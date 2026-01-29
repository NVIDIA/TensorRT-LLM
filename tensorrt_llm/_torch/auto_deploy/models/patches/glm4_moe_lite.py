"""Patches for GLM4 MoE Lite model to enable torch export with fused MLA attention.

This module patches GLM4 MoE Lite's attention and rotary embedding modules to use
AutoDeploy's fused MLA custom op, which enables KV cache insertion later in the pipeline.

The GLM4 MoE Lite model uses Multi-head Latent Attention (MLA), similar to DeepSeek V3.
"""

import types
import warnings
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM


def glm4_moe_lite_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Glm4MoeLiteAttention forward function rewritten to use fused MLA custom op.

    This patches the attention to use torch.ops.auto_deploy.torch_attention_deepseek_fused_mla,
    which captures the MLA pattern for later KV cache insertion.
    """
    if past_key_values is not None:
        raise ValueError("past_key_values is not supported in patched GLM4 MoE Lite attention")

    if "padding_mask" in kwargs:
        warnings.warn("Passing `padding_mask` is deprecated. Please use `attention_mask` instead.")

    bsz, q_len, _ = hidden_states.size()

    # Q projection (with optional LoRA)
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    # Reshape and split Q into nope (no position embedding) and pe (position embedding) parts
    q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # KV compression: project hidden states to compressed KV + k_pe
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )

    # k_pe shape: [bsz, 1, q_len, qk_rope_head_dim]
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

    # KV decompression: layernorm -> projection -> reshape
    # Keep as combined kv tensor (k_nope + v) for the fused op
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )

    # Get cos, sin from position_embeddings (full cached from patched rotary embedding)
    cos, sin = position_embeddings

    # Get position_ids from kwargs (passed by decoder layer)
    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        # Fallback: create position_ids from cache_position or sequential
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0).expand(bsz, -1)
        else:
            position_ids = (
                torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            )

    # Use fused MLA custom op - this captures the MLA pattern for KV cache insertion
    attn_output = torch.ops.auto_deploy.torch_attention_deepseek_fused_mla(
        q_nope,
        q_pe,
        kv,
        k_pe,
        cos,
        sin,
        position_ids,
        attention_mask,
        self.scaling,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def glm4_moe_lite_rope(self, x, position_ids=None):
    """Glm4MoeLiteRotaryEmbedding forward function rewritten to return full cached cos/sin.

    This patches the rotary embedding to return the full cached cos/sin tables instead of
    position-indexed values. The indexing happens inside the fused MLA op.

    Similar to DeepSeek's rotary embedding patch.
    """
    # Return cached cos/sin if already computed, moving to the correct device/dtype
    # Use _ad_cos_cached/_ad_sin_cached to avoid conflicts with any existing attributes
    # and to avoid registering as buffers (which would cause state_dict issues)
    if hasattr(self, "_ad_cos_cached") and self._ad_cos_cached is not None:
        return (
            self._ad_cos_cached.to(dtype=x.dtype).to(device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype).to(device=x.device),
        )

    # Compute cos/sin for all positions up to max_seq_len_cached
    seq_len = self.max_seq_len_cached
    inv_freq = self.inv_freq

    # Create position indices for full sequence
    t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)

    # Compute frequencies: outer product of positions and inverse frequencies
    freqs = torch.outer(t, inv_freq)

    # Concatenate for full head_dim (freqs, freqs) pattern - matches HF's emb = cat((freqs, freqs))
    emb = torch.cat((freqs, freqs), dim=-1)

    # Compute cos and sin with attention scaling
    # Store as regular attributes (not buffers) to avoid state_dict tracking issues
    self._ad_cos_cached = emb.cos() * self.attention_scaling
    self._ad_sin_cached = emb.sin() * self.attention_scaling

    return (
        self._ad_cos_cached.to(dtype=x.dtype).to(device=x.device),
        self._ad_sin_cached.to(dtype=x.dtype).to(device=x.device),
    )


# Store original from_config for chaining
_from_config_original = AutoModelForCausalLM.from_config

# Module patches to apply
CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    # "Glm4MoeLiteAttention": glm4_moe_lite_attention,
    # "Glm4MoeLiteRotaryEmbedding": glm4_moe_lite_rope,
}


def get_model_from_config_patched(config, **kwargs):
    """Patched from_config that applies GLM4 MoE Lite module patches."""
    model = _from_config_original(config, **kwargs)

    # Only patch GLM4 MoE Lite models
    config_class_name = type(config).__name__
    if "Glm4MoeLite" not in config_class_name:
        return model

    # Patch modules
    for _, module in model.named_modules():
        module_class_name = type(module).__name__
        if module_class_name in CUSTOM_MODULE_PATCHES:
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[module_class_name], module)

    return model


# Apply the patch to AutoModelForCausalLM.from_config
# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
