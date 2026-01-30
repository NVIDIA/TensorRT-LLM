"""Patches for GLM4 MoE Lite model to enable torch export with torch_mla attention.

This module patches GLM4 MoE Lite's attention and rotary embedding modules to use
AutoDeploy's torch_mla custom op, which enables KV cache insertion later in the pipeline.

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
    """Glm4MoeLiteAttention forward function rewritten to use torch_mla custom op.

    This patches the attention to use torch.ops.auto_deploy.torch_mla,
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

    # Reshape Q to [B, S, N, D] (bsnd layout) and split into nope/pe parts
    q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # KV compression: project hidden states to compressed KV + k_pe
    kv_a_output = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        kv_a_output, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )

    # Apply layernorm to compressed_kv before passing to torch_mla
    compressed_kv = self.kv_a_layernorm(compressed_kv)

    # k_pe shape: [bsz, q_len, 1, qk_rope_head_dim] (bsnd layout, shared across heads)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

    # Get cos, sin from position_embeddings (full cached from patched rotary embedding)
    cos, sin = position_embeddings  # Full table: [max_seq_len, head_dim]

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

    # Index cos/sin with position_ids: [S, D] -> [B, S, D]
    cos = cos[position_ids]  # Indexed: [batch_size, seq_len, head_dim]
    sin = sin[position_ids]

    # Apply RoPE using custom op (unsqueeze_dim=2 for BSND layout)
    q_pe, k_pe = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(q_pe, k_pe, cos, sin, 2)

    # Use torch_mla custom op - this captures the MLA pattern for KV cache insertion
    # torch_mla returns output in bsnd format [B, S, N, v_head_dim]
    attn_output = torch.ops.auto_deploy.torch_mla(
        q_nope,  # [B, S, N, qk_nope_head_dim]
        q_pe,  # [B, S, N, qk_rope_head_dim] - RoPE applied
        compressed_kv,  # [B, S, kv_lora_rank] - after layernorm
        k_pe,  # [B, S, 1, qk_rope_head_dim] - RoPE applied
        self.kv_b_proj.weight,  # [N*(qk_nope+v), kv_lora_rank]
        True,  # is_causal
        self.scaling,  # scale
        "bsnd",  # layout
    )

    # Output: [B, S, N, v_head_dim] -> [B, S, N * v_head_dim]
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def glm4_moe_lite_rope(self, x, position_ids=None):
    """Glm4MoeLiteRotaryEmbedding forward function rewritten to return full cached cos/sin.

    This patches the rotary embedding to return the full cached cos/sin tables instead of
    position-indexed values. The indexing happens in the attention function before calling
    the torch_mla op.

    The cos/sin buffers are pre-registered in get_model_from_config_patched to ensure they
    exist before FX tracing (which uses meta tensors). This allows lift_to_meta to properly
    save and restore them.
    """
    # Use pre-registered buffers - they were registered in get_model_from_config_patched
    return (
        self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
        self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
    )


def _register_rope_buffers(module):
    """Register cos/sin buffers on a Glm4MoeLiteRotaryEmbedding module.

    This must be called before FX tracing so the buffers exist and can be saved/restored
    by lift_to_meta. The buffers are computed from inv_freq which should have real values
    at this point (when skip_loading_weights=False and using from_pretrained).
    """
    seq_len = module.max_seq_len_cached
    inv_freq = module.inv_freq

    # Create position indices for full sequence
    t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)

    # Compute frequencies: outer product of positions and inverse frequencies
    freqs = torch.outer(t, inv_freq)

    # Concatenate for full head_dim (freqs, freqs) pattern
    emb = torch.cat((freqs, freqs), dim=-1)

    # Register as buffers (persistent=False since they're computed from config, not checkpointed)
    # Being registered buffers means they'll be tracked by named_buffers() and properly
    # saved/restored by lift_to_meta during FX tracing
    module.register_buffer("_ad_cos_cached", emb.cos() * module.attention_scaling, persistent=False)
    module.register_buffer("_ad_sin_cached", emb.sin() * module.attention_scaling, persistent=False)


# Store original from_config for chaining
_from_config_original = AutoModelForCausalLM.from_config

# Module patches to apply
CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "Glm4MoeLiteAttention": glm4_moe_lite_attention,
    "Glm4MoeLiteRotaryEmbedding": glm4_moe_lite_rope,
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

        # Pre-register cos/sin buffers for rotary embedding modules
        # This must happen before FX tracing so lift_to_meta can save/restore them
        if module_class_name == "Glm4MoeLiteRotaryEmbedding":
            _register_rope_buffers(module)

        if module_class_name in CUSTOM_MODULE_PATCHES:
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[module_class_name], module)

    return model


# Apply the patch to AutoModelForCausalLM.from_config
# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
