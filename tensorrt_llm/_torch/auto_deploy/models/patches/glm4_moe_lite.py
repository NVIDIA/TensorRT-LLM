"""Patches for GLM4 MoE Lite model to enable torch export with torch_mla attention and torch_moe.

This module patches GLM4 MoE Lite's attention, rotary embedding, and MoE modules to use
AutoDeploy's custom ops (torch_mla, torch_moe), which enables KV cache insertion and
efficient MoE execution later in the pipeline.

The GLM4 MoE Lite model uses Multi-head Latent Attention (MLA), similar to DeepSeek V3.
"""

import types
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from tensorrt_llm._torch.utils import ActivationType


class Glm4MoeLiteExpert(nn.Module):
    """Individual expert MLP matching checkpoint structure.

    This replaces the stacked 3D parameter approach with individual nn.Linear modules
    per expert, enabling direct checkpoint loading and compatibility with EP/TP sharding.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class Glm4MoeLiteExpertsReplacement(nn.ModuleList):
    """Replacement for Glm4MoeLiteNaiveMoe.experts with ModuleList structure.

    The original HF model uses stacked 3D nn.Parameter tensors (gate_up_proj, down_proj),
    but the checkpoint has per-expert 2D tensors (experts.0.gate_proj.weight, etc.).

    This replacement inherits from nn.ModuleList directly so that state_dict keys
    are flat (e.g., "0.gate_proj.weight") without an extra "_experts." prefix.
    When assigned to moe_module.experts, full keys become: experts.0.gate_proj.weight

    Benefits:
    1. Creates state_dict keys that match the checkpoint exactly
    2. Produces individual get_attr nodes in the graph (instead of aten.select from stacked)
    3. Works with existing EP/TP sharding code that expects individual expert params

    For backward compatibility with the original HF forward, this class provides
    gate_up_proj and down_proj properties that stack the individual weights.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        experts = [Glm4MoeLiteExpert(hidden_size, intermediate_size) for _ in range(num_experts)]
        super().__init__(experts)
        self.intermediate_dim = intermediate_size  # Needed by MoE forward

    @property
    def gate_up_proj(self) -> torch.Tensor:
        """Stacked gate+up projection weights for backward compatibility.

        Returns shape: [num_experts, 2*intermediate_dim, hidden_dim]
        This matches the original HF model's gate_up_proj parameter format.
        """
        # Stack gate_proj and up_proj for each expert along dim=1
        # gate_proj: [intermediate, hidden], up_proj: [intermediate, hidden]
        # Result per expert: [2*intermediate, hidden]
        # Stacked: [num_experts, 2*intermediate, hidden]
        stacked = torch.stack(
            [torch.cat([expert.gate_proj.weight, expert.up_proj.weight], dim=0) for expert in self],
            dim=0,
        )
        return stacked

    @property
    def down_proj(self) -> torch.Tensor:
        """Stacked down projection weights for backward compatibility.

        Returns shape: [num_experts, hidden_dim, intermediate_dim]
        This matches the original HF model's down_proj parameter format.
        """
        # Stack down_proj for each expert
        # down_proj: [hidden, intermediate]
        # Stacked: [num_experts, hidden, intermediate]
        return torch.stack([expert.down_proj.weight for expert in self], dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass matching the original Glm4MoeLiteNaiveMoe interface.

        This is provided for backward compatibility with the original HF forward
        which calls self.experts(hidden_states, topk_indices, topk_weights).

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]
            top_k_index: Expert indices [num_tokens, top_k]
            top_k_weights: Expert weights [num_tokens, top_k]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        num_experts = len(self)
        final_hidden_states = torch.zeros_like(hidden_states)

        # Compute expert mask and find which experts are used
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # SwiGLU computation using individual expert
            expert = self[expert_idx]
            gate_out = torch.nn.functional.silu(expert.gate_proj(current_state))
            up_out = expert.up_proj(current_state)
            current_hidden_states = expert.down_proj(gate_out * up_out)

            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


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


def glm4_moe_lite_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Glm4MoeLiteMoE forward function rewritten to use torch_moe custom op.

    This patches the MoE block to use torch.ops.auto_deploy.torch_moe for the routed experts,
    which is compatible with torch.export and enables efficient MoE fusion later in the pipeline.

    The original forward uses data-dependent operations (torch.nonzero, dynamic indexing) that
    are incompatible with torch.export on meta tensors.

    Key differences from original:
    - Uses torch_moe custom op instead of loop-based expert dispatch
    - Preserves the complex sigmoid + group-based routing logic via route_tokens_to_experts
    - Shared experts are called separately and added to output (same as original)
    - Uses ModuleList-based experts structure (individual get_attr nodes) for EP/TP sharding
    """
    residuals = hidden_states
    orig_shape = hidden_states.shape

    # Use existing routing logic (preserves sigmoid + group selection + bias correction)
    router_logits = self.gate(hidden_states)
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

    # Flatten for torch_moe: (batch, seq, hidden) -> (batch * seq, hidden)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # Call torch_moe with per-expert weight lists
    # self.experts is now Glm4MoeLiteExpertsReplacement with individual expert modules
    # Each expert has gate_proj, up_proj, down_proj as nn.Linear (matching checkpoint keys)
    # This creates individual get_attr nodes in the graph, enabling EP/TP sharding
    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],  # gate projection
        w2_weight=[expert.down_proj.weight for expert in self.experts],  # down projection
        w3_weight=[expert.up_proj.weight for expert in self.experts],  # up projection
        is_gated_mlp=True,
        act_fn=int(ActivationType.Silu),
    )

    final_hidden_states = final_hidden_states.view(*orig_shape)

    # Add shared experts output (same as original)
    final_hidden_states = final_hidden_states + self.shared_experts(residuals)

    return final_hidden_states


# Store original from_config for chaining
_from_config_original = AutoModelForCausalLM.from_config

# Module patches to apply
CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "Glm4MoeLiteAttention": glm4_moe_lite_attention,
    "Glm4MoeLiteRotaryEmbedding": glm4_moe_lite_rope,
    "Glm4MoeLiteMoE": glm4_moe_lite_moe,
}


def _replace_experts_with_modulelist(moe_module):
    """Replace stacked 3D parameter experts with ModuleList-based structure.

    The original HF model uses:
    - experts.gate_up_proj: nn.Parameter [num_experts, 2*intermediate, hidden]
    - experts.down_proj: nn.Parameter [num_experts, hidden, intermediate]

    The checkpoint has per-expert keys:
    - experts.0.gate_proj.weight, experts.0.up_proj.weight, experts.0.down_proj.weight, ...

    This replacement creates a ModuleList structure that matches the checkpoint keys exactly.
    """
    # Extract dimensions from the original stacked parameters
    num_experts = moe_module.n_routed_experts
    # gate_up_proj shape: [num_experts, 2*intermediate, hidden]
    hidden_size = moe_module.experts.gate_up_proj.shape[2]
    intermediate_size = moe_module.experts.intermediate_dim

    # Create replacement with ModuleList structure
    # The replacement's state_dict keys will be: 0.gate_proj.weight, 0.up_proj.weight, etc.
    # When accessed as moe_module.experts, keys become: experts.0.gate_proj.weight, etc.
    replacement = Glm4MoeLiteExpertsReplacement(num_experts, hidden_size, intermediate_size)

    # Move to same device/dtype as original (important for meta tensors)
    device = moe_module.experts.gate_up_proj.device
    dtype = moe_module.experts.gate_up_proj.dtype
    replacement = replacement.to(device=device, dtype=dtype)

    # Replace the experts attribute
    moe_module.experts = replacement


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

        # Replace stacked experts with ModuleList-based structure
        # This must happen BEFORE applying forward patches so the forward uses the new structure
        if module_class_name == "Glm4MoeLiteMoE":
            _replace_experts_with_modulelist(module)

        if module_class_name in CUSTOM_MODULE_PATCHES:
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[module_class_name], module)

    return model


# Apply the patch to AutoModelForCausalLM.from_config
# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
