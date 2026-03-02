"""Patches for Qwen3Next to make it compatible with torch.export and reduce export time.

Includes:
  - MoE patch: replaces Qwen3NextSparseMoeBlock.forward with torch_moe op
  - GDN patch: replaces Qwen3NextGatedDeltaNet.forward with autodeploy custom ops
    (torch_causal_conv1d, torch_l2norm, torch_gated_delta_rule)
  - Mask/cache patches: simplify _update_linear_attn_mask and DynamicCache.__bool__
    for torch.export compatibility

Reference HF modeling file (v4.57.1):
  https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDynamicCache,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    apply_mask_to_padding_states,
)

from ...export.interface import BaseExportPatch, ExportPatchRegistry

# =============================================================================
# MoE patch
# =============================================================================


def _forward_moe(self: Qwen3NextSparseMoeBlock, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Routed experts via torch_moe
    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],
        w2_weight=[expert.down_proj.weight for expert in self.experts],
        w3_weight=[expert.up_proj.weight for expert in self.experts],
    )

    # Shared expert path (unique to Qwen3Next vs Qwen3MoE)
    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


@ExportPatchRegistry.register("hf_qwen3_next_moe")
class Qwen3NextMoePatch(BaseExportPatch):
    """Patch for Qwen3Next MoE for torch.export compatibility."""

    def _apply_patch(self):
        """Apply the Qwen3Next MoE patch."""
        self.original_values["Qwen3NextSparseMoeBlock.forward"] = Qwen3NextSparseMoeBlock.forward
        Qwen3NextSparseMoeBlock.forward = _forward_moe  # type: ignore

    def _revert_patch(self):
        """Revert the Qwen3Next MoE patch."""
        Qwen3NextSparseMoeBlock.forward = self.original_values[  # type: ignore
            "Qwen3NextSparseMoeBlock.forward"
        ]


# =============================================================================
# GDN (Gated Delta Net) patch
# =============================================================================

# Original implementation:
# https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L475-L614
# NOTE: we remove cache-related code paths and use autodeploy custom ops.


def _patched_gdn_forward(
    self: Qwen3NextGatedDeltaNet,
    hidden_states: torch.Tensor,
    cache_params: Optional[Qwen3NextDynamicCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    """Patched forward for Qwen3NextGatedDeltaNet.

    Removes cache-dependent control flow and uses autodeploy custom ops:
      - torch_causal_conv1d for the depthwise causal convolution
      - torch_l2norm for L2 normalization of Q and K
      - torch_gated_delta_rule for the core gated delta rule computation
    """
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    # 1. Projections
    projected_states_qkvz = self.in_proj_qkvz(hidden_states)
    projected_states_ba = self.in_proj_ba(hidden_states)
    query, key, value, z, b, a = self.fix_query_key_value_ordering(
        projected_states_qkvz, projected_states_ba
    )
    # Flatten multi-head dims for conv: [B, S, key_dim] etc.
    query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

    # Concatenate QKV for joint convolution: [B, S, conv_dim]
    mixed_qkv = torch.cat((query, key, value), dim=-1)

    # 2. Causal Conv1d via autodeploy op
    # torch_causal_conv1d expects [B, S, C] input, handles transpose internally
    mixed_qkv = torch.ops.auto_deploy.torch_causal_conv1d(
        mixed_qkv,
        self.conv1d.weight,
        self.conv1d.bias,
        self.conv1d.stride[0],
        self.conv1d.padding[0],
        self.conv1d.dilation[0],
        self.conv1d.groups,
        self.conv1d.padding_mode,
    )
    mixed_qkv = F.silu(mixed_qkv)

    # Split back into Q, K, V
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )

    # Reshape to per-head: [B, S, num_heads, head_dim]
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    # 3. L2 normalize Q and K via autodeploy op
    query = torch.ops.auto_deploy.torch_l2norm(query)
    key = torch.ops.auto_deploy.torch_l2norm(key)

    # 4. Compute beta and gating
    beta = b.sigmoid()  # [B, S, num_v_heads]
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [B, S, num_v_heads]

    # Repeat-interleave Q, K if num_v_heads > num_k_heads (GQA for linear attention)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    # 5. Gated Delta Rule via autodeploy custom op
    # Op expects [B, S, H, D] layout (bsnd convention)
    core_attn_out = torch.ops.auto_deploy.torch_gated_delta_rule(query, key, value, g, beta)

    # 6. Gated RMSNorm
    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    # 7. Output projection
    output = self.out_proj(core_attn_out)
    return output


# =============================================================================
# Mask and cache patches for torch.export compatibility
# =============================================================================


def _patched_update_linear_attn_mask(self, attention_mask, cache_position):
    """Patched _update_linear_attn_mask that returns None unconditionally.

    The original checks `cache_position[0] > 0` which is data-dependent and
    incompatible with torch.export.

    Original:
      https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L1082-L1092
    """
    return None


def _cache_bool(self) -> bool:
    """Patched __bool__ for Qwen3NextDynamicCache that returns True unconditionally.

    The base Cache class's __len__ returns 0 when no cache is stored, which makes
    `if past_key_values:` evaluate to False at initialization. The model code should
    really check `if past_key_values is not None`, but patching __bool__ is less
    intrusive than patching the full model forward.

    Same pattern as the Bamba cache bool fix:
      tensorrt_llm/_torch/auto_deploy/models/patches/bamba.py
    """
    return True


@ExportPatchRegistry.register("hf_qwen3_next_gdn")
class Qwen3NextGDNPatch(BaseExportPatch):
    """Patch for Qwen3Next GatedDeltaNet for torch.export compatibility."""

    def _apply_patch(self):
        """Apply the GDN, mask, and cache patches."""
        self.original_values["Qwen3NextGatedDeltaNet.forward"] = Qwen3NextGatedDeltaNet.forward
        self.original_values["Qwen3NextModel._update_linear_attn_mask"] = (
            Qwen3NextModel._update_linear_attn_mask
        )
        # NOTE: Qwen3NextDynamicCache does not have __bool__ by default
        # (it inherits from object), so we just set it and delete on revert.

        Qwen3NextGatedDeltaNet.forward = _patched_gdn_forward  # type: ignore
        Qwen3NextModel._update_linear_attn_mask = _patched_update_linear_attn_mask  # type: ignore
        Qwen3NextDynamicCache.__bool__ = _cache_bool  # type: ignore

    def _revert_patch(self):
        """Revert the GDN, mask, and cache patches."""
        Qwen3NextGatedDeltaNet.forward = self.original_values[  # type: ignore
            "Qwen3NextGatedDeltaNet.forward"
        ]
        Qwen3NextModel._update_linear_attn_mask = self.original_values[  # type: ignore
            "Qwen3NextModel._update_linear_attn_mask"
        ]
        del Qwen3NextDynamicCache.__bool__
