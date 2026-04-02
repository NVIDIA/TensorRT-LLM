# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5 MoE model for auto_deploy (text + vision).

Reference HF modeling file (not yet in a released transformers version):
  transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

This implementation differs from the HuggingFace original in the following ways:
  * External kernel dependencies (flash-linear-attention, causal_conv1d) are replaced with
    autodeploy custom ops.
  * Cache-related code paths have been removed (prefill-only).
  * Training-related code paths have been removed.
  * Unnecessary output fields have been removed.
  * The GatedDeltaNet forward is adapted from the Qwen3Next GDN patch
    (tensorrt_llm/_torch/auto_deploy/models/patches/qwen3_next.py).
  * The MoE implementation uses expert lists (individual nn.Linear layers per expert)
    that directly match the checkpoint structure, dispatched via torch_moe op.
  * The VLM wrapper passes 3D ``position_ids (3, B, S)`` to the text model,
    which computes mRoPE cos/sin internally via its own ``rotary_emb``.
    For text-only inputs the wrapper expands the executor's 2D positions to 3D;
    for multimodal inputs it computes spatial (T, H, W) positions via ``get_rope_index``.

This allows us to have a "pytorch" native reference implementation decoupled from bugs and
dependency issues in the source, while remaining weight-compatible with HF checkpoints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.export import Dim
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
    TextModelExportInfo,
)
from tensorrt_llm.inputs.multimodal import MultimodalInput, apply_mm_hashes, hexdigest_to_int32
from tensorrt_llm.inputs.utils import VideoData

# =============================================================================
# Configuration
# =============================================================================


class Qwen3_5MoeTextConfig(PretrainedConfig):
    """Minimal config class for Qwen3.5 MoE text model.

    Mirrors the attributes of the upstream Qwen3_5MoeTextConfig. Only attributes
    needed by the slimmed-down prefill model are included.
    """

    model_type = "qwen3_5_moe_text"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        # linear attention (GatedDeltaNet) params
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        # MoE params
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        # layer types
        layer_types=None,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
            }
        self.rope_parameters = rope_parameters

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts

        self.layer_types = layer_types
        if self.layer_types is None:
            # Default pattern: every 4th layer is full_attention, rest are linear_attention
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# =============================================================================
# Normalization
# =============================================================================


class Qwen3_5MoeRMSNorm(nn.Module):
    """RMSNorm with weight scaling.

    The HF checkpoint stores weights in ``(1 + w)`` parameterisation (zeros
    init). A load-time pre-hook adds 1.0 so that the forward can use a plain
    ``weight * x`` multiply, which matches the autodeploy RMSNorm pattern and
    gets fused into a single ``flashinfer_rms_norm`` kernel.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._register_load_state_dict_pre_hook(self._offset_weight)

    @staticmethod
    def _offset_weight(state_dict, prefix, *args):
        key = prefix + "weight"
        assert key in state_dict, f"RMSNorm: Key {key} not found in state_dict"
        state_dict[key] = state_dict[key] + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        output = x.to(torch.float32)
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.to(torch.float32) * output).to(input_dtype)


class Qwen3_5MoeRMSNormGated(nn.Module):
    """Gated RMSNorm: norm(x) * weight * silu(gate). Weight is initialized to ones."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# =============================================================================
# Rotary Position Embedding (mRoPE)
# =============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE to query and key tensors.

    Supports partial rotary where only the first `rotary_dim` dimensions are rotated.
    Default unsqueeze_dim=2 is for bsnd layout (B, S, N, D).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class Qwen3_5MoeTextRotaryEmbedding(nn.Module):
    """Simplified mRoPE for text-only prefill. Supports only the "default" rope type."""

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        rope_params = config.rope_parameters
        base = rope_params["rope_theta"]
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings.

        Args:
            x: Hidden states tensor, used only for dtype/device.
            position_ids: Shape (3, B, S) for mRoPE or (B, S) for plain.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # inv_freq: (dim/2,) -> (3, B, dim/2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        # position_ids: (3, B, S) -> (3, B, 1, S)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # freqs: (3, B, S, dim/2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        # Apply interleaved mRoPE: (3, B, S, dim/2) -> (B, S, dim/2)
        freqs = self._apply_interleaved_mrope(freqs)
        # Double for cos/sin: (B, S, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Apply interleaved mRoPE. Merges T/H/W frequency channels into one tensor."""
        freqs_t = freqs[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]
        return freqs_t


# =============================================================================
# GatedDeltaNet (Linear Attention)
# =============================================================================
# Adapted from the Qwen3Next GDN patch:
#   tensorrt_llm/_torch/auto_deploy/models/patches/qwen3_next.py
# Uses autodeploy custom ops: torch_causal_conv1d, torch_gated_delta_rule


class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """Prefill-only GatedDeltaNet using autodeploy custom ops."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        # QKV convolution
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # dt_bias and A_log for gated delta rule
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm (per head_v_dim)
        self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        # Projections
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        # 1. Projections (separate, unlike Qwen3Next which uses combined in_proj_qkvz)
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, S, conv_dim]
        z = self.in_proj_z(hidden_states)  # [B, S, value_dim]
        b = self.in_proj_b(hidden_states)  # [B, S, num_v_heads]
        a = self.in_proj_a(hidden_states)  # [B, S, num_v_heads]

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

        # 3. Gated Delta Rule via autodeploy custom op
        # L2 norm, GQA repeat-interleave, and g/beta computation are handled inside the op.
        core_attn_out = torch.ops.auto_deploy.torch_gated_delta_rule(
            query, key, value, a, b, self.A_log, self.dt_bias
        )

        # 5. Gated RMSNorm + merge heads
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [B, S, num_v_heads, head_v_dim]
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # 6. Output projection
        output = self.out_proj(core_attn_out)
        return output


# =============================================================================
# Attention
# =============================================================================


class Qwen3_5MoeAttention(nn.Module):
    """Multi-headed attention with gating, Q/K norms, and partial RoPE.

    Key differences from standard attention:
      - q_proj outputs 2x (query + gate), gating applied to attention output.
      - q_norm / k_norm applied per-head before RoPE.
      - Partial RoPE: only first rotary_dim dimensions are rotated.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # q_proj outputs 2x for query + gate
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Per-head Q/K norms
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Q projection with gate: output shape (B, S, N, 2*D)
        qg = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(qg, 2, dim=-1)  # each (B, S, N, D)
        gate = gate.reshape(bsz, q_len, -1)  # (B, S, N*D)

        # K, V projections in bsnd layout
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        # Per-head Q/K norms (norm operates on last dim = head_dim)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Partial RoPE in bsnd layout (unsqueeze_dim=2 for the N dimension)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        # Attention via autodeploy op (bsnd layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            layout="bsnd",
        )
        attn_output = attn_output.view(bsz, q_len, -1)  # (B, S, N*D)

        # Gated output
        attn_output = attn_output * torch.sigmoid(gate)

        # Output projection
        attn_output = self.o_proj(attn_output)
        return attn_output


# =============================================================================
# MLP and MoE
# =============================================================================


class Qwen3_5MoeMLP(nn.Module):
    """SwiGLU MLP used for the shared expert."""

    def __init__(self, config: Qwen3_5MoeTextConfig, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5MoeExpert(nn.Module):
    """Single expert with gate, up, and down projections."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


class Qwen3_5MoeTopKRouter(nn.Module):
    """Top-K router with softmax normalization."""

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (T, E)
        routing_weights = F.softmax(router_logits, dtype=torch.float, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # (T, top_k)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        return routing_weights, selected_experts


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """MoE block with expert list implementation.

    Implements routed experts by iterating over selected experts and dispatching
    tokens accordingly. Each expert is a separate nn.Linear triplet (gate, up, down).
    """

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = nn.ModuleList(
            [
                Qwen3_5MoeExpert(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.shared_expert = Qwen3_5MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self._register_load_state_dict_pre_hook(self._load_experts_from_fused_checkpoint)

    @staticmethod
    def _load_experts_from_fused_checkpoint(state_dict, prefix, *args):
        """Load fused MoE expert checkpoint tensors into per-expert ModuleList params.

        Checkpoint format:
            - experts.gate_up_proj: [E, 2*I, H] with [gate, up] stacking
            - experts.down_proj: [E, H, I]

        Target format:
            - experts.{expert_id}.gate_proj.weight: [I, H]
            - experts.{expert_id}.up_proj.weight: [I, H]
            - experts.{expert_id}.down_proj.weight: [H, I]
        """
        gate_up_key = prefix + "experts.gate_up_proj"
        down_key = prefix + "experts.down_proj"

        if gate_up_key in state_dict:
            fused = state_dict.pop(gate_up_key)
            num_experts = fused.shape[0]
            intermediate_dim = fused.shape[1] // 2
            gate_weights = fused[:, :intermediate_dim, :]
            up_weights = fused[:, intermediate_dim:, :]

            for i in range(num_experts):
                state_dict[f"{prefix}experts.{i}.gate_proj.weight"] = gate_weights[i]
                state_dict[f"{prefix}experts.{i}.up_proj.weight"] = up_weights[i]

        if down_key in state_dict:
            fused = state_dict.pop(down_key)
            num_experts = fused.shape[0]
            for i in range(num_experts):
                state_dict[f"{prefix}experts.{i}.down_proj.weight"] = fused[i]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router
        routing_weights, selected_experts = self.gate(hidden_states_flat)

        # Routed experts via torch_moe op with expert list weights
        # Extract weight tensors from expert modules
        w1_weights = [self.experts[i].gate_proj.weight for i in range(len(self.experts))]
        w2_weights = [self.experts[i].down_proj.weight for i in range(len(self.experts))]
        w3_weights = [self.experts[i].up_proj.weight for i in range(len(self.experts))]

        # Shared expert with sigmoid gating
        shared_expert_output = self.shared_expert(hidden_states_flat)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output
        )

        expert_output = torch.ops.auto_deploy.torch_moe(
            hidden_states_flat,
            selected_experts,
            routing_weights,
            w1_weights,
            w2_weights,
            w3_weights,
            is_gated_mlp=True,
        )

        expert_output = expert_output + shared_expert_output

        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


# =============================================================================
# Decoder Layer
# =============================================================================


class Qwen3_5MoeDecoderLayer(nn.Module):
    """Single decoder layer: token mixer (linear_attention or full_attention) + MoE."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Token mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)

        hidden_states = residual + hidden_states

        # Channel mixer (MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model
# =============================================================================


class Qwen3_5MoePreTrainedModel(PreTrainedModel):
    """Base class for Qwen3.5 MoE pretrained models."""

    config_class = Qwen3_5MoeTextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3_5MoeDecoderLayer"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        # Delegate nn.Linear / nn.Embedding / nn.Conv* to the base class, which
        # safely resolves initializer_range via hasattr + get_text_config() fallback.
        super()._init_weights(module)
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, Qwen3_5MoeRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Qwen3_5MoeRMSNormGated):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Qwen3_5MoeGatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, Qwen3_5MoeTopKRouter):
            module.weight.data.normal_(mean=0.0, std=std)


@dataclass
class Qwen3_5MoeOutput(ModelOutput):
    """Output of the Qwen3.5 MoE text model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Qwen3_5MoeCausalLMOutput(ModelOutput):
    """Output of the Qwen3.5 MoE causal language model."""

    logits: Optional[torch.FloatTensor] = None


class Qwen3_5MoeTextModel(Qwen3_5MoePreTrainedModel):
    """Qwen3.5 MoE text model (embed + decoder layers + final norm + lm_head).

    lm_head is included so that the exported GraphModule contains it directly,
    allowing sharding and gather_logits_before_lm_head transforms to see it
    without post-export grafting.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__(config)
        pad_token_id = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.lm_head = None  # set by parent model via set_lm_head()

        # Initialize weights and apply final processing
        self.post_init()

    def set_lm_head(self, lm_head: nn.Module):
        """Set the lm_head from the parent model."""
        self.lm_head = lm_head

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5MoeOutput]:
        """Forward pass.

        There are three ways to provide position information (checked in order):

        1. ``rope_cos`` + ``rope_sin``: separate tensors, each ``(B, S, rotary_dim)``.
           Export-friendly -- each is a proper graph input with its own dynamic shape.
        2. ``position_embeddings``: pre-computed ``(cos, sin)`` tuple. Convenient for
           the multimodal wrapper calling at the plain-PyTorch level.
        3. ``position_ids`` (or ``None``): standard 2D ``(B, S)`` or 3D ``(3, B, S)``
           position IDs.  The internal ``rotary_emb`` computes cos/sin.  This is the
           default text-only path.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Resolve position embeddings from one of the three input modes.
        if rope_cos is not None and rope_sin is not None:
            position_embeddings = (rope_cos, rope_sin)
        elif position_embeddings is None:
            if position_ids is None:
                seq_len = inputs_embeds.shape[1]
                position_ids = torch.arange(seq_len, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings=position_embeddings)

        hidden_states = self.norm(hidden_states)
        assert self.lm_head is not None, (
            "lm_head not set — call set_lm_head() from the parent model before forward()"
        )
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return Qwen3_5MoeCausalLMOutput(logits=logits)


class Qwen3_5MoeForCausalLM(Qwen3_5MoePreTrainedModel, GenerationMixin):
    """Qwen3.5 MoE causal language model (text model + lm_head)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3_5MoeTextConfig, **kwargs):
        super().__init__(config)
        self.model = Qwen3_5MoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.set_lm_head(self.lm_head)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5MoeCausalLMOutput]:
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        logits = outputs.logits
        return Qwen3_5MoeCausalLMOutput(logits=logits)


# =============================================================================
# Vision Configuration
# =============================================================================


class Qwen3_5MoeVisionConfig(PretrainedConfig):
    """Config class for the Qwen3.5 MoE vision tower.

    Mirrors the upstream ``Qwen3_5MoeVisionConfig``.
    """

    model_type = "qwen3_5_moe_vision"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen3_5MoeConfig(PretrainedConfig):
    """Composite config containing both text and vision configs.

    Mirrors the upstream ``Qwen3_5MoeConfig``.
    """

    model_type = "qwen3_5_moe"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Qwen3_5MoeVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen3_5MoeVisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = Qwen3_5MoeTextConfig(**text_config)
        elif text_config is None:
            self.text_config = Qwen3_5MoeTextConfig()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# =============================================================================
# Vision Tower Components (plain PyTorch -- NOT exported)
# =============================================================================


class Qwen3_5MoeVisionRotaryEmbedding(nn.Module):
    """Simple rotary embedding for the vision tower (not mRoPE)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to vision Q/K tensors. Layout: (seq, heads, dim)."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3_5MoeVisionPatchEmbed(nn.Module):
    """3D convolution patch embedding for images/videos."""

    def __init__(self, config: Qwen3_5MoeVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3_5MoeVisionMLP(nn.Module):
    """Feed-forward network for vision blocks."""

    def __init__(self, config: Qwen3_5MoeVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3_5MoeVisionAttention(nn.Module):
    """Bidirectional attention for vision tokens with cu_seqlens support.

    Uses either:
      - Eager path: splits by sequence lengths, runs attention per chunk.
      - (Future) Flash Attention: single call with cu_seqlens.

    Always non-causal (is_causal=False).
    """

    def __init__(self, config: Qwen3_5MoeVisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Layout: (1, num_heads, seq_len, head_dim) per chunk
        query_states = query_states.transpose(0, 1).unsqueeze(0)  # (1, H, S, D)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(query_states, lengths, dim=2)
        k_splits = torch.split(key_states, lengths, dim=2)
        v_splits = torch.split(value_states, lengths, dim=2)

        attn_outputs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            attn_outputs.append(F.scaled_dot_product_attention(q, k, v, is_causal=False))

        attn_output = torch.cat(attn_outputs, dim=2)  # (1, H, total_S, D)
        attn_output = attn_output.squeeze(0).transpose(0, 1)  # (S, H, D)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5MoeVisionBlock(nn.Module):
    """Vision transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual."""

    def __init__(self, config: Qwen3_5MoeVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5MoeVisionAttention(config=config)
        self.mlp = Qwen3_5MoeVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5MoeVisionPatchMerger(nn.Module):
    """Merges spatial_merge_size^2 patches into one token and projects to LLM hidden size."""

    def __init__(self, config: Qwen3_5MoeVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3_5MoeVisionModel(nn.Module):
    """Complete vision tower: PatchEmbed + PositionEmbed + VisionBlocks + PatchMerger.

    This module is NOT exported -- it runs in plain PyTorch.
    """

    def __init__(self, config: Qwen3_5MoeVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.dtype = None  # set after loading weights

        self.patch_embed = Qwen3_5MoeVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3_5MoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5MoeVisionPatchMerger(config=config)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings for vision tokens."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim//2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of learned positional embeddings for variable image sizes."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list: List[List[int]] = [[] for _ in range(4)]
        weight_list: List[List[float]] = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()))
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()))

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)]
        )

        merge_size = self.config.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> BaseModelOutputWithPooling:
        """Run the vision tower.

        Args:
            hidden_states: Raw pixel values reshaped for patch embedding.
            grid_thw: Shape ``(num_images_or_videos, 3)`` -- temporal, height, width.

        Returns:
            ``BaseModelOutputWithPooling`` with ``pooler_output`` containing merged features.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


# =============================================================================
# Multimodal Wrapper (plain PyTorch -- NOT exported)
# =============================================================================


@dataclass
class Qwen3_5MoeConditionalOutput(ModelOutput):
    """Output of the Qwen3.5 MoE conditional generation model."""

    logits: Optional[torch.FloatTensor] = None


def compute_mrope_positions(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D mRoPE position IDs for multimodal sequences.

    Standalone function usable by both the model forward and the input processor.
    For each sample in the batch, scans for vision placeholder tokens and assigns
    spatial (T, H, W) positions to vision tokens while text tokens get sequential
    positions.

    Args:
        input_ids: Token IDs, shape ``(B, S)``.
        image_grid_thw: Grid dimensions ``(N_images, 3)`` with ``(T, H, W)`` per image.
        video_grid_thw: Grid dimensions ``(N_videos, 3)`` with ``(T, H, W)`` per video.
        image_token_id: Token ID for image placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking the start of a vision segment.
        spatial_merge_size: Factor by which the vision patch merger reduces spatial dims.
        attention_mask: Optional mask, shape ``(B, S)``.

    Returns:
        ``(position_ids, mrope_position_deltas)`` where ``position_ids`` has
        shape ``(3, B, S)`` and ``mrope_position_deltas`` has shape ``(B, 1)``.
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, ids in enumerate(total_input_ids):
            ids = ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
            vision_tokens = ids[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            video_nums = int((vision_tokens == video_token_id).sum().item())
            input_tokens = ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                ed_image = (
                    input_tokens.index(image_token_id, st)
                    if image_token_id in input_tokens and remain_images > 0
                    else len(input_tokens) + 1
                )
                ed_video = (
                    input_tokens.index(video_token_id, st)
                    if video_token_id in input_tokens and remain_videos > 0
                    else len(input_tokens) + 1
                )

                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index].tolist()
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index].tolist()
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = int(t)
                llm_grid_h = int(h) // spatial_merge_size
                llm_grid_w = int(w) // spatial_merge_size
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                device=position_ids.device, dtype=position_ids.dtype
            )
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
            )

        return position_ids, mrope_position_deltas


def _normalize_video_grid_for_mrope(
    video_grid_thw: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if video_grid_thw is None:
        return None
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw = video_grid_thw.clone()
    video_grid_thw[:, 0] = 1
    return video_grid_thw


def _extract_mm_item_types_from_input_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> List[int]:
    """Return multimodal item types in prompt order for a single request."""
    flat_ids = input_ids.reshape(-1).tolist()
    item_types: List[int] = []
    for idx in range(len(flat_ids) - 1):
        if flat_ids[idx] != vision_start_token_id:
            continue
        next_token = flat_ids[idx + 1]
        if next_token == image_token_id:
            item_types.append(0)
        elif next_token == video_token_id:
            item_types.append(1)
    return item_types


def _is_qwen_video_frame(value: Any) -> bool:
    return isinstance(value, (Image.Image, torch.Tensor))


def _normalize_qwen_image_items(images: Any) -> list[Any]:
    if images is None:
        return []
    if isinstance(images, list):
        return images
    return [images]


def _normalize_qwen_video_items(videos: Any) -> list[Any]:
    if videos is None:
        return []
    if isinstance(videos, VideoData):
        return [videos]
    if isinstance(videos, list):
        if not videos:
            return []
        if all(_is_qwen_video_frame(frame) for frame in videos):
            return [videos]
        normalized_items = []
        for item in videos:
            if isinstance(item, VideoData):
                normalized_items.append(item)
            elif (
                isinstance(item, list)
                and item
                and all(_is_qwen_video_frame(frame) for frame in item)
            ):
                normalized_items.append(item)
            else:
                normalized_items.append(item)
        return normalized_items
    return [videos]


def _get_qwen_video_num_spans(video: Any) -> int:
    if isinstance(video, VideoData):
        return len(video.frames)
    if isinstance(video, list):
        if not video:
            return 0
        if all(_is_qwen_video_frame(frame) for frame in video):
            return len(video)
    shape = getattr(video, "shape", None)
    if shape is not None and len(shape) >= 4:
        return int(shape[0])
    return 1


def _compute_mm_item_special_counts(
    mm_token_lengths: torch.Tensor,
    mm_special_offsets_cu_seqlen: torch.Tensor,
    mm_special_offsets: torch.Tensor,
    req_idx: int,
) -> List[int]:
    item_lengths = mm_token_lengths.tolist()
    special_start = int(mm_special_offsets_cu_seqlen[req_idx].item())
    special_end = int(mm_special_offsets_cu_seqlen[req_idx + 1].item())
    special_offsets = mm_special_offsets[special_start:special_end].tolist()
    counts: List[int] = []
    mm_offset = 0
    for item_len in item_lengths:
        item_end = mm_offset + int(item_len)
        num_special = sum(1 for off in special_offsets if mm_offset <= int(off) < item_end)
        counts.append(num_special)
        mm_offset = item_end
    return counts


def _compute_request_mrope_delta(
    mm_item_types: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    special_counts: Sequence[int],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> int:
    image_idx = 0
    video_idx = 0
    total_delta = 0
    for item_type, item_len, num_special in zip(
        mm_item_types.tolist(), mm_token_lengths.tolist(), special_counts
    ):
        num_placeholders = int(item_len) - int(num_special)
        if item_type == 0:
            if image_grid_thw is None:
                raise ValueError("Expected image_grid_thw for image multimodal item")
            t, h, w = [int(v) for v in image_grid_thw[image_idx].tolist()]
            image_idx += 1
        else:
            if video_grid_thw is None:
                raise ValueError("Expected video_grid_thw for video multimodal item")
            t, h, w = [int(v) for v in video_grid_thw[video_idx].tolist()]
            video_idx += 1
        llm_grid_t = int(t)
        llm_grid_h = int(h) // spatial_merge_size
        llm_grid_w = int(w) // spatial_merge_size
        total_delta += max(llm_grid_t, llm_grid_h, llm_grid_w) - num_placeholders
    return total_delta


@torch.library.custom_op("auto_deploy::qwen3_mrope_delta", mutates_args=())
def qwen3_mrope_delta(
    batch_info_host: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_special_offsets_cu_seqlen: torch.Tensor,
    mm_special_offsets: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> torch.Tensor:
    num_prefill, _, num_decode = BatchInfo(batch_info_host).get_absorbed_info()
    num_seq = num_prefill + num_decode
    device = mm_item_cu_seqlen.device
    out = torch.zeros((num_seq, 1), dtype=torch.int32, device=device)
    video_grid_norm = _normalize_video_grid_for_mrope(video_grid_thw)
    img_idx = 0
    vid_idx = 0
    for req_idx in range(num_prefill):
        item_start = int(mm_item_cu_seqlen[req_idx].item())
        item_end = int(mm_item_cu_seqlen[req_idx + 1].item())
        req_item_types = mm_item_types[item_start:item_end]
        req_item_lengths = mm_token_lengths[item_start:item_end]
        if req_item_lengths.numel() == 0:
            continue
        num_images = int((req_item_types == 0).sum().item())
        num_videos = int((req_item_types == 1).sum().item())
        req_image_grid = image_grid_thw[img_idx : img_idx + num_images] if num_images > 0 else None
        req_video_grid = video_grid_norm[vid_idx : vid_idx + num_videos] if num_videos > 0 else None
        special_counts = _compute_mm_item_special_counts(
            req_item_lengths,
            mm_special_offsets_cu_seqlen,
            mm_special_offsets,
            req_idx,
        )
        out[req_idx, 0] = _compute_request_mrope_delta(
            req_item_types,
            req_item_lengths,
            special_counts,
            req_image_grid,
            req_video_grid,
            spatial_merge_size,
        )
        img_idx += num_images
        vid_idx += num_videos
    return out


@qwen3_mrope_delta.register_fake
def qwen3_mrope_delta_fake(
    batch_info_host: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_special_offsets_cu_seqlen: torch.Tensor,
    mm_special_offsets: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> torch.Tensor:
    num_prefill, _, num_decode = BatchInfo(batch_info_host).get_absorbed_info()
    num_seq = num_prefill + num_decode
    return torch.zeros((num_seq, 1), dtype=torch.int32, device=batch_info_host.device)


@torch.library.custom_op(
    "auto_deploy::qwen3_mrope_delta_with_cache", mutates_args=("mrope_delta_cache",)
)
def qwen3_mrope_delta_with_cache(
    batch_info_host: torch.Tensor,
    slot_idx: torch.Tensor,
    mm_item_cu_seqlen: Optional[torch.Tensor],
    mm_item_types: Optional[torch.Tensor],
    mm_token_lengths: Optional[torch.Tensor],
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
    mm_special_offsets: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    mrope_delta_cache: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    num_prefill, _, num_decode = BatchInfo(batch_info_host).get_absorbed_info()
    num_seq = num_prefill + num_decode
    out = torch.zeros((num_seq, 1), dtype=torch.int32, device=mrope_delta_cache.device)
    video_grid_norm = _normalize_video_grid_for_mrope(video_grid_thw)
    if num_prefill > 0:
        has_mm_metadata = all(
            arg is not None
            for arg in (
                mm_item_cu_seqlen,
                mm_item_types,
                mm_token_lengths,
                mm_special_offsets_cu_seqlen,
                mm_special_offsets,
            )
        )
        if has_mm_metadata:
            img_idx = 0
            vid_idx = 0
            for req_idx in range(num_prefill):
                item_start = int(mm_item_cu_seqlen[req_idx].item())
                item_end = int(mm_item_cu_seqlen[req_idx + 1].item())
                req_item_types = mm_item_types[item_start:item_end]
                req_item_lengths = mm_token_lengths[item_start:item_end]
                if req_item_lengths.numel() == 0:
                    continue
                num_images = int((req_item_types == 0).sum().item())
                num_videos = int((req_item_types == 1).sum().item())
                req_image_grid = (
                    image_grid_thw[img_idx : img_idx + num_images] if num_images > 0 else None
                )
                req_video_grid = (
                    video_grid_norm[vid_idx : vid_idx + num_videos] if num_videos > 0 else None
                )
                special_counts = _compute_mm_item_special_counts(
                    req_item_lengths,
                    mm_special_offsets_cu_seqlen,
                    mm_special_offsets,
                    req_idx,
                )
                out[req_idx, 0] = _compute_request_mrope_delta(
                    req_item_types,
                    req_item_lengths,
                    special_counts,
                    req_image_grid,
                    req_video_grid,
                    spatial_merge_size,
                )
                img_idx += num_images
                vid_idx += num_videos
        mrope_delta_cache.index_copy_(
            0,
            slot_idx[:num_prefill].to(torch.long),
            out[:num_prefill].to(mrope_delta_cache.dtype),
        )
    if num_decode > 0:
        out[num_prefill:num_seq] = mrope_delta_cache[
            slot_idx[num_prefill:num_seq].to(torch.long)
        ].to(torch.int32)
    return out


@qwen3_mrope_delta_with_cache.register_fake
def qwen3_mrope_delta_with_cache_fake(
    batch_info_host: torch.Tensor,
    slot_idx: torch.Tensor,
    mm_item_cu_seqlen: Optional[torch.Tensor],
    mm_item_types: Optional[torch.Tensor],
    mm_token_lengths: Optional[torch.Tensor],
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
    mm_special_offsets: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    mrope_delta_cache: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    num_prefill, _, num_decode = BatchInfo(batch_info_host).get_absorbed_info()
    num_seq = num_prefill + num_decode
    return torch.zeros((num_seq, 1), dtype=torch.int32, device=slot_idx.device)


class Qwen3_5MoeModel(nn.Module):
    """Multimodal wrapper: vision tower + embedding merge + mRoPE + language model.

    This module is NOT exported. It orchestrates the vision pipeline in plain
    PyTorch and calls the (potentially exported) language model with 3D
    ``position_ids (3, B, S)`` so that the text model's internal ``rotary_emb``
    computes the correct mRoPE cos/sin.
    """

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5MoeVisionModel(config.vision_config)
        self.language_model = Qwen3_5MoeTextModel(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D mRoPE position IDs. Delegates to ``compute_mrope_positions``."""
        return compute_mrope_positions(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            attention_mask=attention_mask,
        )

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor
    ) -> List[torch.Tensor]:
        """Run vision tower on images and split by grid dimensions."""
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values, grid_thw=image_grid_thw
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return list(torch.split(image_embeds, split_sizes))

    def get_video_features(
        self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.LongTensor
    ) -> List[torch.Tensor]:
        """Run vision tower on videos (same as images)."""
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find image/video placeholder token positions in the embedding sequence."""
        special_image_mask = (
            (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        special_video_mask = (
            (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        return special_image_mask, special_video_mask

    def _select_request_chunk_multimodal_embeds(
        self,
        req_input_pos: int,
        req_seq_len: int,
        req_mm_item_types: Sequence[int],
        req_mm_positions: Sequence[int],
        req_mm_lengths: Sequence[int],
        req_special_offsets: Sequence[int],
        image_embeds_list: Optional[Sequence[torch.Tensor]],
        video_embeds_list: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        chunk_end = req_input_pos + req_seq_len
        mm_cumulative_offset = 0
        img_idx = 0
        vid_idx = 0
        chunks: list[torch.Tensor] = []
        hidden_size = self.config.text_config.hidden_size
        special_offsets_set = set(int(x) for x in req_special_offsets)

        for item_type, mm_start, mm_len in zip(req_mm_item_types, req_mm_positions, req_mm_lengths):
            item_mm_offset = mm_cumulative_offset
            item_mm_len = int(mm_len)
            item_abs_start = int(mm_start)
            item_abs_end = item_abs_start + item_mm_len
            overlap_start = max(req_input_pos, item_abs_start)
            overlap_end = min(chunk_end, item_abs_end)

            if item_type == 0:
                if image_embeds_list is None:
                    raise ValueError("Missing image embeddings for image multimodal item")
                item_embeds = image_embeds_list[img_idx]
                img_idx += 1
            elif item_type == 1:
                if video_embeds_list is None:
                    raise ValueError("Missing video embeddings for video multimodal item")
                item_embeds = video_embeds_list[vid_idx]
                vid_idx += 1
            else:
                raise ValueError(f"Unsupported multimodal item type: {item_type}")

            local_to_feature_idx: list[Optional[int]] = []
            feature_idx = 0
            for rel in range(item_mm_len):
                if item_mm_offset + rel in special_offsets_set:
                    local_to_feature_idx.append(None)
                else:
                    local_to_feature_idx.append(feature_idx)
                    feature_idx += 1

            if feature_idx != item_embeds.shape[0]:
                raise ValueError(
                    "Multimodal embedding length mismatch for Qwen3.5 item: "
                    f"type={item_type}, expected={feature_idx}, actual={item_embeds.shape[0]}, "
                    f"mm_len={item_mm_len}, item_start={item_abs_start}, "
                    f"special_offsets={sorted(special_offsets_set)}"
                )

            if overlap_start < overlap_end:
                selected_indices = [
                    local_to_feature_idx[rel]
                    for rel in range(overlap_start - item_abs_start, overlap_end - item_abs_start)
                    if local_to_feature_idx[rel] is not None
                ]
                if selected_indices:
                    chunks.append(item_embeds[selected_indices])

            mm_cumulative_offset += item_mm_len

        if chunks:
            return torch.cat(chunks, dim=0)

        device = None
        dtype = None
        if image_embeds_list:
            device = image_embeds_list[0].device
            dtype = image_embeds_list[0].dtype
        elif video_embeds_list:
            device = video_embeds_list[0].device
            dtype = video_embeds_list[0].dtype
        if device is None or dtype is None:
            raise ValueError(
                "Cannot build empty multimodal chunk without image or video embeddings"
            )
        return torch.empty(0, hidden_size, device=device, dtype=dtype)

    def _expand_video_embeds_by_span(
        self,
        video_embeds_list: Optional[Sequence[torch.Tensor]],
        video_grid_thw: Optional[torch.Tensor],
    ) -> Optional[List[torch.Tensor]]:
        if video_embeds_list is None or video_grid_thw is None:
            return None

        merge = self.config.vision_config.spatial_merge_size
        video_span_embeds: List[torch.Tensor] = []
        for video_embeds, grid in zip(video_embeds_list, video_grid_thw):
            t, h, w = [int(v) for v in grid.tolist()]
            frame_tokens = (int(h) // merge) * (int(w) // merge)
            expected = int(t) * frame_tokens
            if video_embeds.shape[0] != expected:
                raise ValueError(
                    "Video embedding length mismatch in Qwen3.5 VLM forward: "
                    f"expected={expected}, actual={video_embeds.shape[0]}, grid={tuple(grid.tolist())}"
                )
            video_span_embeds.extend(list(torch.split(video_embeds, frame_tokens, dim=0)))
        return video_span_embeds

    def _build_chunked_multimodal_embeds(
        self,
        input_ids: torch.LongTensor,
        batch_info: torch.Tensor,
        cu_seqlen: torch.Tensor,
        input_pos: torch.Tensor,
        seq_len: torch.Tensor,
        image_embeds_list: Optional[Sequence[torch.Tensor]],
        video_span_embeds_list: Optional[Sequence[torch.Tensor]],
        mm_item_cu_seqlen: torch.Tensor,
        mm_item_types: torch.Tensor,
        mm_token_positions: torch.Tensor,
        mm_token_lengths: torch.Tensor,
        mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
        mm_special_offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_prefill_seqs = int(batch_info[0].item())
        img_idx = 0
        vid_idx = 0
        chunks: list[torch.Tensor] = []

        for i in range(num_prefill_seqs):
            item_start = int(mm_item_cu_seqlen[i].item())
            item_end = int(mm_item_cu_seqlen[i + 1].item())
            req_mm_item_types = mm_item_types[item_start:item_end].tolist()
            req_mm_positions = mm_token_positions[item_start:item_end].tolist()
            req_mm_lengths = mm_token_lengths[item_start:item_end].tolist()

            req_special_offsets: list[int] = []
            if mm_special_offsets_cu_seqlen is not None and mm_special_offsets is not None:
                special_start = int(mm_special_offsets_cu_seqlen[i].item())
                special_end = int(mm_special_offsets_cu_seqlen[i + 1].item())
                req_special_offsets = mm_special_offsets[special_start:special_end].tolist()

            num_images = sum(item_type == 0 for item_type in req_mm_item_types)
            num_videos = sum(item_type == 1 for item_type in req_mm_item_types)
            req_image_embeds = (
                image_embeds_list[img_idx : img_idx + num_images]
                if image_embeds_list is not None
                else None
            )
            req_video_embeds = (
                video_span_embeds_list[vid_idx : vid_idx + num_videos]
                if video_span_embeds_list is not None
                else None
            )
            img_idx += num_images
            vid_idx += num_videos

            req_chunk_embeds = self._select_request_chunk_multimodal_embeds(
                req_input_pos=int(input_pos[i].item()),
                req_seq_len=int(seq_len[i].item()),
                req_mm_item_types=req_mm_item_types,
                req_mm_positions=req_mm_positions,
                req_mm_lengths=req_mm_lengths,
                req_special_offsets=req_special_offsets,
                image_embeds_list=req_image_embeds,
                video_embeds_list=req_video_embeds,
            )
            chunks.append(req_chunk_embeds)

        if chunks:
            return torch.cat(chunks, dim=0)

        hidden_size = self.config.text_config.hidden_size
        return torch.empty(
            0, hidden_size, device=input_ids.device, dtype=self.get_input_embeddings().weight.dtype
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mrope_position_deltas: Optional[torch.Tensor] = None,
        batch_info: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Qwen3_5MoeOutput:
        """Multimodal forward: vision encoding + embedding merge + mRoPE + text model.

        3D mRoPE positions are computed per request at forward time using
        ``cu_seqlen`` to identify request boundaries and ``image_grid_thw``
        to derive spatial positions for multimodal requests.

        Position assembly cases:

        1. Images present + ``batch_info`` (mixed or prefill-only with images):
           iterate prefill requests via ``cu_seqlen``, call
           ``compute_mrope_positions`` for multimodal requests and expand 2D
           positions to 3D for text-only requests.  Decode tokens get
           delta-adjusted 3D expansion.
        2. Otherwise (decode-only or text-only prefill without images): expand
           ``position_ids + delta`` to 3D where delta defaults to 0.
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)

        has_images = pixel_values is not None and image_grid_thw is not None
        has_videos = pixel_values_videos is not None and video_grid_thw is not None

        image_embeds_list = None
        if has_images:
            image_embeds_list = [
                embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                for embeds in self.get_image_features(pixel_values, image_grid_thw)
            ]

        video_embeds_list = None
        if has_videos:
            video_embeds_list = [
                embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                for embeds in self.get_video_features(pixel_values_videos, video_grid_thw)
            ]
        video_span_embeds_list = self._expand_video_embeds_by_span(
            video_embeds_list, video_grid_thw
        )

        delta = mrope_position_deltas if mrope_position_deltas is not None else 0

        vision_grid = image_grid_thw if has_images else video_grid_thw if has_videos else None
        if batch_info is None:
            batch_info = kwargs.get("batch_info_host")
        batch_info_host = kwargs.get("batch_info_host", batch_info)
        cu_seqlen = kwargs.get("cu_seqlen")
        if cu_seqlen is None:
            cu_seqlen = kwargs.get("cu_seqlen_host")
        seq_len = kwargs.get("seq_len")
        if seq_len is None and cu_seqlen is not None:
            seq_len = cu_seqlen[1:] - cu_seqlen[:-1]
        input_pos = kwargs.get("input_pos")
        if input_pos is None:
            seq_len_with_cache = kwargs.get("seq_len_with_cache")
            if seq_len_with_cache is None:
                seq_len_with_cache = kwargs.get("seq_len_with_cache_host")
            if seq_len_with_cache is not None and seq_len is not None:
                input_pos = seq_len_with_cache.to(seq_len.device) - seq_len
        mm_item_cu_seqlen = kwargs.get("mm_item_cu_seqlen")
        mm_token_positions = kwargs.get("mm_token_positions")
        mm_token_lengths = kwargs.get("mm_token_lengths")
        mm_item_types = kwargs.get("mm_item_types")
        mm_special_offsets_cu_seqlen = kwargs.get("mm_special_offsets_cu_seqlen")
        mm_special_offsets = kwargs.get("mm_special_offsets")
        slot_idx = kwargs.get("slot_idx")
        mrope_delta_cache = kwargs.get("mrope_delta_cache")
        if mrope_delta_cache is None:
            for key, value in kwargs.items():
                if key.endswith("_mrope_delta_cache"):
                    mrope_delta_cache = value
                    break
        has_chunk_mm_layout = (
            mm_item_cu_seqlen is not None
            and mm_item_types is not None
            and mm_token_positions is not None
            and mm_token_lengths is not None
            and mm_item_cu_seqlen.numel() > 0
            and int(mm_item_cu_seqlen[-1].item()) > 0
            and mm_token_positions.numel() > 0
            and mm_token_lengths.numel() > 0
        )

        if has_images or has_videos:
            multimodal_mask = (
                (
                    (input_ids == self.config.image_token_id)
                    | (input_ids == self.config.video_token_id)
                )
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            num_multimodal_tokens = int(
                (
                    (input_ids == self.config.image_token_id)
                    | (input_ids == self.config.video_token_id)
                )
                .sum()
                .item()
            )
            if (
                batch_info is not None
                and cu_seqlen is not None
                and input_pos is not None
                and seq_len is not None
                and has_chunk_mm_layout
            ):
                multimodal_embeds = self._build_chunked_multimodal_embeds(
                    input_ids=input_ids,
                    batch_info=batch_info,
                    cu_seqlen=cu_seqlen,
                    input_pos=input_pos,
                    seq_len=seq_len,
                    image_embeds_list=image_embeds_list,
                    video_span_embeds_list=video_span_embeds_list,
                    mm_item_cu_seqlen=mm_item_cu_seqlen,
                    mm_item_types=mm_item_types,
                    mm_token_positions=mm_token_positions,
                    mm_token_lengths=mm_token_lengths,
                    mm_special_offsets_cu_seqlen=mm_special_offsets_cu_seqlen,
                    mm_special_offsets=mm_special_offsets,
                )
            else:
                if image_embeds_list is not None:
                    image_embeds = torch.cat(image_embeds_list, dim=0)
                    image_mask = (
                        (input_ids == self.config.image_token_id)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                if video_embeds_list is not None:
                    video_embeds = torch.cat(video_embeds_list, dim=0)
                    video_mask = (
                        (input_ids == self.config.video_token_id)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                multimodal_embeds = None

            if (
                multimodal_embeds is not None
                and multimodal_embeds.shape[0] != num_multimodal_tokens
            ):
                raise ValueError(
                    "Multimodal embedding count mismatch in Qwen3.5 VLM forward: "
                    f"selected={multimodal_embeds.shape[0]}, placeholders={num_multimodal_tokens}, "
                    f"input_shape={tuple(input_ids.shape)}"
                )
            if multimodal_embeds is not None:
                inputs_embeds = inputs_embeds.masked_scatter(multimodal_mask, multimodal_embeds)
        if mrope_delta_cache is not None and batch_info_host is not None and slot_idx is not None:
            delta = torch.ops.auto_deploy.qwen3_mrope_delta_with_cache(
                batch_info_host,
                slot_idx,
                mm_item_cu_seqlen,
                mm_item_types,
                mm_token_lengths,
                mm_special_offsets_cu_seqlen,
                mm_special_offsets,
                image_grid_thw,
                video_grid_thw,
                mrope_delta_cache,
                self.config.vision_config.spatial_merge_size,
            ).to(input_ids.dtype)

        if (
            vision_grid is not None
            and batch_info is not None
            and cu_seqlen is not None
            and input_pos is not None
            and seq_len is not None
            and has_chunk_mm_layout
        ):
            position_ids_3d = self._build_chunked_multimodal_positions(
                input_ids,
                position_ids,
                delta,
                batch_info,
                cu_seqlen,
                input_pos,
                seq_len,
                image_grid_thw if has_images else None,
                video_grid_thw if has_videos else None,
                mm_item_cu_seqlen,
                mm_item_types,
                mm_token_positions,
                mm_token_lengths,
                mm_special_offsets_cu_seqlen,
                mm_special_offsets,
            )
        elif vision_grid is not None and batch_info is not None and cu_seqlen is not None:
            position_ids_3d = self._build_mixed_positions(
                input_ids,
                position_ids,
                delta,
                batch_info,
                cu_seqlen,
                image_grid_thw if has_images else None,
                video_grid_thw if has_videos else None,
            )
        elif vision_grid is not None:
            position_ids_3d, _ = compute_mrope_positions(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw if has_images else None,
                video_grid_thw=video_grid_thw if has_videos else None,
                image_token_id=self.config.image_token_id,
                video_token_id=self.config.video_token_id,
                vision_start_token_id=self.config.vision_start_token_id,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
            )
        else:
            if position_ids is None:
                raise ValueError("position_ids is required for text-only or decode-only forward")
            is_flattened_cached_layout = position_ids.ndim == 1 or (
                position_ids.ndim == 2 and position_ids.shape[0] == 1
            )
            if is_flattened_cached_layout:
                flat_position_ids = position_ids.reshape(-1)
                token_delta = torch.zeros_like(flat_position_ids)
                if (
                    torch.is_tensor(delta)
                    and cu_seqlen is not None
                    and delta.ndim == 2
                    and delta.shape[0] == cu_seqlen.numel() - 1
                ):
                    seq_lens = (cu_seqlen[1:] - cu_seqlen[:-1]).to(torch.long)
                    token_delta = torch.repeat_interleave(
                        delta.squeeze(-1).to(flat_position_ids.device, flat_position_ids.dtype),
                        seq_lens.to(flat_position_ids.device),
                    )
                position_ids_3d = (flat_position_ids + token_delta).view(1, 1, -1).expand(3, 1, -1)
            else:
                position_ids_3d = (position_ids + delta)[None].expand(3, -1, -1)

        for key in (
            "input_pos",
            "mm_chunk_flat_start",
            "mm_chunk_count",
            "mm_item_cu_seqlen",
            "mm_item_types",
            "mm_token_positions",
            "mm_token_lengths",
            "mm_special_offsets_cu_seqlen",
            "mm_special_offsets",
            "mrope_delta_cache",
        ):
            kwargs.pop(key, None)
        for key in list(kwargs.keys()):
            if key.endswith("_mrope_delta_cache"):
                kwargs.pop(key, None)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids_3d,
            **kwargs,
        )

    def _build_chunked_multimodal_positions(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        delta,
        batch_info: torch.Tensor,
        cu_seqlen: torch.Tensor,
        input_pos: torch.Tensor,
        seq_len: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        mm_item_cu_seqlen: torch.Tensor,
        mm_item_types: torch.Tensor,
        mm_token_positions: torch.Tensor,
        mm_token_lengths: torch.Tensor,
        mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
        mm_special_offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build 3D positions using chunk runtime metadata from the executor.

        This path is for chunked multimodal prefill where ``input_ids`` only contains the current
        chunk but full multimodal tensors are still available. It reconstructs the chunk's 3D
        mRoPE positions in absolute request coordinates from:
        - per-request chunk start/end (`input_pos`, `seq_len`)
        - per-request multimodal item layout (`mm_token_positions`, `mm_token_lengths`)
        - full multimodal grids (`image_grid_thw` / `video_grid_thw`)
        """
        num_prefill_seqs = batch_info[0].item()
        num_prefill_tokens = batch_info[1].item()

        img_grid_idx = 0
        vid_grid_idx = 0
        prefill_3d_parts: list[torch.Tensor] = []
        normalized_video_grid_thw = _normalize_video_grid_for_mrope(video_grid_thw)

        for i in range(num_prefill_seqs):
            start = cu_seqlen[i].item()
            end = cu_seqlen[i + 1].item()
            req_input_pos = int(input_pos[i].item())
            req_seq_len = int(seq_len[i].item())

            item_start = int(mm_item_cu_seqlen[i].item())
            item_end = int(mm_item_cu_seqlen[i + 1].item())
            req_mm_item_types = mm_item_types[item_start:item_end].tolist()
            req_mm_positions = mm_token_positions[item_start:item_end].tolist()
            req_mm_lengths = mm_token_lengths[item_start:item_end].tolist()

            req_special_offsets: list[int] = []
            if mm_special_offsets_cu_seqlen is not None and mm_special_offsets is not None:
                special_start = int(mm_special_offsets_cu_seqlen[i].item())
                special_end = int(mm_special_offsets_cu_seqlen[i + 1].item())
                req_special_offsets = mm_special_offsets[special_start:special_end].tolist()

            has_img = image_grid_thw is not None and len(req_mm_positions) > 0
            has_vid = normalized_video_grid_thw is not None and len(req_mm_positions) > 0

            if has_img or has_vid:
                req_img_grid = None
                req_vid_grid = None
                num_images = sum(item_type == 0 for item_type in req_mm_item_types)
                num_videos = sum(item_type == 1 for item_type in req_mm_item_types)
                if has_img:
                    req_img_grid = image_grid_thw[img_grid_idx : img_grid_idx + num_images]
                    img_grid_idx += num_images
                if has_vid:
                    req_vid_grid = normalized_video_grid_thw[
                        vid_grid_idx : vid_grid_idx + num_videos
                    ]
                    vid_grid_idx += num_videos

                pos_3d = self._compute_request_chunk_mrope_positions(
                    req_input_pos=req_input_pos,
                    req_seq_len=req_seq_len,
                    req_mm_item_types=req_mm_item_types,
                    req_mm_positions=req_mm_positions,
                    req_mm_lengths=req_mm_lengths,
                    req_special_offsets=req_special_offsets,
                    image_grid_thw=req_img_grid,
                    video_grid_thw=req_vid_grid,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                prefill_3d_parts.append(pos_3d)
            else:
                if position_ids is not None:
                    req_pos = position_ids[..., start:end]
                else:
                    req_pos = torch.arange(
                        req_input_pos,
                        req_input_pos + req_seq_len,
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    ).unsqueeze(0)
                prefill_3d_parts.append(req_pos[None].expand(3, -1, -1))

        prefill_pos = torch.cat(prefill_3d_parts, dim=-1)

        if num_prefill_tokens < input_ids.shape[-1]:
            if position_ids is None:
                raise ValueError("position_ids is required when decode tokens are present")
            decode_pos_2d = position_ids[..., num_prefill_tokens:]
            if isinstance(delta, torch.Tensor):
                gen_deltas = delta[num_prefill_seqs:]
                decode_adjusted = decode_pos_2d + gen_deltas.T
            else:
                decode_adjusted = decode_pos_2d + delta
            decode_pos_3d = decode_adjusted[None].expand(3, -1, -1)
            return torch.cat([prefill_pos, decode_pos_3d], dim=-1)

        return prefill_pos

    def _compute_request_chunk_mrope_positions(
        self,
        req_input_pos: int,
        req_seq_len: int,
        req_mm_item_types: Sequence[int],
        req_mm_positions: Sequence[int],
        req_mm_lengths: Sequence[int],
        req_special_offsets: Sequence[int],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute chunk-local 3D mRoPE positions for one request in absolute coordinates."""
        chunk_end = req_input_pos + req_seq_len
        out = torch.empty((3, 1, req_seq_len), dtype=dtype, device=device)
        special_offsets_set = set(int(x) for x in req_special_offsets)
        mm_cumulative_offset = 0
        abs_cursor = 0
        comp_cursor = 0
        img_idx = 0
        vid_idx = 0

        def fill_text(abs_start: int, abs_end: int, comp_start: int) -> None:
            ov_start = max(req_input_pos, abs_start)
            ov_end = min(chunk_end, abs_end)
            if ov_start >= ov_end:
                return
            start_pos = comp_start + (ov_start - abs_start)
            text_pos = torch.arange(
                start_pos, start_pos + (ov_end - ov_start), device=device, dtype=dtype
            )
            out[:, 0, ov_start - req_input_pos : ov_end - req_input_pos] = text_pos.unsqueeze(
                0
            ).expand(3, -1)

        def fill_vision(abs_start: int, grid: torch.Tensor, comp_start: int) -> Tuple[int, int]:
            t, h, w = [int(v) for v in grid.tolist()]
            llm_grid_t = int(t)
            llm_grid_h = int(h) // self.config.vision_config.spatial_merge_size
            llm_grid_w = int(w) // self.config.vision_config.spatial_merge_size
            vision_len = llm_grid_t * llm_grid_h * llm_grid_w

            ov_start = max(req_input_pos, abs_start)
            ov_end = min(chunk_end, abs_start + vision_len)
            if ov_start < ov_end:
                t_index = (
                    torch.arange(llm_grid_t, device=device, dtype=dtype)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h, device=device, dtype=dtype)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w, device=device, dtype=dtype)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                positions = torch.stack([t_index, h_index, w_index]) + comp_start
                local_start = ov_start - abs_start
                local_end = ov_end - abs_start
                out[:, 0, ov_start - req_input_pos : ov_end - req_input_pos] = positions[
                    :, local_start:local_end
                ]

            return vision_len, comp_start + max(llm_grid_t, llm_grid_h, llm_grid_w)

        for item_type, mm_start, mm_len in zip(req_mm_item_types, req_mm_positions, req_mm_lengths):
            item_mm_offset = mm_cumulative_offset
            leading_specials = 0
            while item_mm_offset + leading_specials in special_offsets_set:
                leading_specials += 1

            vision_abs_start = int(mm_start) + leading_specials
            fill_text(abs_cursor, vision_abs_start, comp_cursor)
            comp_cursor += vision_abs_start - abs_cursor

            if item_type == 0:
                if image_grid_thw is None:
                    raise ValueError("Missing image_grid_thw for image multimodal item")
                grid = image_grid_thw[img_idx]
                img_idx += 1
            elif item_type == 1:
                if video_grid_thw is None:
                    raise ValueError("Missing video_grid_thw for video multimodal item")
                grid = video_grid_thw[vid_idx]
                vid_idx += 1
            else:
                raise ValueError(f"Unsupported multimodal item type: {item_type}")

            _, next_comp_cursor = fill_vision(vision_abs_start, grid, comp_cursor)
            comp_cursor = next_comp_cursor
            abs_cursor = int(mm_start) + int(mm_len)
            mm_cumulative_offset += int(mm_len)

        fill_text(abs_cursor, chunk_end, comp_cursor)
        return out

    def _build_mixed_positions(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        delta,
        batch_info: torch.Tensor,
        cu_seqlen: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Build 3D mRoPE positions for a batch with per-request granularity.

        Iterates over prefill requests using ``cu_seqlen`` boundaries.  For
        each request that contains vision tokens, calls
        ``compute_mrope_positions`` with the matching ``image_grid_thw`` rows.
        Text-only prefill requests get trivial 3D expansion.  Decode tokens
        are delta-adjusted uniformly.
        """
        num_prefill_seqs = batch_info[0].item()
        num_prefill_tokens = batch_info[1].item()

        img_grid_idx = 0
        vid_grid_idx = 0
        prefill_3d_parts: list = []

        for i in range(num_prefill_seqs):
            start = cu_seqlen[i].item()
            end = cu_seqlen[i + 1].item()
            req_ids = input_ids[..., start:end]

            has_img = image_grid_thw is not None and (req_ids == self.config.image_token_id).any()
            has_vid = video_grid_thw is not None and (req_ids == self.config.video_token_id).any()

            if has_img or has_vid:
                req_img_grid = None
                req_vid_grid = None
                req_item_types = _extract_mm_item_types_from_input_ids(
                    req_ids,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=self.config.vision_start_token_id,
                )
                if has_img:
                    n_img = sum(item_type == 0 for item_type in req_item_types)
                    req_img_grid = image_grid_thw[img_grid_idx : img_grid_idx + n_img]
                    img_grid_idx += n_img
                if has_vid:
                    n_vid = sum(item_type == 1 for item_type in req_item_types)
                    req_vid_grid = video_grid_thw[vid_grid_idx : vid_grid_idx + n_vid]
                    vid_grid_idx += n_vid

                pos_3d, _ = compute_mrope_positions(
                    input_ids=req_ids,
                    image_grid_thw=req_img_grid,
                    video_grid_thw=req_vid_grid,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=self.config.vision_start_token_id,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                )
                prefill_3d_parts.append(pos_3d)
            else:
                req_pos = position_ids[..., start:end]
                prefill_3d_parts.append((req_pos + 0)[None].expand(3, -1, -1))

        prefill_pos = torch.cat(prefill_3d_parts, dim=-1)

        if num_prefill_tokens < input_ids.shape[-1]:
            decode_pos_2d = position_ids[..., num_prefill_tokens:]
            if isinstance(delta, torch.Tensor):
                num_prefill_seqs_t = batch_info[0].item()
                gen_deltas = delta[num_prefill_seqs_t:]
                decode_adjusted = decode_pos_2d + gen_deltas.T
            else:
                decode_adjusted = decode_pos_2d + delta
            decode_pos_3d = decode_adjusted[None].expand(3, -1, -1)
            return torch.cat([prefill_pos, decode_pos_3d], dim=-1)

        return prefill_pos


class Qwen3_5MoeForConditionalGeneration(Qwen3_5MoePreTrainedModel):
    """Top-level multimodal model: vision + language model + lm_head.

    This wraps ``Qwen3_5MoeModel`` (which contains the vision tower and the
    text ``Qwen3_5MoeTextModel`` as ``language_model``) and adds an ``lm_head``
    at the top level -- matching the HF checkpoint weight layout.
    """

    config_class = Qwen3_5MoeConfig

    def __init__(self, config: Qwen3_5MoeConfig, **kwargs):
        super().__init__(config)
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        # Share lm_head with the text model so it's inside the exported graph
        self.model.language_model.set_lm_head(self.lm_head)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3_5MoeConditionalOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        logits = outputs.logits
        return Qwen3_5MoeConditionalOutput(logits=logits)


# =============================================================================
# Custom Export Info and Factory
# =============================================================================


class Qwen3_5MoeTextExportInfo(TextModelExportInfo):
    """Export info for mRoPE models.

    Exports the full model (embed + decoder + norm + lm_head) as a single
    GraphModule so that sharding and graph-level transforms (e.g.
    ``gather_logits_before_lm_head``) can see lm_head directly, and piecewise
    CUDA graph capture receives a GraphModule at the top level.

    When ``submodule_name=""`` (full model export), position_ids enters as 2D
    ``(B, S)`` and is expanded to 3D inside the graph. When exporting an inner
    submodule, position_ids enters as 3D ``(3, B, S)`` with a static dim 0.
    """

    def __init__(self, submodule_name: str):
        super().__init__(submodule_name)

    def _init_dynamic_shape_lookup(self):
        if self.is_full_model_export:
            # Full model export: position_ids enters as 2D (B, S), expanded
            # to 3D inside the graph by the wrapper's fast path.
            batch_size_dyn = Dim.DYNAMIC
            seq_len_dyn = Dim.DYNAMIC
            return {
                "input_ids": {0: batch_size_dyn, 1: seq_len_dyn},
                "position_ids": {0: batch_size_dyn, 1: seq_len_dyn},
            }
        # Inner model export: position_ids enters as 3D (3, B, S).
        base = super()._init_dynamic_shape_lookup()
        batch_size_dyn = Dim.DYNAMIC
        seq_len_dyn = Dim.DYNAMIC
        base["position_ids"] = {1: batch_size_dyn, 2: seq_len_dyn}
        return base

    @classmethod
    def from_autoinferred(cls, model: nn.Module) -> "Qwen3_5MoeTextExportInfo":
        """Export the full model as a single GraphModule.

        Using submodule_name="" exports the entire model (embedding + text model
        + lm_head) as one FX GraphModule. This allows piecewise CUDA graph capture
        since compile_model receives a GraphModule directly.
        """
        return cls("")


class Qwen3_5MoeADInputProcessor:
    """Qwen-specific AD input processor that emits exact multimodal spans from tokenized input."""

    def __init__(self, base_processor):
        self.base_processor = base_processor
        # Bypass the generic hashing wrapper. We produce multimodal_input directly.
        self.multimodal_hashing_supported = False

    def __getattr__(self, name: str):
        return getattr(self.base_processor, name)

    @property
    def get_num_multimodal_tokens(self):
        """Delegate multimodal token counting to the wrapped Qwen HF processor."""
        if hasattr(self.processor, "_get_num_multimodal_tokens"):
            return self.processor._get_num_multimodal_tokens
        raise NotImplementedError(
            f"get_num_multimodal_tokens not implemented for {self.__class__.__name__}. "
            "Please ensure the processor exposes _get_num_multimodal_tokens."
        )

    def get_num_tokens_per_image(self, *, image: Image.Image, **kwargs) -> int:
        image_size = (image.height, image.width)
        return self.get_num_multimodal_tokens([image_size], **kwargs)["num_image_tokens"][0]

    def get_num_tokens_per_video(self, *, video: List[Image.Image], **kwargs) -> int:
        video_size = (len(video), video[0].height, video[0].width)
        num_video_tokens = self.get_num_multimodal_tokens(video_sizes=[video_size], **kwargs).get(
            "num_video_tokens"
        )
        if num_video_tokens is None:
            raise NotImplementedError("Underlying processor does not expose num_video_tokens.")
        return num_video_tokens[0]

    def get_vocab_size(self) -> Optional[int]:
        """Return the tokenizer vocabulary size for Qwen multimodal hashing helpers."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "vocab_size"):
            return int(self.tokenizer.vocab_size)
        wrapped_tokenizer = getattr(self.tokenizer, "tokenizer", None)
        if wrapped_tokenizer is not None and hasattr(wrapped_tokenizer, "vocab_size"):
            return int(wrapped_tokenizer.vocab_size)
        processor_tokenizer = getattr(self.processor, "tokenizer", None)
        if processor_tokenizer is not None and hasattr(processor_tokenizer, "vocab_size"):
            return int(processor_tokenizer.vocab_size)
        return None

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        if hasattr(self.processor, "mm_token_ids"):
            return self.processor.mm_token_ids
        sources = [
            self.processor,
            getattr(self.processor, "tokenizer", None),
            self.tokenizer,
            getattr(self.tokenizer, "tokenizer", None),
        ]
        token_ids = []
        for source in sources:
            if source is None:
                continue
            for attr in ("image_token_id", "video_token_id"):
                value = getattr(source, attr, None)
                if value is not None:
                    token_ids.append(int(value))
        if token_ids:
            return torch.tensor(sorted(set(token_ids)), dtype=torch.int32)
        return None

    def get_mm_special_token_ids(self) -> Optional[torch.Tensor]:
        if hasattr(self.processor, "mm_special_token_ids"):
            return self.processor.mm_special_token_ids
        sources = [
            self.processor,
            getattr(self.processor, "tokenizer", None),
            self.tokenizer,
            getattr(self.tokenizer, "tokenizer", None),
        ]
        token_ids = []
        for source in sources:
            if source is None:
                continue
            for attr in ("vision_start_token_id", "vision_end_token_id"):
                value = getattr(source, attr, None)
                if value is not None:
                    token_ids.append(int(value))
        if token_ids:
            return torch.tensor(sorted(set(token_ids)), dtype=torch.int32)
        return None

    def _build_multimodal_input(
        self,
        token_ids: List[int],
        inputs: Dict[str, Any],
    ) -> Optional[Tuple[MultimodalInput, List[int], List[int]]]:
        mm_data = inputs.get("multi_modal_data")
        if not mm_data or not any(k in mm_data for k in ("image", "video")):
            return None

        image_token_id = int(self.processor.image_token_id)
        video_token_id = int(self.processor.video_token_id)
        vision_start_token_id = int(self.processor.vision_start_token_id)
        ids = token_ids

        starts: List[int] = []
        lengths: List[int] = []
        special_offsets: List[int] = []
        item_types: List[int] = []
        mm_union_offset = 0
        i = 0
        while i < len(ids):
            if ids[i] != vision_start_token_id:
                i += 1
                continue

            if i + 1 >= len(ids):
                i += 1
                continue

            if ids[i + 1] == image_token_id:
                item_token_id = image_token_id
                item_type = 0
            elif ids[i + 1] == video_token_id:
                item_token_id = video_token_id
                item_type = 1
            else:
                i += 1
                continue

            j = i + 1
            while j < len(ids) and ids[j] == item_token_id:
                j += 1
            if j == i + 1:
                i += 1
                continue

            starts.append(i)
            lengths.append(j - i)
            special_offsets.append(mm_union_offset)
            item_types.append(item_type)
            mm_union_offset += j - i
            i = j

        image_items = _normalize_qwen_image_items(mm_data.get("image"))
        video_items = _normalize_qwen_video_items(mm_data.get("video"))

        num_video_spans = sum(item_type == 1 for item_type in item_types)
        video_span_counts = [_get_qwen_video_num_spans(video) for video in video_items]
        if num_video_spans != sum(video_span_counts):
            raise ValueError(
                "Mismatch between Qwen video prompt spans and video inputs: "
                f"spans={num_video_spans}, expected_from_videos={sum(video_span_counts)}"
            )

        if len(starts) != len(image_items) + num_video_spans:
            raise ValueError(
                "Mismatch between multimodal prompt spans and multimodal items: "
                f"spans={len(starts)}, images={len(image_items)}, video_spans={num_video_spans}"
            )

        mm_uuids = inputs.get("multi_modal_uuids", None)
        mm_hash_inputs = {}
        if image_items:
            mm_hash_inputs["image"] = image_items
        if video_items:
            mm_hash_inputs["video"] = video_items
        mm_hashes, _ = apply_mm_hashes(mm_hash_inputs, mm_uuids)

        image_hashes = [hexdigest_to_int32(h) for h in mm_hashes.get("image", [])]
        video_hashes = [hexdigest_to_int32(h) for h in mm_hashes.get("video", [])]
        image_uuids = list((mm_uuids or {}).get("image", [None] * len(image_items)))
        video_uuids = list((mm_uuids or {}).get("video", [None] * len(video_items)))
        image_idx = 0
        video_idx = 0
        remaining_video_spans = video_span_counts[0] if video_span_counts else 0
        mm_hashes_flat: List[List[int]] = []
        mm_uuid_list: List[Optional[str]] = []
        for item_type in item_types:
            if item_type == 0:
                mm_hashes_flat.append(image_hashes[image_idx])
                mm_uuid_list.append(image_uuids[image_idx])
                image_idx += 1
            else:
                if video_idx >= len(video_hashes):
                    raise ValueError("Video span count exceeded available video items")
                mm_hashes_flat.append(video_hashes[video_idx])
                mm_uuid_list.append(video_uuids[video_idx])
                remaining_video_spans -= 1
                if remaining_video_spans == 0:
                    video_idx += 1
                    remaining_video_spans = (
                        video_span_counts[video_idx] if video_idx < len(video_span_counts) else 0
                    )
        return (
            MultimodalInput.from_components(
                mm_hashes_flat,
                starts,
                lengths,
                mm_uuid_list if mm_uuids is not None else None,
            ),
            special_offsets,
            item_types,
        )

    def __call__(self, inputs, sampling_params):
        token_ids, extra_processed_inputs = self.base_processor(inputs, sampling_params)
        if "multi_modal_data" not in inputs:
            return token_ids, extra_processed_inputs

        built = self._build_multimodal_input(token_ids, inputs)
        if built is None:
            return token_ids, extra_processed_inputs

        multimodal_input, special_offsets, item_types = built
        if extra_processed_inputs is None:
            extra_processed_inputs = {}
        extra_processed_inputs["multimodal_input"] = multimodal_input
        multimodal_data = extra_processed_inputs.get("multimodal_data", {})
        multimodal_data["layout_metadata"] = {
            "special_token_offsets": torch.tensor(special_offsets, dtype=torch.int32),
            "item_types": torch.tensor(item_types, dtype=torch.int32),
        }
        extra_processed_inputs["multimodal_data"] = multimodal_data
        return token_ids, extra_processed_inputs


@ModelFactoryRegistry.register("Qwen3_5MoeForConditionalGeneration")
class Qwen3_5MoeFactory(AutoModelForImageTextToTextFactory):
    """Factory for Qwen3.5 MoE that uses 3D mRoPE position_ids export info."""

    def get_export_infos(self, model: nn.Module):
        return [Qwen3_5MoeTextExportInfo.from_autoinferred(model)]

    def init_input_processor(self, base):
        return Qwen3_5MoeADInputProcessor(base)


# =============================================================================
# Registration
# =============================================================================

AutoConfig.register("qwen3_5_moe", Qwen3_5MoeConfig)
AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeTextConfig)

AutoModelForCausalLMFactory.register_custom_model_cls("Qwen3_5MoeTextConfig", Qwen3_5MoeForCausalLM)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "Qwen3_5MoeConfig", Qwen3_5MoeForConditionalGeneration
)
Qwen3_5MoeFactory.register_custom_model_cls("Qwen3_5MoeConfig", Qwen3_5MoeForConditionalGeneration)
