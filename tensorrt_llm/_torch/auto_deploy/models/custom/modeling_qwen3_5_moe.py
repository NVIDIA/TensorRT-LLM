# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Slimmed-down, prefill-only Qwen3.5 MoE text model for auto_deploy.

Reference HF modeling file (not yet in a released transformers version):
  transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

This implementation differs from the HuggingFace original in the following ways:
  * External kernel dependencies (flash-linear-attention, causal_conv1d) are replaced with
    autodeploy custom ops.
  * Cache-related code paths have been removed (prefill-only).
  * Training-related code paths have been removed.
  * Vision model components have been removed (text-only).
  * Unnecessary output fields have been removed.
  * The GatedDeltaNet forward is adapted from the Qwen3Next GDN patch
    (tensorrt_llm/_torch/auto_deploy/models/patches/qwen3_next.py).
  * The MoE forward is adapted from the Qwen3Next MoE patch.

This allows us to have a "pytorch" native reference implementation decoupled from bugs and
dependency issues in the source, while remaining weight-compatible with HF checkpoints.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

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
    """RMSNorm with (1 + weight) scaling. Weight is initialized to zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


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
# Uses autodeploy custom ops: torch_causal_conv1d, torch_l2norm, torch_gated_delta_rule


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
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [B, S, num_v_heads, head_v_dim]
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


class Qwen3_5MoeExperts(nn.Module):
    """Expert weights stored as fused 3D tensors for checkpoint compatibility.

    Parameters:
        gate_up_proj: shape [num_experts, 2 * intermediate_dim, hidden_dim]
        down_proj:    shape [num_experts, hidden_dim, intermediate_dim]
    """

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )


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
    """MoE block using torch_moe custom op for routed experts.

    Adapted from the Qwen3Next MoE patch. Splits the fused gate_up_proj into
    per-expert gate and up weight lists for the torch_moe op.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router
        routing_weights, selected_experts = self.gate(hidden_states_flat)

        # Split fused expert weights into per-expert gate / up / down lists
        I = self.experts.intermediate_dim  # noqa: E741
        E = self.experts.num_experts
        gate_up = self.experts.gate_up_proj  # [E, 2*I, H]
        gate_proj_weights = [gate_up[i, :I, :] for i in range(E)]
        up_proj_weights = [gate_up[i, I:, :] for i in range(E)]
        down_proj_weights = [self.experts.down_proj[i] for i in range(E)]

        # Routed experts via torch_moe (w1=gate, w2=down, w3=up per Qwen3Next convention)
        expert_output = torch.ops.auto_deploy.torch_moe(
            hidden_states_flat,
            selected_experts,
            routing_weights,
            w1_weight=gate_proj_weights,
            w2_weight=down_proj_weights,
            w3_weight=up_proj_weights,
        )

        # Shared expert with sigmoid gating
        shared_expert_output = self.shared_expert(hidden_states_flat)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output
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
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, Qwen3_5MoeRMSNorm):
            module.weight.data.zero_()
        elif isinstance(module, Qwen3_5MoeRMSNormGated):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Qwen3_5MoeGatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, Qwen3_5MoeExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
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
    """Qwen3.5 MoE text model (embed + decoder layers + final norm)."""

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

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5MoeOutput]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute position_ids for mRoPE (3, B, S)
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
        return Qwen3_5MoeOutput(last_hidden_state=hidden_states)


class Qwen3_5MoeForCausalLM(Qwen3_5MoePreTrainedModel, GenerationMixin):
    """Qwen3.5 MoE causal language model (text model + lm_head)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__(config)
        self.model = Qwen3_5MoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        **kwargs,
    ) -> Union[Tuple, Qwen3_5MoeCausalLMOutput]:
        outputs = self.model(input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return Qwen3_5MoeCausalLMOutput(logits=logits)


# =============================================================================
# Registration
# =============================================================================

AutoModelForCausalLMFactory.register_custom_model_cls("Qwen3_5MoeTextConfig", Qwen3_5MoeForCausalLM)
