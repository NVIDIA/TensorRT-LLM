# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3-Next (MoE) model for auto_deploy (prefill-only).

Reference HF modeling file (v4.57.1):
  transformers/models/qwen3_next/modeling_qwen3_next.py

This implementation differs from the HuggingFace original in the following ways:
  * External kernel dependencies (flash-linear-attention, causal_conv1d) are replaced with
    autodeploy custom ops.
  * Cache-related code paths have been removed (prefill-only).
  * Training-related code paths have been removed.
  * Unnecessary output fields have been removed.
  * The GatedDeltaNet forward uses autodeploy custom ops:
    torch_causal_conv1d, torch_gated_delta_rule.
  * The MoE implementation uses expert lists (individual nn.Linear layers per expert)
    that directly match the checkpoint structure, dispatched via torch_moe op.
  * Rotary embedding is computed once at the model level and passed to all layers.

This allows us to have a "pytorch" native reference implementation decoupled from bugs and
dependency issues in the source, while remaining weight-compatible with HF checkpoints.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

# =============================================================================
# Normalization
# =============================================================================


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm with ``(1 + w)`` parameterisation.

    The HF checkpoint stores zero-initialised weights. A load-time pre-hook
    adds 1.0 so that the forward can use a plain ``weight * x`` multiply,
    matching the autodeploy RMSNorm pattern for fusion.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._register_load_state_dict_pre_hook(self._offset_weight)

    @staticmethod
    def _offset_weight(state_dict, prefix, *args):
        key = prefix + "weight"
        if key in state_dict:
            state_dict[key] = state_dict[key] + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


class Qwen3NextRMSNormGated(nn.Module):
    """Gated RMSNorm: norm(x) * weight * silu(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm_gated(
            hidden_states,
            self.weight,
            gate,
            self.variance_epsilon,
            group_size=self.hidden_size,
            norm_before_gate=True,
        )


# =============================================================================
# Rotary Position Embedding
# =============================================================================


class Qwen3NextRotaryEmbedding(nn.Module):
    """Standard RoPE for Qwen3Next (not mRoPE).

    Precomputes and caches cos/sin tables. Returns the full table from forward;
    downstream attention layers slice by position_ids before applying RoPE via
    the canonical ``torch_rope_with_explicit_cos_sin`` op.
    """

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        base = getattr(config, "rope_theta", 10000.0)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache with _ad_ prefix for AutoDeploy compatibility
        max_pos = config.max_position_embeddings
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full cached (cos, sin) tables, shape (max_pos, rotary_dim)."""
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


# =============================================================================
# GatedDeltaNet (Linear Attention)
# =============================================================================


class Qwen3NextGatedDeltaNet(nn.Module):
    """Prefill-only GatedDeltaNet using autodeploy custom ops.

    Uses fused projections (in_proj_qkvz, in_proj_ba) matching the checkpoint.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
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

        # Fused projections matching checkpoint structure
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        # dt_bias and A_log for gated delta rule
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm (per head_v_dim)
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """Derives query, key, value, z, b, a from fused projections."""
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Fused projections
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

        # 3. Gated Delta Rule via autodeploy custom op.
        # L2 norm, GQA expansion, and g/beta computation are handled inside the op.
        core_attn_out = torch.ops.auto_deploy.torch_gated_delta_rule(
            query, key, value, a, b, self.A_log, self.dt_bias
        )

        # 5. Gated RMSNorm
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # 6. Output projection
        output = self.out_proj(core_attn_out)
        return output


# =============================================================================
# Attention
# =============================================================================


class Qwen3NextAttention(nn.Module):
    """Multi-headed attention with gating, Q/K norms, and partial RoPE.

    Key differences from standard attention:
      - q_proj outputs 2x (query + gate), gating applied to attention output.
      - q_norm / k_norm applied per-head before RoPE.
      - Partial RoPE: only first rotary_dim dimensions are rotated, using the
        canonical ``torch_rope_with_explicit_cos_sin`` op.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)

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

        # Per-head Q/K norms (with +1 offset hook)
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

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

        # Position embeddings already sliced by position_ids at model level
        cos, sin = position_embeddings  # (B, S, rotary_dim)

        # Split into rotary and pass-through portions for partial RoPE
        q_rot = query_states[..., : self.rotary_dim]  # (B, S, N, rotary_dim)
        q_pass = query_states[..., self.rotary_dim :]  # (B, S, N, head_dim - rotary_dim)
        k_rot = key_states[..., : self.rotary_dim]
        k_pass = key_states[..., self.rotary_dim :]

        # Apply RoPE using canonical op (unsqueeze_dim=2 for BSND layout)
        q_rot, k_rot = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q_rot, k_rot, cos, sin, 2
        )

        # Concatenate rotary and pass-through portions back
        query_states = torch.cat([q_rot, q_pass], dim=-1)
        key_states = torch.cat([k_rot, k_pass], dim=-1)

        # Attention via autodeploy op (bsnd layout, handles GQA natively)
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
# MLP
# =============================================================================


class Qwen3NextMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: Qwen3NextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Mixture of Experts
# =============================================================================


class Qwen3NextSparseMoeBlock(nn.Module):
    """MoE layer with softmax + topk routing and a shared expert."""

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Gate
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Routed experts as ModuleList (matches checkpoint structure)
        self.experts = nn.ModuleList(
            [
                Qwen3NextMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

        # Shared expert
        self.shared_expert = Qwen3NextMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states_flat)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        # Routed experts via torch_moe
        final_hidden_states = torch.ops.auto_deploy.torch_moe(
            hidden_states_flat,
            selected_experts,
            routing_weights,
            w1_weight=[expert.gate_proj.weight for expert in self.experts],
            w2_weight=[expert.down_proj.weight for expert in self.experts],
            w3_weight=[expert.up_proj.weight for expert in self.experts],
        )

        # Shared expert path
        shared_expert_output = self.shared_expert(hidden_states_flat)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


# =============================================================================
# Decoder Layer
# =============================================================================


class Qwen3NextDecoderLayer(nn.Module):
    """Single decoder layer: token mixer (linear_attention or full_attention) + MLP/MoE."""

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        # MLP or MoE based on decoder_sparse_step
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (
            layer_idx not in mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(
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
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        hidden_states = residual + hidden_states

        # Channel mixer (MLP or MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model
# =============================================================================


class Qwen3NextPreTrainedModel(PreTrainedModel):
    """Base class for Qwen3Next pretrained models."""

    config_class = Qwen3NextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3NextDecoderLayer"]
    supports_gradient_checkpointing = False
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3NextRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Qwen3NextRMSNormGated):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Qwen3NextGatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())


@dataclass
class Qwen3NextOutput(ModelOutput):
    """Output of the Qwen3Next model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Qwen3NextCausalLMOutput(ModelOutput):
    """Output of the Qwen3Next causal language model."""

    logits: Optional[torch.FloatTensor] = None


class Qwen3NextModel(Qwen3NextPreTrainedModel):
    """Qwen3Next transformer decoder model."""

    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Qwen3NextOutput:
        assert position_ids is not None, "position_ids must be provided"

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute position embeddings: get full table then slice once by position_ids
        cos_full, sin_full = self.rotary_emb(inputs_embeds)
        position_embeddings = (
            cos_full[position_ids],  # (B, S, rotary_dim)
            sin_full[position_ids],  # (B, S, rotary_dim)
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return Qwen3NextOutput(last_hidden_state=hidden_states)


class Qwen3NextForCausalLM(Qwen3NextPreTrainedModel, GenerationMixin):
    """Qwen3Next causal language model (text model + lm_head)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3NextConfig, **kwargs):
        super().__init__(config)
        self.model = Qwen3NextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Qwen3NextCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()
        return Qwen3NextCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("Qwen3NextConfig", Qwen3NextForCausalLM)
