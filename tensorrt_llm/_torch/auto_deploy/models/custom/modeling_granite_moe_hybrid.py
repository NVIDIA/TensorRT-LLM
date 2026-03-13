# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prefill-only PyTorch GraniteMoeHybrid model for auto_deploy export.

Source: https://huggingface.co/ibm-granite/granite-4.0-micro
        https://huggingface.co/ibm-granite/granite-4.0-tiny-preview

This implementation differs from the original HuggingFace version:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants, training paths, dropout
* Supports muP-style multipliers (embedding, attention, residual, logits)

Supports the full GraniteMoeHybrid architecture including:
* Mixed Mamba/Attention layers (selected per-layer via config.layer_types)
* Sparse MoE with shared MLP (when config.num_local_experts > 0)
* Optional RoPE (position_embedding_type="rope") or no positional encoding ("nope")
* Attention-only variants (e.g. granite-4.0-micro with all attention layers, no MoE)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

try:
    from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
        GraniteMoeHybridConfig,
    )
except ImportError:
    from transformers import PretrainedConfig

    class GraniteMoeHybridConfig(PretrainedConfig):
        """Fallback config for GraniteMoeHybrid when not in installed transformers."""

        model_type = "granitemoehybrid"

        def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            shared_intermediate_size=1024,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            embedding_multiplier=1.0,
            logits_scaling=1.0,
            residual_multiplier=1.0,
            attention_multiplier=1.0,
            num_local_experts=0,
            num_experts_per_tok=0,
            position_embedding_type=None,
            layer_types=None,
            tie_word_embeddings=False,
            # Mamba SSM parameters
            mamba_d_conv=4,
            mamba_d_state=128,
            mamba_d_head=64,
            mamba_n_heads=48,
            mamba_n_groups=1,
            mamba_expand=2,
            mamba_chunk_size=256,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.shared_intermediate_size = shared_intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = (
                num_key_value_heads if num_key_value_heads is not None else num_attention_heads
            )
            self.hidden_act = hidden_act
            self.max_position_embeddings = max_position_embeddings
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.attention_bias = attention_bias
            self.attention_dropout = attention_dropout
            self.embedding_multiplier = embedding_multiplier
            self.logits_scaling = logits_scaling
            self.residual_multiplier = residual_multiplier
            self.attention_multiplier = attention_multiplier
            self.num_local_experts = num_local_experts
            self.num_experts_per_tok = num_experts_per_tok
            self.position_embedding_type = position_embedding_type
            self.layer_types = layer_types
            self.mamba_d_conv = mamba_d_conv
            self.mamba_d_state = mamba_d_state
            self.mamba_d_head = mamba_d_head
            self.mamba_n_heads = mamba_n_heads
            self.mamba_n_groups = mamba_n_groups
            self.mamba_expand = mamba_expand
            self.mamba_chunk_size = mamba_chunk_size
            self.mamba_conv_bias = mamba_conv_bias
            self.mamba_proj_bias = mamba_proj_bias
            super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        @property
        def layers_block_type(self):
            return self.layer_types

    AutoConfig.register("granitemoehybrid", GraniteMoeHybridConfig, exist_ok=True)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GraniteMoeHybridModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GraniteMoeHybridCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class GraniteMoeHybridRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GraniteMoeHybridRMSNormGated(nn.Module):
    """Gated RMSNorm for Mamba layers: norm(x) * weight * silu(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary embedding (for attention layers with position_embedding_type="rope")
# ---------------------------------------------------------------------------


class GraniteMoeHybridRotaryEmbedding(nn.Module):
    """Rotary position embedding that pre-computes and caches the full cos/sin table."""

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        # Pre-compute full cos/sin table with AD-specific naming
        t = torch.arange(self.max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return full cached cos/sin (not sliced) for export compatibility
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


# ---------------------------------------------------------------------------
# Shared MLP (present on every layer)
# ---------------------------------------------------------------------------


class GraniteMoeHybridMLP(nn.Module):
    """MLP with fused gate+up projection (shared_mlp in HF checkpoint)."""

    def __init__(self, config):
        super().__init__()
        self.input_size = config.hidden_size
        self.hidden_size = config.shared_intermediate_size
        self.activation = ACT2FN[config.hidden_act]
        self.input_linear = nn.Linear(self.input_size, self.hidden_size * 2, bias=False)
        self.output_linear = nn.Linear(self.hidden_size, self.input_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_linear(hidden_states)
        chunked = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(chunked[0]) * chunked[1]
        hidden_states = self.output_linear(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Attention (supports both RoPE and no positional encoding)
# ---------------------------------------------------------------------------


class GraniteMoeHybridAttention(nn.Module):
    """GQA attention with muP scaling and optional RoPE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # muP: use attention_multiplier instead of 1/sqrt(head_dim)
        self.scaling = config.attention_multiplier

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [B, S, N, D] (bsnd layout for torch_attention)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply RoPE only if position_embeddings are provided (not for "nope" mode)
        if position_embeddings is not None:
            cos = position_embeddings[0]  # Full table: [max_seq_len, head_dim]
            sin = position_embeddings[1]  # Full table: [max_seq_len, head_dim]
            cos = cos[position_ids]  # [B, S, head_dim]
            sin = sin[position_ids]  # [B, S, head_dim]
            query_states, key_states = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
                query_states, key_states, cos, sin, 2
            )

        # AD-managed attention with muP scaling
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            scale=self.scaling,
            layout="bsnd",
        )

        # Reshape back: [B, S, N, D] -> [B, S, N*D]
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


# ---------------------------------------------------------------------------
# Mamba SSM layer (using AD custom ops for prefill)
# ---------------------------------------------------------------------------


class GraniteMoeHybridMambaLayer(nn.Module):
    """Mamba2-style SSM layer using AD custom ops for prefill-only inference.

    Uses torch_causal_conv1d for the 1D convolution and torch_ssm for the
    selective state space computation. No caching or decode-time paths.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * self.hidden_size)
        self.n_groups = config.mamba_n_groups
        self.chunk_size = config.mamba_chunk_size
        self.layer_idx = layer_idx
        self.act = ACT2FN[config.hidden_act]

        self.time_step_limit: List[float] = [0.0, float("inf")]

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.mamba_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Projection: gate + conv_input (hidden+B+C) + dt
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.mamba_proj_bias)

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.num_heads))

        self.norm = GraniteMoeHybridRMSNormGated(self.intermediate_size, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mamba_proj_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # 2. Convolution sequence transformation using AD op
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_causal_conv1d(
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                "zeros",
            )
        )

        hidden_states_ssm, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # 3. SSM transformation using AD op
        A = -torch.exp(self.A_log.float())
        y = torch.ops.auto_deploy.torch_ssm(
            hidden_states=hidden_states_ssm.view(batch_size, seq_len, -1, self.head_dim),
            A=A,
            B=B.view(batch_size, seq_len, -1, self.ssm_state_size),
            C=C.view(batch_size, seq_len, -1, self.ssm_state_size),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            time_step_limit=self.time_step_limit,
            chunk_size=self.chunk_size,
        )
        y = y.view(batch_size, seq_len, -1)

        # 4. Gated normalization and output projection
        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states


# ---------------------------------------------------------------------------
# MoE (Mixture of Experts)
# ---------------------------------------------------------------------------


class GraniteMoeHybridExpert(nn.Module):
    """Single MoE expert with gate, up, and down projections."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


class GraniteMoeHybridTopKRouter(nn.Module):
    """Top-K router: computes logits, selects top-k, applies softmax on selected."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        # Attribute named 'layer' to match HF checkpoint key: router.layer.weight
        self.layer = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.layer(hidden_states).float()
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        routing_weights = torch.softmax(top_k_logits, dim=1).to(hidden_states.dtype)
        return routing_weights, top_k_indices


class GraniteMoeHybridMoEBlock(nn.Module):
    """Sparse MoE block using torch_moe AD custom op.

    Uses per-expert nn.ModuleList for checkpoint compatibility with a
    state_dict pre-hook that converts from the HF fused weight format.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts

        self.router = GraniteMoeHybridTopKRouter(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        self.experts = nn.ModuleList(
            [
                GraniteMoeHybridExpert(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_local_experts)
            ]
        )

        # Convert fused HF checkpoint format to per-expert format
        self._register_load_state_dict_pre_hook(self._unfuse_expert_weights)

    @staticmethod
    def _unfuse_expert_weights(state_dict, prefix, *args):
        """Convert fused expert weights from HF format to per-expert ModuleList.

        HF format:
            input_linear.weight: [E, 2*intermediate_size, hidden_size]  (gate+up stacked)
            output_linear.weight: [E, hidden_size, intermediate_size]   (down)

        Target format:
            experts.{i}.gate_proj.weight: [intermediate_size, hidden_size]
            experts.{i}.up_proj.weight: [intermediate_size, hidden_size]
            experts.{i}.down_proj.weight: [hidden_size, intermediate_size]
        """
        input_key = prefix + "input_linear.weight"
        output_key = prefix + "output_linear.weight"

        if input_key in state_dict:
            fused = state_dict.pop(input_key)
            num_experts = fused.shape[0]
            intermediate_size = fused.shape[1] // 2
            gate_weights = fused[:, :intermediate_size, :]
            up_weights = fused[:, intermediate_size:, :]
            for i in range(num_experts):
                state_dict[f"{prefix}experts.{i}.gate_proj.weight"] = gate_weights[i]
                state_dict[f"{prefix}experts.{i}.up_proj.weight"] = up_weights[i]

        if output_key in state_dict:
            fused = state_dict.pop(output_key)
            num_experts = fused.shape[0]
            for i in range(num_experts):
                state_dict[f"{prefix}experts.{i}.down_proj.weight"] = fused[i]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router
        routing_weights, selected_experts = self.router(hidden_states_flat)

        # Extract weight tensors from expert modules for torch_moe
        w1_weights = [self.experts[i].gate_proj.weight for i in range(self.num_experts)]
        w2_weights = [self.experts[i].down_proj.weight for i in range(self.num_experts)]
        w3_weights = [self.experts[i].up_proj.weight for i in range(self.num_experts)]

        expert_output = torch.ops.auto_deploy.torch_moe(
            hidden_states_flat,
            selected_experts,
            routing_weights,
            w1_weights,
            w2_weights,
            w3_weights,
            is_gated_mlp=True,
        )

        return expert_output.view(batch_size, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Decoder layer (supports attention/mamba + optional MoE)
# ---------------------------------------------------------------------------


class GraniteMoeHybridDecoderLayer(nn.Module):
    """Decoder layer supporting either attention or mamba as the token mixer,
    with optional MoE alongside the shared MLP, and muP residual scaling."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        layer_types = getattr(config, "layer_types", None) or getattr(
            config, "layers_block_type", None
        )
        self.layer_type = layer_types[layer_idx] if layer_types else "attention"

        # Token mixer: either attention or mamba (not both)
        if self.layer_type == "mamba":
            self.mamba = GraniteMoeHybridMambaLayer(config, layer_idx)
        else:
            self.self_attn = GraniteMoeHybridAttention(config, layer_idx)

        # MoE (optional, when num_local_experts > 0)
        self.has_experts = getattr(config, "num_local_experts", 0) > 0
        if self.has_experts:
            self.block_sparse_moe = GraniteMoeHybridMoEBlock(config)

        # Shared MLP (always present)
        self.shared_mlp = GraniteMoeHybridMLP(config)

        # Layer norms and residual scaling
        self.input_layernorm = GraniteMoeHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GraniteMoeHybridRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Token mixer (attention or mamba)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "mamba":
            hidden_states = self.mamba(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings)

        hidden_states = residual + hidden_states * self.residual_multiplier

        # Feed-forward (MoE + shared MLP, or shared MLP only)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.has_experts:
            moe_output = self.block_sparse_moe(hidden_states)
            hidden_states = moe_output + self.shared_mlp(hidden_states)
        else:
            hidden_states = self.shared_mlp(hidden_states)

        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class GraniteMoeHybridPreTrainedModel(PreTrainedModel):
    config_class = GraniteMoeHybridConfig
    base_model_prefix = "model"
    _no_split_modules = ["GraniteMoeHybridDecoderLayer"]


class GraniteMoeHybridModel(GraniteMoeHybridPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GraniteMoeHybridDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GraniteMoeHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_multiplier = config.embedding_multiplier

        # Only create RoPE when position_embedding_type is "rope"
        position_embedding_type = getattr(config, "position_embedding_type", None)
        if position_embedding_type == "rope":
            self.rotary_emb = GraniteMoeHybridRotaryEmbedding(config)
        else:
            self.rotary_emb = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> GraniteMoeHybridModelOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # muP: scale embeddings
        hidden_states = inputs_embeds * self.embedding_multiplier

        # Get RoPE table if applicable (shared across all attention layers)
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return GraniteMoeHybridModelOutput(last_hidden_state=hidden_states)


class GraniteMoeHybridForCausalLM(GraniteMoeHybridPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GraniteMoeHybridModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_scaling = config.logits_scaling

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> GraniteMoeHybridCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )
        logits = self.lm_head(outputs.last_hidden_state)

        # muP: scale down logits
        logits = logits / self.logits_scaling

        return GraniteMoeHybridCausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls(
    "GraniteMoeHybridConfig", GraniteMoeHybridForCausalLM
)
