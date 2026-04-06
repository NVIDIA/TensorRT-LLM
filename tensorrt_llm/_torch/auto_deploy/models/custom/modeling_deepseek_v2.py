# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Slimmed down PyTorch DeepSeekV2 model implementation for auto_deploy export.

Source:
https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct

This covers the DeepSeek-V2 model family (model_type="deepseek_v2"), including:
- DeepSeek-Coder-V2-Instruct (236B MoE, group_limited_greedy routing)
- DeepSeek-Coder-V2-Lite-Instruct (16B MoE, greedy routing, no Q LoRA)
- DeepSeek-V2.5 (236B MoE, group_limited_greedy routing)

Key differences from DeepSeek-V3:
- Softmax-based MoE routing (group_limited_greedy or greedy) instead of noaux_tc (sigmoid)
- No e_score_correction_bias in gate
- routed_scaling_factor applied after softmax top-k selection

Simplified for prefill-only inference with auto_deploy canonical ops.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.custom import mla_rope_utils
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType


class DeepSeekV2RMSNorm(nn.Module):
    """RMS Normalization for DeepSeekV2."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        ).to(hidden_states.dtype)


class DeepSeekV2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for DeepSeekV2.

    Simplified version that precomputes and caches cos/sin values.
    Returns full cached values (not sliced by seq_len) to enable export.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached[position_ids].to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached[position_ids].to(dtype=x.dtype, device=x.device)
        return cos, sin


class DeepSeekV2YarnRotaryEmbedding(DeepSeekV2RotaryEmbedding):
    """YaRN-extended rotary embedding for DeepSeekV2."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            self._yarn_get_mscale(self.scaling_factor, self.mscale)
            / self._yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", (emb.cos() * _mscale), persistent=False)
        self.register_buffer("_ad_sin_cached", (emb.sin() * _mscale), persistent=False)

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float, dim: int, base: float = 10000, max_position_embeddings: int = 2048
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self, low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int
    ) -> Tuple[int, int]:
        low = math.floor(
            self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)


class DeepSeekV2MLP(nn.Module):
    """MLP layer for DeepSeekV2 (SwiGLU activation)."""

    def __init__(
        self, config, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekV2MoEGate(nn.Module):
    """MoE Gating for DeepSeekV2 with softmax-based top-k selection.

    Supports two routing methods:
    - "greedy": simple top-k (used by DeepSeek-Coder-V2-Lite)
    - "group_limited_greedy": group-based top-k (used by DeepSeek-Coder-V2, V2.5)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.norm_topk_prob = config.norm_topk_prob
        self.n_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (selected_experts, routing_weights)."""
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits and softmax scores
        logits = F.linear(hidden_states_flat.float(), self.weight.float())
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            n_tokens = bsz * seq_len
            # Group scores: take max within each group
            group_scores = (
                scores.view(n_tokens, self.n_group, -1).max(dim=-1).values
            )  # [n_tokens, n_group]
            # Select top groups
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
                1
            ]  # [n_tokens, topk_group]
            # Build group mask
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)  # [n_tokens, n_group]
            # Expand group mask to expert mask
            experts_per_group = self.n_routed_experts // self.n_group
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(n_tokens, self.n_group, experts_per_group)
                .reshape(n_tokens, -1)
            )
            # Mask out non-selected groups and take top-k
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        else:
            raise ValueError(f"Unknown topk_method: {self.topk_method}")

        # Normalize if configured
        if self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # Apply scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class DeepSeekV2MoE(nn.Module):
    """Mixture of Experts layer for DeepSeekV2."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # Routed experts
        self.experts = nn.ModuleList(
            [
                DeepSeekV2MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )

        # Gate
        self.gate = DeepSeekV2MoEGate(config)

        # Shared experts (if configured)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepSeekV2MLP(config, intermediate_size=intermediate_size)
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        selected_experts, routing_weights = self.gate(hidden_states)

        # Use torch_moe custom op for routed experts
        final_hidden_states = torch.ops.auto_deploy.torch_moe(
            hidden_states.view(-1, hidden_states.shape[-1]),
            selected_experts,
            routing_weights,
            w1_weight=[expert.gate_proj.weight for expert in self.experts],
            w2_weight=[expert.down_proj.weight for expert in self.experts],
            w3_weight=[expert.up_proj.weight for expert in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
        )

        final_hidden_states = final_hidden_states.view(*orig_shape)

        # Add shared experts output if present
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)

        return final_hidden_states.to(hidden_states.dtype)


class DeepSeekV2Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for DeepSeekV2.

    Uses compressed KV representation with latent projections.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Q projection (with optional LoRA)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepSeekV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        # KV projection with MQA
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepSeekV2RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Initialize rotary embedding
        self._init_rope()

        # Softmax scale
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = DeepSeekV2YarnRotaryEmbedding._yarn_get_mscale(
                    scaling_factor, mscale_all_dim
                )
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepSeekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get(
                "type", self.config.rope_scaling.get("rope_type", "default")
            )
            scaling_factor = self.config.rope_scaling["factor"]

            if scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepSeekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                self.rotary_emb = DeepSeekV2RotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Q projection
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        # Shape: [B, S, N, q_head_dim] (BSND layout)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection - keep compressed form
        kv_a_output = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            kv_a_output, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Apply layernorm to compressed_kv
        compressed_kv = self.kv_a_layernorm(compressed_kv)

        # k_pe: [B, S, 1, qk_rope_head_dim] (BSND layout, shared across heads)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Apply RoPE using custom op (weights pre-permuted to NeoX format at load time)
        q_pe_rotated, kpe = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q_pe,
            k_pe,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # Call MLA with compressed KV
        attn_output = torch.ops.auto_deploy.torch_mla(
            q_nope,  # [B, S, N, qk_nope_head_dim]
            q_pe_rotated,  # [B, S, N, qk_rope_head_dim]
            compressed_kv,  # [B, S, kv_lora_rank]
            kpe,  # [B, S, 1, qk_rope_head_dim]
            self.kv_b_proj.weight,  # [N*(qk_nope+v), kv_lora_rank]
            True,  # is_causal
            self.softmax_scale,
            "bsnd",  # layout
        )

        # Output: [B, S, N, v_head_dim] -> [B, S, N * v_head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepSeekV2DecoderLayer(nn.Module):
    """Transformer decoder layer for DeepSeekV2."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = DeepSeekV2Attention(config, layer_idx=layer_idx)

        # MLP or MoE
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        if use_moe:
            self.mlp = DeepSeekV2MoE(config)
        else:
            self.mlp = DeepSeekV2MLP(config)

        # Layer norms
        self.input_layernorm = DeepSeekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        # MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class DeepSeekV2Output(ModelOutput):
    """Output for DeepSeekV2Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class DeepSeekV2CausalLMOutput(ModelOutput):
    """Output for DeepSeekV2ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class DeepSeekV2PreTrainedModel(PreTrainedModel):
    """Base class for DeepSeekV2 models."""

    base_model_prefix = "model"
    _no_split_modules = ["DeepSeekV2DecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, DeepSeekV2MoEGate):
            module.weight.data.normal_(mean=0.0, std=std)


class DeepSeekV2Model(DeepSeekV2PreTrainedModel):
    """DeepSeekV2 transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                DeepSeekV2DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepSeekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeepSeekV2Output:
        assert position_ids is not None, "position_ids is required for DeepSeekV2"

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids)

        hidden_states = self.norm(hidden_states)

        return DeepSeekV2Output(last_hidden_state=hidden_states)


def _q_proj_deinterleave_hook(
    state_dict,
    prefix,
    *args,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_heads: int,
    num_layers: int,
):
    """Pre-load hook for q_proj (used when q_lora_rank is None).

    q_proj.weight has shape [num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden_size].
    The RoPE portion within each head needs de-interleaving from interleaved to NeoX format.
    """
    d = qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    qk_head_dim = qk_nope_head_dim + d

    for layer_idx in range(num_layers):
        q_key = f"{prefix}model.layers.{layer_idx}.self_attn.q_proj.weight"
        if q_key in state_dict:
            w = state_dict[q_key]
            w = w.view(num_heads, qk_head_dim, -1)
            w_nope = w[:, :qk_nope_head_dim, :]
            w_rope = w[:, qk_nope_head_dim:, :]
            w_rope = w_rope[:, perm, :]
            w = torch.cat([w_nope, w_rope], dim=1)
            state_dict[q_key] = w.view(-1, w.shape[-1])


class DeepSeekV2ForCausalLM(DeepSeekV2PreTrainedModel, GenerationMixin):
    """DeepSeekV2 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepSeekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Pre-permute RoPE weight rows from interleaved to NeoX format at load time
        # so the forward can use torch_rope_with_explicit_cos_sin (-> flashinfer_rope).
        # The shared hook handles q_b_proj and kv_a_proj_with_mqa.
        self._register_load_state_dict_pre_hook(
            partial(
                mla_rope_utils._rope_deinterleave_load_hook,
                qk_rope_head_dim=config.qk_rope_head_dim,
                qk_nope_head_dim=config.qk_nope_head_dim,
                num_heads=config.num_attention_heads,
                kv_lora_rank=config.kv_lora_rank,
                num_layers=config.num_hidden_layers,
            )
        )
        # When q_lora_rank is None, the model uses q_proj instead of q_b_proj.
        # q_proj also has interleaved RoPE dims that need de-interleaving.
        if config.q_lora_rank is None:
            self._register_load_state_dict_pre_hook(
                partial(
                    _q_proj_deinterleave_hook,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    num_heads=config.num_attention_heads,
                    num_layers=config.num_hidden_layers,
                )
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeepSeekV2CausalLMOutput:
        assert position_ids is not None, "position_ids is required for DeepSeekV2"

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return DeepSeekV2CausalLMOutput(logits=logits)


# Import the config from transformers (available since transformers 4.x)

# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("DeepseekV2Config", DeepSeekV2ForCausalLM)
