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

"""Slimmed down PyTorch OLMo-3 model implementation for auto_deploy export.

Source:
https://huggingface.co/allenai/Olmo-3-7B-Instruct

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

OLMo-3 architecture features:
* Post-norm residual pattern (RMSNorm after attention/MLP, not before)
* QK normalization on full projection (not per-head)
* Two separate RoPE embeddings: "default" for sliding_attention, "yarn" for full_attention
* Mixed attention types per layer via config.layer_types (sliding_attention vs full_attention)
* GQA support (7B uses MHA, 32B uses GQA)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class Olmo3RMSNorm(nn.Module):
    """RMS Normalization for OLMo-3 using AutoDeploy torch_rmsnorm reference op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class Olmo3RotaryEmbedding(nn.Module):
    """Standard Rotary Position Embedding for OLMo-3 sliding_attention layers.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
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
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos[position_ids], sin[position_ids]


class Olmo3YarnRotaryEmbedding(nn.Module):
    """YaRN-extended Rotary Position Embedding for OLMo-3 full_attention layers.

    Implements YaRN (Yet Another RoPE eXtensioN) for extended context length.
    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attention_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attention_factor = attention_factor

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        # Standard inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Interpolated inverse frequencies
        inv_freq_interpolated = inv_freq / self.scaling_factor

        # YaRN correction mask
        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2)
        # Blend interpolated and original frequencies
        inv_freq = inv_freq_interpolated * (1 - inv_freq_mask) + inv_freq * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos() * self.attention_factor, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * self.attention_factor, persistent=False)

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float, dim: int, base: float = 10000.0, max_position_embeddings: int = 2048
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self, low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
    ) -> Tuple[int, int]:
        low = math.floor(
            self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos[position_ids], sin[position_ids]


class Olmo3MLP(nn.Module):
    """MLP layer for OLMo-3 (SwiGLU activation)."""

    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Olmo3Attention(nn.Module):
    """Attention for OLMo-3 with QK normalization on full projections.

    OLMo-3 applies RMSNorm to query and key states after projection but before
    reshaping to heads. The norm dimension is num_heads*head_dim for Q and
    num_kv_heads*head_dim for K, which differs from Qwen3-style per-head norm.
    """

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.scaling = self.head_dim ** (-0.5)

        # Q/K/V/O projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # QK normalization on full projection (not per-head)
        self.q_norm = Olmo3RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

        # Attention type and sliding window
        assert config.layer_types is not None
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = (
            config.sliding_window if self.attention_type == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project and apply QK norm on full projection BEFORE reshaping to heads
        q = self.q_norm(self.q_proj(hidden_states))
        k = self.k_norm(self.k_proj(hidden_states))
        v = self.v_proj(hidden_states)

        # Reshape to BSND layout: [B, S, N, head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Get pre-sliced cos/sin from position_embeddings (already indexed by position_ids)
        cos, sin = position_embeddings  # [B, S, head_dim]

        # Apply RoPE using custom op (BSND layout, unsqueeze_dim=2)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q,
            k,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # Attention using custom op with GQA support (BSND layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,  # [B, S, N, head_dim]
            k,  # [B, S, N_kv, head_dim]
            v,  # [B, S, N_kv, head_dim]
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,  # scale
            None,  # sinks
            self.sliding_window,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )

        # Reshape [B, S, N, head_dim] -> [B, S, N * head_dim] and project
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Olmo3DecoderLayer(nn.Module):
    """Transformer decoder layer for OLMo-3.

    Uses POST-norm residual pattern: residual + norm(sublayer(x)).
    This differs from the standard pre-norm pattern used in Llama/Qwen.
    """

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Olmo3Attention(config, layer_idx=layer_idx)
        self.mlp = Olmo3MLP(config)

        # Post-norm: applied AFTER sublayer, not before
        self.post_attention_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention with post-norm
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with post-norm
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Olmo3Output(ModelOutput):
    """Output for Olmo3Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Olmo3CausalLMOutput(ModelOutput):
    """Output for Olmo3ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class Olmo3PreTrainedModel(PreTrainedModel):
    """Base class for OLMo-3 models."""

    config_class = Olmo3Config
    base_model_prefix = "model"
    _no_split_modules = ["Olmo3DecoderLayer"]
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


class Olmo3Model(Olmo3PreTrainedModel):
    """OLMo-3 transformer decoder model."""

    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Olmo3DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Two separate RoPE embeddings: "default" for sliding, "yarn" for full attention
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb_sliding = Olmo3RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.rotary_emb_full = self._init_yarn_rope(config, head_dim)

        # Pre-compute layer type list for static dispatch in forward
        assert config.layer_types is not None
        self._layer_is_sliding = [lt == "sliding_attention" for lt in config.layer_types]

        self.post_init()

    def _init_yarn_rope(self, config: Olmo3Config, head_dim: int) -> nn.Module:
        """Initialize YaRN RoPE for full_attention layers."""
        rope_scaling = config.rope_scaling
        if rope_scaling is not None and isinstance(rope_scaling, dict) and "factor" in rope_scaling:
            return Olmo3YarnRotaryEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scaling_factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling.get(
                    "original_max_position_embeddings", 8192
                ),
                beta_fast=rope_scaling.get("beta_fast", 32.0),
                beta_slow=rope_scaling.get("beta_slow", 1.0),
                attention_factor=rope_scaling.get("attention_factor", 1.0),
            )
        # Fallback to default RoPE if no yarn scaling configured
        return Olmo3RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Olmo3Output:
        assert position_ids is not None, "position_ids must be provided for AD export"

        inputs_embeds = self.embed_tokens(input_ids)

        # Cast to compute dtype for FP8 models
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        # Compute both sets of position embeddings once (pre-sliced by position_ids)
        sliding_pos_embeddings = self.rotary_emb_sliding(inputs_embeds, position_ids)
        full_pos_embeddings = self.rotary_emb_full(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            # Static dispatch: select RoPE based on layer type (resolved at trace time)
            if self._layer_is_sliding[idx]:
                pos_emb = sliding_pos_embeddings
            else:
                pos_emb = full_pos_embeddings
            hidden_states = decoder_layer(hidden_states, pos_emb)

        hidden_states = self.norm(hidden_states)

        return Olmo3Output(last_hidden_state=hidden_states)


class Olmo3ForCausalLM(Olmo3PreTrainedModel, GenerationMixin):
    """OLMo-3 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Olmo3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        **kwargs,
    ) -> Olmo3CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return Olmo3CausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("Olmo3Config", Olmo3ForCausalLM)
