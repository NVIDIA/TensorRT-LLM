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

"""Prefill-only DeciLM (Nemotron-NAS) model for auto_deploy export.

Source: https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1

This is a heterogeneous Llama-like architecture where each layer can have:
- Attention (GQA) + FFN with per-layer varying sizes
- FFN-only (attention skipped via no_op)
- Per-layer FFN width controlled by ffn_mult

Simplified for prefill-only inference (no KV caching).
Uses auto_deploy canonical IR ops for export compatibility.

Config is loaded from the HF checkpoint via trust_remote_code=True (not bundled here).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

# =============================================================================
# Helpers
# =============================================================================


def _find_multiple(n: int, k: int) -> int:
    """Round up n to nearest multiple of k."""
    if n % k == 0:
        return n
    return n + k - (n % k)


def _ffn_mult_to_intermediate_size(ffn_mult: float, hidden_size: int) -> int:
    """Convert ffn_mult to intermediate_size (DeciLM-specific formula)."""
    intermediate_size = int(2 * ffn_mult * hidden_size / 3)
    return _find_multiple(intermediate_size, 256)


# =============================================================================
# RMSNorm (using canonical AD IR op)
# =============================================================================


class DeciLMRMSNorm(nn.Module):
    """RMS Normalization using the canonical AutoDeploy torch_rmsnorm op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


# =============================================================================
# Rotary Embedding (Llama3-style)
# =============================================================================


class DeciLMRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with Llama3-style frequency scaling.

    Precomputes cos/sin tables and caches them with _ad_ prefix for
    AutoDeploy's lift_to_meta compatibility. Returns full tables (not sliced).
    """

    def __init__(self, config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        self.dim = head_dim

        inv_freq = self._compute_inv_freq(config, head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(config.max_position_embeddings)

    def _compute_inv_freq(self, config, dim: int) -> torch.Tensor:
        """Compute inverse frequencies with optional Llama3 scaling."""
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        if config.rope_scaling is None:
            return inv_freq

        rope_type = config.rope_scaling.get("rope_type", "default")
        if rope_type == "llama3":
            factor = config.rope_scaling["factor"]
            low_freq_factor = config.rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = config.rope_scaling.get("high_freq_factor", 4.0)
            old_context_len = config.rope_scaling.get("original_max_position_embeddings", 8192)

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq

            # Low freq: scale down by factor
            inv_freq_scaled = inv_freq / factor

            # Smooth interpolation in transition band
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)
            smoothed = (1 - smooth_factor) * inv_freq_scaled + smooth_factor * inv_freq

            # Apply: high freq unchanged, low freq scaled, medium smoothed
            inv_freq = torch.where(wavelen < high_freq_wavelen, inv_freq, smoothed)
            inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq_scaled, inv_freq)

        return inv_freq

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
        """Return cos/sin indexed by position_ids: [B, S, head_dim]."""
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos[position_ids], sin[position_ids]


# =============================================================================
# MLP
# =============================================================================


class DeciLMMLP(nn.Module):
    """SwiGLU MLP with per-layer intermediate size."""

    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Attention
# =============================================================================


class DeciLMAttention(nn.Module):
    """GQA attention using canonical AD IR ops (bsnd layout)."""

    def __init__(self, config, n_heads_in_group: int, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.num_heads // n_heads_in_group

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to bsnd layout: (B, S, N, D)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply RoPE using canonical AD IR op (cos/sin already indexed by position_ids)
        cos, sin = position_embeddings  # [B, S, head_dim]
        query_states, key_states = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            query_states,
            key_states,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for bsnd layout
        )

        # Attention via canonical AD IR op (bsnd layout, handles GQA internally)
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            dropout_p=0.0,
            layout="bsnd",
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


# =============================================================================
# Decoder Layer
# =============================================================================


class DeciLMDecoderLayer(nn.Module):
    """Heterogeneous decoder layer: attention may be skipped (no_op).

    When attention.no_op is True, the layer only has FFN (no self_attn or
    input_layernorm). This matches the checkpoint weight structure exactly.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        block_config = config.block_configs[layer_idx]
        self.has_attention = not block_config.attention.no_op
        self.has_ffn = not block_config.ffn.no_op

        if self.has_attention:
            self.input_layernorm = DeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn = DeciLMAttention(
                config,
                n_heads_in_group=block_config.attention.n_heads_in_group,
                layer_idx=layer_idx,
            )

        if self.has_ffn:
            intermediate_size = _ffn_mult_to_intermediate_size(
                block_config.ffn.ffn_mult, config.hidden_size
            )
            self.post_attention_layernorm = DeciLMRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = DeciLMMLP(config, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Attention (skip if no_op)
        if self.has_attention:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states, position_embeddings)
            hidden_states = residual + hidden_states

        # FFN (skip if no_op)
        if self.has_ffn:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model Outputs
# =============================================================================


@dataclass
class DeciLMModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class DeciLMCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# =============================================================================
# Full Model
# =============================================================================


class DeciLMPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["DeciLMDecoderLayer"]
    _supports_flash_attn_2 = True
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


class DeciLMModel(DeciLMPreTrainedModel):
    """DeciLM transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeciLMDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = DeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding
        self.rotary_emb = DeciLMRotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeciLMModelOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute position embeddings once (shared across layers, indexed by position_ids)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return DeciLMModelOutput(last_hidden_state=hidden_states)


class DeciLMForCausalLM(DeciLMPreTrainedModel, GenerationMixin):
    """DeciLM model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = DeciLMModel(config)
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
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeciLMCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return DeciLMCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("DeciLMConfig", DeciLMForCausalLM)
