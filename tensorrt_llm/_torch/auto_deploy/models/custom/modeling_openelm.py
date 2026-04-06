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

"""Prefill-only OpenELM model for auto_deploy export.

Source: https://huggingface.co/apple/OpenELM-270M-Instruct

OpenELM is a heterogeneous transformer with per-layer varying:
- Number of query/KV heads (GQA)
- FFN intermediate size (via ffn_multipliers)
- Fused QKV projection
- Q/K normalization before RoPE
- Shared input/output embeddings (no separate lm_head)

Config is loaded from the HF checkpoint via trust_remote_code=True.
Uses AutoDeploy canonical IR ops for export compatibility.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

# =============================================================================
# Helpers
# =============================================================================


def _make_divisible(v, divisor=8, min_value=None):
    """Ensure value is divisible by divisor (from HF OpenELM config)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# =============================================================================
# RMSNorm (canonical AD op)
# =============================================================================


class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


# =============================================================================
# Rotary Embedding
# =============================================================================


class OpenELMRotaryEmbedding(nn.Module):
    """Shared rotary embedding with _ad_ prefixed buffers.

    Returns cos/sin indexed by position_ids: [B, S, head_dim].
    """

    def __init__(self, head_dim: int, max_seq_length: int, freq_constant: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            freq_constant ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_length)

    def _set_cos_sin_cache(self, seq_len: int):
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
# FFN
# =============================================================================


class OpenELMFeedForwardNetwork(nn.Module):
    """GLU-style FFN with fused gate+up projection (proj_1) and down projection (proj_2)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            _make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor)
        )

        if config.ffn_with_glu:
            self.proj_1 = nn.Linear(config.model_dim, 2 * intermediate_dim, bias=False)
            self.proj_2 = nn.Linear(intermediate_dim, config.model_dim, bias=False)
            self.ffn_with_glu = True
        else:
            self.proj_1 = nn.Linear(config.model_dim, intermediate_dim, bias=False)
            self.proj_2 = nn.Linear(intermediate_dim, config.model_dim, bias=False)
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, dim=-1)
            return self.proj_2(self.act(y_1) * y_2)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


# =============================================================================
# Attention
# =============================================================================


class OpenELMAttention(nn.Module):
    """GQA attention with fused QKV proj, Q/K norms, canonical AD ops."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_query_heads[layer_idx]
        self.num_k_heads = config.num_kv_heads[layer_idx]
        self.num_v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            config.model_dim,
            (self.num_q_heads + self.num_k_heads + self.num_v_heads) * self.head_dim,
            bias=False,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(config.head_dim)
            self.k_norm = OpenELMRMSNorm(config.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(self.num_q_heads * self.head_dim, config.model_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        # Fused QKV projection → split
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(
            bsz, seq_len, self.num_q_heads + self.num_k_heads + self.num_v_heads, self.head_dim
        )
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=2
        )

        # Q/K normalization (per-head, operates on last dim = head_dim)
        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        # RoPE via canonical AD op (cos/sin already indexed by position_ids)
        cos, sin = position_embeddings  # [B, S, head_dim]
        queries, keys = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            queries,
            keys,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for bsnd layout
        )

        # Attention via canonical AD op (bsnd layout, handles GQA natively)
        attn_output = torch.ops.auto_deploy.torch_attention(
            queries, keys, values, is_causal=True, dropout_p=0.0, layout="bsnd"
        )

        attn_output = attn_output.reshape(bsz, seq_len, self.num_q_heads * self.head_dim)
        return self.out_proj(attn_output)


# =============================================================================
# Decoder Layer
# =============================================================================


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attn = OpenELMAttention(config, layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config, layer_idx)
        self.attn_norm = OpenELMRMSNorm(config.model_dim)
        self.ffn_norm = OpenELMRMSNorm(config.model_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model Outputs
# =============================================================================


@dataclass
class OpenELMModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class OpenELMCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# =============================================================================
# Full Model
# =============================================================================


class OpenELMPreTrainedModel(PreTrainedModel):
    base_model_prefix = "transformer"
    _no_split_modules = ["OpenELMDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class OpenELMModel(OpenELMPreTrainedModel):
    """OpenELM transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList(
            [
                OpenELMDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_transformer_layers)
            ]
        )
        self.norm = OpenELMRMSNorm(config.model_dim)

        # Shared rotary embedding
        self.rotary_emb = OpenELMRotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, value):
        self.token_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> OpenELMModelOutput:
        assert position_ids is not None, "position_ids is required"

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return OpenELMModelOutput(last_hidden_state=hidden_states)


class OpenELMForCausalLM(OpenELMPreTrainedModel, GenerationMixin):
    """OpenELM model with language modeling head (shared embeddings)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size

        # OpenELM shares input/output embeddings; no separate lm_head
        # But we still need the attribute for weight tying
        if not config.share_input_output_layers:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> OpenELMCausalLMOutput:
        assert position_ids is not None, "position_ids is required"

        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.transformer.token_embeddings.weight)
        logits = logits.float()

        return OpenELMCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("OpenELMConfig", OpenELMForCausalLM)
