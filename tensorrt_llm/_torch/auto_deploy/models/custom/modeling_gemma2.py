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

"""Slimmed down PyTorch Gemma 2 model implementation for auto_deploy export.

Source:
https://huggingface.co/google/gemma-2-2b-it

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

Key Gemma 2 architecture features:
* RMSNorm with (1 + weight) scaling (weights initialized to zero)
* 4 layer norms per decoder layer (pre/post attention, pre/post feedforward)
* Attention logit softcapping (tanh-based)
* Final logit softcapping on lm_head output
* Alternating sliding window / full attention layers
* Custom attention scaling via query_pre_attn_scalar (not head_dim)
* Embedding normalization by sqrt(hidden_size)
* GQA with explicit head_dim (can differ from hidden_size / num_heads)
* gelu_pytorch_tanh activation
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class Gemma2RMSNorm(nn.Module):
    """RMS Normalization for Gemma 2.

    Gemma 2 uses (1 + weight) scaling instead of standard weight scaling.
    The weight parameter is zero-initialized so that the initial scale is 1.0.
    We use the canonical torch_rmsnorm op with (1 + weight) passed as the weight.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, 1.0 + self.weight, self.eps)


class Gemma2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Gemma 2.

    Precomputes and caches cos/sin values. Returns pre-sliced values
    indexed by position_ids so layers don't need to slice again.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
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


class Gemma2MLP(nn.Module):
    """MLP layer for Gemma 2 (gelu_pytorch_tanh gated)."""

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma2Attention(nn.Module):
    """Grouped Query Attention for Gemma 2.

    Features:
    * Custom scaling via query_pre_attn_scalar (not head_dim)
    * Attention logit softcapping (tanh-based capping)
    * Per-layer sliding window or full attention based on layer_types config
    """

    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.scaling = config.query_pre_attn_scalar ** (-0.5)

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

        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings

        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q,
            k,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,
            None,  # sinks
            self.sliding_window,
            self.attn_logit_softcapping,
            "bsnd",
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Gemma2DecoderLayer(nn.Module):
    """Transformer decoder layer for Gemma 2.

    Uses 4 layer norms: pre/post attention and pre/post feedforward.
    The post norms apply after the sublayer output but before the residual add.
    """

    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Gemma2Attention(config, layer_idx=layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention with pre/post norms
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre/post norms
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Gemma2Output(ModelOutput):
    """Output for Gemma2Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Gemma2CausalLMOutput(ModelOutput):
    """Output for Gemma2ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class Gemma2PreTrainedModel(PreTrainedModel):
    """Base class for Gemma 2 models."""

    config_class = Gemma2Config
    base_model_prefix = "model"
    _no_split_modules = ["Gemma2DecoderLayer"]
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
        elif isinstance(module, Gemma2RMSNorm):
            module.weight.data.zero_()


class Gemma2Model(Gemma2PreTrainedModel):
    """Gemma 2 transformer decoder model."""

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = Gemma2RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Normalizer for embedding: Gemma 2 scales embeddings by sqrt(hidden_size)
        self.normalizer_value = config.hidden_size**0.5

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
    ) -> Gemma2Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Gemma 2 scales embeddings by sqrt(hidden_size)
        # Cast normalizer to input dtype to match HF behavior (e.g., float16 sqrt(3072) = 55.5)
        normalizer = torch.tensor(self.normalizer_value, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds * normalizer

        # Compute position embeddings once (sliced by position_ids in RoPE)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return Gemma2Output(last_hidden_state=hidden_states)


class Gemma2ForCausalLM(Gemma2PreTrainedModel, GenerationMixin):
    """Gemma 2 model with language modeling head and logit softcapping."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Gemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        assert config.final_logit_softcapping is not None, (
            "Gemma 2 requires final_logit_softcapping"
        )
        self.final_logit_softcapping = config.final_logit_softcapping

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
    ) -> Gemma2CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        # Apply final logit softcapping (always present for Gemma 2)
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping

        return Gemma2CausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("Gemma2Config", Gemma2ForCausalLM)
