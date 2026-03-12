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

"""Slimmed down PyTorch Cohere/Cohere2 model implementation for auto_deploy export.

Source:
https://huggingface.co/CohereForAI/aya-expanse-8b (Cohere v1)
https://huggingface.co/CohereLabs/c4ai-command-a-03-2025 (Cohere2)

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

This file covers both the `cohere` and `cohere2` model families:
- Cohere (v1): CohereForCausalLM — aya-expanse-8b, aya-expanse-32b
- Cohere2: Cohere2ForCausalLM — c4ai-command-a-03-2025 and variants

Key architectural differences from Llama:
- LayerNorm (not RMSNorm) — subtracts mean, has learned weight, no bias
- Interleaved RoPE — uses torch_rope_with_qk_interleaving canonical op
- Parallel attention + MLP — single input_layernorm, both branches read from
  the same normed hidden states, outputs added to residual
- logit_scale applied to output logits
- tie_word_embeddings = True

Cohere2 additionally features:
- Sliding window attention pattern (every Nth layer is full attention, rest sliding)
- RoPE only applied to sliding window attention layers (full attention layers are
  position-agnostic)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.cohere.configuration_cohere import CohereConfig
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class CohereLayerNorm(nn.Module):
    """LayerNorm for Cohere models (subtracts mean, unlike RMSNorm).

    This is a true LayerNorm with learned weight and no bias.
    No canonical AD op exists for full LayerNorm, so plain PyTorch is used.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)


class CohereRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Cohere models.

    Builds cos/sin cache in NeoX format (cat style) for use with
    torch_rope_with_qk_interleaving canonical op. The HF Cohere model
    uses interleaved RoPE (repeat_interleave), but the canonical op handles
    the de-interleaving of Q/K internally.

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
        # NeoX format: cat([freqs, freqs]) — the qk_interleaving op expects this
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return full cached cos/sin (not sliced) for export compatibility.
        # Position slicing happens downstream in the attention layer.
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class CohereMLP(nn.Module):
    """MLP layer for Cohere models (SwiGLU activation)."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CohereAttention(nn.Module):
    """Grouped Query Attention for Cohere/Cohere2 models.

    Handles both model variants:
    - Cohere v1: always applies RoPE, optional QK norm
    - Cohere2: conditional RoPE (only on sliding window layers), sliding window

    Uses torch_rope_with_qk_interleaving for interleaved RoPE and
    torch_attention for GQA-native attention.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
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

        # Cohere v1: optional QK normalization
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = CohereLayerNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = CohereLayerNorm(self.head_dim, eps=config.layer_norm_eps)

        # Cohere2: per-layer sliding window and conditional RoPE
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx is not None:
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            self.use_rope = is_sliding
            self.sliding_window = config.sliding_window if is_sliding else None
        else:
            # Cohere v1: always use RoPE, no sliding window
            self.use_rope = True
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V and reshape to [B, S, N, head_dim] (BSND layout)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Optional QK normalization (Cohere v1 feature)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply interleaved RoPE (Cohere2: only on sliding window layers)
        if self.use_rope:
            cos, sin = position_embeddings  # Full cached tables
            cos = cos[position_ids]  # Slice by position_ids here (once per layer)
            sin = sin[position_ids]
            q, k = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
                q,
                k,
                cos,
                sin,
                2,  # unsqueeze_dim=2 for BSND layout
            )

        # Attention using canonical op with GQA support (BSND layout)
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
            None,  # logit_cap
            "bsnd",
        )

        # Reshape and project output
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class CohereDecoderLayer(nn.Module):
    """Transformer decoder layer for Cohere/Cohere2 models.

    Uses parallel attention + MLP pattern: both branches read from the same
    normed hidden states and their outputs are added to the residual.
    This means there is only one LayerNorm per layer (input_layernorm),
    no post_attention_layernorm.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = CohereAttention(config, layer_idx=layer_idx)
        self.mlp = CohereMLP(config)
        self.input_layernorm = CohereLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel attention + MLP: both take the normed input
        hidden_states_attention = self.self_attn(hidden_states, position_ids, position_embeddings)
        hidden_states_mlp = self.mlp(hidden_states)

        # Add both outputs to residual
        hidden_states = residual + hidden_states_attention + hidden_states_mlp

        return hidden_states


@dataclass
class CohereModelOutput(ModelOutput):
    """Output for CohereModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class CohereCausalLMOutput(ModelOutput):
    """Output for CohereForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class CoherePreTrainedModel(PreTrainedModel):
    """Base class for Cohere models."""

    config_class = CohereConfig
    base_model_prefix = "model"
    _no_split_modules = ["CohereDecoderLayer"]
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


class CohereModel(CoherePreTrainedModel):
    """Cohere/Cohere2 transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CohereDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = CohereLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Shared rotary embedding at model level
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = CohereRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

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
    ) -> CohereModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute position embeddings once (full cached cos/sin)
        position_embeddings = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return CohereModelOutput(last_hidden_state=hidden_states)


class CohereForCausalLM(CoherePreTrainedModel, GenerationMixin):
    """Cohere model with language modeling head.

    Covers both Cohere v1 (CohereForCausalLM) and Cohere2 (Cohere2ForCausalLM)
    architectures. The config object determines per-layer behavior.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = CohereModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logit_scale = config.logit_scale

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
    ) -> CohereCausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()
        logits = logits * self.logit_scale

        return CohereCausalLMOutput(logits=logits)


# Register for both Cohere v1 and Cohere2 config classes
AutoModelForCausalLMFactory.register_custom_model_cls("CohereConfig", CohereForCausalLM)
AutoModelForCausalLMFactory.register_custom_model_cls("Cohere2Config", CohereForCausalLM)
