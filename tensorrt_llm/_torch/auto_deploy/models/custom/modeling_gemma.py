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

"""Slimmed down PyTorch Gemma model implementation for auto_deploy export.

Source:
https://huggingface.co/google/gemma-1.1-7b-it

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

The Gemma family (gemma, codegemma) shares a single architecture with MHA,
SwiGLU MLP (gelu_pytorch_tanh), RMSNorm with (1+weight) scaling, RoPE, and
embedding normalization by sqrt(hidden_size).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


def _gemma_norm_weight_load_hook(state_dict, prefix, *args, **kwargs):
    """Pre-hook to absorb Gemma's (1+weight) RMSNorm convention into the weight.

    HF Gemma stores norm weights as a bias around zero, applying ``(1 + weight)``
    at forward time. We add 1.0 at load time so that the forward can use the
    standard ``torch_rmsnorm(x, weight, eps)`` without the extra add node.
    """
    for key in list(state_dict.keys()):
        if key.startswith(prefix) and (
            key.endswith("layernorm.weight") or key.endswith("norm.weight")
        ):
            state_dict[key] = state_dict[key] + 1.0


class GemmaADRMSNorm(nn.Module):
    """RMS Normalization for Gemma.

    Gemma HF stores norm weights as a bias around zero and applies ``(1 + weight)``.
    To avoid an extra ``aten.add`` node in the export graph (which breaks the
    fused allreduce+residual+rmsnorm transform), we absorb the +1.0 offset into
    the weight at load time via a pre-hook registered on the top-level model.
    At init, weight is set to ones so that the effective scale starts at 1.0.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(hidden_states, self.weight, self.eps)


class GemmaADRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Gemma.

    Supports all rope types via transformers ROPE_INIT_FUNCTIONS. Precomputes
    and caches cos/sin values. Slices by position_ids once and returns
    pre-sliced cos/sin to all layers.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        # Determine rope type from config
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )
        else:
            rope_type = "default"

        # Use HF's ROPE_INIT_FUNCTIONS to compute inv_freq with proper scaling
        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device=None)

        # Build cos/sin cache with AD-specific naming
        max_pos = config.max_position_embeddings
        t = torch.arange(max_pos, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos() * self.attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * self.attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos[position_ids], sin[position_ids]


class GemmaADMLP(nn.Module):
    """MLP layer for Gemma (gelu_pytorch_tanh activation)."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GemmaADAttention(nn.Module):
    """Multi-Head Attention for Gemma.

    Uses AD canonical ops for attention and RoPE. Gemma typically uses MHA
    (num_kv_heads == num_heads), but the implementation handles GQA natively
    via torch_attention.
    """

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
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

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V and reshape to [B, S, N, head_dim] (BSND layout)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

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

        # Attention using custom op (BSND layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,  # [B, S, N, head_dim]
            k,  # [B, S, N_kv, head_dim]
            v,  # [B, S, N_kv, head_dim]
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,  # scale
            None,  # sinks
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )

        # Reshape [B, S, N, head_dim] -> [B, S, N * head_dim] and project
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class GemmaADDecoderLayer(nn.Module):
    """Transformer decoder layer for Gemma."""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaADAttention(config, layer_idx=layer_idx)
        self.mlp = GemmaADMLP(config)
        self.input_layernorm = GemmaADRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaADRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class GemmaADOutput(ModelOutput):
    """Output for GemmaADModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GemmaADCausalLMOutput(ModelOutput):
    """Output for GemmaADForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class GemmaADPreTrainedModel(PreTrainedModel):
    """Base class for Gemma AD models."""

    config_class = GemmaConfig
    base_model_prefix = "model"
    _no_split_modules = ["GemmaADDecoderLayer"]
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


class GemmaADModel(GemmaADPreTrainedModel):
    """Gemma transformer decoder model."""

    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaADDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaADRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = GemmaADRotaryEmbedding(config)

        # Gemma-specific: embedding normalization factor
        self.normalizer = config.hidden_size**0.5

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
    ) -> GemmaADOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Gemma-specific: scale embeddings by sqrt(hidden_size)
        hidden_states = inputs_embeds * self.normalizer

        # Compute position embeddings once (sliced by position_ids in RoPE)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return GemmaADOutput(last_hidden_state=hidden_states)


class GemmaADForCausalLM(GemmaADPreTrainedModel, GenerationMixin):
    """Gemma model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = GemmaADModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Absorb Gemma's (1+weight) RMSNorm convention into the weight at load time
        self._register_load_state_dict_pre_hook(_gemma_norm_weight_load_hook)

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
    ) -> GemmaADCausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return GemmaADCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("GemmaConfig", GemmaADForCausalLM)
