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

"""Slimmed down PyTorch SmolLM3 model implementation for auto_deploy export.

Source:
https://huggingface.co/HuggingFaceTB/SmolLM3-3B

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
* No repeat_kv — AD attention ops handle GQA natively

SmolLM3 is a Llama-like model with GQA, SwiGLU MLP, and RMSNorm.
Its distinguishing feature is NoPE (No Position Embedding) layers: certain layers
skip RoPE entirely, controlled by the `no_rope_layers` config field.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class SmolLM3RMSNorm(nn.Module):
    """RMS Normalization for SmolLM3 using AutoDeploy torch_rmsnorm reference op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class SmolLM3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for SmolLM3.

    Supports all rope types via transformers ROPE_INIT_FUNCTIONS.
    Precomputes and caches cos/sin values.
    Slices by position_ids once and returns pre-sliced cos/sin to all layers.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(self, config: SmolLM3Config):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )
        else:
            rope_type = "default"

        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device=None)

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


class SmolLM3MLP(nn.Module):
    """MLP layer for SmolLM3 (SwiGLU activation)."""

    def __init__(self, config: SmolLM3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SmolLM3Attention(nn.Module):
    """Grouped Query Attention for SmolLM3 with per-layer RoPE toggling.

    Uses AD canonical ops for attention and RoPE. GQA is handled natively
    by torch_attention — no repeat_kv needed.

    Some layers skip RoPE entirely (NoPE layers), controlled by
    config.no_rope_layers[layer_idx].
    """

    def __init__(self, config: SmolLM3Config, layer_idx: Optional[int] = None):
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

        # Whether this layer uses RoPE (1 = use RoPE, 0 = NoPE)
        self.use_rope = config.no_rope_layers[layer_idx]

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

        # Apply RoPE only for layers that use it
        if self.use_rope:
            cos, sin = position_embeddings  # [B, S, head_dim]
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
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )

        # Reshape [B, S, N, head_dim] -> [B, S, N * head_dim] and project
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class SmolLM3DecoderLayer(nn.Module):
    """Transformer decoder layer for SmolLM3."""

    def __init__(self, config: SmolLM3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SmolLM3Attention(config, layer_idx=layer_idx)
        self.mlp = SmolLM3MLP(config)
        self.input_layernorm = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
class SmolLM3Output(ModelOutput):
    """Output for SmolLM3Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class SmolLM3CausalLMOutput(ModelOutput):
    """Output for SmolLM3ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class SmolLM3PreTrainedModel(PreTrainedModel):
    """Base class for SmolLM3 models."""

    config_class = SmolLM3Config
    base_model_prefix = "model"
    _no_split_modules = ["SmolLM3DecoderLayer"]
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


class SmolLM3Model(SmolLM3PreTrainedModel):
    """SmolLM3 transformer decoder model."""

    def __init__(self, config: SmolLM3Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SmolLM3DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = SmolLM3RotaryEmbedding(config)

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
    ) -> SmolLM3Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cast to compute dtype for FP8 models
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        # Compute position embeddings once (sliced by position_ids in RoPE)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return SmolLM3Output(last_hidden_state=hidden_states)


class SmolLM3ForCausalLM(SmolLM3PreTrainedModel, GenerationMixin):
    """SmolLM3 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = SmolLM3Model(config)
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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SmolLM3CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return SmolLM3CausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("SmolLM3Config", SmolLM3ForCausalLM)
