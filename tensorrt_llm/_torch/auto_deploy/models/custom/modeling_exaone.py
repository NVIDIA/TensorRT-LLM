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

"""Slimmed down PyTorch EXAONE model implementation for auto_deploy export.

Source:
https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct

This implementation differs from the original HuggingFace version in the following ways:
* Bundled config class to work with transformers that lack native ExaoneConfig
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
* No repeat_kv — AD attention ops handle GQA natively

The EXAONE 3.5 family uses GQA with SwiGLU MLP, RMSNorm, and llama3-style RoPE scaling.
Note: EXAONE uses non-standard naming (wte, h, ln_1/ln_2, attn.attention, c_fc_0/c_fc_1/c_proj).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class ExaoneConfig(PretrainedConfig):
    """Configuration class for EXAONE 3.5 model.

    Bundled with the custom model implementation to enable loading when the
    installed transformers does not have native ExaoneConfig (model_type "exaone").
    """

    model_type = "exaone"
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "hidden_act": "activation_function",
        "rms_norm_eps": "layer_norm_epsilon",
    }

    def __init__(
        self,
        vocab_size: int = 102400,
        max_position_embeddings: int = 2048,
        hidden_size: int = 2048,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        activation_function: str = "silu",
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        embed_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if head_dim is not None:
            self.head_dim = head_dim
        if intermediate_size is not None:
            self.intermediate_size = intermediate_size
        else:
            self.intermediate_size = hidden_size * 4
        self.activation_function = activation_function
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register config with AutoConfig so it can be loaded from HF hub
try:
    AutoConfig.register("exaone", ExaoneConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("exaone", ExaoneConfig)
    except ValueError:
        pass


class ExaoneRMSNorm(nn.Module):
    """RMS Normalization using AutoDeploy torch_rmsnorm canonical op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class ExaoneRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for EXAONE.

    Supports all rope types (default, llama3, linear, dynamic, etc.) via
    transformers ROPE_INIT_FUNCTIONS. Precomputes and caches cos/sin values.
    Slices by position_ids once and returns pre-sliced cos/sin to all layers.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(self, config):
        super().__init__()
        # Get rope type from config (handle both rope_scaling and rope_parameters)
        rope_config = getattr(config, "rope_scaling", None)
        if rope_config is None:
            rope_config = getattr(config, "rope_parameters", None)
        if isinstance(rope_config, dict):
            rope_type = rope_config.get("rope_type", rope_config.get("type", "default"))
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


class ExaoneMLP(nn.Module):
    """MLP layer for EXAONE (SwiGLU activation).

    Uses EXAONE naming: c_fc_0 (gate), c_fc_1 (up), c_proj (down).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.c_fc_0 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.c_fc_1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.c_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc_0(x)) * self.c_fc_1(x))


class ExaoneAttention(nn.Module):
    """Grouped Query Attention for EXAONE.

    Uses AD canonical ops for attention and RoPE. GQA is handled natively
    by torch_attention — no repeat_kv needed.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

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
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

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
        attn_output = self.out_proj(attn_output)

        return attn_output


class ExaoneAttentionBlock(nn.Module):
    """Wrapper for attention to match HF checkpoint key layout (attn.attention.*)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attention = ExaoneAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self.attention(hidden_states, position_embeddings)


class ExaoneDecoderLayer(nn.Module):
    """Transformer decoder layer for EXAONE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.ln_1 = ExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = ExaoneAttentionBlock(config, layer_idx)
        self.ln_2 = ExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ExaoneMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class ExaoneOutput(ModelOutput):
    """Output for ExaoneModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class ExaoneCausalLMOutput(ModelOutput):
    """Output for ExaoneForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class ExaonePreTrainedModel(PreTrainedModel):
    """Base class for EXAONE models."""

    config_class = ExaoneConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["ExaoneDecoderLayer"]
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


class ExaoneModel(ExaonePreTrainedModel):
    """EXAONE transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.h = nn.ModuleList(
            [ExaoneDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.ln_f = ExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = ExaoneRotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> ExaoneOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # Compute position embeddings once (sliced by position_ids in RoPE)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.h:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.ln_f(hidden_states)

        return ExaoneOutput(last_hidden_state=hidden_states)


class ExaoneForCausalLM(ExaonePreTrainedModel, GenerationMixin):
    """EXAONE model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = ExaoneModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.transformer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> ExaoneCausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return ExaoneCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("ExaoneConfig", ExaoneForCausalLM)
