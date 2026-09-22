# Copyright 2018 The HuggingFace Team
# Licensed under the Apache License, Version 2.0.
# Original source: https://github.com/huggingface/transformers
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Phi-3 / Phi-4 model (sharding IR).

Source:
https://huggingface.co/microsoft/phi-4

Phi-3/Phi-4 share the Llama-style decoder (GQA, SwiGLU MLP, RMSNorm, RoPE) but
use two *fused* projections:

* ``qkv_proj``      -> fused [Q | K | V]  (colwise, ``output_sizes=[q, kv, kv]``)
* ``gate_up_proj``  -> fused [gate | up]  (colwise, ``output_sizes=[inter, inter]``)

These carry explicit sharding hints (``output_sizes`` for proportional column
sharding plus ``split_with_sizes(enable_sharding=True)``) so the IR sharder
splits each fused weight head-aligned per rank, mirroring ``modeling_nemotron_h``'s
fused ``in_proj``. The legacy structural sharder mishandled this fused layout
(see https://github.com/NVIDIA/TensorRT-LLM/issues/14679).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.utils import ModelOutput

from ..hf import AutoModelForCausalLMFactory
from ._rope_utils import init_rope_inv_freq


class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class Phi3RotaryEmbedding(nn.Module):
    """RoPE for Phi-3/Phi-4. Supports default and longrope via init_rope_inv_freq."""

    def __init__(self, config: Phi3Config):
        super().__init__()
        inv_freq, self.attention_scaling = init_rope_inv_freq(config)

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


class Phi3MLP(nn.Module):
    """SwiGLU MLP with fused gate_up_proj.

    Sharding: gate_up_proj -> colwise (output_sizes=[inter, inter]),
    down_proj -> rowwise + all_reduce.
    """

    def __init__(self, config: Phi3Config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.ops.auto_deploy.torch_linear_simple(
            x,
            self.gate_up_proj.weight,
            self.gate_up_proj.bias,
            tp_mode="colwise",
            output_sizes=[self.intermediate_size, self.intermediate_size],
            layer_type="mlp",
        )
        gate, up = torch.ops.auto_deploy.split_with_sizes(
            gate_up,
            [self.intermediate_size, self.intermediate_size],
            dim=-1,
            enable_sharding=True,
            layer_type="mlp",
        )
        down = torch.ops.auto_deploy.torch_linear_simple(
            self.act_fn(gate) * up,
            self.down_proj.weight,
            self.down_proj.bias,
            tp_mode="rowwise",
            layer_type="mlp",
        )
        down = torch.ops.auto_deploy.all_reduce(down, layer_type="mlp")
        return down


class Phi3Attention(nn.Module):
    """GQA with fused qkv_proj.

    Sharding: qkv_proj -> colwise (output_sizes=[q, kv, kv], tp_min_local_shape=head_dim),
    view -> tp_scaled_dim=2, o_proj -> rowwise + all_reduce.
    """

    def __init__(self, config: Phi3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)

        attention_bias = getattr(config, "attention_bias", False)
        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_size + 2 * self.kv_size, bias=attention_bias
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.qkv_proj.weight,
            self.qkv_proj.bias,
            tp_mode="colwise",
            output_sizes=[self.q_size, self.kv_size, self.kv_size],
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        q, k, v = torch.ops.auto_deploy.split_with_sizes(
            qkv,
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
            enable_sharding=True,
            layer_type="mha",
        )
        q = torch.ops.auto_deploy.view(
            q, [bsz, q_len, self.num_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )
        k = torch.ops.auto_deploy.view(
            k, [bsz, q_len, self.num_kv_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )
        v = torch.ops.auto_deploy.view(
            v, [bsz, q_len, self.num_kv_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )

        cos, sin = position_embeddings
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,  # scale
            None,  # sinks
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )

        attn_output = torch.ops.auto_deploy.view(
            attn_output, [bsz, q_len, self.q_size], tp_scaled_dim=2, layer_type="mha"
        )
        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output,
            self.o_proj.weight,
            self.o_proj.bias,
            tp_mode="rowwise",
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")
        return attn_output


class Phi3DecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Phi3Attention(config, layer_idx=layer_idx)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Phi3Output(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Phi3CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Phi3PreTrainedModel(PreTrainedModel):
    config_class = Phi3Config
    base_model_prefix = "model"
    _no_split_modules = ["Phi3DecoderLayer"]
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


class Phi3Model(Phi3PreTrainedModel):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Phi3DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Phi3RotaryEmbedding(config)

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
    ) -> Phi3Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return Phi3Output(last_hidden_state=hidden_states)


class Phi3ForCausalLM(Phi3PreTrainedModel, GenerationMixin):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Phi3Model(config)
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
    ) -> Phi3CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()
        return Phi3CausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("Phi3Config", Phi3ForCausalLM)
