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

"""Slimmed down PyTorch Phi-4 model implementation for auto_deploy export.

Source:
https://huggingface.co/microsoft/phi-4

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout and residual dropout (inference only)
* Removed attention mask (AD manages masking via transforms and runtime)

Phi-4 uses Phi3Config (model_type="phi3") with GQA, fused QKV projection,
fused gate+up MLP (SwiGLU), and standard RoPE.
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

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class Phi4RMSNorm(nn.Module):
    """RMS Normalization for Phi-4.

    Uses standard torch operations so AD fusion passes can replace with
    the appropriate backend (flashinfer/triton) based on config.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Phi4RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Phi-4.

    Precomputes and caches cos/sin values. Returns full cached values
    (not sliced by seq_len) to enable export.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache with AD-specific naming
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Use _ad_ prefix for AutoDeploy compatibility with lift_to_meta
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return full cached cos/sin (not sliced) for export compatibility
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class Phi4MLP(nn.Module):
    """MLP layer for Phi-4 (SwiGLU activation with fused gate+up projection).

    Uses fused gate_up_proj to match HF checkpoint format.
    """

    def __init__(self, config: Phi3Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


class Phi4Attention(nn.Module):
    """Multi-head attention with GQA for Phi-4.

    Uses fused QKV projection to match HF checkpoint format.
    Receives position embeddings from the model level (shared rotary embedding).
    """

    def __init__(self, config: Phi3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)

        # Fused QKV projection (matches HF checkpoint key: qkv_proj.weight)
        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.qkv_proj = nn.Linear(config.hidden_size, op_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Fused QKV projection and split
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        kv_pos = self.num_key_value_heads * self.head_dim

        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + kv_pos]
        value_states = qkv[..., query_pos + kv_pos :]

        # Reshape to [B, S, N, head_dim] (BSND layout)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Get cos/sin from position_embeddings (full cached from shared rotary embedding)
        cos = position_embeddings[0]  # Full table: [max_seq_len, head_dim]
        sin = position_embeddings[1]  # Full table: [max_seq_len, head_dim]
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        # Apply RoPE using custom op (BSND layout, unsqueeze_dim=2)
        query_states, key_states = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            query_states,
            key_states,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # Attention using custom op (BSND layout with GQA support)
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            scale=self.scaling,
            layout="bsnd",
        )

        # Reshape output [B, S, N, head_dim] -> [B, S, hidden_size]
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Phi4DecoderLayer(nn.Module):
    """Transformer decoder layer for Phi-4."""

    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Phi4Attention(config, layer_idx=layer_idx)
        self.mlp = Phi4MLP(config)
        self.input_layernorm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Phi4ModelOutput(ModelOutput):
    """Output for Phi4Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Phi4CausalLMOutput(ModelOutput):
    """Output for Phi4ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class Phi4PreTrainedModel(PreTrainedModel):
    """Base class for Phi-4 models."""

    config_class = Phi3Config
    base_model_prefix = "model"
    _no_split_modules = ["Phi4DecoderLayer"]
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


class Phi4Model(Phi4PreTrainedModel):
    """Phi-4 transformer decoder model."""

    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Phi4DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_dim = int(head_dim * config.partial_rotary_factor)
        self.rotary_emb = Phi4RotaryEmbedding(
            rotary_dim,
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
    ) -> Phi4ModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Compute position embeddings once from shared rotary embedding
        position_embeddings = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return Phi4ModelOutput(last_hidden_state=hidden_states)


class Phi4ForCausalLM(Phi4PreTrainedModel, GenerationMixin):
    """Phi-4 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Phi3Config, **kwargs):
        super().__init__(config)
        self.model = Phi4Model(config)
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
    ) -> Phi4CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return Phi4CausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
# Phi-4 uses Phi3Config (model_type="phi3") so we register against "Phi3Config"
AutoModelForCausalLMFactory.register_custom_model_cls("Phi3Config", Phi4ForCausalLM)
