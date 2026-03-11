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

"""Prefill-only HunYuan Dense V1 model implementation for auto_deploy export.

Source:
https://huggingface.co/tencent/Hunyuan-MT-7B

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

The HunYuan Dense V1 model is a Llama-like dense transformer with:
* Grouped-Query Attention (GQA) with QK normalization (RMSNorm on Q/K)
* Dynamic NTK-Alpha RoPE scaling
* SiLU-gated MLP
* Tied word embeddings
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class HunYuanDenseV1Config(PretrainedConfig):
    """Configuration class for HunYuan Dense V1 model.

    Bundled with the custom model implementation to enable loading on
    transformers versions that don't have this model natively registered.
    """

    model_type = "hunyuan_v1_dense"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_qk_norm: bool = True,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register config with transformers' AutoConfig
try:
    AutoConfig.register("hunyuan_v1_dense", HunYuanDenseV1Config, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("hunyuan_v1_dense", HunYuanDenseV1Config)
    except ValueError:
        pass


class HunYuanDenseV1RMSNorm(nn.Module):
    """RMS Normalization for HunYuan Dense V1.

    Uses standard torch operations so AD fusion passes can replace with
    the appropriate backend based on config.
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


class HunYuanDenseV1RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for HunYuan Dense V1.

    Supports Dynamic NTK-Alpha scaling: base = theta * alpha^(dim/(dim-2)).
    Returns full cached cos/sin values (not sliced by seq_len) to enable export.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Dynamic NTK-Alpha scaling
        if (
            rope_scaling is not None
            and rope_scaling.get("type") == "dynamic"
            and rope_scaling.get("alpha")
        ):
            alpha = rope_scaling["alpha"]
            base = base * alpha ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache with AD-specific naming
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
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


class HunYuanDenseV1MLP(nn.Module):
    """SiLU-gated MLP for HunYuan Dense V1."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class HunYuanDenseV1Attention(nn.Module):
    """GQA attention with QK normalization for HunYuan Dense V1.

    Applies RMSNorm to Q and K after RoPE (matching HF implementation order).
    Uses auto_deploy torch_attention and torch_rope_with_explicit_cos_sin ops.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        # QK normalization - applied per-head after projection, before RoPE
        self.query_layernorm = HunYuanDenseV1RMSNorm(head_dim, eps=rms_norm_eps)
        self.key_layernorm = HunYuanDenseV1RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V -> [B, S, N, D] (BSND layout)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Get cos/sin from position_embeddings (full cached from shared rotary embedding)
        cos = position_embeddings[0]  # [max_seq_len, head_dim]
        sin = position_embeddings[1]  # [max_seq_len, head_dim]
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        # Apply RoPE in BSND layout (unsqueeze_dim=2)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q,
            k,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND
        )

        # QK normalization (per-head RMSNorm, after RoPE — matches HF order).
        # Use torch_rmsnorm custom op directly so the output nodes carry proper
        # fake-tensor metadata for downstream AD transforms (avoids metadata loss
        # that can occur when plain PyTorch RMSNorm is fused by fuse_rmsnorm).
        q = torch.ops.auto_deploy.torch_rmsnorm(
            q, self.query_layernorm.weight, self.query_layernorm.variance_epsilon
        )
        k = torch.ops.auto_deploy.torch_rmsnorm(
            k, self.key_layernorm.weight, self.key_layernorm.variance_epsilon
        )

        # Attention using AD custom op in BSND layout
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scaling,
            layout="bsnd",
        )

        # Reshape and project output
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class HunYuanDenseV1DecoderLayer(nn.Module):
    """Transformer decoder layer for HunYuan Dense V1."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HunYuanDenseV1Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            layer_idx=layer_idx,
        )
        self.mlp = HunYuanDenseV1MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = HunYuanDenseV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanDenseV1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
class HunYuanDenseV1Output(ModelOutput):
    """Output for HunYuanDenseV1Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class HunYuanDenseV1CausalLMOutput(ModelOutput):
    """Output for HunYuanDenseV1ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class HunYuanDenseV1PreTrainedModel(PreTrainedModel):
    """Base class for HunYuan Dense V1 models."""

    config_class = HunYuanDenseV1Config
    base_model_prefix = "model"
    _no_split_modules = ["HunYuanDenseV1DecoderLayer"]
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


class HunYuanDenseV1Model(HunYuanDenseV1PreTrainedModel):
    """HunYuan Dense V1 transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                HunYuanDenseV1DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = HunYuanDenseV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = HunYuanDenseV1RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
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
    ) -> HunYuanDenseV1Output:
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

        return HunYuanDenseV1Output(last_hidden_state=hidden_states)


class HunYuanDenseV1ForCausalLM(HunYuanDenseV1PreTrainedModel, GenerationMixin):
    """HunYuan Dense V1 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = HunYuanDenseV1Model(config)
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
    ) -> HunYuanDenseV1CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return HunYuanDenseV1CausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls(
    "HunYuanDenseV1Config", HunYuanDenseV1ForCausalLM
)
