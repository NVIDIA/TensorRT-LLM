# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyTorch-flow BART / mBART encoder-decoder model for TensorRT-LLM.

Covers ``BartForConditionalGeneration`` and ``MBartForConditionalGeneration``.

Key differences from T5:
    - LayerNorm instead of RMSNorm.
    - Post-norm (residual → add → LayerNorm) instead of pre-norm.
    - Learned absolute positional embeddings (not relative bias).
    - GELU activation (not ReLU / gated).
    - Bias in attention and MLP projections.
    - Embedding scale = sqrt(d_model).
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BartConfig

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PredefinedAttentionMask
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.cross_attention import CrossAttention
from ..modules.embedding import Embedding, LMHead
from ..modules.encoder_decoder_layer import EncoderDecoderLayer, EncoderLayer
from ..modules.layer_norm import LayerNorm
from ..modules.linear import TensorParallelMode
from ..modules.logits_processor import LogitsProcessor
from ..modules.mlp import MLP
from .modeling_utils import PostInitCaller, register_auto_model

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _bart_encoder_hidden_size(config: BartConfig) -> int:
    return config.d_model


def _bart_decoder_hidden_size(config: BartConfig) -> int:
    return config.d_model


def _bart_encoder_num_heads(config: BartConfig) -> int:
    return config.encoder_attention_heads


def _bart_decoder_num_heads(config: BartConfig) -> int:
    return config.decoder_attention_heads


def _bart_encoder_ffn_dim(config: BartConfig) -> int:
    return config.encoder_ffn_dim


def _bart_decoder_ffn_dim(config: BartConfig) -> int:
    return config.decoder_ffn_dim


def _bart_encoder_num_layers(config: BartConfig) -> int:
    return config.encoder_layers


def _bart_decoder_num_layers(config: BartConfig) -> int:
    return config.decoder_layers


def _bart_head_dim(config: BartConfig) -> int:
    return config.d_model // config.encoder_attention_heads


# ---------------------------------------------------------------------------
# BART Attention
# ---------------------------------------------------------------------------


class BartSelfAttention(Attention):
    """BART-style MHA with bias and no positional encoding in the kernel.

    BART uses learned positional embeddings added to the input before the
    attention layer, so no RoPE or other in-kernel positional encoding is
    needed.
    """

    def __init__(
        self,
        model_config: ModelConfig[BartConfig],
        num_heads: int,
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.d_model,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=True,
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def apply_rope(self, q, k, v, position_ids):
        """BART uses learned pos embeddings, not RoPE — pass through."""
        return q, k, v


class BartCrossAttention(CrossAttention):
    """BART-style cross-attention with bias."""

    def __init__(
        self,
        model_config: ModelConfig[BartConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        num_heads = _bart_decoder_num_heads(config)
        super().__init__(
            hidden_size=config.d_model,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            encoder_hidden_size=config.d_model,
            bias=True,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class BartEncoderLayer(EncoderLayer):
    """BART encoder layer: self-attention → add+LN → MLP → add+LN (post-norm)."""

    def __init__(
        self,
        model_config: ModelConfig[BartConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        hidden_size = config.d_model
        ffn_dim = _bart_encoder_ffn_dim(config)
        num_heads = _bart_encoder_num_heads(config)

        self.self_attn = BartSelfAttention(model_config, num_heads=num_heads, layer_idx=layer_idx)

        self.self_attn_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=True,
            activation=F.gelu,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )

        self.final_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class BartDecoderLayer(EncoderDecoderLayer):
    """BART decoder layer: self-attn → add+LN → cross-attn → add+LN → MLP → add+LN."""

    def __init__(
        self,
        model_config: ModelConfig[BartConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        hidden_size = config.d_model
        ffn_dim = _bart_decoder_ffn_dim(config)
        num_heads = _bart_decoder_num_heads(config)

        self.self_attn = BartSelfAttention(model_config, num_heads=num_heads, layer_idx=layer_idx)

        self.self_attn_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

        self.cross_attn = BartCrossAttention(model_config, layer_idx=layer_idx)

        self.cross_attn_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=True,
            activation=F.gelu,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )

        self.final_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Self-attention (post-norm)
        residual = hidden_states
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-attention (post-norm)
        residual = hidden_states
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attn_metadata=attn_metadata,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # MLP (post-norm)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Encoder / Decoder stacks
# ---------------------------------------------------------------------------


class BartEncoder(nn.Module):
    """BART encoder: positional embedding + encoder layers."""

    def __init__(self, model_config: ModelConfig[BartConfig]):
        super().__init__()
        config = model_config.pretrained_config
        num_layers = _bart_encoder_num_layers(config)

        self.embed_positions = Embedding(
            config.max_position_embeddings,
            config.d_model,
            dtype=config.torch_dtype,
        )
        self.layernorm_embedding = LayerNorm(
            hidden_size=config.d_model,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(model_config, layer_idx=i) for i in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        if position_ids is not None:
            hidden_states = hidden_states + self.embed_positions(position_ids)
        hidden_states = self.layernorm_embedding(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
            )
        return hidden_states


class BartDecoder(nn.Module):
    """BART decoder: positional embedding + decoder layers."""

    def __init__(self, model_config: ModelConfig[BartConfig]):
        super().__init__()
        config = model_config.pretrained_config
        num_layers = _bart_decoder_num_layers(config)

        self.embed_positions = Embedding(
            config.max_position_embeddings,
            config.d_model,
            dtype=config.torch_dtype,
        )
        self.layernorm_embedding = LayerNorm(
            hidden_size=config.d_model,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )
        self.layers = nn.ModuleList(
            [BartDecoderLayer(model_config, layer_idx=i) for i in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
    ) -> torch.Tensor:
        if position_ids is not None:
            hidden_states = hidden_states + self.embed_positions(position_ids)
        hidden_states = self.layernorm_embedding(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
                cross_attn_metadata=cross_attn_metadata,
                skip_cross_kv_projection=skip_cross_kv_projection,
            )
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class BartModel(nn.Module):
    """BART encoder-decoder body (no lm_head)."""

    def __init__(self, model_config: ModelConfig[BartConfig]):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.shared_embedding = Embedding(
            config.vocab_size,
            config.d_model,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.embed_scale = math.sqrt(config.d_model)

        self.encoder = BartEncoder(model_config)
        self.decoder = BartDecoder(model_config)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        encoder_input_ids: Optional[torch.IntTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        encoder_position_ids: Optional[torch.IntTensor] = None,
        encoder_attn_metadata: Optional[AttentionMetadata] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None and encoder_input_ids is not None:
            assert encoder_attn_metadata is not None
            encoder_embeds = self.shared_embedding(encoder_input_ids) * self.embed_scale
            encoder_hidden_states = self.encoder(
                hidden_states=encoder_embeds,
                attn_metadata=encoder_attn_metadata,
                position_ids=encoder_position_ids,
            )

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.shared_embedding(input_ids) * self.embed_scale

        decoder_output = self.decoder(
            hidden_states=inputs_embeds,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
        )
        return decoder_output


@register_auto_model("BartForConditionalGeneration")
class BartForConditionalGeneration(nn.Module, metaclass=PostInitCaller):
    """BART encoder-decoder model with LM head."""

    def __init__(self, model_config: ModelConfig[BartConfig]):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.model = BartModel(model_config)

        self.lm_head = LMHead(
            config.vocab_size,
            config.d_model,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            reduce_output=False,
        )

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.shared_embedding.weight

        self.logits_processor = LogitsProcessor()

    def __post_init__(self):
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def __pp_init__(self):
        pass

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        encoder_input_ids: Optional[torch.IntTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_position_ids: Optional[torch.IntTensor] = None,
        encoder_attn_metadata: Optional[AttentionMetadata] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            encoder_input_ids=encoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            position_ids=position_ids,
            encoder_position_ids=encoder_position_ids,
            encoder_attn_metadata=encoder_attn_metadata,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
            inputs_embeds=inputs_embeds,
        )

        return self.logits_processor.forward(
            hidden_states,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def infer_max_seq_len(self) -> int:
        config = self.model_config.pretrained_config
        return getattr(config, "max_position_embeddings", 1024)

    def load_weights(self, weights: Dict, **kwargs):
        # TODO(Step 6): Implement full HF BART → TRT-LLM weight mapping.
        raise NotImplementedError(
            "BART weight loading is deferred to Step 6 of the porting plan "
            "(weight-loading and architecture registration)."
        )


@register_auto_model("MBartForConditionalGeneration")
class MBartForConditionalGeneration(BartForConditionalGeneration):
    """mBART reuses the BART architecture with the same weight schema."""

    pass
