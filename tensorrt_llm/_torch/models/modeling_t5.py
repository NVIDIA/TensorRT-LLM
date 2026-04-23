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
"""PyTorch-flow T5 encoder-decoder model for TensorRT-LLM.

Supports T5 (``T5ForConditionalGeneration``) and Flan-T5 (gated MLP variant).
mBART and BART share a separate ``modeling_bart.py`` file.

Architecture:
    Encoder: stack of self-attention (non-causal) layers with RMSNorm.
    Decoder: stack of self-attention (causal) + cross-attention + MLP layers.
    Top-level: encoder + decoder + lm_head.

HF config normalization:
    T5Config stores dims as ``d_model``, ``d_kv``, ``d_ff``, ``num_heads``,
    ``num_layers``, ``num_decoder_layers``.  The ``hidden_size`` /
    ``num_hidden_layers`` / ``num_attention_heads`` aliases are available via
    HF property accessors but ``num_key_value_heads`` and
    ``intermediate_size`` are not — helper functions below extract them.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5Config

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PredefinedAttentionMask
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.cross_attention import CrossAttention
from ..modules.embedding import Embedding, LMHead
from ..modules.encoder_decoder_layer import EncoderDecoderLayer, EncoderLayer
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.logits_processor import LogitsProcessor
from ..modules.mlp import MLP
from ..modules.rms_norm import RMSNorm
from .modeling_utils import PostInitCaller, register_auto_model

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _t5_num_kv_heads(config: T5Config) -> int:
    """T5 uses MHA — KV heads == Q heads."""
    return config.num_heads


def _t5_intermediate_size(config: T5Config) -> int:
    return config.d_ff


def _t5_is_gated_act(config: T5Config) -> bool:
    return getattr(config, "is_gated_act", False)


def _t5_head_dim(config: T5Config) -> int:
    return config.d_kv


def _t5_dense_act_fn(config: T5Config):
    """Resolve the T5 MLP activation function from the HF config.

    Standard T5 uses ``relu``; Flan-T5 (``gated-gelu``) uses ``gelu_new``.
    """
    act_name = getattr(config, "dense_act_fn", None) or "relu"
    _ACT_FN_MAP = {
        "relu": F.relu,
        "gelu": F.gelu,
        "gelu_new": F.gelu,
        "silu": F.silu,
        "swish": F.silu,
    }
    if act_name not in _ACT_FN_MAP:
        raise ValueError(
            f"Unsupported T5 dense_act_fn '{act_name}'. Supported: {list(_ACT_FN_MAP.keys())}"
        )
    return _ACT_FN_MAP[act_name]


def _t5_encoder_num_layers(config: T5Config) -> int:
    return config.num_layers


def _t5_decoder_num_layers(config: T5Config) -> int:
    return getattr(config, "num_decoder_layers", None) or config.num_layers


# ---------------------------------------------------------------------------
# T5 Attention (self-attention, no RoPE, no positional encoding in attn)
# ---------------------------------------------------------------------------


class T5Attention(Attention):
    """T5-style multi-head self-attention without positional embeddings.

    T5 uses relative position bias instead of absolute position embeddings.
    For stage-1, we omit the relative bias and rely on the base ``Attention``
    class with ``pos_embd_params=None`` and ``q_scaling`` set per T5 convention.
    """

    def __init__(
        self,
        model_config: ModelConfig[T5Config],
        layer_idx: Optional[int] = None,
        is_decoder: bool = True,
    ):
        config = model_config.pretrained_config
        num_heads = config.num_heads
        num_kv_heads = _t5_num_kv_heads(config)
        hidden_size = config.d_model

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=512,
            bias=False,
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            q_scaling=1.0,
        )

    def apply_rope(self, q, k, v, position_ids):
        """T5 has no RoPE — pass through unchanged."""
        return q, k, v


class T5CrossAttention(CrossAttention):
    """T5-style cross-attention with the same sizing conventions."""

    def __init__(
        self,
        model_config: ModelConfig[T5Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        num_heads = config.num_heads
        num_kv_heads = _t5_num_kv_heads(config)
        hidden_size = config.d_model

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            encoder_hidden_size=hidden_size,
            bias=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            q_scaling=1.0,
        )


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class T5EncoderLayer(EncoderLayer):
    """T5 encoder layer: pre-norm self-attention + pre-norm MLP."""

    def __init__(
        self,
        model_config: ModelConfig[T5Config],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        hidden_size = config.d_model
        intermediate_size = _t5_intermediate_size(config)
        is_gated = _t5_is_gated_act(config)

        act_fn = _t5_dense_act_fn(config)

        self.self_attn = T5Attention(model_config, layer_idx=layer_idx, is_decoder=False)

        self.input_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )

        if is_gated:
            self.mlp = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=False,
                activation=act_fn,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
            )
        else:
            self.mlp = MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=False,
                activation=act_fn,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer (self-attn + cross-attn + MLP)
# ---------------------------------------------------------------------------


class T5DecoderLayer(EncoderDecoderLayer):
    """T5 decoder layer: pre-norm self-attention + pre-norm cross-attention +
    pre-norm MLP."""

    def __init__(
        self,
        model_config: ModelConfig[T5Config],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        hidden_size = config.d_model
        intermediate_size = _t5_intermediate_size(config)
        is_gated = _t5_is_gated_act(config)

        act_fn = _t5_dense_act_fn(config)

        self.self_attn = T5Attention(model_config, layer_idx=layer_idx, is_decoder=True)

        self.cross_attn = T5CrossAttention(model_config, layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )
        self.cross_attn_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )

        if is_gated:
            self.mlp = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=False,
                activation=act_fn,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
            )
        else:
            self.mlp = MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=False,
                activation=act_fn,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
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
        # Self-attention (pre-norm)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
        )
        hidden_states = residual + hidden_states

        # Cross-attention (pre-norm)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attn_metadata=attn_metadata,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
        )
        hidden_states = residual + hidden_states

        # MLP (pre-norm)
        residual = hidden_states
        hidden_states = self.cross_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Encoder stack
# ---------------------------------------------------------------------------


class T5Encoder(nn.Module):
    """T5 encoder: shared embedding → encoder layers → final RMSNorm."""

    def __init__(self, model_config: ModelConfig[T5Config]):
        super().__init__()
        config = model_config.pretrained_config
        num_layers = _t5_encoder_num_layers(config)

        self.layers = nn.ModuleList(
            [T5EncoderLayer(model_config, layer_idx=i) for i in range(num_layers)]
        )
        self.final_layernorm = RMSNorm(
            hidden_size=config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder stack
# ---------------------------------------------------------------------------


class T5Decoder(nn.Module):
    """T5 decoder: decoder layers → final RMSNorm."""

    def __init__(self, model_config: ModelConfig[T5Config]):
        super().__init__()
        config = model_config.pretrained_config
        num_layers = _t5_decoder_num_layers(config)

        self.layers = nn.ModuleList(
            [T5DecoderLayer(model_config, layer_idx=i) for i in range(num_layers)]
        )
        self.final_layernorm = RMSNorm(
            hidden_size=config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
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
        for layer in self.layers:
            hidden_states = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
                cross_attn_metadata=cross_attn_metadata,
                skip_cross_kv_projection=skip_cross_kv_projection,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class T5Model(nn.Module):
    """Full T5 encoder-decoder model body (no lm_head).

    The shared embedding table is used for both encoder and decoder inputs
    (T5 ties encoder/decoder embeddings by default).
    """

    def __init__(self, model_config: ModelConfig[T5Config]):
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

        self.encoder = T5Encoder(model_config)
        self.decoder = T5Decoder(model_config)

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
        """Forward the full encoder-decoder model.

        When ``encoder_hidden_states`` is already provided (from a previous
        encoder pass cached by the runtime), skip the encoder entirely.

        Args:
            attn_metadata: Decoder-side attention metadata.
            input_ids: Decoder input token IDs.
            encoder_input_ids: Encoder input token IDs.
            encoder_hidden_states: Pre-computed encoder output.
            position_ids: Decoder position IDs.
            encoder_position_ids: Encoder position IDs.
            encoder_attn_metadata: Encoder-side attention metadata.
            cross_attn_metadata: Metadata for cross-attention layers.
            skip_cross_kv_projection: If ``True``, skip K/V projection in
                cross-attention (generation steps after the first context step).
            inputs_embeds: Pre-computed decoder input embeddings.
        """
        if encoder_hidden_states is None and encoder_input_ids is not None:
            assert encoder_attn_metadata is not None
            encoder_embeds = self.shared_embedding(encoder_input_ids)
            encoder_hidden_states = self.encoder(
                hidden_states=encoder_embeds,
                attn_metadata=encoder_attn_metadata,
                position_ids=encoder_position_ids,
            )

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.shared_embedding(input_ids)

        decoder_output = self.decoder(
            hidden_states=inputs_embeds,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
        )
        return decoder_output


@register_auto_model("T5ForConditionalGeneration")
class T5ForConditionalGeneration(nn.Module, metaclass=PostInitCaller):
    """T5 encoder-decoder model with LM head for conditional generation.

    Registered for the HF architecture name ``T5ForConditionalGeneration``.
    """

    def __init__(self, model_config: ModelConfig[T5Config]):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.model = T5Model(model_config)

        self.lm_head = LMHead(
            config.vocab_size,
            config.d_model,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            reduce_output=False,
        )

        # T5 ties lm_head to shared embedding by default
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.shared_embedding.weight

        self.logits_processor = LogitsProcessor()

        # T5 convention: scale logits by 1/sqrt(d_model)
        self.rescale_before_lm_head = True
        self.d_model = config.d_model

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

        if self.rescale_before_lm_head:
            hidden_states = hidden_states * (self.d_model**-0.5)

        return self.logits_processor.forward(
            hidden_states,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def infer_max_seq_len(self) -> int:
        return 512

    def load_weights(self, weights: Dict, **kwargs):
        # TODO(Step 6): Implement full HF T5 → TRT-LLM weight mapping.
        # HF T5 uses patterns like encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
        # which need non-trivial renaming to model.encoder.layers.{i}.self_attn.qkv_proj etc.
        raise NotImplementedError(
            "T5 weight loading is deferred to Step 6 of the porting plan "
            "(weight-loading and architecture registration)."
        )
