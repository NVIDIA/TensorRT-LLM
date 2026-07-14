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

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5Config

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, PredefinedAttentionMask
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.cross_attention import CrossAttention
from ..modules.embedding import Embedding, LMHead
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


def _t5_q_scaling(config: T5Config) -> float:
    # TRT-LLM attention backends use 1 / (sqrt(head_dim) * q_scaling).
    # T5 matches Hugging Face by leaving QK scores unscaled.
    return 1.0 / math.sqrt(_t5_head_dim(config))


def _t5_dense_act_fn(config: T5Config):
    """Resolve the T5 MLP activation function from the HF config.

    Standard T5 uses ``relu``; Flan-T5 (``gated-gelu``) uses ``gelu_new``.
    """

    def _gelu_new(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )

    act_name = getattr(config, "dense_act_fn", None) or "relu"
    _ACT_FN_MAP = {
        "relu": F.relu,
        "gelu": F.gelu,
        "gelu_new": _gelu_new,
        "silu": F.silu,
        "swish": F.silu,
    }
    if act_name not in _ACT_FN_MAP:
        raise ValueError(
            f"Unsupported T5 dense_act_fn '{act_name}'. Supported: {list(_ACT_FN_MAP.keys())}"
        )
    return _ACT_FN_MAP[act_name]


def _t5_gated_act_fn(config: T5Config):
    act_fn = _t5_dense_act_fn(config)

    def gated_act_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = hidden_states.chunk(2, dim=-1)
        return act_fn(gate) * up

    return gated_act_fn


def _clamp_fp16_infs(hidden_states: torch.Tensor) -> torch.Tensor:
    """Match Hugging Face T5's fp16 overflow guard after residual adds."""
    if hidden_states.dtype != torch.float16:
        return hidden_states

    clamp_value = torch.where(
        torch.isinf(hidden_states).any(),
        torch.finfo(hidden_states.dtype).max - 1000,
        torch.finfo(hidden_states.dtype).max,
    )
    return torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)


class T5LayerNorm(RMSNorm):
    """T5 RMSNorm with HF-compatible architecture-specific precision.

    On Hopper, HF uses Apex FusedRMSNorm when Apex is available. The generic
    TRT-LLM RMSNorm path matches that behavior for ByT5 BF16 decoding. On
    Blackwell, the generic fused path drifts from the HF reference, so use the
    explicit T5 computation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(hidden_size=hidden_size, eps=eps, dtype=dtype)
        self._use_hopper_rms_norm: Optional[bool] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_hopper_rms_norm is None and hidden_states.is_cuda:
            sm_version = get_sm_version()
            self._use_hopper_rms_norm = 90 <= sm_version < 100

        if self._use_hopper_rms_norm and hidden_states.dtype in (torch.float16, torch.bfloat16):
            return super().forward(hidden_states)

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def _t5_encoder_num_layers(config: T5Config) -> int:
    return config.num_layers


def _t5_decoder_num_layers(config: T5Config) -> int:
    return getattr(config, "num_decoder_layers", None) or config.num_layers


# ---------------------------------------------------------------------------
# T5 Relative Position Bias
# ---------------------------------------------------------------------------


class T5RelativePositionBias(nn.Module):
    """Learned relative position bias for T5 attention.

    Only instantiated on the first layer of each stack (encoder / decoder).
    The computed bias is shared across all layers in the same stack.
    """

    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        max_distance: int,
        is_decoder: bool,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.is_decoder = is_decoder
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads, dtype=dtype)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Return position bias of shape ``(1, num_heads, query_length, key_length)``."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        bucket_ids = self._relative_position_bucket(
            relative_position,
            bidirectional=not self.is_decoder,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(bucket_ids)
        # (query_length, key_length, num_heads) → (1, num_heads, q, k)
        return values.permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------------------------------------
# T5 Attention (self-attention with relative position bias support)
# ---------------------------------------------------------------------------


class T5Attention(Attention):
    """T5-style multi-head self-attention.

    When ``position_bias`` is provided (from a ``T5RelativePositionBias``
    module living on layer 0), it is added to the QK^T scores before
    softmax. T5 self-attention requires the TRTLLM backend so the relative
    bias can be routed through the attention backend. Without a KV cache, this
    module passes a precomputed dense relative bias as explicit attention bias.
    With a KV cache, decoder attention passes the learned relative-attention
    table to the TRTLLM backend.
    """

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
            max_position_embeddings=512,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(type=PositionEmbeddingType.relative),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            q_scaling=_t5_q_scaling(config),
            head_dim=_t5_head_dim(config),
        )

    def apply_rope(self, q, k, v, position_ids):
        """T5 has no RoPE — pass through unchanged."""
        return q, k, v

    def _local_position_bias(
        self,
        position_bias: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if position_bias.shape[1] != self.num_heads:
            head_start = self.tp_rank * self.num_heads
            head_end = head_start + self.num_heads
            if position_bias.shape[1] < head_end:
                raise ValueError(
                    f"T5 position bias has {position_bias.shape[1]} heads, "
                    f"but rank {self.tp_rank} needs heads [{head_start}, {head_end})."
                )
            position_bias = position_bias[:, head_start:head_end]

        return position_bias.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).contiguous()

    def _local_relative_attention_bias(
        self,
        relative_attention_bias: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if relative_attention_bias.shape[0] != self.num_heads:
            head_start = self.tp_rank * self.num_heads
            head_end = head_start + self.num_heads
            if relative_attention_bias.shape[0] < head_end:
                raise ValueError(
                    f"T5 relative attention bias has {relative_attention_bias.shape[0]} heads, "
                    f"but rank {self.tp_rank} needs heads [{head_start}, {head_end})."
                )
            relative_attention_bias = relative_attention_bias[head_start:head_end]

        return relative_attention_bias.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).contiguous()

    def forward(
        self,
        position_ids: Optional[torch.IntTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        attention_mask: Optional[PredefinedAttentionMask] = None,
        position_bias: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if self.attn_backend != "TRTLLM":
            raise ValueError(
                "T5 self-attention with relative position bias requires "
                f"attn_backend='TRTLLM'. Current backend: {self.attn_backend}."
            )

        forward_kwargs = dict(kwargs)
        if attn_metadata is not None and attn_metadata.kv_cache_manager is not None:
            if relative_attention_bias is None:
                raise ValueError("Cached T5 attention requires a relative attention bias table.")
            assert hidden_states is not None
            relative_attention_bias = self._local_relative_attention_bias(
                relative_attention_bias,
                hidden_states,
            )
        elif position_bias is not None:
            assert hidden_states is not None
            position_bias = self._local_position_bias(position_bias, hidden_states)
            relative_attention_bias = position_bias.squeeze(0).contiguous()
            relative_attention_max_distance = 0
            forward_kwargs["attention_window_size"] = relative_attention_bias.shape[-1]
        elif relative_attention_bias is not None:
            assert hidden_states is not None
            relative_attention_bias = self._local_relative_attention_bias(
                relative_attention_bias,
                hidden_states,
            )

        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
            **forward_kwargs,
        )


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
            q_scaling=_t5_q_scaling(config),
            head_dim=_t5_head_dim(config),
        )


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class T5EncoderLayer(nn.Module):
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

        act_fn = _t5_gated_act_fn(config) if is_gated else _t5_dense_act_fn(config)

        self.self_attn = T5Attention(model_config, layer_idx=layer_idx)

        self.input_layernorm = T5LayerNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = T5LayerNorm(
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
        position_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
            position_bias=position_bias,
        )
        hidden_states = residual + hidden_states
        hidden_states = _clamp_fp16_infs(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = _clamp_fp16_infs(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer (self-attn + cross-attn + MLP)
# ---------------------------------------------------------------------------


class T5DecoderLayer(nn.Module):
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

        act_fn = _t5_gated_act_fn(config) if is_gated else _t5_dense_act_fn(config)

        self.self_attn = T5Attention(model_config, layer_idx=layer_idx)

        self.cross_attn = T5CrossAttention(model_config, layer_idx=layer_idx)

        self.input_layernorm = T5LayerNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = T5LayerNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=config.torch_dtype,
        )
        self.cross_attn_layernorm = T5LayerNorm(
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
        position_bias: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
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
            position_bias=position_bias,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        hidden_states = residual + hidden_states
        hidden_states = _clamp_fp16_infs(hidden_states)

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
        hidden_states = _clamp_fp16_infs(hidden_states)

        # MLP (pre-norm)
        residual = hidden_states
        hidden_states = self.cross_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = _clamp_fp16_infs(hidden_states)

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

        self.relative_position_bias = T5RelativePositionBias(
            num_buckets=config.relative_attention_num_buckets,
            num_heads=config.num_heads,
            max_distance=config.relative_attention_max_distance,
            is_decoder=False,
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList(
            [T5EncoderLayer(model_config, layer_idx=i) for i in range(num_layers)]
        )
        self.final_layernorm = T5LayerNorm(
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
        max_context_q_len_override = getattr(attn_metadata, "max_context_q_len_override", None)
        if max_context_q_len_override is not None:
            seq_len = int(max_context_q_len_override)
        else:
            seq_lens = attn_metadata.seq_lens
            seq_len = hidden_states.shape[0] if seq_lens is None else int(seq_lens.max().item())
        position_bias = self.relative_position_bias(seq_len, seq_len, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                position_bias=position_bias,
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

        self.relative_position_bias = T5RelativePositionBias(
            num_buckets=config.relative_attention_num_buckets,
            num_heads=config.num_heads,
            max_distance=config.relative_attention_max_distance,
            is_decoder=True,
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList(
            [T5DecoderLayer(model_config, layer_idx=i) for i in range(num_layers)]
        )
        self.final_layernorm = T5LayerNorm(
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
        position_bias = None
        relative_attention_bias = None
        relative_attention_max_distance = 0
        if attn_metadata.kv_cache_manager is None:
            seq_len = hidden_states.shape[0]
            position_bias = self.relative_position_bias(seq_len, seq_len, hidden_states.device)
        else:
            relative_attention_bias = (
                self.relative_position_bias.relative_attention_bias.weight.transpose(0, 1)
            )
            relative_attention_max_distance = self.relative_position_bias.max_distance

        for layer in self.layers:
            hidden_states = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
                cross_attn_metadata=cross_attn_metadata,
                skip_cross_kv_projection=skip_cross_kv_projection,
                position_bias=position_bias,
                relative_attention_bias=relative_attention_bias,
                relative_attention_max_distance=relative_attention_max_distance,
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
        config = self.model_config.pretrained_config
        tllm_weights = _convert_hf_t5_weights(weights, config, dtype=self.model_config.torch_dtype)

        # __init__ aliases lm_head.weight to shared_embedding.weight when
        # tie_word_embeddings=True, so checkpoints that omit lm_head.weight are
        # handled correctly (lm_head picks up the loaded embedding automatically).
        # When lm_head.weight is present in the checkpoint, break the alias so
        # lm_head gets its own independent weight loaded from the checkpoint.
        if "lm_head.weight" in weights:
            self.lm_head.weight = nn.Parameter(torch.empty_like(self.lm_head.weight))

        for name, module in self.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            if name not in tllm_weights:
                continue
            w = tllm_weights[name]
            if hasattr(module, "load_weights"):
                module.load_weights(weights=w)
            else:
                for n, p in module.named_parameters(recurse=False):
                    if n in w[0]:
                        p.data.copy_(w[0][n][:])


def _convert_hf_t5_weights(
    hf_weights: Dict[str, torch.Tensor],
    config: T5Config,
    dtype: Optional[torch.dtype] = None,
) -> Dict:
    """Map HuggingFace T5 state_dict keys to TRT-LLM module-tree keys.

    Returns a dict keyed by TRT-LLM module path, where each value is a list of
    weight dicts suitable for ``module.load_weights(weights=...)``.

    Args:
        hf_weights: HuggingFace model ``state_dict``.
        config: HuggingFace ``T5Config``.
        dtype: Target precision.  When specified, every weight tensor is cast
            to this dtype before being returned — mirroring the legacy TRT path's
            ``convert_weight_to_dtype(params, config.dtype)`` logic.

    HF T5 weight layout:
        shared.weight
        encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
        encoder.block.{i}.layer.0.layer_norm.weight
        encoder.block.{i}.layer.{1}.DenseReluDense.{wi,wo}.weight  (non-gated)
        encoder.block.{i}.layer.{1}.DenseReluDense.{wi_0,wi_1,wo}.weight (gated)
        encoder.block.{i}.layer.{1}.layer_norm.weight
        encoder.final_layer_norm.weight
        decoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
        decoder.block.{i}.layer.1.EncDecAttention.{q,k,v,o}.weight
        decoder.block.{i}.layer.{0,1,2}.layer_norm.weight
        decoder.block.{i}.layer.2.DenseReluDense.{wi,wo|wi_0,wi_1,wo}.weight
        decoder.final_layer_norm.weight
        lm_head.weight
    """
    if dtype is not None:
        hf_weights = {k: v.to(dtype) for k, v in hf_weights.items()}

    out: Dict[str, list] = {}
    is_gated = getattr(config, "is_gated_act", False)
    enc_layers = config.num_layers
    dec_layers = getattr(config, "num_decoder_layers", None) or config.num_layers

    def _get(key: str) -> torch.Tensor:
        if key in hf_weights:
            return hf_weights[key]
        raise KeyError(f"Missing expected HF weight: {key}")

    # Shared embedding
    out["model.shared_embedding"] = [{"weight": _get("shared.weight")}]

    # LM head
    if "lm_head.weight" in hf_weights:
        out["lm_head"] = [{"weight": _get("lm_head.weight")}]

    # Encoder
    for i in range(enc_layers):
        pfx = f"encoder.block.{i}"
        tgt = f"model.encoder.layers.{i}"

        # Self-attention (fused QKV in TRT-LLM)
        out[f"{tgt}.self_attn.qkv_proj"] = [
            {"weight": _get(f"{pfx}.layer.0.SelfAttention.q.weight")},
            {"weight": _get(f"{pfx}.layer.0.SelfAttention.k.weight")},
            {"weight": _get(f"{pfx}.layer.0.SelfAttention.v.weight")},
        ]
        out[f"{tgt}.self_attn.o_proj"] = [{"weight": _get(f"{pfx}.layer.0.SelfAttention.o.weight")}]

        # Pre-attention layer norm
        out[f"{tgt}.input_layernorm"] = [{"weight": _get(f"{pfx}.layer.0.layer_norm.weight")}]

        # MLP (layer.1 for encoder)
        if is_gated:
            out[f"{tgt}.mlp.gate_up_proj"] = [
                {"weight": _get(f"{pfx}.layer.1.DenseReluDense.wi_0.weight")},
                {"weight": _get(f"{pfx}.layer.1.DenseReluDense.wi_1.weight")},
            ]
        else:
            out[f"{tgt}.mlp.up_proj"] = [
                {"weight": _get(f"{pfx}.layer.1.DenseReluDense.wi.weight")}
            ]
        out[f"{tgt}.mlp.down_proj"] = [{"weight": _get(f"{pfx}.layer.1.DenseReluDense.wo.weight")}]

        # Post-attention (pre-MLP) layer norm
        out[f"{tgt}.post_attention_layernorm"] = [
            {"weight": _get(f"{pfx}.layer.1.layer_norm.weight")}
        ]

    # Encoder relative position bias (only layer 0 in HF)
    rpb_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    if rpb_key in hf_weights:
        out["model.encoder.relative_position_bias.relative_attention_bias"] = [
            {"weight": _get(rpb_key)}
        ]

    # Encoder final layer norm
    out["model.encoder.final_layernorm"] = [{"weight": _get("encoder.final_layer_norm.weight")}]

    # Decoder
    for i in range(dec_layers):
        pfx = f"decoder.block.{i}"
        tgt = f"model.decoder.layers.{i}"

        # Self-attention (fused QKV)
        out[f"{tgt}.self_attn.qkv_proj"] = [
            {"weight": _get(f"{pfx}.layer.0.SelfAttention.q.weight")},
            {"weight": _get(f"{pfx}.layer.0.SelfAttention.k.weight")},
            {"weight": _get(f"{pfx}.layer.0.SelfAttention.v.weight")},
        ]
        out[f"{tgt}.self_attn.o_proj"] = [{"weight": _get(f"{pfx}.layer.0.SelfAttention.o.weight")}]

        # Self-attention layer norm
        out[f"{tgt}.input_layernorm"] = [{"weight": _get(f"{pfx}.layer.0.layer_norm.weight")}]

        # Cross-attention (separate projections in CrossAttention module)
        out[f"{tgt}.cross_attn.q_proj"] = [
            {"weight": _get(f"{pfx}.layer.1.EncDecAttention.q.weight")}
        ]
        out[f"{tgt}.cross_attn.k_proj"] = [
            {"weight": _get(f"{pfx}.layer.1.EncDecAttention.k.weight")}
        ]
        out[f"{tgt}.cross_attn.v_proj"] = [
            {"weight": _get(f"{pfx}.layer.1.EncDecAttention.v.weight")}
        ]
        out[f"{tgt}.cross_attn.o_proj"] = [
            {"weight": _get(f"{pfx}.layer.1.EncDecAttention.o.weight")}
        ]

        # Cross-attention layer norm (post_attention_layernorm in T5DecoderLayer)
        out[f"{tgt}.post_attention_layernorm"] = [
            {"weight": _get(f"{pfx}.layer.1.layer_norm.weight")}
        ]

        # MLP (layer.2 for decoder)
        if is_gated:
            out[f"{tgt}.mlp.gate_up_proj"] = [
                {"weight": _get(f"{pfx}.layer.2.DenseReluDense.wi_0.weight")},
                {"weight": _get(f"{pfx}.layer.2.DenseReluDense.wi_1.weight")},
            ]
        else:
            out[f"{tgt}.mlp.up_proj"] = [
                {"weight": _get(f"{pfx}.layer.2.DenseReluDense.wi.weight")}
            ]
        out[f"{tgt}.mlp.down_proj"] = [{"weight": _get(f"{pfx}.layer.2.DenseReluDense.wo.weight")}]

        # Pre-MLP layer norm
        out[f"{tgt}.cross_attn_layernorm"] = [{"weight": _get(f"{pfx}.layer.2.layer_norm.weight")}]

    # Decoder relative position bias (only layer 0 in HF)
    rpb_key = "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    if rpb_key in hf_weights:
        out["model.decoder.relative_position_bias.relative_attention_bias"] = [
            {"weight": _get(rpb_key)}
        ]

    # Decoder final layer norm
    out["model.decoder.final_layernorm"] = [{"weight": _get("decoder.final_layer_norm.weight")}]

    return out
