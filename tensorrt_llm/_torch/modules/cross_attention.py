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
"""Cross-attention module for encoder-decoder models.

Unlike self-attention, cross-attention uses Q from the decoder hidden states
and K/V from the encoder output (or from a cached cross-KV pool after the
first decoder context step).
"""

from typing import Optional

import torch
from torch import nn

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import AttentionBackend, PredefinedAttentionMask
from ..attention_backend.utils import create_attention
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from .linear import Linear, TensorParallelMode


class CrossAttention(nn.Module):
    """Cross-attention layer for encoder-decoder models.

    Computes attention where Q comes from decoder hidden states and K/V come
    from encoder output. On the first decoder context step, K/V are projected
    from encoder_hidden_states and written into the cross-KV cache pool. On
    subsequent generation steps, K/V are read from the cache without
    re-projection.

    The cross-attention backend is initialized with the ``VANILLA`` backend
    by default since the ``TRTLLM`` backend does not yet support cross-attention.
    Step 3 of the porting plan will wire a cross-capable backend.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        encoder_hidden_size: Optional[int] = None,
        max_position_embeddings: int = 512,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size or hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = getattr(config.pretrained_config, "head_dim", None)
        if not isinstance(self.head_dim, int):
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_scaling = q_scaling

        if dense_bias is None:
            dense_bias = bias

        self.mapping = config.mapping
        tp_size = self.mapping.tp_size
        if self.mapping.enable_attention_dp:
            tp_size = 1

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size - 1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        mapping = config.mapping

        self.q_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        self.k_proj = Linear(
            self.encoder_hidden_size,
            tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        self.v_proj = Linear(
            self.encoder_hidden_size,
            tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            reduce_output=True,
        )

        # Stage-1: use VANILLA backend for cross-attention.
        # Step 3 of the porting plan will enable the TRTLLM backend for cross.
        attn_backend = "VANILLA"
        self.attn: AttentionBackend = create_attention(
            attn_backend,
            layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            q_scaling=self.q_scaling,
        )

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        self.q_proj.create_weights()
        self.k_proj.create_weights()
        self.v_proj.create_weights()
        self.o_proj.create_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for cross-attention.

        Args:
            hidden_states: Decoder hidden states ``[num_tokens, hidden_size]``.
            encoder_hidden_states: Encoder output. Required on the first
                decoder context step (when ``skip_cross_kv_projection`` is
                ``False``). ``None`` for generation steps.
            attn_metadata: Decoder-side attention metadata (Q-side lengths).
            cross_attn_metadata: Cross-attention metadata carrying
                ``encoder_seq_lens``, cross-KV block tables, etc. Falls back
                to ``attn_metadata`` if ``None``.
            skip_cross_kv_projection: When ``True``, K/V are read from the
                cross-KV cache without re-projection (generation steps). When
                ``False``, K/V are projected from ``encoder_hidden_states``
                and written into the cache (first context step).
            all_reduce_params: AllReduce parameters for TP output projection.

        Returns:
            Output tensor ``[num_tokens, hidden_size]``.
        """
        metadata = cross_attn_metadata if cross_attn_metadata is not None else attn_metadata

        q = self.q_proj(hidden_states)

        if not skip_cross_kv_projection:
            assert encoder_hidden_states is not None, (
                "encoder_hidden_states is required when cross-KV projection "
                "is not skipped (first decoder context step)."
            )
            k = self.k_proj(encoder_hidden_states)
            v = self.v_proj(encoder_hidden_states)
        else:
            # Step 3/5 of the porting plan will wire a cross-capable attention
            # backend with KV cache support. Until then, the generation-step
            # path (read cross-KV from cache, skip projection) is not usable.
            raise NotImplementedError(
                "skip_cross_kv_projection=True requires a cross-attention "
                "backend with KV cache support (Step 3/5 of the porting plan)."
            )

        num_tokens = attn_metadata.num_tokens
        q = q[:num_tokens, :]

        attn_output = self.attn.forward(
            q,
            k,
            v,
            metadata,
            attention_mask=PredefinedAttentionMask.FULL,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        attn_output = self.o_proj(attn_output, all_reduce_params=all_reduce_params)
        return attn_output
