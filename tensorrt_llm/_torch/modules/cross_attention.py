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

    The cross-attention sub-layer honors ``ModelConfig.attn_backend``: when
    set to ``"TRTLLM"`` it dispatches through the production C++ attention op
    on every supported architecture. Blackwell uses the ``trtllm_gen`` kernels,
    while earlier architectures use the legacy ``thop.attention`` wrapper.

    Encoder and decoder self-attention are unaffected and continue to use
    whatever backend ``ModelConfig.attn_backend`` selects.
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

        # Cross-attention backend selection honors ``ModelConfig.attn_backend``
        # directly, mirroring the behavior of self-attention.
        self.attn: AttentionBackend = create_attention(
            config.attn_backend,
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

    @staticmethod
    def _infer_encoder_seq_lens(
        encoder_hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Infer per-request encoder lengths from ``encoder_hidden_states``.

        Used as a fallback when ``cross_attn_metadata`` is not provided.
        Only single-request batches are unambiguous; multi-request batches
        require the caller to supply ``cross_attn_metadata`` with explicit
        ``seq_lens_kv``.
        """
        num_encoder_tokens = encoder_hidden_states.shape[0]
        num_requests = int(attn_metadata.seq_lens.numel())
        if num_requests == 1:
            return torch.tensor([num_encoder_tokens], dtype=torch.int32)
        if num_encoder_tokens % num_requests != 0:
            raise ValueError(
                "Cannot infer encoder_seq_lens from encoder_hidden_states for "
                f"a multi-request batch (num_requests={num_requests}, "
                f"num_encoder_tokens={num_encoder_tokens}). "
                "Pass an explicit cross_attn_metadata with seq_lens_kv set."
            )
        per_request = num_encoder_tokens // num_requests
        return torch.full((num_requests,), per_request, dtype=torch.int32)

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
            cross_attn_metadata: Cross-attention metadata carrying encoder
                K/V-side lengths, cross-pool block tables, etc. Must satisfy
                ``cross_attn_metadata.is_cross is True`` (i.e. the K/V-side
                ``seq_lens_kv`` differs from the Q-side ``seq_lens``). When
                ``None``, the module auto-builds a stateless cross metadata
                from ``attn_metadata`` and the inferred encoder lengths
                (single-request batches only — see
                :meth:`_infer_encoder_seq_lens`).
            skip_cross_kv_projection: When ``True``, K/V are read from the
                cross-KV cache without re-projection (decoder generation
                steps). When ``False``, K/V are projected from
                ``encoder_hidden_states`` and written into the cache (first
                decoder context step).
            all_reduce_params: AllReduce parameters for TP output projection.

        Returns:
            Output tensor ``[num_tokens, hidden_size]``.
        """
        # Resolve / build the cross-attention metadata. We require that the
        # backend sees ``metadata.is_cross is True`` so that the no-KV-cache
        # path uses the encoder-side cu_seqlens, and so that the with-KV-cache
        # path uses the cross pool.
        metadata = cross_attn_metadata
        if metadata is None:
            if skip_cross_kv_projection:
                raise ValueError(
                    "cross_attn_metadata is required when "
                    "skip_cross_kv_projection=True: the module needs the "
                    "cross-pool block tables and cached encoder lengths to "
                    "read K/V from the cache."
                )
            assert encoder_hidden_states is not None, (
                "encoder_hidden_states is required when cross-KV projection "
                "is not skipped (first decoder context step)."
            )
            encoder_seq_lens = self._infer_encoder_seq_lens(encoder_hidden_states, attn_metadata)
            metadata = attn_metadata.create_cross_metadata(
                encoder_seq_lens=encoder_seq_lens,
                enc_dec_kv_cache_manager=None,
            )
        else:
            assert metadata.is_cross, (
                "cross_attn_metadata.is_cross must be True. Build it via "
                "attn_metadata.create_cross_metadata(encoder_seq_lens, "
                "enc_dec_kv_cache_manager) so seq_lens_kv differs from "
                "seq_lens."
            )

        q = self.q_proj(hidden_states)

        if not skip_cross_kv_projection:
            assert encoder_hidden_states is not None, (
                "encoder_hidden_states is required when cross-KV projection "
                "is not skipped (first decoder context step)."
            )
            k = self.k_proj(encoder_hidden_states)
            v = self.v_proj(encoder_hidden_states)
        else:
            # Generation step: skip projection, K/V are already in the
            # cross-KV cache (written during the first decoder context step).
            # The backend reads them from ``metadata.kv_cache_manager``.
            assert metadata.kv_cache_manager is not None, (
                "skip_cross_kv_projection=True requires a populated "
                "cross-KV cache manager on cross_attn_metadata."
            )
            k = None
            v = None

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
