# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import torch

from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import SkipSoftmaxParams
from tensorrt_llm._torch.visual_gen.attention.skip_softmax_threshold import (
    resolve_skip_softmax_threshold_scale_factor,
)
from tensorrt_llm._torch.visual_gen.catalog.attention.cute_fmha import cute_fmha
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.fmha_interface import (
    _cute_dsl_import_error as _cute_dsl_import_error,
)
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from .....attention.backends.interface import PredefinedAttentionMask
from ..interface import AttentionBackend, AttentionTensorLayout


class CuTeDSLAttention(AttentionBackend):
    """
    CuTe DSL (NVIDIA kernels) backend for diffusion models.

    JIT-compiles BlackwellFusedMultiHeadAttentionForward on first use and caches the
    compiled artifact per (dtype, mask, head_dim, ...) configuration.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        # Legacy static threshold, superseded by `sparse_params`'s timestep-aware
        # scheduler for every in-tree construction path (`create_attention` never
        # forwards this). Kept only for direct-construction debug/testing use
        # (e.g. unit tests that want a fixed threshold without a scheduler);
        # mutually exclusive with `sparse_params` below.
        skip_softmax_threshold_scale: float | None = None,
        sparse_params: SkipSoftmaxParams | None = None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        if skip_softmax_threshold_scale is not None and sparse_params is not None:
            raise ValueError("Set either skip_softmax_threshold_scale or sparse_params, not both.")
        self.skip_softmax_threshold_scale = skip_softmax_threshold_scale
        self.sparse_params = sparse_params
        self.scale = 1.0 / math.sqrt(head_dim)

        # CuTe DSL expects [B, S, H, D] format
        self._preferred_layout = AttentionTensorLayout.NHD

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using CuTe DSL (NVIDIA kernels).

        Dimensions are derived from tensor shapes (NHD layout: ``[B, S, H, D]``).

        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            v: Value tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            attention_mask: Attention mask type (CAUSAL or FULL)

        Returns:
            Output tensor [batch_size, seq_len, num_heads, head_dim]
        """
        output, _ = self.forward_with_lse(q, k, v, attention_mask=attention_mask, **kwargs)
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and log-sum-exp (LSE).

        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
            lse:    [batch_size, num_heads, seq_len] - log-sum-exp per query position,
                    always in float32. Used for numerically stable combination of
                    partial attention results in Attention2D parallelism.
        """
        threshold = resolve_skip_softmax_threshold_scale_factor(
            self.skip_softmax_threshold_scale, self.sparse_params, kwargs.get("timestep")
        )
        recipe = self.quant_attention_config
        return cute_fmha(
            q,
            k,
            v,
            sm_scale=self.scale,
            is_causal=attention_mask == PredefinedAttentionMask.CAUSAL,
            qk_dtype=recipe.qk_dtype if recipe is not None else None,
            v_block_size=recipe.v_block_size if recipe is not None else None,
            static_q_scale=kwargs.get("static_q_scale"),
            static_k_scale=kwargs.get("static_k_scale"),
            static_v_scale=kwargs.get("static_v_scale"),
            skip_softmax_threshold_scale_factor=threshold,
        )

    @classmethod
    def support_lse(cls) -> bool:
        return True

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
