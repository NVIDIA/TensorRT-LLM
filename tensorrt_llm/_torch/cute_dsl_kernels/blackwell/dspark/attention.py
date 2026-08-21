# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Public Blackwell DSpark attention specialization."""

from typing import Tuple, Type

import cutlass

from .attention_kernel import DSparkAttentionKernel


class DSparkAttention(DSparkAttentionKernel):
    """Attention over a 128-token rolling window and one 5/6-token draft block."""

    window_size = 128
    block_size = 6
    num_heads = 128
    head_dim = 512
    qk_tiler_mn = (128, 128)
    pv_tiler_mn = (128, 256)
    qk_tiler_k = 128
    page_size_draft = 8
    page_size_win = 128

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        page_size_draft: int,
        page_size_win: int,
        skip_correction_threshold: float,
        *,
        arch_str: str,
        seq_len_q: int = block_size,
        mma_qk_tiler_k: int = qk_tiler_k,
        inverse_rope_dim: int = 0,
    ):
        expected_config = {
            "acc_dtype": (acc_dtype, cutlass.Float32),
            "mma_qk_tiler_mn": (mma_qk_tiler_mn, self.qk_tiler_mn),
            "mma_pv_tiler_mn": (mma_pv_tiler_mn, self.pv_tiler_mn),
            "page_size_draft": (page_size_draft, self.page_size_draft),
            "page_size_win": (page_size_win, self.page_size_win),
            "mma_qk_tiler_k": (mma_qk_tiler_k, self.qk_tiler_k),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in expected_config.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("Unsupported DSpark kernel configuration: " + ", ".join(mismatches))
        if seq_len_q not in (5, 6):
            raise ValueError(f"DSpark block size must be 5 or 6, got {seq_len_q}")
        if inverse_rope_dim not in (0, 64):
            raise ValueError(f"DSpark inverse_rope_dim must be 0 or 64, got {inverse_rope_dim}")

        super().__init__(
            acc_dtype,
            mma_qk_tiler_mn,
            mma_pv_tiler_mn,
            max_active_clusters,
            page_size_draft,
            page_size_win,
            skip_correction_threshold,
            arch_str=arch_str,
            seq_len_q=seq_len_q,
            mma_qk_tiler_k=mma_qk_tiler_k,
        )
        # The cache ABI stores eight rows per draft block. A 128-row logical
        # descriptor lets one TMA load the valid rows and hardware-zero-fill
        # the rest while preserving the full-tile mbarrier transaction count.
        self.tma_page_size_draft = self.qk_tiler_mn[1]
        self.fixed_cache_seq_len = self.window_size + seq_len_q
        self.inverse_rope_dim = inverse_rope_dim
