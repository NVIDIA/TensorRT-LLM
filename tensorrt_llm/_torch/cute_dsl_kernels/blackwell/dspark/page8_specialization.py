# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Blackwell DSpark Attention specialization for five/six-token draft blocks.

The rolling-window stream keeps its native 128-row page while the compressed
stream uses one eight-row physical page per request. Five or six rows contain
the current DSpark block and the remaining rows are padding masked by the
logical sequence length.
"""

from typing import Tuple, Type

import cutlass

from .attention_kernel import DSparkAttentionKernel


class DSparkAttentionForward(DSparkAttentionKernel):
    """DSpark attention with a five/six-row block in an eight-row page."""

    window_size = 128
    block_size = 6
    seq_len_k = window_size + block_size
    seq_len_q = block_size
    num_heads = 128
    head_dim = 512
    qk_tiler_mn = (128, 128)
    pv_tiler_mn = (128, 256)
    qk_tiler_k = 128
    page_size_cmp = 8
    page_size_win = 128

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        page_size_cmp: int,
        page_size_win: int,
        skip_correction_threshold: float,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        *,
        arch_str: str,
        seq_len_q: int = block_size,
        mma_qk_tiler_k: int = qk_tiler_k,
        inverse_rope_dim: int = 0,
    ):
        config = {
            "acc_dtype": (acc_dtype, cutlass.Float32),
            "mma_qk_tiler_mn": (mma_qk_tiler_mn, self.qk_tiler_mn),
            "mma_pv_tiler_mn": (mma_pv_tiler_mn, self.pv_tiler_mn),
            "page_size_cmp": (page_size_cmp, self.page_size_cmp),
            "page_size_win": (page_size_win, self.page_size_win),
            "is_persistent": (is_persistent, True),
            "is_var_seq": (is_var_seq, False),
            "is_var_split_kv": (is_var_split_kv, False),
            "mma_qk_tiler_k": (mma_qk_tiler_k, self.qk_tiler_k),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in config.items()
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
            page_size_cmp,
            page_size_win,
            skip_correction_threshold,
            is_persistent,
            is_var_seq,
            is_var_split_kv,
            arch_str=arch_str,
            seq_len_q=seq_len_q,
            mma_qk_tiler_k=mma_qk_tiler_k,
        )
        # The page-table/cache ABI stays at eight physical rows. Build the TMA
        # descriptor with a 128-row span so the single valid page is loaded
        # once and rows 8-127 are produced by hardware OOB zero fill. This
        # preserves the full-tile mbarrier transaction count while removing
        # the generic loader's 16 small page-copy issues.
        self.tma_page_size_cmp = self.qk_tiler_mn[1]
        self.fixed_cache_seq_len = self.window_size + seq_len_q
        self.implicit_cmp_page_table = True
        self.attn_sink_is_scaled = True
        self.window_valid_len_from_tensor = True
        # Fused inverse-RoPE epilogue on the last lanes of the output head.
        self.inverse_rope_dim = inverse_rope_dim
