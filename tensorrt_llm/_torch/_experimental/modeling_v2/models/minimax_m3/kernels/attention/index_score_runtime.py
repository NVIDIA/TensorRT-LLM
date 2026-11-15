# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py
"""Shared scorer compilation; subclasses own scheduling flags and compiled caches."""

import cutlass
import torch
from cutlass import cute

from tensorrt_llm.logger import logger

from .m3_index_score import IndexDecodeScoreKernel


class IndexScoreRunner:
    """Compiled ``IndexDecodeScoreKernel`` variants, keyed by (index dtype, index heads, decode query length, head dim,
    split_k, early issue). Batch, query-token count, page count and block-table width stay symbolic, so one compile
    serves every decode step of its key. ``prefetch_old`` is always on; ``early`` also caps the prefetch at one block
    per CTA; ``early_trigger`` is ``INDEX_SCORE_SET`` for every key of the process."""

    kernel_cache: dict
    PRESTAGE_OLD: bool
    EARLY_TRIGGER: bool
    SUPPORTED_HEAD_DIM = 128  # capacity: fixed: the checkpoint's index head dim
    SUPPORTED_PAGE_SIZE = IndexDecodeScoreKernel.BLOCK_K
    # BLOCK_Q must fit one warp's worth of epilogue lanes.
    MAX_BLOCK_Q = 32  # capacity: fixed: the scorer's query tile
    EARLY_PREFETCH_DEPTH = 1

    @classmethod
    def is_supported(
        cls,
        *,
        q_dtype: torch.dtype,
        num_heads: int,
        head_dim: int,
        page_size: int,
        max_decode_query_len: int,
    ) -> bool:
        from tensorrt_llm._torch.kernels.blackwell.cute_ptx_utils import TORCH_TO_CUTE_DTYPE
        from tensorrt_llm._utils import is_sm_100f

        return (
            is_sm_100f()
            and q_dtype in TORCH_TO_CUTE_DTYPE
            and head_dim == cls.SUPPORTED_HEAD_DIM
            and page_size == cls.SUPPORTED_PAGE_SIZE
            and num_heads * max_decode_query_len <= cls.MAX_BLOCK_Q
            and max_decode_query_len >= 1
        )

    @classmethod
    def _compile(
        cls,
        q_dtype: torch.dtype,
        num_heads: int,
        max_decode_query_len: int,
        head_dim: int,
        split_k: int,
        early: bool,
    ) -> None:
        from tensorrt_llm._torch.kernels.blackwell.cute_ptx_utils import TORCH_TO_CUTE_DTYPE

        key = (q_dtype, num_heads, max_decode_query_len, head_dim, split_k, early)
        if key in cls.kernel_cache:
            return
        cute_dtype = TORCH_TO_CUTE_DTYPE[q_dtype]
        page_size = cls.SUPPORTED_PAGE_SIZE
        sym_total_tokens = cute.sym_int()
        sym_batch = cute.sym_int()

        # 16-element divisibility on every non-innermost stride is what the TMA descriptors for Q and K assume.
        def _sym_stride():
            return cute.sym_int64(divisibility=16)

        q_fake = cute.runtime.make_fake_tensor(
            cute_dtype,
            (sym_total_tokens, num_heads, head_dim),
            stride=(_sym_stride(), _sym_stride(), 1),
        )
        # The index-K pool may be coalesced with the main K/V cache (per-page stride above page_size * head_dim).
        k_fake = cute.runtime.make_fake_tensor(
            cute_dtype,
            (cute.sym_int(), page_size, head_dim),
            stride=(_sym_stride(), _sym_stride(), 1),
        )
        bt_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (sym_batch, cute.sym_int()), stride_order=(1, 0)
        )
        # The target passes a transposed view of the [heads, blocks, tokens] selector buffer.
        score_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32,
            (num_heads, sym_total_tokens, cute.sym_int()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
        )
        sl_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (sym_batch,), stride_order=(0,)
        )
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = IndexDecodeScoreKernel(
            cute_dtype,
            num_heads,
            max_decode_query_len,
            split_k,
            head_dim,
            prefetch_old=True,
            early_issue=early,
            prefetch_depth=cls.EARLY_PREFETCH_DEPTH if early else 0,
            prestage_old=early and cls.PRESTAGE_OLD,
            early_trigger=cls.EARLY_TRIGGER,
        )
        if cls.EARLY_TRIGGER and not cls.kernel_cache:
            logger.info(
                "tp8 index scorer: the block selector is released after the scorer's setup (INDEX_SCORE_SET)"
            )
        cls.kernel_cache[key] = cute.compile(
            kernel,
            q_fake,
            k_fake,
            bt_fake,
            score_fake,
            sl_fake,
            fake_stream,
            options="--enable-tvm-ffi",
        )
