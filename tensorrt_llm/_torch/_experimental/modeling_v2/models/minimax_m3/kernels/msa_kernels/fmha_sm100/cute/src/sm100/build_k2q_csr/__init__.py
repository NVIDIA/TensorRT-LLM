# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""JIT-loaded CUDA C++ extension for the q2k -> k2q CSR builder.

This module compiles ``build_k2q_csr.cu`` on first import via
``torch.utils.cpp_extension.load`` and exposes ``run_build_k2q_csr``.
The extension is cached in ``~/.cache/torch_extensions/`` so subsequent
imports are cheap.

The kernel pipeline is tuned and verified for SM100; other
architectures are not supported.
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "build_k2q_csr.cu")

_extra_cflags = ["-O3"]
_extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-arch=sm_100",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "-I/usr/local/cuda/include/cccl",
]

# TensorRT-LLM: the extension name differs from the packaged top-level fmha_sm100's, whose
# build_k2q_csr.cu differs from this copy's, so the two never share or evict one build cache entry.
_ext = load(
    name="sparse_build_k2q_csr_m31_ext",
    sources=[_SRC],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    verbose=False,
)


def run_build_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
) -> None:
    """In-place fill of ``row_ptr`` and ``q_idx``.

    Args:
      q2k:           int32 [H, total_q, topK] contiguous (CUDA).
      cu_seqlens_q:  int32 [B+1] contiguous (CUDA).
      cu_seqlens_k:  int32 [B+1] contiguous (CUDA).
      row_ptr:       int32 [H, total_rows + 1] CUDA, written in place.
      q_idx:         int32 [H, total_q * topK] CUDA, written in place
                     (trailing slots set to -1).
      topk:          must be in {4, 8, 16, 32}.
      blk_kv:        must equal 128.
      total_rows:    sum over batches of ceil(seqlen_k / blk_kv).
      max_kv_blocks: max over batches of ceil(seqlen_k / blk_kv); upper bound
                     used to size the row_map workspace and clamp valid kv ids.
    """
    _ext.run_build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
    )


def run_build_k2q_csr_with_schedule(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    qsplit_idx: torch.Tensor,
    split_counts: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    target_q_per_cta: int,
    work_capacity: int,
    max_seqlen_q: int,
) -> None:
    """In-place fill of CSR plus fused sparse attention schedule metadata."""
    _ext.run_build_k2q_csr_with_schedule(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        scheduler_metadata,
        work_count,
        qsplit_idx,
        split_counts,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(target_q_per_cta),
        int(work_capacity),
        int(max_seqlen_q),
    )


def is_supported(topk: int, blk_kv: int) -> bool:
    return int(topk) in (4, 8, 16, 32) and int(blk_kv) == 128


__all__ = ["run_build_k2q_csr", "run_build_k2q_csr_with_schedule", "is_supported"]
