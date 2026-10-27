# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""JIT-loaded CUDA/C++ extension for paged decode split-KV scheduling."""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "build_decode_schedule.cu")

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

_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="sparse_decode_schedule_ext",
            sources=[_SRC],
            extra_cflags=_extra_cflags,
            extra_cuda_cflags=_extra_cuda_cflags,
            verbose=False,
        )
    return _ext


def build_decode_schedule(
    seqused_k: torch.Tensor,
    *,
    page_size: int,
    seqlen_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    enable_cuda_graph: bool = False,
    max_grid_size: int = 0,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> dict[str, object]:
    """GPU-only schedule build: single CUDA kernel produces all schedule
    index arrays on device.  Only a small summary tensor is D2H'd at the end
    so the wrapper can size O_partial, pick the kernel grid, and choose
    split/non-split compile path.

    ``max_seqlen_k`` is required as the host-side worst-case bound for
    padding the work-tile arrays.
    """

    raw = _load_ext().build_decode_schedule(
        seqused_k,
        int(page_size),
        int(seqlen_q),
        int(num_qo_heads),
        int(num_kv_heads),
        int(head_dim),
        int(max_seqlen_k),
        bool(enable_cuda_graph),
        int(max_grid_size),
        int(fixed_split_size),
        bool(disable_split_kv),
    )
    # The CUDA kernel writes into worst-case-padded buffers (size =
    # batch * num_q_tiles * max_pages_global) but only the first
    # ``padded_work_count`` entries are valid.  Downstream consumers
    # (tile_scheduler) take grid size from ``request_indices.shape[0]``
    # so we narrow the views to that count; the underlying allocation
    # is unchanged so this is a view, no copy.
    pad = int(raw["padded_work_count"])
    for key in (
        "request_indices",
        "qo_tile_indices",
        "kv_tile_indices",
        "block_valid_mask",
    ):
        raw[key] = raw[key].narrow(0, 0, pad)
    return raw


__all__ = ["build_decode_schedule"]
