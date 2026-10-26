# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 index-K cache (``indexer_kv_dtype="nvfp4"``): the writer, the prefill gather and the page remap.

A page is the Q8KV4 decode indexer's packed layout: 8192 bytes of E2M1 codes (token ``t`` at ``t * 64``, low nibble
= even element) then 1024 bytes of E4M3 16-element scales (``8192 + t * 8 + g``), no global scale; 72 bytes per
token instead of FP8's 128. The kernels are built into th_common for sm_107 (C++ / CUDA, ``torch.ops.m31idxk``):

* ``write``: the unit-scale NVFP4 quantization of vLLM's fused index-K writer, from E4M3 or bf16 rows, into the
  token's slot; slots < 0 are skipped.
* ``gather_fp8``: each page's elements as ``E4M3_RN_satfinite(e2m1 * scale)`` into contiguous FP8 pages, so the
  FP8 index scorers (the fmha_sm100 proxy pass and the CuTe DSL scorer) run unchanged on the gathered copy, as
  vLLM's NVFP4 prefill indexer does.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

PAGE = 128
DIM = 128
BYTES_PER_TOKEN = DIM // 2 + DIM // 16  # 72
PAGE_BYTES = PAGE * BYTES_PER_TOKEN  # 9216

_STATE = {"loaded": False, "counts": {}}
# Default ON; the kill restores the original pageable offset copy for item pricing.
_COMPACT_NOSYNC = os.environ.get("TRTLLM_M31_DISABLE_NVFP4_COMPACT_NOSYNC", "0") != "1"


def _load() -> None:
    if _STATE["loaded"]:
        return
    if not all(
        hasattr(torch.ops.m31idxk, name)
        for name in ("nvfp4_idxk_write", "nvfp4_idxk_write_from_sink", "nvfp4_idxk_gather_fp8")
    ):
        raise RuntimeError(
            "MiniMax-M3 indexer_kv_dtype='nvfp4' needs the torch.ops.m31idxk kernels, which this build does not "
            "have (they are built for sm_107); set TRTLLM_M31_DISABLE_AGG_NVFP4_INDEXK=1 to fall back to FP8 index-K"
        )
    _STATE["loaded"] = True
    logger.info("[M3.1] AGG_NVFP4_INDEXK kernels loaded: in-tree th_common")


def _count(kind: str) -> None:
    counts = _STATE["counts"]
    counts[kind] = n = counts.get(kind, 0) + 1
    if n & (n - 1) == 0:
        logger.info(f"[M3.1] AGG_NVFP4_INDEXK {kind} taken n={n}")


def is_nvfp4_index_cache(cache: Optional[torch.Tensor]) -> bool:
    """Whether ``cache`` is an NVFP4 index-K view: [pages, 1, 128, 72] uint8 (or its NHD transpose)."""
    return (
        cache is not None
        and cache.dtype == torch.uint8
        and cache.dim() == 4
        and int(cache.shape[-1]) == BYTES_PER_TOKEN
    )


def page_view(cache: torch.Tensor) -> torch.Tensor:
    """The cache as [pages, 9216] bytes (each page contiguous, the page stride kept)."""
    return cache.as_strided((int(cache.shape[0]), PAGE_BYTES), (int(cache.stride(0)), 1))


def write(rows: torch.Tensor, out_cache_loc: torch.Tensor, cache: torch.Tensor) -> None:
    """Quantize index-K ``rows`` [T, 128] (E4M3 or bf16) into ``cache`` at ``out_cache_loc`` (slots < 0 skipped)."""
    _load()
    torch.ops.m31idxk.nvfp4_idxk_write(rows, out_cache_loc, page_view(cache))
    if not torch.cuda.is_current_stream_capturing():
        _count("write")


def write_from_sink(
    sink: torch.Tensor, out_cache_loc: torch.Tensor, num_tokens: int, cache: torch.Tensor
) -> None:
    """Pack the first ``num_tokens`` slots of ``out_cache_loc`` into ``cache`` from the producer's FP8 ``sink``.

    ``sink`` [pages, 1, 128, 128] holds each token's E4M3 index-K at its slot (written by the fused producer); slots
    < 0 are skipped. Capture-safe (no host sync).
    """
    _load()
    torch.ops.m31idxk.nvfp4_idxk_write_from_sink(
        sink, out_cache_loc, int(num_tokens), page_view(cache)
    )
    if not torch.cuda.is_current_stream_capturing():
        _count("write")


def gather_fp8(
    cache: torch.Tensor, page_ids: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pages ``page_ids`` (int32, < 0 gives zeros) as contiguous FP8 [N, 1, 128, 128]."""
    _load()
    n = int(page_ids.shape[0])
    if out is None:
        out = torch.empty((n, 1, PAGE, DIM), dtype=torch.float8_e4m3fn, device=cache.device)
    torch.ops.m31idxk.nvfp4_idxk_gather_fp8(page_view(cache), page_ids, out)
    if not torch.cuda.is_current_stream_capturing():
        _count("gather")
    return out


def compact_step_pages(
    block_table: torch.Tensor, kv_lens_cpu: torch.Tensor, rows: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The step's pages in the flattened page-table order and the tables that address the compact copy.

    Row ``r`` holds ``ceil(kv_lens_cpu[r] / 128)`` pages, ``block_table[r, :n_r]``; the flattened order is the one
    ``msa_kv_indices`` uses. Returns ``(page_ids [P] int32, kv_indices [P] int32 = 0..P-1, block_table [rows, cols]
    int32`` with row ``r`` = ``offset_r + 0..n_r-1``).
    """
    counts: List[int] = [-(-int(n) // PAGE) for n in kv_lens_cpu[:rows].tolist()]
    device = block_table.device
    parts = [block_table[r, :c] for r, c in enumerate(counts) if c > 0]
    page_ids = (
        torch.cat(parts).to(torch.int32)
        if parts
        else torch.empty((0,), dtype=torch.int32, device=device)
    )
    total = int(page_ids.shape[0])
    if _COMPACT_NOSYNC:
        # The row offsets go through pinned memory with a non-blocking copy: a pageable host-to-device copy synchronizes
        # the stream first, which in a breakable-graph replay made the host wait for all queued work at the first sparse
        # layer of every step with context rows (same values).
        starts: List[int] = [0] * rows
        for r in range(1, rows):
            starts[r] = starts[r - 1] + counts[r - 1]
        offsets = torch.tensor(starts, dtype=torch.int32)
        if device.type == "cuda":
            offsets = maybe_pin_memory(offsets).to(device, non_blocking=True)
        offsets = offsets.unsqueeze(1)
    else:
        offsets = torch.tensor(
            [sum(counts[:r]) for r in range(rows)], dtype=torch.int32, device=device
        ).unsqueeze(1)
    cols = max(1, max(counts) if counts else 1, int(block_table.shape[1]))
    table = offsets + torch.arange(cols, dtype=torch.int32, device=device).unsqueeze(0)
    table = torch.minimum(table, torch.full_like(table, max(total - 1, 0)))
    return page_ids, torch.arange(total, dtype=torch.int32, device=device), table


def decode_history_workspace(seq_lens: torch.Tensor, cols: int) -> torch.Tensor:
    """The Q8KV4 reader's scheduler workspace: each request's history-page count (pages before its local one),
    prefix-summed (uint8 [(B + 1) * 4], int32 view). Depends on the step only."""
    batch = int(seq_lens.shape[0])
    workspace = torch.empty(((batch + 1) * 4,), dtype=torch.uint8, device=seq_lens.device)
    scheduler = workspace.view(torch.int32)
    history = torch.div(seq_lens - 1, PAGE, rounding_mode="floor").clamp_(min=0, max=cols)
    scheduler[:1].zero_()
    torch.cumsum(history, dim=0, out=scheduler[1:])
    return workspace


_READERS = {}


def score_decode_rows(
    idx_q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    num_heads: int,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Index scores of decode rows straight from NVFP4 index-K pages, [heads, cols, rows * query_len] fp32 in the
    selector's layout: the Q8KV4 decode reader (the vendored fmha_sm100_q8kv8 run_scores_hpr; exact FP16 dequant, FP32
    accumulation), the reader every pure-decode step of the agg target uses. History pages are written; a row's local
    and later pages are left as allocated, which the selector never ranks (local blocks are forced)."""
    from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.fmha_sm100_q8kv8 import (
        q8_indexer_interface as qi,
    )

    batch, cols = int(block_table.shape[0]), int(block_table.shape[1])
    device = idx_q.device
    _, view, kernel_out = qi.q8kv4_hpr_buffer(batch, cols, num_heads, int(query_len), device)
    if workspace is None:
        workspace = decode_history_workspace(seq_lens, cols)
    key = (device, int(num_heads))
    w = _READERS.get(key)
    if w is None:
        w = _READERS[key] = qi.BatchDecodeIndexerQ8KV4Wrapper(num_heads=int(num_heads))
    w._block_table = block_table
    w._seq_lens = seq_lens
    w._workspace = workspace
    w._query_len = int(query_len)
    w._scores = torch.empty((batch, 0, 0), dtype=torch.float32, device=device)
    w.run_scores_hpr(
        idx_q, cache.reshape(int(cache.shape[0]), PAGE, int(cache.shape[-1])), kernel_out
    )
    if not torch.cuda.is_current_stream_capturing():
        _count("decode_rows")
    return view


__all__ = [
    "BYTES_PER_TOKEN",
    "PAGE_BYTES",
    "compact_step_pages",
    "decode_history_workspace",
    "gather_fp8",
    "is_nvfp4_index_cache",
    "page_view",
    "score_decode_rows",
    "write",
    "write_from_sink",
]
