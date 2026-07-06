# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent paged-KV table staging for the MiniMax-M3 sparse path.

Provides the CUDA-graph-stable device buffers the M3 sparse attention
reads every step:

  * :func:`build_stable_kv_indices` -- computes the flat paged-KV page
    table (global page ids) into a persistent int32 buffer using
    vectorized ops (avoiding the Python-per-batch loop the reference
    helper uses).  The buffer's ``data_ptr()`` is stable across calls,
    so a captured CUDA graph keeps reading current values on every
    replay.
  * :class:`MsaPlanCache` -- owns those buffers plus the per-rank
    sparse geometry; refreshed once per scheduler step from
    :meth:`MiniMaxM3AttentionMetadata.prepare` (outside any capture
    window).
  * :func:`set_global_msa_geometry` -- process-wide geometry
    registration from the backend constructor, so CUDA-graph metadata
    instances (created separately by the graph runner) can stage
    before any forward has run.

Historical note: this module used to also stage ``fmha_sm100_plan``
outputs for the legacy MSA host-driver decode.  That driver was
replaced by the in-tree graph-safe driver
(``decode_wrapper/``); prefill plans in-forward (it always
runs eagerly) and decode assembles launch arguments from device
tensors, so no plan staging remains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# kv_indices: vectorized, in-place into persistent buffer
# ---------------------------------------------------------------------------


def build_stable_kv_indices(
    *,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    page_size: int,
    dst: torch.Tensor,
    kv_page_indptr_dst: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the flat paged-KV page table into ``dst``.

    Vectorized replacement for
    :func:`msa_backend._build_kv_indices_and_lens`. The old helper
    looped in Python (``for b in range(batch)``) and allocated a fresh
    ``kv_indices`` per call via ``torch.cat``, which is fine in eager
    mode but has no stable ``data_ptr()`` under CUDA graph replay.

    Parameters
    ----------
    req_to_token : ``[max_reqs, max_kv_len]`` int32 on cache device.
    slot_ids : ``[batch]`` int32 on cache device.  ``req_to_token``
        row indices for the current batch.
    seq_lens : ``[batch]`` int32 on cache device.  Per-request effective
        KV length.
    seq_lens_cpu : ``[batch]`` int32 or int64 on CPU.  Same values, used
        for the CPU-side sizing math so we do not force a D2H sync.
    page_size : int, must equal ``req_to_token.stride`` block width
        (the M3 KV cache manager enforces this).
    dst : preallocated int32 buffer sized ``>= max_batch * max_pages``.
        The output ``kv_indices`` view is ``dst[:total_pages]``.
    kv_page_indptr_dst : preallocated int32 buffer sized ``>= batch + 1``.
        The output ``kv_page_indptr`` view is ``kv_page_indptr_dst[:batch+1]``.

    Returns
    -------
    (kv_indices, kv_page_indptr) as views into ``dst`` / ``kv_page_indptr_dst``.
    The two views' ``data_ptr()`` is stable across calls because it
    aliases the destination buffers.
    """
    device = req_to_token.device
    batch = int(seq_lens_cpu.shape[0])
    max_kv_len = int(req_to_token.shape[1])
    max_pages_per_seq = max_kv_len // page_size

    if batch == 0:
        return dst[:0], kv_page_indptr_dst[:1].zero_()

    # Number of pages per request: ceil(seq_len / page_size). Do it on
    # CPU so kv_page_indptr can be prepared as ints for the row/col
    # gather without triggering a D2H sync on seq_lens.
    seq_lens_cpu_long = seq_lens_cpu.to(torch.long).cpu()
    num_pages_cpu = (seq_lens_cpu_long + page_size - 1) // page_size
    num_pages_cpu = num_pages_cpu.clamp_min(0)
    total_pages = int(num_pages_cpu.sum().item())
    if total_pages > int(dst.shape[0]):
        raise RuntimeError(
            f"MSA kv_indices persistent buffer too small: capacity {int(dst.shape[0])} "
            f"< total pages {total_pages}. Increase max_kv_indices on allocate()."
        )

    # Build kv_page_indptr on CPU then copy the prefix into the
    # persistent CPU-shadow-less buffer. Values are per-batch cumulative
    # page counts (starts at 0).
    kv_page_indptr_cpu = torch.empty(batch + 1, dtype=torch.int32)
    kv_page_indptr_cpu[0] = 0
    kv_page_indptr_cpu[1:].copy_(num_pages_cpu.to(torch.int32).cumsum(0))
    kv_page_indptr_dst[: batch + 1].copy_(
        kv_page_indptr_cpu.to(device=device, non_blocking=True), non_blocking=True
    )

    # Vectorized page-index gather:
    #   req_rows = req_to_token[slot_ids] -> [batch, max_kv_len] slot ids
    #   For each request b, valid pages are indices 0..num_pages[b]-1;
    #   the p-th page's first-slot column is p * page_size; its page
    #   id is req_rows[b, p*page_size] // page_size (mirroring the
    #   in-tree helper).  We build a max-sized (batch, max_pages_per_seq)
    #   grid, gather with clamped column indices, then mask trailing
    #   invalid pages before scatter into ``dst``.
    slot_ids_long = slot_ids.to(torch.long)
    req_rows = req_to_token.index_select(0, slot_ids_long).to(torch.long)

    max_valid_pages = max(1, max_pages_per_seq)
    pages_grid = torch.arange(max_valid_pages, device=device, dtype=torch.long)
    # Column index of the first slot of page p: p * page_size, clamped
    # to ``max_kv_len - 1`` for out-of-range pages so the gather does
    # not fault.  Out-of-range page ids are trimmed by the batch mask
    # below.
    col_idx = (pages_grid * page_size).clamp_max(max(0, max_kv_len - 1))
    # Broadcast to [batch, max_pages_per_seq]
    col_idx_b = col_idx.unsqueeze(0).expand(batch, -1)
    # The gathered values are *global* page ids into the paged cache;
    # ``req_rows`` holds valid slot ids by construction so no value
    # clamp is applied.  (An earlier version clamped to
    # ``max_pages_per_seq - 1`` — the per-request page count — which
    # collapsed the page table for every request whose global page ids
    # exceed that count and corrupted all requests after the first;
    # same defect as msa_backend._build_kv_indices_and_lens, fixed in
    # brb/feat/minimax_m3_mxfp8_msa commit 677bcb45e5.)
    gathered = (req_rows.gather(1, col_idx_b) // page_size).to(torch.int32)

    # Build a valid-page mask per request:
    #   mask[b, p] = p < num_pages[b]
    num_pages_dev = num_pages_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    mask = pages_grid.unsqueeze(0) < num_pages_dev.unsqueeze(1)  # [batch, max_pages_per_seq]

    # Compact into the flat ``dst`` prefix using boolean indexing.
    # torch.masked_select preserves row-major (batch, page) order which
    # matches the ``concat([pages_of(seq_0), pages_of(seq_1), ...])``
    # layout kv_page_indptr encodes.
    packed = torch.masked_select(gathered, mask)
    dst[:total_pages].copy_(packed, non_blocking=True)

    return dst[:total_pages], kv_page_indptr_dst[: batch + 1]


# ---------------------------------------------------------------------------
# Cross-run cache
# ---------------------------------------------------------------------------


_GLOBAL_MSA_GEOMETRY: Optional["MsaPlanCacheGeometry"] = None


def set_global_msa_geometry(geometry: "MsaPlanCacheGeometry") -> None:
    """Register the per-rank M3 sparse geometry process-wide.

    Called from ``MiniMaxM3MSARuntimeBackend.__init__`` — i.e. at layer
    construction, before any forward and therefore before any CUDA
    graph capture.  ``MiniMaxM3AttentionMetadata.prepare()`` reads this
    so the MSA plan / kv-indices staging runs for *every* metadata
    instance, including the separate instances the CUDA graph runner
    creates.  Publishing only from the first sparse forward (the old
    scheme) left graph-capture metadata without a geometry: their
    ``prepare()`` skipped the plan pre-build and the captured forward
    fell back to in-forward planning, freezing capture-time host values
    into every replay.

    All sparse layers on a rank share one geometry; the first writer
    wins and later identical writes are no-ops.
    """
    global _GLOBAL_MSA_GEOMETRY
    if _GLOBAL_MSA_GEOMETRY is None:
        _GLOBAL_MSA_GEOMETRY = geometry


def get_global_msa_geometry() -> Optional["MsaPlanCacheGeometry"]:
    return _GLOBAL_MSA_GEOMETRY


@dataclass
class MsaPlanCacheGeometry:
    """Per-rank M3 model geometry needed to allocate the plan cache.

    Populated by the model layer (``MiniMaxM3Attention._sparse_forward``)
    the first time it dispatches through the MSA backend.  All sparse
    layers share the same geometry so the value written by the first
    layer is authoritative for the rest.
    """

    num_q_heads: int
    num_kv_heads: int
    num_index_heads: int
    head_dim: int
    block_size: int
    topk: int
    init_blocks: int
    local_blocks: int


class MsaPlanCache:
    """Persistent paged-KV table staging for the M3 sparse decode path.

    Historically this cache also staged ``fmha_sm100_plan`` outputs for
    the legacy MSA host-driver decode; that path was removed once the
    in-tree graph-safe driver (``decode_wrapper``) reached
    bit-parity and passed GSM8K under CUDA graphs.  What remains is the
    piece both prefill and decode still need: ``kv_indices`` /
    ``kv_page_indptr`` computed each step into persistent device
    buffers whose ``data_ptr()`` is stable across CUDA graph replays.

    Lifecycle
    ---------
      1. Allocated lazily on the first ``build_from_metadata`` call once
         the geometry / device / capacity are known.
      2. Every subsequent ``build_from_metadata`` call rewrites the
         buffer contents in-place for the current scheduler step.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        geometry: MsaPlanCacheGeometry,
        max_batch: int,
        max_kv_indices: int,
    ):
        self.device = device
        self.geometry = geometry
        self.max_batch = int(max_batch)
        self.max_kv_indices = int(max_kv_indices)
        self.kv_indices_buf = torch.zeros(self.max_kv_indices, dtype=torch.int32, device=device)
        self.kv_page_indptr_buf = torch.zeros(self.max_batch + 1, dtype=torch.int32, device=device)
        # Populated on each build_from_metadata call.
        self.kv_indices: Optional[torch.Tensor] = None
        self.kv_page_indptr: Optional[torch.Tensor] = None

    def build_from_metadata(
        self,
        *,
        req_to_token: torch.Tensor,
        slot_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        page_size: int,
    ) -> None:
        """Refresh ``kv_indices`` / ``kv_page_indptr`` for one step.

        Runs entirely outside CUDA graph capture (called from
        :meth:`MiniMaxM3AttentionMetadata.prepare`); the captured
        forward reads the persistent buffers on every replay.
        """
        kv_indices, kv_page_indptr = build_stable_kv_indices(
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            page_size=page_size,
            dst=self.kv_indices_buf,
            kv_page_indptr_dst=self.kv_page_indptr_buf,
        )
        self.kv_indices = kv_indices
        self.kv_page_indptr = kv_page_indptr


__all__ = [
    "MsaPlanCache",
    "MsaPlanCacheGeometry",
    "build_stable_kv_indices",
    "get_global_msa_geometry",
    "set_global_msa_geometry",
]
