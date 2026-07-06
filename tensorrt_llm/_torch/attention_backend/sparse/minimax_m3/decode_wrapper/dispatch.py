# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe decode driver for MiniMax-M3 sparse attention.

Replaces MSA's host-centric ``fmha_sm100_plan`` / ``fmha_sm100`` driver
(``api.py``) for the decode path while invoking the *same* JIT-compiled
SM100 kernel binaries via ``fmha_sm100.jit.get_fmha_variant``: the MSA
plan bakes host-side values into the launch, which CUDA graph replays
would freeze, so only the driver is replaced and the kernels are
reused.

Design contract:

* No plan/run split — every call assembles launch args directly.
* Everything per-step-varying is a device tensor: ``seq_lens``,
  ``kv_page_indptr``, ``kv_indices``, ``kv_block_indexes``, the
  ``max_score`` contents.
* Host-baked values are geometry / per-batch-size constants only:
  head counts, pack factor, page size, ``max_k_tiles`` capacity,
  worklists (a pure function of batch size for decode).
* Every method is callable inside a CUDA graph capture and yields
  correct results at replay: no ``.item()`` / ``.cpu()`` / ``.tolist()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .topk import select_topk_blocks
from .worklist import build_decode_worklist

# Mirrors fmha_sm100.jit._PACK_FACTORS.
_PACK_FACTORS = (1, 2, 4, 6, 8, 16)
_QO_TILE_SIZE = 128
_WORKSPACE_BYTES = 32 * 1024 * 1024


def _compute_pack_factor(max_qo_len: int, num_qo_heads: int, num_kv_heads: int) -> int:
    """Verbatim port of ``fmha_sm100.api._compute_pack_factor``."""
    if num_kv_heads == -1:
        return 1
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in reversed(_PACK_FACTORS):
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1


def _max_k_tiles_capacity(max_kv_len: int) -> int:
    """MSA's ``max_k_tiles`` formula (``api.py:658``) at max capacity.

    Number of 128-token KV blocks rounded up to a multiple of 128 (the
    kernel's max-score tile stride granularity).
    """
    return math.ceil(math.ceil(max_kv_len / 128) / 128) * 128


@dataclass(frozen=True)
class M3DecodeGeometry:
    """Compile/alloc-time constants for one M3 layer family (per rank)."""

    num_q_heads: int
    num_kv_heads: int
    num_index_heads: int
    head_dim: int
    page_size: int
    topk: int
    init_blocks: int
    local_blocks: int
    max_batch: int
    max_kv_len: int

    def __post_init__(self):
        if self.head_dim != 128 or self.page_size != 128:
            raise NotImplementedError(
                "MSA SM100 decode kernels require head_dim=128 and page_size=128; "
                f"got head_dim={self.head_dim}, page_size={self.page_size}."
            )
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.num_index_heads % self.num_kv_heads != 0:
            raise ValueError("num_index_heads must be divisible by num_kv_heads")


class M3DecodeKernelDriver:
    """Persistent-state decode driver for one (device, geometry) pair.

    Allocates every buffer once at construction with max-capacity
    geometry; per-call views are prefixes/strided views of those
    buffers so their ``data_ptr()`` is stable across CUDA graph
    replays.
    """

    def __init__(self, geometry: M3DecodeGeometry, device: torch.device):
        import fmha_sm100  # noqa: F401 — hard dependency of this driver
        from fmha_sm100.jit import _dlpack_dtype_code, get_fmha_variant

        self.geom = g = geometry
        self.device = device

        # --- pack factors and packed head counts (decode: qo_len == 1) ---
        self.pf_proxy = _compute_pack_factor(1, g.num_index_heads, 1)
        self.heads_packed_proxy = g.num_index_heads // self.pf_proxy
        self.pf_sparse = _compute_pack_factor(1, g.num_q_heads, g.num_kv_heads)
        self.heads_packed_sparse = g.num_q_heads // self.pf_sparse

        self.max_k_tiles = _max_k_tiles_capacity(g.max_kv_len)
        self.num_ctas = int(torch.cuda.get_device_properties(device).multi_processor_count)

        # --- kernel variant modules (JIT-compiled once, same binaries the
        #     MSA api path runs) -----------------------------------------
        bf16_code = _dlpack_dtype_code(torch.bfloat16)
        # Proxy: OnlyScore (sparse_mode=2), single_wg, no split.
        self._proxy_module = get_fmha_variant(
            bf16_code, _QO_TILE_SIZE, True, 2, g.page_size, False, self.pf_proxy
        )
        # Sparse GQA: Sparse (sparse_mode=0).
        self._sparse_module = get_fmha_variant(
            bf16_code, _QO_TILE_SIZE, True, 0, g.page_size, False, self.pf_sparse
        )

        # --- persistent buffers -----------------------------------------
        self.workspace_buffer = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self._max_score_flat = torch.empty(
            g.num_index_heads * self.max_k_tiles * g.max_batch,
            dtype=torch.float32,
            device=device,
        )
        self._kv_block_indexes = torch.full(
            (g.max_batch, g.num_kv_heads, g.topk), -1, dtype=torch.int32, device=device
        )
        self._out = torch.empty(
            g.max_batch, g.num_q_heads, g.head_dim, dtype=torch.bfloat16, device=device
        )
        self._kv_segment_offsets = torch.zeros(g.max_batch + 1, dtype=torch.int32, device=device)
        self._valid_pages = torch.zeros(g.max_batch, dtype=torch.int32, device=device)
        self._qo_offset = torch.zeros(g.max_batch, dtype=torch.int32, device=device)

        # Per-batch-size constants, built lazily and cached (bounded by
        # the distinct batch sizes seen: CUDA graph buckets + eager).
        self._qo_const_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._worklist_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Cached per-batch-size constants (host work happens once per shape,
    # outside any capture — callers warm shapes up before capturing).
    # ------------------------------------------------------------------

    def _qo_consts(self, batch: int, pack_factor: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(qo_segment_lens, qo_segment_offsets) for packed decode lens."""
        key = (batch, pack_factor)
        cached = self._qo_const_cache.get(key)
        if cached is None:
            lens = torch.full((batch,), pack_factor, dtype=torch.int32, device=self.device)
            offsets = (
                (torch.arange(batch + 1, dtype=torch.int64) * pack_factor)
                .to(torch.int32)
                .to(self.device)
            )
            cached = (lens, offsets)
            self._qo_const_cache[key] = cached
        return cached

    def _worklist(self, batch: int, num_packed_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (batch, num_packed_heads)
        cached = self._worklist_cache.get(key)
        if cached is None:
            cached = build_decode_worklist(
                batch_size=batch,
                num_packed_heads=num_packed_heads,
                num_ctas=self.num_ctas,
                device=self.device,
            )
            self._worklist_cache[key] = cached
        return cached

    def warmup_shapes(self, batch: int) -> None:
        """Pre-build all per-batch-size constants for ``batch``.

        Call once per CUDA graph bucket before capture so no host-side
        cache misses happen inside the captured region.
        """
        self._qo_consts(batch, self.pf_proxy)
        self._qo_consts(batch, self.pf_sparse)
        self._worklist(batch, self.heads_packed_proxy)
        self._worklist(batch, self.heads_packed_sparse)

    # ------------------------------------------------------------------
    # Shared per-call device-side metadata refresh
    # ------------------------------------------------------------------

    def _kv_offsets_view(self, seq_lens: torch.Tensor, batch: int) -> torch.Tensor:
        """Cumulative KV lengths into the persistent buffer (device op)."""
        view = self._kv_segment_offsets[: batch + 1]
        torch.cumsum(seq_lens, 0, dtype=torch.int32, out=view[1:])
        return view

    def _qo_offset_view(self, seq_lens: torch.Tensor, batch: int) -> torch.Tensor:
        """Per-request causal offset ``kv_len - 1`` (device op).

        The kernel's causal bound is inclusive (attend positions
        ``<= offset + q_idx``); with one query token at position
        ``kv_len - 1`` this unmasks exactly the ``kv_len`` cached
        positions.  ``kv_len`` itself would leak one stale slot from a
        partially-filled last page in *sparse* mode, which has no
        secondary seqlen clip (verified empirically in
        ``test_minimax_m3_decode_driver_vs_msa.py`` hetero cases).
        """
        view = self._qo_offset[:batch]
        torch.sub(seq_lens, 1, out=view)
        return view

    # ------------------------------------------------------------------
    # Proxy MQA pass (indexer): per-KV-block max scores
    # ------------------------------------------------------------------

    def proxy_max_score(
        self,
        idx_q: torch.Tensor,
        idx_k_paged: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        """Dense MQA proxy pass, OnlyScore mode.

        Parameters
        ----------
        idx_q : ``[batch, num_index_heads, 128]`` bf16 (decode: 1 token/req).
        idx_k_paged : ``[num_pages, 1, page_size, 128]`` bf16 (HND).
        seq_lens : ``[batch]`` int32 device — per-request KV length.
        kv_page_indptr : ``[batch + 1]`` int32 device.
        kv_indices : ``[total_pages]`` int32 device page table.
        sm_scale : softmax scale (does not affect max-score ranking).

        Returns
        -------
        ``[num_index_heads, max_k_tiles, batch]`` fp32 view into the
        persistent max-score buffer; unwritten tiles are ``-inf``.
        """
        g = self.geom
        batch = idx_q.shape[0]
        seq_lens = seq_lens[:batch]

        max_score = torch.as_strided(
            self._max_score_flat,
            (g.num_index_heads, self.max_k_tiles, batch),
            (self.max_k_tiles * batch, batch, 1),
        )
        max_score.fill_(float("-inf"))

        qo_lens, qo_offsets = self._qo_consts(batch, self.pf_proxy)
        work_range, work_info = self._worklist(batch, self.heads_packed_proxy)
        kv_offsets = self._kv_offsets_view(seq_lens, batch)
        qo_offset = self._qo_offset_view(seq_lens, batch)

        self._proxy_module.run(
            self.workspace_buffer,
            idx_q,
            idx_k_paged,
            idx_k_paged,
            qo_lens,
            seq_lens,
            qo_offsets,
            kv_offsets,
            work_range,
            work_info,
            None,  # out (OnlyScore)
            float(sm_scale),
            1.0,
            1.0,
            1.0,
            1.0,
            self.pf_proxy,  # max_qo_len after packing
            qo_offset,  # kv_len - 1: inclusive causal bound = last cached pos
            1,  # num_kv_splits
            None,
            None,
            None,  # kv_tile_begin / end / split
            None,
            None,  # workspace_o / workspace_lse
            None,  # num_kv_splits_per_row
            _QO_TILE_SIZE,
            kv_indices,
            kv_page_indptr,
            max_score,
            self.max_k_tiles,
            None,  # kv_block_indexes
            self.pf_proxy,
            True,  # qo_len_uniform
            torch.cuda.current_stream().cuda_stream,
        )
        return max_score

    # ------------------------------------------------------------------
    # Top-k block selection (device-driven)
    # ------------------------------------------------------------------

    def select_blocks(
        self,
        max_score: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Group-reduce index-head scores to KV heads and pick top-k blocks.

        Returns ``[batch, num_kv_heads, topk]`` int32 view into the
        persistent ``kv_block_indexes`` buffer.
        """
        g = self.geom
        batch = max_score.shape[2]
        seq_lens = seq_lens[:batch]

        group = g.num_index_heads // g.num_kv_heads
        if group > 1:
            max_score_kv = max_score.view(g.num_kv_heads, group, max_score.shape[1], batch).amax(
                dim=1
            )
        else:
            max_score_kv = max_score

        valid_pages = self._valid_pages[:batch]
        torch.div(seq_lens + (g.page_size - 1), g.page_size, rounding_mode="floor", out=valid_pages)

        out = self._kv_block_indexes[:batch]
        return select_topk_blocks(
            max_score_kv,
            valid_pages,
            topk=g.topk,
            init_blocks=g.init_blocks,
            local_blocks=g.local_blocks,
            out=out,
        )

    # ------------------------------------------------------------------
    # Sparse block-GQA main pass
    # ------------------------------------------------------------------

    def sparse_attention(
        self,
        q: torch.Tensor,
        k_paged: torch.Tensor,
        v_paged: torch.Tensor,
        kv_block_indexes: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        """Block-sparse paged GQA over the selected KV blocks.

        Parameters
        ----------
        q : ``[batch, num_q_heads, 128]`` bf16.
        k_paged / v_paged : ``[num_pages, num_kv_heads, page_size, 128]`` bf16.
        kv_block_indexes : ``[batch, num_kv_heads, topk]`` int32 ascending,
            ``-1`` tail padded.
        seq_lens / kv_page_indptr / kv_indices : as in ``proxy_max_score``.

        Returns
        -------
        ``[batch, num_q_heads, 128]`` bf16 view into the persistent
        output buffer.
        """
        batch = q.shape[0]
        seq_lens = seq_lens[:batch]

        out = self._out[:batch]
        qo_lens, qo_offsets = self._qo_consts(batch, self.pf_sparse)
        work_range, work_info = self._worklist(batch, self.heads_packed_sparse)
        kv_offsets = self._kv_offsets_view(seq_lens, batch)
        qo_offset = self._qo_offset_view(seq_lens, batch)

        self._sparse_module.run(
            self.workspace_buffer,
            q,
            k_paged,
            v_paged,
            qo_lens,
            seq_lens,
            qo_offsets,
            kv_offsets,
            work_range,
            work_info,
            out,
            float(sm_scale),
            1.0,
            1.0,
            1.0,
            1.0,
            self.pf_sparse,  # max_qo_len after packing
            qo_offset,  # kv_len - 1 (see _qo_offset_view)
            1,  # num_kv_splits
            None,
            None,
            None,
            None,
            None,
            None,
            _QO_TILE_SIZE,
            kv_indices,
            kv_page_indptr,
            None,  # max_score
            -1,  # max_k_tiles
            kv_block_indexes,
            self.pf_sparse,
            True,  # qo_len_uniform
            torch.cuda.current_stream().cuda_stream,
        )
        return out


# ---------------------------------------------------------------------------
# Module-level convenience API (driver cache + functional entry points)
# ---------------------------------------------------------------------------

_driver_cache: Dict[Tuple, M3DecodeKernelDriver] = {}


def get_decode_driver(geometry: M3DecodeGeometry, device: torch.device) -> M3DecodeKernelDriver:
    key = (
        geometry,
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
    )
    driver = _driver_cache.get(key)
    if driver is None:
        driver = M3DecodeKernelDriver(geometry, device)
        _driver_cache[key] = driver
    return driver


def proxy_mqa_decode(
    idx_q: torch.Tensor,
    idx_k_paged: torch.Tensor,
    *,
    geometry: M3DecodeGeometry,
    seq_lens: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Functional proxy pass: returns per-KV-block max scores."""
    driver = get_decode_driver(geometry, idx_q.device)
    return driver.proxy_max_score(
        idx_q,
        idx_k_paged,
        seq_lens=seq_lens,
        kv_page_indptr=kv_page_indptr,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
    )


def sparse_gqa_decode(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    *,
    geometry: M3DecodeGeometry,
    seq_lens: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Functional sparse GQA pass over selected KV blocks."""
    driver = get_decode_driver(geometry, q.device)
    return driver.sparse_attention(
        q,
        k_paged,
        v_paged,
        kv_block_indexes,
        seq_lens=seq_lens,
        kv_page_indptr=kv_page_indptr,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
    )


__all__ = [
    "M3DecodeGeometry",
    "M3DecodeKernelDriver",
    "get_decode_driver",
    "proxy_mqa_decode",
    "sparse_gqa_decode",
]
