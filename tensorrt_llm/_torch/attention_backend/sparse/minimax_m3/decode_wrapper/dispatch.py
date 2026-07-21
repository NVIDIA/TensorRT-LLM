# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe decode planner for MiniMax-M3 sparse attention.

Replaces MSA's host-centric `fmha_sm100_plan` / `fmha_sm100` driver for
the decode path while invoking the same JIT-compiled SM100 kernel
binaries via `fmha_sm100.jit.get_fmha_variant`. MSA's plan bakes
host-side values into the launch, which CUDA graph replays would freeze,
so only the driver is replaced and the kernels are reused.

Ownership follows the FlashInfer backend convention: one
:class:`M3DecodePlanner` per attention-metadata instance (the CUDA graph
runner creates one metadata instance per captured batch size, so each
capture bucket automatically gets its own planner and buffers), with
`plan()` called from `metadata.prepare()` outside any capture window.

Buffer discipline:

* Persistent (cross the capture boundary; written by `plan()` on the
  host, read by the captured kernels): the persistent-CTA worklists and
  the packed qo segment constants.
* Everything else (`max_score`, selected block indexes, cumulative KV
  offsets, causal offsets, the output) is either derived from device
  tensors inside the forward or is a plain intermediate, so it is
  allocated normally and CUDA graph capture bakes the allocations.

Every kernel-facing method is callable inside a CUDA graph capture and
yields correct results at replay: no `.item()`, `.cpu()`, or
`.tolist()`.
"""

from __future__ import annotations

import functools
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
    """Verbatim port of `fmha_sm100.api._compute_pack_factor`."""
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
    """MSA's `max_k_tiles` formula at max capacity.

    Number of 128-token KV blocks rounded up to a multiple of 128 (the
    kernel's max-score tile stride granularity).
    """
    return math.ceil(math.ceil(max_kv_len / 128) / 128) * 128


@functools.lru_cache(maxsize=None)
def _get_variant_module(sparse_mode: int, page_size: int, pack_factor: int):
    """JIT-compiled `fmha_sm100` kernel variant (stateless, process-wide)."""
    import fmha_sm100  # noqa: F401  (hard dependency of this driver)
    from fmha_sm100.jit import _dlpack_dtype_code, get_fmha_variant

    bf16_code = _dlpack_dtype_code(torch.bfloat16)
    return get_fmha_variant(
        bf16_code, _QO_TILE_SIZE, True, sparse_mode, page_size, False, pack_factor
    )


_workspace_buffers: Dict[Tuple[str, int], torch.Tensor] = {}


def _get_workspace(device: torch.device) -> torch.Tensor:
    """Shared per-device kernel scratch (contentless, like FlashInfer's
    `workspace_buffer`; graphs replay serially on a stream so sharing one
    scratch across planners and capture buckets is safe)."""
    key = (device.type, device.index if device.index is not None else torch.cuda.current_device())
    buf = _workspace_buffers.get(key)
    if buf is None:
        buf = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _workspace_buffers[key] = buf
    return buf


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


class M3DecodePlanner:
    """Per-metadata-instance decode planner and kernel launcher.

    Owns exactly the buffers whose contents are produced on the host and
    consumed by captured kernels (worklists and packed qo constants);
    `plan()` refreshes them for the current batch size from
    `metadata.prepare()`, outside any capture window. The owning metadata
    instance is per capture bucket, so buckets never share these buffers.
    """

    def __init__(self, geometry: M3DecodeGeometry, device: torch.device):
        self.geom = g = geometry
        self.device = device

        # Pack factors and packed head counts (decode: qo_len == 1).
        self.pf_proxy = _compute_pack_factor(1, g.num_index_heads, 1)
        self.heads_packed_proxy = g.num_index_heads // self.pf_proxy
        self.pf_sparse = _compute_pack_factor(1, g.num_q_heads, g.num_kv_heads)
        self.heads_packed_sparse = g.num_q_heads // self.pf_sparse

        self.max_k_tiles = _max_k_tiles_capacity(g.max_kv_len)
        self.num_ctas = int(torch.cuda.get_device_properties(device).multi_processor_count)

        # Kernel variant modules (JIT-compiled once per process, same
        # binaries the MSA api path runs). Proxy: OnlyScore
        # (sparse_mode=2); sparse GQA: Sparse (sparse_mode=0).
        self._proxy_module = _get_variant_module(2, g.page_size, self.pf_proxy)
        self._sparse_module = _get_variant_module(0, g.page_size, self.pf_sparse)
        self._workspace = _get_workspace(device)

        # --- persistent host-planned buffers ------------------------------
        # `work_info[i] = head << 16 | batch` in batch-major order is a
        # pure function of the item index, so the batch-`b` worklist is a
        # prefix of the max-batch worklist: fill the info buffers once and
        # take prefix views. Only the CTA split (`work_range`) depends on
        # the batch size and is rewritten by `plan()`.
        max_range, max_info_proxy = build_decode_worklist(
            batch_size=g.max_batch,
            num_packed_heads=self.heads_packed_proxy,
            num_ctas=self.num_ctas,
            device=device,
        )
        _, max_info_sparse = build_decode_worklist(
            batch_size=g.max_batch,
            num_packed_heads=self.heads_packed_sparse,
            num_ctas=self.num_ctas,
            device=device,
        )
        self._work_info_proxy = max_info_proxy
        self._work_info_sparse = max_info_sparse
        self._work_range_proxy = max_range
        self._work_range_sparse = torch.empty_like(max_range)

        # Packed qo segment constants are batch-prefix-stable as well
        # (`lens[i] = pf`, `offsets[i] = i * pf`): fill once, view later.
        self._qo_lens_proxy = torch.full(
            (g.max_batch,), self.pf_proxy, dtype=torch.int32, device=device
        )
        self._qo_lens_sparse = torch.full(
            (g.max_batch,), self.pf_sparse, dtype=torch.int32, device=device
        )
        arange = torch.arange(g.max_batch + 1, dtype=torch.int64)
        self._qo_offsets_proxy = (arange * self.pf_proxy).to(torch.int32).to(device)
        self._qo_offsets_sparse = (arange * self.pf_sparse).to(torch.int32).to(device)

        self._planned_batch = -1
        self.plan(min(1, g.max_batch) or 1)

    # ------------------------------------------------------------------
    # Host-side planning (call from metadata.prepare(), outside capture)
    # ------------------------------------------------------------------

    def plan(self, batch: int) -> None:
        """Refresh the CTA work split for `batch` decode requests.

        Idempotent per batch size; during CUDA graph replay the bucket's
        batch never changes so this is a no-op after the pre-capture
        plan.
        """
        if batch == self._planned_batch:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "M3DecodePlanner.plan() called for a new batch size during CUDA "
                "graph capture; metadata.prepare() must plan before capture."
            )
        if not 0 < batch <= self.geom.max_batch:
            raise ValueError(f"batch {batch} out of range (max_batch={self.geom.max_batch})")

        def _cta_split(n_items: int) -> torch.Tensor:
            bounds = (torch.arange(self.num_ctas + 1, dtype=torch.int64) * n_items) // self.num_ctas
            return (bounds[1:] << 32) | bounds[:-1]

        self._work_range_proxy.copy_(_cta_split(batch * self.heads_packed_proxy), non_blocking=True)
        self._work_range_sparse.copy_(
            _cta_split(batch * self.heads_packed_sparse), non_blocking=True
        )
        self._planned_batch = batch

    def _require_planned(self, batch: int) -> None:
        if batch != self._planned_batch:
            raise RuntimeError(
                f"M3DecodePlanner: batch {batch} was not planned (planned="
                f"{self._planned_batch}); metadata.prepare() must call plan() first."
            )

    # ------------------------------------------------------------------
    # Shared per-call device-side metadata (intra-capture intermediates)
    # ------------------------------------------------------------------

    @staticmethod
    def _kv_offsets(seq_lens: torch.Tensor, batch: int) -> torch.Tensor:
        """Cumulative KV lengths (device op; capture-safe intermediate)."""
        out = torch.zeros(batch + 1, dtype=torch.int32, device=seq_lens.device)
        torch.cumsum(seq_lens, 0, dtype=torch.int32, out=out[1:])
        return out

    @staticmethod
    def _causal_qo_offset(seq_lens: torch.Tensor) -> torch.Tensor:
        """Per-request causal offset `kv_len - 1` (device op).

        The kernel's causal bound is inclusive (attend positions
        `<= offset + q_idx`); with one query token at position
        `kv_len - 1` this unmasks exactly the `kv_len` cached positions.
        `kv_len` itself would leak one stale slot from a partially-filled
        last page in sparse mode, which has no secondary seqlen clip.
        """
        return seq_lens - 1

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
        idx_q : `[batch, num_index_heads, 128]` bf16 (decode: 1 token/req).
        idx_k_paged : `[num_pages, 1, page_size, 128]` bf16 (HND).
        seq_lens : `[batch]` int32 device; per-request KV length.
        kv_page_indptr : `[batch + 1]` int32 device.
        kv_indices : `[total_pages]` int32 device page table.
        sm_scale : softmax scale (does not affect max-score ranking).

        Returns
        -------
        `[num_index_heads, max_k_tiles, batch]` fp32; unwritten tiles are
        -inf.
        """
        g = self.geom
        batch = idx_q.shape[0]
        self._require_planned(batch)
        seq_lens = seq_lens[:batch]

        max_score = torch.full(
            (g.num_index_heads, self.max_k_tiles, batch),
            float("-inf"),
            dtype=torch.float32,
            device=idx_q.device,
        )
        kv_offsets = self._kv_offsets(seq_lens, batch)
        qo_offset = self._causal_qo_offset(seq_lens)

        self._proxy_module.run(
            self._workspace,
            idx_q,
            idx_k_paged,
            idx_k_paged,
            self._qo_lens_proxy[:batch],
            seq_lens,
            self._qo_offsets_proxy[: batch + 1],
            kv_offsets,
            self._work_range_proxy,
            self._work_info_proxy[: batch * self.heads_packed_proxy],
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

        Returns `[batch, num_kv_heads, topk]` int32, ascending, -1 padded.
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

        valid_pages = torch.div(
            seq_lens + (g.page_size - 1), g.page_size, rounding_mode="floor"
        ).to(torch.int32)

        out = torch.full(
            (batch, g.num_kv_heads, g.topk), -1, dtype=torch.int32, device=max_score.device
        )
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
        q : `[batch, num_q_heads, 128]` bf16.
        k_paged / v_paged : `[num_pages, num_kv_heads, page_size, 128]` bf16.
        kv_block_indexes : `[batch, num_kv_heads, topk]` int32 ascending,
            -1 tail padded.
        seq_lens / kv_page_indptr / kv_indices : as in `proxy_max_score`.

        Returns
        -------
        `[batch, num_q_heads, 128]` bf16.
        """
        batch = q.shape[0]
        g = self.geom
        self._require_planned(batch)
        seq_lens = seq_lens[:batch]

        out = torch.empty(batch, g.num_q_heads, g.head_dim, dtype=torch.bfloat16, device=q.device)
        kv_offsets = self._kv_offsets(seq_lens, batch)
        qo_offset = self._causal_qo_offset(seq_lens)

        self._sparse_module.run(
            self._workspace,
            q,
            k_paged,
            v_paged,
            self._qo_lens_sparse[:batch],
            seq_lens,
            self._qo_offsets_sparse[: batch + 1],
            kv_offsets,
            self._work_range_sparse,
            self._work_info_sparse[: batch * self.heads_packed_sparse],
            out,
            float(sm_scale),
            1.0,
            1.0,
            1.0,
            1.0,
            self.pf_sparse,  # max_qo_len after packing
            qo_offset,  # kv_len - 1 (see _causal_qo_offset)
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


__all__ = [
    "M3DecodeGeometry",
    "M3DecodePlanner",
]
