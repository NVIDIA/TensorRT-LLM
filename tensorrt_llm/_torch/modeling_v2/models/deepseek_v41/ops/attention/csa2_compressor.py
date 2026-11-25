# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine's decode compressor of an index-owner layer, in the engine's compressed-row layout.

Two entries, after `attention/backends/sparse/csa2/compressor.py` (`CSA2Compressor.project`) and the metadata's
owner-slot refresh (`CSA2TrtllmMetadata._resolve_owner_slots`):

- `csa2_owner_slots`: once per generation step, for every KV owner in one launch (the engine's
  `kernel.refresh_owner_slots`): the compressed groups the step completes per request (`ends // ratio -
  starts // ratio`, request-major), each group's flat GLOBAL write slot (-1 past the step's groups or unmapped) and
  first position, and for a ratio-2 owner its compression batch (request starts, ends, cumulative group counts).
- `csa2_compressor_project`: the owner's `(latent, index_k)` of those groups from the step's projected rows `src`
  (`[rows, 1024]` value | gate at ratio 2, `[rows, 512]` at ratio 1), as `CSA2Compressor.project` dispatches:
  a step of at most `COMPRESS_DECODE_MAX_ROWS` rows runs the engine's one-launch `_compress_project_kernel` (state
  update, pooling, RMSNorm, index-K projection); a larger one the engine's `compressor_prefill_reduction` (ratio 2),
  RMSNorm and the index-K GEMM. Row `g < groups` of the outputs is group `g`; later rows are zero.

The target publishes row `g` at `slots[g]` with the RoPE of `first[g]` (`csa2_store_rows`).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm

COMPRESS_DECODE_MAX_ROWS = 64  # the engine's: steps up to this many rows compress in one launch
_STATE_ROWS = 128  # rows per page of the compressor-state pools
_N_BLOCK = 16  # index-K outputs per program of the one-launch kernel

# An owner of `csa2_owner_slots`: (GLOBAL page table [requests, pages] int32, slots int64 [cap] (out), first int32
# [cap] (out), ratio, rows per GLOBAL page, and at ratio 2 the batch starts / ends int32 [requests] and cumulative
# group counts int32 [requests + 1] (out), else None x 3).
Owner = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


def owner_slot_descriptors(
    lengths: torch.Tensor, base: torch.Tensor, owners: Sequence[Owner]
) -> torch.Tensor:
    """The int64 `[owners, width]` descriptor table of `csa2_owner_slots` (device pointers of the owners' tables and
    buffers; one host-to-device copy: build it outside graph capture and keep it with the buffers)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import owner_slot_descriptor

    rows = [owner_slot_descriptor(base, lengths, *o[:3], o[3], o[4], *o[5:]) for o in owners]
    return torch.tensor(rows, dtype=torch.int64, device=lengths.device)


class Csa2OwnerSlots(OpWrapper):
    """In place, for every owner of `owners` (`Owner`): the step's compressed groups of each request with
    `starts = ends - lengths` (int32 `[requests]`, the step's rows per request; `base` int32 zeros): `slots[g]` the
    flat GLOBAL row of group `g` (request-major, -1 past the step's groups or on an unmapped page), `first[g]` its
    first position (`group * ratio`, 0 past them) and, at ratio 2, the batch starts / ends / cumulative counts.
    `descriptors` is `owner_slot_descriptors(lengths, base, owners)`. One launch."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8: 40 requests of four rows, the four owners (three ratio 2, one ratio 1)",
            spec=dict(requests=40, rows=4),
        ),
        Cell(why="dep8 production: 320 requests of four rows", spec=dict(requests=320, rows=4)),
        Cell(
            why="dep32: 25 requests of six rows, odd and even starts",
            spec=dict(requests=25, rows=6),
        ),
        Cell(
            why="an eager generation step of mixed lengths: 25 requests of 1 to 6 rows",
            spec=dict(requests=25, rows=(6, 1, 6, 4, 2, 6, 3, 5)),
        ),
    )
    note = "Index arithmetic: every output is compared bit for bit with the engine's PyTorch form."

    def __call__(
        self,
        ends: torch.Tensor,
        lengths: torch.Tensor,
        base: torch.Tensor,
        owners: Sequence[Owner],
        descriptors: torch.Tensor,
    ) -> None:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import refresh_owner_slots

        refresh_owner_slots(
            ends,
            lengths,
            base,
            descriptors,
            int(ends.numel()),
            max(int(o[1].numel()) for o in owners),
        )

    def reference(
        self,
        ends: torch.Tensor,
        lengths: torch.Tensor,
        base: torch.Tensor,
        owners: Sequence[Owner],
        descriptors: torch.Tensor,
    ) -> None:
        """The engine's own PyTorch form (`CSA2TrtllmMetadata._refresh_owner_slots`), owner by owner."""
        from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

        delta = ends - lengths - base
        for pages, slots, first, ratio, page_size, starts, lens, cu in owners:
            CSA2TrtllmMetadata._refresh_owner_slots(
                base, lengths, delta, ratio, page_size, pages, slots, first, starts, lens, cu
            )

    def is_valid(
        self,
        ends: torch.Tensor,
        lengths: torch.Tensor,
        base: torch.Tensor,
        owners: Sequence[Owner],
        descriptors: torch.Tensor,
    ) -> None:
        n = ends.numel()
        assert (
            ends.dtype == lengths.dtype == base.dtype == torch.int32
            and lengths.numel() == base.numel() == n
        )
        assert descriptors.dtype == torch.int64 and descriptors.shape[0] == len(owners)
        for pages, slots, first, ratio, page_size, starts, lens, cu in owners:
            assert ratio in (1, 2) and page_size == _STATE_ROWS // ratio and pages.shape[0] >= n
            assert (
                slots.dtype == torch.int64
                and first.dtype == torch.int32
                and slots.shape == first.shape
            )
            assert (starts is None) == (ratio == 1), "a ratio-2 owner carries its compression batch"


def _batch_rows(src, starts, ends, cu_seq, ratio):
    """Python form of the step's groups: [(request, group, rows of the step)]."""
    out = []
    for r in range(starts.numel()):
        s, e, a = int(starts[r]), int(ends[r]), int(cu_seq[r])
        for g in range(s // ratio, e // ratio):
            out.append((r, g, s, a))
    return out


class Csa2CompressorProject(OpWrapper):
    """`(latent bf16 [rows, 512], index_k bf16 [rows, n_index])` of the step's compressed groups (row `g` = group `g`
    of `csa2_owner_slots`, request-major; rows past them zero), and at ratio 2 every row of the step written to the
    fp32 state pools (in place), as the engine's `CSA2Compressor.project`.

    `src` is the step's projected rows (bf16; ratio 2 `[rows, 1024]` value then gate, ratio 1 `[rows, 512]`),
    `ends` / `starts` int32 `[requests]` (the batch of `csa2_owner_slots`), `cu_seq` int32 `[requests + 1]` the
    step's row offsets, `cu_comp` int32 `[requests + 1]` the cumulative group counts, `max_outputs` the most groups a
    request completes (a host bound, `ceil(rows per request / ratio)`). The state pools / tables are the
    compressor-state pools of the owner (`128` rows per page)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 c7000 ratio 2: 40 requests of four rows (the reduction path)",
            spec=dict(ratio=2, requests=40, rows=4),
        ),
        Cell(
            why="dep8 ratio 1 (layer 20): 40 requests of four rows",
            spec=dict(ratio=1, requests=40, rows=4),
        ),
        Cell(
            why="dep32 ratio 2: 25 requests of six rows, odd starts",
            spec=dict(ratio=2, requests=25, rows=6),
        ),
        Cell(
            why="16 requests of four rows: the one-launch kernel, ratio 2",
            spec=dict(ratio=2, requests=16, rows=4),
        ),
        Cell(
            why="8 requests of six rows: the one-launch kernel, ratio 1",
            spec=dict(ratio=1, requests=8, rows=6),
        ),
        Cell(
            why="dep8 production: 320 requests of four rows, ratio 2",
            spec=dict(ratio=2, requests=320, rows=4),
        ),
        Cell(
            why="an eager generation step of mixed lengths: 25 requests of 1 to 6 rows, ratio 2",
            spec=dict(ratio=2, requests=25, rows=(6, 1, 6, 4, 2, 6, 3, 5)),
        ),
        Cell(
            why="an eager generation step of mixed lengths: 40 requests of 1 to 4 rows, ratio 1",
            spec=dict(ratio=1, requests=40, rows=(4, 1, 4, 3, 2)),
        ),
    )
    note = """
    Latents and index keys are bf16 after an fp32 pooling / norm / projection: gated in bf16 ulps against the
    plain-torch reference; the state pools agree exactly.
    """

    def __call__(
        self,
        src: torch.Tensor,
        state_kv: Optional[torch.Tensor],
        state_score: Optional[torch.Tensor],
        table_kv: Optional[torch.Tensor],
        table_score: Optional[torch.Tensor],
        ends: torch.Tensor,
        starts: torch.Tensor,
        cu_seq: torch.Tensor,
        cu_comp: torch.Tensor,
        max_outputs: int,
        norm_w: torch.Tensor,
        eps: float,
        wk: torch.Tensor,
        ratio: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, d = src.shape[0], norm_w.shape[0]
        if _fuses(src, state_kv, state_score, table_kv, table_score, ends, norm_w, wk, ratio):
            return _compress_project(
                src,
                state_kv,
                state_score,
                table_kv,
                table_score,
                ends,
                starts,
                cu_seq,
                cu_comp,
                norm_w,
                eps,
                wk,
                ratio,
            )
        if ratio == 1:
            latent = src
        else:
            # rows without a completed window read as zero, as the engine's compressor
            latent = torch.zeros(rows, d, dtype=src.dtype, device=src.device)
            torch.ops.trtllm.compressor_prefill_reduction(
                src, torch.zeros(ratio, d, dtype=torch.float32, device=src.device), state_kv, state_score,
                table_kv, table_score, latent, ends, starts, cu_seq, cu_comp, int(ends.numel()), _STATE_ROWS, d,
                ratio, max_outputs,
            )  # fmt: skip
        latent = flashinfer_rmsnorm(latent, norm_w, eps)
        return latent, F.linear(
            latent, wk
        )  # the engine's `bf16_projection(index_wk, latent)`: `F.linear`

    def reference(
        self,
        src: torch.Tensor,
        state_kv: Optional[torch.Tensor],
        state_score: Optional[torch.Tensor],
        table_kv: Optional[torch.Tensor],
        table_score: Optional[torch.Tensor],
        ends: torch.Tensor,
        starts: torch.Tensor,
        cu_seq: torch.Tensor,
        cu_comp: torch.Tensor,
        max_outputs: int,
        norm_w: torch.Tensor,
        eps: float,
        wk: torch.Tensor,
        ratio: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group by group in fp32: the pair's softmax-weighted value (a position before the request's start read from
        the state pools) rounded to bf16, RMSNorm, projection; then every row of the step into the state pools."""
        rows, d = src.shape[0], norm_w.shape[0]
        x = src.float()
        latent = torch.zeros(rows, d, dtype=torch.float32, device=src.device)

        def state(r, p, pool, table):
            page = int(table[r, p // _STATE_ROWS])
            return pool[page, p % _STATE_ROWS].float()

        for out, (r, g, s, a) in enumerate(_batch_rows(src, starts, ends, cu_seq, ratio)):
            if ratio == 1:
                latent[out] = x[a + g - s]
                continue
            kv, sc = [], []
            for p in (2 * g, 2 * g + 1):
                if p >= s:
                    kv.append(x[a + p - s, :d])
                    sc.append(x[a + p - s, d:])
                else:
                    kv.append(state(r, p, state_kv, table_kv))
                    sc.append(state(r, p, state_score, table_score))
            w = torch.softmax(torch.stack(sc), 0)
            latent[out] = (w * torch.stack(kv)).sum(0)
        if ratio == 2:
            for r in range(starts.numel()):
                s, e, a = int(starts[r]), int(ends[r]), int(cu_seq[r])
                for p in range(s, e):
                    for pool, table, cols in (
                        (state_kv, table_kv, slice(0, d)),
                        (state_score, table_score, slice(d, 2 * d)),
                    ):
                        pool[int(table[r, p // _STATE_ROWS]), p % _STATE_ROWS] = x[a + p - s, cols]
        v = latent.to(torch.bfloat16).float()
        normalized = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * norm_w.float()).to(
            torch.bfloat16
        )
        return normalized, (normalized.float() @ wk.float().t()).to(torch.bfloat16)

    def is_valid(
        self,
        src: torch.Tensor,
        state_kv: Optional[torch.Tensor],
        state_score: Optional[torch.Tensor],
        table_kv: Optional[torch.Tensor],
        table_score: Optional[torch.Tensor],
        ends: torch.Tensor,
        starts: torch.Tensor,
        cu_seq: torch.Tensor,
        cu_comp: torch.Tensor,
        max_outputs: int,
        norm_w: torch.Tensor,
        eps: float,
        wk: torch.Tensor,
        ratio: int,
    ) -> None:
        d = norm_w.shape[0]
        assert (
            ratio in (1, 2)
            and d == 512
            and src.dtype == torch.bfloat16
            and src.shape[1] == ratio * d
        )
        assert (
            src.is_contiguous() and wk.dtype == norm_w.dtype == torch.bfloat16 and wk.shape[1] == d
        )
        n = ends.numel()
        assert ends.dtype == starts.dtype == cu_seq.dtype == cu_comp.dtype == torch.int32
        assert (
            starts.numel() == n and cu_seq.numel() == cu_comp.numel() == n + 1 and max_outputs >= 1
        )
        if ratio == 2:
            assert state_kv.dtype == state_score.dtype == torch.float32 and state_kv.shape[1:] == (
                _STATE_ROWS,
                d,
            )
            assert (
                table_kv.dtype == table_score.dtype == torch.int32
                and table_kv.shape == table_score.shape
            )


def _fuses(src, state_kv, state_score, table_kv, table_score, ends, norm_w, wk, ratio) -> bool:
    """`CSA2Compressor._fuses`: one launch for decode-sized steps."""
    rows = src.shape[0]
    if not (
        src.is_cuda
        and 0 < rows <= COMPRESS_DECODE_MAX_ROWS
        and wk.dtype == norm_w.dtype == torch.bfloat16
        and wk.is_contiguous()
        and wk.shape[1] == norm_w.shape[0]
        and wk.shape[0] % _N_BLOCK == 0
        and norm_w.is_contiguous()
    ):
        return False
    if ratio == 1:
        return True
    tables = (table_kv, table_score)
    return (
        state_kv.dtype == state_score.dtype == torch.float32
        and state_kv.is_contiguous()
        and state_score.is_contiguous()
        and all(t.dtype == torch.int32 and t.dim() == 2 and t.is_contiguous() for t in tables)
        and table_kv.shape == table_score.shape
        and ends.shape[0] < 1024
    )


def _compress_project(
    src,
    state_kv,
    state_score,
    table_kv,
    table_score,
    ends,
    starts,
    cu_seq,
    cu_comp,
    norm_w,
    eps,
    wk,
    ratio,
):
    """`CSA2Compressor._compress_project`: the engine's `_compress_project_kernel`, one launch."""
    import triton

    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import (
        _compress_project_kernel,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import _pdl

    rows, d = src.shape[0], norm_w.shape[0]
    latent = torch.empty(rows, d, dtype=src.dtype, device=src.device)
    index_k = torch.empty(rows, wk.shape[0], dtype=src.dtype, device=src.device)
    if ratio == 1:
        paged, batch, max_blocks, page_size = (src,) * 8, 0, 0, 1
    else:
        paged = (state_kv, state_score, table_kv, table_score, ends, starts, cu_seq, cu_comp)
        batch, max_blocks, page_size = int(ends.shape[0]), table_kv.shape[1], _STATE_ROWS
    pdl = _pdl()
    _compress_project_kernel[(rows, wk.shape[0] // _N_BLOCK)](
        src, src.stride(0), *paged, batch, max_blocks, page_size, norm_w, eps, wk, latent, index_k, d, wk.shape[0],
        ratio, triton.next_power_of_2(batch + 1), _N_BLOCK, pdl,
        num_warps=4, enable_fp_fusion=False, launch_pdl=pdl,
    )  # fmt: skip
    return latent, index_k


csa2_owner_slots = Csa2OwnerSlots()
csa2_compressor_project = Csa2CompressorProject()
