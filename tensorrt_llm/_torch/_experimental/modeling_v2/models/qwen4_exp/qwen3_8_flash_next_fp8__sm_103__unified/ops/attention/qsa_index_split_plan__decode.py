# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which requests each TP rank scores in a decode step's split QSA index selection, by work.

A tep decode step splits its QSA index scoring and block top-k by request over the TP group
(`_decode_index_selection`). A split by position would put the CUDA graph padding (always at the
tail) on one rank, so one launch assigns the requests by work, on the device (a graph replay's
batch is not known to the host at capture):

- each request's cost is its first row's visible-block count;
- the requests in descending cost (ties to the lower request index) are dealt to the ranks in
  snake order (0, 1, .., t - 1, t - 1, .., 0, ...): every rank gets exactly `n / t` requests, so
  the kernels' shapes stay the position split's;
- in the permuted order rank `r`'s requests are positions `[r * n / t, (r + 1) * n / t)`: the
  request at sorted position `s = lap * t + seat` goes to permuted position
  `owner * n / t + local(lap)`, where `local` interleaves the laps from both ends
  (0, last, 1, last - 1, ...) so long and short requests alternate in a rank's share.

Outputs, for `rows = n * p` rows of `p` a request:

- `iq_rows` `[rows]` int32: each row's position in the permuted order (where the attention
  prologue stores the row's index query, so each rank's queries are one contiguous slice);
- `perm_rows` `[rows]` int32: the row at each permuted position;
- this rank's rows of the four QSA row vectors (request index, logical position, sequence length,
  visible blocks) and the top-k row starts, in permuted order, `[rows / t]` each, fresh.

Every rank computes the same permutation from the same replicated metadata, so the gathered
selection, scattered back by `perm_rows`, is bitwise the position split's.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Requests compared at once in the ranking (a `[BLOCK, CHUNK]` tile).
_CHUNK = 32


class SplitPlan(NamedTuple):
    """`index_split_plan`'s outputs (see the module docstring)."""

    iq_rows: torch.Tensor
    perm_rows: torch.Tensor
    req_idx: torch.Tensor
    logical: torch.Tensor
    seq_lens: torch.Tensor
    visible: torch.Tensor
    row_starts: torch.Tensor


@triton.jit(do_not_specialize=["num_requests", "rows_per_request", "share", "rank"])
def _split_plan_kernel(
    visible,
    req_idx,
    logical,
    seq_lens,
    row_starts,
    iq_rows,
    perm_rows,
    o_req,
    o_logical,
    o_seq,
    o_visible,
    o_starts,
    num_requests,
    rows_per_request,
    share,
    rank,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    live = i < num_requests
    first = i.to(tl.int64) * rows_per_request
    cost = tl.load(visible + first, mask=live, other=0)
    # Stable descending rank: requests strictly costlier, or as costly with a lower index.
    pos = tl.zeros([BLOCK], dtype=tl.int32)
    for j0 in range(0, num_requests, CHUNK):
        j = j0 + tl.arange(0, CHUNK)
        cj = tl.load(visible + j.to(tl.int64) * rows_per_request, mask=j < num_requests, other=0)
        ahead = (cj[None, :] > cost[:, None]) | (
            (cj[None, :] == cost[:, None]) & (j[None, :] < i[:, None])
        )
        ahead = ahead & (j[None, :] < num_requests)
        pos += tl.sum(ahead.to(tl.int32), axis=1)
    lap = pos // WORLD
    seat = pos % WORLD
    owner = tl.where(lap % 2 == 0, seat, WORLD - 1 - seat)
    half = (share + 1) // 2
    local_lap = tl.where(lap < half, 2 * lap, 2 * (share - 1 - lap) + 1)
    slot = owner * share + local_lap
    mine = live & (owner == rank)
    for t in range(0, rows_per_request):
        g = first + t
        k = slot.to(tl.int64) * rows_per_request + t
        tl.store(iq_rows + g, k.to(tl.int32), mask=live)
        tl.store(perm_rows + k, g.to(tl.int32), mask=live)
        local = local_lap.to(tl.int64) * rows_per_request + t
        tl.store(o_req + local, tl.load(req_idx + g, mask=mine), mask=mine)
        tl.store(o_logical + local, tl.load(logical + g, mask=mine), mask=mine)
        tl.store(o_seq + local, tl.load(seq_lens + g, mask=mine), mask=mine)
        tl.store(o_visible + local, tl.load(visible + g, mask=mine), mask=mine)
        tl.store(o_starts + local, tl.load(row_starts + g, mask=mine), mask=mine)


def _spec(requests: int, p: int, world: int, rank: int, kind: str) -> dict:
    return dict(requests=requests, rows_per_request=p, world=world, rank=rank, kind=kind)


class IndexSplitPlan(OpWrapper):
    """`plan = index_split_plan(visible, req_idx, logical, seq_lens, row_starts, num_requests,
    rows_per_request, world, rank)`: the work-balanced request assignment of a decode step's split
    index selection (module docstring), a `SplitPlan`.

    - `visible`, `req_idx`, `seq_lens`, `row_starts`: int32, `logical`: int64, each a contiguous
      `[>= rows]` QSA row vector (`rows = num_requests * rows_per_request`, `p` rows a request,
      consecutive);
    - `num_requests % world == 0`, `num_requests <= 1024`.

    Bitwise `reference`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="tep2 HT verify step: 64 requests x 3 rows, rank 1, 3 padding requests at the tail",
             spec=_spec(64, 3, 2, 1, "padded")),
        Cell(why="the largest batch (192), ties everywhere (equal windows)", spec=_spec(192, 3, 2, 1, "equal")),
        Cell(why="tep4: 128 requests over 4 ranks, rank 2", spec=_spec(128, 3, 4, 2, "served")),
    )  # fmt: skip

    note = """
    Source: one Triton program (`_split_plan_kernel`, 4 warps), `BLOCK` the next power of two of
    the request count (a compile per block size and world); counts are runtime arguments. The
    ranking compares every pair of requests in `[BLOCK, CHUNK]` tiles.
    """

    def __call__(
        self,
        visible,
        req_idx,
        logical,
        seq_lens,
        row_starts,
        num_requests: int,
        rows_per_request: int,
        world: int,
        rank: int,
    ) -> SplitPlan:
        rows = num_requests * rows_per_request
        local = rows // world
        dev = visible.device
        plan = SplitPlan(
            iq_rows=torch.empty([rows], dtype=torch.int32, device=dev),
            perm_rows=torch.empty([rows], dtype=torch.int32, device=dev),
            req_idx=torch.empty([local], dtype=req_idx.dtype, device=dev),
            logical=torch.empty([local], dtype=logical.dtype, device=dev),
            seq_lens=torch.empty([local], dtype=seq_lens.dtype, device=dev),
            visible=torch.empty([local], dtype=visible.dtype, device=dev),
            row_starts=torch.empty([local], dtype=row_starts.dtype, device=dev),
        )
        if num_requests:
            block = max(_CHUNK, triton.next_power_of_2(num_requests))
            _split_plan_kernel[(1,)](
                visible, req_idx, logical, seq_lens, row_starts, *plan,
                num_requests, rows_per_request, num_requests // world, rank,
                WORLD=world, BLOCK=block, CHUNK=_CHUNK, num_warps=4,
            )  # fmt: skip
        return plan

    def reference(
        self,
        visible,
        req_idx,
        logical,
        seq_lens,
        row_starts,
        num_requests: int,
        rows_per_request: int,
        world: int,
        rank: int,
    ) -> SplitPlan:
        p, n, t = rows_per_request, num_requests, world
        share = n // t
        cost = [int(c) for c in visible[: n * p : p].tolist()]
        order = sorted(range(n), key=lambda r: (-cost[r], r))
        slot = [0] * n
        half = (share + 1) // 2
        for s, r in enumerate(order):
            lap, seat = divmod(s, t)
            owner = seat if lap % 2 == 0 else t - 1 - seat
            local = 2 * lap if lap < half else 2 * (share - 1 - lap) + 1
            slot[r] = owner * share + local
        dev = visible.device
        iq = torch.tensor(
            [slot[g // p] * p + g % p for g in range(n * p)], dtype=torch.int32, device=dev
        )
        perm = torch.empty_like(iq)
        perm[iq.long()] = torch.arange(n * p, dtype=torch.int32, device=dev)
        mine = perm[rank * share * p : (rank + 1) * share * p].long()
        return SplitPlan(
            iq_rows=iq,
            perm_rows=perm,
            req_idx=req_idx.index_select(0, mine),
            logical=logical.index_select(0, mine),
            seq_lens=seq_lens.index_select(0, mine),
            visible=visible.index_select(0, mine),
            row_starts=row_starts.index_select(0, mine),
        )

    def is_valid(
        self,
        visible,
        req_idx,
        logical,
        seq_lens,
        row_starts,
        num_requests: int,
        rows_per_request: int,
        world: int,
        rank: int,
    ) -> None:
        rows = num_requests * rows_per_request
        assert world >= 1 and 0 <= rank < world and num_requests % world == 0
        assert num_requests <= 1024 and rows_per_request >= 1
        for v in (visible, req_idx, logical, seq_lens, row_starts):
            assert v.dim() == 1 and v.is_contiguous() and v.numel() >= rows


index_split_plan = IndexSplitPlan()
