# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-bank staging of a context step for the FP8 sparse context kernel, as stock's `stage_shared_rows`.

Stock's context attention (`CSA2TrtllmAttention.forward_fused_context`) reads one E4M3 bank `[rows, 512]`: row 0 is
zero, then per request the window rows its queries can read (`len + window - 1` positions from the first query's window
start), then per request every compressed row it can select (`0 .. domain - 1`). Each query reads a dense prefix of bank
indices: its valid window rows, then its valid selections, in slot
order. The engine's CuTe kernels build it (`csa2/kernel.py`):

* `SharedCompactKernel` maps every query's selections onto bank rows and marks the window rows read; layers that share
  the KV owner and the index source select the same rows, so only the first layer of such a group compacts;
* `SharedScaleKernel` + `derive_stage_kv_scales` derive one power-of-two staging scale per layer from the marked rows'
  group-scale bytes; the query is E4M3 at unit scale (stock's unit-scale context Q), so BMM1 carries the KV dequant
  scale alone and Q does not cap the range;
* `SharedDecodeKernel` decodes each marked window row and every compressed row once (the compressed rows are kept for
  the next layer of the group when its scale matches) and writes the kernel's BMM scales.

Three entries:

* `csa2_shared_plan`: a context step's bank layout per compression ratio (stock's `CSA2SharedKVPlan`, from the step's
  host request lengths), plus where the window rows live. A step stages its rows in a step-local paged copy (`local`,
  128-row pages, each request's rows contiguous from its first readable position) whose page table the plan holds;
  the bank reads it exactly as stock reads the pool;
* `csa2_local_window`: per layer, copies each request's history rows (the `window - 1` positions before its chunk) from
  the SWA pool into `local`, and every row of the chunk from `local` into its SWA slot of the pool (skipped where the
  page is unmapped), as stock writes each row's SWA slot: prefix reuse publishes whole pages;
* `csa2_shared_stage`: per layer, `stage_shared_rows` with the derived scales: the bank, each query's bank indices and
  count, the query's quantization scale and the kernel's BMM1 / BMM2 scales.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._utils import maybe_pin_memory

from .csa2_pack_rows import csa2_unpack_rows

SWA_ROW, MAIN_ROW, HEAD_DIM = 528, 288, 512
_PAGE = 128  # rows per page of the SWA pool and of the step-local window copy
# Stock's bound of the staging exponent for a context step whose Q is E4M3 at unit scale (`sparse_attn_predict` with
# `projected_q_unit_scale`: Q pays nothing for the KV range, so the range is not capped by Q).
CONTEXT_SCALE_MAX_EXPONENT = 126
_TMA_MIN_BYTES = (
    128 << 10
)  # the context kernel's TMA reads need this much backing from the bank's base (stock)


@dataclass
class Csa2SharedPlan:
    """A context step's shared-bank layout at one compression ratio.

    The first fields are the ones `stage_shared_rows` reads (stock's `CSA2SharedKVPlan`); `main_pages` is the owner's
    page table, set per layer. Then the step-local window copy: `local_rows` rows of 528 bytes, `local_slot` the local
    row of each query row (int64 `[n]`), `hist` / `keep` int64 `[3, H]` / `[3, K]` (request, position, local row) of the
    history rows to copy in and of the chunk's rows to write to the pool (every one, as stock).
    """

    query_requests: torch.Tensor
    query_positions: torch.Tensor
    swa_starts: torch.Tensor
    swa_offsets: torch.Tensor
    main_offsets: torch.Tensor
    swa_pages: torch.Tensor
    main_pages: Optional[torch.Tensor]
    swa_page_size: int
    main_page_size: int
    num_requests: int
    swa_rows: int
    main_rows: int
    swa_first: torch.Tensor
    swa_floors: torch.Tensor
    window: int
    local_rows: int
    local_slot: torch.Tensor
    hist: torch.Tensor
    keep: torch.Tensor


def _host_layout(segs, nreq: int, n: int, window: int, ratio: int) -> Dict[str, np.ndarray]:
    """The plan's host arrays (vectorised over requests)."""
    lens, spans, first = (np.zeros(nreq, dtype=np.int64) for _ in range(3))
    for t0, t1, r, span in segs:
        lens[r], spans[r], first[r] = t1 - t0, span, t0
    present = lens > 0
    swa_offsets = np.concatenate([[1], 1 + np.cumsum(np.where(present, lens + window - 1, 0))])
    domain = np.where(present, spans // ratio, 0) if ratio else np.zeros(nreq, dtype=np.int64)
    main_offsets = np.concatenate([[swa_offsets[-1]], swa_offsets[-1] + np.cumsum(domain)])
    cs = spans - lens
    lo = np.maximum(cs - (window - 1), 0)  # first position a query of the request can read
    col0 = lo // _PAGE
    pages = np.where(present, (spans - col0 * _PAGE + _PAGE - 1) // _PAGE, 0)
    base = np.concatenate([[0], np.cumsum(pages)[:-1]])
    off = base * _PAGE - col0 * _PAGE  # local row of position p of request r: off[r] + p
    ncols = int(max(1, (col0 + pages)[present].max(initial=1)))
    table = np.full((nreq, ncols), -1, dtype=np.int32)
    for r in np.nonzero(present)[0]:
        table[r, col0[r] : col0[r] + pages[r]] = base[r] + np.arange(pages[r])
    rs = np.nonzero(present)[0]
    hcount, kcount = (cs - lo)[rs], (spans - cs)[rs]
    hreq, kreq = np.repeat(rs, hcount), np.repeat(rs, kcount)
    hpos = np.repeat(lo[rs], hcount) + _ragged_arange(hcount)
    kpos = np.repeat(cs[rs], kcount) + _ragged_arange(kcount)
    return dict(
        swa_offsets=swa_offsets,
        main_offsets=main_offsets,
        swa_first=np.where(present, first, 0),
        off=off,
        table=table,
        present=rs.astype(np.int64),
        hist=np.stack([hreq, hpos, off[hreq] + hpos]),
        keep=np.stack([kreq, kpos, off[kreq] + kpos]),
        local_rows=np.int64(max(int(pages.sum()) * _PAGE, n + nreq * (window - 1) + 1)),
    )


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total, dtype=np.int64) - starts


class Csa2SharedPlanOp(OpWrapper):
    """`Csa2SharedPlan` of a context step: `segs` the step's `(t0, t1, request, span)` per request (rows `[t0, t1)` hold
    positions `span - (t1 - t0) .. span - 1`), `nreq` the page-table rows, `pos` int32 / `rid` int64 `[n]` the query
    rows' positions and requests, `floor` int `[n]` the first window position each row may read (None: all), `window`
    the SWA window and `ratio` the compression ratio of the layers that use the plan (0: window rows only).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="two fresh requests, chunks longer than the window",
            spec=dict(reqs=[(0, 700), (0, 300)], ratio=1),
        ),
        Cell(
            why="histories across page edges and a one-row request, ratio 2",
            spec=dict(reqs=[(300, 520), (5000, 40), (77, 1)], ratio=2),
        ),
        Cell(
            why="bounded replay rows (the last 140 of each chunk) with floors, window rows only",
            spec=dict(reqs=[(860, 140), (40, 90)], ratio=0, floor=[860, 40]),
        ),
        Cell(
            why="a request without rows in the step between two with rows",
            spec=dict(reqs=[(129, 200), (0, 0), (1000, 3)], ratio=1),
        ),
    )
    note = (
        "Host index arithmetic and one upload: every field is compared exactly with the reference."
    )

    def __call__(self, segs, nreq: int, pos, rid, floor, window: int, ratio: int) -> Csa2SharedPlan:
        dev, n = pos.device, pos.shape[0]
        h = _host_layout(segs, nreq, n, window, ratio)
        parts = [
            h["swa_offsets"],
            h["main_offsets"],
            h["swa_first"],
            h["off"],
            h["present"],
            h["hist"].reshape(-1),
            h["keep"].reshape(-1),
        ]
        sizes = [len(p) for p in parts]
        host = torch.from_numpy(np.concatenate(parts).astype(np.int64))
        dev_all = (
            maybe_pin_memory(host).to(dev, non_blocking=True) if host.numel() else host.to(dev)
        )
        swa_offsets, main_offsets, swa_first, off, present, hist, keep = torch.split(dev_all, sizes)
        table = maybe_pin_memory(torch.from_numpy(h["table"])).to(dev, non_blocking=True)
        floors = torch.zeros(nreq, dtype=torch.int64, device=dev)
        if floor is not None and present.numel():
            floors[present] = floor.index_select(0, swa_first.index_select(0, present)).long()
        return Csa2SharedPlan(
            query_requests=rid,
            query_positions=pos,
            swa_starts=torch.zeros(nreq, dtype=torch.int64, device=dev),
            swa_offsets=swa_offsets,
            main_offsets=main_offsets,
            swa_pages=table,
            main_pages=None,
            swa_page_size=_PAGE,
            main_page_size=_PAGE // max(ratio, 1),
            num_requests=nreq,
            swa_rows=int(h["swa_offsets"][-1] - 1),
            main_rows=int(h["main_offsets"][-1] - h["swa_offsets"][-1]),
            swa_first=swa_first,
            swa_floors=floors,
            window=window,
            local_rows=int(h["local_rows"]),
            local_slot=off.index_select(0, rid) + pos.long(),
            hist=hist.view(3, -1),
            keep=keep.view(3, -1),
        )

    def reference(
        self, segs, nreq: int, pos, rid, floor, window: int, ratio: int
    ) -> Csa2SharedPlan:
        """Request by request, in Python integers."""
        dev, n = pos.device, pos.shape[0]
        lens, spans, first = [0] * nreq, [0] * nreq, [0] * nreq
        for t0, t1, r, span in segs:
            lens[r], spans[r], first[r] = t1 - t0, span, t0
        swa_off, off, hist, keep, table = [1], [0] * nreq, [], [], {}
        for r in range(nreq):
            swa_off.append(swa_off[-1] + (lens[r] + window - 1 if lens[r] else 0))
        main_off = [swa_off[-1]]
        for r in range(nreq):
            main_off.append(main_off[-1] + (spans[r] // ratio if lens[r] and ratio else 0))
        base = 0
        for r in range(nreq):
            if not lens[r]:
                continue
            cs = spans[r] - lens[r]
            lo = max(cs - (window - 1), 0)
            col0, npages = lo // _PAGE, -(-(spans[r] - lo // _PAGE * _PAGE) // _PAGE)
            for j in range(npages):
                table[(r, col0 + j)] = base + j
            off[r] = base * _PAGE - col0 * _PAGE
            hist += [(r, p, off[r] + p) for p in range(lo, cs)]
            keep += [(r, p, off[r] + p) for p in range(cs, spans[r])]
            base += npages
        ncols = max([c + 1 for _, c in table] + [1])
        tab = torch.full((nreq, ncols), -1, dtype=torch.int32)
        for (r, c), v in table.items():
            tab[r, c] = v
        floors = [0] * nreq
        if floor is not None:
            fl = floor.cpu().tolist()
            floors = [fl[first[r]] if lens[r] else 0 for r in range(nreq)]

        def t(v):
            return torch.tensor(v, dtype=torch.int64, device=dev)

        def trip(v):
            return (
                t(v).view(-1, 3).t().contiguous()
                if v
                else torch.zeros(3, 0, dtype=torch.int64, device=dev)
            )

        return Csa2SharedPlan(
            query_requests=rid,
            query_positions=pos,
            swa_starts=torch.zeros(nreq, dtype=torch.int64, device=dev),
            swa_offsets=t(swa_off),
            main_offsets=t(main_off),
            swa_pages=tab.to(dev),
            main_pages=None,
            swa_page_size=_PAGE,
            main_page_size=_PAGE // max(ratio, 1),
            num_requests=nreq,
            swa_rows=swa_off[-1] - 1,
            main_rows=main_off[-1] - swa_off[-1],
            swa_first=t([first[r] if lens[r] else 0 for r in range(nreq)]),
            swa_floors=t(floors),
            window=window,
            local_rows=max(base * _PAGE, n + nreq * (window - 1) + 1),
            local_slot=t([off[r] for r in rid.cpu().tolist()]) + pos.long(),
            hist=trip(hist),
            keep=trip(keep),
        )

    def is_valid(self, segs, nreq: int, pos, rid, floor, window: int, ratio: int) -> None:
        assert pos.dtype == torch.int32 and rid.dtype == torch.int64 and pos.shape == rid.shape
        assert all(0 <= r < nreq and t1 >= t0 for t0, t1, r, _ in segs), (
            "segs are (t0, t1, request row, span)"
        )
        assert sum(t1 - t0 for t0, t1, _, _ in segs) == pos.shape[0], (
            "the segs cover every query row"
        )
        assert window == _PAGE, "the local window copy pages by the window"
        assert ratio in (0, 1, 2)
        if floor is not None:
            assert floor.shape == pos.shape


@triton.jit
def _local_window_kernel(
    LOCAL,
    POOL,
    TABLE,
    HIST,
    KEEP,
    H,
    K,
    nblk,
    table_stride,
    ROW: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < ROW
    if pid < H:  # a history row: pool -> local (zeros when unmapped)
        r = tl.load(HIST + pid)
        p = tl.load(HIST + H + pid)
        dst = tl.load(HIST + 2 * H + pid)
        blk = p // PAGE
        ok = (p >= 0) & (blk < nblk)
        page = tl.load(TABLE + r * table_stride + tl.where(ok, blk, 0), mask=ok, other=-1)
        ok = ok & (page >= 0)
        src = page.to(tl.int64) * PAGE + (p - blk * PAGE)
        v = tl.load(POOL + src * ROW + offs, mask=m & ok, other=0)
        tl.store(LOCAL + dst * ROW + offs, v, mask=m)
    else:  # a row that stays in the window: local -> pool (skipped when unmapped)
        i = pid - H
        r = tl.load(KEEP + i)
        p = tl.load(KEEP + K + i)
        src = tl.load(KEEP + 2 * K + i)
        blk = p // PAGE
        ok = (p >= 0) & (blk < nblk)
        page = tl.load(TABLE + r * table_stride + tl.where(ok, blk, 0), mask=ok, other=-1)
        ok = ok & (page >= 0)
        dst = page.to(tl.int64) * PAGE + (p - blk * PAGE)
        v = tl.load(LOCAL + src * ROW + offs, mask=m)
        tl.store(POOL + dst * ROW + offs, v, mask=m & ok)


class Csa2LocalWindow(OpWrapper):
    """In place: `pool row of (keep[0], keep[1]) = local[keep[2]]` (skipped when unmapped), then `local[hist[2]] = pool
    row of (hist[0], hist[1])` (zeros when unmapped). `local` uint8 `[R, 528]`, `swa_pool` uint8 `[pages, 128, 528]`
    with the layer's int32 page table `swa_table` `[requests, blocks]`, `hist` / `keep` int64 `[3, H]` / `[3, K]` from
    `csa2_shared_plan`. Two launches in that order, as the decode path publishes the window before reading history:
    history positions precede their request's chunk and kept ones are in it, so they only meet when two requests' page
    tables share a page.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="two requests with histories across page edges, a long chunk and a short one",
            spec=dict(reqs=[(300, 520), (77, 40)]),
        ),
        Cell(
            why="a fresh request (no history) and an unmapped history page",
            spec=dict(reqs=[(0, 700), (900, 5)], unmapped=True),
        ),
    )
    note = "A row copy: compared bit for bit with the reference."

    def __call__(self, local, swa_pool, swa_table, hist, keep) -> None:
        H, K = hist.shape[1], keep.shape[1]
        with torch.cuda.device(local.device):
            if K:  # programs past H copy kept rows: none of the first launch's programs are history rows
                _local_window_kernel[(K,)](
                    local,
                    swa_pool,
                    swa_table,
                    hist,
                    keep,
                    0,
                    K,
                    swa_table.shape[1],
                    swa_table.stride(0),
                    ROW=SWA_ROW,
                    PAGE=_PAGE,
                    BLOCK=1024,
                )
            if H:
                _local_window_kernel[(H,)](
                    local,
                    swa_pool,
                    swa_table,
                    hist,
                    keep,
                    H,
                    K,
                    swa_table.shape[1],
                    swa_table.stride(0),
                    ROW=SWA_ROW,
                    PAGE=_PAGE,
                    BLOCK=1024,
                )

    def reference(self, local, swa_pool, swa_table, hist, keep) -> tuple:
        """Returns `(local, pool)` updated copies; the arguments are left alone."""
        local, flat = local.clone(), swa_pool.reshape(-1, SWA_ROW).clone()

        def slot(r, p):
            b = p // _PAGE
            if p < 0 or b >= swa_table.shape[1] or int(swa_table[r, b]) < 0:
                return -1
            return int(swa_table[r, b]) * _PAGE + p % _PAGE

        for r, p, s in keep.t().tolist():
            d = slot(r, p)
            if d >= 0:
                flat[d] = local[s]
        for r, p, d in hist.t().tolist():
            s = slot(r, p)
            local[d] = flat[s] if s >= 0 else 0
        return local, flat.view_as(swa_pool)

    def is_valid(self, local, swa_pool, swa_table, hist, keep) -> None:
        assert local.dtype == torch.uint8 and local.is_contiguous() and local.shape[1] == SWA_ROW
        assert (
            swa_pool.dtype == torch.uint8
            and swa_pool.is_contiguous()
            and swa_pool.shape[1:] == (_PAGE, SWA_ROW)
        )
        assert swa_table.dtype == torch.int32 and swa_table.stride(1) == 1
        assert (
            hist.dtype == keep.dtype == torch.int64
            and hist.is_contiguous()
            and keep.is_contiguous()
        )


class _Workspace:
    """Grow-only staging buffers of one device (a target-owned scratch, like the engine's per-tile metadata buffers),
    bounded by the targets' use: a step takes the shared bank only within the engine's bound (`swa_rows + main_rows <=
    rows x (window + selection width)`), else the selected-row path."""

    def __init__(self, device):
        self.device = device
        self.bank = self.selected = self.indices = self.lens = None
        z = lambda k, dt: torch.zeros(k, dtype=dt, device=device)  # noqa: E731
        self.counter, self.main_scales, self.bitmap = (
            z(1, torch.uint32),
            z(2, torch.int32),
            z(17, torch.int32),
        )
        self.bmm1, self.bmm2 = z(2, torch.float32), z(1, torch.float32)
        self.kv_dequant, self.kv_quant = torch.ones(1, device=device), torch.ones(1, device=device)
        self.unit = torch.ones(1, device=device)  # Q's scale: E4M3 at unit scale

    def views(self, rows: int, n: int, cap: int):
        if self.bank is None or self.bank.numel() < rows * HEAD_DIM:
            k = max(rows * HEAD_DIM * 5 // 4, _TMA_MIN_BYTES)
            self.bank = torch.empty(k, dtype=torch.float8_e4m3fn, device=self.device)
            self.selected = torch.zeros(k // HEAD_DIM, dtype=torch.int32, device=self.device)
        if self.indices is None or self.indices.numel() < n * cap or self.lens.numel() < n:
            self.indices = torch.empty(n * cap * 5 // 4, dtype=torch.int32, device=self.device)
            self.lens = torch.empty(n * 5 // 4 + 1, dtype=torch.int32, device=self.device)
        return (
            self.bank[: rows * HEAD_DIM].view(rows, HEAD_DIM),
            self.selected[:rows],
            self.indices[: n * cap].view(n, cap),
            self.lens[:n],
        )


_WS: Dict[int, _Workspace] = {}


def _workspace(device) -> _Workspace:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WS.get(idx)
    if ws is None:
        ws = _WS[idx] = _Workspace(device)
    return ws


class Csa2SharedStage(OpWrapper):
    """`(bank E4M3 [rows, 512], indices int32 [n, 128 + k], lens int32 [n], q_scale, bmm1 [2], bmm2 [1])` of one layer
    (`q_scale` is ones: the context kernel quantizes Q to E4M3 at unit scale, as stock's unit-scale context Q).

    `plan` from `csa2_shared_plan` (the layer's ratio), `local` its window copy (`csa2_local_window` done), `main_pool`
    uint8 `[rows, 288]` with the owner's int32 page table `main_table` `[requests, blocks]`, `topk` int `[n, k]` each
    query's selected compressed positions (-1 none; None with no compressed attention), `visible` int64 `[n]` its
    visible compressed rows and `max_positions` the pool's bound. `swa_valid` int64 `[n, window]`: >= 0 where the
    query's window slot `j` (position `pos - window + 1 + j`) may be read (`csa2_chunk_window`'s indices). `q` bf16 the
    rotated query (its row count sizes the step), `scale` the softmax scale. `compact` False reuses the previous call's
    indices and marks (a layer of the same KV owner and index source in the same step: `swa_valid`, `topk` unread),
    `parity` alternates over such calls. The buffers are the device's staging workspace, valid until the next call.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="window rows only (a layer without compressed attention), two requests",
            spec=dict(reqs=[(300, 200), (40, 90)], ratio=0, topk=0),
        ),
        Cell(
            why="ratio 1, 128 selections, a fresh request and one with history",
            spec=dict(reqs=[(0, 500), (500, 120)], ratio=1, topk=128),
        ),
        Cell(
            why="ratio 2, 256 selections, long history, unmapped history slots and replay floors",
            spec=dict(
                reqs=[(3000, 400), (130, 60)], ratio=2, topk=256, floor=[3100, 130], unmapped=True
            ),
        ),
        Cell(
            why="a second layer of the group: no compaction, main rows kept at the same scale",
            spec=dict(reqs=[(700, 300)], ratio=1, topk=128, second=True),
        ),
    )
    note = """
    Window rows are E4M3 under power-of-two group scales and the staging scale is a power of two, so they are staged
    exactly; compressed rows (E2M1 x E4M3 group scale) round once to E4M3, a relative error of 1/16 per element at most.
    """

    def __call__(
        self,
        plan,
        local,
        main_pool,
        main_table,
        topk,
        visible,
        max_positions: int,
        swa_valid,
        q,
        scale: float,
        compact: bool,
        parity: int,
    ) -> tuple:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import stage_shared_rows

        n, dev = q.shape[0], q.device
        ratio_rows = plan.main_page_size
        cap = plan.window + (
            -(-topk.shape[1] // 128) * 128 if topk is not None else 0
        )  # stock's 128-key tiles
        rows = 1 + plan.swa_rows + plan.main_rows
        ws = _workspace(dev)
        bank, selected, indices, lens = ws.views(rows, n, cap)
        meta = SimpleNamespace(
            num_tokens=n,
            num_sparse_topk=cap,
            shared_pool=bank,
            _csa2_shared_selected=selected,
            _csa2_shared_main_scales=ws.main_scales,
            prepared_indices=indices,
            prepared_lens=lens,
            prepared_counter=ws.counter,
            mla_bmm1_scale=ws.bmm1,
            mla_bmm2_scale=ws.bmm2,
            _csa2_stage_scale_bitmap=ws.bitmap,
            _csa2_stage_kv_dequant=ws.kv_dequant,
            _csa2_stage_kv_quant=ws.kv_quant,
        )
        has_main = topk is not None
        p = dataclasses.replace(plan, main_pages=main_table if has_main else None)
        mapping = (
            (main_table, ratio_rows, max_positions, plan.query_requests, visible)
            if has_main
            else None
        )
        with torch.cuda.device(dev):
            stage_shared_rows(
                local,
                swa_valid,
                main_pool if has_main else None,
                None,
                topk if has_main else None,
                meta,
                p,
                main_mapping=mapping,
                kv_scale_orig_quant=ws.kv_quant,
                kv_scale_quant_orig=ws.kv_dequant,
                q_scale_quant_orig=ws.unit,
                softmax_scale=scale,
                scale=(None, True, 0, CONTEXT_SCALE_MAX_EXPONENT),
                compact=compact,
                parity=parity,
            )
        return bank, indices, lens, ws.unit, ws.bmm1, ws.bmm2

    def reference(
        self,
        plan,
        local,
        main_pool,
        main_table,
        topk,
        visible,
        max_positions: int,
        swa_valid,
        q,
        scale: float,
        compact: bool,
        parity: int,
    ) -> tuple:
        """`(rows fp32 [n, 128 + k, 512], lens int32 [n])`: every query's valid window rows (from `local`, slot order),
        then its valid selected rows (from the pool), dequantized, zeros past `lens`; stock's validity rules (a window
        slot from the request's first readable position on, a selection below the request's domain, the query's visible
        rows and the pool bound, at a mapped page). `compact` / `parity` / `q` / `scale` do not change the rows."""
        n, dev, W = q.shape[0], q.device, plan.window
        pos, rid = plan.query_positions.long(), plan.query_requests.long()
        first = plan.swa_first[rid]
        start = torch.maximum(
            (plan.query_positions.long()[first] - W + 1).clamp_min(0), plan.swa_floors[rid]
        )
        logical = pos[:, None] - W + 1 + torch.arange(W, device=dev)
        ok = (swa_valid >= 0) & (logical >= start[:, None])
        off = plan.local_slot - pos  # local row of position p of the query's request: off + p
        swa = csa2_unpack_rows(
            local[(off[:, None] + logical).clamp(0, local.shape[0] - 1)].view(-1, SWA_ROW), "swa"
        )
        parts, valid = [swa.view(n, W, HEAD_DIM).float()], [ok]
        if topk is not None:
            sel = topk.long()
            domain = (plan.main_offsets[rid + 1] - plan.main_offsets[rid])[:, None]
            rpp = plan.main_page_size
            blk = sel.clamp_min(0) // rpp
            page = main_table.long()[
                rid[:, None].expand_as(sel), blk.clamp(max=main_table.shape[1] - 1)
            ]
            slot = page * rpp + sel.clamp_min(0) % rpp
            mok = (
                (sel >= 0)
                & (sel < domain)
                & (sel < visible[:, None])
                & (sel < max_positions)
                & (blk < main_table.shape[1])
                & (page >= 0)
                & (slot < main_pool.shape[0])
            )
            rows = csa2_unpack_rows(
                main_pool[slot.clamp(0, main_pool.shape[0] - 1).view(-1)], "main"
            )
            parts.append(rows.view(n, -1, HEAD_DIM).float())
            valid.append(mok)
        allr, allv = torch.cat(parts, 1), torch.cat(valid, 1)
        width = allv.shape[1]
        order = torch.where(allv, torch.arange(width, device=dev), width).argsort(
            dim=1, stable=True
        )
        cnt = allv.sum(1)
        out = allr.gather(1, order[..., None].expand(-1, -1, HEAD_DIM))
        out = out * (torch.arange(width, device=dev)[None, :] < cnt[:, None])[..., None]
        return out, cnt.clamp_min(1).int()

    def is_valid(
        self,
        plan,
        local,
        main_pool,
        main_table,
        topk,
        visible,
        max_positions: int,
        swa_valid,
        q,
        scale: float,
        compact: bool,
        parity: int,
    ) -> None:
        assert (
            local.dtype == torch.uint8
            and local.is_contiguous()
            and local.shape == (plan.local_rows, SWA_ROW)
        )
        assert (
            q.dtype in (torch.bfloat16, torch.float8_e4m3fn)
            and q.is_contiguous()
            and q.shape[0] == plan.query_positions.shape[0]
        )
        assert plan.window + (0 if topk is None else topk.shape[1]) <= 4096 + 128
        if compact:
            assert swa_valid is not None and swa_valid.shape == (q.shape[0], plan.window)
        if topk is not None:
            assert (
                main_pool.dim() == 2 and main_pool.shape[1] == MAIN_ROW and main_pool.stride(1) == 1
            )
            assert main_table.dtype == torch.int32 and main_table.stride(1) == 1
            assert visible.dtype == torch.int64 and visible.shape == (q.shape[0],)
            assert plan.main_page_size & (plan.main_page_size - 1) == 0


csa2_shared_plan = Csa2SharedPlanOp()
csa2_local_window = Csa2LocalWindow()
csa2_shared_stage = Csa2SharedStage()
