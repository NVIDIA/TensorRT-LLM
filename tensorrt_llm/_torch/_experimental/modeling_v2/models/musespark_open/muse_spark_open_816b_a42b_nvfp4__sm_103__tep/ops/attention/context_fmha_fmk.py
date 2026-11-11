# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host side of the sm_103a / sm_107 multi-instance paged context FMHA (``context_fmha_fmk_kernel``).

``plan_context_fmha`` turns the step's per-request query and key lengths into the kernel's work
units and per-CTA unit ranges; ``paged_context_fmha_fmk`` runs the kernel on a plan and, when the
plan cut a unit's key range, merges the partial results.

Planning. An item is 128 query rows of one request times ``num_inst`` heads of one KV group over
the request's key tiles up to the item's last row. Its cost is its tile count plus a per-item
overhead (``ITEM_OVERHEAD``; a piece of a cut item ``PIECE_OVERHEAD``). With ``split=True`` (the
default) the items, longest first, are laid end to end and cut
into ``num_ctas`` equal shares of the total cost (stream-K): a CTA's share is a list of whole
items plus at most a head and a tail piece of items it shares with its neighbours; a cut is
placed only before an item's first masked tile, so the masked tiles of an item are always in its
last piece. Pieces write ``O / l`` and ``(m, l)`` to a workspace; ``merge`` combines the pieces
of every cut item. A share is at least two minimum pieces plus the overhead (a small call uses
fewer CTAs), and the split is kept only when its busiest CTA has less work than with whole items
assigned longest first to the CTA with the least work (LPT), which ``split=False`` always uses.

Numerics are the kernel's (see ``context_fmha_fmk_kernel``); the merge weights every piece by
``l * 2^(m - max m)`` in fp32 and rounds the merged row to BF16 once.
"""

from __future__ import annotations

import functools
import heapq
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory

#: Head width, tokens per TMA box (a 64-token cache block is two boxes) and query rows per instance
#: the kernel is built for, and the cache block sizes it reads.
HEAD_DIM = 64
PAGE = 32
CACHE_PAGES = (32, 64)
M_TILE = 128
UNIT_FIELDS = 8
#: Planner cost of a unit beyond its key tiles, in key tiles: a whole item (BF16 epilogue,
#: diagonal steps, the next unit's Q wait) and a piece of a cut item (fp32 partial epilogue).
#: Fitted to per-unit timestamps of the kernel's diagnostic build on Rubin (the overheads matter
#: for short units; deep units amortise either). Refitted on GB300 for the kernel with
#: ``cond_rescale`` and ``f16_exp`` (a shorter step makes the fixed per-unit cost count for
#: more): 16 / 26 and a minimum piece of 32 tiles (was 8 / 13 / 16) gave +1.0% summed on the
#: eight benchmark shapes.
ITEM_OVERHEAD = 16.0
PIECE_OVERHEAD = 26.0


@dataclass
class FmkPlan:
    """Work units on the device plus what the merge needs."""

    units: torch.Tensor  # int32 [num_units, 8]
    cta_off: torch.Tensor  # int32 [num_ctas + 1]
    num_units: int
    num_ctas: int
    num_slots: int  # workspace slots (pieces of cut items)
    merge_meta: Optional[torch.Tensor]  # int32 [num_merges, 4]: b, m, hg, num_pieces
    merge_slots: Optional[torch.Tensor]  # int32 [num_merges, max_pieces]
    max_pieces: int
    num_inst: int
    tile_n: int
    makespan: float  # planned cost of the busiest CTA
    ideal: float  # total cost / num_ctas


def _items(q_lens, kv_lens, num_heads, num_kv_heads, num_inst, tile_n):
    groups = num_heads // num_inst
    items = []
    for b, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        if ql <= 0 or kl <= 0:
            continue
        off = kl - ql
        last_tile_req = (kl - 1) // tile_n
        for m in range(-(-ql // M_TILE)):
            t_end = min(last_tile_req, (off + m * M_TILE + M_TILE - 1) // tile_n) + 1
            first_masked = (off + m * M_TILE + 1) // tile_n
            for hg in range(groups):
                items.append((b, m, hg, t_end, min(first_masked, t_end)))
    return items


def _lpt(items, num_ctas, item_overhead):
    """Whole items, longest first, each to the CTA with the least work."""
    per_cta = [[] for _ in range(num_ctas)]
    loads = [0.0] * num_ctas
    heap = [(0.0, k) for k in range(num_ctas)]
    for it in items:
        load, k = heapq.heappop(heap)
        per_cta[k].append((it[0], it[1], it[2], 0, it[3], 0, 0))
        load += it[3] + item_overhead
        loads[k] = load
        heapq.heappush(heap, (load, k))
    return per_cta, loads, {}, 0


def _stream_k(items, num_ctas, share, item_overhead, piece_overhead, min_piece):
    """Items laid end to end and cut into CTA shares of ``share`` (see the module docstring).

    A whole item costs its tiles plus ``item_overhead``, a piece of a cut item its tiles plus
    ``piece_overhead`` (its partial epilogue writes fp32 rows and statistics).
    """
    per_cta = [[] for _ in range(num_ctas)]
    loads = [0.0] * num_ctas
    pieces = {}  # item index -> list of slots
    num_slots = 0
    k = 0
    filled = 0.0
    for idx, (b, m, hg, t_end, fm) in enumerate(items):
        a = 0
        while a < t_end:
            cap = share * (k + 1) - filled
            need = (t_end - a) + (item_overhead if a == 0 else piece_overhead)
            if need <= cap + 0.5 or k == num_ctas - 1:
                if a == 0:
                    per_cta[k].append((b, m, hg, 0, t_end, 0, 0))
                else:
                    per_cta[k].append((b, m, hg, a, t_end, 1, num_slots))
                    pieces[idx].append(num_slots)
                    num_slots += 1
                filled += need
                loads[k] += need
                break
            x = int(math.floor(cap - piece_overhead))
            x = min(x, fm - a)
            if x < min_piece or t_end - (a + x) < min_piece:
                if cap < (t_end - a) / 2 + piece_overhead and k < num_ctas - 1:
                    # Too little room left here for a useful piece: close this CTA.
                    filled = share * (k + 1)
                    k += 1
                    continue
                # The piece cannot be cut usefully: the CTA takes the rest whole.
                if a == 0:
                    per_cta[k].append((b, m, hg, 0, t_end, 0, 0))
                else:
                    per_cta[k].append((b, m, hg, a, t_end, 1, num_slots))
                    pieces[idx].append(num_slots)
                    num_slots += 1
                filled += need
                loads[k] += need
                break
            pieces.setdefault(idx, []).append(num_slots)
            per_cta[k].append((b, m, hg, a, a + x, 1, num_slots))
            num_slots += 1
            filled += x + piece_overhead
            loads[k] += x + piece_overhead
            a += x
            k += 1
    return per_cta, loads, pieces, num_slots


def plan_context_fmha(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    *,
    num_ctas: int,
    num_heads: int = 32,
    num_kv_heads: int = 4,
    num_inst: int = 4,
    tile_n: int = 64,
    split: bool = True,
    item_overhead: float = ITEM_OVERHEAD,
    piece_overhead: float = PIECE_OVERHEAD,
    min_piece: int = 32,
    device: Optional[torch.device] = None,
) -> FmkPlan:
    """The work units of one context call; see the module docstring.

    The plan's int32 arrays reach ``device`` in one copy: from pinned host memory without a
    host wait when ``device`` is a CUDA device (the copy is ordered on the current stream, so a
    kernel launched after this call on that stream reads it), synchronously otherwise.
    """
    items = _items(q_lens, kv_lens, num_heads, num_kv_heads, num_inst, tile_n)
    items.sort(key=lambda it: -it[3])
    per_cta, loads, pieces, num_slots = _lpt(items, num_ctas, item_overhead)
    total = sum(it[3] + item_overhead for it in items)
    ideal = total / max(num_ctas, 1)
    if split and items:
        # A cut turns one whole item into two pieces; a stream-K split makes at most
        # num_ctas - 1 cuts. A share below two minimum pieces would close every CTA without a
        # cut (the last CTA would take the rest), so small calls use fewer CTAs.
        per_cut = max(2.0 * piece_overhead - item_overhead, 0.0)
        share = (total + per_cut * (num_ctas - 1)) / max(num_ctas, 1)
        share = max(share, 2.0 * min_piece + piece_overhead)
        sk = _stream_k(items, num_ctas, share, item_overhead, piece_overhead, min_piece)
        if max(sk[1]) < max(loads):
            per_cta, loads, pieces, num_slots = sk
    units = []
    cta_off = [0]
    for k in range(num_ctas):
        units.extend(per_cta[k])
        cta_off.append(len(units))
    num_units = max(len(units), 1)
    merges = [(items[i][0], items[i][1], items[i][2], s) for i, s in pieces.items()]
    max_pieces = max((len(s) for *_, s in merges), default=0)
    # One int32 buffer: units [num_units, 8], cta_off [num_ctas + 1], then (when items were
    # cut) merge_meta [merges, 4] and merge_slots [merges, max_pieces]; every section starts at
    # a 16-byte boundary.
    lengths = [num_units * UNIT_FIELDS, num_ctas + 1, 4 * len(merges), max_pieces * len(merges)]
    sizes = [-(-n // 4) * 4 for n in lengths]
    starts = np.cumsum([0] + sizes[:-1]).tolist()
    host = np.zeros(sum(sizes), dtype=np.int32)

    def section(i):
        return host[starts[i] : starts[i] + lengths[i]]

    if units:
        section(0)[: len(units) * UNIT_FIELDS].reshape(len(units), UNIT_FIELDS)[:, :7] = np.asarray(
            units, dtype=np.int32
        )
    section(1)[:] = cta_off
    if merges:
        mm = section(2).reshape(len(merges), 4)
        ms = section(3).reshape(len(merges), max_pieces)
        ms.fill(-1)
        for i, (b, m, hg, s) in enumerate(merges):
            mm[i] = (b, m, hg, len(s))
            ms[i, : len(s)] = s
    dev = device if device is not None else torch.device("cuda")
    buf = torch.from_numpy(host)
    if dev.type == "cuda":
        buf = maybe_pin_memory(buf).to(dev, non_blocking=True)
    else:
        buf = buf.to(dev)
    views = [buf[s : s + n] for s, n in zip(starts, lengths)]
    return FmkPlan(
        units=views[0].view(num_units, UNIT_FIELDS),
        cta_off=views[1],
        num_units=len(units),
        num_ctas=num_ctas,
        num_slots=num_slots,
        merge_meta=views[2].view(len(merges), 4) if merges else None,
        merge_slots=views[3].view(len(merges), max_pieces) if merges else None,
        max_pieces=max_pieces,
        num_inst=num_inst,
        tile_n=tile_n,
        makespan=max(loads) if loads else 0.0,
        ideal=ideal,
    )


@functools.lru_cache(maxsize=32)
def _compiled_kernel(
    device_index: int,
    num_heads: int,
    num_kv_heads: int,
    num_inst: int,
    tile_n: int,
    write_stats: bool,
    ablate_mufu: bool,
    kv_stage: int,
    trace: bool = False,
    s_stages: int = 1,
    cache_page: int = PAGE,
    cond_rescale: bool = True,
    f16_exp: bool = True,
):
    """The compiled kernel of one build on one device (every size is a runtime argument)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from . import context_fmha_fmk_kernel as fk

    kernel = fk.PagedContextFmhaFmk(
        num_heads,
        num_kv_heads,
        num_inst=num_inst,
        tile_n=tile_n,
        kv_stage=kv_stage,
        s_stages=s_stages,
        write_stats=write_stats,
        ablate_mufu=ablate_mufu,
        trace=trace,
        cache_page=cache_page,
        cond_rescale=cond_rescale,
        f16_exp=f16_exp,
    )
    gmem = cute.AddressSpace.gmem
    ptrs = [
        make_ptr(cutlass.Float8E4M3FN, 16, gmem, assumed_align=16),  # q
        make_ptr(cutlass.Float8E4M3FN, 16, gmem, assumed_align=16),  # pool
        make_ptr(cutlass.Int32, 16, gmem, assumed_align=4),  # table
        make_ptr(cutlass.Int32, 16, gmem, assumed_align=4),  # cu_q
        make_ptr(cutlass.Int32, 16, gmem, assumed_align=4),  # kv_lens
        make_ptr(cutlass.BFloat16, 16, gmem, assumed_align=16),  # out
        make_ptr(cutlass.Int32, 16, gmem, assumed_align=16),  # units
        make_ptr(cutlass.Int32, 16, gmem, assumed_align=4),  # cta_off
        make_ptr(cutlass.Float32, 16, gmem, assumed_align=16),  # part_o
        make_ptr(cutlass.Float32, 16, gmem, assumed_align=8),  # part_ml
        make_ptr(cutlass.Float32, 16, gmem, assumed_align=8),  # stats
        make_ptr(cutlass.Float32, 16, gmem, assumed_align=4),  # scale_log2
        make_ptr(cutlass.Float32, 16, gmem, assumed_align=4),  # scale_out
    ]
    with torch.cuda.device(device_index):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute.compile(
            kernel,
            *ptrs,
            *[cutlass.Int32(1)] * 7,
            stream,
            # Opt-level 1: level 3 (the DSL default) measured slower on this kernel.
            options="--enable-tvm-ffi --opt-level=1",
        )


_MERGE_KERNEL = None


def _merge_kernel():
    global _MERGE_KERNEL
    if _MERGE_KERNEL is not None:
        return _MERGE_KERNEL
    import triton
    import triton.language as tl

    # ``maxp`` (the row stride of the slot table) is a runtime value, unspecialised, so one
    # compiled kernel serves every plan.
    @triton.jit(do_not_specialize=["maxp"])
    def _merge(
        out_ptr,
        stats_ptr,
        part_o_ptr,
        part_ml_ptr,
        meta_ptr,
        slots_ptr,
        cu_q_ptr,
        maxp,
        H: tl.constexpr,
        NI: tl.constexpr,
        DH: tl.constexpr,
        MT: tl.constexpr,
        WRITE_STATS: tl.constexpr,
    ):
        e = tl.program_id(0)
        j = tl.program_id(1)
        b = tl.load(meta_ptr + e * 4 + 0)
        m = tl.load(meta_ptr + e * 4 + 1)
        hg = tl.load(meta_ptr + e * 4 + 2)
        npc = tl.load(meta_ptr + e * 4 + 3)
        q0 = tl.load(cu_q_ptr + b)
        q_len = tl.load(cu_q_ptr + b + 1) - q0
        rows = tl.arange(0, MT)
        d = tl.arange(0, DH)
        mx = tl.full([MT], float("-inf"), tl.float32)
        for p in range(0, npc):
            s = tl.load(slots_ptr + e * maxp + p).to(tl.int64)
            base = (s * NI + j) * MT + rows
            ml = tl.load(part_ml_ptr + base * 2)
            lp = tl.load(part_ml_ptr + base * 2 + 1)
            mx = tl.where(lp > 0, tl.maximum(mx, ml), mx)
        acc = tl.zeros([MT, DH], tl.float32)
        den = tl.zeros([MT], tl.float32)
        for p in range(0, npc):
            s = tl.load(slots_ptr + e * maxp + p).to(tl.int64)
            base = (s * NI + j) * MT + rows
            ml = tl.load(part_ml_ptr + base * 2)
            lp = tl.load(part_ml_ptr + base * 2 + 1)
            w = tl.where(lp > 0, lp * tl.exp2(ml - mx), 0.0)
            o = tl.load(part_o_ptr + base[:, None] * DH + d[None, :])
            acc += w[:, None] * o
            den += w
        res = acc / tl.where(den > 0, den, 1.0)[:, None]
        r = m * MT + rows
        ok = r < q_len
        head = hg * NI + j
        dst = (q0 + r).to(tl.int64) * H + head
        tl.store(out_ptr + dst[:, None] * DH + d[None, :], res.to(tl.bfloat16), mask=ok[:, None])
        if WRITE_STATS:
            # natural-log max of the scaled scores and the row sum without the 2^8 pre-scale
            tl.store(stats_ptr + dst * 2, mx * 0.6931471805599453, mask=ok)
            tl.store(stats_ptr + dst * 2 + 1, den * 0.00390625, mask=ok)

    _MERGE_KERNEL = _merge
    return _MERGE_KERNEL


def max_slots(num_ctas: int) -> int:
    """The most workspace slots a split plan over ``num_ctas`` CTAs uses: each of its at most
    ``num_ctas - 1`` cuts adds at most two pieces."""
    return 2 * max(num_ctas - 1, 1)


class FmkWorkspace:
    """Partial-result buffers. ``reserve`` sizes them up front; ``get`` grows them on demand
    otherwise (never inside a CUDA-graph capture, which a reserved workspace never needs)."""

    def __init__(self):
        self.part_o = None
        self.part_ml = None

    def reserve(self, slots: int, num_inst: int, device) -> None:
        self.get(slots, num_inst, device)

    def get(self, slots: int, num_inst: int, device):
        need = max(slots, 1) * num_inst * M_TILE
        if self.part_o is None or self.part_o.numel() < need * HEAD_DIM:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("FmkWorkspace: the workspace must be reserved before capture")
            self.part_o = torch.empty(need * HEAD_DIM, dtype=torch.float32, device=device)
            self.part_ml = torch.empty(need * 2, dtype=torch.float32, device=device)
        return self.part_o, self.part_ml

    @property
    def nbytes(self) -> int:
        if self.part_o is None:
            return 0
        return (self.part_o.numel() + self.part_ml.numel()) * 4


_WORKSPACE = FmkWorkspace()


def paged_context_fmha_fmk(
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    table: torch.Tensor,
    cu_q: torch.Tensor,
    kv_lens: torch.Tensor,
    plan: FmkPlan,
    scale_log2: torch.Tensor,
    scale_out: torch.Tensor,
    out: torch.Tensor,
    *,
    stats: Optional[torch.Tensor] = None,
    ablate_mufu: bool = False,
    kv_stage: int = 12,
    workspace: Optional[FmkWorkspace] = None,
    trace: Optional[torch.Tensor] = None,
    s_stages: int = 1,
    cond_rescale: bool = True,
    f16_exp: bool = True,
) -> torch.Tensor:
    """Causal (bottom-right) attention of the packed new rows ``q`` over their paged contexts.

    ``q`` E4M3 ``[total_q, heads, 64]``; ``kv_pool`` E4M3 ``[blocks, kv_heads, cache_page, 64]``
    with ``cache_page`` 32 or 64;
    ``table`` int32 ``[batch, 2, width]`` (K and V block of each page); ``cu_q`` int32
    ``[batch + 1]``; ``kv_lens`` int32 ``[batch]``; ``scale_log2`` and ``scale_out`` fp32 device
    scalars (any tensor whose first element is the value) = the softmax scale times log2(e) and
    the output scale; ``out`` BF16 ``[total_q, heads, 64]``; ``stats`` optional fp32
    ``[total_q, heads, 2]`` (natural-log row max, row sum). ``plan`` from ``plan_context_fmha`` on
    the same lengths. ``trace`` (diagnostic): an int64 buffer of ``plan.num_units *
    TRACE_FIELDS + TRACE_STEP_WORDS`` words (``context_fmha_fmk_kernel``) selects the trace build.
    ``s_stages`` 2 (with ``plan.tile_n`` 32) double-buffers S per instance. ``cond_rescale`` and
    ``f16_exp`` select the softmax builds of ``PagedContextFmhaFmk`` (both on by default; off gives
    the f32 softmax that rescales on every increase of the maximum). No host
    synchronisation: every input is read on the device, in stream order.
    """
    import cuda.bindings.driver as cuda

    heads, kv_heads, cache_page = q.shape[1], kv_pool.shape[1], kv_pool.shape[2]
    assert cache_page in CACHE_PAGES and plan.tile_n % cache_page == 0, (cache_page, plan.tile_n)
    dev = q.device
    assert stats is None or trace is None
    compiled = _compiled_kernel(
        dev.index if dev.index is not None else torch.cuda.current_device(),
        heads,
        kv_heads,
        plan.num_inst,
        plan.tile_n,
        stats is not None,
        bool(ablate_mufu),
        int(kv_stage),
        trace is not None,
        int(s_stages),
        int(cache_page),
        bool(cond_rescale),
        bool(f16_exp),
    )
    ws = workspace if workspace is not None else _WORKSPACE
    part_o, part_ml = ws.get(plan.num_slots, plan.num_inst, dev)
    batch = kv_lens.shape[0]
    stats_ptr = stats.data_ptr() if stats is not None else part_ml.data_ptr()
    if trace is not None:
        stats_ptr = trace.data_ptr()
    compiled(
        q.data_ptr(),
        kv_pool.data_ptr(),
        table.data_ptr(),
        cu_q.data_ptr(),
        kv_lens.data_ptr(),
        out.data_ptr(),
        plan.units.data_ptr(),
        plan.cta_off.data_ptr(),
        part_o.data_ptr(),
        part_ml.data_ptr(),
        stats_ptr,
        scale_log2.data_ptr(),
        scale_out.data_ptr(),
        q.shape[0],
        batch,
        kv_pool.shape[0],
        table.stride(0),
        table.stride(1),
        max(plan.num_units, 1),
        plan.num_ctas,
        cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream),
    )
    if plan.merge_meta is not None:
        kern = _merge_kernel()
        kern[(plan.merge_meta.shape[0], plan.num_inst)](
            out,
            stats if stats is not None else part_ml,
            part_o,
            part_ml,
            plan.merge_meta,
            plan.merge_slots,
            cu_q,
            plan.max_pieces,
            H=heads,
            NI=plan.num_inst,
            DH=HEAD_DIM,
            MT=M_TILE,
            WRITE_STATS=stats is not None,
            num_warps=4,
        )
    return out
