# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-split KDA with a serial FP32 hand-off: the step plan of a context-parallel (cep) rank.

A cep rank holds rows ``[r * rows, (r + 1) * rows)`` of the step. Instead of Ulysses (an all-to-all to
a head split over every row and back), every rank runs the short convolution and the recurrence of
all heads over its own rows, the engine route of a single-rank layer, and a request that spans ranks
hands its state from rank to rank:

* A request's fragment on its first rank starts from its cached state (or zero). A later fragment
  is always at the start of its rank's rows (the rank's *lead*) and starts from the FP32 end state
  of the request's rows on the rank before: the state the chain itself carries, so the split is
  exact up to the 16-token chunk grid restarting at the rank boundary.
* The convolution of a lead starts from the request's three raw rows before the rank's first row:
  a halo from the previous rank, or the request's cached window where it started fewer than three
  rows before.

The KDA pools keep their head-sharded layout (rank ``j`` holds heads ``[j H / N, (j + 1) H / N)``
of every request, the layout the generation server's transfer reads). A layer, every exchange a
push (``ops/comm/rubin_cp_push``) of rows of ``width`` bytes:

1. x1: the owners' slices of every cached request's state and window to the rank of its first
   fragment (and to the next rank when the request started fewer than three rows before it), and
   each rank's last three raw rows to the next rank when that rank has a lead.
2. The convolution of every fragment, then one launch over the fragments that start a request.
3. Hand-off rounds ``j < max_span - 1``: in round ``j`` the ranks ``j`` ranks after their request's
   first rank send their end state to the next rank, which then runs its lead from it. With
   ``prep_ahead`` the lead's chunk factors (the prefill's first kernel, which reads no state) are
   computed before the rounds, beside the other ranks' work, and the round runs only the state
   recursion: the same kernels on the same inputs, so the same bits, with a shorter serial chain
   (leads of ``PREP_AHEAD_MIN_ROWS`` rows or more).
4. x3: the end states and windows of the requests that end on a rank, and its block-reuse
   snapshots, to the owners of their heads.

The plan is host arithmetic over the step's replicated metadata, the same for every KDA layer of
the step; its index vectors go to the device in one copy. A port of the serial hand-off of an earlier
CEP16 implementation onto the family's kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

from ...kernels.ssm.rubin_kda_prefill_ops import KDA_PREFILL_SNAPSHOT_CHUNK, seed_cu_seqlens
from ..comm.rubin_cp_push import CpPushBlock, CpPushWorkspace, Push, push_chunks, rubin_cp_push
from .rubin_kda_prefill import rubin_kda_prefill, rubin_kda_prefill_workspace
from .rubin_kda_prefill_conv import rubin_kda_prefill_conv

# Raw rows of a convolution history (the window width minus one).
HALO = 3
_HEAD_DIM = 128
# ``prep_ahead`` splits a lead of at least this many rows: below it the factor kernel is shorter
# than the second launch's host time, which lands on the serial chain (cep8: a split is slower
# on short leads and faster on leads of a few thousand rows or more in our measurements).
PREP_AHEAD_MIN_ROWS = 2048


@dataclass
class Fragment:
    """Request ``req``'s rows ``[lo, hi)`` of this rank (local rows)."""

    req: int
    lo: int
    hi: int
    first: bool  # the request starts on this rank
    last: bool  # the request ends on this rank


@dataclass
class Launch:
    """One ``rubin_kda_prefill`` call: local rows ``[lo, hi)`` split into sequences."""

    lo: int
    hi: int
    lens: List[int]
    pool_rows: List[int]
    snapshot_rows: Optional[List[int]] = None  # -1: none
    snapshot_chunks: Optional[List[int]] = None
    # device views (``_upload``)
    cu: Optional[torch.Tensor] = None
    order: Optional[torch.Tensor] = None
    cu_chunks: Optional[torch.Tensor] = None
    idx: Optional[torch.Tensor] = None
    snap_rows_t: Optional[torch.Tensor] = None
    snap_chunks_t: Optional[torch.Tensor] = None


@dataclass
class KdaHandoffPlan:
    """One rank's share of a step's sequence-split KDA (the same for every KDA layer)."""

    world: int
    rank: int
    rows: int
    live: int
    heads: int
    max_span: int  # ranks covered by the step's longest request
    max_frag_len: int
    frags: List[Fragment]
    lead: Optional[Fragment]
    # temporary FP32 state rows and BF16 window rows of this rank
    n_ssm_rows: int
    n_conv_rows: int
    # x1: the cached requests' state and window records (+ the halo rows)
    x1_skip: bool
    x1_send_req: List[int]  # batch indices of the records this rank sends, destination-major
    x1_halo_to_next: bool
    x1_halo_cut: int  # records sent to destinations <= rank + 1 (the halo follows them)
    x1_n_rec: int  # records received from every source
    x1_halo_from_prev: bool
    conv_slots: List[int]
    conv_has_init: List[bool]
    lead_halo_init_row: (
        int  # record row of the lead's cached window when it started < 3 rows before
    )
    lead_req_rows_before: int  # rows of the lead request before this rank's first row
    launch1: Optional[Launch]
    launch2: List[Launch]  # the lead, split at a snapshot point that is not chunk-aligned
    lead_row: int
    zero_rows: List[int]  # temp state rows of first fragments without history
    # x3: end states and snapshots to the owners
    x3_items_ssm: List[int]  # temp state rows: ends, then snapshots
    x3_items_conv: List[int]  # temp window rows of the ends (snapshot windows are appended)
    x3_snap_local: List[Tuple[int, int, str]]  # (snapshot, local point, "ckpt" | "split" | "end")
    x3_recv_counts: List[Tuple[int, int]]  # per source: (ends, snapshots)
    num_requests: int
    num_snapshots: int
    snap_ssm_rows: List[int] = field(default_factory=list)
    p2p_send_row: int = -1  # temp state row sent to rank + 1
    width: int = 0  # bytes of a pushed row
    push_x1: List[Push] = field(default_factory=list)
    push_x3: List[Push] = field(default_factory=list)
    push_rounds: List[Push] = field(default_factory=list)
    send_round: int = -1
    recv_round: int = -1
    # device views
    x1_send_idx: Optional[torch.Tensor] = None
    conv_starts: Optional[torch.Tensor] = None
    conv_slots_t: Optional[torch.Tensor] = None
    conv_has_init_t: Optional[torch.Tensor] = None
    x3_items_ssm_t: Optional[torch.Tensor] = None
    x3_items_conv_t: Optional[torch.Tensor] = None
    zero_rows_t: Optional[torch.Tensor] = None
    push_x1_t: List[torch.Tensor] = field(default_factory=list)
    push_x3_t: List[torch.Tensor] = field(default_factory=list)
    push_rounds_t: List[torch.Tensor] = field(default_factory=list)
    # pool slots (int64) of the records sent, of every request and of the snapshots
    slots_send: Optional[torch.Tensor] = None
    slots_all: Optional[torch.Tensor] = None
    snap_slots: Optional[torch.Tensor] = None

    def record_bytes(self) -> Tuple[int, int, int]:
        """(state slice, window slice, halo) bytes of the exchange records."""
        hd = self.heads // self.world * _HEAD_DIM
        return hd * _HEAD_DIM * 4, 3 * hd * HALO * 2, HALO * 3 * self.heads * _HEAD_DIM * 2


def _bounds(lens: Sequence[int]) -> List[int]:
    out = [0]
    for n in lens:
        out.append(out[-1] + int(n))
    return out


def max_span(lengths: Sequence[int], world: int) -> int:
    """Ranks covered by the longest request of a step of ``lengths`` rows over ``world`` ranks."""
    s = _bounds(lengths)
    rows = -(-s[-1] // world)
    if rows == 0:
        return 0
    return max(
        (((s[i + 1] - 1) // rows) - s[i] // rows + 1 for i in range(len(lengths))), default=0
    )


def kda_handoff_plan(
    lengths: Sequence[int],
    has_init: Sequence[bool],
    snapshots: Sequence[Tuple[int, int]],
    world: int,
    rank: int,
    heads: int,
    slots: torch.Tensor,
    region: int,
    snapshot_slots: Optional[torch.Tensor] = None,
) -> KdaHandoffPlan:
    """Plan rank ``rank``'s share of a step of context requests ``lengths`` (new tokens, in row
    order) over ``world`` ranks of ``heads`` KDA heads.

    ``has_init[b]``: request ``b`` continues a cached state and window. ``snapshots``: the
    ``(request, token offset)`` block-reuse snapshots, by request (at most one each, offsets
    multiples of ``KDA_PREFILL_SNAPSHOT_CHUNK``). ``slots``: the requests' pool slots on the device.
    ``region``: bytes of a receive region of the push workspace (an exchange larger than one region
    runs as several pushes); ``snapshot_slots``: the snapshots' pool slots on the device (int64, in
    ``snapshots`` order). The step needs at least ``HALO`` rows per rank.
    """
    n = len(lengths)
    s = _bounds(lengths)
    num_tokens = s[-1]
    rows = -(-num_tokens // world)
    if rows < HALO:
        raise ValueError(f"the KDA hand-off needs at least {HALO} rows per rank, got {rows}")
    D = _HEAD_DIM
    a = [s[i] // rows for i in range(n)]
    b = [(s[i + 1] - 1) // rows for i in range(n)]
    t = max(0, min(rows, num_tokens - rank * rows))
    lo_r = rank * rows

    frags = [
        Fragment(
            i, max(s[i], lo_r) - lo_r, min(s[i + 1], lo_r + t) - lo_r, a[i] == rank, b[i] == rank
        )
        for i in range(n)
        if a[i] <= rank <= b[i]
    ]
    lead = frags[0] if frags and not frags[0].first else None

    # x1: records (an owner's slices of a request's cached state and window) to the rank of the
    # request's first fragment, and to the next rank when the request started fewer than HALO rows
    # before it.
    def recs_to(d: int) -> List[int]:
        out = [i for i in range(n) if has_init[i] and a[i] == d]
        if d >= 1:
            out += [
                i
                for i in range(n)
                if has_init[i] and a[i] == d - 1 and b[i] >= d and s[i] > d * rows - HALO
            ]
        return out

    def has_lead(d: int) -> bool:
        return d >= 1 and any(a[i] < d <= b[i] for i in range(n))

    recs = [recs_to(d) for d in range(world)]
    x1_send_req = [i for d in range(world) for i in recs[d]]
    halo_to_next = rank + 1 < world and has_lead(rank + 1)
    halo_from_prev = has_lead(rank)
    hd = heads // world * D
    ssm_b = hd * D * 4
    conv_b = 3 * hd * HALO * 2
    rec_b = ssm_b + conv_b
    halo_b = HALO * 3 * heads * D * 2
    st_b = heads * D * D * 4
    n_rec = len(recs[rank])
    x1_skip = not any(recs) and not any(has_lead(d) for d in range(world))
    halo_cut = sum(len(recs[d]) for d in range(min(rank + 2, world)))

    # Temporary pools: rows [0, n_rec) hold the records (cached states / windows, record order).
    rec_row = {}
    for j, i in enumerate(recs[rank]):
        rec_row.setdefault(i, j)
    n_ssm = n_rec
    n_conv = n_rec

    def new_ssm() -> int:
        nonlocal n_ssm
        n_ssm += 1
        return n_ssm - 1

    def new_conv() -> int:
        nonlocal n_conv
        n_conv += 1
        return n_conv - 1

    conv_slots, conv_has_init = [], []
    lead_halo_init_row, lead_rows_before = -1, 0
    first_rows = {}  # request -> temp state row of its first fragment
    zero_rows = []
    for f in frags:
        if f is lead:
            conv_slots.append(new_conv())
            conv_has_init.append(True)
            lead_rows_before = lo_r - s[f.req]
            if lead_rows_before < HALO and has_init[f.req]:
                lead_halo_init_row = [j for j, i in enumerate(recs[rank]) if i == f.req][-1]
            continue
        if has_init[f.req]:
            conv_slots.append(rec_row[f.req])
            conv_has_init.append(True)
            first_rows[f.req] = rec_row[f.req]
        else:
            conv_slots.append(new_conv())
            conv_has_init.append(False)
            first_rows[f.req] = new_ssm()
            zero_rows.append(first_rows[f.req])

    # Snapshots: the rank of the token before the point stores it.
    snap_here = []  # (snapshot index, request, local point)
    for j, (i, off) in enumerate(snapshots):
        p = s[i] + off
        if (p - 1) // rows == rank:
            snap_here.append((j, i, p - lo_r))

    launch1 = None
    rest = [f for f in frags if f is not lead]
    snap_ssm_rows, snap_local = [], []
    if rest:
        snap_r, snap_c = [], []
        for f in rest:
            hit = [(j, p) for j, i, p in snap_here if i == f.req]
            if hit:
                j, p = hit[0]
                row = new_ssm()
                snap_ssm_rows.append(row)
                snap_local.append((j, p, "ckpt"))
                snap_r.append(row)
                snap_c.append((p - f.lo) // KDA_PREFILL_SNAPSHOT_CHUNK)
            else:
                snap_r.append(-1)
                snap_c.append(-1)
        any_snap = any(x >= 0 for x in snap_r)
        launch1 = Launch(
            rest[0].lo,
            rest[-1].hi,
            [f.hi - f.lo for f in rest],
            [first_rows[f.req] for f in rest],
            snap_r if any_snap else None,
            snap_c if any_snap else None,
        )

    launch2 = []
    lead_row = -1
    if lead is not None:
        lead_row = new_ssm()
        hit = [(j, p) for j, i, p in snap_here if i == lead.req]
        if not hit:
            launch2 = [Launch(0, lead.hi, [lead.hi], [lead_row])]
        else:
            j, p = hit[0]
            row = new_ssm()
            snap_ssm_rows.append(row)
            if p == lead.hi:  # the end state of the lead (a middle rank's last row)
                snap_local.append((j, p, "end"))
                launch2 = [Launch(0, lead.hi, [lead.hi], [lead_row])]
            elif p % KDA_PREFILL_SNAPSHOT_CHUNK == 0:
                snap_local.append((j, p, "ckpt"))
                launch2 = [
                    Launch(
                        0,
                        lead.hi,
                        [lead.hi],
                        [lead_row],
                        [row],
                        [p // KDA_PREFILL_SNAPSHOT_CHUNK],
                    )
                ]
            else:  # split at the point: the first part's end state is the snapshot
                snap_local.append((j, p, "split"))
                launch2 = [
                    Launch(0, p, [p], [lead_row]),
                    Launch(p, lead.hi, [lead.hi - p], [lead_row]),
                ]
    order_snap = sorted(range(len(snap_local)), key=lambda x: snap_local[x][0])
    snap_local = [snap_local[x] for x in order_snap]
    snap_ssm_rows = [snap_ssm_rows[x] for x in order_snap]

    # The hand-off this rank sends: the end state of its last fragment's request when that
    # request continues on the next rank (a first fragment, or a middle rank's lead).
    mine = [i for i in range(n) if a[i] <= rank <= b[i]]
    send_req = mine[-1] if mine and b[mine[-1]] > rank else -1
    p2p_send_row = -1
    send_after_lead = False
    if send_req >= 0:
        send_after_lead = a[send_req] < rank
        p2p_send_row = lead_row if send_after_lead else first_rows[send_req]

    # x3: end states (requests that end here) and snapshots taken here, to the owners.
    finals = [f for f in frags if f.last]
    items_ssm = [lead_row if f is lead else first_rows[f.req] for f in finals] + snap_ssm_rows
    items_conv = [conv_slots[frags.index(f)] for f in finals]

    def counts_of(r: int) -> Tuple[int, int]:
        nf = sum(1 for i in range(n) if b[i] == r)
        ns = sum(1 for (i, off) in snapshots if (s[i] + off - 1) // rows == r)
        return nf, ns

    counts = [counts_of(r) for r in range(world)]

    plan = KdaHandoffPlan(
        world=world,
        rank=rank,
        rows=rows,
        live=t,
        heads=heads,
        max_span=max((b[i] - a[i] + 1 for i in range(n)), default=0),
        max_frag_len=max((f.hi - f.lo for f in frags), default=0),
        frags=frags,
        lead=lead,
        n_ssm_rows=n_ssm,
        n_conv_rows=n_conv,
        x1_skip=x1_skip,
        x1_send_req=x1_send_req,
        x1_halo_to_next=halo_to_next,
        x1_halo_cut=halo_cut,
        x1_n_rec=n_rec,
        x1_halo_from_prev=halo_from_prev,
        conv_slots=conv_slots,
        conv_has_init=conv_has_init,
        lead_halo_init_row=lead_halo_init_row,
        lead_req_rows_before=lead_rows_before,
        launch1=launch1,
        launch2=launch2,
        lead_row=lead_row,
        zero_rows=zero_rows,
        x3_items_ssm=items_ssm,
        x3_items_conv=items_conv,
        x3_snap_local=snap_local,
        x3_recv_counts=counts,
        num_requests=n,
        num_snapshots=len(snapshots),
        snap_ssm_rows=snap_ssm_rows,
        p2p_send_row=p2p_send_row,
    )
    plan.snap_slots = snapshot_slots
    _plan_pushes(plan, recs, has_lead, a, counts, rec_b, halo_b, st_b, send_after_lead, region)
    _upload(plan, slots)
    return plan


def _plan_pushes(
    plan: KdaHandoffPlan,
    recs,
    has_lead,
    a,
    counts,
    rec_b: int,
    halo_b: int,
    st_b: int,
    send_after_lead: bool,
    region: int,
) -> None:
    """The push tables of x1, x3 and the hand-off rounds (every rank's splits follow from the
    step, so each rank builds its own tables)."""
    world, rank = plan.world, plan.rank
    width = math.gcd(rec_b, math.gcd(halo_b, st_b))
    if width % 16:
        raise ValueError(f"no 16-byte row for the hand-off pushes ({rec_b}, {halo_b}, {st_b} B)")
    cap = region // width
    if cap * width < st_b:
        raise ValueError(f"a {region} byte push region cannot hold a {st_b} byte state")
    plan.width = width
    if not plan.x1_skip:
        send = [
            len(recs[d]) * rec_b + (halo_b if d == rank + 1 and plan.x1_halo_to_next else 0)
            for d in range(world)
        ]
        send_off = _bounds(send)
        blocks = []
        for d in range(world):
            halo_before = halo_b if has_lead(d) and rank >= d else 0
            d0 = rank * len(recs[d]) * rec_b + halo_before
            blocks.append((send_off[d] // width, d0 // width, send[d] // width))
        recv = [
            (world * len(recs[d]) * rec_b + (halo_b if has_lead(d) else 0)) // width
            for d in range(world)
        ]
        plan.push_x1 = push_chunks(blocks, recv, rank, cap)
    if plan.num_requests:
        items = [(nf + ns) * rec_b // width for nf, ns in counts]
        mine = items[rank]
        d0 = sum(items[:rank])
        blocks = [(d * mine, d0, mine) for d in range(world)]
        plan.push_x3 = push_chunks(blocks, [sum(items)] * world, rank, cap)
    st_rows = st_b // width
    if plan.p2p_send_row >= 0:
        plan.send_round = rank - a[plan.lead.req] if send_after_lead else 0
    if plan.lead is not None:
        plan.recv_round = rank - 1 - a[plan.lead.req]
    for j in range(max(0, plan.max_span - 1)):
        vals = [0] * (3 * world)
        sending = j == plan.send_round
        if sending:
            vals[3 * (rank + 1) + 2] = st_rows
        plan.push_rounds.append(
            (vals, st_rows if sending else 0, st_rows if j == plan.recv_round else 0)
        )


def _upload(plan: KdaHandoffPlan, slots: torch.Tensor) -> None:
    """One host-to-device copy of the plan's index vectors; views for each use."""
    device = slots.device
    i64: List[int] = []
    i32: List[int] = []
    spans = {}

    def put64(key, values):
        spans[key] = (64, len(i64), len(values))
        i64.extend(values)

    def put32(key, values):
        while len(i32) % 4:  # 16-byte aligned starts
            i32.append(0)
        spans[key] = (32, len(i32), len(values))
        i32.extend(values)

    put64("x1_send_req", plan.x1_send_req)
    put64("x3_items_ssm", plan.x3_items_ssm)
    put64("x3_items_conv", plan.x3_items_conv)
    put64("zero_rows", plan.zero_rows)
    put32("conv_starts", [0] + [f.hi for f in plan.frags])
    put32("conv_slots", plan.conv_slots)
    put32("conv_has_init", [int(x) for x in plan.conv_has_init])
    launches = ([plan.launch1] if plan.launch1 is not None else []) + plan.launch2
    for li, la in enumerate(launches):
        order = sorted(range(len(la.lens)), key=lambda x: (-la.lens[x], x))
        put64(("cu", li), _bounds(la.lens))
        put32(("order", li), order)
        put32(("cu_chunks", li), _bounds([(x + 15) // 16 for x in la.lens]))
        put32(("idx", li), la.pool_rows)
        if la.snapshot_rows is not None:
            put32(("snap_rows", li), la.snapshot_rows)
            put32(("snap_chunks", li), la.snapshot_chunks)
    for name in ("push_x1", "push_x3", "push_rounds"):
        for k, (vals, _, _) in enumerate(getattr(plan, name)):
            put32((name, k), vals)
    pin = device.type == "cuda"
    t64 = torch.tensor(i64 or [0], dtype=torch.int64, pin_memory=pin).to(device, non_blocking=True)
    t32 = torch.tensor(i32 or [0], dtype=torch.int32, pin_memory=pin).to(device, non_blocking=True)

    def view(key):
        kind, off, length = spans[key]
        return (t64 if kind == 64 else t32)[off : off + length]

    plan.x1_send_idx = view("x1_send_req")
    plan.x3_items_ssm_t = view("x3_items_ssm")
    plan.x3_items_conv_t = view("x3_items_conv")
    plan.zero_rows_t = view("zero_rows")
    plan.conv_starts = view("conv_starts")
    plan.conv_slots_t = view("conv_slots")
    plan.conv_has_init_t = view("conv_has_init").to(torch.bool)
    for name in ("push_x1", "push_x3", "push_rounds"):
        setattr(plan, name + "_t", [view((name, k)) for k in range(len(getattr(plan, name)))])
    for li, la in enumerate(launches):
        la.cu = view(("cu", li))
        if pin:
            seed_cu_seqlens(la.cu, _bounds(la.lens))
        la.order = view(("order", li))
        la.cu_chunks = view(("cu_chunks", li))
        la.idx = view(("idx", li))
        if la.snapshot_rows is not None:
            la.snap_rows_t = view(("snap_rows", li))
            la.snap_chunks_t = view(("snap_chunks", li))
    n = plan.num_requests
    plan.slots_all = slots[:n].to(torch.int64)
    plan.slots_send = plan.slots_all.index_select(0, plan.x1_send_idx) if plan.x1_send_req else None


# --------------------------------------------------------------------------------------------------
# The layer.
# --------------------------------------------------------------------------------------------------


class _Recurrence(NamedTuple):
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float
    lower_bound: float
    mode: str


def _cp_exchange(
    ws: CpPushWorkspace, send: torch.Tensor, pushes: list, tables: list, width: int
) -> torch.Tensor:
    """An exchange of rows of ``width`` bytes planned by ``push_chunks``: this rank's received
    rows as bytes, in source order. One push returns its region (valid until the next push)."""
    total = sum(got for _, _, got in pushes)
    if len(pushes) == 1:
        (_, most, got), table = pushes[0], tables[0]
        block = CpPushBlock(send, width, 0, width, 0, width, 0, most)
        return rubin_cp_push(ws, (block,), table)[: got * width]
    recv = send.new_empty(total * width)
    done = 0
    for (_, most, got), table in zip(pushes, tables):
        region = rubin_cp_push(ws, (CpPushBlock(send, width, 0, width, 0, width, 0, most),), table)
        recv[done * width : (done + got) * width].copy_(region[: got * width])
        done += got
    return recv


def _launch(
    w: "_Recurrence",
    la: Launch,
    q,
    k,
    v,
    g,
    beta,
    tssm,
    out,
    phase: str = "both",
    workspace: Optional[torch.Tensor] = None,
) -> None:
    """One ``rubin_kda_prefill`` of the hand-off: the local rows ``[la.lo, la.hi)`` from the
    temporary FP32 state rows ``la.idx`` (and their snapshots), output into ``out``; ``phase`` /
    ``workspace`` as the entry's (its ``"prep"`` half, then its ``"chain"`` half)."""
    sl = slice(la.lo, la.hi)
    rubin_kda_prefill(
        q=q[:, sl],
        k=k[:, sl],
        v=v[:, sl],
        g=g[:, sl],
        beta=beta[:, sl],
        state_pool=tssm,
        state_indices=la.idx,
        cu_seqlens=la.cu,
        A_log=w.a_log,
        dt_bias=w.dt_bias,
        scale=w.scale,
        lower_bound=w.lower_bound,
        mode=w.mode,
        seq_order=la.order,
        cu_chunks=la.cu_chunks,
        snapshot_rows=la.snap_rows_t,
        snapshot_chunks=la.snap_chunks_t,
        out=out[:, sl],
        phase=phase,
        workspace=workspace,
    )


def _prep_lead(w: "_Recurrence", hp: KdaHandoffPlan, q, k, v, g, beta, tssm, out) -> list:
    """The state-free half of the lead's launches (``hp.launch2``): their chunk factors, into one
    workspace per launch, which ``_run_lead`` hands to their state recursions."""
    spaces = []
    for la in hp.launch2:
        ws = rubin_kda_prefill_workspace(q[:, la.lo : la.hi], tssm, la.idx, w.lower_bound)
        _launch(w, la, q, k, v, g, beta, tssm, out, "prep", ws)
        spaces.append(ws)
    return spaces


def _run_lead(
    w: "_Recurrence", hp: KdaHandoffPlan, q, k, v, g, beta, tssm, out, snaps, spaces=None
) -> None:
    """The lead from its handed-off state (``hp.launch2``), and the snapshots taken in it;
    ``spaces``: the workspaces of ``_prep_lead`` (None: whole launches)."""
    for li, la in enumerate(hp.launch2):
        if spaces is None:
            _launch(w, la, q, k, v, g, beta, tssm, out)
        else:
            _launch(w, la, q, k, v, g, beta, tssm, out, "chain", spaces[li])
        if li == 0 and len(hp.launch2) == 2:  # the first part ended at the snapshot point
            for (_, _, mode), row in snaps:
                if mode == "split":
                    tssm[row].copy_(tssm[hp.lead_row])
    for (_, _, mode), row in snaps:
        if mode == "end":
            tssm[row].copy_(tssm[hp.lead_row])


def kda_handoff_layer(
    hp: KdaHandoffPlan,
    ws: CpPushWorkspace,
    qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    conv_pool: torch.Tensor,
    scale: float,
    lower_bound: float,
    mode: str,
    prep_ahead: bool = False,
) -> torch.Tensor:
    """One KDA layer's convolution and recurrence of all heads over this rank's live rows, with
    the hand-off's exchanges pushed on ``ws``; returns ``[live, H, D]`` BF16.

    ``qkv`` ``[rows, >= 3 H D]``: the raw ``[q | k | v]`` of this rank's rows (heads in order);
    ``g`` ``[rows, H D]``: their forget-gate logits; ``beta`` ``[rows, H]``: their beta logits;
    ``conv_weight`` ``[3 H D, 4]``, ``a_log`` ``[H]``, ``dt_bias`` ``[H D]``: the layer's
    recurrence weights of all heads; ``state_pool`` ``[slots, H / N, D, D]`` FP32 and ``conv_pool``
    ``[slots, 3 H D / N, 3]`` BF16: this rank's head shard of the KDA pools (the end states,
    windows and snapshots of the step's requests land there). Every rank of the group calls it
    with its plan of the same step. ``prep_ahead``: the lead's chunk factors before the rounds
    (module docstring, step 3); the results are the same bits either way.
    """
    world, rank, rows, t = hp.world, hp.rank, hp.rows, hp.live
    H, D = hp.heads, _HEAD_DIM
    P = H * D
    w = _Recurrence(a_log, dt_bias, scale, lower_bound, mode)
    hl = H // world
    hd = hl * D
    ssm_b, conv_b, halo_b = hp.record_bytes()
    rec_b = ssm_b + conv_b
    width = hp.width
    tssm = qkv.new_empty((max(hp.n_ssm_rows, 1), H, D, D), dtype=torch.float32)
    tconv = qkv.new_empty((max(hp.n_conv_rows, 1), 3 * P, HALO))
    halo_rows = None

    # x1: the cached states and windows of the requests that start here; the halo of the lead.
    n_rec = hp.x1_n_rec
    if not hp.x1_skip:
        parts = []
        n_send = len(hp.x1_send_req)
        if n_send:
            rec = torch.cat(
                (
                    state_pool.index_select(0, hp.slots_send).view(n_send, -1).view(torch.uint8),
                    conv_pool.index_select(0, hp.slots_send).view(n_send, -1).view(torch.uint8),
                ),
                dim=1,
            )
        if hp.x1_halo_to_next:
            halo = qkv[rows - HALO : rows, : 3 * P].contiguous().view(-1).view(torch.uint8)
            if n_send:
                parts = [rec[: hp.x1_halo_cut].view(-1), halo, rec[hp.x1_halo_cut :].view(-1)]
            else:
                parts = [halo]
        elif n_send:
            parts = [rec.view(-1)]
        send = torch.cat(parts) if len(parts) > 1 else (parts[0] if parts else tssm.view(-1))
        recv = _cp_exchange(ws, send.view(torch.uint8), hp.push_x1, hp.push_x1_t, width)
        cut = rank * n_rec * rec_b
        halo_in = halo_b if hp.x1_halo_from_prev else 0
        if n_rec:
            dst = tssm[:n_rec].view(n_rec, H * D * D).view(torch.uint8).view(n_rec, world, ssm_b)
            cdst = tconv[:n_rec].view(n_rec, 3, world, hd * HALO)
            # the sources before this rank, then (after the previous rank's halo) the others
            for lo, hi, base in ((0, rank, 0), (rank, world, cut + halo_in)):
                if hi == lo:
                    continue
                part = recv[base : base + (hi - lo) * n_rec * rec_b].view(hi - lo, n_rec, rec_b)
                dst[:, lo:hi].copy_(part[:, :, :ssm_b].transpose(0, 1))
                cdst[:, :, lo:hi].copy_(
                    part[:, :, ssm_b:]
                    .contiguous()
                    .view(torch.bfloat16)
                    .view(hi - lo, n_rec, 3, hd * HALO)
                    .permute(1, 2, 0, 3)
                )
        if hp.lead is not None:
            # The lead's window: the three raw rows before this rank's first row, oldest first.
            halo_rows = recv[cut : cut + halo_in].view(torch.bfloat16).view(HALO, 3 * P)
            lead_conv = tconv[hp.conv_slots[0]]
            lead_conv.copy_(halo_rows.t())
            short = HALO - hp.lead_req_rows_before
            if short > 0:  # the request started fewer rows before: those taps are its window's
                if hp.lead_halo_init_row >= 0:
                    lead_conv[:, :short].copy_(tconv[hp.lead_halo_init_row][:, HALO - short :])
                else:
                    lead_conv[:, :short].zero_()
            # The halo is read again by the snapshot windows, after later pushes.
            halo_rows = halo_rows.clone()

    out = qkv.new_empty((1, t, H, D))
    q = k = v = gg = bb = None
    if t:
        q, k, v = rubin_kda_prefill_conv(
            src=qkv[:t],
            weight=conv_weight,
            conv_state=tconv,
            starts=hp.conv_starts,
            slots=hp.conv_slots_t,
            has_init=hp.conv_has_init_t,
            max_seqlen=hp.max_frag_len,
        )
        gg = g[:t].view(1, t, H, D)
        bb = beta[:t].unsqueeze(0)

    # Launch 1: the fragments that start a request (from the cached state or zero).
    if hp.zero_rows:
        tssm.index_fill_(0, hp.zero_rows_t, 0.0)
    if hp.launch1 is not None:
        _launch(w, hp.launch1, q, k, v, gg, bb, tssm, out)
    # The lead's chunk factors read no state: computed now, the round waits only for the recursion.
    spaces = None
    lead_rows = sum(la.hi - la.lo for la in hp.launch2)
    if prep_ahead and hp.lead is not None and t and lead_rows >= PREP_AHEAD_MIN_ROWS:
        spaces = _prep_lead(w, hp, q, k, v, gg, bb, tssm, out)

    # The hand-off rounds: every rank runs every round; the lead runs right after the round that
    # brings its state.
    snaps = list(zip(hp.x3_snap_local, hp.snap_ssm_rows))
    st_b = H * D * D * 4
    for j, ((_, most, got), table) in enumerate(zip(hp.push_rounds, hp.push_rounds_t)):
        src = tssm[hp.p2p_send_row] if j == hp.send_round else tssm[0]
        block = CpPushBlock(src, width, 0, width, 0, width, 0, most)
        region = rubin_cp_push(ws, (block,), table)
        if got:
            tssm[hp.lead_row].view(-1).view(torch.uint8).copy_(region[:st_b])
            _run_lead(w, hp, q, k, v, gg, bb, tssm, out, snaps, spaces)
    if hp.lead is not None and not hp.push_rounds:
        raise AssertionError("a lead without a hand-off round")

    # x3: end states, windows and snapshots to the owners of their heads.
    if hp.num_requests:
        n_fin = len(hp.x3_items_conv)
        n_it = n_fin + len(hp.x3_snap_local)
        if n_it:
            items = tssm.index_select(0, hp.x3_items_ssm_t)
            conv_items = [tconv.index_select(0, hp.x3_items_conv_t)] if n_fin else []
            for _, p, _ in hp.x3_snap_local:
                # the snapshot's window: the three raw rows before local point p
                if p >= HALO:
                    win = qkv[p - HALO : p, : 3 * P]
                else:
                    win = torch.cat((halo_rows[p:], qkv[:p, : 3 * P]))
                conv_items.append(win.t().unsqueeze(0))
            citems = torch.cat(conv_items)
            ssm_part = items.view(n_it, world, hl * D * D).transpose(0, 1).contiguous()
            conv_part = citems.view(n_it, 3, world, hd * HALO).permute(2, 0, 1, 3).contiguous()
            send = torch.cat(
                (
                    ssm_part.view(world, -1).view(torch.uint8),
                    conv_part.view(world, -1).view(torch.uint8),
                ),
                dim=1,
            ).view(-1)
        else:
            send = tssm.view(-1).view(torch.uint8)
        recv = _cp_exchange(ws, send, hp.push_x3, hp.push_x3_t, width)
        fin_ssm, fin_conv, snap_ssm, snap_conv = [], [], [], []
        off = 0
        for nf, ns in hp.x3_recv_counts:
            m = nf + ns
            seg = recv[off : off + m * rec_b]
            off += m * rec_b
            if nf:
                fin_ssm.append(seg[: nf * ssm_b])
                fin_conv.append(seg[m * ssm_b : m * ssm_b + nf * conv_b])
            if ns:
                snap_ssm.append(seg[nf * ssm_b : m * ssm_b])
                snap_conv.append(seg[m * ssm_b + nf * conv_b : m * rec_b])
        nreq = hp.num_requests
        slots = hp.slots_all
        state_pool.index_copy_(
            0, slots, torch.cat(fin_ssm).view(torch.float32).view(nreq, hl, D, D)
        )
        conv_pool.index_copy_(
            0, slots, torch.cat(fin_conv).view(torch.bfloat16).view(nreq, 3 * hd, HALO)
        )
        if hp.num_snapshots:
            ns = hp.num_snapshots
            snap_slots = hp.snap_slots
            state_pool.index_copy_(
                0, snap_slots, torch.cat(snap_ssm).view(torch.float32).view(ns, hl, D, D)
            )
            conv_pool.index_copy_(
                0,
                snap_slots,
                torch.cat(snap_conv).view(torch.bfloat16).view(ns, 3 * hd, HALO),
            )
    return out[0]


__all__ = [
    "HALO",
    "Fragment",
    "KdaHandoffPlan",
    "Launch",
    "kda_handoff_layer",
    "kda_handoff_plan",
    "max_span",
]
