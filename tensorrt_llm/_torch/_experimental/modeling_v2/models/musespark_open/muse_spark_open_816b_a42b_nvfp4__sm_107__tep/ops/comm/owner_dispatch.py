# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-row pushes of the row-owner MoE island (``row_owner_moe``).

Two transports: ``OwnerExchange`` (below: arrival counters, one launch per exchange that also
waits) and the v2 pair ``OwnerLLDispatch`` / ``OwnerRowPublish`` (the notes before them: LL
words polled by an unpack launch, and a Lamport publication polled in the FFN join).

In the island, rank ``r`` owns rows ``t = ranks * i + r`` of a decode step of ``R`` rows
(the owner map of the wide hop's and the expert finalize's two-shots). After the owner has
run the per-row work of its ``R / ranks`` rows (the MoE prologue; the FFN join and the next
layer's input norm) every rank needs every row again: the expert call needs the NVFP4
payload, its scales and the top-8 selection of all ``R`` rows (it runs EP over the rank's
local experts), and the next layer's QKV GEMM needs every normalized row (TP over heads).
This module is that all-gather, as a push from the owner: one kernel per exchange stores
the owner's compact rows ``i`` into row ``t`` of the segment's receive area on every rank
(its own included), signals each peer, and its program 0 waits until every peer's rows of
the same call arrived, so the consumer launched after it reads a complete ``[R, ...]``
tensor straight out of the local receive memory.

Ordering (no sentinel: an NVFP4 payload word may hold any bits). Each program stores its
row with weak stores, joins its CTA (``bar.sync``), and one thread adds to the local done
word with a system-scope acq_rel add (every call adds exactly ``STEP``: program 0 adds
``STEP + 1 - programs``, every other program one, so the add that completes a multiple of
``STEP`` is the call's last). The last program alone adds ``STEP`` to each peer's arrival
word for this source with a system-scope release add, then reads its own arrival words
with system-scope acquire loads until each peer's reaches ``(n + 1) * STEP``, where ``n``
is the call's index (the done word's ``STEP`` place). Its acquire of every program's add
and its release to the peers order every program's row stores before a peer's acquire;
the kernel's completion then orders every arrived row before the consumer. One remote add
per peer and call (one add per program would serialize ``R / ranks`` remote adds on one
word).

Call index and buffer. ``n`` agrees on every rank (calls are issued in the same order
everywhere). The receive buffer is the call's parity, chosen
on the host (``parity``) so a consumer captured in a CUDA graph reads a fixed address; a
graph must hold an even number of calls of one exchange (the island makes one call per
layer, 62 per step), and the host counter alternates eagerly the same way on every rank.
Buffer reuse: rank ``o`` writes call ``n + 2`` into rank ``r``'s parity-``n`` buffer only
after its call ``n + 1`` completed, which waited for ``r``'s call-``n + 1`` arrival, which
``r`` released only after its consumer of call ``n`` completed (stream order). So no row
is rewritten while it is still read.

Calls on one exchange must form one ordered chain per rank (one stream), with the same
row counts and segment counts in the same order on every rank.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm.mapping import Mapping

BUFFERS = 2
#: What one call adds per source to an arrival word (and to the issue word), whatever its
#: program count (at most ``STEP``: one program per owner row, up to 1024 / 4 rows).
STEP_SHIFT = 10
STEP = 1 << STEP_SHIFT
#: Control words at the start of each rank's receive memory: the issue counter, then one
#: arrival word per source rank, 32 bytes apart.
ISSUE_WORD = 0
ARRIVE_WORD = 32
ARRIVE_STRIDE = 8
CONTROL_WORDS = 128
ISSUE: tl.constexpr = tl.constexpr(ISSUE_WORD)
ARRIVE: tl.constexpr = tl.constexpr(ARRIVE_WORD)
ASTRIDE: tl.constexpr = tl.constexpr(ARRIVE_STRIDE)
MAX_SEGMENTS = 4
#: Owners (the TP group whose rows are pushed) and holders (the ranks receiving every row).
#: The island pushes TP4 owners -> the same 4 ranks; the destinations are addressed up to 8
#: holders so an owner -> EP8 push is the same kernel with ``HOLDERS = 8`` (its
#: non-owner holders then run a wait-only call, not built yet).
MAX_RANKS = 4
MAX_HOLDERS = 8


@triton.jit
def _push_segment(
    src,
    off,
    parity,
    pid,
    row,
    peer0,
    peer1,
    peer2,
    peer3,
    peer4,
    peer5,
    peer6,
    peer7,
    W: tl.constexpr,
    BUF: tl.constexpr,
    HOLDERS: tl.constexpr,
):
    cols = tl.arange(0, W)
    data = tl.load(src + pid.to(tl.int64) * W + cols)
    dst = off + parity * BUF + row.to(tl.int64) * W + cols
    tl.store(peer0 + dst, data)
    tl.store(peer1 + dst, data)
    if HOLDERS > 2:
        tl.store(peer2 + dst, data)
        tl.store(peer3 + dst, data)
    if HOLDERS > 4:
        tl.store(peer4 + dst, data)
        tl.store(peer5 + dst, data)
        tl.store(peer6 + dst, data)
        tl.store(peer7 + dst, data)


@triton.jit
def _signal(peer, word, amount, p: tl.constexpr, rank):
    if p != rank:
        tl.atomic_add(peer + word, amount, sem="release", scope="sys")


@triton.jit
def _await(local, word, expected, p: tl.constexpr, rank):
    if p != rank:
        cur = tl.atomic_add(local + word, 0, sem="acquire", scope="sys")
        while cur - expected < 0:
            cur = tl.atomic_add(local + word, 0, sem="acquire", scope="sys")


@triton.jit
def _owner_push_kernel(
    src0,
    src1,
    src2,
    src3,
    peer0,
    peer1,
    peer2,
    peer3,
    peer4,
    peer5,
    peer6,
    peer7,
    off0,
    off1,
    off2,
    off3,
    rank,
    programs,
    parity,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    W3: tl.constexpr,
    BUF0: tl.constexpr,
    BUF1: tl.constexpr,
    BUF2: tl.constexpr,
    BUF3: tl.constexpr,
    NSEG: tl.constexpr,
    RANKS: tl.constexpr,
    HOLDERS: tl.constexpr,
    STEP: tl.constexpr,
    PDL: tl.constexpr,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    amount = tl.where(pid == 0, STEP + 1 - programs, 1)
    local = peer0
    if rank == 1:
        local = peer1
    if rank == 2:
        local = peer2
    if rank == 3:
        local = peer3
    row = pid * RANKS + rank
    _push_segment(
        src0,
        off0,
        parity,
        pid,
        row,
        peer0,
        peer1,
        peer2,
        peer3,
        peer4,
        peer5,
        peer6,
        peer7,
        W0,
        BUF0,
        HOLDERS,
    )
    if NSEG > 1:
        _push_segment(
            src1,
            off1,
            parity,
            pid,
            row,
            peer0,
            peer1,
            peer2,
            peer3,
            peer4,
            peer5,
            peer6,
            peer7,
            W1,
            BUF1,
            HOLDERS,
        )
    if NSEG > 2:
        _push_segment(
            src2,
            off2,
            parity,
            pid,
            row,
            peer0,
            peer1,
            peer2,
            peer3,
            peer4,
            peer5,
            peer6,
            peer7,
            W2,
            BUF2,
            HOLDERS,
        )
    if NSEG > 3:
        _push_segment(
            src3,
            off3,
            parity,
            pid,
            row,
            peer0,
            peer1,
            peer2,
            peer3,
            peer4,
            peer5,
            peer6,
            peer7,
            W3,
            BUF3,
            HOLDERS,
        )
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    # Every thread's stores of this row precede one system-scope acq_rel add on the local
    # done word (CTA barrier, then one thread); the program whose add completes the call's
    # STEP is the last: it alone releases to every peer (one remote add per peer and call,
    # instead of one per program -- remote same-address adds serialize at the receiver)
    # and then waits for the peers' arrivals.
    tl.debug_barrier()
    old = tl.atomic_add(local + ISSUE, amount, sem="acq_rel", scope="sys")
    call = old // STEP
    last = ((old + amount) & (STEP - 1)) == 0
    if last:
        _signal(peer0, ARRIVE + ASTRIDE * rank, STEP, 0, rank)
        _signal(peer1, ARRIVE + ASTRIDE * rank, STEP, 1, rank)
        if HOLDERS > 2:
            _signal(peer2, ARRIVE + ASTRIDE * rank, STEP, 2, rank)
            _signal(peer3, ARRIVE + ASTRIDE * rank, STEP, 3, rank)
        if HOLDERS > 4:
            _signal(peer4, ARRIVE + ASTRIDE * rank, STEP, 4, rank)
            _signal(peer5, ARRIVE + ASTRIDE * rank, STEP, 5, rank)
            _signal(peer6, ARRIVE + ASTRIDE * rank, STEP, 6, rank)
            _signal(peer7, ARRIVE + ASTRIDE * rank, STEP, 7, rank)
        expected = (call + 1) * STEP
        _await(local, ARRIVE + ASTRIDE * 0, expected, 0, rank)
        _await(local, ARRIVE + ASTRIDE * 1, expected, 1, rank)
        if RANKS > 2:
            _await(local, ARRIVE + ASTRIDE * 2, expected, 2, rank)
            _await(local, ARRIVE + ASTRIDE * 3, expected, 3, rank)


@triton.jit
def _owner_join_push_kernel(
    res_out,
    residual,
    branch,
    gain,
    alpha,
    beta,
    eps,
    next_weight,
    next_eps,
    peer0,
    peer1,
    peer2,
    peer3,
    peer4,
    peer5,
    peer6,
    peer7,
    off0,
    off1,
    rank,
    programs,
    parity,
    BUF0: tl.constexpr,
    BUF1: tl.constexpr,
    NSEG: tl.constexpr,
    RANKS: tl.constexpr,
    HOLDERS: tl.constexpr,
    STEP: tl.constexpr,
    HIDDEN: tl.constexpr,
    PDL: tl.constexpr,
):
    """The served FFN-side join (``_gated_residual_after_rmsnorm_kernel`` with ``HAS_GAIN``,
    ``FUSE_NEXT_NORM``, not ``NORM_FP32_RESIDUAL``), statement for statement, on the owner's
    compact row, whose normalized next-layer input (segment 0, BF16) and, with ``NSEG == 2``,
    updated FP32 residual (segment 1) are stored straight into row ``4 i + rank`` of every
    rank's receive area; then the exchange's arrival protocol."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    offs = tl.arange(0, HIDDEN)
    base = pid.to(tl.int64) * HIDDEN + offs
    b = tl.load(branch + base)
    scaled = b.to(tl.float32)
    g = tl.load(gain + offs).to(tl.float32)
    scaled = (scaled * g).to(b.dtype).to(tl.float32)
    variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
    normed = (scaled * tl.rsqrt(variance + eps)).to(b.dtype).to(tl.float32)
    r = tl.load(residual + base).to(tl.float32)
    a = tl.load(alpha + offs)
    be = tl.load(beta + offs)
    updated = a * r + be * normed
    tl.store(res_out + base, updated)
    next_input = updated.to(b.dtype).to(tl.float32)
    next_variance = tl.sum(next_input * next_input, axis=0) / HIDDEN
    nw = tl.load(next_weight + offs).to(tl.float32)
    nxt = next_input * tl.rsqrt(next_variance + next_eps)
    nxt = (nxt * (nw + 1.0)).to(b.dtype)
    local = peer0
    if rank == 1:
        local = peer1
    if rank == 2:
        local = peer2
    if rank == 3:
        local = peer3
    row = (pid * RANKS + rank).to(tl.int64)
    # Segment 0: the BF16 row (word view: the receive area is int32).
    dst0 = off0 + parity * BUF0 + row * (HIDDEN // 2)
    p0 = (peer0 + dst0).to(tl.pointer_type(tl.bfloat16)) + offs
    p1 = (peer1 + dst0).to(tl.pointer_type(tl.bfloat16)) + offs
    tl.store(p0, nxt)
    tl.store(p1, nxt)
    if HOLDERS > 2:
        p2 = (peer2 + dst0).to(tl.pointer_type(tl.bfloat16)) + offs
        p3 = (peer3 + dst0).to(tl.pointer_type(tl.bfloat16)) + offs
        tl.store(p2, nxt)
        tl.store(p3, nxt)
    if NSEG > 1:
        dst1 = off1 + parity * BUF1 + row * HIDDEN
        q0 = (peer0 + dst1).to(tl.pointer_type(tl.float32)) + offs
        q1 = (peer1 + dst1).to(tl.pointer_type(tl.float32)) + offs
        tl.store(q0, updated)
        tl.store(q1, updated)
        if HOLDERS > 2:
            q2 = (peer2 + dst1).to(tl.pointer_type(tl.float32)) + offs
            q3 = (peer3 + dst1).to(tl.pointer_type(tl.float32)) + offs
            tl.store(q2, updated)
            tl.store(q3, updated)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    amount = tl.where(pid == 0, STEP + 1 - programs, 1)
    tl.debug_barrier()
    old = tl.atomic_add(local + ISSUE, amount, sem="acq_rel", scope="sys")
    call = old // STEP
    last = ((old + amount) & (STEP - 1)) == 0
    if last:
        _signal(peer0, ARRIVE + ASTRIDE * rank, STEP, 0, rank)
        _signal(peer1, ARRIVE + ASTRIDE * rank, STEP, 1, rank)
        if HOLDERS > 2:
            _signal(peer2, ARRIVE + ASTRIDE * rank, STEP, 2, rank)
            _signal(peer3, ARRIVE + ASTRIDE * rank, STEP, 3, rank)
        expected = (call + 1) * STEP
        _await(local, ARRIVE + ASTRIDE * 0, expected, 0, rank)
        _await(local, ARRIVE + ASTRIDE * 1, expected, 1, rank)
        if RANKS > 2:
            _await(local, ARRIVE + ASTRIDE * 2, expected, 2, rank)
            _await(local, ARRIVE + ASTRIDE * 3, expected, 3, rank)


class OwnerExchange:
    """Receive memory, counters and push kernel of one owner exchange for one TP group.

    ``segments`` are the row widths in 32-bit words of the segments a call may push (a call
    pushes a prefix of them). Construction is collective and succeeds or raises on every
    rank alike.
    """

    def __init__(
        self, mapping: Mapping, segments: Sequence[int], *, max_rows: int = 1024, pdl: bool = True
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        if self.ranks not in (2, 4):
            raise ValueError(f"owner exchange serves 2 or 4 ranks, got {self.ranks}")
        if not 1 <= len(segments) <= MAX_SEGMENTS:
            raise ValueError(f"1..{MAX_SEGMENTS} segments, got {len(segments)}")
        for w in segments:
            if w <= 0 or w & (w - 1):
                raise ValueError(f"segment widths are powers of two words, got {w}")
        if max_rows % self.ranks or max_rows // self.ranks > STEP:
            raise ValueError(
                f"max_rows {max_rows}: a multiple of {self.ranks}, at most {STEP * self.ranks}"
            )
        self.segments = [int(w) for w in segments]
        self.max_rows = int(max_rows)
        self.pdl = bool(pdl)
        self.calls = 0
        offsets, words = [], CONTROL_WORDS
        for w in self.segments:
            offsets.append(words)
            words += BUFFERS * self.max_rows * w
        self.offsets = offsets
        self.words = words
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, words * 4, mapping, mapping.is_multi_node()
            )
            self._uc = [
                self._handle.get_uc_buffer(peer, (words,), torch.int32, 0)
                for peer in range(self.ranks)
            ]
            with torch.cuda.device(self.device):
                self._uc[self.rank].zero_()
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "owner exchange workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        self.local = self._uc[self.rank]
        self._peers = list(self._uc) + [self.local] * (MAX_HOLDERS - self.ranks)

    def view(self, segment: int, parity: int, rows: int) -> torch.Tensor:
        """The ``[rows, width]`` int32 receive rows of ``segment`` for buffer ``parity``."""
        w = self.segments[segment]
        base = self.offsets[segment] + parity * self.max_rows * w
        return self.local[base : base + rows * w].view(rows, w)

    def push(
        self, sources: Sequence[torch.Tensor], rows: int, parity: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Push the owner's compact rows of each source (``[rows / ranks, width]``, viewable as
        int32) to every rank; returns the local ``[rows, width]`` int32 receive views."""
        if not 1 <= len(sources) <= len(self.segments):
            raise ValueError("a call pushes a prefix of the segments")
        if rows % self.ranks or not self.ranks <= rows <= self.max_rows:
            raise ValueError(f"rows {rows}: a multiple of {self.ranks} up to {self.max_rows}")
        own = rows // self.ranks
        srcs = []
        for s, src in enumerate(sources):
            v = src.reshape(src.shape[0], -1).view(torch.int32)
            if v.shape[0] < own or v.shape[1] != self.segments[s] or not v.is_contiguous():
                raise ValueError(
                    f"segment {s}: needs [{own}, {self.segments[s]}] int32 words, "
                    f"got {tuple(v.shape)}"
                )
            srcs.append(v)
        if parity is None:
            parity = self.calls & 1
            self.calls += 1
        while len(srcs) < MAX_SEGMENTS:
            srcs.append(srcs[0])
        w = self.segments + [self.segments[0]] * (MAX_SEGMENTS - len(self.segments))
        offs = self.offsets + [self.offsets[0]] * (MAX_SEGMENTS - len(self.offsets))
        _owner_push_kernel[(own,)](
            *srcs,
            *self._peers,
            *offs,
            self.rank,
            own,
            parity,
            W0=w[0],
            W1=w[1],
            W2=w[2],
            W3=w[3],
            BUF0=self.max_rows * w[0],
            BUF1=self.max_rows * w[1],
            BUF2=self.max_rows * w[2],
            BUF3=self.max_rows * w[3],
            NSEG=len(sources),
            RANKS=self.ranks,
            HOLDERS=self.ranks,
            STEP=STEP,
            PDL=self.pdl,
            num_warps=8 if max(w[: len(sources)]) >= 2048 else 4,
            launch_pdl=self.pdl,
        )
        return [self.view(s, parity, rows) for s in range(len(sources))]

    def join_push(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        gain: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        eps: float,
        next_weight: torch.Tensor,
        next_eps: float,
        rows: int,
        with_residual: bool,
        parity: Optional[int] = None,
    ):
        """The FFN join + next norm on the owner's compact rows fused with this exchange's push
        (segments ``[4096, 8192]`` words: the BF16 normalized rows, the FP32 residual rows).
        Returns ``(updated FP32 [rows / ranks, 8192], views)``."""
        if self.segments[:2] != [4096, 8192]:
            raise ValueError("join_push needs segments [4096, 8192]")
        own = rows // self.ranks
        if parity is None:
            parity = self.calls & 1
            self.calls += 1
        out = torch.empty((own, 8192), dtype=torch.float32, device=residual.device)
        nseg = 2 if with_residual else 1
        _owner_join_push_kernel[(own,)](
            out,
            residual,
            branch,
            gain,
            alpha,
            beta,
            eps,
            next_weight,
            next_eps,
            *self._peers,
            self.offsets[0],
            self.offsets[1],
            self.rank,
            own,
            parity,
            BUF0=self.max_rows * 4096,
            BUF1=self.max_rows * 8192,
            NSEG=nseg,
            RANKS=self.ranks,
            HOLDERS=self.ranks,
            STEP=STEP,
            HIDDEN=8192,
            PDL=self.pdl,
            num_warps=16,
            launch_pdl=self.pdl,
        )
        return out, [self.view(s, parity, rows) for s in range(nseg)]


# ------------------------------------------------------------------------------------------
# Transport v2: no arrival counters, no system-scope atomics, no wait kernel of its own.
#
# The exchange above pays a fixed cost per call (a system-scope acq_rel add per program on
# the done word, one remote release add per peer, a spin on remote arrival words, and the
# kernel's completion as the edge before a consumer that cannot poll). The two exchanges below
# put the completion edge into kernels the island runs anyway:
#
# * ``OwnerLLDispatch`` (expert dispatch: NVFP4 payload, scales, top-8 ids and weights): LL
#   protocol. Every 32-bit data word travels with the call's epoch in one 8-byte store; an
#   unpack launch (one program per row, ahead of the expert call) polls its row until every
#   word carries the epoch and writes the dense tensors the expert call reads. Arbitrary NVFP4
#   bits need no sentinel.
# * ``OwnerRowPublish`` (the next layer's normalized input, and the FP32 residual on the
#   layers whose consumers need every row): Lamport sentinel, fused into the FFN join. The
#   "not arrived" words are the all-ones NaNs (BF16 ``0xFFFF``, FP32 ``0xFFFFFFFF``), which no
#   arithmetic result takes: a NaN that the FMA / multiply or the RN conversion to BF16
#   produces is the canonical ``0x7FFF`` / ``0x7FFFFFFF``. Every finite value, infinity and
#   zero of either sign therefore travels unchanged (the join's next norm does produce -0:
#   a gain weight of exactly -1 makes ``w + 1`` zero). The owner still maps an all-ones word to
#   the canonical NaN of its type before it stores or publishes it, so the protocol never
#   depends on that property; the owner keeps the mapped residual too. Then it publishes its
#   row, and the same program polls the rows of its compact index from every rank. The next
#   QKV reads the receive buffer at a fixed address.
#
# Graph-replay safety (what a replay may and may not depend on):
#
# 1. Epoch (dispatch). The rank keeps one device word per parity, ``LL_EPOCH + p``: the epoch
#    of the last call of parity ``p``. A call of parity ``p`` runs epoch ``E + 1``, ``E`` being
#    the other parity's word (the previous call's epoch): the push and every unpack program
#    read only that word, and the unpack's program 0 stores ``E + 1`` into the call's own
#    parity word, which no program of the call reads; the next call (the other parity) reads
#    it after the unpack completed (stream order). No atomic, no host value: a captured call
#    replays with the epoch of its replay, never a baked one. Every rank issues the same calls
#    in the same order (eager, warm-up, capture and replays alike), so the ranks' epochs agree
#    call by call. A word of an earlier call carries a smaller epoch and is never read as
#    current; no reset is needed. The epoch wraps after 2^32 calls (about 70 million decode
#    steps of 62 layers), beyond any server's life.
# 2. Buffers (both). The receive buffer of a call is its host parity, so a consumer captured
#    in a graph reads a fixed address: parities alternate call by call, and a captured
#    sequence holds an even number of calls of each exchange (the island makes one per layer,
#    62 per step). Back-edge: rank ``o``
#    writes call n + 2 into rank ``r``'s parity-n buffer only after its own call n + 1
#    completed, which polled ``r``'s call-n + 1 rows, which ``r`` stored only after its
#    consumer of call n completed (stream order). No row is rewritten while it is read.
# 3. Sentinel (row publish). Before it stores its row, a program clears its share of the rows
#    the other parity buffer's last call received (already consumed) to the sentinel, then
#    ``bar.sync`` + ``fence.acq_rel.sys``, then the stores to the peers: a peer writes the
#    cleared rows again (call n + 1) only after its call n completed, i.e. after it observed
#    every row this rank stored in call n, each of which its program's fence orders after
#    that program's clears. How many rows the other parity holds and whether its residual
#    segment was published are device words (``RP_ROWS + p``, ``RP_DIRTY + p``), stored by
#    program 0 of the call that wrote parity ``p`` and read only by the next call (the other
#    parity, after it in stream order), so a captured call clears what the call before its
#    replay left, whatever graph of whatever row count ran before. The buffers hold the
#    sentinel from construction, before any rank's first call (a collective barrier).
# 4. Words observed whole. The LL word relies on an aligned 8-byte store being observed
#    whole, as NCCL's LL protocol does; the Lamport words are polled per 16-bit (BF16) or
#    32-bit (FP32) element, each stored whole.
# ------------------------------------------------------------------------------------------

#: LL slots per row of the dispatch (8 bytes each: low word data, high word epoch).
LL_SLOTS = 1024
#: Control words of the v2 exchanges, in the rank's own memory below ``CONTROL_WORDS`` (no
#: peer writes there), one per parity each: the dispatch's epochs, the row publish's
#: residual-dirty and row-count words.
LL_EPOCH = 64
RP_DIRTY = 80
RP_ROWS = 88
LLE: tl.constexpr = tl.constexpr(LL_EPOCH)
RPY: tl.constexpr = tl.constexpr(RP_DIRTY)
RPR: tl.constexpr = tl.constexpr(RP_ROWS)
#: The Lamport sentinels: two all-ones BF16 NaNs in one 32-bit word, and the all-ones FP32
#: NaN (both ``0xFFFFFFFF``); an owner word equal to one is published as the canonical NaN.
SENTINEL_BF16X2 = -1  # 0xFFFFFFFF
SENTINEL_F32 = -1  # 0xFFFFFFFF
SENTINEL_BF16 = -1  # 0xFFFF
CANONICAL_NAN_BF16 = 0x7FFF
CANONICAL_NAN_F32 = 0x7FFFFFFF
S_BF16X2: tl.constexpr = tl.constexpr(SENTINEL_BF16X2)
S_F32: tl.constexpr = tl.constexpr(SENTINEL_F32)
S_BF16: tl.constexpr = tl.constexpr(SENTINEL_BF16)
NAN_BF16: tl.constexpr = tl.constexpr(CANONICAL_NAN_BF16)
NAN_F32: tl.constexpr = tl.constexpr(CANONICAL_NAN_F32)


@triton.jit
def _fence_acq_rel_sys():
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys;\n mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _ll_gather_row(
    src0,
    src1,
    src2,
    src3,
    row,
    cols,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    W3: tl.constexpr,
    NSEG: tl.constexpr,
):
    """The row's data words of the (up to four) segments, back to back."""
    m0 = cols < W0
    d = tl.load(src0 + row * W0 + cols, mask=m0, other=0)
    if NSEG > 1:
        m1 = (cols >= W0) & (cols < W0 + W1)
        d = tl.where(m1, tl.load(src1 + row * W1 + (cols - W0), mask=m1, other=0), d)
    if NSEG > 2:
        m2 = (cols >= W0 + W1) & (cols < W0 + W1 + W2)
        d = tl.where(m2, tl.load(src2 + row * W2 + (cols - W0 - W1), mask=m2, other=0), d)
    if NSEG > 3:
        m3 = (cols >= W0 + W1 + W2) & (cols < W0 + W1 + W2 + W3)
        d = tl.where(m3, tl.load(src3 + row * W3 + (cols - W0 - W1 - W2), mask=m3, other=0), d)
    return d


@triton.jit
def _ll_push_kernel(
    src0,
    src1,
    src2,
    src3,
    peer0,
    peer1,
    peer2,
    peer3,
    local,
    off,
    parity,
    rank,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    W3: tl.constexpr,
    NSEG: tl.constexpr,
    TOTAL: tl.constexpr,
    SLOTS: tl.constexpr,
    BUF: tl.constexpr,
    RANKS: tl.constexpr,
    PDL: tl.constexpr,
):
    """One owner row as ``(epoch << 32) | data`` words into row ``ranks * i + rank`` of every
    rank's parity buffer (no fence, no flag)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    epoch = tl.load(local + LLE + (1 - parity), volatile=True) + 1
    cols = tl.arange(0, SLOTS)
    d = _ll_gather_row(src0, src1, src2, src3, pid.to(tl.int64), cols, W0, W1, W2, W3, NSEG)
    lo = d.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = epoch.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    word = (lo | hi).to(tl.int64, bitcast=True)
    mask = cols < TOTAL
    row = (pid * RANKS + rank).to(tl.int64)
    dst = parity * BUF + row * SLOTS + cols
    tl.store((peer0 + off).to(tl.pointer_type(tl.int64)) + dst, word, mask=mask)
    tl.store((peer1 + off).to(tl.pointer_type(tl.int64)) + dst, word, mask=mask)
    if RANKS > 2:
        tl.store((peer2 + off).to(tl.pointer_type(tl.int64)) + dst, word, mask=mask)
        tl.store((peer3 + off).to(tl.pointer_type(tl.int64)) + dst, word, mask=mask)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _ll_unpack_kernel(
    local,
    off,
    parity,
    out0,
    out1,
    out2,
    out3,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    W3: tl.constexpr,
    NSEG: tl.constexpr,
    TOTAL: tl.constexpr,
    SLOTS: tl.constexpr,
    BUF: tl.constexpr,
    PDL: tl.constexpr,
):
    """Poll row ``t`` until every word carries the call's epoch, store its data words into the
    dense segment rows; program 0 records the epoch as its parity's."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    t = pid.to(tl.int64)
    epoch = tl.load(local + LLE + (1 - parity), volatile=True) + 1
    cols = tl.arange(0, SLOTS)
    mask = cols < TOTAL
    src = (local + off).to(tl.pointer_type(tl.int64)) + parity * BUF + t * SLOTS + cols
    v = tl.load(src, mask=mask, other=0, volatile=True)
    tag = (v.to(tl.uint64, bitcast=True) >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
    pending = tl.sum(tl.where(mask & (tag != epoch), 1, 0), axis=0)
    while pending > 0:
        v = tl.load(src, mask=mask, other=0, volatile=True)
        tag = (v.to(tl.uint64, bitcast=True) >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
        pending = tl.sum(tl.where(mask & (tag != epoch), 1, 0), axis=0)
    d = (v.to(tl.uint64, bitcast=True) & 0xFFFFFFFF).to(tl.uint32).to(tl.int32, bitcast=True)
    tl.store(out0 + t * W0 + cols, d, mask=cols < W0)
    if NSEG > 1:
        tl.store(out1 + t * W1 + (cols - W0), d, mask=(cols >= W0) & (cols < W0 + W1))
    if NSEG > 2:
        tl.store(
            out2 + t * W2 + (cols - W0 - W1), d, mask=(cols >= W0 + W1) & (cols < W0 + W1 + W2)
        )
    if NSEG > 3:
        tl.store(
            out3 + t * W3 + (cols - W0 - W1 - W2),
            d,
            mask=(cols >= W0 + W1 + W2) & (cols < W0 + W1 + W2 + W3),
        )
    if pid == 0:
        tl.store(local + LLE + parity, epoch)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _clear_rows(
    local,
    off0,
    off1,
    parity,
    i,
    dirty,
    half,
    offs,
    BUF0: tl.constexpr,
    BUF1: tl.constexpr,
    RANKS: tl.constexpr,
    HIDDEN: tl.constexpr,
):
    """The sentinel into rows ``ranks * i + s`` of ``parity`` (the residual segment's too when
    ``dirty``)."""
    for s in tl.static_range(RANKS):
        row = (i * RANKS + s).to(tl.int64)
        tl.store(
            local + off0 + parity * BUF0 + row * (HIDDEN // 2) + half,
            tl.full([HIDDEN // 2], S_BF16X2, tl.int32),
        )
        if dirty != 0:
            tl.store(
                local + off1 + parity * BUF1 + row * HIDDEN + offs,
                tl.full([HIDDEN], S_F32, tl.int32),
            )


@triton.jit
def _row_publish_kernel(
    res_out,
    residual,
    branch,
    gain,
    alpha,
    beta,
    eps,
    next_weight,
    next_eps,
    peer0,
    peer1,
    peer2,
    peer3,
    local,
    off0,
    off1,
    parity,
    rank,
    programs,
    BUF0: tl.constexpr,
    BUF1: tl.constexpr,
    NSEG: tl.constexpr,
    RANKS: tl.constexpr,
    HIDDEN: tl.constexpr,
    PDL: tl.constexpr,
):
    """The FFN join of ``_owner_join_push_kernel`` (statement for statement) on the owner's
    compact row, the Lamport publication of its normalized row (segment 0, BF16) and, with
    ``NSEG == 2``, of its updated FP32 residual row (segment 1), then the poll of the rows of
    the same compact index from every rank."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    offs = tl.arange(0, HIDDEN)
    base = pid.to(tl.int64) * HIDDEN + offs
    b = tl.load(branch + base)
    scaled = b.to(tl.float32)
    g = tl.load(gain + offs).to(tl.float32)
    scaled = (scaled * g).to(b.dtype).to(tl.float32)
    variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
    normed = (scaled * tl.rsqrt(variance + eps)).to(b.dtype).to(tl.float32)
    r = tl.load(residual + base).to(tl.float32)
    a = tl.load(alpha + offs)
    be = tl.load(beta + offs)
    updated = a * r + be * normed
    ub = updated.to(tl.int32, bitcast=True)
    ub = tl.where(ub == S_F32, tl.full(ub.shape, NAN_F32, tl.int32), ub)
    tl.store(res_out + base, ub.to(tl.float32, bitcast=True))
    next_input = updated.to(b.dtype).to(tl.float32)
    next_variance = tl.sum(next_input * next_input, axis=0) / HIDDEN
    nw = tl.load(next_weight + offs).to(tl.float32)
    nxt = next_input * tl.rsqrt(next_variance + next_eps)
    nb = (nxt * (nw + 1.0)).to(b.dtype).to(tl.int16, bitcast=True)
    nb = tl.where(nb == S_BF16, tl.full(nb.shape, NAN_BF16, tl.int16), nb)
    half = tl.arange(0, HIDDEN // 2)
    # 1) Clear the rows the other parity buffer's last call received (consumed since), this
    #    program's compact indices pid, pid + programs, ...: that call may have had more rows.
    dirty = tl.load(local + RPY + (1 - parity), volatile=True)
    prev = tl.load(local + RPR + (1 - parity), volatile=True) // RANKS
    if pid < prev:
        _clear_rows(
            local, off0, off1, 1 - parity, pid, dirty, half, offs, BUF0, BUF1, RANKS, HIDDEN
        )
    i = pid + programs
    while i < prev:
        _clear_rows(local, off0, off1, 1 - parity, i, dirty, half, offs, BUF0, BUF1, RANKS, HIDDEN)
        i += programs
    tl.debug_barrier()
    _fence_acq_rel_sys()
    # 2) Publish the owner row into every rank's parity buffer.
    mine = (pid * RANKS + rank).to(tl.int64)
    d0 = off0 + parity * BUF0 + mine * (HIDDEN // 2)
    tl.store((peer0 + d0).to(tl.pointer_type(tl.int16)) + offs, nb)
    tl.store((peer1 + d0).to(tl.pointer_type(tl.int16)) + offs, nb)
    if RANKS > 2:
        tl.store((peer2 + d0).to(tl.pointer_type(tl.int16)) + offs, nb)
        tl.store((peer3 + d0).to(tl.pointer_type(tl.int16)) + offs, nb)
    if NSEG > 1:
        d1 = off1 + parity * BUF1 + mine * HIDDEN
        tl.store(peer0 + d1 + offs, ub)
        tl.store(peer1 + d1 + offs, ub)
        if RANKS > 2:
            tl.store(peer2 + d1 + offs, ub)
            tl.store(peer3 + d1 + offs, ub)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    # 3) Poll the rows of this compact index from every rank (the own row included).
    for s in tl.static_range(RANKS):
        row = (pid * RANKS + s).to(tl.int64)
        p0 = local + off0 + parity * BUF0 + row * (HIDDEN // 2) + half
        v = tl.load(p0, volatile=True)
        pending = tl.sum(
            tl.where(((v & 0xFFFF) == 0xFFFF) | (((v >> 16) & 0xFFFF) == 0xFFFF), 1, 0), axis=0
        )
        while pending > 0:
            v = tl.load(p0, volatile=True)
            pending = tl.sum(
                tl.where(((v & 0xFFFF) == 0xFFFF) | (((v >> 16) & 0xFFFF) == 0xFFFF), 1, 0),
                axis=0,
            )
        if NSEG > 1:
            p1 = local + off1 + parity * BUF1 + row * HIDDEN + offs
            u = tl.load(p1, volatile=True)
            pending1 = tl.sum(tl.where(u == S_F32, 1, 0), axis=0)
            while pending1 > 0:
                u = tl.load(p1, volatile=True)
                pending1 = tl.sum(tl.where(u == S_F32, 1, 0), axis=0)
    # 4) Program 0 records what this parity now holds, for the next call's clear (no program of
    #    this call reads these words).
    if pid == 0:
        tl.store(local + RPR + parity, programs * RANKS)
        if NSEG > 1:
            tl.store(local + RPY + parity, 1)
        else:
            tl.store(local + RPY + parity, 0)


class _FabricMemory:
    """Collective, zeroed per-rank memory of ``words`` int32 words on the TP group's fabric:
    ``uc[peer]`` is a peer's memory, ``local`` this rank's."""

    def __init__(self, mapping: Mapping, words: int):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._agree = _mnnvl_workspace_all_succeeded
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, words * 4, mapping, mapping.is_multi_node()
            )
            self.uc = [
                self._handle.get_uc_buffer(peer, (words,), torch.int32, 0)
                for peer in range(self.ranks)
            ]
            with torch.cuda.device(self.device):
                self.uc[self.rank].zero_()
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not self._agree(self._comm, error is None):
            raise RuntimeError(
                "owner exchange workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        self.local = self.uc[self.rank]

    def barrier(self) -> None:
        """A collective over the group, after this rank's device work completed."""
        torch.cuda.synchronize(self.device)
        if not self._agree(self._comm, True):
            raise RuntimeError("owner exchange barrier failed on at least one rank")


class OwnerLLDispatch:
    """The island's expert dispatch on the LL protocol (the v2 notes above).

    ``segments`` are the row widths in int32 words (at most four, at most ``LL_SLOTS`` in
    all); ``dispatch`` takes the owner's compact rows of each and returns every owner's rows as
    dense ``[rows, width]`` int32 tensors at fixed addresses. Construction is collective. The
    calls form one ordered chain per rank with the same row counts in the same order on every
    rank; a CUDA graph holds an even number of calls.

    Caller contract (not checked here): the host call counter ``calls`` is even at every
    capture and at every step boundary (an odd start would run two consecutive calls of one
    parity: the second would reuse the first's epoch and read its words as arrived). The dense
    outputs are overwritten by the next call, so their consumers run before it, in stream order.
    """

    def __init__(
        self,
        mapping: Mapping,
        segments: Sequence[int] = (512, 64, 8, 8),
        *,
        max_rows: int = 1024,
        pdl: bool = True,
    ):
        if mapping.tp_size not in (2, 4):
            raise ValueError(f"owner dispatch serves 2 or 4 ranks, got {mapping.tp_size}")
        if not 1 <= len(segments) <= MAX_SEGMENTS or sum(segments) > LL_SLOTS:
            raise ValueError(
                f"1..{MAX_SEGMENTS} segments of at most {LL_SLOTS} words in all, "
                f"got {list(segments)}"
            )
        if max_rows % mapping.tp_size or max_rows <= 0:
            raise ValueError(f"max_rows {max_rows}: a positive multiple of {mapping.tp_size}")
        self.segments = [int(w) for w in segments]
        self.max_rows = int(max_rows)
        self.pdl = bool(pdl)
        self.calls = 0
        self.buf = self.max_rows * LL_SLOTS
        # int32 words: the control words, then two parity buffers of int64 slots.
        self.mem = _FabricMemory(mapping, CONTROL_WORDS + 2 * BUFFERS * self.buf)
        self.ranks, self.rank = self.mem.ranks, self.mem.rank
        self._peers = list(self.mem.uc) + [self.mem.local] * (4 - self.ranks)
        self.dense = [
            torch.empty((self.max_rows, w), dtype=torch.int32, device=self.mem.device)
            for w in self.segments
        ]

    def epoch(self) -> int:
        """This rank's device epoch: the calls completed on this dispatch (host read)."""
        return int(self.mem.local[LL_EPOCH : LL_EPOCH + 2].max().item())

    def dispatch(
        self, sources: Sequence[torch.Tensor], rows: int, parity: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Every owner's rows of each segment, ``[rows, width]`` int32, from the owner's compact
        ``[rows / ranks, width]`` rows (viewable as int32)."""
        if len(sources) != len(self.segments):
            raise ValueError(f"{len(self.segments)} sources, got {len(sources)}")
        if rows % self.ranks or not self.ranks <= rows <= self.max_rows:
            raise ValueError(f"rows {rows}: a multiple of {self.ranks} up to {self.max_rows}")
        own = rows // self.ranks
        srcs = []
        for s, src in enumerate(sources):
            v = src.reshape(src.shape[0], -1).view(torch.int32)
            if v.shape[0] < own or v.shape[1] != self.segments[s] or not v.is_contiguous():
                raise ValueError(
                    f"segment {s}: needs [{own}, {self.segments[s]}] int32 words, "
                    f"got {tuple(v.shape)}"
                )
            srcs.append(v)
        if parity is None:
            parity = self.calls & 1
            self.calls += 1
        n = len(srcs)
        w = self.segments + [1] * (MAX_SEGMENTS - n)
        srcs = srcs + [srcs[0]] * (MAX_SEGMENTS - n)
        outs = self.dense + [self.dense[0]] * (MAX_SEGMENTS - n)
        meta = dict(
            W0=w[0],
            W1=w[1],
            W2=w[2],
            W3=w[3],
            NSEG=n,
            TOTAL=sum(self.segments),
            SLOTS=LL_SLOTS,
            BUF=self.buf,
            PDL=self.pdl,
            num_warps=4,
            launch_pdl=self.pdl,
        )
        _ll_push_kernel[(own,)](
            *srcs,
            *self._peers[:4],
            self.mem.local,
            CONTROL_WORDS,
            parity,
            self.rank,
            RANKS=self.ranks,
            **meta,
        )
        _ll_unpack_kernel[(rows,)](self.mem.local, CONTROL_WORDS, parity, *outs, **meta)
        return [d[:rows] for d in self.dense]


class OwnerRowPublish:
    """The island's FFN join on the owner's rows fused with the Lamport all-gather of the next
    layer's normalized input (and, per call, of the updated FP32 residual); the v2 notes above.
    Construction is collective; the call-order, parity and graph rules of ``OwnerLLDispatch``
    apply.

    Caller contract (not checked here): the returned views are valid only until the next call
    on the same instance starts (it clears them to the sentinel), so every consumer of a view
    (the next layer's QKV, a drafter capture of the residual rows, the final norm's readers)
    runs before that call, in stream order. The next norm is the built-in join's BF16 ordering
    (``NORM_FP32_RESIDUAL`` off, the ``MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL`` default).
    """

    def __init__(
        self, mapping: Mapping, *, hidden: int = 8192, max_rows: int = 1024, pdl: bool = True
    ):
        if mapping.tp_size not in (2, 4):
            raise ValueError(f"owner row publish serves 2 or 4 ranks, got {mapping.tp_size}")
        if max_rows % mapping.tp_size or max_rows <= 0:
            raise ValueError(f"max_rows {max_rows}: a positive multiple of {mapping.tp_size}")
        self.hidden = int(hidden)
        self.max_rows = int(max_rows)
        self.pdl = bool(pdl)
        self.calls = 0
        self.buf0 = self.max_rows * (self.hidden // 2)
        self.buf1 = self.max_rows * self.hidden
        self.off0 = CONTROL_WORDS
        self.off1 = CONTROL_WORDS + BUFFERS * self.buf0
        self.mem = _FabricMemory(mapping, self.off1 + BUFFERS * self.buf1)
        self.ranks, self.rank = self.mem.ranks, self.mem.rank
        self._peers = list(self.mem.uc) + [self.mem.local] * (4 - self.ranks)
        local = self.mem.local
        with torch.cuda.device(self.mem.device):
            local[self.off0 : self.off1].fill_(SENTINEL_BF16X2)
            local[self.off1 : self.off1 + BUFFERS * self.buf1].fill_(SENTINEL_F32)
        self.mem.barrier()

    def view(self, segment: int, parity: int, rows: int) -> torch.Tensor:
        """The receive rows of ``segment`` (0: BF16 ``[rows, hidden]``, 1: FP32) of ``parity``."""
        if segment == 0:
            base = self.off0 + parity * self.buf0
            words = self.mem.local[base : base + rows * (self.hidden // 2)]
            return words.view(torch.bfloat16).view(rows, self.hidden)
        base = self.off1 + parity * self.buf1
        return (
            self.mem.local[base : base + rows * self.hidden]
            .view(torch.float32)
            .view(rows, self.hidden)
        )

    def join_publish(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        gain: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        eps: float,
        next_weight: torch.Tensor,
        next_eps: float,
        rows: int,
        with_residual: bool,
        parity: Optional[int] = None,
    ):
        """Returns ``(updated FP32 [rows / ranks, hidden], views)``: ``views[0]`` every rank's
        normalized BF16 rows ``[rows, hidden]``, with ``with_residual`` ``views[1]`` every
        rank's updated FP32 rows."""
        if rows % self.ranks or not self.ranks <= rows <= self.max_rows:
            raise ValueError(f"rows {rows}: a multiple of {self.ranks} up to {self.max_rows}")
        own = rows // self.ranks
        if parity is None:
            parity = self.calls & 1
            self.calls += 1
        out = torch.empty((own, self.hidden), dtype=torch.float32, device=residual.device)
        nseg = 2 if with_residual else 1
        _row_publish_kernel[(own,)](
            out,
            residual,
            branch,
            gain,
            alpha,
            beta,
            eps,
            next_weight,
            next_eps,
            *self._peers[:4],
            self.mem.local,
            self.off0,
            self.off1,
            parity,
            self.rank,
            own,
            BUF0=self.buf0,
            BUF1=self.buf1,
            NSEG=nseg,
            RANKS=self.ranks,
            HIDDEN=self.hidden,
            PDL=self.pdl,
            num_warps=16,
            launch_pdl=self.pdl,
        )
        return out, [self.view(s, parity, rows) for s in range(nseg)]
