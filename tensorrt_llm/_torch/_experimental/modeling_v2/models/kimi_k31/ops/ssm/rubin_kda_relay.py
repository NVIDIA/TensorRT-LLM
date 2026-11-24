# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-split KDA of a context-parallel (cep) step: every rank runs the recurrence of its own rows.

The gated delta rule of one head, in the cache's value-first layout (``X`` is ``[V, K]``), is affine
in its state: over a run of tokens ``X_end = X_start @ M + N`` and every output
``o_t = X_start @ m_t + o0_t``, where ``M`` (``[K, K]``), ``N``, ``m_t`` and ``o0_t`` depend only on
the run's own tokens. With ``V == K`` the BT16 engine yields all four in one launch over two copies
of the run (``kda_prefill_segments``): from a zero state with the real values (``o0_t``, ``N``), and
from the identity state with zero values (``m_t`` as the outputs, ``M`` as the final state).

A cep rank holds rows ``[r * rows, (r + 1) * rows)`` of the step, so a context request spans
consecutive ranks. The step plan (:func:`kda_relay_plan`, host arithmetic once a step) cuts every
request into segments at the rank boundaries and at its block-reuse snapshot point; a segment is one
sequence of its rank's launch. The KDA state pools stay head-sharded as under TP N (rank ``j`` holds
heads ``[j * H / N, (j + 1) * H / N)`` of every request: the layout the generation server's KV
transfer reads), so each rank is also the owner of its heads' chain of every request. A layer:

1. A: one all-gather of every rank's last 3 raw ``[q | k | v]`` rows and its channel shard of the
   cached convolution windows of the requests with a cached state (:func:`relay_gather_input`).
2. The convolution of every local segment from its 3-token history (the cached window, or the raw
   rows before the segment, local or gathered: :func:`relay_histories`), then both passes of every
   segment in one launch (:class:`RubinKdaRelayPrefill`). A request's first segment from a fresh
   state is exact after pass 1 and needs no pass 2.
3. B: one all-to-all hands every owner its heads of each segment's ``N`` and ``M`` and its channel
   shard of the end and snapshot convolution windows (:func:`relay_pack`).
4. Each owner composes every request's chain over its segments for its heads from the cached state
   (:class:`RubinKdaRelayChain`, one launch for every chain; :func:`relay_own`): the start state of
   each segment, the snapshot state at a snapshot segment's start, the end state; it stores the end
   and snapshot states and windows into its own pools.
5. C: one all-to-all hands every segment's rank the start state of the segment, all heads, and the
   rank adds ``m_t @ X_start^T`` to the segment's outputs (:class:`RubinKdaRelayCorrect`).

Nothing moves per token except the 3-row histories; a segment moves ``N`` (FP32) and, when it needs
pass 2, ``M`` and its start state (BF16: the operands of the KDA kernel's own state products, as the
SEP ring relay sends them), ``H * 128 * 128`` values each. The composed states and the pools stay
FP32.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ...kernels.ssm.rubin_kda_prefill_ops import (
    KDA_SEQ_INIT_IDENTITY,
    KDA_SEQ_INIT_ZERO,
    KDA_SEQ_ZERO_V,
    kda_prefill_segments,
)
from ...kernels.ssm.rubin_kda_relay_chain import STEP_COLS, relay_chain

# Raw rows of a convolution history (the window width minus one).
WINDOW = 3
_HEAD_DIM = 128
_PASS1_FLAGS = KDA_SEQ_INIT_ZERO
_PASS2_FLAGS = KDA_SEQ_INIT_IDENTITY | KDA_SEQ_ZERO_V


# --------------------------------------------------------------------------------------------------
# The step plan.
# --------------------------------------------------------------------------------------------------


class Segment(NamedTuple):
    """A run of request ``req``'s new tokens on one rank: step rows ``[begin, end)``."""

    req: int
    begin: int
    end: int
    rank: int
    first: bool  # the request's first new token
    last: bool  # its last new token
    snapshot: bool  # begins at the request's snapshot point
    affine: bool  # needs pass 2: not first, or first from a cached state


def relay_segments(
    lengths: Sequence[int],
    has_init: Sequence[bool],
    snapshot_offsets: Sequence[int],
    world: int,
    rows: int,
) -> tuple[tuple[Segment, ...], ...]:
    """Every request's segments, in token order, per request.

    ``snapshot_offsets[b]``: the token count of request ``b`` before its snapshot point, 0 for none.
    """
    out = []
    start = 0
    for req, (length, init, snap) in enumerate(zip(lengths, has_init, snapshot_offsets)):
        if length <= 0:
            raise ValueError(f"context request {req} has {length} new tokens")
        end = start + length
        cuts = {start, end}
        cuts.update(k * rows for k in range(1, world) if start < k * rows < end)
        if snap:
            if not 0 < snap < length:
                raise ValueError(f"snapshot after {snap} of {length} tokens")
            cuts.add(start + snap)
        points = sorted(cuts)
        segs = []
        for lo, hi in zip(points, points[1:]):
            first = lo == start
            segs.append(
                Segment(
                    req=req,
                    begin=lo,
                    end=hi,
                    rank=lo // rows,
                    first=first,
                    last=hi == end,
                    snapshot=bool(snap) and lo == start + snap,
                    affine=not first or bool(init),
                )
            )
        out.append(tuple(segs))
        start = end
    return tuple(out)


class KdaRelayPlan(NamedTuple):
    """One rank's share of a step's sequence-split KDA, the same for every KDA layer.

    Host counts, then device index tensors (int64 unless noted). ``ns`` / ``ms`` / ``ws`` are the
    largest per-rank segment, pass-2 and window counts of the group: the all-to-all chunk shapes.
    """

    world: int
    rank: int
    rows: int
    live: int
    heads: int
    nseg: int  # local segments (pass-1 sequences)
    naff: int  # local segments with pass 2
    out_rows: int  # live + pass-2 rows
    max_seqlen: int
    num_cached: int  # requests with a cached state (the gathered windows)
    ns: int
    ms: int
    ws: int
    corr: tuple  # per local pass-2 segment: (row begin, row end, its first pass-2 row)
    # kernel launch (int32 except cu_seqlens)
    cu_seqlens: torch.Tensor
    src_rows: torch.Tensor
    seq_flags: torch.Tensor
    state_rows: torch.Tensor
    seq_order: torch.Tensor
    cu_chunks: torch.Tensor
    # convolution (int32 starts / slots, bool init)
    conv_starts: torch.Tensor
    conv_slots: torch.Tensor
    conv_init: torch.Tensor
    # histories
    tail_dst: torch.Tensor
    tail_src: torch.Tensor
    cached_req: torch.Tensor
    hist_local_dst: torch.Tensor
    hist_local_src: torch.Tensor
    hist_small_dst: torch.Tensor
    hist_small_src: torch.Tensor
    end_local: torch.Tensor  # local segments whose end window goes to the owners
    snap_local: torch.Tensor  # local segments whose history is a snapshot window
    # owner: every request's pool slot and chain (``kernels/ssm/rubin_kda_relay_chain``; int32)
    slots: torch.Tensor
    chain_steps: torch.Tensor  # [R, max segments, STEP_COLS]
    chain_counts: torch.Tensor  # [R] segments
    chain_init: torch.Tensor  # [R] 1: the chain starts from the pool row
    end_win_req: torch.Tensor
    end_win_q: torch.Tensor
    end_win_l: torch.Tensor
    snap_win_rows: torch.Tensor
    snap_win_q: torch.Tensor
    snap_win_l: torch.Tensor


def kda_relay_plan(
    lengths: Sequence[int],
    has_init: Sequence[bool],
    snapshots: Sequence[tuple[int, int, int]],
    world: int,
    rank: int,
    heads: int,
    slots: torch.Tensor,
) -> KdaRelayPlan:
    """Plan rank ``rank``'s share of a step of context requests ``lengths`` (new tokens, in row
    order) over ``world`` ranks of ``heads`` KDA heads.

    ``has_init[b]``: request ``b`` continues a cached state and window. ``snapshots``: the
    ``(request, pool slot, token offset)`` block-reuse snapshots (at most one per request).
    ``slots``: the requests' pool slots on the device (any integer dtype).
    """
    num_tokens = sum(lengths)
    rows = -(-num_tokens // world)
    begin = min(rank * rows, num_tokens)
    live = min(begin + rows, num_tokens) - begin
    snap_offset = [0] * len(lengths)
    snap_slot = [-1] * len(lengths)
    for req, slot, offset in snapshots:
        snap_offset[req], snap_slot[req] = offset, slot
    chains = relay_segments(lengths, has_init, snap_offset, world, rows)

    # Per rank: its segments in row order, their local and pass-2 indices, their windows.
    per_rank = [[] for _ in range(world)]
    for chain in chains:
        for seg in chain:
            per_rank[seg.rank].append(seg)
    # Window slots of a rank: its segments' end windows, then its snapshot windows.
    local_index, affine_index, window_index = {}, {}, {}
    for q, segs in enumerate(per_rank):
        naff_q = 0
        for i, s in enumerate(segs):
            local_index[s] = (q, i)
            if s.affine:
                affine_index[s] = (q, naff_q)
                naff_q += 1
        windows = [(s, False) for s in segs if s.last] + [(s, True) for s in segs if s.snapshot]
        for i, key in enumerate(windows):
            window_index[key] = (q, i)
    ns = max(len(segs) for segs in per_rank)
    ms = max(sum(1 for s in segs if s.affine) for segs in per_rank)
    ws = max(
        sum(1 for s in segs if s.last) + sum(1 for s in segs if s.snapshot) for segs in per_rank
    )

    mine = per_rank[rank]
    affine = [s for s in mine if s.affine]
    nseg, naff = len(mine), len(affine)
    lens1 = [s.end - s.begin for s in mine]
    lens2 = [s.end - s.begin for s in affine]
    lens = lens1 + lens2
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    src = [s.begin - begin for s in mine] + [s.begin - begin for s in affine]
    flags = [_PASS1_FLAGS] * nseg + [_PASS2_FLAGS] * naff
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    chunks = [0]
    for n in lens:
        chunks.append(chunks[-1] + (n + 15) // 16)
    corr = []
    for k, s in enumerate(affine):
        corr.append((s.begin - begin, s.end - begin, cu[nseg + k]))

    # Histories: the 3 raw rows before each segment, by source.
    cached = [b for b, init in enumerate(has_init) if init]
    cached_pos = {b: i for i, b in enumerate(cached)}
    tail_first = rows - WINDOW  # this rank's tail rows: local rows [rows - 3, rows) that are live
    tail_dst = [i for i in range(WINDOW) if 0 <= tail_first + i < live]
    tail_src = [tail_first + i for i in tail_dst]
    small_zero = world * WINDOW + len(cached) * WINDOW
    hl_dst, hl_src, hs_dst, hs_src = [], [], [], []
    req_start = [0]
    for n in lengths:
        req_start.append(req_start[-1] + n)
    conv_init = []
    for i, s in enumerate(mine):
        init = not s.first or bool(has_init[s.req])
        conv_init.append(init)
        if not init:
            continue
        for col in range(WINDOW):
            p = s.begin - WINDOW + col
            dst = i * WINDOW + col
            if p >= req_start[s.req]:
                if p >= begin:
                    hl_dst.append(dst)
                    hl_src.append(p - begin)
                else:
                    q = p // rows
                    t = p - (q * rows + rows - WINDOW)
                    if not 0 <= t < WINDOW:
                        raise AssertionError(f"history row {p} is not a tail row of rank {q}")
                    hs_dst.append(dst)
                    hs_src.append(q * WINDOW + t)
            elif has_init[s.req]:
                k = p - (req_start[s.req] - WINDOW)
                hs_dst.append(dst)
                hs_src.append(world * WINDOW + cached_pos[s.req] * WINDOW + k)
            else:
                hs_dst.append(dst)
                hs_src.append(small_zero)
    end_local = [i for i, s in enumerate(mine) if s.last]
    snap_local = [i for i, s in enumerate(mine) if s.snapshot]

    # Owner chains: per request, its segments in order (``STEP_COLS`` columns each).
    device = slots.device
    max_steps = max(len(c) for c in chains)
    steps = []
    for chain in chains:
        rows_ = []
        for sg in chain:
            nq, nl = local_index[sg]
            if sg.affine:
                mq, ml = affine_index[sg]
                rows_.append([1, nq, nl, mq, ml, snap_slot[sg.req] if sg.snapshot else -1])
            else:
                rows_.append([0, nq, nl, 0, 0, -1])
        rows_ += [[0] * STEP_COLS] * (max_steps - len(rows_))
        steps.append(rows_)
    lasts = [c[-1] for c in chains]
    snaps = [s for c in chains for s in c if s.snapshot]

    def i64(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return KdaRelayPlan(
        world=world,
        rank=rank,
        rows=rows,
        live=live,
        heads=heads,
        nseg=nseg,
        naff=naff,
        out_rows=cu[-1],
        max_seqlen=max(lens1, default=0),
        num_cached=len(cached),
        ns=ns,
        ms=ms,
        ws=ws,
        corr=tuple(corr),
        cu_seqlens=i64(cu),
        src_rows=i32(src),
        seq_flags=i32(flags),
        state_rows=i32(list(range(len(lens)))),
        seq_order=i32(order),
        cu_chunks=i32(chunks),
        conv_starts=i32([s.begin - begin for s in mine] + [live]),
        conv_slots=i32(list(range(nseg))),
        conv_init=torch.tensor(conv_init, dtype=torch.bool, device=device),
        tail_dst=i64(tail_dst),
        tail_src=i64(tail_src),
        cached_req=i64(cached),
        hist_local_dst=i64(hl_dst),
        hist_local_src=i64(hl_src),
        hist_small_dst=i64(hs_dst),
        hist_small_src=i64(hs_src),
        end_local=i64(end_local),
        snap_local=i64(snap_local),
        slots=slots.to(torch.int64),
        chain_steps=torch.tensor(steps, dtype=torch.int32, device=device),
        chain_counts=i32([len(c) for c in chains]),
        chain_init=i32([int(bool(init)) for init in has_init]),
        end_win_req=i64([s.req for s in lasts]),
        end_win_q=i64([window_index[(s, False)][0] for s in lasts]),
        end_win_l=i64([window_index[(s, False)][1] for s in lasts]),
        snap_win_rows=i64([snap_slot[s.req] for s in snaps]),
        snap_win_q=i64([window_index[(s, True)][0] for s in snaps]),
        snap_win_l=i64([window_index[(s, True)][1] for s in snaps]),
    )


# --------------------------------------------------------------------------------------------------
# The entries.
# --------------------------------------------------------------------------------------------------


class RubinKdaRelayPrefill(OpWrapper):
    """Both passes of a rank's segments in one BT16 launch; return the ``[1, out_rows, H, 128]`` outputs.

    Sequence ``b`` (``cu_seqlens`` partitions the output rows) reads q / k / g / beta rows
    ``[src_rows[b], src_rows[b] + len_b)``, starts from the zero (pass 1) or identity (pass 2, zero
    v) state per ``seq_flags[b]``, and leaves its final state in ``states[state_rows[b]]`` (FP32,
    ``[rows, H, 128, 128]``, none read). Pass-1 sequences read v rows per ``cu_seqlens``.
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})
    note = """
    The family's ``kda_prefill_segments``: the BT16 engine route with the
    per-sequence src_rows / seq_flags build (no snapshot checkpoints). q / k /
    v / g are the convolution's contiguous BF16 [1, T, H, 128] (v of T rows,
    read by the pass-1 sequences only), beta the BF16 [1, T, H] logits;
    pass-2 sequences have output rows past T. seq_order / cu_chunks are the
    stable descending length permutation and the prefix sums of
    ceil(len / 16) over every sequence. Within the BT16 bands of an fp32
    recurrence (2e-2 output, 1e-2 state, relative to the largest magnitude).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        states: torch.Tensor,
        state_rows: torch.Tensor,
        cu_seqlens: torch.Tensor,
        src_rows: torch.Tensor,
        seq_flags: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        seq_order: torch.Tensor,
        cu_chunks: torch.Tensor,
        out_rows: int,
    ) -> torch.Tensor:
        return kda_prefill_segments(
            q,
            k,
            v,
            g,
            beta,
            states,
            state_rows,
            cu_seqlens,
            src_rows,
            seq_flags,
            A_log,
            dt_bias,
            scale,
            lower_bound,
            seq_order,
            cu_chunks,
            out_rows,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        states: torch.Tensor,
        state_rows: torch.Tensor,
        cu_seqlens: torch.Tensor,
        src_rows: torch.Tensor,
        seq_flags: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        seq_order: torch.Tensor,
        cu_chunks: torch.Tensor,
        out_rows: int,
    ) -> torch.Tensor:
        """fp32 token-by-token recurrence; ``seq_order`` and ``cu_chunks`` only schedule."""
        heads = q.shape[2]
        qf = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-12) * scale
        kf = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-12)
        gate = g.float() + dt_bias.float().reshape(q.shape[-2:])
        rate = A_log.float().exp().reshape(-1, 1)
        decay = (lower_bound * torch.sigmoid(rate * gate)).exp()
        bf = beta.float().sigmoid()
        out = torch.zeros((1, out_rows, heads, _HEAD_DIM), dtype=torch.float32, device=q.device)
        eye = torch.eye(_HEAD_DIM, device=q.device).expand(heads, -1, -1)
        bounds = cu_seqlens.tolist()
        for b, (row, src, flags, lo, hi) in enumerate(
            zip(state_rows.tolist(), src_rows.tolist(), seq_flags.tolist(), bounds, bounds[1:])
        ):
            state = eye.clone() if flags & KDA_SEQ_INIT_IDENTITY else eye.new_zeros(eye.shape)
            for t in range(hi - lo):
                s = src + t
                state = state * decay[0, s].unsqueeze(-2)
                key = kf[0, s]
                value = torch.zeros_like(key) if flags & KDA_SEQ_ZERO_V else v[0, lo + t].float()
                err = value - (state * key.unsqueeze(-2)).sum(-1)
                state = state + bf[0, s, :, None, None] * err.unsqueeze(-1) * key.unsqueeze(-2)
                out[0, lo + t] = (state * qf[0, s].unsqueeze(-2)).sum(-1)
            states[row] = state
        return out.to(q.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        states: torch.Tensor,
        state_rows: torch.Tensor,
        cu_seqlens: torch.Tensor,
        src_rows: torch.Tensor,
        seq_flags: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        seq_order: torch.Tensor,
        cu_chunks: torch.Tensor,
        out_rows: int,
    ) -> None:
        bounds = cu_seqlens.tolist()
        lengths = [hi - lo for lo, hi in zip(bounds, bounds[1:])]
        assert bounds[0] == 0 and bounds[-1] == out_rows and min(lengths) > 0, "cu_seqlens"
        for src, flags, lo, n in zip(src_rows.tolist(), seq_flags.tolist(), bounds, lengths):
            assert 0 <= src and src + n <= q.shape[1], "a sequence reads past the input rows"
            assert flags in (_PASS1_FLAGS, _PASS2_FLAGS), f"seq_flags {flags}"
            if not flags & KDA_SEQ_ZERO_V:
                assert lo + n <= v.shape[1], "a pass-1 sequence reads v past its rows"
        rows = state_rows.tolist()
        assert len(set(rows)) == len(rows) and max(rows) < states.shape[0], "state_rows"
        assert seq_order.tolist() == sorted(range(len(lengths)), key=lambda i: -lengths[i])
        chunks = [0]
        for n in lengths:
            chunks.append(chunks[-1] + (n + 15) // 16)
        assert cu_chunks.tolist() == chunks, "cu_chunks"


rubin_kda_relay_prefill = RubinKdaRelayPrefill()


class RubinKdaRelayCorrect(OpWrapper):
    """Add the start state's share to each pass-2 segment's outputs, in place.

    ``o`` ``[T, H, 128]`` holds the pass-1 rows and after them the pass-2 rows ``m_t``; for
    ``(lo, hi, m0)`` of ``bounds[i]``: ``o[lo:hi] += o[m0:m0 + hi - lo] @ starts[i]^T`` per head
    (``starts`` ``[S, H, V, K]``, the relayed BF16 start states).
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})
    note = """
    One torch.bmm per segment over BF16 operands with FP32 accumulation, and an
    in-place add into the BF16 outputs.
    """

    def __call__(self, o: torch.Tensor, starts: torch.Tensor, bounds: tuple) -> None:
        for i, (lo, hi, m0) in enumerate(bounds):
            m = o[m0 : m0 + hi - lo].transpose(0, 1)
            corr = torch.bmm(m, starts[i].to(o.dtype).transpose(1, 2))
            o[lo:hi].add_(corr.transpose(0, 1))

    def reference(self, o: torch.Tensor, starts: torch.Tensor, bounds: tuple) -> None:
        """fp32 products and add, rounded once."""
        for i, (lo, hi, m0) in enumerate(bounds):
            m = o[m0 : m0 + hi - lo].float()
            corr = torch.einsum("thk,hvk->thv", m, starts[i].float())
            o[lo:hi] = (o[lo:hi].float() + corr).to(o.dtype)


rubin_kda_relay_correct = RubinKdaRelayCorrect()


class RubinKdaRelayChain(OpWrapper):
    """Every request's owner chain over this rank's ``heads`` heads, in place.

    ``steps[r, d]`` (``STEP_COLS`` int32: kind, N's rank and slot, M's rank and slot, snapshot pool
    row or -1) for ``d < counts[r]``; the state starts from ``pool[slots[r]]`` when ``init[r]``, else
    zero. Kind 0 sets it to N; kind 1 writes it to ``send[M's rank, M's slot]`` (BF16), to the
    snapshot row (FP32) when named, then becomes ``bf16(X) @ M + N``. The end state goes to
    ``pool[slots[r]]``. ``recv`` is the all-to-all B (``[world, CB]`` FP32 words: N FP32 at word
    ``slot * elems``, M BF16 from word ``m_words``).
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_107})
    note = """
    CuTe DSL, one CTA per (request, head, 8 value rows), one thread per state
    column; the steps of a request run in order inside the CTA (one launch for
    every chain of the layer). BF16 x BF16 products accumulated in FP32 from N
    over j = 0..127 (SIMT FMAs), as the composition the SEP relay runs. The
    per-step BF16 rounding of X turns accumulation-order differences into
    ~2e-4 relative differences against any other order (a tensor-core dot,
    torch's matmul), so the reference is compared within that band.
    """

    def __call__(
        self,
        steps: torch.Tensor,
        counts: torch.Tensor,
        init: torch.Tensor,
        slots: torch.Tensor,
        recv: torch.Tensor,
        m_words: int,
        heads: int,
        pool: torch.Tensor,
        send: Optional[torch.Tensor],
    ) -> None:
        relay_chain(steps, counts, init, slots, recv, m_words, heads, pool, send)

    def reference(
        self,
        steps: torch.Tensor,
        counts: torch.Tensor,
        init: torch.Tensor,
        slots: torch.Tensor,
        recv: torch.Tensor,
        m_words: int,
        heads: int,
        pool: torch.Tensor,
        send: Optional[torch.Tensor],
    ) -> None:
        """Step by step in fp32 on the BF16-rounded operands."""
        elems = heads * _HEAD_DIM * _HEAD_DIM
        shape = (heads, _HEAD_DIM, _HEAD_DIM)
        mbuf = recv.view(torch.bfloat16)
        for r, (count, start, slot) in enumerate(
            zip(counts.tolist(), init.tolist(), slots.tolist())
        ):
            x = pool[slot].float() if start else torch.zeros(shape, device=recv.device)
            for kind, nq, nl, mq, ml, snap in steps[r, :count].tolist():
                n = recv[nq, nl * elems : (nl + 1) * elems].view(shape)
                if kind == 0:
                    x = n.clone()
                    continue
                send[mq, ml] = x.to(torch.bfloat16)
                if snap >= 0:
                    pool[snap] = x
                m0 = 2 * m_words + ml * elems
                m = mbuf[mq, m0 : m0 + elems].view(shape).float()
                x = x.to(torch.bfloat16).float() @ m + n
            pool[slot] = x


rubin_kda_relay_chain = RubinKdaRelayChain()


# --------------------------------------------------------------------------------------------------
# The phases of a layer (the target sequences them around its collectives).
# --------------------------------------------------------------------------------------------------


def relay_gather_input(
    plan: KdaRelayPlan, y: torch.Tensor, conv_pool: torch.Tensor
) -> torch.Tensor:
    """This rank's ``[1, X]`` BF16 part of the all-gather A: its last 3 raw ``[q | k | v]`` rows
    (zero where not live), then its channel shard of the cached requests' convolution windows."""
    width = WINDOW * plan.heads * _HEAD_DIM
    tails = y.new_zeros(WINDOW, width)
    if plan.tail_dst.numel():
        tails[plan.tail_dst] = y.index_select(0, plan.tail_src)[:, :width]
    windows = conv_pool.index_select(0, plan.slots.index_select(0, plan.cached_req))
    return torch.cat((tails.reshape(-1), windows.to(y.dtype).reshape(-1))).unsqueeze(0)


def relay_histories(plan: KdaRelayPlan, y: torch.Tensor, gathered: torch.Tensor) -> torch.Tensor:
    """The ``[nseg, 3P, 3]`` BF16 convolution history of every local segment (zero where the
    convolution starts fresh), from ``gathered`` (``[world, X]``, the all-gather A)."""
    world, heads = plan.world, plan.heads
    width = WINDOW * heads * _HEAD_DIM
    shard = heads // world * _HEAD_DIM
    tails = gathered[:, : WINDOW * width].reshape(world * WINDOW, width)
    cached = gathered[:, WINDOW * width :].reshape(world, plan.num_cached, 3, shard, WINDOW)
    # [request, column, section, rank, channel]: the window's columns as full-width raw rows.
    cached = cached.permute(1, 4, 2, 0, 3).reshape(plan.num_cached * WINDOW, width)
    small = torch.cat((tails, cached, tails.new_zeros(1, width)))
    hist = y.new_zeros(plan.nseg * WINDOW, width)
    if plan.hist_local_dst.numel():
        hist[plan.hist_local_dst] = y.index_select(0, plan.hist_local_src)[:, :width]
    if plan.hist_small_dst.numel():
        hist[plan.hist_small_dst] = small.index_select(0, plan.hist_small_src)
    return hist.view(plan.nseg, WINDOW, width).transpose(1, 2).contiguous()


def _layout(plan: KdaRelayPlan) -> tuple[int, int, int, int]:
    """The all-to-all B row in FP32 words: ``N`` FP32 (``ns`` slots), then ``M`` and the windows in
    BF16 (``ms`` and ``ws`` slots, two values a word). Returns (elements of a state shard, the
    word offsets of the M and window regions, the row's words)."""
    hs = plan.heads // plan.world
    elems = hs * _HEAD_DIM * _HEAD_DIM
    wcols = 3 * hs * _HEAD_DIM * WINDOW
    m0 = plan.ns * elems
    w0 = m0 + plan.ms * elems // 2
    return elems, m0, w0, w0 + plan.ws * wcols // 2


def relay_pack(
    plan: KdaRelayPlan,
    states: Optional[torch.Tensor],
    windows: Optional[torch.Tensor],
    snap_windows: Optional[torch.Tensor],
) -> torch.Tensor:
    """The ``[world, CB]`` FP32-word send buffer of the all-to-all B: to owner ``j`` its heads of
    every local segment's ``N`` (pass-1 final state, FP32), its heads of every pass-2 segment's ``M``
    (BF16: the operand precision of the relay's ``X @ M``), then its channel shard of the end windows
    and the snapshot windows (BF16, as the pools hold them). ``states`` is the launch's
    ``[nseg + naff, H, K, K]``; ``windows`` the convolution's updated windows; ``snap_windows`` the
    snapshot segments' histories."""
    world = plan.world
    hs = plan.heads // world
    shard = hs * _HEAD_DIM
    elems, m0, w0, size = _layout(plan)
    send = torch.empty(world, size, dtype=torch.float32, device=plan.slots.device)
    if plan.nseg == 0:
        return send
    shape = (hs, _HEAD_DIM, _HEAD_DIM)
    n_part = send[:, : plan.nseg * elems].view(world, plan.nseg, *shape)
    n_part.copy_(states[: plan.nseg].view(plan.nseg, world, *shape).transpose(0, 1))
    if plan.naff:
        m_part = send[:, m0 : m0 + plan.naff * elems // 2].view(torch.bfloat16)
        m_part = m_part.view(world, plan.naff, *shape)
        m_part.copy_(states[plan.nseg :].view(plan.naff, world, *shape).transpose(0, 1))
    wins = torch.cat((windows.index_select(0, plan.end_local), snap_windows))
    if wins.shape[0]:
        nw = wins.shape[0]
        w_part = send[:, w0 : w0 + nw * 3 * shard * WINDOW // 2].view(torch.bfloat16)
        w_part = w_part.view(world, nw, 3, shard, WINDOW)
        w_part.copy_(wins.view(nw, 3, world, shard, WINDOW).permute(2, 0, 1, 3, 4))
    return send


def relay_own(
    plan: KdaRelayPlan,
    recv: torch.Tensor,
    state_pool: torch.Tensor,
    conv_pool: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Compose every request's chain over this rank's heads from ``recv`` (``[world, CB]``, the
    all-to-all B); store the end and snapshot states (FP32) and windows into the pools. Return the
    ``[world, ms, H / world, K, K]`` BF16 send buffer of the all-to-all C (each pass-2 segment's
    start state, to its rank), or None when no rank has a pass-2 segment."""
    world = plan.world
    hs = plan.heads // world
    dim = _HEAD_DIM
    shard = hs * dim
    _, m0, w0, _ = _layout(plan)
    w_end = w0 + plan.ws * 3 * shard * WINDOW // 2
    w_view = recv[:, w0:w_end].view(torch.bfloat16).unflatten(1, (plan.ws, 3 * shard, WINDOW))
    send = None
    if plan.ms:
        send = torch.empty(world, plan.ms, hs, dim, dim, dtype=torch.bfloat16, device=recv.device)
    rubin_kda_relay_chain(
        plan.chain_steps,
        plan.chain_counts,
        plan.chain_init,
        plan.slots,
        recv,
        m0,
        hs,
        state_pool,
        send,
    )
    end_windows = w_view[plan.end_win_q, plan.end_win_l]
    conv_pool[plan.slots.index_select(0, plan.end_win_req)] = end_windows.to(conv_pool.dtype)
    if plan.snap_win_rows.numel():
        snap_windows = w_view[plan.snap_win_q, plan.snap_win_l]
        conv_pool[plan.snap_win_rows] = snap_windows.to(conv_pool.dtype)
    return send


def relay_starts(plan: KdaRelayPlan, recv: torch.Tensor) -> torch.Tensor:
    """The BF16 ``[naff, H, K, K]`` start states of this rank's pass-2 segments from ``recv``
    (``[world, ms, H / world, K, K]``, the all-to-all C: rank ``j`` sent heads of share ``j``)."""
    starts = recv[:, : plan.naff].transpose(0, 1)
    return starts.reshape(plan.naff, plan.heads, _HEAD_DIM, _HEAD_DIM)
