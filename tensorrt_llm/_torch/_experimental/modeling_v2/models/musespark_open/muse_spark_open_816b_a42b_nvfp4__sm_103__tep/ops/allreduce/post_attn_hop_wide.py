# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-side hop of a decoder layer for multi-request decode steps, in one launch.

``post_attn_hop`` beside this file fuses the o_proj all-reduce, the gated residual join
and the pre-FFN norm for decode steps of at most 8 rows. A step of several requests
(9..``WIDE_MAX_ROWS`` rows: 2..128 requests at K = 7) runs the built-in pair instead: the
MNNVL all-reduce of the projection's BF16 partial (one-shot up to 16 rows, two-shot
above), then the residual join -- the cluster boundary up to 32 rows, the Triton
``_gated_residual_after_rmsnorm_kernel`` above. This entry is that pair in one kernel.

Arithmetic. The all-reduce is the built-in one: every rank's partial with BF16 ``-0``
cleared, summed in FP32 in rank order starting from ``+0``, rounded once to BF16 (the
MNNVL one-shot and two-shot both compute this; such a sum is never ``-0``, so the
two-shot's output sanitization changes nothing). The join is the cluster boundary on
its 9..32-row split -- a cluster of 4 CTAs per row, 256 threads, one 16-byte column
vector per thread -- statement for statement, at every row count. So:

* 9..32 rows: bit-identical to the built-in pair.
* 33..``WIDE_MAX_ROWS`` rows: every row is what the built-in pair computes for that row in
  a step of at most 32 rows. Against the built-in Triton join of those row counts the
  only difference is the summation order of the two variance reductions (FP32 sums of
  8192 squares); every other rounding point is the same.

Transport (Lamport). Every slot of the receive memory starts as the sentinel word
``0x80000000``; a 32-bit word that still holds it has not arrived. No payload word is
the sentinel: a partial has BF16 ``-0`` cleared, and the BF16 rounding of an FP32 sum
from ``+0`` of such values is never ``-0`` either (the exact sum is a multiple of
BF16's smallest subnormal, so is its FP32 rounding, and a non-zero one rounds to a
non-zero BF16). The first read that finds no sentinel word returns the final value.
Polling is stale-only: after one read of every awaited slot only the slots still
holding a sentinel word are read again, all in flight together. A reader writes the
sentinel back into each slot it consumed. A call takes one of two transports, chosen by
its row count (``TWO_SHOT_ROWS``):

* One-shot. A rank publishes its sanitized partial into slab ``rank`` of the current
  buffer on every peer (one store to the multicast address, or one unicast store per
  peer) and sums the peer slabs of its own buffer. A slot (buffer, source, row, column
  vector) has one writer per call (the source rank's CTA of that row and column slice)
  and one reader (the receiving rank's CTA of the same row and slice). NVLink ingress
  per rank: ``(ranks - 1) * rows * 16 KiB``.
* Two-shot. Row ``t`` is owned by rank ``t % ranks``. A non-owner stores its sanitized
  partial into slab ``rank`` of the owner's scatter area (slot: buffer, source,
  ``t // ranks``, column vector); the owner sums every rank's partial (the one-shot's
  arithmetic) and publishes the four BF16 words of the sum into the gather slot
  (buffer, ``t``, column vector) of every rank, where each non-owner reads it. Each
  slot again has one writer and one reader per call; the owner's own gather slot
  receives its multicast copy and is never read (the owner of a row never changes).
  Ingress per rank: about half the one-shot's, behind a second dependent hop.

Buffer reuse. The buffer index is the call's parity. A rank can write call ``n + 2``
into a peer's buffer only after it read that peer's data of call ``n + 1`` (one-shot:
the peer's slab; two-shot: the peer's gather words of a row it owns, or -- for a gather
store -- the peer's scatter of the same call ``n + 2``, published after its call
``n + 1``), and the peer publishes that only after its call ``n`` -- including the
resets -- completed (calls are serialized on one stream and every kernel between two
calls waits on its predecessor before it completes). Every call of either transport
reads data of every rank (rows > ranks, so every rank owns a row), so this holds across
any sequence of one-shot and two-shot calls. A slot is therefore never rewritten while
it holds unread data, and no value left from an earlier call can satisfy a later read.

Parity on the device. The CTAs of every call add exactly ``GRID_STEP`` to a per-rank
arrival counter (CTA 0 adds ``GRID_STEP + 1 - grid``, every other CTA one), each after
all its threads read the counter; a CTA of call ``n`` reads ``GRID_STEP * n + k`` with
``0 <= k < GRID_STEP``, so ``(counter >> GRID_SHIFT) & 1`` is the call's parity on every
CTA. ``2**32`` is a multiple of ``2 * GRID_STEP``, so the parity survives the wrap.
``GRID_STEP`` is the workspace's: at least ``CLUSTER`` CTAs per row of its capacity
(``grid_shift``).

Progress. A CTA publishes its partial (one-shot: to every peer; two-shot: to the owner,
unless it is the owner) before it polls, and polls only slots of its own row and column
slice, which only the same CTA index of the peers writes; a two-shot owner publishes the
sum as soon as its own polls complete, which needs nothing but that publication from
the same CTA index of the peers. With CTAs dispatched in grid order on every rank (the
dispatch the built-in MNNVL all-reduce kernels and the expert finalize also rely on),
the lowest-indexed unfinished cluster is resident on every rank once every lower one has
finished, publishes, and finishes; so every cluster finishes even when the grid
(``4 * rows`` CTAs) exceeds what is resident at once.

Workspace per rank: the one-shot area ``2 * ranks * (TWO_SHOT_ROWS - 1) * 16 KiB`` plus
the two-shot area ``2 * 2 * max_rows * 16 KiB`` (scatter and gather), its own fabric
allocation and counter, never shared with the 8-row hop, the built-in MNNVL all-reduce
or the expert finalize. Calls on one workspace must be issued in the same order on every
rank and serialized on one stream (graph replay included).

Capacities. The served workspace holds calls of up to ``MAX_ROWS`` rows (64 requests). A
decode step of ``MAX_ROWS + 1..WIDE_MAX_ROWS`` rows (65..128 requests: a generation worker
of ``max_batch_size`` above 64) runs on a second workspace of capacity ``WIDE_MAX_ROWS``,
two-shot only, with its own receive memory, counter and ``GRID_STEP``; it is built on the
first such step (never inside a capture), so an engine whose steps stay within ``MAX_ROWS``
rows allocates nothing for it. Each workspace keeps the ownership, reuse and progress
argument above for its own calls (they are serialized on one stream and issued in the same
order on every rank); the two share no memory, and the call of either depends only on its
row count, which every rank shares. The two-shot kernel of the larger capacity is the same
kernel with the capacity's scatter and gather strides and grid step.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from . import post_attn_hop as _hop

#: The kernels launch with programmatic dependent launch: ``TRTLLM_ENABLE_PDL`` on is a
#: precondition of every composition that builds them.
_LAUNCH_PDL = True

HIDDEN = _hop.HIDDEN
VECTOR = _hop.VECTOR
SENTINEL = _hop.SENTINEL
BUFFERS = 2
#: Fewest rows this entry takes; fewer rows are the 8-row hop's.
MIN_ROWS = _hop.MAX_ROWS + 1
#: Most rows of a call the served transport is sized for: 64 requests at K = 7.
MAX_ROWS = 512
#: Most rows of a call the larger transport is sized for: 128 requests at K = 7, the decode
#: steps of a generation worker of ``max_batch_size`` up to 128 (built on the first such step).
WIDE_MAX_ROWS = 1024
#: The transport capacities, ascending: a call runs on the smallest that holds its rows.
CAPACITIES = (MAX_ROWS, WIDE_MAX_ROWS)
#: Fewest rows a call takes the two-shot transport for (one-shot below): on GB300 at four
#: ranks the one-shot is the faster transport up to 24 rows, the two-shot from 32.
TWO_SHOT_ROWS = 32
#: The served workspace's publication: unicast stores to each peer (the two-shot gather
#: then skips the owner's own copy).
SERVED_MULTICAST = False
#: The served workspace's launch mode: after the o_proj GEMM rather than as its programmatic
#: dependent, so the hop's CTAs do not take SM slots from the GEMM's last wave.
SERVED_EARLY_LAUNCH = False
ROW_BYTES = HIDDEN * 2
#: The cluster boundary's split for 9..32 rows; this entry uses it at every row count.
CLUSTER = 4
THREADS = HIDDEN // (CLUSTER * VECTOR)
#: What one call adds to the arrival counter, whatever its grid (at most CLUSTER * MAX_ROWS CTAs),
#: on the served workspace.
GRID_SHIFT = 11
GRID_STEP = 1 << GRID_SHIFT
assert GRID_STEP >= CLUSTER * MAX_ROWS


def grid_shift(max_rows: int) -> int:
    """log2 of what one call adds to a workspace's arrival counter: the served ``GRID_SHIFT``,
    or the smallest power of two at least ``CLUSTER * max_rows`` above it."""
    return max(GRID_SHIFT, (CLUSTER * max_rows - 1).bit_length())


assert grid_shift(MAX_ROWS) == GRID_SHIFT
assert (1 << grid_shift(WIDE_MAX_ROWS)) >= CLUSTER * WIDE_MAX_ROWS
CONTROL_WORDS = 4

HAS_CUTE_DSL = _hop.HAS_CUTE_DSL

if HAS_CUTE_DSL:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    from .post_attn_hop import (
        _arrive,
        _float_bits,
        _load_counter,
        _load_x4,
        _poll_x4,
        _reset_x4,
        _sanitize_bf16x2,
        _stale_bit,
        _store_multicast_x4,
        _store_peer_x4,
        _unpack_bf16x2,
    )

    @dsl_user_op
    def _pack_bf16x2(low, high, *, loc=None, ip=None):
        """``low`` and ``high`` rounded to BF16 (RN) into the low and high halves of a word."""
        result = llvm.inline_asm(
            T.i32(),
            [Float32(high).ir_value(loc=loc, ip=ip), Float32(low).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Uint32(result)

    class _WideHopKernel:
        """All-reduce + attention-side cluster boundary for 9..``max_rows`` rows.

        CTA ``b`` handles row ``b // CLUSTER`` and the ``b % CLUSTER``-th quarter of the
        8192 columns; each of its 256 threads owns one 16-byte column vector -- the
        cluster boundary's 9..32-row map, which fixes the variance reduction order.
        ``two_shot`` selects the transport; ``one_shot_rows`` sizes the one-shot slabs,
        which come first in the receive memory, followed by the two-shot scatter and
        gather areas. ``early_launch`` launches the grid as a programmatic dependent of
        the previous kernel (its CTAs may become resident before that kernel completes and
        wait in ``griddepcontrol.wait``); without it the grid starts after the previous
        kernel, and the wait returns at once. The early trigger for this grid's own
        dependents is independent of it. ``max_rows`` is the workspace's capacity (the
        scatter and gather strides) and fixes the counter's grid step (``grid_shift``).
        """

        def __init__(
            self,
            ranks: int,
            multicast: bool,
            post_eps: float,
            pre_eps: float,
            norm_fp32: bool,
            pdl: bool,
            two_shot: bool,
            one_shot_rows: int,
            early_launch: bool,
            max_rows: int = MAX_ROWS,
        ):
            self.ranks = ranks
            self.max_rows = max_rows
            self.grid_shift = grid_shift(max_rows)
            self.grid_step = 1 << self.grid_shift
            self.cluster = CLUSTER
            self.threads = THREADS
            self.iters = HIDDEN // (CLUSTER * THREADS * VECTOR)
            self.warps = self.threads // 32
            self.slots = self.cluster * self.warps
            self.multicast = multicast
            self.post_eps, self.pre_eps = float(post_eps), float(pre_eps)
            self.norm_fp32 = norm_fp32
            self.pdl = pdl
            self.early_launch = early_launch
            self.two_shot = two_shot
            self.slab_bytes = one_shot_rows * ROW_BYTES
            self.scatter_rows = max_rows // ranks
            self.scatter_base = BUFFERS * ranks * self.slab_bytes
            self.gather_base = self.scatter_base + BUFFERS * ranks * self.scatter_rows * ROW_BYTES

        @cute.jit
        def __call__(
            self,
            partial: cute.Pointer,
            residual: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            mcast: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            peer4: cute.Pointer,
            peer5: cute.Pointer,
            peer6: cute.Pointer,
            peer7: cute.Pointer,
            rows: Int32,
            rank: Int32,
            dry: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partial,
                residual,
                alpha,
                beta,
                weight,
                output,
                next_output,
                control,
                receive,
                mcast,
                peer0,
                peer1,
                peer2,
                peer3,
                peer4,
                peer5,
                peer6,
                peer7,
                rows,
                rank,
                dry,
            ).launch(
                grid=[rows * self.cluster, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 2 + 16,
                stream=stream,
                use_pdl=self.pdl and self.early_launch,
            )

        @cute.jit
        def load(self, ptr: cute.Pointer, dtype: cutlass.Constexpr):
            layout = cute.make_layout((8, self.iters), stride=(1, self.threads * self.cluster * 8))
            reg = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), dtype)
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
            cute.copy(atom, cute.make_tensor(ptr.align(16), layout), reg)
            return reg

        @cute.jit
        def reduce(
            self, value: Float32, partials: cute.Tensor, bars: cute.Tensor, phase: cutlass.Constexpr
        ):
            # Statement for statement the cluster boundary's reduction.
            tid = cute.arch.thread_idx()[0]
            lane, warp = tid % 32, tid // 32
            rank = cute.arch.block_idx()[0] % self.cluster
            value = cute.arch.warp_reduction_sum(value)
            if lane < self.cluster:
                slot = partials.iterator + (phase * self.slots + rank * self.warps + warp) * 4
                cute.arch.store_async_dsmem(
                    slot.align(16),
                    (_float_bits(value), Int32(0), Int32(0), Int32(0)),
                    bars.iterator + phase,
                    lane,
                )
            cute.arch.mbarrier_wait(bars.iterator + phase, 0)
            total = Float32(0)
            for slot in range(lane, self.slots, 32):
                total += partials[(phase * self.slots + slot) * 4]
            return cute.arch.warp_reduction_sum(total)

        @cute.kernel
        def kernel(
            self,
            partial: cute.Pointer,
            residual: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            mcast: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            peer4: cute.Pointer,
            peer5: cute.Pointer,
            peer6: cute.Pointer,
            peer7: cute.Pointer,
            rows: Int32,
            rank: Int32,
            dry: Int32,
        ):
            peers = (peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7)
            tid = cute.arch.thread_idx()[0]
            crank = cute.arch.block_idx()[0] % self.cluster
            row = cute.arch.block_idx()[0] // self.cluster
            offset = cute.assume((crank * self.threads + tid) * 8, divby=8)
            smem = utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((self.slots * 4 * 2,)), byte_alignment=16
            )
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((2,)), byte_alignment=8)
            # A dry launch only makes the driver load the module.
            if dry == 0:
                control_address = control.toint()
                if tid < 2:
                    cute.arch.mbarrier_init(bars.iterator + tid, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + tid, self.slots * 16)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
                # Nothing above touches global memory: the projection, the residual,
                # the coefficients and the transport are all read after this wait.
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                # The arrival counter holds GRID_STEP per completed call plus less than
                # GRID_STEP of this one (this CTA arrives only after all its threads have
                # read it), so its GRID_STEP place is the call's parity.
                counter = _load_counter(control_address + 4)
                buffer = Int64((counter >> self.grid_shift) & Uint32(1))
                slot = (Int64(row) * HIDDEN + Int64(offset)) * 2
                words = _load_x4(partial.toint() + slot)
                a = self.load(alpha + offset, Float32)
                b = self.load(beta + offset, Float32)
                w = self.load(weight + offset, cutlass.BFloat16)
                r = self.load(residual + row * HIDDEN + offset, Float32)

                # This rank's partial with BF16 -0 cleared: no word is the sentinel.
                own = (
                    _sanitize_bf16x2(words[0]),
                    _sanitize_bf16x2(words[1]),
                    _sanitize_bf16x2(words[2]),
                    _sanitize_bf16x2(words[3]),
                )
                receive_address = receive.toint()
                x = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                if cutlass.const_expr(not self.two_shot):
                    # One-shot: publish into this rank's slab on every rank.
                    own_slab = (buffer * self.ranks + Int64(rank)) * self.slab_bytes
                    if cutlass.const_expr(self.multicast):
                        _store_multicast_x4(mcast.toint() + own_slab + slot, *own)
                    else:
                        for p in cutlass.range_constexpr(self.ranks):
                            if rank != p:
                                _store_peer_x4(peers[p].toint() + own_slab + slot, *own)

                    # Stale-only polling of the peer slots of the local buffer; this rank's
                    # own slot is never read (its value is `own`). Bit `p` of `stale` is set
                    # while slot `p` still holds a sentinel word; a load whose bit is clear
                    # keeps the words it already has.
                    values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                    for p in cutlass.range_constexpr(self.ranks):
                        address = (
                            receive_address + (buffer * self.ranks + p) * self.slab_bytes + slot
                        )
                        v0, v1, v2, v3 = _poll_x4(
                            address, Uint32(rank != p), own[0], own[1], own[2], own[3]
                        )
                        values[p, 0] = v0
                        values[p, 1] = v1
                        values[p, 2] = v2
                        values[p, 3] = v3
                    stale = Uint32(0)
                    for p in cutlass.range_constexpr(self.ranks):
                        stale = stale | _stale_bit(
                            values[p, 0], values[p, 1], values[p, 2], values[p, 3], Uint32(1 << p)
                        )
                    while stale != Uint32(0):
                        polled = stale
                        for p in cutlass.range_constexpr(self.ranks):
                            address = (
                                receive_address + (buffer * self.ranks + p) * self.slab_bytes + slot
                            )
                            v0, v1, v2, v3 = _poll_x4(
                                address,
                                (polled >> Uint32(p)) & Uint32(1),
                                values[p, 0],
                                values[p, 1],
                                values[p, 2],
                                values[p, 3],
                            )
                            values[p, 0] = v0
                            values[p, 1] = v1
                            values[p, 2] = v2
                            values[p, 3] = v3
                        stale = Uint32(0)
                        for p in cutlass.range_constexpr(self.ranks):
                            stale = stale | _stale_bit(
                                values[p, 0],
                                values[p, 1],
                                values[p, 2],
                                values[p, 3],
                                Uint32(1 << p),
                            )
                    # Retire every consumed slot before this grid can complete.
                    for p in cutlass.range_constexpr(self.ranks):
                        address = (
                            receive_address + (buffer * self.ranks + p) * self.slab_bytes + slot
                        )
                        _reset_x4(address, Uint32(rank != p))
                    # Past the last transport access: a dependent may start its prologue now;
                    # its griddepcontrol.wait still returns only once this grid has completed.
                    if cutlass.const_expr(self.pdl):
                        cute.arch.griddepcontrol_launch_dependents()

                    # The built-in all-reduce sum: FP32 from +0, ranks in order, one RN round.
                    totals = [Float32(0.0)] * VECTOR
                    for p in cutlass.range_constexpr(self.ranks):
                        for k in cutlass.range_constexpr(4):
                            low, high = _unpack_bf16x2(values[p, k])
                            totals[2 * k] = totals[2 * k] + low
                            totals[2 * k + 1] = totals[2 * k + 1] + high
                    for v in cutlass.range_constexpr(8):
                        x[v, 0] = cutlass.BFloat16(totals[v])
                else:
                    # Two-shot: row `row` is owned by rank `row % ranks`. `scatter` is the
                    # row's slot of source 0 in the owner's scatter area, `gather` its slot
                    # in every rank's gather area (both in the current buffer).
                    owner = row % self.ranks
                    source_bytes = self.scatter_rows * ROW_BYTES
                    scatter = (
                        Int64(self.scatter_base)
                        + buffer * (self.ranks * source_bytes)
                        + Int64(row // self.ranks) * ROW_BYTES
                        + Int64(offset) * 2
                    )
                    gather = Int64(self.gather_base) + buffer * (self.max_rows * ROW_BYTES) + slot
                    reduced = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                    values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                    if owner != rank:
                        # Scatter this rank's partial into its slab on the owner, then poll
                        # the row's gather slot until the owner's sum has arrived.
                        for p in cutlass.range_constexpr(self.ranks):
                            if owner == p:
                                _store_peer_x4(
                                    peers[p].toint() + scatter + Int64(rank) * source_bytes, *own
                                )
                        g0, g1, g2, g3 = _poll_x4(
                            receive_address + gather, Uint32(1), own[0], own[1], own[2], own[3]
                        )
                        reduced[0] = g0
                        reduced[1] = g1
                        reduced[2] = g2
                        reduced[3] = g3
                        waiting = _stale_bit(
                            reduced[0], reduced[1], reduced[2], reduced[3], Uint32(1)
                        )
                        while waiting != Uint32(0):
                            h0, h1, h2, h3 = _poll_x4(
                                receive_address + gather,
                                Uint32(1),
                                reduced[0],
                                reduced[1],
                                reduced[2],
                                reduced[3],
                            )
                            reduced[0] = h0
                            reduced[1] = h1
                            reduced[2] = h2
                            reduced[3] = h3
                            waiting = _stale_bit(
                                reduced[0], reduced[1], reduced[2], reduced[3], Uint32(1)
                            )
                        _reset_x4(receive_address + gather, Uint32(1))
                    else:
                        # The owner: stale-only polling of the peers' scatter slots of this
                        # row (this rank's own value is `own`), as the one-shot polls.
                        for p in cutlass.range_constexpr(self.ranks):
                            v0, v1, v2, v3 = _poll_x4(
                                receive_address + scatter + p * source_bytes,
                                Uint32(rank != p),
                                own[0],
                                own[1],
                                own[2],
                                own[3],
                            )
                            values[p, 0] = v0
                            values[p, 1] = v1
                            values[p, 2] = v2
                            values[p, 3] = v3
                        stale = Uint32(0)
                        for p in cutlass.range_constexpr(self.ranks):
                            stale = stale | _stale_bit(
                                values[p, 0],
                                values[p, 1],
                                values[p, 2],
                                values[p, 3],
                                Uint32(1 << p),
                            )
                        while stale != Uint32(0):
                            polled = stale
                            for p in cutlass.range_constexpr(self.ranks):
                                v0, v1, v2, v3 = _poll_x4(
                                    receive_address + scatter + p * source_bytes,
                                    (polled >> Uint32(p)) & Uint32(1),
                                    values[p, 0],
                                    values[p, 1],
                                    values[p, 2],
                                    values[p, 3],
                                )
                                values[p, 0] = v0
                                values[p, 1] = v1
                                values[p, 2] = v2
                                values[p, 3] = v3
                            stale = Uint32(0)
                            for p in cutlass.range_constexpr(self.ranks):
                                stale = stale | _stale_bit(
                                    values[p, 0],
                                    values[p, 1],
                                    values[p, 2],
                                    values[p, 3],
                                    Uint32(1 << p),
                                )
                        for p in cutlass.range_constexpr(self.ranks):
                            _reset_x4(
                                receive_address + scatter + p * source_bytes, Uint32(rank != p)
                            )
                        # The one-shot's sum and RN round, packed as the four BF16x2 words
                        # every rank of the group then uses.
                        sums = [Float32(0.0)] * VECTOR
                        for p in cutlass.range_constexpr(self.ranks):
                            for k in cutlass.range_constexpr(4):
                                lo, hi = _unpack_bf16x2(values[p, k])
                                sums[2 * k] = sums[2 * k] + lo
                                sums[2 * k + 1] = sums[2 * k + 1] + hi
                        for k in cutlass.range_constexpr(4):
                            reduced[k] = _pack_bf16x2(sums[2 * k], sums[2 * k + 1])
                        # Publish the sum into the row's gather slot on every rank.
                        if cutlass.const_expr(self.multicast):
                            _store_multicast_x4(
                                mcast.toint() + gather,
                                reduced[0],
                                reduced[1],
                                reduced[2],
                                reduced[3],
                            )
                        else:
                            for p in cutlass.range_constexpr(self.ranks):
                                if rank != p:
                                    _store_peer_x4(
                                        peers[p].toint() + gather,
                                        reduced[0],
                                        reduced[1],
                                        reduced[2],
                                        reduced[3],
                                    )
                    # Past the last transport access (see the one-shot).
                    if cutlass.const_expr(self.pdl):
                        cute.arch.griddepcontrol_launch_dependents()
                    for k in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(reduced[k])
                        x[2 * k, 0] = cutlass.BFloat16(low)
                        x[2 * k + 1, 0] = cutlass.BFloat16(high)

                # From here on: the cluster boundary with gain off, statement for statement.
                scaled = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                updated = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                normed = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                total = Float32(0)
                for p in cutlass.range_constexpr(self.iters):
                    for v in cutlass.range_constexpr(8):
                        value = Float32(x[v, p])
                        scaled[v, p] = value
                        total += value * value
                total = self.reduce(total, partials, bars, 0)
                inv = cute.math.rsqrt(total / Float32(8192) + Float32(self.post_eps), fastmath=True)
                total = Float32(0)
                for p in cutlass.range_constexpr(self.iters):
                    for v in cutlass.range_constexpr(8):
                        value = Float32(cutlass.BFloat16(scaled[v, p] * inv))
                        u = a[v, p] * r[v, p] + b[v, p] * value
                        updated[v, p] = u
                        if cutlass.const_expr(self.norm_fp32):
                            scaled[v, p] = u
                            total += u * u
                        else:
                            rounded = Float32(cutlass.BFloat16(u))
                            scaled[v, p] = rounded
                            total += rounded * rounded
                total = self.reduce(total, partials, bars, 1)
                inv = cute.math.rsqrt(total / Float32(8192) + Float32(self.pre_eps), fastmath=True)
                for p in cutlass.range_constexpr(self.iters):
                    for v in cutlass.range_constexpr(8):
                        normed[v, p] = cutlass.BFloat16(
                            scaled[v, p] * inv * (Float32(w[v, p]) + Float32(1))
                        )
                layout = cute.make_layout(
                    (8, self.iters), stride=(1, self.threads * self.cluster * 8)
                )
                atom32 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                )
                atom16 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
                )
                cute.copy(
                    atom32,
                    updated,
                    cute.make_tensor((output + row * HIDDEN + offset).align(16), layout),
                )
                cute.copy(
                    atom16,
                    normed,
                    cute.make_tensor((next_output + row * HIDDEN + offset).align(16), layout),
                )
                # Both reductions waited on this CTA's own mbarrier for every byte its
                # peers send, so no peer touches this CTA's shared memory any more. This
                # barrier only orders every thread's counter read before the arrival.
                cute.arch.barrier()
                if tid == 0:
                    block = cute.arch.block_idx()[0]
                    blocks = cute.arch.grid_dim()[0]
                    amount = Uint32(1)
                    if block == 0:
                        amount = Uint32(self.grid_step + 1 - blocks)
                    _arrive(control_address + 4, amount)

    _POINTER_DTYPES = (
        BFloat16,  # partial
        Float32,  # residual
        Float32,  # alpha
        Float32,  # beta
        BFloat16,  # weight
        Float32,  # output
        BFloat16,  # next_output
        Uint32,  # control
        Uint32,  # receive
        Uint32,  # mcast
    ) + (Uint32,) * _hop.MAX_RANKS

    @functools.lru_cache(maxsize=32)
    def _compiled_kernel(
        device_index: int,
        ranks: int,
        multicast: bool,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        pdl: bool,
        two_shot: bool,
        one_shot_rows: int,
        early_launch: bool,
        max_rows: int = MAX_ROWS,
    ):
        kernel = _WideHopKernel(
            ranks,
            multicast,
            post_eps,
            pre_eps,
            norm_fp32,
            pdl,
            two_shot,
            one_shot_rows,
            early_launch,
            max_rows,
        )
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *[
                    make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
                    for dtype in _POINTER_DTYPES
                ],
                Int32(MIN_ROWS),
                Int32(0),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


def receive_layout(ranks: int, two_shot_rows: int, max_rows: int = MAX_ROWS) -> Tuple[int, int]:
    """``(one_shot_rows, bytes)`` of the receive memory for calls split at ``two_shot_rows``,
    on a workspace of capacity ``max_rows`` (``CAPACITIES``).

    The one-shot slabs hold ``one_shot_rows`` rows (0 when no call takes the one-shot);
    the two-shot scatter and gather areas follow when some call takes the two-shot.
    """
    if max_rows not in CAPACITIES:
        raise ValueError(f"max_rows must be one of {CAPACITIES}, got {max_rows}")
    if not MIN_ROWS <= two_shot_rows <= max_rows + 1:
        raise ValueError(
            f"two_shot_rows must be in {MIN_ROWS}..{max_rows + 1}, got {two_shot_rows}"
        )
    one_shot_rows = two_shot_rows - 1 if two_shot_rows > MIN_ROWS else 0
    size = BUFFERS * ranks * one_shot_rows * ROW_BYTES
    if two_shot_rows <= max_rows:
        size += BUFFERS * (ranks * (max_rows // ranks) + max_rows) * ROW_BYTES
    return one_shot_rows, size


class WideHopWorkspace:
    """Transport memory and device state of the wide hop for one tensor-parallel group.

    Private to this op (see the module docstring). One instance serves every layer of a
    model; calls must be issued in the same order on every rank and serialized on one
    stream. Construction is collective over the group (fabric allocation, sentinel fill,
    kernel compilation and a dry launch per variant) and either succeeds on every rank or
    raises on every rank. A variant a caller did not pre-build compiles on its first eager
    call (never inside a capture). Calls of ``two_shot_rows`` rows or more take the
    two-shot transport, fewer the one-shot; the receive memory holds the areas of the
    transports that range uses. ``early_launch`` is the kernel's launch mode
    (``_WideHopKernel``). The defaults are the served configuration (``SERVED_MULTICAST``,
    ``TWO_SHOT_ROWS``, ``SERVED_EARLY_LAUNCH``) on the served capacity ``MAX_ROWS``;
    ``max_rows=WIDE_MAX_ROWS`` builds the larger transport.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        multicast: bool = SERVED_MULTICAST,
        variants: Sequence[Tuple[float, float, bool]] = (),
        two_shot_rows: int = TWO_SHOT_ROWS,
        early_launch: bool = SERVED_EARLY_LAUNCH,
        max_rows: int = MAX_ROWS,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.multicast = bool(multicast)
        self.pdl = _LAUNCH_PDL
        self.early_launch = bool(early_launch)
        if self.ranks not in (4, _hop.MAX_RANKS):
            raise ValueError(f"the wide hop serves 4 or 8 ranks, got {self.ranks}")
        self.two_shot_rows = int(two_shot_rows)
        self.max_rows = int(max_rows)
        self.one_shot_rows, self.workspace_bytes = receive_layout(
            self.ranks, self.two_shot_rows, self.max_rows
        )
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._kernels = {}
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("CuTe DSL is unavailable")
            if torch.cuda.get_device_capability(self.device) not in ((10, 3), (10, 7)):
                raise RuntimeError("the wide hop runs on SM103 and SM107 only")
            use_fabric = mapping.is_multi_node()
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, self.workspace_bytes, mapping, use_fabric
            )
            words = self.workspace_bytes // 4
            self._uc = [
                self._handle.get_uc_buffer(peer, (words,), torch.int32, 0)
                for peer in range(self.ranks)
            ]
            self._mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            with torch.cuda.device(self.device):
                self._uc[self.rank].fill_(-(2**31))
                self.control = torch.tensor(
                    [0] * CONTROL_WORDS, dtype=torch.int32, device=self.device
                )
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                f"wide post-attention hop workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        receive = self._uc[self.rank].data_ptr()
        peers = [t.data_ptr() for t in self._uc] + [receive] * (_hop.MAX_RANKS - self.ranks)
        self._fixed = (self.control.data_ptr(), receive, self._mc.data_ptr(), *peers)
        self.build_variants(variants)

    def transports(self) -> Tuple[bool, ...]:
        """The transports this workspace's calls take: ``False`` one-shot, ``True`` two-shot."""
        return tuple(
            two_shot
            for two_shot, used in (
                (False, self.one_shot_rows > 0),
                (True, self.two_shot_rows <= self.max_rows),
            )
            if used
        )

    def build_variants(self, variants: Sequence[Tuple[float, float, bool]]) -> None:
        """Compile and load each ``(post_eps, pre_eps, norm_fp32)`` not built yet (collective)."""
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        error: Optional[BaseException] = None
        try:
            for post_eps, pre_eps, norm_fp32 in variants:
                for two_shot in self.transports():
                    self._kernel(post_eps, pre_eps, norm_fp32, two_shot)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "wide post-attention hop kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _kernel(self, post_eps: float, pre_eps: float, norm_fp32: bool, two_shot: bool):
        key = (float(post_eps), float(pre_eps), bool(norm_fp32), bool(two_shot))
        compiled = self._kernels.get(key)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"wide post-attention hop variant {key} was not built before graph capture"
            )
        compiled = _compiled_kernel(
            self.device.index,
            self.ranks,
            self.multicast,
            float(post_eps),
            float(pre_eps),
            bool(norm_fp32),
            self.pdl,
            bool(two_shot),
            self.one_shot_rows,
            self.early_launch,
            self.max_rows,
        )
        # A dry launch loads the module now, not inside a capture.
        dummy = self.control.data_ptr()
        compiled(*([dummy] * 7), *self._fixed, MIN_ROWS, self.rank, 1, self._stream())
        self._kernels[key] = compiled
        return compiled

    def _stream(self):
        return cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def launch(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if partial.device != self.device:
            raise ValueError(
                f"post_attn_hop_wide inputs are on {partial.device}, its workspace on {self.device}"
            )
        rows = partial.shape[0]
        if rows > self.max_rows:
            raise ValueError(
                f"post_attn_hop_wide: {rows} rows on a workspace of capacity {self.max_rows}"
            )
        compiled = self._kernel(post_eps, pre_eps, norm_fp32, rows >= self.two_shot_rows)
        updated = torch.empty_like(residual)
        normalized = torch.empty_like(partial)
        compiled(
            partial.data_ptr(),
            residual.data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            next_weight.data_ptr(),
            updated.data_ptr(),
            normalized.data_ptr(),
            *self._fixed,
            rows,
            self.rank,
            0,
            self._stream(),
        )
        return updated, normalized


_SHARED: dict = {}


def shared_wide_workspace(
    mapping: Mapping,
    *,
    multicast: bool = SERVED_MULTICAST,
    variants: Sequence[Tuple[float, float, bool]] = (),
    early_launch: bool = SERVED_EARLY_LAUNCH,
    max_rows: int = MAX_ROWS,
) -> WideHopWorkspace:
    """The one wide-hop workspace of this group, publication, launch mode and capacity, built
    on first use. The larger capacity (``WIDE_MAX_ROWS``) takes only calls of more than
    ``MAX_ROWS`` rows, so it is two-shot only.

    Collective on first use.
    """
    two_shot_rows = TWO_SHOT_ROWS if max_rows == MAX_ROWS else MIN_ROWS
    key = (mapping, bool(multicast), _LAUNCH_PDL, two_shot_rows, bool(early_launch), max_rows)
    workspace = _SHARED.get(key)
    if workspace is None:
        workspace = WideHopWorkspace(
            mapping,
            multicast=multicast,
            variants=variants,
            two_shot_rows=two_shot_rows,
            early_launch=early_launch,
            max_rows=max_rows,
        )
        _SHARED[key] = workspace
    else:
        workspace.build_variants(variants)
    return workspace


def wide_takes_rows(rows: int) -> bool:
    """Whether a decode step of ``rows`` rows is this entry's (the same on every rank)."""
    return MIN_ROWS <= rows <= WIDE_MAX_ROWS


def capacity_for(rows: int) -> int:
    """The capacity (``CAPACITIES``) of the workspace a call of ``rows`` rows runs on."""
    return next(capacity for capacity in CAPACITIES if rows <= capacity)


def wide_takes(
    partial: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
) -> bool:
    """Whether the wide hop computes this call: row count, dtypes, shapes and devices only.

    Every rank of the group gives the same answer for the same layer call (no address or
    stride is consulted), which lets a caller choose between this collective and the
    built-in all-reduce on it.
    """
    rows = partial.shape[0] if partial.ndim == 2 else 0
    if not wide_takes_rows(rows):
        return False
    if (
        partial.dtype != torch.bfloat16
        or residual.dtype != torch.float32
        or alpha.dtype != torch.float32
        or beta.dtype != torch.float32
        or next_weight.dtype != torch.bfloat16
    ):
        return False
    if tuple(partial.shape) != (rows, HIDDEN) or tuple(residual.shape) != (rows, HIDDEN):
        return False
    vectors = (alpha, beta, next_weight)
    if any(tuple(t.shape) != (HIDDEN,) for t in vectors):
        return False
    return all(t.is_cuda and t.device == partial.device for t in (partial, residual, *vectors))


def wide_supported(
    partial: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
) -> bool:
    """Whether the kernel reads these inputs correctly as they are: ``wide_takes`` and layout."""
    return wide_takes(partial, residual, alpha, beta, next_weight) and _hop.hop_layout_ok(
        partial, residual, alpha, beta, next_weight
    )


class PostAttnHopWide(OpWrapper):
    """All-reduce the o_proj partial, join it into the residual, apply the pre-FFN norm.

    Takes 9..1024 rows.

    Returns ``(updated, ffn_input)`` as ``post_attn_hop`` does: the new FP32 residual
    ``alpha * residual + beta * bf16(rmsnorm(x))`` and the BF16
    ``bf16(rmsnorm(prenorm_input) * (1 + next_weight))``, where ``x`` is the BF16 sum of
    every rank's ``partial`` and ``prenorm_input`` is ``bf16(updated)`` (or ``updated``
    itself with ``norm_fp32``).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    note = """
    Metadata consumed: none. Call site: once per decoder layer per decode step of
    9..1024 rows, from the target's attention-side residual join, in place of the built-in
    MNNVL all-reduce and residual join; calls of more than 512 rows on the larger transport.

    Bit-identical to the built-in pair at 9..32 rows. At 33..1024 rows each row is what the
    built-in pair computes for it at 9..32 rows; against the built-in Triton join the two
    variance reductions are summed in another order (the only difference), gated by its
    multi-GPU test within a band it states in units of each row's largest magnitude: where
    a gated residual term and the normalized branch nearly cancel, one BF16 rounding of
    the normalized branch moves an element by many ulps of its own value, in either
    implementation. The one-shot and two-shot transports, and the two capacities, give the
    same bits (the owner computes the one-shot's sum and every rank uses its words).

    Loud preconditions: `wide_supported` must hold. A caller choosing between this
    collective and the built-in all-reduce decides on `wide_takes` alone, which every rank
    answers alike, and makes the layout fit with `post_attn_hop.readable`. Not certified:
    rows outside 9..1024, widths other than 8192, a weighted attention-side post norm,
    eight ranks.
    """

    def __call__(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        workspace: WideHopWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not wide_supported(partial, residual, alpha, beta, next_weight):
            raise ValueError(
                f"post_attn_hop_wide takes a contiguous BF16 [{MIN_ROWS}..{WIDE_MAX_ROWS}, 8192] "
                "partial, an FP32 residual of the same shape, FP32 alpha/beta and a BF16 weight "
                f"of 8192, 16-byte aligned; got {partial.dtype} {tuple(partial.shape)} / "
                f"{residual.dtype} {tuple(residual.shape)}"
            )
        return workspace.launch(
            partial, residual, alpha, beta, next_weight, post_eps, pre_eps, norm_fp32
        )

    def reference(
        self,
        per_rank_partials: Sequence[torch.Tensor],
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        workspace: Optional[WideHopWorkspace],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The op's rounding points, float64 in between (``post_attn_hop``'s reference).

        The documented collective exception: takes the group's partials in ascending rank
        order instead of this rank's ``partial``. ``workspace`` is ignored.
        """
        return _hop.post_attn_hop.reference(
            per_rank_partials,
            residual,
            alpha,
            beta,
            next_weight,
            post_eps,
            pre_eps,
            norm_fp32,
            None,
        )


post_attn_hop_wide = PostAttnHopWide()


def log_once_enabled(
    layers: int,
    multicast: bool,
    two_shot_rows: int = TWO_SHOT_ROWS,
    early_launch: bool = SERVED_EARLY_LAUNCH,
    min_rows: int = MIN_ROWS,
) -> None:
    if two_shot_rows > MAX_ROWS:
        transports = f"one-shot AR at {min_rows}..{MAX_ROWS} rows"
    elif two_shot_rows <= min_rows:
        transports = f"two-shot AR at {min_rows}..{MAX_ROWS} rows"
    else:
        transports = (
            f"one-shot AR at {min_rows}..{two_shot_rows - 1} rows, "
            f"two-shot at {two_shot_rows}..{MAX_ROWS}"
        )
    transports += (
        f"; two-shot AR at {MAX_ROWS + 1}..{WIDE_MAX_ROWS} rows on a second transport "
        "built on the first such eager step"
    )
    launch = (
        "launched as a programmatic dependent" if early_launch else "launched after its producer"
    )
    logger.info(
        "[musespark_open] multi_request_post_attention_hop: fused post-attention hop (AR + residual "
        f"join + pre-FFN norm) on the decode path of all {layers} layers: {transports} "
        f"({'multicast' if multicast else 'unicast'} publication, {launch}, early "
        "dependent-launch trigger, stale-only polling)"
    )
