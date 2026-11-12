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

Row split (``cluster``). A row runs on a cluster of 4 CTAs of 256 threads (the default),
2 of 512 or 1 of 1024. Thread ``t`` of the row (``t = cta_in_cluster * threads + tid``)
owns column vector ``t`` in every split, and both variance reductions add the row's 32
warp sums in warp order ``t // 32`` (over DSMEM between the cluster's CTAs, within the
CTA when the cluster is one CTA), so the three splits compute the same bits; they differ
in the CTAs per row (grid ``rows * cluster``). ``cluster_rows`` limits a workspace's split
to calls of those row counts; every other call of the workspace runs the 4-CTA split.
Every rank shares a call's row count, so every rank runs the same split for it, and the
transport is the same for every split (thread ``t`` of a row writes and reads the slots of
column vector ``t``, and every call adds ``GRID_STEP`` to the arrival counter), so calls
of either split may follow each other on one workspace.

Transport (Lamport). Every slot of the receive memory starts as the sentinel word
``0x80000000``; a 32-bit word that still holds it has not arrived. No payload word is
the sentinel: a partial has BF16 ``-0`` cleared, and the BF16 rounding of an FP32 sum
from ``+0`` of such values is never ``-0`` either (the exact sum is a multiple of
BF16's smallest subnormal, so is its FP32 rounding, and a non-zero one rounds to a
non-zero BF16). The first read that finds no sentinel word returns the final value.
Polling is stale-only: after one read of every awaited slot only the slots still
holding a sentinel word are read again, all in flight together. A reader writes the
sentinel back into each slot it consumed. A call takes one of two transports, chosen by
its row count against the workspace's cut (served: ``TWO_SHOT_ROWS_BY_RANKS``, 32 rows at
four ranks and 16 at eight):

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

Workspace per rank: the one-shot area ``2 * ranks * (two_shot_rows - 1) * 16 KiB``
(``two_shot_rows`` the workspace's cut: 3.875 MiB on the served four-rank workspace, 3.75
MiB on the eight-rank one) plus the two-shot area ``2 * 2 * max_rows * 16 KiB`` (scatter
and gather), its own fabric allocation and counter, never shared with the 8-row hop, the
built-in MNNVL all-reduce or the expert finalize. Calls on one workspace must be issued in
the same order on every rank and serialized on one stream (graph replay included).

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
from typing import FrozenSet, Optional, Sequence, Tuple

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
#: Fewest rows a call takes the two-shot transport for (one-shot below), by the group's rank
#: count, on GB300. At four ranks the one-shot is the faster transport up to 24 rows, the
#: two-shot from 32. At eight ranks over two trays' fabric, where a one-shot call moves
#: seven peers' rows instead of three, the two-shot is the faster from 16 rows. The served
#: workspace takes its group's entry (``two_shot_rows_for``); both transports give the same
#: bits, so the cut moves time only.
TWO_SHOT_ROWS_BY_RANKS = {4: 32, 8: 16}
#: The four-rank cut: ``log_once_enabled``'s default, and what callers that predate the
#: per-rank table read.
TWO_SHOT_ROWS = TWO_SHOT_ROWS_BY_RANKS[4]
#: The served workspace's publication: unicast stores to each peer (the two-shot gather
#: then skips the owner's own copy). ``multicast=True`` publishes with one multimem
#: store per published vector instead (the one-shot slab, the two-shot gather): the same values
#: in the same slots, bitwise.
SERVED_MULTICAST = False
#: The served workspace's launch mode: after the o_proj GEMM rather than as its programmatic
#: dependent, so the hop's CTAs do not take SM slots from the GEMM's last wave.
#: ``early_launch=True`` launches the hop as o_proj's programmatic dependent (at TEP8
#: o_proj's split-K is one wave of 128 CTAs and triggers after its A loads, so the hop's
#: ``4 * rows`` CTAs become resident on the free SMs and wait in griddepcontrol.wait). Launch
#: mode only: bitwise. layers.py passes both settings to ``shared_wide_workspace``.
SERVED_EARLY_LAUNCH = False
#: Two rows per CTA on two-shot calls of at least ``SERVED_R2_MIN_ROWS`` (even) rows on 8-rank
#: groups (``r2=True``). CTA ``b`` takes rows ``2 (b // CLUSTER)`` and
#: ``2 (b // CLUSTER) + 1`` and the ``b % CLUSTER``-th quarter of both; every thread keeps its
#: one 16-byte column vector, now of both rows, so each row's sum, cluster reduction slots and
#: order, and boundary are the one-row kernel's statement for statement (bitwise). Halves the
#: grid, so an early-launched hop packs half as many CTAs onto the SMs o_proj leaves free.
#: ``r2_wide=True`` (``SERVED_R2_WIDE``) instead runs two rows per 512-thread CTA (each 256-thread half
#: is the one-row kernel's CTA of its row, so each thread's work is unchanged; bitwise likewise).
SERVED_R2 = False
SERVED_R2_WIDE = False
SERVED_R2_MIN_ROWS = 33
#: Extra dynamic shared memory per CTA on 8-rank groups (bytes, unused by the kernel), to cap
#: how many CTAs of an early-launched hop become resident per free SM (launch placement only:
#: bitwise).
SERVED_SMEM_PAD = 0
#: The row counts ``(lo, hi)`` whose calls take the pad (default every row count); e.g.
#: ``(9, 40)`` pads the one-row grid only where its clusters fit one CTA per SM.
SERVED_PAD_ROWS = (0, 1024)


ROW_BYTES = HIDDEN * 2
#: The cluster boundary's split for 9..32 rows; this entry uses it at every row count.
CLUSTER = 4
THREADS = HIDDEN // (CLUSTER * VECTOR)
#: The row's CTA splits a workspace may take: a cluster of 4, 2 or 1 CTAs of 256, 512 or 1024
#: threads. Every split keeps the thread -> column-vector map and the 32 warp slots of the
#: variance reductions in the same order, so the three give the same bits.
CLUSTERS = (4, 2, 1)
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


def two_shot_rows_for(ranks: int) -> int:
    """The served cut of a group of ``ranks`` ranks (``TWO_SHOT_ROWS_BY_RANKS``).

    Raises:
        ValueError: for a rank count the wide hop does not serve (4 or 8), with the
            workspace's own message.
    """
    cut = TWO_SHOT_ROWS_BY_RANKS.get(int(ranks))
    if cut is None:
        raise ValueError(f"the wide hop serves 4 or 8 ranks, got {ranks}")
    return cut


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

        CTA ``b`` handles row ``b // cluster`` and the ``b % cluster``-th slice of the
        8192 columns; each of its ``8192 / (8 * cluster)`` threads owns one 16-byte column
        vector -- the cluster boundary's 9..32-row map at ``cluster = 4``, which fixes the
        variance reduction order (the same order at every split).
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
            smem_pad: int = 0,
            cluster: int = CLUSTER,
        ):
            if cluster not in CLUSTERS:
                raise ValueError(f"cluster must be one of {CLUSTERS}, got {cluster}")
            self.ranks = ranks
            self.smem_pad = int(smem_pad)
            self.max_rows = max_rows
            self.grid_shift = grid_shift(max_rows)
            self.grid_step = 1 << self.grid_shift
            self.cluster = cluster
            self.threads = HIDDEN // (cluster * VECTOR)
            self.iters = HIDDEN // (cluster * self.threads * VECTOR)
            #: Resident CTAs per SM the registers are bounded for: a 1024-thread CTA needs at
            #: most 64 registers per thread to launch at all.
            self.min_blocks = {4: 0, 2: 2, 1: 1}[cluster]
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
            launcher = self.kernel(
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
            )
            if cutlass.const_expr(self.min_blocks > 0):
                launcher.launch(
                    grid=[rows * self.cluster, 1, 1],
                    block=[self.threads, 1, 1],
                    cluster=[self.cluster, 1, 1],
                    smem=self.slots * 4 * 4 * 2 + 16 + self.smem_pad,
                    stream=stream,
                    use_pdl=self.pdl and self.early_launch,
                    min_blocks_per_mp=self.min_blocks,
                )
            else:
                launcher.launch(
                    grid=[rows * self.cluster, 1, 1],
                    block=[self.threads, 1, 1],
                    cluster=[self.cluster, 1, 1],
                    smem=self.slots * 4 * 4 * 2 + 16 + self.smem_pad,
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

    class _WideHop2Kernel(_WideHopKernel):
        """The two-shot ``_WideHopKernel`` with two rows per CTA (``r2=True``).

        CTA ``b`` handles rows ``2 (b // CLUSTER)`` and ``2 (b // CLUSTER) + 1`` and the
        ``b % CLUSTER``-th quarter of the 8192 columns of both; each of its 256 threads owns the
        same 16-byte column vector of both rows. Per row, every statement is the one-row kernel's:
        the sanitized publication into the owner's scatter slot, the owner's stale-only polling,
        rank-order FP32 sum and RN round, the gather publication and poll, the two variance
        reductions (in-thread, warp butterfly, the cluster's 32 slots ``crank * warps + warp``
        summed by lane) and the boundary. So both rows' bits equal the one-row kernel's.

        Progress: a thread first publishes every row of its pair that another rank owns, then
        does the owner's work for the row this rank owns (at most one of two consecutive rows),
        then waits on the gather slots. The owner's polls need only the publications of the
        same-index CTA of the peers, which precede any wait there; the gather polls need only
        the owners' work. So no wait depends on another wait, as in the one-row kernel.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.two_shot:
                raise ValueError("the two-row hop serves the two-shot transport only")

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
                grid=[(rows // 2) * self.cluster, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 4 + 32 + self.smem_pad,
                stream=stream,
                use_pdl=self.pdl and self.early_launch,
            )

        @cute.jit
        def owner_sum(self, scatter_address, source_bytes, own, reduced: cute.Tensor, rank: Int32):
            # The one-row kernel's owner branch: stale-only polling of the peers' scatter slots of
            # this row (this rank's own value is ``own``), resets, the rank-order FP32 sum from +0
            # and one RN round, packed into ``reduced``.
            values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
            for p in cutlass.range_constexpr(self.ranks):
                v0, v1, v2, v3 = _poll_x4(
                    scatter_address + p * source_bytes,
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
                    values[p, 0], values[p, 1], values[p, 2], values[p, 3], Uint32(1 << p)
                )
            while stale != Uint32(0):
                polled = stale
                for p in cutlass.range_constexpr(self.ranks):
                    v0, v1, v2, v3 = _poll_x4(
                        scatter_address + p * source_bytes,
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
                        values[p, 0], values[p, 1], values[p, 2], values[p, 3], Uint32(1 << p)
                    )
            for p in cutlass.range_constexpr(self.ranks):
                _reset_x4(scatter_address + p * source_bytes, Uint32(rank != p))
            sums = [Float32(0.0)] * VECTOR
            for p in cutlass.range_constexpr(self.ranks):
                for k in cutlass.range_constexpr(4):
                    lo, hi = _unpack_bf16x2(values[p, k])
                    sums[2 * k] = sums[2 * k] + lo
                    sums[2 * k + 1] = sums[2 * k + 1] + hi
            for k in cutlass.range_constexpr(4):
                reduced[k] = _pack_bf16x2(sums[2 * k], sums[2 * k + 1])

        @cute.jit
        def gather_first(self, address, own, reduced: cute.Tensor):
            # The non-owner's first read of the row's gather slot (the one-row kernel's).
            g0, g1, g2, g3 = _poll_x4(address, Uint32(1), own[0], own[1], own[2], own[3])
            reduced[0] = g0
            reduced[1] = g1
            reduced[2] = g2
            reduced[3] = g3

        @cute.jit
        def gather_wait(self, address, reduced: cute.Tensor):
            # Re-read the gather slot until the owner's sum has arrived, then reset it.
            waiting = _stale_bit(reduced[0], reduced[1], reduced[2], reduced[3], Uint32(1))
            while waiting != Uint32(0):
                h0, h1, h2, h3 = _poll_x4(
                    address, Uint32(1), reduced[0], reduced[1], reduced[2], reduced[3]
                )
                reduced[0] = h0
                reduced[1] = h1
                reduced[2] = h2
                reduced[3] = h3
                waiting = _stale_bit(reduced[0], reduced[1], reduced[2], reduced[3], Uint32(1))
            _reset_x4(address, Uint32(1))

        @cute.jit
        def reduce2(
            self,
            value0: Float32,
            value1: Float32,
            partials: cute.Tensor,
            bars: cute.Tensor,
            phase: cutlass.Constexpr,
        ):
            # ``reduce`` for both rows, the cluster stores of both issued before either wait;
            # row j of phase k uses slot area and mbarrier 2 k + j.
            tid = cute.arch.thread_idx()[0]
            lane, warp = tid % 32, tid // 32
            rank = cute.arch.block_idx()[0] % self.cluster
            value0 = cute.arch.warp_reduction_sum(value0)
            value1 = cute.arch.warp_reduction_sum(value1)
            if lane < self.cluster:
                slot0 = (
                    partials.iterator + ((2 * phase) * self.slots + rank * self.warps + warp) * 4
                )
                cute.arch.store_async_dsmem(
                    slot0.align(16),
                    (_float_bits(value0), Int32(0), Int32(0), Int32(0)),
                    bars.iterator + 2 * phase,
                    lane,
                )
                slot1 = (
                    partials.iterator
                    + ((2 * phase + 1) * self.slots + rank * self.warps + warp) * 4
                )
                cute.arch.store_async_dsmem(
                    slot1.align(16),
                    (_float_bits(value1), Int32(0), Int32(0), Int32(0)),
                    bars.iterator + 2 * phase + 1,
                    lane,
                )
            cute.arch.mbarrier_wait(bars.iterator + 2 * phase, 0)
            total0 = Float32(0)
            for slot in range(lane, self.slots, 32):
                total0 += partials[((2 * phase) * self.slots + slot) * 4]
            cute.arch.mbarrier_wait(bars.iterator + 2 * phase + 1, 0)
            total1 = Float32(0)
            for slot in range(lane, self.slots, 32):
                total1 += partials[((2 * phase + 1) * self.slots + slot) * 4]
            return cute.arch.warp_reduction_sum(total0), cute.arch.warp_reduction_sum(total1)

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
            pair = cute.arch.block_idx()[0] // self.cluster
            offset = cute.assume((crank * self.threads + tid) * 8, divby=8)
            smem = utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((self.slots * 4 * 4,)), byte_alignment=16
            )
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((4,)), byte_alignment=8)
            # A dry launch only makes the driver load the module.
            if dry == 0:
                control_address = control.toint()
                if tid < 4:
                    cute.arch.mbarrier_init(bars.iterator + tid, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + tid, self.slots * 16)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
                # Nothing above touches global memory (as in the one-row kernel).
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                counter = _load_counter(control_address + 4)
                buffer = Int64((counter >> self.grid_shift) & Uint32(1))
                a = self.load(alpha + offset, Float32)
                b = self.load(beta + offset, Float32)
                w = self.load(weight + offset, cutlass.BFloat16)
                source_bytes = self.scatter_rows * ROW_BYTES
                receive_address = receive.toint()

                # Both rows' operands, before any transport access (explicit per row: the DSL's
                # dynamic branches below assign nothing, they only call helpers and store).
                row0 = pair * 2
                row1 = pair * 2 + 1
                slot0 = (Int64(row0) * HIDDEN + Int64(offset)) * 2
                slot1 = (Int64(row1) * HIDDEN + Int64(offset)) * 2
                words0 = _load_x4(partial.toint() + slot0)
                words1 = _load_x4(partial.toint() + slot1)
                r0 = self.load(residual + row0 * HIDDEN + offset, Float32)
                r1 = self.load(residual + row1 * HIDDEN + offset, Float32)
                # This rank's partials with BF16 -0 cleared: no word is the sentinel.
                own0 = (
                    _sanitize_bf16x2(words0[0]),
                    _sanitize_bf16x2(words0[1]),
                    _sanitize_bf16x2(words0[2]),
                    _sanitize_bf16x2(words0[3]),
                )
                own1 = (
                    _sanitize_bf16x2(words1[0]),
                    _sanitize_bf16x2(words1[1]),
                    _sanitize_bf16x2(words1[2]),
                    _sanitize_bf16x2(words1[3]),
                )
                owner0 = row0 % self.ranks
                owner1 = row1 % self.ranks
                scatter0 = (
                    Int64(self.scatter_base)
                    + buffer * (self.ranks * source_bytes)
                    + Int64(row0 // self.ranks) * ROW_BYTES
                    + Int64(offset) * 2
                )
                scatter1 = (
                    Int64(self.scatter_base)
                    + buffer * (self.ranks * source_bytes)
                    + Int64(row1 // self.ranks) * ROW_BYTES
                    + Int64(offset) * 2
                )
                gather0 = Int64(self.gather_base) + buffer * (self.max_rows * ROW_BYTES) + slot0
                gather1 = Int64(self.gather_base) + buffer * (self.max_rows * ROW_BYTES) + slot1
                reduced0 = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                reduced1 = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)

                # 1. Publish every row of the pair another rank owns into its owner's scatter slab.
                if owner0 != rank:
                    for p in cutlass.range_constexpr(self.ranks):
                        if owner0 == p:
                            _store_peer_x4(
                                peers[p].toint() + scatter0 + Int64(rank) * source_bytes,
                                own0[0],
                                own0[1],
                                own0[2],
                                own0[3],
                            )
                if owner1 != rank:
                    for p in cutlass.range_constexpr(self.ranks):
                        if owner1 == p:
                            _store_peer_x4(
                                peers[p].toint() + scatter1 + Int64(rank) * source_bytes,
                                own1[0],
                                own1[1],
                                own1[2],
                                own1[3],
                            )

                # 2. The owner's work for the row of the pair this rank owns (at most one).
                if owner0 == rank:
                    self.owner_sum(receive_address + scatter0, source_bytes, own0, reduced0, rank)
                    if cutlass.const_expr(self.multicast):
                        _store_multicast_x4(
                            mcast.toint() + gather0,
                            reduced0[0],
                            reduced0[1],
                            reduced0[2],
                            reduced0[3],
                        )
                    else:
                        for p in cutlass.range_constexpr(self.ranks):
                            if rank != p:
                                _store_peer_x4(
                                    peers[p].toint() + gather0,
                                    reduced0[0],
                                    reduced0[1],
                                    reduced0[2],
                                    reduced0[3],
                                )
                if owner1 == rank:
                    self.owner_sum(receive_address + scatter1, source_bytes, own1, reduced1, rank)
                    if cutlass.const_expr(self.multicast):
                        _store_multicast_x4(
                            mcast.toint() + gather1,
                            reduced1[0],
                            reduced1[1],
                            reduced1[2],
                            reduced1[3],
                        )
                    else:
                        for p in cutlass.range_constexpr(self.ranks):
                            if rank != p:
                                _store_peer_x4(
                                    peers[p].toint() + gather1,
                                    reduced1[0],
                                    reduced1[1],
                                    reduced1[2],
                                    reduced1[3],
                                )

                # 3. The owners' sums of the other rows: both first polls in flight, then each wait.
                if owner0 != rank:
                    self.gather_first(receive_address + gather0, own0, reduced0)
                if owner1 != rank:
                    self.gather_first(receive_address + gather1, own1, reduced1)
                if owner0 != rank:
                    self.gather_wait(receive_address + gather0, reduced0)
                if owner1 != rank:
                    self.gather_wait(receive_address + gather1, reduced1)
                # Past the last transport access (see the one-row kernel).
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_launch_dependents()

                xs = []
                reduced_of = (reduced0, reduced1)
                r_of = (r0, r1)
                row_of = (row0, row1)
                for j in cutlass.range_constexpr(2):
                    x = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                    reduced = reduced_of[j]
                    for k in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(reduced[k])
                        x[2 * k, 0] = cutlass.BFloat16(low)
                        x[2 * k + 1, 0] = cutlass.BFloat16(high)
                    xs.append(x)

                # From here on: the cluster boundary with gain off, statement for statement, per row.
                scaled_of = []
                updated_of = []
                normed_of = []
                totals = []
                for j in cutlass.range_constexpr(2):
                    scaled = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                    updated = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                    normed = cute.make_rmem_tensor(
                        cute.make_layout((8, self.iters)), cutlass.BFloat16
                    )
                    x = xs[j]
                    total = Float32(0)
                    for p in cutlass.range_constexpr(self.iters):
                        for v in cutlass.range_constexpr(8):
                            value = Float32(x[v, p])
                            scaled[v, p] = value
                            total += value * value
                    scaled_of.append(scaled)
                    updated_of.append(updated)
                    normed_of.append(normed)
                    totals.append(total)
                total0, total1 = self.reduce2(totals[0], totals[1], partials, bars, 0)
                invs = [
                    cute.math.rsqrt(total0 / Float32(8192) + Float32(self.post_eps), fastmath=True),
                    cute.math.rsqrt(total1 / Float32(8192) + Float32(self.post_eps), fastmath=True),
                ]
                totals = []
                for j in cutlass.range_constexpr(2):
                    scaled = scaled_of[j]
                    updated = updated_of[j]
                    r = r_of[j]
                    inv = invs[j]
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
                    totals.append(total)
                total0, total1 = self.reduce2(totals[0], totals[1], partials, bars, 1)
                invs = [
                    cute.math.rsqrt(total0 / Float32(8192) + Float32(self.pre_eps), fastmath=True),
                    cute.math.rsqrt(total1 / Float32(8192) + Float32(self.pre_eps), fastmath=True),
                ]
                layout = cute.make_layout(
                    (8, self.iters), stride=(1, self.threads * self.cluster * 8)
                )
                atom32 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                )
                atom16 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
                )
                for j in cutlass.range_constexpr(2):
                    scaled = scaled_of[j]
                    normed = normed_of[j]
                    inv = invs[j]
                    for p in cutlass.range_constexpr(self.iters):
                        for v in cutlass.range_constexpr(8):
                            normed[v, p] = cutlass.BFloat16(
                                scaled[v, p] * inv * (Float32(w[v, p]) + Float32(1))
                            )
                    row = row_of[j]
                    cute.copy(
                        atom32,
                        updated_of[j],
                        cute.make_tensor((output + row * HIDDEN + offset).align(16), layout),
                    )
                    cute.copy(
                        atom16,
                        normed,
                        cute.make_tensor((next_output + row * HIDDEN + offset).align(16), layout),
                    )
                # Both reductions waited on this CTA's own mbarriers for every byte its peers
                # send; this barrier only orders every thread's counter read before the arrival.
                cute.arch.barrier()
                if tid == 0:
                    block = cute.arch.block_idx()[0]
                    blocks = cute.arch.grid_dim()[0]
                    amount = Uint32(1)
                    if block == 0:
                        amount = Uint32(self.grid_step + 1 - blocks)
                    _arrive(control_address + 4, amount)

    class _WideHop2wKernel(_WideHopKernel):
        """The two-shot ``_WideHopKernel`` with two rows per 512-thread CTA (``r2="w"``).

        Threads ``256 h .. 256 h + 255`` of CTA ``b`` are the one-row kernel's CTA for row
        ``2 (b // CLUSTER) + h``: the same column vector per thread, the same transport, and the same
        cluster reduction (row ``h``'s 32 slots and mbarrier of each phase are its own). So each
        row's bits are the one-row kernel's. Half the CTAs and clusters of the one-row grid, with no
        extra work per thread (``_WideHop2Kernel`` doubles each thread's work instead).
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.two_shot:
                raise ValueError("the two-row hop serves the two-shot transport only")

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
                grid=[(rows // 2) * self.cluster, 1, 1],
                block=[2 * self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 4 + 32 + self.smem_pad,
                stream=stream,
                use_pdl=self.pdl and self.early_launch,
            )

        @cute.jit
        def reduce_w(
            self,
            value: Float32,
            partials: cute.Tensor,
            bars: cute.Tensor,
            phase: cutlass.Constexpr,
            half,
            t,
        ):
            # ``reduce`` for row ``half`` of the CTA: its own slot area and mbarrier 2 phase + half.
            lane, warp = t % 32, t // 32
            rank = cute.arch.block_idx()[0] % self.cluster
            area = 2 * phase + half
            value = cute.arch.warp_reduction_sum(value)
            if lane < self.cluster:
                slot = partials.iterator + (area * self.slots + rank * self.warps + warp) * 4
                cute.arch.store_async_dsmem(
                    slot.align(16),
                    (_float_bits(value), Int32(0), Int32(0), Int32(0)),
                    bars.iterator + area,
                    lane,
                )
            cute.arch.mbarrier_wait(bars.iterator + area, 0)
            total = Float32(0)
            for slot in range(lane, self.slots, 32):
                total += partials[(area * self.slots + slot) * 4]
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
            half = tid // self.threads
            t = tid % self.threads
            crank = cute.arch.block_idx()[0] % self.cluster
            row = (cute.arch.block_idx()[0] // self.cluster) * 2 + half
            offset = cute.assume((crank * self.threads + t) * 8, divby=8)
            smem = utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((self.slots * 4 * 4,)), byte_alignment=16
            )
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((4,)), byte_alignment=8)
            # A dry launch only makes the driver load the module.
            if dry == 0:
                control_address = control.toint()
                if tid < 4:
                    cute.arch.mbarrier_init(bars.iterator + tid, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + tid, self.slots * 16)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
                # Nothing above touches global memory (as in the one-row kernel).
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

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
                # Two-shot, the one-row kernel's branch statement for statement.
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
                    waiting = _stale_bit(reduced[0], reduced[1], reduced[2], reduced[3], Uint32(1))
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
                        _reset_x4(receive_address + scatter + p * source_bytes, Uint32(rank != p))
                    sums = [Float32(0.0)] * VECTOR
                    for p in cutlass.range_constexpr(self.ranks):
                        for k in cutlass.range_constexpr(4):
                            lo, hi = _unpack_bf16x2(values[p, k])
                            sums[2 * k] = sums[2 * k] + lo
                            sums[2 * k + 1] = sums[2 * k + 1] + hi
                    for k in cutlass.range_constexpr(4):
                        reduced[k] = _pack_bf16x2(sums[2 * k], sums[2 * k + 1])
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
                # Past the last transport access (see the one-row kernel).
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
                total = self.reduce_w(total, partials, bars, 0, half, t)
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
                total = self.reduce_w(total, partials, bars, 1, half, t)
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
        rows_per_cta: int = 1,
        smem_pad: int = 0,
        cluster: int = CLUSTER,
    ):
        # rows_per_cta: 1 one row per CTA (_WideHopKernel); 2 two rows per thread (_WideHop2Kernel);
        # 3 two rows per 512-thread CTA (_WideHop2wKernel, the key's third value).
        cls = {1: _WideHopKernel, 2: _WideHop2Kernel, 3: _WideHop2wKernel}[rows_per_cta]
        kernel = cls(
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
            smem_pad,
            cluster,
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
    the group's ``two_shot_rows_for`` cut, ``SERVED_EARLY_LAUNCH``) on the served
    capacity ``MAX_ROWS``;
    ``max_rows=WIDE_MAX_ROWS`` builds the larger transport. A setting left None (the
    publication, the launch mode, the two-row grid and the SMEM pad) takes the module's
    ``SERVED_*`` setting at construction. ``cluster`` is the row split of the
    one-row-per-CTA calls of ``cluster_rows`` rows (every such call when None); the others run
    ``CLUSTER``.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        multicast: Optional[bool] = None,
        variants: Sequence[Tuple[float, float, bool]] = (),
        two_shot_rows: Optional[int] = None,
        early_launch: Optional[bool] = None,
        max_rows: int = MAX_ROWS,
        r2: Optional[bool] = None,
        r2_min_rows: Optional[int] = None,
        smem_pad: Optional[int] = None,
        r2_wide: Optional[bool] = None,
        pad_rows: Optional[Tuple[int, int]] = None,
        cluster: int = CLUSTER,
        cluster_rows: Optional[FrozenSet[int]] = None,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        multicast = SERVED_MULTICAST if multicast is None else multicast
        early_launch = SERVED_EARLY_LAUNCH if early_launch is None else early_launch
        r2 = SERVED_R2 if r2 is None else r2
        r2_min_rows = SERVED_R2_MIN_ROWS if r2_min_rows is None else r2_min_rows
        smem_pad = SERVED_SMEM_PAD if smem_pad is None else smem_pad
        r2_wide = SERVED_R2_WIDE if r2_wide is None else r2_wide
        pad_rows = SERVED_PAD_ROWS if pad_rows is None else pad_rows
        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.multicast = bool(multicast)
        self.pdl = _LAUNCH_PDL
        self.early_launch = bool(early_launch)
        # Two rows per CTA on two-shot calls of >= r2_min_rows even rows, and the SMEM pad: both
        # serve 8-rank groups only; a 4-rank group launches one row per CTA, unpadded.
        tep8 = self.ranks == _hop.MAX_RANKS
        self.r2 = bool(r2) and tep8
        self.r2_wide = bool(r2_wide) and tep8
        self.r2_min_rows = max(int(r2_min_rows), TWO_SHOT_ROWS)
        self.smem_pad = int(smem_pad) if tep8 else 0
        self.pad_rows = (int(pad_rows[0]), int(pad_rows[1]))
        if self.ranks not in (4, _hop.MAX_RANKS):
            raise ValueError(f"the wide hop serves 4 or 8 ranks, got {self.ranks}")
        self.two_shot_rows = int(
            two_shot_rows_for(self.ranks) if two_shot_rows is None else two_shot_rows
        )
        self.max_rows = int(max_rows)
        if int(cluster) not in CLUSTERS:
            raise ValueError(f"cluster must be one of {CLUSTERS}, got {cluster}")
        self.cluster = int(cluster)
        # The row counts ``cluster`` applies to (None: every call); the others run ``CLUSTER``.
        self.cluster_rows = (
            None if cluster_rows is None else frozenset(int(r) for r in cluster_rows)
        )
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
            if torch.cuda.get_device_capability(self.device) != (10, 3):
                raise RuntimeError("the wide hop runs on SM103 only")
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
            low = MIN_ROWS if self.max_rows == MAX_ROWS else MAX_ROWS + 1
            kinds = sorted(
                {
                    (*self.variant(rows), self.cluster_for(rows))
                    for rows in range(low, self.max_rows + 1)
                }
            )
            for post_eps, pre_eps, norm_fp32 in variants:
                for two_shot, rows_per_cta, pad, cluster in kinds:
                    self._kernel(post_eps, pre_eps, norm_fp32, two_shot, rows_per_cta, pad, cluster)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "wide post-attention hop kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def clusters(self) -> Tuple[int, ...]:
        """The row splits this workspace's calls run: ``cluster``, and ``CLUSTER`` for the
        row counts outside ``cluster_rows``."""
        if self.cluster_rows is None or self.cluster == CLUSTER:
            return (self.cluster,)
        return (self.cluster, CLUSTER)

    def cluster_for(self, rows: int) -> int:
        """The row split of a call of ``rows`` rows (the same on every rank); a two-row call
        (``takes_r2``) keeps ``CLUSTER``."""
        if self.takes_r2(rows):
            return CLUSTER
        if self.cluster_rows is None or rows in self.cluster_rows:
            return self.cluster
        return CLUSTER

    def _kernel(
        self,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        two_shot: bool,
        rows_per_cta: int = 1,
        pad: int = 0,
        cluster: Optional[int] = None,
    ):
        cluster = self.cluster if cluster is None else int(cluster)
        if int(rows_per_cta) != 1 and cluster != CLUSTER:
            raise ValueError(f"a two-row wide hop runs the {CLUSTER}-CTA split, got {cluster}")
        key = (
            float(post_eps),
            float(pre_eps),
            bool(norm_fp32),
            bool(two_shot),
            int(rows_per_cta),
            int(pad),
        )
        if cluster != CLUSTER:
            # The default split keeps the key of a workspace without a split.
            key += (cluster,)
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
            int(rows_per_cta),
            int(pad),
            cluster,
        )
        # A dry launch loads the module now, not inside a capture (MIN_ROWS: the grid of a dry launch
        # is never run past its dry check, whatever the kernel's rows per CTA).
        dummy = self.control.data_ptr()
        compiled(*([dummy] * 7), *self._fixed, MIN_ROWS, self.rank, 1, self._stream())
        self._kernels[key] = compiled
        return compiled

    def _stream(self):
        return cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def takes_r2(self, rows: int) -> bool:
        """Whether a call of ``rows`` rows runs two rows per CTA (the same on every rank)."""
        return self.r2 and rows >= self.two_shot_rows and rows >= self.r2_min_rows and rows % 2 == 0

    def variant(self, rows: int) -> Tuple[bool, int, int]:
        """``(two_shot, rows_per_cta, pad)`` of a call of ``rows`` rows (row count only, so the
        same on every rank). ``rows_per_cta`` 1 is one row per CTA (``_WideHopKernel``), 2 two
        rows per thread, 3 two rows per 512-thread CTA; ``pad`` the dynamic SMEM pad in bytes."""
        two_shot = rows >= self.two_shot_rows
        kind = (3 if self.r2_wide else 2) if self.takes_r2(rows) else 1
        pad = self.smem_pad if self.pad_rows[0] <= rows <= self.pad_rows[1] else 0
        return two_shot, kind, pad

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
        compiled = self._kernel(
            post_eps, pre_eps, norm_fp32, *self.variant(rows), self.cluster_for(rows)
        )
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
    multicast: Optional[bool] = None,
    variants: Sequence[Tuple[float, float, bool]] = (),
    early_launch: Optional[bool] = None,
    max_rows: int = MAX_ROWS,
    cluster: int = CLUSTER,
    cluster_rows: Optional[FrozenSet[int]] = None,
) -> WideHopWorkspace:
    """The one wide-hop workspace of this group, publication, launch mode, capacity and row
    split, built on first use. The served capacity (``MAX_ROWS``) splits its calls at the
    group's cut (``two_shot_rows_for(mapping.tp_size)``: 32 rows at four ranks, 16 at eight).
    The larger capacity (``WIDE_MAX_ROWS``) takes only calls of more than ``MAX_ROWS`` rows,
    so it is two-shot only. ``multicast`` and ``early_launch``
    left None take the module's settings (``SERVED_MULTICAST``, ``SERVED_EARLY_LAUNCH``).

    Collective on first use.
    """
    multicast = SERVED_MULTICAST if multicast is None else multicast
    early_launch = SERVED_EARLY_LAUNCH if early_launch is None else early_launch
    two_shot_rows = two_shot_rows_for(mapping.tp_size) if max_rows == MAX_ROWS else MIN_ROWS
    key = (mapping, bool(multicast), _LAUNCH_PDL, two_shot_rows, bool(early_launch), max_rows)
    if int(cluster) != CLUSTER:
        key += (int(cluster), None if cluster_rows is None else frozenset(cluster_rows))
    workspace = _SHARED.get(key)
    if workspace is None:
        workspace = WideHopWorkspace(
            mapping,
            multicast=multicast,
            variants=variants,
            two_shot_rows=two_shot_rows,
            early_launch=early_launch,
            max_rows=max_rows,
            cluster=cluster,
            cluster_rows=cluster_rows,
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

    ARCHS = frozenset({Arch.SM_103})

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
    answers alike, and makes the layout fit with `post_attn_hop.readable`. Eight ranks
    (TEP8, two GB300 trays) are certified on the served transport, 16..512 rows: every
    row against the reference, and the one-shot and two-shot transports bitwise against
    each other. Not certified: rows outside 9..1024, widths other than 8192, a weighted
    attention-side post norm, eight ranks on the larger transport (513..1024 rows).
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
    early_launch: Optional[bool] = None,
    min_rows: int = MIN_ROWS,
    r2: Optional[bool] = None,
    r2_wide: Optional[bool] = None,
    smem_pad: Optional[int] = None,
    cluster: int = CLUSTER,
    cluster_rows: Optional[FrozenSet[int]] = None,
    ranks: Optional[int] = None,
) -> None:
    """The install line. It names the workspace's cut as ``two_shot_rows=<cut>`` (and the
    group as ``ranks=<ranks>`` when given), whatever row range the transports cover."""
    # A setting left None reads the module's ``SERVED_*`` setting.
    early_launch = SERVED_EARLY_LAUNCH if early_launch is None else early_launch
    r2 = SERVED_R2 if r2 is None else r2
    r2_wide = SERVED_R2_WIDE if r2_wide is None else r2_wide
    smem_pad = SERVED_SMEM_PAD if smem_pad is None else smem_pad
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
    if cluster != CLUSTER:
        where = "" if cluster_rows is None else f" at {len(cluster_rows)} row counts"
        transports += f"; {cluster} CTA(s) of {HIDDEN // (cluster * VECTOR)} threads per row{where}"
    launch = (
        "launched as a programmatic dependent" if early_launch else "launched after its producer"
    )
    cut = f"two_shot_rows={two_shot_rows}" + (f" ranks={ranks}" if ranks is not None else "")
    logger.info(
        "[musespark_open] multi_request_post_attention_hop: fused post-attention hop (AR + residual "
        f"join + pre-FFN norm) on the decode path of all {layers} layers: {transports} "
        f"({cut}; {'multicast' if multicast else 'unicast'} publication, {launch}, early "
        "dependent-launch trigger, stale-only polling)"
        + (
            "; two rows per CTA "
            f"{('(512 threads)' if r2_wide else '(two per thread)') if r2 else 'off'}"
            f" (two-shot calls of >= {max(SERVED_R2_MIN_ROWS, TWO_SHOT_ROWS)} even rows),"
            f" SMEM pad {smem_pad} B at {SERVED_PAD_ROWS[0]}..{SERVED_PAD_ROWS[1]} rows"
        )
    )
