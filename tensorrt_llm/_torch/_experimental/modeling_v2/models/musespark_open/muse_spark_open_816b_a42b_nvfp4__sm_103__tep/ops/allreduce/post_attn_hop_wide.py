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
from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

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
# ``configure`` sets the settings from here to its definition (``lever_settings.apply`` calls it
# from the model's plan before any install); their defaults keep the paths they refine.
#: The served workspace's publication: unicast stores to each peer (the two-shot gather
#: then skips the owner's own copy). ``configure(multicast=True)`` publishes with one multimem
#: store per published vector instead (the one-shot slab, the two-shot gather): the same values
#: in the same slots, bitwise.
SERVED_MULTICAST = False
#: The served workspace's launch mode: after the o_proj GEMM rather than as its programmatic
#: dependent, so the hop's CTAs do not take SM slots from the GEMM's last wave.
#: ``configure(early_launch=True)`` launches the hop as o_proj's programmatic dependent (at TEP8
#: o_proj's split-K is one wave of 128 CTAs and triggers after its A loads, so the hop's
#: ``4 * rows`` CTAs become resident on the free SMs and wait in griddepcontrol.wait). Launch
#: mode only: bitwise. layers.py passes both settings to ``shared_wide_workspace``.
SERVED_EARLY_LAUNCH = False
#: Two rows per CTA on two-shot calls of at least ``SERVED_R2_MIN_ROWS`` (even) rows on 8-rank
#: groups (``configure(r2="1")``). CTA ``b`` takes rows ``2 (b // CLUSTER)`` and
#: ``2 (b // CLUSTER) + 1`` and the ``b % CLUSTER``-th quarter of both; every thread keeps its
#: one 16-byte column vector, now of both rows, so each row's sum, cluster reduction slots and
#: order, and boundary are the one-row kernel's statement for statement (bitwise). Halves the
#: grid, so an early-launched hop packs half as many CTAs onto the SMs o_proj leaves free.
#: ``r2="w"`` (``SERVED_R2_WIDE``) instead runs two rows per 512-thread CTA (each 256-thread half
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
#: ``tep8_direct_hop``: o_proj of a TEP8 decode step of ``SERVED_DIRECT_ROWS`` rows stores its
#: partial straight into the owners' scatter slots (the wide64 split-K kernel's scatter
#: epilogue, called by the dense GEMM dispatcher) and the joins run the direct consumer
#: (``WideDirectWorkspace``). Built and armed by layers.py.
SERVED_DIRECT = False


def configure(
    *,
    direct: bool = False,
    early_launch: bool = False,
    multicast: bool = False,
    pad_rows: Tuple[int, int] = (0, 1024),
    r2: str = "0",
    r2_min_rows: int = 33,
    smem_pad: int = 0,
) -> None:
    """Set the served wide workspace's publication, launch mode and grid, and the direct hop.

    ``lever_settings.apply`` calls it from the model's plan before any install: each workspace
    and the direct publication take these settings when they are built.

    Args:
        direct: o_proj of the TEP8 decode steps of ``SERVED_DIRECT_ROWS`` rows publishes
            straight into the owners' scatter slots (``SERVED_DIRECT``).
        early_launch: launch the hop as o_proj's programmatic dependent (``SERVED_EARLY_LAUNCH``).
        multicast: publish with one multimem store per vector (``SERVED_MULTICAST``).
        pad_rows: the row counts ``(lo, hi)`` whose calls take the SMEM pad (``SERVED_PAD_ROWS``).
        r2: ``"0"`` one row per CTA, ``"1"`` two rows per CTA (each thread both), ``"w"`` two
            rows per 512-thread CTA (``SERVED_R2``, ``SERVED_R2_WIDE``).
        r2_min_rows: the fewest rows of a two-row call (``SERVED_R2_MIN_ROWS``).
        smem_pad: extra dynamic shared memory per CTA, in bytes (``SERVED_SMEM_PAD``).

    Raises:
        ValueError: for an ``r2`` other than ``"0"``, ``"1"`` or ``"w"``.
    """
    global SERVED_DIRECT, SERVED_EARLY_LAUNCH, SERVED_MULTICAST, SERVED_PAD_ROWS
    global SERVED_R2, SERVED_R2_WIDE, SERVED_R2_MIN_ROWS, SERVED_SMEM_PAD
    if r2 not in ("0", "1", "w"):
        raise ValueError(f"r2 is '0', '1' or 'w', got {r2!r}")
    SERVED_DIRECT = bool(direct)
    SERVED_EARLY_LAUNCH = bool(early_launch)
    SERVED_MULTICAST = bool(multicast)
    SERVED_PAD_ROWS = (int(pad_rows[0]), int(pad_rows[1]))
    SERVED_R2 = r2 in ("1", "w")
    SERVED_R2_WIDE = r2 == "w"
    SERVED_R2_MIN_ROWS = int(r2_min_rows)
    SERVED_SMEM_PAD = int(smem_pad)


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


def cluster_rows_of(ranges: Optional[Iterable[Tuple[int, int]]]) -> Optional[FrozenSet[int]]:
    """The row counts of inclusive ``(lo, hi)`` ranges; None for none (every row count).

    Raises:
        ValueError: for a range outside ``MIN_ROWS..WIDE_MAX_ROWS`` or descending.
    """
    if ranges is None:
        return None
    rows = set()
    for lo, hi in ranges:
        if not MIN_ROWS <= int(lo) <= int(hi) <= WIDE_MAX_ROWS:
            raise ValueError(
                f"row range {lo}-{hi} outside {MIN_ROWS}..{WIDE_MAX_ROWS} or descending"
            )
        rows.update(range(int(lo), int(hi) + 1))
    return frozenset(rows) if rows else None


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
        _direct_reset_x4,
        _direct_stale_bit,
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
        """The two-shot ``_WideHopKernel`` with two rows per CTA (``configure(r2="1")``).

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

    class _WideDirectKernel(_WideHop2wKernel):
        """The two-shot hop's consumer half on a direct transport (``tep8_direct_hop``).

        The o_proj GEMM stores row ``t`` of its BF16 partial straight into the scatter slab of this
        rank on the row's owner ``t % ranks`` (its epilogue, through the group's contiguous unicast
        window), so this kernel neither loads nor publishes the partial. The owner of a row polls the
        row's eight source slots of its local scatter area -- its own included -- with the direct
        transport's sentinel (``0xFFFE`` per BF16 half: no GEMM writes it, so ``-0`` needs no
        clearing), resets them, sums them in rank order from ``+0`` and rounds once (the two-shot
        owner's sum of ``_WideHopKernel``: a ``-0`` addend leaves a sum from ``+0`` unchanged), and
        publishes the BF16 sum into the row's gather slot of every rank (``0x80000000`` sentinel,
        as ``_WideHopKernel``'s gather). Every other rank polls that gather slot. Then the
        boundary, statement for statement. The buffer index is the call site's ``parity`` (the
        producer cannot read a counter). Grid: ``_WideHopKernel``'s 4 CTAs per row, or with
        ``wide`` two rows per 512-thread CTA (``_WideHop2wKernel``'s map).
        ``wait_first``: ``griddepcontrol.wait`` before the polls (the 8-row direct consumer's
        order); otherwise after them, before the residual load (the polls are Lamport, the
        coefficients immutable).
        """

        def __init__(
            self,
            ranks,
            multicast,
            post_eps,
            pre_eps,
            norm_fp32,
            pdl,
            early_launch,
            max_rows,
            wide,
            smem_pad,
            wait_first,
            fb_layout=False,
        ):
            super().__init__(
                ranks,
                multicast,
                post_eps,
                pre_eps,
                norm_fp32,
                pdl,
                True,
                0,
                early_launch,
                max_rows,
                smem_pad,
            )
            # Scatter slots feature-block-major ([64-feature block][max_rows / ranks slots][64])
            # instead of [slot][8192] (``fb_layout``): the producer's per-owner runs are then
            # contiguous.
            self.fb_layout = bool(fb_layout)
            self.wide = bool(wide)
            self.wait_first = bool(wait_first)
            self.src_bytes = (max_rows // ranks) * ROW_BYTES
            self.d_gather_base = BUFFERS * ranks * self.src_bytes

        @cute.jit
        def __call__(
            self,
            residual: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
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
            parity: Int32,
            dry: Int32,
            stream: cuda.CUstream,
        ):
            if cutlass.const_expr(self.wide):
                grid = [(rows // 2) * self.cluster, 1, 1]
                block = [2 * self.threads, 1, 1]
            else:
                grid = [rows * self.cluster, 1, 1]
                block = [self.threads, 1, 1]
            self.kernel(
                residual,
                alpha,
                beta,
                weight,
                output,
                next_output,
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
                parity,
                dry,
            ).launch(
                grid=grid,
                block=block,
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 4 + 32 + self.smem_pad,
                stream=stream,
                use_pdl=self.pdl and self.early_launch,
            )

        @cute.kernel
        def kernel(
            self,
            residual: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
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
            parity: Int32,
            dry: Int32,
        ):
            peers = (peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7)
            tid = cute.arch.thread_idx()[0]
            crank = cute.arch.block_idx()[0] % self.cluster
            if cutlass.const_expr(self.wide):
                half = tid // self.threads
                t = tid % self.threads
                row = (cute.arch.block_idx()[0] // self.cluster) * 2 + half
            else:
                half = 0
                t = tid
                row = cute.arch.block_idx()[0] // self.cluster
            offset = cute.assume((crank * self.threads + t) * 8, divby=8)
            smem = utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((self.slots * 4 * 4,)), byte_alignment=16
            )
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((4,)), byte_alignment=8)
            if dry == 0:
                if tid < 4:
                    cute.arch.mbarrier_init(bars.iterator + tid, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + tid, self.slots * 16)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
                if cutlass.const_expr(self.pdl and self.wait_first):
                    cute.arch.griddepcontrol_wait()
                # Immutable operands (the kernel's whole lifetime): safe before the wait.
                a = self.load(alpha + offset, Float32)
                b = self.load(beta + offset, Float32)
                w = self.load(weight + offset, cutlass.BFloat16)

                buffer = Int64(parity)
                slot = (Int64(row) * HIDDEN + Int64(offset)) * 2
                receive_address = receive.toint()
                owner = row % self.ranks
                if cutlass.const_expr(self.fb_layout):
                    scatter = (
                        buffer * (self.ranks * self.src_bytes)
                        + Int64(offset // 64) * ((self.max_rows // self.ranks) * 128)
                        + Int64(row // self.ranks) * 128
                        + Int64(offset % 64) * 2
                    )
                else:
                    scatter = (
                        buffer * (self.ranks * self.src_bytes)
                        + Int64(row // self.ranks) * ROW_BYTES
                        + Int64(offset) * 2
                    )
                gather = Int64(self.d_gather_base) + buffer * (self.max_rows * ROW_BYTES) + slot
                reduced = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                x = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                if owner == rank:
                    for p in cutlass.range_constexpr(self.ranks):
                        v0, v1, v2, v3 = _poll_x4(
                            receive_address + scatter + p * self.src_bytes,
                            Uint32(1),
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                            Uint32(0),
                        )
                        values[p, 0] = v0
                        values[p, 1] = v1
                        values[p, 2] = v2
                        values[p, 3] = v3
                    stale = Uint32(0)
                    for p in cutlass.range_constexpr(self.ranks):
                        stale = stale | _direct_stale_bit(
                            values[p, 0], values[p, 1], values[p, 2], values[p, 3], Uint32(1 << p)
                        )
                    while stale != Uint32(0):
                        polled = stale
                        for p in cutlass.range_constexpr(self.ranks):
                            v0, v1, v2, v3 = _poll_x4(
                                receive_address + scatter + p * self.src_bytes,
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
                            stale = stale | _direct_stale_bit(
                                values[p, 0],
                                values[p, 1],
                                values[p, 2],
                                values[p, 3],
                                Uint32(1 << p),
                            )
                    for p in cutlass.range_constexpr(self.ranks):
                        _direct_reset_x4(receive_address + scatter + p * self.src_bytes)
                    # The two-shot owner's sum: FP32 from +0, ranks in order, one RN round; a -0
                    # addend (nothing clears it here) leaves a sum from +0 unchanged.
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
                            mcast.toint() + gather, reduced[0], reduced[1], reduced[2], reduced[3]
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
                else:
                    g0, g1, g2, g3 = _poll_x4(
                        receive_address + gather,
                        Uint32(1),
                        Uint32(0),
                        Uint32(0),
                        Uint32(0),
                        Uint32(0),
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
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.pdl and not self.wait_first):
                    cute.arch.griddepcontrol_wait()
                r = self.load(residual + row * HIDDEN + offset, Float32)
                for k in cutlass.range_constexpr(4):
                    low, high = _unpack_bf16x2(reduced[k])
                    x[2 * k, 0] = cutlass.BFloat16(low)
                    x[2 * k + 1, 0] = cutlass.BFloat16(high)

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

    _DIRECT_POINTER_DTYPES = (
        Float32,  # residual
        Float32,  # alpha
        Float32,  # beta
        BFloat16,  # weight
        Float32,  # output
        BFloat16,  # next_output
        Uint32,  # receive
        Uint32,  # mcast
    ) + (Uint32,) * _hop.MAX_RANKS

    @functools.lru_cache(maxsize=32)
    def _compiled_direct(
        device_index: int,
        ranks: int,
        multicast: bool,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        pdl: bool,
        early_launch: bool,
        max_rows: int,
        wide: bool,
        smem_pad: int,
        wait_first: bool,
        fb_layout: bool = False,
    ):
        kernel = _WideDirectKernel(
            ranks,
            multicast,
            post_eps,
            pre_eps,
            norm_fp32,
            pdl,
            early_launch,
            max_rows,
            wide,
            smem_pad,
            wait_first,
            fb_layout,
        )
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *[
                    make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
                    for dtype in _DIRECT_POINTER_DTYPES
                ],
                Int32(MIN_ROWS + 1),
                Int32(0),
                Int32(0),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )

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
    ``TWO_SHOT_ROWS``, ``SERVED_EARLY_LAUNCH``) on the served capacity ``MAX_ROWS``;
    ``max_rows=WIDE_MAX_ROWS`` builds the larger transport. A setting left None (the
    publication, the launch mode, the two-row grid and the SMEM pad) takes the module's
    ``SERVED_*`` setting (``configure``) at construction. ``cluster`` is the row split of the
    one-row-per-CTA calls of ``cluster_rows`` rows (every such call when None); the others run
    ``CLUSTER``.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        multicast: Optional[bool] = None,
        variants: Sequence[Tuple[float, float, bool]] = (),
        two_shot_rows: int = TWO_SHOT_ROWS,
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
        self.two_shot_rows = int(two_shot_rows)
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


class WideDirectWorkspace:
    """The direct transport of the wide two-shot hop for one TP group (``tep8_direct_hop``).

    Receive memory per rank (a fabric allocation of its own, never shared): the scatter area
    ``[2 parities][ranks sources][max_rows / ranks slots][8192]`` BF16 at offset 0, filled with
    the direct sentinel ``0xFFFEFFFE``, then the gather area ``[2][max_rows][8192]`` BF16, filled
    with ``0x80000000``. The producer of call parity ``q`` on rank ``r`` stores token row ``t`` of
    its partial at ``producer_base(q) + (t % ranks) * peer_stride + (t // ranks) * ROW_BYTES``
    (bytes; ``producer_base(q)`` = rank 0's unicast mapping + ``q`` parity area + ``r``'s slab):
    the group's unicast mappings are one contiguous window (checked here).
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        multicast: bool = True,
        variants: Sequence[Tuple[float, float, bool]] = (),
        early_launch: bool = True,
        max_rows: int = 64,
        wide_min_rows: int = 41,
        smem_pad: int = 177408,
        pad_rows: Tuple[int, int] = (9, 40),
        wait_first: bool = True,
        fb_layout: bool = False,
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
        self.max_rows = int(max_rows)
        self.wide_min_rows = int(wide_min_rows)
        self.smem_pad = int(smem_pad)
        self.pad_rows = (int(pad_rows[0]), int(pad_rows[1]))
        self.wait_first = bool(wait_first)
        self.fb_layout = bool(fb_layout)
        if self.ranks not in (4, _hop.MAX_RANKS) or self.max_rows % self.ranks:
            raise ValueError(f"the direct wide hop serves 4 or 8 ranks, got {self.ranks}")
        self.src_bytes = (self.max_rows // self.ranks) * ROW_BYTES
        self.scatter_bytes = BUFFERS * self.ranks * self.src_bytes
        self.workspace_bytes = self.scatter_bytes + BUFFERS * self.max_rows * ROW_BYTES
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._kernels = {}
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("CuTe DSL is unavailable")
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, self.workspace_bytes, mapping, mapping.is_multi_node()
            )
            words = self.workspace_bytes // 4
            self._uc = [
                self._handle.get_uc_buffer(peer, (words,), torch.int32, 0)
                for peer in range(self.ranks)
            ]
            self._mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            base = self._uc[0].data_ptr()
            self.peer_stride = self._uc[1].data_ptr() - base
            if any(
                self._uc[p].data_ptr() - base != p * self.peer_stride for p in range(self.ranks)
            ):
                raise RuntimeError("the group's unicast mappings are not one evenly strided window")
            with torch.cuda.device(self.device):
                local = self._uc[self.rank]
                local[: self.scatter_bytes // 4].fill_(_hop._DIRECT_SENTINEL_I32)
                local[self.scatter_bytes // 4 :].fill_(-(2**31))
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                f"direct wide hop workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        receive = self._uc[self.rank].data_ptr()
        peers = [t.data_ptr() for t in self._uc] + [receive] * (_hop.MAX_RANKS - self.ranks)
        self._fixed = (receive, self._mc.data_ptr(), *peers)
        for post_eps, pre_eps, norm_fp32 in variants:
            for kind in sorted(
                {self.variant(rows) for rows in range(MIN_ROWS + 1, self.max_rows + 1, 2)}
            ):
                self._kernel(post_eps, pre_eps, norm_fp32, *kind)
        torch.cuda.synchronize(self.device)

    def variant(self, rows: int) -> Tuple[bool, int]:
        """``(wide, pad)`` of a call of ``rows`` rows (the row count only: the same on every rank)."""
        wide = rows >= self.wide_min_rows and rows % 2 == 0
        pad = self.smem_pad if self.pad_rows[0] <= rows <= self.pad_rows[1] else 0
        return wide, pad

    def producer_views(self):
        """The producer's output tensors, one per parity: BF16 ``[max_rows, 8192]`` views whose origin is this
        rank's slab on rank 0 of the window (only their address and shape are used by the scatter epilogue)."""
        flat = self._uc[0].view(torch.bfloat16)
        views = []
        for parity in range(BUFFERS):
            off = (self.producer_base(parity) - self._uc[0].data_ptr()) // 2
            views.append(flat[off : off + self.max_rows * HIDDEN].view(self.max_rows, HIDDEN))
        return views

    def scatter_copy(self, branch: torch.Tensor, parity: int) -> None:
        """Store ``branch`` (this rank's partial, BF16 ``[rows, 8192]``) into the owners' slots of ``parity`` as the
        scatter epilogue would (a fallback for a partial that did not come from the scatter producer)."""
        slots = self.max_rows // self.ranks
        for p in range(self.ranks):
            part = branch[p :: self.ranks]
            n = part.shape[0]
            if n == 0:
                continue
            flat = self._uc[p].view(torch.bfloat16)
            off = (parity * self.ranks + self.rank) * self.src_bytes // 2
            slab = flat[off : off + self.src_bytes // 2]
            if self.fb_layout:
                slab.view(HIDDEN // 64, slots, 64)[:, :n, :].copy_(
                    part.view(n, HIDDEN // 64, 64).transpose(0, 1)
                )
            else:
                slab.view(slots, HIDDEN)[:n].copy_(part)

    def reset(self) -> None:
        """Every slot back to its sentinel on every rank (collective)."""
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        torch.cuda.synchronize(self.device)
        local = self._uc[self.rank]
        local[: self.scatter_bytes // 4].fill_(_hop._DIRECT_SENTINEL_I32)
        local[self.scatter_bytes // 4 :].fill_(-(2**31))
        torch.cuda.synchronize(self.device)
        if not _mnnvl_workspace_all_succeeded(self._comm, True):
            raise RuntimeError("direct wide hop reset failed on at least one rank")

    def producer_base(self, parity: int) -> int:
        """Where this rank's producer of a call of ``parity`` writes token row 0 (rank 0's window)."""
        return (
            self._uc[0].data_ptr()
            + parity * self.ranks * self.src_bytes
            + self.rank * self.src_bytes
        )

    def _kernel(self, post_eps, pre_eps, norm_fp32, wide, pad):
        key = (float(post_eps), float(pre_eps), bool(norm_fp32), bool(wide), int(pad))
        compiled = self._kernels.get(key)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"direct wide hop variant {key} was not built before graph capture")
        compiled = _compiled_direct(
            self.device.index,
            self.ranks,
            self.multicast,
            float(post_eps),
            float(pre_eps),
            bool(norm_fp32),
            self.pdl,
            self.early_launch,
            self.max_rows,
            bool(wide),
            int(pad),
            self.wait_first,
            self.fb_layout,
        )
        dummy = self._fixed[0]
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        compiled(*([dummy] * 6), *self._fixed, MIN_ROWS + 1, self.rank, 0, 1, stream)
        self._kernels[key] = compiled
        return compiled

    def launch(self, residual, alpha, beta, next_weight, post_eps, pre_eps, norm_fp32, parity: int):
        rows = residual.shape[0]
        if not MIN_ROWS <= rows <= self.max_rows:
            raise ValueError(f"direct wide hop: {rows} rows on a workspace of {self.max_rows}")
        compiled = self._kernel(post_eps, pre_eps, norm_fp32, *self.variant(rows))
        updated = torch.empty_like(residual)
        normalized = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        compiled(
            residual.data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            next_weight.data_ptr(),
            updated.data_ptr(),
            normalized.data_ptr(),
            *self._fixed,
            rows,
            self.rank,
            int(parity),
            0,
            stream,
        )
        return updated, normalized


#: The row counts ``(lo, hi)`` of the TEP8 decode steps that publish directly when
#: ``SERVED_DIRECT`` is on (``hi`` capped by the workspace's capacity).
SERVED_DIRECT_ROWS = (33, 64)
#: The direct consumer's grid: two rows per 512-thread CTA from this many (even) rows; its SMEM
#: pad (bytes, 0 off).
SERVED_DIRECT_WIDE_MIN_ROWS = 41
SERVED_DIRECT_PAD = 177408


class DirectPublication:
    """One model's direct publication (``tep8_direct_hop``): the workspace, the claimed o_proj
    dispatchers (by identity; layer ``l`` publishes with parity ``l % 2``, as ``DirectPublish``),
    the joins and the armed step."""

    def __init__(self, workspace: "WideDirectWorkspace", dispatchers, joins) -> None:
        if len(dispatchers) != len(joins) or len(joins) % 2:
            raise ValueError(
                "the direct publication needs one o_proj per join and an even layer count"
            )
        self.workspace = workspace
        self.parity = {id(d): i % 2 for i, d in enumerate(dispatchers)}
        self.dispatchers = tuple(dispatchers)
        self.joins = tuple(joins)
        self.views = workspace.producer_views()
        self.rows: Optional[int] = None
        self.poisoned: Optional[str] = None

    def takes(self, rows: int) -> bool:
        """Whether a decode step of ``rows`` rows publishes directly (the row count only: every rank alike)."""
        lo, hi = SERVED_DIRECT_ROWS
        return max(lo, MIN_ROWS) <= rows <= min(hi, self.workspace.max_rows)

    def target(self, dispatcher, rows: int):
        """``(view, peer_stride, fb_layout)`` for a claimed o_proj dispatcher's call of the armed step, else None."""
        if self.rows is None or rows != self.rows:
            return None
        parity = self.parity.get(id(dispatcher))
        if parity is None:
            return None
        return self.views[parity][:rows], self.workspace.peer_stride, self.workspace.fb_layout

    def arm(self, rows: int) -> None:
        if self.poisoned is not None:
            raise RuntimeError(
                "[musespark_open] tep8_direct_hop: the transport was not reset after a failed step "
                f"({self.poisoned})"
            )
        self.rows = int(rows)
        for index, join in enumerate(self.joins):
            join._wide_direct = index % 2

    def disarm(self) -> None:
        self.rows = None
        for join in self.joins:
            join._wide_direct = None

    def recover(self, error: BaseException) -> None:
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except RuntimeError:
            pass
        logger.error(
            f"[musespark_open] tep8_direct_hop: a decode step raised ({error!r}); "
            "resetting the transport"
        )
        try:
            self.workspace.reset()
        except Exception as reset_error:  # noqa: BLE001
            self.poisoned = repr(reset_error)

    def consume(self, parity, branch, residual, alpha, beta, weight, post_eps, pre_eps, norm_fp32):
        if branch.data_ptr() != self.views[parity].data_ptr():
            logger.warning_once(
                "[musespark_open] tep8_direct_hop: an o_proj partial did not come from the scatter "
                "producer; copied into the slots before the consumer",
                key="musespark_open.tep8_direct_hop.copied_partial",
            )
            self.workspace.scatter_copy(branch, parity)
        return self.workspace.launch(
            residual, alpha, beta, weight, post_eps, pre_eps, norm_fp32, parity
        )


#: The model's direct publication (layers.py sets it when it builds one); the dispatcher reads it.
_DIRECT_PUBLICATION: Optional[DirectPublication] = None


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
    split, built on first use. The larger capacity (``WIDE_MAX_ROWS``) takes only calls of
    more than ``MAX_ROWS`` rows, so it is two-shot only. ``multicast`` and ``early_launch``
    left None take the module's settings (``SERVED_MULTICAST``, ``SERVED_EARLY_LAUNCH``).

    Collective on first use.
    """
    multicast = SERVED_MULTICAST if multicast is None else multicast
    early_launch = SERVED_EARLY_LAUNCH if early_launch is None else early_launch
    two_shot_rows = TWO_SHOT_ROWS if max_rows == MAX_ROWS else MIN_ROWS
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
    early_launch: Optional[bool] = None,
    min_rows: int = MIN_ROWS,
    r2: Optional[bool] = None,
    r2_wide: Optional[bool] = None,
    smem_pad: Optional[int] = None,
    cluster: int = CLUSTER,
    cluster_rows: Optional[FrozenSet[int]] = None,
) -> None:
    # A setting left None reads the module's (``configure``).
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
    logger.info(
        "[musespark_open] multi_request_post_attention_hop: fused post-attention hop (AR + residual "
        f"join + pre-FFN norm) on the decode path of all {layers} layers: {transports} "
        f"({'multicast' if multicast else 'unicast'} publication, {launch}, early "
        "dependent-launch trigger, stale-only polling)"
        + (
            "; two rows per CTA "
            f"{('(512 threads)' if r2_wide else '(two per thread)') if r2 else 'off'}"
            f" (two-shot calls of >= {max(SERVED_R2_MIN_ROWS, TWO_SHOT_ROWS)} even rows),"
            f" SMEM pad {smem_pad} B at {SERVED_PAD_ROWS[0]}..{SERVED_PAD_ROWS[1]} rows"
        )
    )


# ------------------------------------------------------------------------------------------------
# The producer of the direct transport (``tep8_direct_hop``): o_proj's wide64 split-K kernel with
# the scatter epilogue (``run_splitk_scatter``), called by the dense GEMM dispatcher.
# ------------------------------------------------------------------------------------------------
try:
    import cuda.bindings.driver as _cuda
    import cutlass.utils.blackwell_helpers as sm100_utils
    import torch as _torch
    from cutlass.cute import experimental as cute_ext

    from ..gemm.splitk_gemm import (
        _AB_BUFFER_ALIGN_BYTES,
        _MAILBOX_ALIGN_BYTES,
        _MBARRIER_BYTES,
        _TMEM_POINTER_BYTES,
        OWNER_RANK,
        SplitKDenseGemmKernel,
        SplitKTactic,
        _make_compile_repr_tensors,
        _store_shared_remote_v4,
        _to_cute_swap,
        _validate_runtime_tensors,
        validate_tactic,
    )

    HAS_SCATTER_GEMM = HAS_CUTE_DSL
except Exception:  # noqa: BLE001 - no scatter producer: the direct transport refuses to build
    HAS_SCATTER_GEMM = False

if HAS_SCATTER_GEMM:

    class SplitKScatterKernel(SplitKDenseGemmKernel):
        """``ops/gemm/splitk_gemm.SplitKDenseGemmKernel`` (flashinfer's, moved into this target) at
        split_k 1 whose epilogue stores the tile as the TEP8 post-attention hop's direct scatter
        (``scatter_store``). Its ``kernel`` / ``epilog_warp_scatter`` are the base class's
        ``kernel`` / ``epilog_warp`` with the scatter scalars threaded through and the final store
        replaced: same mainloop, same rounding, bitwise the same values; only the addresses of the
        final stores differ. Defined here (not in the kernel file) so the split-K kernel's device
        code -- and the TP8 publish certification keyed on it (``producer_key``) -- stays
        unchanged."""

        def __init__(self, *, tactic: SplitKTactic, use_pdl: bool, fb_layout: bool = False) -> None:
            super().__init__(tactic=tactic, use_pdl=use_pdl, has_bias=False, mailbox_v4=True)
            if tactic.split_k != 1 or (fb_layout and (tactic.mma_m != 64 or tactic.mma_n != 64)):
                raise ValueError(
                    f"the scatter epilogue serves split_k 1 (fb layout: 64 x 64 tiles) only, got {tactic}"
                )
            self.scatter = True
            self.fb_layout = bool(fb_layout)

        @cute.experimental.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            c: cute.Tensor,
            peer_stride_el: cutlass.Int64,
            rows: cutlass.Int32,
            stream: _cuda.CUstream,
        ):
            self.kernel(a, b, c, c, peer_stride_el, rows).launch(
                grid=(
                    cute.ceil_div(c.layout.shape[0], self.cta_m),
                    cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                    c.layout.shape[2],
                ),
                block=(self.threads_per_cta, 1, 1),
                cluster=self.cluster_shape,
                smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.experimental.kernel
        def kernel(
            self,
            mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
            mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
            mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
            mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
            peer_stride_el: cutlass.Int64,
            rows: cutlass.Int32,
        ):
            """Allocate storage and dispatch the specialized warps."""
            stages = self.num_ab_stage

            ab_dtype = mA.element_type
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                ab_dtype,
                ab_dtype,
                utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
                utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_mn,
            )

            mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
            block_idx = cute.arch.block_idx()
            bidx = block_idx[0]
            split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            n_idx = block_idx[1] // self.split_k
            l_idx = block_idx[2]
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            sA = cute_ext.allocate(
                ab_dtype,
                cute.AddressSpace.smem,
                sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
                alignment=_AB_BUFFER_ALIGN_BYTES,
            )
            sB = cute_ext.allocate(
                ab_dtype,
                cute.AddressSpace.smem,
                sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
                alignment=_AB_BUFFER_ALIGN_BYTES,
            )

            acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
            c_tiler_mn = (self.cta_m, self.cta_n)

            bar_full = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(stages),
                alignment=_MBARRIER_BYTES,
            ).iterator
            bar_empty = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(stages),
                alignment=_MBARRIER_BYTES,
            ).iterator
            bar_tma_epilog = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
            bar_mma_epilog = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
            bar_tmem_alloc = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
            tmem_base_ptr = cute_ext.allocate(
                cutlass.Int32,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_TMEM_POINTER_BYTES,
            ).iterator

            if cutlass.const_expr(self.split_k > 1):
                mailbox = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(self.mailbox_elements),
                    alignment=_MAILBOX_ALIGN_BYTES,
                )
                bar_reduce = cute_ext.allocate(
                    cutlass.Int64,
                    cute.AddressSpace.smem,
                    cute.make_layout(1),
                    alignment=_MBARRIER_BYTES,
                ).iterator
            else:
                # Dummy operands for the compile-time-elided reduction.
                mailbox = sA
                bar_reduce = bar_mma_epilog

            if warp_idx == 0:
                with cute.arch.elect_one():
                    for i in range(stages):
                        cute.arch.mbarrier_init(bar_full + i, 2)
                        cute.arch.mbarrier_init(bar_empty + i, 1)
                    cute.arch.mbarrier_init(bar_tma_epilog, 32)
                    cute.arch.mbarrier_init(bar_mma_epilog, 1)
                    cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                    if cutlass.const_expr(self.split_k > 1):
                        # Owner arrival plus peer transaction-byte credits.
                        cute.arch.mbarrier_init(bar_reduce, 1)

            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(self.split_k > 1):
                # Publish peer barriers before cross-CTA stores.
                cute.arch.cluster_arrive_relaxed()
            else:
                cute.arch.barrier()

            # Host validation guarantees an equal, tail-free K partition.
            k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
            k_tile_start = split_rank * k_tile_count

            if cutlass.const_expr(self.split_k > 1):
                cute.arch.cluster_wait()

            # Warp 3 is idle; warps 4-7 run the epilogue.
            if warp_idx == 0:
                self.dma_warp(
                    bar_full,
                    bar_empty,
                    bar_tma_epilog,
                    cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                    sA,
                    cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                    k_tile_start,
                    k_tile_count,
                    True,
                )
            elif warp_idx == 1:
                self.dma_warp(
                    bar_full,
                    bar_empty,
                    bar_tma_epilog,
                    cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                    sB,
                    cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                    k_tile_start,
                    k_tile_count,
                    False,
                )
            elif warp_idx == 2:
                self.mma_warp(
                    bar_full,
                    bar_empty,
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tiled_mma,
                    sA,
                    sB,
                    tmem_base_ptr,
                    acc_layout,
                    self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                    k_tile_count,
                )
            elif warp_idx >= 4:
                self.epilog_warp_scatter(
                    bar_tma_epilog,
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tmem_base_ptr,
                    acc_layout,
                    cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                    cute.local_tile(mBias, c_tiler_mn, (bidx, n_idx, l_idx)),
                    cute.arch.thread_idx()[0] - 128,
                    mC.element_type,
                    utils.LayoutEnum.from_tensor(mC),
                    mailbox,
                    bar_reduce,
                    split_rank,
                    sA,
                    peer_stride_el,
                    rows,
                )

        @cute.experimental.jit
        def epilog_warp_scatter(
            self,
            bar_tma_epilog,
            bar_mma_epilog,
            bar_tmem_alloc,
            tmem_base_ptr,
            acc_layout: cutlass.Constexpr,
            gD_tile: cute.Tensor,
            gBias_tile: cute.Tensor,
            epi_tid: cutlass.Int32,
            c_dtype: cutlass.Constexpr,
            d_layout: cutlass.Constexpr,
            mailbox,
            bar_reduce,
            split_rank: cutlass.Int32,
            sA,
            peer_stride_el,
            rows,
        ):
            # Wait until MMA publishes the TMEM base pointer.
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

            acc_view = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
                acc_layout,
            )[((None, None), 0, 0, 0)]

            epi_tile = (self.cta_m, self.cta_n)
            tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.cta_m, self.cta_n, self.cta_k),
                    d_layout,
                    c_dtype,
                    self.acc_dtype,
                    epi_tile,
                    False,
                ),
                acc_view,
            )
            gD_epi = cute.flat_divide(gD_tile, epi_tile)

            # Match each epilogue thread's TMEM partition in RMEM.
            rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
            rAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rD = cute_ext.allocate(
                c_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

            if cutlass.const_expr(self.has_bias):
                bias_dtype = gBias_tile.element_type
                rBias = cute_ext.allocate(
                    bias_dtype,
                    cute.AddressSpace.rmem,
                    rmem_layout,
                    alignment=32,
                )
                rBiasAcc = cute_ext.allocate(
                    self.acc_dtype,
                    cute.AddressSpace.rmem,
                    rmem_layout,
                    alignment=32,
                )
                if split_rank == OWNER_RANK:
                    cute.arch.mbarrier_wait(bar_tma_epilog, 0)
                    cute_ext.partition_and_copy(
                        cute.make_tiled_copy_D(
                            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                            tiled_copy_t2r,
                        ).get_slice(epi_tid),
                        cute.flat_divide(gBias_tile, epi_tile)[None, None, 0, 0],
                        rBias,
                    )
                    rBiasAcc.store(rBias.load().to(self.acc_dtype))

            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
            # Make tcgen05.ld visible before TMEM release and RMEM use.
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(bar_tmem_alloc)

            # Peers publish FP32 partials; only rank 0 reduces and stores.
            if cutlass.const_expr(self.split_k > 1):
                assert cute.size(rmem_layout) == self.mailbox_elements // (
                    (self.split_k - 1) * self.epilog_threads
                )
                values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
                values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
                if split_rank != OWNER_RANK:
                    for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                        if cutlass.const_expr(self.mailbox_v4):
                            # A warp's 16-byte stores of one value group are contiguous (512 bytes).
                            slot = (value_idx // 4) * (self.epilog_threads * 4) + epi_tid * 4
                        else:
                            slot = epi_tid * values_per_thread + value_idx
                        _store_shared_remote_v4(
                            rAcc[value_idx],
                            rAcc[value_idx + 1],
                            rAcc[value_idx + 2],
                            rAcc[value_idx + 3],
                            mailbox.iterator + (split_rank - Int32(1)) * values_per_peer + slot,
                            bar_reduce,
                            Int32(OWNER_RANK),
                        )
                else:
                    if epi_tid == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar_reduce, self.expected_transaction_bytes
                        )
                    cute.arch.mbarrier_wait(bar_reduce, 0)
                    for peer in cutlass.range_constexpr(self.split_k - 1):
                        for value_idx in cutlass.range_constexpr(values_per_thread):
                            if cutlass.const_expr(self.mailbox_v4):
                                slot = (
                                    (value_idx // 4) * (self.epilog_threads * 4)
                                    + epi_tid * 4
                                    + value_idx % 4
                                )
                            else:
                                slot = epi_tid * values_per_thread + value_idx
                            # Same additions in the same order (own, then peers in rank order).
                            rAcc[value_idx] = (
                                rAcc[value_idx] + mailbox[peer * values_per_peer + slot]
                            )

            if split_rank == OWNER_RANK:
                if cutlass.const_expr(self.has_bias):
                    rAcc.store(rAcc.load() + rBiasAcc.load())

                rD.store(rAcc.load().to(c_dtype))
                # Preserve TMEM coordinates; the copy predicates output tails.
                self.scatter_store(thr_t2r, rD, sA, gD_tile, epi_tid, c_dtype, peer_stride_el, rows)

            # The reduction mbarrier covers remote stores; no cluster barrier needed.

        @cute.experimental.jit
        def scatter_store(
            self,
            thr_t2r,
            rD,
            sA,
            gD_tile: cute.Tensor,
            epi_tid: cutlass.Int32,
            c_dtype: cutlass.Constexpr,
            peer_stride_el,
            rows,
        ):
            """The tile's store as the TEP8 hop's direct scatter. Token row t of the
            tile goes to the scatter slab this rank writes on the row's owner t % 8: element (feature m,
            token t) at C + (t % 8) * peer_stride_el + (t // 8) * 8192 + m, where C (gD_tile's origin, feature
            m0 = bidx * cta_m, token 0) is this rank's slab on rank 0 of the group's contiguous unicast window.
            rD's values (the bits) are staged token-major in sA (free once bar_mma_epilog completed: the last
            MMA read it and every TMA load into it landed before), then each token's cta_m features go out as
            16-byte stores of 8 adjacent threads (one 128 B run per token and CTA at cta_m 64)."""
            pitch = self.cta_m + 8
            bidx = cute.arch.block_idx()[0]
            sT = cute.make_tensor(
                cute.recast_ptr(sA.iterator, dtype=c_dtype),
                cute.make_layout((self.cta_m, self.cta_n), stride=(1, pitch)),
            )
            cute_ext.partition_and_copy(thr_t2r, rD, sT)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
            chunks = self.cta_m // 8
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=128)
            vec = cute.make_rmem_tensor(cute.make_layout((8,)), c_dtype)
            if cutlass.const_expr(self.fb_layout):
                # Feature-block-major slots: owner p's area holds [feature block of cta_m][8 slots][cta_m features],
                # so this CTA's 8 tokens for owner p are one contiguous (8 * cta_m * 2)-byte run, and each warp
                # instruction writes 4 consecutive slots (4 x 128 B, contiguous) of one owner.
                warp = epi_tid // 32
                lane = epi_tid % 32
                fchunk = lane % chunks
                for j in cutlass.range_constexpr(4):
                    p = warp + 4 * (j // 2)
                    s = 4 * (j % 2) + lane // chunks
                    tok = p + 8 * s
                    if tok < rows:
                        cute.copy(
                            atom,
                            cute.make_tensor(
                                (sT.iterator + tok * pitch + fchunk * 8).align(16),
                                cute.make_layout((8,)),
                            ),
                            vec,
                        )
                        dst = (
                            gD_tile.iterator
                            + cutlass.Int64(bidx) * (7 * self.cta_m)
                            + cutlass.Int64(p) * peer_stride_el
                            + s * self.cta_m
                            + fchunk * 8
                        )
                        cute.copy(
                            atom, vec, cute.make_tensor(dst.align(16), cute.make_layout((8,)))
                        )
            for j in cutlass.range_constexpr(
                0 if self.fb_layout else (self.cta_m * self.cta_n) // 8 // self.epilog_threads
            ):
                idx = epi_tid + j * self.epilog_threads
                tok = idx // chunks
                chunk = idx % chunks
                if tok < rows:
                    cute.copy(
                        atom,
                        cute.make_tensor(
                            (sT.iterator + tok * pitch + chunk * 8).align(16),
                            cute.make_layout((8,)),
                        ),
                        vec,
                    )
                    dst = (
                        gD_tile.iterator
                        + cutlass.Int64(tok % 8) * peer_stride_el
                        + cutlass.Int64(tok // 8) * 8192
                        + chunk * 8
                    )
                    cute.copy(atom, vec, cute.make_tensor(dst.align(16), cute.make_layout((8,))))

            # The reduction mbarrier covers remote stores; no cluster barrier needed.

    @cute.experimental.jit
    def _bmm_scatter(
        gemm_op: cutlass.Constexpr,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        peer_stride_el: cutlass.Int64,
        rows: cutlass.Int32,
        stream: _cuda.CUstream,
    ):
        gemm_op(
            cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
            cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
            cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
            peer_stride_el,
            rows,
            stream,
        )

    @functools.cache
    def _get_compiled_scatter_kernel(
        dtype, tactic: SplitKTactic, use_pdl: bool, leading_dims, fb_layout: bool = False
    ):
        kernel = SplitKScatterKernel(tactic=tactic, use_pdl=use_pdl, fb_layout=fb_layout)
        a_rep, b_rep, c_rep, _ = _make_compile_repr_tensors(dtype, False, *leading_dims)
        stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
        return cute_ext.compile(
            _bmm_scatter, kernel, a_rep, b_rep, c_rep, cutlass.Int64(0), cutlass.Int32(0), stream
        )

    def run_splitk_scatter(
        a, b, out, pdl: bool, tactic: SplitKTactic, peer_stride_bytes: int, fb_layout: bool = False
    ):
        """``A @ B`` stored as the hop's direct scatter: ``out`` [M, N] is a view whose origin is
        this rank's scatter slab on rank 0 of the window (only its address is used); token row t lands at
        ``out`` + (t % 8) * peer_stride_bytes + (t // 8) * N * 2."""
        m, n, k = _validate_runtime_tensors(a, b, None, out)
        validate_tactic(tactic, m, n, k)
        if m > tactic.mma_n or n != 8192 or peer_stride_bytes % 16:
            raise ValueError(f"scatter epilogue: rows <= mma_n, 8192 columns; got {m} x {n}")
        cute_tensors = _to_cute_swap(a, b, out, None)
        compiled = _get_compiled_scatter_kernel(
            a.dtype, tactic, pdl, cute_tensors[4], bool(fb_layout)
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
        compiled(*cute_tensors[:3], cutlass.Int64(peer_stride_bytes // 2), cutlass.Int32(m), stream)
        return out
