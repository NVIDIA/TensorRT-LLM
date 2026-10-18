# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-side hop of a decoder layer in one launch: all-reduce, residual join, prenorm.

A decode step of the built-in layer spends two launches between the row-parallel
output projection and the MoE: the MNNVL one-shot all-reduce of the projection's
BF16 partial (``mnnvl::oneshotAllreduceFusionKernel``), then the cluster
boundary (``musespark_open_boundary``), which normalizes the reduced branch
without weights, joins it into the FP32 residual through the gated add and
applies the Gemma pre-FFN norm. This entry is both in one kernel: the reduced
branch never leaves registers, and the second launch with its own dependency
wait is gone.

It is bit-identical to that pair, not merely close. The all-reduce part is the
legacy one-shot arithmetic -- every rank's partial with BF16 ``-0`` cleared,
summed in FP32 in rank order starting from ``+0``, rounded once to BF16 -- and
the boundary part is the boundary kernel's statements on the boundary kernel's
thread-to-column map and cluster size, so its variance reductions happen in the
same order.

Transport (one-shot Lamport over the NVLink fabric). Each rank owns a receive
area of ``2 buffers x 8 source ranks x 8 rows x 8192`` BF16, initialized to the
sentinel word ``0x80000000``; a 32-bit word that still holds it has not arrived.
A rank publishes its sanitized partial into slot ``rank`` of the current buffer
on every peer, either with one store to the multicast address or with seven
unicast stores, polls the seven peer slots of its own buffer until no word is
the sentinel, and writes the sentinel back into each slot it consumed. The
buffer index alternates per call: a peer can write call ``n + 2`` into a buffer
only after it read this rank's call ``n + 1``, which this rank publishes only
after its call ``n`` -- including those resets -- completed.

Polling schedule. By default every polling round reads all seven peer slots
again until a round finds no sentinel word. With ``stale_poll`` (the stale-only
schedule) the first round reads every peer slot and each later round reads only
the slots that still held a sentinel word, all of them issued before any is
waited on; a slot whose words have all arrived is not read again. The values
are the same: in one call a slot has exactly one writer (its source rank's
single publication), and the payload never contains the sentinel, so a word
that does not hold it any more already holds its final value, and nothing writes the
slot again before this rank resets it. The read that first finds no sentinel
word in a slot therefore returns what any later read would. A ``stale_poll``
workspace polls stale-only on calls of at least ``STALE_POLL_MIN_ROWS`` rows
and with the default schedule below that; the two schedules share everything
but the reads of the polling loop, so calls of either kind interleave on one
workspace.

The index lives on the device, so graph replays and eager calls interleave
freely: the CTAs of every call add exactly ``GRID_STEP`` (64) to a free-running
per-rank arrival counter -- CTA 0 adds ``65 - grid``, every other CTA one --
each after all its threads have read the counter. A CTA of call ``n``
therefore reads ``64 n + k`` with ``0 <= k < 64``: every earlier call completed
before its dependency wait returned, and its own CTA has not arrived yet. So
``(counter >> 6) & 1`` is the call's parity on every CTA, and nothing waits for
an atomic's result. ``2**32`` is a multiple of 128, so the parity sequence
survives the counter's wrap.

Ordering contract: calls on one workspace must be issued in the same order on
every rank and serialized on one stream (graph replay included), and every
kernel launched with programmatic dependent launch between two calls must wait
on its predecessor before it completes -- the same contract the built-in MNNVL
all-reduce and the expert finalize rely on.

What the tests can and cannot see. The data path (publication, polling, the
rank-order sum, the slot resets) and the counter's accounting (exactly 64 per
call, across the u32 wrap) are tested. The two-buffer alternation itself is
not observable: a fast rank's next publish could only collide with a slow
peer's read of the same slot if that read took longer than the fast rank's
whole next kernel boundary, and no host-side skew opens that window. It rests
on the argument above.

Direct publication (``attention_output_into_hop``, ``PostAttnDirectWorkspace`` and
``post_attn_consumer_hop`` below). The producer of the partial -- the o_proj
GEMM -- writes its output straight into this rank's slot of a second
transport through the multicast address, so its own epilogue stores are the
publication, and the hop is a consumer only: it polls all eight slots of its
local copy (its own included), resets them, sums and runs the boundary. Two
things change with the producer. A library epilogue cannot clear ``-0``, so
the empty-slot sentinel is a value no GEMM writes: the BF16 pair
``0xFFFE 0xFFFE``, a NaN with a payload that float-to-BF16 rounding never
produces (it gives the canonical ``0x7FFF``), tested per 16-bit half, so a
word whose two halves arrive in separate stores is not taken half-written.
And a GEMM cannot read the arrival counter, so the buffer index is a launch
argument the caller fixes per call site (layer ``l`` uses ``l mod 2``, see
``PostAttnDirectWorkspace``).

What the direct transport needs from its users, and what checks it. The
producer must store each output element once and never reduce through its
output: ``check_producer`` reads the kernels a producer captures into a CUDA
graph (never replayed) and accepts exactly one of the device's ``VALIDATED_PRODUCERS``.
Every published call must be consumed before its buffer is published into
again; a call sequence that breaks off between the two (a host exception)
leaves words a later call would take as its own, so ``reset`` refills every
rank's copy with the sentinel (collective). A consumer whose peer never
publishes polls on: after ``DIRECT_GUARD_NS`` it sets the guard word, which
lives in pinned host memory so that a watchdog thread can read it while the
device is still polling and log the stuck rank.
"""

from __future__ import annotations

import functools
import threading
import time
import weakref
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

#: The kernels launch with programmatic dependent launch: ``TRTLLM_ENABLE_PDL`` on is a
#: precondition of every composition that builds them.
_LAUNCH_PDL = True

#: Residual width the kernel's thread map is written for.
HIDDEN = 8192
#: Rows of a decode step the transport is sized for: one request's last token and 7 drafts.
MAX_ROWS = 8
#: Largest tensor-parallel group; slots are laid out for it whatever the group size.
MAX_RANKS = 8
#: BF16 values per 16-byte access, one access per thread.
VECTOR = 8
SENTINEL = 0x80000000
BUFFERS = 2
SLAB_BYTES = MAX_ROWS * HIDDEN * 2
WORKSPACE_BYTES = BUFFERS * MAX_RANKS * SLAB_BYTES
#: Per-rank device state: word 1 is the arrival counter; the rest is reserved.
CONTROL_WORDS = 4
#: What one call adds to the arrival counter, whatever its grid (at most 64 CTAs).
GRID_STEP = 64
GRID_SHIFT = 6
assert GRID_STEP == 1 << GRID_SHIFT and GRID_STEP >= MAX_ROWS * 8 and GRID_STEP >= 16


#: Fewest rows of a call that a ``stale_poll`` workspace polls with the stale-only
#: schedule (8: the K = 7 verify step). Smaller calls keep the default schedule.
STALE_POLL_MIN_ROWS = 8


def cluster_for_rows(rows: int) -> int:
    """The cluster boundary's split for ``rows``; the variance tree depends on it."""
    return 16 if rows == 1 else 8


def uses_stale_poll(stale_poll: bool, rows: int, min_rows: int = STALE_POLL_MIN_ROWS) -> bool:
    """Whether a call of ``rows`` rows polls stale-only on a workspace built with ``stale_poll``."""
    return bool(stale_poll) and rows >= min_rows


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op
except ImportError:
    HAS_CUTE_DSL = False
else:
    HAS_CUTE_DSL = True


if HAS_CUTE_DSL:
    _FOUR_WORDS = "!llvm.struct<(i32,i32,i32,i32)>"

    def _asm(result_type, operands, text, constraints, loc, ip, side_effects=True):
        return llvm.inline_asm(
            result_type,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    def _u32(value, loc, ip):
        return Uint32(value).ir_value(loc=loc, ip=ip)

    def _i64(value, loc, ip):
        return Int64(value).ir_value(loc=loc, ip=ip)

    def _split_words(result, loc, ip):
        return tuple(
            Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4)
        )

    # Every global access of the transport is a side-effecting asm statement, so
    # none of them can move above griddepcontrol.wait or merge across polls.

    @dsl_user_op
    def _float_bits(value, *, loc=None, ip=None):
        """The int32 with the same bit pattern as an FP32 value."""
        bits = llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return Int32(bits)

    @dsl_user_op
    def _load_x4(address, *, loc=None, ip=None):
        """Coherent 16-byte global load (the projection was written by the previous grid)."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip)],
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return _split_words(result, loc, ip)

    @dsl_user_op
    def _poll_x4(address, enabled, o0, o1, o2, o3, *, loc=None, ip=None):
        """Relaxed system-scope 16-byte load when ``enabled``; else the given words, no access."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip), _u32(enabled, loc, ip)]
            + [_u32(w, loc, ip) for w in (o0, o1, o2, o3)],
            "{\n.reg .pred p;\n.reg .b32 a, b, c, d;\n"
            "mov.b32 a, $6;\nmov.b32 b, $7;\nmov.b32 c, $8;\nmov.b32 d, $9;\n"
            "setp.ne.u32 p, $5, 0;\n"
            "@p ld.relaxed.sys.global.v4.b32 {a, b, c, d}, [$4];\n"
            "mov.b32 $0, a;\nmov.b32 $1, b;\nmov.b32 $2, c;\nmov.b32 $3, d;\n}",
            "=r,=r,=r,=r,l,r,r,r,r,r",
            loc,
            ip,
        )
        return _split_words(result, loc, ip)

    @dsl_user_op
    def _stale_bit(w0, w1, w2, w3, bit, *, loc=None, ip=None):
        """``bit`` while any of the four words is the sentinel, else 0, without a branch."""
        result = _asm(
            T.i32(),
            [_u32(w, loc, ip) for w in (w0, w1, w2, w3, bit)],
            "{\n.reg .pred p0, p1, p2, p3;\n"
            "setp.eq.u32 p0, $1, 0x80000000;\nsetp.eq.u32 p1, $2, 0x80000000;\n"
            "setp.eq.u32 p2, $3, 0x80000000;\nsetp.eq.u32 p3, $4, 0x80000000;\n"
            "or.pred p0, p0, p1;\nor.pred p2, p2, p3;\nor.pred p0, p0, p2;\n"
            "selp.b32 $0, $5, 0, p0;\n}",
            "=r,r,r,r,r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _reset_x4(address, enabled, *, loc=None, ip=None):
        """Write the sentinel into one consumed 16-byte slot when ``enabled``."""
        _asm(
            None,
            [_i64(address, loc, ip), _u32(enabled, loc, ip)],
            "{\n.reg .pred p;\n.reg .b32 s;\nmov.b32 s, 0x80000000;\n"
            "setp.ne.u32 p, $1, 0;\n"
            "@p st.global.v4.b32 [$0], {s, s, s, s};\n}",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _store_peer_x4(address, w0, w1, w2, w3, *, loc=None, ip=None):
        """Relaxed system-scope 16-byte store into a peer's unicast mapping."""
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.relaxed.sys.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _store_multicast_x4(address, w0, w1, w2, w3, *, loc=None, ip=None):
        """16-byte store to a multicast address: the fabric writes it to every member."""
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _load_counter(address, *, loc=None, ip=None):
        result = _asm(
            T.i32(),
            [_i64(address, loc, ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
        return Uint32(result)

    @dsl_user_op
    def _arrive(address, amount, *, loc=None, ip=None):
        """Add ``amount`` to the arrival counter; nothing waits for the result."""
        _asm(
            None,
            [_i64(address, loc, ip), _u32(amount, loc, ip)],
            "red.relaxed.gpu.global.add.u32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _sanitize_bf16x2(word, *, loc=None, ip=None):
        """Turn a BF16 ``-0`` in either half into ``+0``, so no word equals the sentinel."""
        result = _asm(
            T.i32(),
            [_u32(word, loc, ip)],
            "{\n.reg .b32 w, m;\n.reg .pred p;\nmov.b32 w, $1;\n"
            "and.b32 m, w, 0xffff;\nsetp.eq.u32 p, m, 0x8000;\n@p and.b32 w, w, 0xffff0000;\n"
            "and.b32 m, w, 0xffff0000;\nsetp.eq.u32 p, m, 0x80000000;\n@p and.b32 w, w, 0xffff;\n"
            "mov.b32 $0, w;\n}",
            "=r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _unpack_bf16x2(word, *, loc=None, ip=None):
        """FP32 values of the low and high BF16 halves of ``word`` (exact)."""
        result = _asm(
            ir.Type.parse("!llvm.struct<(f32,f32)>"),
            [_u32(word, loc, ip)],
            "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
            "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
            "=f,=f,r",
            loc,
            ip,
            side_effects=False,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        )

    class _PostAttnHopKernel:
        """One-shot all-reduce + attention-side cluster boundary for a fixed group and split.

        CTA ``b`` handles row ``b // cluster`` and the ``b % cluster``-th slice of
        the 8192 columns; each thread owns one 16-byte column vector of it -- the
        cluster boundary's map, which fixes the variance reduction order.
        """

        def __init__(
            self,
            ranks: int,
            cluster: int,
            multicast: bool,
            post_eps: float,
            pre_eps: float,
            norm_fp32: bool,
            pdl: bool,
            early_trigger: bool,
            stale_poll: bool = False,
        ):
            self.ranks = ranks
            self.cluster = cluster
            self.threads = HIDDEN // (cluster * VECTOR)
            self.iters = HIDDEN // (cluster * self.threads * VECTOR)
            self.warps = self.threads // 32
            self.slots = cluster * self.warps
            self.multicast = multicast
            self.post_eps, self.pre_eps = float(post_eps), float(pre_eps)
            self.norm_fp32 = norm_fp32
            self.pdl = pdl
            self.early_trigger = early_trigger
            self.stale_poll = stale_poll

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
                use_pdl=self.pdl,
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
            if cutlass.const_expr(self.cluster > 1):
                if lane < self.cluster:
                    slot = partials.iterator + (phase * self.slots + rank * self.warps + warp) * 4
                    cute.arch.store_async_dsmem(
                        slot.align(16),
                        (_float_bits(value), Int32(0), Int32(0), Int32(0)),
                        bars.iterator + phase,
                        lane,
                    )
                cute.arch.mbarrier_wait(bars.iterator + phase, 0)
            else:
                if lane == 0:
                    partials[(phase * self.slots + warp) * 4] = value
                cute.arch.barrier()
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
                if cutlass.const_expr(self.cluster > 1):
                    if tid < 2:
                        cute.arch.mbarrier_init(bars.iterator + tid, 1)
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bars.iterator + tid, self.slots * 16
                        )
                    cute.arch.mbarrier_init_fence()
                    cute.arch.barrier()
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()
                # Nothing above touches global memory: the projection, the
                # residual, the coefficients and the transport are all read
                # after this wait.
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                # The arrival counter holds 64 per completed call plus less than
                # 64 of this one (this CTA arrives only after all its threads
                # have read it), so its 64s place is the call's parity.
                counter = _load_counter(control_address + 4)
                buffer = Int64((counter >> GRID_SHIFT) & Uint32(1))
                slot = (Int64(row) * HIDDEN + Int64(offset)) * 2
                words = _load_x4(partial.toint() + slot)
                a = self.load(alpha + offset, Float32)
                b = self.load(beta + offset, Float32)
                w = self.load(weight + offset, cutlass.BFloat16)
                r = self.load(residual + row * 8192 + offset, Float32)

                # Publish this rank's sanitized partial into its slot on every rank.
                own = (
                    _sanitize_bf16x2(words[0]),
                    _sanitize_bf16x2(words[1]),
                    _sanitize_bf16x2(words[2]),
                    _sanitize_bf16x2(words[3]),
                )
                own_slab = (buffer * MAX_RANKS + Int64(rank)) * SLAB_BYTES
                if cutlass.const_expr(self.multicast):
                    _store_multicast_x4(mcast.toint() + own_slab + slot, *own)
                else:
                    for p in cutlass.range_constexpr(self.ranks):
                        if rank != p:
                            _store_peer_x4(peers[p].toint() + own_slab + slot, *own)

                # Poll every peer slot of the local buffer until none holds the
                # sentinel; this rank's own slot is never read (its value is `own`).
                receive_address = receive.toint()
                values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                if cutlass.const_expr(self.stale_poll):
                    # Stale-only polling: after one read of every peer slot, only
                    # the slots that still hold a sentinel word are read again,
                    # all of them in flight together, and a slot is never read
                    # once all its words have arrived. Bit ``p`` of ``stale`` is
                    # set while slot ``p`` still holds a sentinel word; a load
                    # whose bit is clear keeps the words it already has, so the
                    # rounds carry no branch per slot. Every load of a round is
                    # issued before any of its words is tested.
                    for p in cutlass.range_constexpr(self.ranks):
                        address = receive_address + (buffer * MAX_RANKS + p) * SLAB_BYTES + slot
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
                            address = receive_address + (buffer * MAX_RANKS + p) * SLAB_BYTES + slot
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
                else:
                    pending = Int32(1)
                    while pending != 0:
                        pending = Int32(0)
                        for p in cutlass.range_constexpr(self.ranks):
                            address = receive_address + (buffer * MAX_RANKS + p) * SLAB_BYTES + slot
                            v0, v1, v2, v3 = _poll_x4(
                                address, Uint32(rank != p), own[0], own[1], own[2], own[3]
                            )
                            values[p, 0] = v0
                            values[p, 1] = v1
                            values[p, 2] = v2
                            values[p, 3] = v3
                            late = (
                                (v0 == SENTINEL)
                                | (v1 == SENTINEL)
                                | (v2 == SENTINEL)
                                | (v3 == SENTINEL)
                            )
                            if late:
                                pending = Int32(1)
                # Retire every consumed slot before this grid can complete.
                for p in cutlass.range_constexpr(self.ranks):
                    address = receive_address + (buffer * MAX_RANKS + p) * SLAB_BYTES + slot
                    _reset_x4(address, Uint32(rank != p))
                # Early trigger: past the last transport access. A dependent may
                # start its prologue now; its griddepcontrol.wait still returns
                # only once this grid has completed and its stores are visible.
                if cutlass.const_expr(self.pdl and self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

                # The built-in one-shot sum: FP32 from +0, ranks in order, one RN round.
                totals = [Float32(0.0)] * VECTOR
                for p in cutlass.range_constexpr(self.ranks):
                    for k in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(values[p, k])
                        totals[2 * k] = totals[2 * k] + low
                        totals[2 * k + 1] = totals[2 * k + 1] + high
                x = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                for v in cutlass.range_constexpr(8):
                    x[v, 0] = cutlass.BFloat16(totals[v])

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
                    cute.make_tensor((output + row * 8192 + offset).align(16), layout),
                )
                cute.copy(
                    atom16,
                    normed,
                    cute.make_tensor((next_output + row * 8192 + offset).align(16), layout),
                )
                # No exit cluster barrier: both reductions waited on this CTA's own
                # mbarrier for every byte its peers send, so no peer touches this
                # CTA's shared memory any more. This barrier only orders every
                # thread's counter read before the CTA's arrival below.
                cute.arch.barrier()
                # The call's arrivals add up to GRID_STEP whatever its grid: CTA 0
                # carries the remainder, every other CTA one.
                if tid == 0:
                    block = cute.arch.block_idx()[0]
                    blocks = cute.arch.grid_dim()[0]
                    amount = Uint32(1)
                    if block == 0:
                        amount = Uint32(GRID_STEP + 1 - blocks)
                    _arrive(control_address + 4, amount)
                # Late trigger: after every store, the boundary kernel's own placement.
                if cutlass.const_expr(self.pdl and not self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

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
    ) + (Uint32,) * MAX_RANKS

    @functools.lru_cache(maxsize=64)
    def _compiled_kernel(
        device_index: int,
        ranks: int,
        cluster: int,
        multicast: bool,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        pdl: bool,
        early_trigger: bool,
        stale_poll: bool = False,
    ):
        kernel = _PostAttnHopKernel(
            ranks, cluster, multicast, post_eps, pre_eps, norm_fp32, pdl, early_trigger, stale_poll
        )
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *[
                    make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
                    for dtype in _POINTER_DTYPES
                ],
                Int32(1),
                Int32(0),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


class PostAttnHopWorkspace:
    """Transport memory and device state of the fused hop for one tensor-parallel group.

    Private to this op: it is never shared with the built-in MNNVL all-reduce,
    the expert finalize or the projection gather, whose protocols differ. One
    instance serves every layer of a model; calls must be issued in the same
    order on every rank and serialized on one stream.

    Construction is collective over the group (fabric allocation, sentinel fill,
    kernel compilation and a dry launch per variant), and either succeeds on
    every rank or raises on every rank. Nothing compiles or loads a module
    afterwards unless a caller asks for a variant it did not pre-build, which
    then happens on the first eager call (never inside a capture). A model
    takes its workspace from ``shared_workspace``, which builds one per group
    and transport and hands the same one to every later caller.

    ``early_trigger`` places the kernel's programmatic-dependent-launch trigger
    right after its last transport access instead of after its output stores.
    Either placement is correct for a dependent that waits on this grid; they
    differ only in how early its prologue may start.

    ``stale_poll`` reads a peer slot again only while it still holds a sentinel
    word, with every outstanding read in flight together, instead of reading
    every peer slot again in every polling round. Same protocol and the same
    values (see the module docstring); fewer loads per polling round. It
    applies to calls of at least ``stale_poll_min_rows`` rows (default
    ``STALE_POLL_MIN_ROWS``); smaller calls run the default schedule.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        multicast: bool = True,
        variants: Sequence[Tuple[float, float, bool]] = (),
        early_trigger: bool = False,
        stale_poll: bool = False,
        stale_poll_min_rows: int = STALE_POLL_MIN_ROWS,
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
        self.early_trigger = bool(early_trigger)
        self.stale_poll = bool(stale_poll)
        self.stale_poll_min_rows = int(stale_poll_min_rows)
        if self.ranks not in (4, MAX_RANKS):
            raise ValueError(f"the fused hop serves 4 or 8 ranks, got {self.ranks}")
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._kernels = {}
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("CuTe DSL is unavailable")
            if torch.cuda.get_device_capability(self.device) not in ((10, 3), (10, 7)):
                raise RuntimeError("the fused hop runs on SM103 or SM107 only")
            use_fabric = mapping.is_multi_node()
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, WORKSPACE_BYTES, mapping, use_fabric
            )
            words = WORKSPACE_BYTES // 4
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
                f"post-attention hop workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        receive = self._uc[self.rank].data_ptr()
        peers = [t.data_ptr() for t in self._uc] + [receive] * (MAX_RANKS - self.ranks)
        self._fixed = (self.control.data_ptr(), receive, self._mc.data_ptr(), *peers)
        self.build_variants(variants)

    def build_variants(self, variants: Sequence[Tuple[float, float, bool]]) -> None:
        """Compile and load every split of each ``(post_eps, pre_eps, norm_fp32)`` not built yet.

        Collective: either every rank of the group has them afterwards or every
        rank raises.
        """
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        error: Optional[BaseException] = None
        try:
            # Every (split, schedule) pair some row count 1..MAX_ROWS launches.
            kinds = sorted(
                {
                    (cluster_for_rows(rows), self._stale_for(rows))
                    for rows in range(1, MAX_ROWS + 1)
                },
                reverse=True,
            )
            for post_eps, pre_eps, norm_fp32 in variants:
                for cluster, stale in kinds:
                    self._kernel(cluster, post_eps, pre_eps, norm_fp32, stale)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "post-attention hop kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _stale_for(self, rows: int) -> bool:
        return uses_stale_poll(self.stale_poll, rows, self.stale_poll_min_rows)

    def _kernel(self, cluster: int, post_eps: float, pre_eps: float, norm_fp32: bool, stale: bool):
        key = (cluster, float(post_eps), float(pre_eps), bool(norm_fp32), bool(stale))
        compiled = self._kernels.get(key)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"post-attention hop variant {key} was not built before graph capture"
            )
        compiled = _compiled_kernel(
            self.device.index,
            self.ranks,
            cluster,
            self.multicast,
            float(post_eps),
            float(pre_eps),
            bool(norm_fp32),
            self.pdl,
            self.early_trigger,
            bool(stale),
        )
        # A dry launch loads the module now, not inside a capture.
        dummy = self.control.data_ptr()
        compiled(*([dummy] * 7), *self._fixed, 1, self.rank, 1, self._stream())
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
                f"post_attn_hop inputs are on {partial.device}, its workspace on {self.device}"
            )
        rows = partial.shape[0]
        compiled = self._kernel(
            cluster_for_rows(rows), post_eps, pre_eps, norm_fp32, self._stale_for(rows)
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


def shared_workspace(
    mapping: Mapping,
    *,
    multicast: bool = True,
    variants: Sequence[Tuple[float, float, bool]] = (),
    early_trigger: bool = False,
    stale_poll: bool = False,
) -> PostAttnHopWorkspace:
    """The one workspace of this group and transport, built on first use.

    Collective, like the constructor. A later call with the same group,
    transport, trigger placement, polling schedule and PDL switch -- a model
    constructor run a second time, say -- gets the same workspace back, with
    any variant it names built as well, instead of a second fabric allocation.
    """
    key = (mapping, bool(multicast), bool(early_trigger), _LAUNCH_PDL, bool(stale_poll))
    workspace = _SHARED.get(key)
    if workspace is None:
        workspace = PostAttnHopWorkspace(
            mapping,
            multicast=multicast,
            variants=variants,
            early_trigger=early_trigger,
            stale_poll=stale_poll,
        )
        _SHARED[key] = workspace
    else:
        workspace.build_variants(variants)
    return workspace


def hop_takes(
    partial: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
) -> bool:
    """Whether the hop computes this call: row count, dtypes, shapes and devices only.

    Every rank of the group gives the same answer for the same layer call, which
    is what lets a caller choose between this collective and the built-in
    all-reduce on it. A tensor's address and strides are rank-local and are not
    consulted (``hop_layout_ok`` / ``readable``).
    """
    rows = partial.shape[0] if partial.ndim == 2 else 0
    if not 1 <= rows <= MAX_ROWS:
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


def hop_layout_ok(*tensors: torch.Tensor) -> bool:
    """Whether the kernel's 16-byte vector map can read each tensor as it is."""
    return all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)


def readable(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Each tensor, or a contiguous copy where the kernel could not read it in place.

    A fresh allocation is aligned far beyond 16 bytes. Only rank-local work, so
    a caller that has already chosen the hop on ``hop_takes`` stays on it on
    every rank whatever one rank's layout is.
    """
    out = tuple(
        t if hop_layout_ok(t) else t.clone(memory_format=torch.contiguous_format) for t in tensors
    )
    if any(a is not b for a, b in zip(out, tensors)):
        logger.warning_once(
            "[musespark_open] post_attention_hop: a post_attn_hop input was not contiguous or not 16-byte aligned; "
            "copied before the fused hop",
            key="musespark_open.post_attn_hop.copied_input",
        )
    return out


def hop_supported(
    partial: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
) -> bool:
    """Whether the kernel reads these inputs correctly as they are: ``hop_takes`` and layout."""
    return hop_takes(partial, residual, alpha, beta, next_weight) and hop_layout_ok(
        partial, residual, alpha, beta, next_weight
    )


# ---------------------------------------------------------------------------
# Direct publication (attention_output_into_hop): the producer publishes, the hop consumes.
# ---------------------------------------------------------------------------

#: The empty-slot word of the direct transport: two BF16 NaNs with payload 0xFFFE.
DIRECT_SENTINEL = 0xFFFEFFFE
#: One half of it. Float-to-BF16 rounding returns the canonical NaN 0x7FFF for a NaN input,
#: so no GEMM epilogue stores this half; a half that still holds it has not arrived.
DIRECT_SENTINEL_HALF = 0xFFFE
#: The sentinel word as the int32 the workspace is filled with.
_DIRECT_SENTINEL_I32 = DIRECT_SENTINEL - (1 << 32)
#: Word of ``PostAttnDirectWorkspace.control`` the poll guard sets.
GUARD_WORD = 2
#: Default poll guard of the consumer hop, in ns: past it without every slot arrived, a
#: polling thread sets the guard word and keeps polling (it never consumes a half that
#: has not arrived). Far above any wait the protocol itself produces; a parity or call
#: sequence that differs between ranks shows up there instead of only as a hang.
DIRECT_GUARD_NS = 1 << 33
#: How often the guard watchdog of a direct workspace reads the guard word, in s.
DIRECT_WATCHDOG_S = 10.0
#: The kernels shown to publish into the direct transport correctly, per device compute
#: capability: one store per output element through the multicast address, no reduction
#: through the output (a concurrent-poller test at every row count; NaN / Inf inputs never
#: store the sentinel half). A captured kernel matches an entry its name contains.
#:
#: * ``(10, 3)``: cuBLAS's choice for the 8-row o_proj, ``[8, 1024] x [1024, 8192]`` BF16.
#: * ``(10, 7)``: flashinfer's split-K and direct kernels, the o_proj kernels the dense GEMM
#:   plan names there; cuBLAS's choice on this arch is not one.
VALIDATED_PRODUCERS: Dict[Tuple[int, int], FrozenSet[str]] = {
    (10, 3): frozenset({"nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT"}),
    (10, 7): frozenset(
        {
            "flashinfergemmkernelsdense_bf16_gemm_sm100_splitkSplitKDenseGemmKernel",
            "flashinfergemmkernelsdense_bf16_gemm_directDirectDenseGemmKernel",
        }
    ),
}


def validated_producers(capability: Tuple[int, int]) -> FrozenSet[str]:
    """``VALIDATED_PRODUCERS`` of a device of compute ``capability``; empty on any other."""
    return VALIDATED_PRODUCERS.get(tuple(capability), frozenset())


def validated_producer(name: str, capability: Tuple[int, int]) -> bool:
    """Whether a captured kernel ``name`` is a validated producer on a device of ``capability``."""
    return any(validated in name for validated in validated_producers(capability))


def _driver_check(result) -> tuple:
    from cuda.bindings import driver

    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {status}")
    return tuple(values)


def _graph_node_name(node) -> str:
    """A captured node's kernel name, or ``<node type>`` for a node that is not a kernel."""
    from cuda.bindings import driver

    (kind,) = _driver_check(driver.cuGraphNodeGetType(node))
    if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
        return f"<{kind.name}>"
    (params,) = _driver_check(driver.cuGraphKernelNodeGetParams(node))
    status, raw = driver.cuFuncGetName(params.func)
    if status != driver.CUresult.CUDA_SUCCESS:
        status, raw = driver.cuKernelGetName(params.kern)
    if status != driver.CUresult.CUDA_SUCCESS:
        return "<unnamed kernel>"
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def captured_nodes(
    produce: Callable[[], None],
    device: torch.device,
    warm: Optional[Callable[[], None]] = None,
) -> List[str]:
    """The nodes of a CUDA graph that captures ``produce()`` on ``device``: kernel names,
    ``<node type>`` for any other node.

    The graph is read and destroyed, never instantiated or replayed, so nothing
    ``produce`` would store is stored. ``warm`` runs once, eagerly, on the capture stream
    first (lazy library state is created outside the capture).
    """
    from cuda.bindings import driver

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    if warm is not None:
        with torch.cuda.stream(stream):
            warm()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        produce()
    try:
        handle = driver.CUgraph(graph.raw_cuda_graph())
        count = int(_driver_check(driver.cuGraphGetNodes(handle, 0))[-1])
        nodes = _driver_check(driver.cuGraphGetNodes(handle, count))[0]
        return [_graph_node_name(node) for node in list(nodes)[:count]]
    finally:
        graph.reset()


def _watch_guard(ref: "weakref.ReferenceType", interval: float) -> None:
    """Log once when a direct workspace's guard word is set; stop with the workspace."""
    while True:
        time.sleep(interval)
        workspace = ref()
        if workspace is None:
            return
        if workspace.guard_set():
            logger.error(
                f"[musespark_open] attention_output_into_hop: rank {workspace.rank}'s post-attention consumer hop "
                "has polled "
                f"the direct transport for more than {workspace.guard_ns / 1e9:.1f} s (guard "
                "word set): a peer never published into the buffer this rank waits on, so the "
                "ranks' call sequences or buffer parities differ; the step cannot finish"
            )
            return
        del workspace


if HAS_CUTE_DSL:

    @dsl_user_op
    def _direct_stale_bit(w0, w1, w2, w3, bit, *, loc=None, ip=None):
        """``bit`` while any 16-bit half of the four words is ``DIRECT_SENTINEL_HALF``, else 0."""
        text = "{\n.reg .pred p;\n.reg .b32 x, h, s;\nmov.b32 s, 0;\n"
        for operand in range(1, 5):
            text += (
                f"xor.b32 x, ${operand}, 0xFFFEFFFE;\n"
                "and.b32 h, x, 0xFFFF;\nsetp.eq.u32 p, h, 0;\n@p mov.b32 s, 1;\n"
                "shr.b32 h, x, 16;\nsetp.eq.u32 p, h, 0;\n@p mov.b32 s, 1;\n"
            )
        text += "setp.ne.u32 p, s, 0;\nselp.b32 $0, $5, 0, p;\n}"
        result = _asm(
            T.i32(),
            [_u32(w, loc, ip) for w in (w0, w1, w2, w3, bit)],
            text,
            "=r,r,r,r,r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _direct_reset_x4(address, *, loc=None, ip=None):
        """Write the direct sentinel into one consumed 16-byte slot of the local copy."""
        _asm(
            None,
            [_i64(address, loc, ip)],
            "{\n.reg .b32 s;\nmov.b32 s, 0xFFFEFFFE;\nst.global.v4.b32 [$0], {s, s, s, s};\n}",
            "l",
            loc,
            ip,
        )

    @dsl_user_op
    def _globaltimer(*, loc=None, ip=None):
        """The global nanosecond timer (side-effecting, so every read is a fresh one)."""
        result = _asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip)
        return Int64(result)

    @dsl_user_op
    def _set_guard_word(address, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip)],
            "{\n.reg .b32 one;\nmov.b32 one, 1;\nst.relaxed.sys.global.b32 [$0], one;\n}",
            "l",
            loc,
            ip,
        )

    class _PostAttnConsumerKernel(_PostAttnHopKernel):
        """The consumer half of the hop on the direct transport.

        ``_PostAttnHopKernel``'s thread map, rank-order sum and boundary, statement for
        statement; gone are the arrival counter, the load of this rank's partial and its
        publication -- the producer stored the partial into this rank's slot of every
        rank's copy. Every slot of the local copy is polled, this rank's own (the
        multicast loopback) included, with the stale-only schedule: one read of every
        slot, then reads of only the slots that still hold a sentinel half, all issued
        before any is tested. Then every slot is reset.

        PDL memory order: before ``griddepcontrol.wait`` the kernel touches shared
        memory and the cluster barrier only. The residual, the coefficients and the
        weight are read after it (the residual was written by kernels the wait orders),
        and the transport is read after it too. The transport is not ordered by the
        wait for the peers' slots, nor for this rank's own (the multicast stores of the
        producer reach the local copy asynchronously): the sentinel test is what orders
        them, as in every Lamport exchange.
        """

        def __init__(
            self,
            ranks: int,
            cluster: int,
            post_eps: float,
            pre_eps: float,
            norm_fp32: bool,
            pdl: bool,
            early_trigger: bool,
            guard_ns: int,
            guard_break: bool,
        ):
            super().__init__(ranks, cluster, True, post_eps, pre_eps, norm_fp32, pdl, early_trigger)
            self.guard_ns = int(guard_ns)
            self.guard_break = bool(guard_break)

        @cute.jit
        def __call__(
            self,
            residual: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            rows: Int32,
            parity: Int32,
            dry: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                residual,
                alpha,
                beta,
                weight,
                output,
                next_output,
                control,
                receive,
                rows,
                parity,
                dry,
            ).launch(
                grid=[rows * self.cluster, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 2 + 16,
                stream=stream,
                use_pdl=self.pdl,
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
            control: cute.Pointer,
            receive: cute.Pointer,
            rows: Int32,
            parity: Int32,
            dry: Int32,
        ):
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
                if cutlass.const_expr(self.cluster > 1):
                    if tid < 2:
                        cute.arch.mbarrier_init(bars.iterator + tid, 1)
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bars.iterator + tid, self.slots * 16
                        )
                    cute.arch.mbarrier_init_fence()
                    cute.arch.barrier()
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()
                # Nothing above touches global memory.
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                buffer = Int64(parity)
                slot = (Int64(row) * HIDDEN + Int64(offset)) * 2
                a = self.load(alpha + offset, Float32)
                b = self.load(beta + offset, Float32)
                w = self.load(weight + offset, cutlass.BFloat16)
                r = self.load(residual + row * 8192 + offset, Float32)

                # Poll every slot of the local buffer, this rank's own included, until no
                # half of any word is the sentinel. In one call a slot has exactly one
                # writer (its source rank's producer, one store per element), so a half
                # that does not hold the sentinel holds its final value.
                receive_address = receive.toint()
                values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                for p in cutlass.range_constexpr(self.ranks):
                    address = receive_address + (buffer * MAX_RANKS + p) * SLAB_BYTES + slot
                    v0, v1, v2, v3 = _poll_x4(
                        address, Uint32(1), Uint32(0), Uint32(0), Uint32(0), Uint32(0)
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
                start = Int64(0)
                if cutlass.const_expr(self.guard_ns > 0):
                    start = _globaltimer()
                while stale != Uint32(0):
                    polled = stale
                    for p in cutlass.range_constexpr(self.ranks):
                        address = receive_address + (buffer * MAX_RANKS + p) * SLAB_BYTES + slot
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
                        stale = stale | _direct_stale_bit(
                            values[p, 0],
                            values[p, 1],
                            values[p, 2],
                            values[p, 3],
                            Uint32(1 << p),
                        )
                    if cutlass.const_expr(self.guard_ns > 0):
                        if stale != Uint32(0):
                            if _globaltimer() - start > Int64(self.guard_ns):
                                _set_guard_word(control_address + GUARD_WORD * 4)
                                if cutlass.const_expr(self.guard_break):
                                    stale = Uint32(0)
                # Retire every slot of this call before this grid can complete.
                for p in cutlass.range_constexpr(self.ranks):
                    _direct_reset_x4(receive_address + (buffer * MAX_RANKS + p) * SLAB_BYTES + slot)
                # Early trigger: past the last transport access (the hop's early-trigger placement).
                if cutlass.const_expr(self.pdl and self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

                # The built-in one-shot sum: FP32 from +0, ranks in order, one RN round.
                # A -0 partial now arrives as -0 (nothing clears it); from +0 a
                # round-to-nearest sum never becomes -0, so -0 and +0 add the same.
                totals = [Float32(0.0)] * VECTOR
                for p in cutlass.range_constexpr(self.ranks):
                    for k in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(values[p, k])
                        totals[2 * k] = totals[2 * k] + low
                        totals[2 * k + 1] = totals[2 * k + 1] + high
                x = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                for v in cutlass.range_constexpr(8):
                    x[v, 0] = cutlass.BFloat16(totals[v])

                # From here on: _PostAttnHopKernel's boundary, statement for statement.
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
                    cute.make_tensor((output + row * 8192 + offset).align(16), layout),
                )
                cute.copy(
                    atom16,
                    normed,
                    cute.make_tensor((next_output + row * 8192 + offset).align(16), layout),
                )
                # Late trigger: after every store, the boundary kernel's own placement.
                if cutlass.const_expr(self.pdl and not self.early_trigger):
                    cute.arch.griddepcontrol_launch_dependents()

    _CONSUMER_POINTER_DTYPES = (
        Float32,  # residual
        Float32,  # alpha
        Float32,  # beta
        BFloat16,  # weight
        Float32,  # output
        BFloat16,  # next_output
        Uint32,  # control
        Uint32,  # receive
    )

    @functools.lru_cache(maxsize=64)
    def _compiled_consumer(
        device_index: int,
        ranks: int,
        cluster: int,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        pdl: bool,
        early_trigger: bool,
        guard_ns: int,
        guard_break: bool,
    ):
        kernel = _PostAttnConsumerKernel(
            ranks, cluster, post_eps, pre_eps, norm_fp32, pdl, early_trigger, guard_ns, guard_break
        )
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *[
                    make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
                    for dtype in _CONSUMER_POINTER_DTYPES
                ],
                Int32(1),
                Int32(0),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


class PostAttnDirectWorkspace:
    """The direct transport of the hop (``attention_output_into_hop``) for one tensor-parallel group.

    The layout is ``PostAttnHopWorkspace``'s -- ``[2 buffers][8 source ranks][8 rows][8192]``
    BF16, rank-major -- so this rank's slot of buffer ``p`` is one contiguous ``[rows, 8192]``
    block; ``slot`` hands it out as a tensor on the multicast address. A GEMM given that
    tensor as its output publishes the partial into every rank's copy with its own epilogue
    stores; ``launch`` then runs the consumer hop on the local copy. A fabric allocation of
    its own, filled with ``DIRECT_SENTINEL``: a ``PostAttnHopWorkspace`` call never reads or
    resets its own slot, a consumer call reads and resets all eight, so the two cannot share.

    The buffer index is the caller's: a GEMM cannot read a device counter. The two-buffer
    argument of the module docstring holds when consecutive calls on this workspace
    alternate between the buffers, identically on every rank. Rank ``A``'s producer of call
    ``n + 2`` stores into buffer ``p`` only from its epilogue, after its dependency wait,
    so after ``A``'s call ``n + 1`` completed, which needed every rank's call-``n + 1``
    publication; rank ``B``'s producer of call ``n + 1`` stored only after ``B``'s call
    ``n`` -- whose resets of buffer ``p``, its own slot included, are part of it --
    completed. The target gives layer ``l`` the parity ``l mod 2``, runs this hop on every
    layer of every step of the one row count it serves, and refuses a layer count that is
    odd, so the sequence alternates across steps as well. Like the built-in all-reduce,
    this needs every kernel launched with programmatic dependent launch between two calls
    to wait on its predecessor before it completes, and the producer to store each output
    element once (no reduction through its output).

    Construction is collective over the group, like ``PostAttnHopWorkspace``'s, and needs
    the multicast transport. ``guard_ns`` is the poll guard (``DIRECT_GUARD_NS``; 0 = none);
    ``guard_break`` makes an expired guard stop polling (tests of a broken sequence only:
    the call then returns values of slots that had not arrived). The guard word is in
    ``control``, pinned host memory the kernel writes through its mapped address, so a
    daemon thread reads it every ``watchdog_s`` seconds without waiting for the device and
    logs an error once it is set (0 = no thread). ``reset`` and ``check_producer`` are the
    collective recovery and the producer check (module docstring).
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        variants: Sequence[Tuple[float, float, bool]] = (),
        early_trigger: bool = False,
        guard_ns: int = DIRECT_GUARD_NS,
        guard_break: bool = False,
        watchdog_s: float = DIRECT_WATCHDOG_S,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.pdl = _LAUNCH_PDL
        self.early_trigger = bool(early_trigger)
        self.guard_ns = int(guard_ns)
        self.guard_break = bool(guard_break)
        if self.ranks not in (4, MAX_RANKS):
            raise ValueError(f"the direct hop serves 4 or 8 ranks, got {self.ranks}")
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        #: The device's compute capability, which selects ``check_producer``'s validated set.
        self.capability: Optional[Tuple[int, int]] = None
        self._kernels = {}
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("CuTe DSL is unavailable")
            self.capability = tuple(torch.cuda.get_device_capability(self.device))
            if self.capability not in ((10, 3), (10, 7)):
                raise RuntimeError("the direct hop runs on SM103 or SM107 only")
            use_fabric = mapping.is_multi_node()
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, WORKSPACE_BYTES, mapping, use_fabric
            )
            words = WORKSPACE_BYTES // 4
            self._receive = self._handle.get_uc_buffer(self.rank, (words,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            self._slots = self._mc.view(torch.bfloat16).view(BUFFERS, MAX_RANKS, MAX_ROWS, HIDDEN)
            with torch.cuda.device(self.device):
                self._receive.fill_(_DIRECT_SENTINEL_I32)
                # Pinned host memory: the kernel's guard store reaches the host directly.
                self.control = torch.zeros(
                    CONTROL_WORDS, dtype=torch.int32, pin_memory=prefer_pinned()
                )
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                f"post-attention direct hop workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        self.build_variants(variants)
        if watchdog_s > 0 and self.guard_ns > 0:
            threading.Thread(
                target=_watch_guard,
                args=(weakref.ref(self), float(watchdog_s)),
                name=f"musespark_open.direct_publish_watchdog-r{self.rank}",
                daemon=True,
            ).start()

    def build_variants(self, variants: Sequence[Tuple[float, float, bool]]) -> None:
        """Compile and load both splits of each ``(post_eps, pre_eps, norm_fp32)`` not built yet.

        Collective: either every rank of the group has them afterwards or every rank raises.
        """
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        error: Optional[BaseException] = None
        try:
            for post_eps, pre_eps, norm_fp32 in variants:
                for cluster in (16, 8):
                    self._kernel(cluster, post_eps, pre_eps, norm_fp32)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "post-attention direct hop kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _kernel(self, cluster: int, post_eps: float, pre_eps: float, norm_fp32: bool):
        key = (cluster, float(post_eps), float(pre_eps), bool(norm_fp32))
        compiled = self._kernels.get(key)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"post-attention direct hop variant {key} was not built before graph capture"
            )
        compiled = _compiled_consumer(
            self.device.index,
            self.ranks,
            cluster,
            float(post_eps),
            float(pre_eps),
            bool(norm_fp32),
            self.pdl,
            self.early_trigger,
            self.guard_ns,
            self.guard_break,
        )
        # A dry launch loads the module now, not inside a capture.
        dummy = self.control.data_ptr()
        compiled(*([dummy] * 6), dummy, self._receive.data_ptr(), 1, 0, 1, self._stream())
        self._kernels[key] = compiled
        return compiled

    def _stream(self):
        return cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def slot(self, parity: int, rows: int) -> torch.Tensor:
        """This rank's ``[rows, 8192]`` BF16 slot of buffer ``parity``, on the multicast address.

        Write-only: a store through it reaches every rank's copy; nothing may read it.
        """
        if parity not in (0, 1) or not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"no slot for parity {parity}, {rows} rows")
        return self._slots[parity, self.rank, :rows]

    def guard_set(self) -> bool:
        """Whether a poll of this workspace has outlasted ``guard_ns`` so far (no device wait)."""
        return int(self.control[GUARD_WORD]) != 0

    def guard_fired(self) -> bool:
        """Whether a poll of this workspace ever outlasted ``guard_ns`` (synchronizes)."""
        torch.cuda.synchronize(self.device)
        return self.guard_set()

    def reset(self) -> None:
        """Every slot of both buffers of every rank's copy back to the sentinel. Collective.

        For a call sequence that broke off between a producer and its consumer: every rank
        waits for its device (each store its producers issued has landed), the group
        agrees, every rank refills its local copy and waits again, and the group agrees
        again, so no rank's next producer stores into a copy its owner has not refilled.

        Raises:
            RuntimeError: on every rank, if any rank failed a step.
        """
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        for step in ("wait", "refill"):
            error: Optional[BaseException] = None
            try:
                with torch.cuda.device(self.device):
                    if step == "refill":
                        self._receive.fill_(_DIRECT_SENTINEL_I32)
                    torch.cuda.synchronize(self.device)
            except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
                error = exc
            if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
                raise RuntimeError(
                    f"post-attention direct hop transport reset ({step}) failed on at least one "
                    f"rank{f' (this rank: {error!r})' if error is not None else ''}"
                ) from error

    def check_producer(
        self, produce: Callable[[], None], warm: Optional[Callable[[], None]] = None
    ) -> str:
        """The kernel ``produce`` publishes with, if it is a validated producer. Collective.

        ``produce`` stores into one of this workspace's slots; it is captured into a CUDA
        graph that is never replayed (``captured_nodes``), so nothing is published. The
        capture must hold exactly one node, a kernel of the device's ``VALIDATED_PRODUCERS``,
        on every rank. Returns its name.

        Raises:
            ValueError: on every rank, naming this rank's nodes, if any rank's capture is not.
        """
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        names: List[str] = []
        error: Optional[BaseException] = None
        try:
            names = captured_nodes(produce, self.device, warm)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        validated = validated_producers(self.capability)
        ok = error is None and len(names) == 1 and validated_producer(names[0], self.capability)
        if not _mnnvl_workspace_all_succeeded(self._comm, ok):
            raise ValueError(
                "[musespark_open] attention_output_into_hop: the published o_proj must be exactly one kernel "
                "validated as a "
                f"direct-transport producer on SM {self.capability} ({sorted(validated)}) on every "
                "rank; "
                f"rank {self.rank} captured {names}"
                + (f" ({error!r})" if error is not None else "")
            )
        log_once_producer(names[0])
        return names[0]

    def launch(
        self,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        parity: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual.device != self.device:
            raise ValueError(
                f"post_attn_consumer_hop inputs are on {residual.device}, "
                f"its workspace on {self.device}"
            )
        rows = residual.shape[0]
        compiled = self._kernel(cluster_for_rows(rows), post_eps, pre_eps, norm_fp32)
        updated = torch.empty_like(residual)
        normalized = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
        compiled(
            residual.data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            next_weight.data_ptr(),
            updated.data_ptr(),
            normalized.data_ptr(),
            self.control.data_ptr(),
            self._receive.data_ptr(),
            rows,
            int(parity),
            0,
            self._stream(),
        )
        return updated, normalized


def publish_copy(slot: torch.Tensor, partial: torch.Tensor) -> None:
    """Store ``partial``'s bits into a direct-transport ``slot`` from an elementwise kernel.

    For a producer that wrote its output elsewhere. Not ``Tensor.copy_``: a contiguous
    copy runs on the copy engine, which is not an SM store to the multicast address. The
    kernel is an integer ``x | 0``, so every bit pattern (``-0`` included) is kept.
    """
    torch.bitwise_or(partial.reshape(slot.shape).view(torch.int16), 0, out=slot.view(torch.int16))


_SHARED_DIRECT: dict = {}


def shared_direct_workspace(
    mapping: Mapping,
    *,
    variants: Sequence[Tuple[float, float, bool]] = (),
    early_trigger: bool = False,
) -> PostAttnDirectWorkspace:
    """The one direct workspace of this group and trigger placement, built on first use.

    Collective, like the constructor; a later call gets the same workspace back, with any
    variant it names built as well.
    """
    key = (mapping, bool(early_trigger), _LAUNCH_PDL)
    workspace = _SHARED_DIRECT.get(key)
    if workspace is None:
        workspace = PostAttnDirectWorkspace(mapping, variants=variants, early_trigger=early_trigger)
        _SHARED_DIRECT[key] = workspace
    else:
        workspace.build_variants(variants)
    return workspace


def consumer_hop_supported(
    published: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
    parity: int,
    workspace: PostAttnDirectWorkspace,
) -> bool:
    """``hop_supported`` for the consumer: ``published`` must be ``workspace.slot(parity, rows)``."""
    rows = published.shape[0] if published.ndim == 2 else 0
    return (
        hop_takes(published, residual, alpha, beta, next_weight)
        and hop_layout_ok(residual, alpha, beta, next_weight)
        and parity in (0, 1)
        and published.data_ptr() == workspace.slot(parity, rows).data_ptr()
    )


class PostAttnConsumerHop(OpWrapper):
    """``post_attn_hop`` on the direct transport, whose partial its producer already published.

    ``published`` is this rank's slot of buffer ``parity`` (``workspace.slot(parity, rows)``),
    which the producer -- a GEMM handed it as its output -- has written on every rank. The op
    checks its address and never reads it. Returns ``(updated, ffn_input)``, the values
    ``post_attn_hop`` returns for the same partials.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="no speculation: one row, the 16-CTA cluster split", spec=dict(rows=1)),
        Cell(why="two rows: the smallest 8-CTA split", spec=dict(rows=2)),
        Cell(why="K = 3 verify width", spec=dict(rows=4)),
        Cell(why="K = 5 verify width", spec=dict(rows=6)),
        Cell(why="K = 7 verify width, the one row count the target serves", spec=dict(rows=8)),
        Cell(
            why="the second buffer: odd layers publish into parity 1",
            spec=dict(rows=8, parity=1),
        ),
        Cell(
            why="the FP32-residual prenorm ordering (MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL)",
            spec=dict(rows=8, norm_fp32=True),
        ),
        Cell(
            why="exact zeros and negative zeros published unchanged (0x80000000 is now a payload)",
            spec=dict(rows=8, zeros=True),
        ),
        Cell(
            why="cancelling partials: the FP32 rank-order sum differs from other orders",
            spec=dict(rows=8, cancel=True),
        ),
    )

    note = """
    Metadata consumed: none. Call site: once per decoder layer per decode step of
    the row count the target serves directly (8 at K = 7), from the target's
    attention-side residual join, after the layer's o_proj GEMM wrote its output
    into ``published``; layer ``l`` uses parity ``l mod 2``.

    Bit-identical to ``post_attn_hop`` on the same partials, gated by its own
    multi-GPU test (both outputs, every cell, a real cuBLAS o_proj as the
    producer, graph replay with changing inputs, a delayed rank, a long burst,
    a concurrent poller that records the first non-sentinel value of every
    word). Ranks 4 and 8. The guard word turns a broken call sequence into a
    flag instead of only a hang (tested with ``guard_break``), and the
    workspace's watchdog logs it while the poll still waits. A sequence broken
    between a producer and this op is cleared by the workspace's collective
    ``reset`` (tested after an aborted call).

    Loud preconditions: `consumer_hop_supported` must hold. Not certified:
    producers that reduce through their output (split-K through global memory),
    rows above 8, widths other than 8192; ``check_producer`` accepts only the device's
    ``VALIDATED_PRODUCERS``.
    """

    def __call__(
        self,
        published: torch.Tensor,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        parity: int,
        workspace: PostAttnDirectWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not consumer_hop_supported(
            published, residual, alpha, beta, next_weight, parity, workspace
        ):
            raise ValueError(
                "post_attn_consumer_hop takes this rank's [1..8, 8192] slot of buffer "
                "`parity` of its workspace as `published`, an FP32 residual of the same "
                "shape, FP32 alpha/beta and a BF16 weight of 8192, 16-byte aligned; got "
                f"{published.dtype} {tuple(published.shape)} / {residual.dtype} "
                f"{tuple(residual.shape)}, parity {parity}"
            )
        return workspace.launch(
            residual, alpha, beta, next_weight, post_eps, pre_eps, norm_fp32, parity
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
        parity: int,
        workspace: Optional[PostAttnDirectWorkspace],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``post_attn_hop``'s reference: the group's partials in ascending rank order.

        The documented collective exception, as there. ``parity`` and ``workspace`` are
        ignored.
        """
        summed = torch.zeros_like(per_rank_partials[0], dtype=torch.float32)
        for part in per_rank_partials:
            summed = summed + part.float()
        x = summed.to(torch.bfloat16).double()
        normed = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + post_eps)).to(torch.bfloat16)
        updated = alpha.double() * residual.double() + beta.double() * normed.double()
        updated32 = updated.to(torch.float32)
        prenorm = updated if norm_fp32 else updated32.to(torch.bfloat16).double()
        out = (
            prenorm
            * torch.rsqrt(prenorm.square().mean(-1, keepdim=True) + pre_eps)
            * (next_weight.double() + 1.0)
        )
        return updated32, out.to(torch.bfloat16)


post_attn_consumer_hop = PostAttnConsumerHop()


def log_once_direct(
    layers: int, rows: int, early_trigger: bool = False, prefetch_dropped: int = 0
) -> None:
    logger.info(
        "[musespark_open] attention_output_into_hop: o_proj publishes into the post-attention hop's direct "
        "transport and the "
        f"hop only consumes, on the decode steps of {rows} rows, all {layers} layers "
        f"(parity = layer mod 2; {'early' if early_trigger else 'late'} dependent-launch "
        f"trigger; the o_proj -> pool-weight prefetch dropped at {rows} rows on "
        f"{prefetch_dropped} layers)"
    )


def log_once_producer(name: str) -> None:
    logger.info(
        f"[musespark_open] attention_output_into_hop: the published o_proj captures {name}, a validated producer"
    )


class PostAttnHop(OpWrapper):
    """All-reduce the o_proj partial, join it into the residual, apply the pre-FFN norm.

    Returns ``(updated, ffn_input)``: the new FP32 residual
    ``alpha * residual + beta * bf16(rmsnorm(x))`` and the BF16
    ``bf16(rmsnorm(prenorm_input) * (1 + next_weight))``, where ``x`` is the
    BF16 sum of every rank's ``partial`` and ``prenorm_input`` is
    ``bf16(updated)`` (or ``updated`` itself with ``norm_fp32``).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The decode step of this target at K = 7 carries 8 rows; K = 3 / 5 carry 4 / 6;
    # one row is the no-speculation step and the only shape on the 16-CTA split.
    CELLS: tuple[Cell, ...] = (
        Cell(why="no speculation: one row, the 16-CTA cluster split", spec=dict(rows=1)),
        Cell(why="two rows: the smallest 8-CTA split", spec=dict(rows=2)),
        Cell(why="K = 3 verify width", spec=dict(rows=4)),
        Cell(why="K = 5 verify width", spec=dict(rows=6)),
        Cell(why="K = 7 verify width, the measured configuration", spec=dict(rows=8)),
        Cell(
            why="the FP32-residual prenorm ordering (MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL)",
            spec=dict(rows=8, norm_fp32=True),
        ),
        Cell(
            why="exact zeros and negative zeros in every rank's partial",
            spec=dict(rows=8, zeros=True),
        ),
        Cell(
            why="cancelling partials: the FP32 rank-order sum differs from other orders",
            spec=dict(rows=8, cancel=True),
        ),
    )

    note = """
    Metadata consumed: none. Call site: once per decoder layer per decode step
    of at most 8 rows, from the target's attention-side residual join, in place
    of the built-in MNNVL one-shot all-reduce and cluster boundary.

    Bit-identical to that pair, gated by its own multi-GPU test (both outputs,
    every cell, both transport variants, graph replay with changing inputs and
    shapes, a delayed rank, a long burst, the arrival counter's accounting
    across its wrap). Ranks 4 (one node) and 8 (two nodes); the slot layout is
    sized for 8. The cancelling cell makes the all-reduce's summation order
    observable, so a change of order fails the bitwise gate. The two-buffer
    alternation of the transport is not observable by any test (see the module
    docstring); it rests on the written argument.

    Loud preconditions: `hop_supported` must hold. A caller choosing between
    this collective and the built-in all-reduce decides on `hop_takes` alone,
    which every rank answers alike, and makes the layout fit with `readable`.
    Not certified: rows above 8, widths other than 8192, a weighted
    attention-side post norm.
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
        workspace: PostAttnHopWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hop_supported(partial, residual, alpha, beta, next_weight):
            raise ValueError(
                "post_attn_hop takes a contiguous BF16 [1..8, 8192] partial, an FP32 residual "
                "of the same shape, FP32 alpha/beta and a BF16 weight of 8192, 16-byte aligned; "
                f"got {partial.dtype} {tuple(partial.shape)} / {residual.dtype} "
                f"{tuple(residual.shape)}"
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
        workspace: Optional[PostAttnHopWorkspace],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The op's rounding points, float64 in between.

        The documented collective exception: takes the group's partials in
        ascending rank order instead of this rank's ``partial``. The all-reduce
        is the op's contract, which the built-in one-shot defines: an FP32 sum
        from ``+0`` in ascending rank order, rounded once to BF16 (the value the
        built-in boundary reads). The post norm is rounded to BF16 before the
        gated add; ``updated`` is rounded once to FP32; the prenorm input is
        rounded to BF16 unless ``norm_fp32``; the output is rounded once to BF16.
        Everything after the all-reduce is float64. ``workspace`` is ignored.
        """
        summed = torch.zeros_like(per_rank_partials[0], dtype=torch.float32)
        for part in per_rank_partials:
            summed = summed + part.float()
        x = summed.to(torch.bfloat16).double()
        normed = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + post_eps)).to(torch.bfloat16)
        updated = alpha.double() * residual.double() + beta.double() * normed.double()
        updated32 = updated.to(torch.float32)
        prenorm = updated if norm_fp32 else updated32.to(torch.bfloat16).double()
        out = (
            prenorm
            * torch.rsqrt(prenorm.square().mean(-1, keepdim=True) + pre_eps)
            * (next_weight.double() + 1.0)
        )
        return updated32, out.to(torch.bfloat16)


post_attn_hop = PostAttnHop()


def log_once_enabled(
    layers: int, multicast: bool, early_trigger: bool = False, stale_poll: bool = False
) -> None:
    logger.info(
        "[musespark_open] post_attention_hop: fused post-attention hop (one-shot AR + residual join + pre-FFN norm) "
        "on the "
        f"decode path of all {layers} layers (steps of at most {MAX_ROWS} rows, "
        f"{'multicast' if multicast else 'unicast'} publication, "
        f"{'early' if early_trigger else 'late'} dependent-launch trigger"
        f"{', stale-only polling from %d rows' % STALE_POLL_MIN_ROWS if stale_poll else ''})"
    )
