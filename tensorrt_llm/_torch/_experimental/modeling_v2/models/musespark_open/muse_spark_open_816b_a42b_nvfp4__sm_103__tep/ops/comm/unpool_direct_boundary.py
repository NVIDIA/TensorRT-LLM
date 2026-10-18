# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The post-FFN boundary on a transport the unpool GEMM itself publishes into (``unpool_into_boundary``).

``unpool_gather_boundary`` (the gather-transport boundary) loads this rank's ``[M, 1024]`` unpool
slice from the GEMM's output, publishes it to the peers as tagged 64-bit words,
then polls one peer's words per thread. Here the unpool GEMM's output tensor
*is* this rank's slot of a multicast transport: the GEMM's own epilogue stores
reach every rank's copy, so the boundary kernel only polls, resets and
computes. Off the critical path: the dependency-wait release followed by
the reload of the local slice and the publication.

Transport (Lamport, payload as flag). Every rank holds ``3 buffers x 8 source
ranks x 8 rows x 1024`` BF16 -- rank-major, so rank ``r``'s slot of buffer
``p`` is one contiguous ``[rows, 1024]`` block, a GEMM output as it is -- filled
with the sentinel word ``0xFFFEFFFE``: two BF16 NaNs with payload ``0x7E``.
Float-to-BF16 rounding (``cvt.rn.bf16{x2}.f32``, the GEMM epilogues') turns
every NaN into the canonical ``0x7FFF``, so no GEMM output contains the
half ``0xFFFE``; a 16-bit half that still holds it has not arrived. The test is
per half, not per word or per vector: a store that reached the poller in
pieces still reads as not arrived until every piece did. Each thread of the
boundary covers the same 8 columns of the same row as in ``unpool_gather_boundary`` -- columns of one
peer's slice -- polls those 16 bytes of its local copy until no half is the
sentinel, writes the sentinel back, and runs the gather boundary's arithmetic statement for
statement (``unpack_bf16x2`` is exact). Own slot included: the producer's
multicast stores reach the local copy asynchronously like the peers'.

Ownership and reuse (cuFlow):

* One writer per element per call: rank ``r``'s unpool GEMM stores each element
  of slot ``r`` once. flashinfer's split-K kernel reduces the K slices in the
  cluster's first CTA through DSMEM and stores the rounded tile once from that
  CTA; the direct SIMT kernel stores once too. A cuBLAS kernel never writes the
  slot: the unpool site's producer policy (``unpool_direct.PUBLISH_POLICY``)
  lets only flashinfer's kernels store there, and every other product reaches
  the slot through ``publish_copy``, once per element.
* One reader and resetter per element per call: the thread of each rank's
  boundary that owns those columns, on its own copy.
* Reuse: no two consecutive calls on the workspace use the same buffer, the
  last call of a step and the first of the next included (``buffer_sequence``:
  the buffers alternate call by call and the last call of an odd count takes
  the third). Buffer ``p`` of rank ``B``'s copy is next written by a producer of
  a later call ``m`` on that buffer; rank ``A``'s producer stores only from its
  epilogue, after its dependency wait, so after ``A``'s call ``m - 1``
  completed, which needed ``B``'s call-``(m - 1)`` slot, which ``B``'s producer
  stored only after every earlier call of ``B`` -- among them ``B``'s last call
  on buffer ``p``, call ``m - 2`` or earlier, whose resets are part of it --
  completed. The chain runs through the workspace's own calls only: no other
  collective needs to sit between two calls. As for the built-in MNNVL
  all-reduce, every kernel launched with programmatic dependent launch between
  two calls must wait on its predecessor before it completes.
* Progress: every producer stores without waiting on anything remote; every
  boundary thread waits only for one 16-byte piece of one slot. A poll that
  has seen no arrival for ``guard_ns`` sets a guard word in host memory (and
  keeps polling): a call sequence that differs between ranks, or a peer that
  stopped, shows up there, and ``start_guard_watch`` logs it from the host.

PDL memory order: before ``griddepcontrol.wait`` the kernel touches shared
memory and the cluster barrier only (and, with ``trigger="entry"``, releases
its dependents first). After it: the coefficients and weights (immutable),
the residual (written by kernels the wait orders) and the transport (ordered
by the sentinel test, not by the wait). The capture store (``capture_fold``) and the
outputs come last; the trigger placements are ``unpool_gather_boundary``'s.

Calls on one workspace must be issued in the same order on every rank, with
the same row counts and buffers, serialized on one stream (graph replay
included). The workspace is this op's alone: the gather boundary's tagged inbox, the built-in
gather and every other collective use their own memory.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Optional, Sequence, Tuple

import torch
from cuda.bindings import runtime as _cudart

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from . import unpool_gather_boundary as _f1
from .projection_inbox import HAS_CUTE_DSL, MAX_ROWS, WORLD

#: Width of the residual stream and of one rank's unpool slice.
HIDDEN = _f1.HIDDEN
LOCAL_COLUMNS = _f1.LOCAL_COLUMNS
TRIGGERS = _f1.TRIGGERS
#: Buffers of the transport: two that alternate call by call, and a third for the last
#: call of an odd count, so that no two consecutive calls share one (``buffer_sequence``).
BUFFERS = 3
#: The empty-slot word: two BF16 NaNs with the payload no float-to-BF16 rounding produces.
DIRECT_SENTINEL = 0xFFFEFFFE
DIRECT_SENTINEL_HALF = 0xFFFE
_SENTINEL_I32 = DIRECT_SENTINEL - (1 << 32)
#: Bytes of one rank's slot (``MAX_ROWS`` rows) and of the whole transport.
SLAB_BYTES = MAX_ROWS * LOCAL_COLUMNS * 2
WORKSPACE_BYTES = BUFFERS * WORLD * SLAB_BYTES
#: Per-rank control words, in pinned host memory the kernel stores to at system scope:
#: ``GUARD_WORD`` is set by a poll that outlived the guard.
CONTROL_WORDS = 4
GUARD_WORD = 0
#: Default poll guard in ns (far above any wait the protocol produces itself).
DIRECT_GUARD_NS = 1 << 33
#: Seconds between two reads of the guard word by ``start_guard_watch``'s thread.
GUARD_WATCH_S = 1.0
#: 32-bit payload words of one row of one rank's slice, and threads per peer slice.
_ROW_WORDS = LOCAL_COLUMNS // 2
_PER_PEER = _ROW_WORDS // 4

_AVAILABLE = HAS_CUTE_DSL and _f1.available()

if _AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64, Uint32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    from .projection_inbox import _asm, _i64, _u32, unpack_bf16x2

    _FOUR_WORDS = "!llvm.struct<(i32,i32,i32,i32)>"

    @dsl_user_op
    def _poll_x4(address, *, loc=None, ip=None):
        """Four 32-bit words (16 bytes) of the local copy, relaxed at system scope."""
        result = _asm(
            ir.Type.parse(_FOUR_WORDS),
            [_i64(address, loc, ip)],
            "ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            loc,
            ip,
        )
        return tuple(
            Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4)
        )

    @dsl_user_op
    def _pending(w0, w1, w2, w3, *, loc=None, ip=None):
        """1 while any 16-bit half of the four words is ``DIRECT_SENTINEL_HALF``, else 0."""
        text = "{\n.reg .pred p;\n.reg .b32 x, h, s;\nmov.b32 s, 0;\n"
        for operand in range(1, 5):
            text += (
                f"xor.b32 x, ${operand}, 0xFFFEFFFE;\n"
                "and.b32 h, x, 0xFFFF;\nsetp.eq.u32 p, h, 0;\n@p mov.b32 s, 1;\n"
                "shr.b32 h, x, 16;\nsetp.eq.u32 p, h, 0;\n@p mov.b32 s, 1;\n"
            )
        text += "mov.b32 $0, s;\n}"
        result = _asm(
            T.i32(),
            [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            text,
            "=r,r,r,r,r",
            loc,
            ip,
            side_effects=False,
        )
        return Uint32(result)

    @dsl_user_op
    def _reset_x4(address, *, loc=None, ip=None):
        """Write the sentinel back into 16 consumed bytes of the local copy."""
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
        """The global nanosecond timer (side-effecting: every read is a fresh one)."""
        return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))

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

    class _UnpoolDirectBoundaryKernel(_f1._UnpoolGatherBoundaryKernel):
        """The gather boundary on the direct transport: poll + reset, then its arithmetic.

        ``unpool_gather_boundary``'s geometry, thread-to-column map, trigger placements and
        (with ``capture``) its DFlash capture store, without the per-row tag state, the load
        of the local slice and its publication.
        """

        def __init__(
            self,
            m: int,
            gain: bool,
            post_eps: float,
            pre_eps: float,
            norm_fp32: bool,
            trigger: int,
            capture: bool,
            guard_ns: int,
            guard_break: bool,
        ):
            super().__init__(m, gain, post_eps, pre_eps, norm_fp32, 0, trigger)
            self.capture = bool(capture)
            self.guard_ns = int(guard_ns)
            self.guard_break = bool(guard_break)

        @cute.jit
        def __call__(
            self,
            receive: cute.Pointer,
            control: cute.Pointer,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            capture: cute.Pointer,
            capture_stride: Int64,
            parity: Int32,
            dry: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                receive,
                control,
                residual,
                gain,
                alpha,
                beta,
                weight,
                output,
                next_output,
                capture,
                capture_stride,
                parity,
                dry,
            ).launch(
                grid=[self.m * self.cluster, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 2 + 16,
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            receive: cute.Pointer,
            control: cute.Pointer,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            capture: cute.Pointer,
            capture_stride: Int64,
            parity: Int32,
            dry: Int32,
        ):
            # A dry launch only makes the driver load the module, outside any capture.
            if dry == 0:
                if cutlass.const_expr(self.trigger == TRIGGERS["entry"]):
                    cute.arch.griddepcontrol_launch_dependents()
                tid = cute.arch.thread_idx()[0]
                rank = cute.arch.block_idx()[0] % self.cluster
                row = cute.arch.block_idx()[0] // self.cluster
                offset = cute.assume((rank * self.threads + tid) * 8, divby=8)
                smem = utils.SmemAllocator()
                partials = smem.allocate_tensor(
                    Float32, cute.make_layout((self.slots * 4 * 2,)), byte_alignment=16
                )
                bars = smem.allocate_tensor(
                    cutlass.Uint64, cute.make_layout((2,)), byte_alignment=8
                )
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
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.trigger == TRIGGERS["after_wait"]):
                    cute.arch.griddepcontrol_launch_dependents()

                # Thread (rank, tid) covers columns [offset, offset + 8): words
                # [word, word + 4) of peer ``peer``'s slice, as in the gather boundary.
                flat = rank * self.threads + tid
                peer = flat // _PER_PEER
                word = (flat % _PER_PEER) * 4

                a = self.load(alpha + offset, Float32)
                b = self.load(beta + offset, Float32)
                w = self.load(weight + offset, cutlass.BFloat16)
                g = self.load(gain + offset, cutlass.BFloat16)
                r = self.load(residual + row * HIDDEN + offset, Float32)

                # This thread's 16 bytes of peer ``peer``'s slot in buffer ``parity``
                # of the local copy. In one call they have exactly one writer (that
                # peer's producer, one store per element), so a half that does not
                # hold the sentinel holds its final value.
                box = (
                    receive.toint()
                    + (
                        ((Int64(parity) * WORLD + Int64(peer)) * MAX_ROWS + Int64(row)) * _ROW_WORDS
                        + Int64(word)
                    )
                    * 4
                )
                w0, w1, w2, w3 = _poll_x4(box)
                pending = _pending(w0, w1, w2, w3)
                start = Int64(0)
                if cutlass.const_expr(self.guard_ns > 0):
                    start = _globaltimer()
                while pending != Uint32(0):
                    w0, w1, w2, w3 = _poll_x4(box)
                    pending = _pending(w0, w1, w2, w3)
                    if cutlass.const_expr(self.guard_ns > 0):
                        if pending != Uint32(0):
                            if _globaltimer() - start > Int64(self.guard_ns):
                                _set_guard_word(control.toint() + GUARD_WORD * 4)
                                if cutlass.const_expr(self.guard_break):
                                    pending = Uint32(0)
                # Retire the consumed bytes before this grid can complete.
                _reset_x4(box)
                x = []
                for packed in (w0, w1, w2, w3):
                    low, high = unpack_bf16x2(packed)
                    x.append(low)
                    x.append(high)

                # From here on: the built-in boundary's arithmetic, statement for
                # statement, with x[v] in place of Float32(branch[v, 0]).
                scaled = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                updated = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                normed = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                total = Float32(0)
                for p in cutlass.range_constexpr(self.iters):
                    for v in cutlass.range_constexpr(8):
                        value = x[v]
                        if cutlass.const_expr(self.gain):
                            value = Float32(cutlass.BFloat16(value * Float32(g[v, p])))
                        scaled[v, p] = value
                        total += value * value
                total = self.reduce(total, partials, bars, 0)
                inv = cute.math.rsqrt(
                    total / Float32(HIDDEN) + Float32(self.post_eps), fastmath=True
                )
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
                inv = cute.math.rsqrt(
                    total / Float32(HIDDEN) + Float32(self.pre_eps), fastmath=True
                )
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
                if cutlass.const_expr(self.capture):
                    # The DFlash capture: BF16(updated), rounded to nearest even as
                    # torch's float -> bfloat16 copy does.
                    captured = cute.make_rmem_tensor(
                        cute.make_layout((8, self.iters)), cutlass.BFloat16
                    )
                    for p in cutlass.range_constexpr(self.iters):
                        for v in cutlass.range_constexpr(8):
                            captured[v, p] = cutlass.BFloat16(updated[v, p])
                    cute.copy(
                        atom16,
                        captured,
                        cute.make_tensor(
                            (capture + Int64(row) * capture_stride + offset).align(16), layout
                        ),
                    )
                if cutlass.const_expr(self.cluster > 1):
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()
                if cutlass.const_expr(self.trigger == TRIGGERS["exit"]):
                    cute.arch.griddepcontrol_launch_dependents()

    _POINTER_TYPES = (
        Uint32,  # receive
        Uint32,  # control
        Float32,  # residual
        cutlass.BFloat16,  # gain
        Float32,  # alpha
        Float32,  # beta
        cutlass.BFloat16,  # weight
        Float32,  # output
        cutlass.BFloat16,  # next_output
        cutlass.BFloat16,  # capture
    )

    @functools.lru_cache(maxsize=64)
    def _compile(
        device_index: int,
        rows: int,
        has_gain: bool,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        trigger: int,
        capture: bool,
        guard_ns: int,
        guard_break: bool,
    ):
        kernel = _UnpoolDirectBoundaryKernel(
            rows, has_gain, post_eps, pre_eps, norm_fp32, trigger, capture, guard_ns, guard_break
        )
        pointers = [
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
            for dtype in _POINTER_TYPES
        ]
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *pointers,
                Int64(0),
                Int32(0),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


#: What selects a compiled boundary: ``(rows, gain, post_eps, pre_eps, norm_fp32, trigger, capture)``.
Variant = Tuple[int, bool, float, float, bool, str, bool]


def variant(
    rows: int,
    has_gain: bool,
    post_eps: float,
    pre_eps: float,
    norm_fp32: bool,
    trigger: str,
    capture: bool,
) -> Variant:
    """The workspace's key of the kernel a call with these settings launches."""
    if trigger not in TRIGGERS:
        raise ValueError(f"trigger must be one of {sorted(TRIGGERS)}, got {trigger!r}")
    return (
        int(rows),
        bool(has_gain),
        float(post_eps),
        float(pre_eps),
        bool(norm_fp32),
        str(trigger),
        bool(capture),
    )


def check_buffer_sequence(buffers: Sequence[int]) -> Tuple[int, ...]:
    """Check the buffers of one repeating cycle of calls (a decode step's fused layers).

    The cycle repeats back to back (graph replays), so the last call is followed by the
    first: no two neighbours, the wrap included, may share a buffer.

    Raises:
        ValueError: for an unknown buffer, fewer than two calls, or two neighbours on one
            buffer.
    """
    buffers = tuple(int(b) for b in buffers)
    if len(buffers) < 2 or any(b not in range(BUFFERS) for b in buffers):
        raise ValueError(
            f"a cycle of the direct unpool transport needs >= 2 calls on buffers "
            f"0..{BUFFERS - 1}, got {buffers}"
        )
    for index, buffer in enumerate(buffers):
        following = buffers[(index + 1) % len(buffers)]
        if buffer == following:
            raise ValueError(
                f"calls {index} and {(index + 1) % len(buffers)} of the cycle {buffers} both use "
                f"buffer {buffer}: a fast rank's later producer could overwrite a slot a slow "
                "peer has not consumed"
            )
    return buffers


def buffer_sequence(calls: int) -> Tuple[int, ...]:
    """The buffer of each call of a repeating cycle of ``calls`` calls (``calls >= 2``).

    ``0, 1, 0, 1, ...``; with an odd count the last call takes buffer 2, so the cycle's
    last call and the next cycle's first never share a buffer.
    """
    buffers = [index % 2 for index in range(calls)]
    if calls % 2:
        buffers[-1] = 2
    return check_buffer_sequence(buffers)


class UnpoolDirectWorkspace:
    """The direct transport of the post-FFN boundary (``unpool_into_boundary``) for one TP8 group.

    ``slot(buffer, rows)`` is this rank's ``[rows, 1024]`` BF16 slot of buffer
    ``buffer`` on the multicast address: a GEMM given it as its output publishes
    into every rank's copy. ``kernel(variant)`` is the compiled boundary for a
    call's settings. Construction is collective over the group (fabric
    allocation and sentinel fill), and so is ``build_variants``; both succeed on
    every rank or raise on every rank. Nothing compiles inside a CUDA graph
    capture: a variant not built before is an error there.

    ``guard_ns`` is the poll guard (0: none); ``guard_break`` makes an expired
    guard stop polling (tests of a broken call sequence only: the call then
    consumes slots that had not arrived). The guard word lives in pinned host
    memory, so the host reads it while the device still polls
    (``guard_fired(sync=False)``, ``start_guard_watch``).
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        guard_ns: int = DIRECT_GUARD_NS,
        guard_break: bool = False,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        if mapping.tp_size != WORLD or mapping.enable_attention_dp or mapping.cp_size != 1:
            raise ValueError("the direct unpool transport needs TP8 without attention DP or CP")
        self.rank = mapping.tp_rank
        self.guard_ns = int(guard_ns)
        self.guard_break = bool(guard_break)
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._kernels = {}
        self._watch: Optional[threading.Thread] = None
        self._comm = _get_mnnvl_workspace_comm(mapping)
        # The preconditions are agreed before the multicast allocation, which is
        # collective: a rank that refused alone would leave its peers in it.
        refusal: Optional[str] = None
        if not _AVAILABLE:
            refusal = "CuTe DSL or the built-in cluster boundary is unavailable"
        elif torch.cuda.get_device_capability(self.device) != (10, 3):
            refusal = "the direct unpool boundary is validated on SM103 only"
        elif not prefer_pinned():
            refusal = "the guard words need pinned host memory, which this platform does not pin"
        if not _mnnvl_workspace_all_succeeded(self._comm, refusal is None):
            raise RuntimeError(
                "direct unpool transport refused on at least one rank"
                f"{f' (this rank: {refusal})' if refusal is not None else ''}"
            )
        error: Optional[BaseException] = None
        try:
            use_fabric = mapping.is_multi_node()
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, WORKSPACE_BYTES, mapping, use_fabric
            )
            words = WORKSPACE_BYTES // 4
            self._receive = self._handle.get_uc_buffer(self.rank, (words,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            self._slots = self._mc.view(torch.bfloat16).view(
                BUFFERS, WORLD, MAX_ROWS, LOCAL_COLUMNS
            )
            with torch.cuda.device(self.device):
                self._receive.fill_(_SENTINEL_I32)
                # Pinned host memory is mapped into the device's address space at the
                # same address (unified addressing): the guard store reaches it at
                # system scope and the host reads it without a synchronization.
                self.control = torch.zeros(
                    CONTROL_WORDS, dtype=torch.int32, pin_memory=prefer_pinned()
                )
                status, pointer = _cudart.cudaHostGetDevicePointer(self.control.data_ptr(), 0)
                if status != _cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaHostGetDevicePointer of the guard words: {status}")
                self._control_ptr = int(pointer)
                # What a dry launch passes for every pointer (it dereferences none).
                self._dry = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=self.device)
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "direct unpool transport failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def slot(self, buffer: int, rows: int) -> torch.Tensor:
        """This rank's ``[rows, 1024]`` BF16 slot of ``buffer``, on the multicast address.

        Write-only: a store through it reaches every rank's copy; nothing may read it.
        """
        if buffer not in range(BUFFERS) or not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"no slot for buffer {buffer}, {rows} rows")
        return self._slots[buffer, self.rank, :rows]

    def receive(self) -> torch.Tensor:
        """This rank's copy of the whole transport, ``int32 [3 * 8 * 8 * 512]`` (tests)."""
        return self._receive

    def control_ptr(self) -> int:
        """The device address of the guard words (pinned host memory)."""
        return self._control_ptr

    def is_clean(self) -> bool:
        """Whether every word of this rank's copy holds the sentinel (synchronizes; tests)."""
        torch.cuda.synchronize(self.device)
        return bool((self._receive == _SENTINEL_I32).all().item())

    def reset(self) -> None:
        """Every word of this rank's copy back to the sentinel (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self._receive.fill_(_SENTINEL_I32)
        torch.cuda.synchronize(self.device)

    def guard_fired(self, sync: bool = True) -> bool:
        """Whether a poll of this workspace ever outlasted ``guard_ns``.

        ``sync``: synchronize the device first (tests). Without it the word is read as
        it stands in host memory, while kernels may still be polling.
        """
        if sync:
            torch.cuda.synchronize(self.device)
        return int(self.control[GUARD_WORD]) != 0

    def clear_guard(self) -> None:
        """Zero the guard words (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self.control.zero_()

    def start_guard_watch(self, interval_s: float = GUARD_WATCH_S, abort: bool = False) -> bool:
        """Read the guard word from a host thread every ``interval_s`` seconds (0: no thread).

        An expired guard means a poll waited ``guard_ns`` for a slot that did not come:
        the ranks' call sequences differ, or a peer stopped. The kernel keeps polling,
        so the device makes no progress and the host blocks at its next
        synchronization; the thread logs an error naming the rank at once, and with
        ``abort`` aborts the process instead of leaving it to the executor's hang
        detector. Idempotent; returns whether a thread runs.
        """
        if self._watch is not None:
            return True
        if interval_s <= 0:
            return False
        stop = threading.Event()

        def watch():
            while not stop.wait(interval_s):
                if self.guard_fired(sync=False):
                    logger.error(
                        f"[musespark_open] unpool_direct: rank {self.rank} polled the direct unpool transport for "
                        f"more than {self.guard_ns / 1e9:.1f} s without its slot arriving: the "
                        "ranks' call sequences differ, or a peer stopped. The boundary keeps "
                        "polling, so this rank makes no progress" + ("; aborting" if abort else "")
                    )
                    if abort:
                        os.abort()
                    return

        self._watch_stop = stop
        self._watch = threading.Thread(
            target=watch, name="musespark_open.unpool_direct_watchdog", daemon=True
        )
        self._watch.start()
        return True

    def stop_guard_watch(self) -> None:
        """Stop the watch thread, if one runs (tests)."""
        if self._watch is not None:
            self._watch_stop.set()
            self._watch.join()
            self._watch = None

    def build_variants(self, variants: Sequence[Variant]) -> None:
        """Compile and load every variant not built yet. Collective over the group."""
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        error: Optional[BaseException] = None
        try:
            for key in variants:
                self.kernel(key)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "direct unpool boundary compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def kernel(self, key: Variant):
        """The compiled boundary for ``key`` (see ``variant``), built on first use outside a capture."""
        compiled = self._kernels.get(key)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"[musespark_open] unpool_direct: direct unpool boundary variant {key} was not built before graph "
                "capture"
            )
        rows, has_gain, post_eps, pre_eps, norm_fp32, trigger, capture = key
        compiled = _compile(
            self.device.index,
            rows,
            has_gain,
            post_eps,
            pre_eps,
            norm_fp32,
            TRIGGERS[trigger],
            capture,
            self.guard_ns,
            self.guard_break,
        )
        # A dry launch loads the module now, not inside a capture.
        dummy = self._dry.data_ptr()
        with torch.cuda.device(self.device):
            stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
            compiled(*([dummy] * 10), 0, 0, 1, stream)
        self._kernels[key] = compiled
        return compiled


def publish_copy(slot: torch.Tensor, partial: torch.Tensor) -> None:
    """Store ``partial``'s bits into a direct-transport ``slot`` from an elementwise kernel.

    For a producer that wrote its output elsewhere. Not ``Tensor.copy_``: a
    contiguous copy may run on the copy engine, which is not an SM store to the
    multicast address. An integer ``x | 0`` keeps every bit pattern.
    """
    torch.bitwise_or(partial.reshape(slot.shape).view(torch.int16), 0, out=slot.view(torch.int16))


_SHARED: dict = {}


def shared_workspace(mapping: Mapping) -> UnpoolDirectWorkspace:
    """The one direct workspace of this group, built on first use (collective, like the constructor)."""
    workspace = _SHARED.get(mapping)
    if workspace is None:
        workspace = UnpoolDirectWorkspace(mapping)
        _SHARED[mapping] = workspace
    return workspace


class UnpoolDirectBoundary(OpWrapper):
    """The gather boundary (``unpool_gather_boundary``) on slices the unpool GEMM already published.

    ``published`` is this rank's slot of buffer ``parity`` (0..2,
    ``workspace.slot(parity, rows)``), which the producer -- a GEMM handed it as
    its output -- has written on every rank; the op checks its address and never
    reads it. ``residual``, ``gain``, ``alpha``, ``beta``, ``weight``, the
    epsilons, the norm ordering, ``trigger`` and ``capture`` are the gather boundary's.
    Returns its ``(updated, next_input)`` for the same slices.

    A collective over the TP8 group: every rank calls it with the same row count
    and buffer, in the same order relative to its other calls on ``workspace``,
    and no two consecutive calls on one buffer (``buffer_sequence``). Launched
    with programmatic dependent launch.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DFlash2 K=7 verify: 8 rows, the one row count the target publishes at",
            spec=dict(rows=8, with_gain=True, norm_fp32_residual=False),
        ),
        Cell(
            why="the second buffer: odd layers publish into buffer 1",
            spec=dict(rows=8, with_gain=True, norm_fp32_residual=False, parity=1),
        ),
        Cell(
            why="the third buffer: the last fused layer of an odd count (61 in the model)",
            spec=dict(rows=8, with_gain=True, norm_fp32_residual=False, parity=2),
        ),
        Cell(
            why="early release: dependents launched at entry, the next QKV streams its weight meanwhile",
            spec=dict(rows=8, with_gain=True, norm_fp32_residual=False, trigger="entry"),
        ),
        Cell(
            why="capture with early release: the DFlash capture from a boundary releasing the next QKV at entry",
            spec=dict(
                rows=8, with_gain=True, norm_fp32_residual=False, trigger="entry", capture=True
            ),
        ),
        Cell(
            why="the FP32-residual pre-QKV norm ordering switch",
            spec=dict(rows=8, with_gain=True, norm_fp32_residual=True),
        ),
        Cell(
            why="a weightless first norm (the boundary's other arm)",
            spec=dict(rows=8, with_gain=False, norm_fp32_residual=False),
        ),
        Cell(
            why="dependents launched right after the wait",
            spec=dict(rows=8, with_gain=True, norm_fp32_residual=False, trigger="after_wait"),
        ),
        Cell(
            why="1 row: the cluster of 16 CTAs, two CTAs per peer slice",
            spec=dict(rows=1, with_gain=True, norm_fp32_residual=False),
        ),
        Cell(
            why="3 rows, a short draft between the two cluster shapes",
            spec=dict(rows=3, with_gain=True, norm_fp32_residual=False, parity=1),
        ),
    )

    note = """
    Certified against ``unpool_gather_boundary`` on the same slices bit for bit (the test publishes with
    the production split-K GEMM and with an elementwise copy, through CUDA-graph
    replays with changing inputs, back-to-back 62-layer graphs with in-graph
    rank delays, and a delayed rank), and against `reference` within its band.
    A concurrent poller checks that every half of a slot is first seen with its
    final value. The target serves 8 rows only; the other row counts are cells
    because the kernel has the gather boundary's geometry at every row count 1..8.
    """

    def __call__(
        self,
        published: torch.Tensor,
        workspace: UnpoolDirectWorkspace,
        parity: int,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        trigger: str = "exit",
        capture: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """... and, given ``capture`` (a ``[M, 8192]`` BF16 view with unit column
        stride), also writes ``updated.to(torch.bfloat16)`` into it."""
        rows = published.shape[0]
        compiled = workspace.kernel(
            variant(
                rows,
                gain is not None,
                post_eps,
                pre_eps,
                norm_fp32_residual,
                trigger,
                capture is not None,
            )
        )
        updated = torch.empty_like(residual)
        normalized = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=residual.device)
        stream = cuda.CUstream(torch.cuda.current_stream(residual.device).cuda_stream)
        compiled(
            workspace.receive().data_ptr(),
            workspace.control_ptr(),
            residual.data_ptr(),
            (weight if gain is None else gain).data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            weight.data_ptr(),
            updated.data_ptr(),
            normalized.data_ptr(),
            (normalized if capture is None else capture).data_ptr(),
            0 if capture is None else capture.stride(0),
            int(parity),
            0,
            stream,
        )
        return updated, normalized

    def reference(
        self,
        per_rank_locals: Sequence[torch.Tensor],
        workspace: Optional[UnpoolDirectWorkspace],
        parity: int,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        trigger: str = "exit",
        capture: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The gather boundary's reference: the boundary on the concatenated slices, float64 between BF16 roundings.

        The collective exception to mirroring: ``per_rank_locals`` is every
        rank's published slice in ascending rank order. ``workspace``, ``parity``
        and ``trigger`` are ignored (where the slices travel and when a dependent
        may launch change no value). Given ``capture``, the BF16 rounding of the
        returned residual is written into it, as the op does.
        """
        branch = torch.cat(list(per_rank_locals), dim=1)
        updated, next_input = _f1._reference_boundary(
            branch, residual, gain, alpha, beta, weight, post_eps, pre_eps, norm_fp32_residual
        )
        if capture is not None:
            capture.copy_(updated.to(torch.bfloat16))
        return updated, next_input

    def is_valid(
        self,
        published: torch.Tensor,
        workspace: UnpoolDirectWorkspace,
        parity: int,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        trigger: str = "exit",
        capture: Optional[torch.Tensor] = None,
    ) -> None:
        """Every input is read through raw 16-byte vector accesses at fixed strides.

        ``published`` is never read, but its address names the slot the peers'
        producers wrote: any other tensor would make the op consume slices
        nobody published for this call (it would wait for them forever).
        """
        rows = published.shape[0] if published.dim() == 2 else -1
        if not 1 <= rows <= MAX_ROWS or tuple(published.shape) != (rows, LOCAL_COLUMNS):
            raise ValueError(
                f"published must be [1..{MAX_ROWS}, {LOCAL_COLUMNS}], got {tuple(published.shape)}"
            )
        if (
            parity not in range(BUFFERS)
            or published.data_ptr() != workspace.slot(parity, rows).data_ptr()
        ):
            raise ValueError(f"published is not this rank's slot of buffer {parity} at {rows} rows")
        expected = (
            (residual, (rows, HIDDEN), torch.float32),
            (alpha, (HIDDEN,), torch.float32),
            (beta, (HIDDEN,), torch.float32),
            (weight, (HIDDEN,), torch.bfloat16),
        ) + (((gain, (HIDDEN,), torch.bfloat16),) if gain is not None else ())
        for tensor, shape, dtype in expected:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
                or tensor.device != residual.device
            ):
                raise ValueError(
                    f"expected a contiguous 16-byte aligned {dtype} {shape} tensor on "
                    f"{residual.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )
        if residual.device != workspace.device:
            raise ValueError(f"inputs on {residual.device}, the workspace on {workspace.device}")
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(TRIGGERS)}, got {trigger!r}")
        if capture is not None and not _f1.capture_fits(capture, rows, residual.device):
            raise ValueError(
                f"capture must be a 16-byte aligned BF16 ({rows}, {HIDDEN}) view with unit "
                f"column stride and a row stride divisible by 8 on {residual.device}, got "
                f"{capture.dtype} {tuple(capture.shape)} stride {tuple(capture.stride())}"
            )


unpool_direct_boundary = UnpoolDirectBoundary()


def available() -> bool:
    """Whether this image can build the kernel (CuTe DSL and the built-in cluster boundary)."""
    return _AVAILABLE


def log_once_enabled(layers: int, kernels: str, buffers: Sequence[int] = ()) -> None:
    last = f", the last on buffer {buffers[-1]}" if buffers else ""
    logger.info(
        f"[musespark_open] unpool_direct: the unpool GEMM publishes into the post-FFN boundary's direct transport on "
        f"{layers} layers (8-row decode graphs; buffers alternate per layer{last}; unpool kernel "
        f"{kernels}); eager and other row counts keep the gather boundary's tagged inbox"
    )
