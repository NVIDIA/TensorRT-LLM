# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused expert finalize + all-reduce of 64..512-token calls on a two-shot transport.

The built-in ``FinalizeIpcWorkspace.finalize`` (``tensorrt_llm/_torch/models/
musespark_open_finalize_ar.py``) serves at most 128 tokens and moves every rank's
whole BF16 partial to every peer (one-shot): ``3 * tokens * 8 KiB`` of NVLink ingress
per rank at TP4. This entry keeps its arithmetic and changes the transport and the
row range:

* **Arithmetic** -- the built-in TP4 kernel's, statement for statement, on its thread
  map: one 512-thread CTA per token, thread ``t`` owning columns ``8t .. 8t + 7`` of
  all eight slots. Per slot the thread's sum of squares of ``x * (1 + gamma)`` (word
  order), the warp's butterfly sum, the sixteen warp sums added from ``+0`` in warp
  order; ``scale = w * rsqrt(sum / 4096 + eps)`` (zero for a slot that is not this
  rank's); the slots accumulated in slot order from ``+0``; the partial rounded to
  BF16 (RN) with ``-0`` cleared; the ranks' partials summed in FP32 in rank order
  from ``+0`` and rounded once to BF16. Every token's result is therefore bitwise
  what the built-in kernel returns for it (``finalize`` up to 128 tokens,
  ``finalize_chunked`` above).
* **Transport (two-shot, owner per row).** Token ``t`` is owned by rank
  ``t % ranks``. A non-owner stores its partial into slab ``rank`` of the owner's
  scatter area (slot: buffer, source, ``t // ranks``, column vector); the owner sums
  every rank's partial and stores the four BF16 words of the sum into the gather slot
  (buffer, ``t``, column vector) of every other rank, where that rank's CTA of the
  token reads it. Ingress per rank: ``(ranks - 1) / ranks * tokens * 8 KiB`` twice,
  behind two dependent exchanges.

Lamport slots. Every slot starts as the sentinel word ``0x80000000``; a 32-bit word
that still holds it has not arrived. No payload word is the sentinel: a partial has
BF16 ``-0`` cleared, and the RN rounding of an FP32 sum from ``+0`` of such values
is never ``-0`` (the exact sum is a multiple of BF16's smallest subnormal, so is its
FP32 rounding, and a non-zero one rounds to a non-zero BF16). This needs the FP32
adds of that sum to keep subnormals: a flush-to-zero add would turn a negative
subnormal sum into ``-0``, so the kernel is compiled without ``.ftz``. The first read that
finds no sentinel word returns the final value. Polling is stale-only (only slots
that still hold a sentinel word are read again, all in flight together). A reader
writes the sentinel back into every slot it consumed before its CTA exits.

Ownership, per call:

* scatter slot ``(b, r, t // ranks, v)`` on the owner ``o = t % ranks`` (``r != o``):
  written once by rank ``r``'s thread ``(t, v)``, read and reset once by ``o``'s
  thread ``(t, v)``;
* gather slot ``(b, t, v)`` on a rank ``r != o``: written once by ``o``'s thread
  ``(t, v)`` after its scatter polls complete, read and reset once by ``r``'s thread
  ``(t, v)``. A rank never reads the gather slot of a token it owns.

Buffer reuse (the buffer ``b`` is the call's parity). Scatter: rank ``r`` writes call
``n + 2`` into ``o``'s scatter area only after its call ``n + 1`` completed; that call
read ``o``'s gather words of call ``n + 1`` for a token ``o`` owns (a call has at
least ``ranks`` tokens, so every rank owns one and is a non-owner of one every other
rank owns), which ``o`` published only after its call ``n`` -- resets included --
completed. Gather: ``o`` writes call ``n + 2`` into ``r``'s gather area only after it
read ``r``'s scatter of call ``n + 2`` for that token, which ``r`` publishes only
after its call ``n + 1`` (hence call ``n``, with the gather reset) completed. Both
need what the built-in kernel needs: calls on one workspace are issued in the same
order on every rank and serialized on one stream (graph replay included), and every
kernel launched with programmatic dependent launch between two calls waits on its
predecessor before it completes.

Parity on the device. Every call adds exactly ``GRID_STEP`` to a per-rank arrival
counter with fire-and-forget ``red.add`` (CTA 0 adds ``GRID_STEP + 1 - grid``, every
other CTA one, each right after its thread 0 read the counter). A CTA of call ``n``
reads it after every earlier call completed (its ``griddepcontrol.wait`` returned)
and before its own arrival, so it reads ``GRID_STEP * n + k`` with
``0 <= k < GRID_STEP`` (the grid is at most ``MAX_ROWS <= GRID_STEP`` CTAs);
``(counter >> GRID_SHIFT) & 1`` is the call's buffer on every CTA, and ``2**32`` is a
multiple of ``2 * GRID_STEP``, so the parity survives the counter's wrap.

Progress. A non-owner CTA publishes its scatter before it polls anything and then
polls only its token's gather slot; an owner CTA polls only its token's scatter slots,
which only the same token's CTAs of the peers write, each before any wait, and
publishes the gather as soon as those polls complete. No wait cycle exists. A call of
at most one token per SM runs the build bounded for one resident CTA per SM, a call of
at most two tokens per SM the build bounded for two (``MIN_BLOCKS``,
``WIDE_MIN_BLOCKS``), so on an otherwise idle device every CTA of such a call is
resident at once. Above that (more than ``2 * SMs`` tokens), or when concurrent work
holds SMs, progress rests on CTAs being dispatched in grid order on every rank (the
assumption the built-in MNNVL two-shot and the built-in expert finalize also rely on):
the lowest unfinished token's CTAs are resident on every rank, publish and finish, so
every token finishes. A dependent grid launches only after every CTA of this grid
triggered or exited, so an early dependent never holds back this grid's CTAs.

Poll bound. The served kernel spins without a bound; a rank whose peer never
publishes hangs at this kernel, where the executor's hang detector reports it (the
convention of every Lamport kernel of this target). ``timeout_ns > 0`` compiles a
bounded variant for qualification only: a poll that exceeds it executes ``trap``,
which aborts that rank's context; it never continues on sentinel data.

Workspace per rank (its own allocation and counter, never shared with the built-in
finalize, the ``finalize_transport`` kernel, the MNNVL all-reduce or the hops):
scatter ``2 * ranks * (MAX_ROWS / ranks) * 8 KiB`` + gather ``2 * MAX_ROWS * 8 KiB``
= 16 MiB at ``MAX_ROWS = 512``.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...kernels.persistent_experts import rows as _expert_rows
from . import finalize_allreduce_transport as _transport

#: Expert (pooled) width, selected experts per token, BF16 values per thread.
HIDDEN = 4096
TOP_K = 8
VECTOR = 8
#: One CTA per token, one 16-byte column vector of every slot per thread (the built-in TP4 map).
THREADS = HIDDEN // VECTOR
WARPS = THREADS // 32
#: The tokens a call of this entry takes: from the first row count where the one-shot is
#: ingress-bound (eight requests at K = 7) up to 64 requests.
MIN_ROWS = 64
MAX_ROWS = 512
#: The largest call the built-in fused finalize serves in one launch.
BUILTIN_MAX_TOKENS = 128
#: The tensor-parallel group sizes the transport is laid out for.
RANK_COUNTS = (4,)
MAX_RANKS = 8
SENTINEL = 0x80000000
BUFFERS = 2
ROW_BYTES = HIDDEN * 2
#: What one call adds to the arrival counter, whatever its grid (at most GRID_STEP CTAs).
GRID_SHIFT = 10
GRID_STEP = 1 << GRID_SHIFT
assert GRID_STEP >= MAX_ROWS
#: Resident CTAs per SM the registers are bounded for (one 512-thread CTA per token). Calls of
#: at most one token per SM run the ``MIN_BLOCKS`` build; larger calls run the
#: ``WIDE_MIN_BLOCKS`` build (at most 64 registers per thread), which keeps up to two tokens
#: per SM in one wave instead of starting a second wave.
MIN_BLOCKS = 1
WIDE_MIN_BLOCKS = 2
#: Control words: 2 holds the free-running arrival counter (0, 1 and 3 unused).
CONTROL_WORDS = 4
COUNTER_WORD = 2


def workspace_bytes(ranks: int, max_rows: int = MAX_ROWS) -> int:
    """Bytes of one rank's receive memory: the scatter area, then the gather area."""
    return BUFFERS * (ranks * (max_rows // ranks) + max_rows) * ROW_BYTES


HAS_CUTE_DSL = _transport.HAS_CUTE_DSL


if HAS_CUTE_DSL:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cutlass import BFloat16, Float32, Int32, Int64, Uint32
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    from tensorrt_llm._torch.models.musespark_open_finalize_ar import (
        _asm,
        _load_control,
        _load_input_x4,
        _pack_bf16x2,
        _pack_publishable_bf16x2,
        _store_x4,
        _unpack_bf16x2,
        _zero_unless,
    )

    from .finalize_allreduce_transport import (
        _load_f32,
        _load_row_x4,
        _load_s32,
        _poll_x4,
        _red_add,
        _reset_x4,
        _stale_bit,
        _store_peer_x4_if,
    )

    @dsl_user_op
    def _globaltimer(*, loc=None, ip=None):
        """The global nanosecond timer (bounded-poll variant only)."""
        result = _asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip)
        return Int64(result)

    @dsl_user_op
    def _trap_if(expired, *, loc=None, ip=None):
        """Abort this rank's context when ``expired`` is non-zero (bounded-poll variant only)."""
        _asm(
            None,
            [Uint32(expired).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p trap;\n}",
            "r",
            loc,
            ip,
        )

    class _FinalizeTwoShotKernel:
        """Finalize + two-shot all-reduce for up to ``max_rows`` tokens (served from MIN_ROWS).

        CTA ``b`` handles token ``b``; each of its 512 threads owns one 16-byte column
        vector of all eight slots. ``timeout_ns`` (0 = unbounded) bounds every poll.
        """

        def __init__(self, ranks: int, pdl: bool, timeout_ns: int, max_rows: int, min_blocks: int):
            if ranks not in RANK_COUNTS:
                raise ValueError(f"the two-shot finalize serves {RANK_COUNTS} ranks, got {ranks}")
            self.ranks = ranks
            self.pdl = pdl
            self.timeout_ns = int(timeout_ns)
            self.max_rows = int(max_rows)
            self.min_blocks = int(min_blocks)
            self.source_bytes = (self.max_rows // ranks) * ROW_BYTES
            self.gather_base = BUFFERS * ranks * self.source_bytes

        @cute.jit
        def __call__(
            self,
            gemm: cute.Pointer,
            indices: cute.Pointer,
            weights: cute.Pointer,
            norm: cute.Pointer,
            out: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            peer4: cute.Pointer,
            peer5: cute.Pointer,
            peer6: cute.Pointer,
            peer7: cute.Pointer,
            rows: Int32,
            row_stride: Int32,
            tokens: Int32,
            rank: Int32,
            eps: Float32,
            dry: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                gemm,
                indices,
                weights,
                norm,
                out,
                control,
                receive,
                peer0,
                peer1,
                peer2,
                peer3,
                peer4,
                peer5,
                peer6,
                peer7,
                rows,
                row_stride,
                tokens,
                rank,
                eps,
                dry,
            ).launch(
                grid=[tokens, 1, 1],
                block=[THREADS, 1, 1],
                smem=1024,
                stream=stream,
                use_pdl=self.pdl,
                min_blocks_per_mp=self.min_blocks,
            )

        @cute.jit
        def wait_slots(
            self,
            values: cute.Tensor,
            own0: Uint32,
            own1: Uint32,
            own2: Uint32,
            own3: Uint32,
            stale: Uint32,
            first: Int64,
            stride: Int64,
        ):
            """Fill ``values[p, 0..3]`` from slot ``first + p * stride`` for every bit ``p`` of ``stale``.

            Slots without a bit keep ``own``. Each round reads exactly the slots whose bit
            is set, all loads in flight, then recomputes the bits from the words.
            """
            for p in cutlass.range_constexpr(self.ranks):
                values[p, 0] = own0
                values[p, 1] = own1
                values[p, 2] = own2
                values[p, 3] = own3
            start = Int64(0)
            if cutlass.const_expr(self.timeout_ns > 0):
                start = _globaltimer()
            while stale != Uint32(0):
                polled = stale
                for p in cutlass.range_constexpr(self.ranks):
                    v0, v1, v2, v3 = _poll_x4(
                        first + Int64(p) * stride,
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
                if cutlass.const_expr(self.timeout_ns > 0):
                    elapsed = _globaltimer() - start
                    _trap_if(Uint32((stale != Uint32(0)) & (elapsed > Int64(self.timeout_ns))))

        @cute.kernel
        def kernel(
            self,
            gemm: cute.Pointer,
            indices: cute.Pointer,
            weights: cute.Pointer,
            norm: cute.Pointer,
            out: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            peer4: cute.Pointer,
            peer5: cute.Pointer,
            peer6: cute.Pointer,
            peer7: cute.Pointer,
            rows: Int32,
            row_stride: Int32,
            tokens: Int32,
            rank: Int32,
            eps: Float32,
            dry: Int32,
        ):
            peers = (peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7)
            tid = cute.arch.thread_idx()[0]
            block = cute.arch.block_idx()[0]
            blocks = cute.arch.grid_dim()[0]
            token = block
            column = tid * VECTOR
            lane = tid % 32
            warp = tid // 32

            smem = cutlass_utils.SmemAllocator()
            snapshot = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)
            warp_sums = smem.allocate_tensor(
                Float32, cute.make_layout((WARPS * TOP_K,)), byte_alignment=16
            )
            sums = smem.allocate_tensor(Float32, cute.make_layout((TOP_K,)), byte_alignment=16)

            # A dry launch only makes the driver load the module.
            if dry == 0:
                # Nothing above touches global memory: the expert rows, the row map, the
                # routing weights and the transport are all read after this wait.
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                # Selected rows and routing weights; invalid slots read nothing. The
                # predecessor grids' outputs are read with coherent loads (see
                # finalize_allreduce_transport); only the norm gain keeps ``.nc``.
                selection = Int64(token) * TOP_K
                gemm_address = gemm.toint()
                row_index = []
                row_weight = []
                row_words = []
                for k in cutlass.range_constexpr(TOP_K):
                    index = _load_s32(indices.toint() + (selection + k) * 4)
                    row_index.append(index)
                    row_weight.append(_load_f32(weights.toint() + (selection + k) * 4))
                    valid = Uint32(_expert_rows.routed_row_valid(index, rows))
                    offset = _expert_rows.routed_row_offset(
                        Int64(index), Int64(row_stride), Int64(column)
                    )
                    row_words.append(_load_row_x4(gemm_address + offset * 2, valid))
                norm_words = _load_input_x4(norm.toint() + Int64(column) * 2, Uint32(1))

                # This call's buffer, then this CTA's arrival.
                counter_address = control.toint() + COUNTER_WORD * 4
                if tid == 0:
                    counter = _load_control(counter_address)
                    snapshot[0] = Int32((counter >> Uint32(GRID_SHIFT)) & Uint32(1))
                    amount = Uint32(1)
                    if block == 0:
                        amount = Uint32(GRID_STEP + 1) - Uint32(blocks)
                    _red_add(counter_address, amount)

                gains = []
                for w in cutlass.range_constexpr(4):
                    low, high = _unpack_bf16x2(norm_words[w])
                    gains.append(low + Float32(1.0))
                    gains.append(high + Float32(1.0))

                # Sum of squares of the scaled row, per slot (the built-in's statements).
                partials = []
                for k in cutlass.range_constexpr(TOP_K):
                    square_sum = Float32(0.0)
                    for w in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(row_words[k][w])
                        scaled_low = low * gains[2 * w]
                        scaled_high = high * gains[2 * w + 1]
                        square_sum = (
                            square_sum + scaled_low * scaled_low + scaled_high * scaled_high
                        )
                    partials.append(cute.arch.warp_reduction_sum(square_sum))
                if lane == 0:
                    for k in cutlass.range_constexpr(TOP_K):
                        warp_sums[warp * TOP_K + k] = partials[k]
                cute.arch.barrier()
                if tid < TOP_K:
                    total = Float32(0.0)
                    for w in cutlass.range_constexpr(WARPS):
                        total = total + warp_sums[w * TOP_K + tid]
                    sums[tid] = total
                cute.arch.barrier()

                # Normalize each slot, weight it, and accumulate in FP32.
                result = [Float32(0.0)] * VECTOR
                for k in cutlass.range_constexpr(TOP_K):
                    valid = Uint32(_expert_rows.routed_row_valid(row_index[k], rows))
                    scale = row_weight[k] * cute.math.rsqrt(sums[k] / Float32(HIDDEN) + eps)
                    scale = _zero_unless(scale, valid)
                    for w in cutlass.range_constexpr(4):
                        low, high = _unpack_bf16x2(row_words[k][w])
                        result[2 * w] = result[2 * w] + scale * (low * gains[2 * w])
                        result[2 * w + 1] = result[2 * w + 1] + scale * (high * gains[2 * w + 1])
                own = []
                for w in cutlass.range_constexpr(4):
                    own.append(_pack_publishable_bf16x2(result[2 * w], result[2 * w + 1]))

                # Two-shot: token `token` is owned by rank `token % ranks`. `scatter` is its
                # slot of source 0 in the owner's scatter area, `gather` its slot in a
                # rank's gather area (both in the current buffer).
                buffer = Int64(snapshot[0])
                owner = token % self.ranks
                vector = Int64(column) * 2
                scatter = (
                    buffer * (self.ranks * self.source_bytes)
                    + Int64(token // self.ranks) * ROW_BYTES
                    + vector
                )
                gather = (
                    Int64(self.gather_base)
                    + buffer * (self.max_rows * ROW_BYTES)
                    + Int64(token) * ROW_BYTES
                    + vector
                )
                receive_address = receive.toint()
                values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                reduced = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                if owner != rank:
                    # Scatter this rank's partial into its slab on the owner, then wait
                    # for the owner's sum in this rank's gather slot.
                    for p in cutlass.range_constexpr(self.ranks):
                        _store_peer_x4_if(
                            peers[p].toint() + scatter + Int64(rank) * self.source_bytes,
                            Uint32(owner == p),
                            own[0],
                            own[1],
                            own[2],
                            own[3],
                        )
                    self.wait_slots(
                        values,
                        own[0],
                        own[1],
                        own[2],
                        own[3],
                        Uint32(1),
                        receive_address + gather,
                        Int64(0),
                    )
                    _reset_x4(receive_address + gather, Uint32(1))
                    for w in cutlass.range_constexpr(4):
                        reduced[w] = values[0, w]
                else:
                    # The owner: every peer's partial of this token (its own is `own`).
                    stale = Uint32((1 << self.ranks) - 1) & ~(Uint32(1) << Uint32(rank))
                    self.wait_slots(
                        values,
                        own[0],
                        own[1],
                        own[2],
                        own[3],
                        stale,
                        receive_address + scatter,
                        Int64(self.source_bytes),
                    )
                    for p in cutlass.range_constexpr(self.ranks):
                        _reset_x4(
                            receive_address + scatter + Int64(p) * self.source_bytes,
                            Uint32(rank != p),
                        )
                    # The built-in sum: every rank's partial in rank order, FP32 from +0.
                    total_out = [Float32(0.0)] * VECTOR
                    for p in cutlass.range_constexpr(self.ranks):
                        for w in cutlass.range_constexpr(4):
                            low, high = _unpack_bf16x2(values[p, w])
                            total_out[2 * w] = total_out[2 * w] + low
                            total_out[2 * w + 1] = total_out[2 * w + 1] + high
                    for w in cutlass.range_constexpr(4):
                        reduced[w] = _pack_bf16x2(total_out[2 * w], total_out[2 * w + 1])
                    for p in cutlass.range_constexpr(self.ranks):
                        _store_peer_x4_if(
                            peers[p].toint() + gather,
                            Uint32(rank != p),
                            reduced[0],
                            reduced[1],
                            reduced[2],
                            reduced[3],
                        )
                # Past the last transport access: a dependent may start its prologue now;
                # its griddepcontrol.wait still returns only once this grid has completed.
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                _store_x4(
                    out.toint() + Int64(token) * ROW_BYTES + vector,
                    reduced[0],
                    reduced[1],
                    reduced[2],
                    reduced[3],
                )

    _POINTER_DTYPES = (
        BFloat16,  # gemm
        Int32,  # indices
        Float32,  # weights
        BFloat16,  # norm
        BFloat16,  # out
        Uint32,  # control
        Uint32,  # receive
    ) + (Uint32,) * MAX_RANKS
    #: The built-in's assumptions: 4 bytes for the row map and the routing weights.
    _POINTER_ALIGNMENTS = (16, 4, 4) + (16,) * (len(_POINTER_DTYPES) - 3)

    @functools.lru_cache(maxsize=16)
    def _compiled_kernel(
        device_index: int, ranks: int, pdl: bool, timeout_ns: int, max_rows: int, min_blocks: int
    ):
        kernel = _FinalizeTwoShotKernel(ranks, pdl, timeout_ns, max_rows, min_blocks)
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *[
                    make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)
                    for dtype, align in zip(_POINTER_DTYPES, _POINTER_ALIGNMENTS)
                ],
                Int32(0),
                Int32(HIDDEN),
                Int32(1),
                Int32(0),
                Float32(1e-6),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


def _pdl_enabled() -> bool:
    return _transport._pdl_enabled()


def takes(num_tokens, max_rows: int = MAX_ROWS) -> bool:
    """Whether a call of ``num_tokens`` is in this entry's band (the same on every rank)."""
    return (
        isinstance(num_tokens, int)
        and not isinstance(num_tokens, bool)
        and MIN_ROWS <= num_tokens <= max_rows
    )


class FinalizeTwoShotWorkspace:
    """This transport's memory and state for one TP4 group, in front of another finalize workspace.

    ``finalize`` and ``finalize_chunked`` have the built-in ``FinalizeIpcWorkspace``
    signatures. While ``active`` (set by the fast path's switch around a decode forward,
    identical on every rank), calls of ``MIN_ROWS..max_rows`` tokens run this module's
    kernel; every other call goes to ``wrapped`` (the ``finalize_transport`` workspace or
    the built-in one), whose protocol state is separate, so the two interleave freely.
    Construction is collective over the group (fabric allocation, sentinel fill, compilation
    of both register bounds and a dry launch of each) and either succeeds on every rank or
    raises on every rank. ``timeout_ns > 0`` builds the bounded-poll variant (qualification only).
    """

    def __init__(
        self,
        wrapped,
        mapping: Mapping,
        *,
        timeout_ns: int = 0,
        max_rows: int = MAX_ROWS,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        if mapping.tp_size not in RANK_COUNTS:
            raise ValueError(f"the two-shot finalize serves TP4, got TP{mapping.tp_size}")
        if not MIN_ROWS <= max_rows <= GRID_STEP or max_rows % mapping.tp_size:
            raise ValueError(
                f"max_rows must be a multiple of the TP size in {MIN_ROWS}..{GRID_STEP}, "
                f"got {max_rows}"
            )
        self.wrapped = wrapped
        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.max_rows = int(max_rows)
        self.timeout_ns = int(timeout_ns)
        self.pdl = _pdl_enabled()
        self.active = False
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self.workspace_bytes = workspace_bytes(self.ranks, self.max_rows)
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("the CuTe DSL or the built-in finalize module is unavailable")
            if mapping.is_multi_node():
                raise RuntimeError("the two-shot finalize serves one tray")
            if torch.cuda.get_device_capability(self.device) != (10, 3):
                raise RuntimeError("the two-shot finalize runs on SM103 only")
            self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, self.workspace_bytes, mapping, False
            )
            words = self.workspace_bytes // 4
            self._uc = [
                self._handle.get_uc_buffer(peer, (words,), torch.int32, 0)
                for peer in range(self.ranks)
            ]
            with torch.cuda.device(self.device):
                self._uc[self.rank].fill_(-(2**31))
                self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=self.device)
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "two-shot finalize workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        receive = self._uc[self.rank].data_ptr()
        peers = [t.data_ptr() for t in self._uc]
        self._fixed = (
            self.control.data_ptr(),
            receive,
            # Peer pointers past the group are never dereferenced.
            *(peers + [receive] * (MAX_RANKS - self.ranks)),
        )
        error = None
        try:
            self._compiled = {
                blocks: _compiled_kernel(
                    self.device.index,
                    self.ranks,
                    self.pdl,
                    self.timeout_ns,
                    self.max_rows,
                    blocks,
                )
                for blocks in (MIN_BLOCKS, WIDE_MIN_BLOCKS)
            }
            # Dry launches load both modules now, not inside a capture.
            dummy = self.control.data_ptr()
            self._launch(dummy, dummy, dummy, dummy, dummy, 0, HIDDEN, 1, 1.0, 1)
            self._launch(dummy, dummy, dummy, dummy, dummy, 0, HIDDEN, self.sms + 1, 1.0, 1)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "two-shot finalize kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _launch(self, gemm, indices, weights, norm, out, rows, row_stride, tokens, eps, dry=0):
        self._compiled[WIDE_MIN_BLOCKS if tokens > self.sms else MIN_BLOCKS](
            gemm,
            indices,
            weights,
            norm,
            out,
            *self._fixed,
            rows,
            row_stride,
            tokens,
            self.rank,
            eps,
            dry,
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
        )

    def takes(self, num_tokens) -> bool:
        """Whether a call of ``num_tokens`` runs this kernel while active."""
        return takes(num_tokens, self.max_rows)

    def _validate(self, gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens):
        """The built-in workspace's input contract, for ``MIN_ROWS..max_rows`` tokens."""
        if not self.takes(num_tokens):
            raise ValueError(
                f"num_tokens must be in [{MIN_ROWS}, {self.max_rows}], got {num_tokens!r}"
            )
        if not (isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0):
            raise ValueError(f"eps must be positive and finite, got {eps!r}")
        if (
            gemm2_output.device != self.device
            or gemm2_output.dtype != torch.bfloat16
            or gemm2_output.dim() != 2
            or gemm2_output.shape[1] < HIDDEN
        ):
            raise ValueError(
                "gemm2_output must be a [R, >=4096] BF16 tensor on the workspace device"
            )
        rows = gemm2_output.shape[0]
        row_stride = HIDDEN
        if rows > 0:
            if gemm2_output.stride(1) != 1 or gemm2_output.data_ptr() % 16:
                raise ValueError("gemm2_output needs unit column stride and a 16-byte aligned base")
            if rows > 1:
                row_stride = gemm2_output.stride(0)
                if row_stride % VECTOR:
                    raise ValueError("gemm2_output row stride must be a multiple of 8 elements")
        expected = num_tokens * TOP_K
        for name, tensor, dtype in (
            ("indices", indices, torch.int32),
            ("routing_weights", routing_weights, torch.float32),
        ):
            if (
                tensor.device != self.device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.numel() != expected
            ):
                raise ValueError(
                    f"{name} must be a contiguous {dtype} tensor with {expected} entries"
                )
        if (
            norm_weight.device != self.device
            or norm_weight.dtype != torch.bfloat16
            or not norm_weight.is_contiguous()
            or norm_weight.numel() != HIDDEN
            or norm_weight.data_ptr() % 16
        ):
            raise ValueError(
                "norm_weight must be a contiguous, 16-byte aligned BF16 tensor of 4096"
            )
        return rows, row_stride

    def launch(
        self,
        gemm2_output: torch.Tensor,
        indices: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
    ) -> torch.Tensor:
        """This kernel on ``num_tokens`` tokens, active or not (collective)."""
        rows, row_stride = self._validate(
            gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens
        )
        out = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device)
        self._launch(
            gemm2_output.data_ptr() if rows > 0 else 0,
            indices.data_ptr(),
            routing_weights.data_ptr(),
            norm_weight.data_ptr(),
            out.data_ptr(),
            rows,
            row_stride,
            num_tokens,
            float(eps),
        )
        return out

    def finalize(
        self,
        gemm2_output: torch.Tensor,
        indices: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
    ) -> torch.Tensor:
        """The wrapped ``finalize``, with ``MIN_ROWS..128`` tokens on this kernel while active."""
        if self.active and self.takes(num_tokens) and num_tokens <= BUILTIN_MAX_TOKENS:
            return self.launch(gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens)
        return self.wrapped.finalize(
            gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens
        )

    def finalize_chunked(
        self,
        gemm2_output: torch.Tensor,
        indices: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
    ) -> torch.Tensor:
        """This kernel for ``MIN_ROWS..max_rows`` tokens while active, else the wrapped ``finalize_chunked``.

        Raises:
            RuntimeError: for a call this kernel does not take when ``wrapped`` has no
                ``finalize_chunked``; the fast path's install bounds the layers so that no
                such call is made.
        """
        if self.active and self.takes(num_tokens):
            return self.launch(gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens)
        chunked = getattr(self.wrapped, "finalize_chunked", None)
        if chunked is None:
            raise RuntimeError(
                f"two-shot finalize: an unarmed chunked call of {num_tokens} tokens, and the "
                "wrapped workspace has no finalize_chunked"
            )
        return chunked(gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens)


class FinalizeTwoShot(OpWrapper):
    """The fused expert finalize + all-reduce of a ``FinalizeTwoShotWorkspace`` (64..512 tokens).

    Returns the ``[num_tokens, 4096]`` BF16 group sum of every rank's partial: per
    token, each valid expert row ``x`` (slot index in ``[0, R)``) scaled by
    ``w * rsqrt(mean((x * g)^2) + eps)`` times ``x * g`` with ``g = 1 + norm``, summed
    over the slots, rounded to BF16 with ``-0`` cleared, then summed over the ranks in
    FP32 in rank order and rounded to BF16.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="the fewest tokens: eight requests at K = 7", spec=dict(tokens=MIN_ROWS)),
        Cell(why="the built-in fused finalize's largest call", spec=dict(tokens=128)),
        Cell(why="the first call past the built-in fused finalize", spec=dict(tokens=129)),
        Cell(why="the first call the served unfused combine takes sparse", spec=dict(tokens=160)),
        Cell(why="twenty-four requests: the bs-24 graph", spec=dict(tokens=192)),
        Cell(why="the transport's capacity: 64 requests", spec=dict(tokens=MAX_ROWS)),
        Cell(why="a token count the rank count does not divide", spec=dict(tokens=101)),
        Cell(
            why="every slot of a token on one rank, none on the others",
            spec=dict(tokens=64, placement="concentrated"),
        ),
        Cell(
            why="exact and negative zeros in the rows, tokens with no valid slot",
            spec=dict(tokens=64, zeros=True),
        ),
        Cell(
            why="invalid slots and cancelling rows: the rank-order sum is observable",
            spec=dict(tokens=64, invalid=True, cancel=True),
        ),
    )

    note = """
    Metadata consumed: ``indices`` (the expert op's ``[tokens, 8]`` row map, under the
    ``persistent_experts.rows`` contract) and the routing weights. Call site: the
    built-in MoE's ``workspace.finalize`` (64..128 tokens) or
    ``workspace.finalize_chunked`` (129..512 tokens) once per layer per armed decode
    step; the fast path ``finalize_two_shot`` puts a ``FinalizeTwoShotWorkspace`` in
    that attribute and arms it around decode forwards of 64..512 rows.

    Certified bit for bit against the built-in kernel (``finalize`` up to 128 tokens,
    ``finalize_chunked`` above) by its own four-rank test (every cell, a delayed rank,
    back-to-back calls of changing token counts, CUDA-graph replays with changing
    inputs) and against `reference` within a band. Not certified: TP8, more than 512
    tokens.
    """

    def __call__(
        self,
        gemm2_output: torch.Tensor,
        indices: torch.Tensor,
        routing_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
        workspace: FinalizeTwoShotWorkspace,
    ) -> torch.Tensor:
        if not workspace.takes(num_tokens):
            raise ValueError(f"finalize_two_shot serves {MIN_ROWS}..{workspace.max_rows} tokens")
        return workspace.launch(
            gemm2_output, indices, routing_weights, norm_weight, eps, num_tokens
        )

    def reference(
        self,
        per_rank_gemm2_outputs: Sequence[torch.Tensor],
        per_rank_indices: Sequence[torch.Tensor],
        per_rank_routing_weights: Sequence[torch.Tensor],
        norm_weight: torch.Tensor,
        eps: float,
        num_tokens: int,
        workspace,
    ) -> torch.Tensor:
        """The op's rounding points (the BF16 partial, the BF16 sum), float64 in between.

        The collective exception to mirroring: takes every rank's expert rows, slot
        indices and routing weights in ascending rank order. ``workspace`` is ignored.
        """
        return _transport.finalize_allreduce_transport.reference(
            per_rank_gemm2_outputs,
            per_rank_indices,
            per_rank_routing_weights,
            norm_weight,
            eps,
            num_tokens,
            None,
        )


finalize_two_shot = FinalizeTwoShot()


def log_once_enabled(layers: int, max_rows: int = MAX_ROWS) -> None:
    logger.info(
        f"[musespark_open] finalize_two_shot: fused expert finalize + two-shot all-reduce of all "
        f"{layers} layers on decode steps of {MIN_ROWS}..{max_rows} rows (owner per row, "
        "unicast scatter and gather, stale-only polling); the built-in kernel and the unfused "
        "combine + all-reduce otherwise"
    )
