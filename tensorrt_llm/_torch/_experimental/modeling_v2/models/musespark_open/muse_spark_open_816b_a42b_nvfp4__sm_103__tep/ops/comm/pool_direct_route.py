# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routed pool kernel on a transport its producers publish into themselves (``pool_into_route``).

``pool_gather_route`` (the tagged routed pool kernel of ``moe_prologue``) loads this rank's ``[M, C]`` pool
slice and ``[M, L]`` router logits after its dependency wait (``C`` = 512 columns
and ``L`` = 32 experts at TP8, 1024 and 64 at TP4), publishes both to
the peers as tagged 64-bit words, and polls the peers' words. Here the pool GEMM's
output tensor *is* this rank's pool slot of a multicast transport, and the router
shard (``ops/moe/router_shard.router_shard_publish``, on the router fork's side stream)
stores its logit block into this rank's logit boxes of the same transport: the
producers' own stores reach every rank's copy, so the routed kernel only polls,
resets, and then runs ``pool_gather_route``'s routing, placement, early-id
publication and RMSNorm + NVFP4 quantization statement for statement. Off
the critical path: the dependency-wait release followed by the reload of the
local slice and logits and their publication.

Transport (Lamport, payload as flag). Every rank holds ``3 buffers x world source
ranks`` records of ``record_words(world)`` 32-bit words, filled with the sentinel word
``0xFFFEFFFE``. A record is the source rank's pool slot -- ``[8, C]`` BF16,
row-major, one contiguous block, a GEMM output as it is -- followed by
``CTAS`` logit boxes of ``[8, L]`` FP32, one per CTA of the routed kernel (a
CTA routes every row, so each CTA reads a full copy and resets only its own).
A thread of the quantizing warps polls its ``_VECTORS`` 16-byte pieces of a row
where the tagged kernel would have them: vector ``v`` in sender ``owner(v)``'s slot
(one sender per vector at TP8, two vectors per sender at TP4).

* Pool slot: float-to-BF16 rounding (``cvt.rn.bf16{x2}.f32``, the GEMM
  epilogues') turns every NaN into ``0x7FFF``, so no product holds the half
  ``0xFFFE``; the test is per 16-bit half (``unpool_direct_boundary``'s).
* Logit boxes: each logit is the product of an FP32 sum and the multiplier
  (``mul.f32``), whose NaN results are the canonical ``0x7FFFFFFF``, so no
  logit is the word ``0xFFFEFFFE``; each logit is one aligned 32-bit store, so
  the test is per 32-bit word (a valid FP32 may hold ``0xFFFE`` in either half).

Ownership and reuse (cuFlow):

* One writer per element per call: rank ``r``'s pool GEMM stores each element
  of its pool slot once (flashinfer's split-K / direct kernels, or the site
  policy's copy kernel, ``dense_gemms.PublishPolicy``), and rank ``r``'s router
  stores each logit once per box.
* One reader and resetter per element per call: the thread of each rank's
  routed kernel that owns those words, on its own copy (own slot included: the
  producers' multicast stores reach the local copy asynchronously like the
  peers'); pool words by the quantizing warps of the CTA that owns the row,
  logit words by the routing warps of the box's CTA.
* Reuse: no two consecutive calls on the workspace use the same buffer, the
  last call of a step and the first of the next included (``buffer_sequence``
  of ``unpool_direct_boundary``). Buffer ``p`` of rank ``B``'s copy is next
  written by a producer of a later call ``m`` on that buffer. Rank ``A``'s pool
  GEMM stores only from its epilogue, after its dependency wait, and ``A``'s
  router is a plain launch that starts only after ``A``'s attention hop of the
  same layer completed; either way after ``A``'s call ``m - 1`` completed
  (stream order, and every PDL kernel between two calls waits before it
  completes), which needed ``B``'s call-``(m - 1)`` records, which ``B``'s
  producers stored only after every earlier call of ``B`` -- among them ``B``'s
  last call on buffer ``p``, call ``m - 2`` or earlier, whose resets are part of
  it -- completed.
* Progress: every producer stores without waiting on anything remote; every
  thread of the routed kernel waits only for eight 16-byte pieces of eight
  records, after its own dependency wait. A poll that has seen no arrival for
  ``guard_ns`` sets a guard word in host memory (and keeps polling).

Wide transport (``max_rows=WIDE_MAX_ROWS``; upstream b5ca973 / #21377 at TP4, here also TP8
routing-only, ``tep8_wide_direct_pool`` beside the TEP8 router shard's wide
route): the decode steps of two to four
requests (16..32 rows, ``wide_prologue``) run the wide tagged kernel's statements, in
which each CTA routes only its own two rows (no placement, no early ids). A record is
then the source rank's ``[32, C]`` pool slot followed by one ``[32, L]`` FP32 logit
block, the router's output as it is (``router_shard_rows`` with the block as its
output): CTA ``c`` polls and resets rows ``2c`` and ``2c + 1`` of each sender's block.
A separate workspace with its own buffer sequence; the 8-row layout is unchanged.

PDL memory order: before ``griddepcontrol.wait`` the routing warps read the
expert bias (a parameter), as in ``pool_gather_route``; nothing else, in
particular nothing of the transport. After it: the transport (ordered by the
sentinel test, not by the wait), the gain and scale (parameters), the expert
kernel's counter (``early_expert_ids``, final once the wait returned). The trigger
placements are ``pool_gather_route``'s (the early-id one after the logit poll: never
before the CTA's own wait).

Calls on one workspace must be issued in the same order on every rank, with the
same row counts and buffers, serialized on one stream (graph replay included).
The workspace is this op's alone.

BF16 norm (``bf16_norm=``, a separately compiled variant; the default kernels do not include it):
for the BF16-expert layers (0 and 61), whose expert input is the pre-expert gemma RMSNorm in
BF16 (``moe_prologue.PF_BF16_FRONT`` mode 2), phase 3 writes that norm instead of the NVFP4
payload, with ``pool_gather_route``'s ``bf16_norm`` statements: the same sum of squares and
``rstd`` of phase 2, the weight loaded with flashinfer ``RMSNormKernel``'s row-broadcast tiled
copy over the same thread-value layout, ``y = x * rstd * (w + 1)`` in FP32, one rounding to
BF16. The norm weight is a parameter, so these warps load it before their dependency wait.
No block scale is written and the early-id publication is not available in this variant.
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

from . import pool_gather_route as _route
from . import unpool_direct_boundary as _direct
from .pool_gather_fp4quant import _VECTOR_SPAN_WORDS, _VECTORS, BLOCK, HIDDEN, local_columns
from .projection_inbox import MAX_ROWS, WIDE_MAX_ROWS

#: The TP sizes this transport is laid out for: TP8 over two trays and TP4 on one.
WORLDS = (4, 8)
#: The TP8 layout's values, which the TP8-only callers read; ``record_words(world)``
#: and its siblings give every layout.
WORLD = 8
LOCAL_COLUMNS = local_columns(WORLD)

EXPERTS = _route.EXPERTS
LOCAL_EXPERTS = _route.LOCAL_EXPERTS
TOP_K = _route.TOP_K
PLACE_NONE = _route.PLACE_NONE
PLACE_BALANCE = _route.PLACE_BALANCE
PLACE_BALANCE3 = _route.PLACE_BALANCE3
PLACE_HALF = _route.PLACE_HALF
#: Buffers: two that alternate call by call, a third for the last call of an odd count.
BUFFERS = _direct.BUFFERS
buffer_sequence = _direct.buffer_sequence
check_buffer_sequence = _direct.check_buffer_sequence
#: The empty-slot word (two BF16 NaNs ``0xFFFE``; as FP32 a NaN no arithmetic returns).
DIRECT_SENTINEL = _direct.DIRECT_SENTINEL
_SENTINEL_I32 = DIRECT_SENTINEL - (1 << 32)
#: CTAs of the routed kernel at ``MAX_ROWS`` rows (two rows each): one logit box each.
CTAS = MAX_ROWS // 2
CHUNKS = _route.CHUNKS
#: The wide layout's half-window census (``PLACE_HALF``), per buffer in rank-local device
#: memory: ``CHUNKS`` presence words, the CTAs' arrival count and their read count.
CENSUS_WORDS = 2 * CHUNKS
CENSUS_ARRIVED = CHUNKS
CENSUS_READ = CHUNKS + 1
#: Census words per buffer: the presence, arrival and read words.
CENSUS_STRIDE = CENSUS_WORDS


def pool_row_words(world: int) -> int:
    """32-bit words of one row of a pool slot: ``local_columns(world)`` BF16."""
    return local_columns(world) // 2


def check_layout(world: int, max_rows: int) -> None:
    """Raise unless ``(world, max_rows)`` is a served layout: 8 rows at TP8 or TP4, 32 rows at TP4
    or TP8 (at TP8 routing only, ``PLACE_NONE``: the tagged kernel's wide statements are the
    TP-generic ones, each CTA's ``[2, L]`` logit words; the half window's census is TP4's)."""
    if world not in WORLDS or max_rows not in (MAX_ROWS, WIDE_MAX_ROWS):
        raise ValueError(f"no direct pool layout for TP{world} and {max_rows} rows")


def pool_slot_words(world: int, max_rows: int = MAX_ROWS) -> int:
    """32-bit words of a pool slot: ``max_rows`` rows."""
    return max_rows * pool_row_words(world)


def box_words(world: int) -> int:
    """32-bit words of a logit box of the 8-row layout: ``[MAX_ROWS, local_experts(world)]`` FP32."""
    return MAX_ROWS * _route.local_experts(world)


def logit_words(world: int, max_rows: int = MAX_ROWS) -> int:
    """32-bit words of a record's logits: ``CTAS`` boxes (8 rows), or one ``[max_rows, L]`` block (wide)."""
    if max_rows == MAX_ROWS:
        return CTAS * box_words(world)
    return max_rows * _route.local_experts(world)


def record_words(world: int, max_rows: int = MAX_ROWS) -> int:
    """32-bit words of a record: the pool slot, then the logits (``logit_words``)."""
    return pool_slot_words(world, max_rows) + logit_words(world, max_rows)


def workspace_bytes(world: int, max_rows: int = MAX_ROWS) -> int:
    """Bytes of one rank's copy of the transport: ``BUFFERS x world`` records."""
    return BUFFERS * world * record_words(world, max_rows) * 4


#: 32-bit words of one row of a pool slot, of a pool slot, of a logit box, of a record
#: (the TP8 layout).
POOL_ROW_WORDS = pool_row_words(WORLD)
POOL_SLOT_WORDS = pool_slot_words(WORLD)
BOX_WORDS = box_words(WORLD)
RECORD_WORDS = record_words(WORLD)
WORKSPACE_BYTES = workspace_bytes(WORLD)
#: Per-rank control words in pinned host memory: ``GUARD_WORD`` is set by a poll that
#: outlived the guard.
CONTROL_WORDS = 4
GUARD_WORD = 0
DIRECT_GUARD_NS = _direct.DIRECT_GUARD_NS
GUARD_WATCH_S = _direct.GUARD_WATCH_S

QUANT_WARPS = _route.QUANT_WARPS
QUANT_THREADS = _route.QUANT_THREADS
ROUTE_THREADS = _route.ROUTE_THREADS
QUANT_BARRIER = _route.QUANT_BARRIER
ROUTE_BARRIER = _route.ROUTE_BARRIER

_AVAILABLE = _route.available() and _direct.available()

if _AVAILABLE:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint8, Uint32
    from cutlass.cutlass_dsl import T, dsl_user_op

    from .pool_gather_fp4quant import (
        _fp4,
        _load_shared_words,
        _store_global_words,
        _store_shared_words,
    )
    from .pool_gather_route import (
        _add_rn,
        _bar_sync,
        _bf16_bits,
        _div_rn,
        _div_u32,
        _eq,
        _f32_bits,
        _ld_global_b32,
        _ld_relaxed_gpu_b32,
        _ld_shared_b32,
        _lo32,
        _lt_u,
        _max_s32,
        _min_u32,
        _placed_expert,
        _PoolGatherRouteKernel,
        _popc,
        _red_shared_or,
        _redux_max_s32,
        _route_key,
        _row_sum,
        _sel,
        _shfl_idx_b32,
        _shfl_xor_b32,
        _sigmoid,
        _sort8,
        _st_global_b16,
        _st_global_b32,
        _st_relaxed_gpu_pair,
        _st_shared_b32,
        _stamp_timer,
        _top8_by_redux,
    )
    from .projection_inbox import _asm, _i64, _u32
    from .unpool_direct_boundary import _globaltimer, _pending, _poll_x4, _reset_x4, _set_guard_word

    @dsl_user_op
    def _red_global_or(address, value, *, loc=None, ip=None):
        """A relaxed GPU-scope OR into a rank-local global word (the wide census)."""
        _asm(
            None,
            [_i64(address, loc, ip), _u32(value, loc, ip)],
            "red.relaxed.gpu.global.or.b32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _red_release_gpu_add(address, value, *, loc=None, ip=None):
        """A release GPU-scope add (a CTA's arrival at the wide census)."""
        _asm(
            None,
            [_i64(address, loc, ip), _u32(value, loc, ip)],
            "red.release.gpu.global.add.u32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _atom_acq_rel_gpu_add(address, value, *, loc=None, ip=None):
        """An acquire-release GPU-scope add returning the old value."""
        return Uint32(
            _asm(
                T.i32(),
                [_i64(address, loc, ip), _u32(value, loc, ip)],
                "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
                "=r,l,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _ld_acquire_gpu_b32(address, *, loc=None, ip=None):
        return Uint32(
            _asm(
                T.i32(),
                [_i64(address, loc, ip)],
                "ld.acquire.gpu.global.b32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _st_relaxed_gpu_b32(address, value, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip), _u32(value, loc, ip)],
            "st.relaxed.gpu.global.b32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _pending_words(w0, w1, w2, w3, *, loc=None, ip=None):
        """1 while any of the four 32-bit words is ``DIRECT_SENTINEL``, else 0."""
        text = "{\n.reg .pred p;\n.reg .b32 s;\nmov.b32 s, 0;\n"
        for operand in range(1, 5):
            text += f"setp.eq.u32 p, ${operand}, 0xFFFEFFFE;\n@p mov.b32 s, 1;\n"
        text += "mov.b32 $0, s;\n}"
        return Uint32(
            _asm(
                T.i32(),
                [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
                text,
                "=r,r,r,r,r",
                loc,
                ip,
                side_effects=False,
            )
        )

    class _PoolDirectRouteKernel(_PoolGatherRouteKernel):
        """``pool_gather_route``'s kernel on the direct transport: poll + reset, then its routing,
        placement, early-id publication and quantization statement for statement.

        The per-row tag state, the loads of the local pool slice and logits and
        their publication are gone, and with them the named barrier by which the
        routing warps told the quantizing warps that they had read that state.
        """

        def __init__(
            self,
            peer_rank: int,
            place: int,
            static_from: int,
            publish: int,
            epoch_word: int,
            trigger: int,
            stamps: int,
            guard_ns: int,
            guard_break: bool,
            world: int = WORLD,
            max_rows: int = MAX_ROWS,
            census: int = 0,
            bf16_norm: bool = False,
        ):
            check_layout(world, max_rows)
            wide = max_rows > MAX_ROWS
            if wide and (
                place not in (PLACE_NONE, PLACE_HALF)
                or (place == PLACE_HALF) != bool(census)
                or (place == PLACE_HALF and world != 4)
            ):
                raise ValueError(
                    "the wide layout routes only, or places the half window with its census words"
                )
            # A wide layout compiles the tagged kernel's wide statements (``self.wide``):
            # each CTA routes its own rows, without early ids; the tagged kernel has no
            # wide placement, so it is built routing-only and the half window's placement,
            # over the grid's census (``census``), is this kernel's.
            super().__init__(
                peer_rank,
                PLACE_NONE if wide else place,
                static_from,
                True,
                False,
                publish,
                epoch_word,
                trigger,
                stamps,
                world,
                max_rows,
                bf16_norm,
            )
            self.place = place
            self.census = int(census)
            self.guard_ns = int(guard_ns)
            self.guard_break = bool(guard_break)
            self.pool_row_words = pool_row_words(world)
            self.pool_slot_words = pool_slot_words(world, max_rows)
            self.box_words = box_words(world)
            self.record_words = record_words(world, max_rows)

        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mY: cute.Tensor,
            mS: cute.Tensor,
            mGlobalScale: cute.Tensor,
            M: Int32,
            eps: Float32,
            renorm_eps: Float32,
            receive: Int64,
            control: Int64,
            buffer: Int32,
            bias: Int64,
            logits_full: Int64,
            selected: Int64,
            weights: Int64,
            local_ids: Int64,
            weights_bf16: Int64,
            stats: Int64,
            dry: Int32,
            enable_pdl: cutlass.Constexpr[bool],
            store_gathered: cutlass.Constexpr[bool],
            stream,
        ):
            tv_shape, tv_stride = self._make_tv_layout(
                self.threads_per_row,
                self.rows_per_block,
                self.vec_size,
                self.num_vec_blocks,
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
            tiler_mn = (self.rows_per_block, self.cols_per_tile)
            self.kernel(
                mX,
                mW,
                mY,
                mS,
                mGlobalScale,
                M,
                eps,
                renorm_eps,
                receive,
                control,
                buffer,
                bias,
                logits_full,
                selected,
                weights,
                local_ids,
                weights_bf16,
                stats,
                dry,
                enable_pdl,
                store_gathered,
                tv_layout,
                tiler_mn,
            ).launch(
                grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
                block=[self.num_threads + ROUTE_THREADS, 1, 1],
                smem=self._smem_size_in_bytes() + self._route_smem_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mY: cute.Tensor,
            mS: cute.Tensor,
            mGlobalScale: cute.Tensor,
            M: Int32,
            eps: Float32,
            renorm_eps: Float32,
            receive: Int64,
            control: Int64,
            buffer: Int32,
            bias: Int64,
            logits_full: Int64,
            selected: Int64,
            weights: Int64,
            local_ids: Int64,
            weights_bf16: Int64,
            stats: Int64,
            dry: Int32,
            enable_pdl: cutlass.Constexpr[bool],
            store_gathered: cutlass.Constexpr[bool],
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32

            H = self.H
            block_size = self.block_size
            num_sf_blocks_per_row = self.num_sf_blocks_per_row
            threads_per_row = tv_layout.shape[0][0]
            warps_per_row = max(threads_per_row // 32, 1)
            rows_per_block = tiler_mn[0]

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            sLogits = smem.allocate_tensor(
                Uint32, cute.make_layout((MAX_ROWS * EXPERTS,)), byte_alignment=16
            )
            sSelected = smem.allocate_tensor(
                Uint32, cute.make_layout((MAX_ROWS * TOP_K,)), byte_alignment=16
            )
            sWeights = smem.allocate_tensor(
                Uint32, cute.make_layout((MAX_ROWS * TOP_K,)), byte_alignment=16
            )
            sCensus = smem.allocate_tensor(Uint32, cute.make_layout((CHUNKS,)), byte_alignment=16)
            world = self.world
            local_count = self.local_experts
            record_bytes = self.record_words * 4
            # This buffer's records in the local copy: sender b's at records + b * record_bytes.
            records = receive + Int64(buffer) * Int64(world * record_bytes)

            # A dry launch only makes the driver load the module, outside any capture.
            if dry == Int32(0):
                if warp < QUANT_WARPS:
                    # ======================================================
                    # Warps 0-3: pool_gather_fp4quant statement for
                    # statement, the pool slices polled from the transport.
                    # ======================================================
                    if cutlass.const_expr(self.bf16_norm):
                        # BF16 norm variant: the tile layouts first (address arithmetic only),
                        # then the norm weight (a parameter) into registers, then the wait.
                        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
                        copy_atom_load_async = cute.make_copy_atom(
                            cute.nvgpu.cpasync.CopyG2SOp(),
                            mX.element_type,
                            num_bits_per_copy=_fp4.COPY_BITS,
                        )
                        tiled_copy_load = cute.make_tiled_copy(
                            copy_atom_load_async, tv_layout, tiler_mn
                        )
                        thr_copy_X = tiled_copy_load.get_slice(tidx)
                        tXgX = thr_copy_X.partition_S(gX)
                        tXsX = thr_copy_X.partition_D(sX)
                        tXrX = cute.make_fragment_like(tXgX)
                        mW_2d = cute.make_tensor(
                            mW.iterator,
                            cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))),
                        )
                        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))
                        copy_atom_w = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            mW.element_type,
                            num_bits_per_copy=_fp4.COPY_BITS,
                        )
                        thr_copy_W = cute.make_tiled_copy(
                            copy_atom_w, tv_layout, tiler_mn
                        ).get_slice(tidx)
                        tWgW = thr_copy_W.partition_S(gW)
                        tWrW = cute.make_fragment_like(tWgW)
                        tXrW = thr_copy_X.retile(tWrW)
                        cute.copy(copy_atom_w, tWgW, tWrW)
                        if enable_pdl:
                            cute.arch.griddepcontrol_wait()
                        lane_in_row = tidx % threads_per_row
                        row_in_block = tidx // threads_per_row
                    else:
                        if enable_pdl:
                            cute.arch.griddepcontrol_wait()
                        lane_in_row = tidx % threads_per_row
                        row_in_block = tidx // threads_per_row
                        fp4_max_rcp = _fp4.rcp_approx_ftz(Float32(_fp4.FLOAT4_E2M1_MAX))

                        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
                        copy_atom_load_async = cute.make_copy_atom(
                            cute.nvgpu.cpasync.CopyG2SOp(),
                            mX.element_type,
                            num_bits_per_copy=_fp4.COPY_BITS,
                        )
                        tiled_copy_load = cute.make_tiled_copy(
                            copy_atom_load_async, tv_layout, tiler_mn
                        )
                        thr_copy_X = tiled_copy_load.get_slice(tidx)
                        tXgX = thr_copy_X.partition_S(gX)
                        tXsX = thr_copy_X.partition_D(sX)
                        tXrX = cute.make_fragment_like(tXgX)

                    # Phase 1: the pool gather, from the senders' slots: vector v of a
                    # thread lies in sender owner(v)'s slot at word offset(v) + 4 * lane.
                    actual_row_idx = bidx * rows_per_block + row_in_block
                    if actual_row_idx < M:
                        word = Int64(lane_in_row * 4)
                        row64 = Int64(actual_row_idx)
                        piece = (row64 * self.pool_row_words + word) * 4
                        received = cute.make_rmem_tensor(cute.make_layout((4 * _VECTORS,)), Uint32)
                        start = Int64(0)
                        if cutlass.const_expr(self.guard_ns > 0):
                            start = _globaltimer()
                        pending = Uint32(1)
                        while pending != Uint32(0):
                            pending = Uint32(0)
                            for v in cutlass.range_constexpr(_VECTORS):
                                box = (
                                    records
                                    + Int64(self.owner(v) * record_bytes + self.offset(v) * 4)
                                    + piece
                                )
                                p0, p1, p2, p3 = _poll_x4(box)
                                received[4 * v] = p0
                                received[4 * v + 1] = p1
                                received[4 * v + 2] = p2
                                received[4 * v + 3] = p3
                                pending = pending | _pending(p0, p1, p2, p3)
                            if cutlass.const_expr(self.guard_ns > 0):
                                if pending != Uint32(0):
                                    if _globaltimer() - start > Int64(self.guard_ns):
                                        _set_guard_word(control + Int64(GUARD_WORD * 4))
                                        if cutlass.const_expr(self.guard_break):
                                            pending = Uint32(0)
                        gathered_row = mX.iterator.toint() + row64 * (H * 2)
                        for v in cutlass.range_constexpr(_VECTORS):
                            # Retire the consumed bytes before this grid can complete.
                            _reset_x4(
                                records
                                + Int64(self.owner(v) * record_bytes + self.offset(v) * 4)
                                + piece
                            )
                            w0 = received[4 * v]
                            w1 = received[4 * v + 1]
                            w2 = received[4 * v + 2]
                            w3 = received[4 * v + 3]
                            column = v * (2 * _VECTOR_SPAN_WORDS) + lane_in_row * 8
                            _store_shared_words(
                                sX.iterator + row_in_block * H + column, w0, w1, w2, w3
                            )
                            if cutlass.const_expr(store_gathered):
                                _store_global_words(
                                    gathered_row + Int64(column) * 2, w0, w1, w2, w3
                                )

                    # Phase 2: sum of squares (flashinfer's).
                    _bar_sync(QUANT_BARRIER, QUANT_THREADS)
                    cute.autovec_copy(tXsX, tXrX)
                    x = tXrX.load().to(Float32)
                    x_sq = x * x
                    sum_sq = _row_sum(x_sq, reduction_buffer)
                    mean_sq = sum_sq / H
                    rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
                    if cutlass.const_expr(not self.bf16_norm):
                        global_scale_val = mGlobalScale[0]
                    _bar_sync(QUANT_BARRIER, QUANT_THREADS)

                    if cutlass.const_expr(self.bf16_norm):
                        # Phase 3 (BF16 norm variant): pool_gather_route's bf16_norm phase 3 --
                        # flashinfer RMSNormKernel's pass 2 at this width (gemma, weight_bias 1).
                        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
                        copy_atom_y = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            mY.element_type,
                            num_bits_per_copy=_fp4.COPY_BITS,
                        )
                        thr_copy_O = cute.make_tiled_copy(
                            copy_atom_y, tv_layout, tiler_mn
                        ).get_slice(tidx)
                        tXgO = thr_copy_O.partition_D(gY)
                        tXrO = cute.make_fragment_like(tXgO)
                        cute.autovec_copy(tXsX, tXrX)
                        xn = tXrX.load().to(Float32)
                        w = tXrW.load().to(Float32)
                        y = xn * rstd * (w + Float32(1.0))
                        tXrO.store(y.to(mY.element_type))
                        if actual_row_idx < M:
                            cute.copy(copy_atom_y, tXrO, tXgO)
                    # Phase 3: RMSNorm + quantize (flashinfer's).
                    if cutlass.const_expr(not self.bf16_norm):
                        if actual_row_idx < M:
                            num_sf_per_thread = (
                                num_sf_blocks_per_row + threads_per_row - 1
                            ) // threads_per_row
                            for sf_iter in range(num_sf_per_thread):
                                sf_idx = lane_in_row + sf_iter * threads_per_row
                                if sf_idx < num_sf_blocks_per_row:
                                    block_start = sf_idx * block_size

                                    x_h2 = cute.make_rmem_tensor((8,), Uint32)
                                    w_h2 = cute.make_rmem_tensor((8,), Uint32)
                                    x_tile = sX.iterator + row_in_block * H + block_start
                                    x_h2[0], x_h2[1], x_h2[2], x_h2[3] = _load_shared_words(x_tile)
                                    x_h2[4], x_h2[5], x_h2[6], x_h2[7] = _load_shared_words(
                                        x_tile + 8
                                    )
                                    w_ptr0 = _fp4.get_ptr_as_int64(mW, block_start)
                                    w_ptr1 = _fp4.get_ptr_as_int64(mW, block_start + Int32(8))
                                    w_h2[0], w_h2[1], w_h2[2], w_h2[3] = _fp4.ld_global_v4_u32(
                                        w_ptr0
                                    )
                                    w_h2[4], w_h2[5], w_h2[6], w_h2[7] = _fp4.ld_global_v4_u32(
                                        w_ptr1
                                    )

                                    xw_h2 = _fp4.bfloat2_mul_8(x_h2, w_h2)
                                    max_xw_h2 = _fp4.bfloat2_max_abs_8(xw_h2)
                                    max_xw = _fp4.bfloat2_hmax_to_f32(max_xw_h2)
                                    y_f32 = _fp4.bfloat2_to_float16(xw_h2, rstd)

                                    max_abs = max_xw * rstd

                                    scale_float = global_scale_val * max_abs * fp4_max_rcp
                                    scale_float = _fp4.fmin_f32(
                                        scale_float, Float32(_fp4.FLOAT8_E4M3_MAX)
                                    )

                                    scale_fp8_u32 = _fp4.cvt_f32_to_e4m3(scale_float)
                                    scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))

                                    inv_scale = (
                                        _fp4.fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)
                                        * global_scale_val
                                    )

                                    mS[actual_row_idx, sf_idx] = scale_fp8

                                    packed64 = _fp4.quantize_and_pack_16(y_f32, inv_scale)

                                    out_offset = block_start // 2
                                    out_ptr = _fp4.get_ptr_as_int64(
                                        mY, actual_row_idx * (H // 2) + out_offset
                                    )
                                    _fp4.st_global_u64(out_ptr, packed64)
                    if cutlass.const_expr(self.stamps != 0):
                        if tidx == 0:
                            _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 32))
                    # Early ids with an early trigger: the routing warps' trigger (after the
                    # logit poll) is the CTA's only one, since the first trigger in a
                    # CTA counts and these warps can finish first.
                    if cutlass.const_expr(self.trigger == 0):
                        if enable_pdl:
                            cute.arch.griddepcontrol_launch_dependents()
                else:
                    # ======================================================
                    # Warps 4-7: the logit boxes of this CTA (one per sender),
                    # the routing of all M rows (warp 4 + w takes rows w and
                    # w + 4; lane l holds expert 32k + l of chunk k) and the
                    # placement.
                    # ======================================================
                    rt = tidx - QUANT_THREADS
                    rwarp = warp - QUANT_WARPS
                    # The expert bias is a parameter: read before the wait.
                    bias_k = [
                        _ld_global_b32(bias + Int64((32 * k + lane) * 4)) for k in range(CHUNKS)
                    ]
                    if enable_pdl:
                        cute.arch.griddepcontrol_wait()
                    if cutlass.const_expr(self.publish != 0):
                        # Early ids: every grid before this one has completed (the wait above
                        # returned), so the expert kernel's counter is final; its next
                        # launch uses counter + 1 (never 0). As in pool_gather_route.
                        pub_tag = _ld_relaxed_gpu_b32(Int64(self.epoch_word)) + Uint32(1)
                        pub_tag = _sel(_eq(pub_tag, Uint32(0)), Uint32(1), pub_tag)
                        if cutlass.const_expr(self.trigger == 2):
                            if enable_pdl:
                                cute.arch.griddepcontrol_launch_dependents()
                    if cutlass.const_expr(self.stamps != 0):
                        if rt == 0:
                            _stamp_timer(Int64(self.stamps) + Int64(bidx * 64))
                    if cutlass.const_expr(self.place != PLACE_NONE):
                        if rt < CHUNKS:
                            _st_shared_b32(sCensus.iterator + rt, Uint32(0))
                    if cutlass.const_expr(self.wide):
                        # The CTA's own rows of each sender's block: words 4 rt of its
                        # [rows_per_block, L] rows (fewer when its last row is past M).
                        word_limit = _min_u32(
                            Uint32(rows_per_block * local_count),
                            Uint32((M - bidx * rows_per_block) * local_count),
                        )
                        logits_start = Int64(self.pool_slot_words) + Int64(
                            bidx * rows_per_block * local_count
                        )
                    else:
                        word_limit = M * local_count
                        logits_start = Int64(self.pool_slot_words) + Int64(bidx * self.box_words)
                    if rt * 4 < word_limit:
                        word = Int64(rt * 4)
                        piece = (logits_start + word) * 4
                        received = cute.make_rmem_tensor(cute.make_layout((4 * world,)), Uint32)
                        start = Int64(0)
                        if cutlass.const_expr(self.guard_ns > 0):
                            start = _globaltimer()
                        pending = Uint32(1)
                        while pending != Uint32(0):
                            pending = Uint32(0)
                            for b in cutlass.range_constexpr(world):
                                box = records + Int64(b * record_bytes) + piece
                                q0, q1, q2, q3 = _poll_x4(box)
                                received[4 * b] = q0
                                received[4 * b + 1] = q1
                                received[4 * b + 2] = q2
                                received[4 * b + 3] = q3
                                pending = pending | _pending_words(q0, q1, q2, q3)
                            if cutlass.const_expr(self.guard_ns > 0):
                                if pending != Uint32(0):
                                    if _globaltimer() - start > Int64(self.guard_ns):
                                        _set_guard_word(control + Int64(GUARD_WORD * 4))
                                        if cutlass.const_expr(self.guard_break):
                                            pending = Uint32(0)
                        # This thread's words: row (4 rt) // L of the block, experts
                        # (4 rt) % L .. + 3 of each sender's L.
                        logit_row = (rt * 4) // local_count
                        logit_col = (rt * 4) % local_count
                        for b in cutlass.range_constexpr(world):
                            _reset_x4(records + Int64(b * record_bytes) + piece)
                            _store_shared_words(
                                sLogits.iterator
                                + logit_row * EXPERTS
                                + b * local_count
                                + logit_col,
                                received[4 * b],
                                received[4 * b + 1],
                                received[4 * b + 2],
                                received[4 * b + 3],
                            )
                    _bar_sync(ROUTE_BARRIER, ROUTE_THREADS)
                    if cutlass.const_expr(self.publish != 0):
                        # Early ids: this CTA's logits are in (the logit poll is complete).
                        if cutlass.const_expr(self.trigger == 1):
                            if enable_pdl:
                                cute.arch.griddepcontrol_launch_dependents()
                    if cutlass.const_expr(self.stamps != 0):
                        if rt == 0:
                            _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 8))
                    # This CTA's rows of the full logits: 64 threads per row.
                    full_row = bidx * rows_per_block + rt // 64
                    if full_row < M:
                        lr = full_row * EXPERTS + (rt % 64) * 4
                        if cutlass.const_expr(self.wide):
                            slr = (rt // 64) * EXPERTS + (rt % 64) * 4
                        else:
                            slr = lr
                        f0, f1, f2, f3 = _load_shared_words(sLogits.iterator + slr)
                        _store_global_words(logits_full + Int64(lr) * 4, f0, f1, f2, f3)
                    eps_bits = _f32_bits(renorm_eps)
                    # Each lane's keys of the warp's two rows, sorted; then the warp's
                    # top 8 of each row (key s in lane s). Wide: one row per routing warp,
                    # the CTA's own (warps 0 and 1; warps 2 and 3 route unread shared rows
                    # and store nothing).
                    halves = 1 if self.wide else 2
                    lists = []
                    for half in cutlass.range_constexpr(halves):
                        row = rwarp + 4 * half
                        keys = []
                        for k in cutlass.range_constexpr(CHUNKS):
                            expert = 32 * k + lane
                            logit = _ld_shared_b32(sLogits.iterator + row * EXPERTS + expert)
                            keys.append(
                                _route_key(_sigmoid(logit), bias_k[k], Uint32(255) - Uint32(expert))
                            )
                        lists.append(_sort8(keys))
                    picks, winners = _top8_by_redux(lists, lane)
                    for half in cutlass.range_constexpr(halves):
                        row = rwarp + 4 * half
                        if cutlass.const_expr(self.place != PLACE_NONE):
                            # The census, from the rounds' winners (warp-uniform): one
                            # lane sets each selected expert's bit (wide: of the CTA's own
                            # rows only; the grid's census is merged below).
                            if cutlass.const_expr(self.wide):
                                counts = (
                                    _eq(Uint32(lane), Uint32(0))
                                    & _lt_u(Uint32(row), Uint32(rows_per_block))
                                    & _lt_u(Uint32(bidx * rows_per_block + row), Uint32(M))
                                )
                            else:
                                counts = _eq(Uint32(lane), Uint32(0)) & _lt_u(
                                    Uint32(row), Uint32(M)
                                )
                            if counts != Uint32(0):
                                for s in cutlass.range_constexpr(TOP_K):
                                    chosen = Uint32(255) - winners[half][s]
                                    _red_shared_or(
                                        sCensus.iterator + (chosen >> 5),
                                        Uint32(1) << (chosen & Uint32(31)),
                                    )
                    for half in cutlass.range_constexpr(halves):
                        row = rwarp + 4 * half
                        mine = picks[half]
                        # Masked so that the unused rows' (unwritten) keys stay in bounds.
                        expert_id = (Uint32(255) - _lo32(mine)) & Uint32(255)
                        score = _sigmoid(
                            _ld_shared_b32(sLogits.iterator + row * EXPERTS + expert_id)
                        )
                        pair = _add_rn(score, _shfl_xor_b32(score, 1))
                        quad = _add_rn(pair, _shfl_xor_b32(pair, 4))
                        total = _add_rn(quad, _shfl_xor_b32(quad, 2))
                        weight = _div_rn(score, _add_rn(eps_bits, total))
                        if cutlass.const_expr(self.wide):
                            global_row = bidx * rows_per_block + row
                            keep = (
                                _lt_u(Uint32(row), Uint32(rows_per_block))
                                & _lt_u(Uint32(global_row), Uint32(M))
                                & _lt_u(Uint32(lane), Uint32(TOP_K))
                            )
                            if keep != Uint32(0):
                                if cutlass.const_expr(self.place != PLACE_NONE):
                                    # The CTA's own rows, at their CTA-relative rows.
                                    _st_shared_b32(
                                        sSelected.iterator + row * TOP_K + lane, expert_id
                                    )
                                    _st_shared_b32(sWeights.iterator + row * TOP_K + lane, weight)
                                out = Int64(global_row * TOP_K + lane) * 4
                                _st_global_b32(selected + out, expert_id)
                                _st_global_b32(weights + out, weight)
                        else:
                            keep = _lt_u(Uint32(row), Uint32(M)) & _lt_u(
                                Uint32(lane), Uint32(TOP_K)
                            )
                            if keep != Uint32(0):
                                _st_shared_b32(sSelected.iterator + row * TOP_K + lane, expert_id)
                                _st_shared_b32(sWeights.iterator + row * TOP_K + lane, weight)
                                if row // rows_per_block == bidx:
                                    out = Int64(row * TOP_K + lane) * 4
                                    _st_global_b32(selected + out, expert_id)
                                    _st_global_b32(weights + out, weight)

                    # Placement (replicated experts) from all M rows' selections, in every
                    # routing warp.
                    if cutlass.const_expr(self.place != PLACE_NONE):
                        _bar_sync(ROUTE_BARRIER, ROUTE_THREADS)
                        if cutlass.const_expr(self.wide):
                            # The grid's census: every CTA ORs its rows' bits into this
                            # rank's census words of the buffer, arrives, and waits for
                            # every CTA's arrival (all of the grid's CTAs are resident:
                            # at most 16); then each CTA reads the merged words. The
                            # last CTA to have read them clears them for the next call.
                            census = Int64(self.census) + Int64(buffer) * Int64(CENSUS_STRIDE * 4)
                            if rt < CHUNKS:
                                bits = _ld_shared_b32(sCensus.iterator + rt)
                                if bits != Uint32(0):
                                    _red_global_or(census + Int64(rt * 4), bits)
                            _bar_sync(ROUTE_BARRIER, ROUTE_THREADS)
                            ctas = Uint32((M + rows_per_block - 1) // rows_per_block)
                            if rt == 0:
                                _red_release_gpu_add(census + Int64(CENSUS_ARRIVED * 4), Uint32(1))
                                start = Int64(0)
                                if cutlass.const_expr(self.guard_ns > 0):
                                    start = _globaltimer()
                                arrived = _ld_acquire_gpu_b32(census + Int64(CENSUS_ARRIVED * 4))
                                while _lt_u(arrived, ctas) != Uint32(0):
                                    arrived = _ld_acquire_gpu_b32(
                                        census + Int64(CENSUS_ARRIVED * 4)
                                    )
                                    if cutlass.const_expr(self.guard_ns > 0):
                                        if _globaltimer() - start > Int64(self.guard_ns):
                                            _set_guard_word(control + Int64(GUARD_WORD * 4))
                                            if cutlass.const_expr(self.guard_break):
                                                arrived = ctas
                            _bar_sync(ROUTE_BARRIER, ROUTE_THREADS)
                            if rt < CHUNKS:
                                _st_shared_b32(
                                    sCensus.iterator + rt,
                                    _ld_relaxed_gpu_b32(census + Int64(rt * 4)),
                                )
                            _bar_sync(ROUTE_BARRIER, ROUTE_THREADS)
                            if rt == 0:
                                read = _atom_acq_rel_gpu_add(
                                    census + Int64(CENSUS_READ * 4), Uint32(1)
                                )
                                if _eq(read + Uint32(1), ctas) != Uint32(0):
                                    for word in cutlass.range_constexpr(CENSUS_WORDS):
                                        _st_relaxed_gpu_b32(census + Int64(word * 4), Uint32(0))
                        w0, w1, w2, w3 = _load_shared_words(sCensus.iterator)
                        w4, w5, w6, w7 = _load_shared_words(sCensus.iterator + 4)
                        words = [w0, w1, w2, w3, w4, w5, w6, w7]
                        # Per block: its chunk (TP8) or its chunk pair (TP4).
                        per = CHUNKS // world
                        loads = []
                        for b in cutlass.range_constexpr(world):
                            load_b = _popc(words[per * b])
                            for c in cutlass.range_constexpr(1, per):
                                load_b = load_b + _popc(words[per * b + c])
                            loads.append(load_b)
                        lane_r = Uint32(lane) & Uint32(world - 1)
                        x2s = None
                        if cutlass.const_expr(self.place == PLACE_BALANCE):
                            held = loads[0]
                            for b in cutlass.range_constexpr(1, world):
                                held = _sel(_eq(lane_r, Uint32(b)), loads[b], held)
                            arc = Uint32(0)
                            arcs = []
                            need = Uint32(0)
                            for k in cutlass.range_constexpr(world):
                                span = k + 1
                                arc = arc + _shfl_idx_b32(
                                    held, (lane_r - Uint32(k)) & Uint32(world - 1)
                                )
                                arcs.append(arc)
                                if cutlass.const_expr(span == world):
                                    need = _max_s32(need, _div_u32(arc + Uint32(world - 1), world))
                                else:
                                    need = _max_s32(need, _div_u32(arc + Uint32(span), span + 1))
                            t = _redux_max_s32(need)
                            moved_r = Uint32(0)
                            for k in cutlass.range_constexpr(world):
                                moved_r = _max_s32(moved_r, arcs[k] - Uint32(k + 1) * t)
                            xs = [_shfl_idx_b32(moved_r, Uint32(b)) for b in range(world)]
                        elif cutlass.const_expr(self.place == PLACE_HALF):
                            # pool_gather_route's half window (TP4): the two-window flow with
                            # T >= ceil((arc - movable[r]) / (k + 1)) for every arc ending at
                            # rank r, movable[r] the present experts of r's upper half.
                            held = loads[0]
                            for b in cutlass.range_constexpr(1, world):
                                held = _sel(_eq(lane_r, Uint32(b)), loads[b], held)
                            movable = _popc(words[1])
                            for b in cutlass.range_constexpr(1, world):
                                movable = _sel(
                                    _eq(lane_r, Uint32(b)), _popc(words[2 * b + 1]), movable
                                )
                            arc = Uint32(0)
                            arcs = []
                            need = Uint32(0)
                            for k in cutlass.range_constexpr(world):
                                span = k + 1
                                arc = arc + _shfl_idx_b32(
                                    held, (lane_r - Uint32(k)) & Uint32(world - 1)
                                )
                                arcs.append(arc)
                                if cutlass.const_expr(span == world):
                                    need = _max_s32(need, _div_u32(arc + Uint32(world - 1), world))
                                else:
                                    need = _max_s32(need, _div_u32(arc + Uint32(span), span + 1))
                                # arc >= load[r] >= movable[r]: no wrap.
                                need = _max_s32(
                                    need, _div_u32(arc - movable + Uint32(span - 1), span)
                                )
                            t = _redux_max_s32(need)
                            moved_r = Uint32(0)
                            for k in cutlass.range_constexpr(world):
                                moved_r = _max_s32(moved_r, arcs[k] - Uint32(k + 1) * t)
                            xs = [_shfl_idx_b32(moved_r, Uint32(b)) for b in range(world)]
                        elif cutlass.const_expr(self.place == PLACE_BALANCE3):
                            held = loads[0]
                            for b in cutlass.range_constexpr(1, world):
                                held = _sel(_eq(lane_r, Uint32(b)), loads[b], held)
                            arc = Uint32(0)
                            arcs = []
                            need = Uint32(0)
                            for k in cutlass.range_constexpr(world):
                                reach = min(k + 3, world)
                                arc = arc + _shfl_idx_b32(
                                    held, (lane_r - Uint32(k)) & Uint32(world - 1)
                                )
                                arcs.append(arc)
                                need = _max_s32(need, _div_u32(arc + Uint32(reach - 1), reach))
                            t = _redux_max_s32(need)
                            moved_r = Uint32(0)
                            for k in cutlass.range_constexpr(world):
                                moved_r = _max_s32(moved_r, arcs[k] - Uint32(k + 1) * t)
                            cs = [_shfl_idx_b32(moved_r, Uint32(b)) for b in range(world)]
                            xs = [_min_u32(cs[b], loads[b]) for b in range(world)]
                            x2s = [
                                _max_s32(Uint32(0), cs[(b + 1) % world] - loads[(b + 1) % world])
                                for b in range(world)
                            ]
                        elif cutlass.const_expr(self.place == _route.PLACE_STATIC):
                            if cutlass.const_expr(world == WORLD):
                                xs = [_popc(w >> self.static_from) for w in words]
                            else:
                                # Present experts at index static_from or above of each block.
                                xs = []
                                for b in cutlass.range_constexpr(world):
                                    count = Uint32(0)
                                    for c in cutlass.range_constexpr(per):
                                        shift = self.static_from - _route.CHUNK * c
                                        if shift <= 0:
                                            count = count + _popc(words[per * b + c])
                                        elif shift < _route.CHUNK:
                                            count = count + _popc(words[per * b + c] >> shift)
                                    xs.append(count)
                            moved_r = xs[0]
                            for b in cutlass.range_constexpr(1, world):
                                moved_r = _sel(_eq(lane_r, Uint32(b)), xs[b], moved_r)
                        else:
                            xs = [Uint32(0) for _ in range(world)]
                            moved_r = Uint32(0)
                        from_prev = _shfl_idx_b32(
                            moved_r, (lane_r + Uint32(world - 1)) & Uint32(world - 1)
                        )
                        # Not ``half``: the publication loop below binds that name to its row half.
                        place_half = self.place == PLACE_HALF
                        if cutlass.const_expr(self.publish != 0):
                            # Early ids: CTA 0 publishes every row's placed ids, before its own
                            # rows' stores, as in pool_gather_route.
                            for half in cutlass.range_constexpr(2):
                                prow = rwarp + 4 * half
                                pubs = _eq(Uint32(bidx), Uint32(0)) & _lt_u(
                                    Uint32(lane), Uint32(TOP_K)
                                )
                                if pubs != Uint32(0):
                                    pslot = prow * TOP_K + lane
                                    pg = _ld_shared_b32(sSelected.iterator + pslot)
                                    pplaced = _placed_expert(
                                        pg, words, xs, self.peer_rank, x2s, world, place_half
                                    )
                                    pplaced = _sel(
                                        _lt_u(Uint32(prow), Uint32(M)),
                                        pplaced,
                                        Uint32(0xFFFFFFFF),
                                    )
                                    _st_relaxed_gpu_pair(
                                        Int64(self.publish) + Int64(pslot) * 8, pplaced, pub_tag
                                    )
                            if cutlass.const_expr(self.stamps != 0):
                                if rt == 0:
                                    _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 16))
                        own_row = bidx * rows_per_block + rwarp
                        writes = (
                            _lt_u(Uint32(rwarp), Uint32(rows_per_block))
                            & _lt_u(Uint32(own_row), Uint32(M))
                            & _lt_u(Uint32(lane), Uint32(TOP_K))
                        )
                        if writes != Uint32(0):
                            slot_index = own_row * TOP_K + lane
                            # Wide: the CTA's rows sit at their CTA-relative rows.
                            shared_index = rwarp * TOP_K + lane if self.wide else slot_index
                            g = _ld_shared_b32(sSelected.iterator + shared_index)
                            placed = _placed_expert(
                                g, words, xs, self.peer_rank, x2s, world, place_half
                            )
                            _st_global_b32(local_ids + Int64(slot_index) * 4, placed)
                            w_bits = _ld_shared_b32(sWeights.iterator + shared_index)
                            _st_global_b16(weights_bf16 + Int64(slot_index) * 2, _bf16_bits(w_bits))
                        held_load = loads[0]
                        for b in cutlass.range_constexpr(1, world):
                            held_load = _sel(_eq(lane_r, Uint32(b)), loads[b], held_load)
                        reports = (
                            _eq(Uint32(bidx), Uint32(0))
                            & _eq(Uint32(rwarp), Uint32(2))
                            & _lt_u(Uint32(lane), Uint32(world))
                        )
                        if reports != Uint32(0):
                            _st_global_b32(stats + Int64(lane) * 4, held_load)
                            _st_global_b32(
                                stats + Int64(lane + world) * 4, held_load - moved_r + from_prev
                            )
                    if cutlass.const_expr(self.stamps != 0):
                        if rt == 0:
                            _stamp_timer(Int64(self.stamps) + Int64(bidx * 64 + 24))
                    if enable_pdl:
                        cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=64)
    def _compile(
        device_index: int,
        peer_rank: int,
        place: int,
        static_from: int,
        enable_pdl: bool,
        store_gathered: bool,
        publish: int,
        epoch_word: int,
        trigger: int,
        stamps: int,
        guard_ns: int,
        guard_break: bool,
        world: int = WORLD,
        max_rows: int = MAX_ROWS,
        census: int = 0,
        bf16_norm: bool = False,
    ):
        kernel = _PoolDirectRouteKernel(
            peer_rank,
            place,
            static_from,
            publish,
            epoch_word,
            trigger,
            stamps,
            guard_ns,
            guard_break,
            world,
            max_rows,
            census,
            bf16_norm,
        )
        rows = cute.sym_int()
        fake = cute.runtime.make_fake_compact_tensor
        # The BF16 norm variant's output is the [M, 4096] BF16 normalized row.
        out_fake = (
            fake(cutlass.BFloat16, (rows, HIDDEN), stride_order=(1, 0), assumed_align=128)
            if bf16_norm
            else fake(cutlass.Uint8, (rows, HIDDEN // 2), stride_order=(1, 0), assumed_align=128)
        )
        with torch.cuda.device(device_index):
            return cute.compile(
                kernel,
                fake(cutlass.BFloat16, (rows, HIDDEN), stride_order=(1, 0), assumed_align=128),
                fake(cutlass.BFloat16, (HIDDEN,), assumed_align=128),
                out_fake,
                fake(
                    cutlass.Uint8, (rows, HIDDEN // BLOCK), stride_order=(1, 0), assumed_align=128
                ),
                fake(cutlass.Float32, (1,), assumed_align=4),
                Int32(1),
                Float32(1e-6),
                Float32(1e-20),
                Int64(0),
                Int64(0),
                Int32(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int32(1),
                enable_pdl,
                store_gathered,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )


#: What selects a compiled kernel: ``(place, static_from, enable_pdl, store_gathered,
#: publish, epoch_word, trigger, stamps, bf16_norm)`` (the early-id settings are publish ..
#: stamps, 0 = off; ``bf16_norm`` the BF16 norm variant).
Variant = Tuple[int, int, bool, bool, int, int, int, int, bool]


def variant(
    place: int,
    static_from: int = LOCAL_EXPERTS,
    enable_pdl: bool = True,
    store_gathered: bool = True,
    early=None,
    stamps: int = 0,
    bf16_norm: bool = False,
) -> Variant:
    """The workspace's key of the kernel a call with these settings launches.

    ``early``: the ``EarlyExpertIds`` when this call publishes (the host decides
    per call, ``EarlyExpertIds.may_publish``), else None. ``bf16_norm``: the BF16 norm
    variant (phase 3 writes the gemma RMSNorm in BF16; no early ids).
    """
    if place not in (PLACE_NONE,) + tuple(_route.EARLY_PLACES):
        raise ValueError(f"unknown placement kind {place}")
    if bf16_norm and early is not None:
        raise ValueError("the BF16 norm variant has no early-id publication")
    publish = epoch = trigger = 0
    if early is not None:
        if place not in _route.EARLY_PLACES or not enable_pdl:
            raise ValueError(
                f"early_expert_ids publishes the placements {_route.EARLY_PLACES} with PDL on, got {place}"
            )
        publish, epoch, trigger = early.words_address, early.epoch_address, early.trigger
        stamps = stamps or early.stamps_address
    return (
        int(place),
        int(static_from),
        bool(enable_pdl),
        bool(store_gathered),
        int(publish),
        int(epoch),
        int(trigger),
        int(stamps),
        bool(bf16_norm),
    )


class PoolDirectWorkspace:
    """The direct transport of the routed pool kernel (``pool_into_route``) for one TP8 or TP4 group.

    ``slot(buffer, rows)`` is this rank's ``[rows, local_columns(world)]`` BF16 pool
    slot of ``buffer`` (512 columns at TP8, 1024 at TP4) and ``boxes(buffer, rows)``
    its ``[publish_copies(rows), box_words(world)]`` FP32 logit boxes (the 8-row
    layout) or ``block(buffer, rows)`` its ``[rows, L]`` FP32 logit block (the wide
    layout, ``max_rows=WIDE_MAX_ROWS``, TP4 or TP8), all on the multicast address: a producer
    given them as its output publishes into every rank's copy. ``kernel(variant)`` is the compiled routed
    kernel for a call's settings. Construction is collective over the group (fabric
    allocation and sentinel fill), and so is ``build_variants``; both succeed on
    every rank or raise on every rank. Nothing compiles inside a CUDA graph
    capture: a variant not built before is an error there.

    ``guard_ns`` is the poll guard (0: none); ``guard_break`` makes an expired guard
    stop polling (tests of a broken call sequence only). The guard word lives in
    pinned host memory, so the host reads it while the device still polls.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        guard_ns: int = DIRECT_GUARD_NS,
        guard_break: bool = False,
        max_rows: int = MAX_ROWS,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        if mapping.tp_size not in WORLDS or mapping.enable_attention_dp or mapping.cp_size != 1:
            raise ValueError(
                "the direct pool transport needs TP8 or TP4 without attention DP or CP"
            )
        check_layout(int(mapping.tp_size), int(max_rows))
        self.rank = mapping.tp_rank
        #: The group size and the row capacity, which fix the layout (``record_words``).
        self.world = int(mapping.tp_size)
        self.max_rows = int(max_rows)
        self.wide = self.max_rows > MAX_ROWS
        self.record_words = record_words(self.world, self.max_rows)
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
            refusal = (
                "CuTe DSL, flashinfer's RMSNormFP4QuantKernel or the direct poll is unavailable"
            )
        elif torch.cuda.get_device_capability(self.device) != (10, 3):
            refusal = "the direct routed pool kernel runs on SM103 only"
        elif not prefer_pinned():
            refusal = "the guard words need pinned host memory, which this platform does not pin"
        if not _mnnvl_workspace_all_succeeded(self._comm, refusal is None):
            raise RuntimeError(
                "direct pool transport refused on at least one rank"
                f"{f' (this rank: {refusal})' if refusal is not None else ''}"
            )
        error: Optional[BaseException] = None
        try:
            use_fabric = mapping.is_multi_node()
            nbytes = workspace_bytes(self.world, self.max_rows)
            self._handle = _make_mnnvl_mcast_buffer(self._comm, nbytes, mapping, use_fabric)
            words = nbytes // 4
            self._receive = self._handle.get_uc_buffer(self.rank, (words,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((words,), torch.int32, 0)
            self._records = self._mc.view(BUFFERS, self.world, self.record_words)
            with torch.cuda.device(self.device):
                self._receive.fill_(_SENTINEL_I32)
                # The wide layout's half-window census words (rank-local, zero between calls).
                self._census = (
                    torch.zeros(BUFFERS * CENSUS_STRIDE, dtype=torch.int32, device=self.device)
                    if self.wide
                    else None
                )
                self.control = torch.zeros(
                    CONTROL_WORDS, dtype=torch.int32, pin_memory=prefer_pinned()
                )
                status, pointer = _cudart.cudaHostGetDevicePointer(self.control.data_ptr(), 0)
                if status != _cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaHostGetDevicePointer of the guard words: {status}")
                self._control_ptr = int(pointer)
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "direct pool transport failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _record(self, buffer: int) -> torch.Tensor:
        if buffer not in range(BUFFERS):
            raise ValueError(f"no buffer {buffer}")
        return self._records[buffer, self.rank]

    def slot(self, buffer: int, rows: int) -> torch.Tensor:
        """This rank's ``[rows, local_columns(world)]`` BF16 pool slot of ``buffer``, on the multicast address.

        Write-only: a store through it reaches every rank's copy; nothing may read it.
        """
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"no slot for {rows} rows")
        record = self._record(buffer)
        return (
            record[: rows * pool_row_words(self.world)]
            .view(torch.bfloat16)
            .view(rows, local_columns(self.world))
        )

    def boxes(self, buffer: int, rows: int) -> torch.Tensor:
        """This rank's ``[publish_copies(rows), box_words(world)]`` FP32 logit boxes of ``buffer``.

        Multicast and write-only.
        """
        if self.wide or not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"no boxes for {rows} rows on a {self.max_rows}-row layout")
        copies = (rows + 1) // 2
        record = self._record(buffer)
        start, box = pool_slot_words(self.world), box_words(self.world)
        return record[start : start + copies * box].view(torch.float32).view(copies, box)

    def block(self, buffer: int, rows: int) -> torch.Tensor:
        """This rank's ``[rows, L]`` FP32 logit block of ``buffer`` on the wide layout (multicast; write-only)."""
        if not self.wide or not 1 <= rows <= self.max_rows:
            raise ValueError(f"no logit block for {rows} rows on a {self.max_rows}-row layout")
        local = _route.local_experts(self.world)
        record = self._record(buffer)
        start = pool_slot_words(self.world, self.max_rows)
        return record[start : start + rows * local].view(torch.float32).view(rows, local)

    def receive(self) -> torch.Tensor:
        """This rank's copy of the whole transport, ``int32 [3 * world * record_words(world)]`` (tests)."""
        return self._receive

    def control_ptr(self) -> int:
        """The device address of the guard words (pinned host memory)."""
        return self._control_ptr

    def is_clean(self) -> bool:
        """Whether every word of this rank's copy holds the sentinel and every census word
        is zero (synchronizes; tests)."""
        torch.cuda.synchronize(self.device)
        clean = bool((self._receive == _SENTINEL_I32).all().item())
        if self._census is not None:
            clean = clean and not bool(self._census.any().item())
        return clean

    def reset(self) -> None:
        """Every word of this rank's copy back to the sentinel (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self._receive.fill_(_SENTINEL_I32)
        if self._census is not None:
            self._census.zero_()
        torch.cuda.synchronize(self.device)

    def guard_fired(self, sync: bool = True) -> bool:
        """Whether a poll of this workspace ever outlasted ``guard_ns``."""
        if sync:
            torch.cuda.synchronize(self.device)
        return int(self.control[GUARD_WORD]) != 0

    def clear_guard(self) -> None:
        """Zero the guard words (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self.control.zero_()

    def start_guard_watch(self, interval_s: float = GUARD_WATCH_S, abort: bool = False) -> bool:
        """Read the guard word from a host thread every ``interval_s`` seconds (0: no thread;
        see ``UnpoolDirectWorkspace.start_guard_watch``); ``abort`` aborts. Idempotent."""
        if self._watch is not None:
            return True
        if interval_s <= 0:
            return False
        stop = threading.Event()

        def watch():
            while not stop.wait(interval_s):
                if self.guard_fired(sync=False):
                    logger.error(
                        f"[musespark_open] pool_into_route: rank {self.rank} polled the direct pool transport for "
                        f"more than {self.guard_ns / 1e9:.1f} s without a record arriving: the "
                        "ranks' call sequences differ, or a peer stopped. The routed kernel "
                        "keeps polling, so this rank makes no progress"
                        + ("; aborting" if abort else "")
                    )
                    if abort:
                        os.abort()
                    return

        self._watch_stop = stop
        self._watch = threading.Thread(
            target=watch, name="musespark_open.pool_direct_watchdog", daemon=True
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
                "direct routed pool kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def built(self, key: Variant) -> bool:
        return key in self._kernels

    def kernel(self, key: Variant):
        """The compiled routed kernel for ``key`` (see ``variant``), built on first use outside a capture."""
        compiled = self._kernels.get(key)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"[musespark_open] pool_into_route: direct routed pool kernel variant {key} was not built before "
                "graph capture"
            )
        (
            place,
            static_from,
            enable_pdl,
            store_gathered,
            publish,
            epoch,
            trigger,
            stamps,
            bf16_norm,
        ) = key
        compiled = _compile(
            self.device.index,
            self.rank,
            place,
            static_from,
            enable_pdl,
            store_gathered,
            publish,
            epoch,
            trigger,
            stamps,
            self.guard_ns,
            self.guard_break,
            self.world,
            self.max_rows,
            self._census.data_ptr() if self.wide and place != PLACE_NONE else 0,
            bf16_norm,
        )
        # A dry launch loads the module now, not inside a capture.
        with torch.cuda.device(self.device):
            outputs = _outputs(
                1, self.device, place != PLACE_NONE, store_gathered, self.world, bf16_norm
            )
            gain = torch.zeros(HIDDEN, dtype=torch.bfloat16, device=self.device)
            scale = torch.zeros(1, dtype=torch.float32, device=self.device)
            _launch(compiled, outputs, gain, scale, 1, 1e-6, 1e-20, 0, 0, 0, 0, dry=1)
            torch.cuda.synchronize(self.device)
        self._kernels[key] = compiled
        return compiled


def _outputs(
    rows: int,
    device,
    placed: bool,
    store_gathered: bool,
    world: int = WORLD,
    bf16_norm: bool = False,
) -> _route.RouteOutputs:
    """A call's outputs; the BF16 norm variant writes ``normed`` in place of ``fp4``."""
    return _route.RouteOutputs(
        fp4=None
        if bf16_norm
        else torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device),
        normed=torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        if bf16_norm
        else None,
        scale=torch.empty((rows, HIDDEN // BLOCK), dtype=torch.uint8, device=device),
        gathered=torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device),
        logits=torch.empty((rows, EXPERTS), dtype=torch.float32, device=device),
        selected=torch.empty((rows, TOP_K), dtype=torch.int32, device=device),
        weights=torch.empty((rows, TOP_K), dtype=torch.float32, device=device),
        local_ids=torch.empty((rows, TOP_K), dtype=torch.int32, device=device) if placed else None,
        weights_bf16=(
            torch.empty((rows, TOP_K), dtype=torch.bfloat16, device=device) if placed else None
        ),
        stats=torch.empty((2 * world,), dtype=torch.int32, device=device) if placed else None,
    )


def _launch(
    compiled,
    out: _route.RouteOutputs,
    gain,
    global_scale,
    rows: int,
    eps: float,
    renorm_eps: float,
    receive: int,
    control: int,
    buffer: int,
    bias: int,
    dry: int = 0,
) -> None:
    placed = out.local_ids is not None
    compiled(
        out.gathered,
        gain,
        out.fp4 if out.fp4 is not None else out.normed,
        out.scale,
        global_scale,
        rows,
        eps,
        renorm_eps,
        receive,
        control,
        buffer,
        bias,
        out.logits.data_ptr(),
        out.selected.data_ptr(),
        out.weights.data_ptr(),
        out.local_ids.data_ptr() if placed else 0,
        out.weights_bf16.data_ptr() if placed else 0,
        out.stats.data_ptr() if placed else 0,
        dry,
    )


_SHARED: dict = {}


def shared_workspace(mapping: Mapping, max_rows: int = MAX_ROWS) -> PoolDirectWorkspace:
    """The direct pool workspace of this group and row capacity, built on first use (collective)."""
    key = (mapping, int(max_rows))
    workspace = _SHARED.get(key)
    if workspace is None:
        workspace = PoolDirectWorkspace(mapping, max_rows=max_rows)
        _SHARED[key] = workspace
    return workspace


class PoolDirectRoute(OpWrapper):
    """``pool_gather_route`` on pool slices and logits their producers already published.

    The pool GEMM wrote this rank's ``workspace.slot(buffer, rows)`` and the router
    shard ``workspace.boxes(buffer, rows)`` (``router_shard_publish``) on every rank
    before this call (or will: the op polls). ``gain`` / ``global_scale`` / ``eps``
    / ``bias`` / ``renorm_eps`` / ``place`` / ``static_from`` / ``early`` /
    ``stamps`` are ``pool_gather_route``'s. Returns its ``RouteOutputs`` for the
    same slices and logits.

    A collective over the TP8 or TP4 group: every rank calls it with the same row count
    and buffer, in the same order relative to its other calls on ``workspace``,
    and no two consecutive calls on one buffer (``buffer_sequence``). Launched
    with programmatic dependent launch.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DFlash2 K=7 verify: 8 rows, balanced placement, buffer 0",
            spec=dict(rows=8, place=_route.PLACE_BALANCE, buffer=0),
        ),
        Cell(
            why="8 rows over three ring windows, the second buffer",
            spec=dict(rows=8, place=_route.PLACE_BALANCE3, buffer=1),
        ),
        Cell(
            why="the third buffer (the last call of an odd count)",
            spec=dict(rows=8, place=_route.PLACE_BALANCE, buffer=2),
        ),
        Cell(
            why="8 rows, routing only (no placement)", spec=dict(rows=8, place=PLACE_NONE, buffer=0)
        ),
        Cell(
            why="1 row: one CTA, its second row idle, one logit box",
            spec=dict(rows=1, place=_route.PLACE_BALANCE, buffer=1),
        ),
        Cell(
            why="3 rows: the second CTA carries one row",
            spec=dict(rows=3, place=_route.PLACE_BALANCE, buffer=0),
        ),
        Cell(
            why="8 rows, owner-only placement",
            spec=dict(rows=8, place=_route.PLACE_OWNER, buffer=1),
        ),
        Cell(
            why="8 rows, static placement",
            spec=dict(rows=8, place=_route.PLACE_STATIC, buffer=0),
        ),
        Cell(
            why="tep4 wide layout: two requests' K=7 verify (16 rows), routing only",
            spec=dict(rows=16, place=PLACE_NONE, buffer=0, wide=True),
        ),
        Cell(
            why="tep4 wide layout: three requests (24 rows), the second buffer",
            spec=dict(rows=24, place=PLACE_NONE, buffer=1, wide=True),
        ),
        Cell(
            why="tep4 wide layout: four requests (32 rows), the third buffer",
            spec=dict(rows=32, place=PLACE_NONE, buffer=2, wide=True),
        ),
        Cell(
            why="tep4 wide layout: an odd row count (19 rows), the last CTA one row",
            spec=dict(rows=19, place=PLACE_NONE, buffer=0, wide=True),
        ),
        Cell(
            why="tep4 wide layout: 16 rows on the half window, the grid's census",
            spec=dict(rows=16, place=PLACE_HALF, buffer=1, wide=True),
        ),
        Cell(
            why="tep4 wide layout: 32 rows on the half window (16 CTAs at the census)",
            spec=dict(rows=32, place=PLACE_HALF, buffer=0, wide=True),
        ),
        Cell(
            why="tep4 wide layout: 19 rows on the half window, the last CTA one row",
            spec=dict(rows=19, place=PLACE_HALF, buffer=2, wide=True),
        ),
    )

    note = """
    Certified against `pool_gather_route` on the same slices and logits bit for bit
    (the test publishes with the production split-K pool GEMM and the router's
    publish twin, and with the site policy's copy kernel), through CUDA-graph
    replays with changing inputs, back-to-back 60-layer graphs with in-graph rank
    delays and a delayed rank; a concurrent poller checks that every half of a pool
    slot and every logit word is first seen with its final value.
    """

    def __call__(
        self,
        workspace: PoolDirectWorkspace,
        buffer: int,
        rows: int,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        bias: torch.Tensor,
        renorm_eps: float,
        place: int = PLACE_NONE,
        static_from: int = LOCAL_EXPERTS,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        early=None,
        stamps: Optional[torch.Tensor] = None,
        bf16_norm: Optional[torch.Tensor] = None,
    ) -> _route.RouteOutputs:
        placed = place != PLACE_NONE
        # ``bf16_norm`` (the gemma norm's raw [4096] BF16 weight): the kernel writes the
        # normalized BF16 row into ``normed`` and no NVFP4 payload or scale.
        norm = bf16_norm is not None
        publish = (
            early is not None and not norm and placed and bool(enable_pdl) and early.may_publish()
        )
        key = variant(
            int(place),
            int(static_from),
            bool(enable_pdl),
            bool(return_gathered),
            early if publish else None,
            0 if stamps is None else stamps.data_ptr(),
            norm,
        )
        compiled = workspace.kernel(key)
        out = _outputs(rows, gain.device, placed, True, workspace.world, norm)
        _launch(
            compiled,
            out,
            gain if not norm else bf16_norm,
            global_scale,
            rows,
            eps,
            renorm_eps,
            workspace.receive().data_ptr(),
            workspace.control_ptr(),
            int(buffer),
            bias.data_ptr(),
        )
        if publish:
            early.arm(out.local_ids)
        if not return_gathered:
            out.gathered = None
        if norm:
            out.scale = None
        return out

    def reference(
        self,
        per_rank_locals: Sequence[torch.Tensor],
        per_rank_logits: Sequence[torch.Tensor],
        workspace: Optional[PoolDirectWorkspace],
        buffer: int,
        rows: int,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        bias: torch.Tensor,
        renorm_eps: float,
        place: int = PLACE_NONE,
        static_from: int = LOCAL_EXPERTS,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        rank: int = 0,
    ) -> _route.RouteOutputs:
        """``pool_gather_route.reference`` on every rank's published slice and logits.

        The collective exception to mirroring: ``per_rank_locals`` / ``per_rank_logits``
        are every rank's published operands in rank order; ``workspace``, ``buffer``
        and ``enable_pdl`` are ignored (where the operands travel changes no value).
        """
        del workspace, buffer, rows
        return _route.pool_gather_route.reference(
            per_rank_locals,
            per_rank_logits,
            None,
            gain,
            global_scale,
            eps,
            bias,
            renorm_eps,
            place,
            static_from,
            enable_pdl,
            return_gathered,
            rank,
        )

    def is_valid(
        self,
        workspace: PoolDirectWorkspace,
        buffer: int,
        rows: int,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        bias: torch.Tensor,
        renorm_eps: float,
        place: int = PLACE_NONE,
        static_from: int = LOCAL_EXPERTS,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        early=None,
        stamps: Optional[torch.Tensor] = None,
        bf16_norm: Optional[torch.Tensor] = None,
    ) -> None:
        """Raw vector accesses at fixed strides on the operands' device; the transport's
        buffer and row count must be the ones the producers published into. The wide
        layout routes only (``PLACE_NONE``, no early ids)."""
        if not 1 <= rows <= workspace.max_rows:
            raise ValueError(f"rows must be in 1..{workspace.max_rows}, got {rows}")
        if workspace.wide and (
            place not in (PLACE_NONE, PLACE_HALF)
            or early is not None
            or (place == PLACE_HALF and workspace.world != 4)
        ):
            raise ValueError(
                "the wide layout routes only or places the half window (PLACE_NONE or PLACE_HALF), "
                "without early ids"
            )
        if buffer not in range(BUFFERS):
            raise ValueError(f"buffer must be in 0..{BUFFERS - 1}, got {buffer}")
        if not enable_pdl and early is not None:
            raise ValueError("the early-id publication needs programmatic dependent launch")
        checked = [
            (gain, (HIDDEN,), torch.bfloat16),
            (global_scale, (1,), torch.float32),
            (bias, (EXPERTS,), torch.float32),
        ]
        if bf16_norm is not None:
            checked.append((bf16_norm, (HIDDEN,), torch.bfloat16))
        for tensor, shape, dtype in checked:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
                or tensor.device != workspace.device
            ):
                raise ValueError(
                    f"expected a contiguous 16-byte aligned {dtype} {shape} tensor on "
                    f"{workspace.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )
        if place not in (PLACE_NONE,) + tuple(_route.EARLY_PLACES):
            raise ValueError(f"unknown placement kind {place}")
        if place not in _route.PLACES[workspace.world]:
            raise ValueError(f"placement kind {place} is not served at TP{workspace.world}")
        local = _route.local_experts(workspace.world)
        if not 0 <= static_from <= local:
            raise ValueError(f"static_from must be in [0, {local}]")
        if stamps is not None and (
            stamps.dtype != torch.int64
            or not stamps.is_contiguous()
            or stamps.device != workspace.device
            or stamps.numel() < 8 * ((rows + 1) // 2)
        ):
            raise ValueError("stamps must be a contiguous int64 tensor of 8 words per CTA")


pool_direct_route = PoolDirectRoute()


def available() -> bool:
    """Whether this image can build the kernel (CuTe DSL, flashinfer's kernel, the direct poll)."""
    return _AVAILABLE
