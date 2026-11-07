# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Wan VAE position-walk kernel in CuTe DSL.

One thread group (``lanes`` lanes, ``k`` 16-byte vectors each) owns one spatial position of a
batch-1 channels-last video (``[T, P, C]``, rows contiguous) and walks its frames in order.
Three compile-time stages make it every norm-side kernel of WinoVae:

  input   the x frames, or the temporal output transform of ``order + 2`` conv products (frame
          ``g*R + j`` is ``sum_k AT[j][k] m_k[g]``, index order), then the deferred conv bias and
          the residual; the pre-norm row goes to ``sum`` when the variant writes it.
  middle  nothing (the add_bias_and_residual kernel), RMSNorm, or RMSNorm + SiLU.
  output  the temporal Winograd inputs of order R (``D_k = sum_i BT[k][i] f_i`` over the group's
          R + 2 normalized frames, two of them the cache), the last two normalized frames going
          to ``y_tail`` for the next chunk; or plain rows to ``y`` (``[T, P', C]``, ``P'`` the
          padded position count when the downsampler's zero row and column are appended).

Numerics: fp32 arithmetic with these bf16 roundings. Pass 1 (input stage) leaves the pre-norm
row rounded to bf16: that is what ``sum`` holds and what pass 2 normalizes. Row output and
``reference`` also round after the transform + deferred bias and after the residual; the
Winograd output otherwise rounds the row only at that one point, so its sum of squares
accumulates the unrounded fp32 values while pass 2 normalizes the rounded ones. Pass 2 rounds
each normalized value to bf16 (the output rows and the frames the Winograd inputs are built
from), and each Winograd input once more to its 16-bit type. ``reference`` reproduces the
eager bf16 rounding after every add of the output transform and after every normalize,
multiply and bias step. The sum of squares accumulates element by element with fma; the
Winograd sums run in table index order.
"""

from __future__ import annotations

import functools
import inspect

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op

from ..jit_norm import BLOCK_THREADS, MID_NONE, MID_NORM_SILU, OUT_ROWS, OUT_WINOGRAD, Variant
from .common import (
    CHANNELS_PER_VECTOR,
    WORDS_PER_VECTOR,
    bf16_hi,
    bf16_lo,
    compile_cached,
    current_stream,
    load_words,
    pack2,
    round_pair_bf16,
    store_words,
    u32_view,
    unpack_bf16,
    vec_view,
)

# The three per-channel tables in shared memory, in this order.
_GAMMA, _BIAS, _IN_BIAS = 0, 1, 2


@dsl_user_op
def _opaque(x: Int32, *, loc=None, ip=None) -> Int32:
    """``x`` through a register move the compiler cannot see through. Shared-memory reads
    addressed from it stay where they are written: the per-channel tables are re-read in every
    frame and the ring frames in every Winograd input instead of being hoisted out of the loop
    or merged, which would keep dozens of values live and spill them to local memory."""
    from cutlass._mlir import ir

    res = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [Int32(x).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@functools.lru_cache(maxsize=None)
def _tables(order: int) -> tuple[list[list[float]], list[list[float]]]:
    from ..wino_vae_kernels import TEMPORAL_WINO

    # Order 1 (plain row input and output) has no temporal transform and never reads the
    # tables; it gets the order-2 ones so the plan's table-derived fields stay defined.
    w = TEMPORAL_WINO[max(order, 2)]
    return w["BT"], w["AT"]


def _blocks_per_sm(v: Variant) -> int:
    """Resident blocks per SM the register budget is sized for (256 threads: 4 blocks give 64
    registers, 3 give 80, 2 give 128).

    Row output: 2 blocks for products at four or more vectors per lane on at most 8 lanes
    (160 and 320 channels; the order-4 output transform holds five products per slice and
    spills at 80 registers); 4 for plain input at four or more vectors per lane on 4 lanes
    (160 channels, latency-bound at 3); otherwise 4 up to two vectors per lane and 3 above.
    Winograd output: 3 for the order-4 producer from plain rows at up to two vectors per lane
    (fits 80 registers and hides more load latency); otherwise 2 for order 4 (six inputs per
    frame) or four and more vectors per lane; otherwise 4 for plain rows at up to two vectors
    per lane and 3 for products or three vectors per lane."""
    if v.out == OUT_ROWS:
        if v.in_wino and v.k >= 4 and v.lanes <= 8:
            return 2
        if not v.in_wino and v.k >= 4 and v.lanes == 4:
            return 4
        return 4 if v.k <= 2 else 3
    if v.order == 4 and not v.in_wino and v.k <= 2:
        return 3
    if v.k >= 4 or v.order == 4:
        return 2
    return 4 if (not v.in_wino and v.k <= 2) else 3


def _slice_depth(v: Variant) -> int:
    """Vector slices in flight ahead of the one being processed: three for four or more
    vectors per lane, two otherwise, except where two measured faster at four or more: the
    output transform of products into rows on 16 or more lanes (640 channels, and 1024 and
    up), and the 1024-channel row norm of plain input with a residual (two slices plus the
    cross-frame prefetch of the next frame's first two slices hide the doubled loads better)."""
    if v.out == OUT_ROWS and v.in_wino and v.k >= 4 and v.lanes >= 16:
        return 2
    if (
        v.out == OUT_ROWS
        and not v.in_wino
        and v.has_residual
        and v.mid != MID_NONE
        and v.k == 4
        and v.lanes == 32
    ):
        return 2
    return 3 if v.k >= 4 else 2


def _cross_frame_prefetch(v: Variant, depth: int) -> bool:
    """Whether the next frame's first two slices are loaded while this frame is normalized.
    Never at one vector per lane (whole frames are prefetched there). Plain input: at slice
    depth 2. Products: their slices hold up to five vectors each, so only the order-2
    Winograd output at 1024 channels (three products per slice, 128 registers) does it, where
    it measured faster."""
    if v.k == 1:
        return False
    if v.in_wino:
        return v.out == OUT_WINOGRAD and v.order == 2 and v.k == 4 and v.lanes == 32
    return depth == 2


class _Plan:
    """Compile-time constants of one variant."""

    def __init__(self, v: Variant, has_sum: bool) -> None:
        self.v = v
        self.has_sum = has_sum
        self.order = v.order
        self.lanes = v.lanes
        self.vecs_per_lane = v.k
        self.row_vecs = v.lanes * v.k
        self.channels = self.row_vecs * CHANNELS_PER_VECTOR
        self.row_dtype = BFloat16
        self.winograd_dtype = Float16 if v.fp16_winograd else BFloat16
        self.in_dtype = self.winograd_dtype if v.in_wino else self.row_dtype
        self.bt, self.at = _tables(self.order)
        self.components = self.order + 2
        # Nonzero terms of each AT row: the products frame j of a group is built from.
        self.terms = [
            [i for i in range(self.components) if self.at[j][i] != 0] for j in range(self.order)
        ]
        self.max_terms = max(len(t) for t in self.terms) if v.in_wino else 1
        # Winograd input s is complete once frame last_frame[s] of the group is normalized.
        self.last_frame = [
            max(i for i in range(self.components) if self.bt[s][i] != 0)
            for s in range(self.components)
        ]
        self.winograd_out = v.out == OUT_WINOGRAD
        self.silu = v.mid == MID_NORM_SILU
        # Round the pre-norm row to bf16 after the transform + bias and after the residual.
        self.round_each_add = v.reference or v.out == OUT_ROWS
        # One vector per lane: whole frames are prefetched and the per-channel tables stay in
        # registers. Otherwise slices are pipelined ``slice_depth`` deep.
        self.one_vector_per_lane = v.k == 1
        self.slice_depth = _slice_depth(v)
        self.cross_frame_prefetch = _cross_frame_prefetch(v, self.slice_depth)
        self.next_frame_prefetched = self.one_vector_per_lane or self.cross_frame_prefetch
        self.xor_offsets = [1 << b for b in reversed(range(self.lanes.bit_length() - 1))]
        self.positions_per_warp = 32 // self.lanes
        self.positions_per_block = v.positions_per_block
        self.ring_slots = v.ring_slots
        self.blocks_per_sm = _blocks_per_sm(v)
        self.smem_bytes = v.shared_bytes(self.channels)


class _Ctx:
    """Trace-time bundle of the kernel's views, scalars and register buffers."""


class VaeNormKernel:
    def __init__(self, variant: Variant, has_sum: bool) -> None:
        self.p = _Plan(variant, has_sum)

    @cute.jit
    def wrapper(
        self,
        x_ptr: cute.Pointer,
        m_ptr: cute.Pointer,
        d_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        in_bias_ptr: cute.Pointer,
        cache_ptr: cute.Pointer,
        gamma_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        sum_ptr: cute.Pointer,
        y_tail_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        halo_l_ptr: cute.Pointer,
        halo_r_ptr: cute.Pointer,
        comp_vecs: Int64,
        in_comp_vecs: Int64,
        in_frame: Int64,
        out_frame: Int64,
        halo_frame_l: Int64,
        halo_frame_r: Int64,
        halo_comp_l: Int64,
        halo_comp_r: Int64,
        positions: Int32,
        frames: Int32,
        cache_frames: Int32,
        height: Int32,
        width: Int32,
        pad_h: Int32,
        pad_w: Int32,
        in_w: Int32,
        in_off: Int32,
        out_w: Int32,
        out_off: Int32,
        halo_step_l: Int32,
        halo_step_r: Int32,
        halo_axis: Int32,
        has_l: Int32,
        has_r: Int32,
        scale: Float32,
        eps: Float32,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x_ptr,
            m_ptr,
            d_ptr,
            residual_ptr,
            in_bias_ptr,
            cache_ptr,
            gamma_ptr,
            bias_ptr,
            sum_ptr,
            y_tail_ptr,
            y_ptr,
            halo_l_ptr,
            halo_r_ptr,
            comp_vecs,
            in_comp_vecs,
            in_frame,
            out_frame,
            halo_frame_l,
            halo_frame_r,
            halo_comp_l,
            halo_comp_r,
            positions,
            frames,
            cache_frames,
            height,
            width,
            pad_h,
            pad_w,
            in_w,
            in_off,
            out_w,
            out_off,
            halo_step_l,
            halo_step_r,
            halo_axis,
            has_l,
            has_r,
            scale,
            eps,
        ).launch(
            grid=[blocks, 1, 1],
            block=[BLOCK_THREADS, 1, 1],
            smem=self.p.smem_bytes,
            stream=stream,
            min_blocks_per_mp=self.p.blocks_per_sm,
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        m_ptr: cute.Pointer,
        d_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        in_bias_ptr: cute.Pointer,
        cache_ptr: cute.Pointer,
        gamma_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        sum_ptr: cute.Pointer,
        y_tail_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        halo_l_ptr: cute.Pointer,
        halo_r_ptr: cute.Pointer,
        comp_vecs: Int64,
        in_comp_vecs: Int64,
        in_frame: Int64,
        out_frame: Int64,
        halo_frame_l: Int64,
        halo_frame_r: Int64,
        halo_comp_l: Int64,
        halo_comp_r: Int64,
        positions: Int32,
        frames: Int32,
        cache_frames: Int32,
        height: Int32,
        width: Int32,
        pad_h: Int32,
        pad_w: Int32,
        in_w: Int32,
        in_off: Int32,
        out_w: Int32,
        out_off: Int32,
        halo_step_l: Int32,
        halo_step_r: Int32,
        halo_axis: Int32,
        has_l: Int32,
        has_r: Int32,
        scale: Float32,
        eps: Float32,
    ):
        p = self.p
        c = _Ctx()
        self._global_views(
            c,
            x_ptr,
            m_ptr,
            d_ptr,
            residual_ptr,
            cache_ptr,
            sum_ptr,
            y_tail_ptr,
            y_ptr,
            halo_l_ptr,
            halo_r_ptr,
        )
        self._channel_tables(c, gamma_ptr, bias_ptr, in_bias_ptr, scale)
        self._frame_geometry(
            c,
            positions,
            frames,
            height,
            width,
            pad_h,
            pad_w,
            in_w,
            in_off,
            out_w,
            out_off,
            in_frame,
            out_frame,
            comp_vecs,
            in_comp_vecs,
            scale,
        )
        self._halo_geometry(
            c,
            halo_frame_l,
            halo_frame_r,
            halo_comp_l,
            halo_comp_r,
            halo_step_l,
            halo_step_r,
            halo_axis,
            has_l,
            has_r,
        )
        self._register_buffers(c)

        prefetched = cutlass.Boolean(False)
        src_in = Int64(0)
        src_res = Int64(0)

        bid, _, _ = cute.arch.block_idx()
        out_pos = Int32(bid) * p.positions_per_block + c.pos_in_block
        while out_pos < c.positions_out:
            pos = self._source_pos(c, out_pos)
            pos_src = self._in_index(c, pos)
            pos_out = self._out_index(c, out_pos)
            halo = self._halo_targets(c, pos)

            is_pad = cutlass.Boolean(False)
            if cutlass.const_expr(p.v.padded):
                is_pad = pos < 0
            if is_pad:
                self._store_pad_rows(c, out_pos)
                prefetched = cutlass.Boolean(False)
            else:
                src_in, src_res = self._begin_position(
                    c, pos, pos_src, cache_frames, prefetched, src_in, src_res
                )
                prefetched = cutlass.Boolean(False)
                for t0 in cutlass.range(0, frames, p.order):
                    for j in cutlass.range_constexpr(p.order):
                        src_in, src_res, prefetched = self._frame_step(
                            c,
                            j,
                            t0,
                            pos,
                            pos_src,
                            out_pos,
                            pos_out,
                            halo,
                            eps,
                            src_in,
                            src_res,
                            prefetched,
                        )
                if cutlass.const_expr(p.winograd_out):
                    self._store_cache_tail(c, pos)
            out_pos = out_pos + c.position_stride

    # ------------------------------------------------------------------ kernel setup
    # Plain helpers: they only build views and copy launch arguments into ``c`` (compile-time
    # branches only), so they inline at no cost.

    def _global_views(
        self,
        c,
        x_ptr,
        m_ptr,
        d_ptr,
        residual_ptr,
        cache_ptr,
        sum_ptr,
        y_tail_ptr,
        y_ptr,
        halo_l_ptr,
        halo_r_ptr,
    ):
        """The global buffers as 16-byte vector views (``g*``) or packed-pair views (``u*``)."""
        c.gx = vec_view(x_ptr)
        c.gm = vec_view(m_ptr)
        c.gres = vec_view(residual_ptr)
        c.ud = u32_view(d_ptr)
        c.ucache = u32_view(cache_ptr)
        c.usum = u32_view(sum_ptr)
        c.utail = u32_view(y_tail_ptr)
        c.uy = u32_view(y_ptr)
        c.uhl = u32_view(halo_l_ptr)
        c.uhr = u32_view(halo_r_ptr)

    def _channel_tables(self, c, gamma_ptr, bias_ptr, in_bias_ptr, scale):
        """Fill the per-channel tables (gamma, bias, input bias) into shared memory, set this
        thread's position in its block, and keep the tables in registers when every lane owns
        one vector."""
        p = self.p
        g_gamma = cute.make_tensor(gamma_ptr, cute.make_layout((p.channels,)))
        g_bias = cute.make_tensor(bias_ptr, cute.make_layout((p.channels,)))
        g_in_bias = cute.make_tensor(in_bias_ptr, cute.make_layout((p.channels,)))

        smem_f = cute.arch.get_dyn_smem(Float32, alignment=16)
        # Per-channel tables, the 8 floats of vector vi split into two 16-byte halves in
        # separate planes: float e of half q of table t sits at ((2 * t + q) * NV + vi) * 4 + e
        # (NV vectors per row). A warp reads each half at a 16-byte lane stride, free of bank
        # conflicts.
        s_tab = cute.make_tensor(smem_f, cute.make_layout((3 * p.channels,)))
        c.s_quads = cute.make_tensor(smem_f, cute.make_layout((4, 6 * p.row_vecs), stride=(1, 4)))
        c.uhist = u32_view(cute.recast_ptr(smem_f + 3 * p.channels, dtype=p.row_dtype))

        tid, _, _ = cute.arch.thread_idx()
        tid = Int32(tid)
        lane = tid % 32
        warp = tid // 32
        group = lane // p.lanes
        c.lane_in_group = lane - group * p.lanes
        c.pos_in_block = warp * p.positions_per_warp + group

        self._fill_tables(s_tab, g_gamma, g_bias, g_in_bias, tid, scale)
        cute.arch.sync_threads()

        c.regs = None
        if cutlass.const_expr(p.one_vector_per_lane):
            c.regs = [
                self._table_vec(c, table, 0, c.lane_in_group) for table in (_GAMMA, _BIAS, _IN_BIAS)
            ]

    def _frame_geometry(
        self,
        c,
        positions,
        frames,
        height,
        width,
        pad_h,
        pad_w,
        in_w,
        in_off,
        out_w,
        out_off,
        in_frame,
        out_frame,
        comp_vecs,
        in_comp_vecs,
        scale,
    ):
        """Frame sizes, the input/output position maps and frame strides, and the grid stride
        of the position walk."""
        p = self.p
        v = p.v
        c.positions = positions
        c.frames = frames
        c.height = height
        c.width = width
        c.out_row_width = width + pad_w
        c.in_w = in_w
        c.in_off = in_off
        c.out_w = out_w
        c.out_off = out_off
        c.scale = scale
        c.comp_vecs = comp_vecs
        c.in_comp_vecs = in_comp_vecs
        positions_out = positions
        if cutlass.const_expr(v.padded):
            positions_out = (height + pad_h) * c.out_row_width
        c.positions_out = positions_out
        c.in_frame_pos = Int64(positions)
        if cutlass.const_expr(v.in_map):
            c.in_frame_pos = in_frame
        out_frame_pos = Int64(positions_out)
        if cutlass.const_expr(v.out_map):
            out_frame_pos = out_frame
        c.out_frame_pos = out_frame_pos
        gdim, _, _ = cute.arch.grid_dim()
        c.position_stride = Int32(gdim) * p.positions_per_block

    def _halo_geometry(
        self,
        c,
        halo_frame_l,
        halo_frame_r,
        halo_comp_l,
        halo_comp_r,
        halo_step_l,
        halo_step_r,
        halo_axis,
        has_l,
        has_r,
    ):
        """Where the left / right neighbours' halo lines go (spatially sharded decode)."""
        c.halo_frame_l = halo_frame_l
        c.halo_frame_r = halo_frame_r
        c.halo_comp_l = halo_comp_l
        c.halo_comp_r = halo_comp_r
        c.halo_step_l = halo_step_l
        c.halo_step_r = halo_step_r
        c.halo_axis = halo_axis
        c.has_l = has_l
        c.has_r = has_r

    def _register_buffers(self, c):
        """Three input slices (inputs x terms) and their residual slices; the row being
        normalized travels between the passes as packed bf16 pairs."""
        p = self.p
        c.in_buf = [
            cute.make_rmem_tensor((CHANNELS_PER_VECTOR, p.max_terms), p.in_dtype) for _ in range(3)
        ]
        c.res_buf = [cute.make_rmem_tensor((CHANNELS_PER_VECTOR,), p.row_dtype) for _ in range(3)]

    # ------------------------------------------------------------------ the position walk

    @cute.jit
    def _begin_position(self, c, pos, pos_src, cache_frames, prefetched, src_in, src_res):
        """Start position ``pos``: load the causal cache into the frame ring (Winograd output)
        and, unless the previous position already prefetched it, the first frame's input rows.
        Returns the input / residual row indices of the first frame."""
        p = self.p
        if cutlass.const_expr(p.winograd_out):
            self._load_cache_into_ring(c, pos, cache_frames)
        # An empty then-branch instead of ``if not prefetched``: ``not`` on a dynamic Boolean
        # emits an extra negation in front of the branch.
        if prefetched:
            pass
        else:
            src_in, src_res = self._input_rows(c, Int32(0), pos, pos_src)
            if cutlass.const_expr(p.next_frame_prefetched):
                self._load_frame_start(c, 0, 0, src_in, src_res)
        return src_in, src_res

    @cute.jit
    def _frame_step(
        self,
        c,
        j: cutlass.Constexpr,
        t0,
        pos,
        pos_src,
        out_pos,
        pos_out,
        halo,
        eps,
        src_in,
        src_res,
        prefetched,
    ):
        """Frame ``t0 + j`` of position ``pos``: prefetch the next frame's first slices, pass 1
        (input, bias, residual, sum of squares), the RMS scale and pass 2 (normalize, outputs),
        then hand the next frame's rows over. Returns ``(src_in, src_res, prefetched)`` for
        the next frame (``prefetched``: the next position's first frame is already loaded)."""
        p = self.p
        v = p.v
        t = t0 + j
        lane_opaque = _opaque(c.lane_in_group)
        nxt = self._next_position(c, t, pos, pos_src, out_pos)
        next_in_pos, has_next = nxt[0], nxt[1]
        next_src_in = Int64(0)
        next_src_res = Int64(0)

        if cutlass.const_expr(p.one_vector_per_lane):
            # The next frame's single slice goes to buffer 1 and moves to buffer 0 once this
            # frame is done.
            next_src_in, next_src_res = self._prefetch_next_frame(
                c, (j + 1) % p.order, nxt, next_src_in, next_src_res
            )
        elif cutlass.const_expr(not p.cross_frame_prefetch):
            self._load_slice(c, j, 0, src_in, src_res, 0)

        sum_sq = Float32(0.0)
        orow = Int64(t) * c.out_frame_pos + pos_out
        srow = Int64(t) * c.positions + pos
        sum_sq, row_words = self._pass1_frame(
            c, j, src_in, src_res, sum_sq, orow, srow, lane_opaque
        )

        if cutlass.const_expr(p.cross_frame_prefetch):
            next_src_in, next_src_res = self._prefetch_next_frame(
                c, (j + 1) % p.order, nxt, next_src_in, next_src_res
            )

        if cutlass.const_expr(v.mid != MID_NONE):
            denom, inv_denom = self._rms_scale(sum_sq, eps)
            self._pass2_frame(
                c, j, t, t0, pos_out, orow, row_words, denom, inv_denom, lane_opaque, halo
            )

        if cutlass.const_expr(p.next_frame_prefetched):
            src_in = next_src_in
            src_res = next_src_res
            if cutlass.const_expr(p.one_vector_per_lane):
                if has_next:
                    c.in_buf[0].store(c.in_buf[1].load())
                    if cutlass.const_expr(v.has_residual):
                        c.res_buf[0].store(c.res_buf[1].load())
            if next_in_pos:
                pass
            else:
                prefetched = has_next
        else:
            if next_in_pos:
                src_in, src_res = self._input_rows(c, t + 1, pos, pos_src)
        return src_in, src_res, prefetched

    # ------------------------------------------------------------------ trace-time helpers
    # Everything a helper reads comes in through ``c`` or its arguments (a staged loop may not
    # capture the kernel's locals). The ``@cute.jit`` helpers hold the dynamic branches and
    # loops; the plain methods have only compile-time branches and Python loops.

    @cute.jit
    def _fill_tables(self, s_tab, g_gamma, g_bias, g_in_bias, tid, scale):
        """The per-channel tables into shared memory. The fill is unrolled (the channel count
        is a compile-time constant) so every thread's loads are issued together; with
        ``reference`` off gamma is pre-multiplied by ``scale`` (by ``0.5 * scale`` with SiLU)."""
        p = self.p
        v = p.v
        channels = p.channels
        for it in cutlass.range_constexpr((channels + BLOCK_THREADS - 1) // BLOCK_THREADS):
            ch = tid + it * BLOCK_THREADS
            in_range = cutlass.Boolean(True)
            if cutlass.const_expr((it + 1) * BLOCK_THREADS > channels):
                in_range = ch < channels
            if in_range:
                idx = (
                    ((ch % CHANNELS_PER_VECTOR) // 4) * p.row_vecs + ch // CHANNELS_PER_VECTOR
                ) * 4 + ch % 4
                gamma_v = Float32(0.0)
                if cutlass.const_expr(v.mid != MID_NONE):
                    gamma_v = Float32(g_gamma[ch])
                    if cutlass.const_expr(not v.reference):
                        gamma_v = gamma_v * (Float32(0.5) * scale if p.silu else scale)
                bias_v = Float32(0.0)
                if cutlass.const_expr(v.has_bias):
                    bias_v = Float32(g_bias[ch])
                in_bias_v = Float32(0.0)
                if cutlass.const_expr(v.has_in_bias):
                    in_bias_v = Float32(g_in_bias[ch])
                s_tab[idx] = gamma_v
                s_tab[_BIAS * channels + idx] = bias_v
                s_tab[_IN_BIAS * channels + idx] = in_bias_v

    @cute.jit
    def _halo_targets(self, c, pos):
        """``(use_l, use_r, row_l, row_r)``: whether position ``pos`` lies on the first / last
        line of the split axis with a neighbour on that side, and the line row it goes to in
        that neighbour's halo buffer."""
        use_l = cutlass.Boolean(False)
        use_r = cutlass.Boolean(False)
        row_l = Int64(0)
        row_r = Int64(0)
        if cutlass.const_expr(self.p.v.halo):
            hi = pos // c.width
            hj = pos - hi * c.width
            across = hi
            along = hj
            n_across = c.height
            if c.halo_axis != 0:
                across = hj
                along = hi
                n_across = c.width
            if across == 0 and c.has_l != 0:
                use_l = cutlass.Boolean(True)
                row_l = Int64(along) * c.halo_step_l
            if across == n_across - 1 and c.has_r != 0:
                use_r = cutlass.Boolean(True)
                row_r = Int64(along) * c.halo_step_r
        return use_l, use_r, row_l, row_r

    @cute.jit
    def _store_pad_rows(self, c, out_pos):
        """A pad position: zero rows for every frame, nothing to read."""
        for t in cutlass.range(c.frames):
            for kk in cutlass.range_constexpr(self.p.vecs_per_lane):
                store_words(
                    c.uy, self._vrow(c, Int64(t) * c.positions_out + out_pos, kk), self._zeros()
                )

    @cute.jit
    def _load_cache_into_ring(self, c, pos, cache_frames):
        """Cached frames -2 and -1 into ring slots 0 and 1 (zeros where the cache is
        shorter)."""
        for h in cutlass.range_constexpr(2):
            cache_idx = cache_frames - 2 + h
            if cache_idx >= 0:
                for kk in cutlass.range_constexpr(self.p.vecs_per_lane):
                    ring_vec = self._slot_vec(c, h, kk)
                    cache_row = Int64(cache_idx) * c.positions + pos
                    store_words(
                        c.uhist, ring_vec, load_words(c.ucache, self._vrow(c, cache_row, kk))
                    )
            else:
                for kk in cutlass.range_constexpr(self.p.vecs_per_lane):
                    store_words(c.uhist, self._slot_vec(c, h, kk), self._zeros())

    @cute.jit
    def _next_position(self, c, t, pos, pos_src, out_pos):
        """Where the frame after ``t`` comes from: ``(next_in_pos, has_next, next_t, next_pos,
        next_pos_src)``. Frame ``t + 1`` of this position, or frame 0 of the next position this
        group walks (``has_next`` false past the end or on a pad position)."""
        next_in_pos = t + 1 < c.frames
        next_pos = pos
        has_next = cutlass.Boolean(True)
        next_pos_src = pos_src
        next_t = t + 1
        if next_in_pos:
            pass
        else:
            next_pos = self._source_pos(c, out_pos + c.position_stride)
            has_next = out_pos + c.position_stride < c.positions_out and next_pos >= 0
            next_pos_src = self._in_index(c, next_pos)
            next_t = Int32(0)
        return next_in_pos, has_next, next_t, next_pos, next_pos_src

    @cute.jit
    def _prefetch_next_frame(self, c, next_j: cutlass.Constexpr, nxt, next_src_in, next_src_res):
        """The next frame's first slices (see :meth:`_load_frame_start`), ``next_j`` being its
        index in the group when it belongs to this position; returns its input rows."""
        next_in_pos, has_next, next_t, next_pos, next_pos_src = nxt
        if has_next:
            next_src_in, next_src_res = self._input_rows(c, next_t, next_pos, next_pos_src)
            if next_in_pos:
                self._load_frame_start(
                    c, next_j, self._next_frame_buffer(), next_src_in, next_src_res
                )
            else:
                self._load_frame_start(c, 0, self._next_frame_buffer(), next_src_in, next_src_res)
        return next_src_in, next_src_res

    @cute.jit
    def _push_halo(self, c, halo, frame, kk: cutlass.Constexpr, s: cutlass.Constexpr, words):
        """Store slice ``kk`` of an output line position into the neighbours' halo buffers as
        well. ``s``: the Winograd input index (``frame`` is then the group's Int64 row), or None
        for row output (``frame`` is the frame index)."""
        use_l, use_r, row_l, row_r = halo
        if use_l:
            store_words(
                c.uhl, self._halo_vec(c, frame, row_l, c.halo_frame_l, c.halo_comp_l, kk, s), words
            )
        if use_r:
            store_words(
                c.uhr, self._halo_vec(c, frame, row_r, c.halo_frame_r, c.halo_comp_r, kk, s), words
            )

    def _halo_vec(self, c, frame, line_row, frame_stride, comp_stride, kk: int, s):
        if s is None:
            return self._vrow(c, Int64(frame) * frame_stride + line_row, kk)
        return self._vrow(c, frame * frame_stride + line_row, kk) + s * comp_stride

    def _vrow(self, c, row, kk: int):
        """Vector index of slice ``kk`` of this lane in row ``row`` of a [rows, C] buffer."""
        return Int64(row) * self.p.row_vecs + c.lane_in_group + kk * self.p.lanes

    def _slot_vec(self, c, slot, kk: int, lane=None):
        """Vector index of slice ``kk`` of this lane (``lane``, default the lane in the group)
        for this position in ring slot ``slot``."""
        p = self.p
        if lane is None:
            lane = c.lane_in_group
        return Int64(
            (slot * p.positions_per_block + c.pos_in_block) * p.row_vecs + lane + kk * p.lanes
        )

    def _table_vec(self, c, table: int, kk: int, lane) -> list:
        p = self.p
        vi = lane + kk * p.lanes
        out = []
        for q in range(2):
            quad = c.s_quads[(None, table * 2 * p.row_vecs + q * p.row_vecs + vi)].load()
            for e in range(4):
                out.append(quad[e])
        return out

    def _const_at(self, c, table: int, kk: int, lane_opaque) -> list:
        if self.p.one_vector_per_lane:
            return c.regs[table]
        return self._table_vec(c, table, kk, lane_opaque)

    def _in_index(self, c, pos):
        if self.p.v.in_map:
            i = pos // c.width
            return Int64(i) * c.in_w + (pos - i * c.width) + c.in_off
        return Int64(pos)

    def _out_index(self, c, pos):
        if self.p.v.out_map:
            i = pos // c.width
            return Int64(i) * c.out_w + (pos - i * c.width) + c.out_off
        return Int64(pos)

    def _source_pos(self, c, out_pos):
        """Source position of an output position (the row output may be padded); -1 for a
        pad."""
        if not self.p.v.padded:
            return out_pos
        oi = out_pos // c.out_row_width
        oj = out_pos - oi * c.out_row_width
        sp = Int32(cutlass.select_(oj < c.width, oi * c.width + oj, Int32(-1)))
        return Int32(cutlass.select_(oi < c.height, sp, Int32(-1)))

    def _input_rows(self, c, t, spos, spos_src):
        """Rows of frame ``t``'s inputs: the x row, or the row inside component 0 of the
        products (component k sits ``k * in_comp_vecs`` vectors further); the residual row."""
        s_res = Int64(t) * c.positions + spos
        if self.p.v.in_wino:
            return Int64(t // self.p.order) * c.in_frame_pos + spos_src, s_res
        return s_res, s_res

    def _load_slice(self, c, j: int, buf: int, src_in, src_res, kk: int) -> None:
        """Slice ``kk`` of the inputs of frame ``j`` of the group into buffer ``buf``: the x
        row, or the products of the nonzero terms of AT row ``j``."""
        p = self.p
        if p.v.in_wino:
            for q in range(len(p.terms[j])):
                c.in_buf[buf][(None, q)].store(
                    c.gm[(None, self._vrow(c, src_in, kk) + p.terms[j][q] * c.in_comp_vecs)].load()
                )
        else:
            c.in_buf[buf][(None, 0)].store(c.gx[(None, self._vrow(c, src_in, kk))].load())
        if p.v.has_residual:
            c.res_buf[buf].store(c.gres[(None, self._vrow(c, src_res, kk))].load())

    def _next_frame_buffer(self) -> int:
        # One vector per lane: buffer 0 still holds this frame's slice when the next one loads.
        return 1 if self.p.one_vector_per_lane else 0

    def _load_frame_start(self, c, j: int, first_buf: int, src_in, src_res) -> None:
        """The slices of frame ``j`` loaded ahead of its pass 1: slice 0 into buffer
        ``first_buf`` and, with the cross-frame prefetch, slice 1 into buffer 1."""
        self._load_slice(c, j, first_buf, src_in, src_res, 0)
        if self.p.cross_frame_prefetch:
            self._load_slice(c, j, 1, src_in, src_res, 1)

    def _load_ahead(self, c, j: int, n: int, src_in, src_res) -> None:
        """Slice ``n`` into buffer ``n % slice_depth``, unless the row has no slice ``n`` or the
        cross-frame prefetch already loaded it (slices 0 and 1)."""
        p = self.p
        if n < p.vecs_per_lane and not (p.cross_frame_prefetch and n < 2):
            self._load_slice(c, j, n % p.slice_depth, src_in, src_res, n)

    def _zeros(self) -> list:
        return [Int32(0)] * 4

    def _round(self, x):
        """``x`` rounded to bf16 and back (one packed convert of the duplicated value)."""
        return bf16_lo(pack2(x, x))

    def _pass1_frame(self, c, j: int, src_in, src_res, sum_sq, orow, srow, lane_opaque):
        """Pass 1 over the slices of frame ``j`` of the group: output transform, deferred
        bias, residual. Slice 0 (and with the cross-frame prefetch slice 1) is already loaded;
        the next ``slice_depth - 1`` slices are kept in flight in rotating buffers. Returns the
        updated sum of squares and the bf16 row as packed pairs per slice; the row also goes
        to ``sum`` (or, for the add_bias_and_residual kernel, to ``y``)."""
        p = self.p
        depth = p.slice_depth
        row_words = {}
        for n in range(1, depth - 1):
            self._load_ahead(c, j, n, src_in, src_res)
        for kk in range(p.vecs_per_lane):
            self._load_ahead(c, j, kk + depth - 1, src_in, src_res)
            sum_sq, row_words[kk] = self._pass1_slice(
                c, j, kk % depth, kk, sum_sq, orow, srow, lane_opaque
            )
        return sum_sq, row_words

    def _pass1_slice(self, c, j: int, buf: int, kk: int, sum_sq, orow, srow, lane_opaque):
        """Pass 1 on slice ``kk`` from buffer ``buf``. Returns the updated sum of squares and
        the bf16 slice as four packed pairs."""
        p = self.p
        v = p.v
        in_bias_v = self._const_at(c, _IN_BIAS, kk, lane_opaque)
        n_in = len(p.terms[j]) if v.in_wino else 1
        ins = [c.in_buf[buf][(None, q)].load().to(Float32) for q in range(n_in)]
        res = c.res_buf[buf].load().to(Float32) if v.has_residual else None
        words = []
        for i in range(WORDS_PER_VECTOR):
            pair = [self._combine_input(j, ins, in_bias_v, e) for e in (2 * i, 2 * i + 1)]
            # The packed word of the last rounding is the stored value; pack only if none.
            word = None
            if p.round_each_add:
                word, lo, hi = round_pair_bf16(pair[0], pair[1])
                pair = [lo, hi]
            if v.has_residual:
                pair = [pair[0] + res[2 * i], pair[1] + res[2 * i + 1]]
                word = None
                if p.round_each_add:
                    word, lo, hi = round_pair_bf16(pair[0], pair[1])
                    pair = [lo, hi]
            if v.mid != MID_NONE:
                sum_sq = cute.math.fma(pair[0], pair[0], sum_sq)
                sum_sq = cute.math.fma(pair[1], pair[1], sum_sq)
            words.append(word if word is not None else pack2(pair[0], pair[1]))
        if v.mid == MID_NONE:
            store_words(c.uy, self._vrow(c, orow, kk), words)
        elif p.has_sum:
            store_words(c.usum, self._vrow(c, srow, kk), words)
        return sum_sq, words

    def _combine_input(self, j: int, ins: list, in_bias_v: list, e: int):
        """Channel ``e`` of the slice before rounding: the x value, or the output transform of
        the products (in index order; the products are exact, so the reference rounding is
        one rounding per add), plus the deferred bias."""
        p = self.p
        v = p.v
        val = ins[0][e]
        if v.in_wino:
            terms = p.terms[j]
            val = val * Float32(p.at[j][terms[0]])
            for q in range(1, len(ins)):
                val = val + Float32(p.at[j][terms[q]]) * ins[q][e]
                if v.reference:
                    val = self._round(val)
        if v.has_in_bias:
            val = val + in_bias_v[e]
        return val

    def _rms_scale(self, sum_sq, eps):
        """``(denom, inv_denom)`` of the row from this lane's partial sum of squares: the
        divisor in the reference mode, the reciprocal otherwise (the other one is zero)."""
        p = self.p
        for offset in p.xor_offsets:
            sum_sq = sum_sq + cute.arch.shuffle_sync_bfly(sum_sq, offset)
        denom = Float32(0.0)
        inv_denom = Float32(0.0)
        if p.v.reference:
            denom = cute.arch.fmax(cute.math.sqrt(sum_sq), eps)
        else:
            inv_denom = cute.math.rsqrt(cute.arch.fmax(sum_sq, eps * eps))
        return denom, inv_denom

    def _pass2_frame(
        self, c, j: int, t, t0, pos_out, orow, row_words, denom, inv_denom, lane_opaque, halo
    ):
        """Pass 2: normalize frame ``t`` (frame ``j`` of the group starting at ``t0``) from
        registers into its ring slot, emitting the Winograd inputs it completes, or into the
        output row."""
        p = self.p
        group_row = Int64(t // p.order)
        drow = group_row * c.out_frame_pos + pos_out
        for kk in range(p.vecs_per_lane):
            yo = self._normalize(c, kk, row_words[kk], denom, inv_denom, lane_opaque)
            if p.winograd_out:
                self._store_winograd_inputs(c, j, t, t0, kk, yo, drow, group_row, halo)
            else:
                store_words(c.uy, self._vrow(c, orow, kk), yo)
                if p.v.halo:
                    self._push_halo(c, halo, t, kk, None, yo)

    def _store_winograd_inputs(self, c, j: int, t, t0, kk: int, yo, drow, group_row, halo):
        """Slice ``kk`` of the normalized frame into its ring slot, then the Winograd inputs it
        completes: input s reads the group's frames f_0 .. f_{j+1} from the ring and f_{j+2},
        this frame, from registers."""
        p = self.p
        store_words(c.uhist, self._slot_vec(c, (t + 2) % p.ring_slots, kk), yo)
        cur = unpack_bf16(yo)
        for s in range(p.components):
            if p.last_frame[s] == j + 2:
                o = self._winograd_input(c, s, j, t0, kk, cur)
                store_words(c.ud, self._vrow(c, drow, kk) + s * c.comp_vecs, o)
                if p.v.halo:
                    self._push_halo(c, halo, group_row, kk, s, o)

    def _store_cache_tail(self, c, pos):
        """The last two normalized frames, the next chunk's cache, into ``y_tail``."""
        p = self.p
        for kk in range(p.vecs_per_lane):
            store_words(
                c.utail,
                self._vrow(c, Int64(pos), kk),
                load_words(c.uhist, self._slot_vec(c, c.frames % p.ring_slots, kk)),
            )
            store_words(
                c.utail,
                self._vrow(c, Int64(c.positions) + pos, kk),
                load_words(c.uhist, self._slot_vec(c, (c.frames + 1) % p.ring_slots, kk)),
            )

    def _normalize(self, c, kk: int, words: list, denom, inv_denom, lane_opaque) -> list:
        """Pass 2 on slice ``kk`` (the bf16 row as packed pairs): RMSNorm (+ SiLU), returned as
        packed pairs. The reference mode reproduces the eager rounding sequence; otherwise
        gamma arrives pre-multiplied by scale (by 0.5 * scale with SiLU, which is then
        h * (1 + tanh(h)) with h = r / 2)."""
        p = self.p
        v = p.v
        gamma_v = self._const_at(c, _GAMMA, kk, lane_opaque)
        bias_v = self._const_at(c, _BIAS, kk, lane_opaque)
        out = []
        for i in range(WORDS_PER_VECTOR):
            pair = []
            for e, x in ((2 * i, bf16_lo(words[i])), (2 * i + 1, bf16_hi(words[i]))):
                if v.reference:
                    r = self._round(self._round(x / denom) * c.scale)
                    r = self._round(r * gamma_v[e])
                    if v.has_bias:
                        r = self._round(r + bias_v[e])
                    if p.silu:
                        r = r / (Float32(1.0) + cute.math.exp(-r))
                else:
                    r = x * inv_denom * gamma_v[e]
                    if p.silu:
                        if v.has_bias:
                            r = r + Float32(0.5) * bias_v[e]
                        r = cute.math.fma(r, cute.math.tanh(r, approx=True), r)
                    elif v.has_bias:
                        r = r + bias_v[e]
                pair.append(r)
            out.append(pack2(pair[0], pair[1]))
        return out

    def _winograd_input(self, c, s: int, j: int, t0, kk: int, cur: list) -> list:
        """Winograd input ``s`` of the group starting at frame ``t0``, slice ``kk``, as packed
        pairs: frames f_0 .. f_{j+1} from the ring, f_{j+2} (``cur``, this frame's 8 values)
        from registers. Products of 16-bit values and multiples of 1/2 are exact in fp32; the
        input rounds once."""
        p = self.p
        lane = _opaque(c.lane_in_group)
        ring_frames = {}
        for fi in range(j + 2):
            if p.bt[s][fi] != 0:
                slot = (t0 + fi) % p.ring_slots
                ring_frames[fi] = unpack_bf16(
                    load_words(c.uhist, self._slot_vec(c, slot, kk, lane))
                )
        out = []
        for i in range(WORDS_PER_VECTOR):
            pair = []
            for e in (2 * i, 2 * i + 1):
                acc = None
                for fi in range(j + 3):
                    coef = p.bt[s][fi]
                    if coef != 0:
                        fv = cur[e] if fi == j + 2 else ring_frames[fi][e]
                        acc = (
                            Float32(coef) * fv
                            if acc is None
                            else cute.math.fma(Float32(coef), fv, acc)
                        )
                pair.append(acc)
            out.append(pack2(pair[0], pair[1], p.v.fp16_winograd))
        return out


# The kernel arguments in the order of ``VaeNormKernel.wrapper``'s signature (pointer
# parameters carry a ``_ptr`` suffix there), each with its type: ``_PTR`` for a global-memory
# pointer (element type from :func:`_pointer_dtypes`), else the scalar type. The compile
# placeholders and the launch arguments are both built from this one list.
_PTR = "ptr"
_ARGS = (
    ("x", _PTR),
    ("m", _PTR),
    ("d", _PTR),
    ("residual", _PTR),
    ("in_bias", _PTR),
    ("cache", _PTR),
    ("gamma", _PTR),
    ("bias", _PTR),
    ("sum", _PTR),
    ("y_tail", _PTR),
    ("y", _PTR),
    ("halo_l", _PTR),
    ("halo_r", _PTR),
    ("comp_vecs", Int64),
    ("in_comp_vecs", Int64),
    ("in_frame", Int64),
    ("out_frame", Int64),
    ("halo_frame_l", Int64),
    ("halo_frame_r", Int64),
    ("halo_comp_l", Int64),
    ("halo_comp_r", Int64),
    ("positions", Int32),
    ("frames", Int32),
    ("cache_frames", Int32),
    ("height", Int32),
    ("width", Int32),
    ("pad_h", Int32),
    ("pad_w", Int32),
    ("in_w", Int32),
    ("in_off", Int32),
    ("out_w", Int32),
    ("out_off", Int32),
    ("halo_step_l", Int32),
    ("halo_step_r", Int32),
    ("halo_axis", Int32),
    ("has_l", Int32),
    ("has_r", Int32),
    ("scale", Float32),
    ("eps", Float32),
)
_ARG_NAMES = tuple(name for name, _ in _ARGS)


def _check_arg_order() -> None:
    """``_ARGS`` matches the wrapper's parameters (name and order) up to the grid size and
    the stream."""
    params = list(inspect.signature(VaeNormKernel.wrapper).parameters)[1:]  # without self
    names = [name + "_ptr" if kind is _PTR else name for name, kind in _ARGS]
    if params != [*names, "blocks", "stream"]:
        raise RuntimeError("kernels.norm: _ARGS does not match VaeNormKernel.wrapper")


_check_arg_order()


def _pointer_dtypes(p: _Plan) -> dict:
    """Element type of each pointer argument, keyed by its name in :data:`_ARGS`."""
    halo = p.row_dtype if p.v.out == OUT_ROWS else p.winograd_dtype
    dtypes = {name: p.row_dtype for name, kind in _ARGS if kind is _PTR}
    dtypes.update(m=p.winograd_dtype, d=p.winograd_dtype, halo_l=halo, halo_r=halo)
    return dtypes


def _kernel_name(v: Variant, has_sum: bool) -> str:
    fields = ",".join(f"{k}={getattr(v, k)}" for k in sorted(v.__dataclass_fields__))
    return f"vae_norm[{fields},has_sum={has_sum}]"


@functools.lru_cache(maxsize=None)
def _compile(device: int, variant: Variant, has_sum: bool):
    """The compiled variant (TVM-FFI calling convention: raw addresses and Python scalars) and
    its resident blocks per SM, through the on-disk CuTe DSL kernel cache when flashinfer
    provides it."""
    from cutlass.cute.runtime import make_ptr

    kernel = VaeNormKernel(variant, has_sum)

    def compile_fn():
        dtypes = _pointer_dtypes(kernel.p)
        placeholders = [
            make_ptr(dtypes[name], 16, cute.AddressSpace.gmem, assumed_align=16)
            if kind is _PTR
            else kind(0)
            for name, kind in _ARGS
        ]
        return cute.compile(
            kernel.wrapper,
            *placeholders,
            Int32(1),  # blocks
            current_stream(device),
            options="--enable-tvm-ffi",
        )

    from .. import jit_norm, wino_vae_kernels

    sources = [__file__, jit_norm.__file__, wino_vae_kernels.__file__]
    fn = compile_cached("trtllm_wan_vae_norm", _kernel_name(variant, has_sum), compile_fn, sources)
    return fn, kernel.p.blocks_per_sm


def launch(
    variant: Variant, device: int, has_sum: bool, steps: int, sm_count: int, args: dict, stream
) -> None:
    """Launch the variant (compiled on first use per device) on ``stream`` with ``args``, the
    value of every :data:`_ARGS` entry by name: an address as a Python int (0 for an operand
    the variant does not read), the scalars as Python ints and floats. ``steps`` thread
    blocks cover the output positions, the persistent grid is capped at the resident blocks
    of ``sm_count`` SMs."""
    fn, blocks_per_sm = _compile(device, variant, has_sum)
    fn(*[args[name] for name in _ARG_NAMES], min(steps, sm_count * blocks_per_sm), stream)
