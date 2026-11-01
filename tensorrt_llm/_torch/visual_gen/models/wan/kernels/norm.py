# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Wan VAE position-walk kernel in CuTe DSL.

One thread group (``lanes`` lanes, ``k`` 16-byte vectors each) owns one spatial position of a
batch-1 channels-last video (``[T, P, C]``, rows contiguous) and walks its frames in order.
Three compile-time stages make it every norm-side kernel of WinoVae:

  input   the x frames, or the temporal Winograd combine of ``order + 2`` conv products (frame
          ``g*R + j`` is ``sum_k AT[j][k] m_k[g]``, index order), then the deferred conv bias and
          the residual; the pre-norm row goes to ``sum`` when the variant writes it.
  middle  nothing (the combine kernel), RMSNorm, or RMSNorm + SiLU.
  output  the temporal Winograd inputs of order R (``D_k = sum_i BT[k][i] f_i`` over the group's
          R + 2 normalized frames, two of them the cache), the last two normalized frames going
          to ``y_tail`` for the next chunk; or plain rows to ``y`` (``[T, P', C]``, ``P'`` the
          padded position count when the downsampler's zero row and column are appended).

Numerics: fp32 throughout, one rounding per output unless ``reference`` (the eager bf16
rounding after every add, normalize, multiply and bias). The operation order is the one the
outputs are defined by: the sum of squares accumulates element by element with fma, the
Winograd sums run in table index order.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op

from ..jit_norm import MID_NONE, MID_NORM_SILU, OUT_ROWS, OUT_WINOGRAD, Variant
from .common import (
    VEC,
    bf16_hi,
    bf16_lo,
    compile_cached,
    load_words,
    pack2,
    store_words,
    u32_view,
    vec_view,
)

THREADS = 256
WARPS = THREADS // 32


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

    w = TEMPORAL_WINO[max(order, 2)]
    return w["BT"], w["AT"]


def _blocks_per_sm(v: Variant) -> int:
    """Resident blocks per SM the register budget is sized for (256 threads: 4 blocks give 64
    registers, 3 give 80, 2 give 128).

    Row output: 4 blocks up to two vectors per lane, 3 above; the order-4 product combine at
    160 and 320 channels holds five products per slice and spills at 80 registers, so it runs
    2 blocks; the 160-channel row walk of plain input is latency-bound at 3 and runs 4.
    Winograd output: 2 blocks for order 4 or four and more vectors per lane (the order-4 input
    producer emits six inputs per frame), except the order-4 producer from plain rows at two
    vectors per lane, which fits 80 registers and hides more load latency at 3; otherwise 4
    for plain rows at up to two vectors per lane and 3 for products."""
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
    """Vector slices in flight ahead of the one being combined: three for four or more
    vectors per lane, except where two measured faster: the 640-channel product combine, and
    the 1024-channel row norm with a residual (two slices plus the cross-frame prefetch of the
    next frame's first two slices hide the doubled loads better)."""
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


def _cross_frame_slices(v: Variant, depth: int) -> int:
    """Vector slices of the next frame loaded while this frame is normalized. Two for the row
    walks of plain input at slice depth 2. The products of a Winograd output kernel hold up to
    five vectors per slice, so they are not prefetched across frames, except at 1024 channels
    in order 2 (three products per slice, 128 registers): there two slices measured faster."""
    if v.k == 1:
        return 0
    if v.in_wino:
        return 2 if v.out == OUT_WINOGRAD and v.order == 2 and v.k == 4 and v.lanes == 32 else 0
    return 2 if depth == 2 else 0


class _Plan:
    """Compile-time constants of one variant."""

    def __init__(self, v: Variant, has_sum: bool) -> None:
        self.v = v
        self.has_sum = has_sum
        self.R = v.order
        self.L = v.lanes
        self.K = v.k
        self.C = v.lanes * v.k * VEC
        self.NV = self.C // VEC
        self.T = BFloat16
        self.TD = Float16 if v.fp16_winograd else BFloat16
        self.TIN = self.TD if v.in_wino else self.T
        self.bt, self.at = _tables(self.R)
        self.comps = self.R + 2
        self.terms = [
            [kk for kk in range(self.comps) if self.at[j][kk] != 0] for j in range(max(self.R, 1))
        ]
        self.num_in = max(len(t) for t in self.terms) if v.in_wino else 1
        self.last_frame = [
            max(i for i in range(self.comps) if self.bt[s][i] != 0) for s in range(self.comps)
        ]
        self.winograd_out = v.out == OUT_WINOGRAD
        self.act = v.mid == MID_NORM_SILU
        self.round1 = v.reference or v.out == OUT_ROWS
        # One vector per lane: whole frames are prefetched and the per-channel constants stay
        # in registers. Otherwise slices are pipelined ``depth`` deep, and ``xf`` slices of the
        # next frame are loaded while this one is normalized.
        self.frame_pf = self.K == 1
        self.const_regs = self.K == 1
        self.depth = _slice_depth(v)
        self.xf = _cross_frame_slices(v, self.depth)
        self.xf_pf = self.xf > 0
        self.xf_two = self.xf > 1 and self.K > 1
        self.pf = self.frame_pf or self.xf_pf  # the next frame's first slices are prefetched
        self.shuffles = [1 << b for b in reversed(range(self.L.bit_length() - 1))]
        self.ppw = 32 // self.L
        self.ppb = self.ppw * WARPS
        self.slots = self.R + 1
        self.blocks_per_sm = _blocks_per_sm(v)
        self.smem = v.shared_bytes(self.C)


class VaeNormKernel:
    def __init__(self, variant: Variant, has_sum: bool) -> None:
        self.p = _Plan(variant, has_sum)

    @cute.jit
    def wrapper(
        self,
        x_ptr: cute.Pointer,
        m_ptr: cute.Pointer,
        d_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        inb_ptr: cute.Pointer,
        cache_ptr: cute.Pointer,
        gamma_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        sum_ptr: cute.Pointer,
        tail_ptr: cute.Pointer,
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
            res_ptr,
            inb_ptr,
            cache_ptr,
            gamma_ptr,
            bias_ptr,
            sum_ptr,
            tail_ptr,
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
            block=[THREADS, 1, 1],
            smem=self.p.smem,
            stream=stream,
            min_blocks_per_mp=self.p.blocks_per_sm,
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        m_ptr: cute.Pointer,
        d_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        inb_ptr: cute.Pointer,
        cache_ptr: cute.Pointer,
        gamma_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        sum_ptr: cute.Pointer,
        tail_ptr: cute.Pointer,
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
        v = p.v
        R, K, C, NV = p.R, p.K, p.C, p.NV
        HALO = v.halo
        WOUT = p.winograd_out

        c = _Ctx()
        c.gx = vec_view(x_ptr)
        c.gm = vec_view(m_ptr)
        c.gres = vec_view(res_ptr)
        c.ud = u32_view(d_ptr)
        c.ucache = u32_view(cache_ptr)
        c.usum = u32_view(sum_ptr)
        c.utail = u32_view(tail_ptr)
        c.uy = u32_view(y_ptr)
        c.uhl = u32_view(halo_l_ptr)
        c.uhr = u32_view(halo_r_ptr)
        g_gamma = cute.make_tensor(gamma_ptr, cute.make_layout((C,)))
        g_bias = cute.make_tensor(bias_ptr, cute.make_layout((C,)))
        g_inb = cute.make_tensor(inb_ptr, cute.make_layout((C,)))

        smem_f = cute.arch.get_dyn_smem(Float32, alignment=16)
        # Per-channel tables, the 8 floats of vector vi split into two 16-byte quarters in
        # separate planes ((q * NV + vi) * 4 + e): a warp reads each quarter at a 16-byte lane
        # stride, free of bank conflicts.
        s_tab = cute.make_tensor(smem_f, cute.make_layout((3 * C,)))
        c.s_quads = cute.make_tensor(smem_f, cute.make_layout((4, 6 * NV), stride=(1, 4)))
        c.uhist = u32_view(cute.recast_ptr(smem_f + 3 * C, dtype=p.T))

        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane = tid % 32
        warp = tid // 32
        group = lane // p.L
        c.lig = lane - group * p.L
        c.pos_in_block = warp * p.ppw + group

        # The table fill is unrolled (the channel count is a compile-time constant) so every
        # thread's loads are issued together.
        for it in cutlass.range_constexpr((C + THREADS - 1) // THREADS):
            ch = tid + it * THREADS
            in_range = cutlass.Boolean(True)
            if cutlass.const_expr((it + 1) * THREADS > C):
                in_range = ch < C
            if in_range:
                idx = (((ch % VEC) // 4) * NV + ch // VEC) * 4 + ch % 4
                g = Float32(0.0)
                if cutlass.const_expr(v.mid != MID_NONE):
                    g = Float32(g_gamma[ch])
                    if cutlass.const_expr(not v.reference):
                        g = g * (Float32(0.5) * scale if p.act else scale)
                b_ = Float32(0.0)
                if cutlass.const_expr(v.has_bias):
                    b_ = Float32(g_bias[ch])
                ib_ = Float32(0.0)
                if cutlass.const_expr(v.has_in_bias):
                    ib_ = Float32(g_inb[ch])
                s_tab[idx] = g
                s_tab[C + idx] = b_
                s_tab[2 * C + idx] = ib_
        cute.arch.sync_threads()

        c.regs = None
        if cutlass.const_expr(p.const_regs):
            c.regs = [self._table_vec(c, t, 0, c.lig) for t in range(3)]

        P = positions
        c.P = P
        c.height = height
        c.width = width
        c.out_wp = width + pad_w
        c.in_w = in_w
        c.in_off = in_off
        c.out_w = out_w
        c.out_off = out_off
        c.scale = scale
        c.in_comp_vecs = in_comp_vecs
        POut = P
        if cutlass.const_expr(v.padded):
            POut = (height + pad_h) * c.out_wp
        c.in_frame_pos = Int64(P)
        if cutlass.const_expr(v.in_map):
            c.in_frame_pos = in_frame
        out_frame_pos = Int64(POut)
        if cutlass.const_expr(v.out_map):
            out_frame_pos = out_frame
        grid_pos = Int32(gdim) * p.ppb
        T_ = frames

        # Register buffers: three input slices (inputs x terms) and their residual slices. The
        # row being normalized travels between the passes as packed bf16 pairs.
        c.in_buf = [cute.make_rmem_tensor((VEC, p.num_in), p.TIN) for _ in range(3)]
        c.res_buf = [cute.make_rmem_tensor((VEC,), p.T) for _ in range(3)]

        prefetched = cutlass.Boolean(False)
        src_in = Int64(0)
        src_res = Int64(0)

        opos = Int32(bid) * p.ppb + c.pos_in_block
        while opos < POut:
            pos = self._source_pos(c, opos)
            pos_src = self._in_index(c, pos)
            pos_out = self._out_index(c, opos)

            use_l = cutlass.Boolean(False)
            use_r = cutlass.Boolean(False)
            row_l = Int64(0)
            row_r = Int64(0)
            if cutlass.const_expr(HALO):
                hi = pos // width
                hj = pos - hi * width
                across = hi
                along = hj
                n_across = height
                if halo_axis != 0:
                    across = hj
                    along = hi
                    n_across = width
                if across == 0 and has_l != 0:
                    use_l = cutlass.Boolean(True)
                    row_l = Int64(along) * halo_step_l
                if across == n_across - 1 and has_r != 0:
                    use_r = cutlass.Boolean(True)
                    row_r = Int64(along) * halo_step_r

            is_pad = cutlass.Boolean(False)
            if cutlass.const_expr(v.padded):
                is_pad = pos < 0
            if is_pad:
                # A pad position: zero rows for every frame, nothing to read.
                for t in cutlass.range(T_):
                    for kk in cutlass.range_constexpr(K):
                        store_words(c.uy, self._vrow(c, Int64(t) * POut + opos, kk), self._zeros())
                prefetched = cutlass.Boolean(False)
            else:
                if cutlass.const_expr(WOUT):
                    # Cached frames -2 and -1 into ring slots 0 and 1 (zeros where the cache
                    # is shorter).
                    for h in cutlass.range_constexpr(2):
                        cache_idx = cache_frames - 2 + h
                        if cache_idx >= 0:
                            for kk in cutlass.range_constexpr(K):
                                store_words(
                                    c.uhist,
                                    self._slot_vec(c, h, kk),
                                    load_words(
                                        c.ucache, self._vrow(c, Int64(cache_idx) * P + pos, kk)
                                    ),
                                )
                        else:
                            for kk in cutlass.range_constexpr(K):
                                store_words(c.uhist, self._slot_vec(c, h, kk), self._zeros())

                if prefetched:
                    pass
                else:
                    src_in, src_res = self._src_of(c, Int32(0), pos, pos_src)
                    if cutlass.const_expr(p.pf):
                        self._load_slice(c, 0, 0, src_in, src_res, 0)
                    if cutlass.const_expr(p.xf_two):
                        self._load_slice(c, 0, 1, src_in, src_res, 1)
                prefetched = cutlass.Boolean(False)

                for t0 in cutlass.range(0, T_, R):
                    for J in cutlass.range_constexpr(R):
                        JN = (J + 1) % R
                        t = t0 + J
                        lig_f = _opaque(c.lig)
                        next_in_pos = t + 1 < T_
                        next_pos = pos
                        has_next = cutlass.Boolean(True)
                        next_pos_src = pos_src
                        next_t = t + 1
                        if next_in_pos:
                            pass
                        else:
                            next_pos = self._source_pos(c, opos + grid_pos)
                            has_next = opos + grid_pos < POut and next_pos >= 0
                            next_pos_src = self._in_index(c, next_pos)
                            next_t = Int32(0)
                        nx_in = Int64(0)
                        nx_res = Int64(0)

                        if cutlass.const_expr(p.frame_pf):
                            # The next frame's single slice goes to buffer 1 and moves to
                            # buffer 0 once this frame is done.
                            if has_next:
                                nx_in, nx_res = self._src_of(c, next_t, next_pos, next_pos_src)
                                if next_in_pos:
                                    self._load_slice(c, JN, 1, nx_in, nx_res, 0)
                                else:
                                    self._load_slice(c, 0, 1, nx_in, nx_res, 0)
                        elif cutlass.const_expr(not p.xf_pf):
                            self._load_slice(c, J, 0, src_in, src_res, 0)

                        # Pass 1: combine, deferred bias, residual; the bf16 row stays in
                        # registers (``yw``, packed pairs per slice) and goes to ``sum``; the sum
                        # of squares on the way.
                        ss = Float32(0.0)
                        yw = {}
                        orow = Int64(t) * out_frame_pos + pos_out
                        srow = Int64(t) * P + pos
                        if cutlass.const_expr(p.frame_pf):
                            ss, yw[0] = self._slice(c, J, 0, 0, ss, orow, srow, lig_f)
                        elif cutlass.const_expr(p.depth == 2):
                            for kk in cutlass.range_constexpr(0, K, 2):
                                if cutlass.const_expr(kk + 1 < K and not (p.xf_two and kk == 0)):
                                    self._load_slice(c, J, 1, src_in, src_res, kk + 1)
                                ss, yw[kk] = self._slice(c, J, 0, kk, ss, orow, srow, lig_f)
                                if cutlass.const_expr(kk + 1 < K):
                                    if cutlass.const_expr(kk + 2 < K):
                                        self._load_slice(c, J, 0, src_in, src_res, kk + 2)
                                    ss, yw[kk + 1] = self._slice(
                                        c, J, 1, kk + 1, ss, orow, srow, lig_f
                                    )
                        else:
                            if cutlass.const_expr(1 < K and not p.xf_two):
                                self._load_slice(c, J, 1, src_in, src_res, 1)
                            for kk in cutlass.range_constexpr(0, K, 3):
                                if cutlass.const_expr(kk + 2 < K):
                                    self._load_slice(c, J, 2, src_in, src_res, kk + 2)
                                ss, yw[kk] = self._slice(c, J, 0, kk, ss, orow, srow, lig_f)
                                if cutlass.const_expr(kk + 1 < K):
                                    if cutlass.const_expr(kk + 3 < K):
                                        self._load_slice(c, J, 0, src_in, src_res, kk + 3)
                                    ss, yw[kk + 1] = self._slice(
                                        c, J, 1, kk + 1, ss, orow, srow, lig_f
                                    )
                                if cutlass.const_expr(kk + 2 < K):
                                    if cutlass.const_expr(kk + 4 < K):
                                        self._load_slice(c, J, 1, src_in, src_res, kk + 4)
                                    ss, yw[kk + 2] = self._slice(
                                        c, J, 2, kk + 2, ss, orow, srow, lig_f
                                    )

                        if cutlass.const_expr(p.xf_pf):
                            if has_next:
                                nx_in, nx_res = self._src_of(c, next_t, next_pos, next_pos_src)
                                if next_in_pos:
                                    self._load_slice(c, JN, 0, nx_in, nx_res, 0)
                                    if cutlass.const_expr(p.xf_two):
                                        self._load_slice(c, JN, 1, nx_in, nx_res, 1)
                                else:
                                    self._load_slice(c, 0, 0, nx_in, nx_res, 0)
                                    if cutlass.const_expr(p.xf_two):
                                        self._load_slice(c, 0, 1, nx_in, nx_res, 1)

                        if cutlass.const_expr(v.mid != MID_NONE):
                            for i in cutlass.range_constexpr(len(p.shuffles)):
                                ss = ss + cute.arch.shuffle_sync_bfly(ss, p.shuffles[i])
                            denom = Float32(0.0)
                            inv_denom = Float32(0.0)
                            if cutlass.const_expr(v.reference):
                                denom = cute.arch.fmax(cute.math.sqrt(ss), eps)
                            else:
                                inv_denom = cute.math.rsqrt(cute.arch.fmax(ss, eps * eps))

                            # Pass 2: normalize from registers into the ring slot (emitting the
                            # Winograd inputs this frame completes: input s reads the group's
                            # frames f_0 .. f_{J+1} from the ring and f_{J+2}, this frame, from
                            # registers) or into the output row.
                            drow = Int64(t // R) * out_frame_pos + pos_out
                            for kk in cutlass.range_constexpr(K):
                                yo = self._normalize(c, kk, yw[kk], denom, inv_denom, lig_f)
                                if cutlass.const_expr(WOUT):
                                    store_words(
                                        c.uhist, self._slot_vec(c, (t + 2) % p.slots, kk), yo
                                    )
                                    cur = [f for w in yo for f in (bf16_lo(w), bf16_hi(w))]
                                    for s in cutlass.range_constexpr(p.comps):
                                        if cutlass.const_expr(p.last_frame[s] == J + 2):
                                            o = self._winograd_input(c, s, J, t0, kk, cur)
                                            store_words(
                                                c.ud, self._vrow(c, drow, kk) + s * comp_vecs, o
                                            )
                                            if cutlass.const_expr(HALO):
                                                g_row = Int64(t // R)
                                                if use_l:
                                                    store_words(
                                                        c.uhl,
                                                        self._vrow(
                                                            c, g_row * halo_frame_l + row_l, kk
                                                        )
                                                        + s * halo_comp_l,
                                                        o,
                                                    )
                                                if use_r:
                                                    store_words(
                                                        c.uhr,
                                                        self._vrow(
                                                            c, g_row * halo_frame_r + row_r, kk
                                                        )
                                                        + s * halo_comp_r,
                                                        o,
                                                    )
                                else:
                                    store_words(c.uy, self._vrow(c, orow, kk), yo)
                                    if cutlass.const_expr(HALO):
                                        if use_l:
                                            store_words(
                                                c.uhl,
                                                self._vrow(c, Int64(t) * halo_frame_l + row_l, kk),
                                                yo,
                                            )
                                        if use_r:
                                            store_words(
                                                c.uhr,
                                                self._vrow(c, Int64(t) * halo_frame_r + row_r, kk),
                                                yo,
                                            )

                        if cutlass.const_expr(p.pf):
                            src_in = nx_in
                            src_res = nx_res
                            if cutlass.const_expr(p.frame_pf):
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
                                src_in, src_res = self._src_of(c, t + 1, pos, pos_src)

                if cutlass.const_expr(WOUT):
                    # The last two normalized frames are the next chunk's cache.
                    for kk in cutlass.range_constexpr(K):
                        store_words(
                            c.utail,
                            self._vrow(c, Int64(pos), kk),
                            load_words(c.uhist, self._slot_vec(c, T_ % p.slots, kk)),
                        )
                        store_words(
                            c.utail,
                            self._vrow(c, Int64(P) + pos, kk),
                            load_words(c.uhist, self._slot_vec(c, (T_ + 1) % p.slots, kk)),
                        )
            opos = opos + grid_pos

    # ------------------------------------------------------------------ trace-time helpers
    # Plain methods: everything they read comes in through ``c`` (a staged loop may not
    # capture the kernel's locals), and every branch in them is a compile-time one.

    def _vrow(self, c, row, kk: int):
        """Vector index of slice ``kk`` of this lane in row ``row`` of a [rows, C] buffer."""
        return Int64(row) * self.p.NV + c.lig + kk * self.p.L

    def _slot_vec(self, c, slot, kk: int):
        p = self.p
        return Int64((slot * p.ppb + c.pos_in_block) * p.NV + c.lig + kk * p.L)

    def _table_vec(self, c, t: int, kk: int, lig) -> list:
        p = self.p
        vi = lig + kk * p.L
        out = []
        for q in range(2):
            quad = c.s_quads[(None, t * 2 * p.NV + q * p.NV + vi)].load()
            for e in range(4):
                out.append(quad[e])
        return out

    def _const_at(self, c, t: int, kk: int, lig_f) -> list:
        if self.p.const_regs:
            return c.regs[t]
        return self._table_vec(c, t, kk, lig_f)

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

    def _source_pos(self, c, opos):
        """Source position of an output position (the row output may be padded); -1 for a
        pad."""
        if not self.p.v.padded:
            return opos
        oi = opos // c.out_wp
        oj = opos - oi * c.out_wp
        sp = Int32(cutlass.select_(oj < c.width, oi * c.width + oj, Int32(-1)))
        return Int32(cutlass.select_(oi < c.height, sp, Int32(-1)))

    def _src_of(self, c, t, spos, spos_src):
        """Rows of frame ``t``'s inputs: the x row, or the row inside component 0 of the
        products (component k sits ``k * in_comp_vecs`` vectors further); the residual row."""
        s_res = Int64(t) * c.P + spos
        if self.p.v.in_wino:
            return Int64(t // self.p.R) * c.in_frame_pos + spos_src, s_res
        return s_res, s_res

    def _load_slice(self, c, j: int, b: int, s_in, s_res, kk: int) -> None:
        """Slice ``kk`` of frame ``j``'s inputs into buffer ``b``: the x row, or the products of
        the nonzero terms of AT row ``j``."""
        p = self.p
        if p.v.in_wino:
            for q in range(len(p.terms[j])):
                c.in_buf[b][(None, q)].store(
                    c.gm[(None, self._vrow(c, s_in, kk) + p.terms[j][q] * c.in_comp_vecs)].load()
                )
        else:
            c.in_buf[b][(None, 0)].store(c.gx[(None, self._vrow(c, s_in, kk))].load())
        if p.v.has_residual:
            c.res_buf[b].store(c.gres[(None, self._vrow(c, s_res, kk))].load())

    def _zeros(self) -> list:
        return [Int32(0)] * 4

    def _round(self, x):
        """``x`` rounded to bf16 and back (one packed convert of the duplicated value)."""
        return bf16_lo(pack2(x, x, False))

    def _slice(self, c, J: int, b: int, kk: int, ss, orow, srow, lig_f):
        """Pass 1 on slice ``kk`` from buffer ``b``: combine (in index order; the products are
        exact, so the reference rounding is one rounding per add), deferred bias, residual.
        Returns the updated sum of squares and the bf16 slice as four packed pairs, which also
        go to ``sum`` (or, for the combine kernel, to ``y``)."""
        p = self.p
        v = p.v
        ref = v.reference
        ib = self._const_at(c, 2, kk, lig_f)
        n_in = len(p.terms[J]) if v.in_wino else 1
        ins = [c.in_buf[b][(None, q)].load().to(Float32) for q in range(n_in)]
        res = c.res_buf[b].load().to(Float32) if v.has_residual else None
        words = []
        for i in range(VEC // 2):
            pair = []
            for e in (2 * i, 2 * i + 1):
                val = ins[0][e]
                if v.in_wino:
                    val = val * Float32(p.at[J][p.terms[J][0]])
                    for q in range(1, n_in):
                        val = val + Float32(p.at[J][p.terms[J][q]]) * ins[q][e]
                        if ref:
                            val = self._round(val)
                if v.has_in_bias:
                    val = val + ib[e]
                pair.append(val)
            if p.round1:
                w = pack2(pair[0], pair[1], False)
                pair = [bf16_lo(w), bf16_hi(w)]
            if v.has_residual:
                pair = [pair[0] + res[2 * i], pair[1] + res[2 * i + 1]]
                if p.round1:
                    w = pack2(pair[0], pair[1], False)
                    pair = [bf16_lo(w), bf16_hi(w)]
            if v.mid != MID_NONE:
                ss = cute.math.fma(pair[0], pair[0], ss)
                ss = cute.math.fma(pair[1], pair[1], ss)
            words.append(pack2(pair[0], pair[1], False))
        if v.mid == MID_NONE:
            store_words(c.uy, self._vrow(c, orow, kk), words)
        elif p.has_sum:
            store_words(c.usum, self._vrow(c, srow, kk), words)
        return ss, words

    def _normalize(self, c, kk: int, words: list, denom, inv_denom, lig_f) -> list:
        """Pass 2 on slice ``kk`` (the bf16 row as packed pairs): RMSNorm (+ SiLU), returned as
        packed pairs. The reference mode reproduces the eager rounding sequence; otherwise
        gamma arrives pre-multiplied by scale (by 0.5 * scale with SiLU, which is then
        h * (1 + tanh(h)) with h = r / 2)."""
        p = self.p
        v = p.v
        gmv = self._const_at(c, 0, kk, lig_f)
        bsv = self._const_at(c, 1, kk, lig_f)
        out = []
        for i in range(VEC // 2):
            pair = []
            for e, x in ((2 * i, bf16_lo(words[i])), (2 * i + 1, bf16_hi(words[i]))):
                if v.reference:
                    r = self._round(self._round(x / denom) * c.scale)
                    r = self._round(r * gmv[e])
                    if v.has_bias:
                        r = self._round(r + bsv[e])
                    if p.act:
                        r = r / (Float32(1.0) + cute.math.exp(-r))
                else:
                    r = x * inv_denom * gmv[e]
                    if p.act:
                        if v.has_bias:
                            r = r + Float32(0.5) * bsv[e]
                        r = cute.math.fma(r, cute.math.tanh(r, approx=True), r)
                    elif v.has_bias:
                        r = r + bsv[e]
                pair.append(r)
            out.append(pack2(pair[0], pair[1], False))
        return out

    def _winograd_input(self, c, s: int, J: int, t0, kk: int, cur: list) -> list:
        """Winograd input ``s`` of the group starting at frame ``t0``, slice ``kk``, as packed
        pairs: frames f_0 .. f_{J+1} from the ring, f_{J+2} (``cur``, this frame's 8 values)
        from registers. Products of 16-bit values and multiples of 1/2 are exact in fp32; the
        input rounds once."""
        p = self.p
        lig = _opaque(c.lig)
        hs = {}
        for fi in range(J + 2):
            if p.bt[s][fi] != 0:
                slot = (t0 + fi) % p.slots
                vi = Int64((slot * p.ppb + c.pos_in_block) * p.NV + lig + kk * p.L)
                ws = load_words(c.uhist, vi)
                hs[fi] = [f for w in ws for f in (bf16_lo(w), bf16_hi(w))]
        out = []
        for i in range(VEC // 2):
            pair = []
            for e in (2 * i, 2 * i + 1):
                acc = None
                for fi in range(J + 3):
                    coef = p.bt[s][fi]
                    if coef != 0:
                        fv = cur[e] if fi == J + 2 else hs[fi][e]
                        acc = (
                            Float32(coef) * fv
                            if acc is None
                            else cute.math.fma(Float32(coef), fv, acc)
                        )
                pair.append(acc)
            out.append(pack2(pair[0], pair[1], p.v.fp16_winograd))
        return out


class _Ctx:
    """Trace-time bundle of the kernel's views, scalars and register buffers."""


_POINTER_ARGS = (
    "x",
    "m",
    "d",
    "residual",
    "in_bias",
    "cache",
    "gamma",
    "bias",
    "sum",
    "y_tail",
    "y",
    "halo_l",
    "halo_r",
)
_INT64_ARGS = (
    "comp_vecs",
    "in_comp_vecs",
    "in_frame",
    "out_frame",
    "halo_frame_l",
    "halo_frame_r",
    "halo_comp_l",
    "halo_comp_r",
)
_INT32_ARGS = (
    "positions",
    "frames",
    "cache_frames",
    "height",
    "width",
    "pad_h",
    "pad_w",
    "in_w",
    "in_off",
    "out_w",
    "out_off",
    "halo_step_l",
    "halo_step_r",
    "halo_axis",
    "has_l",
    "has_r",
)
_FLOAT_ARGS = ("scale", "eps")

_compiled: dict = {}


def _pointer_dtypes(v: Variant) -> tuple:
    t = BFloat16
    td = Float16 if v.fp16_winograd else BFloat16
    halo = t if v.out == OUT_ROWS else td
    return (t, td, td, t, t, t, t, t, t, t, t, halo, halo)


def _kernel_name(v: Variant, has_sum: bool) -> str:
    fields = ",".join(f"{k}={getattr(v, k)}" for k in sorted(v.__dataclass_fields__))
    return f"vae_norm[{fields},has_sum={has_sum}]"


def _compile(variant: Variant, has_sum: bool, stream):
    """Compile the variant (TVM-FFI calling convention: raw addresses and Python scalars),
    through the on-disk CuTe DSL kernel cache when flashinfer provides it."""
    from cutlass.cute.runtime import make_ptr

    kernel = VaeNormKernel(variant, has_sum)

    def compile_fn():
        ptrs = [
            make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)
            for dt in _pointer_dtypes(variant)
        ]
        scalars = (
            [Int64(0)] * len(_INT64_ARGS)
            + [Int32(0)] * len(_INT32_ARGS)
            + [Float32(0.0)] * len(_FLOAT_ARGS)
            + [Int32(1)]
        )
        return cute.compile(kernel.wrapper, *ptrs, *scalars, stream, options="--enable-tvm-ffi")

    from .. import jit_norm, wino_vae_kernels

    sources = [__file__, jit_norm.__file__, wino_vae_kernels.__file__]
    fn = compile_cached("trtllm_wan_vae_norm", _kernel_name(variant, has_sum), compile_fn, sources)
    return fn, kernel.p.blocks_per_sm


def launch(
    variant: Variant, device: int, has_sum: bool, steps: int, sm_count: int, args: dict, stream
) -> None:
    """Launch the variant (compiled on first use per device) on ``stream`` with the addresses
    and scalars of ``jit_norm._launch`` (a null address for an operand the variant does not
    read); ``steps`` thread blocks cover the output positions, the persistent grid is capped
    at the resident blocks of ``sm_count`` SMs."""
    key = (device, variant, has_sum)
    entry = _compiled.get(key)
    if entry is None:
        entry = _compile(variant, has_sum, stream)
        _compiled[key] = entry
    fn, bps = entry
    args = dict(args, has_l=int(bool(args["halo_l"])), has_r=int(bool(args["halo_r"])))
    fn(
        *(args[k] for k in _POINTER_ARGS),
        *(int(args[k]) for k in _INT64_ARGS),
        *(int(args[k]) for k in _INT32_ARGS),
        *(float(args[k]) for k in _FLOAT_ARGS),
        int(min(steps, sm_count * bps)),
        stream,
    )
