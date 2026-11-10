# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A verify step's Gated-DeltaNet input projection with the causal conv of its q|k|v columns in
the GEMM's epilogue: `low_m_bf16_mm` then `causal_conv1d_token_major`, in one launch.

The GEMM is FlashInfer's warp split-K kernel (`dense_bf16_gemm_warp_splitk`), subclassed with
its body unchanged. A CTA owns 16 output columns for every row of the step (one token tile), so
the reduce warp keeps the bf16-rounded tile in shared memory and its first 16 lanes run the conv
of their channel as `_causal_conv1d_update_kernel` does (window snapshots, SiLU, the token-major
`[n * p, C]` output, the pool advanced). No two CTAs touch one channel.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import T as _ty
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims
from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as _fw

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ..gemm import low_m_bf16_mm__decode as _lowm

#: The GDN conv (width 4) over a tp4/tep4 rank's q|k|v channels.
CONV_WIDTH = 4
CONV_DIM = 2560
#: The projection a tp4/tep4 rank runs: `[q k v | z | a | b | pad]` = 4128 over hidden 2560.
N_OUT, K_IN = 4128, 2560


def _ir(x, cls, loc, ip):
    return cls(x).ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_i32p(gaddr, pred, other: int, *, loc=None, ip=None):
    """An int32, or `other` where `pred` is 0 (nothing read)."""
    return cutlass.Int32(
        _asm(_ty.i32(), [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
             "{ .reg .pred q; setp.ne.b32 q, $2, 0; mov.b32 $0, " + str(int(other)) + "; "
             "@q ld.global.b32 $0, [$1]; }", "=r,l,r", loc, ip, side=True)
    )  # fmt: skip


@dsl_user_op
def _ldg_bf16p(gaddr, pred, *, loc=None, ip=None):
    """One bf16 widened to f32, or 0 where `pred` is 0 (nothing read)."""
    return cutlass.Float32(
        _asm(_ty.f32(), [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
             "{ .reg .pred q; .reg .b16 h; setp.ne.b32 q, $2, 0; mov.b16 h, 0; "
             "@q ld.global.b16 h, [$1]; cvt.f32.bf16 $0, h; }", "=f,l,r", loc, ip, side=True)
    )  # fmt: skip


@dsl_user_op
def _stg_bf16(gaddr, x, *, loc=None, ip=None):
    """One f32 rounded to bf16 (round to nearest even) and stored."""
    _asm(_ty.i32(), [_ir(gaddr, cutlass.Int64, loc, ip), _ir(x, cutlass.Float32, loc, ip)],
         "{ .reg .b16 h; cvt.rn.bf16.f32 h, $2; st.global.b16 [$1], h; mov.u32 $0, 0; }",
         "=r,l,f", loc, ip, side=True)  # fmt: skip


@dsl_user_op
def _round_bf16(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_ty.f32(), [_ir(x, cutlass.Float32, loc, ip)],
             "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }", "=f,f", loc, ip)
    )  # fmt: skip


@dsl_user_op
def _mul_bf16(a, b, *, loc=None, ip=None):
    """A bf16 x bf16 product as the Triton kernel forms it: exact in f32, rounded to bf16."""
    return cutlass.Float32(
        _asm(_ty.f32(), [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
             "{ .reg .b16 h; .reg .f32 p; mul.rn.f32 p, $1, $2; cvt.rn.bf16.f32 h, p; "
             "cvt.f32.bf16 $0, h; }", "=f,f,f", loc, ip)
    )  # fmt: skip


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_ty.f32(), [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
             "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )  # fmt: skip


@dsl_user_op
def _silu(x, *, loc=None, ip=None):
    """acc / (1 + ex2(-acc * log2(e))): bitwise the Triton kernel's SiLU."""
    return cutlass.Float32(
        _asm(_ty.f32(), [_ir(x, cutlass.Float32, loc, ip)],
             "{ .reg .f32 t, e, d; neg.f32 t, $1; mul.rn.f32 t, t, 0f3FB8AA3B; "
             "ex2.approx.f32 e, t; add.rn.f32 d, e, 0f3F800000; div.full.f32 $0, $1, d; }",
             "=f,f", loc, ip)
    )  # fmt: skip


class _InProjConvKernel(_fw.CpAsyncWarpSplitKKernel):
    """FlashInfer's warp split-K kernel with the conv epilogue (see the module docstring)."""

    def __init__(self, tactic, use_pdl: bool, rows_per_request: int) -> None:
        super().__init__(tactic, use_pdl, False)
        self.P = rows_per_request

    @cute.experimental.jit
    def __call__(
        self,
        weight,
        activation,
        bias,
        output,
        conv_w: cutlass.Int64,
        pool: cutlass.Int64,
        pool_s_seq: cutlass.Int64,
        pool_s_dim: cutlass.Int64,
        pool_s_tok: cutlass.Int64,
        slots: cutlass.Int64,
        win: cutlass.Int64,
        win_s_seq: cutlass.Int64,
        win_s_step: cutlass.Int64,
        win_s_dim: cutlass.Int64,
        win_s_win: cutlass.Int64,
        win_rows: cutlass.Int64,
        conv_out: cutlass.Int64,
        pad_slot: cutlass.Int32,
        stream: _cuda.CUstream,
    ):
        sA_layout = _fw._make_smem_layout(
            weight.element_type, self.output_tile, self.k_tile, self.stages
        )
        sB_layout = _fw._make_smem_layout(
            activation.element_type, self.token_tile, self.k_tile, self.stages
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(weight.element_type, cutlass.Float32, _fw._MMA_SHAPE),
            cute.make_layout((1, 1, _fw._COMPUTE_WARPS)),
            permutation_mnk=(self.output_tile, self.token_tile, self.k_tile),
        )
        self.kernel(
            weight,
            activation,
            bias,
            output,
            sA_layout,
            sB_layout,
            tiled_mma,
            conv_w,
            pool,
            pool_s_seq,
            pool_s_dim,
            pool_s_tok,
            slots,
            win,
            win_s_seq,
            win_s_step,
            win_s_dim,
            win_s_win,
            win_rows,
            conv_out,
            pad_slot,
        ).launch(
            grid=(
                cute.ceil_div(weight.shape[0], self.output_tile),
                cute.ceil_div(activation.shape[0], self.token_tile),
                1,
            ),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes + self.output_tile * self.token_tile * 4 + 64),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        activation: cute.Tensor,
        bias: cute.Tensor,
        output: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        conv_w: cutlass.Int64,
        pool: cutlass.Int64,
        pool_s_seq: cutlass.Int64,
        pool_s_dim: cutlass.Int64,
        pool_s_tok: cutlass.Int64,
        slots: cutlass.Int64,
        win: cutlass.Int64,
        win_s_seq: cutlass.Int64,
        win_s_step: cutlass.Int64,
        win_s_dim: cutlass.Int64,
        win_s_win: cutlass.Int64,
        win_rows: cutlass.Int64,
        conv_out: cutlass.Int64,
        pad_slot: cutlass.Int32,
    ):
        # ---- FlashInfer's kernel body, unchanged (dense_bf16_gemm_warp_splitk.py) ----------
        _COMPUTE_WARPS = _fw._COMPUTE_WARPS
        _A_LOADER_WARPS = _fw._A_LOADER_WARPS
        _MMA_SHAPE = _fw._MMA_SHAPE
        _MAILBOX_GROUP_VALUES = _fw._MAILBOX_GROUP_VALUES
        _MAILBOX_PADDED_GROUP_VALUES = _fw._MAILBOX_PADDED_GROUP_VALUES
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, token_tile_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        reuse_stages = cute.size(weight, mode=[1]) > self.stages * self.k_tile

        sA = cute_ext.allocate(
            weight.element_type, cute.AddressSpace.smem, sA_layout, alignment=1024
        )
        sB = cute_ext.allocate(
            activation.element_type, cute.AddressSpace.smem, sB_layout, alignment=1024
        )
        values_per_lane = self.output_tile * self.token_tile // cute.arch.WARP_SIZE
        partial_stride = (
            self.output_tile * self.token_tile // _MAILBOX_GROUP_VALUES
        ) * _MAILBOX_PADDED_GROUP_VALUES
        partials = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_COMPUTE_WARPS * partial_stride),
            alignment=16,
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(self.stages), alignment=8
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(self.stages), alignment=8
        )
        # The reduced tile, bf16-rounded, for the conv epilogue.
        xtile = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.output_tile * self.token_tile),
            alignment=16,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator

        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile

        # ---- the conv's operands, read before the GEMM (nothing earlier in the step writes them)
        P = self.P
        MAX_REQ = self.token_tile // P
        n_req = cute.size(output, mode=[0]) // P
        conv_lane = cutlass.Int32(0)
        if warp_idx == self.compute_warp_base:
            if lane_idx < self.output_tile:
                if output_tile_idx * self.output_tile < CONV_DIM:
                    conv_lane = cutlass.Int32(1)
        c64 = cutlass.Int64(output_tile_idx * self.output_tile + lane_idx % self.output_tile)
        wbase = conv_w + c64 * (CONV_WIDTH * 2)
        w0 = _ldg_bf16p(wbase, conv_lane)
        w1 = _ldg_bf16p(wbase + 2, conv_lane)
        w2 = _ldg_bf16p(wbase + 4, conv_lane)
        w3 = _ldg_bf16p(wbase + 6, conv_lane)
        req_live = []
        req_cols = []
        req_win = []
        for s in cutlass.range_constexpr(MAX_REQ):
            ok = conv_lane
            if s >= n_req:
                ok = cutlass.Int32(0)
            slot = _ldg_i32p(slots + s * 4, ok, -1)
            live = ok
            if slot == pad_slot:
                live = cutlass.Int32(0)
            prow = pool + (cutlass.Int64(slot) * pool_s_seq + c64 * pool_s_dim) * 2
            req_cols.append(
                (
                    _ldg_bf16p(prow, live),
                    _ldg_bf16p(prow + pool_s_tok * 2, live),
                    _ldg_bf16p(prow + pool_s_tok * 4, live),
                    prow,
                )
            )
            req_win.append(cutlass.Int64(_ldg_i32p(win_rows + s * 4, live, 0)))
            req_live.append(live)

        if warp_idx < _A_LOADER_WARPS:
            _fw._issue_weight_tile(
                sA,
                weight,
                tidx,
                output_tile_idx,
                0,
                0,
                cutlass.Int64(0),
                self.output_tile,
                self.k_tile,
                False,
            )
        if warp_idx == self.compute_warp_base:
            if lane_idx < self.stages:
                cute.arch.mbarrier_init(bar_full + lane_idx, self.load_threads)
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_init(bar_empty + lane_idx, _COMPUTE_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx < _A_LOADER_WARPS:
            weight_policy = _fw._l2_policy_evict_first()
            empty_phase = cutlass.Int32(1)
            prims.cp_async_mbarrier_arrive(bar_full.llvm_ptr, noinc=True)
            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                _fw._issue_weight_tile(
                    sA,
                    weight,
                    tidx,
                    output_tile_idx,
                    k_tile,
                    stage,
                    weight_policy,
                    self.output_tile,
                    self.k_tile,
                    True,
                )
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx < self.compute_warp_base:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            b_loader_threads = self.b_loader_warps * 32
            local_tid = tidx - _A_LOADER_WARPS * 32
            copy_count = self.token_tile * self.k_tile // (8 * b_loader_threads)
            empty_phase = cutlass.Int32(1)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                for copy_idx in cutlass.range_constexpr(copy_count):
                    linear = local_tid * 8 + copy_idx * b_loader_threads * 8
                    row = linear // self.k_tile
                    col = linear % self.k_tile
                    global_row = token_tile_idx * self.token_tile + row
                    if global_row < cute.size(activation, mode=[0]):
                        dst = (
                            sB.iterator.raw_ptr()
                            + stage * self.token_tile * self.k_tile
                            + row * self.k_tile
                            + (col ^ ((row % 8) * 8))
                        )
                        src = activation.iterator.raw_ptr() + activation.layout(
                            (global_row, k_tile * self.k_tile + col)
                        )
                        prims.cp_async_shared_global(dst, src, 16, "cg")
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
        else:
            mma_tid = tidx - self.compute_warp_base * cute.arch.WARP_SIZE
            compute_warp = warp_idx - self.compute_warp_base
            thr_mma = tiled_mma.get_slice(mma_tid)
            acc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.output_tile, self.token_tile)), cutlass.Float32
            )
            acc.fill(0.0)
            k_phases = self.k_tile // (_COMPUTE_WARPS * _MMA_SHAPE[2])
            m_iters = self.output_tile // _MMA_SHAPE[0]
            n_iters = self.token_tile // _MMA_SHAPE[1]
            a_offsets = tuple(
                tuple(
                    _fw._a_fragment_offset(lane_idx, compute_warp, m_iter, phase, self.k_tile)
                    for phase in range(k_phases)
                )
                for m_iter in range(m_iters)
            )
            b_offsets = tuple(
                tuple(
                    _fw._b_pair_offset(
                        lane_idx, compute_warp, n_iter, pair, self.token_tile, self.k_tile
                    )
                    for pair in range(k_phases // 2)
                )
                for n_iter in range(n_iters)
            )
            a_stage_elems = self.output_tile * self.k_tile
            b_stage_elems = self.token_tile * self.k_tile
            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll_full=True):
                stage = k_tile % self.stages
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                a_base = stage * a_stage_elems
                b_base = stage * b_stage_elems
                a_registers = tuple(
                    tuple(
                        _fw._ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase])
                        for phase in range(k_phases)
                    )
                    for m_iter in range(m_iters)
                )
                b_registers = tuple(
                    tuple(
                        _fw._ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair])
                        for pair in range(k_phases // 2)
                    )
                    for n_iter in range(n_iters)
                )
                for phase in cutlass.range_constexpr(k_phases):
                    for n_iter in cutlass.range_constexpr(n_iters):
                        packed_b = b_registers[n_iter][phase // 2]
                        b_offset = (phase % 2) * 2
                        for m_iter in cutlass.range_constexpr(m_iters):
                            acc_base = (m_iter * n_iters + n_iter) * 4
                            result = prims.mma_sync(
                                llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 4),
                                shape=_MMA_SHAPE,
                                layout_a=prims.MMALayout.ROW,
                                layout_b=prims.MMALayout.COL,
                                operand_a=[
                                    a_registers[m_iter][phase][i].ir_value() for i in range(4)
                                ],
                                operand_b=[packed_b[b_offset + i].ir_value() for i in range(2)],
                                operand_c=[acc[acc_base + i].ir_value() for i in range(4)],
                                multiplicand_a_ptx_type=prims.MMAType.BF16,
                                multiplicand_b_ptx_type=prims.MMAType.BF16,
                            )
                            for i in cutlass.range_constexpr(4):
                                acc[acc_base + i] = cutlass.Float32(
                                    llvm.extractvalue(cutlass.Float32.mlir_type, result, [i])
                                )
                if const_expr(reuse_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_empty + stage)
                if stage == self.stages - 1:
                    full_phase = full_phase ^ 1
            for value in cutlass.range_constexpr(values_per_lane):
                atom = value % 4
                fragment = value // 4
                m_iter = fragment // n_iters
                n_iter = fragment % n_iters
                feature = lane_idx // 4 + 8 * (atom // 2) + m_iter * _MMA_SHAPE[0]
                token = 8 * n_iter + 2 * (lane_idx % 4) + atom % 2
                partials[
                    compute_warp * partial_stride
                    + _fw._partial_offset(feature, token, self.token_tile)
                ] = acc[value]
            prims.barrier_cta_sync(1, thread_count=_COMPUTE_WARPS * 32)

        if warp_idx == self.compute_warp_base:
            final_acc = cute.make_rmem_tensor(cute.make_layout(values_per_lane), cutlass.Float32)
            for value in cutlass.range_constexpr(values_per_lane):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                total = cutlass.Float32(0)
                for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                    total = (
                        total
                        + partials[
                            peer * partial_stride
                            + _fw._partial_offset(feature, token, self.token_tile)
                        ]
                    )
                final_acc[value] = total

            output_base = output_tile_idx * self.output_tile
            token_base = token_tile_idx * self.token_tile
            for value in cutlass.range_constexpr(values_per_lane):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                if token_base + token < cute.size(output, mode=[0]):
                    output[token_base + token, output_base + feature] = final_acc[value].to(
                        output.element_type
                    )
            # ---- the conv epilogue -----------------------------------------------------------
            if output_base < CONV_DIM:
                for value in cutlass.range_constexpr(values_per_lane):
                    linear = value * cute.arch.WARP_SIZE + lane_idx
                    feature = linear % self.output_tile
                    token = linear // self.output_tile
                    xtile[token * self.output_tile + feature] = _round_bf16(final_acc[value])
                cute.arch.sync_warp()
                for s in cutlass.range_constexpr(MAX_REQ):
                    if req_live[s] != 0:
                        # One channel's conv over request s.
                        col0, col1, col2, prow = req_cols[s]
                        wb = win + (req_win[s] * win_s_seq + c64 * win_s_dim) * 2
                        for t in cutlass.range_constexpr(P):
                            x = xtile[(s * P + t) * self.output_tile + lane_idx]
                            a = _add_rn(cutlass.Float32(0.0), _mul_bf16(col0, w0))
                            a = _add_rn(a, _mul_bf16(col1, w1))
                            a = _add_rn(a, _mul_bf16(col2, w2))
                            a = _add_rn(a, _mul_bf16(x, w3))
                            col0 = col1
                            col1 = col2
                            col2 = x
                            wt = wb + cutlass.Int64(t) * win_s_step * 2
                            _stg_bf16(wt, col0)
                            _stg_bf16(wt + win_s_win * 2, col1)
                            _stg_bf16(wt + win_s_win * 4, col2)
                            _stg_bf16(
                                conv_out + (cutlass.Int64(s * P + t) * CONV_DIM + c64) * 2,
                                _silu(a),
                            )
                        # The pool advanced to the last 3 inputs of [window | rows].
                        _stg_bf16(prow, col0)
                        _stg_bf16(prow + pool_s_tok * 2, col1)
                        _stg_bf16(prow + pool_s_tok * 4, col2)


def fused_tactic(rows: int):
    """The FlashInfer warp split-K tactic `low_m_bf16_mm` runs this projection with at `rows`
    rows, when it is that kernel in one token tile (the fused launch's domain), else None."""
    p = _lowm.plan(rows, N_OUT, K_IN)
    if p.kind != "warp":
        return None
    tactic = _fw.WarpSplitKTactic(*p.tactic)
    return tactic if rows <= tactic.token_tile else None


def _from_dlpack(tensor: torch.Tensor, *, dynamic_m: bool = False):
    return _fw._from_dlpack(tensor, dynamic_m=dynamic_m)


@functools.cache
def _compile(device_index: int, tactic, use_pdl: bool, rows_per_request: int):
    with torch.cuda.device(device_index):
        i64 = cutlass.Int64(0)
        return cute_ext.compile(
            _InProjConvKernel(tactic, use_pdl, rows_per_request),
            *(
                _from_dlpack(
                    torch.empty(shape, device="cuda", dtype=torch.bfloat16), dynamic_m=dynamic_m
                )
                for shape, dynamic_m in (
                    ((N_OUT, K_IN), False),
                    ((_fw._MAX_M, K_IN), True),
                    ((N_OUT,), False),
                    ((_fw._MAX_M, N_OUT), True),
                )
            ),  # fmt: skip
            *([i64] * 13),
            cutlass.Int32(0),
            _cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


def _launch(
    mat_a,
    mat_b,
    out,
    conv_out,
    conv_weight,
    conv_state,
    slots,
    window,
    window_rows,
    pad_slot_id: int,
    rows_per_request: int,
    tactic,
    pdl: bool,
):
    compiled = _compile(mat_a.device.index, tactic, pdl, rows_per_request)
    compiled(
        _from_dlpack(mat_b.T),
        _from_dlpack(mat_a, dynamic_m=True),
        _from_dlpack(out[0]),
        _from_dlpack(out, dynamic_m=True),
        conv_weight.data_ptr(),
        conv_state.data_ptr(),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        slots.data_ptr(),
        window.data_ptr(),
        window.stride(0),
        window.stride(1),
        window.stride(2),
        window.stride(3),
        window_rows.data_ptr(),
        conv_out.data_ptr(),
        pad_slot_id,
        _cuda.CUstream(torch.cuda.current_stream(mat_a.device).cuda_stream),
    )


def _cells():
    return tuple(
        Cell(
            why=f"tep4 MTP verify, {n} request(s) x {p} rows: in_proj + conv in one launch",
            spec=dict(requests=n, p=p, capture=capture),
        )
        # One per warp split-K tactic `fused_tactic` picks (3, 6 and 16 rows), the first captured.
        for n, p, capture in ((1, 3, True), (2, 3, False), (4, 4, False))
    )


class GdnInProjConv(OpWrapper):
    """`(proj, conv) = gdn_in_proj_conv(x, in_proj, conv_weight, conv_state, slots, window,
    window_rows, pad_slot_id, rows_per_request, tactic, pdl)` for a verify step's rows.

    `x` `[T, 2560]` bf16 contiguous, `T = n * rows_per_request`; `in_proj` the `[2560, 4128]`
    column-major operand; `conv_weight` `[2560, 4]`; `conv_state` the `[slots, 2560, 3]` pool;
    `slots` `[n]` int32 (`pad_slot_id` skipped); `window` the `[rows, p, 2560, 3]` snapshot
    buffer and `window_rows` `[n]` int32; `tactic` the warp split-K tactic `low_m_bf16_mm` picks
    at `T` rows (`fused_tactic`). Returns `proj` `[T, 4128]`, bitwise `low_m_bf16_mm`'s, and
    `conv` `[T, 2560]` token-major with `conv_state` and `window` written, bitwise
    `causal_conv1d_token_major` over `proj`'s q|k|v columns.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Source: FlashInfer's `CpAsyncWarpSplitKKernel` (CuTe DSL) plus the conv epilogue, compiled
    per (tactic, PDL, rows per request) with M dynamic: the warmup forward must compile it
    before capture. Preconditions, refused by `is_valid`: the shapes above, `T` a multiple of
    `rows_per_request`, `T <= tactic.token_tile`.
    """

    def __call__(
        self,
        x,
        in_proj,
        conv_weight,
        conv_state,
        slots,
        window,
        window_rows,
        pad_slot_id: int,
        rows_per_request: int,
        tactic,
        pdl: bool,
    ):
        rows = x.shape[0]
        out = torch.empty([rows, N_OUT], dtype=torch.bfloat16, device=x.device)
        conv = torch.empty([rows, CONV_DIM], dtype=torch.bfloat16, device=x.device)
        _launch(
            x, in_proj, out, conv, conv_weight, conv_state, slots, window, window_rows,
            pad_slot_id, rows_per_request, tactic, pdl,
        )  # fmt: skip
        return out, conv

    def reference(
        self,
        x,
        in_proj,
        conv_weight,
        conv_state,
        slots,
        window,
        window_rows,
        pad_slot_id: int,
        rows_per_request: int,
        tactic,
        pdl: bool,
    ):
        """fp32 projection rounded to bf16, then the conv's per-element arithmetic in torch
        (writes `conv_state` and `window`: call it on copies)."""
        with true_fp32_matmul():
            out = (x.float() @ in_proj.float()).to(torch.bfloat16)
        rows = x.shape[0]
        p = rows_per_request
        conv = torch.zeros([rows, CONV_DIM], dtype=torch.bfloat16, device=x.device)
        w = conv_weight.float()
        for s in range(rows // p):
            slot = int(slots[s])
            if slot == pad_slot_id:
                continue
            cols = [conv_state[slot, :, j].float() for j in range(CONV_WIDTH - 1)]
            for t in range(p):
                xv = out[s * p + t, :CONV_DIM].float()
                acc = torch.zeros(CONV_DIM, dtype=torch.float32, device=x.device)
                for j, v in enumerate(cols + [xv]):
                    acc = acc + (v * w[:, j]).to(torch.bfloat16).float()
                cols = cols[1:] + [xv]
                for j in range(CONV_WIDTH - 1):
                    window[int(window_rows[s]), t, :, j] = cols[j].to(window.dtype)
                conv[s * p + t] = (acc / (1 + torch.exp(-acc))).to(torch.bfloat16)
            for j in range(CONV_WIDTH - 1):
                conv_state[slot, :, j] = cols[j].to(conv_state.dtype)
        return out, conv

    def is_valid(
        self,
        x,
        in_proj,
        conv_weight,
        conv_state,
        slots,
        window,
        window_rows,
        pad_slot_id: int,
        rows_per_request: int,
        tactic,
        pdl: bool,
    ) -> None:
        rows = x.shape[0]
        assert x.dtype == in_proj.dtype == torch.bfloat16 and x.is_contiguous()
        assert tuple(x.shape) == (rows, K_IN) and tuple(in_proj.shape) == (K_IN, N_OUT)
        assert in_proj.stride() == (1, K_IN), "in_proj is the column-major [2560, 4128] operand"
        assert rows % rows_per_request == 0 and 1 <= rows <= tactic.token_tile, (rows, tactic)
        assert conv_weight.dtype == conv_state.dtype == window.dtype == torch.bfloat16, (
            "the epilogue reads and writes the conv taps, pool and window as bf16"
        )
        assert tuple(conv_weight.shape) == (CONV_DIM, CONV_WIDTH) and conv_weight.is_contiguous()
        assert conv_state.dim() == 3 and tuple(conv_state.shape[1:]) == (CONV_DIM, CONV_WIDTH - 1)
        assert window.dim() == 4 and tuple(window.shape[2:]) == (CONV_DIM, CONV_WIDTH - 1)
        assert window.shape[1] >= rows_per_request
        assert slots.dtype == window_rows.dtype == torch.int32
        n = rows // rows_per_request
        assert slots.shape[0] >= n and window_rows.shape[0] >= n


gdn_in_proj_conv = GdnInProjConv()
