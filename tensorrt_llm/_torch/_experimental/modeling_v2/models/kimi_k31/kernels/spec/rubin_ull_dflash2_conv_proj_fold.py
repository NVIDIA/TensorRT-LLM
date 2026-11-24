# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm@e3ddbdbd60 vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/gemm/warp_splitk.py
# (from FlashInfer dense_bf16_gemm_warp_splitk.py); changed: the side-0 convolution runs in the epilogue.
"""The DFlash2 drafter's conv kernel projection with the side-0 convolution in its epilogue (CuTe DSL).

One launch computes ``coef = x @ weight^T`` (``weight`` the ``[2 * taps * groups, hidden]`` kernel projection,
``coef`` its ``[rows, 2 * taps * groups]`` output as the projection writes it) and the side-0 convolution of the
same ``x``::

    conv[t, c] = sum_i (base[i, c] + coef[t, 0, i, c // group]) * x[t - i, c]

where ``coef[t, 0, i, g]`` is ``coef``'s column ``i * groups + g`` rounded to BF16 (the value the projection
stores) and a tap past the start of row ``t``'s block (``t % block_size < i``) multiplies an explicit ``0.0``.

The GEMM is the vendored warp split-K (``kernels/vendor/vllm_main/.../gemm/warp_splitk.py``,
``CpAsyncWarpSplitKKernel``) with two changes:

* **Paired output rows.** Output tile ``T`` (``output_tile`` weight rows) holds, for each of ``output_tile // taps``
  groups, all ``taps`` rows of that group: local row ``r`` is the weight row
  ``side * taps * groups + (r % taps) * groups + (T % tiles_per_side) * (output_tile // taps) + r // taps``, where
  ``side = T // tiles_per_side``. The loader reads that row and the epilogue stores the result at that column, so
  ``coef`` is laid out (and every value accumulated) exactly as the plain kernel does: each output sums the same
  BF16 products in the same per-warp K order and the same 4-warp reduction order.
* **Convolution epilogue.** After the 4-warp reduction, the tiles of side 0 (``T < tiles_per_side``) hold every
  coefficient of their ``output_tile // taps`` groups for all ``token_tile`` rows. The reduce warp puts the
  BF16-rounded coefficients in shared memory; then each of the 128 compute threads convolves one row and 8
  channels (one 16-byte vector) of those groups. The convolution reads ``x`` (loaded right after the compute warps'
  PDL wait, so its L2 latency is hidden behind the K loop) and ``base`` (loaded before it). Each output channel is
  computed by exactly one thread of one CTA: nothing is recomputed.

The convolution arithmetic is the plain kernel's (``kernels/spec/dflash2_grouped_conv.py``): per tap
``base + delta`` (``add.rn``), tap 0's product (``mul.rn``), tap 1's product of the selected ``x`` (``mul.rn``)
added (``add.rn``), never contracted to an FMA, rounded once to BF16 (``cvt.rn``). So both outputs are bitwise
the warp split-K GEMM followed by ``dflash2_grouped_conv``.

Fixed shape class: ``taps == 2``, ``group_size == 16``, ``output_tile == 16``, ``token_tile == 8`` (one request's
block of 8 rows), so the 128 compute threads cover ``8 rows x 8 groups x 16 channels`` exactly.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from ..vendor.vllm_main import module as _vllm_main

_w = _vllm_main("cutedsl.gemm.warp_splitk")

#: Channels a thread convolves: one 16-byte vector of BF16.
VEC = 8
TAPS = 2
GROUP = 16
OUTPUT_TILE = 16
TOKEN_TILE = 8
_CONV_BARRIER = 2


def _asm(res, operands, text, constraints, side_effects, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _address(pointer, *, loc=None, ip=None):
    """The 64-bit global address of a CuTe pointer."""
    return pointer.toint(loc=loc, ip=ip)


@dsl_user_op
def _ld_v4(addr, pred, cache: str, *, loc=None, ip=None):
    """16 bytes at ``addr`` with ``ld.global.<cache>`` (zeros where ``pred`` is 0, memory untouched)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\n"
        f"mov.b32 $3, 0;\n@p ld.global.{cache}.v4.b32 {{$0, $1, $2, $3}}, [$4];\n}}",
        "=r,=r,=r,=r,l,r",
        True,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_v4(addr, words, pred, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in words]
        + [Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}",
        "l,r,r,r,r,r",
        True,
        loc=loc,
        ip=ip,
    )


def _conv8_ptx() -> str:
    """Operands: $0-$3 out (bf16x2), $4-$7 x[t], $8-$11 x[t - 1], $12-$15 base tap 0, $16-$19 base tap 1,
    $20 delta tap 0, $21 delta tap 1 (FP32), $22 keep (tap 1 inside the block)."""
    lines = [
        "{",
        ".reg .pred kp;",
        ".reg .b32 t;",
        ".reg .f32 xa, xb, ba, bb, c, m, s;",
        ".reg .f32 a<8>;",
        "setp.ne.s32 kp, $22, 0;",
    ]
    for k in range(VEC):
        j, hi = k // 2, k % 2

        def widen(dst, src):
            # BF16 -> FP32 is exact: the element's bits in the high half of the word.
            op = f"and.b32 t, {src}, 0xFFFF0000;" if hi else f"shl.b32 t, {src}, 16;"
            return f"{op}\nmov.b32 {dst}, t;"

        lines += [
            widen("xa", f"${4 + j}"),
            widen("ba", f"${12 + j}"),
            "add.rn.f32 c, ba, $20;",
            f"mul.rn.f32 a{k}, c, xa;",
            widen("xb", f"${8 + j}"),
            widen("bb", f"${16 + j}"),
            # Select, not multiply: a tap past the block start multiplies an explicit 0.0.
            "selp.f32 s, xb, 0f00000000, kp;",
            "add.rn.f32 c, bb, $21;",
            "mul.rn.f32 m, c, s;",
            f"add.rn.f32 a{k}, a{k}, m;",
        ]
    lines += [f"cvt.rn.bf16x2.f32 ${j}, a{2 * j + 1}, a{2 * j};" for j in range(VEC // 2)]
    lines.append("}")
    return "\n".join(lines)


_CONV8 = _conv8_ptx()


@dsl_user_op
def _conv8(x0, x1, b0, b1, d0, d1, keep, *, loc=None, ip=None):
    """8 channels of the 2-tap convolution, packed BF16 in and out (see ``_conv8_ptx``)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(v).ir_value(loc=loc, ip=ip) for v in (*x0, *x1, *b0, *b1)]
        + [Float32(d0).ir_value(loc=loc, ip=ip), Float32(d1).ir_value(loc=loc, ip=ip)]
        + [Int32(keep).ir_value(loc=loc, ip=ip)],
        _CONV8,
        "=r,=r,=r,=r," + ",".join(["r"] * 16) + ",f,f,r",
        False,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _paired_row(tile, row, tiles_per_side, groups, *, loc=None, ip=None):
    """Weight row (= output column) of local row ``row`` of output tile ``tile`` (module docstring)."""
    side = tile // tiles_per_side
    return (
        side * (TAPS * groups)
        + (row % TAPS) * groups
        + (tile - side * tiles_per_side) * (OUTPUT_TILE // TAPS)
        + row // TAPS
    )


@dsl_user_op
def _issue_weight_tile_paired(
    smem, weight, tidx, tile, k_tile, stage, policy, k_tile_size, with_policy, tiles_per_side, groups,
    *, loc=None, ip=None,
):  # fmt: skip
    """``warp_splitk._issue_weight_tile`` with the paired weight rows (plain Python control flow)."""
    copy_count = OUTPUT_TILE * k_tile_size // (8 * _w._A_LOADER_THREADS)
    for copy_idx in range(copy_count):
        linear = tidx * 8 + copy_idx * _w._A_LOADER_THREADS * 8
        row = linear // k_tile_size
        col = linear % k_tile_size
        dst_offset = stage * OUTPUT_TILE * k_tile_size + row * k_tile_size + (col ^ ((row % 8) * 8))
        src_row = _paired_row(tile, row, tiles_per_side, groups, loc=loc, ip=ip)
        src_offset = weight.layout((src_row, k_tile * k_tile_size + col))
        if with_policy:
            _w._cp_async_16_with_policy(
                smem.iterator + dst_offset, weight.iterator + src_offset, policy, loc=loc, ip=ip
            )
        else:
            prims.cp_async_shared_global(
                smem.iterator.raw_ptr() + dst_offset,
                weight.iterator.raw_ptr() + src_offset,
                16,
                "cg",
                loc=loc,
                ip=ip,
            )


def smem_bytes(tactic) -> int:
    """The vendored kernel's shared memory plus the BF16-rounded coefficients (FP32) of one tile."""
    return _w._align_up(_w._smem_bytes(tactic), 16) + OUTPUT_TILE * TOKEN_TILE * 4


class ConvProjWarpSplitKKernel:
    """``coef = x @ weight^T`` and ``conv = Conv(x; base, coef side 0)`` in one PDL launch (module docstring)."""

    def __init__(self, tactic, hidden: int, groups: int) -> None:
        if (tactic.output_tile, tactic.token_tile) != (OUTPUT_TILE, TOKEN_TILE):
            raise ValueError(f"the conv epilogue needs output tile {OUTPUT_TILE}, token tile {TOKEN_TILE}")
        if groups * GROUP != hidden or groups % (OUTPUT_TILE // TAPS):
            raise ValueError(f"{groups} groups of {GROUP} for hidden {hidden}")
        self.output_tile = tactic.output_tile
        self.token_tile = tactic.token_tile
        self.k_tile = tactic.k_tile
        self.stages = tactic.stages
        self.b_loader_warps = tactic.b_loader_warps
        self.compute_warp_base = _w._A_LOADER_WARPS + self.b_loader_warps
        self.threads = (self.compute_warp_base + _w._COMPUTE_WARPS) * 32
        self.load_threads = self.compute_warp_base * 32
        self.hidden = hidden
        self.groups = groups
        self.tiles_per_side = groups // (OUTPUT_TILE // TAPS)
        self.smem_bytes = smem_bytes(tactic)
        # Compute thread -> (row, vector): 16 vectors (8 groups x 16 channels) per row.
        self.vectors_per_row = (OUTPUT_TILE // TAPS) * GROUP // VEC
        assert self.vectors_per_row * TOKEN_TILE == _w._COMPUTE_WARPS * 32

    @cute.experimental.jit
    def __call__(self, weight, activation, base, output, conv_out, block_size: Int32, stream: _cuda.CUstream):
        sA_layout = _w._make_smem_layout(weight.element_type, self.output_tile, self.k_tile, self.stages)
        sB_layout = _w._make_smem_layout(activation.element_type, self.token_tile, self.k_tile, self.stages)
        tiled_mma = cute.make_tiled_mma(
            _w.warp.MmaF16BF16Op(weight.element_type, cutlass.Float32, _w._MMA_SHAPE),
            cute.make_layout((1, 1, _w._COMPUTE_WARPS)),
            permutation_mnk=(self.output_tile, self.token_tile, self.k_tile),
        )
        self.kernel(
            weight, activation, base, output, conv_out, block_size, sA_layout, sB_layout, tiled_mma
        ).launch(
            grid=(
                cute.ceil_div(weight.shape[0], self.output_tile),
                cute.ceil_div(activation.shape[0], self.token_tile),
                1,
            ),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        activation: cute.Tensor,
        base: cute.Tensor,  # [taps, hidden]
        output: cute.Tensor,
        conv_out: cute.Tensor,  # [rows, hidden]
        block_size: Int32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, token_tile_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        reuse_stages = cute.size(weight, mode=[1]) > self.stages * self.k_tile

        sA = cute_ext.allocate(weight.element_type, cute.AddressSpace.smem, sA_layout, alignment=1024)
        sB = cute_ext.allocate(activation.element_type, cute.AddressSpace.smem, sB_layout, alignment=1024)
        values_per_lane = self.output_tile * self.token_tile // cute.arch.WARP_SIZE
        partial_stride = (
            self.output_tile * self.token_tile // _w._MAILBOX_GROUP_VALUES
        ) * _w._MAILBOX_PADDED_GROUP_VALUES
        partials = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_w._COMPUTE_WARPS * partial_stride),
            alignment=16,
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(self.stages), alignment=8
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(self.stages), alignment=8
        )
        deltas = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.output_tile * self.token_tile),
            alignment=16,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator

        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile

        if warp_idx < _w._A_LOADER_WARPS:
            _issue_weight_tile_paired(
                sA, weight, tidx, output_tile_idx, 0, 0, cutlass.Int64(0), self.k_tile, False,
                self.tiles_per_side, self.groups,
            )  # fmt: skip
        if warp_idx == self.compute_warp_base:  # noqa: SIM102 (DSL: no dynamic and)
            if lane_idx < self.stages:
                cute.arch.mbarrier_init(bar_full + lane_idx, self.load_threads)
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_init(bar_empty + lane_idx, _w._COMPUTE_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx < _w._A_LOADER_WARPS:
            weight_policy = _w._l2_policy_evict_first()
            empty_phase = cutlass.Int32(1)
            prims.cp_async_mbarrier_arrive(bar_full.llvm_ptr, noinc=True)
            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                _issue_weight_tile_paired(
                    sA, weight, tidx, output_tile_idx, k_tile, stage, weight_policy, self.k_tile, True,
                    self.tiles_per_side, self.groups,
                )  # fmt: skip
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx < self.compute_warp_base:
            cute.arch.griddepcontrol_wait()
            b_loader_threads = self.b_loader_warps * 32
            local_tid = tidx - _w._A_LOADER_WARPS * 32
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
            cute.arch.griddepcontrol_launch_dependents()
        else:
            mma_tid = tidx - self.compute_warp_base * cute.arch.WARP_SIZE
            compute_warp = warp_idx - self.compute_warp_base
            # The convolution operands of this thread: row conv_row, channels conv_channel .. + 8. Every tile
            # loads them (side-1 tiles at in-range channels, unused) so no load sits under a branch.
            rows = cute.size(activation, mode=[0])
            conv_token = mma_tid // self.vectors_per_row
            conv_vec = mma_tid - conv_token * self.vectors_per_row
            conv_group = conv_vec // (GROUP // VEC)
            conv_channel = (
                (output_tile_idx % self.tiles_per_side) * (OUTPUT_TILE // TAPS) + conv_group
            ) * GROUP + (conv_vec % (GROUP // VEC)) * VEC
            conv_row = token_tile_idx * self.token_tile + conv_token
            row_ok = Int32(0)
            if conv_row < rows:
                row_ok = Int32(1)
            keep = Int32(0)
            prev_row = conv_row
            if conv_row % block_size >= 1:
                keep = Int32(1)
                prev_row = conv_row - 1
            base0 = _ld_v4(_address(base.iterator + base.layout((0, conv_channel))), Int32(1), "nc")
            base1 = _ld_v4(_address(base.iterator + base.layout((1, conv_channel))), Int32(1), "nc")
            # x is the previous kernel's output: wait for it, then read it through L2 only.
            cute.arch.griddepcontrol_wait()
            x0 = _ld_v4(
                _address(activation.iterator + activation.layout((conv_row, conv_channel))), row_ok, "cg"
            )
            x1 = _ld_v4(
                _address(activation.iterator + activation.layout((prev_row, conv_channel))), row_ok, "cg"
            )

            thr_mma = tiled_mma.get_slice(mma_tid)
            acc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.output_tile, self.token_tile)), cutlass.Float32
            )
            acc.fill(0.0)
            k_phases = self.k_tile // (_w._COMPUTE_WARPS * _w._MMA_SHAPE[2])
            m_iters = self.output_tile // _w._MMA_SHAPE[0]
            n_iters = self.token_tile // _w._MMA_SHAPE[1]
            a_offsets = tuple(
                tuple(
                    _w._a_fragment_offset(lane_idx, compute_warp, m_iter, phase, self.k_tile)
                    for phase in range(k_phases)
                )
                for m_iter in range(m_iters)
            )
            b_offsets = tuple(
                tuple(
                    _w._b_pair_offset(
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
                        _w._ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase])
                        for phase in range(k_phases)
                    )
                    for m_iter in range(m_iters)
                )
                b_registers = tuple(
                    tuple(
                        _w._ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair])
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
                                shape=_w._MMA_SHAPE,
                                layout_a=prims.MMALayout.ROW,
                                layout_b=prims.MMALayout.COL,
                                operand_a=[a_registers[m_iter][phase][i].ir_value() for i in range(4)],
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
                feature = lane_idx // 4 + 8 * (atom // 2) + m_iter * _w._MMA_SHAPE[0]
                token = 8 * n_iter + 2 * (lane_idx % 4) + atom % 2
                partials[
                    compute_warp * partial_stride + _w._partial_offset(feature, token, self.token_tile)
                ] = acc[value]
            prims.barrier_cta_sync(1, thread_count=_w._COMPUTE_WARPS * 32)

            if compute_warp == 0:
                out_token_base = token_tile_idx * self.token_tile
                for r_value in cutlass.range_constexpr(values_per_lane):
                    r_linear = r_value * cute.arch.WARP_SIZE + lane_idx
                    r_feature = r_linear % self.output_tile
                    r_token = r_linear // self.output_tile
                    total = cutlass.Float32(0)
                    for peer in cutlass.range_constexpr(_w._COMPUTE_WARPS):
                        total = (
                            total
                            + partials[
                                peer * partial_stride
                                + _w._partial_offset(r_feature, r_token, self.token_tile)
                            ]
                        )
                    rounded = total.to(output.element_type)
                    # The coefficient the convolution reads: the value the projection stores, widened.
                    deltas[r_feature * self.token_tile + r_token] = Float32(rounded)
                    if out_token_base + r_token < cute.size(output, mode=[0]):
                        column = _paired_row(output_tile_idx, r_feature, self.tiles_per_side, self.groups)
                        output[out_token_base + r_token, column] = rounded
            prims.barrier_cta_sync(_CONV_BARRIER, thread_count=_w._COMPUTE_WARPS * 32)

            if output_tile_idx < self.tiles_per_side:
                d0 = deltas[(conv_group * TAPS) * self.token_tile + conv_token]
                d1 = deltas[(conv_group * TAPS + 1) * self.token_tile + conv_token]
                words = _conv8(x0, x1, base0, base1, d0, d1, keep)
                _st_v4(
                    _address(conv_out.iterator + conv_out.layout((conv_row, conv_channel))), words, row_ok
                )


def _view(tensor, *, dynamic_m=False):
    return _w._view(tensor, dynamic_m=dynamic_m)


@functools.cache
def compile_conv_proj(device_index: int, n: int, k: int, tactic):
    """Code for an ``[n, k]`` kernel projection (``n = 2 * taps * k // group``) with dynamic M (at most
    ``TOKEN_TILE`` rows), PDL on."""
    _w.check_tactic(tactic, n, k)
    groups = k // GROUP
    if n != 2 * TAPS * groups:
        raise ValueError(f"the kernel projection of hidden {k} has {2 * TAPS * groups} rows, not {n}")
    if smem_bytes(tactic) > _w._SMEM_CAPACITY:
        raise ValueError(f"{tactic} needs {smem_bytes(tactic)} B of shared memory")
    dtype = torch.bfloat16
    with torch.cuda.device(device_index):
        shapes = (((n, k), False), ((TOKEN_TILE, k), True), ((TAPS, k), False), ((TOKEN_TILE, n), True),
                  ((TOKEN_TILE, k), True))  # fmt: skip
        args = tuple(_view(torch.empty(s, device="cuda", dtype=dtype), dynamic_m=d) for s, d in shapes)
        stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute_ext.compile(
            ConvProjWarpSplitKKernel(tactic, k, groups), *args, Int32(TOKEN_TILE), stream
        )


class PreparedConvProj:
    """``coef = x @ weight^T`` and ``conv = Conv(x; base, coef side 0)`` for at most ``TOKEN_TILE`` rows of one
    request's block; ``x``, ``coef`` and ``conv`` packed row-major, 32 B aligned."""

    def __init__(self, weight: torch.Tensor, tactic) -> None:
        n, k = weight.shape
        if weight.dtype != torch.bfloat16 or weight.stride() != (k, 1) or weight.data_ptr() % 32:
            raise ValueError("the conv kernel projection is a packed, 32 B aligned BF16 [N, K]")
        self.weight, self.tactic = weight, tactic
        self.code = compile_conv_proj(weight.device.index, n, k, tactic)
        self.weight_view = _view(weight)

    def run(self, x, base, coef, conv, block_size: int):
        stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        self.code(self.weight_view, _view(x, dynamic_m=True), _view(base), _view(coef, dynamic_m=True),
                  _view(conv, dynamic_m=True), Int32(block_size), stream)  # fmt: skip
        return coef, conv


__all__ = ["ConvProjWarpSplitKKernel", "PreparedConvProj", "compile_conv_proj", "smem_bytes"]
