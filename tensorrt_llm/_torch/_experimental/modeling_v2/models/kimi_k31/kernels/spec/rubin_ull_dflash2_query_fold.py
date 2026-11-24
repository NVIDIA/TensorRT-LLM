# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm@e3ddbdbd60 vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/gemm/warp_splitk.py
# (from FlashInfer dense_bf16_gemm_warp_splitk.py); changed: q_a RMSNorm, q_b, W_UK absorption, RoPE and the block
# latent store in one launch.
"""The DFlash2 drafter's whole query path and block latent store of one MLA layer in one launch (CuTe DSL).

Inputs are the layer's ``[q_a | kv_a]`` projection ``q_kv`` (``[rows, q_lora + kv_lora + rope]``, rows of one
request's block) and the folded query weight ``weight`` (``[heads * (kv_lora + rope), q_lora]``: per head
``(W_UK_h^T q_b_nope_h) * g`` then ``q_b_rope_h * g``, ``g`` the q_a RMSNorm weight, FP64 and rounded once to BF16).
One launch writes:

* ``query[t, h] = [r_t * (x_t @ W'_latent_h^T) | RoPE(r_t * (x_t @ W'_rope_h^T), positions[t])]`` in BF16, where
  ``x_t = q_kv[t, :q_lora]`` and ``r_t = rsqrt(mean(x_t ** 2) + eps)``: the q_a RMSNorm, the q_b projection, the
  W_UK absorption and the query RoPE, rounded to BF16 once before the rotation (as the absorbed query is) and once
  after it;
* the block latent of token ``t`` into the paged pool at ``pages[t] * page_size + offsets[t]``:
  ``[RMSNorm(kv_a) | RoPE(k_pe)]`` saturated to FP8 E4M3, bitwise ``K31DraftMlaBlockRoPEFp8Kernel``'s store.

The query tiles are the vendored warp split-K (``kernels/vendor/vllm_main/.../gemm/warp_splitk.py``,
``CpAsyncWarpSplitKKernel``) over the first ``q_lora`` columns of ``q_kv`` with three changes:

* **The RMSNorm rides on the K loop.** The norm is linear in ``x`` up to the row scale ``r_t`` and its weight is
  folded into ``W'``, so the GEMM runs on the raw ``x``. Each compute lane already holds, in its B fragments, 8
  values of token ``lane // 4`` per ``ldmatrix``; it squares and sums them (FP32 FMA). Two shuffles and a shared
  memory exchange give every token's sum of squares over all ``q_lora`` columns, and the reduce warp scales the
  4-warp sum by ``r_t`` before rounding. Nothing waits for the norm: it adds a few FMAs per K tile.
* **RoPE epilogue.** Output tiles are 16 columns and every head is 36 tiles (``576 / 16``), so tiles 32..35 of a
  head are its rope columns. Rope pairs are adjacent columns (``rope_interleave``), held by adjacent lanes of the
  reduce warp: one shuffle brings the partner. The cos / sin of the lane's columns are loaded before the K loop.
* **Block latent CTA.** One extra CTA (one warp per token) does the block latents' norm, rotation and FP8 store,
  the arithmetic of ``kernels/spec/mla_norm_rope.py`` (torch-order sum of squares), after its PDL wait.

The query is not ``K31DraftMlaBlockRoPEKernel``'s bits: the reference rounds the normalized ``q_a`` to BF16 and
multiplies by ``q_b`` and ``W_UK`` (or the absorbed weight) without the norm weight; here the norm weight is in the
weight, ``q_a`` is never rounded and the sum of squares takes another order. The block latents are bitwise.

Fixed shape class: ``output_tile == 16``, ``token_tile == 8`` (one request's block of at most 8 rows), kv_lora 512,
rope 64.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from ..vendor.vllm_main import module as _vllm_main
from .mla_norm_rope import _E4M3_MAX, _e4m3_byte, _inverse_rms_torch_order, _mul_rn, _norm_column, _rsqrt

_w = _vllm_main("cutedsl.gemm.warp_splitk")

OUTPUT_TILE = 16
TOKEN_TILE = 8
KV_LORA = 512
ROPE = 64
HEAD = KV_LORA + ROPE
TILES_PER_HEAD = HEAD // OUTPUT_TILE
LATENT_TILES = KV_LORA // OUTPUT_TILE


@dsl_user_op
def _ssq4(acc, words, *, loc=None, ip=None):
    """``acc`` plus the squares of the 8 BF16 values packed in 4 words (FP32 FMA, low half first)."""
    lines = ["{", ".reg .b32 t;", ".reg .f32 lo, hi, s;", "mov.f32 s, $1;"]
    for i in range(4):
        lines += [
            f"shl.b32 t, ${2 + i}, 16;",
            "mov.b32 lo, t;",
            f"and.b32 t, ${2 + i}, 0xFFFF0000;",
            "mov.b32 hi, t;",
            "fma.rn.f32 s, lo, lo, s;",
            "fma.rn.f32 s, hi, hi, s;",
        ]
    lines += ["mov.f32 $0, s;", "}"]
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(acc).ir_value(loc=loc, ip=ip)] + [words[i].ir_value(loc=loc, ip=ip) for i in range(4)],
            "\n".join(lines),
            "=f,f,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _fp8_byte(value):
    """``value`` rounded to BF16, saturated to +-448 (NaN passes) and rounded to E4M3: ``to_draft_fp8``."""
    rounded = Float32(value.to(cutlass.BFloat16))
    rounded = Float32(cutlass.select_(rounded > Float32(_E4M3_MAX), Float32(_E4M3_MAX), rounded))
    rounded = Float32(cutlass.select_(rounded < Float32(-_E4M3_MAX), Float32(-_E4M3_MAX), rounded))
    return _e4m3_byte(rounded)


def smem_bytes(tactic) -> int:
    """The vendored kernel's shared memory plus the per-warp, per-token sums of squares."""
    return _w._align_up(_w._smem_bytes(tactic), 16) + _w._COMPUTE_WARPS * TOKEN_TILE * 4


class DraftQueryFoldKernel:
    """Norm + absorbed query projection + RoPE, and the block latent store, in one PDL launch (module docstring)."""

    def __init__(self, tactic, q_lora: int, heads: int, eps: float) -> None:
        if (tactic.output_tile, tactic.token_tile) != (OUTPUT_TILE, TOKEN_TILE):
            raise ValueError(f"the query fold needs output tile {OUTPUT_TILE}, token tile {TOKEN_TILE}")
        self.output_tile = tactic.output_tile
        self.token_tile = tactic.token_tile
        self.k_tile = tactic.k_tile
        self.stages = tactic.stages
        self.b_loader_warps = tactic.b_loader_warps
        self.compute_warp_base = _w._A_LOADER_WARPS + self.b_loader_warps
        self.threads = (self.compute_warp_base + _w._COMPUTE_WARPS) * 32
        self.load_threads = self.compute_warp_base * 32
        if self.threads // 32 < TOKEN_TILE:
            raise ValueError("the block latent CTA needs one warp per token")
        self.q_lora = q_lora
        self.heads = heads
        self.eps = eps
        self.query_tiles = heads * TILES_PER_HEAD
        self.smem_bytes = smem_bytes(tactic)

    @cute.experimental.jit
    def __call__(
        self, weight, activation, output, kv_weight, positions, freqs, pool, pages, offsets, page_size: Int32,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        sA_layout = _w._make_smem_layout(weight.element_type, self.output_tile, self.k_tile, self.stages)
        sB_layout = _w._make_smem_layout(activation.element_type, self.token_tile, self.k_tile, self.stages)
        tiled_mma = cute.make_tiled_mma(
            _w.warp.MmaF16BF16Op(weight.element_type, cutlass.Float32, _w._MMA_SHAPE),
            cute.make_layout((1, 1, _w._COMPUTE_WARPS)),
            permutation_mnk=(self.output_tile, self.token_tile, self.k_tile),
        )
        self.kernel(
            weight, activation, output, kv_weight, positions, freqs, pool, pages, offsets, page_size,
            sA_layout, sB_layout, tiled_mma,
        ).launch(  # fmt: skip
            grid=(self.query_tiles + 1, 1, 1),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,  # [heads * 576, q_lora]
        activation: cute.Tensor,  # [rows, q_lora + 576]: q_kv
        output: cute.Tensor,  # [rows, heads * 576]
        kv_weight: cute.Tensor,  # [512]: the kv_a RMSNorm weight
        positions: cute.Tensor,  # [rows]
        freqs: cute.Tensor,  # [positions * 64] FP32: (cos, sin) per pair
        pool: cute.Tensor,  # [pool rows, 576] uint8 (FP8 E4M3 bytes)
        pages: cute.Tensor,  # [rows]
        offsets: cute.Tensor,  # [rows]
        page_size: Int32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        rows = cute.size(activation, mode=[0])
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
        squares = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_w._COMPUTE_WARPS * self.token_tile),
            alignment=16,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator
        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile

        if output_tile_idx < self.query_tiles:
            if warp_idx < _w._A_LOADER_WARPS:
                _w._issue_weight_tile(
                    sA, weight, tidx, output_tile_idx, 0, 0, cutlass.Int64(0), self.output_tile, self.k_tile, False
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
                    _w._issue_weight_tile(
                        sA, weight, tidx, output_tile_idx, k_tile, stage, weight_policy, self.output_tile,
                        self.k_tile, True,
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
                        if row < rows:
                            dst = (
                                sB.iterator.raw_ptr()
                                + stage * self.token_tile * self.k_tile
                                + row * self.k_tile
                                + (col ^ ((row % 8) * 8))
                            )
                            src = activation.iterator.raw_ptr() + activation.layout(
                                (row, k_tile * self.k_tile + col)
                            )
                            prims.cp_async_shared_global(dst, src, 16, "cg")
                    prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                    if stage == self.stages - 1:
                        empty_phase = empty_phase ^ 1
                cute.arch.griddepcontrol_launch_dependents()
            else:
                mma_tid = tidx - self.compute_warp_base * cute.arch.WARP_SIZE
                compute_warp = warp_idx - self.compute_warp_base
                tile_in_head = output_tile_idx % TILES_PER_HEAD
                # The reduce warp's cos / sin (one rope pair per output value), loaded before the K loop.
                rope_cs = cute.make_rmem_tensor((2 * values_per_lane,), cutlass.Float32)
                rope_cs.fill(0.0)
                if compute_warp == 0:
                    cute.arch.griddepcontrol_wait()
                    if tile_in_head >= LATENT_TILES:
                        for value in cutlass.range_constexpr(values_per_lane):
                            linear = value * cute.arch.WARP_SIZE + lane_idx
                            feature = linear % self.output_tile
                            token = linear // self.output_tile
                            pos_token = token
                            if token >= rows:
                                pos_token = rows - 1
                            position = positions[pos_token]
                            pair = (tile_in_head - LATENT_TILES) * (self.output_tile // 2) + feature // 2
                            rope_cs[2 * value] = freqs[position * ROPE + pair * 2]
                            rope_cs[2 * value + 1] = freqs[position * ROPE + pair * 2 + 1]

                thr_mma = tiled_mma.get_slice(mma_tid)
                acc = cute.make_rmem_tensor(
                    thr_mma.partition_shape_C((self.output_tile, self.token_tile)), cutlass.Float32
                )
                acc.fill(0.0)
                # This lane's running sum of squares of token lane // 4 over its K columns.
                ssq = cute.make_rmem_tensor((1,), cutlass.Float32)
                ssq.fill(0.0)
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
                        _w._b_pair_offset(lane_idx, compute_warp, n_iter, pair, self.token_tile, self.k_tile)
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
                        tuple(_w._ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase]) for phase in range(k_phases))
                        for m_iter in range(m_iters)
                    )
                    b_registers = tuple(
                        tuple(
                            _w._ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair]) for pair in range(k_phases // 2)
                        )
                        for n_iter in range(n_iters)
                    )
                    for pair in cutlass.range_constexpr(k_phases // 2):
                        ssq[0] = _ssq4(ssq[0], b_registers[0][pair])
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
                # Lanes 4j .. 4j + 3 hold token j's squares of this warp's K quarter.
                total_sq = ssq[0]
                total_sq = total_sq + cute.arch.shuffle_sync_bfly(total_sq, offset=1)
                total_sq = total_sq + cute.arch.shuffle_sync_bfly(total_sq, offset=2)
                if lane_idx % 4 == 0:
                    squares[compute_warp * self.token_tile + lane_idx // 4] = total_sq
                prims.barrier_cta_sync(1, thread_count=_w._COMPUTE_WARPS * 32)

                if compute_warp == 0:
                    rope = tile_in_head >= LATENT_TILES
                    even = lane_idx % 2 == 0
                    for r_value in cutlass.range_constexpr(values_per_lane):
                        r_linear = r_value * cute.arch.WARP_SIZE + lane_idx
                        r_feature = r_linear % self.output_tile
                        r_token = r_linear // self.output_tile
                        total = cutlass.Float32(0)
                        for peer in cutlass.range_constexpr(_w._COMPUTE_WARPS):
                            total = (
                                total
                                + partials[
                                    peer * partial_stride + _w._partial_offset(r_feature, r_token, self.token_tile)
                                ]
                            )
                        sum_sq = squares[r_token]
                        for peer in cutlass.range_constexpr(1, _w._COMPUTE_WARPS):
                            sum_sq = sum_sq + squares[peer * self.token_tile + r_token]
                        inverse_rms = _rsqrt(sum_sq * Float32(1.0 / self.q_lora) + Float32(self.eps))
                        q = Float32((total * inverse_rms).to(output.element_type))
                        # The rope partner (column ^ 1, same token) is the neighbouring lane.
                        partner = cute.arch.shuffle_sync_bfly(q, offset=1)
                        cos = rope_cs[2 * r_value]
                        sin = rope_cs[2 * r_value + 1]
                        rotated = Float32(cutlass.select_(even, q * cos - partner * sin, q * cos + partner * sin))
                        out = Float32(cutlass.select_(rope, rotated, q))
                        if r_token < rows:
                            output[r_token, output_tile_idx * self.output_tile + r_feature] = out.to(
                                output.element_type
                            )
        else:
            # The block latents: warp t normalizes, rotates and stores token t (mla_norm_rope's arithmetic).
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            norm_items = KV_LORA // cute.arch.WARP_SIZE
            token = warp_idx
            if token < rows:
                values = cute.make_rmem_tensor((norm_items,), cutlass.Float32)
                scales = cute.make_rmem_tensor((norm_items,), cutlass.Float32)
                position = positions[token]
                dest = pages[token] * page_size + offsets[token]
                for item in cutlass.range_constexpr(norm_items):
                    dim = _norm_column(lane_idx, item)
                    values[item] = cutlass.Float32(activation[token, self.q_lora + dim])
                    scales[item] = cutlass.Float32(kv_weight[dim])
                real = cutlass.Float32(activation[token, self.q_lora + KV_LORA + lane_idx * 2])
                imag = cutlass.Float32(activation[token, self.q_lora + KV_LORA + lane_idx * 2 + 1])
                cos = freqs[position * ROPE + lane_idx * 2]
                sin = freqs[position * ROPE + lane_idx * 2 + 1]
                # torch normalizes the [T, 512] block latents as T rows.
                inverse_rms = _inverse_rms_torch_order(values, rows, self.eps)
                for item in cutlass.range_constexpr(norm_items):
                    dim = _norm_column(lane_idx, item)
                    pool[dest, dim] = _fp8_byte(_mul_rn(_mul_rn(values[item], inverse_rms), scales[item]))
                pool[dest, KV_LORA + lane_idx * 2] = _fp8_byte(real * cos - imag * sin)
                pool[dest, KV_LORA + lane_idx * 2 + 1] = _fp8_byte(imag * cos + real * sin)


def _view_1d(tensor, align: int):
    return from_dlpack(tensor.detach(), assumed_align=align).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,), divisibility=1
    )


@functools.cache
def compile_query_fold(
    device_index: int, heads: int, q_lora: int, eps: float, tactic, positions_dtype, index_dtype
):
    """Code for ``heads`` heads of 576 over a ``q_lora`` q_a, at most ``TOKEN_TILE`` rows (dynamic), PDL on."""
    n, k = heads * HEAD, q_lora
    _w.check_tactic(tactic, n, k)
    if smem_bytes(tactic) > _w._SMEM_CAPACITY:
        raise ValueError(f"{tactic} needs {smem_bytes(tactic)} B of shared memory")
    bf16 = torch.bfloat16
    with torch.cuda.device(device_index):
        dev = torch.device("cuda", device_index)
        args = (
            _w._view(torch.empty(n, k, device=dev, dtype=bf16)),
            _w._view(torch.empty(TOKEN_TILE, k + HEAD, device=dev, dtype=bf16), dynamic_m=True),
            _w._view(torch.empty(TOKEN_TILE, n, device=dev, dtype=bf16), dynamic_m=True),
            _w._view(torch.empty(KV_LORA, device=dev, dtype=bf16)),
            _view_1d(torch.empty(TOKEN_TILE, device=dev, dtype=positions_dtype), 8),
            _view_1d(torch.empty(16 * ROPE, device=dev, dtype=torch.float32), 16),
            _w._view(torch.empty(TOKEN_TILE, HEAD, device=dev, dtype=torch.uint8), dynamic_m=True),
            _view_1d(torch.empty(TOKEN_TILE, device=dev, dtype=index_dtype), 8),
            _view_1d(torch.empty(TOKEN_TILE, device=dev, dtype=index_dtype), 8),
        )
        stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute_ext.compile(DraftQueryFoldKernel(tactic, q_lora, heads, eps), *args, Int32(1), stream)


class PreparedQueryFold:
    """One layer's folded query weight ``[heads * 576, q_lora]`` and kv_a norm weight ``[512]``, with the code of
    ``eps`` and the warp split-K ``tactic``; compiled on construction (outside CUDA-graph capture)."""

    def __init__(self, weight: torch.Tensor, kv_norm: torch.Tensor, eps: float, tactic) -> None:
        n, k = weight.shape
        if weight.dtype != torch.bfloat16 or weight.stride() != (k, 1) or weight.data_ptr() % 32 or n % HEAD:
            raise ValueError("the folded query weight is a packed, 32 B aligned BF16 [heads * 576, q_lora]")
        if kv_norm.shape != (KV_LORA,) or kv_norm.dtype != torch.bfloat16 or not kv_norm.is_contiguous():
            raise ValueError("the kv_a norm weight is a contiguous BF16 [512]")
        self.weight, self.kv_norm, self.eps, self.tactic = weight, kv_norm, float(eps), tactic
        self.heads, self.q_lora = n // HEAD, k
        self.weight_view = _w._view(weight)
        self.kv_norm_view = _w._view(kv_norm)
        self._codes: dict = {}

    def code(self, positions_dtype, index_dtype):
        key = (positions_dtype, index_dtype)
        if key not in self._codes:
            self._codes[key] = compile_query_fold(
                self.weight.device.index, self.heads, self.q_lora, self.eps, self.tactic, positions_dtype,
                index_dtype,
            )  # fmt: skip
        return self._codes[key]

    def run(self, q_kv, query, freqs, positions, pool, pages, offsets, page_size: int):
        """``q_kv`` ``[rows, q_lora + 576]`` and ``query`` ``[rows, heads * 576]`` packed, 32 B aligned; ``freqs``
        the contiguous FP32 ``[positions * 64]`` (cos, sin) table; ``pool`` ``[pool rows, 576]`` uint8."""
        code = self.code(positions.dtype, pages.dtype)
        stream = _cuda.CUstream(torch.cuda.current_stream(q_kv.device).cuda_stream)
        code(
            self.weight_view, _w._view(q_kv, dynamic_m=True), _w._view(query, dynamic_m=True), self.kv_norm_view,
            _view_1d(positions, 8), _view_1d(freqs, 16), _w._view(pool, dynamic_m=True), _view_1d(pages, 8),
            _view_1d(offsets, 8), Int32(page_size), stream,
        )  # fmt: skip
        return query


__all__ = ["DraftQueryFoldKernel", "PreparedQueryFold", "compile_query_fold", "smem_bytes"]
