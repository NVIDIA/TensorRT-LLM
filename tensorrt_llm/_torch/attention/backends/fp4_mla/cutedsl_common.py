# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared host geometry, PTX packing, and online-softmax primitives for FP4 MLA CuTe DSL kernels."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import cutlass
import cutlass as ctm
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch.nvvm_wrappers import inline_ptx as cute_inline_ptx
from cutlass.experimental import primitives as prims

nvvm_add_packed_f32x2 = partial(prims.add_packed_f32x2, rnd=prims.FPRoundingMode.RN)
nvvm_mul_packed_f32x2 = partial(prims.mul_packed_f32x2, rnd=prims.FPRoundingMode.RN)
nvvm_fma_packed_f32x2 = partial(prims.fma_packed_f32x2, rnd=prims.FPRoundingMode.RN)

PREPARED_BUFFER_ALIGNMENT_BYTES = 32
CUDA_GRID_Z_MAX = 65535
INT32_MAX = (1 << 31) - 1
LOG2_6 = math.log2(6.0)
LOG2_E = 1.4426950408889634
FP4_MLA_E4M3_MAX_FINITE = 448.0
FP4_MLA_P_GLOBAL_SCALE = FP4_MLA_E4M3_MAX_FINITE * 6.0


@dataclass(frozen=True)
class KvCache3DLayout:
    num_pages: int
    packed_dim: int
    stride_page: int
    stride_token: int
    stride_packed_dim: int


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@ctm.dsl_user_op
def _as_shared_cta(address: Any, *, loc: Any = None, ip: Any = None) -> Any:
    address_ir = address.ir_value() if hasattr(address, "ir_value") else address
    address_space = llvm.PointerType(address_ir.type).address_space
    if address_space == ctm.AddressSpace.dsmem:
        return prims.cvta_to(address, prims.CvtaSpace.SHARED, loc=loc, ip=ip)
    return address


@ctm.dsl_user_op
def _mapa_shared_cluster(address: Any, rank: Any, *, loc: Any = None, ip: Any = None) -> Any:
    return prims.mapa(_as_shared_cta(address, loc=loc, ip=ip), rank, loc=loc, ip=ip)


@ctm.dsl_user_op
def _mbarrier_arrive_release_cta_shared_cluster(
    mbar: Any, count: int = 1, *, loc: Any = None, ip: Any = None
) -> None:
    prims.mbarrier_arrive(
        mbar,
        count=count,
        scope=prims.MemScope.CTA,
        loc=loc,
        ip=ip,
    )


def _scale_words_for_k(k_dim: int, sf_vec_size: int) -> int:
    return (k_dim + sf_vec_size * 4 - 1) // (sf_vec_size * 4)


def _sfa_tmem_cols_for_scale_chunks(scale_chunks: int, use_128dp_unique: bool) -> int:
    if use_128dp_unique:
        return (scale_chunks + 3) // 4 * 4
    return scale_chunks * 4


def _compact_sfb_mma_cols_per_scale_chunk(m_dim: int, n_dim: int) -> int:
    if m_dim == 128:
        if n_dim > 256:
            raise ValueError(f"Unsupported M=128 compact SFB N dimension {n_dim}")
        return 2 * ((n_dim + 127) // 128)
    return 2 * ((n_dim + 63) // 64)


def _compact_sfb_copy_cols_per_scale_chunk(m_dim: int, n_dim: int) -> int:
    if m_dim == 128:
        if n_dim > 256:
            raise ValueError(f"Unsupported M=128 compact SFB N dimension {n_dim}")
        return 4 * ((n_dim + 127) // 128)
    return _compact_sfb_mma_cols_per_scale_chunk(m_dim, n_dim)


def _qk_compact_sfb_cols_per_scale_chunk(m_dim: int, n_dim: int, warpx2_mode: int) -> int:
    if warpx2_mode != 0 and m_dim == 128:
        return _compact_sfb_mma_cols_per_scale_chunk(m_dim, n_dim)
    return _compact_sfb_copy_cols_per_scale_chunk(m_dim, n_dim)


def _qk_sfb_cols_per_kblock(m_dim: int, n_dim: int, mma_scale_chunks_per_kblock: int) -> int:
    return mma_scale_chunks_per_kblock * _compact_sfb_copy_cols_per_scale_chunk(m_dim, n_dim)


def _qk_sfb_smem_nchunks_per_cta(m_dim: int, n_dim: int) -> int:
    if m_dim == 128:
        if n_dim > 256:
            raise ValueError(f"Unsupported M=128 compact SFB N dimension {n_dim}")
    return (n_dim + 127) // 128


def _derive_qk_scale_burst_kblocks(
    *,
    qk_mma_kblocks: int,
    qk_scale_burst_target_kblocks: int,
    scale_a_col_offset: int,
    num_tmem_alloc_cols: int,
    qk_mma_m_dim: int,
    mma_scale_chunks_per_kblock: int,
    use_sfa_128dp_unique: bool,
    qk_score_slot_n: int,
    qk_compact_sfb_cols_per_scale_chunk: int = 0,
) -> int:
    target = max(1, qk_scale_burst_target_kblocks)
    if qk_compact_sfb_cols_per_scale_chunk > 0:
        sfb_cols_per_kblock = mma_scale_chunks_per_kblock * qk_compact_sfb_cols_per_scale_chunk
    else:
        sfb_cols_per_kblock = _qk_sfb_cols_per_kblock(
            qk_mma_m_dim, qk_score_slot_n, mma_scale_chunks_per_kblock
        )
    best = 0
    for burst_kblocks in range(1, target + 1):
        if qk_mma_kblocks % burst_kblocks != 0:
            continue
        sfa_cols = _sfa_tmem_cols_for_scale_chunks(
            burst_kblocks * mma_scale_chunks_per_kblock, use_sfa_128dp_unique
        )
        sfb_cols = burst_kblocks * sfb_cols_per_kblock
        scale_b_col_offset = (scale_a_col_offset + sfa_cols + 15) // 16 * 16
        qk_scale_feed_col_end = max(scale_a_col_offset + sfa_cols, scale_b_col_offset + sfb_cols)
        if qk_scale_feed_col_end <= num_tmem_alloc_cols:
            best = burst_kblocks
    if best == 0:
        raise ValueError(
            "QK scale feed cannot fit even one K64 block in the TMEM tail; "
            f"scale_a_col_offset={scale_a_col_offset}, "
            f"num_tmem_alloc_cols={num_tmem_alloc_cols}"
        )
    return best


def _mma_sf_ids_per_scale_atom(k_dim: int, sf_vec_size: int) -> int:
    return max(1, sf_vec_size * 4 // k_dim)


def _mma_scale_vec_size_enum(sf_vec_size: int) -> Any:
    if sf_vec_size == 16:
        return prims.Tcgen05MMAScaleVecSize.BLOCK16
    raise ValueError(f"Unsupported FP4 scale vector size {sf_vec_size}")


def _mma_scale_format(sf_vec_size: int) -> int:
    if sf_vec_size == 16:
        return 0
    raise ValueError(f"Unsupported FP4 scale vector size {sf_vec_size}")


def _mma_block_scale_kind(sf_vec_size: int) -> Any:
    if sf_vec_size == 16:
        return prims.MMABlockScaleKind.MXF4NVF4
    raise ValueError(f"Unsupported FP4 scale vector size {sf_vec_size}")


def _validate_output_dtype(output_dtype: torch.dtype) -> None:
    if output_dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(f"output dtype must be torch.float16 or torch.bfloat16, got {output_dtype}")


def _cutlass_output_dtype(output_dtype: torch.dtype) -> Any:
    _validate_output_dtype(output_dtype)
    if output_dtype == torch.bfloat16:
        return cutlass.BFloat16
    return cutlass.Float16


def _validate_tensor_pointer_alignment(
    name: str,
    tensor: torch.Tensor,
    alignment_bytes: int = PREPARED_BUFFER_ALIGNMENT_BYTES,
) -> None:
    pointer = tensor.data_ptr()
    if pointer % alignment_bytes != 0:
        raise ValueError(
            f"{name} data pointer must be {alignment_bytes}B aligned, "
            f"got address remainder {pointer % alignment_bytes}"
        )


@cute.jit
def _pack_e2m1x8(
    p0: ctm.Float32,
    p1: ctm.Float32,
    p2: ctm.Float32,
    p3: ctm.Float32,
    p4: ctm.Float32,
    p5: ctm.Float32,
    p6: ctm.Float32,
    p7: ctm.Float32,
) -> ctm.Int32:
    return cute_inline_ptx(
        "{\n"
        "\t.reg .b8 b0, b1, b2, b3;\n"
        "\tcvt.rn.satfinite.e2m1x2.f32 b0, {$r1}, {$r0};\n"
        "\tcvt.rn.satfinite.e2m1x2.f32 b1, {$r3}, {$r2};\n"
        "\tcvt.rn.satfinite.e2m1x2.f32 b2, {$r5}, {$r4};\n"
        "\tcvt.rn.satfinite.e2m1x2.f32 b3, {$r7}, {$r6};\n"
        "\tmov.b32 {$w0}, {b0, b1, b2, b3};\n"
        "\t}\n",
        write_only_types=[ctm.Int32],
        read_only_args=[p0, p1, p2, p3, p4, p5, p6, p7],
    )


@cute.jit
def _mask_invalid_score(
    score: ctm.Float32, global_k_idx: ctm.Int32, valid_k: ctm.Int32
) -> ctm.Float32:
    masked = score
    if global_k_idx >= valid_k:
        masked = ctm.Float32(-ctm.Float32.inf)
    return masked


@cute.jit
def _pack_e4m3x4_natural(
    sf0: ctm.Float32, sf1: ctm.Float32, sf2: ctm.Float32, sf3: ctm.Float32
) -> ctm.Int32:
    return cute_inline_ptx(
        "{\n"
        "\t.reg .b16 lo, hi;\n"
        "\tcvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n"
        "\tcvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n"
        "\tmov.b32 {$w0}, {lo, hi};\n"
        "\t}\n",
        write_only_types=[ctm.Int32],
        read_only_args=[sf0, sf1, sf2, sf3],
    )


@cute.jit
def _pack_e4m3x4_natural_from_f16x2(pair01: ctm.Int32, pair23: ctm.Int32) -> ctm.Int32:
    return cute_inline_ptx(
        "{\n"
        "\t.reg .b16 lo, hi;\n"
        "\tcvt.rn.satfinite.e4m3x2.f16x2 lo, {$r0};\n"
        "\tcvt.rn.satfinite.e4m3x2.f16x2 hi, {$r1};\n"
        "\tmov.b32 {$w0}, {lo, hi};\n"
        "\t}\n",
        write_only_types=[ctm.Int32],
        read_only_args=[pair01, pair23],
    )


@cute.jit
def _scale_pack_e4m3x4_natural_from_f16x2(
    pair01: ctm.Int32, pair23: ctm.Int32, scale: ctm.Float32
) -> ctm.Int32:
    """Scale two FP16x2 pairs and pack the PV-only E4M3 scale word."""
    return cute_inline_ptx(
        "{\n"
        "\t.reg .f16 scale_h;\n"
        "\t.reg .b32 scale_h2, scaled01, scaled23;\n"
        "\t.reg .b16 lo, hi;\n"
        "\tcvt.rn.f16.f32 scale_h, {$r2};\n"
        "\tmov.b32 scale_h2, {scale_h, scale_h};\n"
        "\tmul.f16x2 scaled01, {$r0}, scale_h2;\n"
        "\tmul.f16x2 scaled23, {$r1}, scale_h2;\n"
        "\tcvt.rn.satfinite.e4m3x2.f16x2 lo, scaled01;\n"
        "\tcvt.rn.satfinite.e4m3x2.f16x2 hi, scaled23;\n"
        "\tmov.b32 {$w0}, {lo, hi};\n"
        "\t}\n",
        write_only_types=[ctm.Int32],
        read_only_args=[pair01, pair23, scale],
    )


@cute.jit
def _softmax_exp2_pair(
    score0: ctm.Float32,
    score1: ctm.Float32,
    softmax_scale_log2: ctm.Float32,
    p_bias: ctm.Float32,
) -> tuple:
    shifted = nvvm_fma_packed_f32x2(
        (score0, score1), (softmax_scale_log2, softmax_scale_log2), (p_bias, p_bias)
    )
    return (cute.exp2(shifted[0], fastmath=True), cute.exp2(shifted[1], fastmath=True))


@cute.jit
def _pack_f32_pair_to_f16x2(value0: ctm.Float32, value1: ctm.Float32) -> ctm.Int32:
    return (
        ctm.Vector.from_elements((value0, value1), dtype=ctm.Float32)
        .to(ctm.Float16)
        .bitcast(ctm.Int32)[0]
    )


@cute.jit
def _softmax_exp2_pair_packed_f16x2(
    score0: ctm.Float32,
    score1: ctm.Float32,
    softmax_scale_log2: ctm.Float32,
    p_bias: ctm.Float32,
) -> ctm.Int32:
    shifted = nvvm_fma_packed_f32x2(
        (score0, score1), (softmax_scale_log2, softmax_scale_log2), (p_bias, p_bias)
    )
    shifted_h2 = _pack_f32_pair_to_f16x2(shifted[0], shifted[1])
    return cute_inline_ptx(
        "ex2.approx.f16x2 {$w0}, {$r0};",
        write_only_types=[ctm.Int32],
        read_only_args=[shifted_h2],
    )


@cute.jit
def _add_f16x2(a: ctm.Int32, b: ctm.Int32) -> ctm.Int32:
    return cute_inline_ptx(
        "add.rn.f16x2 {$w0}, {$r0}, {$r1};",
        write_only_types=[ctm.Int32],
        read_only_args=[a, b],
    )


@cute.jit
def _unpack_f16x2_to_f32_pair(packed: ctm.Int32) -> tuple:
    """Widen one packed probability pair for the denominator only."""
    values_f16 = ctm.Vector.from_elements((packed,), dtype=ctm.Int32).bitcast(ctm.Float16)
    values_f32 = values_f16.to(ctm.Float32)
    return (values_f32[0], values_f32[1])


@cute.jit
def _sum_p4_n256_psf_row_parts(
    row_sum_part0: ctm.Float32,
    row_sum_part1: ctm.Float32,
    row_sum_part2: ctm.Float32,
    row_sum_part3: ctm.Float32,
) -> ctm.Float32:
    row_sum = row_sum_part0 + row_sum_part1
    row_sum = row_sum + row_sum_part2
    return row_sum + row_sum_part3


@cute.jit
def _p4_n256_prequant_rowsum_from_f16x2_psf(
    sf_h2_01: ctm.Int32,
    sf_h2_23: ctm.Int32,
    group_p_sums: Any,
) -> ctm.Float32:
    """Weight FP32 P sums before the FP16 PSF values narrow to E4M3."""
    sf_exact_0, sf_exact_1 = _unpack_f16x2_to_f32_pair(sf_h2_01)
    sf_exact_2, sf_exact_3 = _unpack_f16x2_to_f32_pair(sf_h2_23)
    row_sum_parts01 = nvvm_mul_packed_f32x2(
        (sf_exact_0, sf_exact_1), (group_p_sums[0], group_p_sums[1])
    )
    row_sum_parts23 = nvvm_mul_packed_f32x2(
        (sf_exact_2, sf_exact_3), (group_p_sums[2], group_p_sums[3])
    )
    return _sum_p4_n256_psf_row_parts(
        row_sum_parts01[0],
        row_sum_parts01[1],
        row_sum_parts23[0],
        row_sum_parts23[1],
    )


@cute.jit
def _pack_e2m1x8_from_f16x2(
    pair0: ctm.Int32, pair1: ctm.Int32, pair2: ctm.Int32, pair3: ctm.Int32
) -> ctm.Int32:
    return cute_inline_ptx(
        "{\n"
        "\t.reg .b8 b0, b1, b2, b3;\n"
        "\tcvt.rn.satfinite.e2m1x2.f16x2 b0, {$r0};\n"
        "\tcvt.rn.satfinite.e2m1x2.f16x2 b1, {$r1};\n"
        "\tcvt.rn.satfinite.e2m1x2.f16x2 b2, {$r2};\n"
        "\tcvt.rn.satfinite.e2m1x2.f16x2 b3, {$r3};\n"
        "\tmov.b32 {$w0}, {b0, b1, b2, b3};\n"
        "\t}\n",
        write_only_types=[ctm.Int32],
        read_only_args=[pair0, pair1, pair2, pair3],
    )


@cute.jit
def _pack_p4_16_scores_with_sum(
    group_scores: Any,
    group_base: ctm.Int32,
    valid_k: ctm.Int32,
    group_valid: ctm.Boolean,
    mask_valid_k: ctm.Constexpr,
    use_mixed_imlp: ctm.Constexpr,
    softmax_scale_log2: ctm.Float32,
    p_bias: ctm.Float32,
    elem_base: ctm.Constexpr,
) -> tuple:
    del use_mixed_imlp
    words = ctm.Array(ctm.Int32, 2, space=ctm.AddressSpace.rmem)
    p_sum_pair = (ctm.Float32(0.0), ctm.Float32(0.0))
    for word_part in ctm.range_constexpr(2):
        packed_p0 = ctm.Float32(0.0)
        packed_p1 = ctm.Float32(0.0)
        packed_p2 = ctm.Float32(0.0)
        packed_p3 = ctm.Float32(0.0)
        packed_p4 = ctm.Float32(0.0)
        packed_p5 = ctm.Float32(0.0)
        packed_p6 = ctm.Float32(0.0)
        packed_p7 = ctm.Float32(0.0)
        for pair_idx in ctm.range_constexpr(4):
            elem0: ctm.Constexpr = elem_base + word_part * 8 + pair_idx * 2
            elem1: ctm.Constexpr = elem0 + 1
            p0 = ctm.Float32(0.0)
            p1 = ctm.Float32(0.0)
            should_compute = ctm.Boolean(True)
            if cutlass.const_expr(mask_valid_k):
                should_compute = group_valid
            if should_compute:
                score0 = group_scores[elem0]
                score1 = group_scores[elem1]
                if cutlass.const_expr(mask_valid_k):
                    global_k_idx0 = group_base + ctm.Int32(elem0)
                    global_k_idx1 = group_base + ctm.Int32(elem1)
                    score0 = _mask_invalid_score(score0, global_k_idx0, valid_k)
                    score1 = _mask_invalid_score(score1, global_k_idx1, valid_k)
                p0, p1 = _softmax_exp2_pair(score0, score1, softmax_scale_log2, p_bias)
            if cutlass.const_expr(word_part == 0 and pair_idx == 0):
                p_sum_pair = (p0, p1)
            else:
                p_sum_pair = nvvm_add_packed_f32x2(p_sum_pair, (p0, p1))
            if cutlass.const_expr(pair_idx == 0):
                packed_p0 = p0
                packed_p1 = p1
            elif cutlass.const_expr(pair_idx == 1):
                packed_p2 = p0
                packed_p3 = p1
            elif cutlass.const_expr(pair_idx == 2):
                packed_p4 = p0
                packed_p5 = p1
            else:
                packed_p6 = p0
                packed_p7 = p1
        words[word_part] = _pack_e2m1x8(
            packed_p0,
            packed_p1,
            packed_p2,
            packed_p3,
            packed_p4,
            packed_p5,
            packed_p6,
            packed_p7,
        )
    p_sum = p_sum_pair[0] + p_sum_pair[1]
    return (words[0], words[1], p_sum)


@cute.jit
def _float_to_ordered_u32_for_atomic_max(value: ctm.Float32) -> ctm.Uint32:
    bits = prims.mov_b32(value, target_type=ctm.Int32)
    sign_mask = bits >> ctm.Int32(31) | ctm.Int32(2147483648)
    encoded = bits ^ sign_mask
    return prims.mov_b32(encoded, target_type=ctm.Uint32)


@cute.jit
def _ordered_u32_to_float_after_atomic_max(value: ctm.Uint32) -> ctm.Float32:
    encoded = prims.mov_b32(value, target_type=ctm.Int32)
    sign_mask = ~(encoded >> ctm.Int32(31)) | ctm.Int32(2147483648)
    bits = encoded ^ sign_mask
    return prims.mov_b32(bits, target_type=ctm.Float32)


@cute.jit
def _smem_atomic_max_ordered_u32(pointer: Any, value: ctm.Uint32) -> None:
    prims.atomicrmw(
        prims.AtomicOp.MAX,
        pointer,
        value,
        syncscope=prims.MemScope.CTA,
        space=prims.SharedSpace.shared_cta,
    )


@cute.jit
def _tcgen05_ld_red_32x32b_x16_max_f32(tmem: Any) -> tuple:
    tmem_addr = tmem.toint(ctm.Int32)
    return cute_inline_ptx(
        "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}, {$w4}, {$w5}, {$w6}, {$w7}, "
        "{$w8}, {$w9}, {$w10}, {$w11}, {$w12}, {$w13}, {$w14}, {$w15}}, "
        "{$w16}, [{$r0}];",
        write_only_types=[ctm.Int32] * 17,
        read_only_args=[tmem_addr],
    )


@cute.jit
def _publish_p4_n256_atomic_rowmax(
    sAtomicRunningRowMax: Any,
    tile_row_max: ctm.Float32,
    warp_idx: ctm.Int32,
    tidx: ctm.Int32,
    producer_warp_base: ctm.Int32,
) -> None:
    local_warp_idx = warp_idx - producer_warp_base
    lane_idx = tidx & ctm.Int32(31)
    row_band = local_warp_idx & ctm.Int32(1)
    local_row = row_band * ctm.Int32(32) + lane_idx
    _smem_atomic_max_ordered_u32(
        sAtomicRunningRowMax.data_ptr() + local_row,
        _float_to_ordered_u32_for_atomic_max(tile_row_max),
    )


@cute.jit
def _fence_async_shared_cta() -> None:
    prims.fence_proxy(kind=prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)


@cute.jit
def _lookup_physical_page(
    mPageTable_pl: cute.Tensor,
    logical_page: ctm.Int32,
    batch: ctm.Int32,
    page_begin: ctm.Int32,
    page_count: ctm.Int32,
) -> ctm.Int32:
    has_pages = ctm.Int32(page_count > ctm.Int32(0))
    last_logical_page = ctm.max(page_count - ctm.Int32(1), ctm.Int32(0))
    safe_logical_page = ctm.min(logical_page, last_logical_page)
    page_table_idx = (page_begin + safe_logical_page) * has_pages
    return ctm.Int32(mPageTable_pl[page_table_idx])


@cute.jit
def _compute_smem_p4_lazy_rebase_scale(
    lane_rebase: ctm.Boolean, anchor_delta_log2: ctm.Float32
) -> ctm.Float32:
    rebase_scale = ctm.Float32(1.0)
    if lane_rebase:
        rebase_scale = cute.exp2(-anchor_delta_log2, fastmath=True)
    return rebase_scale


@cute.jit
def _rescale_smem_p4_resident_o_in_tmem(o_acc_tmem, old_scale: ctm.Float32) -> None:
    scaled_accum = ctm.Array(ctm.Float32, 128, space=ctm.AddressSpace.rmem)
    c_rmem = prims.tcgen05_ld(prims.Tcgen05LdStShape.SHAPE_32X32B, o_acc_tmem, num=128)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    old_scale_f32x2 = (old_scale, old_scale)
    for pair_idx in ctm.range_constexpr(64):
        elem_idx: ctm.Constexpr = pair_idx * 2
        scaled_pair = nvvm_mul_packed_f32x2(
            (c_rmem[elem_idx], c_rmem[elem_idx + 1]), old_scale_f32x2
        )
        scaled_accum[elem_idx] = scaled_pair[0]
        scaled_accum[elem_idx + 1] = scaled_pair[1]
    prims.tcgen05_st(
        prims.Tcgen05LdStShape.SHAPE_32X32B,
        o_acc_tmem,
        scaled_accum.load(0, 128, alignment=32),
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
