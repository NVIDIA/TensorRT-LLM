# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Softmax / P-compute helpers for FMHA decode TS resources.

Used by ``TmemSResource`` (softmax reduction, atomic-max scratch),
``SmemPResource`` (S→P conversion and quantization), ``TmemCorrResource``
(attention-sink normalization), and the softmax-stats resources.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32

from cutlass.experimental import primitives as prims

from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import (
    Constexpr,
    fadd2,
    ffma2,
    fmul2,
    _fp8_log2_quant_scale,
    _neg_max_f32,
    _q_row_token_and_local_head,
)


@cute.jit
def _pack_float4_to_fp8_e4m3(
    v0: Float32, v1: Float32, v2: Float32, v3: Float32
) -> Int32:
    """Pack four FP32 values into one FP8 E4M3x4 register."""
    return cute.arch.inline_ptx(
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n"
        "  cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n"
        "  mov.b32 {$w0}, {lo, hi};\n"
        "}",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )


@cute.jit
def _pack_float4_to_fp8_e4m3_inline(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Pack FP8x4 in one public inline-PTX block."""
    return cute.arch.inline_ptx(
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n"
        "  cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n"
        "  mov.b32 {$w0}, {lo, hi};\n"
        "}",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
        loc=loc,
        ip=ip,
    )


@cute.jit
def _compute_fp8_p_regs_and_local_sums(
    scale_softmax_log2: Float32,
    new_max_0: Float32,
    new_max_1: Float32,
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
) -> tuple[Int32, Int32, Float32, Float32]:
    """Compute masked FP8 P registers and local softmax sums."""
    # Safe path: masked tiles can produce NEG_FLT_MAX as new_max. Treat that
    # as zero for the exponent offset so invalid rows generate zero P instead
    # of NaNs while local sums remain zero.
    safe_new_max_0 = new_max_0
    safe_new_max_1 = new_max_1
    if safe_new_max_0 == _neg_max_f32():
        safe_new_max_0 = Float32(0.0)
    if safe_new_max_1 == _neg_max_f32():
        safe_new_max_1 = Float32(0.0)
    neg_scaled_max_pair = ffma2(
        (safe_new_max_0, safe_new_max_1),
        (-scale_softmax_log2, -scale_softmax_log2),
        (_fp8_log2_quant_scale(), _fp8_log2_quant_scale()),
    )

    # Scale S by log2(e) * softmax_scale and include the FP8 quantization
    # shift. The packed f32x2 operations keep paired scale groups aligned.
    scaled_pair_01 = ffma2(
        (s0, s1),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    scaled_pair_23 = ffma2(
        (s2, s3),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob0 = cute.math.exp2(scaled_pair_01[0], fastmath=True)
    prob1 = cute.math.exp2(scaled_pair_01[1], fastmath=True)
    scaled_pair_45 = ffma2(
        (s4, s5),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob2 = cute.math.exp2(scaled_pair_23[0], fastmath=True)
    prob3 = cute.math.exp2(scaled_pair_23[1], fastmath=True)
    # Accumulate the local softmax sums while packing P for the BMM2 operand.
    local_sum_pair_01 = fadd2((prob0, prob1), (prob2, prob3))
    packed_p_0 = _pack_float4_to_fp8_e4m3(prob0, prob1, prob2, prob3)
    scaled_pair_67 = ffma2(
        (s6, s7),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob4 = cute.math.exp2(scaled_pair_45[0], fastmath=True)
    prob5 = cute.math.exp2(scaled_pair_45[1], fastmath=True)
    prob6 = cute.math.exp2(scaled_pair_67[0], fastmath=True)
    prob7 = cute.math.exp2(scaled_pair_67[1], fastmath=True)
    local_sum_pair_45 = fadd2((prob4, prob5), (prob6, prob7))
    packed_p_1 = _pack_float4_to_fp8_e4m3(prob4, prob5, prob6, prob7)
    local_sum_pair = fadd2(local_sum_pair_01, local_sum_pair_45)

    return packed_p_0, packed_p_1, local_sum_pair[0], local_sum_pair[1]


@cute.jit
def _compute_fp8_p_regs_and_local_sums_dense(
    scale_softmax_log2: Float32,
    new_max_0: Float32,
    new_max_1: Float32,
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
) -> tuple[Int32, Int32, Float32, Float32]:
    """Compute dense FP8 P registers and local softmax sums."""
    # Dense path: all S entries are valid, so no NEG_FLT_MAX guard is needed.
    neg_scaled_max_pair = ffma2(
        (new_max_0, new_max_1),
        (-scale_softmax_log2, -scale_softmax_log2),
        (_fp8_log2_quant_scale(), _fp8_log2_quant_scale()),
    )

    scaled_pair_01 = ffma2(
        (s0, s1),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    scaled_pair_23 = ffma2(
        (s2, s3),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob0 = cute.math.exp2(scaled_pair_01[0], fastmath=True)
    prob1 = cute.math.exp2(scaled_pair_01[1], fastmath=True)
    scaled_pair_45 = ffma2(
        (s4, s5),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob2 = cute.math.exp2(scaled_pair_23[0], fastmath=True)
    prob3 = cute.math.exp2(scaled_pair_23[1], fastmath=True)
    local_sum_pair_01 = fadd2((prob0, prob1), (prob2, prob3))
    packed_p_0 = _pack_float4_to_fp8_e4m3(prob0, prob1, prob2, prob3)
    scaled_pair_67 = ffma2(
        (s6, s7),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob4 = cute.math.exp2(scaled_pair_45[0], fastmath=True)
    prob5 = cute.math.exp2(scaled_pair_45[1], fastmath=True)
    prob6 = cute.math.exp2(scaled_pair_67[0], fastmath=True)
    prob7 = cute.math.exp2(scaled_pair_67[1], fastmath=True)
    local_sum_pair_45 = fadd2((prob4, prob5), (prob6, prob7))
    packed_p_1 = _pack_float4_to_fp8_e4m3(prob4, prob5, prob6, prob7)
    local_sum_pair = fadd2(local_sum_pair_01, local_sum_pair_45)

    return packed_p_0, packed_p_1, local_sum_pair[0], local_sum_pair[1]


@cute.jit
def _compute_p_values_and_local_sums_dense(
    scale_softmax_log2: Float32,
    new_max_0: Float32,
    new_max_1: Float32,
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
) -> tuple[
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
]:
    """Compute dense 16-bit P values and paired local softmax sums."""
    # Dense 16-bit path: compute eight P values and the two local sums without
    # per-row validity checks.
    neg_scaled_max_pair = fmul2(
        (new_max_0, new_max_1),
        (-scale_softmax_log2, -scale_softmax_log2),
    )
    scaled_pair_01 = ffma2(
        (s0, s1),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    scaled_pair_23 = ffma2(
        (s2, s3),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob0 = cute.math.exp2(scaled_pair_01[0], fastmath=True)
    prob1 = cute.math.exp2(scaled_pair_01[1], fastmath=True)
    scaled_pair_45 = ffma2(
        (s4, s5),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob2 = cute.math.exp2(scaled_pair_23[0], fastmath=True)
    prob3 = cute.math.exp2(scaled_pair_23[1], fastmath=True)
    scaled_pair_67 = ffma2(
        (s6, s7),
        (scale_softmax_log2, scale_softmax_log2),
        neg_scaled_max_pair,
    )
    prob4 = cute.math.exp2(scaled_pair_45[0], fastmath=True)
    prob5 = cute.math.exp2(scaled_pair_45[1], fastmath=True)
    prob6 = cute.math.exp2(scaled_pair_67[0], fastmath=True)
    prob7 = cute.math.exp2(scaled_pair_67[1], fastmath=True)
    local_sum_pair_02 = fadd2((prob0, prob1), (prob2, prob3))
    local_sum_pair_46 = fadd2((prob4, prob5), (prob6, prob7))
    local_sum_pair = fadd2(local_sum_pair_02, local_sum_pair_46)
    return (
        prob0,
        prob1,
        prob2,
        prob3,
        prob4,
        prob5,
        prob6,
        prob7,
        local_sum_pair[0],
        local_sum_pair[1],
    )


@cute.jit
def _float_to_u32_for_atomic_max(val: Float32) -> Uint32:
    """Encode a float so unsigned atomic max preserves float ordering."""
    # Encode signed floats so unsigned atomic max has the same ordering.
    bits = prims.mov_b32(val, target_type=Int32)
    mask = (bits >> Int32(31)) | Int32(0x80000000)
    encoded = bits ^ mask
    return prims.mov_b32(encoded, target_type=Uint32)


@cute.jit
def _u32_to_float_for_atomic_max(val: Uint32) -> Float32:
    """Decode the unsigned atomic-max representation back to float."""
    # Decode the monotonic unsigned representation back to float.
    encoded = prims.mov_b32(val, target_type=Int32)
    mask = (~(encoded >> Int32(31))) | Int32(0x80000000)
    bits = encoded ^ mask
    return prims.mov_b32(bits, target_type=Float32)


@cute.jit
def _smem_atomic_max_u32(ptr: cute.Pointer, val: Uint32) -> None:
    """Atomically update a CTA-scope SMEM max encoded as unsigned int."""
    # Softmax row max is reduced through CTA SMEM using unsigned atomics over
    # the encoded float representation. CTA scope is sufficient because all
    # participating softmax warps are inside one CTA.
    prims.atomicrmw(
        prims.AtomicOp.MAX,
        ptr,
        val,
        syncscope=prims.MemScope.CTA,
        space=prims.SharedSpace.shared_cta,
    )


@cute.jit
def _wspro_reduce_max4(
    val0: Float32,
    val1: Float32,
    val2: Float32,
    val3: Float32,
    local_row_idx: Int32,
) -> Float32:
    """Reduce four independent maxima across four strided warp rows."""
    # The four column groups are interleaved every four lanes. The conditional
    # swaps transpose independent scale groups into row ownership before each
    # full-warp butterfly; every lane executes all three shuffle operations.
    left01 = val0
    right01 = val1
    left23 = val2
    right23 = val3
    if (local_row_idx & Int32(1)) == Int32(0):
        tmp = left01
        left01 = right01
        right01 = tmp
        tmp = left23
        left23 = right23
        right23 = tmp
    left01 = Float32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=left01,
            offset=4,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    )
    left23 = Float32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=left23,
            offset=4,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    )
    reduced01 = cute.math.max(left01, right01, ftz=True)
    reduced23 = cute.math.max(left23, right23, ftz=True)

    if local_row_idx < Int32(2):
        tmp = reduced01
        reduced01 = reduced23
        reduced23 = tmp
    reduced01 = Float32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=reduced01,
            offset=8,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    )
    return cute.math.max(reduced01, reduced23, ftz=True)


@cute.jit
def _init_softmax_scratch_u32(
    scratch: cutlass.Array, warp_grp_thread_idx: Int32, num_entries: int
) -> None:
    """Initialize encoded softmax scratch maxima across the warp group."""
    encoded = _float_to_u32_for_atomic_max(_neg_max_f32())
    # Initialize scratch with a warp-group-thread-index-stepped loop instead
    # of a simple `thread < 8` predicate. This avoids the widened store pattern
    # that the TS helper currently lowers into.
    for scratch_idx in cutlass.range(
        warp_grp_thread_idx, Int32(num_entries), Int32(128), unroll=1
    ):
        scratch[scratch_idx] = encoded


@cute.jit
def _attention_sink_for_local_head(
    cfg: FmhaDecodeConfig,
    attention_sinks_ptr: cute.Pointer | None,
    scale_softmax_log2: Float32,
    max_val: Float32,
    logical_h_k_idx: Int32,
    h_r: Int32,
    num_heads_kv: Int32,
    local_head_idx: Int32,
) -> Float32:
    """Return the attention-sink denominator contribution for one local head."""
    # Attention sinks add a synthetic key/value entry to the normalization
    # denominator. Return zero when the feature is disabled so call sites can
    # use the same reduction flow.
    if cutlass.const_expr(not cfg.use_attention_sinks):
        return Float32(0.0)
    head_idx = cute.math.min(
        logical_h_k_idx * h_r + local_head_idx,
        h_r * num_heads_kv - Int32(1),
    )
    sink_ptr = cutlass.inttoptr(
        attention_sinks_ptr.toint() + cutlass.Int64(head_idx * Int32(4)),
        mem_space=1,
        dtype=Float32,
    )
    sink_val = sink_ptr.load(count=1, alignment=4)[0]
    sink_exp = cute.math.exp2(
        sink_val * Float32(1.4426950408889634) - max_val * scale_softmax_log2,
        fastmath=True,
    )
    if cutlass.const_expr(cfg.use_fp8_qkv):
        sink_exp = sink_exp * Float32(448.0)
    return sink_exp


@cute.jit
def _attention_sink_for_scale_idx(
    cfg: FmhaDecodeConfig,
    attention_sinks_ptr: cute.Pointer | None,
    scale_softmax_log2: Float32,
    max_val: Float32,
    logical_h_k_idx: Int32,
    h_r: Int32,
    num_heads_kv: Int32,
    logical_q_group_idx: Int32,
    col_group_idx: Int32,
    scale_idx: Constexpr[int],
) -> Float32:
    """Map a softmax scale group to its attention-sink contribution."""
    tile_row_idx = (
        Int32((scale_idx // 2) * 8) + col_group_idx * Int32(2) + Int32(scale_idx % 2)
    )
    _, local_head_idx = _q_row_token_and_local_head(
        cfg, h_r, logical_q_group_idx, tile_row_idx
    )
    head_stride = h_r
    if cutlass.const_expr(cfg.heads_q_per_kv > 0):
        # Multi-token profiles flatten token/head rows differently, but
        # attention sinks remain one value per physical Q head. Resolve the
        # CTA row through the shared Q geometry and stride by the true ratio.
        head_stride = Int32(cfg.heads_q_per_kv)
    return _attention_sink_for_local_head(
        cfg,
        attention_sinks_ptr,
        scale_softmax_log2,
        max_val,
        logical_h_k_idx,
        head_stride,
        num_heads_kv,
        local_head_idx,
    )
