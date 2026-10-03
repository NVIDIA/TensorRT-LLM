# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CuTe DSL FP4 (MXFP4) paged MQA logits kernel for Blackwell (SM100).

Implement from DeepGEMM's `sm100_fp4_paged_mqa_logits.cuh` and reuse the FP8 DSL kernel's
warp partition / barrier choreography with the following deltas (see plan
`study-deepseek-v4/cute_dsl_fp4_indexer/plan_detailed.md` for full design):

  * A/B element type = `Float4E2M1FN` (0.5B / packed `uint8` `e2m1x2`)
  * Per-(token, K-group) UE8M0 SF, sf_vec_size=32, head_dim/sf_vec=4 SFs/token
  * MMA = `tcgen05` MXF4 SS (block-scaled), UMMA_K=64
  * UTCCP copies SF SMEM -> TMEM in MMA-required chunk layout. SMEM is flat
    (matches paged KV writer); a lane-cooperative warp transpose runs in SMEM
    before UTCCP to reshuffle flat -> chunk byte layout.
  * SF Q (4 UE8M0 packed per int32 token) and SF KV (4 UE8M0 packed per int32
    token, embedded in the fused KV buffer tail) flow as flat int32.
  * Scale apply: Block-scaled MMA bakes the SF into acc; epilogue drops the
    `* scale_val` multiply (compared to FP8). Math warp no longer waits on
    KV+SF pipe — UMMA owns it.

Architecture:
  - 384 threads: 256 math (2 WGs) + 128 specialized (2 TMA + 2 UMMA)
  - 1 TMA per KV block [128, 128] (FP4 packed bytes = 64 / row)
  - 2 warp groups process 2 KV blocks per iteration (kNumMathWarpGroups=2)
  - Q reloaded via TMA pipeline when q_idx (batch) changes
  - Persistent kernel: CTAs iterate through assigned (q_idx, kv_idx) pairs
  - Weights cached in registers: preloaded once per q_idx change

Fused KV layout (same shape as FP8, half the data bytes):
  [num_phys_blocks, phys_block_kv, 1, head_dim/2 + 4] uint8
  Per block: [KV data (phys_block_kv * head_dim/2 bytes)] [SF (phys_block_kv * 4 bytes)]

Epilogue dtype flows:
  acc=fp32 (FP4 MXF4 SS only emits fp32 acc).
  epi ∈ {fp32, bf16, fp16}, output ∈ {fp32, bf16, fp16}.

  fp32 path: tmem_load fp32 → ReLU → scalar FMA → cvt output_dtype → store
  fp16 path: tmem_load fp32 → cvt fp16 → packed ReLU/FMA (fma.rn.f16x2) → store
  bf16 path: tmem_load fp32 → cvt bf16 → packed ReLU/FMA (fma.rn.bf16x2) → store

  Op surface accepts fp32 weights; the host wrapper casts to `epi_dtype`
  before passing to the kernel, so the tma_atom_w dtype matches `self.epi_bytes`.
"""

import math
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import BFloat16, Float4E2M1FN, Float8E8M0FNU, Float16, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait

# CuTe DSL CUDA 13 validates rounding modes as string literals. The string
# form is also accepted by older wrappers, so keep it version-independent.
_RND_RN = "rn"

# Reduction identities for emit_block_meta; must match the GVR kernel's
# FLT_MAX/NEG_FLT_MAX sentinels (gvr_topk_decode.py).
_META_FLT_MAX = 3.4028235e38
_META_NEG_FLT_MAX = -3.4028235e38


# Global-memory reductions for the per-row hit aggregate (emit_hit_stats).
# fp32 min/max use the order-preserving int encoding
# enc(f) = bits(f) >= 0 ? bits(f) : bits(f) ^ 0x7FFFFFFF (an involution)
# with red.global.{min,max}.s32; sum uses red.global.add.f32.
@dsl_user_op
def _fabs_f32(x, *, loc=None, ip=None):
    """abs.f32 as inline PTX so ptxas folds it into the FADD2 |R| operand modifier
    (relu(x) = (x + |x|) * 0.5, the 0.5 pre-folded into the cached weights)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_f32(addr_i64, fval, *, loc=None, ip=None):
    """st.global.f32 [addr], val (block_max record store by precomputed byte address)."""
    llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), fval.ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_global_fmin_ordered(addr_i64, fval, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), fval.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 k;\n\t"
        ".reg .pred p;\n\t"
        "mov.b32 k, $1;\n\t"
        "setp.lt.s32 p, k, 0;\n\t"
        "@p xor.b32 k, k, 0x7FFFFFFF;\n\t"
        "red.global.min.s32 [$0], k;\n\t"
        "}",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_global_fmax_ordered(addr_i64, fval, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), fval.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 k;\n\t"
        ".reg .pred p;\n\t"
        "mov.b32 k, $1;\n\t"
        "setp.lt.s32 p, k, 0;\n\t"
        "@p xor.b32 k, k, 0x7FFFFFFF;\n\t"
        "red.global.max.s32 [$0], k;\n\t"
        "}",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_global_add_s32(addr_i64, ival, *, loc=None, ip=None):
    """atom.global.add.s32 returning the OLD value (warp batch-claim)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr_i64.ir_value(loc=loc, ip=ip), ival.ir_value(loc=loc, ip=ip)],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_global_add_s32(addr_i64, ival, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), ival.ir_value(loc=loc, ip=ip)],
        "red.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _red_global_add_f32(addr_i64, fval, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), fval.ir_value(loc=loc, ip=ip)],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_relaxed_gpu_s32(addr_i64, *, loc=None, ip=None):
    """L1-bypassing load of a word other CTAs mutate (claim counters)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr_i64.ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _trap(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_s32(addr_i64, ival, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), ival.ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_f16x2(
    a: Float16,
    b: Float16,
    *,
    loc=None,
    ip=None,
) -> Int32:
    f16_ty = Float16.mlir_type
    i32_ty = Int32.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = vector.from_elements(
        vec2_f16,
        (Float16(a).ir_value(loc=loc, ip=ip), Float16(b).ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.bitcast(i32_ty, v, loc=loc, ip=ip))


@dsl_user_op
def unpack_f16x2(
    packed: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float16, Float16]:
    f16_ty = Float16.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = llvm.bitcast(vec2_f16, Int32(packed).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    r0 = Float16(vector.extract(v, dynamic_position=[], static_position=[0], loc=loc, ip=ip))
    r1 = Float16(vector.extract(v, dynamic_position=[], static_position=[1], loc=loc, ip=ip))
    return r0, r1


@dsl_user_op
def fma_f16x2(
    a: Int32,
    b: Int32,
    c: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# bf16x2 packed helpers — same shape as f16x2 helpers, different PTX op.
# Used by epi_dtype = BFloat16 path (FP4 only; FP8 doesn't support bf16 epi).


@dsl_user_op
def pack_bf16x2(
    a: BFloat16,
    b: BFloat16,
    *,
    loc=None,
    ip=None,
) -> Int32:
    bf16_ty = BFloat16.mlir_type
    i32_ty = Int32.mlir_type
    vec2_bf16 = ir.VectorType.get([2], bf16_ty, loc=loc)
    v = vector.from_elements(
        vec2_bf16,
        (BFloat16(a).ir_value(loc=loc, ip=ip), BFloat16(b).ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.bitcast(i32_ty, v, loc=loc, ip=ip))


@dsl_user_op
def fma_bf16x2(
    a: Int32,
    b: Int32,
    c: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_bf16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_bf16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def unpack_bf16x2(
    packed: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[BFloat16, BFloat16]:
    bf16_ty = BFloat16.mlir_type
    vec2_bf16 = ir.VectorType.get([2], bf16_ty, loc=loc)
    v = llvm.bitcast(vec2_bf16, Int32(packed).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    r0 = BFloat16(vector.extract(v, dynamic_position=[], static_position=[0], loc=loc, ip=ip))
    r1 = BFloat16(vector.extract(v, dynamic_position=[], static_position=[1], loc=loc, ip=ip))
    return r0, r1


# SMEM b32 load/store wrappers — `cute.arch.ld_shared / st_shared` are not
# exposed in the DSL surface, so we wrap raw PTX. Used by
# `utccp_required_smem_warp_transpose` (Step 4 of plan) to reshuffle flat
# UE8M0 SF SMEM bytes into the chunk byte layout that UTCCP / MMA require.


@dsl_user_op
def ld_shared_b32(smem_ptr, *, loc=None, ip=None) -> Int32:
    # `_Pointer.toint()` returns Int32 (SMEM addr fits in 32 bits).
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [smem_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_b32(smem_ptr, val: Int32, *, loc=None, ip=None) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.jit
def utccp_required_smem_warp_transpose(smem_ptr) -> None:
    """1 warp cooperatively reshuffles 128 int32 (= one 128-token x 4-K-group
    SF atom = 512 bytes) from flat M-row-major layout to the chunk byte layout
    required by UTCCP / MMA for block-scaled FP4.

    Learn from DeepGEMM's implementation `sm100_fp4_paged_mqa_logits.cuh:267-277`.
    The XOR pattern `i ^ (lane_idx >> 3)` avoids SMEM bank conflicts and must be
    preserved (M4 reminder).

    Args:
        smem_ptr: cute.Pointer to int32, must be 128-int (= 512-byte) aligned.
    """
    lane_idx = cute.arch.lane_idx()
    values = cute.make_rmem_tensor(4, cutlass.Int32)
    for i in cutlass.range_constexpr(4):
        offset = (i ^ (lane_idx >> 3)) * 32 + lane_idx
        values[i] = ld_shared_b32(smem_ptr + offset)
    cute.arch.sync_warp()
    for i in cutlass.range_constexpr(4):
        offset = lane_idx * 4 + (i ^ (lane_idx >> 3))
        st_shared_b32(smem_ptr + offset, values[i])


class FP4MQALogitsKernel:
    """FP4 (MXFP4) paged MQA logits kernel for Blackwell (SM100).

    Each CTA processes a range of (q_idx, kv_split) pairs.
    A split = 2 consecutive KV blocks within a sequence (one per warp group).
    Q is shared between warp groups and reloaded when q_idx changes.

    Differs from FP8 sibling: A/B are Float4E2M1FN, MMA is block-scaled MXF4 SS
    (UMMA_K=64), per-(token, K-group) UE8M0 SF feeds the MMA via UTCCP/TMEM.
    KV+SF pipeline is owned by the UMMA warp; Math warp epilogue is identical
    in shape but drops the `* scale_val` multiply (the SF is baked into acc by
    the block-scaled MMA itself).
    """

    def __init__(
        self,
        block_kv: int = 128,
        phys_block_kv: int = 128,
        num_heads: int = 64,
        head_dim: int = 128,
        next_n: int = 1,
        num_sms: int = 148,
        num_epi_subtiles: int = 1,
        epi_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        remove_online_sf_transpose: bool = False,
        use_batched_store: bool = True,
        emit_block_meta: bool = False,
        emit_hit_stats: bool = True,
        emit_seed_counts: bool = False,
        seed_packed: bool = False,
        emit_cand: bool = False,
        cand_cap: int = 5120,
        emit_cand_bucketed: bool = False,
        accept_cap: int = 8192,
        dynamic_sched: bool = False,
        ring_depth: int = 128,
        b_cap: int = 1024,
        pdl: bool | None = None,
        pdl_trigger: int = 2,
        num_kv_stages: int = 6,
    ):
        # Static FP4 invariants — see plan Sanity checklist.
        assert num_heads == 64, "FP4 kernel hardcodes num_heads=64 for TMEM/SMEM budget"
        assert head_dim == 128, "FP4 kernel hardcodes head_dim=128"
        assert next_n in (1, 2, 3), (
            f"FP4 supports next_n in {{1,2,3}}; got {next_n}. next_n=4 is out-of-scope (TMEM cap)."
        )
        assert epi_dtype in (
            cutlass.Float32,
            cutlass.BFloat16,
            cutlass.Float16,
        ), f"FP4 epi_dtype must be fp32/bf16/fp16; got {epi_dtype}"
        assert output_dtype in (
            cutlass.Float32,
            cutlass.BFloat16,
            cutlass.Float16,
        ), f"FP4 output_dtype must be fp32/bf16/fp16; got {output_dtype}"
        assert block_kv == 128, "FP4 compute tile (block_kv) hardcoded to 128"
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.num_blocks_per_mma = block_kv // phys_block_kv
        assert block_kv % phys_block_kv == 0, (
            f"block_kv={block_kv} must be divisible by phys_block_kv={phys_block_kv}"
        )
        assert self.num_blocks_per_mma <= 4, (
            f"num_blocks_per_mma={self.num_blocks_per_mma} exceeds max 4"
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n = next_n
        self.N = next_n * num_heads
        self.num_sms = num_sms
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        # When True, skip the in-kernel SMEM warp_transpose for KV SF; assume
        # the host has pre-arranged GMEM SF into UTCCP chunk layout. Only valid
        # for phys_block_kv=128 (1 phys block = 1 UTCCP atom). Q SF transpose
        # is NOT affected by this flag (deferred to a separate phase).
        if remove_online_sf_transpose and phys_block_kv != 128:
            remove_online_sf_transpose = False
        self.remove_online_sf_transpose = remove_online_sf_transpose
        # When True, defer per-t STG to register array and emit all STGs in
        # one contiguous LSU phase after the for-t loop (epilogue micro-opt).
        self.use_batched_store = use_batched_store
        # When True, the epilogue additionally emits per-128-token-block
        # metadata consumed by the fused GVR top-k (gvr_topk_decode.py).
        # Emission is warp-autonomous: each of the WG's 4 warps writes one
        # partial record per tile per t (record index = tile*4 + warp);
        # the GVR consumer folds the 4 partials per block. No cross-warp
        # barrier.
        #   block_max [num_rows, nb_pad*4] fp32 — warp-partial max of
        #     f32(stored logit) over valid positions (kv_pos < ctx),
        #     computed on the POST-conversion value so it bounds what GVR
        #     reads back bit-exactly.
        #   hit_agg [num_rows, 4] fp32 — per-row aggregate
        #     {enc_min, enc_max, sum, cnt} of stored logits at positions
        #     flagged in hit_bitmap; min/max slots hold the
        #     order-preserving int encoding (see _red_global_fmin_ordered).
        #     Buffer must be pre-initialized to
        #     {enc(+FLT_MAX), enc(-FLT_MAX), 0, 0} per step.
        #   hit_bitmap [batch, nb_pad*4] int32 — 1 bit per kv position
        #     (request-level; from the previous step's top-k).
        # Hit accumulators are lane-local and flush once per q-transition
        # via atomics (_flush_hit_agg).
        #
        # emit_hit_stats sub-knob (only meaningful with emit_block_meta):
        # False emits block_max ONLY (no bitmap read, no hit aggregate).
        self.emit_block_meta = emit_block_meta
        self.emit_hit_stats = emit_hit_stats
        # Deferred block_max emission: with only block_max requested, a
        # tile's warp fmax + record store are issued under the NEXT tile's
        # TMEM load instead of at the end of its epilogue chain. The
        # sub-knobs below consume r_bmax in place, so they keep the in-place
        # path.
        self.emit_block_meta_deferred = (
            emit_block_meta and not emit_hit_stats and not emit_seed_counts
        )
        # plain build: each tile's logits store is deferred into the next
        # tile's TMEM-load shadow (emitting builds place block_max there)
        self.defer_logits = not emit_block_meta
        # emit_seed_counts (requires emit_block_meta): per row, count
        # stored logits >= each of the T=3 caller-provided thresholds.
        # Counts are computed on the POST-conversion value over valid
        # positions only.
        if emit_seed_counts and not emit_block_meta:
            raise ValueError("emit_seed_counts requires emit_block_meta")
        self.emit_seed_counts = emit_seed_counts
        # seed_packed: single [num_rows, 8] fp32 seed row per the top-k
        # pre-packed contract - lines at cols 0..2, counts accumulated as
        # floats at cols 3..5 (exact to 2^24; red.global.add.f32). The
        # caller zeroes cols 3..7 and writes the lines each step.
        if seed_packed and not emit_seed_counts:
            raise ValueError("seed_packed requires emit_seed_counts")
        self.seed_packed = seed_packed
        # emit_cand: unordered pre-collect of all (value, index) pairs >=
        # the t_0 seed threshold. claimed >= K certifies the candidate set
        # covers the true top-K.
        if emit_cand and not emit_seed_counts:
            raise ValueError("emit_cand requires emit_seed_counts (t_0 source)")
        self.emit_cand = emit_cand
        self.cand_cap = cand_cap
        # emit_cand_bucketed: three fixed SoA segments (A=[0,segA) holds
        # >= t2, B=[segA,2segA) holds [t1,t2), C=[2segA,2segA+capC) holds
        # [t0,t1)); a full segment spills to the next looser one. A/B use
        # EXACT ballot claims (their prefixes must stay pad-free - the
        # consumer's prefix math assumes it), C keeps the claim-window
        # scheme (pads are legal there). Cursors live in caller-zeroed
        # cand_cur [rows,4]; ctl [rows,4] carries {n0 incl C pads, void,
        # n1, n2} with n1/n2 flushed from the seed counters.
        if emit_cand_bucketed and not emit_seed_counts:
            raise ValueError("emit_cand_bucketed requires emit_seed_counts")
        if emit_cand_bucketed and emit_cand:
            raise ValueError("emit_cand_bucketed and emit_cand are exclusive")
        self.emit_cand_bucketed = emit_cand_bucketed
        self.accept_cap = accept_cap
        # Per-warp claim window: one atomic claims (hits + CAND_WIN) slots.
        # The unconsumed tail is sentinel-filled (idx = -1) at
        # q-transition/loop end, so `claimed` over-approximates the true
        # count (counts[r][0] exact).
        self.CAND_WIN = 8
        # dynamic_sched: tail-only work stealing over the DG ranges. Every
        # CTA walks the head of its range unchanged; the last
        # (range >> tail_shift) pairs form its donation region, split into
        # C-pair chunks claimed through one global counter per range (owner
        # first, just in time, then any CTA that ran out of work). TMA warp 0
        # publishes row segments {row|flag<<16, kv0, n_pairs, ctx} into a
        # smem ring every role consumes in order. The regime is decided per
        # launch from the inputs (total_pairs >= nmin * num_ctas); below it
        # every CTA walks exactly its DG range. A thief scans every range's
        # claim counter and takes a chunk of a range that still holds many
        # (one atomic per probe). Global state lives in a caller-owned int32
        # buffer [0]=arrival, [64..64+NC)=per-range claim counters, restored
        # to zero by the last arriving CTA; launches sharing a buffer must be
        # stream-ordered.
        # Under PDL the kernel-entry griddepcontrol.wait orders the first
        # claim after the previous grid's reset of these words.
        self.dynamic_sched = dynamic_sched
        # pdl: launch with the programmatic-dependent-launch attribute and
        # wait at kernel entry before any role's first input read; default
        # follows TRTLLM_ENABLE_PDL. pdl_trigger: release the dependent top-k
        # 0 never, 1 at kernel entry, 2 at each CTA's tile-loop exit (the
        # dependent still waits for this grid's completion before reading)
        self.pdl = TRTLLM_ENABLE_PDL if pdl is None else bool(pdl)
        self.pdl_trigger = int(pdl_trigger)
        if self.pdl_trigger not in (0, 1, 2):
            raise ValueError("pdl_trigger must be 0, 1 or 2")
        self.ring_depth = ring_depth
        self.b_cap = b_cap
        # setmaxnreg split (producer warps, math WGs) of the 168 x 384 pool;
        # the fetcher and ring pops need 56 producer registers
        self.prod_regs, self.math_regs = (56, 224) if dynamic_sched else (24, 240)
        assert ring_depth >= 16 and ring_depth % 2 == 0
        assert b_cap % 32 == 0 and b_cap <= 1024
        assert num_sms <= 256, "dynamic scheduler encodes CTA ids in 8 bits"
        assert not (dynamic_sched and (emit_cand or emit_cand_bucketed)), (
            "dynamic_sched is incompatible with candidate emission (per-segment "
            "CAND_WIN flushes inflate `claimed` against cand_cap)"
        )
        # epi_bytes covers fp16 and bf16 (FP8 only handled fp16).
        self.epi_bytes = 2 if epi_dtype in (cutlass.Float16, cutlass.BFloat16) else 4
        # sW stage stride padded to 128-byte SMEM alignment for TMA bulk copy.
        # Without padding, e.g. fp16 + N=32 gives 64B per stage, so stage 1
        # at +64 would be misaligned (TMA requires 128-byte aligned SMEM dest).
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.epi_bytes
        self.output_dtype = output_dtype
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "num_heads // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 2

        self.num_math_threads = 256
        self.num_specialized_threads = 128
        self.threads_per_cta = 384
        self.num_math_warps = 8
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        self.num_q_stages = 3
        # next_n == 1: two accumulator stages per math warpgroup so the next
        # tile's MMA overlaps this tile's epilogue; next_n > 1 keeps one (TMEM).
        self.num_umma_stages = 2 if next_n == 1 else 1
        # KV pipeline depth (op knob TRTLLM_DSL_FP4_KV_STAGES): 6 = 117 KB dynamic
        # SMEM, each stage adds 17 KB (8 KB KV + 512 B SF per group), 10 = 185 KB
        assert 2 <= num_kv_stages <= 10, f"num_kv_stages={num_kv_stages} must be in [2, 10]"
        self.num_kv_stages = num_kv_stages
        # the fetcher throttles the work ring at D - ring_slack unpopped entries;
        # the laggard roles trail it by at most kv + umma stages
        self.ring_slack = max(8, self.num_kv_stages + self.num_umma_stages)
        assert self.ring_slack <= ring_depth // 2 - 4
        # Step 5.11: smem_pad_bytes (FP8 sub-partition opt knob) dropped.

        # acc_dtype is locked to fp32 for FP4 MXF4 SS (cannot be exposed).
        self.acc_dtype = cutlass.Float32
        # SF (UE8M0) static knobs.
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.sf_vec_size = 32
        # MXF4 inst K is 64 (FP8 was 32).
        self.umma_inst_k = 64
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        # Block-scaled MXF4 MMA. ab_dtype = Float4E2M1FN, sf_vec_size=32.
        # Inst K is locked to 64 inside `MmaMXF4Op`; the helper picks
        # SWIZZLE_64B for FP4 (head_dim/2 = 64 byte innermost).
        # NOTE: installed nvidia_cutlass_dsl ships the **legacy single-ab_dtype**
        # signature (7 positional args). The newer dkg-repo split a_dtype/b_dtype
        # signature is not available here.
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            a_dtype,  # ab_dtype: a and b share the same FP4 dtype
            a_major,
            b_major,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 64 for MXF4

        # Full-K: tile K = head_dim (128 FP4 elem), 1 TMA per block.
        # mma_inst_tile_k = 128 / 64 = 2 (FP4) vs 128 / 32 = 4 (FP8).
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k
        full_k = mma_inst_shape_k * mma_inst_tile_k  # 128
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            full_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]

        # KV SMEM: helper picks SWIZZLE_64B for FP4 (M7 reminder; verify when
        # printing the layout in dev).
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            a_dtype,
            self.num_kv_stages,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            b_dtype,
            self.num_q_stages,
        )

        # acc TMEM (per math WG, per UMMA stage)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, rounding=False)

        # TMEM SF layouts (single region, NOT staged — see plan U1).
        # Build a virtual 1-stage chunk SMEM layout to feed the TMEM helpers
        # (smem_layout param is only used for shape inference, doesn't
        # bind a runtime alloc; our actual SF SMEM is flat, see Step 4 plan).
        sfa_chunk_smem_for_inference = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_chunk_smem_for_inference = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        # NOTE: post-4.4.x DSL (CFK-3348x MMA Refactor — landed in 4.5.0 and
        # internal) tightened the C++ binding `_cute_nvgpu_ir.make_tmem_layout_sf{a,b}`
        # to require a rank-3 input layout ((MMA_inner), num_MMA_M, num_MMA_K)
        # and emit a rank-3 output. On 4.4.x both input and output preserved
        # the trailing stages mode. Slice the stages mode off the input to
        # satisfy 4.5.0+, then append a degenerate size-1 stages mode back to
        # the output so the downstream 4-tuple slice
        # `cute.slice_(tCtSF*, (None, None, k*sf_k_step, None))` and S2T copy
        # ranks remain unchanged across versions.
        _tmem_sfa_rank3 = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_chunk_smem_for_inference, (None, None, None, 0)),
        )
        _tmem_sfb_rank3 = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfb_chunk_smem_for_inference, (None, None, None, 0)),
        )
        self.tmem_sfa_layout = cute.append(_tmem_sfa_rank3, cute.make_layout(1, stride=0))
        self.tmem_sfb_layout = cute.append(_tmem_sfb_rank3, cute.make_layout(1, stride=0))
        # SF TMEM K-mode count differs by DSL version (CFK-29747).
        #   4.4.x              (hardcoded mma_tile_inst_k=4) → SF K-mode = 4, sf_k_step = 4//2 = 2
        #   4.5.0+ / internal  (dynamic = mma_tiler_k/UMMA_K)  → SF K-mode = 2, sf_k_step = 2//2 = 1
        # Body slices `cute.slice_(tCtSF*, (None, None, k_block * sf_k_step, None))`;
        # iterator-start address is invariant across versions.
        _sf_kmode_size = cute.size(_tmem_sfa_rank3, mode=[2])
        _mma_inst_tile_k_for_sf = self.head_dim // self.umma_inst_k  # = 2 for FP4
        assert _sf_kmode_size % _mma_inst_tile_k_for_sf == 0, (
            f"SF K-mode ({_sf_kmode_size}) must be divisible by UMMA K-inst count "
            f"({_mma_inst_tile_k_for_sf})"
        )
        self.sf_k_step = _sf_kmode_size // _mma_inst_tile_k_for_sf
        # SF TMEM col count — compute directly per plan formula.
        # `cute.cosize(tmem_layout)` returns the stride span (huge), not the
        # actual TMEM cell count, so we don't use it here.
        # SFA cols = UMMA_M / sf_atom_mn * (head_dim / UMMA_K)
        # SFB cols = N_padded / sf_atom_mn * (head_dim / UMMA_K)
        sf_atom_mn = 32  # BlockScaledBasicChunk M atom
        mma_inst_tile_k = self.head_dim // self.umma_inst_k  # 128/64 = 2
        N_padded = ((self.N + 127) // 128) * 128
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (N_padded // sf_atom_mn) * mma_inst_tile_k

        # Total TMEM = staged acc (per WG) + per-WG SFA + shared SFB (Q SF).
        # SFA must be per-WG: each UMMA warp does its own UTCCP+MMA over its
        # own KV block, so they must NOT share a single SFA TMEM region or
        # they race on writes (DeepGEMM uses kTmemStartColOfSFKV + i*4 per WG).
        raw_total = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            + self.num_sfa_tmem_cols * self.num_groups
            + self.num_sfb_tmem_cols
        )
        # TMEM allocator requires num_columns to be a power of two AND a
        # multiple of 32, between 32 and 512. Round up to next valid value.
        # Equivalent to utils.get_num_tmem_alloc_cols(..., rounding=True) but
        # without needing a tmem tensor handle (we already have raw_total).
        self.num_tmem_alloc_cols_total = max(1 << math.ceil(math.log2(raw_total)), 32)
        assert self.num_tmem_alloc_cols_total <= 512, (
            f"FP4 TMEM exceeds 512 cols: raw={raw_total}, "
            f"acc={self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages}, "
            f"sfa_per_wg={self.num_sfa_tmem_cols} x{self.num_groups}, "
            f"sfb={self.num_sfb_tmem_cols}, "
            f"total={self.num_tmem_alloc_cols_total}. next_n={self.next_n}, "
            f"num_umma_stages={self.num_umma_stages} — see plan TMEM table."
        )

        # Stash the chunk SMEM layouts; UMMA warp uses them as a reference
        # view onto our flat SF SMEM after the in-place transpose (Step 4).
        self.sfa_chunk_smem_layout = sfa_chunk_smem_for_inference
        self.sfb_chunk_smem_layout = sfb_chunk_smem_for_inference

        # UTCCP S2T copy atoms for SF SMEM -> TMEM SF.
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        # Stash atoms; partitioning happens at the use site in the kernel body
        # (chunk-view of post-transpose SMEM). Keeping the raw atom is enough
        # because `make_s2t_copy` is called per-tile when issuing UTCCP.
        self.copy_atom_s2t = copy_atom_s2t

        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,  # Fused KV: [num_phys_blocks, block_bytes] uint8
        # Per phys block: [data: phys_block_kv * head_dim/2 bytes]
        #                 [SF:   phys_block_kv * 4 bytes (UE8M0 packed int32)]
        b: cute.Tensor,  # Q: [N, head_dim/2, batch_size] uint8 (FP4 packed)
        sf_q: cute.Tensor,  # Q SF: [N, batch_size] int32 (4 UE8M0 packed per token)
        weights: cute.Tensor,  # [N, batch_size] epi_dtype (cast by host wrapper)
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        context_lens: cute.Tensor,  # [batch_size]
        schedule_meta: cute.Tensor,  # [num_sms+1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
        # emission-only tensors; defaulted so the positional signature
        # stays the one callers already use
        block_max: cute.Tensor = None,  # [num_rows, nb_pad*4] fp32 warp-partials
        hit_stats: cute.Tensor = None,  # [num_rows, 4] fp32 (emit_hit_stats)
        hit_bitmap: cute.Tensor = None,  # [batch, nb_pad*4] int32 (emit_block_meta)
        seed_thr: cute.Tensor = None,  # [num_rows, 3] fp32 (emit_seed_counts)
        seed_counts: cute.Tensor = None,  # [num_rows, 3] int32 out, caller-zeroed
        cand: cute.Tensor = None,  # [num_rows, CAP*2] int32 {val bits, idx} pairs
        cand_ctl: cute.Tensor = None,  # [num_rows, 2] int32 {claimed, void}, zeroed
        cand_idx_t: cute.Tensor = None,  # bucketed: [num_rows, 2*segA+capC] int32 SoA
        cand_cur: cute.Tensor = None,  # bucketed: [num_rows, 4] int32 cursors, zeroed
        dyn_state: cute.Tensor = None,  # dynamic_sched: int32 [>= 64 + roundup32(num_sms)]
        dyn_chunk: cutlass.Int32 = 16,  # bits 0-7 pairs per chunk (<= ring_depth/2 - 4), bits 8-15 tail shift
        dyn_nmin: cutlass.Int32 = 64,  # dynamic regime iff total_pairs >= nmin * num_ctas
    ):
        # Derive KV data and SF views from the fused uint8 buffer.
        # Fused layout per phys block: [data half_head_dim*phys_block_kv bytes]
        #                              [SF   phys_block_kv*4         bytes (= phys_block_kv int32)]
        phys_block_kv = self.phys_block_kv
        half_head_dim = self.head_dim // 2  # FP4 packed bytes per row
        scale_offset_bytes = phys_block_kv * half_head_dim  # to SF region of each phys block

        # Recast the fused buffer to FP4. Each uint8 byte becomes 2 FP4 elements,
        # so layout positions and the iterator scale accordingly.
        kv_fp4 = cute.recast_tensor(kv_fused, Float4E2M1FN)

        # Q (b) was passed as uint8 (FP4-packed bytes); recast to FP4 so MMA
        # type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, Float4E2M1FN)

        # Read the real per-block stride (bytes) from the input tensor.
        # When KV is the indexer K-cache pool view, the pool is laid out as
        # [num_blocks, num_layers, kvFactor, blockSize], so dim-0 stride =
        # num_layers * kvFactor * phys_block_bytes (not phys_block_bytes).
        # Using the input stride keeps both the contiguous test path and
        # the strided prod path correct.
        kv_block_stride_bytes = kv_fused.layout.stride[0]

        # KV data view: [phys_block_kv, head_dim, num_phys_blocks] FP4 elements.
        # Innermost stride 1 = consecutive FP4 elem = packed pair share a byte.
        # Per-row stride = head_dim FP4 elem = head_dim/2 bytes.
        # Per-block stride (FP4 elem) = kv_block_stride_bytes * 2 (uint8→FP4 doubles).
        kv_layout = cute.make_layout(
            (phys_block_kv, self.head_dim, num_phys_blocks),
            stride=(self.head_dim, 1, kv_block_stride_bytes * 2),
        )
        a = cute.make_tensor(kv_fp4.iterator, kv_layout)

        # SF KV view: int32 (4 UE8M0 packed). Build a uint8 view at the SF
        # offset, then recast to int32.
        # Layout in bytes: (phys_block_kv * 4, num_phys_blocks) stride (1, kv_block_stride_bytes)
        # After recast int32: (phys_block_kv, num_phys_blocks) stride (1, kv_block_stride_bytes/4)
        sf_kv_uint8_layout = cute.make_layout(
            (phys_block_kv * 4, num_phys_blocks),
            stride=(1, kv_block_stride_bytes),
        )
        sf_kv_uint8 = cute.make_tensor(kv_fused.iterator + scale_offset_bytes, sf_kv_uint8_layout)
        sf_kv = cute.recast_tensor(sf_kv_uint8, cutlass.Int32)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA for KV (A) — fmha_decode_paged pattern.
        # Build a TMA SMEM layout via tiled_divide on the full compute-tile
        # layout, then select to drop trivial K dim. Atom uses mode [0] as
        # single-tile SMEM layout and (phys, head) as cta_tiler.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_view_layout = sm100_utils.make_smem_layout(
            tcgen05.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        self.a_tma_view_layout = cute.tiled_divide(
            self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
        )
        # ((tile_M, tile_K), rest_M, rest_K, stages) → drop trivial rest_K
        self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
        # ((tile_M, tile_K), rest_M=num_sub_blocks, stages)
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            a,
            self.a_tma_view_layout[0],  # atom SMEM = single-tile (mode 0)
            (self.phys_block_kv, self.head_dim),
        )

        # TMA for Q (B) — full K=128, L dim = batch_size (unchanged)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA for Weights — [N, batch_size], tile [N], L=batch_size
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.w_stage_stride),
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            self.w_smem_layout_staged.shape[:1],
        )

        # TMA for SF KV — [phys_block_kv, num_phys_blocks] int32, tile [phys_block_kv]
        # SMEM holds compute_block_kv int32 SF per stage (= block_kv tokens);
        # filled by num_blocks_per_mma sub-block TMAs at consecutive offsets.
        # FLAT M-row-major; UMMA warp does an in-place SMEM
        # transpose to chunk byte layout before issuing UTCCP (Step 4).
        self.sf_kv_smem_layout_staged = cute.make_layout((self.block_kv, self.num_kv_stages))
        sf_kv_smem_per_subblock = cute.make_layout((phys_block_kv,))
        tma_atom_sf_kv, tma_tensor_sf_kv = cpasync.make_tiled_tma_atom(
            tma_load_op,
            sf_kv,
            sf_kv_smem_per_subblock,
            (phys_block_kv,),
        )

        # TMA for SF Q — [N, batch_size] int32, tile [N] (1D, like weights).
        # SMEM single stage size = N_padded int32 (UTCCP atom is 128-token
        # aligned, so SMEM allocation must round up; UTCCP reads the full
        # N_padded region). TMA descriptor tile = real N: TMA fetches only
        # the valid GMEM region; SMEM positions [N, N_padded) are left as
        # garbage. The garbage propagates through UTCCP into TMEM SFB cols
        # ≥ N, but MMA reads only SFB cols [0, N) since UMMA_N = N, and
        # the epilogue writes acc cols [0, N) — so the tail never affects
        # output. Mirrors DeepGEMM's tma::copy<kRealNumSFQAtom, ...> with
        # kNumSFQAtom-sized SMEM (sm100_fp4_paged_mqa_logits.cuh:202).
        N_padded = ((self.N + 127) // 128) * 128
        self.N_padded = N_padded
        # sf_q stage stride in int32 elem (= 4 bytes/elem). Pad to 128B (= 32 int32).
        sf_q_stage_bytes = N_padded * 4
        sf_q_stage_stride_int32 = ((sf_q_stage_bytes + 127) // 128 * 128) // 4
        self.sf_q_smem_layout_staged = cute.make_layout(
            (N_padded, self.num_q_stages),
            stride=(1, sf_q_stage_stride_int32),
        )
        # TMA atom uses a smaller per-stage layout matching real N (not N_padded)
        # so the DSL helper's symmetry check passes (it requires
        # cosize(smem_layout) == cosize(cta_v_map)). The actual SMEM allocation
        # still uses sf_q_smem_layout_staged with N_padded for UTCCP alignment;
        # only TMA atom construction + tma_partition use this smaller view.
        # Stride matches the staged layout so per-stage offsets are consistent.
        self.sf_q_tma_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, sf_q_stage_stride_int32),
        )
        sf_q_tma_smem_per_stage = cute.select(self.sf_q_tma_smem_layout_staged, mode=[0])
        tma_atom_sf_q, tma_tensor_sf_q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            sf_q,
            sf_q_tma_smem_per_stage,
            (self.N,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        # Per sub-block (FP4):
        #   phys_block_kv * (head_dim/2) bytes data + phys_block_kv * 4 bytes SF
        kv_tma_bytes_per_subblock = phys_block_kv * half_head_dim
        sf_kv_tma_bytes_per_subblock = phys_block_kv * 4
        # Total per compute tile = num_blocks_per_mma sub-blocks
        self.num_kv_sf_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + sf_kv_tma_bytes_per_subblock
        )
        # Q + SF_Q + Weights share barrier (Q-pipe). SF Q TMA descriptor tile
        # is now real N (see TMA atom construction above), so the actual
        # GMEM→SMEM transfer is N int32 = self.N * 4 bytes. Barrier tx_count
        # must match the real fetch.
        sf_q_tma_bytes = self.N * 4
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + sf_q_tma_bytes + w_copy_size

        num_ctas = self.num_sms
        ring_d = self.ring_depth if self.dynamic_sched else 1
        b_cap = self.b_cap if self.dynamic_sched else 1
        nc_d = (self.num_sms + 31) // 32 * 32 if self.dynamic_sched else 0

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            ring_mbar: cute.struct.MemRange[cutlass.Int64, ring_d * 2]
            tmem_holding_buf: cutlass.Int32
            # scheduler words re-read per warp role after setmaxnreg (a prologue
            # value carried across the role split is spilled function-wide)
            sched_state: cute.struct.MemRange[cutlass.Int32, 4]
            # dynamic_sched: ring entries, pair-prefix / ctx tables, fetcher state
            # (64 words) + per-range donation table (2 words per range)
            ring_ent: cute.struct.MemRange[cutlass.Int32, ring_d * 4]
            sched_P: cute.struct.MemRange[cutlass.Int32, b_cap + 1]
            sched_ctx: cute.struct.MemRange[cutlass.Int32, b_cap]
            sched_ctl: cute.struct.MemRange[cutlass.Int32, 64 + 2 * nc_d]

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_sf_kv,
            tma_tensor_sf_kv,
            tma_atom_sf_q,
            tma_tensor_sf_q,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            batch_size,
            block_max,
            hit_stats,
            hit_bitmap,
            seed_thr,
            seed_counts,
            cand,
            cand_ctl,
            cand_idx_t,
            cand_cur,
            dyn_state,
            dyn_chunk,
            dyn_nmin,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.sf_kv_smem_layout_staged,
            self.sf_q_smem_layout_staged,
            self.sf_q_tma_smem_layout_staged,
            self.tmem_sfa_layout,
            self.tmem_sfb_layout,
            self.sfa_chunk_smem_layout,
            self.sfb_chunk_smem_layout,
            self.copy_atom_s2t,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def _flush_hit_agg(
        self,
        mHitAgg,  # [num_rows, 4] fp32 {enc_min, enc_max, sum, cnt}
        q_flush,  # request whose accumulators are being flushed
        hacc_min,
        hacc_max,
        hacc_sum,
        hacc_cnt,
        meta_lane,
    ):
        """Warp-reduce the per-lane hit accumulators and merge them into the
        per-row global aggregate via one set of atomics (encoded-int
        min/max + fp32 adds), then reset to identities. Called once per
        q-transition per warp. The aggregate buffer must be pre-initialized
        to {enc(+FLT_MAX), enc(-FLT_MAX), 0, 0} per step."""
        next_n = cutlass.const_expr(self.next_n)
        base_addr = mHitAgg.iterator.toint()
        for t in cutlass.range_constexpr(next_n):
            w_min = cute.arch.warp_redux_sync(hacc_min[t], "fmin")
            w_max = cute.arch.warp_redux_sync(hacc_max[t], "fmax")
            w_sum = cute.arch.warp_reduction_sum(hacc_sum[t])
            w_cnt = cute.arch.warp_redux_sync(hacc_cnt[t], "add")
            if meta_lane == cutlass.Int32(0):
                if w_cnt > cutlass.Int32(0):
                    row = q_flush * cutlass.Int32(next_n) + cutlass.Int32(t)
                    row_addr = base_addr + cutlass.Int64(row) * cutlass.Int64(16)
                    _red_global_fmin_ordered(row_addr, w_min)
                    _red_global_fmax_ordered(row_addr + cutlass.Int64(4), w_max)
                    _red_global_add_f32(row_addr + cutlass.Int64(8), w_sum)
                    _red_global_add_f32(row_addr + cutlass.Int64(12), cutlass.Float32(w_cnt))
            hacc_min[t] = cutlass.Float32(_META_FLT_MAX)
            hacc_max[t] = cutlass.Float32(_META_NEG_FLT_MAX)
            hacc_sum[t] = cutlass.Float32(0.0)
            hacc_cnt[t] = cutlass.Int32(0)

    @cute.jit
    def _flush_seed_counts(self, mSeedCounts, q_idx, scnt, meta_lane, spass=None, cand_ctl=None):
        """Warp-redux the lane-local seed counters and fire one lane-0
        red.global.add per (t, threshold). Caller zero-initializes the
        count slots each step; cross-CTA totals accumulate atomically.

        seed_packed: mSeedCounts IS the [num_rows, 8] packed seed row -
        counts land as fp32 at cols 3..5 (exact to 2^24)."""
        next_n = cutlass.const_expr(self.next_n)
        base_addr = mSeedCounts.iterator.toint()
        for t in cutlass.range_constexpr(next_n):
            for j in cutlass.range_constexpr(3):
                w_cnt = cute.arch.warp_redux_sync(scnt[t * 3 + j], "add")
                if meta_lane == cutlass.Int32(0):
                    row = q_idx * cutlass.Int32(next_n) + cutlass.Int32(t)
                    if cutlass.const_expr(self.seed_packed):
                        addr = base_addr + (
                            cutlass.Int64(row) * cutlass.Int64(8) + cutlass.Int64(3 + j)
                        ) * cutlass.Int64(4)
                        _red_global_add_f32(addr, cutlass.Float32(w_cnt))
                    else:
                        addr = base_addr + (
                            cutlass.Int64(row) * cutlass.Int64(3) + cutlass.Int64(j)
                        ) * cutlass.Int64(4)
                        _red_global_add_s32(addr, w_cnt)
                if cutlass.const_expr(self.emit_cand_bucketed):
                    if j >= 1:
                        # consumer contract: ctl = {n0, void, n1, n2}
                        if meta_lane == cutlass.Int32(0):
                            row_b = q_idx * cutlass.Int32(next_n) + cutlass.Int32(t)
                            ctl_a = cand_ctl.iterator.toint() + (
                                cutlass.Int64(row_b) * cutlass.Int64(4) + cutlass.Int64(j + 1)
                            ) * cutlass.Int64(4)
                            _red_global_add_s32(ctl_a, w_cnt)
                scnt[t * 3 + j] = cutlass.Int32(0)
        if cutlass.const_expr(self.seed_packed and spass is not None):
            # packed col 6: adaptive-skip pass count (lane0-accumulated)
            for t in cutlass.range_constexpr(next_n):
                w_bp = cute.arch.warp_redux_sync(spass[t], "add")
                if meta_lane == cutlass.Int32(0):
                    row = q_idx * cutlass.Int32(next_n) + cutlass.Int32(t)
                    addr = base_addr + (
                        cutlass.Int64(row) * cutlass.Int64(8) + cutlass.Int64(6)
                    ) * cutlass.Int64(4)
                    _red_global_add_f32(addr, cutlass.Float32(w_bp))
                spass[t] = cutlass.Int32(0)

    @cute.jit
    def _flush_cand_window_bucketed(self, mCand, mCandIdx, q_idx, cwbase, cwleft, meta_lane):
        """Sentinel-fill the unconsumed C-window tail in BOTH SoA columns
        (score -inf, idx -1: the consumer pads by score) and invalidate
        the window. Segment C sits at base 2*segA in each row."""
        next_n = cutlass.const_expr(self.next_n)
        segA_f = cutlass.const_expr(self.accept_cap)
        capC_f = cutlass.const_expr(self.cand_cap)
        wtot_f = cutlass.const_expr(2 * self.accept_cap + self.cand_cap)
        vbase_f = mCand.iterator.toint()
        ibase_f = mCandIdx.iterator.toint()
        for t in cutlass.range_constexpr(next_n):
            if cwleft[t] > cutlass.Int32(0):
                row_f = q_idx * cutlass.Int32(next_n) + cutlass.Int32(t)
                sl_f = cwbase[t] + meta_lane
                if meta_lane < cwleft[t] and sl_f < cutlass.Int32(capC_f):
                    off_f = (
                        cutlass.Int64(row_f) * cutlass.Int64(wtot_f)
                        + cutlass.Int64(2 * segA_f + sl_f)
                    ) * cutlass.Int64(4)
                    vp_f = cute.make_ptr(
                        cutlass.Float32,
                        vbase_f + off_f,
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    cute.make_tensor(vp_f, cute.make_layout((1,)))[0] = cutlass.Float32(
                        _META_NEG_FLT_MAX
                    )
                    ip_f = cute.make_ptr(
                        cutlass.Int32,
                        ibase_f + off_f,
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    cute.make_tensor(ip_f, cute.make_layout((1,)))[0] = cutlass.Int32(-1)
                cwbase[t] = cutlass.Int32(0)
                cwleft[t] = cutlass.Int32(0)

    @cute.jit
    def _flush_cand_window(self, mCand, q_idx, cwbase, cwleft, meta_lane):
        """Sentinel-fill the unconsumed tail of each per-(warp, t) claim
        window (idx word = -1; consumers skip sentinels) and invalidate the
        window. wleft <= CAND_WIN + 31 always fits one lane round."""
        next_n = cutlass.const_expr(self.next_n)
        CAP_C = cutlass.const_expr(self.cand_cap)
        cand_base = mCand.iterator.toint()
        for t in cutlass.range_constexpr(next_n):
            if cwleft[t] > cutlass.Int32(0):
                row_c = q_idx * cutlass.Int32(next_n) + cutlass.Int32(t)
                sl_f = cwbase[t] + meta_lane
                if meta_lane < cwleft[t] and sl_f < cutlass.Int32(CAP_C):
                    pair_f = cand_base + (
                        cutlass.Int64(row_c) * cutlass.Int64(CAP_C) + cutlass.Int64(sl_f)
                    ) * cutlass.Int64(8)
                    iptr_f = cute.make_ptr(
                        cutlass.Int32,
                        pair_f + cutlass.Int64(4),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    cute.make_tensor(iptr_f, cute.make_layout((1,)))[0] = cutlass.Int32(-1)
            cwbase[t] = cutlass.Int32(0)
            cwleft[t] = cutlass.Int32(0)

    # ---- dynamic_sched helpers (warp-collective; all lanes of one warp call) ----
    # sched_ctl words: 0 mode (0 own, 1 steal, 2 finished, 3 static-tail),
    # 1/2 own donation region [t, e) flat pairs, 3 own chunk limit,
    # 6/7 pending segment [pf0, pf1), 8 pending first-of-chunk flag,
    # 9 chunks published (incl. terminal), 10 chunks popped by the fetcher,
    # 12 donation table built, 13/14 scan reductions, 16..18 ring producer
    # (count, index, phase), 32..63 strip totals (prologue scratch).

    @cute.jit
    def _ring_pop(self, ring, ring_cons, ring_ent, lane_idx):
        ring.consumer_wait(ring_cons)
        eb = ring_cons.index * 4
        re0 = ring_ent[eb]
        re1 = ring_ent[eb + 1]
        re2 = ring_ent[eb + 2]
        re3 = ring_ent[eb + 3]
        cute.arch.sync_warp()
        if lane_idx == cutlass.Int32(0):
            ring.consumer_release(ring_cons)
        return re0, re1, re2, re3

    @cute.jit
    def _ring_publish(self, ring, ring_ent, s_ctl, lane_idx, w0, w1, w2, w3):
        st = pipeline.PipelineState(self.ring_depth, s_ctl[16], s_ctl[17], s_ctl[18])
        ring.producer_acquire(st)
        if lane_idx == cutlass.Int32(0):
            eb = st.index * 4
            ring_ent[eb] = w0
            ring_ent[eb + 1] = w1
            ring_ent[eb + 2] = w2
            ring_ent[eb + 3] = w3
            ring.producer_commit(st)
        st.advance()
        if lane_idx == cutlass.Int32(0):
            s_ctl[16] = st.count
            s_ctl[17] = st.index
            s_ctl[18] = st.phase
        cute.arch.sync_warp()

    @cute.jit
    def _row_of(self, s_P, batch_size, lane_idx, f):
        """row r with P[r] <= f < P[r+1] (two 32-way ballot rounds; f < P[B])."""
        i1 = min(cutlass.Int32(32) * (lane_idx + cutlass.Int32(1)), batch_size)
        c1 = cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(s_P[i1] <= f)))
        i2 = min(cutlass.Int32(32) * c1 + lane_idx + cutlass.Int32(1), batch_size)
        c2 = cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(s_P[i2] <= f)))
        return cutlass.Int32(32) * c1 + c2

    @cute.jit
    def _dyn_drain(self, ring, ring_ent, s_P, s_ctx, s_ctl, batch_size, lane_idx, cons_count):
        """Publish up to 8 row segments of the pending range while the ring
        holds fewer than ring_depth - ring_slack entries past the fetcher's own pop."""
        D = cutlass.const_expr(self.ring_depth)
        SL = cutlass.const_expr(self.ring_slack)
        pf0 = s_ctl[6]
        pf1 = s_ctl[7]
        flag = s_ctl[8]
        k_it = cutlass.Int32(0)
        go = cutlass.Int32(1)
        while go == cutlass.Int32(1):
            pc = s_ctl[16]
            if pf0 < pf1 and pc - cons_count < cutlass.Int32(D - SL) and k_it < cutlass.Int32(8):
                row = self._row_of(s_P, batch_size, lane_idx, pf0)
                pb = s_P[row]
                pn = s_P[row + cutlass.Int32(1)]
                n = min(pf1, pn) - pf0
                self._ring_publish(
                    ring,
                    ring_ent,
                    s_ctl,
                    lane_idx,
                    row | (flag << 16),
                    (pf0 - pb) * cutlass.Int32(2),
                    n,
                    s_ctx[row],
                )
                pf0 = pf0 + n
                flag = cutlass.Int32(0)
                k_it = k_it + cutlass.Int32(1)
            else:
                go = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            s_ctl[6] = pf0
            s_ctl[8] = flag
        cute.arch.sync_warp()

    @cute.jit
    def _dyn_claim(
        self,
        ring,
        ring_ent,
        s_P,
        s_rng,
        s_ctl,
        mScheduleMeta,
        dyn_base,
        sm_idx,
        lane_idx,
        chunk,
        tail_sh,
    ):
        """One claim probe (at most one global atomic): the next own donation
        chunk (mode 0), else a chunk of a rich range (mode 1: the first range
        in ring order after sm_idx among those holding at least half of the
        maximum unclaimed chunks; a lost race leaves that counter past its
        limit, so the range drops out of later scans). With nothing left
        anywhere: arrive (the last arriver restores the zeros) and publish
        the terminal entry."""
        num_ctas = cutlass.const_expr(self.num_sms)
        NW = cutlass.const_expr((self.num_sms + 31) // 32)
        mode = s_ctl[0]
        if s_ctl[12] == cutlass.Int32(0) and mode != cutlass.Int32(3):
            # donation table [end, size) per range, built at this CTA's first
            # claim (its own range still has two chunks of work queued); own
            # and out-of-range entries get size 0
            qs_v = cute.make_rmem_tensor(NW, cutlass.Int32)
            ks_v = cute.make_rmem_tensor(NW, cutlass.Int32)
            qe_v = cute.make_rmem_tensor(NW, cutlass.Int32)
            ke_v = cute.make_rmem_tensor(NW, cutlass.Int32)
            for i in cutlass.range_constexpr(NW):
                vc = min(lane_idx + cutlass.Int32(32 * i), cutlass.Int32(num_ctas - 1))
                qs_v[i] = mScheduleMeta[(vc, 0)]
                ks_v[i] = mScheduleMeta[(vc, 1)]
                qe_v[i] = mScheduleMeta[(vc + cutlass.Int32(1), 0)]
                ke_v[i] = mScheduleMeta[(vc + cutlass.Int32(1), 1)]
            for i in cutlass.range_constexpr(NW):
                v = lane_idx + cutlass.Int32(32 * i)
                e_v = s_P[qe_v[i]] + ke_v[i]
                d_v = (e_v - s_P[qs_v[i]] - ks_v[i]) >> tail_sh
                if v >= cutlass.Int32(num_ctas) or v == sm_idx:
                    d_v = cutlass.Int32(0)
                s_rng[v * 2] = e_v
                s_rng[v * 2 + 1] = d_v
            cute.arch.sync_warp()
            if lane_idx == cutlass.Int32(0):
                s_ctl[12] = cutlass.Int32(1)
        got = cutlass.Int32(0)
        pf0 = cutlass.Int32(0)
        pf1 = cutlass.Int32(0)
        if mode == cutlass.Int32(0):
            lim = s_ctl[3]
            s0 = s_ctl[1]
            e0 = s_ctl[2]
            old = cutlass.Int32(0)
            if lane_idx == cutlass.Int32(0):
                old = _atom_global_add_s32(
                    dyn_base + cutlass.Int64(64 + sm_idx) * cutlass.Int64(4), cutlass.Int32(1)
                )
            old = cute.arch.shuffle_sync(old, cutlass.Int32(0))
            k = old + cutlass.Int32(1)
            if k < lim:
                pf0 = s0 + k * chunk
                pf1 = min(pf0 + chunk, e0)
                got = cutlass.Int32(1)
            else:
                mode = cutlass.Int32(1)
        go_done = cutlass.Int32(0)
        if mode == cutlass.Int32(1) and got == cutlass.Int32(0):
            # lane l scans ranges v = l + 32 i
            cnt = cute.make_rmem_tensor(NW, cutlass.Int32)
            for i in cutlass.range_constexpr(NW):
                vc = min(lane_idx + cutlass.Int32(32 * i), cutlass.Int32(num_ctas - 1))
                cnt[i] = _ld_relaxed_gpu_s32(dyn_base + cutlass.Int64(64 + vc) * cutlass.Int64(4))
            rem_max = cutlass.Int32(0)
            for i in cutlass.range_constexpr(NW):
                v = lane_idx + cutlass.Int32(32 * i)
                rem = (
                    (s_rng[v * 2 + 1] + chunk - cutlass.Int32(1)) // chunk
                    - cutlass.Int32(1)
                    - cnt[i]
                )
                rem_max = max(rem_max, rem)
            for k in cutlass.range_constexpr(5):
                rem_max = max(rem_max, cute.arch.shuffle_sync_bfly(rem_max, 1 << k))
            if lane_idx == cutlass.Int32(0):
                s_ctl[13] = rem_max
            cute.arch.sync_warp()
            rem_max = s_ctl[13]
            if rem_max <= cutlass.Int32(0):
                go_done = cutlass.Int32(1)
            else:
                thr = (rem_max + cutlass.Int32(1)) >> 1
                pick = cutlass.Int32(0x7FFFFFFF)
                for i in cutlass.range_constexpr(NW):
                    v = lane_idx + cutlass.Int32(32 * i)
                    rem = (
                        (s_rng[v * 2 + 1] + chunk - cutlass.Int32(1)) // chunk
                        - cutlass.Int32(1)
                        - cnt[i]
                    )
                    if rem >= thr:
                        dist = (v - sm_idx - cutlass.Int32(1)) & cutlass.Int32(0xFF)
                        pick = min(pick, (dist << 8) | v)
                for k in cutlass.range_constexpr(5):
                    pick = min(pick, cute.arch.shuffle_sync_bfly(pick, 1 << k))
                if lane_idx == cutlass.Int32(0):
                    s_ctl[14] = pick
                cute.arch.sync_warp()
                v = s_ctl[14] & cutlass.Int32(0xFF)
                old = cutlass.Int32(0)
                if lane_idx == cutlass.Int32(0):
                    old = _atom_global_add_s32(
                        dyn_base + cutlass.Int64(64 + v) * cutlass.Int64(4), cutlass.Int32(1)
                    )
                e_v = s_rng[v * 2]
                d_v = s_rng[v * 2 + 1]
                lim_v = (d_v + chunk - cutlass.Int32(1)) // chunk
                old = cute.arch.shuffle_sync(old, cutlass.Int32(0))
                k = old + cutlass.Int32(1)
                if k < lim_v:
                    pf0 = e_v - d_v + k * chunk
                    pf1 = min(pf0 + chunk, e_v)
                    got = cutlass.Int32(1)
        if got == cutlass.Int32(1):
            if lane_idx == cutlass.Int32(0):
                s_ctl[0] = mode
                s_ctl[6] = pf0
                s_ctl[7] = pf1
                s_ctl[8] = cutlass.Int32(1)
                s_ctl[9] = s_ctl[9] + cutlass.Int32(1)
        if go_done == cutlass.Int32(1):
            # one arrival per CTA; the last arriver restores the zeros
            a = cutlass.Int32(0)
            if lane_idx == cutlass.Int32(0):
                cute.arch.fence_acq_rel_gpu()
                a = _atom_global_add_s32(dyn_base, cutlass.Int32(1))
            a = cute.arch.shuffle_sync(a, cutlass.Int32(0))
            if a == cutlass.Int32(num_ctas - 1):
                cute.arch.fence_acq_rel_gpu()
                if lane_idx == cutlass.Int32(0):
                    _st_global_s32(dyn_base, cutlass.Int32(0))
                for _j in cutlass.range_constexpr(NW):
                    _st_global_s32(
                        dyn_base + cutlass.Int64(64 + 32 * _j + lane_idx) * cutlass.Int64(4),
                        cutlass.Int32(0),
                    )
        if go_done == cutlass.Int32(1) or mode == cutlass.Int32(3):
            self._ring_publish(
                ring,
                ring_ent,
                s_ctl,
                lane_idx,
                cutlass.Int32(1 << 16),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            )
            if lane_idx == cutlass.Int32(0):
                s_ctl[0] = cutlass.Int32(2)
                s_ctl[9] = s_ctl[9] + cutlass.Int32(1)
        cute.arch.sync_warp()

    @cute.jit
    def _fetch_step(
        self,
        ring,
        ring_ent,
        s_P,
        s_ctx,
        s_rng,
        s_ctl,
        mScheduleMeta,
        dyn_base,
        batch_size,
        sm_idx,
        lane_idx,
        cons_count,
        chunk,
        tail_sh,
    ):
        """Claim the next chunk when the pending range is drained and fewer
        than two chunks are claimed ahead of the fetcher's pops; then drain.
        Returns 1 while a further step can make progress."""
        mode = s_ctl[0]
        pf0 = s_ctl[6]
        pf1 = s_ctl[7]
        ahead = s_ctl[9] - s_ctl[10]
        if pf0 == pf1 and mode != cutlass.Int32(2) and ahead < cutlass.Int32(2):
            self._dyn_claim(
                ring,
                ring_ent,
                s_P,
                s_rng,
                s_ctl,
                mScheduleMeta,
                dyn_base,
                sm_idx,
                lane_idx,
                chunk,
                tail_sh,
            )
        self._dyn_drain(ring, ring_ent, s_P, s_ctx, s_ctl, batch_size, lane_idx, cons_count)
        need = cutlass.Int32(0)
        if s_ctl[6] < s_ctl[7]:
            need = cutlass.Int32(1)
        if s_ctl[0] != cutlass.Int32(2) and s_ctl[9] - s_ctl[10] < cutlass.Int32(2):
            need = cutlass.Int32(1)
        return need

    @cute.jit
    def _issue_q(
        self,
        q_pipeline,
        q_prod_state,
        tma_atom_b,
        tBgB,
        tBsB,
        b_mcast_mask,
        tma_atom_sf_q,
        tSF_Q_gSF_Q,
        tSF_Q_sSF_Q,
        tma_atom_w,
        tWgW,
        tWsW,
        row,
    ):
        q_pipeline.producer_acquire(q_prod_state)
        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
        cute.copy(
            tma_atom_b,
            tBgB[(None, 0, row)],
            tBsB[(None, q_prod_state.index)],
            tma_bar_ptr=q_bar,
            mcast_mask=b_mcast_mask,
        )
        cute.copy(
            tma_atom_sf_q,
            tSF_Q_gSF_Q[(None, row)],
            tSF_Q_sSF_Q[(None, q_prod_state.index)],
            tma_bar_ptr=q_bar,
        )
        cute.copy(
            tma_atom_w,
            tWgW[(None, row)],
            tWsW[(None, q_prod_state.index)],
            tma_bar_ptr=q_bar,
        )

    @cute.jit
    def _load_schedule_row(self, mScheduleMeta, s_sched, sm_idx, num_math_wg):
        # s_sched = [start tile, start row, end tile, end row] (tiles = half-units x num_math_wg)
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_half = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_half = mScheduleMeta[(sm_idx + 1, 1)]
        s_sched[0] = start_kv_half * num_math_wg
        s_sched[1] = start_q
        s_sched[2] = end_kv_half * num_math_wg
        s_sched[3] = end_q_idx

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV pool (FP4)
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = batch_size, FP4)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, batch_size]
        tma_atom_sf_kv: cute.CopyAtom,
        mSF_KV_tma: cute.Tensor,  # SF KV TMA coord tensor [phys_block_kv, num_phys_blocks] int32
        tma_atom_sf_q: cute.CopyAtom,
        mSF_Q_tma: cute.Tensor,  # SF Q TMA coord tensor [N, batch_size] int32
        mLogits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        mBlockTable: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        mContextLens: cute.Tensor,  # [batch_size]
        mScheduleMeta: cute.Tensor,  # [num_sms+1, 2] int32
        batch_size: cutlass.Int32,
        mBlockMax: cute.Tensor,  # [num_rows, nb_pad*4] fp32 warp-partials (or None)
        mHitAgg: cute.Tensor,  # [num_rows, 4] fp32 {enc_min, enc_max, sum, cnt} (or None)
        mHitBitmap: cute.Tensor,  # [batch, nb_pad*4] int32 (or None)
        mSeedThr: cute.Tensor,  # [num_rows, 3] fp32 seed thresholds (or None)
        mSeedCounts: cute.Tensor,  # [num_rows, 3] int32 counts out (or None)
        mCand: cute.Tensor,  # [num_rows, CAP*2] int32 pair scatter (or None)
        mCandCtl: cute.Tensor,  # [num_rows, 2] int32 {claimed, void} (or None)
        mCandIdx: cute.Tensor,  # bucketed: [num_rows, 2*segA+capC] int32 SoA (or None)
        mCandCur: cute.Tensor,  # bucketed: [num_rows, 4] int32 cursors (or None)
        mDynState: cute.Tensor,  # dynamic_sched state words (or None)
        dyn_chunk: cutlass.Int32,
        dyn_nmin: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        sf_kv_smem_layout_staged: cute.Layout,
        sf_q_smem_layout_staged: cute.Layout,
        sf_q_tma_smem_layout_staged: cute.Layout,
        tmem_sfa_layout: cute.Layout,
        tmem_sfb_layout: cute.Layout,
        sfa_chunk_smem_layout: cute.Layout,
        sfb_chunk_smem_layout: cute.Layout,
        copy_atom_s2t: cute.CopyAtom,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Warp roles
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        is_umma_warp_1 = warp_idx == 11

        # Early schedule metadata load: issue global loads ASAP so their
        # ~200-cycle L2 latency overlaps with subsequent prologue setup
        # (SMEM alloc, TMA partition, MMA fragment creation, etc.)
        NUM_MATH_WG = 2  # kNumMathWarpGroups
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        sm_idx = bidz
        if cutlass.const_expr(self.pdl):
            # every warp waits here, before the role split: the reads below
            # (schedule_meta, context_lens, block table, q / K / weights via
            # TMA, the dynamic scheduler's state words) all follow it
            griddepcontrol_wait()
            if cutlass.const_expr(self.pdl_trigger == 1):
                griddepcontrol_launch_dependents()

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_sf_kv)
            cpasync.prefetch_descriptor(tma_atom_sf_q)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        lane_idx = tidx % 32
        left = cutlass.Int32(0)
        trig = cutlass.Int32(-1)
        ring = None
        ring_cons = None
        ring_ent = None
        s_P = None
        s_ctx = None
        s_ctl = None
        s_rng = None
        if cutlass.const_expr(self.dynamic_sched):
            D_RING = self.ring_depth
            B_CAP = self.b_cap
            NSTRIP = B_CAP // 32
            ring_ent = cute.make_tensor(
                storage.ring_ent.data_ptr(), cute.make_layout((4 * D_RING,))
            )
            s_P = cute.make_tensor(storage.sched_P.data_ptr(), cute.make_layout((B_CAP + 1,)))
            s_ctx = cute.make_tensor(storage.sched_ctx.data_ptr(), cute.make_layout((B_CAP,)))
            s_ctl = cute.make_tensor(storage.sched_ctl.data_ptr(), cute.make_layout((64,)))
            s_rng = cute.make_tensor(
                storage.sched_ctl.data_ptr() + 64,
                cute.make_layout((2 * ((self.num_sms + 31) // 32 * 32),)),
            )
            if batch_size > cutlass.Int32(B_CAP):
                _trap()
            # P[r] = pairs of rows < r: warp w scans strips w, w+12, w+24 of 32 rows
            strip_incl = cute.make_rmem_tensor(3, cutlass.Int32)
            for s_i in cutlass.range_constexpr(3):
                strip = warp_idx + 12 * s_i
                strip_incl[s_i] = cutlass.Int32(0)
                if strip < NSTRIP:
                    p = cutlass.Int32(0)
                    if strip * 32 < batch_size:
                        row = strip * 32 + lane_idx
                        if row < batch_size:
                            c = mContextLens[row]
                            s_ctx[row] = c
                            # a zero-tile row still costs one pair (static visits it once)
                            p = max(
                                (((c + cutlass.Int32(127)) >> 7) + cutlass.Int32(1)) >> 1,
                                cutlass.Int32(1),
                            )
                        for k in cutlass.range_constexpr(5):
                            o = cute.arch.shuffle_sync_up(p, 1 << k, mask_and_clamp=0)
                            if lane_idx >= cutlass.Int32(1 << k):
                                p = p + o
                    strip_incl[s_i] = p
                    if lane_idx == cutlass.Int32(31):
                        s_ctl[32 + strip] = p
            cute.arch.barrier()
            tot_l = s_ctl[32 + lane_idx]
            incl_t = tot_l
            for k in cutlass.range_constexpr(5):
                o = cute.arch.shuffle_sync_up(incl_t, 1 << k, mask_and_clamp=0)
                if lane_idx >= cutlass.Int32(1 << k):
                    incl_t = incl_t + o
            for s_i in cutlass.range_constexpr(3):
                strip = warp_idx + 12 * s_i
                if strip < NSTRIP:
                    base = cute.arch.shuffle_sync(incl_t, strip) - cute.arch.shuffle_sync(
                        tot_l, strip
                    )
                    row = strip * 32 + lane_idx
                    if row < batch_size:
                        s_P[row + 1] = base + strip_incl[s_i]
            if tidx == cutlass.Int32(0):
                s_P[0] = cutlass.Int32(0)

        # Schedule row: one thread loads it and broadcasts it through smem, so
        # no other warp consumes a global load before the block barrier (warp 0's
        # mbarrier init overlaps the load). Every role reads s_sched after it.
        s_sched = cute.make_tensor(storage.sched_state.data_ptr(), cute.make_layout((4,)))
        if tidx == cutlass.Int32(11 * 32):
            self._load_schedule_row(mScheduleMeta, s_sched, sm_idx, NUM_MATH_WG)

        block_kv_val = self.block_kv
        num_heads = self.num_heads
        next_n = self.next_n
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer → Math consumer (8 math warps)
        # PipelineTmaAsync: consumer_release uses is_signalling_thread
        # (lane 0 per warp). 8 math warps × 1 lane-0 = 8 arrives.
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        # Both Math WGs share the same pipeline state (advance in lockstep)
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        # UMMA warps observe Q pipeline (wait only, no release)
        # to ensure Q is in SMEM before GEMM. Critical for UMMA warp 1
        # since TMA warp 1 only loads KV1 (not Q).
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        # Merged KV+Scale pipelines (per-group, num_kv_stages each).
        # Step 5.10: For FP4, UMMA owns release (it consumes both KV data and
        # SF for the block-scaled MMA). Math warp does NOT wait/release on
        # this pipeline anymore (the SF is baked into the acc by the MMA).
        # consumer_group = 1 thread = lane 0 of the single UMMA warp.
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_sf_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_sf_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )

        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        # Step 5.10: UMMA consumer states (wait + release, owns the pipeline).
        # FP8 had a separate math-side consumer state — removed since the math
        # warp no longer waits on KV+SF (block-scaled MMA bakes SF into acc).
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )

        if cutlass.const_expr(self.dynamic_sched):
            # work ring: fetcher lane 0 produces, lane 0 of all 12 warps releases
            ring = pipeline.PipelineAsync.create(
                barrier_storage=storage.ring_mbar.data_ptr(),
                num_stages=self.ring_depth,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 12),
                defer_sync=True,
            )
            ring_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ring_depth
            )
            ring_prod_init = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ring_depth
            )
            if tidx == cutlass.Int32(0):
                s_ctl[16] = ring_prod_init.count
                s_ctl[17] = ring_prod_init.index
                s_ctl[18] = ring_prod_init.phase

        # TMEM — only Math warps (8×32=256) + UMMA warps (2×32=64) = 320 threads
        # TMA warps do NOT participate, so they can start TMA loads earlier.
        # Math warp 0 is the allocator (like fp16_gemm_3's epilogue warp 0),
        # because math warps are the last TMEM consumers (epilogue reads).
        tmem_alloc_num_threads = 320  # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=tmem_alloc_num_threads)
        # SFB (Q SF) cross-warp TMEM-write visibility:
        # umma_warp_0 owns the SMEM transpose + s2t copy of SF Q into TMEM SFB
        # on every q_idx transition; umma_warp_1's MMA reads the same SFB TMEM
        # region. consumer_wait on the Q pipeline only orders TMA→SMEM, not the
        # cross-warp TMEM write. Without this barrier, umma_warp_1 can fire its
        # MMA before umma_warp_0's s2t lands, reading stale SFB from the prior
        # batch — the source of B>1 numerical mismatches at large ctx + next_n.
        # 64 threads = 32 (umma_warp_0) + 32 (umma_warp_1). DeepGEMM avoids this
        # entirely by issuing both groups' UMMAs from a single UMMA warp.
        sfb_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,  # math warp 0 does alloc+free (last TMEM consumer)
            is_two_cta=False,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        # tcgen05.alloc by math warp 0; the 320-thread retrieve barrier stays
        # in the math / UMMA branches.
        tmem.allocate(self.num_tmem_alloc_cols_total)

        # SMEM allocation: per-group KV + shared Q
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # Step 5.1: SF_Q SMEM (int32, packed UE8M0; flat layout, transposed
        # in-place by UMMA warp before UTCCP — see plan U2).
        sSF_Q = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_q_smem_layout_staged,
            byte_alignment=128,
        )
        # Step 5.11: smem_pad_bytes block removed — perf knob always 0 in FP4.
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        # Step 5.1: SF_KV SMEM (int32, packed UE8M0; flat layout, transposed
        # in-place by UMMA warp before UTCCP). Renamed from sScales (FP8 fp32).
        sSF_KV_0 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_kv_smem_layout_staged,
            byte_alignment=128,
        )
        sSF_KV_1 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=sf_kv_smem_layout_staged,
            byte_alignment=128,
        )
        # Block-meta emission is warp-autonomous: each warp's lane 0 writes
        # its own warp-partial record straight to GMEM; the GVR side folds
        # 4 partials per block. No SMEM scratch, no named barrier.

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): fmha_decode_paged pattern.
        # SMEM view is ((tile), num_sub_blocks, stages) — built in __call__.
        # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
        # GMEM: local_tile by (phys, head), then group first 2 modes into tile.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
        sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(
            mA_mkl,
            (self.phys_block_kv, self.head_dim),
            coord=(None, None, None),
        )
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_0_for_tma,
            cute.group_modes(gA, 0, 2),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_1_for_tma,
            cute.group_modes(gA, 0, 2),
        )

        # Partition Q (B): shared SMEM, L dim = batch_size
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, None)]  # [tma, K, L]

        # Partition Weights: standalone TMA, [N, batch_size] → [N] per stage
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )

        # Step 5.2: Partition SF KV: explicit sub_blocks + stages dims.
        # Layout (phys_block_kv, num_sub, stages) K-major with custom strides.
        # Strides are int32-element units, same numeric stride values as FP8's
        # fp32 case (4-byte element / element-major-1) — so no shape change.
        sf_kv_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sSF_KV_0_for_tma = cute.make_tensor(sSF_KV_0.iterator, sf_kv_tma_view_layout)
        sSF_KV_1_for_tma = cute.make_tensor(sSF_KV_1.iterator, sf_kv_tma_view_layout)
        # GMEM: local_tile by phys to match atom's tile size
        gSF_KV = cute.local_tile(mSF_KV_tma, (self.phys_block_kv,), coord=(None, None))
        tSsSF_KV_0, tSgSF_KV_0 = cpasync.tma_partition(
            tma_atom_sf_kv,
            0,
            cute.make_layout(1),
            sSF_KV_0_for_tma,
            gSF_KV,
        )
        tSsSF_KV_1, tSgSF_KV_1 = cpasync.tma_partition(
            tma_atom_sf_kv,
            0,
            cute.make_layout(1),
            sSF_KV_1_for_tma,
            gSF_KV,
        )

        # Step 5.3: Partition SF Q: standalone TMA, [N, batch_size] → [N] per
        # stage (parallel to weights). Single SMEM ring buffer (sSF_Q has
        # num_q_stages stages).
        # The TMA atom was built with a smaller (N, num_q_stages) layout to
        # satisfy the DSL helper's symmetry check; here we make a matching
        # logical view of sSF_Q (sharing the same SMEM iterator) so
        # tma_partition produces N-element tiles. The physical SMEM is still
        # sf_q_smem_layout_staged-shaped (N_padded per stage); positions
        # ≥ self.N in each stage are untouched by TMA but still readable by
        # UTCCP via the original sSF_Q.
        sSF_Q_for_tma = cute.make_tensor(sSF_Q.iterator, sf_q_tma_smem_layout_staged)
        sf_q_cta_layout = cute.make_layout((1,))
        tSF_Q_sSF_Q, tSF_Q_gSF_Q = cpasync.tma_partition(
            tma_atom_sf_q,
            0,
            sf_q_cta_layout,
            cute.group_modes(sSF_Q_for_tma, 0, 1),
            cute.group_modes(mSF_Q_tma, 0, 1),
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)  # shared
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # Staged acc (fp16_gemm_3 pattern): append UMMA stage dim
        # shape: (*acc_shape, STAGE) — dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)

        # Epilogue setup
        c_layout = utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], num_heads // num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # Scheduler state is derived per role from s_sched after the barrier
        # (a prologue value carried across setmaxnreg is spilled function-wide).
        # Sentinel: no previous batch (q_idx = batch_size)
        q_idx = batch_size

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            # TMA warp 0: loads Q (prefetch) + KV for group 0
            cute.arch.warpgroup_reg_dealloc(self.prod_regs)
            next_kv_idx = s_sched[0]
            next_q_idx = s_sched[1]
            end_kv_idx = s_sched[2]
            end_q_idx = s_sched[3]
            lane_idx = tidx % 32
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
            blk_row = min(next_q_idx, batch_size - 1)
            # num_kv of the first tile is derived in the advance step, after its
            # KV burst (a load here would be consumed before the first TMA issue).
            next_num_kv = cutlass.Int32(0)

            # Block table prefetch: 32 lanes cache block indices,
            # distributed via shuffle. Each lane holds num_blocks_per_mma
            # physical block indices per compute tile.
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use
            q_row = cutlass.Int32(-1)
            dyn_base = cutlass.Int64(0)
            chunk_c = cutlass.Int32(1)
            tail_sh = cutlass.Int32(0)

            if cutlass.const_expr(self.dynamic_sched):
                dyn_base = mDynState.iterator.toint()
                chunk_c = dyn_chunk & cutlass.Int32(0xFF)
                tail_sh = dyn_chunk >> 8
                own_s = s_P[next_q_idx] + (next_kv_idx >> 1)
                own_e = s_P[end_q_idx] + (end_kv_idx >> 1)
                total_pairs = s_P[batch_size]
                # donation region [own_t, own_e); the head [own_s, own_t) plus
                # chunk 0 is this CTA's initial pending segment
                own_t = own_e - ((own_e - own_s) >> tail_sh)
                lim_i = (own_e - own_t + chunk_c - cutlass.Int32(1)) // chunk_c
                dyn_on = total_pairs >= dyn_nmin * cutlass.Int32(self.num_sms)
                if lane_idx == cutlass.Int32(0):
                    s_ctl[1] = own_t
                    s_ctl[2] = own_e
                    s_ctl[3] = lim_i
                    mode0 = cutlass.Int32(3)
                    pf1_0 = own_e
                    if dyn_on:
                        mode0 = cutlass.Int32(0)
                        pf1_0 = min(own_t + chunk_c, own_e)
                    s_ctl[0] = mode0
                    s_ctl[6] = own_s
                    s_ctl[7] = pf1_0
                    s_ctl[8] = cutlass.Int32(1)
                    cpub0 = cutlass.Int32(1)
                    if own_e == own_s:
                        cpub0 = cutlass.Int32(0)
                    s_ctl[9] = cpub0
                    s_ctl[10] = cutlass.Int32(0)
                    s_ctl[12] = cutlass.Int32(0)
                cute.arch.sync_warp()
                need_first = s_ctl[16] == ring_cons.count
                while need_first:
                    self._fetch_step(
                        ring,
                        ring_ent,
                        s_P,
                        s_ctx,
                        s_rng,
                        s_ctl,
                        mScheduleMeta,
                        dyn_base,
                        batch_size,
                        sm_idx,
                        lane_idx,
                        ring_cons.count,
                        chunk_c,
                        tail_sh,
                    )
                    need_first = s_ctl[16] == ring_cons.count
                re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                ring_cons.advance()
                next_q_idx = re0 & cutlass.Int32(0xFFFF)
                next_kv_idx = re1
                left = re2
                next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                has_work = re2 > cutlass.Int32(0)
                # one scheduler step per segment, two chunks (or half the
                # segment) before its end
                trig = left - (chunk_c + chunk_c)
                if trig < (left >> 1):
                    trig = left >> 1
                if lane_idx == cutlass.Int32(0):
                    s_ctl[10] = s_ctl[10] + (re0 >> 16)
                cute.arch.sync_warp()
                if has_work:
                    self._issue_q(
                        q_pipeline,
                        q_prod_state,
                        tma_atom_b,
                        tBgB,
                        tBsB,
                        b_mcast_mask,
                        tma_atom_sf_q,
                        tSF_Q_gSF_Q,
                        tSF_Q_sSF_Q,
                        tma_atom_w,
                        tWgW,
                        tWsW,
                        next_q_idx,
                    )
                    q_prod_state.advance()
                    q_row = next_q_idx
            else:
                # First tile pair's page indices: issued before the Q TMA so the
                # load overlaps the TMA issue and the KV producer_acquire.
                kv_blk_ptr = cutlass.Int32(0)
                # Lanes past the row read in-bounds entries that are never consumed.
                pf_base = (next_kv_idx + lane_idx * NUM_MATH_WG) * NUM_BLOCKS_PER_MMA
                pf_last = cute.size(mBlockTable, mode=[1]) - 1
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cached_blks[i] = mBlockTable[(blk_row, min(pf_base + i, pf_last))]
                # Prefetch first Q before loop
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, next_q_idx)],
                    tBsB[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                # Step 5.4: SF_Q TMA load — under same q_bar as Q + W.
                cute.copy(
                    tma_atom_sf_q,
                    tSF_Q_gSF_Q[(None, next_q_idx)],
                    tSF_Q_sSF_Q[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                )
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, next_q_idx)],
                    tWsW[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                )
                q_prod_state.advance()

            while has_work:
                # fetch_next_task: commit next → current
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Block table prefetch for group 0.
                # Each lane loads num_blocks_per_mma physical block indices
                # for one compute tile (kv_idx counts compute tiles).
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + lane_idx * NUM_MATH_WG
                    if prefetch_kv < num_kv:
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[(q_idx, base_phys + i)]
                    else:
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = cutlass.Int32(0)

                # Get block indices via shuffle before barrier.
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1

                # Load KV + Scale for group 0: num_blocks_per_mma TMAs per tile.
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                stage = kv_prod_state_0.index
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cute.copy(
                        tma_atom_a,
                        tAgA_0[(None, 0, 0, phys_blks[i])],
                        tAsA_0[(None, i, stage)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sf_kv,
                        tSgSF_KV_0[(None, 0, phys_blks[i])],
                        tSsSF_KV_0[(None, i, stage)],
                        tma_bar_ptr=bar,
                    )
                kv_prod_state_0.advance()

                # Q prefetch for the next row, after this tile's KV burst.
                if cutlass.const_expr(not self.dynamic_sched):
                    if q_idx != q_idx_old:
                        prefetch_next = q_idx + 1
                        if prefetch_next < end_q_idx:
                            q_pipeline.producer_acquire(q_prod_state)
                            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, 0, prefetch_next)],
                                tBsB[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                                mcast_mask=b_mcast_mask,
                            )
                            # Step 5.4: SF_Q TMA load — under same q_bar.
                            cute.copy(
                                tma_atom_sf_q,
                                tSF_Q_gSF_Q[(None, prefetch_next)],
                                tSF_Q_sSF_Q[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                            )
                            cute.copy(
                                tma_atom_w,
                                tWgW[(None, prefetch_next)],
                                tWsW[(None, q_prod_state.index)],
                                tma_bar_ptr=q_bar,
                            )
                            q_prod_state.advance()
                        elif prefetch_next == end_q_idx:
                            if end_kv_idx > 0:
                                q_pipeline.producer_acquire(q_prod_state)
                                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                                cute.copy(
                                    tma_atom_b,
                                    tBgB[(None, 0, prefetch_next)],
                                    tBsB[(None, q_prod_state.index)],
                                    tma_bar_ptr=q_bar,
                                    mcast_mask=b_mcast_mask,
                                )
                                # Step 5.4: SF_Q TMA load — under same q_bar.
                                cute.copy(
                                    tma_atom_sf_q,
                                    tSF_Q_gSF_Q[(None, prefetch_next)],
                                    tSF_Q_sSF_Q[(None, q_prod_state.index)],
                                    tma_bar_ptr=q_bar,
                                )
                                cute.copy(
                                    tma_atom_w,
                                    tWgW[(None, prefetch_next)],
                                    tWsW[(None, q_prod_state.index)],
                                    tma_bar_ptr=q_bar,
                                )
                                q_prod_state.advance()

                if cutlass.const_expr(self.dynamic_sched):
                    left = left - cutlass.Int32(1)
                    if left == trig:
                        # claim ahead (at most two probes) and publish; then Q
                        # for the entry after the current one if it is visible
                        k_it = cutlass.Int32(0)
                        go = cutlass.Int32(1)
                        while go == cutlass.Int32(1):
                            need = self._fetch_step(
                                ring,
                                ring_ent,
                                s_P,
                                s_ctx,
                                s_rng,
                                s_ctl,
                                mScheduleMeta,
                                dyn_base,
                                batch_size,
                                sm_idx,
                                lane_idx,
                                ring_cons.count,
                                chunk_c,
                                tail_sh,
                            )
                            k_it = k_it + cutlass.Int32(1)
                            go = cutlass.Int32(0)
                            if need == cutlass.Int32(1) and k_it < cutlass.Int32(2):
                                go = cutlass.Int32(1)
                        if s_ctl[16] > ring_cons.count:
                            nb_e = ring_cons.index * 4
                            n_row = ring_ent[nb_e] & cutlass.Int32(0xFFFF)
                            n_cnt = ring_ent[nb_e + 2]
                            if n_cnt > cutlass.Int32(0) and n_row != q_row:
                                self._issue_q(
                                    q_pipeline,
                                    q_prod_state,
                                    tma_atom_b,
                                    tBgB,
                                    tBsB,
                                    b_mcast_mask,
                                    tma_atom_sf_q,
                                    tSF_Q_gSF_Q,
                                    tSF_Q_sSF_Q,
                                    tma_atom_w,
                                    tWgW,
                                    tWsW,
                                    n_row,
                                )
                                q_prod_state.advance()
                                q_row = n_row
                    if left > cutlass.Int32(0):
                        next_kv_idx = kv_idx + NUM_MATH_WG
                    else:
                        # segment end: publish (claim or terminal) until an entry is visible, then pop it
                        empty = s_ctl[16] == ring_cons.count
                        while empty:
                            self._fetch_step(
                                ring,
                                ring_ent,
                                s_P,
                                s_ctx,
                                s_rng,
                                s_ctl,
                                mScheduleMeta,
                                dyn_base,
                                batch_size,
                                sm_idx,
                                lane_idx,
                                ring_cons.count,
                                chunk_c,
                                tail_sh,
                            )
                            empty = s_ctl[16] == ring_cons.count
                        re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                        ring_cons.advance()
                        next_q_idx = re0 & cutlass.Int32(0xFFFF)
                        next_kv_idx = re1
                        left = re2
                        next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                        has_work = re2 > cutlass.Int32(0)
                        kv_blk_ptr = cutlass.Int32(32)
                        trig = left - (chunk_c + chunk_c)
                        if trig < (left >> 1):
                            trig = left >> 1
                        if lane_idx == cutlass.Int32(0):
                            s_ctl[10] = s_ctl[10] + (re0 >> 16)
                        cute.arch.sync_warp()
                        if has_work and next_q_idx != q_row:
                            self._issue_q(
                                q_pipeline,
                                q_prod_state,
                                tma_atom_b,
                                tBgB,
                                tBsB,
                                b_mcast_mask,
                                tma_atom_sf_q,
                                tSF_Q_gSF_Q,
                                tSF_Q_sSF_Q,
                                tma_atom_w,
                                tWgW,
                                tWsW,
                                next_q_idx,
                            )
                            q_prod_state.advance()
                            q_row = next_q_idx
                        if has_work and s_ctl[16] > ring_cons.count:
                            nb_e = ring_cons.index * 4
                            n_row = ring_ent[nb_e] & cutlass.Int32(0xFFFF)
                            n_cnt = ring_ent[nb_e + 2]
                            if n_cnt > cutlass.Int32(0) and n_row != q_row:
                                self._issue_q(
                                    q_pipeline,
                                    q_prod_state,
                                    tma_atom_b,
                                    tBgB,
                                    tBsB,
                                    b_mcast_mask,
                                    tma_atom_sf_q,
                                    tSF_Q_gSF_Q,
                                    tSF_Q_sSF_Q,
                                    tma_atom_w,
                                    tWgW,
                                    tWsW,
                                    n_row,
                                )
                                q_prod_state.advance()
                                q_row = n_row
                else:
                    # Advance: inline fetch_next_task
                    if q_idx_old == batch_size:
                        num_kv = (mContextLens[q_idx] + block_kv_val - 1) // block_kv_val
                        next_num_kv = num_kv
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        kv_blk_ptr = cutlass.Int32(32)
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_tma_warp_1:
            # TMA warp 1: loads KV + Scale for group 1 only
            cute.arch.warpgroup_reg_dealloc(self.prod_regs)
            next_kv_idx = s_sched[0]
            next_q_idx = s_sched[1]
            end_kv_idx = s_sched[2]
            end_q_idx = s_sched[3]
            lane_idx = tidx % 32
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
            blk_row = min(next_q_idx, batch_size - 1)
            # num_kv of the first tile is derived in the advance step, after its
            # KV burst (a load here would be consumed before the first TMA issue).
            next_num_kv = cutlass.Int32(0)

            # Block table prefetch for group 1
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use

            if cutlass.const_expr(self.dynamic_sched):
                re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                ring_cons.advance()
                next_q_idx = re0 & cutlass.Int32(0xFFFF)
                next_kv_idx = re1
                left = re2
                next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                has_work = re2 > cutlass.Int32(0)
            else:
                # First tile pair's page indices, issued at role entry.
                kv_blk_ptr = cutlass.Int32(0)
                pf_base = (next_kv_idx + 1 + lane_idx * NUM_MATH_WG) * NUM_BLOCKS_PER_MMA
                pf_last = cute.size(mBlockTable, mode=[1]) - 1
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cached_blks[i] = mBlockTable[(blk_row, min(pf_base + i, pf_last))]

            while has_work:
                # fetch_next_task: commit next → current
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                if cutlass.const_expr(self.dynamic_sched):
                    # New q_idx → force block table re-prefetch
                    if q_idx != q_idx_old:
                        kv_blk_ptr = cutlass.Int32(32)

                # Block table prefetch for group 1
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + 1 + lane_idx * NUM_MATH_WG
                    if prefetch_kv < num_kv:
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[(q_idx, base_phys + i)]
                    else:
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = cutlass.Int32(0)

                # Get block indices via shuffle before barrier
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1

                # Load KV + Scale for group 1: num_blocks_per_mma TMAs per tile.
                kv_pipeline_1.producer_acquire(kv_prod_state_1)
                bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                stage = kv_prod_state_1.index
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cute.copy(
                        tma_atom_a,
                        tAgA_1[(None, 0, 0, phys_blks[i])],
                        tAsA_1[(None, i, stage)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sf_kv,
                        tSgSF_KV_1[(None, 0, phys_blks[i])],
                        tSsSF_KV_1[(None, i, stage)],
                        tma_bar_ptr=bar,
                    )
                kv_prod_state_1.advance()

                if cutlass.const_expr(self.dynamic_sched):
                    left = left - cutlass.Int32(1)
                    if left > cutlass.Int32(0):
                        next_kv_idx = kv_idx + NUM_MATH_WG
                    else:
                        re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                        ring_cons.advance()
                        next_q_idx = re0 & cutlass.Int32(0xFFFF)
                        next_kv_idx = re1
                        left = re2
                        next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                        has_work = re2 > cutlass.Int32(0)
                        kv_blk_ptr = cutlass.Int32(32)
                else:
                    # Advance: inline fetch_next_task
                    if q_idx_old == batch_size:
                        num_kv = (mContextLens[q_idx] + block_kv_val - 1) // block_kv_val
                        next_num_kv = num_kv
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_idx = q_idx + 1
                        next_kv_idx = 0
                        kv_blk_ptr = cutlass.Int32(32)
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                    # Update while-loop condition
                    has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_umma_warp_0:
            # UMMA warp for group 0
            # Must wait on Q pipeline: TMA operations with different
            # barriers are NOT visibility-ordered even within the same
            # warp. KV0 barrier arriving does not guarantee Q SMEM
            # writes are visible.
            cute.arch.warpgroup_reg_dealloc(self.prod_regs)
            next_kv_idx = s_sched[0]
            next_q_idx = s_sched[1]
            end_kv_idx = s_sched[2]
            end_q_idx = s_sched[3]
            lane_idx = tidx % 32
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
            next_num_kv = (
                mContextLens[min(next_q_idx, batch_size - 1)] + block_kv_val - 1
            ) // block_kv_val

            # TMEM: wait for math warp 0's allocation, retrieve pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)

            # Step 5.6: SF TMEM tensors. Per-WG SFA region (group 0 owns
            # cols [sf_base..sf_base+num_sfa_tmem_cols)). SFB is shared (Q SF)
            # and starts after BOTH groups' SFA regions.
            sf_base_offset = self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + sf_base_offset, dtype=self.sf_dtype)
            sfb_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols * self.num_groups,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)

            if is_leader_cta:
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_stage_0 = cutlass.Int32(0)

                if cutlass.const_expr(self.dynamic_sched):
                    re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                    ring_cons.advance()
                    next_q_idx = re0 & cutlass.Int32(0xFFFF)
                    next_kv_idx = re1
                    left = re2
                    next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                    has_work = re2 > cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Wait for Q pipeline when batch changes
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index

                        # Step 5.6: New Q stage → re-issue UTCCP for SF Q.
                        # SMEM Q is N_padded tokens; loop over UTCCP atoms (128
                        # tokens each).
                        sf_q_atoms = self.N_padded // 128
                        for atom_idx in cutlass.range_constexpr(sf_q_atoms):
                            atom_offset = atom_idx * 128
                            stage_offset = q_stage_0 * sSF_Q.layout.stride[1]
                            utccp_required_smem_warp_transpose(
                                sSF_Q.iterator + stage_offset + atom_offset
                            )
                        cute.arch.fence_view_async_shared()
                        # UTCCP atom is UE8M0-typed; the int32 SMEM (4 packed
                        # UE8M0 per int32) needs a recast for the s2t copy +
                        # chunk layout (which counts UE8M0 cells, 4× int32).
                        sSF_Q_ue8m0 = cute.recast_tensor(sSF_Q, Float8E8M0FNU)
                        stage_off_ue8m0 = q_stage_0 * sSF_Q_ue8m0.layout.stride[1]
                        sSF_Q_chunk = cute.make_tensor(
                            sSF_Q_ue8m0.iterator + stage_off_ue8m0,
                            sfb_chunk_smem_layout,
                        )
                        tCsSFB_compact = cute.filter_zeros(sSF_Q_chunk)
                        tCtSFB_compact = cute.filter_zeros(tCtSFB)
                        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
                        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                        tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
                        tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                            tiled_copy_s2t_sfb, tCsSFB_s2t_
                        )
                        tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t)
                        # Make SFB TMEM write visible to umma_warp_1 before its
                        # MMA reads the same SFB region. fence orders the async
                        # s2t; barrier crosses the warp boundary.
                        cute.arch.fence_view_async_tmem_store()
                        sfb_sync_barrier.arrive_and_wait()

                    # Process KV block for group 0 (kv_idx + 0)
                    # Unconditional UMMA: OOB iterations
                    # compute on garbage data; results written to aligned
                    # padding region in logits buffer.
                    # Wait KV first, then TMEM empty
                    kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                    umma_pipeline_0.producer_acquire(umma_prod_state_0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage = kv_cons_state_umma_0.index

                    # Step 5.6: SF KV transpose + UTCCP. block_kv = 128 = 1
                    # UTCCP atom; loop is constexpr-1 but kept for clarity.
                    # When remove_online_sf_transpose=True, the host has already
                    # pre-arranged GMEM SF into UTCCP chunk layout, so the
                    # in-kernel SMEM transpose (and its fence) can be skipped.
                    if cutlass.const_expr(not self.remove_online_sf_transpose):
                        sf_kv_atoms = self.block_kv // 128
                        for atom_idx in cutlass.range_constexpr(sf_kv_atoms):
                            atom_offset = atom_idx * 128
                            stage_offset = kv_stage * sSF_KV_0.layout.stride[1]
                            utccp_required_smem_warp_transpose(
                                sSF_KV_0.iterator + stage_offset + atom_offset
                            )
                        cute.arch.fence_view_async_shared()
                    # int32 SMEM → UE8M0 view for UTCCP atom + chunk layout.
                    sSF_KV_0_ue8m0 = cute.recast_tensor(sSF_KV_0, Float8E8M0FNU)
                    stage_off_kv0_ue8m0 = kv_stage * sSF_KV_0_ue8m0.layout.stride[1]
                    sSF_KV_0_chunk = cute.make_tensor(
                        sSF_KV_0_ue8m0.iterator + stage_off_kv0_ue8m0,
                        sfa_chunk_smem_layout,
                    )
                    tCsSFA_compact = cute.filter_zeros(sSF_KV_0_chunk)
                    tCtSFA_compact = cute.filter_zeros(tCtSFA)
                    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
                    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
                    tCsSFA_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
                    tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, tCsSFA_s2t_)
                    tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t)

                    tCtAcc_0 = tCtAcc_base_0[(None, None, None, umma_prod_state_0.index)]
                    # Rank-4 slice of tmem SFA/SFB selects one K-instruction's
                    # SF region. tmem_sf*_layout shape is
                    # ((MMA_M_atom),MMA,MMA_K,STAGE) (STAGE is the synthetic
                    # size-1 appended in __init__). Mode 2 is MMA_K, whose size
                    # is version-dependent (CFK-29747) — sf_k_step computed in
                    # __init__ absorbs the difference.
                    # 4.4.x → sf_k_step=2 (slice at 0,2); 4.5.0+/internal → 1 (slice at 0,1).
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        tCtSFA_k = cute.slice_(tCtSFA, (None, None, k_block * self.sf_k_step, None))
                        tCtSFB_k = cute.slice_(tCtSFB, (None, None, k_block * self.sf_k_step, None))
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_0,
                            tCrA_0[None, None, k_block, kv_stage],
                            tCrB[None, None, k_block, q_stage_0],
                            tCtAcc_0,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # Step 5.6: UMMA owns the KV+SF release (was Math WG in FP8).
                    kv_pipeline_0.consumer_release(kv_cons_state_umma_0)
                    kv_cons_state_umma_0.advance()

                    umma_pipeline_0.producer_commit(umma_prod_state_0)
                    umma_prod_state_0.advance()

                    # Per-iter sync with umma_warp_1: SFB TMEM is a single
                    # region (no staging). Without this, warp 0 can race
                    # ahead and overwrite SFB at the next q transition while
                    # warp 1's previous-batch MMA is still reading the old
                    # SFB. DeepGEMM avoids this implicitly via single-warp
                    # ordering. arrive_and_wait here lock-steps the two UMMA
                    # warps every tile so the next transition's s2t cannot
                    # land before warp 1's previous MMA has committed.
                    sfb_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(self.dynamic_sched):
                        left = left - cutlass.Int32(1)
                        if left > cutlass.Int32(0):
                            next_kv_idx = kv_idx + NUM_MATH_WG
                        else:
                            re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                            ring_cons.advance()
                            next_q_idx = re0 & cutlass.Int32(0xFFFF)
                            next_kv_idx = re1
                            left = re2
                            next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                            has_work = re2 > cutlass.Int32(0)
                    else:
                        # Advance: inline fetch_next_task
                        next_kv_idx = kv_idx + NUM_MATH_WG
                        if next_kv_idx >= num_kv:
                            next_q_idx = q_idx + 1
                            next_kv_idx = 0
                            if next_q_idx < batch_size:
                                next_num_kv = (
                                    mContextLens[next_q_idx] + block_kv_val - 1
                                ) // block_kv_val
                        # Update while-loop condition
                        has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
        elif is_umma_warp_1:
            # UMMA warp for group 1
            # Explicitly waits on Q pipeline — critical because TMA warp 1
            # only loads KV1, not Q. Without this wait, UMMA warp 1 can
            # start GEMM before TMA warp 0 finishes loading Q into SMEM.
            cute.arch.warpgroup_reg_dealloc(self.prod_regs)
            next_kv_idx = s_sched[0]
            next_q_idx = s_sched[1]
            end_kv_idx = s_sched[2]
            end_q_idx = s_sched[3]
            lane_idx = tidx % 32
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
            next_num_kv = (
                mContextLens[min(next_q_idx, batch_size - 1)] + block_kv_val - 1
            ) // block_kv_val

            # TMEM: wait for umma_warp_0's allocation, retrieve pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)

            # Step 5.6: SF TMEM tensors. Group 1 owns SFA cols
            # [sf_base + num_sfa_tmem_cols, sf_base + 2*num_sfa_tmem_cols).
            # SFB is shared (Q SF); warp 0 issued the SFB UTCCP for Q and
            # warp 1 must NOT re-issue (would double-write).
            sf_base_offset = self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            sfa_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            sfb_tmem_ptr = cute.recast_ptr(
                tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols * self.num_groups,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)

            if is_leader_cta:
                num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                q_stage_1 = cutlass.Int32(0)

                if cutlass.const_expr(self.dynamic_sched):
                    re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                    ring_cons.advance()
                    next_q_idx = re0 & cutlass.Int32(0xFFFF)
                    next_kv_idx = re1
                    left = re2
                    next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                    has_work = re2 > cutlass.Int32(0)

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Wait for Q pipeline when batch changes
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index
                        # Step 5.6: UMMA warp 1 does NOT re-issue UTCCP_SF_Q —
                        # warp 0 owns it (same Q, same SFB TMEM region). But
                        # consumer_wait only orders TMA→SMEM Q; warp 0's s2t to
                        # TMEM SFB is a separate cross-warp dependency. Sync
                        # with warp 0 so its SFB write is visible before our
                        # MMA reads it.
                        sfb_sync_barrier.arrive_and_wait()

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional UMMA
                    # Wait KV first, then TMEM empty
                    kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                    umma_pipeline_1.producer_acquire(umma_prod_state_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage_1 = kv_cons_state_umma_1.index

                    # Step 5.6: SF KV (group 1) transpose + UTCCP.
                    # When remove_online_sf_transpose=True, the host has already
                    # pre-arranged GMEM SF into UTCCP chunk layout, so the
                    # in-kernel SMEM transpose (and its fence) can be skipped.
                    if cutlass.const_expr(not self.remove_online_sf_transpose):
                        sf_kv_atoms_1 = self.block_kv // 128
                        for atom_idx in cutlass.range_constexpr(sf_kv_atoms_1):
                            atom_offset = atom_idx * 128
                            stage_offset = kv_stage_1 * sSF_KV_1.layout.stride[1]
                            utccp_required_smem_warp_transpose(
                                sSF_KV_1.iterator + stage_offset + atom_offset
                            )
                        cute.arch.fence_view_async_shared()
                    sSF_KV_1_ue8m0 = cute.recast_tensor(sSF_KV_1, Float8E8M0FNU)
                    stage_off_kv1_ue8m0 = kv_stage_1 * sSF_KV_1_ue8m0.layout.stride[1]
                    sSF_KV_1_chunk = cute.make_tensor(
                        sSF_KV_1_ue8m0.iterator + stage_off_kv1_ue8m0,
                        sfa_chunk_smem_layout,
                    )
                    tCsSFA_compact_1 = cute.filter_zeros(sSF_KV_1_chunk)
                    tCtSFA_compact_1 = cute.filter_zeros(tCtSFA)
                    tiled_copy_s2t_sfa_1 = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact_1)
                    thr_copy_s2t_sfa_1 = tiled_copy_s2t_sfa_1.get_slice(0)
                    tCsSFA_s2t_1_ = thr_copy_s2t_sfa_1.partition_S(tCsSFA_compact_1)
                    tCsSFA_s2t_1 = tcgen05.get_s2t_smem_desc_tensor(
                        tiled_copy_s2t_sfa_1, tCsSFA_s2t_1_
                    )
                    tCtSFA_s2t_1 = thr_copy_s2t_sfa_1.partition_D(tCtSFA_compact_1)
                    cute.copy(tiled_copy_s2t_sfa_1, tCsSFA_s2t_1, tCtSFA_s2t_1)

                    tCtAcc_1 = tCtAcc_base_1[(None, None, None, umma_prod_state_1.index)]
                    # Rank-4 slice mode 2 at k_block*sf_k_step per K-instr.
                    # See umma_warp_0 for sf_k_step rationale.
                    for k_block in cutlass.range_constexpr(num_k_blocks_1):
                        tCtSFA_k = cute.slice_(tCtSFA, (None, None, k_block * self.sf_k_step, None))
                        tCtSFB_k = cute.slice_(tCtSFB, (None, None, k_block * self.sf_k_step, None))
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_1,
                            tCrA_1[None, None, k_block, kv_stage_1],
                            tCrB[None, None, k_block, q_stage_1],
                            tCtAcc_1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # Step 5.6: UMMA owns the KV+SF release.
                    kv_pipeline_1.consumer_release(kv_cons_state_umma_1)
                    kv_cons_state_umma_1.advance()

                    umma_pipeline_1.producer_commit(umma_prod_state_1)
                    umma_prod_state_1.advance()

                    # Per-iter sync with umma_warp_0 — see umma_warp_0 for
                    # rationale. Lock-steps the two UMMA warps every tile so
                    # warp 0 cannot overwrite SFB while warp 1's MMA is still
                    # reading it.
                    sfb_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(self.dynamic_sched):
                        left = left - cutlass.Int32(1)
                        if left > cutlass.Int32(0):
                            next_kv_idx = kv_idx + NUM_MATH_WG
                        else:
                            re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                            ring_cons.advance()
                            next_q_idx = re0 & cutlass.Int32(0xFFFF)
                            next_kv_idx = re1
                            left = re2
                            next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                            has_work = re2 > cutlass.Int32(0)
                    else:
                        # Advance: inline fetch_next_task
                        next_kv_idx = kv_idx + NUM_MATH_WG
                        if next_kv_idx >= num_kv:
                            next_q_idx = q_idx + 1
                            next_kv_idx = 0
                            if next_q_idx < batch_size:
                                next_num_kv = (
                                    mContextLens[next_q_idx] + block_kv_val - 1
                                ) // block_kv_val
                        # Update while-loop condition
                        has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_regs)
            lane_idx = tidx % 32
            next_kv_idx = s_sched[0]
            next_q_idx = s_sched[1]
            end_kv_idx = s_sched[2]
            end_q_idx = s_sched[3]
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
            next_num_kv = (
                mContextLens[min(next_q_idx, batch_size - 1)] + block_kv_val - 1
            ) // block_kv_val

            # TMEM: allocated by math warp 0 in the prologue; wait + retrieve
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)

            local_tidx = tidx % 128
            cC = cute.make_identity_tensor(epi_sub_mn)

            if warpgroup_idx == 0:
                # Math WG 0: process group 0
                # Reference setup (stage 0) for m_coord
                # flat_divide by sub-tile to get sub-tile partitions
                tAcc_0_ref = tCtAcc_base_0[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_0_ref_epi = cute.flat_divide(tAcc_0_ref, epi_sub_mn)
                tiled_copy_ref_0 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_0_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_0 = tiled_copy_ref_0.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_0.partition_D(cC)
                m_coord = tTR_cC[0][0]

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                # Step 5.8: MAX_NUM_W_IN_REG. Both fp16 and bf16 use 2-byte
                # weights (pack 2 per 32-bit reg) — same budget. Only fp32
                # differs. (`!= Float32` because const_expr `in tuple` behavior
                # is unverified across DSL versions.)
                # fp32 values determined empirically by SASS spill check
                # (cuobjdump --dump-sass | grep LDL/STL) with 240-reg math
                # warpgroup: next_n=1,2 fit 64 weights/slot with 0 spill;
                # next_n=3 spills at 60/64, max safe is 56. Must be a
                # multiple of 4 to match the packed-FMA (h_g, h_g+1, h_g+2,
                # h_g+3) layout below.
                if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                    # 2-byte weights (fp16 or bf16): next_n in {1,2,3} all fit
                    MAX_NUM_W_IN_REG = 64
                else:  # fp32, 4-byte weights
                    MAX_NUM_W_IN_REG = 56 if next_n == 3 else 64
                if cutlass.const_expr(self.emit_block_meta and next_n > 1):
                    # Free ~8 registers for the meta accumulators/fragments;
                    # the epilogue's weight cache sits at the spill edge for
                    # next_n >= 2. next_n == 1 has the headroom and keeps the
                    # full cache.
                    MAX_NUM_W_IN_REG = MAX_NUM_W_IN_REG - 8
                    if cutlass.const_expr(self.emit_hit_stats):
                        # Hit accumulators + bitmap word add ~6 more live
                        # registers across the tile loop.
                        MAX_NUM_W_IN_REG = MAX_NUM_W_IN_REG - 8
                    # emit_seed_counts needs no extra budget cut.
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_rmem_tensor(NUM_W_IN_REG * next_n, self.epi_dtype)
                # Batched STG: hold reduced result per t in register; the
                # actual STG happens once after the for-t loop to land all
                # STGs in one contiguous LSU phase.
                if cutlass.const_expr(self.use_batched_store):
                    result_arr = cute.make_rmem_tensor(next_n, self.output_dtype)
                else:
                    result_arr = None
                q_stage_local = cutlass.Int32(0)
                # loop-carried: the deferred logits / block_max flushes read
                # the last tile's position after the loop
                kv_pos = cutlass.Int32(0)
                # deferred logits store (plain build): the previous tile's
                # values are stored under the next tile's first TMEM load
                lg_pend = cute.make_rmem_tensor(next_n, self.output_dtype)
                lg_row = cutlass.Int32(0)
                lg_pos = cutlass.Int32(0)
                lg_on = cutlass.Int32(0)
                if cutlass.const_expr(self.emit_block_meta):
                    ctx_cur = cutlass.Int32(0)
                    meta_warp = local_tidx // 32
                    meta_lane = local_tidx % 32
                    # block_max row base (bytes) + record pitch: hoisted per q so
                    # the per-tile store is one address add instead of a full
                    # tensor-index chain (LDC + 64-bit IMAD/LEA) on the
                    # epilogue's critical path
                    bm_base = cutlass.Int64(0)
                    bm_nrec = cutlass.Int32(mBlockMax.shape[1])
                    if cutlass.const_expr(self.emit_block_meta_deferred):
                        # pending emission of the previous tile: per-lane
                        # values (masked only at the row's last tile) and the
                        # record address (row base + warp*4 + tile*16)
                        pend_v = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        for _i in cutlass.range_constexpr(next_n):
                            pend_v[_i] = cutlass.Float32(_META_NEG_FLT_MAX)
                        pend_addr = cutlass.Int64(0)
                        bm_base_w = cutlass.Int64(0)
                        # 16 B tile pitch kept opaque to the compiler (bm_nrec
                        # >= 0) so the arm lowers to one wide multiply-add
                        bm_pitch = cutlass.Int32(16) + (bm_nrec >> cutlass.Int32(31))
                    if cutlass.const_expr(self.emit_seed_counts):
                        sthr = cute.make_rmem_tensor(next_n * 3, cutlass.Float32)
                        scnt = cute.make_rmem_tensor(next_n * 3, cutlass.Int32)
                        spass = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        for _i in cutlass.range_constexpr(next_n):
                            spass[_i] = cutlass.Int32(0)
                        for _i in cutlass.range_constexpr(next_n * 3):
                            sthr[_i] = cutlass.Float32(_META_FLT_MAX)
                            scnt[_i] = cutlass.Int32(0)
                    if cutlass.const_expr(self.emit_cand or self.emit_cand_bucketed):
                        cwbase = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        cwleft = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        for _i in cutlass.range_constexpr(next_n):
                            cwbase[_i] = cutlass.Int32(0)
                            cwleft[_i] = cutlass.Int32(0)
                    if cutlass.const_expr(self.emit_hit_stats):
                        # Per-lane hit accumulators, carried across all
                        # tiles of the same q and flushed once per
                        # q-transition — no warp-wide ops per tile.
                        hacc_min = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        hacc_max = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        hacc_sum = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        hacc_cnt = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        for _t in cutlass.range_constexpr(next_n):
                            hacc_min[_t] = cutlass.Float32(_META_FLT_MAX)
                            hacc_max[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            hacc_sum[_t] = cutlass.Float32(0.0)
                            hacc_cnt[_t] = cutlass.Int32(0)
                        # Batched bitmap read state: all 32 lanes of a warp
                        # need the SAME word per tile, so lane l loads the
                        # word for tile j+l once per 32 tiles and each tile
                        # takes its word via one shuffle.
                        meta_j = cutlass.Int32(0)
                        hitw_batch = cutlass.Int32(0)

                if cutlass.const_expr(self.dynamic_sched):
                    re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                    ring_cons.advance()
                    next_q_idx = re0 & cutlass.Int32(0xFFFF)
                    next_kv_idx = re1
                    left = re2
                    next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                    has_work = re2 > cutlass.Int32(0)
                if cutlass.const_expr(self.emit_block_meta_deferred):
                    # pre-arm to this warp's first record: the first flush
                    # stores -FLT_MAX there and the same lanes overwrite it
                    # with the true value one tile later (no armed flag)
                    pend_addr = (
                        mBlockMax.iterator.toint()
                        + cutlass.Int64(next_q_idx * next_n)
                        * cutlass.Int64(bm_nrec)
                        * cutlass.Int64(4)
                        + cutlass.Int64((next_kv_idx) * cutlass.Int32(4) + meta_warp)
                        * cutlass.Int64(4)
                    )

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Q pipeline consumer: wait for Q+Weights SMEM
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        # Preload first NUM_W_IN_REG weights per slot
                        for t_i in cutlass.range_constexpr(next_n):
                            for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                                if cutlass.const_expr(self.epi_dtype == cutlass.Float32):
                                    # cached weights carry the 0.5 of (x + |x|) / 2
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                        (t_i * num_heads + w_j, q_stage_local)
                                    ] * cutlass.Float32(0.5)
                                else:
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                        (t_i * num_heads + w_j, q_stage_local)
                                    ]
                        if cutlass.const_expr(self.emit_block_meta):
                            # Flush the PREVIOUS request's hit accumulators
                            # before switching context.
                            if cutlass.const_expr(self.emit_hit_stats):
                                if q_idx_old < batch_size:
                                    self._flush_hit_agg(
                                        mHitAgg,
                                        q_idx_old,
                                        hacc_min,
                                        hacc_max,
                                        hacc_sum,
                                        hacc_cnt,
                                        meta_lane,
                                    )
                                # New bitmap row: invalidate the batched
                                # word cache (forces a reload).
                                meta_j = cutlass.Int32(0)
                            # Compressed-space context len; the meta valid
                            # mask (kv_pos < ctx_cur) keeps GEMM garbage in
                            # the aligned padding region out of block_max.
                            if cutlass.const_expr(self.emit_seed_counts):
                                if q_idx_old < batch_size:
                                    self._flush_seed_counts(
                                        mSeedCounts,
                                        q_idx_old,
                                        scnt,
                                        meta_lane,
                                        spass=spass,
                                        cand_ctl=mCandCtl,
                                    )
                                # (re)load this q's thresholds - gated on
                                # emit_seed_counts, NOT emit_cand: counts-
                                # only mode needs them too (a stale
                                # FLT_MAX default zeroes every counter)
                                for _t in cutlass.range_constexpr(next_n):
                                    for _j in cutlass.range_constexpr(3):
                                        sthr[_t * 3 + _j] = mSeedThr[(q_idx * next_n + _t, _j)]
                            if cutlass.const_expr(self.emit_cand):
                                if q_idx_old < batch_size:
                                    self._flush_cand_window(
                                        mCand, q_idx_old, cwbase, cwleft, meta_lane
                                    )
                            if cutlass.const_expr(self.emit_cand_bucketed):
                                if q_idx_old < batch_size:
                                    self._flush_cand_window_bucketed(
                                        mCand, mCandIdx, q_idx_old, cwbase, cwleft, meta_lane
                                    )
                            ctx_cur = mContextLens[q_idx]
                            bm_base = mBlockMax.iterator.toint() + cutlass.Int64(
                                q_idx * next_n
                            ) * cutlass.Int64(bm_nrec) * cutlass.Int64(4)
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                bm_base_w = bm_base + cutlass.Int64(meta_warp * cutlass.Int32(4))

                    # Process KV block for group 0 (kv_idx + 0)
                    # Unconditional Math: OOB results
                    # written to aligned padding region in logits buffer.
                    kv_pos = kv_idx * block_kv_val + m_coord
                    if cutlass.const_expr(self.emit_block_meta):
                        meta_kv_tile = kv_idx
                        if cutlass.const_expr(not self.emit_block_meta_deferred):
                            meta_valid = kv_pos < ctx_cur
                        if cutlass.const_expr(self.emit_hit_stats):
                            # Warp-uniform reload once per 32 tiles: this
                            # WG's tile at counter j+l is kv_tile + 2*l,
                            # whose warp word index is (kv_tile+2*l)*4 +
                            # warp. Clamp keeps end-of-row lanes in
                            # bounds (their tiles are never consumed).
                            if (meta_j & cutlass.Int32(31)) == cutlass.Int32(0):
                                w_idx = (
                                    meta_kv_tile + cutlass.Int32(2) * meta_lane
                                ) * cutlass.Int32(4) + meta_warp
                                w_idx = min(w_idx, mHitBitmap.shape[1] - cutlass.Int32(1))
                                hitw_batch = mHitBitmap[(q_idx, w_idx)]
                            hit_word = cute.arch.shuffle_sync(
                                hitw_batch, meta_j & cutlass.Int32(31)
                            )
                            meta_j = meta_j + cutlass.Int32(1)

                    # Step 5.7: drop kv_pipeline.consumer_wait/release and
                    # scale_val LDS — UMMA owns KV+SF pipe; SF is baked into
                    # acc by block-scaled MMA.
                    umma_pipeline_0.consumer_wait(umma_cons_state_0)

                    # --- TMEM sub-tile setup ---
                    # flat_divide accumulator by sub-tile shape;
                    # partition once, then loop over sub-tiles.
                    tCtAcc_c0 = tCtAcc_base_0[(None, None, None, umma_cons_state_0.index)]
                    tAcc_c0 = tCtAcc_c0[((None, None), 0, 0)]
                    tAcc_c0_epi = cute.flat_divide(tAcc_c0, epi_sub_mn)
                    tc_0 = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_c0_epi[(None, None, 0, 0)])
                    tr_0 = tc_0.get_slice(local_tidx)
                    tTR_0 = tr_0.partition_S(tAcc_c0_epi)

                    # --- First sub-tile LDTM ---
                    cute.copy(tc_0, tTR_0[(None, None, None, 0, 0)], tTR_rAcc)
                    if cutlass.const_expr(self.defer_logits):
                        if lg_on != cutlass.Int32(0):
                            for _t in cutlass.range_constexpr(next_n):
                                mLogits[(lg_row + _t, lg_pos)] = lg_pend[_t]
                    if cutlass.const_expr(self.emit_block_meta_deferred):
                        # previous tile's block_max under this tile's TMEM
                        # load; all 32 lanes store the warp-uniform value
                        for _t in cutlass.range_constexpr(next_n):
                            r_pend = cute.arch.warp_redux_sync(pend_v[_t], "fmax")
                            _st_global_f32(
                                pend_addr
                                + cutlass.Int64(_t) * cutlass.Int64(bm_nrec) * cutlass.Int64(4),
                                r_pend,
                            )
                        # arm this tile: record = tile*4 + warp -> 16 B per tile
                        pend_addr = cutlass.Int64(
                            cutlass.Uint64(bm_base_w)
                            + cutlass.Uint64(cutlass.Uint32(meta_kv_tile))
                            * cutlass.Uint64(cutlass.Uint32(bm_pitch))
                        )
                    cute.arch.fence_view_async_tmem_load()

                    # --- Sub-tile compute loop ---
                    # Each sub-tile: LDTM.xN → fence → load →
                    # ReLU+FMA. Breaks FMA chain (16→4 per chunk)
                    # and interleaves LDTM with FP32 compute to
                    # reduce ShadowPipeThrottle.
                    subtile_n = num_heads // num_epi_subtiles
                    # Step 5.9: packed_zero needed for both fp16 and bf16
                    # paths; pre-compute once outside loop.
                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                        packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
                    elif cutlass.const_expr(self.epi_dtype == cutlass.BFloat16):
                        packed_zero = pack_bf16x2(BFloat16(0.0), BFloat16(0.0))
                    for t in cutlass.range_constexpr(next_n):
                        # Step 5.9: !=Float32 catches both fp16 and bf16
                        if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                            ps0 = packed_zero
                            ps1 = packed_zero
                        else:
                            s0x = cutlass.Float32(0.0)
                            s0y = cutlass.Float32(0.0)
                            s1x = cutlass.Float32(0.0)
                            s1y = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            # LDTM for sub-tiles 1..N-1
                            # (sub-tile 0 handled above)
                            if t > 0 or i > 0:
                                cute.copy(
                                    tc_0,
                                    tTR_0[(None, None, None, 0, t * num_epi_subtiles + i)],
                                    tTR_rAcc,
                                )
                                cute.arch.fence_view_async_tmem_load()
                            # Release UMMA after last LDTM+fence
                            if t == next_n - 1 and i == num_epi_subtiles - 1:
                                umma_pipeline_0.consumer_release(umma_cons_state_0)
                                umma_cons_state_0.advance()
                            acc_vec = tTR_rAcc.load()
                            # Reg-path: weights from registers
                            reg_h_end = min(subtile_n, max(0, NUM_W_IN_REG - i * subtile_n))
                            for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                n0 = h
                                h_g = i * subtile_n + h
                                # Step 5.9: packed path catches fp16 & bf16
                                if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                        pa01 = pack_f16x2(
                                            Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_f16x2(
                                            Float16(acc_vec[n0 + 2]), Float16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_f16x2(pa01, packed_zero)
                                        pa23 = max_f16x2(pa23, packed_zero)
                                        r0 = t * NUM_W_IN_REG + h_g
                                        pw01 = pack_f16x2(w_cache[r0], w_cache[r0 + 1])
                                        pw23 = pack_f16x2(w_cache[r0 + 2], w_cache[r0 + 3])
                                        ps0 = fma_f16x2(pa01, pw01, ps0)
                                        ps1 = fma_f16x2(pa23, pw23, ps1)
                                    else:  # bf16
                                        pa01 = pack_bf16x2(
                                            BFloat16(acc_vec[n0]), BFloat16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_bf16x2(
                                            BFloat16(acc_vec[n0 + 2]), BFloat16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_bf16x2(pa01, packed_zero)
                                        pa23 = max_bf16x2(pa23, packed_zero)
                                        r0 = t * NUM_W_IN_REG + h_g
                                        pw01 = pack_bf16x2(w_cache[r0], w_cache[r0 + 1])
                                        pw23 = pack_bf16x2(w_cache[r0 + 2], w_cache[r0 + 3])
                                        ps0 = fma_bf16x2(pa01, pw01, ps0)
                                        ps1 = fma_bf16x2(pa23, pw23, ps1)
                                else:
                                    # relu(x) * w == (x + |x|) * (w / 2), bit-exact in fp32
                                    x0 = acc_vec[n0]
                                    x1 = acc_vec[n0 + 1]
                                    x2 = acc_vec[n0 + 2]
                                    x3 = acc_vec[n0 + 3]
                                    a0, a1 = cute.arch.add_packed_f32x2(
                                        (x0, x1), (_fabs_f32(x0), _fabs_f32(x1))
                                    )
                                    a2, a3 = cute.arch.add_packed_f32x2(
                                        (x2, x3), (_fabs_f32(x2), _fabs_f32(x3))
                                    )
                                    r0 = t * NUM_W_IN_REG + h_g
                                    w0 = w_cache[r0]
                                    w1 = w_cache[r0 + 1]
                                    w2 = w_cache[r0 + 2]
                                    w3 = w_cache[r0 + 3]
                                    s0x, s0y = cute.arch.fma_packed_f32x2(
                                        (a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN
                                    )
                                    s1x, s1y = cute.arch.fma_packed_f32x2(
                                        (a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN
                                    )
                            # SMEM-path: weights from shared mem
                            smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
                            for h in cutlass.range_constexpr(smem_h_start, subtile_n, 4):
                                n0 = h
                                h_g = i * subtile_n + h
                                # Step 5.9: packed path catches fp16 & bf16
                                if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                        pa01 = pack_f16x2(
                                            Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_f16x2(
                                            Float16(acc_vec[n0 + 2]), Float16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_f16x2(pa01, packed_zero)
                                        pa23 = max_f16x2(pa23, packed_zero)
                                        pw01 = pack_f16x2(
                                            sW[(t * num_heads + h_g, q_stage_local)],
                                            sW[(t * num_heads + h_g + 1, q_stage_local)],
                                        )
                                        pw23 = pack_f16x2(
                                            sW[(t * num_heads + h_g + 2, q_stage_local)],
                                            sW[(t * num_heads + h_g + 3, q_stage_local)],
                                        )
                                        ps0 = fma_f16x2(pa01, pw01, ps0)
                                        ps1 = fma_f16x2(pa23, pw23, ps1)
                                    else:  # bf16
                                        pa01 = pack_bf16x2(
                                            BFloat16(acc_vec[n0]), BFloat16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_bf16x2(
                                            BFloat16(acc_vec[n0 + 2]), BFloat16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_bf16x2(pa01, packed_zero)
                                        pa23 = max_bf16x2(pa23, packed_zero)
                                        pw01 = pack_bf16x2(
                                            sW[(t * num_heads + h_g, q_stage_local)],
                                            sW[(t * num_heads + h_g + 1, q_stage_local)],
                                        )
                                        pw23 = pack_bf16x2(
                                            sW[(t * num_heads + h_g + 2, q_stage_local)],
                                            sW[(t * num_heads + h_g + 3, q_stage_local)],
                                        )
                                        ps0 = fma_bf16x2(pa01, pw01, ps0)
                                        ps1 = fma_bf16x2(pa23, pw23, ps1)
                                else:
                                    a0 = cutlass.max(acc_vec[n0], cutlass.Float32(0.0))
                                    a1 = cutlass.max(acc_vec[n0 + 1], cutlass.Float32(0.0))
                                    a2 = cutlass.max(acc_vec[n0 + 2], cutlass.Float32(0.0))
                                    a3 = cutlass.max(acc_vec[n0 + 3], cutlass.Float32(0.0))
                                    w0 = sW[(t * num_heads + h_g, q_stage_local)]
                                    w1 = sW[(t * num_heads + h_g + 1, q_stage_local)]
                                    w2 = sW[(t * num_heads + h_g + 2, q_stage_local)]
                                    w3 = sW[(t * num_heads + h_g + 3, q_stage_local)]
                                    s0x, s0y = cute.arch.fma_packed_f32x2(
                                        (a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN
                                    )
                                    s1x, s1y = cute.arch.fma_packed_f32x2(
                                        (a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN
                                    )
                        # Step 5.9: result reduction — packed path catches both
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps_sum = add_f16x2(ps0, ps1)
                            sum_lo, sum_hi = unpack_f16x2(ps_sum)
                            result_t = sum_lo + sum_hi
                        elif cutlass.const_expr(self.epi_dtype == cutlass.BFloat16):
                            ps_sum = add_bf16x2(ps0, ps1)
                            sum_lo, sum_hi = unpack_bf16x2(ps_sum)
                            result_t = sum_lo + sum_hi
                        else:
                            result_t = s0x + s0y + s1x + s1y
                        # Step 5.7: drop * scale_val (FP4 SF baked into acc).
                        stored_t = self.output_dtype(result_t)
                        if cutlass.const_expr(self.use_batched_store):
                            result_arr[t] = stored_t
                        elif cutlass.const_expr(self.defer_logits):
                            lg_pend[t] = stored_t
                        else:
                            out_row = q_idx * next_n + t
                            mLogits[(out_row, kv_pos)] = stored_t
                        if cutlass.const_expr(self.emit_block_meta):
                            # Meta reduction on the POST-conversion value so
                            # block_max bounds what GVR reads back bit-exactly.
                            f32_t = cutlass.Float32(stored_t)
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                # unmasked here: only the row's last tile pair
                                # can straddle ctx and it is always the last
                                # tile of a static row / dynamic run
                                pend_v[t] = f32_t
                            else:
                                bmax_v = cutlass.Float32(_META_NEG_FLT_MAX)
                                if meta_valid:
                                    bmax_v = f32_t
                                r_bmax = cute.arch.warp_redux_sync(bmax_v, "fmax")
                                # Warp-autonomous store: record index =
                                # tile*4 + warp; the GVR consumer folds the
                                # 4 warp-partials per block.
                                if meta_lane == cutlass.Int32(0):
                                    rec_m = (
                                        cutlass.Int32(t) * bm_nrec
                                        + meta_kv_tile * cutlass.Int32(4)
                                        + meta_warp
                                    )
                                    _st_global_f32(
                                        bm_base + cutlass.Int64(rec_m) * cutlass.Int64(4), r_bmax
                                    )
                            if cutlass.const_expr(self.emit_seed_counts):
                                # Seed-count accumulation: branchless 0/1
                                # adds on the post-conversion value; the
                                # valid mask keeps aligned-padding garbage
                                # out (same contract as block_max).
                                valid_i1 = cutlass.Int32(meta_valid)
                                for _j in cutlass.range_constexpr(3):
                                    ge_j = cutlass.Int32(f32_t >= sthr[t * 3 + _j])
                                    scnt[t * 3 + _j] = scnt[t * 3 + _j] + (ge_j & valid_i1)
                                if cutlass.const_expr(self.seed_packed):
                                    # adaptive-skip pass count: one record
                                    # per (tile, warp); r_bmax is warp-
                                    # uniform so lane0 alone accumulates
                                    if meta_lane == cutlass.Int32(0):
                                        spass[t] = spass[t] + cutlass.Int32(
                                            r_bmax >= sthr[t * 3 + 0]
                                        )
                            if cutlass.const_expr(self.emit_cand):
                                # Candidate pre-collect at t_0 with per-warp
                                # claim windows: one atomic claims
                                # (hits + CAND_WIN) slots per refill.
                                # Unconsumed tail is sentinel-filled on
                                # flush; counts[r][0] stays the exact count.
                                # Gated on the warp-uniform 32-position
                                # bound: r_bmax < t_0 proves zero hits;
                                # bound >= t_0 guarantees a nonzero ballot
                                # (exact per-lane max, invalid -> -FLT_MAX).
                                if r_bmax >= sthr[t * 3 + 0]:
                                    pred_c = cutlass.Int32(0)
                                    if meta_valid:
                                        if f32_t >= sthr[t * 3 + 0]:
                                            pred_c = cutlass.Int32(1)
                                    mask_c = cute.arch.vote_ballot_sync(pred_c != cutlass.Int32(0))
                                    row_c = q_idx * next_n + t
                                    cnt_c = cutlass.Int32(cute.arch.popc(mask_c))
                                    lm_c = (
                                        cutlass.Uint32(1) << cutlass.Uint32(meta_lane)
                                    ) - cutlass.Uint32(1)
                                    off_c = cutlass.Int32(cute.arch.popc(mask_c & lm_c))
                                    CAP_C = cutlass.const_expr(self.cand_cap)
                                    cand_b = mCand.iterator.toint()
                                    if cnt_c > cwleft[t]:
                                        # sentinel-fill the old tail, then
                                        # refill: one atomic per window.
                                        sl_o = cwbase[t] + meta_lane
                                        if meta_lane < cwleft[t] and sl_o < cutlass.Int32(CAP_C):
                                            pair_o = cand_b + (
                                                cutlass.Int64(row_c) * cutlass.Int64(CAP_C)
                                                + cutlass.Int64(sl_o)
                                            ) * cutlass.Int64(8)
                                            iptr_o = cute.make_ptr(
                                                cutlass.Int32,
                                                pair_o + cutlass.Int64(4),
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            cute.make_tensor(iptr_o, cute.make_layout((1,)))[0] = (
                                                cutlass.Int32(-1)
                                            )
                                        m_c = cnt_c + cutlass.Int32(self.CAND_WIN)
                                        ctl_addr = mCandCtl.iterator.toint() + (
                                            cutlass.Int64(row_c) * cutlass.Int64(8)
                                        )
                                        nb_c = cutlass.Int32(0)
                                        if meta_lane == cutlass.Int32(0):
                                            nb_c = _atom_global_add_s32(ctl_addr, m_c)
                                        nb_c = cute.arch.shuffle_sync(nb_c, cutlass.Int32(0))
                                        cwbase[t] = nb_c
                                        cwleft[t] = m_c
                                        # one-shot void mark on crossing CAP
                                        if meta_lane == cutlass.Int32(0):
                                            if nb_c + m_c > cutlass.Int32(
                                                CAP_C
                                            ) and nb_c <= cutlass.Int32(CAP_C):
                                                vdptr = cute.make_ptr(
                                                    cutlass.Int32,
                                                    ctl_addr + cutlass.Int64(4),
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=4,
                                                )
                                                cute.make_tensor(vdptr, cute.make_layout((1,)))[
                                                    0
                                                ] = cutlass.Int32(1)
                                    slot_c = cwbase[t] + off_c
                                    if pred_c != cutlass.Int32(0) and slot_c < cutlass.Int32(CAP_C):
                                        pair_addr = cand_b + (
                                            cutlass.Int64(row_c) * cutlass.Int64(CAP_C)
                                            + cutlass.Int64(slot_c)
                                        ) * cutlass.Int64(8)
                                        vptr_c = cute.make_ptr(
                                            cutlass.Float32,
                                            pair_addr,
                                            cute.AddressSpace.gmem,
                                            assumed_align=8,
                                        )
                                        cute.make_tensor(vptr_c, cute.make_layout((1,)))[0] = f32_t
                                        iptr_c = cute.make_ptr(
                                            cutlass.Int32,
                                            pair_addr + cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(iptr_c, cute.make_layout((1,)))[0] = kv_pos
                                    cwbase[t] = cwbase[t] + cnt_c
                                    cwleft[t] = cwleft[t] - cnt_c
                            if cutlass.const_expr(self.emit_cand_bucketed):
                                # Bucketed SoA: A/B EXACT ballot claims
                                # (their prefixes must stay pad-free for
                                # the consumer's prefix math), C keeps the
                                # claim-window; a full segment spills to
                                # the next looser one. Every warp
                                # collective sits at the TOP level of this
                                # warp-uniform bound gate - no collectives
                                # inside nested dynamic branches (DSL).
                                if r_bmax >= sthr[t * 3 + 0]:
                                    segA_k = cutlass.const_expr(self.accept_cap)
                                    capC_k = cutlass.const_expr(self.cand_cap)
                                    wtot_k = cutlass.const_expr(2 * self.accept_cap + self.cand_cap)
                                    row_k = q_idx * next_n + t
                                    vb_k = mCand.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(wtot_k) * cutlass.Int64(4)
                                    ib_k = mCandIdx.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(wtot_k) * cutlass.Int64(4)
                                    cur_k = mCandCur.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(16)
                                    ctl_k = mCandCtl.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(16)
                                    lmk_k = (
                                        cutlass.Uint32(1) << cutlass.Uint32(meta_lane)
                                    ) - cutlass.Uint32(1)
                                    # exclusive class predicates
                                    pA_k = cutlass.Int32(0)
                                    pB_k = cutlass.Int32(0)
                                    pC_k = cutlass.Int32(0)
                                    if meta_valid:
                                        if f32_t >= sthr[t * 3 + 2]:
                                            pA_k = cutlass.Int32(1)
                                        if f32_t >= sthr[t * 3 + 1] and pA_k == cutlass.Int32(0):
                                            pB_k = cutlass.Int32(1)
                                        if (
                                            f32_t >= sthr[t * 3 + 0]
                                            and pA_k == cutlass.Int32(0)
                                            and pB_k == cutlass.Int32(0)
                                        ):
                                            pC_k = cutlass.Int32(1)
                                    # ---- A: exact claim ----
                                    mA_k = cute.arch.vote_ballot_sync(pA_k != cutlass.Int32(0))
                                    cntA_k = cutlass.Int32(cute.arch.popc(mA_k))
                                    offA_k = cutlass.Int32(cute.arch.popc(mA_k & lmk_k))
                                    baseA_k = cutlass.Int32(0)
                                    if meta_lane == cutlass.Int32(0) and cntA_k > cutlass.Int32(0):
                                        baseA_k = _atom_global_add_s32(cur_k, cntA_k)
                                    baseA_k = cute.arch.shuffle_sync(baseA_k, cutlass.Int32(0))
                                    slotA_k = baseA_k + offA_k
                                    spA_k = cutlass.Int32(0)
                                    if pA_k != cutlass.Int32(0) and slotA_k >= cutlass.Int32(
                                        segA_k
                                    ):
                                        spA_k = cutlass.Int32(1)
                                    if pA_k != cutlass.Int32(0) and slotA_k < cutlass.Int32(segA_k):
                                        vp_k = cute.make_ptr(
                                            cutlass.Float32,
                                            vb_k + cutlass.Int64(slotA_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(vp_k, cute.make_layout((1,)))[0] = f32_t
                                        ip_k = cute.make_ptr(
                                            cutlass.Int32,
                                            ib_k + cutlass.Int64(slotA_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(ip_k, cute.make_layout((1,)))[0] = kv_pos
                                    # ---- B: exact claim (native + A spill) ----
                                    pBe_k = cutlass.Int32(0)
                                    if pB_k != cutlass.Int32(0) or spA_k != cutlass.Int32(0):
                                        pBe_k = cutlass.Int32(1)
                                    mB_k = cute.arch.vote_ballot_sync(pBe_k != cutlass.Int32(0))
                                    cntB_k = cutlass.Int32(cute.arch.popc(mB_k))
                                    offB_k = cutlass.Int32(cute.arch.popc(mB_k & lmk_k))
                                    baseB_k = cutlass.Int32(0)
                                    if meta_lane == cutlass.Int32(0) and cntB_k > cutlass.Int32(0):
                                        baseB_k = _atom_global_add_s32(
                                            cur_k + cutlass.Int64(4), cntB_k
                                        )
                                    baseB_k = cute.arch.shuffle_sync(baseB_k, cutlass.Int32(0))
                                    slotB_k = baseB_k + offB_k
                                    spB_k = cutlass.Int32(0)
                                    if pBe_k != cutlass.Int32(0) and slotB_k >= cutlass.Int32(
                                        segA_k
                                    ):
                                        spB_k = cutlass.Int32(1)
                                    if pBe_k != cutlass.Int32(0) and slotB_k < cutlass.Int32(
                                        segA_k
                                    ):
                                        vp2_k = cute.make_ptr(
                                            cutlass.Float32,
                                            vb_k
                                            + cutlass.Int64(segA_k + slotB_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(vp2_k, cute.make_layout((1,)))[0] = f32_t
                                        ip2_k = cute.make_ptr(
                                            cutlass.Int32,
                                            ib_k
                                            + cutlass.Int64(segA_k + slotB_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(ip2_k, cute.make_layout((1,)))[0] = kv_pos
                                    # n0 += exact placements in A and B
                                    plc_k = (
                                        cntA_k
                                        - cutlass.Int32(
                                            cute.arch.popc(
                                                cute.arch.vote_ballot_sync(
                                                    spA_k != cutlass.Int32(0)
                                                )
                                            )
                                        )
                                    ) + (
                                        cntB_k
                                        - cutlass.Int32(
                                            cute.arch.popc(
                                                cute.arch.vote_ballot_sync(
                                                    spB_k != cutlass.Int32(0)
                                                )
                                            )
                                        )
                                    )
                                    if meta_lane == cutlass.Int32(0) and plc_k > cutlass.Int32(0):
                                        _atom_global_add_s32(ctl_k, plc_k)
                                    # ---- C: claim window (native + B spill) ----
                                    pCe_k = cutlass.Int32(0)
                                    if pC_k != cutlass.Int32(0) or spB_k != cutlass.Int32(0):
                                        pCe_k = cutlass.Int32(1)
                                    mC_k = cute.arch.vote_ballot_sync(pCe_k != cutlass.Int32(0))
                                    cntC_k = cutlass.Int32(cute.arch.popc(mC_k))
                                    offC_k = cutlass.Int32(cute.arch.popc(mC_k & lmk_k))
                                    if cntC_k > cwleft[t]:
                                        # sentinel-fill the old window tail
                                        # (BOTH columns: the consumer pads
                                        # by score -inf, idx -1)
                                        slo_k = cwbase[t] + meta_lane
                                        if meta_lane < cwleft[t] and slo_k < cutlass.Int32(capC_k):
                                            vpo_k = cute.make_ptr(
                                                cutlass.Float32,
                                                vb_k
                                                + cutlass.Int64(2 * segA_k + slo_k)
                                                * cutlass.Int64(4),
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            cute.make_tensor(vpo_k, cute.make_layout((1,)))[0] = (
                                                cutlass.Float32(_META_NEG_FLT_MAX)
                                            )
                                            ipo_k = cute.make_ptr(
                                                cutlass.Int32,
                                                ib_k
                                                + cutlass.Int64(2 * segA_k + slo_k)
                                                * cutlass.Int64(4),
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            cute.make_tensor(ipo_k, cute.make_layout((1,)))[0] = (
                                                cutlass.Int32(-1)
                                            )
                                        mC2_k = cntC_k + cutlass.Int32(self.CAND_WIN)
                                        nbC_k = cutlass.Int32(0)
                                        if meta_lane == cutlass.Int32(0):
                                            nbC_k = _atom_global_add_s32(
                                                cur_k + cutlass.Int64(8), mC2_k
                                            )
                                            _atom_global_add_s32(ctl_k, mC2_k)
                                            if nbC_k + mC2_k > cutlass.Int32(
                                                capC_k
                                            ) and nbC_k <= cutlass.Int32(capC_k):
                                                vdp_k = cute.make_ptr(
                                                    cutlass.Int32,
                                                    ctl_k + cutlass.Int64(4),
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=4,
                                                )
                                                cute.make_tensor(vdp_k, cute.make_layout((1,)))[
                                                    0
                                                ] = cutlass.Int32(1)
                                        nbC_k = cute.arch.shuffle_sync(nbC_k, cutlass.Int32(0))
                                        cwbase[t] = nbC_k
                                        cwleft[t] = mC2_k
                                    slotC_k = cwbase[t] + offC_k
                                    if pCe_k != cutlass.Int32(0) and slotC_k < cutlass.Int32(
                                        capC_k
                                    ):
                                        vpc_k = cute.make_ptr(
                                            cutlass.Float32,
                                            vb_k
                                            + cutlass.Int64(2 * segA_k + slotC_k)
                                            * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(vpc_k, cute.make_layout((1,)))[0] = f32_t
                                        ipc_k = cute.make_ptr(
                                            cutlass.Int32,
                                            ib_k
                                            + cutlass.Int64(2 * segA_k + slotC_k)
                                            * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(ipc_k, cute.make_layout((1,)))[0] = kv_pos
                                    cwbase[t] = cwbase[t] + cntC_k
                                    cwleft[t] = cwleft[t] - cntC_k
                            if cutlass.const_expr(self.emit_hit_stats):
                                # Lane-local accumulation; keep it branchless
                                # (`if meta_hit` compiles to real divergent
                                # branches). Bit-mask select is NaN-safe for
                                # OOB-tile garbage logits. No valid-mask
                                # needed: the bitmap contract only sets bits
                                # inside [0, ctx).
                                meta_hit = (
                                    hit_word >> (kv_pos & cutlass.Int32(31))
                                ) & cutlass.Int32(1)
                                msk = cutlass.Int32(0) - meta_hit  # 0 / ~0
                                inv = cutlass.Int32(-1) - msk
                                fbits = cutlass.Int32(
                                    llvm.bitcast(cutlass.Int32.mlir_type, f32_t.ir_value())
                                )
                                # bits(+FLT_MAX)=0x7F7FFFFF,
                                # bits(-FLT_MAX)=0xFF7FFFFF (as i32: neg).
                                selmin = cutlass.Float32(
                                    llvm.bitcast(
                                        cutlass.Float32.mlir_type,
                                        (
                                            (fbits & msk) | (cutlass.Int32(0x7F7FFFFF) & inv)
                                        ).ir_value(),
                                    )
                                )
                                selmax = cutlass.Float32(
                                    llvm.bitcast(
                                        cutlass.Float32.mlir_type,
                                        (
                                            (fbits & msk) | (cutlass.Int32(-8388609) & inv)
                                        ).ir_value(),
                                    )
                                )
                                seladd = cutlass.Float32(
                                    llvm.bitcast(
                                        cutlass.Float32.mlir_type, (fbits & msk).ir_value()
                                    )
                                )
                                hacc_min[t] = cutlass.min(hacc_min[t], selmin)
                                hacc_max[t] = cutlass.max(hacc_max[t], selmax)
                                hacc_sum[t] = hacc_sum[t] + seladd
                                hacc_cnt[t] = hacc_cnt[t] + meta_hit

                    if cutlass.const_expr(self.use_batched_store and self.defer_logits):
                        for t in cutlass.range_constexpr(next_n):
                            lg_pend[t] = result_arr[t]
                    elif cutlass.const_expr(self.use_batched_store):
                        # Batched STG: all result_arr[t] → mLogits in one pass.
                        for t in cutlass.range_constexpr(next_n):
                            out_row = q_idx * next_n + t
                            mLogits[(out_row, kv_pos)] = result_arr[t]
                    if cutlass.const_expr(self.defer_logits):
                        lg_row = q_idx * next_n
                        lg_pos = kv_pos
                        lg_on = cutlass.Int32(1)

                    if cutlass.const_expr(self.dynamic_sched):
                        left = left - cutlass.Int32(1)
                        if left > cutlass.Int32(0):
                            next_kv_idx = kv_idx + NUM_MATH_WG
                        else:
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                # run end: the row's last tile (or WG1's OOB
                                # tile) may hold lanes >= ctx; mask before flush
                                for _t in cutlass.range_constexpr(next_n):
                                    if kv_pos >= ctx_cur:
                                        pend_v[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                            ring_cons.advance()
                            next_q_idx = re0 & cutlass.Int32(0xFFFF)
                            next_kv_idx = re1
                            left = re2
                            next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                            has_work = re2 > cutlass.Int32(0)
                            if cutlass.const_expr(self.emit_block_meta and self.emit_hit_stats):
                                meta_j = cutlass.Int32(0)
                    else:
                        # Advance: inline fetch_next_task
                        next_kv_idx = kv_idx + NUM_MATH_WG
                        if next_kv_idx >= num_kv:
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                # row end: see the dynamic branch
                                for _t in cutlass.range_constexpr(next_n):
                                    if kv_pos >= ctx_cur:
                                        pend_v[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            next_q_idx = q_idx + 1
                            next_kv_idx = 0
                            if next_q_idx < batch_size:
                                next_num_kv = (
                                    mContextLens[next_q_idx] + block_kv_val - 1
                                ) // block_kv_val
                        # Update while-loop condition
                        has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

                if cutlass.const_expr(self.pdl and self.pdl_trigger == 2):
                    # past this CTA's last work tile
                    griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.defer_logits):
                    if lg_on != cutlass.Int32(0):
                        for _t in cutlass.range_constexpr(next_n):
                            mLogits[(lg_row + _t, lg_pos)] = lg_pend[_t]
                if cutlass.const_expr(self.emit_block_meta_deferred):
                    # flush the last tile's pending block_max (WG 0); a
                    # segment ending mid-row skipped the tail mask, so mask here
                    if q_idx < batch_size:
                        for _t in cutlass.range_constexpr(next_n):
                            if kv_pos >= ctx_cur:
                                pend_v[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            r_pend = cute.arch.warp_redux_sync(pend_v[_t], "fmax")
                            _st_global_f32(
                                pend_addr
                                + cutlass.Int64(_t) * cutlass.Int64(bm_nrec) * cutlass.Int64(4),
                                r_pend,
                            )
                # Flush the final request's hit accumulators (WG 0).
                if cutlass.const_expr(self.emit_block_meta and self.emit_hit_stats):
                    if q_idx < batch_size:
                        self._flush_hit_agg(
                            mHitAgg, q_idx, hacc_min, hacc_max, hacc_sum, hacc_cnt, meta_lane
                        )
                if cutlass.const_expr(self.emit_seed_counts):
                    if q_idx < batch_size:
                        self._flush_seed_counts(
                            mSeedCounts, q_idx, scnt, meta_lane, spass=spass, cand_ctl=mCandCtl
                        )
                if cutlass.const_expr(self.emit_cand):
                    if q_idx < batch_size:
                        self._flush_cand_window(mCand, q_idx, cwbase, cwleft, meta_lane)
                if cutlass.const_expr(self.emit_cand_bucketed):
                    if q_idx < batch_size:
                        self._flush_cand_window_bucketed(
                            mCand, mCandIdx, q_idx, cwbase, cwleft, meta_lane
                        )

                # Release last Q stage (WG 0)
                if q_idx < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            else:
                # Math WG 1: process group 1
                tAcc_1_ref = tCtAcc_base_1[(None, None, None, 0)][((None, None), 0, 0)]
                tAcc_1_ref_epi = cute.flat_divide(tAcc_1_ref, epi_sub_mn)
                tiled_copy_ref_1 = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_1_ref_epi[(None, None, 0, 0)]
                )
                thr_copy_ref_1 = tiled_copy_ref_1.get_slice(local_tidx)
                tTR_cC = thr_copy_ref_1.partition_D(cC)
                m_coord = tTR_cC[0][0]

                tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

                # Step 5.8: see WG 0 for rationale.
                if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                    MAX_NUM_W_IN_REG = 64
                else:
                    MAX_NUM_W_IN_REG = 56 if next_n == 3 else 64
                if cutlass.const_expr(self.emit_block_meta and next_n > 1):
                    # Free ~8 registers for the meta accumulators/fragments;
                    # the epilogue's weight cache sits at the spill edge for
                    # next_n >= 2. next_n == 1 has the headroom and keeps the
                    # full cache.
                    MAX_NUM_W_IN_REG = MAX_NUM_W_IN_REG - 8
                    if cutlass.const_expr(self.emit_hit_stats):
                        # Hit accumulators + bitmap word add ~6 more live
                        # registers across the tile loop.
                        MAX_NUM_W_IN_REG = MAX_NUM_W_IN_REG - 8
                    # emit_seed_counts needs no extra budget cut.
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_rmem_tensor(NUM_W_IN_REG * next_n, self.epi_dtype)
                # Batched STG: hold reduced result per t in register; the
                # actual STG happens once after the for-t loop to land all
                # STGs in one contiguous LSU phase.
                if cutlass.const_expr(self.use_batched_store):
                    result_arr = cute.make_rmem_tensor(next_n, self.output_dtype)
                else:
                    result_arr = None
                q_stage_local = cutlass.Int32(0)
                # loop-carried: the deferred logits / block_max flushes read
                # the last tile's position after the loop
                kv_pos = cutlass.Int32(0)
                # deferred logits store (plain build): the previous tile's
                # values are stored under the next tile's first TMEM load
                lg_pend = cute.make_rmem_tensor(next_n, self.output_dtype)
                lg_row = cutlass.Int32(0)
                lg_pos = cutlass.Int32(0)
                lg_on = cutlass.Int32(0)
                if cutlass.const_expr(self.emit_block_meta):
                    ctx_cur = cutlass.Int32(0)
                    meta_warp = local_tidx // 32
                    meta_lane = local_tidx % 32
                    # block_max row base (bytes) + record pitch: hoisted per q so
                    # the per-tile store is one address add instead of a full
                    # tensor-index chain (LDC + 64-bit IMAD/LEA) on the
                    # epilogue's critical path
                    bm_base = cutlass.Int64(0)
                    bm_nrec = cutlass.Int32(mBlockMax.shape[1])
                    if cutlass.const_expr(self.emit_block_meta_deferred):
                        # pending emission of the previous tile: per-lane
                        # values (masked only at the row's last tile) and the
                        # record address (row base + warp*4 + tile*16)
                        pend_v = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        for _i in cutlass.range_constexpr(next_n):
                            pend_v[_i] = cutlass.Float32(_META_NEG_FLT_MAX)
                        pend_addr = cutlass.Int64(0)
                        bm_base_w = cutlass.Int64(0)
                        # 16 B tile pitch kept opaque to the compiler (bm_nrec
                        # >= 0) so the arm lowers to one wide multiply-add
                        bm_pitch = cutlass.Int32(16) + (bm_nrec >> cutlass.Int32(31))
                    if cutlass.const_expr(self.emit_seed_counts):
                        sthr = cute.make_rmem_tensor(next_n * 3, cutlass.Float32)
                        scnt = cute.make_rmem_tensor(next_n * 3, cutlass.Int32)
                        spass = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        for _i in cutlass.range_constexpr(next_n):
                            spass[_i] = cutlass.Int32(0)
                        for _i in cutlass.range_constexpr(next_n * 3):
                            sthr[_i] = cutlass.Float32(_META_FLT_MAX)
                            scnt[_i] = cutlass.Int32(0)
                    if cutlass.const_expr(self.emit_cand or self.emit_cand_bucketed):
                        cwbase = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        cwleft = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        for _i in cutlass.range_constexpr(next_n):
                            cwbase[_i] = cutlass.Int32(0)
                            cwleft[_i] = cutlass.Int32(0)
                    if cutlass.const_expr(self.emit_hit_stats):
                        # Per-lane hit accumulators, carried across all
                        # tiles of the same q and flushed once per
                        # q-transition — no warp-wide ops per tile.
                        hacc_min = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        hacc_max = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        hacc_sum = cute.make_rmem_tensor(next_n, cutlass.Float32)
                        hacc_cnt = cute.make_rmem_tensor(next_n, cutlass.Int32)
                        for _t in cutlass.range_constexpr(next_n):
                            hacc_min[_t] = cutlass.Float32(_META_FLT_MAX)
                            hacc_max[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            hacc_sum[_t] = cutlass.Float32(0.0)
                            hacc_cnt[_t] = cutlass.Int32(0)
                        # Batched bitmap read state: all 32 lanes of a warp
                        # need the SAME word per tile, so lane l loads the
                        # word for tile j+l once per 32 tiles and each tile
                        # takes its word via one shuffle.
                        meta_j = cutlass.Int32(0)
                        hitw_batch = cutlass.Int32(0)

                if cutlass.const_expr(self.dynamic_sched):
                    re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                    ring_cons.advance()
                    next_q_idx = re0 & cutlass.Int32(0xFFFF)
                    next_kv_idx = re1
                    left = re2
                    next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                    has_work = re2 > cutlass.Int32(0)
                if cutlass.const_expr(self.emit_block_meta_deferred):
                    # pre-arm to this warp's first record: the first flush
                    # stores -FLT_MAX there and the same lanes overwrite it
                    # with the true value one tile later (no armed flag)
                    pend_addr = (
                        mBlockMax.iterator.toint()
                        + cutlass.Int64(next_q_idx * next_n)
                        * cutlass.Int64(bm_nrec)
                        * cutlass.Int64(4)
                        + cutlass.Int64(
                            (next_kv_idx + cutlass.Int32(1)) * cutlass.Int32(4) + meta_warp
                        )
                        * cutlass.Int64(4)
                    )

                while has_work:
                    # fetch_next_task: commit next → current
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv

                    # Q pipeline consumer: wait for Q+Weights SMEM
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        # Preload first NUM_W_IN_REG weights per slot
                        for t_i in cutlass.range_constexpr(next_n):
                            for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                                if cutlass.const_expr(self.epi_dtype == cutlass.Float32):
                                    # cached weights carry the 0.5 of (x + |x|) / 2
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                        (t_i * num_heads + w_j, q_stage_local)
                                    ] * cutlass.Float32(0.5)
                                else:
                                    w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                        (t_i * num_heads + w_j, q_stage_local)
                                    ]
                        if cutlass.const_expr(self.emit_block_meta):
                            # Flush the PREVIOUS request's hit accumulators
                            # before switching context.
                            if cutlass.const_expr(self.emit_hit_stats):
                                if q_idx_old < batch_size:
                                    self._flush_hit_agg(
                                        mHitAgg,
                                        q_idx_old,
                                        hacc_min,
                                        hacc_max,
                                        hacc_sum,
                                        hacc_cnt,
                                        meta_lane,
                                    )
                                # New bitmap row: invalidate the batched
                                # word cache (forces a reload).
                                meta_j = cutlass.Int32(0)
                            # Compressed-space context len; the meta valid
                            # mask (kv_pos < ctx_cur) keeps GEMM garbage in
                            # the aligned padding region out of block_max.
                            if cutlass.const_expr(self.emit_seed_counts):
                                if q_idx_old < batch_size:
                                    self._flush_seed_counts(
                                        mSeedCounts,
                                        q_idx_old,
                                        scnt,
                                        meta_lane,
                                        spass=spass,
                                        cand_ctl=mCandCtl,
                                    )
                                # (re)load this q's thresholds - gated on
                                # emit_seed_counts, NOT emit_cand (see the
                                # WG0 twin above)
                                for _t in cutlass.range_constexpr(next_n):
                                    for _j in cutlass.range_constexpr(3):
                                        sthr[_t * 3 + _j] = mSeedThr[(q_idx * next_n + _t, _j)]
                            if cutlass.const_expr(self.emit_cand):
                                if q_idx_old < batch_size:
                                    self._flush_cand_window(
                                        mCand, q_idx_old, cwbase, cwleft, meta_lane
                                    )
                            if cutlass.const_expr(self.emit_cand_bucketed):
                                if q_idx_old < batch_size:
                                    self._flush_cand_window_bucketed(
                                        mCand, mCandIdx, q_idx_old, cwbase, cwleft, meta_lane
                                    )
                            ctx_cur = mContextLens[q_idx]
                            bm_base = mBlockMax.iterator.toint() + cutlass.Int64(
                                q_idx * next_n
                            ) * cutlass.Int64(bm_nrec) * cutlass.Int64(4)
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                bm_base_w = bm_base + cutlass.Int64(meta_warp * cutlass.Int32(4))

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional Math
                    kv_idx_1 = kv_idx + 1

                    kv_pos = kv_idx_1 * block_kv_val + m_coord
                    if cutlass.const_expr(self.emit_block_meta):
                        meta_kv_tile = kv_idx_1
                        # See WG 0. For the odd-num_kv OOB tile
                        # (kv_idx_1 == num_kv) every lane has
                        # kv_pos >= ctx_cur, so identities land in the
                        # nb_pad padding slot — never read by GVR.
                        if cutlass.const_expr(not self.emit_block_meta_deferred):
                            meta_valid = kv_pos < ctx_cur
                        if cutlass.const_expr(self.emit_hit_stats):
                            # Warp-uniform reload once per 32 tiles: this
                            # WG's tile at counter j+l is kv_tile + 2*l,
                            # whose warp word index is (kv_tile+2*l)*4 +
                            # warp. Clamp keeps end-of-row lanes in
                            # bounds (their tiles are never consumed).
                            if (meta_j & cutlass.Int32(31)) == cutlass.Int32(0):
                                w_idx = (
                                    meta_kv_tile + cutlass.Int32(2) * meta_lane
                                ) * cutlass.Int32(4) + meta_warp
                                w_idx = min(w_idx, mHitBitmap.shape[1] - cutlass.Int32(1))
                                hitw_batch = mHitBitmap[(q_idx, w_idx)]
                            hit_word = cute.arch.shuffle_sync(
                                hitw_batch, meta_j & cutlass.Int32(31)
                            )
                            meta_j = meta_j + cutlass.Int32(1)

                    # Step 5.7: drop kv_pipeline.consumer_wait/release and
                    # scale_val LDS — UMMA owns KV+SF pipe.
                    umma_pipeline_1.consumer_wait(umma_cons_state_1)

                    # --- TMEM sub-tile setup (WG1) ---
                    tCtAcc_c1 = tCtAcc_base_1[(None, None, None, umma_cons_state_1.index)]
                    tAcc_c1 = tCtAcc_c1[((None, None), 0, 0)]
                    tAcc_c1_epi = cute.flat_divide(tAcc_c1, epi_sub_mn)
                    tc_1 = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_c1_epi[(None, None, 0, 0)])
                    tr_1 = tc_1.get_slice(local_tidx)
                    tTR_1 = tr_1.partition_S(tAcc_c1_epi)

                    # --- First sub-tile LDTM (WG1) ---
                    cute.copy(tc_1, tTR_1[(None, None, None, 0, 0)], tTR_rAcc)
                    if cutlass.const_expr(self.defer_logits):
                        if lg_on != cutlass.Int32(0):
                            for _t in cutlass.range_constexpr(next_n):
                                mLogits[(lg_row + _t, lg_pos)] = lg_pend[_t]
                    if cutlass.const_expr(self.emit_block_meta_deferred):
                        # previous tile's block_max under this tile's TMEM
                        # load; all 32 lanes store the warp-uniform value
                        for _t in cutlass.range_constexpr(next_n):
                            r_pend = cute.arch.warp_redux_sync(pend_v[_t], "fmax")
                            _st_global_f32(
                                pend_addr
                                + cutlass.Int64(_t) * cutlass.Int64(bm_nrec) * cutlass.Int64(4),
                                r_pend,
                            )
                        # arm this tile: record = tile*4 + warp -> 16 B per tile
                        pend_addr = cutlass.Int64(
                            cutlass.Uint64(bm_base_w)
                            + cutlass.Uint64(cutlass.Uint32(meta_kv_tile))
                            * cutlass.Uint64(cutlass.Uint32(bm_pitch))
                        )
                    cute.arch.fence_view_async_tmem_load()

                    # --- Sub-tile compute loop (WG1) ---
                    subtile_n = num_heads // num_epi_subtiles
                    # Step 5.9: packed_zero pre-compute (fp16 or bf16)
                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                        packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
                    elif cutlass.const_expr(self.epi_dtype == cutlass.BFloat16):
                        packed_zero = pack_bf16x2(BFloat16(0.0), BFloat16(0.0))
                    for t in cutlass.range_constexpr(next_n):
                        # Step 5.9: !=Float32 catches both fp16 and bf16
                        if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                            ps0 = packed_zero
                            ps1 = packed_zero
                        else:
                            s0x = cutlass.Float32(0.0)
                            s0y = cutlass.Float32(0.0)
                            s1x = cutlass.Float32(0.0)
                            s1y = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(num_epi_subtiles):
                            if t > 0 or i > 0:
                                cute.copy(
                                    tc_1,
                                    tTR_1[(None, None, None, 0, t * num_epi_subtiles + i)],
                                    tTR_rAcc,
                                )
                                cute.arch.fence_view_async_tmem_load()
                            if t == next_n - 1 and i == num_epi_subtiles - 1:
                                umma_pipeline_1.consumer_release(umma_cons_state_1)
                                umma_cons_state_1.advance()
                            acc_vec = tTR_rAcc.load()
                            # Reg-path
                            reg_h_end = min(subtile_n, max(0, NUM_W_IN_REG - i * subtile_n))
                            for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                n0 = h
                                h_g = i * subtile_n + h
                                # Step 5.9: packed (fp16/bf16) vs fp32
                                if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                        pa01 = pack_f16x2(
                                            Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_f16x2(
                                            Float16(acc_vec[n0 + 2]), Float16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_f16x2(pa01, packed_zero)
                                        pa23 = max_f16x2(pa23, packed_zero)
                                        r0 = t * NUM_W_IN_REG + h_g
                                        pw01 = pack_f16x2(w_cache[r0], w_cache[r0 + 1])
                                        pw23 = pack_f16x2(w_cache[r0 + 2], w_cache[r0 + 3])
                                        ps0 = fma_f16x2(pa01, pw01, ps0)
                                        ps1 = fma_f16x2(pa23, pw23, ps1)
                                    else:  # bf16
                                        pa01 = pack_bf16x2(
                                            BFloat16(acc_vec[n0]), BFloat16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_bf16x2(
                                            BFloat16(acc_vec[n0 + 2]), BFloat16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_bf16x2(pa01, packed_zero)
                                        pa23 = max_bf16x2(pa23, packed_zero)
                                        r0 = t * NUM_W_IN_REG + h_g
                                        pw01 = pack_bf16x2(w_cache[r0], w_cache[r0 + 1])
                                        pw23 = pack_bf16x2(w_cache[r0 + 2], w_cache[r0 + 3])
                                        ps0 = fma_bf16x2(pa01, pw01, ps0)
                                        ps1 = fma_bf16x2(pa23, pw23, ps1)
                                else:
                                    # relu(x) * w == (x + |x|) * (w / 2), bit-exact in fp32
                                    x0 = acc_vec[n0]
                                    x1 = acc_vec[n0 + 1]
                                    x2 = acc_vec[n0 + 2]
                                    x3 = acc_vec[n0 + 3]
                                    a0, a1 = cute.arch.add_packed_f32x2(
                                        (x0, x1), (_fabs_f32(x0), _fabs_f32(x1))
                                    )
                                    a2, a3 = cute.arch.add_packed_f32x2(
                                        (x2, x3), (_fabs_f32(x2), _fabs_f32(x3))
                                    )
                                    r0 = t * NUM_W_IN_REG + h_g
                                    w0 = w_cache[r0]
                                    w1 = w_cache[r0 + 1]
                                    w2 = w_cache[r0 + 2]
                                    w3 = w_cache[r0 + 3]
                                    s0x, s0y = cute.arch.fma_packed_f32x2(
                                        (a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN
                                    )
                                    s1x, s1y = cute.arch.fma_packed_f32x2(
                                        (a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN
                                    )
                            # SMEM-path
                            smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
                            for h in cutlass.range_constexpr(smem_h_start, subtile_n, 4):
                                n0 = h
                                h_g = i * subtile_n + h
                                # Step 5.9: packed (fp16/bf16) vs fp32
                                if cutlass.const_expr(self.epi_dtype != cutlass.Float32):
                                    if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                                        pa01 = pack_f16x2(
                                            Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_f16x2(
                                            Float16(acc_vec[n0 + 2]), Float16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_f16x2(pa01, packed_zero)
                                        pa23 = max_f16x2(pa23, packed_zero)
                                        pw01 = pack_f16x2(
                                            sW[(t * num_heads + h_g, q_stage_local)],
                                            sW[(t * num_heads + h_g + 1, q_stage_local)],
                                        )
                                        pw23 = pack_f16x2(
                                            sW[(t * num_heads + h_g + 2, q_stage_local)],
                                            sW[(t * num_heads + h_g + 3, q_stage_local)],
                                        )
                                        ps0 = fma_f16x2(pa01, pw01, ps0)
                                        ps1 = fma_f16x2(pa23, pw23, ps1)
                                    else:  # bf16
                                        pa01 = pack_bf16x2(
                                            BFloat16(acc_vec[n0]), BFloat16(acc_vec[n0 + 1])
                                        )
                                        pa23 = pack_bf16x2(
                                            BFloat16(acc_vec[n0 + 2]), BFloat16(acc_vec[n0 + 3])
                                        )
                                        pa01 = max_bf16x2(pa01, packed_zero)
                                        pa23 = max_bf16x2(pa23, packed_zero)
                                        pw01 = pack_bf16x2(
                                            sW[(t * num_heads + h_g, q_stage_local)],
                                            sW[(t * num_heads + h_g + 1, q_stage_local)],
                                        )
                                        pw23 = pack_bf16x2(
                                            sW[(t * num_heads + h_g + 2, q_stage_local)],
                                            sW[(t * num_heads + h_g + 3, q_stage_local)],
                                        )
                                        ps0 = fma_bf16x2(pa01, pw01, ps0)
                                        ps1 = fma_bf16x2(pa23, pw23, ps1)
                                else:
                                    a0 = cutlass.max(acc_vec[n0], cutlass.Float32(0.0))
                                    a1 = cutlass.max(acc_vec[n0 + 1], cutlass.Float32(0.0))
                                    a2 = cutlass.max(acc_vec[n0 + 2], cutlass.Float32(0.0))
                                    a3 = cutlass.max(acc_vec[n0 + 3], cutlass.Float32(0.0))
                                    w0 = sW[(t * num_heads + h_g, q_stage_local)]
                                    w1 = sW[(t * num_heads + h_g + 1, q_stage_local)]
                                    w2 = sW[(t * num_heads + h_g + 2, q_stage_local)]
                                    w3 = sW[(t * num_heads + h_g + 3, q_stage_local)]
                                    s0x, s0y = cute.arch.fma_packed_f32x2(
                                        (a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN
                                    )
                                    s1x, s1y = cute.arch.fma_packed_f32x2(
                                        (a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN
                                    )
                        # Step 5.9: result reduction
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps_sum = add_f16x2(ps0, ps1)
                            sum_lo, sum_hi = unpack_f16x2(ps_sum)
                            result_t = sum_lo + sum_hi
                        elif cutlass.const_expr(self.epi_dtype == cutlass.BFloat16):
                            ps_sum = add_bf16x2(ps0, ps1)
                            sum_lo, sum_hi = unpack_bf16x2(ps_sum)
                            result_t = sum_lo + sum_hi
                        else:
                            result_t = s0x + s0y + s1x + s1y
                        # Step 5.7: drop * scale_val (FP4 SF baked into acc).
                        stored_t = self.output_dtype(result_t)
                        if cutlass.const_expr(self.use_batched_store):
                            result_arr[t] = stored_t
                        elif cutlass.const_expr(self.defer_logits):
                            lg_pend[t] = stored_t
                        else:
                            out_row = q_idx * next_n + t
                            mLogits[(out_row, kv_pos)] = stored_t
                        if cutlass.const_expr(self.emit_block_meta):
                            # Meta reduction on the POST-conversion value so
                            # block_max bounds what GVR reads back bit-exactly.
                            f32_t = cutlass.Float32(stored_t)
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                # unmasked here: only the row's last tile pair
                                # can straddle ctx and it is always the last
                                # tile of a static row / dynamic run
                                pend_v[t] = f32_t
                            else:
                                bmax_v = cutlass.Float32(_META_NEG_FLT_MAX)
                                if meta_valid:
                                    bmax_v = f32_t
                                r_bmax = cute.arch.warp_redux_sync(bmax_v, "fmax")
                                # Warp-autonomous store: record index =
                                # tile*4 + warp; the GVR consumer folds the
                                # 4 warp-partials per block.
                                if meta_lane == cutlass.Int32(0):
                                    rec_m = (
                                        cutlass.Int32(t) * bm_nrec
                                        + meta_kv_tile * cutlass.Int32(4)
                                        + meta_warp
                                    )
                                    _st_global_f32(
                                        bm_base + cutlass.Int64(rec_m) * cutlass.Int64(4), r_bmax
                                    )
                            if cutlass.const_expr(self.emit_seed_counts):
                                # Seed-count accumulation: branchless 0/1
                                # adds on the post-conversion value; the
                                # valid mask keeps aligned-padding garbage
                                # out (same contract as block_max).
                                valid_i1 = cutlass.Int32(meta_valid)
                                for _j in cutlass.range_constexpr(3):
                                    ge_j = cutlass.Int32(f32_t >= sthr[t * 3 + _j])
                                    scnt[t * 3 + _j] = scnt[t * 3 + _j] + (ge_j & valid_i1)
                                if cutlass.const_expr(self.seed_packed):
                                    # adaptive-skip pass count: one record
                                    # per (tile, warp); r_bmax is warp-
                                    # uniform so lane0 alone accumulates
                                    if meta_lane == cutlass.Int32(0):
                                        spass[t] = spass[t] + cutlass.Int32(
                                            r_bmax >= sthr[t * 3 + 0]
                                        )
                            if cutlass.const_expr(self.emit_cand):
                                # Candidate pre-collect at t_0 with per-warp
                                # claim windows: one atomic claims
                                # (hits + CAND_WIN) slots per refill.
                                # Unconsumed tail is sentinel-filled on
                                # flush; counts[r][0] stays the exact count.
                                # Gated on the warp-uniform 32-position
                                # bound: r_bmax < t_0 proves zero hits;
                                # bound >= t_0 guarantees a nonzero ballot
                                # (exact per-lane max, invalid -> -FLT_MAX).
                                if r_bmax >= sthr[t * 3 + 0]:
                                    pred_c = cutlass.Int32(0)
                                    if meta_valid:
                                        if f32_t >= sthr[t * 3 + 0]:
                                            pred_c = cutlass.Int32(1)
                                    mask_c = cute.arch.vote_ballot_sync(pred_c != cutlass.Int32(0))
                                    row_c = q_idx * next_n + t
                                    cnt_c = cutlass.Int32(cute.arch.popc(mask_c))
                                    lm_c = (
                                        cutlass.Uint32(1) << cutlass.Uint32(meta_lane)
                                    ) - cutlass.Uint32(1)
                                    off_c = cutlass.Int32(cute.arch.popc(mask_c & lm_c))
                                    CAP_C = cutlass.const_expr(self.cand_cap)
                                    cand_b = mCand.iterator.toint()
                                    if cnt_c > cwleft[t]:
                                        # sentinel-fill the old tail, then
                                        # refill: one atomic per window.
                                        sl_o = cwbase[t] + meta_lane
                                        if meta_lane < cwleft[t] and sl_o < cutlass.Int32(CAP_C):
                                            pair_o = cand_b + (
                                                cutlass.Int64(row_c) * cutlass.Int64(CAP_C)
                                                + cutlass.Int64(sl_o)
                                            ) * cutlass.Int64(8)
                                            iptr_o = cute.make_ptr(
                                                cutlass.Int32,
                                                pair_o + cutlass.Int64(4),
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            cute.make_tensor(iptr_o, cute.make_layout((1,)))[0] = (
                                                cutlass.Int32(-1)
                                            )
                                        m_c = cnt_c + cutlass.Int32(self.CAND_WIN)
                                        ctl_addr = mCandCtl.iterator.toint() + (
                                            cutlass.Int64(row_c) * cutlass.Int64(8)
                                        )
                                        nb_c = cutlass.Int32(0)
                                        if meta_lane == cutlass.Int32(0):
                                            nb_c = _atom_global_add_s32(ctl_addr, m_c)
                                        nb_c = cute.arch.shuffle_sync(nb_c, cutlass.Int32(0))
                                        cwbase[t] = nb_c
                                        cwleft[t] = m_c
                                        # one-shot void mark on crossing CAP
                                        if meta_lane == cutlass.Int32(0):
                                            if nb_c + m_c > cutlass.Int32(
                                                CAP_C
                                            ) and nb_c <= cutlass.Int32(CAP_C):
                                                vdptr = cute.make_ptr(
                                                    cutlass.Int32,
                                                    ctl_addr + cutlass.Int64(4),
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=4,
                                                )
                                                cute.make_tensor(vdptr, cute.make_layout((1,)))[
                                                    0
                                                ] = cutlass.Int32(1)
                                    slot_c = cwbase[t] + off_c
                                    if pred_c != cutlass.Int32(0) and slot_c < cutlass.Int32(CAP_C):
                                        pair_addr = cand_b + (
                                            cutlass.Int64(row_c) * cutlass.Int64(CAP_C)
                                            + cutlass.Int64(slot_c)
                                        ) * cutlass.Int64(8)
                                        vptr_c = cute.make_ptr(
                                            cutlass.Float32,
                                            pair_addr,
                                            cute.AddressSpace.gmem,
                                            assumed_align=8,
                                        )
                                        cute.make_tensor(vptr_c, cute.make_layout((1,)))[0] = f32_t
                                        iptr_c = cute.make_ptr(
                                            cutlass.Int32,
                                            pair_addr + cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(iptr_c, cute.make_layout((1,)))[0] = kv_pos
                                    cwbase[t] = cwbase[t] + cnt_c
                                    cwleft[t] = cwleft[t] - cnt_c
                            if cutlass.const_expr(self.emit_cand_bucketed):
                                # Bucketed SoA: A/B EXACT ballot claims
                                # (their prefixes must stay pad-free for
                                # the consumer's prefix math), C keeps the
                                # claim-window; a full segment spills to
                                # the next looser one. Every warp
                                # collective sits at the TOP level of this
                                # warp-uniform bound gate - no collectives
                                # inside nested dynamic branches (DSL).
                                if r_bmax >= sthr[t * 3 + 0]:
                                    segA_k = cutlass.const_expr(self.accept_cap)
                                    capC_k = cutlass.const_expr(self.cand_cap)
                                    wtot_k = cutlass.const_expr(2 * self.accept_cap + self.cand_cap)
                                    row_k = q_idx * next_n + t
                                    vb_k = mCand.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(wtot_k) * cutlass.Int64(4)
                                    ib_k = mCandIdx.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(wtot_k) * cutlass.Int64(4)
                                    cur_k = mCandCur.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(16)
                                    ctl_k = mCandCtl.iterator.toint() + cutlass.Int64(
                                        row_k
                                    ) * cutlass.Int64(16)
                                    lmk_k = (
                                        cutlass.Uint32(1) << cutlass.Uint32(meta_lane)
                                    ) - cutlass.Uint32(1)
                                    # exclusive class predicates
                                    pA_k = cutlass.Int32(0)
                                    pB_k = cutlass.Int32(0)
                                    pC_k = cutlass.Int32(0)
                                    if meta_valid:
                                        if f32_t >= sthr[t * 3 + 2]:
                                            pA_k = cutlass.Int32(1)
                                        if f32_t >= sthr[t * 3 + 1] and pA_k == cutlass.Int32(0):
                                            pB_k = cutlass.Int32(1)
                                        if (
                                            f32_t >= sthr[t * 3 + 0]
                                            and pA_k == cutlass.Int32(0)
                                            and pB_k == cutlass.Int32(0)
                                        ):
                                            pC_k = cutlass.Int32(1)
                                    # ---- A: exact claim ----
                                    mA_k = cute.arch.vote_ballot_sync(pA_k != cutlass.Int32(0))
                                    cntA_k = cutlass.Int32(cute.arch.popc(mA_k))
                                    offA_k = cutlass.Int32(cute.arch.popc(mA_k & lmk_k))
                                    baseA_k = cutlass.Int32(0)
                                    if meta_lane == cutlass.Int32(0) and cntA_k > cutlass.Int32(0):
                                        baseA_k = _atom_global_add_s32(cur_k, cntA_k)
                                    baseA_k = cute.arch.shuffle_sync(baseA_k, cutlass.Int32(0))
                                    slotA_k = baseA_k + offA_k
                                    spA_k = cutlass.Int32(0)
                                    if pA_k != cutlass.Int32(0) and slotA_k >= cutlass.Int32(
                                        segA_k
                                    ):
                                        spA_k = cutlass.Int32(1)
                                    if pA_k != cutlass.Int32(0) and slotA_k < cutlass.Int32(segA_k):
                                        vp_k = cute.make_ptr(
                                            cutlass.Float32,
                                            vb_k + cutlass.Int64(slotA_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(vp_k, cute.make_layout((1,)))[0] = f32_t
                                        ip_k = cute.make_ptr(
                                            cutlass.Int32,
                                            ib_k + cutlass.Int64(slotA_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(ip_k, cute.make_layout((1,)))[0] = kv_pos
                                    # ---- B: exact claim (native + A spill) ----
                                    pBe_k = cutlass.Int32(0)
                                    if pB_k != cutlass.Int32(0) or spA_k != cutlass.Int32(0):
                                        pBe_k = cutlass.Int32(1)
                                    mB_k = cute.arch.vote_ballot_sync(pBe_k != cutlass.Int32(0))
                                    cntB_k = cutlass.Int32(cute.arch.popc(mB_k))
                                    offB_k = cutlass.Int32(cute.arch.popc(mB_k & lmk_k))
                                    baseB_k = cutlass.Int32(0)
                                    if meta_lane == cutlass.Int32(0) and cntB_k > cutlass.Int32(0):
                                        baseB_k = _atom_global_add_s32(
                                            cur_k + cutlass.Int64(4), cntB_k
                                        )
                                    baseB_k = cute.arch.shuffle_sync(baseB_k, cutlass.Int32(0))
                                    slotB_k = baseB_k + offB_k
                                    spB_k = cutlass.Int32(0)
                                    if pBe_k != cutlass.Int32(0) and slotB_k >= cutlass.Int32(
                                        segA_k
                                    ):
                                        spB_k = cutlass.Int32(1)
                                    if pBe_k != cutlass.Int32(0) and slotB_k < cutlass.Int32(
                                        segA_k
                                    ):
                                        vp2_k = cute.make_ptr(
                                            cutlass.Float32,
                                            vb_k
                                            + cutlass.Int64(segA_k + slotB_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(vp2_k, cute.make_layout((1,)))[0] = f32_t
                                        ip2_k = cute.make_ptr(
                                            cutlass.Int32,
                                            ib_k
                                            + cutlass.Int64(segA_k + slotB_k) * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(ip2_k, cute.make_layout((1,)))[0] = kv_pos
                                    # n0 += exact placements in A and B
                                    plc_k = (
                                        cntA_k
                                        - cutlass.Int32(
                                            cute.arch.popc(
                                                cute.arch.vote_ballot_sync(
                                                    spA_k != cutlass.Int32(0)
                                                )
                                            )
                                        )
                                    ) + (
                                        cntB_k
                                        - cutlass.Int32(
                                            cute.arch.popc(
                                                cute.arch.vote_ballot_sync(
                                                    spB_k != cutlass.Int32(0)
                                                )
                                            )
                                        )
                                    )
                                    if meta_lane == cutlass.Int32(0) and plc_k > cutlass.Int32(0):
                                        _atom_global_add_s32(ctl_k, plc_k)
                                    # ---- C: claim window (native + B spill) ----
                                    pCe_k = cutlass.Int32(0)
                                    if pC_k != cutlass.Int32(0) or spB_k != cutlass.Int32(0):
                                        pCe_k = cutlass.Int32(1)
                                    mC_k = cute.arch.vote_ballot_sync(pCe_k != cutlass.Int32(0))
                                    cntC_k = cutlass.Int32(cute.arch.popc(mC_k))
                                    offC_k = cutlass.Int32(cute.arch.popc(mC_k & lmk_k))
                                    if cntC_k > cwleft[t]:
                                        # sentinel-fill the old window tail
                                        # (BOTH columns: the consumer pads
                                        # by score -inf, idx -1)
                                        slo_k = cwbase[t] + meta_lane
                                        if meta_lane < cwleft[t] and slo_k < cutlass.Int32(capC_k):
                                            vpo_k = cute.make_ptr(
                                                cutlass.Float32,
                                                vb_k
                                                + cutlass.Int64(2 * segA_k + slo_k)
                                                * cutlass.Int64(4),
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            cute.make_tensor(vpo_k, cute.make_layout((1,)))[0] = (
                                                cutlass.Float32(_META_NEG_FLT_MAX)
                                            )
                                            ipo_k = cute.make_ptr(
                                                cutlass.Int32,
                                                ib_k
                                                + cutlass.Int64(2 * segA_k + slo_k)
                                                * cutlass.Int64(4),
                                                cute.AddressSpace.gmem,
                                                assumed_align=4,
                                            )
                                            cute.make_tensor(ipo_k, cute.make_layout((1,)))[0] = (
                                                cutlass.Int32(-1)
                                            )
                                        mC2_k = cntC_k + cutlass.Int32(self.CAND_WIN)
                                        nbC_k = cutlass.Int32(0)
                                        if meta_lane == cutlass.Int32(0):
                                            nbC_k = _atom_global_add_s32(
                                                cur_k + cutlass.Int64(8), mC2_k
                                            )
                                            _atom_global_add_s32(ctl_k, mC2_k)
                                            if nbC_k + mC2_k > cutlass.Int32(
                                                capC_k
                                            ) and nbC_k <= cutlass.Int32(capC_k):
                                                vdp_k = cute.make_ptr(
                                                    cutlass.Int32,
                                                    ctl_k + cutlass.Int64(4),
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=4,
                                                )
                                                cute.make_tensor(vdp_k, cute.make_layout((1,)))[
                                                    0
                                                ] = cutlass.Int32(1)
                                        nbC_k = cute.arch.shuffle_sync(nbC_k, cutlass.Int32(0))
                                        cwbase[t] = nbC_k
                                        cwleft[t] = mC2_k
                                    slotC_k = cwbase[t] + offC_k
                                    if pCe_k != cutlass.Int32(0) and slotC_k < cutlass.Int32(
                                        capC_k
                                    ):
                                        vpc_k = cute.make_ptr(
                                            cutlass.Float32,
                                            vb_k
                                            + cutlass.Int64(2 * segA_k + slotC_k)
                                            * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(vpc_k, cute.make_layout((1,)))[0] = f32_t
                                        ipc_k = cute.make_ptr(
                                            cutlass.Int32,
                                            ib_k
                                            + cutlass.Int64(2 * segA_k + slotC_k)
                                            * cutlass.Int64(4),
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.make_tensor(ipc_k, cute.make_layout((1,)))[0] = kv_pos
                                    cwbase[t] = cwbase[t] + cntC_k
                                    cwleft[t] = cwleft[t] - cntC_k
                            if cutlass.const_expr(self.emit_hit_stats):
                                # Lane-local accumulation; keep it branchless
                                # (`if meta_hit` compiles to real divergent
                                # branches). Bit-mask select is NaN-safe for
                                # OOB-tile garbage logits. No valid-mask
                                # needed: the bitmap contract only sets bits
                                # inside [0, ctx).
                                meta_hit = (
                                    hit_word >> (kv_pos & cutlass.Int32(31))
                                ) & cutlass.Int32(1)
                                msk = cutlass.Int32(0) - meta_hit  # 0 / ~0
                                inv = cutlass.Int32(-1) - msk
                                fbits = cutlass.Int32(
                                    llvm.bitcast(cutlass.Int32.mlir_type, f32_t.ir_value())
                                )
                                # bits(+FLT_MAX)=0x7F7FFFFF,
                                # bits(-FLT_MAX)=0xFF7FFFFF (as i32: neg).
                                selmin = cutlass.Float32(
                                    llvm.bitcast(
                                        cutlass.Float32.mlir_type,
                                        (
                                            (fbits & msk) | (cutlass.Int32(0x7F7FFFFF) & inv)
                                        ).ir_value(),
                                    )
                                )
                                selmax = cutlass.Float32(
                                    llvm.bitcast(
                                        cutlass.Float32.mlir_type,
                                        (
                                            (fbits & msk) | (cutlass.Int32(-8388609) & inv)
                                        ).ir_value(),
                                    )
                                )
                                seladd = cutlass.Float32(
                                    llvm.bitcast(
                                        cutlass.Float32.mlir_type, (fbits & msk).ir_value()
                                    )
                                )
                                hacc_min[t] = cutlass.min(hacc_min[t], selmin)
                                hacc_max[t] = cutlass.max(hacc_max[t], selmax)
                                hacc_sum[t] = hacc_sum[t] + seladd
                                hacc_cnt[t] = hacc_cnt[t] + meta_hit

                    if cutlass.const_expr(self.use_batched_store and self.defer_logits):
                        for t in cutlass.range_constexpr(next_n):
                            lg_pend[t] = result_arr[t]
                    elif cutlass.const_expr(self.use_batched_store):
                        # Batched STG: all result_arr[t] → mLogits in one pass.
                        for t in cutlass.range_constexpr(next_n):
                            out_row = q_idx * next_n + t
                            mLogits[(out_row, kv_pos)] = result_arr[t]
                    if cutlass.const_expr(self.defer_logits):
                        lg_row = q_idx * next_n
                        lg_pos = kv_pos
                        lg_on = cutlass.Int32(1)

                    if cutlass.const_expr(self.dynamic_sched):
                        left = left - cutlass.Int32(1)
                        if left > cutlass.Int32(0):
                            next_kv_idx = kv_idx + NUM_MATH_WG
                        else:
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                # run end: the row's last tile (or WG1's OOB
                                # tile) may hold lanes >= ctx; mask before flush
                                for _t in cutlass.range_constexpr(next_n):
                                    if kv_pos >= ctx_cur:
                                        pend_v[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            re0, re1, re2, re3 = self._ring_pop(ring, ring_cons, ring_ent, lane_idx)
                            ring_cons.advance()
                            next_q_idx = re0 & cutlass.Int32(0xFFFF)
                            next_kv_idx = re1
                            left = re2
                            next_num_kv = (re3 + cutlass.Int32(127)) >> 7
                            has_work = re2 > cutlass.Int32(0)
                            if cutlass.const_expr(self.emit_block_meta and self.emit_hit_stats):
                                meta_j = cutlass.Int32(0)
                    else:
                        # Advance: inline fetch_next_task
                        next_kv_idx = kv_idx + NUM_MATH_WG
                        if next_kv_idx >= num_kv:
                            if cutlass.const_expr(self.emit_block_meta_deferred):
                                # row end: see the dynamic branch
                                for _t in cutlass.range_constexpr(next_n):
                                    if kv_pos >= ctx_cur:
                                        pend_v[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            next_q_idx = q_idx + 1
                            next_kv_idx = 0
                            if next_q_idx < batch_size:
                                next_num_kv = (
                                    mContextLens[next_q_idx] + block_kv_val - 1
                                ) // block_kv_val
                        # Update while-loop condition
                        has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

                if cutlass.const_expr(self.pdl and self.pdl_trigger == 2):
                    # past this CTA's last work tile
                    griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.defer_logits):
                    if lg_on != cutlass.Int32(0):
                        for _t in cutlass.range_constexpr(next_n):
                            mLogits[(lg_row + _t, lg_pos)] = lg_pend[_t]
                if cutlass.const_expr(self.emit_block_meta_deferred):
                    # flush the last tile's pending block_max (WG 1); a
                    # segment ending mid-row skipped the tail mask, so mask here
                    if q_idx < batch_size:
                        for _t in cutlass.range_constexpr(next_n):
                            if kv_pos >= ctx_cur:
                                pend_v[_t] = cutlass.Float32(_META_NEG_FLT_MAX)
                            r_pend = cute.arch.warp_redux_sync(pend_v[_t], "fmax")
                            _st_global_f32(
                                pend_addr
                                + cutlass.Int64(_t) * cutlass.Int64(bm_nrec) * cutlass.Int64(4),
                                r_pend,
                            )
                # Flush the final request's hit accumulators (WG 1).
                if cutlass.const_expr(self.emit_block_meta and self.emit_hit_stats):
                    if q_idx < batch_size:
                        self._flush_hit_agg(
                            mHitAgg, q_idx, hacc_min, hacc_max, hacc_sum, hacc_cnt, meta_lane
                        )
                if cutlass.const_expr(self.emit_seed_counts):
                    if q_idx < batch_size:
                        self._flush_seed_counts(
                            mSeedCounts, q_idx, scnt, meta_lane, spass=spass, cand_ctl=mCandCtl
                        )
                if cutlass.const_expr(self.emit_cand):
                    if q_idx < batch_size:
                        self._flush_cand_window(mCand, q_idx, cwbase, cwleft, meta_lane)
                if cutlass.const_expr(self.emit_cand_bucketed):
                    if q_idx < batch_size:
                        self._flush_cand_window_bucketed(
                            mCand, mCandIdx, q_idx, cwbase, cwleft, meta_lane
                        )

                # Release last Q stage (WG 1)
                if q_idx < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            # TMEM dealloc: math warps are allocator + last consumer
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        else:
            cute.arch.warpgroup_reg_dealloc(self.prod_regs)
