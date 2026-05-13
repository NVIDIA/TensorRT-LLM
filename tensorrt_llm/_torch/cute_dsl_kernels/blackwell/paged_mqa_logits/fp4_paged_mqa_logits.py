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
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

# Compat across CuTe DSL versions:
#   4.4.x              : nvvm.RoundingModeKind.RN (enum, accepted by cute.arch.*)
#   4.5.0+ / internal  : nvvm.RoundingModeKind removed; cute.arch.* strictly require
#                        the string literal 'rn' (passing FPRoundingMode.RN enum
#                        raises TypeError per nvvm_wrappers.from_str).
_RND_RN = getattr(getattr(nvvm, "RoundingModeKind", None), "RN", None) or "rn"


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
    values = cute.make_fragment(4, cutlass.Int32)
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
        # Note, when next_n=1, num_umma_stages could be 2. But test on silicon shows,
        # num_umma_stages=2 don't improve perf much, so we use num_umma_stages=1 now.
        # This parameter could be tuned when needed.
        # self.num_umma_stages = 2 if next_n == 1 else 1
        self.num_umma_stages = 1
        # KV pipeline depth
        self.num_kv_stages = 6
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
    ):
        # Derive KV data and SF views from the fused uint8 buffer.
        # Fused layout per phys block: [data half_head_dim*phys_block_kv bytes]
        #                              [SF   phys_block_kv*4         bytes (= phys_block_kv int32)]
        phys_block_kv = self.phys_block_kv
        half_head_dim = self.head_dim // 2  # FP4 packed bytes per row
        phys_block_bytes = phys_block_kv * (half_head_dim + 4)
        scale_offset_bytes = phys_block_kv * half_head_dim  # to SF region of each phys block

        # Recast the fused buffer to FP4. Each uint8 byte becomes 2 FP4 elements,
        # so layout positions and the iterator scale accordingly.
        kv_fp4 = cute.recast_tensor(kv_fused, Float4E2M1FN)

        # Q (b) was passed as uint8 (FP4-packed bytes); recast to FP4 so MMA
        # type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, Float4E2M1FN)

        # KV data view: [phys_block_kv, head_dim, num_phys_blocks] FP4 elements.
        # Innermost stride 1 = consecutive FP4 elem = packed pair share a byte.
        # Per-row stride = head_dim FP4 elem = head_dim/2 bytes.
        # Per-block stride = phys_block_bytes * 2 FP4 elem (data + SF region).
        kv_layout = cute.make_layout(
            (phys_block_kv, self.head_dim, num_phys_blocks),
            stride=(self.head_dim, 1, phys_block_bytes * 2),
        )
        a = cute.make_tensor(kv_fp4.iterator, kv_layout)

        # SF KV view: int32 (4 UE8M0 packed). Build a uint8 view at the SF
        # offset, then recast to int32.
        # Layout in bytes: (phys_block_kv * 4, num_phys_blocks) stride (1, phys_block_bytes)
        # After recast int32: (phys_block_kv, num_phys_blocks) stride (1, phys_block_bytes/4)
        sf_kv_uint8_layout = cute.make_layout(
            (phys_block_kv * 4, num_phys_blocks),
            stride=(1, phys_block_bytes),
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

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

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
        )

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
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_half = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_half = mScheduleMeta[(sm_idx + 1, 1)]
        # Early mContextLens load: overlap ~200-cycle L2 latency with the
        # entire prologue setup (pipelines, SMEM alloc, TMA partition, etc.)
        # Clamp to avoid OOB when start_q == batch_size (zero-work CTA sentinel).
        # Note: zero-work CTAs get a stale current_num_kv (from the last batch
        # element), but it is never used because has_work will be False.
        start_q_clamped = min(start_q, batch_size - 1)
        current_num_kv = (mContextLens[start_q_clamped] + self.block_kv - 1) // self.block_kv

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_sf_kv)
            cpasync.prefetch_descriptor(tma_atom_sf_q)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

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
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

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

        # ===== SCHEDULER: derive values from early-loaded schedule metadata =====
        end_kv_idx = end_kv_half * NUM_MATH_WG

        # Convert start to KV block units
        current_q_idx = start_q
        current_kv_idx = start_kv_half * NUM_MATH_WG

        # ===== COMMON SCHEDULER STATE (before warp branches) =====
        # Each warp role independently maintains its own copy of these
        # variables (like DeepGEMM where each role creates its own scheduler).
        # Pre-fetch first task (current_num_kv loaded early above for latency hiding)
        next_q_idx = current_q_idx
        next_kv_idx = current_kv_idx
        next_num_kv = current_num_kv
        # Sentinel: no previous batch (q_idx = batch_size)
        q_idx = batch_size
        # While-loop termination flag (fetch_next_task pattern).
        # True if this CTA has work assigned (start != end in schedule_meta).
        has_work = (current_q_idx != end_q_idx) | (current_kv_idx != end_kv_idx)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            # TMA warp 0: loads Q (prefetch) + KV for group 0
            cute.arch.warpgroup_reg_dealloc(24)
            lane_idx = tidx % 32

            # Block table prefetch: 32 lanes cache block indices,
            # distributed via shuffle. Each lane holds num_blocks_per_mma
            # physical block indices per compute tile.
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use

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

                # Q prefetch: when batch changes, load Q for NEXT batch
                if q_idx != q_idx_old:
                    kv_blk_ptr = cutlass.Int32(32)  # force re-prefetch
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

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (mContextLens[next_q_idx] + block_kv_val - 1) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_tma_warp_1:
            # TMA warp 1: loads KV + Scale for group 1 only
            cute.arch.warpgroup_reg_dealloc(24)
            lane_idx = tidx % 32

            # Block table prefetch for group 1
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use

            while has_work:
                # fetch_next_task: commit next → current
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

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

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (mContextLens[next_q_idx] + block_kv_val - 1) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        elif is_umma_warp_0:
            # UMMA warp for group 0
            # Must wait on Q pipeline: TMA operations with different
            # barriers are NOT visibility-ordered even within the same
            # warp. KV0 barrier arriving does not guarantee Q SMEM
            # writes are visible.
            cute.arch.warpgroup_reg_dealloc(24)

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
            cute.arch.warpgroup_reg_dealloc(24)

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
            cute.arch.warpgroup_reg_alloc(240)

            # TMEM: math warp 0 is the allocator; all math warps wait + retrieve
            tmem.allocate(num_tmem_alloc_cols_total)
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
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_fragment(NUM_W_IN_REG * next_n, self.epi_dtype)
                q_stage_local = cutlass.Int32(0)

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
                                w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                    (t_i * num_heads + w_j, q_stage_local)
                                ]

                    # Process KV block for group 0 (kv_idx + 0)
                    # Unconditional Math: OOB results
                    # written to aligned padding region in logits buffer.
                    kv_pos = kv_idx * block_kv_val + m_coord

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
                                    a0 = cutlass.max(acc_vec[n0], cutlass.Float32(0.0))
                                    a1 = cutlass.max(acc_vec[n0 + 1], cutlass.Float32(0.0))
                                    a2 = cutlass.max(acc_vec[n0 + 2], cutlass.Float32(0.0))
                                    a3 = cutlass.max(acc_vec[n0 + 3], cutlass.Float32(0.0))
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
                        out_row = q_idx * next_n + t
                        # Step 5.7: drop * scale_val (FP4 SF baked into acc).
                        mLogits[(out_row, kv_pos)] = self.output_dtype(result_t)

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
                NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
                w_cache = cute.make_fragment(NUM_W_IN_REG * next_n, self.epi_dtype)
                q_stage_local = cutlass.Int32(0)

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
                                w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                    (t_i * num_heads + w_j, q_stage_local)
                                ]

                    # Process KV block for group 1 (kv_idx + 1)
                    # Unconditional Math
                    kv_idx_1 = kv_idx + 1

                    kv_pos = kv_idx_1 * block_kv_val + m_coord

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
                                    a0 = cutlass.max(acc_vec[n0], cutlass.Float32(0.0))
                                    a1 = cutlass.max(acc_vec[n0 + 1], cutlass.Float32(0.0))
                                    a2 = cutlass.max(acc_vec[n0 + 2], cutlass.Float32(0.0))
                                    a3 = cutlass.max(acc_vec[n0 + 3], cutlass.Float32(0.0))
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
                        out_row = q_idx * next_n + t
                        # Step 5.7: drop * scale_val (FP4 SF baked into acc).
                        mLogits[(out_row, kv_pos)] = self.output_dtype(result_t)

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

                # Release last Q stage (WG 1)
                if q_idx < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()

            # TMEM dealloc: math warps are allocator + last consumer
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        else:
            cute.arch.warpgroup_reg_dealloc(24)
