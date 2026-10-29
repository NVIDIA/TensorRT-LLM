# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M split-K BF16 GEMM with fused epilogues for the MLA q_b projection (SM10x, CuTe DSL).

``SplitKFusedEpilogueGemmKernel`` runs the mainloop and the cluster split-K reduction of
``_low_m_splitk.SplitKDenseGemmKernel`` unchanged. Only the owner CTA's epilogue differs:
it stages the reduced FP32 tile in shared memory and then stores four consecutive output
columns per thread, so per-row and per-column-pair operations do not depend on the TMEM load
layout. Options (``SplitKEpilogue``):

``row_rms``
    B (the public activation ``a``) is the raw input x of an RMSNorm whose weight is folded
    into the GEMM weight (``W * gamma``). The owner's epilogue warps compute every row's mean
    square from global memory while the mainloop runs, and the row's outputs are scaled by
    ``rsqrt(mean + eps)``: ``(x * rstd) @ (W * gamma)^T``, which is ``RMSNorm(x) @ W^T`` up to
    rounding.
``rms_out``
    Also write ``RMSNorm(x) = x * rstd * gamma`` (BF16 [M, K]) for other readers of the
    normalized activation. The owner CTAs of a row tile split its K columns.
``fp8_rope``
    Write FP8 E4M3 (times a device scale) instead of BF16, after GPT-J RoPE on the last
    ``rope_dim`` columns of every ``head_dim``-wide head: the MLA generation Q layout
    [tokens, heads, kv_lora_rank + qk_rope_head_dim] that ``mla_rope_generation`` rotates
    and quantizes, with its position rule (``kv_lens[seq] - seq_len + token_in_seq``), its
    cos/sin table layout and its saturating E4M3 conversion. The BF16 rounding the unfused
    path applies before the rotation and before the quantization is skipped.

``SplitKIndexerQTailGemmKernel`` (``run_splitk_idx_qtail``) is the ``row_rms`` kernel of the DSA
indexer wq_b whose owner epilogue finishes the indexer Q (``IndexerQTail``) instead of storing it:
it rounds every output to BF16 as the BF16 store does, then applies the per-(token, head) tail of
the fused DSA indexer prologue to those values with the same operations in the same order (RoPE,
BF16 rounding, the per-head E4M3 scale, the E4M3 Q, the scale, and the attention weights times
the scale). Head-wide tiles finish a head per warp; half-head tiles run
in clusters of the two halves, whose owners exchange their row amax through DSMEM, so the GEMM
keeps its 64-row tactic and its K split.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch as _torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cutlass_dsl import T, dsl_user_op

from ._low_m_splitk import (
    _AB_BUFFER_ALIGN_BYTES,
    _CTA_K,
    _LAUNCH_SMEM_HEADROOM_BYTES,
    _MAILBOX_ALIGN_BYTES,
    _MBARRIER_BYTES,
    _SMEM_CAPACITY_BYTES,
    _TMEM_POINTER_BYTES,
    OWNER_RANK,
    SplitKDenseGemmKernel,
    SplitKTactic,
    _align_up,
    _detect_leading_dim,
    _from_dlpack_dynamic,
    _smem_bytes,
    _store_shared_remote_v4,
    default_tactic,
    device_limits,
    splitk_default_tactic,
    validate_tactic,
)

__all__ = [
    "IndexerQTail",
    "Q_B_PLANS",
    "SplitKEpilogue",
    "fused_epilogue_tactic",
    "idx_qtail_plan",
    "idx_qtail_tactic",
    "q_b_plan",
    "run_splitk_fused",
    "run_splitk_idx_qtail",
]

#: Alignment of the epilogue's shared-memory staging buffers.
_EPI_ALIGN_BYTES = 16

#: Named barrier of the four epilogue warps (barrier 0 is the CTA-wide one).
_EPI_BARRIER_ID = 1

#: BF16 elements per 16-byte vector.
_BF16_PER_VEC = 8

#: Lanes of a warp.
_WARP_LANES = 32


@dataclasses.dataclass(frozen=True)
class SplitKEpilogue:
    """Fused epilogue options; every field is a compile-time constant of the kernel."""

    row_rms: bool = False
    rms_k: int = 0
    rms_eps: float = 0.0
    rms_out: bool = False
    fp8_rope: bool = False
    head_dim: int = 0
    rope_dim: int = 0
    seq_len: int = 1

    def validate(self) -> None:
        if not (self.row_rms or self.fp8_rope):
            raise ValueError("a fused epilogue needs row_rms or fp8_rope")
        if self.row_rms and (self.rms_k <= 0 or self.rms_k % _BF16_PER_VEC or self.rms_eps <= 0):
            raise ValueError(f"row_rms needs a positive rms_k divisible by 8 and eps > 0: {self}")
        if self.rms_out and not self.row_rms:
            raise ValueError("rms_out needs row_rms")
        if self.fp8_rope and not (
            0 < self.rope_dim < self.head_dim
            and self.rope_dim % 4 == 0
            and (self.head_dim - self.rope_dim) % 4 == 0
            and self.seq_len >= 1
        ):
            raise ValueError(f"invalid fp8_rope geometry: {self}")


def _fused_smem_bytes(tactic: SplitKTactic, ab_stages: int) -> int:
    """Shared memory of the fused kernel: the base layout, then the FP32 tile and row scales."""
    cursor = _align_up(_smem_bytes(tactic, ab_stages), _EPI_ALIGN_BYTES)
    cursor += tactic.mma_m * tactic.mma_n * 4
    return _align_up(cursor, _EPI_ALIGN_BYTES) + tactic.mma_n * 4


def fused_epilogue_tactic(tactic: SplitKTactic) -> SplitKTactic:
    """``tactic`` with at most as many A/B stages as fit beside the epilogue staging buffers."""
    stages = tactic.ab_stages
    while stages > 1 and _fused_smem_bytes(tactic, stages) > _SMEM_CAPACITY_BYTES:
        stages -= 1
    return dataclasses.replace(tactic, ab_stages=stages)


@dataclasses.dataclass(frozen=True)
class IndexerQTail:
    """The DSA indexer Q tail of ``run_splitk_idx_qtail``; every field is a compile-time constant.

    Per (token, head) of the BF16-rounded wq_b output, as ``_dsa_indexer_prologue_fused_kernel``:
    RoPE on the first ``rope_dim`` columns (GPT-J pairs with ``interleave``, NeoX halves otherwise)
    with the cos/sin row of the token's position, rounded to BF16; the head's E4M3 scale
    max(amax, 1e-12) / 448, rounded up to a power of two with ``ue8m0``; the E4M3 Q (times the
    MUFU reciprocal of the scale); the scale; and the attention weight times the scale times
    ``weight_scale``.
    """

    n_heads: int
    head_dim: int = 128
    rope_dim: int = 64
    interleave: bool = True
    ue8m0: bool = True
    weight_scale: float = 1.0

    def validate(self) -> None:
        if not (
            self.n_heads > 0
            and self.head_dim % 8 == 0
            and 0 < self.rope_dim <= self.head_dim // 2
            and self.rope_dim % 8 == 0
            and self.weight_scale > 0
        ):
            raise ValueError(f"invalid indexer Q tail geometry: {self}")


def _qtail_pair(tactic: SplitKTactic, qtail: IndexerQTail) -> bool:
    """Whether a tile holds half a head (clusters of the two halves) rather than a whole head."""
    if tactic.mma_m == qtail.head_dim:
        return False
    if 2 * tactic.mma_m == qtail.head_dim:
        return True
    raise ValueError(f"tile of {tactic.mma_m} columns is neither a head nor half a head: {qtail}")


def _qtail_smem_bytes(tactic: SplitKTactic, ab_stages: int, qtail: IndexerQTail) -> int:
    """Shared memory of the indexer Q tail kernel: the fused layout, then the cos/sin rows and the
    attention weights of the tile's tokens, then (half-head tiles) both halves' row amax and their
    mbarrier."""
    cursor = _align_up(_fused_smem_bytes(tactic, ab_stages), _EPI_ALIGN_BYTES)
    cursor += tactic.mma_n * qtail.rope_dim * 4
    cursor = _align_up(cursor, _EPI_ALIGN_BYTES) + tactic.mma_n * 4
    if _qtail_pair(tactic, qtail):
        cursor = _align_up(cursor, _EPI_ALIGN_BYTES) + tactic.mma_n * 4
        cursor = _align_up(cursor, _EPI_ALIGN_BYTES) + tactic.mma_n * 4
        cursor = _align_up(cursor, _MBARRIER_BYTES) + _MBARRIER_BYTES
    return cursor


def idx_qtail_tactic(tactic: SplitKTactic, qtail: IndexerQTail) -> SplitKTactic:
    """``tactic`` with at most as many A/B stages as fit beside the indexer Q tail's buffers."""
    if _qtail_pair(tactic, qtail) and 2 * tactic.split_k > 8:
        raise ValueError(f"half-head tiles need a cluster of at most 8 CTAs: {tactic}")
    stages = tactic.ab_stages
    while stages > 1 and _qtail_smem_bytes(tactic, stages, qtail) > _SMEM_CAPACITY_BYTES:
        stages -= 1
    return dataclasses.replace(tactic, ab_stages=stages)


#: A/B pipeline stages of the absorbed q_b queued beside the DSA indexer top-k.
_BESIDE_TOPK_AB_STAGES = 3

#: The launch plans of the fused q_b (``q_b_plan``).
Q_B_PLANS = ("regular", "beside_topk", "early_gather")


def q_b_plan(plan: str, m: int, n: int, k: int, device_index: int) -> SplitKTactic:
    """The split-K tactic of the fused q_b ``[m, k] @ [k, n]`` under ``plan``.

    ``regular``: ``splitk_default_tactic``. ``beside_topk``: 128-row tiles, no split-K and three
    A/B stages, so that a q_b queued beside the DSA indexer top-k takes few SMs and little memory
    bandwidth from it; the regular tactic where that one does not serve the shape.
    ``early_gather``: the source default (split-K 4) where the regular tactic is split-K 8, which
    leaves the clusters of a flash decode that gathers before its grid dependency wait room to
    land while q_b runs. ``run_splitk_fused`` then lowers the A/B stages to fit its buffers.
    """
    if plan not in Q_B_PLANS:
        raise ValueError(f"unknown q_b plan {plan!r}; expected one of {Q_B_PLANS}")
    sm_count, smem_capacity = device_limits(device_index)
    tactic = splitk_default_tactic(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity)
    if plan == "beside_topk":
        candidate = dataclasses.replace(
            tactic, mma_m=128, split_k=1, ab_stages=_BESIDE_TOPK_AB_STAGES
        )
        try:
            validate_tactic(candidate, m, n, k)
        except ValueError:
            return tactic
        return candidate
    if plan == "early_gather" and tactic.split_k == 8:
        return default_tactic(m, n, k)
    return tactic


def idx_qtail_plan(qtail: IndexerQTail, m: int, n: int, k: int, device_index: int) -> SplitKTactic:
    """The split-K tactic of the indexer wq_b ``[m, k] @ [k, n]`` with the Q tail: the
    ``splitk_default_tactic`` of the shape, its 64-row tiles as half-head tiles, its A/B stages
    lowered to fit the Q tail's buffers. Raises ``ValueError`` where it does not serve the shape.
    """
    sm_count, smem_capacity = device_limits(device_index)
    tactic = splitk_default_tactic(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity)
    tactic = idx_qtail_tactic(tactic, qtail)
    validate_tactic(tactic, m, n, k)
    return tactic


# ----------------------------------------------------------------------------- PTX helpers


@dsl_user_op
def _ld_global_v4(addr: Int64, *, loc=None, ip=None):
    """16-byte global load as four 32-bit words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [addr.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _bf16x2_to_f32x2(word: Int32, *, loc=None, ip=None):
    """The two BF16 halves of a 32-bit word (low half first) as FP32."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [word.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 bflo, bfhi;\n\tshl.b32 bflo, $2, 16;\n\tand.b32 bfhi, $2, 0xFFFF0000;\n\t"
        "mov.b32 $0, bflo;\n\tmov.b32 $1, bfhi;\n\t}",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def _cvt_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """Round two FP32 values to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_e4m3x4(v0: Float32, v1: Float32, v2: Float32, v3: Float32, *, loc=None, ip=None) -> Int32:
    """Four FP32 values to saturated E4M3 (nearest even), ``v0`` in the lowest byte."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [v.ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n\t.reg .b16 fp8lo, fp8hi;\n\tcvt.rn.satfinite.e4m3x2.f32 fp8lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 fp8hi, $4, $3;\n\tmov.b32 $0, {fp8lo, fp8hi};\n\t}",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_global_b32(addr: Int64, v: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip), v.ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_global_v2(addr: Int64, v0: Int32, v1: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip), v0.ir_value(loc=loc, ip=ip), v1.ir_value(loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_global_v4(
    addr: Int64, v0: Int32, v1: Int32, v2: Int32, v3: Int32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip)] + [v.ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b + c`` rounded once (nearest even)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b`` rounded once (nearest even), never contracted into an FMA."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _neg_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """``-a`` (sign flip, exact)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip)],
            "neg.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fp8_ratio(amax: Float32, *, loc=None, ip=None) -> Float32:
    """max(amax, 1e-12) * fp32(1 / 448): the per-row E4M3 ratio of the indexer prologue
    (``_fp8_row_scale``), with its FP32 constants."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [amax.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 fpfloor;\n\tmax.f32 fpfloor, $1, 0f2B8CBCCC;\n\t"
            "mul.rn.f32 $0, fpfloor, 0f3B124925;\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ue8m0_up(ratio: Float32, *, loc=None, ip=None) -> Float32:
    """The positive ``ratio`` with its mantissa cleared and, if the mantissa was nonzero, its
    exponent raised by one: the UE8M0 scale of the indexer prologue."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [ratio.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 uebits, uemant, ueexp;\n\t.reg .pred uenz;\n\tmov.b32 uebits, $1;\n\t"
            "and.b32 uemant, uebits, 0x007FFFFF;\n\tand.b32 ueexp, uebits, 0x7F800000;\n\t"
            "setp.ne.u32 uenz, uemant, 0;\n\t@uenz add.u32 ueexp, ueexp, 0x00800000;\n\t"
            "mov.b32 $0, ueexp;\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _bf16_round(lo: Float32, hi: Float32):
    """Two FP32 values rounded to BF16 (nearest even), back as FP32."""
    return _bf16x2_to_f32x2(_cvt_bf16x2(lo, hi))


# --------------------------------------------------------------------------------- kernel


class SplitKFusedEpilogueGemmKernel(SplitKDenseGemmKernel):
    """``SplitKDenseGemmKernel`` (no bias) with a fused owner epilogue."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        epilogue: SplitKEpilogue,
        reserve_full_smem: bool = True,
    ) -> None:
        super().__init__(
            tactic=tactic, use_pdl=use_pdl, has_bias=False, reserve_full_smem=reserve_full_smem
        )
        epilogue.validate()
        self.epi = epilogue
        self.launch_smem_bytes = min(
            _align_up(_fused_smem_bytes(tactic, tactic.ab_stages), 1024)
            + _LAUNCH_SMEM_HEADROOM_BYTES,
            _SMEM_CAPACITY_BYTES,
        )
        if (self.cta_m * self.cta_n // 4) % self.epilog_threads:
            raise ValueError(
                f"tile ({self.cta_m}, {self.cta_n}) does not split into 4-column groups"
            )
        self.rows_per_tile = self.cta_n
        self.threads_per_row = self.epilog_threads // self.cta_n
        if epilogue.row_rms and (
            self.threads_per_row & (self.threads_per_row - 1)
            or self.threads_per_row > 32
            or (epilogue.rms_k // _BF16_PER_VEC) % self.threads_per_row
        ):
            raise ValueError(
                f"row_rms cannot split K={epilogue.rms_k} over a ({self.cta_m}, {self.cta_n}) tile"
            )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        gamma: cute.Tensor,
        norm_out: cute.Tensor,
        kv_lens: cute.Tensor,
        cos_sin: cute.Tensor,
        q_scale: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        self.kernel(a, b, c, gamma, norm_out, kv_lens, cos_sin, q_scale).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(
                utils.get_smem_capacity_in_bytes("sm_100")
                if self.reserve_full_smem
                else self.launch_smem_bytes
            ),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation (rows = tokens)
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major: BF16, or uint8 for fp8_rope
        mGamma: cute.Tensor,  # [K] BF16 norm weight (rms_out)
        mNormOut: cute.Tensor,  # [M, K] BF16 normalized activation (rms_out)
        mKvLens: cute.Tensor,  # [sequences] int32 KV lengths (fp8_rope)
        mCosSin: cute.Tensor,  # flat FP32 (cos, sin) table (fp8_rope)
        mQScale: cute.Tensor,  # [1] FP32 quantization scale (fp8_rope)
    ):
        """Allocate storage and dispatch the specialized warps (as the base kernel)."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            mailbox = sA
            bar_reduce = bar_mma_epilog

        # Epilogue staging: the reduced FP32 tile (M-major, as the output tile) and the
        # per-row scales.
        sAcc = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n)),
            alignment=_EPI_ALIGN_BYTES,
        )
        sRstd = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.cta_n),
            alignment=_EPI_ALIGN_BYTES,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_fused(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                sAcc,
                sRstd,
                mB,
                mC,
                mGamma,
                mNormOut,
                mKvLens,
                mCosSin,
                mQScale,
                bidx,
                n_idx,
            )

    @cute.experimental.jit
    def epilog_fused(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        sAcc: cute.Tensor,
        sRstd: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mGamma: cute.Tensor,
        mNormOut: cute.Tensor,
        mKvLens: cute.Tensor,
        mCosSin: cute.Tensor,
        mQScale: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
    ):
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        # The TMEM load op of the BF16 kernel, whatever the output type: the tile is staged
        # in shared memory, so only the accumulator partition matters.
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                cutlass.BFloat16,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # Row statistics overlap the mainloop: only the owner needs them.
        if cutlass.const_expr(self.epi.row_rms):
            if split_rank == OWNER_RANK:
                self._row_rstd(mB, sRstd, n_idx, epi_tid)
                if cutlass.const_expr(self.epi.rms_out):
                    self._rms_out(mB, mC, mGamma, mNormOut, sRstd, bidx, n_idx, epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            # Stage the tile by output coordinate, then store four columns per thread.
            cute_ext.partition_and_copy(thr_t2r, rAcc, sAcc)
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=self.epilog_threads)
            self._store_tile(sAcc, sRstd, mC, mKvLens, mCosSin, mQScale, bidx, n_idx, epi_tid)

    @cute.experimental.jit
    def _row_rstd(
        self,
        mB: cute.Tensor,
        sRstd: cute.Tensor,
        n_idx: cutlass.Int32,
        epi_tid: cutlass.Int32,
    ):
        """sRstd[row] = rsqrt(mean(x[row]^2) + eps) for the tile's rows; ends with a barrier."""
        tpr = cutlass.const_expr(self.threads_per_row)
        vecs_per_thread = cutlass.const_expr(self.epi.rms_k // _BF16_PER_VEC // tpr)
        # The activation is the previous grid's output.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        row = epi_tid // tpr
        lane = epi_tid - row * tpr
        token = n_idx * self.cta_n + row
        ssq = Float32(0.0)
        if token < cute.size(mB, mode=[0]):
            base = (mB.iterator + cute.crd2idx((token, 0, 0), mB.layout)).toint() + Int64(lane * 16)
            for i in cutlass.range_constexpr(vecs_per_thread):
                words = _ld_global_v4(base + Int64(i * tpr * 16))
                for j in cutlass.range_constexpr(4):
                    pair = _bf16x2_to_f32x2(words[j])
                    ssq = ssq + pair[0] * pair[0] + pair[1] * pair[1]
        for i in cutlass.range_constexpr(int(math.log2(tpr))):
            ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, offset=1 << i)
        if lane == 0:
            sRstd[row] = cute.math.rsqrt(
                ssq / Float32(self.epi.rms_k) + Float32(self.epi.rms_eps), fastmath=True
            )
        cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=self.epilog_threads)

    @cute.experimental.jit
    def _rms_out(
        self,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mGamma: cute.Tensor,
        mNormOut: cute.Tensor,
        sRstd: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        epi_tid: cutlass.Int32,
    ):
        """Write x * rstd * gamma for this tile's rows and its share of the K columns."""
        vecs = cutlass.const_expr(self.epi.rms_k // _BF16_PER_VEC)
        n_tiles = cute.ceil_div(cute.size(mC, mode=[0]), self.cta_m)
        chunk = (vecs + n_tiles - 1) // n_tiles
        total = chunk * self.cta_n
        iters = (total + self.epilog_threads - 1) // self.epilog_threads
        tokens = cute.size(mB, mode=[0])
        for i in cutlass.range(iters, unroll=1):
            idx = epi_tid + i * self.epilog_threads
            row = idx // chunk
            vec = bidx * chunk + (idx - row * chunk)
            token = n_idx * self.cta_n + row
            if idx < total:
                if vec < vecs:
                    if token < tokens:
                        col = vec * _BF16_PER_VEC
                        x_words = _ld_global_v4(
                            (mB.iterator + cute.crd2idx((token, col, 0), mB.layout)).toint()
                        )
                        g_words = _ld_global_v4((mGamma.iterator + col).toint())
                        rstd = sRstd[row]
                        out_words = []
                        for j in cutlass.range_constexpr(4):
                            x_pair = _bf16x2_to_f32x2(x_words[j])
                            g_pair = _bf16x2_to_f32x2(g_words[j])
                            out_words.append(
                                _cvt_bf16x2(
                                    x_pair[0] * rstd * g_pair[0], x_pair[1] * rstd * g_pair[1]
                                )
                            )
                        out_ptr = mNormOut.iterator + cute.crd2idx((token, col), mNormOut.layout)
                        _st_global_v4(
                            out_ptr.toint(),
                            out_words[0],
                            out_words[1],
                            out_words[2],
                            out_words[3],
                        )

    @cute.experimental.jit
    def _store_tile(
        self,
        sAcc: cute.Tensor,
        sRstd: cute.Tensor,
        mC: cute.Tensor,
        mKvLens: cute.Tensor,
        mCosSin: cute.Tensor,
        mQScale: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        epi_tid: cutlass.Int32,
    ):
        """Store the staged tile: four consecutive output columns of one row per thread."""
        groups_m = cutlass.const_expr(self.cta_m // 4)
        groups_per_thread = cutlass.const_expr(self.cta_m * self.cta_n // 4 // self.epilog_threads)
        n_feat = cute.size(mC, mode=[0])
        n_tok = cute.size(mC, mode=[1])
        q_scale = Float32(1.0)
        if cutlass.const_expr(self.epi.fp8_rope):
            q_scale = mQScale[0]
        for g in cutlass.range_constexpr(groups_per_thread):
            gi = epi_tid + g * self.epilog_threads
            row = gi // groups_m
            m4 = (gi - row * groups_m) * 4
            v0 = sAcc[m4, row]
            v1 = sAcc[m4 + 1, row]
            v2 = sAcc[m4 + 2, row]
            v3 = sAcc[m4 + 3, row]
            if cutlass.const_expr(self.epi.row_rms):
                rstd = sRstd[row]
                v0 = v0 * rstd
                v1 = v1 * rstd
                v2 = v2 * rstd
                v3 = v3 * rstd
            token = n_idx * self.cta_n + row
            feature = bidx * self.cta_m + m4
            if token < n_tok:
                if feature < n_feat:
                    addr = (mC.iterator + cute.crd2idx((feature, token, 0), mC.layout)).toint()
                    if cutlass.const_expr(self.epi.fp8_rope):
                        nope = cutlass.const_expr(self.epi.head_dim - self.epi.rope_dim)
                        col = feature % self.epi.head_dim
                        if col >= nope:
                            seq = token // self.epi.seq_len
                            pos = mKvLens[seq] - self.epi.seq_len + (token - seq * self.epi.seq_len)
                            # The table holds rope_dim (cos, sin) pairs per position; pair p
                            # rotates columns 2p and 2p + 1 of the head's RoPE slice.
                            cs = (pos * self.epi.rope_dim + (col - nope) // 2) * 2
                            c0 = mCosSin[cs]
                            s0 = mCosSin[cs + 1]
                            c1 = mCosSin[cs + 2]
                            s1 = mCosSin[cs + 3]
                            r0 = c0 * v0 - s0 * v1
                            r1 = c0 * v1 + s0 * v0
                            r2 = c1 * v2 - s1 * v3
                            r3 = c1 * v3 + s1 * v2
                            v0 = r0
                            v1 = r1
                            v2 = r2
                            v3 = r3
                        _st_global_b32(
                            addr,
                            _cvt_e4m3x4(v0 * q_scale, v1 * q_scale, v2 * q_scale, v3 * q_scale),
                        )
                    else:
                        _st_global_v2(addr, _cvt_bf16x2(v0, v1), _cvt_bf16x2(v2, v3))


class SplitKIndexerQTailGemmKernel(SplitKFusedEpilogueGemmKernel):
    """The ``row_rms`` kernel of the DSA indexer wq_b with the indexer Q tail (``IndexerQTail``)
    in the owner epilogue, which writes the E4M3 Q, its per-head scale and the scaled attention
    weights instead of the BF16 output.

    A tile is one head (``mma_m == head_dim``: a warp finishes one token's head) or half a head
    (``2 * mma_m == head_dim``: half a warp per token, and the two halves of a head run in one
    cluster of shape (2, split_k, 1), whose owners exchange their row amax through DSMEM). The
    mainloop, the split-K reduction and the row RMS are those of the fused kernel, so the GEMM of
    a tactic is unchanged.
    """

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        epilogue: SplitKEpilogue,
        qtail: IndexerQTail,
        reserve_full_smem: bool = True,
    ) -> None:
        super().__init__(
            tactic=tactic, use_pdl=use_pdl, epilogue=epilogue, reserve_full_smem=reserve_full_smem
        )
        qtail.validate()
        if not epilogue.row_rms or epilogue.rms_out or epilogue.fp8_rope:
            raise ValueError("the indexer Q tail follows the row_rms epilogue alone")
        self.qtail = qtail
        self.pair = _qtail_pair(tactic, qtail)
        self.cluster_shape = (2 if self.pair else 1, tactic.split_k, 1)
        self.lanes_per_row = self.cta_m // 4
        if _WARP_LANES % self.lanes_per_row or (self.epilog_threads // self.cta_n) == 0:
            raise ValueError(f"tile ({self.cta_m}, {self.cta_n}) does not map rows onto lanes")
        if qtail.rope_dim % (self.epilog_threads // self.cta_n):
            raise ValueError(f"rope_dim {qtail.rope_dim} does not split over a tile row's threads")
        self.launch_smem_bytes = min(
            _align_up(_qtail_smem_bytes(tactic, tactic.ab_stages, qtail), 1024)
            + _LAUNCH_SMEM_HEADROOM_BYTES,
            _SMEM_CAPACITY_BYTES,
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        positions: cute.Tensor,
        cos_sin: cute.Tensor,
        weights_in: cute.Tensor,
        q_scale_out: cute.Tensor,
        weights_out: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        self.kernel(a, b, c, positions, cos_sin, weights_in, q_scale_out, weights_out).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(
                utils.get_smem_capacity_in_bytes("sm_100")
                if self.reserve_full_smem
                else self.launch_smem_bytes
            ),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation (rows = tokens)
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major: uint8 (E4M3 bits)
        mPos: cute.Tensor,  # [tokens] int32 positions
        mCosSin: cute.Tensor,  # [positions, >= rope_dim] FP32 [cos | sin] rows
        mWIn: cute.Tensor,  # [tokens, heads] BF16 attention weights (weights_proj output)
        mQScaleOut: cute.Tensor,  # [tokens, heads] FP32 E4M3 scales
        mWOut: cute.Tensor,  # [tokens, heads] FP32 attention weights * scale * weight_scale
    ):
        """Allocate storage and dispatch the specialized warps (as the fused kernel)."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        # The cluster's ranks: x is the half of the head (half-head tiles), y the K slice.
        split_rank = cute.arch.make_warp_uniform(block_idx[1] % self.split_k)
        half = Int32(0)
        if cutlass.const_expr(self.pair):
            half = cute.arch.make_warp_uniform(bidx % 2)
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            mailbox = sA
            bar_reduce = bar_mma_epilog

        sAcc = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n)),
            alignment=_EPI_ALIGN_BYTES,
        )
        sRstd = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.cta_n),
            alignment=_EPI_ALIGN_BYTES,
        )
        # The tile's tokens: their cos/sin rows and attention weights, loaded while the mainloop
        # runs.
        sCS = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.cta_n * self.qtail.rope_dim),
            alignment=_EPI_ALIGN_BYTES,
        )
        sW = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.cta_n),
            alignment=_EPI_ALIGN_BYTES,
        )
        if cutlass.const_expr(self.pair):
            # Row amax of this half, and of the other half (written by its owner).
            sAmaxMine = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.cta_n),
                alignment=_EPI_ALIGN_BYTES,
            )
            sAmaxPeer = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.cta_n),
                alignment=_EPI_ALIGN_BYTES,
            )
            bar_amax = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            sAmaxMine = sRstd
            sAmaxPeer = sRstd
            bar_amax = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)
                if cutlass.const_expr(self.pair):
                    # The local arrival plus the other owner's transaction bytes.
                    cute.arch.mbarrier_init(bar_amax, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1 or self.pair):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1 or self.pair):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_qtail(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                half,
                sAcc,
                sRstd,
                sCS,
                sW,
                sAmaxMine,
                sAmaxPeer,
                bar_amax,
                mB,
                mC,
                mPos,
                mCosSin,
                mWIn,
                mQScaleOut,
                mWOut,
                bidx,
                n_idx,
            )

    @cute.experimental.jit
    def epilog_qtail(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        half: cutlass.Int32,
        sAcc: cute.Tensor,
        sRstd: cute.Tensor,
        sCS: cute.Tensor,
        sW: cute.Tensor,
        sAmaxMine: cute.Tensor,
        sAmaxPeer: cute.Tensor,
        bar_amax,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mPos: cute.Tensor,
        mCosSin: cute.Tensor,
        mWIn: cute.Tensor,
        mQScaleOut: cute.Tensor,
        mWOut: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
    ):
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                cutlass.BFloat16,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        head = bidx
        if cutlass.const_expr(self.pair):
            head = bidx // 2
        # The owner's row statistics, cos/sin rows and attention weights overlap the mainloop.
        if split_rank == OWNER_RANK:
            self._row_rstd(mB, sRstd, n_idx, epi_tid)
            self._qtail_prefetch(mB, mPos, mCosSin, mWIn, sCS, sW, half, head, n_idx, epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            # The owner of this tile is the cluster's CTA of the same half at K slice 0.
            owner_rank = half
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        owner_rank,
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            cute_ext.partition_and_copy(thr_t2r, rAcc, sAcc)
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=self.epilog_threads)
            self._store_qtail(
                sAcc,
                sRstd,
                sCS,
                sW,
                sAmaxMine,
                sAmaxPeer,
                bar_amax,
                mC,
                mQScaleOut,
                mWOut,
                bidx,
                half,
                head,
                n_idx,
                epi_tid,
            )

    @cute.experimental.jit
    def _qtail_prefetch(
        self,
        mB: cute.Tensor,
        mPos: cute.Tensor,
        mCosSin: cute.Tensor,
        mWIn: cute.Tensor,
        sCS: cute.Tensor,
        sW: cute.Tensor,
        half: cutlass.Int32,
        head: cutlass.Int32,
        n_idx: cutlass.Int32,
        epi_tid: cutlass.Int32,
    ):
        """sCS[row] = the cos/sin row of the token's position and sW[row] = its attention weight,
        for the tile's tokens, in the tile that holds the head's RoPE columns. Runs after
        ``_row_rstd``, whose grid dependency wait covers the positions and the weights."""
        rope = cutlass.const_expr(self.qtail.rope_dim)
        tpr = cutlass.const_expr(self.epilog_threads // self.cta_n)
        per = cutlass.const_expr(rope // tpr)
        row = epi_tid // tpr
        lane = epi_tid - row * tpr
        token = n_idx * self.cta_n + row
        if token < cute.size(mB, mode=[0]):
            if half == 0:
                pos = mPos[token]
                for i in cutlass.range_constexpr(per):
                    col = lane * per + i
                    sCS[row * rope + col] = mCosSin[pos, col]
                if lane == 0:
                    sW[row] = mWIn[token, head].to(Float32)

    @cute.experimental.jit
    def _store_qtail(
        self,
        sAcc: cute.Tensor,
        sRstd: cute.Tensor,
        sCS: cute.Tensor,
        sW: cute.Tensor,
        sAmaxMine: cute.Tensor,
        sAmaxPeer: cute.Tensor,
        bar_amax,
        mC: cute.Tensor,
        mQScaleOut: cute.Tensor,
        mWOut: cute.Tensor,
        bidx: cutlass.Int32,
        half: cutlass.Int32,
        head: cutlass.Int32,
        n_idx: cutlass.Int32,
        epi_tid: cutlass.Int32,
    ):
        """Finish the staged tile: four consecutive columns of one token row per thread, the
        lanes of a row reducing the head's amax with butterfly shuffles."""
        qt = self.qtail
        rope = cutlass.const_expr(qt.rope_dim)
        rhalf = cutlass.const_expr(qt.rope_dim // 2)
        lanes = cutlass.const_expr(self.lanes_per_row)
        groups_per_thread = cutlass.const_expr(self.cta_m * self.cta_n // 4 // self.epilog_threads)
        n_tok = cute.size(mC, mode=[1])
        col0 = half * self.cta_m
        rows = []
        m4s = []
        vals = []
        amaxes = []
        for g in cutlass.range_constexpr(groups_per_thread):
            gi = epi_tid + g * self.epilog_threads
            row = gi // lanes
            m4 = (gi - row * lanes) * 4
            rstd = sRstd[row]
            # The BF16 wq_b output, as the BF16 store rounds it.
            x01 = _bf16_round(sAcc[m4, row] * rstd, sAcc[m4 + 1, row] * rstd)
            x23 = _bf16_round(sAcc[m4 + 2, row] * rstd, sAcc[m4 + 3, row] * rstd)
            q0 = x01[0]
            q1 = x01[1]
            q2 = x23[0]
            q3 = x23[1]
            col = col0 + m4
            if col < rope:
                # x * cos + (+-partner) * sin as fma(x, cos, rn(+-partner * sin)), then BF16.
                if cutlass.const_expr(qt.interleave):
                    cs = row * rope + col // 2
                    c0 = sCS[cs]
                    s0 = sCS[cs + rhalf]
                    c1 = sCS[cs + 1]
                    s1 = sCS[cs + rhalf + 1]
                    r01 = _bf16_round(
                        _fma_rn(q0, c0, _mul_rn(_neg_f32(q1), s0)),
                        _fma_rn(q1, c0, _mul_rn(q0, s0)),
                    )
                    r23 = _bf16_round(
                        _fma_rn(q2, c1, _mul_rn(_neg_f32(q3), s1)),
                        _fma_rn(q3, c1, _mul_rn(q2, s1)),
                    )
                else:
                    # NeoX: column d pairs with d +- rope / 2; four columns share a side.
                    lower = col < rhalf
                    pm4 = m4 - rhalf
                    cs = row * rope + col - rhalf
                    if lower:
                        pm4 = m4 + rhalf
                        cs = row * rope + col
                    p01 = _bf16_round(sAcc[pm4, row] * rstd, sAcc[pm4 + 1, row] * rstd)
                    p23 = _bf16_round(sAcc[pm4 + 2, row] * rstd, sAcc[pm4 + 3, row] * rstd)
                    p0 = p01[0]
                    p1 = p01[1]
                    p2 = p23[0]
                    p3 = p23[1]
                    if lower:
                        p0 = _neg_f32(p0)
                        p1 = _neg_f32(p1)
                        p2 = _neg_f32(p2)
                        p3 = _neg_f32(p3)
                    r01 = _bf16_round(
                        _fma_rn(q0, sCS[cs], _mul_rn(p0, sCS[cs + rhalf])),
                        _fma_rn(q1, sCS[cs + 1], _mul_rn(p1, sCS[cs + rhalf + 1])),
                    )
                    r23 = _bf16_round(
                        _fma_rn(q2, sCS[cs + 2], _mul_rn(p2, sCS[cs + rhalf + 2])),
                        _fma_rn(q3, sCS[cs + 3], _mul_rn(p3, sCS[cs + rhalf + 3])),
                    )
                q0 = r01[0]
                q1 = r01[1]
                q2 = r23[0]
                q3 = r23[1]
            amax = cute.arch.fmax(
                cute.arch.fmax(abs(q0), abs(q1)), cute.arch.fmax(abs(q2), abs(q3))
            )
            for i in cutlass.range_constexpr(int(math.log2(lanes))):
                amax = cute.arch.fmax(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=lanes >> (i + 1))
                )
            rows.append(row)
            m4s.append(m4)
            vals.append((q0, q1, q2, q3))
            amaxes.append(amax)

        if cutlass.const_expr(self.pair):
            # The other half's row amax: each owner sends its rows' amax to the other owner.
            for g in cutlass.range_constexpr(groups_per_thread):
                if m4s[g] == 0:
                    sAmaxMine[rows[g]] = amaxes[g]
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=self.epilog_threads)
            if epi_tid < self.cta_n // 4:
                v = epi_tid * 4
                _store_shared_remote_v4(
                    sAmaxMine[v],
                    sAmaxMine[v + 1],
                    sAmaxMine[v + 2],
                    sAmaxMine[v + 3],
                    sAmaxPeer.iterator + v,
                    bar_amax,
                    Int32(1) - half,
                )
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(bar_amax, self.cta_n * 4)
            cute.arch.mbarrier_wait(bar_amax, 0)
            for g in cutlass.range_constexpr(groups_per_thread):
                amaxes[g] = cute.arch.fmax(amaxes[g], sAmaxPeer[rows[g]])

        for g in cutlass.range_constexpr(groups_per_thread):
            row = rows[g]
            m4 = m4s[g]
            q0, q1, q2, q3 = vals[g]
            token = n_idx * self.cta_n + row
            if token < n_tok:
                scale = _fp8_ratio(amaxes[g])
                if cutlass.const_expr(qt.ue8m0):
                    scale = _ue8m0_up(scale)
                inv = cute.arch.rcp_approx(scale)
                feature = bidx * self.cta_m + m4
                addr = (mC.iterator + cute.crd2idx((feature, token, 0), mC.layout)).toint()
                _st_global_b32(
                    addr,
                    _cvt_e4m3x4(
                        _mul_rn(q0, inv), _mul_rn(q1, inv), _mul_rn(q2, inv), _mul_rn(q3, inv)
                    ),
                )
                if col0 + m4 == 0:
                    mQScaleOut[token, head] = scale
                    mWOut[token, head] = _mul_rn(_mul_rn(sW[row], scale), Float32(qt.weight_scale))


# ----------------------------------------------------------------------------------- host


@cute.experimental.jit
def _bmm_fused(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    gamma: cute.Tensor,
    norm_out: cute.Tensor,
    kv_lens: cute.Tensor,
    cos_sin: cute.Tensor,
    q_scale: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        gamma,
        norm_out,
        kv_lens,
        cos_sin,
        q_scale,
        stream,
    )


@functools.cache
def _placeholders(device_index: int) -> dict:
    """Stand-ins for the operands an epilogue does not read (one compiled signature)."""
    device = _torch.device("cuda", device_index)
    return {
        "gamma": _torch.zeros(_BF16_PER_VEC, dtype=_torch.bfloat16, device=device),
        "norm_out": _torch.zeros((1, _BF16_PER_VEC), dtype=_torch.bfloat16, device=device),
        "kv_lens": _torch.zeros(4, dtype=_torch.int32, device=device),
        "cos_sin": _torch.zeros(4, dtype=_torch.float32, device=device),
        "q_scale": _torch.ones(4, dtype=_torch.float32, device=device),
    }


def _aux_cute_tensors(gamma, norm_out, kv_lens, cos_sin, q_scale):
    # detach(): a module parameter (the norm weight) cannot be exported through DLPack otherwise.
    return (
        _from_dlpack_dynamic(gamma.detach(), 0, assumed_align=16),
        _from_dlpack_dynamic(norm_out.detach(), 1, assumed_align=16),
        _from_dlpack_dynamic(kv_lens.detach(), 0, assumed_align=4),
        _from_dlpack_dynamic(cos_sin.detach(), 0, assumed_align=4),
        _from_dlpack_dynamic(q_scale.detach(), 0, assumed_align=4),
    )


def _swap_cute_tensors(a, b, out):
    a_swap = b.detach().unsqueeze(0).transpose(-2, -1)
    b_swap = a.detach().unsqueeze(0).transpose(-2, -1)
    c_swap = out.detach().unsqueeze(0).transpose(-2, -1)
    leading_dims = tuple(_detect_leading_dim(t) for t in (a_swap, b_swap, c_swap))
    tensors = tuple(
        _from_dlpack_dynamic(t, ld)
        for t, ld in zip((a_swap, b_swap, c_swap), leading_dims, strict=True)
    )
    return tensors, leading_dims


@functools.cache
def _get_compiled_fused_kernel(
    tactic: SplitKTactic,
    use_pdl: bool,
    epilogue: SplitKEpilogue,
    leading_dims: tuple[int, int, int],
    out_dtype: _torch.dtype,
    reserve_full_smem: bool,
    device_index: int,
):
    kernel = SplitKFusedEpilogueGemmKernel(
        tactic=tactic, use_pdl=use_pdl, epilogue=epilogue, reserve_full_smem=reserve_full_smem
    )
    device = _torch.device("cuda", device_index)
    m, n, k = 8, 64 * 9, _CTA_K * 2
    weight = _torch.empty((n, k), dtype=_torch.bfloat16, device=device)
    act = _torch.empty((m, k), dtype=_torch.bfloat16, device=device)
    out = _torch.empty((m, n), dtype=out_dtype, device=device)
    tensors, repr_leading = _swap_cute_tensors(act, weight.t(), out)
    if repr_leading != leading_dims:
        raise ValueError(f"unsupported operand layouts {leading_dims} (expected {repr_leading})")
    ph = _placeholders(device_index)
    aux = _aux_cute_tensors(
        ph["gamma"], ph["norm_out"], ph["kv_lens"], ph["cos_sin"], ph["q_scale"]
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(device).cuda_stream)
    return cute_ext.compile(_bmm_fused, kernel, *tensors, *aux, stream)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _validate_fused(a, b, out, epilogue, gamma, norm_out, kv_lens, cos_sin, q_scale):
    _check(all(isinstance(t, _torch.Tensor) for t in (a, b, out)), "a, b and out must be tensors")
    _check(a.ndim == 2 and b.ndim == 2 and out.ndim == 2, "a, b and out must be 2D")
    _check(
        a.is_cuda and b.device == a.device and out.device == a.device,
        "a, b and out must be on one CUDA device",
    )
    _check(a.dtype == _torch.bfloat16 and b.dtype == _torch.bfloat16, "a and b must be BF16")
    m, k = a.shape
    _check(b.shape[0] == k, f"incompatible shapes {tuple(a.shape)} x {tuple(b.shape)}")
    n = b.shape[1]
    _check(tuple(out.shape) == (m, n), f"out must be {(m, n)}, got {tuple(out.shape)}")
    _check(1 <= m <= 32 and k % _CTA_K == 0 and n % 4 == 0, f"unsupported shape M={m} N={n} K={k}")
    # a may be a row view of a wider buffer (the raw q_a slice of the fused-A output).
    _check(
        a.stride(1) == 1 and a.stride(0) >= k and (a.stride(0) * 2) % 16 == 0,
        f"a needs unit inner stride and a 16-byte row pitch, got strides {a.stride()}",
    )
    _check(b.t().is_contiguous(), "b must be column-major (the transpose of a contiguous [N, K])")
    _check(out.is_contiguous(), "out must be contiguous")
    _check(all(t.data_ptr() % 32 == 0 for t in (a, b, out)), "a, b and out must be 32-byte aligned")
    if epilogue.fp8_rope:
        _check(out.dtype == _torch.uint8, "fp8_rope writes uint8 (E4M3 bits)")
        _check(
            n % epilogue.head_dim == 0, f"N={n} is not a multiple of head_dim {epilogue.head_dim}"
        )
        _check(m % epilogue.seq_len == 0, f"M={m} is not a multiple of seq_len {epilogue.seq_len}")
        for t, name, dtype in (
            (kv_lens, "kv_lens", _torch.int32),
            (cos_sin, "cos_sin", _torch.float32),
            (q_scale, "q_scale", _torch.float32),
        ):
            _check(
                isinstance(t, _torch.Tensor)
                and t.dtype == dtype
                and t.is_contiguous()
                and t.device == a.device,
                f"fp8_rope needs a contiguous {dtype} {name} on the GEMM's device",
            )
        _check(kv_lens.numel() >= m // epilogue.seq_len, "kv_lens has fewer entries than sequences")
        _check(q_scale.numel() >= 1, "q_scale is empty")
    else:
        _check(out.dtype == _torch.bfloat16, "out must be BF16 without fp8_rope")
    if epilogue.row_rms:
        _check(k == epilogue.rms_k, f"row_rms was specialized for K={epilogue.rms_k}, got {k}")
    if epilogue.rms_out:
        _check(
            isinstance(gamma, _torch.Tensor)
            and gamma.dtype == _torch.bfloat16
            and gamma.is_contiguous()
            and gamma.numel() == k
            and gamma.data_ptr() % 16 == 0,
            "rms_out needs a contiguous 16-byte aligned BF16 gamma of K elements",
        )
        _check(
            isinstance(norm_out, _torch.Tensor)
            and norm_out.dtype == _torch.bfloat16
            and norm_out.is_contiguous()
            and tuple(norm_out.shape) == (m, k)
            and norm_out.data_ptr() % 16 == 0,
            "rms_out needs a contiguous 16-byte aligned BF16 norm_out [M, K]",
        )
    return m, n, k


def run_splitk_fused(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    tactic: SplitKTactic,
    epilogue: SplitKEpilogue,
    *,
    gamma: _torch.Tensor | None = None,
    norm_out: _torch.Tensor | None = None,
    kv_lens: _torch.Tensor | None = None,
    cos_sin: _torch.Tensor | None = None,
    q_scale: _torch.Tensor | None = None,
    reserve_full_smem: bool = True,
) -> _torch.Tensor:
    """Run ``a[M, K] @ b[K, N]`` with the split-K kernel and a fused epilogue into ``out``.

    Args:
        a: Activation [M, K] BF16 with unit inner stride (may be a row view of a wider
            buffer). With ``epilogue.row_rms`` it is the raw RMSNorm input.
        b: Weight [K, N], column-major (``W.t()`` of a contiguous [N, K]); with ``row_rms``
            the RMSNorm weight is already folded in.
        out: [M, N] contiguous: BF16, or uint8 (E4M3 bits) with ``epilogue.fp8_rope``.
        pdl: Launch with Programmatic Dependent Launch.
        tactic: The split-K tactic; its A/B stages are reduced to fit the epilogue buffers.
        epilogue: The fused epilogue.
        gamma, norm_out: ``rms_out`` operands (BF16 [K], BF16 [M, K]).
        kv_lens, cos_sin, q_scale: ``fp8_rope`` operands (int32 KV lengths per sequence, the
            FP32 (cos, sin) table of ``mla_rope_generation``, FP32 [1] scale).
        reserve_full_smem: Reserve the full shared-memory capacity per CTA.
    """
    m, n, k = _validate_fused(a, b, out, epilogue, gamma, norm_out, kv_lens, cos_sin, q_scale)
    tactic = fused_epilogue_tactic(tactic)
    validate_tactic(tactic, m, n, k)
    _check(
        _fused_smem_bytes(tactic, tactic.ab_stages) <= _SMEM_CAPACITY_BYTES,
        f"tactic {tactic} does not fit the epilogue buffers",
    )
    tensors, leading_dims = _swap_cute_tensors(a, b, out)
    device_index = a.device.index if a.device.index is not None else _torch.cuda.current_device()
    compiled = _get_compiled_fused_kernel(
        tactic, pdl, epilogue, leading_dims, out.dtype, reserve_full_smem, device_index
    )
    ph = _placeholders(device_index)
    aux = _aux_cute_tensors(
        gamma if epilogue.rms_out else ph["gamma"],
        norm_out if epilogue.rms_out else ph["norm_out"],
        kv_lens if epilogue.fp8_rope else ph["kv_lens"],
        cos_sin.view(-1) if epilogue.fp8_rope else ph["cos_sin"],
        q_scale if epilogue.fp8_rope else ph["q_scale"],
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*tensors, *aux, stream)
    return out


@cute.experimental.jit
def _bmm_qtail(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    positions: cute.Tensor,
    cos_sin: cute.Tensor,
    weights_in: cute.Tensor,
    q_scale_out: cute.Tensor,
    weights_out: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        positions,
        cos_sin,
        weights_in,
        q_scale_out,
        weights_out,
        stream,
    )


def _qtail_cute_tensors(positions, cos_sin, weights_in, q_scale_out, weights_out):
    return (
        _from_dlpack_dynamic(positions.detach(), 0, assumed_align=4),
        _from_dlpack_dynamic(cos_sin.detach(), 1, assumed_align=4),
        _from_dlpack_dynamic(weights_in.detach(), 1, assumed_align=2),
        _from_dlpack_dynamic(q_scale_out.detach(), 1, assumed_align=4),
        _from_dlpack_dynamic(weights_out.detach(), 1, assumed_align=4),
    )


@functools.cache
def _get_compiled_qtail_kernel(
    tactic: SplitKTactic,
    use_pdl: bool,
    epilogue: SplitKEpilogue,
    qtail: IndexerQTail,
    leading_dims: tuple[int, int, int],
    reserve_full_smem: bool,
    device_index: int,
):
    kernel = SplitKIndexerQTailGemmKernel(
        tactic=tactic,
        use_pdl=use_pdl,
        epilogue=epilogue,
        qtail=qtail,
        reserve_full_smem=reserve_full_smem,
    )
    device = _torch.device("cuda", device_index)
    m, n, k = 8, qtail.n_heads * qtail.head_dim, epilogue.rms_k
    weight = _torch.empty((n, k), dtype=_torch.bfloat16, device=device)
    act = _torch.empty((m, k), dtype=_torch.bfloat16, device=device)
    out = _torch.empty((m, n), dtype=_torch.uint8, device=device)
    tensors, repr_leading = _swap_cute_tensors(act, weight.t(), out)
    if repr_leading != leading_dims:
        raise ValueError(f"unsupported operand layouts {leading_dims} (expected {repr_leading})")
    aux = _qtail_cute_tensors(
        _torch.zeros(m, dtype=_torch.int32, device=device),
        _torch.zeros((m, qtail.rope_dim), dtype=_torch.float32, device=device),
        _torch.zeros((m, qtail.n_heads), dtype=_torch.bfloat16, device=device),
        _torch.zeros((m, qtail.n_heads), dtype=_torch.float32, device=device),
        _torch.zeros((m, qtail.n_heads), dtype=_torch.float32, device=device),
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(device).cuda_stream)
    return cute_ext.compile(_bmm_qtail, kernel, *tensors, *aux, stream)


def _validate_qtail(
    a, b, out, epilogue, qtail, positions, cos_sin, weights_in, q_scale_out, weights_out
):
    _check(all(isinstance(t, _torch.Tensor) for t in (a, b, out)), "a, b and out must be tensors")
    _check(a.ndim == 2 and b.ndim == 2 and out.ndim == 2, "a, b and out must be 2D")
    _check(
        a.is_cuda and b.device == a.device and out.device == a.device,
        "a, b and out must be on one CUDA device",
    )
    _check(a.dtype == _torch.bfloat16 and b.dtype == _torch.bfloat16, "a and b must be BF16")
    m, k = a.shape
    _check(b.shape[0] == k, f"incompatible shapes {tuple(a.shape)} x {tuple(b.shape)}")
    n = b.shape[1]
    _check(tuple(out.shape) == (m, n), f"out must be {(m, n)}, got {tuple(out.shape)}")
    _check(1 <= m <= 32 and k % _CTA_K == 0, f"unsupported shape M={m} N={n} K={k}")
    _check(
        a.stride(1) == 1 and a.stride(0) >= k and (a.stride(0) * 2) % 16 == 0,
        f"a needs unit inner stride and a 16-byte row pitch, got strides {a.stride()}",
    )
    _check(b.t().is_contiguous(), "b must be column-major (the transpose of a contiguous [N, K])")
    _check(out.is_contiguous() and out.dtype == _torch.uint8, "out must be contiguous uint8")
    _check(all(t.data_ptr() % 32 == 0 for t in (a, b, out)), "a, b and out must be 32-byte aligned")
    _check(
        epilogue.row_rms and not epilogue.rms_out and not epilogue.fp8_rope,
        "the indexer Q tail follows the row_rms epilogue alone",
    )
    _check(k == epilogue.rms_k, f"row_rms was specialized for K={epilogue.rms_k}, got {k}")
    qtail.validate()
    _check(
        n == qtail.n_heads * qtail.head_dim,
        f"N={n} is not {qtail.n_heads} heads of {qtail.head_dim}",
    )
    _check(
        isinstance(positions, _torch.Tensor)
        and positions.dtype == _torch.int32
        and positions.dim() == 1
        and positions.is_contiguous()
        and positions.numel() >= m,
        "positions must be a contiguous int32 vector of at least M entries",
    )
    _check(
        isinstance(cos_sin, _torch.Tensor)
        and cos_sin.dtype == _torch.float32
        and cos_sin.dim() == 2
        and cos_sin.stride(1) == 1
        and cos_sin.shape[1] >= qtail.rope_dim,
        "cos_sin must be FP32 [positions, >= rope_dim] with unit inner stride",
    )
    _check(
        isinstance(weights_in, _torch.Tensor)
        and weights_in.dtype == _torch.bfloat16
        and tuple(weights_in.shape) == (m, qtail.n_heads)
        and weights_in.stride(1) == 1,
        "weights_in must be BF16 [M, n_heads] with unit inner stride",
    )
    for t, name in ((q_scale_out, "q_scale_out"), (weights_out, "weights_out")):
        _check(
            isinstance(t, _torch.Tensor)
            and t.dtype == _torch.float32
            and tuple(t.shape) == (m, qtail.n_heads)
            and t.is_contiguous(),
            f"{name} must be a contiguous FP32 [M, n_heads]",
        )
    for t in (positions, cos_sin, weights_in, q_scale_out, weights_out):
        _check(t.device == a.device, "every operand must be on the GEMM's device")
    return m, n, k


def run_splitk_idx_qtail(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    tactic: SplitKTactic,
    epilogue: SplitKEpilogue,
    qtail: IndexerQTail,
    *,
    positions: _torch.Tensor,
    cos_sin: _torch.Tensor,
    weights_in: _torch.Tensor,
    q_scale_out: _torch.Tensor,
    weights_out: _torch.Tensor,
    reserve_full_smem: bool = True,
) -> _torch.Tensor:
    """The DSA indexer wq_b ``a[M, K] @ b[K, N]`` with the row-RMS epilogue and the indexer Q tail.

    Args:
        a: The raw q_a [M, K] BF16 with unit inner stride (may be a row view of a wider buffer).
        b: The norm-folded indexer wq_b [K, N], column-major (``W.t()`` of a contiguous [N, K]).
        out: [M, N] uint8: the E4M3 indexer Q, [tokens, heads * head_dim].
        pdl: Launch with Programmatic Dependent Launch.
        tactic: The split-K tactic: ``mma_m`` is one head or half a head; its A/B stages are
            reduced to fit the epilogue buffers.
        epilogue: ``row_rms`` alone.
        qtail: The Q tail.
        positions: int32 [>= M] RoPE positions of the tokens.
        cos_sin: FP32 [positions, >= rope_dim] cos/sin rows ([cos | sin] halves).
        weights_in: BF16 [M, n_heads] attention weights (the weights_proj output; may be a column
            view of a wider buffer).
        q_scale_out: FP32 [M, n_heads] output: the E4M3 scales.
        weights_out: FP32 [M, n_heads] output: weights_in * scale * qtail.weight_scale.
        reserve_full_smem: Reserve the full shared-memory capacity per CTA.
    """
    m, n, k = _validate_qtail(
        a, b, out, epilogue, qtail, positions, cos_sin, weights_in, q_scale_out, weights_out
    )
    tactic = idx_qtail_tactic(tactic, qtail)
    validate_tactic(tactic, m, n, k)
    _check(
        _qtail_smem_bytes(tactic, tactic.ab_stages, qtail) <= _SMEM_CAPACITY_BYTES,
        f"tactic {tactic} does not fit the indexer Q tail buffers",
    )
    tensors, leading_dims = _swap_cute_tensors(a, b, out)
    device_index = a.device.index if a.device.index is not None else _torch.cuda.current_device()
    compiled = _get_compiled_qtail_kernel(
        tactic, pdl, epilogue, qtail, leading_dims, reserve_full_smem, device_index
    )
    aux = _qtail_cute_tensors(positions, cos_sin, weights_in, q_scale_out, weights_out)
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*tensors, *aux, stream)
    return out
