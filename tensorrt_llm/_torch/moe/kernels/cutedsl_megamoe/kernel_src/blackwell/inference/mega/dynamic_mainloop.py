# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamic UMMA N for fused fc1+fc2 swap-AB MegaMoE kernel.
#
# Emits the block-scaled UMMA instruction as literal PTX text (via
# ``llvm.inline_asm``) so the instruction descriptor (``idesc``) is under our
# control -- specifically so its ``n_dim_`` bitfield becomes a runtime SSA
# value (= ``align16(valid_tokens_in_tile) >> 3``).
#
# Why PTX text instead of ``cutlass._mlir.dialects.nvvm`` (the dialect Python
# binding): that binding is a private/unstable surface that has already
# broken once across a cutedsl upgrade (enum classes and kwargs renamed:
# ``Tcgen05GroupKind``/``mma_kind``/``Tcgen05MMAScaleVecSize.X4`` ->
# ``CTAGroupKind``/``kind``/``Tcgen05MMABlockScale.BLOCK16``). The PTX ISA
# text syntax for ``tcgen05.mma...block_scale`` has been stable across CUDA
# 12.9-13.3 (PTX ISA 8.8-9.3): the only change in that window was the
# addition of the ``.block16``/``.block32`` aliases in 8.8, and the old
# spelling was never removed. This module targets CUDA >= 13 and emits the
# ``.block``-suffixed spelling.
#
# Three MMA kinds reach this module, selected by ``QuantKind.umma_kind``:
#
#   kind::mxf4nvf4.block_scale.block16   nvfp4          UMMA_K = 64
#   kind::mxf4.block_scale.block32       mxfp4          UMMA_K = 64
#   kind::mxf8f6f4.block_scale           mxfp8, mixed   UMMA_K = 32
#
# They differ only in the mnemonic: operand order, operand types and the
# constraint string are identical across all three (verified against CUTLASS
# ``cute/arch/mma_sm100_umma.hpp``: SM100_MMA_MXF4_SS at :4738 and
# SM100_MMA_MXF8F6F4_SS at :1573 emit the same
# ``[%0], %1, %2, %3, [%5], [%6], p`` with ``"r,l,l,r,r,r,r"``). Note that
# mxf8f6f4 takes NO scale-vec suffix -- its scale vector size is implicitly 32.

from typing import Optional

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin, llvm
from cutlass.cutlass_dsl import Boolean, Int32, dsl_user_op

from .....quant_def import QuantKind

# =============================================================================
# Alignment policy (single source of truth)
# =============================================================================


def _align16(x):
    """Round Int32 SSA ``x`` up to a multiple of 16 (mask off bottom 4 bits)."""
    return (Int32(x) + Int32(15)) & Int32(-16)


@dsl_user_op
def compute_non_leader_cta_load_shift(
    *,
    valid_tokens_in_tile,  # Int32 SSA
    mma_tiler_n: int,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """Token offset that non-leader CTA's TMA-B read must shift by under 2cta.

    Under dynamic UMMA + 2cta:
      - MMA splits N at align16(valid) / 2
      - TMA static partition splits at mma_tiler_n / 2

    The result ∈ (-mma_tiler_n/2, 0]; apply via
    ``cute.domain_offset((shift, 0, 0), real_b)`` on non-leader CTA only.
    """
    return (_align16(valid_tokens_in_tile) >> Int32(1)) - Int32(mma_tiler_n // 2)


# =============================================================================
# Static idesc base builder (compile-time Python int)
# =============================================================================
#
# Bit layout of the block-scaled instruction descriptor:
#
#   bit [ 0, 2) : sparse_id2_      (0)
#   bit [ 2, 3) : sparse_flag_     (0 = dense)
#   bit [ 3, 4) : saturate_        (0)
#   bit [ 4, 6) : b_sf_id_         (runtime, OR'd in)
#   bit [ 6, 7) : sparse_format_   (0)
#   bit [ 7,10) : a_format_        (per QuantKind.weight_format_code)
#   bit [10,13) : b_format_        (per QuantKind.activation_format_code)
#   bit [13,14) : a_negate_        (0)
#   bit [14,15) : b_negate_        (0)
#   bit [15,16) : a_major_         (0 = K-major)
#   bit [16,17) : b_major_         (0 = K-major)
#   bit [17,23) : n_dim_           (runtime, OR'd in: align16(valid) >> 3)
#   bit [23,24) : scale_format_    (0 = UE4M3 SF, 1 = UE8M0 SF)
#   bit [24,29) : m_dim_           (M >> 4: 256 -> 16, 128 -> 8)
#   bit [29,31) : a_sf_id_         (runtime, OR'd in)
#   bit [31,32) : k_size_          (0 for every kind we build: mxf8f6f4 has no
#                                   other legal value, and MXF4Format's 1 means
#                                   the sm103-only K=96 Ultra variant)

_BIT_A_FORMAT = 7  # width 3
_BIT_B_FORMAT = 10  # width 3
_BIT_A_MAJOR = 15  # width 1
_BIT_B_MAJOR = 16  # width 1
_BIT_N_DIM = 17  # width 6
_BIT_SCALE_FORMAT = 23  # width 1
_BIT_M_DIM = 24  # width 5
_BIT_A_SF_ID = 29  # width 2
_BIT_B_SF_ID = 4  # width 2
_BIT_K_SIZE = 31  # width 1


def build_static_idesc_base(
    *,
    umma_m: int,  # 64, 128, or 256
    a_format: int,
    b_format: int,
    scale_format: int,
    a_major: int = 0,  # 0 = K-major
    b_major: int = 0,
    k_size_bit: int = 0,
) -> int:
    """Pack the static-field portion of the idesc into a u32.

    Runtime fields (n_dim_, a_sf_id_, b_sf_id_, a_negate_, b_negate_) are left
    at zero; the call site OR's them in.  For UMMA_M=256 / NVFP4 / K-major /
    E4M3 SF the base is ``0x10000480``; for UMMA_M=128 it is ``0x08000480``.
    """
    assert umma_m in (64, 128, 256), f"Unsupported UMMA_M={umma_m}"
    assert 0 <= a_format < (1 << 3)
    assert 0 <= b_format < (1 << 3)
    assert 0 <= scale_format < (1 << 1)

    m_dim = umma_m >> 4  # 256 -> 16, 128 -> 8, 64 -> 4

    desc = 0
    desc |= (a_format & 0x7) << _BIT_A_FORMAT
    desc |= (b_format & 0x7) << _BIT_B_FORMAT
    desc |= (a_major & 0x1) << _BIT_A_MAJOR
    desc |= (b_major & 0x1) << _BIT_B_MAJOR
    desc |= (scale_format & 0x1) << _BIT_SCALE_FORMAT
    desc |= (m_dim & 0x1F) << _BIT_M_DIM
    desc |= (k_size_bit & 0x1) << _BIT_K_SIZE
    return desc & 0xFFFFFFFF


# =============================================================================
# Runtime idesc finalization (per-MMA-call OR steps)
# =============================================================================


@dsl_user_op
def compute_idesc(
    *,
    static_base: int,  # from build_static_idesc_base()
    n_dim_value,  # Int32 SSA (= align16(valid_tokens_in_tile) >> 3)
    sfa_tmem_addr_i32,  # i32 SSA -- runtime SF-A TMEM address
    sfb_tmem_addr_i32,  # i32 SSA -- runtime SF-B TMEM address
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """OR runtime fields (n_dim_, a_sf_id_, b_sf_id_) into the static base."""
    idesc = Int32(static_base) | (Int32(n_dim_value) << _BIT_N_DIM)
    sfa_top = Int32(sfa_tmem_addr_i32) & Int32(0xC0000000)
    sfb_top = Int32(sfb_tmem_addr_i32) & Int32(0xC0000000)
    # SF address top 2 bits -> idesc.{a,b}_sf_id_ slots.
    sfa_id = (sfa_top >> Int32(30 - _BIT_A_SF_ID)) & Int32(0x3 << _BIT_A_SF_ID)
    sfb_id = (sfb_top >> Int32(30 - _BIT_B_SF_ID)) & Int32(0x3 << _BIT_B_SF_ID)
    idesc = idesc | sfa_id
    idesc = idesc | sfb_id
    return idesc


# =============================================================================
# Type-cast helpers
# =============================================================================


def _smem_desc_to_i64(smem_desc_value: ir.Value) -> ir.Value:
    """Bit-cast cute_nvgpu.smem_desc value -> i64."""
    i64_ty = ir.IntegerType.get_signless(64)
    return builtin.unrealized_conversion_cast([i64_ty], [smem_desc_value])


def _tmem_ptr_to_i32(tmem_ptr_value: ir.Value) -> ir.Value:
    """Bit-cast cute.ptr<tmem> -> i32."""
    i32_ty = ir.IntegerType.get_signless(32)
    return builtin.unrealized_conversion_cast([i32_ty], [tmem_ptr_value])


def _as_value(it) -> ir.Value:
    """Unwrap to underlying ir.Value (cute Pointer has .value)."""
    return it.value if hasattr(it, "value") else it


# =============================================================================
# PTX-text MMA emission (see module docstring for why this isn't the nvvm
# dialect binding)
# =============================================================================


@dsl_user_op
def _tcgen05_mma_block_scaled(
    *,
    cta_group: int,  # 1 or 2 (Python int, folds into the asm mnemonic)
    umma_kind: str,  # "mxf4nvf4" | "mxf4" | "mxf8f6f4"
    scale_vec_suffix: str,  # ".block16" | ".block32" | "" (mxf8f6f4)
    d_tmem_i32,  # ir.Value (i32) -- accumulator TMEM address
    a_desc_i64,  # ir.Value (i64) -- A operand smem descriptor
    b_desc_i64,  # ir.Value (i64) -- B operand smem descriptor
    idesc_i32,  # ir.Value (i32) -- instruction descriptor
    enable_input_d_i32,  # ir.Value (i32) -- 0/1, D = A@B (+ C if nonzero)
    sfa_tmem_i32,  # ir.Value (i32) -- SFA TMEM address
    sfb_tmem_i32,  # ir.Value (i32) -- SFB TMEM address
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Emit ``tcgen05.mma.cta_group::{1,2}.kind::<umma_kind>.block_scale<suffix>``.

    Mirrors CUTLASS's own reference lowering for these instructions
    (``cute/arch/mma_sm100_umma.hpp``, ``SM100_MMA_MXF4_2x1SM_SS::fma`` and
    ``SM100_MMA_MXF8F6F4_SS::fma``): ``enable_input_d`` travels as a plain u32
    register and is turned into the hardware predicate in-asm via
    ``setp.ne.b32``, so no operand needs a dialect-specific predicate type.
    """
    assert cta_group in (1, 2), f"cta_group must be 1 or 2, got {cta_group}"
    assert umma_kind in ("mxf4nvf4", "mxf4", "mxf8f6f4"), f"Unsupported UMMA kind {umma_kind!r}"
    assert scale_vec_suffix in ("", ".block16", ".block32"), (
        f"Unsupported suffix {scale_vec_suffix!r}"
    )
    llvm.inline_asm(
        None,
        [
            d_tmem_i32,
            a_desc_i64,
            b_desc_i64,
            idesc_i32,
            enable_input_d_i32,
            sfa_tmem_i32,
            sfb_tmem_i32,
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, $4, 0;\n\t"
        f"tcgen05.mma.cta_group::{cta_group}.kind::{umma_kind}.block_scale{scale_vec_suffix} "
        "[$0], $1, $2, $3, [$5], [$6], p;\n\t"
        "}\n",
        "r,l,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# Main entry: 1 K-tile = (mma_tiler_k / UMMA_K) inner-K MMAs
# =============================================================================


@dsl_user_op
def issue_dynamic_block_scaled_mma_tile(
    *,
    # Selects the MMA kind, the idesc format fields and UMMA_K.
    quant_kind: QuantKind,
    # Logical shape: (V, MMA_M, MMA_N). TMEM tensor for one accumulator stage.
    acc_tensor: cute.Tensor,
    # A/B fragment tensors at current AB pipeline stage (smem_desc iterators),
    # logical shape (V, MMA_M|N, MMA_K).
    a_frag_tile: cute.Tensor,
    b_frag_tile: cute.Tensor,
    # Logical shape: (V, MMA_M|N, MMA_K). TMEM scale-factor tensors.
    sfa_tensor: cute.Tensor,
    sfb_tensor: cute.Tensor,
    # Outer K-tile index (Int32 SSA).  Drives accumulate flag.
    k_tile_idx: Int32,
    # Logical valid token count for this tile (Int32 SSA).  Rounded up to 16
    # before encoding into idesc.n_dim_.
    valid_tokens_in_tile: Int32,
    # ``cta_group`` (1 vs 2) is inferred from M: 256 -> 2cta, 128 -> 1cta
    # (kernel constraint per_cta_m == 128).
    mma_tiler_mnk: tuple = (256, 256, 256),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue (mma_tiler_k / UMMA_K) block-scaled MMAs for one K-tile.

    ``idesc.n_dim_ = align16(valid_tokens_in_tile) >> 3``.

    For 2cta MMA, HW's ``cta_group::2`` semantics encode ``n_dim_ * 8`` as the
    cluster-total N (CTA0 + CTA1 combined).  Each CTA writes half in its own
    TMEM.  The caller aligns the per-CTA TMA load offset with HW's split via
    a ``cute.domain_offset`` on non-leader CTA's GMEM tensor, so we just
    encode the cluster-total directly here.

    Pipeline / mbarrier / TMEM alloc / arrive are owned by the caller.
    """
    # Compile-time-fold base idesc from the per-config static fields.
    static_idesc_base = build_static_idesc_base(
        umma_m=mma_tiler_mnk[0],
        a_format=quant_kind.weight_format_code,
        b_format=quant_kind.activation_format_code,
        scale_format=quant_kind.scale_format_code,
    )

    # Runtime n_dim_ field: align16(valid) >> 3.
    n_dim_value = _align16(valid_tokens_in_tile) >> Int32(3)

    # 4 for nvfp4/mxfp4 at K=256, 4 for mxfp8 at K=128, 8 for mxfp8 at K=256.
    num_k_inner = mma_tiler_mnk[2] // quant_kind.instruction_k("1x")

    # 256 -> 2cta, 128 -> 1cta (kernel constraint per_cta_m == 128).
    cta_group = 2 if mma_tiler_mnk[0] == 256 else 1

    # m / n inner indices both 0 for v1 (m_count = n_count = 1).
    m_inner = 0
    n_inner = 0

    for k_inner in range(num_k_inner):  # Python int -> compile-time unroll
        a_atom = a_frag_tile[(None, m_inner, k_inner)]
        b_atom = b_frag_tile[(None, n_inner, k_inner)]
        # SF id per-K-iter is encoded in SF TMEM address top bits; the
        # per-k_inner slice advances those bits via the SF layout.
        sfa_atom = sfa_tensor[(None, m_inner, k_inner)]
        sfb_atom = sfb_tensor[(None, n_inner, k_inner)]
        acc_atom = acc_tensor[(None, m_inner, n_inner)]

        # Cast operands to plain integer registers (PTX operand types).
        a_iter_val = _as_value(a_atom.iterator)
        b_iter_val = _as_value(b_atom.iterator)
        acc_iter_val = _as_value(acc_atom.iterator)
        sfa_iter_val = _as_value(sfa_atom.iterator)
        sfb_iter_val = _as_value(sfb_atom.iterator)

        operand_a = _smem_desc_to_i64(a_iter_val)
        operand_b = _smem_desc_to_i64(b_iter_val)
        operand_sfa_i32 = _tmem_ptr_to_i32(sfa_iter_val)
        operand_sfb_i32 = _tmem_ptr_to_i32(sfb_iter_val)
        operand_acc_i32 = _tmem_ptr_to_i32(acc_iter_val)

        idesc = compute_idesc(
            static_base=static_idesc_base,
            n_dim_value=n_dim_value,
            sfa_tmem_addr_i32=operand_sfa_i32,
            sfb_tmem_addr_i32=operand_sfb_i32,
        )

        # Accumulate flag: True except for the very first iter
        # (k_tile_idx == 0 AND k_inner == 0).
        if k_inner == 0:
            accum_flag = k_tile_idx != 0
        else:
            accum_flag = True

        with cute.arch.elect_one():
            _tcgen05_mma_block_scaled(
                cta_group=cta_group,
                umma_kind=quant_kind.umma_kind,
                scale_vec_suffix=quant_kind.umma_scale_vec_suffix,
                d_tmem_i32=operand_acc_i32,
                a_desc_i64=operand_a,
                b_desc_i64=operand_b,
                idesc_i32=idesc.ir_value(),
                enable_input_d_i32=Int32(Boolean(accum_flag)).ir_value(),
                sfa_tmem_i32=operand_sfa_i32,
                sfb_tmem_i32=operand_sfb_i32,
            )
