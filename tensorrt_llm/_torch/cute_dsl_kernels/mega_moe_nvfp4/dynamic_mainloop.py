# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Dynamic UMMA N for fused fc1+fc2 swap-AB MegaMoE kernel.
#
# Emits raw ``nvvm.tcgen05_mma_block_scale(...)`` so the instruction descriptor
# (``idesc``) is under our control -- specifically so its ``n_dim_`` bitfield
# becomes a runtime SSA value (= ``align16(valid_tokens_in_tile) >> 3``).

from typing import Optional

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin, llvm
from cutlass._mlir.dialects import nvvm as _nvvm_raw

# auto-Int32/Boolean -> ir.Value wrapper for nvvm.* calls.
from cutlass.cute.arch.nvvm_wrappers import nvvm
from cutlass.cutlass_dsl import Boolean, Int32, dsl_user_op

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
#   bit [ 7,10) : a_format_        (1 for NVFP4 mxf4nvf4)
#   bit [10,13) : b_format_        (1 for NVFP4 mxf4nvf4)
#   bit [13,14) : a_negate_        (0)
#   bit [14,15) : b_negate_        (0)
#   bit [15,16) : a_major_         (0 = K-major)
#   bit [16,17) : b_major_         (0 = K-major)
#   bit [17,23) : n_dim_           (runtime, OR'd in: align16(valid) >> 3)
#   bit [23,24) : scale_format_    (0 = E4M3 SF)
#   bit [24,29) : m_dim_           (M >> 4: 256 -> 16, 128 -> 8)
#   bit [29,31) : a_sf_id_         (runtime, OR'd in)
#   bit [31,32) : k_size_          (0 for NVFP4 K=64)

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

# tcgen05.mma atom K-size for NVFP4 (UMMA_K = 64 for mxf4nvf4).
_UMMA_K_NVFP4 = 64


def build_static_idesc_base(
    *,
    umma_m: int,  # 64, 128, or 256
    a_format: int = 1,  # NVFP4 mxf4nvf4
    b_format: int = 1,
    a_major: int = 0,  # 0 = K-major
    b_major: int = 0,
    scale_format: int = 0,  # 0 = E4M3 SF
    k_size_bit: int = 0,  # 0 for NVFP4 (K=64)
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
    idesc = idesc | (sfa_top >> Int32(30 - _BIT_A_SF_ID))
    idesc = idesc | (sfb_top >> Int32(30 - _BIT_B_SF_ID))
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


def _i32_to_tmem_llvm_ptr(i32_value: ir.Value) -> ir.Value:
    """Cast i32 TMEM address -> ``!llvm.ptr<6>``."""
    tmem_llvm_ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    return llvm.inttoptr(tmem_llvm_ptr_ty, i32_value)


def _as_value(it) -> ir.Value:
    """Unwrap to underlying ir.Value (cute Pointer has .value)."""
    return it.value if hasattr(it, "value") else it


# =============================================================================
# Main entry: 1 K-tile = (mma_tiler_k / UMMA_K) inner-K MMAs
# =============================================================================


@dsl_user_op
def issue_dynamic_block_scaled_mma_tile(
    *,
    # Acc cute tensor (TMEM cute.ptr-typed iterator).  Carries the full
    # per-task-tile TMEM region; K-axis advance is encoded in its layout.
    acc_tensor,
    # A/B fragment tensors at current AB pipeline stage (smem_desc iterators),
    # shape (V, M_count, K_count) where K_count = mma_tiler_k / UMMA_K.
    a_frag_tile,
    b_frag_tile,
    # SFA / SFB cute tensors (TMEM cute.ptr-typed iterators).
    sfa_tensor,
    sfb_tensor,
    # Outer K-tile index (Int32 SSA).  Drives accumulate flag.
    k_tile_idx,
    # Logical valid token count for this tile (Int32 SSA).  Rounded up to 16
    # before encoding into idesc.n_dim_.
    valid_tokens_in_tile,
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
    static_idesc_base = build_static_idesc_base(umma_m=mma_tiler_mnk[0])

    # Runtime n_dim_ field: align16(valid) >> 3.
    n_dim_value = _align16(valid_tokens_in_tile) >> Int32(3)

    num_k_inner = mma_tiler_mnk[2] // _UMMA_K_NVFP4  # = 4 for NVFP4 K=256

    compatible_to_old_nvvm = False
    if hasattr(_nvvm_raw, "Tcgen05GroupKind"):
        compatible_to_old_nvvm = True

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

        # Cast operands to NVVM-op-acceptable types.
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

        operand_d_ptr = _i32_to_tmem_llvm_ptr(operand_acc_i32)
        operand_sfa_ptr = _i32_to_tmem_llvm_ptr(operand_sfa_i32)
        operand_sfb_ptr = _i32_to_tmem_llvm_ptr(operand_sfb_i32)

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
            if compatible_to_old_nvvm:
                nvvm_args = {
                    "mma_kind": _nvvm_raw.Tcgen05MMAKind.MXF4NVF4,
                    "cta_group": _nvvm_raw.Tcgen05GroupKind.CTA_2
                    if mma_tiler_mnk[0] == 256
                    else _nvvm_raw.Tcgen05GroupKind.CTA_1,
                    "d": operand_d_ptr,
                    "a": operand_a,
                    "b": operand_b,
                    "idesc": idesc.ir_value(),
                    "enable_input_d": Boolean(accum_flag).ir_value(),
                    "scale_a": operand_sfa_ptr,
                    "scale_b": operand_sfb_ptr,
                    "scale_vec_size": _nvvm_raw.Tcgen05MMAScaleVecSize.X4,
                }
            else:
                nvvm_args = {
                    "kind": _nvvm_raw.Tcgen05MMAKind.MXF4NVF4,
                    "cta_group": _nvvm_raw.CTAGroupKind.CTA_2
                    if mma_tiler_mnk[0] == 256
                    else _nvvm_raw.CTAGroupKind.CTA_1,
                    "matrix_d": operand_d_ptr,
                    "matrix_a": operand_a,
                    "matrix_b": operand_b,
                    "idesc": idesc.ir_value(),
                    "enable_input_d": Boolean(accum_flag).ir_value(),
                    "scale_a": operand_sfa_ptr,
                    "scale_b": operand_sfb_ptr,
                    "block_scale": _nvvm_raw.Tcgen05MMABlockScale.BLOCK16,
                }
            nvvm.tcgen05_mma_block_scale(**nvvm_args)
