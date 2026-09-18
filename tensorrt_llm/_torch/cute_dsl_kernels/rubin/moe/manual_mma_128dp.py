# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is copied from the dynamic-kernel-generator repository
# (cutlass_ir/compiler/python/examples/CuTeDSL/cute/rubin/kernel/moe/manual_mma_128dp.py,
#  branch junliu/fc12-mxfp8 on top of MR !27225, commit a956f6f6039).
# No local modifications.

# Hand-encoded Rubin (sm107) block-scaled UTCOMMA for the non-swapAB gather fc1
# MoE kernel.  Emits ``tcgen05.mma...kind::mxf4nvf4.block_scale.block16`` (NVFP4)
# or ``tcgen05.mma...kind::mxf8f6f4.block_scale`` (MXFP8) as literal PTX
# (via ``llvm.inline_asm``) so the instruction descriptor (``idesc``) is fully
# under our control -- specifically so bit 26 (SFA layout: 0 = SFA_32dp_4xCopy,
# 1 = SFA_128dp_Unique) can be set.
#
# Mechanism copied from megaMOE swapAB ``dynamic_mainloop.py`` (the user's
# confirmed Rubin-correct reference).  Differences here (non-swapAB fc1):
#   * A = tokens (M), B = weights (N).  n_dim is STATIC (= mma_inst_n >> 3),
#     folded into the static idesc base -- no dynamic-token n_dim.
#   * bit 26 (sfa_layout) is exposed as a build-time knob.
#   * Per-atom entry (kernel already loops k-blocks externally), single MMA
#     issued per call.
#
# idesc bit layout (SM107 OMMA table, cute/arch/mma_sm107_desc.hpp). The two
# block-scaled kinds share every field except the operand-format widths and
# the K-size encoding:
#   bit [ 3]     k_size upper            (mxf4nvf4 only)
#   bit [ 4, 6)  b_sf_id   (runtime, OR'd from SFB tmem addr top bits)
#   bit [ 7,10)  a_format  (mxf4nvf4: 1 = E2M1; mxf8f6f4: 0 = E4M3, 1 = E5M2)
#   bit [10,12)  b_format  (mxf4nvf4, 2 bits: 1 = E2M1)
#   bit [10,13)  b_format  (mxf8f6f4, 3 bits: 0 = E4M3, 1 = E5M2)
#   bit [15]     a_major   (0 = K-major)
#   bit [16]     b_major   (0 = K-major)
#   bit [17,23)  n_dim     (N >> 3, static for non-swapAB)
#   bit [23,25)  scale_format (mxf4nvf4: 0 = UE4M3, 1 = UE8M0; mxf8f6f4: 1 = UE8M0)
#   bit [26]     sfa_layout (0 = 32dp_4xCopy, 1 = 128dp_Unique)  <-- the knob
#   bit [27,29)  m_dim     (M >> 7)
#   bit [29,31)  a_sf_id   (runtime, OR'd from SFA tmem addr top bits)
#   bit [31]     k_size lower (mxf4nvf4: {upper,lower} 0=K64 1=K96 2=K128;
#                             mxf8f6f4: 0 = K32, 1 = K64)

from typing import Optional

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin, llvm
from cutlass.cutlass_dsl import Boolean, Int32, Uint32, dsl_user_op

_BIT_A_FORMAT = 7  # width 3
_BIT_B_FORMAT = 10  # width 2 (mxf4nvf4) or 3 (mxf8f6f4)
_BIT_A_MAJOR = 15  # width 1
_BIT_B_MAJOR = 16  # width 1
_BIT_N_DIM = 17  # width 6
_BIT_SCALE_FORMAT = 23  # width 2
_BIT_SFA_LAYOUT = 26  # width 1  <-- 0 = 32dp_4xCopy, 1 = 128dp_Unique
_BIT_M_DIM = 24  # m_dim = umma_m >> 4 placed here (== umma_m >> 7 at bit 27)
_BIT_A_SF_ID = 29  # width 2
_BIT_B_SF_ID = 4  # width 2
_BIT_K_SIZE_LO = 31  # K-size lower bit
_BIT_K_SIZE_HI = 3  # K-size upper bit (Rubin repurposes this reserved bit)

KIND_MXF4NVF4 = "mxf4nvf4"
KIND_MXF8F6F4 = "mxf8f6f4"
_KINDS = (KIND_MXF4NVF4, KIND_MXF8F6F4)

_UMMA_K_NVFP4 = 128
# K-size field per kind. mxf4nvf4 uses two bits ({upper, lower}); mxf8f6f4
# uses only the lower bit.
_K_SIZE_FIELD = {
    KIND_MXF4NVF4: {64: 0, 96: 1, 128: 2},
    KIND_MXF8F6F4: {32: 0, 64: 1},
}
# PTX block-scale vector qualifier per kind. NVFP4 needs the explicit
# ".block16" (16-element SF vectors). MXFP8 only supports 32-element vectors,
# which is the default for kind::mxf8f6f4, so no qualifier is emitted (this
# matches the PTX the DSL generates for SM107BlockScaledMmaMXF8F6F4Op).
_PTX_BLOCK_QUALIFIER = {
    KIND_MXF4NVF4: ".block16",
    KIND_MXF8F6F4: "",
}
# Operand-format field values per kind (see header comment).
_A_FORMAT = {
    KIND_MXF4NVF4: {"Float4E2M1FN": 1},
    KIND_MXF8F6F4: {"Float8E4M3FN": 0, "Float8E5M2": 1},
}
_B_FORMAT = _A_FORMAT
_B_FORMAT_MASK = {KIND_MXF4NVF4: 0x3, KIND_MXF8F6F4: 0x7}
# Scale-factor format field values per kind.
_SCALE_FORMAT = {
    KIND_MXF4NVF4: {"Float8E4M3FN": 0, "Float8E8M0FNU": 1},
    KIND_MXF8F6F4: {"Float8E8M0FNU": 1},
}


def _dtype_name(dtype) -> str:
    return dtype if isinstance(dtype, str) else dtype.__name__


def mma_kind_for_dtypes(a_dtype, b_dtype) -> str:
    """Return the block-scaled tcgen05 kind that multiplies ``a_dtype * b_dtype``."""
    a_name, b_name = _dtype_name(a_dtype), _dtype_name(b_dtype)
    if a_name == "Float4E2M1FN" and b_name == "Float4E2M1FN":
        return KIND_MXF4NVF4
    if a_name in _A_FORMAT[KIND_MXF8F6F4] and b_name in _B_FORMAT[KIND_MXF8F6F4]:
        return KIND_MXF8F6F4
    raise ValueError(f"no Rubin block-scaled MMA kind for A={a_name}, B={b_name}")


def build_static_idesc_base(
    *,
    umma_m: int,
    umma_n: int,
    kind: str = KIND_MXF4NVF4,
    a_format: Optional[int] = None,
    b_format: Optional[int] = None,
    a_major: int = 0,
    b_major: int = 0,
    scale_format: Optional[int] = None,
    umma_k: Optional[int] = None,
    sfa_layout: int = 0,
) -> int:
    """Pack the static-field portion of the idesc into a u32.

    Runtime fields (a_sf_id_, b_sf_id_) are left at zero; ``compute_idesc``
    OR's them in from the SF TMEM addresses.  n_dim is static (non-swapAB) and
    folded in here. Format fields default to the historical NVFP4/UE4M3
    encoding for ``mxf4nvf4`` and to E4M3/UE8M0 for ``mxf8f6f4``.
    """
    assert kind in _KINDS, f"Unsupported kind={kind}"
    assert umma_m in (64, 128, 256), f"Unsupported UMMA_M={umma_m}"
    if umma_k is None:
        umma_k = _UMMA_K_NVFP4 if kind == KIND_MXF4NVF4 else 64
    assert umma_k in _K_SIZE_FIELD[kind], f"Unsupported UMMA_K={umma_k} for {kind}"
    assert 0 <= sfa_layout < 2
    if a_format is None:
        a_format = 1 if kind == KIND_MXF4NVF4 else 0
    if b_format is None:
        b_format = 1 if kind == KIND_MXF4NVF4 else 0
    if scale_format is None:
        scale_format = 0 if kind == KIND_MXF4NVF4 else 1

    m_dim = umma_m >> 4  # placed at bit 24; == (umma_m >> 7) << 27
    n_dim = umma_n >> 3

    desc = 0
    desc |= (a_format & 0x7) << _BIT_A_FORMAT
    desc |= (b_format & _B_FORMAT_MASK[kind]) << _BIT_B_FORMAT
    desc |= (a_major & 0x1) << _BIT_A_MAJOR
    desc |= (b_major & 0x1) << _BIT_B_MAJOR
    desc |= (n_dim & 0x3F) << _BIT_N_DIM
    desc |= (scale_format & 0x3) << _BIT_SCALE_FORMAT
    desc |= (sfa_layout & 0x1) << _BIT_SFA_LAYOUT
    desc |= (m_dim & 0x1F) << _BIT_M_DIM
    k_size_value = _K_SIZE_FIELD[kind][umma_k]
    desc |= (k_size_value & 0x1) << _BIT_K_SIZE_LO
    desc |= ((k_size_value >> 1) & 0x1) << _BIT_K_SIZE_HI
    return desc & 0xFFFFFFFF


def build_static_idesc_base_for_dtypes(
    *,
    umma_m: int,
    umma_n: int,
    umma_k: int,
    a_dtype,
    b_dtype,
    sf_dtype,
    sfa_layout: int = 0,
) -> tuple[str, int]:
    """Return ``(kind, idesc_base)`` for the given operand and SF dtypes."""
    kind = mma_kind_for_dtypes(a_dtype, b_dtype)
    sf_name = _dtype_name(sf_dtype)
    if sf_name not in _SCALE_FORMAT[kind]:
        raise ValueError(f"scale-factor dtype {sf_name} is invalid for kind {kind}")
    idesc = build_static_idesc_base(
        umma_m=umma_m,
        umma_n=umma_n,
        kind=kind,
        a_format=_A_FORMAT[kind][_dtype_name(a_dtype)],
        b_format=_B_FORMAT[kind][_dtype_name(b_dtype)],
        scale_format=_SCALE_FORMAT[kind][sf_name],
        umma_k=umma_k,
        sfa_layout=sfa_layout,
    )
    return kind, idesc


@dsl_user_op
def compute_idesc(
    *,
    static_base: int,
    sfa_tmem_addr_i32,
    sfb_tmem_addr_i32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Uint32:
    """OR runtime a_sf_id_ / b_sf_id_ (SF TMEM addr top 2 bits) into the base.

    The shifts must be logical: a sub-word SF byte offset of 2 or 3 sets bit
    31 of the TMEM address, and an arithmetic shift would sign-extend it over
    the rest of the descriptor (an illegal-instruction trap on Rubin). NVFP4
    k-blocks land on whole columns, so only MXFP8 (2 SF bytes per UMMA_K=64
    k-block) exercises this.
    """
    idesc = Uint32(static_base)
    sfa_top = Uint32(sfa_tmem_addr_i32) & Uint32(0xC0000000)
    sfb_top = Uint32(sfb_tmem_addr_i32) & Uint32(0xC0000000)
    idesc = idesc | (sfa_top >> Uint32(30 - _BIT_A_SF_ID))
    idesc = idesc | (sfb_top >> Uint32(30 - _BIT_B_SF_ID))
    return idesc


def _smem_desc_to_i64(smem_desc_value: ir.Value) -> ir.Value:
    i64_ty = ir.IntegerType.get_signless(64)
    return builtin.unrealized_conversion_cast([i64_ty], [smem_desc_value])


def _tmem_ptr_to_i32(tmem_ptr_value: ir.Value) -> ir.Value:
    i32_ty = ir.IntegerType.get_signless(32)
    return builtin.unrealized_conversion_cast([i32_ty], [tmem_ptr_value])


def _as_value(it) -> ir.Value:
    return it.value if hasattr(it, "value") else it


def block_scale_mma_opcode(kind: str, cta_group: int) -> str:
    """Return the PTX opcode (with qualifiers) of the block-scaled UTCOMMA.

    Pure helper shared with the unit tests: ``kind::mxf4nvf4`` carries the
    explicit ``.block16`` SF-vector qualifier, ``kind::mxf8f6f4`` relies on
    its only legal (default) 32-element vector, and A is always discarded
    from the collector buffer.
    """
    assert cta_group in (1, 2), f"cta_group must be 1 or 2, got {cta_group}"
    assert kind in _KINDS, f"Unsupported kind={kind}"
    return (
        f"tcgen05.mma.cta_group::{cta_group}.kind::{kind}"
        f".block_scale{_PTX_BLOCK_QUALIFIER[kind]}.collector::a::discard"
    )


@dsl_user_op
def _tcgen05_mma_block_scale(
    *,
    kind: str,
    cta_group: int,
    d_tmem_i32,
    a_desc_i64,
    b_desc_i64,
    idesc_i32,
    enable_input_d_i32,
    sfa_tmem_i32,
    sfb_tmem_i32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Emit ``tcgen05.mma.cta_group::{1,2}.kind::<kind>.block_scale<block>``."""
    opcode = block_scale_mma_opcode(kind, cta_group)
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
        f"{opcode} "
        "[$0], $1, $2, $3, [$5], [$6], p;\n\t"
        "}\n",
        "r,l,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_manual_block_scaled_mma_atom(
    *,
    acc_frag,  # tCtAcc  (V, M_count, N_count)   -- TMEM f32 accumulator
    a_frag,  # tCrA[(None,None,kblk,stage)]   (V, M_count) -- smem desc
    sfa_frag,  # tCtSFA[(None,None,kblk)]        (V, MN_count) -- TMEM sf
    b_frag,  # tCrB[(None,None,kblk,stage)]   (V, N_count) -- smem desc
    sfb_frag,  # tCtSFB[(None,None,kblk)]        (V, MN_count) -- TMEM sf
    static_idesc_base: int,
    accumulate,  # Boolean SSA: D = A@B (+ C) when True
    cta_group: int = 1,
    kind: str = KIND_MXF4NVF4,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ONE block-scaled UTCOMMA for the current (k-block, stage) atom.

    Mirrors megaMOE ``dynamic_mainloop`` per-atom extraction: slice each
    operand down to its atom (V,), cast to the PTX register types, OR the
    runtime SF-id bits into the static idesc base, then emit the asm.
    """
    a_atom = a_frag[(None, 0)]
    b_atom = b_frag[(None, 0)]
    sfa_atom = sfa_frag[(None, 0)]
    sfb_atom = sfb_frag[(None, 0)]
    acc_atom = acc_frag[(None, 0, 0)]

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
        sfa_tmem_addr_i32=operand_sfa_i32,
        sfb_tmem_addr_i32=operand_sfb_i32,
    )

    with cute.arch.elect_one():
        _tcgen05_mma_block_scale(
            kind=kind,
            cta_group=cta_group,
            d_tmem_i32=operand_acc_i32,
            a_desc_i64=operand_a,
            b_desc_i64=operand_b,
            idesc_i32=idesc.ir_value(),
            enable_input_d_i32=Int32(Boolean(accumulate)).ir_value(),
            sfa_tmem_i32=operand_sfa_i32,
            sfb_tmem_i32=operand_sfb_i32,
        )
