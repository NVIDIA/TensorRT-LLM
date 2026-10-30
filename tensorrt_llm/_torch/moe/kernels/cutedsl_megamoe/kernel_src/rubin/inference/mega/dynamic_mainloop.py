"""Raw SM107 dynamic-N block-scaled MMA emission for Rubin MegaMoE."""

from typing import Optional

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin, llvm
from cutlass.cutlass_dsl import Boolean, Int32, dsl_user_op

from .....quant_def import QuantKind


_bit_k_size_upper = 3
_bit_b_sf_id = 4
_bit_a_format = 7
_bit_b_format = 10
_bit_a_major = 15
_bit_b_major = 16
_bit_n_dim = 17
_bit_scale_format = 23
_bit_sfa_layout = 26
_bit_m_dim = 27
_bit_a_sf_id = 29
_bit_k_size = 31


def _align16(value):
    """Round a runtime Int32 value up to sixteen."""
    return (Int32(value) + Int32(15)) & Int32(-16)


@dsl_user_op
def compute_non_leader_cta_load_shift(
    *,
    valid_tokens_in_tile,
    mma_tiler_n: int,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """Return the 2CTA nonleader TMA-B shift for dynamic instruction N."""
    return (_align16(valid_tokens_in_tile) >> Int32(1)) - Int32(mma_tiler_n // 2)


def build_static_idesc_base(
    *,
    quant_kind: QuantKind,
    instruction_mnk: tuple[int, int, int],
    a_major: int = 0,
    b_major: int = 0,
) -> int:
    """Pack all static SM107 block-scaled descriptor fields."""
    instruction_m, instruction_n, instruction_k = instruction_mnk
    if instruction_m not in (128, 256):
        raise ValueError(f"SM107 MegaMoE requires instruction M 128 or 256, got {instruction_m}.")
    instruction_n_granularity = 16 if instruction_m == 256 else 8
    if instruction_n <= 0 or instruction_n > 256 or instruction_n % instruction_n_granularity != 0:
        raise ValueError(f"Invalid SM107 instruction N {instruction_n} for instruction M {instruction_m}.")

    expected_instruction_k = quant_kind.instruction_k("2x")
    if instruction_k != expected_instruction_k:
        raise ValueError(
            f"SM107 MegaMoE requires the 2x instruction K {expected_instruction_k} for {quant_kind}, "
            f"got {instruction_k}."
        )

    is_dual_fp4 = quant_kind.umma_kind != "mxf8f6f4"
    k_size_upper = 1 if is_dual_fp4 else 0
    k_size = 0 if is_dual_fp4 else 1

    descriptor = 0
    descriptor |= k_size_upper << _bit_k_size_upper
    descriptor |= (quant_kind.weight_format_code & 0x7) << _bit_a_format
    descriptor |= (quant_kind.activation_format_code & 0x7) << _bit_b_format
    descriptor |= (a_major & 0x1) << _bit_a_major
    descriptor |= (b_major & 0x1) << _bit_b_major
    descriptor |= (quant_kind.scale_format_code & 0x7) << _bit_scale_format
    descriptor |= 0 << _bit_sfa_layout
    descriptor |= ((instruction_m >> 7) & 0x3) << _bit_m_dim
    descriptor |= (k_size & 0x1) << _bit_k_size
    return descriptor & 0xFFFFFFFF


@dsl_user_op
def compute_idesc(
    *,
    static_base: int,
    valid_tokens_in_tile,
    sfa_tmem_addr_i32,
    sfb_tmem_addr_i32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """Add dynamic N and scale-factor IDs to an SM107 descriptor."""
    n_dim_value = _align16(valid_tokens_in_tile) >> Int32(3)
    descriptor = Int32(static_base) | (n_dim_value << _bit_n_dim)
    sfa_top_bits = Int32(sfa_tmem_addr_i32) & Int32(0xC0000000)
    sfb_top_bits = Int32(sfb_tmem_addr_i32) & Int32(0xC0000000)
    descriptor = descriptor | (
        (sfa_top_bits >> Int32(30 - _bit_a_sf_id)) & Int32(0x3 << _bit_a_sf_id)
    )
    descriptor = descriptor | (
        (sfb_top_bits >> Int32(30 - _bit_b_sf_id)) & Int32(0x3 << _bit_b_sf_id)
    )
    return descriptor


def _smem_desc_to_i64(value: ir.Value) -> ir.Value:
    return builtin.unrealized_conversion_cast([ir.IntegerType.get_signless(64)], [value])


def _tmem_ptr_to_i32(value: ir.Value) -> ir.Value:
    return builtin.unrealized_conversion_cast([ir.IntegerType.get_signless(32)], [value])


def _as_value(value) -> ir.Value:
    return value.value if hasattr(value, "value") else value


@dsl_user_op
def _tcgen05_mma_block_scaled(
    *,
    cta_group: int,
    mma_kind: str,
    scale_vec_suffix: str,
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
    """Emit one SM107 block-scaled tcgen05 instruction."""
    if cta_group not in (1, 2):
        raise ValueError(f"cta_group must be one or two, got {cta_group}.")
    if mma_kind not in ("mxf4nvf4", "mxf8f6f4"):
        raise ValueError(f"Unsupported SM107 MMA kind {mma_kind!r}.")
    if scale_vec_suffix not in ("", ".block16", ".block32"):
        raise ValueError(f"Unsupported SM107 scale-vector suffix {scale_vec_suffix!r}.")

    llvm.inline_asm(
        None,
        [d_tmem_i32, a_desc_i64, b_desc_i64, idesc_i32, enable_input_d_i32, sfa_tmem_i32, sfb_tmem_i32],
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, $4, 0;\n\t"
        f"tcgen05.mma.cta_group::{cta_group}.kind::{mma_kind}.block_scale{scale_vec_suffix} "
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
def issue_dynamic_block_scaled_mma_window(
    *,
    quant_kind: QuantKind,
    acc_tensor: cute.Tensor,
    a_window_frag: cute.Tensor,
    b_window_frag: cute.Tensor,
    sfa_window_tensor: cute.Tensor,
    sfb_window_tensor: cute.Tensor,
    valid_tokens_in_tile: Int32,
    mma_instruction_mnk: tuple,
    window_instruction_offset: int,
    window_instruction_count: int,
    first_instruction_accumulate,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one SF window while preserving caller-owned accumulation state."""
    if window_instruction_count <= 0:
        raise ValueError("window_instruction_count must be positive.")
    if window_instruction_offset < 0:
        raise ValueError("window_instruction_offset must be nonnegative.")
    if mma_instruction_mnk[2] != quant_kind.instruction_k("2x"):
        raise ValueError("mma_instruction_mnk must use the SM107 2x instruction K.")

    static_idesc_base = build_static_idesc_base(
        quant_kind=quant_kind,
        instruction_mnk=mma_instruction_mnk,
    )
    cta_group = 2 if mma_instruction_mnk[0] == 256 else 1
    mma_kind = "mxf8f6f4" if quant_kind.umma_kind == "mxf8f6f4" else "mxf4nvf4"
    scale_vec_suffix = "" if mma_kind == "mxf8f6f4" else quant_kind.umma_scale_vec_suffix

    for instruction_index in range(window_instruction_count):
        fragment_instruction_index = window_instruction_offset + instruction_index
        a_atom = a_window_frag[(None, 0, fragment_instruction_index)]
        b_atom = b_window_frag[(None, 0, fragment_instruction_index)]
        sfa_atom = sfa_window_tensor[(None, 0, instruction_index)]
        sfb_atom = sfb_window_tensor[(None, 0, instruction_index)]
        acc_atom = acc_tensor[(None, 0, 0)]

        operand_a = _smem_desc_to_i64(_as_value(a_atom.iterator))
        operand_b = _smem_desc_to_i64(_as_value(b_atom.iterator))
        operand_sfa_i32 = _tmem_ptr_to_i32(_as_value(sfa_atom.iterator))
        operand_sfb_i32 = _tmem_ptr_to_i32(_as_value(sfb_atom.iterator))
        operand_acc_i32 = _tmem_ptr_to_i32(_as_value(acc_atom.iterator))
        descriptor = compute_idesc(
            static_base=static_idesc_base,
            valid_tokens_in_tile=valid_tokens_in_tile,
            sfa_tmem_addr_i32=operand_sfa_i32,
            sfb_tmem_addr_i32=operand_sfb_i32,
        )
        accumulate = first_instruction_accumulate if instruction_index == 0 else True

        with cute.arch.elect_one():
            _tcgen05_mma_block_scaled(
                cta_group=cta_group,
                mma_kind=mma_kind,
                scale_vec_suffix=scale_vec_suffix,
                d_tmem_i32=operand_acc_i32,
                a_desc_i64=operand_a,
                b_desc_i64=operand_b,
                idesc_i32=descriptor.ir_value(),
                enable_input_d_i32=Int32(Boolean(accumulate)).ir_value(),
                sfa_tmem_i32=operand_sfa_i32,
                sfb_tmem_i32=operand_sfb_i32,
            )


__all__ = [
    "build_static_idesc_base",
    "compute_idesc",
    "compute_non_leader_cta_load_shift",
    "issue_dynamic_block_scaled_mma_window",
]
