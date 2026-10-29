# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal inline-PTX primitives required by the greenfield NVFP4 kernels."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, dsl_user_op

TmaCacheHintEvictFirst = 0x12F0000000000000


def _address_value(pointer_or_address, *, loc=None, ip=None):
    if isinstance(pointer_or_address, Int64):
        return pointer_or_address.ir_value()
    return pointer_or_address.toint(loc=loc, ip=ip).ir_value()


@dsl_user_op
def nanosleep(
    sleep_cycles: int, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> None:
    """Suspend the calling thread for up to the requested clock cycles."""
    if cutlass.const_expr(hasattr(cute.arch, "nanosleep")):
        cute.arch.nanosleep(sleep_time=sleep_cycles, loc=loc, ip=ip)
        return

    llvm.inline_asm(
        res=None,
        operands_=[Int32(sleep_cycles).ir_value(loc=loc, ip=ip)],
        asm_string="nanosleep.u32 $0;",
        constraints="r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def exit(*, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> None:
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="exit;",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def read_clock64(
    *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> Int64:
    """Read the per-SM 64-bit cycle counter."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def movmatrix_b16(input_regs: cute.Tensor) -> cute.Tensor:
    """Transpose every packed m8n8 b16 register fragment across the warp."""
    if cutlass.const_expr(input_regs.element_type.width != 32):
        raise TypeError(
            f"movmatrix_b16 expects packed 32-bit registers, got {input_regs.element_type}."
        )

    input_words = cute.coalesce(cute.flatten(cute.recast_tensor(input_regs, Int32)))
    output_words = cute.make_rmem_tensor((cute.size(input_words),), Int32)
    for word_idx in cutlass.range_constexpr(cute.size(input_words)):
        output_words[word_idx] = Int32(
            llvm.inline_asm(
                T.i32(),
                [Int32(input_words[word_idx]).ir_value()],
                "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
                "=r,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    output_regs = cute.make_rmem_tensor(input_regs.layout, input_regs.element_type)
    output_regs_words = cute.coalesce(cute.flatten(cute.recast_tensor(output_regs, Int32)))
    for word_idx in cutlass.range_constexpr(cute.size(output_words)):
        output_regs_words[word_idx] = output_words[word_idx]
    return output_regs


@dsl_user_op
def cvt_f32_to_fp8_to_f32(
    value, fp8_type, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> Float32:
    """Round one f32 through the selected FP8 format and widen it back."""
    if cutlass.const_expr(fp8_type is cutlass.Float8E8M0FNU):
        downcast_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
        upcast_instruction = "cvt.rn.bf16x2.ue8m0x2"
    elif cutlass.const_expr(fp8_type is cutlass.Float8E4M3FN):
        downcast_instruction = "cvt.rn.satfinite.e4m3x2.f32"
        upcast_instruction = "cvt.rn.bf16x2.e4m3x2"
    elif cutlass.const_expr(fp8_type is cutlass.Float8E5M2):
        downcast_instruction = "cvt.rn.satfinite.e5m2x2.f32"
        upcast_instruction = "cvt.rn.bf16x2.e5m2x2"
    else:
        raise ValueError(f"Unsupported FP8 type {fp8_type}.")

    packed_bf16 = llvm.inline_asm(
        T.i32(),
        [Float32(value).ir_value(loc=loc, ip=ip)],
        "{\n"
        "  .reg .b16 converted;\n"
        f"  {downcast_instruction} converted, 0f00000000, $1;\n"
        f"  {upcast_instruction} $0, converted;\n"
        "}",
        "=r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    bf16_pair_type = ir.Type.parse("vector<2xbf16>")
    bf16_pair = llvm.bitcast(bf16_pair_type, packed_bf16, loc=loc, ip=ip)
    rounded_bf16 = vector.extract(bf16_pair, [], [0], loc=loc, ip=ip)
    return Float32(arith.extf(Float32.mlir_type, rounded_bf16, loc=loc, ip=ip))


@dsl_user_op
def tma_load_1d(
    destination_smem, source_gmem, mbarrier_smem, num_bytes, *, loc=None, ip=None
) -> None:
    """Issue a cache-hinted 1D GMEM-to-SMEM bulk copy."""
    llvm.inline_asm(
        None,
        [
            destination_smem.toint(loc=loc, ip=ip).ir_value(),
            _address_value(source_gmem, loc=loc, ip=ip),
            num_bytes.ir_value(),
            mbarrier_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(TmaCacheHintEvictFirst).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_i32_to_peer_cluster_smem_async(
    smem_pointer,
    value: Int32,
    mbarrier_pointer,
    destination_cta_rank,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Store one Int32 to peer SMEM and complete its transaction barrier."""
    smem_address = llvm.ptrtoint(T.i32(), smem_pointer.llvm_ptr, loc=loc, ip=ip)
    mbarrier_address = llvm.ptrtoint(T.i32(), mbarrier_pointer.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            smem_address,
            value.ir_value(loc=loc, ip=ip),
            mbarrier_address,
            Int32(destination_cta_rank).ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_addr;
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_addr, $0, $3;
            mapa.shared::cluster.u32 remote_mbar, $2, $3;
            st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [remote_addr], $1, [remote_mbar];
        }}""",
        constraints="r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_arrive_expect_tx_on_peer(
    mbarrier_pointer,
    transaction_bytes: Int32,
    destination_cta_rank,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Declare an expected peer-CTA SMEM transaction."""
    mbarrier_address = llvm.ptrtoint(T.i32(), mbarrier_pointer.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            mbarrier_address,
            Int32(destination_cta_rank).ir_value(loc=loc, ip=ip),
            transaction_bytes.ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_mbar, $0, $1;
            mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote_mbar], $2;
        }}""",
        constraints="r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_bulk_s2g(
    destination_gmem,
    source_smem,
    num_bytes,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    llvm.inline_asm(
        None,
        [
            destination_gmem.toint(loc=loc, ip=ip).ir_value(),
            source_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_reduce_async_bulk_add_bf16_s2g(
    destination_gmem,
    source_smem,
    num_bytes,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    llvm.inline_asm(
        None,
        [
            destination_gmem.toint(loc=loc, ip=ip).ir_value(),
            source_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_reduce_async_bulk_add_u32_s2g(
    destination_gmem,
    source_smem,
    num_bytes,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    llvm.inline_asm(
        None,
        [
            destination_gmem.toint(loc=loc, ip=ip).ir_value(),
            source_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds128_v4_b32(smem_pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [smem_pointer.toint(loc=loc, ip=ip).ir_value()],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [index])) for index in range(4))


@dsl_user_op
def stg_f32(
    address: Int64, value: Float32, predicate: Optional[Int32] = None, *, loc=None, ip=None
) -> None:
    if predicate is None:
        llvm.inline_asm(
            None,
            [address.ir_value(), value.ir_value()],
            "st.global.f32 [$0], $1;",
            "l,f",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value(), predicate.ir_value()],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.f32 [$0], $1;\n\t}",
        "l,f,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b64(
    address: Int64, value: Int64, predicate: Optional[Int32] = None, *, loc=None, ip=None
) -> None:
    if predicate is None:
        llvm.inline_asm(
            None,
            [address.ir_value(), value.ir_value()],
            "st.global.u64 [$0], $1;",
            "l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value(), predicate.ir_value()],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.u64 [$0], $1;\n\t}",
        "l,l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "red.relaxed.sys.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_f32(
    address,
    value: Float32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    llvm.inline_asm(
        None,
        [address.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        "red.relaxed.sys.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_sys_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "red.release.sys.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_gpu_s32(
    counter_pointer,
    value: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    llvm.inline_asm(
        None,
        [counter_pointer.toint(loc=loc, ip=ip).ir_value(), value.ir_value()],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_async_add_release_gpu_s32(
    counter_pointer,
    value: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    llvm.inline_asm(
        None,
        [counter_pointer.toint(loc=loc, ip=ip).ir_value(), value.ir_value()],
        "red.async.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_async_add_release_sys_u32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """Fire-and-forget cross-rank counter bump carrying its own release ordering.

    The issuing warp does not wait for the L2/HBM round trip, and the release is
    enforced by the memory system, so this replaces an explicit ``membar.sys``
    followed by a relaxed reduction. ``u32`` rather than ``s32`` only because a
    counter bump is the same two's-complement add either way and this is the
    spelling already proven on this path.

    Operands go through uniform registers, so the address must be warp-uniform:
    a fan-out where lanes target different peers cannot use this.
    """
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "red.async.release.sys.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_v2_bf16x2(
    address,
    value0,
    value1,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    llvm.inline_asm(
        None,
        [address.toint(loc=loc, ip=ip).ir_value(), value0.ir_value(), value1.ir_value()],
        "red.relaxed.sys.global.add.noftz.v2.bf16x2 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def cvt_f32x4_to_f8x4_pack_i32(fp32x4: cute.Tensor, fp8_type, *, loc=None, ip=None) -> Int32:
    """Round four f32 lanes to the selected FP8 format and pack them into one i32."""
    fp32x4 = fp32x4.load()
    src_vec4 = fp32x4.ir_value(loc=loc, ip=ip) if hasattr(fp32x4, "ir_value") else fp32x4

    src0 = Float32(vector.extract(src_vec4, [], [0])).ir_value(loc=loc, ip=ip)
    src1 = Float32(vector.extract(src_vec4, [], [1])).ir_value(loc=loc, ip=ip)
    src2 = Float32(vector.extract(src_vec4, [], [2])).ir_value(loc=loc, ip=ip)
    src3 = Float32(vector.extract(src_vec4, [], [3])).ir_value(loc=loc, ip=ip)

    if cutlass.const_expr(fp8_type is cutlass.Float8E8M0FNU):
        cvt_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
    elif cutlass.const_expr(fp8_type is cutlass.Float8E4M3FN):
        cvt_instruction = "cvt.rn.satfinite.e4m3x2.f32"
    elif cutlass.const_expr(fp8_type is cutlass.Float8E5M2):
        cvt_instruction = "cvt.rn.satfinite.e5m2x2.f32"
    else:
        raise ValueError(f"Unsupported FP8 type {fp8_type}.")

    packed_i32 = llvm.inline_asm(
        T.i32(),
        [src0, src1, src2, src3],
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        f"  {cvt_instruction} lo, $2, $1;\n"
        f"  {cvt_instruction} hi, $4, $3;\n"
        "  mov.b32 $0, {lo, hi};\n"
        "}",
        "=r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(packed_i32)


@dsl_user_op
def stg_e8m0_from_f32(addr: Int64, fp32_val: Float32, *, loc=None, ip=None) -> None:
    """Convert ``fp32_val`` to E8M0 via PTX and store the 1-byte result to global memory.

    Uses ``cvt.rp.satfinite.ue8m0x2.f32`` -- the correct fp32 -> E8M0 path -- rather
    than the DSL's generic ``.to(Float8E8M0FNU)``, which does not lower correctly for
    the non-IEEE-754 E8M0 type.
    """
    llvm.inline_asm(
        None,
        [addr.ir_value(), fp32_val.ir_value()],
        "{\n"
        "  .reg .b16 bf_lo;\n"
        "  .reg .u32 tmp;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 bf_lo, 0f00000000, $1;\n"
        "  cvt.u32.u16 tmp, bf_lo;\n"
        "  st.global.b8 [$0], tmp;\n"
        "}",
        "l,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_e8m0x8_from_f32(
    addr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Convert 8 fp32 values to E8M0 and store them as 8 contiguous bytes in one shot.

    Batched form of ``stg_e8m0_from_f32``: four ``cvt.rp.satfinite.ue8m0x2.f32`` each
    pack two E8M0 bytes, the four ``.b16`` results are assembled into two ``.b32`` words,
    and a single ``st.global.v2.u32`` writes all 8 bytes. ``addr`` must be 8-byte aligned;
    output byte ``k`` holds E8M0(``v{k}``).
    """
    llvm.inline_asm(
        None,
        [
            addr.ir_value(),
            v0.ir_value(),
            v1.ir_value(),
            v2.ir_value(),
            v3.ir_value(),
            v4.ir_value(),
            v5.ir_value(),
            v6.ir_value(),
            v7.ir_value(),
        ],
        "{\n"
        "  .reg .b16 p0, p1, p2, p3;\n"
        "  .reg .b32 w0, w1;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p0, $2, $1;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p1, $4, $3;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p2, $6, $5;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 p3, $8, $7;\n"
        "  mov.b32 w0, {p0, p1};\n"
        "  mov.b32 w1, {p2, p3};\n"
        "  st.global.v2.u32 [$0], {w0, w1};\n"
        "}",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


__all__ = [
    "TmaCacheHintEvictFirst",
    "cp_async_bulk_s2g",
    "cp_reduce_async_bulk_add_bf16_s2g",
    "cp_reduce_async_bulk_add_u32_s2g",
    "cvt_f32_to_fp8_to_f32",
    "cvt_f32x4_to_f8x4_pack_i32",
    "exit",
    "lds128_v4_b32",
    "mbarrier_arrive_expect_tx_on_peer",
    "movmatrix_b16",
    "nanosleep",
    "read_clock64",
    "red_add_relaxed_sys_f32",
    "red_add_relaxed_sys_s32",
    "red_add_relaxed_sys_v2_bf16x2",
    "red_async_add_release_gpu_s32",
    "red_async_add_release_sys_u32",
    "red_add_release_gpu_s32",
    "red_add_release_sys_s32",
    "store_i32_to_peer_cluster_smem_async",
    "stg_b64",
    "stg_e8m0_from_f32",
    "stg_e8m0x8_from_f32",
    "stg_f32",
    "tma_load_1d",
]
