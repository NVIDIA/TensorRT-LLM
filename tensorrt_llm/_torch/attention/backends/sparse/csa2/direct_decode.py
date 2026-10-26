# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct sparse decode attention over CSA2's persistent caches for small generation batches.

One CuTe DSL launch computes the sparse attention of every query row straight from the packed
caches: the selected SWA rows (E4M3 with power-of-two scales per 32 channels) and main rows
(E2M1 with E4M3 scales per 16 channels) are decoded into BF16 in shared memory -- exactly, both
formats are subsets of BF16 -- and attended by the BF16 query. Neither an FP8 staging pool nor
a per-forward staging scale is involved.

Work split: one cluster of up to 8 CTAs per query row. CTA ``c`` owns an equal share of the
row's main columns and of its SWA columns, decodes those rows, and attends each share as a
part: ``S = Q K^T`` with BF16 ``mma.sync`` and FP32 accumulation (heads are the M dimension,
the 512 channels split over the warps and summed in shared memory), a softmax numerator in
FP32 (probabilities rounded to BF16 for the second product) and ``O = P V`` (V is the decoded
row). The two parts are merged in registers, then every CTA stores its partial output and its
(max, sum) per head into the CTA owning each block of ``512 / cluster`` channels (distributed
shared memory, one cluster barrier), which adds the attention sink and writes BF16. Invalid
selections (negative or out-of-range slots, unmapped or invisible logical positions) get zero
probability, and a row without valid keys outputs zeros.

The launch can also take the query before its RoPE and rotate it in shared memory, bit for bit
as a separate ``mla_rope_inplace`` launch does.

Work that does not depend on the launch preceding this one on the stream happens before the
programmatic dependent wait: the query and the SWA slots when ``inputs_ready`` promises they
were final before that launch started, and the main rows (slots, page-table mapping, payload)
when ``main_ready`` promises the same for them; with both, the whole main part runs before
the wait and only the SWA rows the preceding cache write may have produced are left after it.
"""

import functools

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm.math_utils import ceil_div, pad_up

#: Largest query-row count per rank the direct decode serves (a cluster of up to 8 CTAs per row).
MAX_ROWS = 16
HEAD_DIM = 512
SWA_ROW_BYTES = 528
MAIN_ROW_BYTES = 288
_THREADS = 256
_KEY_ROW = 2 * HEAD_DIM + 16  # bytes per BF16 key row in shared memory (16 B bank shift)
LOG2E = 1.4426950408889634


def direct_decode_device_supported(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


#: Dynamic shared memory one CTA may use (SM100/SM103 opt-in limit).
_SMEM_LIMIT = 227 * 1024


def plan_split(swa_width: int, main_width: int) -> int:
    """CTAs per query row (one cluster) for a selection width."""
    width = swa_width + main_width
    cluster = 8
    while cluster > 1 and width <= 16 * (cluster // 2):
        cluster //= 2
    return cluster


def cta_keys(swa_width: int, main_width: int, cluster: int) -> tuple[int, int]:
    """(main, SWA) key rows per CTA: each width split over the cluster in multiples of 16."""

    def share(width):
        return 0 if width == 0 else pad_up(ceil_div(width, cluster), 16)

    return share(main_width), share(swa_width)


def smem_layout(heads: int, main_keys: int, swa_keys: int, cluster: int) -> dict:
    """Byte offsets of one CTA's shared memory. ``recv`` holds every source CTA's partial output
    for this CTA's channels (``cluster`` blocks of ``block`` bytes: FP32 [heads, 512 / cluster]
    then (max, sum) per head). It gets its own space when that fits (``dedicated``: the sources
    store into it while this CTA may still be computing); otherwise it reuses the working region
    after a cluster barrier."""
    keys = main_keys + swa_keys
    part = max(main_keys, swa_keys, 16)
    prow = 2 * part + 16
    sp = keys * _KEY_ROW
    q = sp + (8 // (heads // 16)) * heads * part * 4
    p = q + heads * _KEY_ROW
    work = pad_up(p + heads * prow, 16)
    block = heads * (HEAD_DIM // cluster) * 4 + heads * 8
    recv_bytes = cluster * block
    extras = keys * 12 + heads * 20 + 6 * 128 + 64 * 4
    dedicated = work + recv_bytes + extras <= _SMEM_LIMIT
    recv = work if dedicated else 0
    total_bytes = recv + recv_bytes if dedicated else pad_up(max(work, recv_bytes), 16)
    return {
        "sp": sp,
        "q": q,
        "p": p,
        "prow": prow,
        "block": block,
        "recv": recv,
        "bytes": total_bytes,
        "total": total_bytes + extras,
        "dedicated": dedicated,
    }


def mma_heads(heads: int) -> int:
    """Heads the kernel computes for ``heads`` query heads per rank (the ``mma.sync`` M dimension
    is 16 heads: 8 query heads, e.g. at TP8, fill one tile with 8 zero heads), 0 if unsupported."""
    if heads == 8:
        return 16
    return heads if heads % 16 == 0 and heads // 16 in (1, 2, 4) else 0


def direct_decode_fits(heads: int, swa_width: int, main_width: int) -> bool:
    """Whether one CTA's shared memory holds this geometry's key rows and merge buffers for
    ``heads`` query heads per rank."""
    heads = mma_heads(heads)
    if not heads or swa_width + main_width == 0:
        return False
    cluster = plan_split(swa_width, main_width)
    keys = cta_keys(swa_width, main_width, cluster)
    return smem_layout(heads, *keys, cluster)["total"] <= _SMEM_LIMIT


@functools.lru_cache(maxsize=1)
def _kernel_type():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint8, const_expr
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    from .kernel import _cache_kernels

    numerics = _cache_kernels()
    mul_rn, fma_rn = numerics.mul_rn, numerics.fma_rn

    def asm(res, operands, text, constraints, *, side_effects=True, loc=None, ip=None):
        return llvm.inline_asm(
            res,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    def words(out, count, loc, ip):
        return tuple(
            Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(count)
        )

    def floats(out, count, loc, ip):
        return tuple(
            Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(count)
        )

    # ------------------------------------------------------------------ memory

    @dsl_user_op
    def ld_global_index(addr: Int64, wide: bool, *, loc=None, ip=None) -> Int64:
        if wide:
            return Int64(
                asm(
                    T.i64(),
                    [Int64(addr).ir_value(loc=loc, ip=ip)],
                    "ld.global.s64 $0, [$1];",
                    "=l,l,~{memory}",
                    loc=loc,
                    ip=ip,
                )
            )
        return Int64(
            asm(
                T.i64(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .s32 v;\n\tld.global.s32 v, [$1];\n\tcvt.s64.s32 $0, v;\n\t}",
                "=l,l,~{memory}",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def ld_global_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.s32 $0, [$1];",
                "=r,l,~{memory}",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
        return Float32(
            asm(
                T.f32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.f32 $0, [$1];",
                "=f,l,~{memory}",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.u8 $0, [$1];",
                "=r,l,~{memory}",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def ld_global_b32(addr: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.b32 $0, [$1];",
                "=r,l,~{memory}",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def ld_global_v2(addr: Int64, *, loc=None, ip=None):
        out = asm(
            llvm.StructType.get_literal([T.i32()] * 2),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.v2.b32 {$0, $1}, [$2];",
            "=r,=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
        return words(out, 2, loc, ip)

    @dsl_user_op
    def ld_global_v4(addr: Int64, *, loc=None, ip=None):
        out = asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
        return words(out, 4, loc, ip)

    @dsl_user_op
    def st_global_v2(addr: Int64, w0: Int32, w1: Int32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(w0).ir_value(loc=loc, ip=ip),
                Int32(w1).ir_value(loc=loc, ip=ip),
            ],
            "st.global.v2.b32 [$0], {$1, $2};",
            "l,r,r,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def cp_async16(dst: Int32, src: Int64, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
            "cp.async.cg.shared.global [$0], [$1], 16;",
            "r,l,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def cp_async_wait_all(*, loc=None, ip=None) -> None:
        asm(
            None,
            [],
            "cp.async.commit_group;\n\tcp.async.wait_group 0;",
            "~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def st_shared_v4(
        addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None
    ) -> None:
        asm(
            None,
            [Int32(addr).ir_value(loc=loc, ip=ip)]
            + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
            "r,r,r,r,r,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def st_shared_f32x2(addr: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "st.shared.v2.f32 [$0], {$1, $2};",
            "r,f,f,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def st_shared_f32(addr: Int32, a: Float32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int32(addr).ir_value(loc=loc, ip=ip), Float32(a).ir_value(loc=loc, ip=ip)],
            "st.shared.f32 [$0], $1;",
            "r,f,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
        return Float32(
            asm(
                T.f32(),
                [Int32(addr).ir_value(loc=loc, ip=ip)],
                "ld.shared.f32 $0, [$1];",
                "=f,r,~{memory}",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def st_shared_b16(addr: Int32, w: Int32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.shared.b16 [$0], h;\n\t}",
            "r,r,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def mapa(addr: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int32(addr).ir_value(loc=loc, ip=ip), Int32(rank).ir_value(loc=loc, ip=ip)],
                "mapa.shared::cluster.u32 $0, $1, $2;",
                "=r,r,r",
                side_effects=False,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def ldmatrix_x4(addr: Int32, trans: bool, *, loc=None, ip=None):
        op = (
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            if trans
            else "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
        )
        out = asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            op + " {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
        return words(out, 4, loc, ip)

    @dsl_user_op
    def mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
        out = asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [Int32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
            + [Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)],
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
        return floats(out, 4, loc, ip)

    @dsl_user_op
    def swa_scale(byte: Int32, *, loc=None, ip=None) -> Float32:
        """UE8M0 byte -> 2^(byte - 127) (byte 0 is the FP32 subnormal 2^-127)."""
        return Float32(
            asm(
                T.f32(),
                [Int32(byte).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .b32 r;\n\t.reg .pred p;\n\tsetp.eq.u32 p, $1, 0;\n\tshl.b32 r, $1, 23;\n\t"
                "selp.b32 r, 0x00400000, r, p;\n\tmov.b32 $0, r;\n\t}",
                "=f,r",
                side_effects=False,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def e4m3_scale(byte: Int32, *, loc=None, ip=None) -> Float32:
        """E4M3 byte -> FP32 (exact)."""
        return Float32(
            asm(
                T.f32(),
                [Int32(byte).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .b16 b, lo, hi;\n\t.reg .b32 h;\n\tcvt.u16.u32 b, $1;\n\t"
                "cvt.rn.f16x2.e4m3x2 h, b;\n\tmov.b32 {lo, hi}, h;\n\tcvt.f32.f16 $0, lo;\n\t}",
                "=f,r",
                side_effects=False,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def swa_decode4(w: Int32, scale: Float32, *, loc=None, ip=None):
        """Four E4M3 channels (byte k = channel k) times ``scale`` -> two BF16 pairs."""
        out = asm(
            llvm.StructType.get_literal([T.i32()] * 2),
            [Int32(w).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 lo, hi, x0, x1, x2, x3;\n\t.reg .b32 h0, h1;\n\t.reg .f32 f0, f1, f2, f3;\n\t"
            "mov.b32 {lo, hi}, $2;\n\tcvt.rn.f16x2.e4m3x2 h0, lo;\n\tcvt.rn.f16x2.e4m3x2 h1, hi;\n\t"
            "mov.b32 {x0, x1}, h0;\n\tmov.b32 {x2, x3}, h1;\n\t"
            "cvt.f32.f16 f0, x0;\n\tcvt.f32.f16 f1, x1;\n\tcvt.f32.f16 f2, x2;\n\tcvt.f32.f16 f3, x3;\n\t"
            "mul.rn.f32 f0, f0, $3;\n\tmul.rn.f32 f1, f1, $3;\n\tmul.rn.f32 f2, f2, $3;\n\t"
            "mul.rn.f32 f3, f3, $3;\n\tcvt.rn.bf16x2.f32 $0, f1, f0;\n\tcvt.rn.bf16x2.f32 $1, f3, f2;\n\t}",
            "=r,=r,r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
        return words(out, 2, loc, ip)

    @dsl_user_op
    def main_decode8(w: Int32, scale: Float32, *, loc=None, ip=None):
        """Eight E2M1 channels (byte k = channels 2k (low nibble), 2k + 1) times ``scale`` ->
        four BF16 pairs."""
        out = asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [Int32(w).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b8 b0, b1, b2, b3;\n\t.reg .b16 x0, x1, x2, x3, x4, x5, x6, x7;\n\t"
            ".reg .b32 h0, h1, h2, h3;\n\t.reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;\n\t"
            "mov.b32 {b0, b1, b2, b3}, $4;\n\t"
            "cvt.rn.f16x2.e2m1x2 h0, b0;\n\tcvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
            "cvt.rn.f16x2.e2m1x2 h2, b2;\n\tcvt.rn.f16x2.e2m1x2 h3, b3;\n\t"
            "mov.b32 {x0, x1}, h0;\n\tmov.b32 {x2, x3}, h1;\n\tmov.b32 {x4, x5}, h2;\n\tmov.b32 {x6, x7}, h3;\n\t"
            "cvt.f32.f16 f0, x0;\n\tcvt.f32.f16 f1, x1;\n\tcvt.f32.f16 f2, x2;\n\tcvt.f32.f16 f3, x3;\n\t"
            "cvt.f32.f16 f4, x4;\n\tcvt.f32.f16 f5, x5;\n\tcvt.f32.f16 f6, x6;\n\tcvt.f32.f16 f7, x7;\n\t"
            "mul.rn.f32 f0, f0, $5;\n\tmul.rn.f32 f1, f1, $5;\n\tmul.rn.f32 f2, f2, $5;\n\t"
            "mul.rn.f32 f3, f3, $5;\n\tmul.rn.f32 f4, f4, $5;\n\tmul.rn.f32 f5, f5, $5;\n\t"
            "mul.rn.f32 f6, f6, $5;\n\tmul.rn.f32 f7, f7, $5;\n\t"
            "cvt.rn.bf16x2.f32 $0, f1, f0;\n\tcvt.rn.bf16x2.f32 $1, f3, f2;\n\t"
            "cvt.rn.bf16x2.f32 $2, f5, f4;\n\tcvt.rn.bf16x2.f32 $3, f7, f6;\n\t}",
            "=r,=r,=r,=r,r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
        return words(out, 4, loc, ip)

    @dsl_user_op
    def bf16_pair(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
                "cvt.rn.bf16x2.f32 $0, $2, $1;",
                "=r,f,f",
                side_effects=False,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def bf16_bits(x: Float32, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Float32(x).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
                "=r,f",
                side_effects=False,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def exp2(x: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(
            asm(
                T.f32(),
                [Float32(x).ir_value(loc=loc, ip=ip)],
                "ex2.approx.ftz.f32 $0, $1;",
                "=f,f",
                side_effects=False,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def shfl_max(v: Float32, mask: int, *, loc=None, ip=None) -> Float32:
        return Float32(
            asm(
                T.f32(),
                [Float32(v).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .f32 o;\n\tshfl.sync.bfly.b32 o, $1, "
                + str(mask)
                + ", 0x1f, 0xffffffff;\n\t"
                "max.f32 $0, $1, o;\n\t}",
                "=f,f",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def shfl_sum(v: Float32, mask: int, *, loc=None, ip=None) -> Float32:
        return Float32(
            asm(
                T.f32(),
                [Float32(v).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .f32 o;\n\tshfl.sync.bfly.b32 o, $1, "
                + str(mask)
                + ", 0x1f, 0xffffffff;\n\t"
                "add.f32 $0, $1, o;\n\t}",
                "=f,f",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def st_cluster_f32x2(addr: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "st.shared::cluster.v2.f32 [$0], {$1, $2};",
            "r,f,f,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_shared_f32x2(addr: Int32, *, loc=None, ip=None):
        out = asm(
            llvm.StructType.get_literal([T.f32()] * 2),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.v2.f32 {$0, $1}, [$2];",
            "=f,=f,r,~{memory}",
            loc=loc,
            ip=ip,
        )
        return floats(out, 2, loc, ip)

    @dsl_user_op
    def ld_shared_f32x4(addr: Int32, *, loc=None, ip=None):
        out = asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
            "=f,=f,=f,=f,r,~{memory}",
            loc=loc,
            ip=ip,
        )
        return floats(out, 4, loc, ip)

    # ------------------------------------------------- query RoPE, as mla_rope_inplace computes it

    @dsl_user_op
    def bf16_halves(w: Int32, *, loc=None, ip=None):
        """BF16 pair (element 0 in the low half) -> two FP32 values, exact."""
        out = asm(
            llvm.StructType.get_literal([T.f32()] * 2),
            [Int32(w).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 lo, hi;\n\tshl.b32 lo, $2, 16;\n\tand.b32 hi, $2, 0xffff0000;\n\t"
            "mov.b32 $0, lo;\n\tmov.b32 $1, hi;\n\t}",
            "=f,=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
        return floats(out, 2, loc, ip)

    @dsl_user_op
    def ld_shared_b32(addr: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int32(addr).ir_value(loc=loc, ip=ip)],
                "ld.shared.b32 $0, [$1];",
                "=r,r,~{memory}",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def st_shared_b32(addr: Int32, w: Int32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
            "st.shared.b32 [$0], $1;",
            "r,r,~{memory}",
            loc=loc,
            ip=ip,
        )

    NEG_INF = float("-inf")

    class DirectSparseDecode:
        """One launch per (heads, selection widths, cluster, index/mapping types, readiness).

        ``heads`` is the MMA head count (16, 32 or 64); ``q_heads`` (default ``heads``) query
        heads are read and written: with 8 of 16, the other 8 rows of the query tile are zero
        and their results are discarded, so the real heads come out as the 16-head kernel
        computes them from a zero-padded query. Only the real heads' partial outputs and
        (max, sum) pairs go to the channel owners.

        CTA ``rank`` of a row's cluster owns main columns ``[rank * main_keys, ...)`` (key rows
        ``0 .. main_keys``) and SWA columns ``[rank * swa_keys, ...)`` (key rows after them). The
        two ranges are attended as separate parts and merged in registers, so with ``main_ready``
        and ``inputs_ready`` the whole main part runs before the dependent wait and only the SWA
        rows this layer's cache write may have produced are left for after it.
        """

        def __init__(
            self,
            heads: int,
            swa_width: int,
            main_width: int,
            cluster: int,
            mapped: bool,
            swa_index64: bool,
            main_index64: bool,
            inputs_ready: bool,
            main_ready: bool,
            use_pdl: bool,
            swa_vec: int = 16,
            main_vec: int = 8,
            q_heads: int = 0,
            q_rope: bool = False,
        ):
            self.q_heads = q_heads or heads
            self.heads = heads
            self.head_tiles = heads // 16
            self.ksplit = 8 // self.head_tiles
            self.swa_width = swa_width
            self.main_width = main_width
            self.cluster = cluster
            self.main_keys, self.swa_keys = cta_keys(swa_width, main_width, cluster)
            self.keys = self.main_keys + self.swa_keys
            self.mapped = mapped
            self.swa_index64 = swa_index64
            self.main_index64 = main_index64
            # Widest load (bytes) every row's payload start is aligned to.
            self.swa_vec = swa_vec
            self.main_vec = main_vec
            self.inputs_ready = inputs_ready
            self.main_ready = main_ready and main_width > 0
            # The main part runs before the dependent wait when its rows and the query are final.
            self.pre_main = self.main_ready and inputs_ready
            self.use_pdl = use_pdl
            self.q_rope = q_rope
            self.layout = smem_layout(heads, self.main_keys, self.swa_keys, cluster)

        @cute.jit
        def __call__(
            self,
            q: cute.Pointer,
            swa: cute.Pointer,
            swa_index: cute.Pointer,
            main: cute.Pointer,
            main_index: cute.Pointer,
            table: cute.Pointer,
            requests: cute.Pointer,
            visible: cute.Pointer,
            sink: cute.Pointer,
            out: cute.Pointer,
            rows: Int32,
            swa_stride: Int64,
            main_stride: Int64,
            swa_index_stride: Int64,
            main_index_stride: Int64,
            table_rows: Int32,
            table_cols: Int32,
            table_stride: Int64,
            page_size: Int64,
            max_positions: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            scale_log2: Float32,
            q_positions: cute.Pointer,
            q_cos_sin: cute.Pointer,
            stream,
        ):
            self.kernel(
                q,
                swa,
                swa_index,
                main,
                main_index,
                table,
                requests,
                visible,
                sink,
                out,
                swa_stride,
                main_stride,
                swa_index_stride,
                main_index_stride,
                table_rows,
                table_cols,
                table_stride,
                page_size,
                max_positions,
                swa_capacity,
                main_capacity,
                scale_log2,
                q_positions,
                q_cos_sin,
            ).launch(
                grid=[self.cluster, rows, 1],
                block=[_THREADS, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.layout["total"],
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.jit
        def _resolve(
            self,
            local,
            rank,
            row,
            swa_index,
            main_index,
            table,
            requests,
            visible,
            swa_index_stride,
            main_index_stride,
            table_rows,
            table_cols,
            table_stride,
            page_size,
            max_positions,
            swa_capacity,
            main_capacity,
            do_swa: cutlass.Constexpr,
            do_main: cutlass.Constexpr,
            sSlot,
            sKind,
        ):
            """Slot and kind (1 SWA, 2 main, 0 none) of key row ``local``, CSA2's mapping rules."""
            if const_expr(do_main and self.main_keys > 0):
                column = rank * self.main_keys + local
                if (local < self.main_keys) & (column < self.main_width):
                    logical = ld_global_index(
                        Int64(main_index.toint())
                        + (Int64(row) * main_index_stride + Int64(column))
                        * (8 if self.main_index64 else 4),
                        self.main_index64,
                    )
                    slot = logical
                    if const_expr(self.mapped):
                        slot = Int64(-1)
                        request = ld_global_index(Int64(requests.toint()) + Int64(row) * 8, True)
                        seen = ld_global_index(Int64(visible.toint()) + Int64(row) * 8, True)
                        if (table_rows > 0) & (table_cols > 0) & (logical >= 0):
                            bounded = logical
                            if bounded > max_positions:
                                bounded = max_positions
                            if bounded > Int64(2**31 - 1):
                                bounded = Int64(2**31 - 1)
                            page = Int64(Int32(bounded) // Int32(page_size))
                            safe_request = request
                            if safe_request < 0:
                                safe_request = Int64(0)
                            if safe_request > Int64(table_rows) - 1:
                                safe_request = Int64(table_rows) - 1
                            safe_page = page
                            if safe_page > Int64(table_cols) - 1:
                                safe_page = Int64(table_cols) - 1
                            physical = Int64(
                                ld_global_s32(
                                    Int64(table.toint())
                                    + (safe_request * table_stride + safe_page) * 4
                                )
                            )
                            valid = (logical < max_positions) & (page < Int64(table_cols))
                            valid = valid & (request >= 0) & (request < Int64(table_rows))
                            valid = valid & (physical >= 0) & (logical < seen)
                            if valid:
                                slot = physical * page_size + Int64(
                                    Int32(bounded) % Int32(page_size)
                                )
                    kind = Int32(0)
                    if (slot >= 0) & (slot < main_capacity):
                        kind = Int32(2)
                    sSlot[local] = slot
                    sKind[local] = kind
            if const_expr(do_swa and self.swa_keys > 0):
                column = rank * self.swa_keys + (local - self.main_keys)
                if (local >= self.main_keys) & (column < self.swa_width):
                    slot = ld_global_index(
                        Int64(swa_index.toint())
                        + (Int64(row) * swa_index_stride + Int64(column))
                        * (8 if self.swa_index64 else 4),
                        self.swa_index64,
                    )
                    kind = Int32(0)
                    if (slot >= 0) & (slot < swa_capacity):
                        kind = Int32(1)
                    sSlot[local] = slot
                    sKind[local] = kind

        @cute.jit
        def _decode_rows(
            self,
            warp,
            lane,
            k_addr,
            swa,
            main,
            swa_stride,
            main_stride,
            sSlot,
            sKind,
            first: cutlass.Constexpr,
            count: cutlass.Constexpr,
            want: cutlass.Constexpr,
        ):
            """Decode key rows ``first .. first + count`` of the wanted kinds (bit 1 SWA, bit 2
            main, bit 4 empty) into BF16 shared-memory rows; lane ``l`` owns channels
            ``16l .. 16l + 15``."""
            per_warp: cutlass.Constexpr = ceil_div(count, 8)
            data = cute.make_rmem_tensor(cute.make_layout((per_warp, 4), stride=(4, 1)), Int32)
            scale_byte = cute.make_rmem_tensor(cute.make_layout((per_warp,)), Int32)
            kinds = cute.make_rmem_tensor(cute.make_layout((per_warp,)), Int32)
            swa_base = Int64(swa.toint())
            main_base = Int64(main.toint())
            for i in cutlass.range_constexpr(per_warp):
                r = first + warp + 8 * i
                kinds[i] = Int32(-1)
                if r < first + count:
                    kind = sKind[r]
                    slot = sSlot[r]
                    if const_expr(want & 1):
                        if kind == 1:
                            row_base = swa_base + slot * swa_stride
                            src = row_base + Int64(16 * lane)
                            if const_expr(self.swa_vec == 16):
                                w0, w1, w2, w3 = ld_global_v4(src)
                                data[i, 0] = w0
                                data[i, 1] = w1
                                data[i, 2] = w2
                                data[i, 3] = w3
                            elif const_expr(self.swa_vec == 8):
                                w0, w1 = ld_global_v2(src)
                                w2, w3 = ld_global_v2(src + 8)
                                data[i, 0] = w0
                                data[i, 1] = w1
                                data[i, 2] = w2
                                data[i, 3] = w3
                            else:
                                for wi in cutlass.range_constexpr(4):
                                    data[i, wi] = ld_global_b32(src + 4 * wi)
                            scale_byte[i] = ld_global_u8(row_base + Int64(HEAD_DIM + lane // 2))
                            kinds[i] = Int32(1)
                    if const_expr(want & 2):
                        if kind == 2:
                            row_base = main_base + slot * main_stride
                            src = row_base + Int64(8 * lane)
                            if const_expr(self.main_vec == 8):
                                w0, w1 = ld_global_v2(src)
                                data[i, 0] = w0
                                data[i, 1] = w1
                            else:
                                data[i, 0] = ld_global_b32(src)
                                data[i, 1] = ld_global_b32(src + 4)
                            scale_byte[i] = ld_global_u8(row_base + Int64(HEAD_DIM // 2 + lane))
                            kinds[i] = Int32(2)
                    if const_expr(want & 4):
                        if kind == 0:
                            kinds[i] = Int32(0)
            for i in cutlass.range_constexpr(per_warp):
                r = first + warp + 8 * i
                dst = k_addr + r * _KEY_ROW + 32 * lane
                if kinds[i] == 1:
                    s = swa_scale(scale_byte[i])
                    o0, o1 = swa_decode4(data[i, 0], s)
                    o2, o3 = swa_decode4(data[i, 1], s)
                    o4, o5 = swa_decode4(data[i, 2], s)
                    o6, o7 = swa_decode4(data[i, 3], s)
                    st_shared_v4(dst, o0, o1, o2, o3)
                    st_shared_v4(dst + 16, o4, o5, o6, o7)
                if kinds[i] == 2:
                    s = e4m3_scale(scale_byte[i])
                    o0, o1, o2, o3 = main_decode8(data[i, 0], s)
                    o4, o5, o6, o7 = main_decode8(data[i, 1], s)
                    st_shared_v4(dst, o0, o1, o2, o3)
                    st_shared_v4(dst + 16, o4, o5, o6, o7)
                if kinds[i] == 0:
                    st_shared_v4(dst, Int32(0), Int32(0), Int32(0), Int32(0))
                    st_shared_v4(dst + 16, Int32(0), Int32(0), Int32(0), Int32(0))

        @cute.jit
        def _part(
            self,
            tidx,
            warp,
            lane,
            q_addr,
            k_addr,
            sp_addr,
            p_addr,
            sKind,
            sML,
            oacc,
            scale_log2: Float32,
            key0: cutlass.Constexpr,
            nkeys: cutlass.Constexpr,
            part: cutlass.Constexpr,
        ):
            """Attention of every head over key rows ``key0 .. key0 + nkeys``: unnormalized
            ``O = exp2(S - m) V`` into ``oacc`` and per-head ``(m, sum exp2(S - m))`` into
            ``sML[part]``, with ``S`` in log2 units."""
            heads: cutlass.Constexpr = self.heads
            head_tiles: cutlass.Constexpr = self.head_tiles
            ksplit: cutlass.Constexpr = self.ksplit
            prow: cutlass.Constexpr = self.layout["prow"]
            g = lane // 4
            t = lane % 4
            ht = warp % head_tiles
            ks = warp // head_tiles
            a_row = 16 * ht + lane % 16
            a_col = 8 * (lane // 16)

            # S = Q K^T: every warp a (head tile, channel slice), partial sums over its slice.
            n_tiles: cutlass.Constexpr = nkeys // 8
            k_slice: cutlass.Constexpr = HEAD_DIM // ksplit
            acc = cute.make_rmem_tensor(cute.make_layout((n_tiles, 4), stride=(4, 1)), Float32)
            acc.fill(0.0)
            b_key = lane % 8 + 8 * (lane // 16)
            b_col = 8 * ((lane // 8) % 2)
            for kk in cutlass.range_constexpr(k_slice // 16):
                d0 = ks * k_slice + 16 * kk
                a0, a1, a2, a3 = ldmatrix_x4(q_addr + a_row * _KEY_ROW + (d0 + a_col) * 2, False)
                for np in cutlass.range_constexpr(n_tiles // 2):
                    b0, b1, b2, b3 = ldmatrix_x4(
                        k_addr + (key0 + 16 * np + b_key) * _KEY_ROW + (d0 + b_col) * 2, False
                    )
                    c0, c1, c2, c3 = mma_bf16(
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        acc[2 * np, 0],
                        acc[2 * np, 1],
                        acc[2 * np, 2],
                        acc[2 * np, 3],
                    )
                    acc[2 * np, 0] = c0
                    acc[2 * np, 1] = c1
                    acc[2 * np, 2] = c2
                    acc[2 * np, 3] = c3
                    e0, e1, e2, e3 = mma_bf16(
                        a0,
                        a1,
                        a2,
                        a3,
                        b2,
                        b3,
                        acc[2 * np + 1, 0],
                        acc[2 * np + 1, 1],
                        acc[2 * np + 1, 2],
                        acc[2 * np + 1, 3],
                    )
                    acc[2 * np + 1, 0] = e0
                    acc[2 * np + 1, 1] = e1
                    acc[2 * np + 1, 2] = e2
                    acc[2 * np + 1, 3] = e3
            part_addr = sp_addr + ks * (heads * nkeys * 4)
            for n in cutlass.range_constexpr(n_tiles):
                h0 = 16 * ht + g
                st_shared_f32x2(part_addr + (h0 * nkeys + 8 * n + 2 * t) * 4, acc[n, 0], acc[n, 1])
                st_shared_f32x2(
                    part_addr + ((h0 + 8) * nkeys + 8 * n + 2 * t) * 4, acc[n, 2], acc[n, 3]
                )
            cute.arch.sync_threads()
            # Sum the channel slices into slice 0, in log2 units; invalid keys -inf.
            for i in cutlass.range_constexpr(ceil_div(heads * nkeys, _THREADS)):
                idx = tidx + _THREADS * i
                if idx < heads * nkeys:
                    total = Float32(0.0)
                    for s in cutlass.range_constexpr(ksplit):
                        total = total + ld_shared_f32(sp_addr + (s * heads * nkeys + idx) * 4)
                    value = total * scale_log2
                    if sKind[key0 + idx % nkeys] == 0:
                        value = Float32(NEG_INF)
                    st_shared_f32(sp_addr + idx * 4, value)
            cute.arch.sync_threads()

            # Softmax numerators per head (a warp per head, heads / 8 each), BF16 probabilities.
            per_lane: cutlass.Constexpr = ceil_div(nkeys, 32)
            for hh in cutlass.range_constexpr(heads // 8):
                h = warp * (heads // 8) + hh
                vals = cute.make_rmem_tensor(cute.make_layout((per_lane,)), Float32)
                m = Float32(NEG_INF)
                for i in cutlass.range_constexpr(per_lane):
                    j = lane + 32 * i
                    vals[i] = Float32(NEG_INF)
                    if j < nkeys:
                        vals[i] = ld_shared_f32(sp_addr + (h * nkeys + j) * 4)
                    m = cute.arch.fmax(m, vals[i])
                for mask in (16, 8, 4, 2, 1):
                    m = shfl_max(m, mask)
                safe = m
                if m == Float32(NEG_INF):
                    safe = Float32(0.0)
                total = Float32(0.0)
                for i in cutlass.range_constexpr(per_lane):
                    j = lane + 32 * i
                    p = exp2(vals[i] - safe)
                    total = total + p
                    if j < nkeys:
                        st_shared_b16(p_addr + h * prow + 2 * j, bf16_bits(p))
                for mask in (16, 8, 4, 2, 1):
                    total = shfl_sum(total, mask)
                if lane == 0:
                    sML[part * 2 * heads + 2 * h] = m
                    sML[part * 2 * heads + 2 * h + 1] = total
            cute.arch.sync_threads()

            # O = P V: every warp a (head tile, channel slice).
            d_slice: cutlass.Constexpr = HEAD_DIM // self.ksplit
            v_key = lane % 8 + 8 * ((lane // 8) % 2)
            v_col = 8 * (lane // 16)
            for kk in cutlass.range_constexpr(nkeys // 16):
                a0, a1, a2, a3 = ldmatrix_x4(p_addr + a_row * prow + (16 * kk + a_col) * 2, False)
                for np in cutlass.range_constexpr(d_slice // 16):
                    n0 = ks * d_slice + 16 * np
                    b0, b1, b2, b3 = ldmatrix_x4(
                        k_addr + (key0 + 16 * kk + v_key) * _KEY_ROW + (n0 + v_col) * 2, True
                    )
                    c0, c1, c2, c3 = mma_bf16(
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        oacc[2 * np, 0],
                        oacc[2 * np, 1],
                        oacc[2 * np, 2],
                        oacc[2 * np, 3],
                    )
                    oacc[2 * np, 0] = c0
                    oacc[2 * np, 1] = c1
                    oacc[2 * np, 2] = c2
                    oacc[2 * np, 3] = c3
                    e0, e1, e2, e3 = mma_bf16(
                        a0,
                        a1,
                        a2,
                        a3,
                        b2,
                        b3,
                        oacc[2 * np + 1, 0],
                        oacc[2 * np + 1, 1],
                        oacc[2 * np + 1, 2],
                        oacc[2 * np + 1, 3],
                    )
                    oacc[2 * np + 1, 0] = e0
                    oacc[2 * np + 1, 1] = e1
                    oacc[2 * np + 1, 2] = e2
                    oacc[2 * np + 1, 3] = e3

        @cute.kernel
        def kernel(
            self,
            q: cute.Pointer,
            swa: cute.Pointer,
            swa_index: cute.Pointer,
            main: cute.Pointer,
            main_index: cute.Pointer,
            table: cute.Pointer,
            requests: cute.Pointer,
            visible: cute.Pointer,
            sink: cute.Pointer,
            out: cute.Pointer,
            swa_stride: Int64,
            main_stride: Int64,
            swa_index_stride: Int64,
            main_index_stride: Int64,
            table_rows: Int32,
            table_cols: Int32,
            table_stride: Int64,
            page_size: Int64,
            max_positions: Int64,
            swa_capacity: Int64,
            main_capacity: Int64,
            scale_log2: Float32,
            q_positions: cute.Pointer,
            q_cos_sin: cute.Pointer,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            rank, row, _ = cute.arch.block_idx()
            warp = tidx // 32
            lane = tidx % 32
            g = lane // 4
            t = lane % 4

            heads: cutlass.Constexpr = self.heads
            keys: cutlass.Constexpr = self.keys
            main_keys: cutlass.Constexpr = self.main_keys
            swa_keys: cutlass.Constexpr = self.swa_keys
            head_tiles: cutlass.Constexpr = self.head_tiles
            cluster: cutlass.Constexpr = self.cluster
            lay = self.layout

            smem = SmemAllocator()
            sBase = smem.allocate_tensor(
                Uint8, cute.make_layout((lay["bytes"],)), byte_alignment=128
            )
            sSlot = smem.allocate_tensor(Int64, cute.make_layout((keys,)), byte_alignment=16)
            sKind = smem.allocate_tensor(Int32, cute.make_layout((keys,)), byte_alignment=16)
            sML = smem.allocate_tensor(Float32, cute.make_layout((4 * heads,)), byte_alignment=16)
            sSink = smem.allocate_tensor(Float32, cute.make_layout((heads,)), byte_alignment=16)
            # The query row's RoPE cos (32) then sin (32).
            sRope = smem.allocate_tensor(Float32, cute.make_layout((64,)), byte_alignment=16)
            base = Int32(sBase.iterator.toint())
            share: cutlass.Constexpr = HEAD_DIM // cluster
            d_slice: cutlass.Constexpr = HEAD_DIM // self.ksplit
            o_tiles: cutlass.Constexpr = d_slice // 8
            k_addr = base
            sp_addr = base + lay["sp"]
            q_addr = base + lay["q"]
            p_addr = base + lay["p"]
            recv_addr = base + lay["recv"]
            block: cutlass.Constexpr = lay["block"]
            ml_off: cutlass.Constexpr = heads * (HEAD_DIM // self.cluster) * 4

            if const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()

            q_heads: cutlass.Constexpr = self.q_heads
            q_base = Int64(q.toint()) + Int64(row) * Int64(q_heads * HEAD_DIM * 2)
            q_chunks: cutlass.Constexpr = q_heads * HEAD_DIM // 8
            if tidx < q_heads:
                sSink[tidx] = ld_global_f32(Int64(sink.toint()) + Int64(4 * tidx)) * LOG2E
            if const_expr(q_heads < heads):
                # Padding heads: a zero query row (their outputs are not stored).
                if tidx >= q_heads:
                    if tidx < heads:
                        sSink[tidx] = Float32(0.0)
                for i in cutlass.range_constexpr((heads - q_heads) * HEAD_DIM // 8 // _THREADS):
                    c = q_chunks + tidx + _THREADS * i
                    st_shared_v4(
                        q_addr + (c // 64) * _KEY_ROW + (c % 64) * 16,
                        Int32(0),
                        Int32(0),
                        Int32(0),
                        Int32(0),
                    )
            oacc0 = cute.make_rmem_tensor(cute.make_layout((o_tiles, 4), stride=(4, 1)), Float32)
            oacc1 = cute.make_rmem_tensor(cute.make_layout((o_tiles, 4), stride=(4, 1)), Float32)
            oacc0.fill(0.0)
            oacc1.fill(0.0)
            # The row's RoPE coefficients (its position is final long before this layer).
            if const_expr(self.q_rope):
                if tidx < 64:
                    sRope[tidx] = ld_global_f32(
                        Int64(q_cos_sin.toint())
                        + Int64(ld_global_s32(Int64(q_positions.toint()) + Int64(row) * 4))
                        * Int64(64 * 4)
                        + Int64(4 * tidx)
                    )

            # ---- before the dependent wait: inputs final before the preceding launch ----
            if const_expr(self.inputs_ready):
                for i in cutlass.range_constexpr(q_chunks // _THREADS):
                    c = tidx + _THREADS * i
                    cp_async16(
                        q_addr + (c // 64) * _KEY_ROW + (c % 64) * 16, q_base + Int64(16 * c)
                    )
            if tidx < keys:
                sKind[tidx] = Int32(0)
                sSlot[tidx] = Int64(-1)
                self._resolve(
                    tidx,
                    rank,
                    row,
                    swa_index,
                    main_index,
                    table,
                    requests,
                    visible,
                    swa_index_stride,
                    main_index_stride,
                    table_rows,
                    table_cols,
                    table_stride,
                    page_size,
                    max_positions,
                    swa_capacity,
                    main_capacity,
                    self.inputs_ready,
                    self.main_ready,
                    sSlot,
                    sKind,
                )
            if const_expr(self.pre_main):
                cute.arch.sync_threads()
                self._decode_rows(
                    warp,
                    lane,
                    k_addr,
                    swa,
                    main,
                    swa_stride,
                    main_stride,
                    sSlot,
                    sKind,
                    0,
                    main_keys,
                    6,
                )
                cp_async_wait_all()
                cute.arch.sync_threads()
                if const_expr(self.q_rope):
                    self._rope_query(tidx, q_addr, sRope)
                    cute.arch.sync_threads()
                self._part(
                    tidx,
                    warp,
                    lane,
                    q_addr,
                    k_addr,
                    sp_addr,
                    p_addr,
                    sKind,
                    sML,
                    oacc0,
                    scale_log2,
                    0,
                    main_keys,
                    0,
                )

            if const_expr(self.use_pdl):
                griddepcontrol_wait()

            # ---- after it: this layer's cache rows (and anything not promised ready) ----
            if const_expr(not self.inputs_ready):
                for i in cutlass.range_constexpr(q_chunks // _THREADS):
                    c = tidx + _THREADS * i
                    cp_async16(
                        q_addr + (c // 64) * _KEY_ROW + (c % 64) * 16, q_base + Int64(16 * c)
                    )
            if const_expr((not self.inputs_ready) or (not self.main_ready)):
                if tidx < keys:
                    self._resolve(
                        tidx,
                        rank,
                        row,
                        swa_index,
                        main_index,
                        table,
                        requests,
                        visible,
                        swa_index_stride,
                        main_index_stride,
                        table_rows,
                        table_cols,
                        table_stride,
                        page_size,
                        max_positions,
                        swa_capacity,
                        main_capacity,
                        not self.inputs_ready,
                        not self.main_ready,
                        sSlot,
                        sKind,
                    )
            cute.arch.sync_threads()
            if const_expr(self.pre_main):
                self._decode_rows(
                    warp,
                    lane,
                    k_addr,
                    swa,
                    main,
                    swa_stride,
                    main_stride,
                    sSlot,
                    sKind,
                    main_keys,
                    swa_keys,
                    5,
                )
            else:
                self._decode_rows(
                    warp,
                    lane,
                    k_addr,
                    swa,
                    main,
                    swa_stride,
                    main_stride,
                    sSlot,
                    sKind,
                    0,
                    keys,
                    7,
                )
            cp_async_wait_all()
            cute.arch.sync_threads()
            if const_expr(self.q_rope and not self.pre_main):
                self._rope_query(tidx, q_addr, sRope)
                cute.arch.sync_threads()
            if const_expr((not self.pre_main) and main_keys > 0):
                self._part(
                    tidx,
                    warp,
                    lane,
                    q_addr,
                    k_addr,
                    sp_addr,
                    p_addr,
                    sKind,
                    sML,
                    oacc0,
                    scale_log2,
                    0,
                    main_keys,
                    0,
                )
            if const_expr(swa_keys > 0):
                self._part(
                    tidx,
                    warp,
                    lane,
                    q_addr,
                    k_addr,
                    sp_addr,
                    p_addr,
                    sKind,
                    sML,
                    oacc1,
                    scale_log2,
                    main_keys,
                    swa_keys,
                    1,
                )
            cute.arch.sync_threads()

            # ---- merge the parts, then send this CTA's partial to the channel owners ----
            ht = warp % head_tiles
            ds = warp // head_tiles
            # This CTA's block in every destination's receive buffer.
            send_base = recv_addr + rank * block
            # Each thread holds fragment rows ``g`` and ``g + 8``; with 8 query heads the second are
            # padding.
            halves: cutlass.Constexpr = 1 if q_heads < heads else 2
            for half in cutlass.range_constexpr(halves):
                h = 16 * ht + g + 8 * half
                f0 = Float32(0.0)
                f1 = Float32(0.0)
                if const_expr(main_keys > 0 and swa_keys > 0):
                    mm0 = sML[2 * h]
                    mm1 = sML[2 * heads + 2 * h]
                    mtop = cute.arch.fmax(mm0, mm1)
                    if mm0 != Float32(NEG_INF):
                        f0 = exp2(mm0 - mtop)
                    if mm1 != Float32(NEG_INF):
                        f1 = exp2(mm1 - mtop)
                elif const_expr(main_keys > 0):
                    f0 = Float32(1.0)
                else:
                    f1 = Float32(1.0)
                for n in cutlass.range_constexpr(o_tiles):
                    d = ds * d_slice + 8 * n + 2 * t
                    v0 = oacc0[n, 2 * half] * f0 + oacc1[n, 2 * half] * f1
                    v1 = oacc0[n, 2 * half + 1] * f0 + oacc1[n, 2 * half + 1] * f1
                    if const_expr(lay["dedicated"]):
                        dest, offset = self._channel_owner(h, d)
                        st_cluster_f32x2(mapa(send_base + offset * 4, dest), v0, v1)
                    else:
                        oacc0[n, 2 * half] = v0
                        oacc0[n, 2 * half + 1] = v1
            # This CTA's (max, sum) per head over both parts, sent with its partial outputs.
            ml_top = Float32(NEG_INF)
            ml_total = Float32(0.0)
            pm0 = Float32(NEG_INF)
            pl0 = Float32(0.0)
            pm1 = Float32(NEG_INF)
            pl1 = Float32(0.0)
            if tidx < q_heads:
                if const_expr(main_keys > 0):
                    pm0 = sML[2 * tidx]
                    pl0 = sML[2 * tidx + 1]
                if const_expr(swa_keys > 0):
                    pm1 = sML[2 * heads + 2 * tidx]
                    pl1 = sML[2 * heads + 2 * tidx + 1]
                ml_top = cute.arch.fmax(pm0, pm1)
                if pm0 != Float32(NEG_INF):
                    ml_total = ml_total + pl0 * exp2(pm0 - ml_top)
                if pm1 != Float32(NEG_INF):
                    ml_total = ml_total + pl1 * exp2(pm1 - ml_top)
                if const_expr(lay["dedicated"]):
                    for dest in cutlass.range_constexpr(cluster):
                        st_cluster_f32x2(
                            mapa(send_base + ml_off + tidx * 8, dest), ml_top, ml_total
                        )
            if const_expr(not lay["dedicated"]):
                # The receive buffer reuses the key rows: every CTA must be done with them.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                for half in cutlass.range_constexpr(halves):
                    h = 16 * ht + g + 8 * half
                    for n in cutlass.range_constexpr(o_tiles):
                        d = ds * d_slice + 8 * n + 2 * t
                        dest, offset = self._channel_owner(h, d)
                        st_cluster_f32x2(
                            mapa(send_base + offset * 4, dest),
                            oacc0[n, 2 * half],
                            oacc0[n, 2 * half + 1],
                        )
                if tidx < q_heads:
                    for dest in cutlass.range_constexpr(cluster):
                        st_cluster_f32x2(
                            mapa(send_base + ml_off + tidx * 8, dest), ml_top, ml_total
                        )
            # Every source's block (and its arrive's release) is visible after the wait.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # ---- this CTA's channels of every head: sum the sources with the sink ----
            items: cutlass.Constexpr = q_heads * share // 4
            out_base = Int64(out.toint()) + Int64(row) * Int64(q_heads * HEAD_DIM * 2)
            for i in cutlass.range_constexpr(ceil_div(items, _THREADS)):
                item = tidx + _THREADS * i
                if item < items:
                    h = item // (share // 4)
                    dd = 4 * (item % (share // 4))
                    w01, w23 = self._merge4(recv_addr, sSink, h, h * share + dd)
                    st_global_v2(out_base + Int64((h * HEAD_DIM + rank * share + dd) * 2), w01, w23)

        @cute.jit
        def _channel_owner(self, h, d):
            """(CTA, offset in its receive block) of channel ``d`` of head ``h``: a CTA owns
            ``512 / cluster`` channels of every head."""
            share: cutlass.Constexpr = HEAD_DIM // self.cluster
            return d // share, h * share + d % share

        @cute.jit
        def _merge4(self, recv_addr, sSink, h, offset):
            """Four consecutive channels of head ``h`` (at ``offset`` of every source's receive
            block) merged over the cluster with the sink: BF16 pairs."""
            cluster: cutlass.Constexpr = self.cluster
            block: cutlass.Constexpr = self.layout["block"]
            ml_off: cutlass.Constexpr = self.heads * (HEAD_DIM // cluster) * 4
            sink_log2 = sSink[h]
            rtop = sink_log2
            for src in cutlass.range_constexpr(cluster):
                ms, _ = ld_shared_f32x2(recv_addr + src * block + ml_off + h * 8)
                rtop = cute.arch.fmax(rtop, ms)
            denom = exp2(sink_log2 - rtop)
            v0 = Float32(0.0)
            v1 = Float32(0.0)
            v2 = Float32(0.0)
            v3 = Float32(0.0)
            for src in cutlass.range_constexpr(cluster):
                ms, ls = ld_shared_f32x2(recv_addr + src * block + ml_off + h * 8)
                f = Float32(0.0)
                if ms != Float32(NEG_INF):
                    f = exp2(ms - rtop)
                denom = denom + f * ls
                x0, x1, x2, x3 = ld_shared_f32x4(recv_addr + src * block + offset * 4)
                v0 = v0 + f * x0
                v1 = v1 + f * x1
                v2 = v2 + f * x2
                v3 = v3 + f * x3
            inv = Float32(1.0) / denom
            return bf16_pair(v0 * inv, v1 * inv), bf16_pair(v2 * inv, v3 * inv)

        @cute.jit
        def _rope_query(self, tidx, q_addr, sRope):
            """The query's interleaved RoPE on its last 64 channels, in shared memory, as
            mla_rope_inplace computes it (FP32, one BF16 rounding)."""
            rope_n: cutlass.Constexpr = ceil_div(self.q_heads * 32, _THREADS)
            for i in cutlass.range_constexpr(rope_n):
                pair = tidx + _THREADS * i
                addr = q_addr + (pair // 32) * _KEY_ROW + (HEAD_DIM - 64 + 2 * (pair % 32)) * 2
                a, b = bf16_halves(ld_shared_b32(addr))
                c = sRope[pair % 32]
                sn = sRope[32 + pair % 32]
                # mla_rope_inplace's contraction (verified bitwise on 51M values).
                x = fma_rn(c, a, -mul_rn(sn, b))
                y = fma_rn(sn, a, mul_rn(c, b))
                st_shared_b32(addr, bf16_pair(x, y))

    return DirectSparseDecode


_COMPILED: dict = {}


def _pointer(tensor: torch.Tensor | None, dtype, align: int):
    import cutlass.cute as cute

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    address = 0 if tensor is None else tensor.data_ptr()
    while align > 1 and address % align:
        align //= 2
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def direct_sparse_decode(
    q: torch.Tensor,
    swa_pool: torch.Tensor,
    swa_indices: torch.Tensor,
    main_pool: torch.Tensor | None,
    main_indices: torch.Tensor | None,
    sink: torch.Tensor,
    scale: float,
    *,
    main_mapping=None,
    output: torch.Tensor | None = None,
    inputs_ready: bool = False,
    main_ready: bool = False,
    query_rope=None,
) -> torch.Tensor:
    """Sparse attention of ``q [T, H, 512]`` (BF16) over its selected packed cache rows.

    ``swa_pool [*, 528]`` / ``main_pool [*, 288]`` are uint8 rows (unit column stride, any row
    stride); ``swa_indices [T, Ws]`` are SWA slots and ``main_indices [T, Wm]`` main slots, or
    logical positions mapped in-kernel when ``main_mapping = (page_table, page_size,
    max_positions, requests, visible)`` is given (CSA2's owner page-table rules). ``sink [H]``
    FP32 logits, ``scale`` the softmax scale. Returns BF16 ``[T, H, 512]`` (``output`` if given).
    ``inputs_ready`` / ``main_ready``: the query, SWA slots and page tables / the main
    selections and rows were final before the launch preceding this one on the stream started.

    ``query_rope = (positions, cos_sin)``: ``q`` is the projected query before its interleaved RoPE
    (int32 positions ``[T]``, FP32 table of 32 cos then 32 sin per position), rotated in the
    launch.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    rows, q_heads, dim = q.shape
    if dim != HEAD_DIM or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("CSA2 direct decode needs a contiguous BF16 query [T, H, 512]")
    heads = mma_heads(q_heads)
    if not 0 < rows <= MAX_ROWS or not heads:
        raise ValueError(
            f"CSA2 direct decode serves 1-{MAX_ROWS} query rows of 8, 16, 32 or 64 heads"
        )
    has_main = main_indices is not None and main_indices.shape[1] > 0
    for pool, width in (
        (swa_pool, SWA_ROW_BYTES),
        (main_pool if has_main else None, MAIN_ROW_BYTES),
    ):
        if pool is not None and (
            pool.dtype != torch.uint8
            or pool.ndim != 2
            or pool.shape[1] != width
            or pool.stride(1) != 1
        ):
            raise ValueError("CSA2 direct decode needs packed uint8 cache rows")
    for indices in (swa_indices, main_indices if has_main else None):
        if indices is not None and (
            indices.dtype not in (torch.int32, torch.int64)
            or indices.ndim != 2
            or indices.shape[0] != rows
            or (indices.shape[1] > 1 and indices.stride(1) != 1)
        ):
            raise ValueError("CSA2 direct decode needs [T, W] integer selections")
    if sink.dtype != torch.float32 or sink.shape != (q_heads,) or not sink.is_contiguous():
        raise ValueError("CSA2 direct decode needs FP32 sinks per head")
    if output is None:
        output = torch.empty_like(q)
    elif output.shape != q.shape or output.dtype != q.dtype or not output.is_contiguous():
        raise ValueError("CSA2 direct decode output must match the contiguous query")
    if q.data_ptr() % 16 or output.data_ptr() % 8:
        raise ValueError("CSA2 direct decode needs a 16-byte aligned query and 8-byte output")
    swa_width = int(swa_indices.shape[1])
    main_width = int(main_indices.shape[1]) if has_main else 0
    mapped = has_main and main_mapping is not None
    if mapped:
        table, page_size, max_positions, requests, visible = main_mapping
        if (
            table.dtype != torch.int32
            or table.ndim != 2
            or (table.shape[1] > 1 and table.stride(1) != 1)
        ):
            raise ValueError("CSA2 direct decode page table must be int32 [rows, pages]")
        requests = requests.reshape(-1).to(torch.int64).contiguous()
        visible = visible.reshape(-1).to(torch.int64).contiguous()
        if requests.numel() != rows or visible.numel() != rows:
            raise ValueError("CSA2 direct decode mapping rows must match the queries")
        table_rows, table_cols = int(table.shape[0]), int(table.shape[1])
        table_stride = int(table.stride(0)) if table_rows > 1 else table_cols
    else:
        table = requests = visible = None
        page_size, max_positions = 1, 2**62
        table_rows = table_cols = table_stride = 1

    def widest(pool, widths):
        return next((w for w in widths if pool.data_ptr() % w == 0 and pool.stride(0) % w == 0), 0)

    swa_vec = widest(swa_pool, (16, 8, 4))
    main_vec = widest(main_pool, (8, 4)) if has_main else 8
    if not swa_vec or not main_vec:
        raise ValueError("CSA2 direct decode needs 4-byte aligned cache rows")
    cluster = plan_split(swa_width, main_width)
    if not direct_decode_fits(q_heads, swa_width, main_width):
        raise ValueError("CSA2 direct decode: selection too wide for one CTA's shared memory")
    q_positions = q_table = None
    if query_rope is not None:
        q_positions, q_table = query_rope
        if (
            q_positions.dtype != torch.int32
            or q_positions.numel() != rows
            or not q_positions.is_contiguous()
            or q_table.dtype != torch.float32
            or q_table.numel() % 64
            or not q_table.is_contiguous()
        ):
            raise ValueError(
                "CSA2 direct decode query RoPE needs int32 positions and a 64-wide table"
            )
    q_rope = query_rope is not None
    swa_index64 = swa_indices.dtype == torch.int64
    main_index64 = has_main and main_indices.dtype == torch.int64
    use_pdl = get_env_enable_pdl()
    config = (
        heads,
        swa_width,
        main_width,
        cluster,
        mapped,
        swa_index64,
        main_index64,
        bool(inputs_ready),
        bool(main_ready),
        use_pdl,
        swa_vec,
        main_vec,
        q_heads,
        q_rope,
    )
    key = (q.device.index, *config)
    swa_index_type = cutlass.Int64 if swa_index64 else cutlass.Int32
    main_index_type = cutlass.Int64 if main_index64 else cutlass.Int32
    # Fixed pointer alignments keep one compiled signature; the kernel addresses raw bytes
    # with the vector widths derived above.
    args = (
        _pointer(q, cutlass.BFloat16, 2),
        _pointer(swa_pool, cutlass.Uint8, 1),
        _pointer(swa_indices, swa_index_type, swa_indices.element_size()),
        _pointer(main_pool if has_main else swa_pool, cutlass.Uint8, 1),
        _pointer(
            main_indices if has_main else swa_indices,
            main_index_type,
            (main_indices if has_main else swa_indices).element_size(),
        ),
        _pointer(table, cutlass.Int32, 4),
        _pointer(requests, cutlass.Int64, 8),
        _pointer(visible, cutlass.Int64, 8),
        _pointer(sink, cutlass.Float32, 4),
        _pointer(output, cutlass.BFloat16, 2),
        cutlass.Int32(rows),
        cutlass.Int64(swa_pool.stride(0)),
        cutlass.Int64(main_pool.stride(0) if has_main else SWA_ROW_BYTES),
        cutlass.Int64(swa_indices.stride(0)),
        cutlass.Int64(main_indices.stride(0) if has_main else 0),
        cutlass.Int32(table_rows),
        cutlass.Int32(table_cols),
        cutlass.Int64(table_stride),
        cutlass.Int64(page_size),
        cutlass.Int64(max_positions),
        cutlass.Int64(swa_pool.shape[0]),
        cutlass.Int64(main_pool.shape[0] if has_main else 0),
        cutlass.Float32(scale * LOG2E),
        _pointer(q_positions, cutlass.Int32, 4),
        _pointer(q_table, cutlass.Float32, 4),
        cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up CSA2 direct decode before CUDA Graph capture")
        kernel = _kernel_type()(*config)
        compiled = _COMPILED[key] = cute.compile(kernel, *args)
    compiled(*args)
    return output


__all__ = [
    "MAX_ROWS",
    "direct_decode_device_supported",
    "direct_sparse_decode",
]
