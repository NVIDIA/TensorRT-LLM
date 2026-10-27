# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES

"""CuTe DSL Q8KV4 paged decode indexer proxy scores for two or four index heads.

K pages use the vLLM packed NVFP4 layout: 8192 bytes of E2M1 values (64 bytes
per token, low nibble first) followed by 1024 bytes of E4M3 scales for groups
of 16 values. Dequant warps expand each TMA-loaded page straight into an E4M3
UMMA A operand in tensor memory, so shared memory only carries the raw page.
"""

from __future__ import annotations

import enum

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute.typing import Float16, Float32, Int32, Int64
from cutlass.cutlass_dsl import BaseDSL, T, dsl_user_op

import cuda.bindings.driver as cuda

from ..common import utils as common_utils

# QMUL4 is public PTX from CUDA 13.4; older DSL backends and SM107, whose ptxas
# rejects it, take the exact FP16 path.
_HAS_QMUL4 = cutlass.CUDA_VERSION.major > 13 or (
    cutlass.CUDA_VERSION.major == 13 and cutlass.CUDA_VERSION.minor >= 4
)


@dsl_user_op
def _e2m1x8_scaled_to_e4m3x8(
    packed: Int32, scale: Int32, *, loc=None, ip=None
) -> tuple[Int32, Int32]:
    """Multiply eight E2M1 values by the E4M3 scale in byte 0 with RN saturation."""

    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(packed).ir_value(loc=loc, ip=ip), Int32(scale).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        ".reg .b32 sf;\n\t"
        "prmt.b32 sf, $3, 0, 0;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 $0, lo, sf;\n\t"
        "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 $1, hi, sf;\n\t"
        "}\n",
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e4m3_scale_to_f16x2(scales: Int32, byte: int, *, loc=None, ip=None) -> Int32:
    """Broadcast E4M3 byte ``byte`` of ``scales`` to both halves of an f16x2."""

    selector = byte | (byte << 4)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(scales).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b32 sf_bytes;\n\t"
            ".reg .b16 sf_pair;\n\t"
            f"prmt.b32 sf_bytes, $1, 0, {selector};\n\t"
            "mov.b32 {sf_pair, _}, sf_bytes;\n\t"
            "cvt.rn.f16x2.e4m3x2 $0, sf_pair;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _e2m1x8_f16x2_scaled_to_e4m3x8(
    packed: Int32, scale_f16x2: Int32, *, loc=None, ip=None
) -> tuple[Int32, Int32]:
    """``cvt_fp4x8_e2m1_scaled_e4m3x8`` with the scale already converted to f16x2."""

    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(packed).ir_value(loc=loc, ip=ip), Int32(scale_f16x2).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 e0, e1, e2, e3;\n\t"
        ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
        ".reg .b32 h0, h1, h2, h3;\n\t"
        "mov.b32 {byte0, byte1, byte2, byte3}, $2;\n\t"
        "cvt.rn.f16x2.e2m1x2 h0, byte0;\n\t"
        "cvt.rn.f16x2.e2m1x2 h1, byte1;\n\t"
        "cvt.rn.f16x2.e2m1x2 h2, byte2;\n\t"
        "cvt.rn.f16x2.e2m1x2 h3, byte3;\n\t"
        "mul.rn.f16x2 h0, h0, $3;\n\t"
        "mul.rn.f16x2 h1, h1, $3;\n\t"
        "mul.rn.f16x2 h2, h2, $3;\n\t"
        "mul.rn.f16x2 h3, h3, $3;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
        "mov.b32 $0, {e0, e1};\n\t"
        "mov.b32 $1, {e2, e3};\n\t"
        "}\n",
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e2m1x8_f16x2_scaled_to_f16x8(
    packed: Int32, scale_f16x2: Int32, *, loc=None, ip=None
) -> tuple[Int32, Int32, Int32, Int32]:
    """Eight E2M1 values times an f16x2 block scale, kept as four f16x2 words (exact)."""

    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(packed).ir_value(loc=loc, ip=ip), Int32(scale_f16x2).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b8 byte0, byte1, byte2, byte3;\n\t"
        "mov.b32 {byte0, byte1, byte2, byte3}, $4;\n\t"
        "cvt.rn.f16x2.e2m1x2 $0, byte0;\n\t"
        "cvt.rn.f16x2.e2m1x2 $1, byte1;\n\t"
        "cvt.rn.f16x2.e2m1x2 $2, byte2;\n\t"
        "cvt.rn.f16x2.e2m1x2 $3, byte3;\n\t"
        "mul.rn.f16x2 $0, $0, $5;\n\t"
        "mul.rn.f16x2 $1, $1, $5;\n\t"
        "mul.rn.f16x2 $2, $2, $5;\n\t"
        "mul.rn.f16x2 $3, $3, $5;\n\t"
        "}\n",
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _e4m3x4_to_f16x4(word: Int32, *, loc=None, ip=None) -> tuple[Int32, Int32]:
    """Widen four packed E4M3 values to two f16x2 words (exact)."""

    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, hi;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
    )


class _NamedBarrier(enum.IntEnum):
    TmemPtr = enum.auto()
    Score = enum.auto()
    Final = enum.auto()


class Q8KV4DecodeIndexerSm100:
    """Compute NVFP4-K page scores for two or four index heads with persistent workers.

    The heads share the single K head, so the eight MTP tokens of all heads
    form one MMA N tile of ``8 * num_heads`` columns (``token * num_heads +
    head``). Each K page is loaded and dequantized once per request.
    """

    supported_num_heads = (2, 4)
    query_length = 8
    page_size = 128
    head_dim = 128
    m_tile = page_size
    k_tile = head_dim
    packed_bytes_per_token = head_dim // 2
    packed_page_bytes = page_size * packed_bytes_per_token
    # The TMA loads the packed page as 64 rows of 128 bytes (two tokens per
    # row) so every request is a full L2 line; 64-byte rows cost an extra
    # sector per request. Each dequant thread reads its token's 64 bytes as
    # 16-byte chunks, which the 128-byte swizzle Swizzle<3, 4, 3> permutes by
    # row so a quarter warp hits distinct banks.
    packed_chunk_bytes = 16
    packed_tma_row_bytes = 128
    packed_tma_rows = packed_page_bytes // packed_tma_row_bytes
    packed_swizzle = (3, 4, 3)
    scale_group = 16
    scales_per_token = head_dim // scale_group
    scale_page_bytes = page_size * scales_per_token
    # A TMA box dimension holds at most 256 elements, so the contiguous scale
    # bytes of a page are loaded as rows of 128.
    scale_row_bytes = 128
    scale_rows = scale_page_bytes // scale_row_bytes
    raw_page_bytes = packed_page_bytes + scale_page_bytes
    raw_stages = 8
    # Two K and two accumulator stages measured faster than four of each.
    k_stages = 2
    acc_stages = 2
    # TMEM columns hold 32 bits per lane: an accumulator stage uses one column
    # per query and a K stage packs four E4M3 values per column.
    e4m3_per_tmem_col = Float32.width // cutlass.Float8E4M3FN.width
    k_stage_tmem_cols = head_dim // e4m3_per_tmem_col
    threads_per_warp = 32
    score_warp_begin = 3
    score_warps = 4
    score_threads = score_warps * threads_per_warp
    # Warp w accesses TMEM lanes 32 * (w % 4), the lanes ``tidx % 128`` names,
    # so each of the four dequant warps owns the token rows of its lanes.
    dequant_warp_begin = score_warp_begin + score_warps
    dequant_warps = 4
    threads_per_cta = (dequant_warp_begin + dequant_warps) * threads_per_warp
    tmem_copy_threads = 128
    target_ctas_per_sm = 2
    sm107_mma_k = 64
    q_stages = 1
    q_loader_warp = 2
    merge_q_loader = False
    empty_cta_pages = -1
    # SM107 (set in __init__): the dequantized K stays FP16 in TMEM and the QK MMA
    # is kind::f16. SM107's dequant has to go through FP16 anyway (ptxas rejects
    # QMUL4 there); rounding the products back to E4M3 for an FP8 MMA cost a third
    # of the dequant instructions and up to ~8% per-score error, while the FP8 MMA
    # rate was never the limiter. q stays E4M3 at the interface: the MMA warp
    # widens it into shared memory right before its first MMA of a segment.
    use_fp16_mma = False
    split_tmem = False
    tmem_cols_k = 0
    tmem_cols_acc = 0
    k_dtype = cutlass.Float8E4M3FN
    k_per_tmem_col = e4m3_per_tmem_col
    dequant_store_halves = 1

    def __init__(self, *, sm_count: int, num_heads: int, out_query_len: int = 0) -> None:
        # MiniMax-M3.1 (MSA #7, PREREG_M31_MSA_NVFP4_INDEXK_HPR_v1): out_query_len > 0 writes the scores straight into the
        # select's [heads, pages, rows] layout (rows = batch * out_query_len, the request's last out_query_len of the eight
        # query slots), skipping the padded slots; 0 keeps the upstream [batch, 8 * heads, pages] layout. Same scores.
        if not 0 <= out_query_len <= self.query_length:
            raise ValueError(f"out_query_len must be in [0, {self.query_length}]")
        self.out_query_len = out_query_len
        if sm_count <= 0:
            raise ValueError("sm_count must be positive")
        if num_heads not in self.supported_num_heads:
            raise ValueError(f"num_heads must be one of {self.supported_num_heads}")
        self.num_heads = num_heads
        self.num_queries = self.query_length * num_heads
        self.n_tile = self.num_queries
        arch = BaseDSL._get_dsl().get_arch_enum()
        rubin_arch = getattr(arch.__class__, "sm_107", None)
        is_rubin = rubin_arch is not None and arch.is_family_of(rubin_arch)
        if is_rubin:
            # The page-loader warp also issues the Q TMA, so a CTA has ten
            # warps; two Q stages let it run a request segment ahead.
            self.merge_q_loader = True
            self.q_loader_warp = -1
            self.q_stages = 2
            self.score_warp_begin = 2
            self.dequant_warp_begin = self.score_warp_begin + self.score_warps
            self.threads_per_cta = (self.dequant_warp_begin + self.dequant_warps) * self.threads_per_warp
            self.target_ctas_per_sm = 3
            self.empty_cta_pages = 0
            self.raw_stages = 7
            self.use_fp16_mma = True
        if self.use_fp16_mma:
            if not is_rubin:
                raise NotImplementedError("the FP16 K / kind::f16 path is implemented for SM107")
            self.k_dtype = Float16
            self.k_per_tmem_col = Float32.width // Float16.width
            self.k_stage_tmem_cols = self.head_dim // self.k_per_tmem_col
            # Keep the dequant register tile at 32 words: store each TMEM K stage in two halves.
            self.dequant_store_halves = 2
        self.acc_tmem_cols = self.acc_stages * self.num_queries
        # The allocation is a power of two of at least 32 columns: 128 for both
        # two heads (2 * 16 + 2 * 32 columns) and four heads (2 * 32 + 2 * 32).
        used_tmem_cols = self.acc_tmem_cols + self.k_stages * self.k_stage_tmem_cols
        self.tmem_cols = max(32, 1 << (used_tmem_cols - 1).bit_length())
        if self.use_fp16_mma:
            # FP16 K stages are 128 columns. SM107 has 576 TMEM columns (SM100/SM103:
            # 512): two allocations (K stages, accumulators: 160/192 per CTA) fit three
            # CTAs where one power-of-two allocation (256) would allow only two.
            self.split_tmem = True
            self.tmem_cols_k = self.k_stages * self.k_stage_tmem_cols
            self.tmem_cols_acc = max(32, 1 << (self.acc_tmem_cols - 1).bit_length())
            per_cta = self.tmem_cols_k + self.tmem_cols_acc
            if per_cta * self.target_ctas_per_sm > 576:
                self.target_ctas_per_sm = 576 // per_cta
            # FP16 doubles the Q stages (8 KB at four heads); one raw page stage less
            # (9 KB) keeps three CTAs within the SM's shared memory. (One Q stage with
            # seven raw stages measured 1.8% slower.)
            if self.target_ctas_per_sm == 3 and self.num_heads == 4:
                self.raw_stages = 6
        self.grid_ctas = sm_count * self.target_ctas_per_sm
        self.use_qmul4 = _HAS_QMUL4 and not is_rubin
        # SM107's FP16 dequant path needs more registers; without a register
        # bound only one CTA fits per SM there and the grid runs in two waves.
        self.min_blocks_per_mp = self.target_ctas_per_sm if is_rubin else 1
        # SM107 has a K=64 FP8 UMMA, two instructions per page instead of four.
        self.use_sm107_fp8_mma = is_rubin and not self.use_fp16_mma
        # One scale byte covers two packed words, so SM107 converts each of a
        # row's eight scales to f16x2 once per page instead of once per word.
        self.hoist_scale_cvt = is_rubin and not self.use_qmul4

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mPageTable: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut: cute.Tensor,
        mSchedulerStorage: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if cutlass.const_expr(mQ.element_type is not cutlass.Float8E4M3FN):
            raise TypeError("q must be Float8E4M3FN")
        if cutlass.const_expr(mKCache.element_type is not cutlass.Uint8):
            raise TypeError("k_cache must be Uint8 NVFP4 pages")
        if cutlass.const_expr(
            mPageTable.element_type is not Int32 or mSeqLens.element_type is not Int32
        ):
            raise TypeError("page_table and seq_lens must be Int32")
        if cutlass.const_expr(mOut.element_type is not Float32):
            raise TypeError("out must be Float32")

        batch_size = mPageTable.shape[0]
        # q holds 1..8 queries per request; they fill the last slots of its
        # eight, so the TMA box starts that many rows before the request.
        query_rows = mQ.shape[0] // batch_size * self.num_heads
        physical_pages = mKCache.shape[0]
        # Each page is contiguous; vLLM may pad the stride between pages.
        page_stride = mKCache.stride[0]
        mKPacked = cute.make_tensor(
            mKCache.iterator,
            cute.make_layout(
                (self.packed_tma_row_bytes, self.packed_tma_rows, physical_pages),
                stride=(1, self.packed_tma_row_bytes, page_stride),
            ),
        )
        mKScale = cute.make_tensor(
            mKCache.iterator + self.packed_page_bytes,
            cute.make_layout(
                (self.scale_row_bytes, self.scale_rows, physical_pages),
                stride=(1, self.scale_row_bytes, page_stride),
            ),
        )
        mQ_krb = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (self.head_dim, query_rows * batch_size),
                stride=(1, self.head_dim),
            ),
        )
        mPageTable_lb = cute.make_tensor(
            mPageTable.iterator,
            cute.select(mPageTable.layout, mode=[1, 0]),
        )
        if cutlass.const_expr(self.out_query_len == 0):
            mOut_pqb = cute.make_tensor(
                mOut.iterator,
                cute.select(mOut.layout, mode=[2, 1, 0]),
            )
        else:
            # mOut = [heads, pages, rows] over a buffer whose select view starts (8 - out_query_len) elements later:
            # query slot s of request b lands on row b * out_query_len + s - (8 - out_query_len) of that view.
            mOut_pqb = cute.make_tensor(
                mOut.iterator,
                cute.make_layout(
                    (mOut.shape[1], (self.num_heads, self.query_length), batch_size),
                    stride=(mOut.stride[1], (mOut.stride[0], 1), self.out_query_len),
                ),
            )
        mScheduler = cute.make_tensor(
            cute.recast_ptr(mSchedulerStorage.iterator, dtype=Int32),
            cute.make_layout(batch_size + 1),
        )

        qk_tiler = (self.m_tile, self.n_tile, self.k_tile)
        cta_group = tcgen05.CtaGroup.ONE
        # Both operands are K-major: dequantized K rows in TMEM and Q rows.
        if cutlass.const_expr(self.use_fp16_mma):
            tiled_mma = cute.make_tiled_mma(
                tcgen05.MmaF16BF16Op(
                    Float16,
                    Float32,
                    (self.m_tile, self.n_tile, 16),
                    cta_group,
                    tcgen05.OperandSource.TMEM,
                    OperandMajorMode.K,
                    OperandMajorMode.K,
                )
            )
        elif cutlass.const_expr(self.use_sm107_fp8_mma):
            tiled_mma = cute.make_tiled_mma(
                tcgen05.SM107MmaFP8Op(
                    cutlass.Float8E4M3FN,
                    mQ.element_type,
                    Float32,
                    (self.m_tile, self.n_tile, self.sm107_mma_k),
                    cta_group,
                    tcgen05.OperandSource.TMEM,
                    OperandMajorMode.K,
                    OperandMajorMode.K,
                )
            )
        else:
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                cutlass.Float8E4M3FN,
                mQ.element_type,
                OperandMajorMode.K,
                OperandMajorMode.K,
                Float32,
                cta_group,
                qk_tiler[:2],
                tcgen05.OperandSource.TMEM,
            )
        tK_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            qk_tiler,
            self.k_dtype,
            1,
        )
        sQ_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            qk_tiler,
            self.k_dtype,
            self.q_stages,
        )
        # On the FP16-K path the loader warp widens q into sQ itself; this TMA
        # layout and atom are then unused but still built (E4M3 gmem, 128B rows).
        sQ_tma_layout = cute.make_composed_layout(
            sQ_layout.inner,
            0,
            cute.make_layout(
                (self.head_dim, self.num_queries),
                stride=(1, self.head_dim),
            ),
        )
        sPacked_layout = cute.make_composed_layout(
            cute.make_swizzle(*self.packed_swizzle),
            0,
            cute.make_layout(
                (self.packed_tma_row_bytes, self.packed_tma_rows, self.raw_stages),
                stride=(1, self.packed_tma_row_bytes, self.packed_page_bytes),
            ),
        )
        sScale_layout = cute.make_layout(
            (self.scale_row_bytes, self.scale_rows, self.raw_stages),
            stride=(1, self.scale_row_bytes, self.scale_page_bytes),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_packed, tma_tensor_packed = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mKPacked,
            cute.slice_(sPacked_layout, (None, None, 0)),
            (self.packed_tma_row_bytes, self.packed_tma_rows),
        )
        tma_atom_scale, tma_tensor_scale = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mKScale,
            cute.select(sScale_layout, mode=[0, 1]),
            (self.scale_row_bytes, self.scale_rows),
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mQ_krb,
            sQ_tma_layout,
            (self.head_dim, self.num_queries),
        )

        sPartialMax_layout = cute.make_layout(
            (2, self.score_warps, self.num_queries),
            stride=(self.score_warps * self.num_queries, self.num_queries, 1),
        )

        @cute.struct
        class SharedStorage:
            raw_mbar_ptr: cute.struct.MemRange[Int64, self.raw_stages * 2]
            k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stages * 2]
            q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stages * 2]
            acc_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stages * 2]
            tmem_holding_buf: Int32
            tmem_holding_buf2: Int32
            sPacked: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(sPacked_layout)], 1024
            ]
            sScale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(sScale_layout)], 128
            ]
            # 1024-byte aligned so the layout's 128B swizzle (applied to offsets from the
            # buffer base by the FP16-path q widen) coincides with the hardware's, which
            # permutes on absolute shared-memory address bits.
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sQ_layout)], 1024
            ]
            sPartialMax: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(sPartialMax_layout)], 16
            ]

        self.shared_storage = SharedStorage
        # The DSL sizes the shared-memory carveout from an explicit byte count
        # whenever more than one block per SM is requested.
        smem_bytes = SharedStorage.size_in_bytes() if self.min_blocks_per_mp > 1 else None
        self.kernel(
            tiled_mma,
            tma_atom_packed,
            tma_tensor_packed,
            tma_atom_scale,
            tma_tensor_scale,
            tma_atom_Q,
            tma_tensor_Q,
            mQ_krb,
            mPageTable_lb,
            mSeqLens,
            mOut_pqb,
            mScheduler,
            batch_size,
            query_rows,
            sPacked_layout,
            sScale_layout,
            tK_layout,
            sQ_layout,
            sQ_tma_layout,
            sPartialMax_layout,
        ).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_packed: cute.CopyAtom,
        mKPacked: cute.Tensor,
        tma_atom_scale: cute.CopyAtom,
        mKScale: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mQ_krb: cute.Tensor,
        mQ_raw: cute.Tensor,
        mPageTable_lb: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut_pqb: cute.Tensor,
        mScheduler: cute.Tensor,
        batch_size: Int32,
        query_rows: Int32,
        sPacked_layout: cute.ComposedLayout,
        sScale_layout: cute.Layout,
        tK_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
        sQ_tma_layout: cute.ComposedLayout,
        sPartialMax_layout: cute.Layout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        worker_idx, _, _ = cute.arch.block_idx()

        if warp_idx == Int32(1):
            cpasync.prefetch_descriptor(tma_atom_packed)
            cpasync.prefetch_descriptor(tma_atom_scale)
            if cutlass.const_expr(self.merge_q_loader):
                cpasync.prefetch_descriptor(tma_atom_Q)
        elif warp_idx == Int32(self.q_loader_warp):
            cpasync.prefetch_descriptor(tma_atom_Q)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sPacked = storage.sPacked.get_tensor(sPacked_layout.outer, swizzle=sPacked_layout.inner)
        sScale = storage.sScale.get_tensor(sScale_layout)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ_tma = cute.make_tensor(sQ.iterator, sQ_tma_layout.outer)
        if cutlass.const_expr(self.use_fp16_mma):
            # The MMA warp's store view of sQ: ((8-element vector, 16-byte chunk, 64-element
            # atom half), query row, stage), the same bytes and the same swizzle as sQ_layout;
            # this view only exposes the 16-byte vector the widen stores.
            sQ_vec = storage.sQ.get_tensor(
                cute.make_layout(
                    ((8, 8, self.head_dim // 64), self.num_queries, self.q_stages),
                    stride=((1, 8, 64 * self.num_queries), 64, self.num_queries * self.head_dim),
                ),
                swizzle=sQ_layout.inner,
            )
        sPartialMax = storage.sPartialMax.get_tensor(sPartialMax_layout)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.TmemPtr),
            num_threads=self.threads_per_cta,
        )
        score_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.Score),
            num_threads=self.score_threads,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        if cutlass.const_expr(self.split_tmem):
            # K stages first (the larger, 128-column-aligned block), then accumulators.
            tmem.allocate(self.tmem_cols_k)
            tmem_acc = utils.TmemAllocator(
                storage.tmem_holding_buf2,
                barrier_for_retrieve=tmem_alloc_barrier,
            )
            tmem_acc.allocate(self.tmem_cols_acc)
        else:
            tmem.allocate(self.tmem_cols)
        # Release the permit right away so the other CTA on the SM can allocate.
        tmem.relinquish_alloc_permit()

        # Only lane 0 of each consumer warp signals a TMA-async empty barrier.
        raw_producer, raw_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.raw_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.dequant_warps),
            tx_count=self.raw_page_bytes,
            barrier_storage=storage.raw_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        ).make_participants()
        # One elected lane per dequant warp commits a TMEM K stage.
        k_producer, k_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.dequant_warps),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        if cutlass.const_expr(self.use_fp16_mma):
            # The MMA warp both fills sQ and consumes it through its own MMAs, so
            # the stage ring needs only the "empty" side of a pipeline: one
            # mbarrier per stage, released by the UMMA commit of the MMAs that
            # read it. The warp tracks it with a producer PipelineState.
            q_stage_empty = pipeline.MbarrierArray(
                storage.q_mbar_ptr.data_ptr(),
                self.q_stages,
                (pipeline.PipelineOp.TCGen05Mma, pipeline.CooperativeGroup(pipeline.Agent.Thread)),
            )
        else:
            q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.q_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                tx_count=self.num_queries * self.head_dim,
                barrier_storage=storage.q_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.tmem_copy_threads,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        pipeline.pipeline_init_arrive(is_relaxed=True)

        tPsP, tPgP = cpasync.tma_partition(
            tma_atom_packed,
            0,
            cute.make_layout(1),
            cute.group_modes(sPacked, 0, 2),
            cute.group_modes(
                cute.flat_divide(mKPacked, (self.packed_tma_row_bytes, self.packed_tma_rows)), 0, 2
            ),
        )
        tSsS, tSgS = cpasync.tma_partition(
            tma_atom_scale,
            0,
            cute.make_layout(1),
            cute.group_modes(sScale, 0, 2),
            cute.group_modes(
                cute.flat_divide(mKScale, (self.scale_row_bytes, self.scale_rows)), 0, 2
            ),
        )

        tCrQ = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C((self.m_tile, self.n_tile))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        pipeline.pipeline_init_wait()
        # PDL: the predecessor writes q and the current pages of the index cache.
        cute.arch.griddepcontrol_wait()
        # The plan writes the scheduler prefix sums just before this launch, so
        # they are read only after the PDL wait. Lane 0 of every warp reads
        # them; a read before the wait could give warps different page counts.
        total_pages = Int32(0)
        if lane_idx == Int32(0):
            total_pages = mScheduler[batch_size]
        total_pages = cute.arch.shuffle_sync(total_pages, 0)
        pages_per_worker = (total_pages + Int32(self.grid_ctas - 1)) // Int32(self.grid_ctas)
        global_page_begin = worker_idx * pages_per_worker
        num_pages = Int32(0)
        if global_page_begin < total_pages:
            global_page_end = global_page_begin + pages_per_worker
            if global_page_end > total_pages:  # noqa: PLR1730
                global_page_end = total_pages
            num_pages = global_page_end - global_page_begin

        batch_idx = Int32(0)
        if lane_idx == Int32(0):
            batch_hi = batch_size
            while batch_idx < batch_hi:
                batch_mid = (batch_idx + batch_hi) // Int32(2)
                if mScheduler[batch_mid + 1] <= global_page_begin:
                    batch_idx = batch_mid + Int32(1)
                else:
                    batch_hi = batch_mid
        batch_idx = cute.arch.shuffle_sync(batch_idx, 0)
        logical_page_begin = global_page_begin - mScheduler[batch_idx]
        thr_mma = tiled_mma.get_slice(0)
        tmem_ptr = tmem.retrieve_ptr(Float32)
        if cutlass.const_expr(self.split_tmem):
            acc_ptr = tmem_acc.retrieve_ptr(Float32)
            k_col_offset = 0
        else:
            acc_ptr = tmem_ptr
            k_col_offset = self.acc_tmem_cols
        tCtAcc_staged = cute.make_tensor(acc_ptr, tCtAcc_fake.layout)
        # TMEM K stages follow the accumulators. The TMEM A fragment starts at
        # column 0 rather than at the input tensor, so rebase it onto this
        # CTA's allocation; fragment offsets count E4M3 values.
        tCrK_base = tiled_mma.make_fragment_A(cute.make_tensor(tmem_ptr, tK_layout.outer))[
            None, None, None, 0
        ]
        k_tmem_col = tmem_ptr.toint() + Int32(k_col_offset)
        tCrK = cute.make_tensor(
            tCrK_base.iterator + k_tmem_col * Int32(self.k_per_tmem_col),
            cute.append(
                tCrK_base.layout,
                cute.make_layout(
                    (self.k_stages,),
                    stride=(self.k_stage_tmem_cols * self.k_per_tmem_col,),
                ),
            ),
        )
        # Dequant warps write the same K stages through an FP32 view: lane is
        # the token row, each column packs four E4M3 values. A K stage is
        # k_stage_tmem_cols wide for any head count, so the view takes the C
        # layout of a (page_size, k_stage_tmem_cols) MMA, not the accumulator's.
        store_cols = self.k_stage_tmem_cols // self.dequant_store_halves
        k_view_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (self.m_tile, store_cols),
            tcgen05.OperandSource.TMEM,
        )
        k_view_layout = k_view_mma.make_fragment_C(
            k_view_mma.partition_shape_C((self.m_tile, store_cols))
        ).layout
        tKtK = cute.make_tensor(
            tmem_ptr + k_col_offset,
            cute.append(
                cute.append(
                    cute.composition(
                        k_view_layout,
                        cute.make_layout((self.page_size, store_cols)),
                    ),
                    cute.make_layout((self.dequant_store_halves,), stride=(store_cols,)),
                ),
                cute.make_layout((self.k_stages,), stride=(self.k_stage_tmem_cols,)),
            ),
        )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(getattr(tcgen05.Repetition, f"x{self.num_queries}")),
            Float32,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_load_atom,
            tCtAcc_staged[(None, None, None, 0)],
        )
        copy_tidx = tidx % Int32(self.tmem_copy_threads)
        thr_tmem_copy = tmem_tiled_copy.get_slice(copy_tidx)
        tTR_tAcc_staged = thr_tmem_copy.partition_S(tCtAcc_staged)
        cScores = cute.make_identity_tensor((self.m_tile, self.n_tile))
        tCcScores = thr_mma.partition_C(cScores)
        tTR_cScores = thr_tmem_copy.partition_D(tCcScores)

        # Small steps leave most CTAs without pages: on SM107 those go straight
        # to the final barrier and TMEM free (num_pages >= 0 elsewhere).
        if num_pages > Int32(self.empty_cta_pages):
            if warp_idx == Int32(1):
                current_batch = batch_idx
                logical_page = logical_page_begin
                global_page = global_page_begin
                pages_remaining = num_pages
                while pages_remaining > Int32(0):
                    segment_pages = mScheduler[current_batch + 1] - global_page
                    if segment_pages > pages_remaining:  # noqa: PLR1730
                        segment_pages = pages_remaining
                    # On the FP16 path the MMA warp widens q itself; the loader only streams pages.
                    if cutlass.const_expr(self.merge_q_loader and not self.use_fp16_mma):
                        q_empty = q_producer.acquire_and_advance()
                        q_row = (current_batch + Int32(1)) * query_rows - Int32(self.num_queries)
                        sQ_stage = cute.make_tensor(
                            sQ.iterator + q_empty.index * Int32(self.head_dim * self.num_queries),
                            sQ_tma_layout.outer,
                        )
                        tQsQ, tQgQ = cpasync.tma_partition(
                            tma_atom_Q,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sQ_stage, 0, 2),
                            cute.group_modes(
                                cute.flat_divide(
                                    cute.domain_offset((0, q_row), mQ_krb),
                                    (self.head_dim, self.num_queries),
                                ),
                                0,
                                2,
                            ),
                        )
                        cute.copy(tma_atom_Q, tQgQ[(None, 0, 0)], tQsQ, tma_bar_ptr=q_empty.barrier)
                    # Each lane fetches one of the next 32 page-table entries, so
                    # the page lookup latency is paid once per 32 raw pages.
                    for chunk_begin in cutlass.range(0, segment_pages, self.threads_per_warp, unroll=1):
                        chunk_pages = segment_pages - chunk_begin
                        if chunk_pages > Int32(self.threads_per_warp):  # noqa: PLR1730
                            chunk_pages = Int32(self.threads_per_warp)
                        lane_page = Int32(0)
                        if lane_idx < chunk_pages:
                            lane_page = mPageTable_lb[logical_page + lane_idx, current_batch]
                        for chunk_page in cutlass.range(chunk_pages, unroll=1):
                            physical_page = cute.arch.shuffle_sync(lane_page, chunk_page)
                            raw_empty = raw_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_packed,
                                tPgP[(None, 0, 0, physical_page)],
                                tPsP[(None, raw_empty.index)],
                                tma_bar_ptr=raw_empty.barrier,
                            )
                            cute.copy(
                                tma_atom_scale,
                                tSgS[(None, 0, 0, physical_page)],
                                tSsS[(None, raw_empty.index)],
                                tma_bar_ptr=raw_empty.barrier,
                            )
                        global_page += chunk_pages
                        logical_page += chunk_pages
                    pages_remaining -= segment_pages
                    while (
                        current_batch < batch_size - Int32(1)
                        and mScheduler[current_batch + 1] <= global_page
                    ):
                        current_batch += Int32(1)
                        logical_page = global_page - mScheduler[current_batch]
                raw_producer.tail()
                if cutlass.const_expr(self.merge_q_loader and not self.use_fp16_mma):
                    q_producer.tail()
            elif warp_idx == Int32(self.q_loader_warp):
                # SM100/SM103 layout only; this branch is still traced on SM107.
                if cutlass.const_expr(not self.use_fp16_mma):
                    current_batch = batch_idx
                    global_page = global_page_begin
                    pages_remaining = num_pages
                    while pages_remaining > Int32(0):
                        segment_pages = mScheduler[current_batch + 1] - global_page
                        if segment_pages > pages_remaining:  # noqa: PLR1730
                            segment_pages = pages_remaining
                        q_empty = q_producer.acquire_and_advance()
                        # Rows before the request belong to the previous one, or are
                        # out of bounds and zero-filled for the first: they only fill
                        # the unused leading slots.
                        q_row = (current_batch + Int32(1)) * query_rows - Int32(self.num_queries)
                        tQsQ, tQgQ = cpasync.tma_partition(
                            tma_atom_Q,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sQ_tma, 0, 2),
                            cute.group_modes(
                                cute.flat_divide(
                                    cute.domain_offset((0, q_row), mQ_krb),
                                    (self.head_dim, self.num_queries),
                                ),
                                0,
                                2,
                            ),
                        )
                        cute.copy(
                            tma_atom_Q,
                            tQgQ[(None, 0, 0)],
                            tQsQ,
                            tma_bar_ptr=q_empty.barrier,
                        )
                        global_page += segment_pages
                        pages_remaining -= segment_pages
                        while (
                            current_batch < batch_size - Int32(1)
                            and mScheduler[current_batch + 1] <= global_page
                        ):
                            current_batch += Int32(1)
                    q_producer.tail()
            elif warp_idx == Int32(0):
                num_k_blocks = cute.size(tCrK, mode=[2])
                if cutlass.const_expr(self.use_fp16_mma):
                    # Producer state: starts with phase 1, so the first use of each
                    # sQ stage passes its empty wait without a prior commit.
                    q_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.q_stages
                    )
                current_batch = batch_idx
                global_page = global_page_begin
                pages_remaining = num_pages
                while pages_remaining > Int32(0):
                    segment_pages = mScheduler[current_batch + 1] - global_page
                    if segment_pages > pages_remaining:  # noqa: PLR1730
                        segment_pages = pages_remaining
                    if cutlass.const_expr(self.use_fp16_mma):
                        # Wait until the MMAs that last read this sQ stage have
                        # committed, then widen the request's q into it and fence
                        # the generic stores for the async proxy.
                        q_stage_empty.wait(q_state.index, q_state.phase)
                        q_index = q_state.index
                        q_row = (current_batch + Int32(1)) * query_rows - Int32(self.num_queries)
                        self._widen_q_stage(mQ_raw, sQ_vec, q_index, q_row, lane_idx)
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_warp()
                    else:
                        q_full = q_consumer.wait_and_advance()
                        q_index = q_full.index
                    for _ in cutlass.range(segment_pages, unroll=1):
                        acc_empty = acc_producer.acquire_and_advance()
                        tCtAcc = tCtAcc_staged[(None, None, None, acc_empty.index)]
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        k_full = k_consumer.wait_and_advance()
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrK[(None, None, k_block_idx, k_full.index)],
                                tCrQ[(None, None, k_block_idx, q_index)],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_full.release()
                        acc_empty.commit()
                    if cutlass.const_expr(self.use_fp16_mma):
                        # UMMA commit releases the stage once this segment's MMAs are done.
                        q_stage_empty.arrive(q_state.index, None, tcgen05.CtaGroup.ONE)
                        q_state.advance()
                    else:
                        q_full.release()
                    global_page += segment_pages
                    pages_remaining -= segment_pages
                    while (
                        current_batch < batch_size - Int32(1)
                        and mScheduler[current_batch + 1] <= global_page
                    ):
                        current_batch += Int32(1)
                acc_producer.tail()

            rScores = cute.make_rmem_tensor(tTR_cScores.shape, Float32)
            if warp_idx >= Int32(self.dequant_warp_begin):
                self._dequant_pages(
                    num_pages,
                    raw_consumer,
                    k_producer,
                    sPacked,
                    sScale,
                    tKtK,
                    copy_tidx,
                )
            elif warp_idx >= Int32(self.score_warp_begin):
                score_warp_idx = warp_idx - Int32(self.score_warp_begin)
                rScoresFlat = cute.make_tensor(
                    rScores.iterator,
                    cute.make_layout(self.num_queries),
                )
                current_batch = batch_idx
                logical_page = logical_page_begin
                global_page = global_page_begin
                last_global_page = global_page_begin + num_pages - Int32(1)
                pages_remaining = num_pages
                local_block = Int32(0)
                if (
                    num_pages > Int32(0)
                    and score_warp_idx == Int32(0)
                    and lane_idx < Int32(self.num_queries)
                ):
                    seq_len = mSeqLens[current_batch]
                    query_position = (
                        seq_len - Int32(self.query_length) + lane_idx // Int32(self.num_heads)
                    )
                    local_block = query_position // Int32(self.page_size)
                while pages_remaining > Int32(0):
                    segment_pages = mScheduler[current_batch + 1] - global_page
                    if segment_pages > pages_remaining:  # noqa: PLR1730
                        segment_pages = pages_remaining
                    for _ in cutlass.range(segment_pages, unroll=1):
                        partial_stage = global_page % Int32(2)
                        acc_full = acc_consumer.wait_and_advance()
                        cute.copy(
                            tmem_tiled_copy,
                            tTR_tAcc_staged[(None, None, None, None, acc_full.index)],
                            rScores,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        for query_idx in cutlass.range_constexpr(self.num_queries):
                            partial_max = cute.arch.warp_redux_sync(rScoresFlat[query_idx], "fmax")
                            if lane_idx == Int32(0):
                                sPartialMax[partial_stage, score_warp_idx, query_idx] = partial_max
                        acc_full.release()
                        score_barrier.arrive_and_wait()

                        if score_warp_idx == Int32(0) and lane_idx < Int32(self.num_queries):
                            query_idx = lane_idx
                            row_max = -Float32.inf
                            for partial_idx in cutlass.range_constexpr(self.score_warps):
                                row_max = cute.arch.fmax(
                                    row_max, sPartialMax[partial_stage, partial_idx, query_idx]
                                )
                            if global_page == last_global_page:
                                # The CTA's last scores: the top-k that reads them
                                # waits for this grid, so it may start launching.
                                cute.arch.griddepcontrol_launch_dependents()
                            if logical_page < local_block:
                                if cutlass.const_expr(self.out_query_len == 0):
                                    mOut_pqb[logical_page, query_idx, current_batch] = row_max
                                elif query_idx >= Int32(
                                    (self.query_length - self.out_query_len) * self.num_heads
                                ):
                                    mOut_pqb[logical_page, query_idx, current_batch] = row_max
                        global_page += Int32(1)
                        logical_page += Int32(1)
                    pages_remaining -= segment_pages
                    while (
                        current_batch < batch_size - Int32(1)
                        and mScheduler[current_batch + 1] <= global_page
                    ):
                        current_batch += Int32(1)
                        logical_page = global_page - mScheduler[current_batch]
                    if score_warp_idx == Int32(0) and lane_idx < Int32(self.num_queries):
                        seq_len = mSeqLens[current_batch]
                        query_position = (
                            seq_len - Int32(self.query_length) + lane_idx // Int32(self.num_heads)
                        )
                        local_block = query_position // Int32(self.page_size)

        pipeline.sync(barrier_id=int(_NamedBarrier.Final))
        tmem.free(tmem_ptr)
        if cutlass.const_expr(self.split_tmem):
            tmem_acc.free(acc_ptr)

    @cute.jit
    def _widen_q_stage(
        self,
        mQ_krb: cute.Tensor,
        sQ_vec: cute.Tensor,
        stage: Int32,
        q_row: Int32,
        lane_idx: Int32,
    ) -> None:
        """MMA warp: E4M3 q rows [q_row, q_row + num_queries) -> FP16 sQ stage.

        The warp is tiled as 8 lanes x 16 values along the head dimension and 4 rows
        per step, so one tiled load reads the (head_dim, num_queries) q tile with
        16-byte loads and one tiled store writes it to the stage with 16-byte stores.
        Rows before 0 are the first request's unused leading slots: their loads are
        predicated off and the fragment stays zero, as the TMA path's out-of-bounds
        fill does. ``sQ_vec`` is the swizzled ((vector, chunk, atom half), row, stage)
        view of sQ, so partitioning it places each 8-value store where TMA would have
        put it and where the UMMA descriptor (SWIZZLE_128B, K-major) reads it.
        """

        lanes_per_row = self.head_dim // 16  # 16 E4M3 per 16-byte global chunk
        thr_layout = cute.make_layout(
            (lanes_per_row, 32 // lanes_per_row), stride=(1, lanes_per_row)
        )
        val_layout = cute.make_layout((16, 1))
        tiled_load = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mQ_krb.element_type, num_bits_per_copy=128
            ),
            thr_layout,
            val_layout,
        )
        tiled_store = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float16, num_bits_per_copy=128),
            thr_layout,
            val_layout,
        )
        thr_load = tiled_load.get_slice(lane_idx)
        thr_store = tiled_store.get_slice(lane_idx)

        mQ_rows = cute.local_tile(
            cute.domain_offset((0, q_row), mQ_krb), (self.head_dim, self.num_queries), (0, 0)
        )
        tQgQ = thr_load.partition_S(mQ_rows)  # ((16, 1), 1, rows / 4) E4M3
        tQrQ = cute.make_fragment_like(tQgQ)
        tQsQ = thr_store.partition_D(sQ_vec[(None, None, stage)])  # ((8, 2), 1, rows / 4) FP16
        tQrQ16 = cute.make_fragment_like(tQsQ)

        # Predicate on the row coordinate: rows before q_row + ... < 0 are not loaded.
        cQ = cute.make_identity_tensor(mQ_rows.shape)
        tQcQ = thr_load.partition_S(cQ)
        tQpQ = cute.make_rmem_tensor(
            (cute.size(tQcQ, mode=[0, 1]), cute.size(tQcQ, mode=[1]), cute.size(tQcQ, mode=[2])),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(cute.size(tQpQ, mode=[0])):
            for m in cutlass.range_constexpr(cute.size(tQpQ, mode=[2])):
                tQpQ[rest_v, 0, m] = q_row + tQcQ[(0, rest_v), 0, m][1] >= Int32(0)
        tQrQ.fill(0)
        cute.copy(tiled_load, tQgQ, tQrQ, pred=tQpQ)

        # Widen in place: each lane's 16 E4M3 (4 words) of a row become 16 FP16 (8 words).
        tQrQ_w = cute.recast_tensor(tQrQ, Int32)  # ((4, 1), 1, rows / 4)
        tQrQ16_w = cute.recast_tensor(tQrQ16, Int32)  # ((4, 2), 1, rows / 4)
        for m in cutlass.range_constexpr(cute.size(tQrQ_w, mode=[2])):
            for w in cutlass.range_constexpr(4):
                lo, hi = _e4m3x4_to_f16x4(tQrQ_w[((w, 0), 0, m)])
                tQrQ16_w[(((2 * w) % 4, (2 * w) // 4), 0, m)] = lo
                tQrQ16_w[(((2 * w + 1) % 4, (2 * w + 1) // 4), 0, m)] = hi
        cute.copy(tiled_store, tQrQ16, tQsQ)

    @cute.jit
    def _dequant_pages(
        self,
        num_pages: Int32,
        raw_consumer,
        k_producer,
        sPacked: cute.Tensor,
        sScale: cute.Tensor,
        tKtK: cute.Tensor,
        row: Int32,
    ) -> None:
        """Expand raw NVFP4 pages into E4M3 TMEM K stages, one token row per thread."""

        store_cols = self.k_stage_tmem_cols // self.dequant_store_halves
        tmem_store = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(store_cols)),
                Float32,
            ),
            tKtK[(None, None, 0, 0)],
        ).get_slice(row)
        rK = cute.make_rmem_tensor(
            tmem_store.partition_S(
                cute.make_identity_tensor((self.page_size, store_cols))
            ).shape,
            Float32,
        )
        rK_words = cute.recast_tensor(rK, Int32)

        word_bytes = Int32.width // 8
        values_per_word = 2 * word_bytes
        chunk_words = self.packed_chunk_bytes // word_bytes
        packed_words = self.packed_bytes_per_token // word_bytes
        rPacked = cute.make_rmem_tensor((packed_words,), Int32)
        rScales = cute.make_rmem_tensor((self.scales_per_token // word_bytes,), Int32)
        rScalesF16x2 = cute.make_rmem_tensor((self.scales_per_token,), Int32)
        chunk_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=self.packed_chunk_bytes * 8
        )
        scale_load = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=self.scales_per_token * 8
        )
        smem = sPacked.iterator.memspace
        # Apply the TMA's swizzle to this token's page offsets by hand: it XORs
        # offset bits [base + shift, base + shift + bits) into [base, base + bits).
        # The chunk offsets stay below bit ``base + shift``, so the XOR term
        # depends on the token only.
        swizzle_bits, swizzle_base, swizzle_shift = self.packed_swizzle
        row_offset = row * Int32(self.packed_bytes_per_token)
        chunk_xor = (row_offset >> Int32(swizzle_shift)) & Int32(
            ((1 << swizzle_bits) - 1) << swizzle_base
        )
        packed_base = sPacked.iterator.toint()
        scale_row = sScale.iterator.toint() + row * Int32(self.scales_per_token)

        for _ in cutlass.range(num_pages, unroll=1):
            raw_full = raw_consumer.wait_and_advance()
            k_empty = k_producer.acquire_and_advance()
            packed_page = packed_base + raw_full.index * Int32(self.packed_page_bytes)
            for chunk in cutlass.range_constexpr(packed_words // chunk_words):
                chunk_ptr = cute.make_ptr(
                    Int32,
                    packed_page
                    + ((row_offset + Int32(chunk * self.packed_chunk_bytes)) ^ chunk_xor),
                    mem_space=smem,
                    assumed_align=self.packed_chunk_bytes,
                )
                cute.copy(
                    chunk_load,
                    cute.make_tensor(chunk_ptr, cute.make_layout(chunk_words)),
                    cute.make_tensor(
                        rPacked.iterator + chunk * chunk_words, cute.make_layout(chunk_words)
                    ),
                )
            scale_ptr = cute.make_ptr(
                Int32,
                scale_row + raw_full.index * Int32(self.scale_page_bytes),
                mem_space=smem,
                assumed_align=self.scales_per_token,
            )
            cute.copy(scale_load, cute.make_tensor(scale_ptr, rScales.layout), rScales)
            if cutlass.const_expr(self.hoist_scale_cvt):
                for group in cutlass.range_constexpr(self.scales_per_token):
                    rScalesF16x2[group] = _e4m3_scale_to_f16x2(
                        rScales[group // word_bytes], group % word_bytes
                    )
            words_per_half = packed_words // self.dequant_store_halves
            for half in cutlass.range_constexpr(self.dequant_store_halves):
                for local_word in cutlass.range_constexpr(words_per_half):
                    word = half * words_per_half + local_word
                    group = word * values_per_word // self.scale_group
                    scale = rScales[group // word_bytes] >> Int32(8 * (group % word_bytes))
                    if cutlass.const_expr(self.use_fp16_mma):
                        h0, h1, h2, h3 = _e2m1x8_f16x2_scaled_to_f16x8(
                            rPacked[word], rScalesF16x2[group]
                        )
                        rK_words[4 * local_word] = h0
                        rK_words[4 * local_word + 1] = h1
                        rK_words[4 * local_word + 2] = h2
                        rK_words[4 * local_word + 3] = h3
                    else:
                        if cutlass.const_expr(self.hoist_scale_cvt):
                            lo, hi = _e2m1x8_f16x2_scaled_to_e4m3x8(rPacked[word], rScalesF16x2[group])
                        elif cutlass.const_expr(self.use_qmul4):
                            lo, hi = _e2m1x8_scaled_to_e4m3x8(rPacked[word], scale)
                        else:
                            lo, hi = common_utils.cvt_fp4x8_e2m1_scaled_e4m3x8(rPacked[word], scale)
                        rK_words[2 * local_word] = lo
                        rK_words[2 * local_word + 1] = hi
                cute.copy(
                    tmem_store, rK, tmem_store.partition_D(tKtK[(None, None, half, k_empty.index)])
                )
            # Complete this thread's TMEM stores before one lane per warp
            # signals the stage to the MMA warp.
            cute.arch.fence_view_async_tmem_store()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                k_empty.commit()
            raw_full.release()
        k_producer.tail()


__all__ = ["Q8KV4DecodeIndexerSm100"]
