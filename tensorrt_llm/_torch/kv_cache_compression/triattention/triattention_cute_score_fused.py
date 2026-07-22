# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 CuTe-DSL scorer for the TriAttention mean-score path.

This is the ONLY score implementation: the per-head modes launch its score-only
entry and union eviction launches its fused score+stats+union pipeline. It
uses split real/imag TMA loads, BF16 and FP16 compensated UMMA, sqrt FTZ,
and producer-only page-ID lookahead. Geometries outside the exact contract
validated here raise loudly at setup
(``triattention.prepare_eviction_workspace``); there is no fallback path.
"""

from __future__ import annotations

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op


def _cute_sqrt_keyword_mode() -> str:
    """Probe which fast-sqrt spelling this CuTe DSL's ``cute.math.sqrt`` takes.

    The approximate-sqrt control was renamed across DSL releases: some expose
    ``approx``/``ftz`` keywords, cutlass 4.5 exposes a single ``fastmath``
    flag, and older releases expose a plain one-argument ``sqrt``. Passing an
    unknown keyword raises TypeError at trace time (inside ``cute.compile``,
    where it cannot be caught), so the capability is probed once at import
    time via signature inspection and folded into a trace-time constant.
    """
    import inspect

    try:
        parameters = inspect.signature(cute.math.sqrt).parameters
    except (TypeError, ValueError):
        return "plain"
    if "approx" in parameters and "ftz" in parameters:
        return "approx_ftz"
    if "fastmath" in parameters:
        return "fastmath"
    return "plain"


_CUTE_SQRT_KWARG_MODE = _cute_sqrt_keyword_mode()


@dsl_user_op
def _sqrt_approx_ftz(value: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Emit the approximate FTZ square root missing from CuTe DSL 4.5."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


CTA_M = 128
# Minimum tcgen05 MMA tile N: GQA groups below 8 ride zero-padded head
# columns (see the weight-builder loop and the partial-stats epilogue).
N = 8
THREADS = 256
EPILOGUE_THREADS = 128
RAW_PAGE_BUFFERS = 2

RAW_K_VECTOR_ELEMENTS = 8
TMA_DESCRIPTOR_QWORDS = 16
_SUPPORTED_PAGE_SHARDS = (2, 3)


class _TriScoreEpilogue:
    """Minimal TMEM-to-global epilogue for the score specialization."""

    def __init__(self) -> None:
        self.acc_dtype = cutlass.Float32

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        accumulator: cute.Tensor,
        output: cute.Tensor,
        epilogue_tile: cute.Tile,
        use_2cta_instrs: bool,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epilogue_tile,
            use_2cta_instrs,
        )
        accumulator_epilogue = cute.flat_divide(
            accumulator[((None, None), 0, 0)],
            epilogue_tile,
        )
        tiled_copy = tcgen05.make_tmem_copy(
            copy_atom,
            accumulator_epilogue[(None, None, 0, 0)],
        )
        thread_copy = tiled_copy.get_slice(tidx)
        thread_accumulator = thread_copy.partition_S(accumulator_epilogue)
        output_epilogue = cute.flat_divide(
            output[((None, None), 0, 0, None, None, None)],
            epilogue_tile,
        )
        thread_output = thread_copy.partition_D(output_epilogue)
        register_accumulator = cute.make_rmem_tensor(
            thread_output[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy, thread_accumulator, register_accumulator

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tiled_copy: cute.TiledCopy,
        output: cute.Tensor,
        epilogue_tile: cute.Tile,
        _unused_smem: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        output_epilogue = cute.flat_divide(
            output[((None, None), 0, 0, None, None, None)],
            epilogue_tile,
        )
        thread_copy = tiled_copy.get_slice(tidx)
        thread_output = thread_copy.partition_D(output_epilogue)
        register_output = cute.make_rmem_tensor(
            thread_output[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.c_dtype,
        )
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        return copy_atom, register_output, thread_output


class _TriAttentionScoreKernel(_TriScoreEpilogue):
    """Assign one CTA to each segment/KV-head task and retain W across pages."""

    def __init__(
        self,
        *,
        num_layers: int,
        seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_freqs: int,
        tokens_per_block: int,
        pool_shape: tuple[int, int, int, int, int],
        pool_strides: tuple[int, int, int, int, int],
        pool_dtype: type[cutlass.Numeric],
        page_shards: int,
        write_partial_stats: bool = False,
    ) -> None:
        """Build the single validated production specialization."""
        super().__init__()
        if pool_dtype is not cutlass.BFloat16:
            raise ValueError("TriAttention CuTe score requires BF16 K pages")
        if num_freqs not in (32, 64):
            raise ValueError(
                "TriAttention CuTe score requires 32 or 64 frequencies (head size 64/128)"
            )
        if tokens_per_block not in (32, 128):
            raise ValueError("TriAttention CuTe score requires 32- or 128-token pages")
        if num_q_heads % num_kv_heads or num_q_heads // num_kv_heads not in (4, 8):
            raise ValueError("TriAttention CuTe score requires GQA group 4 or 8")
        if page_shards not in _SUPPORTED_PAGE_SHARDS:
            raise ValueError("TriAttention CuTe score has unsupported page shards")

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.page_shards = page_shards
        self.write_partial_stats = write_partial_stats
        self.num_freqs = num_freqs
        # cos/sin/mlr coefficient planes per frequency.
        self.k_coeff = 3 * num_freqs
        self.tokens_per_block = tokens_per_block
        # One 128-token compute tile either matches a page exactly (the
        # validated 128-token geometry, one TMA box per phase) or spans
        # several pages (32-token pages: four page fragments per phase,
        # one TMA box each into the same transaction barrier). The
        # single-fragment schedule of the validated geometry is unchanged.
        self.box_tokens = min(CTA_M, tokens_per_block)
        self.fragments_per_phase = CTA_M // self.box_tokens
        self.pages_per_tile = self.fragments_per_phase
        self.halves_per_page = max(1, tokens_per_block // CTA_M)
        self.tile_tokens = self.halves_per_page * CTA_M
        self.max_tiles = (seq_len + self.tile_tokens - 1) // self.tile_tokens

        # Measured final choices that still shape layouts or generated code.
        self.prefetch_depth = 4
        self.sqrt_mode = "approx"
        self.k_staging_mode = "half_page_tma"
        self.use_tma = True
        self.cpasync_schedule = "sync_each_half"
        self.split_raw_tma = True
        self.raw_tma_feature_extent = num_freqs
        # Barrier transaction bytes for one phase: the full 128-token tile
        # of one coefficient plane, regardless of how many page fragments
        # deliver it.
        self.raw_tma_copy_bytes = CTA_M * num_freqs * (cutlass.BFloat16.width // 8)
        self.raw_tma_pipeline_stages = 2 * RAW_PAGE_BUFFERS if write_partial_stats else 1
        self.accumulator_pipeline_stages = 1
        self.umma_accumulator_partitions = 1
        self.raw_cpasync_direct_a = True
        self.weight_builder_mode = "coefficient_scalar_bf16_two_term"
        self.main_operand_mode = "bf16_raw_three_term_weight"
        self.numerical_policy = "three_term"
        self.magnitude_residual_mode = "fp16_mma_two_term_single_commit"
        self.fp16_magnitude_two_term = True
        self.magnitude_sqrt_ftz = True
        self.producer_page_id_prefetch = True
        self.producer_warp_id = 0
        self.physical_threads = THREADS
        self.shared_a_raw_alias = False
        self.compact_token_loop = True

        self.num_physical_pages, _, pool_kv_heads, pool_tokens, pool_dim = pool_shape
        if (
            pool_kv_heads != num_kv_heads
            or pool_tokens != tokens_per_block
            or pool_dim != 2 * num_freqs
        ):
            raise ValueError("K pool shape does not match the CuTe score specialization")
        self.s_page, _, self.s_kv_head, self.s_slot, self.s_dim = pool_strides
        if self.s_slot != 2 * num_freqs or self.s_dim != 1:
            raise ValueError(f"K pages must be contiguous [{tokens_per_block}, {2 * num_freqs}]")
        if self.s_page % RAW_K_VECTOR_ELEMENTS or self.s_kv_head % RAW_K_VECTOR_ELEMENTS:
            raise ValueError("K page and KV-head strides must preserve 16-byte alignment")

    @cute.jit
    def __call__(
        self,
        page_ids: cute.Tensor,
        seg_page_off: cute.Tensor,
        seg_req_id: cute.Tensor,
        seg_layer_id: cute.Tensor,
        seg_seq_len: cute.Tensor,
        seg_out_offset: cute.Tensor,
        token_starts: cute.Tensor,
        q_real: cute.Tensor,
        q_imag: cute.Tensor,
        mlr_coef: cute.Tensor,
        mean_cos: cute.Tensor,
        mean_sin: cute.Tensor,
        freq_scale_sq: cute.Tensor,
        output: cute.Tensor,
        partial_stats: cute.Tensor,
        pool_template: cute.Tensor,
        raw_tma_descriptors: cute.Tensor,
        request_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.c_dtype = output.element_type
        self.c_layout = utils.LayoutEnum.COL_MAJOR
        self.mma_tiler = (CTA_M, N, self.k_coeff)
        self.cta_tile_shape_mnk = self.mma_tiler
        self.epi_tile = (CTA_M, N)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float32,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        raw_bf16_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        main_a_shape = (
            (CTA_M, N, self.num_freqs)
            if self.main_operand_mode == "bf16_raw_three_term_weight"
            else self.mma_tiler
        )
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, main_a_shape, cutlass.Float32, 1)
        raw_bf16_a_smem_layout = sm100_utils.make_smem_layout_a(
            raw_bf16_tiled_mma,
            (CTA_M, N, 2 * self.num_freqs),
            cutlass.BFloat16,
            1,
        )
        raw_bf16_split_a_smem_layout = sm100_utils.make_smem_layout_a(
            raw_bf16_tiled_mma,
            (CTA_M, N, self.num_freqs),
            cutlass.BFloat16,
            2 * RAW_PAGE_BUFFERS,
        )
        # The full transport uses one K_SW128 8-KiB tile.  The split transport
        # packs two K_SW64 4-KiB stages into that same allocation, one each for
        # real and imaginary data; the compile-time schedule selects the view.
        raw_bf16_direct_a_smem_layout = (
            raw_bf16_split_a_smem_layout if self.split_raw_tma else raw_bf16_a_smem_layout
        )
        raw_tma_smem_layout = cute.make_composed_layout(
            raw_bf16_direct_a_smem_layout.inner,
            0,
            cute.make_layout(
                (self.raw_tma_feature_extent, self.box_tokens),
                stride=(1, self.raw_tma_feature_extent),
            ),
        )
        raw_tma_source_layout = cute.make_layout(
            (
                2 * self.num_freqs,
                self.tokens_per_block,
                (self.num_kv_heads, self.num_physical_pages),
            ),
            stride=(
                self.s_dim,
                self.s_slot,
                (self.s_kv_head, self.s_page),
            ),
        )
        raw_tma_source = cute.make_tensor(
            pool_template.iterator,
            raw_tma_source_layout,
        )
        raw_tma_atom, raw_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            raw_tma_source,
            raw_tma_smem_layout,
            (self.raw_tma_feature_extent, self.box_tokens),
        )
        raw_bf16_b_smem_layout = sm100_utils.make_smem_layout_b(
            raw_bf16_tiled_mma,
            (CTA_M, N, 2 * self.num_freqs),
            cutlass.BFloat16,
            1,
        )
        # The magnitude residual has only the frequency-count K.  A separate
        # compact descriptor lets the producer issue its UMMA steps before the
        # first commit, rather than waiting for and overwriting the main A tile.
        magnitude_lo_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, (CTA_M, N, self.num_freqs), cutlass.Float32, 1
        )
        magnitude_lo_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        magnitude_lo_fp16_smem_layout = sm100_utils.make_smem_layout_a(
            magnitude_lo_tiled_mma,
            (CTA_M, N, self.num_freqs),
            cutlass.Float16,
            1,
        )
        magnitude_hi_fp16_smem_layout = sm100_utils.make_smem_layout_b(
            magnitude_lo_tiled_mma,
            (CTA_M, N, self.num_freqs),
            cutlass.Float16,
            1,
        )
        magnitude_fp16_a_smem_layout = sm100_utils.make_smem_layout_a(
            magnitude_lo_tiled_mma,
            (CTA_M, N, self.num_freqs),
            cutlass.Float16,
            1,
        )
        magnitude_fp16_b_smem_layout = sm100_utils.make_smem_layout_b(
            magnitude_lo_tiled_mma,
            (CTA_M, N, self.num_freqs),
            cutlass.Float16,
            1,
        )
        main_b_shape = (
            (CTA_M, N, self.num_freqs)
            if self.main_operand_mode == "bf16_raw_three_term_weight"
            else self.mma_tiler
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, main_b_shape, cutlass.Float32, 1)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # Keep an explicit stage mode even for the one-stage control.  The
        # singleton mode folds away in codegen and lets both specializations
        # share the same producer/consumer slicing protocol.
        self.num_accumulator_slots = (
            self.accumulator_pipeline_stages * self.umma_accumulator_partitions
        )
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_accumulator_slots))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        b_hi_elements = cute.cosize(b_smem_layout.outer) * int(not self.fp16_magnitude_two_term)
        b_lo_elements = cute.cosize(b_smem_layout.outer) * int(
            self.numerical_policy == "three_term" and not self.fp16_magnitude_two_term
        )
        magnitude_lo_elements = cute.cosize(magnitude_lo_fp16_smem_layout.outer) * int(
            self.numerical_policy == "three_term"
            and self.magnitude_residual_mode in ("fp16_smem", "fp16_mma_single_commit")
        )
        magnitude_lo_fp32_elements = cute.cosize(magnitude_lo_smem_layout.outer) * int(
            self.numerical_policy == "three_term"
            and self.magnitude_residual_mode == "fp32_smem_single_commit"
        )
        magnitude_hi_fp16_elements = cute.cosize(magnitude_hi_fp16_smem_layout.outer) * int(
            self.numerical_policy == "three_term"
            and self.magnitude_residual_mode == "fp16_mma_single_commit"
        )
        a_elements = cute.cosize(a_smem_layout.outer) * int(
            not self.shared_a_raw_alias and not self.fp16_magnitude_two_term
        )
        raw_k_half_elements = CTA_M * 2 * self.num_freqs * RAW_PAGE_BUFFERS
        raw_k_elements = raw_k_half_elements * int(
            self.k_staging_mode in ("half_page_cpasync", "half_page_tma")
            and not self.shared_a_raw_alias
        )
        alias_a_elements = cute.cosize(a_smem_layout.outer) * int(self.shared_a_raw_alias)
        alias_raw_k_elements = raw_k_half_elements * int(self.shared_a_raw_alias)
        raw_bf16_a_elements = cute.cosize(raw_bf16_a_smem_layout.outer) * int(
            self.main_operand_mode == "bf16_raw_three_term_weight"
            and (
                not self.raw_cpasync_direct_a
                or self.cpasync_schedule not in ("sync_each_half", "intra_half_overlap")
            )
        )
        raw_bf16_b_elements = cute.cosize(raw_bf16_b_smem_layout.outer) * int(
            self.main_operand_mode == "bf16_raw_three_term_weight"
        )
        raw_bf16_b2_elements = raw_bf16_b_elements * int(
            self.weight_builder_mode != "coefficient_scalar_bf16_two_term"
        )
        magnitude_fp16_a_elements = cute.cosize(magnitude_fp16_a_smem_layout.outer) * int(
            self.fp16_magnitude_two_term
        )
        magnitude_fp16_b_elements = cute.cosize(magnitude_fp16_b_smem_layout.outer) * int(
            self.fp16_magnitude_two_term
        )
        stats_scratch_elements = 72 * int(self.write_partial_stats)

        @cute.union
        class SharedARawAlias:
            # The two descriptors are byte-identical in size (8 KiB) and have
            # disjoint lifetimes in the alias specialization.
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, alias_a_elements],
                1024,
            ]
            sRawK: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, alias_raw_k_elements],
                16,
            ]

        @cute.struct
        class SharedStorage:
            # PipelineUmmaAsync uses one full and one empty barrier per stage.
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.accumulator_pipeline_stages * 2]
            raw_tma_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64,
                2 * self.raw_tma_pipeline_stages * int(self.use_tma),
            ]
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, a_elements],
                1024,
            ]
            sB_hi: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, b_hi_elements],
                1024,
            ]
            sB_lo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, b_lo_elements],
                1024,
            ]
            sMagnitudeLo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, magnitude_lo_elements],
                1024,
            ]
            sMagnitudeLoFp32: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, magnitude_lo_fp32_elements],
                1024,
            ]
            sMagnitudeHiFp16: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, magnitude_hi_fp16_elements],
                1024,
            ]
            sRawK: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, raw_k_elements],
                1024,
            ]
            sRawBf16A: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, raw_bf16_a_elements],
                1024,
            ]
            sRawBf16B0: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, raw_bf16_b_elements],
                1024,
            ]
            sRawBf16B1: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, raw_bf16_b_elements],
                1024,
            ]
            sRawBf16B2: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, raw_bf16_b2_elements],
                1024,
            ]
            sMagnitudeFp16A0: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, magnitude_fp16_a_elements],
                1024,
            ]
            sMagnitudeFp16A1: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, magnitude_fp16_a_elements],
                1024,
            ]
            sMagnitudeFp16B0: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, magnitude_fp16_b_elements],
                1024,
            ]
            sMagnitudeFp16B1: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, magnitude_fp16_b_elements],
                1024,
            ]
            sStats: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, stats_scratch_elements],
                16,
            ]
            sARawAlias: SharedARawAlias

        self.shared_storage = SharedStorage
        # 64-bit: at large request counts this product exceeds 2^31 (the
        # score scratch spans request*layer segments of seq_len columns), so
        # the head-plane stride must reach the kernel as Int64.
        sum_seq = cutlass.Int64(request_count * self.num_layers * self.seq_len)
        num_ctas = request_count * self.num_layers * self.num_kv_heads * self.page_shards
        self.kernel(
            tiled_mma,
            raw_bf16_tiled_mma,
            magnitude_lo_tiled_mma,
            raw_tma_atom,
            raw_tma_tensor,
            raw_tma_descriptors,
            page_ids,
            seg_page_off,
            seg_req_id,
            seg_layer_id,
            seg_seq_len,
            seg_out_offset,
            token_starts,
            q_real,
            q_imag,
            mlr_coef,
            mean_cos,
            mean_sin,
            freq_scale_sq,
            output,
            partial_stats,
            sum_seq,
            a_smem_layout,
            raw_bf16_a_smem_layout,
            raw_bf16_direct_a_smem_layout,
            raw_bf16_split_a_smem_layout,
            raw_tma_smem_layout,
            raw_bf16_b_smem_layout,
            magnitude_lo_smem_layout,
            magnitude_lo_fp16_smem_layout,
            magnitude_hi_fp16_smem_layout,
            magnitude_fp16_a_smem_layout,
            magnitude_fp16_b_smem_layout,
            b_smem_layout,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(self.physical_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        raw_bf16_tiled_mma: cute.TiledMma,
        magnitude_lo_tiled_mma: cute.TiledMma,
        raw_tma_atom: cute.CopyAtom,
        raw_tma_source: cute.Tensor,
        raw_tma_descriptors: cute.Tensor,
        page_ids: cute.Tensor,
        seg_page_off: cute.Tensor,
        seg_req_id: cute.Tensor,
        seg_layer_id: cute.Tensor,
        seg_seq_len: cute.Tensor,
        seg_out_offset: cute.Tensor,
        token_starts: cute.Tensor,
        q_real: cute.Tensor,
        q_imag: cute.Tensor,
        mlr_coef: cute.Tensor,
        mean_cos: cute.Tensor,
        mean_sin: cute.Tensor,
        freq_scale_sq: cute.Tensor,
        output: cute.Tensor,
        partial_stats: cute.Tensor,
        sum_seq: cutlass.Int64,
        a_smem_layout: cute.ComposedLayout,
        raw_bf16_a_smem_layout: cute.ComposedLayout,
        raw_bf16_direct_a_smem_layout: cute.ComposedLayout,
        raw_bf16_split_a_smem_layout: cute.ComposedLayout,
        raw_tma_smem_layout: cute.ComposedLayout,
        raw_bf16_b_smem_layout: cute.ComposedLayout,
        magnitude_lo_smem_layout: cute.ComposedLayout,
        magnitude_lo_fp16_smem_layout: cute.ComposedLayout,
        magnitude_hi_fp16_smem_layout: cute.ComposedLayout,
        magnitude_fp16_a_smem_layout: cute.ComposedLayout,
        magnitude_fp16_b_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta_index, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % 32
        task = cta_index // self.page_shards
        page_shard = cta_index % self.page_shards
        segment = task // self.num_kv_heads
        kv_head = task % self.num_kv_heads
        req_id = seg_req_id[segment]
        layer_id = seg_layer_id[segment]
        valid_seq_len = seg_seq_len[segment]
        page_off = seg_page_off[segment]
        out_base = seg_out_offset[segment]
        # Per-request score window start (the request's pinned prompt
        # length), loaded like the other per-segment metadata: each CTA
        # owns one segment, so its whole schedule derives from one start.
        # Scratch writes stay absolute; only the scoring/stats domain and
        # the first scored page move per request.
        score_start = cutlass.Int32(token_starts[req_id])

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sMagnitudeFp16A0 = storage.sMagnitudeFp16A0.get_tensor(
            magnitude_fp16_a_smem_layout.outer,
            swizzle=magnitude_fp16_a_smem_layout.inner,
        )
        sMagnitudeFp16A1 = storage.sMagnitudeFp16A1.get_tensor(
            magnitude_fp16_a_smem_layout.outer,
            swizzle=magnitude_fp16_a_smem_layout.inner,
        )
        sMagnitudeFp16B0 = storage.sMagnitudeFp16B0.get_tensor(
            magnitude_fp16_b_smem_layout.outer,
            swizzle=magnitude_fp16_b_smem_layout.inner,
        )
        sMagnitudeFp16B1 = storage.sMagnitudeFp16B1.get_tensor(
            magnitude_fp16_b_smem_layout.outer,
            swizzle=magnitude_fp16_b_smem_layout.inner,
        )
        if cutlass.const_expr(self.write_partial_stats):
            sStats = storage.sStats.get_tensor(cute.make_layout(72))
        raw_k_storage = storage.sRawK
        cpasync_raw_k_0 = raw_k_storage.get_tensor(
            raw_bf16_direct_a_smem_layout.outer,
            swizzle=raw_bf16_direct_a_smem_layout.inner,
        )
        cpasync_raw_k_real = cpasync_raw_k_0[(None, None, None, 0)]
        cpasync_raw_k_imag = cpasync_raw_k_0[(None, None, None, 1)]
        cpasync_raw_k_real_next = cpasync_raw_k_0[(None, None, None, 2)]
        cpasync_raw_k_imag_next = cpasync_raw_k_0[(None, None, None, 3)]
        # Each stage slice retains the K_SW64 pointer flags.  Reuse only
        # the feature-first outer mapping for the corresponding TMA
        # destination so the swizzle is not applied twice.
        raw_tma_source_tiles = cute.local_tile(
            raw_tma_source,
            (self.raw_tma_feature_extent, self.box_tokens),
            coord=(None, None, None),
        )
        # One smem view and TMA partition per page fragment of the
        # 128-token tile, for each of the four stage slices. Fragment f
        # lands box_tokens rows deeper in the same stage; the offset is a
        # whole multiple of the swizzle period, so the descriptor swizzle
        # stays phase-aligned.
        raw_tma_shared_partition_real = []
        raw_tma_shared_partition_imag = []
        raw_tma_shared_partition_real_next = []
        raw_tma_shared_partition_imag_next = []
        raw_tma_global_partition = None
        for fragment in cutlass.range_constexpr(self.fragments_per_phase):
            fragment_offset = fragment * self.box_tokens * self.raw_tma_feature_extent
            fragment_real = cute.make_tensor(
                cpasync_raw_k_real.iterator + fragment_offset,
                raw_tma_smem_layout.outer,
            )
            fragment_imag = cute.make_tensor(
                cpasync_raw_k_imag.iterator + fragment_offset,
                raw_tma_smem_layout.outer,
            )
            fragment_real_next = cute.make_tensor(
                cpasync_raw_k_real_next.iterator + fragment_offset,
                raw_tma_smem_layout.outer,
            )
            fragment_imag_next = cute.make_tensor(
                cpasync_raw_k_imag_next.iterator + fragment_offset,
                raw_tma_smem_layout.outer,
            )
            partition_real, global_partition = cpasync.tma_partition(
                raw_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(fragment_real, 0, 2),
                cute.group_modes(raw_tma_source_tiles, 0, 2),
            )
            partition_imag, _ = cpasync.tma_partition(
                raw_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(fragment_imag, 0, 2),
                cute.group_modes(raw_tma_source_tiles, 0, 2),
            )
            partition_real_next, _ = cpasync.tma_partition(
                raw_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(fragment_real_next, 0, 2),
                cute.group_modes(raw_tma_source_tiles, 0, 2),
            )
            partition_imag_next, _ = cpasync.tma_partition(
                raw_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(fragment_imag_next, 0, 2),
                cute.group_modes(raw_tma_source_tiles, 0, 2),
            )
            raw_tma_shared_partition_real.append(partition_real)
            raw_tma_shared_partition_imag.append(partition_imag)
            raw_tma_shared_partition_real_next.append(partition_real_next)
            raw_tma_shared_partition_imag_next.append(partition_imag_next)
            raw_tma_global_partition = global_partition
        raw_tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.GMEM,
            128,
        )
        raw_tma_descriptor_ptr = raw_tensormap_manager.get_tensormap_ptr(
            (raw_tma_descriptors.iterator + layer_id * TMA_DESCRIPTOR_QWORDS).align(128),
            cute.AddressSpace.generic,
        )
        sRawBf16B0 = storage.sRawBf16B0.get_tensor(
            raw_bf16_b_smem_layout.outer,
            swizzle=raw_bf16_b_smem_layout.inner,
        )
        sRawBf16B1 = storage.sRawBf16B1.get_tensor(
            raw_bf16_b_smem_layout.outer,
            swizzle=raw_bf16_b_smem_layout.inner,
        )

        page_index = score_start // self.tile_tokens + page_shard
        page_start = page_index * self.tile_tokens
        shard_first_page_start = page_start
        pages_processed = cutlass.Int32(0)
        if cutlass.const_expr(self.write_partial_stats):
            stats_page_scores_m128 = cute.make_rmem_tensor((N,), cutlass.Float32)
            stats_origins_m128 = cute.make_rmem_tensor((N,), cutlass.Float32)
            stats_sums_m128 = cute.make_rmem_tensor((N,), cutlass.Float32)
            stats_square_sums_m128 = cute.make_rmem_tensor((N,), cutlass.Float32)
            for stats_head in cutlass.range_constexpr(N):
                stats_sums_m128[stats_head] = cutlass.Float32(0.0)
                stats_square_sums_m128[stats_head] = cutlass.Float32(0.0)
        producer_prefetched_page_id_lane0 = cutlass.Int32(0)
        if cutlass.const_expr(self.fragments_per_phase > 1):
            # Per-fragment page-id registers for multi-page compute tiles.
            # Slot 0 is unused: fragment 0 keeps the scalar broadcast
            # registers of the validated single-fragment schedule.
            producer_prefetched_page_ids_lane0 = cute.make_rmem_tensor(
                (self.pages_per_tile,), cutlass.Int32
            )
            physical_page_fragments = cute.make_rmem_tensor((self.pages_per_tile,), cutlass.Int32)
            prefetched_page_fragments = cute.make_rmem_tensor((self.pages_per_tile,), cutlass.Int32)
        shard_has_page = valid_seq_len > score_start and page_start < valid_seq_len
        empty_shard = valid_seq_len <= score_start or page_start >= valid_seq_len
        if cutlass.dynamic_expr(shard_has_page):
            if warp_idx == self.producer_warp_id:
                if lane_idx == 0:
                    # The staged K-plane entries encode physical_page *
                    # kv_factor (2); decode to the pool page index here,
                    # matching the production score kernel.
                    producer_prefetched_page_id_lane0 = (
                        cutlass.Int32(page_ids[page_off + page_index * self.pages_per_tile]) // 2
                    )
                    if cutlass.const_expr(self.fragments_per_phase > 1):
                        for fragment in cutlass.range_constexpr(1, self.pages_per_tile):
                            # The tail tile may not reach this fragment's
                            # page; clamp to the first fragment (those
                            # scores lie past the valid width and are
                            # masked downstream) so the TMA never
                            # dereferences an unstaged block entry.
                            fragment_page_id = producer_prefetched_page_id_lane0
                            if page_start + fragment * self.box_tokens < valid_seq_len:
                                fragment_page_id = (
                                    cutlass.Int32(
                                        page_ids[
                                            page_off + page_index * self.pages_per_tile + fragment
                                        ]
                                    )
                                    // 2
                                )
                            producer_prefetched_page_ids_lane0[fragment] = fragment_page_id
        tCrRawBf16ASplit = raw_bf16_tiled_mma.make_fragment_A(cpasync_raw_k_0)
        tCrRawBf16B0 = raw_bf16_tiled_mma.make_fragment_B(sRawBf16B0)
        tCrRawBf16B1 = raw_bf16_tiled_mma.make_fragment_B(sRawBf16B1)
        tCrMagnitudeFp16A0 = magnitude_lo_tiled_mma.make_fragment_A(sMagnitudeFp16A0)
        tCrMagnitudeFp16A1 = magnitude_lo_tiled_mma.make_fragment_A(sMagnitudeFp16A1)
        tCrMagnitudeFp16B0 = magnitude_lo_tiled_mma.make_fragment_B(sMagnitudeFp16B0)
        tCrMagnitudeFp16B1 = magnitude_lo_tiled_mma.make_fragment_B(sMagnitudeFp16B1)
        raw_tma_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.raw_tma_mbar_ptr.data_ptr(),
            num_stages=self.raw_tma_pipeline_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS // 32),
            tx_count=self.raw_tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            defer_sync=True,
        )
        raw_tma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.raw_tma_pipeline_stages,
        )
        raw_tma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.raw_tma_pipeline_stages,
        )

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.accumulator_pipeline_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS // 32),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.accumulator_pipeline_stages
        )
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.accumulator_pipeline_stages
        )
        stats_epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=EPILOGUE_THREADS,
        )
        cute.arch.mbarrier_init_fence()
        for weight_round in cutlass.range_constexpr(N * self.k_coeff // THREADS):
            linear_index = tidx + weight_round * THREADS
            qg = linear_index // self.k_coeff
            feature = linear_index % self.k_coeff
            coefficient_kind = feature // self.num_freqs
            frequency = feature % self.num_freqs
            mean_offset = req_id * self.num_freqs + frequency
            # GQA groups below the minimum MMA tile N=8 ride padded
            # columns: they read the group's first head (any valid
            # address) and force zero coefficients, so the padded score
            # columns come out zero and land in scratch rows the union
            # finalizer never reads.
            qg_read = qg
            if cutlass.const_expr(self.group_size < N):
                if qg_read >= self.group_size:
                    qg_read = cutlass.Int32(0)
            q_head = kv_head * self.group_size + qg_read
            calib_offset = (layer_id * self.num_q_heads + q_head) * self.num_freqs + frequency
            qr = cutlass.Float32(q_real[calib_offset])
            qi = cutlass.Float32(q_imag[calib_offset])
            mcos = cutlass.Float32(mean_cos[mean_offset])
            msin = cutlass.Float32(mean_sin[mean_offset])
            scale = cutlass.Float32(freq_scale_sq[frequency])
            value = cutlass.Float32(0.0)
            if coefficient_kind == 0:
                value = scale * (qr * mcos - qi * msin)
            elif coefficient_kind == 1:
                value = scale * (qr * msin + qi * mcos)
            else:
                value = scale * cutlass.Float32(mlr_coef[calib_offset])
            if cutlass.const_expr(self.group_size < N):
                if qg >= self.group_size:
                    value = cutlass.Float32(0.0)
            raw_k_block = feature // 16
            magnitude_k_block = frequency // 16
            if coefficient_kind < 2:
                value_bf16_0 = cutlass.BFloat16(value)
                residual_1 = value - cutlass.Float32(value_bf16_0)
                value_bf16_1 = cutlass.BFloat16(residual_1)
                raw_coord = (
                    (qg, feature % 16),
                    0,
                    raw_k_block,
                    0,
                )
                sRawBf16B0[raw_coord] = value_bf16_0
                sRawBf16B1[raw_coord] = value_bf16_1
            else:
                value_fp16_0 = cutlass.Float16(value)
                value_fp16_1 = cutlass.Float16(value - cutlass.Float32(value_fp16_0))
                magnitude_coord_fp16 = (
                    (qg, frequency % 16),
                    0,
                    magnitude_k_block,
                    0,
                )
                sMagnitudeFp16B0[magnitude_coord_fp16] = value_fp16_0
                sMagnitudeFp16B1[magnitude_coord_fp16] = value_fp16_1
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_accumulator_slots))
        if warp_idx == 0:
            cute.arch.alloc_tmem(
                self.num_tmem_alloc_cols,
                storage.tmem_holding_buf,
                is_two_cta=False,
            )
        cute.arch.barrier()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32,
            alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        if cutlass.dynamic_expr(empty_shard):
            if warp_idx == self.producer_warp_id:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        thr_mma = tiled_mma.get_slice(0)
        if cutlass.const_expr(self.write_partial_stats):
            if cutlass.dynamic_expr(shard_has_page):
                if warp_idx == self.producer_warp_id:
                    prefetched_physical_page = cute.arch.shuffle_sync(
                        producer_prefetched_page_id_lane0,
                        0,
                    )
                    if cutlass.const_expr(self.fragments_per_phase > 1):
                        for fragment in cutlass.range_constexpr(1, self.pages_per_tile):
                            prefetched_page_fragments[fragment] = cute.arch.shuffle_sync(
                                producer_prefetched_page_ids_lane0[fragment],
                                0,
                            )
                    raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                    for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                        fragment_page = prefetched_physical_page
                        if cutlass.const_expr(fragment > 0):
                            fragment_page = prefetched_page_fragments[fragment]
                        cute.copy(
                            raw_tma_atom,
                            raw_tma_global_partition[
                                (
                                    None,
                                    0,
                                    0,
                                    (kv_head, fragment_page),
                                )
                            ],
                            raw_tma_shared_partition_real[fragment],
                            tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                raw_tma_producer_state
                            ),
                            tma_desc_ptr=raw_tma_descriptor_ptr,
                        )
                    raw_tma_producer_state.advance()
                    raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                    for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                        fragment_page = prefetched_physical_page
                        if cutlass.const_expr(fragment > 0):
                            fragment_page = prefetched_page_fragments[fragment]
                        cute.copy(
                            raw_tma_atom,
                            raw_tma_global_partition[
                                (
                                    None,
                                    1,
                                    0,
                                    (kv_head, fragment_page),
                                )
                            ],
                            raw_tma_shared_partition_imag[fragment],
                            tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                raw_tma_producer_state
                            ),
                            tma_desc_ptr=raw_tma_descriptor_ptr,
                        )
                    raw_tma_producer_state.advance()
        while (
            valid_seq_len > score_start
            and page_start < valid_seq_len
            and pages_processed < self.max_tiles
        ):
            physical_page = cutlass.Int32(0)
            if warp_idx == self.producer_warp_id:
                physical_page = cute.arch.shuffle_sync(
                    producer_prefetched_page_id_lane0,
                    0,
                )
                if cutlass.const_expr(self.fragments_per_phase > 1):
                    for fragment in cutlass.range_constexpr(1, self.pages_per_tile):
                        physical_page_fragments[fragment] = cute.arch.shuffle_sync(
                            producer_prefetched_page_ids_lane0[fragment],
                            0,
                        )
            for page_half in cutlass.range_constexpr(self.halves_per_page):
                raw_page_buffer = cutlass.Int32(0)
                if cutlass.const_expr(self.write_partial_stats):
                    raw_page_buffer = pages_processed % RAW_PAGE_BUFFERS
                raw_real_stage = raw_page_buffer * 2
                raw_imag_stage = raw_real_stage + 1
                if cutlass.const_expr(not self.write_partial_stats):
                    if warp_idx == self.producer_warp_id:
                        # Phase 0 fills the packed 4-KiB K_SW64 real
                        # view. Every producer-warp lane participates
                        # in the PipelineTmaAsync barrier election.
                        raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                        for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                            fragment_page = physical_page
                            if cutlass.const_expr(fragment > 0):
                                fragment_page = physical_page_fragments[fragment]
                            cute.copy(
                                raw_tma_atom,
                                raw_tma_global_partition[
                                    (
                                        None,
                                        0,
                                        page_half,
                                        (kv_head, fragment_page),
                                    )
                                ],
                                raw_tma_shared_partition_real[fragment],
                                tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                    raw_tma_producer_state
                                ),
                                tma_desc_ptr=raw_tma_descriptor_ptr,
                            )
                        raw_tma_producer_state.advance()
                raw_tma_pipeline.consumer_wait(raw_tma_consumer_state)
                raw_tma_pipeline.consumer_release(raw_tma_consumer_state)
                raw_tma_consumer_state.advance()
                if cutlass.const_expr(not self.write_partial_stats):
                    if warp_idx == self.producer_warp_id:
                        raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                        for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                            fragment_page = physical_page
                            if cutlass.const_expr(fragment > 0):
                                fragment_page = physical_page_fragments[fragment]
                            cute.copy(
                                raw_tma_atom,
                                raw_tma_global_partition[
                                    (
                                        None,
                                        1,
                                        page_half,
                                        (kv_head, fragment_page),
                                    )
                                ],
                                raw_tma_shared_partition_imag[fragment],
                                tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                    raw_tma_producer_state
                                ),
                                tma_desc_ptr=raw_tma_descriptor_ptr,
                            )
                        raw_tma_producer_state.advance()

                if cutlass.const_expr(
                    self.producer_page_id_prefetch and page_half == self.halves_per_page - 1
                ):
                    next_page_id_lane0 = cutlass.Int32(0)
                    if warp_idx == self.producer_warp_id:
                        if lane_idx == 0:
                            next_page_start = page_start + self.tile_tokens * self.page_shards
                            next_pages_processed = pages_processed + 1
                            if (
                                next_page_start < valid_seq_len
                                and next_pages_processed < self.max_tiles
                            ):
                                next_page_id_lane0 = (
                                    cutlass.Int32(
                                        page_ids[
                                            page_off
                                            + (page_index + self.page_shards) * self.pages_per_tile
                                        ]
                                    )
                                    // 2
                                )
                                if cutlass.const_expr(self.fragments_per_phase > 1):
                                    for fragment in cutlass.range_constexpr(1, self.pages_per_tile):
                                        # Same tail-tile clamp as the initial
                                        # prefetch: fall back to the first
                                        # fragment's page.
                                        next_fragment_page_id = next_page_id_lane0
                                        if (
                                            next_page_start + fragment * self.box_tokens
                                            < valid_seq_len
                                        ):
                                            next_fragment_page_id = (
                                                cutlass.Int32(
                                                    page_ids[
                                                        page_off
                                                        + (page_index + self.page_shards)
                                                        * self.pages_per_tile
                                                        + fragment
                                                    ]
                                                )
                                                // 2
                                            )
                                        producer_prefetched_page_ids_lane0[fragment] = (
                                            next_fragment_page_id
                                        )
                    producer_prefetched_page_id_lane0 = next_page_id_lane0

                # Submit B0-real while the imaginary TMA is in flight.
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                if warp_idx == self.producer_warp_id:
                    acc_pipeline.producer_acquire(acc_producer_state)
                    raw_bf16_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        False,
                    )
                    for raw_k_block in cutlass.range_constexpr(self.num_freqs // 16):
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, raw_real_stage)],
                            tCrRawBf16B0[(None, None, raw_k_block, 0)],
                            tCtAcc,
                        )
                        raw_bf16_tiled_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            True,
                        )
                raw_tma_pipeline.consumer_wait(raw_tma_consumer_state)
                if cutlass.const_expr(self.write_partial_stats):
                    if warp_idx == self.producer_warp_id:
                        if cutlass.const_expr(page_half + 1 < self.halves_per_page):
                            prefetched_physical_page = physical_page
                            prefetch_next_raw = True
                            prefetched_page_half = page_half + 1
                        else:
                            next_page_start = page_start + self.tile_tokens * self.page_shards
                            next_pages_processed = pages_processed + 1
                            prefetch_next_raw = (
                                next_page_start < valid_seq_len
                                and next_pages_processed < self.max_tiles
                            )
                            prefetched_physical_page = cute.arch.shuffle_sync(
                                producer_prefetched_page_id_lane0,
                                0,
                            )
                            if cutlass.const_expr(self.fragments_per_phase > 1):
                                for fragment in cutlass.range_constexpr(1, self.pages_per_tile):
                                    prefetched_page_fragments[fragment] = cute.arch.shuffle_sync(
                                        producer_prefetched_page_ids_lane0[fragment],
                                        0,
                                    )
                            prefetched_page_half = 0
                        if cutlass.dynamic_expr(prefetch_next_raw):
                            next_raw_page_buffer = (raw_page_buffer + 1) % RAW_PAGE_BUFFERS
                            raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                            if cutlass.dynamic_expr(next_raw_page_buffer == 0):
                                for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                                    fragment_page = prefetched_physical_page
                                    if cutlass.const_expr(fragment > 0):
                                        fragment_page = prefetched_page_fragments[fragment]
                                    cute.copy(
                                        raw_tma_atom,
                                        raw_tma_global_partition[
                                            (
                                                None,
                                                0,
                                                prefetched_page_half,
                                                (kv_head, fragment_page),
                                            )
                                        ],
                                        raw_tma_shared_partition_real[fragment],
                                        tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                            raw_tma_producer_state
                                        ),
                                        tma_desc_ptr=raw_tma_descriptor_ptr,
                                    )
                            else:
                                for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                                    fragment_page = prefetched_physical_page
                                    if cutlass.const_expr(fragment > 0):
                                        fragment_page = prefetched_page_fragments[fragment]
                                    cute.copy(
                                        raw_tma_atom,
                                        raw_tma_global_partition[
                                            (
                                                None,
                                                0,
                                                prefetched_page_half,
                                                (kv_head, fragment_page),
                                            )
                                        ],
                                        raw_tma_shared_partition_real_next[fragment],
                                        tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                            raw_tma_producer_state
                                        ),
                                        tma_desc_ptr=raw_tma_descriptor_ptr,
                                    )
                            raw_tma_producer_state.advance()
                            raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                            if cutlass.dynamic_expr(next_raw_page_buffer == 0):
                                for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                                    fragment_page = prefetched_physical_page
                                    if cutlass.const_expr(fragment > 0):
                                        fragment_page = prefetched_page_fragments[fragment]
                                    cute.copy(
                                        raw_tma_atom,
                                        raw_tma_global_partition[
                                            (
                                                None,
                                                1,
                                                prefetched_page_half,
                                                (kv_head, fragment_page),
                                            )
                                        ],
                                        raw_tma_shared_partition_imag[fragment],
                                        tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                            raw_tma_producer_state
                                        ),
                                        tma_desc_ptr=raw_tma_descriptor_ptr,
                                    )
                            else:
                                for fragment in cutlass.range_constexpr(self.fragments_per_phase):
                                    fragment_page = prefetched_physical_page
                                    if cutlass.const_expr(fragment > 0):
                                        fragment_page = prefetched_page_fragments[fragment]
                                    cute.copy(
                                        raw_tma_atom,
                                        raw_tma_global_partition[
                                            (
                                                None,
                                                1,
                                                prefetched_page_half,
                                                (kv_head, fragment_page),
                                            )
                                        ],
                                        raw_tma_shared_partition_imag_next[fragment],
                                        tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(
                                            raw_tma_producer_state
                                        ),
                                        tma_desc_ptr=raw_tma_descriptor_ptr,
                                    )
                            raw_tma_producer_state.advance()
                # Each of the 32 lanes stages one frequency per pass; 64-
                # frequency heads take two passes.
                for freq_rep in cutlass.range_constexpr(self.num_freqs // 32):
                    frequency = lane_idx + 32 * freq_rep
                    # Issue several independent token loads before consuming
                    # any of them.  This bounded RMEM window is unchanged; the
                    # optional half-page staging only switches its K source
                    # from global to the single raw shared buffer.
                    for token_base in cutlass.range(
                        0,
                        CTA_M // (THREADS // 32),
                        self.prefetch_depth,
                        unroll_full=not self.compact_token_loop,
                    ):
                        staged_real = cute.make_rmem_tensor((self.prefetch_depth,), cutlass.Float32)
                        staged_imag = cute.make_rmem_tensor((self.prefetch_depth,), cutlass.Float32)
                        for prefetch_index in cutlass.range_constexpr(self.prefetch_depth):
                            token_round = token_base + prefetch_index
                            token = warp_idx + token_round * (THREADS // 32)
                            staged_real[prefetch_index] = cutlass.Float32(
                                cpasync_raw_k_0[
                                    (
                                        (token, frequency % 16),
                                        0,
                                        frequency // 16,
                                        raw_real_stage,
                                    )
                                ]
                            )
                            staged_imag[prefetch_index] = cutlass.Float32(
                                cpasync_raw_k_0[
                                    (
                                        (token, frequency % 16),
                                        0,
                                        frequency // 16,
                                        raw_imag_stage,
                                    )
                                ]
                            )

                        for prefetch_index in cutlass.range_constexpr(self.prefetch_depth):
                            token_round = token_base + prefetch_index
                            token = warp_idx + token_round * (THREADS // 32)
                            real = staged_real[prefetch_index]
                            imag = staged_imag[prefetch_index]
                            norm2 = real * real + imag * imag
                            if cutlass.const_expr(_CUTE_SQRT_KWARG_MODE == "approx_ftz"):
                                magnitude = cute.math.sqrt(
                                    norm2,
                                    approx=self.sqrt_mode == "approx",
                                    ftz=self.magnitude_sqrt_ftz,
                                )
                            elif cutlass.const_expr(_CUTE_SQRT_KWARG_MODE == "fastmath"):
                                magnitude = _sqrt_approx_ftz(norm2)
                            else:
                                # DSLs with neither spelling get the plain (IEEE)
                                # sqrt, which is strictly MORE accurate than the
                                # measured approx choice; the equivalence test's
                                # tolerance absorbs the difference.
                                magnitude = cute.math.sqrt(norm2)
                            magnitude_fp16_0 = cutlass.Float16(magnitude)
                            magnitude_fp16_1 = cutlass.Float16(
                                magnitude - cutlass.Float32(magnitude_fp16_0)
                            )
                            magnitude_k_block_fp16 = frequency // 16
                            magnitude_coord_fp16 = (
                                (token, frequency % 16),
                                0,
                                magnitude_k_block_fp16,
                                0,
                            )
                            sMagnitudeFp16A0[magnitude_coord_fp16] = magnitude_fp16_0
                            sMagnitudeFp16A1[magnitude_coord_fp16] = magnitude_fp16_1

                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                if warp_idx == self.producer_warp_id:
                    # Finish B0-imag, then issue B1-real and B1-imag.
                    raw_bf16_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        True,
                    )
                    for raw_k_block in cutlass.range_constexpr(self.num_freqs // 16):
                        imag_b_block = self.num_freqs // 16 + raw_k_block
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, raw_imag_stage)],
                            tCrRawBf16B0[(None, None, imag_b_block, 0)],
                            tCtAcc,
                        )
                    for raw_k_block in cutlass.range_constexpr(self.num_freqs // 16):
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, raw_real_stage)],
                            tCrRawBf16B1[(None, None, raw_k_block, 0)],
                            tCtAcc,
                        )
                    for raw_k_block in cutlass.range_constexpr(self.num_freqs // 16):
                        imag_b_block = self.num_freqs // 16 + raw_k_block
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, raw_imag_stage)],
                            tCrRawBf16B1[(None, None, imag_b_block, 0)],
                            tCtAcc,
                        )
                    # Keep the full four-product control unchanged.
                    # The independent omit_a1b1 mode drops only the
                    # second-order residual product; all other FP16
                    # K16 products retain their original order.
                    magnitude_lo_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        True,
                    )
                    for magnitude_k_block in cutlass.range_constexpr(self.num_freqs // 16):
                        cute.gemm(
                            magnitude_lo_tiled_mma,
                            tCtAcc,
                            tCrMagnitudeFp16A0[(None, None, magnitude_k_block, 0)],
                            tCrMagnitudeFp16B0[(None, None, magnitude_k_block, 0)],
                            tCtAcc,
                        )
                    for magnitude_k_block in cutlass.range_constexpr(self.num_freqs // 16):
                        cute.gemm(
                            magnitude_lo_tiled_mma,
                            tCtAcc,
                            tCrMagnitudeFp16A0[(None, None, magnitude_k_block, 0)],
                            tCrMagnitudeFp16B1[(None, None, magnitude_k_block, 0)],
                            tCtAcc,
                        )
                    for magnitude_k_block in cutlass.range_constexpr(self.num_freqs // 16):
                        cute.gemm(
                            magnitude_lo_tiled_mma,
                            tCtAcc,
                            tCrMagnitudeFp16A1[(None, None, magnitude_k_block, 0)],
                            tCrMagnitudeFp16B0[(None, None, magnitude_k_block, 0)],
                            tCtAcc,
                        )
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                    if pages_processed == 0:
                        if cutlass.const_expr(page_half == 0):
                            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(self.write_partial_stats):
                    # The alternate raw-page buffer was filled while this
                    # page's UMMA completed. Release only the current imag
                    # phase after all of its asynchronous consumers finish.
                    raw_tma_pipeline.consumer_release(raw_tma_consumer_state)
                    raw_tma_consumer_state.advance()
                # Every term multiplying sum_seq must stay 64-bit: with
                # request*layer segments of seq_len columns the head-plane
                # stride alone can exceed 2^31. The scratch head axis is
                # padded to the MMA tile N=8 per KV head (group-4 columns
                # 4..7 land in padded planes holding zero scores).
                output_offset = (
                    cutlass.Int64(kv_head * N) * sum_seq + out_base + page_start + page_half * CTA_M
                )
                page_output = cute.make_tensor(
                    output.iterator + output_offset,
                    cute.make_layout(
                        (CTA_M, N, 1),
                        stride=(
                            1,
                            sum_seq,
                            N * sum_seq,
                        ),
                    ),
                )
                gC_mnl = cute.local_tile(page_output, self.epi_tile, (None, None, None))
                tCgC = thr_mma.partition_C(gC_mnl)
                epilogue_tidx = tidx % EPILOGUE_THREADS
                tiled_copy_t2r, tTR_tAcc, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                    epilogue_tidx, tCtAcc, tCgC, self.epi_tile, False
                )
                simt_atom, tTR_rC, tTR_gC = self.epilog_gmem_copy_and_partition(
                    epilogue_tidx, tiled_copy_t2r, tCgC, self.epi_tile, None
                )
                tTR_gC = tTR_gC[(None, None, None, None, None, 0, 0, 0)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                if tidx < EPILOGUE_THREADS:
                    for subtile_idx in cutlass.range_constexpr(cute.size(tTR_tAcc.shape, mode=[3])):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, subtile_idx)],
                            tTR_rAcc,
                        )
                        tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
                        # The window start is a per-request runtime value, so
                        # the tile-interior fast path is a dynamic predicate;
                        # only the straddling first tile takes the per-token
                        # branch.
                        if cutlass.dynamic_expr(
                            page_start >= score_start and page_start + CTA_M <= self.seq_len
                        ):
                            cute.copy(
                                simt_atom,
                                tTR_rC,
                                tTR_gC[(None, None, None, subtile_idx)],
                            )
                        else:
                            output_token = page_start + epilogue_tidx
                            if cutlass.dynamic_expr(
                                output_token >= score_start and output_token < self.seq_len
                            ):
                                cute.copy(
                                    simt_atom,
                                    tTR_rC,
                                    tTR_gC[(None, None, None, subtile_idx)],
                                )
                        if cutlass.const_expr(self.write_partial_stats):
                            stats_output = cute.coalesce(tTR_rC)
                            stats_head_base = subtile_idx * cute.size(stats_output)
                            for stats_value in cutlass.range_constexpr(cute.size(stats_output)):
                                stats_head = stats_head_base + stats_value
                                stats_page_scores_m128[stats_head] = cutlass.Float32(
                                    stats_output[stats_value]
                                )
                                if pages_processed == 0:
                                    if cutlass.const_expr(page_half == 0):
                                        if tidx == 0:
                                            sStats[stats_head] = cutlass.Float32(
                                                stats_output[stats_value]
                                            )
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                if cutlass.const_expr(self.write_partial_stats):
                    if tidx < EPILOGUE_THREADS:
                        stats_epilogue_barrier.wait_unaligned()
                else:
                    cute.arch.barrier()
                if cutlass.const_expr(not self.write_partial_stats):
                    raw_tma_pipeline.consumer_release(raw_tma_consumer_state)
                    raw_tma_consumer_state.advance()
                if cutlass.const_expr(self.write_partial_stats):
                    if pages_processed == 0:
                        if cutlass.const_expr(page_half == 0):
                            for stats_head in cutlass.range_constexpr(N):
                                stats_origins_m128[stats_head] = sStats[stats_head]
                    stats_token = page_start + tidx
                    if tidx < EPILOGUE_THREADS:
                        if cutlass.dynamic_expr(
                            stats_token >= score_start and stats_token < valid_seq_len
                        ):
                            for stats_head in cutlass.range_constexpr(N):
                                stats_delta = (
                                    stats_page_scores_m128[stats_head]
                                    - stats_origins_m128[stats_head]
                                )
                                stats_sums_m128[stats_head] = (
                                    stats_sums_m128[stats_head] + stats_delta
                                )
                                stats_square_sums_m128[stats_head] = (
                                    stats_square_sums_m128[stats_head] + stats_delta * stats_delta
                                )
            page_index += self.page_shards
            page_start += self.tile_tokens * self.page_shards
            pages_processed += 1
        if warp_idx == self.producer_warp_id:
            raw_tma_pipeline.producer_tail(raw_tma_producer_state)
            acc_pipeline.producer_tail(acc_producer_state)
        if cutlass.const_expr(self.write_partial_stats):
            for stats_head in cutlass.range_constexpr(N):
                stats_sum = stats_sums_m128[stats_head]
                stats_square_sum = stats_square_sums_m128[stats_head]
                for stats_offset in (16, 8, 4, 2, 1):
                    stats_sum = stats_sum + cute.arch.shuffle_sync_bfly(stats_sum, stats_offset)
                    stats_square_sum = stats_square_sum + cute.arch.shuffle_sync_bfly(
                        stats_square_sum, stats_offset
                    )
                if lane_idx == 0 and warp_idx < EPILOGUE_THREADS // 32:
                    stats_scratch_base = 8 + (warp_idx * N + stats_head) * 2
                    sStats[stats_scratch_base] = stats_sum
                    sStats[stats_scratch_base + 1] = stats_square_sum
            cute.arch.barrier()
            if warp_idx == 0:
                # Padded head columns (GQA group below the MMA tile N=8)
                # carry zero scores; only the real heads' statistics are
                # merged and written, in the compact row layout the union
                # finalizer reads (row = segment * num_q_heads + q_head).
                if lane_idx < self.group_size:
                    stats_sum = cutlass.Float32(0.0)
                    stats_square_sum = cutlass.Float32(0.0)
                    for stats_warp in cutlass.range_constexpr(EPILOGUE_THREADS // 32):
                        stats_scratch_base = 8 + (stats_warp * N + lane_idx) * 2
                        stats_sum = stats_sum + sStats[stats_scratch_base]
                        stats_square_sum = stats_square_sum + sStats[stats_scratch_base + 1]
                    stats_count_i32 = pages_processed * self.tile_tokens
                    if pages_processed > 0:
                        stats_invalid_prefix = score_start - shard_first_page_start
                        if cutlass.dynamic_expr(stats_invalid_prefix > 0):
                            stats_count_i32 = stats_count_i32 - stats_invalid_prefix
                        stats_last_page = page_start - self.tile_tokens * self.page_shards
                        stats_invalid_tail = stats_last_page + self.tile_tokens - valid_seq_len
                        if cutlass.dynamic_expr(stats_invalid_tail > 0):
                            stats_count_i32 = stats_count_i32 - stats_invalid_tail
                    stats_count = cutlass.Float32(stats_count_i32)
                    stats_mean = cutlass.Float32(0.0)
                    stats_m2 = cutlass.Float32(0.0)
                    if cutlass.dynamic_expr(stats_count_i32 > 0):
                        inverse_count = cutlass.Float32(1.0) / stats_count
                        stats_origin = sStats[lane_idx]
                        stats_mean = stats_origin + stats_sum * inverse_count
                        stats_m2 = cute.arch.fmax(
                            stats_square_sum - stats_sum * stats_sum * inverse_count,
                            cutlass.Float32(0.0),
                        )
                    stats_row = task * self.group_size + lane_idx
                    stats_base = (stats_row * self.page_shards + page_shard) * 3
                    partial_stats[stats_base] = stats_count
                    partial_stats[stats_base + 1] = stats_mean
                    partial_stats[stats_base + 2] = stats_m2
        cute.arch.barrier()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=False)


_COMPILED_KERNELS: dict[tuple, object] = {}
_COMPILE_LOCK = threading.Lock()


def _encode_tma_descriptors(
    layer_pools: list[torch.Tensor],
    layer_indices: list[int],
    num_freqs: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Encode one immutable feature-first TensorMap per layer index."""
    anchor = layer_pools[layer_indices[0]]
    active_layers = set(layer_indices)
    uint32 = cuda.cuuint32_t
    uint64 = cuda.cuuint64_t
    descriptor_rows = []
    for layer, maybe_pool in enumerate(layer_pools):
        pool = maybe_pool if layer in active_layers else anchor
        if pool.dtype != torch.bfloat16:
            raise TypeError("TriAttention CuTe score requires BF16 layer pools")
        if tuple(pool.shape[1:]) != tuple(anchor.shape[1:]):
            raise ValueError("TriAttention CuTe score requires uniform scored-layer pool geometry")
        _, kv_factor, num_kv_heads, pool_tokens, head_dim = pool.shape
        if (kv_factor, pool_tokens, head_dim) != (2, tokens_per_block, 2 * num_freqs):
            raise ValueError(
                f"TriAttention CuTe score requires [page, 2, Hkv, {tokens_per_block}, "
                f"{2 * num_freqs}] pools"
            )
        s_page, _, s_kv_head, s_token, s_dim = map(int, pool.stride())
        if s_dim != 1:
            raise ValueError("TriAttention CuTe score requires contiguous K features")

        global_dims = [2 * num_freqs, tokens_per_block]
        global_strides_bytes = [s_token * pool.element_size()]
        if num_kv_heads > 1:
            global_dims.append(int(num_kv_heads))
            global_strides_bytes.append(s_kv_head * pool.element_size())
        if pool.shape[0] > 1:
            global_dims.append(int(pool.shape[0]))
            global_strides_bytes.append(s_page * pool.element_size())
        tensor_rank = len(global_dims)
        # One TMA box covers one coefficient plane of one page fragment
        # (the whole page for the validated 128-token geometry).
        box_dims = [num_freqs, min(CTA_M, tokens_per_block)] + [1] * (tensor_rank - 2)
        status, tensor_map = cuda.cuTensorMapEncodeTiled(
            cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            uint32(tensor_rank),
            pool.data_ptr(),
            [uint64(value) for value in global_dims],
            [uint64(value) for value in global_strides_bytes],
            [uint32(value) for value in box_dims],
            [uint32(1) for _ in range(tensor_rank)],
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            # The swizzle must match the smem layout the TMA lands in; the
            # sm100 helpers pick it from the inner-row byte count (one
            # coefficient plane: num_freqs bf16 elements).
            (
                cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B
                if num_freqs * 2 == 64
                else cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
            ),
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed for layer {layer}: {status}")
        descriptor_rows.append(
            [
                value if value < 1 << 63 else value - (1 << 64)
                for value in map(int, tensor_map.opaque)
            ]
        )

    descriptors = torch.tensor(
        descriptor_rows,
        dtype=torch.int64,
        device=anchor.device,
    )
    if descriptors.shape != (len(layer_pools), TMA_DESCRIPTOR_QWORDS):
        raise AssertionError("each TriAttention TMA descriptor must occupy 128 bytes")
    if descriptors.data_ptr() % 128 or descriptors.stride(0) != TMA_DESCRIPTOR_QWORDS:
        raise AssertionError("TriAttention TMA descriptor rows must be 128-byte aligned")
    return descriptors


def _tensor_spec(tensor: torch.Tensor) -> tuple:
    return (
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _to_cute(tensor: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=assumed_align)


class TriAttentionCuteScoreRunner:
    """Compile and launch the exact SM100 mean-score specialization."""

    def __init__(
        self,
        *,
        layer_pools: list[torch.Tensor],
        layer_indices: list[int],
        max_requests: int,
        num_layers: int,
        seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_freqs: int,
        tokens_per_block: int,
        page_ids: torch.Tensor,
        seg_page_off: torch.Tensor,
        seg_req_id: torch.Tensor,
        seg_layer_id: torch.Tensor,
        seg_seq_len: torch.Tensor,
        seg_out_offset: torch.Tensor,
        token_starts: torch.Tensor,
        q_real: torch.Tensor,
        q_imag: torch.Tensor,
        mlr_coef: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        output: torch.Tensor,
        enable_partial_stats: bool = False,
        small_workload_page_shards: int = 3,
    ) -> None:
        self.max_requests = int(max_requests)
        self.num_layers = int(num_layers)
        # The score window start is a per-request runtime input
        # (``token_starts``), so the widest window — the whole bucket —
        # sizes every start-dependent buffer.
        self.width = int(seq_len)
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.sm_count = int(torch.cuda.get_device_properties(output.device).multi_processor_count)
        self.enable_partial_stats = bool(enable_partial_stats)
        if small_workload_page_shards not in _SUPPORTED_PAGE_SHARDS:
            raise ValueError("TriAttention CuTe score has unsupported page shards")
        self.small_workload_page_shards = int(small_workload_page_shards)
        partial_stats_elements = (
            max_requests * num_layers * num_q_heads * self.small_workload_page_shards * 3
            if self.enable_partial_stats
            else 1
        )
        self.partial_stats = torch.empty(
            partial_stats_elements,
            dtype=torch.float32,
            device=output.device,
        )
        self.descriptors = _encode_tma_descriptors(
            layer_pools, layer_indices, int(num_freqs), int(tokens_per_block)
        )
        self._torch_prefix = (
            page_ids,
            seg_page_off,
            seg_req_id,
            seg_layer_id,
            seg_seq_len,
            seg_out_offset,
            token_starts,
            q_real,
            q_imag,
            mlr_coef,
        )
        self._torch_tail = (
            freq_scale_sq,
            output,
            self.partial_stats,
            layer_pools[layer_indices[0]],
            self.descriptors,
        )
        self._cute_prefix = tuple(_to_cute(tensor) for tensor in self._torch_prefix)
        self._cute_tail = (
            _to_cute(freq_scale_sq),
            _to_cute(output),
            _to_cute(self.partial_stats),
            _to_cute(layer_pools[layer_indices[0]]),
            _to_cute(self.descriptors, assumed_align=128),
        )
        self._compiled: dict[int, object] = {}
        self._compiled_stats: dict[int, object] = {}
        self._compiled_normalize_union: dict[int, object] = {}
        self._page_shards: dict[int, int] = {}
        compile_output_rows = max_requests if self.enable_partial_stats else 1
        self._normalize_union_compile_output = torch.empty(
            (compile_output_rows, self.width),
            dtype=torch.float32,
            device=output.device,
        )
        self._cute_selection_prefix = (
            _to_cute(output),
            _to_cute(seg_seq_len),
            _to_cute(seg_out_offset),
            _to_cute(token_starts),
        )
        static_geometry = (
            max_requests,
            num_layers,
            seq_len,
            num_q_heads,
            num_kv_heads,
            num_freqs,
            tokens_per_block,
            tuple(int(value) for value in layer_pools[layer_indices[0]].shape),
            tuple(int(value) for value in layer_pools[layer_indices[0]].stride()),
        )
        tensor_specs = tuple(
            _tensor_spec(tensor)
            for tensor in (
                *self._torch_prefix,
                mean_cos.view(-1),
                mean_sin.view(-1),
                *self._torch_tail,
            )
        )
        variants = [(1, self.small_workload_page_shards)]
        if max_requests > 1:
            variants.append((max_requests, 2))
        for request_count, page_shards in variants:
            cache_key = (
                "triattention_cute_score",
                static_geometry,
                tensor_specs,
                request_count,
                page_shards,
            )
            with _COMPILE_LOCK:
                compiled = _COMPILED_KERNELS.get(cache_key)
                if compiled is None:
                    kernel = _TriAttentionScoreKernel(
                        num_layers=num_layers,
                        seq_len=seq_len,
                        num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        num_freqs=num_freqs,
                        tokens_per_block=tokens_per_block,
                        pool_shape=tuple(
                            int(value) for value in layer_pools[layer_indices[0]].shape
                        ),
                        pool_strides=tuple(
                            int(value) for value in layer_pools[layer_indices[0]].stride()
                        ),
                        pool_dtype=cutlass.BFloat16,
                        page_shards=page_shards,
                    )
                    stream = cuda.CUstream(torch.cuda.current_stream(output.device).cuda_stream)
                    compiled = cute.compile(
                        kernel,
                        *self._cute_prefix,
                        _to_cute(mean_cos.view(-1)),
                        _to_cute(mean_sin.view(-1)),
                        *self._cute_tail,
                        cutlass.Int32(1),
                        stream,
                    )
                    _COMPILED_KERNELS[cache_key] = compiled
            self._compiled[request_count] = compiled
            self._page_shards[request_count] = page_shards
            if self.enable_partial_stats:
                stats_cache_key = (
                    "triattention_cute_score_stats",
                    static_geometry,
                    tensor_specs,
                    request_count,
                    page_shards,
                )
                with _COMPILE_LOCK:
                    compiled_stats = _COMPILED_KERNELS.get(stats_cache_key)
                    if compiled_stats is None:
                        stats_kernel = _TriAttentionScoreKernel(
                            num_layers=num_layers,
                            seq_len=seq_len,
                            num_q_heads=num_q_heads,
                            num_kv_heads=num_kv_heads,
                            num_freqs=num_freqs,
                            tokens_per_block=tokens_per_block,
                            pool_shape=tuple(
                                int(value) for value in layer_pools[layer_indices[0]].shape
                            ),
                            pool_strides=tuple(
                                int(value) for value in layer_pools[layer_indices[0]].stride()
                            ),
                            pool_dtype=cutlass.BFloat16,
                            page_shards=page_shards,
                            write_partial_stats=True,
                        )
                        stream = cuda.CUstream(torch.cuda.current_stream(output.device).cuda_stream)
                        compiled_stats = cute.compile(
                            stats_kernel,
                            *self._cute_prefix,
                            _to_cute(mean_cos.view(-1)),
                            _to_cute(mean_sin.view(-1)),
                            *self._cute_tail,
                            cutlass.Int32(1),
                            stream,
                        )
                        _COMPILED_KERNELS[stats_cache_key] = compiled_stats
                self._compiled_stats[request_count] = compiled_stats

        if max_requests > 1:
            small_score = self._compiled[1]
            large_score = self._compiled[max_requests]
            small_stats = self._compiled_stats.get(1)
            large_stats = self._compiled_stats.get(max_requests)
            for request_count in range(1, max_requests + 1):
                two_shard_ctas = request_count * num_layers * num_kv_heads * 2
                use_extra_score_shard = two_shard_ctas < 2 * self.sm_count
                self._compiled[request_count] = (
                    small_score if use_extra_score_shard else large_score
                )
                self._page_shards[request_count] = (
                    self.small_workload_page_shards if use_extra_score_shard else 2
                )
                if self.enable_partial_stats:
                    self._compiled_stats[request_count] = (
                        small_stats if use_extra_score_shard else large_stats
                    )

        if self.enable_partial_stats:
            from .triattention_cute_selection import (
                _select_normalize_union_config,
                _TriAttentionNormalizeUnionKernel,
            )

            compiled_configs: dict[tuple[int, int, int, int], object] = {}
            for request_count in range(1, max_requests + 1):
                page_shards = self._page_shards[request_count]
                config = _select_normalize_union_config(
                    request_count,
                    self.width,
                    self.sm_count,
                )
                config_key = (page_shards, *config)
                compiled_selection = compiled_configs.get(config_key)
                if compiled_selection is None:
                    cache_key = (
                        "triattention_cute_normalize_union",
                        static_geometry,
                        tensor_specs,
                        config_key,
                        _tensor_spec(self._normalize_union_compile_output),
                        _tensor_spec(self.partial_stats),
                    )
                    with _COMPILE_LOCK:
                        compiled_selection = _COMPILED_KERNELS.get(cache_key)
                        if compiled_selection is None:
                            tokens_per_lane, token_subtiles, row_cluster_ctas = config
                            kernel = _TriAttentionNormalizeUnionKernel(
                                num_layers=num_layers,
                                seq_len=seq_len,
                                num_q_heads=num_q_heads,
                                # The score scratch pads each KV head's group
                                # of head planes to the MMA tile N=8; the
                                # finalizer maps real head rows onto those
                                # padded planes (identity for GQA group 8).
                                num_kv_heads=num_kv_heads,
                                page_shards=page_shards,
                                tokens_per_lane=tokens_per_lane,
                                token_subtiles=token_subtiles,
                                row_cluster_ctas=row_cluster_ctas,
                            )
                            stream = cuda.CUstream(
                                torch.cuda.current_stream(output.device).cuda_stream
                            )
                            compiled_selection = cute.compile(
                                kernel,
                                _to_cute(self.partial_stats),
                                *self._cute_selection_prefix,
                                _to_cute(self._normalize_union_compile_output.view(-1)),
                                cutlass.Int32(1),
                                stream,
                            )
                            _COMPILED_KERNELS[cache_key] = compiled_selection
                    compiled_configs[config_key] = compiled_selection
                self._compiled_normalize_union[request_count] = compiled_selection

    def supports(self, request_count: int) -> bool:
        """Return whether the dynamic specialization covers this request count."""
        return request_count in self._compiled

    def launch(
        self,
        request_count: int,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
    ) -> None:
        """Launch the CuTe score kernel on the current PyTorch stream."""
        stream = cuda.CUstream(torch.cuda.current_stream(mean_cos.device).cuda_stream)
        self._compiled[request_count](
            *self._cute_prefix,
            _to_cute(mean_cos.view(-1)),
            _to_cute(mean_sin.view(-1)),
            *self._cute_tail,
            request_count,
            stream,
        )

    def supports_union_fusion(self, request_count: int) -> bool:
        """Return whether the score/stats/union pipeline was precompiled."""
        return (
            request_count in self._compiled_stats
            and request_count in self._compiled_normalize_union
        )

    def launch_union_fusion(
        self,
        request_count: int,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        union_scores: torch.Tensor,
    ) -> None:
        """Launch score plus stats followed by normalized union reduction."""
        stream = cuda.CUstream(torch.cuda.current_stream(mean_cos.device).cuda_stream)
        self._compiled_stats[request_count](
            *self._cute_prefix,
            _to_cute(mean_cos.view(-1)),
            _to_cute(mean_sin.view(-1)),
            *self._cute_tail,
            request_count,
            stream,
        )
        self._launch_union_finalize(request_count, union_scores, stream)

    def _launch_union_finalize(
        self,
        request_count: int,
        union_scores: torch.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        if (
            union_scores.shape != (request_count, self.width)
            or union_scores.dtype != torch.float32
            or union_scores.device != self.partial_stats.device
            or not union_scores.is_contiguous()
        ):
            raise ValueError("TriAttention union output does not match the compiled geometry")
        self._compiled_normalize_union[request_count](
            _to_cute(self.partial_stats),
            *self._cute_selection_prefix,
            _to_cute(union_scores.view(-1)),
            request_count,
            stream,
        )
