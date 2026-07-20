# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 CuTe-DSL scorer for the TriAttention mean-score path.

This is the production specialization of the final workbench kernel. It
uses split real/imag TMA loads, BF16 and FP16 compensated UMMA, sqrt FTZ, and
producer-only page-ID lookahead. The public integration keeps the compiled
C++ score ops as the implementation for every geometry outside the exact
contract validated here.

Page-table contract: ``page_ids`` is the flattened native block-offset
staging buffer ([pool_slot, request, K/V plane, block] int32) shared with
the C++ score op; K-plane entries encode ``physical_page * kv_factor`` and
are decoded inline (kv_factor == 2), so no per-round conversion pass is
needed.
"""

from __future__ import annotations

import inspect
import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack


def _cute_sqrt_keyword_mode() -> str:
    """Probe which fast-sqrt spelling this CuTe DSL's ``cute.math.sqrt`` takes.

    The approximate-sqrt control was renamed across DSL releases: some expose
    ``approx``/``ftz`` keywords, cutlass 4.5 exposes a single ``fastmath``
    flag, and older releases expose a plain one-argument ``sqrt``. Passing an
    unknown keyword raises TypeError at trace time (inside ``cute.compile``,
    where it cannot be caught), so the capability is probed once at import
    time via signature inspection and folded into a trace-time constant.
    """
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

CTA_M = 64
K = 96
N = 8
NUM_FREQS = 32
THREADS = 128

PAGE_TOKENS = 128
RAW_K_HALF_ELEMENTS = CTA_M * 2 * NUM_FREQS
RAW_K_VECTOR_ELEMENTS = 8
RAW_K_SPLIT_PHASE_ELEMENTS = CTA_M * NUM_FREQS
RAW_K_SPLIT_TMA_COPY_BYTES = RAW_K_SPLIT_PHASE_ELEMENTS * (cutlass.BFloat16.width // 8)
TMA_DESCRIPTOR_QWORDS = 16


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
        num_segments: int,
        seq_len: int,
        score_start: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_freqs: int,
        tokens_per_block: int,
        pool_shape: tuple[int, int, int, int, int],
        pool_strides: tuple[int, int, int, int, int],
        pool_dtype: type[cutlass.Numeric],
        page_shards: int,
    ) -> None:
        """Build the single validated production specialization."""
        super().__init__()
        if pool_dtype is not cutlass.BFloat16:
            raise ValueError("TriAttention CuTe score requires BF16 K pages")
        if num_freqs != NUM_FREQS:
            raise ValueError("TriAttention CuTe score requires 32 frequencies")
        if tokens_per_block != PAGE_TOKENS:
            raise ValueError("TriAttention CuTe score requires 128-token pages")
        if num_q_heads % num_kv_heads or num_q_heads // num_kv_heads != N:
            raise ValueError("TriAttention CuTe score requires GQA group 8")
        if score_start % PAGE_TOKENS:
            raise ValueError("TriAttention CuTe score requires page-aligned score_start")
        if page_shards not in (2, 3):
            raise ValueError("TriAttention CuTe score requires two or three page shards")

        self.score_start = score_start
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.sum_seq = num_segments * seq_len
        self.num_tasks = num_segments * num_kv_heads
        self.page_shards = page_shards
        self.num_ctas = self.num_tasks * page_shards
        self.max_pages = (seq_len + PAGE_TOKENS - 1) // PAGE_TOKENS
        self.halves_per_page = PAGE_TOKENS // CTA_M

        # Measured final choices that still shape layouts or generated code.
        self.prefetch_depth = 4
        self.sqrt_mode = "approx"
        self.k_staging_mode = "half_page_tma"
        self.use_tma = True
        self.cpasync_schedule = "sync_each_half"
        self.split_raw_tma = True
        self.raw_tma_feature_extent = NUM_FREQS
        self.raw_tma_copy_bytes = RAW_K_SPLIT_TMA_COPY_BYTES
        self.raw_tma_pipeline_stages = 1
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
        if pool_kv_heads != num_kv_heads or pool_tokens != PAGE_TOKENS or pool_dim != 2 * NUM_FREQS:
            raise ValueError("K pool shape does not match the CuTe score specialization")
        self.s_page, _, self.s_kv_head, self.s_slot, self.s_dim = pool_strides
        if self.s_slot != 2 * NUM_FREQS or self.s_dim != 1:
            raise ValueError("K pages must be contiguous [128, 64]")
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
        q_real: cute.Tensor,
        q_imag: cute.Tensor,
        mlr_coef: cute.Tensor,
        mean_cos: cute.Tensor,
        mean_sin: cute.Tensor,
        freq_scale_sq: cute.Tensor,
        output: cute.Tensor,
        pool_template: cute.Tensor,
        raw_tma_descriptors: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.c_dtype = output.element_type
        self.c_layout = utils.LayoutEnum.COL_MAJOR
        self.mma_tiler = (CTA_M, N, K)
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
            (CTA_M, N, NUM_FREQS)
            if self.main_operand_mode == "bf16_raw_three_term_weight"
            else self.mma_tiler
        )
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, main_a_shape, cutlass.Float32, 1)
        raw_bf16_a_smem_layout = sm100_utils.make_smem_layout_a(
            raw_bf16_tiled_mma,
            (CTA_M, N, 2 * NUM_FREQS),
            cutlass.BFloat16,
            1,
        )
        raw_bf16_split_a_smem_layout = sm100_utils.make_smem_layout_a(
            raw_bf16_tiled_mma,
            (CTA_M, N, NUM_FREQS),
            cutlass.BFloat16,
            2,
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
                (self.raw_tma_feature_extent, CTA_M),
                stride=(1, self.raw_tma_feature_extent),
            ),
        )
        raw_tma_source_layout = cute.make_layout(
            (
                2 * NUM_FREQS,
                PAGE_TOKENS,
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
            (self.raw_tma_feature_extent, CTA_M),
        )
        raw_bf16_b_smem_layout = sm100_utils.make_smem_layout_b(
            raw_bf16_tiled_mma,
            (CTA_M, N, 2 * NUM_FREQS),
            cutlass.BFloat16,
            1,
        )
        # The magnitude residual has only K=32.  A separate compact descriptor
        # lets the producer issue its four UMMA steps before the first commit,
        # rather than waiting for and overwriting the K=96 main A tile.
        magnitude_lo_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, (CTA_M, N, NUM_FREQS), cutlass.Float32, 1
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
            (CTA_M, N, NUM_FREQS),
            cutlass.Float16,
            1,
        )
        magnitude_hi_fp16_smem_layout = sm100_utils.make_smem_layout_b(
            magnitude_lo_tiled_mma,
            (CTA_M, N, NUM_FREQS),
            cutlass.Float16,
            1,
        )
        magnitude_fp16_a_smem_layout = sm100_utils.make_smem_layout_a(
            magnitude_lo_tiled_mma,
            (CTA_M, N, NUM_FREQS),
            cutlass.Float16,
            1,
        )
        magnitude_fp16_b_smem_layout = sm100_utils.make_smem_layout_b(
            magnitude_lo_tiled_mma,
            (CTA_M, N, NUM_FREQS),
            cutlass.Float16,
            1,
        )
        main_b_shape = (
            (CTA_M, N, NUM_FREQS)
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
        raw_k_elements = RAW_K_HALF_ELEMENTS * int(
            self.k_staging_mode in ("half_page_cpasync", "half_page_tma")
            and not self.shared_a_raw_alias
        )
        alias_a_elements = cute.cosize(a_smem_layout.outer) * int(self.shared_a_raw_alias)
        alias_raw_k_elements = RAW_K_HALF_ELEMENTS * int(self.shared_a_raw_alias)
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
            sARawAlias: SharedARawAlias

        self.shared_storage = SharedStorage
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
            q_real,
            q_imag,
            mlr_coef,
            mean_cos,
            mean_sin,
            freq_scale_sq,
            output,
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
            grid=(self.num_ctas, 1, 1),
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
        q_real: cute.Tensor,
        q_imag: cute.Tensor,
        mlr_coef: cute.Tensor,
        mean_cos: cute.Tensor,
        mean_sin: cute.Tensor,
        freq_scale_sq: cute.Tensor,
        output: cute.Tensor,
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
        raw_k_storage = storage.sRawK
        cpasync_raw_k_0 = raw_k_storage.get_tensor(
            raw_bf16_direct_a_smem_layout.outer,
            swizzle=raw_bf16_direct_a_smem_layout.inner,
        )
        cpasync_raw_k_real = cpasync_raw_k_0[(None, None, None, 0)]
        cpasync_raw_k_imag = cpasync_raw_k_0[(None, None, None, 1)]
        # Each stage slice retains the K_SW64 pointer flags.  Reuse only
        # the feature-first outer mapping for the corresponding TMA
        # destination so the swizzle is not applied twice.
        raw_tma_shared_real = cute.make_tensor(
            cpasync_raw_k_real.iterator,
            raw_tma_smem_layout.outer,
        )
        raw_tma_shared_imag = cute.make_tensor(
            cpasync_raw_k_imag.iterator,
            raw_tma_smem_layout.outer,
        )
        raw_tma_source_tiles = cute.local_tile(
            raw_tma_source,
            (self.raw_tma_feature_extent, CTA_M),
            coord=(None, None, None),
        )
        raw_tma_shared_partition_real, raw_tma_global_partition = cpasync.tma_partition(
            raw_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(raw_tma_shared_real, 0, 2),
            cute.group_modes(raw_tma_source_tiles, 0, 2),
        )
        raw_tma_shared_partition_imag, _ = cpasync.tma_partition(
            raw_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(raw_tma_shared_imag, 0, 2),
            cute.group_modes(raw_tma_source_tiles, 0, 2),
        )
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

        page_index = self.score_start // PAGE_TOKENS + page_shard
        page_start = page_index * PAGE_TOKENS
        pages_processed = cutlass.Int32(0)
        producer_prefetched_page_id_lane0 = cutlass.Int32(0)
        if warp_idx == self.producer_warp_id:
            if lane_idx == 0:
                # ``page_ids`` is the flattened native block-offset staging
                # buffer ([pool_slot, request, K/V plane, block] int32) and
                # ``page_off`` points at one request's K plane. K-plane
                # entries encode ``physical_page * kv_factor`` (kv_factor is
                # 2 for the interleaved K/V pools this kernel requires); the
                # C++ score op decodes the same buffer with
                # ``encoded / kvFactor`` (triAttentionScoreKernels.cu), so
                # divide by two here as well. V-plane entries are never read.
                producer_prefetched_page_id_lane0 = (
                    cutlass.Int32(page_ids[page_off + page_index]) // 2
                )
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
        cute.arch.mbarrier_init_fence()
        for weight_round in cutlass.range_constexpr(N * K // THREADS):
            linear_index = tidx + weight_round * THREADS
            qg = linear_index // K
            feature = linear_index % K
            coefficient_kind = feature // NUM_FREQS
            frequency = feature % NUM_FREQS
            mean_offset = req_id * NUM_FREQS + frequency
            q_head = kv_head * self.group_size + qg
            calib_offset = (layer_id * self.num_q_heads + q_head) * NUM_FREQS + frequency
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

        thr_mma = tiled_mma.get_slice(0)
        while page_start < valid_seq_len and pages_processed < self.max_pages:
            physical_page = cutlass.Int32(0)
            if warp_idx == self.producer_warp_id:
                physical_page = cute.arch.shuffle_sync(
                    producer_prefetched_page_id_lane0,
                    0,
                )
            for page_half in cutlass.range_constexpr(self.halves_per_page):
                if warp_idx == self.producer_warp_id:
                    # Phase 0 fills the packed 4-KiB K_SW64 real
                    # view. Every producer-warp lane participates
                    # in the PipelineTmaAsync barrier election.
                    raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                    cute.copy(
                        raw_tma_atom,
                        raw_tma_global_partition[
                            (
                                None,
                                0,
                                page_half,
                                (kv_head, physical_page),
                            )
                        ],
                        raw_tma_shared_partition_real,
                        tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(raw_tma_producer_state),
                        tma_desc_ptr=raw_tma_descriptor_ptr,
                    )
                    raw_tma_producer_state.advance()
                raw_tma_pipeline.consumer_wait(raw_tma_consumer_state)
                raw_tma_pipeline.consumer_release(raw_tma_consumer_state)
                raw_tma_consumer_state.advance()
                if warp_idx == self.producer_warp_id:
                    raw_tma_pipeline.producer_acquire(raw_tma_producer_state)
                    cute.copy(
                        raw_tma_atom,
                        raw_tma_global_partition[
                            (
                                None,
                                1,
                                page_half,
                                (kv_head, physical_page),
                            )
                        ],
                        raw_tma_shared_partition_imag,
                        tma_bar_ptr=raw_tma_pipeline.producer_get_barrier(raw_tma_producer_state),
                        tma_desc_ptr=raw_tma_descriptor_ptr,
                    )
                    raw_tma_producer_state.advance()

                if cutlass.const_expr(self.producer_page_id_prefetch and page_half == 1):
                    next_page_id_lane0 = cutlass.Int32(0)
                    if warp_idx == self.producer_warp_id:
                        if lane_idx == 0:
                            next_page_start = page_start + PAGE_TOKENS * self.page_shards
                            next_pages_processed = pages_processed + 1
                            if (
                                next_page_start < valid_seq_len
                                and next_pages_processed < self.max_pages
                            ):
                                # Same K-plane decode as the initial
                                # prefetch: entries are physical_page * 2.
                                next_page_id_lane0 = (
                                    cutlass.Int32(
                                        page_ids[page_off + page_index + self.page_shards]
                                    )
                                    // 2
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
                    for raw_k_block in cutlass.range_constexpr(NUM_FREQS // 16):
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, 0)],
                            tCrRawBf16B0[(None, None, raw_k_block, 0)],
                            tCtAcc,
                        )
                        raw_bf16_tiled_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            True,
                        )
                raw_tma_pipeline.consumer_wait(raw_tma_consumer_state)
                frequency = lane_idx
                # Issue several independent token loads before consuming
                # any of them.  This bounded RMEM window is unchanged; the
                # optional half-page staging only switches its K source
                # from global to the single raw shared buffer.
                for token_base in cutlass.range(
                    0,
                    CTA_M // 4,
                    self.prefetch_depth,
                    unroll_full=not self.compact_token_loop,
                ):
                    staged_real = cute.make_rmem_tensor((self.prefetch_depth,), cutlass.Float32)
                    staged_imag = cute.make_rmem_tensor((self.prefetch_depth,), cutlass.Float32)
                    for prefetch_index in cutlass.range_constexpr(self.prefetch_depth):
                        token_round = token_base + prefetch_index
                        token = warp_idx + token_round * 4
                        staged_real[prefetch_index] = cutlass.Float32(
                            cpasync_raw_k_0[
                                (
                                    (token, frequency % 16),
                                    0,
                                    frequency // 16,
                                    0,
                                )
                            ]
                        )
                        staged_imag[prefetch_index] = cutlass.Float32(
                            cpasync_raw_k_0[
                                (
                                    (token, frequency % 16),
                                    0,
                                    frequency // 16,
                                    1,
                                )
                            ]
                        )

                    for prefetch_index in cutlass.range_constexpr(self.prefetch_depth):
                        token_round = token_base + prefetch_index
                        token = warp_idx + token_round * 4
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
                            # cutlass 4.5 renamed the approximate-sqrt control
                            # to ``fastmath``; map the measured approx choice
                            # onto it to preserve the authored behavior.
                            magnitude = cute.math.sqrt(norm2, fastmath=self.sqrt_mode == "approx")
                        else:
                            # DSLs with neither spelling get the plain (IEEE)
                            # sqrt, which is strictly MORE accurate than the
                            # measured approx choice above; the unit test's
                            # 5e-3 oracle tolerance absorbs the difference.
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
                    for raw_k_block in cutlass.range_constexpr(NUM_FREQS // 16):
                        imag_b_block = NUM_FREQS // 16 + raw_k_block
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, 1)],
                            tCrRawBf16B0[(None, None, imag_b_block, 0)],
                            tCtAcc,
                        )
                    for raw_k_block in cutlass.range_constexpr(NUM_FREQS // 16):
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, 0)],
                            tCrRawBf16B1[(None, None, raw_k_block, 0)],
                            tCtAcc,
                        )
                    for raw_k_block in cutlass.range_constexpr(NUM_FREQS // 16):
                        imag_b_block = NUM_FREQS // 16 + raw_k_block
                        cute.gemm(
                            raw_bf16_tiled_mma,
                            tCtAcc,
                            tCrRawBf16ASplit[(None, None, raw_k_block, 1)],
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
                    for magnitude_k_block in cutlass.range_constexpr(NUM_FREQS // 16):
                        cute.gemm(
                            magnitude_lo_tiled_mma,
                            tCtAcc,
                            tCrMagnitudeFp16A0[(None, None, magnitude_k_block, 0)],
                            tCrMagnitudeFp16B0[(None, None, magnitude_k_block, 0)],
                            tCtAcc,
                        )
                    for magnitude_k_block in cutlass.range_constexpr(NUM_FREQS // 16):
                        cute.gemm(
                            magnitude_lo_tiled_mma,
                            tCtAcc,
                            tCrMagnitudeFp16A0[(None, None, magnitude_k_block, 0)],
                            tCrMagnitudeFp16B1[(None, None, magnitude_k_block, 0)],
                            tCtAcc,
                        )
                    for magnitude_k_block in cutlass.range_constexpr(NUM_FREQS // 16):
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
                output_offset = (
                    kv_head * self.group_size * self.sum_seq
                    + out_base
                    + page_start
                    + page_half * CTA_M
                )
                page_output = cute.make_tensor(
                    output.iterator + output_offset,
                    cute.make_layout(
                        (CTA_M, N, 1),
                        stride=(
                            1,
                            self.sum_seq,
                            self.group_size * self.sum_seq,
                        ),
                    ),
                )
                gC_mnl = cute.local_tile(page_output, self.epi_tile, (None, None, None))
                tCgC = thr_mma.partition_C(gC_mnl)
                tiled_copy_t2r, tTR_tAcc, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                    tidx, tCtAcc, tCgC, self.epi_tile, False
                )
                simt_atom, tTR_rC, tTR_gC = self.epilog_gmem_copy_and_partition(
                    tidx, tiled_copy_t2r, tCgC, self.epi_tile, None
                )
                tTR_gC = tTR_gC[(None, None, None, None, None, 0, 0, 0)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                for subtile_idx in range(cute.size(tTR_tAcc.shape, mode=[3])):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, subtile_idx)],
                        tTR_rAcc,
                    )
                    tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
                    cute.copy(
                        simt_atom,
                        tTR_rC,
                        tTR_gC[(None, None, None, subtile_idx)],
                    )

                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                cute.arch.barrier()
                raw_tma_pipeline.consumer_release(raw_tma_consumer_state)
                raw_tma_consumer_state.advance()

            page_index += self.page_shards
            page_start += PAGE_TOKENS * self.page_shards
            pages_processed += 1
        if warp_idx == self.producer_warp_id:
            raw_tma_pipeline.producer_tail(raw_tma_producer_state)
        if warp_idx == self.producer_warp_id:
            acc_pipeline.producer_tail(acc_producer_state)
        cute.arch.barrier()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=False)


_COMPILED_KERNELS: dict[tuple, object] = {}
_COMPILE_LOCK = threading.Lock()


def _encode_tma_descriptors(
    layer_pools: list[torch.Tensor],
    layer_indices: list[int],
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
        _, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape
        if (kv_factor, tokens_per_block, head_dim) != (2, PAGE_TOKENS, 2 * NUM_FREQS):
            raise ValueError("TriAttention CuTe score requires [page, 2, Hkv, 128, 64] pools")
        s_page, _, s_kv_head, s_token, s_dim = map(int, pool.stride())
        if s_dim != 1:
            raise ValueError("TriAttention CuTe score requires contiguous K features")

        global_dims = [2 * NUM_FREQS, PAGE_TOKENS]
        global_strides_bytes = [s_token * pool.element_size()]
        if num_kv_heads > 1:
            global_dims.append(int(num_kv_heads))
            global_strides_bytes.append(s_kv_head * pool.element_size())
        if pool.shape[0] > 1:
            global_dims.append(int(pool.shape[0]))
            global_strides_bytes.append(s_page * pool.element_size())
        tensor_rank = len(global_dims)
        box_dims = [NUM_FREQS, CTA_M] + [1] * (tensor_rank - 2)
        status, tensor_map = cuda.cuTensorMapEncodeTiled(
            cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            uint32(tensor_rank),
            pool.data_ptr(),
            [uint64(value) for value in global_dims],
            [uint64(value) for value in global_strides_bytes],
            [uint32(value) for value in box_dims],
            [uint32(1) for _ in range(tensor_rank)],
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
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
        score_start: int,
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
        q_real: torch.Tensor,
        q_imag: torch.Tensor,
        mlr_coef: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self.max_requests = int(max_requests)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.descriptors = _encode_tma_descriptors(layer_pools, layer_indices)
        self._torch_prefix = (
            page_ids,
            seg_page_off,
            seg_req_id,
            seg_layer_id,
            seg_seq_len,
            seg_out_offset,
            q_real,
            q_imag,
            mlr_coef,
        )
        self._torch_tail = (
            freq_scale_sq,
            output,
            layer_pools[layer_indices[0]],
            self.descriptors,
        )
        self._cute_prefix = tuple(_to_cute(tensor) for tensor in self._torch_prefix)
        self._cute_tail = (
            _to_cute(freq_scale_sq),
            _to_cute(output),
            _to_cute(layer_pools[layer_indices[0]]),
            _to_cute(self.descriptors, assumed_align=128),
        )
        self._compiled: dict[int, object] = {}
        static_geometry = (
            max_requests * num_layers,
            seq_len,
            score_start,
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
        variants = [(1, 3)]
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
                        num_segments=request_count * num_layers,
                        seq_len=seq_len,
                        score_start=score_start,
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
                        stream,
                    )
                    _COMPILED_KERNELS[cache_key] = compiled
            self._compiled[request_count] = compiled

    def supports(self, request_count: int) -> bool:
        """Return whether an exact static specialization was precompiled."""
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
            stream,
        )
