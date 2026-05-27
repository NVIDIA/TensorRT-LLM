# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL FP4 MLA decode backend.

This module intentionally preserves the Python FP4 MLA decode contract from
``fp4_mla_kv.run_fp4_mla_attention_decode``.  It consumes the same packed Q,
packed paged KV cache, swizzled scale tensors, page tables, and workspace
buffers as the Triton backend.
"""

import math

import torch

from ...logger import logger
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda

    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import T, dsl_user_op

    class _CUDAGraphCompatibleWrapper:
        """Wrapper to make DLPack export safe during CUDA graph capture."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor

        def __dlpack__(self, stream=None):
            return self._tensor.__dlpack__(stream=-1)

        def __dlpack_device__(self):
            return self._tensor.__dlpack_device__()

    def _to_cute(tensor: torch.Tensor) -> cute.Tensor:
        return from_dlpack(
            _CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=16
        ).mark_layout_dynamic()

    @cute.jit
    def _swizzled_sf_offset(row_idx, col_idx, sf_per_token: cutlass.Constexpr):
        padded_cols = ((sf_per_token + 3) // 4) * 4
        return (
            col_idx % 4
            + (col_idx // 4) * (4 * 128)
            + (row_idx % 32) * 16
            + ((row_idx % 128) // 32) * 4
            + (row_idx // 128) * (128 * padded_cols)
        )

    @dsl_user_op
    def _ptx_fp4_e2m1x2_to_f16x2(byte, *, loc=None, ip=None) -> cutlass.Uint32:
        return cutlass.Uint32(
            llvm.inline_asm(
                T.i32(),
                [cutlass.Uint32(byte).ir_value(loc=loc, ip=ip)],
                """
                {
                    .reg .b8 in_8;
                    .reg .f16x2 out;
                    cvt.u8.u32 in_8, $1;
                    cvt.rn.f16x2.e2m1x2 out, in_8;
                    mov.b32 $0, out;
                }
                """,
                "=r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _ptx_fp8_e4m3x2_to_f16x2(byte, *, loc=None, ip=None) -> cutlass.Uint32:
        return cutlass.Uint32(
            llvm.inline_asm(
                T.i32(),
                [cutlass.Uint32(byte).ir_value(loc=loc, ip=ip)],
                """
                {
                    .reg .b16 in_16;
                    .reg .f16x2 out;
                    cvt.u16.u32 in_16, $1;
                    cvt.rn.f16x2.e4m3x2 out, in_16;
                    mov.b32 $0, out;
                }
                """,
                "=r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _ptx_fp4_e2m1x2_from_f32(even, odd, *, loc=None, ip=None) -> cutlass.Uint32:
        return cutlass.Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    cutlass.Float32(odd).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(even).ir_value(loc=loc, ip=ip),
                ],
                """
                {
                    .reg .b8 out;
                    cvt.rn.satfinite.e2m1x2.f32 out, $1, $2;
                    mov.b32 $0, {out, out, out, out};
                }
                """,
                "=r,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _ptx_fp8_e4m3x2_from_f32(low, high, *, loc=None, ip=None) -> cutlass.Uint32:
        return cutlass.Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    cutlass.Float32(low).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(high).ir_value(loc=loc, ip=ip),
                ],
                """
                {
                    .reg .b16 out;
                    cvt.rn.satfinite.e4m3x2.f32 out, $2, $1;
                    mov.b32 $0, {out, out};
                }
                """,
                "=r,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def _low_f16x2_lane_to_f32(bits):
        half_bits = cutlass.Uint16(bits & cutlass.Uint32(0xFFFF))
        half_value = cutlass.Float16(llvm.bitcast(cutlass.Float16.mlir_type, half_bits.ir_value()))
        return half_value.to(cutlass.Float32)

    @cute.jit
    def _high_f16x2_lane_to_f32(bits):
        half_bits = cutlass.Uint16((bits >> 16) & cutlass.Uint32(0xFFFF))
        half_value = cutlass.Float16(llvm.bitcast(cutlass.Float16.mlir_type, half_bits.ir_value()))
        return half_value.to(cutlass.Float32)

    @cute.jit
    def _fp4_e2m1_to_f32(nibble):
        bits = _ptx_fp4_e2m1x2_to_f16x2(cute.Uint8(nibble & cute.Uint8(0x0F)))
        return _low_f16x2_lane_to_f32(bits)

    @cute.jit
    def _fp4_e2m1_quantize_packed(even, odd):
        bits = _ptx_fp4_e2m1x2_from_f32(even, odd)
        return cute.Uint8(bits & cutlass.Uint32(0xFF))

    @cute.jit
    def _load_fp4_value(packed_tensor, packed_offset, elem_idx):
        packed = packed_tensor[packed_offset]
        nibble = cute.Uint8(packed & 0x0F)
        if (elem_idx & 1) != 0:
            nibble = cute.Uint8((packed >> 4) & 0x0F)
        return _fp4_e2m1_to_f32(nibble)

    @cute.jit
    def _load_fp4_byte_pair(packed_tensor, packed_offset):
        """Load one packed FP4 byte and return both nibbles as (low, high) f32.

        The PTX ``cvt.f16x2.e2m1x2`` instruction converts both nibbles in a
        single op; the previous scalar path discarded the high half.
        """
        packed = packed_tensor[packed_offset]
        bits = _ptx_fp4_e2m1x2_to_f16x2(cute.Uint8(packed))
        return _low_f16x2_lane_to_f32(bits), _high_f16x2_lane_to_f32(bits)

    @cute.jit
    def _fp8_e4m3fn_to_f32(byte):
        bits = _ptx_fp8_e4m3x2_to_f16x2(cute.Uint8(byte))
        return _low_f16x2_lane_to_f32(bits)

    @cute.jit
    def _fp8_e4m3fn_positive_from_f32(value):
        bits = _ptx_fp8_e4m3x2_from_f32(value, cutlass.Float32(0.0))
        return cute.Uint8(bits & cutlass.Uint32(0xFF))

    class _Fp4MlaDecodeCuteKernel:
        def __init__(
            self,
            *,
            num_heads: int,
            kv_lora_rank: int,
            qk_rope_head_dim: int,
            q_residual_dim: int,
            page_size: int,
            max_pages: int,
            q_fp4_stride0: int,
            q_fp4_stride1: int,
            kv_stride0: int,
            kv_stride2: int,
            kv_stride4: int,
            sf_stride0: int,
            v_sf_stride0: int,
            p_stride0: int,
            p_stride1: int,
            page_stats_stride0: int,
            page_stats_stride1: int,
            page_stats_stride2: int,
            output_dtype: torch.dtype,
        ) -> None:
            self.num_heads = num_heads
            self.kv_lora_rank = kv_lora_rank
            self.qk_rope_head_dim = qk_rope_head_dim
            self.q_residual_dim = q_residual_dim
            self.k_head_dim = kv_lora_rank + qk_rope_head_dim
            self.q_head_dim = self.k_head_dim + q_residual_dim
            self.page_size = page_size
            self.max_pages = max_pages
            self.q_fp4_stride0 = q_fp4_stride0
            self.q_fp4_stride1 = q_fp4_stride1
            self.kv_stride0 = kv_stride0
            self.kv_stride2 = kv_stride2
            self.kv_stride4 = kv_stride4
            self.sf_stride0 = sf_stride0
            self.v_sf_stride0 = v_sf_stride0
            self.p_stride0 = p_stride0
            self.p_stride1 = p_stride1
            self.page_stats_stride0 = page_stats_stride0
            self.page_stats_stride1 = page_stats_stride1
            self.page_stats_stride2 = page_stats_stride2
            self.output_dtype = output_dtype
            self.fp4_block = 16
            self.k_sf_per_token = self.k_head_dim // self.fp4_block
            self.q_sf_per_token = self.q_head_dim // self.fp4_block
            self.p_sf_per_page = page_size // self.fp4_block
            self.non_residual_groups = self.k_sf_per_token - q_residual_dim // self.fp4_block
            self.log2_e = math.log2(math.e)
            self.p_global_scale = 448.0 * 6.0
            self.head_tile = 128
            self.v_tile = 128

        @cute.jit
        def __call__(
            self,
            output,
            max_scores,
            denom,
            page_max,
            page_sum,
            p_fp4,
            p_sf,
            q_fp4,
            q_sf,
            kv_cache,
            sf_cache,
            v_sf,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            sm_scale: cutlass.Float32,
            stream: cuda.CUstream,
        ) -> None:
            num_head_blocks = cute.ceil_div(self.num_heads, self.head_tile)
            self._page_stats_pack_kernel(
                page_max,
                page_sum,
                p_fp4,
                p_sf,
                q_fp4,
                q_sf,
                kv_cache,
                sf_cache,
                global_scale,
                src_page_ids,
                paged_kv_indptr_decode,
                kv_lens,
                sm_scale,
            ).launch(
                grid=(output.shape[0], num_head_blocks, self.max_pages),
                block=(self.head_tile, 1, 1),
                stream=stream,
            )

            self._reduce_stats_kernel(
                max_scores,
                denom,
                page_max,
                page_sum,
            ).launch(
                grid=(output.shape[0], num_head_blocks, 1),
                block=(self.head_tile, 1, 1),
                stream=stream,
            )

            self._prob_scale_kernel(
                max_scores,
                denom,
                page_max,
                p_sf,
                paged_kv_indptr_decode,
                kv_lens,
            ).launch(
                grid=(output.shape[0], num_head_blocks, self.max_pages),
                block=(self.head_tile, 1, 1),
                stream=stream,
            )

            self._pv_kernel(
                output,
                p_fp4,
                p_sf,
                kv_cache,
                v_sf,
                global_scale,
                src_page_ids,
                paged_kv_indptr_decode,
                kv_lens,
            ).launch(
                grid=(
                    output.shape[0],
                    self.num_heads,
                    cute.ceil_div(self.kv_lora_rank, self.v_tile),
                ),
                block=(self.v_tile, 1, 1),
                stream=stream,
            )

        @cute.jit
        def _qk_score(
            self,
            q_fp4,
            q_sf,
            kv_cache,
            sf_cache,
            global_scale,
            src_page_ids,
            compact_page,
            token_idx,
            q_row,
            sm_scale,
        ):
            physical_page = src_page_ids[compact_page]
            score = cutlass.Float32(0.0)
            bytes_per_group: cutlass.Constexpr = self.fp4_block // 2
            q_row_base = q_row * self.q_fp4_stride0
            kv_token_base = physical_page * self.kv_stride0 + token_idx * self.kv_stride2
            k_sf_page_base = physical_page * self.sf_stride0

            # Keep q_group as a runtime loop (44 iters): unrolling it together
            # with the inner byte-pair loop and the outer 128-token loop blows
            # up compile time and instruction footprint.
            for q_group in cutlass.range(self.q_sf_per_token, unroll=1):
                k_group = q_group
                if q_group >= self.non_residual_groups:
                    k_group = self.non_residual_groups + (q_group - self.non_residual_groups) // 2

                # Scales only change at fp4_block boundaries — hoist out of the
                # inner byte-pair loop instead of reloading per element.
                q_scale = _fp8_e4m3fn_to_f32(
                    q_sf[_swizzled_sf_offset(q_row, q_group, self.q_sf_per_token)]
                )
                k_scale = _fp8_e4m3fn_to_f32(
                    sf_cache[
                        k_sf_page_base
                        + _swizzled_sf_offset(token_idx, k_group, self.k_sf_per_token)
                    ]
                )
                qk_scale = q_scale * k_scale

                # Each FP4 byte holds 2 nibbles; process them together so the
                # single cvt.f16x2.e2m1x2 produces 2 useful f32 values.
                for byte_idx in cutlass.range_constexpr(bytes_per_group):
                    q_packed_col = q_group * bytes_per_group + byte_idx
                    k_packed_col = k_group * bytes_per_group + byte_idx
                    q_lo, q_hi = _load_fp4_byte_pair(
                        q_fp4,
                        q_row_base + q_packed_col * self.q_fp4_stride1,
                    )
                    k_lo, k_hi = _load_fp4_byte_pair(
                        kv_cache,
                        kv_token_base + k_packed_col * self.kv_stride4,
                    )
                    score += (q_lo * k_lo + q_hi * k_hi) * qk_scale

            scale = sm_scale / (global_scale[0] * global_scale[0])
            return score * scale

        @cute.jit
        def _page_stats_offset(self, gen_idx, page_rel, head_idx):
            return (
                gen_idx * self.page_stats_stride0
                + page_rel * self.page_stats_stride1
                + head_idx * self.page_stats_stride2
            )

        @cute.kernel
        def _page_stats_pack_kernel(
            self,
            page_max,
            page_sum,
            p_fp4,
            p_sf,
            q_fp4,
            q_sf,
            kv_cache,
            sf_cache,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            sm_scale: cutlass.Float32,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            gen_idx, head_block, page_rel = cute.arch.block_idx()
            head_idx = head_block * self.head_tile + tidx
            max_score = -cutlass.Float32.inf
            sum_value = cutlass.Float32(0.0)

            if head_idx < self.num_heads:
                q_row = gen_idx * self.num_heads + head_idx
                kv_len = kv_lens[gen_idx]
                page_start = page_rel * self.page_size
                if page_start < kv_len:
                    page_table_start = paged_kv_indptr_decode[gen_idx]
                    compact_page = page_table_start + page_rel
                    scores = cute.make_fragment((self.page_size,), cutlass.Float32)
                    for token_offset in cutlass.range_constexpr(self.page_size):
                        token_abs = page_start + token_offset
                        score = -cutlass.Float32.inf
                        if token_abs < kv_len:
                            score = self._qk_score(
                                q_fp4,
                                q_sf,
                                kv_cache,
                                sf_cache,
                                global_scale,
                                src_page_ids,
                                compact_page,
                                token_offset,
                                q_row,
                                sm_scale,
                            )
                            if score > max_score:
                                max_score = score
                        scores[token_offset] = score

                    p_row = compact_page * self.num_heads + head_idx
                    for token_group in cutlass.range_constexpr(self.p_sf_per_page):
                        local_max = cutlass.Float32(0.0)
                        probs = cute.make_fragment((self.fp4_block,), cutlass.Float32)
                        for idx in cutlass.range_constexpr(self.fp4_block):
                            token_offset = token_group * self.fp4_block + idx
                            token_abs = page_start + token_offset
                            prob = cutlass.Float32(0.0)
                            if token_abs < kv_len:
                                prob = cute.math.exp2(
                                    (scores[token_offset] - max_score) * self.log2_e,
                                    fastmath=True,
                                )
                                sum_value += prob
                            probs[idx] = prob
                            if prob > local_max:
                                local_max = prob

                        local_scale = cutlass.Float32(1.0)
                        stored_scale = cutlass.Float32(1.0)
                        if local_max > cutlass.Float32(0.0):
                            local_scale = local_max / cutlass.Float32(6.0)
                            stored_scale = local_scale * self.p_global_scale
                            if stored_scale > cutlass.Float32(448.0):
                                stored_scale = cutlass.Float32(448.0)

                        p_sf[
                            _swizzled_sf_offset(
                                p_row,
                                token_group,
                                self.p_sf_per_page,
                            )
                        ] = _fp8_e4m3fn_positive_from_f32(stored_scale)

                        for byte_idx in cutlass.range_constexpr(self.fp4_block // 2):
                            packed = _fp4_e2m1_quantize_packed(
                                probs[byte_idx * 2] / local_scale,
                                probs[byte_idx * 2 + 1] / local_scale,
                            )
                            p_fp4[
                                p_row * self.p_stride0
                                + (token_group * (self.fp4_block // 2) + byte_idx) * self.p_stride1
                            ] = packed

                stats_offset = self._page_stats_offset(gen_idx, page_rel, head_idx)
                page_max[stats_offset] = max_score
                page_sum[stats_offset] = sum_value

        @cute.kernel
        def _reduce_stats_kernel(self, max_scores, denom, page_max, page_sum) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            gen_idx, head_block, _ = cute.arch.block_idx()
            head_idx = head_block * self.head_tile + tidx
            if head_idx < self.num_heads:
                max_score = -cutlass.Float32.inf
                for page_rel in cutlass.range(self.max_pages, unroll=1):
                    stats_offset = self._page_stats_offset(gen_idx, page_rel, head_idx)
                    page_max_value = page_max[stats_offset]
                    if page_max_value > max_score:
                        max_score = page_max_value

                denom_value = cutlass.Float32(0.0)
                for page_rel in cutlass.range(self.max_pages, unroll=1):
                    stats_offset = self._page_stats_offset(gen_idx, page_rel, head_idx)
                    page_sum_value = page_sum[stats_offset]
                    if page_sum_value > cutlass.Float32(0.0):
                        denom_value += page_sum_value * cute.math.exp2(
                            (page_max[stats_offset] - max_score) * self.log2_e,
                            fastmath=True,
                        )

                q_row = gen_idx * self.num_heads + head_idx
                max_scores[q_row] = max_score
                denom[q_row] = denom_value

        @cute.kernel
        def _prob_scale_kernel(
            self,
            max_scores,
            denom,
            page_max,
            p_sf,
            paged_kv_indptr_decode,
            kv_lens,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            gen_idx, head_block, page_rel = cute.arch.block_idx()
            head_idx = head_block * self.head_tile + tidx
            if head_idx < self.num_heads:
                kv_len = kv_lens[gen_idx]
                page_start = page_rel * self.page_size
                if page_start < kv_len:
                    stats_offset = self._page_stats_offset(gen_idx, page_rel, head_idx)
                    q_row = gen_idx * self.num_heads + head_idx
                    denom_value = denom[q_row]
                    factor = cutlass.Float32(0.0)
                    if denom_value > cutlass.Float32(0.0):
                        factor = (
                            cute.math.exp2(
                                (page_max[stats_offset] - max_scores[q_row]) * self.log2_e,
                                fastmath=True,
                            )
                            / denom_value
                        )

                    page_table_start = paged_kv_indptr_decode[gen_idx]
                    p_row = (page_table_start + page_rel) * self.num_heads + head_idx
                    for token_group in cutlass.range_constexpr(self.p_sf_per_page):
                        sf_offset = _swizzled_sf_offset(
                            p_row,
                            token_group,
                            self.p_sf_per_page,
                        )
                        scaled = _fp8_e4m3fn_to_f32(p_sf[sf_offset]) * factor
                        p_sf[sf_offset] = _fp8_e4m3fn_positive_from_f32(scaled)

        @cute.kernel
        def _pv_kernel(
            self,
            output,
            p_fp4,
            p_sf,
            kv_cache,
            v_sf,
            global_scale,
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            gen_idx, head_idx, dim_block = cute.arch.block_idx()
            v_dim = dim_block * self.v_tile + tidx
            if v_dim < self.kv_lora_rank:
                kv_len = kv_lens[gen_idx]
                page_table_start = paged_kv_indptr_decode[gen_idx]
                v_packed_col = v_dim // 2  # constant per thread
                acc = cutlass.Float32(0.0)
                bytes_per_group: cutlass.Constexpr = self.fp4_block // 2
                for page_rel in cutlass.range(self.max_pages, unroll=1):
                    page_start = page_rel * self.page_size
                    if page_start < kv_len:
                        compact_page = page_table_start + page_rel
                        physical_page = src_page_ids[compact_page]
                        p_row = compact_page * self.num_heads + head_idx
                        p_row_base = p_row * self.p_stride0
                        v_page_base = (
                            physical_page * self.kv_stride0 + v_packed_col * self.kv_stride4
                        )
                        v_sf_page_base = physical_page * self.v_sf_stride0
                        # Process 16 tokens at a time — both scales are
                        # constant within each fp4_block, so hoist them.
                        for token_group in cutlass.range(self.p_sf_per_page, unroll=1):
                            group_start = token_group * self.fp4_block
                            if page_start + group_start < kv_len:
                                p_scale = _fp8_e4m3fn_to_f32(
                                    p_sf[
                                        _swizzled_sf_offset(
                                            p_row,
                                            token_group,
                                            self.p_sf_per_page,
                                        )
                                    ]
                                )
                                v_scale = _fp8_e4m3fn_to_f32(
                                    v_sf[
                                        v_sf_page_base
                                        + _swizzled_sf_offset(
                                            v_dim,
                                            token_group,
                                            self.p_sf_per_page,
                                        )
                                    ]
                                )
                                pv_scale = p_scale * v_scale
                                # Each P byte holds 2 adjacent tokens; load
                                # once and use both nibbles.
                                for byte_idx in cutlass.range_constexpr(bytes_per_group):
                                    token_a = group_start + byte_idx * 2
                                    token_b = token_a + 1
                                    token_abs_a = page_start + token_a
                                    if token_abs_a < kv_len:
                                        p_packed_col = token_group * bytes_per_group + byte_idx
                                        p_lo, p_hi = _load_fp4_byte_pair(
                                            p_fp4,
                                            p_row_base + p_packed_col * self.p_stride1,
                                        )
                                        v_a = _load_fp4_value(
                                            kv_cache,
                                            v_page_base + token_a * self.kv_stride2,
                                            v_dim,
                                        )
                                        acc += p_lo * v_a * pv_scale
                                        if token_abs_a + 1 < kv_len:
                                            v_b = _load_fp4_value(
                                                kv_cache,
                                                v_page_base + token_b * self.kv_stride2,
                                                v_dim,
                                            )
                                            acc += p_hi * v_b * pv_scale

                output[gen_idx, head_idx, v_dim] = (
                    acc / (global_scale[0] * self.p_global_scale)
                ).to(output.element_type)

    _COMPILE_CACHE: dict[tuple[int, ...], object] = {}

    def _storage_span(tensor: torch.Tensor) -> int:
        if tensor.numel() == 0:
            return 0
        return 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))

    def _flatten(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_contiguous():
            return tensor.reshape(-1)
        return torch.as_strided(
            tensor,
            size=(_storage_span(tensor),),
            stride=(1,),
            storage_offset=tensor.storage_offset(),
        )

    def run_fp4_mla_attention_decode_cute(
        *,
        output: torch.Tensor,
        max_scores: torch.Tensor,
        denom: torch.Tensor,
        page_max: torch.Tensor,
        page_sum: torch.Tensor,
        p_fp4: torch.Tensor,
        p_sf: torch.Tensor,
        q_fp4: torch.Tensor,
        q_sf: torch.Tensor,
        kv_cache: torch.Tensor,
        sf_cache: torch.Tensor,
        v_sf: torch.Tensor,
        global_scale: torch.Tensor,
        src_page_ids: torch.Tensor,
        paged_kv_indptr_decode: torch.Tensor,
        kv_lens: torch.Tensor,
        sm_scale: float,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        q_residual_dim: int,
        page_size: int,
        max_pages: int,
    ) -> None:
        """Run FP4 MLA decode using the page-parallel CuTe DSL backend."""

        q_fp4_flat = _flatten(q_fp4.view(torch.uint8))
        q_sf_flat = _flatten(q_sf.view(torch.uint8))
        kv_cache_flat = _flatten(kv_cache.view(torch.uint8))
        sf_cache_flat = _flatten(sf_cache.view(torch.uint8))
        v_sf_flat = _flatten(v_sf.view(torch.uint8))
        p_fp4_flat = _flatten(p_fp4.view(torch.uint8))
        p_sf_flat = _flatten(p_sf.view(torch.uint8))
        max_scores_flat = _flatten(max_scores)
        denom_flat = _flatten(denom)
        page_max_flat = _flatten(page_max)
        page_sum_flat = _flatten(page_sum)

        stream = cuda.CUstream(torch.cuda.current_stream(output.device).cuda_stream)
        cute_args = (
            _to_cute(output),
            _to_cute(max_scores_flat),
            _to_cute(denom_flat),
            _to_cute(page_max_flat),
            _to_cute(page_sum_flat),
            _to_cute(p_fp4_flat),
            _to_cute(p_sf_flat),
            _to_cute(q_fp4_flat),
            _to_cute(q_sf_flat),
            _to_cute(kv_cache_flat),
            _to_cute(sf_cache_flat),
            _to_cute(v_sf_flat),
            _to_cute(global_scale),
            _to_cute(src_page_ids),
            _to_cute(paged_kv_indptr_decode),
            _to_cute(kv_lens),
            cutlass.Float32(sm_scale),
            stream,
        )
        compile_key = (
            output.device.index or 0,
            output.dtype is torch.bfloat16,
            output.shape[0],
            output.shape[1],
            kv_lora_rank,
            qk_rope_head_dim,
            q_residual_dim,
            page_size,
            max_pages,
            q_fp4.stride(0),
            q_fp4.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
            sf_cache.stride(0),
            v_sf.stride(0),
            p_fp4.stride(0),
            p_fp4.stride(1),
            page_max.stride(0),
            page_max.stride(1),
            page_max.stride(2),
        )
        if compile_key not in _COMPILE_CACHE:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "CuTe FP4 MLA decode must be compiled before CUDA graph capture."
                )
            logger.info(
                "Compiling CuTe FP4 MLA decode kernel for "
                f"num_gen={output.shape[0]}, num_heads={output.shape[1]}, "
                f"kv_lora_rank={kv_lora_rank}, rope_dim={qk_rope_head_dim}, "
                f"max_pages={max_pages}"
            )
            kernel = _Fp4MlaDecodeCuteKernel(
                num_heads=output.shape[1],
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                q_residual_dim=q_residual_dim,
                page_size=page_size,
                max_pages=max_pages,
                q_fp4_stride0=q_fp4.stride(0),
                q_fp4_stride1=q_fp4.stride(1),
                kv_stride0=kv_cache.stride(0),
                kv_stride2=kv_cache.stride(2),
                kv_stride4=kv_cache.stride(4),
                sf_stride0=sf_cache.stride(0),
                v_sf_stride0=v_sf.stride(0),
                p_stride0=p_fp4.stride(0),
                p_stride1=p_fp4.stride(1),
                page_stats_stride0=page_max.stride(0),
                page_stats_stride1=page_max.stride(1),
                page_stats_stride2=page_max.stride(2),
                output_dtype=output.dtype,
            )
            _COMPILE_CACHE[compile_key] = cute.compile(kernel, *cute_args)

        _COMPILE_CACHE[compile_key](*cute_args)

else:

    def run_fp4_mla_attention_decode_cute(**_: object) -> None:
        raise RuntimeError("CuTe DSL is not available in this environment.")
