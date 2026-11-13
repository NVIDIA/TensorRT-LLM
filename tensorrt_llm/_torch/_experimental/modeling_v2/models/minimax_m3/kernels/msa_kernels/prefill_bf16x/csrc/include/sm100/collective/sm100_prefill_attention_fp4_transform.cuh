// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cute/arch/config.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "sm100/common/nvfp4_to_e4m3.cuh"
#include "sm100/common/prefill_attention_config.cuh"
#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
#include "sm100/common/nvfp4_to_bf16.cuh"
#endif

namespace fmha_sm100::prefill_bf16x::detail {

struct Fp4DequantInput {
  uint32_t packed_fp4;
  uint32_t scale_word;
};

CUTLASS_DEVICE Fp4DequantInput load_fp4_dequant_input(
    uint8_t const* packed_fp4, uint8_t scale_e4m3) {
  if constexpr (kE4M3X) {
    return {*reinterpret_cast<uint32_t const*>(packed_fp4),
            ::fmha_sm100::prefill_bf16x::sm100::common::e4m3x_scale_f16x2(scale_e4m3)};
  }
  return {
      *reinterpret_cast<uint32_t const*>(packed_fp4),
      ::fmha_sm100::prefill_bf16x::sm100::common::make_dequant_scale_word(scale_e4m3),
  };
}

CUTLASS_DEVICE void dequant_store_fp8_smem(
    Fp4DequantInput const& input, uint32_t output_address, float global_scale = 1.0f) {
  uint32_t output_lo;
  uint32_t output_hi;
  if constexpr (kE4M3X) {
    ::fmha_sm100::prefill_bf16x::sm100::common::dequant_e2m1x8_scale_global_e4m3x(
        output_lo, output_hi, input.packed_fp4, input.scale_word, global_scale);
  } else {
    ::fmha_sm100::prefill_bf16x::sm100::common::dequant_e2m1x8_uniform_scale(
        output_lo, output_hi, input.packed_fp4, input.scale_word);
  }
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
               :
               : "r"(output_address), "r"(output_lo), "r"(output_hi));
}

// Scales are linear (byte token * 8 + group, the K layout) or, with TokenQuadScales, in the
// token-quad order (byte (token / 4) * 32 + group * 4 + token % 4) that vLLM writes for V.
//
// Rows [valid_rows, NumRows) are past the request's KV length and are dequantized with a zero
// scale, so they come out as exact zeros. For V this is required: those tokens have P = 0, but the
// PV MMA still multiplies them, and the cache bytes there are unspecified (vLLM reuses blocks
// across cache groups of different formats), so a NaN block scale (0x7F / 0xFF) would give
// 0 x NaN = NaN.
template <int NumRows, int HeadDim, int ScaleGroupSize, bool TokenQuadScales>
CUTLASS_DEVICE void dequant_fp4_tile_to_fp8_smem(
    uint8_t const* packed_fp4, uint8_t const* scale,
    uint8_t* output_smem, int thread_idx, int valid_rows = NumRows, float global_scale = 1.0f) {
  static_assert(NumRows == 128);
  static_assert(HeadDim == 128);
  static_assert(ScaleGroupSize == 16);

  constexpr int kNumThreads = 128;
  constexpr int kElementsPerThread = 8;
  constexpr int kThreadsPerRow = HeadDim / kElementsPerThread;
  constexpr int kRowsPerIteration = kNumThreads / kThreadsPerRow;
  constexpr int kRowIterations = NumRows / kRowsPerIteration;
  constexpr int kDataRowStride = HeadDim / 2;
  constexpr int kScaleRowStride = HeadDim / ScaleGroupSize;
  constexpr int kDataIterationStride =
      kDataRowStride * kRowsPerIteration;
  constexpr int kScaleIterationStride =
      kScaleRowStride * kRowsPerIteration;
  constexpr int kSwizzleAtomBytes = 8 * 128;
  constexpr int kLinearRowStep =
      (kRowsPerIteration / 8) * kSwizzleAtomBytes;

  static_assert(kThreadsPerRow == 16);
  static_assert(kRowsPerIteration == 8);
  static_assert(kRowIterations == 16);
  // Eight rows are two whole token quads, so both scale orders advance 64 bytes per iteration.
  static_assert(kScaleIterationStride == (kRowsPerIteration / 4) * 32);

  int const row_in_iteration = thread_idx / kThreadsPerRow;
  int const lane_in_row = thread_idx % kThreadsPerRow;
  int const d_base = lane_in_row * kElementsPerThread;
  int const fp4_byte_base = d_base / 2;
  int const scale_group = d_base / ScaleGroupSize;

  int const initial_linear =
      d_base + (row_in_iteration % 8) * 128 +
      (row_in_iteration / 8) * kSwizzleAtomBytes;
  int const xor_offset = (initial_linear & 0x380) >> 3;
  uint32_t output_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(output_smem)) +
      (initial_linear ^ xor_offset);

  uint8_t const* current_data =
      packed_fp4 + row_in_iteration * kDataRowStride + fp4_byte_base;
  int const scale_offset =
      TokenQuadScales
          ? (row_in_iteration / 4) * 32 + scale_group * 4 + row_in_iteration % 4
          : row_in_iteration * kScaleRowStride + scale_group;
  uint8_t const* current_scale = scale + scale_offset;
  // Scale byte of the row `iterations_ahead` iterations past the current one, 0 past valid_rows.
  int row = row_in_iteration;
  auto row_scale = [&](int iterations_ahead) -> uint8_t {
    return row + iterations_ahead * kRowsPerIteration < valid_rows
               ? current_scale[iterations_ahead * kScaleIterationStride]
               : uint8_t{0};
  };

  Fp4DequantInput input_a = load_fp4_dequant_input(current_data, row_scale(0));
  Fp4DequantInput input_b = load_fp4_dequant_input(
      current_data + kDataIterationStride, row_scale(1));
  Fp4DequantInput input_c = load_fp4_dequant_input(
      current_data + 2 * kDataIterationStride, row_scale(2));

  CUTLASS_PRAGMA_UNROLL
  for (int iteration = 0; iteration < kRowIterations - 3; ++iteration) {
    Fp4DequantInput input_d = load_fp4_dequant_input(
        current_data + 3 * kDataIterationStride, row_scale(3));
    dequant_store_fp8_smem(input_a, output_address, global_scale);
    output_address += kLinearRowStep;
    current_data += kDataIterationStride;
    current_scale += kScaleIterationStride;
    row += kRowsPerIteration;
    input_a = input_b;
    input_b = input_c;
    input_c = input_d;
  }

  dequant_store_fp8_smem(input_a, output_address, global_scale);
  dequant_store_fp8_smem(
      input_b, output_address + kLinearRowStep, global_scale);
  dequant_store_fp8_smem(
      input_c, output_address + 2 * kLinearRowStep, global_scale);
}

template <int NumRows, int HeadDim>
CUTLASS_DEVICE void clear_fp8_tile_smem(
    uint8_t* output_smem, int thread_idx) {
  constexpr int kNumThreads = 128;
  constexpr int kElementsPerThread = 8;
  constexpr int kThreadsPerRow = HeadDim / kElementsPerThread;
  constexpr int kRowsPerIteration = kNumThreads / kThreadsPerRow;
  constexpr int kRowIterations = NumRows / kRowsPerIteration;
  constexpr int kSwizzleAtomBytes = 8 * 128;
  constexpr int kLinearRowStep =
      (kRowsPerIteration / 8) * kSwizzleAtomBytes;

  int const row_in_iteration = thread_idx / kThreadsPerRow;
  int const lane_in_row = thread_idx % kThreadsPerRow;
  int const d_base = lane_in_row * kElementsPerThread;
  int const initial_linear =
      d_base + (row_in_iteration % 8) * 128 +
      (row_in_iteration / 8) * kSwizzleAtomBytes;
  int const xor_offset = (initial_linear & 0x380) >> 3;
  uint32_t output_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(output_smem)) +
      (initial_linear ^ xor_offset);
  uint32_t constexpr kZero = 0;

  CUTLASS_PRAGMA_UNROLL
  for (int iteration = 0; iteration < kRowIterations; ++iteration) {
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                 :
                 : "r"(output_address), "r"(kZero), "r"(kZero));
    output_address += kLinearRowStep;
  }
  cutlass::arch::fence_view_async_shared();
}

#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1

// BF16 data path: SMEM byte offset (before the SW128 XOR) of element (row, d) of a 128 x 128 BF16
// tile, row = token. Both tiles are built from 1 KB SW128 atoms of 8 rows x 128 bytes:
//  * K (QK B operand, K-major, Layout_K_SW128_Atom<bf16> = 8 tokens x 64 dims): atoms run over
//    the 16 token groups first, then over the two 64-dim halves, so dims [64, 128) start 16 KB in.
//  * V (PV B operand, MN-major, Layout_MN_SW128_Atom<bf16> = 64 dims x 8 tokens): atoms run over
//    the two 64-dim halves first, then over the 16 token groups (2 KB per group of 8 tokens).
// This is tile_to_mma_shape's order (M/N mode first, then K); the launch static_asserts these
// formulas against the cute layouts. The SW128 XOR (Swizzle<3,4,3> on byte addresses) swaps the
// 16-byte chunks of a 128-byte row by the row's index within its atom, bits [7, 10).
template <bool MnMajor>
CUTLASS_HOST_DEVICE constexpr int bf16_tile_linear_byte_offset(int row, int d) {
  return MnMajor ? (d / 64) * 1024 + (row / 8) * 2048 + (row % 8) * 128 + (d % 64) * 2
                 : (d / 64) * (128 * 128) + (row / 8) * 1024 + (row % 8) * 128 + (d % 64) * 2;
}

CUTLASS_HOST_DEVICE constexpr int sw128_swizzle_byte_offset(int linear) {
  return linear ^ ((linear & 0x380) >> 3);
}

// NVFP4 -> BF16 tile dequant for the BF16 data path. Same thread mapping as the E4M3 path: 16
// threads per row, 8 consecutive dims (one 32-bit packed word, one block scale) per thread, 8
// rows per iteration; each thread writes one 16-byte chunk per row with st.shared.v4.
//  * MnMajor = false: K tile (K-major); MnMajor = true: V tile (MN-major).
//  * TokenQuadScales: V scale order, as in dequant_fp4_tile_to_fp8_smem.
//  * global_scale_bf16x2: folded into every block scale with mul.rn.bf16x2 when
//    FoldGlobalScale (K); unused otherwise (the V global scale is applied by the combine).
//  * Rows [valid_rows, NumRows) get a zero scale before the multiply, so they come out as exact
//    zeros (see dequant_fp4_tile_to_fp8_smem for why V needs this).
template <int NumRows, int HeadDim, int ScaleGroupSize, bool TokenQuadScales, bool MnMajor,
          bool FoldGlobalScale>
CUTLASS_DEVICE void dequant_fp4_tile_to_bf16_smem(uint8_t const* packed_fp4, uint8_t const* scale,
                                                  uint8_t* output_smem, int thread_idx,
                                                  uint32_t global_scale_bf16x2,
                                                  int valid_rows = NumRows) {
  namespace common = ::fmha_sm100::prefill_bf16x::sm100::common;
  static_assert(NumRows == 128);
  static_assert(HeadDim == 128);
  static_assert(ScaleGroupSize == 16);

  constexpr int kNumThreads = 128;
  constexpr int kElementsPerThread = 8;
  constexpr int kThreadsPerRow = HeadDim / kElementsPerThread;
  constexpr int kRowsPerIteration = kNumThreads / kThreadsPerRow;
  constexpr int kRowIterations = NumRows / kRowsPerIteration;
  constexpr int kDataRowStride = HeadDim / 2;
  constexpr int kScaleRowStride = HeadDim / ScaleGroupSize;
  constexpr int kDataIterationStride = kDataRowStride * kRowsPerIteration;
  constexpr int kScaleIterationStride = kScaleRowStride * kRowsPerIteration;
  // One iteration advances 8 rows = one atom row group.
  constexpr int kOutputIterationStride = bf16_tile_linear_byte_offset<MnMajor>(kRowsPerIteration, 0);

  static_assert(kThreadsPerRow == 16);
  static_assert(kRowsPerIteration == 8);
  static_assert(kRowIterations == 16);
  static_assert(kScaleIterationStride == (kRowsPerIteration / 4) * 32);
  // The XOR only reads bits [7, 10) (the row within the atom), which the per-iteration stride
  // (a multiple of 1 KB) does not touch, so the swizzled start address can be advanced linearly.
  static_assert(kOutputIterationStride % 1024 == 0);

  int const row_in_iteration = thread_idx / kThreadsPerRow;
  int const lane_in_row = thread_idx % kThreadsPerRow;
  int const d_base = lane_in_row * kElementsPerThread;
  int const fp4_byte_base = d_base / 2;
  int const scale_group = d_base / ScaleGroupSize;

  uint32_t output_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(output_smem)) +
      static_cast<uint32_t>(
          sw128_swizzle_byte_offset(bf16_tile_linear_byte_offset<MnMajor>(row_in_iteration, d_base)));

  uint8_t const* current_data = packed_fp4 + row_in_iteration * kDataRowStride + fp4_byte_base;
  int const scale_offset =
      TokenQuadScales ? (row_in_iteration / 4) * 32 + scale_group * 4 + row_in_iteration % 4
                      : row_in_iteration * kScaleRowStride + scale_group;
  uint8_t const* current_scale = scale + scale_offset;

  CUTLASS_PRAGMA_UNROLL
  for (int iteration = 0; iteration < kRowIterations; ++iteration) {
    int const row = row_in_iteration + iteration * kRowsPerIteration;
    uint32_t const packed = *reinterpret_cast<uint32_t const*>(current_data);
    // Zero the scale of rows past the KV length before the multiply (exact zeros out).
    uint8_t const scale_byte = row < valid_rows ? *current_scale : uint8_t{0};
    uint32_t scale_bf16x2 = common::e4m3_to_bf16x2_broadcast(scale_byte);
    if constexpr (FoldGlobalScale) {
      // combined = bf16(block scale) x bf16(global scale), one RNE rounding in BF16.
      scale_bf16x2 = common::mul_bf16x2_rn(scale_bf16x2, global_scale_bf16x2);
    }
    uint32_t out[4];
    common::dequant_e2m1x8_to_bf16x8(out, packed, scale_bf16x2);
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(output_address), "r"(out[0]), "r"(out[1]), "r"(out[2]), "r"(out[3])
                 : "memory");
    output_address += kOutputIterationStride;
    current_data += kDataIterationStride;
    current_scale += kScaleIterationStride;
  }
  // The tile is read by tcgen05.mma (async proxy).
  cutlass::arch::fence_view_async_shared();
}

// Zero fill of a 128 x 128 BF16 tile (32 KB): a linear clear, the layout is irrelevant.
template <int NumRows, int HeadDim>
CUTLASS_DEVICE void clear_bf16_tile_smem(uint8_t* output_smem, int thread_idx) {
  constexpr int kNumThreads = 128;
  constexpr int kTileBytes = NumRows * HeadDim * 2;
  constexpr int kChunkBytes = 16;
  constexpr int kIterations = kTileBytes / (kNumThreads * kChunkBytes);
  static_assert(kTileBytes % (kNumThreads * kChunkBytes) == 0);
  uint32_t output_address = static_cast<uint32_t>(__cvta_generic_to_shared(output_smem)) +
                            static_cast<uint32_t>(thread_idx * kChunkBytes);
  uint32_t constexpr kZero = 0;
  CUTLASS_PRAGMA_UNROLL
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(output_address), "r"(kZero), "r"(kZero), "r"(kZero), "r"(kZero)
                 : "memory");
    output_address += kNumThreads * kChunkBytes;
  }
  cutlass::arch::fence_view_async_shared();
}

#endif  // FMHA_SM100_PREFILL_BF16X_DTYPE == 1

}  // namespace fmha_sm100::prefill_bf16x::detail
