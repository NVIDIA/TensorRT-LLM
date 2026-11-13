// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

#if !defined(FMHA_SM100_PREFILL_BF16X_HAS_QMUL4)
#error "The BF16X prefill JIT must define FMHA_SM100_PREFILL_BF16X_HAS_QMUL4"
#endif
#if !defined(FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT)
#error "The BF16X prefill JIT must define FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT"
#endif

#if FMHA_SM100_PREFILL_BF16X_HAS_QMUL4 &&                                                    \
    (!defined(__CUDACC_VER_MAJOR__) || (__CUDACC_VER_MAJOR__ < 13) ||                        \
     (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 4))
#error "BF16X prefill attention QMUL4 requires CUDA Toolkit 13.4 or newer"
#endif

namespace fmha_sm100::prefill_bf16x::sm100::common {

// One E4M3 block scale duplicated into a 16-bit pair. With block-scale staging the scale is
// divided by 2^shift first: exact through f16, rounded back to E4M3 (scales below 2^(shift - 6)
// become subnormal, below 2^(shift - 10) zero), the decode kernel's rounding. Both dequant paths
// start from this pair, so they produce the same E4M3 values.
#if FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT == 0
CUTLASS_DEVICE uint16_t stage_block_scale_pair(uint8_t scale_e4m3) {
  return static_cast<uint16_t>(scale_e4m3 * 0x0101u);
}
#else
CUTLASS_DEVICE uint16_t stage_block_scale_pair(uint8_t scale_e4m3) {
  constexpr uint32_t kStageF16x2 =
      ((15u - FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT) << 10) * 0x00010001u;
  uint16_t const pair = static_cast<uint16_t>(scale_e4m3 * 0x0101u);
  uint16_t staged;
  asm("{\n"
      ".reg .b32 f16x2;\n"
      "cvt.rn.f16x2.e4m3x2 f16x2, %1;\n"
      "mul.rn.f16x2 f16x2, f16x2, %2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 %0, f16x2;\n"
      "}\n"
      : "=h"(staged)
      : "h"(pair), "r"(kStageF16x2));
  return staged;
}
#endif

// Dequantizes eight consecutive E2M1 values (one 32-bit word, element i in nibble i) that share
// one block scale into eight E4M3 values: `output_lo` holds elements 0-3 and `output_hi` elements
// 4-7, each byte i being element i. `scale_word` comes from `make_dequant_scale_word`, so the
// per-element scale broadcast happens once per group of eight.

#if FMHA_SM100_PREFILL_BF16X_HAS_QMUL4

// QMUL4 operand: the staged scale in all four lanes.
CUTLASS_DEVICE uint32_t make_dequant_scale_word(uint8_t scale_e4m3) {
  return static_cast<uint32_t>(stage_block_scale_pair(scale_e4m3)) * 0x00010001u;
}

CUTLASS_DEVICE void nvfp4_to_e4m3x4(
    uint32_t& output, uint16_t packed_fp4, uint32_t scale_e4m3x4) {
  asm volatile(
      "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;"
      : "=r"(output)
      : "h"(packed_fp4), "r"(scale_e4m3x4));
}

CUTLASS_DEVICE void dequant_e2m1x8_uniform_scale(
    uint32_t& output_lo, uint32_t& output_hi,
    uint32_t packed_fp4, uint32_t scale_word) {
  nvfp4_to_e4m3x4(output_lo, static_cast<uint16_t>(packed_fp4), scale_word);
  nvfp4_to_e4m3x4(
      output_hi, static_cast<uint16_t>(packed_fp4 >> 16), scale_word);
}

#else

// Architectures without QMUL4 (SM107) multiply in packed FP16. An E2M1 x E4M3 product has at
// most six significant bits and stays within the FP16 normal range, so the product is exact
// and the single rounding happens in the final E4M3 conversion, exactly as in QMUL4.
CUTLASS_DEVICE uint32_t make_dequant_scale_word(uint8_t scale_e4m3) {
  uint32_t scale_f16x2;
  asm("{\n"
      ".reg .b16 scale_pair;\n"
      "mov.b16 scale_pair, %1;\n"
      "cvt.rn.f16x2.e4m3x2 %0, scale_pair;\n"
      "}\n"
      : "=r"(scale_f16x2)
      : "h"(stage_block_scale_pair(scale_e4m3)));
  return scale_f16x2;
}

CUTLASS_DEVICE void dequant_e2m1x8_uniform_scale(
    uint32_t& output_lo, uint32_t& output_hi,
    uint32_t packed_fp4, uint32_t scale_f16x2) {
  asm volatile(
      "{\n"
      ".reg .b8 e0, e1, e2, e3;\n"
      ".reg .b32 f0, f1, f2, f3;\n"
      ".reg .b16 q0, q1, q2, q3;\n"
      "mov.b32 {e0, e1, e2, e3}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 f0, e0;\n"
      "cvt.rn.f16x2.e2m1x2 f1, e1;\n"
      "cvt.rn.f16x2.e2m1x2 f2, e2;\n"
      "cvt.rn.f16x2.e2m1x2 f3, e3;\n"
      "mul.rn.f16x2 f0, f0, %3;\n"
      "mul.rn.f16x2 f1, f1, %3;\n"
      "mul.rn.f16x2 f2, f2, %3;\n"
      "mul.rn.f16x2 f3, f3, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 q0, f0;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 q1, f1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 q2, f2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 q3, f3;\n"
      "mov.b32 %0, {q0, q1};\n"
      "mov.b32 %1, {q2, q3};\n"
      "}\n"
      : "=&r"(output_lo), "=&r"(output_hi)
      : "r"(packed_fp4), "r"(scale_f16x2));
}

#endif

// E4M3X (the CuTe-DSL fp8 forward's K / V dequant with a global scale, utils cvt_fp4x8_e2m1_scaled_e4m3x8 fallback):
// the block scale as an exact f16 pair; per E2M1 pair: cvt.rn.f16x2.e2m1x2, mul.rn.f16x2 by the scale (exact),
// cvt.f32.f16 (exact), mul.rn.f32 by the fp32 global scale (rounding 1), cvt.rn.satfinite.e4m3x2.f32 (rounding 2).
// Byte i = element i (low nibble = even element). A 0 scale gives exact +-0; a NaN scale byte gives NaN.
CUTLASS_DEVICE uint32_t e4m3x_scale_f16x2(uint8_t scale_e4m3) {
  uint32_t scale_f16x2;
  uint16_t const pair = static_cast<uint16_t>(scale_e4m3 * 0x0101u);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(scale_f16x2) : "h"(pair));
  return scale_f16x2;
}

CUTLASS_DEVICE uint16_t e4m3x_dequant_pair(uint32_t e2m1_byte, uint32_t scale_f16x2, float global_scale) {
  uint16_t out;
  asm("{\n"
      ".reg .b8 e;\n"
      ".reg .b32 f;\n"
      ".reg .b16 h0, h1;\n"
      ".reg .f32 x0, x1;\n"
      "cvt.u8.u32 e, %1;\n"
      "cvt.rn.f16x2.e2m1x2 f, e;\n"
      "mul.rn.f16x2 f, f, %2;\n"
      "mov.b32 {h0, h1}, f;\n"
      "cvt.f32.f16 x0, h0;\n"
      "cvt.f32.f16 x1, h1;\n"
      "mul.rn.f32 x0, x0, %3;\n"
      "mul.rn.f32 x1, x1, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f32 %0, x1, x0;\n"
      "}\n"
      : "=h"(out)
      : "r"(e2m1_byte), "r"(scale_f16x2), "f"(global_scale));
  return out;
}

CUTLASS_DEVICE void dequant_e2m1x8_scale_global_e4m3x(uint32_t& output_lo, uint32_t& output_hi, uint32_t packed_fp4,
                                                      uint32_t scale_f16x2, float global_scale) {
  uint32_t const q0 = e4m3x_dequant_pair(packed_fp4 & 0xffu, scale_f16x2, global_scale);
  uint32_t const q1 = e4m3x_dequant_pair((packed_fp4 >> 8) & 0xffu, scale_f16x2, global_scale);
  uint32_t const q2 = e4m3x_dequant_pair((packed_fp4 >> 16) & 0xffu, scale_f16x2, global_scale);
  uint32_t const q3 = e4m3x_dequant_pair(packed_fp4 >> 24, scale_f16x2, global_scale);
  output_lo = q0 | (q1 << 16);
  output_hi = q2 | (q3 << 16);
}

}  // namespace fmha_sm100::prefill_bf16x::sm100::common
