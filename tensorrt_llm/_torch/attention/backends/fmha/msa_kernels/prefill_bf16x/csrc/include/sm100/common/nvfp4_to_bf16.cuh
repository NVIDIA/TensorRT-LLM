// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

// NVFP4 (E2M1 code x E4M3 block scale) -> BF16 dequant helpers for the BF16 data path
// (FMHA_SM100_PREFILL_BF16X_DTYPE == 1).
//
// Exactness:
//  * E2M1 -> BF16 is exact: the E2M1 values (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}) have at most two
//    significant bits. The path is cvt.rn.f16x2.e2m1x2 (exact, F16 holds them), cvt.f32.f16
//    (exact, widening) and cvt.rn.bf16x2.f32 (exact: 2 significant bits fit BF16's 8, and the
//    exponent range is inside BF16's). A direct cvt.rn.bf16x2.e2m1x2 is not used: CUTLASS 4.8 has
//    no use of it and it could not be verified against the toolchain without compiling.
//  * E4M3 -> BF16 is exact: E4M3 values have at most 4 significant bits and magnitudes in
//    [2^-9, 448], so F16 (cvt.rn.f16x2.e4m3x2) holds them exactly, and so does BF16. A NaN scale
//    (0x7F / 0xFF) stays NaN.
//  * code x scale in mul.rn.bf16x2 is exact: at most 2 + 4 = 6 significant bits <= 8, magnitude
//    in [2^-10, 2688]; no rounding happens.
//  * For K the block scale is first multiplied by the K global scale (cvt.rn.bf16 of the fp32
//    global), mul.rn.bf16x2(scale_bf16, kglobal_bf16): that product and the following
//    code x combined product each round once to nearest-even in BF16. This is the defined
//    reference semantics of the BF16X K dequant.

#include <cstdint>

#include "cutlass/cutlass.h"

namespace fmha_sm100::prefill_bf16x::sm100::common {

// fp32 -> BF16 (round to nearest even), duplicated into both halves of a bf16x2.
CUTLASS_DEVICE uint32_t f32_to_bf16x2_broadcast(float value) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(packed) : "f"(value));
  return packed;
}

// Two fp32 -> one bf16x2 with `lo` in bits [15:0] and `hi` in bits [31:16] (cvt.rn.bf16x2.f32
// puts its first source in the upper half).
CUTLASS_DEVICE uint32_t pack_bf16x2_rn(float lo, float hi) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(hi), "f"(lo));
  return packed;
}

CUTLASS_DEVICE uint32_t mul_bf16x2_rn(uint32_t lhs, uint32_t rhs) {
  uint32_t product;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(product) : "r"(lhs), "r"(rhs));
  return product;
}

// One E4M3 block-scale byte -> BF16, duplicated into both halves (exact, see above).
// BF16X-M3: the reference's bit decode (utils cvt_fp8x4_e4m3_bf16x4 on the byte replicated x4):
// sign | (exp, mant) >> 4 as a BF16 pattern, then fma.rn.bf16x2 by 2^120. Equal to the hardware
// cvt on every finite code; the NaN codes 0x7f / 0xff decode to +-480 as in the reference.
CUTLASS_DEVICE uint32_t e4m3_to_bf16x2_broadcast(uint8_t scale_e4m3) {
  uint32_t const replicated = static_cast<uint32_t>(scale_e4m3) * 0x01010101u;
  uint32_t packed;
  asm("{\n"
      ".reg .b32 q, mant, out, bias, zero;\n"
      "prmt.b32 q, %1, %1, 0x1302;\n"
      "and.b32 out, q, 0x80008000;\n"
      "and.b32 mant, q, 0x7f007f00;\n"
      "shr.u32 mant, mant, 4;\n"
      "or.b32 out, out, mant;\n"
      "mov.b32 bias, 0x7b807b80;\n"
      "mov.b32 zero, 0;\n"
      "fma.rn.bf16x2 %0, out, bias, zero;\n"
      "}\n"
      : "=r"(packed)
      : "r"(replicated));
  return packed;
}

// Dequantizes eight consecutive E2M1 values (one 32-bit word, element i in nibble i, so the low
// nibble of each byte is the even element) that share one scale into eight BF16 values:
// out[j] holds elements 2j (bits [15:0]) and 2j + 1 (bits [31:16]), the little-endian SMEM order.
// `scale_bf16x2` is the (combined) scale in both halves.
CUTLASS_DEVICE void dequant_e2m1x8_to_bf16x8(uint32_t (&out)[4], uint32_t packed_fp4,
                                             uint32_t scale_bf16x2) {
  asm("{\n"
      ".reg .b8 e0, e1, e2, e3;\n"
      ".reg .b32 h0, h1, h2, h3;\n"
      ".reg .b16 l0, u0, l1, u1, l2, u2, l3, u3;\n"
      ".reg .f32 fl0, fu0, fl1, fu1, fl2, fu2, fl3, fu3;\n"
      ".reg .b32 b0, b1, b2, b3;\n"
      "mov.b32 {e0, e1, e2, e3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 h0, e0;\n"
      "cvt.rn.f16x2.e2m1x2 h1, e1;\n"
      "cvt.rn.f16x2.e2m1x2 h2, e2;\n"
      "cvt.rn.f16x2.e2m1x2 h3, e3;\n"
      "mov.b32 {l0, u0}, h0;\n"
      "mov.b32 {l1, u1}, h1;\n"
      "mov.b32 {l2, u2}, h2;\n"
      "mov.b32 {l3, u3}, h3;\n"
      "cvt.f32.f16 fl0, l0;\n"
      "cvt.f32.f16 fu0, u0;\n"
      "cvt.f32.f16 fl1, l1;\n"
      "cvt.f32.f16 fu1, u1;\n"
      "cvt.f32.f16 fl2, l2;\n"
      "cvt.f32.f16 fu2, u2;\n"
      "cvt.f32.f16 fl3, l3;\n"
      "cvt.f32.f16 fu3, u3;\n"
      "cvt.rn.bf16x2.f32 b0, fu0, fl0;\n"
      "cvt.rn.bf16x2.f32 b1, fu1, fl1;\n"
      "cvt.rn.bf16x2.f32 b2, fu2, fl2;\n"
      "cvt.rn.bf16x2.f32 b3, fu3, fl3;\n"
      "mul.rn.bf16x2 %0, b0, %5;\n"
      "mul.rn.bf16x2 %1, b1, %5;\n"
      "mul.rn.bf16x2 %2, b2, %5;\n"
      "mul.rn.bf16x2 %3, b3, %5;\n"
      "}\n"
      : "=&r"(out[0]), "=&r"(out[1]), "=&r"(out[2]), "=&r"(out[3])
      : "r"(packed_fp4), "r"(scale_bf16x2));
}

}  // namespace fmha_sm100::prefill_bf16x::sm100::common
