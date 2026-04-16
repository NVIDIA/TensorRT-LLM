/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cute/tensor.hpp>
#include <cute/arch/config.hpp>
#include <cutlass/array.h>

namespace trtllm {
namespace dev {

// FlashAttention4-style exp2 approximation
CUTLASS_DEVICE cutlass::Array<float, 2> exp2f2Emu(float x0, float x1) {
  using namespace cute;

  const uint32_t input_min_bits = 0xC2FE0000; // -127.0
  const uint32_t round_out_bits = 0x4B400000; // 1.5 * 2^23
  // Polynomials to approximate exp2(x) in the [0, 1] interval
  const uint32_t c1_bits = 0x3F31F519;
  const uint32_t c2_bits = 0x3E6906A4;
  const uint32_t c3_bits = 0x3D9DF09D;
  // Constants converted to float
  const float input_min = *reinterpret_cast<const float*>(&input_min_bits);
  const float round_out = *reinterpret_cast<const float*>(&round_out_bits);
  const float c1 = *reinterpret_cast<const float*>(&c1_bits);
  const float c2 = *reinterpret_cast<const float*>(&c2_bits);
  const float c3 = *reinterpret_cast<const float*>(&c3_bits);

  // Safeguard minimum (rounding-out won't work otherwise)
  float2 x_x2 = make_float2(cutlass::fast_max(input_min, x0), cutlass::fast_max(input_min, x1));

  float2 round_out_x2 = make_float2(round_out, round_out);

  float2 int_x_at_lower_bits_x2, subnorm_x_x2, temp_x2;
#if defined(CUTE_ARCH_FFMA2_SM100_ENABLED)
  asm volatile(
    "add.rm.ftz.f32x2 %0, %3, %4;\n" // [s] [exponent bit represents 2^23] [mantissa has (1<<22) +
                                     // floor(x)]. The (1<22) ensures exponent bit untouched if
                                     // floor(x)<0.
    "sub.rn.ftz.f32x2 %2, %0, %4;\n" // %2.x = floor(x). We don't directly call floor(x) because we
                                     // need int32(floor(x)) to be in lower bits of %0.x
    "sub.rn.ftz.f32x2 %1, %3, %2;\n" // 0 <= %3.x = x - floor(x) < 1
    : "=l"(reinterpret_cast<uint64_t&>(int_x_at_lower_bits_x2)), // %0
      "=l"(reinterpret_cast<uint64_t&>(subnorm_x_x2)),           // %1
      "=l"(reinterpret_cast<uint64_t&>(temp_x2))                 // %2
    : "l"(reinterpret_cast<uint64_t const&>(x_x2)),              // %3
      "l"(reinterpret_cast<uint64_t const&>(round_out_x2)));     // %4
#else
  asm volatile("add.rm.ftz.f32  %0, %6, %8;\n"
               "sub.rn.ftz.f32  %4, %0, %8;\n"
               "sub.rn.ftz.f32  %2, %6, %4;\n"
               "add.rm.ftz.f32  %1, %7, %9;\n"
               "sub.rn.ftz.f32  %5, %1, %9;\n"
               "sub.rn.ftz.f32  %3, %7, %5;\n"
               : "=f"(int_x_at_lower_bits_x2.x), // %0
                 "=f"(int_x_at_lower_bits_x2.y), // %1
                 "=f"(subnorm_x_x2.x),           // %2
                 "=f"(subnorm_x_x2.y),           // %3
                 "=f"(temp_x2.x),                // %4
                 "=f"(temp_x2.y)                 // %5
               : "f"(x_x2.x),                    // %6
                 "f"(x_x2.y),                    // %7
                 "f"(round_out_x2.x),            // %8
                 "f"(round_out_x2.y));           // %9
#endif

  float2 c0_x2 = make_float2(1.f, 1.f);
  float2 c1_x2 = make_float2(c1, c1);
  float2 c2_x2 = make_float2(c2, c2);
  float2 c3_x2 = make_float2(c3, c3);

  cute::fma(temp_x2, subnorm_x_x2, c3_x2, c2_x2); // temp <- (f * c3 + c2)
  cute::fma(temp_x2,
            subnorm_x_x2,
            temp_x2,
            c1_x2); // temp <- f * temp + c1 = f * (f * c3 + c2) + c1
  cute::fma(temp_x2,
            subnorm_x_x2,
            temp_x2,
            c0_x2); // f * (f * (f * c3 + c2) + c1) + 1 \approx exp2(f) for 0 < f < 1

  uint32_t int_x0_at_lower_bits, int_x1_at_lower_bits;
  asm volatile("mov.b64 {%0, %1}, %2;\n"
               : "=r"(int_x0_at_lower_bits), "=r"(int_x1_at_lower_bits)
               : "l"(reinterpret_cast<uint64_t const&>(int_x_at_lower_bits_x2)));

  // Mul exp2f(int_x) <=> Mul ((int_x << 23 + 127) <=> Add (int_x << 23)
  uint32_t y0_bits = (int_x0_at_lower_bits << 23) + *reinterpret_cast<uint32_t*>(&temp_x2.x);
  uint32_t y1_bits = (int_x1_at_lower_bits << 23) + *reinterpret_cast<uint32_t*>(&temp_x2.y);

  cutlass::Array<float, 2> y;
  y[0] = *reinterpret_cast<float*>(&y0_bits);
  y[1] = *reinterpret_cast<float*>(&y1_bits);
  return y;
}
} // namespace dev
} // namespace trtllm