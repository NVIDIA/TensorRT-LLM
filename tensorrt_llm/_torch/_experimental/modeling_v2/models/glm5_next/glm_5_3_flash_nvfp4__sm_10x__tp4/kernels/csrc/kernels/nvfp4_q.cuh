/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// NVFP4 activation quantization of 16-value blocks (nvfp4.cu's kernels).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "../common.h"

namespace glm {
namespace k {
namespace nvq {

__device__ __forceinline__ float rcp_approx(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// v[0..7] -> eight e2m1 nibbles, v[0] in the lowest
__device__ __forceinline__ uint32_t e2m1x8(const float* v) {
  uint32_t r;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(r)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return r;
}

// One 16-element block: scale byte and the 8 bytes of packed values. sfs = 1 / gs.
__device__ __forceinline__ uint8_t quant16(const float* v, float sfs, uint2& out) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  float sfv = sfs * (amax * rcp_approx(6.f));
  const __nv_fp8_e4m3 s8(sfv);
  sfv = static_cast<float>(s8);
  const float osc = amax != 0.f ? rcp_approx(sfv * rcp_approx(sfs)) : 0.f;
  float w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = v[i] * osc;
  out.x = e2m1x8(w);
  out.y = e2m1x8(w + 8);
  return s8.__x;
}

__device__ __forceinline__ void load16_bf16(const bf16* p, float* v) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&a);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
  h = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[8 + 2 * i] = f.x;
    v[8 + 2 * i + 1] = f.y;
  }
}

}  // namespace nvq
}  // namespace k
}  // namespace glm
