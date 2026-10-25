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
// Warp / block reduction helpers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace glm {
namespace k {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

// Sum over the whole block; all threads receive the result. `sh` needs 32 floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nw = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = lane < nw ? sh[lane] : 0.f;
  return warp_sum(v);
}

__device__ __forceinline__ float block_max(float v, float* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nw = (blockDim.x + 31) >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = lane < nw ? sh[lane] : -INFINITY;
  return warp_max(v);
}

// IEEE round-to-nearest a / b given y = __frcp_rn(b) (Markstein: q = RN(a y), r = a - b q exact
// by FMA, RN(q + r y)): bitwise a / b for the FP8 activation quantization's operands (normal
// numbers; 1.24e9 random samples of x / (max(amax, 1e-12) / 448) agreed), without the division
// routine's per-call cost.
__device__ __forceinline__ float div_rn(float a, float b, float y) {
  const float q = __fmul_rn(a, y);
  const float r = __fmaf_rn(-q, b, a);
  return __fmaf_rn(r, y, q);
}

__device__ __forceinline__ float bf16r(float x) {  // round-trip through bf16
  return __bfloat162float(__float2bfloat16(x));
}

// 8 x bf16 <-> float helpers
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* f) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float* f) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}

}  // namespace k
}  // namespace glm
