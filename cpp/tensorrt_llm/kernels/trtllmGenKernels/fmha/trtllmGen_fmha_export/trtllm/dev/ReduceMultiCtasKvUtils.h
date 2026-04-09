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

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "CutlassUtils.h"
#include <float.h>
#include "Fp4Utils.h"

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype, int32_t NumElts>
inline __device__ void convertAndStoreToGmem(char* gmemPtr, float (&input)[NumElts]) {
  static_assert(sizeof(Dtype) == 0, "Not implemented.");
}

template <typename Dtype, int32_t NumElts>
inline __device__ void convertAndStoreToGmem(char* gmemPtr,
                                             char* oSfPtr,
                                             float (&input)[NumElts],
                                             float sfScale,
                                             bool isValidRow) {
  static_assert(sizeof(Dtype) == 0, "Not implemented.");
}

template <>
inline __device__ void convertAndStoreToGmem<cutlass::half_t, 8>(char* gmemPtr, float (&input)[8]) {
  uint4 output;
  output.x = convert_float2_to_half(input[0], input[1]);
  output.y = convert_float2_to_half(input[2], input[3]);
  output.z = convert_float2_to_half(input[4], input[5]);
  output.w = convert_float2_to_half(input[6], input[7]);
  *reinterpret_cast<uint4*>(gmemPtr) = output;
}

template <>
inline __device__ void convertAndStoreToGmem<cutlass::bfloat16_t, 8>(char* gmemPtr,
                                                                     float (&input)[8]) {
  uint4 output;
  output.x = convert_float2_to_bfloat16(input[0], input[1]);
  output.y = convert_float2_to_bfloat16(input[2], input[3]);
  output.z = convert_float2_to_bfloat16(input[4], input[5]);
  output.w = convert_float2_to_bfloat16(input[6], input[7]);
  *reinterpret_cast<uint4*>(gmemPtr) = output;
}

template <>
inline __device__ void convertAndStoreToGmem<cutlass::float_e4m3_t, 8>(char* gmemPtr,
                                                                       float (&input)[8]) {
  uint2 output;
  output.x = convert_float4_to_e4m3(input[0], input[1], input[2], input[3]);
  output.y = convert_float4_to_e4m3(input[4], input[5], input[6], input[7]);
  *reinterpret_cast<uint2*>(gmemPtr) = output;
}

template <>
inline __device__ void convertAndStoreToGmem<cutlass::float_e2m1_t, 8>(char* gmemPtr,
                                                                       char* gmemSfPtr,
                                                                       float (&input)[8],
                                                                       float sfScale,
                                                                       bool isValidRow) {
  cutlass::float_e4m3_t sfOut;
  uint32_t valOut;
  convertFloatToE2m1<8>(valOut, sfOut, input, sfScale);
  if (isValidRow) {
    *reinterpret_cast<uint32_t*>(gmemPtr) = valOut;
    if (threadIdx.x % 2 == 0) {
      *reinterpret_cast<cutlass::float_e4m3_t*>(gmemSfPtr) = sfOut;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype, int32_t NumElts>
inline __device__ void convertToFloatAndAccumulate(float (&output)[NumElts],
                                                   uint4 input,
                                                   float scale0,
                                                   float scale1) {
  static_assert(sizeof(Dtype) == 0, "Not implemented.");
}

template <>
inline __device__ void convertToFloatAndAccumulate<cutlass::half_t, 8>(float (&output)[8],
                                                                       uint4 input,
                                                                       float scale0,
                                                                       float scale1) {
  float2 scales0 = make_float2(scale0, scale0);
  float2 scales1 = make_float2(scale1, scale1);
#pragma unroll
  for (int32_t ii = 0; ii < 4; ++ii) {
    float2 a = __half22float2(reinterpret_cast<__half2*>(&input)[ii]);
    float2& c = reinterpret_cast<float2(&)[4]>(output)[ii];
    // FFMA2: output = input * scale1 + output * scale0.
    cute::mul(c, c, scales0);
    cute::fma(c, a, scales1, c);
  }
}

template <>
inline __device__ void convertToFloatAndAccumulate<cutlass::bfloat16_t, 8>(float (&output)[8],
                                                                           uint4 input,
                                                                           float scale0,
                                                                           float scale1) {
  float2 scales0 = make_float2(scale0, scale0);
  float2 scales1 = make_float2(scale1, scale1);
#pragma unroll
  for (int32_t ii = 0; ii < 4; ++ii) {
    float2 a = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&input)[ii]);
    float2& c = reinterpret_cast<float2(&)[4]>(output)[ii];
    // FFMA2: output = input * scale1 + output * scale0.
    cute::mul(c, c, scales0);
    cute::fma(c, a, scales1, c);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
