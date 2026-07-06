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

#include "CutlassUtils.h"

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeMxE4m3SfAndOutputScale(float& outputScale,
                                                     cutlass::float_ue8m0_t& sfOut,
                                                     float amax,
                                                     float const& sfScale) {
  float const amaxPow2 = trunc_abs_float_to_pow2(amax);
  float const sfVal = amaxPow2 * (1.f / 256.f) * sfScale;
  cutlass::Array<float, 1> sfArrayFp32;
  sfArrayFp32[0] = sfVal;
  sfOut = castArray<cutlass::float_ue8m0_t>(sfArrayFp32)[0];
  outputScale = sfVal != 0.f ? scale_rcp_exp_only(sfVal) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeMxE4m3SfAndOutputScale(float& outputScale,
                                                     cutlass::float_ue8m0_t& sfOut,
                                                     float amax,
                                                     float const& sfScale,
                                                     float const& sfScaleInv) {
  float const amaxPow2 = trunc_abs_float_to_pow2(amax);
  float const sfVal = amaxPow2 * (1.f / 256.f) * sfScale;
  cutlass::Array<float, 1> sfArrayFp32;
  sfArrayFp32[0] = sfVal;
  sfOut = castArray<cutlass::float_ue8m0_t>(sfArrayFp32)[0];
  outputScale = sfVal != 0.f ? scale_rcp_exp_only(sfVal * sfScaleInv) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumEltsPerThread, typename OutT>
inline __device__ void convertFloatToMxE4m3(OutT& out,
                                            cutlass::float_ue8m0_t& sfOut,
                                            float const (&input)[NumEltsPerThread],
                                            float sfScale) {
  // MxE4m3 uses one UE8M0 scale for each group of 32 E4M3 elements.
  int32_t constexpr NumEltsPerSf = 32;
  int32_t constexpr NumThreadsPerVec = NumEltsPerSf / NumEltsPerThread;
  static_assert(NumEltsPerSf % NumEltsPerThread == 0 &&
                NumEltsPerThread % 4 == 0, "NumEltsPerThread not supported.");
  static_assert(sizeof(OutT) == NumEltsPerThread, "Output type not supported."); // 1 byte per element.

  float localAmax = 0.f;
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread; ++i) {
    localAmax = fmaxf(localAmax, fabsf(input[i]));
  }

#pragma unroll
  for (int32_t step = 1; step < NumThreadsPerVec; step *= 2) {
    localAmax = fmaxf(__shfl_xor_sync(uint32_t(-1), localAmax, step), localAmax);
  }

  float outputScale;
  computeMxE4m3SfAndOutputScale(outputScale, sfOut, localAmax, sfScale);

  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread; ++i) {
    scaled[i] = input[i] * outputScale;
  }

  using OutVec = cutlass::Array<cutlass::float_e4m3_t, NumEltsPerThread>;
  reinterpret_cast<OutVec&>(out) = castArray<cutlass::float_e4m3_t>(scaled);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumEltsPerThread, typename OutT>
inline __device__ void convertFp16ToMxE4m3(OutT& out,
                                            cutlass::float_ue8m0_t& sfOut,
                                            cutlass::half_t const (&in)[NumEltsPerThread],
                                            float sfScale) {
  // MxE4m3 uses one UE8M0 scale for each group of 32 E4M3 elements.
  int32_t constexpr NumEltsPerSf = 32;
  int32_t constexpr NumThreadsPerVec = NumEltsPerSf / NumEltsPerThread;
  static_assert(NumEltsPerSf % NumEltsPerThread == 0 &&
                NumEltsPerThread % 4 == 0, "NumEltsPerThread not supported.");
  static_assert(sizeof(OutT) == NumEltsPerThread, "Output type not supported."); // 1 byte per element.

  auto inH2 = reinterpret_cast<half2 const *>(&in[0]);
  auto localAmaxH2 = __habs2(inH2[0]);
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread / 2; ++i) {
    localAmaxH2 = __hmax2(localAmaxH2, __habs2(inH2[i]));
  }

  // Perform warp-level reduction to achieve the amax of the vector of 16 elements.
  if constexpr (NumThreadsPerVec > 1) {
    static_assert(NumThreadsPerVec == 2 || NumThreadsPerVec == 4, "Not supported.");
    for (int32_t step = 1; step < NumThreadsPerVec; step *= 2) {
      localAmaxH2 = __hmax2(__shfl_xor_sync(uint32_t(-1), localAmaxH2, step), localAmaxH2);
    }
  }

  float localAmax = float(__hmax(localAmaxH2.x, localAmaxH2.y));
  float outputScale;
  computeMxE4m3SfAndOutputScale(outputScale, sfOut, localAmax, sfScale);

  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread / 2; ++i) {
    float2 tmp = __half22float2(inH2[i]);
    scaled[i * 2 + 0] = tmp.x * outputScale;
    scaled[i * 2 + 1] = tmp.y * outputScale;
  }

  using OutVec = cutlass::Array<cutlass::float_e4m3_t, NumEltsPerThread>;
  reinterpret_cast<OutVec&>(out) = castArray<cutlass::float_e4m3_t>(scaled);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
