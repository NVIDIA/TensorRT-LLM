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
  static_assert(NumEltsPerSf % NumEltsPerThread == 0 && NumEltsPerThread % 4 == 0,
                "NumEltsPerThread not supported.");
  static_assert(sizeof(OutT) == NumEltsPerThread,
                "Output type not supported."); // 1 byte per element.

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
  static_assert(NumEltsPerSf % NumEltsPerThread == 0 && NumEltsPerThread % 4 == 0,
                "NumEltsPerThread not supported.");
  static_assert(sizeof(OutT) == NumEltsPerThread,
                "Output type not supported."); // 1 byte per element.

  auto inH2 = reinterpret_cast<half2 const*>(&in[0]);
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

template <int32_t NumReduceVals, int32_t NumVals>
__device__ __forceinline__ float reduceMaxAbs(cutlass::Array<float, NumVals> const& vals,
                                              float init = 1.0e-12f) {
  static_assert(NumReduceVals % 4 == 0, "Number of reduced values must be divisible by 4.");
  static_assert(NumReduceVals <= NumVals, "Reduced value range exceeds value array.");

  float max0 = init;
  float max1 = init;
  float max2 = init;
  float max3 = init;
#pragma unroll
  for (int32_t i = 0; i < NumReduceVals / 4; ++i) {
    int32_t const ii = i * 4;
    max0 = fmaxf(max0, fabsf(vals[ii + 0]));
    max1 = fmaxf(max1, fabsf(vals[ii + 1]));
    max2 = fmaxf(max2, fabsf(vals[ii + 2]));
    max3 = fmaxf(max3, fabsf(vals[ii + 3]));
  }
  return fmaxf(fmaxf(max0, max1), fmaxf(max2, max3));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Write the FP32 dequant scale for one output block and pack the already-finalized values to E4M3.
// The stored scale is amax / 448, so dequantization is `fp8_value * scalePtr[scaleOffset]`.
template <int32_t NumVals, int32_t NumPackedRegs, typename OutRegs>
inline __device__ void e4m3PackEpilogue(OutRegs& out,
                                        cutlass::Array<float, NumVals> const& vals,
                                        float* scalePtr,
                                        int64_t scaleOffset,
                                        float amax) {
  static_assert(NumVals == NumPackedRegs * 4, "One packed register stores four E4M3 values.");

  // E4M3's largest finite value is 448; store scale = amax / 448 and quantize with 448 / amax.
  float constexpr fp8Max = 448.f;
  float constexpr fp8MaxRcp = 1.f / fp8Max;
  float const outScale = amax * fp8MaxRcp;
  float const invScale = __fdividef(fp8Max, amax);
  scalePtr[scaleOffset] = outScale;

  // Keep these multiplies scalar: fmul2 introduces extra array temporaries here and increases
  // register spills in this epilogue.
#pragma unroll
  for (int32_t regIdx = 0; regIdx < NumPackedRegs; ++regIdx) {
    int32_t const ii = regIdx * 4;
    out[regIdx] = convert_float4_to_e4m3(vals[ii + 0] * invScale,
                                         vals[ii + 1] * invScale,
                                         vals[ii + 2] * invScale,
                                         vals[ii + 3] * invScale);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Quantize one output block to packed E4M3 and write its FP32 dequant scale. The input values are
// expected to already be in the final output layout.
template <int32_t NumVals, int32_t NumPackedRegs, typename OutRegs>
inline __device__ void e4m3QuantEpilogue(OutRegs& out,
                                         cutlass::Array<float, NumVals> const& vals,
                                         float* scalePtr,
                                         int64_t scaleOffset) {
  static_assert(NumVals == NumPackedRegs * 4, "One packed register stores four E4M3 values.");

  float const amax = reduceMaxAbs<NumVals>(vals);
  e4m3PackEpilogue<NumVals, NumPackedRegs>(out, vals, scalePtr, scaleOffset, amax);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Apply the DSv4 inverse-RoPE transform to the final 64 dimensions of a 512-dim head when this
// 1x128 block covers that range, then quantize the block to packed E4M3 and write its FP32 dequant
// scale. Blocks outside the inverse-RoPE range follow the same path as e4m3QuantEpilogue.
template <int32_t NumVals, int32_t NumPackedRegs, typename OutRegs>
inline __device__ void dsv4InvRopeFp8QuantEpilogue(OutRegs& out,
                                                   cutlass::Array<float, NumVals>& vals,
                                                   float* scalePtr,
                                                   float const* cosSinCache,
                                                   int64_t scaleOffset,
                                                   int32_t position,
                                                   int32_t headDimOffset) {
  static_assert(NumVals == NumPackedRegs * 4, "One packed register stores four E4M3 values.");
  static_assert(NumVals == 128, "DSv4 fused epilogue processes one 1x128 quant group.");

  // DSv4 inverse-RoPE applies to the last 64 dimensions, [448, 512), of a 512-dim head. The
  // helper processes one 128-value quant block, so the RoPE block starts at 384 and the RoPE
  // values begin at offset 64 inside that block. DSv4 uses non-NeoX interleaved RoPE: adjacent
  // element pairs share one cos/sin value. Each cos/sin cache row is laid out as
  // [cos(32), sin(32)].
  int32_t constexpr ropeStart = 448;
  int32_t constexpr ropeHalf = 32;
  int32_t constexpr ropeBlockStart = ropeStart - ropeHalf * 2;
  int32_t constexpr ropeOffset = ropeStart - ropeBlockStart;
  int32_t constexpr cosSinStride = ropeHalf * 2;

  // TODO: use headDimOffset as template parameter to drop warp divergence.
  if (headDimOffset != ropeBlockStart) {
    e4m3QuantEpilogue<NumVals, NumPackedRegs>(out, vals, scalePtr, scaleOffset);
    return;
  }

  float const* csRow = cosSinCache + position * cosSinStride;

  static_assert(ropeOffset % 4 == 0, "The RoPE offset must be aligned to packed E4M3 registers.");
  int32_t constexpr ropePackedRegStart = ropeOffset / 4;
  float amax = reduceMaxAbs<ropeOffset>(vals);

  // Match the standalone TRT-LLM inverse-RoPE FP8 kernel's per-4-value structure: each packed
  // register in the RoPE half contains two interleaved RoPE pairs.
#pragma unroll
  for (int32_t regIdx = ropePackedRegStart; regIdx < NumPackedRegs; ++regIdx) {
    int32_t const ii = regIdx * 4;
    int32_t const csIdx = (regIdx - ropePackedRegStart) * 2;
    // TODO: use float2 for cos, sin pair. Currently do not use because register spills.
    float const cos0 = csRow[csIdx + 0];
    float const sin0 = csRow[ropeHalf + csIdx + 0];
    float const first0 = vals[ii + 0];
    float const second0 = vals[ii + 1];
    float const rotatedFirst0 = first0 * cos0 + second0 * sin0;
    float const rotatedSecond0 = second0 * cos0 - first0 * sin0;
    vals[ii + 0] = rotatedFirst0;
    vals[ii + 1] = rotatedSecond0;
    amax = fmaxf(amax, fmaxf(fabsf(rotatedFirst0), fabsf(rotatedSecond0)));

    float const cos1 = csRow[csIdx + 1];
    float const sin1 = csRow[ropeHalf + csIdx + 1];
    float const first1 = vals[ii + 2];
    float const second1 = vals[ii + 3];
    float const rotatedFirst1 = first1 * cos1 + second1 * sin1;
    float const rotatedSecond1 = second1 * cos1 - first1 * sin1;
    vals[ii + 2] = rotatedFirst1;
    vals[ii + 3] = rotatedSecond1;
    amax = fmaxf(amax, fmaxf(fabsf(rotatedFirst1), fabsf(rotatedSecond1)));
  }

  e4m3PackEpilogue<NumVals, NumPackedRegs>(out, vals, scalePtr, scaleOffset, amax);
}

} // namespace dev
} // namespace trtllm
