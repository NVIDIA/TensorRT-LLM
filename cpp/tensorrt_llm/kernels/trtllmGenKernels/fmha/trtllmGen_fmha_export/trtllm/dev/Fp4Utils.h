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

#include <cute/tensor.hpp>
#include "CutlassUtils.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cuda_fp4.h> // __nvfp4*
#pragma GCC diagnostic pop

typedef __nv_fp8x4_storage_t __nv_fp4x8_storage_t;
typedef unsigned long __nv_fp8x8_storage_t;

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmUnpackFp4Mode8x16Numx2(uint32_t& out0,
                                                  uint32_t& out1,
                                                  void const* srcMem) {
  asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];"
               : "=r"(out0), "=r"(out1)
               : "l"(srcMem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldsmUnpackFp4ModeTranspose16x16Numx1(uint32_t& out0,
                                                            uint32_t& out1,
                                                            void const* srcMem) {
  asm volatile("ldmatrix.sync.aligned.shared::cta.m16n16.x1.trans.b8x16.b4x16_p64 {%0, %1}, [%2];"
               : "=r"(out0), "=r"(out1)
               : "l"(srcMem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cute::uint128_t e2m1ToFp16(uint32_t src) {
  uint32_t outFp16[4];
  asm volatile("{\n"
               ".reg .b8 byte0, byte1, byte2, byte3;\n"
               "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
               "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
               "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
               "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
               "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
               "}\n"
               : "=r"(outFp16[0]), "=r"(outFp16[1]), "=r"(outFp16[2]), "=r"(outFp16[3])
               : "r"(src));
  return *reinterpret_cast<cute::uint128_t*>(&outFp16[0]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeSfAndOutputScale(float& outputScale,
                                               cutlass::float_e4m3_t& sfOut,
                                               float const& amax,
                                               float const& sfScale) {
  // The reciprocal of E2M1_MAX.
  float constexpr reciprocalOfE2m1Max = 1.f / 6.f;
  // The FP32 sf.
  float sfVal = sfScale * amax * reciprocalOfE2m1Max;
  // The FP8 sf.
  sfOut = cutlass::float_e4m3_t{sfVal};
  // Cast back to FP32.
  sfVal = static_cast<float>(sfOut);

  // The output scale.
  // outputScale = reciprocal(fp32(fp8(sfVal * sfScale))) * reciprocal(sfScale))
  outputScale =
    sfVal != 0.f ? reciprocal_approximate_ftz(sfVal * reciprocal_approximate_ftz(sfScale)) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeSfAndOutputScale(float& outputScale,
                                               cutlass::float_e4m3_t& sfOut,
                                               float const& amax,
                                               float const& sfScale,
                                               float const& sfScaleInv) {
  // The reciprocal of E2M1_MAX.
  float constexpr reciprocalOfE2m1Max = 1.f / 6.f;
  // The FP32 sf.
  float sfVal = sfScale * amax * reciprocalOfE2m1Max;
  // The FP8 sf.
  sfOut = cutlass::float_e4m3_t{sfVal};
  // Cast back to FP32.
  sfVal = static_cast<float>(sfOut);

  // The output scale.
  // outputScale = reciprocal(fp32(fp8(sfVal * sfScale))) * reciprocal(sfScale))
  // Same as
  outputScale = sfVal != 0.f ? reciprocal_approximate_ftz(sfVal * sfScaleInv) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float16 type Tensor to E2m1.
template <int NumEltsPerThread, typename OutT>
inline __device__ void convertFp16ToE2m1(OutT& out,
                                         cutlass::float_e4m3_t& sfOut,
                                         uint32_t const (&in)[NumEltsPerThread / 2],
                                         float sfScale) {

  // This function converts fp16 elements to E2m1.
  // Inputs:
  // - in: fp16 values.
  // - sfScale: The global scaling factor.
  // Outputs:
  // - return value: The casted E2m1 values.
  // - sfOut: The local scaling factor.

  // The number of half2 elements per thread.
  int32_t constexpr NumHalf2PerThread = NumEltsPerThread / 2;
  // The number of threads per vector of 16 elements.
  int32_t constexpr NumThreadsPerVec = 16 / NumEltsPerThread;

  // Cast the input to half2 array.
  half2 const(&inH2)[NumHalf2PerThread] = reinterpret_cast<half2 const(&)[NumHalf2PerThread]>(in);

  // Get absolute maximum values among the local 8 values.
  auto localAmax = __habs2(inH2[0]);
#pragma unroll
  for (int32_t ii = 1; ii < NumHalf2PerThread; ++ii) {
    localAmax = __hmax2(localAmax, __habs2(inH2[ii]));
  }

  // Perform warp-level reduction to achieve the amax of the vector of 16 elements.
  if constexpr (NumThreadsPerVec > 1) {
    static_assert(NumThreadsPerVec == 2 || NumThreadsPerVec == 4, "Not supported.");
    for (int32_t step = 1; step < NumThreadsPerVec; step *= 2) {
      localAmax = __hmax2(__shfl_xor_sync(uint32_t(-1), localAmax, step), localAmax);
    }
  }

  // Get the final absolute maximum values.
  float vecAmax = float(__hmax(localAmax.x, localAmax.y));

  // Compute the sf and output scale.
  float outputScale;
  computeSfAndOutputScale(outputScale, sfOut, vecAmax, sfScale);

  // Apply the output scale.
  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int32_t ii = 0; ii < NumHalf2PerThread; ++ii) {
    float2 tmp = __half22float2(inH2[ii]);
    scaled[ii * 2 + 0] = tmp.x * outputScale;
    scaled[ii * 2 + 1] = tmp.y * outputScale;
  }

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(scaled);
  // Make sure the type size is as expected.
  static_assert(sizeof(arrFp4) == NumEltsPerThread / 2 && sizeof(arrFp4) == sizeof(OutT));
  // Cast to E2m1.
  out = reinterpret_cast<OutT const&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to E2m1.
template <int NumEltsPerThread, typename OutT>
inline __device__ void convertFloatToE2m1(OutT& out,
                                          cutlass::float_e4m3_t& sfOut,
                                          float const (&in)[NumEltsPerThread],
                                          float sfScale) {

  // This function converts float to E2m1.
  // Inputs:
  // - in: The tensor holding float values.
  // - sfScale: The global scaling factor.
  // Outputs:
  // - return value: The casted E2m1 values.
  // - sfOut: The local scaling factor.

  // The number of threads per vector of 16 elements.
  int32_t constexpr NumThreadsPerVec = 16 / NumEltsPerThread;

  // Find the loacl amax.
  float localAmax = 0;
#pragma unroll
  for (int ii = 0; ii < 16; ii++) {
    localAmax = fmaxf(localAmax, fabsf(in[ii]));
  }

  // Get the absolute maximum among all 16 values.
  if constexpr (NumThreadsPerVec > 1) {
    static_assert(NumThreadsPerVec == 2, "Not supported.");
    localAmax = fmaxf(__shfl_xor_sync(uint32_t(-1), localAmax, 1), localAmax);
  }

  // Compute the sf and output scale.
  float outputScale;
  computeSfAndOutputScale(outputScale, sfOut, localAmax, sfScale);

  // Apply the output scale.
  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int ii = 0; ii < 16; ii++) {
    scaled[ii] = in[ii] * outputScale;
  }

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(scaled);
  // Cast to E2m1.
  out = reinterpret_cast<OutT&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to E2m1.
template <int NumEltsPerThread, typename OutT>
inline __device__ void convertFloatToE2m1(OutT& out, float const (&in)[NumEltsPerThread]) {
  // Apply the output scale.
  auto arrFp32{reinterpret_cast<cutlass::Array<float, NumEltsPerThread> const&>(in[0])};

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(arrFp32);
  // Cast to E2m1.
  out = reinterpret_cast<OutT&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to E2m1.
inline __device__ uint32_t convertFloat8ToE2m1(float in0,
                                               float in1,
                                               float in2,
                                               float in3,
                                               float in4,
                                               float in5,
                                               float in6,
                                               float in7) {

  cutlass::Array<float, 8> arrFp32{in0, in1, in2, in3, in4, in5, in6, in7};

  // Array of E2m1.
  auto arrFp4 = castArray<cutlass::float_e2m1_t>(arrFp32);
  // Cast to uint32_t.
  return reinterpret_cast<uint32_t&>(arrFp4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF tensor where
// dataRowIdx and dataBlkColIdx are the respective indices of the row and the block of 16 elts
// from the K dim in the tensor of data.
inline __device__ int64_t getSfOffset(int32_t dataRowIdx,
                                      int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {

  // The number of rows of SF per block.
  static int32_t constexpr NumRowsPerSfBlock = 128;
  // The number of cols of SF per block.
  static int32_t constexpr NumColsPerSfBlock = 4;
  // The size of each SF block.
  static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

  // The number of rows of data per SF block.
  static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
  // The number of cols of blocks of data per SF block.
  static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;

  // The row of the SF block in the SF tensor.
  int sfBlkRowIdx = dataRowIdx / NumDataRowsPerSfBlock;
  // The col of the SF block in the SF tensor.
  int sfBlkColIdx = dataBlkColIdx / NumDataBlkColsPerSfBlock;
  // The blocks are stored row-major in the tensor of scaling factors.
  int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumDataBlkColsPerSfBlock + sfBlkColIdx;

  // Find the row in the SF block.
  int sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % NumDataRowsPerSfBlock) / 32;
  // Find the col in the SF block.
  int sfColIdx = (dataBlkColIdx % 4);

  // Compute the offset in bytes.
  return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given the GMEM address of an output element, compute the offset of the corresponding scaling
// factor in the SF tensor. Optionally, a startTokenIndex can be provided if the first token is not
// the start token in the SF tensor. This is useful when inflight batching is enabled in TRT-LLM,
// where the context and generation output are stored as one output tensor. In this case, the
// generation output may not start with zero offset in the SF output tensor.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(int64_t gmemOffsetInBytes,
                                      int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  // The number of elements per sf.
  int32_t constexpr NumEltsPerSf = 16;
  // The GMEM offset of the output element.
  int64_t gmemOffset = gmemOffsetInBytes * 8 /*bits*/ / NumBitsPerElt;
  // The row/col indices of the corresponding SF element.
  int32_t sfRowIdx = gmemOffset / hiddenDim + startTokenIdx;
  int32_t sfColIdx = (gmemOffset % hiddenDim) / NumEltsPerSf;
  // Compute the SF offset.
  return getSfOffset(sfRowIdx, sfColIdx, hiddenDim / NumEltsPerSf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(tizheng): Refactor to track gmem offset instead of doing pointer subtraction.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(void const* gmemOutPtr,
                                      void const* gmemBasePtr,
                                      int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  return getSfOffset<NumBitsPerElt>(reinterpret_cast<char const*>(gmemOutPtr) -
                                      reinterpret_cast<char const*>(gmemBasePtr),
                                    hiddenDim,
                                    startTokenIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
