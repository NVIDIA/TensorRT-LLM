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

// NOTE: Using a guard here since NVRTC fails if using #pragma once
#ifndef TRTLLM_DEV_REDUCECOL_H
#define TRTLLM_DEV_REDUCECOL_H

#include <cutlass/numeric_conversion.h>
#include "CutlassBarrier.h"

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following functions are helpers to implement an atomicMax in shared memory for float.
//
// Cutlass implements atomicMax for float using the following method:
//
// if (signbit(x)) {
//   atomicMin(smemPtr, reinterpret_cast<uint32_t const&>(x));
// } else {
//   atomicMax(smemPtr, reinterpret_cast<int32_t const&>(x));
// }
//
// This code leads to multiple branches and two ATOMS.* per floating-point value. We avoid the
// branching and stick to one ATOMS.MAX per floating-point value using a trick described in
// http://stereopsis.com/radix.html and implemented in CUB's radix sort, for example.

////////////////////////////////////////////////////////////////////////////////////////////////////

// XOR the sign bit and take two-complement if the sign bit was set.

inline __device__ uint32_t floatToUInt32ForAtomicMax(float f) {
  uint32_t mask = -int32_t(reinterpret_cast<uint32_t&>(f) >> 31) | 0x80000000;
  return reinterpret_cast<uint32_t&>(f) ^ mask;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Undo the two-complement encoding and reset the sign bit.

inline __device__ float uint32ToFloatForAtomicMax(uint32_t u) {
  uint32_t mask = ((u >> 31) - 1) | 0x80000000;
  uint32_t f = (u ^ mask);
  return reinterpret_cast<float&>(f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Perform an atomicMax for float using atomicMax for unsigned int32.

inline __device__ void atomicMaxFloat(float* smemPtr, float val) {
  atomicMax((uint32_t*)smemPtr, floatToUInt32ForAtomicMax(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following functions load max values (stored as uint32_t) from shared memory and convert
// those values back to floats (see above conversion code and the link to the trick).

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void loadMaxFromSmem(float (&regsMax)[2], uint32_t const* smemMax) {
  // Issue the LDS.64 to load from shared memory.
  uint2 vals = *reinterpret_cast<uint2 const*>(smemMax);
  // Convert.
  regsMax[0] = uint32ToFloatForAtomicMax(vals.x);
  regsMax[1] = uint32ToFloatForAtomicMax(vals.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void loadMaxFromSmem(float (&regsMax)[4], uint32_t const* smemMax) {
  // Issue the LDS.128 to load from shared memory.
  uint4 vals = *reinterpret_cast<uint4 const*>(smemMax);
  // Convert.
  regsMax[0] = uint32ToFloatForAtomicMax(vals.x);
  regsMax[1] = uint32ToFloatForAtomicMax(vals.y);
  regsMax[2] = uint32ToFloatForAtomicMax(vals.z);
  regsMax[3] = uint32ToFloatForAtomicMax(vals.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void loadMaxFromSmem(float (&regsMax)[8], uint32_t const* smemMax) {
  // Issue 2 LDS.128 loads.
  for (int ii = 0; ii < 2; ++ii) {
    loadMaxFromSmem(reinterpret_cast<float(&)[4]>(regsMax[ii * 4]), smemMax + ii * 4);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void loadMaxFromSmem(float (&regsMax)[16], uint32_t const* smemMax) {
  // Issue 4 LDS.128 loads.
  for (int ii = 0; ii < 4; ++ii) {
    loadMaxFromSmem(reinterpret_cast<float(&)[4]>(regsMax[ii * 4]), smemMax + ii * 4);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void loadMaxFromSmem(float (&regsMax)[32], uint32_t const* smemMax) {
  // Issue 8 LDS.128 loads.
  for (int ii = 0; ii < 8; ++ii) {
    loadMaxFromSmem(reinterpret_cast<float(&)[4]>(regsMax[ii * 4]), smemMax + ii * 4);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void loadMaxFromSmem(float (&regsMax)[64], uint32_t const* smemMax) {
  // Issue 16 LDS.128 loads.
  for (int ii = 0; ii < 16; ++ii) {
    loadMaxFromSmem(reinterpret_cast<float(&)[4]>(regsMax[ii * 4]), smemMax + ii * 4);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This code computes the max for each column of a tile NumRows x NumCols distributed inside a
// warp-group. In this warp-group, there are 4 consecutive threads per row and 4x8 rows of threads.
// That's typically the result of loading the accumulators from TMEM using the 16DP/256b version of
// LDTM (LDTM.16dp256bit).
//
// Distribution of threads:
//
// Row ..0: ..0 ..1 ..2 ..3
// Row ..1: ..4 ..5 ..6 ..7
// Row ..2: ..8 ..9 .10 .11
// Row ..3: .12 .13 .14 .15
// Row ..4: .16 .17 .18 .19
// Row ...: ...
// Row ..7: .28 .29 .30 .31
// Row ..8: .32 .33 .34 .35
// Row ...: ...
// Row .31: 124 125 126 127
//
// Distribution of values (even if it does not matter too much). The indices in the following dia-
// gram are the indices of the thread owners.
//
// Row ..0: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row ..1: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ..2: ..8 ..8 ..9 ..9 .10 .10 .11 .11
// Row ..3: .12 .12 .13 .13 .14 .14 .15 .15
// Row ..4: .16 .16 .17 .17 .18 .18 .19 .19
// Row ...: ... ...
// Row ..7: .28 .28 .29 .29 .30 .30 .31 .31
// Row ..8: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row ..9: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .16: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row .17: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .24: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row .25: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .32: .32 .32 .33 .33 .34 .34 .35 .35
// Row ...: ... ...
// Row .64: .64 .64 .65 .65 .66 .66 .67 .67
// Row ...: ... ...
// Row .96: .96 .96 .97 .97 .98 .98 .99 .99
// Row ...: ... ...
// Row 127: 124 124 125 125 126 126 127 127

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumInstsPerCol,
          int NumInstsPerRow,
          int NumRepeats,
          bool AbsoluteMax,
          int DstN_,
          int SrcN_>
inline __device__ void reduceColMax16dp256bit(float (&dst)[DstN_],
                                              cutlass::Array<float, SrcN_> const& src) {

  // Make sure the number of dst registers is as expected.
  static_assert(DstN_ == NumInstsPerRow * NumRepeats * 2);
  // Make sure the number of src registers is as expected.
  static_assert(SrcN_ == NumInstsPerCol * NumInstsPerRow * NumRepeats * 4);

  for (int ii = 0; ii < NumInstsPerCol; ++ii) {
    for (int jj = 0; jj < NumInstsPerRow; ++jj) {
      for (int kk = 0; kk < NumRepeats; ++kk) {
        // The indices of the columns.
        int const dstIdx0 = jj * NumRepeats * 2 + kk * 2 + 0;
        int const dstIdx1 = jj * NumRepeats * 2 + kk * 2 + 1;

        // The indices of the source.
        int const srcIdx0 = ii * NumInstsPerRow * NumRepeats * 4 + jj * NumRepeats * 4 + kk * 4 + 0;
        int const srcIdx1 = ii * NumInstsPerRow * NumRepeats * 4 + jj * NumRepeats * 4 + kk * 4 + 1;
        int const srcIdx2 = ii * NumInstsPerRow * NumRepeats * 4 + jj * NumRepeats * 4 + kk * 4 + 2;
        int const srcIdx3 = ii * NumInstsPerRow * NumRepeats * 4 + jj * NumRepeats * 4 + kk * 4 + 3;

        // Registers srcIdx + 0 and srcIdx + 2 map to the same column.
        dst[dstIdx0] = fmaxf(dst[dstIdx0], AbsoluteMax ? fabsf(src[srcIdx0]) : src[srcIdx0]);
        dst[dstIdx0] = fmaxf(dst[dstIdx0], AbsoluteMax ? fabsf(src[srcIdx2]) : src[srcIdx2]);

        // Registers srcIdx + 1 and srcIdx + 3 map to the same column.
        dst[dstIdx1] = fmaxf(dst[dstIdx1], AbsoluteMax ? fabsf(src[srcIdx1]) : src[srcIdx1]);
        dst[dstIdx1] = fmaxf(dst[dstIdx1], AbsoluteMax ? fabsf(src[srcIdx3]) : src[srcIdx3]);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumColsPerThread>
inline __device__ void reduceColMax(float (&regsMax)[NumColsPerThread],
                                    float* smemMaxPtr,
                                    int numWarpGrpThreads,
                                    int warpGrpThreadIdx,
                                    int namedBarId) {

  // The number of threads per row.
  int constexpr NumThreadsPerRow{4};
  // The number of threads participating in atomicMax. It's an optim between SHFL and ATOMS.
  int constexpr NumThreadsPerAtomicMax{8};

  // Then, reduce maximum values among the threads in the warp.
  for (int col = 0; col < NumColsPerThread; ++col) {
    for (int laneMask = 16; laneMask >= NumThreadsPerAtomicMax; laneMask /= 2) {
      regsMax[col] =
        fmaxf(__shfl_xor_sync(uint32_t{0xffffffff}, regsMax[col], laneMask), regsMax[col]);
    }
  }

  // The lane inside the warp-group.
  int const laneIdx{warpGrpThreadIdx % 32};
  // The column inside the warp-group.
  int const colIdx{warpGrpThreadIdx % NumThreadsPerRow};

  // The shared memory location where to perform the atomic max.
  float* smemMax = &smemMaxPtr[colIdx * NumColsPerThread];

  // Compute the warp-group maxes using atomics in shared memory.
  for (int col = 0; col < NumColsPerThread; ++col) {
    if (laneIdx < NumThreadsPerAtomicMax) {
      atomicMaxFloat(&smemMax[col], regsMax[col]);
    }
  }

  // Sync the threads in the warpgroup.
  trtllm::dev::CutlassNamedBarrier::sync(numWarpGrpThreads, namedBarId);

  // Fetch the values from shared memory and populate the output buffer.
  loadMaxFromSmem(regsMax, reinterpret_cast<uint32_t const*>(smemMax));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumFloat2s>
inline __device__ void loadSumFromSmem(float2 (&sum)[NumFloat2s], float const* smem) {

  // The number of float4s to load.
  int constexpr NumFloat4s{NumFloat2s / 2};

  // Load the sums from shared memory.
  for (int ii = 0; ii < NumFloat4s; ++ii) {
    float4 tmp = reinterpret_cast<float4 const*>(smem)[ii];
    sum[ii * 2 + 0].x = tmp.x;
    sum[ii * 2 + 0].y = tmp.y;
    sum[ii * 2 + 1].x = tmp.z;
    sum[ii * 2 + 1].y = tmp.w;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void loadSumFromSmem(float2 (&sum)[1], float const* smem) {
  sum[0] = *reinterpret_cast<float2 const*>(smem);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumFloat2s>
inline __device__ void storeSumToSmem(float* smem, float2 const (&sum)[NumFloat2s]) {

  // The number of float4s to store.
  int constexpr NumFloat4s{NumFloat2s / 2};

  // Store the sums to shared memory.
  for (int ii = 0; ii < NumFloat4s; ++ii) {
    float4 tmp;
    tmp.x = sum[ii * 2 + 0].x;
    tmp.y = sum[ii * 2 + 0].y;
    tmp.z = sum[ii * 2 + 1].x;
    tmp.w = sum[ii * 2 + 1].y;
    reinterpret_cast<float4*>(smem)[ii] = tmp;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void storeSumToSmem(float* smem, float2 const (&sum)[1]) {
  reinterpret_cast<float2*>(smem)[0] = sum[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This code computes the sum for each column of a tile NumRows x NumCols distributed inside a
// warp-group. In this warp-group, there are 4 consecutive threads per row and 4x8 rows of threads.
// That's typically the result of loading the accumulators from TMEM using the 16DP/256b version of
// LDTM (LDTM.16dp256bit).
//
// Distribution of threads:
//
// Row ..0: ..0 ..1 ..2 ..3
// Row ..1: ..4 ..5 ..6 ..7
// Row ..2: ..8 ..9 .10 .11
// Row ..3: .12 .13 .14 .15
// Row ..4: .16 .17 .18 .19
// Row ...: ...
// Row ..7: .28 .29 .30 .31
// Row ..8: .32 .33 .34 .35
// Row ...: ...
// Row .31: 124 125 126 127
//
// Distribution of values (even if it does not matter too much). The indices in the following dia-
// gram are the indices of the thread owners.
//
// Row ..0: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row ..1: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ..2: ..8 ..8 ..9 ..9 .10 .10 .11 .11
// Row ..3: .12 .12 .13 .13 .14 .14 .15 .15
// Row ..4: .16 .16 .17 .17 .18 .18 .19 .19
// Row ...: ... ...
// Row ..7: .28 .28 .29 .29 .30 .30 .31 .31
// Row ..8: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row ..9: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .16: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row .17: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .24: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row .25: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .32: .32 .32 .33 .33 .34 .34 .35 .35
// Row ...: ... ...
// Row .64: .64 .64 .65 .65 .66 .66 .67 .67
// Row ...: ... ...
// Row .96: .96 .96 .97 .97 .98 .98 .99 .99
// Row ...: ... ...
// Row 127: 124 124 125 125 126 126 127 127
//
// The function is "unrolled" to deal with two instances per warp-group (and reduce the number of
// synchronization calls).

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumColsPerThread>
inline __device__ void reduceColSum(float* regsSum,
                                    float* smemSumPtr,
                                    int32_t numWarps,
                                    int32_t numWarpGrpThreads,
                                    int32_t warpGrpThreadIdx,
                                    int32_t namedBarId) {

  // Number of threads per row.
  int32_t constexpr NumThreadsPerRow{4};
  // The number of columns stored by each warp - the warps work on the same columns.
  int32_t constexpr NumColsPerWarp{NumThreadsPerRow * NumColsPerThread};

  // Declare explicit float2 arrays.
  float2 sums[NumColsPerThread / 2];

  // Issue the copies to populate those float2 arrays (the compiler will remove those copies).
  for (int32_t ii = 0; ii < NumColsPerThread / 2; ++ii) {
    sums[ii].x = regsSum[2 * ii + 0];
    sums[ii].y = regsSum[2 * ii + 1];
  }

  // Reduce sums across threads in the warp.
  for (int32_t laneMask = 16; laneMask >= NumThreadsPerRow; laneMask /= 2) {
    for (int32_t ii = 0; ii < NumColsPerThread / 2; ++ii) {
      // Shuffle the values
      float2 other;
      other.x = __shfl_xor_sync(uint32_t{0xffffffff}, sums[ii].x, laneMask);
      other.y = __shfl_xor_sync(uint32_t{0xffffffff}, sums[ii].y, laneMask);

      // Perform the summations using explicit FADD2s.
      cute::add(sums[ii], sums[ii], other);
    }
  }

  // The warp inside the warp-group.
  int const warpIdx{warpGrpThreadIdx / 32};
  // The lane inside the warp-group.
  int const laneIdx{warpGrpThreadIdx % 32};
  // The column inside the warp-group.
  int const colIdx{warpGrpThreadIdx % NumThreadsPerRow};

  // The shared memory location where to store the sums of each warp.
  float* smemSumDst = &smemSumPtr[warpIdx * NumColsPerWarp + colIdx * NumColsPerThread];

  // Store the warp sums to shared memory. Only the first quad of each warp issues STS operations.
  if (laneIdx < NumThreadsPerRow) {
    storeSumToSmem(&smemSumDst[0], sums);
  }

  // Sync the threads in the warpgroup.
  trtllm::dev::CutlassNamedBarrier::sync(numWarpGrpThreads, namedBarId);

  // The shared memory locations where to read the sums from.
  float const* smemSumSrc = &smemSumPtr[colIdx * NumColsPerThread];

  // Fetch the sums from the 1st warp. All the threads perform the load.
  loadSumFromSmem(sums, smemSumSrc);

  // Iterate over the other warps to fetch the other sums and add them to the existing sums.
  for (int ii = 1; ii < numWarps; ++ii) {
    float2 other[NumColsPerThread / 2];
    loadSumFromSmem(other, &smemSumSrc[ii * NumColsPerWarp]);

    // Perform the sums - use explicit FADD2.
    for (int jj = 0; jj < NumColsPerThread / 2; ++jj) {
      cute::add(sums[jj], sums[jj], other[jj]);
    }
  }

  // Store back to registers. The compiler will remove the MOVs.
  for (int ii = 0; ii < NumColsPerThread / 2; ++ii) {
    regsSum[2 * ii + 0] = sums[ii].x;
    regsSum[2 * ii + 1] = sums[ii].y;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm

#endif // TRTLLM_DEV_REDUCECOL_H
