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

#include <cutlass/numeric_conversion.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions to do the reduction in the epilogue of the slice-K slices via SMEM.
// A and B matrices are transposed, such that every numSlicesForSliceK of MmaK subtiles
// are mapped along M/N dimension for the MMA. During the epilogue warp `x` has only one
// element `Sxy` for N `y` that is valid. The final result `Sz` is the sum of `Sxz` across all
// warps.
//
//                          +-----++-----++-----++-----+++-----++-----++-----++-----+
//                          |                        T ileN                         |
//                          +-----++-----++-----++-----+++-----++-----++-----++-----+
//                          | N0  || N0  || N0  || N0  | | N1  || N2  || N3  || N4  |
//                          | K0  || K1  || K2  || K3  | | K0  || K1  || K2  || K3  |
//                          +-----++-----++-----++-----+++-----++-----++-----++-----+
//                         +----------------------------+----------------------------+
// +-------+-------++----+ |+-----+                     |+-----+                     |
// |       | Warp0 || W0 | || S00 |                     || S10 |                     |
// |       |       || K0 | ||     |                     ||     |                     |
// +       +-------++----+ |+-----++-----+              |+-----++-----+              |
// |       | Warp1 || W0 | |       | S01 |              |       | S11 |              |
// |       |       || K1 | |       |     |              |       |     |              |
// + TileM +-------++----+ |       +-----++-----+       |       +-----++-----+       |
// |       | Warp2 || W0 | |              | S02 |       |              | S12 |       |
// |       |       || K2 | |              |     |       |              |     |       |
// +       +-------++----+ |              +-----++-----+|              +-----++-----+|
// |       | Warp3 || W0 | |                     | S03 ||                     | S13 ||
// |       |       || K3 | |                     |     ||                     |     ||
// +-------+-------++----+ |                     +-----+|                     +-----+|
//                         +----------------------------+----------------------------+
//
// sliceKSaveToSmem writes corresponding whole TMEM to SMEM
// SMEM layout is [NumInstsPerCol * NumInstsPerRow * NumRepeats * 2, 64]
// Each column in SMEM contains 64 float values written by 32 threads -- in 16dp256b thread owns
// 2 elements per row per repeat. Additionally, to minimize bank conflicts, we apply 4 rows swizzle.
// To the written columns.
// Rows in SMEM are mapped as follows:
// +-----------------+-----------------+---------+---------+-------------------------------------+
// |                 |                 |         | Repeat0 | t0 | t0 | t1 | t1 | ... | t31 | t31 |
// |                 |                 |   Row0  | Repeat1 | t1 | t1 | t0 | t0 | ... | t30 | t30 |
// |                 |                 |         | ....... | ...                                 |
// +                 + NumInstsPerRow0 +---------+---------+-------------------------------------+
// |                 |                 |         | Repeat0 | t0 | t0 | t1 | t1 | ... | t31 | t31 |
// |                 |                 |   Row8  | Repeat1 | t1 | t1 | t0 | t0 | ... | t30 | t30 |
// |                 |                 |         | ....... | ...                                 |
// + NumInstsPerCol0 +-----------------+---------+---------+-------------------------------------+
// |                 |                 |         | Repeat0 | t0 | t0 | t1 | t1 | ... | t31 | t31 |
// |                 |                 |   Row0  | Repeat1 | t1 | t1 | t0 | t0 | ... | t30 | t30 |
// |                 |                 |         | ....... | ...                                 |
// +                 + NumInstsPerRow1 +---------+---------+-------------------------------------+
// |                 |                 |         | Repeat0 | t0 | t0 | t1 | t1 | ... | t31 | t31 |
// |                 |                 |   Row8  | Repeat1 | t1 | t1 | t0 | t0 | ... | t30 | t30 |
// |                 |                 |         | ....... | ...                                 |
// +-----------------+-----------------+---------+---------+-------------------------------------+
//
// In sliceKReduce, warp 0 reduces all subtiles and stores the result to the registers.
// TODO it is inefficient to do it this way -- ideally, we don't need to load whole TMEM and
// do need to store it to SMEM. Only valid elements Sxy must be stored.
// We do that because of 2 reasons:
// - We use 16dp256b, which is not the optimal layout for that problem. We should use 32dp32b with
// proper offset for each warp. This way we read out only valid subtiles Sxy.
// - Index to the valid element in the array of registers is depending on the warp idx
// and can't be figured out at compilation time. Thus, if saving only valid elems
// compiler puts accumulators, the result of the LDTM, to the local memory.
// It turns out to be more efficient to save whole tile and load only valid parts from SMEM.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumInstsPerCol, int NumInstsPerRow, int NumRepeats, int SrcN_>
inline __device__ void sliceKReduce16dp256bit(int warpIdx,
                                              int laneIdx,
                                              float* smemPtr,
                                              cutlass::Array<float, SrcN_, true>& acc) {
  auto constexpr numRowsPerWarpPerInst = 16;
  auto constexpr numThreadsPerCol = 4;
  auto constexpr numEltsPerThreadPerInstPerRepeat = 2;
  auto constexpr numColsPerInstPerRepeat = numThreadsPerCol * numEltsPerThreadPerInstPerRepeat;
  // Number of rows in slice.
  auto constexpr numRowsInSlice = NumInstsPerCol * numRowsPerWarpPerInst;
  // Number of cols in slices.
  auto constexpr numColsInSlice = NumInstsPerRow * NumRepeats * numColsPerInstPerRepeat;
  // Number of elements produced by a warp.
  auto constexpr numEltsInSlice = numRowsInSlice * numColsInSlice;
  // Number of original tile columns per output valid element.
  auto constexpr numColsPerValidElem = 4;
  // Number of columns in SMEM tile.
  auto constexpr numSmemCols = 32 * numEltsPerThreadPerInstPerRepeat;

  // Warp0 reduces all slices
  if (warpIdx == 0) {
#pragma unroll
    for (int ii = 0; ii < NumInstsPerCol; ++ii) {
#pragma unroll
      for (int jj = 0; jj < NumInstsPerRow; ++jj) {
// Number of output elements is 4 times less (as we have 4 slices)
#pragma unroll
        for (int kk = 0; kk < NumRepeats / 4; ++kk) {
          // Reduced values
          cutlass::Array<float, 2> sum[2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
          // Loop over slices
          for (int si = 0; si < 4; ++si) {
            // For warp 0 and 2, we use the 1st element.
            // For warp 1 and 3, we use the 2nd element.
            auto validElemIndexInCol = si % 2;
            // Valid threads are 0 and 2 for warps 0 and 1.
            // And threads 1 and 3 are for warps 2 and 3.
            auto validElemThreadIdx = si / 2;

            // Row index in the smem.
            // Each thread loads 2 rows per instruction (row 0 and row 8).
            // Check the layout diagram above to understand the mapping.
            auto const srcRowIdx0 = ii * NumInstsPerRow * NumRepeats * 2 + jj * NumRepeats * 2 +
                                    0 * NumRepeats + ((laneIdx % numThreadsPerCol) + kk * 4);
            auto const srcRowIdx1 = ii * NumInstsPerRow * NumRepeats * 2 + jj * NumRepeats * 2 +
                                    1 * NumRepeats + ((laneIdx % numThreadsPerCol) + kk * 4);

            // Col index in the smem. Each lane loads 2 elements from col k and k + 2 (col width is
            // uint64_t).
            auto const srcColIdx0 =
              (laneIdx / numThreadsPerCol) * numThreadsPerCol + validElemThreadIdx;
            auto const srcColIdx1 =
              (laneIdx / numThreadsPerCol) * numThreadsPerCol + validElemThreadIdx + 2;
            // Swizzle cols to minimize banl conflicts.
            auto const swizzledColIdx00 =
              (srcColIdx0 ^ (srcRowIdx0 % numThreadsPerCol)) * 2 + validElemIndexInCol;
            auto const swizzledColIdx01 =
              (srcColIdx1 ^ (srcRowIdx0 % numThreadsPerCol)) * 2 + validElemIndexInCol;
            auto const swizzledColIdx10 =
              (srcColIdx0 ^ (srcRowIdx1 % numThreadsPerCol)) * 2 + validElemIndexInCol;
            auto const swizzledColIdx11 =
              (srcColIdx1 ^ (srcRowIdx1 % numThreadsPerCol)) * 2 + validElemIndexInCol;

            // Src indices in the smem to the element in `si`th slice.
            int const srcIdx00 = si * numEltsInSlice + srcRowIdx0 * numSmemCols + swizzledColIdx00;
            int const srcIdx01 = si * numEltsInSlice + srcRowIdx0 * numSmemCols + swizzledColIdx01;
            int const srcIdx10 = si * numEltsInSlice + srcRowIdx1 * numSmemCols + swizzledColIdx10;
            int const srcIdx11 = si * numEltsInSlice + srcRowIdx1 * numSmemCols + swizzledColIdx11;

            // Loading elements from SMEM.
            auto val00 = smemPtr[srcIdx00];
            auto val01 = smemPtr[srcIdx01];
            auto val10 = smemPtr[srcIdx10];
            auto val11 = smemPtr[srcIdx11];

            cutlass::Array<float, 2> val0 = {val00, val01};
            cutlass::Array<float, 2> val1 = {val10, val11};

            // Do the reduction using FFMA2.
            cute::add(sum[0], sum[0], val0);
            cute::add(sum[1], sum[1], val1);
          }

          // Save reduced vals back to accumulators at the "original" positions.
          // It is done that way to keep the rest of the epilogue assuming 16dp256b_xR
          // structure unmodified and enable more modular code that allows to combine
          // slice-k with split-k, AR, etc.
          auto accIdx = ii * NumInstsPerRow * NumRepeats * 4 + jj * NumRepeats * 4 + kk * 4;
          acc[accIdx + 0] = sum[0][0];
          acc[accIdx + 1] = sum[0][1];
          acc[accIdx + 2] = sum[1][0];
          acc[accIdx + 3] = sum[1][1];
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumInstsPerCol, int NumInstsPerRow, int NumRepeats, int SrcN_>
inline __device__ void sliceKSaveToSmem16dp256bit(int warpIdx,
                                                  int laneIdx,
                                                  float* smemPtr,
                                                  cutlass::Array<float, SrcN_, true>& acc) {
  auto constexpr numRowsPerWarpPerInst = 16;
  auto constexpr numThreadsPerCol = 4;
  auto constexpr numEltsPerThreadPerInstPerRepeat = 2;
  auto constexpr numColsPerInstPerRepeat = numThreadsPerCol * numEltsPerThreadPerInstPerRepeat;
  // Number of rows in slice.
  auto constexpr numRowsInSlice = NumInstsPerCol * numRowsPerWarpPerInst;
  // Number of cols in slices.
  auto constexpr numColsInSlice = NumInstsPerRow * NumRepeats * numColsPerInstPerRepeat;
  // Number of elements produced by a warp.
  auto constexpr numEltsInSlice = numRowsInSlice * numColsInSlice;
  // Number of columns in SMEM tile.
  auto constexpr numSmemCols = 32 * numEltsPerThreadPerInstPerRepeat;

  // Valid threads are 0 and 2 for warps 0 and 1.
  // And threads 1 and 3 are for warps 2 and 3.
  auto validElemThreadIdx = warpIdx / 2;

// Write slice to SMEM.
#pragma unroll
  for (int ii = 0; ii < NumInstsPerCol; ++ii) {
#pragma unroll
    for (int jj = 0; jj < NumInstsPerRow; ++jj) {
      if ((laneIdx % 2) == validElemThreadIdx) {
#pragma unroll
        for (int kk = 0; kk < NumRepeats; ++kk) {
#pragma unroll
          for (int mi = 0; mi < 2; ++mi) {
            // Index of the acc in the registers array.
            int const srcIdx =
              ii * NumInstsPerRow * NumRepeats * 4 + jj * NumRepeats * 4 + kk * 4 + mi * 2;

            // Read elem from accs.
            auto const elem = reinterpret_cast<uint64_t*>(&acc)[srcIdx / 2];

            // Row index in the dst SMEM.
            auto const dstRowIdx =
              ii * NumInstsPerRow * NumRepeats * 2 + jj * NumRepeats * 2 + mi * NumRepeats + kk;
            // Row index in the dst SMEM.
            auto const dstColIdx = laneIdx;
            // Swizzle column to avoid bank conflicts.
            auto const swizzledCol = (dstColIdx ^ (dstRowIdx % numThreadsPerCol)) * 2;

            // Dst indices in the SMEM.
            auto const dstIdx = warpIdx * numEltsInSlice + dstRowIdx * numSmemCols + swizzledCol;

            // Saving elements to SMEM.
            reinterpret_cast<uint64_t*>(smemPtr)[dstIdx / 2] = elem;
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
