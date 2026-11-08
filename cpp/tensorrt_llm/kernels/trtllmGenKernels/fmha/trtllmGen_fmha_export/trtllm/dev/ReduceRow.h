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
#include "ReduceCol.h"
#include "CutlassBarrier.h"

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// This code computes the max for each row of a tile NumRows x NumCols distributed inside a
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
inline __device__ void reduceRowMax16dp256bit(float (&dst)[DstN_],
                                              cutlass::Array<float, SrcN_> const& src,
                                              float* smemMaxPtr,
                                              int numThreadsPerWarpGrp,
                                              int warpGrpThreadIdx,
                                              int namedBarId) {

  // Make sure the number of dst registers is as expected.
  static_assert(DstN_ == NumInstsPerCol * 2);
  // Make sure the number of src registers is as expected.
  static_assert(SrcN_ == NumInstsPerCol * NumInstsPerRow * NumRepeats * 4);

  // For each row, each thread computes the abs max amongst its registers. Each LDTM.16dp256bit
  // (with repeats) loads elements from two rows in TMEM. Registers 0, 1, 4, 5, 8, 9, ... map to the
  // first row when registers 2, 3, 6, 7, 10, 11, ... map to the second row.

  // Iterate over the instructions in the column - produce two max per iteration.
  for (int mi = 0; mi < NumInstsPerCol; ++mi) {

    // The indices of the rows.
    int const row0 = mi * 2 + 0;
    int const row1 = mi * 2 + 1;

    // The number of registers per repeat. LDTM.16dp256bit loads 4 registers per repeat.
    int constexpr NumRegsPerRepeat = 4;
    // The number of registers per LDTM.16dp256bit with repeats.
    int constexpr NumRegsPerInst = NumRepeats * NumRegsPerRepeat;
    // The number of registers per row of instructions.
    auto constexpr NumRegsPerInstRow = NumInstsPerRow * NumRegsPerInst;

    // Iterate over the instructions in the row.
    for (int ni = 0; ni < NumInstsPerRow; ++ni) {
      for (int ri = 0; ri < NumRepeats; ++ri) {

        // The indices of the source.
        int const src0 = mi * NumRegsPerInstRow + ni * NumRegsPerInst + ri * NumRegsPerRepeat + 0;
        int const src1 = mi * NumRegsPerInstRow + ni * NumRegsPerInst + ri * NumRegsPerRepeat + 1;
        int const src2 = mi * NumRegsPerInstRow + ni * NumRegsPerInst + ri * NumRegsPerRepeat + 2;
        int const src3 = mi * NumRegsPerInstRow + ni * NumRegsPerInst + ri * NumRegsPerRepeat + 3;

        // Within an instruction registers 0 and 1 are for the 1st row.
        dst[row0] = fmaxf(dst[row0], AbsoluteMax ? fabsf(src[src0]) : src[src0]);
        dst[row0] = fmaxf(dst[row0], AbsoluteMax ? fabsf(src[src1]) : src[src1]);

        // Within an instruction registers 2 and 3 are for the 2nd row.
        dst[row1] = fmaxf(dst[row1], AbsoluteMax ? fabsf(src[src2]) : src[src2]);
        dst[row1] = fmaxf(dst[row1], AbsoluteMax ? fabsf(src[src3]) : src[src3]);
      }
    }
  }

  // For each row, we have to compute the max accross the 4 lanes of the quad.
  for (int mi = 0; mi < DstN_; ++mi) {
    for (int laneMask = 2; laneMask >= 1; laneMask /= 2) {
      dst[mi] = fmaxf(__shfl_xor_sync(uint32_t{0xffffffff}, dst[mi], laneMask), dst[mi]);
    }
  }

  // The lane inside the warp-group.
  int const laneIdx{warpGrpThreadIdx % 32};
  // The index of the quad in the warp.
  int const quadIdx{laneIdx / 4};
  // The index of the warp.
  int const warpIdx{warpGrpThreadIdx / 32};
  // Number of rows loaded by a warp. Each quad owns 2 rows per instruction.
  // There are NumInstsPerCol instructions.
  int const numRowsPerWarp = DstN_ * 8;
  // The shared memory location where to perform the atomic max.
  float* smemMax = &smemMaxPtr[warpIdx * numRowsPerWarp + quadIdx * DstN_];

  // Compute the max within warp-groups using atomics in shared memory.
  // Only the first thread of the quad writes to smem.
  if (warpGrpThreadIdx % 4 == 0) {
    for (int ii = 0; ii < DstN_; ++ii) {
      atomicMaxFloat(&smemMax[ii], dst[ii]);
    }
  }

  // Sync the threads in the warpgroup.
  trtllm::dev::CutlassNamedBarrier::sync(numThreadsPerWarpGrp, namedBarId);

  // Fetch the values from shared memory and populate the output buffer.
  loadMaxFromSmem(dst, reinterpret_cast<uint32_t const*>(smemMax));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The reduce operation.
template <bool IsMax> struct ReduceOp {};

template <> struct ReduceOp<true> {
  inline __device__ float operator()(float a, float b) { return fmaxf(a, b); }
};

template <> struct ReduceOp<false> {
  inline __device__ float operator()(float a, float b) { return a + b; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The reduce operation for the 2x2 warp-group layout of the M=128 2CTA Utcmma instruction.
template <bool IsMax>
inline __device__ void reduceWarpGrp2x2(float& val,
                                        float* smem,
                                        int warpGrpThreadIdx,
                                        int namedBarId) {

  //
  // The 2x2 warp-group layout.
  // Need to reduce the values among the 2 warps (warp0 and warp2, warp1 and warp3).
  //
  // warp0   warp2
  // warp1   warp3
  //

  // The warpIdx in the warp-group.
  int const warpIdx{warpGrpThreadIdx / 32};
  // The smem rowIdx.
  int const smemRowIdx{warpGrpThreadIdx % 64};
  // The smem colIdx.
  int const smemColIdx{warpGrpThreadIdx / 64};

  // The named barrier to sync lanes that belong to the same row of warps (warps 0/2 and warps 1/3,
  // resp).
  int const namedBarrierId{(warpIdx % 2) + namedBarId};

  // Store to smem.
  smem[smemRowIdx * 2 + smemColIdx] = val;

  // Sync the threads among the row of warps.
  trtllm::dev::CutlassNamedBarrier::sync(64, namedBarrierId);

  // Load from smem, and perform the reduction with the other warp in the pair.
  val = ReduceOp<IsMax>{}(val, smem[smemRowIdx * 2 + (smemColIdx ^ 1)]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
