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

#include <cuda_runtime_api.h>
#include <cstdint>
#include <cute/container/tuple.hpp>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Structure with all the members from Cutlass counterpart class, to be interchangable with it
struct WorkTileInfo {
  int32_t M_idx;
  int32_t N_idx;
  int32_t L_idx;
  bool is_valid_tile;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// This schedule applies a simple euristic to make the load more balanced for workers.
// It relies on the observation that there is frequently a pattern of increasing (or decreasing)
// workload size as the flattened blockIdx increases.
// Assigning (workerId + i * numWorkers) at i-th iteration to a worker with ID workerId
// might result in some workers getting much more work than others, thus increasing kernel runtime.
// What we are doing here is alternating direction of worker assignement at each iteration:
// We assign workTile with this flattened blockIdx:
// ((i % 2 == 0) ? workerId : (numWorkers - 1 - workerId)) + i * numWorkers
class StaticScheduler {
public:
  __device__ StaticScheduler(int32_t workerId,
                             int32_t numWorkers,
                             int32_t maxM,
                             int32_t maxN,
                             int32_t maxL)
    : mMaxM(maxM)
    , mMaxN(maxN)
    , mMaxL(maxL)
    , mIsEvenStep(true) {

    int32_t divNL = workerId / mMaxM;
    mCurrentWorkTileInfo.M_idx = workerId - divNL * mMaxM;
    mCurrentWorkTileInfo.L_idx = divNL / mMaxN;
    mCurrentWorkTileInfo.N_idx = divNL - mCurrentWorkTileInfo.L_idx * mMaxN;
    mCurrentWorkTileInfo.is_valid_tile = mCurrentWorkTileInfo.L_idx < mMaxL;

    int evenStep = numWorkers * 2 - workerId * 2 - 1;
    int32_t divNLEvenStep = evenStep / mMaxM;
    mEvenStepM = evenStep - divNLEvenStep * mMaxM;
    mEvenStepL = divNLEvenStep / mMaxN;
    mEvenStepN = divNLEvenStep - mEvenStepL * mMaxN;

    int oddStep = numWorkers * 2 - evenStep;
    int32_t divNLOddStep = oddStep / mMaxM;
    mOddStepM = oddStep - divNLOddStep * mMaxM;
    mOddStepL = divNLOddStep / mMaxN;
    mOddStepN = divNLOddStep - mOddStepL * mMaxN;
  }

  // Advance to the next work tile.
  __device__ void advance_to_next_work() {
    mCurrentWorkTileInfo.M_idx += mIsEvenStep ? mEvenStepM : mOddStepM;
    mCurrentWorkTileInfo.N_idx += mIsEvenStep ? mEvenStepN : mOddStepN;
    mCurrentWorkTileInfo.L_idx += mIsEvenStep ? mEvenStepL : mOddStepL;
    if (mCurrentWorkTileInfo.M_idx >= mMaxM) {
      mCurrentWorkTileInfo.M_idx -= mMaxM;
      mCurrentWorkTileInfo.N_idx++;
    }
    if (mCurrentWorkTileInfo.N_idx >= mMaxN) {
      mCurrentWorkTileInfo.N_idx -= mMaxN;
      mCurrentWorkTileInfo.L_idx++;
    }
    mCurrentWorkTileInfo.is_valid_tile = mCurrentWorkTileInfo.L_idx < mMaxL;
    mIsEvenStep = !mIsEvenStep;
  }

  // Return the coordinates of the current work tile.
  __device__ cute::tuple<WorkTileInfo, bool> fetch_next_work(WorkTileInfo) {
    advance_to_next_work();
    return cute::make_tuple(mCurrentWorkTileInfo, true);
  }

  // Return the coordinates of the 1st work tile.
  __device__ WorkTileInfo initial_work_tile_info() { return mCurrentWorkTileInfo; }

private:
  // Logical grid size
  int32_t mMaxM, mMaxN, mMaxL;
  // How much we need to change each coordinate when advancing to the next work tile,
  // separately for even and odd steps.
  // They are precomputed to avoid integer division at each advance_to_next_work
  int32_t mEvenStepM, mEvenStepN, mEvenStepL;
  int32_t mOddStepM, mOddStepN, mOddStepL;
  // Is the current step even or odd?
  bool mIsEvenStep;
  // Coordinates of the current work tile.
  WorkTileInfo mCurrentWorkTileInfo;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
