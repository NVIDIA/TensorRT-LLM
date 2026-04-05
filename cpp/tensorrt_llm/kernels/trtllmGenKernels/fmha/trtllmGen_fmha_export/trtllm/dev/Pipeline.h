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

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumStages> class PipelineState {
public:
  // Ctor.
  inline __device__ PipelineState()
    : mIndex{0} {}

  // Get access to the index.
  inline __device__ int index() const { return mIndex; }

  // Increment the state.
  inline __device__ PipelineState& operator++() {
    mIndex = (mIndex == NumStages - 1) ? 0 : (mIndex + 1);
    return *this;
  }

private:
  // The state.
  int mIndex;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumStages, int NumThreads> class Pipeline {

  // Offset to the first full barrier.
  static constexpr int FullBarOffset{0};
  // Offset to the first empty barrier.
  static constexpr int EmptyBarOffset{NumStages};

  // The state.
  using State = PipelineState<NumStages>;

public:
  // Ctor.
  inline explicit __device__ Pipeline(int barId)
    : mBarId{barId} {}

  // Consumer release the barrier.
  inline __device__ void consumer_release(State const& state) {
    barrier_arrive(EmptyBarOffset + mBarId + state.index());
  }

  // Consumer test if they have to wait at the barrier. It does nothing.
  [[nodiscard]] inline __device__ int32_t consumer_test_wait(State const&) { return int32_t{0}; }

  // Consumer try to wait at the barrier. It does nothing.
  [[nodiscard]] inline __device__ int32_t consumer_try_wait(State const&) { return int32_t{0}; }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(State const& state, int32_t = int32_t{0}) {
    barrier_sync(FullBarOffset + mBarId + state.index());
  }

  // Producer acquire the barrier.
  inline __device__ void producer_acquire(State const& state, int32_t t = int32_t{0}) {
    if (t != int32_t{1}) {
      barrier_sync(EmptyBarOffset + mBarId + state.index());
    }
  }

  // Producer arrive at the barrier.
  inline __device__ void producer_commit(State const& state) {
    barrier_arrive(FullBarOffset + mBarId + state.index());
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(State const&) {}

  // Producer try to acquire the barrier. It does nothing.
  [[nodiscard]] inline __device__ int32_t producer_try_acquire(State const&) { return int32_t{0}; }

  // Decrement state index for n positions.
  [[nodiscard]] inline __device__ int32_t decr_state_index(State const& state, int32_t n) {
    auto index = (state.index() - n) % NumStages;
    return index < 0 ? index + NumStages : index;
  }

private:
  // Barrier arrive.
  inline __device__ void barrier_arrive(int barId) {
    asm volatile("barrier.cta.arrive %0, %1;" ::"r"(barId), "n"(NumThreads));
  }

  // Barrier wait.
  inline __device__ void barrier_sync(int barId) {
    asm volatile("barrier.cta.sync %0, %1;" ::"r"(barId), "n"(NumThreads));
  }

private:
  // The 1st barrier id. This pipeline uses 2 x NumStages named barriers.
  int mBarId;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
