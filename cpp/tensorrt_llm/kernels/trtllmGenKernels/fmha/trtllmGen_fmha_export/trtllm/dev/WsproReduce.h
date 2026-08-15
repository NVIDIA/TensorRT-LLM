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


/* Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    *  Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *    *  Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    *  Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Adapted from WIP CUB PR: https://github.com/NVIDIA/cccl/pull/7692
// (hence the above BSD 3-Clause license)

// NOTE: Using a guard here since NVRTC fails if using #pragma once
#ifndef TRTLLM_DEV_WSPROREDUCE_H
#define TRTLLM_DEV_WSPROREDUCE_H

#include <cstdint>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// WsproReduce provides WSPRO (Warp Shuffle Parallel Reduction Optimization) based
// batched parallel reduction of floats partitioned across a CUDA thread warp.
//
// NumGroups - Number of independent reductions to perform in batch
// GroupSize - Number of threads per reduction group (must be a power-of-two)
template <int NumGroups, int GroupSize> struct WsproReduce {
  static_assert((GroupSize & (GroupSize - 1)) == 0, "GroupSize must be a power of two");
  static_assert(GroupSize > 1 && GroupSize <= 32, "GroupSize must be in the range [2, 32]");
  static_assert(NumGroups >= 1, "NumGroups must be >= 1");

  static constexpr int max_out_per_thread = (NumGroups + GroupSize - 1) / GroupSize;

  int logical_lane_id;
  uint32_t lane_mask;
  int group_stride;

  __device__ __forceinline__ WsproReduce(int lane_id_,
                                         uint32_t lane_mask_ = 0xffffffffu,
                                         int group_stride_ = 1)
    : logical_lane_id(lane_id_)
    , lane_mask(lane_mask_)
    , group_stride(group_stride_) {}

  template <typename ReductionOp>
  __device__ __forceinline__ void Reduce(float const (&inputs)[NumGroups],
                                         float (&outputs)[max_out_per_thread],
                                         ReductionOp reduction_op) {
    float values[NumGroups];
#pragma unroll
    for (int i = 0; i < NumGroups; ++i) {
      values[i] = inputs[i];
    }

    ReduceInplace(values, reduction_op);

#pragma unroll
    for (int i = 0; i < max_out_per_thread; ++i) {
      const int group_idx = i * GroupSize + logical_lane_id;
      if (group_idx < NumGroups) {
        outputs[i] = values[i];
      }
    }
  }

  template <int LeftIdx, int StrideInterReduce = GroupSize, typename ReductionOp>
  __device__ __forceinline__ void RecurseReductionTree(float* inputs, ReductionOp reduction_op) {
    constexpr int stride_intra_reduce = StrideInterReduce / 2;
    constexpr int right_idx = LeftIdx + stride_intra_reduce;
    constexpr bool base_case = stride_intra_reduce == 1;
    if constexpr (!base_case) {
      RecurseReductionTree<LeftIdx, stride_intra_reduce>(inputs, reduction_op);
      if constexpr (right_idx < NumGroups) {
        RecurseReductionTree<right_idx, stride_intra_reduce>(inputs, reduction_op);
      }
    }
    float left_value = inputs[LeftIdx];
    constexpr int safe_right_idx = right_idx < NumGroups ? right_idx : NumGroups - 1;
    float right_value = inputs[safe_right_idx];
    const bool is_left_lane = logical_lane_id % StrideInterReduce < stride_intra_reduce;
    if (is_left_lane) {
      float tmp = left_value;
      left_value = right_value;
      right_value = tmp;
    }
    left_value = __shfl_xor_sync(lane_mask, left_value, stride_intra_reduce * group_stride);
    inputs[LeftIdx] = reduction_op(left_value, right_value);
  }

  template <typename ReductionOp>
  __device__ __forceinline__ void ReduceInplace(float (&inputs)[NumGroups],
                                                ReductionOp reduction_op) {
    reduceAllGroups<0>(inputs, reduction_op);
#pragma unroll
    for (int i = 1; i < max_out_per_thread; ++i) {
      if (i * GroupSize < NumGroups) {
        inputs[i] = inputs[i * GroupSize];
      }
    }
  }

  struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
  };

  struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
  };

  struct MinOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return fminf(a, b); }
  };

  __device__ __forceinline__ void Sum(float const (&inputs)[NumGroups],
                                      float (&outputs)[max_out_per_thread]) {
    Reduce(inputs, outputs, SumOp{});
  }

  __device__ __forceinline__ void Max(float const (&inputs)[NumGroups],
                                      float (&outputs)[max_out_per_thread]) {
    Reduce(inputs, outputs, MaxOp{});
  }

  __device__ __forceinline__ void Min(float const (&inputs)[NumGroups],
                                      float (&outputs)[max_out_per_thread]) {
    Reduce(inputs, outputs, MinOp{});
  }

private:
  template <int Idx, typename ReductionOp>
  __device__ __forceinline__ void reduceAllGroups(float* inputs, ReductionOp reduction_op) {
    if constexpr (Idx < max_out_per_thread) {
      RecurseReductionTree<Idx * GroupSize>(inputs, reduction_op);
      reduceAllGroups<Idx + 1>(inputs, reduction_op);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm

#endif // TRTLLM_DEV_WSPROREDUCE_H
