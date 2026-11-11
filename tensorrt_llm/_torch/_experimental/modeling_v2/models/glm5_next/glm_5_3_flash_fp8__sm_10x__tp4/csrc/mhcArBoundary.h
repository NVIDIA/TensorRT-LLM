/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// The tensor-parallel all-reduce of a decode step's partial rows fused with the mHC boundary after it
// (GLM-5.3-Flash at tp4).
#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

constexpr int kMhcArBoundaryMaxRows = 8;

struct MhcArBoundaryArgs
{
    void** workspace;        // the TP group's TensorRT-LLM all-reduce workspace (Lamport buffers, flag, sizes)
    int rank;                // this rank in the group of 4
    void const* partial;     // [M, 4096] bf16 this rank's partial output of the block
    void const* residual;    // [M, 4, 4096] bf16 the streams before the block's post-mapping
    float const* post_prev;  // [M, 4]
    float const* comb_prev;  // [M, 4, 4]
    float const* fn;         // [24, 4 * 4096] the next block's mixer weights
    float const* scale;      // [3]
    float const* base;       // [24]
    void const* norm_weight; // [4096] bf16 the next block's input RMSNorm
    void* residual_out;      // [M, 4, 4096] bf16
    float* post_out;         // [M, 4]
    float* comb_out;         // [M, 4, 4]
    void* layer_input;       // [M, 4096] bf16 RMS-normalized
    int rows;
    int sinkhorn_iters;
    float norm_eps; // the mixer's RMS epsilon
    float hc_eps;   // added to pre
    float sinkhorn_eps;
    float post_mult;
    float layer_input_eps; // the next block's RMSNorm epsilon
    bool pdl;              // programmatic dependent launch (only the weights are read before the wait)
};

// One launch on `stream`: the all-reduce reads and rotates the workspace's Lamport state, so it is ordered with the
// other users of that workspace (the library's one-shot all-reduce) by the stream.
cudaError_t mhcArBoundary(MhcArBoundaryArgs const& args, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
