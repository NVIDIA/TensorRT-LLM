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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! The live-row word of a forward (tensorrt_llm/_torch/moe/live_rows.py): bits 0-15 hold its real requests R, bits
//! 16-31 its requests P.
constexpr int kLiveRowRequestBits = 16;
constexpr int32_t kLiveRowRequestMask = (1 << kLiveRowRequestBits) - 1;

//! The leading live rows of n rows under a live-row word: none when P is 0, all when R >= P or P does not divide n,
//! else n / P * R (the host reference is live_row_count in live_rows.py).
__host__ __device__ inline int64_t megaMoeLiveRows(int32_t word, int64_t n)
{
    int64_t const real = word & kLiveRowRequestMask;
    int64_t const requests = (word >> kLiveRowRequestBits) & kLiveRowRequestMask;
    if (requests == 0)
    {
        return 0;
    }
    return (real >= requests || n % requests != 0) ? n : n / requests * real;
}

//! Stages one MegaMoE launch's routes into its bucket in one launch. Of the n route rows, the live ones (all n, or
//! megaMoeLiveRows(*liveRowWord, n) when liveRowWord is not null) go to rows [0, live) of the contiguous [rows, topK]
//! buffers as expert ids (converted to OutT) and float32 gate weights; rows [live, rows) get id -1 and weight 0.
template <typename IdT, typename OutT>
void megaMoeStageRoutes(IdT const* ids, float const* weights, int64_t n, int32_t const* liveRowWord, OutT* idsOut,
    float* weightsOut, int64_t rows, int topK, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
