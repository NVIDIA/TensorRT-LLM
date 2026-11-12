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

#include "tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageRoutes.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 1024; // a larger bucket loops: one launch either way

template <typename IdT, typename OutT>
__global__ void megaMoeStageRoutesKernel(IdT const* __restrict__ ids, float const* __restrict__ weights,
    int64_t liveElements, OutT* __restrict__ idsOut, float* __restrict__ weightsOut, int64_t elements)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < elements;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        bool const live = i < liveElements;
        idsOut[i] = live ? static_cast<OutT>(ids[i]) : static_cast<OutT>(-1);
        weightsOut[i] = live ? weights[i] : 0.F;
    }
}

} // namespace

template <typename IdT, typename OutT>
void megaMoeStageRoutes(IdT const* ids, float const* weights, int64_t liveRows, OutT* idsOut, float* weightsOut,
    int64_t rows, int topK, cudaStream_t stream)
{
    int64_t const elements = rows * topK;
    if (elements == 0)
    {
        return;
    }
    auto const blocks = static_cast<int>(std::min((elements + kThreads - 1) / kThreads, kMaxBlocks));
    megaMoeStageRoutesKernel<IdT, OutT>
        <<<blocks, kThreads, 0, stream>>>(ids, weights, liveRows * topK, idsOut, weightsOut, elements);
}

#define INSTANTIATE_MEGAMOE_STAGE_ROUTES(IdT, OutT)                                                                    \
    template void megaMoeStageRoutes<IdT, OutT>(                                                                       \
        IdT const*, float const*, int64_t, OutT*, float*, int64_t, int, cudaStream_t)

INSTANTIATE_MEGAMOE_STAGE_ROUTES(int32_t, int32_t);
INSTANTIATE_MEGAMOE_STAGE_ROUTES(int32_t, int64_t);
INSTANTIATE_MEGAMOE_STAGE_ROUTES(int64_t, int32_t);
INSTANTIATE_MEGAMOE_STAGE_ROUTES(int64_t, int64_t);

#undef INSTANTIATE_MEGAMOE_STAGE_ROUTES

} // namespace kernels

TRTLLM_NAMESPACE_END
