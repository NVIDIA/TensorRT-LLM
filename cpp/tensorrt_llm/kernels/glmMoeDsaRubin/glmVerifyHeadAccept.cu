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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmVerifyHeadAccept.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{

constexpr int kAcceptBlockSize = 128;

// One CTA; a thread per request. The forced-acceptance counter is read and advanced once per call.
template <typename TargetT>
__global__ void __launch_bounds__(kAcceptBlockSize) verifyHeadAcceptKernel(VerifyHeadAcceptParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // The target tokens are written by the preceding kernel.
    cudaGridDependencySynchronize();
#endif
    __shared__ int64_t counter;
    bool const randomForce = params.forceMode == VerifyHeadForceMode::kRandom;
    if (randomForce && threadIdx.x == 0)
    {
        counter = *params.forceCounter + 1;
        *params.forceCounter = counter;
    }
    __syncthreads();

    auto const* target = static_cast<TargetT const*>(params.targetTokens);
    int const batchSize = params.numContexts + params.numGens;
    int const rowLen = params.draftLen + 1;
    for (int req = static_cast<int>(threadIdx.x); req < batchSize; req += kAcceptBlockSize)
    {
        int64_t const rowOffset = static_cast<int64_t>(req) * rowLen;
        if (req < params.numContexts)
        {
            params.numAcceptedTokens[req] = 1;
            if (params.acceptedTokens != nullptr)
            {
                params.acceptedTokens[rowOffset] = static_cast<int32_t>(target[req]);
            }
            continue;
        }
        int const gen = req - params.numContexts;
        TargetT const* genTarget = target + params.numContexts + static_cast<int64_t>(gen) * rowLen;
        int32_t const* genDraft = params.draftTokens + static_cast<int64_t>(gen) * params.draftLen;
        // The target token plus the leading draft tokens that equal the target tokens.
        int accepted = 1;
        while (accepted <= params.draftLen
            && static_cast<int64_t>(genDraft[accepted - 1]) == static_cast<int64_t>(genTarget[accepted - 1]))
        {
            ++accepted;
        }
        if (params.forceMode == VerifyHeadForceMode::kFixed)
        {
            accepted = params.forceBase;
        }
        else if (randomForce)
        {
            int64_t const poolIdx
                = (counter * params.counterStride + static_cast<int64_t>(gen) * params.slotStride) & params.poolMask;
            int const extra = params.forcePool[poolIdx] < params.forceFrac ? 1 : 0;
            accepted = min(params.forceBase + extra, params.forceMax);
        }
        params.numAcceptedTokens[req] = accepted;
        if (params.acceptedTokens != nullptr)
        {
            for (int j = 0; j < rowLen; ++j)
            {
                params.acceptedTokens[rowOffset + j] = static_cast<int32_t>(genTarget[j]);
            }
        }
    }
}

} // namespace

void invokeVerifyHeadAccept(VerifyHeadAcceptParams const& params)
{
    TLLM_CHECK_WITH_INFO(params.forceMode != VerifyHeadForceMode::kRandom
            || (params.forcePool != nullptr && params.forceCounter != nullptr),
        "verifyHeadAccept: the random forced count needs the pool and the counter");
    if (params.numContexts + params.numGens == 0)
    {
        return;
    }
    if (params.targetTokensInt64)
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled("verifyHeadAccept", verifyHeadAcceptKernel<int64_t>, dim3(1),
            dim3(kAcceptBlockSize), 0, params.stream, params);
    }
    else
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled("verifyHeadAccept", verifyHeadAcceptKernel<int32_t>, dim3(1),
            dim3(kAcceptBlockSize), 0, params.stream, params);
    }
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
