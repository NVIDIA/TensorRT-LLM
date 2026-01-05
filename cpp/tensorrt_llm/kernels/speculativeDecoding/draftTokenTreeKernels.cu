/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <cuda_runtime_api.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include "draftTokenTreeKernels.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

__global__ void extractRealDraftTokensKernel(int const curDraftIdx, int const batchSize, int const maxDraftLen,
    int const maxTotalDraftTokens, int const maxTopK, int const numTokensExpandThisLayer,
    int* tokensGatherIdxForDrafterModel, int* topKList, int* draftTokensIndicesCumsum, int64_t* newDraftTokens,
    int64_t* draftTokensBuffer)
{
    // curDraftIdx: int
    // batchSize: int
    // maxTotalDraftTokens: int
    // maxTopK: int
    // tokensGatherIdxForDrafterModel: int32_t*, indices of the draft tokens that need to be expand this layer
    //     shape: [numTokensExpandThisLayer]
    // topKList: int32_t*, top k value for each expandable token
    //     shape: [numTokensExpandThisLayer]
    // draftTokensIndicesCumsum: int32_t*, the cumulative sum of the write back indices for each draft layer
    //     shape: [maxDraftLen + 1]
    // newDraftTokens: int64_t*, the new draft tokens. We only need to extract this layer's tokens and write back to
    // the draftTokensBuffer shape: [batchSize, maxTotalDraftTokens + 1 if curDraftIdx > 0 else 1, maxTopK]
    // draftTokensBuffer: int64_t*, the buffer to store the real draft tokens
    //     shape: [maxBatchSize, maxTotalDraftTokens + 1]

    // Each thread handles one request
    auto const tix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto const isValid{tix < batchSize};

    if (isValid)
    {
        int newDraftTokensOffset = curDraftIdx == 0 ? 1 : maxTotalDraftTokens + 1;
        auto newDraftTokensStartPtr = newDraftTokens + tix * newDraftTokensOffset * maxTopK;
        auto draftTokensBufferPtr
            = draftTokensBuffer + tix * (maxTotalDraftTokens + 1) + draftTokensIndicesCumsum[curDraftIdx];

        int cnt = 0;
        for (int i = 0; i < numTokensExpandThisLayer; i++)
        {
            int tokenGatherIdx = tokensGatherIdxForDrafterModel[i];
            auto newDraftTokenPtr = newDraftTokensStartPtr + tokenGatherIdx * maxTopK;

            int topKValue = topKList[i];
            for (int j = 0; j < topKValue; j++)
            {
                int64_t newGenDraftToken = newDraftTokenPtr[j];
                draftTokensBufferPtr[cnt] = newGenDraftToken;
                cnt++;
            }
        }
    }
}

void invokeExtractRealDraftTokens(ExtractRealDraftTokensParam& params, cudaStream_t const stream)
{
    int constexpr BLOCK_SIZE = 64;
    int NUM_BLOCKS = divUp(params.batchSize, BLOCK_SIZE);

    extractRealDraftTokensKernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(params.curDraftIdx, params.batchSize,
        params.maxDraftLen, params.maxTotalDraftTokens, params.maxTopK, params.numTokensExpandThisLayer,
        params.tokensGatherIdxForDrafterModel, params.topKList, params.draftTokensIndicesCumsum, params.newDraftTokens,
        params.draftTokensBuffer);

    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
