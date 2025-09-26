/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/allocateKvCache.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

void tensorrt_llm::batch_manager::AllocateKvCache::operator()(BaseKVCacheManager& kvCacheManager,
    RequestVector& contextRequests, RequestVector const& generationRequests, runtime::ModelConfig const& modelConfig,
    OptionalRef<BaseKVCacheManager> crossKvCacheManager) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(allocateKvCache);

    for (auto const& llmReq : contextRequests)
    {
        if (llmReq->isFirstContextChunk())
        {
            auto const requestId = llmReq->mRequestId;
            auto const promptLen = llmReq->mPromptLen;
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
            auto draftLength = llmReq->getNumDraftTokens();

            // Allocate/Reuse KV cache
            kvCacheManager.addSequence(requestId, promptLen, reqBeamWidth, llmReq);

            // EagleNet will increment kv cache up to maxPathLen to account for accepted tokens.
            // Then up to maxDecodingDraftTokens will be used to generate next draft tokens.
            if (modelConfig.getSpeculativeDecodingMode().isEagle())
            {
                draftLength = modelConfig.getSpeculativeDecodingModule().getMaxPathLen()
                    + modelConfig.getSpeculativeDecodingModule().getMaxDecodingTokens();
            }

            // Allocate more KV cache for speculative decoding
            if (draftLength > 0)
            {
                for (SizeType32 di = 0; di < draftLength; ++di)
                {
                    kvCacheManager.addToken(requestId);
                }
            }

            if (crossKvCacheManager)
            {
                crossKvCacheManager->addSequence(requestId, llmReq->getEncoderOutputLen(), reqBeamWidth, llmReq);
            }
        }
    }

    for (auto const& llmReq : generationRequests)
    {
        auto const requestId = llmReq->mRequestId;
        auto decodingTokens = llmReq->getNumDraftTokens() + 1;

        // EagleNet will increment kv cache up to maxPathLen to account for accepted tokens.
        // Then up to maxDecodingDraftTokens will be used to generate next draft tokens.
        if (modelConfig.getSpeculativeDecodingMode().isEagle())
        {
            decodingTokens = modelConfig.getSpeculativeDecodingModule().getMaxPathLen()
                + modelConfig.getSpeculativeDecodingModule().getMaxDecodingTokens();
        }

        for (SizeType32 di = 0; di < decodingTokens; ++di)
        {
            kvCacheManager.addToken(requestId);
        }
    }

    kvCacheManager.refreshBlocks();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
