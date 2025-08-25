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

#include "tensorrt_llm/batch_manager/logitsPostProcessor.h"

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

using BufferManager = tensorrt_llm::runtime::BufferManager;
using TensorPtr = runtime::ITensor::SharedPtr;
using ITensor = runtime::ITensor;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

bool LogitsPostProcessor::operator()(RequestVector const& contextRequests, RequestVector const& generationRequests,
    bool replicateLogitsPostProcessor, std::vector<TensorPtr>& seqSlotLogits, tr::WorldConfig const& worldConfig,
    tr::TllmRuntime& runtime, std::optional<LogitsPostProcessorBatched> const& logitsPostProcessorBatched) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(LogitsPostProcessor);

    // Arguments for batched processor
    std::vector<LlmRequest::RequestIdType> reqIdsVec;
    std::vector<LlmRequest::TensorPtr> logitsVec;
    std::vector<std::reference_wrapper<LlmRequest::BeamTokens const>> beamTokensVec;
    std::vector<std::optional<LlmRequest::RequestIdType>> clientIdsVec;

    bool logitsPostProcessorIsApplied = false;
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->isContextInitState() ? llmReq->isLastContextChunk() : llmReq->isGenerationInProgressState())
            {
                // Invoke non-batched processor or collect arguments for batched processor
                if (llmReq->mLogitsPostProcessor)
                {
                    logitsPostProcessorIsApplied = true;
                    if (replicateLogitsPostProcessor || worldConfig.isFirstTensorParallelRank())
                    {
                        auto& logits = seqSlotLogits.at(llmReq->mSeqSlot.value());
                        (*llmReq->mLogitsPostProcessor)(
                            llmReq->mRequestId, logits, llmReq->getTokens(), runtime.getStreamPtr(), llmReq->mClientId);
                    }
                }
                else if (llmReq->mApplyLogitsPostProcessorBatched)
                {
                    reqIdsVec.push_back(llmReq->mRequestId);

                    auto& logits = seqSlotLogits.at(llmReq->mSeqSlot.value());
                    logitsVec.push_back(logits);

                    beamTokensVec.emplace_back(llmReq->getTokens());
                    clientIdsVec.push_back(llmReq->mClientId);
                }
            }
        }
    }

    // Invoke batched processor
    if (!reqIdsVec.empty())
    {
        logitsPostProcessorIsApplied = true;
        if (replicateLogitsPostProcessor || worldConfig.isFirstTensorParallelRank())
        {
            (*logitsPostProcessorBatched)(reqIdsVec, logitsVec, beamTokensVec, runtime.getStreamPtr(), clientIdsVec);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return logitsPostProcessorIsApplied;
}

} // namespace tensorrt_llm::batch_manager
