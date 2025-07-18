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

#include "tensorrt_llm/batch_manager/handleGenerationLogits.h"

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"

namespace tr = tensorrt_llm::runtime;
namespace tru = tensorrt_llm::runtime::utils;

namespace tensorrt_llm::batch_manager
{

using BufferManager = tensorrt_llm::runtime::BufferManager;
using TensorPtr = runtime::ITensor::SharedPtr;
using ITensor = runtime::ITensor;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace
{

//! @brief Copy logits from generation phase under streaming mode.
void copyStreamingGenerationLogits(BufferManager const& bufferManager, LlmRequest& llmReq)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If llmRequest is streaming, directly copy to host.
    // Only one token's logits needs to be copied each time.
    TLLM_CHECK(llmReq.getGenerationLogitsFragmentsSize() == 1);

    SizeType32 numGenerationToken = llmReq.getMaxBeamNumTokens() - llmReq.mPromptLen;
    TensorPtr const& generationLogitsHost
        = llmReq.getGenerationLogitsHost(); // [mMaxNewTokens (or 1), beamWidth, vocabSizePadded]

    TensorPtr hostTensorPtr
        = ITensor::slice(generationLogitsHost, numGenerationToken, 1); // [1, beamWidth, vocabSizePadded]
    TensorPtr deviceTensorPtr = *(llmReq.getGenerationLogitsFragments().begin());

    bufferManager.copy(*deviceTensorPtr, *hostTensorPtr);
    llmReq.clearGenerationLogitsFragments();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void setupMedusaLogits(std::vector<TensorPtr>& medusaLogitsHeads, TensorPtr const& medusaLogitsDevice,
    SizeType32 medusaHeads, SizeType32 logitsIndex, SizeType32 numLogits)
{
    for (SizeType32 hi = 0; hi < medusaHeads; ++hi)
    {
        TensorPtr logitsHead = ITensor::slice(medusaLogitsDevice, hi, 1);
        logitsHead->squeeze(0);
        medusaLogitsHeads[hi] = ITensor::slice(logitsHead, logitsIndex, numLogits);
    }
}

} // namespace

void HandleGenerationLogits::operator()(DecoderInputBuffers& inputBuffers, RequestVector const& generationRequests,
    tr::ITensor::SharedPtr const& logits, tr::SizeType32 logitsIndex, tr::ModelConfig const& modelConfig,
    tr::BufferManager const& manager, OptionalRef<RuntimeBuffers> genRuntimeBuffers,
    OptionalRef<MedusaBuffers> medusaBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(HandleGenerationLogits);

    auto& decoderRequests = inputBuffers.decoderRequests;
    decoderRequests.reserve(decoderRequests.size() + generationRequests.size());
    auto& allDecoderLogits = inputBuffers.logits;
    allDecoderLogits.reserve(allDecoderLogits.size() + generationRequests.size());

    for (auto const& llmReq : generationRequests)
    {
        auto const reqBeamWidth = llmReq->getBeamWidthByIter();
        auto const seqSlot = llmReq->mSeqSlot.value();

        auto const draftLength = llmReq->getNumDraftTokens();
        auto const numLogits = draftLength + reqBeamWidth;

        TLLM_CHECK(draftLength == 0 || reqBeamWidth == 1);

        TLLM_LOG_DEBUG("logitsIndex: %d", logitsIndex);
        TLLM_LOG_DEBUG("draftLength: %d", draftLength);
        TLLM_LOG_DEBUG("reqBeamWidth: %d", reqBeamWidth);

        // genRuntimeBuffers.logits shape: [numGen*reqBeamWidth, vocabSize]
        // logitsView shape: [numLogits, vocabSize]
        TensorPtr logitsView = ITensor::slice(logits, logitsIndex, numLogits);
        TLLM_CHECK_DEBUG_WITH_INFO(tru::tensorHasInvalid<float>(*logitsView, manager, "logits") == false,
            "Found invalid number (NaN or Inf) in logits");

        TLLM_CHECK(llmReq->isGenerationInProgressState());
        TensorPtr decoderLogits;
        if (reqBeamWidth > 1)
        {
            decoderLogits = logitsView;
            decoderLogits->unsqueeze(0);
        }
        else
        {
            decoderLogits = logitsView;
            decoderLogits->unsqueeze(1);
        }
        decoderRequests.push_back(llmReq);
        allDecoderLogits.emplace_back(std::move(decoderLogits));

        if (llmReq->getReturnGenerationLogits())
        {
            TLLM_CHECK_WITH_INFO(modelConfig.getSpeculativeDecodingMode().isNone()
                    || modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal(),
                "Only speculative decoding with external draft tokens supports returning generation logits");

            // Push into fragments vector
            llmReq->addGenerationLogitsFragment(logitsView);
            TLLM_CHECK(
                llmReq->getGenerationLogitsFragmentsSize() <= RuntimeBuffers::GenerationLogitsCache::kCACHE_LENGTH);
            if (llmReq->isStreaming())
            {
                copyStreamingGenerationLogits(manager, *llmReq);
            }
            // Copy back to host for every kCACHE_LENGTH steps to mitigate GPU memory pressure
            else if (llmReq->getGenerationLogitsFragmentsSize() == RuntimeBuffers::GenerationLogitsCache::kCACHE_LENGTH)
            {
                TLLM_CHECK(genRuntimeBuffers);
                auto constexpr beforeDecoder = true;
                utils::copyGenerationLogits(genRuntimeBuffers->generationLogitsCache, manager, *llmReq, beforeDecoder);
            }
        }
        if (modelConfig.getSpeculativeDecodingMode().hasDraftLogits())
        {
            auto& medusaLogitsHeads = inputBuffers.predictedDraftLogits.at(seqSlot);
            TLLM_CHECK(medusaBuffers);
            setupMedusaLogits(medusaLogitsHeads, medusaBuffers->medusaLogitsDevice,
                modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(), logitsIndex, draftLength);
        }
        logitsIndex += numLogits;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
