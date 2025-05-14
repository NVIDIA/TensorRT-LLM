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

#include "tensorrt_llm/batch_manager/generateRequestOptions.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/utils/logitsThread.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

#include <NvInferRuntimeBase.h>

using namespace tensorrt_llm::runtime;

namespace te = tensorrt_llm::executor;

namespace tensorrt_llm::batch_manager
{

std::tuple<ITensor::SharedPtr, std::vector<decoder_batch::Request>, std::vector<SamplingConfig>>
GenerateRequestOptions::operator()(tr::ModelConfig const& modelConfig, tr::WorldConfig const& worldConfig,
    te::DecodingConfig const& decodingConfig, RequestVector const& contextRequests, BufferManager const& bufferManager,
    nvinfer1::DataType logitsType, DecoderInputBuffers& inputBuffers, OptionalRef<RuntimeBuffers const> buffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(GenerateRequestOptions);

    SizeType32 batchSize{0};
    unsigned decoderInputSize{0};
    for (auto const& llmReq : contextRequests)
    {
        auto const& reqTokens = llmReq->getTokens(0);
        if (llmReq->isLastContextChunk())
        {
            decoderInputSize += reqTokens.size();
            ++batchSize;
        }
    }
    inputBuffers.inputsIds->resize(decoderInputSize);

    TensorPtr batchSlotsView = runtime::ITensor::slice(inputBuffers.setupBatchSlots, 0, batchSize);
    auto batchSlotsRange = BufferRange<SizeType32>(*batchSlotsView);
    std::vector<decoder_batch::Request> decoderRequests;
    decoderRequests.reserve(batchSize);
    std::vector<SamplingConfig> samplingConfigs;
    samplingConfigs.reserve(batchSize);

    SizeType32 batchIdx{0};
    SizeType32 inputOffset{0};
    for (auto const& llmReq : contextRequests)
    {
        if (!llmReq->isLastContextChunk())
        {
            continue;
        }

        auto const promptLen = llmReq->getPromptLen();
        auto const& reqTokens = llmReq->getTokens(0);
        TLLM_CHECK(reqTokens.size() == static_cast<decltype(reqTokens.size())>(promptLen));
        TensorPtr inputView = ITensor::slice(inputBuffers.inputsIds, inputOffset, promptLen);
        bufferManager.copy(reqTokens.data(), *inputView);

        auto decoderRequest = decoder_batch::Request{inputView, promptLen, llmReq->mMaxNewTokens, llmReq->mEndId};

        llmReq->mSamplingConfig.normalizeLogProbs = mIsNormalizeLogProbs;
        if (modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
        {
            if (llmReq->hasDraftTokens())
            {
                auto const& draftTokens = llmReq->getDraftTokens();
                decoderRequest.draftTokens = bufferManager.copyFrom(*draftTokens, MemoryType::kPINNEDPOOL);
                auto const& draftLogits = llmReq->getDraftLogits();
                if (draftLogits.has_value())
                {
                    decoderRequest.draftLogits
                        = retrieveDraftLogits(modelConfig, worldConfig, draftLogits.value(), bufferManager);
                }
                decoderRequest.generatedTokensPerEngineStep = draftTokens->size() + 1;
            }
            else
            {
                decoderRequest.generatedTokensPerEngineStep = 1;
            }
        }
        else if (!modelConfig.getSpeculativeDecodingMode().isNone())
        {
            decoderRequest.generatedTokensPerEngineStep = modelConfig.getMaxDecodingTokens();
        }
        if (modelConfig.getSpeculativeDecodingMode().isMedusa())
        {
            TLLM_CHECK(buffers);
            llmReq->mSamplingConfig.topKMedusaHeads = {buffers->medusaBuffers->mTopKs};
            // FIXME: we must set medusa paths and tree ids not from seq slot, but from llmRequest?
            // When multiple microbatches buffers are used, runtime buffers can not be addressed with seqSlot.
            decoderRequest.medusaPaths = ITensor::slice(buffers->medusaBuffers->medusaPathsDevice, 0, 1);
            decoderRequest.medusaTreeIds = ITensor::slice(buffers->medusaBuffers->medusaTreeIdsDevice, 0, 1);
        }
        else if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
        {
            decoderRequest.lookaheadRuntimeConfig = llmReq->getLookaheadConfig()
                ? llmReq->getLookaheadConfig()
                : decodingConfig.getLookaheadDecodingConfig();
        }
        else if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
        {
            // Only Explicit draft tokens model needs dtype to WAR the lack of bf16 decoder.
            decoderRequest.dtype = modelConfig.getDataType();
        }
        else if (modelConfig.getSpeculativeDecodingMode().isEagle())
        {
            decoderRequest.eagleConfig
                = llmReq->getEagleConfig() ? llmReq->getEagleConfig() : decodingConfig.getEagleConfig();
        }
        if (llmReq->getEmbeddingBias().has_value())
        {
            decoderRequest.embeddingBias = getEmbeddingBias(logitsType, llmReq->getEmbeddingBias().value());
        }
        if (llmReq->getBadWordsList().has_value())
        {
            // Move to GPU and remove leading bs1 dimension since this is what decoderRequest expects
            decoderRequest.badWordsList = bufferManager.copyFrom(*llmReq->getBadWordsList().value(), MemoryType::kGPU);
            decoderRequest.badWordsList->squeeze(0);
        }
        if (llmReq->getStopWordsList().has_value())
        {
            decoderRequest.stopWordsList
                = bufferManager.copyFrom(*llmReq->getStopWordsList().value(), MemoryType::kGPU);
            decoderRequest.stopWordsList->squeeze(0);
        }
        batchSlotsRange[batchIdx] = llmReq->mSeqSlot.value();
        decoderRequests.push_back(decoderRequest);
        samplingConfigs.push_back(llmReq->mSamplingConfig);

        inputOffset += promptLen;
        ++batchIdx;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {std::move(batchSlotsView), std::move(decoderRequests), std::move(samplingConfigs)};
}

std::shared_ptr<runtime::ITensor> GenerateRequestOptions::retrieveDraftLogits(tr::ModelConfig const& modelConfig,
    tr::WorldConfig const& worldConfig, std::shared_ptr<runtime::ITensor> const& tensor,
    BufferManager const& bufferManager) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (!mSpeculativeDecodingFastLogits)
    {
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return bufferManager.copyFrom(*tensor, MemoryType::kPINNEDPOOL);
    }

    if (mIsLeaderInOrchMode)
    {
        te::SpeculativeDecodingFastLogitsInfo fastLogitsInfo;
        std::memcpy(&fastLogitsInfo, tensor->data(), sizeof(fastLogitsInfo));
        auto logits = utils::targetModelReceiveLogits(fastLogitsInfo, modelConfig).value();

        // Broadcast to other ranks if needed
        if (worldConfig.isTensorParallel())
        {
            auto const& commSession = COMM_SESSION;
            auto shape = logits->getShape();
            commSession.bcastValue(shape.d[0], 0);
            commSession.bcastValue(shape.d[1], 0);
            commSession.bcast(logits->data(), logits->getSizeInBytes(), mpi::MpiType::kUINT8, 0);
        }
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return logits;
    }

    // Get logits from leader rank
    auto const& commSession = COMM_SESSION;
    int64_t dims[2];
    commSession.bcastValue(dims[0], 0);
    commSession.bcastValue(dims[1], 0);
    auto const logitsDtype = modelConfig.getLogitsDtype();
    auto logits = tensorrt_llm::runtime::BufferManager::pinnedPool(ITensor::makeShape({dims[0], dims[1]}), logitsDtype);
    commSession.bcast(logits->data(), logits->getSizeInBytes(), mpi::MpiType::kUINT8, 0);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return logits;
};

GenerateRequestOptions::TensorPtr GenerateRequestOptions::getEmbeddingBias(
    nvinfer1::DataType logitsType, TensorPtr const& tensor) const
{
    // Check that embedding bias type is same as logits type. If so, we can return the tensor right away
    if (tensor->getDataType() == logitsType)
    {
        return tensor;
    }

    // Support FP32 input for FP16 embedding bias (in the case of FP8 models)
    if (tensor->getDataType() == nvinfer1::DataType::kFLOAT && logitsType == nvinfer1::DataType::kHALF)
    {
        // Do a deep copy of the tensor to the expected type
        TLLM_LOG_WARNING(
            "Embedding bias data type must be same as model logits type, will copy the tensor from float to half");

        TLLM_CHECK_WITH_INFO(
            tensor->getMemoryType() != MemoryType::kGPU, "Embedding bias tensor needs to be in CPU memory for casting");

        auto const shape = tensor->getShape();
        TLLM_CHECK(shape.nbDims == 2); // [1, vocabSizePadded]
        TLLM_CHECK(shape.d[0] == 1);
        auto newTensor = tensorrt_llm::runtime::BufferManager::pinnedPool(shape, logitsType);

        auto const tensorRange = BufferRange<float>(*tensor);
        auto newTensorRange = BufferRange<half>(*newTensor);

        std::transform(tensorRange.begin(), tensorRange.end(), newTensorRange.begin(),
            [](float value) -> half { return static_cast<half>(value); });

        return newTensor;
    }

    TLLM_THROW("Embedding bias data type must be same as model logits type.");
}

} // namespace tensorrt_llm::batch_manager
