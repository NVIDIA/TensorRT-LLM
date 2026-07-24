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

#include "encoderBuffers.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <valarray>

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

EncoderBuffers::EncoderBuffers(
    SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    // init empty buffers on cpu/gpu/pinned
    init(maxBatchSize, modelConfig, worldConfig, runtime);

    // pre-allocate based on max buffer sizes
    // Note: pre-allocation can be done directly instead of empty-->reshape, but it is ok extract the common reshape()
    // utility because the buffer shapes can be dynamically set during runtime as well
    initBufferSizes(maxBatchSize, modelConfig, worldConfig, runtime);
}

void EncoderBuffers::init(
    SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();

    auto hiddenStatesType = modelConfig.getDataType();

    inputFeatures = manager.emptyTensor(MemoryType::kGPU, hiddenStatesType);
    inputIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    // in PP, only rank 0 needs the following input fields
    if (modelConfig.usePositionEmbedding() && worldConfig.isFirstPipelineParallelRank())
    {
        positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        positionIdsReserved.resize(maxBatchSize * modelConfig.getMaxInputLen());
        std::iota(positionIdsReserved.begin(), positionIdsReserved.end(), 0);
    }
    if (modelConfig.useTokenTypeEmbedding() && worldConfig.isFirstPipelineParallelRank())
    {
        tokenTypeIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        tokenTypeIdsReserved.resize(maxBatchSize * modelConfig.getMaxInputLen());
        std::fill(tokenTypeIdsReserved.begin(), tokenTypeIdsReserved.end(), 0);
    }

    inputLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    maxInputLength = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, hiddenStatesType);
    }
    if (worldConfig.isLastPipelineParallelRank())
    {
        encoderOutput = manager.emptyTensor(MemoryType::kGPU, hiddenStatesType);
    }

    if (modelConfig.useLanguageAdapter())
    {
        languageAdapterRoutings = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType32>::value);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::initBufferSizes(
    SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // get buffer shape based on max values
    numRequests = maxBatchSize;
    encoderInputLen = maxBatchSize * modelConfig.getMaxInputLen();
    encoderOutputLen = maxBatchSize * modelConfig.getMaxInputLen(); // assume output length <= input length
    maxInputLengthInBatch = modelConfig.getMaxInputLen();

    // update buffer shapes
    reshape(runtime, modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::updateBufferSizes(RequestVector const& requests, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    numRequests = requests.size();
    encoderInputLen = 0;
    encoderOutputLen = 0;
    maxInputLengthInBatch = 0;

    // get buffer shape based on actual batched requests
    for (auto const& req : requests)
    {
        encoderInputLen += req->getEncoderInputLen();
        encoderOutputLen += req->getEncoderOutputLen();
        maxInputLengthInBatch
            = std::max(maxInputLengthInBatch, req->getEncoderInputLen()); // Decoder input is encoder output
    }

    // update buffer shapes
    reshape(runtime, modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::reshape(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (modelConfig.isMultiModal())
    {
        return; // multimodal models do not need to set position id, etc. or any output tensors
    }

    inputIds->reshape(ITensor::makeShape({encoderInputLen}));
    if (positionIds)
    {
        if (modelConfig.isWhisper())
        {
            positionIds->reshape(ITensor::makeShape({encoderOutputLen}));
        }
        else
        {
            positionIds->reshape(ITensor::makeShape({encoderInputLen}));
        }
    }
    if (tokenTypeIds)
    {
        tokenTypeIds->reshape(ITensor::makeShape({encoderInputLen}));
    }

    inputLengths->reshape(ITensor::makeShape({numRequests}));
    maxInputLength->reshape(ITensor::makeShape({maxInputLengthInBatch}));

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates->reshape(
            ITensor::makeShape({encoderOutputLen, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
    }
    if (worldConfig.isLastPipelineParallelRank())
    {
        encoderOutput->reshape(
            ITensor::makeShape({encoderOutputLen, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
    }
    if (modelConfig.useLanguageAdapter())
    {
        languageAdapterRoutings->reshape(ITensor::makeShape({encoderInputLen, 1}));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::setFromInputs(RequestVector const& requests, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(encoderBuffersSetFromInputs);

    if (!worldConfig.isFirstPipelineParallelRank())
    {
        return;
    }

    auto const& manager = runtime.getBufferManager();

    std::vector<TokenIdType> inputIdsAll;
    std::vector<SizeType32> positionIdsAll;
    std::vector<SizeType32> tokenTypeIdsAll;
    std::vector<SizeType32> inputLengthsAll;
    std::vector<SizeType32> languageAdapterRoutingAll;
    // use shape to indicates max input length, content is not important
    // TODO: change to a scalar value for this from engine side
    std::vector<SizeType32> maxInputLengthAll(maxInputLengthInBatch, 0);

    if (requests.front()->getEncoderInputFeatures())
    {
        if (modelConfig.isMultiModal())
        {
            auto batchedInputShape = requests.front()->getEncoderInputFeatures()->getShape(); // [1, 3, H, W]
            batchedInputShape.d[0] = encoderInputLen;                                         // [batch_size, 3, H, W]
            inputFeatures->reshape(batchedInputShape);
        }
        else
        {
            SizeType32 const featureDim = requests.front()->getEncoderInputFeatures()->getShape().d[1];
            TLLM_LOG_DEBUG("EncoderBuffers::setFromInputs - featureDim = %d", featureDim);
            inputFeatures->reshape(ITensor::makeShape({encoderInputLen, featureDim}));
        }
    }

    SizeType32 offset = 0;

    for (auto const& llmReq : requests)
    {
        SizeType32 const inputLength = llmReq->getEncoderInputLen();
        SizeType32 const outputLength = llmReq->getEncoderOutputLen();
        if (llmReq->getEncoderInputFeatures())
        {
            auto const& reqFeatures
                = llmReq
                      ->getEncoderInputFeatures(); // whisper: [length, featureDim]; Vision: [batch_size, channel, W, H]
            TLLM_LOG_DEBUG("EncoderBuffers::setFromInputs - request id = %d, input features length = %d",
                llmReq->mRequestId, inputLength);
            manager.copy(*reqFeatures, *ITensor::slice(inputFeatures, offset, inputLength));
            offset += inputLength;
        }
        else
        {
            auto const& reqTokens = *llmReq->getEncoderTokens().value();
            inputIdsAll.insert(inputIdsAll.end(), reqTokens.begin(), reqTokens.end());
            if (tokenTypeIds)
            {
                tokenTypeIdsAll.insert(
                    tokenTypeIdsAll.end(), tokenTypeIdsReserved.begin(), tokenTypeIdsReserved.begin() + inputLength);
            }
        }
        if (positionIds)
        {
            SizeType32 const length = modelConfig.isWhisper() ? outputLength : inputLength;
            positionIdsAll.insert(
                positionIdsAll.end(), positionIdsReserved.begin(), positionIdsReserved.begin() + length);
        }
        if (modelConfig.useLanguageAdapter())
        {
            auto const languageAdapterRouting
                = llmReq->getLanguageAdapterRouting(modelConfig.getNumLanguages().value(), inputLength);
            languageAdapterRoutingAll.insert(
                languageAdapterRoutingAll.end(), std::begin(languageAdapterRouting), std::end(languageAdapterRouting));
        }
        inputLengthsAll.push_back(inputLength);
    }

    // copy inputs from host to device
    {
        NVTX3_SCOPED_RANGE(bufferCopies);
        if (requests.front()->getEncoderTokens())
        {
            manager.copy(inputIdsAll.data(), *inputIds);
            if (tokenTypeIds)
            {
                manager.copy(tokenTypeIdsAll.data(), *tokenTypeIds);
            }
            manager.copy(maxInputLengthAll.data(), *maxInputLength);
        }
        if (positionIds)
        {
            manager.copy(positionIdsAll.data(), *positionIds);
        }
        manager.copy(inputLengthsAll.data(), *inputLengths);
        if (modelConfig.useLanguageAdapter())
        {
            manager.copy(languageAdapterRoutingAll.data(), *languageAdapterRoutings);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::fillIOMaps(ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersFillIOMaps);

    inputMap.clear();
    outputMap.clear();

    // inputs
    if (modelConfig.isMultiModal())
    {
        inputMap.insert_or_assign("input", inputFeatures);
    }
    else if (modelConfig.isWhisper())
    {
        inputMap.insert_or_assign("input_features", inputFeatures);
        inputMap.insert_or_assign("input_lengths", inputLengths);
        inputMap.insert_or_assign("position_ids", positionIds);
    }
    else
    {
        if (worldConfig.isFirstPipelineParallelRank())
        {
            inputMap.insert_or_assign("input_ids", inputIds);
            if (positionIds)
            {
                inputMap.insert_or_assign("position_ids", positionIds);
            }
            if (tokenTypeIds)
            {
                inputMap.insert_or_assign("token_type_ids", tokenTypeIds);
            }
        }
        else
        {
            inputMap.insert_or_assign("hidden_states_input", hiddenStates);
        }
        inputMap.insert_or_assign("input_lengths", inputLengths);
        inputMap.insert_or_assign("max_input_length", maxInputLength);
        if (modelConfig.useLanguageAdapter())
        {
            inputMap.insert_or_assign("language_adapter_routings", languageAdapterRoutings);
        }
    }

    // outputs
    if (worldConfig.isLastPipelineParallelRank())
    {
        outputMap.insert_or_assign("encoder_output", encoderOutput);
    }
    else
    {
        outputMap.insert_or_assign("hidden_states_output", hiddenStates);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::pair<EncoderBuffers::TensorMap const&, EncoderBuffers::TensorMap&> EncoderBuffers::prepareIO(
    RequestVector const& requests, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    updateBufferSizes(requests, modelConfig, worldConfig, runtime);

    setFromInputs(requests, modelConfig, worldConfig, runtime);

    fillIOMaps(modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);

    return {inputMap, outputMap};
}

void EncoderBuffers::rearrangeOutputs(RequestVector const& requests, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(encoderBuffersRearrangeOutput);

    auto const& manager = runtime.getBufferManager();

    SizeType32 offset = 0, size = 0;

    updateReqOutputShape(requests, runtime, worldConfig, modelConfig);

    for (auto const& req : requests)
    {
        // copy from internal buffer to request-owned external buffers
        size = req->getEncoderOutputLen();
        TLLM_LOG_DEBUG("EncoderBuffers::rearrangeOutputs - req: %d, encoderOutput shape = (%d, %d)", req->mClientId,
            req->getEncoderOutput()->getShape().d[0], req->getEncoderOutput()->getShape().d[1]);
        TLLM_LOG_DEBUG("EncoderBuffers::rearrangeOutputs - req: %d, enc output size = %d", req->mClientId, size);

        if (worldConfig.isPipelineParallel())
        {
            manager.copy(*ITensor::slice(hiddenStates, offset, size), *req->getEncoderHiddenStates());
        }
        if (worldConfig.isLastPipelineParallelRank())
        {
            if (modelConfig.isMultiModal())
            {
                manager.copy(
                    *ITensor::slice(encoderOutput, offset, size), *(req->getPromptEmbeddingTableMutable().value()));
            }
            else
            {
                manager.copy(*ITensor::slice(encoderOutput, offset, size), *req->getEncoderOutput());
            }
        }
        offset += size;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::updateReqOutputShape(RequestVector const& requests, TllmRuntime const& runtime,
    WorldConfig const& worldConfig, ModelConfig const& modelConfig)
{
    auto const& manager = runtime.getBufferManager();

    for (auto const& req : requests)
    {
        if (modelConfig.isMultiModal())
        {
            auto shape = encoderOutput->getShape(); // [batch_size, prompt_vocab_size, feature_dim]
            shape.d[0] = req->getEncoderOutputLen();
            req->getPromptEmbeddingTableMutable() = manager.emptyTensor(MemoryType::kGPU, encoderOutput->getDataType());
            req->getPromptEmbeddingTableMutable().value()->reshape(shape);
            req->setPromptVocabSize(shape.d[1]);
            // TODO: extra ids for kv cache reuse
        }
        else
        {
            auto encOutLen = req->getEncoderOutputLen();
            // update request-owned external buffer for each request
            if (worldConfig.isPipelineParallel())
            {
                req->getEncoderHiddenStates()->reshape(
                    ITensor::makeShape({encOutLen, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
            }
            if (worldConfig.isLastPipelineParallelRank())
            {
                req->getEncoderOutput()->reshape(
                    ITensor::makeShape({encOutLen, modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()}));
            }
        }
    }
}

void EncoderBuffers::create(SizeType32 maxBatchSize, ModelConfig const& modelConfig, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();

    inputLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    maxInputLength = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    hiddenSize = modelConfig.getEncoderHiddenSize(); // full hidden size
    // assume encoder & decoder use the same data type
    encoderOutput = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    encoderOutputReserved = manager.gpu(ITensor::makeShape({1, hiddenSize}), modelConfig.getDataType());

    crossKvCacheGen = manager.gpu(ITensor::makeShape({1}), nvinfer1::DataType::kBOOL);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::setMaxBufferSizes(SizeType32 maxBatchSize, runtime::ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    numRequests = maxBatchSize;
    encoderInputLen = maxBatchSize * modelConfig.getMaxEncoderLen();
    encoderOutputLen = maxBatchSize * modelConfig.getMaxEncoderLen();
    maxInputLengthInBatch = modelConfig.getMaxEncoderLen();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    numRequests = 0; /// total number of requests that need encoder information (context requests +
                     /// generation requests * beam width)
    encoderInputLen = 0;
    encoderOutputLen = 0;
    maxInputLengthInBatch = 1; /// maximum encoder length in a batch

    for (auto const& llmReq : contextRequests)
    {
        numRequests += 1;
        encoderInputLen += llmReq->getEncoderInputLen();
        encoderOutputLen += llmReq->getEncoderOutputLen();
        maxInputLengthInBatch = std::max(maxInputLengthInBatch, llmReq->getEncoderInputLen());
    }

    for (auto const& llmReq : genRequests)
    {
        auto const reqBeamWidth = llmReq->getBeamWidthByIter();
        numRequests += reqBeamWidth; // tile by beam width
        maxInputLengthInBatch = std::max(maxInputLengthInBatch, llmReq->getEncoderInputLen());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::reshape()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    inputLengths->reshape(ITensor::makeShape({numRequests}));
    maxInputLength->reshape(ITensor::makeShape({maxInputLengthInBatch}));
    encoderOutput->reshape(ITensor::makeShape({encoderOutputLen, hiddenSize}));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::fill(
    RequestVector const& ctxRequests, RequestVector const& genRequests, runtime::BufferManager const& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(encoderBufferCopies);

    std::vector<SizeType32> inputLengthsAll;
    std::vector<SizeType32> maxInputLengthAll(maxInputLength->getShape().d[0], 0);

    SizeType32 offset = 0, size = 0;
    for (auto const& requests : {ctxRequests, genRequests})
    {
        for (auto const& llmReq : requests)
        {
            // 1. only ctx requests should gather the encoder output
            // 2. only gen requests should tile encoder input lengths info by beam width
            bool isCtx = llmReq->isContextInitState();
            if (isCtx)
            {
                size = llmReq->getEncoderOutputLen();
                auto const encoderOutputSlice = runtime::ITensor::slice(encoderOutput, offset, size);
                manager.copy(*llmReq->getEncoderOutput(), *encoderOutputSlice);
                offset += size;

                inputLengthsAll.emplace_back(size);
            }
            else
            {
                auto const reqBeamWidth = llmReq->getBeamWidthByIter();
                std::fill_n(std::back_inserter(inputLengthsAll), reqBeamWidth,
                    llmReq->getEncoderOutputLen()); // although encoder output is not needed, gen phase still needs the
                                                    // encoder length info for cross kv cache. Also tile by beam width
            }
        }
    }
    manager.copy(inputLengthsAll.data(), *inputLengths);
    manager.copy(maxInputLengthAll.data(), *maxInputLength);
    // crossKvCacheGen unused in engine for now, use default tensor

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EncoderBuffers::insertInputTensors(TensorMap& inputMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    inputMap.insert_or_assign("encoder_output", encoderOutput);
    inputMap.insert_or_assign("encoder_input_lengths", inputLengths);
    inputMap.insert_or_assign("encoder_max_input_length", maxInputLength);
    inputMap.insert_or_assign("cross_kv_cache_gen", crossKvCacheGen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
} // namespace tensorrt_llm::batch_manager
