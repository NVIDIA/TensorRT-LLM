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

#include "tensorrt_llm/batch_manager/runtimeBuffers.h"

#include "tensorrt_llm/batch_manager/encoderBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/loraBuffers.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/promptTuningBuffers.h"
#include "tensorrt_llm/batch_manager/rnnStateBuffers.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/transformerBuffers.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

RuntimeBuffers::RuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, bool gatherGenerationLogits, std::optional<SizeType32> maxNumTokens,
    std::optional<std::vector<executor::AdditionalModelOutput>> const& additionalModelOutputs,
    bool promptTableOffloadingParam)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    promptTableOffloading = promptTableOffloadingParam;

    create(maxBatchSize, maxBeamWidth, maxAttentionWindowVec, maxAttentionWindow, sinkTokenLen, runtime, modelConfig,
        worldConfig, decodingConfig, gatherGenerationLogits, additionalModelOutputs);

    // pre-allocate
    setMaxBufferSizes(maxBatchSize, maxBeamWidth, modelConfig, maxNumTokens);
    reshape(runtime, modelConfig, worldConfig, gatherGenerationLogits);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

RuntimeBuffers::~RuntimeBuffers() = default;

void RuntimeBuffers::create(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, bool gatherGenerationLogits,
    std::optional<std::vector<executor::AdditionalModelOutput>> const& additionalModelOutputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    if (modelConfig.isTransformerBased())
    {
        transformerBuffers = std::make_unique<TransformerBuffers>(maxBatchSize, maxBeamWidth, maxAttentionWindowVec,
            maxAttentionWindow, sinkTokenLen, runtime, modelConfig, worldConfig);
    }
    if (modelConfig.isRnnBased())
    {
        rnnStateBuffers = std::make_unique<RnnStateBuffers>(maxBatchSize, runtime);
    }

    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    inputsIds = manager.emptyTensor(MemoryType::kGPU, nvTokenIdType);

    mropeRotaryCosSin = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    mropePositionDeltas = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = engine.getTensorDataType(batch_manager::RuntimeBuffers::kLogitsTensorName);
        logits = manager.emptyTensor(MemoryType::kGPU, logitsType);
    }

    // TODO: check which tensors can be allocated as pinned for max size
    requestTypes = manager.emptyTensor(MemoryType::kCPU, TRTDataType<runtime::RequestType>::value);

    contextLengthsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    contextLengthsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    sequenceLengthsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    sequenceLengthsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    lastTokenIdsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    lastTokenIdsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    logitsIdsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);

    inputsIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    }

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    seqSlots = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT32);
    seqSlotsDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT32);

    cacheIndirDecoderIOBatchedCopySrcOffsets
        = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    cacheIndirDecoderIOBatchedCopyDstOffsets
        = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    cacheIndirDecoderIOBatchedCopySizes
        = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    mCacheIndirDecoderIOBatchedCopyCopySizesDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT64);

    // Pre-allocate buffer for saving generation logits for model w/o draft tokens
    if (gatherGenerationLogits
        && (modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal()
            || modelConfig.getSpeculativeDecodingMode().isNone())
        && worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        auto const logitsType = engine.getTensorDataType(batch_manager::RuntimeBuffers::kLogitsTensorName);

        generationLogitsCache.transposedLogits = manager.gpu(
            ITensor::makeShape({maxBeamWidth, GenerationLogitsCache::kCACHE_LENGTH, vocabSizePadded}), logitsType);
        generationLogitsCache.logits = manager.gpu(
            ITensor::makeShape({GenerationLogitsCache::kCACHE_LENGTH, maxBatchSize * maxBeamWidth, vocabSizePadded}),
            logitsType);

        generationLogitsCache.fragmentPointerDevice
            = manager.gpu(ITensor::makeShape({GenerationLogitsCache::kCACHE_LENGTH}), nvinfer1::DataType::kINT64);
        generationLogitsCache.fragmentPointerHost = tensorrt_llm::runtime::BufferManager::pinnedPool(
            ITensor::makeShape({maxBatchSize, GenerationLogitsCache::kCACHE_LENGTH}), nvinfer1::DataType::kINT64);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers = std::make_unique<EncoderBuffers>();
        encoderBuffers->create(maxBatchSize, modelConfig, runtime);
    }

    if (modelConfig.usePromptTuning())
    {
        promptTuningBuffers = std::make_unique<PromptTuningBuffers>(
            maxBatchSize, manager, modelConfig, worldConfig, promptTableOffloading);
    }

    if (modelConfig.useLoraPlugin())
    {
        loraBuffers = std::make_unique<LoraBuffers>(maxBatchSize, maxBeamWidth, runtime, modelConfig, worldConfig);
    }

    if (modelConfig.getSpeculativeDecodingMode().isMedusa())
    {
        mMedusaBuffers = std::make_unique<MedusaBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig, runtime);
    }
    else if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        mLookaheadBuffers = std::make_unique<runtime::LookaheadRuntimeBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig, runtime);
    }
    else if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        mExplicitDraftTokensBuffers = std::make_unique<runtime::ExplicitDraftTokensBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig);
    }
    else if (modelConfig.getSpeculativeDecodingMode().isEagle())
    {
        mEagleBuffers = std::make_unique<runtime::EagleBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig);
    }

    if (modelConfig.useLanguageAdapter())
    {
        languageAdapterRoutings = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType32>::value);
    }

    for (auto const& output : additionalModelOutputs.value_or(std::vector<executor::AdditionalModelOutput>{}))
    {
        auto const& engine = runtime.getEngine();
        auto const dataType = engine.getTensorDataType(output.name.c_str());
        mAdditionalOutputTensors.emplace(output.name, manager.emptyTensor(runtime::MemoryType::kGPU, dataType));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setMaxBufferSizes(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    runtime::ModelConfig const& modelConfig, std::optional<SizeType32> maxNumRuntimeTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // `maxNumSequences` is reached when all requests are in generation
    numContextRequests = 0;
    numGenRequests = maxBatchSize;
    numGenSequences = maxBatchSize * maxBeamWidth;

    auto const maxDraftTokens = modelConfig.getMaxDecodingDraftTokens();
    // Draft-Tokens and Beam-Search are mutually exclusive
    numLogits = maxBatchSize * std::max(1 + maxDraftTokens, maxBeamWidth);
    auto const maxNumModelTokens = modelConfig.getMaxNumTokens();
    auto const maxNumContextTokens = maxBatchSize * modelConfig.getMaxInputLen();
    auto const maxNumGenTokens = numLogits;
    // For pre-allocation
    numContextTokens = 0; // Set in `setBufferSizes` rather than here for `computeContextLogits`
    numGenTokens
        = maxNumRuntimeTokens.value_or(maxNumModelTokens.value_or(std::max(maxNumContextTokens, maxNumGenTokens)));

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->setMaxBufferSizes(maxBatchSize, modelConfig);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersSetBufferSizes);

    // set context sizes
    numContextRequests = static_cast<SizeType32>(contextRequests.size());
    auto numContextLogits = numContextRequests;
    numContextTokens = 0;
    maxContextLength = 0;
    for (auto const& llmReq : contextRequests)
    {
        auto const draftLength = llmReq->isLastContextChunk() ? llmReq->getNumDraftTokens() : 0;
        numContextLogits += draftLength;

        auto const contextChunkSize = llmReq->getContextChunkSize();
        numContextTokens += contextChunkSize + draftLength;
        if (maxContextLength < llmReq->mPromptLen)
        {
            maxContextLength = llmReq->mPromptLen;
        }
    }

    // set generation sizes
    numGenRequests = static_cast<SizeType32>(genRequests.size());
    numGenSequences = 0;
    numGenTokens = 0;
    for (auto const& llmReq : genRequests)
    {
        auto const reqBeamWidth = llmReq->getBeamWidthByIter();
        numGenSequences += reqBeamWidth;
        auto const draftLen = llmReq->getNumDraftTokens();
        numGenTokens += draftLen + reqBeamWidth;
    }

    numLogits = numContextLogits + numGenTokens;

    if (encoderBuffers)
    {
        encoderBuffers->setBufferSizes(contextRequests, genRequests);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::reshape(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    bool gatherGenerationLogits)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersReshape);

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

        if (modelConfig.computeContextLogits() && (numContextRequests > 0))
        {
            // Only when need to return context logits, and there are new requests will execute context phase,
            // logits buffer need to be re-allocated with size of [numContextTokens + numGenSequences, vocabSizePadded]
            auto const& engine = runtime.getEngine();
            auto const& manager = runtime.getBufferManager();
            auto const logitsType = engine.getTensorDataType(kLogitsTensorName);
            logits = manager.gpu(ITensor::makeShape({numContextTokens + numGenSequences, vocabSizePadded}), logitsType);
        }
        else if (gatherGenerationLogits && modelConfig.getSpeculativeDecodingMode().isNone())
        {
            // If need to return generation logits, re-point the logit buffer to avoid overwrite,
            // so we could write back GenerationLogitsCache::kCACHE_LENGTH steps' logits together
            // logits shape: [1, maxBatchSize * maxBeamWidth, vocabSizePadded]
            // which is large enough to cover both numContextRequests and numGenSequences
            logits = ITensor::slice(generationLogitsCache.logits, generationLogitsCache.offset, 1);
            generationLogitsCache.offset = (generationLogitsCache.offset + 1) % GenerationLogitsCache::kCACHE_LENGTH;
            logits->squeeze(0);
        }
        else
        {
            logits->reshape(ITensor::makeShape({numLogits, vocabSizePadded}));
        }
    }

    auto const numSequences = getNumSequences();
    auto const numSequencesShape = ITensor::makeShape({numSequences});
    requestTypes->reshape(numSequencesShape);
    contextLengthsHost->reshape(numSequencesShape);
    contextLengthsDevice->reshape(numSequencesShape);
    sequenceLengthsHost->reshape(numSequencesShape);
    sequenceLengthsDevice->reshape(numSequencesShape);

    auto const numLogitsShape = ITensor::makeShape({numLogits});
    lastTokenIdsHost->reshape(numLogitsShape);
    lastTokenIdsDevice->reshape(numLogitsShape);
    logitsIdsHost->reshape(numLogitsShape);

    if (transformerBuffers)
    {
        transformerBuffers->reshape(numSequences, numContextTokens + numGenTokens);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->reshape(numSequences);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->reshape();
    }

    if (modelConfig.useLoraPlugin())
    {
        loraBuffers->reshape(numSequences);
    }

    if (mMedusaBuffers)
    {
        mMedusaBuffers->reshape(
            numContextRequests, numGenRequests, modelConfig.getSpeculativeDecodingModulePtr()->getMaxDecodingTokens());
    }

    if (mLookaheadBuffers && modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        mLookaheadBuffers->reshape(
            numContextRequests, numGenRequests, modelConfig.getSpeculativeDecodingModulePtr()->getMaxDecodingTokens());
    }

    if (mExplicitDraftTokensBuffers)
    {
        mExplicitDraftTokensBuffers->reshape(numContextRequests, numGenRequests, modelConfig);
    }

    if (mEagleBuffers)
    {
        mEagleBuffers->reshape(numContextRequests, numGenRequests, modelConfig);
    }

    auto const numRequests = getNumRequests();
    auto const numRequestsShape = ITensor::makeShape({numRequests});
    seqSlots->reshape(numRequestsShape);
    seqSlotsDevice->reshape(numRequestsShape);

    auto const numTokens = getNumTokens();
    inputsIds->reshape(ITensor::makeShape({numTokens}));

    if (modelConfig.useMrope())
    {
        auto const mropeRotaryCosSinSize = modelConfig.getMaxPositionEmbeddings() * modelConfig.getRotaryEmbeddingDim();
        mropeRotaryCosSin->reshape(ITensor::makeShape({numSequences, mropeRotaryCosSinSize}));
        mropePositionDeltas->reshape(ITensor::makeShape({numSequences, 1}));
    }

    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = (!modelConfig.getPpReduceScatter() || worldConfig.isFirstPipelineParallelRank())
            ? modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()
            : modelConfig.getHiddenSize();

        auto const hiddenStatesShape = ITensor::makeShape({numTokens, hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    if (modelConfig.useLanguageAdapter())
    {
        languageAdapterRoutings->reshape(ITensor::makeShape({numTokens, 1}));
    }

    for (auto const& outputTensor : mAdditionalOutputTensors)
    {
        auto const& [name, tensor] = outputTensor;
        auto const& engine = runtime.getEngine();
        auto shape = engine.getTensorShape(name.c_str());
        TLLM_CHECK_WITH_INFO(
            shape.d[0] == -1, "First dimension of additional output tensor '%s' must be dynamic", name.c_str());
        shape.d[0] = numTokens;
        tensor->reshape(shape);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareBuffersForCudaGraph(SizeType32 maxSequenceLength)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(prepareBuffersForCudaGraph);

    TLLM_CHECK(numContextRequests == 0);

    if (transformerBuffers)
    {
        // Set pastKeyValueLength for graph capturing. This way we will capture graph with
        // maxKvCacheLengthRounded rounded to the next kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE.
        // MMHA will launch excessive amount of blocks and some of them will exit early during the actual launch.
        // We can reuse the same graph for the next kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE iterations.

        // make sure the size does not overflow the max allowed pastKvCacheLength
        auto const pastKvCacheLength = std::min(maxSequenceLength - 1, maxKvCacheLengthRounded);

        auto* pastKeyValueLengthsPtr = bufferCast<SizeType32>(*transformerBuffers->pastKeyValueLengths);
        std::fill_n(pastKeyValueLengthsPtr, getNumSequences(), pastKvCacheLength);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
    SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, runtime::decoder::DecoderState const& decoderState,
    kv_cache_manager::BaseKVCacheManager* kvCacheManagerPtr,
    kv_cache_manager::BaseKVCacheManager* crossKvCacheManagerPtr,
    rnn_state_manager::RnnStateManager* rnnStateManagerPtr, PeftTable const& peftTable,
    runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, bool trtOverlap, OptionalRef<runtime::ITensor const> newOutputTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersSetFromInputs);

    auto const& manager = runtime.getBufferManager();
    auto const& stream = runtime.getStream();

    // Fill requestTypes
    {
        auto* hostRequestTypes = bufferCast<runtime::RequestType>(*requestTypes);
        std::fill_n(hostRequestTypes, numContextRequests, runtime::RequestType::kCONTEXT);
        std::fill_n(hostRequestTypes + numContextRequests, numGenSequences, runtime::RequestType::kGENERATION);
    }

    SizeType32 totalInputSize = 0;
    std::vector<TokenIdType> inputHost;
    std::vector<SizeType32> positionIdsHost;
    std::vector<SizeType32> positionIdsHostRow2;
    std::vector<SizeType32> mropePositionDeltasHost;
    std::vector<SizeType32> languageAdapterRoutingsHost;

    auto* contextLengthsHostPtr = bufferCast<SizeType32>(*contextLengthsHost);
    auto* sequenceLengthsHostPtr = bufferCast<SizeType32>(*sequenceLengthsHost);
    auto* pastKeyValueLengthsPtr
        = transformerBuffers ? bufferCast<SizeType32>(*transformerBuffers->pastKeyValueLengths) : nullptr;
    SizeType32 totalNumLogits{0};
    auto* logitsIdsHostPtr = bufferCast<SizeType32>(*logitsIdsHost);
    bool const isChatGlm = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kChatGlm;
    bool const isGlm = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kGlm;
    auto const mropeRotaryCosSinSize = modelConfig.getMaxPositionEmbeddings() * modelConfig.getRotaryEmbeddingDim();

    {
        NVTX3_SCOPED_RANGE(seqSlotsLoop);
        auto* seqSlotIndices = bufferCast<SizeType32>(*seqSlots);

        SizeType32 batchIdx{0};
        for (auto const& requests : {contextRequests, genRequests})
        {
            for (auto const& llmReq : requests)
            {
                // Get position of the current sequence in the decoder
                auto const seqSlot = llmReq->mSeqSlot.value();
                seqSlotIndices[batchIdx] = seqSlot;
                ++batchIdx;
            }
        }

        TLLM_CHECK(seqSlots->getSize() == static_cast<std::size_t>(batchIdx));
        manager.copy(*seqSlots, *seqSlotsDevice);
    }

    // context preparation loop
    if (!contextRequests.empty())
    {
        NVTX3_SCOPED_RANGE(contextPrepareLoop);
        numContextLogits.resize(contextRequests.size());

        SizeType32 batchIdx{0};
        for (auto const& llmReq : contextRequests)
        {
            TLLM_CHECK_WITH_INFO(llmReq->isContextInitState() || llmReq->isDisaggGenerationTransmissionComplete(),
                "The request should be in context phase or disaggregated generation tranmissionComplete phase.");
            TLLM_CHECK_WITH_INFO(
                llmReq->getMaxNumGeneratedTokens() == 0, "Context request should not have generated tokens.");

            auto const& reqTokens = llmReq->getTokens(0);
            auto const& draftTokens = llmReq->getDraftTokens();
            auto const draftLength = llmReq->getNumDraftTokens();
            auto const& positionIds = llmReq->getPositionIds();

            auto const contextChunkSize = llmReq->getContextChunkSize();
            auto const beginCompute = llmReq->getContextCurrentPosition();
            auto const endCompute = beginCompute + contextChunkSize;
            inputHost.insert(inputHost.end(), reqTokens.begin() + beginCompute, reqTokens.begin() + endCompute);

            logitsIdsHostPtr[totalNumLogits++] = contextChunkSize;
            numContextLogits.at(batchIdx) = modelConfig.computeContextLogits() ? contextChunkSize : 1;

            if (llmReq->isLastContextChunk())
            {
                inputHost.insert(inputHost.end(), draftTokens->begin(), draftTokens->end());
                std::fill_n(logitsIdsHostPtr + totalNumLogits, draftLength, 1);
                totalNumLogits += draftLength;
            }
            auto const inputLength = contextChunkSize + (llmReq->isLastContextChunk() ? draftLength : 0);
            contextLengthsHostPtr[batchIdx] = inputLength;
            auto const sequenceLen = inputLength + llmReq->getContextCurrentPosition();
            sequenceLengthsHostPtr[batchIdx] = sequenceLen;

            if (static_cast<bool>(pastKeyValueLengthsPtr))
            {
                pastKeyValueLengthsPtr[batchIdx] = beginCompute + inputLength;
            }

            if (positionIds.has_value())
            {
                TLLM_CHECK_WITH_INFO(!(isChatGlm || isGlm), "ChatGLM-6B and Glm only use the default initialization");
                positionIdsHost.insert(positionIdsHost.end(), positionIds.value()->begin() + beginCompute,
                    positionIds.value()->begin() + endCompute);
            }
            else
            {
                if (isChatGlm)
                {
                    // Specialize for ChatGLM-6B with 2D-Position-Embedding
                    positionIdsHost.resize(totalInputSize + inputLength);
                    std::iota(std::begin(positionIdsHost) + totalInputSize, std::end(positionIdsHost), 0);
                    positionIdsHost.back() = positionIdsHost.back() - 1;

                    positionIdsHostRow2.resize(totalInputSize + inputLength);
                    positionIdsHostRow2.back() = 1;
                }
                else if (isGlm)
                {
                    // Specialize for GLM-10B with 2D-Position-Embedding and special value of the mask id position
                    auto start = inputHost.begin() + totalInputSize;
                    auto end = start + inputLength;
                    auto it = std::find_if(
                        start, end, [](SizeType32 id) { return id == 50260 || id == 50263 || id == 50264; });
                    llmReq->mMaskPosition = (it != end) ? std::distance(start, it) : maxContextLength;

                    positionIdsHost.resize(totalInputSize + inputLength);
                    std::iota(std::begin(positionIdsHost) + totalInputSize, std::end(positionIdsHost), 0);
                    positionIdsHost.back() = llmReq->mMaskPosition;

                    positionIdsHostRow2.resize(totalInputSize + inputLength);
                    positionIdsHostRow2.back() = 1;
                }
                else
                {
                    // Other models
                    positionIdsHost.resize(totalInputSize + inputLength);
                    std::iota(std::begin(positionIdsHost) + totalInputSize,
                        std::begin(positionIdsHost) + totalInputSize + inputLength, beginCompute);
                }
            }
            if (modelConfig.useMrope())
            {
                auto optMropeRotaryCosSin = llmReq->getMropeRotaryCosSin().value();
                TLLM_CHECK_WITH_INFO(optMropeRotaryCosSin->getShape().d[0] == mropeRotaryCosSinSize,
                    "Provided MropeRotarySinCos is %ld and expected is %d.\n", optMropeRotaryCosSin->getShape().d[0],
                    int(mropeRotaryCosSinSize));

                auto const mropeRotaryCosSinCtx = ITensor::slice(mropeRotaryCosSin, batchIdx, 1);
                manager.copy(*optMropeRotaryCosSin, *mropeRotaryCosSinCtx);
            }

            if (modelConfig.useLanguageAdapter())
            {
                auto const languageAdapterRouting = llmReq->getLanguageAdapterRouting(
                    modelConfig.getNumLanguages().value(), endCompute - beginCompute);
                languageAdapterRoutingsHost.insert(languageAdapterRoutingsHost.end(),
                    std::begin(languageAdapterRouting), std::end(languageAdapterRouting));
            }
            totalInputSize += inputLength;
            ++batchIdx;
        }

        if (rnnStateBuffers)
        {
            rnnStateBuffers->fillSlotMappings(contextRequests, rnnStateManagerPtr);
        }
    }

    // generation preparation loop
    if (!genRequests.empty())
    {
        NVTX3_SCOPED_RANGE(genPrepareLoop);

        auto const numContextRequests = static_cast<SizeType32>(contextRequests.size());
        auto numSequences = numContextRequests;
        for (auto const& llmReq : genRequests)
        {
            auto const reqBeamWidth = llmReq->getBeamWidthByIter();
            auto const draftLength = llmReq->getNumDraftTokens();
            auto const& draftTokens = llmReq->getDraftTokens();
            auto const numLogits = draftLength + reqBeamWidth;
            TLLM_CHECK(draftLength == 0 || reqBeamWidth == 1);

            auto const promptLen = llmReq->mPromptLen;
            auto const sequenceLen
                = promptLen + llmReq->getMaxNumGeneratedTokens() + static_cast<SizeType32>(trtOverlap);
            auto const& positionIds = llmReq->getPositionIds();
            for (int beam = 0; beam < reqBeamWidth; ++beam)
            {
                auto const numTokens = llmReq->getNumTokens(beam) + static_cast<SizeType32>(trtOverlap);
                // TODO: can this be removed completely?
                if (!trtOverlap)
                {
                    auto const lastToken = llmReq->getLastTokens(beam);
                    inputHost.push_back(lastToken);
                    if (draftLength > 0)
                    {
                        inputHost.insert(inputHost.end(), draftTokens->begin(), draftTokens->end());
                    }
                }

                // If model updates generation position ids do not append them here.
                if (!modelConfig.getSpeculativeDecodingMode().updatesPositionIds())
                {
                    if (positionIds.has_value())
                    {
                        TLLM_CHECK_WITH_INFO(
                            !(isChatGlm || isGlm), "ChatGLM-6B and Glm only use the default initialization");
                        auto last_context_position_id = positionIds.value()->back();
                        positionIdsHost.push_back(
                            static_cast<SizeType32>(last_context_position_id + sequenceLen - promptLen));
                    }
                    else
                    {
                        if (isChatGlm) // ChatGLM-6B
                        {
                            positionIdsHost.push_back(static_cast<SizeType32>(promptLen - 2));
                            positionIdsHostRow2.push_back(static_cast<SizeType32>(sequenceLen - promptLen + 1));
                        }
                        else if (isGlm)
                        {
                            positionIdsHost.push_back(llmReq->mMaskPosition);
                            positionIdsHostRow2.push_back(static_cast<SizeType32>(sequenceLen - promptLen + 1));
                        }
                        else // GPT / ChatGLM2-6B / ChatGLM3-6B / BART
                        {
                            // positionIds is just the size of tokens -1
                            positionIdsHost.push_back(numTokens - 1);
                        }
                    }
                }

                if (modelConfig.useMrope())
                {
                    auto optMropePositionDeltas = llmReq->getMropePositionDeltas().value();
                    mropePositionDeltasHost.push_back(optMropePositionDeltas);
                }

                if (modelConfig.useLanguageAdapter())
                {
                    // Generation requests only have one token per sequence
                    auto const languageAdapterRouting
                        = llmReq->getLanguageAdapterRouting(modelConfig.getNumLanguages().value(), 1);
                    languageAdapterRoutingsHost.insert(languageAdapterRoutingsHost.end(),
                        std::begin(languageAdapterRouting), std::end(languageAdapterRouting));
                }
            }

            if (static_cast<bool>(pastKeyValueLengthsPtr))
            {
                SizeType32 pastKeyValueLength = sequenceLen - 1;
                std::fill_n(pastKeyValueLengthsPtr + numSequences, reqBeamWidth, pastKeyValueLength);
            }
            totalInputSize += numLogits;

            std::fill_n(logitsIdsHostPtr + totalNumLogits, numLogits, 1);

            totalNumLogits += numLogits;

            if (rnnStateBuffers)
            {
                auto const seqSlot = llmReq->mSeqSlot.value();
                auto& rnnStateManager = *rnnStateManagerPtr;
                rnnStateManager.fillSlotMapping(*rnnStateBuffers->slotMappingHost, numSequences, seqSlot, reqBeamWidth);
            }
            numSequences += reqBeamWidth;
        }

        if (transformerBuffers && maxBeamWidth > 1)
        {
            transformerBuffers->copyCacheIndirection(genRequests, decoderState.getCacheIndirectionOutput(), stream);
        }

        numSequences = numContextRequests;
        for (auto const& llmReq : genRequests)
        {
            auto const reqBeamWidth = llmReq->getBeamWidthByIter();
            auto const draftLength = llmReq->getNumDraftTokens();

            auto const contextQLength = llmReq->mPromptLen + draftLength;
            auto const sequenceLen
                = contextQLength + llmReq->getMaxNumGeneratedTokens() + static_cast<SizeType32>(trtOverlap);

            std::fill_n(contextLengthsHostPtr + numSequences, reqBeamWidth, contextQLength);
            std::fill_n(sequenceLengthsHostPtr + numSequences, reqBeamWidth, sequenceLen);
            numSequences += reqBeamWidth;
        }
        if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
        {
            // copy from lookahead decoding buffer
            mLookaheadBuffers->setFromInputs(numContextRequests, numGenRequests, *requestTypes, *seqSlots,
                decoderState.getLookaheadBuffers(), runtime, modelConfig, worldConfig);
        }
    }

    // check skipCrossAttnBlocks
    if (transformerBuffers && modelConfig.skipCrossAttnBlocks())
    {
        bool isSkipCrossAttn = true;
        for (auto const& requests : {contextRequests, genRequests})
        {
            for (auto const& llmReq : requests)
            {
                bool tmpValue = false;
                if (llmReq->getSkipCrossAttnBlocks() != nullptr)
                {
                    manager.copy(*llmReq->getSkipCrossAttnBlocks(), &tmpValue);
                }
                isSkipCrossAttn &= tmpValue;
            }
        }
        transformerBuffers->copySkipCrossAttnBlocks(isSkipCrossAttn, runtime);
    }

    if (isChatGlm || isGlm)
    {
        positionIdsHost.reserve(totalInputSize * 2);
        positionIdsHost.insert(positionIdsHost.end(), positionIdsHostRow2.begin(), positionIdsHostRow2.end());
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->fill(contextRequests, genRequests, manager);
    }
    if (modelConfig.usePromptTuning())
    {
        promptTuningBuffers->fill(contextRequests, genRequests, manager, modelConfig.usePackedInput());
    }
    if (modelConfig.useLoraPlugin())
    {
        loraBuffers->fill(contextRequests, genRequests, peftTable, manager, modelConfig, worldConfig);
    }
    if (modelConfig.useMrope())
    {
        if (!mropePositionDeltasHost.empty())
        {
            auto mropePositionDeltasGen = ITensor::slice(mropePositionDeltas, 0, numGenSequences);
            manager.copy(mropePositionDeltasHost.data(), *mropePositionDeltasGen);
        }
    }

    {
        NVTX3_SCOPED_RANGE(bufferCopies);
        if (trtOverlap)
        {
            auto contextInputsIds = ITensor::slice(inputsIds, 0, numContextTokens);
            manager.copy(inputHost.data(), *contextInputsIds);

            if (!genRequests.empty())
            {
                auto generationInputsIds = ITensor::slice(inputsIds, numContextTokens);
                auto seqSlotsDeviceSlice = ITensor::slice(seqSlotsDevice, numContextRequests);
                runtime::kernels::invokeGatherBatch(
                    *generationInputsIds, *newOutputTokens, *seqSlotsDeviceSlice, maxBeamWidth, stream);
            }
        }
        else
        {
            manager.copy(inputHost.data(), *inputsIds);
        }
        // In generation phase, device ptr of context lengths need to be tiled.
        manager.copy(*contextLengthsHost, *contextLengthsDevice);
        manager.copy(*sequenceLengthsHost, *sequenceLengthsDevice);
        auto const logitsIdsHostRange = BufferRange<SizeType32>(*logitsIdsHost);
        auto lastTokenIdsHostRange = BufferRange<SizeType32>(*lastTokenIdsHost);
        common::stl_utils::inclusiveScan(
            logitsIdsHostRange.begin(), logitsIdsHostRange.end(), lastTokenIdsHostRange.begin());
        manager.copy(*lastTokenIdsHost, *lastTokenIdsDevice);
        if (transformerBuffers)
        {
            TensorPtr decoderPositionIds = modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
                ? mLookaheadBuffers->positionIdsDevice
                : nullptr;
            transformerBuffers->copyPositionIds(runtime, positionIdsHost, isChatGlm || isGlm, decoderPositionIds);
        }
        if (rnnStateBuffers)
        {
            rnnStateBuffers->copySlotMappingH2D(runtime);
        }
        if (modelConfig.useLanguageAdapter())
        {
            manager.copy(languageAdapterRoutingsHost.data(), *languageAdapterRoutings);
        }
    }

    if (transformerBuffers && static_cast<bool>(kvCacheManagerPtr))
    {
        transformerBuffers->copyKvBlockOffsets(
            contextRequests, genRequests, kvCacheManagerPtr, crossKvCacheManagerPtr, manager);
    }

    if (modelConfig.useCrossAttention())
    {
        transformerBuffers->copyCrossAttentionMasks(contextRequests, genRequests, contextLengthsDevice,
            encoderBuffers->inputLengths, maxContextLength, encoderBuffers->getMaxInputLengthInBatch(), runtime);
    }

    maxKvCacheLengthRounded = 0;
    if (static_cast<bool>(pastKeyValueLengthsPtr))
    {
        auto const maxKvCacheLength
            = *std::max_element(pastKeyValueLengthsPtr, pastKeyValueLengthsPtr + getNumSequences());
        // Round up kv cache length
        maxKvCacheLengthRounded = common::ceilDiv(maxKvCacheLength, kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE)
            * kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE;
    }

    if (modelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
    {
        if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
        {
            prepareExplicitDraftTokenBuffers(
                decoderState.getExplicitDraftTokensBuffers(), runtime, modelConfig, worldConfig);
        }
        if (modelConfig.getSpeculativeDecodingMode().isEagle())
        {
            prepareEagleBuffers(
                contextRequests, genRequests, decoderState.getEagleBuffers(), runtime, modelConfig, worldConfig);
        }
    }

    sync_check_cuda_error(stream.get());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareExplicitDraftTokenBuffers(
    runtime::ExplicitDraftTokensBuffers::Inputs const& explicitDraftTokensBuffers, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mExplicitDraftTokensBuffers);

    mExplicitDraftTokensBuffers->setFromInputs(numContextRequests, numGenRequests, *requestTypes, *seqSlots,
        explicitDraftTokensBuffers, *transformerBuffers->positionIds, modelConfig, worldConfig,
        runtime.getBufferManager(), runtime.getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareEagleBuffers(RequestVector const& contextRequests, RequestVector const& genRequests,
    runtime::EagleBuffers::Inputs const& eagleBuffers, TllmRuntime const& runtime, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mEagleBuffers);

    mEagleBuffers->setFromInputs(contextRequests, genRequests, *requestTypes, *seqSlots, eagleBuffers,
        runtime.getBufferManager(), modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::tuple<SizeType32, RuntimeBuffers::TensorMap const&, RuntimeBuffers::TensorMap&> RuntimeBuffers::prepareStep(
    RequestVector const& contextRequests, RequestVector const& genRequests, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, runtime::decoder::DecoderState const& decoderState,
    kv_cache_manager::BaseKVCacheManager* kvCacheManager, kv_cache_manager::BaseKVCacheManager* crossKvCacheManager,
    rnn_state_manager::RnnStateManager* rnnStateManager, PeftTable const& peftTable, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, bool gatherGenerationLogits, bool trtOverlap,
    OptionalRef<runtime::ITensor const> newOutputTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersPrepareStep);

    setBufferSizes(contextRequests, genRequests);
    reshape(runtime, modelConfig, worldConfig, gatherGenerationLogits);

    setFromInputs(contextRequests, genRequests, maxBeamWidth, maxAttentionWindow, decoderState, kvCacheManager,
        crossKvCacheManager, rnnStateManager, peftTable, runtime, modelConfig, worldConfig, trtOverlap,
        newOutputTokens);

    fillIOMaps(modelConfig, worldConfig);

    auto const numTokens = getNumTokens();
    auto const optProfileId = runtime.getOptProfileId(numTokens, ModelConfig::getOptProfilesSplitPoints());
    setContextIndex(optProfileId);
    TLLM_LOG_DEBUG("numTokens: %d, optProfileId: %d", numTokens, optProfileId);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {optProfileId, inputMap, outputMap};
}

void RuntimeBuffers::fillIOMaps(ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersFillIOMaps);

    inputMap.clear();
    outputMap.clear();

    if (transformerBuffers)
    {
        transformerBuffers->getBuffers(inputMap, outputMap, modelConfig);
    }
    if (rnnStateBuffers)
    {
        rnnStateBuffers->getBuffers(inputMap);
    }

    if (worldConfig.isLastPipelineParallelRank())
    {
        // feed a view to TensorRT runtime so reshaping does not change logits buffer
        outputMap.insert_or_assign(kLogitsTensorName, ITensor::view(logits));
    }
    else
    {
        outputMap.insert_or_assign(kHiddenStatesOutputTensorName, hiddenStates);
    }

    if (worldConfig.isFirstPipelineParallelRank())
    {
        inputMap.insert_or_assign(kInputIdsTensorName, inputsIds);
    }
    else
    {
        inputMap.insert_or_assign(kHiddenStatesInputTensorName, hiddenStates);
    }

    inputMap.insert_or_assign(kLastTokenIdsTensorName, lastTokenIdsDevice);

    inputMap.insert_or_assign(kHostRequestTypesTensorName, requestTypes);
    // In the generation phase, we still pass context lengths.
    inputMap.insert_or_assign(kContextLengthsTensorName, contextLengthsDevice);
    inputMap.insert_or_assign(kHostContextLengthsTensorName, contextLengthsHost);
    inputMap.insert_or_assign(kSequenceLengthsTensorName, sequenceLengthsDevice);

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->insertInputTensors(inputMap);
    }
    if (modelConfig.usePromptTuning())
    {
        auto const& promptTuningParams = promptTuningBuffers->mPromptTuningParams;
        inputMap.insert_or_assign(kPromptEmbeddingTableTensorName, promptTuningParams.embeddingTable);
        inputMap.insert_or_assign(kTasksTensorName, promptTuningParams.tasks);
        inputMap.insert_or_assign(kPromptVocabSizeTensorName, promptTuningParams.vocabSize);
    }
    if (modelConfig.useMrope())
    {

        inputMap.insert_or_assign(kMRopeRotaryCosSinTensorName, mropeRotaryCosSin);
        inputMap.insert_or_assign(kMRopePositionDeltasTensorName, mropePositionDeltas);
    }
    if (modelConfig.useLoraPlugin())
    {
        loraBuffers->insertInputTensors(inputMap, loraBuffers->mLoraWeightsPointersHost,
            loraBuffers->mLoraAdapterSizesHost, modelConfig, worldConfig);
    }
    if (modelConfig.useLanguageAdapter())
    {
        inputMap.insert_or_assign("language_adapter_routings", languageAdapterRoutings);
    }

    if (mMedusaBuffers)
    {
        mMedusaBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }
    if (mLookaheadBuffers)
    {
        mLookaheadBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }
    if (mExplicitDraftTokensBuffers)
    {
        mExplicitDraftTokensBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }
    if (mEagleBuffers)
    {
        mEagleBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }

    for (auto const& outputTensor : mAdditionalOutputTensors)
    {
        outputMap.insert_or_assign(outputTensor.first, outputTensor.second);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
