/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/runtimeBuffers.h"

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <algorithm>
#include <iostream>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

RuntimeBuffers::GenerationConfig RuntimeBuffers::GenerationConfig::fromInput(ITensor const& inputIds,
    ITensor const& inputLengthsHost, bool const inputPacked, SizeType const beamWidth,
    SizeType const maxAttentionWindow, SizeType const sinkTokenLength, SizeType const maxSequenceLength)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = static_cast<SizeType>(inputLengthsHost.getSize());

    auto const* inputLengthsPtr = bufferCast<SizeType>(inputLengthsHost);
    SizeType maxInputLength = *std::max_element(inputLengthsPtr, inputLengthsPtr + batchSize);

    auto const& inputShape = inputIds.getShape();
    SizeType inputLengthSum{0};
    if (inputPacked)
    {
        inputLengthSum = std::accumulate(inputLengthsPtr, inputLengthsPtr + batchSize, 0);
        TLLM_CHECK_WITH_INFO(inputShape.nbDims == 1 || inputShape.nbDims == 2,
            "Packed input must have shape [<sum of input lengths>] or [1, <sum of input lengths>].");
        if (inputShape.nbDims == 1)
        {
            TLLM_CHECK_WITH_INFO(inputShape.d[0] == inputLengthSum,
                "Packed 1D input must have shape [<sum of input lengths>]. Expected (Infer from inputLengths): [%d], "
                "supplied: [%d]",
                inputLengthSum, inputShape.d[0]);
        }
        else if (inputShape.nbDims == 2)
        {
            TLLM_CHECK_WITH_INFO(inputShape.d[1] == inputLengthSum,
                "Packed 2D input must have shape [1, <sum of input lengths>]. Expected (Infer from inputLengths): [1, "
                "%d], supplied: [%d, %d]",
                inputLengthSum, inputShape.d[0], inputShape.d[1]);
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(inputShape.d[0] == batchSize && inputShape.d[1] >= maxInputLength,
            "Padded input must have shape [batch size, max input length]");
        maxInputLength = inputShape.d[1];
    }

    TLLM_CHECK_WITH_INFO(maxInputLength < maxSequenceLength,
        "Max input length is equal to or larger that maxSequenceLength given in setup. No new tokens can be "
        "generated.");

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return GenerationConfig{
        batchSize, beamWidth, maxInputLength, maxAttentionWindow, sinkTokenLength, maxSequenceLength, inputLengthSum};
}

void RuntimeBuffers::clear()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    contextLengthsHost = nullptr;
    contextLengthsDevice = nullptr;

    logits = nullptr;
    sequenceLengths = nullptr;
    pastKeyValueLengths = nullptr;
    attentionMask = nullptr;
    positionIds = nullptr;
    lastTokenIds = nullptr;
    requestTypes = nullptr;

    presentKeysVals.clear();
    presentKeysValsAlt.clear();
    kvCacheBlockPointersHost = nullptr;
    kvCacheBlockPointersDevice = nullptr;

    cacheIndirectionDecoderInput = nullptr;
    cacheIndirectionDecoderOutput = nullptr;

    cumLogProbs = nullptr;
    logProbs = nullptr;

    hiddenStates = nullptr;

    allocated = false;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::clearTensorMaps()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    for (auto& buffer : inputBuffers)
        buffer.clear();
    for (auto& buffer : outputBuffers)
        buffer.clear();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::create(TllmRuntime& runtime, GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = runtime.getBufferManager();
    auto& engine = runtime.getEngine();

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = engine.getTensorDataType("logits");
        logits = manager.emptyTensor(MemoryType::kGPU, logitsType);
        originalLogitsPtr = logits;

        allGenerationLogits = manager.emptyTensor(MemoryType::kGPU, logitsType);
        if (modelConfig.computeGenerationLogits())
        {
            cacheGenerationFragmentPointerDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT64);
            cacheGenerationFragmentPointerHost = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT64);

            generationLogitsFragments = std::make_shared<std::vector<TensorPtr>>();
        }
    }

    contextLengthsHost = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT32);
    lastTokenIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    nvinfer1::DataType kvDtype;
    if (modelConfig.usePagedKvCache())
    {
        if (modelConfig.getQuantMode().hasFp8KvCache())
        {
            kvDtype = nvinfer1::DataType::kFP8;
        }
        else if (modelConfig.getQuantMode().hasInt8KvCache())
        {
            kvDtype = nvinfer1::DataType::kINT8;
        }
        else
        {
            kvDtype = modelConfig.getDataType();
        }
    }
    else
    {
        kvDtype = modelConfig.getQuantMode().hasFp8KvCache()
            ? nvinfer1::DataType::kFP8
            : engine.getTensorDataType(("present_key_value_" + std::to_string(firstLayerId)).c_str());
    }

    if (modelConfig.usePagedKvCache())
    {
        auto const kvCacheBlockPointersType
            = engine.getTensorDataType(("kv_cache_block_pointers_" + std::to_string(firstLayerId)).c_str());
        kvCacheBlockPointersHost = manager.emptyTensor(MemoryType::kCPU, kvCacheBlockPointersType);
        kvCacheBlockPointersDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockPointersType);
    }
    else
    {
        presentKeysVals = utils::createBufferVector(runtime, localNbLayers, MemoryType::kGPU, kvDtype);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        maxAttentionWindows
            = utils::createBufferVector(runtime, localNbLayers, MemoryType::kCPU, nvinfer1::DataType::kINT32);
        sinkTokenLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    }
    else
    {
        presentKeysValsAlt = utils::createBufferVector(runtime, localNbLayers, MemoryType::kGPU, kvDtype);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        requestTypes = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    }

    cacheIndirectionDecoderInput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    cacheIndirectionDecoderOutput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    nbFinished = BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::initFromInput(ITensor const& inputIds, TensorPtr const& inputLengths, bool inputPacked,
    SizeType beamWidth, SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxSequenceLength,
    BufferManager& manager)
{
    contextLengthsDevice = inputLengths;
    contextLengthsHost->reshape(inputLengths->getShape());
    manager.copy(*contextLengthsDevice, *contextLengthsHost);
    manager.getStream().synchronize(); // wait for context lengths to be copied to host

    generationConfig = RuntimeBuffers::GenerationConfig::fromInput(
        inputIds, *contextLengthsHost, inputPacked, beamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength);
}

void RuntimeBuffers::reshape(GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxAttentionWindow = generationConfig.maxAttentionWindow;
    auto const sinkTokenLen = generationConfig.sinkTokenLength;
    auto const maxSeqLength = generationConfig.maxSeqLength;
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    if (worldConfig.isLastPipelineParallelRank())
    {
        if (modelConfig.computeContextLogits())
        {
            if (!modelConfig.computeGenerationLogits())
            {
                // If only enable computeContextLogits, also need to have a generation buffer to store the last token of
                // context
                allGenerationLogits->reshape(ITensor::makeShape({1, batchSize, beamWidth, vocabSizePadded}));
            }
        }
        else
        {
            // If only gather generation logits
            if (modelConfig.computeGenerationLogits())
            {
                logits = originalLogitsPtr; // logits point to original buffer
            }
            logits->reshape(ITensor::makeShape({batchSize, 1, vocabSizePadded}));
        }

        if (modelConfig.computeGenerationLogits())
        {
            allGenerationLogits->reshape(
                ITensor::makeShape({(generationConfig.maxSeqLength - generationConfig.maxInputLength), batchSize,
                    beamWidth, vocabSizePadded}));

            cacheGenerationFragmentPointerDevice->reshape(
                ITensor::makeShape({batchSize, (generationConfig.maxSeqLength - generationConfig.maxInputLength)}));
            cacheGenerationFragmentPointerHost->reshape(
                ITensor::makeShape({batchSize, (generationConfig.maxSeqLength - generationConfig.maxInputLength)}));
        }
    }

    lastTokenIds->reshape(ITensor::makeShape({batchSize}));

    auto kvCacheReserve = ITensor::makeShape(
        {batchSize, 2, modelConfig.getNbKvHeads(), maxAttentionWindow, modelConfig.getSizePerHead()});
    auto kvCacheShape
        = ITensor::makeShape({batchSize, 2, modelConfig.getNbKvHeads(), maxInputLength, modelConfig.getSizePerHead()});
    if (modelConfig.usePagedKvCache())
    {
        auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
        auto const tokensPerBlock = modelConfig.getTokensPerBlock();
        SizeType bubbleLen
            = (sinkTokenLen % tokensPerBlock == 0) ? 0 : tokensPerBlock - (sinkTokenLen % tokensPerBlock);
        auto maxBlocksPerSeq = tc::ceilDiv(maxAttentionWindow + bubbleLen, tokensPerBlock);
        // If beamWidth > 1, use one more block for each sequence in the paged kv cache to avoid dropping the needed
        // tokens, when enabling cyclic kv cache.
        if (beamWidth > 1 && maxSeqLength > maxAttentionWindow)
        {
            maxBlocksPerSeq += 1;
        }

        // reserve batchSize * beamWidth and resize to batchSize
        auto cacheBlockPointersShape = ITensor::makeShape({localNbLayers, batchSize * beamWidth, 2, maxBlocksPerSeq});
        kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
        cacheBlockPointersShape.d[1] = batchSize;
        kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
    }
    else
    {
        utils::reshapeBufferVector(presentKeysVals, kvCacheReserve);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        pastKeyValueLengths->reshape(ITensor::makeShape({batchSize}));
        requestTypes->reshape(ITensor::makeShape({batchSize}));
        utils::reshapeBufferVector(maxAttentionWindows, ITensor::makeShape({1}));
        sinkTokenLengths->reshape(ITensor::makeShape({1}));
    }
    else
    {
        utils::reshapeBufferVector(presentKeysValsAlt, kvCacheReserve);
        // present KV cache tensors will be reshaped by shape inference.
        // reshape to the required shape here to make context batch slicing work correctly.
        utils::reshapeBufferVector(presentKeysVals, kvCacheShape);
    }

    auto const cacheIndirShape = ITensor::makeShape({batchSize, beamWidth, maxAttentionWindow});
    cacheIndirectionDecoderInput->reshape(cacheIndirShape);
    cacheIndirectionDecoderOutput->reshape(cacheIndirShape);

    if (worldConfig.isPipelineParallel())
    {
        // reserve max size
        auto const maxNumTokens = std::max(beamWidth, maxInputLength);
        auto const hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();
        auto const hiddenStatesShape = ITensor::makeShape(
            {batchSize, maxNumTokens, hiddenSize}); // reserve space in traditional [bs, seq_len, hidden_state] way.
        hiddenStates->reshape(hiddenStatesShape);
    }

    allocated = true;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::reset(BufferManager& manager)
{
    clearTensorMaps();
    manager.setZero(*cacheIndirectionDecoderInput);
    manager.setZero(*cacheIndirectionDecoderOutput);
}

std::vector<RuntimeBuffers> RuntimeBuffers::split(
    SizeType contextBatchSize, GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    std::vector<RuntimeBuffers> bufferSlices;
    auto const generationBatchSize = generationConfig.batchSize;
    bufferSlices.reserve(tc::ceilDiv(generationBatchSize, contextBatchSize));
    if (contextBatchSize >= generationBatchSize)
    {
        bufferSlices.emplace_back(*this);
    }
    else
    {
        for (auto offset = 0; offset < generationBatchSize; offset += contextBatchSize)
        {
            auto const batchSize = std::min(contextBatchSize, generationBatchSize - offset);
            auto& buffers = bufferSlices.emplace_back();
            buffers.generationConfig = generationConfig;
            buffers.generationConfig.batchSize = batchSize;

            buffers.contextLengthsHost = ITensor::slice(contextLengthsHost, offset, batchSize);
            buffers.contextLengthsDevice = ITensor::slice(contextLengthsDevice, offset, batchSize);

            if (worldConfig.isLastPipelineParallelRank() && !modelConfig.computeContextLogits())
            {
                buffers.logits = ITensor::slice(logits, offset, batchSize);
            }

            buffers.lastTokenIds = ITensor::slice(lastTokenIds, offset, batchSize);

            if (modelConfig.usePagedKvCache())
            {
                auto const& realCacheBlockPointersShape = kvCacheBlockPointersHost->getShape();
                auto const localNbLayers = realCacheBlockPointersShape.d[0];
                auto const maxBlocksPerSeq = realCacheBlockPointersShape.d[3];

                // enable slicing by moving generationBatchSize to first dim
                auto const fakeCacheBlockPointersShape
                    = ITensor::makeShape({generationBatchSize, localNbLayers, 2, maxBlocksPerSeq});
                TensorPtr kvCacheBlockPointersHostView{
                    ITensor::view(kvCacheBlockPointersHost, fakeCacheBlockPointersShape)};
                TensorPtr kvCacheBlockPointersDeviceView{
                    ITensor::view(kvCacheBlockPointersDevice, fakeCacheBlockPointersShape)};

                // slice and reshape to correct shape
                auto const cacheBlockPointersShape = ITensor::makeShape({localNbLayers, batchSize, 2, maxBlocksPerSeq});
                buffers.kvCacheBlockPointersHost = ITensor::slice(kvCacheBlockPointersHostView, offset, batchSize);
                buffers.kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
                buffers.kvCacheBlockPointersDevice = ITensor::slice(kvCacheBlockPointersDeviceView, offset, batchSize);
                buffers.kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
            }
            else
            {
                buffers.presentKeysVals = utils::sliceBufferVector(presentKeysVals, offset, batchSize);
            }

            if (modelConfig.useGptAttentionPlugin())
            {
                buffers.pastKeyValueLengths = ITensor::slice(pastKeyValueLengths, offset, batchSize);
                buffers.maxAttentionWindows = maxAttentionWindows;
                buffers.sinkTokenLengths = sinkTokenLengths;
                buffers.requestTypes = ITensor::slice(requestTypes, offset, batchSize);
            }
            else
            {
                buffers.presentKeysValsAlt = utils::sliceBufferVector(presentKeysValsAlt, offset, batchSize);
            }

            if (worldConfig.isPipelineParallel())
            {
                TLLM_CHECK_WITH_INFO(hiddenStates->getShape().nbDims == 3,
                    "Invalid shape for hiddenStates."); // Expect hiddens states shape to be [bs, seq_len, hidden_size]
                                                        // at generation buffer split stage.
                buffers.hiddenStates = ITensor::slice(hiddenStates, offset, batchSize);
            }

            buffers.cacheIndirectionDecoderOutput = ITensor::slice(cacheIndirectionDecoderOutput, offset, batchSize);

            if (modelConfig.usePromptTuning())
            {
                auto const& ptuningEnabled = promptTuningParams.promptTuningEnabled;
                buffers.promptTuningParams.promptTuningEnabled
                    = std::vector<bool>(ptuningEnabled.begin() + offset, ptuningEnabled.begin() + offset + batchSize);

                buffers.promptTuningParams.tasks = ITensor::slice(promptTuningParams.tasks, offset, batchSize);
            }
        }
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return bufferSlices;
}

void RuntimeBuffers::gatherLastTokenLogits(
    BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(modelConfig.computeContextLogits(),
        "Gather last token logits is only necessary when context logits are computed");

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        TensorPtr tiledTensor = ITensor::slice(allGenerationLogits, 0, 1);
        tiledTensor->squeeze(0);
        kernels::gatherLastTokenLogits(*tiledTensor, *logits, *lastTokenIds, manager.getStream());
        manager.getStream().synchronize();

        std::swap(logits, tiledTensor);
        if (modelConfig.usePackedInput())
        {
            tiledTensor->reshape(
                ITensor::makeShape({generationConfig.inputLengthSum, vocabSizePadded})); // [packedSize, vocabSize]
        }
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::copyAttentionMasks(std::vector<RuntimeBuffers> const& contextBatches, BufferManager& manager)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = generationConfig.batchSize;
    auto const maxInputLength = generationConfig.maxInputLength;

    // TODO(rkobus) include tiling
    attentionMask = manager.gpu(ITensor::makeShape({batchSize, maxInputLength}), nvinfer1::DataType::kINT32);

    auto const numContextBatches = static_cast<SizeType>(contextBatches.size());
    auto offset = 0;
    for (auto contextBatchId = 0; contextBatchId < numContextBatches; ++contextBatchId)
    {
        auto& buffers = contextBatches.at(contextBatchId);
        auto contextBatchSize = buffers.generationConfig.batchSize;
        auto attentionMaskSlice = ITensor::slice(attentionMask, offset, contextBatchSize);
        manager.copy(*buffers.attentionMask, *attentionMaskSlice);
        offset += contextBatchSize;
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::tile(BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const beamWidth = generationConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(beamWidth > 1, "Tiling is only necessary for beam search.");

    // Note: If computeContextLogits is true, the copy/expansion is performed in gatherLastTokenLogits.
    if (worldConfig.isLastPipelineParallelRank() && !modelConfig.computeContextLogits())
    {
        // logits needs beamWidth in second dimension
        auto logitsShape = logits->getShape();
        logitsShape.d[1] *= beamWidth;
        utils::tileBufferReplace(logits, beamWidth, manager);
        logits->reshape(logitsShape);
    }

    utils::tileBufferReplace(contextLengthsDevice, beamWidth, manager);

    if (modelConfig.useGptAttentionPlugin())
    {
        utils::tileCpuBufferReplace(contextLengthsHost, beamWidth, manager);
        utils::tileCpuBufferReplace(pastKeyValueLengths, beamWidth, manager);
    }
    else
    {
        utils::tileBufferReplace(attentionMask, beamWidth, manager);
    }

    if (!modelConfig.usePagedKvCache())
    {
        for (auto& buffer : presentKeysVals)
            utils::tileBufferReplace(buffer, beamWidth, manager);
        for (auto& buffer : presentKeysValsAlt)
            utils::tileBufferReplace(buffer, beamWidth, manager);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::postContextStep(std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;

    if (modelConfig.useGptAttentionPlugin())
    {
        requestTypes->reshape(ITensor::makeShape({batchSize * beamWidth}));
        auto hostRequestTypes = bufferCast<int32_t>(*requestTypes);
        std::fill_n(hostRequestTypes, requestTypes->getSize(), 1);
    }
    else
    {
        copyAttentionMasks(contextBuffers, manager);
    }

    // TODO(rkobus) handle this more gracefully
    positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (modelConfig.computeContextLogits())
    {
        gatherLastTokenLogits(manager, modelConfig, worldConfig);
    }

    if (beamWidth > 1)
    {
        tile(manager, modelConfig, worldConfig);
    }

    // use output lengths after context step
    manager.copy(*contextLengthsDevice, *outputLengths);
    sequenceLengths = ITensor::view(outputLengths);
    sequenceLengths->reshape(ITensor::makeShape({batchSize * beamWidth}));
    // no need to copy data in lastTokenIds because it is overwritten in prepareNextStep
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth}));

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto cacheBlockPointersShape = kvCacheBlockPointersHost->getShape();
        cacheBlockPointersShape.d[1] = batchSize * beamWidth;
        kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
    }

    if (modelConfig.usePromptTuning())
    {
        std::vector<SizeType> reqBeamWidths(batchSize, beamWidth);
        //// Note: reqPromptLenghts won't be used
        std::vector<SizeType> reqPromptLengths;
        // Copy the generationInput tasks to host
        promptTuningTasksHost = manager.copyFrom(*promptTuningParams.tasks, MemoryType::kPINNED);
        // Update the promptTuningParams tasks tensor
        promptTuningParams.fillTasksTensor(promptTuningTasksHost, batchSize, 0, reqBeamWidths, reqPromptLengths,
            manager, modelConfig.usePackedInput());
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareContextStep(TensorPtr const& inputIds, TokenIdType const padId, BufferManager& manager,
    KvCacheManager const* kvCacheManager, SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const maxInputLength = generationConfig.maxInputLength;

    // use context lengths only in context step
    sequenceLengths = contextLengthsDevice;

    // get local number of layers.
    auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());

    if (modelConfig.useGptAttentionPlugin())
    {
        auto pastKeyValueLengthsPtr = bufferCast<SizeType>(*pastKeyValueLengths);
        TLLM_CHECK(pastKeyValueLengths->getSize() == static_cast<std::size_t>(batchSize));

        auto RequestTypesPtr = bufferCast<int32_t>(*requestTypes);
        TLLM_CHECK(requestTypes->getSize() == static_cast<std::size_t>(batchSize));
        std::fill_n(RequestTypesPtr, batchSize, 0);

        // Set maxAttentionWindows buffer and sinkTokenLengths to the same value currently.
        for (auto layer = 0; layer < localNbLayers; ++layer)
        {
            bufferCast<SizeType>(*maxAttentionWindows[layer])[0] = generationConfig.maxAttentionWindow;
        }
        bufferCast<SizeType>(*sinkTokenLengths)[0] = generationConfig.sinkTokenLength;

        auto const& inputShape = inputIds->getShape();
        auto const contextLengthsHostPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const modelVariant = modelConfig.getModelVariant();

        if (modelVariant == GptModelConfig::ModelVariant::kGpt)
        {
            auto const inputSize = inputIds->getSize();
            std::vector<SizeType> positionIdsVec(inputSize);
            auto begin = std::begin(positionIdsVec);
            for (SizeType i = 0; i < batchSize; ++i)
            {
                auto end = begin + (modelConfig.usePackedInput() ? contextLengthsHostPtr[i] : maxInputLength);
                std::iota(begin, end, 0);
                begin = end;
            }
            positionIds = manager.copyFrom(positionIdsVec, inputShape, MemoryType::kGPU);
        }
        else if (modelVariant == GptModelConfig::ModelVariant::kGlm)
        {
            auto const positionIdsVec = getPositionIdsContextPhaseGlm(batchSize, maxInputLength, contextLengthsHostPtr,
                modelConfig.useGptAttentionPlugin(), modelConfig.usePackedInput());
            if (modelConfig.usePackedInput())
            {
                int num_tokens = (int) positionIdsVec.size() / 2;
                auto const positionIdsShape = ITensor::makeShape({2, num_tokens});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
            else
            {
                auto const positionIdsShape = ITensor::makeShape({batchSize, 2, maxInputLength});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
        }
        else
        {
            TLLM_THROW("Unsupported model variant");
        }

        for (SizeType i = 0; i < batchSize; ++i)
        {
            pastKeyValueLengthsPtr[i] = contextLengthsHostPtr[i];
        }

        if (worldConfig.isPipelineParallel())
        {
            auto const hiddenSize
                = hiddenStates->getShape().nbDims == 2 ? hiddenStates->getShape().d[1] : hiddenStates->getShape().d[2];
            auto const hiddenStatesShape = modelConfig.usePackedInput()
                ? ITensor::makeShape({inputShape.d[0], hiddenSize})
                : ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
            hiddenStates->reshape(hiddenStatesShape);
        }

        if (modelConfig.usePromptTuning())
        {
            std::vector<SizeType> reqBeamWidths(batchSize, 1);
            std::vector<SizeType> reqPromptLengths;
            for (SizeType i = 0; i < batchSize; ++i)
            {
                reqPromptLengths.push_back(contextLengthsHostPtr[i]);
            }

            // Copy the generationInput tasks to host
            promptTuningTasksHost = manager.copyFrom(*promptTuningParams.tasks, MemoryType::kPINNED);

            // Update the tasks tensor
            promptTuningParams.fillTasksTensor(promptTuningTasksHost, batchSize, batchSize, reqBeamWidths,
                reqPromptLengths, manager, modelConfig.usePackedInput());
        }
    }
    else
    {
        attentionMask = manager.copyFrom(*inputIds, MemoryType::kGPU);
        kernels::invokeBuildAttentionMask(*attentionMask, padId, stream);

        auto attentionMaskHost = manager.copyFrom(*attentionMask, MemoryType::kCPU);
        auto const* attentionMaskData = reinterpret_cast<SizeType const*>(attentionMaskHost->data());
        std::vector<SizeType> positionIdsVec(attentionMask->getSize());
        for (SizeType i = 0; i < batchSize; ++i)
        {
            tc::stl_utils::exclusiveScan(attentionMaskData + i * maxInputLength,
                attentionMaskData + (i + 1) * maxInputLength, std::begin(positionIdsVec) + i * maxInputLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskData[i] == 0)
                positionIdsVec[i] = 1;
        positionIds = manager.copyFrom(positionIdsVec, attentionMask->getShape(), MemoryType::kGPU);
    }

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto constexpr contextBeamWidth = 1;
        kvCacheManager->getBlockPointersOfBatch(
            *kvCacheBlockPointersHost, firstBatchSlotIdx, batchSize, contextBeamWidth);
        manager.copy(*kvCacheBlockPointersHost, *kvCacheBlockPointersDevice);
    }

    if (modelConfig.usePackedInput())
    {
        kernels::invokeInclusiveSum(*lastTokenIds, *contextLengthsDevice, manager, stream);
    }
    else
    {
        manager.copy(*contextLengthsDevice, *lastTokenIds);
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

RuntimeBuffers::TensorPtr RuntimeBuffers::prepareNextStep(SizeType const step, BufferManager& manager,
    KvCacheManager* kvCacheManager, SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const beamWidth = generationConfig.beamWidth;

    nvinfer1::Dims inputShape;
    if (modelConfig.usePackedInput())
    {
        // batch in last dim
        inputShape = ITensor::makeShape({batchSize * beamWidth});
    }
    else
    {
        // batch in first dim
        inputShape = ITensor::makeShape({batchSize * beamWidth, 1});
    }
    auto nextInputIds = newTokens ? ITensor::view(newTokens, inputShape) : TensorPtr{};

    if (modelConfig.useGptAttentionPlugin())
    {
        auto const contextLengthsHostPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const pastKeyValueLengthsPtr = bufferCast<SizeType>(*pastKeyValueLengths);
        auto const tensorBatchSize = static_cast<SizeType>(pastKeyValueLengths->getSize());
        SizeType const srcStride{modelConfig.useGptAttentionPlugin() ? 1 : beamWidth};
        TLLM_CHECK(static_cast<std::size_t>(tensorBatchSize * srcStride) == contextLengthsDevice->getSize());
        for (SizeType i = 0; i < tensorBatchSize; ++i)
        {
            pastKeyValueLengthsPtr[i] = contextLengthsHostPtr[i * srcStride] + step;
        }

        auto const modelVariant = modelConfig.getModelVariant();

        if (modelVariant == GptModelConfig::ModelVariant::kGpt)
        {
            positionIds->reshape(inputShape);
            manager.copy(*contextLengthsDevice, *positionIds);
            kernels::invokeAdd(*positionIds, step, stream);
        }
        else if (modelVariant == GptModelConfig::ModelVariant::kGlm)
        {
            auto const positionIdsVec = getPositionIdsGenerationPhaseGlm(batchSize, beamWidth, step,
                contextLengthsHostPtr, modelConfig.useGptAttentionPlugin(), modelConfig.usePackedInput());
            if (modelConfig.usePackedInput())
            {
                auto const positionIdsShape = ITensor::makeShape({2, batchSize * beamWidth});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
            else
            {
                auto const positionIdsShape = ITensor::makeShape({batchSize * beamWidth, 2, 1});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
        }
        else
        {
            TLLM_THROW("Unsupported model variant");
        }

        if (worldConfig.isPipelineParallel())
        {
            auto const hiddenSize
                = hiddenStates->getShape().nbDims == 2 ? hiddenStates->getShape().d[1] : hiddenStates->getShape().d[2];
            auto const hiddenStatesShape = modelConfig.usePackedInput()
                ? ITensor::makeShape({inputShape.d[0], hiddenSize})
                : ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
            hiddenStates->reshape(hiddenStatesShape);
        }
    }
    else
    {
        auto const& shape = attentionMask->getShape();
        auto const nbInputs = shape.d[0];
        auto const oldLength = shape.d[1];
        auto const newLength = oldLength + 1;
        auto const newShape = ITensor::makeShape({nbInputs, newLength});

        TensorPtr newAttentionMask = manager.gpu(newShape, attentionMask->getDataType());
        kernels::invokeExtendAttentionMask(*newAttentionMask, *attentionMask, stream);
        attentionMask = newAttentionMask;

        auto attentionMaskHost = manager.copyFrom(*attentionMask, MemoryType::kCPU);
        auto const* attentionMaskPtr = bufferCast<SizeType>(*attentionMaskHost);

        // TODO old positionIds could be recovered to avoid scan
        std::vector<SizeType> positionIdsVec(attentionMask->getSize());
        for (SizeType i = 0; i < nbInputs; ++i)
        {
            tc::stl_utils::exclusiveScan(attentionMaskPtr + i * newLength, attentionMaskPtr + (i + 1) * newLength,
                std::begin(positionIdsVec) + i * newLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskPtr[i] == 0)
                positionIdsVec[i] = 1;
        std::vector<SizeType> positionIdsEndVec(nbInputs);
        for (SizeType i = 0; i < nbInputs; ++i)
            positionIdsEndVec[i] = positionIdsVec[(i + 1) * newLength - 1];

        positionIds = manager.copyFrom(positionIdsEndVec, ITensor::makeShape({nbInputs, 1}), MemoryType::kGPU);
    }

    if (modelConfig.usePagedKvCache())
    {
        for (auto batchIdx = firstBatchSlotIdx; batchIdx < firstBatchSlotIdx + batchSize; ++batchIdx)
        {
            kvCacheManager->addToken(batchIdx);
        }
        kvCacheManager->getBlockPointersOfBatch(*kvCacheBlockPointersHost, firstBatchSlotIdx, batchSize, beamWidth);
        manager.copy(*kvCacheBlockPointersHost, *kvCacheBlockPointersDevice);
    }

    kernels::invokeFill(*lastTokenIds, 1, stream);
    if (modelConfig.usePackedInput())
    {
        kernels::invokeInclusiveSum(*lastTokenIds, *lastTokenIds, manager, stream);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return nextInputIds;
}

void RuntimeBuffers::getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType const step,
    TensorPtr const& inputIds, TensorPtr const& commPtrs, GptModelConfig const& modelConfig,
    WorldConfig const& worldConfig) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    inputBuffers.clear();
    outputBuffers.clear();

    if (worldConfig.isLastPipelineParallelRank())
    {
        // feed a view to TensorRT runtime so reshaping does not change logits buffer
        outputBuffers.insert_or_assign("logits", ITensor::view(logits));
    }
    else
    {
        outputBuffers.insert_or_assign("hidden_states_output", hiddenStates);
    }

    if (worldConfig.isFirstPipelineParallelRank())
    {
        inputBuffers.insert_or_assign("input_ids", inputIds);
    }
    else
    {
        inputBuffers.insert_or_assign("hidden_states_input", hiddenStates);
    }

    inputBuffers.insert_or_assign("context_lengths", contextLengthsDevice);
    if (!modelConfig.computeContextLogits())
    {
        inputBuffers.insert_or_assign("last_token_ids", lastTokenIds);
    }
    inputBuffers.insert_or_assign("position_ids", positionIds);

    auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    if (modelConfig.useGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("cache_indirection", cacheIndirectionDecoderOutput);
        inputBuffers.insert_or_assign("host_past_key_value_lengths", pastKeyValueLengths);
        inputBuffers.insert_or_assign("host_request_types", requestTypes);
        inputBuffers.insert_or_assign("sequence_length", sequenceLengths);
        inputBuffers.insert_or_assign("host_sink_token_length", sinkTokenLengths);
        utils::insertTensorVector(inputBuffers, "host_max_attention_window_size_", maxAttentionWindows, firstLayerId);

        if (modelConfig.usePackedInput())
        {
            inputBuffers.insert_or_assign("host_context_lengths", contextLengthsHost);
        }
        if (modelConfig.usePagedKvCache())
        {
            utils::insertTensorSlices(
                inputBuffers, "kv_cache_block_pointers_", kvCacheBlockPointersDevice, firstLayerId);
            utils::insertTensorSlices(
                inputBuffers, "host_kv_cache_block_pointers_", kvCacheBlockPointersHost, firstLayerId);
        }
        else
        {
            utils::insertTensorVector(inputBuffers, "past_key_value_", presentKeysVals, firstLayerId);
            utils::insertTensorVector(outputBuffers, "present_key_value_", presentKeysVals, firstLayerId);
        }
    }
    else
    {
        inputBuffers.insert_or_assign("attention_mask", attentionMask);
        inputBuffers.insert_or_assign("cache_indirection", cacheIndirectionDecoderOutput);
        utils::insertTensorVector(
            outputBuffers, "present_key_value_", (step % 2) ? presentKeysValsAlt : presentKeysVals, firstLayerId);

        if (step == 0)
        {
            auto kvCacheShape = presentKeysValsAlt.at(0)->getShape();
            kvCacheShape.d[3] = 0;

            for (SizeType i = firstLayerId; i < firstLayerId + localNbLayers; ++i)
            {
                std::string name = "past_key_value_" + std::to_string(i);
                TensorPtr tmp = ITensor::view(presentKeysValsAlt[i], kvCacheShape);
                inputBuffers.insert_or_assign(name, std::move(tmp));
            }
        }
        else
        {
            utils::insertTensorVector(
                inputBuffers, "past_key_value_", (step % 2) ? presentKeysVals : presentKeysValsAlt, firstLayerId);
        }
    }

    if (modelConfig.useCustomAllReduce() && worldConfig.getTensorParallelism())
    {
        inputBuffers.insert_or_assign("all_reduce_workspace", commPtrs);
    }

    if (modelConfig.usePromptTuning())
    {
        inputBuffers.insert_or_assign("prompt_embedding_table", promptTuningParams.embeddingTable);
        inputBuffers.insert_or_assign("tasks", promptTuningParams.tasks);
        inputBuffers.insert_or_assign("prompt_vocab_size", promptTuningParams.vocabSize);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::vector<SizeType> RuntimeBuffers::getPositionIdsContextPhaseGlm(const SizeType& batchSize,
    const SizeType& maxInputLength, const SizeType* pInputLengths, bool useGptAttentionPlugin, bool usePackedInput)
{
    TLLM_CHECK(pInputLengths != nullptr);

    std::vector<SizeType> positionIdsVec(1, 0);
    if (useGptAttentionPlugin)
    {
        if (usePackedInput)
        {
            std::vector<int> pInputLengthsAcc = std::vector<int>(batchSize + 1, 0);
            for (int i = 0; i < batchSize; ++i)
            {
                pInputLengthsAcc[i + 1] = pInputLengthsAcc[i] + pInputLengths[i];
            }

            auto const size = 1 * 2 * pInputLengthsAcc[batchSize];
            positionIdsVec.resize(size, 0);
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + pInputLengthsAcc[b];
                auto const length = pInputLengths[b];
                std::iota(pIdB, pIdB + length, 0);

                pIdB[length - 1] = length - 2;
                pIdB[length - 1 + pInputLengthsAcc[batchSize]] = 1;
            }
        }
        else
        {
            auto const size = batchSize * 2 * maxInputLength;
            positionIdsVec.resize(size, 0);
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + b * 2 * maxInputLength;
                auto const length = pInputLengths[b];
                std::iota(pIdB, pIdB + length, 0);

                pIdB[length - 1] = length - 2;
                pIdB[length - 1 + maxInputLength] = 1;
            }
        }
    }
    else
    {
        TLLM_THROW("Unsupported model without GPT Attention Plugin");
    }

    return positionIdsVec;
}

std::vector<SizeType> RuntimeBuffers::getPositionIdsGenerationPhaseGlm(const SizeType& batchSize,
    const SizeType& beamSize, const SizeType& step, const SizeType* pInputLengths, bool useGptAttentionPlugin,
    bool usePackedInput)
{
    TLLM_CHECK(pInputLengths != nullptr);

    auto const size = 2 * batchSize * beamSize;
    std::vector<SizeType> positionIdsVec(size, 0);
    if (useGptAttentionPlugin)
    {
        if (usePackedInput)
        {
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + b * beamSize * 2;
                auto const length = pInputLengths[b * beamSize];

                for (SizeType bm = 0; bm < beamSize; ++bm)
                {
                    pIdB[bm * 2 + 0] = length - 2;
                    pIdB[bm * 2 + 1] = step + 2;
                }
            }
        }
        else
        {
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + b * beamSize * 2;
                auto const length = pInputLengths[b * beamSize];

                for (SizeType bm = 0; bm < beamSize; ++bm)
                {
                    pIdB[bm * 2 + 0] = length - 2;
                    pIdB[bm * 2 + 1] = step + 2;
                }
            }
        }
    }
    else
    {
        TLLM_THROW("Unsupported model without GPT Attention Plugin");
    }

    return positionIdsVec;
}
