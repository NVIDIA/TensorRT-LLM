//
// Created by martinma on 5/24/23.
//
/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <algorithm>
#include <iostream>

#include "ipcUtils.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

RuntimeBuffers::GenerationConfig RuntimeBuffers::GenerationConfig::fromInput(ITensor const& inputIds,
    ITensor const& inputLengthsHost, bool const inputPacked, SizeType const beamWidth, SizeType const maxSequenceLength,
    std::optional<SizeType> const& maxNewTokensOpt)
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
        TLLM_CHECK_WITH_INFO(inputShape.d[0] == 1 && inputShape.d[1] == inputLengthSum,
            "Packed input must have shape [1, <sum of input lengths>].");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(inputShape.d[0] == batchSize && inputShape.d[1] >= maxInputLength,
            "Padded input must have shape [batch size, max input length]");
        maxInputLength = inputShape.d[1];
    }

    auto const maxNewTokens = maxNewTokensOpt.value_or(maxSequenceLength - maxInputLength);
    TLLM_CHECK_WITH_INFO(1 <= maxNewTokens && maxNewTokens <= maxSequenceLength - maxInputLength,
        "Max input length is equal to or larger that maxSequenceLength given in setup. No new tokens can be "
        "generated.");

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return GenerationConfig{batchSize, beamWidth, maxInputLength, maxNewTokens, maxSequenceLength, inputLengthSum};
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

    hiddenStates = nullptr;

    allocated = false;
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
    }

    contextLengthsHost = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT32);
    sequenceLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
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

void RuntimeBuffers::initContextLengths(TensorPtr const& inputLengths, BufferManager& manager)
{
    contextLengthsDevice = inputLengths;
    contextLengthsHost->reshape(inputLengths->getShape());
    manager.copy(*contextLengthsDevice, *contextLengthsHost);
    manager.getStream().synchronize(); // wait for context lengths to be copied to host
}

void RuntimeBuffers::createCustomAllReduceWorkspace(SizeType maxBatchSize, SizeType maxBeamWidth,
    SizeType maxSequenceLength, SizeType hiddenSize, WorldConfig const& worldConfig, BufferManager& manager)
{
    mIpcMemoryHandles.clear();
    const std::size_t bufferSize = static_cast<std::size_t>(maxBatchSize) * maxBeamWidth * maxSequenceLength
        * hiddenSize * worldConfig.getTensorParallelism() * sizeof(float);
    mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(worldConfig, bufferSize));
    mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(worldConfig, IpcMemory::FLAGS_SIZE * sizeof(int32_t)));
    mIpcMemoryHandles.emplace_back(std::make_shared<IpcMemory>(worldConfig, IpcMemory::FLAGS_SIZE * sizeof(int32_t)));

    commPtrs = manager.cpu(
        ITensor::makeShape({static_cast<SizeType>(mIpcMemoryHandles.size()) * worldConfig.getTensorParallelism()}),
        nvinfer1::DataType::kINT64);
    const auto commPtrsData = bufferCast<void*>(*commPtrs);

    for (size_t memIdx = 0; memIdx < mIpcMemoryHandles.size(); memIdx++)
    {
        const auto& memCommPtrs = mIpcMemoryHandles[memIdx]->getCommPtrsTensor();
        for (SizeType tpIdx = 0; tpIdx < worldConfig.getTensorParallelism(); tpIdx++)
        {
            commPtrsData[memIdx * worldConfig.getTensorParallelism() + tpIdx] = memCommPtrs[tpIdx];
        }
    }
}

void RuntimeBuffers::reshape(
    GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxSeqLength = generationConfig.maxSeqLength;

    if (worldConfig.isLastPipelineParallelRank() && !modelConfig.computeContextLogits())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        logits->reshape(ITensor::makeShape({batchSize, 1, vocabSizePadded}));
    }

    sequenceLengths->reshape(ITensor::makeShape({batchSize}));
    lastTokenIds->reshape(ITensor::makeShape({batchSize}));

    auto kvCacheShape
        = ITensor::makeShape({batchSize, 2, modelConfig.getNbKvHeads(), maxSeqLength, modelConfig.getSizePerHead()});
    if (modelConfig.usePagedKvCache())
    {
        auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
        auto const tokensPerBlock = modelConfig.getTokensPerBlock();
        auto const maxBlocksPerSeq = (maxSeqLength + tokensPerBlock - 1) / tokensPerBlock;

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
        utils::reshapeBufferVector(presentKeysVals, kvCacheShape);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        pastKeyValueLengths->reshape(ITensor::makeShape({batchSize}));
        requestTypes->reshape(ITensor::makeShape({batchSize}));
    }
    else
    {
        utils::reshapeBufferVector(presentKeysValsAlt, kvCacheShape);
    }

    auto const cacheIndirShape = ITensor::makeShape({batchSize, beamWidth, maxSeqLength});
    cacheIndirectionDecoderInput->reshape(cacheIndirShape);
    cacheIndirectionDecoderOutput->reshape(cacheIndirShape);

    if (worldConfig.isPipelineParallel())
    {
        // reserve max size
        auto const maxNumTokens = std::max(batchSize * beamWidth, batchSize * maxInputLength);
        auto const hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();
        auto const hiddenStatesShape = ITensor::makeShape({1, maxNumTokens, hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    allocated = true;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::gatherLastTokenLogits(BufferManager& manager, GenerationConfig const& generationConfig,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(modelConfig.computeContextLogits(),
        "Gather last token logits is only necessary when context logits are computed");

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const batchSize = generationConfig.batchSize;
        auto const beamWidth = generationConfig.beamWidth;
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

        auto const tiledTensorShape = ITensor::makeShape({batchSize, beamWidth, vocabSizePadded});
        auto tiledTensor = std::shared_ptr(manager.gpu(tiledTensorShape, logits->getDataType()));
        kernels::gatherLastTokenLogits(*tiledTensor, *logits, *lastTokenIds, manager.getStream());
        manager.getStream().synchronize();

        std::swap(logits, tiledTensor);
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::tile(BufferManager& manager, GenerationConfig const& generationConfig,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
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
    utils::tileBufferReplace(sequenceLengths, beamWidth, manager);

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

void RuntimeBuffers::postContextStep(BufferManager& manager, GenerationConfig const& generationConfig,
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

    if (modelConfig.computeContextLogits())
    {
        gatherLastTokenLogits(manager, generationConfig, modelConfig, worldConfig);
    }

    if (beamWidth > 1)
    {
        tile(manager, generationConfig, modelConfig, worldConfig);
    }

    // no need to copy data in lastTokenIds because it is overwritten in prepareNextStep
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth}));

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto cacheBlockPointersShape = kvCacheBlockPointersHost->getShape();
        cacheBlockPointersShape.d[1] = batchSize * beamWidth;
        kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareContextStep(TensorPtr const& inputIds, TokenIdType const padId, BufferManager& manager,
    KvCacheManager const* kvCacheManager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const maxInputLength = generationConfig.maxInputLength;

    manager.copy(*contextLengthsDevice, *sequenceLengths);

    if (modelConfig.useGptAttentionPlugin())
    {
        auto pastKeyValueLengthsPtr = bufferCast<SizeType>(*pastKeyValueLengths);
        TLLM_CHECK(pastKeyValueLengths->getSize() == static_cast<std::size_t>(batchSize));
        std::fill_n(pastKeyValueLengthsPtr, batchSize, 0);
        if (modelConfig.useGptAttentionPlugin())
        {
            auto RequestTypesPtr = bufferCast<int32_t>(*requestTypes);
            TLLM_CHECK(requestTypes->getSize() == static_cast<std::size_t>(batchSize));
            std::fill_n(RequestTypesPtr, batchSize, 0);
        }

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
            auto const positionIdsVec = getPositionIdsContextPhaseGlm(
                batchSize, maxInputLength, contextLengthsHostPtr, modelConfig.useGptAttentionPlugin());
            auto const positionIdsShape = ITensor::makeShape({batchSize, 2, maxInputLength});
            positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
        }
        else
        {
            TLLM_THROW("Unsupported model variant");
        }

        if (worldConfig.isPipelineParallel())
        {
            auto const hiddenSize = hiddenStates->getShape().d[2];
            auto const hiddenStatesShape = ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
            hiddenStates->reshape(hiddenStatesShape);
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
        kvCacheManager->getBlockPointersOfBatch(kvCacheBlockPointersHost, batchSize, contextBeamWidth);
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

    manager.setZero(*cacheIndirectionDecoderInput);
    manager.setZero(*cacheIndirectionDecoderOutput);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

RuntimeBuffers::TensorPtr RuntimeBuffers::prepareNextStep(SizeType const step, TensorPtr const& outputIds,
    BufferManager& manager, KvCacheManager* kvCacheManager, GenerationConfig const& generationConfig,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const beamWidth = generationConfig.beamWidth;

    nvinfer1::Dims inputShape;
    if (modelConfig.usePackedInput())
    {
        // batch in last dim
        inputShape = ITensor::makeShape({1, batchSize * beamWidth});
    }
    else
    {
        // batch in first dim
        inputShape = ITensor::makeShape({batchSize * beamWidth, 1});
    }
    auto nextInputIds = outputIds ? ITensor::view(outputIds, inputShape) : TensorPtr{};

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
            auto const positionIdsVec = getPositionIdsGenerationPhaseGlm(
                batchSize, beamWidth, step, contextLengthsHostPtr, modelConfig.useGptAttentionPlugin());
            auto const positionIdsShape = ITensor::makeShape({batchSize * beamWidth, 2, 1});
            positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
        }
        else
        {
            TLLM_THROW("Unsupported model variant");
        }

        if (worldConfig.isPipelineParallel())
        {
            auto const hiddenSize = hiddenStates->getShape().d[2];
            auto const hiddenStatesShape = ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
            hiddenStates->reshape(hiddenStatesShape);
        }
    }
    else
    {
        auto const shape = attentionMask->getShape();
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
        for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            kvCacheManager->addToken(batchIdx);
        }
        kvCacheManager->getBlockPointersOfBatch(kvCacheBlockPointersHost, batchSize, beamWidth);
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
    TensorPtr const& inputIds, GptModelConfig const& modelConfig, WorldConfig const& worldConfig) const
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

        if (modelConfig.usePackedInput())
        {
            inputBuffers.insert_or_assign("host_context_lengths", contextLengthsHost);
        }
        if (modelConfig.usePagedKvCache())
        {
            utils::insertTensorSlices(
                inputBuffers, "kv_cache_block_pointers_", kvCacheBlockPointersDevice, firstLayerId);
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
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

std::vector<SizeType> RuntimeBuffers::getPositionIdsContextPhaseGlm(
    SizeType batchSize, SizeType maxInputLength, SizeType const* pInputLengths, bool useGptAttentionPlugin)
{
    TLLM_CHECK(pInputLengths != nullptr);
    TLLM_CHECK_WITH_INFO(useGptAttentionPlugin, "GptAttentionPlugin must be used");

    auto const size = batchSize * 2 * maxInputLength;
    std::vector<SizeType> positionIdsVec(size, 0);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        auto* pIdB = positionIdsVec.data() + b * 2 * maxInputLength;
        auto const length = pInputLengths[b];
        std::iota(pIdB, pIdB + length, 0);

        pIdB[length - 1] = length - 2;
        pIdB[length - 1 + maxInputLength] = 1;
    }

    return positionIdsVec;
}

std::vector<SizeType> RuntimeBuffers::getPositionIdsGenerationPhaseGlm(
    SizeType batchSize, SizeType beamSize, SizeType step, SizeType const* pInputLengths, bool useGptAttentionPlugin)
{
    TLLM_CHECK(pInputLengths != nullptr);
    TLLM_CHECK_WITH_INFO(useGptAttentionPlugin, "GptAttentionPlugin must be used");

    auto const size = batchSize * beamSize * 2;
    std::vector<SizeType> positionIdsVec(size, 0);

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

    return positionIdsVec;
}
