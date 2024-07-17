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

#include "tensorrt_llm/runtime/transformerBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"
#include <cstdlib> // std::getenv

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

TransformerBuffers::TransformerBuffers()
{
    pastKeyValueLengths = nullptr;
    attentionMask = nullptr;
    positionIds = nullptr;

    presentKeysVals.clear();
    presentKeysValsAlt.clear();
    kvCacheBlockPoolPointers = nullptr;
    kvCacheBlockOffsetsHost = nullptr;
    kvCacheBlockOffsetsDevice = nullptr;
}

TransformerBuffers::TransformerBuffers(
    TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(modelConfig.isTransformerBased());
    auto& manager = runtime.getBufferManager();
    auto& engine = runtime.getEngine();

    auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());
    auto firstAttentionLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    auto const& layerTypes = modelConfig.getLayerTypes();
    if (!layerTypes.empty())
    {
        firstAttentionLayerId
            = std::find(layerTypes.begin(), layerTypes.end(), ModelConfig::LayerType::kATTENTION) - layerTypes.begin();
    }

    nvinfer1::DataType kvDtype;
    if (modelConfig.usePagedKvCache())
    {
        kvDtype = modelConfig.getKvDataType();
    }
    else
    {
        kvDtype = modelConfig.getQuantMode().hasFp8KvCache()
            ? nvinfer1::DataType::kFP8
            : engine.getTensorDataType(("present_key_value_" + std::to_string(firstAttentionLayerId)).c_str());
    }

    if (modelConfig.usePagedKvCache())
    {
        auto const kvCacheBlockOffsetsType = engine.getTensorDataType("kv_cache_block_offsets");
        kvCacheBlockOffsetsHost = manager.emptyTensor(MemoryType::kCPU, kvCacheBlockOffsetsType);
        kvCacheBlockOffsetsDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockOffsetsType);
    }
    else
    {
        presentKeysVals = utils::createBufferVector(runtime, localNbLayers, MemoryType::kGPU, kvDtype);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        maxAttentionWindows = BufferManager::cpu(ITensor::makeShape({localNbLayers}), nvinfer1::DataType::kINT32);
        sinkTokenLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    }
    else
    {
        constexpr int32_t extraKeyValBufferNum = 1;
        presentKeysValsAlt = utils::createBufferVector(runtime, extraKeyValBufferNum, MemoryType::kGPU, kvDtype);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::reshape(
    GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = generationConfig.batchSize;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxAttentionWindow = generationConfig.maxAttentionWindow;

    auto const kvCacheReserve = ITensor::makeShape(
        {batchSize, 2, modelConfig.getNbKvHeads(), maxAttentionWindow, modelConfig.getSizePerHead()});
    auto const kvCacheShape
        = ITensor::makeShape({batchSize, 2, modelConfig.getNbKvHeads(), maxInputLength, modelConfig.getSizePerHead()});
    if (modelConfig.usePagedKvCache())
    {
        auto cacheBlockOffsetsShape = kvCacheBlockOffsetsHost->getShape();
        if (cacheBlockOffsetsShape.nbDims > 0)
        {
            cacheBlockOffsetsShape.d[0] = batchSize;
            kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
            kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
        }
        else
        {
            TLLM_LOG_DEBUG("kvCacheBlockOffsets not allocated yet");
        }
    }
    else
    {
        utils::reshapeBufferVector(presentKeysVals, kvCacheReserve);
    }

    auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());

    if (modelConfig.useGptAttentionPlugin())
    {
        pastKeyValueLengths->reshape(ITensor::makeShape({batchSize}));
        maxAttentionWindows->reshape(ITensor::makeShape({localNbLayers}));
        sinkTokenLengths->reshape(ITensor::makeShape({1}));
    }
    else
    {
        utils::reshapeBufferVector(presentKeysValsAlt, kvCacheShape);
        // present KV cache tensors will be reshaped by shape inference.
        // reshape to the required shape here to make context batch slicing work correctly.
        utils::reshapeBufferVector(presentKeysVals, kvCacheShape);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::reshapeKvTensors(
    SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq, runtime::TllmRuntime const& runtime)
{
    auto const& manager = runtime.getBufferManager();

    auto const cacheBlockOffsetsShape = ITensor::makeShape({maxBatchSize * maxBeamWidth, 2, maxBlocksPerSeq});

    kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
    manager.setZero(*kvCacheBlockOffsetsHost);

    kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
    manager.setZero(*kvCacheBlockOffsetsDevice);
}

void TransformerBuffers::setKvPoolPointers(KvCacheManager const* kvCacheManager)
{
    kvCacheBlockPoolPointers = kvCacheManager->getBlockPoolPointers();
}

TransformerBuffers TransformerBuffers::sliceTo(
    GenerationConfig const& generationConfig, ModelConfig const& modelConfig, SizeType32 offset, SizeType32 batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TransformerBuffers buffers;
    auto const generationBatchSize = generationConfig.batchSize;
    if (modelConfig.usePagedKvCache())
    {
        auto const& realCacheBlockOffsetsShape = kvCacheBlockOffsetsHost->getShape();
        auto const maxBlocksPerSeq = realCacheBlockOffsetsShape.d[2];

        // enable slicing by moving generationBatchSize to first dim
        auto const fakeCacheBlockOffsetsShape = ITensor::makeShape({generationBatchSize, 2, maxBlocksPerSeq});
        TensorPtr kvCacheBlockOffsetsHostView{ITensor::view(kvCacheBlockOffsetsHost, fakeCacheBlockOffsetsShape)};
        TensorPtr kvCacheBlockOffsetsDeviceView{ITensor::view(kvCacheBlockOffsetsDevice, fakeCacheBlockOffsetsShape)};

        // slice and reshape to correct shape
        auto const cacheBlockOffsetsShape = ITensor::makeShape({batchSize, 2, maxBlocksPerSeq});
        buffers.kvCacheBlockOffsetsHost = ITensor::slice(kvCacheBlockOffsetsHostView, offset, batchSize);
        buffers.kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
        buffers.kvCacheBlockOffsetsDevice = ITensor::slice(kvCacheBlockOffsetsDeviceView, offset, batchSize);
        buffers.kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);

        buffers.kvCacheBlockPoolPointers = kvCacheBlockPoolPointers;
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
    }
    else
    {
        buffers.presentKeysValsAlt = utils::sliceBufferVector(presentKeysValsAlt, offset, batchSize);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return buffers;
}

static std::vector<SizeType32> getPositionIdsContextPhaseGlm(SizeType32 const& batchSize,
    SizeType32 const& maxInputLength, SizeType32 const* pInputLengths, bool useGptAttentionPlugin, bool usePackedInput)
{
    TLLM_CHECK(pInputLengths != nullptr);

    std::vector<SizeType32> positionIdsVec(1, 0);
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
            for (SizeType32 b = 0; b < batchSize; ++b)
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
            for (SizeType32 b = 0; b < batchSize; ++b)
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

void TransformerBuffers::prepareContextStep(RuntimeBuffers* runtimeBuffers, TensorPtr const& inputIds,
    TokenIdType const padId, BufferManager& manager, KvCacheManager const* kvCacheManager, SizeType32 firstBatchSlotIdx,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
    auto& promptTuningTasksHost = runtimeBuffers->promptTuningTasksHost;
    auto& promptTuningParams = runtimeBuffers->promptTuningParams;
    auto& stream = manager.getStream();
    SizeType32 const batchSize = generationConfig.batchSize;
    SizeType32 const maxInputLength = generationConfig.maxInputLength;
    auto const& inputShape = inputIds->getShape();

    // get local number of layers.
    auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());

    if (modelConfig.useGptAttentionPlugin())
    {
        auto pastKeyValueLengthsPtr = bufferCast<SizeType32>(*pastKeyValueLengths);
        TLLM_CHECK(pastKeyValueLengths->getSize() == static_cast<std::size_t>(batchSize));

        auto RequestTypesPtr = bufferCast<int32_t>(*requestTypes);
        TLLM_CHECK(requestTypes->getSize() == static_cast<std::size_t>(batchSize));
        std::fill_n(RequestTypesPtr, batchSize, 0);

        auto maxAttentionWindowsPtr = bufferCast<SizeType32>(*maxAttentionWindows);
        std::fill_n(maxAttentionWindowsPtr, localNbLayers, generationConfig.maxAttentionWindow);

        bufferCast<SizeType32>(*sinkTokenLengths)[0] = generationConfig.sinkTokenLength;

        auto const contextLengthsHostPtr = bufferCast<SizeType32 const>(*contextLengthsHost);
        auto const modelVariant = modelConfig.getModelVariant();

        if (modelVariant == ModelConfig::ModelVariant::kGpt
            || modelVariant == ModelConfig::ModelVariant::kRecurrentGemma)
        {
            auto const inputSize = inputIds->getSize();
            std::vector<SizeType32> positionIdsVec(inputSize);
            auto begin = std::begin(positionIdsVec);
            for (SizeType32 i = 0; i < batchSize; ++i)
            {
                auto end = begin + (modelConfig.usePackedInput() ? contextLengthsHostPtr[i] : maxInputLength);
                std::iota(begin, end, 0);
                begin = end;
            }
            positionIds = manager.copyFrom(positionIdsVec, inputShape, MemoryType::kGPU);
        }
        else if (modelVariant == ModelConfig::ModelVariant::kGlm)
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

        for (SizeType32 i = 0; i < batchSize; ++i)
        {
            pastKeyValueLengthsPtr[i] = contextLengthsHostPtr[i];
        }

        if (modelConfig.usePromptTuning())
        {
            std::vector<SizeType32> reqBeamWidths(batchSize, 1);
            std::vector<SizeType32> reqPromptLengths;
            for (SizeType32 i = 0; i < batchSize; ++i)
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
        auto const* attentionMaskData = reinterpret_cast<SizeType32 const*>(attentionMaskHost->data());
        std::vector<SizeType32> positionIdsVec(attentionMask->getSize());
        for (SizeType32 i = 0; i < batchSize; ++i)
        {
            tc::stl_utils::exclusiveScan(attentionMaskData + i * maxInputLength,
                attentionMaskData + (i + 1) * maxInputLength, std::begin(positionIdsVec) + i * maxInputLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskData[i] == 0)
                positionIdsVec[i] = 1;
        positionIds = manager.copyFrom(positionIdsVec, attentionMask->getShape(), MemoryType::kGPU);
    }

    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = hiddenStates->getShape().d[hiddenStates->getShape().nbDims - 1];
        auto const hiddenStatesShape = modelConfig.usePackedInput()
            ? ITensor::makeShape({inputShape.d[0], hiddenSize})
            : ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto constexpr contextBeamWidth = 1;
        kvCacheManager->getBlockOffsetsOfBatch(
            *kvCacheBlockOffsetsHost, firstBatchSlotIdx, batchSize, contextBeamWidth);
        manager.copy(*kvCacheBlockOffsetsHost, *kvCacheBlockOffsetsDevice);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

static std::vector<SizeType32> getPositionIdsGenerationPhaseGlm(SizeType32 const& batchSize, SizeType32 const& beamSize,
    SizeType32 const& step, SizeType32 const* pInputLengths, bool useGptAttentionPlugin, bool usePackedInput)
{
    TLLM_CHECK(pInputLengths != nullptr);

    auto const size = 2 * batchSize * beamSize;
    std::vector<SizeType32> positionIdsVec(size, 0);
    if (useGptAttentionPlugin)
    {
        // Share the same layout regardless of usePackedInput or not
        for (SizeType32 b = 0; b < batchSize; ++b)
        {
            auto* pIdB = positionIdsVec.data() + b * beamSize * 2;
            auto const length = pInputLengths[b * beamSize];

            for (SizeType32 bm = 0; bm < beamSize; ++bm)
            {
                pIdB[bm * 2 + 0] = length - 2;
                pIdB[bm * 2 + 1] = step + 2;
            }
        }
    }
    else
    {
        TLLM_THROW("Unsupported model without GPT Attention Plugin");
    }

    return positionIdsVec;
}

void TransformerBuffers::copyAttentionMasks(
    RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBatches, BufferManager& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto const batchSize = generationConfig.batchSize;
    auto const maxInputLength = generationConfig.maxInputLength;

    // TODO(rkobus) include tiling
    attentionMask = manager.gpu(ITensor::makeShape({batchSize, maxInputLength}), nvinfer1::DataType::kINT32);

    auto const numContextBatches = static_cast<SizeType32>(contextBatches.size());
    auto offset = 0;
    for (auto contextBatchId = 0; contextBatchId < numContextBatches; ++contextBatchId)
    {
        auto& buffers = contextBatches.at(contextBatchId);
        auto contextBatchSize = buffers.generationConfig.batchSize;
        auto attentionMaskSlice = ITensor::slice(attentionMask, offset, contextBatchSize);
        manager.copy(*buffers.transformerBuffers->attentionMask, *attentionMaskSlice);
        offset += contextBatchSize;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& logits = runtimeBuffers->logits;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
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
        utils::tileCpuBufferReplace(contextLengthsHost, beamWidth);
        utils::tileCpuBufferReplace(pastKeyValueLengths, beamWidth);
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
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::postContextStep(RuntimeBuffers* runtimeBuffers,
    std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& requestTypes = runtimeBuffers->requestTypes;
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
        copyAttentionMasks(runtimeBuffers, contextBuffers, manager);
    }

    // TODO(rkobus) handle this more gracefully
    positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (modelConfig.computeContextLogits())
    {
        runtimeBuffers->gatherLastTokenLogits(manager, modelConfig, worldConfig);
    }

    if (beamWidth > 1)
    {
        tile(runtimeBuffers, manager, modelConfig, worldConfig);
    }

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto cacheBlockOffsetsShape = kvCacheBlockOffsetsHost->getShape();
        cacheBlockOffsetsShape.d[0] = batchSize * beamWidth;
        kvCacheBlockOffsetsHost->reshape(cacheBlockOffsetsShape);
        kvCacheBlockOffsetsDevice->reshape(cacheBlockOffsetsShape);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::prepareNextStep(RuntimeBuffers* runtimeBuffers, SizeType32 const step, BufferManager& manager,
    KvCacheManager* kvCacheManager, SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& stream = manager.getStream();
    SizeType32 const batchSize = generationConfig.batchSize;
    SizeType32 const beamWidth = generationConfig.beamWidth;
    auto const inputShape = [&modelConfig, batchSize, beamWidth]()
    {
        if (modelConfig.usePackedInput())
        {
            // batch in last dim
            return ITensor::makeShape({batchSize * beamWidth});
        }
        else
        {
            // batch in first dim
            return ITensor::makeShape({batchSize * beamWidth, 1});
        }
    }();
    if (modelConfig.useGptAttentionPlugin())
    {
        auto const contextLengthsHostPtr = bufferCast<SizeType32 const>(*contextLengthsHost);
        auto const pastKeyValueLengthsPtr = bufferCast<SizeType32>(*pastKeyValueLengths);
        auto const tensorBatchSize = static_cast<SizeType32>(pastKeyValueLengths->getSize());
        SizeType32 const srcStride{modelConfig.useGptAttentionPlugin() ? 1 : beamWidth};
        TLLM_CHECK(static_cast<std::size_t>(tensorBatchSize * srcStride) == contextLengthsDevice->getSize());
        for (SizeType32 i = 0; i < tensorBatchSize; ++i)
        {
            pastKeyValueLengthsPtr[i] = contextLengthsHostPtr[i * srcStride] + step;
        }

        auto const modelVariant = modelConfig.getModelVariant();

        if (modelVariant == ModelConfig::ModelVariant::kGpt
            || modelVariant == ModelConfig::ModelVariant::kRecurrentGemma)
        {
            positionIds->reshape(inputShape);
            manager.copy(*contextLengthsDevice, *positionIds);
            kernels::invokeAdd(*positionIds, step, stream);
        }
        else if (modelVariant == ModelConfig::ModelVariant::kGlm)
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
        auto const* attentionMaskPtr = bufferCast<SizeType32>(*attentionMaskHost);

        // TODO old positionIds could be recovered to avoid scan
        std::vector<SizeType32> positionIdsVec(attentionMask->getSize());
        for (SizeType32 i = 0; i < nbInputs; ++i)
        {
            tc::stl_utils::exclusiveScan(attentionMaskPtr + i * newLength, attentionMaskPtr + (i + 1) * newLength,
                std::begin(positionIdsVec) + i * newLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskPtr[i] == 0)
                positionIdsVec[i] = 1;
        std::vector<SizeType32> positionIdsEndVec(nbInputs);
        for (SizeType32 i = 0; i < nbInputs; ++i)
            positionIdsEndVec[i] = positionIdsVec[(i + 1) * newLength - 1];

        positionIds = manager.copyFrom(positionIdsEndVec, ITensor::makeShape({nbInputs, 1}), MemoryType::kGPU);
    }

    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = hiddenStates->getShape().d[hiddenStates->getShape().nbDims - 1];
        auto const hiddenStatesShape = modelConfig.usePackedInput()
            ? ITensor::makeShape({inputShape.d[0], hiddenSize})
            : ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    if (modelConfig.usePagedKvCache())
    {
        for (auto batchIdx = firstBatchSlotIdx; batchIdx < firstBatchSlotIdx + batchSize; ++batchIdx)
        {
            kvCacheManager->addToken(batchIdx);
        }
        kvCacheManager->getBlockOffsetsOfBatch(*kvCacheBlockOffsetsHost, firstBatchSlotIdx, batchSize, beamWidth);
        manager.copy(*kvCacheBlockOffsetsHost, *kvCacheBlockOffsetsDevice);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers,
    TensorMap& outputBuffers, SizeType32 const step, TensorPtr const& inputIds, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.clear();
    outputBuffers.clear();

    auto& logits = runtimeBuffers->logits;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto& lastTokenIds = runtimeBuffers->lastTokenIds;
    auto& requestTypes = runtimeBuffers->requestTypes;

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

    auto const localNbLayers = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;
    auto const& layerTypes = modelConfig.getLayerTypes();

    if (modelConfig.useGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("cache_indirection", runtimeBuffers->cacheIndirectionDecoderOutput);
        inputBuffers.insert_or_assign("host_past_key_value_lengths", pastKeyValueLengths);
        inputBuffers.insert_or_assign("host_request_types", requestTypes);
        inputBuffers.insert_or_assign("sequence_length", runtimeBuffers->sequenceLengths);
        inputBuffers.insert_or_assign("host_sink_token_length", sinkTokenLengths);
        inputBuffers.insert_or_assign("host_max_attention_window_sizes", maxAttentionWindows);

        if (modelConfig.usePackedInput())
        {
            inputBuffers.insert_or_assign("host_context_lengths", contextLengthsHost);
        }
        if (modelConfig.usePagedKvCache())
        {
            inputBuffers.insert_or_assign("kv_cache_block_offsets", kvCacheBlockOffsetsDevice);
            inputBuffers.insert_or_assign("host_kv_cache_block_offsets", kvCacheBlockOffsetsHost);
            inputBuffers.insert_or_assign("host_kv_cache_pool_pointers", kvCacheBlockPoolPointers);
        }
        else
        {
            utils::insertTensorVector(inputBuffers, "past_key_value_", presentKeysVals, firstLayerId, layerTypes,
                ModelConfig::LayerType::kATTENTION);
            utils::insertTensorVector(outputBuffers, "present_key_value_", presentKeysVals, firstLayerId, layerTypes,
                ModelConfig::LayerType::kATTENTION);
        }
    }
    else
    {
        inputBuffers.insert_or_assign("attention_mask", attentionMask);
        inputBuffers.insert_or_assign("cache_indirection", runtimeBuffers->cacheIndirectionDecoderOutput);

        nvinfer1::Dims kvCacheShape{0};
        if (step == 0)
        {
            kvCacheShape = presentKeysValsAlt.at(0)->getShape();
            kvCacheShape.d[3] = 0;
        }

        // TODO: fix for recurrentgemma
        for (int32_t idx = 0; idx < localNbLayers; ++idx)
        {
            TensorPtr input;
            TensorPtr output;
            // We will make current layer's output KV-cache overwrite previous layers input KV-cache
            // buffer id: ...  5,  6,  7,  8,  9, ...
            // layer n:        out in
            // layer n+1:          out in
            // layer n+2               out in
            // And when finish a step, we will make every layer's in/out buffer index subtract 1 in
            // a circular buffer way to make sure current outputs become next step's inputs.
            int32_t input_ind = idx - (step % (localNbLayers + 1)); // Subtract 1 for every step.
            if (input_ind < 0)
            {
                // When underflow, go to the back to achieve a circular buffers.
                input_ind = localNbLayers + 1 + input_ind;
            }
            // Output buffer is just before input buffer. When input is buffer 0,
            // output should use the back buffer to achieve circular buffers.
            int32_t output_ind = input_ind > 0 ? input_ind - 1 : localNbLayers;

            // We only allocate localNbLayers of normal buffers. If index is overflow, use the extra buffer.
            input = input_ind < localNbLayers ? presentKeysVals[input_ind] : presentKeysValsAlt[0];
            output = output_ind < localNbLayers ? presentKeysVals[output_ind] : presentKeysValsAlt[0];

            if (step == 0)
            {
                TensorPtr tmp = ITensor::view(input, kvCacheShape);
                inputBuffers.insert_or_assign("past_key_value_" + std::to_string(firstLayerId + idx), std::move(tmp));
            }
            else
            {
                inputBuffers.insert_or_assign("past_key_value_" + std::to_string(firstLayerId + idx), input);
            }
            outputBuffers.insert_or_assign("present_key_value_" + std::to_string(firstLayerId + idx), output);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
