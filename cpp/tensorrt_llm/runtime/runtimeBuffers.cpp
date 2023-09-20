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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

using namespace tensorrt_llm::runtime;

RuntimeBuffers::GenerationConfig RuntimeBuffers::GenerationConfig::fromInput(ITensor::SharedPtr const& inputIds,
    ITensor::SharedPtr const& inputLengthsHost, bool const inputPacked, SizeType const beamWidth,
    SizeType const maxSequenceLength, std::optional<SizeType> const& maxNewTokensOpt, BufferManager& manager)
{
    auto const batchSize = static_cast<SizeType>(inputLengthsHost->getSize());

    auto const* inputLengthsPtr = bufferCast<SizeType>(*inputLengthsHost);
    auto const maxInputLength = *std::max_element(inputLengthsPtr, inputLengthsPtr + batchSize);

    if (inputPacked)
    {
        auto const inputLengthSum = std::reduce(inputLengthsPtr, inputLengthsPtr + batchSize);
        TLLM_CHECK_WITH_INFO(inputIds->getShape().d[0] == 1 && inputIds->getShape().d[1] == inputLengthSum,
            "Packed input must have shape [1, <sum of input lengths>].");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(inputIds->getShape().d[0] == batchSize && inputIds->getShape().d[1] == maxInputLength,
            "Padded input must have shape [batch size, max input length]");
    }

    auto const maxNewTokens = maxNewTokensOpt.value_or(maxSequenceLength - maxInputLength);
    TLLM_CHECK_WITH_INFO(1 <= maxNewTokens && maxNewTokens <= maxSequenceLength - maxInputLength,
        "Max input length is equal to or larger that maxSequenceLength given in setup. No new tokens can be "
        "generated.");

    return GenerationConfig{batchSize, beamWidth, maxInputLength, maxNewTokens, maxSequenceLength};
}

void RuntimeBuffers::clear()
{
    logits = nullptr;
    sequenceLengths = nullptr;
    pastKeyValueLengths = nullptr;
    attentionMask = nullptr;
    positionIds = nullptr;
    lastTokenIds = nullptr;

    presentKeysVals.clear();
    presentKeysValsAlt.clear();

    contextLengthsHost = nullptr;
    requestTypes = nullptr;

    allocated = false;
}

void RuntimeBuffers::create(TllmRuntime& runtime, GptModelConfig const& modelConfig)
{
    auto& manager = runtime.getBufferManager();

    auto const logitsType = utils::getTensorDataType(runtime.getEngine(), "logits");
    logits = manager.emptyTensor(MemoryType::kGPU, logitsType);

    contextLengthsHost = manager.emptyTensor(MemoryType::kPINNED, nvinfer1::DataType::kINT32);
    inputOffsets = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    presentKeysVals
        = utils::createBufferVector(runtime, modelConfig.getNbLayers(), "present_key_value_", MemoryType::kGPU);

    if (modelConfig.useGptAttentionPlugin())
    {
        sequenceLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    }
    else
    {
        presentKeysValsAlt
            = utils::createBufferVector(runtime, modelConfig.getNbLayers(), "present_key_value_", MemoryType::kGPU);
    }

    if (modelConfig.usePagedKvCache())
    {
        kvCacheBlockPointers = utils::createBufferVector(
            runtime, modelConfig.getNbLayers(), "kv_cache_block_pointers_", MemoryType::kGPU);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        requestTypes = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    }

    cacheIndirectionDecoderInput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    cacheIndirectionDecoderOutput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
}

void RuntimeBuffers::reshape(
    GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, SizeType worldSize)
{
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxSeqLength = generationConfig.maxSeqLength;

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldSize);
    // logits are tiled to {batchSize, beamWidth, vocabSizePadded} after context step of engine
    logits->reshape(ITensor::makeShape({batchSize, 1, vocabSizePadded}));

    auto kvCacheShape
        = ITensor::makeShape({batchSize, 2, modelConfig.getNbKvHeads(), maxSeqLength, modelConfig.getSizePerHead()});
    if (modelConfig.usePagedKvCache())
    {
        auto const tokensPerBlock = modelConfig.getTokensPerBlock();
        auto const maxBlocksPerSeq = (maxSeqLength + tokensPerBlock - 1) / tokensPerBlock;

        // reserve batchSize * beamWidth and resize to batchSize
        auto cacheBlockPointersShape = ITensor::makeShape({batchSize * beamWidth, 2, maxBlocksPerSeq * 2});
        utils::reshapeBufferVector(kvCacheBlockPointers, cacheBlockPointersShape);
        cacheBlockPointersShape.d[0] = batchSize;
        utils::reshapeBufferVector(kvCacheBlockPointers, cacheBlockPointersShape);
    }
    else
    {
        utils::reshapeBufferVector(presentKeysVals, kvCacheShape);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        sequenceLengths->reshape(ITensor::makeShape({batchSize}));
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

    allocated = true;
}

void RuntimeBuffers::tile(
    BufferManager& manager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig)
{
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(beamWidth > 1, "Tiling is only necessary for beam search.");

    // logits needs beamWidth in second dimension
    auto logitsShape = logits->getShape();
    logitsShape.d[1] *= beamWidth;
    utils::tileBufferReplace(logits, beamWidth, manager);
    logits->reshape(logitsShape);

    utils::tileBufferReplace(contextLengthsDevice, beamWidth, manager);

    if (modelConfig.useGptAttentionPlugin())
    {
        utils::tileBufferReplace(sequenceLengths, beamWidth, manager);
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
}

void RuntimeBuffers::postContextStep(
    BufferManager& manager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig)
{
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxSeqLength = generationConfig.maxSeqLength;

    if (modelConfig.useGptAttentionPlugin())
    {
        requestTypes->reshape(ITensor::makeShape({batchSize * beamWidth}));
        auto hostRequestTypes = bufferCast<int32_t>(*requestTypes);
        std::fill_n(hostRequestTypes, requestTypes->getSize(), 1);
    }

    if (beamWidth > 1)
    {
        tile(manager, generationConfig, modelConfig);
    }

    // no need to copy data in lastTokenIds because it is overwritten in prepareNextStep
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth}));

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto const& pointersShape = kvCacheBlockPointers[0]->getShape();
        auto const maxBlocksPerSeq = pointersShape.d[pointersShape.nbDims - 1] / 2;
        auto cacheBlockPointersShape = ITensor::makeShape({batchSize * beamWidth, 2, maxBlocksPerSeq * 2});
        utils::reshapeBufferVector(kvCacheBlockPointers, cacheBlockPointersShape);
    }
}

void RuntimeBuffers::prepareContextStep(TensorPtr const& inputIds, TokenIdType const padId, BufferManager& manager,
    KvCacheManager& kvCacheManager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig)
{
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const beamWidth = generationConfig.beamWidth;
    SizeType const maxInputLength = generationConfig.maxInputLength;
    SizeType const maxSeqLength = generationConfig.maxSeqLength;

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

        if (modelConfig.usePackedInput())
        {
            auto const inputOffsetsHost = manager.copyFrom(*inputOffsets, MemoryType::kCPU);
            auto const* inputOffsetsPtr = bufferCast<SizeType>(*inputOffsetsHost);

            std::vector<SizeType> positionIdsVec(inputIds->getShape().d[1]);
            for (SizeType i = 0; i < batchSize; ++i)
                std::iota(std::begin(positionIdsVec) + inputOffsetsPtr[i],
                    std::begin(positionIdsVec) + inputOffsetsPtr[i + 1], 0);
            positionIds = manager.copyFrom(positionIdsVec, inputIds->getShape(), MemoryType::kGPU);
        }
        else
        {
            std::vector<SizeType> positionIdsVec(inputIds->getSize());
            for (SizeType i = 0; i < batchSize; ++i)
                std::iota(std::begin(positionIdsVec) + i * maxInputLength,
                    std::begin(positionIdsVec) + (i + 1) * maxInputLength, 0);
            positionIds = manager.copyFrom(positionIdsVec, inputIds->getShape(), MemoryType::kGPU);
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
            std::exclusive_scan(attentionMaskData + i * maxInputLength, attentionMaskData + (i + 1) * maxInputLength,
                std::begin(positionIdsVec) + i * maxInputLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskData[i] == 0)
                positionIdsVec[i] = 1;
        positionIds = manager.copyFrom(positionIdsVec, attentionMask->getShape(), MemoryType::kGPU);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        manager.copy(*contextLengthsDevice, *sequenceLengths);
    }

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto constexpr contextBeamWidth = 1;
        auto const& pointersShape = kvCacheBlockPointers[0]->getShape();
        auto const maxBlocksPerSeq = pointersShape.d[pointersShape.nbDims - 1] / 2;
        auto const& blockPointersBatch
            = kvCacheManager.getBlockPointersOfBatch(batchSize, contextBeamWidth, maxBlocksPerSeq);
        for (auto layer = 0; layer < modelConfig.getNbLayers(); ++layer)
        {
            TLLM_CHECK(blockPointersBatch[layer]->getSizeInBytes() == kvCacheBlockPointers[layer]->getSizeInBytes());
            auto pointersPtr = bufferCast<int64_t>(*blockPointersBatch[layer]);
            auto pointersPtr32 = reinterpret_cast<int32_t*>(pointersPtr);
            manager.copy(pointersPtr32, *kvCacheBlockPointers[layer]);
        }
    }

    if (modelConfig.usePackedInput())
    {
        lastTokenIds = manager.copyFrom(*ITensor::slice(inputOffsets, 1), MemoryType::kGPU);
    }
    else
    {
        lastTokenIds = manager.copyFrom(*contextLengthsDevice, MemoryType::kGPU);
    }

    manager.setZero(*cacheIndirectionDecoderInput);
    manager.setZero(*cacheIndirectionDecoderOutput);
};

RuntimeBuffers::TensorPtr RuntimeBuffers::prepareNextStep(SizeType const step, TensorPtr const& outputIds,
    BufferManager& manager, KvCacheManager& kvCacheManager, GenerationConfig const& generationConfig,
    GptModelConfig const& modelConfig)
{
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const beamWidth = generationConfig.beamWidth;
    SizeType const maxSeqLength = generationConfig.maxSeqLength;

    nvinfer1::Dims nextInputIdsShape;
    if (modelConfig.usePackedInput())
    {
        // squeeze first dim and batch in last dim
        nextInputIdsShape = ITensor::makeShape({1, batchSize * beamWidth});
    }
    else
    {
        // squeeze first dim
        nextInputIdsShape = ITensor::makeShape({batchSize * beamWidth, 1});
    }

    auto nextInputIds = ITensor::view(outputIds, nextInputIdsShape);

    if (modelConfig.useGptAttentionPlugin())
    {
        auto const contextLengthsHostPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const pastKeyValueLengthsPtr = bufferCast<SizeType>(*pastKeyValueLengths);
        SizeType const tensorBatchSize = pastKeyValueLengths->getSize();
        SizeType const srcStride = (modelConfig.useGptAttentionPlugin() ? 1 : beamWidth);
        TLLM_CHECK(static_cast<std::size_t>(tensorBatchSize * srcStride) == contextLengthsDevice->getSize());
        for (SizeType i = 0; i < tensorBatchSize; ++i)
        {
            pastKeyValueLengthsPtr[i] = contextLengthsHostPtr[i * srcStride] + step;
        }

        // The sequence_lengths = context_lengths + step for generation stage.
        kernels::invokeAdd(*sequenceLengths, 1, stream);

        positionIds->reshape(contextLengthsDevice->getShape());
        manager.copy(*contextLengthsDevice, *positionIds);
        kernels::invokeAdd(*positionIds, step, stream);

        auto const size = static_cast<SizeType>(positionIds->getSize());
        if (modelConfig.usePackedInput())
            positionIds->reshape(ITensor::makeShape({1, size}));
        else
            positionIds->reshape(ITensor::makeShape({size, 1}));
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
            std::exclusive_scan(attentionMaskPtr + i * newLength, attentionMaskPtr + (i + 1) * newLength,
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
            kvCacheManager.addToken(batchIdx);
        }
        auto const& pointersShape = kvCacheBlockPointers[0]->getShape();
        auto const maxBlocksPerSeq = pointersShape.d[pointersShape.nbDims - 1] / 2;
        auto const& blockPointersBatch = kvCacheManager.getBlockPointersOfBatch(batchSize, beamWidth, maxBlocksPerSeq);
        for (auto layer = 0; layer < modelConfig.getNbLayers(); ++layer)
        {
            TLLM_CHECK(blockPointersBatch[layer]->getSizeInBytes() == kvCacheBlockPointers[layer]->getSizeInBytes());
            auto pointersPtr = bufferCast<int64_t>(*blockPointersBatch[layer]);
            auto pointersPtr32 = reinterpret_cast<int32_t*>(pointersPtr);
            manager.copy(pointersPtr32, *kvCacheBlockPointers[layer]);
        }
    }

    kernels::invokeFill(*lastTokenIds, 1, stream);
    if (modelConfig.usePackedInput())
    {
        kernels::invokeInclusiveSum(*lastTokenIds, *lastTokenIds, manager, stream);
    }

    return nextInputIds;
};

void RuntimeBuffers::getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType const step,
    TensorPtr const& inputIds, KvCacheManager& kvCacheManager, GptModelConfig const& modelConfig) const
{
    inputBuffers.clear();
    outputBuffers.clear();

    outputBuffers.insert_or_assign("logits", ITensor::view(logits)); // feed a view to TensorRT runtime

    inputBuffers.insert_or_assign("input_ids", inputIds);

    inputBuffers.insert_or_assign("context_lengths", contextLengthsDevice);
    inputBuffers.insert_or_assign("last_token_ids", lastTokenIds);
    inputBuffers.insert_or_assign("position_ids", positionIds);

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
            utils::insertTensorVector(inputBuffers, "past_key_value_", kvCacheManager.getMemoryPools());
            utils::insertTensorVector(outputBuffers, "present_key_value_", kvCacheManager.getMemoryPools());
            utils::insertTensorVector(inputBuffers, "kv_cache_block_pointers_", kvCacheBlockPointers);
        }
        else
        {
            utils::insertTensorVector(inputBuffers, "past_key_value_", presentKeysVals);
            utils::insertTensorVector(outputBuffers, "present_key_value_", presentKeysVals);
        }
    }
    else
    {
        inputBuffers.insert_or_assign("attention_mask", attentionMask);
        inputBuffers.insert_or_assign("cache_indirection", cacheIndirectionDecoderOutput);
        utils::insertTensorVector(
            outputBuffers, "present_key_value_", (step % 2) ? presentKeysValsAlt : presentKeysVals);

        if (step == 0)
        {
            auto kvCacheShape = presentKeysValsAlt.at(0)->getShape();
            kvCacheShape.d[3] = 0;

            for (SizeType i = 0; i < modelConfig.getNbLayers(); ++i)
            {
                std::string name = "past_key_value_" + std::to_string(i);
                TensorPtr tmp = ITensor::view(presentKeysValsAlt[i], kvCacheShape);
                inputBuffers.insert_or_assign(name, std::move(tmp));
            }
        }
        else
        {
            utils::insertTensorVector(
                inputBuffers, "past_key_value_", (step % 2) ? presentKeysVals : presentKeysValsAlt);
        }
    }
}
