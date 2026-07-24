/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"

namespace tensorrt_llm::runtime
{

LookaheadDecodingBuffers::LookaheadDecodingBuffers(
    SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, BufferManager const& bufferManager)
    : generationLengths(bufferManager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32))
    , positionOffsets(
          bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep}), nvinfer1::DataType::kINT32))
    , packedMasks(bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep,
                                        static_cast<ITensor::DimType64>(common::divUp(maxTokensPerStep, 32))}),
          nvinfer1::DataType::kINT32))
    , positionIds(
          bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep}), nvinfer1::DataType::kINT32))
{
}

LookaheadRuntimeBuffers::LookaheadRuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    BufferManager const& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    executor::DecodingConfig const& /* decodingConfig */, TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(maxBeamWidth == 1, "Lookahead decoding does not support beam search");

    auto const tokensPerStep = modelConfig.getMaxDecodingTokens();
    auto const numPackedMasks = static_cast<ITensor::DimType64>(tensorrt_llm::common::divUp(tokensPerStep, 32));

    cumSumLength = manager.pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    packedMasksDevice
        = manager.gpu(ITensor::makeShape({maxBatchSize * tokensPerStep, numPackedMasks}), nvinfer1::DataType::kINT32);
    positionOffsetsDevice = manager.gpu(ITensor::makeShape({maxBatchSize, tokensPerStep}), nvinfer1::DataType::kINT32);
    generationLengthsDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    positionIdsDevice = manager.gpu(ITensor::makeShape({maxBatchSize, tokensPerStep}), nvinfer1::DataType::kINT32);

    packedMaskHost = manager.cpu(packedMasksDevice->getShape(), nvinfer1::DataType::kINT32);
    positionOffsetsHost = manager.cpu(positionOffsetsDevice->getShape(), nvinfer1::DataType::kINT32);
    generationLengthsHost = manager.cpu(generationLengthsDevice->getShape(), nvinfer1::DataType::kINT32);
    positionIdsHost = manager.cpu(positionIdsDevice->getShape(), nvinfer1::DataType::kINT32);

    packedMaskHostCopy = manager.cpu(packedMasksDevice->getShape(), nvinfer1::DataType::kINT32);
    positionOffsetsHostCopy = manager.cpu(positionOffsetsDevice->getShape(), nvinfer1::DataType::kINT32);
    generationLengthsHostCopy = manager.cpu(generationLengthsDevice->getShape(), nvinfer1::DataType::kINT32);
    positionIdsHostCopy = manager.cpu(positionIdsDevice->getShape(), nvinfer1::DataType::kINT32);

    batchSlotsHostCopy = manager.cpu(generationLengthsDevice->getShape(), nvinfer1::DataType::kINT32);

    useSpecDecoding = manager.cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    bufferCast<SizeType32>(*useSpecDecoding)[0] = 1;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadRuntimeBuffers::setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences,
    ITensor const& requestTypes, ITensor const& seqSlots, LookaheadDecodingBuffers const& decoderLookaheadBuffers,
    TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();

    auto const tokensPerStep = modelConfig.getMaxDecodingTokens();

    manager.copy(seqSlots, *batchSlotsHostCopy);
    manager.copy(*decoderLookaheadBuffers.generationLengths, *generationLengthsHostCopy);
    manager.copy(*decoderLookaheadBuffers.positionOffsets, *positionOffsetsHostCopy);
    manager.copy(*decoderLookaheadBuffers.packedMasks, *packedMaskHostCopy);
    manager.copy(*decoderLookaheadBuffers.positionIds, *positionIdsHostCopy);

    manager.getStream().synchronize();

    BufferRange<SizeType32 const> batchSlotsRange(*batchSlotsHostCopy);
    BufferRange<SizeType32> cumSumLengthRange(*cumSumLength);

    SizeType32 maxGenerationLength = 0;
    for (SizeType32 bi = 0; bi < numGenSequences; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi + numCtxSequences];
        SizeType32 theLength = BufferRange<SizeType32>(*generationLengthsHostCopy)[gbi];
        maxGenerationLength = std::max(maxGenerationLength, theLength);
    }

    auto positionOffsetShape = positionOffsetsHost->getShape();
    positionOffsetShape.d[1] = maxGenerationLength;
    positionOffsetsHost->reshape(positionOffsetShape);
    positionOffsetsDevice->reshape(positionOffsetShape);

    auto positionIdsShape = positionIdsHostCopy->getShape();
    auto positionIdsShape1D = ITensor::makeShape({ITensor::volume(positionIdsShape)});
    positionIdsHostCopy->reshape(positionIdsShape1D);
    positionIdsHost->reshape(positionIdsShape1D);

    cumSumLengthRange[0] = 0;
    for (SizeType32 bi = 0; bi < numGenSequences; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi + numCtxSequences];
        SizeType32 theLength = BufferRange<SizeType32>(*generationLengthsHostCopy)[gbi];

        manager.copy(*ITensor::at(generationLengthsHostCopy, {gbi}), *ITensor::at(generationLengthsHost, {bi}));

        manager.copy(*ITensor::slice(positionOffsetsHostCopy, {gbi, 0}, theLength),
            *ITensor::slice(positionOffsetsHost, {bi, 0}, theLength));

        manager.copy(*ITensor::slice(packedMaskHostCopy, gbi * tokensPerStep, theLength),
            *ITensor::slice(packedMaskHost, cumSumLengthRange[0], theLength));

        manager.copy(*ITensor::slice(positionIdsHostCopy, gbi * tokensPerStep, theLength),
            *ITensor::slice(positionIdsHost, cumSumLengthRange[0], theLength));

        cumSumLengthRange[0] += theLength;
    }

    positionIdsHostCopy->reshape(positionIdsShape);
    positionIdsHost->reshape(positionIdsShape);
    positionIdsDevice->reshape(positionIdsShape);

    manager.copy(*ITensor::slice(generationLengthsHost, 0, numGenSequences),
        *ITensor::slice(generationLengthsDevice, 0, numGenSequences));
    manager.copy(*ITensor::slice(positionOffsetsHost, 0, numGenSequences),
        *ITensor::slice(positionOffsetsDevice, 0, numGenSequences));
    manager.copy(*ITensor::slice(packedMaskHost, 0, numGenSequences * tokensPerStep),
        *ITensor::slice(packedMasksDevice, 0, numGenSequences * tokensPerStep));
    manager.copy(
        *ITensor::slice(positionIdsHost, 0, numGenSequences), *ITensor::slice(positionIdsDevice, 0, numGenSequences));
    positionIdsDevice->reshape(ITensor::makeShape({cumSumLengthRange[0]}));

    manager.getStream().synchronize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadRuntimeBuffers::reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 tokensPerStep)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const numSequences = numGenSequences;

    auto packedMaskShape = packedMasksDevice->getShape();
    packedMaskShape.d[0] = numSequences * tokensPerStep;
    packedMasksDevice->reshape(packedMaskShape);
    packedMaskHost->reshape(packedMaskShape);

    auto generationLengthsShape = generationLengthsDevice->getShape();
    generationLengthsShape.d[0] = numSequences;
    generationLengthsDevice->reshape(generationLengthsShape);
    generationLengthsHost->reshape(generationLengthsShape);

    auto positionOffsetsShape = positionOffsetsDevice->getShape();
    positionOffsetsShape.d[0] = numSequences;
    positionOffsetsDevice->reshape(positionOffsetsShape);
    positionOffsetsHost->reshape(positionOffsetsShape);

    auto positionIdsShape = positionIdsDevice->getShape();
    positionIdsShape.d[0] = numSequences;
    positionIdsDevice->reshape(positionIdsShape);
    positionIdsHost->reshape(positionIdsShape);

    auto batchSlotsShape = batchSlotsHostCopy->getShape();
    batchSlotsShape.d[0] = numCtxSequences + numGenSequences;
    batchSlotsHostCopy->reshape(batchSlotsShape);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadRuntimeBuffers::enableLookaheadDecoding(SizeType32 maxBatchSize, SizeType32 tokensPerStep)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const numPackedMasks = static_cast<ITensor::DimType64>(tensorrt_llm::common::divUp(tokensPerStep, 32));
    packedMasksDevice->reshape(ITensor::makeShape({maxBatchSize * tokensPerStep, numPackedMasks}));
    generationLengthsDevice->reshape(ITensor::makeShape({maxBatchSize}));
    positionOffsetsDevice->reshape(ITensor::makeShape({maxBatchSize, tokensPerStep}));
    bufferCast<SizeType32>(*useSpecDecoding)[0] = 1;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadRuntimeBuffers::disableLookaheadDecoding()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    packedMasksDevice->reshape(ITensor::makeShape({1, 1}));
    generationLengthsDevice->reshape(ITensor::makeShape({1}));
    positionOffsetsDevice->reshape(ITensor::makeShape({1, 1}));
    bufferCast<SizeType32>(*useSpecDecoding)[0] = 0;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadRuntimeBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& /* outputBuffers */, WorldConfig const& /* worldConfig */) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    inputBuffers.insert_or_assign("spec_decoding_packed_mask", packedMasksDevice);
    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", generationLengthsDevice);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", positionOffsetsDevice);
    inputBuffers.insert_or_assign("spec_decoding_use", useSpecDecoding);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::runtime
