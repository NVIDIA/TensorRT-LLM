/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::runtime
{

LookaheadDecodingBuffers::LookaheadDecodingBuffers(
    SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, runtime::BufferManager const& bufferManager)
    : generationLengths(bufferManager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32))
    , positionOffsets(
          bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep}), nvinfer1::DataType::kINT32))
    , packedMasks(bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep,
                                        static_cast<ITensor::DimType64>(common::divUp(maxTokensPerStep, 32))}),
          nvinfer1::DataType::kINT32))
    , positionIds(
          bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep}), nvinfer1::DataType::kINT32))
{
    TLLM_LOG_DEBUG(
        "LookaheadDecodingBuffers, maxNumSequences = %d, maxTokensPerStep = %d", maxNumSequences, maxTokensPerStep);
}

LookaheadRuntimeBuffers::LookaheadRuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, executor::DecodingConfig const& /* decodingConfig */,
    runtime::TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(maxBeamWidth == 1, "Lookahead decoding does not support beam search");

    // auto const tokensPerStep = modelConfig.getMaxTokensPerStep();
    auto const tokensPerStep = modelConfig.getMaxDecodingTokens();
    auto const numPackedMasks = static_cast<ITensor::DimType64>(tensorrt_llm::common::divUp(tokensPerStep, 32));

    // Copy buffers to device
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

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadRuntimeBuffers::setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences,
    ITensor const& requestTypes, ITensor const& seqSlots, LookaheadDecodingBuffers const& decoderLookaheadBuffers,
    TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();

    auto const tokensPerStep = modelConfig.getMaxDecodingTokens();

    manager.copy(*decoderLookaheadBuffers.positionOffsets, *positionOffsetsHostCopy);
    manager.copy(*decoderLookaheadBuffers.packedMasks, *packedMaskHostCopy);
    manager.copy(*decoderLookaheadBuffers.positionIds, *positionIdsHostCopy);
    manager.copy(seqSlots, *batchSlotsHostCopy);
    manager.copy(*decoderLookaheadBuffers.generationLengths, *generationLengthsHostCopy);

    manager.getStream().synchronize();

    BufferRange<SizeType32 const> batchSlotsRange(*batchSlotsHostCopy);
    for (SizeType32 bi = 0; bi < numGenSequences; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi + numCtxSequences];
        manager.copy(*ITensor::at(generationLengthsHostCopy, {gbi}), *ITensor::at(generationLengthsHost, {bi}));
        manager.copy(*ITensor::at(positionOffsetsHostCopy, {gbi}), *ITensor::at(positionOffsetsHost, {bi}));
        manager.copy(*ITensor::slice(packedMaskHostCopy, gbi * tokensPerStep, tokensPerStep),
            *ITensor::slice(packedMaskHost, bi * tokensPerStep, tokensPerStep));
        manager.copy(*ITensor::at(positionIdsHostCopy, {gbi}), *ITensor::at(positionIdsHost, {bi}));
    }
    manager.copy(*ITensor::slice(generationLengthsHost, 0, numGenSequences),
        *ITensor::slice(generationLengthsDevice, 0, numGenSequences));
    manager.copy(*ITensor::slice(positionOffsetsHost, 0, numGenSequences),
        *ITensor::slice(positionOffsetsDevice, 0, numGenSequences));
    manager.copy(*ITensor::slice(packedMaskHost, 0, numGenSequences * tokensPerStep),
        *ITensor::slice(packedMasksDevice, 0, numGenSequences * tokensPerStep));
    manager.copy(
        *ITensor::slice(positionIdsHost, 0, numGenSequences), *ITensor::slice(positionIdsDevice, 0, numGenSequences));

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

void LookaheadRuntimeBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& /* outputBuffers */, runtime::WorldConfig const& /* worldConfig */) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.insert_or_assign("spec_decoding_packed_mask", packedMasksDevice);
    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", generationLengthsDevice);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", positionOffsetsDevice);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::runtime
