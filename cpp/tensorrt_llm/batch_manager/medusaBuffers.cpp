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

#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/medusaModule.h"
#include "tensorrt_llm/runtime/utils/speculativeChoicesUtils.h"

namespace tensorrt_llm::batch_manager
{

MedusaBuffers::MedusaBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(maxBeamWidth == 1, "Medusa does not support beam search");

    auto const& engine = runtime.getEngine();

    auto const maxNumSequences = maxBatchSize;

    auto const medusaModule = std::dynamic_pointer_cast<tensorrt_llm::runtime::MedusaModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());

    auto const medusaHeads = medusaModule->getMaxDraftPathLen();
    auto const maxPathLen = medusaModule->getMaxPathLen();               // medusaHeads + 1
    auto const maxMedusaTokens = medusaModule->getMaxDecodingDraftTokens();
    auto const maxDecodingTokens = medusaModule->getMaxDecodingTokens(); // maxMedusaTokens + 1
    auto const numPackedMasks = medusaModule->getNumPackedMasks();

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto logitsType = engine.getTensorDataType("medusa_logits");
        medusaLogitsDevice = manager.gpu(
            ITensor::makeShape({medusaHeads, maxBatchSize, maxDecodingTokens, vocabSizePadded}), logitsType);
    }

    // Note: reserved for variable sequence length support.
    medusaGenerationLengthsHost
        = runtime::BufferManager::pinned(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    // TODO: pack batch and tokensPerStep into one dim to support variable sequence length without padddings.
    attentionPackedMaskHost = runtime::BufferManager::pinned(
        ITensor::makeShape({maxNumSequences, maxDecodingTokens, numPackedMasks}), nvinfer1::DataType::kINT32);
    medusaPositionOffsetsHost = runtime::BufferManager::pinned(
        ITensor::makeShape({maxNumSequences, maxDecodingTokens}), nvinfer1::DataType::kINT32);
    medusaTreeIdsHost = runtime::BufferManager::pinned(
        ITensor::makeShape({maxNumSequences, maxMedusaTokens}), nvinfer1::DataType::kINT32);
    medusaPathsHost = runtime::BufferManager::pinned(
        ITensor::makeShape({maxNumSequences, maxDecodingTokens, maxPathLen}), nvinfer1::DataType::kINT32);

    TensorPtr medusaPositionOffsetsHostSlice = ITensor::slice(medusaPositionOffsetsHost, 0, 1);
    medusaPositionOffsetsHostSlice->squeeze(0);
    TensorPtr medusaTreeIdsHostSlice = ITensor::slice(medusaTreeIdsHost, 0, 1);
    medusaTreeIdsHostSlice->squeeze(0);
    TensorPtr medusaPathsHostSlice = ITensor::slice(medusaPathsHost, 0, 1);
    medusaPathsHostSlice->squeeze(0);
    TensorPtr attentionPackedMaskHostSlice = ITensor::slice(attentionPackedMaskHost, 0, 1);
    attentionPackedMaskHostSlice->squeeze(0);

    // Init buffers for 1 request
    auto const& choices = decodingConfig.getMedusaChoices().value_or(medusaModule->getMedusaChoices());
    runtime::utils::initTensorsFromChoices(*medusaModule, choices, mTopKs, medusaGenerationLengthsHost,
        medusaPositionOffsetsHostSlice, medusaTreeIdsHostSlice, medusaPathsHostSlice, attentionPackedMaskHostSlice);

    auto scatterToBatch = [maxBatchSize, &manager](TensorPtr& data)
    {
        auto srcSlice = ITensor::slice(data, 0, 1);
        // Populate data from the 1st request to the other requests in the batch
        for (SizeType32 bi = 1; bi < maxBatchSize; ++bi)
        {
            auto dstSlice = ITensor::slice(data, bi, 1);
            manager.copy(*srcSlice, *dstSlice);
        }
    };

    scatterToBatch(medusaPositionOffsetsHost);
    scatterToBatch(medusaTreeIdsHost);
    scatterToBatch(medusaPathsHost);
    scatterToBatch(attentionPackedMaskHost);

    // Copy buffers to device
    // 1st dimension of packed mask is num_total_generation_tokens now (packed without paddings).
    attentionPackedMaskHost->reshape(ITensor::makeShape({maxNumSequences * maxDecodingTokens, numPackedMasks}));
    attentionPackedMaskDevice = manager.copyFrom(*attentionPackedMaskHost, runtime::MemoryType::kGPU);
    medusaGenerationLengthsDevice = manager.copyFrom(*medusaGenerationLengthsHost, runtime::MemoryType::kGPU);
    medusaPositionOffsetsDevice = manager.copyFrom(*medusaPositionOffsetsHost, runtime::MemoryType::kGPU);
    medusaTreeIdsDevice = manager.copyFrom(*medusaTreeIdsHost, runtime::MemoryType::kGPU);
    medusaPathsDevice = manager.copyFrom(*medusaPathsHost, runtime::MemoryType::kGPU);

    // use speculative decoding buffer
    medusaUseSpecDecoding = manager.cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    runtime::bufferCast<SizeType32>(*medusaUseSpecDecoding)[0] = 1;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void MedusaBuffers::reshape(SizeType32 /* numCtxSequences */, SizeType32 numGenSequences, SizeType32 tokensPerStep)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto attentionPackedMaskShape = attentionPackedMaskDevice->getShape();
    attentionPackedMaskShape.d[0] = numGenSequences * tokensPerStep;
    attentionPackedMaskDevice->reshape(attentionPackedMaskShape);

    auto medusaGenerationLengthsShape = medusaGenerationLengthsDevice->getShape();
    medusaGenerationLengthsShape.d[0] = numGenSequences;
    medusaGenerationLengthsDevice->reshape(medusaGenerationLengthsShape);

    auto medusaPositionOffsetsShape = medusaPositionOffsetsDevice->getShape();
    medusaPositionOffsetsShape.d[0] = numGenSequences;
    medusaPositionOffsetsDevice->reshape(medusaPositionOffsetsShape);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void MedusaBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.insert_or_assign("spec_decoding_packed_mask", attentionPackedMaskDevice);
    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", medusaGenerationLengthsDevice);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", medusaPositionOffsetsDevice);
    inputBuffers.insert_or_assign("spec_decoding_use", medusaUseSpecDecoding);
    if (worldConfig.isLastPipelineParallelRank())
    {
        outputBuffers.insert_or_assign("medusa_logits", medusaLogitsDevice);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
