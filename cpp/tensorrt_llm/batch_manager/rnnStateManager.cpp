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

#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

RnnStateManager::RnnStateManager(SizeType32 maxNumSequences, tensorrt_llm::runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, tensorrt_llm::runtime::BufferManager const& bufferManager)
    : mMaxNumSequences(maxNumSequences)
    , mMaxBeamWidth{modelConfig.getMaxBeamWidth()}
{
    TLLM_CHECK_WITH_INFO(modelConfig.usePagedState(), "RnnStateManager should be used with Paged State enabled.");
    TLLM_CHECK_WITH_INFO(modelConfig.useMambaConv1dPlugin(), "RnnStateManager should be used with MambaConv1dPlugin.");
    TLLM_CHECK_WITH_INFO(mMaxBeamWidth == 1, "Beam search is not supported for Mamba now.");
    mBeamSlotsPerSequence = mMaxBeamWidth == 1 ? mMaxBeamWidth : mMaxBeamWidth + 1;
    // If we need support beam search, we may need mMaxBeamWidth + 1 slots and use separate input / output states.
    auto const& rnnConfig = modelConfig.getRnnConfig();
    TLLM_CHECK_WITH_INFO(rnnConfig.has_value(), "RnnStateManager should be used with rnnConfig");
    auto const convKernel = rnnConfig->convKernel;
    auto const stateSize = rnnConfig->stateSize;
    auto const rnnHiddenSize = rnnConfig->rnnHiddenSize;
    auto const rnnHeadSize = rnnConfig->rnnHeadSize;
    auto const rnnConvDimSize = rnnConfig->rnnConvDimSize;
    auto const localNbLayers
        = modelConfig.getNbRnnLayers(worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    auto const dataType = modelConfig.getDataType();

    auto const rnnStateShape = [&]()
    {
        if (rnnHeadSize > 0)
        {
            return tensorrt_llm::runtime::ITensor::makeShape({localNbLayers, mMaxNumSequences * mBeamSlotsPerSequence,
                rnnHiddenSize / rnnHeadSize, stateSize, rnnHeadSize});
        }
        else
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {localNbLayers, mMaxNumSequences * mBeamSlotsPerSequence, stateSize, rnnHiddenSize});
        }
    }();
    auto const convStateShape = tensorrt_llm::runtime::ITensor::makeShape(
        {localNbLayers, mMaxNumSequences * mBeamSlotsPerSequence, convKernel - 1, rnnConvDimSize});
    pagedRnnStates = bufferManager.gpu(rnnStateShape, nvinfer1::DataType::kFLOAT);
    pagedConvStates = bufferManager.gpu(convStateShape, dataType);

    auto const statePtrsShape = tensorrt_llm::runtime::ITensor::makeShape({localNbLayers});
    rnnStatePtrs = tensorrt_llm::runtime::BufferManager::cpu(statePtrsShape, TRTDataType<void*>::value);
    convStatePtrs = tensorrt_llm::runtime::BufferManager::cpu(statePtrsShape, TRTDataType<void*>::value);
    auto* rnnStatePtrArray = bufferCast<void*>(*rnnStatePtrs);
    auto* convStatePtrArray = bufferCast<void*>(*convStatePtrs);

    rnnStatePtr.resize(localNbLayers);
    convStatePtr.resize(localNbLayers);
    for (int i = 0; i < localNbLayers; i++)
    {
        auto layerRnnStates = tensorrt_llm::runtime::ITensor::slice(pagedRnnStates, i, 1);
        auto layerConvStates = tensorrt_llm::runtime::ITensor::slice(pagedConvStates, i, 1);
        rnnStatePtrArray[i] = layerRnnStates->data();
        convStatePtrArray[i] = layerConvStates->data();
        rnnStatePtr[i] = tensorrt_llm::runtime::ITensor::slice(rnnStatePtrs, i, 1);
        convStatePtr[i] = tensorrt_llm::runtime::ITensor::slice(convStatePtrs, i, 1);
    }
}

void RnnStateManager::getPtrBuffers(
    TensorMap& inputBuffers, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const
{
    auto const firstLayerId
        = modelConfig.getFirstLocalLayer(worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    auto const& layerTypes = modelConfig.getLayerTypes();

    utils::insertTensorVector(
        inputBuffers, "conv_state_ptr_", convStatePtr, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
    utils::insertTensorVector(
        inputBuffers, "rnn_state_ptr_", rnnStatePtr, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
}

void RnnStateManager::fillSlotMapping(
    runtime::ITensor& dstPointers, SizeType32 dstSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const
{
    TLLM_CHECK(seqSlotIdx < mMaxNumSequences);
    TLLM_CHECK(beamWidth <= mMaxBeamWidth);

    auto* dstPtr = bufferCast<SizeType32>(dstPointers);
    if (beamWidth == 1)
    {
        dstPtr[dstSlotOffset] = seqSlotIdx * mBeamSlotsPerSequence;
    }
    else
    {
        // leave first for context.
        std::iota(dstPtr + dstSlotOffset, dstPtr + dstSlotOffset + beamWidth, seqSlotIdx * mBeamSlotsPerSequence + 1);
    }
}

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
