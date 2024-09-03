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

#include "tensorrt_llm/runtime/rnnStateBuffers.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

RnnStateBuffers::RnnStateBuffers()
{
    rnnStates = nullptr;
    convStates = nullptr;
    convStatesAlt = nullptr;
    slotMappingHost = nullptr;
    slotMappingDevice = nullptr;
    rnnStatePtrs = nullptr;
    convStatePtrs = nullptr;
}

RnnStateBuffers::RnnStateBuffers(
    TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(modelConfig.isRnnBased());
    TLLM_CHECK_WITH_INFO(modelConfig.hasRnnConfig(), "RNN only support Mamba1/Mamba2/RecurrentGemma now.");
    auto maxBatchSize = modelConfig.getMaxBatchSize();
    auto maxBeamWidth = modelConfig.getMaxBeamWidth();
    auto maxBatchBeam = maxBatchSize * maxBeamWidth;
    auto rnnConfig = modelConfig.getRnnConfig();
    TLLM_CHECK_WITH_INFO(rnnConfig.has_value(), "RnnStateBuffers should be used with rnnConfig.");
    mConvKernel = rnnConfig->convKernel;
    mStateSize = rnnConfig->stateSize;
    mRnnHiddenSize = rnnConfig->rnnHiddenSize;
    mRnnHeadSize = rnnConfig->rnnHeadSize;
    mRnnConvDimSize = rnnConfig->rnnConvDimSize;
    auto dType = modelConfig.getDataType();
    auto const localNbLayers = modelConfig.getNbRnnLayers(worldConfig.getPipelineParallelism());
    mLocalNbLayers = localNbLayers;
    mMaxBeamWidth = maxBeamWidth;
    mUseMambaConv1dPlugin = modelConfig.useMambaConv1dPlugin();
    auto const rnnStatesShape = [&]()
    {
        if (mRnnHeadSize > 0)
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mRnnHiddenSize / mRnnHeadSize, mStateSize, mRnnHeadSize});
        }
        else
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mStateSize, mRnnHiddenSize});
        }
    }();
    auto const convStatesShape = [&]()
    {
        if (mUseMambaConv1dPlugin)
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mConvKernel - 1, mRnnConvDimSize});
        }
        else
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mRnnConvDimSize, mConvKernel - 1});
        }
    }();
    auto& bufferManager = runtime.getBufferManager();
    auto const isRecurrentGemma = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kRecurrentGemma;
    auto stateDType = isRecurrentGemma ? nvinfer1::DataType::kFLOAT : dType;
    rnnStates = bufferManager.gpu(rnnStatesShape, stateDType);
    convStates = bufferManager.gpu(convStatesShape, dType);
    convStatesAlt = bufferManager.gpu(convStatesShape, dType);

    if (modelConfig.usePagedState())
    {
        auto slotMappingShape = ITensor::makeShape({maxBatchSize});
        auto statePtrsShape = ITensor::makeShape({localNbLayers});
        slotMappingDevice = bufferManager.gpu(slotMappingShape, nvinfer1::DataType::kINT32);
        slotMappingHost = BufferManager::cpu(slotMappingShape, nvinfer1::DataType::kINT32);
        rnnStatePtrs = BufferManager::cpu(statePtrsShape, nvinfer1::DataType::kINT64);
        convStatePtrs = BufferManager::cpu(statePtrsShape, nvinfer1::DataType::kINT64);
    }
    else
    {
        slotMappingHost = nullptr;
        slotMappingDevice = nullptr;
        rnnStatePtrs = nullptr;
        convStatePtrs = nullptr;
    }

    reshape(maxBatchSize);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::reshape(SizeType32 batchSize)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const rnnStatesShape = [&]()
    {
        if (mRnnHeadSize > 0)
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mRnnHiddenSize / mRnnHeadSize, mStateSize, mRnnHeadSize});
        }
        else
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mStateSize, mRnnHiddenSize});
        }
    }();
    auto const convStatesShape = [&]()
    {
        if (mUseMambaConv1dPlugin)
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mConvKernel - 1, mRnnConvDimSize});
        }
        else
        {
            return tensorrt_llm::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mRnnConvDimSize, mConvKernel - 1});
        }
    }();
    rnnStates->reshape(rnnStatesShape);
    convStates->reshape(convStatesShape);
    convStatesAlt->reshape(convStatesShape);

    rnnState.resize(mLocalNbLayers);
    convState.resize(mLocalNbLayers);
    convStateAlt.resize(mLocalNbLayers);
    for (int i = 0; i < mLocalNbLayers; i++)
    {
        size_t offset = batchSize * mMaxBeamWidth * i;
        rnnState[i] = tensorrt_llm::runtime::ITensor::slice(rnnStates, offset, batchSize * mMaxBeamWidth);
        convState[i] = tensorrt_llm::runtime::ITensor::slice(convStates, offset, batchSize * mMaxBeamWidth);
        convStateAlt[i] = tensorrt_llm::runtime::ITensor::slice(convStatesAlt, offset, batchSize * mMaxBeamWidth);
    }
    if (slotMappingDevice != nullptr)
    {
        TLLM_CHECK(slotMappingHost != nullptr);
        TLLM_CHECK(rnnStates != nullptr && convStates != nullptr);
        TLLM_CHECK(rnnStatePtrs != nullptr && convStatePtrs != nullptr);

        auto slotMappingShape = ITensor::makeShape({batchSize});
        slotMappingDevice->reshape(slotMappingShape);
        slotMappingHost->reshape(slotMappingShape);

        int* slotMapping = static_cast<int*>(slotMappingHost->data());
        for (int b = 0; b < batchSize; b++)
        {
            slotMapping[b] = b;
        }
        fillStatePtrs();
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::fillStatePtrs()
{
    auto statePtrsShape = ITensor::makeShape({mLocalNbLayers});
    rnnStatePtrs->reshape(statePtrsShape);
    convStatePtrs->reshape(statePtrsShape);

    rnnStatePtr.resize(mLocalNbLayers);
    convStatePtr.resize(mLocalNbLayers);

    void** rnnStatePtrArray = static_cast<void**>(rnnStatePtrs->data());
    void** convStatePtrArray = static_cast<void**>(convStatePtrs->data());

    for (int i = 0; i < mLocalNbLayers; i++)
    {
        rnnStatePtrArray[i] = rnnState[i]->data();
        convStatePtrArray[i] = convState[i]->data();
        rnnStatePtr[i] = tensorrt_llm::runtime::ITensor::slice(rnnStatePtrs, i, 1);
        convStatePtr[i] = tensorrt_llm::runtime::ITensor::slice(convStatePtrs, i, 1);
    }
}

void RnnStateBuffers::reshape(
    GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    auto const batchSize = generationConfig.batchSize;

    reshape(batchSize);
}

void RnnStateBuffers::reset(BufferManager& manager)
{
    // This is not need in Plugin path, but may be needed for OOTB path.
    manager.setZero(*rnnStates);
    manager.setZero(*convStates);
    manager.setZero(*convStatesAlt);
}

RnnStateBuffers RnnStateBuffers::sliceTo(SizeType32 offset, SizeType32 size)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    RnnStateBuffers buffers;
    buffers.rnnState = utils::sliceBufferVector(rnnState, offset, size);
    buffers.convState = utils::sliceBufferVector(convState, offset, size);
    buffers.convStateAlt = utils::sliceBufferVector(convStateAlt, offset, size);

    if (slotMappingDevice != nullptr)
    {
        TLLM_CHECK(slotMappingHost != nullptr);
        TLLM_CHECK(rnnStates != nullptr && convStates != nullptr);
        TLLM_CHECK(rnnStatePtrs != nullptr && convStatePtrs != nullptr);
        buffers.slotMappingHost = ITensor::slice(slotMappingHost, offset, size);
        buffers.slotMappingDevice = ITensor::slice(slotMappingHost, offset, size);
        int* slotMapping = static_cast<int*>(buffers.slotMappingHost->data());
        for (int b = 0; b < size; b++)
        {
            slotMapping[b] = b;
        }
        buffers.fillStatePtrs();
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return buffers;
}

void RnnStateBuffers::prepareContextStep(RuntimeBuffers* runtimeBuffers, BufferManager& manager)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType32 const batchSize = runtimeBuffers->generationConfig.batchSize;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto RequestTypesPtr = bufferCast<int32_t>(*requestTypes);
    TLLM_CHECK(requestTypes->getSize() == static_cast<std::size_t>(batchSize));
    std::fill_n(RequestTypesPtr, batchSize, 0);

    manager.setZero(*convStates);
    if (slotMappingDevice != nullptr)
    {
        manager.copy(*slotMappingHost, *slotMappingDevice);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(false, "Beam search for mamba is not supported now.");
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
    utils::tileCpuBufferReplace(contextLengthsHost, beamWidth);

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
    BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& outputLengths = runtimeBuffers->outputLengths;
    auto& lastTokenIds = runtimeBuffers->lastTokenIds;
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    requestTypes->reshape(ITensor::makeShape({batchSize * beamWidth}));
    auto hostRequestTypes = bufferCast<int32_t>(*requestTypes);
    std::fill_n(hostRequestTypes, requestTypes->getSize(), 1);

    if (modelConfig.computeContextLogits())
    {
        runtimeBuffers->gatherLastTokenLogits(manager, modelConfig, worldConfig);
    }

    if (beamWidth > 1)
    {
        tile(runtimeBuffers, manager, modelConfig, worldConfig);
    }

    // use output lengths after context step
    manager.copy(*contextLengthsDevice, *outputLengths);
    // no need to copy data in lastTokenIds because it is overwritten in prepareNextStep
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth}));
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers,
    TensorMap& outputBuffers, SizeType32 const step, TensorPtr const& inputIds, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& logits = runtimeBuffers->logits;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
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

    inputBuffers.insert_or_assign("last_token_ids", lastTokenIds);

    auto const localNbLayers = modelConfig.getNbRnnLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;
    auto const& layerTypes = modelConfig.getLayerTypes();

    if (modelConfig.usePagedState())
    {
        inputBuffers.insert_or_assign("slot_mapping", slotMappingDevice);
        utils::insertTensorVector(inputBuffers, "conv_state_ptr_", convStatePtr, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(
            inputBuffers, "rnn_state_ptr_", rnnStatePtr, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
    }
    else
    {
        utils::insertTensorVector(inputBuffers, "past_conv_state_", (step % 2) ? convState : convStateAlt, firstLayerId,
            layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(outputBuffers, "present_conv_state_", (step % 2) ? convStateAlt : convState,
            firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(
            inputBuffers, "past_rnn_state_", rnnState, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(outputBuffers, "present_rnn_state_", rnnState, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
    }

    inputBuffers.insert_or_assign("host_request_types", requestTypes);
    inputBuffers.insert_or_assign("host_context_lengths", runtimeBuffers->contextLengthsHost);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}
