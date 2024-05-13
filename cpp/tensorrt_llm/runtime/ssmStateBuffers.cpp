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

#include "tensorrt_llm/runtime/ssmStateBuffers.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

SsmStateBuffers::SsmStateBuffers()
{
    mambaSsmStates = nullptr;
    mambaConvStates = nullptr;
    mambaConvStatesAlt = nullptr;
    slotMappingHost = nullptr;
    slotMappingDevice = nullptr;
    mambaSsmStatePtrs = nullptr;
    mambaConvStatePtrs = nullptr;
}

SsmStateBuffers::SsmStateBuffers(
    TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(modelConfig.isSsmBased());
    // TODO: support RecurrentGemma: the code mostly works but returns incorrect tokens in the generation phase
    TLLM_CHECK_WITH_INFO(modelConfig.hasMambaConfig(), "SSM only support Mamba for now.");
    auto maxBatchSize = modelConfig.getMaxBatchSize();
    auto maxBeamWidth = modelConfig.getMaxBeamWidth();
    auto maxBatchBeam = maxBatchSize * maxBeamWidth;
    auto rnnConfig = modelConfig.getRnnConfig();
    mIsRecurrentGemma = rnnConfig.has_value();

    if (mIsRecurrentGemma)
    {
        mDConv = rnnConfig->dConv;
        mDInner = rnnConfig->hiddenSize;
    }
    else
    {
        auto mambaConfig = modelConfig.getMambaConfig();
        mDConv = mambaConfig->dConv;
        mDState = mambaConfig->dState;
        auto expand = mambaConfig->expand;
        auto hiddenSize = modelConfig.getHiddenSize();
        mDInner = expand * hiddenSize;
    }

    auto dType = modelConfig.getDataType();
    auto const localNbLayers = modelConfig.getNbSsmLayers(worldConfig.getPipelineParallelism());
    mLocalNbLayers = localNbLayers;
    mMaxBeamWidth = maxBeamWidth;
    mUseMambaConv1dPlugin = modelConfig.useMambaConv1dPlugin();
    auto const ssmStatesShape = [&]()
    {
        if (mIsRecurrentGemma)
        {
            return ITensor::makeShape({localNbLayers * maxBatchBeam, mDInner});
        }
        else
        {
            return ITensor::makeShape({localNbLayers * maxBatchBeam, mDState, mDInner});
        }
    }();
    auto const convStatesShape = [&]()
    {
        if (mUseMambaConv1dPlugin)
        {
            return ITensor::makeShape({localNbLayers * maxBatchBeam, mDConv - 1, mDInner});
        }
        else
        {
            return ITensor::makeShape({localNbLayers * maxBatchBeam, mDInner, mDConv - 1});
        }
    }();
    auto& bufferManager = runtime.getBufferManager();
    mambaSsmStates = bufferManager.gpu(ssmStatesShape, dType);
    mambaConvStates = bufferManager.gpu(convStatesShape, dType);
    mambaConvStatesAlt = bufferManager.gpu(convStatesShape, dType);

    if (modelConfig.usePagedState())
    {
        auto slotMappingShape = ITensor::makeShape({maxBatchSize});
        auto statePtrsShape = ITensor::makeShape({localNbLayers});
        slotMappingDevice = bufferManager.gpu(slotMappingShape, nvinfer1::DataType::kINT32);
        slotMappingHost = BufferManager::cpu(slotMappingShape, nvinfer1::DataType::kINT32);
        mambaSsmStatePtrs = BufferManager::cpu(statePtrsShape, nvinfer1::DataType::kINT64);
        mambaConvStatePtrs = BufferManager::cpu(statePtrsShape, nvinfer1::DataType::kINT64);
    }
    else
    {
        slotMappingHost = nullptr;
        slotMappingDevice = nullptr;
        mambaSsmStatePtrs = nullptr;
        mambaConvStatePtrs = nullptr;
    }

    reshape(maxBatchSize);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void SsmStateBuffers::reshape(SizeType batchSize)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const ssmStatesShape = [&]()
    {
        if (mIsRecurrentGemma)
        {
            return ITensor::makeShape({mLocalNbLayers * batchSize * mMaxBeamWidth, mDInner});
        }
        else
        {
            return ITensor::makeShape({mLocalNbLayers * batchSize * mMaxBeamWidth, mDState, mDInner});
        }
    }();
    auto const convStatesShape = [&]()
    {
        if (mUseMambaConv1dPlugin)
        {
            return ITensor::makeShape({mLocalNbLayers * batchSize * mMaxBeamWidth, mDConv - 1, mDInner});
        }
        else
        {
            return ITensor::makeShape({mLocalNbLayers * batchSize * mMaxBeamWidth, mDInner, mDConv - 1});
        }
    }();
    mambaSsmStates->reshape(ssmStatesShape);
    mambaConvStates->reshape(convStatesShape);
    mambaConvStatesAlt->reshape(convStatesShape);

    mambaSsmState.resize(mLocalNbLayers);
    mambaConvState.resize(mLocalNbLayers);
    mambaConvStateAlt.resize(mLocalNbLayers);
    for (int i = 0; i < mLocalNbLayers; i++)
    {
        size_t offset = batchSize * mMaxBeamWidth * i;
        mambaSsmState[i] = ITensor::slice(mambaSsmStates, offset, batchSize * mMaxBeamWidth);
        mambaConvState[i] = ITensor::slice(mambaConvStates, offset, batchSize * mMaxBeamWidth);
        mambaConvStateAlt[i] = ITensor::slice(mambaConvStatesAlt, offset, batchSize * mMaxBeamWidth);
    }
    if (slotMappingDevice != nullptr)
    {
        TLLM_CHECK(slotMappingHost != nullptr);
        TLLM_CHECK(mambaSsmStates != nullptr && mambaConvStates != nullptr);
        TLLM_CHECK(mambaSsmStatePtrs != nullptr && mambaConvStatePtrs != nullptr);

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

void SsmStateBuffers::fillStatePtrs()
{
    auto statePtrsShape = ITensor::makeShape({mLocalNbLayers});
    mambaSsmStatePtrs->reshape(statePtrsShape);
    mambaConvStatePtrs->reshape(statePtrsShape);

    mambaSsmStatePtr.resize(mLocalNbLayers);
    mambaConvStatePtr.resize(mLocalNbLayers);

    void** mambaSsmStatePtrArray = static_cast<void**>(mambaSsmStatePtrs->data());
    void** mambaConvStatePtrArray = static_cast<void**>(mambaConvStatePtrs->data());

    for (int i = 0; i < mLocalNbLayers; i++)
    {
        mambaSsmStatePtrArray[i] = mambaSsmState[i]->data();
        mambaConvStatePtrArray[i] = mambaConvState[i]->data();
        mambaSsmStatePtr[i] = ITensor::slice(mambaSsmStatePtrs, i, 1);
        mambaConvStatePtr[i] = ITensor::slice(mambaConvStatePtrs, i, 1);
    }
}

void SsmStateBuffers::reshape(
    GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    auto const batchSize = generationConfig.batchSize;

    reshape(batchSize);
}

void SsmStateBuffers::reset(BufferManager& manager)
{
    // This is not need in Plugin path, but may be needed for OOTB path.
    manager.setZero(*mambaSsmStates);
    manager.setZero(*mambaConvStates);
    manager.setZero(*mambaConvStatesAlt);
}

SsmStateBuffers SsmStateBuffers::sliceTo(SizeType offset, SizeType size)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SsmStateBuffers buffers;
    buffers.mambaSsmState = utils::sliceBufferVector(mambaSsmState, offset, size);
    buffers.mambaConvState = utils::sliceBufferVector(mambaConvState, offset, size);
    buffers.mambaConvStateAlt = utils::sliceBufferVector(mambaConvStateAlt, offset, size);

    if (slotMappingDevice != nullptr)
    {
        TLLM_CHECK(slotMappingHost != nullptr);
        TLLM_CHECK(mambaSsmStates != nullptr && mambaConvStates != nullptr);
        TLLM_CHECK(mambaSsmStatePtrs != nullptr && mambaConvStatePtrs != nullptr);
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

void SsmStateBuffers::prepareContextStep(RuntimeBuffers* runtimeBuffers, BufferManager& manager)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType const batchSize = runtimeBuffers->generationConfig.batchSize;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto RequestTypesPtr = bufferCast<int32_t>(*requestTypes);
    TLLM_CHECK(requestTypes->getSize() == static_cast<std::size_t>(batchSize));
    std::fill_n(RequestTypesPtr, batchSize, 0);

    manager.setZero(*mambaConvStates);
    if (slotMappingDevice != nullptr)
    {
        manager.copy(*slotMappingHost, *slotMappingDevice);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void SsmStateBuffers::tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, ModelConfig const& modelConfig,
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

void SsmStateBuffers::postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
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

void SsmStateBuffers::getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers,
    TensorMap& outputBuffers, SizeType const step, TensorPtr const& inputIds, ModelConfig const& modelConfig,
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

    auto const localNbLayers = modelConfig.getNbSsmLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;
    auto const& layerTypes = modelConfig.getLayerTypes();

    if (modelConfig.usePagedState())
    {
        auto const* const ssmStatePtrName = mIsRecurrentGemma ? "rnn_state_ptr_" : "ssm_state_ptr_";
        inputBuffers.insert_or_assign("slot_mapping", slotMappingDevice);
        utils::insertTensorVector(inputBuffers, "conv_state_ptr_", mambaConvStatePtr, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(inputBuffers, ssmStatePtrName, mambaSsmStatePtr, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
    }
    else
    {
        auto const* const ssmPastStatePtrName = mIsRecurrentGemma ? "past_rnn_state_" : "past_ssm_state_";
        auto const* const ssmPresentStatePtrName = mIsRecurrentGemma ? "present_rnn_state_" : "present_ssm_state_";
        utils::insertTensorVector(inputBuffers, "past_conv_state_", (step % 2) ? mambaConvState : mambaConvStateAlt,
            firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(outputBuffers, "present_conv_state_", (step % 2) ? mambaConvStateAlt : mambaConvState,
            firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(inputBuffers, ssmPastStatePtrName, mambaSsmState, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(outputBuffers, ssmPresentStatePtrName, mambaSsmState, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
    }

    inputBuffers.insert_or_assign("host_request_types", requestTypes);
    inputBuffers.insert_or_assign("host_context_lengths", runtimeBuffers->contextLengthsHost);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}
