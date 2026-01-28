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
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"

#include <unordered_set>

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

    // TODO(shreyasm): This might not be correct with ADP cause of getPipelineParallelRank method. 
    // This constructor is not used so should be ok for now.
    SizeType32 totalRnnLayers = modelConfig.getNbRnnLayers();
    SizeType32 ppSize = worldConfig.getPipelineParallelism();
    SizeType32 ppRank = worldConfig.getPipelineParallelRank();

    SizeType32 layersPerRank = totalRnnLayers / ppSize;
    SizeType32 remainder = totalRnnLayers % ppSize;
    SizeType32 startLayer = ppRank * layersPerRank + std::min(ppRank, static_cast<SizeType32>(remainder));

    mGlobalLayerNumsPerPP.resize(localNbLayers);
    for (SizeType32 i = 0; i < localNbLayers; i++)
    {
        mGlobalLayerNumsPerPP[i] = startLayer + i;
        mLayerOffsets[startLayer + i] = i;
    }

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

    mDtype = dataType;
    mSsmCacheDtype = nvinfer1::DataType::kFLOAT;

    // Store RNN model config for CacheTransceiver
    mDState = stateSize;
    mDConv = convKernel;
    mHiddenSize = rnnHiddenSize;
    mHeadDim = rnnHeadSize;
    mConvDimSize = rnnConvDimSize;
    mNGroups = 0; // Not available in ModelConfig-based constructor
    mNumLayers = modelConfig.getNbRnnLayers();
    mNumHeads = rnnHeadSize > 0 ? (rnnHiddenSize / rnnHeadSize) : 0;
    mNumLocalLayers = localNbLayers;

    pagedRnnStates = bufferManager.gpu(rnnStateShape, mSsmCacheDtype);
    pagedConvStates = bufferManager.gpu(convStateShape, mDtype);

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

RnnStateManager::RnnStateManager(SizeType32 dState, SizeType32 dConv, SizeType32 numHeads, SizeType32 nGroups,
    SizeType32 headDim, SizeType32 maxBatchSize, WorldConfig const& worldConfig, int64_t stream,
    nvinfer1::DataType dtype, nvinfer1::DataType ssmCacheDtype, std::vector<SizeType32> const& ppLayers)
    : mMaxNumSequences(maxBatchSize)
    , mMaxBeamWidth{1}
    , mBeamSlotsPerSequence{1}
    , mBufferManager{std::make_shared<CudaStream>(reinterpret_cast<cudaStream_t>(stream))}
    , mDtype{dtype}
    , mSsmCacheDtype{ssmCacheDtype} // Store global RNN model config
    , mDState{dState}
    , mDConv{dConv}
    , mHiddenSize{headDim * numHeads}
    , mHeadDim{headDim}
    , mConvDimSize{headDim * numHeads + 2 * nGroups * dState}
    , mNGroups{nGroups}
    , mNumLayers{numLayers}
    , mNumHeads{numHeads}
// Note: mNumLocalLayers is set in the body after ppLayers is computed
{
    auto const tpSize = worldConfig.getTensorParallelism();

    auto const dInner = headDim * numHeads;
    auto convDim = dInner + 2 * nGroups * dState;
    auto nheads = numHeads;

    TLLM_CHECK_WITH_INFO(nheads % tpSize == 0, "nheads must be divisible by tp_size");
    TLLM_CHECK_WITH_INFO(convDim % tpSize == 0, "conv_dim must be divisible by tp_size");

    convDim = convDim / tpSize;
    nheads = nheads / tpSize;

    auto const numLocalLayers = static_cast<SizeType32>(ppLayers.size());

    // Store local layer count
    mNumLocalLayers = numLocalLayers;
    mGlobalLayerNumsPerPP.resize(numLocalLayers);
    for (SizeType32 offset = 0; offset < numLocalLayers; ++offset)
    {
        mGlobalLayerNumsPerPP[offset] = ppLayers[offset];
        mLayerOffsets[ppLayers[offset]] = offset;
    }

    auto const convStateShape = ITensor::makeShape({numLocalLayers, maxBatchSize, convDim, dConv - 1});
    pagedConvStates = mBufferManager->gpu(convStateShape, dtype);

    auto const rnnStateShape = ITensor::makeShape({numLocalLayers, maxBatchSize, nheads, headDim, dState});
    pagedRnnStates = mBufferManager->gpu(rnnStateShape, ssmCacheDtype);

    mFreeBlocks.reserve(maxBatchSize);
    for (SizeType32 i = 0; i < maxBatchSize; ++i)
    {
        mFreeBlocks.push_back(i);
    }

    auto const statePtrsShape = ITensor::makeShape({numLocalLayers});
    rnnStatePtrs = BufferManager::cpu(statePtrsShape, TRTDataType<void*>::value);
    convStatePtrs = BufferManager::cpu(statePtrsShape, TRTDataType<void*>::value);
    auto* rnnStatePtrArray = bufferCast<void*>(*rnnStatePtrs);
    auto* convStatePtrArray = bufferCast<void*>(*convStatePtrs);

    rnnStatePtr.resize(numLocalLayers);
    convStatePtr.resize(numLocalLayers);
    for (SizeType32 i = 0; i < numLocalLayers; i++)
    {
        auto layerRnnStates = ITensor::slice(pagedRnnStates, i, 1);
        auto layerConvStates = ITensor::slice(pagedConvStates, i, 1);
        rnnStatePtrArray[i] = layerRnnStates->data();
        convStatePtrArray[i] = layerConvStates->data();
        rnnStatePtr[i] = ITensor::slice(rnnStatePtrs, i, 1);
        convStatePtr[i] = ITensor::slice(convStatePtrs, i, 1);
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

void RnnStateManager::allocateCacheBlocks(std::vector<RequestIdType> const& requestIds)
{
    for (auto const& requestId : requestIds)
    {
        auto it = mCacheIndex.find(requestId);
        if (it == mCacheIndex.end())
        {
            TLLM_CHECK_WITH_INFO(!mFreeBlocks.empty(), "Run out of RNN state cache blocks");
            SizeType32 const block = mFreeBlocks.back();
            mFreeBlocks.pop_back();
            mCacheIndex[requestId] = block;
        }
    }
}

void RnnStateManager::freeCacheBlock(RequestIdType requestId)
{
    auto it = mCacheIndex.find(requestId);
    if (it != mCacheIndex.end())
    {
        mFreeBlocks.push_back(it->second);
        mCacheIndex.erase(it);
    }
}

RnnStateManager::SizeType32 RnnStateManager::getCacheIndex(RequestIdType requestId) const
{
    auto it = mCacheIndex.find(requestId);
    TLLM_CHECK_WITH_INFO(it != mCacheIndex.end(), "Request ID not found in cache index");
    return it->second;
}

std::vector<RnnStateManager::SizeType32> RnnStateManager::getStateIndices(
    std::vector<RequestIdType> const& requestIds, std::vector<bool> const& isPadding)
{
    TLLM_CHECK_WITH_INFO(requestIds.size() == isPadding.size(), "requestIds and isPadding must have the same size");

    std::unordered_set<SizeType32> availableSlots;
    availableSlots.reserve(mMaxNumSequences);
    for (SizeType32 i = 0; i < mMaxNumSequences; ++i)
    {
        availableSlots.insert(i);
    }

    for (size_t i = 0; i < requestIds.size(); ++i)
    {
        if (!isPadding[i])
        {
            availableSlots.erase(getCacheIndex(requestIds[i]));
        }
    }

    std::vector<SizeType32> result;
    result.reserve(requestIds.size());
    auto availableIt = availableSlots.begin();

    for (size_t i = 0; i < requestIds.size(); ++i)
    {
        if (isPadding[i])
        {
            TLLM_CHECK_WITH_INFO(availableIt != availableSlots.end(), "Run out of available slots for padding");
            result.push_back(*availableIt);
            ++availableIt;
        }
        else
        {
            result.push_back(getCacheIndex(requestIds[i]));
        }
    }

    return result;
}

RnnStateManager::TensorPtr RnnStateManager::getConvStates(SizeType32 layerIdx) const
{
    auto it = mLayerOffsets.find(layerIdx);
    TLLM_CHECK_WITH_INFO(it != mLayerOffsets.end(), "Layer index not found in layer offsets");
    auto result = ITensor::slice(pagedConvStates, it->second, 1);
    result->squeeze(0);
    return result;
}

RnnStateManager::TensorPtr RnnStateManager::getSsmStates(SizeType32 layerIdx) const
{
    auto it = mLayerOffsets.find(layerIdx);
    TLLM_CHECK_WITH_INFO(it != mLayerOffsets.end(), "Layer index not found in layer offsets");
    auto result = ITensor::slice(pagedRnnStates, it->second, 1);
    result->squeeze(0);
    return result;
}

nvinfer1::DataType RnnStateManager::getConvStateDataType() const noexcept
{
    return mDtype;
}

nvinfer1::DataType RnnStateManager::getSsmStateDataType() const noexcept
{
    return mSsmCacheDtype;
}

executor::rnn_cache::RnnCacheState::ModelConfig RnnStateManager::getRnnCacheStateModelConfig() const noexcept
{
    return executor::rnn_cache::RnnCacheState::ModelConfig{
        mDState, mDConv, mHiddenSize, mHeadDim, mConvDimSize, mNGroups, mNumLayers, mNumHeads};
}

RnnStateManager::SizeType32 RnnStateManager::getNumLocalLayers() const noexcept
{
    return mNumLocalLayers;
}

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
