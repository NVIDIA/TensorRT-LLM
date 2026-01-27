/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <optional>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

class RnnStateManager
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using RequestIdType = tensorrt_llm::batch_manager::RequestIdType;

    RnnStateManager(SizeType32 maxNumSequences, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, tensorrt_llm::runtime::BufferManager const& bufferManager);

    RnnStateManager(SizeType32 dState, SizeType32 dConv, SizeType32 numHeads, SizeType32 nGroups, SizeType32 headDim,
        SizeType32 maxBatchSize, runtime::WorldConfig const& worldConfig, int64_t stream, nvinfer1::DataType dtype,
        nvinfer1::DataType ssmCacheDtype, std::vector<SizeType32> const& ppLayers);

    void getPtrBuffers(TensorMap& inputBuffers, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void fillSlotMapping(
        runtime::ITensor& dstPointers, SizeType32 dstSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const;

    void allocateCacheBlocks(std::vector<RequestIdType> const& requestIds);

    void freeCacheBlock(RequestIdType requestId);

    [[nodiscard]] SizeType32 getCacheIndex(RequestIdType requestId) const;

    [[nodiscard]] std::vector<SizeType32> getStateIndices(
        std::vector<RequestIdType> const& requestIds, std::vector<bool> const& isPadding);

    [[nodiscard]] TensorPtr getConvStates(SizeType32 layerIdx) const;

    [[nodiscard]] TensorPtr getSsmStates(SizeType32 layerIdx) const;

    [[nodiscard]] nvinfer1::DataType getConvStateDataType() const noexcept;

    [[nodiscard]] nvinfer1::DataType getSsmStateDataType() const noexcept;

    [[nodiscard]] executor::rnn_cache::RnnCacheState::ModelConfig getRnnCacheStateModelConfig() const noexcept;

    /// Returns the number of local RNN layers on this PP rank
    [[nodiscard]] SizeType32 getNumLocalLayers() const noexcept;

    /// Returns the buffer manager
    [[nodiscard]] runtime::BufferManager const& getBufferManager() const noexcept
    {
        return mBufferManager.value();
    }

private:
    static std::vector<SizeType32> getPpLayers(SizeType32 numLayers, runtime::WorldConfig const& worldConfig,
        std::optional<std::vector<bool>> const& layerMask);

    // If we need support beam search, we may need mMaxBeamWidth + 1 slots and use separate input / output states.
    TensorPtr pagedRnnStates;  // [local_nb_layers, max_seq_num * max_beam_width, state_size, rnn_hidden_size] or
                               // [local_nb_layers, max_seq_num * max_beam_width, num_heads, state_size, rnn_head_size]
    TensorPtr pagedConvStates; // [local_nb_layers, max_seq_num * max_beam_width, conv_kernel - 1, rnn_hidden_size]

    TensorPtr rnnStatePtrs;    // [layer_count]
    TensorPtr convStatePtrs;   // [layer_count]

    std::vector<TensorPtr> rnnStatePtr;  // [1]
    std::vector<TensorPtr> convStatePtr; // [1]

    SizeType32 mMaxNumSequences = 0;
    SizeType32 mMaxBeamWidth = 0;
    SizeType32 mBeamSlotsPerSequence = 0;
    std::unordered_map<SizeType32, SizeType32> mLayerOffsets;
    std::vector<SizeType32> mFreeBlocks;
    std::unordered_map<RequestIdType, SizeType32> mCacheIndex;
    std::optional<runtime::BufferManager> mBufferManager;
    nvinfer1::DataType mDtype{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mSsmCacheDtype{nvinfer1::DataType::kFLOAT};

    // RNN model config (global values before TP/PP split)
    SizeType32 mDState{0};
    SizeType32 mDConv{0};
    SizeType32 mHiddenSize{0};
    SizeType32 mHeadDim{0};
    SizeType32 mConvDimSize{0};
    SizeType32 mNGroups{0};
    SizeType32 mNumLayers{0};
    SizeType32 mNumHeads{0};
    SizeType32 mNumLocalLayers{0};
};

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
