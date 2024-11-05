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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

class RnnStateManager
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    RnnStateManager(SizeType32 maxNumSequences, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, tensorrt_llm::runtime::BufferManager const& bufferManager);

    void getPtrBuffers(TensorMap& inputBuffers, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void fillSlotMapping(
        runtime::ITensor& dstPointers, SizeType32 dstSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const;

private:
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
};

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
