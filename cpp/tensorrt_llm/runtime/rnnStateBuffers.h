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

#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/generationConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{

class RuntimeBuffers;

class RnnStateBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TensorMap = StringPtrMap<ITensor>;

    TensorPtr rnnStates;                 // [layer_count * batch_beam, state_size, rnn_hidden_size]
    TensorPtr convStates;                // [layer_count * batch_beam, conv_kernel - 1, rnn_hidden_size]
    TensorPtr convStatesAlt;             // [layer_count * batch_beam, conv_kernel - 1, rnn_hidden_size]

    std::vector<TensorPtr> rnnState;     // [batch_beam, state_size, rnn_hidden_size]
    std::vector<TensorPtr> convState;    // [batch_beam, conv_kernel - 1, rnn_hidden_size]
    std::vector<TensorPtr> convStateAlt; // [batch_beam, conv_kernel - 1, rnn_hidden_size]

    TensorPtr slotMappingHost;           // [batch_size]
    TensorPtr slotMappingDevice;         // [batch_size]
    TensorPtr rnnStatePtrs;              // [layer_count]
    TensorPtr convStatePtrs;             // [layer_count]

    std::vector<TensorPtr> rnnStatePtr;  // [1]
    std::vector<TensorPtr> convStatePtr; // [1]

    RnnStateBuffers();

    RnnStateBuffers(
        TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void reshape(SizeType32 batchSize);
    void reshape(
        GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    RnnStateBuffers sliceTo(SizeType32 offset, SizeType32 size);

    void prepareContextStep(RuntimeBuffers* runtimeBuffers, BufferManager& manager);

    void postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
        BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
        SizeType32 const step, TensorPtr const& inputIds, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

protected:
    void tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void fillStatePtrs();

private:
    SizeType32 mConvKernel = 0;
    SizeType32 mStateSize = 0;
    SizeType32 mRnnHiddenSize = 0;

    int mLocalNbLayers = 0;
    int mMaxBeamWidth = 0;

    bool mUseMambaConv1dPlugin = true;

    bool mIsRecurrentGemma = false;
};

} // namespace tensorrt_llm::runtime
