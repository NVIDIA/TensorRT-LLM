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
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{

class RuntimeBuffers;

class SsmStateBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TensorMap = StringPtrMap<ITensor>;

    // Mamba states:  mamba_d_inner = mamba_expand * hidden_size
    TensorPtr mambaSsmStates;                 // [layer_count * batch_beam, mamba_d_state, mamba_d_inner]
    TensorPtr mambaConvStates;                // [layer_count * batch_beam, mamba_d_conv - 1, mamba_d_inner]
    TensorPtr mambaConvStatesAlt;             // [layer_count * batch_beam, mamba_d_conv - 1, mamba_d_inner]

    std::vector<TensorPtr> mambaSsmState;     // [batch_beam, mamba_d_state, mamba_d_inner]
    std::vector<TensorPtr> mambaConvState;    // [batch_beam, mamba_d_conv - 1, mamba_d_inner]
    std::vector<TensorPtr> mambaConvStateAlt; // [batch_beam, mamba_d_conv - 1, mamba_d_inner]

    TensorPtr slotMappingHost;                // [batch_size]
    TensorPtr slotMappingDevice;              // [batch_size]
    TensorPtr mambaSsmStatePtrs;              // [layer_count]
    TensorPtr mambaConvStatePtrs;             // [layer_count]

    std::vector<TensorPtr> mambaSsmStatePtr;  // [1]
    std::vector<TensorPtr> mambaConvStatePtr; // [1]

    SsmStateBuffers();

    SsmStateBuffers(TllmRuntime const& runtime, runtime::GptModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void reshape(SizeType batchSize);
    void reshape(
        GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    SsmStateBuffers sliceTo(SizeType offset, SizeType size);

    void prepareContextStep(RuntimeBuffers* runtimeBuffers, BufferManager& manager);

    void postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
        BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
        SizeType const step, TensorPtr const& inputIds, TensorPtr const& commPtrs, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

protected:
    void tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void fillStatePtrs();

private:
    SizeType mDConv = 0;
    SizeType mDState = 0;
    SizeType mDInner = 0;

    int mLocalNbLayers = 0;
    int mMaxBeamWidth = 0;

    bool mUseMambaConv1dPlugin = true;
};

} // namespace tensorrt_llm::runtime
