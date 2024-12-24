/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"

#include <curand_kernel.h>

namespace tensorrt_llm::layers
{

//! \brief Decoding layer for EAGLE speculative decoding technique.
template <typename T>
class EagleDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using PathsVec = std::vector<std::vector<std::vector<runtime::SizeType32>>>;

    EagleDecodingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void allocateBuffer();

    void fillContextBuffers(runtime::SizeType32 batchSize, BufferConstPtr batchSlots,
        EagleSetupParams const& setupParams, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void augmentBatchSlots(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void convertToPackedMask(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void packAcceptedPaths(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void unpackData(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

private:
    using Base::mDecoderDomain;

    size_t mWorkspaceSize{0};

    TensorPtr mTemperature;

    TensorPtr mCurandStatesDevice;
    TensorPtr mTemperatureDevice;

    TensorPtr mEagleNetCtxRequestTypes;
    TensorPtr mEagleNetCtxContextLengths;
    TensorPtr mEagleNetCtxPastKeyValueLengths;
    TensorPtr mEagleNetGenRequestTypes;
    TensorPtr mEagleNetGenContextLengths;
    TensorPtr mEagleNetGenPastKeyValueLengths;
};

} // namespace tensorrt_llm::layers
