/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <curand_kernel.h>

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"

namespace tensorrt_llm::layers
{

//! \brief Layer applies penalties to the logits. Supports:
//! 1. Temperature
//! 2. Repetition penalty
//! 3. Presence penalty
//! 4. Frequency penalty
//! 5. Min length penalty
template <typename T>
class PenaltyLayer : public BaseLayer
{
public:
    PenaltyLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! \brief Modifies 'outputs->logits' in-place with -INF for banned words
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void initialize();
    void allocateWorkspace();
    void allocateBuffer();

private:
    using BaseLayer::mDecoderDomain;

    executor::DecodingMode mDecodingMode;

    size_t mWorkspaceSize{};
    TensorPtr mTemperatureDevice;
    TensorPtr mRepetitionPenaltyDevice;
    TensorPtr mPresencePenaltyDevice;
    TensorPtr mFrequencyPenaltyDevice;
    TensorPtr mMinLengthDevice;

    TensorPtr mTemperature;
    TensorPtr mRepetitionPenalty;
    TensorPtr mPresencePenalty;
    TensorPtr mFrequencyPenalty;
    TensorPtr mMinLength;

    bool mUseTemperature{false};
    bool mUseRepetitionPenalty{false};
    bool mUsePresencePenalty{false};
    bool mUseFrequencyPenalty{false};
    bool mUseMinLength{false};

    runtime::SizeType32 mCyclicStep{0};
    runtime::SizeType32 mRuntimeMaxSeqLen{0};
    runtime::SizeType32 mConfiguredBeamWidth{-1};

    BufferPtr mPenaltyWorkspaceDevice;
    BufferPtr mPenaltyWorkspacePrevDevice;
    TensorPtr mLogitsPtrsHost;
};

} // namespace tensorrt_llm::layers
