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
#include "tensorrt_llm/runtime/iTensor.h"

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
    PenaltyLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);

    ~PenaltyLayer() override;

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams) override;

    //! \brief Modifies 'outputs->logits' in-place with -INF for banned words
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs) override;

    T* getRuntimeLogitsDevice()
    {
        return mRuntimeLogitsDevice;
    }

private:
    void initialize();
    void allocateWorkspace();
    void allocateBuffer();
    void freeBuffer();

private:
    using BaseLayer::mWorkspaceSize;
    using BaseLayer::mAllocatedSize;

    using BaseLayer::mStream;
    using BaseLayer::mAllocator;

    using BaseLayer::mDecoderDomain;

    executor::DecodingMode mDecodingMode;

    float* mTemperatureDevice{nullptr};
    float* mRepetitionPenaltyDevice{nullptr};
    float* mPresencePenaltyDevice{nullptr};
    float* mFrequencyPenaltyDevice{nullptr};
    runtime::SizeType32* mMinLengthDevice{nullptr};
    T* mRuntimeLogitsDevice{nullptr};

    std::vector<float> mTemperature;
    std::vector<float> mRepetitionPenalty;
    std::vector<float> mPresencePenalty;
    std::vector<float> mFrequencyPenalty;
    std::vector<SizeType32> mMinLength;

    bool mUseTemperature{false};
    bool mUseRepetitionPenalty{false};
    bool mUsePresencePenalty{false};
    bool mUseFrequencyPenalty{false};
    bool mUseMinLength{false};

    runtime::SizeType32 mCyclicStep{0};
    runtime::SizeType32 mRuntimeMaxSeqLen{0};
    runtime::SizeType32 mConfiguredBeamWidth{-1};

    runtime::TokenIdType* mPenaltyWorkspaceDevice{nullptr};
    runtime::TokenIdType* mPenaltyWorkspacePrevDevice{nullptr};
    runtime::ITensor::SharedPtr mLogitsPtrsHost;
};

} // namespace tensorrt_llm::layers
