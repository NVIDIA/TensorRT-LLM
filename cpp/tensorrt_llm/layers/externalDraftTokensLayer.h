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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

#include <curand_kernel.h>

namespace tensorrt_llm::layers
{

//! \brief Top class for sampling layers.
//! It sets up and executes TopKSamplingLayer and TopPSamplingLayer samplings
template <typename T>
class ExternalDraftTokensLayer : public BaseLayer
{
public:
    using Base = BaseLayer;

    ExternalDraftTokensLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager, bool isDeterministic = true, bool isAirTopP = true);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    using Base::mDecoderDomain;

    executor::DecodingMode mDecodingMode;

    size_t mWorkspaceSize{0};
    size_t mSetupWorkspaceSize{0};

    TensorPtr mCurandStatesDevice;
    TensorPtr mSkipTopKDecodeDevice;
    TensorPtr mSkipTopKDecodeHost;
    TensorPtr mSkipTopPDecodeDevice;
    TensorPtr mSkipTopPDecodeHost;

    TensorPtr mBatchIsAccepted;
    TensorPtr mRuntimeMultinomialDevice;

    TensorPtr mOutputIdsAfterSampling;
    TensorPtr mOutputIdsAfterSamplingPtrsHost;
    TensorPtr mOutputIdsAfterSamplingPtrsDevice;
    TensorPtr mTargetOutputIds;
    TensorPtr mRuntimeTopKDevice;
    TensorPtr mRuntimeTopKHost;
    TensorPtr mRuntimeTopPDevice;
    TensorPtr mReturnAllSelectedTokensPerSlotHost;
    TensorPtr mReturnAllSelectedTokensPerSlotDevice;
    TensorPtr mMaskBuffer;

    TensorPtr mTargetLogits;

    // AirTopP
    cudaDeviceProp mDeviceProp;
    runtime::SizeType32 mAirTopPBlockNum{0};
    bool mIsDeterministic{true};
    bool mIsAirTopP{false};

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
    void prepareInputs(
        std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs);
    void targetSoftmax(std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void acceptDraftTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void multinomialSampling(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void getAllTopKs(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void getAllTopPs(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void forwardAcceptedTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
};

} // namespace tensorrt_llm::layers
