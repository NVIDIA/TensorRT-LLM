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

#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::layers
{

//! \brief Layer to randomly sample tokens from TopP logits.
//! Layer expects probs precomputed in "logits" tensor
template <typename T>
class TopPSamplingLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    TopPSamplingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager,
        bool isDeterministic = true, bool isAirTopP = true);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

protected:
    TensorPtr mRuntimeTopKDevice;
    TensorPtr mRuntimeTopPDevice;
    TensorPtr mInitialTopPDevice;
    TensorPtr mTopPDecayDevice;
    TensorPtr mTopPMinDevice;
    TensorPtr mTopPResetIdsDevice;

    TensorPtr mSkipDecodeDevice;
    TensorPtr mSkipDecodeHost;
    size_t mWorkspaceSize{0};
    size_t mSetupWorkspaceSize{0};

    // AirTopP
    cudaDeviceProp mDeviceProp;
    runtime::SizeType32 mAirTopPBlockNum{0};
    bool mIsDeterministic{true};
    bool mIsAirTopP{false};

    using Base::mDecoderDomain;

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
};

} // namespace tensorrt_llm::layers
