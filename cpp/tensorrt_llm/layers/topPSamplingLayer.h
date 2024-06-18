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
    TopPSamplingLayer(DecoderDomain const& decoderDomain, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, bool isDeterministic = true,
        bool isAirTopP = true);
    ~TopPSamplingLayer();

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams) override;
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs) override;

    [[nodiscard]] bool const* getSkipDecodeHost() const
    {
        return mSkipDecodeHost;
    }

protected:
    runtime::SizeType32* mRuntimeTopKDevice{nullptr};
    float* mRuntimeTopPDevice{nullptr};
    float mRuntimeMaxTopP{0.f};
    float* mInitialTopPDevice{nullptr};
    float* mTopPDecayDevice{nullptr};
    float* mTopPMinDevice{nullptr};
    runtime::TokenIdType* mTopPResetIdsDevice{nullptr};
    void* mSetupWorkspaceDevice{nullptr};

    bool* mSkipDecodeDevice{nullptr};
    bool* mSkipDecodeHost{nullptr};
    runtime::SizeType32 mAirTopPBlockNum{0};

    cudaDeviceProp mDeviceProp;
    bool mIsDeterministic{true};
    bool mIsAirTopP{false};

    using Base::mWorkspaceSize;
    using Base::mAllocatedSize;

    using Base::mStream;
    using Base::mAllocator;

    using Base::mDecoderDomain;

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
    void freeBuffer();
};

} // namespace tensorrt_llm::layers
