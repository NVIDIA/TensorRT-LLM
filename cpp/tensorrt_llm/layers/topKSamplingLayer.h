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

//! \brief Layer to randomly sample tokens from TopK logits.
//! When both TopK and TopP are specified, layer jointly samples using TopK and TopP.
//! When no TopK param is specified, sampling is skipped for particular request.
template <typename T>
class TopKSamplingLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    TopKSamplingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

protected:
    bool mNormalizeLogProbs{true};
    size_t mWorkspaceSize{0};
    size_t mSetupWorkspaceSize{0};
    TensorPtr mRuntimeTopKDevice;
    TensorPtr mRuntimeTopPDevice;
    TensorPtr mSkipDecodeDevice;
    TensorPtr mRuntimeTopKHost;
    TensorPtr mSkipDecodeHost;

    using Base::mDecoderDomain;

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
};

} // namespace tensorrt_llm::layers
