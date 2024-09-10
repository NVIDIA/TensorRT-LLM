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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"

#include <curand_kernel.h>

namespace tensorrt_llm::layers
{

//! \brief Layer to process stop criteria. Supports:
//! 1. Stop words criteria
//! 2. Maximum length criteria
template <typename T>
class StopCriteriaLayer : public BaseLayer
{
public:
    StopCriteriaLayer(executor::DecodingMode const& mode, DecoderDomain const& /* decoderDomain */,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    static void checkMaxLengthStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
        runtime::BufferManager const& bufferManager, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    static void checkStopWordsStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
        runtime::SizeType32 maxSeqLen, runtime::BufferManager const& bufferManager,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    static void checkEosToken(std::shared_ptr<BaseDecodingOutputs>& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
        runtime::BufferManager const& bufferManager, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

private:
    using BaseLayer::mDecoderDomain;

    executor::DecodingMode mDecodingMode;
    size_t mWorkspaceSize{0};
};

} // namespace tensorrt_llm::layers
