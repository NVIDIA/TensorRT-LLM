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

//! \brief Layer performs token decoding using sampling (beamWidth=1), beam search (beamWidth>1) or Medusa.
template <typename T>
class DecodingLayer : public BaseLayer
{
public:
    DecodingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! \brief Calls single SamplingLayer::forwardAsync or MedusaDecodingLayer::forwardAsync in batched mode
    //! or runs BeamSearchLayer::forwardAsync in the loop for each request.
    //! Modifies outputs->logits in-place.
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! \brief Calls forwardSync of configured decoding layer.
    void forwardSync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    [[nodiscard]] std::tuple<std::shared_ptr<BaseDecodingOutputs>, std::shared_ptr<BaseDecodingInputs>> prepareParams(
        std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& inputs) const;

private:
    using BaseLayer::mDecoderDomain;

    executor::DecodingMode mDecodingMode;

    std::unique_ptr<BaseLayer> mDecodingLayer;
};

} // namespace tensorrt_llm::layers
