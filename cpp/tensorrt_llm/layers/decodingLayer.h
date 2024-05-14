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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/beamSearchLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/medusaDecodingLayer.h"
#include "tensorrt_llm/layers/samplingLayer.h"
#include "tensorrt_llm/runtime/decodingMode.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

//! \brief Layer performs token decoding using sampling (beamWidth=1), beam search (beamWidth>1) or Medusa.
template <typename T>
class DecodingLayer : public BaseLayer
{
public:
    DecodingLayer(runtime::DecodingMode const& mode, DecoderDomain const& decoderDomain, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);

    ~DecodingLayer() override = default;

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> setupParams) override;

    //! \brief Calls single SamplingLayer::forward in batched mode
    //! or runs BeamSearchLayer::forward in the loop for each request.
    //! Modifies outputs->logits in-place.
    void forward(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) override;

private:
    using BaseLayer::mWorkspaceSize;
    using BaseLayer::mAllocatedSize;

    using BaseLayer::mStream;
    using BaseLayer::mAllocator;

    runtime::DecodingMode mDecodingMode;

    std::unique_ptr<BaseLayer> mDecodingLayer;

    bool mHasDiffRuntimeArgs{false};
};

} // namespace layers
} // namespace tensorrt_llm
