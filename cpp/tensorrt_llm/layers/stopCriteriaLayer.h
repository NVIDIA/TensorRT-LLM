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
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/decodingMode.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm
{
namespace layers
{

//! \brief Layer to process stop criteria. Supports:
//! 1. Stop words criteria
//! 2. Maximum length criteria
template <typename T>
class StopCriteriaLayer : public BaseLayer
{
public:
    StopCriteriaLayer(runtime::DecodingMode const& mode, DecoderDomain const& /* decoderDomain */, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);

    ~StopCriteriaLayer() override = default;

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> setupParams) override;

    void forward(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) override;

private:
    static void checkMaxLengthStopCriteria(std::shared_ptr<DynamicDecodeOutputParams>& outputs,
        std::shared_ptr<DynamicDecodeInputParams> const& inputs, runtime::SizeType32 const* batchSlots,
        runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen,
        cudaStream_t stream);
    static void checkStopWordsStopCriteria(std::shared_ptr<DynamicDecodeOutputParams>& outputs,
        std::shared_ptr<DynamicDecodeInputParams> const& inputs, runtime::SizeType32 const* batchSlots,
        runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen,
        cudaStream_t stream);

private:
    using BaseLayer::mWorkspaceSize;
    using BaseLayer::mAllocatedSize;

    using BaseLayer::mStream;
    using BaseLayer::mAllocator;

    runtime::DecodingMode mDecodingMode;
};

} // namespace layers
} // namespace tensorrt_llm
