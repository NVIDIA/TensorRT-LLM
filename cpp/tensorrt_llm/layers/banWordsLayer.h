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

//! \brief Layer to ban specific words from being sampled.
//! Supports banning bad words and repeating N grams.
//! Set badWordsPtr, maxBadWordsLen and badWordsLengths to ban bad words.
//! Set noRepeatNgramSize in input params to ban repeat Ngrams.
//! Layer modifies logits in-place.
template <typename T>
class BanWordsLayer : public BaseLayer
{
public:
    BanWordsLayer(runtime::DecodingMode const& mode, DecoderDomain const& decoderDomain, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);

    ~BanWordsLayer() override = default;

    void setup(runtime::SizeType batchSize, runtime::SizeType beamWidth, runtime::SizeType const* batchSlots,
        std::shared_ptr<BaseSetupParams> setupParams) override;

    //! \brief Modifies 'outputs->logits' in-place with -INF for banned words
    void forward(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) override;

private:
    static void banRepeatNGrams(tc::Tensor& logits, std::shared_ptr<DynamicDecodeOutputParams> const& outputs,
        std::shared_ptr<DynamicDecodeInputParams> const& params, runtime::SizeType32 const* batchSlots,
        runtime::SizeType batchSize, runtime::SizeType beamWidth, runtime::SizeType maxSeqLen,
        runtime::SizeType vocabSizePadded, cudaStream_t stream);
    static void banBadWords(tc::Tensor& logits, std::shared_ptr<DynamicDecodeOutputParams> const& outputs,
        std::shared_ptr<DynamicDecodeInputParams> const& params, runtime::SizeType32 const* batchSlots,
        runtime::SizeType batchSize, runtime::SizeType beamWidth, runtime::SizeType maxSeqLen,
        runtime::SizeType vocabSizePadded, cudaStream_t stream);

private:
    using BaseLayer::mWorkspaceSize;
    using BaseLayer::mAllocatedSize;

    using BaseLayer::mStream;
    using BaseLayer::mAllocator;

    runtime::DecodingMode mDecodingMode;
};

} // namespace layers
} // namespace tensorrt_llm
