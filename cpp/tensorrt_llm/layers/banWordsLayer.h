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

//! \brief Layer to ban specific words from being sampled.
//! Supports banning bad words and repeating N grams.
//! Set badWordsPtr, maxBadWordsLen and badWordsLengths to ban bad words.
//! Set noRepeatNgramSize in input params to ban repeat Ngrams.
//! Layer modifies logits in-place.
template <typename T>
class BanWordsLayer : public BaseLayer
{

public:
    BanWordsLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& baseSetupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! \brief Modifies 'outputs->logits' in-place with -INF for banned words
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

private:
    void allocateBuffer();
    void banBadWords(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots,
        DecoderDomain const& decoderDomain, runtime::SizeType32 maxSeqLen);
    void banRepeatNGrams(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots,
        BufferPtr noRepeatNgramSizeDevice, DecoderDomain const& decoderDomain, runtime::SizeType32 maxSeqLen,
        bool useNoRepeatNgramSize);

private:
    executor::DecodingMode mDecodingMode;

    TensorPtr mNoRepeatNgramSizeDevice;
    TensorPtr mNoRepeatNgramSize;
    bool mUseNoRepeatNgramSize{false};
};

} // namespace tensorrt_llm::layers
