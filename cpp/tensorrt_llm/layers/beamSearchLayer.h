/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

#include <optional>
#include <utility>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::layers
{

class BeamSearchInputParams : public DecodingInputs
{
public:
    explicit BeamSearchInputParams(runtime::SizeType32 step, runtime::SizeType32 ite, tc::Tensor logits,
        tc::Tensor endIds, tc::Tensor srcCacheIndirection, runtime::SizeType32 maxAttentionWindow,
        runtime::SizeType32 sinkTokenLength, runtime::SizeType32 maxSeqLen, runtime::SizeType32 localBatchSize)
        : DecodingInputs(std::move(endIds), step, ite, localBatchSize)
        , logits{std::move(logits)}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , maxSeqLen{maxSeqLen}
        , srcCacheIndirection{std::move(srcCacheIndirection)}
    {
    }

    // mandatory parameters
    tc::Tensor logits; // [maxBatchSize, beamWidth, vocabSizePadded]
    runtime::SizeType32 maxAttentionWindow;
    runtime::SizeType32 sinkTokenLength;
    runtime::SizeType32 maxSeqLen;
    tc::Tensor srcCacheIndirection;         // [BS, BM, mSL]
    std::optional<tc::Tensor> inputLengths; // [BS, BM]
};

template <typename T>
class BeamSearchLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    BeamSearchLayer(DecoderDomain const& decoderDomain, cudaStream_t stream, std::shared_ptr<tc::IAllocator> allocator);

    ~BeamSearchLayer() override;

    void setup(runtime::SizeType32 const batchSize, runtime::SizeType32 const beamWidth,
        runtime::SizeType32 const* batchSlots, std::shared_ptr<BaseSetupParams> const& setupParams) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs) override;

private:
    void forwardAsyncSingleRequest(
        std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& inputs);

    void allocateBuffer(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth);
    void freeBuffer();

private:
    using Base::mAllocator;
    using Base::mStream;

    using Base::mDecoderDomain;

    bool mIsAllocateBuffer;
    runtime::SizeType32 mVocabSize{0};
    runtime::SizeType32 mVocabSizePadded{0};
    size_t mWorkspaceSize{0};
    void* mWorkspace{nullptr};
    // TODO: use pinned memory to simplify the buffers?
    float* mDiversityRateDevice;
    float* mLengthPenaltyDevice;
    int* mEarlyStoppingDevice;
    std::vector<float> mDiversityRateHost;
    std::vector<float> mLengthPenaltyHost;
    std::vector<int> mEarlyStoppingHost;
    bool mHasDiffRuntimeArgs{false};
};

} // namespace tensorrt_llm::layers
