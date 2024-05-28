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
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

#include <utility>

#include <optional>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

// BS: batch_size, lBS: local_batch_size, BM: beam_width, mSL: max_seq_length
class BeamSearchSetupParams : public BaseSetupParams
{
public:
    std::optional<std::vector<float>> beam_search_diversity_rate; // [BS] on cpu
    std::optional<std::vector<float>> length_penalty;             // [BS] on cpu
    std::optional<std::vector<int>> early_stopping;               // [BS] on cpu
    bool hasDiffRuntimeArgs{false};
};

class BeamSearchInputParams : public BaseInputParams
{
public:
    explicit BeamSearchInputParams(runtime::SizeType32 step, runtime::SizeType32 ite, tc::Tensor logits,
        tc::Tensor endIds, tc::Tensor src_cache_indirection, runtime::SizeType32 max_attention_window,
        runtime::SizeType32 sink_token_length, runtime::SizeType32 max_seq_len)
        : BaseInputParams(step, ite, std::move(endIds))
        , logits{std::move(logits)}
        , max_attention_window{max_attention_window}
        , sink_token_length{sink_token_length}
        , max_seq_len{max_seq_len}
        , src_cache_indirection{std::move(src_cache_indirection)}
    {
    }

    // mandatory parameters
    tc::Tensor logits; // [maxBatchSize, beamWidth, vocabSizePadded]
    runtime::SizeType32 max_attention_window;
    runtime::SizeType32 sink_token_length;
    runtime::SizeType32 max_seq_len;
    tc::Tensor src_cache_indirection;        // [BS, BM, mSL]
    std::optional<tc::Tensor> input_lengths; // [BS, BM]
};

class BeamSearchOutputParams : public BaseOutputParams
{
public:
    explicit BeamSearchOutputParams(tc::Tensor outputIds, tc::Tensor parentIds, tc::Tensor tgt_cache_indirection)
        : BaseOutputParams{std::move(outputIds)}
        , parent_ids{std::move(parentIds)}
        , tgt_cache_indirection{std::move(tgt_cache_indirection)}
    {
    }

    std::shared_ptr<kernels::BeamHypotheses> beamHypotheses;
    tc::Tensor parent_ids;            // [BS, BM, mSL]
    tc::Tensor tgt_cache_indirection; // [BS, BM, mSL]
    tc::Tensor parent_ids_ptr;        // [BS][BM, mSL]
};

template <typename T>
class BeamSearchLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    BeamSearchLayer(DecoderDomain const& decoderDomain, cudaStream_t stream, std::shared_ptr<tc::IAllocator> allocator);

    ~BeamSearchLayer() override;

    void setup(runtime::SizeType32 const batch_size, runtime::SizeType32 const beamWidth,
        runtime::SizeType32 const* batchSlots, std::shared_ptr<BaseSetupParams> setupParams) override;

    void forwardAsync(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) override;

private:
    void forwardAsyncSingleRequest(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs);

    void allocateBuffer(runtime::SizeType32 const batch_size, runtime::SizeType32 const beam_width);
    void freeBuffer();

private:
    using Base::mAllocator;
    using Base::mStream;

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

} // namespace layers
} // namespace tensorrt_llm
