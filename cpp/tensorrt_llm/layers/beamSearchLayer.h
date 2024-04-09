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

template <typename T>
class BeamSearchLayer : public BaseLayer
{
public:
    // BS: batch_size, lBS: local_batch_size, BM: beam_width, mSL: max_seq_length
    class SetupParams : public DecodingSetupParams
    {
    public:
        std::optional<std::vector<float>> beam_search_diversity_rate; // [BS] on cpu
        std::optional<std::vector<float>> length_penalty;             // [BS] on cpu
        std::optional<std::vector<int>> early_stopping;               // [BS] on cpu
    };

    class ForwardParams : public DecodingParams
    {
    public:
        explicit ForwardParams(runtime::SizeType step, runtime::SizeType ite, tc::Tensor logits, tc::Tensor endIds,
            tc::Tensor src_cache_indirection, runtime::SizeType max_attention_window,
            runtime::SizeType sink_token_length, runtime::SizeType max_seq_len)
            : DecodingParams(step, ite, std::move(logits), std::move(endIds))
            , src_cache_indirection{std::move(src_cache_indirection)}
            , max_attention_window{max_attention_window}
            , sink_token_length{sink_token_length}
            , max_seq_len{max_seq_len}
        {
        }

        // mandatory parameters
        runtime::SizeType max_attention_window;
        runtime::SizeType sink_token_length;
        runtime::SizeType max_seq_len;
        std::optional<tc::Tensor> input_lengths; // [BS, BM]
        tc::Tensor src_cache_indirection;        // [BS, BM, mSL]
    };

    class OutputParams : public DecodingOutputParams
    {
    public:
        explicit OutputParams(tc::Tensor outputIds, tc::Tensor parentIds, tc::Tensor tgt_cache_indirection)
            : DecodingOutputParams{std::move(outputIds)}
            , parent_ids{std::move(parentIds)}
            , tgt_cache_indirection{std::move(tgt_cache_indirection)}
        {
        }

        std::shared_ptr<kernels::BeamHypotheses> beamHypotheses;
        tc::Tensor parent_ids;            // [BS, BM, mSL]
        tc::Tensor tgt_cache_indirection; // [BS, BM, mSL]
        tc::Tensor parent_ids_ptr;        // [BS][BM, mSL]
    };

    BeamSearchLayer(runtime::SizeType const vocab_size, runtime::SizeType const vocab_size_padded, cudaStream_t stream,
        std::shared_ptr<tc::IAllocator> allocator);

    BeamSearchLayer(BeamSearchLayer<T> const& beam_search_layer);

    ~BeamSearchLayer() override;

    void setup(runtime::SizeType const batch_size, runtime::SizeType const beam_width, SetupParams const& setupParams);

    void forward(OutputParams& outputs, ForwardParams const& params);

protected:
    using BaseLayer::mAllocator;
    using BaseLayer::mStream;

    bool mIsAllocateBuffer;
    runtime::SizeType mVocabSize{0};
    runtime::SizeType mVocabSizePadded{0};
    size_t mWorkspaceSize{0};
    void* mWorkspace{nullptr};
    // TODO: use pinned memory to simplify the buffers?
    float* mDiversityRateDevice;
    float* mLengthPenaltyDevice;
    int* mEarlyStoppingDevice;
    std::vector<float> mDiversityRateHost;
    std::vector<float> mLengthPenaltyHost;
    std::vector<int> mEarlyStoppingHost;

private:
    void allocateBuffer(runtime::SizeType const batch_size, runtime::SizeType const beam_width);
    void freeBuffer();
};

} // namespace layers
} // namespace tensorrt_llm
