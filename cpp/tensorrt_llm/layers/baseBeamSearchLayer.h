/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"

#include <utility>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
struct BeamHypotheses;
}

namespace layers
{

template <typename T>
class BaseBeamSearchLayer : public BaseLayer
{
public:
    using SetupParams = DecodingSetupParams;

    BaseBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, tc::IAllocator* allocator,
        bool is_free_buffer_after_forward);

    BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer);

    ~BaseBeamSearchLayer() override;

    using SoftmaxParams = DecodingParams;

    class ForwardParams : public SoftmaxParams
    {
    public:
        ForwardParams(int step, int ite, tc::Tensor logits, tc::Tensor endIds, tc::Tensor src_cache_indirection,
            int max_attention_window, int max_seq_len)
            : SoftmaxParams(step, ite, std::move(logits), std::move(endIds))
            , src_cache_indirection{std::move(src_cache_indirection)}
            , max_attention_window{max_attention_window}
            , max_seq_len{max_seq_len}
        {
        }

        // mandatory parameters
        int max_attention_window;
        int max_seq_len;
        tc::Tensor src_cache_indirection; // [local_batch_size, beam_width, max_seq_len]

        // optional parameters
        std::optional<tc::Tensor> embedding_bias; // [vocab_size_padded]
        std::optional<tc::Tensor> input_lengths;  // [local_batch_size * beam_width]
    };

    class BeamSearchOutputParams : public DecodingOutputParams
    {
    public:
        explicit BeamSearchOutputParams(tc::Tensor outputIds, tc::Tensor parentIds, tc::Tensor tgt_cache_indirection)
            : DecodingOutputParams{std::move(outputIds)}
            , parent_ids{std::move(parentIds)}
            , tgt_cache_indirection{std::move(tgt_cache_indirection)}
        {
        }

        tc::Tensor parent_ids;     // [max_seq_len, batch_size * beam_width], necessary in beam search
        tc::Tensor
            tgt_cache_indirection; // [local_batch_size, beam_width, max_seq_len], the k/v cache index for beam search
        std::shared_ptr<kernels::BeamHypotheses>
            beamHypotheses;        // a special structure which maintains some pointers of beam search

        tc::Tensor
            parent_ids_ptr; // [batch_size] int*, each array is [beam_width, max_seq_len], necessary in beam search
    };

    void forward(BeamSearchOutputParams& outputs, ForwardParams const& params);

protected:
    // meta data
    size_t vocab_size_;
    size_t vocab_size_padded_;

    size_t topk_softmax_workspace_size_;
    void* topk_softmax_workspace_ = nullptr;

    std::vector<float> mTemperature;
    std::vector<int> mMinLength;
    std::vector<float> mRepetitionPenalty;
    float* temperature_buf_;
    int* min_lengths_buf_;
    float* repetition_penalty_buf_;
    tensorrt_llm::kernels::RepetitionPenaltyType mRepetitionPenaltyType;

    virtual void invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params) = 0;

    void setupBase(size_t batch_size, SetupParams const& setupParams);

private:
    void allocateBuffer(size_t batch_size);
    void freeBuffer();
};

void update_indir_cache_kernelLauncher(int* tgt_indir_cache, const int* src_indir_cache, const int* beam_ids,
    const tensorrt_llm::kernels::FinishedState* finished, int batch_dim, int beam_width, int max_seq_len, int ite,
    cudaStream_t stream);

} // namespace layers
} // namespace tensorrt_llm
