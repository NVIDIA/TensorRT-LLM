/*
 * Copyright (c) 2022-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <optional>
#include <string>
#include <unordered_map>
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
class DynamicDecodeLayer : public BaseLayer
{
public:
    DynamicDecodeLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
        std::shared_ptr<tc::IAllocator> allocator, bool is_free_buffer_after_forward, cudaDeviceProp* cuda_device_prop);

    ~DynamicDecodeLayer() override;
    DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer);

    class SetupParams
    {
    public:
        std::optional<std::vector<float>> temperature;        // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> repetition_penalty; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> presence_penalty;   // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> frequency_penalty;  // [1] or [batch_size] on cpu
        std::optional<std::vector<std::int32_t>> min_length;  // [1] or [batch_size] on cpu

        // baseSamplingLayer
        std::optional<std::vector<std::uint32_t>> runtime_top_k; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> runtime_top_p;         // [1] or [batch_size] on cpu
        std::optional<std::vector<uint64_t>> randomSeed;         // [1] or [batch_size] on cpu

        // topPSamplingLayer
        std::optional<std::vector<float>> top_p_decay;            // [batch_size], must between [0, 1]
        std::optional<std::vector<float>> top_p_min;              // [batch_size], must between [0, 1]
        std::optional<std::vector<std::int32_t>> top_p_reset_ids; // [batch_size]

        // omlineBeamSearchLayer
        std::optional<std::vector<float>> beam_search_diversity_rate;
        std::optional<std::vector<float>> length_penalty;

        std::optional<bool> normalize_log_probs;
    };

    void setup(size_t batch_size, size_t beam_width, SetupParams const& setupParams);

    class ForwardParams
    {
    public:
        ForwardParams(int step, int ite, int maxInputLength, int maxAttentionWindow, int sinkTokenLength,
            int localBatchSize, tc::Tensor logits, tc::Tensor endIds)
            : step{step}
            , ite{ite}
            , max_input_length{maxInputLength}
            , max_attention_window{maxAttentionWindow}
            , sink_token_length{sinkTokenLength}
            , local_batch_size{localBatchSize}
            , logits{std::move(logits)}
            , end_ids{std::move(endIds)}
        {
        }

        // mandatory parameters
        int step;
        int ite;
        int max_input_length;
        int max_attention_window;
        int sink_token_length;
        int local_batch_size;
        tc::Tensor logits;  // [batch_size, beam_width, vocab_size_padded], on gpu
        tc::Tensor end_ids; // [batch_size], on gpu

        // optional parameters
        std::optional<tc::Tensor> finished;              // [batch_size * beam_width], optional
        std::optional<tc::Tensor> src_cache_indirection; // [local_batch_size, beam_width, max_seq_len] - the k/v cache
                                                         // index for beam search, mandatory for beam search, on gpu
        std::optional<tc::Tensor> sequence_limit_length; // [batch_size], on gpu
        std::optional<tc::Tensor> embedding_bias;        // [vocab_size_padded], on gpu
        std::optional<tc::Tensor> input_lengths;         // [batch_size, beam_width], on gpu
        std::optional<tc::Tensor> bad_words_list;  // [2, bad_words_length] or [batch_size, 2, bad_words_length], on gpu
        std::optional<tc::Tensor> stop_words_list; // [batch_size, 2, stop_words_length], on gpu
        std::optional<tc::Tensor> no_repeat_ngram_size; // [batch_size], optional
    };

    class OutputParams
    {
    public:
        explicit OutputParams(tc::Tensor outputIds)
            : output_ids{std::move(outputIds)}
        {
        }

        // mandatory parameters
        tc::Tensor output_ids; // [batch_size, beam_width. max_seq_len]
        tc::Tensor newTokens;  // [batch_size, beam_width]
        // optional parameters
        std::optional<tc::Tensor> finished;        // [batch_size * beam_width], optional
        std::optional<tc::Tensor> finished_sum;    // [1], optional, in pinned host memory
        std::optional<tc::Tensor> cum_log_probs;   // [batch_size * beam_width], necessary in beam search
        std::optional<tc::Tensor> parent_ids;      // [max_seq_len, batch_size * beam_width], necessary in beam search
        std::optional<tc::Tensor> sequence_length; // [batch_size * beam_width], optional
        std::optional<tc::Tensor>
            output_log_probs_tiled; // [request_output_length, batch_size, beam_width], must be float*, optional
        std::optional<tc::Tensor>
            output_log_probs;       // [batchSize, beam_width, request_ouptut_length], must be float*, optional
        std::optional<tc::Tensor>
            tgt_cache_indirection;  // [local_batch_size, beam_width, max_seq_len], the k/v cache index for beam search
        std::shared_ptr<kernels::BeamHypotheses>
            beamHypotheses;         // a special structure which maintains some pointers of beam search

        tc::Tensor output_ids_ptr;  // [batch_size] int* (2-d array), each int* has [beam_width, max_seq_len]
        tc::Tensor parent_ids_ptr;  // [batch_size] int* (2-d array), each int* has [beam_width, max_seq_len]
    };

    void forward(OutputParams& outputs, ForwardParams const& params);
    void allocateBuffer(size_t batch_size, size_t beam_width, size_t max_seq_len);
    void freeBuffer();

private:
    void initialize();

    std::unique_ptr<OnlineBeamSearchLayer<T>> mOnlineBeamsearchDecode;
    std::unique_ptr<TopKSamplingLayer<T>> mTopKDecode;
    std::unique_ptr<TopPSamplingLayer<T>> mTopPDecode;

    size_t vocab_size_;
    size_t vocab_size_padded_;
    cudaDeviceProp* cuda_device_prop_;
    int* zero_parent_ids = nullptr;
    int* top_k_workspace = nullptr;
    int* top_p_workspace = nullptr;
    int* beam_search_workspace_0 = nullptr;
    int* beam_search_workspace_1 = nullptr;
    runtime::IBuffer::SharedPtr mIdsPtrHost;

    bool has_diff_runtime_args_ = false;
    int* h_pinned_finished_sum_ = nullptr;
};

} // namespace layers
} // namespace tensorrt_llm
