/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;

namespace torch_ext
{

class IFtDynamicDecode
{
public:
    virtual void setup(size_t batch_size, size_t beam_width, th::optional<th::Tensor> runtime_top_k_opt,
        th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
        th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
        th::optional<th::Tensor> frequency_penalty_opt, th::optional<th::Tensor> min_length_opt,
        th::optional<th::Tensor> length_penalty_opt, th::optional<th::Tensor> beam_search_diversity_rate_opt,
        th::optional<th::Tensor> random_seed_opt, th::optional<th::Tensor> top_p_decay_opt,
        th::optional<th::Tensor> top_p_min_opt, th::optional<th::Tensor> top_p_reset_ids_opt)
        = 0;

    virtual void forward(th::Tensor& logits, // (batch_size, beam_width, hidden_size)
        int step, int max_input_length, int max_attention_window, int sink_token_length, uint64_t ite,
        int local_batch_size, th::Tensor end_id, th::optional<th::Tensor> embedding_bias_opt,
        th::optional<th::Tensor> input_lengths_opt, th::optional<th::Tensor> sequence_limit_length_opt,
        th::optional<th::Tensor> stop_words_list_opt, th::optional<th::Tensor> bad_words_list_opt,
        th::optional<th::Tensor> no_repeat_ngram_size_opt, th::optional<th::Tensor> src_cache_indirection_opt,
        // Outputs
        th::Tensor& output_token_ids, th::Tensor& newTokens, th::Tensor& should_stop,
        th::optional<th::Tensor> finished_input, th::optional<th::Tensor> finished_output,
        th::optional<th::Tensor> sequence_lengths_opt, th::optional<th::Tensor> cum_log_probs_opt,
        th::optional<th::Tensor> output_log_probs_opt, th::optional<th::Tensor> parent_ids_opt,
        th::optional<th::Tensor> tgt_cache_indirection_opt, th::optional<th::Tensor> beam_hyps_output_ids_tgt_opt,
        th::optional<th::Tensor> beam_hyps_sequence_lengths_tgt_opt,
        th::optional<th::Tensor> beam_hyps_cum_log_probs_opt, th::optional<th::Tensor> beam_hyps_normed_scores_opt,
        th::optional<th::Tensor> beam_hyps_log_probs_opt, th::optional<th::Tensor> beam_hyps_min_normed_scores_opt,
        th::optional<th::Tensor> beam_hyps_num_beams_opt, th::optional<th::Tensor> beam_hyps_is_done_opt,
        bool use_beam_hyps)
        = 0;
};

template <typename T>
class FtDynamicDecode : public IFtDynamicDecode
{
public:
    using SetupParams = typename tensorrt_llm::layers::DynamicDecodeLayer<T>::SetupParams;

    FtDynamicDecode(const size_t vocab_size, const size_t vocab_size_padded, const int tensor_para_size,
        const int pipeline_para_size);

    void setup(size_t batch_size, size_t beam_width, th::optional<th::Tensor> runtime_top_k_opt,
        th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
        th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
        th::optional<th::Tensor> frequency_penalty_opt, th::optional<th::Tensor> min_length_opt,
        th::optional<th::Tensor> length_penalty_opt, th::optional<th::Tensor> beam_search_diversity_rate_opt,
        th::optional<th::Tensor> random_seed_opt, th::optional<th::Tensor> top_p_decay_opt,
        th::optional<th::Tensor> top_p_min_opt, th::optional<th::Tensor> top_p_reset_ids_opt) override;

    void forward(th::Tensor& logits, // (batch_size, beam_width, hidden_size)
        int step, int max_input_length, int max_attention_window, int sink_token_length, uint64_t ite,
        int local_batch_size, th::Tensor end_id, th::optional<th::Tensor> embedding_bias_opt,
        th::optional<th::Tensor> input_lengths_opt, th::optional<th::Tensor> sequence_limit_length_opt,
        th::optional<th::Tensor> stop_words_list_opt, th::optional<th::Tensor> bad_words_list_opt,
        th::optional<th::Tensor> no_repeat_ngram_size_opt, th::optional<th::Tensor> src_cache_indirection_opt,
        // Outputs
        th::Tensor& output_token_ids, th::Tensor& newTokens, th::Tensor& should_stop,
        th::optional<th::Tensor> finished_input, th::optional<th::Tensor> finished_output,
        th::optional<th::Tensor> sequence_lengths_opt, th::optional<th::Tensor> cum_log_probs_opt,
        th::optional<th::Tensor> output_log_probs_opt, th::optional<th::Tensor> parent_ids_opt,
        th::optional<th::Tensor> tgt_cache_indirection_opt, th::optional<th::Tensor> beam_hyps_output_ids_tgt_opt,
        th::optional<th::Tensor> beam_hyps_sequence_lengths_tgt_opt,
        th::optional<th::Tensor> beam_hyps_cum_log_probs_opt, th::optional<th::Tensor> beam_hyps_normed_scores_opt,
        th::optional<th::Tensor> beam_hyps_log_probs_opt, th::optional<th::Tensor> beam_hyps_min_normed_scores_opt,
        th::optional<th::Tensor> beam_hyps_num_beams_opt, th::optional<th::Tensor> beam_hyps_is_done_opt,
        bool use_beam_hyps) override;

private:
    const size_t vocab_size_;
    const size_t vocab_size_padded_;

    cudaDeviceProp prop_;

    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> dynamic_decode_layer_;
    tensorrt_llm::runtime::ITensor::SharedPtr finished_sum_;
};

class DynamicDecodeOp : public th::jit::CustomClassHolder
{
public:
    DynamicDecodeOp(const int64_t vocab_size, const int64_t vocab_size_padded, const int64_t tensor_para_size,
        const int64_t pipeline_para_size, at::ScalarType scalar_type);

    void setup(int64_t batch_size, int64_t beam_width, th::optional<th::Tensor> runtime_top_k_opt,
        th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
        th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
        th::optional<th::Tensor> frequency_penalty_opt, th::optional<th::Tensor> min_length_opt,
        th::optional<th::Tensor> length_penalty_opt, th::optional<th::Tensor> beam_search_diversity_rate_opt,
        th::optional<th::Tensor> random_seed_opt, th::optional<th::Tensor> top_p_decay_opt,
        th::optional<th::Tensor> top_p_min_opt, th::optional<th::Tensor> top_p_reset_ids_opt);

    th::Tensor forward(th::Tensor logits, // (batch_size, beam_width, vocab_size)
        int64_t step, int64_t max_input_length, int64_t max_attention_window, int64_t sink_token_length, int64_t ite,
        int64_t local_batch_size, th::Tensor end_id, th::optional<th::Tensor> embedding_bias_opt,
        th::optional<th::Tensor> input_lengths_opt, // length of input contexts.
        th::optional<th::Tensor> sequence_limit_length_opt, th::optional<th::Tensor> stop_words_list_opt,
        th::optional<th::Tensor> bad_words_list_opt, th::optional<th::Tensor> no_repeat_ngram_size_opt,
        th::optional<th::Tensor> src_cache_indirection_opt,
        // output buffers.
        th::Tensor output_token_ids, th::Tensor newTokens, th::optional<th::Tensor> finished_input,
        th::optional<th::Tensor> finished_output,
        th::optional<th::Tensor> sequence_lengths_opt, // length of the current sequences.
        th::optional<th::Tensor> cum_log_probs_opt, th::optional<th::Tensor> output_log_probs_opt,
        th::optional<th::Tensor> parent_ids_opt, th::optional<th::Tensor> tgt_cache_indirection_opt,
        th::optional<th::Tensor> beam_hyps_output_ids_tgt_opt,
        th::optional<th::Tensor> beam_hyps_sequence_lengths_tgt_opt,
        th::optional<th::Tensor> beam_hyps_cum_log_probs_opt, th::optional<th::Tensor> beam_hyps_normed_scores_opt,
        th::optional<th::Tensor> beam_hyps_log_probs_opt, th::optional<th::Tensor> beam_hyps_min_normed_scores_opt,
        th::optional<th::Tensor> beam_hyps_num_beams_opt, th::optional<th::Tensor> beam_hyps_is_done_opt,
        bool use_beam_hyps);

private:
    size_t const vocab_size_;
    size_t const vocab_size_padded_;
    int const tensor_para_size_;
    int const pipeline_para_size_;

    // Data type of expected input logits.
    at::ScalarType scalar_type_;
    // FT Dynamic decode layer wrapper instance.
    std::unique_ptr<IFtDynamicDecode> dynamic_decode_;

    void createInstance();
};

} // namespace torch_ext
