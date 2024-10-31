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
    virtual void setup(size_t const batch_size, size_t const beam_width, std::optional<th::Tensor> runtime_top_k_opt,
        std::optional<th::Tensor> runtime_top_p_opt, std::optional<th::Tensor> temperature_opt,
        std::optional<th::Tensor> repetition_penalty_opt, std::optional<th::Tensor> presence_penalty_opt,
        std::optional<th::Tensor> frequency_penalty_opt, std::optional<th::Tensor> min_length_opt,
        std::optional<th::Tensor> length_penalty_opt, std::optional<th::Tensor> early_stopping_opt,
        std::optional<th::Tensor> beam_search_diversity_rate_opt, std::optional<th::Tensor> random_seed_opt,
        std::optional<th::Tensor> top_p_decay_opt, std::optional<th::Tensor> top_p_min_opt,
        std::optional<th::Tensor> top_p_reset_ids_opt, std::optional<th::Tensor> no_repeat_ngram_size_opt,
        bool output_log_probs, bool cum_log_probs)
        = 0;

    virtual void forward(th::Tensor const& logits, int const step, int const max_input_length,
        int const max_attention_window, int const sink_token_length, uint64_t const ite, int const local_batch_size,
        th::Tensor end_id, std::optional<th::Tensor> embedding_bias_opt, std::optional<th::Tensor> input_lengths_opt,
        std::optional<th::Tensor> sequence_limit_length_opt, std::optional<th::Tensor> stop_words_list_ptrs_opt,
        std::optional<th::Tensor> stop_words_lens_opt, int32_t const max_stop_words_len,
        std::optional<th::Tensor> bad_words_list_ptrs_opt, std::optional<th::Tensor> bad_words_lens_opt,
        int32_t const max_bad_words_len, std::optional<th::Tensor> src_cache_indirection_opt,
        th::Tensor& output_token_ids, th::Tensor& newTokens, th::Tensor& should_stop,
        std::optional<th::Tensor> finished_input, std::optional<th::Tensor> finished_output,
        std::optional<th::Tensor> sequence_lengths_opt, std::optional<th::Tensor> cum_log_probs_opt,
        std::optional<th::Tensor> output_log_probs_opt, std::optional<th::Tensor> output_log_probs_tiled_opt,
        std::optional<th::Tensor> parent_ids_opt, std::optional<th::Tensor> tgt_cache_indirection_opt,
        std::optional<th::Tensor> beam_hyps_output_ids_cba_opt, std::optional<th::Tensor> beam_hyps_seq_len_cba_opt,
        std::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt,
        std::optional<th::Tensor> beam_hyps_normed_scores_cba_opt,
        std::optional<th::Tensor> beam_hyps_log_probs_cba_opt,
        std::optional<th::Tensor> beam_hyps_min_normed_scores_opt, std::optional<th::Tensor> beam_hyps_num_beams_opt,
        std::optional<th::Tensor> beam_hyps_is_done_opt, bool const use_beam_hyps)
        = 0;
};

template <typename T>
class FtDynamicDecode : public IFtDynamicDecode
{
public:
    FtDynamicDecode(size_t const max_batch_size, size_t const max_beam_width, size_t const vocab_size,
        size_t const vocab_size_padded, int const tensor_para_size, int const pipeline_para_size);

    void setup(size_t const batch_size, size_t const beam_width, std::optional<th::Tensor> runtime_top_k_opt,
        std::optional<th::Tensor> runtime_top_p_opt, std::optional<th::Tensor> temperature_opt,
        std::optional<th::Tensor> repetition_penalty_opt, std::optional<th::Tensor> presence_penalty_opt,
        std::optional<th::Tensor> frequency_penalty_opt, std::optional<th::Tensor> min_length_opt,
        std::optional<th::Tensor> length_penalty_opt, std::optional<th::Tensor> early_stopping_opt,
        std::optional<th::Tensor> beam_search_diversity_rate_opt, std::optional<th::Tensor> random_seed_opt,
        std::optional<th::Tensor> top_p_decay_opt, std::optional<th::Tensor> top_p_min_opt,
        std::optional<th::Tensor> top_p_reset_ids_opt, std::optional<th::Tensor> no_repeat_ngram_size_opt,
        bool output_log_probs, bool cum_log_probs) override;

    void forward(th::Tensor const& logits, int const step, int const max_input_length, int const max_attention_window,
        int const sink_token_length, uint64_t const ite, int const local_batch_size, th::Tensor end_id,
        std::optional<th::Tensor> embedding_bias_opt, std::optional<th::Tensor> input_lengths_opt,
        std::optional<th::Tensor> sequence_limit_length_opt, std::optional<th::Tensor> stop_words_list_ptrs_opt,
        std::optional<th::Tensor> stop_words_lens_opt, int32_t const max_stop_words_len,
        std::optional<th::Tensor> bad_words_list_ptrs_opt, std::optional<th::Tensor> bad_words_lens_opt,
        int32_t const max_bad_words_len, std::optional<th::Tensor> src_cache_indirection_opt,
        th::Tensor& output_token_ids, th::Tensor& newTokens, th::Tensor& should_stop,
        std::optional<th::Tensor> finished_input, std::optional<th::Tensor> finished_output,
        std::optional<th::Tensor> sequence_lengths_opt, std::optional<th::Tensor> cum_log_probs_opt,
        std::optional<th::Tensor> output_log_probs_opt, std::optional<th::Tensor> output_log_probs_tiled_opt,
        std::optional<th::Tensor> parent_ids_opt, std::optional<th::Tensor> tgt_cache_indirection_opt,
        std::optional<th::Tensor> beam_hyps_output_ids_cba_opt, std::optional<th::Tensor> beam_hyps_seq_len_cba_opt,
        std::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt,
        std::optional<th::Tensor> beam_hyps_normed_scores_cba_opt,
        std::optional<th::Tensor> beam_hyps_log_probs_cba_opt,
        std::optional<th::Tensor> beam_hyps_min_normed_scores_opt, std::optional<th::Tensor> beam_hyps_num_beams_opt,
        std::optional<th::Tensor> beam_hyps_is_done_opt, bool const use_beam_hyps) override;

private:
    tensorrt_llm::runtime::ITensor::SharedPtr mFinishedSum; // [batch_size] pinned
    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
    std::shared_ptr<tensorrt_llm::runtime::DecodingLayerWorkspace> mDecodingWorkspace;
    std::optional<size_t> mBeamWidth;
    tensorrt_llm::runtime::ITensor::SharedConstPtr mBatchSlots;
};

class DynamicDecodeOp : public th::jit::CustomClassHolder
{
public:
    DynamicDecodeOp(int64_t const max_batch_size, int64_t const max_beam_width, int64_t const vocab_size,
        int64_t const vocab_size_padded, int64_t const tensor_para_size, int64_t const pipeline_para_size,
        at::ScalarType const scalar_type);

    void setup(int64_t const batch_size, int64_t const beam_width, std::optional<th::Tensor> runtime_top_k_opt,
        std::optional<th::Tensor> runtime_top_p_opt, std::optional<th::Tensor> temperature_opt,
        std::optional<th::Tensor> repetition_penalty_opt, std::optional<th::Tensor> presence_penalty_opt,
        std::optional<th::Tensor> frequency_penalty_opt, std::optional<th::Tensor> min_length_opt,
        std::optional<th::Tensor> length_penalty_opt, std::optional<th::Tensor> early_stopping_opt,
        std::optional<th::Tensor> beam_search_diversity_rate_opt, std::optional<th::Tensor> random_seed_opt,
        std::optional<th::Tensor> top_p_decay_opt, std::optional<th::Tensor> top_p_min_opt,
        std::optional<th::Tensor> top_p_reset_ids_opt, std::optional<th::Tensor> no_repeat_ngram_size_opt,
        bool output_log_probs, bool cum_log_probs);

    th::Tensor forward(th::Tensor const& logits, int64_t const step, int64_t const max_input_length,
        int64_t const max_attention_window, int64_t const sink_token_length, int64_t const ite,
        int64_t const local_batch_size, th::Tensor end_id, std::optional<th::Tensor> embedding_bias_opt,
        std::optional<th::Tensor> input_lengths_opt, std::optional<th::Tensor> sequence_limit_length_opt,
        std::optional<th::Tensor> stop_words_list_ptrs_opt, std::optional<th::Tensor> stop_words_lens_opt,
        int64_t const max_stop_words_len, std::optional<th::Tensor> bad_words_list_ptrs_opt,
        std::optional<th::Tensor> bad_words_lens_opt, int64_t const max_bad_words_len,
        std::optional<th::Tensor> src_cache_indirection_opt, th::Tensor output_token_ids, th::Tensor newTokens,
        std::optional<th::Tensor> finished_input, std::optional<th::Tensor> finished_output,
        std::optional<th::Tensor> sequence_lengths_opt, std::optional<th::Tensor> cum_log_probs_opt,
        std::optional<th::Tensor> output_log_probs_opt, std::optional<th::Tensor> output_log_probs_tiled_opt,
        std::optional<th::Tensor> parent_ids_opt, std::optional<th::Tensor> tgt_cache_indirection_opt,
        std::optional<th::Tensor> beam_hyps_output_ids_cba_opt, std::optional<th::Tensor> beam_hyps_seq_len_cba_opt,
        std::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt,
        std::optional<th::Tensor> beam_hyps_normed_scores_cba_opt,
        std::optional<th::Tensor> beam_hyps_log_probs_cba_opt,
        std::optional<th::Tensor> beam_hyps_min_normed_scores_opt, std::optional<th::Tensor> beam_hyps_num_beams_opt,
        std::optional<th::Tensor> beam_hyps_is_done_opt, bool const use_beam_hyps);

private:
    // Members initialized in constructor and used in call of createInstance()
    size_t const maxBatchSize_;
    size_t const maxBeamWidth_;
    size_t const vocabSize_;
    size_t const vocabSizePadded_;
    int const tensorParaSize_;
    int const pipelineParaSize_;
    at::ScalarType const scalarType_;                 // Data type of expected input logits
    std::unique_ptr<IFtDynamicDecode> dynamicDecode_; // FT Dynamic decode layer wrapper instance

    void createInstance();
};

} // namespace torch_ext
