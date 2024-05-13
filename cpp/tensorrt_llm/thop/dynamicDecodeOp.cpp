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

#include "tensorrt_llm/thop/dynamicDecodeOp.h"

#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/torchAllocator.h"

namespace th = torch;

namespace tr = tensorrt_llm::runtime;
namespace tcc = tensorrt_llm::common::conversion;

namespace torch_ext
{

template <typename T>
FtDynamicDecode<T>::FtDynamicDecode(size_t const max_batch_size, size_t const max_beam_width, size_t const vocab_size,
    size_t const vocab_size_padded, int const tensor_para_size, int const pipeline_para_size)
    : finished_sum_(tr::BufferManager::pinned(
        tr::ITensor::makeShape({static_cast<int32_t>(max_batch_size)}), nvinfer1::DataType::kINT32))
{
    TLLM_CHECK_WITH_INFO(vocab_size_padded % tensor_para_size == 0,
        tensorrt_llm::common::fmtstr(
            "vocab_size (%ld) is not multiple of tensor_para_size (%d).", vocab_size_padded, tensor_para_size));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto allocator = std::make_shared<tensorrt_llm::thop::TorchAllocator>(stream);

    auto const decodingDomain
        = tensorrt_llm::layers::DecoderDomain(max_batch_size, max_beam_width, vocab_size, vocab_size_padded);

    dynamic_decode_layer_ = std::make_shared<tensorrt_llm::layers::DynamicDecodeLayer<T>>(
        tr::DecodingMode::None(), decodingDomain, stream, std::move(allocator));
}

namespace
{

template <typename T>
void safeInsert(th::optional<th::Tensor>& tensor, std::optional<std::vector<T>>& arg)
{
    using value_type = T;
    if (tensor.has_value())
    {
        auto ptr = get_ptr<value_type>(tensor.value());
        auto shape = convert_shape(tensor.value());
        size_t const size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        arg = std::vector<value_type>(ptr, ptr + size);
    }
}

template <typename T>
void safeUpdate(th::optional<th::Tensor>& tensor, std::optional<tc::Tensor>& arg)
{
    if (tensor.has_value())
    {
        arg = convert_tensor<T>(tensor.value());
    }
}

template <typename T>
void safeUpdateScalar(th::optional<th::Tensor>& tensor, std::optional<T>& arg, std::string const& name)
{
    if (tensor.has_value())
    {
        auto accessor = tensor->accessor<T, 1>();
        TLLM_CHECK_WITH_INFO(accessor.size(0) == 1, name + " must be a scalar");
        arg = accessor[0];
    }
}

template <typename T>
void safeUpdatePtr(th::optional<th::Tensor>& tensor, T*& ptr)
{
    if (tensor.has_value())
    {
        ptr = get_ptr<T>(tensor.value());
    }
}

} // namespace

template <typename T>
void FtDynamicDecode<T>::setup(size_t const batch_size, size_t const beam_width,
    th::optional<th::Tensor> runtime_top_k_opt, th::optional<th::Tensor> runtime_top_p_opt,
    th::optional<th::Tensor> temperature_opt, th::optional<th::Tensor> repetition_penalty_opt,
    th::optional<th::Tensor> presence_penalty_opt, th::optional<th::Tensor> frequency_penalty_opt,
    th::optional<th::Tensor> min_length_opt, th::optional<th::Tensor> length_penalty_opt,
    th::optional<th::Tensor> early_stopping_opt, th::optional<th::Tensor> beam_search_diversity_rate_opt,
    th::optional<th::Tensor> random_seed_opt, th::optional<th::Tensor> top_p_decay_opt,
    th::optional<th::Tensor> top_p_min_opt, th::optional<th::Tensor> top_p_reset_ids_opt)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dynamic_decode_layer_->setStream(stream);

    auto setupParams = std::make_shared<tensorrt_llm::layers::DynamicDecodeSetupParams>();
    safeInsert(temperature_opt, setupParams->penaltyParams.temperature);
    safeInsert(repetition_penalty_opt, setupParams->penaltyParams.repetitionPenalty);
    safeInsert(presence_penalty_opt, setupParams->penaltyParams.presencePenalty);
    safeInsert(frequency_penalty_opt, setupParams->penaltyParams.frequencyPenalty);
    safeInsert(min_length_opt, setupParams->penaltyParams.minLength);
    safeInsert(runtime_top_k_opt, setupParams->samplingParams.runtime_top_k);
    safeInsert(runtime_top_p_opt, setupParams->samplingParams.runtime_top_p);
    safeInsert(random_seed_opt, setupParams->randomSeed);
    safeInsert(top_p_decay_opt, setupParams->samplingParams.top_p_decay);
    safeInsert(top_p_min_opt, setupParams->samplingParams.top_p_min);
    safeInsert(top_p_reset_ids_opt, setupParams->samplingParams.top_p_reset_ids);
    safeInsert(beam_search_diversity_rate_opt, setupParams->beamSearchParams.beam_search_diversity_rate);
    safeInsert(length_penalty_opt, setupParams->beamSearchParams.length_penalty);
    safeInsert(early_stopping_opt, setupParams->beamSearchParams.early_stopping);
    // TODO: insert "normalize_log_probs" and "topKMedusaHeads"

    dynamic_decode_layer_->setup(batch_size, beam_width, nullptr, setupParams);
}

template <typename T>
void FtDynamicDecode<T>::forward(th::Tensor const& logits, int const step, int const max_input_length,
    int const max_attention_window, int const sink_token_length, uint64_t const ite, int const local_batch_size,
    th::Tensor end_id, th::optional<th::Tensor> embedding_bias_opt, th::optional<th::Tensor> input_lengths_opt,
    th::optional<th::Tensor> sequence_limit_length_opt, th::optional<th::Tensor> stop_words_list_ptrs_opt,
    th::optional<th::Tensor> stop_words_lens_opt, int32_t const max_stop_words_len,
    th::optional<th::Tensor> bad_words_list_ptrs_opt, th::optional<th::Tensor> bad_words_lens_opt,
    int32_t const max_bad_words_len, th::optional<th::Tensor> no_repeat_ngram_size_opt,
    th::optional<th::Tensor> src_cache_indirection_opt, th::Tensor& output_token_ids, th::Tensor& newTokens,
    th::Tensor& should_stop, th::optional<th::Tensor> finished_input, th::optional<th::Tensor> finished_output,
    th::optional<th::Tensor> sequence_lengths_opt, th::optional<th::Tensor> cum_log_probs_opt,
    th::optional<th::Tensor> output_log_probs_opt, th::optional<th::Tensor> output_log_probs_tiled_opt,
    th::optional<th::Tensor> parent_ids_opt, th::optional<th::Tensor> tgt_cache_indirection_opt,
    th::optional<th::Tensor> beam_hyps_output_ids_cba_opt, th::optional<th::Tensor> beam_hyps_seq_len_cba_opt,
    th::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt, th::optional<th::Tensor> beam_hyps_normed_scores_cba_opt,
    th::optional<th::Tensor> beam_hyps_log_probs_cba_opt, th::optional<th::Tensor> beam_hyps_min_normed_scores_opt,
    th::optional<th::Tensor> beam_hyps_num_beams_opt, th::optional<th::Tensor> beam_hyps_is_done_opt,
    bool const use_beam_hyps)
{
    auto forwardParams = std::make_shared<tensorrt_llm::layers::DynamicDecodeInputParams>(step, static_cast<int>(ite),
        max_input_length, max_attention_window, sink_token_length, local_batch_size, convert_tensor<int>(end_id));

    forwardParams->logits = convert_tensor<float>(logits);

    safeUpdate<T>(embedding_bias_opt, forwardParams->embedding_bias);
    safeUpdate<int>(input_lengths_opt, forwardParams->input_lengths);
    safeUpdate<int>(sequence_limit_length_opt, forwardParams->sequence_limit_length);
    safeUpdate<uint64_t>(stop_words_list_ptrs_opt, forwardParams->stop_words_ptr);
    safeUpdate<int>(stop_words_lens_opt, forwardParams->stop_words_lengths);
    forwardParams->max_stop_words_len = max_stop_words_len;
    safeUpdate<uint64_t>(bad_words_list_ptrs_opt, forwardParams->bad_words_ptr);
    safeUpdate<int>(bad_words_lens_opt, forwardParams->bad_words_lengths);
    forwardParams->max_bad_words_len = max_bad_words_len;
    safeUpdate<int>(no_repeat_ngram_size_opt, forwardParams->no_repeat_ngram_size);
    safeUpdate<int>(src_cache_indirection_opt, forwardParams->src_cache_indirection);

    auto const& output_ids_converted = convert_tensor<int>(output_token_ids);
    auto outputParams = std::make_shared<tensorrt_llm::layers::DynamicDecodeOutputParams>(output_ids_converted);
    outputParams->newTokens = std::move(convert_tensor<int>(newTokens));
    safeUpdate<uint8_t>(finished_input, forwardParams->finished);
    safeUpdate<uint8_t>(finished_output, outputParams->finished);
    safeUpdate<int>(sequence_lengths_opt, outputParams->sequence_length);
    safeUpdate<float>(cum_log_probs_opt, outputParams->cum_log_probs);
    safeUpdate<float>(output_log_probs_opt, outputParams->output_log_probs);
    safeUpdate<float>(output_log_probs_tiled_opt, outputParams->output_log_probs_tiled);
    safeUpdate<int>(parent_ids_opt, outputParams->parent_ids);
    safeUpdate<int>(tgt_cache_indirection_opt, outputParams->tgt_cache_indirection);

    std::int32_t* finished_sum_host = nullptr;
    if (forwardParams->sequence_limit_length && outputParams->finished.has_value())
    {
        // Skip the initialization and later calculation if there is no limit of sequence length or no finished beam
        outputParams->finished_sum = tcc::toTllmTensor(*finished_sum_);
        finished_sum_host = tr::bufferCast<std::int32_t>(*finished_sum_);
        for (int32_t bi = 0; bi < local_batch_size; ++bi)
        {
            finished_sum_host[bi] = 0;
        }
    }

    if (use_beam_hyps)
    {
        // Additional parameters for beam search
        outputParams->beamHypotheses = std::make_unique<tensorrt_llm::kernels::BeamHypotheses>();
        safeUpdatePtr<bool>(beam_hyps_is_done_opt, outputParams->beamHypotheses->batchDones);
        safeUpdatePtr<float>(beam_hyps_cum_log_probs_cba_opt, outputParams->beamHypotheses->cumLogProbsCBA);
        safeUpdatePtr<float>(beam_hyps_log_probs_cba_opt, outputParams->beamHypotheses->logProbsCBA);
        safeUpdatePtr<float>(beam_hyps_min_normed_scores_opt, outputParams->beamHypotheses->minNormedScoresCBA);
        safeUpdatePtr<float>(beam_hyps_normed_scores_cba_opt, outputParams->beamHypotheses->normedScoresCBA);
        safeUpdatePtr<int>(beam_hyps_num_beams_opt, outputParams->beamHypotheses->numBeamsCBA);
        safeUpdatePtr<int>(beam_hyps_output_ids_cba_opt, outputParams->beamHypotheses->outputIdsCBA);
        safeUpdatePtr<int>(beam_hyps_seq_len_cba_opt, outputParams->beamHypotheses->sequenceLengthsCBA);
        // TODO: move the assignment below into beamSearchLayer.cu
        safeUpdatePtr<int32_t const>(input_lengths_opt, outputParams->beamHypotheses->inputLengths);
    }

    dynamic_decode_layer_->forward(outputParams, forwardParams);

    if (finished_sum_host)
    {
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(dynamic_decode_layer_->getStream()));
        int32_t numRealFinished = 0;
        for (int32_t bi = 0; bi < local_batch_size; ++bi)
        {
            numRealFinished += finished_sum_host[bi];
        }
        auto const numToFinish = outputParams->finished->size();
        auto should_stop_accessor = should_stop.accessor<bool, 1>();
        should_stop_accessor[0] = numToFinish == numRealFinished;
    }
}

DynamicDecodeOp::DynamicDecodeOp(int64_t const max_batch_size, int64_t const max_beam_width, int64_t const vocab_size,
    int64_t const vocab_size_padded, int64_t const tensor_para_size, int64_t const pipeline_para_size,
    at::ScalarType const scalar_type)
    : max_batch_size_(static_cast<size_t>(max_batch_size))
    , max_beam_width_(static_cast<size_t>(max_beam_width))
    , vocab_size_(static_cast<size_t>(vocab_size))
    , vocab_size_padded_(static_cast<size_t>(vocab_size_padded))
    , tensor_para_size_(static_cast<int>(tensor_para_size))
    , pipeline_para_size_(static_cast<int>(pipeline_para_size))
    , scalar_type_(scalar_type)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    createInstance();
}

void DynamicDecodeOp::createInstance()
{
    dynamic_decode_.reset();
    switch (scalar_type_)
    {
    case at::ScalarType::Float:
        dynamic_decode_ = std::make_unique<FtDynamicDecode<float>>(
            max_batch_size_, max_beam_width_, vocab_size_, vocab_size_padded_, tensor_para_size_, pipeline_para_size_);
        break;
    case at::ScalarType::Half:
        dynamic_decode_ = std::make_unique<FtDynamicDecode<half>>(
            max_batch_size_, max_beam_width_, vocab_size_, vocab_size_padded_, tensor_para_size_, pipeline_para_size_);
        break;
    default: throw std::runtime_error("Wrong tensor type.");
    }
}

void DynamicDecodeOp::setup(int64_t const batch_size, int64_t const beam_width,
    th::optional<th::Tensor> runtime_top_k_opt, th::optional<th::Tensor> runtime_top_p_opt,
    th::optional<th::Tensor> temperature_opt, th::optional<th::Tensor> repetition_penalty_opt,
    th::optional<th::Tensor> presence_penalty_opt, th::optional<th::Tensor> frequency_penalty_opt,
    th::optional<th::Tensor> min_length_opt, th::optional<th::Tensor> length_penalty_opt,
    th::optional<th::Tensor> early_stopping_opt, th::optional<th::Tensor> beam_search_diversity_rate_opt,
    th::optional<th::Tensor> random_seed_opt, th::optional<th::Tensor> top_p_decay_opt,
    th::optional<th::Tensor> top_p_min_opt, th::optional<th::Tensor> top_p_reset_ids_opt)
{
    // TODO: Revise DynamicDecodeLayer and make the decode arguments consistent.
    // TODO: add parameters "normalize_log_probs" and "topKMedusaHeads"
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_k_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_p_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(temperature_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(repetition_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(presence_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(frequency_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(min_length_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(length_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(early_stopping_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(beam_search_diversity_rate_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(random_seed_opt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(top_p_decay_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_min_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_reset_ids_opt, torch::kInt32);

    dynamic_decode_->setup(static_cast<size_t>(batch_size), static_cast<size_t>(beam_width), runtime_top_k_opt,
        runtime_top_p_opt, temperature_opt, repetition_penalty_opt, presence_penalty_opt, frequency_penalty_opt,
        min_length_opt, length_penalty_opt, early_stopping_opt, beam_search_diversity_rate_opt, random_seed_opt,
        top_p_decay_opt, top_p_min_opt, top_p_reset_ids_opt);
}

th::Tensor DynamicDecodeOp::forward(
    // Inputs  BS: batch_size, BM: beam_width, MSL: max_seq_length, V: vocab_size, VP: vocab_size_padded
    th::Tensor const& logits,                           // [BS, BM, VP], T, variables for input
    int64_t const step,                                 //
    int64_t const max_input_length,                     //
    int64_t const max_attention_window,                 //
    int64_t const sink_token_length,                    //
    int64_t const ite,                                  //
    int64_t const local_batch_size,                     //
    th::Tensor const end_id,                            // [BS*BM], int
    th::optional<th::Tensor> embedding_bias_opt,        // [VP], T
    th::optional<th::Tensor> input_lengths_opt,         // [BS*BM], int, length of input contexts
    th::optional<th::Tensor> sequence_limit_length_opt, // [BS, 1], int
    th::optional<th::Tensor> stop_words_list_ptrs_opt,  // [BS][2, stop_words_length], int64
    th::optional<th::Tensor> stop_words_lens_opt,       // [BS], int
    int64_t const max_stop_words_len,                   //
    th::optional<th::Tensor> bad_words_list_ptrs_opt,   // [BS][2, bad_words_length], int64
    th::optional<th::Tensor> bad_words_lens_opt,        // [BS], int
    int64_t const max_bad_words_len,                    //
    th::optional<th::Tensor> no_repeat_ngram_size_opt,  // [BS], int
    th::optional<th::Tensor> src_cache_indirection_opt, // [local_BS, BM, MSL], int
    // Outputs
    th::Tensor output_token_ids,                              // [BS, BM, MSL], variables for output
    th::Tensor newTokens,                                     // [BS, BM, 1], int
    th::optional<th::Tensor> finished_input,                  // [BS, BM], uint8
    th::optional<th::Tensor> finished_output,                 // [BS, BM], uint8
    th::optional<th::Tensor> sequence_lengths_opt,            // [BS*BM], int, length of the current sequences
    th::optional<th::Tensor> cum_log_probs_opt,               // [BS, BM], float
    th::optional<th::Tensor> output_log_probs_opt,            // [BS, BM, MSL], float
    th::optional<th::Tensor> output_log_probs_tiled_opt,      // [MSL, BS, BM], float, transpose of output_log_probs_opt
    th::optional<th::Tensor> parent_ids_opt,                  // [BS, BM, MSL], int
    th::optional<th::Tensor> tgt_cache_indirection_opt,       // [local_BS, BM, MSL], int
    th::optional<th::Tensor> beam_hyps_output_ids_cba_opt,    // [BS, BM*2, MSL], int
    th::optional<th::Tensor> beam_hyps_seq_len_cba_opt,       // [BS, BM*2], int
    th::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt, // [BS, BM*2], float
    th::optional<th::Tensor> beam_hyps_normed_scores_cba_opt, // [BS, BM*2], float
    th::optional<th::Tensor> beam_hyps_log_probs_cba_opt,     // [BS, BM*2, MSL], float
    th::optional<th::Tensor> beam_hyps_min_normed_scores_opt, // [BS], float
    th::optional<th::Tensor> beam_hyps_num_beams_opt,         // [BS], int
    th::optional<th::Tensor> beam_hyps_is_done_opt,           // [BS], bool
    bool const use_beam_hyps                                  //
)
{
    CHECK_INPUT(logits, scalar_type_);
    TLLM_CHECK_WITH_INFO(logits.dim() == 3,
        "logits is of shape (batch_size, beam_width, vocab_size_padded), but got dim=%d shape=%s", (int) logits.dim(),
        tensorrt_llm::common::vec2str(convert_shape(logits)).c_str());
    TLLM_CHECK_WITH_INFO(static_cast<size_t>(logits.size(2)) == vocab_size_padded_,
        "logits is of shape (batch_size, beam_width, vocab_size(%ld)), but got the last dim=%ld.", vocab_size_padded_,
        static_cast<size_t>(logits.size(2)));
    CHECK_INPUT(end_id, torch::kInt32);
    CHECK_OPTIONAL_INPUT(embedding_bias_opt, scalar_type_);
    CHECK_OPTIONAL_INPUT(input_lengths_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(sequence_limit_length_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(stop_words_list_ptrs_opt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(stop_words_lens_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(bad_words_list_ptrs_opt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(bad_words_lens_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(no_repeat_ngram_size_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(src_cache_indirection_opt, torch::kInt32);
    CHECK_INPUT(output_token_ids, torch::kInt32);
    CHECK_INPUT(newTokens, torch::kInt32);
    CHECK_OPTIONAL_INPUT(finished_input, torch::kUInt8);
    CHECK_OPTIONAL_INPUT(finished_output, torch::kUInt8);
    CHECK_OPTIONAL_INPUT(sequence_lengths_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(cum_log_probs_opt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(output_log_probs_opt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(output_log_probs_tiled_opt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(parent_ids_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(tgt_cache_indirection_opt, torch::kInt32);

    th::Tensor should_stop = torch::zeros({1}, torch::dtype(torch::kBool).requires_grad(false));

    dynamic_decode_->forward(
        // Inputs
        logits, static_cast<int>(step), static_cast<int>(max_input_length), static_cast<int>(max_attention_window),
        static_cast<int>(sink_token_length), static_cast<uint32_t>(ite), static_cast<int>(local_batch_size), end_id,
        embedding_bias_opt, input_lengths_opt, sequence_limit_length_opt, stop_words_list_ptrs_opt, stop_words_lens_opt,
        static_cast<int32_t>(max_stop_words_len), bad_words_list_ptrs_opt, bad_words_lens_opt,
        static_cast<int32_t>(max_bad_words_len), no_repeat_ngram_size_opt, src_cache_indirection_opt,
        // Outputs
        output_token_ids, newTokens, should_stop, finished_input, finished_output, sequence_lengths_opt,
        cum_log_probs_opt, output_log_probs_opt, output_log_probs_tiled_opt, parent_ids_opt, tgt_cache_indirection_opt,
        beam_hyps_output_ids_cba_opt, beam_hyps_seq_len_cba_opt, beam_hyps_cum_log_probs_cba_opt,
        beam_hyps_normed_scores_cba_opt, beam_hyps_log_probs_cba_opt, beam_hyps_min_normed_scores_opt,
        beam_hyps_num_beams_opt, beam_hyps_is_done_opt, use_beam_hyps);

    return should_stop;
}

} // namespace torch_ext

static auto trtllmGptContextDecoderTHS
    = torch::jit::class_<torch_ext::DynamicDecodeOp>("trtllm", "DynamicDecodeOp")
          .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, at::ScalarType>())
          .def("setup", &torch_ext::DynamicDecodeOp::setup)
          .def("forward", &torch_ext::DynamicDecodeOp::forward);
