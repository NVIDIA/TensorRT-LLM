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

#pragma once

#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include <tensorrt_llm/common/tensor.h>
#include <tensorrt_llm/runtime/common.h>

#include <optional>
#include <vector>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::layers
{

class DecoderDomain
{
public:
    DecoderDomain(runtime::SizeType maxBatchSize, runtime::SizeType maxBeamWidth, runtime::SizeType vocabSize,
        runtime::SizeType vocabSizePadded, std::optional<runtime::SizeType> maxTokensPerStep = std::nullopt,
        std::optional<runtime::SizeType> maxNumMedusaHeads = std::nullopt)
        : mMaxBatchSize(maxBatchSize)
        , mMaxBeamWidth(maxBeamWidth)
        , mVocabSize(vocabSize)
        , mVocabSizePadded(vocabSizePadded)
        , mMaxTokensPerStep(maxTokensPerStep.value_or(1))
        , mMaxNumMedusaHeads(maxNumMedusaHeads.value_or(0))
    {
    }

    [[nodiscard]] runtime::SizeType getMaxBatchSize() const
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] runtime::SizeType getMaxBeamWidth() const
    {
        return mMaxBeamWidth;
    }

    [[nodiscard]] runtime::SizeType getVocabSize() const
    {
        return mVocabSize;
    }

    [[nodiscard]] runtime::SizeType getVocabSizePadded() const
    {
        return mVocabSizePadded;
    }

    [[nodiscard]] runtime::SizeType getMaxTokensPerStep() const
    {
        return mMaxTokensPerStep;
    }

    [[nodiscard]] runtime::SizeType getMaxNumMedusaHeads() const
    {
        return mMaxNumMedusaHeads;
    }

private:
    runtime::SizeType mMaxBatchSize;
    runtime::SizeType mMaxBeamWidth;
    runtime::SizeType mVocabSize;
    runtime::SizeType mVocabSizePadded;
    runtime::SizeType mMaxTokensPerStep;
    runtime::SizeType mMaxNumMedusaHeads;
};

class BaseSetupParams
{
public:
    virtual ~BaseSetupParams() {}
};

class DynamicDecodeSetupParams : public BaseSetupParams
{
public:
    // Penalty layer
    struct PenaltyParams
    {
        std::optional<std::vector<float>> temperature;             // [1] or [batch_size] on cpu
        std::optional<std::vector<runtime::SizeType32>> minLength; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> repetitionPenalty;       // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> presencePenalty;         // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> frequencyPenalty;        // [1] or [batch_size] on cpu
    };

    struct SamplingParams
    {
        // baseSamplingLayer
        std::optional<std::vector<runtime::SizeType>> runtime_top_k; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> runtime_top_p;             // [1] or [batch_size] on cpu

        // topPSamplingLayer
        std::optional<std::vector<float>> top_p_decay;                    // [batch_size], must between [0, 1]
        std::optional<std::vector<float>> top_p_min;                      // [batch_size], must between [0, 1]
        std::optional<std::vector<runtime::TokenIdType>> top_p_reset_ids; // [batch_size]
        std::optional<bool> normalize_log_probs;
    };

    struct BeamSearchParams
    {
        // BeamSearchLayer
        std::optional<std::vector<float>> beam_search_diversity_rate; // [batch_size] on cpu
        std::optional<std::vector<float>> length_penalty;             // [batch_size] on cpu
        std::optional<std::vector<int>> early_stopping;               // [batch_size] on cpu
    };

    struct MedusaParams
    {
        // Medusa params
        std::optional<std::vector<std::vector<runtime::SizeType>>> topKMedusaHeads; // [batchSize, maxMedusaHeads]
    };

    std::optional<std::vector<uint64_t>> randomSeed; // [1] or [batch_size] on cpu

    PenaltyParams penaltyParams;

    SamplingParams samplingParams;

    BeamSearchParams beamSearchParams;

    MedusaParams medusaParams;
};

class BaseInputParams
{
public:
    explicit BaseInputParams(runtime::SizeType32 step, runtime::SizeType32 ite, tc::Tensor endIds)
        : step{step}
        , ite{ite}
        , end_ids{std::move(endIds)}
    {
    }

    virtual ~BaseInputParams() {}

    // mandatory parameters
    runtime::SizeType32 step;
    runtime::SizeType32 ite;
    tc::Tensor end_ids;                    // [local_batch_size]
    std::optional<tc::Tensor> batch_slots; // [local_batch_size], on pinned memory
    std::optional<tc::Tensor> finished;    // [batch_size * beam_width]
};

class DynamicDecodeInputParams : public BaseInputParams
{
public:
    DynamicDecodeInputParams(runtime::SizeType32 step, runtime::SizeType32 ite, runtime::SizeType maxInputLength,
        runtime::SizeType maxAttentionWindow, runtime::SizeType sinkTokenLength, runtime::SizeType localBatchSize,
        tc::Tensor endIds)
        : BaseInputParams(step, ite, std::move(endIds))
        , max_input_length{maxInputLength}
        , max_attention_window{maxAttentionWindow}
        , sink_token_length{sinkTokenLength}
        , local_batch_size{localBatchSize}
        , max_stop_words_len{0}
        , max_bad_words_len{0}
    {
    }

    // mandatory parameters
    runtime::SizeType max_input_length;
    runtime::SizeType max_attention_window;
    runtime::SizeType sink_token_length;
    runtime::SizeType local_batch_size;
    runtime::SizeType max_stop_words_len;
    runtime::SizeType max_bad_words_len;

    // One of these two fields has to be set
    // DynamicDecodeLayer::forward checks for it
    // Need both of these fields to support legacy code during transition period to the batched decoder
    std::optional<tc::Tensor> logits;                  // [maxBatchSize, beamWidth, vocabSizePadded]
    std::optional<std::vector<tc::Tensor>> logits_vec; // [batch_size], on gpu

    // optional parameters
    std::optional<tc::Tensor> src_cache_indirection; // [local_batch_size, beam_width, max_seq_len] - the k/v cache
                                                     // index for beam search, mandatory for beam search, on gpu
    std::optional<tc::Tensor> sequence_limit_length; // [batch_size], on gpu
    std::optional<tc::Tensor> embedding_bias;        // [vocab_size_padded], on gpu
    std::optional<tc::Tensor> input_lengths;         // [batch_size, beam_width], on gpu
    std::optional<tc::Tensor> bad_words_ptr;         // [batch_size][2, bad_words_length], on gpu
    std::optional<tc::Tensor> bad_words_lengths;     // [batch_size], on gpu
    std::optional<tc::Tensor> stop_words_ptr;        // [batch_size][2, stop_words_length], on gpu
    std::optional<tc::Tensor> stop_words_lengths;    // [batch_size], on gpu
    std::optional<tc::Tensor> no_repeat_ngram_size;  // [batch_size], on gpu

    // Medusa inputs
    class MedusaInputs
    {
    public:
        tc::Tensor medusaCurTokensPerStep;    // [batch_size], optional, on gpu
        tc::Tensor medusaTargetTokensPerStep; // [batch_size], optional, on gpu
        tc::Tensor medusaPaths;               // [batch_size, max_tokens_per_step, max_num_heads + 1], optional, on gpu
        tc::Tensor medusaTreeIds;             // [batch_size, max_tokens_per_step], optional, on gpu
        std::vector<std::vector<tc::Tensor>>
            medusaLogits; // [max_batch_size][max_num_heads][tokens_per_step, vocab_size], optional, on gpu
    };

    std::optional<MedusaInputs> medusaInputs;
};

class BaseOutputParams
{
public:
    explicit BaseOutputParams(tc::Tensor outputIds)
        : output_ids{std::move(outputIds)}
    {
    }

    virtual ~BaseOutputParams() {}

    // mandatory parameters
    tc::Tensor output_ids; // [max_seq_len, batch_size]

    // optional parameters
    std::optional<tc::Tensor> finished;        // [batch_size * beam_width], optional
    std::optional<tc::Tensor> sequence_length; // [batch_size * beam_width], optional
    std::optional<tc::Tensor> cum_log_probs;   // [batch_size * beam_width], necessary in beam search
    std::optional<tc::Tensor>
        output_log_probs;                 // [request_ouptut_length, batch_size * beam_width], must be float*, optional
    std::optional<tc::Tensor> parent_ids; // [max_seq_len, batch_size * beam_width], necessary in beam search

    tc::Tensor output_ids_ptr;            // [batch_size] int* (2-d array), each int* has [beam_width, max_seq_len]

    // Medusa outputs
    class MedusaOutputs
    {
    public:
        tc::Tensor nextDraftTokens;       // [batch_size, max_tokens_per_step], draft tokens predicted by Medusa heads
        tc::Tensor acceptedLengths;       // [batch_size], lengths of the accepted draft tokens + 1
        tc::Tensor acceptedLengthsCumSum; // [batch_size + 1]
        tc::Tensor pathsOffsets;          // [batch_size * max_medusa_heads]
    };

    std::optional<MedusaOutputs> medusaOutputs;
};

class DynamicDecodeOutputParams : public BaseOutputParams
{
public:
    explicit DynamicDecodeOutputParams(tc::Tensor outputIds)
        : BaseOutputParams{std::move(outputIds)}
    {
    }

    // mandatory parameters
    tc::Tensor newTokens; // [batch_size, beam_width]
    // optional parameters
    std::optional<tc::Tensor> finished_sum;           // [1] in pinned host memory
    std::optional<tc::Tensor> output_log_probs_tiled; // [request_output_length, batch_size, beam_width], must be float*
    std::optional<tc::Tensor>
        tgt_cache_indirection; // [local_batch_size, beam_width, max_seq_len], the k/v cache index for beam search
    std::unique_ptr<kernels::BeamHypotheses> beamHypotheses; // structure maintains some pointers of beam search

    tc::Tensor parent_ids_ptr; // [batch_size] int* (2-d array), each int* has [beam_width, max_seq_len]
};

} // namespace tensorrt_llm::layers
