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
#include <tensorrt_llm/runtime/speculativeDecodingModule.h>

#include <optional>
#include <vector>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::layers
{

//!
//! \brief In a DecodingLayer's life cycle, it is constructed once;
//! `setup` repeatedly, but once per request; `forward*` repeatedly, many times per request.
//! A possible sequence would be, construct(maxBatchSize) -> setup({1,3}) -> forward({1, 3})
//! -> forward({1, 3}) -> setup({2,4}) -> forward({1, 3, 2, 4}) -> forward({1, 3, 2, 4})
//! -> forward({1, 2, 4}), where {a,b} are batchSlots, and {3} ends at last step.
//! As a result there are three types of batches.
//! 1. `maxBatchSize` for each layers to reserve resources.
//!    It is passed through class constructor, in DecoderDomain.getBatchSize().
//! 2. `setupBatchSize` for setting up layers for a batch of new requests.
//!    It is passed through `setup` method.
//! 3. `forwardBatchSize` for layers forwarding for a batch of existing active requests.
//!    it is passed through `forwardAsync` and `forwardSync` methods.
//! `setup` and `forward` always provide `batch_slots` indexed by
//! local batch index ranging in [0, setupBatchSize) or [0, forwardBatchSize),
//! holding the global batch index ranging in [0, maxBatchSize).
//! In case of beam search, maxBatchSize = forwardBatchSize = 1.
class DecoderDomain
{
public:
    DecoderDomain(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 vocabSize,
        std::optional<runtime::SizeType32> vocabSizePadded = std::nullopt,
        std::shared_ptr<runtime::SpeculativeDecodingModule const> speculativeDecodingModule = nullptr)
        : mBatchSize(batchSize)
        , mBeamWidth(beamWidth)
        , mVocabSize(vocabSize)
        , mVocabSizePadded(vocabSizePadded.value_or(vocabSize))
        , mSpeculativeDecodingModule(speculativeDecodingModule)
    {
    }

    [[nodiscard]] runtime::SizeType32 getBatchSize() const
    {
        return mBatchSize;
    }

    [[nodiscard]] runtime::SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] runtime::SizeType32 getVocabSize() const
    {
        return mVocabSize;
    }

    [[nodiscard]] runtime::SizeType32 getVocabSizePadded() const
    {
        return mVocabSizePadded;
    }

    [[nodiscard]] runtime::SizeType32 getMaxDecodingTokens() const
    {
        return mSpeculativeDecodingModule ? mSpeculativeDecodingModule->getMaxDecodingTokens() : 1;
    }

    [[nodiscard]] std::shared_ptr<runtime::SpeculativeDecodingModule const> getSpeculativeDecodingModule() const
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set to decoder domain");
        return mSpeculativeDecodingModule;
    }

private:
    runtime::SizeType32 mBatchSize;
    runtime::SizeType32 mBeamWidth;
    runtime::SizeType32 mVocabSize;
    runtime::SizeType32 mVocabSizePadded;
    std::shared_ptr<runtime::SpeculativeDecodingModule const> mSpeculativeDecodingModule;
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
        std::optional<std::vector<float>> temperature;                     // [1] or [setupBatchSize] on cpu
        std::optional<std::vector<runtime::SizeType32>> minLength;         // [1] or [setupBatchSize] on cpu
        std::optional<std::vector<float>> repetitionPenalty;               // [1] or [setupBatchSize] on cpu
        std::optional<std::vector<float>> presencePenalty;                 // [1] or [setupBatchSize] on cpu
        std::optional<std::vector<float>> frequencyPenalty;                // [1] or [setupBatchSize] on cpu
        std::optional<std::vector<runtime::SizeType32>> noRepeatNgramSize; // [1] or [setupBatchSize] on cpu
    };

    struct SamplingParams
    {
        // baseSamplingLayer
        std::optional<std::vector<runtime::SizeType32>> runtime_top_k; // [1] or [setupBatchSize] on cpu
        std::optional<std::vector<float>> runtime_top_p;               // [1] or [setupBatchSize] on cpu

        // topPSamplingLayer
        std::optional<std::vector<float>> top_p_decay;                    // [setupBatchSize], must between [0, 1]
        std::optional<std::vector<float>> top_p_min;                      // [setupBatchSize], must between [0, 1]
        std::optional<std::vector<runtime::TokenIdType>> top_p_reset_ids; // [setupBatchSize]
        std::optional<bool> normalize_log_probs;
        std::optional<std::vector<bool>> outputLogProbs;                  // [setupBatchSize]
        std::optional<std::vector<bool>> cumLogProbs;                     // [setupBatchSize]
    };

    struct BeamSearchParams
    {
        // BeamSearchLayer
        std::optional<std::vector<float>> beam_search_diversity_rate; // [setupBatchSize] on cpu
        std::optional<std::vector<float>> length_penalty;             // [setupBatchSize] on cpu
        std::optional<std::vector<int>> early_stopping;               // [setupBatchSize] on cpu
    };

    struct MedusaParams
    {
        // Medusa params
        std::optional<std::vector<std::vector<runtime::SizeType32>>>
            topKMedusaHeads; // [setupBatchSize, maxMedusaHeads]
    };

    std::optional<std::vector<uint64_t>> randomSeed; // [1] or [setupBatchSize] on cpu

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
    tc::Tensor end_ids;                    // [maxBatchSize]
    std::optional<tc::Tensor> batch_slots; // [forwardBatchSize], on pinned memory
    std::optional<tc::Tensor> finished;    // [maxBatchSize, maxBeamWidth]
};

class DynamicDecodeInputParams : public BaseInputParams
{
public:
    DynamicDecodeInputParams(runtime::SizeType32 step, runtime::SizeType32 ite, runtime::SizeType32 maxInputLength,
        runtime::SizeType32 maxAttentionWindow, runtime::SizeType32 sinkTokenLength, runtime::SizeType32 localBatchSize,
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
    runtime::SizeType32 max_input_length;
    runtime::SizeType32 max_attention_window;
    runtime::SizeType32 sink_token_length;
    runtime::SizeType32 local_batch_size;
    runtime::SizeType32 max_stop_words_len;
    runtime::SizeType32 max_bad_words_len;

    // One of these two fields has to be set
    // DynamicDecodeLayer::forward checks for it
    // Need both of these fields to support legacy code during transition period to the batched decoder
    std::optional<tc::Tensor> logits;                  // [maxBatchSize, beamWidth, vocabSizePadded]
    std::optional<std::vector<tc::Tensor>> logits_vec; // [forwardBatchSize][beamWidth, vocabSizePadded], on gpu

    // optional parameters
    std::optional<tc::Tensor> src_cache_indirection; // [forwardBatchSize, maxBeamWidth, maxSeqLen] - the k/v cache
                                                     // index for beam search, mandatory for beam search, on gpu
    std::optional<tc::Tensor> sequence_limit_length; // [maxBatchSize], on gpu
    std::optional<tc::Tensor> embedding_bias;        // [vocabSizePadded], on gpu
    std::optional<tc::Tensor> input_lengths;         // [maxBatchSize, maxBeamWidth], on gpu
    std::optional<tc::Tensor> bad_words_ptr;         // [maxBatchSize][2, bad_words_length], on gpu
    std::optional<tc::Tensor> bad_words_lengths;     // [maxBatchSize], on gpu
    std::optional<tc::Tensor> stop_words_ptr;        // [maxBatchSize][2, stop_words_length], on gpu
    std::optional<tc::Tensor> stop_words_lengths;    // [maxBatchSize], on gpu

    // Medusa inputs
    class MedusaInputs
    {
    public:
        tc::Tensor medusaCurTokensPerStep;                 // [maxBatchSize], optional, on gpu
        tc::Tensor medusaTargetTokensPerStep;              // [maxBatchSize], optional, on gpu
        tc::Tensor medusaPaths;                            // [maxBatchSize, maxPathLen, maxPathLen]
                                                           // optional, on gpu
        tc::Tensor medusaTreeIds;                          // [maxBatchSize, maxDecodingTokens], optional, on gpu
        std::vector<std::vector<tc::Tensor>> medusaLogits; // [maxBatchSize][maxDraftPathLen]
                                                           // [maxDecodingTokens, vocabSizePadded], optional, on gpu
    };

    // Explicit draft tokens inputs
    // FIXME(nkorobov): this should be ExplicitDraftTokensBuffers?
    class ExplicitDraftTokensInputs
    {
    public:
    };

    std::optional<MedusaInputs> medusaInputs;

    std::optional<ExplicitDraftTokensInputs> explicitDraftTokensInputs;
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
    tc::Tensor output_ids; // [maxBatchSize, maxSeqLen]

    // optional parameters
    std::optional<tc::Tensor> finished;         // [maxBatchSize * maxBeamWidth], optional
    std::optional<tc::Tensor> sequence_length;  // [maxBatchSize * maxBeamWidth], optional
    std::optional<tc::Tensor> cum_log_probs;    // [maxBatchSize * maxBeamWidth], necessary in beam search
    std::optional<tc::Tensor> output_log_probs; // [maxBatchSize, maxBeamWidth, maxSeqLen], must be float*, optional
    std::optional<tc::Tensor> parent_ids;       // [maxBatchSize, maxBeamWidth, maxSeqLen], necessary in beam search

    tc::Tensor output_ids_ptr; // [maxBatchSize] int* (2-d array), each int* has [maxBeamWidth, maxSeqLen]

    //!
    //! \brief SpeculativeDecodingOutputs outputs.
    //!
    //! For one example sequence [a, b] [c] <x, y, z>, where, [a, b, c] is the accepted sequence,
    //! [c] is the last accepted token, and <x, y, z> is the draft tokens from `nextDraftTokens` saved by last step.
    //! [c]'s position id is known, only position ids for <x, y, z> need to be provided in `nextDraftPosIds`.
    //! LLM inputs {c, x, y, z} and generates {c', x', y', z'}.
    //!
    //! {c'} is always accepted and {x', z'} is supposed to be accepted.
    //! The accepted tokens [c', x', z'] is saved in `output_ids` in-place, starting from `sequence_length`.
    //! The `acceptedLength` is 3, and the accepted draft tokens length is 2.
    //! `sequence_length` is also increaded by `acceptedLength` in-place.
    //! The pathsOffset is {0, 1, 3} for {c', x', z'}.
    //! [] for accepted, <> for draft, {} for input/output.
    //!
    //! For a batchSlots {1, 3}, `acceptedLengthsCumSum` is an exclusive sum of `acceptedLength` over the batch,
    //! the `acceptedLengths` may be {3, 5}, `acceptedLengthsCumSum` is {0, 3, 8}.
    class SpeculativeDecodingOutputs
    {
    public:
        tc::Tensor nextDraftTokens;       // [maxBatchSize, maxDecodingDraftTokens], draft tokens for the next step
        tc::Tensor nextDraftPosIds;       // [maxBatchSize, maxDecodingDraftTokens], draft token position IDs
        tc::Tensor nextDraftLengths;      // [maxBatchSize], next step draft tokens lengths
        tc::Tensor acceptedLengths;       // [maxBatchSize], lengths of the accepted draft tokens + 1.
        tc::Tensor acceptedLengthsCumSum; // [maxBatchSize + 1] accumulative sum along batchSlots.
        tc::Tensor pathsOffsets;          // [maxBatchSize, maxPathLen]
        tc::Tensor packedMasks;           // [maxBatchSize, maxDecodingTokens, divUp(maxDecodingTokens, 32)]
    };

    class ExplicitDraftTokensOutputs : public SpeculativeDecodingOutputs
    {
    public:
        //! Draft tokens for the next iteration. The first token in each path is the last accepted token at current
        //! iteration. E.g. if batchSize == 1, maxNumPaths == 2, maxPathLen== 3, [[[0, 1, 2], [0, 1, 10]]]
        tc::Tensor unpackedNextDraftTokens; // [maxBatchSize, maxNumPaths, maxPathLen] on gpu
        //! Indices of draft tokens in the compressed `nextFlatTokens` for the next iteration.
        //! Using example above, [[[0, 1, 2], [0, 1, 3]]]
        tc::Tensor unpackedNextDraftIndices; // [maxBatchSize, maxNumPaths, maxPathLen] on gpu
        //! Probabilities of the next draft tokens.
        tc::Tensor nextDraftProbs; // [maxBatchSize, maxNumPaths, maxPathDraftLen, vocabSize] on gpu
        //! Baseline for the position ids.
        tc::Tensor positionIdsBase; // [maxBatchSize] on gpu
        //! Randomly sampled data (between 0.f and 1.f)
        tc::Tensor randomDataSample; // [maxBatchSize] on gpu
        //! Randomly sampled data (between 0.f and 1.f)
        tc::Tensor randomDataValidation; // [maxBatchSize, maxNumPaths, maxDraftPathLen] on gpu
        //! Sampling temperature.
        tc::Tensor temperatures; // [maxBatchSize] on gpu
    };

    std::optional<SpeculativeDecodingOutputs> speculativeDecodingOutputs;

    std::optional<ExplicitDraftTokensOutputs> explicitDraftTokensOutputs;
};

class DynamicDecodeOutputParams : public BaseOutputParams
{
public:
    explicit DynamicDecodeOutputParams(tc::Tensor outputIds)
        : BaseOutputParams{std::move(outputIds)}
    {
    }

    // mandatory parameters
    tc::Tensor newTokens; // [maxBatchSize, maxBeamWidth]
    // optional parameters
    std::optional<tc::Tensor> finished_sum;           // [1] in pinned host memory
    std::optional<tc::Tensor> output_log_probs_tiled; // [maxSeqLen, maxBatchSize, maxBeamWidth], must be float*
    std::optional<tc::Tensor>
        tgt_cache_indirection; // [forwardBatchSize, maxBeamWidth, maxSeqLen], the k/v cache index for beam search
    std::unique_ptr<kernels::BeamHypotheses> beamHypotheses; // structure maintains some pointers of beam search

    tc::Tensor parent_ids_ptr; // [maxBatchSize] int* (2-d array), each int* has [maxBeamWidth, maxSeqLen]
};

} // namespace tensorrt_llm::layers
