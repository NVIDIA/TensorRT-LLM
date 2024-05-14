/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "tensorrt_llm/layers/decodingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/samplingLayer.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace
{
template <typename T>
bool allSame(std::optional<std::vector<T>> const& vOpt)
{
    if (!vOpt)
    {
        return true;
    }

    auto const& v = *vOpt;

    if (v.size() <= 1)
    {
        return true;
    }
    auto first = v[0];
    for (std::size_t i = 1; i < v.size(); ++i)
    {
        if (v[i] != first)
        {
            return false;
        }
    }
    return true;
}

bool hasDiffRuntimeArgs(std::shared_ptr<tensorrt_llm::layers::DynamicDecodeSetupParams> const& params)
{
    return !allSame(params->penaltyParams.frequencyPenalty) || !allSame(params->penaltyParams.presencePenalty)
        || !allSame(params->penaltyParams.repetitionPenalty) || !allSame(params->penaltyParams.temperature)
        || !allSame(params->penaltyParams.minLength);
}
} // namespace

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
DecodingLayer<T>::DecodingLayer(DecodingMode const& mode, DecoderDomain const& decoderDomain, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isTopKorTopP())
    {
        mDecodingLayer = std::make_unique<SamplingLayer<T>>(mDecodingMode, decoderDomain, mStream, mAllocator);
    }
    else if (mDecodingMode.isBeamSearch())
    {
        mDecodingLayer = std::make_unique<BeamSearchLayer<T>>(decoderDomain, mStream, mAllocator);
    }
    else if (mDecodingMode.isMedusa())
    {
        mDecodingLayer = std::make_unique<MedusaDecodingLayer<T>>(decoderDomain, mStream, mAllocator);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa}");
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    if (mDecodingMode.isTopKorTopP())
    { // sampling layers
        TLLM_CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", beamWidth);
        auto samplingParams = std::make_shared<SamplingSetupParams>();

        samplingParams->runtime_top_k = setupParams->samplingParams.runtime_top_k;
        samplingParams->runtime_top_p = setupParams->samplingParams.runtime_top_p;
        samplingParams->randomSeed = setupParams->randomSeed;

        samplingParams->top_p_decay = setupParams->samplingParams.top_p_decay;
        samplingParams->top_p_min = setupParams->samplingParams.top_p_min;
        samplingParams->top_p_reset_ids = setupParams->samplingParams.top_p_reset_ids;
        samplingParams->normalize_log_probs = setupParams->samplingParams.normalize_log_probs;

        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, samplingParams);
    }
    else if (mDecodingMode.isBeamSearch())
    { // beam search layer
        TLLM_CHECK_WITH_INFO(beamWidth > 1, "Decoding mode is beam search, but beamWidth <= 1 (%d <= 1)", beamWidth);
        auto beamSearchParams = std::make_shared<BeamSearchSetupParams>();

        beamSearchParams->beam_search_diversity_rate = setupParams->beamSearchParams.beam_search_diversity_rate;
        beamSearchParams->length_penalty = setupParams->beamSearchParams.length_penalty;
        beamSearchParams->early_stopping = setupParams->beamSearchParams.early_stopping;

        mHasDiffRuntimeArgs = hasDiffRuntimeArgs(setupParams);
        mDecodingLayer->setup(batchSize, beamWidth, nullptr, beamSearchParams);
    }
    else if (mDecodingMode.isMedusa())
    {
        auto medusaSetupParams = std::make_shared<MedusaSetupParams>();
        medusaSetupParams->runtimeTopK = setupParams->samplingParams.runtime_top_k;
        medusaSetupParams->runtimeHeadsTopK = setupParams->medusaParams.topKMedusaHeads;
        medusaSetupParams->randomSeed = setupParams->randomSeed;
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, medusaSetupParams);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa}");
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forward(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto outputs = std::dynamic_pointer_cast<DynamicDecodeOutputParams>(baseOutputs);
    auto params = std::dynamic_pointer_cast<DynamicDecodeInputParams>(baseInputs);

    SizeType32 batchSize{0};
    SizeType32 beamWidth{0};
    SizeType32 vocabSize{0};
    auto const maxSeqLen = outputs->output_ids.shape[outputs->output_ids.shape.size() - 1];
    auto batchSlots = params->batch_slots ? params->batch_slots->template getPtr<SizeType32 const>() : nullptr;
    if (params->logits)
    {
        auto const& logitsShape = params->logits->shape;
        TLLM_CHECK(logitsShape.size() == 3 || logitsShape.size() == 4);
        batchSize = logitsShape[0];
        auto const idxOffset = logitsShape.size() - 3;
        beamWidth = logitsShape[idxOffset + 1];
        vocabSize = logitsShape[idxOffset + 2];
    }
    else
    {
        TLLM_CHECK(params->logits_vec->size());
        auto const& logitsShape = params->logits_vec.value()[0].shape;
        TLLM_CHECK(logitsShape.size() == 3 || logitsShape.size() == 4);
        auto const idxOffset = logitsShape.size() - 3;
        batchSize = params->logits_vec->size();
        beamWidth = logitsShape[idxOffset + 1];
        vocabSize = logitsShape[idxOffset + 2];
    }

    auto const ite = params->ite;
    auto const step = params->step;

    // common inputs
    auto const& endIds = params->end_ids;
    auto const localBatchSize = static_cast<std::size_t>(params->local_batch_size);

    // dynamic decode GPT
    if (mDecodingMode.isBeamSearch())
    {
        TLLM_CHECK_WITH_INFO(beamWidth > 1, "Decoding mode is beam search, but beamWidth <= 1 (%d <= 1)", beamWidth);
        TLLM_CHECK_WITH_INFO(
            params->src_cache_indirection.has_value(), "src_cache_indirection is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(
            outputs->tgt_cache_indirection.has_value(), "tgt_cache_indirection is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs->parent_ids.has_value(), "parent_ids tensor is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs->finished.has_value(), "finished tensor is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs->cum_log_probs.has_value(), "cum_log_probs tensor is mandatory in beam search.");

        // Compute one by one if there are different runtime arguments
        //     due to Batch-Beam-Search is not supported yet, so we need to compute
        size_t const dynamic_decode_batch_size = mHasDiffRuntimeArgs ? 1 : localBatchSize;
        auto const dynamic_decode_total_iteration = mHasDiffRuntimeArgs ? localBatchSize : 1;

        for (uint32_t dynamic_ite = 0; dynamic_ite < dynamic_decode_total_iteration; ++dynamic_ite)
        {
            auto const dynamic_id_offset = dynamic_ite * dynamic_decode_batch_size * beamWidth;
            auto const dynamic_decode_vocab_size_units_offset = dynamic_id_offset * mDecoderDomain.getVocabSizePadded();

            auto const logits_offset
                = params->logits->slice({dynamic_decode_batch_size, params->logits->shape[1], params->logits->shape[2]},
                    dynamic_decode_vocab_size_units_offset);
            auto const end_id_offset
                = endIds.slice({dynamic_decode_batch_size}, dynamic_ite * dynamic_decode_batch_size);

            auto forwardParams = std::make_shared<BeamSearchInputParams>(step, ite, logits_offset, end_id_offset,
                *params->src_cache_indirection, static_cast<std::int32_t>(params->max_attention_window),
                static_cast<std::int32_t>(params->sink_token_length), static_cast<std::int32_t>(maxSeqLen));

            if (params->input_lengths)
            {
                forwardParams->input_lengths
                    = params->input_lengths->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);
            }

            auto outputParams = std::make_shared<BeamSearchOutputParams>(
                outputs->output_ids, outputs->parent_ids.value(), outputs->tgt_cache_indirection.value());

            outputParams->output_ids_ptr = std::move(outputs->output_ids_ptr);
            outputParams->parent_ids_ptr = std::move(outputs->parent_ids_ptr);
            outputParams->sequence_length
                = outputs->sequence_length->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);
            outputParams->finished
                = outputs->finished->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);
            outputParams->cum_log_probs
                = outputs->cum_log_probs->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);
            outputParams->output_log_probs = outputs->output_log_probs_tiled;
            outputParams->beamHypotheses = std::move(outputs->beamHypotheses);

            // beam_search_diversity_rate is only supported when using BeamHypotheses
            mDecodingLayer->forward(outputParams, forwardParams);
        } // end of dynamic_ite
    }
    else if (mDecodingMode.isTopKorTopP())
    { // beamWidth == 1
        TLLM_CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", beamWidth);

        // In sampling, we have supported batch sampling. So, we always compute all
        // sentences once.
        Tensor const logits_slice{
            params->logits->slice({localBatchSize, static_cast<size_t>(beamWidth), params->logits->shape[2]}, 0)};
        Tensor const end_id_slice{endIds.slice({localBatchSize}, 0)};
        auto decode_input_tensors = std::make_shared<SamplingInputParams>(
            step, ite, logits_slice, end_id_slice, static_cast<SizeType32>(maxSeqLen));

        decode_input_tensors->finished = params->finished;

        if (params->input_lengths)
        {
            auto& input_lengths = params->input_lengths.value();
            decode_input_tensors->input_lengths
                = input_lengths.slice({localBatchSize, static_cast<size_t>(beamWidth)}, 0);
        }
        decode_input_tensors->batch_slots = params->batch_slots;

        auto decode_outputs = std::make_shared<SamplingOutputParams>(outputs->output_ids);
        decode_outputs->output_ids_ptr = std::move(outputs->output_ids_ptr);
        if (outputs->sequence_length)
        {
            decode_outputs->sequence_length = outputs->sequence_length->slice({localBatchSize * beamWidth}, 0);
        }
        if (outputs->finished)
        {
            decode_outputs->finished = outputs->finished->slice({localBatchSize * beamWidth}, 0);
        }
        if (outputs->cum_log_probs)
        {
            decode_outputs->cum_log_probs = outputs->cum_log_probs->slice({localBatchSize * beamWidth}, 0);
        }
        if (outputs->output_log_probs_tiled)
        {
            Tensor& output_log_probs = outputs->output_log_probs_tiled.value();
            decode_outputs->output_log_probs = output_log_probs.slice({1, localBatchSize * beamWidth}, 0);
        }

        // Run TopK + TopP decode layers.
        mDecodingLayer->forward(decode_outputs, decode_input_tensors);
    }
    else if (mDecodingMode.isMedusa())
    {
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "Decoding mode is Medusa, but beamWidth != 1 (%d != 1)", beamWidth);

        auto medusaInputParams = std::make_shared<MedusaInputParams>(params->logits.value(), endIds);
        medusaInputParams->finished = outputs->finished.value();
        medusaInputParams->batch_slots = params->batch_slots;
        medusaInputParams->paths = params->medusaInputs->medusaPaths;
        medusaInputParams->medusaLogits = params->medusaInputs->medusaLogits;
        medusaInputParams->medusaCurTokensPerStep = params->medusaInputs->medusaCurTokensPerStep;
        medusaInputParams->medusaTargetTokensPerStep = params->medusaInputs->medusaTargetTokensPerStep;
        medusaInputParams->treeIds = params->medusaInputs->medusaTreeIds;

        auto medusaOutputParams = std::make_shared<MedusaOutputParams>(outputs->output_ids);
        medusaOutputParams->sequence_length = outputs->sequence_length.value();
        medusaOutputParams->finished = outputs->finished.value();
        medusaOutputParams->medusaOutputs = MedusaOutputParams::MedusaOutputs();
        medusaOutputParams->medusaOutputs->nextDraftTokens = outputs->medusaOutputs->nextDraftTokens;
        medusaOutputParams->medusaOutputs->acceptedLengths = outputs->medusaOutputs->acceptedLengths;
        medusaOutputParams->medusaOutputs->acceptedLengthsCumSum = outputs->medusaOutputs->acceptedLengthsCumSum;
        medusaOutputParams->medusaOutputs->pathsOffsets = outputs->medusaOutputs->pathsOffsets;

        mDecodingLayer->forward(medusaOutputParams, medusaInputParams);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class DecodingLayer<float>;
template class DecodingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
