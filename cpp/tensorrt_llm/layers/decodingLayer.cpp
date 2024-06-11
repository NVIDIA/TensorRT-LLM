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
#include "tensorrt_llm/layers/layerUtils.h"
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
        || !allSame(params->penaltyParams.minLength) || !allSame(params->penaltyParams.noRepeatNgramSize);
}
} // namespace

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
DecodingLayer<T>::DecodingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
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
    else if (mDecodingMode.isLookahead())
    {
        // TODO(nkorobov) add lookahead layer
        TLLM_LOG_WARNING("Lookahead decoding is not supported yet.");
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        mDecodingLayer = std::make_unique<ExplicitDraftTokensLayer<T>>(decoderDomain, mStream, mAllocator);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens}");
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
        samplingParams->outputLogProbs = setupParams->samplingParams.outputLogProbs;
        samplingParams->cumLogProbs = setupParams->samplingParams.cumLogProbs;

        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, samplingParams);
    }
    else if (mDecodingMode.isBeamSearch())
    { // beam search layer
        TLLM_CHECK_WITH_INFO(beamWidth > 1, "Decoding mode is beam search, but beamWidth <= 1 (%d <= 1)", beamWidth);
        auto beamSearchParams = std::make_shared<BeamSearchSetupParams>();

        beamSearchParams->beam_search_diversity_rate = setupParams->beamSearchParams.beam_search_diversity_rate;
        beamSearchParams->length_penalty = setupParams->beamSearchParams.length_penalty;
        beamSearchParams->early_stopping = setupParams->beamSearchParams.early_stopping;
        beamSearchParams->hasDiffRuntimeArgs = hasDiffRuntimeArgs(setupParams);

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
    else if (mDecodingMode.isLookahead())
    {
        // TODO(nkorobov) add lookahead layer
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        auto explicitDraftTokensSetupParams = std::make_shared<ExplicitDraftTokensSetupParams>();
        explicitDraftTokensSetupParams->temperature = setupParams->penaltyParams.temperature;
        explicitDraftTokensSetupParams->randomSeed = setupParams->randomSeed;
        mDecodingLayer->setup(batchSize, /* beamWidth */ 1, batchSlots, explicitDraftTokensSetupParams);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens}");
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forwardAsync(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardAsync(outputParams, inputParams);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forwardSync(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardSync(outputParams, inputParams);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
std::tuple<std::shared_ptr<BaseOutputParams>, std::shared_ptr<BaseInputParams>> DecodingLayer<T>::prepareParams(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto outputs = std::dynamic_pointer_cast<DynamicDecodeOutputParams>(baseOutputs);
    auto params = std::dynamic_pointer_cast<DynamicDecodeInputParams>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(params, mDecoderDomain);
    auto const maxSeqLen = outputs->output_ids.shape[outputs->output_ids.shape.size() - 1];
    auto const& endIds = params->end_ids;

    std::shared_ptr<BaseOutputParams> preparedOutputs;
    std::shared_ptr<BaseInputParams> preparedInputs;

    // dynamic decode GPT
    if (mDecodingMode.isBeamSearch())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isTopKorTopP())
    { // beamWidth == 1
        auto const ite = params->ite;
        auto const step = params->step;
        auto const localBatchSize = static_cast<std::size_t>(params->local_batch_size);

        TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        // In sampling, we have supported batch sampling. So, we always compute all
        // sentences once.
        Tensor const logitsSlice{params->logits->slice(
            {localBatchSize, static_cast<size_t>(localDecoderDomain.getBeamWidth()), params->logits->shape[2]}, 0)};
        Tensor const endIdSlice{endIds.slice({localBatchSize}, 0)};
        auto decodeInputs = std::make_shared<SamplingInputParams>(
            step, ite, logitsSlice, endIdSlice, static_cast<SizeType32>(maxSeqLen));

        decodeInputs->finished = params->finished;

        if (params->input_lengths)
        {
            auto& inputLengths = params->input_lengths.value();
            decodeInputs->input_lengths
                = inputLengths.slice({localBatchSize, static_cast<size_t>(localDecoderDomain.getBeamWidth())}, 0);
        }
        decodeInputs->batch_slots = params->batch_slots;

        auto decodeOutputs = std::make_shared<SamplingOutputParams>(outputs->output_ids);
        decodeOutputs->output_ids_ptr = std::move(outputs->output_ids_ptr);
        if (outputs->sequence_length)
        {
            decodeOutputs->sequence_length
                = outputs->sequence_length->slice({localBatchSize * localDecoderDomain.getBeamWidth()}, 0);
        }
        if (outputs->finished)
        {
            decodeOutputs->finished = outputs->finished->slice({localBatchSize * localDecoderDomain.getBeamWidth()}, 0);
        }
        if (outputs->cum_log_probs)
        {
            decodeOutputs->cum_log_probs
                = outputs->cum_log_probs->slice({localBatchSize * localDecoderDomain.getBeamWidth()}, 0);
        }
        if (outputs->output_log_probs_tiled)
        {
            Tensor& output_log_probs = outputs->output_log_probs_tiled.value();
            decodeOutputs->output_log_probs
                = output_log_probs.slice({1, localBatchSize * localDecoderDomain.getBeamWidth()}, 0);
        }

        preparedInputs = decodeInputs;
        preparedOutputs = decodeOutputs;
    }
    else if (mDecodingMode.isMedusa())
    {
        TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is Medusa, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        auto medusaInputParams = std::make_shared<MedusaInputParams>(params->logits.value(), endIds);
        medusaInputParams->finished = outputs->finished.value();
        medusaInputParams->batch_slots = params->batch_slots;
        medusaInputParams->paths = params->medusaInputs->medusaPaths;
        medusaInputParams->medusaLogits = params->medusaInputs->medusaLogits;
        medusaInputParams->medusaCurTokensPerStep = params->medusaInputs->medusaCurTokensPerStep;
        medusaInputParams->medusaTargetTokensPerStep = params->medusaInputs->medusaTargetTokensPerStep;
        medusaInputParams->treeIds = params->medusaInputs->medusaTreeIds;

        preparedInputs = medusaInputParams;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isLookahead())
    {
        // TODO(nkorobov) add lookahead layer
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        // TODO(nkorobov) add explicit draft tokens layer param prep
        // Simply forward params for now
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens}");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {preparedOutputs, preparedInputs};
}

template class DecodingLayer<float>;
template class DecodingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
