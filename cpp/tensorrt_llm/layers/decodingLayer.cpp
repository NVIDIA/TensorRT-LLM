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
#include "tensorrt_llm/layers/beamSearchLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/explicitDraftTokensLayer.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/layers/medusaDecodingLayer.h"
#include "tensorrt_llm/layers/samplingLayer.h"

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
    // return !allSame(params->penaltyParams.frequencyPenalty) || !allSame(params->penaltyParams.presencePenalty)
    //     || !allSame(params->penaltyParams.repetitionPenalty) || !allSame(params->penaltyParams.temperature)
    //     || !allSame(params->penaltyParams.minLength) || !allSame(params->banWordsInputs.noRepeatNgramSize);
    return false;
}
} // namespace

namespace tensorrt_llm::layers
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
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    TLLM_CHECK_WITH_INFO(setupParams->decodingParams, "decodingParams for setup is not set");

    if (mDecodingMode.isTopKorTopP())
    { // sampling layers
        TLLM_CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams);
    }
    else if (mDecodingMode.isBeamSearch())
    { // beam search layer
        TLLM_CHECK_WITH_INFO(beamWidth > 1, "Decoding mode is beam search, but beamWidth <= 1 (%d <= 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams);
    }
    else if (mDecodingMode.isMedusa())
    {
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "Decoding mode is Medusa, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams);
    }
    else if (mDecodingMode.isLookahead())
    {
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "Decoding mode is Lookahead, but beamWidth != 1 (%d != 1)", beamWidth);
        // TODO(nkorobov) add lookahead layer
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        TLLM_CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is ExplicitDraftTokens, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams);
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
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardAsync(outputParams, inputParams);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forwardSync(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardSync(outputParams, inputParams);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
std::tuple<std::shared_ptr<BaseDecodingOutputs>, std::shared_ptr<BaseDecodingInputs>> DecodingLayer<T>::prepareParams(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto params = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(params, mDecoderDomain);
    auto const maxSeqLen = baseOutputs->outputIds.shape[baseOutputs->outputIds.shape.size() - 1];
    auto const& endIds = params->endIds;

    std::shared_ptr<BaseDecodingOutputs> preparedOutputs;
    std::shared_ptr<BaseDecodingInputs> preparedInputs;

    if (mDecodingMode.isBeamSearch())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isTopKorTopP())
    {
        auto const ite = params->ite;
        auto const step = params->step;
        auto const localBatchSize = static_cast<std::size_t>(params->localBatchSize);

        TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        // In sampling, we have supported batch sampling. So, we always compute all
        // sentences once.
        Tensor const logitsSlice{params->logits->slice(
            {localBatchSize, static_cast<size_t>(localDecoderDomain.getBeamWidth()), params->logits->shape[2]}, 0)};
        Tensor const endIdSlice{endIds.slice({localBatchSize}, 0)};
        auto decodeInputs = std::make_shared<SamplingInputs>(endIdSlice, step, ite, localBatchSize);

        decodeInputs->finished = params->finished;

        decodeInputs->logits = logitsSlice;

        if (params->inputLengths)
        {
            auto& inputLengths = params->inputLengths.value();
            decodeInputs->inputLengths
                = inputLengths.slice({localBatchSize, static_cast<size_t>(localDecoderDomain.getBeamWidth())}, 0);
        }
        decodeInputs->batchSlots = params->batchSlots;

        preparedInputs = decodeInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isMedusa())
    {
        TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is Medusa, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        preparedInputs = baseInputs;
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

} // namespace tensorrt_llm::layers
