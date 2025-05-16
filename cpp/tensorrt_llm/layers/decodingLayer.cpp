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

#include "decodingLayer.h"
#include "tensorrt_llm/layers/beamSearchLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/eagleDecodingLayer.h"
#include "tensorrt_llm/layers/explicitDraftTokensLayer.h"
#include "tensorrt_llm/layers/externalDraftTokensLayer.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/layers/lookaheadDecodingLayer.h"
#include "tensorrt_llm/layers/medusaDecodingLayer.h"
#include "tensorrt_llm/layers/samplingLayer.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
DecodingLayer<T>::DecodingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isTopKorTopP())
    {
        mDecodingLayer = std::make_unique<SamplingLayer<T>>(mDecodingMode, decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isBeamSearch())
    {
        mDecodingLayer = std::make_unique<BeamSearchLayer<T>>(mDecodingMode, decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isMedusa())
    {
        mDecodingLayer = std::make_unique<MedusaDecodingLayer<T>>(decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isLookahead())
    {
        mDecodingLayer = std::make_unique<LookaheadDecodingLayer<T>>(mDecoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        mDecodingLayer = std::make_unique<ExplicitDraftTokensLayer<T>>(decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isExternalDraftTokens())
    {
        mDecodingLayer = std::make_unique<ExternalDraftTokensLayer<T>>(mDecodingMode, decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isEagle())
    {
        mDecodingLayer = std::make_unique<EagleDecodingLayer<T>>(decoderDomain, mBufferManager);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens, ExternalDraftTokens, Eagle}");
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    TLLM_CHECK_WITH_INFO(setupParams->decodingParams, "decodingParams for setup is not set");

    if (mDecodingMode.isBeamSearch())
    {
        TLLM_CHECK_WITH_INFO(
            beamWidth > 1, "Decoding mode is %s, but beamWidth <= 1 (%d <= 1)", mDecodingMode.getName(), beamWidth);
    }
    else if (mDecodingMode.isTopKorTopP() || mDecodingMode.isMedusa() || mDecodingMode.isLookahead()
        || mDecodingMode.isExplicitDraftTokens() || mDecodingMode.isExternalDraftTokens() || mDecodingMode.isEagle())
    {
        TLLM_CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is %s, but beamWidth != 1 (%d != 1)", mDecodingMode.getName(), beamWidth);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens, ExternalDraftTokens, Eagle}");
    }

    mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardAsync(outputParams, inputParams, workspace);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forwardSync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardSync(outputParams, inputParams, workspace);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t DecodingLayer<T>::getWorkspaceSize() const noexcept
{
    return mDecodingLayer->getWorkspaceSize();
}

template <typename T>
std::tuple<std::shared_ptr<BaseDecodingOutputs>, std::shared_ptr<BaseDecodingInputs>> DecodingLayer<T>::prepareParams(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto params = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(params, mDecoderDomain);
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
        auto const localBatchSize = static_cast<int64_t>(params->localBatchSize);

        TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        // In sampling, we have supported batch sampling. So, we always compute all
        // sentences once.
        TensorConstPtr logitsSlice = ITensor::slice(*params->logits, 0, localBatchSize);
        TensorConstPtr endIdSlice = ITensor::slice(endIds, 0, localBatchSize);
        auto decodeInputs = std::make_shared<SamplingInputs>(endIdSlice, params->batchSlots, step, ite, localBatchSize);

        decodeInputs->finished = params->finished;

        decodeInputs->logits = logitsSlice;

        if (params->inputLengths)
        {
            auto& inputLengths = params->inputLengths.value();
            decodeInputs->inputLengths = ITensor::slice(inputLengths, 0, localBatchSize);
        }
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
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isExternalDraftTokens())
    {
        auto externalDraftTokenParams = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);
        auto const ite = externalDraftTokenParams->ite;
        auto const step = externalDraftTokenParams->step;
        auto const localBatchSize = static_cast<int64_t>(externalDraftTokenParams->localBatchSize);

        TLLM_CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        // Compute all sentences once since batch-sampling is supported
        TensorConstPtr logitsSlice = ITensor::slice(*externalDraftTokenParams->logits, 0, localBatchSize);
        TensorConstPtr endIdSlice = ITensor::slice(endIds, 0, localBatchSize);
        auto decodeInputs = std::make_shared<ExternalDraftTokensInputs>(
            endIdSlice, externalDraftTokenParams->batchSlots, step, ite, localBatchSize);

        decodeInputs->finished = externalDraftTokenParams->finished;

        decodeInputs->logits = logitsSlice;

        if (externalDraftTokenParams->inputLengths)
        {
            auto& inputLengths = externalDraftTokenParams->inputLengths.value();
            decodeInputs->inputLengths = ITensor::slice(inputLengths, 0, localBatchSize);
        }
        decodeInputs->draftLogits = externalDraftTokenParams->draftLogits;
        decodeInputs->draftProbs = externalDraftTokenParams->draftProbs;
        decodeInputs->targetProbs = externalDraftTokenParams->targetProbs;
        decodeInputs->numDraftTokens = externalDraftTokenParams->numDraftTokens;
        decodeInputs->numDraftTokensHost = externalDraftTokenParams->numDraftTokensHost;
        decodeInputs->draftTokenIds = externalDraftTokenParams->draftTokenIds;
        decodeInputs->constantThreshold = externalDraftTokenParams->constantThreshold;
        decodeInputs->useRandomAcceptanceThreshold = externalDraftTokenParams->useRandomAcceptanceThreshold;
        decodeInputs->step = externalDraftTokenParams->step;
        decodeInputs->useDraftLogits = externalDraftTokenParams->useDraftLogits;
        decodeInputs->useDraftLogitsHost = externalDraftTokenParams->useDraftLogitsHost;

        preparedInputs = decodeInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isEagle())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens, ExternalDraftTokens, Eagle}");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {preparedOutputs, preparedInputs};
}

template class DecodingLayer<float>;
template class DecodingLayer<half>;

} // namespace tensorrt_llm::layers
