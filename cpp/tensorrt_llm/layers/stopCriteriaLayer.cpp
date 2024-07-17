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

#include "tensorrt_llm/layers/stopCriteriaLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"
#include "tensorrt_llm/layers/layerUtils.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
StopCriteriaLayer<T>::StopCriteriaLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds.shape[outputs->outputIds.shape.size() - 1];
    auto batchSlots = inputs->batchSlots ? inputs->batchSlots->template getPtr<SizeType32 const>() : nullptr;

    TLLM_CHECK_WITH_INFO(inputs->stopCriteriaInputs, "stopCriteriaInputs for forward is not set");

    if (mDecodingMode.isUseStopWords())
    {
        checkStopWordsStopCriteria(outputs, inputs, batchSlots, localDecoderDomain, maxSeqLen, mStream);
    }
    if (mDecodingMode.isUseExplicitEosStop())
    {
        checkEosToken(outputs, inputs, batchSlots, localDecoderDomain, maxSeqLen, mStream);
    }
    if (mDecodingMode.isUseMaxLengthStop())
    {
        checkMaxLengthStopCriteria(outputs, inputs, batchSlots, localDecoderDomain, maxSeqLen, mStream);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkStopWordsStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlots, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxStopWordsLength = inputs->stopCriteriaInputs->maxStopWordsLen;
    if (maxStopWordsLength)
    {
        auto numNewTokens = outputs->numNewTokens ? outputs->numNewTokens->template getPtr<SizeType32>() : nullptr;
        invokeStopWordsCriterion(outputs->outputIdsPtr.template getPtr<TokenIdType const*>(),
            outputs->parentIdsPtr.template getPtr<SizeType32 const*>(),
            inputs->stopCriteriaInputs->stopWordsPtr->template getPtr<TokenIdType const*>(),
            reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>()),
            outputs->sequenceLength->template getPtr<SizeType32>(), batchSlots,
            inputs->stopCriteriaInputs->stopWordsLengths->template getPtr<SizeType32 const>(), numNewTokens,
            maxStopWordsLength, decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), maxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkMaxLengthStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlots, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (inputs->stopCriteriaInputs->sequenceLimitLength)
    {
        auto numNewTokens = outputs->numNewTokens ? outputs->numNewTokens->template getPtr<SizeType32>() : nullptr;

        invokeLengthCriterion(
            reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>()),
            outputs->finishedSum ? outputs->finishedSum->template getPtr<SizeType32>() : nullptr,
            inputs->stopCriteriaInputs->sequenceLimitLength->template getPtr<SizeType32 const>(),
            outputs->sequenceLength->template getPtr<SizeType32>(), numNewTokens, batchSlots,
            decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), stream);
        sync_check_cuda_error();
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkEosToken(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlots, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto numNewTokens = outputs->numNewTokens ? outputs->numNewTokens->template getPtr<SizeType32>() : nullptr;

    invokeExplicitEOSCriterion(outputs->outputIdsPtr.template getPtr<TokenIdType const*>(),
        inputs->endIds.template getPtr<TokenIdType const>(),
        reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>()),
        outputs->sequenceLength->template getPtr<SizeType32>(), numNewTokens, batchSlots, decoderDomain.getBatchSize(),
        decoderDomain.getBeamWidth(), decoderDomain.getMaxDecodingTokens(), stream);
    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class StopCriteriaLayer<float>;
template class StopCriteriaLayer<half>;

} // namespace tensorrt_llm::layers
