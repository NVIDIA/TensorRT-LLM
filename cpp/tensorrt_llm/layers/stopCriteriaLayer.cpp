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

#include "stopCriteriaLayer.h"
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
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, BufferConstPtr batchSlots,
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
    auto const maxSeqLen = outputs->outputIds->getDimension<-1>();
    auto batchSlotsPtr = bufferCast<SizeType32>(*inputs->batchSlots);

    TLLM_CHECK_WITH_INFO(inputs->stopCriteriaInputs, "stopCriteriaInputs for forward is not set");

    if (mDecodingMode.isUseStopWords())
    {
        checkStopWordsStopCriteria(outputs, inputs, batchSlotsPtr, localDecoderDomain, maxSeqLen, getStream());
    }
    if (mDecodingMode.isUseExplicitEosStop())
    {
        checkEosToken(outputs, inputs, batchSlotsPtr, localDecoderDomain, maxSeqLen, getStream());
    }
    if (mDecodingMode.isUseMaxLengthStop())
    {
        checkMaxLengthStopCriteria(outputs, inputs, batchSlotsPtr, localDecoderDomain, maxSeqLen, getStream());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkStopWordsStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlotsPtr, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxStopWordsLength = inputs->stopCriteriaInputs->maxStopWordsLen;
    if (maxStopWordsLength)
    {
        auto numNewTokens = bufferCastOrNull<SizeType32>(outputs->numNewTokens);
        auto outputIdsPtr = bufferCast<SizeType32 const*>(*outputs->outputIdsPtr);
        auto parentIdsPtr = bufferCast<SizeType32 const*>(*outputs->parentIdsPtr);
        invokeStopWordsCriterion(outputIdsPtr, parentIdsPtr,
            bufferCastOrNull<TokenIdType const*>(inputs->stopCriteriaInputs->stopWordsPtr),
            reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished)),
            bufferCastOrNull<SizeType32>(outputs->sequenceLength), batchSlotsPtr,
            bufferCastOrNull<SizeType32>(inputs->stopCriteriaInputs->stopWordsLengths), numNewTokens,
            maxStopWordsLength, decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), maxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkMaxLengthStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlotsPtr, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (inputs->stopCriteriaInputs->sequenceLimitLength)
    {
        auto numNewTokens = bufferCastOrNull<SizeType32>(outputs->numNewTokens);

        invokeLengthCriterion(
            reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished)),
            bufferCastOrNull<SizeType32>(outputs->finishedSum),
            bufferCastOrNull<SizeType32>(inputs->stopCriteriaInputs->sequenceLimitLength),
            bufferCastOrNull<SizeType32>(outputs->sequenceLength), numNewTokens, batchSlotsPtr,
            decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), stream);
        sync_check_cuda_error();
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkEosToken(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlotsPtr, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto numNewTokens = bufferCastOrNull<SizeType32>(outputs->numNewTokens);

    auto sequenceLengthsPtr = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto endIdsPtr = bufferCastOrNull<TokenIdType>(inputs->endIds);
    auto finishedStatePtr
        = reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished));
    invokeExplicitEOSCriterion(bufferCastOrNull<TokenIdType const*>(outputs->outputIdsPtr), endIdsPtr, finishedStatePtr,
        sequenceLengthsPtr, numNewTokens, batchSlotsPtr, decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(),
        decoderDomain.getMaxDecodingTokens(), stream);
    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class StopCriteriaLayer<float>;
template class StopCriteriaLayer<half>;

} // namespace tensorrt_llm::layers
