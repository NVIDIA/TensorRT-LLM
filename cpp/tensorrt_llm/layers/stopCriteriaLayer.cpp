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
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"
#include "tensorrt_llm/layers/layerUtils.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
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
    std::shared_ptr<BaseSetupParams> setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::forwardAsync(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<DynamicDecodeInputParams>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<DynamicDecodeOutputParams>(baseOutputs);

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const maxSeqLen = outputs->output_ids.shape[outputs->output_ids.shape.size() - 1];
    auto batchSlots = inputs->batch_slots ? inputs->batch_slots->template getPtr<SizeType32 const>() : nullptr;

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
void StopCriteriaLayer<T>::checkStopWordsStopCriteria(std::shared_ptr<DynamicDecodeOutputParams>& outputs,
    std::shared_ptr<DynamicDecodeInputParams> const& inputs, SizeType32 const* batchSlots,
    DecoderDomain const& decoderDomain, SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxStopWordsLength = inputs->max_stop_words_len;
    if (maxStopWordsLength)
    {
        auto numNewTokens = outputs->speculativeDecodingOutputs
            ? outputs->speculativeDecodingOutputs->acceptedLengths.template getPtr<SizeType32>()
            : nullptr;
        invokeStopWordsCriterion(outputs->output_ids_ptr.template getPtr<TokenIdType const*>(),
            outputs->parent_ids_ptr.template getPtr<SizeType32 const*>(),
            inputs->stop_words_ptr->template getPtr<TokenIdType const*>(),
            reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>()),
            outputs->sequence_length->template getPtr<SizeType32>(), batchSlots,
            inputs->stop_words_lengths->template getPtr<SizeType32 const>(), numNewTokens, maxStopWordsLength,
            decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), maxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkMaxLengthStopCriteria(std::shared_ptr<DynamicDecodeOutputParams>& outputs,
    std::shared_ptr<DynamicDecodeInputParams> const& inputs, SizeType32 const* batchSlots,
    DecoderDomain const& decoderDomain, SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (inputs->sequence_limit_length)
    {
        invokeLengthCriterion(
            reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>()),
            outputs->finished_sum ? outputs->finished_sum->template getPtr<SizeType32>() : nullptr,
            inputs->sequence_limit_length->template getPtr<SizeType32 const>(),
            outputs->sequence_length->template getPtr<SizeType32>(), batchSlots, decoderDomain.getBatchSize(),
            decoderDomain.getBeamWidth(), stream);
        sync_check_cuda_error();
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkEosToken(std::shared_ptr<DynamicDecodeOutputParams>& outputs,
    std::shared_ptr<DynamicDecodeInputParams> const& inputs, SizeType32 const* batchSlots,
    DecoderDomain const& decoderDomain, SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    invokeExplicitEOSCriterion(outputs->output_ids_ptr.template getPtr<TokenIdType const*>(),
        inputs->end_ids.template getPtr<TokenIdType const>(),
        reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>()),
        outputs->sequence_length->template getPtr<SizeType32>(),
        // FIXME(nkorobov): add tokens per step tensor when necessary
        /* tokensPerStep */ nullptr, batchSlots, decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(),
        decoderDomain.getMaxDecodingTokens(), stream);
    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class StopCriteriaLayer<float>;
template class StopCriteriaLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
