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

#include "tensorrt_llm/layers/banWordsLayer.h"
#include "tensorrt_llm/kernels/banBadWords.h"
#include "tensorrt_llm/kernels/banRepeatNgram.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
BanWordsLayer<T>::BanWordsLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    initialize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
BanWordsLayer<T>::~BanWordsLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::initialize()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    allocateBuffer();

    mNoRepeatNgramSize.resize(mDecoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseNoRepeatNgramSize())
    {
        mNoRepeatNgramSizeDevice
            = mAllocator->reMalloc(mNoRepeatNgramSizeDevice, sizeof(SizeType32) * mDecoderDomain.getBatchSize(), false);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseNoRepeatNgramSize())
    {
        mAllocator->free((void**) (&mNoRepeatNgramSizeDevice));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);
    std::vector<SizeType32> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = batchSlots ? batchSlots : batchSlotsVec.data();
    auto const& banWordsParams = setupParams->banWordsParams;
    TLLM_CHECK_WITH_INFO(banWordsParams, "banWordsParams for setup is not set");
    bool const useNoRepeatNgramSize
        = mDecodingMode.isUseNoRepeatNgramSize() && banWordsParams->noRepeatNgramSize.has_value();
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mStream};
    mUseNoRepeatNgramSize |= useNoRepeatNgramSize;
    if (mUseNoRepeatNgramSize)
    {
        fillBuffers(banWordsParams->noRepeatNgramSize, DefaultDecodingParams::getNoRepeatNgramSize(),
            mNoRepeatNgramSize, mNoRepeatNgramSizeDevice, batchSlotsHost,
            std::make_pair(0.f, std::numeric_limits<float>::max()), "no_repeat_ngram_size");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::banRepeatNGrams(Tensor& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlots,
    SizeType32 const* noRepeatNgramSizeDevice, DecoderDomain const& decoderDomain, SizeType32 maxSeqLen,
    bool useNoRepeatNgramSize, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // auto const maxStep = inputs->step; // TODO (bhsueh) Should we use step? but current inputs->step is always 0.
    auto const maxStep = maxSeqLen;
    if (useNoRepeatNgramSize)
    {
        invokeBanRepeatNgram(logits.template getPtr<T>(), outputs->outputIdsPtr.template getPtr<TokenIdType const*>(),
            reinterpret_cast<FinishedState*>(
                inputs->finished.value_or(Tensor{}).template getPtr<FinishedState::UnderlyingType>()),
            outputs->parentIdsPtr.template getPtr<SizeType32 const*>(), batchSlots,
            outputs->sequenceLength->template getPtr<SizeType32>(), decoderDomain.getBatchSize(),
            decoderDomain.getBeamWidth(), maxSeqLen, noRepeatNgramSizeDevice, decoderDomain.getVocabSizePadded(),
            maxStep, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::banBadWords(Tensor& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, SizeType32 const* batchSlots, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxBadWordsLength = inputs->banWordsInputs->maxBadWordsLen;
    if (maxBadWordsLength)
    {
        auto const** badWordsPtr = inputs->banWordsInputs->badWordsPtr->template getPtr<TokenIdType const*>();
        auto const* badWordsLens = inputs->banWordsInputs->badWordsLengths->template getPtr<SizeType32>();

        invokeBanBadWords((T*) logits.template getPtr<T>(), outputs->outputIdsPtr.template getPtr<TokenIdType const*>(),
            decoderDomain.getBeamWidth() > 1 ? outputs->parentIdsPtr.template getPtr<SizeType32 const*>() : nullptr,
            batchSlots, decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), badWordsPtr, badWordsLens,
            maxBadWordsLength, decoderDomain.getVocabSizePadded(),
            outputs->sequenceLength->template getPtr<SizeType32>(), maxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);

    TLLM_CHECK_WITH_INFO(inputs->banWordsInputs, "banWordsInputs for forward is not set");

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds.shape[outputs->outputIds.shape.size() - 1];
    auto batchSlots = inputs->batchSlots ? inputs->batchSlots->template getPtr<SizeType32 const>() : nullptr;

    banRepeatNGrams(inputs->logits.value(), outputs, inputs, batchSlots, mNoRepeatNgramSizeDevice, localDecoderDomain,
        maxSeqLen, mUseNoRepeatNgramSize, mStream);
    banBadWords(inputs->logits.value(), outputs, inputs, batchSlots, localDecoderDomain, maxSeqLen, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BanWordsLayer<float>;
template class BanWordsLayer<half>;

} // namespace tensorrt_llm::layers
