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

#include "banWordsLayer.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/banBadWords.h"
#include "tensorrt_llm/kernels/banRepeatNgram.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
BanWordsLayer<T>::BanWordsLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    allocateBuffer();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseNoRepeatNgramSize())
    {
        mNoRepeatNgramSizeDevice
            = mBufferManager->gpu(ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value);
    }

    mNoRepeatNgramSize = mBufferManager->pinnedPool(
        ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(BanWordsLayer_setup);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);
    auto const& banWordsParams = setupParams->banWordsParams;
    TLLM_CHECK_WITH_INFO(banWordsParams, "banWordsParams for setup is not set");
    bool const useNoRepeatNgramSize
        = mDecodingMode.isUseNoRepeatNgramSize() && banWordsParams->noRepeatNgramSize.has_value();
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};
    mUseNoRepeatNgramSize |= useNoRepeatNgramSize;
    if (mUseNoRepeatNgramSize)
    {
        fillBuffers(banWordsParams->noRepeatNgramSize, DefaultDecodingParams::getNoRepeatNgramSize(),
            mNoRepeatNgramSize, mNoRepeatNgramSizeDevice, batchSlots,
            std::make_pair(0.f, std::numeric_limits<float>::max()), "no_repeat_ngram_size");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::banRepeatNGrams(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots, BufferPtr noRepeatNgramSizeDevice,
    DecoderDomain const& decoderDomain, SizeType32 maxSeqLen, bool useNoRepeatNgramSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (useNoRepeatNgramSize)
    {
        // auto const maxStep = inputs->step; // TODO Should we use step? but current inputs->step is always 0.
        auto const maxStep = maxSeqLen;
        // Temporary variables to store dereferenced inputs
        auto logitsPtr = bufferCast<T>(*logits);
        auto outputIdsPtr = bufferCast<TokenIdType const*>(*outputs->outputIdsPtr);
        auto finishedPtr
            = reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished));
        auto parentIdsPtr = bufferCast<SizeType32 const*>(*outputs->parentIdsPtr);
        auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);
        auto sequenceLengthPtr = bufferCast<SizeType32>(*outputs->sequenceLength.value());
        auto noRepeatNgramSizeDevicePtr = bufferCastOrNull<SizeType32>(noRepeatNgramSizeDevice);

        // Call to invokeBanRepeatNgram with dereferenced inputs
        invokeBanRepeatNgram(logitsPtr, outputIdsPtr, finishedPtr, parentIdsPtr, batchSlotsPtr, sequenceLengthPtr,
            decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), maxSeqLen, noRepeatNgramSizeDevicePtr,
            decoderDomain.getVocabSizePadded(), maxStep, getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::banBadWords(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxBadWordsLength = inputs->banWordsInputs->maxBadWordsLen;
    if (maxBadWordsLength != 0)
    {
        // Temporary variables to store dereferenced inputs
        auto badWordsPtr = bufferCast<TokenIdType const*>(*inputs->banWordsInputs->badWordsPtr.value());
        auto badWordsLens = bufferCast<SizeType32>(*inputs->banWordsInputs->badWordsLengths.value());
        auto logitsPtr = bufferCast<T>(*logits);
        auto outputIdsPtr = bufferCast<TokenIdType const*>(*outputs->outputIdsPtr);
        auto parentIdsPtr
            = decoderDomain.getBeamWidth() > 1 ? bufferCast<SizeType32 const*>(*outputs->parentIdsPtr) : nullptr;
        auto sequenceLengthPtr = bufferCast<SizeType32>(*outputs->sequenceLength.value());
        auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);

        // Call to invokeBanBadWords with dereferenced inputs
        invokeBanBadWords(logitsPtr, outputIdsPtr, parentIdsPtr, batchSlotsPtr, decoderDomain.getBatchSize(),
            decoderDomain.getBeamWidth(), badWordsPtr, badWordsLens, maxBadWordsLength,
            decoderDomain.getVocabSizePadded(), sequenceLengthPtr, maxSeqLen, getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(BanWordsLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);

    TLLM_CHECK_WITH_INFO(inputs->banWordsInputs, "banWordsInputs for forward is not set");

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds->getDimension<-1>();

    banRepeatNGrams(workspace->getDeviceRuntimeLogits(), outputs, inputs, workspace->getDeviceBatchSlots(),
        mNoRepeatNgramSizeDevice, localDecoderDomain, maxSeqLen, mUseNoRepeatNgramSize);
    banBadWords(workspace->getDeviceRuntimeLogits(), outputs, inputs, workspace->getDeviceBatchSlots(),
        localDecoderDomain, maxSeqLen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BanWordsLayer<float>;
template class BanWordsLayer<half>;

} // namespace tensorrt_llm::layers
