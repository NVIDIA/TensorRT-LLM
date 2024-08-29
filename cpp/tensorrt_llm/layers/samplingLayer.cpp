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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

#include "samplingLayer.h"
#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
SamplingLayer<T>::SamplingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(!mDecodingMode.isBeamSearch(), "SamplingLayer does not support Beam search mode");
    TLLM_CHECK_WITH_INFO(mDecodingMode.isTopKorTopP(), "SamplingLayer requires TopK or TopP mode");
    if (mDecodingMode.isTopK())
    {
        mSamplingLayers.emplace_back(std::make_unique<TopKSamplingLayer<T>>(decoderDomain, mBufferManager));
    }

    if (mDecodingMode.isTopP())
    {
        mSamplingLayers.emplace_back(
            std::make_unique<TopPSamplingLayer<T>>(decoderDomain, mBufferManager, /* deterministic */ true));
    }

    allocateBuffer(decoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void SamplingLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    size_t workspaceSize = 0;
    for (auto&& layer : mSamplingLayers)
    {
        workspaceSize = std::max(workspaceSize, layer->getWorkspaceSize());
    }

    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize, sizeof(curandState_t)}), TRTDataType<int8_t>::value);
    auto const batchSizeShape = ITensor::makeShape({batchSize});
    mRandomSeedsDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<uint64_t>::value);
    mSkipDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mSamplingWorkspaceDevice = mBufferManager->gpu(workspaceSize, TRTDataType<int8_t>::value);

    // host buffers.
    mSkipDecodeHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    TLLM_CHECK(mSkipDecodeHost != nullptr);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void SamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, BufferConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);

    // If runtime argument has single random seed, using this random seed to
    // initialize the random table of all sentences. If the argument has
    // [batchSize] random seeds, initializing the random table by different
    // random seeds respectively. If no random seed, initialize the random table
    // of all sentences by 0 directly.
    auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);
    if (setupParams->randomSeed)
    {
        auto curandStateDevicePtr = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
        if (setupParams->randomSeed->size() == 1)
        {
            invokeCurandInitialize(
                curandStateDevicePtr, batchSlotsPtr, batchSize, setupParams->randomSeed->front(), getStream());
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(setupParams->randomSeed->size() == batchSize, "Random seed vector size mismatch.");
            auto randomSeedsDevicePtr = bufferCast<uint64_t>(*mRandomSeedsDevice);
            cudaAutoCpy(randomSeedsDevicePtr, setupParams->randomSeed->data(), batchSize, getStream());
            invokeCurandBatchInitialize(
                curandStateDevicePtr, batchSlotsPtr, batchSize, randomSeedsDevicePtr, getStream());
            sync_check_cuda_error();
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        auto curandStatesDevicePtr = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
        invokeCurandInitialize(curandStatesDevicePtr, batchSlotsPtr, batchSize, 0, getStream());
    }

    if (setupParams->outputLogProbs)
    {
        // FIXME(nkorobov): monotonically growing
        mOutputLogProbs = std::any_of(setupParams->outputLogProbs->begin(), setupParams->outputLogProbs->end(),
            [this](bool outputLogProbs) { return this->mOutputLogProbs | outputLogProbs; });
    }

    if (setupParams->cumLogProbs)
    {
        // FIXME(nkorobov): monotonically growing
        mCumLogProbs = std::any_of(setupParams->cumLogProbs->begin(), setupParams->cumLogProbs->end(),
            [this](bool cumLogProbs) { return this->mCumLogProbs | cumLogProbs; });
    }

    for (auto&& layer : mSamplingLayers)
    {
        layer->setup(batchSize, beamWidth, batchSlots, setupParams);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void SamplingLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto logits = bufferCast<T>(*inputs->logits.value());
    auto endIds = bufferCast<TokenIdType>(*inputs->endIds);
    auto batchSlots = bufferCast<SizeType32>(*inputs->batchSlots);

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCast<FinishedState::UnderlyingType>(*inputs->finished.value()))
        : nullptr;

    auto const skipTopP = !mDecodingMode.isTopP();

    // Compute probabilities either for TopP or if cumLogProbs or outputLogProbs are specified
    bool const skipSoftMax = skipTopP && !mOutputLogProbs && !mCumLogProbs;

    inputs->curandStates = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    inputs->samplingWorkspace = mSamplingWorkspaceDevice->data();
    inputs->probsComputed = !skipSoftMax;
    if (!skipSoftMax)
    {
        invokeAddBiasSoftMax(logits, (T**) nullptr, logits, (T*) (nullptr), endIds, finishedInput, batchSlots,
            batchSize, mDecoderDomain.getBatchSize(), /* bw */ 1, mDecoderDomain.getVocabSize(),
            mDecoderDomain.getVocabSizePadded(), skipSoftMax, /* batchSlotLogits */ false, getStream());
        sync_check_cuda_error();
    }

    for (auto&& layer : mSamplingLayers)
    {
        layer->forwardAsync(outputs, baseInputs);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t SamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return mSamplingWorkspaceDevice->getSizeInBytes();
}

template class SamplingLayer<float>;
template class SamplingLayer<half>;

} // namespace tensorrt_llm::layers
