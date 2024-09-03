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

#include "topPSamplingLayer.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

#include <algorithm>
#include <cfloat>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager, bool isDeterministic, bool isAirTopP)
    : BaseLayer(decoderDomain, bufferManager)
    , mIsDeterministic(isDeterministic)
    , mIsAirTopP(isAirTopP)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const deviceId = getDevice();
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProp, deviceId));

    allocateBuffer(mDecoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (!mIsAirTopP)
    {
        mWorkspaceSize = getTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());
    }
    else
    {
        mWorkspaceSize = getAirTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded(), mIsDeterministic);
    }

    auto const batchSizeShape = ITensor::makeShape({batchSize});
    mRuntimeTopKDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mRuntimeTopPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mInitialTopPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mTopPDecayDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mTopPMinDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mTopPResetIdsDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<TokenIdType>::value);
    mSkipDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mSkipDecodeHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    auto skipDecodeHostRange = BufferRange<bool>(*mSkipDecodeHost);
    std::fill(skipDecodeHostRange.begin(), skipDecodeHostRange.end(), true);

    mSetupWorkspaceSize = std::max({mRuntimeTopKDevice->getSizeInBytes(), mRuntimeTopPDevice->getSizeInBytes(),
        mInitialTopPDevice->getSizeInBytes(), mTopPDecayDevice->getSizeInBytes(), mTopPMinDevice->getSizeInBytes(),
        mTopPResetIdsDevice->getSizeInBytes(), mSkipDecodeDevice->getSizeInBytes()});

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);

    auto const defaultTopK = DefaultDecodingParams::getTopK();
    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector<SizeType32>(batchSize, defaultTopK));
    auto runtimeTopP = setupParams->runtimeTopP.value_or(std::vector<float>{});

    auto const runtimeTopKSize = runtimeTopK.size();
    auto const runtimeTopPSize = runtimeTopP.size();

    auto const defaultTopPDecay = DefaultDecodingParams::getTopPDecay();
    auto decayVec = setupParams->topPDecay.value_or(std::vector<float>(batchSize, defaultTopPDecay));

    auto const defaultTopPMin = DefaultDecodingParams::getTopPMin(); // prevent TopP becoming 0.0
    auto topPMinVec = setupParams->topPMin.value_or(std::vector<float>(batchSize, defaultTopPMin));

    auto const defaultTopPResetId = DefaultDecodingParams::getTopPResetId();
    auto topPResetIdsVec = setupParams->topPResetIds.value_or(std::vector<TokenIdType>(batchSize, defaultTopPResetId));

    auto const* batchSlotsPtr = bufferCastOrNull<SizeType32>(batchSlots);
    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    if (runtimeTopPSize == 0)
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const bid = batchSlotsPtr[bi];
            skipDecodeHostPtr[bid] = true;
        }
        auto const batchSize = mDecoderDomain.getBatchSize();
        auto skipDecodeHostSlice = IBuffer::slice(mSkipDecodeHost, 0, batchSize);
        mBufferManager->copy(*skipDecodeHostSlice, *mSkipDecodeDevice);
        return;
    }

    for (auto& topK : runtimeTopK)
    {
        if (topK < 0 || topK > TOP_K_MAX)
        {
            TLLM_LOG_WARNING(
                "TopK (%d) is larger than max supported number (%d). Clip to max supported number.", topK, TOP_K_MAX);
            topK = std::clamp(topK, 0, static_cast<SizeType32>(TOP_K_MAX));
        }
    }

    for (auto& topP : runtimeTopP)
    {
        if (topP < 0.f || topP > 1.0f)
        {
            TLLM_LOG_WARNING("TopP (%f) is out of range ([0.0, 1.0f]). Clip to closest number.", topP);
            topP = std::clamp(topP, 0.f, 1.f);
        }
    }

    for (auto& decay : decayVec)
    {
        if (decay <= 0.f || decay > 1.0f)
        {
            TLLM_LOG_WARNING(
                "Decay (%f) is out of range ((0.0, 1.0f]). Change to default (%f).", decay, defaultTopPDecay);
            decay = defaultTopPDecay;
        }
    }

    for (auto& topPMin : topPMinVec)
    {
        if (topPMin <= 0.f || topPMin > 1.0f)
        {
            TLLM_LOG_WARNING(
                "TopP min (%f) is out of range ([0.0, 1.0f]). Change to default (%f).", topPMin, defaultTopPMin);
            topPMin = defaultTopPMin;
        }
    }

    auto const topK = runtimeTopK.at(0);
    auto const topP = runtimeTopP.at(0);

    auto* setupWorkspaceDevicePtr = workspace->getWorkspaceDevicePtrAs<SizeType32>();
    auto* setupWorkspaceDeviceAsFloatPtr = reinterpret_cast<float*>(setupWorkspaceDevicePtr);
    auto* runtimeTopKDevicePtr = bufferCastOrNull<SizeType32>(mRuntimeTopKDevice);
    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();
    if (runtimeTopKSize > 1)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(runtimeTopK.size()) == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopK.size(), batchSize));
        DecodingLayerWorkspace::copyToWorkspace(*mBufferManager, runtimeTopK, workspace->getWorkspaceDeviceBuffer());
        invokeScatterDecodingParams(
            setupWorkspaceDevicePtr, runtimeTopKDevicePtr, batchSlotsDevicePtr, batchSize, getStream());
    }
    auto* runtimeTopPDevicePtr = bufferCast<float>(*mRuntimeTopPDevice);
    if (runtimeTopPSize > 1)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(runtimeTopP.size()) == batchSize,
            fmtstr("runtimeTopP.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopP.size(), batchSize));
        DecodingLayerWorkspace::copyToWorkspace(*mBufferManager, runtimeTopP, workspace->getWorkspaceDeviceBuffer());
        invokeScatterDecodingParams(
            setupWorkspaceDeviceAsFloatPtr, runtimeTopPDevicePtr, batchSlotsDevicePtr, batchSize, getStream());
    }

    auto fillBuffers = [this, batchSize, batchSlotsDevicePtr](
                           std::string const& name, auto const& vector, auto deviceTmpBuffer, auto deviceBuffer)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(vector.size()) == batchSize,
            fmtstr("%s.size() (%lu) == batchSize (%d) is not satisfied!", name.c_str(), vector.size(), batchSize));
        cudaAutoCpy(deviceTmpBuffer, vector.data(), batchSize, getStream());
        invokeScatterDecodingParams(deviceTmpBuffer, deviceBuffer, batchSlotsDevicePtr, batchSize, getStream());
    };

    auto* topPDecayDevicePtr = bufferCastOrNull<float>(mTopPDecayDevice);
    fillBuffers("topPDecay", decayVec, setupWorkspaceDeviceAsFloatPtr, topPDecayDevicePtr);

    auto* topPMinDevicePtr = bufferCastOrNull<float>(mTopPMinDevice);
    fillBuffers("topPMin", topPMinVec, setupWorkspaceDeviceAsFloatPtr, topPMinDevicePtr);

    auto* topPRestIdsDevicePtr = bufferCastOrNull<SizeType32>(mTopPResetIdsDevice);
    fillBuffers("topPResetIds", topPResetIdsVec, setupWorkspaceDevicePtr, topPRestIdsDevicePtr);

    auto* skipDecodeDevicePtr = bufferCast<bool>(*mSkipDecodeDevice);
    auto* initialTopPDevicePtr = bufferCast<float>(*mInitialTopPDevice);
    invokeSetTopPRuntimeArgs(batchSize, topK, runtimeTopKDevicePtr, runtimeTopKSize, topP, runtimeTopPDevicePtr,
        runtimeTopPSize, skipDecodeDevicePtr, batchSlotsDevicePtr, initialTopPDevicePtr, getStream());

    auto const skipHostDecodeDeviceSlice = ITensor::slice(mSkipDecodeDevice, 0, mDecoderDomain.getBatchSize());
    auto skipDecodeHostSlice = ITensor::slice(mSkipDecodeHost, 0, mDecoderDomain.getBatchSize());
    mBufferManager->copy(*skipHostDecodeDeviceSlice, *skipDecodeHostSlice);

    if (mIsAirTopP)
    {
        auto smCnt = mDeviceProp.multiProcessorCount;
        if (smCnt <= 0)
        {
            auto const deviceId = getDevice();
            cudaDeviceProp prop{};
            TLLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
            smCnt = prop.multiProcessorCount;
        }
        mAirTopPBlockNum
            = calcAirTopPBlockNum<T>(batchSize, mDecoderDomain.getVocabSizePadded(), smCnt, mIsDeterministic);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    auto const skip = allOfBatchSlots(bufferCast<SizeType32>(*inputs->batchSlots), skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    // Probabilities must be already computed instead of logits
    auto probs = bufferCastOrNull<T>(inputs->logits);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    auto const* finishedInput = (inputs->finished) ? reinterpret_cast<FinishedState const*>(
                                    bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished.value()))
                                                   : nullptr;
    auto* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished.value()))
        : nullptr;

    auto* cumLogProbs = bufferCastOrNull<float>(outputs->cumLogProbs);
    auto* outputLogProbs = bufferCastOrNull<float>(outputs->outputLogProbsTiled);
    auto* sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);

    TopPSamplingKernelParams<T> params{};
    params.probs = probs;
    params.outputIds = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.sequenceLength = sequenceLength;
    params.endIds = endIds;
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipDecodeDevice);
    params.cumLogProbs = cumLogProbs;
    params.outputLogProbs = outputLogProbs;
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    if (!mIsAirTopP)
    {
        invokeBatchTopPSampling<T>(params, getStream());
    }
    else
    {
        params.blockNum = mAirTopPBlockNum;
        params.isDeterministic = mIsDeterministic;
        invokeBatchAirTopPSampling<T>(params, getStream());
    }

    sync_check_cuda_error();
    auto* runtimeTopPDevicePtr = bufferCastOrNull<float>(mRuntimeTopPDevice);
    auto* initialTopPDevicePtr = bufferCastOrNull<float>(mInitialTopPDevice);
    auto* topPDecayDevicePtr = bufferCastOrNull<float>(mTopPDecayDevice);
    auto* topPMinDevicePtr = bufferCastOrNull<float>(mTopPMinDevice);
    auto* topPResetIdsDevice = bufferCastOrNull<TokenIdType>(mTopPResetIdsDevice);
    auto* outputIdsPtr = bufferCastOrNull<TokenIdType const*>(outputs->outputIdsPtr);
    invokeComputeToppDecay(runtimeTopPDevicePtr, initialTopPDevicePtr, outputIdsPtr, topPDecayDevicePtr,
        topPMinDevicePtr, topPResetIdsDevice, sequenceLength, workspace->getDeviceBatchSlotsPtr(), batchSize,
        getStream());
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t TopPSamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mSetupWorkspaceSize, mWorkspaceSize);
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

} // namespace tensorrt_llm::layers
