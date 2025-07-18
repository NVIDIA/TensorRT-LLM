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
#include "tensorrt_llm/common/nvtxUtils.h"
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
    NVTX3_SCOPED_RANGE(TopPSamplingLayer_setup);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);

    auto constexpr defaultTopPDecay = DefaultDecodingParams::getTopPDecay();
    auto constexpr defaultTopPMin = DefaultDecodingParams::getTopPMin(); // prevent TopP becoming 0.0

    auto const* batchSlotsHostPtr = bufferCastOrNull<SizeType32>(batchSlots);
    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    if (!setupParams->runtimeTopP.has_value() || setupParams->runtimeTopP.value().empty())
    {
        // Fast path to disable TopP for slots
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const bid = batchSlotsHostPtr[bi];
            skipDecodeHostPtr[bid] = true;
        }
        auto const maxBatchSize = mDecoderDomain.getBatchSize();
        auto skipDecodeHostSlice = IBuffer::slice(mSkipDecodeHost, 0, maxBatchSize);
        mBufferManager->copy(*skipDecodeHostSlice, *mSkipDecodeDevice);
        return;
    }

    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector{DefaultDecodingParams::getTopK()});
    auto runtimeTopP = setupParams->runtimeTopP.value();
    auto decayVec = setupParams->topPDecay.value_or(std::vector{defaultTopPDecay});
    auto topPMinVec = setupParams->topPMin.value_or(std::vector{defaultTopPMin});
    auto topPResetIdsVec = setupParams->topPResetIds.value_or(std::vector{DefaultDecodingParams::getTopPResetId()});

    auto const paramsSize
        = expandMatchElements(batchSize, runtimeTopK, runtimeTopP, decayVec, topPMinVec, topPResetIdsVec);
    TLLM_CHECK_WITH_INFO(paramsSize != 0,
        fmtstr("TopPSamplingLayer got parameter with unexpected size, want 1 or batchSize(%d), got"
               "runtimeTopK.size() = %zu, "
               "runtimeTopP.size() = %zu, "
               "topPDecay.size() = %zu, "
               "topPMin.size() = %zu, "
               "topPResetIds.size() = %zu",
            batchSize, runtimeTopK.size(), runtimeTopP.size(), decayVec.size(), topPMinVec.size(),
            topPResetIdsVec.size()));

    for (size_t i = 0; i < paramsSize; ++i)
    {
        // support topK up to TOP_K_MAX.
        auto& topK = runtimeTopK[i];
        auto& topP = runtimeTopP[i];
        clampTopK(topK);
        clampTopP(topP);
        regularizeTopKTopP(topK, topP);

        auto& decay = decayVec[i];
        if (decay <= 0.f || decay > 1.0f)
        {
            TLLM_LOG_WARNING(
                "Decay (%f) is out of range ((0.0, 1.0f]). Change to default (%f).", decay, defaultTopPDecay);
            decay = defaultTopPDecay;
        }

        auto& topPMin = topPMinVec[i];
        if (topPMin <= 0.f || topPMin > 1.0f)
        {
            TLLM_LOG_WARNING(
                "TopP min (%f) is out of range ([0.0, 1.0f]). Change to default (%f).", topPMin, defaultTopPMin);
            topPMin = defaultTopPMin;
        }
    }

    // Update parameters on both device and host, so we can
    // determine whether we can skip launch kernel by examine mSkipDecodeHost
    // without consulting device memory, or we'll have to do an expensive synchronization.
    SizeType32* topKsPtr = nullptr;
    float* topPsPtr = nullptr;
    float* topPDecayPtr = nullptr;
    float* topPMinPtr = nullptr;
    SizeType32* topPResetIdsPtr = nullptr;

    if (paramsSize > 1)
    {
        auto initWorkspaceSizes = getTopPInitWorkspaceSizes(batchSize);
        std::vector<void*> alignedPointers;
        calcAlignedPointers(workspace->getRawWorkspaceDevicePtr(), initWorkspaceSizes)(
            topKsPtr, topPsPtr, topPDecayPtr, topPMinPtr, topPResetIdsPtr);
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopK, IBuffer::wrap(topKsPtr, initWorkspaceSizes[0] / sizeof(*topKsPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopP, IBuffer::wrap(topPsPtr, initWorkspaceSizes[1] / sizeof(*topPsPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, decayVec, IBuffer::wrap(topPDecayPtr, initWorkspaceSizes[2] / sizeof(*topPDecayPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, topPMinVec, IBuffer::wrap(topPMinPtr, initWorkspaceSizes[3] / sizeof(*topPMinPtr)));
        DecodingLayerWorkspace::copyToWorkspace(*mBufferManager, topPResetIdsVec,
            IBuffer::wrap(topPResetIdsPtr, initWorkspaceSizes[4] / sizeof(*topPResetIdsPtr)));
    }

    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();
    auto* skipDecodeDevicePtr = bufferCastOrNull<bool>(mSkipDecodeDevice);
    auto* initialTopPDevicePtr = bufferCast<float>(*mInitialTopPDevice);
    invokeSetTopPRuntimeArgs(batchSize,                                               //
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKDevice)}, //
        {topPsPtr, runtimeTopP.front(), bufferCast<float>(*mRuntimeTopPDevice)},      //
        skipDecodeDevicePtr, initialTopPDevicePtr, batchSlotsDevicePtr, true, getStream());

    invokeScatterDecodingParams(topPDecayPtr, decayVec.front(), bufferCast<float>(*mTopPDecayDevice),
        batchSlotsDevicePtr, batchSize, getStream());
    invokeScatterDecodingParams(topPMinPtr, topPMinVec.front(), bufferCast<float>(*mTopPMinDevice), batchSlotsDevicePtr,
        batchSize, getStream());
    invokeScatterDecodingParams(topPResetIdsPtr, topPResetIdsVec.front(), bufferCast<TokenIdType>(*mTopPResetIdsDevice),
        batchSlotsDevicePtr, batchSize, getStream());

    topKsPtr = paramsSize > 1 ? runtimeTopK.data() : nullptr;
    invokeSetTopPRuntimeArgs(batchSize,               //
        {topKsPtr, runtimeTopK.front(), nullptr}, {}, //
        skipDecodeHostPtr, nullptr, batchSlotsHostPtr, false);

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
    NVTX3_SCOPED_RANGE(TopPSamplingLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    auto const skip = allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
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
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
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

    sync_check_cuda_error(getStream());
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
