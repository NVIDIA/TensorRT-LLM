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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "topPSamplingLayer.h"

#include <algorithm>
#include <float.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

static __global__ void setTopPRuntimeArgs(SizeType32 batchSize, SizeType32 topK, SizeType32* topKs,
    SizeType32 topKsSize, float topP, float* topPs, SizeType32 topPsSize, bool* skipDecode,
    SizeType32 const* batchSlots, float* initialTopPBuf)
{
    /**
     * @brief Setup the runtime arguments for topp, broadcasting top_p to top_ps
              and top_k to top_ks.
     */

    auto index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    for (SizeType32 bi = index; bi < batchSize; bi += static_cast<SizeType32>(gridDim.x * blockDim.x))
    {
        auto const batchSlot = batchSlots != nullptr ? batchSlots[bi] : bi;
        auto k = topKsSize > 1 ? topKs[batchSlot] : topK;
        auto const p = topPsSize > 1 ? topPs[batchSlot] : topP;
        if (k == 0 && p == 0.0f)
        {
            // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
            // equivalent to greedy search. So, we set the topk = 1 as an alternative
            // solution.
            k = 1;
        }
        topKs[batchSlot] = k;
        topPs[batchSlot] = p;
        skipDecode[batchSlot] = k > 0;

        initialTopPBuf[batchSlot] = topPs[batchSlot];
    }
}

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager, bool isDeterministic, bool isAirTopP)
    : BaseLayer(decoderDomain, bufferManager)
    , mIsDeterministic(isDeterministic)
    , mIsAirTopP(isAirTopP)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    int deviceId;
    tc::check_cuda_error(cudaGetDevice(&deviceId)); // Get the correct device id
    tc::check_cuda_error(cudaGetDeviceProperties(&mDeviceProp, deviceId));

    allocateBuffer(mDecoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mIsAirTopP == false)
    {
        mWorkspaceSize = getTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());
    }
    else
    {
        mWorkspaceSize = getAirTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded(), mIsDeterministic);
    }

    mRuntimeTopKDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<SizeType32>::value);
    mRuntimeTopPDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<float>::value);
    mInitialTopPDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<float>::value);
    mTopPDecayDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<float>::value);
    mTopPMinDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<float>::value);
    mTopPResetIdsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<TokenIdType>::value);
    mSkipDecodeDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<bool>::value);
    mSkipDecodeHost = mBufferManager->pinnedPool(ITensor::makeShape({batchSize}), TRTDataType<bool>::value);
    auto skipDecodeHostRange = BufferRange<bool>(*mSkipDecodeHost);
    std::fill(skipDecodeHostRange.begin(), skipDecodeHostRange.end(), true);

    auto workspaceSize = std::max({mRuntimeTopKDevice->getSizeInBytes(), mRuntimeTopPDevice->getSizeInBytes(),
        mInitialTopPDevice->getSizeInBytes(), mTopPDecayDevice->getSizeInBytes(), mTopPMinDevice->getSizeInBytes(),
        mTopPResetIdsDevice->getSizeInBytes(), mSkipDecodeDevice->getSizeInBytes()});
    mSetupWorkspaceDevice
        = mBufferManager->gpu(ITensor::makeShape({static_cast<int64_t>(workspaceSize)}), TRTDataType<int8_t>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::setup(SizeType32 const batchSize, SizeType32 const beamWidth, BufferConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
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

    auto batchSlotsPtr = bufferCastOrNull<SizeType32>(batchSlots);
    auto skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    if (runtimeTopPSize == 0)
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto bid = bi;
            if (batchSlotsPtr)
            {
                bid = batchSlotsPtr[bi];
            }
            skipDecodeHostPtr[bid] = true;
        }
        auto const batchSize = mDecoderDomain.getBatchSize();
        auto skipDecodeHostSlice = IBuffer::slice(mSkipDecodeHost, 0, batchSize);
        mBufferManager->copy(*skipDecodeHostSlice, *mSkipDecodeDevice);
        return;
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

    auto setupWorkspaceDevicePtr = reinterpret_cast<SizeType32*>(bufferCastOrNull<int8_t>(mSetupWorkspaceDevice));
    auto setupWorkspaceDeviceAsFloatPtr = reinterpret_cast<float*>(setupWorkspaceDevicePtr);
    auto runtimeTopKDevicePtr = bufferCastOrNull<SizeType32>(mRuntimeTopKDevice);
    if (runtimeTopKSize > 1)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(runtimeTopK.size()) == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopK.size(), batchSize));
        mBufferManager->copy(runtimeTopK.data(), *mSetupWorkspaceDevice, runtime::MemoryType::kCPU);
        invokeScatterDecodingParams(
            setupWorkspaceDevicePtr, runtimeTopKDevicePtr, batchSlotsPtr, batchSize, getStream());
    }
    auto runtimeTopPDevicePtr = bufferCastOrNull<float>(mRuntimeTopPDevice);
    if (runtimeTopPSize > 1)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(runtimeTopP.size()) == batchSize,
            fmtstr("runtimeTopP.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopP.size(), batchSize));
        mBufferManager->copy(runtimeTopP.data(), *mSetupWorkspaceDevice, runtime::MemoryType::kCPU);
        invokeScatterDecodingParams(
            setupWorkspaceDeviceAsFloatPtr, runtimeTopPDevicePtr, batchSlotsPtr, batchSize, getStream());
    }

    auto fillBuffers = [this, batchSize, batchSlotsPtr](
                           std::string name, auto const& vector, auto deviceTmpBuffer, auto deviceBuffer)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(vector.size()) == batchSize,
            fmtstr("%s.size() (%lu) == batchSize (%d) is not satisfied!", name.c_str(), vector.size(), batchSize));
        cudaAutoCpy(deviceTmpBuffer, vector.data(), batchSize, getStream());
        invokeScatterDecodingParams(deviceTmpBuffer, deviceBuffer, batchSlotsPtr, batchSize, getStream());
    };

    auto topPDecayDevicePtr = bufferCastOrNull<float>(mTopPDecayDevice);
    fillBuffers("topPDecay", decayVec, setupWorkspaceDeviceAsFloatPtr, topPDecayDevicePtr);

    auto topPMinDevicePtr = bufferCastOrNull<float>(mTopPMinDevice);
    fillBuffers("topPMin", topPMinVec, setupWorkspaceDeviceAsFloatPtr, topPMinDevicePtr);

    auto topPRestIdsDevicePtr = bufferCastOrNull<SizeType32>(mTopPResetIdsDevice);
    fillBuffers("topPResetIds", topPResetIdsVec, setupWorkspaceDevicePtr, topPRestIdsDevicePtr);

    {
        auto skipDecodeDevicePtr = bufferCastOrNull<bool>(mSkipDecodeDevice);
        auto initialTopPDevicePtr = bufferCastOrNull<float>(mInitialTopPDevice);
        dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
        dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
        setTopPRuntimeArgs<<<grid, block, 0, getStream()>>>(batchSize, topK, runtimeTopKDevicePtr, runtimeTopKSize,
            topP, runtimeTopPDevicePtr, runtimeTopPSize, skipDecodeDevicePtr, batchSlotsPtr, initialTopPDevicePtr);
        sync_check_cuda_error();
    }

    auto const skipHostDecodeDeviceSlice = ITensor::slice(mSkipDecodeDevice, 0, mDecoderDomain.getBatchSize());
    auto skipDecodeHostSlice = ITensor::slice(mSkipDecodeHost, 0, mDecoderDomain.getBatchSize());
    mBufferManager->copy(*skipHostDecodeDeviceSlice, *skipDecodeHostSlice);
    std::vector<float> runtimeTopPs(mDecoderDomain.getBatchSize());
    auto const runtimeTopPDeviceSlice = ITensor::slice(mRuntimeTopPDevice, 0, mDecoderDomain.getBatchSize());
    mBufferManager->copy(*runtimeTopPDeviceSlice, runtimeTopPs.data(), runtime::MemoryType::kCPU);
    {
        auto maxTopP = 0.f;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const bid = batchSlotsPtr ? batchSlotsPtr[bi] : bi;
            maxTopP = std::max(maxTopP, runtimeTopPs[bid]);
        }
        mRuntimeMaxTopP = std::max(mRuntimeMaxTopP, maxTopP);
    }

    if (mIsAirTopP == true)
    {
        auto smCnt = mDeviceProp.multiProcessorCount;
        if (smCnt <= 0)
        {
            int deviceId;
            check_cuda_error(cudaGetDevice(&deviceId)); // Get the correct device id
            cudaDeviceProp prop;
            check_cuda_error(cudaGetDeviceProperties(&prop, deviceId));
            smCnt = prop.multiProcessorCount;
        }
        mAirTopPBlockNum
            = calcAirTopPBlockNum<T>(batchSize, (int) mDecoderDomain.getVocabSizePadded(), smCnt, mIsDeterministic);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto batchSlotsHost
        = inputs->batchSlots ? inputs->batchSlots.value() : getDefaultBatchSlots(batchSize, *mBufferManager);
    auto skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    auto const skip = allOfBatchSlots(bufferCastOrNull<SizeType32>(batchSlotsHost), skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    // Probabilities must be already computed instead of logits
    auto probs = bufferCastOrNull<T>(inputs->logits);
    auto endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);
    auto batchSlots = bufferCastOrNull<SizeType32>(inputs->batchSlots);
    auto curandStatesDevice = inputs->curandStates;
    auto samplingWorkspaceDevice = inputs->samplingWorkspace;

    TLLM_CHECK_WITH_INFO(curandStatesDevice, "No curand states provided");
    TLLM_CHECK_WITH_INFO(samplingWorkspaceDevice, "No sampling workspace provided");

    auto finishedInput = (inputs->finished) ? reinterpret_cast<FinishedState const*>(
                             bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished.value()))
                                            : nullptr;
    auto finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished.value()))
        : nullptr;

    auto cumLogProbs = bufferCastOrNull<float>(outputs->cumLogProbs);
    auto outputLogProbs = bufferCastOrNull<float>(outputs->outputLogProbsTiled);
    auto sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);

    TopPSamplingKernelParams<T> params;
    params.probs = probs;
    params.outputIds = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    params.workspace = samplingWorkspaceDevice;
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.sequenceLength = sequenceLength;
    params.endIds = endIds;
    params.batchSlots = batchSlots;
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipDecodeDevice);
    params.cumLogProbs = cumLogProbs;
    params.outputLogProbs = outputLogProbs;
    params.curandState = curandStatesDevice;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    if (mIsAirTopP == false)
    {
        invokeBatchTopPSampling<T>(params, getStream());
        sync_check_cuda_error();
    }
    else
    {
        params.blockNum = mAirTopPBlockNum;
        params.isDeterministic = mIsDeterministic;
        invokeBatchAirTopPSampling<T>(params, getStream());
        sync_check_cuda_error();
    }

    sync_check_cuda_error();
    auto runtimeTopPDevicePtr = bufferCastOrNull<float>(mRuntimeTopPDevice);
    auto initialTopPDevicePtr = bufferCastOrNull<float>(mInitialTopPDevice);
    auto topPDecayDevicePtr = bufferCastOrNull<float>(mTopPDecayDevice);
    auto topPMinDevicePtr = bufferCastOrNull<float>(mTopPMinDevice);
    auto topPResetIdsDevice = bufferCastOrNull<TokenIdType>(mTopPResetIdsDevice);
    auto outputIdsPtr = bufferCastOrNull<TokenIdType const*>(outputs->outputIdsPtr);
    invokeComputeToppDecay(runtimeTopPDevicePtr, initialTopPDevicePtr, outputIdsPtr, topPDecayDevicePtr,
        topPMinDevicePtr, topPResetIdsDevice, sequenceLength, batchSlots, batchSize, getStream());
    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t TopPSamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

} // namespace tensorrt_llm::layers
