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
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

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
TopPSamplingLayer<T>::TopPSamplingLayer(DecoderDomain const& decoderDomain, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator, bool isDeterministic, bool isAirTopP)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
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
TopPSamplingLayer<T>::~TopPSamplingLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

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

    std::array<size_t, 8> deviceBufferSizes = {sizeof(SizeType32) * batchSize, sizeof(float) * batchSize,
        sizeof(float) * batchSize, sizeof(float) * batchSize, sizeof(float) * batchSize,
        sizeof(TokenIdType) * batchSize, sizeof(bool) * batchSize, 0};
    deviceBufferSizes[7] = *std::max_element(&deviceBufferSizes[0], &deviceBufferSizes[6]);

    mRuntimeTopKDevice = mAllocator->reMalloc(mRuntimeTopKDevice, deviceBufferSizes[0], false);
    mRuntimeTopPDevice = mAllocator->reMalloc(mRuntimeTopPDevice, deviceBufferSizes[1], false);
    mInitialTopPDevice = mAllocator->reMalloc(mInitialTopPDevice, deviceBufferSizes[2], false);
    mTopPDecayDevice = mAllocator->reMalloc(mTopPDecayDevice, deviceBufferSizes[3], false);
    mTopPMinDevice = mAllocator->reMalloc(mTopPMinDevice, deviceBufferSizes[4], false);
    mTopPResetIdsDevice = mAllocator->reMalloc(mTopPResetIdsDevice, deviceBufferSizes[5], false);
    mSkipDecodeDevice = mAllocator->reMalloc(mSkipDecodeDevice, deviceBufferSizes[6], false);
    mSetupWorkspaceDevice = mAllocator->reMalloc(mSetupWorkspaceDevice, deviceBufferSizes[7], false);

    mSkipDecodeHost = static_cast<bool*>(std::realloc(mSkipDecodeHost, sizeof(bool) * batchSize));
    std::fill(mSkipDecodeHost, mSkipDecodeHost + batchSize, true);

    mAllocatedSize = std::accumulate(deviceBufferSizes.begin(), deviceBufferSizes.end(), 0);
    TLLM_LOG_DEBUG("topPSamplingLayer allocated %lu bytes on GPU", mAllocatedSize);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mAllocator->free((void**) (&mRuntimeTopKDevice));
    mAllocator->free((void**) (&mRuntimeTopPDevice));
    mAllocator->free((void**) (&mInitialTopPDevice));
    mAllocator->free((void**) (&mTopPDecayDevice));
    mAllocator->free((void**) (&mTopPMinDevice));
    mAllocator->free((void**) (&mTopPResetIdsDevice));
    mAllocator->free((void**) (&mSkipDecodeDevice));
    mAllocator->free((void**) (&mSetupWorkspaceDevice));
    std::free(mSkipDecodeHost);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::setup(SizeType32 const batchSize, SizeType32 const beamWidth, SizeType32 const* batchSlots,
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

    if (runtimeTopPSize == 0)
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto bid = bi;
            if (batchSlots)
            {
                bid = batchSlots[bi];
            }
            mSkipDecodeHost[bid] = true;
        }
        cudaAutoCpy(mSkipDecodeDevice, mSkipDecodeHost, mDecoderDomain.getBatchSize(), mStream);
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

    if (runtimeTopKSize > 1)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(runtimeTopK.size()) == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopK.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<SizeType32*>(mSetupWorkspaceDevice), runtimeTopK.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<SizeType32*>(mSetupWorkspaceDevice), mRuntimeTopKDevice, batchSlots, batchSize, mStream);
    }
    if (runtimeTopPSize > 1)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(runtimeTopP.size()) == batchSize,
            fmtstr("runtimeTopP.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopP.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<float*>(mSetupWorkspaceDevice), runtimeTopP.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<float*>(mSetupWorkspaceDevice), mRuntimeTopPDevice, batchSlots, batchSize, mStream);
    }

    auto fillBuffers
        = [this, &batchSize, &batchSlots](std::string name, auto const& vector, auto deviceTmpBuffer, auto deviceBuffer)
    {
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(vector.size()) == batchSize,
            fmtstr("%s.size() (%lu) == batchSize (%d) is not satisfied!", name.c_str(), vector.size(), batchSize));
        cudaAutoCpy(deviceTmpBuffer, vector.data(), batchSize, mStream);
        invokeScatterDecodingParams(deviceTmpBuffer, deviceBuffer, batchSlots, batchSize, mStream);
    };

    fillBuffers("topPDecay", decayVec, reinterpret_cast<float*>(mSetupWorkspaceDevice), mTopPDecayDevice);

    fillBuffers("topPMin", topPMinVec, reinterpret_cast<float*>(mSetupWorkspaceDevice), mTopPMinDevice);

    fillBuffers(
        "topPResetIds", topPResetIdsVec, reinterpret_cast<TokenIdType*>(mSetupWorkspaceDevice), mTopPResetIdsDevice);

    {
        dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
        dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
        setTopPRuntimeArgs<<<grid, block, 0, mStream>>>(batchSize, topK, mRuntimeTopKDevice, runtimeTopKSize, topP,
            mRuntimeTopPDevice, runtimeTopPSize, mSkipDecodeDevice, batchSlots, mInitialTopPDevice);
        sync_check_cuda_error();
    }

    cudaAutoCpy(mSkipDecodeHost, mSkipDecodeDevice, mDecoderDomain.getBatchSize(), mStream);
    std::vector<float> runtimeTopPs(mDecoderDomain.getBatchSize());
    cudaAutoCpy(runtimeTopPs.data(), mRuntimeTopPDevice, mDecoderDomain.getBatchSize(), mStream);
    {
        auto maxTopP = 0.f;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const bid = batchSlots ? batchSlots[bi] : bi;
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

    auto const batchSize = inputs->logits->shape[0];

    std::vector<int32_t> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = inputs->batchSlots ? inputs->batchSlots->template getPtr<int const>() : batchSlotsVec.data();
    auto const skip = allOfBatchSlots(batchSlotsHost, mSkipDecodeHost, batchSize, true);
    if (skip)
    {
        return;
    }

    // Probabilities must be already computed instead of logits
    auto probs = inputs->logits->template getPtr<T>();
    auto endIds = inputs->endIds.template getPtr<TokenIdType const>();
    auto batchSlots = inputs->batchSlots ? inputs->batchSlots->template getPtr<SizeType32 const>() : nullptr;
    auto curandStatesDevice = inputs->curandStates;
    auto samplingWorkspaceDevice = inputs->samplingWorkspace;

    TLLM_CHECK_WITH_INFO(curandStatesDevice, "No curand states provided");
    TLLM_CHECK_WITH_INFO(samplingWorkspaceDevice, "No sampling workspace provided");

    FinishedState* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState*>(inputs->finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;

    auto cumLogProbs = (outputs->cumLogProbs) ? outputs->cumLogProbs->template getPtr<float>() : nullptr;
    auto outputLogProbs
        = (outputs->outputLogProbsTiled) ? outputs->outputLogProbsTiled->template getPtr<float>() : nullptr;
    auto sequenceLength = (outputs->sequenceLength) ? outputs->sequenceLength->template getPtr<SizeType32>() : nullptr;

    TopPSamplingKernelParams<T> params;
    params.probs = probs;
    params.outputIds = outputs->outputIdsPtr.template getPtr<TokenIdType*>();
    params.workspace = samplingWorkspaceDevice;
    params.topPs = mRuntimeTopPDevice;
    params.sequenceLength = sequenceLength;
    params.endIds = endIds;
    params.batchSlots = batchSlots;
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = mSkipDecodeDevice;
    params.cumLogProbs = cumLogProbs;
    params.outputLogProbs = outputLogProbs;
    params.curandState = curandStatesDevice;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    if (mIsAirTopP == false)
    {
        invokeBatchTopPSampling<T>(params, mStream);
        sync_check_cuda_error();
    }
    else
    {
        params.blockNum = mAirTopPBlockNum;
        params.isDeterministic = mIsDeterministic;
        invokeBatchAirTopPSampling<T>(params, mStream);
        sync_check_cuda_error();
    }

    sync_check_cuda_error();
    invokeComputeToppDecay(mRuntimeTopPDevice, mInitialTopPDevice,
        outputs->outputIdsPtr.template getPtr<TokenIdType const*>(), mTopPDecayDevice, mTopPMinDevice,
        mTopPResetIdsDevice, sequenceLength, batchSlots, batchSize, mStream);
    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

} // namespace tensorrt_llm::layers
