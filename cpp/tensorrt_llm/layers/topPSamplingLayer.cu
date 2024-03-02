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
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

#include <algorithm>
#include <float.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

static __global__ void setTopPRuntimeArgs(int batchSize, uint32_t topK, uint32_t* topKs, int topKsSize, float topP,
    float* topPs, int topPsSize, bool* skipDecode, const int* batchSlots, float* initialTopPBuf)
{
    /**
     * @brief Setup the runtime arguments for topp, broadcasting top_p to top_ps
              and top_k to top_ks.
     */

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int bi = index; bi < batchSize; bi += gridDim.x * blockDim.x)
    {
        auto const batchSlot = batchSlots != nullptr ? batchSlots[bi] : bi;
        std::uint32_t k = topKsSize > 1 ? topKs[batchSlot] : topK;
        float p = topPsSize > 1 ? topPs[batchSlot] : topP;
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
void TopPSamplingLayer<T>::allocateBuffer(size_t batchSize)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (mIsDeterministic)
    {
        invokeTopPSampling<T>(nullptr, // workspace
            mSamplingWorkspaceSize, mCubTempStorageSize,
            nullptr,                   // output_ids
            nullptr,                   // sequence_length
            nullptr,                   // finished_input_buffer
            nullptr,                   // finished_output_buffer
            nullptr,                   // cum_log_probs
            nullptr,                   // output_log_probs
            nullptr,                   // log_probs
            mTopPIdValsDevice, mTopPOffsetDevice, mBeginTopPOffsetDevice, nullptr, batchSize, mMaxBatchSize,
            mVocabSizePadded, nullptr, 0.f, mStream, nullptr, nullptr);
    }
    else
    {
        invokeAirTopPSampling<T>(nullptr, mSamplingWorkspaceSize,
            nullptr, // output_ids
            nullptr, // sequence_length
            nullptr, // finished_input_buffer
            nullptr, // finished_output_buffer
            nullptr, // cum_log_probs
            nullptr, // output_log_probs
            nullptr, // log_probs)
            nullptr, batchSize, mMaxBatchSize, mVocabSizePadded, nullptr, 0.f, mStream, mAirTopPBlockNum, nullptr,
            nullptr);
    }

    std::array<size_t, 11> deviceBufferSizes;
    deviceBufferSizes[0] = sizeof(int32_t) * batchSize * mVocabSizePadded;
    deviceBufferSizes[1] = sizeof(int32_t) * (batchSize + 1);
    deviceBufferSizes[2] = sizeof(int32_t) * (batchSize + 1);
    deviceBufferSizes[3] = sizeof(uint32_t) * batchSize;
    deviceBufferSizes[4] = sizeof(float) * batchSize;
    deviceBufferSizes[5] = sizeof(float) * batchSize;
    deviceBufferSizes[6] = sizeof(float) * batchSize;
    deviceBufferSizes[7] = sizeof(float) * batchSize;
    deviceBufferSizes[8] = sizeof(int32_t) * batchSize;
    deviceBufferSizes[9] = sizeof(bool) * batchSize;
    deviceBufferSizes[10] = *std::max_element(&deviceBufferSizes[3], &deviceBufferSizes[9]);

    mTopPIdValsDevice = mAllocator->reMalloc(mTopPIdValsDevice, deviceBufferSizes[0], false);
    mTopPOffsetDevice = mAllocator->reMalloc(mTopPOffsetDevice, deviceBufferSizes[1], false);
    mBeginTopPOffsetDevice = mAllocator->reMalloc(mBeginTopPOffsetDevice, deviceBufferSizes[2], false);
    mRuntimeTopKDevice = mAllocator->reMalloc(mRuntimeTopKDevice, deviceBufferSizes[3], false);
    mRuntimeTopPDevice = mAllocator->reMalloc(mRuntimeTopPDevice, deviceBufferSizes[4], false);
    mInitialTopPDevice = mAllocator->reMalloc(mInitialTopPDevice, deviceBufferSizes[5], false);
    mTopPDecayDevice = mAllocator->reMalloc(mTopPDecayDevice, deviceBufferSizes[6], false);
    mTopPMinDevice = mAllocator->reMalloc(mTopPMinDevice, deviceBufferSizes[7], false);
    mTopPResetIdsDevice = mAllocator->reMalloc(mTopPResetIdsDevice, deviceBufferSizes[8], false);
    mSkipDecodeDevice = mAllocator->reMalloc(mSkipDecodeDevice, deviceBufferSizes[9], false);
    mSetupWorkspaceDevice = mAllocator->reMalloc(mSetupWorkspaceDevice, deviceBufferSizes[10], false);

    mSkipDecodeHost = (bool*) std::realloc(mSkipDecodeHost, sizeof(bool) * batchSize);
    std::fill(mSkipDecodeHost, mSkipDecodeHost + batchSize, true);

    mAllocatedSize = std::accumulate(deviceBufferSizes.begin(), deviceBufferSizes.end(), 0);
    TLLM_LOG_DEBUG("topPSamplingLayer allocated %lu bytes on GPU", mAllocatedSize);
}

template <typename T>
void TopPSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    mAllocator->free((void**) (&mTopPIdValsDevice));
    mAllocator->free((void**) (&mTopPOffsetDevice));
    mAllocator->free((void**) (&mBeginTopPOffsetDevice));
    mAllocator->free((void**) (&mRuntimeTopKDevice));
    mAllocator->free((void**) (&mRuntimeTopPDevice));
    mAllocator->free((void**) (&mInitialTopPDevice));
    mAllocator->free((void**) (&mTopPDecayDevice));
    mAllocator->free((void**) (&mTopPMinDevice));
    mAllocator->free((void**) (&mTopPResetIdsDevice));
    mAllocator->free((void**) (&mSkipDecodeDevice));
    mAllocator->free((void**) (&mSetupWorkspaceDevice));
    std::free(mSkipDecodeHost);
}

template <typename T>
void TopPSamplingLayer<T>::setup(size_t const batchSize, int32_t const* batchSlots, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    uint32_t const defaultTopK = 0;
    auto runtimeTopK = setupParams.runtime_top_k.value_or(std::vector<uint32_t>{defaultTopK});
    auto runtimeTopP = setupParams.runtime_top_p.value_or(std::vector<float>{});

    size_t const runtimeTopKSize = runtimeTopK.size();
    size_t const runtimeTopPSize = runtimeTopP.size();

    float const defaultTopPDecay{1.0f};
    auto decayVec = setupParams.top_p_decay.value_or(std::vector<float>(batchSize, defaultTopPDecay));

    float const defaultTopPMin{1e-6f}; // prevent topp becoming 0.0
    auto topPMinVec = setupParams.top_p_min.value_or(std::vector<float>(batchSize, defaultTopPMin));

    int32_t const defaultTopPResetId{-1};
    auto topPResetIdsVec = setupParams.top_p_reset_ids.value_or(std::vector<int32_t>(batchSize, defaultTopPResetId));

    if (runtimeTopPSize == 0)
    {
        for (size_t bi = 0; bi < batchSize; ++bi)
        {
            int32_t bid = bi;
            if (batchSlots)
            {
                bid = batchSlots[bi];
            }
            mSkipDecodeHost[bid] = true;
        }
        cudaAutoCpy(mSkipDecodeDevice, mSkipDecodeHost, mMaxBatchSize, mStream);
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
            TLLM_LOG_WARNING("Decay (%f) is out of range ([0.0, 1.0f]). Change to 1.0.", decay);
            decay = 1.0f;
        }
    }

    for (auto& topPMin : topPMinVec)
    {
        if (topPMin <= 0.f || topPMin > 1.0f)
        {
            TLLM_LOG_WARNING("TopP min (%f) is out of range ([0.0, 1.0f]). Change to 0.5.", topPMin);
            topPMin = 0.5f;
        }
    }

    uint32_t const topK = runtimeTopK.at(0);
    float const topP = runtimeTopP.at(0);

    if (runtimeTopKSize > 1)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopK.size() == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%lu) is not satisfied!", runtimeTopK.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<uint32_t*>(mSetupWorkspaceDevice), runtimeTopK.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<uint32_t*>(mSetupWorkspaceDevice), mRuntimeTopKDevice, batchSlots, batchSize, mStream);
    }
    if (runtimeTopPSize > 1)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopP.size() == batchSize,
            fmtstr("runtime_top_p.size() (%lu) == batchSize (%lu) is not satisfied!", runtimeTopP.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<float*>(mSetupWorkspaceDevice), runtimeTopP.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<float*>(mSetupWorkspaceDevice), mRuntimeTopPDevice, batchSlots, batchSize, mStream);
    }

    auto fillBuffers
        = [this, &batchSize, &batchSlots](std::string name, auto const& vector, auto deviceTmpBuffer, auto deviceBuffer)
    {
        TLLM_CHECK_WITH_INFO(vector.size() == batchSize,
            fmtstr("%s.size() (%lu) == batchSize (%lu) is not satisfied!", name.c_str(), vector.size(), batchSize));
        cudaAutoCpy(deviceTmpBuffer, vector.data(), batchSize, mStream);
        invokeScatterDecodingParams(deviceTmpBuffer, deviceBuffer, batchSlots, batchSize, mStream);
    };

    fillBuffers("top_p_decay", decayVec, reinterpret_cast<float*>(mSetupWorkspaceDevice), mTopPDecayDevice);

    fillBuffers("top_p_min", topPMinVec, reinterpret_cast<float*>(mSetupWorkspaceDevice), mTopPMinDevice);

    fillBuffers(
        "top_p_reset_ids", topPResetIdsVec, reinterpret_cast<int32_t*>(mSetupWorkspaceDevice), mTopPResetIdsDevice);

    {
        dim3 block(std::min((int) batchSize, 256));
        dim3 grid(divUp((int) batchSize, (int) block.x));
        setTopPRuntimeArgs<<<grid, block, 0, mStream>>>(batchSize, topK, mRuntimeTopKDevice, runtimeTopKSize, topP,
            mRuntimeTopPDevice, runtimeTopPSize, mSkipDecodeDevice, batchSlots, mInitialTopPDevice);
        sync_check_cuda_error();
    }

    cudaAutoCpy(mSkipDecodeHost, mSkipDecodeDevice, mMaxBatchSize, mStream);
    std::vector<float> runtimeTopPs(mMaxBatchSize);
    cudaAutoCpy(runtimeTopPs.data(), mRuntimeTopPDevice, mMaxBatchSize, mStream);
    {
        float maxTopP = 0.f;
        for (size_t bi = 0; bi < batchSize; ++bi)
        {
            int32_t bid = bi;
            if (batchSlots)
            {
                bid = batchSlots[bi];
            }
            maxTopP = std::max(maxTopP, runtimeTopPs[bid]);
        }
        mRuntimeMaxTopP = std::max(mRuntimeMaxTopP, maxTopP);
    }

    if (!mIsDeterministic)
    {
        int smCnt = mCudaDeviceProp->multiProcessorCount;
        mAirTopPBlockNum = calcAirTopPBlockNum<T, int, float>(batchSize, (int) mVocabSizePadded, smCnt);
    }
}

template <typename T>
void TopPSamplingLayer<T>::forward(DecodingOutputParams& outputs, ForwardParams& inputs)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];

    // Probabilities must be already computed instead of logits
    auto probs = inputs.logits.template getPtr<T>();
    auto endIds = inputs.end_ids.template getPtr<const int>();
    auto batchSlots = inputs.batch_slots ? inputs.batch_slots->template getPtr<const int>() : nullptr;
    auto curandStatesDevice = inputs.curand_states;
    auto samplingWorkspaceDevice = inputs.sampling_workspace;

    TLLM_CHECK_WITH_INFO(curandStatesDevice, "No curand states provided");
    TLLM_CHECK_WITH_INFO(samplingWorkspaceDevice, "No sampling workspace provided");

    if (mIsDeterministic)
    {
        invokeTopPInitialize(
            mTopPIdValsDevice, mTopPOffsetDevice, mBeginTopPOffsetDevice, batchSize, mVocabSizePadded, mStream);
        sync_check_cuda_error();
    }

    FinishedState* finishedInput = (inputs.finished)
        ? reinterpret_cast<FinishedState*>(inputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    FinishedState* finishedOutput = (outputs.finished)
        ? reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;

    float* cumLogProbs = (outputs.cum_log_probs) ? outputs.cum_log_probs->template getPtr<float>() : nullptr;
    float* outputLogProbs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    int* sequenceLength = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;

    if (mIsDeterministic)
    {
        invokeBatchTopPSampling<T>(samplingWorkspaceDevice, mSamplingWorkspaceSize, mCubTempStorageSize,
            outputs.output_ids_ptr.template getPtr<int*>(), sequenceLength, finishedInput, finishedOutput, cumLogProbs,
            outputLogProbs, probs, mTopPIdValsDevice, mTopPOffsetDevice, mBeginTopPOffsetDevice, curandStatesDevice,
            batchSize, mMaxBatchSize, mVocabSizePadded, endIds, mRuntimeMaxTopP, mRuntimeTopPDevice, mStream,
            mSkipDecodeDevice, batchSlots);
        sync_check_cuda_error();
        invokeComputeToppDecay(mRuntimeTopPDevice, mInitialTopPDevice,
            outputs.output_ids_ptr.template getPtr<const int*>(), mTopPDecayDevice, mTopPMinDevice, mTopPResetIdsDevice,
            sequenceLength, batchSlots, batchSize, mStream);
        sync_check_cuda_error();
    }
    else
    {
        invokeBatchAirTopPSampling<T>(samplingWorkspaceDevice, mSamplingWorkspaceSize,
            outputs.output_ids_ptr.template getPtr<int*>(), sequenceLength, finishedInput, finishedOutput, cumLogProbs,
            outputLogProbs, probs, curandStatesDevice, batchSize, mMaxBatchSize, mVocabSizePadded, endIds,
            mRuntimeMaxTopP, mRuntimeTopPDevice, mStream, mAirTopPBlockNum, mSkipDecodeDevice, batchSlots);
        sync_check_cuda_error();
    }
}

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(std::size_t maxBatchSize, std::size_t vocabSize, std::size_t vocabSizePadded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator, cudaDeviceProp* prop, bool isDeterministic)
    : BaseSamplingLayer<T>(maxBatchSize, vocabSize, vocabSizePadded, stream, std::move(allocator), prop)
    , mIsDeterministic(isDeterministic)
{
    allocateBuffer(mMaxBatchSize);
}

template <typename T>
TopPSamplingLayer<T>::~TopPSamplingLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    freeBuffer();
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
