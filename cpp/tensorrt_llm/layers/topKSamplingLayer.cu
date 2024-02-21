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
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <algorithm>
#include <float.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

template <uint32_t TOP_K_MAX>
__global__ void setupTopKRuntimeArgs(int batchSize, uint32_t topK, uint32_t* topKs, int topKsSize, float topP,
    float* topPs, int topPsSize, bool* skipDecode, const int* batchSlots)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int bi = index; bi < batchSize; bi += gridDim.x * blockDim.x)
    {
        auto const batchSlot = batchSlots != nullptr ? batchSlots[bi] : bi;
        uint32_t k = topKsSize > 1 ? topKs[batchSlot] : topK;
        float p = topPsSize > 1 ? topPs[batchSlot] : topP;
        if (k == 0 && p == 0.0f)
        {
            // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
            // equivalent to greedy search. So, we set the topk = 1 as an alternative
            // solution.
            k = 1;
        }
        if (k > 0 && p == 0.0f)
        {
            // This case corresponds to the old topk sampling, which is equivalent to
            // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
            // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
            // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
            // compatibility.
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX.
        topKs[batchSlot] = k;
        // Clip p value if it is out of range. range = [0.0, 1.0].
        topPs[batchSlot] = p;
        skipDecode[batchSlot] = k == 0;
    }
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer(size_t const batchSize)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    invokeTopKSampling<T>(nullptr, mSamplingWorkspaceSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, TOP_K_MAX, 1.0f, mVocabSizePadded, nullptr, nullptr, mStream, batchSize, mMaxBatchSize,
        nullptr, mNormalizeLogProbs, false);

    std::array<size_t, 4> deviceBufferSizes;
    deviceBufferSizes[0] = sizeof(uint32_t) * batchSize;
    deviceBufferSizes[1] = sizeof(float) * batchSize;
    deviceBufferSizes[2] = sizeof(bool) * batchSize;
    deviceBufferSizes[3] = std::max(deviceBufferSizes[0], deviceBufferSizes[1]);

    mRuntimeTopKDevice = mAllocator->reMalloc(mRuntimeTopKDevice, deviceBufferSizes[0], false);
    mRuntimeTopPDevice = mAllocator->reMalloc(mRuntimeTopPDevice, deviceBufferSizes[1], false);
    mSkipDecodeDevice = mAllocator->reMalloc(mSkipDecodeDevice, deviceBufferSizes[2], false);
    mSetupWorkspaceDevice = mAllocator->reMalloc(mSetupWorkspaceDevice, deviceBufferSizes[3], false);

    mSkipDecodeHost = (bool*) std::realloc(mSkipDecodeHost, sizeof(bool) * batchSize);

    mAllocatedSize = std::accumulate(deviceBufferSizes.begin(), deviceBufferSizes.end(), 0);
    TLLM_LOG_DEBUG("topKSamplingLayer allocated %lu bytes on GPU", mAllocatedSize);
}

template <typename T>
void TopKSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    mAllocator->free((void**) (&mRuntimeTopKDevice));
    mAllocator->free((void**) (&mRuntimeTopPDevice));
    mAllocator->free((void**) (&mSkipDecodeDevice));
    mAllocator->free((void**) (&mSetupWorkspaceDevice));
    std::free(mSkipDecodeHost);
}

template <typename T>
void TopKSamplingLayer<T>::setup(size_t const batchSize, int32_t const* batchSlots, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    uint32_t constexpr defaultTopK = 0;
    auto runtimeTopK = setupParams.runtime_top_k.value_or(std::vector<uint32_t>{defaultTopK});
    auto runtimeTopP = setupParams.runtime_top_p.value_or(std::vector<float>{});

    size_t const runtimeTopKSize = runtimeTopK.size();
    size_t const runtimeTopPSize = runtimeTopP.size();
    mNormalizeLogProbs = setupParams.normalize_log_probs.has_value() && setupParams.normalize_log_probs.value();

    for (auto& topP : runtimeTopP)
    {
        if (topP < 0.f || topP > 1.0f)
        {
            TLLM_LOG_WARNING("TopP (%f) is out of range ([0.0, 1.0f]). Clip to closest number.", topP);
            topP = std::clamp(topP, 0.f, 1.f);
        }
    }
    for (auto& topK : runtimeTopK)
    {
        if (topK > TOP_K_MAX)
        {
            TLLM_LOG_WARNING(
                "TopK (%d) is larger than max supported number (%d). Clip to max supported number.", topK, TOP_K_MAX);
            topK = TOP_K_MAX;
        }
    }

    uint32_t const topK = *std::max_element(std::begin(runtimeTopK), std::end(runtimeTopK));
    float const topP = (runtimeTopPSize == 0) ? 0.0f : runtimeTopP.front();

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
            fmtstr("runtimeTopP.size() (%lu) == batchSize (%lu) is not satisfied!", runtimeTopP.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<float*>(mSetupWorkspaceDevice), runtimeTopP.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<float*>(mSetupWorkspaceDevice), mRuntimeTopPDevice, batchSlots, batchSize, mStream);
    }

    {
        dim3 block(std::min((int) batchSize, 256));
        dim3 grid(divUp((int) batchSize, (int) block.x));
        // support topK up to TOP_K_MAX.
        setupTopKRuntimeArgs<TOP_K_MAX><<<grid, block, 0, mStream>>>(batchSize, topK, mRuntimeTopKDevice,
            runtimeTopKSize, topP, mRuntimeTopPDevice, runtimeTopPSize, mSkipDecodeDevice, batchSlots);
    }

    cudaAutoCpy(mSkipDecodeHost, mSkipDecodeDevice, mMaxBatchSize, mStream);
    std::vector<uint32_t> runtimeTopKs(mMaxBatchSize);
    cudaAutoCpy(runtimeTopKs.data(), mRuntimeTopKDevice, mMaxBatchSize, mStream);
    {
        uint32_t maxTopK = 0;
        for (size_t bi = 0; bi < batchSize; ++bi)
        {
            uint32_t bid = bi;
            if (batchSlots)
            {
                bid = batchSlots[bi];
            }
            maxTopK = std::max(maxTopK, runtimeTopKs[bid]);
        }
        mRuntimeMaxTopK = std::max(mRuntimeMaxTopK, maxTopK);
    }
}

template <typename T>
void TopKSamplingLayer<T>::forward(DecodingOutputParams& outputs, ForwardParams& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];

    auto logits = inputs.logits.template getPtr<T>();
    auto endIds = inputs.end_ids.template getPtr<const int>();
    auto batchSlots = inputs.batch_slots ? inputs.batch_slots->template getPtr<const int>() : nullptr;
    auto curandStatesDevice = inputs.curand_states;
    auto samplingWorkspaceDevice = inputs.sampling_workspace;
    auto const probsComputed = inputs.probs_computed;

    TLLM_CHECK_WITH_INFO(curandStatesDevice, "No curand states provided");
    TLLM_CHECK_WITH_INFO(samplingWorkspaceDevice, "No sampling workspace provided");

    FinishedState* finishedInput = (inputs.finished)
        ? reinterpret_cast<FinishedState*>(inputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    FinishedState* finishedOutput = (outputs.finished)
        ? reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;

    float* cumLogProbs = (outputs.cum_log_probs) ? outputs.cum_log_probs->template getPtr<float>() : nullptr;
    float* outputLogProbs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    int* sequenceLength = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;

    invokeBatchTopKSampling(samplingWorkspaceDevice, mSamplingWorkspaceSize, logits,
        outputs.output_ids_ptr.template getPtr<int*>(), sequenceLength, finishedInput, finishedOutput, cumLogProbs,
        outputLogProbs, curandStatesDevice, (int) mRuntimeMaxTopK, (int*) (mRuntimeTopKDevice), 1.0f,
        mRuntimeTopPDevice, mVocabSizePadded, endIds, batchSlots, mStream, batchSize, mMaxBatchSize, mSkipDecodeDevice,
        mNormalizeLogProbs, probsComputed);
    sync_check_cuda_error();
}

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(size_t maxBatchSize, size_t vocabSize, size_t vocabSizePadded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseSamplingLayer<T>(maxBatchSize, vocabSize, vocabSizePadded, stream, std::move(allocator), nullptr)
{
    allocateBuffer(mMaxBatchSize);
}

template <typename T>
TopKSamplingLayer<T>::~TopKSamplingLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    freeBuffer();
}

template class TopKSamplingLayer<float>;
template class TopKSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
