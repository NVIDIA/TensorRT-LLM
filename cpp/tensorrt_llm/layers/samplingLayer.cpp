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

#include "tensorrt_llm/layers/samplingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
void SamplingLayer<T>::allocateBuffer(size_t batchSize)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    mSamplingWorkspaceSize = 0;
    if (mDecodingMode.isTopK())
    {
        mSamplingWorkspaceSize = std::max(mSamplingWorkspaceSize, mTopKDecode->getWorkspaceSize());
    }
    if (mDecodingMode.isTopP())
    {
        mSamplingWorkspaceSize = std::max(mSamplingWorkspaceSize, mTopPDecode->getWorkspaceSize());
    }

    std::array<size_t, 4> deviceBufferSizes;
    deviceBufferSizes[0] = sizeof(curandState_t) * batchSize;
    deviceBufferSizes[1] = sizeof(uint64_t) * batchSize;
    deviceBufferSizes[2] = sizeof(bool) * batchSize;
    deviceBufferSizes[3] = mSamplingWorkspaceSize;

    mCurandStatesDevice = mAllocator->reMalloc(mCurandStatesDevice, deviceBufferSizes[0], false);
    mRandomSeedsDevice = mAllocator->reMalloc(mRandomSeedsDevice, deviceBufferSizes[1], false);
    mSkipDecodeDevice = mAllocator->reMalloc(mSkipDecodeDevice, deviceBufferSizes[2], false);
    mSamplingWorkspaceDevice = mAllocator->reMalloc(mSamplingWorkspaceDevice, deviceBufferSizes[3], false);

    auto const bytesAllocated = std::accumulate(deviceBufferSizes.begin(), deviceBufferSizes.end(), 0);
    TLLM_LOG_DEBUG("SamplingLayer allocated %d bytes on GPU", bytesAllocated);

    mAllocatedSize = bytesAllocated;
    if (mDecodingMode.isTopK())
    {
        mAllocatedSize += mTopKDecode->getAllocatedSize();
    }
    if (mDecodingMode.isTopP())
    {
        mAllocatedSize += mTopPDecode->getAllocatedSize();
    }

    // host buffers.
    mSkipDecodeHost = (bool*) std::realloc(mSkipDecodeHost, sizeof(bool) * batchSize);
    TLLM_CHECK(mSkipDecodeHost != nullptr);
}

template <typename T>
void SamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    mAllocator->free((void**) (&mCurandStatesDevice));
    mAllocator->free((void**) (&mRandomSeedsDevice));
    mAllocator->free((void**) (&mSkipDecodeDevice));
    mAllocator->free((void**) (&mSamplingWorkspaceDevice));
    std::free(mSkipDecodeHost);
}

template <typename T>
SamplingLayer<T>::SamplingLayer(DecodingMode const& mode, size_t maxBatchSize, size_t vocabSize, size_t vocabSizePadded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator, cudaDeviceProp* prop)
    : BaseSamplingLayer<T>(maxBatchSize, vocabSize, vocabSizePadded, stream, std::move(allocator), nullptr)
    , mDecodingMode(mode)
{
    TLLM_CHECK_WITH_INFO(!mDecodingMode.isBeamSearch(), "Beam search mode has been requested from Sampling Layer");
    TLLM_CHECK_WITH_INFO(mDecodingMode.isTopKorTopP(), "Requested mode is neither TopK nor TopP");
    if (mDecodingMode.isTopK())
    {
        mTopKDecode
            = std::make_unique<TopKSamplingLayer<T>>(maxBatchSize, vocabSize, vocabSizePadded, mStream, mAllocator);
    }

    if (mDecodingMode.isTopP())
    {
        mTopPDecode = std::make_unique<TopPSamplingLayer<T>>(
            maxBatchSize, vocabSize, vocabSizePadded, mStream, mAllocator, prop, /* deterministic */ true);
    }

    allocateBuffer(maxBatchSize);
}

template <typename T>
void SamplingLayer<T>::setup(const size_t batchSize, int32_t const* batchSlots, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    // If runtime argument has single random seed, using this random seed to
    // initialize the random table of all sentences. If the argument has
    // [batchSize] random seeds, initializing the random table by different
    // random seeds respectively. If no random seed, initialize the random table
    // of all sentences by 0 directly.
    if (setupParams.randomSeed)
    {
        if (setupParams.randomSeed->size() == 1)
        {
            invokeCurandInitialize(
                mCurandStatesDevice, batchSlots, batchSize, setupParams.randomSeed->front(), mStream);
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(setupParams.randomSeed->size() == batchSize, "Random seed vector size mismatch.");
            cudaAutoCpy(mRandomSeedsDevice, setupParams.randomSeed->data(), batchSize, mStream);
            invokeCurandBatchInitialize(mCurandStatesDevice, batchSlots, batchSize, mRandomSeedsDevice, mStream);
            sync_check_cuda_error();
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(mCurandStatesDevice, batchSlots, batchSize, 0, mStream);
    }

    if (mDecodingMode.isTopK())
    {
        mTopKDecode->setup(batchSize, batchSlots, setupParams);
    }
    if (mDecodingMode.isTopP())
    {
        mTopPDecode->setup(batchSize, batchSlots, setupParams);
    }
}

template <typename T>
void SamplingLayer<T>::forward(DecodingOutputParams& outputs, ForwardParams& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];

    auto logits = inputs.logits.template getPtr<T>();
    auto endIds = inputs.end_ids.template getPtr<const int>();
    auto batchSlots = inputs.batch_slots ? inputs.batch_slots->template getPtr<const int>() : nullptr;
    float* cumLogProbs = (outputs.cum_log_probs) ? outputs.cum_log_probs->template getPtr<float>() : nullptr;
    float* outputLogProbs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;

    FinishedState* finishedInput = (inputs.finished)
        ? reinterpret_cast<FinishedState*>(inputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;

    std::vector<int32_t> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = inputs.batch_slots ? inputs.batch_slots->template getPtr<const int>() : batchSlotsVec.data();

    bool skipTopK = !mDecodingMode.isTopK();
    if (!skipTopK)
    {
        skipTopK = allOfBatchSlots(batchSlotsHost, mTopKDecode->getSkipDecodeHost(), batchSize, true);
    }

    bool skipTopP = !mDecodingMode.isTopP();
    if (!skipTopP)
    {
        skipTopP = allOfBatchSlots(batchSlotsHost, mTopPDecode->getSkipDecodeHost(), batchSize, true);
    }

    // Compute probabilities either for TopP or if cumLogProbs or outputLogProbs are specified
    bool const skipSoftMax = skipTopP && cumLogProbs == nullptr && outputLogProbs == nullptr;

    inputs.curand_states = mCurandStatesDevice;
    inputs.sampling_workspace = mSamplingWorkspaceDevice;
    inputs.probs_computed = !skipSoftMax;

    invokeAddBiasSoftMax(logits, (T**) nullptr, logits, (T*) (nullptr), endIds, finishedInput, batchSlots, batchSize,
        mMaxBatchSize, /* bw */ 1, mVocabSize, mVocabSizePadded, skipSoftMax, /* batchSlotLogits */ false, mStream);
    sync_check_cuda_error();

    if (!skipTopK)
    {
        mTopKDecode->forward(outputs, inputs);
    }

    if (!skipTopP)
    {
        mTopPDecode->forward(outputs, inputs);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class SamplingLayer<float>;
template class SamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
