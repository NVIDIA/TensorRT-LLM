/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/baseSamplingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/fillBuffers.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
void BaseSamplingLayer<T>::allocateBuffer(size_t batchSize)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    std::array<size_t, 10> deviceBufferSizes;
    deviceBufferSizes[0] = sizeof(curandState_t) * batchSize;
    deviceBufferSizes[1] = sizeof(uint64_t) * batchSize;
    deviceBufferSizes[2] = sizeof(float) * batchSize;
    deviceBufferSizes[3] = sizeof(float) * batchSize;
    deviceBufferSizes[4] = sizeof(float) * batchSize;
    deviceBufferSizes[5] = sizeof(float) * batchSize;
    deviceBufferSizes[6] = sizeof(int) * batchSize;
    deviceBufferSizes[7] = sizeof(T) * batchSize * mVocabSizePadded;
    deviceBufferSizes[8] = sizeof(bool) * batchSize;
    deviceBufferSizes[9] = sizeof(float) * batchSize;

    mCurandStatesDevice = mAllocator->reMalloc(mCurandStatesDevice, deviceBufferSizes[0], false);
    mRandomSeedsDevice = mAllocator->reMalloc(mRandomSeedsDevice, deviceBufferSizes[1], false);
    mTemperaturesDevice = mAllocator->reMalloc(mTemperaturesDevice, deviceBufferSizes[2], false);
    mRepetitionPenaltiesDevice = mAllocator->reMalloc(mRepetitionPenaltiesDevice, deviceBufferSizes[3], false);
    mPresencePenaltiesDevice = mAllocator->reMalloc(mPresencePenaltiesDevice, deviceBufferSizes[4], false);
    mFrequencyPenaltiesDevice = mAllocator->reMalloc(mFrequencyPenaltiesDevice, deviceBufferSizes[5], false);
    mMinLengthsDevice = mAllocator->reMalloc(mMinLengthsDevice, deviceBufferSizes[6], false);
    mRuntimeLogitsDevice = mAllocator->reMalloc(mRuntimeLogitsDevice, deviceBufferSizes[7], false);
    mSkipDecodeDevice = mAllocator->reMalloc(mSkipDecodeDevice, deviceBufferSizes[8], false);
    mSetupWorkspaceDevice = mAllocator->reMalloc(mSetupWorkspaceDevice, deviceBufferSizes[9], false);

    auto const bytesAllocated = std::accumulate(deviceBufferSizes.begin(), deviceBufferSizes.end(), 0);
    TLLM_LOG_DEBUG("baseSamplingLayer allocated %d bytes on GPU", bytesAllocated);

    // host buffers.
    mSkipDecodeHost = (bool*) std::realloc(mSkipDecodeHost, sizeof(bool) * batchSize);
    TLLM_CHECK(mSkipDecodeHost != nullptr);

    mIsAllocateBuffer = true;
}

template <typename T>
void BaseSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (mIsAllocateBuffer)
    {
        mAllocator->free((void**) (&mCurandStatesDevice));
        mAllocator->free((void**) (&mRandomSeedsDevice));
        mAllocator->free((void**) (&mTemperaturesDevice));
        mAllocator->free((void**) (&mRepetitionPenaltiesDevice));
        mAllocator->free((void**) (&mPresencePenaltiesDevice));
        mAllocator->free((void**) (&mFrequencyPenaltiesDevice));
        mAllocator->free((void**) (&mMinLengthsDevice));
        mAllocator->free((void**) (&mRuntimeLogitsDevice));
        mAllocator->free((void**) (&mSkipDecodeDevice));
        mAllocator->free((void**) (&mSetupWorkspaceDevice));
        std::free(mSkipDecodeHost);
        mIsAllocateBuffer = false;
    }
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t maxBatchSize, size_t vocabSize, size_t vocabSizePadded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator, cudaDeviceProp* prop)
    : BaseLayer(stream, std::move(allocator), prop)
    , mMaxBatchSize(maxBatchSize)
    , mVocabSize(vocabSize)
    , mVocabSizePadded(vocabSizePadded)
{
    allocateBuffer(maxBatchSize);
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(BaseSamplingLayer const& samplingLayer)
    : BaseLayer(samplingLayer)
    , mMaxBatchSize(samplingLayer.mMaxBatchSize)
    , mVocabSize(samplingLayer.mVocabSize)
    , mVocabSizePadded(samplingLayer.mVocabSizePadded)
    , mSamplingWorkspaceSize(samplingLayer.mSamplingWorkspaceSize)
{
    allocateBuffer(mMaxBatchSize);
}

template <typename T>
void BaseSamplingLayer<T>::setupBase(const size_t batchSize, int const* batchSlots, SetupParams const& setupParams)
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

    // Setup penalties.
    FillBuffers const fillBuffers{batchSize, mStream};

    mUseTemperature = static_cast<bool>(setupParams.temperature);
    mUseRepetitionPenalty = static_cast<bool>(setupParams.repetition_penalty);
    mUsePresencePenalty = static_cast<bool>(setupParams.presence_penalty);
    mUseFrequencyPenalty = static_cast<bool>(setupParams.frequency_penalty);
    mUseMinLengths = static_cast<bool>(setupParams.min_length);
    if (mUseTemperature)
    {
        fillBuffers(setupParams.temperature, getDefaultPenaltyValue(RepetitionPenaltyType::Temperature), mTemperature,
            mTemperaturesDevice, mSetupWorkspaceDevice, batchSlots);
    }
    if (mUseRepetitionPenalty)
    {
        fillBuffers(setupParams.repetition_penalty, getDefaultPenaltyValue(RepetitionPenaltyType::Repetition),
            mRepetitionPenalty, mRepetitionPenaltiesDevice, mSetupWorkspaceDevice, batchSlots);
    }
    if (mUsePresencePenalty)
    {
        fillBuffers(setupParams.presence_penalty, getDefaultPenaltyValue(RepetitionPenaltyType::Presence),
            mPresencePenalty, mPresencePenaltiesDevice, mSetupWorkspaceDevice, batchSlots);
    }
    if (mUseFrequencyPenalty)
    {
        fillBuffers(setupParams.frequency_penalty, getDefaultPenaltyValue(RepetitionPenaltyType::Frequency),
            mFrequencyPenalty, mFrequencyPenaltiesDevice, mSetupWorkspaceDevice, batchSlots);
    }
    if (mUseMinLengths)
    {
        fillBuffers(setupParams.min_length, (int) getDefaultPenaltyValue(RepetitionPenaltyType::MinLength), mMinLengths,
            mMinLengthsDevice, mSetupWorkspaceDevice, batchSlots);
    }
}

template <typename T>
void BaseSamplingLayer<T>::forward(DecodingOutputParams& outputs, ForwardParams const& inputs, int* penaltyWorkspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];
    auto const step = inputs.step;
    auto* const inputLengths = inputs.input_lengths ? inputs.input_lengths->template getPtr<const int>() : nullptr;

    auto* logits = inputs.logits.template getPtr<T>();
    TLLM_CHECK_WITH_INFO((inputs.batch_slots_host.has_value() ^ inputs.batch_slots.has_value()) == 0,
        "either both batch_slots_host and batch_slots have to be provided or neither of them");
    auto* batchSlots = inputs.batch_slots ? inputs.batch_slots->template getPtr<const int>() : nullptr;
    std::vector<int32_t> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto* batchSlotsHost
        = inputs.batch_slots_host ? inputs.batch_slots_host->template getPtr<const int>() : batchSlotsVec.data();

#define ALL_OF(addrs_, p_, sz_, v_) (std::all_of(addrs_, addrs_ + sz_, [&](int32_t b) { return p_[b] == v_; }))

    if (ALL_OF(batchSlotsHost, mSkipDecodeHost, batchSize, true))
    {
        // No sample in the current batch to do TopX sampling.
        return;
    }
    mSkipAny = std::any_of(batchSlotsHost, batchSlotsHost + batchSize,
        [this](int32_t batchSlot) { return this->mSkipDecodeHost[batchSlot]; });
    if (mSkipAny)
    {
        // A TopX Sampling layer directly changes the logit values. In case of
        // skip_any==true, meaning topk and topp layers will run simultaneously for
        // a batch in the same step. We copy the logits to an internal buffer, not
        // affecting the other sampling layers.
        TLLM_CHECK(inputs.logits.size() == batchSize * mVocabSizePadded);
        cudaD2Dcpy(mRuntimeLogitsDevice, logits, inputs.logits.size(), mStream);
        logits = mRuntimeLogitsDevice;
    }

    auto* embeddingBias = inputs.embedding_bias ? inputs.embedding_bias->template getPtr<T const>() : nullptr;
    auto* temperatures = (mUseTemperature
                             && !ALL_OF(batchSlotsHost, mTemperature, batchSize,
                                 getDefaultPenaltyValue(RepetitionPenaltyType::Temperature)))
        ? mTemperaturesDevice
        : nullptr;
    auto* repetitionPenalties = (mUseRepetitionPenalty
                                    && !ALL_OF(batchSlotsHost, mRepetitionPenalty, batchSize,
                                        getDefaultPenaltyValue(RepetitionPenaltyType::Repetition)))
        ? mRepetitionPenaltiesDevice
        : nullptr;
    auto* presencePenalties = (mUsePresencePenalty
                                  && !ALL_OF(batchSlotsHost, mPresencePenalty, batchSize,
                                      getDefaultPenaltyValue(RepetitionPenaltyType::Presence)))
        ? mPresencePenaltiesDevice
        : nullptr;
    auto* frequencyPenalties = (mUseFrequencyPenalty
                                   && !ALL_OF(batchSlotsHost, mFrequencyPenalty, batchSize,
                                       getDefaultPenaltyValue(RepetitionPenaltyType::Frequency)))
        ? mFrequencyPenaltiesDevice
        : nullptr;
    auto* minLengths = (mUseMinLengths
                           && !ALL_OF(batchSlotsHost, mMinLengths, batchSize,
                               (int) getDefaultPenaltyValue(RepetitionPenaltyType::MinLength)))
        ? mMinLengthsDevice
        : nullptr;

    InvokeBatchApplyPenaltyParams<T> penaltyParams{logits, embeddingBias, penaltyWorkspace, nullptr, temperatures,
        repetitionPenalties, presencePenalties, frequencyPenalties,
        (mUseRepetitionPenalty || mUsePresencePenalty || mUseFrequencyPenalty), batchSize, 1, inputs.max_seq_len,
        mVocabSize, mVocabSizePadded, outputs.output_ids_ptr.template getPtr<const int*>(), nullptr, inputLengths,
        outputs.sequence_length->getPtr<const int>(), minLengths, inputs.end_ids.template getPtr<const int>(),
        batchSlots, mStream};
    invokeBatchApplyPenalty(penaltyParams);
    sync_check_cuda_error();
#undef ALL_OF

    runSampling(outputs, inputs);

    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BaseSamplingLayer<float>;
template class BaseSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
