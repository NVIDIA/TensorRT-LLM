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

#include "tensorrt_llm/layers/medusaDecodingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
MedusaDecodingLayer<T>::MedusaDecodingLayer(SizeType maxBatchSize, SizeType vocabSize, SizeType vocabSizePadded,
    SizeType maxTokensPerStep, SizeType maxNumHeads, cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(stream, std::move(allocator), nullptr)
    , mMaxBatchSize(maxBatchSize)
    , mVocabSize(vocabSize)
    , mVocabSizePadded(vocabSizePadded)
    , mMaxTokensPerStep(maxTokensPerStep)
    , mMaxNumHeads(maxNumHeads)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
MedusaDecodingLayer<T>::~MedusaDecodingLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Get sampling workspace size
    {
        auto samplingSizePrimarySampling = getTopKWorkspaceSize<T>(mMaxBatchSize, 1, TOP_K_MAX, mVocabSizePadded);

        auto const maxBatchSizeHeadNums = mMaxBatchSize * mMaxNumHeads;
        auto samplingSizeMedusaHeadsSampling
            = getTopKWorkspaceSize<T>(maxBatchSizeHeadNums, 1, TOP_K_MAX, mVocabSizePadded);

        mSamplingWorkspaceSize = std::max(samplingSizePrimarySampling, samplingSizeMedusaHeadsSampling);
    }

    mDraftIdsPtrHost
        = runtime::BufferManager::pinned(ITensor::makeShape({static_cast<SizeType>(mMaxBatchSize), mMaxNumHeads}),
            runtime::TRTDataType<TokenIdType*>::value);
    mCummulativeTopK.resize(mMaxBatchSize * mMaxNumHeads);

    std::array<size_t, 10> deviceBufferSizes;
    deviceBufferSizes[0] = mMaxBatchSize * sizeof(curandState_t);
    deviceBufferSizes[1] = mMaxBatchSize * sizeof(SizeType);
    deviceBufferSizes[2] = mMaxBatchSize * mMaxNumHeads * sizeof(SizeType);
    deviceBufferSizes[3] = mSamplingWorkspaceSize;
    deviceBufferSizes[4] = mMaxBatchSize * sizeof(SizeType);
    deviceBufferSizes[5] = mMaxBatchSize * mMaxTokensPerStep * sizeof(TokenIdType);
    deviceBufferSizes[6] = mMaxBatchSize * mMaxNumHeads * sizeof(uint64_t);
    deviceBufferSizes[7] = mMaxBatchSize * mMaxNumHeads * sizeof(T*);
    deviceBufferSizes[8] = mMaxBatchSize * mMaxNumHeads * sizeof(curandState_t);
    deviceBufferSizes[9] = mMaxBatchSize * mMaxNumHeads * sizeof(SizeType);

    mCurandStatesDevice = mAllocator->reMalloc(mCurandStatesDevice, deviceBufferSizes[0], false);
    mTokensPerStepDevice = mAllocator->reMalloc(mTokensPerStepDevice, deviceBufferSizes[1], false);
    mSetupWorkspaceDevice = mAllocator->reMalloc(mSetupWorkspaceDevice, deviceBufferSizes[2], false);
    mSamplingWorkspaceDevice = mAllocator->reMalloc(mSamplingWorkspaceDevice, deviceBufferSizes[3], false);
    mRuntimeTopKDevice = mAllocator->reMalloc(mRuntimeTopKDevice, deviceBufferSizes[4], false);
    mTargetTokensDevice = mAllocator->reMalloc(mTargetTokensDevice, deviceBufferSizes[5], false);
    mRandomSeedsDevice = mAllocator->reMalloc(mRandomSeedsDevice, deviceBufferSizes[6], false);
    mMedusaLogitsPtrsDevice = mAllocator->reMalloc(mMedusaLogitsPtrsDevice, deviceBufferSizes[7], false);
    mCurandStatesMedusaLogitsDevice
        = mAllocator->reMalloc(mCurandStatesMedusaLogitsDevice, deviceBufferSizes[8], false);
    mRuntimeTopKPerRequestPerMedusaHeadDevice
        = mAllocator->reMalloc(mRuntimeTopKPerRequestPerMedusaHeadDevice, deviceBufferSizes[9], false);

    mTiledBatchSlotsSetup = BufferManager::pinnedPool(
        ITensor::makeShape({static_cast<SizeType>(mMaxBatchSize * mMaxNumHeads)}), nvinfer1::DataType::kINT32);
    mTiledBatchSlotsForward = BufferManager::pinnedPool(
        ITensor::makeShape({static_cast<SizeType>(mMaxBatchSize * mMaxNumHeads)}), nvinfer1::DataType::kINT32);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mAllocator->free((void**) (&mCurandStatesDevice));
    mAllocator->free((void**) (&mTokensPerStepDevice));
    mAllocator->free((void**) (&mSetupWorkspaceDevice));
    mAllocator->free((void**) (&mSamplingWorkspaceDevice));
    mAllocator->free((void**) (&mRuntimeTopKDevice));
    mAllocator->free((void**) (&mTargetTokensDevice));
    mAllocator->free((void**) (&mRandomSeedsDevice));
    mAllocator->free((void**) (&mMedusaLogitsPtrsDevice));
    mAllocator->free((void**) (&mCurandStatesMedusaLogitsDevice));
    mAllocator->free((void**) (&mRuntimeTopKPerRequestPerMedusaHeadDevice));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::setup(SizeType batchSize, SizeType const* batchSlots, MedusaSetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Prepare random seed
    auto initCurandStates = [this](std::optional<std::vector<uint64_t>> const& randomSeed, SizeType batchSize,
                                SizeType const* batchSlots, curandState_t* statesDevice)
    {
        if (randomSeed)
        {
            if (randomSeed->size() == 1)
            {
                invokeCurandInitialize(statesDevice, batchSlots, batchSize, randomSeed->front(), this->mStream);
                sync_check_cuda_error();
            }
            else
            {
                TLLM_CHECK_WITH_INFO(randomSeed->size() == batchSize, "Random seed vector size mismatch.");
                cudaAutoCpy(this->mRandomSeedsDevice, randomSeed->data(), batchSize, this->mStream);
                invokeCurandBatchInitialize(
                    statesDevice, batchSlots, batchSize, this->mRandomSeedsDevice, this->mStream);
                sync_check_cuda_error();
            }
        }
        else
        {
            // Initialize curand states using the default seed 0.
            invokeCurandInitialize(statesDevice, batchSlots, batchSize, 0, this->mStream);
        }
    };

    initCurandStates(setupParams.randomSeed, batchSize, batchSlots, mCurandStatesDevice);

    auto batchSizeMaxNumHeads = batchSize * mMaxNumHeads;
    auto randomSeed = setupParams.randomSeed.value_or(std::vector<uint64_t>(batchSize, uint64_t{0}));
    std::vector<uint64_t> tiledRandomSeed(batchSizeMaxNumHeads);
    if (randomSeed.size() > 1)
    {
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType hi = 0; hi < mMaxNumHeads; ++hi)
            {
                tiledRandomSeed[bi * mMaxNumHeads + hi] = randomSeed[bi];
            }
        }
    }
    auto tiledBatchSlots = bufferCast<SizeType>(*mTiledBatchSlotsSetup);
    for (SizeType bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType hi = 0; hi < mMaxNumHeads; ++hi)
        {
            tiledBatchSlots[bi * mMaxNumHeads + hi] = batchSlots[bi] + hi;
        }
    }
    initCurandStates({tiledRandomSeed}, batchSizeMaxNumHeads, tiledBatchSlots, mCurandStatesMedusaLogitsDevice);

    // Prepare tokens per step
    {
        auto tokensPerStep = setupParams.tokensPerStep.value_or(std::vector<SizeType>{batchSize, mMaxTokensPerStep});
        TLLM_CHECK_WITH_INFO(tokensPerStep.size() == batchSize,
            fmtstr("tokensPerStep.size() (%lu) == batchSize (%d) is not satisfied!", tokensPerStep.size(), batchSize));

        cudaAutoCpy(reinterpret_cast<SizeType*>(mSetupWorkspaceDevice), tokensPerStep.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<SizeType*>(mSetupWorkspaceDevice), mTokensPerStepDevice, batchSlots, batchSize, mStream);
    }

    // Prepare runtime top K
    auto prepareRuntimeTopK = [this](std::vector<SizeType> const& runtimeTopK, SizeType batchSize,
                                  SizeType const* batchSlots, SizeType* runtimeTopKDevice)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopK.size() == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopK.size(), batchSize));

        cudaAutoCpy(
            reinterpret_cast<SizeType*>(this->mSetupWorkspaceDevice), runtimeTopK.data(), batchSize, this->mStream);
        invokeScatterDecodingParams(reinterpret_cast<SizeType*>(this->mSetupWorkspaceDevice), runtimeTopKDevice,
            batchSlots, batchSize, this->mStream);

        // FIXME(nkorobov): monotonically growing
        auto const curMaxTopK = *std::max_element(std::begin(runtimeTopK), std::end(runtimeTopK));
        return curMaxTopK;
    };

    auto constexpr defaultTopK = 1u;
    {
        auto runtimeTopK = setupParams.runtimeTopK.value_or(std::vector<SizeType>{batchSize, defaultTopK});
        auto const curMaxTopK = prepareRuntimeTopK(runtimeTopK, batchSize, batchSlots, mRuntimeTopKDevice);
        mRuntimeMaxTopK = std::max(mRuntimeMaxTopK, curMaxTopK);
    }
    {
        auto runtimeHeadsTopK = setupParams.runtimeHeadsTopK;
        std::vector<SizeType> runtimeHeadsTopKFlatten;
        if (runtimeHeadsTopK.has_value())
        {
            for (auto const& sub : runtimeHeadsTopK.value())
            {
                runtimeHeadsTopKFlatten.insert(runtimeHeadsTopKFlatten.end(), sub.begin(), sub.end());
            }
        }
        else
        {
            runtimeHeadsTopKFlatten = std::vector<SizeType>(batchSizeMaxNumHeads, defaultTopK);
        }

        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            auto const slot = batchSlots[bi];
            SizeType cummulativeTopK = 0;
            for (SizeType hi = 0; hi < mMaxNumHeads; ++hi)
            {
                mCummulativeTopK[slot * mMaxNumHeads + hi] = cummulativeTopK;
                cummulativeTopK += runtimeHeadsTopKFlatten[bi * mMaxNumHeads + hi];
            }
        }

        auto tiledBatchSlots = bufferCast<SizeType>(*mTiledBatchSlotsSetup);
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType hi = 0; hi < mMaxNumHeads; ++hi)
            {
                tiledBatchSlots[bi * mMaxNumHeads + hi] = mMaxNumHeads * batchSlots[bi] + hi;
            }
        }

        auto const curMaxTopK = prepareRuntimeTopK(runtimeHeadsTopKFlatten, static_cast<SizeType>(batchSizeMaxNumHeads),
            tiledBatchSlots, mRuntimeTopKPerRequestPerMedusaHeadDevice);
        mRuntimeMaxTopKPerRequestPerMedusaHead = std::max(mRuntimeMaxTopKPerRequestPerMedusaHead, curMaxTopK);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::forward(DecodingOutputParams& outputs, MedusaForwardParams& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    samplePrimeHeadTokens(outputs, inputs);

    acceptDraftTokens(outputs, inputs);

    sampleNewDraftTokens(outputs, inputs);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::samplePrimeHeadTokens(DecodingOutputParams& outputs, MedusaForwardParams& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];

    auto logits = inputs.logits.template getPtr<T>();
    auto batchSlots
        = inputs.batch_slots ? inputs.batch_slots->template getPtr<SizeType const>() : static_cast<SizeType*>(nullptr);
    auto sequenceLengths = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<SizeType>()
                                                     : static_cast<SizeType*>(nullptr);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");

    // Sample multiple tokens per request and store them to separate to be accepted/rejected later
    // Sequence length is not modified, endIds is not checked, outputLogProbs are not supported.
    // Finished state is not set.
    invokeBatchTopKSampling(mSamplingWorkspaceDevice, logits, /* logProbsPtrs */ static_cast<T const* const*>(nullptr),
        /* outputIdsPtrs */ nullptr, mTargetTokensDevice, sequenceLengths,
        /* finishedInput */ nullptr, /* finishedOutput */ nullptr,
        /* cumLogProbs */ nullptr, /* outputLogProbs */ nullptr, mCurandStatesDevice, mRuntimeMaxTopK,
        mRuntimeTopKDevice, 1.0f, /* runtimeTopPDevice */ nullptr, mVocabSizePadded, /* endIds */ nullptr, batchSlots,
        mStream, batchSize, mMaxBatchSize, mTokensPerStepDevice, mMaxTokensPerStep, mMaxTokensPerStep,
        /* skipDecode */ nullptr, /* normalizeLogProbs */ false,
        /* probsComputed */ false, /* return all Top-K*/ false);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::acceptDraftTokens(DecodingOutputParams& outputs, MedusaForwardParams& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];
    auto const maxSeqLen = outputs.output_ids.shape[outputs.output_ids.shape.size() - 1];

    auto outputIds = outputs.output_ids.template getPtr<TokenIdType>();
    auto endIds = inputs.end_ids.template getPtr<TokenIdType const>();
    auto paths = inputs.paths.template getPtr<SizeType const>();
    auto medusaLogits = inputs.medusaLogits.template getPtr<T const>();

    auto batchSlots
        = inputs.batch_slots ? inputs.batch_slots->template getPtr<SizeType const>() : static_cast<SizeType*>(nullptr);
    auto sequenceLengths = outputs.sequence_length ? outputs.sequence_length->template getPtr<SizeType>()
                                                   : static_cast<SizeType*>(nullptr);
    auto acceptedLengths = outputs.acceptedLengths ? outputs.acceptedLengths->template getPtr<SizeType>()
                                                   : static_cast<SizeType*>(nullptr);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(acceptedLengths != nullptr, "Accepted lengths must be provided for MedusaDecoding");

    auto finishedStates
        = reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>());

    // Compare draft tokens from outputIds with sampled target tokens at mTargetTokensDevice using paths.
    // Select the longest accepted path, modify outputIds in-place, increment sequenceLengths accordingly.
    // Fill mMedusaLogitsPtrsDevice with respective Medusa logits
    acceptDraftTokensByIdsWithPaths(outputIds, mTargetTokensDevice, sequenceLengths, acceptedLengths, finishedStates,
        batchSlots, paths, endIds, medusaLogits, const_cast<T const**>(mMedusaLogitsPtrsDevice), batchSize, mVocabSize,
        mMaxBatchSize, maxSeqLen, mMaxTokensPerStep, mMaxNumHeads, mMaxTokensPerStep, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::sampleNewDraftTokens(DecodingOutputParams& outputs, MedusaForwardParams& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];
    auto batchSlots
        = inputs.batch_slots ? inputs.batch_slots->template getPtr<SizeType const>() : static_cast<SizeType*>(nullptr);
    auto sequenceLengths = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<SizeType>()
                                                     : static_cast<SizeType*>(nullptr);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");

    // For each request we sample Head Num times for topK[hi] tokens
    auto const batchSizeHeadNums = batchSize * mMaxNumHeads;
    auto const maxBatchSizeHeadNums = mMaxBatchSize * mMaxNumHeads;

    auto tiledBatchSlots = bufferCast<SizeType>(*mTiledBatchSlotsForward);
    for (SizeType bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType hi = 0; hi < mMaxNumHeads; ++hi)
        {
            tiledBatchSlots[bi * mMaxNumHeads + hi] = mMaxNumHeads * batchSlots[bi] + hi;
        }
    }

    auto draftIdsPtrs = reinterpret_cast<TokenIdType**>(bufferCast<int64_t>(*mDraftIdsPtrHost));
    auto draftIds = (outputs.nextDraftTokens) ? outputs.nextDraftTokens->template getPtr<TokenIdType>()
                                              : static_cast<TokenIdType*>(nullptr);
    TLLM_CHECK_WITH_INFO(draftIds != nullptr, "Draft ids must be provided for MedusaDecoding");

    for (SizeType bi = 0; bi < batchSize; ++bi)
    {
        auto slot = batchSlots[bi];
        for (SizeType hi = 0; hi < mMaxNumHeads; ++hi)
        {
            draftIdsPtrs[slot * mMaxNumHeads + hi]
                = draftIds + slot * mMaxTokensPerStep + mCummulativeTopK[slot * mMaxNumHeads + hi];
        }
    }

    invokeBatchTopKSampling(mSamplingWorkspaceDevice,
        /* logits */ static_cast<T const*>(nullptr), const_cast<T const* const*>(mMedusaLogitsPtrsDevice), draftIdsPtrs,
        /* outputIds */ nullptr, /* sequenceLength */ nullptr,
        /* finishedInput */ nullptr, /* finishedOutput */ nullptr,
        /* cumLogProbs */ nullptr, /* outputLogProbs */ nullptr, mCurandStatesMedusaLogitsDevice,
        mRuntimeMaxTopKPerRequestPerMedusaHead, mRuntimeTopKPerRequestPerMedusaHeadDevice, 1.0f,
        /* runtimeTopPDevice */ nullptr, mVocabSizePadded, /* endIds */ nullptr, tiledBatchSlots, mStream,
        batchSizeHeadNums, maxBatchSizeHeadNums,
        /* tokensPerStep */ nullptr, /* maxTokensPerStep */ 1,
        /* maxSeqLen (not required as outputIds is nullptr) */ 0,
        /* skipDecode */ nullptr, /* normalizeLogProbs */ false,
        /* probsComputed */ false, /* return all Top-K*/ true);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class MedusaDecodingLayer<float>;
template class MedusaDecodingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
