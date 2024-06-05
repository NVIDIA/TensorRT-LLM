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
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
MedusaDecodingLayer<T>::MedusaDecodingLayer(
    DecoderDomain const& decoderDomain, cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
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
        auto samplingSizePrimarySampling = getTopKWorkspaceSize<T>(mDecoderDomain.getMaxBatchSize(),
            mDecoderDomain.getMaxTokensPerStep(), TOP_K_MAX, mDecoderDomain.getVocabSizePadded());

        auto const maxBatchSizeHeadNums = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads();
        auto samplingSizeMedusaHeadsSampling
            = getTopKWorkspaceSize<T>(maxBatchSizeHeadNums, 1, TOP_K_MAX, mDecoderDomain.getVocabSizePadded());

        mWorkspaceSize = std::max(samplingSizePrimarySampling, samplingSizeMedusaHeadsSampling);
    }

    mDraftIdsPtrHost = runtime::BufferManager::pinned(
        ITensor::makeShape(
            {static_cast<SizeType32>(mDecoderDomain.getMaxBatchSize()), mDecoderDomain.getMaxNumMedusaHeads()}),
        runtime::TRTDataType<TokenIdType*>::value);
    mCummulativeTopK.resize(mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads());

    std::array<size_t, 11> deviceBufferSizes;
    deviceBufferSizes[0] = mDecoderDomain.getMaxBatchSize() * sizeof(curandState_t);
    deviceBufferSizes[1]
        = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads() * sizeof(SizeType32);
    deviceBufferSizes[2] = mWorkspaceSize;
    deviceBufferSizes[3] = mDecoderDomain.getMaxBatchSize() * sizeof(SizeType32);
    deviceBufferSizes[4]
        = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxTokensPerStep() * sizeof(TokenIdType);
    deviceBufferSizes[5] = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads() * sizeof(uint64_t);
    deviceBufferSizes[6] = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads() * sizeof(T*);
    deviceBufferSizes[7]
        = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads() * sizeof(curandState_t);
    deviceBufferSizes[8]
        = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads() * sizeof(SizeType32);
    deviceBufferSizes[9]
        = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxTokensPerStep() * sizeof(TokenIdType);
    deviceBufferSizes[10] = mDecoderDomain.getMaxBatchSize() * sizeof(SizeType32);

    mCurandStatesDevice = mAllocator->reMalloc(mCurandStatesDevice, deviceBufferSizes[0], false);
    mSetupWorkspaceDevice = mAllocator->reMalloc(mSetupWorkspaceDevice, deviceBufferSizes[1], false);
    mSamplingWorkspaceDevice = mAllocator->reMalloc(mSamplingWorkspaceDevice, deviceBufferSizes[2], false);
    mRuntimeTopKDevice = mAllocator->reMalloc(mRuntimeTopKDevice, deviceBufferSizes[3], false);
    mTargetTokensDevice = mAllocator->reMalloc(mTargetTokensDevice, deviceBufferSizes[4], false);
    mRandomSeedsDevice = mAllocator->reMalloc(mRandomSeedsDevice, deviceBufferSizes[5], false);
    mMedusaSelectedLogitsPtrsDevice
        = mAllocator->reMalloc(mMedusaSelectedLogitsPtrsDevice, deviceBufferSizes[6], false);
    mCurandStatesMedusaLogitsDevice
        = mAllocator->reMalloc(mCurandStatesMedusaLogitsDevice, deviceBufferSizes[7], false);
    mRuntimeTopKPerRequestPerMedusaHeadDevice
        = mAllocator->reMalloc(mRuntimeTopKPerRequestPerMedusaHeadDevice, deviceBufferSizes[8], false);
    mNewDraftTokensDevice = mAllocator->reMalloc(mNewDraftTokensDevice, deviceBufferSizes[9], false);
    mBestPathIdsDevice = mAllocator->reMalloc(mBestPathIdsDevice, deviceBufferSizes[10], false);

    mTiledBatchSlotsSetup = BufferManager::pinnedPool(
        ITensor::makeShape(
            {static_cast<SizeType32>(mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads())}),
        nvinfer1::DataType::kINT32);
    mTiledBatchSlotsForward = BufferManager::pinnedPool(
        ITensor::makeShape(
            {static_cast<SizeType32>(mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads())}),
        nvinfer1::DataType::kINT32);
    mMedusaInputLogitsPtrs = BufferManager::pinnedPool(
        ITensor::makeShape(
            {static_cast<SizeType32>(mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads())}),
        TRTDataType<T*>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mAllocator->free((void**) (&mCurandStatesDevice));
    mAllocator->free((void**) (&mSetupWorkspaceDevice));
    mAllocator->free((void**) (&mSamplingWorkspaceDevice));
    mAllocator->free((void**) (&mRuntimeTopKDevice));
    mAllocator->free((void**) (&mTargetTokensDevice));
    mAllocator->free((void**) (&mRandomSeedsDevice));
    mAllocator->free((void**) (&mMedusaSelectedLogitsPtrsDevice));
    mAllocator->free((void**) (&mCurandStatesMedusaLogitsDevice));
    mAllocator->free((void**) (&mRuntimeTopKPerRequestPerMedusaHeadDevice));
    mAllocator->free((void**) (&mNewDraftTokensDevice));
    mAllocator->free((void**) (&mBestPathIdsDevice));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<MedusaSetupParams>(baseSetupParams);

    // Prepare random seed
    auto initCurandStates = [this](std::optional<std::vector<uint64_t>> const& randomSeed, SizeType32 batchSize,
                                SizeType32 const* batchSlots, curandState_t* statesDevice)
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
            invokeCurandInitialize(
                statesDevice, batchSlots, batchSize, DefaultDecodingParams::getSeed(), this->mStream);
        }
    };

    initCurandStates(setupParams->randomSeed, batchSize, batchSlots, mCurandStatesDevice);

    auto batchSizeMaxNumHeads = batchSize * mDecoderDomain.getMaxNumMedusaHeads();
    auto randomSeed = setupParams->randomSeed.value_or(std::vector<uint64_t>(batchSize, uint64_t{0}));
    std::vector<uint64_t> tiledRandomSeed(batchSizeMaxNumHeads);
    if (randomSeed.size() > 1)
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType32 hi = 0; hi < mDecoderDomain.getMaxNumMedusaHeads(); ++hi)
            {
                tiledRandomSeed[bi * mDecoderDomain.getMaxNumMedusaHeads() + hi] = randomSeed[bi];
            }
        }
    }
    auto tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsSetup);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType32 hi = 0; hi < mDecoderDomain.getMaxNumMedusaHeads(); ++hi)
        {
            tiledBatchSlots[bi * mDecoderDomain.getMaxNumMedusaHeads() + hi] = batchSlots[bi] + hi;
        }
    }
    initCurandStates({tiledRandomSeed}, batchSizeMaxNumHeads, tiledBatchSlots, mCurandStatesMedusaLogitsDevice);

    // Prepare runtime top K
    auto prepareRuntimeTopK = [this](std::vector<SizeType32> const& runtimeTopK, SizeType32 batchSize,
                                  SizeType32 const* batchSlots, SizeType32* runtimeTopKDevice)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopK.size() == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopK.size(), batchSize));

        cudaAutoCpy(
            reinterpret_cast<SizeType32*>(this->mSetupWorkspaceDevice), runtimeTopK.data(), batchSize, this->mStream);
        invokeScatterDecodingParams(reinterpret_cast<SizeType32*>(this->mSetupWorkspaceDevice), runtimeTopKDevice,
            batchSlots, batchSize, this->mStream);

        // FIXME(nkorobov): monotonically growing
        auto const curMaxTopK = *std::max_element(std::begin(runtimeTopK), std::end(runtimeTopK));
        return curMaxTopK;
    };

    auto constexpr defaultTopK = 1u;
    {
        auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector<SizeType32>(batchSize, defaultTopK));
        auto const curMaxTopK = prepareRuntimeTopK(runtimeTopK, batchSize, batchSlots, mRuntimeTopKDevice);
        mRuntimeMaxTopK = std::max(mRuntimeMaxTopK, curMaxTopK);
    }
    {
        auto runtimeHeadsTopK = setupParams->runtimeHeadsTopK;
        std::vector<SizeType32> runtimeHeadsTopKFlatten;
        if (runtimeHeadsTopK.has_value() && runtimeHeadsTopK->size())
        {
            for (auto const& sub : runtimeHeadsTopK.value())
            {
                runtimeHeadsTopKFlatten.insert(runtimeHeadsTopKFlatten.end(), sub.begin(), sub.end());
            }
        }
        else
        {
            runtimeHeadsTopKFlatten = std::vector<SizeType32>(batchSizeMaxNumHeads, defaultTopK);
        }

        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const slot = batchSlots[bi];
            SizeType32 cummulativeTopK = 0;
            for (SizeType32 hi = 0; hi < mDecoderDomain.getMaxNumMedusaHeads(); ++hi)
            {
                mCummulativeTopK[slot * mDecoderDomain.getMaxNumMedusaHeads() + hi] = cummulativeTopK;
                cummulativeTopK += runtimeHeadsTopKFlatten[bi * mDecoderDomain.getMaxNumMedusaHeads() + hi];
            }
        }

        auto tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsSetup);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType32 hi = 0; hi < mDecoderDomain.getMaxNumMedusaHeads(); ++hi)
            {
                tiledBatchSlots[bi * mDecoderDomain.getMaxNumMedusaHeads() + hi]
                    = mDecoderDomain.getMaxNumMedusaHeads() * batchSlots[bi] + hi;
            }
        }

        auto const curMaxTopK = prepareRuntimeTopK(runtimeHeadsTopKFlatten,
            static_cast<SizeType32>(batchSizeMaxNumHeads), tiledBatchSlots, mRuntimeTopKPerRequestPerMedusaHeadDevice);
        mRuntimeMaxTopKPerRequestPerMedusaHead = std::max(mRuntimeMaxTopKPerRequestPerMedusaHead, curMaxTopK);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::forward(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<MedusaInputParams>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<MedusaOutputParams>(baseOutputs);

    samplePrimeHeadTokens(outputs, inputs);

    acceptDraftTokens(outputs, inputs);

    sampleNewDraftTokens(outputs, inputs);

    scatterNewDraftTokens(outputs, inputs);

    packAcceptedPaths(outputs, inputs);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::samplePrimeHeadTokens(
    std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs->logits.shape[0];

    auto logits = inputs->logits.template getPtr<T>();
    auto batchSlots = inputs->batch_slots ? inputs->batch_slots->template getPtr<SizeType32 const>() : nullptr;
    auto sequenceLengths = outputs->sequence_length ? outputs->sequence_length->template getPtr<SizeType32>() : nullptr;
    auto tokensPerStepDevice = inputs->medusaCurTokensPerStep.template getPtr<SizeType32>();

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");

    TopKSamplingKernelParams<T> params;
    params.logProbs = logits;
    params.outputIds = mTargetTokensDevice;
    params.workspace = mSamplingWorkspaceDevice;
    params.maxTopK = mRuntimeMaxTopK;
    params.topKs = mRuntimeTopKDevice;
    params.batchSlots = batchSlots;
    params.curandState = mCurandStatesDevice;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getMaxBatchSize();
    params.tokensPerStep = tokensPerStepDevice;
    params.maxTokensPerStep = mDecoderDomain.getMaxTokensPerStep();
    params.maxSeqLen = mDecoderDomain.getMaxTokensPerStep();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    // Sample multiple tokens per request and store them to separate to be accepted/rejected later
    // Sequence length is not modified, endIds is not checked, outputLogProbs are not supported.
    // Finished state is not set.
    invokeBatchTopKSampling(params, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::acceptDraftTokens(
    std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs->logits.shape[0];
    auto const maxSeqLen = outputs->output_ids.shape[outputs->output_ids.shape.size() - 1];

    auto outputIds = outputs->output_ids.template getPtr<TokenIdType>();
    auto endIds = inputs->end_ids.template getPtr<TokenIdType const>();
    auto paths = inputs->paths.template getPtr<SizeType32 const>();

    auto batchSlots = inputs->batch_slots ? inputs->batch_slots->template getPtr<SizeType32 const>() : nullptr;
    auto sequenceLengths = outputs->sequence_length ? outputs->sequence_length->template getPtr<SizeType32>() : nullptr;
    auto acceptedLengths = outputs->medusaOutputs->acceptedLengths.template getPtr<SizeType32>();
    auto curTokensPerStepDevice = inputs->medusaCurTokensPerStep.template getPtr<SizeType32>();
    auto targetTokensPerStepDevice = inputs->medusaTargetTokensPerStep.template getPtr<SizeType32>();

    auto medusaInputLogitsPtrs = BufferRange<T*>(*mMedusaInputLogitsPtrs);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto const slot = batchSlots[bi];
        for (SizeType32 hi = 0; hi < mDecoderDomain.getMaxNumMedusaHeads(); ++hi)
        {
            medusaInputLogitsPtrs[slot * mDecoderDomain.getMaxNumMedusaHeads() + hi]
                = inputs->medusaLogits[slot][hi].template getPtr<T>();
        }
    }

    auto draftIds = outputs->medusaOutputs->nextDraftTokens.template getPtr<TokenIdType>();

    TLLM_CHECK_WITH_INFO(draftIds != nullptr, "Draft ids must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(acceptedLengths != nullptr, "Accepted lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(
        curTokensPerStepDevice != nullptr, "Current tokens per step must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(
        targetTokensPerStepDevice != nullptr, "Target tokens per step must be provided for MedusaDecoding");

    auto finishedStates
        = reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>());

    // Compare draft tokens from outputIds with sampled target tokens at mTargetTokensDevice using paths.
    // Select the longest accepted path, modify outputIds in-place, increment sequenceLengths accordingly.
    // Fill mMedusaSelectedLogitsPtrsDevice with respective Medusa logits
    acceptDraftTokensByIdsWithPaths(outputIds, draftIds, mTargetTokensDevice, sequenceLengths, acceptedLengths,
        finishedStates, batchSlots, paths, endIds,
        reinterpret_cast<T const**>(bufferCast<int64_t>(*mMedusaInputLogitsPtrs)),
        const_cast<T const**>(mMedusaSelectedLogitsPtrsDevice), curTokensPerStepDevice, targetTokensPerStepDevice,
        mBestPathIdsDevice, batchSize, mDecoderDomain.getVocabSize(), mDecoderDomain.getMaxBatchSize(),
        mDecoderDomain.getMaxTokensPerStep(), maxSeqLen, mDecoderDomain.getMaxNumMedusaHeads(),
        mDecoderDomain.getMaxTokensPerStep(), mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::sampleNewDraftTokens(
    std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs->logits.shape[0];
    auto batchSlots = inputs->batch_slots ? inputs->batch_slots->template getPtr<SizeType32 const>() : nullptr;
    auto sequenceLengths
        = (outputs->sequence_length) ? outputs->sequence_length->template getPtr<SizeType32>() : nullptr;

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");

    // For each request we sample Head Num times for topK[hi] tokens
    auto const batchSizeHeadNums = batchSize * mDecoderDomain.getMaxNumMedusaHeads();
    auto const maxBatchSizeHeadNums = mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxNumMedusaHeads();

    auto tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsForward);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType32 hi = 0; hi < mDecoderDomain.getMaxNumMedusaHeads(); ++hi)
        {
            tiledBatchSlots[bi * mDecoderDomain.getMaxNumMedusaHeads() + hi]
                = mDecoderDomain.getMaxNumMedusaHeads() * batchSlots[bi] + hi;
        }
    }

    auto draftIdsPtrs = reinterpret_cast<TokenIdType**>(bufferCast<int64_t>(*mDraftIdsPtrHost));

    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto slot = batchSlots[bi];
        for (SizeType32 hi = 0; hi < mDecoderDomain.getMaxNumMedusaHeads(); ++hi)
        {
            draftIdsPtrs[slot * mDecoderDomain.getMaxNumMedusaHeads() + hi] = mNewDraftTokensDevice
                + slot * mDecoderDomain.getMaxTokensPerStep()
                + mCummulativeTopK[slot * mDecoderDomain.getMaxNumMedusaHeads() + hi];
        }
    }

    TopKSamplingKernelParams<T> params;
    params.logProbsPtrs = const_cast<T const* const*>(mMedusaSelectedLogitsPtrsDevice);
    params.outputIdsPtrs = draftIdsPtrs;
    params.workspace = mSamplingWorkspaceDevice;
    params.maxTopK = mRuntimeMaxTopKPerRequestPerMedusaHead;
    params.topKs = mRuntimeTopKPerRequestPerMedusaHeadDevice;
    params.batchSlots = tiledBatchSlots;
    params.curandState = mCurandStatesMedusaLogitsDevice;
    params.batchSize = batchSizeHeadNums;
    params.maxBatchSize = maxBatchSizeHeadNums;
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.returnAllTopK = true;

    invokeBatchTopKSampling(params, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::scatterNewDraftTokens(
    std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs->logits.shape[0];
    auto batchSlots = inputs->batch_slots ? inputs->batch_slots->template getPtr<SizeType32 const>()
                                          : static_cast<SizeType32*>(nullptr);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");

    auto draftIds = outputs->medusaOutputs->nextDraftTokens.template getPtr<TokenIdType>();
    auto tokensPerStepDevice = inputs->medusaCurTokensPerStep.template getPtr<SizeType32>();
    auto treeIds = inputs->treeIds.template getPtr<SizeType32>();
    TLLM_CHECK_WITH_INFO(draftIds != nullptr, "Draft ids must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(tokensPerStepDevice != nullptr, "Tokens per step must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(treeIds != nullptr, "Tree ids must be provided for MedusaDecoding");

    scatterMedusaDraftTokens(draftIds, mNewDraftTokensDevice, treeIds, tokensPerStepDevice, batchSlots,
        mDecoderDomain.getMaxTokensPerStep(), batchSize, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::packAcceptedPaths(
    std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs->logits.shape[0];
    auto paths = inputs->paths.template getPtr<SizeType32 const>();
    auto batchSlots = inputs->batch_slots ? inputs->batch_slots->template getPtr<SizeType32 const>() : nullptr;
    auto acceptedLengths = outputs->medusaOutputs->acceptedLengths.template getPtr<SizeType32>();
    auto acceptedLengthsCumSum = outputs->medusaOutputs->acceptedLengthsCumSum.template getPtr<SizeType32>();
    auto pathsOffsets = outputs->medusaOutputs->pathsOffsets.template getPtr<SizeType32>();

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(acceptedLengths != nullptr, "Accepted lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(acceptedLengthsCumSum != nullptr, "acceptedLengthsCumSum must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for MedusaDecoding");
    invokePackAcceptedPaths(acceptedLengthsCumSum, pathsOffsets, acceptedLengths, mBestPathIdsDevice, paths, batchSlots,
        batchSize, mDecoderDomain.getMaxTokensPerStep(), mDecoderDomain.getMaxNumMedusaHeads() + 1, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class MedusaDecodingLayer<float>;
template class MedusaDecodingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
