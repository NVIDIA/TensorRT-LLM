/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/medusaDecodingKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::kernels::speculative_decoding;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
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

    auto const maxDraftPathLen = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDraftPathLen();
    // Get sampling workspace size
    {
        auto samplingSizePrimarySampling = getTopKWorkspaceSize<T>(mDecoderDomain.getBatchSize(),
            mDecoderDomain.getMaxDecodingTokens(), TOP_K_MAX, mDecoderDomain.getVocabSizePadded());

        auto const maxBatchSizeHeadNums = mDecoderDomain.getBatchSize() * maxDraftPathLen;
        auto samplingSizeMedusaHeadsSampling
            = getTopKWorkspaceSize<T>(maxBatchSizeHeadNums, 1, TOP_K_MAX, mDecoderDomain.getVocabSizePadded());

        mWorkspaceSize = std::max(samplingSizePrimarySampling, samplingSizeMedusaHeadsSampling);
    }

    mDraftIdsPtrHost = runtime::BufferManager::pinned(
        ITensor::makeShape({static_cast<SizeType32>(mDecoderDomain.getBatchSize()), maxDraftPathLen}),
        runtime::TRTDataType<TokenIdType*>::value);
    mCummulativeTopK.resize(mDecoderDomain.getBatchSize() * maxDraftPathLen);

    std::array<size_t, 11> deviceBufferSizes;
    deviceBufferSizes[0] = mDecoderDomain.getBatchSize() * sizeof(curandState_t);
    deviceBufferSizes[1] = mDecoderDomain.getBatchSize() * maxDraftPathLen * sizeof(SizeType32);
    deviceBufferSizes[2] = mWorkspaceSize;
    deviceBufferSizes[3] = mDecoderDomain.getBatchSize() * sizeof(SizeType32);
    deviceBufferSizes[4] = mDecoderDomain.getBatchSize() * mDecoderDomain.getMaxDecodingTokens() * sizeof(TokenIdType);
    deviceBufferSizes[5] = mDecoderDomain.getBatchSize() * maxDraftPathLen * sizeof(uint64_t);
    deviceBufferSizes[6] = mDecoderDomain.getBatchSize() * maxDraftPathLen * sizeof(T*);
    deviceBufferSizes[7] = mDecoderDomain.getBatchSize() * maxDraftPathLen * sizeof(curandState_t);
    deviceBufferSizes[8] = mDecoderDomain.getBatchSize() * maxDraftPathLen * sizeof(SizeType32);
    deviceBufferSizes[9] = mDecoderDomain.getBatchSize() * mDecoderDomain.getMaxDecodingTokens() * sizeof(TokenIdType);
    deviceBufferSizes[10] = mDecoderDomain.getBatchSize() * sizeof(SizeType32);

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
        ITensor::makeShape({static_cast<SizeType32>(mDecoderDomain.getBatchSize() * maxDraftPathLen)}),
        nvinfer1::DataType::kINT32);
    mTiledBatchSlotsForward = BufferManager::pinnedPool(
        ITensor::makeShape({static_cast<SizeType32>(mDecoderDomain.getBatchSize() * maxDraftPathLen)}),
        nvinfer1::DataType::kINT32);
    mMedusaInputLogitsPtrs = BufferManager::pinnedPool(
        ITensor::makeShape({static_cast<SizeType32>(mDecoderDomain.getBatchSize() * maxDraftPathLen)}),
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
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
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

    auto const maxDraftPathLen = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDraftPathLen();
    auto const batchSizeMaxNumHeads = batchSize * maxDraftPathLen;
    auto randomSeed = setupParams->randomSeed.value_or(std::vector<uint64_t>(batchSize, uint64_t{0}));
    std::vector<uint64_t> tiledRandomSeed(batchSizeMaxNumHeads);
    if (randomSeed.size() > 1)
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
            {
                tiledRandomSeed[bi * maxDraftPathLen + hi] = randomSeed[bi];
            }
        }
    }
    auto tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsSetup);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            tiledBatchSlots[bi * maxDraftPathLen + hi] = batchSlots[bi] + hi;
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
            for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
            {
                mCummulativeTopK[slot * maxDraftPathLen + hi] = cummulativeTopK;
                cummulativeTopK += runtimeHeadsTopKFlatten[bi * maxDraftPathLen + hi];
            }
        }

        auto tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsSetup);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
            {
                tiledBatchSlots[bi * maxDraftPathLen + hi] = maxDraftPathLen * batchSlots[bi] + hi;
            }
        }

        auto const curMaxTopK = prepareRuntimeTopK(runtimeHeadsTopKFlatten,
            static_cast<SizeType32>(batchSizeMaxNumHeads), tiledBatchSlots, mRuntimeTopKPerRequestPerMedusaHeadDevice);
        mRuntimeMaxTopKPerRequestPerMedusaHead = std::max(mRuntimeMaxTopKPerRequestPerMedusaHead, curMaxTopK);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<MedusaDecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<SpeculativeDecodingOutputs>(baseOutputs);

    samplePrimeHeadTokens(*outputs, *inputs);

    acceptDraftTokens(*outputs, *inputs);

    sampleNewDraftTokens(*outputs, *inputs);

    scatterNewDraftTokens(*outputs, *inputs);

    packAcceptedPaths(*outputs, *inputs);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::samplePrimeHeadTokens(
    SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits->shape[0];

    auto logits = inputs.logits->template getPtr<T>();
    auto batchSlots = inputs.batchSlots ? inputs.batchSlots->template getPtr<SizeType32 const>() : nullptr;
    auto sequenceLengths = outputs.sequenceLength ? outputs.sequenceLength->template getPtr<SizeType32>() : nullptr;
    auto tokensPerStepDevice = inputs.curTokensPerStep->template getPtr<SizeType32>();

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
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.tokensPerStep = tokensPerStepDevice;
    params.maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    params.maxSeqLen = mDecoderDomain.getMaxDecodingTokens();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    // Sample multiple tokens per request and store them to separate to be accepted/rejected later
    // Sequence length is not modified, endIds is not checked, outputLogProbs are not supported.
    // Finished state is not set.
    invokeBatchTopKSampling(params, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::acceptDraftTokens(
    SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits->shape[0];
    auto const maxSeqLen = outputs.outputIds.shape[outputs.outputIds.shape.size() - 1];

    auto outputIds = outputs.outputIds.template getPtr<TokenIdType>();
    auto endIds = inputs.endIds.template getPtr<TokenIdType const>();
    auto paths = inputs.paths.template getPtr<SizeType32 const>();

    auto batchSlots = inputs.batchSlots ? inputs.batchSlots->template getPtr<SizeType32 const>() : nullptr;
    auto sequenceLengths = outputs.sequenceLength ? outputs.sequenceLength->template getPtr<SizeType32>() : nullptr;
    auto numNewTokens = outputs.numNewTokens->template getPtr<SizeType32>();
    auto curTokensPerStepDevice = inputs.curTokensPerStep->template getPtr<SizeType32>();
    auto targetTokensPerStepDevice = inputs.targetTokensPerStep.template getPtr<SizeType32>();

    auto const maxDraftPathLen = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDraftPathLen();

    auto medusaInputLogitsPtrs = BufferRange<T*>(*mMedusaInputLogitsPtrs);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto const slot = batchSlots[bi];
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            medusaInputLogitsPtrs[slot * maxDraftPathLen + hi] = inputs.medusaLogits[slot][hi].template getPtr<T>();
        }
    }

    auto draftIds = outputs.nextDraftTokens.template getPtr<TokenIdType>();

    TLLM_CHECK_WITH_INFO(draftIds != nullptr, "Draft ids must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(
        curTokensPerStepDevice != nullptr, "Current tokens per step must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(
        targetTokensPerStepDevice != nullptr, "Target tokens per step must be provided for MedusaDecoding");

    auto finishedStates
        = reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>());

    // Compare draft tokens from outputIds with sampled target tokens at mTargetTokensDevice using paths.
    // Select the longest accepted path, modify outputIds in-place, increment sequenceLengths accordingly.
    // Fill mMedusaSelectedLogitsPtrsDevice with respective Medusa logits
    acceptDraftTokensByIdsWithPaths(outputIds, draftIds, mTargetTokensDevice, sequenceLengths, numNewTokens,
        finishedStates, batchSlots, paths, endIds,
        reinterpret_cast<T const**>(bufferCast<int64_t>(*mMedusaInputLogitsPtrs)),
        const_cast<T const**>(mMedusaSelectedLogitsPtrsDevice), curTokensPerStepDevice, targetTokensPerStepDevice,
        mBestPathIdsDevice, batchSize, mDecoderDomain.getVocabSize(), mDecoderDomain.getBatchSize(), maxSeqLen,
        maxDraftPathLen, mDecoderDomain.getMaxDecodingTokens(), mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::sampleNewDraftTokens(
    SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits->shape[0];
    auto batchSlots = inputs.batchSlots ? inputs.batchSlots->template getPtr<SizeType32 const>() : nullptr;
    auto sequenceLengths = (outputs.sequenceLength) ? outputs.sequenceLength->template getPtr<SizeType32>() : nullptr;

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");

    auto const maxDraftPathLen = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDraftPathLen();
    // For each request we sample Head Num times for topK[hi] tokens
    auto const batchSizeHeadNums = batchSize * maxDraftPathLen;
    auto const maxBatchSizeHeadNums = mDecoderDomain.getBatchSize() * maxDraftPathLen;

    auto tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsForward);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            tiledBatchSlots[bi * maxDraftPathLen + hi] = maxDraftPathLen * batchSlots[bi] + hi;
        }
    }

    auto draftIdsPtrs = reinterpret_cast<TokenIdType**>(bufferCast<int64_t>(*mDraftIdsPtrHost));

    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto slot = batchSlots[bi];
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            draftIdsPtrs[slot * maxDraftPathLen + hi] = mNewDraftTokensDevice
                + slot * mDecoderDomain.getMaxDecodingTokens() + mCummulativeTopK[slot * maxDraftPathLen + hi];
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
    SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits->shape[0];
    auto batchSlots = inputs.batchSlots ? inputs.batchSlots->template getPtr<SizeType32 const>()
                                        : static_cast<SizeType32*>(nullptr);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");

    auto draftIds = outputs.nextDraftTokens.template getPtr<TokenIdType>();
    auto tokensPerStepDevice = inputs.curTokensPerStep->template getPtr<SizeType32>();
    auto treeIds = inputs.treeIds.template getPtr<SizeType32>();
    TLLM_CHECK_WITH_INFO(draftIds != nullptr, "Draft ids must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(tokensPerStepDevice != nullptr, "Tokens per step must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(treeIds != nullptr, "Tree ids must be provided for MedusaDecoding");

    scatterMedusaDraftTokens(draftIds, mNewDraftTokensDevice, treeIds, tokensPerStepDevice, batchSlots,
        mDecoderDomain.getMaxDecodingTokens(), batchSize, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::packAcceptedPaths(
    SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits->shape[0];
    auto paths = inputs.paths.template getPtr<SizeType32 const>();
    auto batchSlots = inputs.batchSlots ? inputs.batchSlots->template getPtr<SizeType32 const>() : nullptr;
    auto numNewTokens = outputs.numNewTokens->template getPtr<SizeType32>();
    auto numNewTokensCumSum = outputs.numNewTokensCumSum.template getPtr<SizeType32>();
    auto pathsOffsets = outputs.pathsOffsets.template getPtr<SizeType32>();

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(numNewTokensCumSum != nullptr, "numNewTokensCumSum must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for MedusaDecoding");
    invokePackAcceptedPaths(numNewTokensCumSum, pathsOffsets, numNewTokens, mBestPathIdsDevice, paths, batchSlots,
        batchSize, mDecoderDomain.getMaxDecodingTokens(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen(), false, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class MedusaDecodingLayer<float>;
template class MedusaDecodingLayer<half>;

} // namespace tensorrt_llm::layers
