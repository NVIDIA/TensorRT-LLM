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

#include "medusaDecodingLayer.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/medusaDecodingKernels.h"
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
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

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

    mDraftIdsPtrHost = BufferManager::pinnedPool(
        ITensor::makeShape({static_cast<SizeType32>(mDecoderDomain.getBatchSize()), maxDraftPathLen}),
        TRTDataType<TokenIdType*>::value);
    mCummulativeTopK.resize(mDecoderDomain.getBatchSize() * maxDraftPathLen);

    auto const batchSize = mDecoderDomain.getBatchSize();
    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mCurandStatesDevice = mBufferManager->gpu(
        ITensor::makeShape({static_cast<int32_t>(batchSize * sizeof(curandState_t))}), TRTDataType<int8_t>::value);
    mRuntimeTopKDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mTargetTokensDevice = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getMaxDecodingTokens()}), TRTDataType<TokenIdType>::value);
    mRandomSeedsDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize * maxDraftPathLen}), TRTDataType<uint64_t>::value);
    mMedusaSelectedLogitsPtrsDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize, maxDraftPathLen}), TRTDataType<T*>::value);
    mCurandStatesMedusaLogitsDevice = mBufferManager->gpu(
        ITensor::makeShape({batchSize, maxDraftPathLen, sizeof(curandState_t)}), TRTDataType<int8_t>::value);
    mRuntimeTopKPerRequestPerMedusaHeadDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize, maxDraftPathLen}), TRTDataType<SizeType32>::value);
    mNewDraftTokensDevice = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getMaxDecodingTokens()}), TRTDataType<TokenIdType>::value);
    mBestPathIdsDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);

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
void MedusaDecodingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(MedusaDecodingLayer_setup);

    auto setupParams = std::dynamic_pointer_cast<MedusaSetupParams>(baseSetupParams);

    workspace->initializeDeviceCurandStates(
        setupParams->randomSeed, batchSize, workspace->getDeviceBatchSlots(), mCurandStatesDevice);

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
    auto* tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsSetup);
    BufferRange<SizeType32 const> batchSlotsRange(*batchSlots);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            tiledBatchSlots[bi * maxDraftPathLen + hi] = batchSlotsRange[bi] + hi;
        }
    }
    auto tiledRandomSeedOpt = std::make_optional(std::move(tiledRandomSeed));
    workspace->initializeDeviceCurandStates(
        tiledRandomSeedOpt, batchSizeMaxNumHeads, mTiledBatchSlotsSetup, mCurandStatesMedusaLogitsDevice);

    // Prepare runtime top K
    auto prepareRuntimeTopK = [this, workspace](std::vector<SizeType32> const& runtimeTopK, SizeType32 batchSize,
                                  BufferConstPtr const& batchSlots, BufferPtr const& runtimeTopKDevice)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopK.size() == 1 || runtimeTopK.size() == static_cast<size_t>(batchSize),
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopK.size(), batchSize));
        SizeType32* topKSetupPtr = nullptr;
        if (runtimeTopK.size() > 1)
        {
            DecodingLayerWorkspace::copyToWorkspace<SizeType32>(
                *this->mBufferManager, runtimeTopK, workspace->getWorkspaceDeviceBuffer());
            topKSetupPtr = workspace->getWorkspaceDevicePtrAs<SizeType32>();
        }
        auto* runtimeTopKDevicePtr = bufferCastOrNull<SizeType32>(runtimeTopKDevice);
        auto const* batchSlotsPtr = bufferCastOrNull<SizeType32 const>(batchSlots);
        invokeScatterDecodingParams(
            topKSetupPtr, runtimeTopK.front(), runtimeTopKDevicePtr, batchSlotsPtr, batchSize, getStream());

        // FIXME: monotonically growing
        auto const curMaxTopK = *std::max_element(std::begin(runtimeTopK), std::end(runtimeTopK));
        return curMaxTopK;
    };

    SizeType32 constexpr defaultTopK = 1;
    {
        auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector{defaultTopK});
        auto const curMaxTopK
            = prepareRuntimeTopK(runtimeTopK, batchSize, workspace->getDeviceBatchSlots(), mRuntimeTopKDevice);
        mRuntimeMaxTopK = std::max(mRuntimeMaxTopK, curMaxTopK);
    }
    {
        auto runtimeHeadsTopK = setupParams->runtimeHeadsTopK;
        std::vector<SizeType32> runtimeHeadsTopKFlatten;
        if (runtimeHeadsTopK.has_value() && static_cast<bool>(runtimeHeadsTopK->size()))
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

        BufferRange<SizeType32 const> batchSlotsRange(*batchSlots);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const slot = batchSlotsRange[bi];
            SizeType32 cummulativeTopK = 0;
            for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
            {
                mCummulativeTopK[slot * maxDraftPathLen + hi] = cummulativeTopK;
                cummulativeTopK += runtimeHeadsTopKFlatten[bi * maxDraftPathLen + hi];
            }
        }

        auto* tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsSetup);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
            {
                tiledBatchSlots[bi * maxDraftPathLen + hi] = maxDraftPathLen * batchSlotsRange[bi] + hi;
            }
        }

        auto const curMaxTopK
            = prepareRuntimeTopK(runtimeHeadsTopKFlatten, static_cast<SizeType32>(batchSizeMaxNumHeads),
                mTiledBatchSlotsSetup, mRuntimeTopKPerRequestPerMedusaHeadDevice);
        mRuntimeMaxTopKPerRequestPerMedusaHead = std::max(mRuntimeMaxTopKPerRequestPerMedusaHead, curMaxTopK);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(MedusaDecodingLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<MedusaDecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<SpeculativeDecodingOutputs>(baseOutputs);

    // TODO add typical acceptance similarly to EagleSampleAndAcceptDraftTokensPlugin::doTypicalAcceptance.
    samplePrimeHeadTokens(*outputs, *inputs, workspace);

    acceptDraftTokens(*outputs, *inputs, workspace);

    sampleNewDraftTokens(*outputs, *inputs, workspace);

    scatterNewDraftTokens(*outputs, *inputs);

    packAcceptedPaths(*outputs, *inputs, workspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t MedusaDecodingLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mWorkspaceSize, mSetupWorkspaceSize);
}

template <typename T>
void MedusaDecodingLayer<T>::samplePrimeHeadTokens(SpeculativeDecodingOutputs const& outputs,
    MedusaDecodingInputs const& inputs, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.value()->getDimension<0>();

    auto logits = bufferCast<T>(*inputs.logits.value());
    auto const* batchSlots = workspace->getDeviceBatchSlotsPtr();
    auto* sequenceLengths = bufferCastOrNull<SizeType32>(outputs.sequenceLength);
    auto* tokensPerStepDevice = bufferCast<SizeType32>(*inputs.curTokensPerStep.value());

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");

    TopKSamplingKernelParams<T> params;
    params.logProbs = logits;
    params.outputIds = bufferCastOrNull<SizeType32>(mTargetTokensDevice);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.maxTopK = mRuntimeMaxTopK;
    params.topKs = bufferCastOrNull<SizeType32>(mRuntimeTopKDevice);
    params.batchSlots = batchSlots;
    params.curandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesDevice));
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.tokensPerStep = tokensPerStepDevice;
    params.maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    params.maxSeqLen = mDecoderDomain.getMaxDecodingTokens();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    // Sample multiple tokens per request and store them to separate to be accepted/rejected later
    // Sequence length is not modified, endIds is not checked, outputLogProbs are not supported.
    // Finished state is not set.
    invokeBatchTopKSampling(params, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::acceptDraftTokens(SpeculativeDecodingOutputs const& outputs,
    MedusaDecodingInputs const& inputs, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.value()->getDimension<0>();
    auto const maxSeqLen = outputs.outputIds->getDimension<-1>();

    auto* outputIds = bufferCast<TokenIdType>(*outputs.outputIds);
    auto const* endIds = bufferCast<TokenIdType>(*inputs.endIds);
    auto const* paths = bufferCast<SizeType32>(*inputs.paths);

    auto const* batchSlots = bufferCast<SizeType32>(*inputs.batchSlots);
    auto* sequenceLengths = bufferCastOrNull<SizeType32>(outputs.sequenceLength);
    auto* numNewTokens = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    auto* curTokensPerStepDevice = bufferCast<SizeType32>(*inputs.curTokensPerStep.value());
    auto const* targetTokensPerStepDevice = bufferCast<SizeType32>(*inputs.targetTokensPerStep);

    auto const maxDraftPathLen = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDraftPathLen();

    auto medusaInputLogitsPtrs = BufferRange<T*>(*mMedusaInputLogitsPtrs);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto const slot = batchSlots[bi];
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            medusaInputLogitsPtrs[slot * maxDraftPathLen + hi] = bufferCast<T>(*inputs.medusaLogits[slot][hi]);
        }
    }

    auto* draftIds = bufferCast<TokenIdType>(*outputs.nextDraftTokens);

    TLLM_CHECK_WITH_INFO(draftIds != nullptr, "Draft ids must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(
        curTokensPerStepDevice != nullptr, "Current tokens per step must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(
        targetTokensPerStepDevice != nullptr, "Target tokens per step must be provided for MedusaDecoding");

    // Compare draft tokens from outputIds with sampled target tokens at mTargetTokensDevice using paths.
    // Select the longest accepted path, modify outputIds in-place, increment sequenceLengths accordingly.
    // Fill mMedusaSelectedLogitsPtrsDevice with respective Medusa logits
    auto* targetTokensDevicePtr = bufferCast<SizeType32>(*mTargetTokensDevice);
    auto* finishedStatesPtr
        = reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs.finished));
    auto* bestPathIdsDevicePtr = bufferCastOrNull<SizeType32>(mBestPathIdsDevice);
    auto medusaInputLogitsPtrsPtr = reinterpret_cast<T const**>(bufferCast<int64_t>(*mMedusaInputLogitsPtrs));
    auto medusaSelectedLogitsPtrsDevicePtr
        = const_cast<T const**>(bufferCastOrNull<T const*>(mMedusaSelectedLogitsPtrsDevice));

    AcceptDraftTokensByIdsWithPathsParams<T> params;
    params.outputIds = outputIds;
    params.draftIds = draftIds;
    params.targetIds = targetTokensDevicePtr;
    params.sequenceLengths = sequenceLengths;
    params.acceptedLengths = numNewTokens;
    params.finishedFinal = finishedStatesPtr;
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.paths = paths;
    params.endIds = endIds;
    params.medusaLogits = medusaInputLogitsPtrsPtr;
    params.logitsPtrs = medusaSelectedLogitsPtrsDevicePtr;
    params.curTokensPerStep = curTokensPerStepDevice;
    params.targetTokensPerStep = targetTokensPerStepDevice;
    params.bestPathIds = bestPathIdsDevicePtr;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSize = mDecoderDomain.getVocabSize();
    params.maxSeqLen = maxSeqLen;
    params.maxDraftPathLen = maxDraftPathLen;
    params.maxDecodingTokens = mDecoderDomain.getMaxDecodingTokens();
    params.stream = getStream();

    params.checkParams();

    acceptDraftTokensByIdsWithPaths(params);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::sampleNewDraftTokens(SpeculativeDecodingOutputs const& outputs,
    MedusaDecodingInputs const& inputs, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.value()->getDimension<0>();
    auto const* batchSlots = bufferCast<SizeType32>(*inputs.batchSlots);
    auto* sequenceLengths = bufferCastOrNull<SizeType32>(outputs.sequenceLength);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(sequenceLengths != nullptr, "Sequence lengths must be provided for MedusaDecoding");

    auto const maxDraftPathLen = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDraftPathLen();
    // For each request we sample Head Num times for topK[hi] tokens
    auto const batchSizeHeadNums = batchSize * maxDraftPathLen;
    auto const maxBatchSizeHeadNums = mDecoderDomain.getBatchSize() * maxDraftPathLen;

    auto* tiledBatchSlots = bufferCast<SizeType32>(*mTiledBatchSlotsForward);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            tiledBatchSlots[bi * maxDraftPathLen + hi] = maxDraftPathLen * batchSlots[bi] + hi;
        }
    }

    auto* draftIdsPtrs = reinterpret_cast<TokenIdType**>(bufferCast<int64_t>(*mDraftIdsPtrHost));

    auto* newDraftTokensDeviceRange = bufferCast<TokenIdType>(*mNewDraftTokensDevice);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto slot = batchSlots[bi];
        for (SizeType32 hi = 0; hi < maxDraftPathLen; ++hi)
        {
            draftIdsPtrs[slot * maxDraftPathLen + hi] = newDraftTokensDeviceRange
                + slot * mDecoderDomain.getMaxDecodingTokens() + mCummulativeTopK[slot * maxDraftPathLen + hi];
        }
    }

    TopKSamplingKernelParams<T> params{};
    params.logProbsPtrs = bufferCastOrNull<T const*>(mMedusaSelectedLogitsPtrsDevice);
    params.outputIdsPtrs = draftIdsPtrs;
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.maxTopK = mRuntimeMaxTopKPerRequestPerMedusaHead;
    params.topKs = bufferCastOrNull<SizeType32>(mRuntimeTopKPerRequestPerMedusaHeadDevice);
    params.batchSlots = tiledBatchSlots;
    params.curandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesMedusaLogitsDevice));
    params.batchSize = batchSizeHeadNums;
    params.maxBatchSize = maxBatchSizeHeadNums;
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.returnAllSelectedTokens = true;

    invokeBatchTopKSampling(params, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::scatterNewDraftTokens(
    SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.value()->getDimension<0>();
    auto const* batchSlots = bufferCast<SizeType32>(*inputs.batchSlots);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");

    auto* draftIds = bufferCastOrNull<TokenIdType>(outputs.nextDraftTokens);
    auto* tokensPerStepDevice = bufferCastOrNull<SizeType32>(inputs.curTokensPerStep);
    auto const* treeIds = bufferCastOrNull<SizeType32>(inputs.treeIds);
    TLLM_CHECK_WITH_INFO(draftIds != nullptr, "Draft ids must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(tokensPerStepDevice != nullptr, "Tokens per step must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(treeIds != nullptr, "Tree ids must be provided for MedusaDecoding");

    auto* newDraftTokensDevice = bufferCastOrNull<TokenIdType>(mNewDraftTokensDevice);
    scatterMedusaDraftTokens(draftIds, newDraftTokensDevice, treeIds, tokensPerStepDevice, batchSlots,
        mDecoderDomain.getMaxDecodingTokens(), batchSize, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MedusaDecodingLayer<T>::packAcceptedPaths(SpeculativeDecodingOutputs const& outputs,
    MedusaDecodingInputs const& inputs, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.value()->getDimension<0>();
    auto const* paths = bufferCast<SizeType32>(*inputs.paths);
    auto const* batchSlots = workspace->getDeviceBatchSlotsPtr();
    auto* numNewTokens = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    auto* numNewTokensCumSum = bufferCast<SizeType32>(*outputs.numNewTokensCumSum);
    auto* pathsOffsets = bufferCast<SizeType32>(*outputs.pathsOffsets);
    auto* bestPathIdsDevicePtr = bufferCastOrNull<SizeType32>(mBestPathIdsDevice);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(numNewTokensCumSum != nullptr, "numNewTokensCumSum must be provided for MedusaDecoding");
    TLLM_CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for MedusaDecoding");
    invokePackAcceptedPaths(numNewTokensCumSum, pathsOffsets, numNewTokens, bestPathIdsDevicePtr, paths, batchSlots,
        batchSize, batchSize, mDecoderDomain.getMaxDecodingTokens(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen(), false, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class MedusaDecodingLayer<float>;
template class MedusaDecodingLayer<half>;

} // namespace tensorrt_llm::layers
