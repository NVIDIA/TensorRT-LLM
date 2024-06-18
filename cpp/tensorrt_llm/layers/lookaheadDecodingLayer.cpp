/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/lookaheadDecodingLayer.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <cstdint>
#include <ios>
#include <memory>
#include <tuple>

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

template <typename T>
LookaheadDecodingLayer<T>::CpuAlgorithmResources::CpuAlgorithmResources(DecoderDomain const& decoderDomain)
{
    auto maxBatchSize = decoderDomain.getBatchSize();
    auto lookaheadModule
        = std::dynamic_pointer_cast<LookaheadModule const>(decoderDomain.getSpeculativeDecodingModule());
    auto const [maxW, maxN, maxG] = lookaheadModule->getExecutionConfig().get();

    for (runtime::SizeType32 id = 0; id < maxBatchSize; id++)
    {
        mAlgos.emplace_back(maxW, maxN, maxG, id);
    }

    SizeType32 maxTokensPerStep, maxNumNewTokens, maxDraftLen;
    std::tie(maxTokensPerStep, maxNumNewTokens, maxDraftLen, std::ignore)
        = executor::LookaheadDecodingConfig(maxW, maxN, maxG).calculateSpeculativeResource();

    auto const maxBatchShape1D = ITensor::makeShape({maxBatchSize});
    mBatchSlots = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mTargetTokens
        = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxTokensPerStep}), nvinfer1::DataType::kINT32);
    mTokensPerStep = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mEndIds = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);

    mOutputIds = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxNumNewTokens}), nvinfer1::DataType::kINT32);
    mPathsOffsets = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxNumNewTokens}), nvinfer1::DataType::kINT32);
    mNumNewTokens = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mNumNewTokensCumSum = BufferManager::cpu(ITensor::makeShape({maxBatchSize + 1}), nvinfer1::DataType::kINT32);
    mNextDraftTokens = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxDraftLen}), nvinfer1::DataType::kINT32);
    mNextDraftPosIds = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxDraftLen}), nvinfer1::DataType::kINT32);
    auto divUp32 = [](SizeType32 x) { return x / 32 + ((x % 32) ? 1 : 0); };
    mPackedMasks = BufferManager::cpu(
        ITensor::makeShape({maxBatchSize, maxTokensPerStep, divUp32(maxTokensPerStep)}), nvinfer1::DataType::kINT32);
    mSamplingMask = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxDraftLen}), nvinfer1::DataType::kBOOL);
    mNextDraftLengths = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mSequenceLengths = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
}

template <typename T>
LookaheadDecodingLayer<T>::LookaheadDecodingLayer(
    DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> const& bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mCpuAlgo(std::make_optional<CpuAlgorithmResources>(decoderDomain))
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const maxBatchSize = mDecoderDomain.getBatchSize();
    auto const maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    auto const vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    auto const maxTopK = 1;
    auto const maxBatchShape1D = ITensor::makeShape({maxBatchSize});
    auto const maxBatchShape2D = ITensor::makeShape({maxBatchSize, maxTokensPerStep});

    mWorkspaceSize = getTopKWorkspaceSize<T>(maxBatchSize, maxTokensPerStep, maxTopK, vocabSizePadded);
    TLLM_LOG_DEBUG("mWorkspaceSize=%d", mWorkspaceSize);

    mSamplingWorkspaceDevice
        = mBufferManager->gpu(ITensor::makeShape({static_cast<int32_t>(mWorkspaceSize)}), nvinfer1::DataType::kINT8);
    mTargetTokensDevice = mBufferManager->gpu(maxBatchShape2D, nvinfer1::DataType::kINT32);
    mRandomSeedsDevice = mBufferManager->gpu(maxBatchShape1D, nvinfer1::DataType::kINT64);
    mSamplingMaskDevice = mBufferManager->gpu(maxBatchShape2D, nvinfer1::DataType::kBOOL);
    mCurandStatesDevice = mBufferManager->gpu(maxBatchShape1D, nvinfer1::DataType::kINT8);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void LookaheadDecodingLayer<T>::setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth,
    runtime::SizeType32 const* batchSlots, std::shared_ptr<BaseSetupParams> const& baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<LookaheadSetupParams>(baseSetupParams);

    if (mCpuAlgo)
    {
        auto& algoConfigs = setupParams->algoConfigs;
        TLLM_CHECK_WITH_INFO(algoConfigs.size() == 1 || algoConfigs.size() == batchSize,
            "Lookahead runtime configuration size should be either 1 or batchSize");
        for (runtime::SizeType32 bi = 0; bi < batchSize; bi++)
        {
            SizeType32 gbi = batchSlots[bi];
            SizeType32 bi1orN = (algoConfigs.size() == 1) ? 0 : bi;
            TLLM_LOG_DEBUG("CPU ALGO [ %d ] setup", gbi);
            PRINT_TOKENS(setupParams->prompt[bi]);
            auto [w, n, g] = algoConfigs[bi1orN].get();
            SizeType32 runtimeTokensPerStep;
            std::tie(runtimeTokensPerStep, std::ignore, std::ignore, std::ignore)
                = executor::LookaheadDecodingConfig(w, n, g).calculateSpeculativeResource();
            TLLM_CHECK_WITH_INFO(runtimeTokensPerStep <= mDecoderDomain.getMaxDecodingTokens(),
                "runtime w(%d) n(%d) g(%d) exceeds maxTokensPerStep(%d)", w, n, g,
                mDecoderDomain.getMaxDecodingTokens());
            mCpuAlgo->mAlgos[gbi].setup(setupParams->prompt[bi], w, n, g);
        }
    }

    auto curandStatesDevicePtr = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    if (setupParams->randomSeed)
    {
        auto& randomSeed = setupParams->randomSeed.value();
        if (randomSeed.size() == 1)
        {
            invokeCurandInitialize(curandStatesDevicePtr, batchSlots, batchSize, randomSeed.front(), mStream);
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(randomSeed.size() == batchSize, "Random seed vector size mismatch.");
            cudaAutoCpy(bufferCast<uint64_t>(*mRandomSeedsDevice), randomSeed.data(), batchSize, mStream);
            invokeCurandBatchInitialize(
                curandStatesDevicePtr, batchSlots, batchSize, bufferCast<uint64_t>(*mRandomSeedsDevice), mStream);
            sync_check_cuda_error();
        }
    }
    else
    {
        invokeCurandInitialize(curandStatesDevicePtr, batchSlots, batchSize, DefaultDecodingParams::getSeed(), mStream);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void LookaheadDecodingLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& outputParams, std::shared_ptr<BaseDecodingInputs> const& inputParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto inputs = std::dynamic_pointer_cast<LookaheadDecodingInputs>(inputParams);
    auto outputs = std::dynamic_pointer_cast<SpeculativeDecodingOutputs>(outputParams);
    auto batchSize = inputs->localBatchSize;

    TLLM_CHECK_WITH_INFO(inputs->batchSlots, "Batch slots must be provided for LookaheadDecoding");
    TLLM_CHECK_WITH_INFO(inputs->curTokensPerStep, "curTokensPerStep must be provided for LookaheadDecoding");
    TLLM_CHECK_WITH_INFO(outputs->sequenceLength, "sequenceLength must be provided for LookaheadDecoding");
    // TODO(liweim) to be confirmed.
    TLLM_CHECK(inputs->logits);

    mBufferManager->copy(
        inputs->batchSlots->template getPtr<SizeType32 const>(), *mCpuAlgo->mBatchSlots, runtime::MemoryType::kGPU);
    mBufferManager->copy(inputs->curTokensPerStep->template getPtr<SizeType32 const>(), *mCpuAlgo->mTokensPerStep,
        runtime::MemoryType::kGPU);
    mBufferManager->copy(
        inputs->endIds.template getPtr<TokenIdType const>(), *mCpuAlgo->mEndIds, runtime::MemoryType::kGPU);
    mBufferManager->copy(outputs->sequenceLength->template getPtr<SizeType32 const>(), *mCpuAlgo->mSequenceLengths,
        runtime::MemoryType::kGPU);

    TopKSamplingKernelParams<T> params;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.batchSize = batchSize;
    params.maxTopK = 1;
    params.returnAllTopK = true;
    params.maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    params.maxSeqLen = mDecoderDomain.getMaxDecodingTokens();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.batchSlots = inputs->batchSlots->template getPtr<SizeType32 const>();
    TLLM_LOG_DEBUG("batchSize = %d", batchSize);
    params.logProbs = inputs->logits ? inputs->logits->template getPtr<T>() : nullptr;
    params.outputIds = bufferCast<TokenIdType>(*mTargetTokensDevice);
    params.workspace = bufferCast<int8_t>(*mSamplingWorkspaceDevice);
    params.curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    params.tokensPerStep = inputs->curTokensPerStep->template getPtr<SizeType32 const>();

    TLLM_LOG_DEBUG(
        "invokeBatchTopKSampling: maxBatchSize=%d, batchSize=%d, maxTopK=%d, maxTokensPerStep=%d, maxSeqLen=%d, "
        "vocabSizePadded=%d",
        params.maxBatchSize, params.batchSize, params.maxTopK, params.maxTokensPerStep, params.maxSeqLen,
        params.vocabSizePadded);

    // Sample multiple tokens per request and store them to separate to be accepted/rejected later
    // Sequence length is not modified, endIds is not checked, outputLogProbs are not supported.
    // Finished state is not set.
    invokeBatchTopKSampling(params, mStream);

    mBufferManager->copy(*mTargetTokensDevice, *mCpuAlgo->mTargetTokens);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void LookaheadDecodingLayer<T>::forwardSync(
    std::shared_ptr<BaseDecodingOutputs> const& outputParams, std::shared_ptr<BaseDecodingInputs> const& inputParams)
{
    if (mCpuAlgo)
    {
        forwardSyncCPU(outputParams, inputParams);
    }
}

template <typename T>
void LookaheadDecodingLayer<T>::posIdsToMask(TensorPtr mask, TensorConstPtr posIds)
{
    auto len = ITensor::volume(posIds->getShape());
    TLLM_CHECK(mask->getShape().d[0] > len);
    TLLM_CHECK(mask->getShape().d[1] * 32 > len);
    auto posIdsRange = BufferRange<SizeType32 const>(*posIds);
    auto maskLocation = BufferLocation<SizeType32>(*mask);

    for (auto i = 0; i < maskLocation.size(); i++)
    {
        maskLocation[i] = 0;
    }
    maskLocation.at(0, 0) = 1;

    auto setBit = [](SizeType32& x, SizeType32 idx) { x |= (1 << idx); };
    if (len > 0)
    {
        std::vector<std::pair<SizeType32, SizeType32>> stack;
        stack.push_back(std::make_pair(0, posIdsRange[0] - 1));
        for (auto i = 1; i < len + 1; i++)
        {
            auto cur = posIdsRange[i - 1];
            while (stack.size() > 0 && cur <= stack.back().second)
            {
                stack.pop_back();
            }
            TLLM_CHECK(stack.size() > 0 ? cur == stack.back().second + 1 : true);
            stack.push_back(std::make_pair(i, cur));
            for (auto prev : stack)
            {
                setBit(maskLocation.at(i, prev.first / 32), prev.first % 32);
            }
        }
    }
}

template <typename T>
void LookaheadDecodingLayer<T>::forwardSyncCPU(
    std::shared_ptr<BaseDecodingOutputs> const& outputParams, std::shared_ptr<BaseDecodingInputs> const& inputParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto inputs = std::dynamic_pointer_cast<LookaheadDecodingInputs>(inputParams);
    auto outputs = std::dynamic_pointer_cast<SpeculativeDecodingOutputs>(outputParams);
    auto const batchSize = inputs->localBatchSize;

    TensorPtr outputIds(wrap(outputs->outputIds));
    BufferRange<SizeType32 const> tokensPerStepRange(*mCpuAlgo->mTokensPerStep);
    BufferRange<SizeType32> numNewTokensRange(*mCpuAlgo->mNumNewTokens);
    BufferRange<SizeType32> numNewTokensCumSumRange(*mCpuAlgo->mNumNewTokensCumSum);
    BufferRange<SizeType32> batchSlotsRange(*mCpuAlgo->mBatchSlots);
    BufferRange<SizeType32> nextDraftLengthsRange(*mCpuAlgo->mNextDraftLengths);
    BufferRange<SizeType32> sequenceLengthsRange(*mCpuAlgo->mSequenceLengths);

    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi];
        LookaheadAlgorithm& theAlgo(mCpuAlgo->mAlgos[gbi]);

        SizeType32 const tokensPerStep = tokensPerStepRange[gbi];
        TensorPtr sampledTokens = ITensor::slice(mCpuAlgo->mTargetTokens, {gbi, 0}, tokensPerStep);

        if (tokensPerStep == 1)
        { // The first step in generation phase has no draft tokens.
            theAlgo.accept(sampledTokens);
            mBufferManager->copy(*sampledTokens, *ITensor::slice(mCpuAlgo->mOutputIds, {gbi, 0}, tokensPerStep));
            BufferLocation<SizeType32>(*mCpuAlgo->mPathsOffsets).at(gbi, 0) = 0;
            numNewTokensRange[gbi] = tokensPerStep;
            BufferLocation<SizeType32>(*mCpuAlgo->mNextDraftLengths).at(gbi) = 0;
        }
        else
        {
            theAlgo.update(                                  //
                ITensor::at(mCpuAlgo->mOutputIds, {gbi}),    //
                ITensor::at(mCpuAlgo->mPathsOffsets, {gbi}), //
                ITensor::at(mCpuAlgo->mNumNewTokens, {gbi}), //
                sampledTokens,                               //
                ITensor::at(mCpuAlgo->mEndIds, {gbi}));
        }

        auto maxNumNewTokens = mCpuAlgo->mOutputIds->getShape().d[1];
        mBufferManager->copy(*ITensor::at(mCpuAlgo->mOutputIds, {gbi}),
            *ITensor::slice(outputIds, {gbi, sequenceLengthsRange[gbi]}, maxNumNewTokens));

        sequenceLengthsRange[gbi] += numNewTokensRange[gbi];

        theAlgo.prepare(                                     //
            ITensor::at(mCpuAlgo->mNextDraftTokens, {gbi}),  //
            ITensor::at(mCpuAlgo->mNextDraftPosIds, {gbi}),  //
            ITensor::at(mCpuAlgo->mSamplingMask, {gbi}),     //
            ITensor::at(mCpuAlgo->mNextDraftLengths, {gbi}), //
            ITensor::at(mCpuAlgo->mSequenceLengths, {gbi}),  //
            ITensor::at(mCpuAlgo->mOutputIds, {gbi, numNewTokensRange[gbi] - 1}));

        posIdsToMask(                                   //
            ITensor::at(mCpuAlgo->mPackedMasks, {gbi}), //
            ITensor::slice(mCpuAlgo->mNextDraftPosIds, {gbi, 0}, nextDraftLengthsRange[gbi]));
    }

    numNewTokensCumSumRange[0] = 0;
    for (SizeType32 i = 0; i < numNewTokensRange.size(); i++)
    {
        numNewTokensCumSumRange[i + 1] = numNewTokensCumSumRange[i] + numNewTokensRange[i];
    }

    TLLM_CHECK(outputs->numNewTokens);

    mBufferManager->copy(*mCpuAlgo->mSequenceLengths,    //
        const_cast<void*>(outputs->sequenceLength.value().data), runtime::MemoryType::kGPU);
    mBufferManager->copy(*mCpuAlgo->mPathsOffsets,       //
        const_cast<void*>(outputs->pathsOffsets.data), runtime::MemoryType::kGPU);
    mBufferManager->copy(*mCpuAlgo->mNumNewTokens,       //
        const_cast<void*>(outputs->numNewTokens->data), runtime::MemoryType::kGPU);
    mBufferManager->copy(*mCpuAlgo->mNumNewTokensCumSum, //
        const_cast<void*>(outputs->numNewTokensCumSum.data), runtime::MemoryType::kGPU);
    mBufferManager->copy(*mCpuAlgo->mNextDraftTokens,    //
        const_cast<void*>(outputs->nextDraftTokens.data), runtime::MemoryType::kGPU);
    mBufferManager->copy(*mCpuAlgo->mNextDraftPosIds,    //
        const_cast<void*>(outputs->nextDraftPosIds.data), runtime::MemoryType::kGPU);
    mBufferManager->copy(*mCpuAlgo->mPackedMasks,        //
        const_cast<void*>(outputs->packedMasks.data), runtime::MemoryType::kGPU);
    mBufferManager->copy(*mCpuAlgo->mNextDraftLengths,   //
        const_cast<void*>(outputs->nextDraftLengths.data), runtime::MemoryType::kGPU);

    // TODO(liweim) do we need this?
    // mBufferManager->getStream().synchronize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
template class LookaheadDecodingLayer<float>;
template class LookaheadDecodingLayer<half>;

} // namespace tensorrt_llm::layers
