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

#include "lookaheadDecodingLayer.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"
#include <cstddef>
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
    auto const maxBatchSize = decoderDomain.getBatchSize();
    auto const beamWidth = decoderDomain.getBeamWidth();
    auto const decodingTokens = decoderDomain.getMaxDecodingTokens();
    auto lookaheadModule
        = std::dynamic_pointer_cast<LookaheadModule const>(decoderDomain.getSpeculativeDecodingModule());
    auto const [maxW, maxN, maxG] = lookaheadModule->getExecutionConfig().get();
    SizeType32 maxTokensPerStep, maxNumNewTokens, maxDraftLen, maxAcceptedDraftLen;
    std::tie(maxTokensPerStep, maxNumNewTokens, maxDraftLen, maxAcceptedDraftLen)
        = executor::LookaheadDecodingConfig(maxW, maxN, maxG).calculateSpeculativeResource();
    TLLM_CHECK_WITH_INFO(beamWidth == 1, "Lookahead requires beam width = 1");
    TLLM_CHECK_WITH_INFO(maxTokensPerStep == decodingTokens, "%d != %d", maxTokensPerStep, decodingTokens);

    for (SizeType32 id = 0; id < maxBatchSize; id++)
    {
        mAlgos.emplace_back(maxW, maxN, maxG, id);
    }

    mPrompts.reserve(maxBatchSize);
    for (auto bi = 0; bi < maxBatchSize; bi++)
    {
        mPrompts.emplace_back(BufferManager::cpu(ITensor::makeShape({0}), nvinfer1::DataType::kINT32));
    }

    auto const maxBatchShape1D = ITensor::makeShape({maxBatchSize});
    mBatchSlots = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mTargetTokens
        = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxTokensPerStep}), nvinfer1::DataType::kINT32);
    mTokensPerStep = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mEndIds = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);

    mOutputIds = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxNumNewTokens}), nvinfer1::DataType::kINT32);
    mNewTokens = BufferManager::cpu(
        ITensor::makeShape({maxTokensPerStep, maxBatchSize, beamWidth}), nvinfer1::DataType::kINT32);
    mPathsOffsets
        = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxAcceptedDraftLen}), nvinfer1::DataType::kINT32);
    mPathsOffsetsBatch
        = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxAcceptedDraftLen}), nvinfer1::DataType::kINT32);
    mNumNewTokens = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mNumNewTokensCumSum = BufferManager::cpu(ITensor::makeShape({maxBatchSize + 1}), nvinfer1::DataType::kINT32);
    mNextDraftTokens = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxDraftLen}), nvinfer1::DataType::kINT32);
    mNextDraftPosIds = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxDraftLen}), nvinfer1::DataType::kINT32);
    mGenerationLengths = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mPositionOffsets
        = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxTokensPerStep}), nvinfer1::DataType::kINT32);
    mPositionIds = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxTokensPerStep}), nvinfer1::DataType::kINT32);
    mAttentionMask
        = BufferManager::cpu(ITensor::makeShape({maxTokensPerStep, maxTokensPerStep}), nvinfer1::DataType::kBOOL);
    mPackedMask = BufferManager::cpu(ITensor::makeShape({maxBatchSize, maxTokensPerStep,
                                         static_cast<ITensor::DimType64>(divUp(maxTokensPerStep, 32))}),
        nvinfer1::DataType::kINT32);
    mNextDraftLengths = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mSequenceLengths = BufferManager::cpu(maxBatchShape1D, nvinfer1::DataType::kINT32);
}

template <typename T>
LookaheadDecodingLayer<T>::LookaheadDecodingLayer(
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mCpuAlgo(std::make_optional<CpuAlgorithmResources>(decoderDomain))
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto lookaheadModule
        = std::dynamic_pointer_cast<LookaheadModule const>(decoderDomain.getSpeculativeDecodingModule());

    auto const maxBatchSize = mDecoderDomain.getBatchSize();
    auto const maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    auto const vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    auto const maxTopK = 1;
    auto const maxBatchShape1D = ITensor::makeShape({maxBatchSize});
    auto const maxBatchShape2D = ITensor::makeShape({maxBatchSize, maxTokensPerStep});

    mWorkspaceSize = getTopKWorkspaceSize<T>(maxBatchSize, maxTokensPerStep, maxTopK, vocabSizePadded);
    mTargetTokensDevice = mBufferManager->gpu(maxBatchShape2D, nvinfer1::DataType::kINT32);
    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({maxBatchSize, sizeof(curandState_t)}), nvinfer1::DataType::kINT8);

    mSetupWorkspaceSize = DecodingLayerWorkspace::calculateRequiredWorkspaceSize(
        std::make_pair(maxBatchShape1D, nvinfer1::DataType::kINT64));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void LookaheadDecodingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(LookaheadDecodingLayer_setup);

    auto setupParams = std::dynamic_pointer_cast<LookaheadSetupParams>(baseSetupParams);

    if (mCpuAlgo)
    {
        auto& algoConfigs = setupParams->algoConfigs;
        TLLM_CHECK_WITH_INFO(algoConfigs.size() == 1 || algoConfigs.size() == static_cast<size_t>(batchSize),
            "Lookahead runtime configuration size should be either 1 or batchSize");

        for (auto bi = 0; bi < batchSize; bi++)
        {
            PRINT_SHAPE(setupParams->prompt[bi]);
            PRINT_TOKENS(setupParams->prompt[bi]);
            mCpuAlgo->mPrompts[bi]->reshape(setupParams->prompt[bi]->getShape());
            mBufferManager->copy(*setupParams->prompt[bi], *mCpuAlgo->mPrompts[bi]);
        }

        mBufferManager->getStream().synchronize(); // sync prompt gpu to cpu

        auto const batchSlotsRange = BufferRange<SizeType32 const>(*batchSlots);
        for (SizeType32 bi = 0; bi < batchSize; bi++)
        {
            auto const gbi = batchSlotsRange[bi];
            SizeType32 bi1orN = (algoConfigs.size() == 1) ? 0 : bi;
            TLLM_LOG_DEBUG("CPU ALGO [ %d ] setup prompt %s", gbi, D(mCpuAlgo->mPrompts[bi]).values().c_str());
            auto [w, n, g] = algoConfigs[bi1orN].get();
            SizeType32 runtimeTokensPerStep = 0;
            std::tie(runtimeTokensPerStep, std::ignore, std::ignore, std::ignore)
                = executor::LookaheadDecodingConfig(w, n, g).calculateSpeculativeResource();
            TLLM_CHECK_WITH_INFO(runtimeTokensPerStep <= mDecoderDomain.getMaxDecodingTokens(),
                "runtime w(%d) n(%d) g(%d) exceeds maxTokensPerStep(%d)", w, n, g,
                mDecoderDomain.getMaxDecodingTokens());
            PRINT_VALUES(mCpuAlgo->mPrompts[bi]);
            auto seed = DefaultDecodingParams::getSeed();
            if (setupParams->randomSeed)
            {
                auto& seeds = setupParams->randomSeed.value();
                seed = seeds.size() == 1 ? seeds[0] : seeds[bi];
            }
            mCpuAlgo->mAlgos[gbi].setup(mCpuAlgo->mPrompts[bi], w, n, g, seed);
        }

        for (runtime::SizeType32 bi = 0; bi < batchSize; bi++)
        {
            SizeType32 gbi = batchSlotsRange[bi];
            (BufferRange<SizeType32>(*mCpuAlgo->mGenerationLengths))[gbi] = 1;
            (BufferRange<SizeType32>(*mCpuAlgo->mNextDraftLengths))[gbi] = 0;
            BufferLocation<SizeType32>(*mCpuAlgo->mPositionOffsets).at(gbi, 0) = 0;
            BufferRange<SizeType32> packedMaskRange(*ITensor::at(mCpuAlgo->mPackedMask, {gbi}));
            for (auto& mask : packedMaskRange)
            {
                mask = 0;
            }
            packedMaskRange[0] = 1;

            PRINT_SHAPE(mCpuAlgo->mGenerationLengths);
            PRINT_SHAPE(setupParams->generationLengths);
            PRINT_SHAPE(mCpuAlgo->mPositionOffsets);
            PRINT_SHAPE(setupParams->positionOffsets);
            PRINT_SHAPE(mCpuAlgo->mPackedMask);
            PRINT_SHAPE(setupParams->attentionPackedMasks);
            mBufferManager->copy(
                *ITensor::at(mCpuAlgo->mGenerationLengths, {gbi}), *ITensor::at(setupParams->generationLengths, {gbi}));
            mBufferManager->copy(
                *ITensor::at(mCpuAlgo->mPositionOffsets, {gbi}), *ITensor::at(setupParams->positionOffsets, {gbi}));
            mBufferManager->copy(
                *ITensor::at(mCpuAlgo->mPackedMask, {gbi}), *ITensor::at(setupParams->attentionPackedMasks, {gbi}));
        }

        mBufferManager->getStream().synchronize(); // sync outputs cpu to gpu
    }

    workspace->initializeDeviceCurandStates(
        setupParams->randomSeed, batchSize, workspace->getDeviceBatchSlots(), mCurandStatesDevice);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void LookaheadDecodingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputParams,
    std::shared_ptr<BaseDecodingInputs> const& inputParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(LookaheadDecodingLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<LookaheadDecodingInputs>(inputParams);
    auto outputs = std::dynamic_pointer_cast<LookaheadDecodingOutputs>(outputParams);
    auto batchSize = inputs->localBatchSize;

    TLLM_CHECK_WITH_INFO(inputs->batchSlots, "Batch slots must be provided for LookaheadDecoding");
    TLLM_CHECK_WITH_INFO(inputs->curTokensPerStep, "curTokensPerStep must be provided for LookaheadDecoding");
    TLLM_CHECK_WITH_INFO(outputs->sequenceLength, "sequenceLength must be provided for LookaheadDecoding");
    TLLM_CHECK_WITH_INFO(inputs->logits, "logits must be provided for LookaheadDecoding");
    TLLM_CHECK_WITH_INFO(inputs->localBatchSize > 0, "batchSize must be");

    TopKSamplingKernelParams<T> params;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.batchSize = batchSize;
    params.maxTopK = 1;
    params.returnAllSelectedTokens = true;
    params.maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    params.maxSeqLen = mDecoderDomain.getMaxDecodingTokens();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.logProbs = bufferCastOrNull<T>(inputs->logits);
    params.outputIds = bufferCast<TokenIdType>(*mTargetTokensDevice);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    params.tokensPerStep = bufferCast<SizeType32>(*inputs->curTokensPerStep.value());

    TLLM_LOG_DEBUG(
        "invokeBatchTopKSampling: maxBatchSize=%d, batchSize=%d, maxTopK=%d, maxTokensPerStep=%d, maxSeqLen=%d, "
        "vocabSizePadded=%d",
        params.maxBatchSize, params.batchSize, params.maxTopK, params.maxTokensPerStep, params.maxSeqLen,
        params.vocabSizePadded);

    // Sample multiple tokens per request and store them to separate to be accepted/rejected later
    // Sequence length is not modified, endIds is not checked, outputLogProbs are not supported.
    // Finished state is not set.
    invokeBatchTopKSampling(params, getStream());

    if (mCpuAlgo)
    {
        forwardSyncCPU(outputs, inputs);
        mGlobalSteps += 1;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t LookaheadDecodingLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mWorkspaceSize, mSetupWorkspaceSize);
}

inline void initAttentionMask(TensorPtr const& mask, std::shared_ptr<runtime::BufferManager>& bufferManager)
{
    bufferManager->setZero(*mask);
    BufferLocation<bool> maskLocation(*mask);
    auto maskShape = mask->getShape();
    for (SizeType32 i = 0; i < maskShape.d[0]; i++)
    {
        maskLocation.at(i, 0) = true;
    }
}

inline void convertBoolToInt32(TensorPtr const& dst, TensorConstPtr const& src)
{
    auto dstShape = dst->getShape();
    auto srcShape = src->getShape();
    TLLM_CHECK(dstShape.d[0] == srcShape.d[0]);
    TLLM_CHECK(dstShape.d[1] * 32 >= srcShape.d[1]);
    BufferLocation<SizeType32> dstLocation(*dst);
    BufferLocation<bool const> srcLocation(*src);

    auto setBit = [](SizeType32& x, SizeType32 idx, bool value) { x |= (value << idx); };
    for (auto i = 0; i < srcShape.d[0]; i++)
    {
        for (auto j = 0; j < srcShape.d[1]; j++)
        {
            setBit(dstLocation.at(i, j / 32), j % 32, srcLocation.at(i, j));
        }
    }
}

template <typename T>
void LookaheadDecodingLayer<T>::forwardSyncCPU(
    std::shared_ptr<LookaheadDecodingOutputs> const& outputs, std::shared_ptr<LookaheadDecodingInputs> const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(LookaheadDecodingLayer_forwardSyncCPU);

    mCpuAlgo->mBatchSlots->reshape(inputs->batchSlots->getShape());
    mBufferManager->copy(*inputs->batchSlots, *mCpuAlgo->mBatchSlots);
    mBufferManager->copy(*inputs->curTokensPerStep.value(), *mCpuAlgo->mTokensPerStep);
    mBufferManager->copy(*inputs->endIds, *mCpuAlgo->mEndIds);
    mBufferManager->copy(*outputs->sequenceLength.value(), *mCpuAlgo->mSequenceLengths);

    mBufferManager->copy(*mTargetTokensDevice, *mCpuAlgo->mTargetTokens);

    if (outputs->prevDraftLengths)
    {
        mBufferManager->copy(*mCpuAlgo->mNextDraftLengths, *outputs->prevDraftLengths);
    }

    mBufferManager->getStream().synchronize();

    auto const batchSize = inputs->localBatchSize;

    BufferRange<SizeType32 const> tokensPerStepRange(*mCpuAlgo->mTokensPerStep);
    BufferRange<TokenIdType> endIdsRange(*mCpuAlgo->mEndIds);
    BufferLocation<TokenIdType> newTokensLocation(*mCpuAlgo->mNewTokens);
    BufferRange<SizeType32> numNewTokensRange(*mCpuAlgo->mNumNewTokens);
    BufferRange<SizeType32> numNewTokensCumSumRange(*mCpuAlgo->mNumNewTokensCumSum);
    BufferRange<SizeType32> batchSlotsRange(*mCpuAlgo->mBatchSlots);
    BufferRange<SizeType32> generationLengthsRange(*mCpuAlgo->mGenerationLengths);
    BufferRange<SizeType32> nextDraftLengthsRange(*mCpuAlgo->mNextDraftLengths);
    BufferRange<SizeType32> sequenceLengthsRange(*mCpuAlgo->mSequenceLengths);
    BufferLocation<SizeType32> pathsOffsetLocation(*mCpuAlgo->mPathsOffsets);
    BufferLocation<SizeType32> pathsOffsetBatchLocation(*mCpuAlgo->mPathsOffsetsBatch);
    BufferLocation<TokenIdType> outputIdsLocation(*mCpuAlgo->mOutputIds);

    mBufferManager->setZero(*mCpuAlgo->mPathsOffsets);
    mBufferManager->setZero(*mCpuAlgo->mNumNewTokens);
    mBufferManager->setZero(*mCpuAlgo->mNumNewTokensCumSum);
    mBufferManager->setZero(*mCpuAlgo->mPackedMask);

    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi];
        LookaheadAlgorithm& theAlgo(mCpuAlgo->mAlgos[gbi]);

        SizeType32 const tokensPerStep = generationLengthsRange[gbi];
        TensorPtr sampledTokens = ITensor::slice(mCpuAlgo->mTargetTokens, {gbi, 0}, tokensPerStep);

        if (tokensPerStep == 1)
        {
            // The first step in generation phase has no draft tokens.
            theAlgo.accept(sampledTokens);
            mBufferManager->copy(*sampledTokens, *ITensor::slice(mCpuAlgo->mOutputIds, {gbi, 0}, tokensPerStep));
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
            *ITensor::slice(outputs->outputIds, {gbi, 0, sequenceLengthsRange[gbi]}, maxNumNewTokens));

        sequenceLengthsRange[gbi] += numNewTokensRange[gbi];

        initAttentionMask(mCpuAlgo->mAttentionMask, mBufferManager);

        theAlgo.prepare(                                     //
            ITensor::at(mCpuAlgo->mNextDraftTokens, {gbi}),  //
            ITensor::at(mCpuAlgo->mNextDraftPosIds, {gbi}),  //
            ITensor::at(mCpuAlgo->mNextDraftLengths, {gbi}), //
            mCpuAlgo->mAttentionMask, 1,                     //
            ITensor::at(mCpuAlgo->mSequenceLengths, {gbi}),  //
            ITensor::at(mCpuAlgo->mOutputIds, {gbi, numNewTokensRange[gbi] - 1}));

        convertBoolToInt32(ITensor::at(mCpuAlgo->mPackedMask, {gbi}), mCpuAlgo->mAttentionMask);

        BufferLocation<SizeType32> posIdsLocation(*ITensor::at(mCpuAlgo->mPositionIds, {gbi}));
        for (auto& posid : posIdsLocation)
        {
            posid = sequenceLengthsRange[gbi] - 1;
        }
        mBufferManager->copy(*ITensor::slice(mCpuAlgo->mNextDraftPosIds, {gbi, 0}, nextDraftLengthsRange[gbi]),
            *ITensor::slice(mCpuAlgo->mPositionIds, {gbi, 1}, nextDraftLengthsRange[gbi]));

        BufferRange<SizeType32> offsetRange(*ITensor::at(mCpuAlgo->mPositionOffsets, {gbi}));
        for (size_t i = 0; i < posIdsLocation.size(); i++)
        {
            offsetRange[i] = posIdsLocation[i] - posIdsLocation[0];
        }

        TensorPtr accepted = ITensor::slice(mCpuAlgo->mOutputIds, {gbi, 0}, numNewTokensRange[gbi]);
        TensorPtr draft = ITensor::slice(mCpuAlgo->mNextDraftTokens, {gbi, 0}, nextDraftLengthsRange[gbi]);
        TLLM_LOG_DEBUG("CPU ALGO [ %d ] forward, %s", gbi, D(sampledTokens).values().c_str());
        TLLM_LOG_DEBUG("[%d][%d] CPU ALGO [ %d ] forward, %s, %s", mGlobalSteps, batchSize, gbi,
            D(accepted).values().c_str(), D(draft).values().c_str());
    }

    size_t pi = 0;
    numNewTokensCumSumRange[0] = 0;
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi];
        SizeType32 acceptedDraftLen = numNewTokensRange[gbi] <= 1 ? 0 : (numNewTokensRange[gbi] - 1);
        numNewTokensCumSumRange[bi + 1] = numNewTokensCumSumRange[bi] + acceptedDraftLen;
        for (SizeType32 tj = 0; tj < acceptedDraftLen; tj++)
        {
            pathsOffsetBatchLocation[pi++] = pathsOffsetLocation.at(gbi, tj);
        }
    }

    while (pi < pathsOffsetBatchLocation.size())
    {
        pathsOffsetBatchLocation[pi++] = 0;
    }

    TLLM_CHECK(outputs->numNewTokens);

    mBufferManager->copy(*mCpuAlgo->mSequenceLengths, *outputs->sequenceLength.value());
    mBufferManager->copy(*mCpuAlgo->mNewTokens, *outputs->newTokens);

    mBufferManager->copy(*mCpuAlgo->mNumNewTokens, *outputs->numNewTokens.value());
    mBufferManager->copy(*mCpuAlgo->mPathsOffsetsBatch, *outputs->pathsOffsets);
    mBufferManager->copy(*mCpuAlgo->mNumNewTokensCumSum, *outputs->numNewTokensCumSum); //
    mBufferManager->copy(*mCpuAlgo->mNextDraftTokens, *outputs->nextDraftTokens);

    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi];
        // nextDraftLengthsRange[gbi] = mDecoderDomain.getMaxDecodingTokens() - 1;
        generationLengthsRange[gbi] = nextDraftLengthsRange[gbi] + 1;
    }

    if (outputs->nextDraftLengths)
    {
        mBufferManager->copy(*mCpuAlgo->mNextDraftLengths, *outputs->nextDraftLengths);
    }

    mBufferManager->copy(*mCpuAlgo->mPackedMask, *outputs->packedMasks);
    mBufferManager->copy(*mCpuAlgo->mGenerationLengths, *outputs->generationLengths);
    mBufferManager->copy(*mCpuAlgo->mPositionOffsets, *outputs->positionOffsets);
    mBufferManager->copy(*mCpuAlgo->mPositionIds, *outputs->positionIds);

    mBufferManager->getStream().synchronize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
template class LookaheadDecodingLayer<float>;
template class LookaheadDecodingLayer<half>;

} // namespace tensorrt_llm::layers
