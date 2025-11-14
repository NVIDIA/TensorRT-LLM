/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"

namespace tensorrt_llm::batch_manager
{

template <typename TTensor, typename TStream>
runtime::SizeType32 GenericLlmRequest<TTensor, TStream>::getBeamWidthByIter(bool const forNextIteration)
{
    runtime::SizeType32 beamWidth = mSamplingConfig.beamWidth; // For non-Variable-Beam-Width-Search
    auto const& beamWidthArray = mSamplingConfig.beamWidthArray;
    if (beamWidthArray.has_value())
    {
        auto const iter = mDecodingIter + (forNextIteration ? 1 : 0);
        // Clamped `decodingIter` into [0,kMaxBeamWidthArrayLength-1] as index
        int const index
            = std::max(std::min(iter, static_cast<int>(tensorrt_llm::kernels::kMaxBeamWidthArrayLength)) - 1, 0);
        beamWidth = beamWidthArray.value()[0][index];
    }
    return beamWidth;
}

template class GenericLlmRequest<runtime::ITensor::SharedPtr>;

std::optional<executor::Response> LlmRequest::createResponse(bool useFastLogits, int32_t mpiWorldRank)
{
    auto requestId = isChild() ? mParentRequestId : mRequestId;
    auto result = createResult(useFastLogits, mpiWorldRank);
    if (result.has_value())
    {
        return executor::Response(requestId, result.value(), mClientId);
    }
    return std::nullopt;
}

void LlmRequest::createSerializedResult(
    std::vector<char>& serializedResult, bool& isFinal, bool useFastLogits, int32_t mpiWorldRank)
{
    auto result = createResult(useFastLogits, mpiWorldRank);
    if (result.has_value())
    {
        std::ostringstream oStream;
        executor::serialize_utils::serialize(result.value(), oStream);
        auto str = oStream.str();
        serializedResult.resize(str.size());
        std::copy(str.begin(), str.end(), serializedResult.begin());
        isFinal = result.value().isFinal;
    }
}

/// Note that there is some dependency on the order of operations in this method. Modify with care!
std::optional<executor::Result> LlmRequest::createResult(bool useFastLogits, int32_t mpiWorldRank)
{
    if (!(isFinished() || (mIsStreaming && mState == LlmRequestState::kGENERATION_IN_PROGRESS)))
    {
        return std::nullopt;
    }

    TLLM_LOG_DEBUG("Creating response for request %lu", mRequestId);

    executor::Result result;
    result.sequenceIndex = mSequenceIndex;

    result.isSequenceFinal = isFinished();
    mSequenceFinalVec->at(mSequenceIndex) = result.isSequenceFinal;

    result.isFinal = std::all_of(
        mSequenceFinalVec->begin(), mSequenceFinalVec->end(), [](bool isSequenceFinal) { return isSequenceFinal; });

    auto const maxNbTokens = getMaxBeamNumTokens();

    if (isDisaggContextTransmissionState() && isContextOnlyRequest())
    {
        auto const reqBeamWidth = mSamplingConfig.beamWidth;
        std::vector<TokenIdType> firstGenTokens;
        for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
        {
            firstGenTokens.push_back(getTokens().at(beam).back());
        }
        if (!hasDraftTokens())
        {
            result.contextPhaseParams = executor::ContextPhaseParams{
                std::move(firstGenTokens), mRequestId, mContextPhaseParams.value().releaseState(), std::nullopt};
        }
        else
        {
            result.contextPhaseParams = executor::ContextPhaseParams{
                std::move(firstGenTokens), mRequestId, mContextPhaseParams.value().releaseState(), *getDraftTokens()};
        }
    }

    auto const calculateNbTokensOut = [this](SizeType32 maxNbTokens)
    {
        if (!mIsStreaming)
        {
            return maxNbTokens - (mExcludeInputFromOutput ? getOrigPromptLen() : 0);
        }
        return mReturnAllGeneratedTokens ? maxNbTokens - getOrigPromptLen() : maxNbTokens - getMaxSentTokenLen();
    };

    auto const maxNbTokensOut = calculateNbTokensOut(maxNbTokens);

    auto const nbBeams = mSamplingConfig.getNumReturnBeams();

    result.outputTokenIds.resize(nbBeams);

    auto const startTokenPos = maxNbTokens - maxNbTokensOut;

    auto const shouldSendResponse = isFinished() || (mIsStreaming && maxNbTokens > getMaxSentTokenLen());

    if (!shouldSendResponse)
    {
        return std::nullopt;
    }

    for (SizeType32 beam = 0; beam < nbBeams; ++beam)
    {
        auto const& tokens = getTokens(beam);
        auto const nbTokensOut = calculateNbTokensOut(tokens.size());

        if (nbTokensOut > 0)
        {
            auto const first = tokens.data() + startTokenPos;
            result.outputTokenIds.at(beam).assign(first, first + nbTokensOut);
        }
    }

    auto sliceBeams = [&nbBeams](auto beams)
    { return std::vector<typename decltype(beams)::value_type>(beams.begin(), beams.begin() + nbBeams); };

    if (returnLogProbs())
    {
        result.cumLogProbs = sliceBeams(getCumLogProbs());
        result.logProbs = sliceBeams(getLogProbs());
    }

    if (getReturnContextLogits())
    {
        result.contextLogits = executor::detail::ofITensor(getContextLogitsHost());
    }

    if (getReturnGenerationLogits())
    {
        bool hasDraftTokens = mDraftTokens && !mDraftTokens->empty();
        if (isStreaming() && !hasDraftTokens)
        {
            auto startGenTokenPos = startTokenPos - getOrigPromptLen();
            TensorPtr generationLogitsHostCurrentStep
                = runtime::ITensor::slice(getGenerationLogitsHost(), startGenTokenPos, maxNbTokensOut);
            result.generationLogits = executor::detail::ofITensor(generationLogitsHostCurrentStep);
        }
        else if (useFastLogits)
        {
            result.specDecFastLogitsInfo = executor::SpeculativeDecodingFastLogitsInfo{mRequestId, mpiWorldRank};
        }
        else
        {
            result.generationLogits
                = executor::detail::ofITensor(runtime::ITensor::slice(getGenerationLogitsHost(), 0, nbBeams));
        }
    }

    if (getReturnEncoderOutput())
    {
        result.encoderOutput = executor::detail::ofITensor(getEncoderOutputHost());
    }

    if (getReturnPerfMetrics())
    {
        mPerfMetrics.kvCacheMetrics.kvCacheHitRate = getKVCacheHitRatePerRequest();

        auto& specDecMetrics = mPerfMetrics.speculativeDecoding;
        if (specDecMetrics.totalDraftTokens != 0)
        {
            specDecMetrics.acceptanceRate
                = static_cast<float>(specDecMetrics.totalAcceptedDraftTokens) / specDecMetrics.totalDraftTokens;
        }

        result.requestPerfMetrics = mPerfMetrics;
    }

    result.finishReasons = sliceBeams(mFinishReasons);
    result.decodingIter = mDecodingIter;
    result.avgDecodedTokensPerIter = getAvgDecodedTokensPerIter();

    if (hasAdditionalOutputs())
    {
        std::string prefix = "context_";
        for (auto const& outputTensorMap : {mAdditionalContextOutputTensors, mAdditionalGenerationOutputTensors})
        {
            for (auto const& outputTensor : outputTensorMap)
            {
                TLLM_LOG_DEBUG("Adding tensor %s with shape %s to result.", outputTensor.first.c_str(),
                    runtime::ITensor::toString(outputTensor.second->getShape()).c_str());
                result.additionalOutputs.emplace_back(
                    prefix + outputTensor.first, executor::detail::ofITensor(outputTensor.second));
            }
            prefix = "generation_";
        }
    }

    // Update position of last sent response
    setMaxSentTokenLen(maxNbTokens);
    return result;
}

bool LlmRequest::checkTokenIdRange(SizeType32 vocabSize)
{
    TLLM_CHECK_WITH_INFO(!isContextFinished(), "not supported after prefill");

    if (mSamplingConfig.beamWidth == 0)
    {
        return true;
    }

    // Before generation, all beams contain the same tokens
    auto const& tokens = getTokens(0);
    return std::all_of(
        tokens.begin(), tokens.end(), [&vocabSize](auto const& token) { return token >= 0 && token < vocabSize; });
}

void LlmRequest::validate(SizeType32 maxInputLen, SizeType32 maxSequenceLen, SizeType32 maxDraftLen,
    SizeType32 vocabSizePadded, std::optional<SizeType32> maxEncoderInputLen, bool enableKVCacheReuse)
{
    if (mEndId.has_value())
    {
        TLLM_CHECK_WITH_INFO(*mEndId >= -1 && *mEndId < vocabSizePadded,
            "EndId (%d) is not within acceptable range [-1, %d).", *mEndId, vocabSizePadded);
    }
    if (getEncoderInputFeatures()
        && getEncoderInputFeatures()->getShape().nbDims < 4) // skip encoder shape validation for image inputs
    {
        TLLM_CHECK_WITH_INFO(!(maxEncoderInputLen.has_value() && getEncoderInputLen() > maxEncoderInputLen.value()),
            "Encoder length (%d) exceeds maximum encoder input length (%d).", getEncoderInputLen(),
            maxEncoderInputLen.value());
    }

    if (mPromptLen > maxInputLen)
    {
        TLLM_THROW(
            "Prompt length (%d) exceeds maximum input length (%d). Set log level to info and check "
            "TRTGptModel logs for how maximum input length is set",
            mPromptLen, maxInputLen);
    }

    // Maximum number of draft tokens per request we pass to the engine for single runtime iteration.
    // It depends on the speculative decoding mode.
    auto draftLenPerEngineStep = maxDraftLen;
    auto const& draftTokens = getDraftTokens();
    if (draftTokens && !draftTokens->empty())
    {
        auto const inputDraftTokensLen = static_cast<SizeType32>(draftTokens->size());
        if (inputDraftTokensLen > maxDraftLen)
        {
            TLLM_THROW(
                "Draft tokens length (%d) exceeds maximum draft tokens length (%d).", inputDraftTokensLen, maxDraftLen);
        }
        draftLenPerEngineStep = inputDraftTokensLen;

        if (mPromptLen + draftLenPerEngineStep > maxInputLen)
        {
            auto const newDraftLenPerEngineStep = maxInputLen - mPromptLen;
            TLLM_LOG_WARNING(
                "Prompt length + number of draft tokens (%d + %d) exceeds maximum input length (%d)."
                "Number of draft tokens is changed to (%d)",
                mPromptLen, draftLenPerEngineStep, maxInputLen, newDraftLenPerEngineStep);
            draftLenPerEngineStep = newDraftLenPerEngineStep;
            mDraftTokens->resize(draftLenPerEngineStep);
        }
    }

    if (mPromptLen + mMaxNewTokens + draftLenPerEngineStep > maxSequenceLen)
    {
        auto const maxNewTokens = maxSequenceLen - mPromptLen - draftLenPerEngineStep;
        TLLM_LOG_WARNING(
            "Prompt length + number of requested output tokens + draft tokens per step (%d + %d + %d) exceeds "
            "maximum sequence length (%d). "
            "Number of requested output tokens is changed to (%d).",
            mPromptLen, mMaxNewTokens, draftLenPerEngineStep, maxSequenceLen, maxNewTokens);
        mMaxNewTokens = maxNewTokens;
    }

    TLLM_CHECK_WITH_INFO(mSamplingConfig.validate(), "Incorrect sampling config");

    // validate extra ids when enabling kv cache reuse with prompt table
    if (enableKVCacheReuse && mPromptEmbeddingTable.has_value() && mPromptVocabSize.has_value())
    {
        TLLM_CHECK_WITH_INFO(mInputTokenExtraIds.has_value() && mInputTokenExtraIds.value(),
            "Input token extra ids must be provided when enabling kv cache reuse with prompt table");
        TLLM_CHECK_WITH_INFO(mInputTokenExtraIds.value()->size() == static_cast<size_t>(mOrigPromptLen),
            "inputTokenExtraIds vector size (%lu) must be the same as input token vector size (%lu).",
            mInputTokenExtraIds.value()->size(), static_cast<size_t>(mOrigPromptLen));
    }
}

std::shared_ptr<LlmRequest> LlmRequest::createChildRequest(RequestIdType requestId)
{
    TLLM_CHECK_WITH_INFO(!isChild(), "A child request cannot create its own child.");
    TLLM_CHECK_WITH_INFO(mChildRequests.size() + 1 < static_cast<size_t>(getNumSubRequests()),
        "Cannot create child requests more than the number of return sequences (%d)", getNumSubRequests());
    auto childReq = std::make_shared<LlmRequest>(*this);
    childReq->mRequestId = requestId;
    childReq->mSequenceIndex = mChildRequests.size() + 1;
    childReq->mParentRequestId = this->mRequestId;
    childReq->mSequenceFinalVec = this->mSequenceFinalVec;
    childReq->mSeqSlot.reset();

    // To ensure different randomness across children, assign a unique random seed to each child
    // by adding its sequence index to the base seed. If no seed is provided, the parent's seed defaults to 0.
    using RandomSeedType = tensorrt_llm::executor::RandomSeedType;
    if (childReq->mSamplingConfig.randomSeed.has_value())
    {
        childReq->mSamplingConfig.randomSeed->at(0) += static_cast<RandomSeedType>(childReq->mSequenceIndex);
    }
    else
    {
        RandomSeedType defaultSeed{0};
        mSamplingConfig.randomSeed = std::vector<RandomSeedType>(1, defaultSeed);
        childReq->mSamplingConfig.randomSeed
            = std::vector<RandomSeedType>(1, defaultSeed + static_cast<RandomSeedType>(childReq->mSequenceIndex));
    }

    mChildRequests.push_back(childReq);
    return childReq;
}

void LlmRequest::movePromptEmbeddingTableToGpu(runtime::BufferManager const& manager)
{
    if (!mPromptEmbeddingTable.has_value()
        || mPromptEmbeddingTable.value()->getMemoryType() == runtime::MemoryType::kGPU)
    {
        return;
    }

    TensorPtr gpuPromptEmbeddingTable = manager.copyFrom(*mPromptEmbeddingTable.value(), runtime::MemoryType::kGPU);
    mPromptEmbeddingTable = gpuPromptEmbeddingTable;
}

void LlmRequest::moveLoraWeightsToGpu(runtime::BufferManager const& manager)
{
    if (!mLoraWeights.has_value() || mLoraWeights.value()->getMemoryType() == runtime::MemoryType::kGPU)
    {
        return;
    }
    // TODO for tp / pp models we only need to move the bit that belong on the local device
    TensorPtr gpuLoraWeights = manager.copyFrom(*mLoraWeights.value(), runtime::MemoryType::kGPU);
    mLoraWeights = gpuLoraWeights;
}

void LlmRequest::removeLoraTensors()
{
    mLoraWeights.reset();
    mLoraConfig.reset();
}

} // namespace tensorrt_llm::batch_manager
