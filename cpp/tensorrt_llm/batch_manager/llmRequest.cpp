/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace tensorrt_llm::batch_manager
{

namespace
{
template <typename T>
std::optional<T> copyOptionalInput(std::optional<std::shared_ptr<T>> const& input)
{
    return input && *input ? std::make_optional(**input) : std::nullopt;
}
} // namespace

// Single, process-global storage for the steady-clock offset. Keeping it in this
// translation unit (rather than as an inline-static member reachable from every
// shared object) guarantees that libtensorrt_llm.so and the nanobind extension
// module observe the same value once either side calibrates it.
std::optional<std::chrono::steady_clock::duration>& globalSteadyClockOffset()
{
    static std::optional<std::chrono::steady_clock::duration> offset{std::nullopt};
    return offset;
}

template <typename TTensor>
runtime::SizeType32 GenericLlmRequest<TTensor>::getBeamWidthByIter(bool const forNextIteration)
{
    runtime::SizeType32 beamWidth = mSamplingConfig.getBeamWidth(); // For non-Variable-Beam-Width-Search
    auto const beamWidthArray = mSamplingConfig.getBeamWidthArray();
    if (beamWidthArray.has_value() && !beamWidthArray.value().empty())
    {
        auto const& requestBeamWidthArray = beamWidthArray.value();
        auto const iter = mDecodingIter + (forNextIteration ? 1 : 0);
        // Clamp `decodingIter` with the actual array length, so that decoding
        // longer than the array holds the last width instead of reading past
        // the end. kMaxBeamWidthArrayLength is only the capacity limit; the
        // user array is not padded up to it.
        int const index = std::max(std::min(iter, static_cast<int>(requestBeamWidthArray.size())) - 1, 0);
        beamWidth = requestBeamWidthArray[index];
    }
    return beamWidth;
}

template class GenericLlmRequest<runtime::ITensor::SharedPtr>;

void LlmRequest::createSerializedResult(std::vector<char>& serializedResult, bool& isFinal)
{
    auto result = createResult();
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
std::optional<executor::Result> LlmRequest::createResult()
{
    auto const streamingInProgress = mIsStreaming
        && (mState == LlmRequestState::kGENERATION_IN_PROGRESS || mState == LlmRequestState::kGENERATION_TO_COMPLETE);
    if (!(isFinished() || streamingInProgress))
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

    if ((isDisaggContextTransmissionState() || isDisaggContextCompleteState()) && isContextOnlyRequest())
    {
        auto const reqBeamWidth = mSamplingConfig.getBeamWidth();
        std::vector<TokenIdType> firstGenTokens;
        for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
        {
            firstGenTokens.push_back(getTokens().at(beam).back());
        }
        if (!hasDraftTokens())
        {
            result.contextPhaseParams = executor::ContextPhaseParams{std::move(firstGenTokens), mRequestId,
                mContextPhaseParams.value().releaseState(), std::nullopt, mContextPhaseParams.value().getCtxDpRank(),
                mContextPhaseParams.value().getDisaggInfoEndpoint()};
        }
        else
        {
            result.contextPhaseParams = executor::ContextPhaseParams{std::move(firstGenTokens), mRequestId,
                mContextPhaseParams.value().releaseState(), *getDraftTokens(),
                mContextPhaseParams.value().getCtxDpRank(), mContextPhaseParams.value().getDisaggInfoEndpoint()};
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

    // Update position of last sent response
    setMaxSentTokenLen(maxNbTokens);
    return result;
}

bool LlmRequest::checkTokenIdRange(SizeType32 vocabSize)
{
    TLLM_CHECK_WITH_INFO(!isContextFinished(), "not supported after prefill");

    if (mSamplingConfig.getBeamWidth() == 0)
    {
        return true;
    }

    // Before generation, all beams contain the same tokens
    auto const& tokens = getTokens(0);
    return std::all_of(
        tokens.begin(), tokens.end(), [&vocabSize](auto const& token) { return token >= 0 && token < vocabSize; });
}

void LlmRequest::validate(
    SizeType32 maxInputLen, SizeType32 maxSequenceLen, SizeType32 maxDraftLen, bool /*enableKVCacheReuse*/)
{
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
}

LlmRequest::LlmRequest(LlmRequest const& parent, RequestIdType requestId)
    : LlmRequest(requestId, parent.mMaxNewTokens, parent.mTokens.at(0), parent.mSamplingConfig, parent.mIsStreaming,
        copyOptionalInput(parent.mMultimodalHashes), copyOptionalInput(parent.mMultimodalPositions),
        copyOptionalInput(parent.mMultimodalLengths), copyOptionalInput(parent.mMultimodalUuids), parent.mLoraTaskId,
        parent.mLoraWeights, parent.mLoraConfig, parent.mKvCacheRetentionConfig,
        parent.mDraftTokens ? *parent.mDraftTokens : VecTokens{}, parent.mExcludeInputFromOutput,
        copyOptionalInput(parent.mEncoderTokens), parent.mPriority, parent.mEncoderInputLength,
        parent.mEncoderOutputLength, parent.mLlmRequestType, copyOptionalInput(parent.mInputTokenExtraIds),
        parent.mReturnPerfMetrics, parent.mContextPhaseParams,
        parent.mReturnPerfMetrics ? std::make_optional(parent.mPerfMetrics.timingMetrics.arrivalTime) : std::nullopt,
        parent.mAgentHierarchy, copyOptionalInput(parent.mMultimodalItemRunCuOffsets),
        copyOptionalInput(parent.mMultimodalRunPositions), copyOptionalInput(parent.mMultimodalRunLengths),
        parent.mCacheSalt)
{
    TLLM_CHECK_WITH_INFO(!parent.isChild(), "A child request cannot create its own child.");
    TLLM_CHECK_WITH_INFO(parent.mChildRequests.size() + 1 < static_cast<size_t>(parent.getNumSubRequests()),
        "Cannot create child requests more than the number of return sequences (%d)", parent.getNumSubRequests());
    TLLM_CHECK_WITH_INFO(parent.mDecodingIter == 0 && parent.mContextCurrentPositionTarget == 0
            && parent.mContextCurrentPositionDraft == 0 && parent.mPromptLen == parent.mOrigPromptLen
            && parent.mTokens.at(0).size() == static_cast<size_t>(parent.mOrigPromptLen)
            && (parent.mState == LlmRequestState::kCONTEXT_INIT || parent.mState == LlmRequestState::kENCODER_INIT
                || parent.mState == LlmRequestState::kDISAGG_GENERATION_INIT),
        "Children must be constructed before request execution.");
    mSequenceIndex = static_cast<SizeType32>(parent.mChildRequests.size() + 1);
    mParentRequestId = parent.mRequestId;
    mSequenceFinalVec = parent.mSequenceFinalVec;
    mReturnAllGeneratedTokens = parent.mReturnAllGeneratedTokens;
    mIsDummyRequest = parent.mIsDummyRequest;
    mUseDraftModel = parent.mUseDraftModel;

    // Tensor inputs are shared read-only; mutable vector inputs have independent storage.
    using RandomSeedType = tensorrt_llm::executor::RandomSeedType;
    mSamplingConfig.setSeed(
        parent.mSamplingConfig.getSeed().value_or(RandomSeedType{0}) + static_cast<RandomSeedType>(mSequenceIndex));
}

void LlmRequest::registerChildRequest(std::shared_ptr<LlmRequest> const& child)
{
    TLLM_CHECK_WITH_INFO(child && child->isChild() && child->mParentRequestId == mRequestId
            && child->mSequenceFinalVec == mSequenceFinalVec
            && child->mSequenceIndex == static_cast<SizeType32>(mChildRequests.size() + 1)
            && mChildRequests.size() + 1 < static_cast<size_t>(getNumSubRequests()) && !isChild(),
        "Child does not belong to the next available sequence of this parent.");
    TLLM_CHECK_WITH_INFO(child->mRequestId != mRequestId
            && std::none_of(mChildRequests.begin(), mChildRequests.end(),
                [&child](auto const& existing) { return existing->mRequestId == child->mRequestId; }),
        "Child request ID must be unique within its sequence group.");
    mChildRequests.push_back(child);
    if (!mSamplingConfig.getSeed().has_value())
    {
        mSamplingConfig.setSeed(executor::RandomSeedType{0});
    }
}

std::shared_ptr<LlmRequest> LlmRequest::createChildRequest(RequestIdType requestId)
{
    auto child = std::make_shared<LlmRequest>(*this, requestId);
    registerChildRequest(child);
    return child;
}

void LlmRequest::removeLoraTensors()
{
    mLoraWeights.reset();
    mLoraConfig.reset();
}

} // namespace tensorrt_llm::batch_manager
