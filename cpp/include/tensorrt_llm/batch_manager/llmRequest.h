/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager
{

/**
 * @brief The state of the request.
 *
 * Enum order must follow chronological order for state dependency check, @see hasReachedState().
 */
enum class LlmRequestState : int32_t
{
    kUNKNOWN = 0,                             ///< Unknown state
    kENCODER_INIT = 1,                        ///< Encoder phase starts (for encoder-decoder models)

    kDISAGG_GENERATION_INIT = 8,              ///< New Generation request arrived at generation model
    kDISAGG_GENERATION_TRANS_IN_PROGRESS = 9, ///< Transmitting the kv cache

    // schedulable states starts
    kCONTEXT_INIT = 10,                     ///< Context phase starts
    kDISAGG_CONTEXT_INIT_AND_TRANS = 11,    ///< Context phase starts and cache transmission is in progress,
                                            /// used in layer-wise transmission
    kDISAGG_GENERATION_TRANS_COMPLETE = 12, ///< Kv cache transmission are finished
    kGENERATION_IN_PROGRESS = 13,           ///< Generation phase is in progress

    // schedulable states ends
    kGENERATION_TO_COMPLETE = 14,           ///< Generation phase is to be completed
    kGENERATION_COMPLETE = 20,              ///< Generation phase completed
    kDISAGG_CONTEXT_TRANS_IN_PROGRESS = 21, ///< Waiting context-only request transmitting the kv cache,
                                            /// after computation finished
    kDISAGG_CONTEXT_COMPLETE = 22,          ///< Context-only request finished kv cache transmission.

    // error states
    kDISAGG_TRANS_ERROR = -1, ///< Error occurred during kv cache transmission
};

enum LlmRequestType
{
    LLMREQUEST_TYPE_CONTEXT_AND_GENERATION = 0, // Normal request will inference both context phase and generation phase
    LLMREQUEST_TYPE_CONTEXT_ONLY = 1,           // Only inference context phase
    LLMREQUEST_TYPE_GENERATION_ONLY = 2         // only inference generation phase
};

class ContextProgress;

template <typename TTensor, typename TStream = runtime::BufferManager::CudaStreamPtr>
class GenericLlmRequest
{
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

public:
    using SizeType32 = runtime::SizeType32;
    using TokenIdType = runtime::TokenIdType;
    using RequestIdType = std::uint64_t;
    using LoraTaskIdType = runtime::LoraTaskIdType;
    using VecTokens = std::vector<TokenIdType>;
    using TokenExtraIdType = runtime::TokenExtraIdType;
    using VecTokenExtraIds = runtime::VecTokenExtraIds;
    using VecLogProbs = std::vector<float>;
    using BeamTokens = std::vector<VecTokens>;
    using UniqueToken = runtime::UniqueToken;
    using VecUniqueTokens = runtime::VecUniqueTokens;
    using BeamUniqueTokens = std::vector<VecUniqueTokens>;
    using TensorPtr = TTensor;
    using LogitsPostProcessor = std::function<void(
        RequestIdType, TensorPtr&, BeamTokens const&, TStream const&, std::optional<RequestIdType>)>;
    using RequestPtr = std::shared_ptr<GenericLlmRequest>;
    using MillisecondsType = std::chrono::milliseconds;
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
    using Duration = std::chrono::time_point<std::chrono::steady_clock>::duration;
    using CacheSaltIDType = runtime::CacheSaltIDType;

    GenericLlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, std::shared_ptr<VecTokens> const& inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<std::shared_ptr<std::vector<std::vector<SizeType32>>>> multimodalHashes = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> multimodalPositions = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> multimodalLengths = std::nullopt,
        std::optional<TensorPtr> multimodalEmbedding = std::nullopt,
        std::optional<TensorPtr> mropeRotaryCosSin = std::nullopt,
        std::optional<SizeType32> mropePositionDeltas = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<executor::KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<std::shared_ptr<VecTokens>> const& draftTokens = std::nullopt,
        std::optional<TensorPtr> draftLogits = std::nullopt, bool excludeInputFromOutput = false,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false,
        std::optional<std::shared_ptr<VecTokens>> encoderInputTokens = std::nullopt, bool returnEncoderOutput = false,
        std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority,
        std::optional<TensorPtr> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<TensorPtr> crossAttentionMask = std::nullopt,
        LlmRequestType llmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<std::shared_ptr<VecTokenExtraIds>> inputTokenExtraIds = std::nullopt,
        SizeType32 numReturnSequences = 1, std::optional<executor::EagleConfig> eagleConfig = std::nullopt,
        std::optional<TensorPtr> skipCrossAttnBlocks = std::nullopt, bool returnPerfMetrics = false,
        std::optional<executor::GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<SizeType32> languageAdapterUid = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt,
        std::optional<executor::ContextPhaseParams> const& contextPhaseParams = std::nullopt,
        std::optional<CacheSaltIDType> cacheSaltID = std::nullopt, std::optional<TimePoint> arrivalTime = std::nullopt)
        : mRequestId(requestId)
        , mPromptLen(inputTokens->size())
        , mMaxNewTokens(maxNewTokens)
        , mSamplingConfig(samplingConfig)
        , mEndId(endId)
        , mPadId(padId)
        , mLogitsPostProcessor(std::move(logitsPostProcessor))
        , mApplyLogitsPostProcessorBatched(applyLogitsPostProcessorBatched)
        , mClientId(clientId)
        , mIsStreaming(isStreaming)
        , mOrigPromptLen(mPromptLen)
        , mNumPreDecodedTokens(samplingConfig.beamWidth, 0)
        , mMaxSentTokenLen(mPromptLen)
        , mEmbeddingBias(std::move(embeddingBias))
        , mBadWordsList(std::move(badWordsList))
        , mStopWordsList(std::move(stopWordsList))
        , mPositionIds(std::move(positionIds))
        , mPromptEmbeddingTable(std::move(promptEmbeddingTable))
        , mPromptVocabSize(promptVocabSize)
        , mMultimodalHashes(std::move(multimodalHashes))
        , mMultimodalPositions(std::move(multimodalPositions))
        , mMultimodalLengths(std::move(multimodalLengths))
        , mMultimodalEmbedding(std::move(multimodalEmbedding))
        , mMropeRotaryCosSin(std::move(mropeRotaryCosSin))
        , mMropePositionDeltas(mropePositionDeltas)
        , mLoraTaskId(loraTaskId)
        , mLoraWeights(std::move(loraWeights))
        , mLoraConfig(std::move(loraConfig))
        , mLookaheadConfig(std::move(lookaheadConfig))
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mContextChunkSize{mPromptLen}
        , mLogProbs(samplingConfig.beamWidth)
        , mCumLogProbs(samplingConfig.beamWidth)
        , mDraftTokens(draftTokens.value_or(std::make_shared<VecTokens>()))
        , mDraftLogits(std::move(draftLogits))
        , mReturnAllGeneratedTokens(isStreaming && (samplingConfig.beamWidth > 1))
        , mReturnContextLogits(returnContextLogits)
        , mReturnGenerationLogits(returnGenerationLogits)
        , mExcludeInputFromOutput(excludeInputFromOutput)
        , mEncoderTokens(std::move(encoderInputTokens))
        , mReturnEncoderOutput(returnEncoderOutput)
        , mPriority(priority)
        , mFinishReasons(samplingConfig.beamWidth)
        , mEncoderInputFeatures(std::move(encoderInputFeatures))
        , mEncoderOutputLength(encoderOutputLength)
        , mCrossAttentionMask(std::move(crossAttentionMask))
        , mLlmRequestType(llmRequestType)
        , mContextPhaseParams(contextPhaseParams)
        , mInputTokenExtraIds(std::move(inputTokenExtraIds))
        , mNumReturnSequences(numReturnSequences)
        , mEagleConfig(std::move(eagleConfig))
        , mSkipCrossAttnBlocks(std::move(skipCrossAttnBlocks))
        , mReturnPerfMetrics(returnPerfMetrics)
        , mGuidedDecodingParams(std::move(guidedDecodingParams))
        , mLanguageAdapterUid(languageAdapterUid)
        , mAllottedTimeMs(allottedTimeMs)
        , mCacheSaltID(cacheSaltID)
    {
        if (mEncoderTokens.has_value() || encoderInputFeatures.has_value())
        {
            mState = LlmRequestState::kENCODER_INIT;
        }

        initialize(*inputTokens, returnLogProbs, arrivalTime);
    }

    GenericLlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, VecTokens const& inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt, bool returnLogProbs = false,
        bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<VecTokens> draftTokens = std::nullopt, std::optional<TensorPtr> draftLogits = std::nullopt,
        bool excludeInputFromOutput = false, std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false, std::optional<VecTokens> encoderInputTokens = std::nullopt,
        bool returnEncoderOutput = false, std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority, SizeType32 numReturnSequences = 1,
        std::optional<SizeType32> languageAdapterUid = std::nullopt,
        std::optional<executor::ContextPhaseParams> const& contextPhaseParams = std::nullopt,
        std::optional<CacheSaltIDType> cacheSaltID = std::nullopt)
        : mRequestId(requestId)
        , mPromptLen(inputTokens.size())
        , mMaxNewTokens(maxNewTokens)
        , mSamplingConfig(samplingConfig)
        , mEndId(endId)
        , mPadId(padId)
        , mLogitsPostProcessor(logitsPostProcessor)
        , mApplyLogitsPostProcessorBatched(applyLogitsPostProcessorBatched)
        , mClientId(clientId)
        , mIsStreaming(isStreaming)
        , mOrigPromptLen(mPromptLen)
        , mNumPreDecodedTokens(samplingConfig.beamWidth, 0)
        , mMaxSentTokenLen(mPromptLen)
        , mEmbeddingBias(std::move(embeddingBias))
        , mBadWordsList(std::move(badWordsList))
        , mStopWordsList(std::move(stopWordsList))
        , mPositionIds(std::move(positionIds))
        , mPromptEmbeddingTable(std::move(promptEmbeddingTable))
        , mPromptVocabSize(promptVocabSize)
        , mLoraTaskId(loraTaskId)
        , mLoraWeights(std::move(loraWeights))
        , mLoraConfig(std::move(loraConfig))
        , mLookaheadConfig(lookaheadConfig)
        , mContextChunkSize(mPromptLen)
        , mLogProbs(samplingConfig.beamWidth)
        , mCumLogProbs(samplingConfig.beamWidth)
        , mDraftTokens(std::make_shared<VecTokens>(draftTokens.value_or(VecTokens())))
        , mDraftLogits(draftLogits)
        , mReturnAllGeneratedTokens(isStreaming && (samplingConfig.beamWidth > 1))
        , mReturnContextLogits(returnContextLogits)
        , mReturnGenerationLogits(returnGenerationLogits)
        , mExcludeInputFromOutput(excludeInputFromOutput)
        , mEncoderTokens(std::make_shared<VecTokens>(encoderInputTokens.value_or(VecTokens())))
        , mReturnEncoderOutput(returnEncoderOutput)
        , mPriority(priority)
        , mFinishReasons(samplingConfig.beamWidth)
        , mContextPhaseParams(contextPhaseParams)
        , mNumReturnSequences(numReturnSequences)
        , mLanguageAdapterUid(languageAdapterUid)
        , mCacheSaltID(cacheSaltID)
    {
        if (mEncoderTokens.has_value())
        {
            mState = LlmRequestState::kENCODER_INIT;
        }
        initialize(inputTokens, returnLogProbs);
    }

    GenericLlmRequest(RequestIdType requestId, executor::Request const& req)
        : mRequestId(requestId)
        , mPromptLen(req.getInputTokenIds().size())
        , mMaxNewTokens(req.getMaxTokens())
        , mSamplingConfig(req.getSamplingConfig(), req.getExternalDraftTokensConfig())
        , mEndId(req.getEndId())
        , mPadId(req.getPadId())
        , mClientId(req.getClientId())
        , mIsStreaming(req.getStreaming())
        , mOrigPromptLen(mPromptLen)
        , mNumPreDecodedTokens(mSamplingConfig.beamWidth, 0)
        , mMaxSentTokenLen(mPromptLen)
        , mContextChunkSize{mPromptLen}
        , mLogProbs(mSamplingConfig.beamWidth)
        , mCumLogProbs(mSamplingConfig.beamWidth)
        , mDraftTokens(std::make_shared<VecTokens>())
        , mReturnAllGeneratedTokens(req.getReturnAllGeneratedTokens())
        , mReturnContextLogits(req.getOutputConfig().returnContextLogits)
        , mReturnGenerationLogits(req.getOutputConfig().returnGenerationLogits)
        , mExcludeInputFromOutput(req.getOutputConfig().excludeInputFromOutput)
        , mReturnEncoderOutput(req.getOutputConfig().returnEncoderOutput)
        , mPriority(req.getPriority())
        , mFinishReasons(mSamplingConfig.beamWidth)
        , mEncoderOutputLength(req.getEncoderOutputLength())
        , mContextPhaseParams(req.getContextPhaseParams())
        , mEagleConfig(req.getEagleConfig())
        , mReturnPerfMetrics(req.getOutputConfig().returnPerfMetrics)
        , mGuidedDecodingParams(req.getGuidedDecodingParams())
        , mLanguageAdapterUid(req.getLanguageAdapterUid())
        , mAllottedTimeMs(req.getAllottedTimeMs())
        , mCacheSaltID(req.getCacheSaltID())
    {
        if (req.getRequestType() == executor::RequestType::REQUEST_TYPE_GENERATION_ONLY)
        {
            mState = LlmRequestState::kDISAGG_GENERATION_INIT;
        }
        if (mIsStreaming && mSamplingConfig.beamWidth > 1 && !mReturnAllGeneratedTokens)
        {
            TLLM_LOG_WARNING(
                "Setting mReturnAllGeneratedTokens to True since streaming AND beam search are done simultaneously. "
                "Returning the full beams at each streaming step is needed because beam search + streaming can change "
                "previous outputs. Initialize request with mReturnAllGeneratedTokens = True to dismiss this error. "
                "WARNING: using this option may increase network usage significantly (quadratically w.r.t output "
                "length).");
            mReturnAllGeneratedTokens = true;
        }

        if (mIsStreaming && mSamplingConfig.beamWidth > 1 && mReturnGenerationLogits)
        {
            TLLM_LOG_WARNING(
                "Returning generation logits when streaming is enabled and beamWidth > 1 is not allowed. "
                "This is because the logits may appear in irrelevant order when the beams are gathered, "
                "since logits are not. Disabling returnGenerationLogits.");
            mReturnGenerationLogits = false;
        }

        if (req.getEncoderInputTokenIds().has_value() || req.getEncoderInputFeatures().has_value())
        {
            mState = LlmRequestState::kENCODER_INIT;
            if (req.getEncoderInputTokenIds().has_value())
            {
                mEncoderTokens = std::make_shared<VecTokens>(req.getEncoderInputTokenIds().value());
            }
        }

        if (req.getEmbeddingBias())
        {
            mEmbeddingBias
                = tensorrt_llm::runtime::ITensor::view(executor::detail::toITensor(req.getEmbeddingBias().value()));
            // Add leading 1 dimension since that's what IFB code expects
            mEmbeddingBias.value()->unsqueeze(0);
        }
        if (req.getBadWords())
        {
            mBadWordsList = createListTensor(req.getBadWords().value());
        }
        if (req.getStopWords())
        {
            mStopWordsList = createListTensor(req.getStopWords().value());
        }

        if (req.getPositionIds())
        {
            mPositionIds = std::make_shared<std::vector<SizeType32>>(req.getPositionIds().value());
        }

        auto pTuningConfig = req.getPromptTuningConfig();
        if (pTuningConfig)
        {
            mPromptEmbeddingTable = tensorrt_llm::runtime::ITensor::view(
                executor::detail::toITensor(pTuningConfig.value().getEmbeddingTable()));
            TLLM_CHECK(mPromptEmbeddingTable.value()->getShape().nbDims == 2);
            mPromptVocabSize = mPromptEmbeddingTable.value()->getShape().d[0];
            mPromptEmbeddingTable.value()->unsqueeze(0);

            if (pTuningConfig->getInputTokenExtraIds())
            {
                mInputTokenExtraIds
                    = std::make_shared<VecTokenExtraIds>(pTuningConfig->getInputTokenExtraIds().value());
            }
        }
        auto mRopeConfig = req.getMropeConfig();
        if (mRopeConfig)
        {
            mMropeRotaryCosSin = executor::detail::toITensor(mRopeConfig.value().getMRopeRotaryCosSin());
            mMropePositionDeltas = mRopeConfig.value().getMRopePositionDeltas();
        }

        auto loraConfig = req.getLoraConfig();
        if (loraConfig)
        {
            mLoraTaskId = loraConfig->getTaskId();
            if (loraConfig.value().getWeights())
            {
                mLoraWeights = tensorrt_llm::runtime::ITensor::view(
                    executor::detail::toITensor(loraConfig.value().getWeights().value()));
                mLoraWeights.value()->unsqueeze(0);
            }

            if (loraConfig.value().getConfig())
            {
                mLoraConfig = tensorrt_llm::runtime::ITensor::view(
                    executor::detail::toITensor(loraConfig.value().getConfig().value()));
                mLoraConfig.value()->unsqueeze(0);
            }
        }

        auto externalDraftTokensConfig = req.getExternalDraftTokensConfig();
        if (externalDraftTokensConfig)
        {
            mDraftTokens = std::make_shared<VecTokens>(externalDraftTokensConfig.value().getTokens());

            if (externalDraftTokensConfig.value().getLogits())
            {
                mDraftLogits = executor::detail::toITensor(externalDraftTokensConfig.value().getLogits().value());
            }

            // NOTE: Draft acceptance threshold is stored in mSamplingConfig
        }

        if (req.getOutputConfig().additionalModelOutputs.has_value())
        {
            auto const& outputConfig = req.getOutputConfig();
            auto const& additionalModelOutputs = outputConfig.additionalModelOutputs.value();
            for (auto const& modelOutput : additionalModelOutputs)
            {
                if (modelOutput.gatherContext)
                {
                    mAdditionalContextOutputTensors.emplace(modelOutput.name, TensorPtr{});
                }
                mAdditionalGenerationOutputTensors.emplace(modelOutput.name, TensorPtr{});
            }
        }

        auto const& encoderInputFeatures = req.getEncoderInputFeatures();
        if (encoderInputFeatures.has_value())
        {
            mEncoderInputFeatures = executor::detail::toITensor(encoderInputFeatures.value());
        }
        else
        {
            mEncoderInputFeatures = std::nullopt;
        }

        auto const& crossAttentionMask = req.getCrossAttentionMask();
        if (crossAttentionMask.has_value())
        {
            mCrossAttentionMask = executor::detail::toITensor(crossAttentionMask.value());
        }
        else
        {
            mCrossAttentionMask = std::nullopt;
        }

        auto const& skipCrossAttnBlocks = req.getSkipCrossAttnBlocks();
        if (skipCrossAttnBlocks.has_value())
        {
            mSkipCrossAttnBlocks = executor::detail::toITensor(skipCrossAttnBlocks.value());
        }
        else
        {
            mSkipCrossAttnBlocks = std::nullopt;
        }

        switch (req.getRequestType())
        {
        case executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION:
            mLlmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION;
            break;
        case executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY:
            mLlmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_ONLY;
            break;
        case executor::RequestType::REQUEST_TYPE_GENERATION_ONLY:
            mLlmRequestType = LlmRequestType::LLMREQUEST_TYPE_GENERATION_ONLY;
            break;
        default: throw std::runtime_error("Unsupported request type found.");
        }

        initialize(req.getInputTokenIds(), req.getOutputConfig().returnLogProbs);
    }

    GenericLlmRequest(GenericLlmRequest&& request) = default;
    GenericLlmRequest(GenericLlmRequest const& request) = default;

    void setExcludeInputFromOutput(bool exclude)
    {
        mExcludeInputFromOutput = exclude;
    }

    /// @brief Get the params of the context
    /// @return The params of the context
    [[nodiscard]] std::optional<executor::ContextPhaseParams> const& getContextPhaseParams() const noexcept
    {
        return mContextPhaseParams;
    }

    void setContextPhaseParams(executor::ContextPhaseParams contextPhaseParams)
    {
        mContextPhaseParams = std::move(contextPhaseParams);
    }

    /// @brief Get the state params of the context
    /// @return The state params of the context
    [[nodiscard]] executor::DataTransceiverState const& getDataTransceiverState() const
    {
        TLLM_CHECK(mContextPhaseParams.has_value());
        return *static_cast<executor::DataTransceiverState const*>(mContextPhaseParams.value().getState());
    }

    [[nodiscard]] std::shared_ptr<ContextProgress> const& getContextProgress() const noexcept
    {
        return mContextProgress;
    }

    void setContextProgress(std::shared_ptr<ContextProgress> const& progress)
    {
        mContextProgress = progress;
    }

    /// @brief Get total number of tokens for this req (prompt + generated)
    /// @param beam The beam index
    /// @return  The number of tokens
    [[nodiscard]] SizeType32 getNumTokens(SizeType32 beam) const
    {
        return mTokens.at(beam).size() - mNumPreDecodedTokens[beam];
    }

    /// @brief Get the number of subrequests, the expected number of responses under non-streaming mode. In sampling
    /// mode, it will be equal to mSamplingConfig.numReturnSequences, while it will be equal to 1 in beam search.
    /// @return  The number of subrequests in total  request size.
    [[nodiscard]] SizeType32 getNumSubRequests() const
    {
        return mSamplingConfig.beamWidth == 1 ? mSamplingConfig.numReturnSequences.value_or(1) : 1;
    }

    /// @brief Get child requests spawned by this req.
    /// @return A vector of child requests.
    [[nodiscard]] std::vector<RequestPtr> const& getChildRequests() const
    {
        return mChildRequests;
    }

    /// @brief Get max number of tokens across all beams
    /// @return  The number of tokens
    [[nodiscard]] SizeType32 getMaxBeamNumTokens() const
    {
        SizeType32 maxTokens = 0;
        for (SizeType32 beam = 0; beam < mSamplingConfig.beamWidth; ++beam)
        {
            maxTokens = std::max(maxTokens, getNumTokens(beam));
        }
        return maxTokens;
    }

    /// @brief Get a token at a given position and beam index
    /// @param beam  The beam index
    /// @param pos The position of the token relative to beginning of the prompt
    /// @return  The token index
    [[nodiscard]] TokenIdType getToken(SizeType32 beam, SizeType32 pos) const
    {
        return mTokens.at(beam).at(pos);
    }

    /// @brief Get the tokens at a given beam index
    /// @param beam The beam index
    /// @return A vector of tokens for this beam index, includes the prompt
    [[nodiscard]] VecTokens const& getTokens(SizeType32 beam) const
    {
        return mTokens.at(beam);
    }

    /// @brief Get mutable reference to tokens for a specific beam
    /// @param beam The beam index
    /// @return Mutable reference to the tokens vector
    [[nodiscard]] VecTokens& getTokensMutable(SizeType32 beam)
    {
        return mTokens.at(beam);
    }

    /// @brief Get all tokens (input+output) for all beams
    /// @return A vector of vector of tokens.
    [[nodiscard]] BeamTokens const& getTokens() const
    {
        return mTokens;
    }

    /// @brief Get the unique tokens at a given beam index
    /// @param beam The beam index
    /// @return A vector of UniqueTokens for this beam index, includes the prompt
    [[nodiscard]] VecUniqueTokens const& getUniqueTokens(SizeType32 beam) const
    {
        return mUniqueTokens.at(beam);
    }

    /// @brief Get all unique tokens (input+output) for all beams
    /// @return A vector of vector of UniqueTokens.
    [[nodiscard]] BeamUniqueTokens const& getUniqueTokens() const
    {
        return mUniqueTokens;
    }

    /// @brief Get all extra input token ids
    /// @return A optional shared pointer to a vector of extra ids.
    [[nodiscard]] std::optional<std::shared_ptr<VecTokenExtraIds>> const& getInputTokensExtraIds() const
    {
        return mInputTokenExtraIds;
    }

    /// @brief Get input tokens to encoder
    /// @return A vector of tokens.
    [[nodiscard]] std::optional<std::shared_ptr<VecTokens>> const& getEncoderTokens() const
    {
        return mEncoderTokens;
    }

    /// @brief Get the unique tokens to encoder
    /// @return A vector of UniqueTokens for encoder
    [[nodiscard]] std::optional<std::shared_ptr<VecUniqueTokens>> const& getEncoderUniqueTokens() const
    {
        return mEncoderUniqueTokens;
    }

    /// @brief Get length of encoder input (could be tokens or features length)
    /// @return An integer.
    [[nodiscard]] SizeType32 getEncoderInputLen() const
    {
        if (mEncoderInputFeatures.has_value())
        {
            return getEncoderInputFeatures()->getShape().d[0];
        }
        if (getEncoderTokens().has_value())
        {
            return getEncoderTokens().value()->size();
        }

        TLLM_THROW("GenericLlmRequest::getEncoderInputLen - Do not have encoder length!");
    }

    /// @brief Get length of encoder output. Fall back to encoder input length if not present
    /// @return An integer.
    [[nodiscard]] SizeType32 getEncoderOutputLen() const
    {
        if (mEncoderOutputLength.has_value())
        {
            return mEncoderOutputLength.value();
        }

        return getEncoderInputLen();
    }

    [[nodiscard]] std::optional<std::shared_ptr<std::vector<SizeType32>>> getPositionIds() const
    {
        return mPositionIds;
    }

    /// @brief Get the draft tokens
    /// @return shared_ptr to vector of draft tokens
    [[nodiscard]] std::shared_ptr<VecTokens> const& getDraftTokens() const
    {
        return mDraftTokens;
    }

    /// @brief Get the logits for the draft tokens
    /// @return Tensor of draft logits
    [[nodiscard]] std::optional<TensorPtr> getDraftLogits() const
    {
        return mDraftLogits;
    }

    /// @brief Returns true if request has draft tokens
    /// @return flag
    [[nodiscard]] bool hasDraftTokens() const
    {
        return mDraftTokens && !mDraftTokens->empty();
    }

    /// @brief Get the maximum number of generated tokens among all rays in beam
    /// @return The number of generated tokens (doesn't include the prompt tokens)
    [[nodiscard]] SizeType32 getMaxNumGeneratedTokens() const
    {
        return getMaxBeamNumTokens() - mPromptLen;
    }

    /// @brief Returns true if request reaches max number of tokens in the next iteration.
    [[nodiscard]] bool willCompleteNextIteration() const
    {
        return getMaxNumGeneratedTokens() + mNumTokensPerIteration >= mMaxNewTokens;
    }

    [[nodiscard]] LlmRequestType getLlmRequestType() const
    {
        return mLlmRequestType;
    }

    /// @brief Add new generated tokens to the vector of tokens and set mLastTokens
    /// @param token The token to add
    /// @param beam The beam to which to add the new token
    /// @return  The number of tokens after the new token is added
    SizeType32 addNewToken(TokenIdType token, SizeType32 beam)
    {
        mLastTokens[beam] = token;
        mTokens.at(beam).push_back(token);
        // New token's extra id is 0
        mUniqueTokens.at(beam).push_back({token, 0});
        return getNumTokens(beam);
    }

    /// @brief Add new generated tokens to the vector of tokens and set mLastTokens
    /// @param beamTokens A vector containing the tokens to add for each beam index
    ///                   beamTokens is expected to be of size beamWidth
    void addNewTokens(VecTokens const& beamTokens)
    {
        assert(static_cast<size_t>(mSamplingConfig.beamWidth) == beamTokens.size());
        mLastTokens = beamTokens;
        for (std::size_t beam = 0; beam < beamTokens.size(); ++beam)
        {
            auto const outputId = beamTokens[beam];
            mTokens.at(beam).push_back(outputId);
            // New token's extra id is 0
            mUniqueTokens.at(beam).push_back({outputId, 0});
        }
    }

    /// @brief Set the number of pre-decoded tokens
    /// @param num_tokens The number of pre-decoded tokens
    /// @param beam The beam to which to set the number of pre-decoded tokens
    void setNumPreDecodedTokens(SizeType32 num_tokens, SizeType32 beam)
    {
        mNumPreDecodedTokens[beam] = num_tokens;
    }

    /// @brief Erases all previous generated tokens, only leaving the prompt.
    void clearGeneratedTokens()
    {
        TLLM_LOG_DEBUG("Clearing generated tokens for request %ld with promptlen %d", mRequestId, mPromptLen);
        for (auto& beam : mTokens)
        {
            beam.resize(mPromptLen);
        }
    }

    /// @brief Sets the generated tokens for all beams after gatherTree. Erases all previous generated tokens.
    /// @param generatedBeamTokens The generated tokens for all beams (vector of vector of tokens)
    void setGeneratedTokens(BeamTokens const& generatedBeamTokens)
    {
        TLLM_LOG_DEBUG("Setting generated tokens for request %ld", mRequestId);
        assert(generatedBeamTokens.size() == static_cast<size_t>(mSamplingConfig.beamWidth));

        for (size_t beamId = 0; beamId < generatedBeamTokens.size(); ++beamId)
        {
            auto& beamTokens = mTokens[beamId];
            beamTokens.resize(mPromptLen);
            beamTokens.insert(beamTokens.end(), generatedBeamTokens[beamId].begin(), generatedBeamTokens[beamId].end());
            auto& beamUniqueTokens = mUniqueTokens[beamId];
            beamUniqueTokens.resize(mPromptLen);
            for (auto const token : generatedBeamTokens[beamId])
            {
                beamUniqueTokens.push_back({token, 0});
            }
        }
    }

    /// @brief Sets the number of return sequences.
    /// @param numReturnSequences The number of return sequences.
    void setNumReturnSequences(SizeType32 const& numReturnSequences)
    {
        TLLM_CHECK_WITH_INFO(!isChild(), "A child request cannot change numReturnSequences.");
        TLLM_CHECK_WITH_INFO(
            numReturnSequences > 0, "numReturnSequences should be a positive integer, got %d.", numReturnSequences);
        TLLM_CHECK_WITH_INFO(mChildRequests.size() <= static_cast<size_t>(numReturnSequences),
            "Cannot set numReturnSequences %d smaller than the number %ld of child requests that have already created.",
            numReturnSequences, mChildRequests.size());
        mSamplingConfig.numReturnSequences = numReturnSequences;
        mSequenceFinalVec->resize(numReturnSequences);
    }

    [[nodiscard]] bool constexpr isChild() const noexcept
    {
        return mSequenceIndex > 0;
    }

    [[nodiscard]] RequestIdType getParentRequestId() const
    {
        return mParentRequestId;
    }

    /// @brief Return a vector of the last-generated tokens of shape [num_beams]
    [[nodiscard]] VecTokens const& getLastTokens()
    {
        return mLastTokens;
    }

    /// @brief Return the last-generated token of from a particular beam
    [[nodiscard]] TokenIdType const& getLastTokens(SizeType32 beam)
    {
        return mLastTokens[beam];
    }

    /// @brief Pause a request by moving the generated tokens to the prompt
    /// @param maxInputLen The maximum prompt len.
    void pause(SizeType32 maxInputLen)
    {
        // TODO: For beamWidth > 1, we would need to support swapping to avoid
        // recomputing from the start
        // As a temporary solution, we currently reset the tokens to the prompt
        if (mSamplingConfig.beamWidth > 1)
        {
            for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
            {
                auto& beamTokens = mTokens.at(beam);
                beamTokens.resize(mPromptLen);
                auto& beamUniqueTokens = mUniqueTokens.at(beam);
                beamUniqueTokens.resize(mPromptLen);
                if (returnLogProbs())
                {
                    mLogProbs.at(beam).clear();
                }
            }
        }
        else
        {
            SizeType32 newPromptLen = std::min(maxInputLen, mPromptLen + getMaxNumGeneratedTokens());
            for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
            {
                auto& beamTokens = mTokens.at(beam);
                beamTokens.resize(newPromptLen);
                auto& beamUniqueTokens = mUniqueTokens.at(beam);
                beamUniqueTokens.resize(newPromptLen);

                if (returnLogProbs())
                {
                    auto& logProb = mLogProbs.at(beam);
                    logProb.resize(newPromptLen - mPromptLen);
                }
            }
            mMaxNewTokens -= (newPromptLen - mPromptLen);
            mPromptLen = newPromptLen;
        }

        // for enc-dec models, pause means saving generated tokens to prompt but need to re-do encoder phase
        mState = mEncoderTokens.has_value() || mEncoderInputFeatures ? LlmRequestState::kENCODER_INIT
                                                                     : LlmRequestState::kCONTEXT_INIT;
        mContextCurrentPositionTarget = 0;
        mContextCurrentPositionDraft = 0;
        mPrepopulatedPromptLenTarget = 0;
        mPrepopulatedPromptLenDraft = 0;
        mContextChunkSize = mPromptLen;
        mSeqSlot.reset();
    }

    /// @brief Get the maximum length of tokens returned to the client. Use to ensure we don't return to
    /// client duplicated tokens.
    /// @return The maximum length of the tokens sent to the client.
    [[nodiscard]] SizeType32 getMaxSentTokenLen() const
    {
        return mMaxSentTokenLen;
    }

    /// @brief Sets the maximum length of tokens returned to the client. Use to ensure we don't return to
    /// client duplicated tokens.
    /// @param maxSentLength The new maximum length.
    void setMaxSentTokenLen(SizeType32 maxSentLength)
    {
        mMaxSentTokenLen = maxSentLength;
    }

    [[nodiscard]] std::optional<TensorPtr> getPromptEmbeddingTable() const
    {
        return mPromptEmbeddingTable;
    }

    [[nodiscard]] std::optional<TensorPtr>& getPromptEmbeddingTableMutable()
    {
        return mPromptEmbeddingTable;
    }

    [[nodiscard]] std::optional<SizeType32> getPromptVocabSize() const
    {
        return mPromptVocabSize;
    }

    [[nodiscard]] std::optional<std::shared_ptr<std::vector<std::vector<SizeType32>>>> getMultimodalHashes() const
    {
        return mMultimodalHashes;
    }

    [[nodiscard]] std::optional<std::shared_ptr<std::vector<SizeType32>>> getMultimodalPositions() const
    {
        return mMultimodalPositions;
    }

    [[nodiscard]] std::optional<std::shared_ptr<std::vector<SizeType32>>> getMultimodalLengths() const
    {
        return mMultimodalLengths;
    }

    [[nodiscard]] std::optional<TensorPtr> getMultimodalEmbedding() const
    {
        return mMultimodalEmbedding;
    }

    [[nodiscard]] std::optional<TensorPtr> getMropeRotaryCosSin() const
    {
        return mMropeRotaryCosSin;
    }

    [[nodiscard]] std::optional<SizeType32> getMropePositionDeltas() const
    {
        return mMropePositionDeltas;
    }

    [[nodiscard]] std::optional<LoraTaskIdType> getLoraTaskId() const
    {
        return mLoraTaskId;
    }

    void setLoraTaskId(LoraTaskIdType taskId)
    {
        mLoraTaskId = taskId;
    }

    [[nodiscard]] std::optional<TensorPtr> getLoraWeights() const
    {
        return mLoraWeights;
    }

    void setLoraWeights(TensorPtr weights)
    {
        mLoraWeights = weights;
    }

    void setPromptVocabSize(SizeType32 size)
    {
        mPromptVocabSize = size;
    }

    void clearLoraWeights()
    {
        mLoraWeights = std::nullopt;
    }

    [[nodiscard]] std::optional<TensorPtr> getLoraConfig() const
    {
        return mLoraConfig;
    }

    void setLoraConfig(TensorPtr config)
    {
        mLoraConfig = config;
    }

    void clearLoraConfig()
    {
        mLoraConfig = std::nullopt;
    }

    [[nodiscard]] std::optional<executor::LookaheadDecodingConfig> getLookaheadConfig() const
    {
        return mLookaheadConfig;
    }

    void setLookaheadConfig(executor::LookaheadDecodingConfig config)
    {
        mLookaheadConfig = config;
    }

    void clearLookaheadConfig()
    {
        mLookaheadConfig = std::nullopt;
    }

    [[nodiscard]] std::optional<executor::KvCacheRetentionConfig> getKvCacheRetentionConfig() const
    {
        return mKvCacheRetentionConfig;
    }

    void setKvCacheRetentionConfig(executor::KvCacheRetentionConfig config)
    {
        mKvCacheRetentionConfig = config;
    }

    [[nodiscard]] std::optional<executor::EagleConfig> getEagleConfig() const
    {
        return mEagleConfig;
    }

    void setEagleConfig(executor::EagleConfig config)
    {
        mEagleConfig = config;
    }

    [[nodiscard]] std::optional<executor::GuidedDecodingParams> getGuidedDecodingParams() const
    {
        return mGuidedDecodingParams;
    }

    void setGuidedDecodingParams(executor::GuidedDecodingParams guidedDecodingParams)
    {
        mGuidedDecodingParams = guidedDecodingParams;
    }

    [[nodiscard]] std::optional<TensorPtr> getEmbeddingBias() const
    {
        return mEmbeddingBias;
    }

    [[nodiscard]] std::optional<TensorPtr> getBadWordsList() const
    {
        return mBadWordsList;
    }

    [[nodiscard]] std::optional<TensorPtr> getStopWordsList() const
    {
        return mStopWordsList;
    }

    [[nodiscard]] bool returnLogProbs() const
    {
        return mSamplingConfig.outputLogProbs.has_value() ? mSamplingConfig.outputLogProbs->at(0) : false;
    }

    void setReturnLogProbs(bool returnLogProbs)
    {
        mSamplingConfig.outputLogProbs = {{returnLogProbs}};
        mSamplingConfig.cumLogProbs = {{returnLogProbs}};
    }

    [[nodiscard]] std::vector<VecLogProbs> const& getLogProbs() const
    {
        return mLogProbs;
    }

    [[nodiscard]] VecLogProbs const& getLogProbs(SizeType32 beam) const
    {
        return mLogProbs.at(beam);
    }

    void setLogProbs(VecLogProbs const& logProbs, SizeType32 beam)
    {
        mLogProbs.at(beam).resize(mPromptLen - mOrigPromptLen);
        mLogProbs.at(beam).insert(mLogProbs.at(beam).end(), logProbs.begin(), logProbs.end());
    }

    [[nodiscard]] VecLogProbs const& getCumLogProbs() const
    {
        return mCumLogProbs;
    }

    void setCumLogProb(float cumLogProb, SizeType32 beam)
    {
        mCumLogProbs.at(beam) = cumLogProb;
    }

    [[nodiscard]] SizeType32 getOrigPromptLen() const
    {
        return mOrigPromptLen;
    }

    [[nodiscard]] SizeType32 getPromptLen() const
    {
        return mPromptLen;
    }

    [[nodiscard]] SizeType32 getPrepopulatedPromptLen() const
    {
        return mUseDraftModel ? mPrepopulatedPromptLenDraft : mPrepopulatedPromptLenTarget;
    }

    void setPrepopulatedPromptLen(SizeType32 prepopulatedPromptLen, SizeType32 kvTokensPerBlock)
    {
        // Add debug log for prepopulatedPromptLen
        TLLM_LOG_DEBUG("Setting pre-populated prompt length for request %lu to %i (promptLen=%i).", mRequestId,
            prepopulatedPromptLen, getPromptLen());

        auto const promptLen = getPromptLen();

        // This check is make sure prepopulated prompt length (tokens already cached in KV cache) is less than prompt
        // length (total tokens in the prompt)
        TLLM_CHECK_WITH_INFO(prepopulatedPromptLen < promptLen,
            "Invalid state: prepopulatedPromptLen (%d) >= promptLen (%d) for request %lu", prepopulatedPromptLen,
            promptLen, mRequestId);

        auto& prePromptLen = mUseDraftModel ? mPrepopulatedPromptLenDraft : mPrepopulatedPromptLenTarget;
        auto& contextCurrentPosition = mUseDraftModel ? mContextCurrentPositionDraft : mContextCurrentPositionTarget;
        prePromptLen = prepopulatedPromptLen;

        if (prepopulatedPromptLen > 0)
        {
            // Currently, the runtime process is to apply for cache first and then determine prepopulation.
            // Use the prepopulated length to advance the context position and decrease chunk size if necessary.
            auto chunkSize = getContextChunkSize();
            if (prepopulatedPromptLen + chunkSize < promptLen)
            {
                // make sure to end at block boundary after current chunk
                auto const flooredEndPosition
                    = (prepopulatedPromptLen + chunkSize) / kvTokensPerBlock * kvTokensPerBlock;
                chunkSize = flooredEndPosition - prepopulatedPromptLen;
                TLLM_CHECK(chunkSize <= getContextChunkSize());
            }
            contextCurrentPosition = prepopulatedPromptLen;
            setContextChunkSize(chunkSize);

            if (!isLastContextChunk())
            {
                TLLM_CHECK_WITH_INFO((getContextCurrentPosition() + getContextChunkSize()) % kvTokensPerBlock == 0,
                    "To prevent cache fragmentation, the context position after current chunk should be divisible "
                    "by the number of tokens per block, except for the last chunk.");
            }
        }
    }

    void setDraftTokens(std::shared_ptr<VecTokens> const& draftTokens)
    {
        mDraftTokens = draftTokens;
    }

    void setDraftLogits(std::optional<TensorPtr> const& draftLogits)
    {
        mDraftLogits = draftLogits;
    }

    [[nodiscard]] SizeType32 getNumDraftTokens() const noexcept
    {
        return hasDraftTokens() ? static_cast<SizeType32>(mDraftTokens->size()) : 0;
    }

    void discardDraftTokens(SizeType32 numTokensToDiscard)
    {
        TLLM_CHECK_WITH_INFO(
            numTokensToDiscard > 0, "Can only discard a positive amount of draft tokens, got %d", numTokensToDiscard);
        TLLM_CHECK_WITH_INFO(numTokensToDiscard <= getNumDraftTokens(),
            "Can't discard more draft tokens (%d) than exists (%d).", numTokensToDiscard, getNumDraftTokens());
        mDraftTokens->resize(getNumDraftTokens() - numTokensToDiscard);

        if (mDraftLogits)
        {
            auto shape = mDraftLogits.value()->getShape();
            shape.d[0] = getNumDraftTokens();
            mDraftLogits.value()->reshape(shape);
        }
    }

    void updateNumTokensPerIteration(SizeType32 numTokensPerIteration, runtime::ModelConfig const& modelConfig)
    {
        mNumTokensPerIteration = std::max(1, numTokensPerIteration);

        if (modelConfig.hasSpeculativeDecodingModule() && getReturnPerfMetrics() && hasDraftTokens())
        {
            auto& specDecMetrics = mPerfMetrics.speculativeDecoding;
            specDecMetrics.totalAcceptedDraftTokens += mNumTokensPerIteration - 1;
            auto const maxAcceptedDraftTokens = modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen();
            specDecMetrics.totalDraftTokens += std::min(getNumDraftTokens(), maxAcceptedDraftTokens);
        }
    }

    [[nodiscard]] SizeType32 getNumTokensPerIteration() const
    {
        return mNumTokensPerIteration;
    }

    void setReturnEncoderOutput(bool const returnEncoderOutput)
    {
        mReturnEncoderOutput = returnEncoderOutput;
    }

    [[nodiscard]] bool getReturnEncoderOutput() const
    {
        return mReturnEncoderOutput;
    }

    [[nodiscard]] TensorPtr const& getEncoderOutputHost() const
    {
        return mEncoderOutputHost;
    }

    [[nodiscard]] TensorPtr getEncoderInputFeatures() const
    {
        return mEncoderInputFeatures.value_or(nullptr);
    }

    void setEncoderOutputHost(TensorPtr encoderOutputHost)
    {
        mEncoderOutputHost = std::move(encoderOutputHost);
    }

    void setEncoderOutput(TensorPtr encoderOutput)
    {
        mEncoderOutput = std::move(encoderOutput);
    }

    void allocEncoderOutputHost(SizeType32 encoderHiddenSize, nvinfer1::DataType dataType)
    {
        mEncoderOutputHost = runtime::BufferManager::pinned(
            runtime::ITensor::makeShape({getEncoderOutputLen(), encoderHiddenSize}), dataType);
    }

    [[nodiscard]] TensorPtr const& getEncoderOutput() const noexcept
    {
        return mEncoderOutput;
    }

    [[nodiscard]] TensorPtr const& getEncoderHiddenStates() const noexcept
    {
        return mEncoderHiddenStates;
    }

    void allocEncoderOutput(runtime::BufferManager const& manager, nvinfer1::DataType dataType)
    {
        // unique_ptr --> shared_ptr ownership move
        mEncoderOutput = std::move(manager.emptyTensor(runtime::MemoryType::kGPU, dataType));
    }

    void allocEncoderHiddenStates(runtime::BufferManager const& manager, nvinfer1::DataType dataType)
    {
        // unique_ptr --> shared_ptr ownership move
        mEncoderHiddenStates = std::move(manager.emptyTensor(runtime::MemoryType::kGPU, dataType));
    }

    void freeEncoderOutputBuffers()
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

        TLLM_LOG_DEBUG(
            "Encoder output buffers use count: %u, %u", mEncoderOutput.use_count(), mEncoderHiddenStates.use_count());

        // TODO: better ways to free shared_ptr buffers
        mEncoderOutput.reset();
        mEncoderHiddenStates.reset();

        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    }

    [[nodiscard]] TensorPtr getCrossAttentionMask() const
    {
        return mCrossAttentionMask.value_or(nullptr);
    }

    [[nodiscard]] TensorPtr getSkipCrossAttnBlocks() const
    {
        return mSkipCrossAttnBlocks.value_or(nullptr);
    }

    [[nodiscard]] bool constexpr getReturnPerfMetrics() const noexcept
    {
        return mReturnPerfMetrics;
    }

    void constexpr setReturnPerfMetrics(bool returnPerfMetrics) noexcept
    {
        mReturnPerfMetrics = returnPerfMetrics;
    }

    [[nodiscard]] executor::RequestPerfMetrics const& getPerfMetrics() const noexcept
    {
        return mPerfMetrics;
    }

    void setFirstScheduledTime()
    {
        if (mPerfMetrics.timingMetrics.firstScheduledTime == executor::RequestPerfMetrics::TimePoint{})
        {
            mPerfMetrics.timingMetrics.firstScheduledTime = getSteadyClockNow();
        }
    }

    [[nodiscard]] bool constexpr isStreaming() const noexcept
    {
        return mIsStreaming;
    }

    void constexpr setStreaming(bool isStreaming) noexcept
    {
        mIsStreaming = isStreaming;
    }

    void setPriority(executor::PriorityType priority) noexcept
    {
        mPriority = priority;
    }

    void setReturnAllGeneratedTokens(bool const returnAllGeneratedTokens)
    {
        TLLM_CHECK_WITH_INFO(!mIsStreaming || mSamplingConfig.beamWidth == 1 || returnAllGeneratedTokens,
            "returnAllGeneratedTokens must be true if streaming AND beam search are used.");
        mReturnAllGeneratedTokens = returnAllGeneratedTokens;
    }

    [[nodiscard]] bool getReturnAllGeneratedTokens()
    {
        return mReturnAllGeneratedTokens;
    }

    void setAllottedTimeMs(MillisecondsType allottedTimeMs)
    {
        mAllottedTimeMs = allottedTimeMs;
    }

    void setReturnContextLogits(bool const returnContextLogits)
    {
        mReturnContextLogits = returnContextLogits;
    }

    [[nodiscard]] bool getReturnContextLogits() const
    {
        return mReturnContextLogits;
    }

    void setReturnGenerationLogits(bool const returnGenerationLogits)
    {
        TLLM_CHECK_WITH_INFO(!(mIsStreaming && mSamplingConfig.beamWidth > 1 && returnGenerationLogits),
            "returnGenerationLogits must be false if streaming AND beam search are used.");
        mReturnGenerationLogits = returnGenerationLogits;
    }

    [[nodiscard]] bool getReturnGenerationLogits() const
    {
        return mReturnGenerationLogits;
    }

    [[nodiscard]] TensorPtr const& getContextLogitsHost() const
    {
        return mContextLogitsHost;
    }

    /// @param contextLogitsHost Expected shape [promtLen, vocabSizePadded]
    void setContextLogitsHost(TensorPtr contextLogitsHost)
    {
        mContextLogitsHost = std::move(contextLogitsHost);
    }

    void allocContextLogitsHost(SizeType32 vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        mContextLogitsHost = runtime::BufferManager::pinnedPool(
            runtime::ITensor::makeShape({mPromptLen, vocabSizePadded}), logitsDataType);
    }

    [[nodiscard]] TensorPtr const& getGenerationLogitsHost() const
    {
        return mGenerationLogitsHost;
    }

    /// @param generationLogitsHost Expected shape
    /// * [beamWidth, maxNewTokens, vocabSizePadded] for non-speculative decoding
    /// * [1, numDraftTokens + 1, vocabSizePadded] for speculative decoding
    void setGenerationLogitsHost(TensorPtr generationLogitsHost)
    {
        mGenerationLogitsHost = std::move(generationLogitsHost);
    }

    void allocGenerationLogitsHost(SizeType32 vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        if (mIsStreaming)
        {
            // If streaming mode, the complete generation logits shape will be [1, beamWidth, vocabSizePadded],
            // or [allGeneratedTokens, beamWidth, vocabSizePadded] if mReturnAllGeneratedTokens is True.
            // This could reduce unnecessary format conversions and allows the data to be returned directly.
            mGenerationLogitsHost = runtime::BufferManager::pinnedPool(
                runtime::ITensor::makeShape({mMaxNewTokens, mSamplingConfig.beamWidth, vocabSizePadded}),
                logitsDataType);
        }
        else
        {
            mGenerationLogitsHost = runtime::BufferManager::pinnedPool(
                runtime::ITensor::makeShape({mSamplingConfig.beamWidth, mMaxNewTokens, vocabSizePadded}),
                logitsDataType);
        }
    }

    void allocTargetModelAcceptedTokenLogitsHost(SizeType32 vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        mGenerationLogitsHost = runtime::BufferManager::pinnedPool(
            runtime::ITensor::makeShape({1, getNumDraftTokens() + 1, vocabSizePadded}), logitsDataType);
    }

    [[nodiscard]] std::vector<TensorPtr> const& getGenerationLogitsFragments() const
    {
        return mGenerationLogitsFragments;
    }

    void addGenerationLogitsFragment(TensorPtr& genLogits)
    {
        mGenerationLogitsFragments.push_back(genLogits);
    }

    [[nodiscard]] SizeType32 getGenerationLogitsFragmentsSize() const noexcept
    {
        return static_cast<SizeType32>(mGenerationLogitsFragments.size());
    }

    void clearGenerationLogitsFragments() noexcept
    {
        mGenerationLogitsFragments.clear();
    }

    [[nodiscard]] bool hasAdditionalOutputs() const noexcept
    {
        return !mAdditionalContextOutputTensors.empty() || !mAdditionalGenerationOutputTensors.empty();
    }

    [[nodiscard]] TensorMap const& getAdditionalContextOutputs() const
    {
        return mAdditionalContextOutputTensors;
    }

    [[nodiscard]] TensorMap const& getAdditionalGenerationOutputs() const
    {
        return mAdditionalGenerationOutputTensors;
    }

    template <typename TypeFunc, typename ShapeFunc>
    void allocAdditionalOutputs(TypeFunc getTensorDataType, ShapeFunc getTensorShape)
    {
        for (auto& outputTensor : mAdditionalContextOutputTensors)
        {
            auto const& outputTensorName = outputTensor.first;
            auto const dataType = getTensorDataType(outputTensorName);
            auto shape = getTensorShape(outputTensorName);
            TLLM_CHECK_WITH_INFO(shape.d[0] == -1, "First dimension of additional output tensor '%s' must be dynamic",
                outputTensorName.c_str());
            shape.d[0] = mPromptLen;
            auto tensor = runtime::BufferManager::pinnedPool(shape, dataType);
            outputTensor.second = std::move(tensor);
        }
        for (auto& outputTensor : mAdditionalGenerationOutputTensors)
        {
            auto const& outputTensorName = outputTensor.first;
            auto const dataType = getTensorDataType(outputTensorName);
            auto shape = getTensorShape(outputTensorName);
            TLLM_CHECK_WITH_INFO(shape.d[0] == -1, "First dimension of additional output tensor '%s' must be dynamic",
                outputTensorName.c_str());
            shape.d[0] = mMaxNewTokens;
            shape = runtime::ITensor::unsqueeze(shape, 0);
            shape.d[0] = mSamplingConfig.beamWidth;
            auto tensor = runtime::BufferManager::pinnedPool(shape, dataType);
            outputTensor.second = std::move(tensor);
        }
    }

    void setState(LlmRequestState state)
    {
        TLLM_LOG_DEBUG("Set request %lu from state %d to %d", mRequestId, mState, state);
        mState = state;
    }

    [[nodiscard]] LlmRequestState getState() const noexcept
    {
        return mState;
    }

    [[nodiscard]] bool hasReachedState(LlmRequestState state) const noexcept
    {
        return mState >= state;
    }

    [[nodiscard]] bool isEncoderInitState() const noexcept
    {
        return mState == LlmRequestState::kENCODER_INIT;
    }

    [[nodiscard]] bool isContextInitState() const noexcept
    {
        return mState == LlmRequestState::kCONTEXT_INIT || mState == LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS;
    }

    [[nodiscard]] bool isContextFinished() const noexcept
    {
        return isGenerationInProgressState() || mState == LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS;
    }

    [[nodiscard]] bool isGenerationInProgressState() const noexcept
    {
        return mState == LlmRequestState::kGENERATION_IN_PROGRESS || mState == LlmRequestState::kGENERATION_TO_COMPLETE
            || mState == LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE;
    }

    [[nodiscard]] bool isGenerationToCompleteState() const noexcept
    {
        return mState == LlmRequestState::kGENERATION_TO_COMPLETE;
    }

    [[nodiscard]] bool isGenerationCompleteState() const noexcept
    {
        return mState == LlmRequestState::kGENERATION_COMPLETE;
    }

    [[nodiscard]] bool isDisaggGenerationInitState() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_GENERATION_INIT;
    }

    [[nodiscard]] bool isDisaggGenerationTransmissionComplete() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE;
    }

    [[nodiscard]] bool isDisaggGenerationTransmissionInProgress() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS;
    }

    [[nodiscard]] bool isDisaggContextTransmissionState() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS
            || mState == LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS;
    }

    [[nodiscard]] bool isDisaggContextCompleteState() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_CONTEXT_COMPLETE;
    }

    [[nodiscard]] executor::RequestStage getRequestStage() const
    {
        switch (mState)
        {
        case batch_manager::LlmRequestState::kENCODER_INIT: return executor::RequestStage::kENCODER_IN_PROGRESS; break;
        case batch_manager::LlmRequestState::kCONTEXT_INIT: return executor::RequestStage::kCONTEXT_IN_PROGRESS; break;
        case batch_manager::LlmRequestState::kGENERATION_IN_PROGRESS:
        case batch_manager::LlmRequestState::kGENERATION_TO_COMPLETE:
        case batch_manager::LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE:
        case batch_manager::LlmRequestState::kDISAGG_GENERATION_INIT:
        case batch_manager::LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS:
            return executor::RequestStage::kGENERATION_IN_PROGRESS;
            break;
        default: TLLM_LOG_ERROR("Unexpected request state."); return executor::RequestStage::kGENERATION_COMPLETE;
        }
    }

    [[nodiscard]] bool isContextOnlyRequest() const noexcept
    {
        return mLlmRequestType == LlmRequestType::LLMREQUEST_TYPE_CONTEXT_ONLY;
    }

    [[nodiscard]] bool isGenerationOnlyRequest() const noexcept
    {
        return mLlmRequestType == LlmRequestType::LLMREQUEST_TYPE_GENERATION_ONLY;
    }

    void setContextCurrentPosition(SizeType32 contextCurrentPosition)
    {
        mContextCurrentPositionDraft = contextCurrentPosition;
        mContextCurrentPositionTarget = contextCurrentPosition;
    }

    /// When chunked, the position of the current chunk is returned. Otherwise, only the beginning
    /// or end of the context is returned.
    [[nodiscard]] SizeType32 getContextCurrentPosition() const noexcept
    {
        return mUseDraftModel ? mContextCurrentPositionDraft : mContextCurrentPositionTarget;
    }

    /// Return the length of the context that has not yet been processed.
    [[nodiscard]] SizeType32 getContextRemainingLength() const noexcept
    {
        return mPromptLen - getContextCurrentPosition();
    }

    [[nodiscard]] SizeType32 getContextChunkSize() const
    {
        TLLM_CHECK_WITH_INFO(
            isContextInitState() || isDisaggGenerationInitState() || isDisaggGenerationTransmissionComplete(),
            "getContextChunkSize is only possible during the context phase or generation init phase.");
        return mContextChunkSize;
    }

    /// To set the context chunk size, throw an exception when the chunk size is negative. If the chunk
    /// size is greater than the remaining length of the context, the size will be reduced to fit the
    /// remaining length.
    void setContextChunkSize(SizeType32 size)
    {
        TLLM_CHECK_WITH_INFO(
            isContextInitState() || isDisaggGenerationInitState() || isDisaggGenerationTransmissionComplete(),
            "setContextChunkSize is only possible during the context phase or generation init phase.");
        TLLM_CHECK_WITH_INFO(size >= 0, "The chunk size of context (%d) can't be negative.", size);
        mContextChunkSize = std::min(size, getContextRemainingLength());
    }

    /// Determines whether the current position is only one chunk away from the end of the context.
    [[nodiscard]] bool isLastContextChunk() const
    {
        return isDisaggGenerationInitState() || isDisaggGenerationTransmissionComplete()
            || getContextCurrentPosition() + getContextChunkSize() == mPromptLen;
    }

    /// Returns whether the position is at the beginning of the context.
    [[nodiscard]] bool isFirstContextChunk() const noexcept
    {
        // The number of cached token is encountered in mContextCurrentPosition,
        // so the start position of the context is mPrepopulatedPromptLen.
        return getContextCurrentPosition() == getPrepopulatedPromptLen();
    }

    /// Move the cursor forward one chunk. When not chunked, move forward to the end of the context.
    void moveToNextContextChunk()
    {
        TLLM_CHECK_WITH_INFO(isContextInitState(), "Chunking is only possible during the context phase.");

        mContextCurrentPositionDraft += getContextChunkSize();
        mContextCurrentPositionTarget += getContextChunkSize();
        setContextChunkSize(0);
    }

    [[nodiscard]] executor::PriorityType priority() const noexcept
    {
        return mPriority;
    }

    /// Get the counter of decoding iterations.
    SizeType32 getDecodingIter()
    {
        return mDecodingIter;
    }

    /// Increment the counter of decoding iterations.
    void advanceDecodingIter()
    {
        mDecodingIter++;
    }

    /// @brief  Return the average number of decoded tokens per iteration. For standard model it is 1.
    /// For speculative decoding model >= 1 -- number of draft tokens accepted per step + 1.
    [[nodiscard]] float getAvgDecodedTokensPerIter() const noexcept
    {
        if (mDecodingIter == 0)
        {
            return 0.F;
        }
        return static_cast<float>(getMaxNumGeneratedTokens()) / mDecodingIter;
    }

    /// @brief Get the beam width of the current decoding step.
    /// @details Return `mSamplingConfig.beamWidth` in decoding modes beside Variable-Beam-Width-Search (VBWS).
    /// Or returns a scalar value from `mSamplingConfig.beamWidthArray` indexing by `mDecodingIter` in VBWS.
    ///
    /// Calling in context phase, it returns the beam width of the first generation step, which is used for copying
    /// logits (function `copyGenerationLogits` as example).
    ///
    /// Calling in generation phase, it returns the beam width of the input tokens in the current generation step, which
    /// is used for computing I/O tensor shapes for TRT engine (function `RuntimeBuffers::setBufferSizes` as example).
    ///
    /// For example, we have a request with beamWidthArray = [2,3,4], the generation process can be:
    ///
    /// input_ids[1,inputLength] --->
    /// ---> [Forward, step == 0] ---> logits[1, 1, vocabSize] ---> [BeamSearchDecoder] ---> tokens[1, 2]
    ///     Context Phase, getBeamWidthByIter() returns 2 for copying logits
    ///     Decoder uses beamWidthIn=2, beamWidthOut=2 to get top 2 tokens
    /// ---> [Forward, step == 1] ---> logits[1, 2, vocabSize] ---> [BeamSearchDecoder] ---> tokens[1, 3]
    ///     Generation phase, getBeamWidthByIter() returns 2 for computing tensor shapes
    ///     Decoder uses beamWidthIn=2, beamWidthOut=3 to get top 3 tokens
    /// ---> [Forward, step == 2] ---> logits[1, 3, vocabSize] ---> [BeamSearchDecoder] ---> tokens[1, 4]
    ///     Generation phase, getBeamWidthByIter() returns 3 for computing tensor shapes
    ///     Decoder uses beamWidthIn=3, beamWidthOut=4 to get top 4 tokens
    /// ---> [Forward, step == 3] ---> logits[1, 4, vocabSize] ---> [BeamSearchDecoder] ---> tokens[1, 4]
    ///     Generation phase, getBeamWidthByIter() returns 4 for computing tensor shapes
    ///     Decoder uses beamWidthIn=4, beamWidthOut=4 to get top 4 tokens
    ///     i.e. the same as normal Beam Search of `beamWidth==4`
    /// @param: forNextIteration: get beam width for next step rather than current beam width.
    [[nodiscard]] SizeType32 getBeamWidthByIter(bool forNextIteration = false);

    [[nodiscard]] bool isFinished() const noexcept
    {
        return isGenerationCompleteState() || mState == LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS;
    }

    /// Returns true if finished_reason is length for all beams
    [[nodiscard]] bool isFinishedDueToLength() const noexcept
    {
        return std::all_of(mFinishReasons.begin(), mFinishReasons.end(),
            [](auto reason) { return reason == executor::FinishReason::kLENGTH; });
    }

    [[nodiscard]] bool isTimedOut() const
    {
        if (!mAllottedTimeMs.has_value())
        {
            return false;
        }
        auto const currentTime = std::chrono::steady_clock::now();
        auto const elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - mStartTime));
        TLLM_LOG_DEBUG("Checked timeOut for request %ld with allotted Time %ld after time %ld and got %d", mRequestId,
            mAllottedTimeMs->count(), elapsed.count(), (elapsed >= mAllottedTimeMs));

        return elapsed >= *mAllottedTimeMs;
    }

    void setFinishedReason(executor::FinishReason reason, SizeType32 beam)
    {
        mFinishReasons.at(beam) = reason;
    }

    void setDecodingIter(SizeType32 iter)
    {
        mDecodingIter = iter;
    }

    void setKvCacheTransferStart(TimePoint const& time)
    {
        mPerfMetrics.timingMetrics.kvCacheTransferStart = maybeToGlobalSteadyClock(time);
    }

    void setKvCacheTransferEnd(TimePoint const& time)
    {
        mPerfMetrics.timingMetrics.kvCacheTransferEnd = maybeToGlobalSteadyClock(time);
    }

    TimePoint getKvCacheTransferStart()
    {
        return mPerfMetrics.timingMetrics.kvCacheTransferStart;
    }

    TimePoint getKvCacheTransferEnd()
    {
        return mPerfMetrics.timingMetrics.kvCacheTransferEnd;
    }

    [[nodiscard]] double getKvCacheTransferTimeMS() const
    {
        // get max with 0 in case this function is called while end time is not recorded
        return std::max(0.0,
            std::chrono::duration<double, std::milli>(
                mPerfMetrics.timingMetrics.kvCacheTransferEnd - mPerfMetrics.timingMetrics.kvCacheTransferStart)
                .count());
    }

    void updateKvCacheSize(size_t targetBufferSize) const
    {
        mPerfMetrics.timingMetrics.kvCacheSize += targetBufferSize;
    }

    void setKvCacheSize(size_t targetBufferSize) const
    {
        mPerfMetrics.timingMetrics.kvCacheSize = targetBufferSize;
    }

    [[nodiscard]] size_t getKvCacheSize() const
    {
        return mPerfMetrics.timingMetrics.kvCacheSize;
    }

    void updateAllocTotalBlocksPerRequest(SizeType32 allocTotalBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numTotalAllocatedBlocks += allocTotalBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getAllocTotalBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numTotalAllocatedBlocks;
    }

    void updateAllocNewBlocksPerRequest(SizeType32 allocNewBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numNewAllocatedBlocks += allocNewBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getAllocNewBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numNewAllocatedBlocks;
    }

    void updateReusedBlocksPerRequest(SizeType32 reusedBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numReusedBlocks += reusedBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getReusedBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numReusedBlocks;
    }

    [[nodiscard]] std::optional<SizeType32> getLanguageAdapterUid() const
    {
        return mLanguageAdapterUid;
    }

    [[nodiscard]] std::optional<CacheSaltIDType> getCacheSaltID() const
    {
        return mCacheSaltID;
    }

    std::vector<SizeType32> getLanguageAdapterRouting(
        SizeType32 const reqNumLanguages, SizeType32 const inputLength) const
    {
        auto const reqLanguageAdapterUid = getLanguageAdapterUid().value();
        TLLM_CHECK_WITH_INFO(reqLanguageAdapterUid < reqNumLanguages, "Language adapter uid is out of range.\n");
        // Copy the same routing info for all the tokens in this request
        return std::vector<SizeType32>(inputLength, reqLanguageAdapterUid);
    }

    /// @brief mark all beams as finished by the given reason. Marks only unfinished beams.
    void finishByReason(executor::FinishReason finishReason)
    {
        if (finishReason == executor::FinishReason::kTIMED_OUT)
        {
            TLLM_LOG_DEBUG("Request %ld finished by timeout after %f sec", mRequestId,
                std::chrono::duration<float>(getSteadyClockNow() - mStartTime).count());
        }
        if (finishReason == executor::FinishReason::kCANCELLED)
        {
            TLLM_LOG_DEBUG("Request %ld finished by cancel", mRequestId);
        }

        for (int beam = 0; beam < mSamplingConfig.beamWidth; ++beam)
        {
            if (mFinishReasons.at(beam) == executor::FinishReason::kNOT_FINISHED)
            {
                setFinishedReason(finishReason, beam);
            }
        }
        mState = LlmRequestState::kGENERATION_COMPLETE;
    }

    void updateMissedBlocksPerRequest(SizeType32 missedBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numMissedBlocks += missedBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getMissedBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numMissedBlocks;
    }

    [[nodiscard]] float getKVCacheHitRatePerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numReusedBlocks == 0
            ? 0
            : static_cast<float>(mPerfMetrics.kvCacheMetrics.numReusedBlocks)
                / (static_cast<float>(
                    mPerfMetrics.kvCacheMetrics.numReusedBlocks + mPerfMetrics.kvCacheMetrics.numMissedBlocks));
    }

    void updatePerfMetrics(executor::IterationType iter)
    {
        auto const currentTokenTime = getSteadyClockNow();

        if (!mPerfMetrics.firstIter)
        {
            mPerfMetrics.firstIter = iter;
            mPerfMetrics.timingMetrics.firstTokenTime = currentTokenTime;
        }

        mPerfMetrics.iter = iter;

        if (isFinished())
        {
            mPerfMetrics.lastIter = iter;
            mPerfMetrics.timingMetrics.lastTokenTime = currentTokenTime;
        }
    }

    void setIsDummyRequest(bool isDummyRequest)
    {
        mIsDummyRequest = isDummyRequest;
    }

    [[nodiscard]] bool isDummyRequest() const
    {
        return mIsDummyRequest;
    }

    void setUseDraftModel(bool useDraftModel)
    {
        mUseDraftModel = useDraftModel;
    }

    [[nodiscard]] bool useDraftModel() const
    {
        return mUseDraftModel;
    }

    // If mGlobalSteadyClockOffset is set, return a global steady clock time point, otherwise return local steady clock
    // time point
    [[nodiscard]] TimePoint getSteadyClockNow() const
    {
        const TimePoint time_point = std::chrono::steady_clock::now();

        return maybeToGlobalSteadyClock(time_point);
    }

    RequestIdType mRequestId;
    SizeType32 mPromptLen;
    SizeType32 mMaxNewTokens;
    runtime::SamplingConfig mSamplingConfig;
    std::optional<TokenIdType> mEndId{std::nullopt};
    std::optional<TokenIdType> mPadId{std::nullopt};
    std::optional<SizeType32> mSeqSlot{std::nullopt};
    std::optional<LogitsPostProcessor> mLogitsPostProcessor{std::nullopt};
    bool mApplyLogitsPostProcessorBatched{false};
    std::optional<RequestIdType> mClientId{std::nullopt};

    // Position of mask token in GLM model inputs
    SizeType32 mMaskPosition{0};

    LlmRequestState mState{LlmRequestState::kCONTEXT_INIT};

    // current position of the prompt tuning table (only used in chunked prefill mode)
    SizeType32 mPtableCurrentPosition{0};

    // The offset between local steady clock and global steady clock (at rank 0)
    inline static std::optional<Duration> mGlobalSteadyClockOffset{std::nullopt};

protected:
    bool mIsStreaming;

    // List of tokens generated at the current step, used as the input tokens to the next step, [beamSize]
    // `mLastTokens[beam]` is not equal to `mTokens.back()[beam]` in "Streaming + Beam Search" mode
    // since `mTokens` will be overwritten by the gathered tokens.
    VecTokens mLastTokens;

    // List of tokens including input prompt and generated part, [beamSize, mPromptLen + getMaxNumGeneratedTokens()]
    BeamTokens mTokens;

    // Length of input prompt tokens, never changes during generation process.
    SizeType32 mOrigPromptLen;

    // List of numbers of pre-deocded tokens on the last PP rank when using pipeline parallelism.
    // It is introduced as a WAR to solve the hanging problem caused by overestimating the used KV cache on the last PP
    // rank (because new tokens are decoded earlier). By excluding the numbers of pre-decoded tokens, the used KV cache
    // can be estimated correctly.
    std::vector<SizeType32> mNumPreDecodedTokens;

    // Number of tokens already in KV cache before context phase.
    // A value > 0 indicates cached KV cache blocks were reused.
    // Up to inputLen - 1 tokens can be reused.
    SizeType32 mPrepopulatedPromptLenTarget{0};
    SizeType32 mPrepopulatedPromptLenDraft{0};

    SizeType32 mMaxSentTokenLen;

    std::optional<TensorPtr> mEmbeddingBias{std::nullopt};
    std::optional<TensorPtr> mBadWordsList{std::nullopt};
    std::optional<TensorPtr> mStopWordsList{std::nullopt};

    std::optional<std::shared_ptr<std::vector<SizeType32>>> mPositionIds{std::nullopt};

    std::optional<TensorPtr> mPromptEmbeddingTable{std::nullopt};
    std::optional<SizeType32> mPromptVocabSize{std::nullopt};
    std::optional<std::shared_ptr<std::vector<std::vector<SizeType32>>>> mMultimodalHashes{std::nullopt};
    std::optional<std::shared_ptr<std::vector<SizeType32>>> mMultimodalPositions{std::nullopt};
    std::optional<std::shared_ptr<std::vector<SizeType32>>> mMultimodalLengths{std::nullopt};
    std::optional<TensorPtr> mMultimodalEmbedding{std::nullopt};
    std::optional<TensorPtr> mMropeRotaryCosSin{std::nullopt};
    std::optional<SizeType32> mMropePositionDeltas{std::nullopt};

    std::optional<LoraTaskIdType> mLoraTaskId{std::nullopt};
    std::optional<TensorPtr> mLoraWeights{std::nullopt};
    std::optional<TensorPtr> mLoraConfig{std::nullopt};

    std::optional<executor::LookaheadDecodingConfig> mLookaheadConfig{std::nullopt};

    std::optional<executor::KvCacheRetentionConfig> mKvCacheRetentionConfig{std::nullopt};

    // Paged-KV-Cache must be enabled while enabling Chunked-Context.
    // The size of the context chunk must be multiple of the KV-Cache block size except the last one.
    // Value `0` means Chunked-Context is disabled.
    SizeType32 mContextChunkSize{0};
    SizeType32 mContextCurrentPositionTarget{0};
    SizeType32 mContextCurrentPositionDraft{0};

    std::vector<VecLogProbs> mLogProbs; // [beamSize, seqLen]
    VecLogProbs mCumLogProbs;           // [beamSize]
    std::shared_ptr<VecTokens> mDraftTokens{nullptr};
    std::optional<TensorPtr> mDraftLogits{std::nullopt};
    SizeType32 mNumTokensPerIteration{1};

    // whether to return the full beams on each iteration. True when doing streaming + beamsearch
    bool mReturnAllGeneratedTokens;
    // Save logits
    bool mReturnContextLogits;
    bool mReturnGenerationLogits;
    bool mReturnLogProbs;
    TensorPtr mContextLogitsHost;    // [mPromptLen, vocabSizePadded]
    TensorPtr mGenerationLogitsHost; // [beamSize, mMaxNewTokens, vocabSizePadded]
    std::vector<TensorPtr> mGenerationLogitsFragments;

    bool mExcludeInputFromOutput;

    // Encoder-only and Encoder-Decoder models
    // Encoder input tokens
    std::optional<std::shared_ptr<VecTokens>> mEncoderTokens{std::nullopt};

    bool mReturnEncoderOutput;

    // Encoder output, used to compute cross attention KV-Cache.
    TensorPtr mEncoderOutput;       // [numTokens, hidden_size]
    TensorPtr mEncoderHiddenStates; // [numTokens, hiddenSize] for for Pipeline-Parallelism
    TensorPtr mEncoderOutputHost;   // [mEncoderOutputLength, encoderHiddenSize]

    SizeType32 mDecodingIter{0};

    executor::PriorityType mPriority;

    std::vector<executor::FinishReason> mFinishReasons;

    // Input features of encoder for multimodal models.
    std::optional<TensorPtr> mEncoderInputFeatures{std::nullopt};

    // Setting buffer sizes correctly for models like Whisper,
    // which encoder output shape cannot be inferred from encoder input shape due to downsampling.
    std::optional<SizeType32> mEncoderOutputLength{std::nullopt};

    // Input cross attention mask.
    std::optional<TensorPtr> mCrossAttentionMask{std::nullopt};

    LlmRequestType mLlmRequestType;

    std::optional<executor::ContextPhaseParams> mContextPhaseParams{std::nullopt};

    std::shared_ptr<ContextProgress> mContextProgress{nullptr};

    std::optional<std::shared_ptr<VecTokenExtraIds>> mInputTokenExtraIds{std::nullopt};

    BeamUniqueTokens mUniqueTokens;

    // TODO: add real extra id for encoder tokens.
    std::optional<std::shared_ptr<VecUniqueTokens>> mEncoderUniqueTokens{std::nullopt};

    SizeType32 mNumReturnSequences{1};

    // Config for Eagle speculative decoding.
    std::optional<executor::EagleConfig> mEagleConfig{std::nullopt};

    SizeType32 mSequenceIndex{0};

    std::vector<RequestPtr> mChildRequests;

    RequestIdType mParentRequestId;

    // Indicators whether each sibling completes generation.
    std::shared_ptr<std::vector<bool>> mSequenceFinalVec;

    std::optional<TensorPtr> mSkipCrossAttnBlocks{std::nullopt};

    // Performance metrics.
    bool mReturnPerfMetrics{false};
    executor::RequestPerfMetrics mPerfMetrics;

    // Guided decoding params.
    std::optional<executor::GuidedDecodingParams> mGuidedDecodingParams{std::nullopt};

    std::optional<SizeType32> mLanguageAdapterUid{std::nullopt};

    // Timepoint at which the request started. Used for tracking the timeout
    std::chrono::steady_clock::time_point mStartTime;
    // Time in milliseconds after which the model is finished with a `timeout` finishReason.
    std::optional<MillisecondsType> mAllottedTimeMs{std::nullopt};

    // Tensors containing the additional context output.
    TensorMap mAdditionalContextOutputTensors;

    // Tensors containing the additional generation output.
    TensorMap mAdditionalGenerationOutputTensors;

    bool mIsDummyRequest{false};

    bool mUseDraftModel{false};

    // Cache salt id for each request.
    std::optional<CacheSaltIDType> mCacheSaltID{std::nullopt};

private:
    void initialize(
        VecTokens const& inputTokens, bool outputLogProbs, std::optional<TimePoint> arrivalTime = std::nullopt)
    {
        if (mLlmRequestType == LlmRequestType::LLMREQUEST_TYPE_GENERATION_ONLY)
        {
            mState = LlmRequestState::kDISAGG_GENERATION_INIT;
        }

        // Scatter the input tokens to other beam
        mTokens = BeamTokens(mSamplingConfig.beamWidth, inputTokens);
        mLastTokens = VecTokens(mSamplingConfig.beamWidth, inputTokens.back());

        // Init mUniqueTokens
        VecUniqueTokens uniqueTokens{inputTokens.size()};
        if (mInputTokenExtraIds.has_value() && mInputTokenExtraIds.value())
        {
            if (mInputTokenExtraIds.value()->size() != inputTokens.size())
            {
                TLLM_THROW("inputTokenExtraIds vector size (%lu) must be the same as input token vector size (%lu).",
                    mInputTokenExtraIds.value()->size(), inputTokens.size());
            }
            std::transform(inputTokens.cbegin(), inputTokens.cend(), mInputTokenExtraIds.value()->cbegin(),
                uniqueTokens.begin(),
                [](auto const inputToken, auto const tokenExtraId) {
                    return UniqueToken{inputToken, tokenExtraId};
                });
        }
        else
        {
            // Default extra id is 0
            std::transform(inputTokens.cbegin(), inputTokens.cend(), uniqueTokens.begin(),
                [](auto const inputToken) {
                    return UniqueToken{inputToken, 0};
                });
        }
        mUniqueTokens = BeamUniqueTokens(mSamplingConfig.beamWidth, uniqueTokens);

        // Init mEncoderUniqueTokens
        // TODO: use real extra id instead of default zero value
        if (mEncoderTokens.has_value() && mEncoderTokens.value())
        {
            auto const& encoderTokens = *(mEncoderTokens.value());
            auto encoderUniqueTokens = std::make_shared<VecUniqueTokens>(encoderTokens.size());
            std::transform(encoderTokens.cbegin(), encoderTokens.cend(), encoderUniqueTokens->begin(),
                [](auto const encoderToken) {
                    return UniqueToken{encoderToken, 0};
                });
            mEncoderUniqueTokens = encoderUniqueTokens;
        }

        if ((mPromptEmbeddingTable.has_value() && !mPromptVocabSize.has_value())
            || (!mPromptEmbeddingTable.has_value() && mPromptVocabSize.has_value()))
        {
            std::string errStr
                = "Prompt embedding table and prompt vocab size tensors must both be provided for requests with "
                  "prompt "
                  "tuning enabled.";
            TLLM_THROW(errStr);
        }

        if (mDraftLogits.has_value() && mDraftTokens->empty())
        {
            TLLM_THROW("Draft tokens must be specified when draft logits are given.");
        }

        setReturnLogProbs(outputLogProbs);

        // Handling the backward compatibility of numReturnSequences.
        if (mNumReturnSequences > 1)
        {
            if (!mSamplingConfig.numReturnSequences)
            {
                TLLM_LOG_WARNING(
                    "In the Executor class, mNumReturnSequences is deprecated. Please set numReturnSequences in "
                    "SamplingConfig directly.");
            }
            else if (mSamplingConfig.numReturnSequences
                && mSamplingConfig.numReturnSequences.value() != mNumReturnSequences)
            {
                TLLM_THROW(
                    "In the Executor class, both mSamplingConfig.numReturnSequences (%d) and mNumReturnSequences (%d) "
                    "are provided but unmatched. Please use numReturnSequences in SamplingConfig directly.",
                    mSamplingConfig.numReturnSequences.value(), mNumReturnSequences);
            }
            mSamplingConfig.numReturnSequences = mNumReturnSequences;
        }

        if (!isChild())
        {
            // Initialize result states unless it is a child and a child request should share parent's one.
            mSequenceFinalVec = std::make_shared<std::vector<bool>>(getNumSubRequests(), false);
        }

        if (mReturnPerfMetrics)
        {
            // arrivalTime is assumed to be recorded at the rank 0, so no need to convert it to global clock
            mPerfMetrics.timingMetrics.arrivalTime = arrivalTime.value_or(getSteadyClockNow());
        }
        mStartTime = std::chrono::steady_clock::now();
    }

    TensorPtr createListTensor(std::list<VecTokens> const& wordsList)
    {
        std::vector<SizeType32> offsets;
        VecTokens words;
        SizeType32 offsetCnt = 0;
        for (auto const& tokens : wordsList)
        {
            offsetCnt += tokens.size();
            offsets.push_back(offsetCnt);
            words.insert(words.end(), tokens.begin(), tokens.end());
        }
        offsets.resize(words.size(), -1);

        auto const numWords = static_cast<SizeType32>(words.size());
        auto const shape = runtime::ITensor::makeShape({2, numWords});
        auto tensor = runtime::BufferManager::pinnedPool(shape, nvinfer1::DataType::kINT32);
        auto* data = runtime::bufferCast<int32_t>(*tensor);
        std::memcpy(data, words.data(), numWords * sizeof(int32_t));
        std::memcpy(data + numWords, offsets.data(), numWords * sizeof(int32_t));

        // Add leading dim of 1
        tensor->unsqueeze(0);

        return tensor;
    }

    TimePoint maybeToGlobalSteadyClock(TimePoint const& time_point) const
    {
        if (mGlobalSteadyClockOffset.has_value())
        {
            return time_point + *mGlobalSteadyClockOffset;
        }
        else
        {
            return time_point;
        }
    }
};

class LlmRequest : public GenericLlmRequest<runtime::ITensor::SharedPtr>
{
    friend class LlmRequestBindings;

public:
    using Base = GenericLlmRequest<runtime::ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;
    using SizeType32 = Base::SizeType32;
    using TokenIdType = Base::TokenIdType;
    using RequestIdType = Base::RequestIdType;
    using VecLogProbs = Base::VecLogProbs;
    using BeamTokens = Base::BeamTokens;
    using VecTokens = Base::VecTokens;
    using LoraTaskIdType = Base::LoraTaskIdType;
    using TokenExtraIdType = Base::TokenExtraIdType;
    using VecTokenExtraIds = Base::VecTokenExtraIds;

    // inherit constructors
    using Base::Base;

    LlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, std::vector<TokenIdType> inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::vector<SizeType32>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<std::vector<std::vector<SizeType32>>> multimodalHashes = std::nullopt,
        std::optional<std::vector<SizeType32>> multimodalPositions = std::nullopt,
        std::optional<std::vector<SizeType32>> multimodalLengths = std::nullopt,
        std::optional<TensorPtr> multimodalEmbedding = std::nullopt,
        std::optional<TensorPtr> mropeRotaryCosSin = std::nullopt,
        std::optional<SizeType32> mropePositionDeltas = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<executor::KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<VecTokens> draftTokens = std::nullopt, std::optional<TensorPtr> draftLogits = std::nullopt,
        bool excludeInputFromOutput = false, std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false, std::optional<VecTokens> encoderInputTokens = std::nullopt,
        bool returnEncoderOutput = false, std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority,
        std::optional<TensorPtr> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<TensorPtr> crossAttentionMask = std::nullopt,
        LlmRequestType llmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<VecTokenExtraIds> inputTokenExtraIds = std::nullopt, SizeType32 numReturnSequences = 1,
        std::optional<executor::EagleConfig> eagleConfig = std::nullopt,
        std::optional<TensorPtr> skipCrossAttnBlocks = std::nullopt, bool returnPerfMetrics = false,
        std::optional<executor::GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<SizeType32> languageAdapterUid = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt,
        std::optional<executor::ContextPhaseParams> const& contextPhaseParams = std::nullopt,
        std::optional<CacheSaltIDType> cacheSaltID = std::nullopt, std::optional<TimePoint> arrivalTime = std::nullopt)
        : Base(requestId, maxNewTokens, std::make_shared<std::vector<TokenIdType>>(std::move(inputTokens)),
            samplingConfig, isStreaming, endId, padId, std::move(embeddingBias), std::move(badWordsList),
            std::move(stopWordsList),
            positionIds.has_value() ? std::make_shared<std::vector<SizeType32>>(std::move(positionIds.value()))
                                    : std::optional<std::shared_ptr<std::vector<SizeType32>>>(std::nullopt),
            std::move(promptEmbeddingTable), promptVocabSize,
            multimodalHashes.has_value()
                ? std::make_shared<std::vector<std::vector<SizeType32>>>(std::move(multimodalHashes.value()))
                : std::optional<std::shared_ptr<std::vector<std::vector<SizeType32>>>>(std::nullopt),
            multimodalPositions.has_value()
                ? std::make_shared<std::vector<SizeType32>>(std::move(multimodalPositions.value()))
                : std::optional<std::shared_ptr<std::vector<SizeType32>>>(std::nullopt),
            multimodalLengths.has_value()
                ? std::make_shared<std::vector<SizeType32>>(std::move(multimodalLengths.value()))
                : std::optional<std::shared_ptr<std::vector<SizeType32>>>(std::nullopt),
            std::move(multimodalEmbedding), std::move(mropeRotaryCosSin), mropePositionDeltas, loraTaskId,
            std::move(loraWeights), std::move(loraConfig), lookaheadConfig, std::move(kvCacheRetentionConfig),
            returnLogProbs, returnContextLogits, returnGenerationLogits,
            draftTokens.has_value() ? std::make_shared<VecTokens>(std::move(draftTokens.value()))
                                    : std::make_shared<VecTokens>(),
            std::move(draftLogits), excludeInputFromOutput, std::move(logitsPostProcessor),
            applyLogitsPostProcessorBatched,
            encoderInputTokens ? std::make_optional(std::make_shared<VecTokens>(std::move(*encoderInputTokens)))
                               : std::optional<std::shared_ptr<VecTokens>>(std::nullopt),
            returnEncoderOutput, clientId, priority, std::move(encoderInputFeatures), encoderOutputLength,
            std::move(crossAttentionMask), llmRequestType,
            inputTokenExtraIds ? std::make_optional(std::make_shared<VecTokenExtraIds>(std::move(*inputTokenExtraIds)))
                               : std::optional<std::shared_ptr<VecTokenExtraIds>>(std::nullopt),
            numReturnSequences, std::move(eagleConfig), skipCrossAttnBlocks, returnPerfMetrics,
            std::move(guidedDecodingParams), languageAdapterUid, allottedTimeMs, contextPhaseParams, cacheSaltID,
            arrivalTime)
    {
    }

    LlmRequest(RequestIdType requestId, executor::Request const& request,
        std::optional<Base::LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false)
        : Base(requestId, request)
    {
        mLogitsPostProcessor = std::move(logitsPostProcessor);
        mApplyLogitsPostProcessorBatched = applyLogitsPostProcessorBatched;
        mLookaheadConfig = request.getLookaheadConfig();
        mKvCacheRetentionConfig = request.getKvCacheRetentionConfig();
    }

    LlmRequest(LlmRequest&& request) = default;
    LlmRequest(LlmRequest const& request) = default;

    /// @brief  Create a Response from the current state of the request
    /// @details Note that there is some dependency on the order of operations in this method. Modify with care!
    /// @return An optional Response
    std::optional<executor::Response> createResponse(bool useFastLogits = false, int32_t mpiWorldRank = 0);

    std::optional<executor::Result> createResult(bool useFastLogits = false, int32_t mpiWorldRank = 0);

    void createSerializedResult(
        std::vector<char>& serializedResult, bool& isFinal, bool useFastLogits = false, int32_t mpiWorldRank = 0);

    /// @brief Check if the (user-provided) tokens fall within the vocabulary range.
    /// @details Currently only supports invocation before context phase is completed.
    /// @return True if tokens are within range.
    bool checkTokenIdRange(SizeType32 vocabSize);

    void validate(SizeType32 maxInputLen, SizeType32 maxSequenceLen, SizeType32 maxDraftLen, SizeType32 vocabSizePadded,
        std::optional<SizeType32> maxEncoderInputLen = std::nullopt, bool enableKVCacheReuse = false);

    std::shared_ptr<LlmRequest> createChildRequest(RequestIdType requestId);

    void movePromptEmbeddingTableToGpu(runtime::BufferManager const& manager);

    void moveLoraWeightsToGpu(runtime::BufferManager const& manager);

    // Remove LoRA weights and LoRA config tensors
    void removeLoraTensors();
};

} // namespace tensorrt_llm::batch_manager
