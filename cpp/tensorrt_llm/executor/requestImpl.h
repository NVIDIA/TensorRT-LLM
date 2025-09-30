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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

#include <utility>
#include <vector>

namespace tensorrt_llm::executor
{
class Request::Impl
{

public:
    Impl(VecTokens inputTokenIds, SizeType32 maxNewTokens, bool streaming, SamplingConfig const& samplingConfig,
        OutputConfig outputConfig, std::optional<TokenIdType> const& endId, std::optional<TokenIdType> const& padId,
        std::optional<std::vector<SizeType32>> positionIds, std::optional<std::list<VecTokens>> badWords,
        std::optional<std::list<VecTokens>> stopWords, std::optional<Tensor> embeddingBias,
        std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig,
        std::optional<PromptTuningConfig> pTuningConfig, std::optional<MultimodalInput> multimodalInput,
        std::optional<Tensor> multimodalEmbedding, std::optional<MropeConfig> mRopeConfig,
        std::optional<LoraConfig> loraConfig, std::optional<LookaheadDecodingConfig> lookaheadConfig,
        std::optional<KvCacheRetentionConfig> kvCacheRetentionConfig,
        std::optional<std::string> logitsPostProcessorName, std::optional<LogitsPostProcessor> logitsPostProcessor,
        std::optional<VecTokens> encoderInputTokenIds, std::optional<IdType> clientId, bool returnAllGeneratedTokens,
        PriorityType priority, RequestType type, std::optional<ContextPhaseParams> contextPhaseParams,
        std::optional<Tensor> encoderInputFeatures, std::optional<SizeType32> encoderOutputLength,
        std::optional<Tensor> crossAttentionMask, SizeType32 numReturnSequences, std::optional<EagleConfig> eagleConfig,
        std::optional<Tensor> skipCrossAttnBlocks, std::optional<GuidedDecodingParams> guidedDecodingParams,
        std::optional<SizeType32> languageAdapterUid, std::optional<MillisecondsType> allottedTimeMs,
        std::optional<CacheSaltIDType> cacheSaltID)
        : mInputTokenIds(std::move(inputTokenIds))
        , mMaxNewTokens(maxNewTokens)
        , mStreaming(streaming)
        , mSamplingConfig(samplingConfig)
        , mOutputConfig(std::move(outputConfig))
        , mEndId(endId)
        , mPadId(padId)
        , mPositionIds(std::move(positionIds))
        , mBadWords(std::move(badWords))
        , mStopWords(std::move(stopWords))
        , mEmbeddingBias(checkEmbeddingBias(std::move(embeddingBias)))
        , mExternalDraftTokensConfig(std::move(externalDraftTokensConfig))
        , mPTuningConfig(std::move(pTuningConfig))
        , mMultimodalInput(std::move(multimodalInput))
        , mMultimodalEmbedding(std::move(multimodalEmbedding))
        , mMropeConfig(std::move(mRopeConfig))
        , mLoraConfig(std::move(loraConfig))
        , mLookaheadConfig(lookaheadConfig)
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mLogitsPostProcessorName(std::move(logitsPostProcessorName))
        , mLogitsPostProcessor(std::move(logitsPostProcessor))
        , mEncoderInputTokenIds(std::move(encoderInputTokenIds))
        , mClientId(clientId)
        , mReturnAllGeneratedTokens(returnAllGeneratedTokens)
        , mPriority(priority)
        , mType(type)
        , mContextPhaseParams(std::move(contextPhaseParams))
        , mEncoderInputFeatures(std::move(encoderInputFeatures))
        , mEncoderOutputLength(encoderOutputLength)
        , mCrossAttentionMask(std::move(crossAttentionMask))
        , mNumReturnSequences(numReturnSequences)
        , mEagleConfig(std::move(eagleConfig))
        , mSkipCrossAttnBlocks(std::move(skipCrossAttnBlocks))
        , mGuidedDecodingParams(std::move(guidedDecodingParams))
        , mLanguageAdapterUid(languageAdapterUid)
        , mAllottedTimeMs(allottedTimeMs)
        , mCacheSaltID(cacheSaltID)
    {
        validate();
    }

    ~Impl() = default;
    Impl(Impl const& other) = default;
    Impl(Impl&& other) noexcept = default;
    Impl& operator=(Impl const& other) = default;
    Impl& operator=(Impl&& other) noexcept = default;

    void serialize(std::ostream& ostream) const
    {
        // Dynamic logitsPostProcessor is only supported with replicate=false or no tensor parallelism.
        TLLM_CHECK_WITH_INFO(!mLogitsPostProcessor.has_value(),
            "Serialization of Request with logitsPostProcessor is currently not supported.");
        visitMembers([&ostream](auto const& member) { serialize_utils::serialize(member, ostream); });
    }

    [[nodiscard]] size_t serializedSize() const
    {
        // Dynamic logitsPostProcessor is only supported with replicate=false or no tensor parallelism.
        TLLM_CHECK_WITH_INFO(!mLogitsPostProcessor.has_value(),
            "Serialization of Request with logitsPostProcessor is currently not supported.");
        size_t totalSize = 0;
        visitMembers([&totalSize](auto const& member) { totalSize += serialize_utils::serializedSize(member); });
        return totalSize;
    }

    [[nodiscard]] VecTokens getInputTokenIds() const
    {
        return mInputTokenIds;
    }

    [[nodiscard]] SizeType32 getMaxNewTokens() const
    {
        return mMaxNewTokens;
    }

    [[nodiscard]] bool getStreaming() const
    {
        return mStreaming;
    }

    [[nodiscard]] SamplingConfig getSamplingConfig() const
    {
        return mSamplingConfig;
    }

    [[nodiscard]] OutputConfig getOutputConfig() const
    {
        return mOutputConfig;
    }

    [[nodiscard]] std::optional<SizeType32> getEndId() const
    {
        return mEndId;
    }

    [[nodiscard]] std::optional<SizeType32> getPadId() const
    {
        return mPadId;
    }

    [[nodiscard]] std::optional<std::vector<SizeType32>> getPositionIds() const
    {
        return mPositionIds;
    }

    [[nodiscard]] std::optional<std::list<VecTokens>> getBadWords() const
    {
        return mBadWords;
    }

    [[nodiscard]] std::optional<std::list<VecTokens>> getStopWords() const
    {
        return mStopWords;
    }

    [[nodiscard]] std::optional<Tensor> getEmbeddingBias() const
    {
        return mEmbeddingBias;
    }

    [[nodiscard]] std::optional<ExternalDraftTokensConfig> getExternalDraftTokensConfig() const
    {
        return mExternalDraftTokensConfig;
    }

    [[nodiscard]] std::optional<PromptTuningConfig> getPromptTuningConfig() const
    {
        return mPTuningConfig;
    }

    [[nodiscard]] std::optional<Tensor> getMultimodalEmbedding() const
    {
        return mMultimodalEmbedding;
    }

    [[nodiscard]] std::optional<MultimodalInput> getMultimodalInput() const
    {
        return mMultimodalInput;
    }

    [[nodiscard]] std::optional<MropeConfig> getMropeConfig() const
    {
        return mMropeConfig;
    }

    [[nodiscard]] std::optional<LoraConfig> getLoraConfig() const
    {
        return mLoraConfig;
    }

    [[nodiscard]] std::optional<LookaheadDecodingConfig> getLookaheadConfig() const
    {
        return mLookaheadConfig;
    }

    [[nodiscard]] std::optional<KvCacheRetentionConfig> getKvCacheRetentionConfig() const
    {
        return mKvCacheRetentionConfig;
    }

    [[nodiscard]] std::optional<std::string> getLogitsPostProcessorName() const
    {
        return mLogitsPostProcessorName;
    }

    std::optional<LogitsPostProcessor> getLogitsPostProcessor() const
    {
        return mLogitsPostProcessor;
    }

    [[nodiscard]] std::optional<VecTokens> getEncoderInputTokenIds() const
    {
        return mEncoderInputTokenIds;
    }

    [[nodiscard]] std::optional<IdType> getClientId() const
    {
        return mClientId;
    }

    [[nodiscard]] PriorityType getPriority() const
    {
        return mPriority;
    }

    [[nodiscard]] std::optional<MillisecondsType> getAllottedTimeMs() const
    {
        return mAllottedTimeMs;
    }

    [[nodiscard]] bool getReturnAllGeneratedTokens() const
    {
        return mReturnAllGeneratedTokens;
    }

    [[nodiscard]] RequestType getRequestType() const
    {
        return mType;
    }

    [[nodiscard]] std::optional<ContextPhaseParams> const& getContextPhaseParams() const
    {
        return mContextPhaseParams;
    }

    [[nodiscard]] std::optional<Tensor> getEncoderInputFeatures() const
    {
        return mEncoderInputFeatures;
    }

    [[nodiscard]] std::optional<Tensor> getCrossAttentionMask() const
    {
        return mCrossAttentionMask;
    }

    [[nodiscard]] std::optional<SizeType32> getEncoderOutputLength() const
    {
        return mEncoderOutputLength;
    }

    [[nodiscard]] std::optional<SizeType32> getNumReturnSequences() const
    {
        TLLM_LOG_WARNING(
            "The 'getNumReturnSequences' method in the Request class is deprecated and will be removed in a future "
            "release. Please use 'getNumReturnSequences' directly from the 'SamplingConfig' object.");
        return mSamplingConfig.getNumReturnSequences();
    }

    [[nodiscard]] std::optional<EagleConfig> getEagleConfig() const
    {
        return mEagleConfig;
    }

    [[nodiscard]] std::optional<Tensor> getSkipCrossAttnBlocks() const
    {
        return mSkipCrossAttnBlocks;
    }

    [[nodiscard]] std::optional<GuidedDecodingParams> getGuidedDecodingParams() const
    {
        return mGuidedDecodingParams;
    }

    [[nodiscard]] std::optional<SizeType32> getLanguageAdapterUid() const
    {
        return mLanguageAdapterUid;
    }

    [[nodiscard]] std::optional<CacheSaltIDType> getCacheSaltID() const
    {
        return mCacheSaltID;
    }

    void setStreaming(bool streaming)
    {
        mStreaming = streaming;
    }

    void setSamplingConfig(SamplingConfig const& config)
    {
        mSamplingConfig = config;
    }

    void setOutputConfig(OutputConfig const& outputConfig)
    {
        mOutputConfig = outputConfig;
    }

    void setEndId(SizeType32 endId)
    {
        mEndId = endId;
    }

    void setPadId(SizeType32 padId)
    {
        mPadId = padId;
    }

    void setPositionIds(std::vector<SizeType32> const& positionIds)
    {
        mPositionIds = positionIds;
    }

    void setBadWords(std::list<VecTokens> const& badWords)
    {
        mBadWords = badWords;
    }

    void setStopWords(std::list<VecTokens> const& stopWords)
    {
        mStopWords = stopWords;
    }

    void setEmbeddingBias(Tensor const& embeddingBias)
    {
        mEmbeddingBias = checkEmbeddingBias(embeddingBias);
    }

    void setExternalDraftTokensConfig(ExternalDraftTokensConfig const& externalDraftTokensConfig)
    {
        mExternalDraftTokensConfig = externalDraftTokensConfig;
    }

    void setPromptTuningConfig(PromptTuningConfig const& pTuningConfig)
    {
        mPTuningConfig = pTuningConfig;
    }

    void setMultimodalEmbedding(Tensor const& multimodalEmbedding)
    {
        mMultimodalEmbedding = multimodalEmbedding;
    }

    void setMultimodalInput(MultimodalInput const& multimodalInput)
    {
        mMultimodalInput = multimodalInput;
    }

    void setMropeConfig(MropeConfig const& mRopeConfig)
    {
        mMropeConfig = mRopeConfig;
    }

    void setLoraConfig(LoraConfig const& loraConfig)
    {
        mLoraConfig = loraConfig;
    }

    void setLookaheadConfig(LookaheadDecodingConfig const& lookaheadConfig)
    {
        mLookaheadConfig = lookaheadConfig;
    }

    void setKvCacheRetentionConfig(KvCacheRetentionConfig const& kvCacheRetentionConfig)
    {
        mKvCacheRetentionConfig = kvCacheRetentionConfig;
    }

    void setLogitsPostProcessorName(std::string const& logitsPostProcessorName)
    {
        mLogitsPostProcessorName = logitsPostProcessorName;
    }

    void setLogitsPostProcessor(std::optional<LogitsPostProcessor> const& logitsPostProcessor)
    {
        mLogitsPostProcessor = logitsPostProcessor;
    }

    void setEncoderInputTokenIds(VecTokens const& encoderInputTokenIds)
    {
        mEncoderInputTokenIds = encoderInputTokenIds;
    }

    void setClientId(IdType clientId)
    {
        mClientId = clientId;
    }

    void setPriority(PriorityType priority)
    {
        mPriority = priority;
    }

    void setReturnAllGeneratedTokens(bool returnAllGeneratedTokens)
    {
        mReturnAllGeneratedTokens = returnAllGeneratedTokens;
    }

    void setRequestType(RequestType requestType)
    {
        mType = requestType;
    }

    void setContextPhaseParams(ContextPhaseParams contextPhaseParams)
    {
        mContextPhaseParams = std::move(contextPhaseParams);
    }

    void setEncoderInputFeatures(Tensor encoderInputFeatures)
    {
        mEncoderInputFeatures = encoderInputFeatures;
    }

    void setCrossAttentionMask(Tensor crossAttentionMask)
    {
        mCrossAttentionMask = crossAttentionMask;
    }

    void setEncoderOutputLength(SizeType32 encoderOutputLength)
    {
        mEncoderOutputLength = encoderOutputLength;
    }

    void setNumReturnSequences(SizeType32 numReturnSequences)
    {
        TLLM_LOG_WARNING(
            "The 'setNumReturnSequences' method in the Request class is deprecated and will be removed in a future "
            "release. Please use 'setNumReturnSequences' directly on the 'SamplingConfig' object.");
        mNumReturnSequences = numReturnSequences;
        mSamplingConfig.setNumReturnSequences(numReturnSequences);
    }

    void setEagleConfig(std::optional<EagleConfig> eagleConfig)
    {
        mEagleConfig = std::move(eagleConfig);
    }

    void setSkipCrossAttnBlocks(Tensor skipCrossAttnBlocks)
    {
        mSkipCrossAttnBlocks = skipCrossAttnBlocks;
    }

    void setGuidedDecodingParams(GuidedDecodingParams const& guidedDecodingParams)
    {
        mGuidedDecodingParams = guidedDecodingParams;
    }

    void setAllottedTimeMs(MillisecondsType allottedTimeMs)
    {
        mAllottedTimeMs = allottedTimeMs;
    }

    void setLanguageAdapterUid(SizeType32 languageAdapterUid)
    {
        mLanguageAdapterUid = languageAdapterUid;
    }

    void setCacheSaltID(CacheSaltIDType cacheSaltID)
    {
        mCacheSaltID = cacheSaltID;
    }

private:
    void validate()
    {
        TLLM_CHECK(!mInputTokenIds.empty());
        TLLM_CHECK(mMaxNewTokens > 0);

        // Show warning message unless mNumReturnSequences is the default value.
        if (mNumReturnSequences > 1)
        {
            TLLM_LOG_WARNING(
                "The 'numReturnSequences' in the Request class is deprecated and will be removed in a future release. "
                "Please set the number of return sequences directly in 'SamplingConfig'.");
            mSamplingConfig.setNumReturnSequences(mNumReturnSequences);
        }

        if (mLogitsPostProcessorName.has_value() && mLogitsPostProcessor.has_value())
        {
            TLLM_THROW("Only one of 'logitsPostProcessorName' and 'logitsPostProcessor' can be specified.");
        }

        if (mGuidedDecodingParams.has_value() && mSamplingConfig.getBeamWidth() > 1)
        {
            TLLM_THROW("Guided decoding does not support with beam search.");
        }
    }

    static std::optional<Tensor> checkEmbeddingBias(std::optional<Tensor> bias)
    {
        if (bias)
        {
            TLLM_CHECK(bias.value().getShape().size() == 1);
        }
        return bias;
    }

    template <typename Lambda>
    void visitMembers(Lambda const& lambda) const
    {
        lambda(mInputTokenIds);
        lambda(mMaxNewTokens);
        lambda(mStreaming);
        lambda(mSamplingConfig);
        lambda(mOutputConfig);
        lambda(mEndId);
        lambda(mPadId);
        lambda(mPositionIds);
        lambda(mBadWords);
        lambda(mStopWords);
        lambda(mEmbeddingBias);
        lambda(mExternalDraftTokensConfig);
        lambda(mPTuningConfig);
        lambda(mMultimodalInput);
        lambda(mMultimodalEmbedding);
        lambda(mMropeConfig);
        lambda(mLoraConfig);
        lambda(mLookaheadConfig);
        lambda(mKvCacheRetentionConfig);
        lambda(mLogitsPostProcessorName);
        lambda(mEncoderInputTokenIds);
        lambda(mClientId);
        lambda(mReturnAllGeneratedTokens);
        lambda(mPriority);
        lambda(mType);
        lambda(mContextPhaseParams);
        lambda(mEncoderInputFeatures);
        lambda(mEncoderOutputLength);
        lambda(mCrossAttentionMask);
        lambda(mNumReturnSequences);
        lambda(mEagleConfig);
        lambda(mSkipCrossAttnBlocks);
        lambda(mGuidedDecodingParams);
        lambda(mLanguageAdapterUid);
        lambda(mAllottedTimeMs ? std::make_optional(mAllottedTimeMs->count()) : std::nullopt);
        lambda(mCacheSaltID);
    }

    VecTokens mInputTokenIds;
    SizeType32 mMaxNewTokens;
    bool mStreaming;
    SamplingConfig mSamplingConfig;
    OutputConfig mOutputConfig;
    std::optional<SizeType32> mEndId;
    std::optional<SizeType32> mPadId;
    std::optional<std::vector<SizeType32>> mPositionIds;
    std::optional<std::list<VecTokens>> mBadWords;
    std::optional<std::list<VecTokens>> mStopWords;
    std::optional<Tensor> mEmbeddingBias;
    std::optional<ExternalDraftTokensConfig> mExternalDraftTokensConfig;
    std::optional<PromptTuningConfig> mPTuningConfig;
    std::optional<MultimodalInput> mMultimodalInput;
    std::optional<Tensor> mMultimodalEmbedding;
    std::optional<MropeConfig> mMropeConfig;
    std::optional<LoraConfig> mLoraConfig;
    std::optional<LookaheadDecodingConfig> mLookaheadConfig;
    std::optional<KvCacheRetentionConfig> mKvCacheRetentionConfig;
    std::optional<std::string> mLogitsPostProcessorName;
    std::optional<LogitsPostProcessor> mLogitsPostProcessor;
    std::optional<VecTokens> mEncoderInputTokenIds;
    std::optional<IdType> mClientId;
    bool mReturnAllGeneratedTokens;
    PriorityType mPriority;
    RequestType mType;
    std::optional<ContextPhaseParams> mContextPhaseParams;
    std::optional<Tensor> mEncoderInputFeatures;
    std::optional<SizeType32> mEncoderOutputLength;
    std::optional<Tensor> mCrossAttentionMask;
    SizeType32 mNumReturnSequences;
    std::optional<EagleConfig> mEagleConfig;
    std::optional<Tensor> mSkipCrossAttnBlocks;
    std::optional<GuidedDecodingParams> mGuidedDecodingParams;
    std::optional<SizeType32> mLanguageAdapterUid;
    std::optional<MillisecondsType> mAllottedTimeMs;
    std::optional<CacheSaltIDType> mCacheSaltID;
};

} // namespace tensorrt_llm::executor
