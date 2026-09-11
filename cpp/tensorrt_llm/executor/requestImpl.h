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
    //! Maximum allowed length of a cache salt string. Cache salts are copied into every BlockKey and emitted
    //! with KV cache events, so unbounded strings would inflate memory and serialization cost proportional to
    //! the number of blocks.
    static constexpr std::size_t kMaxCacheSaltLength{256};

    static std::optional<std::string> validateCacheSalt(std::optional<std::string> cacheSalt)
    {
        if (cacheSalt.has_value() && cacheSalt->size() > kMaxCacheSaltLength)
        {
            TLLM_THROW("cacheSalt length (%zu) exceeds the maximum supported length (%zu).", cacheSalt->size(),
                kMaxCacheSaltLength);
        }
        return cacheSalt;
    }

    Impl(VecTokens inputTokenIds, SizeType32 maxNewTokens, bool streaming, SamplingConfig const& samplingConfig,
        OutputConfig outputConfig, std::optional<TokenIdType> const& endId,
        std::optional<std::vector<SizeType32>> positionIds, std::optional<std::list<VecTokens>> badWords,
        std::optional<std::list<VecTokens>> stopWords, std::optional<Tensor> embeddingBias,
        std::optional<PromptTuningConfig> pTuningConfig, std::optional<MultimodalInput> multimodalInput,
        std::optional<Tensor> multimodalEmbedding, std::optional<MropeConfig> mRopeConfig,
        std::optional<LoraConfig> loraConfig, std::optional<KvCacheRetentionConfig> kvCacheRetentionConfig,
        std::optional<VecTokens> encoderInputTokenIds, std::optional<IdType> clientId, bool returnAllGeneratedTokens,
        PriorityType priority, RequestType type, std::optional<ContextPhaseParams> contextPhaseParams,
        std::optional<Tensor> encoderInputFeatures, std::optional<SizeType32> encoderOutputLength,
        std::optional<GuidedDecodingParams> guidedDecodingParams, std::optional<SizeType32> languageAdapterUid,
        std::optional<MillisecondsType> allottedTimeMs, std::optional<IdType> disaggRequestId,
        std::optional<std::string> cacheSalt = std::nullopt)
        : mInputTokenIds(std::move(inputTokenIds))
        , mMaxNewTokens(maxNewTokens)
        , mStreaming(streaming)
        , mSamplingConfig(samplingConfig)
        , mOutputConfig(std::move(outputConfig))
        , mEndId(endId)
        , mPositionIds(std::move(positionIds))
        , mBadWords(std::move(badWords))
        , mStopWords(std::move(stopWords))
        , mEmbeddingBias(checkEmbeddingBias(std::move(embeddingBias)))
        , mPTuningConfig(std::move(pTuningConfig))
        , mMultimodalInput(std::move(multimodalInput))
        , mMultimodalEmbedding(std::move(multimodalEmbedding))
        , mMropeConfig(std::move(mRopeConfig))
        , mLoraConfig(std::move(loraConfig))
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mEncoderInputTokenIds(std::move(encoderInputTokenIds))
        , mClientId(clientId)
        , mReturnAllGeneratedTokens(returnAllGeneratedTokens)
        , mPriority(priority)
        , mType(type)
        , mContextPhaseParams(std::move(contextPhaseParams))
        , mEncoderInputFeatures(std::move(encoderInputFeatures))
        , mEncoderOutputLength(encoderOutputLength)
        , mGuidedDecodingParams(std::move(guidedDecodingParams))
        , mLanguageAdapterUid(languageAdapterUid)
        , mAllottedTimeMs(allottedTimeMs)
        , mCacheSalt(validateCacheSalt(std::move(cacheSalt)))
        , mDisaggRequestId(disaggRequestId)
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
        visitMembers([&ostream](auto const& member) { serialize_utils::serialize(member, ostream); });
    }

    [[nodiscard]] size_t serializedSize() const
    {
        size_t totalSize = 0;
        visitMembers([&totalSize](auto const& member) { totalSize += serialize_utils::serializedSize(member); });
        return totalSize;
    }

    [[nodiscard]] VecTokens getInputTokenIds() const
    {
        return mInputTokenIds;
    }

    [[nodiscard]] SizeType32 getNumInputTokens() const
    {
        return static_cast<SizeType32>(mInputTokenIds.size());
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

    [[nodiscard]] std::optional<KvCacheRetentionConfig> getKvCacheRetentionConfig() const
    {
        return mKvCacheRetentionConfig;
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

    [[nodiscard]] std::optional<SizeType32> getEncoderOutputLength() const
    {
        return mEncoderOutputLength;
    }

    [[nodiscard]] std::optional<GuidedDecodingParams> getGuidedDecodingParams() const
    {
        return mGuidedDecodingParams;
    }

    [[nodiscard]] std::optional<SizeType32> getLanguageAdapterUid() const
    {
        return mLanguageAdapterUid;
    }

    [[nodiscard]] std::optional<std::string> getCacheSalt() const
    {
        return mCacheSalt;
    }

    [[nodiscard]] std::optional<IdType> getDisaggRequestId() const
    {
        return mDisaggRequestId;
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

    void setKvCacheRetentionConfig(KvCacheRetentionConfig const& kvCacheRetentionConfig)
    {
        mKvCacheRetentionConfig = kvCacheRetentionConfig;
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

    void setEncoderOutputLength(SizeType32 encoderOutputLength)
    {
        mEncoderOutputLength = encoderOutputLength;
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

    void setCacheSalt(std::optional<std::string> cacheSalt)
    {
        mCacheSalt = validateCacheSalt(std::move(cacheSalt));
    }

    void setDisaggRequestId(IdType disaggRequestId)
    {
        mDisaggRequestId = disaggRequestId;
    }

private:
    void validate()
    {
        TLLM_CHECK(!mInputTokenIds.empty());
        TLLM_CHECK(mMaxNewTokens > 0);

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
        lambda(mPositionIds);
        lambda(mBadWords);
        lambda(mStopWords);
        lambda(mEmbeddingBias);
        lambda(mPTuningConfig);
        lambda(mMultimodalInput);
        lambda(mMultimodalEmbedding);
        lambda(mMropeConfig);
        lambda(mLoraConfig);
        lambda(mKvCacheRetentionConfig);
        lambda(mEncoderInputTokenIds);
        lambda(mClientId);
        lambda(mReturnAllGeneratedTokens);
        lambda(mPriority);
        lambda(mType);
        lambda(mContextPhaseParams);
        lambda(mEncoderInputFeatures);
        lambda(mEncoderOutputLength);
        lambda(mGuidedDecodingParams);
        lambda(mLanguageAdapterUid);
        lambda(mAllottedTimeMs ? std::make_optional(mAllottedTimeMs->count()) : std::nullopt);
        lambda(mDisaggRequestId);
        lambda(mCacheSalt);
    }

    VecTokens mInputTokenIds;
    SizeType32 mMaxNewTokens;
    bool mStreaming;
    SamplingConfig mSamplingConfig;
    OutputConfig mOutputConfig;
    std::optional<SizeType32> mEndId;
    std::optional<std::vector<SizeType32>> mPositionIds;
    std::optional<std::list<VecTokens>> mBadWords;
    std::optional<std::list<VecTokens>> mStopWords;
    std::optional<Tensor> mEmbeddingBias;
    std::optional<PromptTuningConfig> mPTuningConfig;
    std::optional<MultimodalInput> mMultimodalInput;
    std::optional<Tensor> mMultimodalEmbedding;
    std::optional<MropeConfig> mMropeConfig;
    std::optional<LoraConfig> mLoraConfig;
    std::optional<KvCacheRetentionConfig> mKvCacheRetentionConfig;
    std::optional<VecTokens> mEncoderInputTokenIds;
    std::optional<IdType> mClientId;
    bool mReturnAllGeneratedTokens;
    PriorityType mPriority;
    RequestType mType;
    std::optional<ContextPhaseParams> mContextPhaseParams;
    std::optional<Tensor> mEncoderInputFeatures;
    std::optional<SizeType32> mEncoderOutputLength;
    std::optional<GuidedDecodingParams> mGuidedDecodingParams;
    std::optional<SizeType32> mLanguageAdapterUid;
    std::optional<MillisecondsType> mAllottedTimeMs;
    std::optional<std::string> mCacheSalt;
    std::optional<IdType> mDisaggRequestId;
};

} // namespace tensorrt_llm::executor
