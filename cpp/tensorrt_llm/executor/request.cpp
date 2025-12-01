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

#include <utility>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestImpl.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{
Request::Request(VecTokens inputTokenIds, SizeType32 maxTokens, bool streaming, SamplingConfig const& samplingConfig,
    OutputConfig const& outputConfig, std::optional<SizeType32> const& endId, std::optional<SizeType32> const& padId,
    std::optional<std::vector<SizeType32>> positionIds, std::optional<std::list<VecTokens>> badWords,
    std::optional<std::list<VecTokens>> stopWords, std::optional<Tensor> embeddingBias,
    std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig, std::optional<PromptTuningConfig> pTuningConfig,
    std::optional<MultimodalInput> multimodalInput, std::optional<Tensor> multimodalEmbedding,
    std::optional<MropeConfig> mRopeConfig, std::optional<LoraConfig> loraConfig,
    std::optional<LookaheadDecodingConfig> lookaheadConfig,
    std::optional<KvCacheRetentionConfig> kvCacheRetentionConfig, std::optional<std::string> logitsPostProcessorName,
    std::optional<LogitsPostProcessor> logitslogitsPostProcessor, std::optional<VecTokens> encoderInputTokenIds,
    std::optional<IdType> clientId, bool returnAllGeneratedTokens, float priority, RequestType type,
    std::optional<ContextPhaseParams> contextPhaseParams, std::optional<Tensor> encoderInputFeatures,
    std::optional<SizeType32> encoderOutputLength, std::optional<Tensor> crossAttentionMask,
    SizeType32 numReturnSequences, std::optional<EagleConfig> eagleConfig, std::optional<Tensor> skipCrossAttnBlocks,
    std::optional<GuidedDecodingParams> guidedDecodingParams, std::optional<SizeType32> languageAdapterUid,
    std::optional<MillisecondsType> allottedTimeMs, std::optional<CacheSaltIDType> cacheSaltID)
    : mImpl(std::make_unique<Impl>(std::move(inputTokenIds), maxTokens, streaming, samplingConfig, outputConfig, endId,
        padId, std::move(positionIds), std::move(badWords), std::move(stopWords), std::move(embeddingBias),
        std::move(externalDraftTokensConfig), std::move(pTuningConfig), std::move(multimodalInput),
        std::move(multimodalEmbedding), std::move(mRopeConfig), std::move(loraConfig), lookaheadConfig,
        std::move(kvCacheRetentionConfig), std::move(logitsPostProcessorName), std::move(logitslogitsPostProcessor),
        std::move(encoderInputTokenIds), clientId, returnAllGeneratedTokens, priority, type,
        std::move(contextPhaseParams), std::move(encoderInputFeatures), encoderOutputLength, crossAttentionMask,
        numReturnSequences, eagleConfig, skipCrossAttnBlocks, std::move(guidedDecodingParams), languageAdapterUid,
        allottedTimeMs, cacheSaltID))
{
}

Request::~Request() = default;

Request::Request(Request const& other)
    : mImpl(std::make_unique<Impl>(*other.mImpl))
{
}

Request::Request(Request&& other) noexcept = default;

Request& Request::operator=(Request const& other)
{
    if (this != &other)
    {
        mImpl = std::make_unique<Impl>(*other.mImpl);
    }
    return *this;
}

Request& Request::operator=(Request&& other) noexcept = default;

VecTokens Request::getInputTokenIds() const
{
    return mImpl->getInputTokenIds();
}

SizeType32 Request::getMaxTokens() const
{
    return mImpl->getMaxNewTokens();
}

bool Request::getStreaming() const
{
    return mImpl->getStreaming();
}

SamplingConfig Request::getSamplingConfig() const
{
    return mImpl->getSamplingConfig();
}

OutputConfig Request::getOutputConfig() const
{
    return mImpl->getOutputConfig();
}

std::optional<SizeType32> Request::getEndId() const
{
    return mImpl->getEndId();
}

std::optional<SizeType32> Request::getPadId() const
{
    return mImpl->getPadId();
}

std::optional<std::vector<SizeType32>> Request::getPositionIds() const
{
    return mImpl->getPositionIds();
}

std::optional<std::list<VecTokens>> Request::getBadWords() const
{
    return mImpl->getBadWords();
}

std::optional<std::list<VecTokens>> Request::getStopWords() const
{
    return mImpl->getStopWords();
}

std::optional<Tensor> Request::getEmbeddingBias() const
{
    return mImpl->getEmbeddingBias();
}

std::optional<ExternalDraftTokensConfig> Request::getExternalDraftTokensConfig() const
{
    return mImpl->getExternalDraftTokensConfig();
}

std::optional<PromptTuningConfig> Request::getPromptTuningConfig() const
{
    return mImpl->getPromptTuningConfig();
}

std::optional<Tensor> Request::getMultimodalEmbedding() const
{
    return mImpl->getMultimodalEmbedding();
}

std::optional<MultimodalInput> Request::getMultimodalInput() const
{
    return mImpl->getMultimodalInput();
}

std::optional<MropeConfig> Request::getMropeConfig() const
{
    return mImpl->getMropeConfig();
}

std::optional<LoraConfig> Request::getLoraConfig() const
{
    return mImpl->getLoraConfig();
}

std::optional<LookaheadDecodingConfig> Request::getLookaheadConfig() const
{
    return mImpl->getLookaheadConfig();
}

std::optional<KvCacheRetentionConfig> Request::getKvCacheRetentionConfig() const
{
    return mImpl->getKvCacheRetentionConfig();
}

std::optional<std::string> Request::getLogitsPostProcessorName() const
{
    return mImpl->getLogitsPostProcessorName();
}

std::optional<LogitsPostProcessor> Request::getLogitsPostProcessor() const
{
    return mImpl->getLogitsPostProcessor();
}

std::optional<VecTokens> Request::getEncoderInputTokenIds() const
{
    return mImpl->getEncoderInputTokenIds();
}

std::optional<IdType> Request::getClientId() const
{
    return mImpl->getClientId();
}

PriorityType Request::getPriority() const
{
    return mImpl->getPriority();
}

std::optional<MillisecondsType> Request::getAllottedTimeMs() const
{
    return mImpl->getAllottedTimeMs();
}

bool Request::getReturnAllGeneratedTokens() const
{
    return mImpl->getReturnAllGeneratedTokens();
}

RequestType Request::getRequestType() const
{
    return mImpl->getRequestType();
}

std::optional<ContextPhaseParams> const& Request::getContextPhaseParams() const
{
    return mImpl->getContextPhaseParams();
}

std::optional<Tensor> Request::getEncoderInputFeatures() const
{
    return mImpl->getEncoderInputFeatures();
}

std::optional<SizeType32> Request::getEncoderOutputLength() const
{
    return mImpl->getEncoderOutputLength();
}

std::optional<Tensor> Request::getCrossAttentionMask() const
{
    return mImpl->getCrossAttentionMask();
}

std::optional<EagleConfig> Request::getEagleConfig() const
{
    return mImpl->getEagleConfig();
}

std::optional<Tensor> Request::getSkipCrossAttnBlocks() const
{
    return mImpl->getSkipCrossAttnBlocks();
}

std::optional<GuidedDecodingParams> Request::getGuidedDecodingParams() const
{
    return mImpl->getGuidedDecodingParams();
}

std::optional<SizeType32> Request::getLanguageAdapterUid() const
{
    return mImpl->getLanguageAdapterUid();
}

std::optional<CacheSaltIDType> Request::getCacheSaltID() const
{
    return mImpl->getCacheSaltID();
}

void Request::setStreaming(bool streaming)
{
    mImpl->setStreaming(streaming);
}

void Request::setSamplingConfig(SamplingConfig const& config)
{
    mImpl->setSamplingConfig(config);
}

void Request::setOutputConfig(OutputConfig const& outputConfig)
{
    mImpl->setOutputConfig(outputConfig);
}

void Request::setEndId(SizeType32 endId)
{
    mImpl->setEndId(endId);
}

void Request::setPadId(SizeType32 padId)
{
    mImpl->setPadId(padId);
}

void Request::setPositionIds(std::vector<SizeType32> const& positionIds)
{
    mImpl->setPositionIds(positionIds);
}

void Request::setBadWords(std::list<VecTokens> const& badWords)
{
    mImpl->setBadWords(badWords);
}

void Request::setStopWords(std::list<VecTokens> const& stopWords)
{
    mImpl->setStopWords(stopWords);
}

void Request::setEmbeddingBias(Tensor const& embeddingBias)
{
    mImpl->setEmbeddingBias(embeddingBias);
}

void Request::setExternalDraftTokensConfig(ExternalDraftTokensConfig const& specDecodingConfig)
{
    mImpl->setExternalDraftTokensConfig(specDecodingConfig);
}

void Request::setPromptTuningConfig(PromptTuningConfig const& pTuningConfig)
{
    mImpl->setPromptTuningConfig(pTuningConfig);
}

void Request::setMultimodalEmbedding(Tensor const& multimodalEmbedding)
{
    return mImpl->setMultimodalEmbedding(multimodalEmbedding);
}

void Request::setMultimodalInput(MultimodalInput const& multimodalInput)
{
    return mImpl->setMultimodalInput(multimodalInput);
}

void Request::setMropeConfig(MropeConfig const& mRopeConfig)
{
    mImpl->setMropeConfig(mRopeConfig);
}

void Request::setLoraConfig(LoraConfig const& loraConfig)
{
    mImpl->setLoraConfig(loraConfig);
}

void Request::setLookaheadConfig(LookaheadDecodingConfig const& lookaheadConfig)
{
    mImpl->setLookaheadConfig(lookaheadConfig);
}

void Request::setKvCacheRetentionConfig(KvCacheRetentionConfig const& kvCacheRetentionConfig)
{
    mImpl->setKvCacheRetentionConfig(kvCacheRetentionConfig);
}

void Request::setLogitsPostProcessorName(std::string const& logitsPostProcessorName)
{
    mImpl->setLogitsPostProcessorName(logitsPostProcessorName);
}

void Request::setLogitsPostProcessor(std::optional<LogitsPostProcessor> const& logitsPostProcessor)
{
    mImpl->setLogitsPostProcessor(logitsPostProcessor);
}

void Request::setEncoderInputTokenIds(VecTokens const& encoderInputTokenIds)
{
    mImpl->setEncoderInputTokenIds(encoderInputTokenIds);
}

void Request::setClientId(IdType clientId)
{
    mImpl->setClientId(clientId);
}

void Request::setPriority(PriorityType priority)
{
    mImpl->setPriority(priority);
}

void Request::setReturnAllGeneratedTokens(bool returnAllGeneratedTokens)
{
    mImpl->setReturnAllGeneratedTokens(returnAllGeneratedTokens);
}

void Request::setRequestType(RequestType const& requestType)
{
    mImpl->setRequestType(requestType);
}

void Request::setContextPhaseParams(ContextPhaseParams contextPhaseParams)
{
    mImpl->setContextPhaseParams(std::move(contextPhaseParams));
}

void Request::setEncoderInputFeatures(Tensor encoderInputFeatures)
{
    mImpl->setEncoderInputFeatures(encoderInputFeatures);
}

void Request::setEncoderOutputLength(SizeType32 encoderOutputLength)
{
    mImpl->setEncoderOutputLength(encoderOutputLength);
}

void Request::setCrossAttentionMask(Tensor crossAttentionMask)
{
    mImpl->setCrossAttentionMask(crossAttentionMask);
}

void Request::setEagleConfig(std::optional<EagleConfig> const& eagleConfig)
{
    mImpl->setEagleConfig(eagleConfig);
}

void Request::setSkipCrossAttnBlocks(Tensor skipCrossAttnBlocks)
{
    return mImpl->setSkipCrossAttnBlocks(skipCrossAttnBlocks);
}

void Request::setGuidedDecodingParams(GuidedDecodingParams const& guidedDecodingParams)
{
    mImpl->setGuidedDecodingParams(guidedDecodingParams);
}

void Request::setAllottedTimeMs(MillisecondsType allottedTimeMs)
{
    return mImpl->setAllottedTimeMs(allottedTimeMs);
}

void Request::setLanguageAdapterUid(SizeType32 languageAdapterUid)
{
    return mImpl->setLanguageAdapterUid(languageAdapterUid);
}

void Request::setCacheSaltID(CacheSaltIDType cacheSaltID)
{
    return mImpl->setCacheSaltID(cacheSaltID);
}
} // namespace tensorrt_llm::executor
