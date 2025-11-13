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

#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestImpl.h"
#include "tensorrt_llm/executor/responseImpl.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>

namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::executor
{

// TimePoint
RequestPerfMetrics::TimePoint Serialization::deserializeTimePoint(std::istream& is)
{
    auto serialized_time = su::deserialize<uint64_t>(is);
    std::chrono::microseconds duration(serialized_time);
    return std::chrono::steady_clock::time_point(duration);
}

void Serialization::serialize(RequestPerfMetrics::TimePoint const& tp, std::ostream& os)
{
    su::serialize(std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count(), os);
}

size_t Serialization::serializedSize(RequestPerfMetrics::TimePoint const& /*unused*/)
{
    return sizeof(RequestPerfMetrics::TimePoint);
}

// RequestPerfMetrics
RequestPerfMetrics Serialization::deserializeRequestPerfMetrics(std::istream& is)
{
    auto arrivalTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto firstScheduledTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto firstTokenTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto lastTokenTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto kvCacheTransferStart = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto kvCacheTransferEnd = su::deserialize<RequestPerfMetrics::TimePoint>(is);

    auto kvCacheSize = su::deserialize<size_t>(is);

    auto numTotalAllocatedBlocks = su::deserialize<SizeType32>(is);
    auto numNewAllocatedBlocks = su::deserialize<SizeType32>(is);
    auto numReusedBlocks = su::deserialize<SizeType32>(is);
    auto numMissedBlocks = su::deserialize<SizeType32>(is);
    auto kvCacheHitRate = su::deserialize<FloatType>(is);

    auto acceptanceRate = su::deserialize<FloatType>(is);
    auto totalAcceptedDraftTokens = su::deserialize<SizeType32>(is);
    auto totalDraftTokens = su::deserialize<SizeType32>(is);

    auto firstIter = su::deserialize<std::optional<IterationType>>(is);
    auto lastIter = su::deserialize<std::optional<IterationType>>(is);
    auto iter = su::deserialize<std::optional<IterationType>>(is);

    RequestPerfMetrics::TimingMetrics timingMetrics{arrivalTime, firstScheduledTime, firstTokenTime, lastTokenTime,
        kvCacheTransferStart, kvCacheTransferEnd, kvCacheSize};
    RequestPerfMetrics::KvCacheMetrics kvCacheMetrics{
        numTotalAllocatedBlocks, numNewAllocatedBlocks, numReusedBlocks, numMissedBlocks, kvCacheHitRate};
    RequestPerfMetrics::SpeculativeDecodingMetrics specDecMetrics{
        acceptanceRate, totalAcceptedDraftTokens, totalDraftTokens};
    return RequestPerfMetrics{timingMetrics, kvCacheMetrics, specDecMetrics, firstIter, lastIter, iter};
}

void Serialization::serialize(RequestPerfMetrics const& metrics, std::ostream& os)
{
    su::serialize(metrics.timingMetrics.arrivalTime, os);
    su::serialize(metrics.timingMetrics.firstScheduledTime, os);
    su::serialize(metrics.timingMetrics.firstTokenTime, os);
    su::serialize(metrics.timingMetrics.lastTokenTime, os);
    su::serialize(metrics.timingMetrics.kvCacheTransferStart, os);
    su::serialize(metrics.timingMetrics.kvCacheTransferEnd, os);

    su::serialize(metrics.timingMetrics.kvCacheSize, os);

    su::serialize(metrics.kvCacheMetrics.numTotalAllocatedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.numNewAllocatedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.numReusedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.numMissedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.kvCacheHitRate, os);

    su::serialize(metrics.speculativeDecoding.acceptanceRate, os);
    su::serialize(metrics.speculativeDecoding.totalAcceptedDraftTokens, os);
    su::serialize(metrics.speculativeDecoding.totalDraftTokens, os);

    su::serialize(metrics.firstIter, os);
    su::serialize(metrics.lastIter, os);
    su::serialize(metrics.iter, os);
}

size_t Serialization::serializedSize(RequestPerfMetrics const& metrics)
{
    size_t totalSize = 0;

    totalSize += su::serializedSize(metrics.timingMetrics.arrivalTime);
    totalSize += su::serializedSize(metrics.timingMetrics.firstScheduledTime);
    totalSize += su::serializedSize(metrics.timingMetrics.firstTokenTime);
    totalSize += su::serializedSize(metrics.timingMetrics.lastTokenTime);
    totalSize += su::serializedSize(metrics.timingMetrics.kvCacheTransferStart);
    totalSize += su::serializedSize(metrics.timingMetrics.kvCacheTransferEnd);

    totalSize += su::serializedSize(metrics.timingMetrics.kvCacheSize);

    totalSize += su::serializedSize(metrics.kvCacheMetrics.numTotalAllocatedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.numNewAllocatedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.numReusedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.numMissedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.kvCacheHitRate);

    totalSize += su::serializedSize(metrics.speculativeDecoding.acceptanceRate);
    totalSize += su::serializedSize(metrics.speculativeDecoding.totalAcceptedDraftTokens);
    totalSize += su::serializedSize(metrics.speculativeDecoding.totalDraftTokens);

    totalSize += su::serializedSize(metrics.firstIter);
    totalSize += su::serializedSize(metrics.lastIter);
    totalSize += su::serializedSize(metrics.iter);

    return totalSize;
}

// SamplingConfig
SamplingConfig Serialization::deserializeSamplingConfig(std::istream& is)
{
    auto beamWidth = su::deserialize<SizeType32>(is);
    auto topK = su::deserialize<std::optional<SizeType32>>(is);
    auto topP = su::deserialize<std::optional<FloatType>>(is);
    auto topPMin = su::deserialize<std::optional<FloatType>>(is);
    auto topPResetIds = su::deserialize<std::optional<TokenIdType>>(is);
    auto topPDecay = su::deserialize<std::optional<FloatType>>(is);
    auto randomSeed = su::deserialize<std::optional<RandomSeedType>>(is);
    auto temperature = su::deserialize<std::optional<FloatType>>(is);
    auto minLength = su::deserialize<std::optional<SizeType32>>(is);
    auto beamSearchDiversityRate = su::deserialize<std::optional<FloatType>>(is);
    auto repetitionPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto presencePenalty = su::deserialize<std::optional<FloatType>>(is);
    auto frequencyPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto promptIgnoreLength = su::deserialize<std::optional<SizeType32>>(is);
    auto lengthPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto earlyStopping = su::deserialize<std::optional<SizeType32>>(is);
    auto noRepeatNgramSize = su::deserialize<std::optional<SizeType32>>(is);
    auto numReturnSequences = su::deserialize<std::optional<SizeType32>>(is);
    auto minP = su::deserialize<std::optional<FloatType>>(is);
    auto beamWidthArray = su::deserialize<std::optional<std::vector<SizeType32>>>(is);

    return SamplingConfig{beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed, temperature, minLength,
        beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, promptIgnoreLength,
        lengthPenalty, earlyStopping, noRepeatNgramSize, numReturnSequences, minP, beamWidthArray};
}

void Serialization::serialize(SamplingConfig const& config, std::ostream& os)
{
    su::serialize(config.mBeamWidth, os);
    su::serialize(config.mTopK, os);
    su::serialize(config.mTopP, os);
    su::serialize(config.mTopPMin, os);
    su::serialize(config.mTopPResetIds, os);
    su::serialize(config.mTopPDecay, os);
    su::serialize(config.mSeed, os);
    su::serialize(config.mTemperature, os);
    su::serialize(config.mMinTokens, os);
    su::serialize(config.mBeamSearchDiversityRate, os);
    su::serialize(config.mRepetitionPenalty, os);
    su::serialize(config.mPresencePenalty, os);
    su::serialize(config.mFrequencyPenalty, os);
    su::serialize(config.mPromptIgnoreLength, os);
    su::serialize(config.mLengthPenalty, os);
    su::serialize(config.mEarlyStopping, os);
    su::serialize(config.mNoRepeatNgramSize, os);
    su::serialize(config.mNumReturnSequences, os);
    su::serialize(config.mMinP, os);
    su::serialize(config.mBeamWidthArray, os);
}

size_t Serialization::serializedSize(SamplingConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mBeamWidth);
    totalSize += su::serializedSize(config.mTopK);
    totalSize += su::serializedSize(config.mTopP);
    totalSize += su::serializedSize(config.mTopPMin);
    totalSize += su::serializedSize(config.mTopPResetIds);
    totalSize += su::serializedSize(config.mTopPDecay);
    totalSize += su::serializedSize(config.mSeed);
    totalSize += su::serializedSize(config.mTemperature);
    totalSize += su::serializedSize(config.mMinTokens);
    totalSize += su::serializedSize(config.mBeamSearchDiversityRate);
    totalSize += su::serializedSize(config.mRepetitionPenalty);
    totalSize += su::serializedSize(config.mPresencePenalty);
    totalSize += su::serializedSize(config.mFrequencyPenalty);
    totalSize += su::serializedSize(config.mPromptIgnoreLength);
    totalSize += su::serializedSize(config.mLengthPenalty);
    totalSize += su::serializedSize(config.mEarlyStopping);
    totalSize += su::serializedSize(config.mNoRepeatNgramSize);
    totalSize += su::serializedSize(config.mNumReturnSequences);
    totalSize += su::serializedSize(config.mMinP);
    totalSize += su::serializedSize(config.mBeamWidthArray);
    return totalSize;
}

// OutputConfig
OutputConfig Serialization::deserializeOutputConfig(std::istream& is)
{
    auto returnLogProbs = su::deserialize<bool>(is);
    auto returnContextLogits = su::deserialize<bool>(is);
    auto returnGenerationLogits = su::deserialize<bool>(is);
    auto excludeInputFromOutput = su::deserialize<bool>(is);
    auto returnEncoderOutput = su::deserialize<bool>(is);
    auto returnPerfMetrics = su::deserialize<bool>(is);
    auto additionalOutputs = su::deserialize<std::optional<std::vector<AdditionalModelOutput>>>(is);
    return OutputConfig{returnLogProbs, returnContextLogits, returnGenerationLogits, excludeInputFromOutput,
        returnEncoderOutput, returnPerfMetrics, additionalOutputs};
}

void Serialization::serialize(OutputConfig const& config, std::ostream& os)
{
    su::serialize(config.returnLogProbs, os);
    su::serialize(config.returnContextLogits, os);
    su::serialize(config.returnGenerationLogits, os);
    su::serialize(config.excludeInputFromOutput, os);
    su::serialize(config.returnEncoderOutput, os);
    su::serialize(config.returnPerfMetrics, os);
    su::serialize(config.additionalModelOutputs, os);
}

size_t Serialization::serializedSize(OutputConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.returnLogProbs);
    totalSize += su::serializedSize(config.returnContextLogits);
    totalSize += su::serializedSize(config.returnGenerationLogits);
    totalSize += su::serializedSize(config.excludeInputFromOutput);
    totalSize += su::serializedSize(config.returnEncoderOutput);
    totalSize += su::serializedSize(config.returnPerfMetrics);
    totalSize += su::serializedSize(config.additionalModelOutputs);
    return totalSize;
}

// AdditionalModelOutput
AdditionalModelOutput Serialization::deserializeAdditionalModelOutput(std::istream& is)
{
    auto name = su::deserialize<std::string>(is);
    auto gatherContext = su::deserialize<bool>(is);
    return AdditionalModelOutput{name, gatherContext};
}

void Serialization::serialize(AdditionalModelOutput const& additionalModelOutput, std::ostream& os)
{
    su::serialize(additionalModelOutput.name, os);
    su::serialize(additionalModelOutput.gatherContext, os);
}

size_t Serialization::serializedSize(AdditionalModelOutput const& additionalModelOutput)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(additionalModelOutput.name);
    totalSize += su::serializedSize(additionalModelOutput.gatherContext);
    return totalSize;
}

// ExternalDraftTokensConfig
ExternalDraftTokensConfig Serialization::deserializeExternalDraftTokensConfig(std::istream& is)
{
    auto tokens = su::deserialize<VecTokens>(is);
    auto logits = su::deserialize<std::optional<Tensor>>(is);
    auto acceptanceThreshold = su::deserialize<std::optional<FloatType>>(is);
    auto fastLogits = su::deserialize<std::optional<bool>>(is);

    return ExternalDraftTokensConfig{std::move(tokens), std::move(logits), acceptanceThreshold, fastLogits};
}

void Serialization::serialize(ExternalDraftTokensConfig const& config, std::ostream& os)
{
    su::serialize(config.mTokens, os);
    su::serialize(config.mLogits, os);
    su::serialize(config.mAcceptanceThreshold, os);
    su::serialize(config.mFastLogits, os);
}

size_t Serialization::serializedSize(ExternalDraftTokensConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mTokens);
    totalSize += su::serializedSize(config.mLogits);
    totalSize += su::serializedSize(config.mAcceptanceThreshold);
    totalSize += su::serializedSize(config.mFastLogits);
    return totalSize;
}

// PromptTuningConfig
PromptTuningConfig Serialization::deserializePromptTuningConfig(std::istream& is)
{
    auto tensor = su::deserialize<Tensor>(is);
    auto inputTokenExtraIds = su::deserialize<std::optional<VecTokenExtraIds>>(is);
    return PromptTuningConfig{std::move(tensor), std::move(inputTokenExtraIds)};
}

void Serialization::serialize(PromptTuningConfig const& config, std::ostream& os)
{
    su::serialize(config.mEmbeddingTable, os);
    su::serialize(config.mInputTokenExtraIds, os);
}

size_t Serialization::serializedSize(PromptTuningConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mEmbeddingTable);
    totalSize += su::serializedSize(config.mInputTokenExtraIds);
    return totalSize;
}

// MultimodalInput
MultimodalInput Serialization::deserializeMultimodalInput(std::istream& is)
{
    auto multimodalHashes = su::deserialize<std::vector<std::vector<SizeType32>>>(is);
    auto multimodalPositions = su::deserialize<std::vector<SizeType32>>(is);
    auto multimodalLengths = su::deserialize<std::vector<SizeType32>>(is);
    return MultimodalInput{std::move(multimodalHashes), std::move(multimodalPositions), std::move(multimodalLengths)};
}

void Serialization::serialize(MultimodalInput const& multimodalInput, std::ostream& os)
{
    su::serialize(multimodalInput.mMultimodalHashes, os);
    su::serialize(multimodalInput.mMultimodalPositions, os);
    su::serialize(multimodalInput.mMultimodalLengths, os);
}

size_t Serialization::serializedSize(MultimodalInput const& multimodalInput)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(multimodalInput.mMultimodalHashes);
    totalSize += su::serializedSize(multimodalInput.mMultimodalPositions);
    totalSize += su::serializedSize(multimodalInput.mMultimodalLengths);
    return totalSize;
}

// MropeConfig
MropeConfig Serialization::deserializeMropeConfig(std::istream& is)
{
    auto mropeRotaryCosSin = su::deserialize<Tensor>(is);
    auto mropePositionDeltas = su::deserialize<SizeType32>(is);
    return MropeConfig{std::move(mropeRotaryCosSin), std::move(mropePositionDeltas)};
}

void Serialization::serialize(MropeConfig const& config, std::ostream& os)
{
    su::serialize(config.mMRopeRotaryCosSin, os);
    su::serialize(config.mMRopePositionDeltas, os);
}

size_t Serialization::serializedSize(MropeConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mMRopeRotaryCosSin);
    totalSize += su::serializedSize(config.mMRopePositionDeltas);

    return totalSize;
}

// LoraConfig
LoraConfig Serialization::deserializeLoraConfig(std::istream& is)
{
    auto taskId = su::deserialize<IdType>(is);
    auto weights = su::deserialize<std::optional<Tensor>>(is);
    auto config = su::deserialize<std::optional<Tensor>>(is);
    return LoraConfig{taskId, std::move(weights), std::move(config)};
}

void Serialization::serialize(LoraConfig const& config, std::ostream& os)
{
    su::serialize(config.mTaskId, os);
    su::serialize(config.mWeights, os);
    su::serialize(config.mConfig, os);
}

size_t Serialization::serializedSize(LoraConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mTaskId);
    totalSize += su::serializedSize(config.mWeights);
    totalSize += su::serializedSize(config.mConfig);
    return totalSize;
}

// SocketState
kv_cache::SocketState Serialization::deserializeSocketState(std::istream& is)
{
    auto port = su::deserialize<decltype(kv_cache::SocketState::mPort)>(is);
    auto ip = su::deserialize<decltype(kv_cache::SocketState::mIp)>(is);
    return kv_cache::SocketState{port, std::move(ip)};
}

void Serialization::serialize(kv_cache::SocketState const& state, std::ostream& os)
{
    su::serialize(state.mPort, os);
    su::serialize(state.mIp, os);
}

size_t Serialization::serializedSize(kv_cache::SocketState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mPort);
    totalSize += su::serializedSize(state.mIp);
    return totalSize;
}

// AgentState
kv_cache::AgentState Serialization::deserializeAgentState(std::istream& is)
{
    auto agentName = su::deserialize<decltype(kv_cache::AgentState::mAgentName)>(is);
    auto connectionInfo = su::deserialize<decltype(kv_cache::AgentState::mConnectionInfo)>(is);
    return kv_cache::AgentState{std::move(agentName), std::move(connectionInfo)};
}

void Serialization::serialize(kv_cache::AgentState const& state, std::ostream& os)
{
    su::serialize(state.mAgentName, os);
    su::serialize(state.mConnectionInfo, os);
}

size_t Serialization::serializedSize(kv_cache::AgentState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mAgentName);
    totalSize += su::serializedSize(state.mConnectionInfo);
    return totalSize;
}

// CommState
kv_cache::CommState Serialization::deserializeCommState(std::istream& is)
{
    auto selfIdx = su::deserialize<decltype(kv_cache::CommState::mSelfIdx)>(is);
    auto variantIdx = su::deserialize<std::size_t>(is);
    constexpr std::size_t mpiIdx{1};
    constexpr std::size_t socketIdx{2};
    constexpr std::size_t agentIdx{3};
    static_assert(
        std::is_same_v<kv_cache::MpiState, std::variant_alternative_t<mpiIdx, decltype(kv_cache::CommState::mState)>>);
    static_assert(std::is_same_v<std::vector<kv_cache::SocketState>,
        std::variant_alternative_t<socketIdx, decltype(kv_cache::CommState::mState)>>);
    if (variantIdx == mpiIdx)
    {
        auto ranks = su::deserialize<decltype(kv_cache::MpiState::mRanks)>(is);
        return kv_cache::CommState{std::move(ranks), selfIdx};
    }
    if (variantIdx == socketIdx)
    {
        auto state = su::deserialize<std::vector<kv_cache::SocketState>>(is);
        return kv_cache::CommState{std::move(state), selfIdx};
    }
    if (variantIdx == agentIdx)
    {
        auto state = su::deserialize<std::vector<kv_cache::AgentState>>(is);
        return kv_cache::CommState{std::move(state), selfIdx};
    }
    return {};
}

void Serialization::serialize(kv_cache::CommState const& state, std::ostream& os)
{
    su::serialize(state.mSelfIdx, os);
    su::serialize(state.mState.index(), os);
    if (state.isMpiState())
    {
        su::serialize(state.getMpiState().mRanks, os);
    }
    else if (state.isSocketState())
    {
        su::serialize(state.getSocketState(), os);
    }
    else if (state.isAgentState())
    {
        su::serialize(state.getAgentState(), os);
    }
    else
    {
        TLLM_THROW("Unknown context phase state communication type.");
    }
}

size_t Serialization::serializedSize(kv_cache::CommState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mSelfIdx);
    totalSize += su::serializedSize(state.mState.index());
    if (state.isMpiState())
    {
        totalSize += su::serializedSize(state.getMpiState().mRanks);
    }
    else if (state.isSocketState())
    {
        totalSize += su::serializedSize(state.getSocketState());
    }
    else if (state.isAgentState())
    {
        totalSize += su::serializedSize(state.getAgentState());
    }
    else
    {
        TLLM_THROW("Unknown context phase state communication type.");
    }
    return totalSize;
}

// CacheState
kv_cache::CacheState Serialization::deserializeCacheState(std::istream& is)
{
    using CacheState = kv_cache::CacheState;
    auto nbKvHeadsPerLayer = su::deserialize<decltype(CacheState::ModelConfig::mNbKvHeadsPerLayer)>(is);
    auto sizePerHead = su::deserialize<decltype(CacheState::ModelConfig::mSizePerHead)>(is);
    auto tokensPerBlock = su::deserialize<decltype(CacheState::ModelConfig::mTokensPerBlock)>(is);
    auto tensorParallelism = su::deserialize<decltype(CacheState::ParallelConfig::mTensorParallelism)>(is);
    auto pipelineParallelism = su::deserialize<decltype(CacheState::ParallelConfig::mPipelineParallelism)>(is);
    auto contextParallelism = su::deserialize<decltype(CacheState::ParallelConfig::mContextParallelism)>(is);
    auto enableAttentionDP = su::deserialize<decltype(CacheState::ParallelConfig::mEnableAttentionDP)>(is);
    auto DPrank = su::deserialize<decltype(CacheState::ParallelConfig::mDPrank)>(is);
    auto DPsize = su::deserialize<decltype(CacheState::ParallelConfig::mDPsize)>(is);
    auto attentionLayerNumPerPP = su::deserialize<decltype(CacheState::ParallelConfig::mAttentionLayerNumPerPP)>(is);
    auto dataType = su::deserialize<decltype(CacheState::mDataType)>(is);
    auto attentionType = su::deserialize<decltype(CacheState::AttentionConfig::mAttentionType)>(is);
    auto kvFactor = su::deserialize<decltype(CacheState::AttentionConfig::mKvFactor)>(is);
    auto enableBlockReuse = su::deserialize<bool>(is);
    auto hasIndexerKCache = su::deserialize<bool>(is);
    auto indexerDimPerHead = su::deserialize<decltype(CacheState::ModelConfig::mSizePerHead)>(is);
    auto indexerKCacheQuantBlockSize = su::deserialize<decltype(CacheState::ModelConfig::mTokensPerBlock)>(is);
    return CacheState{nbKvHeadsPerLayer, sizePerHead, tokensPerBlock, tensorParallelism, pipelineParallelism,
        contextParallelism, attentionLayerNumPerPP, dataType, attentionType, kvFactor, enableAttentionDP, DPrank,
        DPsize, enableBlockReuse, hasIndexerKCache, indexerDimPerHead, indexerKCacheQuantBlockSize};
}

void Serialization::serialize(kv_cache::CacheState const& state, std::ostream& os)
{
    su::serialize(state.mModelConfig.mNbKvHeadsPerLayer, os);
    su::serialize(state.mModelConfig.mSizePerHead, os);
    su::serialize(state.mModelConfig.mTokensPerBlock, os);
    su::serialize(state.mParallelConfig.mTensorParallelism, os);
    su::serialize(state.mParallelConfig.mPipelineParallelism, os);
    su::serialize(state.mParallelConfig.mContextParallelism, os);
    su::serialize(state.mParallelConfig.mEnableAttentionDP, os);
    su::serialize(state.mParallelConfig.mDPrank, os);
    su::serialize(state.mParallelConfig.mDPsize, os);
    su::serialize(state.mParallelConfig.mAttentionLayerNumPerPP, os);
    su::serialize(state.mDataType, os);
    su::serialize(state.mAttentionConfig.mAttentionType, os);
    su::serialize(state.mAttentionConfig.mKvFactor, os);
    su::serialize(state.mEnableBlockReuse, os);
    su::serialize(state.getHasIndexerKCache(), os);
    su::serialize(state.getIndexerDimPerHead(), os);
    su::serialize(state.getIndexerKCacheQuantBlockSize(), os);
}

size_t Serialization::serializedSize(kv_cache::CacheState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mModelConfig.mNbKvHeadsPerLayer);
    totalSize += su::serializedSize(state.mModelConfig.mSizePerHead);
    totalSize += su::serializedSize(state.mModelConfig.mTokensPerBlock);
    totalSize += su::serializedSize(state.mParallelConfig.mTensorParallelism);
    totalSize += su::serializedSize(state.mParallelConfig.mPipelineParallelism);
    totalSize += su::serializedSize(state.mParallelConfig.mContextParallelism);
    totalSize += su::serializedSize(state.mParallelConfig.mEnableAttentionDP);
    totalSize += su::serializedSize(state.mParallelConfig.mDPrank);
    totalSize += su::serializedSize(state.mParallelConfig.mDPsize);
    totalSize += su::serializedSize(state.mParallelConfig.mAttentionLayerNumPerPP);
    totalSize += su::serializedSize(state.mDataType);
    totalSize += su::serializedSize(state.mAttentionConfig.mAttentionType);
    totalSize += su::serializedSize(state.mAttentionConfig.mKvFactor);
    totalSize += su::serializedSize(state.mEnableBlockReuse);
    totalSize += su::serializedSize(state.getHasIndexerKCache());
    totalSize += su::serializedSize(state.getIndexerDimPerHead());
    totalSize += su::serializedSize(state.getIndexerKCacheQuantBlockSize());
    return totalSize;
}

// DataTransceiverState

DataTransceiverState Serialization::deserializeDataTransceiverState(std::vector<char>& buffer)
{
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    return deserializeDataTransceiverState(is);
}

DataTransceiverState Serialization::deserializeDataTransceiverState(std::istream& is)
{
    DataTransceiverState state;
    auto commState = su::deserialize<decltype(DataTransceiverState::mCommState)>(is);
    if (commState)
    {
        state.setCommState(std::move(commState).value());
    }
    auto cacheState = su::deserialize<decltype(DataTransceiverState::mCacheState)>(is);
    if (cacheState)
    {
        state.setCacheState(std::move(cacheState).value());
    }
    return state;
}

void Serialization::serialize(DataTransceiverState const& state, std::ostream& os)
{
    su::serialize(state.mCommState, os);
    su::serialize(state.mCacheState, os);
}

std::vector<char> Serialization::serialize(DataTransceiverState const& state)
{
    std::vector<char> buffer(serializedSize(state));

    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);
    serialize(state, os);
    return buffer;
}

size_t Serialization::serializedSize(DataTransceiverState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mCommState);
    totalSize += su::serializedSize(state.mCacheState);
    return totalSize;
}

// ContextPhaseParams
ContextPhaseParams Serialization::deserializeContextPhaseParams(std::istream& is)
{
    auto reqId = su::deserialize<decltype(ContextPhaseParams::mReqId)>(is);
    auto firstGenTokens = su::deserialize<decltype(ContextPhaseParams::mFirstGenTokens)>(is);
    auto draftTokens = su::deserialize<decltype(ContextPhaseParams::mDraftTokens)>(is);
    auto hasState = su::deserialize<bool>(is);
    if (hasState)
    {
        auto state = std::make_unique<DataTransceiverState>();
        *state = deserializeDataTransceiverState(is);
        return ContextPhaseParams{std::move(firstGenTokens), reqId, state.release(), std::move(draftTokens)};
    }
    return ContextPhaseParams{std::move(firstGenTokens), reqId, nullptr, std::move(draftTokens)};
}

void Serialization::serialize(ContextPhaseParams const& contextPhaseParams, std::ostream& os)
{
    su::serialize(contextPhaseParams.mReqId, os);
    su::serialize(contextPhaseParams.mFirstGenTokens, os);
    su::serialize(contextPhaseParams.mDraftTokens, os);
    su::serialize(static_cast<bool>(contextPhaseParams.mState), os);
    if (contextPhaseParams.mState)
    {
        serialize(*static_cast<DataTransceiverState*>(contextPhaseParams.mState.get()), os);
    }
}

size_t Serialization::serializedSize(ContextPhaseParams const& contextPhaseParams)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(contextPhaseParams.mReqId);
    totalSize += su::serializedSize(contextPhaseParams.mFirstGenTokens);
    totalSize += su::serializedSize(contextPhaseParams.mDraftTokens);
    totalSize += su::serializedSize(bool{});
    if (contextPhaseParams.mState)
    {
        totalSize += serializedSize(*static_cast<DataTransceiverState*>(contextPhaseParams.mState.get()));
    }
    return totalSize;
}

// Request
Request Serialization::deserializeRequest(std::istream& is)
{
    // Serialization of Request with logitsPostProcessor is currently not supported.
    // Dynamic logitsPostProcessor only supported with replicate=false or no tensor parallelism.
    auto inputTokenIds = su::deserialize<VecTokens>(is);
    auto maxNewTokens = su::deserialize<SizeType32>(is);
    auto streaming = su::deserialize<bool>(is);
    auto samplingConfig = su::deserialize<SamplingConfig>(is);
    auto outputConfig = su::deserialize<OutputConfig>(is);
    auto endId = su::deserialize<std::optional<SizeType32>>(is);
    auto padId = su::deserialize<std::optional<SizeType32>>(is);
    auto positionIds = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto badWords = su::deserialize<std::optional<std::list<VecTokens>>>(is);
    auto stopWords = su::deserialize<std::optional<std::list<VecTokens>>>(is);
    auto embeddingBias = su::deserialize<std::optional<Tensor>>(is);
    auto externalDraftTokensConfig = su::deserialize<std::optional<ExternalDraftTokensConfig>>(is);
    auto pTuningConfig = su::deserialize<std::optional<PromptTuningConfig>>(is);
    auto multimodalInput = su::deserialize<std::optional<MultimodalInput>>(is);
    auto multimodalEmbedding = su::deserialize<std::optional<Tensor>>(is);
    auto mRopeConfig = su::deserialize<std::optional<MropeConfig>>(is);
    auto loraConfig = su::deserialize<std::optional<LoraConfig>>(is);
    auto lookaheadConfig = su::deserialize<std::optional<LookaheadDecodingConfig>>(is);
    auto kvCacheRetentionConfig = su::deserialize<std::optional<KvCacheRetentionConfig>>(is);
    auto logitsPostProcessorName = su::deserialize<std::optional<std::string>>(is);
    auto encoderInputTokenIds = su::deserialize<std::optional<VecTokens>>(is);
    auto clientId = su::deserialize<std::optional<IdType>>(is);
    auto returnAllGeneratedTokens = su::deserialize<bool>(is);
    auto priority = su::deserialize<executor::PriorityType>(is);
    auto requestType = su::deserialize<executor::RequestType>(is);
    auto contextPhaseParams = su::deserialize<std::optional<ContextPhaseParams>>(is);
    auto encoderInputFeatures = su::deserialize<std::optional<Tensor>>(is);
    auto encoderOutputLength = su::deserialize<std::optional<SizeType32>>(is);
    auto crossAttentionMask = su::deserialize<std::optional<Tensor>>(is);
    auto numReturnSequences = su::deserialize<SizeType32>(is);
    auto eagleConfig = su::deserialize<std::optional<EagleConfig>>(is);
    auto skipCrossAttnBlocks = su::deserialize<std::optional<Tensor>>(is);
    auto guidedDecodingParams = su::deserialize<std::optional<GuidedDecodingParams>>(is);
    auto languageAdapterUid = su::deserialize<std::optional<SizeType32>>(is);
    auto allottedTimeInt = su::deserialize<std::optional<std::chrono::milliseconds::rep>>(is);
    auto allottedTimeMs = allottedTimeInt
        ? std::optional<std::chrono::milliseconds>(std::chrono::milliseconds(*allottedTimeInt))
        : std::nullopt;
    auto cacheSaltID = su::deserialize<std::optional<CacheSaltIDType>>(is);

    return Request(std::move(inputTokenIds), maxNewTokens, streaming, samplingConfig, outputConfig, endId, padId,
        std::move(positionIds), std::move(badWords), std::move(stopWords), std::move(embeddingBias),
        std::move(externalDraftTokensConfig), std::move(pTuningConfig), std::move(multimodalInput),
        std::move(multimodalEmbedding), std::move(mRopeConfig), std::move(loraConfig), lookaheadConfig,
        std::move(kvCacheRetentionConfig), std::move(logitsPostProcessorName), std::nullopt,
        std::move(encoderInputTokenIds), clientId, returnAllGeneratedTokens, priority, requestType,
        std::move(contextPhaseParams), std::move(encoderInputFeatures), encoderOutputLength,
        std::move(crossAttentionMask), numReturnSequences, std::move(eagleConfig), std::move(skipCrossAttnBlocks),
        std::move(guidedDecodingParams), languageAdapterUid, allottedTimeMs, cacheSaltID);
}

void Serialization::serialize(Request const& request, std::ostream& os)
{
    request.mImpl->serialize(os);
}

size_t Serialization::serializedSize(Request const& request)
{
    return request.mImpl->serializedSize();
}

// Tensor
Tensor Serialization::deserializeTensor(std::istream& is)
{
    // DataType
    DataType dataType{};
    is.read(reinterpret_cast<char*>(&dataType), sizeof(dataType));

    // Shape
    size_t shapeSize{0};
    is.read(reinterpret_cast<char*>(&shapeSize), sizeof(size_t));
    static constexpr int32_t MAX_DIMS{8};
    TLLM_CHECK(shapeSize < MAX_DIMS);

    Shape::DimType64 dims[MAX_DIMS];
    is.read(reinterpret_cast<char*>(&dims[0]), shapeSize * sizeof(Shape::DimType64));
    Shape shape(&dims[0], shapeSize);

    // Memory Type
    MemoryType memoryType{};
    is.read(reinterpret_cast<char*>(&memoryType), sizeof(memoryType));

    // Size in bytes
    size_t sizeInBytes{0};
    is.read(reinterpret_cast<char*>(&sizeInBytes), sizeof(size_t));

    Tensor tensor;
    switch (memoryType)
    {
    case MemoryType::kCPU:
    {
        tensor = Tensor::cpu(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kCPU_PINNED:
    {
        tensor = Tensor::pinned(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kUVM:
    {
        tensor = Tensor::managed(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kGPU:
    {
        // TODO: Eventually we might want to support serialization/deserialization in GPU memory
        //       Until then created Pinned tensor and move to GPU
        auto pinnedTensor = Tensor::pinned(dataType, shape);
        is.read(reinterpret_cast<char*>(pinnedTensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        tensor = pinnedTensor.copyToGpu(stream);
        stream->synchronize();
        break;
    }
    case MemoryType::kUNKNOWN:
    {
        TLLM_THROW("Cannot deserialize tensor with UNKNOWN type.");
        break;
    }
    default:
    {
        TLLM_THROW("Memory type not supported in deserializeTensor.");
        break;
    }
    }

    return tensor;
}

void Serialization::serialize(Tensor const& tensor, std::ostream& os)
{
    auto dataType = tensor.getDataType();
    os.write(reinterpret_cast<char const*>(&dataType), sizeof(dataType));
    auto shape = tensor.getShape();
    auto shapeSize = shape.size();
    os.write(reinterpret_cast<char const*>(&shapeSize), sizeof(shapeSize));
    os.write(reinterpret_cast<char const*>(&shape[0]), shapeSize * sizeof(Shape::DimType64));

    // Memory Type
    auto memoryType = tensor.getMemoryType();
    os.write(reinterpret_cast<char const*>(&memoryType), sizeof(memoryType));

    std::size_t sizeInBytes = tensor.getSizeInBytes();
    os.write(reinterpret_cast<char const*>(&sizeInBytes), sizeof(sizeInBytes));

    if (memoryType == MemoryType::kCPU || memoryType == MemoryType::kCPU_PINNED || memoryType == MemoryType::kUVM)
    {
        void const* data = tensor.getData();
        os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
    }
    // Need special treatment for GPU type
    else if (memoryType == MemoryType::kGPU)
    {
        auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        auto pinnedTensor = tensor.copyToPinned(stream);
        stream->synchronize();
        void const* data = pinnedTensor.getData();
        os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
    }
    else if (memoryType == MemoryType::kUNKNOWN)
    {
        TLLM_THROW("Memory type unknown when serializing tensor");
    }
}

size_t Serialization::serializedSize(Tensor const& tensor)
{
    size_t totalSize = 0;
    totalSize += sizeof(tensor.getDataType()); // datatype
    auto const shape = tensor.getShape();
    auto const shapeSize = shape.size();
    totalSize += sizeof(decltype(shapeSize)); // number of dims
    TLLM_CHECK(shapeSize > 0);
    totalSize += shapeSize * sizeof(decltype(shape[0]));

    auto memoryType = tensor.getMemoryType();
    totalSize += sizeof(memoryType); // memory type

    totalSize += sizeof(size_t);     // Size in bytes
    totalSize += tensor.getSizeInBytes();
    return totalSize;
}

// SpeculativeDecodingFastLogitsInfo
SpeculativeDecodingFastLogitsInfo Serialization::deserializeSpecDecFastLogitsInfo(std::istream& is)
{
    auto draftRequestId = su::deserialize<uint64_t>(is);
    auto draftParticipantId = su::deserialize<int32_t>(is);

    return SpeculativeDecodingFastLogitsInfo{draftRequestId, draftParticipantId};
}

void Serialization::serialize(SpeculativeDecodingFastLogitsInfo const& info, std::ostream& os)
{
    su::serialize(info.draftRequestId, os);
    su::serialize(info.draftParticipantId, os);
}

size_t Serialization::serializedSize(SpeculativeDecodingFastLogitsInfo const& info)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(info.draftRequestId);
    totalSize += su::serializedSize(info.draftParticipantId);
    return totalSize;
}

// Result
Result Serialization::deserializeResult(std::istream& is)
{
    Result result{};
    result.isFinal = su::deserialize<bool>(is);
    result.outputTokenIds = su::deserialize<BeamTokens>(is);
    result.cumLogProbs = su::deserialize<std::optional<VecLogProbs>>(is);
    result.logProbs = su::deserialize<std::optional<std::vector<VecLogProbs>>>(is);
    result.contextLogits = su::deserialize<std::optional<Tensor>>(is);
    result.generationLogits = su::deserialize<std::optional<Tensor>>(is);
    result.specDecFastLogitsInfo = su::deserialize<std::optional<SpeculativeDecodingFastLogitsInfo>>(is);
    result.encoderOutput = su::deserialize<std::optional<Tensor>>(is);
    result.finishReasons = su::deserialize<std::vector<FinishReason>>(is);
    result.contextPhaseParams = su::deserialize<std::optional<ContextPhaseParams>>(is);
    result.decodingIter = su::deserialize<SizeType32>(is);
    result.avgDecodedTokensPerIter = su::deserialize<float>(is);
    result.sequenceIndex = su::deserialize<SizeType32>(is);
    result.isSequenceFinal = su::deserialize<bool>(is);
    result.requestPerfMetrics = su::deserialize<std::optional<RequestPerfMetrics>>(is);
    result.additionalOutputs = su::deserialize<std::vector<AdditionalOutput>>(is);
    return result;
}

void Serialization::serialize(Result const& result, std::ostream& os)
{
    su::serialize(result.isFinal, os);
    su::serialize(result.outputTokenIds, os);
    su::serialize(result.cumLogProbs, os);
    su::serialize(result.logProbs, os);
    su::serialize(result.contextLogits, os);
    su::serialize(result.generationLogits, os);
    su::serialize(result.specDecFastLogitsInfo, os);
    su::serialize(result.encoderOutput, os);
    su::serialize(result.finishReasons, os);
    su::serialize(result.contextPhaseParams, os);
    su::serialize(result.decodingIter, os);
    su::serialize(result.avgDecodedTokensPerIter, os);
    su::serialize(result.sequenceIndex, os);
    su::serialize(result.isSequenceFinal, os);
    su::serialize(result.requestPerfMetrics, os);
    su::serialize(result.additionalOutputs, os);
}

size_t Serialization::serializedSize(Result const& result)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(result.isFinal);
    totalSize += su::serializedSize(result.outputTokenIds);
    totalSize += su::serializedSize(result.cumLogProbs);
    totalSize += su::serializedSize(result.logProbs);
    totalSize += su::serializedSize(result.contextLogits);
    totalSize += su::serializedSize(result.specDecFastLogitsInfo);
    totalSize += su::serializedSize(result.generationLogits);
    totalSize += su::serializedSize(result.encoderOutput);
    totalSize += su::serializedSize(result.finishReasons);
    totalSize += su::serializedSize(result.contextPhaseParams);
    totalSize += su::serializedSize(result.decodingIter);
    totalSize += su::serializedSize(result.avgDecodedTokensPerIter);
    totalSize += su::serializedSize(result.sequenceIndex);
    totalSize += su::serializedSize(result.isSequenceFinal);
    totalSize += su::serializedSize(result.requestPerfMetrics);
    totalSize += su::serializedSize(result.additionalOutputs);
    return totalSize;
}

// Response
Response Serialization::deserializeResponse(std::istream& is)
{
    auto requestId = su::deserialize<IdType>(is);
    auto errOrResult = su::deserialize<std::variant<std::string, Result>>(is);
    auto clientId = su::deserialize<std::optional<IdType>>(is);

    return std::holds_alternative<std::string>(errOrResult)
        ? Response{requestId, std::get<std::string>(errOrResult), clientId}
        : Response{requestId, std::get<Result>(errOrResult), clientId};
}

void Serialization::serialize(Response const& response, std::ostream& os)
{
    su::serialize(response.mImpl->mRequestId, os);
    su::serialize(response.mImpl->mErrOrResult, os);
    su::serialize(response.mImpl->mClientId, os);
}

size_t Serialization::serializedSize(Response const& response)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(response.mImpl->mRequestId);
    totalSize += su::serializedSize(response.mImpl->mErrOrResult);
    totalSize += su::serializedSize(response.mImpl->mClientId);
    return totalSize;
}

// Vector of responses
std::vector<Response> Serialization::deserializeResponses(std::vector<char>& buffer)
{
    std::vector<Response> responses;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numResponses = su::deserialize<std::size_t>(is);
    for (std::size_t resp = 0; resp < numResponses; ++resp)
    {
        responses.emplace_back(Serialization::deserializeResponse(is));
    }
    return responses;
}

std::vector<char> Serialization::serialize(std::vector<Response> const& responses)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& response : responses)
    {
        totalSize += su::serializedSize(response);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    su::serialize(responses.size(), os);
    for (auto const& response : responses)
    {
        su::serialize(response, os);
    }
    return buffer;
}

AdditionalOutput Serialization::deserializeAdditionalOutput(std::istream& is)
{
    return {
        Serialization::deserializeString(is),
        Serialization::deserializeTensor(is),
    };
}

void Serialization::serialize(AdditionalOutput const& additionalOutput, std::ostream& os)
{
    su::serialize(additionalOutput.name, os);
    su::serialize(additionalOutput.output, os);
}

size_t serializedSize(AdditionalOutput const& additionalOutput)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(additionalOutput.name);
    totalSize += su::serializedSize(additionalOutput.output);
    return totalSize;
}

// ExecutorConfig
ExecutorConfig Serialization::deserializeExecutorConfig(std::istream& is)
{
    auto maxBeamWidth = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxBeamWidth)>(is);
    auto maxBatchSize = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxBatchSize)>(is);
    auto maxNumTokens = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxNumTokens)>(is);
    auto schedulerConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getSchedulerConfig)>(is);
    auto kvCacheConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getKvCacheConfig)>(is);
    auto enableChunkedContext = su::deserializeWithGetterType<decltype(&ExecutorConfig::getEnableChunkedContext)>(is);
    auto normalizeLogProbs = su::deserializeWithGetterType<decltype(&ExecutorConfig::getNormalizeLogProbs)>(is);
    auto iterStatsMaxIterations
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getIterStatsMaxIterations)>(is);
    auto requestStatsMaxIterations
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getRequestStatsMaxIterations)>(is);
    auto batchingType = su::deserializeWithGetterType<decltype(&ExecutorConfig::getBatchingType)>(is);
    auto parallelConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getParallelConfig)>(is);
    auto peftCacheConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getPeftCacheConfig)>(is);
    auto decodingConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getDecodingConfig)>(is);
    auto useGpuDirectStorage = su::deserializeWithGetterType<decltype(&ExecutorConfig::getUseGpuDirectStorage)>(is);
    auto gpuWeightsPercent = su::deserializeWithGetterType<decltype(&ExecutorConfig::getGpuWeightsPercent)>(is);
    auto maxQueueSize = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxQueueSize)>(is);
    auto extendedRuntimePerfKnobConfig
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getExtendedRuntimePerfKnobConfig)>(is);
    auto debugConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getDebugConfig)>(is);
    auto recvPollPeriodMs = su::deserializeWithGetterType<decltype(&ExecutorConfig::getRecvPollPeriodMs)>(is);
    auto maxSeqIdleMicroseconds
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxSeqIdleMicroseconds)>(is);
    auto specDecConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getSpecDecConfig)>(is);
    auto guidedDecodingConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getGuidedDecodingConfig)>(is);
    auto additionalModelOutputs
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getAdditionalModelOutputs)>(is);
    auto cacheTransceiverConfig
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getCacheTransceiverConfig)>(is);
    auto gatherGenerationLogits
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getGatherGenerationLogits)>(is);
    auto promptTableOffloading = su::deserializeWithGetterType<decltype(&ExecutorConfig::getPromptTableOffloading)>(is);
    auto enableTrtOverlap = su::deserializeWithGetterType<decltype(&ExecutorConfig::getEnableTrtOverlap)>(is);

    return ExecutorConfig{maxBeamWidth, schedulerConfig, kvCacheConfig, enableChunkedContext, normalizeLogProbs,
        iterStatsMaxIterations, requestStatsMaxIterations, batchingType, maxBatchSize, maxNumTokens, parallelConfig,
        peftCacheConfig, std::nullopt, decodingConfig, useGpuDirectStorage, gpuWeightsPercent, maxQueueSize,
        extendedRuntimePerfKnobConfig, debugConfig, recvPollPeriodMs, maxSeqIdleMicroseconds, specDecConfig,
        guidedDecodingConfig, additionalModelOutputs, cacheTransceiverConfig, gatherGenerationLogits,
        promptTableOffloading, enableTrtOverlap};
}

size_t Serialization::serializedSize(ExecutorConfig const& executorConfig)
{
    TLLM_CHECK_WITH_INFO(!executorConfig.getLogitsPostProcessorConfig().has_value(),
        "Serialization of executorConfig with logitsPostProcessor is currently not supported.");

    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += su::serializedSize(executorConfig.getMaxBeamWidth());
    totalSize += su::serializedSize(executorConfig.getMaxBatchSize());
    totalSize += su::serializedSize(executorConfig.getMaxNumTokens());
    totalSize += su::serializedSize(executorConfig.getSchedulerConfig());
    totalSize += su::serializedSize(executorConfig.getKvCacheConfig());
    totalSize += su::serializedSize(executorConfig.getEnableChunkedContext());
    totalSize += su::serializedSize(executorConfig.getNormalizeLogProbs());
    totalSize += su::serializedSize(executorConfig.getIterStatsMaxIterations());
    totalSize += su::serializedSize(executorConfig.getRequestStatsMaxIterations());
    totalSize += su::serializedSize(executorConfig.getBatchingType());
    totalSize += su::serializedSize(executorConfig.getParallelConfig());
    totalSize += su::serializedSize(executorConfig.getPeftCacheConfig());
    totalSize += su::serializedSize(executorConfig.getDecodingConfig());
    totalSize += su::serializedSize(executorConfig.getUseGpuDirectStorage());
    totalSize += su::serializedSize(executorConfig.getGpuWeightsPercent());
    totalSize += su::serializedSize(executorConfig.getMaxQueueSize());
    totalSize += su::serializedSize(executorConfig.getExtendedRuntimePerfKnobConfig());
    totalSize += su::serializedSize(executorConfig.getDebugConfig());
    totalSize += su::serializedSize(executorConfig.getRecvPollPeriodMs());
    totalSize += su::serializedSize(executorConfig.getMaxSeqIdleMicroseconds());
    totalSize += su::serializedSize(executorConfig.getSpecDecConfig());
    totalSize += su::serializedSize(executorConfig.getGuidedDecodingConfig());
    totalSize += su::serializedSize(executorConfig.getAdditionalModelOutputs());
    totalSize += su::serializedSize(executorConfig.getCacheTransceiverConfig());
    totalSize += su::serializedSize(executorConfig.getGatherGenerationLogits());
    totalSize += su::serializedSize(executorConfig.getPromptTableOffloading());
    totalSize += su::serializedSize(executorConfig.getEnableTrtOverlap());

    return totalSize;
}

void Serialization::serialize(ExecutorConfig const& executorConfig, std::ostream& os)
{
    TLLM_CHECK_WITH_INFO(!executorConfig.getLogitsPostProcessorConfig().has_value(),
        "Serialization of executorConfig with logitsPostProcessor is currently not supported.");

    su::serialize(executorConfig.getMaxBeamWidth(), os);
    su::serialize(executorConfig.getMaxBatchSize(), os);
    su::serialize(executorConfig.getMaxNumTokens(), os);
    su::serialize(executorConfig.getSchedulerConfig(), os);
    su::serialize(executorConfig.getKvCacheConfig(), os);
    su::serialize(executorConfig.getEnableChunkedContext(), os);
    su::serialize(executorConfig.getNormalizeLogProbs(), os);
    su::serialize(executorConfig.getIterStatsMaxIterations(), os);
    su::serialize(executorConfig.getRequestStatsMaxIterations(), os);
    su::serialize(executorConfig.getBatchingType(), os);
    su::serialize(executorConfig.getParallelConfig(), os);
    su::serialize(executorConfig.getPeftCacheConfig(), os);
    su::serialize(executorConfig.getDecodingConfig(), os);
    su::serialize(executorConfig.getUseGpuDirectStorage(), os);
    su::serialize(executorConfig.getGpuWeightsPercent(), os);
    su::serialize(executorConfig.getMaxQueueSize(), os);
    su::serialize(executorConfig.getExtendedRuntimePerfKnobConfig(), os);
    su::serialize(executorConfig.getDebugConfig(), os);
    su::serialize(executorConfig.getRecvPollPeriodMs(), os);
    su::serialize(executorConfig.getMaxSeqIdleMicroseconds(), os);
    su::serialize(executorConfig.getSpecDecConfig(), os);
    su::serialize(executorConfig.getGuidedDecodingConfig(), os);
    su::serialize(executorConfig.getAdditionalModelOutputs(), os);
    su::serialize(executorConfig.getCacheTransceiverConfig(), os);
    su::serialize(executorConfig.getGatherGenerationLogits(), os);
    su::serialize(executorConfig.getPromptTableOffloading(), os);
    su::serialize(executorConfig.getEnableTrtOverlap(), os);
}

// KvCacheConfig
KvCacheConfig Serialization::deserializeKvCacheConfig(std::istream& is)
{
    auto enableBlockReuse = su::deserialize<bool>(is);
    auto enablePartialReuse = su::deserialize<bool>(is);
    auto copyOnPartialReuse = su::deserialize<bool>(is);
    auto maxTokens = su::deserialize<std::optional<SizeType32>>(is);
    auto maxAttentionWindowVec = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto sinkTokenLength = su::deserialize<std::optional<SizeType32>>(is);
    auto freeGpuMemoryFraction = su::deserialize<std::optional<FloatType>>(is);
    auto hostCacheSize = su::deserialize<std::optional<size_t>>(is);
    auto onboardBlocks = su::deserialize<bool>(is);
    auto crossKvCacheFraction = su::deserialize<std::optional<FloatType>>(is);
    auto secondaryOffloadMinPriority = su::deserialize<std::optional<executor::RetentionPriority>>(is);
    auto eventBufferMaxSize = su::deserialize<size_t>(is);
    auto useUvm = su::deserialize<bool>(is);
    auto attentionDpEventsGatherPeriodMs = su::deserialize<SizeType32>(is);

    return KvCacheConfig{enableBlockReuse, maxTokens, maxAttentionWindowVec, sinkTokenLength, freeGpuMemoryFraction,
        hostCacheSize, onboardBlocks, crossKvCacheFraction, secondaryOffloadMinPriority, eventBufferMaxSize,
        enablePartialReuse, copyOnPartialReuse, useUvm, attentionDpEventsGatherPeriodMs};
}

void Serialization::serialize(KvCacheConfig const& kvCacheConfig, std::ostream& os)
{
    su::serialize(kvCacheConfig.getEnableBlockReuse(), os);
    su::serialize(kvCacheConfig.getEnablePartialReuse(), os);
    su::serialize(kvCacheConfig.getCopyOnPartialReuse(), os);
    su::serialize(kvCacheConfig.getMaxTokens(), os);
    su::serialize(kvCacheConfig.getMaxAttentionWindowVec(), os);
    su::serialize(kvCacheConfig.getSinkTokenLength(), os);
    su::serialize(kvCacheConfig.getFreeGpuMemoryFraction(), os);
    su::serialize(kvCacheConfig.getHostCacheSize(), os);
    su::serialize(kvCacheConfig.getOnboardBlocks(), os);
    su::serialize(kvCacheConfig.getCrossKvCacheFraction(), os);
    su::serialize(kvCacheConfig.getSecondaryOffloadMinPriority(), os);
    su::serialize(kvCacheConfig.getEventBufferMaxSize(), os);
    su::serialize(kvCacheConfig.getUseUvm(), os);
    su::serialize(kvCacheConfig.getAttentionDpEventsGatherPeriodMs(), os);
}

size_t Serialization::serializedSize(KvCacheConfig const& kvCacheConfig)
{

    size_t totalSize = 0;
    totalSize += su::serializedSize(kvCacheConfig.getEnableBlockReuse());
    totalSize += su::serializedSize(kvCacheConfig.getEnablePartialReuse());
    totalSize += su::serializedSize(kvCacheConfig.getCopyOnPartialReuse());
    totalSize += su::serializedSize(kvCacheConfig.getMaxTokens());
    totalSize += su::serializedSize(kvCacheConfig.getMaxAttentionWindowVec());
    totalSize += su::serializedSize(kvCacheConfig.getSinkTokenLength());
    totalSize += su::serializedSize(kvCacheConfig.getFreeGpuMemoryFraction());
    totalSize += su::serializedSize(kvCacheConfig.getHostCacheSize());
    totalSize += su::serializedSize(kvCacheConfig.getOnboardBlocks());
    totalSize += su::serializedSize(kvCacheConfig.getCrossKvCacheFraction());
    totalSize += su::serializedSize(kvCacheConfig.getSecondaryOffloadMinPriority());
    totalSize += su::serializedSize(kvCacheConfig.getEventBufferMaxSize());
    totalSize += su::serializedSize(kvCacheConfig.getUseUvm());
    totalSize += su::serializedSize(kvCacheConfig.getAttentionDpEventsGatherPeriodMs());
    return totalSize;
}

// DynamicBatchConfig
DynamicBatchConfig Serialization::deserializeDynamicBatchConfig(std::istream& is)
{
    auto enableBatchSizeTuning = su::deserialize<bool>(is);
    auto enableMaxNumTokensTuning = su::deserialize<bool>(is);
    auto dynamicBatchMovingAvgWindow = su::deserialize<SizeType32>(is);
    return DynamicBatchConfig{enableBatchSizeTuning, enableMaxNumTokensTuning, dynamicBatchMovingAvgWindow};
}

void Serialization::serialize(DynamicBatchConfig const& DynamicBatchConfig, std::ostream& os)
{
    su::serialize(DynamicBatchConfig.getEnableBatchSizeTuning(), os);
    su::serialize(DynamicBatchConfig.getEnableMaxNumTokensTuning(), os);
    su::serialize(DynamicBatchConfig.getDynamicBatchMovingAverageWindow(), os);
}

size_t Serialization::serializedSize(DynamicBatchConfig const& DynamicBatchConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(DynamicBatchConfig.getEnableBatchSizeTuning());
    totalSize += su::serializedSize(DynamicBatchConfig.getEnableMaxNumTokensTuning());
    totalSize += su::serializedSize(DynamicBatchConfig.getDynamicBatchMovingAverageWindow());
    return totalSize;
}

// SchedulerConfig
SchedulerConfig Serialization::deserializeSchedulerConfig(std::istream& is)
{
    auto capacitySchedulerPolicy = su::deserialize<CapacitySchedulerPolicy>(is);
    auto contextChunkingPolicy = su::deserialize<std::optional<ContextChunkingPolicy>>(is);
    auto dynamicBatchConfig = su::deserialize<std::optional<DynamicBatchConfig>>(is);
    return SchedulerConfig{capacitySchedulerPolicy, contextChunkingPolicy, dynamicBatchConfig};
}

void Serialization::serialize(SchedulerConfig const& schedulerConfig, std::ostream& os)
{
    su::serialize(schedulerConfig.getCapacitySchedulerPolicy(), os);
    su::serialize(schedulerConfig.getContextChunkingPolicy(), os);
    su::serialize(schedulerConfig.getDynamicBatchConfig(), os);
}

size_t Serialization::serializedSize(SchedulerConfig const& schedulerConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(schedulerConfig.getCapacitySchedulerPolicy());
    totalSize += su::serializedSize(schedulerConfig.getContextChunkingPolicy());
    totalSize += su::serializedSize(schedulerConfig.getDynamicBatchConfig());
    return totalSize;
}

// CacheTransceiverConfig
CacheTransceiverConfig Serialization::deserializeCacheTransceiverConfig(std::istream& is)
{
    auto backendType = su::deserialize<std::optional<CacheTransceiverConfig::BackendType>>(is);
    auto maxTokensInBuffer = su::deserialize<std::optional<size_t>>(is);
    return CacheTransceiverConfig{backendType, maxTokensInBuffer};
}

void Serialization::serialize(CacheTransceiverConfig const& cacheTransceiverConfig, std::ostream& os)
{
    su::serialize(cacheTransceiverConfig.getBackendType(), os);
    su::serialize(cacheTransceiverConfig.getMaxTokensInBuffer(), os);
}

size_t Serialization::serializedSize(CacheTransceiverConfig const& cacheTransceiverConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(cacheTransceiverConfig.getBackendType());
    totalSize += su::serializedSize(cacheTransceiverConfig.getMaxTokensInBuffer());
    return totalSize;
}

// ExtendedRuntimePerfKnobConfig
ExtendedRuntimePerfKnobConfig Serialization::deserializeExtendedRuntimePerfKnobConfig(std::istream& is)
{
    auto multiBlockMode = su::deserialize<bool>(is);
    auto enableContextFMHAFP32Acc = su::deserialize<bool>(is);
    auto cudaGraphMode = su::deserialize<bool>(is);
    auto cudaGraphCacheSize = su::deserialize<SizeType32>(is);
    return ExtendedRuntimePerfKnobConfig{multiBlockMode, enableContextFMHAFP32Acc, cudaGraphMode, cudaGraphCacheSize};
}

void Serialization::serialize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, std::ostream& os)
{
    su::serialize(extendedRuntimePerfKnobConfig.getMultiBlockMode(), os);
    su::serialize(extendedRuntimePerfKnobConfig.getEnableContextFMHAFP32Acc(), os);
    su::serialize(extendedRuntimePerfKnobConfig.getCudaGraphMode(), os);
    su::serialize(extendedRuntimePerfKnobConfig.getCudaGraphCacheSize(), os);
}

size_t Serialization::serializedSize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getMultiBlockMode());
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getEnableContextFMHAFP32Acc());
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getCudaGraphMode());
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getCudaGraphCacheSize());
    return totalSize;
}

// ParallelConfig
ParallelConfig Serialization::deserializeParallelConfig(std::istream& is)
{
    auto commType = su::deserialize<CommunicationType>(is);
    auto commMode = su::deserialize<CommunicationMode>(is);
    auto deviceIds = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto participantids = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto orchestratorConfig = su::deserialize<std::optional<OrchestratorConfig>>(is);
    auto numNodes = su::deserialize<std::optional<SizeType32>>(is);

    return ParallelConfig{commType, commMode, deviceIds, participantids, orchestratorConfig, numNodes};
}

void Serialization::serialize(ParallelConfig const& parallelConfig, std::ostream& os)
{
    su::serialize(parallelConfig.getCommunicationType(), os);
    su::serialize(parallelConfig.getCommunicationMode(), os);
    su::serialize(parallelConfig.getDeviceIds(), os);
    su::serialize(parallelConfig.getParticipantIds(), os);
    su::serialize(parallelConfig.getOrchestratorConfig(), os);
    su::serialize(parallelConfig.getNumNodes(), os);
}

size_t Serialization::serializedSize(ParallelConfig const& parallelConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(parallelConfig.getCommunicationType());
    totalSize += su::serializedSize(parallelConfig.getCommunicationMode());
    totalSize += su::serializedSize(parallelConfig.getDeviceIds());
    totalSize += su::serializedSize(parallelConfig.getParticipantIds());
    totalSize += su::serializedSize(parallelConfig.getOrchestratorConfig());
    totalSize += su::serializedSize(parallelConfig.getNumNodes());
    return totalSize;
}

// PeftCacheConfig
PeftCacheConfig Serialization::deserializePeftCacheConfig(std::istream& is)
{
    auto numHostModuleLayer = su::deserialize<SizeType32>(is);
    auto numDeviceModuleLayer = su::deserialize<SizeType32>(is);
    auto optimalAdapterSize = su::deserialize<SizeType32>(is);
    auto maxAdapterSize = su::deserialize<SizeType32>(is);
    auto numPutWorkers = su::deserialize<SizeType32>(is);
    auto numEnsureWorkers = su::deserialize<SizeType32>(is);
    auto numCopyStreams = su::deserialize<SizeType32>(is);
    auto maxPagesPerBlockHost = su::deserialize<SizeType32>(is);
    auto maxPagesPerBlockDevice = su::deserialize<SizeType32>(is);
    auto deviceCachePercent = su::deserialize<std::optional<FloatType>>(is);
    auto hostCacheSize = su::deserialize<std::optional<size_t>>(is);

    return PeftCacheConfig{numHostModuleLayer, numDeviceModuleLayer, optimalAdapterSize, maxAdapterSize, numPutWorkers,
        numEnsureWorkers, numCopyStreams, maxPagesPerBlockHost, maxPagesPerBlockDevice, deviceCachePercent,
        hostCacheSize};
}

void Serialization::serialize(PeftCacheConfig const& peftCacheConfig, std::ostream& os)
{
    su::serialize(peftCacheConfig.getNumHostModuleLayer(), os);
    su::serialize(peftCacheConfig.getNumDeviceModuleLayer(), os);
    su::serialize(peftCacheConfig.getOptimalAdapterSize(), os);
    su::serialize(peftCacheConfig.getMaxAdapterSize(), os);
    su::serialize(peftCacheConfig.getNumPutWorkers(), os);
    su::serialize(peftCacheConfig.getNumEnsureWorkers(), os);
    su::serialize(peftCacheConfig.getNumCopyStreams(), os);
    su::serialize(peftCacheConfig.getMaxPagesPerBlockHost(), os);
    su::serialize(peftCacheConfig.getMaxPagesPerBlockDevice(), os);
    su::serialize(peftCacheConfig.getDeviceCachePercent(), os);
    su::serialize(peftCacheConfig.getHostCacheSize(), os);
}

size_t Serialization::serializedSize(PeftCacheConfig const& peftCacheConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(peftCacheConfig.getNumHostModuleLayer());
    totalSize += su::serializedSize(peftCacheConfig.getNumDeviceModuleLayer());
    totalSize += su::serializedSize(peftCacheConfig.getOptimalAdapterSize());
    totalSize += su::serializedSize(peftCacheConfig.getMaxAdapterSize());
    totalSize += su::serializedSize(peftCacheConfig.getNumPutWorkers());
    totalSize += su::serializedSize(peftCacheConfig.getNumEnsureWorkers());
    totalSize += su::serializedSize(peftCacheConfig.getNumCopyStreams());
    totalSize += su::serializedSize(peftCacheConfig.getMaxPagesPerBlockHost());
    totalSize += su::serializedSize(peftCacheConfig.getMaxPagesPerBlockDevice());
    totalSize += su::serializedSize(peftCacheConfig.getDeviceCachePercent());
    totalSize += su::serializedSize(peftCacheConfig.getHostCacheSize());
    return totalSize;
}

// OrchestratorConfig
OrchestratorConfig Serialization::deserializeOrchestratorConfig(std::istream& is)
{
    auto isOrchestrator = su::deserialize<bool>(is);
    auto path = su::deserialize<std::string>(is);
    auto spawnProcesses = su::deserialize<bool>(is);
    // Note we ignore mpiComm since we don't need to exchange it
    return OrchestratorConfig{isOrchestrator, path, nullptr, spawnProcesses};
}

void Serialization::serialize(OrchestratorConfig const& orchestratorConfig, std::ostream& os)
{
    su::serialize(orchestratorConfig.getIsOrchestrator(), os);
    su::serialize(orchestratorConfig.getWorkerExecutablePath(), os);
    su::serialize(orchestratorConfig.getSpawnProcesses(), os);
}

size_t Serialization::serializedSize(OrchestratorConfig const& orchestratorConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(orchestratorConfig.getIsOrchestrator());
    totalSize += su::serializedSize(orchestratorConfig.getWorkerExecutablePath());
    totalSize += su::serializedSize(orchestratorConfig.getSpawnProcesses());
    return totalSize;
}

// DecodingMode
DecodingMode Serialization::deserializeDecodingMode(std::istream& is)
{
    auto mode = su::deserialize<DecodingMode::UnderlyingType>(is);

    return DecodingMode{mode};
}

void Serialization::serialize(DecodingMode const& decodingMode, std::ostream& os)
{
    su::serialize(decodingMode.getState(), os);
}

size_t Serialization::serializedSize(DecodingMode const& decodingMode)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(decodingMode.getState());
    return totalSize;
}

// LookaheadDecodingConfig
LookaheadDecodingConfig Serialization::deserializeLookaheadDecodingConfig(std::istream& is)
{
    auto ngramSize = su::deserialize<SizeType32>(is);
    auto windowSize = su::deserialize<SizeType32>(is);
    auto verificationSetSize = su::deserialize<SizeType32>(is);

    return LookaheadDecodingConfig{windowSize, ngramSize, verificationSetSize};
}

void Serialization::serialize(LookaheadDecodingConfig const& lookaheadDecodingConfig, std::ostream& os)
{
    su::serialize(lookaheadDecodingConfig.getNgramSize(), os);
    su::serialize(lookaheadDecodingConfig.getWindowSize(), os);
    su::serialize(lookaheadDecodingConfig.getVerificationSetSize(), os);
}

size_t Serialization::serializedSize(LookaheadDecodingConfig const& lookaheadDecodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(lookaheadDecodingConfig.getNgramSize());
    totalSize += su::serializedSize(lookaheadDecodingConfig.getWindowSize());
    totalSize += su::serializedSize(lookaheadDecodingConfig.getVerificationSetSize());
    return totalSize;
}

// EagleConfig
EagleConfig Serialization::deserializeEagleConfig(std::istream& is)
{
    auto eagleChoices = su::deserialize<std::optional<EagleChoices>>(is);
    auto isGreedySampling = su::deserialize<bool>(is);
    auto posteriorThreshold = su::deserialize<std::optional<float>>(is);
    auto useDynamicTree = su::deserialize<bool>(is);
    auto dynamicTreeMaxTopK = su::deserialize<std::optional<SizeType32>>(is);

    return EagleConfig{eagleChoices, isGreedySampling, posteriorThreshold, useDynamicTree, dynamicTreeMaxTopK};
}

void Serialization::serialize(EagleConfig const& eagleConfig, std::ostream& os)
{
    su::serialize(eagleConfig.getEagleChoices(), os);
    su::serialize(eagleConfig.isGreedySampling(), os);
    su::serialize(eagleConfig.getPosteriorThreshold(), os);
    su::serialize(eagleConfig.useDynamicTree(), os);
    su::serialize(eagleConfig.getDynamicTreeMaxTopK(), os);
}

size_t Serialization::serializedSize(EagleConfig const& eagleConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(eagleConfig.getEagleChoices());
    totalSize += su::serializedSize(eagleConfig.isGreedySampling());
    totalSize += su::serializedSize(eagleConfig.getPosteriorThreshold());
    totalSize += su::serializedSize(eagleConfig.useDynamicTree());
    totalSize += su::serializedSize(eagleConfig.getDynamicTreeMaxTopK());

    return totalSize;
}

// SpeculativeDecodingConfig
SpeculativeDecodingConfig Serialization::deserializeSpeculativeDecodingConfig(std::istream& is)
{
    auto fastLogits = su::deserialize<decltype(SpeculativeDecodingConfig::fastLogits)>(is);
    return SpeculativeDecodingConfig(fastLogits);
}

void Serialization::serialize(SpeculativeDecodingConfig const& specDecConfig, std::ostream& os)
{
    su::serialize(specDecConfig.fastLogits, os);
}

size_t Serialization::serializedSize(SpeculativeDecodingConfig const& specDecConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(specDecConfig.fastLogits);
    return totalSize;
}

// GuidedDecodingConfig
GuidedDecodingConfig Serialization::deserializeGuidedDecodingConfig(std::istream& is)
{
    auto backend = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getBackend)>(is);
    auto encodedVocab = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getEncodedVocab)>(is);
    auto tokenizerStr = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getTokenizerStr)>(is);
    auto stopTokenIds = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getStopTokenIds)>(is);
    return GuidedDecodingConfig(backend, encodedVocab, tokenizerStr, stopTokenIds);
}

void Serialization::serialize(GuidedDecodingConfig const& guidedDecodingConfig, std::ostream& os)
{
    su::serialize(guidedDecodingConfig.getBackend(), os);
    su::serialize(guidedDecodingConfig.getEncodedVocab(), os);
    su::serialize(guidedDecodingConfig.getTokenizerStr(), os);
    su::serialize(guidedDecodingConfig.getStopTokenIds(), os);
}

size_t Serialization::serializedSize(GuidedDecodingConfig const& guidedDecodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(guidedDecodingConfig.getBackend());
    totalSize += su::serializedSize(guidedDecodingConfig.getEncodedVocab());
    totalSize += su::serializedSize(guidedDecodingConfig.getTokenizerStr());
    totalSize += su::serializedSize(guidedDecodingConfig.getStopTokenIds());
    return totalSize;
}

// GuidedDecodingParams
GuidedDecodingParams Serialization::deserializeGuidedDecodingParams(std::istream& is)
{
    auto guideType = su::deserializeWithGetterType<decltype(&GuidedDecodingParams::getGuideType)>(is);
    auto guide = su::deserializeWithGetterType<decltype(&GuidedDecodingParams::getGuide)>(is);
    return GuidedDecodingParams(guideType, guide);
}

void Serialization::serialize(GuidedDecodingParams const& guidedDecodingParams, std::ostream& os)
{
    su::serialize(guidedDecodingParams.getGuideType(), os);
    su::serialize(guidedDecodingParams.getGuide(), os);
}

size_t Serialization::serializedSize(GuidedDecodingParams const& guidedDecodingParams)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(guidedDecodingParams.getGuideType());
    totalSize += su::serializedSize(guidedDecodingParams.getGuide());
    return totalSize;
}

// KV Cache Retention Config
std::optional<size_t> durationToInt(std::optional<std::chrono::milliseconds> const& duration)
{
    return duration.has_value() ? std::optional(static_cast<size_t>(duration->count())) : std::nullopt;
}

std::optional<std::chrono::milliseconds> intToDuration(std::optional<size_t> const& duration)
{
    return duration.has_value() ? std::optional(std::chrono::milliseconds(*duration)) : std::nullopt;
}

KvCacheRetentionConfig Serialization::deserializeKvCacheRetentionConfig(std::istream& is)
{
    auto tokenRangeRetentionPriorities
        = su::deserialize<std::vector<executor::KvCacheRetentionConfig::TokenRangeRetentionConfig>>(is);
    auto decodePriority = su::deserialize<executor::RetentionPriority>(is);
    auto decodeDurationMs = intToDuration(su::deserialize<std::optional<size_t>>(is));
    auto transferMode = su::deserialize<executor::KvCacheTransferMode>(is);
    auto directory = su::deserialize<std::string>(is);

    return KvCacheRetentionConfig{
        tokenRangeRetentionPriorities, decodePriority, decodeDurationMs, transferMode, directory};
}

void Serialization::serialize(KvCacheRetentionConfig const& kvCacheRetentionConfig, std::ostream& os)
{
    su::serialize(kvCacheRetentionConfig.getTokenRangeRetentionConfigs(), os);
    su::serialize(kvCacheRetentionConfig.getDecodeRetentionPriority(), os);
    su::serialize(durationToInt(kvCacheRetentionConfig.getDecodeDurationMs()), os);
    su::serialize(kvCacheRetentionConfig.getTransferMode(), os);
    su::serialize(kvCacheRetentionConfig.getDirectory(), os);
}

size_t Serialization::serializedSize(KvCacheRetentionConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.getTokenRangeRetentionConfigs());
    totalSize += su::serializedSize(config.getDecodeRetentionPriority());
    totalSize += su::serializedSize(durationToInt(config.getDecodeDurationMs()));
    totalSize += su::serializedSize(config.getTransferMode());
    totalSize += su::serializedSize(config.getDirectory());
    return totalSize;
}

// TokenRangeRetentionConfig
KvCacheRetentionConfig::TokenRangeRetentionConfig Serialization::deserializeTokenRangeRetentionConfig(std::istream& is)
{
    auto tokenStart = su::deserialize<SizeType32>(is);
    auto tokenEnd = su::deserialize<std::optional<SizeType32>>(is);
    auto priority = static_cast<RetentionPriority>(su::deserialize<RetentionPriority>(is));
    auto durationMs = intToDuration(su::deserialize<std::optional<size_t>>(is));

    return KvCacheRetentionConfig::TokenRangeRetentionConfig(tokenStart, tokenEnd, priority, durationMs);
}

void Serialization::serialize(
    KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig, std::ostream& os)
{
    su::serialize(tokenRangeRetentionConfig.tokenStart, os);
    su::serialize(tokenRangeRetentionConfig.tokenEnd, os);
    su::serialize(tokenRangeRetentionConfig.priority, os);
    su::serialize(durationToInt(tokenRangeRetentionConfig.durationMs), os);
}

size_t Serialization::serializedSize(KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(tokenRangeRetentionConfig.tokenStart);
    totalSize += su::serializedSize(tokenRangeRetentionConfig.tokenEnd);
    totalSize += su::serializedSize(static_cast<RetentionPriority>(tokenRangeRetentionConfig.priority));
    totalSize += su::serializedSize(durationToInt(tokenRangeRetentionConfig.durationMs));
    return totalSize;
}

// DecodingConfig
DecodingConfig Serialization::deserializeDecodingConfig(std::istream& is)
{
    auto decodingMode = su::deserialize<std::optional<DecodingMode>>(is);
    auto lookaheadDecodingConfig = su::deserialize<std::optional<LookaheadDecodingConfig>>(is);
    auto medusaChoices = su::deserialize<std::optional<MedusaChoices>>(is);
    auto eagleConfig = su::deserialize<std::optional<EagleConfig>>(is);

    return DecodingConfig{decodingMode, lookaheadDecodingConfig, medusaChoices, eagleConfig};
}

void Serialization::serialize(DecodingConfig const& decodingConfig, std::ostream& os)
{
    su::serialize(decodingConfig.getDecodingMode(), os);
    su::serialize(decodingConfig.getLookaheadDecodingConfig(), os);
    su::serialize(decodingConfig.getMedusaChoices(), os);
    su::serialize(decodingConfig.getEagleConfig(), os);
}

size_t Serialization::serializedSize(DecodingConfig const& decodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(decodingConfig.getDecodingMode());
    totalSize += su::serializedSize(decodingConfig.getLookaheadDecodingConfig());
    totalSize += su::serializedSize(decodingConfig.getMedusaChoices());
    totalSize += su::serializedSize(decodingConfig.getEagleConfig());
    return totalSize;
}

// DebugConfig
DebugConfig Serialization::deserializeDebugConfig(std::istream& is)
{
    auto debugInputTensors = su::deserializeWithGetterType<decltype(&DebugConfig::getDebugInputTensors)>(is);
    auto debugOutputTensors = su::deserializeWithGetterType<decltype(&DebugConfig::getDebugOutputTensors)>(is);
    auto debugTensorNames = su::deserialize<std::remove_cv_t<
        std::remove_reference_t<std::invoke_result_t<decltype(&DebugConfig::getDebugTensorNames), DebugConfig>>>>(is);
    auto debugTensorsMaxIterations
        = su::deserializeWithGetterType<decltype(&DebugConfig::getDebugTensorsMaxIterations)>(is);

    return DebugConfig{debugInputTensors, debugOutputTensors, debugTensorNames, debugTensorsMaxIterations};
}

void Serialization::serialize(DebugConfig const& debugConfig, std::ostream& os)
{
    su::serialize(debugConfig.getDebugInputTensors(), os);
    su::serialize(debugConfig.getDebugOutputTensors(), os);
    su::serialize(debugConfig.getDebugTensorNames(), os);
    su::serialize(debugConfig.getDebugTensorsMaxIterations(), os);
}

size_t Serialization::serializedSize(DebugConfig const& debugConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(debugConfig.getDebugInputTensors());
    totalSize += su::serializedSize(debugConfig.getDebugOutputTensors());
    totalSize += su::serializedSize(debugConfig.getDebugTensorNames());
    totalSize += su::serializedSize(debugConfig.getDebugTensorsMaxIterations());
    return totalSize;
}

// KvCacheStats
KvCacheStats Serialization::deserializeKvCacheStats(std::istream& is)
{
    auto maxNumBlocks = su::deserialize<SizeType32>(is);
    auto freeNumBlocks = su::deserialize<SizeType32>(is);
    auto usedNumBlocks = su::deserialize<SizeType32>(is);
    auto tokensPerBlock = su::deserialize<SizeType32>(is);
    auto allocTotalBlocks = su::deserialize<SizeType32>(is);
    auto allocNewBlocks = su::deserialize<SizeType32>(is);
    auto reusedBlocks = su::deserialize<SizeType32>(is);
    auto missedBlocks = su::deserialize<SizeType32>(is);
    auto cacheHitRate = su::deserialize<float>(is);

    return KvCacheStats{maxNumBlocks, freeNumBlocks, usedNumBlocks, tokensPerBlock, allocTotalBlocks, allocNewBlocks,
        reusedBlocks, missedBlocks, cacheHitRate};
}

void Serialization::serialize(KvCacheStats const& kvCacheStats, std::ostream& os)
{
    su::serialize(kvCacheStats.maxNumBlocks, os);
    su::serialize(kvCacheStats.freeNumBlocks, os);
    su::serialize(kvCacheStats.usedNumBlocks, os);
    su::serialize(kvCacheStats.tokensPerBlock, os);
    su::serialize(kvCacheStats.allocTotalBlocks, os);
    su::serialize(kvCacheStats.allocNewBlocks, os);
    su::serialize(kvCacheStats.reusedBlocks, os);
    su::serialize(kvCacheStats.missedBlocks, os);
    su::serialize(kvCacheStats.cacheHitRate, os);
}

size_t Serialization::serializedSize(KvCacheStats const& kvCacheStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(kvCacheStats.maxNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.freeNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.usedNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.tokensPerBlock);
    totalSize += su::serializedSize(kvCacheStats.allocTotalBlocks);
    totalSize += su::serializedSize(kvCacheStats.allocNewBlocks);
    totalSize += su::serializedSize(kvCacheStats.reusedBlocks);
    totalSize += su::serializedSize(kvCacheStats.missedBlocks);
    totalSize += su::serializedSize(kvCacheStats.cacheHitRate);
    return totalSize;
}

// StaticBatchingStats
StaticBatchingStats Serialization::deserializeStaticBatchingStats(std::istream& is)
{
    auto numScheduledRequests = su::deserialize<SizeType32>(is);
    auto numContextRequests = su::deserialize<SizeType32>(is);
    auto numCtxTokens = su::deserialize<SizeType32>(is);
    auto numGenTokens = su::deserialize<SizeType32>(is);
    auto emptyGenSlots = su::deserialize<SizeType32>(is);
    return StaticBatchingStats{numScheduledRequests, numContextRequests, numCtxTokens, numGenTokens, emptyGenSlots};
}

void Serialization::serialize(StaticBatchingStats const& staticBatchingStats, std::ostream& os)
{
    su::serialize(staticBatchingStats.numScheduledRequests, os);
    su::serialize(staticBatchingStats.numContextRequests, os);
    su::serialize(staticBatchingStats.numCtxTokens, os);
    su::serialize(staticBatchingStats.numGenTokens, os);
    su::serialize(staticBatchingStats.emptyGenSlots, os);
}

size_t Serialization::serializedSize(StaticBatchingStats const& staticBatchingStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(staticBatchingStats.numScheduledRequests);
    totalSize += su::serializedSize(staticBatchingStats.numContextRequests);
    totalSize += su::serializedSize(staticBatchingStats.numCtxTokens);
    totalSize += su::serializedSize(staticBatchingStats.numGenTokens);
    totalSize += su::serializedSize(staticBatchingStats.emptyGenSlots);
    return totalSize;
}

// InflightBatchingStats
InflightBatchingStats Serialization::deserializeInflightBatchingStats(std::istream& is)
{
    auto numScheduledRequests = su::deserialize<SizeType32>(is);
    auto numContextRequests = su::deserialize<SizeType32>(is);
    auto numGenRequests = su::deserialize<SizeType32>(is);
    auto numPausedRequests = su::deserialize<SizeType32>(is);
    auto numCtxTokens = su::deserialize<SizeType32>(is);
    auto microBatchId = su::deserialize<SizeType32>(is);
    auto avgNumDecodedTokensPerIter = su::deserialize<float>(is);
    return InflightBatchingStats{numScheduledRequests, numContextRequests, numGenRequests, numPausedRequests,
        numCtxTokens, microBatchId, avgNumDecodedTokensPerIter};
}

void Serialization::serialize(InflightBatchingStats const& inflightBatchingStats, std::ostream& os)
{
    su::serialize(inflightBatchingStats.numScheduledRequests, os);
    su::serialize(inflightBatchingStats.numContextRequests, os);
    su::serialize(inflightBatchingStats.numGenRequests, os);
    su::serialize(inflightBatchingStats.numPausedRequests, os);
    su::serialize(inflightBatchingStats.numCtxTokens, os);
    su::serialize(inflightBatchingStats.microBatchId, os);
    su::serialize(inflightBatchingStats.avgNumDecodedTokensPerIter, os);
}

size_t Serialization::serializedSize(InflightBatchingStats const& inflightBatchingStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(inflightBatchingStats.numScheduledRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numContextRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numGenRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numPausedRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numCtxTokens);
    totalSize += su::serializedSize(inflightBatchingStats.microBatchId);
    totalSize += su::serializedSize(inflightBatchingStats.avgNumDecodedTokensPerIter);
    return totalSize;
}

// SpecDecodingStats
SpecDecodingStats Serialization::deserializeSpecDecodingStats(std::istream& is)
{
    auto numDraftTokens = su::deserialize<SizeType64>(is);
    auto numAcceptedTokens = su::deserialize<SizeType64>(is);
    auto numRequestsWithDraftTokens = su::deserialize<SizeType64>(is);
    auto acceptanceLength = su::deserialize<double>(is);
    auto iterLatencyMS = su::deserialize<double>(is);
    auto draftOverhead = su::deserialize<double>(is);

    return SpecDecodingStats{
        numDraftTokens, numAcceptedTokens, numRequestsWithDraftTokens, acceptanceLength, iterLatencyMS, draftOverhead};
}

void Serialization::serialize(SpecDecodingStats const& state, std::ostream& os)
{
    su::serialize(state.numDraftTokens, os);
    su::serialize(state.numAcceptedTokens, os);
    su::serialize(state.numRequestsWithDraftTokens, os);
    su::serialize(state.acceptanceLength, os);
    su::serialize(state.iterLatencyMS, os);
    su::serialize(state.draftOverhead, os);
}

size_t Serialization::serializedSize(SpecDecodingStats const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.numDraftTokens);
    totalSize += su::serializedSize(state.numAcceptedTokens);
    totalSize += su::serializedSize(state.numRequestsWithDraftTokens);
    totalSize += su::serializedSize(state.acceptanceLength);
    totalSize += su::serializedSize(state.iterLatencyMS);
    totalSize += su::serializedSize(state.draftOverhead);
    return totalSize;
}

// IterationStats

IterationStats Serialization::deserializeIterationStats(std::istream& is)
{
    IterationStats iterStats;
    auto timestamp = su::deserialize<std::string>(is);
    auto iter = su::deserialize<IterationType>(is);
    auto iterLatencyMS = su::deserialize<double>(is);
    auto newActiveRequestsQueueLatencyMS = su::deserialize<double>(is);
    auto numNewActiveRequests = su::deserialize<SizeType32>(is);
    auto numActiveRequests = su::deserialize<SizeType32>(is);
    auto numQueuedRequests = su::deserialize<SizeType32>(is);
    auto numCompletedRequests = su::deserialize<SizeType32>(is);
    auto maxNumActiveRequests = su::deserialize<SizeType32>(is);
    auto maxBatchSizeStatic = su::deserialize<SizeType32>(is);
    auto maxBatchSizeTunerRecommended = su::deserialize<SizeType32>(is);
    auto maxBatchSizeRuntime = su::deserialize<SizeType32>(is);
    auto maxNumTokensStatic = su::deserialize<SizeType32>(is);
    auto maxNumTokensTunerRecommended = su::deserialize<SizeType32>(is);
    auto maxNumTokensRuntime = su::deserialize<SizeType32>(is);
    auto gpuMemUsage = su::deserialize<size_t>(is);
    auto cpuMemUsage = su::deserialize<size_t>(is);
    auto pinnedMemUsage = su::deserialize<size_t>(is);
    auto kvCacheStats = su::deserialize<std::optional<KvCacheStats>>(is);
    auto crossKvCacheStats = su::deserialize<std::optional<KvCacheStats>>(is);
    auto staticBatchingStats = su::deserialize<std::optional<StaticBatchingStats>>(is);
    auto inflightBatchingStats = su::deserialize<std::optional<InflightBatchingStats>>(is);
    auto specDecodingStats = su::deserialize<std::optional<SpecDecodingStats>>(is);

    return IterationStats{timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS, numNewActiveRequests,
        numActiveRequests, numQueuedRequests, numCompletedRequests, maxNumActiveRequests, maxBatchSizeStatic,
        maxBatchSizeTunerRecommended, maxBatchSizeRuntime, maxNumTokensStatic, maxNumTokensTunerRecommended,
        maxNumTokensRuntime, gpuMemUsage, cpuMemUsage, pinnedMemUsage, kvCacheStats, crossKvCacheStats,
        staticBatchingStats, inflightBatchingStats, specDecodingStats};
}

IterationStats Serialization::deserializeIterationStats(std::vector<char>& buffer)
{
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);

    return Serialization::deserializeIterationStats(is);
}

size_t Serialization::serializedSize(IterationStats const& iterStats)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;

    totalSize += su::serializedSize(iterStats.timestamp);
    totalSize += su::serializedSize(iterStats.iter);
    totalSize += su::serializedSize(iterStats.iterLatencyMS);
    totalSize += su::serializedSize(iterStats.newActiveRequestsQueueLatencyMS);
    totalSize += su::serializedSize(iterStats.numNewActiveRequests);
    totalSize += su::serializedSize(iterStats.numActiveRequests);
    totalSize += su::serializedSize(iterStats.numQueuedRequests);
    totalSize += su::serializedSize(iterStats.numCompletedRequests);
    totalSize += su::serializedSize(iterStats.maxNumActiveRequests);
    totalSize += su::serializedSize(iterStats.maxBatchSizeStatic);
    totalSize += su::serializedSize(iterStats.maxBatchSizeTunerRecommended);
    totalSize += su::serializedSize(iterStats.maxBatchSizeRuntime);
    totalSize += su::serializedSize(iterStats.maxNumTokensStatic);
    totalSize += su::serializedSize(iterStats.maxNumTokensTunerRecommended);
    totalSize += su::serializedSize(iterStats.maxNumTokensRuntime);
    totalSize += su::serializedSize(iterStats.gpuMemUsage);
    totalSize += su::serializedSize(iterStats.cpuMemUsage);
    totalSize += su::serializedSize(iterStats.pinnedMemUsage);
    totalSize += su::serializedSize(iterStats.kvCacheStats);
    totalSize += su::serializedSize(iterStats.crossKvCacheStats);
    totalSize += su::serializedSize(iterStats.staticBatchingStats);
    totalSize += su::serializedSize(iterStats.inflightBatchingStats);
    totalSize += su::serializedSize(iterStats.specDecodingStats);

    return totalSize;
}

void Serialization::serialize(IterationStats const& iterStats, std::ostream& os)
{
    su::serialize(iterStats.timestamp, os);
    su::serialize(iterStats.iter, os);
    su::serialize(iterStats.iterLatencyMS, os);
    su::serialize(iterStats.newActiveRequestsQueueLatencyMS, os);
    su::serialize(iterStats.numNewActiveRequests, os);
    su::serialize(iterStats.numActiveRequests, os);
    su::serialize(iterStats.numQueuedRequests, os);
    su::serialize(iterStats.numCompletedRequests, os);
    su::serialize(iterStats.maxNumActiveRequests, os);
    su::serialize(iterStats.maxBatchSizeStatic, os);
    su::serialize(iterStats.maxBatchSizeTunerRecommended, os);
    su::serialize(iterStats.maxBatchSizeRuntime, os);
    su::serialize(iterStats.maxNumTokensStatic, os);
    su::serialize(iterStats.maxNumTokensTunerRecommended, os);
    su::serialize(iterStats.maxNumTokensRuntime, os);
    su::serialize(iterStats.gpuMemUsage, os);
    su::serialize(iterStats.cpuMemUsage, os);
    su::serialize(iterStats.pinnedMemUsage, os);
    su::serialize(iterStats.kvCacheStats, os);
    su::serialize(iterStats.crossKvCacheStats, os);
    su::serialize(iterStats.staticBatchingStats, os);
    su::serialize(iterStats.inflightBatchingStats, os);
    su::serialize(iterStats.specDecodingStats, os);
}

std::vector<char> Serialization::serialize(IterationStats const& iterStats)
{
    auto totalSize = Serialization::serializedSize(iterStats);
    std::vector<char> buffer(totalSize);

    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    Serialization::serialize(iterStats, os);

    return buffer;
}

std::vector<IterationStats> Serialization::deserializeIterationStatsVec(std::vector<char>& buffer)
{
    std::vector<IterationStats> iterStatsVec;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numIterStats = su::deserialize<std::size_t>(is);
    for (std::size_t iterStats = 0; iterStats < numIterStats; ++iterStats)
    {
        iterStatsVec.emplace_back(Serialization::deserializeIterationStats(is));
    }
    return iterStatsVec;
}

std::vector<char> Serialization::serialize(std::vector<IterationStats> const& iterStatsVec)
{
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& iterStats : iterStatsVec)
    {
        totalSize += su::serializedSize(iterStats);
    }
    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);
    su::serialize(iterStatsVec.size(), os);
    for (auto const& iterStats : iterStatsVec)
    {
        su::serialize(iterStats, os);
    }
    return buffer;
}

// DistServinigState

DisServingRequestStats Serialization::deserializeDisServingRequestStats(std::istream& is)
{
    auto kvCacheTransferMs = su::deserialize<double>(is);
    auto kvCacheSize = su::deserialize<size_t>(is);
    return DisServingRequestStats{kvCacheTransferMs, kvCacheSize};
}

void Serialization::serialize(DisServingRequestStats const& stats, std::ostream& os)
{
    su::serialize(stats.kvCacheTransferMS, os);
    su::serialize(stats.kvCacheSize, os);
}

size_t Serialization::serializedSize(DisServingRequestStats const& disServingRequestStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(disServingRequestStats.kvCacheTransferMS);
    totalSize += su::serializedSize(disServingRequestStats.kvCacheSize);
    return totalSize;
}

// ReqeuestStage

RequestStage Serialization::deserializeRequestStage(std::istream& is)
{
    int stage = su::deserialize<int>(is);
    return RequestStage{stage};
}

void Serialization::serialize(RequestStage const& requestStage, std::ostream& os)
{
    su::serialize(static_cast<int>(requestStage), os);
}

size_t Serialization::serializedSize(RequestStage const& requestStage)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(static_cast<int>(requestStage));
    return totalSize;
}

// RequestStats
RequestStats Serialization::deserializeRequestStats(std::istream& is)
{
    auto id = su::deserialize<IdType>(is);
    auto stage = su::deserialize<RequestStage>(is);
    auto contextPrefillPosition = su::deserialize<SizeType32>(is);
    auto numGeneratedTokens = su::deserialize<SizeType32>(is);
    auto avgNumDecodedTokensPerIter = su::deserialize<float>(is);
    auto scheduled = su::deserialize<bool>(is);
    auto paused = su::deserialize<bool>(is);
    auto disServingStats = su::deserialize<std::optional<DisServingRequestStats>>(is);
    auto allocTotalBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto allocNewBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto reusedBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto missedBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto kvCacheHitRatePerRequest = su::deserialize<FloatType>(is);

    return RequestStats{id, stage, contextPrefillPosition, numGeneratedTokens, avgNumDecodedTokensPerIter, scheduled,
        paused, disServingStats, allocTotalBlocksPerRequest, allocNewBlocksPerRequest, reusedBlocksPerRequest,
        missedBlocksPerRequest, kvCacheHitRatePerRequest};
}

void Serialization::serialize(RequestStats const& state, std::ostream& os)
{
    su::serialize(state.id, os);
    su::serialize(state.stage, os);
    su::serialize(state.contextPrefillPosition, os);
    su::serialize(state.numGeneratedTokens, os);
    su::serialize(state.avgNumDecodedTokensPerIter, os);
    su::serialize(state.scheduled, os);
    su::serialize(state.paused, os);
    su::serialize(state.disServingStats, os);
    su::serialize(state.allocTotalBlocksPerRequest, os);
    su::serialize(state.allocNewBlocksPerRequest, os);
    su::serialize(state.reusedBlocksPerRequest, os);
    su::serialize(state.missedBlocksPerRequest, os);
    su::serialize(state.kvCacheHitRatePerRequest, os);
}

size_t Serialization::serializedSize(RequestStats const& state)
{

    size_t totalSize = 0;
    totalSize += su::serializedSize(state.id);
    totalSize += su::serializedSize((state.stage));
    totalSize += su::serializedSize(state.contextPrefillPosition);
    totalSize += su::serializedSize(state.numGeneratedTokens);
    totalSize += su::serializedSize(state.avgNumDecodedTokensPerIter);
    totalSize += su::serializedSize(state.scheduled);
    totalSize += su::serializedSize(state.paused);
    totalSize += su::serializedSize(state.disServingStats);
    totalSize += su::serializedSize(state.allocTotalBlocksPerRequest);
    totalSize += su::serializedSize(state.allocNewBlocksPerRequest);
    totalSize += su::serializedSize(state.reusedBlocksPerRequest);
    totalSize += su::serializedSize(state.missedBlocksPerRequest);
    totalSize += su::serializedSize(state.kvCacheHitRatePerRequest);
    return totalSize;
}

// RequestStatsPerIteration
RequestStatsPerIteration Serialization::deserializeRequestStatsPerIteration(std::istream& is)
{
    auto iter = su::deserialize<IterationType>(is);
    auto requestStats = su::deserialize<std::vector<RequestStats>>(is);

    return RequestStatsPerIteration{iter, requestStats};
}

RequestStatsPerIteration Serialization::deserializeRequestStatsPerIteration(std::vector<char>& buffer)
{
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);

    return Serialization::deserializeRequestStatsPerIteration(is);
}

void Serialization::serialize(RequestStatsPerIteration const& state, std::ostream& os)
{
    su::serialize(state.iter, os);
    su::serialize(state.requestStats, os);
}

std::vector<char> Serialization::serialize(RequestStatsPerIteration const& state)
{
    auto totalSize = Serialization::serializedSize(state);
    std::vector<char> buffer(totalSize);

    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    Serialization::serialize(state, os);

    return buffer;
}

size_t Serialization::serializedSize(RequestStatsPerIteration const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.iter);
    totalSize += su::serializedSize(state.requestStats);
    return totalSize;
}

std::vector<char> Serialization::serialize(std::vector<RequestStatsPerIteration> const& requestStatsVec)
{
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& requestStat : requestStatsVec)
    {
        totalSize += su::serializedSize(requestStat);
    }
    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);
    su::serialize(requestStatsVec.size(), os);
    for (auto const& requestStat : requestStatsVec)
    {
        su::serialize(requestStat, os);
    }
    return buffer;
}

std::vector<RequestStatsPerIteration> Serialization::deserializeRequestStatsPerIterationVec(std::vector<char>& buffer)
{

    std::vector<RequestStatsPerIteration> iterRequestStatsVec;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numIterStats = su::deserialize<std::size_t>(is);
    for (std::size_t iterStats = 0; iterStats < numIterStats; ++iterStats)
    {
        iterRequestStatsVec.emplace_back(Serialization::deserializeRequestStatsPerIteration(is));
    }
    return iterRequestStatsVec;
}

//  KVCacheEvents deque
std::vector<char> Serialization::serialize(std::deque<KVCacheEvent> const& eventQueue)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& event : eventQueue)
    {
        totalSize += su::serializedSize(event);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    su::serialize(eventQueue.size(), os);
    for (auto const& event : eventQueue)
    {
        su::serialize(event, os);
    }
    return buffer;
}

std::deque<KVCacheEvent> Serialization::deserializeKVCacheEvents(std::vector<char>& buffer)
{
    std::deque<KVCacheEvent> kvCacheEvents;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numEvents = su::deserialize<std::size_t>(is);
    for (std::size_t event = 0; event < numEvents; ++event)
    {
        kvCacheEvents.emplace_back(Serialization::deserializeKVCacheEvent(is));
    }
    return kvCacheEvents;
}

//  KVCacheEvent
size_t Serialization::serializedSize(KVCacheEvent const& event)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(event.eventId);
    totalSize += su::serializedSize(event.data);
    totalSize += su::serializedSize(event.windowSize);
    totalSize += su::serializedSize(event.attentionDpRank);
    return totalSize;
}

void Serialization::serialize(KVCacheEvent const& event, std::ostream& os)
{
    su::serialize(event.eventId, os);
    su::serialize(event.data, os);
    su::serialize(event.windowSize, os);
    su::serialize(event.attentionDpRank, os);
}

KVCacheEvent Serialization::deserializeKVCacheEvent(std::istream& is)
{
    auto eventId = su::deserialize<IdType>(is);
    auto data = su::deserialize<KVCacheEventData>(is);
    auto windowSize = su::deserialize<SizeType32>(is);
    auto attentionDpRank = su::deserialize<std::optional<SizeType32>>(is);

    return KVCacheEvent{eventId, data, windowSize, attentionDpRank};
}

// KVCacheCreatedData
size_t Serialization::serializedSize(KVCacheCreatedData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.numBlocksPerCacheLevel);
    return totalSize;
}

void Serialization::serialize(KVCacheCreatedData const& data, std::ostream& os)
{
    su::serialize(data.numBlocksPerCacheLevel, os);
}

KVCacheCreatedData Serialization::deserializeKVCacheCreatedData(std::istream& is)
{
    auto numBlocksPerCacheLevel = su::deserialize<std::vector<SizeType32>>(is);
    return KVCacheCreatedData{numBlocksPerCacheLevel};
}

// KVCacheStoredData
size_t Serialization::serializedSize(KVCacheStoredData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.parentHash);
    totalSize += su::serializedSize(data.blocks);
    return totalSize;
}

void Serialization::serialize(KVCacheStoredData const& data, std::ostream& os)
{
    su::serialize(data.parentHash, os);
    su::serialize(data.blocks, os);
}

KVCacheStoredData Serialization::deserializeKVCacheStoredData(std::istream& is)
{
    auto parentHash = su::deserialize<std::optional<IdType>>(is);
    auto blocks = su::deserialize<std::vector<KVCacheStoredBlockData>>(is);
    return KVCacheStoredData{parentHash, blocks};
}

// KVCacheStoredBlockData
size_t Serialization::serializedSize(KVCacheStoredBlockData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.blockHash);
    totalSize += su::serializedSize(data.tokens);
    totalSize += su::serializedSize(data.loraId);
    totalSize += su::serializedSize(data.cacheLevel);
    totalSize += su::serializedSize(data.priority);
    return totalSize;
}

void Serialization::serialize(KVCacheStoredBlockData const& data, std::ostream& os)
{
    su::serialize(data.blockHash, os);
    su::serialize(data.tokens, os);
    su::serialize(data.loraId, os);
    su::serialize(data.cacheLevel, os);
    su::serialize(data.priority, os);
}

KVCacheStoredBlockData Serialization::deserializeKVCacheStoredBlockData(std::istream& is)
{
    auto blockHash = su::deserialize<IdType>(is);
    auto tokens = su::deserialize<tensorrt_llm::runtime::VecUniqueTokens>(is);
    auto loraId = su::deserialize<std::optional<tensorrt_llm::runtime::LoraTaskIdType>>(is);
    auto cacheLevel = su::deserialize<SizeType32>(is);
    auto priority = su::deserialize<SizeType32>(is);

    return KVCacheStoredBlockData{blockHash, tokens, loraId, cacheLevel, priority};
}

// KVcacheRemovedData

size_t Serialization::serializedSize(KVCacheRemovedData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.blockHashes);
    return totalSize;
}

void Serialization::serialize(KVCacheRemovedData const& data, std::ostream& os)
{
    su::serialize(data.blockHashes, os);
}

KVCacheRemovedData Serialization::deserializeKVCacheRemovedData(std::istream& is)
{
    auto blockHashes = su::deserialize<std::vector<IdType>>(is);
    return KVCacheRemovedData{blockHashes};
}

// KVCacheEventDiff
template <typename T>
size_t Serialization::serializedSize(KVCacheEventDiff<T> const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.oldValue);
    totalSize += su::serializedSize(data.newValue);
    return totalSize;
}

template <typename T>
void Serialization::serialize(KVCacheEventDiff<T> const& data, std::ostream& os)
{
    su::serialize(data.oldValue, os);
    su::serialize(data.newValue, os);
}

template <typename T>
KVCacheEventDiff<T> Serialization::deserializeKVCacheEventDiff(std::istream& is)
{
    auto oldValue = su::deserialize<T>(is);
    auto newValue = su::deserialize<T>(is);
    return KVCacheEventDiff<T>{oldValue, newValue};
}

// KVCacheUpdatedData
size_t Serialization::serializedSize(KVCacheUpdatedData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.blockHash);
    totalSize += su::serializedSize(data.cacheLevel);
    totalSize += su::serializedSize(data.priority);
    return totalSize;
}

void Serialization::serialize(KVCacheUpdatedData const& data, std::ostream& os)
{
    su::serialize(data.blockHash, os);
    su::serialize(data.cacheLevel, os);
    su::serialize(data.priority, os);
}

KVCacheUpdatedData Serialization::deserializeKVCacheUpdatedData(std::istream& is)
{
    auto blockHash = su::deserialize<IdType>(is);
    auto cacheLevel = su::deserialize<std::optional<KVCacheEventDiff<SizeType32>>>(is);
    auto priority = su::deserialize<std::optional<KVCacheEventDiff<SizeType32>>>(is);
    return KVCacheUpdatedData{blockHash, cacheLevel, priority};
}

// UniqueToken
size_t Serialization::serializedSize(tensorrt_llm::runtime::UniqueToken const& token)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(token.tokenId);
    totalSize += su::serializedSize(token.tokenExtraId);
    return totalSize;
}

void Serialization::serialize(tensorrt_llm::runtime::UniqueToken const& token, std::ostream& os)
{
    su::serialize(token.tokenId, os);
    su::serialize(token.tokenExtraId, os);
}

tensorrt_llm::runtime::UniqueToken Serialization::deserializeUniqueToken(std::istream& is)
{
    auto tokenId = su::deserialize<tensorrt_llm::runtime::TokenIdType>(is);
    auto tokenExtraId = su::deserialize<tensorrt_llm::runtime::TokenExtraIdType>(is);
    return tensorrt_llm::runtime::UniqueToken{tokenId, tokenExtraId};
}

// String
std::string Serialization::deserializeString(std::istream& is)
{
    return su::deserialize<std::string>(is);
}

// Bool
bool Serialization::deserializeBool(std::istream& is)
{
    return su::deserialize<bool>(is);
}

// ModelType
ModelType Serialization::deserializeModelType(std::istream& is)
{
    return su::deserialize<ModelType>(is);
}

// BlockKey (KV cache)
size_t Serialization::serializedSize(tensorrt_llm::batch_manager::kv_cache_manager::BlockKey const& key)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(key.usesExtraIds);
    totalSize += su::serializedSize(key.loraTaskId);
    totalSize += su::serializedSize(key.uniqueTokens);
    // std::vector<MmKey> where MmKey is pair<std::array<uint8_t,32>, SizeType32>
    totalSize += su::serializedSize(key.extraKeys);
    return totalSize;
}

void Serialization::serialize(tensorrt_llm::batch_manager::kv_cache_manager::BlockKey const& key, std::ostream& os)
{
    su::serialize(key.usesExtraIds, os);
    su::serialize(key.loraTaskId, os);
    su::serialize(key.uniqueTokens, os);
    su::serialize(key.extraKeys, os);
}

tensorrt_llm::batch_manager::kv_cache_manager::BlockKey Serialization::deserializeBlockKey(std::istream& is)
{
    auto usesExtraIds = su::deserialize<bool>(is);
    auto loraTaskId = su::deserialize<std::optional<tensorrt_llm::batch_manager::kv_cache_manager::LoraTaskIdType>>(is);
    auto uniqueTokens = su::deserialize<std::vector<tensorrt_llm::runtime::UniqueToken>>(is);
    auto extraKeys = su::deserialize<std::vector<tensorrt_llm::batch_manager::kv_cache_manager::MmKey>>(is);
    tensorrt_llm::batch_manager::kv_cache_manager::BlockKey key;
    key.usesExtraIds = usesExtraIds;
    key.loraTaskId = std::move(loraTaskId);
    key.uniqueTokens = std::move(uniqueTokens);
    key.extraKeys = std::move(extraKeys);
    return key;
}

} // namespace tensorrt_llm::executor
