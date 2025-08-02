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

#include "tensorrt_llm/executor/executor.h"

#include <utility>

namespace tensorrt_llm::executor
{

ExecutorConfig::ExecutorConfig(SizeType32 maxBeamWidth, SchedulerConfig schedulerConfig, KvCacheConfig kvCacheConfig,
    bool enableChunkedContext, bool normalizeLogProbs, SizeType32 iterStatsMaxIterations,
    SizeType32 requestStatsMaxIterations, BatchingType batchingType, std::optional<SizeType32> maxBatchSize,
    std::optional<SizeType32> maxNumTokens, std::optional<ParallelConfig> parallelConfig,
    std::optional<PeftCacheConfig> const& peftCacheConfig,
    std::optional<LogitsPostProcessorConfig> logitsPostProcessorConfig, std::optional<DecodingConfig> decodingConfig,
    bool useGpuDirectStorage, float gpuWeightPercent, std::optional<SizeType32> maxQueueSize,
    ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, std::optional<DebugConfig> debugConfig,
    SizeType32 recvPollPeriodMs, uint64_t maxSeqIdleMicroseconds,
    std::optional<SpeculativeDecodingConfig> specDecConfig, std::optional<GuidedDecodingConfig> guidedDecodingConfig,
    std::optional<std::vector<AdditionalModelOutput>> additionalModelOutputs,
    std::optional<CacheTransceiverConfig> cacheTransceiverConfig, bool gatherGenerationLogits,
    bool promptTableOffloading, bool enableTrtOverlap, bool failFastOnAttentionWindowTooLarge)
    : mMaxBeamWidth(maxBeamWidth)
    , mSchedulerConfig(std::move(schedulerConfig))
    , mKvCacheConfig(std::move(kvCacheConfig))
    , mEnableChunkedContext(enableChunkedContext)
    , mNormalizeLogProbs(normalizeLogProbs)
    , mIterStatsMaxIterations(iterStatsMaxIterations)
    , mRequestStatsMaxIterations(requestStatsMaxIterations)
    , mBatchingType(batchingType)
    , mMaxBatchSize(maxBatchSize)
    , mMaxNumTokens(maxNumTokens)
    , mParallelConfig(std::move(parallelConfig))
    , mPeftCacheConfig(peftCacheConfig)
    , mLogitsPostProcessorConfig(std::move(logitsPostProcessorConfig))
    , mDecodingConfig(std::move(decodingConfig))
    , mUseGpuDirectStorage((useGpuDirectStorage))
    , mGpuWeightsPercent(gpuWeightPercent)
    , mMaxQueueSize(maxQueueSize)
    , mExtendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig)
    , mDebugConfig(std::move(debugConfig))
    , mRecvPollPeriodMs(recvPollPeriodMs)
    , mMaxSeqIdleMicroseconds(maxSeqIdleMicroseconds)
    , mSpeculativeDecodingConfig(specDecConfig)
    , mGuidedDecodingConfig(std::move(guidedDecodingConfig))
    , mAdditionalModelOutputs(std::move(additionalModelOutputs))
    , mCacheTransceiverConfig(std::move(cacheTransceiverConfig))
    , mGatherGenerationLogits(gatherGenerationLogits)
    , mPromptTableOffloading(promptTableOffloading)
    , mEnableTrtOverlap(enableTrtOverlap)
    , mFailFastOnAttentionWindowTooLarge(failFastOnAttentionWindowTooLarge)
{
    TLLM_CHECK(iterStatsMaxIterations >= 0);
    TLLM_CHECK(requestStatsMaxIterations >= 0);
    TLLM_CHECK(mMaxBeamWidth > 0);
    TLLM_CHECK(maxSeqIdleMicroseconds > 0);
}

// getters

SizeType32 ExecutorConfig::getMaxBeamWidth() const
{
    return mMaxBeamWidth;
}

SchedulerConfig ExecutorConfig::getSchedulerConfig() const
{
    return mSchedulerConfig;
}

SchedulerConfig& ExecutorConfig::getSchedulerConfigRef()
{
    return mSchedulerConfig;
}

KvCacheConfig ExecutorConfig::getKvCacheConfig() const
{
    return mKvCacheConfig;
}

KvCacheConfig& ExecutorConfig::getKvCacheConfigRef()
{
    return mKvCacheConfig;
}

bool ExecutorConfig::getEnableChunkedContext() const
{
    return mEnableChunkedContext;
}

bool ExecutorConfig::getNormalizeLogProbs() const
{
    return mNormalizeLogProbs;
}

SizeType32 ExecutorConfig::getIterStatsMaxIterations() const
{
    return mIterStatsMaxIterations;
}

SizeType32 ExecutorConfig::getRequestStatsMaxIterations() const
{
    return mRequestStatsMaxIterations;
}

BatchingType ExecutorConfig::getBatchingType() const
{
    return mBatchingType;
}

std::optional<SizeType32> ExecutorConfig::getMaxBatchSize() const
{
    return mMaxBatchSize;
}

std::optional<SizeType32> ExecutorConfig::getMaxNumTokens() const
{
    return mMaxNumTokens;
}

std::optional<ParallelConfig> ExecutorConfig::getParallelConfig() const
{
    return mParallelConfig;
}

std::optional<PeftCacheConfig> ExecutorConfig::getPeftCacheConfig() const
{
    return mPeftCacheConfig;
}

std::optional<LogitsPostProcessorConfig> ExecutorConfig::getLogitsPostProcessorConfig() const
{
    return mLogitsPostProcessorConfig;
}

std::optional<DecodingConfig> ExecutorConfig::getDecodingConfig() const
{
    return mDecodingConfig;
}

bool ExecutorConfig::getUseGpuDirectStorage() const
{
    return mUseGpuDirectStorage;
}

float ExecutorConfig::getGpuWeightsPercent() const
{
    return mGpuWeightsPercent;
}

std::optional<SizeType32> ExecutorConfig::getMaxQueueSize() const
{
    return mMaxQueueSize;
}

ExtendedRuntimePerfKnobConfig ExecutorConfig::getExtendedRuntimePerfKnobConfig() const
{
    return mExtendedRuntimePerfKnobConfig;
}

std::optional<DebugConfig> ExecutorConfig::getDebugConfig() const
{
    return mDebugConfig;
}

SizeType32 ExecutorConfig::getRecvPollPeriodMs() const
{
    return mRecvPollPeriodMs;
}

uint64_t ExecutorConfig::getMaxSeqIdleMicroseconds() const
{
    return mMaxSeqIdleMicroseconds;
}

std::optional<SpeculativeDecodingConfig> ExecutorConfig::getSpecDecConfig() const
{
    return mSpeculativeDecodingConfig;
}

std::optional<GuidedDecodingConfig> ExecutorConfig::getGuidedDecodingConfig() const
{
    return mGuidedDecodingConfig;
}

std::optional<std::vector<AdditionalModelOutput>> ExecutorConfig::getAdditionalModelOutputs() const
{
    return mAdditionalModelOutputs;
}

std::optional<CacheTransceiverConfig> ExecutorConfig::getCacheTransceiverConfig() const
{
    return mCacheTransceiverConfig;
}

bool ExecutorConfig::getGatherGenerationLogits() const
{
    return mGatherGenerationLogits;
}

bool ExecutorConfig::getPromptTableOffloading() const
{
    return mPromptTableOffloading;
}

bool ExecutorConfig::getEnableTrtOverlap() const
{
    return mEnableTrtOverlap;
}

bool ExecutorConfig::getFailFastOnAttentionWindowTooLarge() const
{
    return mFailFastOnAttentionWindowTooLarge;
}

// setters

void ExecutorConfig::setMaxBeamWidth(SizeType32 maxBeamWidth)
{
    mMaxBeamWidth = maxBeamWidth;
    TLLM_CHECK(mMaxBeamWidth > 0);
}

void ExecutorConfig::setMaxBatchSize(SizeType32 maxBatchSize)
{
    mMaxBatchSize = maxBatchSize;
    TLLM_CHECK(mMaxBatchSize > 0);
}

void ExecutorConfig::setMaxNumTokens(SizeType32 maxNumTokens)
{
    mMaxNumTokens = maxNumTokens;
    TLLM_CHECK(mMaxNumTokens > 0);
}

void ExecutorConfig::setSchedulerConfig(SchedulerConfig const& schedulerConfig)
{
    mSchedulerConfig = schedulerConfig;
}

void ExecutorConfig::setKvCacheConfig(KvCacheConfig const& kvCacheConfig)
{
    mKvCacheConfig = kvCacheConfig;
}

void ExecutorConfig::setEnableChunkedContext(bool enableChunkedContext)
{
    mEnableChunkedContext = enableChunkedContext;
}

void ExecutorConfig::setNormalizeLogProbs(bool normalizeLogProbs)
{
    mNormalizeLogProbs = normalizeLogProbs;
}

void ExecutorConfig::setIterStatsMaxIterations(SizeType32 iterStatsMaxIterations)
{
    mIterStatsMaxIterations = iterStatsMaxIterations;
    TLLM_CHECK(mIterStatsMaxIterations >= 0);
}

void ExecutorConfig::setRequestStatsMaxIterations(SizeType32 requestStatsMaxIterations)
{
    mRequestStatsMaxIterations = requestStatsMaxIterations;
    TLLM_CHECK(mRequestStatsMaxIterations >= 0);
}

void ExecutorConfig::setBatchingType(BatchingType batchingType)
{
    mBatchingType = batchingType;
}

void ExecutorConfig::setParallelConfig(ParallelConfig const& parallelConfig)
{
    mParallelConfig = parallelConfig;
}

void ExecutorConfig::setPeftCacheConfig(PeftCacheConfig const& peftCacheConfig)
{
    mPeftCacheConfig = peftCacheConfig;
}

void ExecutorConfig::setLogitsPostProcessorConfig(LogitsPostProcessorConfig const& logitsPostProcessorConfig)
{
    mLogitsPostProcessorConfig = logitsPostProcessorConfig;
}

void ExecutorConfig::setDecodingConfig(DecodingConfig const& decodingConfig)
{
    mDecodingConfig = decodingConfig;
}

void ExecutorConfig::setUseGpuDirectStorage(bool const& useGpuDirectStorage)
{
    mUseGpuDirectStorage = useGpuDirectStorage;
}

void ExecutorConfig::setGpuWeightsPercent(float const& gpuWeightsPercent)
{
    mGpuWeightsPercent = gpuWeightsPercent;
}

void ExecutorConfig::setMaxQueueSize(std::optional<SizeType32> const& maxQueueSize)
{
    mMaxQueueSize = maxQueueSize;
}

void ExecutorConfig::setExtendedRuntimePerfKnobConfig(
    ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig)
{
    mExtendedRuntimePerfKnobConfig = extendedRuntimePerfKnobConfig;
}

void ExecutorConfig::setDebugConfig(DebugConfig const& debugConfig)
{
    mDebugConfig = debugConfig;
}

void ExecutorConfig::setRecvPollPeriodMs(SizeType32 const& recvPollPeriodMs)
{
    mRecvPollPeriodMs = recvPollPeriodMs;
}

void ExecutorConfig::setMaxSeqIdleMicroseconds(uint64_t maxSeqIdleMicroseconds)
{
    mMaxSeqIdleMicroseconds = maxSeqIdleMicroseconds;
    TLLM_CHECK(mMaxSeqIdleMicroseconds > 0);
}

void ExecutorConfig::setSpecDecConfig(SpeculativeDecodingConfig const& specDecConfig)
{
    mSpeculativeDecodingConfig = specDecConfig;
}

void ExecutorConfig::setGuidedDecodingConfig(GuidedDecodingConfig const& guidedDecodingConfig)
{
    mGuidedDecodingConfig = guidedDecodingConfig;
}

void ExecutorConfig::setAdditionalModelOutputs(std::vector<AdditionalModelOutput> const& additionalModelOutputs)
{
    mAdditionalModelOutputs = additionalModelOutputs;
}

void ExecutorConfig::setCacheTransceiverConfig(CacheTransceiverConfig const& cacheTransceiverConfig)
{
    mCacheTransceiverConfig = cacheTransceiverConfig;
}

void ExecutorConfig::setGatherGenerationLogits(bool gatherGenerationLogits)
{
    mGatherGenerationLogits = gatherGenerationLogits;
}

void ExecutorConfig::setPromptTableOffloading(bool promptTableOffloading)
{
    mPromptTableOffloading = promptTableOffloading;
}

void ExecutorConfig::setEnableTrtOverlap(bool enableTrtOverlap)
{
    mEnableTrtOverlap = enableTrtOverlap;
}

void ExecutorConfig::setFailFastOnAttentionWindowTooLarge(bool failFastOnAttentionWindowTooLarge)
{
    mFailFastOnAttentionWindowTooLarge = failFastOnAttentionWindowTooLarge;
}

} // namespace tensorrt_llm::executor
