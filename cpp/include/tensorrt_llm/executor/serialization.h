/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include <istream>
#include <ostream>

namespace tensorrt_llm::executor
{

namespace kv_cache
{
class CommState;
class CacheState;
struct SocketState;
} // namespace kv_cache

class Serialization
{
public:
    // BlockKey (KV cache)
    static size_t serializedSize(tensorrt_llm::batch_manager::kv_cache_manager::BlockKey const& key);
    static void serialize(tensorrt_llm::batch_manager::kv_cache_manager::BlockKey const& key, std::ostream& os);
    static tensorrt_llm::batch_manager::kv_cache_manager::BlockKey deserializeBlockKey(std::istream& is);
    // TimePoint
    [[nodiscard]] static RequestPerfMetrics::TimePoint deserializeTimePoint(std::istream& is);
    static void serialize(RequestPerfMetrics::TimePoint const& tp, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestPerfMetrics::TimePoint const&);

    // RequestPerfMetrics
    [[nodiscard]] static RequestPerfMetrics deserializeRequestPerfMetrics(std::istream& is);
    static void serialize(RequestPerfMetrics const& metrics, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestPerfMetrics const& metrics);

    // SamplingConfig
    [[nodiscard]] static SamplingConfig deserializeSamplingConfig(std::istream& is);
    static void serialize(SamplingConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(SamplingConfig const& config);

    // OutputConfig
    [[nodiscard]] static OutputConfig deserializeOutputConfig(std::istream& is);
    static void serialize(OutputConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(OutputConfig const& config);

    // OutputConfig::AdditionalModelOutput
    [[nodiscard]] static AdditionalModelOutput deserializeAdditionalModelOutput(std::istream& is);
    static void serialize(AdditionalModelOutput const& additionalModelOutput, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(AdditionalModelOutput const& additionalModelOutput);

    // ExternalDraftTokensConfig
    [[nodiscard]] static ExternalDraftTokensConfig deserializeExternalDraftTokensConfig(std::istream& is);
    static void serialize(ExternalDraftTokensConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(ExternalDraftTokensConfig const& config);

    // PromptTuningConfig
    [[nodiscard]] static PromptTuningConfig deserializePromptTuningConfig(std::istream& is);
    static void serialize(PromptTuningConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(PromptTuningConfig const& config);

    // MultimodalInput
    [[nodiscard]] static MultimodalInput deserializeMultimodalInput(std::istream& is);
    static void serialize(MultimodalInput const& multimodalInput, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(MultimodalInput const& multimodalInput);

    // MropeConfig
    [[nodiscard]] static MropeConfig deserializeMropeConfig(std::istream& is);
    static void serialize(MropeConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(MropeConfig const& config);

    // LoraConfig
    [[nodiscard]] static LoraConfig deserializeLoraConfig(std::istream& is);
    static void serialize(LoraConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(LoraConfig const& config);

    // CommState
    [[nodiscard]] static kv_cache::CommState deserializeCommState(std::istream& is);
    static void serialize(kv_cache::CommState const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(kv_cache::CommState const& state);

    // SocketState
    [[nodiscard]] static kv_cache::SocketState deserializeSocketState(std::istream& is);
    static void serialize(kv_cache::SocketState const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(kv_cache::SocketState const& state);

    // AgentState
    [[nodiscard]] static kv_cache::AgentState deserializeAgentState(std::istream& is);
    static void serialize(kv_cache::AgentState const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(kv_cache::AgentState const& state);

    // CacheState
    [[nodiscard]] static kv_cache::CacheState deserializeCacheState(std::istream& is);
    static void serialize(kv_cache::CacheState const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(kv_cache::CacheState const& state);

    // DataTransceiverState
    [[nodiscard]] static DataTransceiverState deserializeDataTransceiverState(std::istream& is);
    [[nodiscard]] static DataTransceiverState deserializeDataTransceiverState(std::vector<char>& buffer);
    static void serialize(DataTransceiverState const& dataTransceiverState, std::ostream& os);
    static std::vector<char> serialize(DataTransceiverState const& dataTransceiverState);
    [[nodiscard]] static size_t serializedSize(DataTransceiverState const& dataTransceiverState);

    // ContextPhaseParams
    [[nodiscard]] static ContextPhaseParams deserializeContextPhaseParams(std::istream& is);
    static void serialize(ContextPhaseParams const& contextPhaseParams, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(ContextPhaseParams const& contextPhaseParams);

    // Request
    [[nodiscard]] static Request deserializeRequest(std::istream& is);
    static void serialize(Request const& request, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Request const& request);

    // Tensor
    [[nodiscard]] static Tensor deserializeTensor(std::istream& is);
    static void serialize(Tensor const& tensor, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Tensor const& tensor);

    // SpeculativeDecodingFastLogitsInfo
    [[nodiscard]] static SpeculativeDecodingFastLogitsInfo deserializeSpecDecFastLogitsInfo(std::istream& is);
    static void serialize(SpeculativeDecodingFastLogitsInfo const& info, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(SpeculativeDecodingFastLogitsInfo const& info);

    // Result
    [[nodiscard]] static Result deserializeResult(std::istream& is);
    static void serialize(Result const& result, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Result const& result);

    // AdditionalOutput
    [[nodiscard]] static AdditionalOutput deserializeAdditionalOutput(std::istream& is);
    static void serialize(AdditionalOutput const& additionalOutput, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(AdditionalOutput const& additionalOutput);

    // Response
    [[nodiscard]] static Response deserializeResponse(std::istream& is);
    static void serialize(Response const& response, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Response const& response);

    // Vector of responses
    static std::vector<Response> deserializeResponses(std::vector<char>& buffer);
    static std::vector<char> serialize(std::vector<Response> const& responses);

    // KvCacheConfig
    static KvCacheConfig deserializeKvCacheConfig(std::istream& is);
    static void serialize(KvCacheConfig const& kvCacheConfig, std::ostream& os);
    static size_t serializedSize(KvCacheConfig const& kvCacheConfig);

    // DynamicBatchConfig
    static DynamicBatchConfig deserializeDynamicBatchConfig(std::istream& is);
    static void serialize(DynamicBatchConfig const& dynamicBatchConfig, std::ostream& os);
    static size_t serializedSize(DynamicBatchConfig const& dynamicBatchConfig);

    // SchedulerConfig
    static SchedulerConfig deserializeSchedulerConfig(std::istream& is);
    static void serialize(SchedulerConfig const& schedulerConfig, std::ostream& os);
    static size_t serializedSize(SchedulerConfig const& schedulerConfig);

    // ExtendedRuntimePerfKnobConfig
    static ExtendedRuntimePerfKnobConfig deserializeExtendedRuntimePerfKnobConfig(std::istream& is);
    static void serialize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, std::ostream& os);
    static size_t serializedSize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig);

    // ParallelConfig
    static ParallelConfig deserializeParallelConfig(std::istream& is);
    static void serialize(ParallelConfig const& parallelConfig, std::ostream& os);
    static size_t serializedSize(ParallelConfig const& parallelConfig);

    // PeftCacheConfig
    static PeftCacheConfig deserializePeftCacheConfig(std::istream& is);
    static void serialize(PeftCacheConfig const& peftCacheConfig, std::ostream& os);
    static size_t serializedSize(PeftCacheConfig const& peftCacheConfig);

    // OrchestratorConfig
    static OrchestratorConfig deserializeOrchestratorConfig(std::istream& is);
    static void serialize(OrchestratorConfig const& orchestratorConfig, std::ostream& os);
    static size_t serializedSize(OrchestratorConfig const& orchestratorConfig);

    // DecodingMode
    static DecodingMode deserializeDecodingMode(std::istream& is);
    static void serialize(DecodingMode const& decodingMode, std::ostream& os);
    static size_t serializedSize(DecodingMode const& decodingMode);

    // LookaheadDecodingConfig
    static LookaheadDecodingConfig deserializeLookaheadDecodingConfig(std::istream& is);
    static void serialize(LookaheadDecodingConfig const& lookaheadDecodingConfig, std::ostream& os);
    static size_t serializedSize(LookaheadDecodingConfig const& lookaheadDecodingConfig);

    // EagleConfig
    static EagleConfig deserializeEagleConfig(std::istream& is);
    static void serialize(EagleConfig const& eagleConfig, std::ostream& os);
    static size_t serializedSize(EagleConfig const& eagleConfig);

    // SpeculativeDecodingConfig
    static SpeculativeDecodingConfig deserializeSpeculativeDecodingConfig(std::istream& is);
    static void serialize(SpeculativeDecodingConfig const& specDecConfig, std::ostream& os);
    static size_t serializedSize(SpeculativeDecodingConfig const& specDecConfig);

    // GuidedDecodingConfig
    static GuidedDecodingConfig deserializeGuidedDecodingConfig(std::istream& is);
    static void serialize(GuidedDecodingConfig const& guidedDecodingConfig, std::ostream& os);
    static size_t serializedSize(GuidedDecodingConfig const& guidedDecodingConfig);

    // GuidedDecodingParams
    static GuidedDecodingParams deserializeGuidedDecodingParams(std::istream& is);
    static void serialize(GuidedDecodingParams const& guidedDecodingParams, std::ostream& os);
    static size_t serializedSize(GuidedDecodingParams const& guidedDecodingParams);

    // KvCacheRetentionConfig
    static KvCacheRetentionConfig deserializeKvCacheRetentionConfig(std::istream& is);
    static void serialize(KvCacheRetentionConfig const& kvCacheRetentionConfig, std::ostream& os);
    static size_t serializedSize(KvCacheRetentionConfig const& kvCacheRetentionConfig);

    // TokenRangeRetentionConfig
    static KvCacheRetentionConfig::TokenRangeRetentionConfig deserializeTokenRangeRetentionConfig(std::istream& is);
    static void serialize(
        KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig, std::ostream& os);
    static size_t serializedSize(KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig);

    // DecodingConfig
    static DecodingConfig deserializeDecodingConfig(std::istream& is);
    static void serialize(DecodingConfig const& decodingConfig, std::ostream& os);
    static size_t serializedSize(DecodingConfig const& decodingConfig);

    // DebugConfig
    static DebugConfig deserializeDebugConfig(std::istream& is);
    static void serialize(DebugConfig const& debugConfig, std::ostream& os);
    static size_t serializedSize(DebugConfig const& debugConfig);

    // CacheTransceiverConfig
    static CacheTransceiverConfig deserializeCacheTransceiverConfig(std::istream& is);
    static void serialize(CacheTransceiverConfig const& cacheTransceiverConfig, std::ostream& os);
    static size_t serializedSize(CacheTransceiverConfig const& cacheTransceiverConfig);

    // ExecutorConfig
    static ExecutorConfig deserializeExecutorConfig(std::istream& is);
    static void serialize(ExecutorConfig const& executorConfig, std::ostream& os);
    static size_t serializedSize(ExecutorConfig const& executorConfig);

    // KvCacheStats
    static KvCacheStats deserializeKvCacheStats(std::istream& is);
    static void serialize(KvCacheStats const& kvCacheStats, std::ostream& os);
    static size_t serializedSize(KvCacheStats const& kvCacheStats);

    // StaticBatchingStats
    static StaticBatchingStats deserializeStaticBatchingStats(std::istream& is);
    static void serialize(StaticBatchingStats const& staticBatchingStats, std::ostream& os);
    static size_t serializedSize(StaticBatchingStats const& staticBatchingStats);

    // InflightBatchingStats
    static InflightBatchingStats deserializeInflightBatchingStats(std::istream& is);
    static void serialize(InflightBatchingStats const& inflightBatchingStats, std::ostream& os);
    static size_t serializedSize(InflightBatchingStats const& inflightBatchingStats);

    // SpecDecodingStats
    static SpecDecodingStats deserializeSpecDecodingStats(std::istream& is);
    static void serialize(SpecDecodingStats const& specDecodingStats, std::ostream& os);
    static size_t serializedSize(SpecDecodingStats const& specDecodingStats);

    // IterationStats
    static IterationStats deserializeIterationStats(std::vector<char>& buffer);
    static IterationStats deserializeIterationStats(std::istream& is);
    static void serialize(IterationStats const& iterStats, std::ostream& os);
    static std::vector<char> serialize(IterationStats const& iterStats);
    static size_t serializedSize(IterationStats const& iterStats);
    static std::vector<char> serialize(std::vector<IterationStats> const& iterStatsVec);
    static std::vector<IterationStats> deserializeIterationStatsVec(std::vector<char>& buffer);

    // DisServingStats
    [[nodiscard]] static DisServingRequestStats deserializeDisServingRequestStats(std::istream& is);
    static void serialize(DisServingRequestStats const& stats, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(DisServingRequestStats const& disServingRequestStats);

    // RequestStage
    [[nodiscard]] static RequestStage deserializeRequestStage(std::istream& is);
    static void serialize(RequestStage const& requestStage, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestStage const& requestStage);

    // RequestStats
    [[nodiscard]] static RequestStats deserializeRequestStats(std::istream& is);
    static void serialize(RequestStats const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestStats const& state);

    // RequestStatsPerIteration
    [[nodiscard]] static RequestStatsPerIteration deserializeRequestStatsPerIteration(std::istream& is);
    [[nodiscard]] static RequestStatsPerIteration deserializeRequestStatsPerIteration(std::vector<char>& buffer);
    static void serialize(RequestStatsPerIteration const& state, std::ostream& os);
    [[nodiscard]] static std::vector<char> serialize(RequestStatsPerIteration const& state);
    [[nodiscard]] static size_t serializedSize(RequestStatsPerIteration const& state);
    [[nodiscard]] static std::vector<char> serialize(std::vector<RequestStatsPerIteration> const& requestStatsVec);
    [[nodiscard]] static std::vector<RequestStatsPerIteration> deserializeRequestStatsPerIterationVec(
        std::vector<char>& buffer);

    // KVCacheEvent deque
    [[nodiscard]] static std::vector<char> serialize(std::deque<KVCacheEvent> const& kvCacheEvents);
    [[nodiscard]] static std::deque<KVCacheEvent> deserializeKVCacheEvents(std::vector<char>& buffer);

    // KVCacheEvent
    [[nodiscard]] static size_t serializedSize(KVCacheEvent const& event);
    static void serialize(KVCacheEvent const& event, std::ostream& os);
    [[nodiscard]] static KVCacheEvent deserializeKVCacheEvent(std::istream& is);

    // KVCacheCreatedData
    [[nodiscard]] static size_t serializedSize(KVCacheCreatedData const& data);
    static void serialize(KVCacheCreatedData const& data, std::ostream& os);
    [[nodiscard]] static KVCacheCreatedData deserializeKVCacheCreatedData(std::istream& is);

    // KVCacheStoredData
    [[nodiscard]] static size_t serializedSize(KVCacheStoredData const& data);
    static void serialize(KVCacheStoredData const& data, std::ostream& os);
    [[nodiscard]] static KVCacheStoredData deserializeKVCacheStoredData(std::istream& is);

    // KVCacheStoredBlockData
    [[nodiscard]] static size_t serializedSize(KVCacheStoredBlockData const& data);
    static void serialize(KVCacheStoredBlockData const& data, std::ostream& os);
    [[nodiscard]] static KVCacheStoredBlockData deserializeKVCacheStoredBlockData(std::istream& is);

    // KVCacheRemovedData
    [[nodiscard]] static size_t serializedSize(KVCacheRemovedData const& data);
    static void serialize(KVCacheRemovedData const& data, std::ostream& os);
    [[nodiscard]] static KVCacheRemovedData deserializeKVCacheRemovedData(std::istream& is);

    // KVCacheEventDiff
    template <typename T>
    [[nodiscard]] static size_t serializedSize(KVCacheEventDiff<T> const& data);
    template <typename T>
    static void serialize(KVCacheEventDiff<T> const& data, std::ostream& os);
    template <typename T>
    [[nodiscard]] static KVCacheEventDiff<T> deserializeKVCacheEventDiff(std::istream& is);

    // KVCacheUpdateData
    [[nodiscard]] static size_t serializedSize(KVCacheUpdatedData const& data);
    static void serialize(KVCacheUpdatedData const& data, std::ostream& os);
    [[nodiscard]] static KVCacheUpdatedData deserializeKVCacheUpdatedData(std::istream& is);

    // UniqueToken
    [[nodiscard]] static size_t serializedSize(tensorrt_llm::runtime::UniqueToken const& token);
    static void serialize(tensorrt_llm::runtime::UniqueToken const& token, std::ostream& os);
    [[nodiscard]] static tensorrt_llm::runtime::UniqueToken deserializeUniqueToken(std::istream& is);

    // String
    static std::string deserializeString(std::istream& is);

    // Bool
    static bool deserializeBool(std::istream& is);

    // ModelType
    static ModelType deserializeModelType(std::istream& is);
};

} // namespace tensorrt_llm::executor
