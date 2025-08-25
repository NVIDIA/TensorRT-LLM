/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bindings.h"
#include "executor.h"
#include "executorConfig.h"
#include "request.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <optional>

namespace nb = nanobind;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tle::SizeType32;

namespace tensorrt_llm::nanobind::executor
{

template <typename T>
void instantiateEventDiff(nb::module_& m, std::string const& name)
{
    nb::class_<tle::KVCacheEventDiff<T>>(m, ("KVCacheEventDiff" + name).c_str())
        .def_ro("old_value", &tle::KVCacheEventDiff<T>::oldValue)
        .def_ro("new_value", &tle::KVCacheEventDiff<T>::newValue);
}

void initBindings(nb::module_& m)
{
    m.attr("__version__") = tle::version();
    nb::enum_<tle::ModelType>(m, "ModelType")
        .value("DECODER_ONLY", tle::ModelType::kDECODER_ONLY)
        .value("ENCODER_ONLY", tle::ModelType::kENCODER_ONLY)
        .value("ENCODER_DECODER", tle::ModelType::kENCODER_DECODER);

    auto decodingModeGetstate = [](tle::DecodingMode const& self) { return nb::make_tuple(self.getState()); };
    auto decodingModeSetstate = [](tle::DecodingMode& self, nb::tuple const& state)
    {
        if (state.size() != 1)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::DecodingMode(nb::cast<tle::DecodingMode::UnderlyingType>(state[0]));
    };
    nb::class_<tle::DecodingMode>(m, "DecodingMode")
        .def("Auto", &tle::DecodingMode::Auto)
        .def("TopK", &tle::DecodingMode::TopK)
        .def("TopP", &tle::DecodingMode::TopP)
        .def("TopKTopP", &tle::DecodingMode::TopKTopP)
        .def("BeamSearch", &tle::DecodingMode::BeamSearch)
        .def("Medusa", &tle::DecodingMode::Medusa)
        .def("Lookahead", &tle::DecodingMode::Lookahead)
        .def("ExplicitDraftTokens", &tle::DecodingMode::ExplicitDraftTokens)
        .def("Eagle", &tle::DecodingMode::Eagle)
        .def("isAuto", &tle::DecodingMode::isAuto)
        .def("isTopK", &tle::DecodingMode::isTopK)
        .def("isTopP", &tle::DecodingMode::isTopP)
        .def("isTopKorTopP", &tle::DecodingMode::isTopKorTopP)
        .def("isTopKandTopP", &tle::DecodingMode::isTopKandTopP)
        .def("isBeamSearch", &tle::DecodingMode::isBeamSearch)
        .def("isMedusa", &tle::DecodingMode::isMedusa)
        .def("isLookahead", &tle::DecodingMode::isLookahead)
        .def("isExplicitDraftTokens", &tle::DecodingMode::isExplicitDraftTokens)
        .def("isEagle", &tle::DecodingMode::isEagle)
        .def("useVariableBeamWidthSearch", &tle::DecodingMode::useVariableBeamWidthSearch)
        .def_prop_ro("name", &tle::DecodingMode::getName)
        .def("__getstate__", decodingModeGetstate)
        .def("__setstate__", decodingModeSetstate);

    nb::enum_<tle::CapacitySchedulerPolicy>(m, "CapacitySchedulerPolicy")
        .value("MAX_UTILIZATION", tle::CapacitySchedulerPolicy::kMAX_UTILIZATION)
        .value("GUARANTEED_NO_EVICT", tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
        .value("STATIC_BATCH", tle::CapacitySchedulerPolicy::kSTATIC_BATCH);

    nb::enum_<tle::ContextChunkingPolicy>(m, "ContextChunkingPolicy")
        .value("EQUAL_PROGRESS", tle::ContextChunkingPolicy::kEQUAL_PROGRESS)
        .value("FIRST_COME_FIRST_SERVED", tle::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED);

    nb::enum_<tle::CommunicationType>(m, "CommunicationType").value("MPI", tle::CommunicationType::kMPI);

    nb::enum_<tle::CommunicationMode>(m, "CommunicationMode")
        .value("LEADER", tle::CommunicationMode::kLEADER)
        .value("ORCHESTRATOR", tle::CommunicationMode::kORCHESTRATOR);

    nb::class_<tle::KvCacheStats>(m, "KvCacheStats")
        .def(nb::init<>())
        .def_rw("max_num_blocks", &tle::KvCacheStats::maxNumBlocks)
        .def_rw("free_num_blocks", &tle::KvCacheStats::freeNumBlocks)
        .def_rw("used_num_blocks", &tle::KvCacheStats::usedNumBlocks)
        .def_rw("tokens_per_block", &tle::KvCacheStats::tokensPerBlock)
        .def_rw("alloc_total_blocks", &tle::KvCacheStats::allocTotalBlocks)
        .def_rw("alloc_new_blocks", &tle::KvCacheStats::allocNewBlocks)
        .def_rw("reused_blocks", &tle::KvCacheStats::reusedBlocks)
        .def_rw("missed_blocks", &tle::KvCacheStats::missedBlocks)
        .def_rw("cache_hit_rate", &tle::KvCacheStats::cacheHitRate);

    nb::class_<tle::StaticBatchingStats>(m, "StaticBatchingStats")
        .def(nb::init<>())
        .def_rw("num_scheduled_requests", &tle::StaticBatchingStats::numScheduledRequests)
        .def_rw("num_context_requests", &tle::StaticBatchingStats::numContextRequests)
        .def_rw("num_ctx_tokens", &tle::StaticBatchingStats::numCtxTokens)
        .def_rw("num_gen_tokens", &tle::StaticBatchingStats::numGenTokens)
        .def_rw("empty_gen_slots", &tle::StaticBatchingStats::emptyGenSlots);

    nb::class_<tle::InflightBatchingStats>(m, "InflightBatchingStats")
        .def(nb::init<>())
        .def_rw("num_scheduled_requests", &tle::InflightBatchingStats::numScheduledRequests)
        .def_rw("num_context_requests", &tle::InflightBatchingStats::numContextRequests)
        .def_rw("num_gen_requests", &tle::InflightBatchingStats::numGenRequests)
        .def_rw("num_paused_requests", &tle::InflightBatchingStats::numPausedRequests)
        .def_rw("num_ctx_tokens", &tle::InflightBatchingStats::numCtxTokens)
        .def_rw("micro_batch_id", &tle::InflightBatchingStats::microBatchId)
        .def_rw("avg_num_decoded_tokens_per_iter", &tle::InflightBatchingStats::avgNumDecodedTokensPerIter);

    nb::class_<tle::SpecDecodingStats>(m, "SpecDecodingStats")
        .def(nb::init<>())
        .def_rw("num_draft_tokens", &tle::SpecDecodingStats::numDraftTokens)
        .def_rw("num_accepted_tokens", &tle::SpecDecodingStats::numAcceptedTokens)
        .def_rw("num_requests_with_draft_tokens", &tle::SpecDecodingStats::numRequestsWithDraftTokens)
        .def_rw("acceptance_length", &tle::SpecDecodingStats::acceptanceLength)
        .def_rw("iter_latency_ms", &tle::SpecDecodingStats::iterLatencyMS)
        .def_rw("draft_overhead", &tle::SpecDecodingStats::draftOverhead);

    nb::class_<tle::IterationStats>(m, "IterationStats")
        .def(nb::init<>())
        .def_rw("timestamp", &tle::IterationStats::timestamp)
        .def_rw("iter", &tle::IterationStats::iter)
        .def_rw("iter_latency_ms", &tle::IterationStats::iterLatencyMS)
        .def_rw("new_active_requests_queue_latency_ms", &tle::IterationStats::newActiveRequestsQueueLatencyMS)
        .def_rw("num_new_active_requests", &tle::IterationStats::numNewActiveRequests)
        .def_rw("num_active_requests", &tle::IterationStats::numActiveRequests)
        .def_rw("num_queued_requests", &tle::IterationStats::numQueuedRequests)
        .def_rw("num_completed_requests", &tle::IterationStats::numCompletedRequests)
        .def_rw("max_num_active_requests", &tle::IterationStats::maxNumActiveRequests)
        .def_rw("gpu_mem_usage", &tle::IterationStats::gpuMemUsage)
        .def_rw("cpu_mem_usage", &tle::IterationStats::cpuMemUsage)
        .def_rw("pinned_mem_usage", &tle::IterationStats::pinnedMemUsage)
        .def_rw("kv_cache_stats", &tle::IterationStats::kvCacheStats)
        .def_rw("cross_kv_cache_stats", &tle::IterationStats::crossKvCacheStats)
        .def_rw("static_batching_stats", &tle::IterationStats::staticBatchingStats)
        .def_rw("inflight_batching_stats", &tle::IterationStats::inflightBatchingStats)
        .def_rw("specdec_stats", &tle::IterationStats::specDecodingStats)
        .def("to_json_str",
            [](tle::IterationStats const& iterationStats)
            { return tle::JsonSerialization::toJsonStr(iterationStats); });

    nb::class_<tle::DebugTensorsPerIteration>(m, "DebugTensorsPerIteration")
        .def(nb::init<>())
        .def_rw("iter", &tle::DebugTensorsPerIteration::iter)
        .def_rw("debug_tensors", &tle::DebugTensorsPerIteration::debugTensors);

    nb::enum_<tle::RequestStage>(m, "RequestStage")
        .value("QUEUED", tle::RequestStage::kQUEUED)
        .value("ENCODER_IN_PROGRESS", tle::RequestStage::kENCODER_IN_PROGRESS)
        .value("CONTEXT_IN_PROGRESS", tle::RequestStage::kCONTEXT_IN_PROGRESS)
        .value("GENERATION_IN_PROGRESS", tle::RequestStage::kGENERATION_IN_PROGRESS)
        .value("GENERATION_COMPLETE", tle::RequestStage::kGENERATION_COMPLETE);

    nb::class_<tle::DisServingRequestStats>(m, "DisServingRequestStats")
        .def(nb::init<>())
        .def_rw("kv_cache_transfer_ms", &tle::DisServingRequestStats::kvCacheTransferMS)
        .def_rw("kv_cache_size", &tle::DisServingRequestStats::kvCacheSize);

    nb::class_<tle::RequestStats>(m, "RequestStats")
        .def(nb::init<>())
        .def_rw("id", &tle::RequestStats::id)
        .def_rw("stage", &tle::RequestStats::stage)
        .def_rw("context_prefill_position", &tle::RequestStats::contextPrefillPosition)
        .def_rw("num_generated_tokens", &tle::RequestStats::numGeneratedTokens)
        .def_rw("avg_num_decoded_tokens_per_iter", &tle::RequestStats::avgNumDecodedTokensPerIter)
        .def_rw("scheduled", &tle::RequestStats::scheduled)
        .def_rw("paused", &tle::RequestStats::paused)
        .def_rw("dis_serving_stats", &tle::RequestStats::disServingStats)
        .def_rw("alloc_total_blocks_per_request", &tle::RequestStats::allocTotalBlocksPerRequest)
        .def_rw("alloc_new_blocks_per_request", &tle::RequestStats::allocNewBlocksPerRequest)
        .def_rw("reused_blocks_per_request", &tle::RequestStats::reusedBlocksPerRequest)
        .def_rw("missed_blocks_per_request", &tle::RequestStats::missedBlocksPerRequest)
        .def_rw("kv_cache_hit_rate_per_request", &tle::RequestStats::kvCacheHitRatePerRequest)
        .def("to_json_str",
            [](tle::RequestStats const& iterationStats) { return tle::JsonSerialization::toJsonStr(iterationStats); });

    nb::class_<tle::RequestStatsPerIteration>(m, "RequestStatsPerIteration")
        .def(nb::init<>())
        .def_rw("iter", &tle::RequestStatsPerIteration::iter)
        .def_rw("request_stats", &tle::RequestStatsPerIteration::requestStats)
        .def("to_json_str",
            [](tle::RequestStatsPerIteration const& iterationStats)
            { return tle::JsonSerialization::toJsonStr(iterationStats); });

    nb::module_ executor_kv_cache = m.def_submodule("kv_cache", "Executor KV Cache Manager");

    nb::class_<tle::KVCacheCreatedData>(executor_kv_cache, "KVCacheCreatedData")
        .def_ro("num_blocks_per_cache_level", &tle::KVCacheCreatedData::numBlocksPerCacheLevel);

    nb::class_<tensorrt_llm::runtime::UniqueToken>(executor_kv_cache, "UniqueToken")
        .def_ro("token_id", &tensorrt_llm::runtime::UniqueToken::tokenId)
        .def_ro("token_extra_id", &tensorrt_llm::runtime::UniqueToken::tokenExtraId);

    nb::class_<tle::KVCacheStoredBlockData>(executor_kv_cache, "KVCacheStoredBlockData")
        .def_ro("block_hash", &tle::KVCacheStoredBlockData::blockHash)
        .def_ro("tokens", &tle::KVCacheStoredBlockData::tokens)
        .def_ro("lora_id", &tle::KVCacheStoredBlockData::loraId)
        .def_ro("cache_level", &tle::KVCacheStoredBlockData::cacheLevel)
        .def_ro("priority", &tle::KVCacheStoredBlockData::priority);

    nb::class_<tle::KVCacheStoredData>(executor_kv_cache, "KVCacheStoredData")
        .def_ro("parent_hash", &tle::KVCacheStoredData::parentHash)
        .def_ro("blocks", &tle::KVCacheStoredData::blocks);

    nb::class_<tle::KVCacheRemovedData>(executor_kv_cache, "KVCacheRemovedData")
        .def_ro("block_hashes", &tle::KVCacheRemovedData::blockHashes);

    instantiateEventDiff<SizeType32>(executor_kv_cache, "Int");

    nb::class_<tle::KVCacheUpdatedData>(executor_kv_cache, "KVCacheUpdatedData")
        .def_ro("block_hash", &tle::KVCacheUpdatedData::blockHash)
        .def_ro("cache_level", &tle::KVCacheUpdatedData::cacheLevel)
        .def_ro("priority", &tle::KVCacheUpdatedData::priority);

    nb::class_<tle::KVCacheEvent>(executor_kv_cache, "KVCacheEvent")
        .def_ro("event_id", &tle::KVCacheEvent::eventId)
        .def_ro("data", &tle::KVCacheEvent::data)
        .def_ro("window_size", &tle::KVCacheEvent::windowSize)
        .def_ro("attention_dp_rank", &tle::KVCacheEvent::attentionDpRank);

    nb::class_<tle::KVCacheEventManager>(executor_kv_cache, "KVCacheEventManager")
        .def(
            "get_latest_events",
            [](tle::KVCacheEventManager& self, std::optional<double> timeout_ms = std::nullopt)
            {
                if (timeout_ms)
                {
                    return self.getLatestEvents(std::chrono::milliseconds(static_cast<int64_t>(*timeout_ms)));
                }
                return self.getLatestEvents(std::nullopt);
            },
            nb::arg("timeout_ms") = std::nullopt);

    tensorrt_llm::nanobind::executor::initRequestBindings(m);
    tensorrt_llm::nanobind::executor::initConfigBindings(m);
    tensorrt_llm::nanobind::executor::Executor::initBindings(m);
}

} // namespace tensorrt_llm::nanobind::executor
