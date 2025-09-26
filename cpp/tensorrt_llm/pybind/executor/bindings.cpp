/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

namespace py = pybind11;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tle::SizeType32;

namespace tensorrt_llm::pybind::executor
{

template <typename T>
void instantiateEventDiff(pybind11::module& m, std::string const& name)
{
    py::class_<tle::KVCacheEventDiff<T>>(m, ("KVCacheEventDiff" + name).c_str())
        .def_readonly("old_value", &tle::KVCacheEventDiff<T>::oldValue)
        .def_readonly("new_value", &tle::KVCacheEventDiff<T>::newValue);
}

void initBindings(pybind11::module_& m)
{
    m.attr("__version__") = tle::version();
    py::enum_<tle::ModelType>(m, "ModelType")
        .value("DECODER_ONLY", tle::ModelType::kDECODER_ONLY)
        .value("ENCODER_ONLY", tle::ModelType::kENCODER_ONLY)
        .value("ENCODER_DECODER", tle::ModelType::kENCODER_DECODER);

    auto decodingModeGetstate = [](tle::DecodingMode const& self) { return py::make_tuple(self.getState()); };
    auto decodingModeSetstate = [](py::tuple const& state)
    {
        if (state.size() != 1)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DecodingMode(state[0].cast<tle::DecodingMode::UnderlyingType>());
    };
    py::class_<tle::DecodingMode>(m, "DecodingMode")
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
        .def_property_readonly("name", &tle::DecodingMode::getName)
        .def(py::pickle(decodingModeGetstate, decodingModeSetstate));

    py::enum_<tle::CapacitySchedulerPolicy>(m, "CapacitySchedulerPolicy")
        .value("MAX_UTILIZATION", tle::CapacitySchedulerPolicy::kMAX_UTILIZATION)
        .value("GUARANTEED_NO_EVICT", tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
        .value("STATIC_BATCH", tle::CapacitySchedulerPolicy::kSTATIC_BATCH);

    py::enum_<tle::ContextChunkingPolicy>(m, "ContextChunkingPolicy")
        .value("EQUAL_PROGRESS", tle::ContextChunkingPolicy::kEQUAL_PROGRESS)
        .value("FIRST_COME_FIRST_SERVED", tle::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED);

    py::enum_<tle::CommunicationType>(m, "CommunicationType").value("MPI", tle::CommunicationType::kMPI);

    py::enum_<tle::CommunicationMode>(m, "CommunicationMode")
        .value("LEADER", tle::CommunicationMode::kLEADER)
        .value("ORCHESTRATOR", tle::CommunicationMode::kORCHESTRATOR);

    py::class_<tle::KvCacheStats>(m, "KvCacheStats")
        .def(py::init<>())
        .def_readwrite("max_num_blocks", &tle::KvCacheStats::maxNumBlocks)
        .def_readwrite("free_num_blocks", &tle::KvCacheStats::freeNumBlocks)
        .def_readwrite("used_num_blocks", &tle::KvCacheStats::usedNumBlocks)
        .def_readwrite("tokens_per_block", &tle::KvCacheStats::tokensPerBlock)
        .def_readwrite("alloc_total_blocks", &tle::KvCacheStats::allocTotalBlocks)
        .def_readwrite("alloc_new_blocks", &tle::KvCacheStats::allocNewBlocks)
        .def_readwrite("reused_blocks", &tle::KvCacheStats::reusedBlocks)
        .def_readwrite("missed_blocks", &tle::KvCacheStats::missedBlocks)
        .def_readwrite("cache_hit_rate", &tle::KvCacheStats::cacheHitRate);

    py::class_<tle::StaticBatchingStats>(m, "StaticBatchingStats")
        .def(py::init<>())
        .def_readwrite("num_scheduled_requests", &tle::StaticBatchingStats::numScheduledRequests)
        .def_readwrite("num_context_requests", &tle::StaticBatchingStats::numContextRequests)
        .def_readwrite("num_ctx_tokens", &tle::StaticBatchingStats::numCtxTokens)
        .def_readwrite("num_gen_tokens", &tle::StaticBatchingStats::numGenTokens)
        .def_readwrite("empty_gen_slots", &tle::StaticBatchingStats::emptyGenSlots);

    py::class_<tle::InflightBatchingStats>(m, "InflightBatchingStats")
        .def(py::init<>())
        .def_readwrite("num_scheduled_requests", &tle::InflightBatchingStats::numScheduledRequests)
        .def_readwrite("num_context_requests", &tle::InflightBatchingStats::numContextRequests)
        .def_readwrite("num_gen_requests", &tle::InflightBatchingStats::numGenRequests)
        .def_readwrite("num_paused_requests", &tle::InflightBatchingStats::numPausedRequests)
        .def_readwrite("num_ctx_tokens", &tle::InflightBatchingStats::numCtxTokens)
        .def_readwrite("micro_batch_id", &tle::InflightBatchingStats::microBatchId)
        .def_readwrite("avg_num_decoded_tokens_per_iter", &tle::InflightBatchingStats::avgNumDecodedTokensPerIter);

    py::class_<tle::SpecDecodingStats>(m, "SpecDecodingStats")
        .def(py::init<>())
        .def_readwrite("num_draft_tokens", &tle::SpecDecodingStats::numDraftTokens)
        .def_readwrite("num_accepted_tokens", &tle::SpecDecodingStats::numAcceptedTokens)
        .def_readwrite("num_requests_with_draft_tokens", &tle::SpecDecodingStats::numRequestsWithDraftTokens)
        .def_readwrite("acceptance_length", &tle::SpecDecodingStats::acceptanceLength)
        .def_readwrite("iter_latency_ms", &tle::SpecDecodingStats::iterLatencyMS)
        .def_readwrite("draft_overhead", &tle::SpecDecodingStats::draftOverhead);

    py::class_<tle::IterationStats>(m, "IterationStats")
        .def(py::init<>())
        .def_readwrite("timestamp", &tle::IterationStats::timestamp)
        .def_readwrite("iter", &tle::IterationStats::iter)
        .def_readwrite("iter_latency_ms", &tle::IterationStats::iterLatencyMS)
        .def_readwrite("new_active_requests_queue_latency_ms", &tle::IterationStats::newActiveRequestsQueueLatencyMS)
        .def_readwrite("num_new_active_requests", &tle::IterationStats::numNewActiveRequests)
        .def_readwrite("num_active_requests", &tle::IterationStats::numActiveRequests)
        .def_readwrite("num_queued_requests", &tle::IterationStats::numQueuedRequests)
        .def_readwrite("num_completed_requests", &tle::IterationStats::numCompletedRequests)
        .def_readwrite("max_num_active_requests", &tle::IterationStats::maxNumActiveRequests)
        .def_readwrite("gpu_mem_usage", &tle::IterationStats::gpuMemUsage)
        .def_readwrite("cpu_mem_usage", &tle::IterationStats::cpuMemUsage)
        .def_readwrite("pinned_mem_usage", &tle::IterationStats::pinnedMemUsage)
        .def_readwrite("kv_cache_stats", &tle::IterationStats::kvCacheStats)
        .def_readwrite("cross_kv_cache_stats", &tle::IterationStats::crossKvCacheStats)
        .def_readwrite("static_batching_stats", &tle::IterationStats::staticBatchingStats)
        .def_readwrite("inflight_batching_stats", &tle::IterationStats::inflightBatchingStats)
        .def_readwrite("specdec_stats", &tle::IterationStats::specDecodingStats)
        .def("to_json_str",
            [](tle::IterationStats const& iterationStats)
            { return tle::JsonSerialization::toJsonStr(iterationStats); });

    py::class_<tle::DebugTensorsPerIteration>(m, "DebugTensorsPerIteration")
        .def(py::init<>())
        .def_readwrite("iter", &tle::DebugTensorsPerIteration::iter)
        .def_readwrite("debug_tensors", &tle::DebugTensorsPerIteration::debugTensors);

    py::enum_<tle::RequestStage>(m, "RequestStage")
        .value("QUEUED", tle::RequestStage::kQUEUED)
        .value("ENCODER_IN_PROGRESS", tle::RequestStage::kENCODER_IN_PROGRESS)
        .value("CONTEXT_IN_PROGRESS", tle::RequestStage::kCONTEXT_IN_PROGRESS)
        .value("GENERATION_IN_PROGRESS", tle::RequestStage::kGENERATION_IN_PROGRESS)
        .value("GENERATION_COMPLETE", tle::RequestStage::kGENERATION_COMPLETE);

    py::class_<tle::DisServingRequestStats>(m, "DisServingRequestStats")
        .def(py::init<>())
        .def_readwrite("kv_cache_transfer_ms", &tle::DisServingRequestStats::kvCacheTransferMS)
        .def_readwrite("kv_cache_size", &tle::DisServingRequestStats::kvCacheSize);

    py::class_<tle::RequestStats>(m, "RequestStats")
        .def(py::init<>())
        .def_readwrite("id", &tle::RequestStats::id)
        .def_readwrite("stage", &tle::RequestStats::stage)
        .def_readwrite("context_prefill_position", &tle::RequestStats::contextPrefillPosition)
        .def_readwrite("num_generated_tokens", &tle::RequestStats::numGeneratedTokens)
        .def_readwrite("avg_num_decoded_tokens_per_iter", &tle::RequestStats::avgNumDecodedTokensPerIter)
        .def_readwrite("scheduled", &tle::RequestStats::scheduled)
        .def_readwrite("paused", &tle::RequestStats::paused)
        .def_readwrite("dis_serving_stats", &tle::RequestStats::disServingStats)
        .def_readwrite("alloc_total_blocks_per_request", &tle::RequestStats::allocTotalBlocksPerRequest)
        .def_readwrite("alloc_new_blocks_per_request", &tle::RequestStats::allocNewBlocksPerRequest)
        .def_readwrite("reused_blocks_per_request", &tle::RequestStats::reusedBlocksPerRequest)
        .def_readwrite("missed_blocks_per_request", &tle::RequestStats::missedBlocksPerRequest)
        .def_readwrite("kv_cache_hit_rate_per_request", &tle::RequestStats::kvCacheHitRatePerRequest)
        .def("to_json_str",
            [](tle::RequestStats const& iterationStats) { return tle::JsonSerialization::toJsonStr(iterationStats); });

    py::class_<tle::RequestStatsPerIteration>(m, "RequestStatsPerIteration")
        .def(py::init<>())
        .def_readwrite("iter", &tle::RequestStatsPerIteration::iter)
        .def_readwrite("request_stats", &tle::RequestStatsPerIteration::requestStats)
        .def("to_json_str",
            [](tle::RequestStatsPerIteration const& iterationStats)
            { return tle::JsonSerialization::toJsonStr(iterationStats); });

    py::module_ executor_kv_cache = m.def_submodule("kv_cache", "Executor KV Cache Manager");

    py::class_<tle::KVCacheCreatedData>(executor_kv_cache, "KVCacheCreatedData")
        .def_readonly("num_blocks_per_cache_level", &tle::KVCacheCreatedData::numBlocksPerCacheLevel);

    py::class_<tensorrt_llm::runtime::UniqueToken>(executor_kv_cache, "UniqueToken")
        .def_readonly("token_id", &tensorrt_llm::runtime::UniqueToken::tokenId)
        .def_readonly("token_extra_id", &tensorrt_llm::runtime::UniqueToken::tokenExtraId);

    py::class_<tle::KVCacheStoredBlockData>(executor_kv_cache, "KVCacheStoredBlockData")
        .def_readonly("block_hash", &tle::KVCacheStoredBlockData::blockHash)
        .def_readonly("tokens", &tle::KVCacheStoredBlockData::tokens)
        .def_readonly("lora_id", &tle::KVCacheStoredBlockData::loraId)
        .def_readonly("cache_level", &tle::KVCacheStoredBlockData::cacheLevel)
        .def_readonly("priority", &tle::KVCacheStoredBlockData::priority);

    py::class_<tle::KVCacheStoredData>(executor_kv_cache, "KVCacheStoredData")
        .def_readonly("parent_hash", &tle::KVCacheStoredData::parentHash)
        .def_readonly("blocks", &tle::KVCacheStoredData::blocks);

    py::class_<tle::KVCacheRemovedData>(executor_kv_cache, "KVCacheRemovedData")
        .def_readonly("block_hashes", &tle::KVCacheRemovedData::blockHashes);

    instantiateEventDiff<SizeType32>(executor_kv_cache, "Int");

    py::class_<tle::KVCacheUpdatedData>(executor_kv_cache, "KVCacheUpdatedData")
        .def_readonly("block_hash", &tle::KVCacheUpdatedData::blockHash)
        .def_readonly("cache_level", &tle::KVCacheUpdatedData::cacheLevel)
        .def_readonly("priority", &tle::KVCacheUpdatedData::priority);

    py::class_<tle::KVCacheEvent>(executor_kv_cache, "KVCacheEvent")
        .def_readonly("event_id", &tle::KVCacheEvent::eventId)
        .def_readonly("data", &tle::KVCacheEvent::data)
        .def_readonly("window_size", &tle::KVCacheEvent::windowSize)
        .def_readonly("attention_dp_rank", &tle::KVCacheEvent::attentionDpRank);

    py::class_<tle::KVCacheEventManager, std::shared_ptr<tle::KVCacheEventManager>>(
        executor_kv_cache, "KVCacheEventManager")
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
            py::arg("timeout_ms") = std::nullopt);

    tensorrt_llm::pybind::executor::initRequestBindings(m);
    tensorrt_llm::pybind::executor::initConfigBindings(m);
    tensorrt_llm::pybind::executor::Executor::initBindings(m);
}

} // namespace tensorrt_llm::pybind::executor
