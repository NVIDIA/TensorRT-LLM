/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "executorConfig.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>
#include <torch/torch.h>
#include <vector>

namespace nb = nanobind;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tle::SizeType32;
using RuntimeDefaults = tensorrt_llm::runtime::RuntimeDefaults;

namespace tensorrt_llm::nanobind::executor
{

void initConfigBindings(nb::module_& m)
{
    nb::enum_<tle::BatchingType>(m, "BatchingType")
        .value("STATIC", tle::BatchingType::kSTATIC)
        .value("INFLIGHT", tle::BatchingType::kINFLIGHT);

    auto dynamicBatchConfigGetstate = [](tle::DynamicBatchConfig const& self)
    {
        return nb::make_tuple(self.getEnableBatchSizeTuning(), self.getEnableMaxNumTokensTuning(),
            self.getDynamicBatchMovingAverageWindow(), self.getBatchSizeTable());
    };
    auto dynamicBatchConfigSetstate = [](tle::DynamicBatchConfig& self, nb::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::DynamicBatchConfig(nb::cast<bool>(state[0]), nb::cast<bool>(state[1]),
            nb::cast<SizeType32>(state[2]), nb::cast<std::vector<std::pair<SizeType32, SizeType32>>>(state[3]));
    };
    nb::class_<tle::DynamicBatchConfig>(m, "DynamicBatchConfig")
        .def(nb::init<bool, bool, SizeType32>(), nb::arg("enable_batch_size_tuning"),
            nb::arg("enable_max_num_tokens_tuning"), nb::arg("dynamic_batch_moving_average_window"))
        .def_prop_ro("enable_batch_size_tuning", &tle::DynamicBatchConfig::getEnableBatchSizeTuning)
        .def_prop_ro("enable_max_num_tokens_tuning", &tle::DynamicBatchConfig::getEnableMaxNumTokensTuning)
        .def_prop_ro(
            "dynamic_batch_moving_average_window", &tle::DynamicBatchConfig::getDynamicBatchMovingAverageWindow)
        .def("__getstate__", dynamicBatchConfigGetstate)
        .def("__setstate__", dynamicBatchConfigSetstate);

    auto schedulerConfigSetstate = [](tle::SchedulerConfig& self, nb::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::SchedulerConfig(nb::cast<tle::CapacitySchedulerPolicy>(state[0]),
            nb::cast<std::optional<tle::ContextChunkingPolicy>>(state[1]),
            nb::cast<std::optional<tle::DynamicBatchConfig>>(state[2]), nb::cast<bool>(state[3]));
    };
    auto schedulerConfigGetstate = [](tle::SchedulerConfig const& self)
    {
        return nb::make_tuple(self.getCapacitySchedulerPolicy(), self.getContextChunkingPolicy(),
            self.getDynamicBatchConfig(), self.getEnablePrefixAwareScheduling());
    };
    nb::class_<tle::SchedulerConfig>(m, "SchedulerConfig")
        .def(nb::init<tle::CapacitySchedulerPolicy, std::optional<tle::ContextChunkingPolicy>,
                 std::optional<tle::DynamicBatchConfig>, bool>(),
            nb::arg("capacity_scheduler_policy") = tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
            nb::arg("context_chunking_policy") = nb::none(), nb::arg("dynamic_batch_config") = nb::none(),
            nb::arg("enable_prefix_aware_scheduling") = true)
        .def_prop_ro("capacity_scheduler_policy", &tle::SchedulerConfig::getCapacitySchedulerPolicy)
        .def_prop_ro("context_chunking_policy", &tle::SchedulerConfig::getContextChunkingPolicy)
        .def_prop_ro("dynamic_batch_config", &tle::SchedulerConfig::getDynamicBatchConfig)
        .def_prop_ro("enable_prefix_aware_scheduling", &tle::SchedulerConfig::getEnablePrefixAwareScheduling)
        .def("__getstate__", schedulerConfigGetstate)
        .def("__setstate__", schedulerConfigSetstate);

    nb::class_<RuntimeDefaults>(m, "RuntimeDefaults")
        .def(nb::init<std::optional<std::vector<SizeType32>>, std::optional<SizeType32>>(),
            nb::arg("max_attention_window") = nb::none(), nb::arg("sink_token_length") = nb::none())
        .def_ro("max_attention_window", &RuntimeDefaults::maxAttentionWindowVec)
        .def_ro("sink_token_length", &RuntimeDefaults::sinkTokenLength);

    auto kvCacheConfigGetstate = [](tle::KvCacheConfig const& self)
    {
        return nb::make_tuple(self.getEnableBlockReuse(), self.getMaxTokens(), self.getMaxAttentionWindowVec(),
            self.getSinkTokenLength(), self.getFreeGpuMemoryFraction(), self.getHostCacheSize(),
            self.getCrossKvCacheFraction(), self.getSecondaryOffloadMinPriority(), self.getEventBufferMaxSize(),
            self.getEnablePartialReuse(), self.getCopyOnPartialReuse(), self.getUseUvm(),
            self.getAttentionDpEventsGatherPeriodMs(), self.getMaxGpuTotalBytes());
    };
    auto kvCacheConfigSetstate = [](tle::KvCacheConfig& self, nb::tuple const& state)
    {
        if (state.size() != 14)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::KvCacheConfig(nb::cast<bool>(state[0]), nb::cast<std::optional<SizeType32>>(state[1]),
            nb::cast<std::optional<std::vector<SizeType32>>>(state[2]), nb::cast<std::optional<SizeType32>>(state[3]),
            nb::cast<std::optional<float>>(state[4]), nb::cast<std::optional<size_t>>(state[5]),
            nb::cast<std::optional<float>>(state[6]), nb::cast<std::optional<tle::RetentionPriority>>(state[7]),
            nb::cast<size_t>(state[8]), nb::cast<bool>(state[9]), nb::cast<bool>(state[10]), nb::cast<bool>(state[11]),
            nb::cast<SizeType32>(state[12]), std::nullopt, nb::cast<uint64_t>(state[13]));
    };
    nb::class_<tle::KvCacheConfig>(m, "KvCacheConfig")
        .def(nb::init<bool, std::optional<SizeType32> const&, std::optional<std::vector<SizeType32>> const&,
                 std::optional<SizeType32> const&, std::optional<float> const&, std::optional<size_t> const&,
                 std::optional<float> const&, std::optional<tle::RetentionPriority>, size_t const&, bool, bool, bool,
                 SizeType32, std::optional<RuntimeDefaults> const&, uint64_t const&>(),
            nb::arg("enable_block_reuse") = true, nb::arg("max_tokens") = nb::none(),
            nb::arg("max_attention_window") = nb::none(), nb::arg("sink_token_length") = nb::none(),
            nb::arg("free_gpu_memory_fraction") = nb::none(), nb::arg("host_cache_size") = nb::none(),
            nb::arg("cross_kv_cache_fraction") = nb::none(), nb::arg("secondary_offload_min_priority") = nb::none(),
            nb::arg("event_buffer_max_size") = 0, nb::kw_only(), nb::arg("enable_partial_reuse") = true,
            nb::arg("copy_on_partial_reuse") = true, nb::arg("use_uvm") = false,
            nb::arg("attention_dp_events_gather_period_ms") = 5, nb::arg("runtime_defaults") = nb::none(),
            nb::arg("max_gpu_total_bytes") = 0)
        .def_prop_rw(
            "enable_block_reuse", &tle::KvCacheConfig::getEnableBlockReuse, &tle::KvCacheConfig::setEnableBlockReuse)
        .def_prop_rw("max_tokens", &tle::KvCacheConfig::getMaxTokens, &tle::KvCacheConfig::setMaxTokens)
        .def_prop_rw("max_attention_window", &tle::KvCacheConfig::getMaxAttentionWindowVec,
            &tle::KvCacheConfig::setMaxAttentionWindowVec)
        .def_prop_rw(
            "sink_token_length", &tle::KvCacheConfig::getSinkTokenLength, &tle::KvCacheConfig::setSinkTokenLength)
        .def_prop_rw("free_gpu_memory_fraction", &tle::KvCacheConfig::getFreeGpuMemoryFraction,
            &tle::KvCacheConfig::setFreeGpuMemoryFraction)
        .def_prop_rw("host_cache_size", &tle::KvCacheConfig::getHostCacheSize, &tle::KvCacheConfig::setHostCacheSize)
        .def_prop_rw("cross_kv_cache_fraction", &tle::KvCacheConfig::getCrossKvCacheFraction,
            &tle::KvCacheConfig::setCrossKvCacheFraction)
        .def_prop_rw("secondary_offload_min_priority", &tle::KvCacheConfig::getSecondaryOffloadMinPriority,
            &tle::KvCacheConfig::setSecondaryOffloadMinPriority)
        .def_prop_rw("event_buffer_max_size", &tle::KvCacheConfig::getEventBufferMaxSize,
            &tle::KvCacheConfig::setEventBufferMaxSize)
        .def_prop_rw("enable_partial_reuse", &tle::KvCacheConfig::getEnablePartialReuse,
            &tle::KvCacheConfig::setEnablePartialReuse)
        .def_prop_rw("copy_on_partial_reuse", &tle::KvCacheConfig::getCopyOnPartialReuse,
            &tle::KvCacheConfig::setCopyOnPartialReuse)
        .def_prop_rw("use_uvm", &tle::KvCacheConfig::getUseUvm, &tle::KvCacheConfig::setUseUvm)
        .def_prop_rw("attention_dp_events_gather_period_ms", &tle::KvCacheConfig::getAttentionDpEventsGatherPeriodMs,
            &tle::KvCacheConfig::setAttentionDpEventsGatherPeriodMs)
        .def_prop_rw(
            "max_gpu_total_bytes", &tle::KvCacheConfig::getMaxGpuTotalBytes, &tle::KvCacheConfig::setMaxGpuTotalBytes)
        .def("fill_empty_fields_from_runtime_defaults", &tle::KvCacheConfig::fillEmptyFieldsFromRuntimeDefaults)
        .def("__getstate__", kvCacheConfigGetstate)
        .def("__setstate__", kvCacheConfigSetstate);

    auto peftCacheConfigSetstate = [](tle::PeftCacheConfig& self, nb::tuple const& state)
    {
        if (state.size() != 11)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::PeftCacheConfig(nb::cast<SizeType32>(state[0]), nb::cast<SizeType32>(state[1]),
            nb::cast<SizeType32>(state[2]), nb::cast<SizeType32>(state[3]), nb::cast<SizeType32>(state[4]),
            nb::cast<SizeType32>(state[5]), nb::cast<SizeType32>(state[6]), nb::cast<SizeType32>(state[7]),
            nb::cast<SizeType32>(state[8]), nb::cast<std::optional<float>>(state[9]),
            nb::cast<std::optional<size_t>>(state[10]));
    };
    auto peftCacheConfigGetstate = [](tle::PeftCacheConfig const& self)
    {
        return nb::make_tuple(self.getNumHostModuleLayer(), self.getNumDeviceModuleLayer(),
            self.getOptimalAdapterSize(), self.getMaxAdapterSize(), self.getNumPutWorkers(), self.getNumEnsureWorkers(),
            self.getNumCopyStreams(), self.getMaxPagesPerBlockHost(), self.getMaxPagesPerBlockDevice(),
            self.getDeviceCachePercent(), self.getHostCacheSize());
    };
    nb::class_<tle::PeftCacheConfig>(m, "PeftCacheConfig")
        .def(nb::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, std::optional<float> const&, std::optional<size_t> const&,
                 std::optional<std::string> const&>(),
            nb::arg("num_host_module_layer") = 0, nb::arg("num_device_module_layer") = 0,
            nb::arg("optimal_adapter_size") = 8, nb::arg("max_adapter_size") = 64, nb::arg("num_put_workers") = 1,
            nb::arg("num_ensure_workers") = 1, nb::arg("num_copy_streams") = 1,
            nb::arg("max_pages_per_block_host") = 24, nb::arg("max_pages_per_block_device") = 8,
            nb::arg("device_cache_percent") = nb::none(), nb::arg("host_cache_size") = nb::none(),
            nb::arg("lora_prefetch_dir") = nb::none())
        .def_prop_ro("num_host_module_layer", &tle::PeftCacheConfig::getNumHostModuleLayer)
        .def_prop_ro("num_device_module_layer", &tle::PeftCacheConfig::getNumDeviceModuleLayer)
        .def_prop_ro("optimal_adapter_size", &tle::PeftCacheConfig::getOptimalAdapterSize)
        .def_prop_ro("max_adapter_size", &tle::PeftCacheConfig::getMaxAdapterSize)
        .def_prop_ro("num_put_workers", &tle::PeftCacheConfig::getNumPutWorkers)
        .def_prop_ro("num_ensure_workers", &tle::PeftCacheConfig::getNumEnsureWorkers)
        .def_prop_ro("num_copy_streams", &tle::PeftCacheConfig::getNumCopyStreams)
        .def_prop_ro("max_pages_per_block_host", &tle::PeftCacheConfig::getMaxPagesPerBlockHost)
        .def_prop_ro("max_pages_per_block_device", &tle::PeftCacheConfig::getMaxPagesPerBlockDevice)
        .def_prop_ro("device_cache_percent", &tle::PeftCacheConfig::getDeviceCachePercent)
        .def_prop_ro("host_cache_size", &tle::PeftCacheConfig::getHostCacheSize)
        .def_prop_ro("lora_prefetch_dir", &tle::PeftCacheConfig::getLoraPrefetchDir)
        .def("__getstate__", peftCacheConfigGetstate)
        .def("__setstate__", peftCacheConfigSetstate);

    auto decodingConfigGetstate = [](tle::DecodingConfig const& self)
    { return nb::make_tuple(self.getDecodingMode(), self.getLookaheadDecodingConfig(), self.getMedusaChoices()); };
    auto decodingConfigSetstate = [](tle::DecodingConfig& self, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::DecodingConfig(nb::cast<std::optional<tle::DecodingMode>>(state[0]), // DecodingMode
            nb::cast<std::optional<tle::LookaheadDecodingConfig>>(state[1]),                  // LookaheadDecodingConfig
            nb::cast<std::optional<tle::MedusaChoices>>(state[2]));                           // MedusaChoices
    };
    nb::class_<tle::DecodingConfig>(m, "DecodingConfig")
        .def(nb::init<std::optional<tle::DecodingMode>, std::optional<tle::LookaheadDecodingConfig>,
                 std::optional<tle::MedusaChoices>>(),
            nb::arg("decoding_mode") = nb::none(), nb::arg("lookahead_decoding_config") = nb::none(),
            nb::arg("medusa_choices") = nb::none())
        .def_prop_rw("decoding_mode", &tle::DecodingConfig::getDecodingMode, &tle::DecodingConfig::setDecodingMode)
        .def_prop_rw("lookahead_decoding_config", &tle::DecodingConfig::getLookaheadDecodingConfig,
            &tle::DecodingConfig::setLookaheadDecodingConfig)
        .def_prop_rw("medusa_choices", &tle::DecodingConfig::getMedusaChoices, &tle::DecodingConfig::setMedusaChoices)
        .def("__getstate__", decodingConfigGetstate)
        .def("__setstate__", decodingConfigSetstate);

    auto extendedRuntimePerfKnobConfigSetstate = [](tle::ExtendedRuntimePerfKnobConfig& self, nb::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid extendedRuntimePerfKnobConfig state!");
        }
        new (&self) tle::ExtendedRuntimePerfKnobConfig(nb::cast<bool>(state[0]), nb::cast<bool>(state[1]),
            nb::cast<bool>(state[2]), nb::cast<SizeType32>(state[3]));
    };
    auto extendedRuntimePerfKnobConfigGetstate = [](tle::ExtendedRuntimePerfKnobConfig const& self)
    {
        return nb::make_tuple(self.getMultiBlockMode(), self.getEnableContextFMHAFP32Acc(), self.getCudaGraphMode(),
            self.getCudaGraphCacheSize());
    };
    nb::class_<tle::ExtendedRuntimePerfKnobConfig>(m, "ExtendedRuntimePerfKnobConfig")
        .def(
            nb::init<bool, bool>(), nb::arg("multi_block_mode") = true, nb::arg("enable_context_fmha_fp32_acc") = false)
        .def_prop_rw("multi_block_mode", &tle::ExtendedRuntimePerfKnobConfig::getMultiBlockMode,
            &tle::ExtendedRuntimePerfKnobConfig::setMultiBlockMode)
        .def_prop_rw("enable_context_fmha_fp32_acc", &tle::ExtendedRuntimePerfKnobConfig::getEnableContextFMHAFP32Acc,
            &tle::ExtendedRuntimePerfKnobConfig::setEnableContextFMHAFP32Acc)
        .def_prop_rw("cuda_graph_mode", &tle::ExtendedRuntimePerfKnobConfig::getCudaGraphMode,
            &tle::ExtendedRuntimePerfKnobConfig::setCudaGraphMode)
        .def_prop_rw("cuda_graph_cache_size", &tle::ExtendedRuntimePerfKnobConfig::getCudaGraphCacheSize,
            &tle::ExtendedRuntimePerfKnobConfig::setCudaGraphCacheSize)
        .def("__getstate__", extendedRuntimePerfKnobConfigGetstate)
        .def("__setstate__", extendedRuntimePerfKnobConfigSetstate);

    // Guided decoding config
    auto pyGuidedDecodingConfig = nb::class_<tle::GuidedDecodingConfig>(m, "GuidedDecodingConfig");

    nb::enum_<tle::GuidedDecodingConfig::GuidedDecodingBackend>(pyGuidedDecodingConfig, "GuidedDecodingBackend")
        .value("XGRAMMAR", tle::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR)
        .value("LLGUIDANCE", tle::GuidedDecodingConfig::GuidedDecodingBackend::kLLGUIDANCE);

    auto guidedDecodingConfigGetstate = [](tle::GuidedDecodingConfig const& self) {
        return nb::make_tuple(
            self.getBackend(), self.getEncodedVocab(), self.getTokenizerStr(), self.getStopTokenIds());
    };
    auto guidedDecodingConfigSetstate = [](tle::GuidedDecodingConfig& self, nb::tuple state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid GuidedDecodingConfig state!");
        }
        new (&self) tle::GuidedDecodingConfig(nb::cast<tle::GuidedDecodingConfig::GuidedDecodingBackend>(state[0]),
            nb::cast<std::optional<std::vector<std::string>>>(state[1]), nb::cast<std::optional<std::string>>(state[2]),
            nb::cast<std::optional<std::vector<tle::TokenIdType>>>(state[3]));
    };

    pyGuidedDecodingConfig
        .def(nb::init<tle::GuidedDecodingConfig::GuidedDecodingBackend, std::optional<std::vector<std::string>>,
                 std::optional<std::string>, std::optional<std::vector<tle::TokenIdType>>>(),
            nb::arg("backend"), nb::arg("encoded_vocab") = nb::none(), nb::arg("tokenizer_str") = nb::none(),
            nb::arg("stop_token_ids") = nb::none())
        .def_prop_rw("backend", &tle::GuidedDecodingConfig::getBackend, &tle::GuidedDecodingConfig::setBackend)
        .def_prop_rw(
            "encoded_vocab", &tle::GuidedDecodingConfig::getEncodedVocab, &tle::GuidedDecodingConfig::setEncodedVocab)
        .def_prop_rw(
            "tokenizer_str", &tle::GuidedDecodingConfig::getTokenizerStr, &tle::GuidedDecodingConfig::setTokenizerStr)
        .def_prop_rw(
            "stop_token_ids", &tle::GuidedDecodingConfig::getStopTokenIds, &tle::GuidedDecodingConfig::setStopTokenIds)
        .def("__getstate__", guidedDecodingConfigGetstate)
        .def("__setstate__", guidedDecodingConfigSetstate);

    auto cacheTransceiverConfigGetstate = [](tle::CacheTransceiverConfig const& self)
    {
        return nb::make_tuple(self.getBackendType(), self.getMaxTokensInBuffer(), self.getKvTransferTimeoutMs(),
            self.getKvTransferSenderFutureTimeoutMs(), self.getKvTransferPollIntervalMs());
    };
    auto cacheTransceiverConfigSetstate = [](tle::CacheTransceiverConfig& self, nb::tuple const& state)
    {
        if (state.size() < 3 || state.size() > 5)
        {
            throw std::runtime_error("Invalid CacheTransceiverConfig state!");
        }
        auto kvTransferSenderFutureTimeoutMs
            = state.size() >= 4 ? nb::cast<std::optional<int>>(state[3]) : std::optional<int>{std::nullopt};
        auto kvTransferPollIntervalMs = state.size() >= 5
            ? nb::cast<std::optional<int>>(state[4])
            : std::optional<int>{tle::CacheTransceiverConfig::kDefaultKvTransferPollIntervalMs};
        auto backendType = nb::cast<std::optional<tle::CacheTransceiverConfig::BackendType>>(state[0]);
        new (&self) tle::CacheTransceiverConfig(backendType, nb::cast<std::optional<size_t>>(state[1]),
            nb::cast<std::optional<int>>(state[2]), kvTransferSenderFutureTimeoutMs, kvTransferPollIntervalMs);
    };

    nb::enum_<tle::CacheTransceiverConfig::BackendType>(m, "CacheTransceiverBackendType")
        .value("DEFAULT", tle::CacheTransceiverConfig::BackendType::DEFAULT)
        .value("MPI", tle::CacheTransceiverConfig::BackendType::MPI)
        .value("UCX", tle::CacheTransceiverConfig::BackendType::UCX)
        .value("NIXL", tle::CacheTransceiverConfig::BackendType::NIXL)
        .value("MOONCAKE", tle::CacheTransceiverConfig::BackendType::MOONCAKE)
        .def("from_string",
            [](std::string const& str)
            {
                if (str == "DEFAULT" || str == "default")
                    return tle::CacheTransceiverConfig::BackendType::DEFAULT;
                if (str == "MPI" || str == "mpi")
                    return tle::CacheTransceiverConfig::BackendType::MPI;
                if (str == "UCX" || str == "ucx")
                    return tle::CacheTransceiverConfig::BackendType::UCX;
                if (str == "NIXL" || str == "nixl")
                    return tle::CacheTransceiverConfig::BackendType::NIXL;
                if (str == "MOONCAKE" || str == "mooncake")
                    return tle::CacheTransceiverConfig::BackendType::MOONCAKE;
                throw std::runtime_error("Invalid backend type: " + str);
            });

    nb::class_<tle::CacheTransceiverConfig>(m, "CacheTransceiverConfig")
        .def(nb::init<std::optional<tle::CacheTransceiverConfig::BackendType>, std::optional<size_t>,
                 std::optional<int>, std::optional<int>, std::optional<int>>(),
            nb::arg("backend") = std::nullopt, nb::arg("max_tokens_in_buffer") = std::nullopt,
            nb::arg("kv_transfer_timeout_ms") = std::nullopt,
            nb::arg("kv_transfer_sender_future_timeout_ms") = std::nullopt,
            nb::arg("kv_transfer_poll_interval_ms") = tle::CacheTransceiverConfig::kDefaultKvTransferPollIntervalMs)
        .def_prop_rw(
            "backend", &tle::CacheTransceiverConfig::getBackendType, &tle::CacheTransceiverConfig::setBackendType)
        .def_prop_rw("max_tokens_in_buffer", &tle::CacheTransceiverConfig::getMaxTokensInBuffer,
            &tle::CacheTransceiverConfig::setMaxTokensInBuffer)
        .def_prop_rw("kv_transfer_timeout_ms", &tle::CacheTransceiverConfig::getKvTransferTimeoutMs,
            &tle::CacheTransceiverConfig::setKvTransferTimeoutMs)
        .def_prop_rw("kv_transfer_sender_future_timeout_ms",
            &tle::CacheTransceiverConfig::getKvTransferSenderFutureTimeoutMs,
            &tle::CacheTransceiverConfig::setKvTransferSenderFutureTimeoutMs)
        .def_prop_rw("kv_transfer_poll_interval_ms", &tle::CacheTransceiverConfig::getKvTransferPollIntervalMs,
            &tle::CacheTransceiverConfig::setKvTransferPollIntervalMs)
        .def("__getstate__", cacheTransceiverConfigGetstate)
        .def("__setstate__", cacheTransceiverConfigSetstate);
}

} // namespace tensorrt_llm::nanobind::executor
