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
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::SchedulerConfig(nb::cast<tle::CapacitySchedulerPolicy>(state[0]),
            nb::cast<std::optional<tle::ContextChunkingPolicy>>(state[1]),
            nb::cast<std::optional<tle::DynamicBatchConfig>>(state[2]));
    };
    auto schedulerConfigGetstate = [](tle::SchedulerConfig const& self)
    {
        return nb::make_tuple(
            self.getCapacitySchedulerPolicy(), self.getContextChunkingPolicy(), self.getDynamicBatchConfig());
    };
    nb::class_<tle::SchedulerConfig>(m, "SchedulerConfig")
        .def(nb::init<tle::CapacitySchedulerPolicy, std::optional<tle::ContextChunkingPolicy>,
                 std::optional<tle::DynamicBatchConfig>>(),
            nb::arg("capacity_scheduler_policy") = tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
            nb::arg("context_chunking_policy") = nb::none(), nb::arg("dynamic_batch_config") = nb::none())
        .def_prop_ro("capacity_scheduler_policy", &tle::SchedulerConfig::getCapacitySchedulerPolicy)
        .def_prop_ro("context_chunking_policy", &tle::SchedulerConfig::getContextChunkingPolicy)
        .def_prop_ro("dynamic_batch_config", &tle::SchedulerConfig::getDynamicBatchConfig)
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
            self.getOnboardBlocks(), self.getCrossKvCacheFraction(), self.getSecondaryOffloadMinPriority(),
            self.getEventBufferMaxSize(), self.getEnablePartialReuse(), self.getCopyOnPartialReuse(), self.getUseUvm(),
            self.getAttentionDpEventsGatherPeriodMs(), self.getMaxGpuTotalBytes());
    };
    auto kvCacheConfigSetstate = [](tle::KvCacheConfig& self, nb::tuple const& state)
    {
        if (state.size() != 15)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::KvCacheConfig(nb::cast<bool>(state[0]), nb::cast<std::optional<SizeType32>>(state[1]),
            nb::cast<std::optional<std::vector<SizeType32>>>(state[2]), nb::cast<std::optional<SizeType32>>(state[3]),
            nb::cast<std::optional<float>>(state[4]), nb::cast<std::optional<size_t>>(state[5]),
            nb::cast<bool>(state[6]), nb::cast<std::optional<float>>(state[7]),
            nb::cast<std::optional<tle::RetentionPriority>>(state[8]), nb::cast<size_t>(state[9]),
            nb::cast<bool>(state[10]), nb::cast<bool>(state[11]), nb::cast<bool>(state[12]),
            nb::cast<SizeType32>(state[13]), std::nullopt, nb::cast<uint64_t>(state[14]));
    };
    nb::class_<tle::KvCacheConfig>(m, "KvCacheConfig")
        .def(nb::init<bool, std::optional<SizeType32> const&, std::optional<std::vector<SizeType32>> const&,
                 std::optional<SizeType32> const&, std::optional<float> const&, std::optional<size_t> const&, bool,
                 std::optional<float> const&, std::optional<tle::RetentionPriority>, size_t const&, bool, bool, bool,
                 SizeType32, std::optional<RuntimeDefaults> const&, uint64_t const&>(),
            nb::arg("enable_block_reuse") = true, nb::arg("max_tokens") = nb::none(),
            nb::arg("max_attention_window") = nb::none(), nb::arg("sink_token_length") = nb::none(),
            nb::arg("free_gpu_memory_fraction") = nb::none(), nb::arg("host_cache_size") = nb::none(),
            nb::arg("onboard_blocks") = true, nb::arg("cross_kv_cache_fraction") = nb::none(),
            nb::arg("secondary_offload_min_priority") = nb::none(), nb::arg("event_buffer_max_size") = 0, nb::kw_only(),
            nb::arg("enable_partial_reuse") = true, nb::arg("copy_on_partial_reuse") = true, nb::arg("use_uvm") = false,
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
        .def_prop_rw("onboard_blocks", &tle::KvCacheConfig::getOnboardBlocks, &tle::KvCacheConfig::setOnboardBlocks)
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

    nb::class_<tle::OrchestratorConfig>(m, "OrchestratorConfig")
        .def(nb::init<bool, std::string, std::shared_ptr<mpi::MpiComm>, bool>(), nb::arg("is_orchestrator") = true,
            nb::arg("worker_executable_path") = "", nb::arg("orch_leader_comm").none() = nullptr,
            nb::arg("spawn_processes") = true)
        .def_prop_rw(
            "is_orchestrator", &tle::OrchestratorConfig::getIsOrchestrator, &tle::OrchestratorConfig::setIsOrchestrator)
        .def_prop_rw("worker_executable_path", &tle::OrchestratorConfig::getWorkerExecutablePath,
            &tle::OrchestratorConfig::setWorkerExecutablePath)
        .def_prop_rw("orch_leader_comm", &tle::OrchestratorConfig::getOrchLeaderComm,
            &tle::OrchestratorConfig::setOrchLeaderComm)
        .def_prop_rw("spawn_processes", &tle::OrchestratorConfig::getSpawnProcesses,
            &tle::OrchestratorConfig::setSpawnProcesses);

    auto parallelConfigGetstate = [](tle::ParallelConfig const& self)
    {
        return nb::make_tuple(self.getCommunicationType(), self.getCommunicationMode(), self.getDeviceIds(),
            self.getParticipantIds(), self.getOrchestratorConfig(), self.getNumNodes());
    };
    auto parallelConfigSetstate = [](tle::ParallelConfig& self, nb::tuple const& state)
    {
        if (state.size() != 6)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::ParallelConfig(nb::cast<tle::CommunicationType>(state[0]),
            nb::cast<tle::CommunicationMode>(state[1]), nb::cast<std::optional<std::vector<SizeType32>>>(state[2]),
            nb::cast<std::optional<std::vector<SizeType32>>>(state[3]),
            nb::cast<std::optional<tle::OrchestratorConfig>>(state[4]), nb::cast<std::optional<SizeType32>>(state[5]));
    };
    nb::class_<tle::ParallelConfig>(m, "ParallelConfig")
        .def(nb::init<tle::CommunicationType, tle::CommunicationMode, std::optional<std::vector<SizeType32>> const&,
                 std::optional<std::vector<SizeType32>> const&, std::optional<tle::OrchestratorConfig> const&,
                 std::optional<SizeType32> const&>(),
            nb::arg("communication_type") = tle::CommunicationType::kMPI,
            nb::arg("communication_mode") = tle::CommunicationMode::kLEADER, nb::arg("device_ids") = nb::none(),
            nb::arg("participant_ids") = nb::none(), nb::arg("orchestrator_config") = nb::none(),
            nb::arg("num_nodes") = nb::none())
        .def_prop_rw("communication_type", &tle::ParallelConfig::getCommunicationType,
            &tle::ParallelConfig::setCommunicationType)
        .def_prop_rw("communication_mode", &tle::ParallelConfig::getCommunicationMode,
            &tle::ParallelConfig::setCommunicationMode)
        .def_prop_rw("device_ids", &tle::ParallelConfig::getDeviceIds, &tle::ParallelConfig::setDeviceIds)
        .def_prop_rw(
            "participant_ids", &tle::ParallelConfig::getParticipantIds, &tle::ParallelConfig::setParticipantIds)
        .def_prop_rw("orchestrator_config", &tle::ParallelConfig::getOrchestratorConfig,
            &tle::ParallelConfig::setOrchestratorConfig)
        .def_prop_rw("num_nodes", &tle::ParallelConfig::getNumNodes, &tle::ParallelConfig::setNumNodes)
        .def("__getstate__", parallelConfigGetstate)
        .def("__setstate__", parallelConfigSetstate);

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
    {
        return nb::make_tuple(
            self.getDecodingMode(), self.getLookaheadDecodingConfig(), self.getMedusaChoices(), self.getEagleConfig());
    };
    auto decodingConfigSetstate = [](tle::DecodingConfig& self, nb::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::DecodingConfig(nb::cast<std::optional<tle::DecodingMode>>(state[0]), // DecodingMode
            nb::cast<std::optional<tle::LookaheadDecodingConfig>>(state[1]),                  // LookaheadDecodingConfig
            nb::cast<std::optional<tle::MedusaChoices>>(state[2]),                            // MedusaChoices
            nb::cast<std::optional<tle::EagleConfig>>(state[3])                               // EagleConfig
        );
    };
    nb::class_<tle::DecodingConfig>(m, "DecodingConfig")
        .def(nb::init<std::optional<tle::DecodingMode>, std::optional<tle::LookaheadDecodingConfig>,
                 std::optional<tle::MedusaChoices>, std::optional<tle::EagleConfig>>(),
            nb::arg("decoding_mode") = nb::none(), nb::arg("lookahead_decoding_config") = nb::none(),
            nb::arg("medusa_choices") = nb::none(), nb::arg("eagle_config") = nb::none())
        .def_prop_rw("decoding_mode", &tle::DecodingConfig::getDecodingMode, &tle::DecodingConfig::setDecodingMode)
        .def_prop_rw("lookahead_decoding_config", &tle::DecodingConfig::getLookaheadDecodingConfig,
            &tle::DecodingConfig::setLookaheadDecodingConfig)
        .def_prop_rw("medusa_choices", &tle::DecodingConfig::getMedusaChoices, &tle::DecodingConfig::setMedusaChoices)
        .def_prop_rw("eagle_config", &tle::DecodingConfig::getEagleConfig, &tle::DecodingConfig::setEagleConfig)
        .def("__getstate__", decodingConfigGetstate)
        .def("__setstate__", decodingConfigSetstate);

    auto debugConfigGetstate = [](tle::DebugConfig const& self)
    {
        return nb::make_tuple(self.getDebugInputTensors(), self.getDebugOutputTensors(), self.getDebugTensorNames(),
            self.getDebugTensorsMaxIterations());
    };
    auto debugConfigSetstate = [](tle::DebugConfig& self, nb::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&self) tle::DebugConfig(nb::cast<bool>(state[0]), nb::cast<bool>(state[1]),
            nb::cast<std::vector<std::string>>(state[2]), nb::cast<SizeType32>(state[3]));
    };
    nb::class_<tle::DebugConfig>(m, "DebugConfig")
        .def(nb::init<bool, bool, std::vector<std::string>, SizeType32>(), nb::arg("debug_input_tensors") = false,
            nb::arg("debug_output_tensors") = false, nb::arg("debug_tensor_names") = nb::none(),
            nb::arg("debug_tensors_max_iterations") = false)
        .def_prop_rw(
            "debug_input_tensors", &tle::DebugConfig::getDebugInputTensors, &tle::DebugConfig::setDebugInputTensors)
        .def_prop_rw(
            "debug_output_tensors", &tle::DebugConfig::getDebugOutputTensors, &tle::DebugConfig::setDebugOutputTensors)
        .def_prop_rw(
            "debug_tensor_names", &tle::DebugConfig::getDebugTensorNames, &tle::DebugConfig::setDebugTensorNames)
        .def_prop_rw("debug_tensors_max_iterations", &tle::DebugConfig::getDebugTensorsMaxIterations,
            &tle::DebugConfig::setDebugTensorsMaxIterations)
        .def("__getstate__", debugConfigGetstate)
        .def("__setstate__", debugConfigSetstate);

    auto logitsPostProcessorConfigGetstate = [](tle::LogitsPostProcessorConfig const& self)
    { return nb::make_tuple(self.getProcessorMap(), self.getProcessorBatched(), self.getReplicate()); };

    auto logitsPostProcessorConfigSetstate = [](tle::LogitsPostProcessorConfig& self, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LogitsPostProcessorConfig state!");
        }
        new (&self) tle::LogitsPostProcessorConfig(nb::cast<std::optional<tle::LogitsPostProcessorMap>>(state[0]),
            nb::cast<std::optional<tle::LogitsPostProcessorBatched>>(state[1]), nb::cast<bool>(state[2]));
    };

    nb::class_<tle::LogitsPostProcessorConfig>(m, "LogitsPostProcessorConfig")
        .def(nb::init<std::optional<tle::LogitsPostProcessorMap>, std::optional<tle::LogitsPostProcessorBatched>,
                 bool>(),
            nb::arg("processor_map") = nb::none(), nb::arg("processor_batched") = nb::none(),
            nb::arg("replicate") = true)
        .def_prop_rw("processor_map", &tle::LogitsPostProcessorConfig::getProcessorMap,
            &tle::LogitsPostProcessorConfig::setProcessorMap)
        .def_prop_rw("processor_batched", &tle::LogitsPostProcessorConfig::getProcessorBatched,
            &tle::LogitsPostProcessorConfig::setProcessorBatched)
        .def_prop_rw(
            "replicate", &tle::LogitsPostProcessorConfig::getReplicate, &tle::LogitsPostProcessorConfig::setReplicate)
        .def("__getstate__", logitsPostProcessorConfigGetstate)
        .def("__setstate__", logitsPostProcessorConfigSetstate);

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

    auto SpeculativeDecodingConfigGetState
        = [](tle::SpeculativeDecodingConfig const& self) { return nb::make_tuple(self.fastLogits); };
    auto SpeculativeDecodingConfigSetState = [](tle::SpeculativeDecodingConfig& self, nb::tuple const& state)
    {
        if (state.size() != 1)
        {
            throw std::runtime_error("Invalid SpeculativeDecodingConfig state!");
        }
        new (&self) tle::SpeculativeDecodingConfig(nb::cast<bool>(state[0]));
    };
    nb::class_<tle::SpeculativeDecodingConfig>(m, "SpeculativeDecodingConfig")
        .def(nb::init<bool>(), nb::arg("fast_logits") = false)
        .def_rw("fast_logits", &tle::SpeculativeDecodingConfig::fastLogits)
        .def("__getstate__", SpeculativeDecodingConfigGetState)
        .def("__setstate__", SpeculativeDecodingConfigSetState);

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
    { return nb::make_tuple(self.getBackendType(), self.getMaxTokensInBuffer(), self.getKvTransferTimeoutMs()); };
    auto cacheTransceiverConfigSetstate = [](tle::CacheTransceiverConfig& self, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid CacheTransceiverConfig state!");
        }
        new (&self) tle::CacheTransceiverConfig(nb::cast<tle::CacheTransceiverConfig::BackendType>(state[0]),
            nb::cast<std::optional<size_t>>(state[1]), nb::cast<std::optional<int>>(state[2]));
    };

    nb::enum_<tle::CacheTransceiverConfig::BackendType>(m, "CacheTransceiverBackendType")
        .value("DEFAULT", tle::CacheTransceiverConfig::BackendType::DEFAULT)
        .value("MPI", tle::CacheTransceiverConfig::BackendType::MPI)
        .value("UCX", tle::CacheTransceiverConfig::BackendType::UCX)
        .value("NIXL", tle::CacheTransceiverConfig::BackendType::NIXL)
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
                throw std::runtime_error("Invalid backend type: " + str);
            });

    nb::class_<tle::CacheTransceiverConfig>(m, "CacheTransceiverConfig")
        .def(nb::init<std::optional<tle::CacheTransceiverConfig::BackendType>, std::optional<size_t>,
                 std::optional<int>>(),
            nb::arg("backend") = std::nullopt, nb::arg("max_tokens_in_buffer") = std::nullopt,
            nb::arg("kv_transfer_timeout_ms") = std::nullopt)
        .def_prop_rw(
            "backend", &tle::CacheTransceiverConfig::getBackendType, &tle::CacheTransceiverConfig::setBackendType)
        .def_prop_rw("max_tokens_in_buffer", &tle::CacheTransceiverConfig::getMaxTokensInBuffer,
            &tle::CacheTransceiverConfig::setMaxTokensInBuffer)
        .def_prop_rw("kv_transfer_timeout_ms", &tle::CacheTransceiverConfig::getKvTransferTimeoutMs,
            &tle::CacheTransceiverConfig::setKvTransferTimeoutMs)
        .def("__getstate__", cacheTransceiverConfigGetstate)
        .def("__setstate__", cacheTransceiverConfigSetstate);

    auto executorConfigGetState = [](nb::object const& self)
    {
        auto& c = nb::cast<tle::ExecutorConfig&>(self);
        // Return a tuple containing C++ data and the Python __dict__
        auto cpp_states = nb::make_tuple(c.getMaxBeamWidth(), c.getSchedulerConfig(), c.getKvCacheConfig(),
            c.getEnableChunkedContext(), c.getNormalizeLogProbs(), c.getIterStatsMaxIterations(),
            c.getRequestStatsMaxIterations(), c.getBatchingType(), c.getMaxBatchSize(), c.getMaxNumTokens(),
            c.getParallelConfig(), c.getPeftCacheConfig(), c.getLogitsPostProcessorConfig(), c.getDecodingConfig(),
            c.getUseGpuDirectStorage(), c.getGpuWeightsPercent(), c.getMaxQueueSize(),
            c.getExtendedRuntimePerfKnobConfig(), c.getDebugConfig(), c.getRecvPollPeriodMs(),
            c.getMaxSeqIdleMicroseconds(), c.getSpecDecConfig(), c.getGuidedDecodingConfig(),
            c.getAdditionalModelOutputs(), c.getCacheTransceiverConfig(), c.getGatherGenerationLogits(),
            c.getPromptTableOffloading(), c.getEnableTrtOverlap(), c.getFailFastOnAttentionWindowTooLarge());
        auto pickle_tuple = nb::make_tuple(cpp_states, nb::getattr(self, "__dict__"));
        return pickle_tuple;
    };

    auto executorConfigSetState = [](nb::object self, nb::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid state!");
        }

        auto cpp_states = nb::cast<nb::tuple>(state[0]);
        if (cpp_states.size() != 29)
        {
            throw std::runtime_error("Invalid cpp_states!");
        }

        // Restore C++ data
        tle::ExecutorConfig* cpp_self = nb::inst_ptr<tle::ExecutorConfig>(self);
        new (cpp_self) tle::ExecutorConfig(                                          //
            nb::cast<SizeType32>(cpp_states[0]),                                     // MaxBeamWidth
            nb::cast<tle::SchedulerConfig>(cpp_states[1]),                           // SchedulerConfig
            nb::cast<tle::KvCacheConfig>(cpp_states[2]),                             // KvCacheConfig
            nb::cast<bool>(cpp_states[3]),                                           // EnableChunkedContext
            nb::cast<bool>(cpp_states[4]),                                           // NormalizeLogProbs
            nb::cast<SizeType32>(cpp_states[5]),                                     // IterStatsMaxIterations
            nb::cast<SizeType32>(cpp_states[6]),                                     // RequestStatsMaxIterations
            nb::cast<tle::BatchingType>(cpp_states[7]),                              // BatchingType
            nb::cast<std::optional<SizeType32>>(cpp_states[8]),                      // MaxBatchSize
            nb::cast<std::optional<SizeType32>>(cpp_states[9]),                      // MaxNumTokens
            nb::cast<std::optional<tle::ParallelConfig>>(cpp_states[10]),            // ParallelConfig
            nb::cast<std::optional<tle::PeftCacheConfig>>(cpp_states[11]),           // PeftCacheConfig
            nb::cast<std::optional<tle::LogitsPostProcessorConfig>>(cpp_states[12]), // LogitsPostProcessorConfig
            nb::cast<std::optional<tle::DecodingConfig>>(cpp_states[13]),            // DecodingConfig
            nb::cast<bool>(cpp_states[14]),                                          // UseGpuDirectStorage
            nb::cast<float>(cpp_states[15]),                                         // GpuWeightsPercent
            nb::cast<std::optional<SizeType32>>(cpp_states[16]),                     // MaxQueueSize
            nb::cast<tle::ExtendedRuntimePerfKnobConfig>(cpp_states[17]),            // ExtendedRuntimePerfKnobConfig
            nb::cast<std::optional<tle::DebugConfig>>(cpp_states[18]),               // DebugConfig
            nb::cast<SizeType32>(cpp_states[19]),                                    // RecvPollPeriodMs
            nb::cast<uint64_t>(cpp_states[20]),                                      // MaxSeqIdleMicroseconds
            nb::cast<std::optional<tle::SpeculativeDecodingConfig>>(cpp_states[21]), // SpecDecConfig
            nb::cast<std::optional<tle::GuidedDecodingConfig>>(cpp_states[22]),      // GuidedDecodingConfig
            nb::cast<std::optional<std::vector<tle::AdditionalModelOutput>>>(cpp_states[23]), // AdditionalModelOutputs
            nb::cast<std::optional<tle::CacheTransceiverConfig>>(cpp_states[24]),             // CacheTransceiverConfig
            nb::cast<bool>(cpp_states[25]),                                                   // GatherGenerationLogits
            nb::cast<bool>(cpp_states[26]),                                                   // PromptTableOffloading
            nb::cast<bool>(cpp_states[27]),                                                   // EnableTrtOverlap
            nb::cast<bool>(cpp_states[28]) // FailFastOnAttentionWindowTooLarge
        );

        // Restore Python data
        auto py_state = nb::cast<nb::dict>(state[1]);
        self.attr("__dict__").attr("update")(py_state);

        nb::inst_mark_ready(self);
    };

    nb::class_<tle::ExecutorConfig>(m, "ExecutorConfig", nb::dynamic_attr())
        .def(nb::init<                                                   //
                 SizeType32,                                             // MaxBeamWidth
                 tle::SchedulerConfig const&,                            // SchedulerConfig
                 tle::KvCacheConfig const&,                              // KvCacheConfig
                 bool,                                                   // EnableChunkedContext
                 bool,                                                   // NormalizeLogProbs
                 SizeType32,                                             // IterStatsMaxIterations
                 SizeType32,                                             // RequestStatsMaxIterations
                 tle::BatchingType,                                      // BatchingType
                 std::optional<SizeType32>,                              // MaxBatchSize
                 std::optional<SizeType32>,                              // MaxNumTokens
                 std::optional<tle::ParallelConfig>,                     // ParallelConfig
                 tle::PeftCacheConfig const&,                            // PeftCacheConfig
                 std::optional<tle::LogitsPostProcessorConfig>,          // LogitsPostProcessorConfig
                 std::optional<tle::DecodingConfig>,                     // DecodingConfig
                 bool,                                                   // UseGpuDirectStorage
                 float,                                                  // GpuWeightsPercent
                 std::optional<SizeType32>,                              // MaxQueueSize
                 tle::ExtendedRuntimePerfKnobConfig const&,              // ExtendedRuntimePerfKnobConfig
                 std::optional<tle::DebugConfig>,                        // DebugConfig
                 SizeType32,                                             // RecvPollPeriodMs
                 uint64_t,                                               // MaxSeqIdleMicroseconds
                 std::optional<tle::SpeculativeDecodingConfig>,          // SpecDecConfig
                 std::optional<tle::GuidedDecodingConfig>,               // GuidedDecodingConfig
                 std::optional<std::vector<tle::AdditionalModelOutput>>, // AdditionalModelOutputs
                 std::optional<tle::CacheTransceiverConfig>,             // CacheTransceiverConfig
                 bool,                                                   // GatherGenerationLogits
                 bool,                                                   // PromptTableOffloading
                 bool,                                                   // EnableTrtOverlap
                 bool                                                    // FailFastOnAttentionWindowTooLarge
                 >(),
            nb::arg("max_beam_width") = 1, nb::arg("scheduler_config") = tle::SchedulerConfig(),
            nb::arg("kv_cache_config") = tle::KvCacheConfig(), nb::arg("enable_chunked_context") = false,
            nb::arg("normalize_log_probs") = true,
            nb::arg("iter_stats_max_iterations") = tle::ExecutorConfig::kDefaultIterStatsMaxIterations,
            nb::arg("request_stats_max_iterations") = tle::ExecutorConfig::kDefaultRequestStatsMaxIterations,
            nb::arg("batching_type") = tle::BatchingType::kINFLIGHT, nb::arg("max_batch_size") = nb::none(),
            nb::arg("max_num_tokens") = nb::none(), nb::arg("parallel_config") = nb::none(),
            nb::arg("peft_cache_config") = tle::PeftCacheConfig(), nb::arg("logits_post_processor_config") = nb::none(),
            nb::arg("decoding_config") = nb::none(), nb::arg("use_gpu_direct_storage") = false,
            nb::arg("gpu_weights_percent") = 1.0, nb::arg("max_queue_size") = nb::none(),
            nb::arg("extended_runtime_perf_knob_config") = tle::ExtendedRuntimePerfKnobConfig(),
            nb::arg("debug_config") = nb::none(), nb::arg("recv_poll_period_ms") = 0,
            nb::arg("max_seq_idle_microseconds") = tle::ExecutorConfig::kDefaultMaxSeqIdleMicroseconds,
            nb::arg("spec_dec_config") = nb::none(), nb::arg("guided_decoding_config") = nb::none(),
            nb::arg("additional_model_outputs") = nb::none(), nb::arg("cache_transceiver_config") = nb::none(),
            nb::arg("gather_generation_logits") = false, nb::arg("mm_embedding_offloading") = false,
            nb::arg("enable_trt_overlap") = false, nb::arg("fail_fast_on_attention_window_too_large") = false)
        .def_prop_rw("max_beam_width", &tle::ExecutorConfig::getMaxBeamWidth, &tle::ExecutorConfig::setMaxBeamWidth)
        .def_prop_rw("max_batch_size", &tle::ExecutorConfig::getMaxBatchSize, &tle::ExecutorConfig::setMaxBatchSize)
        .def_prop_rw("max_num_tokens", &tle::ExecutorConfig::getMaxNumTokens, &tle::ExecutorConfig::setMaxNumTokens)
        .def_prop_rw(
            "scheduler_config", &tle::ExecutorConfig::getSchedulerConfigRef, &tle::ExecutorConfig::setSchedulerConfig)
        .def_prop_rw(
            "kv_cache_config", &tle::ExecutorConfig::getKvCacheConfigRef, &tle::ExecutorConfig::setKvCacheConfig)
        .def_prop_rw("enable_chunked_context", &tle::ExecutorConfig::getEnableChunkedContext,
            &tle::ExecutorConfig::setEnableChunkedContext)
        .def_prop_rw("normalize_log_probs", &tle::ExecutorConfig::getNormalizeLogProbs,
            &tle::ExecutorConfig::setNormalizeLogProbs)
        .def_prop_rw("iter_stats_max_iterations", &tle::ExecutorConfig::getIterStatsMaxIterations,
            &tle::ExecutorConfig::setIterStatsMaxIterations)
        .def_prop_rw("request_stats_max_iterations", &tle::ExecutorConfig::getRequestStatsMaxIterations,
            &tle::ExecutorConfig::setRequestStatsMaxIterations)
        .def_prop_rw("batching_type", &tle::ExecutorConfig::getBatchingType, &tle::ExecutorConfig::setBatchingType)
        .def_prop_rw(
            "parallel_config", &tle::ExecutorConfig::getParallelConfig, &tle::ExecutorConfig::setParallelConfig)
        .def_prop_rw(
            "peft_cache_config", &tle::ExecutorConfig::getPeftCacheConfig, &tle::ExecutorConfig::setPeftCacheConfig)
        .def_prop_rw("logits_post_processor_config", &tle::ExecutorConfig::getLogitsPostProcessorConfig,
            &tle::ExecutorConfig::setLogitsPostProcessorConfig)
        .def_prop_rw(
            "decoding_config", &tle::ExecutorConfig::getDecodingConfig, &tle::ExecutorConfig::setDecodingConfig)
        .def_prop_rw("use_gpu_direct_storage", &tle::ExecutorConfig::getUseGpuDirectStorage,
            &tle::ExecutorConfig::setUseGpuDirectStorage)
        .def_prop_rw("gpu_weights_percent", &tle::ExecutorConfig::getGpuWeightsPercent,
            &tle::ExecutorConfig::setGpuWeightsPercent)
        .def_prop_rw("max_queue_size", &tle::ExecutorConfig::getMaxQueueSize, &tle::ExecutorConfig::setMaxQueueSize)
        .def_prop_rw("extended_runtime_perf_knob_config", &tle::ExecutorConfig::getExtendedRuntimePerfKnobConfig,
            &tle::ExecutorConfig::setExtendedRuntimePerfKnobConfig)
        .def_prop_rw("debug_config", &tle::ExecutorConfig::getDebugConfig, &tle::ExecutorConfig::setDebugConfig)
        .def_prop_rw(
            "recv_poll_period_ms", &tle::ExecutorConfig::getRecvPollPeriodMs, &tle::ExecutorConfig::setRecvPollPeriodMs)
        .def_prop_rw("max_seq_idle_microseconds", &tle::ExecutorConfig::getMaxSeqIdleMicroseconds,
            &tle::ExecutorConfig::setMaxSeqIdleMicroseconds)
        .def_prop_rw("spec_dec_config", &tle::ExecutorConfig::getSpecDecConfig, &tle::ExecutorConfig::setSpecDecConfig)
        .def_prop_rw("guided_decoding_config", &tle::ExecutorConfig::getGuidedDecodingConfig,
            &tle::ExecutorConfig::setGuidedDecodingConfig)
        .def_prop_rw("additional_model_outputs", &tle::ExecutorConfig::getAdditionalModelOutputs,
            &tle::ExecutorConfig::setAdditionalModelOutputs)
        .def_prop_rw("cache_transceiver_config", &tle::ExecutorConfig::getCacheTransceiverConfig,
            &tle::ExecutorConfig::setCacheTransceiverConfig)
        .def_prop_rw("gather_generation_logits", &tle::ExecutorConfig::getGatherGenerationLogits,
            &tle::ExecutorConfig::setGatherGenerationLogits)
        .def_prop_rw("mm_embedding_offloading", &tle::ExecutorConfig::getPromptTableOffloading,
            &tle::ExecutorConfig::setPromptTableOffloading)
        .def_prop_rw(
            "enable_trt_overlap", &tle::ExecutorConfig::getEnableTrtOverlap, &tle::ExecutorConfig::setEnableTrtOverlap)
        .def_prop_rw("fail_fast_on_attention_window_too_large",
            &tle::ExecutorConfig::getFailFastOnAttentionWindowTooLarge,
            &tle::ExecutorConfig::setFailFastOnAttentionWindowTooLarge)
        .def("__getstate__", executorConfigGetState)
        .def("__setstate__", executorConfigSetState);
}

} // namespace tensorrt_llm::nanobind::executor
