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

#include "executorConfig.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <optional>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <vector>

namespace py = pybind11;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tle::SizeType32;
using RuntimeDefaults = tensorrt_llm::runtime::RuntimeDefaults;

namespace tensorrt_llm::pybind::executor
{

void initConfigBindings(pybind11::module_& m)
{
    py::enum_<tle::BatchingType>(m, "BatchingType")
        .value("STATIC", tle::BatchingType::kSTATIC)
        .value("INFLIGHT", tle::BatchingType::kINFLIGHT);

    auto dynamicBatchConfigGetstate = [](tle::DynamicBatchConfig const& self)
    {
        return py::make_tuple(self.getEnableBatchSizeTuning(), self.getEnableMaxNumTokensTuning(),
            self.getDynamicBatchMovingAverageWindow(), self.getBatchSizeTable());
    };
    auto dynamicBatchConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DynamicBatchConfig(state[0].cast<bool>(), state[1].cast<bool>(), state[2].cast<SizeType32>(),
            state[3].cast<std::vector<std::pair<SizeType32, SizeType32>>>());
    };
    py::class_<tle::DynamicBatchConfig>(m, "DynamicBatchConfig")
        .def(py::init<bool, bool, SizeType32>(), py::arg("enable_batch_size_tuning"),
            py::arg("enable_max_num_tokens_tuning"), py::arg("dynamic_batch_moving_average_window"))
        .def_property_readonly("enable_batch_size_tuning", &tle::DynamicBatchConfig::getEnableBatchSizeTuning)
        .def_property_readonly("enable_max_num_tokens_tuning", &tle::DynamicBatchConfig::getEnableMaxNumTokensTuning)
        .def_property_readonly(
            "dynamic_batch_moving_average_window", &tle::DynamicBatchConfig::getDynamicBatchMovingAverageWindow)
        .def(py::pickle(dynamicBatchConfigGetstate, dynamicBatchConfigSetstate));

    auto schedulerConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::SchedulerConfig(state[0].cast<tle::CapacitySchedulerPolicy>(),
            state[1].cast<std::optional<tle::ContextChunkingPolicy>>(),
            state[2].cast<std::optional<tle::DynamicBatchConfig>>());
    };
    auto schedulerConfigGetstate = [](tle::SchedulerConfig const& self)
    {
        return py::make_tuple(
            self.getCapacitySchedulerPolicy(), self.getContextChunkingPolicy(), self.getDynamicBatchConfig());
    };
    py::class_<tle::SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<tle::CapacitySchedulerPolicy, std::optional<tle::ContextChunkingPolicy>,
                 std::optional<tle::DynamicBatchConfig>>(),
            py::arg_v("capacity_scheduler_policy", tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
                "CapacitySchedulerPolicy.GUARANTEED_NO_EVICT"),
            py::arg("context_chunking_policy") = py::none(), py::arg("dynamic_batch_config") = py::none())
        .def_property_readonly("capacity_scheduler_policy", &tle::SchedulerConfig::getCapacitySchedulerPolicy)
        .def_property_readonly("context_chunking_policy", &tle::SchedulerConfig::getContextChunkingPolicy)
        .def_property_readonly("dynamic_batch_config", &tle::SchedulerConfig::getDynamicBatchConfig)
        .def(py::pickle(schedulerConfigGetstate, schedulerConfigSetstate));

    py::class_<RuntimeDefaults>(m, "RuntimeDefaults")
        .def(py::init<std::optional<std::vector<SizeType32>>, std::optional<SizeType32>>(),
            py::arg("max_attention_window") = py::none(), py::arg("sink_token_length") = py::none())
        .def_readonly("max_attention_window", &RuntimeDefaults::maxAttentionWindowVec)
        .def_readonly("sink_token_length", &RuntimeDefaults::sinkTokenLength);

    auto kvCacheConfigGetstate = [](tle::KvCacheConfig const& self)
    {
        return py::make_tuple(self.getEnableBlockReuse(), self.getMaxTokens(), self.getMaxAttentionWindowVec(),
            self.getSinkTokenLength(), self.getFreeGpuMemoryFraction(), self.getHostCacheSize(),
            self.getOnboardBlocks(), self.getCrossKvCacheFraction(), self.getSecondaryOffloadMinPriority(),
            self.getEventBufferMaxSize(), self.getEnablePartialReuse(), self.getCopyOnPartialReuse(), self.getUseUvm(),
            self.getAttentionDpEventsGatherPeriodMs(), self.getMaxGpuTotalBytes());
    };
    auto kvCacheConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 15)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::KvCacheConfig(state[0].cast<bool>(), state[1].cast<std::optional<SizeType32>>(),
            state[2].cast<std::optional<std::vector<SizeType32>>>(), state[3].cast<std::optional<SizeType32>>(),
            state[4].cast<std::optional<float>>(), state[5].cast<std::optional<size_t>>(), state[6].cast<bool>(),
            state[7].cast<std::optional<float>>(), state[8].cast<std::optional<tle::RetentionPriority>>(),
            state[9].cast<size_t>(), state[10].cast<bool>(), state[11].cast<bool>(), state[12].cast<bool>(),
            state[13].cast<SizeType32>(), std::nullopt, state[14].cast<uint64_t>());
    };
    py::class_<tle::KvCacheConfig>(m, "KvCacheConfig")
        .def(py::init<bool, std::optional<SizeType32> const&, std::optional<std::vector<SizeType32>> const&,
                 std::optional<SizeType32> const&, std::optional<float> const&, std::optional<size_t> const&, bool,
                 std::optional<float> const&, std::optional<tle::RetentionPriority>, size_t const&, bool, bool, bool,
                 SizeType32, std::optional<RuntimeDefaults> const&, uint64_t const&>(),
            py::arg("enable_block_reuse") = true, py::arg("max_tokens") = py::none(),
            py::arg("max_attention_window") = py::none(), py::arg("sink_token_length") = py::none(),
            py::arg("free_gpu_memory_fraction") = py::none(), py::arg("host_cache_size") = py::none(),
            py::arg("onboard_blocks") = true, py::arg("cross_kv_cache_fraction") = py::none(),
            py::arg("secondary_offload_min_priority") = py::none(), py::arg("event_buffer_max_size") = 0, py::kw_only(),
            py::arg("enable_partial_reuse") = true, py::arg("copy_on_partial_reuse") = true, py::arg("use_uvm") = false,
            py::arg("attention_dp_events_gather_period_ms") = 5, py::arg("runtime_defaults") = py::none(),
            py::arg("max_gpu_total_bytes") = 0)
        .def_property(
            "enable_block_reuse", &tle::KvCacheConfig::getEnableBlockReuse, &tle::KvCacheConfig::setEnableBlockReuse)
        .def_property("max_tokens", &tle::KvCacheConfig::getMaxTokens, &tle::KvCacheConfig::setMaxTokens)
        .def_property("max_attention_window", &tle::KvCacheConfig::getMaxAttentionWindowVec,
            &tle::KvCacheConfig::setMaxAttentionWindowVec)
        .def_property(
            "sink_token_length", &tle::KvCacheConfig::getSinkTokenLength, &tle::KvCacheConfig::setSinkTokenLength)
        .def_property("free_gpu_memory_fraction", &tle::KvCacheConfig::getFreeGpuMemoryFraction,
            &tle::KvCacheConfig::setFreeGpuMemoryFraction)
        .def_property(
            "max_gpu_total_bytes", &tle::KvCacheConfig::getMaxGpuTotalBytes, &tle::KvCacheConfig::setMaxGpuTotalBytes)
        .def_property("host_cache_size", &tle::KvCacheConfig::getHostCacheSize, &tle::KvCacheConfig::setHostCacheSize)
        .def_property("onboard_blocks", &tle::KvCacheConfig::getOnboardBlocks, &tle::KvCacheConfig::setOnboardBlocks)
        .def_property("cross_kv_cache_fraction", &tle::KvCacheConfig::getCrossKvCacheFraction,
            &tle::KvCacheConfig::setCrossKvCacheFraction)
        .def_property("secondary_offload_min_priority", &tle::KvCacheConfig::getSecondaryOffloadMinPriority,
            &tle::KvCacheConfig::setSecondaryOffloadMinPriority)
        .def_property("event_buffer_max_size", &tle::KvCacheConfig::getEventBufferMaxSize,
            &tle::KvCacheConfig::setEventBufferMaxSize)
        .def_property("enable_partial_reuse", &tle::KvCacheConfig::getEnablePartialReuse,
            &tle::KvCacheConfig::setEnablePartialReuse)
        .def_property("copy_on_partial_reuse", &tle::KvCacheConfig::getCopyOnPartialReuse,
            &tle::KvCacheConfig::setCopyOnPartialReuse)
        .def_property("use_uvm", &tle::KvCacheConfig::getUseUvm, &tle::KvCacheConfig::setUseUvm)
        .def_property("attention_dp_events_gather_period_ms", &tle::KvCacheConfig::getAttentionDpEventsGatherPeriodMs,
            &tle::KvCacheConfig::setAttentionDpEventsGatherPeriodMs)
        .def("fill_empty_fields_from_runtime_defaults", &tle::KvCacheConfig::fillEmptyFieldsFromRuntimeDefaults)
        .def(py::pickle(kvCacheConfigGetstate, kvCacheConfigSetstate));

    py::class_<tle::OrchestratorConfig>(m, "OrchestratorConfig")
        .def(py::init<bool, std::string, std::shared_ptr<mpi::MpiComm>, bool>(), py::arg("is_orchestrator") = true,
            py::arg("worker_executable_path") = "", py::arg("orch_leader_comm") = nullptr,
            py::arg("spawn_processes") = true)
        .def_property(
            "is_orchestrator", &tle::OrchestratorConfig::getIsOrchestrator, &tle::OrchestratorConfig::setIsOrchestrator)
        .def_property("worker_executable_path", &tle::OrchestratorConfig::getWorkerExecutablePath,
            &tle::OrchestratorConfig::setWorkerExecutablePath)
        .def_property("orch_leader_comm", &tle::OrchestratorConfig::getOrchLeaderComm,
            &tle::OrchestratorConfig::setOrchLeaderComm)
        .def_property("spawn_processes", &tle::OrchestratorConfig::getSpawnProcesses,
            &tle::OrchestratorConfig::setSpawnProcesses);

    auto parallelConfigGetstate = [](tle::ParallelConfig const& self)
    {
        return py::make_tuple(self.getCommunicationType(), self.getCommunicationMode(), self.getDeviceIds(),
            self.getParticipantIds(), self.getOrchestratorConfig(), self.getNumNodes());
    };
    auto parallelConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 6)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::ParallelConfig(state[0].cast<tle::CommunicationType>(), state[1].cast<tle::CommunicationMode>(),
            state[2].cast<std::optional<std::vector<SizeType32>>>(),
            state[3].cast<std::optional<std::vector<SizeType32>>>(),
            state[4].cast<std::optional<tle::OrchestratorConfig>>(), state[5].cast<std::optional<SizeType32>>());
    };
    py::class_<tle::ParallelConfig>(m, "ParallelConfig")
        .def(py::init<tle::CommunicationType, tle::CommunicationMode, std::optional<std::vector<SizeType32>> const&,
                 std::optional<std::vector<SizeType32>> const&, std::optional<tle::OrchestratorConfig> const&,
                 std::optional<SizeType32> const&>(),
            py::arg_v("communication_type", tle::CommunicationType::kMPI, "CommunicationType.MPI"),
            py::arg_v("communication_mode", tle::CommunicationMode::kLEADER, "CommunicationMode.LEADER"),
            py::arg("device_ids") = py::none(), py::arg("participant_ids") = py::none(),
            py::arg("orchestrator_config") = py::none(), py::arg("num_nodes") = py::none())
        .def_property("communication_type", &tle::ParallelConfig::getCommunicationType,
            &tle::ParallelConfig::setCommunicationType)
        .def_property("communication_mode", &tle::ParallelConfig::getCommunicationMode,
            &tle::ParallelConfig::setCommunicationMode)
        .def_property("device_ids", &tle::ParallelConfig::getDeviceIds, &tle::ParallelConfig::setDeviceIds)
        .def_property(
            "participant_ids", &tle::ParallelConfig::getParticipantIds, &tle::ParallelConfig::setParticipantIds)
        .def_property("orchestrator_config", &tle::ParallelConfig::getOrchestratorConfig,
            &tle::ParallelConfig::setOrchestratorConfig)
        .def_property("num_nodes", &tle::ParallelConfig::getNumNodes, &tle::ParallelConfig::setNumNodes)
        .def(py::pickle(parallelConfigGetstate, parallelConfigSetstate));

    auto peftCacheConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 11)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::PeftCacheConfig(state[0].cast<SizeType32>(), state[1].cast<SizeType32>(),
            state[2].cast<SizeType32>(), state[3].cast<SizeType32>(), state[4].cast<SizeType32>(),
            state[5].cast<SizeType32>(), state[6].cast<SizeType32>(), state[7].cast<SizeType32>(),
            state[8].cast<SizeType32>(), state[9].cast<std::optional<float>>(),
            state[10].cast<std::optional<size_t>>());
    };
    auto peftCacheConfigGetstate = [](tle::PeftCacheConfig const& self)
    {
        return py::make_tuple(self.getNumHostModuleLayer(), self.getNumDeviceModuleLayer(),
            self.getOptimalAdapterSize(), self.getMaxAdapterSize(), self.getNumPutWorkers(), self.getNumEnsureWorkers(),
            self.getNumCopyStreams(), self.getMaxPagesPerBlockHost(), self.getMaxPagesPerBlockDevice(),
            self.getDeviceCachePercent(), self.getHostCacheSize());
    };
    py::class_<tle::PeftCacheConfig>(m, "PeftCacheConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, std::optional<float> const&, std::optional<size_t> const&,
                 std::optional<std::string> const&>(),
            py::arg("num_host_module_layer") = 0, py::arg("num_device_module_layer") = 0,
            py::arg("optimal_adapter_size") = 8, py::arg("max_adapter_size") = 64, py::arg("num_put_workers") = 1,
            py::arg("num_ensure_workers") = 1, py::arg("num_copy_streams") = 1,
            py::arg("max_pages_per_block_host") = 24, py::arg("max_pages_per_block_device") = 8,
            py::arg("device_cache_percent") = py::none(), py::arg("host_cache_size") = py::none(),
            py::arg("lora_prefetch_dir") = py::none())
        .def_property_readonly("num_host_module_layer", &tle::PeftCacheConfig::getNumHostModuleLayer)
        .def_property_readonly("num_device_module_layer", &tle::PeftCacheConfig::getNumDeviceModuleLayer)
        .def_property_readonly("optimal_adapter_size", &tle::PeftCacheConfig::getOptimalAdapterSize)
        .def_property_readonly("max_adapter_size", &tle::PeftCacheConfig::getMaxAdapterSize)
        .def_property_readonly("num_put_workers", &tle::PeftCacheConfig::getNumPutWorkers)
        .def_property_readonly("num_ensure_workers", &tle::PeftCacheConfig::getNumEnsureWorkers)
        .def_property_readonly("num_copy_streams", &tle::PeftCacheConfig::getNumCopyStreams)
        .def_property_readonly("max_pages_per_block_host", &tle::PeftCacheConfig::getMaxPagesPerBlockHost)
        .def_property_readonly("max_pages_per_block_device", &tle::PeftCacheConfig::getMaxPagesPerBlockDevice)
        .def_property_readonly("device_cache_percent", &tle::PeftCacheConfig::getDeviceCachePercent)
        .def_property_readonly("host_cache_size", &tle::PeftCacheConfig::getHostCacheSize)
        .def_property_readonly("lora_prefetch_dir", &tle::PeftCacheConfig::getLoraPrefetchDir)
        .def(py::pickle(peftCacheConfigGetstate, peftCacheConfigSetstate));

    auto decodingConfigGetstate = [](tle::DecodingConfig const& self)
    {
        return py::make_tuple(
            self.getDecodingMode(), self.getLookaheadDecodingConfig(), self.getMedusaChoices(), self.getEagleConfig());
    };
    auto decodingConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DecodingConfig(state[0].cast<std::optional<tle::DecodingMode>>(), // DecodingMode
            state[1].cast<std::optional<tle::LookaheadDecodingConfig>>(),             // LookaheadDecodingConfig
            state[2].cast<std::optional<tle::MedusaChoices>>(),                       // MedusaChoices
            state[3].cast<std::optional<tle::EagleConfig>>()                          // EagleConfig
        );
    };
    py::class_<tle::DecodingConfig>(m, "DecodingConfig")
        .def(py::init<std::optional<tle::DecodingMode>, std::optional<tle::LookaheadDecodingConfig>,
                 std::optional<tle::MedusaChoices>, std::optional<tle::EagleConfig>>(),
            py::arg("decoding_mode") = py::none(), py::arg("lookahead_decoding_config") = py::none(),
            py::arg("medusa_choices") = py::none(), py::arg("eagle_config") = py::none())
        .def_property("decoding_mode", &tle::DecodingConfig::getDecodingMode, &tle::DecodingConfig::setDecodingMode)
        .def_property("lookahead_decoding_config", &tle::DecodingConfig::getLookaheadDecodingConfig,
            &tle::DecodingConfig::setLookaheadDecodingConfig)
        .def_property("medusa_choices", &tle::DecodingConfig::getMedusaChoices, &tle::DecodingConfig::setMedusaChoices)
        .def_property("eagle_config", &tle::DecodingConfig::getEagleConfig, &tle::DecodingConfig::setEagleConfig)
        .def(py::pickle(decodingConfigGetstate, decodingConfigSetstate));

    auto debugConfigGetstate = [](tle::DebugConfig const& self)
    {
        return py::make_tuple(self.getDebugInputTensors(), self.getDebugOutputTensors(), self.getDebugTensorNames(),
            self.getDebugTensorsMaxIterations());
    };
    auto debugConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DebugConfig(state[0].cast<bool>(), state[1].cast<bool>(), state[2].cast<std::vector<std::string>>(),
            state[3].cast<SizeType32>());
    };
    py::class_<tle::DebugConfig>(m, "DebugConfig")
        .def(py::init<bool, bool, std::vector<std::string>, SizeType32>(), py::arg("debug_input_tensors") = false,
            py::arg("debug_output_tensors") = false, py::arg("debug_tensor_names") = py::none(),
            py::arg("debug_tensors_max_iterations") = false)
        .def_property(
            "debug_input_tensors", &tle::DebugConfig::getDebugInputTensors, &tle::DebugConfig::setDebugInputTensors)
        .def_property(
            "debug_output_tensors", &tle::DebugConfig::getDebugOutputTensors, &tle::DebugConfig::setDebugOutputTensors)
        .def_property(
            "debug_tensor_names", &tle::DebugConfig::getDebugTensorNames, &tle::DebugConfig::setDebugTensorNames)
        .def_property("debug_tensors_max_iterations", &tle::DebugConfig::getDebugTensorsMaxIterations,
            &tle::DebugConfig::setDebugTensorsMaxIterations)
        .def(py::pickle(debugConfigGetstate, debugConfigSetstate));

    auto logitsPostProcessorConfigGetstate = [](tle::LogitsPostProcessorConfig const& self)
    { return py::make_tuple(self.getProcessorMap(), self.getProcessorBatched(), self.getReplicate()); };

    auto logitsPostProcessorConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LogitsPostProcessorConfig state!");
        }
        return tle::LogitsPostProcessorConfig(state[0].cast<std::optional<tle::LogitsPostProcessorMap>>(),
            state[1].cast<std::optional<tle::LogitsPostProcessorBatched>>(), state[2].cast<bool>());
    };

    py::class_<tle::LogitsPostProcessorConfig>(m, "LogitsPostProcessorConfig")
        .def(py::init<std::optional<tle::LogitsPostProcessorMap>, std::optional<tle::LogitsPostProcessorBatched>,
                 bool>(),
            py::arg("processor_map") = py::none(), py::arg("processor_batched") = py::none(),
            py::arg("replicate") = true)
        .def_property("processor_map", &tle::LogitsPostProcessorConfig::getProcessorMap,
            &tle::LogitsPostProcessorConfig::setProcessorMap)
        .def_property("processor_batched", &tle::LogitsPostProcessorConfig::getProcessorBatched,
            &tle::LogitsPostProcessorConfig::setProcessorBatched)
        .def_property(
            "replicate", &tle::LogitsPostProcessorConfig::getReplicate, &tle::LogitsPostProcessorConfig::setReplicate)
        .def(py::pickle(logitsPostProcessorConfigGetstate, logitsPostProcessorConfigSetstate));

    auto extendedRuntimePerfKnobConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid extendedRuntimePerfKnobConfig state!");
        }
        return tle::ExtendedRuntimePerfKnobConfig(
            state[0].cast<bool>(), state[1].cast<bool>(), state[2].cast<bool>(), state[3].cast<SizeType32>());
    };
    auto extendedRuntimePerfKnobConfigGetstate = [](tle::ExtendedRuntimePerfKnobConfig const& self)
    {
        return py::make_tuple(self.getMultiBlockMode(), self.getEnableContextFMHAFP32Acc(), self.getCudaGraphMode(),
            self.getCudaGraphCacheSize());
    };
    py::class_<tle::ExtendedRuntimePerfKnobConfig>(m, "ExtendedRuntimePerfKnobConfig")
        .def(
            py::init<bool, bool>(), py::arg("multi_block_mode") = true, py::arg("enable_context_fmha_fp32_acc") = false)
        .def_property("multi_block_mode", &tle::ExtendedRuntimePerfKnobConfig::getMultiBlockMode,
            &tle::ExtendedRuntimePerfKnobConfig::setMultiBlockMode)
        .def_property("enable_context_fmha_fp32_acc", &tle::ExtendedRuntimePerfKnobConfig::getEnableContextFMHAFP32Acc,
            &tle::ExtendedRuntimePerfKnobConfig::setEnableContextFMHAFP32Acc)
        .def_property("cuda_graph_mode", &tle::ExtendedRuntimePerfKnobConfig::getCudaGraphMode,
            &tle::ExtendedRuntimePerfKnobConfig::setCudaGraphMode)
        .def_property("cuda_graph_cache_size", &tle::ExtendedRuntimePerfKnobConfig::getCudaGraphCacheSize,
            &tle::ExtendedRuntimePerfKnobConfig::setCudaGraphCacheSize)
        .def(py::pickle(extendedRuntimePerfKnobConfigGetstate, extendedRuntimePerfKnobConfigSetstate));

    auto SpeculativeDecodingConfigGetState
        = [](tle::SpeculativeDecodingConfig const& self) { return py::make_tuple(self.fastLogits); };
    auto SpeculativeDecodingConfigSetState = [](py::tuple const& state)
    {
        if (state.size() != 1)
        {
            throw std::runtime_error("Invalid SpeculativeDecodingConfig state!");
        }
        return tle::SpeculativeDecodingConfig(state[0].cast<bool>());
    };
    py::class_<tle::SpeculativeDecodingConfig>(m, "SpeculativeDecodingConfig")
        .def(py::init<bool>(), py::arg("fast_logits") = false)
        .def_readwrite("fast_logits", &tle::SpeculativeDecodingConfig::fastLogits)
        .def(py::pickle(SpeculativeDecodingConfigGetState, SpeculativeDecodingConfigSetState));

    // Guided decoding config
    auto pyGuidedDecodingConfig = py::class_<tle::GuidedDecodingConfig>(m, "GuidedDecodingConfig");

    py::enum_<tle::GuidedDecodingConfig::GuidedDecodingBackend>(pyGuidedDecodingConfig, "GuidedDecodingBackend")
        .value("XGRAMMAR", tle::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR)
        .value("LLGUIDANCE", tle::GuidedDecodingConfig::GuidedDecodingBackend::kLLGUIDANCE);

    auto guidedDecodingConfigGetstate = [](tle::GuidedDecodingConfig const& self) {
        return py::make_tuple(
            self.getBackend(), self.getEncodedVocab(), self.getTokenizerStr(), self.getStopTokenIds());
    };
    auto guidedDecodingConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid GuidedDecodingConfig state!");
        }
        return tle::GuidedDecodingConfig(state[0].cast<tle::GuidedDecodingConfig::GuidedDecodingBackend>(),
            state[1].cast<std::optional<std::vector<std::string>>>(), state[2].cast<std::optional<std::string>>(),
            state[3].cast<std::optional<std::vector<tle::TokenIdType>>>());
    };

    pyGuidedDecodingConfig
        .def(py::init<tle::GuidedDecodingConfig::GuidedDecodingBackend, std::optional<std::vector<std::string>>,
                 std::optional<std::string>, std::optional<std::vector<tle::TokenIdType>>>(),
            py::arg("backend"), py::arg("encoded_vocab") = py::none(), py::arg("tokenizer_str") = py::none(),
            py::arg("stop_token_ids") = py::none())
        .def_property("backend", &tle::GuidedDecodingConfig::getBackend, &tle::GuidedDecodingConfig::setBackend)
        .def_property(
            "encoded_vocab", &tle::GuidedDecodingConfig::getEncodedVocab, &tle::GuidedDecodingConfig::setEncodedVocab)
        .def_property(
            "tokenizer_str", &tle::GuidedDecodingConfig::getTokenizerStr, &tle::GuidedDecodingConfig::setTokenizerStr)
        .def_property(
            "stop_token_ids", &tle::GuidedDecodingConfig::getStopTokenIds, &tle::GuidedDecodingConfig::setStopTokenIds)
        .def(py::pickle(guidedDecodingConfigGetstate, guidedDecodingConfigSetstate));

    auto cacheTransceiverConfigGetstate = [](tle::CacheTransceiverConfig const& self)
    { return py::make_tuple(self.getBackendType(), self.getMaxTokensInBuffer(), self.getKvTransferTimeoutMs()); };
    auto cacheTransceiverConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid CacheTransceiverConfig state!");
        }
        return tle::CacheTransceiverConfig(state[0].cast<tle::CacheTransceiverConfig::BackendType>(),
            state[1].cast<std::optional<size_t>>(), state[2].cast<std::optional<int>>());
    };

    py::enum_<tle::CacheTransceiverConfig::BackendType>(m, "CacheTransceiverBackendType")
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

    py::class_<tle::CacheTransceiverConfig>(m, "CacheTransceiverConfig")
        .def(py::init<std::optional<tle::CacheTransceiverConfig::BackendType>, std::optional<size_t>,
                 std::optional<int>>(),
            py::arg("backend") = std::nullopt, py::arg("max_tokens_in_buffer") = std::nullopt,
            py::arg("kv_transfer_timeout_ms") = std::nullopt)
        .def_property(
            "backend", &tle::CacheTransceiverConfig::getBackendType, &tle::CacheTransceiverConfig::setBackendType)
        .def_property("max_tokens_in_buffer", &tle::CacheTransceiverConfig::getMaxTokensInBuffer,
            &tle::CacheTransceiverConfig::setMaxTokensInBuffer)
        .def_property("kv_transfer_timeout_ms", &tle::CacheTransceiverConfig::getKvTransferTimeoutMs,
            &tle::CacheTransceiverConfig::setKvTransferTimeoutMs)
        .def(py::pickle(cacheTransceiverConfigGetstate, cacheTransceiverConfigSetstate));

    auto executorConfigGetState = [](py::object const& self)
    {
        auto& c = self.cast<tle::ExecutorConfig&>();
        // Return a tuple containing C++ data and the Python __dict__
        auto cpp_states = py::make_tuple(c.getMaxBeamWidth(), c.getSchedulerConfig(), c.getKvCacheConfig(),
            c.getEnableChunkedContext(), c.getNormalizeLogProbs(), c.getIterStatsMaxIterations(),
            c.getRequestStatsMaxIterations(), c.getBatchingType(), c.getMaxBatchSize(), c.getMaxNumTokens(),
            c.getParallelConfig(), c.getPeftCacheConfig(), c.getLogitsPostProcessorConfig(), c.getDecodingConfig(),
            c.getUseGpuDirectStorage(), c.getGpuWeightsPercent(), c.getMaxQueueSize(),
            c.getExtendedRuntimePerfKnobConfig(), c.getDebugConfig(), c.getRecvPollPeriodMs(),
            c.getMaxSeqIdleMicroseconds(), c.getSpecDecConfig(), c.getGuidedDecodingConfig(),
            c.getAdditionalModelOutputs(), c.getCacheTransceiverConfig(), c.getGatherGenerationLogits(),
            c.getPromptTableOffloading(), c.getEnableTrtOverlap(), c.getFailFastOnAttentionWindowTooLarge());
        auto pickle_tuple = py::make_tuple(cpp_states, py::getattr(self, "__dict__"));
        return pickle_tuple;
    };
    auto executorConfigSetState = [](py::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid state!");
        }

        // Restore C++ data
        auto cpp_states = state[0].cast<py::tuple>();
        if (cpp_states.size() != 29)
        {
            throw std::runtime_error("Invalid cpp_states!");
        }

        auto ec = tle::ExecutorConfig(                                            //
            cpp_states[0].cast<SizeType32>(),                                     // MaxBeamWidth
            cpp_states[1].cast<tle::SchedulerConfig>(),                           // SchedulerConfig
            cpp_states[2].cast<tle::KvCacheConfig>(),                             // KvCacheConfig
            cpp_states[3].cast<bool>(),                                           // EnableChunkedContext
            cpp_states[4].cast<bool>(),                                           // NormalizeLogProbs
            cpp_states[5].cast<SizeType32>(),                                     // IterStatsMaxIterations
            cpp_states[6].cast<SizeType32>(),                                     // RequestStatsMaxIterations
            cpp_states[7].cast<tle::BatchingType>(),                              // BatchingType
            cpp_states[8].cast<std::optional<SizeType32>>(),                      // MaxBatchSize
            cpp_states[9].cast<std::optional<SizeType32>>(),                      // MaxNumTokens
            cpp_states[10].cast<std::optional<tle::ParallelConfig>>(),            // ParallelConfig
            cpp_states[11].cast<std::optional<tle::PeftCacheConfig>>(),           // PeftCacheConfig
            cpp_states[12].cast<std::optional<tle::LogitsPostProcessorConfig>>(), // LogitsPostProcessorConfig
            cpp_states[13].cast<std::optional<tle::DecodingConfig>>(),            // DecodingConfig
            cpp_states[14].cast<bool>(),                                          // UseGpuDirectStorage
            cpp_states[15].cast<float>(),                                         // GpuWeightsPercent
            cpp_states[16].cast<std::optional<SizeType32>>(),                     // MaxQueueSize
            cpp_states[17].cast<tle::ExtendedRuntimePerfKnobConfig>(),            // ExtendedRuntimePerfKnobConfig
            cpp_states[18].cast<std::optional<tle::DebugConfig>>(),               // DebugConfig
            cpp_states[19].cast<SizeType32>(),                                    // RecvPollPeriodMs
            cpp_states[20].cast<uint64_t>(),                                      // MaxSeqIdleMicroseconds
            cpp_states[21].cast<std::optional<tle::SpeculativeDecodingConfig>>(), // SpecDecConfig
            cpp_states[22].cast<std::optional<tle::GuidedDecodingConfig>>(),      // GuidedDecodingConfig
            cpp_states[23].cast<std::optional<std::vector<tle::AdditionalModelOutput>>>(), // AdditionalModelOutputs
            cpp_states[24].cast<std::optional<tle::CacheTransceiverConfig>>(),             // CacheTransceiverConfig
            cpp_states[25].cast<bool>(),                                                   // GatherGenerationLogits
            cpp_states[26].cast<bool>(),                                                   // PromptTableOffloading
            cpp_states[27].cast<bool>(),                                                   // EnableTrtOverlap
            cpp_states[28].cast<bool>() // FailFastOnAttentionWindowTooLarge
        );

        auto py_state = state[1].cast<py::dict>();

        return std::make_pair(ec, py_state);
    };

    py::class_<tle::ExecutorConfig>(m, "ExecutorConfig", pybind11::dynamic_attr())
        .def(py::init<                                                   //
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
            py::arg("max_beam_width") = 1, py::arg_v("scheduler_config", tle::SchedulerConfig(), "SchedulerConfig()"),
            py::arg_v("kv_cache_config", tle::KvCacheConfig(), "KvCacheConfig()"),
            py::arg("enable_chunked_context") = false, py::arg("normalize_log_probs") = true,
            py::arg("iter_stats_max_iterations") = tle::ExecutorConfig::kDefaultIterStatsMaxIterations,
            py::arg("request_stats_max_iterations") = tle::ExecutorConfig::kDefaultRequestStatsMaxIterations,
            py::arg_v("batching_type", tle::BatchingType::kINFLIGHT, "BatchingType.INFLIGHT"),
            py::arg("max_batch_size") = py::none(), py::arg("max_num_tokens") = py::none(),
            py::arg("parallel_config") = py::none(),
            py::arg_v("peft_cache_config", tle::PeftCacheConfig(), "PeftCacheConfig()"),
            py::arg("logits_post_processor_config") = py::none(), py::arg("decoding_config") = py::none(),
            py::arg("use_gpu_direct_storage") = false, py::arg("gpu_weights_percent") = 1.0,
            py::arg("max_queue_size") = py::none(),
            py::arg_v("extended_runtime_perf_knob_config", tle::ExtendedRuntimePerfKnobConfig(),
                "ExtendedRuntimePerfKnobConfig()"),
            py::arg("debug_config") = py::none(), py::arg("recv_poll_period_ms") = 0,
            py::arg("max_seq_idle_microseconds") = tle::ExecutorConfig::kDefaultMaxSeqIdleMicroseconds,
            py::arg("spec_dec_config") = py::none(), py::arg("guided_decoding_config") = py::none(),
            py::arg("additional_model_outputs") = py::none(), py::arg("cache_transceiver_config") = py::none(),
            py::arg("gather_generation_logits") = false, py::arg("mm_embedding_offloading") = false,
            py::arg("enable_trt_overlap") = false, py::arg("fail_fast_on_attention_window_too_large") = false)
        .def_property("max_beam_width", &tle::ExecutorConfig::getMaxBeamWidth, &tle::ExecutorConfig::setMaxBeamWidth)
        .def_property("max_batch_size", &tle::ExecutorConfig::getMaxBatchSize, &tle::ExecutorConfig::setMaxBatchSize)
        .def_property("max_num_tokens", &tle::ExecutorConfig::getMaxNumTokens, &tle::ExecutorConfig::setMaxNumTokens)
        .def_property(
            "scheduler_config", &tle::ExecutorConfig::getSchedulerConfigRef, &tle::ExecutorConfig::setSchedulerConfig)
        .def_property(
            "kv_cache_config", &tle::ExecutorConfig::getKvCacheConfigRef, &tle::ExecutorConfig::setKvCacheConfig)
        .def_property("enable_chunked_context", &tle::ExecutorConfig::getEnableChunkedContext,
            &tle::ExecutorConfig::setEnableChunkedContext)
        .def_property("normalize_log_probs", &tle::ExecutorConfig::getNormalizeLogProbs,
            &tle::ExecutorConfig::setNormalizeLogProbs)
        .def_property("iter_stats_max_iterations", &tle::ExecutorConfig::getIterStatsMaxIterations,
            &tle::ExecutorConfig::setIterStatsMaxIterations)
        .def_property("request_stats_max_iterations", &tle::ExecutorConfig::getRequestStatsMaxIterations,
            &tle::ExecutorConfig::setRequestStatsMaxIterations)
        .def_property("batching_type", &tle::ExecutorConfig::getBatchingType, &tle::ExecutorConfig::setBatchingType)
        .def_property(
            "parallel_config", &tle::ExecutorConfig::getParallelConfig, &tle::ExecutorConfig::setParallelConfig)
        .def_property(
            "peft_cache_config", &tle::ExecutorConfig::getPeftCacheConfig, &tle::ExecutorConfig::setPeftCacheConfig)
        .def_property("logits_post_processor_config", &tle::ExecutorConfig::getLogitsPostProcessorConfig,
            &tle::ExecutorConfig::setLogitsPostProcessorConfig)
        .def_property(
            "decoding_config", &tle::ExecutorConfig::getDecodingConfig, &tle::ExecutorConfig::setDecodingConfig)
        .def_property("use_gpu_direct_storage", &tle::ExecutorConfig::getUseGpuDirectStorage,
            &tle::ExecutorConfig::setUseGpuDirectStorage)
        .def_property("gpu_weights_percent", &tle::ExecutorConfig::getGpuWeightsPercent,
            &tle::ExecutorConfig::setGpuWeightsPercent)
        .def_property("max_queue_size", &tle::ExecutorConfig::getMaxQueueSize, &tle::ExecutorConfig::setMaxQueueSize)
        .def_property("extended_runtime_perf_knob_config", &tle::ExecutorConfig::getExtendedRuntimePerfKnobConfig,
            &tle::ExecutorConfig::setExtendedRuntimePerfKnobConfig)
        .def_property("debug_config", &tle::ExecutorConfig::getDebugConfig, &tle::ExecutorConfig::setDebugConfig)
        .def_property(
            "recv_poll_period_ms", &tle::ExecutorConfig::getRecvPollPeriodMs, &tle::ExecutorConfig::setRecvPollPeriodMs)
        .def_property("max_seq_idle_microseconds", &tle::ExecutorConfig::getMaxSeqIdleMicroseconds,
            &tle::ExecutorConfig::setMaxSeqIdleMicroseconds)
        .def_property("spec_dec_config", &tle::ExecutorConfig::getSpecDecConfig, &tle::ExecutorConfig::setSpecDecConfig)
        .def_property("guided_decoding_config", &tle::ExecutorConfig::getGuidedDecodingConfig,
            &tle::ExecutorConfig::setGuidedDecodingConfig)
        .def_property("additional_model_outputs", &tle::ExecutorConfig::getAdditionalModelOutputs,
            &tle::ExecutorConfig::setAdditionalModelOutputs)
        .def_property("cache_transceiver_config", &tle::ExecutorConfig::getCacheTransceiverConfig,
            &tle::ExecutorConfig::setCacheTransceiverConfig)
        .def_property("gather_generation_logits", &tle::ExecutorConfig::getGatherGenerationLogits,
            &tle::ExecutorConfig::setGatherGenerationLogits)
        .def_property("mm_embedding_offloading", &tle::ExecutorConfig::getPromptTableOffloading,
            &tle::ExecutorConfig::setPromptTableOffloading)
        .def_property(
            "enable_trt_overlap", &tle::ExecutorConfig::getEnableTrtOverlap, &tle::ExecutorConfig::setEnableTrtOverlap)
        .def_property("fail_fast_on_attention_window_too_large",
            &tle::ExecutorConfig::getFailFastOnAttentionWindowTooLarge,
            &tle::ExecutorConfig::setFailFastOnAttentionWindowTooLarge)
        .def(py::pickle(executorConfigGetState, executorConfigSetState));
}

} // namespace tensorrt_llm::pybind::executor
