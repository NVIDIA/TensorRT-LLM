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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "bindings.h"
#include "executor.h"
#include "streamCaster.h"
#include "tensorCaster.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

namespace py = pybind11;
namespace tle = tensorrt_llm::executor;
using Tensor = tle::Tensor;
using SizeType = tle::SizeType;
using FloatType = tle::FloatType;
using VecTokens = tle::VecTokens;
using IdType = tle::IdType;

namespace tensorrt_llm::pybind::executor
{

void InitBindings(pybind11::module_& m)
{
    py::enum_<tle::ModelType>(m, "ModelType").value("DECODER_ONLY", tle::ModelType::kDECODER_ONLY);

    py::enum_<tle::BatchingType>(m, "BatchingType")
        .value("STATIC", tle::BatchingType::kSTATIC)
        .value("INFLIGHT", tle::BatchingType::kINFLIGHT);

    py::enum_<tle::SchedulerPolicy>(m, "SchedulerPolicy")
        .value("MAX_UTILIZATION", tle::SchedulerPolicy::kMAX_UTILIZATION)
        .value("GUARANTEED_NO_EVICT", tle::SchedulerPolicy::kGUARANTEED_NO_EVICT);

    py::enum_<tle::CommunicationType>(m, "CommunicationType").value("MPI", tle::CommunicationType::kMPI);

    py::enum_<tle::CommunicationMode>(m, "CommunicationMode").value("LEADER", tle::CommunicationMode::kLEADER);

    py::class_<tle::KvCacheStats>(m, "KvCacheStats")
        .def(py::init<>())
        .def_readwrite("max_num_blocks", &tle::KvCacheStats::maxNumBlocks)
        .def_readwrite("free_num_blocks", &tle::KvCacheStats::freeNumBlocks)
        .def_readwrite("used_num_blocks", &tle::KvCacheStats::usedNumBlocks)
        .def_readwrite("tokens_per_block", &tle::KvCacheStats::tokensPerBlock);

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
        .def_readwrite("micro_batch_id", &tle::InflightBatchingStats::microBatchId);

    py::class_<tle::IterationStats>(m, "IterationStats")
        .def(py::init<>())
        .def_readwrite("timestamp", &tle::IterationStats::timestamp)
        .def_readwrite("iter", &tle::IterationStats::iter)
        .def_readwrite("num_active_requests", &tle::IterationStats::numActiveRequests)
        .def_readwrite("max_num_active_requests", &tle::IterationStats::maxNumActiveRequests)
        .def_readwrite("gpu_mem_usage", &tle::IterationStats::gpuMemUsage)
        .def_readwrite("cpu_mem_usage", &tle::IterationStats::cpuMemUsage)
        .def_readwrite("pinned_mem_usage", &tle::IterationStats::pinnedMemUsage)
        .def_readwrite("kv_cache_stats", &tle::IterationStats::kvCacheStats)
        .def_readwrite("static_batching_stats", &tle::IterationStats::staticBatchingStats)
        .def_readwrite("inflight_batching_stats", &tle::IterationStats::inflightBatchingStats);

    py::enum_<tle::RequestStage>(m, "RequestStage")
        .value("QUEUED", tle::RequestStage::kQUEUED)
        .value("CONTEXT_IN_PROGRESS", tle::RequestStage::kCONTEXT_IN_PROGRESS)
        .value("GENERATION_IN_PROGRESS", tle::RequestStage::kGENERATION_IN_PROGRESS)
        .value("GENERATION_COMPLETE", tle::RequestStage::kGENERATION_COMPLETE);

    py::class_<tle::RequestStats>(m, "RequestStats")
        .def(py::init<>())
        .def_readwrite("id", &tle::RequestStats::id)
        .def_readwrite("stage", &tle::RequestStats::stage)
        .def_readwrite("context_prefill_position", &tle::RequestStats::contextPrefillPosition)
        .def_readwrite("num_generated_tokens", &tle::RequestStats::numGeneratedTokens)
        .def_readwrite("scheduled", &tle::RequestStats::scheduled)
        .def_readwrite("paused", &tle::RequestStats::paused);

    py::class_<tle::RequestStatsPerIteration>(m, "RequestStatsPerIteration")
        .def(py::init<>())
        .def_readwrite("iter", &tle::RequestStatsPerIteration::iter)
        .def_readwrite("request_stats", &tle::RequestStatsPerIteration::requestStats);

    py::class_<tle::SamplingConfig>(m, "SamplingConfig")
        .def(py::init<SizeType, std::optional<SizeType> const&, std::optional<FloatType> const&,
                 std::optional<FloatType> const&, std::optional<SizeType> const&, std::optional<FloatType> const&,
                 std::optional<tle::RandomSeedType> const&, std::optional<FloatType> const&,
                 std::optional<SizeType> const&, std::optional<FloatType> const&, std::optional<FloatType> const&,
                 std::optional<FloatType> const&, std::optional<FloatType> const&, std::optional<FloatType> const&,
                 std::optional<SizeType> const&>(),
            py::arg("beam_width") = 1, py::arg("top_k") = py::none(), py::arg("top_p") = py::none(),
            py::arg("top_p_min") = py::none(), py::arg("top_p_reset_ids") = py::none(),
            py::arg("top_p_decay") = py::none(), py::arg("random_seed") = py::none(),
            py::arg("temperature") = py::none(), py::arg("min_length") = py::none(),
            py::arg("beam_search_diversity_rate") = py::none(), py::arg("repetition_penalty") = py::none(),
            py::arg("presence_penalty") = py::none(), py::arg("frequency_penalty") = py::none(),
            py::arg("length_penalty") = py::none(), py::arg("early_stopping") = py::none())
        .def_property_readonly("beam_width", &tle::SamplingConfig::getBeamWidth)
        .def_property_readonly("top_k", &tle::SamplingConfig::getTopK)
        .def_property_readonly("top_p", &tle::SamplingConfig::getTopP)
        .def_property_readonly("top_p_min", &tle::SamplingConfig::getTopPMin)
        .def_property_readonly("top_p_reset_ids", &tle::SamplingConfig::getTopPResetIds)
        .def_property_readonly("top_p_decay", &tle::SamplingConfig::getTopPDecay)
        .def_property_readonly("random_seed", &tle::SamplingConfig::getRandomSeed)
        .def_property_readonly("temperature", &tle::SamplingConfig::getTemperature)
        .def_property_readonly("min_length", &tle::SamplingConfig::getMinLength)
        .def_property_readonly("beam_search_diversity_rate", &tle::SamplingConfig::getBeamSearchDiversityRate)
        .def_property_readonly("repetition_penalty", &tle::SamplingConfig::getRepetitionPenalty)
        .def_property_readonly("presence_penalty", &tle::SamplingConfig::getPresencePenalty)
        .def_property_readonly("frequency_penalty", &tle::SamplingConfig::getFrequencyPenalty)
        .def_property_readonly("length_penalty", &tle::SamplingConfig::getLengthPenalty)
        .def_property_readonly("early_stopping", &tle::SamplingConfig::getEarlyStopping);

    py::class_<tle::OutputConfig>(m, "OutputConfig")
        .def(py::init<bool, bool, bool, bool>(), py::arg("return_log_probs") = false,
            py::arg("return_context_logits") = false, py::arg("return_generation_logits") = false,
            py::arg("exclude_input_from_output") = false)
        .def_readwrite("return_log_probs", &tle::OutputConfig::returnLogProbs)
        .def_readwrite("return_context_logits", &tle::OutputConfig::returnContextLogits)
        .def_readwrite("return_generation_logits", &tle::OutputConfig::returnGenerationLogits)
        .def_readwrite("exclude_input_from_output", &tle::OutputConfig::excludeInputFromOutput);

    py::class_<tle::SpeculativeDecodingConfig>(m, "SpeculativeDecodingConfig")
        .def(py::init<VecTokens, std::optional<Tensor>, std::optional<FloatType> const&>(), py::arg("tokens"),
            py::arg("logits") = py::none(), py::arg("acceptance_threshold") = py::none())
        .def_property_readonly("tokens", &tle::SpeculativeDecodingConfig::getTokens)
        .def_property_readonly("logits", &tle::SpeculativeDecodingConfig::getLogits)
        .def_property_readonly("acceptance_threshold", &tle::SpeculativeDecodingConfig::getAcceptanceThreshold);

    py::class_<tle::PromptTuningConfig>(m, "PromptTuningConfig")
        .def(py::init<Tensor>(), py::arg("embedding_table"))
        .def_property_readonly("embedding_table", &tle::PromptTuningConfig::getEmbeddingTable);

    py::class_<tle::LoraConfig>(m, "LoraConfig")
        .def(py::init<uint64_t, std::optional<Tensor>, std::optional<Tensor>>(), py::arg("task_id"),
            py::arg("weights") = py::none(), py::arg("config") = py::none())
        .def_property_readonly("task_id", &tle::LoraConfig::getTaskId)
        .def_property_readonly("weights", &tle::LoraConfig::getWeights)
        .def_property_readonly("config", &tle::LoraConfig::getConfig);

    py::class_<tle::Request>(m, "Request")
        .def(py::init<VecTokens, SizeType, bool, tle::SamplingConfig const&, tle::OutputConfig const&,
                 std::optional<SizeType> const&, std::optional<SizeType> const&, std::optional<std::list<VecTokens>>,
                 std::optional<std::list<VecTokens>>, std::optional<Tensor>,
                 std::optional<tle::SpeculativeDecodingConfig>, std::optional<tle::PromptTuningConfig>,
                 std::optional<tle::LoraConfig>>(),
            py::arg("input_token_ids"), py::arg("max_new_tokens"), py::arg("streaming") = false,
            py::arg_v("sampling_config", tle::SamplingConfig(), "SamplingConfig()"),
            py::arg_v("output_config", tle::OutputConfig(), "OutputConfig()"), py::arg("end_id") = py::none(),
            py::arg("pad_id") = py::none(), py::arg("bad_words") = py::none(), py::arg("stop_words") = py::none(),
            py::arg("embedding_bias") = py::none(), py::arg("speculative_decoding_config") = py::none(),
            py::arg("prompt_tuning_config") = py::none(), py::arg("lora_config") = py::none())
        .def_property_readonly("input_token_ids", &tle::Request::getInputTokenIds)
        .def_property_readonly("max_new_tokens", &tle::Request::getMaxNewTokens)
        .def_property("streaming", &tle::Request::getStreaming, &tle::Request::setStreaming)
        .def_property("sampling_config", &tle::Request::getSamplingConfig, &tle::Request::setSamplingConfig)
        .def_property("output_config", &tle::Request::getOutputConfig, &tle::Request::setOutputConfig)
        .def_property("end_id", &tle::Request::getEndId, &tle::Request::setEndId)
        .def_property("pad_id", &tle::Request::getPadId, &tle::Request::setPadId)
        .def_property("bad_words", &tle::Request::getBadWords, &tle::Request::setBadWords)
        .def_property("stop_words", &tle::Request::getStopWords, &tle::Request::setStopWords)
        .def_property("embedding_bias", &tle::Request::getEmbeddingBias, &tle::Request::setEmbeddingBias)
        .def_property("speculative_decoding_config", &tle::Request::getSpeculativeDecodingConfig,
            &tle::Request::setSpeculativeDecodingConfig)
        .def_property(
            "prompt_tuning_config", &tle::Request::getPromptTuningConfig, &tle::Request::setPromptTuningConfig)
        .def_property("lora_config", &tle::Request::getLoraConfig, &tle::Request::setLoraConfig)
        .def_property("logits_post_processor_name", &tle::Request::getLogitsPostProcessorName,
            &tle::Request::setLogitsPostProcessorName);

    py::class_<tle::Result>(m, "Result")
        .def(py::init<>())
        .def_readwrite("is_final", &tle::Result::isFinal)
        .def_readwrite("output_token_ids", &tle::Result::outputTokenIds)
        .def_readwrite("cum_log_probs", &tle::Result::cumLogProbs)
        .def_readwrite("log_probs", &tle::Result::logProbs)
        .def_readwrite("context_logits", &tle::Result::contextLogits)
        .def_readwrite("generation_logits", &tle::Result::generationLogits);

    py::class_<tle::Response>(m, "Response")
        .def(py::init<IdType, std::string>(), py::arg("request_id"), py::arg("error_msg"))
        .def(py::init<IdType, tle::Result>(), py::arg("request_id"), py::arg("result"))
        .def_property_readonly("request_id", &tle::Response::getRequestId)
        .def("has_error", &tle::Response::hasError)
        .def_property_readonly("error_msg", &tle::Response::getErrorMsg)
        .def_property_readonly("result", &tle::Response::getResult);

    py::class_<tle::SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<tle::SchedulerPolicy>(),
            py::arg_v("policy", tle::SchedulerPolicy::kGUARANTEED_NO_EVICT, "SchedulerPolicy.GUARANTEED_NO_EVICT"))
        .def_property_readonly("policy", &tle::SchedulerConfig::getPolicy);

    py::class_<tle::KvCacheConfig>(m, "KvCacheConfig")
        .def(py::init<bool, std::optional<SizeType> const&, std::optional<SizeType> const&,
                 std::optional<SizeType> const&, std::optional<float> const&>(),
            py::arg("enable_block_reuse") = false, py::arg("max_tokens") = py::none(),
            py::arg("max_attention_window") = py::none(), py::arg("sink_token_length") = py::none(),
            py::arg("free_gpu_memory_fraction") = py::none())
        .def_property_readonly("enable_block_reuse", &tle::KvCacheConfig::getEnableBlockReuse)
        .def_property_readonly("max_tokens", &tle::KvCacheConfig::getMaxTokens)
        .def_property_readonly("max_attention_window", &tle::KvCacheConfig::getMaxAttentionWindow)
        .def_property_readonly("sink_token_length", &tle::KvCacheConfig::getSinkTokenLength)
        .def_property_readonly("free_gpu_memory_fraction", &tle::KvCacheConfig::getFreeGpuMemoryFraction);

    py::class_<tle::ParallelConfig>(m, "ParallelConfig")
        .def(py::init<tle::CommunicationType, tle::CommunicationMode, std::optional<std::vector<SizeType>> const&,
                 std::optional<std::vector<SizeType>> const&>(),
            py::arg_v("communication_type", tle::CommunicationType::kMPI, "CommunicationType.MPI"),
            py::arg_v("communication_mode", tle::CommunicationMode::kLEADER, "CommunicationMode.LEADER"),
            py::arg("device_ids") = py::none(), py::arg("participant_ids") = py::none())
        .def_property("communication_type", &tle::ParallelConfig::getCommunicationType,
            &tle::ParallelConfig::setCommunicationType)
        .def_property("communication_mode", &tle::ParallelConfig::getCommunicationMode,
            &tle::ParallelConfig::setCommunicationMode)
        .def_property("device_ids", &tle::ParallelConfig::getDeviceIds, &tle::ParallelConfig::setDeviceIds)
        .def_property(
            "participant_ids", &tle::ParallelConfig::getParticipantIds, &tle::ParallelConfig::setParticipantIds);

    py::class_<tle::PeftCacheConfig>(m, "PeftCacheConfig")
        .def(py::init<SizeType, SizeType, SizeType, SizeType, SizeType, SizeType, SizeType, SizeType, SizeType,
                 std::optional<float> const&, std::optional<size_t> const&>(),
            py::arg("num_host_module_layer") = 0, py::arg("num_device_module_layer") = 0,
            py::arg("optimal_adapter_size") = 8, py::arg("max_adapter_size") = 64, py::arg("num_put_workers") = 1,
            py::arg("num_ensure_workers") = 1, py::arg("num_copy_streams") = 1,
            py::arg("max_pages_per_block_host") = 24, py::arg("max_pages_per_block_device") = 8,
            py::arg("device_cache_percent") = py::none(), py::arg("host_cache_size") = py::none())
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
        .def_property_readonly("host_cache_size", &tle::PeftCacheConfig::getHostCacheSize);

    py::class_<tle::ExecutorConfig>(m, "ExecutorConfig")
        .def(py::init<SizeType, tle::SchedulerConfig const&, tle::KvCacheConfig const&, bool, bool, SizeType, SizeType,
                 tle::BatchingType, std::optional<tle::ParallelConfig>, tle::PeftCacheConfig const&,
                 std::optional<tle::LogitsPostProcessorMap>, std::optional<tle::MedusaChoices>>(),
            py::arg("max_beam_width") = 1, py::arg_v("scheduler_config", tle::SchedulerConfig(), "SchedulerConfig()"),
            py::arg_v("kv_cache_config", tle::KvCacheConfig(), "KvCacheConfig()"),
            py::arg("enable_chunked_context") = false, py::arg("normalize_log_probs") = true,
            py::arg("iter_stats_max_iterations") = tle::kDefaultIterStatsMaxIterations,
            py::arg("request_stats_max_iterations") = tle::kDefaultRequestStatsMaxIterations,
            py::arg_v("batching_type", tle::BatchingType::kINFLIGHT, "BatchingType.INFLIGHT"),
            py::arg("parallel_config") = py::none(),
            py::arg_v("peft_cache_config", tle::PeftCacheConfig(), "PeftCacheConfig()"),
            py::arg("logits_post_processor_map") = py::none(), py::arg("medusa_choices") = py::none())
        .def_property("max_beam_width", &tle::ExecutorConfig::getMaxBeamWidth, &tle::ExecutorConfig::setMaxBeamWidth)
        .def_property(
            "scheduler_config", &tle::ExecutorConfig::getSchedulerConfig, &tle::ExecutorConfig::setSchedulerConfig)
        .def_property("kv_cache_config", &tle::ExecutorConfig::getKvCacheConfig, &tle::ExecutorConfig::setKvCacheConfig)
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
        .def_property("logits_post_processor_map", &tle::ExecutorConfig::getLogitsPostProcessorMap,
            &tle::ExecutorConfig::setLogitsPostProcessorMap)
        .def_property("medusa_choices", &tle::ExecutorConfig::getMedusaChoices, &tle::ExecutorConfig::setMedusaChoices);

    tensorrt_llm::pybind::executor::Executor::initBindings(m);
}

} // namespace tensorrt_llm::pybind::executor
