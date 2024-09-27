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

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

#include "tensorrt_llm/pybind/batch_manager/gptManager.h"
#include "tensorrt_llm/pybind/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/pybind/batch_manager/llmRequest.h"
#include "tensorrt_llm/pybind/batch_manager/namedTensor.h"
#include "tensorrt_llm/pybind/executor/bindings.h"
#include "tensorrt_llm/pybind/utils/pathCaster.h"

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tpb = tensorrt_llm::pybind::batch_manager;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;
using SizeType32 = tr::SizeType32;
using TokenIdType = tr::TokenIdType;
template <typename T>
using OptVec = std::optional<std::vector<T>>;

#if not defined(TRTLLM_PYBIND_MODULE)
#error "TRTLLM_PYBIND_MODULE must be defined"
#endif

PYBIND11_MODULE(TRTLLM_PYBIND_MODULE, m)
{
    m.doc() = "TensorRT-LLM Python bindings for C++ runtime";

    // Create submodule for executor bindings.
    py::module_ executor_submodule = m.def_submodule("executor", "Executor bindings");
    tensorrt_llm::pybind::executor::InitBindings(executor_submodule);

    auto buildInfo = m.def_submodule("BuildInfo");
    buildInfo.attr("ENABLE_MULTI_DEVICE") = py::int_(ENABLE_MULTI_DEVICE);

    auto kvCacheConfigGetState = [](tbk::KvCacheConfig const& config)
    {
        return py::make_tuple(config.maxTokens, config.maxAttentionWindowVec, config.sinkTokenLength,
            config.freeGpuMemoryFraction, config.enableBlockReuse, config.useUvm);
    };
    auto kvCacheConfigSetState = [](py::tuple t)
    {
        return tbk::KvCacheConfig(t[0].cast<std::optional<SizeType32>>(),
            t[1].cast<std::optional<std::vector<SizeType32>>>(), t[2].cast<std::optional<SizeType32>>(),
            t[3].cast<std::optional<float>>(), t[4].cast<bool>(), t[5].cast<bool>());
    };
    py::class_<tbk::KvCacheConfig>(m, "KvCacheConfig")
        .def(py::init<std::optional<SizeType32>, std::optional<std::vector<SizeType32>>, std::optional<SizeType32>,
                 std::optional<float>, bool>(),
            py::arg("max_tokens") = py::none(), py::arg("max_attention_window") = py::none(),
            py::arg("sink_token_length") = py::none(), py::arg("free_gpu_memory_fraction") = py::none(),
            py::arg("enable_block_reuse") = false)
        .def_readwrite("max_tokens", &tbk::KvCacheConfig::maxTokens)
        .def_readwrite("max_attention_window", &tbk::KvCacheConfig::maxAttentionWindowVec)
        .def_readwrite("sink_token_length", &tbk::KvCacheConfig::sinkTokenLength)
        .def_readwrite("free_gpu_memory_fraction", &tbk::KvCacheConfig::freeGpuMemoryFraction)
        .def_readwrite("enable_block_reuse", &tbk::KvCacheConfig::enableBlockReuse)
        .def(py::pickle(kvCacheConfigGetState, kvCacheConfigSetState))
        .def("__eq__", &tbk::KvCacheConfig::operator==);

    py::class_<tb::PeftCacheManagerConfig>(m, "PeftCacheManagerConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, std::optional<float>, std::optional<size_t>>(),
            py::arg("num_host_module_layer") = 0, py::arg("num_device_module_layer") = 0,
            py::arg("optimal_adapter_size") = 8, py::arg("max_adapter_size") = 64, py::arg("num_put_workers") = 1,
            py::arg("num_ensure_workers") = 1, py::arg("num_copy_streams") = 1,
            py::arg("max_pages_per_block_host") = 24, py::arg("max_pages_per_block_device") = 8,
            py::arg("device_cache_percent") = std::nullopt, py::arg("host_cache_size") = std::nullopt)
        .def_readwrite("num_host_module_layer", &tb::PeftCacheManagerConfig::numHostModuleLayer)
        .def_readwrite("num_device_module_layer", &tb::PeftCacheManagerConfig::numDeviceModuleLayer)
        .def_readwrite("optimal_adapter_size", &tb::PeftCacheManagerConfig::optimalAdapterSize)
        .def_readwrite("max_adapter_size", &tb::PeftCacheManagerConfig::maxAdapterSize)
        .def_readwrite("num_put_workers", &tb::PeftCacheManagerConfig::numPutWorkers)
        .def_readwrite("num_ensure_workers", &tb::PeftCacheManagerConfig::numEnsureWorkers)
        .def_readwrite("num_copy_streams", &tb::PeftCacheManagerConfig::numCopyStreams)
        .def_readwrite("max_pages_per_block_host", &tb::PeftCacheManagerConfig::maxPagesPerBlockHost)
        .def_readwrite("max_pages_per_block_device", &tb::PeftCacheManagerConfig::maxPagesPerBlockDevice)
        .def_readwrite("device_cache_percent", &tb::PeftCacheManagerConfig::deviceCachePercent)
        .def_readwrite("host_cache_size", &tb::PeftCacheManagerConfig::hostCacheSize);

    py::enum_<nvinfer1::DataType>(m, "DataType")
        .value("FLOAT", nvinfer1::DataType::kFLOAT)
        .value("HALF", nvinfer1::DataType::kHALF)
        .value("INT8", nvinfer1::DataType::kINT8)
        .value("INT32", nvinfer1::DataType::kINT32)
        .value("BOOL", nvinfer1::DataType::kBOOL)
        .value("UINT8", nvinfer1::DataType::kUINT8)
        .value("FP8", nvinfer1::DataType::kFP8)
        .value("BF16", nvinfer1::DataType::kBF16)
        .value("INT64", nvinfer1::DataType::kINT64)
        .export_values();

    py::enum_<tr::ModelConfig::ModelVariant>(m, "GptModelVariant")
        .value("GPT", tr::ModelConfig::ModelVariant::kGpt)
        .value("GLM", tr::ModelConfig::ModelVariant::kGlm)
        .value("CHATGLM", tr::ModelConfig::ModelVariant::kChatGlm)
        .value("MAMBA", tr::ModelConfig::ModelVariant::kMamba)
        .value("RECURRENTGEMMA", tr::ModelConfig::ModelVariant::kRecurrentGemma);

    py::enum_<tr::ModelConfig::KVCacheType>(m, "KVCacheType")
        .value("CONTINUOUS", tr::ModelConfig::KVCacheType::kCONTINUOUS)
        .value("PAGED", tr::ModelConfig::KVCacheType::kPAGED)
        .value("DISABLED", tr::ModelConfig::KVCacheType::kDISABLED)
        .def(py::init(&tr::ModelConfig::KVCacheTypeFromString));

    py::class_<tc::QuantMode>(m, "QuantMode")
        .def_static("none", &tc::QuantMode::none)
        .def_static("int4_weights", &tc::QuantMode::int4Weights)
        .def_static("int8_weights", &tc::QuantMode::int8Weights)
        .def_static("activations", &tc::QuantMode::activations)
        .def_static("per_channel_scaling", &tc::QuantMode::perChannelScaling)
        .def_static("per_token_scaling", &tc::QuantMode::perTokenScaling)
        .def_static("per_group_scaling", &tc::QuantMode::perGroupScaling)
        .def_static("int8_kv_cache", &tc::QuantMode::int8KvCache)
        .def_static("fp8_kv_cache", &tc::QuantMode::fp8KvCache)
        .def_static("fp8_qdq", &tc::QuantMode::fp8Qdq)
        .def_property_readonly("value", &tc::QuantMode::value)
        .def("is_set", &tc::QuantMode::isSet, py::arg("mode"))
        .def_property_readonly("has_int4_weights", &tc::QuantMode::hasInt4Weights)
        .def_property_readonly("has_int8_weights", &tc::QuantMode::hasInt8Weights)
        .def_property_readonly("has_activations", &tc::QuantMode::hasActivations)
        .def_property_readonly("has_per_channel_scaling", &tc::QuantMode::hasPerChannelScaling)
        .def_property_readonly("has_per_token_scaling", &tc::QuantMode::hasPerTokenScaling)
        .def_property_readonly("has_per_group_scaling", &tc::QuantMode::hasPerGroupScaling)
        .def_property_readonly("has_static_activation_scaling", &tc::QuantMode::hasStaticActivationScaling)
        .def_property_readonly("has_int8_kv_cache", &tc::QuantMode::hasInt8KvCache)
        .def_property_readonly("has_fp8_kv_cache", &tc::QuantMode::hasFp8KvCache)
        .def_property_readonly("has_fp8_qdq", &tc::QuantMode::hasFp8Qdq)
        .def_property_readonly("has_kv_cache_quant", &tc::QuantMode::hasKvCacheQuant)
        .def_static("from_description", &tc::QuantMode::fromDescription, py::arg("quantize_weights") = false,
            py::arg("quantize_activations") = false, py::arg("per_token") = false, py::arg("per_channel") = false,
            py::arg("per_group") = false, py::arg("use_int4_weights") = false, py::arg("use_int8_kv_cache") = false,
            py::arg("use_fp8_kv_kache") = false, py::arg("use_fp8_qdq") = false, py::arg("use_fp8_rowwise") = false)
        .def_static("use_smooth_quant", &tc::QuantMode::useSmoothQuant, py::arg("per_token") = false,
            py::arg("per_channel") = false)
        .def_static("use_weight_only", &tc::QuantMode::useWeightOnly, py::arg("use_int4_weights") = false,
            py::arg("per_group") = false)
        .def_static("from_quant_algo", &tc::QuantMode::fromQuantAlgo, py::arg("quant_algo") = py::none(),
            py::arg("kv_cache_quant_algo") = py::none())
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<tr::ModelConfig>(m, "ModelConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, nvinfer1::DataType>(),
            py::arg("vocab_size"), py::arg("num_attention_layers"), py::arg("num_rnn_layers"), py::arg("num_heads"),
            py::arg("hidden_size"), py::arg("data_type"))
        .def_property_readonly("vocab_size", &tr::ModelConfig::getVocabSize)
        .def("vocab_size_padded", &tr::ModelConfig::getVocabSizePadded, py::arg("world_size"))
        .def("num_attention_layers", &tr::ModelConfig::getNbAttentionLayers, py::arg("pipeline_parallelism") = 1)
        .def("num_rnn_layers", &tr::ModelConfig::getNbRnnLayers, py::arg("pipeline_parallelism") = 1)
        .def_property_readonly("num_heads", &tr::ModelConfig::getNbHeads)
        .def_property_readonly("hidden_size", &tr::ModelConfig::getHiddenSize)
        .def_property_readonly("size_per_head", &tr::ModelConfig::getSizePerHead)
        .def_property_readonly("data_type", &tr::ModelConfig::getDataType)
        .def_property("num_kv_heads", &tr::ModelConfig::getNbKvHeads, &tr::ModelConfig::setNbKvHeads)
        .def_property("head_size", &tr::ModelConfig::getSizePerHead, &tr::ModelConfig::setSizePerHead)
        .def_property("use_gpt_attention_plugin",
            py::overload_cast<>(&tr::ModelConfig::useGptAttentionPlugin, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::useGptAttentionPlugin))
        .def_property("use_packed_input", py::overload_cast<>(&tr::ModelConfig::usePackedInput, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::usePackedInput))
        .def_property("kv_cache_type", py::overload_cast<>(&tr::ModelConfig::getKVCacheType, py::const_),
            py::overload_cast<tr::ModelConfig::KVCacheType>(&tr::ModelConfig::setKVCacheType))
        .def_property("tokens_per_block", &tr::ModelConfig::getTokensPerBlock, &tr::ModelConfig::setTokensPerBlock)
        .def_property("quant_mode", &tr::ModelConfig::getQuantMode, &tr::ModelConfig::setQuantMode)
        .def_property_readonly("supports_inflight_batching", &tr::ModelConfig::supportsInflightBatching)
        .def_property("max_batch_size", &tr::ModelConfig::getMaxBatchSize, &tr::ModelConfig::setMaxBatchSize)
        .def_property("max_beam_width", &tr::ModelConfig::getMaxBeamWidth, &tr::ModelConfig::setMaxBeamWidth)
        .def_property("max_input_len", &tr::ModelConfig::getMaxInputLen, &tr::ModelConfig::setMaxInputLen)
        .def_property("max_seq_len", &tr::ModelConfig::getMaxSequenceLen, &tr::ModelConfig::getMaxSequenceLen)
        .def_property("max_num_tokens", &tr::ModelConfig::getMaxNumTokens, &tr::ModelConfig::setMaxNumTokens)
        .def_property("max_prompt_embedding_table_size", &tr::ModelConfig::getMaxPromptEmbeddingTableSize,
            &tr::ModelConfig::setMaxPromptEmbeddingTableSize)
        .def_property_readonly("use_prompt_tuning", &tr::ModelConfig::usePromptTuning)
        .def_property("use_lora_plugin", py::overload_cast<>(&tr::ModelConfig::useLoraPlugin, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::useLoraPlugin))
        .def_property("compute_context_logits", py::overload_cast<>(&tr::ModelConfig::computeContextLogits, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::computeContextLogits))
        .def_property("compute_generation_logits",
            py::overload_cast<>(&tr::ModelConfig::computeGenerationLogits, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::computeGenerationLogits))
        .def_property("model_variant", &tr::ModelConfig::getModelVariant, &tr::ModelConfig::setModelVariant);

    py::class_<tr::WorldConfig>(m, "WorldConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, std::optional<std::vector<SizeType32>> const&>(),
            py::arg("tensor_parallelism") = 1, py::arg("pipeline_parallelism") = 1, py::arg("rank") = 0,
            py::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode, py::arg("device_ids") = py::none())
        .def_property_readonly("size", &tr::WorldConfig::getSize)
        .def_property_readonly("tensor_parallelism", &tr::WorldConfig::getTensorParallelism)
        .def_property_readonly("pipeline_parallelism", &tr::WorldConfig::getPipelineParallelism)
        .def_property_readonly("is_tensor_parallel", &tr::WorldConfig::isTensorParallel)
        .def_property_readonly("is_pipeline_parallel", &tr::WorldConfig::isPipelineParallel)
        .def_property_readonly("rank", &tr::WorldConfig::getRank)
        .def_property_readonly("local_rank", &tr::WorldConfig::getLocalRank)
        .def_property_readonly("node_rank", &tr::WorldConfig::getNodeRank)
        .def_property_readonly("gpus_per_node", &tr::WorldConfig::getGpusPerNode)
        .def_property_readonly("gpus_per_group", &tr::WorldConfig::getGpusPerGroup)
        .def_property_readonly("device", &tr::WorldConfig::getDevice)
        .def_property_readonly("pipeline_parallel_rank", &tr::WorldConfig::getPipelineParallelRank)
        .def_property_readonly("tensor_parallel_rank", &tr::WorldConfig::getTensorParallelRank)
        .def_static("mpi",
            py::overload_cast<SizeType32, std::optional<SizeType32>, std::optional<SizeType32>,
                std::optional<std::vector<SizeType32>> const&>(&tr::WorldConfig::mpi),
            py::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode, py::arg("tensor_parallelism") = py::none(),
            py::arg("pipeline_parallelism") = py::none(), py::arg("device_ids") = py::none());

    auto SamplingConfigGetState = [](tr::SamplingConfig const& config) -> py::tuple
    {
        return py::make_tuple(config.beamWidth, config.temperature, config.minLength, config.repetitionPenalty,
            config.presencePenalty, config.frequencyPenalty, config.topK, config.topP, config.randomSeed,
            config.topPDecay, config.topPMin, config.topPResetIds, config.beamSearchDiversityRate, config.lengthPenalty,
            config.earlyStopping, config.noRepeatNgramSize);
    };
    auto SamplingConfigSetState = [](py::tuple t) -> tr::SamplingConfig
    {
        assert(t.size() == 16);

        tr::SamplingConfig config;
        config.beamWidth = t[0].cast<SizeType32>();
        config.temperature = t[1].cast<OptVec<float>>();
        config.minLength = t[2].cast<OptVec<SizeType32>>();
        config.repetitionPenalty = t[3].cast<OptVec<float>>();
        config.presencePenalty = t[4].cast<OptVec<float>>();
        config.frequencyPenalty = t[5].cast<OptVec<float>>();
        config.topK = t[6].cast<OptVec<SizeType32>>();
        config.topP = t[7].cast<OptVec<float>>();
        config.randomSeed = t[8].cast<OptVec<uint64_t>>();
        config.topPDecay = t[9].cast<OptVec<float>>();
        config.topPMin = t[10].cast<OptVec<float>>();
        config.topPResetIds = t[11].cast<OptVec<TokenIdType>>();
        config.beamSearchDiversityRate = t[12].cast<OptVec<float>>();
        config.lengthPenalty = t[13].cast<OptVec<float>>();
        config.earlyStopping = t[14].cast<OptVec<SizeType32>>();
        config.noRepeatNgramSize = t[15].cast<OptVec<SizeType32>>();

        return std::move(config);
    };

    py::class_<tr::SamplingConfig>(m, "SamplingConfig")
        .def(py::init<SizeType32>(), py::arg("beam_width") = 1)
        .def_readwrite("beam_width", &tr::SamplingConfig::beamWidth)
        .def_readwrite("temperature", &tr::SamplingConfig::temperature)
        .def_readwrite("min_length", &tr::SamplingConfig::minLength)
        .def_readwrite("repetition_penalty", &tr::SamplingConfig::repetitionPenalty)
        .def_readwrite("presence_penalty", &tr::SamplingConfig::presencePenalty)
        .def_readwrite("frequency_penalty", &tr::SamplingConfig::frequencyPenalty)
        .def_readwrite("top_k", &tr::SamplingConfig::topK)
        .def_readwrite("top_p", &tr::SamplingConfig::topP)
        .def_readwrite("random_seed", &tr::SamplingConfig::randomSeed)
        .def_readwrite("top_p_decay", &tr::SamplingConfig::topPDecay)
        .def_readwrite("top_p_min", &tr::SamplingConfig::topPMin)
        .def_readwrite("top_p_reset_ids", &tr::SamplingConfig::topPResetIds)
        .def_readwrite("beam_search_diversity_rate", &tr::SamplingConfig::beamSearchDiversityRate)
        .def_readwrite("length_penalty", &tr::SamplingConfig::lengthPenalty)
        .def_readwrite("early_stopping", &tr::SamplingConfig::earlyStopping)
        .def_readwrite("no_repeat_ngram_size", &tr::SamplingConfig::noRepeatNgramSize)
        .def(py::pickle(SamplingConfigGetState, SamplingConfigSetState))
        .def("__eq__", &tr::SamplingConfig::operator==);

    py::class_<tr::GptJsonConfig>(m, "GptJsonConfig")
        .def(py::init<std::string, std::string, std::string, SizeType32, SizeType32, SizeType32, tr::ModelConfig>(),
            py::arg("name"), py::arg("version"), py::arg("precision"), py::arg("tensor_parallelism"),
            py::arg("pipeline_parallelism"), py::arg("gpus_per_node"), py::arg("model_config"))
        .def_static("parse", py::overload_cast<std::string const&>(&tr::GptJsonConfig::parse), py::arg("json"))
        .def_static(
            "parse_file", py::overload_cast<std::filesystem::path const&>(&tr::GptJsonConfig::parse), py::arg("path"))
        .def_property_readonly("model_config", &tr::GptJsonConfig::getModelConfig)
        .def_property_readonly("name", &tr::GptJsonConfig::getName)
        .def_property_readonly("version", &tr::GptJsonConfig::getVersion)
        .def_property_readonly("precision", &tr::GptJsonConfig::getPrecision)
        .def_property_readonly("tensor_parallelism", &tr::GptJsonConfig::getTensorParallelism)
        .def_property_readonly("pipeline_parallelism", &tr::GptJsonConfig::getPipelineParallelism)
        .def_property_readonly("gpus_per_node", &tr::GptJsonConfig::getGpusPerNode)
        .def_property_readonly("world_size", &tr::GptJsonConfig::getWorldSize)
        .def("engine_filename",
            py::overload_cast<tr::WorldConfig const&, std::string const&>(
                &tr::GptJsonConfig::engineFilename, py::const_),
            py::arg("world_config"), py::arg("model"))
        .def("engine_filename",
            py::overload_cast<tr::WorldConfig const&>(&tr::GptJsonConfig::engineFilename, py::const_),
            py::arg("world_config"));

    py::enum_<tb::LlmRequestState>(m, "LlmRequestState")
        .value("UNKNOWN", tb::LlmRequestState::kUNKNOWN)
        .value("ENCODER_INIT", tb::LlmRequestState::kENCODER_INIT)
        .value("CONTEXT_INIT", tb::LlmRequestState::kCONTEXT_INIT)
        .value("GENERATION_IN_PROGRESS", tb::LlmRequestState::kGENERATION_IN_PROGRESS)
        .value("GENERATION_TO_COMPLETE", tb::LlmRequestState::kGENERATION_TO_COMPLETE)
        .value("GENERATION_COMPLETE", tb::LlmRequestState::kGENERATION_COMPLETE);

    tpb::NamedTensor::initBindings(m);
    tpb::LlmRequest::initBindings(m);

    auto tensorNames = m.def_submodule("tensor_names");
    // Input tensor names
    tensorNames.attr("INPUT_IDS") = py::str(tb::inference_request::kInputIdsTensorName);
    tensorNames.attr("POSITION_IDS") = py::str(tb::inference_request::kPositionIdsTensorName);
    tensorNames.attr("DRAFT_INPUT_IDS") = py::str(tb::inference_request::kDraftInputIdsTensorName);
    tensorNames.attr("DRAFT_LOGITS") = py::str(tb::inference_request::kDraftLogitsTensorName);
    tensorNames.attr("MAX_NEW_TOKENS") = py::str(tb::inference_request::kMaxNewTokensTensorName);
    tensorNames.attr("NUM_RETURN_SEQUENCES") = py::str(tb::inference_request::kNumReturnSequencesTensorName);
    tensorNames.attr("BEAM_WIDTH") = py::str(tb::inference_request::kBeamWidthTensorName);
    tensorNames.attr("END_ID") = py::str(tb::inference_request::kEndIdTensorName);
    tensorNames.attr("PAD_ID") = py::str(tb::inference_request::kPadIdTensorName);
    tensorNames.attr("BAD_WORDS_LIST") = py::str(tb::inference_request::kBadWordsListTensorName);
    tensorNames.attr("STOP_WORDS_LIST") = py::str(tb::inference_request::kStopWordsListTensorName);
    tensorNames.attr("EMBEDDING_BIAS") = py::str(tb::inference_request::kEmbeddingBiasTensorName);
    tensorNames.attr("TEMPERATURE") = py::str(tb::inference_request::kTemperatureTensorName);
    tensorNames.attr("RUNTIME_TOP_K") = py::str(tb::inference_request::kRuntimeTopKTensorName);
    tensorNames.attr("RUNTIME_TOP_P") = py::str(tb::inference_request::kRuntimeTopPTensorName);
    tensorNames.attr("LENGTH_PENALTY") = py::str(tb::inference_request::kLengthPenaltyTensorName);
    tensorNames.attr("EARLY_STOPPING") = py::str(tb::inference_request::kEarlyStoppingTensorName);
    tensorNames.attr("REPETITION_PENALTY") = py::str(tb::inference_request::kRepetitionPenaltyTensorName);
    tensorNames.attr("MIN_LENGTH") = py::str(tb::inference_request::kMinLengthTensorName);
    tensorNames.attr("PRESENCE_PENALTY") = py::str(tb::inference_request::kPresencePenaltyTensorName);
    tensorNames.attr("FREQUENCY_PENALTY") = py::str(tb::inference_request::kFrequencyPenaltyTensorName);
    tensorNames.attr("RANDOM_SEED") = py::str(tb::inference_request::kRandomSeedTensorName);
    tensorNames.attr("RETURN_LOG_PROBS") = py::str(tb::inference_request::kReturnLogProbsTensorName);
    tensorNames.attr("RETURN_CONTEXT_LOGITS") = py::str(tb::inference_request::kReturnContextLogitsTensorName);
    tensorNames.attr("RETURN_GENERATION_LOGITS") = py::str(tb::inference_request::kReturnGenerationLogitsTensorName);
    tensorNames.attr("PROMPT_EMBEDDING_TABLE") = py::str(tb::inference_request::kPromptEmbeddingTableName);
    tensorNames.attr("PROMPT_VOCAB_SIZE") = py::str(tb::inference_request::kPromptVocabSizeName);
    tensorNames.attr("NO_REPEAT_NGRAM_SIZE") = py::str(tb::inference_request::kNoRepeatNgramSizeTensorName);

    // Output tensor names
    tensorNames.attr("OUTPUT_IDS") = py::str(tb::inference_request::kOutputIdsTensorName);
    tensorNames.attr("SEQUENCE_LENGTH") = py::str(tb::inference_request::kSequenceLengthTensorName);
    tensorNames.attr("OUTPUT_LOG_PROBS") = py::str(tb::inference_request::kLogProbsTensorName);
    tensorNames.attr("CUM_LOG_PROBS") = py::str(tb::inference_request::kCumLogProbsTensorName);

    tpb::InferenceRequest::initBindings(m);

    py::enum_<tb::TrtGptModelType>(m, "TrtGptModelType")
        .value("V1", tb::TrtGptModelType::V1)
        .value("InflightBatching", tb::TrtGptModelType::InflightBatching)
        .value("InflightFusedBatching", tb::TrtGptModelType::InflightFusedBatching);

    auto gptModelParamsGetState = [&kvCacheConfigGetState](tb::TrtGptModelOptionalParams const& params)
    {
        auto kvCacheState = kvCacheConfigGetState(params.kvCacheConfig);
        return py::make_tuple(kvCacheState, params.enableTrtOverlap, params.deviceIds, params.normalizeLogProbs,
            params.enableChunkedContext, params.decodingConfig.getDecodingMode());
    };
    auto gptModelParamsSetState = [&kvCacheConfigSetState](py::tuple t)
    {
        auto kvCacheConfig = kvCacheConfigSetState(t[0]);
        return tb::TrtGptModelOptionalParams(kvCacheConfig, t[1].cast<bool>(),
            t[2].cast<std::optional<std::vector<SizeType32>>>(), t[3].cast<bool>(), t[4].cast<bool>(),
            tb::PeftCacheManagerConfig{},
            tensorrt_llm::executor::DecodingConfig(t[5].cast<std::optional<tensorrt_llm::executor::DecodingMode>>()));
    };

    py::class_<tb::TrtGptModelOptionalParams>(m, "TrtGptModelOptionalParams")
        .def(py::init<tbk::KvCacheConfig, bool, std::optional<std::vector<SizeType32>> const&, bool, bool,
                 tb::PeftCacheManagerConfig const&>(),
            py::arg_v("kv_cache_config", tbk::KvCacheConfig{}, "KvCacheConfig()"),
            py::arg("enable_trt_overlap") = false, py::arg("device_ids") = std::nullopt,
            py::arg("normalize_log_probs") = true, py::arg("enable_chunked_context") = false,
            py::arg_v("peft_cache_manager_config", tb::PeftCacheManagerConfig{}, "PeftCacheManagerConfig()"))
        .def_readwrite("kv_cache_config", &tb::TrtGptModelOptionalParams::kvCacheConfig)
        .def_readwrite("enable_trt_overlap", &tb::TrtGptModelOptionalParams::enableTrtOverlap)
        .def_readwrite("device_ids", &tb::TrtGptModelOptionalParams::deviceIds)
        .def_readwrite("enable_chunked_context", &tb::TrtGptModelOptionalParams::enableChunkedContext)
        .def_readwrite("normalize_log_probs", &tb::TrtGptModelOptionalParams::normalizeLogProbs)
        .def_readwrite("decoding_config", &tb::TrtGptModelOptionalParams::decodingConfig)
        .def_readwrite("gpu_weights_percent", &tb::TrtGptModelOptionalParams::gpuWeightsPercent)
        .def_readwrite("max_beam_width", &tb::TrtGptModelOptionalParams::maxBeamWidth)
        .def_readwrite("scheduler_config", &tb::TrtGptModelOptionalParams::schedulerConfig)
        .def(py::pickle(gptModelParamsGetState, gptModelParamsSetState))
        .def("__eq__", &tb::TrtGptModelOptionalParams::operator==);

    tpb::GptManager::initBindings(m);

    py::class_<tr::MemoryCounters>(m, "MemoryCounters")
        .def_static("instance", &tr::MemoryCounters::getInstance, py::return_value_policy::reference)
        .def_property_readonly("gpu", &tr::MemoryCounters::getGpu)
        .def_property_readonly("cpu", &tr::MemoryCounters::getCpu)
        .def_property_readonly("pinned", &tr::MemoryCounters::getPinned)
        .def_property_readonly("uvm", &tr::MemoryCounters::getUVM);

    py::class_<tensorrt_llm::mpi::MpiComm>(m, "MpiComm")
        .def_static("rank",
            []()
            {
                auto& session = tensorrt_llm::mpi::MpiComm::session();
                return session.tensorrt_llm::mpi::MpiComm::getRank();
            })
        .def_static("size",
            []()
            {
                auto& session = tensorrt_llm::mpi::MpiComm::session();
                return session.tensorrt_llm::mpi::MpiComm::getSize();
            })
        .def_static("local_size",
            []()
            {
                auto& session = tensorrt_llm::mpi::MpiComm::localSession();
                return session.tensorrt_llm::mpi::MpiComm::getSize();
            })
        .def_static("local_init", []() { tensorrt_llm::mpi::MpiComm::localSession(); })
        .def_static("split",
            [](size_t color, size_t rank)
            {
                auto& world = tensorrt_llm::mpi::MpiComm::world();
                tensorrt_llm::mpi::MpiComm::setSession(world.split(color, rank));
            });
}
