/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/operators.h>
#include <torch/extension.h>

#include "runtime/generationInput.h"
#include "runtime/generationOutput.h"

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;
namespace tpr = tensorrt_llm::pybind::runtime;

#if not defined(TRTLLM_PYBIND_MODULE)
#error "TRTLLM_PYBIND_MODULE must be defined"
#endif

PYBIND11_MODULE(TRTLLM_PYBIND_MODULE, m)
{
    m.doc() = "TensorRT-LLM Python bindings for C++ runtime";

    py::class_<tpr::PromptTuningParams>(m, "PromptTuningParams")
        .def(py::init<tpr::PromptTuningParams::TensorPtr, tpr::PromptTuningParams::TensorPtr,
                 tpr::PromptTuningParams::TensorPtr>(),
            py::arg("embedding_table") = py::none(), py::arg("tasks") = py::none(), py::arg("vocab_size") = py::none())
        .def_readwrite("embedding_table", &tpr::PromptTuningParams::embeddingTable)
        .def_readwrite("tasks", &tpr::PromptTuningParams::tasks)
        .def_readwrite("vocab_size", &tpr::PromptTuningParams::vocabSize)
        .def_readwrite("prompt_tuning_enabled", &tpr::PromptTuningParams::promptTuningEnabled);

    py::class_<tpr::GenerationInput>(m, "GenerationInput")
        .def(py::init<tr::SizeType, tr::SizeType, tpr::GenerationInput::TensorPtr, tpr::GenerationInput::TensorPtr,
                 bool>(),
            py::arg("end_id"), py::arg("pad_id"), py::arg("ids"), py::arg("lengths"), py::arg("packed") = false)
        .def_readwrite("end_id", &tpr::GenerationInput::endId)
        .def_readwrite("pad_id", &tpr::GenerationInput::padId)
        .def_readwrite("ids", &tpr::GenerationInput::ids)
        .def_readwrite("lengths", &tpr::GenerationInput::lengths)
        .def_readwrite("packed", &tpr::GenerationInput::packed)
        .def_readwrite("embedding_bias", &tpr::GenerationInput::embeddingBias)
        .def_readwrite("bad_words_list", &tpr::GenerationInput::badWordsList)
        .def_readwrite("stop_words_list", &tpr::GenerationInput::stopWordsList)
        .def_readwrite("max_new_tokens", &tpr::GenerationInput::maxNewTokens)
        .def_readwrite("prompt_tuning_params", &tpr::GenerationInput::promptTuningParams);

    py::class_<tpr::GenerationOutput>(m, "GenerationOutput")
        .def(py::init<tpr::GenerationOutput::TensorPtr, tpr::GenerationOutput::TensorPtr>(), py::arg("ids"),
            py::arg("lengths"))
        .def_readwrite("ids", &tpr::GenerationOutput::ids)
        .def_readwrite("lengths", &tpr::GenerationOutput::lengths)
        .def_readwrite("log_probs", &tpr::GenerationOutput::logProbs)
        .def_readwrite("context_logits", &tpr::GenerationOutput::contextLogits)
        .def_readwrite("on_token_generated", &tpr::GenerationOutput::onTokenGenerated);

    py::class_<tb::kv_cache_manager::KvCacheConfig>(m, "KvCacheConfig")
        .def(py::init<std::optional<tr::SizeType>, std::optional<tr::SizeType>, std::optional<float>>(),
            py::arg("max_tokens") = py::none(), py::arg("max_kv_cache_length") = py::none(),
            py::arg("free_gpu_memory_fraction") = py::none())
        .def_readwrite("max_tokens", &tb::kv_cache_manager::KvCacheConfig::maxTokens)
        .def_readwrite("max_kv_cache_length", &tb::kv_cache_manager::KvCacheConfig::maxKvCacheLength)
        .def_readwrite("free_gpu_memory_fraction", &tb::kv_cache_manager::KvCacheConfig::freeGpuMemoryFraction);

    py::class_<tr::GptSession::Config>(m, "GptSessionConfig")
        .def(py::init<tr::SizeType, tr::SizeType, tr::SizeType>(), py::arg("max_batch_size"), py::arg("max_beam_width"),
            py::arg("max_sequence_length"))
        .def_readwrite("max_batch_size", &tr::GptSession::Config::maxBatchSize)
        .def_readwrite("max_beam_width", &tr::GptSession::Config::maxBeamWidth)
        .def_readwrite("max_sequence_length", &tr::GptSession::Config::maxSequenceLength)
        .def_readwrite("decoder_per_request", &tr::GptSession::Config::decoderPerRequest)
        .def_readwrite("cuda_graph_mode", &tr::GptSession::Config::cudaGraphMode)
        .def_readwrite("ctx_micro_batch_size", &tr::GptSession::Config::ctxMicroBatchSize)
        .def_readwrite("gen_micro_batch_size", &tr::GptSession::Config::genMicroBatchSize)
        .def_readwrite("kv_cache_config", &tr::GptSession::Config::kvCacheConfig);

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

    py::enum_<tr::GptModelConfig::ModelVariant>(m, "GptModelVariant")
        .value("GPT", tr::GptModelConfig::ModelVariant::kGpt)
        .value("GLM", tr::GptModelConfig::ModelVariant::kGlm);

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
            py::arg("use_int4_weights") = false, py::arg("use_int8_kv_cache") = false,
            py::arg("use_fp8_kv_kache") = false, py::arg("use_fp8_qdq") = false)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<tr::GptModelConfig>(m, "GptModelConfig")
        .def(py::init<tr::SizeType, tr::SizeType, tr::SizeType, tr::SizeType, nvinfer1::DataType>(),
            py::arg("vocab_size"), py::arg("num_layers"), py::arg("num_heads"), py::arg("hidden_size"),
            py::arg("data_type"))
        .def_property_readonly("vocab_size", &tr::GptModelConfig::getVocabSize)
        .def("vocab_size_padded", &tr::GptModelConfig::getVocabSizePadded, py::arg("world_size"))
        .def("num_layers", &tr::GptModelConfig::getNbLayers, py::arg("pipeline_parallelism") = 1)
        .def_property_readonly("num_heads", &tr::GptModelConfig::getNbHeads)
        .def_property_readonly("hidden_size", &tr::GptModelConfig::getHiddenSize)
        .def_property_readonly("size_per_head", &tr::GptModelConfig::getSizePerHead)
        .def_property_readonly("data_type", &tr::GptModelConfig::getDataType)
        .def_property("num_kv_heads", &tr::GptModelConfig::getNbKvHeads, &tr::GptModelConfig::setNbKvHeads)
        .def_property("use_gpt_attention_plugin",
            py::overload_cast<>(&tr::GptModelConfig::useGptAttentionPlugin, py::const_),
            py::overload_cast<bool>(&tr::GptModelConfig::useGptAttentionPlugin))
        .def_property("use_packed_input", py::overload_cast<>(&tr::GptModelConfig::usePackedInput, py::const_),
            py::overload_cast<bool>(&tr::GptModelConfig::usePackedInput))
        .def_property("use_paged_kv_cache", py::overload_cast<>(&tr::GptModelConfig::usePagedKvCache, py::const_),
            py::overload_cast<bool>(&tr::GptModelConfig::usePagedKvCache))
        .def_property(
            "tokens_per_block", &tr::GptModelConfig::getTokensPerBlock, &tr::GptModelConfig::setTokensPerBlock)
        .def_property("quant_mode", &tr::GptModelConfig::getQuantMode, &tr::GptModelConfig::setQuantMode)
        .def_property_readonly("supports_inflight_batching", &tr::GptModelConfig::supportsInflightBatching)
        .def_property("max_batch_size", &tr::GptModelConfig::getMaxBatchSize, &tr::GptModelConfig::setMaxBatchSize)
        .def_property("max_input_len", &tr::GptModelConfig::getMaxInputLen, &tr::GptModelConfig::setMaxInputLen)
        .def_property("max_output_len", &tr::GptModelConfig::getMaxOutputLen, &tr::GptModelConfig::setMaxOutputLen)
        .def_property("max_num_tokens", &tr::GptModelConfig::getMaxNumTokens, &tr::GptModelConfig::setMaxNumTokens)
        .def_property("compute_context_logits",
            py::overload_cast<>(&tr::GptModelConfig::computeContextLogits, py::const_),
            py::overload_cast<bool>(&tr::GptModelConfig::computeContextLogits))
        .def_property("compute_generation_logits",
            py::overload_cast<>(&tr::GptModelConfig::computeGenerationLogits, py::const_),
            py::overload_cast<bool>(&tr::GptModelConfig::computeGenerationLogits))
        .def_property("model_variant", &tr::GptModelConfig::getModelVariant, &tr::GptModelConfig::setModelVariant)
        .def_property("use_custom_all_reduce", py::overload_cast<>(&tr::GptModelConfig::useCustomAllReduce, py::const_),
            py::overload_cast<bool>(&tr::GptModelConfig::useCustomAllReduce));

    py::class_<tr::WorldConfig>(m, "WorldConfig")
        .def(py::init<tr::SizeType, tr::SizeType, tr::SizeType, tr::SizeType>(), py::arg("tensor_parallelism") = 1,
            py::arg("pipeline_parallelism") = 1, py::arg("rank") = 0,
            py::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode)
        .def_property_readonly("size", &tr::WorldConfig::getSize)
        .def_property_readonly("tensor_parallelism", &tr::WorldConfig::getTensorParallelism)
        .def_property_readonly("pipeline_parallelism", &tr::WorldConfig::getPipelineParallelism)
        .def_property_readonly("is_tensor_parallel", &tr::WorldConfig::isTensorParallel)
        .def_property_readonly("is_pipeline_parallel", &tr::WorldConfig::isPipelineParallel)
        .def_property_readonly("rank", &tr::WorldConfig::getRank)
        .def_property_readonly("gpus_per_node", &tr::WorldConfig::getGpusPerNode)
        .def_property_readonly("device", &tr::WorldConfig::getDevice)
        .def_property_readonly("pipeline_parallel_rank", &tr::WorldConfig::getPipelineParallelRank)
        .def_property_readonly("tensor_parallel_rank", &tr::WorldConfig::getTensorParallelRank)
        .def_static("mpi",
            py::overload_cast<tr::SizeType, std::optional<tr::SizeType>, std::optional<tr::SizeType>>(
                &tr::WorldConfig::mpi),
            py::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode, py::arg("tensor_parallelism") = py::none(),
            py::arg("pipeline_parallelism") = py::none());

    py::class_<tr::SamplingConfig>(m, "SamplingConfig")
        .def(py::init<tr::SizeType>(), py::arg("beam_width") = 1)
        .def_readwrite("beam_width", &tr::SamplingConfig::beamWidth)
        .def_readwrite("temperature", &tr::SamplingConfig::temperature)
        .def_readwrite("min_length", &tr::SamplingConfig::minLength)
        .def_readwrite("repetition_penalty", &tr::SamplingConfig::repetitionPenalty)
        .def_readwrite("presence_penalty", &tr::SamplingConfig::presencePenalty)
        .def_readwrite("top_k", &tr::SamplingConfig::topK)
        .def_readwrite("top_p", &tr::SamplingConfig::topP)
        .def_readwrite("random_seed", &tr::SamplingConfig::randomSeed)
        .def_readwrite("top_p_decay", &tr::SamplingConfig::topPDecay)
        .def_readwrite("top_p_min", &tr::SamplingConfig::topPMin)
        .def_readwrite("top_p_reset_ids", &tr::SamplingConfig::topPResetIds)
        .def_readwrite("beam_search_diversity_rate", &tr::SamplingConfig::beamSearchDiversityRate)
        .def_readwrite("length_penalty", &tr::SamplingConfig::lengthPenalty);

    py::class_<tr::GptJsonConfig>(m, "GptJsonConfig")
        .def(py::init<std::string, std::string, tr::SizeType, tr::SizeType, tr::GptModelConfig>(), py::arg("name"),
            py::arg("precision"), py::arg("tensor_parallelism"), py::arg("pipeline_parallelism"),
            py::arg("model_config"))
        .def_static("parse", py::overload_cast<std::string const&>(&tr::GptJsonConfig::parse), py::arg("json"))
        .def_static(
            "parse_file", [](std::string const& file) { return tr::GptJsonConfig::parse(std::filesystem::path(file)); },
            py::arg("file"))
        .def_property_readonly("model_config", &tr::GptJsonConfig::getModelConfig)
        .def_property_readonly("name", &tr::GptJsonConfig::getName)
        .def_property_readonly("precision", &tr::GptJsonConfig::getPrecision)
        .def_property_readonly("tensor_parallelism", &tr::GptJsonConfig::getTensorParallelism)
        .def_property_readonly("pipeline_parallelism", &tr::GptJsonConfig::getPipelineParallelism)
        .def_property_readonly("world_size", &tr::GptJsonConfig::getWorldSize)
        .def("engine_filename",
            py::overload_cast<const tr::WorldConfig&, const std::string&>(
                &tr::GptJsonConfig::engineFilename, py::const_),
            py::arg("world_config"), py::arg("model"))
        .def("engine_filename",
            py::overload_cast<const tr::WorldConfig&>(&tr::GptJsonConfig::engineFilename, py::const_),
            py::arg("world_config"));

    py::class_<tr::GptSession>(m, "GptSession")
        .def(py::init<tr::GptSession::Config, tr::GptModelConfig, tr::WorldConfig, std::string>(), py::arg("config"),
            py::arg("model_config"), py::arg("world_config"), py::arg("engine_file"))
        .def_property_readonly("model_config", &tr::GptSession::getModelConfig)
        .def_property_readonly("world_config", &tr::GptSession::getWorldConfig)
        .def_property_readonly("device", &tr::GptSession::getDevice)
        .def(
            "generate",
            [](tr::GptSession& self, tpr::GenerationOutput& outputs, tpr::GenerationInput const& inputs,
                tr::SamplingConfig const& samplingConfig)
            { self.generate(*outputs.toTrtLlm(), *inputs.toTrtLlm(), samplingConfig); },
            py::arg("outputs"), py::arg("inputs"), py::arg("sampling_config"));
}
