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

#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

#include "tensorrt_llm/batch_manager/peftCacheManagerConfig.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/pybind/batch_manager/algorithms.h"
#include "tensorrt_llm/pybind/batch_manager/bindings.h"
#include "tensorrt_llm/pybind/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/pybind/batch_manager/kvCacheConnector.h"
#include "tensorrt_llm/pybind/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/pybind/batch_manager/llmRequest.h"
#include "tensorrt_llm/pybind/common/tllmExceptions.h"
#include "tensorrt_llm/pybind/executor/bindings.h"
#include "tensorrt_llm/pybind/process_group/bindings.h"
#include "tensorrt_llm/pybind/runtime/bindings.h"
#include "tensorrt_llm/pybind/testing/modelSpecBinding.h"
#include "tensorrt_llm/pybind/thop/bindings.h"
#include "tensorrt_llm/pybind/userbuffers/bindings.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tpb = tensorrt_llm::pybind::batch_manager;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tr::SizeType32;
using TokenIdType = tr::TokenIdType;
template <typename T>
using OptVec = std::optional<std::vector<T>>;

#if not defined(TRTLLM_PYBIND_MODULE)
#error "TRTLLM_PYBIND_MODULE must be defined"
#endif

namespace
{
tr::SamplingConfig makeSamplingConfig(std::vector<tr::SamplingConfig> const& configs)
{
    return tr::SamplingConfig(configs);
}
} // namespace

PYBIND11_MODULE(TRTLLM_PYBIND_MODULE, m)
{
    m.doc() = "TensorRT LLM Python bindings for C++ runtime";
    m.attr("binding_type") = "pybind";

    // Create MpiComm binding first since it's used in the executor bindings
    py::classh<tensorrt_llm::mpi::MpiComm>(m, "MpiComm")
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
        .def_static("set_raw_mpi_session_by_fortran_handle",
            [](int64_t fortran_handle) { tensorrt_llm::mpi::MpiComm::setRawSessionByFortran(fortran_handle); })
        .def_static("split",
            [](size_t color, size_t rank)
            {
                auto& world = tensorrt_llm::mpi::MpiComm::world();
                tensorrt_llm::mpi::MpiComm::setSession(world.split(color, rank));
            });

    py::classh<tr::CudaStream>(m, "CudaStream")
        .def(py::init(
                 [](py::object py_stream)
                 {
                     cudaStream_t stream = reinterpret_cast<cudaStream_t>(py_stream.cast<uintptr_t>());
                     return tr::CudaStream{stream};
                 }),
            py::arg("stream_ptr"))
        .def("get_device", &tr::CudaStream::getDevice);

    // Create submodule for executor bindings.
    auto mExecutor = m.def_submodule("executor", "Executor bindings");
    auto mInternal = m.def_submodule("internal", "Internal submodule of TRTLLM runtime");
    auto mInternalProcessGroup = mInternal.def_submodule("process_group", "PyTorch ProcessGroup internal bindings");
    auto mInternalRuntime = mInternal.def_submodule("runtime", "Runtime internal bindings");
    auto mInternalTesting = mInternal.def_submodule("testing", "Testing internal bindings");
    auto mInternalBatchManager = mInternal.def_submodule("batch_manager", "Batch manager internal bindings");
    auto mInternalThop = mInternal.def_submodule("thop", "Torch op internal bindings");
    auto mExceptions = m.def_submodule("exceptions", "Exceptions internal bindings");

    tensorrt_llm::pybind::executor::initBindings(mExecutor);
    tensorrt_llm::pybind::runtime::initBindingsEarly(mInternalRuntime);
    tensorrt_llm::pybind::common::initExceptionsBindings(mExceptions);
    tensorrt_llm::pybind::thop::initBindings(mInternalThop);

    auto buildInfo = m.def_submodule("BuildInfo");
    buildInfo.attr("ENABLE_MULTI_DEVICE") = py::int_(ENABLE_MULTI_DEVICE);

    py::class_<tb::PeftCacheManagerConfig>(m, "PeftCacheManagerConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, std::optional<float>, std::optional<size_t>, std::optional<std::string>>(),
            py::arg("num_host_module_layer") = 0, py::arg("num_device_module_layer") = 0,
            py::arg("optimal_adapter_size") = 8, py::arg("max_adapter_size") = 64, py::arg("num_put_workers") = 1,
            py::arg("num_ensure_workers") = 1, py::arg("num_copy_streams") = 1,
            py::arg("max_pages_per_block_host") = 24, py::arg("max_pages_per_block_device") = 8,
            py::arg("device_cache_percent") = std::nullopt, py::arg("host_cache_size") = std::nullopt,
            py::arg("lora_prefetch_dir") = std::nullopt)
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
        .def_readwrite("host_cache_size", &tb::PeftCacheManagerConfig::hostCacheSize)
        .def_readwrite("lora_prefetch_dir", &tb::PeftCacheManagerConfig::loraPrefetchDir);

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
        .value("NVFP4", nvinfer1::DataType::kFP4)
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
        .def("from_string", &tr::ModelConfig::KVCacheTypeFromString);

    py::enum_<tr::ModelConfig::LayerType>(m, "LayerType")
        .value("ATTENTION", tr::ModelConfig::LayerType::kATTENTION)
        .value("RECURRENT", tr::ModelConfig::LayerType::kRECURRENT);

    py::enum_<tr::LoraModule::ModuleType>(m, "LoraModuleType")
        .value("INVALID", tr::LoraModule::ModuleType::kINVALID)
        .value("ATTN_QKV", tr::LoraModule::ModuleType::kATTN_QKV)
        .value("ATTN_Q", tr::LoraModule::ModuleType::kATTN_Q)
        .value("ATTN_K", tr::LoraModule::ModuleType::kATTN_K)
        .value("ATTN_V", tr::LoraModule::ModuleType::kATTN_V)
        .value("ATTN_DENSE", tr::LoraModule::ModuleType::kATTN_DENSE)
        .value("MLP_H_TO_4H", tr::LoraModule::ModuleType::kMLP_H_TO_4H)
        .value("MLP_4H_TO_H", tr::LoraModule::ModuleType::kMLP_4H_TO_H)
        .value("MLP_GATE", tr::LoraModule::ModuleType::kMLP_GATE)
        .value("CROSS_ATTN_QKV", tr::LoraModule::ModuleType::kCROSS_ATTN_QKV)
        .value("CROSS_ATTN_Q", tr::LoraModule::ModuleType::kCROSS_ATTN_Q)
        .value("CROSS_ATTN_K", tr::LoraModule::ModuleType::kCROSS_ATTN_K)
        .value("CROSS_ATTN_V", tr::LoraModule::ModuleType::kCROSS_ATTN_V)
        .value("CROSS_ATTN_DENSE", tr::LoraModule::ModuleType::kCROSS_ATTN_DENSE)
        .value("MOE_H_TO_4H", tr::LoraModule::ModuleType::kMOE_H_TO_4H)
        .value("MOE_4H_TO_H", tr::LoraModule::ModuleType::kMOE_4H_TO_H)
        .value("MOE_GATE", tr::LoraModule::ModuleType::kMOE_GATE)
        .value("MOE_ROUTER", tr::LoraModule::ModuleType::kMOE_ROUTER)
        .value("MLP_ROUTER", tr::LoraModule::ModuleType::kMLP_ROUTER)
        .value("MLP_GATE_UP", tr::LoraModule::ModuleType::kMLP_GATE_UP);

    py::class_<tr::LoraModule>(m, "LoraModule")
        .def(py::init<tr::LoraModule::ModuleType, SizeType32, SizeType32, bool, bool, SizeType32, SizeType32>(),
            py::arg("module_type"), py::arg("in_dim"), py::arg("out_dim"), py::arg("in_dim_first"),
            py::arg("out_dim_first"), py::arg("in_tp_split_dim"), py::arg("out_tp_split_dim"))
        .def_property_readonly("module_type", &tr::LoraModule::name)
        .def_property_readonly("in_dim", &tr::LoraModule::inDim)
        .def_property_readonly("out_dim", &tr::LoraModule::outDim)
        .def_property_readonly("in_dim_first", &tr::LoraModule::inDimFirst)
        .def_property_readonly("out_dim_first", &tr::LoraModule::outDimFirst)
        .def_property_readonly("in_tp_split_dim", &tr::LoraModule::inTpSplitDim)
        .def_property_readonly("out_tp_split_dim", &tr::LoraModule::outTpSplitDim)
        .def_static("create_lora_modules", &tr::LoraModule::createLoraModules, py::arg("lora_module_names"),
            py::arg("hidden_size"), py::arg("mlp_hidden_size"), py::arg("num_attention_heads"),
            py::arg("num_kv_attention_heads"), py::arg("attention_head_size"), py::arg("tp_size") = 1,
            py::arg("num_experts") = 0);

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
        .def_property_readonly("has_fp4_kv_cache", &tc::QuantMode::hasFp4KvCache)
        .def_property_readonly("has_fp8_kv_cache", &tc::QuantMode::hasFp8KvCache)
        .def_property_readonly("has_fp8_qdq", &tc::QuantMode::hasFp8Qdq)
        .def_property_readonly("has_nvfp4", &tc::QuantMode::hasNvfp4)
        .def_property_readonly("has_w4a8_mxfp4_fp8", &tc::QuantMode::hasW4a8Mxfp4Fp8)
        .def_property_readonly("has_w4a8_mxfp4_mxfp8", &tc::QuantMode::hasW4a8Mxfp4Mxfp8)
        .def_property_readonly("has_w4a16_mxfp4", &tc::QuantMode::hasW4a16Mxfp4)
        .def_property_readonly("has_kv_cache_quant", &tc::QuantMode::hasKvCacheQuant)
        .def_static("from_description", &tc::QuantMode::fromDescription, py::arg("quantize_weights"),
            py::arg("quantize_activations"), py::arg("per_token"), py::arg("per_channel"), py::arg("per_group"),
            py::arg("use_int4_weights"), py::arg("use_int8_kv_cache"), py::arg("use_fp8_kv_kache"),
            py::arg("use_fp8_qdq"), py::arg("use_fp8_rowwise"), py::arg("use_w4a8_qserve"), py::arg("use_nvfp4"),
            py::arg("use_fp8_block_scales"), py::arg("use_w4a8_mxfp4_fp8"), py::arg("use_w4a8_mxfp4_mxfp8"),
            py::arg("use_w4a16_mxfp4"))
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
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, nvinfer1::DataType>(),
            py::arg("vocab_size"), py::arg("num_layers"), py::arg("num_attention_layers"), py::arg("num_rnn_layers"),
            py::arg("num_heads"), py::arg("hidden_size"), py::arg("data_type"))
        .def_property_readonly("vocab_size", &tr::ModelConfig::getVocabSize)
        .def("vocab_size_padded", &tr::ModelConfig::getVocabSizePadded, py::arg("world_size"))
        .def("num_layers", &tr::ModelConfig::getNbLayers, py::arg("pipeline_parallelism") = 1,
            py::arg("pipeline_parallelism_rank") = 0)
        .def("num_attention_layers", &tr::ModelConfig::getNbAttentionLayers, py::arg("pipeline_parallelism") = 1,
            py::arg("pipeline_parallelism_rank") = 0)
        .def("num_rnn_layers", &tr::ModelConfig::getNbRnnLayers, py::arg("pipeline_parallelism") = 1,
            py::arg("pipeline_parallelism_rank") = 0)
        .def("num_kv_heads", &tr::ModelConfig::getNbKvHeads, py::arg("layer_idx"))
        .def("set_num_kv_heads", &tr::ModelConfig::setNbKvHeads, py::arg("num_kv_heads"))
        .def_property_readonly("num_heads", &tr::ModelConfig::getNbHeads)
        .def_property_readonly("hidden_size", &tr::ModelConfig::getHiddenSize)
        .def_property_readonly("size_per_head", &tr::ModelConfig::getSizePerHead)
        .def_property_readonly("data_type", &tr::ModelConfig::getDataType)
        .def_property_readonly("speculative_decoding_mode", &tr::ModelConfig::getSpeculativeDecodingMode)
        .def_property("head_size", &tr::ModelConfig::getSizePerHead, &tr::ModelConfig::setSizePerHead)
        .def_property(
            "num_kv_heads_per_layer", &tr::ModelConfig::getNumKvHeadsPerLayer, &tr::ModelConfig::setNumKvHeadsPerLayer)
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
        .def_property("max_seq_len", &tr::ModelConfig::getMaxSequenceLen, &tr::ModelConfig::setMaxSequenceLen)
        .def_property("max_num_tokens", &tr::ModelConfig::getMaxNumTokens, &tr::ModelConfig::setMaxNumTokens)
        .def_property("max_prompt_embedding_table_size", &tr::ModelConfig::getMaxPromptEmbeddingTableSize,
            &tr::ModelConfig::setMaxPromptEmbeddingTableSize)
        .def_property_readonly("use_prompt_tuning", &tr::ModelConfig::usePromptTuning)
        .def_property_readonly("use_mrope", &tr::ModelConfig::useMrope)
        .def_property("use_lora_plugin", py::overload_cast<>(&tr::ModelConfig::useLoraPlugin, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::useLoraPlugin))
        .def_property("layer_types", &tr::ModelConfig::getLayerTypes, &tr::ModelConfig::setLayerTypes)
        .def_property("compute_context_logits", py::overload_cast<>(&tr::ModelConfig::computeContextLogits, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::computeContextLogits))
        .def_property("compute_generation_logits",
            py::overload_cast<>(&tr::ModelConfig::computeGenerationLogits, py::const_),
            py::overload_cast<bool>(&tr::ModelConfig::computeGenerationLogits))
        .def_property("model_variant", &tr::ModelConfig::getModelVariant, &tr::ModelConfig::setModelVariant)
        .def_property(
            "use_cross_attention", &tr::ModelConfig::useCrossAttention, &tr::ModelConfig::setUseCrossAttention)
        .def_property("lora_modules", &tr::ModelConfig::getLoraModules, &tr::ModelConfig::setLoraModules)
        .def_property("max_lora_rank", &tr::ModelConfig::getMaxLoraRank, &tr::ModelConfig::setMaxLoraRank)
        .def_property("mlp_hidden_size", &tr::ModelConfig::getMlpHiddenSize, &tr::ModelConfig::setMlpHiddenSize)
        .def_property("size_per_head", &tr::ModelConfig::getSizePerHead, &tr::ModelConfig::setSizePerHead);

    py::class_<tr::WorldConfig>(m, "WorldConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 std::optional<std::vector<SizeType32>> const&, bool>(),
            py::arg("tensor_parallelism") = 1, py::arg("pipeline_parallelism") = 1, py::arg("context_parallelism") = 1,
            py::arg("rank") = 0, py::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode,
            py::arg("device_ids") = py::none(), py::arg("enable_attention_dp") = false)
        .def_property_readonly("size", &tr::WorldConfig::getSize)
        .def_property_readonly("tensor_parallelism", &tr::WorldConfig::getTensorParallelism)
        .def_property_readonly("pipeline_parallelism", &tr::WorldConfig::getPipelineParallelism)
        .def_property_readonly("context_parallelism", &tr::WorldConfig::getContextParallelism)
        .def_property_readonly("is_tensor_parallel", &tr::WorldConfig::isTensorParallel)
        .def_property_readonly("is_pipeline_parallel", &tr::WorldConfig::isPipelineParallel)
        .def_property_readonly("is_context_parallel", &tr::WorldConfig::isContextParallel)
        .def_property_readonly("rank", &tr::WorldConfig::getRank)
        .def_property_readonly("local_rank", &tr::WorldConfig::getLocalRank)
        .def_property_readonly("node_rank", &tr::WorldConfig::getNodeRank)
        .def_property_readonly("gpus_per_node", &tr::WorldConfig::getGpusPerNode)
        .def_property_readonly("gpus_per_group", &tr::WorldConfig::getGpusPerGroup)
        .def_property_readonly("device", &tr::WorldConfig::getDevice)
        .def_property_readonly("pipeline_parallel_rank", &tr::WorldConfig::getPipelineParallelRank)
        .def_property_readonly("tensor_parallel_rank", &tr::WorldConfig::getTensorParallelRank)
        .def_property_readonly("context_parallel_rank", &tr::WorldConfig::getContextParallelRank)
        .def_property_readonly("enable_attention_dp", &tr::WorldConfig::enableAttentionDP)
        .def_static("mpi",
            py::overload_cast<SizeType32, std::optional<SizeType32>, std::optional<SizeType32>,
                std::optional<SizeType32>, std::optional<std::vector<SizeType32>> const&, bool>(&tr::WorldConfig::mpi),
            py::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode, py::arg("tensor_parallelism") = py::none(),
            py::arg("pipeline_parallelism") = py::none(), py::arg("context_parallelism") = py::none(),
            py::arg("device_ids") = py::none(), py::arg("enable_attention_dp") = false);

    auto SamplingConfigGetState = [](tr::SamplingConfig const& config) -> py::tuple
    {
        return py::make_tuple(config.beamWidth, config.temperature, config.minLength, config.repetitionPenalty,
            config.presencePenalty, config.frequencyPenalty, config.topK, config.topP, config.randomSeed,
            config.topPDecay, config.topPMin, config.topPResetIds, config.beamSearchDiversityRate, config.lengthPenalty,
            config.earlyStopping, config.noRepeatNgramSize, config.numReturnSequences, config.minP,
            config.beamWidthArray);
    };
    auto SamplingConfigSetState = [](py::tuple t) -> tr::SamplingConfig
    {
        if (t.size() != 19)
        {
            throw std::runtime_error("Invalid SamplingConfig state!");
        }

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
        config.numReturnSequences = t[16].cast<SizeType32>();
        config.minP = t[17].cast<OptVec<float>>();
        config.beamWidthArray = t[18].cast<OptVec<std::vector<SizeType32>>>();

        return config;
    };

    py::classh<tr::SamplingConfig>(m, "SamplingConfig")
        .def(py::init<SizeType32>(), py::arg("beam_width") = 1)
        .def(py::init<tle::SamplingConfig, std::optional<tle::ExternalDraftTokensConfig>>(),
            py::arg("executor_sample_config"), py::arg("external_draft_tokens_config") = std::nullopt)
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
        .def_readwrite("num_return_sequences", &tr::SamplingConfig::numReturnSequences)
        .def_readwrite("min_p", &tr::SamplingConfig::minP)
        .def_readwrite("beam_width_array", &tr::SamplingConfig::beamWidthArray)
        .def_readwrite("normalize_log_probs", &tr::SamplingConfig::normalizeLogProbs)
        .def(py::pickle(SamplingConfigGetState, SamplingConfigSetState))
        .def("__eq__", &tr::SamplingConfig::operator==);

    m.def("make_sampling_config", &makeSamplingConfig, py::arg("configs"));

    py::class_<tr::GptJsonConfig>(m, "GptJsonConfig")
        .def(py::init<std::string, std::string, std::string, SizeType32, SizeType32, SizeType32, SizeType32,
                 tr::ModelConfig, std::optional<tr::RuntimeDefaults>>(),
            py::arg("name"), py::arg("version"), py::arg("precision"), py::arg("tensor_parallelism"),
            py::arg("pipeline_parallelism"), py::arg("context_parallelism"), py::arg("gpus_per_node"),
            py::arg("model_config"), py::arg("runtime_defaults") = py::none())
        .def_static("parse", py::overload_cast<std::string const&>(&tr::GptJsonConfig::parse), py::arg("json"))
        .def_static(
            "parse_file", py::overload_cast<std::filesystem::path const&>(&tr::GptJsonConfig::parse), py::arg("path"))
        .def_property_readonly("model_config", &tr::GptJsonConfig::getModelConfig)
        .def_property_readonly("name", &tr::GptJsonConfig::getName)
        .def_property_readonly("version", &tr::GptJsonConfig::getVersion)
        .def_property_readonly("precision", &tr::GptJsonConfig::getPrecision)
        .def_property_readonly("tensor_parallelism", &tr::GptJsonConfig::getTensorParallelism)
        .def_property_readonly("pipeline_parallelism", &tr::GptJsonConfig::getPipelineParallelism)
        .def_property_readonly("context_parallelism", &tr::GptJsonConfig::getContextParallelism)
        .def_property_readonly("gpus_per_node", &tr::GptJsonConfig::getGpusPerNode)
        .def_property_readonly("world_size", &tr::GptJsonConfig::getWorldSize)
        .def_property_readonly("runtime_defaults", &tr::GptJsonConfig::getRuntimeDefaults)
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
        .value("GENERATION_COMPLETE", tb::LlmRequestState::kGENERATION_COMPLETE)
        .value("DISAGG_GENERATION_INIT", tb::LlmRequestState::kDISAGG_GENERATION_INIT)
        .value("DISAGG_CONTEXT_TRANS_IN_PROGRESS", tb::LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS)
        .value("DISAGG_CONTEXT_COMPLETE", tb::LlmRequestState::kDISAGG_CONTEXT_COMPLETE)
        .value("DISAGG_GENERATION_TRANS_IN_PROGRESS", tb::LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS)
        .value("DISAGG_TRANS_ERROR", tb::LlmRequestState::kDISAGG_TRANS_ERROR)
        .value("DISAGG_GENERATION_TRANS_COMPLETE", tb::LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE)
        .value("DISAGG_CONTEXT_INIT_AND_TRANS", tb::LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS);

    py::class_<tr::MemoryCounters>(m, "MemoryCounters")
        .def_static("instance", &tr::MemoryCounters::getInstance, py::return_value_policy::reference)
        .def_property_readonly("gpu", &tr::MemoryCounters::getGpu)
        .def_property_readonly("cpu", &tr::MemoryCounters::getCpu)
        .def_property_readonly("pinned", &tr::MemoryCounters::getPinned)
        .def_property_readonly("uvm", &tr::MemoryCounters::getUVM);

    tensorrt_llm::pybind::process_group::initBindings(mInternalProcessGroup);
    tensorrt_llm::pybind::runtime::initBindings(mInternalRuntime);
    tensorrt_llm::pybind::testing::initBindings(mInternalTesting);
    tpb::initBindings(mInternalBatchManager);
    tb::kv_cache_manager::KVCacheManagerConnectorBindings::initBindings(mInternalBatchManager);
    tb::kv_cache_manager::KVCacheManagerBindings::initBindings(mInternalBatchManager);
    tb::BasePeftCacheManagerBindings::initBindings(mInternalBatchManager);
    tb::CacheTransceiverBindings::initBindings(mInternalBatchManager);

    auto mInternalAlgorithms = mInternal.def_submodule("algorithms", "Algorithms internal bindings");
    tpb::algorithms::initBindings(mInternalAlgorithms);

    auto mUserbuffers = mInternal.def_submodule("userbuffers", "User buffers internal bindings");
    tensorrt_llm::kernels::userbuffers::UserBufferBindings::initBindings(mUserbuffers);

    // NVLS allocators
    py::class_<tr::IpcNvlsHandle>(m, "IpcNvlsHandle")
        .def(py::init<>())
        .def_readwrite("uc_ptr", &tr::IpcNvlsHandle::uc_ptr)
        .def_readwrite("mc_ptr", &tr::IpcNvlsHandle::mc_ptr)
        .def_readwrite("size", &tr::IpcNvlsHandle::size)
        .def("get_ipc_ptrs",
            [](tr::IpcNvlsHandle& self) { return reinterpret_cast<uintptr_t>(self.ipc_uc_ptrs.data()); });

    m.def("ipc_nvls_allocate", &tr::ipcNvlsAllocate, py::return_value_policy::reference);
    m.def("ipc_nvls_free", &tr::ipcNvlsFree);
    m.def("ipc_nvls_supported", &tr::ipcNvlsSupported);

    m.def("steady_clock_now", []() { return std::chrono::steady_clock::now(); });
}
