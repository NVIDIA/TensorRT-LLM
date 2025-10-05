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

#include "tensorrt_llm/nanobind/common/customCasters.h"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>

#include <torch/extension.h>
#include <vector>

#include "tensorrt_llm/batch_manager/peftCacheManagerConfig.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/nanobind/batch_manager/algorithms.h"
#include "tensorrt_llm/nanobind/batch_manager/bindings.h"
#include "tensorrt_llm/nanobind/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/nanobind/batch_manager/kvCacheConnector.h"
#include "tensorrt_llm/nanobind/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/nanobind/batch_manager/llmRequest.h"
#include "tensorrt_llm/nanobind/common/tllmExceptions.h"
#include "tensorrt_llm/nanobind/executor/bindings.h"
#include "tensorrt_llm/nanobind/process_group/bindings.h"
#include "tensorrt_llm/nanobind/runtime/bindings.h"
#include "tensorrt_llm/nanobind/testing/modelSpecBinding.h"
#include "tensorrt_llm/nanobind/thop/bindings.h"
#include "tensorrt_llm/nanobind/userbuffers/bindings.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace nb = nanobind;
namespace tb = tensorrt_llm::batch_manager;
namespace tpb = tensorrt_llm::nanobind::batch_manager;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;
using SizeType32 = tr::SizeType32;
using TokenIdType = tr::TokenIdType;
template <typename T>
using OptVec = std::optional<std::vector<T>>;

#if not defined(TRTLLM_NB_MODULE)
#error "TRTLLM_NB_MODULE must be defined"
#endif

namespace
{
tr::SamplingConfig makeSamplingConfig(std::vector<tr::SamplingConfig> const& configs)
{
    return tr::SamplingConfig(configs);
}
} // namespace

NB_MODULE(TRTLLM_NB_MODULE, m)
{
    m.doc() = "TensorRT LLM Python bindings for C++ runtime";
    m.attr("binding_type") = "nanobind";
    nb::set_leak_warnings(false);

    // Create MpiComm binding first since it's used in the executor bindings
    nb::class_<tensorrt_llm::mpi::MpiComm>(m, "MpiComm")
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

    nb::class_<tr::CudaStream>(m, "CudaStream")
        .def(
            "__init__",
            [](tr::CudaStream* self, nb::object py_stream)
            {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(py_stream));
                new (self) tr::CudaStream{stream};
            },
            nb::arg("stream_ptr"))
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

    tensorrt_llm::nanobind::executor::initBindings(mExecutor);
    tensorrt_llm::nanobind::runtime::initBindingsEarly(mInternalRuntime);
    tensorrt_llm::nanobind::common::initExceptionsBindings(mExceptions);
    tensorrt_llm::nanobind::thop::initBindings(mInternalThop);

    auto buildInfo = m.def_submodule("BuildInfo");
    buildInfo.attr("ENABLE_MULTI_DEVICE") = nb::int_(ENABLE_MULTI_DEVICE);

    nb::class_<tb::PeftCacheManagerConfig>(m, "PeftCacheManagerConfig")
        .def(nb::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, std::optional<float>, std::optional<size_t>, std::optional<std::string>>(),
            nb::arg("num_host_module_layer") = 0, nb::arg("num_device_module_layer") = 0,
            nb::arg("optimal_adapter_size") = 8, nb::arg("max_adapter_size") = 64, nb::arg("num_put_workers") = 1,
            nb::arg("num_ensure_workers") = 1, nb::arg("num_copy_streams") = 1,
            nb::arg("max_pages_per_block_host") = 24, nb::arg("max_pages_per_block_device") = 8,
            nb::arg("device_cache_percent") = std::nullopt, nb::arg("host_cache_size") = std::nullopt,
            nb::arg("lora_prefetch_dir") = std::nullopt)
        .def_rw("num_host_module_layer", &tb::PeftCacheManagerConfig::numHostModuleLayer)
        .def_rw("num_device_module_layer", &tb::PeftCacheManagerConfig::numDeviceModuleLayer)
        .def_rw("optimal_adapter_size", &tb::PeftCacheManagerConfig::optimalAdapterSize)
        .def_rw("max_adapter_size", &tb::PeftCacheManagerConfig::maxAdapterSize)
        .def_rw("num_put_workers", &tb::PeftCacheManagerConfig::numPutWorkers)
        .def_rw("num_ensure_workers", &tb::PeftCacheManagerConfig::numEnsureWorkers)
        .def_rw("num_copy_streams", &tb::PeftCacheManagerConfig::numCopyStreams)
        .def_rw("max_pages_per_block_host", &tb::PeftCacheManagerConfig::maxPagesPerBlockHost)
        .def_rw("max_pages_per_block_device", &tb::PeftCacheManagerConfig::maxPagesPerBlockDevice)
        .def_rw("device_cache_percent", &tb::PeftCacheManagerConfig::deviceCachePercent)
        .def_rw("host_cache_size", &tb::PeftCacheManagerConfig::hostCacheSize)
        .def_rw("lora_prefetch_dir", &tb::PeftCacheManagerConfig::loraPrefetchDir);

    nb::enum_<nvinfer1::DataType>(m, "DataType")
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

    nb::enum_<tr::ModelConfig::ModelVariant>(m, "GptModelVariant")
        .value("GPT", tr::ModelConfig::ModelVariant::kGpt)
        .value("GLM", tr::ModelConfig::ModelVariant::kGlm)
        .value("CHATGLM", tr::ModelConfig::ModelVariant::kChatGlm)
        .value("MAMBA", tr::ModelConfig::ModelVariant::kMamba)
        .value("RECURRENTGEMMA", tr::ModelConfig::ModelVariant::kRecurrentGemma);

    nb::enum_<tr::ModelConfig::KVCacheType>(m, "KVCacheType")
        .value("CONTINUOUS", tr::ModelConfig::KVCacheType::kCONTINUOUS)
        .value("PAGED", tr::ModelConfig::KVCacheType::kPAGED)
        .value("DISABLED", tr::ModelConfig::KVCacheType::kDISABLED)
        .def("from_string", tr::ModelConfig::KVCacheTypeFromString);

    nb::enum_<tr::ModelConfig::LayerType>(m, "LayerType")
        .value("ATTENTION", tr::ModelConfig::LayerType::kATTENTION)
        .value("RECURRENT", tr::ModelConfig::LayerType::kRECURRENT);

    nb::enum_<tr::LoraModule::ModuleType>(m, "LoraModuleType")
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

    nb::class_<tr::LoraModule>(m, "LoraModule")
        .def(nb::init<tr::LoraModule::ModuleType, SizeType32, SizeType32, bool, bool, SizeType32, SizeType32>(),
            nb::arg("module_type"), nb::arg("in_dim"), nb::arg("out_dim"), nb::arg("in_dim_first"),
            nb::arg("out_dim_first"), nb::arg("in_tp_split_dim"), nb::arg("out_tp_split_dim"))
        .def_prop_ro("module_type", &tr::LoraModule::name)
        .def_prop_ro("in_dim", &tr::LoraModule::inDim)
        .def_prop_ro("out_dim", &tr::LoraModule::outDim)
        .def_prop_ro("in_dim_first", &tr::LoraModule::inDimFirst)
        .def_prop_ro("out_dim_first", &tr::LoraModule::outDimFirst)
        .def_prop_ro("in_tp_split_dim", &tr::LoraModule::inTpSplitDim)
        .def_prop_ro("out_tp_split_dim", &tr::LoraModule::outTpSplitDim)
        .def_static("create_lora_modules", &tr::LoraModule::createLoraModules, nb::arg("lora_module_names"),
            nb::arg("hidden_size"), nb::arg("mlp_hidden_size"), nb::arg("num_attention_heads"),
            nb::arg("num_kv_attention_heads"), nb::arg("attention_head_size"), nb::arg("tp_size") = 1,
            nb::arg("num_experts") = 0);

    nb::class_<tc::QuantMode>(m, "QuantMode")
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
        .def_prop_ro("value", &tc::QuantMode::value)
        .def("is_set", &tc::QuantMode::isSet, nb::arg("mode"))
        .def_prop_ro("has_int4_weights", &tc::QuantMode::hasInt4Weights)
        .def_prop_ro("has_int8_weights", &tc::QuantMode::hasInt8Weights)
        .def_prop_ro("has_activations", &tc::QuantMode::hasActivations)
        .def_prop_ro("has_per_channel_scaling", &tc::QuantMode::hasPerChannelScaling)
        .def_prop_ro("has_per_token_scaling", &tc::QuantMode::hasPerTokenScaling)
        .def_prop_ro("has_per_group_scaling", &tc::QuantMode::hasPerGroupScaling)
        .def_prop_ro("has_static_activation_scaling", &tc::QuantMode::hasStaticActivationScaling)
        .def_prop_ro("has_int8_kv_cache", &tc::QuantMode::hasInt8KvCache)
        .def_prop_ro("has_fp4_kv_cache", &tc::QuantMode::hasFp4KvCache)
        .def_prop_ro("has_fp8_kv_cache", &tc::QuantMode::hasFp8KvCache)
        .def_prop_ro("has_fp8_qdq", &tc::QuantMode::hasFp8Qdq)
        .def_prop_ro("has_nvfp4", &tc::QuantMode::hasNvfp4)
        .def_prop_ro("has_w4a8_mxfp4_fp8", &tc::QuantMode::hasW4a8Mxfp4Fp8)

        .def_prop_ro("has_w4a8_mxfp4_mxfp8", &tc::QuantMode::hasW4a8Mxfp4Mxfp8)
        .def_prop_ro("has_w4a16_mxfp4", &tc::QuantMode::hasW4a16Mxfp4)

        .def_prop_ro("has_kv_cache_quant", &tc::QuantMode::hasKvCacheQuant)
        .def_static("from_description", &tc::QuantMode::fromDescription, nb::arg("quantize_weights"),
            nb::arg("quantize_activations"), nb::arg("per_token"), nb::arg("per_channel"), nb::arg("per_group"),
            nb::arg("use_int4_weights"), nb::arg("use_int8_kv_cache"), nb::arg("use_fp8_kv_kache"),
            nb::arg("use_fp8_qdq"), nb::arg("use_fp8_rowwise"), nb::arg("use_w4a8_qserve"), nb::arg("use_nvfp4"),
            nb::arg("use_fp8_block_scales"), nb::arg("use_w4a8_mxfp4_fp8"), nb::arg("use_w4a8_mxfp4_mxfp8"),
            nb::arg("use_w4a16_mxfp4"))
        .def_static("use_smooth_quant", &tc::QuantMode::useSmoothQuant, nb::arg("per_token") = false,
            nb::arg("per_channel") = false)
        .def_static("use_weight_only", &tc::QuantMode::useWeightOnly, nb::arg("use_int4_weights") = false,
            nb::arg("per_group") = false)
        .def_static("from_quant_algo", &tc::QuantMode::fromQuantAlgo, nb::arg("quant_algo") = nb::none(),
            nb::arg("kv_cache_quant_algo") = nb::none())
        .def(nb::self + nb::self)
        .def(nb::self += nb::self)
        .def(nb::self - nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    nb::class_<tr::ModelConfig>(m, "ModelConfig")
        .def(nb::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, nvinfer1::DataType>(),
            nb::arg("vocab_size"), nb::arg("num_layers"), nb::arg("num_attention_layers"), nb::arg("num_rnn_layers"),
            nb::arg("num_heads"), nb::arg("hidden_size"), nb::arg("data_type"))
        .def_prop_ro("vocab_size", &tr::ModelConfig::getVocabSize)
        .def("vocab_size_padded", &tr::ModelConfig::getVocabSizePadded, nb::arg("world_size"))
        .def("num_layers", &tr::ModelConfig::getNbLayers, nb::arg("pipeline_parallelism") = 1,
            nb::arg("pipeline_parallelism_rank") = 0)
        .def("num_attention_layers", &tr::ModelConfig::getNbAttentionLayers, nb::arg("pipeline_parallelism") = 1,
            nb::arg("pipeline_parallelism_rank") = 0)
        .def("num_rnn_layers", &tr::ModelConfig::getNbRnnLayers, nb::arg("pipeline_parallelism") = 1,
            nb::arg("pipeline_parallelism_rank") = 0)
        .def("num_kv_heads", &tr::ModelConfig::getNbKvHeads, nb::arg("layer_idx"))
        .def("set_num_kv_heads", &tr::ModelConfig::setNbKvHeads, nb::arg("num_kv_heads"))
        .def_prop_ro("num_heads", &tr::ModelConfig::getNbHeads)
        .def_prop_ro("hidden_size", &tr::ModelConfig::getHiddenSize)
        .def_prop_ro("size_per_head", &tr::ModelConfig::getSizePerHead)
        .def_prop_ro("data_type", &tr::ModelConfig::getDataType)
        .def_prop_ro("speculative_decoding_mode", &tr::ModelConfig::getSpeculativeDecodingMode)
        .def_prop_rw("head_size", &tr::ModelConfig::getSizePerHead, &tr::ModelConfig::setSizePerHead)
        .def_prop_rw(
            "num_kv_heads_per_layer", &tr::ModelConfig::getNumKvHeadsPerLayer, &tr::ModelConfig::setNumKvHeadsPerLayer)
        .def_prop_rw("use_gpt_attention_plugin",
            nb::overload_cast<>(&tr::ModelConfig::useGptAttentionPlugin, nb::const_),
            nb::overload_cast<bool>(&tr::ModelConfig::useGptAttentionPlugin))
        .def_prop_rw("use_packed_input", nb::overload_cast<>(&tr::ModelConfig::usePackedInput, nb::const_),
            nb::overload_cast<bool>(&tr::ModelConfig::usePackedInput))
        .def_prop_rw("kv_cache_type", nb::overload_cast<>(&tr::ModelConfig::getKVCacheType, nb::const_),
            nb::overload_cast<tr::ModelConfig::KVCacheType>(&tr::ModelConfig::setKVCacheType))
        .def_prop_rw("tokens_per_block", &tr::ModelConfig::getTokensPerBlock, &tr::ModelConfig::setTokensPerBlock)
        .def_prop_rw("quant_mode", &tr::ModelConfig::getQuantMode, &tr::ModelConfig::setQuantMode)
        .def_prop_ro("supports_inflight_batching", &tr::ModelConfig::supportsInflightBatching)
        .def_prop_rw("max_batch_size", &tr::ModelConfig::getMaxBatchSize, &tr::ModelConfig::setMaxBatchSize)
        .def_prop_rw("max_beam_width", &tr::ModelConfig::getMaxBeamWidth, &tr::ModelConfig::setMaxBeamWidth)
        .def_prop_rw("max_input_len", &tr::ModelConfig::getMaxInputLen, &tr::ModelConfig::setMaxInputLen)
        .def_prop_rw("max_seq_len", &tr::ModelConfig::getMaxSequenceLen, &tr::ModelConfig::setMaxSequenceLen)
        .def_prop_rw("max_num_tokens", &tr::ModelConfig::getMaxNumTokens, &tr::ModelConfig::setMaxNumTokens)
        .def_prop_rw("max_prompt_embedding_table_size", &tr::ModelConfig::getMaxPromptEmbeddingTableSize,
            &tr::ModelConfig::setMaxPromptEmbeddingTableSize)
        .def_prop_ro("use_prompt_tuning", &tr::ModelConfig::usePromptTuning)
        .def_prop_ro("use_mrope", &tr::ModelConfig::useMrope)
        .def_prop_rw("use_lora_plugin", nb::overload_cast<>(&tr::ModelConfig::useLoraPlugin, nb::const_),
            nb::overload_cast<bool>(&tr::ModelConfig::useLoraPlugin))
        .def_prop_rw("layer_types", &tr::ModelConfig::getLayerTypes, &tr::ModelConfig::setLayerTypes)
        .def_prop_rw("compute_context_logits", nb::overload_cast<>(&tr::ModelConfig::computeContextLogits, nb::const_),
            nb::overload_cast<bool>(&tr::ModelConfig::computeContextLogits))
        .def_prop_rw("compute_generation_logits",
            nb::overload_cast<>(&tr::ModelConfig::computeGenerationLogits, nb::const_),
            nb::overload_cast<bool>(&tr::ModelConfig::computeGenerationLogits))
        .def_prop_rw("model_variant", &tr::ModelConfig::getModelVariant, &tr::ModelConfig::setModelVariant)
        .def_prop_rw("use_cross_attention", &tr::ModelConfig::useCrossAttention, &tr::ModelConfig::setUseCrossAttention)
        .def_prop_rw("lora_modules", &tr::ModelConfig::getLoraModules, &tr::ModelConfig::setLoraModules)
        .def_prop_rw("max_lora_rank", &tr::ModelConfig::getMaxLoraRank, &tr::ModelConfig::setMaxLoraRank)
        .def_prop_rw("mlp_hidden_size", &tr::ModelConfig::getMlpHiddenSize, &tr::ModelConfig::setMlpHiddenSize)
        .def_prop_rw("size_per_head", &tr::ModelConfig::getSizePerHead, &tr::ModelConfig::setSizePerHead);

    nb::class_<tr::WorldConfig>(m, "WorldConfig")
        .def(nb::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 std::optional<std::vector<SizeType32>> const&, bool>(),
            nb::arg("tensor_parallelism") = 1, nb::arg("pipeline_parallelism") = 1, nb::arg("context_parallelism") = 1,
            nb::arg("rank") = 0, nb::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode,
            nb::arg("device_ids") = nb::none(), nb::arg("enable_attention_dp") = false)
        .def_prop_ro("size", &tr::WorldConfig::getSize)
        .def_prop_ro("tensor_parallelism", &tr::WorldConfig::getTensorParallelism)
        .def_prop_ro("pipeline_parallelism", &tr::WorldConfig::getPipelineParallelism)
        .def_prop_ro("context_parallelism", &tr::WorldConfig::getContextParallelism)
        .def_prop_ro("is_tensor_parallel", &tr::WorldConfig::isTensorParallel)
        .def_prop_ro("is_pipeline_parallel", &tr::WorldConfig::isPipelineParallel)
        .def_prop_ro("is_context_parallel", &tr::WorldConfig::isContextParallel)
        .def_prop_ro("rank", &tr::WorldConfig::getRank)
        .def_prop_ro("local_rank", &tr::WorldConfig::getLocalRank)
        .def_prop_ro("node_rank", &tr::WorldConfig::getNodeRank)
        .def_prop_ro("gpus_per_node", &tr::WorldConfig::getGpusPerNode)
        .def_prop_ro("gpus_per_group", &tr::WorldConfig::getGpusPerGroup)
        .def_prop_ro("device", &tr::WorldConfig::getDevice)
        .def_prop_ro("pipeline_parallel_rank", &tr::WorldConfig::getPipelineParallelRank)
        .def_prop_ro("tensor_parallel_rank", &tr::WorldConfig::getTensorParallelRank)
        .def_prop_ro("context_parallel_rank", &tr::WorldConfig::getContextParallelRank)
        .def_prop_ro("enable_attention_dp", &tr::WorldConfig::enableAttentionDP)
        .def_static("mpi",
            nb::overload_cast<SizeType32, std::optional<SizeType32>, std::optional<SizeType32>,
                std::optional<SizeType32>, std::optional<std::vector<SizeType32>> const&, bool>(&tr::WorldConfig::mpi),
            nb::arg("gpus_per_node") = tr::WorldConfig::kDefaultGpusPerNode, nb::arg("tensor_parallelism") = nb::none(),
            nb::arg("pipeline_parallelism") = nb::none(), nb::arg("context_parallelism") = nb::none(),
            nb::arg("device_ids") = nb::none(), nb::arg("enable_attention_dp") = false);

    auto SamplingConfigGetState = [](tr::SamplingConfig const& config) -> nb::tuple
    {
        return nb::make_tuple(config.beamWidth, config.temperature, config.minLength, config.repetitionPenalty,
            config.presencePenalty, config.frequencyPenalty, config.topK, config.topP, config.randomSeed,
            config.topPDecay, config.topPMin, config.topPResetIds, config.beamSearchDiversityRate, config.lengthPenalty,
            config.earlyStopping, config.noRepeatNgramSize, config.numReturnSequences, config.minP,
            config.beamWidthArray);
    };
    auto SamplingConfigSetState = [](tr::SamplingConfig& self, nb::tuple t)
    {
        if (t.size() != 19)
        {
            throw std::runtime_error("Invalid SamplingConfig state!");
        }

        tr::SamplingConfig config;
        config.beamWidth = nb::cast<SizeType32>(t[0]);
        config.temperature = nb::cast<OptVec<float>>(t[1]);
        config.minLength = nb::cast<OptVec<SizeType32>>(t[2]);
        config.repetitionPenalty = nb::cast<OptVec<float>>(t[3]);
        config.presencePenalty = nb::cast<OptVec<float>>(t[4]);
        config.frequencyPenalty = nb::cast<OptVec<float>>(t[5]);
        config.topK = nb::cast<OptVec<SizeType32>>(t[6]);
        config.topP = nb::cast<OptVec<float>>(t[7]);
        config.randomSeed = nb::cast<OptVec<uint64_t>>(t[8]);
        config.topPDecay = nb::cast<OptVec<float>>(t[9]);
        config.topPMin = nb::cast<OptVec<float>>(t[10]);
        config.topPResetIds = nb::cast<OptVec<TokenIdType>>(t[11]);
        config.beamSearchDiversityRate = nb::cast<OptVec<float>>(t[12]);
        config.lengthPenalty = nb::cast<OptVec<float>>(t[13]);
        config.earlyStopping = nb::cast<OptVec<SizeType32>>(t[14]);
        config.noRepeatNgramSize = nb::cast<OptVec<SizeType32>>(t[15]);
        config.numReturnSequences = nb::cast<SizeType32>(t[16]);
        config.minP = nb::cast<OptVec<float>>(t[17]);
        config.beamWidthArray = nb::cast<OptVec<std::vector<SizeType32>>>(t[18]);

        new (&self) tr::SamplingConfig(config);
    };

    nb::class_<tr::SamplingConfig>(m, "SamplingConfig")
        .def(nb::init<SizeType32>(), nb::arg("beam_width") = 1)
        .def(nb::init<tle::SamplingConfig, std::optional<tle::ExternalDraftTokensConfig>>(),
            nb::arg("executor_sample_config"), nb::arg("external_draft_tokens_config") = std::nullopt)
        .def_rw("beam_width", &tr::SamplingConfig::beamWidth)
        .def_rw("temperature", &tr::SamplingConfig::temperature)
        .def_rw("min_length", &tr::SamplingConfig::minLength)
        .def_rw("repetition_penalty", &tr::SamplingConfig::repetitionPenalty)
        .def_rw("presence_penalty", &tr::SamplingConfig::presencePenalty)
        .def_rw("frequency_penalty", &tr::SamplingConfig::frequencyPenalty)
        .def_rw("top_k", &tr::SamplingConfig::topK)
        .def_rw("top_p", &tr::SamplingConfig::topP)
        .def_rw("random_seed", &tr::SamplingConfig::randomSeed)
        .def_rw("top_p_decay", &tr::SamplingConfig::topPDecay)
        .def_rw("top_p_min", &tr::SamplingConfig::topPMin)
        .def_rw("top_p_reset_ids", &tr::SamplingConfig::topPResetIds)
        .def_rw("beam_search_diversity_rate", &tr::SamplingConfig::beamSearchDiversityRate)
        .def_rw("length_penalty", &tr::SamplingConfig::lengthPenalty)
        .def_rw("early_stopping", &tr::SamplingConfig::earlyStopping)
        .def_rw("no_repeat_ngram_size", &tr::SamplingConfig::noRepeatNgramSize)
        .def_rw("num_return_sequences", &tr::SamplingConfig::numReturnSequences)
        .def_rw("min_p", &tr::SamplingConfig::minP)
        .def_rw("beam_width_array", &tr::SamplingConfig::beamWidthArray)
        .def_rw("normalize_log_probs", &tr::SamplingConfig::normalizeLogProbs)
        .def("__getstate__", SamplingConfigGetState)
        .def("__setstate__", SamplingConfigSetState)
        .def("__eq__", &tr::SamplingConfig::operator==);

    nb::bind_vector<std::vector<tr::SamplingConfig>>(m, "SamplingConfigVector");

    m.def("make_sampling_config", &makeSamplingConfig, nb::arg("configs"));

    nb::class_<tr::GptJsonConfig>(m, "GptJsonConfig")
        .def(nb::init<std::string, std::string, std::string, SizeType32, SizeType32, SizeType32, SizeType32,
                 tr::ModelConfig, std::optional<tr::RuntimeDefaults>>(),
            nb::arg("name"), nb::arg("version"), nb::arg("precision"), nb::arg("tensor_parallelism"),
            nb::arg("pipeline_parallelism"), nb::arg("context_parallelism"), nb::arg("gpus_per_node"),
            nb::arg("model_config"), nb::arg("runtime_defaults") = nb::none())
        .def_static("parse", nb::overload_cast<std::string const&>(&tr::GptJsonConfig::parse), nb::arg("json"))
        .def_static(
            "parse_file", nb::overload_cast<std::filesystem::path const&>(&tr::GptJsonConfig::parse), nb::arg("path"))
        .def_prop_ro("model_config", &tr::GptJsonConfig::getModelConfig)
        .def_prop_ro("name", &tr::GptJsonConfig::getName)
        .def_prop_ro("version", &tr::GptJsonConfig::getVersion)
        .def_prop_ro("precision", &tr::GptJsonConfig::getPrecision)
        .def_prop_ro("tensor_parallelism", &tr::GptJsonConfig::getTensorParallelism)
        .def_prop_ro("pipeline_parallelism", &tr::GptJsonConfig::getPipelineParallelism)
        .def_prop_ro("context_parallelism", &tr::GptJsonConfig::getContextParallelism)
        .def_prop_ro("gpus_per_node", &tr::GptJsonConfig::getGpusPerNode)
        .def_prop_ro("world_size", &tr::GptJsonConfig::getWorldSize)
        .def_prop_ro("runtime_defaults", &tr::GptJsonConfig::getRuntimeDefaults)
        .def("engine_filename",
            nb::overload_cast<tr::WorldConfig const&, std::string const&>(
                &tr::GptJsonConfig::engineFilename, nb::const_),
            nb::arg("world_config"), nb::arg("model"))
        .def("engine_filename",
            nb::overload_cast<tr::WorldConfig const&>(&tr::GptJsonConfig::engineFilename, nb::const_),
            nb::arg("world_config"));

    nb::enum_<tb::LlmRequestState>(m, "LlmRequestState")
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
        .value("DISAGG_GENERATION_TRANS_COMPLETE", tb::LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE)
        .value("DISAGG_CONTEXT_INIT_AND_TRANS", tb::LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS)
        .value("DISAGG_TRANS_ERROR", tb::LlmRequestState::kDISAGG_TRANS_ERROR);

    nb::class_<tr::MemoryCounters>(m, "MemoryCounters")
        .def_static("instance", &tr::MemoryCounters::getInstance, nb::rv_policy::reference)
        .def_prop_ro("gpu", &tr::MemoryCounters::getGpu)
        .def_prop_ro("cpu", &tr::MemoryCounters::getCpu)
        .def_prop_ro("pinned", &tr::MemoryCounters::getPinned)
        .def_prop_ro("uvm", &tr::MemoryCounters::getUVM);

    tensorrt_llm::nanobind::process_group::initBindings(mInternalProcessGroup);
    tensorrt_llm::nanobind::runtime::initBindings(mInternalRuntime);
    tensorrt_llm::nanobind::testing::initBindings(mInternalTesting);
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
    nb::class_<tr::IpcNvlsHandle>(m, "IpcNvlsHandle")
        .def(nb::init<>())
        .def_rw("uc_ptr", &tr::IpcNvlsHandle::uc_ptr)
        .def_rw("mc_ptr", &tr::IpcNvlsHandle::mc_ptr)
        .def_rw("size", &tr::IpcNvlsHandle::size)
        .def("get_ipc_ptrs",
            [](tr::IpcNvlsHandle& self) { return reinterpret_cast<uintptr_t>(self.ipc_uc_ptrs.data()); });

    m.def("ipc_nvls_allocate", &tr::ipcNvlsAllocate, nb::rv_policy::reference);
    m.def("ipc_nvls_free", &tr::ipcNvlsFree);
    m.def("ipc_nvls_supported", &tr::ipcNvlsSupported);

    m.def("steady_clock_now", []() { return std::chrono::steady_clock::now(); });
}
