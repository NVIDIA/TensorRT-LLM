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
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/delayStream.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/request.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

namespace tr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;
using CudaStreamPtr = std::shared_ptr<tr::CudaStream>;

class PyITensor : public tensorrt_llm::runtime::ITensor
{
public:
    /* Inherit the constructors */
    using ITensor::ITensor;

    [[nodiscard]] void* data() override
    {
        PYBIND11_OVERRIDE_PURE(void*, /* Return type */
            ITensor,                  /* Parent class */
            data                      /* Name of function in C++ (must match Python name) */
                                      /* Argument(s) */
        );
    }

    [[nodiscard]] void const* data() const override
    {
        PYBIND11_OVERRIDE_PURE(void const*, /* Return type */
            ITensor,                        /* Parent class */
            data                            /* Name of function in C++ (must match Python name) */
                                            /* Argument(s) */
        );
    }

    [[nodiscard]] std::size_t getSize() const override
    {
        PYBIND11_OVERRIDE_PURE(std::size_t, /* Return type */
            ITensor,                        /* Parent class */
            getSize                         /* Name of function in C++ (must match Python name) */
                                            /* Argument(s) */
        );
    }

    [[nodiscard]] std::size_t getCapacity() const override
    {
        PYBIND11_OVERRIDE_PURE(std::size_t, /* Return type */
            ITensor,                        /* Parent class */
            getCapacity                     /* Name of function in C++ (must match Python name) */
                                            /* Argument(s) */
        );
    }

    [[nodiscard]] DataType getDataType() const override
    {
        PYBIND11_OVERRIDE_PURE(DataType, /* Return type */
            ITensor,                     /* Parent class */
            getDataType                  /* Name of function in C++ (must match Python name) */
                                         /* Argument(s) */
        );
    }

    [[nodiscard]] tr::MemoryType getMemoryType() const override
    {
        PYBIND11_OVERRIDE_PURE(tr::MemoryType, /* Return type */
            ITensor,                           /* Parent class */
            getMemoryType                      /* Name of function in C++ (must match Python name) */
                                               /* Argument(s) */
        );
    }

    [[nodiscard]] char const* getMemoryTypeName() const override
    {
        PYBIND11_OVERRIDE_PURE(char const*, /* Return type */
            ITensor,                        /* Parent class */
            getMemoryTypeName               /* Name of function in C++ (must match Python name) */
                                            /* Argument(s) */
        );
    }

    virtual void resize(std::size_t newSize) override
    {
        PYBIND11_OVERRIDE_PURE(void, /* Return type */
            ITensor,                 /* Parent class */
            resize                   /* Name of function in C++ (must match Python name) */
                                     /* Argument(s) */
        );
    }

    void release() override
    {
        PYBIND11_OVERRIDE_PURE(void, /* Return type */
            ITensor,                 /* Parent class */
            release                  /* Name of function in C++ (must match Python name) */
                                     /* Argument(s) */
        );
    }

    [[nodiscard]] Shape const& getShape() const override
    {
        PYBIND11_OVERRIDE_PURE(Shape const&, /* Return type */
            ITensor,                         /* Parent class */
            getShape                         /* Name of function in C++ (must match Python name) */
                                             /* Argument(s) */
        );
    }

    void reshape(Shape const& dims) override
    {
        PYBIND11_OVERRIDE_PURE(void, /* Return type */
            ITensor,                 /* Parent class */
            reshape,                 /* Name of function in C++ (must match Python name) */
            dims                     /* Argument(s) */
        );
    }
};

class PyIGptDecoder : public tr::IGptDecoder
{
public:
    using tr::IGptDecoder::IGptDecoder; // Inherit constructors

    void setup(tr::SamplingConfig const& samplingConfig, size_t batchSize,
        tr::DecodingInput::TensorConstPtr const& batchSlots,
        std::optional<tr::DecodingOutput> const& output = std::nullopt,
        std::optional<std::vector<tr::decoder_batch::Request> const> const& requests = std::nullopt) override
    {
        PYBIND11_OVERRIDE_PURE(void, IGptDecoder, setup, samplingConfig, batchSize, batchSlots, output, requests);
    }

    void forwardAsync(tr::DecodingOutput& output, tr::DecodingInput const& input) override
    {
        PYBIND11_OVERRIDE_PURE(void, IGptDecoder, forwardAsync, output, input);
    }

    void forwardSync(tr::DecodingOutput& output, tr::DecodingInput const& input) override
    {
        PYBIND11_OVERRIDE_PURE(void, IGptDecoder, forwardSync, output, input);
    }

    tr::SamplingConfig const& getSamplingConfig() override
    {
        PYBIND11_OVERRIDE_PURE(tr::SamplingConfig const&, IGptDecoder, getSamplingConfig);
    }

    void disableLookahead(std::optional<tr::SamplingConfig> const& samplingConfig, tr::SizeType32 batchSize,
        tr::DecodingInput::TensorConstPtr batchSlots) override
    {
        PYBIND11_OVERRIDE_PURE(void, IGptDecoder, disableLookahead, samplingConfig, batchSize, batchSlots);
    }
};

namespace tensorrt_llm::pybind::runtime
{

void initBindings(pybind11::module_& m)
{
    py::classh<tr::ITensor, PyITensor>(m, "ITensor").def(py::init());
    py::class_<tr::LoraCache::TaskLayerModuleConfig>(m, "TaskLayerModuleConfig")
        .def(py::init<>())
        .def_readwrite("pageId", &tr::LoraCache::TaskLayerModuleConfig::pageId)
        .def_readwrite("slotIdx", &tr::LoraCache::TaskLayerModuleConfig::slotIdx)
        .def_readwrite("inSize", &tr::LoraCache::TaskLayerModuleConfig::inSize)
        .def_readwrite("outSize", &tr::LoraCache::TaskLayerModuleConfig::outSize)
        .def_readwrite("moduleId", &tr::LoraCache::TaskLayerModuleConfig::moduleId)
        .def_readwrite("layerId", &tr::LoraCache::TaskLayerModuleConfig::layerId)
        .def_readwrite("adapterSize", &tr::LoraCache::TaskLayerModuleConfig::adapterSize)
        .def_readwrite("numSlots", &tr::LoraCache::TaskLayerModuleConfig::numSlots)
        .def_readwrite("weightsInPointer", &tr::LoraCache::TaskLayerModuleConfig::weightsInPointer)
        .def_readwrite("weightsOutPointer", &tr::LoraCache::TaskLayerModuleConfig::weightsOutPointer)
        .def_readwrite("scalingVecPointer", &tr::LoraCache::TaskLayerModuleConfig::scalingVecPointer)
        .def(py::self == py::self);

    py::classh<tr::BufferManager>(m, "BufferManager")
        .def(py::init<tr::BufferManager::CudaStreamPtr, bool>(), py::arg("stream"), py::arg("trim_pool") = false)
        .def_property_readonly("stream", &tr::BufferManager::getStream);

    py::classh<tr::TllmRuntime>(m, "TllmRuntime")
        .def(py::init(
            [](std::filesystem::path engine_path, float gpu_weights_percent = 1.0f, bool use_shape_inference = true)
            {
                // Using default logger by passing nullptr
                return new tr::TllmRuntime(
                    tr::RawEngine(engine_path), nullptr, gpu_weights_percent, use_shape_inference);
            }))
        .def(py::init(
            [](py::buffer engine_buffer, float gpu_weights_percent = 1.0f, bool use_shape_inference = true)
            {
                py::buffer_info info = engine_buffer.request();
                if (info.ndim != 1)
                    throw std::runtime_error("Expected 1-D array for engine buffer");
                return new tr::TllmRuntime(
                    tr::RawEngine(info.ptr, info.shape[0]), nullptr, gpu_weights_percent, use_shape_inference);
            }))
        .def_property_readonly("num_contexts", &tr::TllmRuntime::getNbContexts)
        .def_property_readonly("num_profiles", &tr::TllmRuntime::getNbProfiles)
        .def("get_opt_profile_id", &tr::TllmRuntime::getOptProfileId, py::arg("num_tokens"), py::arg("split_points"))
        .def("clear_contexts", &tr::TllmRuntime::clearContexts)
        .def("execute_context", &tr::TllmRuntime::executeContext, py::arg("context_id"))
        .def_property_readonly("stream_ptr", &tr::TllmRuntime::getStreamPtr)
        .def_property_readonly("buffer_manager",
            static_cast<tr::BufferManager& (tr::TllmRuntime::*) ()>(&tr::TllmRuntime::getBufferManager))
        .def("set_layer_profiler", &tr::TllmRuntime::setLayerProfiler)
        .def("has_layer_profiler", &tr::TllmRuntime::hasLayerProfiler, py::arg("context_id"))
        .def_property_readonly("layer_profiler_info", &tr::TllmRuntime::getLayerProfileInfo)
        .def("report_to_profiler", &tr::TllmRuntime::reportToProfiler, py::arg("context_id"))
        .def_property_readonly("logits_dtype_from_engine",
            [](tr::TllmRuntime& self) { return self.getEngine().getTensorDataType("logits"); });

    py::class_<tr::decoder_batch::Request>(m, "Request")
        .def(py::init<tr::decoder_batch::Request::TensorConstPtr, tr::SizeType32, std::optional<tr::SizeType32>,
                 std::optional<tr::SizeType32>>(),
            py::arg("ids"), py::arg("input_len"), py::arg("max_new_tokens") = std::nullopt,
            py::arg("end_id") = std::nullopt)
        .def_readwrite("ids", &tr::decoder_batch::Request::ids)
        .def_readwrite("input_len", &tr::decoder_batch::Request::inputLen)
        .def_readwrite("max_new_tokens", &tr::decoder_batch::Request::maxNewTokens)
        .def_readwrite("end_id", &tr::decoder_batch::Request::endId)
        .def_readwrite("draft_logits", &tr::decoder_batch::Request::draftLogits)
        .def_readwrite("embedding_bias", &tr::decoder_batch::Request::embeddingBias)
        .def_readwrite("bad_words_list", &tr::decoder_batch::Request::badWordsList)
        .def_readwrite("stop_words_list", &tr::decoder_batch::Request::stopWordsList)
        .def_readwrite("generated_tokens_per_engine_step", &tr::decoder_batch::Request::generatedTokensPerEngineStep)
        .def_readwrite("medusa_paths", &tr::decoder_batch::Request::medusaPaths)
        .def_readwrite("medusa_tree_ids", &tr::decoder_batch::Request::medusaTreeIds)
        .def_readwrite("lookahead_runtime_config", &tr::decoder_batch::Request::lookaheadRuntimeConfig);

    py::class_<tr::decoder_batch::Input>(m, "DecoderBatchInput")
        .def(py::init<std::vector<std::vector<tr::ITensor::SharedConstPtr>>, tr::SizeType32>(), py::arg("logits"),
            py::arg("max_decoding_engine_tokens"))
        .def(py::init<std::vector<tr::ITensor::SharedConstPtr>>(), py::arg("logits"))
        .def_readwrite("logits", &tr::decoder_batch::Input::logits)
        .def_readwrite("max_decoder_steps", &tr::decoder_batch::Input::maxDecoderSteps)
        .def_readwrite("cache_indirection", &tr::decoder_batch::Input::cacheIndirection)
        .def_readwrite("predicted_draft_logits", &tr::decoder_batch::Input::predictedDraftLogits)
        .def_readwrite("batch_slots", &tr::decoder_batch::Input::batchSlots);

    py::class_<tr::decoder_batch::Output>(m, "DecoderBatchOutput")
        .def(py::init())
        .def_readwrite("cache_indirection", &tr::decoder_batch::Output::cacheIndirection);

    py::class_<tr::LookaheadDecodingBuffers>(m, "LookaheadDecodingBuffers")
        .def(py::init<tr::SizeType32, tr::SizeType32, tr::BufferManager const&>(), py::arg("max_num_sequences"),
            py::arg("max_tokens_per_step"), py::arg("buffer_manager"))
        .def_readwrite("generation_lengths", &tr::LookaheadDecodingBuffers::generationLengths)
        .def_readwrite("position_offsets", &tr::LookaheadDecodingBuffers::positionOffsets)
        .def_readwrite("packed_masks", &tr::LookaheadDecodingBuffers::packedMasks)
        .def_readwrite("position_ids", &tr::LookaheadDecodingBuffers::positionIds);

    py::class_<tr::ExplicitDraftTokensBuffers::Inputs>(m, "ExplicitDraftTokensBuffersInputs")
        .def("create", &tr::ExplicitDraftTokensBuffers::Inputs::create, py::arg("max_num_sequences"),
            py::arg("runtime"), py::arg("model_config"), py::arg("world_config"))
        .def_readwrite("temperatures", &tr::ExplicitDraftTokensBuffers::Inputs::temperatures)
        .def_readwrite("position_ids_base", &tr::ExplicitDraftTokensBuffers::Inputs::positionIdsBase)
        .def_readwrite("generation_lengths", &tr::ExplicitDraftTokensBuffers::Inputs::generationLengths)
        .def_readwrite("random_data_sample", &tr::ExplicitDraftTokensBuffers::Inputs::randomDataSample)
        .def_readwrite("random_data_validation", &tr::ExplicitDraftTokensBuffers::Inputs::randomDataValidation)
        .def_readwrite("draft_tokens", &tr::ExplicitDraftTokensBuffers::Inputs::draftTokens)
        .def_readwrite("draft_indices", &tr::ExplicitDraftTokensBuffers::Inputs::draftIndices)
        .def_readwrite("draft_probs", &tr::ExplicitDraftTokensBuffers::Inputs::draftProbs)
        .def_readwrite("packed_masks", &tr::ExplicitDraftTokensBuffers::Inputs::packedMasks)
        .def_readwrite("position_ids", &tr::ExplicitDraftTokensBuffers::Inputs::positionIds)
        .def_readwrite("max_gen_length_host", &tr::ExplicitDraftTokensBuffers::Inputs::maxGenLengthHost)
        .def_readwrite("generation_lengths_host", &tr::ExplicitDraftTokensBuffers::Inputs::generationLengthsHost);

    py::class_<tr::DecodingInput>(m, "DecodingInput");
    py::class_<tr::DecodingOutput>(m, "DecodingOutput");

    py::class_<tr::CudaEvent>(m, "CudaEvent")
        .def(py::init<unsigned int>(), py::arg("flags") = cudaEventDisableTiming)
        .def("synchronize", &tr::CudaEvent::synchronize);

    py::class_<tr::IGptDecoder, PyIGptDecoder>(m, "IGptDecoder")
        .def(
            "setup",
            [](tr::IGptDecoder& self, tr::SamplingConfig const& samplingConfig, size_t batchSize,
                at::Tensor const& batchSlots, std::optional<tr::DecodingOutput> const& output = std::nullopt,
                std::optional<std::vector<tr::decoder_batch::Request> const> const& requests = std::nullopt)
            {
                auto tensorPtrBatchSlots = tr::TorchView::of(batchSlots);
                return self.setup(samplingConfig, batchSize, std::move(tensorPtrBatchSlots), output, requests);
            },
            py::arg("sampling_config"), py::arg("batch_size"), py::arg("batch_slots"), py::arg("output") = std::nullopt,
            py::arg("requests") = std::nullopt);

    py::class_<tr::decoder::DecoderState>(m, "DecoderState")
        .def(py::init<nvinfer1::DataType, tr::BufferManager const&>(), py::arg("dtype"), py::arg("buffer_manager"))
        .def("setup", &tr::decoder::DecoderState::setup, py::arg("max_batch_size"), py::arg("max_beam_width"),
            py::arg("max_attention_window"), py::arg("sink_token_length"), py::arg("max_sequence_length"),
            py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"))
        .def_property_readonly("joint_decoding_input", &tr::decoder::DecoderState::getJointDecodingInput)
        .def_property_readonly("joint_decoding_output", &tr::decoder::DecoderState::getJointDecodingOutput)
        .def_property_readonly("sequence_lengths",
            [](tr::decoder::DecoderState& self) { return tr::Torch::tensor(self.getSequenceLengths()); })
        .def_property_readonly(
            "all_new_tokens", [](tr::decoder::DecoderState& self) { return tr::Torch::tensor(self.getAllNewTokens()); })
        .def_property_readonly(
            "finished_sum", [](tr::decoder::DecoderState& self) { return tr::Torch::tensor(self.getFinishedSum()); })
        .def_property_readonly("finish_reasons",
            [](tr::decoder::DecoderState& self) { return tr::Torch::tensor(self.getFinishReasons()); });

    py::class_<tr::GptDecoderBatched>(m, "GptDecoderBatched")
        .def(py::init<tr::GptDecoderBatched::CudaStreamPtr, tr::SpeculativeDecodingMode const&, nvinfer1::DataType>(),
            py::arg("stream"), py::arg("speculative_decoding_mode"), py::arg("dtype"))
        .def("setup", &tr::GptDecoderBatched::setup, py::arg("mode"), py::arg("max_batch_size"),
            py::arg("max_beam_width"), py::arg("max_attention_window"), py::arg("sink_token_length"),
            py::arg("max_sequence_length"), py::arg("max_tokens_per_step"), py::arg("dtype"), py::arg("model_config"),
            py::arg("world_config"))
        .def("forward_async", &tr::GptDecoderBatched::forwardAsync, py::arg("output"), py::arg("input"))
        .def("underlying_decoder", &tr::GptDecoderBatched::getUnderlyingDecoder, py::return_value_policy::reference)
        .def_property_readonly("stream_ptr", &tr::GptDecoderBatched::getDecoderStream)
        .def_property_readonly(
            "decoder_state", py::overload_cast<>(&tr::GptDecoderBatched::getDecoderState, py::const_));

    m.def(
        "lamport_initialize_all",
        [](intptr_t buffer_0, intptr_t buffer_1, intptr_t buffer_2, size_t size)
        {
            tr::lamportInitializeAll(reinterpret_cast<void*>(buffer_0), reinterpret_cast<void*>(buffer_1),
                reinterpret_cast<void*>(buffer_2), size);
        },
        "Lamport initialize all buffers");
    m.def(
        "lamport_initialize",
        [](intptr_t buffer, size_t size)
        { tensorrt_llm::kernels::ar_fusion::lamport_initialize(reinterpret_cast<void*>(buffer), size, 0); },
        "Lmaport initialize buffer");
    m.def(
        "delay_kernel",
        [](int64_t delay_micro_secs, py::object py_stream)
        {
            // Get the raw stream handle from PyTorch stream object
            auto stream_ptr = py_stream.attr("cuda_stream").cast<int64_t>();
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            tensorrt_llm::kernels::invokeDelayStreamKernel(delay_micro_secs, stream);
        },
        "Delay kernel launch on the default stream");

    py::enum_<tensorrt_llm::kernels::AllReduceFusionOp>(m, "AllReduceFusionOp")
        .value("NONE", tensorrt_llm::kernels::AllReduceFusionOp::NONE)
        .value("RESIDUAL_RMS_NORM", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM)
        .value("LAST_PROCESS_FOR_UB", tensorrt_llm::kernels::AllReduceFusionOp::LAST_PROCESS_FOR_UB)
        .value("RESIDUAL_RMS_PREPOST_NORM", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
        .value("RESIDUAL_RMS_NORM_QUANT_FP8", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8)
        .value("RESIDUAL_RMS_NORM_QUANT_NVFP4", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4)
        .value("RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4",
            tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4)
        .value("RESIDUAL_RMS_NORM_OUT_QUANT_FP8",
            tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8);

    py::enum_<tensorrt_llm::kernels::AllReduceStrategyType>(m, "AllReduceStrategy")
        .value("NCCL", tensorrt_llm::kernels::AllReduceStrategyType::NCCL)
        .value("MIN_LATENCY", tensorrt_llm::kernels::AllReduceStrategyType::MIN_LATENCY)
        .value("AUTO", tensorrt_llm::kernels::AllReduceStrategyType::AUTO)
        .value("UB", tensorrt_llm::kernels::AllReduceStrategyType::UB)
        .value("ONESHOT", tensorrt_llm::kernels::AllReduceStrategyType::ONESHOT)
        .value("TWOSHOT", tensorrt_llm::kernels::AllReduceStrategyType::TWOSHOT);
}

void initBindingsEarly(py::module_& m)
{
    py::class_<tr::SpeculativeDecodingMode>(m, "SpeculativeDecodingMode")
        .def(py::init<tr::SpeculativeDecodingMode::UnderlyingType>(), py::arg("state"))
        .def_static("NoneType", &tr::SpeculativeDecodingMode::None)
        .def_static("DraftTokensExternal", &tr::SpeculativeDecodingMode::DraftTokensExternal)
        .def_static("Medusa", &tr::SpeculativeDecodingMode::Medusa)
        .def_static("Eagle", &tr::SpeculativeDecodingMode::Eagle)
        .def_static("LookaheadDecoding", &tr::SpeculativeDecodingMode::LookaheadDecoding)
        .def_static("ExplicitDraftTokens", &tr::SpeculativeDecodingMode::ExplicitDraftTokens)
        .def_property_readonly("is_none", &tr::SpeculativeDecodingMode::isNone)
        .def_property_readonly("is_draft_tokens_external", &tr::SpeculativeDecodingMode::isDraftTokensExternal)
        .def_property_readonly("is_medusa", &tr::SpeculativeDecodingMode::isMedusa)
        .def_property_readonly("is_eagle", &tr::SpeculativeDecodingMode::isEagle)
        .def_property_readonly("is_lookahead_decoding", &tr::SpeculativeDecodingMode::isLookaheadDecoding)
        .def_property_readonly("is_explicit_draft_tokens", &tr::SpeculativeDecodingMode::isExplicitDraftTokens)
        .def_property_readonly("needs_kv_cache_rewind", &tr::SpeculativeDecodingMode::needsKVCacheRewind)
        .def_property_readonly("needs_decoder_prologue", &tr::SpeculativeDecodingMode::needsDecoderPrologue)
        .def_property_readonly("predicts_draft_tokens", &tr::SpeculativeDecodingMode::predictsDraftTokens)
        .def_property_readonly("needs_kv_cache_rewind", &tr::SpeculativeDecodingMode::needsKVCacheRewind);
}
} // namespace tensorrt_llm::pybind::runtime
