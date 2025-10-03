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

#include "bindings.h"
#include "hostfunc.h"
#include "moeBindings.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.h"
#include "tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/delayStream.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/mcastGPUBuffer.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/virtualMemory.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

namespace tr = tensorrt_llm::runtime;
namespace te = tensorrt_llm::executor;

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
        std::optional<nvinfer1::DataType> explicitDraftTokensDType = std::nullopt,
        std::optional<std::vector<tr::ITensor::SharedConstPtr>> const& lookaheadPrompt = std::nullopt,
        std::optional<std::vector<te::LookaheadDecodingConfig>> const& lookaheadAlgoConfigs = std::nullopt) override
    {
        PYBIND11_OVERRIDE_PURE(void, IGptDecoder, setup, samplingConfig, batchSize, batchSlots, output,
            explicitDraftTokensDType, lookaheadPrompt, lookaheadAlgoConfigs);
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
        .def_readwrite("page_id", &tr::LoraCache::TaskLayerModuleConfig::pageId)
        .def_readwrite("slot_idx", &tr::LoraCache::TaskLayerModuleConfig::slotIdx)
        .def_readwrite("in_size", &tr::LoraCache::TaskLayerModuleConfig::inSize)
        .def_readwrite("out_size", &tr::LoraCache::TaskLayerModuleConfig::outSize)
        .def_readwrite("module_id", &tr::LoraCache::TaskLayerModuleConfig::moduleId)
        .def_readwrite("layer_id", &tr::LoraCache::TaskLayerModuleConfig::layerId)
        .def_readwrite("adapter_size", &tr::LoraCache::TaskLayerModuleConfig::adapterSize)
        .def_readwrite("num_slots", &tr::LoraCache::TaskLayerModuleConfig::numSlots)
        .def_readwrite("weights_in_pointer", &tr::LoraCache::TaskLayerModuleConfig::weightsInPointer)
        .def_readwrite("weights_out_pointer", &tr::LoraCache::TaskLayerModuleConfig::weightsOutPointer)
        .def_readwrite("scaling_vec_pointer", &tr::LoraCache::TaskLayerModuleConfig::scalingVecPointer)
        .def(py::self == py::self);

    py::class_<tr::CudaVirtualMemoryManager>(m, "CudaVirtualMemoryManager")
        .def("release_with_tag", &tr::CudaVirtualMemoryManager::releaseWithTag, py::arg("tag"),
            py::call_guard<py::gil_scoped_release>())
        .def("materialize_with_tag", &tr::CudaVirtualMemoryManager::materializeWithTag, py::arg("tag"),
            py::call_guard<py::gil_scoped_release>());

    py::classh<tr::BufferManager>(m, "BufferManager")
        .def(py::init<tr::BufferManager::CudaStreamPtr, bool>(), py::arg("stream"), py::arg("trim_pool") = false,
            py::call_guard<py::gil_scoped_release>())
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
        .def("get_opt_profile_id", &tr::TllmRuntime::getOptProfileId, py::arg("num_tokens"), py::arg("split_points"),
            py::call_guard<py::gil_scoped_release>())
        .def("clear_contexts", &tr::TllmRuntime::clearContexts, py::call_guard<py::gil_scoped_release>())
        .def("execute_context", &tr::TllmRuntime::executeContext, py::arg("context_id"),
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("stream_ptr", &tr::TllmRuntime::getStreamPtr)
        .def_property_readonly("buffer_manager",
            static_cast<tr::BufferManager& (tr::TllmRuntime::*) ()>(&tr::TllmRuntime::getBufferManager))
        .def("set_layer_profiler", &tr::TllmRuntime::setLayerProfiler, py::call_guard<py::gil_scoped_release>())
        .def("has_layer_profiler", &tr::TllmRuntime::hasLayerProfiler, py::arg("context_id"),
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("layer_profiler_info", &tr::TllmRuntime::getLayerProfileInfo)
        .def("report_to_profiler", &tr::TllmRuntime::reportToProfiler, py::arg("context_id"),
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("logits_dtype_from_engine",
            [](tr::TllmRuntime& self) { return self.getEngine().getTensorDataType("logits"); });

    py::class_<tr::decoder_batch::Input>(m, "DecoderBatchInput")
        .def(py::init<std::vector<std::vector<tr::ITensor::SharedConstPtr>>, tr::SizeType32>(), py::arg("logits"),
            py::arg("max_decoding_engine_tokens"), py::call_guard<py::gil_scoped_release>())
        .def(py::init<std::vector<tr::ITensor::SharedConstPtr>>(), py::arg("logits"),
            py::call_guard<py::gil_scoped_release>())
        .def_readwrite("logits", &tr::decoder_batch::Input::logits)
        .def_readwrite("max_decoder_steps", &tr::decoder_batch::Input::maxDecoderSteps)
        .def_readwrite("batch_slots", &tr::decoder_batch::Input::batchSlots);

    py::class_<tr::LookaheadDecodingBuffers>(m, "LookaheadDecodingBuffers")
        .def(py::init<tr::SizeType32, tr::SizeType32, tr::BufferManager const&>(), py::arg("max_num_sequences"),
            py::arg("max_tokens_per_step"), py::arg("buffer_manager"), py::call_guard<py::gil_scoped_release>())
        .def_readwrite("generation_lengths", &tr::LookaheadDecodingBuffers::generationLengths)
        .def_readwrite("position_offsets", &tr::LookaheadDecodingBuffers::positionOffsets)
        .def_readwrite("packed_masks", &tr::LookaheadDecodingBuffers::packedMasks)
        .def_readwrite("position_ids", &tr::LookaheadDecodingBuffers::positionIds);

    py::class_<tr::ExplicitDraftTokensBuffers::Inputs>(m, "ExplicitDraftTokensBuffersInputs")
        .def("create", &tr::ExplicitDraftTokensBuffers::Inputs::create, py::arg("max_num_sequences"),
            py::arg("runtime"), py::arg("model_config"), py::arg("world_config"),
            py::call_guard<py::gil_scoped_release>())
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
        .def(py::init<unsigned int>(), py::arg("flags") = cudaEventDisableTiming,
            py::call_guard<py::gil_scoped_release>())
        .def("synchronize", &tr::CudaEvent::synchronize, py::call_guard<py::gil_scoped_release>());

    py::class_<tr::IGptDecoder, PyIGptDecoder>(m, "IGptDecoder")
        .def(
            "setup",
            [](tr::IGptDecoder& self, tr::SamplingConfig const& samplingConfig, size_t batchSize,
                at::Tensor const& batchSlots, std::optional<tr::DecodingOutput> const& output = std::nullopt,
                std::optional<nvinfer1::DataType> explicitDraftTokensDType = std::nullopt,
                std::optional<std::vector<tr::ITensor::SharedConstPtr>> const& lookaheadPrompt = std::nullopt,
                std::optional<std::vector<te::LookaheadDecodingConfig>> const& lookaheadAlgoConfigs = std::nullopt)
            {
                auto tensorPtrBatchSlots = tr::TorchView::of(batchSlots);
                self.setup(samplingConfig, batchSize, std::move(tensorPtrBatchSlots), output, explicitDraftTokensDType,
                    lookaheadPrompt, lookaheadAlgoConfigs);
            },
            py::arg("sampling_config"), py::arg("batch_size"), py::arg("batch_slots"), py::arg("output") = std::nullopt,
            py::arg("explicit_draft_tokens_d_type") = std::nullopt, py::arg("lookahead_prompt") = std::nullopt,
            py::arg("lookahead_algo_configs") = std::nullopt, py::call_guard<py::gil_scoped_release>());

    py::class_<tr::decoder::DecoderState>(m, "DecoderState")
        .def(py::init<>(), py::call_guard<py::gil_scoped_release>())
        .def("setup", &tr::decoder::DecoderState::setup, py::arg("max_num_sequences"), py::arg("max_beam_width"),
            py::arg("max_attention_window"), py::arg("sink_token_length"), py::arg("max_sequence_length"),
            py::arg("dtype"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"),
            py::call_guard<py::gil_scoped_release>())
        .def("setup_cache_indirection", &tr::decoder::DecoderState::setupCacheIndirection, py::arg("max_num_sequences"),
            py::arg("max_beam_width"), py::arg("max_attention_window"), py::arg("buffer_manager"),
            py::call_guard<py::gil_scoped_release>())
        .def("setup_speculative_decoding", &tr::decoder::DecoderState::setupSpeculativeDecoding,
            py::arg("speculative_decoding_mode"), py::arg("max_tokens_per_engine_step"), py::arg("dtype"),
            py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"),
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("joint_decoding_input", &tr::decoder::DecoderState::getJointDecodingInput)
        .def_property_readonly("joint_decoding_output", &tr::decoder::DecoderState::getJointDecodingOutput)
        .def_property_readonly("cache_indirection_input", &tr::decoder::DecoderState::getCacheIndirectionInput)
        .def_property_readonly("cache_indirection_output", &tr::decoder::DecoderState::getCacheIndirectionOutput)
        .def_property_readonly(
            "sequence_lengths", py::overload_cast<>(&tr::decoder::DecoderState::getSequenceLengths, py::const_))
        .def("get_sequence_lengths",
            py::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getSequenceLengths, py::const_),
            py::arg("batch_idx"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("all_new_tokens", &tr::decoder::DecoderState::getAllNewTokens)
        .def_property_readonly("finished_sum", &tr::decoder::DecoderState::getFinishedSum)
        .def_property_readonly("finish_reasons", &tr::decoder::DecoderState::getFinishReasons)
        .def_property_readonly("ids", py::overload_cast<>(&tr::decoder::DecoderState::getIds, py::const_))
        .def("get_ids", py::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getIds, py::const_),
            py::arg("batch_idx"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly(
            "gathered_ids", py::overload_cast<>(&tr::decoder::DecoderState::getGatheredIds, py::const_))
        .def("get_gathered_ids",
            py::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getGatheredIds, py::const_),
            py::arg("batch_idx"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("parent_ids", &tr::decoder::DecoderState::getParentIds)
        .def_property_readonly(
            "cum_log_probs", py::overload_cast<>(&tr::decoder::DecoderState::getCumLogProbs, py::const_))
        .def("get_cum_log_probs",
            py::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getCumLogProbs, py::const_),
            py::arg("batch_idx"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("log_probs", py::overload_cast<>(&tr::decoder::DecoderState::getLogProbs, py::const_))
        .def("get_log_probs", py::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getLogProbs, py::const_),
            py::arg("batch_idx"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("next_draft_tokens", &tr::decoder::DecoderState::getNextDraftTokens)
        .def_property_readonly("prev_draft_tokens_lengths", &tr::decoder::DecoderState::getPrevDraftTokensLengths)
        .def_property_readonly("next_draft_tokens_lengths", &tr::decoder::DecoderState::getNextDraftTokensLengths)
        .def_property_readonly("accepted_lengths_cum_sum", &tr::decoder::DecoderState::getAcceptedLengthsCumSum)
        .def_property_readonly("accepted_packed_paths", &tr::decoder::DecoderState::getAcceptedPackedPaths)
        .def_property_readonly("max_beam_width", &tr::decoder::DecoderState::getMaxBeamWidth)
        .def_property_readonly("max_sequence_length", &tr::decoder::DecoderState::getMaxSequenceLength)
        .def_property_readonly("max_decoding_decoder_tokens", &tr::decoder::DecoderState::getMaxDecodingDecoderTokens)
        .def_property_readonly("max_decoding_engine_tokens", &tr::decoder::DecoderState::getMaxDecodingEngineTokens)
        .def_property_readonly("num_decoding_engine_tokens",
            py::overload_cast<>(&tr::decoder::DecoderState::getNumDecodingEngineTokens, py::const_))
        .def("get_num_decoding_engine_tokens",
            py::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getNumDecodingEngineTokens, py::const_),
            py::arg("batch_idx"), py::call_guard<py::gil_scoped_release>())
        .def("set_num_decoding_engine_tokens", &tr::decoder::DecoderState::setNumDecodingEngineTokens,
            py::arg("batch_idx"), py::arg("num_tokens"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("speculative_decoding_mode", &tr::decoder::DecoderState::getSpeculativeDecodingMode)
        .def_property("generation_steps", &tr::decoder::DecoderState::getGenerationSteps,
            &tr::decoder::DecoderState::setGenerationSteps);

    py::class_<tr::GptDecoderBatched>(m, "GptDecoderBatched")
        .def(py::init<tr::GptDecoderBatched::CudaStreamPtr>(), py::arg("stream"),
            py::call_guard<py::gil_scoped_release>())
        .def("setup", &tr::GptDecoderBatched::setup, py::arg("mode"), py::arg("max_num_sequences"),
            py::arg("max_beam_width"), py::arg("dtype"), py::arg("model_config"), py::arg("world_config"),
            py::call_guard<py::gil_scoped_release>())
        .def("forward_async", &tr::GptDecoderBatched::forwardAsync, py::arg("decoder_state"), py::arg("input"),
            py::call_guard<py::gil_scoped_release>())
        .def("underlying_decoder", &tr::GptDecoderBatched::getUnderlyingDecoder, py::return_value_policy::reference)
        .def("finalize", &tr::GptDecoderBatched::finalize, py::arg("decoder_state"), py::arg("batch_idx"),
            py::arg("sampling_config"), py::arg("streaming"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly(
            "decoder_stream",
            [](tr::GptDecoderBatched& self) -> tr::CudaStream const& { return *self.getDecoderStream(); },
            py::return_value_policy::reference);

    m.def(
        "lamport_initialize_all",
        [](intptr_t buffer_0, intptr_t buffer_1, intptr_t buffer_2, size_t size)
        {
            tr::lamportInitializeAll(reinterpret_cast<void*>(buffer_0), reinterpret_cast<void*>(buffer_1),
                reinterpret_cast<void*>(buffer_2), size);
        },
        "Lamport initialize all buffers", py::call_guard<py::gil_scoped_release>());
    m.def(
        "lamport_initialize",
        [](intptr_t buffer, size_t size)
        { tensorrt_llm::kernels::ar_fusion::lamport_initialize(reinterpret_cast<void*>(buffer), size, 0); },
        "Lmaport initialize buffer", py::call_guard<py::gil_scoped_release>());
    m.def(
        "delay_kernel",
        [](int64_t delay_micro_secs, py::object py_stream)
        {
            // Get the raw stream handle from PyTorch stream object
            auto stream_ptr = py_stream.attr("cuda_stream").cast<int64_t>();
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            py::gil_scoped_release release;
            tensorrt_llm::kernels::invokeDelayStreamKernel(delay_micro_secs, stream);
        },
        "Delay kernel launch on the default stream");
    m.def(
        "max_workspace_size_lowprecision",
        [](int32_t tp_size) { return tensorrt_llm::kernels::max_workspace_size_lowprecision(tp_size); },
        "Calculate the maximum workspace size needed for low precision all-reduce operations",
        py::call_guard<py::gil_scoped_release>());

    py::enum_<tr::CudaVirtualMemoryAllocator::RestoreMode>(m, "CudaVirtualMemoryAllocatorRestoreMode")
        .value("NONE", tr::CudaVirtualMemoryAllocator::RestoreMode::NONE)
        .value("CPU", tr::CudaVirtualMemoryAllocator::RestoreMode::CPU)
        .value("PINNED", tr::CudaVirtualMemoryAllocator::RestoreMode::PINNED)
        .value("MEMSET", tr::CudaVirtualMemoryAllocator::RestoreMode::MEMSET);

    m.def("get_virtual_memory_manager", &tr::getVirtualMemoryManager, "Get the virtual memory manager",
        py::return_value_policy::reference);

    m.def(
        "set_virtual_memory_allocator",
        [](std::string const& tag, tr::CudaVirtualMemoryAllocator::RestoreMode mode, uintptr_t stream)
        {
            static_assert(sizeof(uintptr_t) == sizeof(cudaStream_t));
            tr::setVirtualMemoryAllocator(tag, mode,
                std::make_shared<tr::CudaStream>(
                    reinterpret_cast<cudaStream_t>(stream), tensorrt_llm::common::getDevice(), false));
        },
        "Set the virtual memory allocator and start allocating virtual memory for CUDA allocations",
        py::call_guard<py::gil_scoped_release>());

    m.def("clear_virtual_memory_allocator", &tr::clearVirtualMemoryAllocator,
        "Reset the current virtual memory allocator and stop allocating virtual memory for CUDA allocations",
        py::call_guard<py::gil_scoped_release>());

    py::class_<tensorrt_llm::runtime::McastGPUBuffer>(m, "McastGPUBuffer")
        .def(py::init<size_t, uint32_t, uint32_t, uint32_t, uint32_t, bool>(), py::arg("buf_size"),
            py::arg("group_size"), py::arg("group_rank"), py::arg("split_color"), py::arg("device_idx"),
            py::arg("mn_nvlink"), py::call_guard<py::gil_scoped_release>())
        .def("get_uc_buffer", &tensorrt_llm::runtime::McastGPUBuffer::getUCBuffer,
            py::call_guard<py::gil_scoped_release>())
        .def("get_mc_buffer", &tensorrt_llm::runtime::McastGPUBuffer::getMCBuffer,
            py::call_guard<py::gil_scoped_release>());

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
        .value("TWOSHOT", tensorrt_llm::kernels::AllReduceStrategyType::TWOSHOT)
        .value("NCCL_SYMMETRIC", tensorrt_llm::kernels::AllReduceStrategyType::NCCL_SYMMETRIC);

    // Initialize MoeLoadBalancer bindings
    initMoeBindings(m);
    // Initialize HostFunc bindings
    initHostFuncBindings(m);
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
        .def_property_readonly("updates_position_ids", &tr::SpeculativeDecodingMode::updatesPositionIds)
        .def_property_readonly("requires_attention_mask", &tr::SpeculativeDecodingMode::requiresAttentionMask)
        .def_property_readonly("predicts_draft_tokens", &tr::SpeculativeDecodingMode::predictsDraftTokens)
        .def_property_readonly("needs_kv_cache_rewind", &tr::SpeculativeDecodingMode::needsKVCacheRewind)
        .def_property_readonly("variable_draft_length", &tr::SpeculativeDecodingMode::variableDraftLength)
        .def_property_readonly("has_draft_logits", &tr::SpeculativeDecodingMode::hasDraftLogits)
        .def_property_readonly("needs_decoder_prologue", &tr::SpeculativeDecodingMode::needsDecoderPrologue);
}
} // namespace tensorrt_llm::pybind::runtime
