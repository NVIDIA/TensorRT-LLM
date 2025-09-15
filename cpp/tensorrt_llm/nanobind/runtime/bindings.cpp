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
#include "tensorrt_llm/nanobind/common/customCasters.h"
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
#include <nanobind/stl/vector.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/trampoline.h>
#include <torch/extension.h>
namespace tr = tensorrt_llm::runtime;
namespace te = tensorrt_llm::executor;

class PyIGptDecoder : public tr::IGptDecoder
{
public:
    NB_TRAMPOLINE(tr::IGptDecoder, 5);

    void setup(tr::SamplingConfig const& samplingConfig, size_t batchSize,
        tr::DecodingInput::TensorConstPtr const& batchSlots,
        std::optional<tr::DecodingOutput> const& output = std::nullopt,
        std::optional<nvinfer1::DataType> explicitDraftTokensDType = std::nullopt,
        std::optional<std::vector<tr::ITensor::SharedConstPtr>> const& lookaheadPrompt = std::nullopt,
        std::optional<std::vector<te::LookaheadDecodingConfig>> const& lookaheadAlgoConfigs = std::nullopt) override
    {
        NB_OVERRIDE_PURE(setup, samplingConfig, batchSize, batchSlots, output, explicitDraftTokensDType,
            lookaheadPrompt, lookaheadAlgoConfigs);
    }

    void forwardAsync(tr::DecodingOutput& output, tr::DecodingInput const& input) override
    {
        NB_OVERRIDE_PURE(forwardAsync, output, input);
    }

    void forwardSync(tr::DecodingOutput& output, tr::DecodingInput const& input) override
    {
        NB_OVERRIDE_PURE(forwardSync, output, input);
    }

    tr::SamplingConfig const& getSamplingConfig() override
    {
        NB_OVERRIDE_PURE(getSamplingConfig);
    }

    void disableLookahead(std::optional<tr::SamplingConfig> const& samplingConfig, tr::SizeType32 batchSize,
        tr::DecodingInput::TensorConstPtr batchSlots) override
    {
        NB_OVERRIDE_PURE(disableLookahead, samplingConfig, batchSize, batchSlots);
    }
};

namespace tensorrt_llm::nanobind::runtime
{

void initBindings(nb::module_& m)
{

    nb::class_<tr::LoraCache::TaskLayerModuleConfig>(m, "TaskLayerModuleConfig")
        .def(nb::init<>())
        .def_rw("page_id", &tr::LoraCache::TaskLayerModuleConfig::pageId)
        .def_rw("slot_idx", &tr::LoraCache::TaskLayerModuleConfig::slotIdx)
        .def_rw("in_size", &tr::LoraCache::TaskLayerModuleConfig::inSize)
        .def_rw("out_size", &tr::LoraCache::TaskLayerModuleConfig::outSize)
        .def_rw("module_id", &tr::LoraCache::TaskLayerModuleConfig::moduleId)
        .def_rw("layer_id", &tr::LoraCache::TaskLayerModuleConfig::layerId)
        .def_rw("adapter_size", &tr::LoraCache::TaskLayerModuleConfig::adapterSize)
        .def_rw("num_slots", &tr::LoraCache::TaskLayerModuleConfig::numSlots)
        .def_rw("weights_in_pointer", &tr::LoraCache::TaskLayerModuleConfig::weightsInPointer)
        .def_rw("weights_out_pointer", &tr::LoraCache::TaskLayerModuleConfig::weightsOutPointer)
        .def_rw("scaling_vec_pointer", &tr::LoraCache::TaskLayerModuleConfig::scalingVecPointer)
        .def(nb::self == nb::self);

    nb::class_<tr::CudaVirtualMemoryManager>(m, "CudaVirtualMemoryManager")
        .def("release_with_tag", &tr::CudaVirtualMemoryManager::releaseWithTag, nb::arg("tag"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("materialize_with_tag", &tr::CudaVirtualMemoryManager::materializeWithTag, nb::arg("tag"),
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tr::BufferManager>(m, "BufferManager")
        .def(nb::init<tr::BufferManager::CudaStreamPtr, bool>(), nb::arg("stream"), nb::arg("trim_pool") = false,
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("stream", &tr::BufferManager::getStream);

    nb::class_<tr::TllmRuntime>(m, "TllmRuntime")
        .def(
            "__init__",
            [](tr::TllmRuntime* self, std::filesystem::path engine_path, float gpu_weights_percent = 1.0f,
                bool use_shape_inference = true)
            {
                // Using default logger by passing nullptr
                new (self)
                    tr::TllmRuntime(tr::RawEngine(engine_path), nullptr, gpu_weights_percent, use_shape_inference);
            },
            nb::arg("engine_path"), nb::arg("gpu_weights_percent") = 1.0f, nb::arg("use_shape_inference") = true)
        .def(
            "__init__",
            [](tr::TllmRuntime* self, nb::ndarray<nb::numpy, uint8_t> engine_buffer, float gpu_weights_percent = 1.0f,
                bool use_shape_inference = true)
            {
                if (engine_buffer.ndim() != 1)
                    throw std::runtime_error("Expected 1-D array for engine buffer");
                new (self) tr::TllmRuntime(tr::RawEngine(engine_buffer.data(), engine_buffer.size()), nullptr,
                    gpu_weights_percent, use_shape_inference);
            },
            nb::arg("engine_buffer"), nb::arg("gpu_weights_percent") = 1.0f, nb::arg("use_shape_inference") = true)
        .def_prop_ro("num_contexts", &tr::TllmRuntime::getNbContexts)
        .def_prop_ro("num_profiles", &tr::TllmRuntime::getNbProfiles)
        .def("get_opt_profile_id", &tr::TllmRuntime::getOptProfileId, nb::arg("num_tokens"), nb::arg("split_points"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("clear_contexts", &tr::TllmRuntime::clearContexts, nb::call_guard<nb::gil_scoped_release>())
        .def("execute_context", &tr::TllmRuntime::executeContext, nb::arg("context_id"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("stream_ptr", &tr::TllmRuntime::getStreamPtr)
        .def_prop_ro("buffer_manager",
            static_cast<tr::BufferManager& (tr::TllmRuntime::*) ()>(&tr::TllmRuntime::getBufferManager))
        .def("set_layer_profiler", &tr::TllmRuntime::setLayerProfiler, nb::call_guard<nb::gil_scoped_release>())
        .def("has_layer_profiler", &tr::TllmRuntime::hasLayerProfiler, nb::arg("context_id"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("layer_profiler_info", &tr::TllmRuntime::getLayerProfileInfo)
        .def("report_to_profiler", &tr::TllmRuntime::reportToProfiler, nb::arg("context_id"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("logits_dtype_from_engine",
            [](tr::TllmRuntime& self) { return self.getEngine().getTensorDataType("logits"); });

    nb::class_<tr::decoder_batch::Input>(m, "DecoderBatchInput")
        .def(nb::init<std::vector<std::vector<tr::ITensor::SharedConstPtr>>, tr::SizeType32>(), nb::arg("logits"),
            nb::arg("max_decoding_engine_tokens"), nb::call_guard<nb::gil_scoped_release>())
        .def(nb::init<std::vector<tr::ITensor::SharedConstPtr>>(), nb::arg("logits"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_rw("logits", &tr::decoder_batch::Input::logits)
        .def_rw("max_decoder_steps", &tr::decoder_batch::Input::maxDecoderSteps)
        .def_rw("batch_slots", &tr::decoder_batch::Input::batchSlots);

    nb::class_<tr::LookaheadDecodingBuffers>(m, "LookaheadDecodingBuffers")
        .def(nb::init<tr::SizeType32, tr::SizeType32, tr::BufferManager const&>(), nb::arg("max_num_sequences"),
            nb::arg("max_tokens_per_step"), nb::arg("buffer_manager"), nb::call_guard<nb::gil_scoped_release>())
        .def_rw("generation_lengths", &tr::LookaheadDecodingBuffers::generationLengths)
        .def_rw("position_offsets", &tr::LookaheadDecodingBuffers::positionOffsets)
        .def_rw("packed_masks", &tr::LookaheadDecodingBuffers::packedMasks)
        .def_rw("position_ids", &tr::LookaheadDecodingBuffers::positionIds);

    nb::class_<tr::ExplicitDraftTokensBuffers::Inputs>(m, "ExplicitDraftTokensBuffersInputs")
        .def("create", &tr::ExplicitDraftTokensBuffers::Inputs::create, nb::arg("max_num_sequences"),
            nb::arg("runtime"), nb::arg("model_config"), nb::arg("world_config"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_rw("temperatures", &tr::ExplicitDraftTokensBuffers::Inputs::temperatures)
        .def_rw("position_ids_base", &tr::ExplicitDraftTokensBuffers::Inputs::positionIdsBase)
        .def_rw("generation_lengths", &tr::ExplicitDraftTokensBuffers::Inputs::generationLengths)
        .def_rw("random_data_sample", &tr::ExplicitDraftTokensBuffers::Inputs::randomDataSample)
        .def_rw("random_data_validation", &tr::ExplicitDraftTokensBuffers::Inputs::randomDataValidation)
        .def_rw("draft_tokens", &tr::ExplicitDraftTokensBuffers::Inputs::draftTokens)
        .def_rw("draft_indices", &tr::ExplicitDraftTokensBuffers::Inputs::draftIndices)
        .def_rw("draft_probs", &tr::ExplicitDraftTokensBuffers::Inputs::draftProbs)
        .def_rw("packed_masks", &tr::ExplicitDraftTokensBuffers::Inputs::packedMasks)
        .def_rw("position_ids", &tr::ExplicitDraftTokensBuffers::Inputs::positionIds)
        .def_rw("max_gen_length_host", &tr::ExplicitDraftTokensBuffers::Inputs::maxGenLengthHost)
        .def_rw("generation_lengths_host", &tr::ExplicitDraftTokensBuffers::Inputs::generationLengthsHost);

    nb::class_<tr::DecodingInput>(m, "DecodingInput");
    nb::class_<tr::DecodingOutput>(m, "DecodingOutput");

    nb::class_<tr::CudaEvent>(m, "CudaEvent")
        .def(nb::init<unsigned int>(), nb::arg("flags") = cudaEventDisableTiming,
            nb::call_guard<nb::gil_scoped_release>())
        .def("synchronize", &tr::CudaEvent::synchronize, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tr::IGptDecoder, PyIGptDecoder>(m, "IGptDecoder")
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
            nb::arg("sampling_config"), nb::arg("batch_size"), nb::arg("batch_slots"), nb::arg("output") = std::nullopt,
            nb::arg("explicit_draft_tokens_d_type") = std::nullopt, nb::arg("lookahead_prompt") = std::nullopt,
            nb::arg("lookahead_algo_configs") = std::nullopt, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tr::decoder::DecoderState>(m, "DecoderState")
        .def(nb::init<>(), nb::call_guard<nb::gil_scoped_release>())
        .def("setup", &tr::decoder::DecoderState::setup, nb::arg("max_num_sequences"), nb::arg("max_beam_width"),
            nb::arg("max_attention_window"), nb::arg("sink_token_length"), nb::arg("max_sequence_length"),
            nb::arg("dtype"), nb::arg("model_config"), nb::arg("world_config"), nb::arg("buffer_manager"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("setup_cache_indirection", &tr::decoder::DecoderState::setupCacheIndirection, nb::arg("max_num_sequences"),
            nb::arg("max_beam_width"), nb::arg("max_attention_window"), nb::arg("buffer_manager"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("setup_speculative_decoding", &tr::decoder::DecoderState::setupSpeculativeDecoding,
            nb::arg("speculative_decoding_mode"), nb::arg("max_tokens_per_engine_step"), nb::arg("dtype"),
            nb::arg("model_config"), nb::arg("world_config"), nb::arg("buffer_manager"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("joint_decoding_input", &tr::decoder::DecoderState::getJointDecodingInput)
        .def_prop_ro("joint_decoding_output", &tr::decoder::DecoderState::getJointDecodingOutput)
        .def_prop_ro("cache_indirection_input", &tr::decoder::DecoderState::getCacheIndirectionInput)
        .def_prop_ro("cache_indirection_output", &tr::decoder::DecoderState::getCacheIndirectionOutput)
        .def_prop_ro(
            "sequence_lengths", nb::overload_cast<>(&tr::decoder::DecoderState::getSequenceLengths, nb::const_))
        .def("get_sequence_lengths",
            nb::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getSequenceLengths, nb::const_),
            nb::arg("batch_idx"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("all_new_tokens", &tr::decoder::DecoderState::getAllNewTokens)
        .def_prop_ro("finished_sum", &tr::decoder::DecoderState::getFinishedSum)
        .def_prop_ro("finish_reasons", &tr::decoder::DecoderState::getFinishReasons)
        .def_prop_ro("ids", nb::overload_cast<>(&tr::decoder::DecoderState::getIds, nb::const_))
        .def("get_ids", nb::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getIds, nb::const_),
            nb::arg("batch_idx"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("gathered_ids", nb::overload_cast<>(&tr::decoder::DecoderState::getGatheredIds, nb::const_))
        .def("get_gathered_ids",
            nb::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getGatheredIds, nb::const_),
            nb::arg("batch_idx"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("parent_ids", &tr::decoder::DecoderState::getParentIds)
        .def_prop_ro("cum_log_probs", nb::overload_cast<>(&tr::decoder::DecoderState::getCumLogProbs, nb::const_))
        .def("get_cum_log_probs",
            nb::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getCumLogProbs, nb::const_),
            nb::arg("batch_idx"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("log_probs", nb::overload_cast<>(&tr::decoder::DecoderState::getLogProbs, nb::const_))
        .def("get_log_probs", nb::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getLogProbs, nb::const_),
            nb::arg("batch_idx"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("next_draft_tokens", &tr::decoder::DecoderState::getNextDraftTokens)
        .def_prop_ro("prev_draft_tokens_lengths", &tr::decoder::DecoderState::getPrevDraftTokensLengths)
        .def_prop_ro("next_draft_tokens_lengths", &tr::decoder::DecoderState::getNextDraftTokensLengths)
        .def_prop_ro("accepted_lengths_cum_sum", &tr::decoder::DecoderState::getAcceptedLengthsCumSum)
        .def_prop_ro("accepted_packed_paths", &tr::decoder::DecoderState::getAcceptedPackedPaths)
        .def_prop_ro("max_beam_width", &tr::decoder::DecoderState::getMaxBeamWidth)
        .def_prop_ro("max_sequence_length", &tr::decoder::DecoderState::getMaxSequenceLength)
        .def_prop_ro("max_decoding_decoder_tokens", &tr::decoder::DecoderState::getMaxDecodingDecoderTokens)
        .def_prop_ro("max_decoding_engine_tokens", &tr::decoder::DecoderState::getMaxDecodingEngineTokens)
        .def_prop_ro("num_decoding_engine_tokens",
            nb::overload_cast<>(&tr::decoder::DecoderState::getNumDecodingEngineTokens, nb::const_))
        .def("get_num_decoding_engine_tokens",
            nb::overload_cast<tr::SizeType32>(&tr::decoder::DecoderState::getNumDecodingEngineTokens, nb::const_),
            nb::arg("batch_idx"), nb::call_guard<nb::gil_scoped_release>())
        .def("set_num_decoding_engine_tokens", &tr::decoder::DecoderState::setNumDecodingEngineTokens,
            nb::arg("batch_idx"), nb::arg("num_tokens"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("speculative_decoding_mode", &tr::decoder::DecoderState::getSpeculativeDecodingMode)
        .def_prop_rw("generation_steps", &tr::decoder::DecoderState::getGenerationSteps,
            &tr::decoder::DecoderState::setGenerationSteps);

    nb::class_<tr::GptDecoderBatched>(m, "GptDecoderBatched")
        .def(nb::init<tr::GptDecoderBatched::CudaStreamPtr>(), nb::arg("stream"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("setup", &tr::GptDecoderBatched::setup, nb::arg("mode"), nb::arg("max_num_sequences"),
            nb::arg("max_beam_width"), nb::arg("dtype"), nb::arg("model_config"), nb::arg("world_config"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("forward_async", &tr::GptDecoderBatched::forwardAsync, nb::arg("decoder_state"), nb::arg("input"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("underlying_decoder", &tr::GptDecoderBatched::getUnderlyingDecoder, nb::rv_policy::reference)
        .def("finalize", &tr::GptDecoderBatched::finalize, nb::arg("decoder_state"), nb::arg("batch_idx"),
            nb::arg("sampling_config"), nb::arg("streaming"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro(
            "decoder_stream",
            [](tr::GptDecoderBatched& self) -> tr::CudaStream const& { return *self.getDecoderStream(); },
            nb::rv_policy::reference);

    m.def(
        "lamport_initialize_all",
        [](intptr_t buffer_0, intptr_t buffer_1, intptr_t buffer_2, size_t size)
        {
            tr::lamportInitializeAll(reinterpret_cast<void*>(buffer_0), reinterpret_cast<void*>(buffer_1),
                reinterpret_cast<void*>(buffer_2), size);
        },
        "Lamport initialize all buffers", nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "lamport_initialize",
        [](intptr_t buffer, size_t size)
        { tensorrt_llm::kernels::ar_fusion::lamport_initialize(reinterpret_cast<void*>(buffer), size, 0); },
        "Lmaport initialize buffer", nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "delay_kernel",
        [](int64_t delay_micro_secs, nb::object py_stream)
        {
            // Get the raw stream handle from PyTorch stream object
            auto stream_ptr = nb::cast<int64_t>(py_stream.attr("cuda_stream"));
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            nb::gil_scoped_release release;
            tensorrt_llm::kernels::invokeDelayStreamKernel(delay_micro_secs, stream);
        },
        "Delay kernel launch on the default stream");
    m.def(
        "max_workspace_size_lowprecision",
        [](int32_t tp_size) { return tensorrt_llm::kernels::max_workspace_size_lowprecision(tp_size); },
        "Calculate the maximum workspace size needed for low precision all-reduce operations",
        nb::call_guard<nb::gil_scoped_release>());

    nb::enum_<tr::CudaVirtualMemoryAllocator::RestoreMode>(m, "CudaVirtualMemoryAllocatorRestoreMode")
        .value("NONE", tr::CudaVirtualMemoryAllocator::RestoreMode::NONE)
        .value("CPU", tr::CudaVirtualMemoryAllocator::RestoreMode::CPU)
        .value("PINNED", tr::CudaVirtualMemoryAllocator::RestoreMode::PINNED)
        .value("MEMSET", tr::CudaVirtualMemoryAllocator::RestoreMode::MEMSET);

    m.def("get_virtual_memory_manager", &tr::getVirtualMemoryManager, "Get the virtual memory manager",
        nb::rv_policy::reference);

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
        nb::call_guard<nb::gil_scoped_release>());

    m.def("clear_virtual_memory_allocator", &tr::clearVirtualMemoryAllocator,
        "Reset the current virtual memory allocator and stop allocating virtual memory for CUDA allocations",
        nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tensorrt_llm::runtime::McastGPUBuffer>(m, "McastGPUBuffer")
        .def(nb::init<size_t, uint32_t, uint32_t, uint32_t, uint32_t, bool>(), nb::arg("buf_size"),
            nb::arg("group_size"), nb::arg("group_rank"), nb::arg("split_color"), nb::arg("device_idx"),
            nb::arg("mn_nvlink"), nb::call_guard<nb::gil_scoped_release>())
        .def("get_uc_buffer", &tensorrt_llm::runtime::McastGPUBuffer::getUCBuffer,
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_mc_buffer", &tensorrt_llm::runtime::McastGPUBuffer::getMCBuffer,
            nb::call_guard<nb::gil_scoped_release>());

    nb::enum_<tensorrt_llm::kernels::AllReduceFusionOp>(m, "AllReduceFusionOp")
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

    nb::enum_<tensorrt_llm::kernels::AllReduceStrategyType>(m, "AllReduceStrategy")
        .value("NCCL", tensorrt_llm::kernels::AllReduceStrategyType::NCCL)
        .value("MIN_LATENCY", tensorrt_llm::kernels::AllReduceStrategyType::MIN_LATENCY)
        .value("AUTO", tensorrt_llm::kernels::AllReduceStrategyType::AUTO)
        .value("UB", tensorrt_llm::kernels::AllReduceStrategyType::UB)
        .value("ONESHOT", tensorrt_llm::kernels::AllReduceStrategyType::ONESHOT)
        .value("TWOSHOT", tensorrt_llm::kernels::AllReduceStrategyType::TWOSHOT);

    // Initialize MoeLoadBalancer bindings
    initMoeBindings(m);
    // Initialize HostFunc bindings
    initHostFuncBindings(m);
}

void initBindingsEarly(nb::module_& m)
{
    nb::class_<tr::SpeculativeDecodingMode>(m, "SpeculativeDecodingMode")
        .def(nb::init<tr::SpeculativeDecodingMode::UnderlyingType>(), nb::arg("state"))
        .def_static("NoneType", &tr::SpeculativeDecodingMode::None)
        .def_static("DraftTokensExternal", &tr::SpeculativeDecodingMode::DraftTokensExternal)
        .def_static("Medusa", &tr::SpeculativeDecodingMode::Medusa)
        .def_static("Eagle", &tr::SpeculativeDecodingMode::Eagle)
        .def_static("LookaheadDecoding", &tr::SpeculativeDecodingMode::LookaheadDecoding)
        .def_static("ExplicitDraftTokens", &tr::SpeculativeDecodingMode::ExplicitDraftTokens)
        .def_prop_ro("is_none", &tr::SpeculativeDecodingMode::isNone)
        .def_prop_ro("is_draft_tokens_external", &tr::SpeculativeDecodingMode::isDraftTokensExternal)
        .def_prop_ro("is_medusa", &tr::SpeculativeDecodingMode::isMedusa)
        .def_prop_ro("is_eagle", &tr::SpeculativeDecodingMode::isEagle)
        .def_prop_ro("is_lookahead_decoding", &tr::SpeculativeDecodingMode::isLookaheadDecoding)
        .def_prop_ro("is_explicit_draft_tokens", &tr::SpeculativeDecodingMode::isExplicitDraftTokens)
        .def_prop_ro("updates_position_ids", &tr::SpeculativeDecodingMode::updatesPositionIds)
        .def_prop_ro("requires_attention_mask", &tr::SpeculativeDecodingMode::requiresAttentionMask)
        .def_prop_ro("predicts_draft_tokens", &tr::SpeculativeDecodingMode::predictsDraftTokens)
        .def_prop_ro("needs_kv_cache_rewind", &tr::SpeculativeDecodingMode::needsKVCacheRewind)
        .def_prop_ro("variable_draft_length", &tr::SpeculativeDecodingMode::variableDraftLength)
        .def_prop_ro("has_draft_logits", &tr::SpeculativeDecodingMode::hasDraftLogits)
        .def_prop_ro("needs_decoder_prologue", &tr::SpeculativeDecodingMode::needsDecoderPrologue);
}
} // namespace tensorrt_llm::nanobind::runtime
