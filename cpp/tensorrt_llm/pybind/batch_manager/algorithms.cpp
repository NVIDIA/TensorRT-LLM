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

#include "algorithms.h"
#include "tensorrt_llm/batch_manager/allocateKvCache.h"
#include "tensorrt_llm/batch_manager/assignReqSeqSlots.h"
#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/createNewDecoderRequests.h"
#include "tensorrt_llm/batch_manager/generateRequestOptions.h"
#include "tensorrt_llm/batch_manager/handleContextLogits.h"
#include "tensorrt_llm/batch_manager/handleGenerationLogits.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/logitsPostProcessor.h"
#include "tensorrt_llm/batch_manager/makeDecodingBatchInputOutput.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/pauseRequests.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/updateDecoderBuffers.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/core/TensorBody.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <optional>

namespace py = pybind11;

namespace tr = tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;

void tensorrt_llm::pybind::batch_manager::algorithms::initBindings(pybind11::module_& m)
{
    py::class_<CapacityScheduler>(m, CapacityScheduler::name)
        .def(py::init<SizeType32, executor::CapacitySchedulerPolicy, bool, bool, LlmRequestState, LlmRequestState>(),
            py::arg("max_num_requests"), py::arg("capacity_scheduler_policy"), py::arg("has_kv_cache_manager"),
            py::arg("many_micro_batches") = false,
            py::arg_v("no_schedule_until_state", LlmRequestState::kCONTEXT_INIT, "LlmRequestState.CONTEXT_INIT"),
            py::arg_v("no_schedule_after_state", LlmRequestState::kGENERATION_COMPLETE,
                "LlmRequestState.GENERATION_COMPLETE"))
        .def("__call__", &CapacityScheduler::operator(), py::arg("active_requests"),
            py::arg("kv_cache_manager") = nullptr, py::arg("peft_cache_manager") = nullptr,
            py::arg("cross_kv_cache_manager") = nullptr)
        .def("name", [](CapacityScheduler const&) { return CapacityScheduler::name; });

    py::class_<MicroBatchScheduler>(m, MicroBatchScheduler::name)
        .def(py::init<std::optional<batch_scheduler::ContextChunkingConfig>, std::optional<SizeType32>, LlmRequestState,
                 LlmRequestState>(),
            py::arg("ctx_chunk_config") = std::nullopt, py::arg("max_context_length") = std::nullopt,
            py::arg_v("no_schedule_until_state", LlmRequestState::kCONTEXT_INIT, "LlmRequestState.CONTEXT_INIT"),
            py::arg_v("no_schedule_after_state", LlmRequestState::kGENERATION_COMPLETE,
                "LlmRequestState.GENERATION_COMPLETE"))
        .def("__call__", &MicroBatchScheduler::operator(), py::arg("active_requests"), py::arg("inflight_req_ids"),
            py::arg("max_batch_size_runtime"), py::arg("max_num_tokens_runtime"))
        .def("name", [](MicroBatchScheduler const&) { return MicroBatchScheduler::name; });

    py::class_<PauseRequests>(m, PauseRequests::name)
        .def(py::init<SizeType32>(), py::arg("max_input_len"))
        .def("__call__", &PauseRequests::operator(), py::arg("requests_to_pause"), py::arg("inflight_req_ids"),
            py::arg("req_ids_to_pause"), py::arg("pause_flagged"), py::arg("seq_slot_manager"),
            py::arg("kv_cache_manager") = std::nullopt, py::arg("cross_kv_cache_manager") = std::nullopt,
            py::arg("peft_cache_manager") = std::nullopt)
        .def("name", [](PauseRequests const&) { return PauseRequests::name; });

    py::class_<AssignReqSeqSlots>(m, AssignReqSeqSlots::name)
        .def(py::init())
        .def("__call__", &AssignReqSeqSlots::operator(), py::arg("seq_slot_manager"), py::arg("context_requests"),
            py::arg("generation_requests"))
        .def("name", [](AssignReqSeqSlots const&) { return AssignReqSeqSlots::name; });

    py::class_<AllocateKvCache>(m, AllocateKvCache::name)
        .def(py::init())
        .def("__call__", &AllocateKvCache::operator(), py::arg("kv_cache_manager"), py::arg("context_requests"),
            py::arg("generation_requests"), py::arg("model_config"), py::arg("cross_kv_cache_manager") = std::nullopt)
        .def("name", [](AllocateKvCache const&) { return AllocateKvCache::name; });

    py::class_<HandleContextLogits>(m, HandleContextLogits::name)
        .def(py::init())
        .def(
            "__call__",
            [](HandleContextLogits const& self, RequestVector const& contextRequests,
                std::vector<tr::SizeType32> const& numContextLogitsVec, at::Tensor const& logits,
                DecoderBuffers& decoderBuffers, tr::ModelConfig const& modelConfig, tr::BufferManager const& manager,
                tensorrt_llm::runtime::CudaStream const& stream,
                OptionalRef<MedusaBuffers> medusaBuffers = std::nullopt)
            {
                return self(contextRequests, numContextLogitsVec, tr::TorchView::of(logits), decoderBuffers,
                    modelConfig, manager, stream, medusaBuffers);
            },
            py::arg("context_requests"), py::arg("num_context_logits"), py::arg("logits"), py::arg("decoder_buffers"),
            py::arg("model_config"), py::arg("buffer_manager"), py::arg("stream"),
            py::arg("medusa_buffers") = std::nullopt)
        .def("name", [](HandleContextLogits const&) { return HandleContextLogits::name; });

    py::class_<HandleGenerationLogits>(m, HandleGenerationLogits::name)
        .def(py::init())
        .def(
            "__call__",
            [](HandleGenerationLogits const& self, tr::SizeType32 logitsIndex, RequestVector const& generationRequests,
                DecoderBuffers& decoderBuffers, tr::ModelConfig const& modelConfig, tr::BufferManager const& manager,
                at::Tensor const& logits, OptionalRef<RuntimeBuffers> genRuntimeBuffers = std::nullopt)
            {
                self(logitsIndex, generationRequests, decoderBuffers, modelConfig, manager, tr::TorchView::of(logits),
                    genRuntimeBuffers);
            },
            py::arg("logits_index"), py::arg("generation_requests"), py::arg("decoder_buffers"),
            py::arg("model_config"), py::arg("buffer_manager"), py::arg("logits"),
            py::arg("gen_runtime_buffers") = std::nullopt)
        .def("name", [](HandleGenerationLogits const&) { return HandleGenerationLogits::name; });

    py::class_<GenerateRequestOptions>(m, GenerateRequestOptions::name)
        .def(py::init<bool, bool, bool>(), py::arg("speculative_decoding_fast_logits"),
            py::arg("is_leader_in_orch_mode"), py::arg("is_normalize_log_probs"))
        .def(
            "__call__",
            [](GenerateRequestOptions& self, tr::ModelConfig const& modelConfig, tr::WorldConfig const& worldConfig,
                executor::DecodingConfig const& decodingConfig, RequestVector const& contextRequests,
                tr::BufferManager const& bufferManager, nvinfer1::DataType logitsType,
                DecoderInputBuffers& inputBuffers, OptionalRef<RuntimeBuffers const> buffers = std::nullopt)
            {
                auto [batchSlots, decoderRequests, samplingConfigs] = self(modelConfig, worldConfig, decodingConfig,
                    contextRequests, bufferManager, logitsType, inputBuffers, buffers);

                return std::tuple{
                    runtime::Torch::tensor(batchSlots), std::move(decoderRequests), std::move(samplingConfigs)};
            },
            py::arg("model_config"), py::arg("world_config"), py::arg("decoding_config"), py::arg("context_requests"),
            py::arg("buffer_manager"), py::arg("logits_type"), py::arg("decoder_input_buffers"),
            py::arg("buffers") = std::nullopt)
        .def("name", [](GenerateRequestOptions const&) { return GenerateRequestOptions::name; });

    py::class_<MakeDecodingBatchInputOutput>(m, MakeDecodingBatchInputOutput::name)
        .def(py::init())
        .def("__call__", &MakeDecodingBatchInputOutput::operator(), py::arg("context_requests"),
            py::arg("generation_requests"), py::arg("decoder_buffers"), py::arg("decoder_input_buffers"),
            py::arg("decoder_state"), py::arg("model_config"), py::arg("max_num_sequences"), py::arg("beam_width"),
            py::arg("is_trt_overlap"), py::arg("buffer_manager"), py::arg("stream"),
            py::arg("fused_runtime_buffers") = std::nullopt)
        .def("name", [](MakeDecodingBatchInputOutput const&) { return MakeDecodingBatchInputOutput::name; });

    py::class_<LogitsPostProcessor>(m, LogitsPostProcessor::name)
        .def(py::init())
        .def("__call__", &LogitsPostProcessor::operator(), py::arg("context_requests"), py::arg("generation_requests"),
            py::arg("replicate_logits_post_processor"), py::arg("decoder_buffers"), py::arg("world_config"),
            py::arg("runtime"), py::arg("logits_post_processor_batched") = std::nullopt)
        .def("name", [](LogitsPostProcessor const&) { return LogitsPostProcessor::name; });

    py::class_<CreateNewDecoderRequests>(m, CreateNewDecoderRequests::name)
        .def(py::init())
        .def(
            "__call__",
            [](CreateNewDecoderRequests& self, at::Tensor const& batchSlots,
                std::vector<runtime::decoder_batch::Request> const& requests,
                std::vector<runtime::SamplingConfig> const& samplingConfigs, runtime::ModelConfig const& modelConfig,
                runtime::GptDecoderBatched& decoder, tensorrt_llm::runtime::CudaStream const& runtimeStream,
                SizeType32 maxSequenceLength)
            {
                self(tr::TorchView::of(batchSlots), requests, samplingConfigs, modelConfig, decoder, runtimeStream,
                    maxSequenceLength);
            },
            py::arg("batch_slots"), py::arg("requests"), py::arg("sampling_configs"), py::arg("model_config"),
            py::arg("decoder"), py::arg("runtime_stream"), py::arg("max_sequence_length"))
        .def("name", [](CreateNewDecoderRequests const&) { return CreateNewDecoderRequests::name; });

    py::class_<UpdateDecoderBuffers>(m, UpdateDecoderBuffers::name)
        .def(py::init())
        .def("__call__", &UpdateDecoderBuffers::operator(), py::arg("model_config"), py::arg("decoder_buffers"),
            py::arg("decoder_output_buffers"), py::arg("copy_buffer_manager"), py::arg("decoder"),
            py::arg("return_log_probs"), py::arg("decoder_finish_event"))
        .def("name", [](UpdateDecoderBuffers const&) { return UpdateDecoderBuffers::name; });
}
