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
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/core/TensorBody.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>

#include <optional>

namespace nb = nanobind;

namespace tr = tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;

void tensorrt_llm::nanobind::batch_manager::algorithms::initBindings(nb::module_& m)
{
    nb::class_<CapacityScheduler>(m, CapacityScheduler::name)
        .def(nb::init<SizeType32, executor::CapacitySchedulerPolicy, bool, bool, LlmRequestState, LlmRequestState>(),
            nb::arg("max_num_requests"), nb::arg("capacity_scheduler_policy"), nb::arg("has_kv_cache_manager"),
            nb::arg("two_step_lookahead") = false, nb::arg("no_schedule_until_state") = LlmRequestState::kCONTEXT_INIT,
            nb::arg("no_schedule_after_state") = LlmRequestState::kGENERATION_COMPLETE)
        .def("__call__", &CapacityScheduler::operator(), nb::arg("active_requests"),
            nb::arg("kv_cache_manager") = nullptr, nb::arg("peft_cache_manager") = nullptr,
            nb::arg("cross_kv_cache_manager") = nullptr)
        .def("name", [](CapacityScheduler const&) { return CapacityScheduler::name; });

    nb::class_<MicroBatchScheduler>(m, MicroBatchScheduler::name)
        .def(nb::init<std::optional<batch_scheduler::ContextChunkingConfig>, std::optional<SizeType32>, LlmRequestState,
                 LlmRequestState>(),
            nb::arg("ctx_chunk_config") = std::nullopt, nb::arg("max_context_length") = std::nullopt,
            nb::arg("no_schedule_until_state") = LlmRequestState::kCONTEXT_INIT,
            nb::arg("no_schedule_after_state") = LlmRequestState::kGENERATION_COMPLETE)
        .def("__call__", &MicroBatchScheduler::operator(), nb::arg("active_requests"), nb::arg("inflight_req_ids"),
            nb::arg("max_batch_size_runtime"), nb::arg("max_num_tokens_runtime"))
        .def("name", [](MicroBatchScheduler const&) { return MicroBatchScheduler::name; });

    nb::class_<PauseRequests>(m, PauseRequests::name)
        .def(nb::init<SizeType32>(), nb::arg("max_input_len"))
        .def("__call__", &PauseRequests::operator(), nb::arg("requests_to_pause"), nb::arg("inflight_req_ids"),
            nb::arg("req_ids_to_pause"), nb::arg("pause_flagged"), nb::arg("seq_slot_manager"),
            nb::arg("kv_cache_manager") = std::nullopt, nb::arg("cross_kv_cache_manager") = std::nullopt,
            nb::arg("peft_cache_manager") = std::nullopt)
        .def("name", [](PauseRequests const&) { return PauseRequests::name; });

    nb::class_<AssignReqSeqSlots>(m, AssignReqSeqSlots::name)
        .def(nb::init<>())
        .def("__call__", &AssignReqSeqSlots::operator(), nb::arg("seq_slot_manager"), nb::arg("context_requests"),
            nb::arg("generation_requests"))
        .def("name", [](AssignReqSeqSlots const&) { return AssignReqSeqSlots::name; });

    nb::class_<AllocateKvCache>(m, AllocateKvCache::name)
        .def(nb::init<>())
        .def("__call__", &AllocateKvCache::operator(), nb::arg("kv_cache_manager"), nb::arg("context_requests"),
            nb::arg("generation_requests"), nb::arg("model_config"), nb::arg("cross_kv_cache_manager") = std::nullopt)
        .def("name", [](AllocateKvCache const&) { return AllocateKvCache::name; });

    nb::class_<HandleContextLogits>(m, HandleContextLogits::name)
        .def(nb::init<>())
        .def(
            "__call__",
            [](HandleContextLogits const& self, DecoderInputBuffers& inputBuffers, RequestVector const& contextRequests,
                at::Tensor const& logits, std::vector<tr::SizeType32> const& numContextLogitsVec,
                tr::ModelConfig const& modelConfig, tr::BufferManager const& manager,
                OptionalRef<MedusaBuffers> medusaBuffers = std::nullopt)
            {
                return self(inputBuffers, contextRequests, tr::TorchView::of(logits), numContextLogitsVec, modelConfig,
                    manager, medusaBuffers);
            },
            nb::arg("decoder_input_buffers"), nb::arg("context_requests"), nb::arg("logits"),
            nb::arg("num_context_logits"), nb::arg("model_config"), nb::arg("buffer_manager"),
            nb::arg("draft_buffers") = std::nullopt)
        .def("name", [](HandleContextLogits const&) { return HandleContextLogits::name; });

    nb::class_<HandleGenerationLogits>(m, HandleGenerationLogits::name)
        .def(nb::init<>())
        .def(
            "__call__",
            [](HandleGenerationLogits const& self, DecoderInputBuffers& inputBuffers,
                RequestVector const& generationRequests, at::Tensor const& logits, tr::SizeType32 logitsIndex,
                tr::ModelConfig const& modelConfig, tr::BufferManager const& manager,
                OptionalRef<RuntimeBuffers> genRuntimeBuffers = std::nullopt,
                OptionalRef<MedusaBuffers> medusaBuffers = std::nullopt)
            {
                self(inputBuffers, generationRequests, tr::TorchView::of(logits), logitsIndex, modelConfig, manager,
                    genRuntimeBuffers, medusaBuffers);
            },
            nb::arg("decoder_input_buffers"), nb::arg("generation_requests"), nb::arg("logits"),
            nb::arg("logits_index"), nb::arg("model_config"), nb::arg("buffer_manager"),
            nb::arg("gen_runtime_buffers") = std::nullopt, nb::arg("medusa_buffers") = std::nullopt)
        .def("name", [](HandleGenerationLogits const&) { return HandleGenerationLogits::name; });

    nb::class_<MakeDecodingBatchInputOutput>(m, MakeDecodingBatchInputOutput::name)
        .def(nb::init<>())
        .def("__call__", &MakeDecodingBatchInputOutput::operator(), nb::arg("context_requests"),
            nb::arg("generation_requests"), nb::arg("decoder_input_buffers"), nb::arg("decoder_state"),
            nb::arg("model_config"), nb::arg("max_num_sequences"), nb::arg("fused_runtime_buffers") = std::nullopt)
        .def("name", [](MakeDecodingBatchInputOutput const&) { return MakeDecodingBatchInputOutput::name; });

    nb::class_<LogitsPostProcessor>(m, LogitsPostProcessor::name)
        .def(nb::init<>())
        .def("__call__", &LogitsPostProcessor::operator(), nb::arg("context_requests"), nb::arg("generation_requests"),
            nb::arg("replicate_logits_post_processor"), nb::arg("decoder_buffers"), nb::arg("world_config"),
            nb::arg("runtime"), nb::arg("logits_post_processor_batched") = std::nullopt)
        .def("name", [](LogitsPostProcessor const&) { return LogitsPostProcessor::name; });

    nb::class_<CreateNewDecoderRequests>(m, CreateNewDecoderRequests::name)
        .def(nb::init<bool, bool, bool>(), nb::arg("speculative_decoding_fast_logits"),
            nb::arg("is_leader_in_orch_mode"), nb::arg("is_normalize_log_probs"))
        .def(
            "__call__",
            [](CreateNewDecoderRequests& self, tr::ModelConfig const& modelConfig, tr::WorldConfig const& worldConfig,
                executor::DecodingConfig const& decodingConfig, RequestVector const& contextRequests,
                tr::BufferManager const& bufferManager, nvinfer1::DataType logitsType,
                DecoderInputBuffers& inputBuffers, runtime::decoder::DecoderState& decoderState,
                tensorrt_llm::runtime::CudaStream const& runtimeStream,
                tensorrt_llm::runtime::CudaStream const& decoderStream, SizeType32 maxSequenceLength,
                SizeType32 beamWidth, OptionalRef<MedusaBuffers const> medusaBuffers = std::nullopt)
            {
                auto [batchSlots, samplingConfigs, lookaheadPrompt, lookaheadAlgoConfigs] = self(modelConfig,
                    worldConfig, decodingConfig, contextRequests, bufferManager, logitsType, inputBuffers, decoderState,
                    runtimeStream, decoderStream, maxSequenceLength, beamWidth, medusaBuffers);

                return std::tuple{runtime::Torch::tensor(batchSlots), std::move(samplingConfigs),
                    std::move(lookaheadPrompt), std::move(lookaheadAlgoConfigs)};
            },
            nb::arg("model_config"), nb::arg("world_config"), nb::arg("decoding_config"), nb::arg("context_requests"),
            nb::arg("buffer_manager"), nb::arg("logits_type"), nb::arg("decoder_input_buffers"),
            nb::arg("decoder_state"), nb::arg("runtime_stream"), nb::arg("decoder_stream"),
            nb::arg("max_sequence_length"), nb::arg("beam_width"), nb::arg("medusa_buffers") = std::nullopt)
        .def("name", [](CreateNewDecoderRequests const&) { return CreateNewDecoderRequests::name; });

    nb::class_<UpdateDecoderBuffers>(m, "UpdateDecoderBuffers")
        .def(nb::init<>())
        .def("__call__", &UpdateDecoderBuffers::operator(), nb::arg("model_config"), nb::arg("decoder_output_buffers"),
            nb::arg("copy_buffer_manager"), nb::arg("decoder_state"), nb::arg("return_log_probs"),
            nb::arg("decoder_finish_event"))
        .def("name", [](UpdateDecoderBuffers const&) { return UpdateDecoderBuffers::name; });
}
