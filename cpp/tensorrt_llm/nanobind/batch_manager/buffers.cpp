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

#include "buffers.h"

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/nanobind/batch_manager/llmRequest.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/torch.h"

#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <torch/extension.h>

namespace nb = nanobind;
namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using tr::SizeType32;

namespace tensorrt_llm::nanobind::batch_manager
{

void Buffers::initBindings(nb::module_& m)
{
    nb::class_<tb::DecoderInputBuffers>(m, "DecoderInputBuffers")
        .def(nb::init<runtime::SizeType32, runtime::SizeType32, tr::BufferManager>(), nb::arg("max_batch_size"),
            nb::arg("max_tokens_per_engine_step"), nb::arg("manager"))
        .def_rw("setup_batch_slots", &tb::DecoderInputBuffers::setupBatchSlots)
        .def_rw("setup_batch_slots_device", &tb::DecoderInputBuffers::setupBatchSlotsDevice)
        .def_rw("fill_values", &tb::DecoderInputBuffers::fillValues)
        .def_rw("fill_values_device", &tb::DecoderInputBuffers::fillValuesDevice)
        .def_rw("inputs_ids", &tb::DecoderInputBuffers::inputsIds)
        .def_rw("forward_batch_slots", &tb::DecoderInputBuffers::forwardBatchSlots)
        .def_rw("decoder_logits", &tb::DecoderInputBuffers::decoderLogits)
        .def_rw("decoder_requests", &tb::DecoderInputBuffers::decoderRequests);

    nb::class_<tb::DecoderOutputBuffers>(m, "DecoderOutputBuffers")
        .def_rw("sequence_lengths_host", &tb::DecoderOutputBuffers::sequenceLengthsHost)
        .def_rw("finished_sum_host", &tb::DecoderOutputBuffers::finishedSumHost)
        .def_prop_ro("new_output_tokens_host",
            [](tb::DecoderOutputBuffers& self) { return tr::Torch::tensor(self.newOutputTokensHost); })
        .def_rw("cum_log_probs_host", &tb::DecoderOutputBuffers::cumLogProbsHost)
        .def_rw("log_probs_host", &tb::DecoderOutputBuffers::logProbsHost)
        .def_rw("finish_reasons_host", &tb::DecoderOutputBuffers::finishReasonsHost);

    nb::class_<tb::SlotDecoderBuffers>(m, "SlotDecoderBuffers")
        .def(nb::init<runtime::SizeType32, runtime::SizeType32, runtime::BufferManager const&>(),
            nb::arg("max_beam_width"), nb::arg("max_seq_len"), nb::arg("buffer_manager"))
        .def_rw("output_ids", &tb::SlotDecoderBuffers::outputIds)
        .def_rw("output_ids_host", &tb::SlotDecoderBuffers::outputIdsHost)
        .def_rw("sequence_lengths_host", &tb::SlotDecoderBuffers::sequenceLengthsHost)
        .def_rw("cum_log_probs", &tb::SlotDecoderBuffers::cumLogProbs)
        .def_rw("cum_log_probs_host", &tb::SlotDecoderBuffers::cumLogProbsHost)
        .def_rw("log_probs", &tb::SlotDecoderBuffers::logProbs)
        .def_rw("log_probs_host", &tb::SlotDecoderBuffers::logProbsHost)
        .def_rw("finish_reasons_host", &tb::SlotDecoderBuffers::finishReasonsHost);
}
} // namespace tensorrt_llm::nanobind::batch_manager
