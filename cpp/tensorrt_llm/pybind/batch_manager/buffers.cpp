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
#include "tensorrt_llm/runtime/torch.h"

#include <ATen/ATen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using tr::SizeType32;

namespace tensorrt_llm::pybind::batch_manager
{

void Buffers::initBindings(pybind11::module_& m)
{
    py::class_<tb::DecoderInputBuffers>(m, "DecoderInputBuffers")
        .def(py::init<tr::SizeType32, tr::SizeType32, tr::BufferManager>(), py::arg("max_batch_size"),
            py::arg("max_tokens_per_engine_step"), py::arg("manager"))
        .def_readwrite("setup_batch_slots", &tb::DecoderInputBuffers::setupBatchSlots)
        .def_readwrite("setup_batch_slots_device", &tb::DecoderInputBuffers::setupBatchSlotsDevice)
        .def_readwrite("fill_values", &tb::DecoderInputBuffers::fillValues)
        .def_readwrite("fill_values_device", &tb::DecoderInputBuffers::fillValuesDevice)
        .def_readwrite("inputs_ids", &tb::DecoderInputBuffers::inputsIds)
        .def_readwrite("batch_logits", &tb::DecoderInputBuffers::batchLogits)
        .def_readwrite("forward_batch_slots", &tb::DecoderInputBuffers::forwardBatchSlots)
        .def_readwrite("decoder_logits", &tb::DecoderInputBuffers::decoderLogits)
        .def_readwrite("max_decoder_steps", &tb::DecoderInputBuffers::maxDecoderSteps);

    py::class_<tb::DecoderOutputBuffers>(m, "DecoderOutputBuffers")
        .def_readwrite("sequence_lengths_host", &tb::DecoderOutputBuffers::sequenceLengthsHost)
        .def_readwrite("finished_sum_host", &tb::DecoderOutputBuffers::finishedSumHost)
        .def_property_readonly("new_output_tokens_host",
            [](tb::DecoderOutputBuffers& self) { return tr::Torch::tensor(self.newOutputTokensHost); })
        .def_readwrite("cum_log_probs_host", &tb::DecoderOutputBuffers::cumLogProbsHost)
        .def_readwrite("log_probs_host", &tb::DecoderOutputBuffers::logProbsHost)
        .def_readwrite("finish_reasons_host", &tb::DecoderOutputBuffers::finishReasonsHost);

    py::class_<tb::SlotDecoderBuffers>(m, "SlotDecoderBuffers")
        .def(py::init<runtime::SizeType32, runtime::SizeType32, runtime::BufferManager const&>(),
            py::arg("max_beam_width"), py::arg("max_seq_len"), py::arg("buffer_manager"))
        .def_readwrite("output_ids", &tb::SlotDecoderBuffers::outputIds)
        .def_readwrite("output_ids_host", &tb::SlotDecoderBuffers::outputIdsHost)
        .def_readwrite("sequence_lengths_host", &tb::SlotDecoderBuffers::sequenceLengthsHost)
        .def_readwrite("cum_log_probs", &tb::SlotDecoderBuffers::cumLogProbs)
        .def_readwrite("cum_log_probs_host", &tb::SlotDecoderBuffers::cumLogProbsHost)
        .def_readwrite("log_probs", &tb::SlotDecoderBuffers::logProbs)
        .def_readwrite("log_probs_host", &tb::SlotDecoderBuffers::logProbsHost)
        .def_readwrite("finish_reasons_host", &tb::SlotDecoderBuffers::finishReasonsHost);
}
} // namespace tensorrt_llm::pybind::batch_manager
