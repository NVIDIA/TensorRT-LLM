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
#include "generationOutput.h"

#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::pybind::runtime;

std::shared_ptr<tr::GenerationOutput> GenerationOutput::toTrtLlm() const
{
    auto output
        = std::make_shared<tr::GenerationOutput>(tr::TorchView::of(ids.value()), tr::TorchView::of(lengths.value()));
    if (logProbs)
    {
        output->logProbs = tr::TorchView::of(logProbs.value());
    }
    if (contextLogits)
    {
        output->contextLogits = tr::TorchView::of(contextLogits.value());
    }
    if (generationLogits)
    {
        output->generationLogits = tr::TorchView::of(generationLogits.value());
    }

    if (onTokenGenerated)
    {
        output->onTokenGenerated = [delegate = onTokenGenerated](
                                       tr::GenerationOutput::TensorPtr const& ids, tr::SizeType step, bool finished)
        { delegate(tr::Torch::tensor(ids), step, finished); };
    }
    return output;
}

void GenerationOutput::initBindings(py::module_& m)
{
    py::class_<GenerationOutput>(m, "GenerationOutput")
        .def(py::init<GenerationOutput::TensorPtr, GenerationOutput::TensorPtr>(), py::arg("ids"), py::arg("lengths"))
        .def_readwrite("ids", &GenerationOutput::ids)
        .def_readwrite("lengths", &GenerationOutput::lengths)
        .def_readwrite("log_probs", &GenerationOutput::logProbs)
        .def_readwrite("context_logits", &GenerationOutput::contextLogits)
        .def_readwrite("generation_logits", &GenerationOutput::generationLogits)
        .def_readwrite("on_token_generated", &GenerationOutput::onTokenGenerated);
}
