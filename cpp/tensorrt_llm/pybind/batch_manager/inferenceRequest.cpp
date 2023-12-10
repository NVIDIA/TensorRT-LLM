/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "inferenceRequest.h"

#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <memory>

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::pybind::batch_manager;

namespace
{

void copy_tensor(NamedTensor const& src, tb::NamedTensor& dst)
{
    TLLM_CHECK_WITH_INFO(src.name == dst.name, "names do not match: %s != %s", src.name.c_str(), dst.name.c_str());
    if (src.tensor.has_value())
    {
        dst.tensor = tr::TorchView::of(src.tensor.value());
    }
}
} // namespace

std::shared_ptr<tb::InferenceRequest> InferenceRequest::toTrtLlm() const
{
    tb::InferenceRequest::TensorMap tensorMap;
    for (auto const& [name, tensor] : mInputTensors)
    {
        if (tensor.has_value())
        {
            tensorMap[name] = tr::TorchView::of(tensor.value());
        }
    }
    auto inferenceRequest = std::make_shared<tb::InferenceRequest>(tensorMap, mRequestId);
    inferenceRequest->setIsStreaming(isStreaming());
    return inferenceRequest;
}

void InferenceRequest::initBindings(py::module_& m)
{
    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def(py::init<uint64_t>())
        .def(py::init<uint64_t, InferenceRequest::TensorMap const&>(), "deprecated: use direct tensor access instead")
        .def_property("input_ids", &InferenceRequest::getInputIdsUnchecked, &InferenceRequest::setInputIds)
        .def_property(
            "draft_input_ids", &InferenceRequest::getDraftInputIdsUnchecked, &InferenceRequest::setDraftInputIds)
        .def_property("draft_logits", &InferenceRequest::getDraftLogitsUnchecked, &InferenceRequest::setDraftLogits)
        .def_property("max_new_tokens", &InferenceRequest::getMaxNewTokensUnchecked, &InferenceRequest::setMaxNewTokens)
        .def_property("beam_width", &InferenceRequest::getBeamWidthUnchecked, &InferenceRequest::setBeamWidth)
        .def_property("end_id", &InferenceRequest::getEndIdUnchecked, &InferenceRequest::setEndId)
        .def_property("pad_id", &InferenceRequest::getPadIdUnchecked, &InferenceRequest::setPadId)
        .def_property("bad_words_list", &InferenceRequest::getBadWordsListUnchecked, &InferenceRequest::setBadWordsList)
        .def_property(
            "stop_words_list", &InferenceRequest::getStopWordsListUnchecked, &InferenceRequest::setStopWordsList)
        .def_property(
            "embedding_bias", &InferenceRequest::getEmbeddingBiasUnchecked, &InferenceRequest::setEmbeddingBias)
        .def_property("temperature", &InferenceRequest::getTemperatureUnchecked, &InferenceRequest::setTemperature)
        .def_property("runtime_top_k", &InferenceRequest::getRuntimeTopKUnchecked, &InferenceRequest::setRuntimeTopK)
        .def_property("runtime_top_p", &InferenceRequest::getRuntimeTopPUnchecked, &InferenceRequest::setRuntimeTopP)
        .def_property(
            "length_penalty", &InferenceRequest::getLengthPenaltyUnchecked, &InferenceRequest::setLengthPenalty)
        .def_property("repetition_penalty", &InferenceRequest::getRepetitionPenaltyUnchecked,
            &InferenceRequest::setRepetitionPenalty)
        .def_property("min_length", &InferenceRequest::getMinLengthUnchecked, &InferenceRequest::setMinLength)
        .def_property(
            "presence_penalty", &InferenceRequest::getPresencePenaltyUnchecked, &InferenceRequest::setPresencePenalty)
        .def_property("random_seed", &InferenceRequest::getRandomSeedUnchecked, &InferenceRequest::setRandomSeed)
        .def_property(
            "return_log_probs", &InferenceRequest::getReturnLogProbsUnchecked, &InferenceRequest::setReturnLogProbs)
        .def_property("prompt_embedding_table", &InferenceRequest::getPromptEmbeddingTableUnchecked,
            &InferenceRequest::setPromptEmbeddingTable)
        .def_property(
            "prompt_vocab_size", &InferenceRequest::getPromptVocabSizeUnchecked, &InferenceRequest::setPromptVocabSize)
        .def_property("is_streaming", &InferenceRequest::isStreaming, &InferenceRequest::setIsStreaming)
        .def_property_readonly("request_id", &InferenceRequest::getRequestId);
}
