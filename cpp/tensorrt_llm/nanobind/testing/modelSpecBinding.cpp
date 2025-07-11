/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "modelSpecBinding.h"
#include "tensorrt_llm/testing/modelSpec.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using tensorrt_llm::testing::ModelSpec;
using tensorrt_llm::testing::KVCacheType;
using tensorrt_llm::testing::QuantMethod;
using tensorrt_llm::testing::OutputContentType;

namespace tensorrt_llm::pybind::testing
{

void initBindings(py::module_& m)
{
    py::enum_<QuantMethod>(m, "QuantMethod", py::arithmetic(), "Quantization Method")
        .value("NONE", QuantMethod::kNONE, "No Quantization")
        .value("SMOOTH_QUANT", QuantMethod::kSMOOTH_QUANT, "Smooth Quantization");

    py::enum_<OutputContentType>(m, "OutputContentType", py::arithmetic(), "Output Content Type")
        .value("NONE", OutputContentType::kNONE, "No Output Content")
        .value("CONTEXT_LOGITS", OutputContentType::kCONTEXT_LOGITS, "Context Logits")
        .value("GENERATION_LOGITS", OutputContentType::kGENERATION_LOGITS, "Generation Logits")
        .value("LOG_PROBS", OutputContentType::kLOG_PROBS, "Log Probs")
        .value("CUM_LOG_PROBS", OutputContentType::kCUM_LOG_PROBS, "Cumulative Log");

    py::class_<ModelSpec>(m, "ModelSpec")
        .def(py::init<std::string const&, nvinfer1::DataType>())
        .def("use_gpt_plugin", &ModelSpec::useGptAttentionPlugin)
        .def("use_packed_input", &ModelSpec::usePackedInput)
        .def("set_kv_cache_type", &ModelSpec::setKVCacheType)
        .def("use_decoder_per_request", &ModelSpec::useDecoderPerRequest)
        .def("use_tensor_parallelism", &ModelSpec::useTensorParallelism)
        .def("use_pipeline_parallelism", &ModelSpec::usePipelineParallelism)
        .def("use_context_parallelism", &ModelSpec::useContextParallelism)
        .def("set_draft_tokens", &ModelSpec::setDraftTokens)
        .def("use_accept_by_logits", &ModelSpec::useAcceptByLogits)
        .def("use_mamba_plugin", &ModelSpec::useMambaPlugin)
        .def("gather_logits", &ModelSpec::gatherLogits)
        .def("replace_logits", &ModelSpec::replaceLogits)
        .def("return_log_probs", &ModelSpec::returnLogProbs)
        .def("smoke_test", &ModelSpec::smokeTest)
        .def("use_medusa", &ModelSpec::useMedusa)
        .def("use_eagle", &ModelSpec::useEagle)
        .def("use_lookahead_decoding", &ModelSpec::useLookaheadDecoding)
        .def("use_explicit_draft_tokens_decoding", &ModelSpec::useExplicitDraftTokensDecoding)
        .def("use_draft_tokens_external_decoding", &ModelSpec::useDraftTokensExternalDecoding)
        .def("use_logits", &ModelSpec::useLogits)
        .def("use_multiple_profiles", &ModelSpec::useMultipleProfiles)
        .def("set_max_input_length", &ModelSpec::setMaxInputLength)
        .def("set_max_output_length", &ModelSpec::setMaxOutputLength)
        .def("set_quant_method", &ModelSpec::setQuantMethod)
        .def("use_lora_plugin", &ModelSpec::useLoraPlugin)
        .def("get_input_file", &ModelSpec::getInputFile)
        .def("get_model_path", &ModelSpec::getModelPath)
        .def("get_results_file", &ModelSpec::getResultsFile)
        .def("get_generation_logits_file", &ModelSpec::getGenerationLogitsFile)
        .def("get_context_logits_file", &ModelSpec::getContextLogitsFile)
        .def("get_cum_log_probs_file", &ModelSpec::getCumLogProbsFile)
        .def("get_log_probs_file", &ModelSpec::getLogProbsFile)
        .def("enable_context_fmha_fp32_acc", &ModelSpec::enableContextFMHAFp32Acc)
        .def("get_enable_context_fmha_fp32_acc", &ModelSpec::getEnableContextFMHAFp32Acc)
        .def("__copy__", [](ModelSpec const& self) { return ModelSpec(self); });
}

} // namespace tensorrt_llm::pybind::testing
