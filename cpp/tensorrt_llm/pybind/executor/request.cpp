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

#include "request.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "streamCaster.h"
#include "tensorCaster.h"

#include <optional>
#include <vector>

namespace py = pybind11;
namespace tle = tensorrt_llm::executor;
using Tensor = tle::Tensor;
using SizeType32 = tle::SizeType32;
using FloatType = tle::FloatType;
using VecTokens = tle::VecTokens;
using IdType = tle::IdType;
using VecTokenExtraIds = tle::VecTokenExtraIds;

namespace tensorrt_llm::pybind::executor
{

void initRequestBindings(pybind11::module_& m)
{
    py::enum_<tle::RequestType>(m, "RequestType")
        .value("REQUEST_TYPE_CONTEXT_AND_GENERATION", tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION)
        .value("REQUEST_TYPE_CONTEXT_ONLY", tle::RequestType::REQUEST_TYPE_CONTEXT_ONLY)
        .value("REQUEST_TYPE_GENERATION_ONLY", tle::RequestType::REQUEST_TYPE_GENERATION_ONLY);

    py::enum_<tle::FinishReason>(m, "FinishReason")
        .value("NOT_FINISHED", tle::FinishReason::kNOT_FINISHED)
        .value("END_ID", tle::FinishReason::kEND_ID)
        .value("STOP_WORDS", tle::FinishReason::kSTOP_WORDS)
        .value("LENGTH", tle::FinishReason::kLENGTH)
        .value("TIMED_OUT", tle::FinishReason::kTIMED_OUT)
        .value("CANCELLED", tle::FinishReason::kCANCELLED);

    auto samplingConfigGetstate = [](tle::SamplingConfig const& self)
    {
        return py::make_tuple(self.getBeamWidth(), self.getTopK(), self.getTopP(), self.getTopPMin(),
            self.getTopPResetIds(), self.getTopPDecay(), self.getSeed(), self.getTemperature(), self.getMinTokens(),
            self.getBeamSearchDiversityRate(), self.getRepetitionPenalty(), self.getPresencePenalty(),
            self.getFrequencyPenalty(), self.getLengthPenalty(), self.getEarlyStopping(), self.getNoRepeatNgramSize(),
            self.getNumReturnSequences(), self.getMinP());
    };
    auto samplingConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 18)
        {
            throw std::runtime_error("Invalid SamplingConfig state!");
        }
        return tle::SamplingConfig(state[0].cast<SizeType32>(), state[1].cast<std::optional<SizeType32>>(),
            state[2].cast<std::optional<FloatType>>(), state[3].cast<std::optional<FloatType>>(),
            state[4].cast<std::optional<tle::TokenIdType>>(), state[5].cast<std::optional<FloatType>>(),
            state[6].cast<std::optional<tle::RandomSeedType>>(), state[7].cast<std::optional<FloatType>>(),
            state[8].cast<std::optional<SizeType32>>(), state[9].cast<std::optional<FloatType>>(),
            state[10].cast<std::optional<FloatType>>(), state[11].cast<std::optional<FloatType>>(),
            state[12].cast<std::optional<FloatType>>(), state[13].cast<std::optional<FloatType>>(),
            state[14].cast<std::optional<SizeType32>>(), state[15].cast<std::optional<SizeType32>>(),
            state[16].cast<std::optional<SizeType32>>(), state[17].cast<std::optional<FloatType>>());
    };
    py::class_<tle::SamplingConfig>(m, "SamplingConfig")
        // A modified version of constructor to accpect deprecated args randomSeed and minLength
        // TODO(enweiz): use the original constructor after the deprecated args are removed
        .def(
            py::init(
                [](tle::SizeType32 beamWidth, std::optional<tle::SizeType32> const& topK,
                    std::optional<tle::FloatType> const& topP, std::optional<tle::FloatType> const& topPMin,
                    std::optional<tle::TokenIdType> const& topPResetIds, std::optional<tle::FloatType> const& topPDecay,
                    std::optional<tle::RandomSeedType> seed, std::optional<tle::RandomSeedType> const& randomSeed,
                    std::optional<tle::FloatType> const& temperature, std::optional<tle::SizeType32> minTokens,
                    std::optional<tle::SizeType32> const& minLength,
                    std::optional<tle::FloatType> const& beamSearchDiversityRate,
                    std::optional<tle::FloatType> const& repetitionPenalty,
                    std::optional<tle::FloatType> const& presencePenalty,
                    std::optional<tle::FloatType> const& frequencyPenalty,
                    std::optional<tle::FloatType> const& lengthPenalty,
                    std::optional<tle::SizeType32> const& earlyStopping,
                    std::optional<tle::SizeType32> const& noRepeatNgramSize,
                    std::optional<tle::SizeType32> const& numReturnSequences, std::optional<tle::FloatType> const& minP)
                {
                    if (randomSeed.has_value())
                    {
                        TLLM_LOG_WARNING("random_seed is being deprecated; please use seed instead.");
                        if (!seed.has_value())
                        {
                            seed = randomSeed;
                        }
                    }
                    if (minLength.has_value())
                    {
                        TLLM_LOG_WARNING("min_length is being deprecated; please use min_tokens instead.");
                        if (!minTokens.has_value())
                        {
                            minTokens = minLength;
                        }
                    }
                    return std::make_unique<tle::SamplingConfig>(beamWidth, topK, topP, topPMin, topPResetIds,
                        topPDecay, seed, temperature, minTokens, beamSearchDiversityRate, repetitionPenalty,
                        presencePenalty, frequencyPenalty, lengthPenalty, earlyStopping, noRepeatNgramSize,
                        numReturnSequences, minP);
                }),
            py::arg("beam_width") = 1, py::kw_only(), py::arg("top_k") = py::none(), py::arg("top_p") = py::none(),
            py::arg("top_p_min") = py::none(), py::arg("top_p_reset_ids") = py::none(),
            py::arg("top_p_decay") = py::none(), py::arg("seed") = py::none(), py::arg("random_seed") = py::none(),
            py::arg("temperature") = py::none(), py::arg("min_tokens") = py::none(), py::arg("min_length") = py::none(),
            py::arg("beam_search_diversity_rate") = py::none(), py::arg("repetition_penalty") = py::none(),
            py::arg("presence_penalty") = py::none(), py::arg("frequency_penalty") = py::none(),
            py::arg("length_penalty") = py::none(), py::arg("early_stopping") = py::none(),
            py::arg("no_repeat_ngram_size") = py::none(), py::arg("num_return_sequences") = py::none(),
            py::arg("min_p") = py::none())
        .def_property("beam_width", &tle::SamplingConfig::getBeamWidth, &tle::SamplingConfig::setBeamWidth)
        .def_property("top_k", &tle::SamplingConfig::getTopK, &tle::SamplingConfig::setTopK)
        .def_property("top_p", &tle::SamplingConfig::getTopP, &tle::SamplingConfig::setTopP)
        .def_property("top_p_min", &tle::SamplingConfig::getTopPMin, &tle::SamplingConfig::setTopPMin)
        .def_property("top_p_reset_ids", &tle::SamplingConfig::getTopPResetIds, &tle::SamplingConfig::setTopPResetIds)
        .def_property("top_p_decay", &tle::SamplingConfig::getTopPDecay, &tle::SamplingConfig::setTopPDecay)
        .def_property("seed", &tle::SamplingConfig::getSeed, &tle::SamplingConfig::setSeed)
        .def_property("random_seed", &tle::SamplingConfig::getRandomSeed, &tle::SamplingConfig::setRandomSeed)
        .def_property("temperature", &tle::SamplingConfig::getTemperature, &tle::SamplingConfig::setTemperature)
        .def_property("min_tokens", &tle::SamplingConfig::getMinTokens, &tle::SamplingConfig::setMinTokens)
        .def_property("min_length", &tle::SamplingConfig::getMinLength, &tle::SamplingConfig::setMinLength)
        .def_property("beam_search_diversity_rate", &tle::SamplingConfig::getBeamSearchDiversityRate,
            &tle::SamplingConfig::setBeamSearchDiversityRate)
        .def_property("repetition_penalty", &tle::SamplingConfig::getRepetitionPenalty,
            &tle::SamplingConfig::setRepetitionPenalty)
        .def_property("presence_penalty", &tle::SamplingConfig::getPresencePenalty,
            [](tle::SamplingConfig& self, std::optional<FloatType> v) { self.setPresencePenalty(v); })
        .def_property(
            "frequency_penalty", &tle::SamplingConfig::getFrequencyPenalty, &tle::SamplingConfig::setFrequencyPenalty)
        .def_property("length_penalty", &tle::SamplingConfig::getLengthPenalty, &tle::SamplingConfig::setLengthPenalty)
        .def_property("early_stopping", &tle::SamplingConfig::getEarlyStopping, &tle::SamplingConfig::setEarlyStopping)
        .def_property("no_repeat_ngram_size", &tle::SamplingConfig::getNoRepeatNgramSize,
            &tle::SamplingConfig::setNoRepeatNgramSize)
        .def_property("num_return_sequences", &tle::SamplingConfig::getNumReturnSequences,
            &tle::SamplingConfig::setNumReturnSequences)
        .def_property("min_p", &tle::SamplingConfig::getMinP, &tle::SamplingConfig::setMinP)
        .def(py::pickle(samplingConfigGetstate, samplingConfigSetstate));

    auto additionalModelOutputGetstate = [](tle::OutputConfig::AdditionalModelOutput const& self)
    { return py::make_tuple(self.name, self.gatherContext); };
    auto additionalModelOutputSetstate = [](py::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid AdditionalModelOutput state!");
        }
        return tle::OutputConfig::AdditionalModelOutput(state[0].cast<std::string>(), state[1].cast<bool>());
    };
    py::class_<tle::OutputConfig::AdditionalModelOutput>(m, "AdditionalModelOutput")
        .def(py::init<std::string, bool>(), py::arg("name"), py::arg("gather_context") = false)
        .def_readwrite("name", &tle::OutputConfig::AdditionalModelOutput::name)
        .def_readwrite("gather_context", &tle::OutputConfig::AdditionalModelOutput::gatherContext)
        .def(py::pickle(additionalModelOutputGetstate, additionalModelOutputSetstate));

    auto outputConfigGetstate = [](tle::OutputConfig const& self)
    {
        return py::make_tuple(self.returnLogProbs, self.returnContextLogits, self.returnGenerationLogits,
            self.excludeInputFromOutput, self.returnEncoderOutput, self.returnPerfMetrics, self.additionalModelOutputs);
    };
    auto outputConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 7)
        {
            throw std::runtime_error("Invalid OutputConfig state!");
        }
        return tle::OutputConfig(state[0].cast<bool>(), state[1].cast<bool>(), state[2].cast<bool>(),
            state[3].cast<bool>(), state[4].cast<bool>(), state[5].cast<bool>(),
            state[6].cast<std::optional<std::vector<tle::OutputConfig::AdditionalModelOutput>>>());
    };
    py::class_<tle::OutputConfig>(m, "OutputConfig")
        .def(py::init<bool, bool, bool, bool, bool, bool,
                 std::optional<std::vector<tle::OutputConfig::AdditionalModelOutput>>>(),
            py::arg("return_log_probs") = false, py::arg("return_context_logits") = false,
            py::arg("return_generation_logits") = false, py::arg("exclude_input_from_output") = false,
            py::arg("return_encoder_output") = false, py::arg("return_perf_metrics") = false,
            py::arg("additional_model_outputs") = py::none())
        .def_readwrite("return_log_probs", &tle::OutputConfig::returnLogProbs)
        .def_readwrite("return_context_logits", &tle::OutputConfig::returnContextLogits)
        .def_readwrite("return_generation_logits", &tle::OutputConfig::returnGenerationLogits)
        .def_readwrite("exclude_input_from_output", &tle::OutputConfig::excludeInputFromOutput)
        .def_readwrite("return_encoder_output", &tle::OutputConfig::returnEncoderOutput)
        .def_readwrite("return_perf_metrics", &tle::OutputConfig::returnPerfMetrics)
        .def_readwrite("additional_model_outputs", &tle::OutputConfig::additionalModelOutputs)
        .def(py::pickle(outputConfigGetstate, outputConfigSetstate));

    auto externalDraftTokensConfigGetstate = [](tle::ExternalDraftTokensConfig const& self)
    { return py::make_tuple(self.getTokens(), self.getLogits(), self.getAcceptanceThreshold()); };
    auto externalDraftTokensConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid ExternalDraftTokensConfig state!");
        }
        return tle::ExternalDraftTokensConfig(state[0].cast<VecTokens>(), state[1].cast<std::optional<Tensor>>(),
            state[2].cast<std::optional<FloatType>>());
    };
    py::class_<tle::ExternalDraftTokensConfig>(m, "ExternalDraftTokensConfig")
        .def(py::init<VecTokens, std::optional<Tensor>, std::optional<FloatType> const&, std::optional<bool>>(),
            py::arg("tokens"), py::arg("logits") = py::none(), py::arg("acceptance_threshold") = py::none(),
            py::arg("fast_logits") = py::none())
        .def_property_readonly("tokens", &tle::ExternalDraftTokensConfig::getTokens)
        .def_property_readonly("logits", &tle::ExternalDraftTokensConfig::getLogits)
        .def_property_readonly("acceptance_threshold", &tle::ExternalDraftTokensConfig::getAcceptanceThreshold)
        .def(py::pickle(externalDraftTokensConfigGetstate, externalDraftTokensConfigSetstate))
        .def_property_readonly("fast_logits", &tle::ExternalDraftTokensConfig::getFastLogits);

    auto promptTuningConfigGetstate = [](tle::PromptTuningConfig const& self)
    { return py::make_tuple(self.getEmbeddingTable(), self.getInputTokenExtraIds()); };
    auto promptTuningConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid PromptTuningConfig state!");
        }
        return tle::PromptTuningConfig(state[0].cast<Tensor>(), state[1].cast<std::optional<VecTokenExtraIds>>());
    };
    py::class_<tle::PromptTuningConfig>(m, "PromptTuningConfig")
        .def(py::init<Tensor, std::optional<VecTokenExtraIds>>(), py::arg("embedding_table"),
            py::arg("input_token_extra_ids") = py::none())
        .def_property_readonly("embedding_table", &tle::PromptTuningConfig::getEmbeddingTable)
        .def_property_readonly("input_token_extra_ids", &tle::PromptTuningConfig::getInputTokenExtraIds)
        .def(py::pickle(promptTuningConfigGetstate, promptTuningConfigSetstate));

    auto loraConfigGetstate = [](tle::LoraConfig const& self)
    { return py::make_tuple(self.getTaskId(), self.getWeights(), self.getConfig()); };
    auto loraConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LoraConfig state!");
        }
        return tle::LoraConfig(
            state[0].cast<IdType>(), state[1].cast<std::optional<Tensor>>(), state[2].cast<std::optional<Tensor>>());
    };
    py::class_<tle::LoraConfig>(m, "LoraConfig")
        .def(py::init<uint64_t, std::optional<Tensor>, std::optional<Tensor>>(), py::arg("task_id"),
            py::arg("weights") = py::none(), py::arg("config") = py::none())
        .def_property_readonly("task_id", &tle::LoraConfig::getTaskId)
        .def_property_readonly("weights", &tle::LoraConfig::getWeights)
        .def_property_readonly("config", &tle::LoraConfig::getConfig)
        .def(py::pickle(loraConfigGetstate, loraConfigSetstate));

    auto MropeConfigGetstate = [](tle::MropeConfig const& self)
    { return py::make_tuple(self.getMRopeRotaryCosSin(), self.getMRopePositionDeltas()); };
    auto MropeConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid MropeConfig state!");
        }
        return tle::MropeConfig(state[0].cast<tle::Tensor>(), state[1].cast<SizeType32>());
    };
    py::class_<tle::MropeConfig>(m, "MropeConfig")
        .def(py::init<Tensor, SizeType32>(), py::arg("mrope_rotary_cos_sin"), py::arg("mrope_position_deltas"))
        .def_property_readonly("mrope_rotary_cos_sin", &tle::MropeConfig::getMRopeRotaryCosSin)
        .def_property_readonly("mrope_position_deltas", &tle::MropeConfig::getMRopePositionDeltas)
        .def(py::pickle(MropeConfigGetstate, MropeConfigSetstate));

    auto lookaheadDecodingConfigGetstate = [](tle::LookaheadDecodingConfig const& self)
    { return py::make_tuple(self.getWindowSize(), self.getNgramSize(), self.getVerificationSetSize()); };
    auto lookaheadDecodingConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LookaheadDecodingConfig state!");
        }
        return tle::LookaheadDecodingConfig(
            state[0].cast<SizeType32>(), state[1].cast<SizeType32>(), state[2].cast<SizeType32>());
    };
    py::class_<tle::LookaheadDecodingConfig>(m, "LookaheadDecodingConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32>(), py::arg("max_window_size"), py::arg("max_ngram_size"),
            py::arg("max_verification_set_size"))
        .def_property_readonly("max_window_size", &tle::LookaheadDecodingConfig::getWindowSize)
        .def_property_readonly("max_ngram_size", &tle::LookaheadDecodingConfig::getNgramSize)
        .def_property_readonly("max_verification_set_size", &tle::LookaheadDecodingConfig::getVerificationSetSize)
        .def("calculate_speculative_resource", &tle::LookaheadDecodingConfig::calculateSpeculativeResource)
        .def_static(
            "calculate_speculative_resource_tuple", &tle::LookaheadDecodingConfig::calculateSpeculativeResourceTuple)
        .def(py::pickle(lookaheadDecodingConfigGetstate, lookaheadDecodingConfigSetstate))
        .def_static("get_default_lookahead_decoding_window",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingWindow; })
        .def_static("get_default_lookahead_decoding_ngram",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingNgram; })
        .def_static("get_default_lookahead_decoding_verification_set",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingVerificationSet; });

    auto TokenRangeRetentionConfigGetstate = [](tle::KvCacheRetentionConfig::TokenRangeRetentionConfig const& self)
    { return py::make_tuple(self.tokenStart, self.tokenEnd, self.priority, self.durationMs); };
    auto TokenRangeRetentionConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(state[0].cast<SizeType32>(),
            state[1].cast<std::optional<SizeType32>>(), state[2].cast<tle::RetentionPriority>(),
            state[3].cast<std::optional<std::chrono::milliseconds>>());
    };
    auto kvCacheRetentionConfigGetstate = [](tle::KvCacheRetentionConfig const& self)
    {
        return py::make_tuple(
            self.getTokenRangeRetentionConfigs(), self.getDecodeRetentionPriority(), self.getDecodeDurationMs());
    };
    auto kvCacheRetentionConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::KvCacheRetentionConfig(
            state[0].cast<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>>(),
            state[1].cast<tle::RetentionPriority>(), state[2].cast<std::optional<std::chrono::milliseconds>>());
    };

    auto kvCacheRetentionConfig = py::class_<tle::KvCacheRetentionConfig>(m, "KvCacheRetentionConfig");

    py::class_<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>(
        kvCacheRetentionConfig, "TokenRangeRetentionConfig")
        .def(py::init<SizeType32, std::optional<SizeType32>, tle::RetentionPriority,
                 std::optional<std::chrono::milliseconds>>(),
            py::arg("token_start"), py::arg("token_end"), py::arg("priority"), py::arg("duration_ms") = py::none())
        .def_readwrite("token_start", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::tokenStart)
        .def_readwrite("token_end", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::tokenEnd)
        .def_readwrite("priority", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::priority)
        .def_readwrite("duration_ms", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::durationMs)
        .def(py::pickle(TokenRangeRetentionConfigGetstate, TokenRangeRetentionConfigSetstate))
        .def("__eq__", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::operator==);

    // There's a circular dependency between the declaration of the TokenRangeRetentionPriority and
    // KvCacheRetentionConfig bindings. Defer definition of the KvCacheRetentionConfig bindings until the
    // TokenRangeRetentionPriority bindings have been defined.
    kvCacheRetentionConfig
        .def(py::init<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>, tle::RetentionPriority,
                 std::optional<std::chrono::milliseconds>>(),
            py::arg("token_range_retention_configs"),
            py::arg("decode_retention_priority") = tle::KvCacheRetentionConfig::kDefaultRetentionPriority,
            py::arg("decode_duration_ms") = py::none())
        .def_property_readonly(
            "token_range_retention_configs", &tle::KvCacheRetentionConfig::getTokenRangeRetentionConfigs)
        .def_property_readonly("decode_retention_priority", &tle::KvCacheRetentionConfig::getDecodeRetentionPriority)
        .def_property_readonly("decode_duration_ms", &tle::KvCacheRetentionConfig::getDecodeDurationMs)
        .def(py::pickle(kvCacheRetentionConfigGetstate, kvCacheRetentionConfigSetstate))
        .def("__eq__", &tle::KvCacheRetentionConfig::operator==);

    auto ContextPhaseParamsGetState = [](tle::ContextPhaseParams const& self)
    {
        auto serializedState = self.getSerializedState();
        return py::make_tuple(self.getFirstGenTokens(), self.getReqId(),
            py::bytes(serializedState.data(), serializedState.size()), self.getDraftTokens());
    };

    auto ContextPhaseParamsSetState = [](py::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid ContextPhaseParams state!");
        }
        auto opaque_state = state[2].cast<py::bytes>();
        auto opaque_state_str_view = std::string_view(opaque_state.cast<std::string_view>());
        return std::make_unique<tle::ContextPhaseParams>(state[0].cast<VecTokens>(),
            state[1].cast<tle::ContextPhaseParams::RequestIdType>(),
            std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()),
            state[3].cast<std::optional<VecTokens>>());
    };

    py::class_<tle::ContextPhaseParams>(m, "ContextPhaseParams")
        .def(py::init(
            [](VecTokens const& first_gen_tokens, tle::ContextPhaseParams::RequestIdType req_id,
                py::bytes const& opaque_state, std::optional<VecTokens> const& draft_tokens)
            {
                auto opaque_state_str_view = std::string_view(opaque_state.cast<std::string_view>());
                return std::make_unique<tle::ContextPhaseParams>(first_gen_tokens, req_id,
                    std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()), draft_tokens);
            }))
        .def_property_readonly("first_gen_tokens", &tle::ContextPhaseParams::getFirstGenTokens)
        .def_property_readonly("draft_tokens", &tle::ContextPhaseParams::getDraftTokens)
        .def_property_readonly("req_id", &tle::ContextPhaseParams::getReqId)
        .def_property_readonly("opaque_state",
            [](tle::ContextPhaseParams const& self)
            {
                auto serializedState = self.getSerializedState();
                return py::bytes(serializedState.data(), serializedState.size());
            })
        .def(py::pickle(ContextPhaseParamsGetState, ContextPhaseParamsSetState));

    auto EagleDecodingConfigGetstate = [](tle::EagleConfig const& self)
    {
        return py::make_tuple(self.getEagleChoices(), self.isGreedySampling(), self.getPosteriorThreshold(),
            self.useDynamicTree(), self.getDynamicTreeMaxTopK());
    };
    auto EagleDecodingConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid EagleConfig state!");
        }
        return tle::EagleConfig(state[0].cast<tle::EagleChoices>(), state[1].cast<bool>(),
            state[2].cast<std::optional<float>>(), state[3].cast<bool>(), state[4].cast<std::optional<SizeType32>>());
    };
    py::class_<tle::EagleConfig>(m, "EagleConfig")
        .def(py::init<std::optional<tle::EagleChoices>, bool, std::optional<float>, bool, std::optional<SizeType32>>(),
            py::arg("eagle_choices") = py::none(), py::arg("greedy_sampling") = true,
            py::arg("posterior_threshold") = py::none(), py::arg("use_dynamic_tree") = false,
            py::arg("dynamic_tree_max_topK") = py::none())
        .def_property_readonly("eagle_choices", &tle::EagleConfig::getEagleChoices)
        .def_property_readonly("greedy_sampling", &tle::EagleConfig::isGreedySampling)
        .def_property_readonly("posterior_threshold", &tle::EagleConfig::getPosteriorThreshold)
        .def_property_readonly("use_dynamic_tree", &tle::EagleConfig::useDynamicTree)
        .def_property_readonly("dynamic_tree_max_topK", &tle::EagleConfig::getDynamicTreeMaxTopK)
        .def(py::pickle(EagleDecodingConfigGetstate, EagleDecodingConfigSetstate));

    // Guided decoding params
    auto pyGuidedDecodingParams = py::class_<tle::GuidedDecodingParams>(m, "GuidedDecodingParams");

    py::enum_<tle::GuidedDecodingParams::GuideType>(pyGuidedDecodingParams, "GuideType")
        .value("JSON", tle::GuidedDecodingParams::GuideType::kJSON)
        .value("JSON_SCHEMA", tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA)
        .value("REGEX", tle::GuidedDecodingParams::GuideType::kREGEX)
        .value("EBNF_GRAMMAR", tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR);

    auto guidedDecodingParamsGetstate
        = [](tle::GuidedDecodingParams const& self) { return py::make_tuple(self.getGuideType(), self.getGuide()); };

    auto guidedDecodingParamsSetstate = [](py::tuple state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid GuidedDecodingParams state!");
        }
        return tle::GuidedDecodingParams(
            state[0].cast<tle::GuidedDecodingParams::GuideType>(), state[1].cast<std::optional<std::string>>());
    };

    pyGuidedDecodingParams
        .def(py::init<tle::GuidedDecodingParams::GuideType, std::optional<std::string>>(), py::arg("guide_type"),
            py::arg("guide") = py::none())
        .def_property_readonly("guide_type", &tle::GuidedDecodingParams::getGuideType)
        .def_property_readonly("guide", &tle::GuidedDecodingParams::getGuide)
        .def(py::pickle(guidedDecodingParamsGetstate, guidedDecodingParamsSetstate));

    auto requestGetstate = [](tle::Request const& self)
    {
        return py::make_tuple(self.getInputTokenIds(), self.getMaxTokens(), self.getStreaming(),
            self.getSamplingConfig(), self.getOutputConfig(), self.getEndId(), self.getPadId(), self.getPositionIds(),
            self.getBadWords(), self.getStopWords(), self.getEmbeddingBias(), self.getExternalDraftTokensConfig(),
            self.getPromptTuningConfig(), self.getMropeConfig(), self.getLoraConfig(), self.getLookaheadConfig(),
            self.getKvCacheRetentionConfig(), self.getLogitsPostProcessorName(), self.getLogitsPostProcessor(),
            self.getEncoderInputTokenIds(), self.getClientId(), self.getReturnAllGeneratedTokens(), self.getPriority(),
            self.getRequestType(), self.getContextPhaseParams(), self.getEncoderInputFeatures(),
            self.getEncoderOutputLength(), self.getCrossAttentionMask(), self.getEagleConfig(),
            self.getSkipCrossAttnBlocks(), self.getGuidedDecodingParams());
    };
    auto requestSetstate = [](py::tuple const& state)
    {
        if (state.size() != 31)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        return std::make_unique<tle::Request>(state[0].cast<VecTokens>(), state[1].cast<SizeType32>(),
            state[2].cast<bool>(), state[3].cast<tle::SamplingConfig>(), state[4].cast<tle::OutputConfig>(),
            state[5].cast<std::optional<SizeType32>>(), state[6].cast<std::optional<SizeType32>>(),
            state[7].cast<std::optional<std::vector<SizeType32>>>(),
            state[8].cast<std::optional<std::list<VecTokens>>>(), state[9].cast<std::optional<std::list<VecTokens>>>(),
            state[10].cast<std::optional<Tensor>>(), state[11].cast<std::optional<tle::ExternalDraftTokensConfig>>(),
            state[12].cast<std::optional<tle::PromptTuningConfig>>(), state[13].cast<std::optional<tle::MropeConfig>>(),
            state[14].cast<std::optional<tle::LoraConfig>>(),
            state[15].cast<std::optional<tle::LookaheadDecodingConfig>>(),
            state[16].cast<std::optional<tle::KvCacheRetentionConfig>>(), state[17].cast<std::optional<std::string>>(),
            state[18].cast<std::optional<tle::LogitsPostProcessor>>(), state[19].cast<std::optional<VecTokens>>(),
            state[20].cast<std::optional<IdType>>(), state[21].cast<bool>(), state[22].cast<tle::PriorityType>(),
            state[23].cast<tle::RequestType>(), state[24].cast<std::optional<tle::ContextPhaseParams>>(),
            state[25].cast<std::optional<tle::Tensor>>(), state[26].cast<std::optional<SizeType32>>(),
            state[27].cast<std::optional<tle::Tensor>>(), 1, state[28].cast<std::optional<tle::EagleConfig>>(),
            state[29].cast<std::optional<tle::Tensor>>(), state[30].cast<std::optional<tle::GuidedDecodingParams>>());
    };

    py::class_<tle::Request> request(m, "Request");
    request
        // A modified version of constructor to accpect deprecated args maxNewTokens
        // TODO(enweiz): use the original constructor after the deprecated args are removed
        .def(py::init(
                 [](tle::VecTokens const& inputTokenIds, std::optional<tle::SizeType32> maxTokens,
                     std::optional<tle::SizeType32> maxNewTokens, bool streaming,
                     tle::SamplingConfig const& samplingConfig, tle::OutputConfig const& outputConfig,
                     std::optional<tle::SizeType32> const& endId, std::optional<tle::SizeType32> const& padId,
                     std::optional<std::vector<SizeType32>> const& positionIds,
                     std::optional<std::list<tle::VecTokens>> const& badWords,
                     std::optional<std::list<tle::VecTokens>> const& stopWords,
                     std::optional<tle::Tensor> const& embeddingBias,
                     std::optional<tle::ExternalDraftTokensConfig> const& externalDraftTokensConfig,
                     std::optional<tle::PromptTuningConfig> const& pTuningConfig,
                     std::optional<tle::MropeConfig> const& mRopeConfig,
                     std::optional<tle::LoraConfig> const& loraConfig,
                     std::optional<tle::LookaheadDecodingConfig> lookaheadConfig,
                     std::optional<tle::KvCacheRetentionConfig> const& kvCacheRetentionConfig,
                     std::optional<std::string> const& logitsPostProcessorName,
                     std::optional<tle::LogitsPostProcessor> const& logitsPostProcessor,
                     std::optional<tle::VecTokens> const& encoderInputTokenIds, std::optional<tle::IdType> clientId,
                     bool returnAllGeneratedTokens, tle::PriorityType priority, tle::RequestType type,
                     std::optional<tle::ContextPhaseParams> const& contextPhaseParams,
                     std::optional<tle::Tensor> const& encoderInputFeatures,
                     std::optional<tle::SizeType32> encoderOutputLength,
                     std::optional<tle::Tensor> const& crossAttentionMask,
                     std::optional<tle::EagleConfig> const& eagleConfig,
                     std::optional<tle::Tensor> const& skipCrossAttnBlocks,
                     std::optional<tle::GuidedDecodingParams> const& guidedDecodingParams,
                     std::optional<tle::SizeType32> const& languageAdapterUid)
                 {
                     if (maxNewTokens.has_value())
                     {
                         TLLM_LOG_WARNING("max_new_tokens is being deprecated; please use max_tokens instead.");
                         if (!maxTokens.has_value())
                         {
                             maxTokens = maxNewTokens;
                         }
                     }
                     TLLM_CHECK_WITH_INFO(maxTokens.has_value(), "missing required argument max_tokens");
                     return std::make_unique<tle::Request>(inputTokenIds, maxTokens.value(), streaming, samplingConfig,
                         outputConfig, endId, padId, positionIds, badWords, stopWords, embeddingBias,
                         externalDraftTokensConfig, pTuningConfig, mRopeConfig, loraConfig, lookaheadConfig,
                         kvCacheRetentionConfig, logitsPostProcessorName, logitsPostProcessor, encoderInputTokenIds,
                         clientId, returnAllGeneratedTokens, priority, type, contextPhaseParams, encoderInputFeatures,
                         encoderOutputLength, crossAttentionMask, 1, eagleConfig, skipCrossAttnBlocks,
                         guidedDecodingParams, languageAdapterUid);
                 }),
            py::arg("input_token_ids"), py::kw_only(), py::arg("max_tokens") = py::none(),
            py::arg("max_new_tokens") = py::none(), py::arg("streaming") = false,
            py::arg_v("sampling_config", tle::SamplingConfig(), "SamplingConfig()"),
            py::arg_v("output_config", tle::OutputConfig(), "OutputConfig()"), py::arg("end_id") = py::none(),
            py::arg("pad_id") = py::none(), py::arg("position_ids") = py::none(), py::arg("bad_words") = py::none(),
            py::arg("stop_words") = py::none(), py::arg("embedding_bias") = py::none(),
            py::arg("external_draft_tokens_config") = py::none(), py::arg("prompt_tuning_config") = py::none(),
            py::arg("mrope_config") = py::none(), py::arg("lora_config") = py::none(),
            py::arg("lookahead_config") = py::none(), py::arg("kv_cache_retention_config") = py::none(),
            py::arg("logits_post_processor_name") = py::none(), py::arg("logits_post_processor") = py::none(),
            py::arg("encoder_input_token_ids") = py::none(), py::arg("client_id") = py::none(),
            py::arg("return_all_generated_tokens") = false, py::arg("priority") = tle::Request::kDefaultPriority,
            py::arg_v("type", tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
                "RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION"),
            py::arg("context_phase_params") = py::none(), py::arg("encoder_input_features") = py::none(),
            py::arg("encoder_output_length") = py::none(), py::arg("cross_attention_mask") = py::none(),
            py::arg("eagle_config") = py::none(), py::arg("skip_cross_attn_blocks") = py::none(),
            py::arg("guided_decoding_params") = py::none(), py::arg("language_adapter_uid") = py::none())
        .def_property_readonly("input_token_ids", &tle::Request::getInputTokenIds)
        .def_property_readonly("max_tokens", &tle::Request::getMaxTokens)
        .def_property_readonly("max_new_tokens", &tle::Request::getMaxNewTokens)
        .def_property("streaming", &tle::Request::getStreaming, &tle::Request::setStreaming)
        .def_property("sampling_config", &tle::Request::getSamplingConfig, &tle::Request::setSamplingConfig)
        .def_property("output_config", &tle::Request::getOutputConfig, &tle::Request::setOutputConfig)
        .def_property("end_id", &tle::Request::getEndId, &tle::Request::setEndId)
        .def_property("pad_id", &tle::Request::getPadId, &tle::Request::setPadId)
        .def_property("position_ids", &tle::Request::getPositionIds, &tle::Request::setPositionIds)
        .def_property("bad_words", &tle::Request::getBadWords, &tle::Request::setBadWords)
        .def_property("stop_words", &tle::Request::getStopWords, &tle::Request::setStopWords)
        .def_property("embedding_bias", &tle::Request::getEmbeddingBias, &tle::Request::setEmbeddingBias)
        .def_property("external_draft_tokens_config", &tle::Request::getExternalDraftTokensConfig,
            &tle::Request::setExternalDraftTokensConfig)
        .def_property(
            "prompt_tuning_config", &tle::Request::getPromptTuningConfig, &tle::Request::setPromptTuningConfig)
        .def_property("mrope_config", &tle::Request::getMropeConfig, &tle::Request::setMropeConfig)
        .def_property("lora_config", &tle::Request::getLoraConfig, &tle::Request::setLoraConfig)
        .def_property("lookahead_config", &tle::Request::getLookaheadConfig, &tle::Request::setLookaheadConfig)
        .def_property("kv_cache_retention_config", &tle::Request::getKvCacheRetentionConfig,
            &tle::Request::setKvCacheRetentionConfig)
        .def_property("logits_post_processor_name", &tle::Request::getLogitsPostProcessorName,
            &tle::Request::setLogitsPostProcessorName)
        .def_property(
            "logits_post_processor", &tle::Request::getLogitsPostProcessor, &tle::Request::setLogitsPostProcessor)
        .def_property(
            "encoder_input_token_ids", &tle::Request::getEncoderInputTokenIds, &tle::Request::setEncoderInputTokenIds)
        .def_property("client_id", &tle::Request::getClientId, &tle::Request::setClientId)
        .def_property("return_all_generated_tokens", &tle::Request::getReturnAllGeneratedTokens,
            &tle::Request::setReturnAllGeneratedTokens)
        .def_property("request_type", &tle::Request::getRequestType, &tle::Request::setRequestType)
        .def_property(
            "encoder_input_features", &tle::Request::getEncoderInputFeatures, &tle::Request::setEncoderInputFeatures)
        .def_property(
            "cross_attention_mask", &tle::Request::getCrossAttentionMask, &tle::Request::setCrossAttentionMask)
        .def_property("eagle_config", &tle::Request::getEagleConfig, &tle::Request::setEagleConfig)
        .def_property(
            "skip_cross_attn_blocks", &tle::Request::getSkipCrossAttnBlocks, &tle::Request::setSkipCrossAttnBlocks)
        .def_property(
            "guided_decoding_params", &tle::Request::getGuidedDecodingParams, &tle::Request::setGuidedDecodingParams)
        .def_property("allotted_time_ms", &tle::Request::getAllottedTimeMs, &tle::Request::setAllottedTimeMs)
        .def_property(
            "context_phase_params", &tle::Request::getContextPhaseParams, &tle::Request::setContextPhaseParams)
        .def(py::pickle(requestGetstate, requestSetstate));
    request.attr("BATCHED_POST_PROCESSOR_NAME") = tle::Request::kBatchedPostProcessorName;

    py::class_<tle::SpeculativeDecodingFastLogitsInfo>(m, "SpeculativeDecodingFastLogitsInfo")
        .def(py::init<>())
        .def_readwrite("draft_request_id", &tle::SpeculativeDecodingFastLogitsInfo::draftRequestId)
        .def_readwrite("draft_participant_id", &tle::SpeculativeDecodingFastLogitsInfo::draftParticipantId)
        .def("to_tensor", &tle::SpeculativeDecodingFastLogitsInfo::toTensor);

    auto requestPerfMetrics = py::class_<tle::RequestPerfMetrics>(m, "RequestPerfMetrics");

    py::class_<tle::RequestPerfMetrics::TimingMetrics>(m, "TimingMetrics")
        .def(py::init<>())
        .def_readwrite("arrival_time", &tle::RequestPerfMetrics::TimingMetrics::arrivalTime)
        .def_readwrite("first_scheduled_time", &tle::RequestPerfMetrics::TimingMetrics::firstScheduledTime)
        .def_readwrite("first_token_time", &tle::RequestPerfMetrics::TimingMetrics::firstTokenTime)
        .def_readwrite("last_token_time", &tle::RequestPerfMetrics::TimingMetrics::lastTokenTime)
        .def_readwrite("kv_cache_transfer_start", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferStart)
        .def_readwrite("kv_cache_transfer_end", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferEnd);

    py::class_<tle::RequestPerfMetrics::KvCacheMetrics>(m, "KvCacheMetrics")
        .def(py::init<>())
        .def_readwrite("num_total_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numTotalAllocatedBlocks)
        .def_readwrite("num_new_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numNewAllocatedBlocks)
        .def_readwrite("num_reused_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numReusedBlocks)
        .def_readwrite("num_missed_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numMissedBlocks)
        .def_readwrite("kv_cache_hit_rate", &tle::RequestPerfMetrics::KvCacheMetrics::kvCacheHitRate);

    py::class_<tle::RequestPerfMetrics::SpeculativeDecodingMetrics>(m, "SpeculativeDecodingMetrics")
        .def(py::init<>())
        .def_readwrite("acceptance_rate", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::acceptanceRate)
        .def_readwrite("total_accepted_draft_tokens",
            &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalAcceptedDraftTokens)
        .def_readwrite("total_draft_tokens", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalDraftTokens);

    // There's a circular dependency between the declaration of the TimingMetrics and RequestPerfMetrics bindings.
    // Defer definition of the RequestPerfMetrics bindings until the TimingMetrics have been defined.
    requestPerfMetrics.def(py::init<>())
        .def_readwrite("timing_metrics", &tle::RequestPerfMetrics::timingMetrics)
        .def_readwrite("kv_cache_metrics", &tle::RequestPerfMetrics::kvCacheMetrics)
        .def_readwrite("speculative_decoding", &tle::RequestPerfMetrics::speculativeDecoding)
        .def_readwrite("first_iter", &tle::RequestPerfMetrics::firstIter)
        .def_readwrite("last_iter", &tle::RequestPerfMetrics::lastIter)
        .def_readwrite("iter", &tle::RequestPerfMetrics::iter);

    py::class_<tle::AdditionalOutput>(m, "AdditionalOutput")
        .def(py::init([](std::string const& name, tle::Tensor const& output)
            { return std::make_unique<tle::AdditionalOutput>(name, output); }))
        .def_readwrite("name", &tle::AdditionalOutput::name)
        .def_readwrite("output", &tle::AdditionalOutput::output);

    auto resultSetstate = [](py::tuple const& state)
    {
        if (state.size() != 12)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        tle::Result result;
        result.isFinal = state[0].cast<bool>();
        result.outputTokenIds = state[1].cast<std::vector<VecTokens>>();
        result.cumLogProbs = state[2].cast<std::optional<std::vector<float>>>();
        result.logProbs = state[3].cast<std::optional<std::vector<std::vector<float>>>>();
        result.contextLogits = state[4].cast<std::optional<Tensor>>();
        result.generationLogits = state[5].cast<std::optional<Tensor>>();
        result.encoderOutput = state[6].cast<std::optional<Tensor>>();
        result.finishReasons = state[7].cast<std::vector<tle::FinishReason>>();
        result.sequenceIndex = state[8].cast<SizeType32>();
        result.isSequenceFinal = state[9].cast<bool>();
        result.decodingIter = state[10].cast<SizeType32>();
        result.contextPhaseParams = state[11].cast<std::optional<tle::ContextPhaseParams>>();
        return std::make_unique<tle::Result>(result);
    };

    auto resultGetstate = [](tle::Result const& self)
    {
        return py::make_tuple(self.isFinal, self.outputTokenIds, self.cumLogProbs, self.logProbs, self.contextLogits,
            self.generationLogits, self.encoderOutput, self.finishReasons, self.sequenceIndex, self.isSequenceFinal,
            self.decodingIter, self.contextPhaseParams);
    };

    py::class_<tle::Result>(m, "Result")
        .def(py::init<>())
        .def_readwrite("is_final", &tle::Result::isFinal)
        .def_readwrite("output_token_ids", &tle::Result::outputTokenIds)
        .def_readwrite("cum_log_probs", &tle::Result::cumLogProbs)
        .def_readwrite("log_probs", &tle::Result::logProbs)
        .def_readwrite("context_logits", &tle::Result::contextLogits)
        .def_readwrite("generation_logits", &tle::Result::generationLogits)
        .def_readwrite("spec_dec_fast_logits_info", &tle::Result::specDecFastLogitsInfo)
        .def_readwrite("encoder_output", &tle::Result::encoderOutput)
        .def_readwrite("finish_reasons", &tle::Result::finishReasons)
        .def_readwrite("sequence_index", &tle::Result::sequenceIndex)
        .def_readwrite("is_sequence_final", &tle::Result::isSequenceFinal)
        .def_readwrite("decoding_iter", &tle::Result::decodingIter)
        .def_readwrite("context_phase_params", &tle::Result::contextPhaseParams)
        .def_readwrite("request_perf_metrics", &tle::Result::requestPerfMetrics)
        .def_readwrite("additional_outputs", &tle::Result::additionalOutputs)
        .def_readwrite("context_phase_params", &tle::Result::contextPhaseParams)
        .def(py::pickle(resultGetstate, resultSetstate));

    auto responseGetstate = [](tle::Response const& self)
    { return py::make_tuple(self.getRequestId(), self.getResult(), self.getClientId()); };

    auto responseSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        return std::make_unique<tle::Response>(
            state[0].cast<SizeType32>(), state[1].cast<tle::Result>(), state[2].cast<SizeType32>());
    };

    py::class_<tle::Response>(m, "Response")
        .def(py::init<IdType, std::string, std::optional<IdType>>(), py::arg("request_id"), py::arg("error_msg"),
            py::arg("client_id") = std::nullopt)
        .def(py::init<IdType, tle::Result, std::optional<IdType>>(), py::arg("request_id"), py::arg("result"),
            py::arg("client_id") = std::nullopt)
        .def_property_readonly("request_id", &tle::Response::getRequestId)
        .def_property_readonly("client_id", &tle::Response::getClientId)
        .def("has_error", &tle::Response::hasError)
        .def_property_readonly("error_msg", &tle::Response::getErrorMsg)
        .def_property_readonly("result", &tle::Response::getResult)
        .def(py::pickle(responseGetstate, responseSetstate));
}

} // namespace tensorrt_llm::pybind::executor
