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
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

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

    py::enum_<tle::KvCacheTransferMode>(m, "KvCacheTransferMode")
        .value("DRAM", tle::KvCacheTransferMode::DRAM)
        .value("GDS", tle::KvCacheTransferMode::GDS)
        .value("POSIX_DEBUG_FALLBACK", tle::KvCacheTransferMode::POSIX_DEBUG_FALLBACK);

    auto samplingConfigGetstate = [](tle::SamplingConfig const& self)
    {
        return py::make_tuple(self.getBeamWidth(), self.getTopK(), self.getTopP(), self.getTopPMin(),
            self.getTopPResetIds(), self.getTopPDecay(), self.getSeed(), self.getTemperature(), self.getMinTokens(),
            self.getBeamSearchDiversityRate(), self.getRepetitionPenalty(), self.getPresencePenalty(),
            self.getFrequencyPenalty(), self.getLengthPenalty(), self.getEarlyStopping(), self.getNoRepeatNgramSize(),
            self.getNumReturnSequences(), self.getMinP(), self.getBeamWidthArray());
    };
    auto samplingConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 19)
        {
            throw std::runtime_error("Invalid SamplingConfig state!");
        }
        return tle::SamplingConfig(state[0].cast<SizeType32>(),      // BeamWidth
            state[1].cast<std::optional<SizeType32>>(),              // TopK
            state[2].cast<std::optional<FloatType>>(),               // TopP
            state[3].cast<std::optional<FloatType>>(),               // TopPMin
            state[4].cast<std::optional<tle::TokenIdType>>(),        // TopPResetIds
            state[5].cast<std::optional<FloatType>>(),               // TopPDecay
            state[6].cast<std::optional<tle::RandomSeedType>>(),     // Seed
            state[7].cast<std::optional<FloatType>>(),               // Temperature
            state[8].cast<std::optional<SizeType32>>(),              // MinTokens
            state[9].cast<std::optional<FloatType>>(),               // BeamSearchDiversityRate
            state[10].cast<std::optional<FloatType>>(),              // RepetitionPenalty
            state[11].cast<std::optional<FloatType>>(),              // PresencePenalty
            state[12].cast<std::optional<FloatType>>(),              // FrequencyPenalty
            state[13].cast<std::optional<FloatType>>(),              // LengthPenalty
            state[14].cast<std::optional<SizeType32>>(),             // EarlyStopping
            state[15].cast<std::optional<SizeType32>>(),             // NoRepeatNgramSize
            state[16].cast<std::optional<SizeType32>>(),             // NumReturnSequences
            state[17].cast<std::optional<FloatType>>(),              // MinP
            state[18].cast<std::optional<std::vector<SizeType32>>>() // BeamWidthArray
        );
    };
    py::class_<tle::SamplingConfig>(m, "SamplingConfig")
        .def(py::init<tle::SizeType32,
                 std::optional<tle::SizeType32> const&,             // beamWidth
                 std::optional<tle::FloatType> const&,              // topP
                 std::optional<tle::FloatType> const&,              // topPMin
                 std::optional<tle::TokenIdType> const&,            // topPResetIds
                 std::optional<tle::FloatType> const&,              // topPDecay
                 std::optional<tle::RandomSeedType> const&,         // seed
                 std::optional<tle::FloatType> const&,              // temperature
                 std::optional<tle::SizeType32> const&,             // minTokens
                 std::optional<tle::FloatType> const&,              // beamSearchDiversityRate
                 std::optional<tle::FloatType> const&,              // repetitionPenalty
                 std::optional<tle::FloatType> const&,              // presencePenalty
                 std::optional<tle::FloatType> const&,              // frequencyPenalty
                 std::optional<tle::FloatType> const&,              // lengthPenalty
                 std::optional<tle::SizeType32> const&,             // earlyStopping
                 std::optional<tle::SizeType32> const&,             // noRepeatNgramSize
                 std::optional<tle::SizeType32> const&,             // numReturnSequences
                 std::optional<tle::FloatType> const&,              // minP
                 std::optional<std::vector<tle::SizeType32>> const& // beamWidthArray
                 >(),
            // clang-format off
            py::arg("beam_width") = 1,
            py::kw_only(),
            py::arg("top_k") = py::none(),
            py::arg("top_p") = py::none(),
            py::arg("top_p_min") = py::none(),
            py::arg("top_p_reset_ids") = py::none(),
            py::arg("top_p_decay") = py::none(),
            py::arg("seed") = py::none(),
            py::arg("temperature") = py::none(),
            py::arg("min_tokens") = py::none(),
            py::arg("beam_search_diversity_rate") = py::none(),
            py::arg("repetition_penalty") = py::none(),
            py::arg("presence_penalty") = py::none(),
            py::arg("frequency_penalty") = py::none(),
            py::arg("length_penalty") = py::none(),
            py::arg("early_stopping") = py::none(),
            py::arg("no_repeat_ngram_size") = py::none(),
            py::arg("num_return_sequences") = py::none(),
            py::arg("min_p") = py::none(),
            py::arg("beam_width_array") = py::none())               // clang-format on
        .def_property("beam_width", &tle::SamplingConfig::getBeamWidth, &tle::SamplingConfig::setBeamWidth)
        .def_property("top_k", &tle::SamplingConfig::getTopK, &tle::SamplingConfig::setTopK)
        .def_property("top_p", &tle::SamplingConfig::getTopP, &tle::SamplingConfig::setTopP)
        .def_property("top_p_min", &tle::SamplingConfig::getTopPMin, &tle::SamplingConfig::setTopPMin)
        .def_property("top_p_reset_ids", &tle::SamplingConfig::getTopPResetIds, &tle::SamplingConfig::setTopPResetIds)
        .def_property("top_p_decay", &tle::SamplingConfig::getTopPDecay, &tle::SamplingConfig::setTopPDecay)
        .def_property("seed", &tle::SamplingConfig::getSeed, &tle::SamplingConfig::setSeed)
        .def_property("temperature", &tle::SamplingConfig::getTemperature, &tle::SamplingConfig::setTemperature)
        .def_property("min_tokens", &tle::SamplingConfig::getMinTokens, &tle::SamplingConfig::setMinTokens)
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
        .def_property(
            "beam_width_array", &tle::SamplingConfig::getBeamWidthArray, &tle::SamplingConfig::setBeamWidthArray)
        .def(py::pickle(samplingConfigGetstate, samplingConfigSetstate));

    auto additionalModelOutputGetstate
        = [](tle::AdditionalModelOutput const& self) { return py::make_tuple(self.name, self.gatherContext); };
    auto additionalModelOutputSetstate = [](py::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid AdditionalModelOutput state!");
        }
        return tle::AdditionalModelOutput(state[0].cast<std::string>(), state[1].cast<bool>());
    };
    py::class_<tle::AdditionalModelOutput>(m, "AdditionalModelOutput")
        .def(py::init<std::string, bool>(), py::arg("name"), py::arg("gather_context") = false)
        .def_readwrite("name", &tle::AdditionalModelOutput::name)
        .def_readwrite("gather_context", &tle::AdditionalModelOutput::gatherContext)
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
            state[6].cast<std::optional<std::vector<tle::AdditionalModelOutput>>>());
    };
    py::class_<tle::OutputConfig>(m, "OutputConfig")
        .def(py::init<bool, bool, bool, bool, bool, bool, std::optional<std::vector<tle::AdditionalModelOutput>>>(),
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

    auto multimodalInputGetstate = [](tle::MultimodalInput const& self)
    { return py::make_tuple(self.getMultimodalHashes(), self.getMultimodalPositions(), self.getMultimodalLengths()); };
    auto multimodalInputSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid MultimodalInput state!");
        }
        return tle::MultimodalInput(state[0].cast<std::vector<std::vector<SizeType32>>>(),
            state[1].cast<std::vector<SizeType32>>(), state[2].cast<std::vector<SizeType32>>());
    };
    py::class_<tle::MultimodalInput>(m, "MultimodalInput")
        .def(py::init<std::vector<std::vector<SizeType32>>, std::vector<SizeType32>, std::vector<SizeType32>>(),
            py::arg("multimodal_hashes"), py::arg("multimodal_positions"), py::arg("multimodal_lengths"))
        .def_property_readonly("multimodal_hashes", &tle::MultimodalInput::getMultimodalHashes)
        .def_property_readonly("multimodal_positions", &tle::MultimodalInput::getMultimodalPositions)
        .def_property_readonly("multimodal_lengths", &tle::MultimodalInput::getMultimodalLengths)
        .def(py::pickle(multimodalInputGetstate, multimodalInputSetstate));

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
        return py::make_tuple(self.getTokenRangeRetentionConfigs(), self.getDecodeRetentionPriority(),
            self.getDecodeDurationMs(), self.getTransferMode(), self.getDirectory());
    };
    auto kvCacheRetentionConfigSetstate = [](py::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::KvCacheRetentionConfig(
            state[0].cast<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>>(),
            state[1].cast<tle::RetentionPriority>(), state[2].cast<std::optional<std::chrono::milliseconds>>(),
            state[3].cast<tle::KvCacheTransferMode>(), state[4].cast<std::string>());
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
                 std::optional<std::chrono::milliseconds>, tle::KvCacheTransferMode, std::string>(),
            py::arg("token_range_retention_configs"),
            py::arg("decode_retention_priority") = tle::KvCacheRetentionConfig::kDefaultRetentionPriority,
            py::arg("decode_duration_ms") = py::none(),
            py::arg_v("transfer_mode", tle::KvCacheTransferMode::DRAM, "DRAM"), py::arg("directory") = py::none())
        .def_property_readonly(
            "token_range_retention_configs", &tle::KvCacheRetentionConfig::getTokenRangeRetentionConfigs)
        .def_property_readonly("decode_retention_priority", &tle::KvCacheRetentionConfig::getDecodeRetentionPriority)
        .def_property_readonly("decode_duration_ms", &tle::KvCacheRetentionConfig::getDecodeDurationMs)
        .def_property_readonly("transfer_mode", &tle::KvCacheRetentionConfig::getTransferMode)
        .def_property_readonly("directory", &tle::KvCacheRetentionConfig::getDirectory)
        .def(py::pickle(kvCacheRetentionConfigGetstate, kvCacheRetentionConfigSetstate))
        .def("__eq__", &tle::KvCacheRetentionConfig::operator==);

    auto ContextPhaseParamsGetState = [](tle::ContextPhaseParams const& self)
    {
        if (self.getState() != nullptr)
        {
            auto serializedState = self.getSerializedState();
            return py::make_tuple(self.getFirstGenTokens(), self.getReqId(),
                py::bytes(serializedState.data(), serializedState.size()), self.getDraftTokens());
        }
        return py::make_tuple(self.getFirstGenTokens(), self.getReqId(), py::none(), self.getDraftTokens());
    };

    auto ContextPhaseParamsSetState = [](py::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid ContextPhaseParams state!");
        }
        if (!state[2].is_none())
        {
            auto opaque_state = state[2].cast<py::bytes>();
            auto opaque_state_str_view = std::string_view(opaque_state.cast<std::string_view>());
            return std::make_unique<tle::ContextPhaseParams>(state[0].cast<VecTokens>(),
                state[1].cast<tle::ContextPhaseParams::RequestIdType>(),
                std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()),
                state[3].cast<std::optional<VecTokens>>());
        }
        return std::make_unique<tle::ContextPhaseParams>(state[0].cast<VecTokens>(),
            state[1].cast<tle::ContextPhaseParams::RequestIdType>(), state[3].cast<std::optional<VecTokens>>());
    };

    py::class_<tle::ContextPhaseParams>(m, "ContextPhaseParams")
        .def(py::init(
            [](VecTokens const& first_gen_tokens, tle::ContextPhaseParams::RequestIdType req_id,
                std::optional<py::bytes> const& opaque_state, std::optional<VecTokens> const& draft_tokens)
            {
                if (opaque_state)
                {
                    auto opaque_state_str_view = std::string_view(opaque_state.value().cast<std::string_view>());
                    return std::make_unique<tle::ContextPhaseParams>(first_gen_tokens, req_id,
                        std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()), draft_tokens);
                }
                return std::make_unique<tle::ContextPhaseParams>(first_gen_tokens, req_id, draft_tokens);
            }))
        .def_property_readonly("first_gen_tokens", &tle::ContextPhaseParams::getFirstGenTokens)
        .def_property_readonly("draft_tokens", &tle::ContextPhaseParams::getDraftTokens)
        .def_property_readonly("req_id", &tle::ContextPhaseParams::getReqId)
        .def_property_readonly("opaque_state",
            [](tle::ContextPhaseParams const& self)
            {
                std::optional<py::bytes> opaque_state{std::nullopt};
                if (self.getState() != nullptr)
                {
                    auto serializedState = self.getSerializedState();
                    opaque_state = py::bytes(serializedState.data(), serializedState.size());
                }
                return opaque_state;
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
        return tle::EagleConfig(state[0].cast<std::optional<tle::EagleChoices>>(), state[1].cast<bool>(),
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
        .value("EBNF_GRAMMAR", tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR)
        .value("STRUCTURAL_TAG", tle::GuidedDecodingParams::GuideType::kSTRUCTURAL_TAG);

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
            self.getPromptTuningConfig(), self.getMultimodalInput(), self.getMultimodalEmbedding(),
            self.getMropeConfig(), self.getLoraConfig(), self.getLookaheadConfig(), self.getKvCacheRetentionConfig(),
            self.getLogitsPostProcessorName(), self.getLogitsPostProcessor(), self.getEncoderInputTokenIds(),
            self.getClientId(), self.getReturnAllGeneratedTokens(), self.getPriority(), self.getRequestType(),
            self.getContextPhaseParams(), self.getEncoderInputFeatures(), self.getEncoderOutputLength(),
            self.getCrossAttentionMask(), self.getEagleConfig(), self.getSkipCrossAttnBlocks(),
            self.getGuidedDecodingParams(), self.getCacheSaltID());
    };
    auto requestSetstate = [](py::tuple const& state)
    {
        if (state.size() != 34)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        return std::make_unique<tle::Request>(state[0].cast<VecTokens>(), state[1].cast<SizeType32>(),
            state[2].cast<bool>(), state[3].cast<tle::SamplingConfig>(), state[4].cast<tle::OutputConfig>(),
            state[5].cast<std::optional<SizeType32>>(), state[6].cast<std::optional<SizeType32>>(),
            state[7].cast<std::optional<std::vector<SizeType32>>>(),
            state[8].cast<std::optional<std::list<VecTokens>>>(), state[9].cast<std::optional<std::list<VecTokens>>>(),
            state[10].cast<std::optional<Tensor>>(), state[11].cast<std::optional<tle::ExternalDraftTokensConfig>>(),
            state[12].cast<std::optional<tle::PromptTuningConfig>>(),
            state[13].cast<std::optional<tle::MultimodalInput>>(), state[14].cast<std::optional<Tensor>>(),
            state[15].cast<std::optional<tle::MropeConfig>>(), state[16].cast<std::optional<tle::LoraConfig>>(),
            state[17].cast<std::optional<tle::LookaheadDecodingConfig>>(),
            state[18].cast<std::optional<tle::KvCacheRetentionConfig>>(), state[19].cast<std::optional<std::string>>(),
            state[20].cast<std::optional<tle::LogitsPostProcessor>>(), state[21].cast<std::optional<VecTokens>>(),
            state[22].cast<std::optional<IdType>>(), state[23].cast<bool>(), state[24].cast<tle::PriorityType>(),
            state[25].cast<tle::RequestType>(), state[26].cast<std::optional<tle::ContextPhaseParams>>(),
            state[27].cast<std::optional<tle::Tensor>>(), state[28].cast<std::optional<SizeType32>>(),
            state[29].cast<std::optional<tle::Tensor>>(), 1, state[30].cast<std::optional<tle::EagleConfig>>(),
            state[31].cast<std::optional<tle::Tensor>>(), state[32].cast<std::optional<tle::GuidedDecodingParams>>(),
            state[33].cast<std::optional<tle::CacheSaltIDType>>());
    };

    py::class_<tle::Request> request(m, "Request", pybind11::dynamic_attr());
    request
        .def(py::init<tle::VecTokens,                           // inputTokenIds
                 tle::SizeType32,                               // maxTokens
                 bool,                                          // streaming
                 tle::SamplingConfig const&,                    // samplingConfig
                 tle::OutputConfig const&,                      // outputConfig
                 std::optional<tle::SizeType32> const&,         // endId
                 std::optional<tle::SizeType32> const&,         // padId
                 std::optional<std::vector<SizeType32>>,        // positionIds
                 std::optional<std::list<tle::VecTokens>>,      // badWords
                 std::optional<std::list<tle::VecTokens>>,      // stopWords
                 std::optional<tle::Tensor>,                    // embeddingBias
                 std::optional<tle::ExternalDraftTokensConfig>, // externalDraftTokensConfig
                 std::optional<tle::PromptTuningConfig>,        // pTuningConfig
                 std::optional<tle::MultimodalInput>,           // multimodalInput
                 std::optional<tle::Tensor>,                    // multimodalEmbedding
                 std::optional<tle::MropeConfig>,               // mRopeConfig
                 std::optional<tle::LoraConfig>,                // loraConfig
                 std::optional<tle::LookaheadDecodingConfig>,   // lookaheadConfig
                 std::optional<tle::KvCacheRetentionConfig>,    // kvCacheRetentionConfig
                 std::optional<std::string>,                    // logitsPostProcessorName
                 std::optional<tle::LogitsPostProcessor>,       // logitsPostProcessor
                 std::optional<tle::VecTokens>,                 // encoderInputTokenIds
                 std::optional<tle::IdType>,                    // clientId
                 bool,                                          // returnAllGeneratedTokens
                 tle::PriorityType,                             // priority
                 tle::RequestType,                              // type
                 std::optional<tle::ContextPhaseParams>,        // contextPhaseParams
                 std::optional<tle::Tensor>,                    // encoderInputFeatures
                 std::optional<tle::SizeType32>,                // encoderOutputLength
                 std::optional<tle::Tensor>,                    // crossAttentionMask
                 SizeType32,                                    // numReturnSequences
                 std::optional<tle::EagleConfig>,               // eagleConfig
                 std::optional<tle::Tensor>,                    // skipCrossAttnBlocks
                 std::optional<tle::GuidedDecodingParams>,      // guidedDecodingParams
                 std::optional<tle::SizeType32>,                // languageAdapterUid
                 std::optional<tle::MillisecondsType>,          // allottedTimeMs
                 std::optional<tle::CacheSaltIDType>            // cacheSaltID
                 >(),
            // clang-format off
        py::arg("input_token_ids"),
        py::arg("max_tokens"),
        py::kw_only(),
        py::arg("streaming") = false,
        py::arg_v("sampling_config", tle::SamplingConfig(), "SamplingConfig()"),
        py::arg_v("output_config", tle::OutputConfig(), "OutputConfig()"),
        py::arg("end_id") = py::none(),
        py::arg("pad_id") = py::none(),
        py::arg("position_ids") = py::none(),
        py::arg("bad_words") = py::none(),
        py::arg("stop_words") = py::none(),
        py::arg("embedding_bias") = py::none(),
        py::arg("external_draft_tokens_config") = py::none(),
        py::arg("prompt_tuning_config") = py::none(),
        py::arg("multimodal_input") = py::none(),
        py::arg("multimodal_embedding") = py::none(),
        py::arg("mrope_config") = py::none(),
        py::arg("lora_config") = py::none(),
        py::arg("lookahead_config") = py::none(),
        py::arg("kv_cache_retention_config") = py::none(),
        py::arg("logits_post_processor_name") = py::none(),
        py::arg("logits_post_processor") = py::none(),
        py::arg("encoder_input_token_ids") = py::none(),
        py::arg("client_id") = py::none(),
        py::arg("return_all_generated_tokens") = false,
        py::arg("priority") = tle::Request::kDefaultPriority,
        py::arg_v("type", tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
            "RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION"),
        py::arg("context_phase_params") = py::none(),
        py::arg("encoder_input_features") = py::none(),
        py::arg("encoder_output_length") = py::none(),
        py::arg("cross_attention_mask") = py::none(),
        py::arg("num_return_sequences") = 1,
        py::arg("eagle_config") = py::none(),
        py::arg("skip_cross_attn_blocks") = py::none(),
        py::arg("guided_decoding_params") = py::none(),
        py::arg("language_adapter_uid") = py::none(),
        py::arg("allotted_time_ms") = py::none(),
        py::arg("cache_salt_id") = py::none()
    )             // clang-format on
        .def_property_readonly("input_token_ids", &tle::Request::getInputTokenIds)
        .def_property_readonly("max_tokens", &tle::Request::getMaxTokens)
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
        .def_property("multimodal_input", &tle::Request::getMultimodalInput, &tle::Request::setMultimodalInput)
        .def_property(
            "multimodal_embedding", &tle::Request::getMultimodalEmbedding, &tle::Request::setMultimodalEmbedding)
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
        .def_property("cache_salt_id", &tle::Request::getCacheSaltID, &tle::Request::setCacheSaltID)
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

    auto timingMetricsGetstate = [](tle::RequestPerfMetrics::TimingMetrics const& self)
    {
        return py::make_tuple(self.arrivalTime, self.firstScheduledTime, self.firstTokenTime, self.lastTokenTime,
            self.kvCacheTransferStart, self.kvCacheTransferEnd, self.kvCacheSize);
    };
    auto timingMetricsSetstate = [](py::tuple const& state)
    {
        if (state.size() != 7)
        {
            throw std::runtime_error("Invalid TimingMetrics state!");
        }
        return tle::RequestPerfMetrics::TimingMetrics{state[0].cast<tle::RequestPerfMetrics::TimePoint>(),
            state[1].cast<tle::RequestPerfMetrics::TimePoint>(), state[2].cast<tle::RequestPerfMetrics::TimePoint>(),
            state[3].cast<tle::RequestPerfMetrics::TimePoint>(), state[4].cast<tle::RequestPerfMetrics::TimePoint>(),
            state[5].cast<tle::RequestPerfMetrics::TimePoint>(), state[6].cast<size_t>()};
    };
    py::class_<tle::RequestPerfMetrics::TimingMetrics>(m, "TimingMetrics")
        .def(py::init<>())
        .def_readwrite("arrival_time", &tle::RequestPerfMetrics::TimingMetrics::arrivalTime)
        .def_readwrite("first_scheduled_time", &tle::RequestPerfMetrics::TimingMetrics::firstScheduledTime)
        .def_readwrite("first_token_time", &tle::RequestPerfMetrics::TimingMetrics::firstTokenTime)
        .def_readwrite("last_token_time", &tle::RequestPerfMetrics::TimingMetrics::lastTokenTime)
        .def_readwrite("kv_cache_transfer_start", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferStart)
        .def_readwrite("kv_cache_transfer_end", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferEnd)
        .def_readwrite("kv_cache_size", &tle::RequestPerfMetrics::TimingMetrics::kvCacheSize)
        .def(py::pickle(timingMetricsGetstate, timingMetricsSetstate));

    auto kvCacheMetricsGetstate = [](tle::RequestPerfMetrics::KvCacheMetrics const& self)
    {
        return py::make_tuple(self.numTotalAllocatedBlocks, self.numNewAllocatedBlocks, self.numReusedBlocks,
            self.numMissedBlocks, self.kvCacheHitRate);
    };
    auto kvCacheMetricsSetstate = [](py::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid KvCacheMetrics state!");
        }
        return tle::RequestPerfMetrics::KvCacheMetrics{state[0].cast<SizeType32>(), state[1].cast<SizeType32>(),
            state[2].cast<SizeType32>(), state[3].cast<SizeType32>(), state[4].cast<float>()};
    };
    py::class_<tle::RequestPerfMetrics::KvCacheMetrics>(m, "KvCacheMetrics")
        .def(py::init<>())
        .def_readwrite("num_total_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numTotalAllocatedBlocks)
        .def_readwrite("num_new_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numNewAllocatedBlocks)
        .def_readwrite("num_reused_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numReusedBlocks)
        .def_readwrite("num_missed_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numMissedBlocks)
        .def_readwrite("kv_cache_hit_rate", &tle::RequestPerfMetrics::KvCacheMetrics::kvCacheHitRate)
        .def(py::pickle(kvCacheMetricsGetstate, kvCacheMetricsSetstate));

    auto speculativeDecodingMetricsGetstate = [](tle::RequestPerfMetrics::SpeculativeDecodingMetrics const& self)
    { return py::make_tuple(self.acceptanceRate, self.totalAcceptedDraftTokens, self.totalDraftTokens); };
    auto speculativeDecodingMetricsSetstate = [](py::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid SpeculativeDecodingMetrics state!");
        }
        return tle::RequestPerfMetrics::SpeculativeDecodingMetrics{
            state[0].cast<float>(), state[1].cast<SizeType32>(), state[2].cast<SizeType32>()};
    };

    py::class_<tle::RequestPerfMetrics::SpeculativeDecodingMetrics>(m, "SpeculativeDecodingMetrics")
        .def(py::init<>())
        .def_readwrite("acceptance_rate", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::acceptanceRate)
        .def_readwrite("total_accepted_draft_tokens",
            &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalAcceptedDraftTokens)
        .def_readwrite("total_draft_tokens", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalDraftTokens)
        .def(py::pickle(speculativeDecodingMetricsGetstate, speculativeDecodingMetricsSetstate));

    auto requestPerfMetricsGetstate = [](tle::RequestPerfMetrics const& self)
    {
        return py::make_tuple(self.timingMetrics, self.kvCacheMetrics, self.speculativeDecoding, self.firstIter,
            self.lastIter, self.iter);
    };
    auto requestPerfMetricsSetstate = [](py::tuple const& state)
    {
        if (state.size() != 6)
        {
            throw std::runtime_error("Invalid RequestPerfMetrics state!");
        }
        return tle::RequestPerfMetrics{state[0].cast<tle::RequestPerfMetrics::TimingMetrics>(),
            state[1].cast<tle::RequestPerfMetrics::KvCacheMetrics>(),
            state[2].cast<tle::RequestPerfMetrics::SpeculativeDecodingMetrics>(),
            state[3].cast<std::optional<tle::IterationType>>(), state[4].cast<std::optional<tle::IterationType>>(),
            state[5].cast<std::optional<tle::IterationType>>()};
    };

    // There's a circular dependency between the declaration of the TimingMetrics and RequestPerfMetrics bindings.
    // Defer definition of the RequestPerfMetrics bindings until the TimingMetrics have been defined.
    requestPerfMetrics.def(py::init<>())
        .def_readwrite("timing_metrics", &tle::RequestPerfMetrics::timingMetrics)
        .def_readwrite("kv_cache_metrics", &tle::RequestPerfMetrics::kvCacheMetrics)
        .def_readwrite("speculative_decoding", &tle::RequestPerfMetrics::speculativeDecoding)
        .def_readwrite("first_iter", &tle::RequestPerfMetrics::firstIter)
        .def_readwrite("last_iter", &tle::RequestPerfMetrics::lastIter)
        .def_readwrite("iter", &tle::RequestPerfMetrics::iter)
        .def(py::pickle(requestPerfMetricsGetstate, requestPerfMetricsSetstate));

    py::class_<tle::AdditionalOutput>(m, "AdditionalOutput")
        .def(py::init([](std::string const& name, tle::Tensor const& output)
            { return std::make_unique<tle::AdditionalOutput>(name, output); }))
        .def_readwrite("name", &tle::AdditionalOutput::name)
        .def_readwrite("output", &tle::AdditionalOutput::output);

    auto resultSetstate = [](py::tuple const& state)
    {
        if (state.size() != 14)
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
        result.avgDecodedTokensPerIter = state[11].cast<float>();
        result.contextPhaseParams = state[12].cast<std::optional<tle::ContextPhaseParams>>();
        result.requestPerfMetrics = state[13].cast<std::optional<tle::RequestPerfMetrics>>();
        return std::make_unique<tle::Result>(result);
    };

    auto resultGetstate = [](tle::Result const& self)
    {
        return py::make_tuple(self.isFinal, self.outputTokenIds, self.cumLogProbs, self.logProbs, self.contextLogits,
            self.generationLogits, self.encoderOutput, self.finishReasons, self.sequenceIndex, self.isSequenceFinal,
            self.decodingIter, self.avgDecodedTokensPerIter, self.contextPhaseParams, self.requestPerfMetrics);
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
        .def_readwrite("avg_decoded_tokens_per_iter", &tle::Result::avgDecodedTokensPerIter)
        .def_readwrite("context_phase_params", &tle::Result::contextPhaseParams)
        .def_readwrite("request_perf_metrics", &tle::Result::requestPerfMetrics)
        .def_readwrite("additional_outputs", &tle::Result::additionalOutputs)
        .def(py::pickle(resultGetstate, resultSetstate));

    m.def("deserialize_result",
        [](std::string& x)
        {
            std::istringstream is(x);
            return tle::serialize_utils::deserialize<tle::Result>(is);
        });

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
        .def("clear_context_logits",
            [](tle::Response& self)
            {
                if (!self.hasError())
                {
                    auto& result = const_cast<tle::Result&>(self.getResult());
                    result.contextLogits.reset();
                }
            })
        .def("clear_generation_logits",
            [](tle::Response& self)
            {
                if (!self.hasError())
                {
                    auto& result = const_cast<tle::Result&>(self.getResult());
                    result.generationLogits.reset();
                }
            })
        .def(py::pickle(responseGetstate, responseSetstate));
}

} // namespace tensorrt_llm::pybind::executor
