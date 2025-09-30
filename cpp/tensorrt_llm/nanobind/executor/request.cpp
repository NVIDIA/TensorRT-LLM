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

#include "request.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <sstream>

#include <optional>
#include <vector>

namespace nb = nanobind;
namespace tle = tensorrt_llm::executor;
using Tensor = tle::Tensor;
using SizeType32 = tle::SizeType32;
using FloatType = tle::FloatType;
using VecTokens = tle::VecTokens;
using IdType = tle::IdType;
using VecTokenExtraIds = tle::VecTokenExtraIds;

namespace tensorrt_llm::nanobind::executor
{

void initRequestBindings(nb::module_& m)
{
    nb::enum_<tle::RequestType>(m, "RequestType")
        .value("REQUEST_TYPE_CONTEXT_AND_GENERATION", tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION)
        .value("REQUEST_TYPE_CONTEXT_ONLY", tle::RequestType::REQUEST_TYPE_CONTEXT_ONLY)
        .value("REQUEST_TYPE_GENERATION_ONLY", tle::RequestType::REQUEST_TYPE_GENERATION_ONLY);

    nb::enum_<tle::FinishReason>(m, "FinishReason")
        .value("NOT_FINISHED", tle::FinishReason::kNOT_FINISHED)
        .value("END_ID", tle::FinishReason::kEND_ID)
        .value("STOP_WORDS", tle::FinishReason::kSTOP_WORDS)
        .value("LENGTH", tle::FinishReason::kLENGTH)
        .value("TIMED_OUT", tle::FinishReason::kTIMED_OUT)
        .value("CANCELLED", tle::FinishReason::kCANCELLED);

    nb::enum_<tle::KvCacheTransferMode>(m, "KvCacheTransferMode")
        .value("DRAM", tle::KvCacheTransferMode::DRAM)
        .value("GDS", tle::KvCacheTransferMode::GDS)
        .value("POSIX_DEBUG_FALLBACK", tle::KvCacheTransferMode::POSIX_DEBUG_FALLBACK);

    auto samplingConfigGetstate = [](tle::SamplingConfig const& self)
    {
        return nb::make_tuple(self.getBeamWidth(), self.getTopK(), self.getTopP(), self.getTopPMin(),
            self.getTopPResetIds(), self.getTopPDecay(), self.getSeed(), self.getTemperature(), self.getMinTokens(),
            self.getBeamSearchDiversityRate(), self.getRepetitionPenalty(), self.getPresencePenalty(),
            self.getFrequencyPenalty(), self.getLengthPenalty(), self.getEarlyStopping(), self.getNoRepeatNgramSize(),
            self.getNumReturnSequences(), self.getMinP(), self.getBeamWidthArray());
    };
    auto samplingConfigSetstate = [](tle::SamplingConfig& samplingConfig, nb::tuple const& state)
    {
        if (state.size() != 19)
        {
            throw std::runtime_error("Invalid SamplingConfig state!");
        }
        new (&samplingConfig) tle::SamplingConfig(nb::cast<SizeType32>(state[0]), // BeamWidth
            nb::cast<std::optional<SizeType32>>(state[1]),                        // TopK
            nb::cast<std::optional<FloatType>>(state[2]),                         // TopP
            nb::cast<std::optional<FloatType>>(state[3]),                         // TopPMin
            nb::cast<std::optional<tle::TokenIdType>>(state[4]),                  // TopPResetIds
            nb::cast<std::optional<FloatType>>(state[5]),                         // TopPDecay
            nb::cast<std::optional<tle::RandomSeedType>>(state[6]),               // Seed
            nb::cast<std::optional<FloatType>>(state[7]),                         // Temperature
            nb::cast<std::optional<SizeType32>>(state[8]),                        // MinTokens
            nb::cast<std::optional<FloatType>>(state[9]),                         // BeamSearchDiversityRate
            nb::cast<std::optional<FloatType>>(state[10]),                        // RepetitionPenalty
            nb::cast<std::optional<FloatType>>(state[11]),                        // PresencePenalty
            nb::cast<std::optional<FloatType>>(state[12]),                        // FrequencyPenalty
            nb::cast<std::optional<FloatType>>(state[13]),                        // LengthPenalty
            nb::cast<std::optional<SizeType32>>(state[14]),                       // EarlyStopping
            nb::cast<std::optional<SizeType32>>(state[15]),                       // NoRepeatNgramSize
            nb::cast<std::optional<SizeType32>>(state[16]),                       // NumReturnSequences
            nb::cast<std::optional<FloatType>>(state[17]),                        // MinP
            nb::cast<std::optional<std::vector<SizeType32>>>(state[18])           // BeamWidthArray
        );
    };
    nb::class_<tle::SamplingConfig>(m, "SamplingConfig")
        .def(nb::init<tle::SizeType32,
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
            nb::arg("beam_width") = 1,
            nb::kw_only(),
            nb::arg("top_k") = nb::none(),
            nb::arg("top_p") = nb::none(),
            nb::arg("top_p_min") = nb::none(),
            nb::arg("top_p_reset_ids") = nb::none(),
            nb::arg("top_p_decay") = nb::none(),
            nb::arg("seed") = nb::none(),
            nb::arg("temperature") = nb::none(),
            nb::arg("min_tokens") = nb::none(),
            nb::arg("beam_search_diversity_rate") = nb::none(),
            nb::arg("repetition_penalty") = nb::none(),
            nb::arg("presence_penalty") = nb::none(),
            nb::arg("frequency_penalty") = nb::none(),
            nb::arg("length_penalty") = nb::none(),
            nb::arg("early_stopping") = nb::none(),
            nb::arg("no_repeat_ngram_size") = nb::none(),
            nb::arg("num_return_sequences") = nb::none(),
            nb::arg("min_p") = nb::none(),
            nb::arg("beam_width_array") = nb::none())               // clang-format on
        .def_prop_rw("beam_width", &tle::SamplingConfig::getBeamWidth, &tle::SamplingConfig::setBeamWidth)
        .def_prop_rw("top_k", &tle::SamplingConfig::getTopK, &tle::SamplingConfig::setTopK)
        .def_prop_rw("top_p", &tle::SamplingConfig::getTopP, &tle::SamplingConfig::setTopP)
        .def_prop_rw("top_p_min", &tle::SamplingConfig::getTopPMin, &tle::SamplingConfig::setTopPMin)
        .def_prop_rw("top_p_reset_ids", &tle::SamplingConfig::getTopPResetIds, &tle::SamplingConfig::setTopPResetIds)
        .def_prop_rw("top_p_decay", &tle::SamplingConfig::getTopPDecay, &tle::SamplingConfig::setTopPDecay)
        .def_prop_rw("seed", &tle::SamplingConfig::getSeed, &tle::SamplingConfig::setSeed)
        .def_prop_rw("temperature", &tle::SamplingConfig::getTemperature, &tle::SamplingConfig::setTemperature)
        .def_prop_rw("min_tokens", &tle::SamplingConfig::getMinTokens, &tle::SamplingConfig::setMinTokens)
        .def_prop_rw("beam_search_diversity_rate", &tle::SamplingConfig::getBeamSearchDiversityRate,
            &tle::SamplingConfig::setBeamSearchDiversityRate)
        .def_prop_rw("repetition_penalty", &tle::SamplingConfig::getRepetitionPenalty,
            &tle::SamplingConfig::setRepetitionPenalty)
        .def_prop_rw("presence_penalty", &tle::SamplingConfig::getPresencePenalty,
            [](tle::SamplingConfig& self, std::optional<FloatType> v) { self.setPresencePenalty(v); })
        .def_prop_rw(
            "frequency_penalty", &tle::SamplingConfig::getFrequencyPenalty, &tle::SamplingConfig::setFrequencyPenalty)
        .def_prop_rw("length_penalty", &tle::SamplingConfig::getLengthPenalty, &tle::SamplingConfig::setLengthPenalty)
        .def_prop_rw("early_stopping", &tle::SamplingConfig::getEarlyStopping, &tle::SamplingConfig::setEarlyStopping)
        .def_prop_rw("no_repeat_ngram_size", &tle::SamplingConfig::getNoRepeatNgramSize,
            &tle::SamplingConfig::setNoRepeatNgramSize)
        .def_prop_rw("num_return_sequences", &tle::SamplingConfig::getNumReturnSequences,
            &tle::SamplingConfig::setNumReturnSequences)
        .def_prop_rw("min_p", &tle::SamplingConfig::getMinP, &tle::SamplingConfig::setMinP)
        .def_prop_rw(
            "beam_width_array", &tle::SamplingConfig::getBeamWidthArray, &tle::SamplingConfig::setBeamWidthArray)
        .def("__getstate__", samplingConfigGetstate)
        .def("__setstate__", samplingConfigSetstate);

    auto additionalModelOutputGetstate
        = [](tle::AdditionalModelOutput const& self) { return nb::make_tuple(self.name, self.gatherContext); };
    auto additionalModelOutputSetstate = [](tle::AdditionalModelOutput& additionalModelOutput, nb::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid AdditionalModelOutput state!");
        }
        new (&additionalModelOutput)
            tle::AdditionalModelOutput(nb::cast<std::string>(state[0]), nb::cast<bool>(state[1]));
    };
    nb::class_<tle::AdditionalModelOutput>(m, "AdditionalModelOutput")
        .def(nb::init<std::string, bool>(), nb::arg("name"), nb::arg("gather_context") = false)
        .def_rw("name", &tle::AdditionalModelOutput::name)
        .def_rw("gather_context", &tle::AdditionalModelOutput::gatherContext)
        .def("__getstate__", additionalModelOutputGetstate)
        .def("__setstate__", additionalModelOutputSetstate);

    auto outputConfigGetstate = [](tle::OutputConfig const& self)
    {
        return nb::make_tuple(self.returnLogProbs, self.returnContextLogits, self.returnGenerationLogits,
            self.excludeInputFromOutput, self.returnEncoderOutput, self.returnPerfMetrics, self.additionalModelOutputs);
    };
    auto outputConfigSetstate = [](tle::OutputConfig& outputConfig, nb::tuple const& state)
    {
        if (state.size() != 7)
        {
            throw std::runtime_error("Invalid OutputConfig state!");
        }
        new (&outputConfig) tle::OutputConfig(nb::cast<bool>(state[0]), nb::cast<bool>(state[1]),
            nb::cast<bool>(state[2]), nb::cast<bool>(state[3]), nb::cast<bool>(state[4]), nb::cast<bool>(state[5]),
            nb::cast<std::optional<std::vector<tle::AdditionalModelOutput>>>(state[6]));
    };
    nb::class_<tle::OutputConfig>(m, "OutputConfig")
        .def(
            "__init__",
            [](tle::OutputConfig& self, std::optional<bool> return_log_probs, std::optional<bool> return_context_logits,
                std::optional<bool> return_generation_logits, std::optional<bool> exclude_input_from_output,
                std::optional<bool> return_encoder_output, std::optional<bool> return_perf_metrics,
                std::optional<std::vector<tle::AdditionalModelOutput>> additional_model_outputs)
            {
                new (&self) tle::OutputConfig(return_log_probs.value_or(false), return_context_logits.value_or(false),
                    return_generation_logits.value_or(false), exclude_input_from_output.value_or(false),
                    return_encoder_output.value_or(false), return_perf_metrics.value_or(false),
                    additional_model_outputs);
            },
            nb::arg("return_log_probs") = nb::none(), nb::arg("return_context_logits") = nb::none(),
            nb::arg("return_generation_logits") = nb::none(), nb::arg("exclude_input_from_output") = nb::none(),
            nb::arg("return_encoder_output") = nb::none(), nb::arg("return_perf_metrics") = nb::none(),
            nb::arg("additional_model_outputs") = nb::none())
        .def_rw("return_log_probs", &tle::OutputConfig::returnLogProbs)
        .def_rw("return_context_logits", &tle::OutputConfig::returnContextLogits)
        .def_rw("return_generation_logits", &tle::OutputConfig::returnGenerationLogits)
        .def_rw("exclude_input_from_output", &tle::OutputConfig::excludeInputFromOutput)
        .def_rw("return_encoder_output", &tle::OutputConfig::returnEncoderOutput)
        .def_rw("return_perf_metrics", &tle::OutputConfig::returnPerfMetrics)
        .def_rw("additional_model_outputs", &tle::OutputConfig::additionalModelOutputs)
        .def("__getstate__", outputConfigGetstate)
        .def("__setstate__", outputConfigSetstate);

    auto externalDraftTokensConfigGetstate = [](tle::ExternalDraftTokensConfig const& self)
    { return nb::make_tuple(self.getTokens(), self.getLogits(), self.getAcceptanceThreshold()); };
    auto externalDraftTokensConfigSetstate
        = [](tle::ExternalDraftTokensConfig& externalDraftTokensConfig, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid ExternalDraftTokensConfig state!");
        }
        new (&externalDraftTokensConfig) tle::ExternalDraftTokensConfig(nb::cast<VecTokens>(state[0]),
            nb::cast<std::optional<Tensor>>(state[1]), nb::cast<std::optional<FloatType>>(state[2]));
    };
    nb::class_<tle::ExternalDraftTokensConfig>(m, "ExternalDraftTokensConfig")
        .def(nb::init<VecTokens, std::optional<Tensor>, std::optional<FloatType> const&, std::optional<bool>>(),
            nb::arg("tokens"), nb::arg("logits") = nb::none(), nb::arg("acceptance_threshold") = nb::none(),
            nb::arg("fast_logits") = nb::none())
        .def_prop_ro("tokens", &tle::ExternalDraftTokensConfig::getTokens)
        .def_prop_ro("logits", &tle::ExternalDraftTokensConfig::getLogits)
        .def_prop_ro("acceptance_threshold", &tle::ExternalDraftTokensConfig::getAcceptanceThreshold)
        .def("__getstate__", externalDraftTokensConfigGetstate)
        .def("__setstate__", externalDraftTokensConfigSetstate)
        .def_prop_ro("fast_logits", &tle::ExternalDraftTokensConfig::getFastLogits);

    auto promptTuningConfigGetstate = [](tle::PromptTuningConfig const& self)
    { return nb::make_tuple(self.getEmbeddingTable(), self.getInputTokenExtraIds()); };
    auto promptTuningConfigSetstate = [](tle::PromptTuningConfig& promptTuningConfig, nb::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid PromptTuningConfig state!");
        }
        new (&promptTuningConfig)
            tle::PromptTuningConfig(nb::cast<Tensor>(state[0]), nb::cast<std::optional<VecTokenExtraIds>>(state[1]));
    };
    nb::class_<tle::PromptTuningConfig>(m, "PromptTuningConfig")
        .def(nb::init<Tensor, std::optional<VecTokenExtraIds>>(), nb::arg("embedding_table"),
            nb::arg("input_token_extra_ids") = nb::none())
        .def_prop_ro("embedding_table", &tle::PromptTuningConfig::getEmbeddingTable)
        .def_prop_ro("input_token_extra_ids", &tle::PromptTuningConfig::getInputTokenExtraIds)
        .def("__getstate__", promptTuningConfigGetstate)
        .def("__setstate__", promptTuningConfigSetstate);

    auto loraConfigGetstate = [](tle::LoraConfig const& self)
    { return nb::make_tuple(self.getTaskId(), self.getWeights(), self.getConfig()); };
    auto loraConfigSetstate = [](tle::LoraConfig& loraConfig, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LoraConfig state!");
        }
        new (&loraConfig) tle::LoraConfig(nb::cast<IdType>(state[0]), nb::cast<std::optional<Tensor>>(state[1]),
            nb::cast<std::optional<Tensor>>(state[2]));
    };
    nb::class_<tle::LoraConfig>(m, "LoraConfig")
        .def(nb::init<uint64_t, std::optional<Tensor>, std::optional<Tensor>>(), nb::arg("task_id"),
            nb::arg("weights") = nb::none(), nb::arg("config") = nb::none())
        .def_prop_ro("task_id", &tle::LoraConfig::getTaskId)
        .def_prop_ro("weights", &tle::LoraConfig::getWeights)
        .def_prop_ro("config", &tle::LoraConfig::getConfig)
        .def("__getstate__", loraConfigGetstate)
        .def("__setstate__", loraConfigSetstate);

    auto multimodalInputGetstate = [](tle::MultimodalInput const& self)
    { return nb::make_tuple(self.getMultimodalHashes(), self.getMultimodalPositions(), self.getMultimodalLengths()); };
    auto multimodalInputSetstate = [](tle::MultimodalInput& multimodalInput, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid MultimodalInput state!");
        }
        new (&multimodalInput) tle::MultimodalInput(nb::cast<std::vector<std::vector<SizeType32>>>(state[0]),
            nb::cast<std::vector<SizeType32>>(state[1]), nb::cast<std::vector<SizeType32>>(state[2]));
    };
    nb::class_<tle::MultimodalInput>(m, "MultimodalInput")
        .def(nb::init<std::vector<std::vector<SizeType32>>, std::vector<SizeType32>, std::vector<SizeType32>>(),
            nb::arg("multimodal_hashes"), nb::arg("multimodal_positions"), nb::arg("multimodal_lengths"))
        .def_prop_ro("multimodal_hashes", &tle::MultimodalInput::getMultimodalHashes)
        .def_prop_ro("multimodal_positions", &tle::MultimodalInput::getMultimodalPositions)
        .def_prop_ro("multimodal_lengths", &tle::MultimodalInput::getMultimodalLengths)
        .def("__getstate__", multimodalInputGetstate)
        .def("__setstate__", multimodalInputSetstate);

    auto MropeConfigGetstate = [](tle::MropeConfig const& self)
    { return nb::make_tuple(self.getMRopeRotaryCosSin(), self.getMRopePositionDeltas()); };
    auto MropeConfigSetstate = [](tle::MropeConfig& mropeConfig, nb::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid MropeConfig state!");
        }
        new (&mropeConfig) tle::MropeConfig(nb::cast<tle::Tensor>(state[0]), nb::cast<SizeType32>(state[1]));
    };
    nb::class_<tle::MropeConfig>(m, "MropeConfig")
        .def(nb::init<Tensor, SizeType32>(), nb::arg("mrope_rotary_cos_sin"), nb::arg("mrope_position_deltas"))
        .def_prop_ro("mrope_rotary_cos_sin", &tle::MropeConfig::getMRopeRotaryCosSin)
        .def_prop_ro("mrope_position_deltas", &tle::MropeConfig::getMRopePositionDeltas)
        .def("__getstate__", MropeConfigGetstate)
        .def("__setstate__", MropeConfigSetstate);

    auto lookaheadDecodingConfigGetstate = [](tle::LookaheadDecodingConfig const& self)
    { return nb::make_tuple(self.getWindowSize(), self.getNgramSize(), self.getVerificationSetSize()); };
    auto lookaheadDecodingConfigSetstate
        = [](tle::LookaheadDecodingConfig& lookaheadDecodingConfig, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LookaheadDecodingConfig state!");
        }
        new (&lookaheadDecodingConfig) tle::LookaheadDecodingConfig(
            nb::cast<SizeType32>(state[0]), nb::cast<SizeType32>(state[1]), nb::cast<SizeType32>(state[2]));
    };
    nb::class_<tle::LookaheadDecodingConfig>(m, "LookaheadDecodingConfig")
        .def(nb::init<SizeType32, SizeType32, SizeType32>(), nb::arg("max_window_size"), nb::arg("max_ngram_size"),
            nb::arg("max_verification_set_size"))
        .def_prop_ro("max_window_size", &tle::LookaheadDecodingConfig::getWindowSize)
        .def_prop_ro("max_ngram_size", &tle::LookaheadDecodingConfig::getNgramSize)
        .def_prop_ro("max_verification_set_size", &tle::LookaheadDecodingConfig::getVerificationSetSize)
        .def("calculate_speculative_resource", &tle::LookaheadDecodingConfig::calculateSpeculativeResource)
        .def_static(
            "calculate_speculative_resource_tuple", &tle::LookaheadDecodingConfig::calculateSpeculativeResourceTuple)
        .def("__getstate__", lookaheadDecodingConfigGetstate)
        .def("__setstate__", lookaheadDecodingConfigSetstate)
        .def_static("get_default_lookahead_decoding_window",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingWindow; })
        .def_static("get_default_lookahead_decoding_ngram",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingNgram; })
        .def_static("get_default_lookahead_decoding_verification_set",
            []() { return tle::LookaheadDecodingConfig::kDefaultLookaheadDecodingVerificationSet; });

    auto TokenRangeRetentionConfigGetstate = [](tle::KvCacheRetentionConfig::TokenRangeRetentionConfig const& self)
    { return nb::make_tuple(self.tokenStart, self.tokenEnd, self.priority, self.durationMs); };
    auto TokenRangeRetentionConfigSetstate
        = [](tle::KvCacheRetentionConfig::TokenRangeRetentionConfig& tokenRangeRetentionConfig, nb::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&tokenRangeRetentionConfig) tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(
            nb::cast<SizeType32>(state[0]), nb::cast<std::optional<SizeType32>>(state[1]),
            nb::cast<tle::RetentionPriority>(state[2]), nb::cast<std::optional<std::chrono::milliseconds>>(state[3]));
    };
    auto kvCacheRetentionConfigGetstate = [](tle::KvCacheRetentionConfig const& self)
    {
        return nb::make_tuple(self.getTokenRangeRetentionConfigs(), self.getDecodeRetentionPriority(),
            self.getDecodeDurationMs(), self.getTransferMode(), self.getDirectory());
    };
    auto kvCacheRetentionConfigSetstate
        = [](tle::KvCacheRetentionConfig& kvCacheRetentionConfig, nb::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid state!");
        }
        new (&kvCacheRetentionConfig) tle::KvCacheRetentionConfig(
            nb::cast<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>>(state[0]),
            nb::cast<tle::RetentionPriority>(state[1]), nb::cast<std::optional<std::chrono::milliseconds>>(state[2]),
            nb::cast<tle::KvCacheTransferMode>(state[3]), nb::cast<std::string>(state[4]));
    };

    auto kvCacheRetentionConfig = nb::class_<tle::KvCacheRetentionConfig>(m, "KvCacheRetentionConfig");

    nb::class_<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>(
        kvCacheRetentionConfig, "TokenRangeRetentionConfig")
        .def(nb::init<SizeType32, std::optional<SizeType32>, tle::RetentionPriority,
                 std::optional<std::chrono::milliseconds>>(),
            nb::arg("token_start"), nb::arg("token_end"), nb::arg("priority"), nb::arg("duration_ms") = nb::none())
        .def_rw("token_start", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::tokenStart)
        .def_rw("token_end", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::tokenEnd)
        .def_rw("priority", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::priority)
        .def_rw("duration_ms", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::durationMs)
        .def("__getstate__", TokenRangeRetentionConfigGetstate)
        .def("__setstate__", TokenRangeRetentionConfigSetstate)
        .def("__eq__", &tle::KvCacheRetentionConfig::TokenRangeRetentionConfig::operator==);

    // There's a circular dependency between the declaration of the TokenRangeRetentionPriority and
    // KvCacheRetentionConfig bindings. Defer definition of the KvCacheRetentionConfig bindings until the
    // TokenRangeRetentionPriority bindings have been defined.
    kvCacheRetentionConfig
        .def(nb::init<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig>, tle::RetentionPriority,
                 std::optional<std::chrono::milliseconds>, tle::KvCacheTransferMode, std::string>(),
            nb::arg("token_range_retention_configs"),
            nb::arg("decode_retention_priority") = tle::KvCacheRetentionConfig::kDefaultRetentionPriority,
            nb::arg("decode_duration_ms") = nb::none(), nb::arg("transfer_mode") = tle::KvCacheTransferMode::DRAM,
            nb::arg("directory") = nb::none())
        .def_prop_ro("token_range_retention_configs", &tle::KvCacheRetentionConfig::getTokenRangeRetentionConfigs)
        .def_prop_ro("decode_retention_priority", &tle::KvCacheRetentionConfig::getDecodeRetentionPriority)
        .def_prop_ro("decode_duration_ms", &tle::KvCacheRetentionConfig::getDecodeDurationMs)
        .def_prop_ro("transfer_mode", &tle::KvCacheRetentionConfig::getTransferMode)
        .def_prop_ro("directory", &tle::KvCacheRetentionConfig::getDirectory)
        .def("__getstate__", kvCacheRetentionConfigGetstate)
        .def("__setstate__", kvCacheRetentionConfigSetstate)
        .def("__eq__", &tle::KvCacheRetentionConfig::operator==);

    auto ContextPhaseParamsGetState = [](tle::ContextPhaseParams const& self)
    {
        if (self.getState() != nullptr)
        {
            auto serializedState = self.getSerializedState();
            return nb::make_tuple(self.getFirstGenTokens(), self.getReqId(),
                nb::bytes(serializedState.data(), serializedState.size()), self.getDraftTokens());
        }
        return nb::make_tuple(self.getFirstGenTokens(), self.getReqId(), nb::none(), self.getDraftTokens());
    };

    auto ContextPhaseParamsSetState = [](tle::ContextPhaseParams& contextPhaseParams, nb::tuple const& state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid ContextPhaseParams state!");
        }
        if (!state[2].is_none())
        {
            auto opaque_state = nb::cast<nb::bytes>(state[2]);
            auto opaque_state_str_view = std::string_view(opaque_state.c_str(), opaque_state.size());
            new (&contextPhaseParams) tle::ContextPhaseParams(nb::cast<VecTokens>(state[0]),
                nb::cast<tle::ContextPhaseParams::RequestIdType>(state[1]),
                std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()),
                nb::cast<std::optional<VecTokens>>(state[3]));
        }
        else
        {
            new (&contextPhaseParams) tle::ContextPhaseParams(nb::cast<VecTokens>(state[0]),
                nb::cast<tle::ContextPhaseParams::RequestIdType>(state[1]),
                nb::cast<std::optional<VecTokens>>(state[3]));
        }
    };

    nb::class_<tle::ContextPhaseParams>(m, "ContextPhaseParams")
        .def(
            "__init__",
            [](tle::ContextPhaseParams& self, VecTokens const& first_gen_tokens,
                tle::ContextPhaseParams::RequestIdType req_id, std::optional<nb::bytes> const& opaque_state,
                std::optional<VecTokens> const& draft_tokens)
            {
                if (opaque_state)
                {
                    auto opaque_state_str_view
                        = std::string_view(opaque_state.value().c_str(), opaque_state.value().size());
                    new (&self) tle::ContextPhaseParams(first_gen_tokens, req_id,
                        std::vector<char>(opaque_state_str_view.begin(), opaque_state_str_view.end()), draft_tokens);
                }
                else
                {
                    new (&self) tle::ContextPhaseParams(first_gen_tokens, req_id, draft_tokens);
                }
            },
            nb::arg("first_gen_tokens"), nb::arg("req_id"), nb::arg("opaque_state").none(),
            nb::arg("draft_tokens").none())
        .def_prop_ro("first_gen_tokens", [](tle::ContextPhaseParams const& self) { return self.getFirstGenTokens(); })
        .def_prop_ro("draft_tokens", [](tle::ContextPhaseParams const& self) { return self.getDraftTokens(); })
        .def_prop_ro("req_id", &tle::ContextPhaseParams::getReqId)
        .def_prop_ro("opaque_state",
            [](tle::ContextPhaseParams const& self)
            {
                std::optional<nb::bytes> opaque_state{std::nullopt};
                if (self.getState() != nullptr)
                {
                    auto serializedState = self.getSerializedState();
                    opaque_state = nb::bytes(serializedState.data(), serializedState.size());
                }
                return opaque_state;
            })
        .def("__getstate__", ContextPhaseParamsGetState)
        .def("__setstate__", ContextPhaseParamsSetState);

    auto EagleDecodingConfigGetstate = [](tle::EagleConfig const& self)
    {
        return nb::make_tuple(self.getEagleChoices(), self.isGreedySampling(), self.getPosteriorThreshold(),
            self.useDynamicTree(), self.getDynamicTreeMaxTopK());
    };
    auto EagleDecodingConfigSetstate = [](tle::EagleConfig& self, nb::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid EagleConfig state!");
        }
        new (&self) tle::EagleConfig(nb::cast<std::optional<tle::EagleChoices>>(state[0]), nb::cast<bool>(state[1]),
            nb::cast<std::optional<float>>(state[2]), nb::cast<bool>(state[3]),
            nb::cast<std::optional<SizeType32>>(state[4]));
    };
    nb::class_<tle::EagleConfig>(m, "EagleConfig")
        .def(nb::init<std::optional<tle::EagleChoices>, bool, std::optional<float>, bool, std::optional<SizeType32>>(),
            nb::arg("eagle_choices") = nb::none(), nb::arg("greedy_sampling") = true,
            nb::arg("posterior_threshold") = nb::none(), nb::arg("use_dynamic_tree") = false,
            nb::arg("dynamic_tree_max_topK") = nb::none())
        .def_prop_ro("eagle_choices", &tle::EagleConfig::getEagleChoices)
        .def_prop_ro("greedy_sampling", &tle::EagleConfig::isGreedySampling)
        .def_prop_ro("posterior_threshold", &tle::EagleConfig::getPosteriorThreshold)
        .def_prop_ro("use_dynamic_tree", &tle::EagleConfig::useDynamicTree)
        .def_prop_ro("dynamic_tree_max_topK", &tle::EagleConfig::getDynamicTreeMaxTopK)
        .def("__getstate__", EagleDecodingConfigGetstate)
        .def("__setstate__", EagleDecodingConfigSetstate);

    // Guided decoding params
    auto pyGuidedDecodingParams = nb::class_<tle::GuidedDecodingParams>(m, "GuidedDecodingParams");

    nb::enum_<tle::GuidedDecodingParams::GuideType>(pyGuidedDecodingParams, "GuideType")
        .value("JSON", tle::GuidedDecodingParams::GuideType::kJSON)
        .value("JSON_SCHEMA", tle::GuidedDecodingParams::GuideType::kJSON_SCHEMA)
        .value("REGEX", tle::GuidedDecodingParams::GuideType::kREGEX)
        .value("EBNF_GRAMMAR", tle::GuidedDecodingParams::GuideType::kEBNF_GRAMMAR)
        .value("STRUCTURAL_TAG", tle::GuidedDecodingParams::GuideType::kSTRUCTURAL_TAG);

    auto guidedDecodingParamsGetstate
        = [](tle::GuidedDecodingParams const& self) { return nb::make_tuple(self.getGuideType(), self.getGuide()); };

    auto guidedDecodingParamsSetstate = [](tle::GuidedDecodingParams& self, nb::tuple const& state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid GuidedDecodingParams state!");
        }
        new (&self) tle::GuidedDecodingParams(
            nb::cast<tle::GuidedDecodingParams::GuideType>(state[0]), nb::cast<std::optional<std::string>>(state[1]));
    };

    pyGuidedDecodingParams
        .def(nb::init<tle::GuidedDecodingParams::GuideType, std::optional<std::string>>(), nb::arg("guide_type"),
            nb::arg("guide") = nb::none())
        .def_prop_ro("guide_type", &tle::GuidedDecodingParams::getGuideType)
        .def_prop_ro("guide", &tle::GuidedDecodingParams::getGuide)
        .def("__getstate__", guidedDecodingParamsGetstate)
        .def("__setstate__", guidedDecodingParamsSetstate);

    auto requestGetstate = [](tle::Request const& self)
    {
        return nb::make_tuple(self.getInputTokenIds(), self.getMaxTokens(), self.getStreaming(),
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
    auto requestSetstate = [](tle::Request& self, nb::tuple const& state)
    {
        if (state.size() != 34)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        new (&self) tle::Request(nb::cast<VecTokens>(state[0]), nb::cast<SizeType32>(state[1]),
            nb::cast<bool>(state[2]), nb::cast<tle::SamplingConfig>(state[3]), nb::cast<tle::OutputConfig>(state[4]),
            nb::cast<std::optional<SizeType32>>(state[5]), nb::cast<std::optional<SizeType32>>(state[6]),
            nb::cast<std::optional<std::vector<SizeType32>>>(state[7]),
            nb::cast<std::optional<std::list<VecTokens>>>(state[8]),
            nb::cast<std::optional<std::list<VecTokens>>>(state[9]), nb::cast<std::optional<Tensor>>(state[10]),
            nb::cast<std::optional<tle::ExternalDraftTokensConfig>>(state[11]),
            nb::cast<std::optional<tle::PromptTuningConfig>>(state[12]),
            nb::cast<std::optional<tle::MultimodalInput>>(state[13]), nb::cast<std::optional<Tensor>>(state[14]),
            nb::cast<std::optional<tle::MropeConfig>>(state[15]), nb::cast<std::optional<tle::LoraConfig>>(state[16]),
            nb::cast<std::optional<tle::LookaheadDecodingConfig>>(state[17]),
            nb::cast<std::optional<tle::KvCacheRetentionConfig>>(state[18]),
            nb::cast<std::optional<std::string>>(state[19]),
            nb::cast<std::optional<tle::LogitsPostProcessor>>(state[20]), nb::cast<std::optional<VecTokens>>(state[21]),
            nb::cast<std::optional<IdType>>(state[22]), nb::cast<bool>(state[23]),
            nb::cast<tle::PriorityType>(state[24]), nb::cast<tle::RequestType>(state[25]),
            nb::cast<std::optional<tle::ContextPhaseParams>>(state[26]),
            nb::cast<std::optional<tle::Tensor>>(state[27]), nb::cast<std::optional<SizeType32>>(state[28]),
            nb::cast<std::optional<tle::Tensor>>(state[29]), 1, nb::cast<std::optional<tle::EagleConfig>>(state[30]),
            nb::cast<std::optional<tle::Tensor>>(state[31]),
            nb::cast<std::optional<tle::GuidedDecodingParams>>(state[32]),
            nb::cast<std::optional<tle::CacheSaltIDType>>(state[33]));
    };

    nb::class_<tle::Request> request(m, "Request", nb::dynamic_attr());
    request
        .def(nb::init<tle::VecTokens,                           // inputTokenIds
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
        nb::arg("input_token_ids"),
        nb::arg("max_tokens"),
        nb::kw_only(),
        nb::arg("streaming") = false,
        nb::arg("sampling_config") = tle::SamplingConfig(),
        nb::arg("output_config") = tle::OutputConfig(),
        nb::arg("end_id") = nb::none(),
        nb::arg("pad_id") = nb::none(),
        nb::arg("position_ids") = nb::none(),
        nb::arg("bad_words") = nb::none(),
        nb::arg("stop_words") = nb::none(),
        nb::arg("embedding_bias") = nb::none(),
        nb::arg("external_draft_tokens_config") = nb::none(),
        nb::arg("prompt_tuning_config") = nb::none(),
        nb::arg("multimodal_input") = nb::none(),
        nb::arg("multimodal_embedding") = nb::none(),
        nb::arg("mrope_config") = nb::none(),
        nb::arg("lora_config") = nb::none(),
        nb::arg("lookahead_config") = nb::none(),
        nb::arg("kv_cache_retention_config") = nb::none(),
        nb::arg("logits_post_processor_name") = nb::none(),
        nb::arg("logits_post_processor") = nb::none(),
        nb::arg("encoder_input_token_ids") = nb::none(),
        nb::arg("client_id") = nb::none(),
        nb::arg("return_all_generated_tokens") = false,
        nb::arg("priority") = tle::Request::kDefaultPriority,
        nb::arg("type") = tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
        nb::arg("context_phase_params") = nb::none(),
        nb::arg("encoder_input_features") = nb::none(),
        nb::arg("encoder_output_length") = nb::none(),
        nb::arg("cross_attention_mask") = nb::none(),
        nb::arg("num_return_sequences") = 1,
        nb::arg("eagle_config") = nb::none(),
        nb::arg("skip_cross_attn_blocks") = nb::none(),
        nb::arg("guided_decoding_params") = nb::none(),
        nb::arg("language_adapter_uid") = nb::none(),
        nb::arg("allotted_time_ms") = nb::none(),
        nb::arg("cache_salt_id") = nb::none()
    )             // clang-format on
        .def_prop_ro("input_token_ids", &tle::Request::getInputTokenIds)
        .def_prop_ro("max_tokens", &tle::Request::getMaxTokens)
        .def_prop_rw("streaming", &tle::Request::getStreaming, &tle::Request::setStreaming)
        .def_prop_rw("sampling_config", &tle::Request::getSamplingConfig, &tle::Request::setSamplingConfig)
        .def_prop_rw("output_config", &tle::Request::getOutputConfig, &tle::Request::setOutputConfig)
        .def_prop_rw("end_id", &tle::Request::getEndId, &tle::Request::setEndId)
        .def_prop_rw("pad_id", &tle::Request::getPadId, &tle::Request::setPadId)
        .def_prop_rw("position_ids", &tle::Request::getPositionIds, &tle::Request::setPositionIds)
        .def_prop_rw("bad_words", &tle::Request::getBadWords, &tle::Request::setBadWords)
        .def_prop_rw("stop_words", &tle::Request::getStopWords, &tle::Request::setStopWords)
        .def_prop_rw("embedding_bias", &tle::Request::getEmbeddingBias, &tle::Request::setEmbeddingBias)
        .def_prop_rw("external_draft_tokens_config", &tle::Request::getExternalDraftTokensConfig,
            &tle::Request::setExternalDraftTokensConfig)
        .def_prop_rw("prompt_tuning_config", &tle::Request::getPromptTuningConfig, &tle::Request::setPromptTuningConfig)
        .def_prop_rw("multimodal_input", &tle::Request::getMultimodalInput, &tle::Request::setMultimodalInput)
        .def_prop_rw(
            "multimodal_embedding", &tle::Request::getMultimodalEmbedding, &tle::Request::setMultimodalEmbedding)
        .def_prop_rw("mrope_config", &tle::Request::getMropeConfig, &tle::Request::setMropeConfig)
        .def_prop_rw("lora_config", &tle::Request::getLoraConfig, &tle::Request::setLoraConfig)
        .def_prop_rw("lookahead_config", &tle::Request::getLookaheadConfig, &tle::Request::setLookaheadConfig)
        .def_prop_rw("kv_cache_retention_config", &tle::Request::getKvCacheRetentionConfig,
            &tle::Request::setKvCacheRetentionConfig)
        .def_prop_rw("logits_post_processor_name", &tle::Request::getLogitsPostProcessorName,
            &tle::Request::setLogitsPostProcessorName)
        .def_prop_rw(
            "logits_post_processor", &tle::Request::getLogitsPostProcessor, &tle::Request::setLogitsPostProcessor)
        .def_prop_rw(
            "encoder_input_token_ids", &tle::Request::getEncoderInputTokenIds, &tle::Request::setEncoderInputTokenIds)
        .def_prop_rw("client_id", &tle::Request::getClientId, &tle::Request::setClientId)
        .def_prop_rw("return_all_generated_tokens", &tle::Request::getReturnAllGeneratedTokens,
            &tle::Request::setReturnAllGeneratedTokens)
        .def_prop_rw("request_type", &tle::Request::getRequestType, &tle::Request::setRequestType)
        .def_prop_rw(
            "encoder_input_features", &tle::Request::getEncoderInputFeatures, &tle::Request::setEncoderInputFeatures)
        .def_prop_rw("cross_attention_mask", &tle::Request::getCrossAttentionMask, &tle::Request::setCrossAttentionMask)
        .def_prop_rw("eagle_config", &tle::Request::getEagleConfig, &tle::Request::setEagleConfig)
        .def_prop_rw(
            "skip_cross_attn_blocks", &tle::Request::getSkipCrossAttnBlocks, &tle::Request::setSkipCrossAttnBlocks)
        .def_prop_rw(
            "guided_decoding_params", &tle::Request::getGuidedDecodingParams, &tle::Request::setGuidedDecodingParams)
        .def_prop_rw("allotted_time_ms", &tle::Request::getAllottedTimeMs, &tle::Request::setAllottedTimeMs)
        .def_prop_rw("cache_salt_id", &tle::Request::getCacheSaltID, &tle::Request::setCacheSaltID)
        .def_prop_rw("context_phase_params", &tle::Request::getContextPhaseParams, &tle::Request::setContextPhaseParams)
        .def("__getstate__", requestGetstate)
        .def("__setstate__", requestSetstate);
    request.attr("BATCHED_POST_PROCESSOR_NAME") = tle::Request::kBatchedPostProcessorName;

    nb::class_<tle::SpeculativeDecodingFastLogitsInfo>(m, "SpeculativeDecodingFastLogitsInfo")
        .def(nb::init<>())
        .def_rw("draft_request_id", &tle::SpeculativeDecodingFastLogitsInfo::draftRequestId)
        .def_rw("draft_participant_id", &tle::SpeculativeDecodingFastLogitsInfo::draftParticipantId)
        .def("to_tensor", &tle::SpeculativeDecodingFastLogitsInfo::toTensor);

    auto requestPerfMetrics = nb::class_<tle::RequestPerfMetrics>(m, "RequestPerfMetrics");

    auto timingMetricsGetstate = [](tle::RequestPerfMetrics::TimingMetrics const& self)
    {
        return nb::make_tuple(self.arrivalTime, self.firstScheduledTime, self.firstTokenTime, self.lastTokenTime,
            self.kvCacheTransferStart, self.kvCacheTransferEnd, self.kvCacheSize);
    };
    auto timingMetricsSetstate = [](tle::RequestPerfMetrics::TimingMetrics& timingMetrics, nb::tuple const& state)
    {
        if (state.size() != 7)
        {
            throw std::runtime_error("Invalid TimingMetrics state!");
        }
        new (&timingMetrics)
            tle::RequestPerfMetrics::TimingMetrics{nb::cast<tle::RequestPerfMetrics::TimePoint>(state[0]),
                nb::cast<tle::RequestPerfMetrics::TimePoint>(state[1]),
                nb::cast<tle::RequestPerfMetrics::TimePoint>(state[2]),
                nb::cast<tle::RequestPerfMetrics::TimePoint>(state[3]),
                nb::cast<tle::RequestPerfMetrics::TimePoint>(state[4]),
                nb::cast<tle::RequestPerfMetrics::TimePoint>(state[5]), nb::cast<size_t>(state[6])};
    };
    nb::class_<tle::RequestPerfMetrics::TimingMetrics>(m, "TimingMetrics")
        .def(nb::init<>())
        .def_rw("arrival_time", &tle::RequestPerfMetrics::TimingMetrics::arrivalTime)
        .def_rw("first_scheduled_time", &tle::RequestPerfMetrics::TimingMetrics::firstScheduledTime)
        .def_rw("first_token_time", &tle::RequestPerfMetrics::TimingMetrics::firstTokenTime)
        .def_rw("last_token_time", &tle::RequestPerfMetrics::TimingMetrics::lastTokenTime)
        .def_rw("kv_cache_transfer_start", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferStart)
        .def_rw("kv_cache_transfer_end", &tle::RequestPerfMetrics::TimingMetrics::kvCacheTransferEnd)
        .def_rw("kv_cache_size", &tle::RequestPerfMetrics::TimingMetrics::kvCacheSize)
        .def("__getstate__", timingMetricsGetstate)
        .def("__setstate__", timingMetricsSetstate);

    auto kvCacheMetricsGetstate = [](tle::RequestPerfMetrics::KvCacheMetrics const& self)
    {
        return nb::make_tuple(self.numTotalAllocatedBlocks, self.numNewAllocatedBlocks, self.numReusedBlocks,
            self.numMissedBlocks, self.kvCacheHitRate);
    };
    auto kvCacheMetricsSetstate = [](tle::RequestPerfMetrics::KvCacheMetrics& kvCacheMetrics, nb::tuple const& state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid KvCacheMetrics state!");
        }
        new (&kvCacheMetrics)
            tle::RequestPerfMetrics::KvCacheMetrics{nb::cast<SizeType32>(state[0]), nb::cast<SizeType32>(state[1]),
                nb::cast<SizeType32>(state[2]), nb::cast<SizeType32>(state[3]), nb::cast<float>(state[4])};
    };
    nb::class_<tle::RequestPerfMetrics::KvCacheMetrics>(m, "KvCacheMetrics")
        .def(nb::init<>())
        .def_rw("num_total_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numTotalAllocatedBlocks)
        .def_rw("num_new_allocated_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numNewAllocatedBlocks)
        .def_rw("num_reused_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numReusedBlocks)
        .def_rw("num_missed_blocks", &tle::RequestPerfMetrics::KvCacheMetrics::numMissedBlocks)
        .def_rw("kv_cache_hit_rate", &tle::RequestPerfMetrics::KvCacheMetrics::kvCacheHitRate)
        .def("__getstate__", kvCacheMetricsGetstate)
        .def("__setstate__", kvCacheMetricsSetstate);

    auto speculativeDecodingMetricsGetstate = [](tle::RequestPerfMetrics::SpeculativeDecodingMetrics const& self)
    { return nb::make_tuple(self.acceptanceRate, self.totalAcceptedDraftTokens, self.totalDraftTokens); };
    auto speculativeDecodingMetricsSetstate
        = [](tle::RequestPerfMetrics::SpeculativeDecodingMetrics& speculativeDecodingMetrics, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid SpeculativeDecodingMetrics state!");
        }
        new (&speculativeDecodingMetrics) tle::RequestPerfMetrics::SpeculativeDecodingMetrics{
            nb::cast<float>(state[0]), nb::cast<SizeType32>(state[1]), nb::cast<SizeType32>(state[2])};
    };

    nb::class_<tle::RequestPerfMetrics::SpeculativeDecodingMetrics>(m, "SpeculativeDecodingMetrics")
        .def(nb::init<>())
        .def_rw("acceptance_rate", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::acceptanceRate)
        .def_rw("total_accepted_draft_tokens",
            &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalAcceptedDraftTokens)
        .def_rw("total_draft_tokens", &tle::RequestPerfMetrics::SpeculativeDecodingMetrics::totalDraftTokens)
        .def("__getstate__", speculativeDecodingMetricsGetstate)
        .def("__setstate__", speculativeDecodingMetricsSetstate);

    auto requestPerfMetricsGetstate = [](tle::RequestPerfMetrics const& self)
    {
        return nb::make_tuple(self.timingMetrics, self.kvCacheMetrics, self.speculativeDecoding, self.firstIter,
            self.lastIter, self.iter);
    };
    auto requestPerfMetricsSetstate = [](tle::RequestPerfMetrics& self, nb::tuple const& state)
    {
        if (state.size() != 6)
        {
            throw std::runtime_error("Invalid RequestPerfMetrics state!");
        }
        new (&self) tle::RequestPerfMetrics{nb::cast<tle::RequestPerfMetrics::TimingMetrics>(state[0]),
            nb::cast<tle::RequestPerfMetrics::KvCacheMetrics>(state[1]),
            nb::cast<tle::RequestPerfMetrics::SpeculativeDecodingMetrics>(state[2]),
            nb::cast<std::optional<tle::IterationType>>(state[3]),
            nb::cast<std::optional<tle::IterationType>>(state[4]),
            nb::cast<std::optional<tle::IterationType>>(state[5])};
    };

    // There's a circular dependency between the declaration of the TimingMetrics and RequestPerfMetrics bindings.
    // Defer definition of the RequestPerfMetrics bindings until the TimingMetrics have been defined.
    requestPerfMetrics.def(nb::init<>())
        .def_rw("timing_metrics", &tle::RequestPerfMetrics::timingMetrics)
        .def_rw("kv_cache_metrics", &tle::RequestPerfMetrics::kvCacheMetrics)
        .def_rw("speculative_decoding", &tle::RequestPerfMetrics::speculativeDecoding)
        .def_rw("first_iter", &tle::RequestPerfMetrics::firstIter)
        .def_rw("last_iter", &tle::RequestPerfMetrics::lastIter)
        .def_rw("iter", &tle::RequestPerfMetrics::iter)
        .def("__getstate__", requestPerfMetricsGetstate)
        .def("__setstate__", requestPerfMetricsSetstate);

    nb::class_<tle::AdditionalOutput>(m, "AdditionalOutput")
        .def(nb::init<std::string, tle::Tensor>(), nb::arg("name"), nb::arg("output"))
        .def_rw("name", &tle::AdditionalOutput::name)
        .def_rw("output", &tle::AdditionalOutput::output);

    auto resultSetstate = [](tle::Result& self, nb::tuple const& state)
    {
        if (state.size() != 14)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        tle::Result result;
        result.isFinal = nb::cast<bool>(state[0]);
        result.outputTokenIds = nb::cast<std::vector<VecTokens>>(state[1]);
        result.cumLogProbs = nb::cast<std::optional<std::vector<float>>>(state[2]);
        result.logProbs = nb::cast<std::optional<std::vector<std::vector<float>>>>(state[3]);
        result.contextLogits = nb::cast<std::optional<Tensor>>(state[4]);
        result.generationLogits = nb::cast<std::optional<Tensor>>(state[5]);
        result.encoderOutput = nb::cast<std::optional<Tensor>>(state[6]);
        result.finishReasons = nb::cast<std::vector<tle::FinishReason>>(state[7]);
        result.sequenceIndex = nb::cast<SizeType32>(state[8]);
        result.isSequenceFinal = nb::cast<bool>(state[9]);
        result.decodingIter = nb::cast<SizeType32>(state[10]);
        result.avgDecodedTokensPerIter = nb::cast<float>(state[11]);
        result.contextPhaseParams = nb::cast<std::optional<tle::ContextPhaseParams>>(state[12]);
        result.requestPerfMetrics = nb::cast<std::optional<tle::RequestPerfMetrics>>(state[13]);
        new (&self) tle::Result(result);
    };

    auto resultGetstate = [](tle::Result const& self)
    {
        return nb::make_tuple(self.isFinal, self.outputTokenIds, self.cumLogProbs, self.logProbs, self.contextLogits,
            self.generationLogits, self.encoderOutput, self.finishReasons, self.sequenceIndex, self.isSequenceFinal,
            self.decodingIter, self.avgDecodedTokensPerIter, self.contextPhaseParams, self.requestPerfMetrics);
    };

    nb::class_<tle::Result>(m, "Result")
        .def(nb::init<>())
        .def_rw("is_final", &tle::Result::isFinal)
        .def_rw("output_token_ids", &tle::Result::outputTokenIds)
        .def_rw("cum_log_probs", &tle::Result::cumLogProbs, nb::arg("cum_log_probs").none())
        .def_rw("log_probs", &tle::Result::logProbs, nb::arg("log_probs").none())
        .def_rw("context_logits", &tle::Result::contextLogits, nb::arg("context_logits").none())
        .def_rw("generation_logits", &tle::Result::generationLogits, nb::arg("generation_logits").none())
        .def_rw("spec_dec_fast_logits_info", &tle::Result::specDecFastLogitsInfo,
            nb::arg("spec_dec_fast_logits_info").none())
        .def_rw("encoder_output", &tle::Result::encoderOutput, nb::arg("encoder_output").none())
        .def_rw("finish_reasons", &tle::Result::finishReasons)
        .def_rw("sequence_index", &tle::Result::sequenceIndex)
        .def_rw("is_sequence_final", &tle::Result::isSequenceFinal)
        .def_rw("decoding_iter", &tle::Result::decodingIter)
        .def_rw("avg_decoded_tokens_per_iter", &tle::Result::avgDecodedTokensPerIter)
        .def_rw("context_phase_params", &tle::Result::contextPhaseParams, nb::arg("context_phase_params").none())
        .def_rw("request_perf_metrics", &tle::Result::requestPerfMetrics, nb::arg("request_perf_metrics").none())
        .def_rw("additional_outputs", &tle::Result::additionalOutputs)
        .def("__getstate__", resultGetstate)
        .def("__setstate__", resultSetstate);

    m.def("deserialize_result",
        [](nb::bytes& x)
        {
            std::string str(x.c_str(), x.size());
            std::istringstream is(str);
            return tle::serialize_utils::deserialize<tle::Result>(is);
        });

    auto responseGetstate = [](tle::Response const& self)
    { return nb::make_tuple(self.getRequestId(), self.getResult(), self.getClientId()); };

    auto responseSetstate = [](tle::Response& response, nb::tuple const& state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid Request state!");
        }
        new (&response) tle::Response(
            nb::cast<SizeType32>(state[0]), nb::cast<tle::Result>(state[1]), nb::cast<SizeType32>(state[2]));
    };

    nb::class_<tle::Response>(m, "Response")
        .def(nb::init<IdType, std::string, std::optional<IdType>>(), nb::arg("request_id"), nb::arg("error_msg"),
            nb::arg("client_id") = std::nullopt)
        .def(nb::init<IdType, tle::Result, std::optional<IdType>>(), nb::arg("request_id"), nb::arg("result"),
            nb::arg("client_id") = std::nullopt)
        .def_prop_ro("request_id", &tle::Response::getRequestId)
        .def_prop_ro("client_id", &tle::Response::getClientId)
        .def("has_error", &tle::Response::hasError)
        .def_prop_ro("error_msg", &tle::Response::getErrorMsg)
        .def_prop_ro("result", &tle::Response::getResult)
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
        .def("__getstate__", responseGetstate)
        .def("__setstate__", responseSetstate);
}

} // namespace tensorrt_llm::nanobind::executor
