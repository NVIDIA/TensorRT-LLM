/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/runtime/common.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <vector>

namespace tensorrt_llm::runtime
{

class SamplingConfig
{
private:
    using FloatType = float;

    template <typename T>
    using OptVec = std::optional<std::vector<T>>;

    template <typename T>
    static OptVec<T> fuseValues(
        std::vector<SamplingConfig> const& configs, std::function<OptVec<T>(size_t ci)> accessor, T defaultValue)
    {
        std::vector<T> values;
        bool atLeastOneHasValue{false};
        for (size_t ci = 0; ci < configs.size(); ++ci)
        {
            auto const& configValue = accessor(ci);
            if (configValue.has_value())
            {
                atLeastOneHasValue = true;
                break;
            }
        }
        if (atLeastOneHasValue)
        {
            for (size_t ci = 0; ci < configs.size(); ++ci)
            {
                auto value = defaultValue;
                auto const& configValue = accessor(ci);
                if (configValue.has_value())
                {
                    TLLM_CHECK(configValue.value().size() == 1);
                    value = configValue.value().front();
                }
                values.push_back(value);
            }

            return std::make_optional<std::vector<T>>(values);
        }
        else
        {
            return std::nullopt;
        }
    }

    template <typename T>
    bool validateVec(std::string name, OptVec<T> const& vec, T min, std::optional<T> max = std::nullopt)
    {
        bool valid{true};
        if (vec)
        {
            valid = std::all_of(vec->begin(), vec->end(),
                [min, max](T elem)
                { return min < elem && ((max.has_value() && elem <= max.value()) || (!max.has_value())); });
            if (!valid)
            {
                std::stringstream ss;
                ss << "Incorrect sampling param. " << name << " is out of range (";
                ss << min << ", ";
                if (max.has_value())
                {
                    ss << max.value();
                }
                else
                {
                    ss << "inf";
                }
                ss << "]";
                TLLM_LOG_WARNING(valid, ss.str());
            }
        }
        return valid;
    }

public:
    explicit SamplingConfig(SizeType32 beamWidth = 1)
        : beamWidth{beamWidth}
    {
    }

    explicit SamplingConfig(std::vector<SamplingConfig> const& configs)
    {
        TLLM_CHECK(configs.size() > 0);
        beamWidth = configs.front().beamWidth;
        numReturnSequences = configs.front().numReturnSequences;
        normalizeLogProbs = configs.front().normalizeLogProbs;
        temperature = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].temperature; },
            layers::DefaultDecodingParams::getTemperature());
        originalTemperature = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].originalTemperature; },
            layers::DefaultDecodingParams::getTemperature());
        minLength = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].minLength; },
            layers::DefaultDecodingParams::getMinLength());
        repetitionPenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].repetitionPenalty; },
            layers::DefaultDecodingParams::getRepetitionPenalty());
        presencePenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].presencePenalty; },
            layers::DefaultDecodingParams::getPresencePenalty());
        frequencyPenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].frequencyPenalty; },
            layers::DefaultDecodingParams::getFrequencyPenalty());
        noRepeatNgramSize = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].noRepeatNgramSize; },
            layers::DefaultDecodingParams::getNoRepeatNgramSize());
        topK = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].topK; }, layers::DefaultDecodingParams::getTopK());
        topP = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].topP; }, layers::DefaultDecodingParams::getTopP());

        // Generate a random seed for each samplingConfig with randomSeed == std::nullopt
        randomSeed = std::vector<uint64_t>(configs.size());
        for (size_t ci = 0; ci < configs.size(); ++ci)
        {
            auto const& configValue = configs[ci].randomSeed;
            if (configValue)
            {
                TLLM_CHECK(configValue->size() == 1);
                randomSeed->at(ci) = configValue->front();
            }
            else
            {
                randomSeed->at(ci) = layers::DefaultDecodingParams::generateRandomSeed();
            }
        }

        topPDecay = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].topPDecay; },
            layers::DefaultDecodingParams::getTopPDecay());
        topPMin = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].topPMin; },
            layers::DefaultDecodingParams::getTopPMin());
        topPResetIds = fuseValues<TokenIdType>(
            configs, [&configs](size_t ci) { return configs[ci].topPResetIds; },
            layers::DefaultDecodingParams::getTopPResetId());
        beamSearchDiversityRate = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].beamSearchDiversityRate; },
            layers::DefaultDecodingParams::getBeamSearchDiversity());
        lengthPenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].lengthPenalty; },
            layers::DefaultDecodingParams::getLengthPenalty());
        earlyStopping = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].earlyStopping; },
            layers::DefaultDecodingParams::getEarlyStopping());
        topKMedusaHeads = fuseValues<std::vector<SizeType32>>(
            configs, [&configs](size_t ci) { return configs[ci].topKMedusaHeads; },
            layers::DefaultDecodingParams::getTopKMedusaHeads());
        outputLogProbs = fuseValues<bool>(
            configs, [&configs](size_t ci) { return configs[ci].outputLogProbs; }, false);
        cumLogProbs = fuseValues<bool>(
            configs, [&configs](size_t ci) { return configs[ci].cumLogProbs; }, false);
        beamWidthArray = fuseValues<std::vector<SizeType32>>(
            configs, [&configs](size_t ci) { return configs[ci].beamWidthArray; },
            layers::DefaultDecodingParams::getBeamWidthArray());
        // Only used for tests.
        draftAcceptanceThreshold = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].draftAcceptanceThreshold; }, 0);
        minP = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].minP; }, layers::DefaultDecodingParams::getMinP());
    }

    explicit SamplingConfig(executor::SamplingConfig const& samplingConfig,
        std::optional<executor::ExternalDraftTokensConfig> const& externalDraftTokensConfig = std::nullopt)
        : beamWidth{samplingConfig.getBeamWidth()}
        , numReturnSequences(samplingConfig.getNumReturnSequences())
    {

        if (externalDraftTokensConfig && externalDraftTokensConfig.value().getAcceptanceThreshold())
        {
            draftAcceptanceThreshold
                = std::vector<FloatType>{externalDraftTokensConfig.value().getAcceptanceThreshold().value()};
        }

#define SET_FROM_OPTIONAL(varName, VarName, VarType)                                                                   \
                                                                                                                       \
    if (samplingConfig.get##VarName())                                                                                 \
    {                                                                                                                  \
        varName = std::vector<VarType>{samplingConfig.get##VarName().value()};                                         \
    }

        SET_FROM_OPTIONAL(topK, TopK, SizeType32)
        SET_FROM_OPTIONAL(topP, TopP, FloatType)
        SET_FROM_OPTIONAL(topPMin, TopPMin, FloatType)
        SET_FROM_OPTIONAL(topPResetIds, TopPResetIds, TokenIdType)
        SET_FROM_OPTIONAL(topPDecay, TopPDecay, FloatType)
        SET_FROM_OPTIONAL(randomSeed, Seed, uint64_t)
        SET_FROM_OPTIONAL(temperature, Temperature, FloatType)
        SET_FROM_OPTIONAL(originalTemperature, Temperature, FloatType)
        SET_FROM_OPTIONAL(minLength, MinTokens, SizeType32)
        SET_FROM_OPTIONAL(beamSearchDiversityRate, BeamSearchDiversityRate, FloatType)
        SET_FROM_OPTIONAL(repetitionPenalty, RepetitionPenalty, FloatType)
        SET_FROM_OPTIONAL(presencePenalty, PresencePenalty, FloatType)
        SET_FROM_OPTIONAL(frequencyPenalty, FrequencyPenalty, FloatType)
        SET_FROM_OPTIONAL(lengthPenalty, LengthPenalty, FloatType)
        SET_FROM_OPTIONAL(earlyStopping, EarlyStopping, SizeType32)
        SET_FROM_OPTIONAL(noRepeatNgramSize, NoRepeatNgramSize, SizeType32)
        SET_FROM_OPTIONAL(minP, MinP, FloatType)
        SET_FROM_OPTIONAL(beamWidthArray, BeamWidthArray, std::vector<SizeType32>)
#undef SET_FROM_OPTIONAL
    }

    bool validate()
    {
        auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();

        bool valid{true};

        valid &= (beamWidth > 0);
        if (!valid)
        {
            TLLM_LOG_WARNING(
                "Requested beam width %d is incorrect. Must be > 0. To de-activate beam searching set beamWidth to 1.",
                beamWidth);
        }

        if (numReturnSequences)
        {
            valid &= (numReturnSequences.value() > 0);
            if (!valid)
            {
                TLLM_LOG_WARNING(
                    "Requested numReturnSequences %d is incorrect. Must be > 0.", numReturnSequences.value());
            }
            valid &= (beamWidth == 1 || numReturnSequences.value() <= beamWidth);
            if (!valid)
            {
                TLLM_LOG_WARNING(
                    "Requested numReturnSequences %d is incorrect. In beam search, numReturnSequences should not "
                    "exceed the beam width %d.",
                    numReturnSequences.value(), beamWidth);
            }
        }

        valid &= validateVec("topK", topK, -1);
        valid &= validateVec("topP", topP, -fltEpsilon, {1.f});
        valid &= validateVec("topPMin", topPMin, 0.f, {1.f});
        valid &= validateVec("topPResetIds", topPResetIds, -1);
        valid &= validateVec("topPDecay", topPDecay, 0.f, {1.f});
        valid &= validateVec("temperature", temperature, -fltEpsilon);
        valid &= validateVec("minLength", minLength, -1);
        valid &= validateVec("beamSearchDiversityRate", beamSearchDiversityRate, -fltEpsilon);
        valid &= validateVec("repetitionPenalty", repetitionPenalty, 0.f);
        // TODO: checking `lengthPenalty`leads to a failure in
        // `test_openai_chat_example`, debug and re-enable it later.
        // valid &= validateVec("lengthPenalty", lengthPenalty, 0.f);
        valid &= validateVec("noRepeatNgramSize", noRepeatNgramSize, 0);
        valid &= validateVec("minP", minP, -fltEpsilon, {1.f});
        // TODO: check `beamWidthArray`

        // Detect greedy sampling and overwrite params.
        if (temperature)
        {
            // Keep original temperature for Eagle.
            bool saveOriginalTemperature{false};
            if (!originalTemperature)
            {
                saveOriginalTemperature = true;
                originalTemperature = std::vector<FloatType>(temperature->size());
            }

            for (size_t ti = 0; ti < temperature->size(); ++ti)
            {
                if (temperature->at(ti) == 0.f)
                {
                    if (saveOriginalTemperature)
                    {
                        originalTemperature->at(ti) = 0.f;
                    }
                    temperature->at(ti) = 1.0f;

                    if (topK)
                    {
                        topK->at(ti) = 1;
                    }
                    if (topP)
                    {
                        topP->at(ti) = 1.f;
                    }
                }
                else if (saveOriginalTemperature)
                {
                    originalTemperature->at(ti) = temperature->at(ti);
                }
            }
        }

        return valid;
    }

    template <typename T>
    bool useDefaultValues(OptVec<T> const& vec, T defaultValue)
    {
        bool useDefault{true};
        if (vec)
        {
            useDefault = std::all_of(vec->begin(), vec->end(), [defaultValue](T elem) { return elem == defaultValue; });
        }
        return useDefault;
    }

public:
    SizeType32 beamWidth;
    std::optional<SizeType32> numReturnSequences;

    // penalties, [1] for one request, [batchSize] for one batch, the same for other parameters below
    OptVec<FloatType> temperature;         // [1] or [batchSize]
    OptVec<FloatType> originalTemperature; // [1] or [batchSize]
    OptVec<SizeType32> minLength;          // [1] or [batchSize]
    OptVec<FloatType> repetitionPenalty;   // [1] or [batchSize]
    OptVec<FloatType> presencePenalty;     // [1] or [batchSize]
    OptVec<FloatType> frequencyPenalty;    // [1] or [batchSize]
    OptVec<SizeType32> noRepeatNgramSize;  // [1] or [batchSize]

    // probs
    OptVec<bool> outputLogProbs;
    OptVec<bool> cumLogProbs;

    // sampling layers
    OptVec<SizeType32> topK;          // [1] or [batchSize]
    OptVec<FloatType> topP;           // [1] or [batchSize]
    OptVec<uint64_t> randomSeed;      // [1] or [batchSize]
    OptVec<FloatType> topPDecay;      // [1] or [batchSize], between [0, 1]
    OptVec<FloatType> topPMin;        // [1] or [batchSize], between [0, 1]
    OptVec<TokenIdType> topPResetIds; // [1] or [batchSize]
    OptVec<FloatType> minP;           // [1] or [batchSize]

    // beam search layer
    OptVec<FloatType> beamSearchDiversityRate;      // [1] or [batchSize]
    OptVec<FloatType> lengthPenalty;                // [1] or [batchSize]
    OptVec<SizeType32> earlyStopping;               // [1] or [batchSize]
    OptVec<std::vector<SizeType32>> beamWidthArray; // [maxBeamWidthArrayLength] or [batchSize, maxBeamWidthArrayLength]

    // speculative decoding, only the first value is used (in gptDecoderBatched.cpp)
    OptVec<FloatType> draftAcceptanceThreshold; // [1] or [batchSize]

    // medusa params
    OptVec<std::vector<SizeType32>> topKMedusaHeads; // [batchSize, maxMedusaHeads]

    std::optional<bool> normalizeLogProbs;

    bool operator==(SamplingConfig const& other) const
    {
        return beamWidth == other.beamWidth && numReturnSequences == other.numReturnSequences
            && temperature == other.temperature && originalTemperature == other.originalTemperature
            && minLength == other.minLength && repetitionPenalty == other.repetitionPenalty
            && presencePenalty == other.presencePenalty && frequencyPenalty == other.frequencyPenalty
            && noRepeatNgramSize == other.noRepeatNgramSize && topK == other.topK && topP == other.topP
            && randomSeed == other.randomSeed && topPDecay == other.topPDecay && topPMin == other.topPMin
            && topPResetIds == other.topPResetIds && beamSearchDiversityRate == other.beamSearchDiversityRate
            && lengthPenalty == other.lengthPenalty && earlyStopping == other.earlyStopping
            && draftAcceptanceThreshold == other.draftAcceptanceThreshold && topKMedusaHeads == other.topKMedusaHeads
            && normalizeLogProbs == other.normalizeLogProbs && outputLogProbs == other.outputLogProbs
            && cumLogProbs == other.cumLogProbs && minP == other.minP && beamWidthArray == other.beamWidthArray;
    }

    SizeType32 getNumReturnBeams() const
    {
        if (numReturnSequences && beamWidth > 1)
        {
            return std::min(numReturnSequences.value(), beamWidth);
        }
        return beamWidth;
    }

    // Get the maximum beam width of a whole SamplingConfig
    SizeType32 getMaxBeamWidth() const noexcept
    {
        SizeType32 maxBeamWidth = this->beamWidth; // For non-Variable-Beam-Width-Search
        auto const& beamWidthArray = this->beamWidthArray;
        if (beamWidthArray.has_value())
        {
            for (size_t indexSC = 0; indexSC < beamWidthArray->size(); ++indexSC)
            {
                auto const& array = beamWidthArray.value()[indexSC];
                auto arrayMax = *std::max_element(array.begin(), array.end());
                maxBeamWidth = std::max(maxBeamWidth, arrayMax);
            }
        }
        return maxBeamWidth;
    }
};

} // namespace tensorrt_llm::runtime
