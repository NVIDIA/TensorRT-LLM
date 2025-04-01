/* * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.  * * Licensed under the Apache License,
 * Version 2.0 (the "License"); * you may not use this file except in compliance with the License.  * You may obtain a
 * copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

namespace tr = tensorrt_llm::runtime;
namespace te = tensorrt_llm::executor;
using namespace tensorrt_llm::common;

using te::SizeType32;
using te::FloatType;
using te::TokenIdType;
using te::RandomSeedType;

static std::nullopt_t constexpr no = std::nullopt;

void test(bool const useExternalDraftTokensConfig, SizeType32 beamWidth = 1, std::optional<SizeType32> topK = no,
    std::optional<FloatType> topP = no, std::optional<FloatType> topPMin = no,
    std::optional<TokenIdType> topPResetIds = no, std::optional<FloatType> topPDecay = no,
    std::optional<RandomSeedType> randomSeed = no, std::optional<FloatType> temperature = no,
    std::optional<SizeType32> minLength = no, std::optional<FloatType> beamSearchDiversityRate = no,
    std::optional<FloatType> repetitionPenalty = no, std::optional<FloatType> presencePenalty = no,
    std::optional<FloatType> frequencyPenalty = no, std::optional<FloatType> lengthPenalty = no,
    std::optional<SizeType32> earlyStopping = no, std::optional<SizeType32> noRepeatNgramSize = no,
    std::optional<SizeType32> numReturnSequences = no, std::optional<FloatType> minP = no,
    std::optional<std::vector<SizeType32>> beamWidthArray = no)
{
    // 19 parameters for SamplingConfig, from `beamWidth` to `beamWidthArray`
    try
    {
        te::SamplingConfig execSamplingCfg(beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed,
            temperature, minLength, beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty,
            lengthPenalty, earlyStopping, noRepeatNgramSize, numReturnSequences, minP, beamWidthArray);
        std::optional<te::ExternalDraftTokensConfig> specCfg = std::nullopt;
        if (useExternalDraftTokensConfig)
        {
            specCfg = te::ExternalDraftTokensConfig({1}, no, 0.5f);
        }
        tr::SamplingConfig samplingCfg(execSamplingCfg, specCfg);

        EXPECT_EQ(samplingCfg.beamWidth, execSamplingCfg.getBeamWidth());
        EXPECT_EQ(samplingCfg.numReturnSequences, execSamplingCfg.getNumReturnSequences());

        if (useExternalDraftTokensConfig)
        {
            EXPECT_TRUE(samplingCfg.draftAcceptanceThreshold.has_value());
            EXPECT_THAT(samplingCfg.draftAcceptanceThreshold.value(), testing::ElementsAre(0.5f));
        }
        else
        {
            EXPECT_EQ(samplingCfg.draftAcceptanceThreshold, no);
        }
    }
    catch (TllmException& e)
    {
        // Come here if `sc` is invalid and the exception is caught
        FAIL() << "Expected TllmException";
    }
    catch (std::exception const& e)
    {
        // Come here if `sc` is invalid but the exception is not caught
        FAIL() << "Expected TllmException";
    }
}

TEST(samplingConfigTest, validInputs)
{
    // Auto
    test(false, 1);
    // Use ExternalDraftTokensConfig
    test(true, 1);
    // TopK
    test(false, 1, 2);
    // TopP
    test(false, 1, no, 0.5f);
    // TopPMin
    test(false, 1, no, no, 0.5f);
    // TopP reset ids
    test(false, 1, no, no, no, 0);
    // TopP decay
    test(false, 1, no, no, no, no, 0.5f);
    // Seed
    test(false, 1, no, no, no, no, no, 65536);
    // Temperature
    test(false, 1, no, no, no, no, no, no, 0.5f);
    // Min token
    test(false, 1, no, no, no, no, no, no, no, 64);
    // Beam divirsity rate
    test(false, 2, no, no, no, no, no, no, no, no, 0.5f);
    // Repetition penalty
    test(false, 1, no, no, no, no, no, no, no, no, no, 1.f);
    // Presence penalty
    test(false, 1, no, no, no, no, no, no, no, no, no, no, 1.f);
    // Frequency penalty
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, 1.f);
    // Length penalty
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, no, 1.f);
    // Early stopping
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, no, no, 1.f);
    // No repeat ngram size
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 2);
    // NumReturnSequences
    test(false, 4, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 2);
    // MinP, 18 arguments
    test(false, 1, no, 0.9, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 0.5f);
    // BeamWidthArray
    test(false, 5, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no,
        std::vector<SizeType32>{2, 3, 4, 5});

    // All parameters
    {
        te::SizeType32 beamWidth{5};
        te::SizeType32 topK{1};
        te::FloatType topP{0.5f};
        te::FloatType topPMin{0.5f};
        te::SizeType32 topPResetIds{1};
        te::FloatType topPDecay{0.5f};
        te::RandomSeedType randomSeed{65536};
        te::FloatType temperature{0.5f};
        te::SizeType32 minLength{64};
        te::FloatType beamSearchDiversityRate{0.5f};
        te::FloatType repetitionPenalty{0.5f};
        te::FloatType presencePenalty{0.5f};
        te::FloatType frequencyPenalty{0.5f};
        te::FloatType lengthPenalty{0.5f};
        te::SizeType32 earlyStopping{1};
        te::SizeType32 noRepeatNgramSize{5};
        te::SizeType32 numReturnSequences{1};
        te::FloatType minP{0.5f};
        std::vector<te::SizeType32> beamWidthArray{2, 3, 4, 5};

        te::SamplingConfig execSamplingCfg(beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed,
            temperature, minLength, beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty,
            lengthPenalty, earlyStopping, noRepeatNgramSize, numReturnSequences, minP, beamWidthArray);
        te::ExternalDraftTokensConfig specCfg({1}, no, 0.5f);
        tr::SamplingConfig samplingCfg(execSamplingCfg, specCfg);
        EXPECT_EQ(samplingCfg.beamWidth, execSamplingCfg.getBeamWidth());
        EXPECT_EQ(samplingCfg.numReturnSequences, execSamplingCfg.getNumReturnSequences());
        EXPECT_THAT(samplingCfg.draftAcceptanceThreshold.value(), testing::ElementsAre(0.5f));
        EXPECT_THAT(samplingCfg.topK.value(), testing::ElementsAre(topK));
        EXPECT_THAT(samplingCfg.topP.value(), testing::ElementsAre(topP));
        EXPECT_THAT(samplingCfg.topPMin.value(), testing::ElementsAre(topPMin));
        EXPECT_THAT(samplingCfg.topPResetIds.value(), testing::ElementsAre(topPResetIds));
        EXPECT_THAT(samplingCfg.topPDecay.value(), testing::ElementsAre(topPDecay));
        EXPECT_THAT(samplingCfg.randomSeed.value(), testing::ElementsAre(randomSeed));
        EXPECT_THAT(samplingCfg.temperature.value(), testing::ElementsAre(temperature));
        EXPECT_THAT(samplingCfg.minLength.value(), testing::ElementsAre(minLength));
        EXPECT_THAT(samplingCfg.beamSearchDiversityRate.value(), testing::ElementsAre(beamSearchDiversityRate));
        EXPECT_THAT(samplingCfg.repetitionPenalty.value(), testing::ElementsAre(repetitionPenalty));
        EXPECT_THAT(samplingCfg.presencePenalty.value(), testing::ElementsAre(presencePenalty));
        EXPECT_THAT(samplingCfg.frequencyPenalty.value(), testing::ElementsAre(frequencyPenalty));
        EXPECT_THAT(samplingCfg.lengthPenalty.value(), testing::ElementsAre(lengthPenalty));
        EXPECT_THAT(samplingCfg.earlyStopping.value(), testing::ElementsAre(earlyStopping));
        EXPECT_THAT(samplingCfg.noRepeatNgramSize.value(), testing::ElementsAre(noRepeatNgramSize));
        EXPECT_THAT(samplingCfg.minP.value(), testing::ElementsAre(minP));
        auto const beamWidthArrayReturn = samplingCfg.beamWidthArray.value()[0];
        EXPECT_EQ(beamWidthArrayReturn.size(), beamWidthArray.size());
        for (int i = 0; i < (int) beamWidthArrayReturn.size(); ++i)
        {
            EXPECT_EQ(beamWidthArrayReturn[i], beamWidthArray[i]);
        }
    }
}
