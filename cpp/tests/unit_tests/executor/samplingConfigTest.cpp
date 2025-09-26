/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>

using ::testing::_;
using ::testing::Invoke;

using namespace tensorrt_llm::executor;
using namespace tensorrt_llm::common;

static std::nullopt_t constexpr no = std::nullopt;

void test(bool const isTestValid, SizeType32 beamWidth = 1, std::optional<SizeType32> topK = no,
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
        auto sc = SamplingConfig(beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed, temperature,
            minLength, beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, lengthPenalty,
            earlyStopping, noRepeatNgramSize, numReturnSequences, minP, beamWidthArray);

        // Come here if `sc` is valid
        if (!isTestValid)
        {
            // Failed in `invalidInputs` tests
            FAIL() << "Expected TllmException";
        }
    }
    catch (TllmException& e)
    {
        // Come here if `sc` is invalid and caught
        if (isTestValid)
        {
            // Failed in `validInputs` tests
            FAIL() << "Expected TllmException";
        }
        else
        {
            // Succeeded in `invalidInputs` tests
            EXPECT_THAT(e.what(), testing::HasSubstr("Assertion failed"));
        }
    }
    catch (std::exception const& e)
    {
        // Come here if `sc` is invalid but not caught
        FAIL() << "Expected TllmException";
    }
}

TEST(SamplingConfigTest, validInputs)
{
    // Auto
    test(true, 1);
    // TopK
    test(true, 1, 2);
    // TopP
    test(true, 1, no, 0.5f);
    // TopPMin
    test(true, 1, no, no, 0.5f);
    // TopP reset ids
    test(true, 1, no, no, no, 0);
    // TopP decay
    test(true, 1, no, no, no, no, 0.5f);
    // Seed
    test(true, 1, no, no, no, no, no, 65536);
    // Temperature
    test(true, 1, no, no, no, no, no, no, 0.5f);
    // Min token
    test(true, 1, no, no, no, no, no, no, no, 64);
    // Beam divirsity rate
    test(true, 2, no, no, no, no, no, no, no, no, 0.5f);
    // Repetition penalty
    test(true, 1, no, no, no, no, no, no, no, no, no, 1.f);
    // Presence penalty
    test(true, 1, no, no, no, no, no, no, no, no, no, no, 1.f);
    // Frequency penalty
    test(true, 1, no, no, no, no, no, no, no, no, no, no, no, 1.f);
    // Length penalty
    test(true, 1, no, no, no, no, no, no, no, no, no, no, no, no, 1.f);
    // Early stopping
    test(true, 1, no, no, no, no, no, no, no, no, no, no, no, no, no, 1.f);
    // No repeat ngram size
    test(true, 1, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 2);
    // NumReturnSequences
    test(true, 4, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 2);
    // MinP
    test(true, 1, no, 0.9, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 0.5f);
    // BeamWidthArray
    test(true, 5, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no,
        std::vector<SizeType32>{2, 3, 4, 5});
}

TEST(SamplingConfigTest, invalidInputs)
{
    // Neg or zero beamWidth
    test(false, 0);

    // Neg topK
    test(false, 1, -1);

    // Neg / large topP
    test(false, 1, no, -1.f);
    test(false, 1, no, +2.f);

    // Neg / large TopPMin
    test(false, 1, no, no, -1.f);
    test(false, 1, no, no, +2.f);

    // Neg topP reset ids
    test(false, 1, no, no, no, -1);

    // Neg / large TopP decay
    test(false, 1, no, no, no, no, -1.f);
    test(false, 1, no, no, no, no, +2.f);

    // Skip seed, no test

    // Neg temperature
    test(false, 1, no, no, no, no, no, no, -0.9f);

    // Neg min length
    test(false, 1, no, no, no, no, no, no, no, -1);

    // Neg beam divirsity rate
    test(false, 2, no, no, no, no, no, no, no, no, -1.f);

    // Neg or zero repetition penalty
    test(false, 1, no, no, no, no, no, no, no, no, no, 0.f);

    // Skip presence penalty, frequency penalty, no test

    // Neg length penalty
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, no, -1);

    // Neg early stopping
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, no, no, -1);

    // Neg no repeat ngram size
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, no, no, no, -1);

    // Neg or zero numReturnSequences
    test(false, 1, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 0);

    // numReturnSequences > beamWidth
    test(false, 2, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, 4);

    // Neg minP
    test(false, 1, no, 0.9, no, no, no, no, no, no, no, no, no, no, no, no, no, no, -1.f);

    // Neg / Large minP
    test(false, 1, no, 0.9, no, no, no, no, no, no, no, no, no, no, no, no, no, no, -1.f);
    test(false, 1, no, 0.9, no, no, no, no, no, no, no, no, no, no, no, no, no, no, +2.f);

    // BeamWidthArray with neg / large beamWidth
    test(false, 4, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no,
        std::vector<SizeType32>{2, 3, 4, -1});
    test(false, 4, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no,
        std::vector<SizeType32>{2, 3, 4, 65536});
}

TEST(SamplingConfigTest, getterSetter)
{
    // Auto
    {
        auto sc = SamplingConfig();
        sc.setBeamWidth(2);
        EXPECT_EQ(sc.getBeamWidth(), 2);
    }
    // TopK
    {
        auto sc = SamplingConfig();
        sc.setTopK(2);
        EXPECT_EQ(sc.getTopK(), 2);
    }
    // TopP
    {
        auto sc = SamplingConfig();
        sc.setTopP(0.5f);
        EXPECT_EQ(sc.getTopP(), 0.5f);
    }
    // TopPMin
    {
        auto sc = SamplingConfig();
        sc.setTopPMin(0.5f);
        EXPECT_EQ(sc.getTopPMin(), 0.5f);
    }
    // TopP reset ids
    {
        auto sc = SamplingConfig();
        sc.setTopPResetIds(0);
        EXPECT_EQ(sc.getTopPResetIds(), 0);
    }
    // TopP decay
    {
        auto sc = SamplingConfig();
        sc.setTopPDecay(0.5f);
        EXPECT_EQ(sc.getTopPDecay(), 0.5f);
    }
    // Seed
    {
        auto sc = SamplingConfig();
        sc.setSeed(65536);
        EXPECT_EQ(sc.getSeed(), 65536);
    }
    // Temperature
    {
        auto sc = SamplingConfig();
        sc.setTemperature(0.5f);
        EXPECT_EQ(sc.getTemperature(), 0.5f);
    }
    // Min token
    {
        auto sc = SamplingConfig();
        sc.setMinTokens(64);
        EXPECT_EQ(sc.getMinTokens(), 64);
    }
    // Beam divirsity rate
    {
        auto sc = SamplingConfig();
        sc.setBeamSearchDiversityRate(0.5f);
        EXPECT_EQ(sc.getBeamSearchDiversityRate(), 0.5f);
    }
    // Repetition penalty
    {
        auto sc = SamplingConfig();
        sc.setRepetitionPenalty(1.f);
        EXPECT_EQ(sc.getRepetitionPenalty(), 1.f);
    }
    // Presence penalty
    {
        auto sc = SamplingConfig();
        sc.setPresencePenalty(0.5f);
        EXPECT_EQ(sc.getPresencePenalty(), 0.5f);
    }
    // Frequency penalty
    {
        auto sc = SamplingConfig();
        sc.setFrequencyPenalty(0.5f);
        EXPECT_EQ(sc.getFrequencyPenalty(), 0.5f);
    }
    // Length penalty
    {
        auto sc = SamplingConfig();
        sc.setLengthPenalty(0.5f);
        EXPECT_EQ(sc.getLengthPenalty(), 0.5f);
    }
    // Early stopping
    {
        auto sc = SamplingConfig();
        sc.setEarlyStopping(1);
        EXPECT_EQ(sc.getEarlyStopping(), 1);
    }
    // No repeat ngram size
    {
        auto sc = SamplingConfig();
        sc.setNoRepeatNgramSize(2);
        EXPECT_EQ(sc.getNoRepeatNgramSize(), 2);
    }
    // NumReturnSequences
    {
        auto sc = SamplingConfig(2);
        sc.setNumReturnSequences(2);
        EXPECT_EQ(sc.getNumReturnSequences(), 2);
    }
    // MinP
    {
        auto sc = SamplingConfig(1, no, 0.9f);
        sc.setMinP(0.5f);
        EXPECT_EQ(sc.getMinP(), 0.5f);
    }
    // BeamWidthArray
    {
        auto sc = SamplingConfig();
        std::vector<SizeType32> beamWidthArray{2, 3, 4, 5};
        sc.setBeamWidthArray(beamWidthArray);
        auto const beamWidthArrayReturn = sc.getBeamWidthArray().value();
        EXPECT_EQ(beamWidthArrayReturn.size(), beamWidthArray.size());
        for (int i = 0; i < (int) beamWidthArrayReturn.size(); ++i)
        {
            EXPECT_EQ(beamWidthArrayReturn[i], beamWidthArray[i]);
        }
    }
}

TEST(SamplingConfigTest, serializeDeserialize)
{
    auto sc = SamplingConfig(1, 10, 0.77, no, no, no, 999, 0.1);
    auto serializedSize = Serialization::serializedSize(sc);

    std::ostringstream os;
    Serialization::serialize(sc, os);
    EXPECT_EQ(os.str().size(), serializedSize);

    std::istringstream is(os.str());
    auto newSamplingConfig = Serialization::deserializeSamplingConfig(is);

    EXPECT_EQ(newSamplingConfig, sc);
}
