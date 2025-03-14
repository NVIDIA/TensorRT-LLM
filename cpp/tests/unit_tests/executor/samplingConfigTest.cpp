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

TEST(SamplingConfigTest, validInputs)
{
    {
        auto samplingConfig = SamplingConfig(1);
    }
    {
        auto samplingConfig = SamplingConfig(4);
    }

    // TopK
    {
        auto samplingConfig = SamplingConfig(4, 1);
    }
    // TopP
    {
        auto samplingConfig = SamplingConfig(4, std::nullopt, 0.8);
    }
    // Temperature
    {
        auto samplingConfig
            = SamplingConfig(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0.9);
    }
}

void testInvalid(SizeType32 beamWidth, std::optional<SizeType32> topK = std::nullopt,
    std::optional<FloatType> topP = std::nullopt, std::optional<FloatType> topPMin = std::nullopt,
    std::optional<FloatType> topPDecay = std::nullopt, std::optional<TokenIdType> topPResetIds = std::nullopt,
    std::optional<RandomSeedType> randomSeed = std::nullopt, std::optional<FloatType> temperature = std::nullopt,
    std::optional<SizeType32> minLength = std::nullopt, std::optional<FloatType> beamSearchDiversityRate = std::nullopt,
    std::optional<FloatType> repetitionPenalty = std::nullopt, std::optional<FloatType> presencePenalty = std::nullopt,
    std::optional<FloatType> frequencePenalty = std::nullopt, std::optional<FloatType> lengthPenalty = std::nullopt,
    std::optional<SizeType32> earlyStopping = std::nullopt, std::optional<SizeType32> noRepeatNgramSize = std::nullopt,
    std::optional<FloatType> minP = std::nullopt)
{
    try
    {
        auto samplingConfig = SamplingConfig(beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed,
            temperature, minLength, beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencePenalty,
            lengthPenalty, earlyStopping, noRepeatNgramSize, minP);
        FAIL() << "Expected TllmException";
    }
    catch (TllmException& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Assertion failed"));
    }
    catch (std::exception const& e)
    {
        FAIL() << "Expected TllmException";
    }
}

TEST(SamplingConfigTest, invalidInputs)
{
    // TODO: Add more validation
    // TODO: If adding setters, test setters

    // BeamWidth
    testInvalid(-1);

    // Neg topK
    testInvalid(1, -1);

    // Neg topP
    testInvalid(1, std::nullopt, -1.0f);

    // Neg TopP min
    testInvalid(4, std::nullopt, std::nullopt, -1.0f);

    // Large TopP min
    testInvalid(4, std::nullopt, std::nullopt, 2.0f);

    // Neg TopP decay
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, -1.0f);

    // Large TopP decay
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, 2.0f);

    // Neg TopP reset ids
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, -1);

    // Neg temperature
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, -0.9f);

    // Neg min length
    testInvalid(
        4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, -1);

    // Neg beam divirsity rate
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, -1.0f);

    // Zero repetition penalty
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, 0.0f);

    // Neg repetition penalty
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, -1.0f);

    // Zero no repeat ngram size
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0);

    // Neg no repeat ngram size
    testInvalid(4, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, -1);

    // min_p = 0.5 under top_p 0.9
    testInvalid(1, std::nullopt, 0.9, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        0.5);
}

TEST(SamplingConfigTest, serializeDeserialize)
{
    auto samplingConfig = SamplingConfig(1, 10, 0.77, std::nullopt, std::nullopt, std::nullopt, 999, 0.1);
    auto serializedSize = Serialization::serializedSize(samplingConfig);

    std::ostringstream os;
    Serialization::serialize(samplingConfig, os);
    EXPECT_EQ(os.str().size(), serializedSize);

    std::istringstream is(os.str());
    auto newSamplingConfig = Serialization::deserializeSamplingConfig(is);

    EXPECT_EQ(newSamplingConfig, samplingConfig);
}
