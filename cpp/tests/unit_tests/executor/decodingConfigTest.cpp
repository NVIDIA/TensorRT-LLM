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

using ::testing::_;
using ::testing::Invoke;

using namespace tensorrt_llm::executor;
using namespace tensorrt_llm::common;

TEST(ExternalDraftTokensConfigTest, ctorValidInputs)
{
    {
        auto externalDraftTokensConfig = ExternalDraftTokensConfig({1});
    }

    {
        auto tokens = VecTokens{1, 2, 3};
        SizeType32 vocabSize = 256;
        auto draftLogits = Tensor::cpu<float>({static_cast<SizeType32>(tokens.size()), vocabSize});
        auto externalDraftTokensConfig = ExternalDraftTokensConfig(tokens, draftLogits, 0.5f);
    }

    {
        auto externalDraftTokensConfig = ExternalDraftTokensConfig({1}, std::nullopt, 0.5f);
    }
}

void testInvalid(VecTokens tokens, std::optional<Tensor> logits = std::nullopt,
    std::optional<FloatType> acceptanceThreshold = std::nullopt)
{
    try
    {
        auto externalDraftTokensConfig = ExternalDraftTokensConfig(tokens, logits, acceptanceThreshold);
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

TEST(ExternalDraftTokensConfigTest, ctorInvalidInputs)
{
    // Draft tokens shouldn't be empty if using a spec decoding config
    testInvalid({});

    // Invalid acceptance threshold
    testInvalid({1, 2, 3}, std::nullopt, -1.0f);

    // Invalid draft logits shape
    SizeType32 vocabSize = 256;
    auto draftLogits = Tensor::cpu<float>({1, vocabSize});
    testInvalid({1, 2, 3}, draftLogits);
}

TEST(ExternalDraftTokensConfigTest, serializeDeserialize)
{
    auto config = ExternalDraftTokensConfig({11, 22});
    auto serializedSize = Serialization::serializedSize(config);

    std::ostringstream os;
    Serialization::serialize(config, os);
    EXPECT_EQ(os.str().size(), serializedSize);

    std::istringstream is(os.str());
    auto newConfig = Serialization::deserializeExternalDraftTokensConfig(is);

    EXPECT_EQ(newConfig.getTokens(), config.getTokens());
    EXPECT_EQ(newConfig.getLogits(), std::nullopt);
    EXPECT_EQ(newConfig.getAcceptanceThreshold(), config.getAcceptanceThreshold());
    EXPECT_EQ(newConfig.getFastLogits(), config.getFastLogits());
}
