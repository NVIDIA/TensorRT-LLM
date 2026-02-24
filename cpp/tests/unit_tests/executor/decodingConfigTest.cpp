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
