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

TEST(RequestTest, validInputs)
{
    {
        {
            auto request = Request({1, 2, 3}, 10);
        }
        {
            auto request = Request({1, 2, 3}, 10, true);
        }
        {
            auto request = Request({1, 2, 3}, 10, true);
        }
        {
            auto samplingConfig = SamplingConfig{1};
            auto request = Request({1, 2, 3}, 10, true, std::move(samplingConfig));
        }
        {
            auto request = Request({1, 1, 2}, 1);
            auto embeddingBias = Tensor::cpu(DataType::kFP32, {20});
            request.setEmbeddingBias(embeddingBias);

            EXPECT_TRUE(request.getEmbeddingBias());
            EXPECT_EQ(request.getEmbeddingBias().value().getShape().size(), 1);
            EXPECT_EQ(request.getEmbeddingBias().value().getShape()[0], 20);
        }

        {
            auto request = Request({1, 1, 2}, 1);
            SizeType32 vocabSize = 100;
            SizeType32 hiddenSize = 64;
            auto embeddingTable = Tensor::cpu(DataType::kFP32, {vocabSize, hiddenSize});
            PromptTuningConfig config(embeddingTable);
            request.setPromptTuningConfig(config);

            EXPECT_TRUE(request.getPromptTuningConfig());
            EXPECT_EQ(request.getPromptTuningConfig().value().getEmbeddingTable().getShape().size(), 2);
            EXPECT_EQ(request.getPromptTuningConfig().value().getEmbeddingTable().getShape()[0], vocabSize);
            EXPECT_EQ(request.getPromptTuningConfig().value().getEmbeddingTable().getShape()[1], hiddenSize);
        }
        {
            auto request = Request({1, 1, 2}, 1);
            IdType clientId = 1234;
            request.setClientId(clientId);

            EXPECT_TRUE(request.getClientId());
            EXPECT_EQ(request.getClientId().value(), clientId);
        }
    }
}

TEST(RequestTest, invalidInputs)
{

    std::list<std::pair<std::function<void()>, std::string>> lambdaErrMsgs;

    // No input tokens
    {
        auto lambda = []() { auto request = Request({}, 1); };
        lambdaErrMsgs.emplace_back(std::make_pair(lambda, "!mInputTokenIds.empty()"));
    }
    // Neg output length
    {
        auto lambda = []() { auto request = Request({1, 1, 2}, -1); };
        lambdaErrMsgs.emplace_back(std::make_pair(lambda, "mMaxNewTokens > 0"));
    }

    // Embedding bias dims
    {
        auto lambda = []()
        {
            auto request = Request({1, 1, 2}, 1);
            auto embeddingBias = Tensor::cpu(DataType::kFP32, {1, 1, 1});
            request.setEmbeddingBias(embeddingBias);
        };
        lambdaErrMsgs.emplace_back(std::make_pair(lambda, ".size() == 1"));
    }

    // Embedding table has wrong shape
    {
        auto lambda = []()
        {
            SizeType32 vocabSize = 100;
            SizeType32 hiddenSize = 64;
            auto embeddingTable = Tensor::cpu(DataType::kFP32, {vocabSize, hiddenSize, 64});
            PromptTuningConfig config(embeddingTable);
            auto request = Request({1, 1, 2}, 1);
            request.setPromptTuningConfig(config);
        };
        lambdaErrMsgs.emplace_back(
            std::make_pair(lambda, "Expected prompt embedding table to have shape [vocabSize, hiddenSize]"));
    }

    for (auto& lambdaErrMsg : lambdaErrMsgs)
    {
        auto& lambda = lambdaErrMsg.first;
        auto& errMsg = lambdaErrMsg.second;
        try
        {
            lambda();
            FAIL() << "Expected failure with " << errMsg;
        }
        catch (TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr(errMsg));
        }
        catch (std::exception const& e)
        {
            FAIL() << "Expected TllmException with " << errMsg << " got " << e.what();
        }
    }
}

TEST(RequestTest, serializeDeserialize)
{
    auto embeddingTable = Tensor::cpu(DataType::kFP32, {2, 2});
    float* data = reinterpret_cast<float*>(embeddingTable.getData());
    data[0] = 123;
    data[1] = 456;
    data[2] = 789;
    data[3] = 10;

    auto request = Request({1, 2, 3, 4}, 11, true, SamplingConfig(), OutputConfig(), 112, 113,
        std::make_optional<std::vector<SizeType32>>({0, 1, 2, 3}), std::list<VecTokens>{{1, 2, 3}, {2, 3, 4}},
        std::nullopt, std::nullopt, ExternalDraftTokensConfig({2, 2, 2}),
        PromptTuningConfig(embeddingTable, VecTokenExtraIds({1, 2, 3, 4})), std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt,
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 1, 10)}, 10), "Processor",
        std::nullopt, std::nullopt, 1234, false, 0.5, RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, 1, std::nullopt, std::nullopt,
        GuidedDecodingParams(GuidedDecodingParams::GuideType::kREGEX, "\\d+"));

    auto serializedSize = Serialization::serializedSize(request);
    std::ostringstream os;
    Serialization::serialize(request, os);
    EXPECT_EQ(os.str().size(), serializedSize);

    std::istringstream is(os.str());
    auto newRequest = Serialization::deserializeRequest(is);

    EXPECT_EQ(newRequest.getInputTokenIds(), request.getInputTokenIds());
    EXPECT_EQ(newRequest.getMaxTokens(), request.getMaxTokens());
    EXPECT_EQ(newRequest.getStreaming(), request.getStreaming());
    EXPECT_EQ(newRequest.getSamplingConfig(), request.getSamplingConfig());
    EXPECT_EQ(newRequest.getEndId(), request.getEndId());
    EXPECT_EQ(newRequest.getPadId(), request.getPadId());
    EXPECT_EQ(newRequest.getPositionIds(), request.getPositionIds());
    EXPECT_EQ(newRequest.getBadWords(), request.getBadWords());
    EXPECT_EQ(newRequest.getExternalDraftTokensConfig().value().getTokens(),
        request.getExternalDraftTokensConfig().value().getTokens());
    EXPECT_TRUE(request.getLogitsPostProcessorName().has_value());
    EXPECT_TRUE(newRequest.getLogitsPostProcessorName().has_value());
    EXPECT_EQ(newRequest.getLogitsPostProcessorName().value(), request.getLogitsPostProcessorName().value());
    EXPECT_EQ(newRequest.getClientId(), request.getClientId());
    EXPECT_EQ(newRequest.getReturnAllGeneratedTokens(), request.getReturnAllGeneratedTokens());
    EXPECT_EQ(newRequest.getPriority(), request.getPriority());
    EXPECT_EQ(newRequest.getKvCacheRetentionConfig().value().getTokenRangeRetentionConfigs(),
        request.getKvCacheRetentionConfig().value().getTokenRangeRetentionConfigs());
    EXPECT_EQ(newRequest.getKvCacheRetentionConfig().value().getTokenRangeRetentionConfigs(),
        request.getKvCacheRetentionConfig().value().getTokenRangeRetentionConfigs());
    EXPECT_EQ(newRequest.getRequestType(), request.getRequestType());
    EXPECT_TRUE(request.getGuidedDecodingParams().has_value());
    EXPECT_TRUE(newRequest.getGuidedDecodingParams().has_value());
    EXPECT_EQ(request.getGuidedDecodingParams(), newRequest.getGuidedDecodingParams());

    EXPECT_TRUE(request.getPromptTuningConfig().has_value());
    EXPECT_TRUE(newRequest.getPromptTuningConfig().has_value());
    EXPECT_EQ(newRequest.getPromptTuningConfig()->getInputTokenExtraIds(),
        request.getPromptTuningConfig()->getInputTokenExtraIds());
    auto newEmbeddingTable = newRequest.getPromptTuningConfig()->getEmbeddingTable();
    EXPECT_EQ(newEmbeddingTable.getShape().size(), embeddingTable.getShape().size());
    EXPECT_EQ(newEmbeddingTable.getDataType(), embeddingTable.getDataType());
    EXPECT_EQ(newEmbeddingTable.getMemoryType(), embeddingTable.getMemoryType());
    float* newData = reinterpret_cast<float*>(newEmbeddingTable.getData());
    EXPECT_EQ(data[0], newData[0]);
    EXPECT_EQ(data[1], newData[1]);
    EXPECT_EQ(data[2], newData[2]);
    EXPECT_EQ(data[3], newData[3]);
}
