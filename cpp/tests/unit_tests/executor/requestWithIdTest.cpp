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

#include "tensorrt_llm/executor/requestWithId.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

using namespace tensorrt_llm::executor;

TEST(RequestWithIdTest, serializeDeserialize)
{
    std::list<VecTokens> badWords{{1, 2, 4}, {7, 8}};
    std::list<VecTokens> stopWords{{1, 2}, {7, 8, 4}};

    std::list<VecTokens> badWords2{{1}, {9}};
    std::list<VecTokens> stopWords2{{2}, {11}};

    auto embeddingBias = Tensor::cpu(DataType::kFP32, Shape({4}));
    auto encoderInputFeatures = Tensor::cpu(DataType::kFP16, Shape({3000, 1280}));

    float* biasData = reinterpret_cast<float*>(embeddingBias.getData());
    biasData[0] = 16.f;
    biasData[1] = 32.f;
    biasData[2] = 32.f;
    biasData[3] = 48.f;

    auto request1 = Request({1, 2, 3, 4}, 1000, true, SamplingConfig(1, 4, 0.77), OutputConfig(false, true), 177, 234,
        std::make_optional<std::vector<SizeType32>>({0, 1, 2, 3}), badWords, stopWords, embeddingBias,
        ExternalDraftTokensConfig({11, 22}), std::nullopt, std::nullopt);

    auto request2 = Request({100, 200, 300, 400}, 77, false, SamplingConfig(1, 1, 0.33), OutputConfig(true, false), 66,
        99, std::make_optional<std::vector<SizeType32>>({0, 1, 1, 2}), badWords2, stopWords2, embeddingBias,
        ExternalDraftTokensConfig({7, 8, 9, 10}), std::nullopt, std::nullopt);
    request2.setEncoderInputFeatures(encoderInputFeatures);

    auto samplingConfig3 = SamplingConfig(1, 1, 0.9);
    samplingConfig3.setNumReturnSequences(3);
    auto request3 = Request({37, 19, 87, 29}, 4, false, samplingConfig3, OutputConfig(false, false), 66, 99,
        std::make_optional<std::vector<SizeType32>>({0, 1, 1, 2}), badWords2, stopWords2, embeddingBias,
        ExternalDraftTokensConfig({7, 8, 9, 10}), std::nullopt, std::nullopt);

    std::vector<RequestWithId> reqWithIds;
    reqWithIds.emplace_back(RequestWithId{request1, 1});
    reqWithIds.emplace_back(RequestWithId{request2, 2});
    reqWithIds.emplace_back(RequestWithId{request3, 3, {4, 5}, std::chrono::steady_clock::now()});

    auto serialized = RequestWithId::serializeReqWithIds(reqWithIds);
    auto reqWithIdsOut = RequestWithId::deserializeReqWithIds(serialized);

    EXPECT_EQ(reqWithIdsOut.size(), reqWithIds.size());

    for (int i = 0; i < reqWithIdsOut.size(); ++i)
    {
        auto const& reqWithIdOut = reqWithIdsOut.at(i);
        auto const& reqWithId = reqWithIds.at(i);

        EXPECT_EQ(reqWithIdOut.id, reqWithId.id);
        EXPECT_EQ(reqWithIdOut.childReqIds, reqWithId.childReqIds);
        EXPECT_EQ(reqWithIdOut.queuedStart, reqWithId.queuedStart);
        auto const& reqOut = reqWithIdOut.req;
        auto const& req = reqWithId.req;
        EXPECT_EQ(reqOut.getInputTokenIds(), req.getInputTokenIds());
        EXPECT_EQ(reqOut.getMaxTokens(), req.getMaxTokens());
        EXPECT_EQ(reqOut.getSamplingConfig(), req.getSamplingConfig());
        EXPECT_EQ(reqOut.getStopWords(), req.getStopWords());
        EXPECT_EQ(reqOut.getExternalDraftTokensConfig().value().getTokens(),
            req.getExternalDraftTokensConfig().value().getTokens());
    }
}
