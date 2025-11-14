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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace texec = tensorrt_llm::executor;
namespace tb = tensorrt_llm::batch_manager;

using VecTokens = tb::LlmRequest::VecTokens;
using SizeType32 = tb::LlmRequest::SizeType32;
using VecTokenExtraIds = tb::LlmRequest::VecTokenExtraIds;
using VecUniqueTokens = tb::LlmRequest::VecUniqueTokens;

class LlmRequestTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(LlmRequestTest, fromExecutorRequest)
{
    VecTokens inputTokens{1, 2, 3, 4, 5};
    SizeType32 maxNewTokens(66);
    texec::IdType requestId{77};
    {
        texec::Request execReq(inputTokens, maxNewTokens);
        tb::LlmRequest llmReq(requestId, execReq);
        EXPECT_EQ(llmReq.getTokens().size(), 1);
        EXPECT_EQ(llmReq.getTokens().at(0), inputTokens);
        EXPECT_EQ(llmReq.mMaxNewTokens, maxNewTokens);
        EXPECT_EQ(llmReq.mSamplingConfig.numReturnSequences, execReq.getSamplingConfig().getNumReturnSequences());
        EXPECT_EQ(llmReq.getOrigPromptLen(), inputTokens.size());
        EXPECT_EQ(llmReq.getMaxSentTokenLen(), inputTokens.size());
        EXPECT_EQ(llmReq.getState(), tb::LlmRequestState::kCONTEXT_INIT);
        EXPECT_FALSE(llmReq.mSeqSlot);
        // No speculative decoding config, draft tokens should be empty
        EXPECT_EQ(llmReq.getNumDraftTokens(), 0);
        EXPECT_FALSE(llmReq.getEmbeddingBias().has_value());
        EXPECT_FALSE(llmReq.getBadWordsList().has_value());
        EXPECT_FALSE(llmReq.getStopWordsList().has_value());
        EXPECT_FALSE(llmReq.getPromptEmbeddingTable().has_value());
        EXPECT_FALSE(llmReq.getPromptVocabSize().has_value());
    }

    // Embedding bias
    {
        texec::Request execReq(inputTokens, maxNewTokens);
        SizeType32 vocabSize = 100;
        // Try adding embedding bias
        auto embeddingBias = texec::Tensor::cpu(texec::DataType::kFP32, {vocabSize});
        execReq.setEmbeddingBias(embeddingBias);
        tb::LlmRequest llmReq(requestId, execReq);
        EXPECT_TRUE(llmReq.getEmbeddingBias().has_value());
        EXPECT_EQ(llmReq.getEmbeddingBias().value()->getShape().nbDims, 2);
        EXPECT_EQ(llmReq.getEmbeddingBias().value()->getShape().d[0], 1);
        EXPECT_EQ(llmReq.getEmbeddingBias().value()->getShape().d[1], vocabSize);
    }

    // bad/stop words
    {
        texec::Request execReq(inputTokens, maxNewTokens);
        SizeType32 vocabSize = 100;
        // Try adding embedding bias
        std::list<VecTokens> badWords{{1, 2, 3}, {4, 5}, {9}};
        std::list<VecTokens> stopWords{{1, 3}, {4}};
        execReq.setBadWords(badWords);
        execReq.setStopWords(stopWords);
        tb::LlmRequest llmReq(requestId, execReq);
        EXPECT_TRUE(llmReq.getBadWordsList().has_value());
        EXPECT_TRUE(llmReq.getStopWordsList().has_value());
        {
            auto badWordsTensor = llmReq.getBadWordsList().value();
            EXPECT_EQ(badWordsTensor->getDataType(), nvinfer1::DataType::kINT32);
            EXPECT_EQ(badWordsTensor->getShape().nbDims, 3);
            EXPECT_EQ(badWordsTensor->getShape().d[0], 1);
            EXPECT_EQ(badWordsTensor->getShape().d[1], 2);
            EXPECT_EQ(badWordsTensor->getShape().d[2], 6);
            auto data = tr::bufferCast<int32_t>(*badWordsTensor);
            EXPECT_EQ(data[0], 1);
            EXPECT_EQ(data[1], 2);
            EXPECT_EQ(data[2], 3);
            EXPECT_EQ(data[3], 4);
            EXPECT_EQ(data[4], 5);
            EXPECT_EQ(data[5], 9);
            EXPECT_EQ(data[6 + 0], 3);
            EXPECT_EQ(data[6 + 1], 5);
            EXPECT_EQ(data[6 + 2], 6);
            EXPECT_EQ(data[6 + 3], -1);
            EXPECT_EQ(data[6 + 4], -1);
            EXPECT_EQ(data[6 + 5], -1);
        }

        {
            auto stopWordsTensor = llmReq.getStopWordsList().value();
            EXPECT_EQ(stopWordsTensor->getDataType(), nvinfer1::DataType::kINT32);
            EXPECT_EQ(stopWordsTensor->getShape().nbDims, 3);
            EXPECT_EQ(stopWordsTensor->getShape().d[0], 1);
            EXPECT_EQ(stopWordsTensor->getShape().d[1], 2);
            EXPECT_EQ(stopWordsTensor->getShape().d[2], 3);
            auto data = tr::bufferCast<int32_t>(*stopWordsTensor);
            EXPECT_EQ(data[0], 1);
            EXPECT_EQ(data[1], 3);
            EXPECT_EQ(data[2], 4);
            EXPECT_EQ(data[3 + 0], 2);
            EXPECT_EQ(data[3 + 1], 3);
            EXPECT_EQ(data[3 + 2], -1);
        }
    }

    // Prompt tuning
    {
        texec::Request execReq(inputTokens, maxNewTokens);
        SizeType32 vocabSize = 100;
        SizeType32 hiddenSize = 64;
        auto embeddingTable = texec::Tensor::cpu(texec::DataType::kFP32, {vocabSize, hiddenSize});
        VecTokenExtraIds extraIds{1, 1, 1, 0, 0};
        texec::PromptTuningConfig config(embeddingTable, extraIds);
        execReq.setPromptTuningConfig(config);
        tb::LlmRequest llmReq(requestId, execReq);

        EXPECT_TRUE(llmReq.getPromptEmbeddingTable().has_value());
        EXPECT_TRUE(llmReq.getPromptVocabSize().has_value());
        EXPECT_EQ(llmReq.getPromptEmbeddingTable().value()->getShape().nbDims, 3);
        EXPECT_EQ(llmReq.getPromptEmbeddingTable().value()->getShape().d[0], 1);
        EXPECT_EQ(llmReq.getPromptEmbeddingTable().value()->getShape().d[1], vocabSize);
        EXPECT_EQ(llmReq.getPromptEmbeddingTable().value()->getShape().d[2], hiddenSize);
        EXPECT_EQ(llmReq.getPromptEmbeddingTable().value()->getDataType(), nvinfer1::DataType::kFLOAT);
        EXPECT_EQ(llmReq.getPromptVocabSize().value(), vocabSize);
        VecUniqueTokens uniqueTokens;
        for (size_t i = 0; i < inputTokens.size(); ++i)
        {
            uniqueTokens.push_back({inputTokens[i], extraIds[i]});
        }
        EXPECT_EQ(llmReq.getUniqueTokens(0), uniqueTokens);
    }
}

TEST_F(LlmRequestTest, invalidExecRequest)
{
    VecTokens inputTokens{1, 2, 3, 4, 5};
    SizeType32 maxNewTokens(66);
    texec::IdType requestId{77};

    // Input is too long
    std::list<std::pair<std::function<void()>, std::string>> lambdaErrMsgs;
    {
        auto lambda = [&inputTokens, maxNewTokens, requestId]()
        {
            texec::Request execReq(inputTokens, maxNewTokens);
            tb::LlmRequest llmReq(requestId, execReq);

            llmReq.validate(2, 1000, 0, 32000);
        };
        lambdaErrMsgs.emplace_back(lambda, "exceeds maximum input");
    }
    // Invalid beam width
    {
        auto lambda = [&inputTokens, maxNewTokens, requestId]()
        {
            texec::Request execReq(inputTokens, maxNewTokens);
            execReq.setSamplingConfig(texec::SamplingConfig(-1));
            tb::LlmRequest llmReq(requestId, execReq);

            llmReq.validate(500, 1000, 0, 32000);
        };
        lambdaErrMsgs.emplace_back(lambda, "beamWidth > 0");
    }
    // Invalid input draft len
    {
        auto lambda = [&inputTokens, maxNewTokens, requestId]()
        {
            texec::Request execReq(inputTokens, maxNewTokens);
            execReq.setExternalDraftTokensConfig(texec::ExternalDraftTokensConfig({1, 2}));
            tb::LlmRequest llmReq(requestId, execReq);

            llmReq.validate(500, 1000, 1, 32000);
        };
        lambdaErrMsgs.emplace_back(lambda, "exceeds maximum draft");
    }

    // Invalid ptable shape
    {
        auto lambda = [&inputTokens, maxNewTokens, requestId]()
        {
            texec::Request execReq(inputTokens, maxNewTokens);
            auto embeddingTable = texec::Tensor::cpu(texec::DataType::kFP32, {17, 32, 69});
            texec::PromptTuningConfig config(embeddingTable);
            execReq.setPromptTuningConfig(config);
            tb::LlmRequest llmReq(requestId, execReq);
        };
        lambdaErrMsgs.emplace_back(lambda, "Expected prompt embedding table to have shape");
    }

    // Invalid extra id vector's size
    {
        auto lambda = [&inputTokens, maxNewTokens, requestId]()
        {
            texec::Request execReq(inputTokens, maxNewTokens);
            auto embeddingTable = texec::Tensor::cpu(texec::DataType::kFP32, {4, 8});
            VecTokenExtraIds extraIds(inputTokens.size() - 1, 0);
            texec::PromptTuningConfig config(embeddingTable, extraIds);
            execReq.setPromptTuningConfig(config);
            tb::LlmRequest llmReq(requestId, execReq);
        };
        lambdaErrMsgs.emplace_back(lambda, "must be the same as input token vector size");
    }

    // Extra ids not provided when enabling kv cache reuse with prompt table
    {
        auto lambda = [&inputTokens, maxNewTokens, requestId]()
        {
            texec::Request execReq(inputTokens, maxNewTokens);
            auto embeddingTable = texec::Tensor::cpu(texec::DataType::kFP32, {4, 8});
            texec::PromptTuningConfig config(embeddingTable);
            execReq.setPromptTuningConfig(config);
            tb::LlmRequest llmReq(requestId, execReq);

            llmReq.validate(500, 1000, 1, 32000, std::nullopt, true);
        };
        lambdaErrMsgs.emplace_back(lambda, "Input token extra ids must be provided");
    }

    // Invalid endId
    {
        auto lambda = [&inputTokens, maxNewTokens, requestId]()
        {
            texec::Request execReq(
                inputTokens, maxNewTokens, false, texec::SamplingConfig(), texec::OutputConfig(), -2);
            tb::LlmRequest llmReq(requestId, execReq);
            llmReq.validate(500, 1000, 1, 32000);
        };
        lambdaErrMsgs.emplace_back(lambda, "EndId (-2) is not within acceptable range [-1, 32000)");
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
        catch (tc::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr(errMsg));
        }
        catch (std::exception const& e)
        {
            FAIL() << "Expected TllmException with " << errMsg << " got " << e.what();
        }
    }

    {
        // Validate output len truncation w/o draft tokens
        texec::Request execReq(inputTokens, maxNewTokens);
        tb::LlmRequest llmReq(requestId, execReq);
        llmReq.validate(10, 60, 0, 32000);
        EXPECT_EQ(llmReq.mMaxNewTokens, 60 - inputTokens.size());
    }
    {
        // Validate output len truncation w draft tokens
        texec::Request execReq(inputTokens, maxNewTokens);
        tb::LlmRequest llmReq(requestId, execReq);
        llmReq.validate(10, 60, 2, 32000);
        EXPECT_EQ(llmReq.mMaxNewTokens, 60 - inputTokens.size() - 2);
    }
    {
        // Validate extra ids when enabling kv cache reuse with prompt table
        texec::Request execReq(inputTokens, maxNewTokens);
        auto embeddingTable = texec::Tensor::cpu(texec::DataType::kFP32, {6, 42});
        VecTokenExtraIds extraIds(inputTokens.size(), 1);
        texec::PromptTuningConfig config(embeddingTable, extraIds);
        execReq.setPromptTuningConfig(config);
        tb::LlmRequest llmReq(requestId, execReq);

        EXPECT_EQ(static_cast<size_t>(llmReq.getOrigPromptLen()), inputTokens.size());
        llmReq.validate(500, 1000, 1, 32000, std::nullopt, true);
    }
    {
        using AdditionalModelOutput = texec::AdditionalModelOutput;
        // Validate additional context and gen outputs
        texec::Request execReq(inputTokens, maxNewTokens);
        std::vector<AdditionalModelOutput> additionalModelOutputs{
            AdditionalModelOutput{"context_gen_output", true}, AdditionalModelOutput{"gen_output", false}};
        texec::OutputConfig outputConfig;
        outputConfig.additionalModelOutputs = additionalModelOutputs;
        execReq.setOutputConfig(outputConfig);
        tb::LlmRequest llmReq(requestId, execReq);
        llmReq.validate(10, 60, 2, 32000, std::nullopt, false);
        auto const& additionalContextOutputs = llmReq.getAdditionalContextOutputs();
        EXPECT_EQ(additionalContextOutputs.count("context_gen_output"), 1);
        EXPECT_EQ(additionalContextOutputs.count("gen_output"), 0);
        auto const& additionalGenerationOutputs = llmReq.getAdditionalGenerationOutputs();
        EXPECT_EQ(additionalGenerationOutputs.count("context_gen_output"), 1);
        EXPECT_EQ(additionalGenerationOutputs.count("gen_output"), 1);
    }
}

TEST_F(LlmRequestTest, pause)
{

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3, 4, 5});
    SizeType32 maxNewTokens(66);
    tb::LlmRequest::RequestIdType requestId{77};

    tb::LlmRequest llmReq(requestId, maxNewTokens, inputTokens, tr::SamplingConfig(1), false);

    llmReq.addNewToken(1, 0);
    llmReq.addNewToken(1, 0);
    llmReq.addNewToken(1, 0);
    llmReq.addNewToken(1, 0);
    llmReq.addNewToken(1, 0);

    EXPECT_EQ(llmReq.getMaxNumGeneratedTokens(), 5);
    // maxInput is larger then num tokens
    llmReq.pause(12);
    EXPECT_EQ(llmReq.mPromptLen, 10);
    EXPECT_EQ(llmReq.mMaxNewTokens, 61);
    EXPECT_EQ(llmReq.getState(), tb::LlmRequestState::kCONTEXT_INIT);
    EXPECT_EQ(llmReq.getMaxNumGeneratedTokens(), 0);

    llmReq.addNewToken(1, 0);
    llmReq.addNewToken(1, 0);
    llmReq.addNewToken(1, 0);
    llmReq.addNewToken(1, 0);
    EXPECT_EQ(llmReq.getMaxNumGeneratedTokens(), 4);

    llmReq.pause(12);

    // max Input is now smaller than num tokens
    EXPECT_EQ(llmReq.mPromptLen, 12);
    EXPECT_EQ(llmReq.mMaxNewTokens, 59);
    EXPECT_EQ(llmReq.getState(), tb::LlmRequestState::kCONTEXT_INIT);
    EXPECT_EQ(llmReq.getMaxNumGeneratedTokens(), 0);
}

TEST_F(LlmRequestTest, testAllocateLogitsBuffer)
{
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3, 4, 5});
    SizeType32 maxNewTokens(60);
    tb::LlmRequest::RequestIdType requestId{77};

    tb::LlmRequest llmReq(requestId, maxNewTokens, inputTokens, tr::SamplingConfig(1), false);

    EXPECT_EQ(llmReq.mPromptLen, 5);

    SizeType32 vocabSizePadded = 32000;
    nvinfer1::DataType logitsDataType = nvinfer1::DataType::kFLOAT;

    // Test the allocation of context logits
    EXPECT_EQ(llmReq.getContextLogitsHost(), nullptr);
    llmReq.allocContextLogitsHost(vocabSizePadded, logitsDataType);
    auto contextLogitsHostShape = llmReq.getContextLogitsHost()->getShape();
    EXPECT_EQ(contextLogitsHostShape.nbDims, 2);
    EXPECT_EQ(contextLogitsHostShape.d[0], 5);
    EXPECT_EQ(contextLogitsHostShape.d[1], vocabSizePadded);

    // Test the allocation of generation logits
    EXPECT_EQ(llmReq.getGenerationLogitsHost(), nullptr);
    llmReq.allocGenerationLogitsHost(vocabSizePadded, logitsDataType);
    auto generationLogitsHostShape = llmReq.getGenerationLogitsHost()->getShape();
    EXPECT_EQ(generationLogitsHostShape.nbDims, 3);
    EXPECT_EQ(generationLogitsHostShape.d[0], 1);
    EXPECT_EQ(generationLogitsHostShape.d[1], maxNewTokens);
    EXPECT_EQ(generationLogitsHostShape.d[2], vocabSizePadded);

    // Test the allocation of target model's accepted token logits
    // Set draft token
    EXPECT_EQ(llmReq.getNumDraftTokens(), 0);
    auto draftTokens = std::make_shared<VecTokens>(VecTokens{7, 8, 9});
    llmReq.setDraftTokens(draftTokens);
    EXPECT_EQ(llmReq.getNumDraftTokens(), 3);
    // Clean the generation logits
    llmReq.setGenerationLogitsHost(nullptr);
    EXPECT_EQ(llmReq.getGenerationLogitsHost(), nullptr);
    llmReq.allocTargetModelAcceptedTokenLogitsHost(vocabSizePadded, logitsDataType);
    auto targetModelAcceptedTokenLogitShape = llmReq.getGenerationLogitsHost()->getShape();
    EXPECT_EQ(targetModelAcceptedTokenLogitShape.nbDims, 3);
    EXPECT_EQ(targetModelAcceptedTokenLogitShape.d[0], 1);
    EXPECT_EQ(targetModelAcceptedTokenLogitShape.d[1], 4);
    EXPECT_EQ(targetModelAcceptedTokenLogitShape.d[2], vocabSizePadded);
}

TEST_F(LlmRequestTest, testLastTokensSetIndependence)
{
    tb::LlmRequest::RequestIdType requestId{77};
    SizeType32 maxNewTokens(66);
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3, 4, 5});
    SizeType32 beamWidth = 3;
    bool streaming = false;
    tb::LlmRequest::BeamTokens expectedInitialOutput
        = {{1, 2, 3, 4, 5, 10, 20}, {1, 2, 3, 4, 5, 11, 21}, {1, 2, 3, 4, 5, 12, 22}};
    tb::LlmRequest::BeamTokens expectedOverwrittenOutput
        = {{1, 2, 3, 4, 5, 100, 200}, {1, 2, 3, 4, 5, 101, 201}, {1, 2, 3, 4, 5, 102, 202}};
    tb::LlmRequest llmReq(requestId, maxNewTokens, inputTokens, tr::SamplingConfig(beamWidth), streaming);

    // check individually set tokens
    llmReq.addNewToken(10, 0);
    llmReq.addNewToken(11, 1);
    llmReq.addNewToken(12, 2);
    auto lastTokens = llmReq.getLastTokens();
    EXPECT_EQ(lastTokens.size(), beamWidth);
    EXPECT_THAT(lastTokens, testing::ElementsAreArray({10, 11, 12}));

    // check tokens set all-at-once
    VecTokens expectedLastTokens = VecTokens({20, 21, 22});
    llmReq.addNewTokens(expectedLastTokens);
    for (SizeType32 beam = 0; beam < beamWidth; beam++)
    {
        EXPECT_EQ(llmReq.getLastTokens(beam), expectedLastTokens[beam]);
    }

    // check mTokens when written by addNewToken
    for (SizeType32 beam = 0; beam < beamWidth; beam++)
    {
        EXPECT_THAT(llmReq.getTokens(beam), testing::ElementsAreArray(expectedInitialOutput[beam]));
    }

    // check that setGeneratedTokens sets mTokens, but doesn't change lastTokens
    tb::LlmRequest::BeamTokens overwriteTokens = {{100, 200}, {101, 201}, {102, 202}};
    llmReq.setGeneratedTokens(overwriteTokens);

    for (SizeType32 beam = 0; beam < beamWidth; beam++)
    {
        EXPECT_THAT(llmReq.getTokens(beam), testing::ElementsAreArray(expectedOverwrittenOutput[beam]));
    }

    EXPECT_THAT(llmReq.getLastTokens(), testing::ElementsAreArray({20, 21, 22}));
}

TEST_F(LlmRequestTest, testCreateRequests)
{
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3, 4, 5});
    SizeType32 maxNewTokens{60};
    tb::LlmRequest::RequestIdType requestId{77};
    SizeType32 vocabSize{32};
    nvinfer1::DataType dtype{nvinfer1::DataType::kHALF};

    tr::SamplingConfig samplingConfig(1);
    samplingConfig.randomSeed = std::vector<texec::RandomSeedType>{7};

    tb::LlmRequest llmReq(requestId, maxNewTokens, inputTokens, samplingConfig, false);
    try
    {
        auto childReq = llmReq.createChildRequest(1837);
        FAIL() << "Expected an exception.";
    }
    catch (tc::TllmException const& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Cannot create child requests more than"));
    }

    samplingConfig.numReturnSequences = 3;
    tb::LlmRequest llmReq2(requestId, maxNewTokens, inputTokens, samplingConfig, false);

    auto childReq1 = llmReq2.createChildRequest(78);

    {
        EXPECT_EQ(llmReq2.getChildRequests().size(), 1);
        EXPECT_EQ(childReq1->mRequestId, 78);
        EXPECT_EQ(childReq1->getTokens().at(0), *inputTokens);
        EXPECT_EQ(childReq1->getNumTokens(0), llmReq.getNumTokens(0));
        EXPECT_EQ(childReq1->getOrigPromptLen(), llmReq.getOrigPromptLen());
        EXPECT_EQ(childReq1->mMaxNewTokens, llmReq.mMaxNewTokens);
        EXPECT_EQ(childReq1->getState(), llmReq.getState());
        EXPECT_EQ(childReq1->mSamplingConfig.randomSeed.value(), std::vector<texec::RandomSeedType>{8});
        EXPECT_EQ(llmReq2.mSamplingConfig.randomSeed.value(), std::vector<texec::RandomSeedType>{7});
        EXPECT_FALSE(childReq1->mSeqSlot);
    }

    {
        auto childReq2 = llmReq2.createChildRequest(79);
        auto childRequests = llmReq2.getChildRequests();
        EXPECT_EQ(childRequests.size(), 2);
        EXPECT_EQ(childRequests.at(0), childReq1);
        EXPECT_EQ(childRequests.at(1), childReq2);
        EXPECT_EQ(childReq2->mSamplingConfig.randomSeed.value(), std::vector<texec::RandomSeedType>{9});
        EXPECT_EQ(childReq1->mSamplingConfig.randomSeed.value(), std::vector<texec::RandomSeedType>{8});
        EXPECT_EQ(llmReq2.mSamplingConfig.randomSeed.value(), std::vector<texec::RandomSeedType>{7});
    }
}

using ParamType = std::tuple<bool, bool, bool, SizeType32, SizeType32, SizeType32>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const streaming = std::get<0>(info.param);
    auto const excludeInputFromOutput = std::get<1>(info.param);
    auto const returnAllGeneratedTokens = std::get<2>(info.param);
    auto const beamWdith = std::get<3>(info.param);
    auto const tokensPerIteration = std::get<4>(info.param);
    auto const numReturnSequences = std::get<5>(info.param);
    std::string name = "llmRequestTest";
    if (streaming)
    {
        name += "Streaming";
    }
    if (excludeInputFromOutput)
    {
        name += "ExclInput";
    }
    if (returnAllGeneratedTokens)
    {
        name += "RetAllTokens";
    }
    name += "Bw" + std::to_string(beamWdith);
    name += "TokensPerIt" + std::to_string(tokensPerIteration);
    name += "N" + std::to_string(numReturnSequences);
    return name;
}

class ParamTest : public LlmRequestTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, createResponse)
{
    bool const streaming{std::get<0>(GetParam())};
    bool const excludeInputFromOutput{std::get<1>(GetParam())};
    bool const returnAllGeneratedTokens{std::get<2>(GetParam())};
    SizeType32 const beamWidth{std::get<3>(GetParam())};
    SizeType32 const tokensPerIteration{std::get<4>(GetParam())};
    SizeType32 const numReturnSequences{std::get<5>(GetParam())};

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3, 4, 5});
    SizeType32 maxNewTokens(66);
    tb::LlmRequest::RequestIdType requestId{77};

    tr::SamplingConfig samplingConfig(beamWidth);
    // numReturnSequences = nullopt, otherwise.
    if (beamWidth == 1 || numReturnSequences < beamWidth)
    {
        samplingConfig.numReturnSequences = numReturnSequences;
    }
    auto numReturnBeams = samplingConfig.getNumReturnBeams();
    // Expect one sequence per request in beam search.
    auto numSequences = beamWidth > 1 ? 1 : numReturnSequences;

    std::vector<std::shared_ptr<tb::LlmRequest>> llmRequests;
    llmRequests.emplace_back(
        std::make_shared<tb::LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, streaming));

    {
        auto llmReq = llmRequests.at(0);
        llmReq->setExcludeInputFromOutput(excludeInputFromOutput);
        if (streaming && beamWidth > 1 && !returnAllGeneratedTokens)
        {
            EXPECT_THROW(
                llmReq->setReturnAllGeneratedTokens(returnAllGeneratedTokens), tensorrt_llm::common::TllmException);
            return;
        }
        llmReq->setReturnAllGeneratedTokens(returnAllGeneratedTokens);
    }

    if (beamWidth == 1)
    {
        auto llmReq = llmRequests.at(0);
        for (auto seqIdx = 1; seqIdx < numReturnSequences; seqIdx++)
        {
            tb::LlmRequest::RequestIdType childReqId{77 + static_cast<tb::LlmRequest::RequestIdType>(seqIdx)};
            auto childReq = llmReq->createChildRequest(childReqId);
            EXPECT_EQ(childReq->getReturnAllGeneratedTokens(), llmReq->getReturnAllGeneratedTokens());
            EXPECT_TRUE(childReq->isChild());
            llmRequests.emplace_back(std::move(childReq));
        }
    }

    for (auto& llmReq : llmRequests)
    {
        auto response = llmReq->createResponse();
        EXPECT_FALSE(response);
    }

    SizeType32 constexpr numIterations{5};
    std::vector<texec::TokenIdType> newTokens(numSequences);
    std::iota(newTokens.begin(), newTokens.end(), 1);

    for (auto seqIdx = 0; seqIdx < numSequences; seqIdx++)
    {
        auto llmReq = llmRequests.at(seqIdx);
        for (int i = 0; i < numIterations - 1; ++i)
        {
            for (int j = 0; j < tokensPerIteration; ++j)
            {
                llmReq->addNewTokens(VecTokens(numReturnBeams, newTokens.at(seqIdx)));
            }

            llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
            auto response = llmReq->createResponse();
            EXPECT_TRUE(streaming == response.has_value());

            for (int beamIdx = 0; beamIdx < numReturnBeams; ++beamIdx)
            {
                if (streaming)
                {
                    EXPECT_EQ(response.value().getRequestId(), requestId);
                    auto result = response.value().getResult();
                    EXPECT_EQ(result.outputTokenIds.size(), numReturnBeams);
                    auto const& beamTokens = result.outputTokenIds.at(beamIdx);
                    if (returnAllGeneratedTokens)
                    {
                        auto const expectedSize = (i + 1) * tokensPerIteration;
                        EXPECT_EQ(beamTokens.size(), expectedSize);
                        VecTokens expectedTokens(expectedSize, newTokens.at(seqIdx));
                        EXPECT_THAT(beamTokens, testing::ElementsAreArray(expectedTokens));
                    }
                    else
                    {
                        auto const expectedSize = tokensPerIteration;
                        EXPECT_EQ(beamTokens.size(), expectedSize);
                        VecTokens expectedTokens(expectedSize, newTokens.at(seqIdx));
                        EXPECT_THAT(beamTokens, testing::ElementsAreArray(expectedTokens));
                    }
                }
            }

            response = llmReq->createResponse();
            EXPECT_FALSE(response);
        }
    }

    for (auto seqIdx = 0; seqIdx < numSequences; seqIdx++)
    {
        for (int j = 0; j < tokensPerIteration; ++j)
        {
            llmRequests.at(seqIdx)->addNewTokens(VecTokens(numReturnBeams, newTokens.at(seqIdx)));
        }
    }

    llmRequests.at(0)->setState(tb::LlmRequestState::kGENERATION_COMPLETE);

    auto const numNewTokens = numIterations * tokensPerIteration;

    for (auto seqIdx = 0; seqIdx < numSequences; seqIdx++)
    {
        auto llmReq = llmRequests.at(seqIdx);
        auto response = llmReq->createResponse();

        if (!streaming && llmRequests.at(seqIdx)->getState() != tb::LlmRequestState::kGENERATION_COMPLETE)
        {
            EXPECT_FALSE(response);
            continue;
        }

        EXPECT_TRUE(response) << "seqIdx " << seqIdx;
        EXPECT_FALSE(response.value().hasError()) << "seqIdx " << seqIdx;

        // All response should have the same request id of the original request.
        EXPECT_EQ(response.value().getRequestId(), requestId);

        auto result = response.value().getResult();
        EXPECT_EQ(result.outputTokenIds.size(), numReturnBeams);

        // Only the first sequence has finished.
        EXPECT_EQ(result.isSequenceFinal, seqIdx == 0) << "seqIdx " << seqIdx;
        EXPECT_EQ(result.isFinal, numSequences == 1) << "seqIdx " << seqIdx;

        auto newToken = newTokens.at(seqIdx);

        for (int beamIdx = 0; beamIdx < numReturnBeams; ++beamIdx)
        {
            auto const& beamTokens = result.outputTokenIds.at(beamIdx);

            if (!streaming)
            {
                if (excludeInputFromOutput)
                {
                    EXPECT_EQ(beamTokens.size(), numNewTokens);
                    VecTokens expectedTokens(numNewTokens, newToken);
                    EXPECT_THAT(beamTokens, testing::ElementsAreArray(expectedTokens));
                }
                else
                {
                    auto const expectedSize = inputTokens->size() + numNewTokens;
                    EXPECT_EQ(beamTokens.size(), expectedSize);
                    VecTokens expectedTokens(*inputTokens);
                    expectedTokens.resize(expectedSize, newToken);
                    EXPECT_THAT(beamTokens, testing::ElementsAreArray(expectedTokens));
                }
            }
            else
            {
                if (returnAllGeneratedTokens)
                {
                    EXPECT_EQ(beamTokens.size(), numNewTokens);
                    VecTokens expectedTokens(numNewTokens, newToken);
                    EXPECT_THAT(beamTokens, testing::ElementsAreArray(expectedTokens));
                }
                else
                {
                    auto const expectedSize = tokensPerIteration;
                    EXPECT_EQ(beamTokens.size(), expectedSize);
                    VecTokens expectedTokens(expectedSize, newToken);
                    EXPECT_THAT(beamTokens, testing::ElementsAreArray(expectedTokens));
                }
            }
        }
    }

    if (numSequences > 1)
    {
        for (auto seqIdx = 1; seqIdx < numSequences; seqIdx++)
        {
            auto llmReq = llmRequests.at(seqIdx);
            for (int j = 0; j < tokensPerIteration; ++j)
            {
                llmReq->addNewTokens(VecTokens(beamWidth, newTokens.at(seqIdx)));
            }
            llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
        }

        for (auto seqIdx = 1; seqIdx < numSequences; seqIdx++)
        {
            auto response = llmRequests.at(seqIdx)->createResponse();
            EXPECT_TRUE(response) << "seqIdx " << seqIdx;
            EXPECT_FALSE(response.value().hasError()) << "seqIdx " << seqIdx;

            auto result = response.value().getResult();
            // All sequences have finished.
            EXPECT_TRUE(result.isSequenceFinal) << "seqIdx " << seqIdx;
            EXPECT_TRUE(result.isFinal) << "seqIdx " << seqIdx;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(LlmRequestTest, ParamTest,
    testing::Combine(
        // TODO: Support and add coverage for streamLLM
        testing::Values(false),
        // excludeInputFromOutput
        testing::Values(false, true),
        // returnAllGeneratedTokens
        testing::Values(false, true),
        // beamWidth
        testing::Values(1, 2),
        // tokensPerIteration
        testing::Values(1, 3),
        // numReturnSequences
        testing::Values(1, 2)),
    generateTestName);
