/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>

#include <limits>
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

class LlmRequestTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(LlmRequestTest, encoderInputLengthValidation)
{
    auto const inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3});
    auto const makeRequest = [&inputTokens](std::optional<SizeType32> encoderInputLength)
    {
        return tb::LlmRequest(/*requestId=*/77, /*maxNewTokens=*/10, inputTokens, texec::SamplingConfig{},
            /*isStreaming=*/false, /*multimodalHashes=*/std::nullopt, /*multimodalPositions=*/std::nullopt,
            /*multimodalLengths=*/std::nullopt,
            /*multimodalUuids=*/std::nullopt, /*loraTaskId=*/std::nullopt, /*loraWeights=*/std::nullopt,
            /*loraConfig=*/std::nullopt, /*kvCacheRetentionConfig=*/std::nullopt, /*draftTokens=*/std::nullopt,
            /*excludeInputFromOutput=*/false, /*encoderInputTokens=*/std::nullopt, texec::Request::kDefaultPriority,
            encoderInputLength);
    };

    for (SizeType32 const length : {-1, std::numeric_limits<SizeType32>::min()})
    {
        EXPECT_THROW(makeRequest(length), tc::TllmException);
    }

    for (auto const length : {std::optional<SizeType32>{}, std::optional<SizeType32>{0}, std::optional<SizeType32>{12},
             std::optional<SizeType32>{std::numeric_limits<SizeType32>::max()}})
    {
        auto const request = makeRequest(length);
        EXPECT_EQ(request.tryGetEncoderInputLen(), length);
        EXPECT_EQ(request.getState(),
            length.has_value() ? tb::LlmRequestState::kENCODER_INIT : tb::LlmRequestState::kCONTEXT_INIT);
    }
}

TEST_F(LlmRequestTest, encoderFeatureMetadata)
{
    for (bool const imageInput : {false, true})
    {
        texec::Request execReq(VecTokens{1, 2, 3}, 10);
        texec::SamplingConfig samplingConfig;
        samplingConfig.setNumReturnSequences(2);
        execReq.setSamplingConfig(samplingConfig);
        execReq.setEncoderInputFeatures(imageInput ? texec::Tensor::cpu(texec::DataType::kFP32, {12, 3, 2, 2})
                                                   : texec::Tensor::cpu(texec::DataType::kFP32, {12, 8}));
        execReq.setEncoderOutputLength(6);
        tb::LlmRequest llmReq(77, execReq);
        EXPECT_EQ(llmReq.getState(), tb::LlmRequestState::kENCODER_INIT);
        EXPECT_EQ(llmReq.tryGetEncoderInputLen(), 12);
        EXPECT_EQ(llmReq.tryGetEncoderOutputLen(), 6);
        EXPECT_NO_THROW(llmReq.validate(10, 30, 0));
        llmReq.setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
        llmReq.pause(20);
        EXPECT_EQ(llmReq.getState(), tb::LlmRequestState::kENCODER_INIT);
        EXPECT_EQ(llmReq.tryGetEncoderInputLen(), 12);
        auto child = llmReq.createChildRequest(78);
        EXPECT_EQ(child->tryGetEncoderInputLen(), 12);
        EXPECT_EQ(child->tryGetEncoderOutputLen(), 6);
    }

    texec::Request tokenReq(VecTokens{1, 2, 3}, 10);
    tokenReq.setEncoderInputTokenIds(VecTokens{4, 5});
    tb::LlmRequest encoderReq(79, tokenReq);
    EXPECT_EQ(encoderReq.tryGetEncoderInputLen(), 2);
    EXPECT_EQ(encoderReq.tryGetEncoderOutputLen(), 2);
    tb::LlmRequest decoderReq(80, texec::Request(VecTokens{1, 2, 3}, 10));
    EXPECT_FALSE(decoderReq.tryGetEncoderInputLen().has_value());
    EXPECT_FALSE(decoderReq.tryGetEncoderOutputLen().has_value());
}

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
        EXPECT_EQ(llmReq.mSamplingConfig.getNumReturnSequences(), execReq.getSamplingConfig().getNumReturnSequences());
        EXPECT_EQ(llmReq.getOrigPromptLen(), inputTokens.size());
        EXPECT_EQ(llmReq.getMaxSentTokenLen(), inputTokens.size());
        EXPECT_EQ(llmReq.getState(), tb::LlmRequestState::kCONTEXT_INIT);
        EXPECT_FALSE(llmReq.mSeqSlot);
        // No speculative decoding config, draft tokens should be empty
        EXPECT_EQ(llmReq.getNumDraftTokens(), 0);
    }
}

TEST_F(LlmRequestTest, extraIdsWithoutPromptTuning)
{
    auto const inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3});
    auto const makeRequest = [&inputTokens](VecTokenExtraIds extraIds)
    {
        texec::SamplingConfig samplingConfig;
        samplingConfig.setNumReturnSequences(2);
        return tb::LlmRequest(77, 10, inputTokens, samplingConfig, /*isStreaming=*/false,
            /*multimodalHashes=*/std::nullopt, /*multimodalPositions=*/std::nullopt,
            /*multimodalLengths=*/std::nullopt, /*multimodalUuids=*/std::nullopt,
            /*loraTaskId=*/std::nullopt, /*loraWeights=*/std::nullopt, /*loraConfig=*/std::nullopt,
            /*kvCacheRetentionConfig=*/std::nullopt, /*draftTokens=*/std::nullopt,
            /*excludeInputFromOutput=*/false, /*encoderInputTokens=*/std::nullopt, texec::Request::kDefaultPriority,
            /*encoderInputLength=*/std::nullopt,
            /*encoderOutputLength=*/std::nullopt, tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
            std::make_shared<VecTokenExtraIds>(std::move(extraIds)));
    };

    EXPECT_THROW(makeRequest(VecTokenExtraIds{1, 1}), tc::TllmException);
    auto request = makeRequest(VecTokenExtraIds{1, 1, 0});
    auto const other = makeRequest(VecTokenExtraIds{2, 2, 0});
    EXPECT_NE(request.getUniqueTokens(0), other.getUniqueTokens(0));
    auto const child = request.createChildRequest(78);
    EXPECT_EQ(child->getUniqueTokens(0), request.getUniqueTokens(0));
    EXPECT_EQ(child->getOrigPromptLen(), inputTokens->size());
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

            llmReq.validate(2, 1000, 0);
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

            llmReq.validate(500, 1000, 0);
        };
        lambdaErrMsgs.emplace_back(lambda, "beamWidth > 0");
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
        llmReq.validate(10, 60, 0);
        EXPECT_EQ(llmReq.mMaxNewTokens, 60 - inputTokens.size());
    }
    {
        // Validate output len truncation w draft tokens
        texec::Request execReq(inputTokens, maxNewTokens);
        tb::LlmRequest llmReq(requestId, execReq);
        llmReq.validate(10, 60, 2);
        EXPECT_EQ(llmReq.mMaxNewTokens, 60 - inputTokens.size() - 2);
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
        llmReq.validate(10, 60, 2, false);
    }
}

TEST_F(LlmRequestTest, pause)
{

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{1, 2, 3, 4, 5});
    SizeType32 maxNewTokens(66);
    tb::LlmRequest::RequestIdType requestId{77};

    tb::LlmRequest llmReq(requestId, maxNewTokens, inputTokens, tensorrt_llm::executor::SamplingConfig(1), false);

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
    tb::LlmRequest llmReq(
        requestId, maxNewTokens, inputTokens, tensorrt_llm::executor::SamplingConfig(beamWidth), streaming);

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
    tensorrt_llm::DataType dtype{tensorrt_llm::DataType::kHALF};

    tensorrt_llm::executor::SamplingConfig samplingConfig(1);
    samplingConfig.setSeed(texec::RandomSeedType{7});

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

    samplingConfig.setNumReturnSequences(3);
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
        EXPECT_EQ(childReq1->mSamplingConfig.getSeed().value(), texec::RandomSeedType{8});
        EXPECT_EQ(llmReq2.mSamplingConfig.getSeed().value(), texec::RandomSeedType{7});
        EXPECT_FALSE(childReq1->mSeqSlot);
    }

    {
        auto childReq2 = llmReq2.createChildRequest(79);
        auto childRequests = llmReq2.getChildRequests();
        EXPECT_EQ(childRequests.size(), 2);
        EXPECT_EQ(childRequests.at(0), childReq1);
        EXPECT_EQ(childRequests.at(1), childReq2);
        EXPECT_EQ(childReq2->mSamplingConfig.getSeed().value(), texec::RandomSeedType{9});
        EXPECT_EQ(childReq1->mSamplingConfig.getSeed().value(), texec::RandomSeedType{8});
        EXPECT_EQ(llmReq2.mSamplingConfig.getSeed().value(), texec::RandomSeedType{7});
    }
}

TEST_F(LlmRequestTest, childrenOwnMutableInputsAndFreshRuntimeState)
{
    texec::SamplingConfig samplingConfig;
    samplingConfig.setNumReturnSequences(3);
    texec::Request input(VecTokens{1, 2, 3}, 10, false, samplingConfig);
    input.setEncoderInputTokenIds(VecTokens{4, 5});
    input.setEncoderOutputLength(1);
    tb::LlmRequest parent(77, input);
    parent.setDraftTokens(std::make_shared<VecTokens>(VecTokens{6, 7}));
    parent.mSeqSlot = 9;

    auto const child = parent.createChildRequest(78);
    auto const sibling = parent.createChildRequest(79);
    EXPECT_EQ(parent.getChildRequests().at(0).get(), child.get());
    EXPECT_EQ(parent.getChildRequests().at(1).get(), sibling.get());
    EXPECT_TRUE(child->getChildRequests().empty());
    EXPECT_TRUE(sibling->getChildRequests().empty());
    EXPECT_FALSE(child->mSeqSlot.has_value());
    EXPECT_EQ(child->getDecodingIter(), 0);
    EXPECT_EQ(child->getContextCurrentPosition(), 0);
    EXPECT_FALSE(child->getContextProgress());
    EXPECT_EQ(child->tryGetEncoderOutputLen(), 1);
    EXPECT_EQ(parent.mSamplingConfig.getSeed(), texec::RandomSeedType{0});
    EXPECT_EQ(child->mSamplingConfig.getSeed(), texec::RandomSeedType{1});
    EXPECT_EQ(sibling->mSamplingConfig.getSeed(), texec::RandomSeedType{2});

    child->getDraftTokens()->pop_back();
    child->getEncoderTokens().value()->push_back(8);
    EXPECT_EQ(*parent.getDraftTokens(), (VecTokens{6, 7}));
    EXPECT_EQ(*sibling->getDraftTokens(), (VecTokens{6, 7}));
    EXPECT_EQ(*parent.getEncoderTokens().value(), (VecTokens{4, 5}));
    EXPECT_EQ(*sibling->getEncoderTokens().value(), (VecTokens{4, 5}));
}

TEST_F(LlmRequestTest, rejectChildConstructionAfterExecution)
{
    texec::SamplingConfig samplingConfig;
    samplingConfig.setNumReturnSequences(2);
    tb::LlmRequest parent(77, texec::Request(VecTokens{1, 2, 3}, 10, false, samplingConfig));
    parent.addNewTokens(VecTokens{4});
    parent.setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
    EXPECT_THROW(parent.createChildRequest(78), tc::TllmException);
    EXPECT_TRUE(parent.getChildRequests().empty());
    EXPECT_FALSE(parent.mSamplingConfig.getSeed().has_value());
}

TEST_F(LlmRequestTest, childRegistrationRejectsDuplicateIdentity)
{
    texec::SamplingConfig samplingConfig;
    samplingConfig.setNumReturnSequences(3);
    tb::LlmRequest parent(77, texec::Request(VecTokens{1, 2, 3}, 10, false, samplingConfig));
    parent.setDraftTokens(nullptr);
    EXPECT_THROW(parent.createChildRequest(77), tc::TllmException);
    EXPECT_TRUE(parent.getChildRequests().empty());
    auto const child = parent.createChildRequest(78);
    EXPECT_EQ(child->getNumDraftTokens(), 0);
    EXPECT_THROW(parent.createChildRequest(78), tc::TllmException);
    EXPECT_THROW(parent.registerChildRequest(child), tc::TllmException);
    EXPECT_THROW(child->createChildRequest(79), tc::TllmException);
    EXPECT_EQ(parent.getChildRequests().size(), 1);
    auto const sibling = parent.createChildRequest(79);
    EXPECT_EQ(sibling->mSamplingConfig.getSeed(), texec::RandomSeedType{2});
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

TEST_P(ParamTest, createResult)
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

    tensorrt_llm::executor::SamplingConfig samplingConfig(beamWidth);
    // numReturnSequences = nullopt, otherwise.
    if (beamWidth == 1 || numReturnSequences < beamWidth)
    {
        samplingConfig.setNumReturnSequences(numReturnSequences);
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
        auto resultOpt = llmReq->createResult();
        EXPECT_FALSE(resultOpt);
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
            auto resultOpt = llmReq->createResult();
            EXPECT_TRUE(streaming == resultOpt.has_value());

            for (int beamIdx = 0; beamIdx < numReturnBeams; ++beamIdx)
            {
                if (streaming)
                {
                    auto const& result = resultOpt.value();
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

            resultOpt = llmReq->createResult();
            EXPECT_FALSE(resultOpt);
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
        auto resultOpt = llmReq->createResult();

        if (!streaming && llmRequests.at(seqIdx)->getState() != tb::LlmRequestState::kGENERATION_COMPLETE)
        {
            EXPECT_FALSE(resultOpt);
            continue;
        }

        EXPECT_TRUE(resultOpt) << "seqIdx " << seqIdx;

        auto const& result = resultOpt.value();
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
            auto resultOpt = llmRequests.at(seqIdx)->createResult();
            EXPECT_TRUE(resultOpt) << "seqIdx " << seqIdx;

            auto const& result = resultOpt.value();
            // All sequences have finished.
            EXPECT_TRUE(result.isSequenceFinal) << "seqIdx " << seqIdx;
            EXPECT_TRUE(result.isFinal) << "seqIdx " << seqIdx;
        }
    }
}

// Regression test for nvbug/5961736: createResult() must produce a valid
// response with contextPhaseParams when the request is in
// kDISAGG_CONTEXT_COMPLETE, not just kDISAGG_CONTEXT_TRANS_IN_PROGRESS.
// Without the fix, createResult() returns nullopt for CONTEXT_COMPLETE,
// causing ctx_request_id=None in the disaggregated serving response.
TEST_F(LlmRequestTest, createResultDisaggContextComplete)
{
    VecTokens inputTokens{1, 2, 3, 4, 5};
    SizeType32 maxNewTokens{10};
    texec::IdType requestId{42};

    // Build an executor::Request and configure it as context-only with ContextPhaseParams.
    texec::Request execReq(inputTokens, maxNewTokens);
    execReq.setRequestType(texec::RequestType::REQUEST_TYPE_CONTEXT_ONLY);
    texec::ContextPhaseParams ctxParams({100}, requestId, static_cast<void*>(nullptr), std::nullopt);
    execReq.setContextPhaseParams(std::move(ctxParams));

    tb::LlmRequest llmReq(requestId, execReq);
    EXPECT_TRUE(llmReq.isContextOnlyRequest());

    // Add a generated token (required by createResult's firstGenTokens extraction).
    llmReq.addNewTokens(VecTokens{42});

    // Verify isFinished() covers DISAGG_CONTEXT_COMPLETE.
    llmReq.setState(tb::LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
    EXPECT_TRUE(llmReq.isFinished());

    // This is the regression case — without the fix, createResult() returns nullopt
    // because DISAGG_CONTEXT_COMPLETE was not handled by createResult's early guard
    // or its context-phase branch.
    auto resultOpt = llmReq.createResult();
    ASSERT_TRUE(resultOpt.has_value()) << "createResult() must not return nullopt for DISAGG_CONTEXT_COMPLETE";
    EXPECT_TRUE(resultOpt->contextPhaseParams.has_value())
        << "contextPhaseParams must be populated for context-only DISAGG_CONTEXT_COMPLETE requests";
    EXPECT_EQ(resultOpt->contextPhaseParams->getReqId(), requestId);
    EXPECT_TRUE(resultOpt->isSequenceFinal);
}

TEST_F(LlmRequestTest, generationOnlyRequestAdoptsContextPhaseDraftTokens)
{
    VecTokens inputTokens{1, 2, 3, 4, 5};
    VecTokens firstGenTokens{100};
    VecTokens draftTokens{101, 102, 103};
    SizeType32 maxNewTokens{10};
    texec::IdType requestId{42};

    texec::Request execReq(inputTokens, maxNewTokens);
    execReq.setRequestType(texec::RequestType::REQUEST_TYPE_GENERATION_ONLY);
    execReq.setContextPhaseParams(
        texec::ContextPhaseParams{firstGenTokens, requestId, static_cast<void*>(nullptr), draftTokens});
    auto const expectedContextPhaseTokens = static_cast<SizeType32>(firstGenTokens.size() + draftTokens.size());
    auto const expectedDraftTokens = static_cast<SizeType32>(draftTokens.size());

    tb::LlmRequest llmReq(requestId, execReq);

    EXPECT_TRUE(llmReq.isGenerationOnlyRequest());
    EXPECT_EQ(llmReq.getNumDraftTokens(), expectedDraftTokens);
    EXPECT_EQ(*llmReq.getDraftTokens(), draftTokens);
    EXPECT_EQ(llmReq.getNumContextPhaseGenerationTokens(), expectedContextPhaseTokens);

    texec::Request lateExecReq(inputTokens, maxNewTokens);
    lateExecReq.setRequestType(texec::RequestType::REQUEST_TYPE_GENERATION_ONLY);
    tb::LlmRequest lateLlmReq(requestId, lateExecReq);
    EXPECT_EQ(lateLlmReq.getNumDraftTokens(), 0);

    lateLlmReq.setContextPhaseParams(
        texec::ContextPhaseParams{firstGenTokens, requestId, static_cast<void*>(nullptr), draftTokens});

    EXPECT_EQ(lateLlmReq.getNumDraftTokens(), expectedDraftTokens);
    EXPECT_EQ(*lateLlmReq.getDraftTokens(), draftTokens);
    EXPECT_EQ(lateLlmReq.getNumContextPhaseGenerationTokens(), expectedContextPhaseTokens);
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
