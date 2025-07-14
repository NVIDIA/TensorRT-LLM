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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "executorTest.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/version.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/testing/modelSpec.h"
#include "tests/utils/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

using namespace tensorrt_llm::testing;
using namespace tensorrt_llm::executor;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using tensorrt_llm::testing::KVCacheType;
using tensorrt_llm::testing::ModelSpec;

namespace
{

auto const LORA_DATA_PATH = DATA_PATH / "lora-test-weights-gpt2-tp1";
auto const LORA_WEIGHTS_FILE = LORA_DATA_PATH / "source.npy";
auto const LORA_CONFIG_FILE = LORA_DATA_PATH / "config.npy";

auto constexpr LLAMA_INPUT_FILE = "input_tokens_llama.npy";
auto constexpr LLAMA_VOCAB_SIZE_PADDED = 128256;
auto constexpr LLAMA_PAD_ID = 128001;
auto constexpr LLAMA_END_ID = 128001;

} // namespace

void testInvalidCtor(std::filesystem::path const& enginePath, ModelType modelType, ExecutorConfig executorConfig,
    std::string expectedErrMsg = "")
{
    try
    {
        auto executor = Executor(enginePath, modelType, executorConfig);

        FAIL() << "Expected TllmException";
    }
    catch (std::exception const& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr(expectedErrMsg));
    }
}

TEST_F(GptExecutorTest, version)
{
    EXPECT_STRNE(kTensorRtLlmVersion, "@TRTLLM_VERSION@");
    EXPECT_STREQ(kTensorRtLlmVersion, version());
}

TEST_F(GptExecutorTest, validCtor)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
}

TEST_F(GptExecutorTest, invalidCtor)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    std::filesystem::path invalidPath{"Bla"};

    // Invalid path
    {
        testInvalidCtor(invalidPath, ModelType::kDECODER_ONLY, executorConfig, "File does not exist");
    }
}

TEST_F(GptExecutorTest, enqueueAfterShutdown)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto requestId = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                done = response.getResult().isFinal;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    executor.shutdown();

    EXPECT_FALSE(executor.canEnqueueRequests());

    std::string expErrMsg{"Shutdown called"};
    EXPECT_THAT([&]() { auto reqId = executor.enqueueRequest(request); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { auto resp = executor.awaitResponses(); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { auto stats = executor.getLatestIterationStats(); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { auto stats = executor.getLatestRequestStats(); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
    EXPECT_THAT([&]() { executor.cancelRequest(requestId); },
        testing::Throws<tensorrt_llm::common::TllmException>(
            testing::Property(&tensorrt_llm::common::TllmException::what, testing::HasSubstr(expErrMsg))));
}

TEST_F(GptExecutorTest, missingPeftTask)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_LORA_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto loraConfig = LoraConfig{10};
    request.setLoraConfig(loraConfig);

    auto requestId = executor.enqueueRequest(request);

    bool done = false;
    std::chrono::milliseconds waitTime(mMaxWaitMs);
    auto responses = executor.awaitResponses(requestId, waitTime);
    for (auto& response : responses)
    {
        if (response.hasError())
        {
            auto err = response.getErrorMsg();
            EXPECT_EQ(err, std::string("LoRA task 10 not found in cache. Please send LoRA weights with request"));
            done = true;
        }
        else
        {
            FAIL() << "Expects error due to missing Lora weights";
        }
    }
    EXPECT_TRUE(done);
}

TEST_F(GptExecutorTest, ReturnAcceptedTokenLogits)
{
    SizeType32 constexpr beamWidth{1};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded

    // Create executor config
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setGatherGenerationLogits(true);

    // Enable kv cache reuse of executorConfig
    bool enableBlockReuse = true;
    FloatType freeGpuMemoryFraction = 0.4;
    auto kvCacheConfig
        = KvCacheConfig(enableBlockReuse, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction);
    executorConfig.setKvCacheConfig(kvCacheConfig);

    // Create executor
    auto trtEnginePath
        = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<bool> streamingOptions{false, true};

    for (auto streaming : streamingOptions)
    {
        auto request = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth));

        // Set draft tokens
        auto draftTokens = VecTokens{9, 10, 11, 12, 13}; // draft tokens
        auto draftLength = draftTokens.size();
        FloatType const acceptanceThreshold = 0.00001f;  // Ensure the draft token can be accepted
        auto externalDraftTokensConfig = ExternalDraftTokensConfig(draftTokens, std::nullopt, acceptanceThreshold);
        request.setExternalDraftTokensConfig(externalDraftTokensConfig);

        // Set return accepted token logits for this request
        OutputConfig outConfig;
        outConfig.returnGenerationLogits = true;
        request.setOutputConfig(outConfig);

        // Enqueue this request
        auto requestId = executor.enqueueRequest(request);

        bool done = false;
        int iter = 0;
        while (!done && iter < 5000)
        {
            std::chrono::milliseconds waitTime(mMaxWaitMs);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {
                    FAIL();
                }
                else
                {
                    auto result = response.getResult();
                    done = result.isFinal;
                    auto& genLogits = result.generationLogits;
                    EXPECT_TRUE(genLogits.has_value());

                    // Expected shape: (1, numAcceptedDraftToken, vocabSizePadded)
                    auto const& acceptedTokenLogitsShape = genLogits->getShape();
                    EXPECT_EQ(acceptedTokenLogitsShape.size(), 3);
                    EXPECT_EQ(acceptedTokenLogitsShape[0], 1);
                    EXPECT_LE(acceptedTokenLogitsShape[1], draftLength);     // number of accepted tokens
                    EXPECT_EQ(acceptedTokenLogitsShape[2], vocabSizePadded); // vocabSizePadded
                }
            }
            ++iter;
        }
    }
}

TEST_F(GptExecutorTest, GenerationLogitsEarlyStop)
{
    SizeType32 constexpr beamWidth{1};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    auto constexpr streaming = false;

    ExtendedRuntimePerfKnobConfig perfKnobConfig = ExtendedRuntimePerfKnobConfig();

    // Create executor config
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setExtendedRuntimePerfKnobConfig(perfKnobConfig);
    executorConfig.setGatherGenerationLogits(true);

    // Create executor
    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto const inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};

    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, modelIds.padId);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(*givenInput);

    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    BeamResult beamResult{beamWidth};
    auto const resultsPath
        = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
    beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE();
    beamResult.contextLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE();
    beamResult.genLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE();

    // Set return generation logits for this request
    OutputConfig outConfig;
    outConfig.returnGenerationLogits = true;
    outConfig.excludeInputFromOutput = true;

    // Load expected outputs for each beam width value
    auto testData = TestData::loadTestData(beamResult, *givenInput, beamWidth, manager, outConfig, modelIds);
    auto const maxSeqLen = testData.maxSeqLen;

    // Load expected outputs and inputs
    std::vector<Request> requests;
    std::vector<SizeType32> reqMaxNewTokens;

    auto constexpr reqIdx = 0;
    SizeType32 inputLen = givenInputLengths.at(reqIdx);
    auto maxNewTokens = maxSeqLen - maxInputLength;
    reqMaxNewTokens.push_back(maxNewTokens);
    auto const* const seqBegin = givenInputData + reqIdx * maxInputLength;

    auto request = Request(VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming,
        tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, modelIds.endId);
    // copy request
    auto request2 = request;

    auto const expectedOutputData = tr::BufferRange<TokenIdType const>(*testData.expectedOutputIds);
    auto const expectedOutputLengths = testData.expectedOutputLengths;
    auto const endPos = expectedOutputLengths[reqIdx] - 3;
    auto const endIndex = tc::flat_index3(reqIdx, beamWidth - 1, endPos, beamWidth, maxSeqLen);
    auto const endToken = expectedOutputData[endIndex];

    // Set end id to stop early
    request.setEndId(endToken);
    requests.emplace_back(std::move(request));

    // Set stop words to stop early
    request2.setStopWords({{endToken}});
    requests.emplace_back(std::move(request2));

    // Enqueue requests
    auto requestIds = executor.enqueueRequests(requests);

    std::map<IdType, SizeType32> expectedNewTokens;
    expectedNewTokens[requestIds.at(0)] = endPos - inputLen;
    expectedNewTokens[requestIds.at(1)] = endPos - inputLen + 1;

    std::map<IdType, FinishReason> expectedFinishReason;
    expectedFinishReason[requestIds.at(0)] = FinishReason::kEND_ID;
    expectedFinishReason[requestIds.at(1)] = FinishReason::kSTOP_WORDS;

    std::map<IdType, bool> done;
    std::for_each(requestIds.begin(), requestIds.end(), [&done](auto id) { done[id] = false; });
    int iter = 0;
    while (!(std::all_of(done.begin(), done.end(), [](auto x) { return x.second; })) && iter < 5000)
    {
        std::chrono::milliseconds waitTime(mMaxWaitMs);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                auto const reqId = response.getRequestId();
                auto const& result = response.getResult();
                EXPECT_TRUE(result.isFinal);
                done.at(reqId) = result.isFinal;

                // only 1 beam
                auto const& outputIds = result.outputTokenIds.at(0);
                EXPECT_EQ(outputIds.size(), expectedNewTokens.at(reqId)) << "req " << reqId;

                auto const& finishReason = result.finishReasons.at(0);
                EXPECT_EQ(finishReason, expectedFinishReason.at(reqId)) << "req " << reqId;

                auto const& genLogits = result.generationLogits;
                EXPECT_TRUE(genLogits.has_value());

                // Expected shape: (1, numAcceptedDraftToken, vocabSizePadded)
                auto const& generationLogitsShape = genLogits->getShape();
                EXPECT_EQ(generationLogitsShape.size(), 3);
                EXPECT_EQ(generationLogitsShape[0], 1);
                EXPECT_LE(generationLogitsShape[1], maxNewTokens);
                EXPECT_EQ(generationLogitsShape[2], vocabSizePadded);

                auto const genLogitsTensor = detail::toITensor(*genLogits);
                genLogitsTensor->squeeze(0); // only 1 beam

                for (size_t outputIdx = 0; outputIdx < expectedNewTokens.at(reqId); ++outputIdx)
                {
                    // logits argmax should be equal to tokenId
                    auto const genLogitsSlice = tr::ITensor::slice(genLogitsTensor, outputIdx, 1);
                    auto const genLogitsRange = tr::BufferRange<float>(*genLogitsSlice);
                    auto const* maxPos = std::max_element(genLogitsRange.begin(), genLogitsRange.end());
                    auto const maxIdx = std::distance(genLogitsRange.begin(), maxPos);

                    auto const tokenId = outputIds.at(outputIdx);
                    // Observed token mismatch at index 2 after building GPT engine with TRT builder optimization
                    // level 3. The testcase is sensitive to slight variation in kernel computation, so we skip checking
                    // for token id at index 2.
                    if (outputIdx != 2)
                    {
                        EXPECT_EQ(tokenId, maxIdx) << "req " << reqId << " outputIdx " << outputIdx;
                    }
                }
            }
        }
        ++iter;
    }
}

TEST_F(GptExecutorTest, GenerationChangeEndId)
{
    SizeType32 constexpr beamWidth{2};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    auto constexpr streaming = false;

    ExtendedRuntimePerfKnobConfig perfKnobConfig = ExtendedRuntimePerfKnobConfig();
    perfKnobConfig.setEnableContextFMHAFP32Acc(true); // use fmha fp32 acc for better accuracy

    // Create executor config
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setExtendedRuntimePerfKnobConfig(perfKnobConfig);

    // Create executor
    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto const inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};

    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, modelIds.padId);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(*givenInput);

    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    BeamResult beamResult{beamWidth};
    auto const resultsPath
        = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
    beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_CONTEXTFMHAFP32ACC_RESULT_FILE();

    // Just return tokens for check
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = true;

    // Load expected outputs for each beam width value
    auto testData = TestData::loadTestData(beamResult, *givenInput, beamWidth, manager, outConfig, modelIds);
    auto const maxSeqLen = testData.maxSeqLen;

    // Load expected outputs and inputs
    std::vector<Request> requests;
    std::vector<SizeType32> reqMaxNewTokens;

    // Only use the first request to test
    auto constexpr reqIdx = 0;
    SizeType32 inputLen = givenInputLengths.at(reqIdx);
    auto maxNewTokens = maxSeqLen - maxInputLength;
    reqMaxNewTokens.push_back(maxNewTokens);
    auto const* const seqBegin = givenInputData + reqIdx * maxInputLength;

    // Use customized `EndId` to enqueue once
    auto request = Request(VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming,
        tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, modelIds.endId);

    TokenIdType customizedEndId = *(seqBegin + 1); // Use a token appeared in ground-truth
    request.setEndId(customizedEndId);
    requests.emplace_back(std::move(request));

    auto requestIds = executor.enqueueRequests(requests);
    std::chrono::milliseconds waitTime(mMaxWaitMs);
    auto responses = executor.awaitResponses(waitTime);
    if (responses.at(0).hasError())
    {
        FAIL();
    }
    requests.clear();

    // Change back to default `EndId` to enqueue again, and check the output
    request = Request(VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming,
        tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, modelIds.endId);

    auto const expectedOutputData = tr::BufferRange<TokenIdType const>(*testData.expectedOutputIds);
    auto const expectedOutputLengths = testData.expectedOutputLengths;
    auto const endPos = expectedOutputLengths[reqIdx];
    auto const endIndex = tc::flat_index3(reqIdx, beamWidth, endPos, beamWidth, maxSeqLen);
    auto const endToken = expectedOutputData[endIndex];

    request.setEndId(endToken);
    requests.emplace_back(std::move(request));
    requestIds = executor.enqueueRequests(requests);
    auto const requestId = requestIds.at(0);

    std::map<IdType, SizeType32> expectedNewTokens;
    expectedNewTokens[requestId] = endPos - inputLen;

    std::map<IdType, FinishReason> expectedFinishReason;
    expectedFinishReason[requestId] = FinishReason::kLENGTH;

    std::map<IdType, bool> done;
    std::for_each(requestIds.begin(), requestIds.end(), [&done](auto id) { done[id] = false; });
    int iter = 0;
    while (!(std::all_of(done.begin(), done.end(), [](auto x) { return x.second; })) && iter < 5000)
    {
        std::chrono::milliseconds waitTime(mMaxWaitMs);
        auto responses = executor.awaitResponses(waitTime);
        auto& response = responses.at(0);
        if (response.hasError())
        {
            FAIL();
        }
        else
        {
            auto const reqId = response.getRequestId();
            auto const& result = response.getResult();
            EXPECT_TRUE(result.isFinal);
            done.at(reqId) = result.isFinal;

            bool anyMismatch = false;
            for (int i = 0; i < result.outputTokenIds.size(); ++i)
            {
                auto const& outputIds = result.outputTokenIds.at(i);
                EXPECT_EQ(outputIds.size(), expectedNewTokens.at(reqId)) << "req " << reqId;
                anyMismatch |= outputIds.size() != expectedNewTokens.at(reqId);

                auto const& finishReason = result.finishReasons.at(i);
                EXPECT_EQ(finishReason, expectedFinishReason.at(reqId)) << "req " << reqId;
                anyMismatch |= finishReason != expectedFinishReason.at(reqId);

                if (anyMismatch)
                {
                    break;
                }

                for (int j = 0; j < outputIds.size(); ++j)
                {
                    auto const resultToken = outputIds[j];
                    auto const groundTruthToken = expectedOutputData[maxSeqLen * i + inputLen + j];
                    EXPECT_EQ(resultToken, groundTruthToken);
                    anyMismatch |= resultToken != groundTruthToken;
                }
            }
            EXPECT_FALSE(anyMismatch);
        }
        ++iter;
    }
}

// stream, excludeInputFromOutput, beamWidth
using ParamType = std::tuple<bool, bool, int>;
// useOrchestratorMode, beamWidth, modelName
using ParamCancelReqType = std::tuple<bool, int, std::string>;
// modelName
using LeaderApiUsageType = std::tuple<std::string>;
// iterStatsMaxIterations, useOrchestratorMode
using ParamStatsType = std::tuple<int, bool>;
// streaming, beamWidth, computeLogProbs, excludeInputInOutput, returnContextLogits, returnGenerationLogits, modelName,
// useOrchestratorMode, returnAllGeneratedTokens, numReturnSequences
using AllParamsType = std::tuple<bool, int, bool, bool, bool, bool, std::string, bool, bool, int>;
// modelName, batched, replicated
using LogitsProcParamsType = std::tuple<std::string, bool, bool>;
// modelName
using GuidedDecodingParamsType = std::tuple<std::string>;
// modelName, useOrchestratorMode, beamWidth
using TimeoutTestParamsType = std::tuple<std::string, bool, int>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const streaming = std::get<0>(info.param);
    auto const excludeInputFromOutput = std::get<1>(info.param);
    auto const beamWidth = std::get<2>(info.param);
    std::string name = "ExecutorTest";
    if (streaming)
    {
        name += "Streaming";
    }
    if (excludeInputFromOutput)
    {
        name += "ExclInput";
    }
    name.append("BW" + std::to_string(beamWidth));
    return name;
}

std::string generateTestNameCancelReq(testing::TestParamInfo<ParamCancelReqType> const& info)
{
    auto const& useOrchestratorMode = std::get<0>(info.param);
    auto const beamWidth = std::get<1>(info.param);
    auto const modelName = std::get<2>(info.param);
    std::string name = "ExecutorTest";
    name.append("BW" + std::to_string(beamWidth));
    name.append("_" + modelName + "_");

    if (useOrchestratorMode)
    {
        name.append("OrchMode");
    }
    else
    {
        name.append("LeaderMode");
    }
    return name;
}

std::string generateTestNameLeaderApiUsage(testing::TestParamInfo<LeaderApiUsageType> const& info)
{
    auto const modelName = std::get<0>(info.param);
    std::string name = "ExecutorTest";
    name.append("_" + modelName);
    return name;
}

std::string generateTestNameLogitsProc(testing::TestParamInfo<LogitsProcParamsType> const& info)
{
    auto const modelName = std::get<0>(info.param);
    bool const batched = std::get<1>(info.param);
    bool const replicated = std::get<2>(info.param);
    std::string name = "ExecutorTest";
    name.append("_" + modelName);
    if (batched)
    {
        name.append("_Batched");
    }
    if (replicated)
    {
        name.append("_Replicated");
    }
    return name;
}

std::string generateTestNameGuidedDecoding(testing::TestParamInfo<GuidedDecodingParamsType> const& info)
{
    auto const modelName = std::get<0>(info.param);
    std::string name = "ExecutorTest";
    name.append("_" + modelName);
    return name;
}

std::string generateTestNameTimeoutTest(testing::TestParamInfo<TimeoutTestParamsType> const& info)
{
    auto const modelName = std::get<0>(info.param);
    auto const& useOrchestratorMode = std::get<1>(info.param);
    auto const beamWidth = std::get<2>(info.param);

    std::string name = "ExecutorTest";
    name.append("_" + modelName);

    if (useOrchestratorMode)
    {
        name.append("_OrchMode");
    }
    else
    {
        name.append("_LeaderMode");
    }
    name.append("_BW" + std::to_string(beamWidth));
    return name;
}

std::string generateTestNameStats(testing::TestParamInfo<ParamStatsType> const& info)
{
    int iterStatsMaxIterations = std::get<0>(info.param);
    auto const& useOrchestratorMode = std::get<1>(info.param);
    std::string name = "ExecutorTest_";
    name.append(std::to_string(iterStatsMaxIterations) + "_");
    if (useOrchestratorMode)
    {
        name.append("OrchMode");
    }
    else
    {
        name.append("LeaderMode");
    }
    return name;
}

std::string generateTestNameAllParams(testing::TestParamInfo<AllParamsType> const& info)
{
    auto const streaming = std::get<0>(info.param);
    auto const& beamWidth = std::get<1>(info.param);
    auto const& computeLogProbs = std::get<2>(info.param);
    auto const& excludeInputInOutput = std::get<3>(info.param);
    auto const& returnContextLogits = std::get<4>(info.param);
    auto const& returnGenerationLogits = std::get<5>(info.param);
    auto const modelName = std::get<6>(info.param);
    auto const& useOrchestratorMode = std::get<7>(info.param);
    auto const& returnAllGeneratedTokens = std::get<8>(info.param);
    auto const& numReturnSequences = std::get<9>(info.param);

    std::string name = "ExecutorTest_";

    if (streaming)
    {
        name += "Streaming";
    }

    name.append("_BW" + std::to_string(beamWidth));
    name.append("Nseq" + std::to_string(numReturnSequences));

    if (computeLogProbs)
    {
        name.append("LogProbs");
    }
    if (excludeInputInOutput)
    {
        name.append("ExcludeInput");
    }
    if (returnContextLogits)
    {
        name.append("ContextLogits");
    }
    if (returnGenerationLogits)
    {
        name.append("GenerationLogits");
    }
    name.append("_" + modelName + "_");
    if (useOrchestratorMode)
    {
        name.append("OrchMode");
    }
    else
    {
        name.append("LeaderMode");
    }

    if (returnAllGeneratedTokens)
    {
        name.append("returnAllGeneratedTokens");
    }
    return name;
}

class ParamTest : public GptExecutorTest, public ::testing::WithParamInterface<ParamType>
{
};

class ParamStatsTest : public GptExecutorTest, public ::testing::WithParamInterface<ParamStatsType>
{
};

class AllParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<AllParamsType>
{
};

class ParamCancelReqTest : public GptExecutorTest, public ::testing::WithParamInterface<ParamCancelReqType>
{
};

class LeaderApiUsageTest : public GptExecutorTest, public ::testing::WithParamInterface<LeaderApiUsageType>
{
};

class LogitsProcParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<LogitsProcParamsType>
{
};

class GuidedDecodingParamsTest : public GptExecutorTest, public ::testing::WithParamInterface<GuidedDecodingParamsType>
{
};

class TimeoutTest : public GptExecutorTest, public ::testing::WithParamInterface<TimeoutTestParamsType>
{
};

TEST_F(GptExecutorTest, GetLatestStats)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    auto requestId = executor.enqueueRequest(std::move(request));

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                done = response.getResult().isFinal;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Expect 6 non-empty iterations
    auto stats = executor.getLatestIterationStats();
    EXPECT_EQ(stats.size(), 6);
    uint64_t currentIter = 0;
    for (auto const& stat : stats)
    {
        EXPECT_EQ(stat.timestamp.size(), 26);
        EXPECT_EQ(stat.iter, currentIter);
        if (currentIter != 5)
        {
            EXPECT_EQ(stat.numActiveRequests, 1);
        }
        else
        {
            // For the last iteration the number of active requests
            // should be zero.
            EXPECT_EQ(stat.numActiveRequests, 0);
        }
        EXPECT_EQ(stat.maxNumActiveRequests, 64);
        // Very loose check to make sure the memory stats are valid
        EXPECT_GT(stat.gpuMemUsage, 16);
        EXPECT_GT(stat.cpuMemUsage, 16);
        EXPECT_GT(stat.pinnedMemUsage, 16);

        // Stats for KV cache
        EXPECT_TRUE(stat.kvCacheStats.has_value());
        KvCacheStats const& kvStats = stat.kvCacheStats.value();
        EXPECT_GT(kvStats.maxNumBlocks, 0);
        EXPECT_GT(kvStats.freeNumBlocks, 0);
        EXPECT_EQ(kvStats.usedNumBlocks, currentIter == maxNewTokens ? 0 : 1);
        EXPECT_GT(kvStats.tokensPerBlock, 0);
        EXPECT_GT(kvStats.allocTotalBlocks, 0);
        EXPECT_GT(kvStats.allocNewBlocks, 0);
        EXPECT_GE(kvStats.reusedBlocks, 0);
        EXPECT_GE(kvStats.missedBlocks, 0);
        EXPECT_GE(kvStats.cacheHitRate, 0);

        // Stats for inflight batching
        EXPECT_TRUE(stat.inflightBatchingStats.has_value() && !stat.staticBatchingStats.has_value());
        InflightBatchingStats const& modelStats = stat.inflightBatchingStats.value();
        EXPECT_EQ(modelStats.numScheduledRequests, currentIter == maxNewTokens ? 0 : 1);
        EXPECT_EQ(modelStats.numContextRequests, currentIter == 0 ? 1 : 0);
        EXPECT_EQ(modelStats.numGenRequests, currentIter == 0 || currentIter == maxNewTokens ? 0 : 1);
        EXPECT_EQ(modelStats.numPausedRequests, 0);
        EXPECT_EQ(modelStats.numCtxTokens, currentIter == 0 ? inputTokens.size() : 0);
        EXPECT_EQ(modelStats.microBatchId, 0);
        EXPECT_NEAR(
            modelStats.avgNumDecodedTokensPerIter, currentIter == 0 || currentIter == maxNewTokens ? 0.f : 1.f, 1e-9f);

        auto jsonStr = JsonSerialization::toJsonStr(stat);
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"iter\":" + std::to_string(currentIter)));
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"staticBatchingStats\":null"));
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"numCtxTokens\":" + std::to_string(modelStats.numCtxTokens)));
        EXPECT_THAT(jsonStr, testing::HasSubstr("\"numGenRequests\":" + std::to_string(modelStats.numGenRequests)));

        ++currentIter;
    }
}

TEST_F(GptExecutorTest, GetLatestStatsWithMultipleRequests)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the requests
    SizeType32 const numRequests = 2;
    std::vector<SizeType32> maxNewTokens{3, 5};
    std::vector<VecTokens> inputTokens{{1, 2, 3, 4}, {5, 6, 7}};
    std::vector<IdType> reqIds;
    for (SizeType32 ireq = 0; ireq < numRequests; ++ireq)
    {
        auto request = Request(inputTokens[ireq], maxNewTokens[ireq], streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
        auto requestId = executor.enqueueRequest(std::move(request));
        reqIds.emplace_back(requestId);
        // sleep for 10 ms before sending the next request
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    for (SizeType32 ireq = 0; ireq < numRequests; ++ireq)
    {
        auto requestId = reqIds[ireq];
        bool done = false;
        int iter = 0;
        while (!done && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {
                    FAIL();
                }
                else
                {
                    done = response.getResult().isFinal;
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, mMaxWaitMs);
    }

    // NOTES:
    // Expect at least max(maxNewTokens) i.e. 5 non-empty iterations
    // 4th iteration should have numCompletedRequests to be 1.
    // Depending on the timing, first iteration will either have:
    //      2 active requests
    //      or
    //      1 active requests and 1 queued requests
    auto stats = executor.getLatestIterationStats();
    EXPECT_GT(stats.size(), 0); // make sure we have at least 1 stat before the accessing 0-th element
    if (stats[0].numActiveRequests == 2)
    {
        // we cannot reliably check queue latency since both started in the same iteration
        // there should be exactly 5 non-empty iterations
        EXPECT_EQ(stats.size(), 5);
        // only check numCompletedRequests in 4th iteration
        EXPECT_EQ(stats[3].numCompletedRequests, 1);
        // 1st iteration shall record all 2 requests queueing time;
        EXPECT_EQ(stats[0].numNewActiveRequests, 2);
        // all rest iterations shall not return any queueing time;
        for (int i = 1; i < stats.size(); ++i)
        {
            EXPECT_EQ(stats[i].numNewActiveRequests, 0);
        }
    }
    else
    {
        // there should be more than 5 non-empty iterations since 2nd request started after 1st iteration
        EXPECT_GT(stats.size(), 5);
        // 1st request's completion is at 4th iteration
        EXPECT_EQ(stats[3].numCompletedRequests, 1);
        // 1st iteration record 1 request's queueing time;
        EXPECT_EQ(stats[0].numNewActiveRequests, 1);
        // the iteration where 2nd request became active, queue latency must be > 0
        uint64_t currentIter = 0;
        for (auto const& stat : stats)
        {
            // To check when 2nd request becomes active, we need to think about 2 cases:
            //  - it overlaps with first request
            //      => only check queue time in this case
            //  - it doesn't overlap with the first request (e.g. 1st request ended too fast)
            //      => little to no queue time, cannot check reliably
            //  so we only check for queue time when numActiveRequests > 1 i.e. overlap happened after first iteration
            if (stat.numActiveRequests > 1)
            {
                EXPECT_GT(currentIter, 0); // it must be after 1st iteration
                EXPECT_GT(stat.newActiveRequestsQueueLatencyMS, 0);
                // 2nd request record queueing time in this iteration
                EXPECT_EQ(stat.numNewActiveRequests, 1);
                break;
            }
            ++currentIter;
        }
    }
}

TEST_F(GptExecutorTest, GetLatestRequestStats)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    executorConfig.setEnableChunkedContext(true);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the requests
    std::vector<std::pair<SizeType32, VecTokens>> requestParams = {
        // {maxNewTokens, inputTokens}
        {5, {1, 2, 3, 4}}, {4, {1, 1, 2, 3, 5}}, {1, {1}},
        {8, VecTokens(383, 1)} // Long enough to be chunked into multiple iterations
    };
    std::vector<Request> requests;
    for (auto requestParam : requestParams)
    {
        requests.emplace_back(requestParam.second, requestParam.first, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    }
    auto requestIdsVec = executor.enqueueRequests(std::move(requests));
    std::map<IdType, SizeType32> requestIdToIndex;
    std::set<IdType> activeRequests;
    for (SizeType32 i = 0; i < requestIdsVec.size(); ++i)
    {
        auto requestId = requestIdsVec[i];
        activeRequests.insert(requestId);
        requestIdToIndex[requestId] = i;
    }

    int iter = 0;
    while (!activeRequests.empty() && iter < mMaxWaitMs)
    {
        for (auto i = activeRequests.begin(); i != activeRequests.end();)
        {
            auto requestId = *i;
            bool thisDone = false;
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
                else
                {
                    thisDone = response.getResult().isFinal;
                }
            }
            if (thisDone)
            {
                // Erase completed request and move to the next one
                i = activeRequests.erase(i);
            }
            else
            {
                ++i;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Expect 5 non-empty iterations
    // Note: The 6th iteration with the last finished request will be reported
    //       but might be unavailable when getLatestRequestStats is called since
    //       it could be updated after the final response has been sent.
    auto stats = executor.getLatestRequestStats();
    EXPECT_GE(stats.size(), 5);
    SizeType32 currentIter = 0;
    auto invalidStart = std::numeric_limits<SizeType32>::max();
    std::vector<SizeType32> genStart(requestParams.size(), invalidStart); // The iteration index when generation started
    std::set<IdType> completedRequests;
    for (auto stat = stats.begin(); stat != stats.begin() + 5; ++stat)
    {
        auto jsonStrIter = JsonSerialization::toJsonStr(*stat);
        EXPECT_EQ(stat->iter, currentIter);
        EXPECT_THAT(jsonStrIter, testing::HasSubstr("\"iter\":" + std::to_string(currentIter)));
        EXPECT_EQ(stat->requestStats.size() + completedRequests.size(), requestParams.size());
        for (auto rStat : stat->requestStats)
        {
            auto jsonStr = JsonSerialization::toJsonStr(rStat);
            // Only a few requests here so all of them should be scheduled. A separate test
            // GetLatestRequestStatsScheduling will target the scheduling stats.
            if (rStat.stage != RequestStage::kGENERATION_COMPLETE)
            {
                EXPECT_TRUE(rStat.scheduled);
                EXPECT_THAT(jsonStr, testing::HasSubstr("\"scheduled\":true"));
            }
            EXPECT_TRUE(!rStat.paused);
            EXPECT_THAT(jsonStr, testing::HasSubstr("\"paused\":false"));
            EXPECT_TRUE(requestIdToIndex.count(rStat.id));
            EXPECT_THAT(jsonStr, testing::HasSubstr("\"id\":" + std::to_string(rStat.id)));
            auto requestIndex = requestIdToIndex[rStat.id];
            auto contextSize = requestParams[requestIndex].second.size();
            if (rStat.contextPrefillPosition == contextSize) // Check generation phase
            {
                bool firstIteration{false};
                // Context phase is done
                EXPECT_TRUE(rStat.stage == RequestStage::kGENERATION_IN_PROGRESS
                    || rStat.stage == RequestStage::kGENERATION_COMPLETE);
                EXPECT_THAT(jsonStr, testing::HasSubstr("\"stage\":\"GENERATION"));
                if (genStart[requestIndex] == invalidStart)
                {
                    // Just started generation
                    genStart[requestIndex] = currentIter;
                    firstIteration = true;
                }

                // One token per iteration
                EXPECT_TRUE(currentIter - genStart[requestIndex] == rStat.numGeneratedTokens);
                EXPECT_NEAR(rStat.avgNumDecodedTokensPerIter, firstIteration ? 0.f : 1.0f, 1e-9);
                if (rStat.stage == RequestStage::kGENERATION_COMPLETE)
                {
                    EXPECT_TRUE(requestParams[requestIndex].first >= rStat.numGeneratedTokens);
                    completedRequests.insert(requestIndex);
                }
                else
                {
                    EXPECT_FALSE(completedRequests.count(requestIndex));
                }
            }
            else if (rStat.contextPrefillPosition < contextSize) // Check context phase
            {
                // Must be chunked
                SizeType32 const maxChunkSize = 128;
                EXPECT_TRUE(rStat.contextPrefillPosition % maxChunkSize == 0);
                // Context phase is on-going
                EXPECT_TRUE(rStat.stage == RequestStage::kCONTEXT_IN_PROGRESS);
                // No tokens are generated
                EXPECT_TRUE(0 == rStat.numGeneratedTokens);
            }
            else
            {
                FAIL() << "Out-of-boundary contextPrefillPosition in stats: " << rStat.contextPrefillPosition
                       << " out of " << contextSize;
            }
            // Sanity check that disaggregated serving stats is not set in typical use case
            EXPECT_FALSE(rStat.disServingStats.has_value());
        }
        ++currentIter;
    }
    // We should have visited all requests.
    // Take into consideration the last request has not been reported
    EXPECT_EQ(completedRequests.size() + 1, requestParams.size());
}

TEST_F(GptExecutorTest, GetLatestRequestStatsScheduling)
{
    // Specifically test the case where there are too many requests to be scheduled for a iteration
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    executorConfig.setEnableChunkedContext(true);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create 100 requests. Note the max batch size for this model is 64 so some requests won't be scheduled right away.
    std::vector<std::pair<SizeType32, VecTokens>> requestParams(100, {5, {1, 2, 3, 4}});
    std::vector<Request> requests;
    requests.reserve(requestParams.size());
    for (auto requestParam : requestParams)
    {
        requests.emplace_back(requestParam.second, requestParam.first, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    }
    auto requestIdsVec = executor.enqueueRequests(std::move(requests));
    std::map<IdType, SizeType32> requestIdToIndex;
    std::set<IdType> activeRequests;
    for (SizeType32 i = 0; i < requestIdsVec.size(); ++i)
    {
        auto requestId = requestIdsVec[i];
        activeRequests.insert(requestId);
        requestIdToIndex[requestId] = i;
    }

    int iter = 0;
    while (!activeRequests.empty() && iter < mMaxWaitMs)
    {
        for (auto i = activeRequests.begin(); i != activeRequests.end();)
        {
            auto requestId = *i;
            bool thisDone = false;
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
                else
                {
                    thisDone = response.getResult().isFinal;
                }
            }
            if (thisDone)
            {
                // Erase completed request and move to the next one
                i = activeRequests.erase(i);
            }
            else
            {
                ++i;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    auto stats = executor.getLatestRequestStats();
    SizeType32 numFinished = 0;
    SizeType32 const maxActiveSize = 64; // Decided by the model

    // The 6th iteration request stat may or may not be available when getLatestRequestStats
    // is called. When there are no other active or inTransmission requests, there will be
    // another request stats to properly reset all the statistics to zero.
    for (auto stat = stats.begin(); stat != stats.begin() + 5; ++stat)
    {
        SizeType32 numReqs = 0;
        SizeType32 numReqsActive = 0;
        SizeType32 numReqsQueued = 0;
        SizeType32 numReqsJustDone = 0;
        for (auto rStat : stat->requestStats)
        {
            ++numReqs;
            numReqsActive += rStat.scheduled ? 1 : 0;
            numReqsQueued += rStat.stage == RequestStage::kQUEUED ? 1 : 0;
            numReqsJustDone += rStat.stage == RequestStage::kGENERATION_COMPLETE ? 1 : 0;
        }
        EXPECT_EQ(numReqs, numReqsActive + numReqsQueued + numReqsJustDone);
        EXPECT_EQ(numReqs + numFinished, requestParams.size()); // Should report all unfinished requests
        EXPECT_TRUE(numReqsActive <= maxActiveSize); // Not all requests are active due to max active size limit.
        numFinished += numReqsJustDone;
    }
}

TEST_F(GptExecutorTest, GetRequestStatsMultipleRequests)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto sendRequestWaitForResponseFn = [&]()
    {
        Request request({1, 2, 3}, 5);
        auto requestId = executor.enqueueRequest(request);
        bool isFinalResponse = false;
        while (!isFinalResponse)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto response : responses)
            {
                if (response.getResult().isFinal)
                {
                    isFinalResponse = true;
                    break;
                }
            }
        }
        return requestId;
    };

    std::unordered_map<IdType, size_t> requestIdToGenerationComplete;
    auto updateStats = [&]()
    {
        auto stats = executor.getLatestRequestStats();
        for (auto& stat : stats)
        {
            for (auto const& request : stat.requestStats)
            {
                // only check and aggregate results when request is completed
                if (request.stage == RequestStage::kGENERATION_COMPLETE)
                {
                    requestIdToGenerationComplete[request.id] += 1;
                }
            }
        }
    };

    auto requestId = sendRequestWaitForResponseFn();
    requestIdToGenerationComplete[requestId] = 0;
    updateStats();

    requestId = sendRequestWaitForResponseFn();
    requestIdToGenerationComplete[requestId] = 0;
    updateStats();

    for (auto [key, value] : requestIdToGenerationComplete)
    {
        EXPECT_EQ(value, 1);
    }
}

TEST_F(GptExecutorTest, BatchSizeTuning)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    executorConfig.setEnableChunkedContext(true);

    DynamicBatchConfig dynamicBatchConfig(true, false, 1); // Set window size to 1
    SchedulerConfig schedulerConfig(CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT, std::nullopt, dynamicBatchConfig);
    executorConfig.setSchedulerConfig(schedulerConfig);

    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    std::vector<SizeType32> tunerRecommendedBatchSizes;

    for (size_t i = 0; i <= 8; ++i)
    {
        auto inputLength = 1 << i; // Note that for this model max input len is 383
        Request request(
            VecTokens(inputLength, 2), 5, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
        auto requestId = executor.enqueueRequest(std::move(request));
        // Wait for current request to finish
        while (true)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            bool done = false;
            if (responses.size() != 0)
            {
                EXPECT_TRUE(responses.size() == 1);
                auto response = responses[0];
                EXPECT_FALSE(response.hasError());
                if (response.getResult().isFinal)
                {
                    break;
                }
            }
        }
        auto reqStats = executor.getLatestIterationStats();
        EXPECT_TRUE(reqStats.size() > 0);
        auto lastStat = reqStats.back();
        tunerRecommendedBatchSizes.push_back(lastStat.maxBatchSizeTunerRecommended);
    }

    EXPECT_TRUE(tunerRecommendedBatchSizes.size() > 0);
    // It's supposed to be decreasing when input length increases
    EXPECT_TRUE(*tunerRecommendedBatchSizes.begin() > *tunerRecommendedBatchSizes.rbegin());
}

TEST_F(GptExecutorTest, GetLatestDebugTensors)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    SizeType32 maxNewTokens = 5;

    tensorrt_llm::executor::DebugConfig debugConfig;
    debugConfig.setDebugTensorNames({{"sequence_length"}});
    debugConfig.setDebugTensorsMaxIterations(maxNewTokens);

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setDebugConfig(debugConfig);

    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    auto requestId = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL();
            }
            else
            {
                done = response.getResult().isFinal;
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    auto stream = std::make_shared<tr::CudaStream>();

    // Expect 5 non-empty iterations
    auto debugTensors = executor.getLatestDebugTensors();
    EXPECT_EQ(debugTensors.size(), 5);
    uint64_t currentIter = 0;
    for (auto const& debugIteration : debugTensors)
    {
        EXPECT_EQ(debugIteration.iter, currentIter);
        EXPECT_EQ(debugIteration.debugTensors.size(), 2);

        {
            auto it = debugIteration.debugTensors.find("request_ids");
            EXPECT_NE(it, debugIteration.debugTensors.end());
            auto const& tensor = it->second;
            auto const& shape = tensor.getShape();
            EXPECT_EQ(shape.size(), 1);
            EXPECT_EQ(shape[0], 1);
            EXPECT_EQ(tensor.getSize(), 1);
            auto const* dataPtr = static_cast<SizeType32 const*>(tensor.getData());
            EXPECT_EQ(dataPtr[0], 1) << "currentIter " << currentIter;
        }
        {
            auto it = debugIteration.debugTensors.find("sequence_length");
            EXPECT_NE(it, debugIteration.debugTensors.end());
            auto const& tensor = it->second;
            auto const& shape = tensor.getShape();
            EXPECT_EQ(shape.size(), 1);
            EXPECT_EQ(tensor.getSize(), 1);
            auto tensorHost = tensor.copyToCpu(stream);
            auto const* dataPtr = static_cast<SizeType32 const*>(tensorHost.getData());
            EXPECT_EQ(dataPtr[0], inputTokens.size() + currentIter);
        }

        ++currentIter;
    }
}

TEST_P(ParamTest, SingleRequestDemo)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(request);

    // Get the new tokens
    VecTokens tokens;
    SizeType32 numResponses{0};
    bool done = false;
    int iter = 0;
    std::chrono::milliseconds waitTime(1);
    while (!done && iter < mMaxWaitMs)
    {
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            ++numResponses;
            if (response.hasError())
            {
                // This request failed for some reason, get error msg
                std::string errStr
                    = "Request id " + std::to_string(requestId) + " failed with err " + response.getErrorMsg();
                FAIL();
            }

            auto result = response.getResult();
            done = result.isFinal;
            auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
            auto const expectedSize = streaming ? (beamWidth > 1 ? numResponses : 1)
                                                : (maxNewTokens + (excludeInputFromOutput ? 0 : inputTokens.size()));
            EXPECT_EQ(newTokens.size(), expectedSize);

            if (streaming && beamWidth > 1)
            {
                // replace tokens
                tokens = newTokens;
            }
            else
            {
                // Append tokens
                tokens.insert(tokens.end(), newTokens.begin(), newTokens.end());
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(numResponses, streaming ? maxNewTokens : 1);
    EXPECT_EQ(
        tokens.size(), streaming ? maxNewTokens : (excludeInputFromOutput ? 0 : inputTokens.size()) + maxNewTokens);

    // Expect awaitResponse to return error message because the request is already terminated (isFinal = True)
    auto response = executor.awaitResponses(requestId, waitTime).at(0);
    EXPECT_TRUE(response.hasError());
    std::string err
        = "ReqId " + std::to_string(response.getRequestId()) + " has already been processed and was terminated.";
    EXPECT_EQ(response.getErrorMsg(), err);
}

TEST_P(ParamTest, MultipleRequestDemo)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 numRequests = 20;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    SizeType32 endId = -1;
    // Enqueue the requests
    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    std::unordered_map<IdType, SizeType32> expectedNumResponses;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
        auto reqId = executor.enqueueRequest(std::move(request));
        tokens[reqId] = {};
        expectedNumTokens[reqId] = ((streaming || excludeInputFromOutput) ? 0 : promptLen) + maxNewTokens;
        expectedNumResponses[reqId] = streaming ? maxNewTokens : 1;
    }

    // Get the new tokens for each requests
    int32_t numFinished = 0;
    int iter = 0;
    std::unordered_map<IdType, SizeType32> numResponses;
    while (numFinished < numRequests && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            auto reqId = response.getRequestId();
            ++numResponses[reqId];
            if (!response.hasError())
            {
                auto result = response.getResult();
                numFinished += result.isFinal;
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                auto const expectedSize
                    = streaming ? (beamWidth > 1 ? numResponses[reqId] : 1) : expectedNumTokens[reqId];
                EXPECT_EQ(newTokens.size(), expectedSize);

                auto& reqTokens = tokens.at(response.getRequestId());
                if (streaming && beamWidth > 1)
                {
                    reqTokens = newTokens;
                }
                else
                {
                    reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                }

                for (SizeType32 b = 0; b < beamWidth; ++b)
                {
                    EXPECT_EQ(result.finishReasons.at(b),
                        result.isFinal ? FinishReason::kLENGTH : FinishReason::kNOT_FINISHED);
                }
            }
            else
            {
                // Allow response with error only if awaitResponse processed a terminated request id
                std::string err = "ReqId " + std::to_string(response.getRequestId())
                    + " has already been processed and was terminated.";
                EXPECT_EQ(response.getErrorMsg(), err);
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Check that number of tokens matches expectations
    for (auto const& [reqId, numTokens] : expectedNumTokens)
    {
        EXPECT_EQ(expectedNumResponses[reqId], numResponses[reqId]) << "reqId " << reqId;
        EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
    }
}

TEST_P(ParamStatsTest, MultipleRequestStats)
{
    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 numRequests = 100;
    auto iterStatsMaxIterations = std::get<0>(GetParam());
    bool useOrchestratorMode = std::get<1>(GetParam());

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setIterStatsMaxIterations(iterStatsMaxIterations);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";

    std::optional<OrchestratorConfig> orchestratorConfig = std::nullopt;
    if (useOrchestratorMode)
    {
        orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
    }
    auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
        useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt, std::nullopt,
        orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    SizeType32 endId = -1;
    // Enqueue the requests
    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
        auto reqId = executor.enqueueRequest(std::move(request));
        tokens[reqId] = {};
        expectedNumTokens[reqId] = (streaming ? 0 : (excludeInputFromOutput ? 0 : promptLen)) + maxNewTokens;
    }

    std::atomic<bool> statsThreadDone = false;
    std::atomic<int32_t> numFinished = 0;
    std::deque<IterationStats> iterStatsReceived;
    // Spawn a thread that continuously get stats
    auto statsThread = std::thread(
        [&executor, &numFinished, numRequests, &iterStatsReceived, &statsThreadDone]()
        {
            while (numFinished < numRequests)
            {
                auto reqStats = executor.getLatestIterationStats();
                iterStatsReceived.insert(iterStatsReceived.end(), std::make_move_iterator(reqStats.begin()),
                    std::make_move_iterator(reqStats.end()));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            statsThreadDone = true;
        });

    // Get the new tokens for each requests
    int iter = 0;
    SizeType32 numResponses = 0;
    while (numFinished < numRequests && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            numResponses++;
            if (!response.hasError())
            {
                auto result = response.getResult();
                numFinished += result.isFinal;
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                auto& reqTokens = tokens.at(response.getRequestId());
                reqTokens.insert(reqTokens.end(), std::make_move_iterator(newTokens.begin()),
                    std::make_move_iterator(newTokens.end()));
            }
            else
            {
                // Allow response with error only if awaitResponse processed a terminated request id
                std::string err = "ReqId " + std::to_string(response.getRequestId())
                    + " has already been processed and was terminated.";
                EXPECT_EQ(response.getErrorMsg(), err);
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Check that number of tokens matches expectations
    for (auto const& [reqId, numTokens] : expectedNumTokens)
    {
        EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
    }

    // Wait for stats thread to be done, fail otherwise
    iter = 0;
    while (!statsThreadDone && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
        iter++;
    }
    ASSERT_TRUE(statsThreadDone);
    if (iterStatsMaxIterations > 0)
    {
        ASSERT_GT(iterStatsReceived.size(), 1);

        for (auto stats : iterStatsReceived)
        {
            EXPECT_GT(stats.numActiveRequests, 0);
            TLLM_LOG_INFO("%d %d", stats.iter, stats.numActiveRequests);

            EXPECT_TRUE(stats.inflightBatchingStats.has_value());
            if (stats.inflightBatchingStats.has_value())
            {
                EXPECT_GT(stats.inflightBatchingStats.value().numScheduledRequests, 0);
            }
        }
    }

    statsThread.join();
}

TEST_P(ParamTest, MultipleRequestBatchResponses)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 constexpr numRequests{20};

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 constexpr maxPromptLen{20};
    SizeType32 constexpr maxMaxNewTokens{20};

    SizeType32 endId = -1;
    // Enqueue the requests
    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    std::vector<IdType> requestIds;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
        auto reqId = executor.enqueueRequest(std::move(request));
        requestIds.push_back(reqId);
        tokens[reqId] = {};
        expectedNumTokens[reqId] = (streaming ? 0 : (excludeInputFromOutput ? 0 : promptLen)) + maxNewTokens;
    }

    // Get the new tokens for each requests
    int32_t numFinished = 0;
    int iter = 0;
    SizeType32 numResponses = 0;
    std::chrono::milliseconds waitTime(1);
    while (numFinished < numRequests && iter < mMaxWaitMs)
    {
        auto idResponses = executor.awaitResponses(requestIds, waitTime);
        for (unsigned i = 0; i < requestIds.size(); ++i)
        {
            auto& responses = idResponses[i];
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                    auto& reqTokens = tokens.at(response.getRequestId());
                    if (streaming && beamWidth > 1)
                    {
                        reqTokens = newTokens;
                    }
                    else
                    {
                        reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                    }
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    // Rerun awaitResponses again and we expect to only see terminated request id error.
    auto idResponses = executor.awaitResponses(requestIds, waitTime);
    for (auto const& responses : idResponses)
    {
        for (auto& response : responses)
        {
            EXPECT_TRUE(response.hasError());
            std::string err = "ReqId " + std::to_string(response.getRequestId())
                + " has already been processed and was terminated.";
            EXPECT_EQ(response.getErrorMsg(), err);
        }
    }

    // Check that number of tokens matches expectations
    for (auto const& [reqId, numTokens] : expectedNumTokens)
    {
        EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
    }
}

TEST_P(ParamTest, GetNumResponsesReadyTest)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 maxNumRequests = 50;
    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    SizeType32 numRequests = rand() % maxNumRequests + 1;
    SizeType32 numExpectedResponses = 0;
    std::map<IdType, SizeType32> reqNumExpectedResponses;
    std::vector<IdType> ids;
    for (SizeType32 req = 0; req < numRequests; ++req)
    {
        SizeType32 promptLen = rand() % maxPromptLen + 1;
        SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

        auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
            tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
        auto id = executor.enqueueRequest(std::move(request));
        ids.emplace_back(id);
        reqNumExpectedResponses[id] = streaming ? maxNewTokens : 1;
        numExpectedResponses += reqNumExpectedResponses.at(id);
    }

    SizeType32 iter = 0;
    SizeType32 numReady = 0;
    while (numReady < numExpectedResponses && iter < mMaxWaitMs)
    {
        numReady = 0;
        for (auto id : ids)
        {
            numReady += executor.getNumResponsesReady(id);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
    // Expect one response per request
    for (auto id : ids)
    {
        SizeType32 numReady = executor.getNumResponsesReady(id);
        EXPECT_EQ(numReady, reqNumExpectedResponses.at(id));
    }
    auto numResponsesReady = executor.getNumResponsesReady();
    EXPECT_EQ(numResponsesReady, numExpectedResponses);
}

namespace
{

void runTest(Executor& executor, fs::path const& inputPath, ModelIds const& modelIds,
    FlakyTestInfo const& flakyTestInfo, bool streaming, SizeType32 const vocabSizePadded, BeamResult const& beamResult,
    OutputConfig const& outConfig, bool isSpeculativeDecoding, int maxWaitMs, bool returnAllGeneratedTokens,
    SizeType32 const numReturnSequences, bool isNonGreedySampling, SizeType32 const modelParallelism)
{
    auto const beamWidth = beamResult.beamWidth;

    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto const& givenInput = tr::utils::loadNpy(manager, inputPath.string(), tr::MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, modelIds.padId);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(*givenInput);

    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    // Load expected outputs for each beam width value
    auto testData = TestData::loadTestData(beamResult, *givenInput, beamWidth, manager, outConfig, modelIds);
    auto const maxSeqLen = testData.maxSeqLen;

    // Load expected outputs and inputs
    SizeType32 numRequests = static_cast<SizeType32>(givenInputLengths.size());
    SizeType32 maxRequests = numRequests;
    std::vector<Request> requests;
    std::vector<SizeType32> reqMaxNewTokens;

    auto samplingConfig = tensorrt_llm::executor::SamplingConfig(beamWidth);
    // top-k will be set by a large number to test non-identical N sequences.
    if (isNonGreedySampling)
    {
        samplingConfig.setTopK(32);
    }
    samplingConfig.setNumReturnSequences(numReturnSequences);

    for (SizeType32 req = 0; req < maxRequests; ++req)
    {
        SizeType32 inputLen = givenInputLengths.at(req);
        auto maxNewTokens = maxSeqLen - maxInputLength;
        reqMaxNewTokens.push_back(maxNewTokens);
        SizeType32 endId = -1;
        auto const* const seqBegin = givenInputData + req * maxInputLength;
        VecTokens tokens(seqBegin, seqBegin + inputLen);
        auto request = Request(
            VecTokens(seqBegin, seqBegin + inputLen), maxNewTokens, streaming, samplingConfig, outConfig, endId);
        request.setReturnAllGeneratedTokens(returnAllGeneratedTokens);
        requests.emplace_back(std::move(request));
    }

    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();

    // Expected return sizes.
    auto const numSequences = beamWidth > 1 ? 1 : numReturnSequences;
    auto const numReturnBeams = std::min(beamWidth, numReturnSequences);

    if (worldRank == 0)
    {
        auto const reqIds = executor.enqueueRequests(requests);

        std::unordered_map<SizeType32, std::vector<BeamTokens>> tokens;
        std::unordered_map<IdType, SizeType32> reqIdToBatchId;

        for (SizeType32 req = 0; req < reqIds.size(); ++req)
        {
            std::vector<BeamTokens> resultTokens(numSequences, BeamTokens(numReturnBeams));
            tokens[req] = std::move(resultTokens);
            reqIdToBatchId[reqIds.at(req)] = req;
        }

        // Get the new tokens for each requests
        int32_t numFinished = 0;
        int iter = 0;
        std::unordered_map<IdType, SizeType32> numResponses;
        while (numFinished < maxRequests && iter < maxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                auto batchId = reqIdToBatchId.at(response.getRequestId());
                numResponses[batchId]++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto seqIdx = result.sequenceIndex;

                    auto const& contextLogits = result.contextLogits;
                    auto const& genLogits = result.generationLogits;
                    auto const& outputTokenIds = result.outputTokenIds;

                    EXPECT_EQ(result.finishReasons.size(), numReturnBeams);
                    for (SizeType32 beam = 0; beam < numReturnBeams; ++beam)
                    {
                        auto const& newTokens = outputTokenIds.at(beam);
                        auto& reqTokens = tokens.at(batchId).at(seqIdx).at(beam);

                        if (!returnAllGeneratedTokens)
                        {
                            reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                        }
                        else
                        {
                            EXPECT_EQ(newTokens.size(),
                                (numResponses.at(batchId) + numReturnSequences - 1) / numReturnSequences);
                            reqTokens = newTokens;
                        }
                        // FinishReason is only supported for bw=1 and inflight batching.
                        if (beamWidth == 1)
                        {
                            EXPECT_EQ(result.finishReasons.at(beam),
                                result.isSequenceFinal ? FinishReason::kLENGTH : FinishReason::kNOT_FINISHED);
                        }
                    }

                    auto const& cumLogProbs = result.cumLogProbs;
                    auto const& logProbs = result.logProbs;
                    auto const& beamTokens = tokens.at(batchId).at(seqIdx);
                    EXPECT_EQ(beamTokens.size(), numReturnBeams);

                    if (!isNonGreedySampling)
                    {
                        float const logitsAtol = modelParallelism > 1 ? 1e-1 : 1e-2;
                        float const logitsRtol = modelParallelism > 1 ? 1e-2 : 1e-3;

                        testData.verifyLogProbs(outConfig.returnLogProbs, streaming, outConfig.excludeInputFromOutput,
                            givenInputLengths.at(batchId), beamWidth, beamTokens, cumLogProbs, logProbs, batchId,
                            flakyTestInfo);
                        testData.validateContextLogits(outConfig.returnContextLogits, givenInputLengths.at(batchId),
                            beamWidth, contextLogits, vocabSizePadded, batchId, logitsAtol, logitsRtol);
                        testData.validateGenerationLogits(outConfig.returnGenerationLogits, result.isSequenceFinal,
                            streaming, outConfig.excludeInputFromOutput, givenInputLengths.at(batchId),
                            reqMaxNewTokens.at(batchId), beamWidth, beamTokens, genLogits, vocabSizePadded, batchId,
                            returnAllGeneratedTokens, logitsAtol, logitsRtol);
                    }

                    // Ignore first iteration as it doesn't use draft tokens
                    if (outConfig.returnPerfMetrics && isSpeculativeDecoding
                        && result.requestPerfMetrics.value().iter > 0)
                    {
                        auto& specDecMetrics = result.requestPerfMetrics.value().speculativeDecoding;
                        // 4 draft tokens are used per step
                        EXPECT_EQ(specDecMetrics.totalDraftTokens, result.requestPerfMetrics.value().iter.value() * 4);
                        EXPECT_EQ(specDecMetrics.acceptanceRate,
                            static_cast<float>(specDecMetrics.totalAcceptedDraftTokens)
                                / specDecMetrics.totalDraftTokens);
                    }
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, maxWaitMs);
        testData.verifyOutput(tokens, givenInputLengths, streaming, outConfig.excludeInputFromOutput, flakyTestInfo,
            isSpeculativeDecoding, beamWidth, numSequences, isNonGreedySampling);
    }
}

void runTest(fs::path const& modelPath, ExecutorConfig const& executorConfig, fs::path const& inputPath,
    ModelIds const& modelIds, FlakyTestInfo const& flakyTestInfo, bool streaming, SizeType32 const vocabSizePadded,
    BeamResult const& beamResult, OutputConfig const& outConfig, bool isSpeculativeDecoding, int maxWaitMs,
    bool returnAllGeneratedTokens, SizeType32 const numReturnSequences, bool isNonGreedySampling,
    SizeType32 const modelParallelism)
{
    auto executor = Executor{modelPath, ModelType::kDECODER_ONLY, executorConfig};

    runTest(executor, inputPath, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult, outConfig,
        isSpeculativeDecoding, maxWaitMs, returnAllGeneratedTokens, numReturnSequences, isNonGreedySampling,
        modelParallelism);
}

ExecutorConfig createExecutorConfig(SizeType32 maxBeamWidth, bool useOrchestratorMode, bool gatherGenerationLogits,
    std::optional<std::vector<SizeType32>> deviceIds = std::nullopt,
    std::optional<std::vector<SizeType32>> participantIds = std::nullopt)
{
    // Note: we reduce memory fraction for cases that return context/generation logits which require more free
    // memory
    FloatType constexpr freeGpuMemoryFraction{0.5F};
    KvCacheConfig kvCacheConfig(false, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction);
    auto executorConfig = ExecutorConfig(maxBeamWidth);
    executorConfig.setKvCacheConfig(kvCacheConfig);
    executorConfig.setNormalizeLogProbs(false);
    executorConfig.setGatherGenerationLogits(gatherGenerationLogits);

    std::optional<OrchestratorConfig> orchestratorConfig = std::nullopt;
    if (useOrchestratorMode)
    {
        orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
    }
    auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
        useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::move(deviceIds),
        std::move(participantIds), orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    return executorConfig;
}

} // namespace

TEST_P(AllParamsTest, TokenComparison)
{
    auto const streaming = std::get<0>(GetParam());
    auto const& beamWidth = std::get<1>(GetParam());
    OutputConfig outConfig;
    outConfig.returnLogProbs = std::get<2>(GetParam());
    outConfig.excludeInputFromOutput = std::get<3>(GetParam());
    outConfig.returnContextLogits = std::get<4>(GetParam());
    outConfig.returnGenerationLogits = std::get<5>(GetParam());
    auto const modelName = std::get<6>(GetParam());
    auto const useOrchestratorMode = std::get<7>(GetParam());
    auto const returnAllGeneratedTokens = std::get<8>(GetParam());
    auto const numReturnSequences = std::get<9>(GetParam());
    if (returnAllGeneratedTokens && !streaming)
    {
        GTEST_SKIP() << "Test does not support returnAllGeneratedTokens without streaming";
    }

    std::optional<std::vector<SizeType32>> participantIds = std::nullopt;

    BeamResult beamResult{beamWidth};

    ASSERT_TRUE(fs::exists(DATA_PATH));

    fs::path modelPath;
    // set defaults and adjust if needed by different models
    fs::path inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};
    bool isSpeculativeDecoding{false};

    SizeType32 vocabSizePadded = 50257;

    // NOTE: This can be used to disable checks for certain prompt batch entries
    FlakyTestInfo flakyTestInfo;

    if (modelName == "gpt")
    {
        auto const resultsPath
            = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        if (outConfig.returnContextLogits || outConfig.returnGenerationLogits)
        {
            modelPath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR() / "tp1-pp1-cp1-gpu";
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE();
            beamResult.contextLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE();
            beamResult.genLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE();
            if (outConfig.returnLogProbs)
            {
                beamResult.cumLogProbsFile
                    = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_CUM_LOG_PROBS_FILE();
                beamResult.logProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_LOG_PROBS_FILE();
            }
        }
        else
        {
            modelPath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_FILE();
            if (outConfig.returnLogProbs)
            {
                beamResult.cumLogProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE();
                beamResult.logProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE();
            }
        }
    }
    else if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1"
        || modelName == "llama_tp1_pp2_cp1")
    {
        inputPath = DATA_PATH / LLAMA_INPUT_FILE;
        modelIds.padId = LLAMA_PAD_ID;
        modelIds.endId = LLAMA_END_ID;

        vocabSizePadded = LLAMA_VOCAB_SIZE_PADDED;

        auto const resultsPath
            = LLAMA_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        if (modelName == "llama_tp4_pp1_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp2_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp2-cp1-gpu";
        }
        else if (modelName == "llama_tp2_pp2_cp1")
        {
            beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE();
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
        }
        beamResult.genLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_TP4_PP1_FILE();
        if (outConfig.returnLogProbs)
        {
            beamResult.cumLogProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_TP4_PP1_FILE();
            beamResult.logProbsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_TP4_PP1_FILE();
        }
    }
    else if (modelName == "medusa")
    {
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "Medusa does not support beam search.");
        auto const resultsPath = MEDUSA_DATA_PATH / "sampling";
        auto modelSpec = ModelSpec::getDefaultModelSpec()
                             .useMedusa()
                             .setInputFile("input_tokens_long.npy")
                             .setMaxOutputLength(128);
        beamResult.resultsFile = resultsPath / modelSpec.getResultsFile();
        modelPath = MEDUSA_MODEL_PATH / modelSpec.getModelPath() / "tp1-pp1-cp1-gpu";

        inputPath = DATA_PATH / "input_vicuna.npy";
        modelIds.padId = 2;
        modelIds.endId = 2;
        isSpeculativeDecoding = true;
        outConfig.returnPerfMetrics = true;
    }
    else if (modelName == "chatglm" || modelName == "chatglm2" || modelName == "chatglm3" || modelName == "glm")
    {
        fs::path resultsPath;
        if (modelName == "chatglm")
        {
            resultsPath = CHATGLM_DATA_PATH;
            modelPath = CHATGLM_MODEL_PATH;
        }
        else if (modelName == "chatglm2")
        {
            resultsPath = CHATGLM2_DATA_PATH;
            modelPath = CHATGLM2_MODEL_PATH;
        }
        else if (modelName == "chatglm3")
        {
            resultsPath = CHATGLM3_DATA_PATH;
            modelPath = CHATGLM3_MODEL_PATH;
        }
        else if (modelName == "glm")
        {
            resultsPath = GLM_DATA_PATH;
            modelPath = GLM_MODEL_PATH;
        }
        resultsPath /= (beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth);
        beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_FILE();
        modelPath = modelPath / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";

        char versionChatglm{0};
        if (size_t index = modelPath.string().find("chatglm"); index != std::string::npos)
        {
            versionChatglm = modelPath.string()[index + 7];
            std::string const vChatglmString
                = (versionChatglm == '-') ? std::string("") : std::string(1, versionChatglm);
            inputPath = DATA_PATH / ("input_tokens_chatglm" + vChatglmString + "-6b.npy");
            modelIds.padId = (versionChatglm == '-') ? 3 : 0;
            modelIds.endId = (versionChatglm == '-') ? 130005 : 2;
        }
        else if (size_t index = modelPath.string().find("glm-10b"); index != std::string::npos)
        {
            inputPath = DATA_PATH / "input_tokens_glm-10b.npy";
            modelIds.padId = 50256;
            modelIds.endId = 50258;
        }

        if (versionChatglm != 0)
        {
            flakyTestInfo.batchIdBeams.insert(std::make_pair(1, 0));
        }
    }
    else
    {
        TLLM_THROW("Unrecognized modelName");
    }

    if (streaming && beamWidth > 1)
    {
        GTEST_SKIP() << "Test does not support streaming with beam search";
    }

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1"
        || modelName == "llama_tp1_pp2_cp1")
    {
        // For llama model, only run for multiple GPUs
        // This is detected by setting an env variable when running the test
        char const* val = getenv("RUN_LLAMA_MULTI_GPU");
        if (val == nullptr)
        {
            GTEST_SKIP() << "Skipping Llama test";
        }

        if (outConfig.returnContextLogits)
        {
            GTEST_SKIP() << "Skipping context logits tests for mpi runs";
        }

        // Check that it was launched with right number of MPI ranks
        if (!useOrchestratorMode && COMM_SESSION.getSize() != 4)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Leader mode and world size is not equal to 4";
        }
        if (useOrchestratorMode && COMM_SESSION.getSize() != 1)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Orchestrator mode and World size is not equal to 1";
        }
    }
    auto decoderJsonConfig = tensorrt_llm::runtime::GptJsonConfig::parse(modelPath / "config.json");

    auto const modelTP = decoderJsonConfig.getTensorParallelism();
    auto const modelPP = decoderJsonConfig.getPipelineParallelism();
    auto const modelParallelism = modelTP * modelPP;
    int deviceCount = -1;
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::optional<std::vector<SizeType32>> deviceIds = std::vector<SizeType32>(modelParallelism);
    for (auto i = 0; i < deviceIds->size(); i++)
    {
        deviceIds->at(i) = i % deviceCount;
    }
    if (modelName == "llama_tp1_pp2_cp1")
    {
        auto const& session = tensorrt_llm::mpi::MpiComm::world();
        if (session.getSize() != 4)
        {
            FAIL() << "Llama-tp1-pp2 is intended solely for testing coexisting engines within the same MPI world,"
                      " which requires a session size of 4. However, the current session size is "
                   << session.getSize() << " .";
        }
        if (session.getRank() / 2 == 0)
        {
            participantIds = std::vector<SizeType32>{0, 1};
            deviceIds = std::vector<SizeType32>{0, 1};
        }
        else
        {
            participantIds = std::vector<SizeType32>{2, 3};
            deviceIds = std::vector<SizeType32>{2, 3};
        }
    }

    if (modelPP > 1)
    {
        std::reverse(deviceIds->begin(), deviceIds->end());
        if (modelTP > 1)
        {
            for (SizeType32 ppRank = 0; ppRank < modelPP; ppRank++)
            {
                std::reverse(deviceIds->begin() + ppRank * modelTP, deviceIds->begin() + (ppRank + 1) * modelPP);
            }
        }
    }

    // Returning logits will bring higher latency
    if (streaming && (outConfig.returnContextLogits || outConfig.returnGenerationLogits))
    {
        mMaxWaitMs = 20000;
    }

    auto executorConfig = createExecutorConfig(beamWidth, useOrchestratorMode, outConfig.returnGenerationLogits,
        std::move(deviceIds), std::move(participantIds));

    runTest(modelPath, executorConfig, inputPath, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult,
        outConfig, isSpeculativeDecoding, mMaxWaitMs, returnAllGeneratedTokens, numReturnSequences, false,
        modelParallelism);
}

TEST_F(GptExecutorTest, ChangeBeamWidth)
{
    SizeType32 constexpr maxBeamWidth{2};
    auto executorConfig = ExecutorConfig(maxBeamWidth);

    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 constexpr beamWidth1{1};
    SizeType32 constexpr beamWidth2{2};
    SizeType32 constexpr maxNewTokens{2};
    VecTokens inputTokens{1, 2, 3, 4};

    // Create requests with different beam widths
    std::vector<Request> requests;
    requests.emplace_back(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth1));
    requests.emplace_back(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth1));
    requests.emplace_back(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth2));
    requests.emplace_back(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth1));

    auto requestIds = executor.enqueueRequests(requests);

    int numFinished = 0;
    int iter = 0;
    while (numFinished < 4 && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                auto err = response.getErrorMsg();
                std::cout << "err:" << err << std::endl;
                FAIL() << "Should not get a response with error";
            }
            else
            {
                auto result = response.getResult();
                numFinished += static_cast<int>(result.isFinal);
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);

    auto stats = executor.getLatestIterationStats();
    uint64_t currentIter = 0;
    for (auto const& stat : stats)
    {
        // TODO: enable this check when stats are cleaned
        // EXPECT_EQ(stat.iter, currentIter);
        if (stat.iter < 2)
        {
            // req 1 and 2 run with same beam width
            EXPECT_EQ(stat.numActiveRequests, 2);
        }
        else if (stat.numActiveRequests != 0) // TODO: remove this check when stats are cleaned
        {
            // req 3 or 4 run width different beam width
            EXPECT_EQ(stat.numActiveRequests, 1);
        }

        ++currentIter;
    }
}

void doTokenComparisonChangeBeamWidth(bool enableReuse, SizeType32 maxWaitMs)
{
    SizeType32 constexpr maxBeamWidth{2};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    auto constexpr streaming = false;

    // Create executor config
    auto kvCacheConfig = KvCacheConfig(enableReuse);
    auto executorConfig = ExecutorConfig(maxBeamWidth, SchedulerConfig(), kvCacheConfig);

    // Create executor
    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto const inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};

    OutputConfig outConfig;
    FlakyTestInfo flakyTestInfo;
    bool constexpr isSpeculativeDecoding{false};

    for (SizeType32 beamWidth : {1, 2})
    {
        BeamResult beamResult{beamWidth};
        auto const resultsPath
            = GPT_DATA_PATH / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE();
        beamResult.contextLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE();
        beamResult.genLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE();

        runTest(executor, inputPath, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult, outConfig,
            isSpeculativeDecoding, maxWaitMs, false, 1, false, 1);
    }
}

TEST_F(GptExecutorTest, TokenComparisonChangeBeamWidth)
{
    doTokenComparisonChangeBeamWidth(false, mMaxWaitMs);
}

TEST_F(GptExecutorTest, TokenComparisonChangeBeamWidthBlockReuse)
{
    doTokenComparisonChangeBeamWidth(true, mMaxWaitMs);
}

TEST_F(GptExecutorTest, NReturnRandomness)
{
    SizeType32 constexpr maxBeamWidth{1};
    SizeType32 constexpr numReturnSequences{2};
    SizeType32 constexpr vocabSizePadded{50257}; // gpt vocabSizePadded
    auto constexpr streaming = false;

    // Create executor config
    auto executorConfig = ExecutorConfig(maxBeamWidth);

    // Create executor
    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto const inputPath = DATA_PATH / "input_tokens.npy";
    ModelIds modelIds{50256, 50256};

    OutputConfig outConfig;
    FlakyTestInfo flakyTestInfo;
    bool constexpr isSpeculativeDecoding{false};

    BeamResult beamResult{maxBeamWidth};
    auto const resultsPath = GPT_DATA_PATH / "sampling";
    beamResult.resultsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE();
    beamResult.contextLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE();
    beamResult.genLogitsFile = resultsPath / PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE();

    runTest(executor, inputPath, modelIds, flakyTestInfo, streaming, vocabSizePadded, beamResult, outConfig,
        isSpeculativeDecoding, mMaxWaitMs, false, 1, true, 1);
}

TEST_F(GptExecutorTest, TimedOut)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // No requests enqueued, expect no responses
    auto numResponsesReady = executor.getNumResponsesReady();
    EXPECT_EQ(numResponsesReady, 0);

    std::chrono::milliseconds waitTime(10);
    auto responses = executor.awaitResponses(waitTime);
    EXPECT_EQ(responses.size(), 0);
}

TEST_F(GptExecutorTest, MaxSeqIdleMicrosecondsError)
{
    auto executorConfig = ExecutorConfig(1);
    // Request will time out
    executorConfig.setMaxSeqIdleMicroseconds(1);
    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 constexpr maxNewTokens{5};
    VecTokens inputTokens{1, 2, 3, 4};

    std::vector<Request> requests;
    requests.emplace_back(inputTokens, maxNewTokens, false);

    auto requestIds = executor.enqueueRequests(requests);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                auto err = response.getErrorMsg();
                std::cout << "err:" << err << std::endl;
                EXPECT_THAT(err, testing::HasSubstr("Unable to get batch slot for request ID"));
                done = true;
            }
            else
            {
                FAIL() << "Should get a response with error";
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
}

void logitsProcessorMixedReqsTest(std::string const& modelDir, SizeType32 worldRank, SizeType32 maxWaitMs,
    bool replicated, std::optional<std::vector<SizeType32>> deviceIds);

TEST_P(LogitsProcParamsTest, All)
{
    auto const modelName = std::get<0>(GetParam());
    auto const batched = std::get<1>(GetParam());
    auto const replicated = std::get<2>(GetParam());

    std::string modelDir;
    int tp_size = 1, pp_size = 1, cp_size = 1;
    std::optional<std::vector<SizeType32>> deviceIds = std::nullopt;

    if (modelName == "llama_tp1_pp1_cp1")
    {
        modelDir = "tp1-pp1-cp1-gpu";
    }
    else if (modelName == "llama_tp4_pp1_cp1")
    {
        modelDir = "tp4-pp1-cp1-gpu";
        tp_size = 4;
    }
    else if (modelName == "llama_tp1_pp4_cp1")
    {
        modelDir = "tp1-pp4-cp1-gpu";
        pp_size = 4;
        deviceIds = std::vector<SizeType32>{3, 2, 1, 0};
    }
    else if (modelName == "llama_tp2_pp2_cp1")
    {
        modelDir = "tp2-pp2-cp1-gpu";
        tp_size = pp_size = 2;
        deviceIds = std::vector<SizeType32>{2, 3, 0, 1};
    }
    else
    {
        TLLM_THROW("Unrecognized modelName");
    }
    std::filesystem::path modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / modelDir;

    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();
    auto const worldSize = comm.getSize();

    if (tp_size * pp_size * cp_size != 1)
    {
        // Run multi GPU test only when env variable is set
        char const* val = getenv("RUN_LLAMA_MULTI_GPU");
        if (val == NULL)
        {
            GTEST_SKIP() << "Skipping multi-gpu logits post processor test";
        }

        if (worldSize != 4)
        {
            FAIL() << "Leader mode and world size is not equal to 4";
        }
    }
    else
    {
        // This has no effect for single-GPU tests
        if (replicated)
        {
            GTEST_SKIP() << "Skipping single-gpu replicated logits post processor test";
        }
    }

    // Configuration options
    bool const streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    SizeType32 numRequests = 20;
    IdType const kClientId = 1234;

    SizeType32 beamWidth = 1;
    SizeType32 maxPromptLen = 20;
    SizeType32 maxMaxNewTokens = 20;

    SizeType32 constexpr endId{2};
    SizeType32 constexpr vocabSizePadded{32000}; // llama-7b vocabSizePadded
    // We just use tokenIdCalculator to generate a token_id based on request index, output position and max new tokens.
    // Then LogitsPostProcessor set all other logits except the generated token_id to large negative value.
    // So the output token should be the generated token by tokenIdCalculator.
    auto tokenIdCalculator = [endId, vocabSizePadded](IdType req, SizeType32 pos)
    {
        SizeType32 tokenId = (req * 1000 + pos) % vocabSizePadded;
        if (tokenId == endId)
        {
            tokenId = 0;
        }
        return tokenId;
    };

    std::unordered_map<IdType, VecTokens> tokens;
    std::unordered_map<IdType, SizeType32> expectedNumTokens;
    std::unordered_map<IdType, VecTokens> expectedOutputTokens;

    // Enqueue the requests
    auto enqueueRequests = [&](Executor& executor, std::optional<std::string const> logitsProcessorName,
                               std::optional<LogitsPostProcessor> logitsProcessor = std::nullopt)
    {
        tokens.clear();
        expectedNumTokens.clear();
        expectedOutputTokens.clear();

        for (SizeType32 req = 0; req < numRequests; ++req)
        {
            SizeType32 promptLen = rand() % maxPromptLen + 1;
            SizeType32 maxNewTokens = rand() % maxMaxNewTokens + 1;

            auto request = Request(VecTokens(promptLen, 1), maxNewTokens, streaming,
                tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig, endId);
            request.setClientId(kClientId);
            if (logitsProcessorName)
            {
                request.setLogitsPostProcessorName(logitsProcessorName.value());
            }
            else if (logitsProcessor)
            {
                request.setLogitsPostProcessor(logitsProcessor.value());
            }
            auto reqId = executor.enqueueRequest(std::move(request));
            tokens[reqId] = {};
            expectedNumTokens[reqId] = (streaming ? 0 : (excludeInputFromOutput ? 0 : promptLen)) + maxNewTokens;
            expectedOutputTokens[reqId] = {};
            if (!streaming && !excludeInputFromOutput)
            {
                expectedOutputTokens[reqId].resize(promptLen, 1);
            }
            for (SizeType32 outputPos = 0; outputPos < maxNewTokens; ++outputPos)
            {
                SizeType32 outputTokenId = tokenIdCalculator(reqId, outputPos + promptLen);
                expectedOutputTokens[reqId].push_back(outputTokenId);
            }
        }
    };

    // Get the new tokens for each requests
    auto collectResponses = [&](Executor& executor)
    {
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < numRequests && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    EXPECT_EQ(response.getClientId().value(), kClientId);
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                    auto& reqTokens = tokens.at(response.getRequestId());
                    reqTokens.insert(reqTokens.end(), std::make_move_iterator(newTokens.begin()),
                        std::make_move_iterator(newTokens.end()));
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, mMaxWaitMs);
    };

    // Check that tokens matches expectations
    auto checkOutput = [&]()
    {
        for (auto const& [reqId, numTokens] : expectedNumTokens)
        {
            EXPECT_EQ(expectedNumTokens[reqId], tokens[reqId].size()) << "reqId " << reqId;
            for (SizeType32 tokenPos = 0;
                 tokenPos < std::min<SizeType32>(expectedNumTokens[reqId], tokens[reqId].size()); ++tokenPos)
            {
                EXPECT_EQ(expectedOutputTokens[reqId][tokenPos], tokens[reqId][tokenPos])
                    << "reqId=" << reqId << ", tokenPos=" << tokenPos;
            }
        }
    };

    // Test non-batched logits processor
    std::string const logitsProcessorName = "SelectToken";

    auto logitsPostProcessorFn = [&](IdType reqId, Tensor& logits, BeamTokens const& tokens, StreamPtr const& streamPtr,
                                     std::optional<IdType> clientId)
    {
        if (replicated)
        {
            EXPECT_TRUE(worldRank <= tp_size - 1);
        }
        else
        {
            EXPECT_TRUE(worldRank == 0);
        }
        EXPECT_TRUE(clientId.value() == kClientId);
        SizeType32 numTokens = tokens.at(0).size();
        SizeType32 pos = numTokens;
        SizeType32 outputTokenId = tokenIdCalculator(reqId, pos);
        auto logitsDataType = logits.getDataType();
        EXPECT_TRUE(logitsDataType == DataType::kFP16 || logitsDataType == DataType::kBF16
            || logitsDataType == DataType::kFP32);
        // logits has shape [draftLength + 1, reqBeamWidth, vocabSize]
        auto logitsCpu = tensorrt_llm::executor::Tensor::cpu(logitsDataType, logits.getShape());
        auto* dataPtr = logitsCpu.getData();
        auto eltSize = logitsCpu.getSizeInBytes() / logitsCpu.getSize();
        EXPECT_TRUE(eltSize == 2 || eltSize == 4);
        if (eltSize == 2)
        {
            auto* dataPtrU16 = static_cast<uint16_t*>(dataPtr);
            uint16_t hugeNegValue = logitsDataType == DataType::kFP16 ? 0xFBFF : 0xFF7F; // a huge negative value
            for (size_t i = 0; i < logitsCpu.getSize(); ++i)
            {
                dataPtrU16[i] = hugeNegValue;
            }
            dataPtrU16[outputTokenId] = 0;
        }
        else
        {
            auto* dataPtrFloat = static_cast<float*>(dataPtr);
            for (size_t i = 0; i < logitsCpu.getSize(); ++i)
            {
                dataPtrFloat[i] = -HUGE_VALF;
            }
            dataPtrFloat[outputTokenId] = 0.0f;
        }

        logits.setFrom(logitsCpu, streamPtr);
    };

    if (!batched)
    {
        auto executorConfig = ExecutorConfig(beamWidth);
        LogitsPostProcessorConfig logitsProcConfig{
            std::unordered_map<std::string, tensorrt_llm::executor::LogitsPostProcessor>{
                {logitsProcessorName, logitsPostProcessorFn}},
            std::nullopt, replicated};
        executorConfig.setLogitsPostProcessorConfig(logitsProcConfig);
        if (deviceIds.has_value())
        {
            auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());
            parallelConfig.setDeviceIds(deviceIds.value());
            executorConfig.setParallelConfig(parallelConfig);
        }
        auto executor = Executor(modelPath, ModelType::kDECODER_ONLY, executorConfig);

        if (worldRank == 0)
        {
            enqueueRequests(executor, logitsProcessorName);
            collectResponses(executor);
            checkOutput();

            if (!replicated || tp_size == 1)
            {
                // Dynamic logits postprocessor must be used with replicate=false or no tensor parallelism.
                enqueueRequests(executor, std::nullopt, logitsPostProcessorFn);
                collectResponses(executor);
                checkOutput();
            }
        }
    }

    // Test batched logits processor
    auto logitsPostProcessorBatchedFn
        = [logitsPostProcessorFn](std::vector<IdType> const& reqIdBatch, std::vector<Tensor>& logitsBatch,
              std::vector<std::reference_wrapper<BeamTokens const>> const& tokensBatch, StreamPtr const& streamPtr,
              std::vector<std::optional<IdType>> const& clientIdBatch)
    {
        for (int sample = 0; sample < reqIdBatch.size(); sample++)
        {
            logitsPostProcessorFn(
                reqIdBatch[sample], logitsBatch[sample], tokensBatch[sample], streamPtr, clientIdBatch[sample]);
        }
    };

    if (batched)
    {
        auto batchedExecutorConfig = ExecutorConfig(beamWidth);
        if (deviceIds.has_value())
        {
            auto parallelConfig = batchedExecutorConfig.getParallelConfig().value_or(ParallelConfig());

            parallelConfig.setDeviceIds(deviceIds.value());
            batchedExecutorConfig.setParallelConfig(parallelConfig);
        }
        LogitsPostProcessorConfig logitsProcConfig{std::nullopt, logitsPostProcessorBatchedFn, replicated};
        batchedExecutorConfig.setLogitsPostProcessorConfig(logitsProcConfig);

        auto batchedExecutor = Executor(modelPath, ModelType::kDECODER_ONLY, batchedExecutorConfig);

        if (worldRank == 0)
        {
            enqueueRequests(batchedExecutor, Request::kBatchedPostProcessorName);
            collectResponses(batchedExecutor);
            checkOutput();
        }
    }

    if (!batched)
    {
        logitsProcessorMixedReqsTest(modelDir, worldRank, mMaxWaitMs, replicated, std::move(deviceIds));
    }
}

// Test for mixing requests with and without logits processor.
void logitsProcessorMixedReqsTest(std::string const& modelDir, SizeType32 worldRank, SizeType32 maxWaitMs,
    bool replicated, std::optional<std::vector<SizeType32>> deviceIds)
{
    std::string const logitsProcessorName = "dummy";
    auto logitsPostProcessorFn = [&](IdType reqId, Tensor& logits, BeamTokens const& tokens, StreamPtr const& streamPtr,
                                     std::optional<IdType> clientId)
    {
        // Dummy callback that does not modify logits
        assert(!clientId.has_value());
    };

    LogitsPostProcessorConfig logitsProcConfig{
        std::unordered_map<std::string, tensorrt_llm::executor::LogitsPostProcessor>{
            {logitsProcessorName, logitsPostProcessorFn}},
        std::nullopt, replicated};

    // Create executor
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    executorConfig.setLogitsPostProcessorConfig(logitsProcConfig);
    if (deviceIds.has_value())
    {
        auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());

        parallelConfig.setDeviceIds(deviceIds.value());
        executorConfig.setParallelConfig(parallelConfig);
    }
    std::filesystem::path modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / modelDir;
    auto executor = Executor(modelPath, ModelType::kDECODER_ONLY, executorConfig);

    if (worldRank == 0)
    {
        SizeType32 numRequests = 2;
        SizeType32 promptLen = 5;

        // First request with no LP and many output tokens
        auto request1 = Request(VecTokens(promptLen, 1), 25);
        // Second request with LP and few output tokens
        auto request2 = Request(VecTokens(promptLen, 1), 5);
        request2.setLogitsPostProcessorName(logitsProcessorName);

        // Enqueue requests
        auto reqId1 = executor.enqueueRequest(request1);
        auto reqId2 = executor.enqueueRequest(request2);

        // Wait for responses
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < numRequests && iter < maxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                }
                else
                {
                    // Allow response with error only if awaitResponse processed a terminated request id
                    std::string err = "ReqId " + std::to_string(response.getRequestId())
                        + " has already been processed and was terminated.";
                    EXPECT_EQ(response.getErrorMsg(), err);
                }
            }
            ++iter;
        }
        EXPECT_LT(iter, maxWaitMs);
    }
}

TEST_F(GptExecutorTest, LogitsPostProcessorThrow)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    std::string const logitsProcessorName = "UnExistProcessor";

    auto request
        = Request(VecTokens(10, 1), 10, false, tensorrt_llm::executor::SamplingConfig(beamWidth), OutputConfig());
    request.setLogitsPostProcessorName(logitsProcessorName);
    EXPECT_THROW({ auto reqId = executor.enqueueRequest(std::move(request)); }, tensorrt_llm::common::TllmException);
}

static Response executeDraftRequest(Executor& executor)
{
    OutputConfig outputConfig;
    outputConfig.returnGenerationLogits = true;

    // Create the request
    SizeType32 maxNewTokens = 4;
    VecTokens inputTokens{1, 2, 3, 4};

    Request request{std::move(inputTokens), maxNewTokens};
    request.setOutputConfig(outputConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    return responses.at(0);
}

static Response executeTargetRequest(Executor& executor, Result const& draftResult)
{
    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};

    Request request{std::move(inputTokens), maxNewTokens};

    VecTokens const& outputTokenIds = draftResult.outputTokenIds.at(0);
    VecTokens draftTokens(outputTokenIds.end() - 4, outputTokenIds.end());

    auto const& logitsInfo = draftResult.specDecFastLogitsInfo.value();
    auto logitsTensor = logitsInfo.toTensor();

    ExternalDraftTokensConfig draftTokensConfig(
        std::move(draftTokens), logitsTensor, std::nullopt /* acceptance threshold */, true /* fastLogits */);
    request.setExternalDraftTokensConfig(draftTokensConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    return responses.at(0);
}

class SpeculativeDecodingTest : public GptExecutorTest
{
};

TEST_F(SpeculativeDecodingTest, SpecDecFastLogits)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtDraftEnginePath
        = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR() / "tp1-pp1-cp1-gpu";
    auto trtEnginePath
        = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR() / "tp1-pp1-cp1-gpu";

    FloatType freeGpuMemoryFraction = 0.3;
    auto kvCacheConfig
        = KvCacheConfig(true /* enableBlockReuse */, std::nullopt, std::nullopt, std::nullopt, freeGpuMemoryFraction);
    executorConfig.setKvCacheConfig(kvCacheConfig);

    tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
    int const worldSize = tensorrt_llm::mpi::MpiComm::world().getSize();
    ASSERT_EQ(worldSize, 3);
    int const myRank = tensorrt_llm::mpi::MpiComm::world().getRank();
    bool const isOrchestrator = (myRank == 0);

    auto orchestratorConfig
        = OrchestratorConfig(isOrchestrator, "" /* workerExecutablePath */, nullptr, false /* spawnPrcesses */);
    auto parallelConfig = ParallelConfig(
        CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR, std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto specDecConfig = SpeculativeDecodingConfig(true /* fastLogits */);
    executorConfig.setSpecDecConfig(specDecConfig);

    std::unique_ptr<Executor> draftExecutor;
    std::unique_ptr<Executor> targetExecutor;

    if (isOrchestrator)
    {
        auto executorConfigDraft = executorConfig;
        parallelConfig.setParticipantIds({1});
        executorConfigDraft.setParallelConfig(parallelConfig);

        draftExecutor = std::make_unique<Executor>(trtDraftEnginePath, ModelType::kDECODER_ONLY, executorConfigDraft);

        parallelConfig.setParticipantIds({2});
        executorConfig.setParallelConfig(parallelConfig);

        targetExecutor = std::make_unique<Executor>(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
    }
    else if (myRank == 1) // draft model process
    {
        parallelConfig.setParticipantIds({1});
        parallelConfig.setDeviceIds({0});
        executorConfig.setParallelConfig(parallelConfig);
        executorConfig.setGatherGenerationLogits(true);
        draftExecutor = std::make_unique<Executor>(trtDraftEnginePath, ModelType::kDECODER_ONLY, executorConfig);
    }
    else if (myRank == 2) // target model process
    {
        parallelConfig.setParticipantIds({2});
        parallelConfig.setDeviceIds({0});
        executorConfig.setParallelConfig(parallelConfig);
        draftExecutor = std::make_unique<Executor>(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
    }

    if (isOrchestrator)
    {
        auto response = executeDraftRequest(*draftExecutor);
        ASSERT_FALSE(response.hasError());
        response = executeTargetRequest(*targetExecutor, response.getResult());
        ASSERT_FALSE(response.hasError());
    }
}

TEST_F(GptExecutorTest, OrchestratorMaxQueueSize)
{
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    SizeType32 maxQueueSize = 6;
    ExecutorConfig executorConfig;
    executorConfig.setMaxQueueSize(maxQueueSize);
    auto orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
    auto parallelConfig = ParallelConfig(
        CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR, std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 100;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens);
    std::vector<IdType> requestIds;
    auto numberOfRequests = maxQueueSize * 5;
    requestIds.reserve(numberOfRequests);

    // Enqueue more requests than the queue can manage
    for (int i = 0; i < numberOfRequests; i++)
    {
        auto requestId = executor.enqueueRequest(request);
        requestIds.emplace_back(requestId);
    }

    auto responseVectors = executor.awaitResponses(std::move(requestIds));
    bool failedWithFullQueue = false;
    for (auto& responseVector : responseVectors)
    {
        for (auto& response : responseVector)
        {
            if (response.hasError())
            {
                EXPECT_THAT(response.getErrorMsg(),
                    testing::HasSubstr("Maximum queue size of 6 has been reached, please try again later"));
                failedWithFullQueue = true;
            }
        }
    }
    EXPECT_TRUE(failedWithFullQueue) << "Expected requests to fail due to maximum queue size reached";

    // Wait for requests to get scheduled to free up space in queue
    std::this_thread::sleep_for(std::chrono::milliseconds(maxQueueSize * 200));
    auto requestId = executor.enqueueRequest(std::move(request));
    auto responses = executor.awaitResponses(requestId);
    for (auto& response : responses)
    {
        EXPECT_FALSE(response.hasError());
    }
}

TEST_F(GptExecutorTest, SingleRequestInvalidInputs)
{
    bool streaming = true;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};

    std::vector<std::string> expectedErrMsgs;
    std::vector<Request> requests;

    // Invalid embedding bias shape
    {
        requests.emplace_back(inputTokens, maxNewTokens, streaming);
        auto embeddingBias = Tensor::cpu(DataType::kFP32, {1});
        requests.back().setEmbeddingBias(embeddingBias);
        expectedErrMsgs.emplace_back("embedding bias shape is not as expected");
    }

    for (auto req = 0; req < requests.size(); ++req)
    {
        auto& request = requests.at(req);
        auto const& expectedErrMsg = expectedErrMsgs.at(req);

        auto requestId = executor.enqueueRequest(std::move(request));

        // Try to get the new tokens
        bool done = false;
        int iter = 0;
        while (!done && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {

                    auto err = response.getErrorMsg();
                    EXPECT_THAT(err, testing::HasSubstr(expectedErrMsg));
                    done = true;
                }
                else
                {
                    FAIL() << "Expected an err: " << expectedErrMsg;
                }
            }
            ++iter;
        }
        EXPECT_EQ(done, true);
    }
}

TEST_F(GptExecutorTest, ExecutorKVCacheManager)
{

    bool streaming = true;
    int numRequests = 3;

    SizeType32 beamWidth = 1;
    SizeType32 maxNewTokens = 5;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto kvCacheConfig = KvCacheConfig(true, 128);
    kvCacheConfig.setEventBufferMaxSize(1024);
    executorConfig.setKvCacheConfig(kvCacheConfig);

    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    auto kvCacheManager = *executor.getKVCacheEventManager();

    // Created event should be available before any requests.
    auto events = kvCacheManager->getLatestEvents(std::chrono::seconds(1));
    EXPECT_EQ(events.size(), 1);
    EXPECT_TRUE(std::holds_alternative<KVCacheCreatedData>(events.front().data));

    // Create requests
    std::vector<Request> requests;
    for (int request = 0; request < 3; request++)
    {
        VecTokens inputTokens;
        for (int i = 0; i < 63; i++)
        {
            inputTokens.emplace_back(i + request);
        }
        requests.emplace_back(inputTokens, maxNewTokens, streaming);
    }

    for (auto req = 0; req < requests.size(); ++req)
    {
        auto& request = requests.at(req);

        auto requestId = executor.enqueueRequest(std::move(request));

        // Get the new tokens
        bool done = false;
        int iter = 0;
        while (!done && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(requestId, waitTime);
            for (auto& response : responses)
            {
                if (response.hasError())
                {
                    // This request failed for some reason, get error msg
                    std::string errStr
                        = "Request id " + std::to_string(requestId) + " failed with err " + response.getErrorMsg();
                    FAIL();
                }
                else
                {
                    auto result = response.getResult();
                    done = result.isFinal;
                    if (done)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        auto events = kvCacheManager->getLatestEvents(std::chrono::milliseconds(100));
                        if (req == 0)
                        {
                            EXPECT_EQ(events.size(), 3);

                            // Store the first context block
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.front().data).parentHash, std::nullopt);
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.front().data).blocks.size(), 1);
                            events.pop_front();
                            // Store the second (now completed) context block and the partial decode block.
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.front().data).blocks.size(), 1);
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.back().data).blocks.size(), 1);
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.front().data).blocks[0].blockHash,
                                std::get<KVCacheStoredData>(events.back().data).parentHash);
                        }
                        else
                        {
                            EXPECT_EQ(events.size(), 5);

                            // Remove a block to make room for the second context block. On the second request, we need
                            // to remove 2 blocks.
                            EXPECT_EQ(std::get<KVCacheRemovedData>(events.front().data).blockHashes.size(), req);
                            events.pop_front();
                            // Store the first filled context block
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.front().data).blocks.size(), 1);
                            events.pop_front();
                            // Remove a block for the decode phase
                            EXPECT_EQ(std::get<KVCacheRemovedData>(events.front().data).blockHashes.size(), 1);
                            events.pop_front();
                            // Store the final context block and the decode block
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.front().data).blocks.size(), 1);
                            events.pop_front();
                            EXPECT_EQ(std::get<KVCacheStoredData>(events.front().data).blocks.size(), 1);
                        }
                    }
                }
            }
            iter++;
        }
        EXPECT_EQ(done, true);
    }
}

TEST_F(GptExecutorTest, SingleRequestLora)
{
    bool streaming = true;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Load lora weights, config
    auto manager = tr::BufferManager(std::make_shared<tr::CudaStream>());
    auto loraWeightsTensor
        = std::shared_ptr(tr::utils::loadNpy(manager, LORA_WEIGHTS_FILE.string(), tr::MemoryType::kCPU));
    auto loraConfigTensor
        = std::shared_ptr(tr::utils::loadNpy(manager, LORA_CONFIG_FILE.string(), tr::MemoryType::kCPU));

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig());
    auto loraConfig = LoraConfig(0, detail::ofITensor(loraWeightsTensor), detail::ofITensor(loraConfigTensor));
    request.setLoraConfig(loraConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Get the new tokens
    VecTokens tokens;
    bool done = false;
    int iter = 0;
    std::chrono::milliseconds waitTime(1);
    while (!done && iter < mMaxWaitMs)
    {
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                // This request failed for some reason, get error msg
                std::string errStr
                    = "Request id " + std::to_string(requestId) + " failed with err " + response.getErrorMsg();
                FAIL();
            }
            else
            {
                auto result = response.getResult();
                done = result.isFinal;
                // Append tokens
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                tokens.insert(
                    tokens.end(), std::make_move_iterator(newTokens.begin()), std::make_move_iterator(newTokens.end()));
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(tokens.size(), maxNewTokens);
}

TEST_P(GuidedDecodingParamsTest, All)
{
    auto const modelName = std::get<0>(GetParam());
    std::filesystem::path enginePath;
    std::filesystem::path tokenizerInfoPath;
    int tp_size = 1, pp_size = 1, cp_size = 1;
    std::optional<std::vector<SizeType32>> deviceIds = std::nullopt;

    if (modelName == "gpt")
    {
        enginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
        tokenizerInfoPath = GPT_XGRAMMAR_TOKENIZER_INFO_PATH;
    }
    else if (modelName == "llama_tp1_pp1_cp1")
    {
        enginePath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
        tokenizerInfoPath = LLAMA_XGRAMMAR_TOKENIZER_INFO_PATH;
    }
    else if (modelName == "llama_tp4_pp1_cp1")
    {
        enginePath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        tokenizerInfoPath = LLAMA_XGRAMMAR_TOKENIZER_INFO_PATH;
        tp_size = 4;
    }
    else if (modelName == "llama_tp1_pp4_cp1")
    {
        enginePath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
        tokenizerInfoPath = LLAMA_XGRAMMAR_TOKENIZER_INFO_PATH;
        pp_size = 4;
        deviceIds = std::vector<SizeType32>{3, 2, 1, 0};
    }
    else if (modelName == "llama_tp2_pp2_cp1")
    {
        enginePath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
        tokenizerInfoPath = LLAMA_XGRAMMAR_TOKENIZER_INFO_PATH;
        tp_size = 2;
        pp_size = 2;
        deviceIds = std::vector<SizeType32>{2, 3, 0, 1};
    }
    else
    {
        TLLM_THROW("Unrecognized modelName");
    }

    auto& comm = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = comm.getRank();
    auto const worldSize = comm.getSize();

    if (tp_size * pp_size * cp_size > 1)
    {
        // Run multi GPU test only when env variable is set
        char const* val = getenv("RUN_LLAMA_MULTI_GPU");
        if (val == NULL)
        {
            GTEST_SKIP() << "Skipping multi-gpu guided decoding test";
        }
        else
        {
            if (worldSize != 4)
            {
                FAIL() << "Leader mode and world size is not equal to 4";
            }
        }
    }

    bool streaming = false;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);

    auto const tokenizerInfo = nlohmann::json::parse(std::ifstream{tokenizerInfoPath});
    auto const encodedVocab = tokenizerInfo["encoded_vocab"].template get<std::vector<std::string>>();
    auto const tokenizerStr = tokenizerInfo["tokenizer_str"].template get<std::string>();
    auto const stopTokenIds = tokenizerInfo["stop_token_ids"].template get<std::vector<TokenIdType>>();
    GuidedDecodingConfig guidedDecodingConfig(
        GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR, encodedVocab, tokenizerStr, stopTokenIds);
    executorConfig.setGuidedDecodingConfig(guidedDecodingConfig);

    if (deviceIds.has_value())
    {
        auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());

        parallelConfig.setDeviceIds(deviceIds.value());
        executorConfig.setParallelConfig(parallelConfig);
    }
    auto executor = Executor(enginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the requests
    VecTokens inputTokens;
    if (modelName == "gpt")
    {
        inputTokens = {2061, 318, 352, 10, 16, 30, 23998, 39559, 287, 257, 8633, 287, 33918, 5794, 25, 220};
    }
    else // llama
    {
        inputTokens = {
            128000, 62, 3923, 7037, 62, 16, 10, 16, 30, 62, 16533, 87710, 1265, 4404, 5356, 1265, 9643, 9132, 25, 62};
    }
    SizeType32 maxNewTokens = 10;
    SamplingConfig samplingConfig{};
    OutputConfig outputConfig{false, false, false, true};

    std::vector<Request> requests;
    requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outputConfig, stopTokenIds[0]);

    requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outputConfig, stopTokenIds[0]);
    requests.back().setGuidedDecodingParams(GuidedDecodingParams(GuidedDecodingParams::GuideType::kJSON));

    requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outputConfig, stopTokenIds[0]);
    std::string jsonSchema{
        R"({"properties": {"answer": {"title": "Answer", "type": "integer"}}, "required": ["answer"], "title": "Answer", "type": "object"})"};
    requests.back().setGuidedDecodingParams(
        GuidedDecodingParams(GuidedDecodingParams::GuideType::kJSON_SCHEMA, jsonSchema));

    requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outputConfig, stopTokenIds[0]);
    std::string regex{R"(\d+)"};
    requests.back().setGuidedDecodingParams(GuidedDecodingParams(GuidedDecodingParams::GuideType::kREGEX, regex));

    requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outputConfig, stopTokenIds[0]);
    std::string ebnfGrammar{R"(root ::= [0-9]+)"};
    requests.back().setGuidedDecodingParams(
        GuidedDecodingParams(GuidedDecodingParams::GuideType::kEBNF_GRAMMAR, ebnfGrammar));

    std::vector<VecTokens> expectedOutputTokens;
    if (modelName == "gpt")
    {
        expectedOutputTokens.push_back({1849, 7, 16, 10, 16, 8, 198, 16, 10, 16});
        expectedOutputTokens.push_back({90, 366, 3672, 1298, 366, 7554, 31780, 1600, 366, 12888});
        expectedOutputTokens.push_back({90, 366, 64, 77, 2032, 68, 81, 1, 1058, 352});
        expectedOutputTokens.push_back({25645, 25645, 25645, 25645, 25645, 25645, 25645, 25645, 25645, 25645});
        expectedOutputTokens.push_back({25645, 25645, 25645, 25645, 25645, 25645, 25645, 25645, 25645, 25645});
    }
    else // llama
    {
        expectedOutputTokens.push_back({16, 10, 16, 28, 17, 198, 62, 3923, 7037, 62});
        expectedOutputTokens.push_back({5018, 16, 794, 330, 16, 498, 330, 17, 794, 330});
        expectedOutputTokens.push_back({5018, 9399, 794, 16, 92});
        expectedOutputTokens.push_back({16});
        expectedOutputTokens.push_back({16});
    }

    if (executor.canEnqueueRequests())
    {
        // Enqueue the requests
        auto reqIds = executor.enqueueRequests(std::move(requests));

        // Get the responses
        int numFinished = 0;
        int iter = 0;
        while (numFinished < 5 && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                auto reqId = response.getRequestId();
                if (response.hasError())
                {
                    // This request failed for some reason, get error msg
                    std::string errStr
                        = "Request id " + std::to_string(reqId) + " failed with err " + response.getErrorMsg();
                    FAIL();
                }
                else
                {
                    auto result = response.getResult();
                    auto& newTokens = result.outputTokenIds.at(0);

                    int reqIdx = std::find(reqIds.begin(), reqIds.end(), reqId) - reqIds.begin();
                    EXPECT_THAT(newTokens, ::testing::ElementsAreArray(expectedOutputTokens[reqIdx]));
                }
                numFinished++;
            }
        }
        EXPECT_LT(iter, mMaxWaitMs);
        EXPECT_EQ(numFinished, 5);
    }
}

TEST_F(GptExecutorTest, GuidedDecodingFailure)
{
    bool streaming = false;

    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);

    std::vector<int> stopTokenIds{50256};
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the requests
    SizeType32 maxNewTokens = 10;
    SamplingConfig samplingConfig{};
    OutputConfig outputConfig{false, false, false, true};
    VecTokens inputTokens{2061, 318, 352, 10, 16, 30, 23998, 39559, 287, 257, 8633, 287, 33918, 5794, 25, 220};

    std::vector<Request> requests;
    requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outputConfig, stopTokenIds[0]);
    requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outputConfig, stopTokenIds[0]);
    requests.back().setGuidedDecodingParams(GuidedDecodingParams(GuidedDecodingParams::GuideType::kJSON));

    // Enqueue the requests
    auto reqIds = executor.enqueueRequests(std::move(requests));

    // Get the responses
    int numFinished = 0;
    int iter = 0;
    while (numFinished < 2 && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            auto reqId = response.getRequestId();
            int reqIdx = std::find(reqIds.begin(), reqIds.end(), reqId) - reqIds.begin();
            if (reqIdx == 0)
            {
                EXPECT_FALSE(response.hasError());
            }
            else
            {
                EXPECT_TRUE(response.hasError());
            }
            numFinished++;
        }
    }
    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(numFinished, 2);
}

TEST_P(ParamTest, SingleRequestCancelRequest)
{
    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;

    auto executorConfig = ExecutorConfig(beamWidth);
    auto trtEnginePath = GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 300;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    auto requestId = executor.enqueueRequest(std::move(request));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    executor.cancelRequest(requestId);

    // Try to get the new tokens
    bool done = false;
    int iter = 0;
    VecTokens tokens;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                FAIL() << "Did not expect errors";
            }
            else
            {
                auto result = response.getResult();
                done = result.isFinal;
                // Append tokens
                auto& newTokens = result.outputTokenIds.at(beamWidth - 1);
                if (done)
                {
                    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
                    {
                        EXPECT_EQ(result.finishReasons[beamIdx], FinishReason::kCANCELLED);
                    }
                }

                if (streaming && beamWidth > 1)
                {
                    tokens = newTokens;
                }
                else
                {
                    tokens.insert(tokens.end(), newTokens.begin(), newTokens.end());
                }
            }
        }
        ++iter;
    }
    EXPECT_EQ(done, true);
    EXPECT_LT(iter, mMaxWaitMs);
    auto expectedNumTokens
        = streaming ? maxNewTokens : (excludeInputFromOutput ? 0 : inputTokens.size()) + maxNewTokens;
    TLLM_LOG_INFO("num tokens: %d, expected %d", tokens.size(), expectedNumTokens);
    EXPECT_LT(tokens.size(), expectedNumTokens);
}

TEST_F(GptExecutorTest, orchModeFetchNewReqErr)
{
    SizeType32 beamWidth = 1;
    auto executorConfig = ExecutorConfig(beamWidth);

    auto orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
    auto parallelConfig = ParallelConfig(
        CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR, std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Create a req with invalid parameters
    SizeType32 maxNewTokens = 5;
    // Create very long prompt which should result in error during request validate
    VecTokens inputTokens(10000000);

    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto requestId = executor.enqueueRequest(request);
    auto requestId2 = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                auto err = response.getErrorMsg();
                EXPECT_THAT(err, testing::HasSubstr("exceeds maximum input length"));
                EXPECT_THAT(err, testing::HasSubstr("Encountered an error when fetching new request:"));
                done = true;
            }
            else
            {
                FAIL() << "Should get a response with error";
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
}

TEST_F(GptExecutorTest, orchModeForwardError)
{
    SizeType32 constexpr maxBeamWidth{1};
    auto executorConfig = ExecutorConfig(maxBeamWidth);

    auto orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
    auto parallelConfig = ParallelConfig(
        CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR, std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu");
    auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);

    // Setting request beam width to 2 which should cause failure
    SizeType32 constexpr beamWidth{2};
    SizeType32 constexpr maxNewTokens{5};
    VecTokens inputTokens{1, 2, 3, 4};

    auto request = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    auto requestId = executor.enqueueRequest(request);
    auto requestId2 = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            if (response.hasError())
            {
                auto err = response.getErrorMsg();
                std::cout << "err:" << err << std::endl;
                EXPECT_THAT(
                    err, testing::HasSubstr("Requested beam width 2 is larger than configured max beam width 1"));
                done = true;
            }
            else
            {
                FAIL() << "Should get a response with error";
            }
        }
        ++iter;
    }
    EXPECT_LT(iter, mMaxWaitMs);
}

TEST_P(ParamCancelReqTest, MultipleRequestsMultiGpuCancelRequest)
{
    auto const useOrchestratorMode = std::get<0>(GetParam());
    auto const beamWidth = std::get<1>(GetParam());
    auto const modelName = std::get<2>(GetParam());

    std::optional<std::vector<SizeType32>> deviceIds = std::nullopt;

    OutputConfig outConfig;

    auto executorConfig = ExecutorConfig(beamWidth);
    std::filesystem::path modelPath;
    if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1")
    {
        if (modelName == "llama_tp4_pp1_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
            deviceIds = std::vector<SizeType32>{3, 2, 1, 0};
        }
        else if (modelName == "llama_tp2_pp2_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
            deviceIds = std::vector<SizeType32>{2, 3, 0, 1};
        }
    }

    // For llama model, only run for multiple GPUs
    // This is detected by setting an env variable when running the test
    char const* val = getenv("RUN_LLAMA_MULTI_GPU");
    if (val == NULL)
    {
        GTEST_SKIP() << "Skipping Llama test";
    }
    else
    {
        // Check that it was launched with right number of MPI ranks
        if (!useOrchestratorMode && COMM_SESSION.getSize() != 4)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Leader mode and world size is not equal to 4";
        }
        else if (useOrchestratorMode && COMM_SESSION.getSize() != 1)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Orchestrator mode and World size is not equal to 1";
        }
    }

    if (useOrchestratorMode)
    {
        auto orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
        auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
            useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt,
            std::nullopt, orchestratorConfig);
        if (deviceIds.has_value())
        {
            parallelConfig.setDeviceIds(deviceIds.value());
        }
        executorConfig.setParallelConfig(parallelConfig);
    }
    else
    {
        if (deviceIds.has_value())
        {
            auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());
            parallelConfig.setDeviceIds(deviceIds.value());
            executorConfig.setParallelConfig(parallelConfig);
        }
    }

    auto executor = Executor(modelPath, ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 50;
    VecTokens inputTokens{1, 2, 3, 4};

    std::vector<Request> requests;
    for (auto streaming : {false, true})
    {
        // Add two requests with numReturnSequences = 1
        auto samplingConfig = tensorrt_llm::executor::SamplingConfig(beamWidth);
        requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outConfig);
        requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig, outConfig);
        // Add a request with numReturnSequences > 1
        auto samplingConfig2 = tensorrt_llm::executor::SamplingConfig(beamWidth);
        auto constexpr numReturnSequences = 2;
        samplingConfig2.setNumReturnSequences(numReturnSequences);
        requests.emplace_back(inputTokens, maxNewTokens, streaming, samplingConfig2, outConfig);
    }
    std::vector<bool> cancelRequests{true, false, true, true, false, true};

    if (executor.canEnqueueRequests())
    {
        auto const requestIds = executor.enqueueRequests(requests);

        // Cancel the first and third requests
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for (SizeType32 i = 0; i < requests.size(); i++)
        {
            if (cancelRequests.at(i))
            {
                executor.cancelRequest(requestIds.at(i));
            }
        }

        std::unordered_map<IdType, bool> isStreaming;
        std::unordered_map<IdType, SizeType32> expectedNumTokens;
        SizeType32 expectedNumResponses = 0;
        for (SizeType32 i = 0; i < requests.size(); i++)
        {
            auto const& request = requests.at(i);
            auto requestId = requestIds.at(i);
            isStreaming[requestId] = request.getStreaming();
            expectedNumTokens[requestId] = (request.getStreaming() ? 0 : inputTokens.size()) + maxNewTokens;
            auto const numResponses = request.getStreaming() ? expectedNumTokens[requestId] : 1;
            auto const numReturnSequences = request.getSamplingConfig().getBeamWidth() > 1
                ? 1
                : request.getSamplingConfig().getNumReturnSequences().value_or(1);
            expectedNumResponses += numResponses * numReturnSequences;
        }

        std::unordered_map<IdType, std::unordered_map<SizeType32, VecTokens>> tokens;

        // Get the new tokens for each requests
        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < requests.size() && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto requestId = response.getRequestId();
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    auto seqIdx = result.sequenceIndex;
                    auto numSequences = result.outputTokenIds.size();
                    auto& newTokens = result.outputTokenIds.at(numSequences - 1);
                    auto& reqResults = tokens[response.getRequestId()];
                    auto& reqTokens = reqResults[seqIdx];
                    if (isStreaming.at(requestId) && beamWidth > 1)
                    {
                        reqTokens = newTokens;
                    }
                    else
                    {
                        reqTokens.insert(reqTokens.end(), newTokens.begin(), newTokens.end());
                    }
                }
                else
                {
                    FAIL() << "Did not expect errors";
                }
            }
            ++iter;
        }

        EXPECT_LE(numResponses, expectedNumResponses);
        EXPECT_EQ(numFinished, requests.size());
        EXPECT_LT(iter, mMaxWaitMs);

        for (auto requestIdx = 0; requestIdx < requests.size(); requestIdx++)
        {
            auto const requestId = requestIds.at(requestIdx);
            for (auto seqIdx = 0; seqIdx < tokens.at(requestId).size(); seqIdx++)
            {
                auto const& seqTokens = tokens.at(requestId).at(seqIdx);
                if (cancelRequests.at(requestIdx))
                {
                    EXPECT_LT(seqTokens.size(), expectedNumTokens.at(requestId));
                }
                else
                {
                    EXPECT_EQ(seqTokens.size(), expectedNumTokens.at(requestId));
                }
            }
        }
    }
}

TEST_P(LeaderApiUsageTest, LeaderModeTest)
{
    auto const modelName = std::get<0>(GetParam());

    SizeType32 beamWidth = 2;
    OutputConfig outConfig;
    std::optional<std::vector<SizeType32>> deviceIds = std::nullopt;

    auto executorConfig = ExecutorConfig(beamWidth);
    std::filesystem::path modelPath;
    if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1")
    {
        if (modelName == "llama_tp4_pp1_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
            deviceIds = std::vector<SizeType32>{3, 2, 1, 0};
        }
        else if (modelName == "llama_tp2_pp2_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
            deviceIds = std::vector<SizeType32>{2, 3, 0, 1};
        }
    }

    // For llama model, only run for multiple GPUs
    // This is detected by setting an env variable when running the test
    char const* val = getenv("RUN_LLAMA_MULTI_GPU");
    if (val == NULL)
    {
        GTEST_SKIP() << "Skipping Llama test";
    }
    else
    {
        // Check that it was launched with right number of MPI ranks
        if (COMM_SESSION.getSize() != 4)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Leader mode and world size is not equal to 4";
        }
    }

    if (deviceIds.has_value())
    {
        auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());

        parallelConfig.setDeviceIds(deviceIds.value());
        executorConfig.setParallelConfig(parallelConfig);
    }
    auto executor = Executor(modelPath, ModelType::kDECODER_ONLY, executorConfig);

    // Since this is leader mode, all ranks should participate
    EXPECT_TRUE(executor.isParticipant());

    // Create the request
    SizeType32 maxNewTokens = 50;
    VecTokens inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);
    auto requestStreaming
        = Request(inputTokens, maxNewTokens, true, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    // Leader enqueues requests and wait for responses
    if (executor.canEnqueueRequests())
    {
        auto requestId = executor.enqueueRequest(request);
        auto requestId2 = executor.enqueueRequest(request);
        auto requestId3 = executor.enqueueRequest(requestStreaming);
        auto requestId4 = executor.enqueueRequest(requestStreaming);

        int32_t numFinished = 0;
        int iter = 0;
        SizeType32 numResponses = 0;
        while (numFinished < 4 && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                numResponses++;
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                }
                else
                {
                    FAIL() << "Did not expect errors";
                }
            }
            ++iter;
        }
        EXPECT_EQ(numFinished, 4);
        EXPECT_LT(iter, mMaxWaitMs);
    }
    else
    {
        // Check that non-leader cannot enqueue requests
        EXPECT_THROW({ auto reqId = executor.enqueueRequest(request); }, tensorrt_llm::common::TllmException);
        EXPECT_THROW({ auto responses = executor.awaitResponses(); }, tensorrt_llm::common::TllmException);
        EXPECT_THROW({ auto numResp = executor.getNumResponsesReady(); }, tensorrt_llm::common::TllmException);
        EXPECT_THROW({ executor.cancelRequest(1); }, tensorrt_llm::common::TllmException);
        EXPECT_THROW({ auto stats = executor.getLatestIterationStats(); }, tensorrt_llm::common::TllmException);
        EXPECT_THROW({ auto stats = executor.getLatestRequestStats(); }, tensorrt_llm::common::TllmException);
    }
}

TEST_F(GptExecutorTest, validateParallelConfig)
{

    auto trtEnginePath = (GPT_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu");
    {
        auto executorConfig = ExecutorConfig();
        auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
    }

    {
        std::string expectedErrMsg = "OrchestratorConfig must be set";
        try
        {
            auto executorConfig = ExecutorConfig();
            auto parallelConfig = ParallelConfig(CommunicationType::kMPI, CommunicationMode::kORCHESTRATOR);
            executorConfig.setParallelConfig(parallelConfig);
            auto executor = Executor(trtEnginePath, ModelType::kDECODER_ONLY, executorConfig);
            FAIL() << "Expected TllmException";
        }
        catch (tc::TllmException& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr(expectedErrMsg));
        }
        catch (std::exception const& e)
        {
            FAIL() << "Expected TllmException";
        }
    }
}

TEST_P(TimeoutTest, TimeoutStreamingTest)
{
    auto const modelName = std::get<0>(GetParam());
    auto const useOrchestratorMode = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());

    auto executorConfig = ExecutorConfig(beamWidth);
    std::filesystem::path modelPath;
    bool isMultiGpu{false};
    std::optional<std::vector<SizeType32>> deviceIds = std::nullopt;

    if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1")
    {
        isMultiGpu = true;
        if (modelName == "llama_tp4_pp1_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
            deviceIds = std::vector<SizeType32>{3, 2, 1, 0};
        }
        else if (modelName == "llama_tp2_pp2_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
            deviceIds = std::vector<SizeType32>{2, 3, 0, 1};
        }
    }
    if (modelName == "llama_tp1_pp1_cp1")
    {
        modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    }
    // For llama model, only run for multiple GPUs
    // This is detected by setting an env variable when running the test
    char const* val = getenv("RUN_LLAMA_MULTI_GPU");
    if (val == NULL && isMultiGpu)
    {
        GTEST_SKIP() << "Skipping MultiGpu tests";
    }
    if (val != NULL && !isMultiGpu)
    {
        GTEST_SKIP() << "Skipping SingleGpu tests";
    }
    if (val != NULL && isMultiGpu)
    {
        // Check that it was launched with right number of MPI ranks
        if (!useOrchestratorMode && COMM_SESSION.getSize() != 4)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Leader mode and world size is not equal to 4";
        }
        if (useOrchestratorMode && COMM_SESSION.getSize() != 1)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Orchestrator mode and World size is not equal to 1";
        }
    }

    if (useOrchestratorMode)
    {
        auto orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
        auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
            useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt,
            std::nullopt, orchestratorConfig);
        executorConfig.setParallelConfig(parallelConfig);
        if (deviceIds.has_value())
        {
            parallelConfig.setDeviceIds(deviceIds.value());
        }
        executorConfig.setParallelConfig(parallelConfig);
    }
    else
    {
        if (deviceIds.has_value())
        {
            auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());
            parallelConfig.setDeviceIds(deviceIds.value());
            executorConfig.setParallelConfig(parallelConfig);
        }
    }
    auto executor = Executor(modelPath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 constexpr maxNewTokens = 10;
    // create 1 request that times out immediately
    // momentarily we don't cancel requests before forwardAsync so it will get scheduled for at least 1 forward
    VecTokens immediateCancelTokens{1, 2, 3, 4};
    auto immediateCancelRequest
        = Request(immediateCancelTokens, maxNewTokens, true, tensorrt_llm::executor::SamplingConfig(beamWidth));
    immediateCancelRequest.setReturnAllGeneratedTokens(true);
    immediateCancelRequest.setAllottedTimeMs(std::chrono::milliseconds(0));
    SizeType32 constexpr immediateCancelMinLength = 0;
    SizeType32 constexpr immediateCancelMaxLength = 1;

    // create 1 request that times out during the first forward
    VecTokens oneForwardTokens{11, 12, 13, 14};
    auto oneForwardRequest
        = Request(oneForwardTokens, maxNewTokens, true, tensorrt_llm::executor::SamplingConfig(beamWidth));
    oneForwardRequest.setReturnAllGeneratedTokens(true);
    oneForwardRequest.setAllottedTimeMs(std::chrono::milliseconds(1));
    SizeType32 constexpr oneForwardlMinLength = 0;
    SizeType32 constexpr oneForwardlMaxLength = 1;

    // Create the request that finishes by the number of tokens
    VecTokens finishedTokens{101, 102, 103, 104};
    auto finishedRequest
        = Request(finishedTokens, maxNewTokens, true, tensorrt_llm::executor::SamplingConfig(beamWidth));
    finishedRequest.setReturnAllGeneratedTokens(true);
    finishedRequest.setAllottedTimeMs(std::chrono::milliseconds(5000));
    SizeType32 constexpr finishedMinLength = 5;
    SizeType32 constexpr finishedMaxLength = maxNewTokens;

    std::vector<FinishReason> referenceFinishReasons
        = {FinishReason::kTIMED_OUT, FinishReason::kTIMED_OUT, FinishReason::kLENGTH};
    std::vector<SizeType32> minLengths = {immediateCancelMinLength, oneForwardlMinLength, finishedMinLength};
    std::vector<SizeType32> maxLengths = {immediateCancelMaxLength, oneForwardlMaxLength, finishedMaxLength};
    // workaround because the last response will be empty, but we want to have at least *some* responses surpass the
    // minLength
    std::vector<SizeType32> achievedLength = {0, 0, 0};
    SizeType32 itNr{0};

    if (executor.canEnqueueRequests())
    {

        std::vector<Request> requests = {immediateCancelRequest, oneForwardRequest, finishedRequest};
        auto requestIds = executor.enqueueRequests(requests);

        auto numFinished = 0;

        while (numFinished < static_cast<SizeType32>(requests.size()))
        {
            itNr++;
            std::chrono::milliseconds waitTime(mMaxWaitMs);
            auto responses = executor.awaitResponses(requestIds, waitTime);
            for (auto const& response : responses)
            {
                for (auto const& responseIt : response)
                {
                    auto const reqId = responseIt.getRequestId();
                    if (responseIt.hasError())
                    {
                        // Allow response with error only if awaitResponse processed a terminated request id
                        std::string err
                            = "ReqId " + std::to_string(reqId) + " has already been processed and was terminated.";
                        if (responseIt.getErrorMsg() != err)
                        {
                            TLLM_THROW("Request id %lu encountered error: %s", reqId, responseIt.getErrorMsg().c_str());
                        }
                        continue;
                    }

                    auto const& result = responseIt.getResult();
                    if (result.isFinal)
                    {
                        requestIds.erase(std::remove(requestIds.begin(), requestIds.end(), reqId), requestIds.end());
                        numFinished++;
                    }

                    auto const finishReason = result.finishReasons;
                    auto const actualResponse = result.outputTokenIds;
                    TLLM_LOG_DEBUG("reqId %d finished %d", reqId, result.isFinal);
                    TLLM_LOG_DEBUG("actual response:");

                    for (auto const& beam : actualResponse)
                    {
                        std::string tokenStr;
                        for (auto tok : beam)
                        {
                            tokenStr += std::to_string(tok) + " ";
                        }
                        TLLM_LOG_DEBUG("%s", tokenStr.c_str());
                    }

                    TLLM_LOG_DEBUG(
                        "beams' length must be in range [%d, %d]", minLengths[reqId - 1], maxLengths[reqId - 1]);

                    if (result.isFinal)
                    {
                        TLLM_LOG_DEBUG("finishReason");
                        std::string reasonStr;
                        for (auto const reason : finishReason)
                        {
                            // cast for easier visibility during debugging
                            EXPECT_EQ(static_cast<int>(reason), static_cast<int>(referenceFinishReasons[reqId - 1]));
                            reasonStr += std::to_string(static_cast<int>(reason)) + " ";
                        }
                        TLLM_LOG_DEBUG("%s", reasonStr.c_str());
                    }

                    EXPECT_EQ(beamWidth, actualResponse.size());
                    for (int beam = 0; beam < beamWidth; beam++)
                    {
                        EXPECT_LE(actualResponse.at(beam).size(), maxLengths[reqId - 1]) << "for request " << reqId;
                        achievedLength[reqId - 1] = std::max(
                            achievedLength[reqId - 1], static_cast<SizeType32>(actualResponse.at(beam).size()));
                    }
                }
            }
        }

        for (int reqIt = 0; reqIt < achievedLength.size(); ++reqIt)
        {
            EXPECT_GE(achievedLength[reqIt], minLengths[reqIt])
                << "request " << reqIt + 1 << " has not achieved min lengths";
        }
    }
}

TEST_P(TimeoutTest, TimeoutNonstreamingTest)
{
    auto const modelName = std::get<0>(GetParam());
    auto const useOrchestratorMode = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());

    std::optional<std::vector<SizeType32>> deviceIds = std::nullopt;

    auto executorConfig = ExecutorConfig(beamWidth);
    std::filesystem::path modelPath;
    bool isMultiGpu{false};
    if (modelName == "llama_tp4_pp1_cp1" || modelName == "llama_tp1_pp4_cp1" || modelName == "llama_tp2_pp2_cp1")
    {
        isMultiGpu = true;
        if (modelName == "llama_tp4_pp1_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp4-pp1-cp1-gpu";
        }
        else if (modelName == "llama_tp1_pp4_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp4-cp1-gpu";
            deviceIds = std::vector<SizeType32>{3, 2, 1, 0};
        }
        else if (modelName == "llama_tp2_pp2_cp1")
        {
            modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp2-pp2-cp1-gpu";
            deviceIds = std::vector<SizeType32>{2, 3, 0, 1};
        }
    }
    if (modelName == "llama_tp1_pp1_cp1")
    {
        modelPath = LLAMA_MODEL_PATH / PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR() / "tp1-pp1-cp1-gpu";
    }
    // For llama model, only run for multiple GPUs
    // This is detected by setting an env variable when running the test
    char const* val = getenv("RUN_LLAMA_MULTI_GPU");
    if (val == NULL && isMultiGpu)
    {
        GTEST_SKIP() << "Skipping MultiGpu tests";
    }
    if (val != NULL && !isMultiGpu)
    {
        GTEST_SKIP() << "Skipping SingleGpu tests";
    }
    if (val != NULL && isMultiGpu)
    {
        // Check that it was launched with right number of MPI ranks
        if (!useOrchestratorMode && COMM_SESSION.getSize() != 4)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Leader mode and world size is not equal to 4";
        }
        if (useOrchestratorMode && COMM_SESSION.getSize() != 1)
        {
            // No orchestrator, need worldSize to match TP*PP
            FAIL() << "Orchestrator mode and World size is not equal to 1";
        }
    }

    if (useOrchestratorMode)
    {
        auto orchestratorConfig = OrchestratorConfig(true, PathUtil::EXECUTOR_WORKER_PATH());
        auto parallelConfig = ParallelConfig(CommunicationType::kMPI,
            useOrchestratorMode ? CommunicationMode::kORCHESTRATOR : CommunicationMode::kLEADER, std::nullopt,
            std::nullopt, orchestratorConfig);
        executorConfig.setParallelConfig(parallelConfig);
        if (deviceIds.has_value())
        {
            parallelConfig.setDeviceIds(deviceIds.value());
        }
        executorConfig.setParallelConfig(parallelConfig);
    }
    else
    {
        if (deviceIds.has_value())
        {
            auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());
            parallelConfig.setDeviceIds(deviceIds.value());
            executorConfig.setParallelConfig(parallelConfig);
        }
    }
    auto executor = Executor(modelPath, ModelType::kDECODER_ONLY, executorConfig);

    SizeType32 constexpr maxNewTokens = 5;
    // create 1 request that times out immediately
    // momentarily we don't cancel requests before forwardAsync so it will get scheduled for at least 1 forward
    VecTokens immediateCancelTokens{1, 2, 3, 4};
    auto immediateCancelRequest
        = Request(immediateCancelTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    immediateCancelRequest.setAllottedTimeMs(std::chrono::milliseconds(0));
    std::vector<std::vector<int>> immediateCancelResponse = {immediateCancelTokens, immediateCancelTokens};

    // create 1 request that times out during the first forward
    VecTokens oneForwardTokens{11, 12, 13, 14};
    auto oneForwardRequest
        = Request(oneForwardTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    oneForwardRequest.setAllottedTimeMs(std::chrono::milliseconds(1));
    std::vector<std::vector<int>> oneForwardResponse = {oneForwardTokens, oneForwardTokens};

    // Create the request that finishes by the number of tokens
    VecTokens finishedTokens{101, 102, 103, 104};
    auto finishedRequest
        = Request(finishedTokens, maxNewTokens, false, tensorrt_llm::executor::SamplingConfig(beamWidth));
    finishedRequest.setAllottedTimeMs(std::chrono::milliseconds(6000));
    std::vector<std::vector<int>> finishedReponse
        = {{101, 102, 103, 104, 49849, 225, 49849, 232, 55742}, {101, 102, 103, 104, 49849, 225, 49849, 232, 29082}};

    // assume responses will come in FIFO order
    std::vector<BeamTokens> refResponses = {immediateCancelResponse, oneForwardResponse, finishedReponse};
    std::vector<FinishReason> referenceFinishReasons
        = {FinishReason::kTIMED_OUT, FinishReason::kTIMED_OUT, FinishReason::kLENGTH};
    if (executor.canEnqueueRequests())
    {

        std::vector<Request> requests = {immediateCancelRequest, oneForwardRequest, finishedRequest};
        auto requestIds = executor.enqueueRequests(requests);

        std::chrono::milliseconds waitTime(mMaxWaitMs);
        auto responses = executor.awaitResponses(requestIds, waitTime);
        for (auto const& response : responses)
        {
            for (auto const& responseIt : response)
            {
                auto const reqId = responseIt.getRequestId();
                if (responseIt.hasError())
                {
                    TLLM_THROW("Request id %lu encountered error: %s", reqId, responseIt.getErrorMsg().c_str());
                }

                auto const& result = responseIt.getResult();

                auto const finishReason = result.finishReasons;
                auto const actualResponse = result.outputTokenIds;
                TLLM_LOG_DEBUG("reqId %d finished %d", reqId, result.isFinal);
                TLLM_LOG_DEBUG("actual response:");

                for (auto const& beam : actualResponse)
                {
                    std::string tokenStr;
                    for (auto tok : beam)
                    {
                        tokenStr += std::to_string(tok) + " ";
                    }
                    TLLM_LOG_DEBUG("%s", tokenStr.c_str());
                }

                TLLM_LOG_DEBUG("reference:");
                auto referenceResponse = refResponses[reqId - 1];
                for (auto const& beam : referenceResponse)
                {
                    std::string tokenStr;
                    for (auto tok : beam)
                    {
                        tokenStr += std::to_string(tok) + " ";
                    }
                    TLLM_LOG_DEBUG("%s", tokenStr.c_str());
                }

                if (result.isFinal)
                {
                    TLLM_LOG_DEBUG("finishReason");
                    std::string reasonStr;
                    for (auto const reason : finishReason)
                    {
                        // cast for easier visibility during debugging
                        EXPECT_EQ(static_cast<int>(reason), static_cast<int>(referenceFinishReasons[reqId - 1]));
                        reasonStr += std::to_string(static_cast<int>(reason)) + " ";
                    }
                    TLLM_LOG_DEBUG("%s", reasonStr.c_str());
                }

                EXPECT_EQ(beamWidth, actualResponse.size());
                for (int beam = 0; beam < beamWidth; beam++)
                {
                    EXPECT_EQ(referenceResponse.at(beam).size(), actualResponse.at(beam).size());
                    EXPECT_THAT(actualResponse.at(beam), testing::ElementsAreArray(referenceResponse.at(beam)));
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(GptExecutorTest, ParamTest,
    testing::Combine(                 //
        testing::Values(false, true), // streaming
        testing::Values(false, true), // excludeInputFromOutput
        testing::Values(1, 2)         // beamWidth
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptExecutorTest, ParamStatsTest,
    testing::Combine(                //
        testing::Values(0, 1000),    // iterStatsMaxIterations
        testing::Values(false, true) // useOrchestratorMode
        ),
    generateTestNameStats);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorTest, ParamCancelReqTest,
    testing::Combine(                                                                  //
        testing::Values(false, true),                                                  // useOrchestratorMode
        testing::Values(1, 2),                                                         // beamWidth
        testing::Values("llama_tp1_pp4_cp1", "llama_tp4_pp1_cp1", "llama_tp2_pp2_cp1") // modelName
        ),
    generateTestNameCancelReq);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorTest, TimeoutTest,
    testing::Combine(                                                                   //
        testing::Values("llama_tp1_pp4_cp1", "llama_tp4_pp1_cp1", "llama_tp1_pp1_cp1"), // modelName
        testing::Values(false, true),                                                   // useOrchestratorMode
        testing::Values(2)                                                              // beamWidth
        ),
    generateTestNameTimeoutTest);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorTest, LeaderApiUsageTest,
    testing::Combine(                                                                  //
        testing::Values("llama_tp1_pp4_cp1", "llama_tp4_pp1_cp1", "llama_tp2_pp2_cp1") // modelName
        ),
    generateTestNameLeaderApiUsage);

INSTANTIATE_TEST_SUITE_P(GptExecutorTest, AllParamsTest,
    testing::Combine(                 //
        testing::Values(false, true), // streaming
        testing::Values(1, 2),        // beamWidth
        testing::Values(true),        // computeLogProbs
        testing::Values(false, true), // excludeInputInOutput
        testing::Values(true),        // returnContextLogits
        testing::Values(true),        // returnGenerationLogits
        testing::Values("gpt"),       // modelName
        testing::Values(false, true), // useOrchestratorMode
        testing::Values(false, true), // returnAllGeneratedTokens
        testing::Values(1, 2)         // numReturnSequences
        ),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorTest, AllParamsTest,
    testing::Combine(                                                                   //
        testing::Values(false, true),                                                   // streaming
        testing::Values(1, 2),                                                          // beamWidth
        testing::Values(true),                                                          // computeLogProbs
        testing::Values(false, true),                                                   // excludeInputInOutput
        testing::Values(false),                                                         // returnContextLogits
        testing::Values(true),                                                          // returnGenerationLogits
        testing::Values("llama_tp1_pp4_cp1", "llama_tp4_pp1_cp1", "llama_tp2_pp2_cp1"), // modelName
        testing::Values(false, true),                                                   // useOrchestratorMode
        testing::Values(false),                                                         // returnAllGeneratedTokens
        testing::Values(1)                                                              // numReturnSequences
        ),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(LlamaMultiExecutorTest, AllParamsTest,
    testing::Combine(                         //
        testing::Values(false, true),         // streaming
        testing::Values(1, 2),                // beamWidth
        testing::Values(false),               // computeLogProbs
        testing::Values(false, true),         // excludeInputInOutput
        testing::Values(false),               // returnContextLogits
        testing::Values(false),               // returnGenerationLogits
        testing::Values("llama_tp1_pp2_cp1"), // modelName
        testing::Values(false),               // useOrchestratorMode
        testing::Values(false),               // returnAllGeneratedTokens
        testing::Values(1)                    // numReturnSequences
        ),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(MedusaExecutorTest, AllParamsTest,
    testing::Combine(                 //
        testing::Values(false, true), // streaming
        testing::Values(1),           // beamWidth
        testing::Values(false),       // computeLogProbs
        testing::Values(false, true), // excludeInputInOutput
        testing::Values(false),       // returnContextLogits
        testing::Values(false),       // returnGenerationLogits
        testing::Values("medusa"),    // modelName
        testing::Values(false, true), // useOrchestratorMode
        testing::Values(false),       // returnAllGeneratedTokens
        testing::Values(1)            // numReturnSequences
        ),
    generateTestNameAllParams);

// Disable some of ChatGLM's tests since they are the same as gpt's.
INSTANTIATE_TEST_SUITE_P(ChatGlmExecutorTest, AllParamsTest,
    testing::Combine(               //
        testing::Values(false),     // streaming
        testing::Values(1, 2),      // beamWidth
        testing::Values(false),     // computeLogProbs
        testing::Values(false),     // excludeInputInOutput
        testing::Values(false),     // returnContextLogits
        testing::Values(false),     // returnGenerationLogits
        testing::Values("chatglm"), // modelName
        testing::Values(false),     // useOrchestratorMode
        testing::Values(false),     // returnAllGeneratedTokens
        testing::Values(1, 2)       // numReturnSequences
        ),
    generateTestNameAllParams);

// ChatGlm0 Test is for glm-10b.
INSTANTIATE_TEST_SUITE_P(ChatGlm0ExecutorTest, AllParamsTest,
    testing::Combine(           //
        testing::Values(false), // streaming
        testing::Values(1),     // beamWidth
        testing::Values(false), // computeLogProbs
        testing::Values(false), // excludeInputInOutput
        testing::Values(false), // returnContextLogits
        testing::Values(false), // returnGenerationLogits
        testing::Values("glm"), // modelName
        testing::Values(false), // useOrchestratorMode
        testing::Values(false), // returnAllGeneratedTokens
        testing::Values(1)      // numReturnSequences
        ),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(ChatGlm2ExecutorTest, AllParamsTest,
    testing::Combine(                //
        testing::Values(false),      // streaming
        testing::Values(1),          // beamWidth
        testing::Values(false),      // computeLogProbs
        testing::Values(false),      // excludeInputInOutput
        testing::Values(false),      // returnContextLogits
        testing::Values(false),      // returnGenerationLogits
        testing::Values("chatglm2"), // modelName
        testing::Values(false),      // useOrchestratorMode
        testing::Values(false),      // returnAllGeneratedTokens
        testing::Values(1)           // numReturnSequences
        ),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(ChatGlm3ExecutorTest, AllParamsTest,
    testing::Combine(                //
        testing::Values(false),      // streaming
        testing::Values(1),          // beamWidth
        testing::Values(false),      // computeLogProbs
        testing::Values(false),      // excludeInputInOutput
        testing::Values(false),      // returnContextLogits
        testing::Values(false),      // returnGenerationLogits
        testing::Values("chatglm3"), // modelName
        testing::Values(false),      // useOrchestratorMode
        testing::Values(false),      // returnAllGeneratedTokens
        testing::Values(1)           // numReturnSequences
        ),
    generateTestNameAllParams);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorTest, LogitsProcParamsTest,
    testing::Combine(                                                                            //
        testing::Values(
            "llama_tp1_pp1_cp1", "llama_tp4_pp1_cp1", "llama_tp2_pp2_cp1", "llama_tp1_pp4_cp1"), // modelName
        testing::Values(false, true),                                                            // batched
        testing::Values(false, true)                                                             // replicated
        ),
    generateTestNameLogitsProc);

INSTANTIATE_TEST_SUITE_P(GptExecutorGuidedDecodingTest, GuidedDecodingParamsTest,
    testing::Combine(testing::Values("gpt")), generateTestNameGuidedDecoding);

INSTANTIATE_TEST_SUITE_P(LlamaExecutorGuidedDecodingTest, GuidedDecodingParamsTest,
    testing::Combine(
        testing::Values("llama_tp1_pp1_cp1", "llama_tp4_pp1_cp1", "llama_tp2_pp2_cp1", "llama_tp1_pp4_cp1")),
    generateTestNameGuidedDecoding);
