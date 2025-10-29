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

#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/batch_manager/trtGptModelFactory.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/testing/modelSpec.h"
#include "tests/utils/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

using namespace tensorrt_llm::testing;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::runtime::utils;
using namespace tensorrt_llm::batch_manager;
namespace fs = std::filesystem;
namespace tc = tensorrt_llm::common;
namespace texec = tensorrt_llm::executor;
using tensorrt_llm::testing::ModelSpec;
using tensorrt_llm::testing::KVCacheType;
using tensorrt_llm::testing::QuantMethod;

namespace
{
using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

auto constexpr GPT_MODEL_DIR = "gpt2";
auto constexpr GPTJ_MODEL_DIR = "gpt-j-6b";
auto constexpr LLAMA_MODEL_DIR = "Llama-3.2-1B";
auto constexpr MEDUSA_MODEL_DIR = "vicuna-7b-medusa";
auto constexpr EAGLE_MODEL_DIR = "vicuna-7b-eagle";
auto constexpr MAMBA_MODEL_DIR = "mamba-2.8b-hf";
auto constexpr RECURRENTGEMMA_MODEL_DIR = "recurrentgemma-2b";
auto constexpr EXPLICIT_DRAFT_MODEL_DIR = "vicuna-7b-redrafter";
auto constexpr CHATGLM_MODEL_DIR = "chatglm-6b";
auto constexpr GLM_MODEL_DIR = "glm-10b";

auto constexpr FP8_GPT_ATTENTION_PLUGIN_IFB_PACKED_PATH = "fp8-plugin";

auto constexpr INPUT_FILE = "input_tokens.npy";
auto constexpr INPUT_LLAMA_FILE = "input_tokens_llama.npy";
auto constexpr INPUT_VICUNA_FILE = "input_vicuna.npy";
auto constexpr LONG_INPUT_FILE = "input_tokens_long.npy";
auto constexpr CHATGLM_INPUT_FILE = "input_tokens_chatglm-6b.npy";
auto constexpr GLM_INPUT_FILE = "input_tokens_glm-10b.npy";

auto constexpr LLAMA_END_ID = 128001;
auto constexpr LLAMA_PAD_ID = 128001;

struct ModelParams
{
    char const* baseDir;
    ModelIds ids;

    friend std::ostream& operator<<(std::ostream& os, ModelParams const& modelParams)
    {
        return os << "baseDir: " << modelParams.baseDir << ", ids: (" << modelParams.ids.padId << ","
                  << modelParams.ids.endId << ")";
    }
};

} // namespace

class TrtModelRealDecoderTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    TrtModelRealDecoderTest() {}

    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount == 0)
        {
            GTEST_SKIP() << "No GPUs found";
        }

        mLogger = std::make_shared<TllmLogger>();

        initTrtLlmPlugins(mLogger.get());
    }

    void TearDown() override {}

    int mDeviceCount{};
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

enum class TrtGptModelIfbTestType
{
    BULK,
    WAVEFRONT,
    RANDOM
};

namespace
{

void verifyOutput(RequestList const& finishedRequestList,
    std::unordered_map<SizeType32, TestData> const& beamWidthTestData, std::vector<SizeType32> const& givenInputLengths,
    SizeType32 nbGivenInputs, ModelSpec const& modelSpec)
{
    auto const checkRawLogits = modelSpec.mOtherModelSpecToCompare ? false : modelSpec.mGatherLogits;
    auto const smokeTest = modelSpec.mSmokeTest;
    auto const returnLogProbs = modelSpec.mReturnLogProbs;
    auto const checkAcceptedTokenLogits = modelSpec.mAcceptDraftByLogits;

    if (smokeTest)
    {
        return;
    }

    for (auto const& llmReqPtr : finishedRequestList)
    {
        auto const& llmReq = *llmReqPtr;
        auto const requestId = llmReq.mRequestId;
        auto const [givenInputIdx, givenInputLength]
            = getRequestGivenInputIdxLength(requestId, nbGivenInputs, givenInputLengths);
        auto const reqBeamWidth = llmReq.mSamplingConfig.beamWidth;
        auto const& testData = beamWidthTestData.at(reqBeamWidth);
        auto const* const expectedOutputData = bufferCast<TokenIdType const>(*testData.expectedOutputIds);
        auto const expectedOutputLengths = testData.expectedOutputLengths;
        auto const acceptedDraftTokensLengths = testData.acceptedDraftTokensLengths;
        auto const endId = testData.endIds[givenInputIdx];
        auto const maxSeqLen = testData.maxSeqLen;
        auto const draftLogits = testData.draftLogits;
        auto const expectedGenerationLogits = testData.expectedGenerationLogits;
        auto const expectedContextLogits = testData.expectedContextLogits;
        auto const expectedCumLogProbs = testData.expectedCumLogProbs;
        auto const expectedLogProbs = testData.expectedLogProbs;
        auto const draftTokens = llmReq.getDraftTokens();
        auto const isDraftTokensExternal = modelSpec.mSpecDecodingMode.isDraftTokensExternal();
        auto const inputLength = givenInputLength + static_cast<SizeType32>(isDraftTokensExternal);

        for (auto beam = 0; beam < reqBeamWidth; ++beam)
        {
            auto const expectedOutputLength = expectedOutputLengths[givenInputIdx * reqBeamWidth + beam];
            auto const predictedTokens = llmReq.getTokens(beam);

            auto numPredTokens = static_cast<SizeType32>(predictedTokens.size() - inputLength);
            if (isDraftTokensExternal && !draftTokens->empty())
            {
                numPredTokens
                    = std::min(numPredTokens, acceptedDraftTokensLengths[givenInputIdx * reqBeamWidth + beam] + 1);
            }
            if (modelSpec.mSpecDecodingMode.isMedusa() || modelSpec.mSpecDecodingMode.isLookaheadDecoding()
                || modelSpec.mSpecDecodingMode.isExplicitDraftTokens() || modelSpec.mSpecDecodingMode.isEagle())
            {
                // WAR to ensure bulk execution of spec decoding.
                // We hope that no request in batch can finish 2x faster than any other request.
                // For the cases when BS < 8, some predicted tokens are mismatched to reference data.
                numPredTokens /= 2;
            }

            if (modelSpec.mKVCacheType == KVCacheType::kDISABLED)
            {
                EXPECT_EQ(numPredTokens, 1) << "b: " << requestId << " beam: " << beam;
            }
            else
            {
                EXPECT_EQ(predictedTokens.size(), expectedOutputLength) << "b: " << requestId << " beam: " << beam;
            }

            bool anyMismatch = false;
            for (auto i = 0; i < numPredTokens; ++i)
            {
                // Use the expected data for that beamWidth
                auto const expectIndex = tc::flat_index3(givenInputIdx, beam, inputLength + i, reqBeamWidth, maxSeqLen);

                auto const expectedToken = expectedOutputData[expectIndex];
                if (expectedToken == endId)
                {
                    break;
                }
                auto const predictIndex = inputLength + i;
                auto const predictedToken = predictedTokens.at(predictIndex);
                EXPECT_EQ(predictedToken, expectedToken) << "b: " << requestId << " beam: " << beam << " i: " << i;
                anyMismatch |= (predictedToken != expectedToken);
            }
            EXPECT_FALSE(anyMismatch) << "b: " << requestId << " beam: " << beam;

            if (returnLogProbs)
            {
                auto cumLogProbs = llmReq.getCumLogProbs();
                auto* const reqExpectedCumLogProbs = bufferCast<float>(*expectedCumLogProbs[requestId]);
                EXPECT_TRUE(almostEqual(reqExpectedCumLogProbs[beam], cumLogProbs[beam]));

                auto logProbs = llmReq.getLogProbs(beam);
                auto expectedLogProbsBeam = std::shared_ptr(ITensor::slice(expectedLogProbs[requestId], beam, 1));
                expectedLogProbsBeam->squeeze(0);
                auto* const reqExpectedLogProbs = bufferCast<float>(*expectedLogProbsBeam);

                for (auto i = 0; i < numPredTokens; ++i)
                {
                    EXPECT_TRUE(almostEqual(reqExpectedLogProbs[inputLength + i], logProbs[i], 5e-2, 5e-2))
                        << "expectedLogProbs : " << reqExpectedLogProbs[inputLength + i]
                        << " logProbs : " << logProbs[i];
                }
            }

            if (checkAcceptedTokenLogits && llmReq.hasDraftTokens())
            {
                TLLM_CHECK_WITH_INFO(reqBeamWidth == 1, "speculative decoding only works for beam width == 1");

                TensorPtr const& acceptedTokensLogits = llmReq.getGenerationLogitsHost();
                auto const acceptedTokensLogitsShape = acceptedTokensLogits->getShape();

                EXPECT_EQ(acceptedTokensLogitsShape.nbDims, 3);
                EXPECT_EQ(1, acceptedTokensLogitsShape.d[0]);
                EXPECT_EQ(numPredTokens, acceptedTokensLogitsShape.d[1]);

                TensorPtr const& expectedLogits = ITensor::slice(expectedGenerationLogits[requestId], 1, numPredTokens);

                // For hyperparameters
                // Greater tolerance for the accepted logits of the target model.
                float atol = 0.f;
                float rtol = 0.01f;
                EXPECT_TRUE(compareLogits(*expectedLogits, *acceptedTokensLogits, atol, rtol));
            }

            if (checkRawLogits)
            {
                // Check generation logits
                TensorPtr const& expectedGenerationLogitsSliced
                    = ITensor::slice(expectedGenerationLogits[requestId], 0, numPredTokens);

                TensorPtr const& llmReqGeneration = llmReq.getGenerationLogitsHost();
                auto llmReqGenerationShape = llmReqGeneration->getShape();

                TensorPtr generationLogitsBeam = nullptr;
                if (llmReq.isStreaming())
                {
                    // Expect generation logits shape: [outputLength, beamWidth, vocabSizePad]
                    EXPECT_EQ(reqBeamWidth, llmReqGenerationShape.d[1]);
                    EXPECT_EQ(reqBeamWidth, 1);   // Streaming mode does not support beam > 1
                    llmReqGeneration->squeeze(1); // [outputLength, vocabSizePad]
                    generationLogitsBeam = llmReqGeneration;
                }
                else
                {
                    // Expect generation logits shape: [beamWidth, outputLength, vocabSizePad]
                    EXPECT_EQ(reqBeamWidth, llmReqGenerationShape.d[0]);
                    generationLogitsBeam
                        = std::shared_ptr(ITensor::slice(llmReqGeneration, beam, 1)); // [1, outputLength, vocabSizePad]
                    generationLogitsBeam->squeeze(0);                                 // [outputLength, vocabSizePad]
                }
                TensorPtr const& generationLogitsSliced = ITensor::slice(generationLogitsBeam, 0, numPredTokens);
                EXPECT_TRUE(compareLogits(*expectedGenerationLogitsSliced, *generationLogitsSliced));
            }
        }

        if (checkRawLogits)
        {
            // Check context logits
            TensorPtr const& llmReqContext = llmReq.getContextLogitsHost();
            auto llmReqContextShape = llmReqContext->getShape();
            EXPECT_EQ(llmReqContextShape.nbDims, 2);
            EXPECT_EQ(llmReq.mPromptLen, llmReqContextShape.d[0]);
            EXPECT_TRUE(compareLogits(*expectedContextLogits[requestId], *llmReqContext));
        }
    }
}

// Pick a different endId at random from one of the expected tokens
std::vector<TokenIdType> pickRandomEndIds(TestData const& testData, std::vector<SizeType32> const& givenInputLengths,
    SizeType32 const maxNewTokens, bool replaceLogits)
{
    auto const nbGivenInputs = testData.nbGivenInputs;
    auto const beamWidth = testData.beamWidth;
    auto* const expectedOutputData = bufferCast<TokenIdType>(*testData.expectedOutputIds);

    std::vector<TokenIdType> endIds;

    // For IFB, pick one of the output tokens as endId
    for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
    {
        TokenIdType skippedEndId0 = 0;
        TokenIdType skippedEndId1 = 0;
        SizeType32 endIdIndex = 0;
        TokenIdType endId = 0;
        auto const endIdRow = bi;
        auto const inputLength = givenInputLengths.at(endIdRow);
        do
        {
            auto const endIdBeam = std::rand() % beamWidth;
            auto const firstOutputIndex
                = tc::flat_index3(endIdRow, endIdBeam, inputLength, beamWidth, testData.maxSeqLen);
            // We do not use the 1st token for EndId because of Speculative Decoding test design
            // We skip 1st token because minLength is 1
            auto const endIdCol = 2 + (std::rand() % std::max(maxNewTokens - 2, 1));
            endIdIndex = firstOutputIndex + endIdCol;
            skippedEndId0 = expectedOutputData[firstOutputIndex];
            skippedEndId1 = expectedOutputData[firstOutputIndex + 1];
            endId = expectedOutputData[endIdIndex];
        } while (endId == skippedEndId0 || endId == skippedEndId1);
        // Workaround: The first example has endIdIndex 14, where the generation logits are almost same at
        // token ids 257 and 373, which causes unstable generation results. Hence, we use the one previous
        // token as endId.
        if (bi == 0 && !replaceLogits)
        {
            endId = expectedOutputData[endIdIndex - 1];
        }
        endIds.push_back(endId);
    }

    return endIds;
}

TestData loadTestData(ModelSpec const& modelSpec, ModelIds const modelIds, BeamResult const& beamResult,
    ITensor const& givenInput, SizeType32 const maxBeamWidth, bool const useRandomEndId, bool const replaceLogits,
    BufferManager& manager)
{
    auto const [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(givenInput, modelIds.padId);
    auto const& [beamWidth, resultsFile, contextLogitsFile, genLogitsFile, cumLogProbsFile, logProbsFile] = beamResult;

    TestData testData{nbGivenInputs, beamWidth};
    testData.expectedOutputIds = loadNpy(manager, resultsFile.string(), MemoryType::kCPU);

    auto* const expectedOutputData = bufferCast<TokenIdType>(*testData.expectedOutputIds);

    auto const& outputShape = testData.expectedOutputIds->getShape();
    EXPECT_EQ(outputShape.nbDims, 2);
    EXPECT_EQ(nbGivenInputs * beamWidth, outputShape.d[0]);
    testData.maxSeqLen = static_cast<SizeType32>(outputShape.d[1]);
    EXPECT_LE(maxInputLength, testData.maxSeqLen);
    EXPECT_LE(beamWidth, maxBeamWidth);

    auto const maxNewTokens = testData.maxSeqLen - maxInputLength;

    std::srand(42);

    if (useRandomEndId)
    {
        testData.endIds = pickRandomEndIds(testData, givenInputLengths, maxNewTokens, replaceLogits);
    }
    else
    {
        testData.endIds.insert(testData.endIds.end(), nbGivenInputs, modelIds.endId);
    }

    if (modelSpec.useLogits())
    {
        testData.loadContextLogits(contextLogitsFile, givenInputLengths, manager);
    }
    if (modelSpec.useLogits() || modelSpec.mAcceptDraftByLogits)
    {
        testData.loadGenerationLogits(genLogitsFile, manager);
    }
    if (modelSpec.mReturnLogProbs)
    {
        testData.loadLogProbs(cumLogProbsFile, logProbsFile, manager);
    }

    for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
    {
        auto const endId = testData.endIds[bi];
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            SizeType32 expectedLen = givenInputLengths[bi] + maxNewTokens;
            for (SizeType32 si = givenInputLengths[bi]; si < testData.maxSeqLen; ++si)
            {
                auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, testData.maxSeqLen);
                if (expectedOutputData[expectIndex] == endId)
                {
                    expectedLen = si;
                    break;
                }
            }
            // Fill new EOS token to the expected data
            for (SizeType32 si = expectedLen; si < testData.maxSeqLen; ++si)
            {
                auto const expectIndex = tc::flat_index2((bi * beamWidth + beam), si, testData.maxSeqLen);
                expectedOutputData[expectIndex] = endId;
            }

            testData.expectedOutputLengths[bi * beamWidth + beam] = expectedLen;
        }
    }

    if (modelSpec.mMaxDraftTokens > 0)
    {
        testData.makeDraft(
            modelSpec.mMaxDraftTokens, modelSpec.mAcceptDraftByLogits, genLogitsFile, givenInputLengths, manager);
    }

    return testData;
}

std::tuple<std::vector<SizeType32>, std::unordered_map<SizeType32, TestData>> loadTestData(ModelSpec const& modelSpec,
    ModelIds const modelIds, BeamResults const& resultsFilesBeamWidths, ITensor const& givenInput,
    SizeType32 const maxBeamWidth, bool const useRandomEndId, bool const replaceLogits, BufferManager& manager)
{
    // Map between beam width, and expected results for that beam width
    std::unordered_map<SizeType32, TestData> beamWidthTestData;
    std::vector<SizeType32> beamWidths;

    for (auto const& beamResult : resultsFilesBeamWidths)
    {
        auto const beamWidth = beamResult.beamWidth;

        EXPECT_EQ(std::find(beamWidths.begin(), beamWidths.end(), beamWidth), beamWidths.end());
        beamWidths.push_back(beamWidth);

        auto testData = loadTestData(
            modelSpec, modelIds, beamResult, givenInput, maxBeamWidth, useRandomEndId, replaceLogits, manager);
        beamWidthTestData.emplace(beamWidth, std::move(testData));
    }

    return {std::move(beamWidths), std::move(beamWidthTestData)};
}

RequestList runGptModelInference(std::shared_ptr<TrtGptModel>& trtGptModel, std::vector<SizeType32> const& beamWidths,
    std::unordered_map<SizeType32, TestData> const& beamWidthTestData, SizeType32 batchSize, SizeType32 nbGivenInputs,
    SizeType32 maxInputLength, SizeType32 padId, std::vector<SizeType32> const& givenInputLengths,
    TokenIdType const* givenInputData, ModelSpec const& modelSpec, TrtGptModelIfbTestType testType, int maxReqPerStep,
    bool prepopulateKVCache, bool enableStreamingMode, bool enableBlockReuse)
{
    // Fill the requests using givenInput
    // requestList will have batchSize requests
    RequestList requestList;

    SizeType32 requestId = 0;
    RequestList finishedRequestList;
    std::vector<SizeType32> reqVec;
    // Advance the requests until they are all finished
    if (COMM_SESSION.getRank() == 0)
    {
        SizeType32 numReq = 0;
        while (numReq < batchSize)
        {
            // Add appropriate number of requests in each iteration. For WAVEFRONT, this is always 1.
            // For RANDOM, it could be any integer <= maxReqPerStep including 0.
            SizeType32 reqThisStep{0};
            switch (testType)
            {
            case TrtGptModelIfbTestType::WAVEFRONT: reqThisStep = 1; break;
            case TrtGptModelIfbTestType::RANDOM: reqThisStep = rand() % (maxReqPerStep + 1); break;
            case TrtGptModelIfbTestType::BULK: [[fallthrough]];
            default: reqThisStep = batchSize; break;
            }
            reqThisStep = std::min(reqThisStep, (batchSize - numReq));
            reqVec.push_back(reqThisStep);
            numReq += reqThisStep;
        }
    }
    COMM_SESSION.bcast(reqVec, 0);

    SizeType32 reqVecIdx = 0;
    while (requestId < batchSize || !requestList.empty())
    {
        SizeType32 reqThisStep = reqVecIdx < reqVec.size() ? reqVec[reqVecIdx++] : 0;
        for (SizeType32 req = 0; req < reqThisStep; req++)
        {
            // Alternate between beamWidths
            SizeType32 beamWidth = beamWidths.at(requestId % beamWidths.size());
            auto const& testData = beamWidthTestData.at(beamWidth);
            auto const* const expectedOutputData = bufferCast<TokenIdType const>(*testData.expectedOutputIds);
            auto const maxSeqLen = testData.maxSeqLen;

            SamplingConfig samplingConfig{beamWidth};
            samplingConfig.temperature = std::vector{1.0f};
            samplingConfig.minLength = std::vector{1};
            samplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ull)};
            samplingConfig.topK = std::vector{1};
            samplingConfig.topP = std::vector{0.0f};
            samplingConfig.draftAcceptanceThreshold = std::vector{0.3f};
            samplingConfig.noRepeatNgramSize = std::vector{1 << 30};

            auto const [givenInputIdx, inputLength]
                = getRequestGivenInputIdxLength(requestId, nbGivenInputs, givenInputLengths);
            SizeType32 endId = testData.endIds[givenInputIdx];

            auto maxNewTokens = maxSeqLen - maxInputLength;
            // Run model only to produce a single token and prepopulate KV cache
            if (prepopulateKVCache || modelSpec.mKVCacheType == KVCacheType::kDISABLED)
            {
                maxNewTokens = 1;
            }
            auto const* const seqBegin = givenInputData + givenInputIdx * maxInputLength;
            auto tokens = std::make_shared<std::vector<int32_t>>(seqBegin, seqBegin + inputLength);
            if (!prepopulateKVCache && modelSpec.mMaxDraftTokens > 0)
            {
                // Append the 1st predicted token to the prompt to get the match with prepopulated KV cache
                auto const expectIndex = tc::flat_index3(givenInputIdx, 0, inputLength, 1, maxSeqLen);
                auto expectedToken = expectedOutputData[expectIndex];
                tokens->push_back(expectedToken);
                // subtract this token from maxNewTokens
                maxNewTokens -= 1;
            }
            auto r = std::make_shared<LlmRequest>(requestId, maxNewTokens, tokens, samplingConfig, false, endId, padId);

            auto const& draftTokens = testData.draftTokens[givenInputIdx];
            auto draftLogits = modelSpec.mAcceptDraftByLogits
                ? std::make_optional<ITensor::SharedPtr>(testData.draftLogits[givenInputIdx])
                : std::nullopt;
            if (!prepopulateKVCache && !draftTokens.empty())
            {
                r->setDraftTokens(std::make_shared<std::vector<TokenIdType>>(draftTokens));
                r->setDraftLogits(draftLogits);
            }

            SizeType32 maxDraftTokens{0};
            if (trtGptModel->getModelConfig().hasSpeculativeDecodingModule())
            {
                maxDraftTokens
                    = trtGptModel->getModelConfig().getSpeculativeDecodingModulePtr()->getMaxDecodingDraftTokens();
            }
            r->validate(trtGptModel->getMaxInputLen(), trtGptModel->getMaxSequenceLen(), maxDraftTokens,
                trtGptModel->getVocabSizePadded(), std::nullopt, enableBlockReuse);

            if (enableStreamingMode)
            {
                r->setReturnAllGeneratedTokens(true); // Test allGeneratedTokens in this test
                r->setStreaming(true);
            }

            auto const vocabSizePadded
                = trtGptModel->getModelConfig().getVocabSizePadded(trtGptModel->getWorldConfig().getSize());
            auto const logitDatatype = trtGptModel->getLogitDataType();
            if (modelSpec.mGatherLogits)
            {
                r->setReturnContextLogits(true);
                r->setReturnGenerationLogits(true);
                r->allocContextLogitsHost(vocabSizePadded, logitDatatype);
                r->allocGenerationLogitsHost(vocabSizePadded, logitDatatype);
            }

            if (!prepopulateKVCache && modelSpec.mAcceptDraftByLogits && !draftTokens.empty())
            {
                r->allocTargetModelAcceptedTokenLogitsHost(vocabSizePadded, logitDatatype);
                r->setReturnGenerationLogits(true);
            }

            if (modelSpec.mReplaceLogits)
            {
                LlmRequest::LogitsPostProcessor logitsCb
                    = [&testData](uint64_t rId, tensorrt_llm::runtime::ITensor::SharedPtr& logits,
                          LlmRequest::BeamTokens const& tokens,
                          tensorrt_llm::runtime::BufferManager::CudaStreamPtr streamPtr, std::optional<uint64_t> cId)
                {
                    auto const expectedGenerationLogits = testData.expectedGenerationLogits[rId];
                    auto const expectedContextLogits = testData.expectedContextLogits[rId];
                    auto const acceptedDraftTokensLengths = testData.acceptedDraftTokensLengths[rId];

                    auto const beamWidth = tokens.size();
                    TLLM_CHECK_WITH_INFO(beamWidth == 1, "Logits substitution is not supported for beam search");

                    auto const genLogitsOffset = tokens[0].size() - expectedContextLogits->getShape().d[0];
                    // TODO: Avoid static cast in TRT 10.0
                    auto const numLogits = static_cast<SizeType32>(logits->getShape().d[0]);
                    auto const numVerifyLogits = std::min(numLogits, acceptedDraftTokensLengths + 1);

                    TensorPtr logitsSlice = ITensor::slice(logits, 0, numVerifyLogits);

                    auto manager = BufferManager(streamPtr);
                    TensorPtr logitsHost = manager.copyFrom(*logitsSlice, MemoryType::kCPU);
                    manager.getStream().synchronize();

                    TensorPtr refLogitsHost
                        = ITensor::slice(expectedGenerationLogits, genLogitsOffset, numVerifyLogits);

                    EXPECT_TRUE(compareLogits(*refLogitsHost, *logitsHost, 0.f, 1e-2)) << "reqId: " << rId;

                    manager.copy(*refLogitsHost, *logitsSlice);
                };

                r->mLogitsPostProcessor = logitsCb;
            }

            if (modelSpec.mReturnLogProbs)
            {
                r->setReturnLogProbs(true);
            }
            requestList.push_back(r);
            ++requestId;
        }

        //  Advance all active requests by one step
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();

        // Check which requests are done, move them out
        for (auto it = requestList.cbegin(); it != requestList.cend();)
        {
            if ((*it)->isGenerationCompleteState())
            {
                finishedRequestList.push_back(*it);
                requestList.erase(it++);
            }
            else
            {
                ++it;
            }
        }
    }
    return finishedRequestList;
}

void runIfbTest(fs::path const& modelPath, ModelSpec const& modelSpec, ModelIds const modelIds,
    TrtGptModelType modelType, std::vector<int32_t> const& batchSizes, BeamResults const& resultsFilesBeamWidths,
    TrtGptModelIfbTestType testType, int maxReqPerStep, texec::ExecutorConfig const& executorConfig,
    bool enableStreamingMode, bool useRandomEndId)
{
    auto manager = BufferManager(std::make_shared<CudaStream>());
    auto const padId = modelIds.padId;

    // Load input data
    ASSERT_TRUE(fs::exists(DATA_PATH));
    auto const inputPath = DATA_PATH / modelSpec.mInputFile;
    auto const& givenInput = loadNpy(manager, inputPath.string(), MemoryType::kCPU);
    auto [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(*givenInput, padId);
    auto const* const givenInputData = bufferCast<TokenIdType const>(*givenInput);

    auto const& inputShape = givenInput->getShape();
    ASSERT_EQ(inputShape.nbDims, 2);
    ASSERT_GT(inputShape.d[0], 0);

    auto const maxBeamWidth = executorConfig.getMaxBeamWidth();
    // Load expected outputs for each beam width value
    auto [beamWidths, beamWidthTestData] = loadTestData(modelSpec, modelIds, resultsFilesBeamWidths, *givenInput,
        maxBeamWidth, useRandomEndId, modelSpec.mReplaceLogits, manager);

    int const worldSize = modelSpec.mTPSize * modelSpec.mPPSize * modelSpec.mCPSize;
    auto const worldConfig = WorldConfig::mpi(worldSize, modelSpec.mTPSize, modelSpec.mPPSize, modelSpec.mCPSize);

    ASSERT_TRUE(fs::exists(modelPath));

    for (auto batchSize : batchSizes)
    {
        std::cout << "=== batchSize:" << batchSize << " ===\n";

        auto trtGptModel = TrtGptModelFactory::create(modelPath, modelType, executorConfig, false);

        if (modelSpec.mKVCacheType == KVCacheType::kDISABLED)
        {
            ASSERT_FALSE(trtGptModel->hasKVCacheManager());
        }

        // Prepopulate KV cache for speculative decoding test
        bool const prepopulateKVCache = modelSpec.mMaxDraftTokens > 0;
        auto finishedRequestList = runGptModelInference(trtGptModel, beamWidths, beamWidthTestData, batchSize,
            nbGivenInputs, maxInputLength, padId, givenInputLengths, givenInputData, modelSpec, testType, maxReqPerStep,
            prepopulateKVCache, enableStreamingMode, modelSpec.mKVCacheReuse);

        if (prepopulateKVCache)
        {
            // Call the 2nd time with prefilled KV cache
            finishedRequestList = runGptModelInference(trtGptModel, beamWidths, beamWidthTestData, batchSize,
                nbGivenInputs, maxInputLength, padId, givenInputLengths, givenInputData, modelSpec, testType,
                maxReqPerStep, false, enableStreamingMode, modelSpec.mKVCacheReuse);
        }

        // WAR: disabled verification because of switched beams for different batch composition
        if (worldConfig.isFirstPipelineParallelRank()
            && (testType == TrtGptModelIfbTestType::BULK || maxBeamWidth == 1))
        {
            bool shouldVerify = true;

            if (testType == TrtGptModelIfbTestType::BULK)
            {
                if (modelSpec.mKVCacheType == KVCacheType::kDISABLED && maxBeamWidth != 1)
                {
                    // For disabled KV cache, only verify when maxBeamWidth is 1, the reason is we only compare with
                    // results with KV cache enabled case and usually, beams search results locate in last token while
                    // disabled KV cache only get exactly one new token.
                    shouldVerify = false;
                }
            }

            if (shouldVerify)
            {
                verifyOutput(finishedRequestList, beamWidthTestData, givenInputLengths, nbGivenInputs, modelSpec);
            }
        }
    }
}

struct BeamConfig
{
    SizeType32 maxBeamWidth;
    std::vector<SizeType32> beamWidths;
};

} // namespace

using ParamType = std::tuple<ModelParams, ModelSpec, TrtGptModelType, TrtGptModelIfbTestType, BeamConfig, // id: 0-4
    std::optional<int32_t>,   // 5. maxTokensInPagedKvCache
    std::optional<float>,     // 6. freeGpuMemoryFraction
    bool,                     // 7. enableTrtOverlap
    bool,                     // 8. enableChunkedContext
    bool,                     // 9. enableStreamingMode
    bool,                     // 10. enableCudaGraphMode
    std::optional<size_t>,    // 11. hostCacheSize
    bool,                     // 12. useRandomEndId
    std::vector<SizeType32>,  // 13. batchSizes
    std::optional<SizeType32> // 14. maxNumTokens
    >;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const modelSpec = std::get<1>(info.param);
    std::string name;
    switch (modelSpec.mDataType)
    {
    case nvinfer1::DataType::kFLOAT: name.append("Float"); break;
    case nvinfer1::DataType::kHALF: name.append("Half"); break;
    case nvinfer1::DataType::kINT8: name.append("Int8"); break;
    case nvinfer1::DataType::kINT32: name.append("Int32");
    case nvinfer1::DataType::kBOOL: name.append("Bool"); break;
    case nvinfer1::DataType::kUINT8: name.append("UInt8"); break;
    case nvinfer1::DataType::kFP8: name.append("Float8"); break;
    case nvinfer1::DataType::kBF16: name.append("BFloat16"); break;
    case nvinfer1::DataType::kINT4: name.append("Int4"); break;
    case nvinfer1::DataType::kFP4: name.append("Fp4"); break;
    default: throw std::runtime_error("Unsupported DataType"); break;
    }

    auto const modelType = std::get<2>(info.param);
    switch (modelType)
    {
    case TrtGptModelType::InflightBatching: name.append("IbModel"); break;
    case TrtGptModelType::InflightFusedBatching: name.append("FusedIbModel"); break;
    default: name.append("DefaultModel"); break;
    }

    switch (modelSpec.mKVCacheType)
    {
    case KVCacheType::kCONTINUOUS: name.append("ContinuousKVCache"); break;
    case KVCacheType::kPAGED: name.append("PagedKVCache"); break;
    case KVCacheType::kDISABLED: name.append("NoKVCache"); break;
    default: throw std::runtime_error("Unknown KVCacheType"); break;
    }

    auto const testType = std::get<3>(info.param);
    switch (testType)
    {
    case TrtGptModelIfbTestType::BULK: name.append("Bulk"); break;
    case TrtGptModelIfbTestType::WAVEFRONT: name.append("Wavefront"); break;
    case TrtGptModelIfbTestType::RANDOM: name.append("Random"); break;
    default: name.append("DefaultTest"); break;
    }
    BeamConfig const beamConfig = std::get<4>(info.param);
    name.append("MaxBeamWidth" + std::to_string(beamConfig.maxBeamWidth));
    for (auto const beamWdith : beamConfig.beamWidths)
    {
        name.append("Bw" + std::to_string(beamWdith));
    }

    auto const maxTokensInPagedKvCache = std::get<5>(info.param);
    if (maxTokensInPagedKvCache.has_value())
    {
        name.append("KvCacheSize" + std::to_string(maxTokensInPagedKvCache.value()));
    }

    auto const freeGpuMemoryFraction = std::get<6>(info.param);
    if (freeGpuMemoryFraction.has_value())
    {
        name.append("GpuFrac");
    }

    auto const enableTrtOverlap = std::get<7>(info.param);
    if (enableTrtOverlap)
    {
        name.append("TrtOverlap");
    }

    auto const enableChunkedContext = std::get<8>(info.param);
    if (enableChunkedContext)
    {
        name.append("Chunked");
    }

    if (modelSpec.mTPSize > 1)
    {
        name.append("TP" + std::to_string(modelSpec.mTPSize));
    }

    if (modelSpec.mPPSize > 1)
    {
        name.append("PP" + std::to_string(modelSpec.mPPSize));
    }

    if (modelSpec.mCPSize > 1)
    {
        name.append("CP" + std::to_string(modelSpec.mCPSize));
    }

    auto const useRandomEndId = std::get<12>(info.param);
    if (useRandomEndId)
    {
        name.append("EndId");
    }

    if (modelSpec.mMaxDraftTokens > 0)
    {
        name.append("DraftTokens" + std::to_string(modelSpec.mMaxDraftTokens));
    }

    if (modelSpec.mAcceptDraftByLogits)
    {
        name.append("AcceptByLogits");
    }

    if (modelSpec.mCapacitySchedulerPolicy)
    {
        name.append(modelSpec.getCapacitySchedulerString());
    }

    auto const enableStreamingMode = std::get<9>(info.param);
    if (enableStreamingMode)
    {
        name.append("Streaming");
    }

    auto const enableCudaGraphMode = std::get<10>(info.param);
    if (enableCudaGraphMode)
    {
        name.append("CudaGraph");
    }

    auto const enableHostCache = std::get<11>(info.param);
    if (enableHostCache)
    {
        name.append("SecondaryOffloading");
    }

    return name;
}

class ParamTest : public TrtModelRealDecoderTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, Test)
{

    auto const& beamConfig = std::get<4>(GetParam());
    auto const& beamWidths = beamConfig.beamWidths;

    auto const modelParams = std::get<0>(GetParam());
    auto const modelIds = modelParams.ids;
    auto const* const modelDir = modelParams.baseDir;
    auto const modelSpec = std::get<1>(GetParam());

    auto const useRandomEndId = std::get<12>(GetParam());

    auto const batchSizes = std::get<13>(GetParam());

    std::ostringstream gpuSizePath;
    gpuSizePath << "tp" << modelSpec.mTPSize << "-pp" << modelSpec.mPPSize << "-cp" << modelSpec.mCPSize;
    gpuSizePath << "-gpu";

    auto const modelPath{ENGINE_PATH / modelDir / modelSpec.getModelPath() / gpuSizePath.str()};

    auto const inputPath = DATA_PATH / modelSpec.mInputFile;

    BeamResults beamResults;
    beamResults.reserve(beamWidths.size());
    for (auto beamWidth : beamWidths)
    {
        fs::path resultsPath
            = DATA_PATH / modelDir / ((beamWidth == 1) ? "sampling" : "beam_search_" + std::to_string(beamWidth));
        fs::path generationLogitsPath
            = modelSpec.mCollectGenerationLogits ? (resultsPath / modelSpec.getGenerationLogitsFile()).string() : "";
        fs::path contextLogitsPath
            = modelSpec.mCollectContextLogits ? (resultsPath / modelSpec.getContextLogitsFile()).string() : "";
        fs::path cumLogProbsPath
            = modelSpec.mCollectCumLogProbs ? (resultsPath / modelSpec.getCumLogProbsFile()).string() : "";
        fs::path logProbsPath = modelSpec.mCollectLogProbs ? (resultsPath / modelSpec.getLogProbsFile()).string() : "";

        beamResults.emplace_back(beamWidth, (resultsPath / modelSpec.getResultsFile()).string(), contextLogitsPath,
            generationLogitsPath, cumLogProbsPath, logProbsPath);
    }

    auto const modelType = std::get<2>(GetParam());
    auto const testType = std::get<3>(GetParam());
    auto const enableStreamingMode = std::get<9>(GetParam());
    auto const cudaGraphMode = std::get<10>(GetParam());

    if (!(modelSpec.mUsePackedInput
            && (modelSpec.mKVCacheType == KVCacheType::kPAGED || modelSpec.mKVCacheType == KVCacheType::kDISABLED)))
    {
        GTEST_SKIP() << "Inflight batching requires packed input and (paged KV cache or disabled KV cache).";
    }

    if (!modelSpec.mUsePackedInput && useRandomEndId)
    {
        GTEST_SKIP() << "Test does not support endId test with padded inputs";
    }

    for (auto beamWidth : beamWidths)
    {
        if (useRandomEndId && beamWidth > 1)
        {
            GTEST_SKIP() << "Test does not support endId test with beam search";
        }

        if (modelSpec.mMaxDraftTokens > 0 && beamWidth > 1)
        {
            GTEST_SKIP() << "Target model in speculative decoding does not support beam search";
        }
    }

    auto executorConfig = texec::ExecutorConfig{};

    auto const maxTokens = std::get<5>(GetParam());
    auto const enableBlockReuse = modelSpec.mMaxDraftTokens > 0 || modelSpec.mKVCacheReuse;
    auto const freeGpuMemoryFraction = std::get<6>(GetParam());
    auto const hostCacheSize = std::get<11>(GetParam());
    auto const kvCacheConfig = texec::KvCacheConfig{
        enableBlockReuse, maxTokens, std::nullopt, std::nullopt, freeGpuMemoryFraction, hostCacheSize};
    executorConfig.setKvCacheConfig(kvCacheConfig);

    executorConfig.setEnableTrtOverlap(std::get<7>(GetParam()));
    executorConfig.setEnableChunkedContext(std::get<8>(GetParam()));
    auto const maxNumTokens = std::get<14>(GetParam());
    if (maxNumTokens.has_value())
    {
        executorConfig.setMaxNumTokens(maxNumTokens.value());
    }
    executorConfig.setNormalizeLogProbs(false);
    executorConfig.setMaxBeamWidth(beamConfig.maxBeamWidth);
    executorConfig.setGatherGenerationLogits(modelSpec.mCollectGenerationLogits);
    auto extendedRuntimePerfKnobConfig = texec::ExtendedRuntimePerfKnobConfig{};
    extendedRuntimePerfKnobConfig.setCudaGraphMode(cudaGraphMode);
    executorConfig.setExtendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig);

    auto const capacitySchedulerPolicy
        = modelSpec.mCapacitySchedulerPolicy.value_or(texec::CapacitySchedulerPolicy::kMAX_UTILIZATION);
    executorConfig.setSchedulerConfig(texec::SchedulerConfig{capacitySchedulerPolicy});

    if (modelSpec.mSpecDecodingMode == SpeculativeDecodingMode::LookaheadDecoding())
    {
        auto decodingConfig = texec::DecodingConfig{};
        decodingConfig.setLookaheadDecodingConfig(texec::LookaheadDecodingConfig(5, 5, 5));
        executorConfig.setDecodingConfig(decodingConfig);
    }

    for (auto beamWidth : beamWidths)
    {
        if (executorConfig.getEnableTrtOverlap() && beamWidth > 1)
        {
            GTEST_SKIP() << "TrtOverlap is not supported with beam search";
        }
    }

    if (executorConfig.getEnableTrtOverlap() && modelSpec.mMaxDraftTokens > 0)
    {
        GTEST_SKIP() << "TrtOverlap is not supported with speculative decoding";
    }

    // Warning: This should be the last check before running the test.
    // It will initialize MPI which can take significant time.
    if (modelSpec.mTPSize * modelSpec.mPPSize * modelSpec.mCPSize != COMM_SESSION.getSize())
    {
        GTEST_SKIP() << "Model's world size " << modelSpec.mPPSize * modelSpec.mTPSize * modelSpec.mCPSize
                     << " is not equal to the system world size";
    }

    runIfbTest(modelPath, modelSpec, modelIds, modelType, batchSizes, beamResults, testType, 2, executorConfig,
        enableStreamingMode, useRandomEndId);
}

auto constexpr gptModelParams = ModelParams{GPT_MODEL_DIR, ModelIds{50256, 50256}};

std::shared_ptr<ModelSpec> getGptDraftTestsCompareModelSpec()
{
    auto pModelSpec = std::make_shared<ModelSpec>(INPUT_FILE, nvinfer1::DataType::kHALF);
    pModelSpec->useGptAttentionPlugin();
    pModelSpec->gatherLogits();
    pModelSpec->usePackedInput();
    pModelSpec->setKVCacheType(KVCacheType::kPAGED);

    return pModelSpec;
}

std::shared_ptr<ModelSpec> getMedusaTestsCompareModelSpec()
{
    auto pModelSpec = std::make_shared<ModelSpec>(LONG_INPUT_FILE, nvinfer1::DataType::kHALF);
    pModelSpec->useGptAttentionPlugin();
    pModelSpec->usePackedInput();
    pModelSpec->setKVCacheType(KVCacheType::kPAGED);
    pModelSpec->setMaxOutputLength(128);

    return pModelSpec;
}

std::shared_ptr<ModelSpec> getEagleTestsCompareModelSpec()
{
    auto pModelSpec = std::make_shared<ModelSpec>(LONG_INPUT_FILE, nvinfer1::DataType::kHALF);
    pModelSpec->useGptAttentionPlugin();
    pModelSpec->usePackedInput();
    pModelSpec->setKVCacheType(KVCacheType::kPAGED);
    pModelSpec->setMaxOutputLength(128);

    return pModelSpec;
}

std::shared_ptr<ModelSpec> getGptChunkedContextTestsCompareModelSpec()
{
    auto pModelSpec = std::make_shared<ModelSpec>(LONG_INPUT_FILE, nvinfer1::DataType::kHALF);
    pModelSpec->useGptAttentionPlugin();
    pModelSpec->usePackedInput();
    pModelSpec->setKVCacheType(KVCacheType::kPAGED);
    pModelSpec->setMaxInputLength(128);

    return pModelSpec;
}

INSTANTIATE_TEST_SUITE_P(GptTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF,
                []() -> std::shared_ptr<ModelSpec>
                {
                    auto pModelSpec = std::make_shared<ModelSpec>(INPUT_FILE, nvinfer1::DataType::kHALF);
                    pModelSpec->useGptAttentionPlugin().setKVCacheType(KVCacheType::kPAGED).usePackedInput();
                    return pModelSpec;
                }()}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kDISABLED)
                .usePackedInput()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}}         // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt, 1280),               // maxTokensInPagedKvCache
        testing::Values(std::nullopt, 0.4),                // freeGpuMemoryFraction
        testing::Values(true),                             // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptRandomEndIdTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}}         // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt, 1280),               // maxTokensInPagedKvCache
        testing::Values(std::nullopt, 0.4),                // freeGpuMemoryFraction
        testing::Values(true),                             // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(true),                             // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptKVOffloadingTest, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{LONG_INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()
                .setKVCacheReuse(true)),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(256),                              // maxTokensInPagedKvCache
        testing::Values(std::nullopt, 0.4),                // freeGpuMemoryFraction
        testing::Values(true),                             // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(100000000),                        // hostCacheSize
        testing::Values(false, true),                      // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptCudaGraphTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()
                .capacitySchedulerPolicy(texec::CapacitySchedulerPolicy::kSTATIC_BATCH),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()
                .capacitySchedulerPolicy(texec::CapacitySchedulerPolicy::kMAX_UTILIZATION)),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}}         // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(true),                             // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(true),                             // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptSwitchBwTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            BeamConfig{2, {1}}                       // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),               // maxTokensInPagedKvCache
        testing::Values(0.4),                        // freeGpuMemoryFraction
        testing::Values(false),                      // enableTrtOverlap
        testing::Values(true),                       // enableChunkedContext
        testing::Values(false),                      // enableStreamingMode
        testing::Values(false),                      // enableCudaGraphMode
        testing::Values(std::nullopt),               // hostCacheSize
        testing::Values(false),                      // useRandomEndId
        testing::Values(std::vector<SizeType32>{4}), // batchSizes
        testing::Values(std::nullopt)                // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptNProfilesTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                            .useGptAttentionPlugin()
                            .usePackedInput()
                            .setKVCacheType(KVCacheType::kPAGED)
                            .useMultipleProfiles()),
        testing::Values(TrtGptModelType::InflightFusedBatching), testing::Values(TrtGptModelIfbTestType::BULK),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}}         // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt, 1280),               // maxTokensInPagedKvCache
        testing::Values(std::nullopt, 0.4),                // freeGpuMemoryFraction
        testing::Values(true),                             // enableTrtOverlap
        testing::Values(true),                             // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(true),                             // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptSqTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                            .useGptAttentionPlugin()
                            .usePackedInput()
                            .setKVCacheType(KVCacheType::kPAGED)
                            .setQuantMethod(QuantMethod::kSMOOTH_QUANT),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF,
                []() -> std::shared_ptr<ModelSpec>
                {
                    auto pModelSpec = std::make_shared<ModelSpec>(INPUT_FILE, nvinfer1::DataType::kHALF);
                    pModelSpec->useGptAttentionPlugin()
                        .usePackedInput()
                        .setKVCacheType(KVCacheType::kPAGED)
                        .setQuantMethod(QuantMethod::kSMOOTH_QUANT);
                    return pModelSpec;
                }()}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kDISABLED)
                .setQuantMethod(QuantMethod::kSMOOTH_QUANT)),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            // FIXME: disabled flaky beam search tests (https://nvbugspro.nvidia.com/bug/4646234)
            BeamConfig{1, {1}}                             //, BeamConfig{2, {2}}
            ),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

// disabled because paused requests generate different tokens after resuming
INSTANTIATE_TEST_SUITE_P(DISABLED_GptChunkedContextTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF, getGptChunkedContextTestsCompareModelSpec()}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .setMaxInputLength(128)),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(TrtGptModelIfbTestType::BULK),     // TrtGptModelIfbTestType
        testing::Values(BeamConfig{1, {1}}),               // beam config
        testing::Values(257),                              // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(true),                             // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptChunkedLongContextTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{LONG_INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .setMaxInputLength(128),
            ModelSpec{LONG_INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDraftTokensExternalDecoding()
                .setDraftTokens(5)),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT,
            TrtGptModelIfbTestType::RANDOM),               // TrtGptModelIfbTestType
        testing::Values(BeamConfig{1, {1}}),               // beam config
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(true),                             // enableTrtOverlap
        testing::Values(true),                             // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(64)                                // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptDraftTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF, getGptDraftTestsCompareModelSpec()}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDraftTokensExternalDecoding()
                .setDraftTokens(5)
                .replaceLogits()
                .collectGenerationLogitsFile()
                .collectContextLogitsFile(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF, getGptDraftTestsCompareModelSpec()}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useDraftTokensExternalDecoding()
                .setDraftTokens(5)
                .useAcceptByLogits()
                .replaceLogits()
                .collectGenerationLogitsFile()
                .collectContextLogitsFile()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),               // beamConfig
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(true),                             // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false, true),                      // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptLogitsTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            // modelSpec
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .gatherLogits()
                .collectGenerationLogitsFile()
                .collectContextLogitsFile()),
        testing::Values(TrtGptModelType::InflightBatching, TrtGptModelType::InflightFusedBatching), // modelType
        testing::Values(TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT,
            TrtGptModelIfbTestType::RANDOM),                                                        // testType
        testing::Values(BeamConfig{1, {1}}),                                                        // beamConfig
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(true),                             // enableChunkedContext
        testing::Values(false, true),                      // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(true),                             // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptLogProbsTests, ParamTest,
    testing::Combine(testing::Values(gptModelParams),
        testing::Values(
            // modelSpec
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .returnLogProbs()
                .collectCumLogProbsFile()
                .collectLogProbsFile()),
        testing::Values(TrtGptModelType::InflightFusedBatching), // modelType
        testing::Values(TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT,
            TrtGptModelIfbTestType::RANDOM),                     // testType
        testing::Values(BeamConfig{1, {1}}),                     // beamConfig
        testing::Values(std::nullopt),                           // maxTokensInPagedKvCache
        testing::Values(0.4),                                    // freeGpuMemoryFraction
        testing::Values(false),                                  // enableTrtOverlap
        testing::Values(true),                                   // enableChunkedContext
        testing::Values(false),                                  // enableStreamingMode
        testing::Values(false),                                  // enableCudaGraphMode
        testing::Values(std::nullopt),                           // hostCacheSize
        testing::Values(false),                                  // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}),       // batchSizes
        testing::Values(std::nullopt)                            // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(GptjTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPTJ_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kCONTINUOUS)
                .usePackedInput(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()

                ),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        // WAR: disable wavefront and random tests on because of switched beams
        testing::Values(TrtGptModelIfbTestType::BULK
            /* , TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM */),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}}         // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(MambaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{MAMBA_MODEL_DIR, {0, 1}}),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kCONTINUOUS)
                .usePackedInput(),
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()

                ),
        testing::Values(TrtGptModelType::InflightBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(RecurrentGemmaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{RECURRENTGEMMA_MODEL_DIR, {0, 1}}),
        testing::Values(ModelSpec{INPUT_FILE, nvinfer1::DataType::kHALF}
                            .useGptAttentionPlugin()
                            .setKVCacheType(KVCacheType::kPAGED)
                            .usePackedInput()

                ),
        testing::Values(TrtGptModelType::InflightBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(LlamaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{LLAMA_MODEL_DIR, {LLAMA_END_ID, LLAMA_PAD_ID}}),
        testing::Values(
            //
            ModelSpec{INPUT_LLAMA_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput(),
            ModelSpec{INPUT_LLAMA_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePipelineParallelism(4),
            ModelSpec{INPUT_LLAMA_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useTensorParallelism(4),
            ModelSpec{INPUT_LLAMA_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePipelineParallelism(2)
                .useTensorParallelism(2)

                ),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // TODO: enable more tests when mixed beam width is supported
            BeamConfig{1, {1}}, BeamConfig{2, {2}}         // , BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(ChatGlmTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{CHATGLM_MODEL_DIR, {130005, 3}}),
        testing::Values(
            //
            ModelSpec{CHATGLM_INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(false, true),                      // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

// ChatGlm0Tests is for glm-10b.
INSTANTIATE_TEST_SUITE_P(ChatGlm0Tests, ParamTest,
    testing::Combine(testing::Values(ModelParams{GLM_MODEL_DIR, {50258, 50256}}),
        testing::Values(
            //
            ModelSpec{GLM_INPUT_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(false),                            // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

// https://nvbugspro.nvidia.com/bug/4640177
// WAVEFRONT and RANDOM are disabled because of the accuracy mismatch
INSTANTIATE_TEST_SUITE_P(MedusaTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{MEDUSA_MODEL_DIR, {2, 2}}),
        testing::Values(
            //
            ModelSpec{INPUT_VICUNA_FILE, nvinfer1::DataType::kHALF, getMedusaTestsCompareModelSpec()}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useMedusa()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt),               // maxTokensInPagedKvCache
        testing::Values(0.4),                        // freeGpuMemoryFraction
        testing::Values(false),                      // enableTrtOverlap
        testing::Values(true),                       // enableChunkedContext
        testing::Values(false),                      // enableStreamingMode
        testing::Values(true, false),                // enableCudaGraphMode
        testing::Values(std::nullopt),               // hostCacheSize
        testing::Values(false),                      // useRandomEndId
        testing::Values(std::vector<SizeType32>{8}), // batchSizes
        testing::Values(std::nullopt)                // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(EagleTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{EAGLE_MODEL_DIR, {2, 2}}),
        testing::Values(
            //
            ModelSpec{INPUT_VICUNA_FILE, nvinfer1::DataType::kHALF, getEagleTestsCompareModelSpec()}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useEagle()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),
        testing::Values(std::nullopt),               // maxTokensInPagedKvCache
        testing::Values(0.4),                        // freeGpuMemoryFraction
        testing::Values(false),                      // enableTrtOverlap
        testing::Values(true),                       // enableChunkedContext
        testing::Values(false),                      // enableStreamingMode
        testing::Values(true, false),                // enableCudaGraphMode
        testing::Values(std::nullopt),               // hostCacheSize
        testing::Values(false),                      // useRandomEndId
        testing::Values(std::vector<SizeType32>{8}), // batchSizes
        testing::Values(std::nullopt)                // maxNumTokens
        ),
    generateTestName);

INSTANTIATE_TEST_SUITE_P(LlamaLookaheadDecodingTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{LLAMA_MODEL_DIR, {LLAMA_END_ID, LLAMA_PAD_ID}}),
        testing::Values(
            //
            ModelSpec{INPUT_LLAMA_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useLookaheadDecoding()),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),             // beamConfig
        testing::Values(std::nullopt),                   // maxTokensInPagedKvCache
        testing::Values(0.4),                            // freeGpuMemoryFraction
        testing::Values(false),                          // enableTrtOverlap
        testing::Values(false),                          // enableChunkedContext
        testing::Values(false),                          // enableStreamingMode
        testing::Values(false),                          // enableCudaGraphMode
        testing::Values(std::nullopt),                   // hostCacheSize
        testing::Values(true),                           // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 16}), // batchSizes
        testing::Values(std::nullopt)                    // maxNumTokens
        ),

    generateTestName);

INSTANTIATE_TEST_SUITE_P(ExplicitDraftTokensDecodingTests, ParamTest,
    testing::Combine(testing::Values(ModelParams{EXPLICIT_DRAFT_MODEL_DIR, {2, 2}}),
        testing::Values(
            //
            ModelSpec{INPUT_VICUNA_FILE, nvinfer1::DataType::kHALF}
                .useGptAttentionPlugin()
                .usePackedInput()
                .setKVCacheType(KVCacheType::kPAGED)
                .useExplicitDraftTokensDecoding()
                .setMaxOutputLength(128)),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(BeamConfig{1, {1}}),         // beamConfig
        testing::Values(std::nullopt),               // maxTokensInPagedKvCache
        testing::Values(0.4),                        // freeGpuMemoryFraction
        testing::Values(false),                      // enableTrtOverlap
        testing::Values(true),                       // enableChunkedContext
        testing::Values(false),                      // enableStreamingMode
        testing::Values(false),                      // enableCudaGraphMode
        testing::Values(std::nullopt),               // hostCacheSize
        testing::Values(false),                      // useRandomEndId
        testing::Values(std::vector<SizeType32>{8}), // batchSizes
        testing::Values(std::nullopt)                // maxNumTokens
        ),

    generateTestName);

#ifdef ENABLE_FP8
// Using IFB-enabled engine
INSTANTIATE_TEST_SUITE_P(GptjFP8Tests, ParamTest,
    testing::Combine(testing::Values(ModelParams{GPTJ_MODEL_DIR, {50256, 50256}}),
        testing::Values(
            //
            ModelSpec{INPUT_FILE, nvinfer1::DataType::kFP8}
                .useGptAttentionPlugin()
                .setKVCacheType(KVCacheType::kPAGED)
                .usePackedInput()

                ),
        testing::Values(TrtGptModelType::InflightFusedBatching),
        testing::Values(
            TrtGptModelIfbTestType::BULK, TrtGptModelIfbTestType::WAVEFRONT, TrtGptModelIfbTestType::RANDOM),
        testing::Values(
            // TODO: enable more tests when supported
            BeamConfig{1, {1}}                             // , BeamConfig{2, {2}}, BeamConfig{2, {1, 2}}
            ),
        testing::Values(std::nullopt),                     // maxTokensInPagedKvCache
        testing::Values(0.4),                              // freeGpuMemoryFraction
        testing::Values(false),                            // enableTrtOverlap
        testing::Values(true),                             // enableChunkedContext
        testing::Values(false),                            // enableStreamingMode
        testing::Values(false),                            // enableCudaGraphMode
        testing::Values(std::nullopt),                     // hostCacheSize
        testing::Values(false),                            // useRandomEndId
        testing::Values(std::vector<SizeType32>{1, 2, 8}), // batchSizes
        testing::Values(std::nullopt)                      // maxNumTokens
        ),
    generateTestName);

#endif
