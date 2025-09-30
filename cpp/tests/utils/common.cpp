/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "common.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/testing/modelSpec.h"
#include "tests/utils/common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace tensorrt_llm::testing
{
namespace fs = std::filesystem;
namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

std::string PathUtil::FP16_GPT_ATTENTION_PACKED_DIR()
{
    return ModelSpec::getDefaultModelSpec().setKVCacheType(KVCacheType::kCONTINUOUS).getModelPath();
}

std::string PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DIR()
{
    return ModelSpec::getDefaultModelSpec().getModelPath();
}

std::string PathUtil::FP16_GPT_LORA_DIR()
{
    return ModelSpec::getDefaultModelSpec().useLoraPlugin().getModelPath();
}

std::string PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR()
{
    return ModelSpec::getDefaultModelSpec().useDraftTokensExternalDecoding().getModelPath();
}

std::string PathUtil::FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().getModelPath();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_FILE()
{
    return ModelSpec::getDefaultModelSpec().getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_LONG_RESULT_FILE()
{
    return ModelSpec::getDefaultModelSpec().setInputFile("input_tokens_long.npy").getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().getGenerationLogitsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().getContextLogitsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE()
{
    return ModelSpec::getDefaultModelSpec().getCumLogProbsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_CUM_LOG_PROBS_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().getCumLogProbsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE()
{
    return ModelSpec::getDefaultModelSpec().getLogProbsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_LOG_PROBS_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().getLogProbsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP1_FILE()
{
    return ModelSpec::getDefaultModelSpec().getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE()
{
    return ModelSpec::getDefaultModelSpec().useTensorParallelism(4).getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE()
{
    return ModelSpec::getDefaultModelSpec().useTensorParallelism(2).usePipelineParallelism(2).getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE()
{
    return ModelSpec::getDefaultModelSpec().usePipelineParallelism(4).getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP2_FILE()
{
    return ModelSpec::getDefaultModelSpec().usePipelineParallelism(2).getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP1_FILE()
{
    return ModelSpec::getDefaultModelSpec().useTensorParallelism(2).getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_TP4_PP1_FILE()
{
    return ModelSpec::getDefaultModelSpec().useTensorParallelism(4).getContextLogitsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_TP4_PP1_FILE()
{
    return ModelSpec::getDefaultModelSpec().useTensorParallelism(4).getGenerationLogitsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_TP4_PP1_FILE()
{
    return ModelSpec::getDefaultModelSpec().useTensorParallelism(4).getCumLogProbsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_TP4_PP1_FILE()
{
    return ModelSpec::getDefaultModelSpec().useTensorParallelism(4).getLogProbsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_GATHER_CONTEXTFMHAFP32ACC_RESULT_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().enableContextFMHAFp32Acc().getResultsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXTFMHAFP32ACC_GENERATION_LOGITS_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().enableContextFMHAFp32Acc().getGenerationLogitsFile();
}

std::string PathUtil::FP16_PLUGIN_PACKED_PAGED_CONTEXTFMHAFP32ACC_CONTEXT_LOGITS_FILE()
{
    return ModelSpec::getDefaultModelSpec().gatherLogits().enableContextFMHAFp32Acc().getContextLogitsFile();
}

void TestData::loadLogProbs(
    fs::path const& cumLogProbsFile, fs::path const& logProbsFile, tr::BufferManager const& manager)
{
    TLLM_CHECK_WITH_INFO(
        cumLogProbsFile != "", "Testing return log probs, but missing the expected cum log probs results file.");
    auto expectedCumLogProbsPtr
        = std::shared_ptr(tr::utils::loadNpy(manager, cumLogProbsFile.string(), MemoryType::kCPU));

    TLLM_CHECK_WITH_INFO(
        logProbsFile != "", "Testing return log probs, but missing the expected log probs results file.");
    auto expectedLogProbsPtr = std::shared_ptr(tr::utils::loadNpy(manager, logProbsFile.string(), MemoryType::kCPU));

    for (SizeType32 inputIdx = 0; inputIdx < nbGivenInputs; ++inputIdx)
    {
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto expectedCumLogProbsBatchSlice = std::shared_ptr(ITensor::slice(expectedCumLogProbsPtr, inputIdx, 1));
            expectedCumLogProbsBatchSlice->squeeze(0);                     // bs
            expectedCumLogProbs[inputIdx] = expectedCumLogProbsBatchSlice; // shape: [beamWidth]

            auto expectedLogProbsBatchSlice = std::shared_ptr(ITensor::slice(expectedLogProbsPtr, inputIdx, 1));
            expectedLogProbsBatchSlice->squeeze(0);                  // bs
            expectedLogProbs[inputIdx] = expectedLogProbsBatchSlice; // shape: [beamWidth, numOutputTokens]
        }
    }
}

void TestData::loadContextLogits(fs::path const& contextLogitsFile, std::vector<SizeType32> const& givenInputLengths,
    tr::BufferManager const& manager)
{
    TLLM_CHECK_WITH_INFO(contextLogitsFile != "",
        "Testing with gather or replace logits, but missing the expected context logits results file.");
    auto expectedContextLogitsPtr
        = std::shared_ptr(tr::utils::loadNpy(manager, contextLogitsFile.string(), MemoryType::kCPU));

    int promptOffset = 0;
    for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
    {
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto expectedContextLogitBatchSlice
                = std::shared_ptr(ITensor::slice(expectedContextLogitsPtr, promptOffset, givenInputLengths.at(bi)));
            expectedContextLogits.at(bi) = expectedContextLogitBatchSlice; // shape: [prompt_length, vocab_size]
        }
        promptOffset += givenInputLengths.at(bi);
    }
}

void TestData::loadGenerationLogits(fs::path const& genLogitsFile, tr::BufferManager const& manager)
{
    TLLM_CHECK_WITH_INFO(genLogitsFile != "",
        "Testing with gather or replace logits, but missing the expected generation logits results file.");
    auto expectedGenerationLogitsPtr
        = std::shared_ptr(tr::utils::loadNpy(manager, genLogitsFile.string(), MemoryType::kCPU));

    for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
    {
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto expectedGenerationLogitBatchSlice
                = std::shared_ptr(ITensor::slice(expectedGenerationLogitsPtr, bi, 1));
            expectedGenerationLogitBatchSlice->squeeze(0);                       // bs
            expectedGenerationLogitBatchSlice->squeeze(0);                       // beam
            expectedGenerationLogits.at(bi) = expectedGenerationLogitBatchSlice; // shape: [max_output_len, vocab_size]
        }
    }
}

void TestData::makeDraft(SizeType32 maxDraftTokens, bool acceptDraftByLogits, fs::path const& genLogitsFile,
    std::vector<SizeType32> const& givenInputLengths, tr::BufferManager const& manager)
{
    TLLM_CHECK(beamWidth == 1);

    ITensor::SharedPtr expectedGenerationLogitsPtr;
    if (acceptDraftByLogits)
    {
        TLLM_CHECK_WITH_INFO(
            genLogitsFile != "", "Testing Draft token, but missing the expected generation logits results file.");
        expectedGenerationLogitsPtr
            = std::shared_ptr(tr::utils::loadNpy(manager, genLogitsFile.string(), MemoryType::kCPU));
    }

    std::vector<SizeType32> draftLengths(givenInputLengths.size());
    // first draft length stays 0
    std::transform(givenInputLengths.begin() + 1, givenInputLengths.end(), draftLengths.begin() + 1,
        [this, &maxDraftTokens](auto inputLength)
        { return std::rand() % std::min((maxSeqLen - (inputLength + 1)), maxDraftTokens) + 1; });

    auto* const expectedOutputData = tr::bufferCast<TokenIdType>(*expectedOutputIds);
    for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
    {
        SizeType32 constexpr beamIdx{0};
        auto const endId = endIds.at(bi);
        auto const draftLen = draftLengths.at(bi);
        auto acceptedLen = draftLen > 0 ? std::rand() % draftLen : 0;

        if (acceptDraftByLogits && draftLen > 0)
        {
            auto expectedLogitBatchSlice = std::shared_ptr(ITensor::slice(expectedGenerationLogitsPtr, bi, 1));
            expectedLogitBatchSlice->squeeze(0); // bs
            expectedLogitBatchSlice->squeeze(0); // beam
            auto expectedLogitBatchStepSlice = std::shared_ptr(ITensor::slice(expectedLogitBatchSlice, 1, draftLen));
            auto expectedLogitBatchStepView = ITensor::view(expectedLogitBatchStepSlice,
                ITensor::makeShape({draftLen, 1, 1, expectedLogitBatchStepSlice->getShape().d[1]}));
            draftLogits.at(bi) = manager.copyFrom(*expectedLogitBatchStepView, MemoryType::kCPU);
        }

        for (SizeType32 si = 0; si < draftLen; ++si)
        {
            auto const draftIndex
                = tc::flat_index3(bi, beamIdx, givenInputLengths.at(bi) + si + 1, beamWidth, maxSeqLen);
            auto draftToken = expectedOutputData[draftIndex];
            if (draftToken == endId)
            {
                acceptedLen = std::min(acceptedLen, si);
            }
            if (si >= acceptedLen)
            {
                draftToken = -1;
                if (acceptDraftByLogits)
                {
                    auto vocabSizePadded = expectedGenerationLogitsPtr->getShape().d[3];
                    auto* draftLogitsPtr = tr::bufferCast<float>(*draftLogits.at(bi));
                    for (SizeType32 vi = 0; vi < vocabSizePadded; ++vi)
                    {
                        draftLogitsPtr[si * vocabSizePadded + vi] = 0.f;
                    }
                }
            }
            draftTokens.at(bi).push_back(draftToken);
        }
        acceptedDraftTokensLengths.at(bi) = acceptedLen;

        auto const expectedLen = expectedOutputLengths.at(bi * beamWidth + beamIdx);
        TLLM_CHECK(expectedLen > 0);
        expectedOutputLengths[bi * beamWidth + beamIdx]
            = draftLen > 0 ? std::min(expectedLen, (givenInputLengths.at(bi) + 1) + acceptedLen + 1) : expectedLen;
    }
}

template <typename T>
bool invokeCompareLogits(ITensor const& groundTruthLogits, ITensor const& outputLogits, float atol, float rtol)
{
    bool allMatch = true;
    T const* const gtLogitsPtr = tr::bufferCast<T>(groundTruthLogits);
    T const* const outputLogitsPtr = tr::bufferCast<T>(outputLogits);

    size_t outputSize = outputLogits.getSize();
    int errorNumber = 0;

    for (size_t i = 0; i < outputSize; i++)
    {
        if (!almostEqual(outputLogitsPtr[i], gtLogitsPtr[i], atol, rtol))
        {
            TLLM_LOG_DEBUG("Mismatch value. Position of logits: %d, expected value: %f, output value: %f", i,
                gtLogitsPtr[i], outputLogitsPtr[i]);
            allMatch = false;
            errorNumber++;
            if (errorNumber == 10)
            {
                break;
            }
        }
    }
    return allMatch;
}

bool compareLogits(ITensor const& groundTruthLogits, ITensor const& outputLogits, float atol, float rtol)
{
    EXPECT_EQ(groundTruthLogits.getDataType(), outputLogits.getDataType());
    switch (groundTruthLogits.getDataType())
    {
    case nvinfer1::DataType::kFLOAT: return invokeCompareLogits<float>(groundTruthLogits, outputLogits, atol, rtol);
    case nvinfer1::DataType::kHALF: return invokeCompareLogits<half>(groundTruthLogits, outputLogits, atol, rtol);
    default: TLLM_THROW("Unsupported data type");
    }
}

std::tuple<SizeType32, SizeType32> getRequestGivenInputIdxLength(
    std::uint64_t requestId, SizeType32 nbGivenInputs, std::vector<SizeType32> const& givenInputLengths)
{
    auto const givenInputIdx = requestId % nbGivenInputs;
    auto const inputLength = givenInputLengths.at(givenInputIdx);
    return {givenInputIdx, inputLength};
}

std::tuple<std::vector<SizeType32>, SizeType32, SizeType32> getGivenInputLengths(
    ITensor const& givenInput, SizeType32 padId)
{
    auto const& inputShape = givenInput.getShape();
    auto const nbGivenInputs = static_cast<SizeType32>(inputShape.d[0]);
    auto const maxInputLength = static_cast<SizeType32>(inputShape.d[1]);
    auto const* const givenInputData = tr::bufferCast<TokenIdType const>(givenInput);

    std::vector<SizeType32> givenInputLengths(nbGivenInputs);
    for (SizeType32 i = 0; i < nbGivenInputs; ++i)
    {
        auto const* const seqBegin = givenInputData + i * maxInputLength;
        auto const* const it = std::find(seqBegin, seqBegin + maxInputLength, padId);
        givenInputLengths[i] = std::distance(seqBegin, it);
    }

    return {givenInputLengths, nbGivenInputs, maxInputLength};
}

std::vector<executor::TokenIdType> createConsecutiveTokenSequence(
    tr::SizeType32 length, tr::SizeType32 vocabSize, tr::TokenIdType firstTokenId)
{
    auto result = std::vector<executor::TokenIdType>(static_cast<size_t>(length), 0);
    std::iota(result.begin(), result.end(), firstTokenId);
    std::transform(result.begin(), result.end(), result.begin(), [&](auto const i) { return i % vocabSize; });
    return result;
}

TestData TestData::loadTestData(BeamResult const& beamResults, ITensor const& givenInput, SizeType32 const maxBeamWidth,
    tr::BufferManager& manager, executor::OutputConfig const& outConfig, ModelIds const& modelIds)
{
    auto const [givenInputLengths, nbGivenInputs, maxInputLength] = getGivenInputLengths(givenInput, modelIds.padId);
    auto const& [beamWidth, resultsFile, contextLogitsFile, genLogitsFile, cumLogProbsFile, logProbsFile] = beamResults;

    TestData testData{nbGivenInputs, beamWidth};
    testData.expectedOutputIds = tr::utils::loadNpy(manager, resultsFile.string(), tr::MemoryType::kCPU);

    auto const& outputShape = testData.expectedOutputIds->getShape();
    EXPECT_EQ(outputShape.nbDims, 2);
    EXPECT_EQ(nbGivenInputs * beamWidth, outputShape.d[0]);
    testData.maxSeqLen = static_cast<SizeType32>(outputShape.d[1]);
    EXPECT_LE(maxInputLength, testData.maxSeqLen);
    EXPECT_LE(beamWidth, maxBeamWidth);

    auto const maxNewTokens = testData.maxSeqLen - maxInputLength;

    testData.endIds.insert(testData.endIds.end(), nbGivenInputs, modelIds.endId);

    if (outConfig.returnContextLogits && beamWidth == 1)
    {
        testData.loadContextLogits(contextLogitsFile, givenInputLengths, manager);
    }
    if (outConfig.returnGenerationLogits && beamWidth == 1)
    {
        testData.loadGenerationLogits(genLogitsFile, manager);
    }
    if (outConfig.returnLogProbs && beamWidth == 1)
    {
        testData.loadLogProbs(cumLogProbsFile, logProbsFile, manager);
    }

    for (SizeType32 inputIdx = 0; inputIdx < nbGivenInputs; ++inputIdx)
    {
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            SizeType32 expectedLen = givenInputLengths[inputIdx] + maxNewTokens;
            testData.expectedOutputLengths[inputIdx * beamWidth + beam] = expectedLen;
        }
    }

    return testData;
}

void TestData::verifyOutput(std::unordered_map<SizeType32, std::vector<executor::BeamTokens>> const& resultTokens,
    std::vector<SizeType32> const& givenInputLengths, bool streaming, bool excludeInputFromOutput,
    FlakyTestInfo flakyTestInfo, bool isSpeculativeDecoding, SizeType32 reqBeamWidth, SizeType32 numReturnSequences,
    bool isNonGreedySampling)
{
    for (auto const& [batchId, beamTokens] : resultTokens)
    {
        for (auto seqIdx = 0; seqIdx < numReturnSequences; seqIdx++)
        {
            auto const& tokens = beamTokens.at(seqIdx);
            auto const inputLength = givenInputLengths.at(batchId);
            SizeType32 const numReturnBeams = tokens.size();
            auto const* const expectedOutputData = tr::bufferCast<TokenIdType const>(*this->expectedOutputIds);
            auto const expectedOutputLengths = this->expectedOutputLengths;
            auto const endId = this->endIds[batchId];
            auto const maxSeqLen = this->maxSeqLen;

            for (SizeType32 beam = 0; beam < numReturnBeams; ++beam)
            {
                bool isFlaky = flakyTestInfo.batchIdBeams.count(std::make_pair(batchId, beam));
                if (isFlaky)
                {
                    TLLM_LOG_WARNING("Disabling token comparison for batchId %d beam %d, test if flaky", batchId, beam);
                }

                auto const expectInputOutputLength
                    = expectedOutputLengths[batchId * reqBeamWidth + beam]; // Ground truth output length
                auto expectedOutputLength
                    = expectInputOutputLength - inputLength;                // Number of new generated output tokens

                bool inputNotIncluded = (streaming || excludeInputFromOutput);
                bool anyMismatch = false;
                auto predictedTokens = tokens.at(beam);
                // Remove the prompt
                if (!inputNotIncluded)
                {
                    predictedTokens.erase(predictedTokens.begin(), predictedTokens.begin() + inputLength);
                }

                if (!isNonGreedySampling)
                {
                    EXPECT_EQ(predictedTokens.size(), expectedOutputLength)
                        << "b: " << batchId << " seq: " << seqIdx << " beam: " << beam;
                }

                auto numPredTokens = static_cast<SizeType32>(predictedTokens.size());

                if (isSpeculativeDecoding)
                {
                    // WAR to ensure bulk execution of spec decoding.
                    // We hope that no request in batch can finish 2x faster than any other request.
                    // For the cases when BS < 8, some predicted tokens are mismatched to reference data.
                    numPredTokens /= 2;
                }

                for (auto i = 0; i < numPredTokens; ++i)
                {
                    // Use the expected data for that beamWidth
                    auto const expectIndex = tc::flat_index3(batchId, beam, inputLength + i, reqBeamWidth, maxSeqLen);
                    auto const expectedToken = expectedOutputData[expectIndex];
                    if (expectedToken == endId)
                    {
                        // TODO: can not find the error when (expectedToken == endId) && (predictedToken != endId)
                        break;
                    }
                    auto const predictedToken = predictedTokens.at(i);
                    if (!isFlaky && !isNonGreedySampling)
                    {
                        EXPECT_EQ(predictedToken, expectedToken)
                            << "b: " << batchId << " seq: " << seqIdx << " beam: " << beam << " i: " << i;
                    }
                    anyMismatch |= (predictedToken != expectedToken);
                }
                if (!isFlaky && !isNonGreedySampling)
                {
                    EXPECT_FALSE(anyMismatch) << "b: " << batchId << " seq: " << seqIdx << " beam: " << beam;
                }
                else if (isNonGreedySampling)
                {
                    EXPECT_TRUE(anyMismatch) << "b: " << batchId << " seq: " << seqIdx << " beam: " << beam;
                }
            }
        }
    }
}

void TestData::verifyLogProbs(bool computeLogProbs, bool streaming, bool excludeInputFromOutput, SizeType32 inputLength,
    SizeType32 beamWidth, executor::BeamTokens const& beamTokens,
    std::optional<executor::VecLogProbs> const& cumLogProbs,
    std::optional<std::vector<executor::VecLogProbs>> const& logProbs, SizeType32 batchId, FlakyTestInfo flakyTestInfo)
{
    auto expectedCumLogProbs = this->expectedCumLogProbs[batchId];
    auto expectedLogProbs = this->expectedLogProbs[batchId];
    auto const expectedOutputLengths = this->expectedOutputLengths;
    auto const numReturnBeams = beamTokens.size();

    if (computeLogProbs)
    {
        EXPECT_TRUE(cumLogProbs.has_value()) << "bid: " << batchId;
        EXPECT_TRUE(logProbs.has_value()) << "bid: " << batchId;
        EXPECT_EQ(cumLogProbs.value().size(), numReturnBeams) << "bid: " << batchId;
        EXPECT_EQ(logProbs.value().size(), numReturnBeams) << "bid: " << batchId;

        bool removeInput = !excludeInputFromOutput && !streaming;

        for (SizeType32 beam = 0; beam < numReturnBeams; ++beam)
        {
            bool isFlaky = flakyTestInfo.batchIdBeams.count(std::make_pair(batchId, beam));
            if (isFlaky)
            {
                TLLM_LOG_WARNING("Disabling token comparison for batchId %d beam %d, test if flaky", batchId, beam);
            }

            auto expectedOutputLength = expectedOutputLengths[batchId * beamWidth + beam];
            expectedOutputLength -= inputLength;

            auto numPredTokens = logProbs.value().at(beam).size();
            // Check shape
            EXPECT_EQ(numPredTokens, beamTokens.at(beam).size() - (removeInput ? inputLength : 0))
                << "bid: " << batchId << " beam: " << beam;

            // If beamWidth == 1, compare log probs against python runtime
            if (beamWidth == 1)
            {
                auto* const reqExpectedCumLogProbs = tr::bufferCast<float>(*expectedCumLogProbs);
                // Only check cumLogProbs for the last generated token
                if (numPredTokens == expectedOutputLength && !isFlaky)
                {
                    EXPECT_TRUE(almostEqual(reqExpectedCumLogProbs[beam], cumLogProbs.value().at(beam), 2e-1, 5e-2))
                        << "expectedCumLogProbs : " << reqExpectedCumLogProbs[beam]
                        << " cumlogProbs : " << cumLogProbs.value().at(beam);
                }

                auto expectedLogProbsBeam = std::shared_ptr(tr::ITensor::slice(expectedLogProbs, beam, 1));
                expectedLogProbsBeam->squeeze(0);
                auto* const reqExpectedLogProbs = tr::bufferCast<float>(*expectedLogProbsBeam);
                for (auto i = 0; i < numPredTokens; ++i)
                {
                    if (!isFlaky)
                    {
                        EXPECT_TRUE(
                            almostEqual(reqExpectedLogProbs[inputLength + i], logProbs.value()[beam][i], 5e-2, 5e-2))
                            << "expectedLogProbs : " << reqExpectedLogProbs[inputLength + i]
                            << " logProbs : " << logProbs.value()[beam][i];
                    }
                }
            }
        }
    }
    else
    {
        EXPECT_FALSE(cumLogProbs.has_value()) << "bid: " << batchId;
        EXPECT_FALSE(logProbs.has_value()) << "bid: " << batchId;
    }
}

void TestData::validateContextLogits(bool getContextLogits, SizeType32 inputLength, SizeType32 beamWidth,
    std::optional<executor::Tensor> const& contextLogits, SizeType32 vocabSizePadded, SizeType32 batchId, float atol,
    float rtol)
{
    if (getContextLogits)
    {
        EXPECT_TRUE(contextLogits.has_value()) << "bid: " << batchId;
        EXPECT_EQ(contextLogits.value().getShape().size(), 2);
        EXPECT_EQ(contextLogits.value().getShape()[0], inputLength);
        EXPECT_EQ(contextLogits.value().getShape()[1], vocabSizePadded);
        auto const expectedContextLogits = this->expectedContextLogits[batchId];

        if (beamWidth == 1)
        {
            cudaDeviceSynchronize(); // Make sure the logits copy is complete.
            EXPECT_TRUE(compareLogits(
                *expectedContextLogits, *(executor::detail::toITensor(contextLogits.value())), atol, rtol));
        }
    }
    else
    {
        EXPECT_FALSE(contextLogits.has_value()) << "bid: " << batchId;
    }
}

void TestData::validateGenerationLogits(bool getGenLogits, bool isFinal, bool streaming, bool excludeInputFromOutput,
    SizeType32 inputLength, SizeType32 maxOutputLen, SizeType32 beamWidth, executor::BeamTokens const& beamTokens,
    std::optional<executor::Tensor> const& genLogits, SizeType32 vocabSizePadded, SizeType32 batchId,
    bool const returnAllGeneratedTokens, float atol, float rtol)
{
    auto const numReturnBeams = beamTokens.size();

    if (getGenLogits)
    {
        EXPECT_TRUE(genLogits.has_value()) << "bid: " << batchId;
        EXPECT_EQ(genLogits.value().getShape().size(), 3);

        // Expected generation logits
        auto const& expectedGenerationLogits
            = this->expectedGenerationLogits[batchId]; // [maxOutputLen, vocabSizePadded]
        // Output generation logits
        // 1. non-streaming: [beamWidth, maxOutputLen, vocabSizePadded]
        // 2. streaming: [maxOutputLen (or 1), beamWidth, vocabSizePadded]
        auto const& outputGenerationLogits = executor::detail::toITensor(genLogits.value());

        if (streaming)
        {
            EXPECT_EQ(genLogits.value().getShape()[1], numReturnBeams);
            EXPECT_EQ(beamWidth, 1); // Only support streaming && beamWidth == 1

            SizeType32 const beamIdx = 0;
            bool removeInput = !excludeInputFromOutput && !streaming;
            // If returnAllGeneratedTokens, will contain duplicate tokens
            auto const& numPredTokens = beamTokens.at(beamIdx).size() - (removeInput ? inputLength : 0);

            SizeType32 numGeneratedToken = genLogits.value().getShape()[0];
            if (returnAllGeneratedTokens)
            {
                EXPECT_EQ(numGeneratedToken, numPredTokens);
            }
            else
            {
                EXPECT_EQ(numGeneratedToken, 1);
            }
            SizeType32 sliceOffset = returnAllGeneratedTokens ? 0 : numPredTokens - 1;

            auto const& expectedGenerationLogitsSlice
                = std::shared_ptr(ITensor::slice(expectedGenerationLogits, sliceOffset,
                    numGeneratedToken)); // [numGeneratedToken, vocabSizePadded]

            cudaDeviceSynchronize();     // Make sure the logits copy is complete.
            EXPECT_TRUE(compareLogits(*expectedGenerationLogitsSlice, *outputGenerationLogits, atol, rtol));
        }
        else
        {
            // Non-streaming
            EXPECT_EQ(genLogits.value().getShape()[0], numReturnBeams);
            EXPECT_EQ(genLogits.value().getShape()[1], maxOutputLen);

            if (isFinal && beamWidth == 1)
            {
                cudaDeviceSynchronize(); // Make sure the logits copy is complete.
                EXPECT_TRUE(compareLogits(*expectedGenerationLogits, *outputGenerationLogits, atol, rtol));
            }
        }
        EXPECT_EQ(genLogits.value().getShape()[2], vocabSizePadded);
    }
    else
    {
        EXPECT_FALSE(genLogits.has_value()) << "bid: " << batchId;
    }
}

} // namespace tensorrt_llm::testing
