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
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <gtest/gtest.h>

namespace tensorrt_llm::testing
{
namespace fs = std::filesystem;
namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

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
    ITensor::SharedPtr expectedGenerationLogitsPtr;
    if (acceptDraftByLogits)
    {
        TLLM_CHECK_WITH_INFO(
            genLogitsFile != "", "Testing Draft token, but missing the expected generation logits results file.");
        expectedGenerationLogitsPtr
            = std::shared_ptr(tr::utils::loadNpy(manager, genLogitsFile.string(), MemoryType::kCPU));
    }

    auto* const expectedOutputData = tr::bufferCast<TokenIdType>(*expectedOutputIds);
    for (SizeType32 bi = 0; bi < nbGivenInputs; ++bi)
    {
        auto const endId = endIds.at(bi);
        for (SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto const draftLen
                = std::rand() % std::min((maxSeqLen - (givenInputLengths.at(bi) + 1)), maxDraftTokens) + 1;
            auto acceptedLen = std::rand() % draftLen;

            if (acceptDraftByLogits)
            {
                auto expectedLogitBatchSlice = std::shared_ptr(ITensor::slice(expectedGenerationLogitsPtr, bi, 1));
                expectedLogitBatchSlice->squeeze(0); // bs
                expectedLogitBatchSlice->squeeze(0); // beam
                auto expectedLogitBatchStepSlice
                    = std::shared_ptr(ITensor::slice(expectedLogitBatchSlice, 1, draftLen));
                auto expectedLogitBatchStepView = ITensor::view(expectedLogitBatchStepSlice,
                    ITensor::makeShape({draftLen, 1, 1, expectedLogitBatchStepSlice->getShape().d[1]}));
                draftLogits.at(bi) = manager.copyFrom(*expectedLogitBatchStepView, MemoryType::kCPU);
            }

            for (SizeType32 si = 0; si < draftLen; ++si)
            {
                auto const draftIndex
                    = tc::flat_index3(bi, beam, givenInputLengths.at(bi) + si + 1, beamWidth, maxSeqLen);
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

            auto const expectedLen = expectedOutputLengths.at(bi * beamWidth + beam);
            TLLM_CHECK(expectedLen > 0);
            expectedOutputLengths[bi * beamWidth + beam]
                = std::min(expectedLen, (givenInputLengths.at(bi) + 1) + acceptedLen + 1);
        }
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

} // namespace tensorrt_llm::testing
