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

#pragma once

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cmath>
#include <filesystem>
#include <random>
#include <utility>
#include <vector>

namespace tensorrt_llm::testing
{
namespace fs = std::filesystem;
namespace tr = tensorrt_llm::runtime;

using tr::SizeType32;
using tr::TokenIdType;
using tr::ITensor;
using tr::MemoryType;

class ModelIds
{
public:
    ModelIds() = default;

    constexpr ModelIds(TokenIdType endId, TokenIdType padId)
        : endId{endId}
        , padId{padId}
    {
    }

    TokenIdType endId{};
    TokenIdType padId{};
};

class BeamResult
{
public:
    explicit BeamResult(SizeType32 beamWidth)
        : beamWidth{beamWidth} {};

    BeamResult(SizeType32 beamWidth, fs::path resultsFile, fs::path contextLogitsFile, fs::path genLogitsFile,
        fs::path cumLogProbsFile, fs::path logProbsFile)
        : beamWidth{beamWidth}
        , resultsFile{std::move(resultsFile)}
        , contextLogitsFile{std::move(contextLogitsFile)}
        , genLogitsFile{std::move(genLogitsFile)}
        , cumLogProbsFile{std::move(cumLogProbsFile)}
        , logProbsFile{std::move(logProbsFile)} {};

    SizeType32 beamWidth;
    fs::path resultsFile;

    fs::path contextLogitsFile;
    fs::path genLogitsFile;

    fs::path cumLogProbsFile;
    fs::path logProbsFile;
};

using BeamResults = std::vector<BeamResult>;

class TestData
{
public:
    explicit TestData(SizeType32 nbGivenInputs, SizeType32 beamWidth)
        : nbGivenInputs{nbGivenInputs}
        , beamWidth{beamWidth}
    {
        expectedOutputLengths.resize(nbGivenInputs * beamWidth);

        draftTokens.resize(nbGivenInputs);
        draftLogits.resize(nbGivenInputs);
        acceptedDraftTokensLengths.resize(nbGivenInputs);
        expectedGenerationLogits.resize(nbGivenInputs);
        expectedContextLogits.resize(nbGivenInputs);
        expectedCumLogProbs.resize(nbGivenInputs);
        expectedLogProbs.resize(nbGivenInputs);
    }

    void loadLogProbs(fs::path const& cumLogProbsFile, fs::path const& logProbsFile, tr::BufferManager const& manager);

    void loadContextLogits(fs::path const& contextLogitsFile, std::vector<SizeType32> const& givenInputLengths,
        tr::BufferManager const& manager);
    void loadGenerationLogits(fs::path const& genLogitsFile, tr::BufferManager const& manager);

    void makeDraft(SizeType32 maxDraftTokens, bool acceptDraftByLogits, fs::path const& genLogitsFile,
        std::vector<SizeType32> const& givenInputLengths, tr::BufferManager const& manager);

    SizeType32 nbGivenInputs{};
    SizeType32 beamWidth{};
    SizeType32 maxSeqLen{};
    ITensor::SharedPtr expectedOutputIds;
    std::vector<SizeType32> expectedOutputLengths;
    std::vector<TokenIdType> endIds;
    std::vector<tensorrt_llm::executor::VecTokens> draftTokens;
    std::vector<ITensor::SharedPtr> draftLogits;
    std::vector<SizeType32> acceptedDraftTokensLengths;
    std::vector<ITensor::SharedPtr> expectedGenerationLogits;
    std::vector<ITensor::SharedPtr> expectedContextLogits;
    std::vector<ITensor::SharedPtr> expectedCumLogProbs;
    std::vector<ITensor::SharedPtr> expectedLogProbs;
};

inline bool almostEqual(float a, float b, float atol = 1e-2, float rtol = 1e-3)
{
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (std::isnan(a) && std::isnan(b))
    {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

bool compareLogits(ITensor const& groundTruthLogits, ITensor const& outputLogits, float atol = 1e-2, float rtol = 1e-3);

std::tuple<SizeType32, SizeType32> getRequestGivenInputIdxLength(
    std::uint64_t requestId, SizeType32 nbGivenInputs, std::vector<SizeType32> const& givenInputLengths);

std::tuple<std::vector<SizeType32>, SizeType32, SizeType32> getGivenInputLengths(
    ITensor const& givenInput, SizeType32 padId);

/// @brief Generates a vector of floating point values summing to 1, that can be used as logits.
///
/// @tparam TEngine The type of the random engine.
/// @tparam TLogits The type of floating point values.
/// @param vocabSize The vocabulary size, i.e. the size of the vector.
/// @param engine A random engine.
/// @return std::vector<TLogits> A vector of floating point values, summing to 1.
template <typename TEngine, typename TLogits>
std::vector<TLogits> randomLogits(runtime::SizeType32 vocabSize, TEngine* engine)
{
    if constexpr (std::disjunction_v<std::is_floating_point<TLogits>, std::is_same<TLogits, half>>)
    {
        // This algorithm ensures the resulting values sum to 1 by:
        // 1. Sampling in the interval 0..1
        // 2. Sorting the sampled values and adding a last value equal to 1
        // 3. Calculating the adjacent differences of the sorted values
        // Since the values are sorted and the last value is 1, we get that all the differences are positive and must
        // sum to 1. It can be proven recursively by seeing that the first value sums to itself, and the n-1 first
        // values must sum to the value at n, minus the difference between the n-th and n-1-th values.
        // It is also helpful to convince yourself of it with a quick drawing.
        auto distribution = std::uniform_real_distribution<float>(0, 1);
        std::vector<float> samples(vocabSize);
        samples.back() = 1.0;
        std::transform(samples.begin(), samples.end() - 1, samples.begin(),
            [&](auto const /*i*/) { return distribution(*engine); });
        std::sort(samples.begin(), samples.end() - 1);
        std::vector<float> result(vocabSize);
        std::adjacent_difference(samples.begin(), samples.end(), result.begin());
        if constexpr (std::is_same_v<TLogits, float>)
        {
            return result;
        }

        if constexpr (std::is_same_v<TLogits, half>)
        {
            std::vector<half> halfResults(vocabSize);
            std::transform(
                result.begin(), result.end(), halfResults.begin(), [&](auto const f) { return __float2half(f); });
            return halfResults;
        }
    }
    TLLM_THROW("Unsupported logits type.");
}

std::vector<tensorrt_llm::executor::TokenIdType> createConsecutiveTokenSequence(
    tensorrt_llm::runtime::SizeType32 length, tensorrt_llm::runtime::TokenIdType vocabLength);

} // namespace tensorrt_llm::testing
