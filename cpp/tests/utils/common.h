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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cmath>
#include <filesystem>
#include <random>
#include <set>
#include <string>
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

auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";

auto const ENGINE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const GPT_MODEL_PATH = ENGINE_PATH / "gpt2";
auto const LLAMA_MODEL_PATH = ENGINE_PATH / "Llama-3.2-1B";
auto const MEDUSA_MODEL_PATH = ENGINE_PATH / "vicuna-7b-medusa";
auto const CHATGLM_MODEL_PATH = ENGINE_PATH / "chatglm-6b";
auto const CHATGLM2_MODEL_PATH = ENGINE_PATH / "chatglm2-6b";
auto const CHATGLM3_MODEL_PATH = ENGINE_PATH / "chatglm3-6b";
auto const GLM_MODEL_PATH = ENGINE_PATH / "glm-10b";
auto const ENC_DEC_ENGINE_BASE = TEST_RESOURCE_PATH / "models/enc_dec/trt_engines";

auto const DATA_PATH = TEST_RESOURCE_PATH / "data";
auto const GPT_DATA_PATH = DATA_PATH / "gpt2";
auto const GPT_XGRAMMAR_TOKENIZER_INFO_PATH = GPT_DATA_PATH / "xgrammar_tokenizer_info.json";
auto const LLAMA_DATA_PATH = DATA_PATH / "Llama-3.2-1B";
auto const LLAMA_XGRAMMAR_TOKENIZER_INFO_PATH = LLAMA_DATA_PATH / "xgrammar_tokenizer_info.json";
auto const MEDUSA_DATA_PATH = DATA_PATH / "vicuna-7b-medusa";
auto const CHATGLM_DATA_PATH = DATA_PATH / "chatglm-6b";
auto const CHATGLM2_DATA_PATH = DATA_PATH / "chatglm2-6b";
auto const CHATGLM3_DATA_PATH = DATA_PATH / "chatglm3-6b";
auto const GLM_DATA_PATH = DATA_PATH / "glm-10b";
auto const ENC_DEC_DATA_BASE = DATA_PATH / "enc_dec";

auto constexpr T5_NAME = "t5-small";
auto constexpr BART_NAME = "bart-large-cnn";
auto constexpr LANGUAGE_ADAPTER_NAME = "language_adapter-enc_dec_language_adapter";

class PathUtil
{
public:
    static std::string EXECUTOR_WORKER_PATH()
    {
        return (std::filesystem::path{TOP_LEVEL_DIR} / "cpp/build/tensorrt_llm/executor_worker/executorWorker")
            .string();
    }

    // model paths
    static std::string FP16_GPT_ATTENTION_PACKED_DIR();
    static std::string FP16_GPT_ATTENTION_PACKED_PAGED_DIR();
    static std::string FP16_GPT_LORA_DIR();
    static std::string FP16_GPT_ATTENTION_PACKED_PAGED_DRAFT_TOKENS_DIR();
    static std::string FP16_GPT_ATTENTION_PACKED_PAGED_GATHER_DIR();
    static std::string FP16_PLUGIN_PACKED_PAGED_RESULT_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_LONG_RESULT_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_GATHER_RESULT_FILE();
    // logits
    static std::string FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_GATHER_CUM_LOG_PROBS_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_GATHER_LOG_PROBS_FILE();
    // results
    static std::string FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP1_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_RESULT_TP4_PP1_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP2_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP4_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_RESULT_TP1_PP2_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_RESULT_TP2_PP1_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_CONTEXT_LOGITS_TP4_PP1_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_GENERATION_LOGITS_TP4_PP1_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_CUM_LOG_PROBS_TP4_PP1_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_LOG_PROBS_TP4_PP1_FILE();
    // GptExecutorTest.GenerationLogitsEarlyStop requires to use context_fmha_fp32_acc flag in runtime for better
    // accuracy
    static std::string FP16_PLUGIN_PACKED_PAGED_GATHER_CONTEXTFMHAFP32ACC_RESULT_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_CONTEXTFMHAFP32ACC_GENERATION_LOGITS_FILE();
    static std::string FP16_PLUGIN_PACKED_PAGED_CONTEXTFMHAFP32ACC_CONTEXT_LOGITS_FILE();
};

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

struct FlakyTestInfo
{
    // Pair of batch ID + beam which are flaky
    std::set<std::pair<SizeType32, SizeType32>> batchIdBeams;
};

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

    static TestData loadTestData(BeamResult const& beamResults, ITensor const& givenInput, SizeType32 maxBeamWidth,
        tr::BufferManager& manager, executor::OutputConfig const& outConfig, ModelIds const& modelIds);

    void verifyOutput(std::unordered_map<SizeType32, std::vector<executor::BeamTokens>> const& resultTokens,
        std::vector<SizeType32> const& givenInputLengths, bool streaming, bool excludeInputFromOutput,
        FlakyTestInfo flakyTestInfo, bool isSpeculativeDecoding, SizeType32 reqBeamWidth, SizeType32 numReturnSequences,
        bool isNonGreedySampling);

    void verifyLogProbs(bool computeLogProbs, bool streaming, bool excludeInputFromOutput, SizeType32 inputLength,
        SizeType32 beamWidth, executor::BeamTokens const& beamTokens,
        std::optional<executor::VecLogProbs> const& cumLogProbs,
        std::optional<std::vector<executor::VecLogProbs>> const& logProbs, SizeType32 batchId,
        FlakyTestInfo flakyTestInfo);

    void validateContextLogits(bool getContextLogits, SizeType32 inputLength, SizeType32 beamWidth,
        std::optional<executor::Tensor> const& contextLogits, SizeType32 vocabSizePadded, SizeType32 batchId,
        float atol = 1e-2, float rtol = 1e-3);

    void validateGenerationLogits(bool getGenLogits, bool isFinal, bool streaming, bool excludeInputFromOutput,
        SizeType32 inputLength, SizeType32 maxOutputLen, SizeType32 beamWidth, executor::BeamTokens const& beamTokens,
        std::optional<executor::Tensor> const& genLogits, SizeType32 vocabSizePadded, SizeType32 batchId,
        bool returnAllGeneratedTokens, float atol = 1e-2, float rtol = 1e-3);

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
    tr::SizeType32 length, tr::SizeType32 vocabSize, tr::TokenIdType firstTokenId);

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;

    /// Construct`or
    GpuTimer()
        : _stream_id(0)
    {
        TLLM_CUDA_CHECK(cudaEventCreate(&_start));
        TLLM_CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        TLLM_CUDA_CHECK(cudaEventDestroy(_start));
        TLLM_CUDA_CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        TLLM_CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        TLLM_CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        TLLM_CUDA_CHECK(cudaEventSynchronize(_stop));
        TLLM_CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

} // namespace tensorrt_llm::testing
