/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace tensorrt_llm::executor
{

class Model;
class Serialization;

/// @brief Sampling configuration
class SamplingConfig
{
public:
    SamplingConfig(SizeType beamWidth = 1, std::optional<SizeType> topK = std::nullopt,
        std::optional<FloatType> topP = std::nullopt, std::optional<FloatType> topPMin = std::nullopt,
        std::optional<SizeType> topPResetIds = std::nullopt, std::optional<FloatType> topPDecay = std::nullopt,
        std::optional<RandomSeedType> randomSeed = std::nullopt, std::optional<FloatType> temperature = std::nullopt,
        std::optional<SizeType> minLength = std::nullopt,
        std::optional<FloatType> beamSearchDiversityRate = std::nullopt,
        std::optional<FloatType> repetitionPenalty = std::nullopt,
        std::optional<FloatType> presencePenalty = std::nullopt,
        std::optional<FloatType> frequencyPenalty = std::nullopt, std::optional<FloatType> lengthPenalty = std::nullopt,
        std::optional<SizeType> earlyStopping = std::nullopt);

    ~SamplingConfig();

    bool operator==(SamplingConfig const& other) const;

    [[nodiscard]] SizeType getBeamWidth() const;
    [[nodiscard]] std::optional<SizeType> getTopK() const;
    [[nodiscard]] std::optional<FloatType> getTopP() const;
    [[nodiscard]] std::optional<FloatType> getTopPMin() const;
    [[nodiscard]] std::optional<SizeType> getTopPResetIds() const;
    [[nodiscard]] std::optional<FloatType> getTopPDecay() const;
    [[nodiscard]] std::optional<RandomSeedType> getRandomSeed() const;
    [[nodiscard]] std::optional<FloatType> getTemperature() const;
    [[nodiscard]] std::optional<SizeType> getMinLength() const;
    [[nodiscard]] std::optional<FloatType> getBeamSearchDiversityRate() const;
    [[nodiscard]] std::optional<FloatType> getRepetitionPenalty() const;
    [[nodiscard]] std::optional<FloatType> getPresencePenalty() const;
    [[nodiscard]] std::optional<FloatType> getFrequencyPenalty() const;
    [[nodiscard]] std::optional<FloatType> getLengthPenalty() const;
    [[nodiscard]] std::optional<SizeType> getEarlyStopping() const;

private:
    friend class Serialization;
    SizeType mBeamWidth;
    std::optional<SizeType> mTopK;
    std::optional<FloatType> mTopP;
    std::optional<FloatType> mTopPMin;
    std::optional<SizeType> mTopPResetIds;
    std::optional<FloatType> mTopPDecay;
    std::optional<RandomSeedType> mRandomSeed;
    std::optional<FloatType> mTemperature;
    std::optional<SizeType> mMinLength;
    std::optional<FloatType> mBeamSearchDiversityRate;
    std::optional<FloatType> mRepetitionPenalty;
    std::optional<FloatType> mPresencePenalty;
    std::optional<FloatType> mFrequencyPenalty;
    std::optional<FloatType> mLengthPenalty;
    std::optional<SizeType> mEarlyStopping;
};

/// @brief  Configuration that controls the outputs of a Result
class OutputConfig
{
public:
    OutputConfig(bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
        bool excludeInputFromOutput = false);

    bool returnLogProbs;
    bool returnContextLogits;
    bool returnGenerationLogits;
    bool excludeInputFromOutput;
};

/// @brief Configuration for speculative decoding. Allows to include draft tokens, draft logits and specify acceptance
/// threshold
class SpeculativeDecodingConfig
{
public:
    explicit SpeculativeDecodingConfig(VecTokens tokens, std::optional<Tensor> logits = std::nullopt,
        std::optional<FloatType> acceptanceThreshold = std::nullopt);

    ~SpeculativeDecodingConfig();

    [[nodiscard]] VecTokens getTokens() const;
    [[nodiscard]] std::optional<Tensor> getLogits() const;
    [[nodiscard]] std::optional<FloatType> getAcceptanceThreshold() const;

private:
    friend class Serialization;
    VecTokens mTokens;
    std::optional<Tensor> mLogits;
    std::optional<FloatType> mAcceptanceThreshold;
};

/// @brief Configuration for prompt tuning
class PromptTuningConfig
{
public:
    /// @brief
    /// @param embeddingTable  The prompt embedding table. Data type must match model weights. Shape [vocabSize,
    /// hiddenSize]
    /// @param vocabSize
    PromptTuningConfig(Tensor embeddingTable);
    ~PromptTuningConfig();

    [[nodiscard]] Tensor getEmbeddingTable() const;

private:
    friend class Serialization;
    Tensor mEmbeddingTable;
};

/// @brief Configuration for LoRA
class LoraConfig
{
public:
    LoraConfig(Tensor weights, Tensor config);
    ~LoraConfig();

    [[nodiscard]] Tensor getWeights() const;
    [[nodiscard]] Tensor getConfig() const;

private:
    friend class Serialization;

    Tensor mWeights;
    Tensor mConfig;
};

/// @brief A class that holds information about the request
class Request
{
public:
    /// @brief
    /// @param inputTokenIds The input token ids
    /// @param maxNewTokens  The maximum number of tokens to generate
    /// @param streaming // Indicates if the responses should be streamed or not
    /// @param samplingConfig // The sampling configuration
    /// @param outputConfig // The output configuration
    /// @param endId // The end token id
    /// @param padId // The pad token id
    /// @param badWords // A list of bad words tokens. Each "word" can be composed of multiple tokens
    /// @param stopWords // A list of stop words tokens. Each "word" can be composed of multiple tokens
    /// @param embeddingBias // The embedding bias tensor. Expected type is kFP32 and shape is [vocab_size]
    /// @param speculativeDecodingConfig // The speculative decoding configuration
    /// @param pTuningConfig // The prompt tuning configuration
    /// @param loraConfig // The LoRA configuration
    Request(VecTokens inputTokenIds, SizeType maxNewTokens, bool streaming = false,
        SamplingConfig samplingConfig = SamplingConfig(), OutputConfig outputConfig = OutputConfig(),
        std::optional<SizeType> endId = std::nullopt, std::optional<SizeType> padId = std::nullopt,
        std::optional<std::list<VecTokens>> badWords = std::nullopt,
        std::optional<std::list<VecTokens>> stopWords = std::nullopt,
        std::optional<Tensor> embeddingBias = std::nullopt,
        std::optional<SpeculativeDecodingConfig> speculativeDecodingConfig = std::nullopt,
        std::optional<PromptTuningConfig> pTuningConfig = std::nullopt,
        std::optional<LoraConfig> loraConfig = std::nullopt);

    Request(Request const& other);
    Request(Request&& other) noexcept;
    Request& operator=(Request const& other);
    Request& operator=(Request&& other) noexcept;
    ~Request();

    [[nodiscard]] VecTokens getInputTokenIds() const;
    [[nodiscard]] SizeType getMaxNewTokens() const;
    [[nodiscard]] bool getStreaming() const;
    [[nodiscard]] SamplingConfig getSamplingConfig() const;
    [[nodiscard]] OutputConfig getOutputConfig() const;
    [[nodiscard]] std::optional<SizeType> getEndId() const;
    [[nodiscard]] std::optional<SizeType> getPadId() const;
    [[nodiscard]] std::optional<std::list<VecTokens>> getBadWords() const;
    [[nodiscard]] std::optional<std::list<VecTokens>> getStopWords() const;
    [[nodiscard]] std::optional<Tensor> getEmbeddingBias() const;
    [[nodiscard]] std::optional<SpeculativeDecodingConfig> getSpeculativeDecodingConfig() const;
    [[nodiscard]] std::optional<PromptTuningConfig> getPromptTuningConfig() const;
    [[nodiscard]] std::optional<LoraConfig> getLoraConfig() const;

    void setStreaming(bool streaming);
    void setSamplingConfig(SamplingConfig config);
    void setOutputConfig(OutputConfig outputConfig);
    void setEndId(SizeType endId);
    void setPadId(SizeType padId);
    void setBadWords(std::list<VecTokens> badWords);
    void setStopWords(std::list<VecTokens> stopWords);
    void setEmbeddingBias(Tensor);
    void setSpeculativeDecodingConfig(SpeculativeDecodingConfig specDecodingConfig);
    void setPromptTuningConfig(PromptTuningConfig pTuningConfig);
    void setLoraConfig(LoraConfig loraConfig);

private:
    friend class Serialization;
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Struct that holds the generation result
struct Result
{
    // Indicates if this is the final result for the request
    bool isFinal;

    /// @brief The output tokens for each beam
    BeamTokens outputTokenIds;

    std::optional<VecLogProbs> cumLogProbs;           // [beamSize]
    std::optional<std::vector<VecLogProbs>> logProbs; // [beamSize, seqLen]
    std::optional<Tensor> contextLogits;              // [promptLen, vocab_size_padded]
    std::optional<Tensor> generationLogits;           // [beam_size, mMaxNewTokens, vocab_size_padded]
};

/// @brief Class that holds either an error or a result
class Response
{
public:
    Response(IdType requestId, std::string errorMsg);
    Response(IdType requestId, Result Result);

    ~Response();
    Response(Response const& other);
    Response(Response&& other) noexcept;
    Response& operator=(Response const& other);
    Response& operator=(Response&& other) noexcept;

    // Get the id of the request for which this response was generated
    IdType getRequestId() const;

    // Indicates if this response has an error or not
    bool hasError() const;

    // Get the error msg for this response
    // Will throw an exception if hasError is false
    std::string getErrorMsg() const;

    // Get the result for this response
    // Will throw an exception if hasResult is true
    Result getResult() const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Configuration class for the scheduler
class SchedulerConfig
{
public:
    explicit SchedulerConfig(SchedulerPolicy policy = SchedulerPolicy::kGUARANTEED_NO_EVICT);
    ~SchedulerConfig();

    [[nodiscard]] SchedulerPolicy getPolicy() const;

private:
    SchedulerPolicy mPolicy;
};

/// @brief Configuration class for the KV cache
class KvCacheConfig
{
public:
    KvCacheConfig(bool enableBlockReuse = false, std::optional<SizeType> maxTokens = std::nullopt,
        std::optional<SizeType> maxAttentionWindow = std::nullopt,
        std::optional<SizeType> sinkTokenLength = std::nullopt,
        std::optional<FloatType> freeGpuMemoryFraction = std::nullopt, bool useUvm = false);

    [[nodiscard]] bool getEnableBlockReuse() const;
    [[nodiscard]] std::optional<SizeType> getMaxTokens() const;
    [[nodiscard]] std::optional<SizeType> getMaxAttentionWindow() const;
    [[nodiscard]] std::optional<SizeType> getSinkTokenLength() const;
    [[nodiscard]] std::optional<FloatType> getFreeGpuMemoryFraction() const;
    [[nodiscard]] bool getUseUvm() const;

private:
    bool mEnableBlockReuse;
    std::optional<SizeType> mMaxTokens;
    std::optional<SizeType> mMaxAttentionWindow;
    std::optional<SizeType> mSinkTokenLength;
    std::optional<FloatType> mFreeGpuMemoryFraction;
    bool mUseUvm;
};

SizeType const kDefaultIterStatsMaxIterations = 1000;

/// @brief A configuration class for the parallel execution parameters
///        Currently only supports commType = CommunicationType::kMPI
class ParallelConfig
{
public:
    /// @brief Constructor
    /// @param commType The communication type. See CommunicationType.
    /// @param commMode The communication mode. See CommunicationMode.
    /// @param deviceIds The IDs of the GPUs involved in the execution of the model
    /// @param participantIds The participant IDs (MPI ranks if commType == kMPI) involved in the execution of the
    /// model. The first participant is considered to be the leader.
    ParallelConfig(CommunicationType commType = CommunicationType::kMPI,
        CommunicationMode commMode = CommunicationMode::kLEADER,
        std::optional<std::vector<SizeType>> deviceIds = std::nullopt,
        std::optional<std::vector<SizeType>> participantIds = std::nullopt);
    ~ParallelConfig();

    [[nodiscard]] CommunicationType getCommunicationType() const;
    [[nodiscard]] CommunicationMode getCommunicationMode() const;
    [[nodiscard]] std::optional<std::vector<SizeType>> getDeviceIds() const;
    [[nodiscard]] std::optional<std::vector<SizeType>> getParticipantIds() const;

    void setCommunicationType(CommunicationType type);
    void setCommunicationMode(CommunicationMode mode);
    void setDeviceIds(std::vector<SizeType> deviceIds);
    void setParticipantIds(std::vector<SizeType> participantIds);

private:
    CommunicationType mCommType;
    CommunicationMode mCommMode;
    std::optional<std::vector<SizeType>> mDeviceIds;
    std::optional<std::vector<SizeType>> mParticipantIds;
};

/// @brief Configuration class for the model executor
class ExecutorConfig
{
public:
    ExecutorConfig(SizeType maxBeamWidth = 1, SchedulerConfig schedulerConfig = SchedulerConfig(),
        KvCacheConfig kvCacheConfig = KvCacheConfig(), bool enableChunkedContext = false, bool normalizeLogProbs = true,
        bool enableTrtOverlap = false, SizeType iterStatsMaxIterations = kDefaultIterStatsMaxIterations,
        BatchingType batchingType = BatchingType::kINFLIGHT,
        std::optional<ParallelConfig> parallelConfig = std::nullopt);

    [[nodiscard]] SizeType getMaxBeamWidth() const;
    [[nodiscard]] SchedulerConfig getSchedulerConfig() const;
    [[nodiscard]] KvCacheConfig getKvCacheConfig() const;
    [[nodiscard]] bool getEnableChunkedContext() const;
    [[nodiscard]] bool getNormalizeLogProbs() const;
    [[nodiscard]] bool getEnableTrtOverlap() const;
    [[nodiscard]] SizeType getIterStatsMaxIterations() const;
    [[nodiscard]] BatchingType getBatchingType() const;
    [[nodiscard]] std::optional<ParallelConfig> getParallelConfig() const;

    void setMaxBeamWidth(SizeType maxBeamWidth);
    void setSchedulerConfig(SchedulerConfig schedulerConfig);
    void setKvCacheConfig(KvCacheConfig kvCacheConfig);
    void setEnableChunkedContext(bool enableChunkedContext);
    void setNormalizeLogProbs(bool normalizeLogProbs);
    void setEnableTrtOverlap(bool enableTrtOverlap);
    void setIterStatsMaxIterations(SizeType iterStatsMaxIterations);
    void setBatchingType(BatchingType batchingType);
    void setParallelConfig(ParallelConfig parallelConfig);

private:
    SizeType mMaxBeamWidth;
    SchedulerConfig mSchedulerConfig;
    KvCacheConfig mKvCacheConfig;
    bool mEnableChunkedContext;
    bool mNormalizeLogProbs;
    bool mEnableTrtOverlap;
    SizeType mIterStatsMaxIterations;
    BatchingType mBatchingType;
    std::optional<ParallelConfig> mParallelConfig;
};

/// @brief The executor is responsible for receiving new requests and sending responses, and running the inference
class Executor
{
    using RequestPtr = std::shared_ptr<Request>;

public:
    /// @brief
    /// @param modelPath Path to the folder that defines the model to run
    /// @param modelType The type of model
    /// @param executorConfig The configuration for the executor
    /// @param comm An optional inter-process communicator configuration
    Executor(std::filesystem::path const& modelPath, ModelType modelType, ExecutorConfig executorConfig);

    Executor(std::vector<uint8_t> const& engineBuffer, std::string const& jsonConfigStr, ModelType modelType,
        ExecutorConfig executorConfig);

    Executor(std::shared_ptr<Model> model, ExecutorConfig executorConfig);

    ~Executor();

    /// @brief Enqueue a new request
    /// @param request The LLM request which contains input tokens and request parameters
    /// @return A unique id that identifies the request
    IdType enqueueRequest(Request request);

    /// @brief Enqueue a batch of request
    std::vector<IdType> enqueueRequests(std::vector<Request> requests);

    /// @brief Await for ready responses
    /// @param id An optional request id. If not specified, responses for any request can be returned
    /// @param timeout The maximum time to wait for new responses
    /// @return A vector of responses
    std::vector<Response> awaitResponses(
        std::optional<IdType> id = std::nullopt, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

    /// @brief Get the number of ready responses
    /// @param id The request id
    /// @return The number of ready responses
    SizeType getNumResponsesReady(std::optional<IdType> id = std::nullopt);

    /// @brief Cancel the request with provided request id
    /// @param id The request id for which to cancel the response
    void cancelRequest(IdType id);

    /// @brief  Signals the server to shutdown
    ///         This call is blocking. Only returns when all requests have terminated or timeout has been reached
    void shutdown();

    /// @brief  Returns the per-iterations statistics computed since last call to getLatestIterationStats
    ///         Contains at most iterStatsMaxIterations iterations
    ///         Will block until stats for at least one iteration are available
    /// TODO: Should we use a class for iterationStats, i.e. std::deque<IterationStats>
    /// @return
    std::deque<std::string> getLatestIterationStats();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace tensorrt_llm::executor
