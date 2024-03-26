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
#include <vector>

namespace tensorrt_llm::executor
{

class Model;
class Serialization;

/// @brief Sampling configuration
class SamplingConfig
{
public:
    /// @brief Constructor for SamplingConfig
    /// See description of parameters below
    explicit SamplingConfig(SizeType beamWidth = 1, std::optional<SizeType> const& topK = std::nullopt,
        std::optional<FloatType> const& topP = std::nullopt, std::optional<FloatType> const& topPMin = std::nullopt,
        std::optional<SizeType> const& topPResetIds = std::nullopt,
        std::optional<FloatType> const& topPDecay = std::nullopt,
        std::optional<RandomSeedType> const& randomSeed = std::nullopt,
        std::optional<FloatType> const& temperature = std::nullopt,
        std::optional<SizeType> const& minLength = std::nullopt,
        std::optional<FloatType> const& beamSearchDiversityRate = std::nullopt,
        std::optional<FloatType> const& repetitionPenalty = std::nullopt,
        std::optional<FloatType> const& presencePenalty = std::nullopt,
        std::optional<FloatType> const& frequencyPenalty = std::nullopt,
        std::optional<FloatType> const& lengthPenalty = std::nullopt,
        std::optional<SizeType> const& earlyStopping = std::nullopt);

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

    /// @brief The beam width. Default is 1 which disables beam search.
    SizeType mBeamWidth;
    /// @brief Controls number of logits to sample from. Default is 0 (all logits).
    std::optional<SizeType> mTopK;
    /// @brief Controls the top-P probability to sample from. Default is 0.f
    std::optional<FloatType> mTopP;
    /// @brief Controls decay in the top-P algorithm. topPMin is lower-bound. Default is 1.e-6.
    std::optional<FloatType> mTopPMin;
    /// @brief Controls decay in the top-P algorithm. Indicates where to reset the decay. Default is 1.
    std::optional<SizeType> mTopPResetIds;
    /// @brief Controls decay in the top-P algorithm. The decay value. Default is 1.f
    std::optional<FloatType> mTopPDecay;
    /// @brief Controls the random seed used by the random number generator in sampling
    std::optional<RandomSeedType> mRandomSeed;
    /// @brief Controls the modulation of logits when sampling new tokens. It can have values > 0.f. Default is 1.0f
    std::optional<FloatType> mTemperature;
    /// @brief Lower bound on the number of tokens to generate. Values < 1 have no effect. Default is 1.
    std::optional<SizeType> mMinLength;
    /// @brief Controls the diversity in beam search.
    std::optional<FloatType> mBeamSearchDiversityRate;
    /// @brief Used to penalize tokens based on how often they appear in the sequence. It can have any value > 0.f.
    /// Values < 1.f encourages repetition, values > 1.f discourages it. Default is 1.f
    std::optional<FloatType> mRepetitionPenalty;
    /// @brief Used to penalize tokens already present in the sequence (irrespective of the number of appearances). It
    /// can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
    std::optional<FloatType> mPresencePenalty;
    /// @brief Used to penalize tokens already present in the sequence (dependent on the number of appearances). It can
    /// have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
    std::optional<FloatType> mFrequencyPenalty;
    /// @brief Controls how to penalize longer sequences in beam search. Default is 0.f
    std::optional<FloatType> mLengthPenalty;
    /// @brief Controls whether the generation process finishes once beamWidth sentences are generated (ends with
    /// end_token)
    std::optional<SizeType> mEarlyStopping;
};

/// @brief Configuration that controls the outputs of a Result
class OutputConfig
{
public:
    explicit OutputConfig(bool returnLogProbs = false, bool returnContextLogits = false,
        bool returnGenerationLogits = false, bool excludeInputFromOutput = false);

    /// @brief Controls if Result should contain log probabilities. Default is false.
    bool returnLogProbs;
    /// @brief Controls if Result should contain the context logits. Default is false.
    bool returnContextLogits;
    /// @brief Controls if Result should contain the generation logits. Default is false.
    bool returnGenerationLogits;
    /// @brief Controls if output tokens in Result should include the input tokens. Default is false.
    bool excludeInputFromOutput;
};

/// @brief Configuration for speculative decoding. Allows to include draft tokens, draft logits and specify acceptance
/// threshold
class SpeculativeDecodingConfig
{
public:
    explicit SpeculativeDecodingConfig(VecTokens tokens, std::optional<Tensor> logits = std::nullopt,
        std::optional<FloatType> const& acceptanceThreshold = std::nullopt);

    [[nodiscard]] VecTokens getTokens() const;
    [[nodiscard]] std::optional<Tensor> getLogits() const;
    [[nodiscard]] std::optional<FloatType> getAcceptanceThreshold() const;

private:
    friend class Serialization;
    /// @brief The draft tokens
    VecTokens mTokens;
    /// @brief The draft logits. Expected shape: [num_draft_tokens, vocab_size].
    std::optional<Tensor> mLogits;
    /// @brief The acceptance threshold. Must be > 0.f and <= 1.f
    std::optional<FloatType> mAcceptanceThreshold;
};

/// @brief Configuration for prompt tuning
class PromptTuningConfig
{
public:
    explicit PromptTuningConfig(Tensor embeddingTable);

    [[nodiscard]] Tensor getEmbeddingTable() const;

private:
    friend class Serialization;
    /// @brief The prompt embedding table. Expected shape: [task vocab_size, hidden_size]. Data type must match model
    /// weights.
    Tensor mEmbeddingTable;
};

/// @brief Configuration for LoRA
class LoraConfig
{
public:
    explicit LoraConfig(
        IdType taskId, std::optional<Tensor> weights = std::nullopt, std::optional<Tensor> config = std::nullopt);

    [[nodiscard]] IdType getTaskId() const;
    [[nodiscard]] std::optional<Tensor> getWeights() const;
    [[nodiscard]] std::optional<Tensor> getConfig() const;

private:
    friend class Serialization;

    /// @brief The Lora task id
    IdType mTaskId;
    /// @brief The Lora weights. See TRT-LLM documentation for expected shapes and types
    std::optional<Tensor> mWeights;
    /// @brief The Lora configuration. See TRT-LLM documentation for detailed description of the config tensor
    std::optional<Tensor> mConfig;
};

/// @brief A class that holds information about the request
class Request
{
public:
    /// @brief The Request constructor

    /// @param inputTokenIds The input token ids
    /// @param maxNewTokens  The maximum number of tokens to generate
    /// @param streaming Indicates if the responses should be streamed or not. Default is false.
    /// @param samplingConfig The sampling configuration
    /// @param outputConfig The output configuration
    /// @param endId The end token id
    /// @param padId The pad token id
    /// @param badWords A list of bad words tokens. Each "word" can be composed of multiple tokens
    /// @param stopWords A list of stop words tokens. Each "word" can be composed of multiple tokens
    /// @param embeddingBias The embedding bias tensor. Expected type is kFP32 and shape is [vocab_size]
    /// @param speculativeDecodingConfig The speculative decoding configuration
    /// @param pTuningConfig The prompt tuning configuration
    /// @param loraConfig The LoRA configuration
    /// @param logitsPostProcessorName The logits postprocessor name. Must correspond to one of the logits postprocessor
    /// name provided to the ExecutorConfig.
    Request(VecTokens inputTokenIds, SizeType maxNewTokens, bool streaming = false,
        SamplingConfig const& samplingConfig = SamplingConfig(), OutputConfig const& outputConfig = OutputConfig(),
        std::optional<SizeType> const& endId = std::nullopt, std::optional<SizeType> const& padId = std::nullopt,
        std::optional<std::list<VecTokens>> badWords = std::nullopt,
        std::optional<std::list<VecTokens>> stopWords = std::nullopt,
        std::optional<Tensor> embeddingBias = std::nullopt,
        std::optional<SpeculativeDecodingConfig> speculativeDecodingConfig = std::nullopt,
        std::optional<PromptTuningConfig> pTuningConfig = std::nullopt,
        std::optional<LoraConfig> loraConfig = std::nullopt,
        std::optional<std::string> logitsPostProcessorName = std::nullopt);

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
    [[nodiscard]] std::optional<std::string> getLogitsPostProcessorName() const;

    void setStreaming(bool streaming);
    void setSamplingConfig(SamplingConfig const& config);
    void setOutputConfig(OutputConfig const& outputConfig);
    void setEndId(SizeType endId);
    void setPadId(SizeType padId);
    void setBadWords(std::list<VecTokens> const& badWords);
    void setStopWords(std::list<VecTokens> const& stopWords);
    void setEmbeddingBias(Tensor const& embeddingBias);
    void setSpeculativeDecodingConfig(SpeculativeDecodingConfig const& specDecodingConfig);
    void setPromptTuningConfig(PromptTuningConfig const& pTuningConfig);
    void setLoraConfig(LoraConfig const& loraConfig);
    void setLogitsPostProcessorName(std::string const& logitsPostProcessorName);

private:
    friend class Serialization;
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Struct that holds the generation result
struct Result
{
    /// @brief Indicates if this is the final result for the request
    bool isFinal;

    /// @brief The output tokens for each beam
    BeamTokens outputTokenIds;

    /// @brief The cumulative log probabilities. Size beamSize.
    std::optional<VecLogProbs> cumLogProbs;

    /// @brief The log probabilities for each generated token. Size [beamSize, outputLen]
    std::optional<std::vector<VecLogProbs>> logProbs;

    /// @brief The context logits. Size [promptLen, vocabSizePadded]
    std::optional<Tensor> contextLogits;

    /// @brief The context logits. Size [beamSize, maxNewTokens, vocabSizePadded]
    std::optional<Tensor> generationLogits;
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

    /// @brief Get the id of the request for which this response was generated
    [[nodiscard]] IdType getRequestId() const;

    /// @brief Indicates if this response has an error or not
    [[nodiscard]] bool hasError() const;

    /// @brief Get the error msg for this response
    /// Will throw an exception if hasError is false
    [[nodiscard]] std::string getErrorMsg() const;

    /// @brief Get the result for this response
    /// Will throw an exception if hasResult is true
    [[nodiscard]] Result getResult() const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Configuration class for the scheduler
class SchedulerConfig
{
public:
    explicit SchedulerConfig(SchedulerPolicy policy = SchedulerPolicy::kGUARANTEED_NO_EVICT);

    [[nodiscard]] SchedulerPolicy getPolicy() const;

private:
    /// @brief The scheduler policy. See SchedulerPolicy.
    SchedulerPolicy mPolicy;
};

/// @brief Configuration class for the KV cache
class KvCacheConfig
{
public:
    explicit KvCacheConfig(bool enableBlockReuse = false, std::optional<SizeType> const& maxTokens = std::nullopt,
        std::optional<SizeType> const& maxAttentionWindow = std::nullopt,
        std::optional<SizeType> const& sinkTokenLength = std::nullopt,
        std::optional<FloatType> const& freeGpuMemoryFraction = std::nullopt);

    [[nodiscard]] bool getEnableBlockReuse() const;
    [[nodiscard]] std::optional<SizeType> getMaxTokens() const;
    [[nodiscard]] std::optional<SizeType> getMaxAttentionWindow() const;
    [[nodiscard]] std::optional<SizeType> getSinkTokenLength() const;
    [[nodiscard]] std::optional<FloatType> getFreeGpuMemoryFraction() const;

private:
    /// @brief Controls if KV cache blocks can be reused for different requests
    bool mEnableBlockReuse;

    /// @brief The maximum number of tokens that should be stored in the KV cache
    /// If both mMaxTokens and mFreeGpuMemoryFraction are specified, memory corresponding to the minimum will be
    /// allocated.
    std::optional<SizeType> mMaxTokens;

    /// @brief Size of the attention window for each sequence. Only the last mMaxAttentionWindow tokens of each sequence
    /// will be stored in the KV cache.
    std::optional<SizeType> mMaxAttentionWindow;

    /// @brief Number of sink tokens (tokens to always keep in attention window)
    std::optional<SizeType> mSinkTokenLength;

    /// @brief The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%.
    /// If both mMaxTokens and mFreeGpuMemoryFraction are specified, memory corresponding to the minimum will be
    /// allocated.
    std::optional<FloatType> mFreeGpuMemoryFraction;
};

SizeType const kDefaultIterStatsMaxIterations = 1000;
// Per request stats may have additional overhead due to going through all requests. Turned off by default.
SizeType const kDefaultRequestStatsMaxIterations = 0;

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
    explicit ParallelConfig(CommunicationType commType = CommunicationType::kMPI,
        CommunicationMode commMode = CommunicationMode::kLEADER,
        std::optional<std::vector<SizeType>> deviceIds = std::nullopt,
        std::optional<std::vector<SizeType>> participantIds = std::nullopt);

    [[nodiscard]] CommunicationType getCommunicationType() const;
    [[nodiscard]] CommunicationMode getCommunicationMode() const;
    [[nodiscard]] std::optional<std::vector<SizeType>> getDeviceIds() const;
    [[nodiscard]] std::optional<std::vector<SizeType>> getParticipantIds() const;

    void setCommunicationType(CommunicationType type);
    void setCommunicationMode(CommunicationMode mode);
    void setDeviceIds(std::vector<SizeType> const& deviceIds);
    void setParticipantIds(std::vector<SizeType> const& participantIds);

private:
    /// @brief The type of communication protocol used. Default is MPI.
    CommunicationType mCommType;

    /// @brief The mode of communication. See CommunicationMode.
    CommunicationMode mCommMode;

    /// @brief The GPU device ids to use for executing this model
    std::optional<std::vector<SizeType>> mDeviceIds;

    /// @brief The participant ids (MPI ranks for example) used for executing this model
    std::optional<std::vector<SizeType>> mParticipantIds;
};

/// @brief config for PeftCacheManager
class PeftCacheConfig
{
public:
    explicit PeftCacheConfig(SizeType numHostModuleLayer = 0, SizeType numDeviceModuleLayer = 0,
        SizeType optimalAdapterSize = 8, SizeType maxAdapterSize = 64, SizeType numPutWorkers = 1,
        SizeType numEnsureWorkers = 1, SizeType numCopyStreams = 1, SizeType maxPagesPerBlockHost = 24,
        SizeType maxPagesPerBlockDevice = 8, std::optional<float> const& deviceCachePercent = std::nullopt,
        std::optional<size_t> const& hostCacheSize = std::nullopt);

    [[nodiscard]] SizeType getNumHostModuleLayer() const;
    [[nodiscard]] SizeType getNumDeviceModuleLayer() const;
    [[nodiscard]] SizeType getOptimalAdapterSize() const;
    [[nodiscard]] SizeType getMaxAdapterSize() const;
    [[nodiscard]] SizeType getNumPutWorkers() const;
    [[nodiscard]] SizeType getNumEnsureWorkers() const;
    [[nodiscard]] SizeType getNumCopyStreams() const;
    [[nodiscard]] SizeType getMaxPagesPerBlockHost() const;
    [[nodiscard]] SizeType getMaxPagesPerBlockDevice() const;
    [[nodiscard]] std::optional<float> getDeviceCachePercent() const;
    [[nodiscard]] std::optional<size_t> getHostCacheSize() const;

private:
    // number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache
    SizeType mNumHostModuleLayer;
    // number of max sized 1-layer 1-module sets of weights that can be stored in host cache
    SizeType mNumDeviceModuleLayer;
    // optimal adapter size used to set page width
    SizeType mOptimalAdapterSize;
    // max supported adapter size. Used to compute minimum
    SizeType mMaxAdapterSize;
    // number of worker threads used to put weights into host cache
    SizeType mNumPutWorkers;
    // number of worker threads used to copy weights from host to device
    SizeType mNumEnsureWorkers;
    // number of streams used to copy weights from host to device
    SizeType mNumCopyStreams;
    // Number of cache pages per allocation block (host)
    SizeType mMaxPagesPerBlockHost;
    // Number of cache pages per allocation block (device)
    SizeType mMaxPagesPerBlockDevice;
    // percent of memory after engine load to use for cache
    std::optional<float> mDeviceCachePercent;
    // size in bytes to use for host cache
    std::optional<size_t> mHostCacheSize;
};

/// @brief Configuration class for the model executor
class ExecutorConfig
{
public:
    explicit ExecutorConfig(SizeType maxBeamWidth = 1, SchedulerConfig const& schedulerConfig = SchedulerConfig(),
        KvCacheConfig const& kvCacheConfig = KvCacheConfig(), bool enableChunkedContext = false,
        bool normalizeLogProbs = true, SizeType iterStatsMaxIterations = kDefaultIterStatsMaxIterations,
        SizeType requestStatsMaxIterations = kDefaultRequestStatsMaxIterations,
        BatchingType batchingType = BatchingType::kINFLIGHT,
        std::optional<ParallelConfig> parallelConfig = std::nullopt,
        std::optional<PeftCacheConfig> const& peftCacheConfig = std::nullopt,
        std::optional<LogitsPostProcessorMap> logitsPostProcessorMap = std::nullopt,
        std::optional<MedusaChoices> medusaChoices = std::nullopt);

    [[nodiscard]] SizeType getMaxBeamWidth() const;
    [[nodiscard]] SchedulerConfig getSchedulerConfig() const;
    [[nodiscard]] KvCacheConfig getKvCacheConfig() const;
    [[nodiscard]] bool getEnableChunkedContext() const;
    [[nodiscard]] bool getNormalizeLogProbs() const;
    [[nodiscard]] SizeType getIterStatsMaxIterations() const;
    [[nodiscard]] SizeType getRequestStatsMaxIterations() const;
    [[nodiscard]] BatchingType getBatchingType() const;
    [[nodiscard]] std::optional<ParallelConfig> getParallelConfig() const;
    [[nodiscard]] std::optional<PeftCacheConfig> getPeftCacheConfig() const;
    [[nodiscard]] std::optional<LogitsPostProcessorMap> getLogitsPostProcessorMap() const;
    [[nodiscard]] std::optional<MedusaChoices> getMedusaChoices() const;

    void setMaxBeamWidth(SizeType maxBeamWidth);
    void setSchedulerConfig(SchedulerConfig const& schedulerConfig);
    void setKvCacheConfig(KvCacheConfig const& kvCacheConfig);
    void setEnableChunkedContext(bool enableChunkedContext);
    void setNormalizeLogProbs(bool normalizeLogProbs);
    void setIterStatsMaxIterations(SizeType iterStatsMaxIterations);
    void setRequestStatsMaxIterations(SizeType requestStatsMaxIterations);
    void setBatchingType(BatchingType batchingType);
    void setParallelConfig(ParallelConfig const& parallelConfig);
    void setPeftCacheConfig(PeftCacheConfig const& peftCacheConfig);
    void setLogitsPostProcessorMap(LogitsPostProcessorMap const& logitsPostProcessorMap);
    void setMedusaChoices(MedusaChoices const& medusaChoices);

private:
    /// @brief The beam width value of requests that will be sent to the executor
    SizeType mMaxBeamWidth;

    /// @brief The scheduler configuration.
    SchedulerConfig mSchedulerConfig;

    /// @brief The KV cache configuration.
    KvCacheConfig mKvCacheConfig;

    /// @brief The KV cache configuration.
    bool mEnableChunkedContext;

    /// @brief Controls if log probabilities should be normalized or not.
    bool mNormalizeLogProbs;

    /// @brief Controls the maximum number of iterations for which to keep statistics.
    SizeType mIterStatsMaxIterations;

    /// @brief Controls the maximum number of iterations for which to keep per-request statistics.
    SizeType mRequestStatsMaxIterations;

    /// @brief The type of batching strategy to use. See BatchingType.
    BatchingType mBatchingType;

    /// @brief The parallel execution configuration.
    std::optional<ParallelConfig> mParallelConfig;
    std::optional<PeftCacheConfig> mPeftCacheConfig;
    std::optional<LogitsPostProcessorMap> mLogitsPostProcessorMap;
    std::optional<MedusaChoices> mMedusaChoices;
};

/// @brief The executor is responsible for receiving new requests and sending responses, and running the inference
class Executor
{

public:
    /// @brief
    /// @param modelPath Path to the folder that defines the model to run
    /// @param modelType The type of model
    /// @param executorConfig The configuration for the executor
    /// @param comm An optional inter-process communicator configuration
    Executor(std::filesystem::path const& modelPath, ModelType modelType, ExecutorConfig const& executorConfig);

    Executor(std::vector<uint8_t> const& engineBuffer, std::string const& jsonConfigStr, ModelType modelType,
        ExecutorConfig const& executorConfig);

    Executor(std::shared_ptr<Model> model, ExecutorConfig const& executorConfig);

    ~Executor();

    /// @brief Enqueue a new request
    /// @param request The LLM request which contains input tokens and request parameters
    /// @return A unique id that identifies the request
    [[nodiscard]] IdType enqueueRequest(Request const& request);

    /// @brief Enqueue a batch of request
    [[nodiscard]] std::vector<IdType> enqueueRequests(std::vector<Request> const& requests);

    /// @brief Await for ready responses
    /// @param id An optional request id. If not specified, responses for any request can be returned
    /// @param timeout The maximum time to wait for new responses
    /// @return A vector of responses
    [[nodiscard]] std::vector<Response> awaitResponses(std::optional<IdType> const& requestId = std::nullopt,
        std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    /// @brief Get the number of ready responses
    /// @param requestId An optional request id
    /// @return The number of ready responses
    [[nodiscard]] SizeType getNumResponsesReady(std::optional<IdType> const& requestId = std::nullopt) const;

    /// @brief Cancel the request with provided request id
    /// @param id The request id for which to cancel the response
    void cancelRequest(IdType requestId);

    /// @brief  Signals the server to shutdown
    ///         This call is blocking. Only returns when all requests have terminated or timeout has been reached
    void shutdown();

    /// @brief  Returns the per-iterations statistics computed since last call to getLatestIterationStats
    ///         Contains at most iterStatsMaxIterations iterations
    /// @return Iteration stats
    std::deque<IterationStats> getLatestIterationStats();

    /// @brief  Returns the request stats of each iteration computed since last call to getLatestRequestStats
    ///         Contains at most requestStatsMaxIterations iterations
    /// @return Request stats grouped by iterations
    std::deque<RequestStatsPerIteration> getLatestRequestStats();

    /// @brief  Indicates if the current process is allowed to enqueueRequests
    [[nodiscard]] bool canEnqueueRequests() const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Class with utility functions to serialize statistics to json string
class JsonSerialization
{
public:
    /// @brief Utility function to convert an iterationStats struct to a json serialized string
    [[nodiscard]] static std::string toJsonStr(IterationStats const& iterationStats);

    /// @brief Utility function to convert a requestStatsPerIteration struct to a json serialized string
    [[nodiscard]] static std::string toJsonStr(RequestStatsPerIteration const& requestStatsPerIter);

    /// @brief Utility function to convert a requestStats struct to a json serialized string
    [[nodiscard]] static std::string toJsonStr(RequestStats const& requestStats);
};

} // namespace tensorrt_llm::executor
