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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensorrt_llm::mpi
{
class MpiComm;
}

namespace tensorrt_llm::executor
{

/// @brief Version of TRT-LLM
char const* version() noexcept;

class Model;
class Serialization;
class ContextPhaseState;

/// @brief Sampling configuration
class SamplingConfig
{
public:
    /// @brief Constructor for SamplingConfig
    /// See description of parameters below
    explicit SamplingConfig(SizeType32 beamWidth = 1, std::optional<SizeType32> const& topK = std::nullopt,
        std::optional<FloatType> const& topP = std::nullopt, std::optional<FloatType> const& topPMin = std::nullopt,
        std::optional<TokenIdType> const& topPResetIds = std::nullopt,
        std::optional<FloatType> const& topPDecay = std::nullopt,
        std::optional<RandomSeedType> const& seed = std::nullopt,
        std::optional<FloatType> const& temperature = std::nullopt,
        std::optional<SizeType32> const& minTokens = std::nullopt,
        std::optional<FloatType> const& beamSearchDiversityRate = std::nullopt,
        std::optional<FloatType> const& repetitionPenalty = std::nullopt,
        std::optional<FloatType> const& presencePenalty = std::nullopt,
        std::optional<FloatType> const& frequencyPenalty = std::nullopt,
        std::optional<FloatType> const& lengthPenalty = std::nullopt,
        std::optional<SizeType32> const& earlyStopping = std::nullopt,
        std::optional<SizeType32> const& noRepeatNgramSize = std::nullopt);

    bool operator==(SamplingConfig const& other) const;

    [[nodiscard]] SizeType32 getBeamWidth() const;
    [[nodiscard]] std::optional<SizeType32> getTopK() const;
    [[nodiscard]] std::optional<FloatType> getTopP() const;
    [[nodiscard]] std::optional<FloatType> getTopPMin() const;
    [[nodiscard]] std::optional<SizeType32> getTopPResetIds() const;
    [[nodiscard]] std::optional<FloatType> getTopPDecay() const;
    [[nodiscard]] std::optional<RandomSeedType> getSeed() const;
    [[nodiscard]] std::optional<RandomSeedType> getRandomSeed() const;
    [[nodiscard]] std::optional<FloatType> getTemperature() const;
    [[nodiscard]] std::optional<SizeType32> getMinTokens() const;
    [[nodiscard]] std::optional<SizeType32> getMinLength() const;
    [[nodiscard]] std::optional<FloatType> getBeamSearchDiversityRate() const;
    [[nodiscard]] std::optional<FloatType> getRepetitionPenalty() const;
    [[nodiscard]] std::optional<FloatType> getPresencePenalty() const;
    [[nodiscard]] std::optional<FloatType> getFrequencyPenalty() const;
    [[nodiscard]] std::optional<FloatType> getLengthPenalty() const;
    [[nodiscard]] std::optional<SizeType32> getEarlyStopping() const;
    [[nodiscard]] std::optional<SizeType32> getNoRepeatNgramSize() const;

    void setBeamWidth(SizeType32 beamWidth);
    void setTopK(std::optional<SizeType32> const& topK);
    void setTopP(std::optional<FloatType> const& topP);
    void setTopPMin(std::optional<FloatType> const& topPMin);
    void setTopPResetIds(std::optional<TokenIdType> const& topPResetIds);
    void setTopPDecay(std::optional<FloatType> const& topPDecay);
    void setSeed(std::optional<RandomSeedType> const& seed);
    void setRandomSeed(std::optional<RandomSeedType> const& randomSeed);
    void setTemperature(std::optional<FloatType> const& temperature);
    void setMinTokens(std::optional<SizeType32> const& minTokens);
    void setMinLength(std::optional<SizeType32> const& minLength);
    void setBeamSearchDiversityRate(std::optional<FloatType> const& beamSearchDiversityRate);
    void setRepetitionPenalty(std::optional<FloatType> const& repetitionPenalty);
    void setPresencePenalty(std::optional<FloatType> const& presencePenalty);
    void setFrequencyPenalty(std::optional<FloatType> const& frequencyPenalty);
    void setLengthPenalty(std::optional<FloatType> const& lengthPenalty);
    void setEarlyStopping(std::optional<SizeType32> const& earlyStopping);
    void setNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize);

private:
    static SizeType32 checkBeamWidth(SizeType32 beamWidth);
    static std::optional<FloatType> const& checkTopK(std::optional<FloatType> const& topK);
    static std::optional<FloatType> const& checkTopP(std::optional<FloatType> const& topP);
    static std::optional<FloatType> const& checkTopPMin(std::optional<FloatType> const& topPMin);
    static std::optional<TokenIdType> const& checkTopPResetIds(std::optional<TokenIdType> const& topPResetIds);
    static std::optional<FloatType> const& checkTopPDecay(std::optional<FloatType> const& topPDecay);
    static std::optional<FloatType> const& checkTemperature(std::optional<FloatType> const& temperature);
    static std::optional<FloatType> const& checkRepetitionPenalty(std::optional<FloatType> const& penalty);
    static std::optional<SizeType32> const& checkMinTokens(std::optional<SizeType32> const& minTokens);
    static std::optional<SizeType32> const& checkNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize);
    static std::optional<FloatType> const& checkBeamSearchDiversityRate(
        std::optional<FloatType> const& beamSearchDiversityRate);

    friend class Serialization;

    /// @brief The beam width. Default is 1 which disables beam search.
    SizeType32 mBeamWidth;
    /// @brief Controls number of logits to sample from. Default is 0 (all logits).
    std::optional<SizeType32> mTopK;
    /// @brief Controls the top-P probability to sample from. Default is 0.f
    std::optional<FloatType> mTopP;
    /// @brief Controls decay in the top-P algorithm. topPMin is lower-bound. Default is 1.e-6.
    std::optional<FloatType> mTopPMin;
    /// @brief Controls decay in the top-P algorithm. Indicates where to reset the decay. Default is 1.
    std::optional<TokenIdType> mTopPResetIds;
    /// @brief Controls decay in the top-P algorithm. The decay value. Default is 1.f
    std::optional<FloatType> mTopPDecay;
    /// @brief Controls the random seed used by the random number generator in sampling
    std::optional<RandomSeedType> mSeed;
    /// @brief Controls the modulation of logits when sampling new tokens. It can have values > 0.f. Default is 1.0f
    std::optional<FloatType> mTemperature;
    /// @brief Lower bound on the number of tokens to generate. Values < 1 have no effect. Default is 1.
    std::optional<SizeType32> mMinTokens;
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
    std::optional<SizeType32> mEarlyStopping;
    /// @brief Controls how many repeat ngram size are acceptable. Default is 1 << 30.
    std::optional<SizeType32> mNoRepeatNgramSize;
};

/// @brief Configuration that controls the outputs of a Result
class OutputConfig
{
public:
    explicit OutputConfig(bool returnLogProbs = false, bool returnContextLogits = false,
        bool returnGenerationLogits = false, bool excludeInputFromOutput = false, bool returnEncoderOutput = false);

    /// @brief Controls if Result should contain log probabilities. Default is false.
    bool returnLogProbs;
    /// @brief Controls if Result should contain the context logits. Default is false.
    bool returnContextLogits;
    /// @brief Controls if Result should contain the generation logits. Default is false.
    bool returnGenerationLogits;
    /// @brief Controls if output tokens in Result should include the input tokens. Default is false.
    bool excludeInputFromOutput;
    /// @brief Controls if Result should contain encoder output hidden states (for encoder-only and encoder-decoder
    /// models). Default is false.
    bool returnEncoderOutput;
};

/// @brief Configuration for speculative decoding with external draft tokens.
/// Allows to include draft tokens, draft logits and specify acceptance threshold.
class ExternalDraftTokensConfig
{
public:
    explicit ExternalDraftTokensConfig(VecTokens tokens, std::optional<Tensor> logits = std::nullopt,
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
    explicit PromptTuningConfig(
        Tensor embeddingTable, std::optional<VecTokenExtraIds> inputTokenExtraIds = std::nullopt);

    [[nodiscard]] Tensor getEmbeddingTable() const;

    [[nodiscard]] std::optional<VecTokenExtraIds> getInputTokenExtraIds() const;

private:
    friend class Serialization;
    /// @brief The prompt embedding table. Expected shape: [task vocab_size, hidden_size]. Data type must match model
    /// weights.
    Tensor mEmbeddingTable;

    /// @brief The input token extra ids for KV Cache reuse when p-tuning is enabled
    std::optional<VecTokenExtraIds> mInputTokenExtraIds;
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

struct LookaheadDecodingConfig
{
    LookaheadDecodingConfig(SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize);

    explicit LookaheadDecodingConfig()
        : LookaheadDecodingConfig(1, 1, 0)
    {
    }

    bool operator==(LookaheadDecodingConfig const& other) const;
    [[nodiscard]] std::tuple<SizeType32 const, SizeType32 const, SizeType32 const> get() const;
    [[nodiscard]] SizeType32 getWindowSize() const;
    [[nodiscard]] SizeType32 getNgramSize() const;
    [[nodiscard]] SizeType32 getVerificationSetSize() const;

    /// @brief return <maxDecodingTokens, maxPathLen, maxDraftTokens, maxDraftPathLen>
    std::tuple<SizeType32, SizeType32, SizeType32, SizeType32> calculateSpeculativeResource() const;

    /// @brief return true when `this` can be executed on resources defined by `that`
    bool isLE(LookaheadDecodingConfig const& that) const;

    /// @brief return true when the parameter combination is valid.
    static bool isLegal(SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize) noexcept;

private:
    friend class Serialization;

    // Number of NGrams in lookahead branch per step.
    SizeType32 mWindowSize;
    // Number of tokens per NGram.
    SizeType32 mNgramSize;
    // Number of NGrams in verification branch per step.
    SizeType32 mVerificationSetSize;
};

class ContextPhaseParams
{
public:
    explicit ContextPhaseParams(VecTokens firstGenTokens);
    ContextPhaseParams(VecTokens firstGenTokens, void* state);

    ContextPhaseParams(ContextPhaseParams const&);
    ContextPhaseParams(ContextPhaseParams&&);
    ContextPhaseParams& operator=(ContextPhaseParams const&);
    ContextPhaseParams& operator=(ContextPhaseParams&&);

    [[nodiscard]] bool operator==(ContextPhaseParams const&) const noexcept;

    [[nodiscard]] VecTokens const& getFirstGenTokens() const& noexcept;
    [[nodiscard]] VecTokens popFirstGenTokens() && noexcept;
    [[nodiscard]] void const* getState() const noexcept;
    [[nodiscard]] void* getState() noexcept;
    [[nodiscard]] void* releaseState() noexcept;

private:
    friend class Serialization;
    static void deleter(void const* data);
    using StatePtr = std::unique_ptr<void, decltype(&deleter)>;

    /// @brief The first tokens generated by context executor
    VecTokens mFirstGenTokens;

    /// @brief Context phase state of this request
    StatePtr mState{nullptr, deleter};
};

/// @brief A class that holds information about the request
class Request
{
public:
    static constexpr PriorityType kDefaultPriority = 0.5;

    /// @brief The Request constructor

    /// @param inputTokenIds The input token ids
    /// @param maxTokens  The maximum number of tokens to generate
    /// @param streaming Indicates if the responses should be streamed or not. Default is false.
    /// @param samplingConfig The sampling configuration
    /// @param outputConfig The output configuration
    /// @param endId The end token id
    /// @param padId The pad token id
    /// @param positionIds The input position ids
    /// @param badWords A list of bad words tokens. Each "word" can be composed of multiple tokens
    /// @param stopWords A list of stop words tokens. Each "word" can be composed of multiple tokens
    /// @param embeddingBias The embedding bias tensor. Expected type is kFP32 and shape is [vocab_size]
    /// @param externalDraftTokensConfig The speculative decoding configuration
    /// @param pTuningConfig The prompt tuning configuration
    /// @param loraConfig The LoRA configuration
    /// @param logitsPostProcessorName The logits postprocessor name. Must correspond to one of the logits postprocessor
    /// name provided to the ExecutorConfig.
    /// @param encoderInputTokenIds The encoder input token ids for encoder-decoder models, or encoder-only models
    /// @param returnAllGeneratedTokens Indicates whether to return the full beams or just the newly generated tokens
    /// after every streaming step.
    /// @param priority Sets the execution priority of this request.
    /// @param encoderInputFeatures Encoder input features for multimodal models.
    /// @param encoderOutputLength Encoder output length if encoder input and output have different lengths (due to
    /// convolution down-sampling, etc.)
    /// @param type Indicate the request type for disaggregated serving mode.
    /// @param contextPhaseParams Generated token ID  from context only executor.
    /// @param numReturnSequences The number of returning sequences.
    Request(VecTokens inputTokenIds, SizeType32 maxTokens, bool streaming = false,
        SamplingConfig const& samplingConfig = SamplingConfig(), OutputConfig const& outputConfig = OutputConfig(),
        std::optional<SizeType32> const& endId = std::nullopt, std::optional<SizeType32> const& padId = std::nullopt,
        std::optional<std::vector<SizeType32>> positionIds = std::nullopt,
        std::optional<std::list<VecTokens>> badWords = std::nullopt,
        std::optional<std::list<VecTokens>> stopWords = std::nullopt,
        std::optional<Tensor> embeddingBias = std::nullopt,
        std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig = std::nullopt,
        std::optional<PromptTuningConfig> pTuningConfig = std::nullopt,
        std::optional<LoraConfig> loraConfig = std::nullopt,
        std::optional<LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<std::string> logitsPostProcessorName = std::nullopt,
        std::optional<VecTokens> encoderInputTokenIds = std::nullopt, std::optional<IdType> clientId = std::nullopt,
        bool returnAllGeneratedTokens = false, PriorityType priority = kDefaultPriority,
        RequestType type = RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<ContextPhaseParams> contextPhaseParams = std::nullopt,
        std::optional<Tensor> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt, SizeType32 numReturnSequences = 1);

    /// @brief This logits postprocessor name will dispatch to the batched logits postprocessor
    static auto constexpr kBatchedPostProcessorName = "batched";

    Request(Request const& other);
    Request(Request&& other) noexcept;
    Request& operator=(Request const& other);
    Request& operator=(Request&& other) noexcept;
    ~Request();

    [[nodiscard]] VecTokens getInputTokenIds() const;
    [[nodiscard]] SizeType32 getMaxTokens() const;
    [[nodiscard]] SizeType32 getMaxNewTokens() const;
    [[nodiscard]] bool getStreaming() const;
    [[nodiscard]] SamplingConfig getSamplingConfig() const;
    [[nodiscard]] OutputConfig getOutputConfig() const;
    [[nodiscard]] std::optional<SizeType32> getEndId() const;
    [[nodiscard]] std::optional<SizeType32> getPadId() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getPositionIds() const;
    [[nodiscard]] std::optional<std::list<VecTokens>> getBadWords() const;
    [[nodiscard]] std::optional<std::list<VecTokens>> getStopWords() const;
    [[nodiscard]] std::optional<Tensor> getEmbeddingBias() const;
    [[nodiscard]] std::optional<ExternalDraftTokensConfig> getExternalDraftTokensConfig() const;
    [[nodiscard]] std::optional<PromptTuningConfig> getPromptTuningConfig() const;
    [[nodiscard]] std::optional<LoraConfig> getLoraConfig() const;
    [[nodiscard]] std::optional<LookaheadDecodingConfig> getLookaheadConfig() const;
    [[nodiscard]] std::optional<std::string> getLogitsPostProcessorName() const;
    [[nodiscard]] std::optional<VecTokens> getEncoderInputTokenIds() const;
    [[nodiscard]] std::optional<IdType> getClientId() const;
    [[nodiscard]] PriorityType getPriority() const;
    [[nodiscard]] bool getReturnAllGeneratedTokens() const;
    [[nodiscard]] std::optional<ContextPhaseParams> const& getContextPhaseParams() const;
    [[nodiscard]] std::optional<Tensor> getEncoderInputFeatures() const;
    [[nodiscard]] std::optional<SizeType32> getEncoderOutputLength() const;
    [[nodiscard]] RequestType getRequestType() const;
    [[nodiscard]] SizeType32 getNumReturnSequences() const;

    void setStreaming(bool streaming);
    void setSamplingConfig(SamplingConfig const& config);
    void setOutputConfig(OutputConfig const& outputConfig);
    void setEndId(SizeType32 endId);
    void setPadId(SizeType32 padId);
    void setPositionIds(std::vector<SizeType32> const& positionIds);
    void setBadWords(std::list<VecTokens> const& badWords);
    void setStopWords(std::list<VecTokens> const& stopWords);
    void setEmbeddingBias(Tensor const& embeddingBias);
    void setExternalDraftTokensConfig(ExternalDraftTokensConfig const& externalDraftTokensConfig);
    void setPromptTuningConfig(PromptTuningConfig const& pTuningConfig);
    void setLoraConfig(LoraConfig const& loraConfig);
    void setLookaheadConfig(LookaheadDecodingConfig const& lookaheadConfig);
    void setLogitsPostProcessorName(std::string const& logitsPostProcessorName);
    void setEncoderInputTokenIds(VecTokens const& encoderInputTokenIds);
    void setClientId(IdType clientId);
    void setPriority(PriorityType priority);
    void setReturnAllGeneratedTokens(bool returnAllGeneratedTokens);
    void setRequestType(RequestType const& requestType);
    void setContextPhaseParams(ContextPhaseParams contextPhaseParams);
    void setEncoderInputFeatures(Tensor encoderInputFeatures);
    void setEncoderOutputLength(SizeType32 encoderOutputLength);
    void setNumReturnSequences(SizeType32 numReturnSequences);

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

    /// @brief The context logits. Size [beamSize, maxNewTokens, vocabSizePadded] (non-streaming)
    /// or [maxNewTokens, beamSize, vocabSizePadded] (streaming and allGeneratedTokens)
    /// or [1, beamSize, vocabSizePadded] (streaming and non-allGeneratedTokens)
    std::optional<Tensor> generationLogits;

    /// @brief The encoder output. Size [encoderLen, hiddenSize]
    std::optional<Tensor> encoderOutput;

    /// @brief The reason why the model stopped generating tokens for each beam in this request. Size [beamSize].
    /// Currently only supported when beamSize is 1 and when using BatchingType::kINFLIGHT.
    std::vector<FinishReason> finishReasons;

    /// @brief The params of the context phase.
    std::optional<ContextPhaseParams> contextPhaseParams;

    /// @brief The decoding iterations it takes.
    SizeType32 decodingIter{0};

    /// @brief The index of the output sequence where 0 <= sequenceIndex < numReturnSequences
    SizeType32 sequenceIndex{0};

    /// @brief Indicates if this is the final result for a given sequence in the request
    bool isSequenceFinal;
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
    [[nodiscard]] std::string const& getErrorMsg() const;

    /// @brief Get the result for this response
    /// Will throw an exception if hasResult is true
    [[nodiscard]] Result const& getResult() const;

private:
    friend class Serialization;
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Configuration class for the scheduler
class SchedulerConfig
{
public:
    explicit SchedulerConfig(
        CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        std::optional<ContextChunkingPolicy> contextChunkingPolicy = std::nullopt);

    bool operator==(SchedulerConfig const& other) const;

    [[nodiscard]] CapacitySchedulerPolicy getCapacitySchedulerPolicy() const;

    [[nodiscard]] std::optional<ContextChunkingPolicy> getContextChunkingPolicy() const;

private:
    friend class Serialization;

    /// @brief The capacity scheduler policy. See CapacitySchedulerPolicy.
    CapacitySchedulerPolicy mCapacitySchedulerPolicy;

    /// @brief The context chunking policy. See ContextChunkingPolicy.
    std::optional<ContextChunkingPolicy> mContextChunkingPolicy;
};

/// @brief Configuration class for the KV cache
class KvCacheConfig
{
public:
    explicit KvCacheConfig(bool enableBlockReuse = false, std::optional<SizeType32> const& maxTokens = std::nullopt,
        std::optional<std::vector<SizeType32>> const& maxAttentionWindowVec = std::nullopt,
        std::optional<SizeType32> const& sinkTokenLength = std::nullopt,
        std::optional<FloatType> const& freeGpuMemoryFraction = std::nullopt,
        std::optional<size_t> const& hostCacheSize = std::nullopt, bool onboardBlocks = true);

    [[nodiscard]] bool getEnableBlockReuse() const;
    [[nodiscard]] std::optional<SizeType32> getMaxTokens() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getMaxAttentionWindowVec() const;
    [[nodiscard]] std::optional<SizeType32> getSinkTokenLength() const;
    [[nodiscard]] std::optional<FloatType> getFreeGpuMemoryFraction() const;
    [[nodiscard]] std::optional<size_t> getHostCacheSize() const;
    [[nodiscard]] bool getOnboardBlocks() const;

    void setEnableBlockReuse(bool enableBlockReuse);
    void setMaxTokens(SizeType32 maxTokens);
    void setMaxAttentionWindowVec(std::vector<SizeType32> maxAttentionWindowVec);
    void setSinkTokenLength(SizeType32 sinkTokenLength);
    void setFreeGpuMemoryFraction(FloatType freeGpuMemoryFraction);
    void setHostCacheSize(size_t hostCacheSize);
    void setOnboardBlocks(bool onboardBlocks);

private:
    friend class Serialization;

    /// @brief Controls if KV cache blocks can be reused for different requests
    bool mEnableBlockReuse;

    /// @brief The maximum number of tokens that should be stored in the KV cache
    /// If both mMaxTokens and mFreeGpuMemoryFraction are specified, memory corresponding to the minimum will be
    /// allocated.
    std::optional<SizeType32> mMaxTokens;

    /// @brief Size of the attention window for each sequence. Only the last mMaxAttentionWindow tokens of each sequence
    /// will be stored in the KV cache. Different layers may have different max attention window sizes.
    /// If the number of elements in mMaxAttentionWindowVec is less than the number of layers, mMaxAttentionWindowVec
    /// will be repeated multiple times to the number of layers.
    std::optional<std::vector<SizeType32>> mMaxAttentionWindowVec;

    /// @brief Number of sink tokens (tokens to always keep in attention window)
    std::optional<SizeType32> mSinkTokenLength;

    /// @brief The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%.
    /// If both mMaxTokens and mFreeGpuMemoryFraction are specified, memory corresponding to the minimum will be
    /// allocated.
    std::optional<FloatType> mFreeGpuMemoryFraction;

    /// @brief Size of secondary memory pool in bytes. Default is 0.
    /// Having a secondary memory pool increases KV cache block reuse potential.
    std::optional<size_t> mHostCacheSize;

    /// @brief Controls whether offloaded blocks should be onboarded back into primary memory before being reused.
    bool mOnboardBlocks;
};

/// @brief Configuration class for the runtime perf knobs
class ExtendedRuntimePerfKnobConfig
{
public:
    explicit ExtendedRuntimePerfKnobConfig(bool multiBlockMode = true, bool enableContextFMHAFP32Acc = false);

    bool operator==(ExtendedRuntimePerfKnobConfig const& other) const
    {
        return mMultiBlockMode == other.mMultiBlockMode && mEnableContextFMHAFP32Acc == other.mEnableContextFMHAFP32Acc;
    }

    [[nodiscard]] bool getMultiBlockMode() const;
    [[nodiscard]] bool getEnableContextFMHAFP32Acc() const;

    void setMultiBlockMode(bool multiBlockMode);
    void setEnableContextFMHAFP32Acc(bool enableContextFMHAFP32Acc);

private:
    friend class Serialization;

    /// @brief Control if multi block mode should be enabled or not.
    bool mMultiBlockMode;

    /// @brief If enable FMHA runner FP32 accumulation.
    bool mEnableContextFMHAFP32Acc;
};

/// @brief Configuration class for debugging output
class DebugConfig
{
    using StringVec = std::vector<std::string>;

public:
    explicit DebugConfig(bool debugInputTensors = false, bool debugOutputTensors = false,
        StringVec debugTensorNames = {}, SizeType32 debugTensorsMaxIterations = 0);

    bool operator==(DebugConfig const& other) const;

    [[nodiscard]] bool getDebugInputTensors() const;
    [[nodiscard]] bool getDebugOutputTensors() const;
    [[nodiscard]] StringVec const& getDebugTensorNames() const;
    [[nodiscard]] SizeType32 getDebugTensorsMaxIterations() const;

    void setDebugInputTensors(bool debugInputTensors);
    void setDebugOutputTensors(bool debugOutputTensors);
    void setDebugTensorNames(StringVec const& debugTensorNames);
    void setDebugTensorsMaxIterations(SizeType32 debugTensorsMaxIterations);

private:
    friend class Serialization;

    /// @brief If true, debug all input tensors.
    bool mDebugInputTensors;
    /// @brief If true, debug all output tensors.
    bool mDebugOutputTensors;
    /// @brief If not empty, only debug tensors in this list.
    StringVec mDebugTensorNames;
    /// @brief If > 0, provide debug tensors for at most debugTensorsMaxIterations past iterations,
    /// else dump them to files.
    SizeType32 mDebugTensorsMaxIterations;
};

SizeType32 const kDefaultIterStatsMaxIterations = 1000;
// Per request stats may have additional overhead due to going through all requests. Turned off by default.
SizeType32 const kDefaultRequestStatsMaxIterations = 0;

class OrchestratorConfig
{
public:
    explicit OrchestratorConfig(bool isOrchestrator = true, std::string workerExecutablePath = "",
        std::shared_ptr<mpi::MpiComm> orchLeaderComm = nullptr, bool spawnProcesses = true);

    [[nodiscard]] bool getIsOrchestrator() const;
    [[nodiscard]] std::string getWorkerExecutablePath() const;
    [[nodiscard]] std::shared_ptr<mpi::MpiComm> getOrchLeaderComm() const;
    [[nodiscard]] bool getSpawnProcesses() const;

    void setIsOrchestrator(bool isOrchestrator);
    void setWorkerExecutablePath(std::string const& workerExecutablePath);
    void setOrchLeaderComm(std::shared_ptr<mpi::MpiComm> const& orchLeaderComm);
    void setSpawnProcesses(bool spawnProcesses);

private:
    bool mIsOrchestrator;
    std::string mWorkerExecutablePath;
    std::shared_ptr<mpi::MpiComm> mOrchLeaderComm;
    bool mSpawnProcesses;
};

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
        std::optional<std::vector<SizeType32>> deviceIds = std::nullopt,
        std::optional<std::vector<SizeType32>> participantIds = std::nullopt,
        std::optional<OrchestratorConfig> const& orchestratorConfig = std::nullopt);

    [[nodiscard]] CommunicationType getCommunicationType() const;
    [[nodiscard]] CommunicationMode getCommunicationMode() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getDeviceIds() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getParticipantIds() const;
    [[nodiscard]] std::optional<OrchestratorConfig> getOrchestratorConfig() const;

    void setCommunicationType(CommunicationType type);
    void setCommunicationMode(CommunicationMode mode);
    void setDeviceIds(std::vector<SizeType32> const& deviceIds);
    void setParticipantIds(std::vector<SizeType32> const& participantIds);
    void setOrchestratorConfig(OrchestratorConfig const& orchestratorConfig);

private:
    friend class Serialization;

    /// @brief The type of communication protocol used. Default is MPI.
    CommunicationType mCommType;

    /// @brief The mode of communication. See CommunicationMode.
    CommunicationMode mCommMode;

    /// @brief The GPU device ids to use for executing this model
    std::optional<std::vector<SizeType32>> mDeviceIds;

    /// @brief The participant ids (MPI ranks for example) used for executing this model
    std::optional<std::vector<SizeType32>> mParticipantIds;

    /// @brief Optional orchestrator configuration
    std::optional<OrchestratorConfig> mOrchestratorConfig;
};

/// @brief config for PeftCacheManager
class PeftCacheConfig
{
public:
    explicit PeftCacheConfig(SizeType32 numHostModuleLayer = 0, SizeType32 numDeviceModuleLayer = 0,
        SizeType32 optimalAdapterSize = 8, SizeType32 maxAdapterSize = 64, SizeType32 numPutWorkers = 1,
        SizeType32 numEnsureWorkers = 1, SizeType32 numCopyStreams = 1, SizeType32 maxPagesPerBlockHost = 24,
        SizeType32 maxPagesPerBlockDevice = 8, std::optional<float> const& deviceCachePercent = std::nullopt,
        std::optional<size_t> const& hostCacheSize = std::nullopt);

    bool operator==(PeftCacheConfig const& other) const;

    [[nodiscard]] SizeType32 getNumHostModuleLayer() const;
    [[nodiscard]] SizeType32 getNumDeviceModuleLayer() const;
    [[nodiscard]] SizeType32 getOptimalAdapterSize() const;
    [[nodiscard]] SizeType32 getMaxAdapterSize() const;
    [[nodiscard]] SizeType32 getNumPutWorkers() const;
    [[nodiscard]] SizeType32 getNumEnsureWorkers() const;
    [[nodiscard]] SizeType32 getNumCopyStreams() const;
    [[nodiscard]] SizeType32 getMaxPagesPerBlockHost() const;
    [[nodiscard]] SizeType32 getMaxPagesPerBlockDevice() const;
    [[nodiscard]] std::optional<float> getDeviceCachePercent() const;
    [[nodiscard]] std::optional<size_t> getHostCacheSize() const;

private:
    friend class Serialization;

    // number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache
    SizeType32 mNumHostModuleLayer;
    // number of max sized 1-layer 1-module sets of weights that can be stored in host cache
    SizeType32 mNumDeviceModuleLayer;
    // optimal adapter size used to set page width
    SizeType32 mOptimalAdapterSize;
    // max supported adapter size. Used to compute minimum
    SizeType32 mMaxAdapterSize;
    // number of worker threads used to put weights into host cache
    SizeType32 mNumPutWorkers;
    // number of worker threads used to copy weights from host to device
    SizeType32 mNumEnsureWorkers;
    // number of streams used to copy weights from host to device
    SizeType32 mNumCopyStreams;
    // Number of cache pages per allocation block (host)
    SizeType32 mMaxPagesPerBlockHost;
    // Number of cache pages per allocation block (device)
    SizeType32 mMaxPagesPerBlockDevice;
    // percent of memory after engine load to use for cache
    std::optional<FloatType> mDeviceCachePercent;
    // size in bytes to use for host cache
    std::optional<size_t> mHostCacheSize;
};

/// @brief Configuration class for the decoding.
class DecodingConfig
{
public:
    explicit DecodingConfig(std::optional<DecodingMode> decodingMode = std::nullopt,
        std::optional<LookaheadDecodingConfig> lookaheadDecodingConfig = std::nullopt,
        std::optional<MedusaChoices> medusaChoices = std::nullopt);

    bool operator==(DecodingConfig const& other) const;

    // Decoding mode.
    /// @brief Sets decoding mode. Some modes require the use of their own setters.
    void setDecodingMode(DecodingMode const&);
    [[nodiscard]] std::optional<DecodingMode> getDecodingMode() const;

    // Lookahead methods.
    /// @brief Sets lookahead decoding mode and config.
    void setLookaheadDecoding(LookaheadDecodingConfig const& lookaheadDecodingConfig);
    [[nodiscard]] std::optional<LookaheadDecodingConfig> getLookaheadDecodingConfig() const;

    // Medusa methods.
    /// @brief Sets medusa mode and config.
    void setMedusaChoices(MedusaChoices const&);
    [[nodiscard]] std::optional<MedusaChoices> getMedusaChoices() const;

private:
    friend class Serialization;

    // Decoding mode.
    std::optional<DecodingMode> mDecodingMode;
    // Lookahead params.
    std::optional<LookaheadDecodingConfig> mLookaheadDecodingConfig;
    // Medusa params.
    std::optional<MedusaChoices> mMedusaChoices;
};

class LogitsPostProcessorConfig
{
public:
    explicit LogitsPostProcessorConfig(std::optional<LogitsPostProcessorMap> processorMap = std::nullopt,
        std::optional<LogitsPostProcessorBatched> processorBatched = std::nullopt, bool replicate = true);

    [[nodiscard]] std::optional<LogitsPostProcessorMap> getProcessorMap() const;
    [[nodiscard]] std::optional<LogitsPostProcessorBatched> getProcessorBatched() const;
    [[nodiscard]] bool getReplicate() const;

    void setProcessorMap(LogitsPostProcessorMap const& processorMap);
    void setProcessorBatched(LogitsPostProcessorBatched const& processorBatched);
    void setReplicate(bool replicate);

private:
    /// @brief mapping from post processor names to non-batched post processors
    std::optional<LogitsPostProcessorMap> mProcessorMap;
    /// @brief single batched post processor
    std::optional<LogitsPostProcessorBatched> mProcessorBatched;
    /// @brief If set to true, logits post processor will run on all TP ranks in last PP rank
    bool mReplicate;
};

/// @brief Configuration class for the model executor
class ExecutorConfig
{
public:
    explicit ExecutorConfig(SizeType32 maxBeamWidth = 1, SchedulerConfig const& schedulerConfig = SchedulerConfig(),
        KvCacheConfig const& kvCacheConfig = KvCacheConfig(), bool enableChunkedContext = false,
        bool normalizeLogProbs = true, SizeType32 iterStatsMaxIterations = kDefaultIterStatsMaxIterations,
        SizeType32 requestStatsMaxIterations = kDefaultRequestStatsMaxIterations,
        BatchingType batchingType = BatchingType::kINFLIGHT, std::optional<SizeType32> maxBatchSize = std::nullopt,
        std::optional<SizeType32> maxNumTokens = std::nullopt,
        std::optional<ParallelConfig> parallelConfig = std::nullopt,
        std::optional<PeftCacheConfig> const& peftCacheConfig = std::nullopt,
        std::optional<LogitsPostProcessorConfig> logitsPostProcessorConfig = std::nullopt,
        std::optional<DecodingConfig> decodingConfig = std::nullopt, float gpuWeightsPercent = 1,
        std::optional<SizeType32> maxQueueSize = std::nullopt,
        ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig = ExtendedRuntimePerfKnobConfig(),
        std::optional<DebugConfig> debugConfig = std::nullopt, SizeType32 recvPollPeriodMs = 0,
        uint64_t maxSeqIdleMicroseconds = 180000000);

    [[nodiscard]] SizeType32 getMaxBeamWidth() const;
    [[nodiscard]] SchedulerConfig getSchedulerConfig() const;
    [[nodiscard]] KvCacheConfig getKvCacheConfig() const;
    [[nodiscard]] bool getEnableChunkedContext() const;
    [[nodiscard]] bool getNormalizeLogProbs() const;
    [[nodiscard]] SizeType32 getIterStatsMaxIterations() const;
    [[nodiscard]] SizeType32 getRequestStatsMaxIterations() const;
    [[nodiscard]] BatchingType getBatchingType() const;
    [[nodiscard]] std::optional<SizeType32> getMaxBatchSize() const;
    [[nodiscard]] std::optional<SizeType32> getMaxNumTokens() const;
    [[nodiscard]] std::optional<ParallelConfig> getParallelConfig() const;
    [[nodiscard]] std::optional<PeftCacheConfig> getPeftCacheConfig() const;
    [[nodiscard]] std::optional<LogitsPostProcessorConfig> getLogitsPostProcessorConfig() const;
    [[nodiscard]] std::optional<DecodingConfig> getDecodingConfig() const;
    [[nodiscard]] float getGpuWeightsPercent() const;
    [[nodiscard]] std::optional<SizeType32> getMaxQueueSize() const;
    [[nodiscard]] ExtendedRuntimePerfKnobConfig getExtendedRuntimePerfKnobConfig() const;
    [[nodiscard]] std::optional<DebugConfig> getDebugConfig() const;
    [[nodiscard]] SizeType32 getRecvPollPeriodMs() const;
    [[nodiscard]] uint64_t getMaxSeqIdleMicroseconds() const;

    void setMaxBeamWidth(SizeType32 maxBeamWidth);
    void setMaxBatchSize(SizeType32 maxBatchSize);
    void setMaxNumTokens(SizeType32 maxNumTokens);
    void setSchedulerConfig(SchedulerConfig const& schedulerConfig);
    void setKvCacheConfig(KvCacheConfig const& kvCacheConfig);
    void setEnableChunkedContext(bool enableChunkedContext);
    void setNormalizeLogProbs(bool normalizeLogProbs);
    void setIterStatsMaxIterations(SizeType32 iterStatsMaxIterations);
    void setRequestStatsMaxIterations(SizeType32 requestStatsMaxIterations);
    void setBatchingType(BatchingType batchingType);
    void setParallelConfig(ParallelConfig const& parallelConfig);
    void setPeftCacheConfig(PeftCacheConfig const& peftCacheConfig);
    void setLogitsPostProcessorConfig(LogitsPostProcessorConfig const& logitsPostProcessorConfig);
    void setDecodingConfig(DecodingConfig const& decodingConfig);
    void setGpuWeightsPercent(float const& gpuWeightsPercent);
    void setMaxQueueSize(std::optional<SizeType32> const& maxQueueSize);
    void setExtendedRuntimePerfKnobConfig(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig);
    void setDebugConfig(DebugConfig const& debugConfig);
    void setRecvPollPeriodMs(SizeType32 const& recvPollPeriodMs);
    void setMaxSeqIdleMicroseconds(uint64_t maxNumTokens);

private:
    friend class Serialization;

    /// @brief The beam width value of requests that will be sent to the executor
    SizeType32 mMaxBeamWidth;

    /// @brief The scheduler configuration.
    SchedulerConfig mSchedulerConfig;

    /// @brief The KV cache configuration.
    KvCacheConfig mKvCacheConfig;

    /// @brief The KV cache configuration.
    bool mEnableChunkedContext;

    /// @brief Controls if log probabilities should be normalized or not.
    bool mNormalizeLogProbs;

    /// @brief Controls the maximum number of iterations for which to keep statistics.
    SizeType32 mIterStatsMaxIterations;

    /// @brief Controls the maximum number of iterations for which to keep per-request statistics.
    SizeType32 mRequestStatsMaxIterations;

    /// @brief The type of batching strategy to use. See BatchingType.
    BatchingType mBatchingType;

    /// @brief The max batch size of requests
    std::optional<SizeType32> mMaxBatchSize;

    /// @brief The max number of tokens per batch
    std::optional<SizeType32> mMaxNumTokens;

    /// @brief The parallel execution configuration.
    std::optional<ParallelConfig> mParallelConfig;
    std::optional<PeftCacheConfig> mPeftCacheConfig;

    /// @brief Logits post processor configuration
    std::optional<LogitsPostProcessorConfig> mLogitsPostProcessorConfig;

    /// @brief Decoding configuration.
    std::optional<DecodingConfig> mDecodingConfig;

    /// @brief GPU weights percent for weight streaming.
    float mGpuWeightsPercent;

    /// @brief The maximum number of requests allowed in queue before rejecting new requests.
    std::optional<SizeType32> mMaxQueueSize;

    /// @brief Config for perf knobs that can be set in runtime.
    ExtendedRuntimePerfKnobConfig mExtendedRuntimePerfKnobConfig;

    /// @brief Debugging configuration.
    std::optional<DebugConfig> mDebugConfig;

    /// @brief The time in ms between polls for new communication in orchestrator mode. Use 0 for busy loop.
    SizeType32 mRecvPollPeriodMs;

    /// @brief The maximum time in microseconds a scheduled request can remain idle before getting terminated. Default
    /// is 3 minutes.
    uint64_t mMaxSeqIdleMicroseconds;
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

    Executor(std::filesystem::path const& encoderModelPath, std::filesystem::path const& decoderModelPath,
        ModelType modelType, ExecutorConfig const& executorConfig);

    Executor(BufferView const& engineBuffer, std::string const& jsonConfigStr, ModelType modelType,
        ExecutorConfig const& executorConfig,
        std::optional<std::map<std::string, Tensor>> const& managedWeights = std::nullopt);

    Executor(BufferView const& encoderEngineBuffer, std::string const& encoderJsonConfigStr,
        BufferView const& decoderEngineBuffer, std::string const& decoderJsonConfigStr, ModelType modelType,
        ExecutorConfig const& executorConfig);

    Executor(std::shared_ptr<Model> model, ExecutorConfig const& executorConfig);

    Executor(
        std::shared_ptr<Model> encoderModel, std::shared_ptr<Model> decoderModel, ExecutorConfig const& executorConfig);

    ~Executor();

    /// @brief Enqueue a new request
    /// @param request The LLM request which contains input tokens and request parameters
    /// @return A unique id that identifies the request
    [[nodiscard]] IdType enqueueRequest(Request const& request);

    /// @brief Enqueue a batch of request
    [[nodiscard]] std::vector<IdType> enqueueRequests(std::vector<Request> const& requests);

    /// @brief Await for ready responses
    ///
    ///        This overload awaits for any ready responses. In particular, if several requests
    ///        have been enqueued, this method will provide any ready responses without order guarantees.
    /// @param timeout The maximum time to wait for new responses
    /// @return A vector of responses
    [[nodiscard]] std::vector<Response> awaitResponses(
        std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    /// @brief Await for ready responses
    /// @param id A request id
    /// @param timeout The maximum time to wait for new responses
    /// @return A vector of responses
    [[nodiscard]] std::vector<Response> awaitResponses(
        IdType const& requestId, std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    /// @brief Await for multiple ready responses
    ///
    ///        A multiple ID request behaves as if awaitResponses(IdType, timeout)
    ///        were invoked on all IDs. The returned vector contains
    ///        a vector of responses per ID in the same order specified by the requestIds.
    ///        The same behaviour as awaitResponses(IdType, timeout) applies:
    ///        * Responses may be empty.
    ///        * If all responses have already been given for one of the requestIds,
    ///          then this method will hang unless a timeout is specified.
    /// @param requestIds Ids requested
    /// @param timeout The maximum time to wait for new responses
    /// @return A vector of vector of responses
    [[nodiscard]] std::vector<std::vector<Response>> awaitResponses(
        std::vector<IdType> const& requestIds, std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    /// @brief Get the number of ready responses
    /// @param requestId An optional request id
    /// @return The number of ready responses
    [[nodiscard]] SizeType32 getNumResponsesReady(std::optional<IdType> const& requestId = std::nullopt) const;

    /// @brief Cancel the request with provided request id
    /// @param id The request id for which to cancel the response
    void cancelRequest(IdType requestId);

    /// @brief   Signals the server to shutdown.
    /// @details This call is blocking. Only returns when all requests have terminated or timeout has been reached
    void shutdown();

    /// @brief  Returns the per-iterations statistics computed since last call to getLatestIterationStats.
    ///         Contains at most iterStatsMaxIterations iterations.
    /// @return Iteration stats
    std::deque<IterationStats> getLatestIterationStats();

    /// @brief  Returns the request stats of each iteration computed since last call to getLatestRequestStats.
    ///         Contains at most requestStatsMaxIterations iterations.
    /// @return Request stats grouped by iterations
    std::deque<RequestStatsPerIteration> getLatestRequestStats();

    /// @brief  Returns the debug tensors of each iteration computed since last call to getLatestDebugTensors.
    ///         Contains at most debugTensorsMaxIterations iterations.
    /// @return Request debug tensors grouped by iterations
    std::deque<DebugTensorsPerIteration> getLatestDebugTensors();

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
