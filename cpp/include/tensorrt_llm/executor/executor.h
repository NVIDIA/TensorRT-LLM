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
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/runtimeDefaults.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tensorrt_llm::mpi
{
class MpiComm;
} // namespace tensorrt_llm::mpi

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class BaseKVCacheManager;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::executor
{

/// @brief Version of TRT-LLM
char const* version() noexcept;

class Model;
class Serialization;
class DataTransceiverState;

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
        std::optional<SizeType32> const& noRepeatNgramSize = std::nullopt,
        std::optional<SizeType32> const& numReturnSequences = std::nullopt,
        std::optional<FloatType> const& minP = std::nullopt,
        std::optional<std::vector<SizeType32>> const& beamWidthArray = std::nullopt);

    bool operator==(SamplingConfig const& other) const;

    [[nodiscard]] SizeType32 getBeamWidth() const;
    [[nodiscard]] SizeType32 getNumReturnBeams() const;
    [[nodiscard]] std::optional<SizeType32> getTopK() const;
    [[nodiscard]] std::optional<FloatType> getTopP() const;
    [[nodiscard]] std::optional<FloatType> getTopPMin() const;
    [[nodiscard]] std::optional<SizeType32> getTopPResetIds() const;
    [[nodiscard]] std::optional<FloatType> getTopPDecay() const;
    [[nodiscard]] std::optional<RandomSeedType> getSeed() const;
    [[nodiscard]] std::optional<FloatType> getTemperature() const;
    [[nodiscard]] std::optional<SizeType32> getMinTokens() const;
    [[nodiscard]] std::optional<FloatType> getBeamSearchDiversityRate() const;
    [[nodiscard]] std::optional<FloatType> getRepetitionPenalty() const;
    [[nodiscard]] std::optional<FloatType> getPresencePenalty() const;
    [[nodiscard]] std::optional<FloatType> getFrequencyPenalty() const;
    [[nodiscard]] std::optional<FloatType> getLengthPenalty() const;
    [[nodiscard]] std::optional<SizeType32> getEarlyStopping() const;
    [[nodiscard]] std::optional<SizeType32> getNoRepeatNgramSize() const;
    [[nodiscard]] std::optional<SizeType32> getNumReturnSequences() const;
    [[nodiscard]] std::optional<FloatType> getMinP() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getBeamWidthArray() const;

    void setBeamWidth(SizeType32 beamWidth);
    void setTopK(std::optional<SizeType32> const& topK);
    void setTopP(std::optional<FloatType> const& topP);
    void setTopPMin(std::optional<FloatType> const& topPMin);
    void setTopPResetIds(std::optional<TokenIdType> const& topPResetIds);
    void setTopPDecay(std::optional<FloatType> const& topPDecay);
    void setSeed(std::optional<RandomSeedType> const& seed);
    void setTemperature(std::optional<FloatType> const& temperature);
    void setMinTokens(std::optional<SizeType32> const& minTokens);
    void setBeamSearchDiversityRate(std::optional<FloatType> const& beamSearchDiversityRate);
    void setRepetitionPenalty(std::optional<FloatType> const& repetitionPenalty);
    void setPresencePenalty(std::optional<FloatType> const& presencePenalty);
    void setFrequencyPenalty(std::optional<FloatType> const& frequencyPenalty);
    void setLengthPenalty(std::optional<FloatType> const& lengthPenalty);
    void setEarlyStopping(std::optional<SizeType32> const& earlyStopping);
    void setNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize);
    void setNumReturnSequences(std::optional<SizeType32> const& numReturnSequences);
    void setMinP(std::optional<FloatType> const& minP);
    void setBeamWidthArray(std::optional<std::vector<SizeType32>> const& beamWidthArray);

private:
    static SizeType32 checkBeamWidth(SizeType32 beamWidth);
    static std::optional<FloatType> const& checkTopK(std::optional<FloatType> const& topK);
    static std::optional<FloatType> const& checkTopP(std::optional<FloatType> const& topP);
    static std::optional<FloatType> const& checkTopPMin(std::optional<FloatType> const& topPMin);
    static std::optional<TokenIdType> const& checkTopPResetIds(std::optional<TokenIdType> const& topPResetIds);
    static std::optional<FloatType> const& checkTopPDecay(std::optional<FloatType> const& topPDecay);
    static std::optional<FloatType> const& checkTemperature(std::optional<FloatType> const& temperature);
    static std::optional<SizeType32> const& checkMinTokens(std::optional<SizeType32> const& minTokens);
    static std::optional<FloatType> const& checkBeamSearchDiversityRate(
        std::optional<FloatType> const& beamSearchDiversityRate);
    static std::optional<FloatType> const& checkRepetitionPenalty(std::optional<FloatType> const& repetitionpenalty);
    static std::optional<FloatType> const& checkLengthPenalty(std::optional<FloatType> const& lengthPenalty);
    static std::optional<SizeType32> const& checkEarlyStopping(std::optional<SizeType32> const& earlyStopping);
    static std::optional<SizeType32> const& checkNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize);
    static std::optional<SizeType32> const& checkNumReturnSequences(
        std::optional<SizeType32> const& numReturnSequences, SizeType32 beamWidth);
    static std::optional<FloatType> const& checkMinP(std::optional<FloatType> const& minP);
    static std::pair<std::optional<std::vector<SizeType32>> const&, SizeType32 const> const checkBeamWidthArray(
        std::optional<std::vector<SizeType32>> const& beamWidthArray, SizeType32 const beamWidth);
    void updateNumReturnBeams();

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
    /// @brief Controls the random seed used by the random number generator in sampling. Default is 0.
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
    /// end_token). Default is 1.
    std::optional<SizeType32> mEarlyStopping;
    /// @brief Controls how many repeat ngram size are acceptable. Default is 1 << 30.
    std::optional<SizeType32> mNoRepeatNgramSize;
    /// @brief The number of return sequences or beams. In beam search, the value should be less than or equal to
    /// mBeamWidth. In sampling, it specifies the total number of independently generated sequences.
    std::optional<SizeType32> mNumReturnSequences;
    /// @brief The number of beams to return. It is equal to beamWidth unless numReturnSequences is set.
    /// If beamWidth > 1 and numReturnSequences is set, then numReturnBeams is equal to numReturnSequences.
    SizeType32 mNumReturnBeams;
    /// @brief Controls the min_p scaling for sampling.
    /// It masks x which P_x < min_p * P_max, where P_x is probability of candidate x. Default is 0.f
    std::optional<FloatType> mMinP;
    /// @brief Controls the beam width for each step for Variable-Beam-Width-Search.
    std::optional<std::vector<SizeType32>> mBeamWidthArray;
};

/// @brief Additional output that should be gathered.
/// @details By default gather output of shape [beamWidth, x] from each generation phase.
///          If gatherContext is true, also gather output of shape [promptLen, x] from context phase.
class AdditionalModelOutput
{
public:
    explicit AdditionalModelOutput(std::string name, bool gatherContext = false);

    bool operator==(AdditionalModelOutput const& other) const;

    std::string name;
    bool gatherContext{false};
};

/// @brief Configuration that controls the outputs of a Result
class OutputConfig
{
public:
    explicit OutputConfig(bool returnLogProbs = false, bool returnContextLogits = false,
        bool returnGenerationLogits = false, bool excludeInputFromOutput = false, bool returnEncoderOutput = false,
        bool returnPerfMetrics = false,
        std::optional<std::vector<AdditionalModelOutput>> additionalModelOutputs = std::nullopt);

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
    /// @brief Controls if Result should contain performance metrics
    bool returnPerfMetrics;

    /// @brief The additional outputs to gather from the model.
    std::optional<std::vector<AdditionalModelOutput>> additionalModelOutputs;
};

/// @brief Configuration for speculative decoding with external draft tokens.
/// Allows to include draft tokens, draft logits and specify acceptance threshold.
class ExternalDraftTokensConfig
{
public:
    explicit ExternalDraftTokensConfig(VecTokens tokens, std::optional<Tensor> logits = std::nullopt,
        std::optional<FloatType> const& acceptanceThreshold = std::nullopt,
        std::optional<bool> const& fastLogits = std::nullopt);

    [[nodiscard]] VecTokens getTokens() const;
    [[nodiscard]] std::optional<Tensor> getLogits() const;
    [[nodiscard]] std::optional<FloatType> getAcceptanceThreshold() const;
    [[nodiscard]] std::optional<bool> getFastLogits() const;

private:
    friend class Serialization;
    /// @brief The draft tokens
    VecTokens mTokens;
    /// @brief The draft logits. Expected shape: [num_draft_tokens, vocab_size].
    std::optional<Tensor> mLogits;
    /// @brief The acceptance threshold. Must be > 0.f and <= 1.f
    std::optional<FloatType> mAcceptanceThreshold;
    /// @brief Use direct transfer for draft logits
    std::optional<bool> mFastLogits;
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

/// @brief Multimodal input data class
class MultimodalInput
{
public:
    explicit MultimodalInput(std::vector<std::vector<SizeType32>> multimodalHashes,
        std::vector<SizeType32> multimodalPositions, std::vector<SizeType32> multimodalLengths);

    [[nodiscard]] std::vector<std::vector<SizeType32>> getMultimodalHashes() const;
    [[nodiscard]] std::vector<SizeType32> getMultimodalPositions() const;
    [[nodiscard]] std::vector<SizeType32> getMultimodalLengths() const;

private:
    friend class Serialization;
    /// @brief The multimodal hashes
    std::vector<std::vector<SizeType32>> mMultimodalHashes;
    /// @brief The multimodal positions
    std::vector<SizeType32> mMultimodalPositions;
    /// @brief The multimodal lengths
    std::vector<SizeType32> mMultimodalLengths;
};

/// @brief Configuration for mrope
class MropeConfig
{
public:
    explicit MropeConfig(Tensor mropeRoratySinCos, SizeType32 mropePositionDeltas);

    [[nodiscard]] Tensor getMRopeRotaryCosSin() const;
    [[nodiscard]] SizeType32 getMRopePositionDeltas() const;

private:
    friend class Serialization;
    /// @brief The mrope rotary sin and cos cache. Expected shape: [maxPositionEmbeddings*rotaryEmbeddingDim],Data type
    /// must float32
    Tensor mMRopeRotaryCosSin;
    /// @brief The mrope position deltas
    SizeType32 mMRopePositionDeltas;
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

/// @brief Configuration for Look-Ahead speculative decoding.
/// Allows to include window size, ngram size and verification set size
struct LookaheadDecodingConfig
{
    LookaheadDecodingConfig(SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize);

    explicit LookaheadDecodingConfig()
        : LookaheadDecodingConfig(
            kDefaultLookaheadDecodingWindow, kDefaultLookaheadDecodingNgram, kDefaultLookaheadDecodingVerificationSet)
    {
    }

    bool operator==(LookaheadDecodingConfig const& other) const;
    [[nodiscard]] std::tuple<SizeType32 const, SizeType32 const, SizeType32 const> get() const;
    [[nodiscard]] SizeType32 getWindowSize() const;
    [[nodiscard]] SizeType32 getNgramSize() const;
    [[nodiscard]] SizeType32 getVerificationSetSize() const;

    /// @brief return <maxDecodingTokens, maxPathLen, maxDraftTokens, maxDraftPathLen>
    [[nodiscard]] std::tuple<SizeType32, SizeType32, SizeType32, SizeType32> calculateSpeculativeResource() const;
    static std::tuple<SizeType32, SizeType32, SizeType32, SizeType32> calculateSpeculativeResourceTuple(
        SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize);

    /// @brief return true when `this` can be executed on resources defined by `that`
    [[nodiscard]] bool isLE(LookaheadDecodingConfig const& that) const;

    /// @brief return true when the parameter combination is valid.
    static bool isLegal(SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize) noexcept;

    static constexpr SizeType32 kDefaultLookaheadDecodingWindow = 4;
    static constexpr SizeType32 kDefaultLookaheadDecodingNgram = 3;
    static constexpr SizeType32 kDefaultLookaheadDecodingVerificationSet = 4;

private:
    friend class Serialization;

    // Number of NGrams in lookahead branch per step.
    SizeType32 mWindowSize;
    // Number of tokens per NGram.
    SizeType32 mNgramSize;
    // Number of NGrams in verification branch per step.
    SizeType32 mVerificationSetSize;
};

struct EagleConfig
{
    explicit EagleConfig(std::optional<EagleChoices> eagleChoices = std::nullopt, bool greedySampling = true,
        std::optional<float> posteriorThreshold = std::nullopt, bool useDynamicTree = false,
        std::optional<SizeType32> dynamicTreeMaxTopK = std::nullopt);

    bool operator==(EagleConfig const& other) const;
    [[nodiscard]] std::optional<EagleChoices> getEagleChoices() const;
    [[nodiscard]] std::optional<float> getPosteriorThreshold() const;
    [[nodiscard]] bool isGreedySampling() const;
    [[nodiscard]] bool useDynamicTree() const;
    [[nodiscard]] std::optional<SizeType32> getDynamicTreeMaxTopK() const;

private:
    std::optional<float> const& checkPosteriorValue(std::optional<float> const& value);

private:
    friend class Serialization;

    /// @brief choices forming tree for EAGLE-1.
    std::optional<EagleChoices> mEagleChoices;

    /// @brief Flag to use greedy or typical acceptance.
    bool mGreedySampling;
    /// @brief Minimum token probability of the typical acceptance.
    /// Corresponds to epsilon in https://arxiv.org/pdf/2401.10774.
    /// Default is 0.09f.
    std::optional<float> mPosteriorThreshold;

    /// @brief Flag to use Eagle-2
    bool mUseDynamicTree;

    /// @brief Number of draft tokens expand for each node in Eagle-2
    std::optional<SizeType32> mDynamicTreeMaxTopK;
};

class ContextPhaseParams
{
public:
    using RequestIdType = std::uint64_t;

    ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId, std::optional<VecTokens> draftTokens);
    ContextPhaseParams(
        VecTokens firstGenTokens, RequestIdType reqId, void* state, std::optional<VecTokens> draftTokens);
    ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId, std::vector<char> const& serializedState,
        std::optional<VecTokens> draftTokens);

    ContextPhaseParams(ContextPhaseParams const&);
    ContextPhaseParams(ContextPhaseParams&&) noexcept;
    ContextPhaseParams& operator=(ContextPhaseParams const&);
    ContextPhaseParams& operator=(ContextPhaseParams&&) noexcept;
    ~ContextPhaseParams();

    [[nodiscard]] bool operator==(ContextPhaseParams const&) const noexcept;

    [[nodiscard]] VecTokens const& getFirstGenTokens() const& noexcept;
    [[nodiscard]] std::optional<VecTokens> const& getDraftTokens() const& noexcept;
    [[nodiscard]] VecTokens popFirstGenTokens() && noexcept;
    [[nodiscard]] RequestIdType getReqId() const noexcept;

    [[nodiscard]] void const* getState() const noexcept;
    [[nodiscard]] void* getState() noexcept;
    [[nodiscard]] void* releaseState() noexcept;
    [[nodiscard]] std::vector<char> getSerializedState() const noexcept;

private:
    friend class Serialization;
    static void deleter(void const* data);
    using StatePtr = std::unique_ptr<void, decltype(&deleter)>;

    /// @brief This request corresponds to the request ID in the context phase.
    RequestIdType mReqId{0};

    /// @brief The first tokens generated by context executor
    VecTokens mFirstGenTokens;

    /// @brief Context phase state of this request
    StatePtr mState{nullptr, deleter};

    /// @brief The draft tokens generated by context executor
    std::optional<VecTokens> mDraftTokens;
};

/// @brief Configuration for speculative decoding (both draft and target models)
class SpeculativeDecodingConfig
{
public:
    explicit SpeculativeDecodingConfig(bool fastLogits = false);

    bool operator==(SpeculativeDecodingConfig const& other) const;

    /// @brief Send logits tensor directly from draft to target model.
    bool fastLogits;

private:
    friend class Serialization;
};

/// @brief Guided decoding parameters for a request.
class GuidedDecodingParams
{
public:
    enum class GuideType
    {
        /// @brief The generated text is amenable to json format.
        kJSON = 0,

        /// @brief The generated text is amenable to json format with additional user-specified restrictions, namely
        /// schema.
        kJSON_SCHEMA = 1,

        /// @brief The generated text is amenable to the user-specified regular expression.
        kREGEX = 2,

        /// @brief The generated text is amenable to the user-specified extended Backus-Naur form (EBNF) grammar.
        /// EBNF grammar is widely-used to express context-free grammars.
        kEBNF_GRAMMAR = 3,

        /// @brief The generated text is amenable to the XGrammar structural tag.
        kSTRUCTURAL_TAG = 4,
    };

    explicit GuidedDecodingParams(GuideType guideType, std::optional<std::string> guide = std::nullopt);

    bool operator==(GuidedDecodingParams const& other) const;
    [[nodiscard]] GuideType getGuideType() const;
    [[nodiscard]] std::optional<std::string> getGuide() const;

private:
    friend class Serialization;

    /// @brief The guide type. See GuideType.
    GuideType mGuideType;
    /// @brief The detailed guide string. It could be a json schema, a regular expression or a EBNF grammar depending on
    /// mGuideType.
    std::optional<std::string> mGuide;
};

using RetentionPriority = SizeType32;

struct RetentionPriorityAndDuration
{

    RetentionPriorityAndDuration(std::optional<RetentionPriority> const& retentionPriority,
        std::optional<std::chrono::milliseconds> const& durationMs)
        : retentionPriority{retentionPriority}
        , durationMs{durationMs}
    {
    }

    std::optional<RetentionPriority> retentionPriority;
    std::optional<std::chrono::milliseconds> durationMs;
};

/// @brief Configuration for the request's retention in the KV Cache
class KvCacheRetentionConfig
{

public:
    static constexpr RetentionPriority kMinRetentionPriority = 0;
    static constexpr RetentionPriority kMaxRetentionPriority = 100;
    static constexpr RetentionPriority kDefaultRetentionPriority = 35;

    /// @brief A single entry to set block priorities over a token range. Earlier ranges always take priority over later
    /// ones. For example, with a block size of 16, a range of [0, 17] would be applied to the first two blocks.
    struct TokenRangeRetentionConfig
    {
    public:
        explicit TokenRangeRetentionConfig(SizeType32 tokenStart, std::optional<SizeType32> tokenEnd = std::nullopt,
            RetentionPriority priority = KvCacheRetentionConfig::kDefaultRetentionPriority,
            std::optional<std::chrono::milliseconds> durationMs = std::nullopt);

        bool operator==(TokenRangeRetentionConfig const& other) const;

        /// @brief The first token of this range.
        SizeType32 tokenStart;
        /// @brief The final token of this range. The end is not included in the range. This can be set to std::nullopt
        /// to extend the range to the end of the sequence.
        std::optional<SizeType32> tokenEnd;
        /// @brief The priority of this token range. Higher priorities are less likely to be evicted or offloaded.
        RetentionPriority priority;
        /// @brief The duration in ms that the block should remain at the given priority level. Set to std::nullopt to
        /// have no expiration time, and keep the block at the given priority level until it gets reclaimed. After the
        /// duration has passed, the block will be moved back to the `kDefaultRetentionPriority` level.
        std::optional<std::chrono::milliseconds> durationMs;
    };

    explicit KvCacheRetentionConfig()
        : KvCacheRetentionConfig({}, kDefaultRetentionPriority)
    {
    }

    explicit KvCacheRetentionConfig(std::vector<TokenRangeRetentionConfig> const& tokenRangeRetentionPriorities,
        RetentionPriority decodeRetentionPriority = kDefaultRetentionPriority,
        std::optional<std::chrono::milliseconds> decodeDurationMs = std::nullopt,
        KvCacheTransferMode transferMode = KvCacheTransferMode::DRAM, std::string const& directory = "");

    [[nodiscard]] std::vector<TokenRangeRetentionConfig> getTokenRangeRetentionConfigs() const;
    [[nodiscard]] RetentionPriority getDecodeRetentionPriority() const;
    [[nodiscard]] std::optional<std::chrono::milliseconds> getDecodeDurationMs() const;
    [[nodiscard]] KvCacheTransferMode getTransferMode() const;
    [[nodiscard]] std::string const& getDirectory() const;

    /// @brief Convert the token range data into an entry per kv block. Returns a tuple of vectors corresponding to the
    /// priorities and durations for each block.
    [[nodiscard]] std::vector<RetentionPriorityAndDuration> getPerBlockRetentionPriorityDuration(
        SizeType32 blockSize, SizeType32 seqLen) const;

    bool operator==(KvCacheRetentionConfig const& other) const
    {
        return mTokenRangeRetentionConfigs == other.mTokenRangeRetentionConfigs
            && mDecodeRetentionPriority == other.mDecodeRetentionPriority
            && mDecodeDurationMs == other.mDecodeDurationMs && mTransferMode == other.mTransferMode
            && mDirectory == other.mDirectory;
    }

private:
    /// @brief The token ranges and priority levels to update. Ranges must be non-overlapping. For example [(0, 64),
    /// (100, 128), (70, 80)] is valid, whereas
    /// [(0, 64), (60, 128)] is not.
    std::vector<TokenRangeRetentionConfig> mTokenRangeRetentionConfigs;
    /// @brief The priority level to assign to blocks allocated in the decode phase
    RetentionPriority mDecodeRetentionPriority;
    /// @brief The duration in ms that decode blocks should remain at their assigned priority level.
    std::optional<std::chrono::milliseconds> mDecodeDurationMs;
    /// @brief The transfer mode for the block.
    KvCacheTransferMode mTransferMode;
    /// @brief Name of the directory if transfer mode is GDS or POSIX_DEBUG_FALLBACK.
    std::string mDirectory;
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
    /// @param embeddingBias The embedding bias tensor. Expected shape is [vocab_size]
    /// @param externalDraftTokensConfig The speculative decoding with external draft tokens configuration
    /// @param pTuningConfig The prompt tuning configuration
    /// @param multimodalInput The multimodal input {multimodalHashes, multimodalPositions, multimodalLengths}
    /// @param multimodalEmbedding The multimodal embedding tensor. Expected shape is [num_multimodal_tokens,
    /// hidden_dim]
    /// @param mRopeConfig The mrope configuration
    /// @param loraConfig The LoRA configuration
    /// @param lookaheadConfig The lookahead speculative decoding configuration
    /// @param kvCacheRetentionConfig The configuration used for KV cache block eviction.
    /// @param logitsPostProcessorName The logits postprocessor name. Must correspond to one of the logits postprocessor
    /// name provided to the ExecutorConfig.
    /// @param logitsPostProcessor The logits postprocessor dynamically specified per request; only supported with
    /// replicate=false or no tensor parallelism.
    /// @param encoderInputTokenIds The encoder input token ids for encoder-decoder models, or encoder-only models
    /// @param clientId
    /// @param returnAllGeneratedTokens Indicates whether to return the full beams or just the newly generated tokens
    /// after every streaming step.
    /// @param priority Sets the execution priority of this request.
    /// @param type Indicate the request type for disaggregated serving mode.
    /// @param contextPhaseParams Generated token ID  from context only executor.
    /// @param encoderInputFeatures Encoder input features for multimodal models.
    /// @param encoderOutputLength Encoder output length if encoder input and output have different lengths (due to
    /// convolution down-sampling, etc.)
    /// @param crossAttentionMask Cross attention mask.
    /// @param numReturnSequences The number of returning sequences.
    /// @param eagleConfig The EAGLE speculative decoding configuration
    /// @param skipCrossAttnBlocks Skip the cross attention transformer blocks or not.
    /// @param guidedDecodingParams The guided decoding parameters.
    /// @param languageAdapterUid Task Uid for language adapter.
    /// @param allottedTimeMs The allotted time in milliseconds after which the request is cancelled with a timedOut
    /// finish reason. The request may exceed this time slightly, but at most by 1 forward pass (in pipeline parallelism
    /// that may involve multiple micro-batches). A request can be timed-out before ever being scheduled.
    /// @param cacheSaltID Salt ID for KV cache blocks to limit the kv cache reuse to the requests with the same string.
    Request(VecTokens inputTokenIds, SizeType32 maxTokens, bool streaming = false,
        SamplingConfig const& samplingConfig = SamplingConfig(), OutputConfig const& outputConfig = OutputConfig(),
        std::optional<SizeType32> const& endId = std::nullopt, std::optional<SizeType32> const& padId = std::nullopt,
        std::optional<std::vector<SizeType32>> positionIds = std::nullopt,
        std::optional<std::list<VecTokens>> badWords = std::nullopt,
        std::optional<std::list<VecTokens>> stopWords = std::nullopt,
        std::optional<Tensor> embeddingBias = std::nullopt,
        std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig = std::nullopt,
        std::optional<PromptTuningConfig> pTuningConfig = std::nullopt,
        std::optional<MultimodalInput> multimodalInput = std::nullopt,
        std::optional<Tensor> multimodalEmbedding = std::nullopt, std::optional<MropeConfig> mRopeConfig = std::nullopt,
        std::optional<LoraConfig> loraConfig = std::nullopt,
        std::optional<LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        std::optional<std::string> logitsPostProcessorName = std::nullopt,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        std::optional<VecTokens> encoderInputTokenIds = std::nullopt, std::optional<IdType> clientId = std::nullopt,
        bool returnAllGeneratedTokens = false, PriorityType priority = kDefaultPriority,
        RequestType type = RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<ContextPhaseParams> contextPhaseParams = std::nullopt,
        std::optional<Tensor> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<Tensor> crossAttentionMask = std::nullopt, SizeType32 numReturnSequences = 1,
        std::optional<EagleConfig> eagleConfig = std::nullopt, std::optional<Tensor> skipCrossAttnBlocks = std::nullopt,
        std::optional<GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<SizeType32> languageAdapterUid = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt,
        std::optional<CacheSaltIDType> cacheSaltID = std::nullopt);

    /// @brief This logits postprocessor name will dispatch to the batched logits postprocessor
    static auto constexpr kBatchedPostProcessorName = "batched";
    /// @brief Dynamic logits postprocessor name will be "dynamic" + requestId
    static auto constexpr kDynamicPostProcessorNamePrefix = "dynamic";

    Request(Request const& other);
    Request(Request&& other) noexcept;
    Request& operator=(Request const& other);
    Request& operator=(Request&& other) noexcept;
    ~Request();

    [[nodiscard]] VecTokens getInputTokenIds() const;
    [[nodiscard]] SizeType32 getMaxTokens() const;
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
    [[nodiscard]] std::optional<MultimodalInput> getMultimodalInput() const;
    [[nodiscard]] std::optional<Tensor> getMultimodalEmbedding() const;
    [[nodiscard]] std::optional<MropeConfig> getMropeConfig() const;
    [[nodiscard]] std::optional<LoraConfig> getLoraConfig() const;
    [[nodiscard]] std::optional<LookaheadDecodingConfig> getLookaheadConfig() const;
    [[nodiscard]] std::optional<KvCacheRetentionConfig> getKvCacheRetentionConfig() const;
    [[nodiscard]] std::optional<std::string> getLogitsPostProcessorName() const;
    [[nodiscard]] std::optional<LogitsPostProcessor> getLogitsPostProcessor() const;
    [[nodiscard]] std::optional<VecTokens> getEncoderInputTokenIds() const;
    [[nodiscard]] std::optional<IdType> getClientId() const;
    [[nodiscard]] PriorityType getPriority() const;
    [[nodiscard]] bool getReturnAllGeneratedTokens() const;
    [[nodiscard]] std::optional<ContextPhaseParams> const& getContextPhaseParams() const;
    [[nodiscard]] std::optional<Tensor> getEncoderInputFeatures() const;
    [[nodiscard]] std::optional<SizeType32> getEncoderOutputLength() const;
    [[nodiscard]] std::optional<Tensor> getCrossAttentionMask() const;
    [[nodiscard]] RequestType getRequestType() const;
    [[nodiscard]] std::optional<EagleConfig> getEagleConfig() const;
    [[nodiscard]] std::optional<Tensor> getSkipCrossAttnBlocks() const;
    [[nodiscard]] std::optional<GuidedDecodingParams> getGuidedDecodingParams() const;
    [[nodiscard]] std::optional<SizeType32> getLanguageAdapterUid() const;
    [[nodiscard]] std::optional<MillisecondsType> getAllottedTimeMs() const;
    [[nodiscard]] std::optional<CacheSaltIDType> getCacheSaltID() const;
    [[nodiscard]] std::optional<std::vector<std::string>> getAdditionalOutputNames() const;

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
    void setMultimodalEmbedding(Tensor const& multimodalEmbedding);
    void setMultimodalInput(MultimodalInput const& multimodalInput);
    void setMropeConfig(MropeConfig const& mRopeConfig);
    void setLoraConfig(LoraConfig const& loraConfig);
    void setLookaheadConfig(LookaheadDecodingConfig const& lookaheadConfig);
    void setKvCacheRetentionConfig(KvCacheRetentionConfig const& kvCacheRetentionConfig);
    void setLogitsPostProcessorName(std::string const& logitsPostProcessorName);
    void setLogitsPostProcessor(std::optional<LogitsPostProcessor> const& logitsPostProcessor);
    void setEncoderInputTokenIds(VecTokens const& encoderInputTokenIds);
    void setClientId(IdType clientId);
    void setPriority(PriorityType priority);
    void setReturnAllGeneratedTokens(bool returnAllGeneratedTokens);
    void setRequestType(RequestType const& requestType);
    void setContextPhaseParams(ContextPhaseParams contextPhaseParams);
    void setEncoderInputFeatures(Tensor encoderInputFeatures);
    void setEncoderOutputLength(SizeType32 encoderOutputLength);
    void setCrossAttentionMask(Tensor crossAttentionMask);
    void setEagleConfig(std::optional<EagleConfig> const& eagleConfig);
    void setSkipCrossAttnBlocks(Tensor skipCrossAttnBlocks);
    void setGuidedDecodingParams(GuidedDecodingParams const& guidedDecodingParams);
    void setLanguageAdapterUid(SizeType32 languageAdapterUid);
    void setAllottedTimeMs(MillisecondsType allottedTimeMs);
    void setCacheSaltID(CacheSaltIDType cacheSaltID);

private:
    friend class Serialization;
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Struct that holds the logits information when using direct transfer
struct SpeculativeDecodingFastLogitsInfo
{
    /// @brief Draft request id
    uint64_t draftRequestId;

    /// @brief MPI world rank of the draft model leader
    int32_t draftParticipantId;

    /// @brief Returns the struct serialized into a tensor that can be used as generation logits input
    [[nodiscard]] Tensor toTensor() const;
};

struct AdditionalOutput
{
    AdditionalOutput(std::string name, Tensor output)
        : name(std::move(name))
        , output(std::move(output))
    {
    }

    AdditionalOutput(AdditionalOutput const& other) = default;
    AdditionalOutput(AdditionalOutput&& other) noexcept = default;
    AdditionalOutput& operator=(AdditionalOutput const& other) = default;
    AdditionalOutput& operator=(AdditionalOutput&& other) noexcept = default;
    ~AdditionalOutput() = default;

    std::string name;
    Tensor output;
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

    /// @brief The generation logits. Size [beamSize, maxTokens, vocabSizePadded] (non-streaming)
    /// or [maxTokens, beamSize, vocabSizePadded] (streaming and allGeneratedTokens)
    /// or [1, beamSize, vocabSizePadded] (streaming and non-allGeneratedTokens)
    std::optional<Tensor> generationLogits;

    /// @brief Logits information for direct transfer when using fast logits
    std::optional<SpeculativeDecodingFastLogitsInfo> specDecFastLogitsInfo;

    /// @brief The encoder output. Size [encoderLen, hiddenSize]
    std::optional<Tensor> encoderOutput;

    /// @brief The reason why the model stopped generating tokens for each beam in this request. Size [beamSize].
    /// Currently only supported when beamSize is 1 and when using BatchingType::kINFLIGHT.
    std::vector<FinishReason> finishReasons;

    /// @brief The params of the context phase.
    std::optional<ContextPhaseParams> contextPhaseParams;

    /// @brief The number of the decoding iterations used to generate the result.
    /// In autoregressive decoding, it is equal to the maximum length of the beam in outputTokenIds.
    /// In speculative decoding, might be less than maximum length of the beam in outputTokenIds as more than
    /// one token can be generated per iteration. Used for speculative decoding statistics.
    SizeType32 decodingIter{0};

    /// @brief The average number of decoded tokens per iteration. For standard model it is 1.
    /// For speculative decoding model >= 1 -- number of draft tokens accepted per step + 1.
    float avgDecodedTokensPerIter{0.0f};

    /// @brief The index of the output sequence of this result where 0 <= sequenceIndex < numReturnSequences.
    /// In beam search (beamWidth > 1), this index will be always zero because all beams to be returned are included
    /// in this result.
    SizeType32 sequenceIndex{0};

    /// @brief Indicates if this is the final result for a given sequence in the request
    /// In beam search (beamWidth > 1), the value will always equal to the value of isFinal.
    bool isSequenceFinal;

    /// @brief Performance metrics if returnPerfMetrics is set in OutputConfig
    std::optional<RequestPerfMetrics> requestPerfMetrics;

    /// @brief The additional outputs
    std::vector<AdditionalOutput> additionalOutputs;
};

/// @brief Class that holds either an error or a result
class Response
{
public:
    Response(IdType requestId, std::string errorMsg, std::optional<IdType> clientId = std::nullopt);
    Response(IdType requestId, Result Result, std::optional<IdType> clientId = std::nullopt);

    ~Response();
    Response(Response const& other);
    Response(Response&& other) noexcept;
    Response& operator=(Response const& other);
    Response& operator=(Response&& other) noexcept;

    /// @brief Get the id of the request for which this response was generated
    [[nodiscard]] IdType getRequestId() const;

    /// @brief Get the client id of the request for which this response was generated
    [[nodiscard]] std::optional<IdType> getClientId() const;

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

/// @brief Configuration class for dynamic tuning of batch size and max num tokens. During runtime the statistics of
/// input and output lengths are recoreded. Based on these statistics, the batch size and max num tokens are tuned
/// dynamically to better serve the requests.
class DynamicBatchConfig
{
public:
    /// @brief The default window size for moving average of input and output length which is used to calculate dynamic
    /// batch size and max num tokens
    static SizeType32 const kDefaultDynamicBatchMovingAverageWindow = 128;

    explicit DynamicBatchConfig(bool enableBatchSizeTuning = false, bool enableMaxNumTokensTuning = false,
        SizeType32 dynamicBatchMovingAverageWindow = kDefaultDynamicBatchMovingAverageWindow,
        std::vector<std::pair<SizeType32, SizeType32>> batchSizeTable = kDefaultBatchSizeTable);

    [[nodiscard]] SizeType32 getDynamicBatchMovingAverageWindow() const;

    [[nodiscard]] bool getEnableBatchSizeTuning() const;

    [[nodiscard]] bool getEnableMaxNumTokensTuning() const;

    [[nodiscard]] std::vector<std::pair<SizeType32, SizeType32>> getBatchSizeTable() const;

    /// @brief The default value of batch size table
    static std::vector<std::pair<SizeType32, SizeType32>> const kDefaultBatchSizeTable;

private:
    friend class Serialization;

    /// @brief Controls if the batch size should be tuned dynamically
    bool mEnableBatchSizeTuning;

    /// @brief Controls if the max num tokens should be tuned dynamically
    bool mEnableMaxNumTokensTuning;

    /// @brief The window size for moving average of input and output length which is used to calculate dynamic batch
    /// size and max num tokens
    SizeType32 mDynamicBatchMovingAverageWindow;

    /// @brief A vector of (batchSizeLimit, batchSize). When max capacity batch size is less than
    // batchSizeLimit_{i} but greater or equal to batchSizeLimit_{i-1}, the batch size will be batchSize_{i}.
    // For max capacity batch size beyond the last batchSizeLimit, the batch size may be rounded down to multiple of 512
    // based on the actual implementation.
    std::vector<std::pair<SizeType32, SizeType32>> mBatchSizeTable;
};

/// @brief Configuration class for the scheduler
class SchedulerConfig
{
public:
    explicit SchedulerConfig(
        CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        std::optional<ContextChunkingPolicy> contextChunkingPolicy = std::nullopt,
        std::optional<DynamicBatchConfig> dynamicBatchConfig = std::nullopt);

    bool operator==(SchedulerConfig const& other) const;

    [[nodiscard]] CapacitySchedulerPolicy getCapacitySchedulerPolicy() const;

    [[nodiscard]] std::optional<ContextChunkingPolicy> getContextChunkingPolicy() const;

    [[nodiscard]] std::optional<DynamicBatchConfig> getDynamicBatchConfig() const;

private:
    friend class Serialization;

    /// @brief The capacity scheduler policy. See CapacitySchedulerPolicy.
    CapacitySchedulerPolicy mCapacitySchedulerPolicy;

    /// @brief The context chunking policy. See ContextChunkingPolicy.
    std::optional<ContextChunkingPolicy> mContextChunkingPolicy;

    /// @brief The config for tuning batch size dynamically. See DynamicBatchSizeConfig.
    std::optional<DynamicBatchConfig> mDynamicBatchConfig;
};

/// @brief Configuration class for the KV cache
class KvCacheConfig
{
public:
    static constexpr auto kDefaultGpuMemFraction = 0.9F;

    explicit KvCacheConfig(bool enableBlockReuse = true, std::optional<SizeType32> const& maxTokens = std::nullopt,
        std::optional<std::vector<SizeType32>> const& maxAttentionWindowVec = std::nullopt,
        std::optional<SizeType32> const& sinkTokenLength = std::nullopt,
        std::optional<FloatType> const& freeGpuMemoryFraction = std::nullopt,
        std::optional<size_t> const& hostCacheSize = std::nullopt, bool onboardBlocks = true,
        std::optional<FloatType> const& crossKvCacheFraction = std::nullopt,
        std::optional<RetentionPriority> secondaryOffloadMinPriority = std::nullopt, size_t eventBufferMaxSize = 0,
        bool enablePartialReuse = true, bool copyOnPartialReuse = true, bool useUvm = false,
        SizeType32 attentionDpEventsGatherPeriodMs = 5,
        std::optional<tensorrt_llm::runtime::RuntimeDefaults> const& runtimeDefaults = std::nullopt,
        uint64_t const& maxGpuTotalBytes = 0);

    [[nodiscard]] bool getEnableBlockReuse() const;
    [[nodiscard]] bool getEnablePartialReuse() const;
    [[nodiscard]] bool getCopyOnPartialReuse() const;
    [[nodiscard]] std::optional<SizeType32> getMaxTokens() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getMaxAttentionWindowVec() const;
    [[nodiscard]] std::optional<SizeType32> getSinkTokenLength() const;
    [[nodiscard]] std::optional<FloatType> getFreeGpuMemoryFraction() const;
    [[nodiscard]] std::optional<FloatType> getCrossKvCacheFraction() const;
    [[nodiscard]] std::optional<size_t> getHostCacheSize() const;
    [[nodiscard]] bool getOnboardBlocks() const;
    [[nodiscard]] std::optional<RetentionPriority> getSecondaryOffloadMinPriority() const;
    [[nodiscard]] size_t getEventBufferMaxSize() const;
    [[nodiscard]] bool getUseUvm() const;
    [[nodiscard]] SizeType32 getAttentionDpEventsGatherPeriodMs() const;
    [[nodiscard]] uint64_t getMaxGpuTotalBytes() const;

    void setEnableBlockReuse(bool enableBlockReuse);
    void setEnablePartialReuse(bool enablePartialReuse);
    void setCopyOnPartialReuse(bool copyOnPartialReuse);
    void setMaxTokens(std::optional<SizeType32> maxTokens);
    void setMaxAttentionWindowVec(std::vector<SizeType32> maxAttentionWindowVec);
    void setSinkTokenLength(SizeType32 sinkTokenLength);
    void setFreeGpuMemoryFraction(FloatType freeGpuMemoryFraction);
    void setCrossKvCacheFraction(FloatType crossKvCacheFraction);
    void setHostCacheSize(size_t hostCacheSize);
    void setOnboardBlocks(bool onboardBlocks);
    void setSecondaryOffloadMinPriority(std::optional<RetentionPriority> secondaryOffloadMinPriority);
    void setEventBufferMaxSize(size_t eventBufferMaxSize);
    void setUseUvm(bool useUvm);
    void setAttentionDpEventsGatherPeriodMs(SizeType32 attentionDpEventsGatherPeriodMs);
    void setMaxGpuTotalBytes(uint64_t maxGpuTotalBytes);

    void fillEmptyFieldsFromRuntimeDefaults(tensorrt_llm::runtime::RuntimeDefaults const& runtimeDefaults);

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

    /// @brief The fraction of the KV Cache memory should be reserved for cross attention
    /// If set to p, self attention will use 1-p of KV Cache memory and cross attention
    /// will use p of KV Cache memory. Default is 50%.
    /// Should only be set when using encoder-decoder model.
    std::optional<FloatType> mCrossKvCacheFraction;

    /// @brief Size of secondary memory pool in bytes. Default is 0.
    /// Having a secondary memory pool increases KV cache block reuse potential.
    std::optional<size_t> mHostCacheSize;

    /// @brief Controls whether offloaded blocks should be onboarded back into primary memory before being reused.
    bool mOnboardBlocks;

    /// @brief Only blocks with priority > mSecondaryOfflineMinPriority can be offloaded to secondary memory.
    std::optional<RetentionPriority> mSecondaryOffloadMinPriority;

    /// @brief Max size of the KV cache event buffer
    size_t mEventBufferMaxSize;

    /// @brief Whether blocks that are only partially matched can be reused
    bool mEnablePartialReuse;

    /// @brief Whether partially matched blocks that are in use can be reused after copying them
    bool mCopyOnPartialReuse;

    /// @brief Whether to use UVM for the KV cache.
    bool mUseUvm;

    /// @brief The period in milliseconds to gather attention DP events across ranks
    SizeType32 mAttentionDpEventsGatherPeriodMs;
    /// @brief The maximum size in bytes of GPU memory that can be allocated for the KV cache.
    /// If both mMaxGpuTotalBytes and mFreeGpuMemoryFraction are specified, memory corresponding to the minimum will
    /// be allocated.
    uint64_t mMaxGpuTotalBytes;
};

/// @brief Configuration class for the runtime perf knobs
class ExtendedRuntimePerfKnobConfig
{
public:
    explicit ExtendedRuntimePerfKnobConfig(bool multiBlockMode = true, bool enableContextFMHAFP32Acc = false,
        bool cudaGraphMode = false, SizeType32 cudaGraphCacheSize = 0);

    bool operator==(ExtendedRuntimePerfKnobConfig const& other) const
    {
        return mMultiBlockMode == other.mMultiBlockMode && mEnableContextFMHAFP32Acc == other.mEnableContextFMHAFP32Acc
            && mCudaGraphMode == other.mCudaGraphMode && mCudaGraphCacheSize == other.mCudaGraphCacheSize;
    }

    [[nodiscard]] bool getMultiBlockMode() const;
    [[nodiscard]] bool getEnableContextFMHAFP32Acc() const;
    [[nodiscard]] bool getCudaGraphMode() const;
    [[nodiscard]] SizeType32 getCudaGraphCacheSize() const;

    void setMultiBlockMode(bool multiBlockMode);
    void setEnableContextFMHAFP32Acc(bool enableContextFMHAFP32Acc);
    void setCudaGraphMode(bool cudaGraphMode);
    void setCudaGraphCacheSize(SizeType32 cacheSize);

private:
    friend class Serialization;

    /// @brief Control if multi block mode should be enabled or not.
    bool mMultiBlockMode;

    /// @brief If enable FMHA runner FP32 accumulation.
    bool mEnableContextFMHAFP32Acc;

    /// @brief Control if enable cuda graph.
    bool mCudaGraphMode;

    /// @brief Number of cuda graphs to be cached in the runtime.
    /// The larger the cache, the better the perf, but more GPU memory is consumed.
    SizeType32 mCudaGraphCacheSize;
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
    /// @param orchestratorConfig The orchestrator configuration. See OrchestratorConfig.
    /// @param numNodes The number of nodes to use for execution. Default is 1.
    explicit ParallelConfig(CommunicationType commType = CommunicationType::kMPI,
        CommunicationMode commMode = CommunicationMode::kLEADER,
        std::optional<std::vector<SizeType32>> deviceIds = std::nullopt,
        std::optional<std::vector<SizeType32>> participantIds = std::nullopt,
        std::optional<OrchestratorConfig> const& orchestratorConfig = std::nullopt,
        std::optional<SizeType32> numNodes = std::nullopt);

    [[nodiscard]] CommunicationType getCommunicationType() const;
    [[nodiscard]] CommunicationMode getCommunicationMode() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getDeviceIds() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getParticipantIds() const;
    [[nodiscard]] std::optional<OrchestratorConfig> getOrchestratorConfig() const;
    [[nodiscard]] std::optional<SizeType32> getNumNodes() const;

    void setCommunicationType(CommunicationType type);
    void setCommunicationMode(CommunicationMode mode);
    void setDeviceIds(std::vector<SizeType32> const& deviceIds);
    void setParticipantIds(std::vector<SizeType32> const& participantIds);
    void setOrchestratorConfig(OrchestratorConfig const& orchestratorConfig);
    void setNumNodes(SizeType32 numNodes);

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

    /// @brief The number of nodes to use for execution. Default is 1.
    std::optional<SizeType32> mNumNodes;
};

/// @brief config for PeftCacheManager
class PeftCacheConfig
{
public:
    static constexpr SizeType32 kDefaultOptimalAdapterSize = 8;
    static constexpr SizeType32 kDefaultMaxAdapterSize = 64;
    static constexpr SizeType32 kDefaultMaxPagesPerBlockHost = 24;
    static constexpr SizeType32 kDefaultMaxPagesPerBlockDevice = 8;

    explicit PeftCacheConfig(SizeType32 numHostModuleLayer = 0, SizeType32 numDeviceModuleLayer = 0,
        SizeType32 optimalAdapterSize = kDefaultOptimalAdapterSize, SizeType32 maxAdapterSize = kDefaultMaxAdapterSize,
        SizeType32 numPutWorkers = 1, SizeType32 numEnsureWorkers = 1, SizeType32 numCopyStreams = 1,
        SizeType32 maxPagesPerBlockHost = kDefaultMaxPagesPerBlockHost,
        SizeType32 maxPagesPerBlockDevice = kDefaultMaxPagesPerBlockDevice,
        std::optional<float> const& deviceCachePercent = std::nullopt,
        std::optional<size_t> const& hostCacheSize = std::nullopt,
        std::optional<std::string> const& loraPrefetchDir = std::nullopt);

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
    [[nodiscard]] std::optional<std::string> getLoraPrefetchDir() const;

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
    // folder to store the LoRA weights we hope to load during engine initialization
    std::optional<std::string> mLoraPrefetchDir;
};

/// @brief Configuration class for the decoding.
class DecodingConfig
{
public:
    explicit DecodingConfig(std::optional<DecodingMode> decodingMode = std::nullopt,
        std::optional<LookaheadDecodingConfig> lookaheadDecodingConfig = std::nullopt,
        std::optional<MedusaChoices> medusaChoices = std::nullopt,
        std::optional<EagleConfig> eagleConfig = std::nullopt);

    bool operator==(DecodingConfig const& other) const;

    // Decoding mode.
    /// @brief Sets decoding mode. Some modes require the use of their own setters.
    void setDecodingMode(DecodingMode const&);
    [[nodiscard]] std::optional<DecodingMode> getDecodingMode() const;

    // Lookahead methods.
    /// @brief Sets lookahead decoding mode and config.
    void setLookaheadDecodingConfig(LookaheadDecodingConfig const& lookaheadDecodingConfig);
    void enableSeamlessLookaheadDecoding();
    [[nodiscard]] std::optional<LookaheadDecodingConfig> getLookaheadDecodingConfig() const;
    [[nodiscard]] SizeType32 getLookaheadDecodingMaxNumRequest() const;

    // Medusa methods.
    /// @brief Sets medusa mode and config.
    void setMedusaChoices(MedusaChoices const&);
    [[nodiscard]] std::optional<MedusaChoices> getMedusaChoices() const;

    // EAGLE methods.
    /// @brief Sets eagle mode and config.
    void setEagleConfig(EagleConfig const&);
    [[nodiscard]] std::optional<EagleConfig> getEagleConfig() const;

private:
    friend class Serialization;

    // Decoding mode.
    std::optional<DecodingMode> mDecodingMode;
    // Lookahead params.
    std::optional<LookaheadDecodingConfig> mLookaheadDecodingConfig;
    // Medusa params.
    std::optional<MedusaChoices> mMedusaChoices;
    // Eagle config.
    std::optional<EagleConfig> mEagleConfig;
    // The max number of requests that can support running with lookahead decoding
    static constexpr SizeType32 mLookaheadDecodingMaxNumRequest = 8;
};

/// @brief Guided decoding configurations for executor.
class GuidedDecodingConfig
{
public:
    enum class GuidedDecodingBackend
    {
        /// @brief Enable guided decoding with XGrammar backend.
        kXGRAMMAR = 0,
        /// @brief Enable guided decoding with LLGuidance backend.
        kLLGUIDANCE = 1,
    };

    explicit GuidedDecodingConfig(GuidedDecodingBackend backend,
        std::optional<std::vector<std::string>> encodedVocab = std::nullopt,
        std::optional<std::string> tokenizerStr = std::nullopt,
        std::optional<std::vector<TokenIdType>> stopTokenIds = std::nullopt);

    bool operator==(GuidedDecodingConfig const& other) const;

    void setBackend(GuidedDecodingBackend const& backend);
    [[nodiscard]] GuidedDecodingBackend getBackend() const;

    void setEncodedVocab(std::vector<std::string> const& encodedVocab);
    [[nodiscard]] std::optional<std::vector<std::string>> getEncodedVocab() const;

    void setTokenizerStr(std::string const& tokenizerStr);
    [[nodiscard]] std::optional<std::string> getTokenizerStr() const;

    void setStopTokenIds(std::vector<TokenIdType> const& stopTokenIds);
    [[nodiscard]] std::optional<std::vector<TokenIdType>> getStopTokenIds() const;

    void validate() const;

private:
    friend class Serialization;

    /// @brief Guided decoding backend. Currently supports XGrammar.
    GuidedDecodingBackend mBackend;
    /// @brief Encoded vocabulary. For a huggingface tokenizer, it can be extracted by:
    /// ```python
    /// encoded_vocab = tokenizer.get_vocab()
    /// encoded_vocab = [token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])]
    /// ```
    std::optional<std::vector<std::string>> mEncodedVocab;
    /// @brief Tokenizer string. For a huggingface fast tokenizer, it can be extracted by:
    /// ```python
    /// tokenizer_str = tokenizer.backend_tokenizer.to_str()
    /// ```
    std::optional<std::string> mTokenizerStr;
    /// @brief Stop token ids. If not provided, it can be automatically detected.
    std::optional<std::vector<TokenIdType>> mStopTokenIds;
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

class CacheTransceiverConfig
{
public:
    enum class BackendType : std::uint8_t
    {
        DEFAULT = 0,
        MPI = 1,
        UCX = 2,
        NIXL = 3
    };
    explicit CacheTransceiverConfig(std::optional<BackendType> backendType = std::nullopt,
        std::optional<size_t> maxNumTokens = std::nullopt, std::optional<int> kvTransferTimeoutMs = std::nullopt);

    bool operator==(CacheTransceiverConfig const& other) const;
    void setBackendType(std::optional<BackendType> backendType);
    void setMaxTokensInBuffer(std::optional<size_t> maxTokensInBuffer);
    void setKvTransferTimeoutMs(std::optional<int> kvTransferTimeoutMs);

    [[nodiscard]] std::optional<int> getKvTransferTimeoutMs() const;
    [[nodiscard]] std::optional<size_t> getMaxTokensInBuffer() const;
    [[nodiscard]] std::optional<BackendType> getBackendType() const;

private:
    std::optional<BackendType> mBackendType;
    /// @brief The maximum number of tokens that the CacheTransceiver's pre-allocated buffer can hold. If the number of
    /// kvCache tokens to be transferred for a single request is greater than this value, the performance of the cache
    /// transfer may be degraded.
    std::optional<size_t> mMaxTokensInBuffer;
    std::optional<int> mKvTransferTimeoutMs;
};

/// @brief Configuration class for the model executor
class ExecutorConfig
{
public:
    static constexpr uint64_t kDefaultMaxSeqIdleMicroseconds
        = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::minutes(3)).count();

    static constexpr SizeType32 kDefaultIterStatsMaxIterations = 1000;

    // Per request stats may have additional overhead due to going through all requests. Turned off by default.
    static constexpr SizeType32 kDefaultRequestStatsMaxIterations = 0;

    explicit ExecutorConfig(SizeType32 maxBeamWidth = 1, SchedulerConfig schedulerConfig = SchedulerConfig(),
        KvCacheConfig kvCacheConfig = KvCacheConfig(), bool enableChunkedContext = true, bool normalizeLogProbs = true,
        SizeType32 iterStatsMaxIterations = kDefaultIterStatsMaxIterations,
        SizeType32 requestStatsMaxIterations = kDefaultRequestStatsMaxIterations,
        BatchingType batchingType = BatchingType::kINFLIGHT, std::optional<SizeType32> maxBatchSize = std::nullopt,
        std::optional<SizeType32> maxNumTokens = std::nullopt,
        std::optional<ParallelConfig> parallelConfig = std::nullopt,
        std::optional<PeftCacheConfig> const& peftCacheConfig = std::nullopt,
        std::optional<LogitsPostProcessorConfig> logitsPostProcessorConfig = std::nullopt,
        std::optional<DecodingConfig> decodingConfig = std::nullopt, bool useGpuDirectStorage = false,
        float gpuWeightsPercent = 1, std::optional<SizeType32> maxQueueSize = std::nullopt,
        ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig = ExtendedRuntimePerfKnobConfig(),
        std::optional<DebugConfig> debugConfig = std::nullopt, SizeType32 recvPollPeriodMs = 0,
        uint64_t maxSeqIdleMicroseconds = kDefaultMaxSeqIdleMicroseconds,
        std::optional<SpeculativeDecodingConfig> specDecConfig = std::nullopt,
        std::optional<GuidedDecodingConfig> guidedDecodingConfig = std::nullopt,
        std::optional<std::vector<AdditionalModelOutput>> additionalModelOutputs = std::nullopt,
        std::optional<CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt,
        bool gatherGenerationLogits = false, bool promptTableOffloading = false, bool enableTrtOverlap = false,
        bool failFastOnAttentionWindowTooLarge = false);

    [[nodiscard]] SizeType32 getMaxBeamWidth() const;
    [[nodiscard]] SchedulerConfig getSchedulerConfig() const;
    [[nodiscard]] KvCacheConfig getKvCacheConfig() const;
    // These functions return references and are useful for defining pybind properties.
    // If we used the normal return by value getters, we would get really confusing
    // behavior on the Python side.
    [[nodiscard]] SchedulerConfig& getSchedulerConfigRef();
    [[nodiscard]] KvCacheConfig& getKvCacheConfigRef();
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
    [[nodiscard]] bool getUseGpuDirectStorage() const;
    [[nodiscard]] float getGpuWeightsPercent() const;
    [[nodiscard]] std::optional<SizeType32> getMaxQueueSize() const;
    [[nodiscard]] ExtendedRuntimePerfKnobConfig getExtendedRuntimePerfKnobConfig() const;
    [[nodiscard]] std::optional<DebugConfig> getDebugConfig() const;
    [[nodiscard]] SizeType32 getRecvPollPeriodMs() const;
    [[nodiscard]] uint64_t getMaxSeqIdleMicroseconds() const;
    [[nodiscard]] std::optional<SpeculativeDecodingConfig> getSpecDecConfig() const;
    [[nodiscard]] std::optional<GuidedDecodingConfig> getGuidedDecodingConfig() const;
    [[nodiscard]] std::optional<std::vector<AdditionalModelOutput>> getAdditionalModelOutputs() const;
    [[nodiscard]] bool getGatherGenerationLogits() const;
    [[nodiscard]] bool getPromptTableOffloading() const;
    [[nodiscard]] std::optional<CacheTransceiverConfig> getCacheTransceiverConfig() const;
    [[nodiscard]] bool getEnableTrtOverlap() const;
    [[nodiscard]] bool getFailFastOnAttentionWindowTooLarge() const;

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
    void setUseGpuDirectStorage(bool const& useGpuDirectStorage);
    void setGpuWeightsPercent(float const& gpuWeightsPercent);
    void setMaxQueueSize(std::optional<SizeType32> const& maxQueueSize);
    void setExtendedRuntimePerfKnobConfig(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig);
    void setDebugConfig(DebugConfig const& debugConfig);
    void setRecvPollPeriodMs(SizeType32 const& recvPollPeriodMs);
    void setMaxSeqIdleMicroseconds(uint64_t maxSeqIdleMicroseconds);
    void setSpecDecConfig(SpeculativeDecodingConfig const& specDecConfig);
    void setGuidedDecodingConfig(GuidedDecodingConfig const& guidedDecodingConfig);
    void setAdditionalModelOutputs(std::vector<AdditionalModelOutput> const& additionalModelOutputs);
    void setGatherGenerationLogits(bool gatherGenerationLogits);
    void setPromptTableOffloading(bool promptTableOffloading);
    void setCacheTransceiverConfig(CacheTransceiverConfig const& cacheTransceiverConfig);
    void setEnableTrtOverlap(bool enableTrtOverlap);
    void setFailFastOnAttentionWindowTooLarge(bool failFastOnAttentionWindowTooLarge);

private:
    friend class Serialization;

    /// @brief The beam width value of requests that will be sent to the executor
    SizeType32 mMaxBeamWidth;

    /// @brief The scheduler configuration.
    SchedulerConfig mSchedulerConfig;

    /// @brief The KV cache configuration.
    KvCacheConfig mKvCacheConfig;

    /// @brief Controls whether context is allowed to be chunked.
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

    /// @brief Enable/disable use of GPU Direct Storage (GDS) to load engines.
    bool mUseGpuDirectStorage;

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

    /// @brief The maximum time in microseconds a scheduled request can remain idle before getting terminated.
    /// Default value is 3 minutes.
    uint64_t mMaxSeqIdleMicroseconds;

    /// @brief The speculative decoding configuration
    std::optional<SpeculativeDecodingConfig> mSpeculativeDecodingConfig;

    /// @brief The guided decoding configuration
    std::optional<GuidedDecodingConfig> mGuidedDecodingConfig;

    /// @brief The additional outputs to gather from the model.
    std::optional<std::vector<AdditionalModelOutput>> mAdditionalModelOutputs;

    /// @brief The cache transceiver configuration
    std::optional<CacheTransceiverConfig> mCacheTransceiverConfig;

    /// @brief Controls if generation logits should be gathered, so that returnGenerationLogits can be requested.
    bool mGatherGenerationLogits{false};

    /// @brief Controls if prompt table offloading is enabled.
    bool mPromptTableOffloading{false};

    /// @brief Controls whether preparation and TRT engine execution should be overlapped.
    bool mEnableTrtOverlap{false};

    /// @brief Controls whether to fail fast when attention window is too large to fit even a single sequence in the KV
    /// cache.
    bool mFailFastOnAttentionWindowTooLarge{false};
};

struct KVCacheCreatedData
{
    /// @brief The amount of blocks at each cache level
    std::vector<SizeType32> numBlocksPerCacheLevel;
};

/// @brief An entry for a single block stored into the tree
struct KVCacheStoredBlockData
{

    KVCacheStoredBlockData(IdType blockHash, tensorrt_llm::runtime::VecUniqueTokens tokens,
        std::optional<tensorrt_llm::runtime::LoraTaskIdType> loraId, SizeType32 cacheLevel, SizeType32 priority)
        : blockHash{blockHash}
        , tokens{std::move(tokens)}
        , loraId{loraId}
        , cacheLevel{cacheLevel}
        , priority{priority}
    {
    }

    /// @brief The hash of the block
    IdType blockHash;
    /// @brief The unique tokens of the block
    tensorrt_llm::runtime::VecUniqueTokens tokens;
    /// @brief The Lora task id of the block
    std::optional<tensorrt_llm::runtime::LoraTaskIdType> loraId;
    /// @brief The cache level of the block
    SizeType32 cacheLevel;
    /// @brief The priority of the block
    SizeType32 priority;
};

struct KVCacheStoredData
{
    /// @brief The parent of this sequence of stored blocks
    std::optional<IdType> parentHash;
    /// @brief A sequence of blocks. The parent of block `i` is block `i-1`
    std::vector<KVCacheStoredBlockData> blocks;
};

struct KVCacheRemovedData
{
    /// @brief The hashes of blocks being removed
    std::vector<IdType> blockHashes;
};

template <typename T>
struct KVCacheEventDiff
{
    T oldValue;
    T newValue;
};

struct KVCacheUpdatedData
{

    explicit KVCacheUpdatedData(IdType blockHash)
        : blockHash{blockHash} {};

    explicit KVCacheUpdatedData(IdType blockHash, std::optional<KVCacheEventDiff<SizeType32>> cacheLevel,
        std::optional<KVCacheEventDiff<SizeType32>> priority)
        : blockHash{blockHash}
        , cacheLevel{cacheLevel}
        , priority{priority} {};

    KVCacheUpdatedData& cacheLevelUpdated(SizeType32 oldValue, SizeType32 newValue)
    {
        cacheLevel = KVCacheEventDiff<SizeType32>{oldValue, newValue};
        return *this;
    }

    KVCacheUpdatedData& priorityUpdated(SizeType32 oldValue, SizeType32 newValue)
    {
        priority = KVCacheEventDiff<SizeType32>{oldValue, newValue};
        return *this;
    }

    /// @brief The hash of the updated block
    IdType blockHash;
    /// @brief The updated value of the cacheLevel field
    std::optional<KVCacheEventDiff<SizeType32>> cacheLevel = std::nullopt;
    /// @brief The updated value of the priority field
    std::optional<KVCacheEventDiff<SizeType32>> priority = std::nullopt;
};

using KVCacheEventData = std::variant<KVCacheCreatedData, KVCacheStoredData, KVCacheRemovedData, KVCacheUpdatedData>;

struct KVCacheEvent
{
    KVCacheEvent(IdType eventId, KVCacheEventData data, SizeType32 windowSize,
        std::optional<SizeType32> attentionDpRank = std::nullopt);

    /// @brief The unique id of this event
    IdType eventId;
    /// @brief The data corresponding to this event
    KVCacheEventData data;
    /// @brief The sliding window size
    SizeType32 windowSize;
    /// @brief The attention DP rank of the event, if applicable
    std::optional<SizeType32> attentionDpRank;
};

/// @brief Exposes a limited set of KV cache manager functionalities
class KVCacheEventManager
{
public:
    KVCacheEventManager(
        std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager> kvCacheManager);

    /// @brief Get the latest KV Cache events.
    /// @param timeout The maximum time to wait for new events. If nullopt, will only return when new events are
    /// available, or when the executor instance has shutdown.
    std::deque<KVCacheEvent> getLatestEvents(std::optional<std::chrono::milliseconds> timeout = std::nullopt);

private:
    std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager> kvCacheManager;
};

/// @brief The executor is responsible for receiving new requests and sending responses, and running the inference
class Executor
{

public:
    /// @brief
    /// @param modelPath Path to the folder that defines the model to run
    /// @param modelType The type of model
    /// @param executorConfig The configuration for the executor
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
    Executor(Executor const& executor) = delete;
    Executor& operator=(Executor const& executor) = delete;
    Executor(Executor&&) = default;
    Executor& operator=(Executor&&) = default;

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

    /// @brief  Indicates if the current process participates in this executor instance
    [[nodiscard]] bool isParticipant() const;

    std::optional<std::shared_ptr<KVCacheEventManager>> getKVCacheEventManager() const;

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
