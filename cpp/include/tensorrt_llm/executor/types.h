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

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm::runtime
{
class CudaStream;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::executor
{

class Request;
class Tensor;

using TensorPtr = std::shared_ptr<Tensor>;
using SizeType32 = std::int32_t;
using SizeType64 = std::int64_t;
using FloatType = float;
using TokenIdType = std::int32_t;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using IdType = std::uint64_t;
using VecTokenExtraIds = std::vector<IdType>;
using IterationType = std::uint64_t;
using RandomSeedType = std::uint64_t;
using VecLogProbs = std::vector<FloatType>;
using StreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;
using MillisecondsType = std::chrono::milliseconds;
using CacheSaltIDType = std::uint64_t;
using LogitsPostProcessor
    = std::function<void(IdType, Tensor&, BeamTokens const&, StreamPtr const&, std::optional<IdType>)>;
using LogitsPostProcessorMap = std::unordered_map<std::string, LogitsPostProcessor>;
using LogitsPostProcessorBatched = std::function<void(std::vector<IdType> const&, std::vector<Tensor>&,
    std::vector<std::reference_wrapper<BeamTokens const>> const&, StreamPtr const&,
    std::vector<std::optional<IdType>> const&)>;
using MedusaChoices = std::vector<std::vector<SizeType32>>;
using EagleChoices = std::vector<std::vector<SizeType32>>;
using PriorityType = float;
using BufferView = std::basic_string_view<uint8_t>;

enum class DataType
{
    kBOOL,
    kUINT8,
    kINT8,
    kINT32,
    kINT64,
    kBF16,
    kFP8,
    kFP16,
    kFP32,
    kUNKNOWN
};

enum class RequestType
{
    REQUEST_TYPE_CONTEXT_AND_GENERATION = 0,
    REQUEST_TYPE_CONTEXT_ONLY = 1,
    REQUEST_TYPE_GENERATION_ONLY = 2
};

//! \brief For converting a C++ data type to a `TrtLmmDataType`.
template <typename T, bool = false>
struct TypeTraits
{
};

template <>
struct TypeTraits<float>
{
    static constexpr auto value = DataType::kFP32;
};

template <>
struct TypeTraits<half>
{
    static constexpr auto value = DataType::kFP16;
};

template <>
struct TypeTraits<std::int8_t>
{
    static constexpr auto value = DataType::kINT8;
};

template <>
struct TypeTraits<std::int32_t>
{
    static constexpr auto value = DataType::kINT32;
};

template <>
struct TypeTraits<std::int64_t>
{
    static constexpr auto value = DataType::kINT64;
};

template <>
struct TypeTraits<bool>
{
    static constexpr auto value = DataType::kBOOL;
};

template <>
struct TypeTraits<std::uint8_t>
{
    static constexpr auto value = DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <>
struct TypeTraits<__nv_bfloat16>
{
    static constexpr auto value = DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TypeTraits<__nv_fp8_e4m3>
{
    static constexpr auto value = DataType::kFP8;
};
#endif

template <typename T>
struct TypeTraits<T*>
{
    // Pointers are stored as int64_t.
    static constexpr auto value = DataType::kINT64;
};

enum class MemoryType
{
    kCPU,
    kCPU_PINNED,
    kCPU_PINNEDPOOL,
    kGPU,
    kUVM,
    kUNKNOWN
};

enum class ModelType
{
    kDECODER_ONLY = 0,
    kENCODER_ONLY = 1,
    kENCODER_DECODER = 2,
};

/// @brief The batching type
enum class BatchingType
{
    /// @brief STATIC refers to the traditional batching scheme with a batch of requests running in lockstep until the
    /// full generation for all of them is complete. Requests in a batch are all padded up to the maximum input and
    /// output sequence length of any member of the batch.
    kSTATIC = 0,

    /// @brief INFLIGHT refers to a scheme where newly arrived requests are dynamically incorporated into the batch
    /// under execution, and requests are returned as soon as the end condition is met without any padding.
    kINFLIGHT = 1,
};

/// @brief The policy used to select the subset of available requests in each iteration of the executor generation loop
enum class CapacitySchedulerPolicy
{
    /// @brief MAX_UTILIZATION packs as many requests as the underlying TRT engine can support in any iteration of the
    /// InflightBatching generation loop. While this is expected to maximize GPU throughput, it might require that some
    /// requests be paused and restarted depending on peak KV cache memory availability.
    kMAX_UTILIZATION = 0,

    /// @brief GUARANTEED_NO_EVICT uses KV cache more conservatively guaranteeing that a request, once started, will run
    /// to completion without eviction.
    kGUARANTEED_NO_EVICT = 1,

    /// @brief kSTATIC_BATCH does not schedule new requests until all requests in current batch are completed.
    /// Similar to kGUARANTEED_NO_EVICT, requests will run to completion without eviction.
    kSTATIC_BATCH = 2
};

std::ostream& operator<<(std::ostream& os, CapacitySchedulerPolicy policy);

enum class ContextChunkingPolicy
{
    /// @brief Sequential chunking, complete the unfinished context phase first.
    kFIRST_COME_FIRST_SERVED = 0,

    /// @brief Iterate through each context request in sequence and attempt to increase its chunk
    /// count until the constraint is exceeded.
    kEQUAL_PROGRESS = 1,
};

std::ostream& operator<<(std::ostream& os, ContextChunkingPolicy policy);

enum class CommunicationType
{
    kMPI = 0
};

enum class CommunicationMode
{
    kLEADER, // With the leader mode, only the leader can enqueue requests. The requests will be
             // broadcasted to the workers. All participants can get response via awaitResponses. The leader is the
             // first participant in the provided participant IDS, or 0 if participant ID is not provided
    kORCHESTRATOR, // With the orchestrator mode, only the orchestrator can enqueue requests and await responses. The
                   // requests will be broadcasted to the workers. The orchestrator will spawn new processes for the
                   // execution of the model
};

/// @brief Struct that holds the stats of a KV cache manager.
// See KvCacheStats definition in kvCacheManager.h for more information about each field.
struct KvCacheStats
{
    /// @brief Max number of blocks
    SizeType32 maxNumBlocks;
    /// @brief Number of free blocks
    SizeType32 freeNumBlocks;
    /// @brief Number of used blocks
    SizeType32 usedNumBlocks;
    /// @brief Number of tokens per block
    SizeType32 tokensPerBlock;
    /// @brief Number of total allocated block
    SizeType32 allocTotalBlocks;
    /// @brief Number of newly allocated block
    SizeType32 allocNewBlocks;
    /// @brief Number of reused block
    SizeType32 reusedBlocks;
    /// @brief Number of not reused block
    SizeType32 missedBlocks;
    /// @brief Measuring the KV Cache reuse rate. cacheHitRate = reusedBlocks / (reusedBlocks + missedBlocks).
    float cacheHitRate;
};

/// @brief Struct that holds the stats of static batching models for a single iteration
struct StaticBatchingStats
{
    /// @brief Number of scheduled requests
    SizeType32 numScheduledRequests;
    /// @brief Number of requests in context stage
    SizeType32 numContextRequests;
    /// @brief Total number of context tokens in the iteration
    SizeType32 numCtxTokens;
    /// @brief Total number of tokens to generate in the iteration
    SizeType32 numGenTokens;
    /// @brief Total number of unused generation token slots
    SizeType32 emptyGenSlots;
};

/// @brief Struct that holds the stats of inflight batching models for a single iteration
struct InflightBatchingStats
{
    /// @brief Number of scheduled requests
    SizeType32 numScheduledRequests;
    /// @brief Number of requests in context stage
    SizeType32 numContextRequests;
    /// @brief Number of requests in generation stage
    SizeType32 numGenRequests;
    /// @brief Number of paused requests
    SizeType32 numPausedRequests;
    /// @brief Total number of context tokens in the iteration
    SizeType32 numCtxTokens;
    /// @brief Index of mirco batch
    SizeType32 microBatchId;
    /// @brief Average number of tokens decoded per request per iteration
    float avgNumDecodedTokensPerIter;
};

/// @brief Struct that holds speculative decoding stats
struct SpecDecodingStats
{
    /// @brief Total number of proposed draft tokens for all requests
    SizeType64 numDraftTokens;
    /// @brief Total number of accepted draft tokens for all requests
    SizeType64 numAcceptedTokens;
    /// @brief Number of requests with at least one draft token in batch
    SizeType64 numRequestsWithDraftTokens;
    /// @brief Acceptance length, defined as average number of tokens produced per step for all requests with at least
    /// one draft token
    double acceptanceLength;
    /// @brief Iteration latency for draft token generation only (ms)
    double iterLatencyMS;
    /// @brief Draft overhead, defined as iterLatencyMS (specdec) / iterLatencyMS (total)
    double draftOverhead;
};

/// @brief Struct that holds the stats of a single iteration
struct IterationStats
{
    /// @brief Ending time of this iteration
    std::string timestamp;
    /// @brief Iteration id
    IterationType iter;
    /// @brief Iteration latency (ms)
    double iterLatencyMS;
    /// @brief The total time spent in queue by the requests that became active in this iteration (ms)
    double newActiveRequestsQueueLatencyMS;
    /// @brief Number of new fetched active requests
    SizeType32 numNewActiveRequests;
    /// @brief Number of active requests
    SizeType32 numActiveRequests;
    /// @brief Number of queued requests
    SizeType32 numQueuedRequests;
    /// @brief  Number of requests that were completed in this iteration
    SizeType32 numCompletedRequests;
    /// @brief Number of max active requests
    SizeType32 maxNumActiveRequests;
    /// @brief Static max batch size passed to the executor
    SizeType32 maxBatchSizeStatic;
    /// @brief Batch size produced by dynamic tuner based on input stats
    SizeType32 maxBatchSizeTunerRecommended;
    /// @brife The min of maxBatchSizeStatic and maxBatchSizeRuntimeUpperbound
    SizeType32 maxBatchSizeRuntime;
    /// @brife Static max num tokens passed to the executor
    SizeType32 maxNumTokensStatic;
    /// @brife Max num tokens produced by dynamic tuner based on input stats
    SizeType32 maxNumTokensTunerRecommended;
    /// @brife The runtime max num tokens
    SizeType32 maxNumTokensRuntime;
    /// @brief GPU memory usage in bytes
    size_t gpuMemUsage;
    /// @brief CPU memory usage in bytes
    size_t cpuMemUsage;
    /// @brief Pinned memory usage in bytes
    size_t pinnedMemUsage;
    /// @brief Stats specific to KV caches
    std::optional<KvCacheStats> kvCacheStats;
    /// @brief Stats specific to cross KV caches
    std::optional<KvCacheStats> crossKvCacheStats;
    /// @brief Stats specific to static batching
    std::optional<StaticBatchingStats> staticBatchingStats;
    /// @brief Stats specific to inflight batching
    std::optional<InflightBatchingStats> inflightBatchingStats;
    /// @brief Stats specific to speculative decoding
    std::optional<SpecDecodingStats> specDecodingStats;
};

/// @brief Enum class that represents the state of a request
enum class RequestStage
{
    /// @brief Request that have been received but not yet included in the active requests (due to constraints such as
    /// maximum batch size for example).
    kQUEUED,
    /// @brief Active request in encoder phase
    kENCODER_IN_PROGRESS,
    /// @brief Active request in context phase
    kCONTEXT_IN_PROGRESS,
    /// @brief Active request in generation phase
    kGENERATION_IN_PROGRESS,
    /// @brief Active request for which generation has completed
    kGENERATION_COMPLETE,
};

/// @brief Struct that holds the request stats in the case of disaggregated serving
struct DisServingRequestStats
{
    /// @brief The total time spent on transferring KV cache from context phase to generation phase (ms)
    double kvCacheTransferMS;
    /// @brief The total size of KV cache transferred from context phase to generation phase (bytes)
    size_t kvCacheSize;
};

/// @brief Struct that holds the stats of a single request
struct RequestStats
{
    /// @brief The request id
    IdType id;
    /// @brief The current stage the request is in
    RequestStage stage;
    /// @brief If using chunked context, the current context prefill position
    SizeType32 contextPrefillPosition;
    /// @brief The number of generated tokens so far
    SizeType32 numGeneratedTokens;
    /// @brief The average number of decoded tokens per iteration. It is >= 1 for speculative decoding.
    float avgNumDecodedTokensPerIter;
    /// @brief Whether the request is scheduled for the current iteration
    bool scheduled;
    /// @brief Whether the request is being paused at the current iteration due to lack of resources (KV cache blocks
    /// exhaustion for example)
    bool paused;
    /// @brief Stats specific to disaggregated serving
    std::optional<DisServingRequestStats> disServingStats;
    /// @brief Number of total allocated blocks per request
    SizeType32 allocTotalBlocksPerRequest;
    /// @brief Number of newly allocated blocks per request
    SizeType32 allocNewBlocksPerRequest;
    /// @brief Number of reused blocks per request
    SizeType32 reusedBlocksPerRequest;
    /// @brief Number of missed blocks per request
    SizeType32 missedBlocksPerRequest;
    /// @brief KV Cache Hit Rate per request, defined as reusedBlocks / (reusedBlocks + missedBlocks)
    FloatType kvCacheHitRatePerRequest;
};

/// @brief Struct that holds the stats of all requests in an iteration
struct RequestStatsPerIteration
{
    /// @brief The iteration id for these stats
    IterationType iter;
    /// @brief The stats of all active requests for this iteration
    std::vector<RequestStats> requestStats;
};

/// @brief Struct that holds the stats of a request
struct RequestPerfMetrics
{
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

    struct TimingMetrics
    {
        /// @brief The time when the request arrived
        TimePoint arrivalTime;
        /// @brief The time when the request was first scheduled
        TimePoint firstScheduledTime;
        /// @brief The time when the first token was generated
        TimePoint firstTokenTime;
        /// @brief The time when the request was finished
        TimePoint lastTokenTime;
        /// @brief Start time of the KV cache transfer for disaggregated serving
        TimePoint kvCacheTransferStart;
        /// @brief End time of the KV cache transfer for disaggregated serving
        TimePoint kvCacheTransferEnd;
        /// @brief KV Cache size transfer for disaggregated serving
        mutable size_t kvCacheSize = 0;
    };

    struct KvCacheMetrics
    {
        /// @brief Number of total allocated blocks
        SizeType32 numTotalAllocatedBlocks{0};
        /// @brief Number of newly allocated blocks
        SizeType32 numNewAllocatedBlocks{0};
        /// @brief Number of reused blocks
        SizeType32 numReusedBlocks{0};
        /// @brief Number of missed blocks
        SizeType32 numMissedBlocks{0};
        /// @brief KV Cache Hit Rate, defined as reusedBlocks / (reusedBlocks + missedBlocks)
        FloatType kvCacheHitRate{0.f};
    };

    struct SpeculativeDecodingMetrics
    {
        /// @brief Token acceptance rate for speculative decoding requests
        FloatType acceptanceRate{0.f};
        /// @brief Total number of accepted draft tokens
        SizeType32 totalAcceptedDraftTokens{0};
        /// @brief Total number of draft tokens used in the request
        SizeType32 totalDraftTokens{0};
    };

    TimingMetrics timingMetrics;
    KvCacheMetrics kvCacheMetrics;
    SpeculativeDecodingMetrics speculativeDecoding;

    /// @brief First iteration where the request was processed
    std::optional<IterationType> firstIter;
    /// @brief Last iteration where a token was generated
    std::optional<IterationType> lastIter;
    /// @brief Current iteration
    std::optional<IterationType> iter;
};

/// @brief Struct that holds the debug tensors in an iteration
struct DebugTensorsPerIteration
{
    /// @brief The iteration id for these tensors
    IterationType iter;
    /// @brief The debug tensors for this iteration
    std::map<std::string, Tensor> debugTensors;
};

/// @brief The reason why the model stopped generating tokens for a request.
enum class FinishReason
{
    /// @brief The request is not finished.
    kNOT_FINISHED = 0,

    /// @brief The request finished because the end id was generated.
    kEND_ID = 1,

    /// @brief The request finished because a stop word was generated.
    kSTOP_WORDS = 2,

    /// @brief The request finished because the maximum number of tokens was reached.
    kLENGTH = 3,

    /// @brief The request finished because it got timed out (via the mAllotedTime parameter)
    kTIMED_OUT = 4,

    /// @brief The request was cancelled by calling cancelRequest.
    kCANCELLED = 5
};

//! \brief Enum describing the transfer mode for KV cache.
enum class KvCacheTransferMode
{
    DRAM = 0,                 //!< Copy to/from CPU memory (original approach).
    GDS = 1,                  //!< Attempt GPUDirect Storage (cuFile).
    POSIX_DEBUG_FALLBACK = 2, //!< Force a POSIX read/write for debugging.
};

/// @brief mode of the decoder
class DecodingMode
{
public:
    /// @brief No mode specified. Config will be determined from the beam width of the first request at runtime
    /// TopKTopP if beamWidth == 1, BeamSearch otherwise
    static auto constexpr Auto()
    {
        return DecodingMode{kAuto};
    }

    static auto constexpr TopK()
    {
        return DecodingMode{kTopK | kUsePenalties | kUseBanTokens | kUseStandardStopCriteria | kUseMinP};
    }

    static auto constexpr TopP()
    {
        return DecodingMode{kTopP | kUsePenalties | kUseBanTokens | kUseStandardStopCriteria | kUseMinP};
    }

    static auto constexpr TopKTopP()
    {
        return DecodingMode{kTopKTopP | kUsePenalties | kUseBanTokens | kUseStandardStopCriteria | kUseMinP};
    }

    static auto constexpr BeamSearch()
    {
        return DecodingMode{kBeamSearch | kUsePenalties | kUseBanTokens | kUseStandardStopCriteria};
    }

    static auto constexpr Medusa()
    {
        return DecodingMode{kMedusa | kUseMinLength | kUseStandardStopCriteria | kUseExplicitEosStop};
    }

    static auto constexpr Lookahead()
    {
        return DecodingMode{kLookahead | kUseMinLength | kUseStandardStopCriteria | kUseExplicitEosStop};
    }

    static auto constexpr ExplicitDraftTokens()
    {
        return DecodingMode{kExplicitDraftTokens | kUseStandardStopCriteria | kUseExplicitEosStop};
    }

    static auto constexpr ExternalDraftTokens()
    {
        return DecodingMode{kExternalDraftTokens | kUsePenalties | kUseBanTokens | kUseStandardStopCriteria};
    }

    static auto constexpr Eagle()
    {
        return DecodingMode{kEagle | kUseStandardStopCriteria | kUseExplicitEosStop};
    }

    auto constexpr useTemperature(bool useTemp)
    {
        mState = setBitTo(kUseTemperature, useTemp);
        return *this;
    }

    auto constexpr useOccurrencePenalties(bool usePenalty)
    {
        mState = setBitTo(kUseOccurrencePenalties, usePenalty);
        return *this;
    }

    auto constexpr usePresencePenalty(bool usePenalty)
    {
        mState = setBitTo(kUsePresencePenalties, usePenalty);
        return *this;
    }

    auto constexpr useRepetitionPenalty(bool usePenalty)
    {
        mState = setBitTo(kUseRepetitionPenalties, usePenalty);
        return *this;
    }

    auto constexpr useFrequencyPenalty(bool usePenalty)
    {
        mState = setBitTo(kUseFrequencyPenalties, usePenalty);
        return *this;
    }

    auto constexpr useMinLength(bool useMinLen)
    {
        mState = setBitTo(kUseMinLength, useMinLen);
        return *this;
    }

    auto constexpr useBanTokens(bool banTokens)
    {
        mState = setBitTo(kUseBanTokens, banTokens);
        return *this;
    }

    auto constexpr useBanWords(bool banWords)
    {
        mState = setBitTo(kUseBanWords, banWords);
        return *this;
    }

    auto constexpr useNoRepeatNgramSize(bool noRepeatNgramSize)
    {
        mState = setBitTo(kUseNoRepeatNgramSize, noRepeatNgramSize);
        return *this;
    }

    auto constexpr useStopWords(bool stopWords)
    {
        mState = setBitTo(kUseStopWords, stopWords);
        return *this;
    }

    auto constexpr useMaxLengthStop(bool maxLengthStop)
    {
        mState = setBitTo(kUseMaxLengthStop, maxLengthStop);
        return *this;
    }

    auto constexpr useExplicitEosStop(bool explicitEosStop)
    {
        mState = setBitTo(kUseExplicitEosStop, explicitEosStop);
        return *this;
    }

    auto constexpr useMinP(bool useMinP)
    {
        mState = setBitTo(kUseMinP, useMinP);
        return *this;
    }

    auto constexpr useVariableBeamWidthSearch(bool useVariableBeamWidthSearch)
    {
        mState = setBitTo(kUseVariableBeamWidthSearch, useVariableBeamWidthSearch);
        return *this;
    }

    [[nodiscard]] bool constexpr isAuto() const
    {
        return anyBitSet(kAuto);
    }

    [[nodiscard]] bool constexpr isTopK() const
    {
        return anyBitSet(kTopK);
    }

    [[nodiscard]] bool constexpr isTopP() const
    {
        return anyBitSet(kTopP);
    }

    [[nodiscard]] bool constexpr isTopKorTopP() const
    {
        return anyBitSet(kTopKTopP);
    }

    [[nodiscard]] bool constexpr isTopKandTopP() const
    {
        return allBitSet(kTopKTopP);
    }

    [[nodiscard]] bool constexpr isBeamSearch() const
    {
        return anyBitSet(kBeamSearch);
    }

    [[nodiscard]] bool constexpr isMedusa() const
    {
        return anyBitSet(kMedusa);
    }

    [[nodiscard]] bool constexpr isLookahead() const
    {
        return anyBitSet(kLookahead);
    }

    [[nodiscard]] bool constexpr isExplicitDraftTokens() const
    {
        return anyBitSet(kExplicitDraftTokens);
    }

    [[nodiscard]] bool constexpr isExternalDraftTokens() const
    {
        return anyBitSet(kExternalDraftTokens);
    }

    [[nodiscard]] bool constexpr isEagle() const
    {
        return anyBitSet(kEagle);
    }

    [[nodiscard]] bool constexpr isUseTemperature() const
    {
        return anyBitSet(kUseTemperature);
    }

    [[nodiscard]] bool constexpr isUsePresencePenalty() const
    {
        return anyBitSet(kUsePresencePenalties);
    }

    [[nodiscard]] bool constexpr isUseFrequencyPenalty() const
    {
        return anyBitSet(kUseFrequencyPenalties);
    }

    [[nodiscard]] bool constexpr isUseRepetitionPenalty() const
    {
        return anyBitSet(kUseRepetitionPenalties);
    }

    [[nodiscard]] bool constexpr isUseMinLength() const
    {
        return anyBitSet(kUseMinLength);
    }

    [[nodiscard]] bool constexpr isUseOccurrencePenalty() const
    {
        return anyBitSet(kUseOccurrencePenalties);
    }

    [[nodiscard]] bool constexpr isUsePenalty() const
    {
        return anyBitSet(kUsePenalties);
    }

    [[nodiscard]] bool constexpr isUseBanWords() const
    {
        return anyBitSet(kUseBanWords);
    }

    bool constexpr isUseNoRepeatNgramSize() const
    {
        return anyBitSet(kUseNoRepeatNgramSize);
    }

    bool constexpr isUseBanTokens() const
    {
        return anyBitSet(kUseBanTokens);
    }

    bool constexpr isUseStopWords() const
    {
        return anyBitSet(kUseStopWords);
    }

    bool constexpr isUseMaxLengthStop() const
    {
        return anyBitSet(kUseMaxLengthStop);
    }

    bool constexpr isUseExplicitEosStop() const
    {
        return anyBitSet(kUseExplicitEosStop);
    }

    bool constexpr isUseStopCriteria() const
    {
        return anyBitSet(kUseStandardStopCriteria | kUseExplicitEosStop);
    }

    bool constexpr isUseMinP() const
    {
        return anyBitSet(kUseMinP);
    }

    bool constexpr isUseVariableBeamWidthSearch() const
    {
        return anyBitSet(kUseVariableBeamWidthSearch);
    }

    using UnderlyingType = uint32_t;

    bool operator==(DecodingMode const& other) const
    {
        return mState == other.mState;
    }

    explicit constexpr DecodingMode(UnderlyingType state)
        : mState(state)
    {
    }

    [[nodiscard]] constexpr UnderlyingType getState() const
    {
        return mState;
    }

    [[nodiscard]] constexpr char const* getName() const
    {
        if (isTopKorTopP())
        {
            return "TopKorTopP";
        }
        if (isBeamSearch())
        {
            return "BeamSearch";
        }
        if (isMedusa())
        {
            return "Medusa";
        }
        if (isLookahead())
        {
            return "Lookahead";
        }
        if (isExplicitDraftTokens())
        {
            return "ExplicitDraftTokens";
        }
        if (isExternalDraftTokens())
        {
            return "ExternalDraftTokens";
        }
        if (isEagle())
        {
            return "Eagle";
        }
        return "Unknown";
    }

private:
    static SizeType32 constexpr kNumFlags{12};
    // Config will be determined from the beam width of the first request at runtime if no mode specified.
    // TopKTopP if beamWidth == 1, BeamSearch otherwise
    static UnderlyingType constexpr kUseRepetitionPenalties{1u << 0};
    static UnderlyingType constexpr kUseFrequencyPenalties{1u << 1};
    static UnderlyingType constexpr kUsePresencePenalties{1u << 2};
    static UnderlyingType constexpr kUseTemperature{1u << 3};
    static UnderlyingType constexpr kUseMinLength{1u << 4};
    static UnderlyingType constexpr kUseBanWords{1u << 5};
    static UnderlyingType constexpr kUseStopWords{1u << 6};
    static UnderlyingType constexpr kUseMaxLengthStop{1u << 7};
    static UnderlyingType constexpr kUseExplicitEosStop{1u << 8};
    static UnderlyingType constexpr kUseNoRepeatNgramSize{1u << 9};
    static UnderlyingType constexpr kUseMinP{1u << 10};
    static UnderlyingType constexpr kUseVariableBeamWidthSearch{1u << 11};

    static UnderlyingType constexpr kUseStandardStopCriteria{kUseStopWords | kUseMaxLengthStop};
    static UnderlyingType constexpr kUseOccurrencePenalties{
        kUseRepetitionPenalties | kUseFrequencyPenalties | kUsePresencePenalties};
    static UnderlyingType constexpr kUsePenalties{kUseOccurrencePenalties | kUseTemperature | kUseMinLength};
    static UnderlyingType constexpr kUseBanTokens{kUseNoRepeatNgramSize | kUseBanWords};

    static UnderlyingType constexpr kAuto{1u << (kNumFlags + 0)};
    static UnderlyingType constexpr kTopK{1u << (kNumFlags + 1)};
    static UnderlyingType constexpr kTopP{1u << (kNumFlags + 2)};
    static UnderlyingType constexpr kBeamSearch{1u << (kNumFlags + 3)};
    static UnderlyingType constexpr kMedusa{1u << (kNumFlags + 4)};
    static UnderlyingType constexpr kLookahead{1u << (kNumFlags + 5)};
    static UnderlyingType constexpr kExplicitDraftTokens{1u << (kNumFlags + 6)};
    static UnderlyingType constexpr kExternalDraftTokens{1u << (kNumFlags + 7)};
    static UnderlyingType constexpr kEagle{1u << (kNumFlags + 8)};

    static UnderlyingType constexpr kTopKTopP{kTopK | kTopP};

    [[nodiscard]] bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    [[nodiscard]] bool constexpr allBitSet(UnderlyingType bits) const
    {
        return (mState & bits) == bits;
    }

    UnderlyingType constexpr setBitTo(UnderlyingType state, bool x)
    {
        return (mState & (~state)) | (state * static_cast<UnderlyingType>(x));
    }

    UnderlyingType mState{};
};

static_assert(!DecodingMode::Auto().isUseBanWords());
static_assert(!DecodingMode::Auto().isUseOccurrencePenalty());
static_assert(!DecodingMode::Auto().isUseStopCriteria());
static_assert(DecodingMode::Auto().isAuto());
static_assert(!DecodingMode::Auto().isTopKorTopP());
static_assert(!DecodingMode::Auto().isBeamSearch());
static_assert(!DecodingMode::Auto().isMedusa());
static_assert(!DecodingMode::Auto().isLookahead());
static_assert(!DecodingMode::Auto().isExplicitDraftTokens());
static_assert(!DecodingMode::Auto().isExternalDraftTokens());
static_assert(!DecodingMode::Auto().isEagle());

static_assert(DecodingMode::TopK().isUseBanWords());
static_assert(DecodingMode::TopK().isUseOccurrencePenalty());
static_assert(DecodingMode::TopK().isUseStopCriteria());
static_assert(!DecodingMode::TopK().useRepetitionPenalty(false).isUseRepetitionPenalty());
static_assert(DecodingMode::TopK().useRepetitionPenalty(false).isUseOccurrencePenalty());
static_assert(!DecodingMode::TopK()
                   .useRepetitionPenalty(false)
                   .usePresencePenalty(false)
                   .useFrequencyPenalty(false)
                   .isUseOccurrencePenalty());
static_assert(!DecodingMode::TopK().isAuto());
static_assert(DecodingMode::TopK().isTopK());
static_assert(!DecodingMode::TopK().isTopP());
static_assert(DecodingMode::TopK().isTopKorTopP());
static_assert(!DecodingMode::TopK().isTopKandTopP());
static_assert(!DecodingMode::TopK().isBeamSearch());
static_assert(!DecodingMode::TopK().isMedusa());
static_assert(!DecodingMode::TopK().isLookahead());
static_assert(!DecodingMode::TopK().isExplicitDraftTokens());
static_assert(!DecodingMode::TopK().isExternalDraftTokens());
static_assert(!DecodingMode::TopK().isEagle());

static_assert(DecodingMode::TopP().isUseBanWords());
static_assert(DecodingMode::TopP().isUseOccurrencePenalty());
static_assert(DecodingMode::TopP().isUseStopCriteria());
static_assert(!DecodingMode::TopP().isAuto());
static_assert(!DecodingMode::TopP().isTopK());
static_assert(DecodingMode::TopP().isTopP());
static_assert(DecodingMode::TopP().isTopKorTopP());
static_assert(!DecodingMode::TopP().isTopKandTopP());
static_assert(!DecodingMode::TopP().isBeamSearch());
static_assert(!DecodingMode::TopP().isMedusa());
static_assert(!DecodingMode::TopP().isLookahead());
static_assert(!DecodingMode::TopP().isExplicitDraftTokens());
static_assert(!DecodingMode::TopP().isExternalDraftTokens());
static_assert(!DecodingMode::TopP().isEagle());

static_assert(DecodingMode::TopKTopP().isUseBanWords());
static_assert(DecodingMode::TopKTopP().isUseOccurrencePenalty());
static_assert(DecodingMode::TopKTopP().isUseStopCriteria());
static_assert(!DecodingMode::TopKTopP().isAuto());
static_assert(DecodingMode::TopKTopP().isTopK());
static_assert(DecodingMode::TopKTopP().isTopP());
static_assert(DecodingMode::TopKTopP().isTopKorTopP());
static_assert(DecodingMode::TopKTopP().isTopKandTopP());
static_assert(!DecodingMode::TopKTopP().isBeamSearch());
static_assert(!DecodingMode::TopKTopP().isMedusa());
static_assert(!DecodingMode::TopKTopP().isLookahead());
static_assert(!DecodingMode::TopKTopP().isExplicitDraftTokens());
static_assert(!DecodingMode::TopKTopP().isExternalDraftTokens());
static_assert(!DecodingMode::TopKTopP().isEagle());

static_assert(DecodingMode::BeamSearch().isUseStopCriteria());
static_assert(!DecodingMode::BeamSearch().isAuto());
static_assert(!DecodingMode::BeamSearch().isTopKorTopP());
static_assert(DecodingMode::BeamSearch().isBeamSearch());
static_assert(!DecodingMode::BeamSearch().isMedusa());
static_assert(!DecodingMode::BeamSearch().isLookahead());
static_assert(!DecodingMode::BeamSearch().isExplicitDraftTokens());
static_assert(!DecodingMode::BeamSearch().isExternalDraftTokens());
static_assert(!DecodingMode::BeamSearch().isEagle());

static_assert(!DecodingMode::Medusa().isUseBanWords());
static_assert(!DecodingMode::Medusa().isUseOccurrencePenalty());
static_assert(!DecodingMode::Medusa().isExplicitDraftTokens());
static_assert(DecodingMode::Medusa().isUseStopCriteria());
static_assert(DecodingMode::Medusa().isUsePenalty());
static_assert(DecodingMode::Medusa().isUseMinLength());
static_assert(!DecodingMode::Medusa().isAuto());
static_assert(!DecodingMode::Medusa().isTopKorTopP());
static_assert(!DecodingMode::Medusa().isBeamSearch());
static_assert(DecodingMode::Medusa().isMedusa());
static_assert(!DecodingMode::Medusa().isLookahead());
static_assert(!DecodingMode::Medusa().isExplicitDraftTokens());
static_assert(!DecodingMode::Medusa().isExternalDraftTokens());
static_assert(!DecodingMode::Medusa().isEagle());

static_assert(DecodingMode::Lookahead().isUseStopCriteria());
static_assert(DecodingMode::Lookahead().isUseStopWords());
static_assert(DecodingMode::Lookahead().isUseExplicitEosStop());
static_assert(!DecodingMode::Lookahead().isAuto());
static_assert(!DecodingMode::Lookahead().isTopKorTopP());
static_assert(!DecodingMode::Lookahead().isBeamSearch());
static_assert(!DecodingMode::Lookahead().isMedusa());
static_assert(DecodingMode::Lookahead().isLookahead());
static_assert(!DecodingMode::Lookahead().isExplicitDraftTokens());
static_assert(!DecodingMode::Lookahead().isExternalDraftTokens());
static_assert(!DecodingMode::Lookahead().isEagle());

static_assert(!DecodingMode::ExplicitDraftTokens().isUsePenalty());
static_assert(DecodingMode::ExplicitDraftTokens().isUseStopCriteria());
static_assert(!DecodingMode::ExplicitDraftTokens().isUseBanWords());
static_assert(!DecodingMode::ExplicitDraftTokens().isAuto());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopKorTopP());
static_assert(!DecodingMode::ExplicitDraftTokens().isBeamSearch());
static_assert(!DecodingMode::ExplicitDraftTokens().isMedusa());
static_assert(!DecodingMode::ExplicitDraftTokens().isLookahead());
static_assert(DecodingMode::ExplicitDraftTokens().isExplicitDraftTokens());
static_assert(!DecodingMode::ExplicitDraftTokens().isExternalDraftTokens());
static_assert(!DecodingMode::ExplicitDraftTokens().isEagle());

static_assert(DecodingMode::ExternalDraftTokens().isUseBanWords());
static_assert(DecodingMode::ExternalDraftTokens().isUseOccurrencePenalty());
static_assert(DecodingMode::ExternalDraftTokens().isUseStopCriteria());
static_assert(!DecodingMode::ExternalDraftTokens().isAuto());
static_assert(!DecodingMode::ExternalDraftTokens().isTopKorTopP());
static_assert(!DecodingMode::ExternalDraftTokens().isBeamSearch());
static_assert(!DecodingMode::ExternalDraftTokens().isMedusa());
static_assert(!DecodingMode::ExternalDraftTokens().isLookahead());
static_assert(!DecodingMode::ExternalDraftTokens().isExplicitDraftTokens());
static_assert(DecodingMode::ExternalDraftTokens().isExternalDraftTokens());
static_assert(!DecodingMode::ExternalDraftTokens().isEagle());

static_assert(!DecodingMode::Eagle().isUseBanWords());
static_assert(!DecodingMode::Eagle().isUseOccurrencePenalty());
static_assert(DecodingMode::Eagle().isUseStopCriteria());
static_assert(!DecodingMode::Eagle().isAuto());
static_assert(!DecodingMode::Eagle().isTopKorTopP());
static_assert(!DecodingMode::Eagle().isBeamSearch());
static_assert(!DecodingMode::Eagle().isMedusa());
static_assert(!DecodingMode::Eagle().isLookahead());
static_assert(!DecodingMode::Eagle().isExplicitDraftTokens());
static_assert(!DecodingMode::Eagle().isExternalDraftTokens());
static_assert(DecodingMode::Eagle().isEagle());
} // namespace tensorrt_llm::executor
