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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
using FloatType = float;
using TokenIdType = std::int32_t;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using IdType = std::uint64_t;
using IterationType = std::uint64_t;
using RandomSeedType = std::uint64_t;
using VecLogProbs = std::vector<FloatType>;
using StreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;
using LogitsPostProcessor = std::function<void(IdType, Tensor&, BeamTokens const&, StreamPtr&)>;
using LogitsPostProcessorMap = std::unordered_map<std::string, LogitsPostProcessor>;
using MedusaChoices = std::vector<std::vector<SizeType32>>;

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
    kGPU,
    kUVM,
    kUNKNOWN
};

enum class ModelType
{
    kDECODER_ONLY = 0,
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

/// @brief Struct that holds the stats of a KV cache manager
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
};

/// @brief Struct that holds the stats of a single iteration
struct IterationStats
{
    /// @brief Ending time of this iteration
    std::string timestamp;
    /// @brief Iteration id
    IterationType iter;
    /// @brief Number of active requests
    SizeType32 numActiveRequests;
    /// @brief Number of max active requests
    SizeType32 maxNumActiveRequests;
    /// @brief GPU memory usage in bytes
    size_t gpuMemUsage;
    /// @brief CPU memory usage in bytes
    size_t cpuMemUsage;
    /// @brief Pinned memory usage in bytes
    size_t pinnedMemUsage;
    /// @brief Stats specific to KV caches
    std::optional<KvCacheStats> kvCacheStats;
    /// @brief Stats specific to static batching
    std::optional<StaticBatchingStats> staticBatchingStats;
    /// @brief Stats specific to inflight batching
    std::optional<InflightBatchingStats> inflightBatchingStats;
};

/// @brief Enum class that represents the state of a request
enum class RequestStage
{
    /// @brief Request that have been received but not yet included in the active requests (due to constraints such as
    /// maximum batch size for example).
    kQUEUED,
    /// @brief Active request in context phase
    kCONTEXT_IN_PROGRESS,
    /// @brief Active request in generation phase
    kGENERATION_IN_PROGRESS,
    /// @brief Active request for which generation has completed
    kGENERATION_COMPLETE,

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
    /// @brief Whether the request is scheduled for the current iteration
    bool scheduled;
    /// @brief Whether the request is being paused at the current iteration due to lack of resources (KV cache blocks
    /// exhaustion for example)
    bool paused;
};

/// @brief Struct that holds the stats of all requests in an iteration
struct RequestStatsPerIteration
{
    /// @brief The iteration id for these stats
    IterationType iter;
    /// @brief The stats of all active requests for this iteration
    std::vector<RequestStats> requestStats;
};

/// @brief Decoding mode
enum class DecodingMode
{
    /// @brief No mode specified. Config will be determined from the beam width of the first request at runtime
    /// TopKTopP if beamWidth == 1, BeamSearch otherwise
    kNONE,
    kTOP_K,
    kTOP_P,
    kBEAM_SEARCH,
    kMEDUSA,
    kTOP_K_TOP_P,
};

} // namespace tensorrt_llm::executor
