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
#include <memory>
#include <vector>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm::executor
{

class Request;
class Tensor;

using TensorPtr = std::shared_ptr<Tensor>;
using SizeType = std::int32_t;
using FloatType = float;
using TokenIdType = std::int32_t;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using IdType = std::uint64_t;
using RandomSeedType = std::uint64_t;
using VecLogProbs = std::vector<FloatType>;

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

enum class BatchingType
{
    kSTATIC = 0,
    kINFLIGHT = 1,
    kINFLIGHT_UNFUSED = 2,
};

enum class SchedulerPolicy
{
    kMAX_UTILIZATION = 0,
    kGUARANTEED_NO_EVICT = 1,
};

enum class CommunicatorType
{
    kMPI = 0
};

enum class CommMode
{
    kLEADER,       // With the leader mode, only the leader will be returning from the executor constructor and
                   // therefore only the leader can enqueue requests and get responses
    kORCHESTRATOR, // With the orchestrator mode, only the orchestrator will be returning from the executor constructor
                   // and therefore only the leader can enqueue requests and get responses The orchestrator doesn't
                   // participate in the computations
    kALL,          // With the ALL mode, all participants are expected to make the same calls to the executor API
                   // So they all need to send the same requests
                   // Responses will be the same for all participants
};

} // namespace tensorrt_llm::executor
