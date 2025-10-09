/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/types.h"
#include <array>
#include <iostream>
#include <istream>
#include <list>
#include <optional>
#include <ostream>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tensorrt_llm::executor::serialize_utils
{

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
class VectorWrapBuf : public std::basic_streambuf<CharT, TraitsT>
{
public:
    explicit VectorWrapBuf(std::vector<CharT>& vec)
    {
        std::streambuf::setg(vec.data(), vec.data(), vec.data() + vec.size());
    }
};

template <typename T, typename = void>
struct ValueType
{
    using type = void;
};

template <typename T>
struct ValueType<T, std::void_t<typename T::value_type>>
{
    using type = typename T::value_type;
};

template <typename T>
struct ValueType<std::optional<T>, void>
{
    using type = T;
};

template <typename T>
struct is_variant : std::false_type
{
};

template <typename... Ts>
struct is_variant<std::variant<Ts...>> : std::true_type
{
};

template <typename T>
constexpr bool is_variant_v = is_variant<T>::value;

// Detect std::array
template <typename T>
struct is_std_array : std::false_type
{
};

template <typename U, std::size_t N>
struct is_std_array<std::array<U, N>> : std::true_type
{
    using value_type = U;
    static constexpr std::size_t size = N;
};

template <typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;

template <typename T>
using array_value_type_t = typename is_std_array<T>::value_type;

template <typename T>
constexpr std::size_t array_size_v = is_std_array<T>::size;

// Detect std::pair
template <typename T>
struct is_std_pair : std::false_type
{
};

template <typename A, typename B>
struct is_std_pair<std::pair<A, B>> : std::true_type
{
    using first_type = A;
    using second_type = B;
};

template <typename T>
constexpr bool is_std_pair_v = is_std_pair<T>::value;

// SerializedSize
template <typename T>
bool constexpr hasSerializedSize(...)
{
    return false;
}

template <typename T>
bool constexpr hasSerializedSize(decltype(Serialization::serializedSize(std::declval<T const&>())))
{
    return true;
}

static_assert(hasSerializedSize<RequestPerfMetrics>(size_t()));
static_assert(hasSerializedSize<Request>(size_t()));
static_assert(hasSerializedSize<SamplingConfig>(size_t()));
static_assert(hasSerializedSize<OutputConfig>(size_t()));
static_assert(hasSerializedSize<AdditionalModelOutput>(size_t()));
static_assert(hasSerializedSize<PromptTuningConfig>(size_t()));
static_assert(hasSerializedSize<MultimodalInput>(size_t()));
static_assert(hasSerializedSize<MropeConfig>(size_t()));
static_assert(hasSerializedSize<LoraConfig>(size_t()));
static_assert(hasSerializedSize<kv_cache::CommState>(size_t()));
static_assert(hasSerializedSize<kv_cache::SocketState>(size_t()));
static_assert(hasSerializedSize<kv_cache::CacheState>(size_t()));
static_assert(hasSerializedSize<DataTransceiverState>(size_t()));
static_assert(hasSerializedSize<ContextPhaseParams>(size_t()));
static_assert(hasSerializedSize<ExternalDraftTokensConfig>(size_t()));
static_assert(hasSerializedSize<Tensor>(size_t()));
static_assert(hasSerializedSize<SpeculativeDecodingFastLogitsInfo>(size_t()));
static_assert(hasSerializedSize<Result>(size_t()));
static_assert(hasSerializedSize<Response>(size_t()));
static_assert(hasSerializedSize<KvCacheConfig>(size_t()));
static_assert(hasSerializedSize<SchedulerConfig>(size_t()));
static_assert(hasSerializedSize<ParallelConfig>(size_t()));
static_assert(hasSerializedSize<PeftCacheConfig>(size_t()));
static_assert(hasSerializedSize<DecodingMode>(size_t()));
static_assert(hasSerializedSize<LookaheadDecodingConfig>(size_t()));
static_assert(hasSerializedSize<EagleConfig>(size_t()));
static_assert(hasSerializedSize<KvCacheRetentionConfig>(size_t()));
static_assert(hasSerializedSize<DecodingConfig>(size_t()));
static_assert(hasSerializedSize<DebugConfig>(size_t()));
static_assert(hasSerializedSize<SpeculativeDecodingConfig>(size_t()));
static_assert(hasSerializedSize<GuidedDecodingConfig>(size_t()));
static_assert(hasSerializedSize<GuidedDecodingParams>(size_t()));
static_assert(!hasSerializedSize<std::string>(size_t()));
static_assert(!hasSerializedSize<std::optional<float>>(size_t()));
static_assert(hasSerializedSize<CacheTransceiverConfig>(size_t()));
static_assert(hasSerializedSize<KVCacheEvent>(size_t()));
static_assert(hasSerializedSize<KVCacheCreatedData>(size_t()));
static_assert(hasSerializedSize<KVCacheStoredData>(size_t()));
static_assert(hasSerializedSize<KVCacheStoredBlockData>(size_t()));
static_assert(hasSerializedSize<KVCacheRemovedData>(size_t()));
static_assert(hasSerializedSize<KVCacheEventDiff<SizeType32>>(size_t()));
static_assert(hasSerializedSize<KVCacheUpdatedData>(size_t()));
static_assert(hasSerializedSize<tensorrt_llm::runtime::UniqueToken>(size_t()));

template <typename T>
size_t serializedSize(T const& data)
{
    // Fundamental types
    if constexpr (std::is_fundamental_v<T>)
    {
        return sizeof(T);
    }
    else if constexpr (hasSerializedSize<T>(size_t()))
    {
        return Serialization::serializedSize(data);
    }
    // Enum class
    else if constexpr (std::is_enum_v<T>)
    {
        using UnderlyingType = std::underlying_type_t<T>;
        auto value = static_cast<UnderlyingType>(data);
        return serializedSize(value);
    }
    // Vectors, lists and strings
    else if constexpr (std::is_same_v<T, std::vector<typename ValueType<T>::type>>
        || std::is_same_v<T, std::list<typename ValueType<T>::type>> || std::is_same_v<T, std::string>)
    {
        size_t size = sizeof(size_t);
        for (auto const& elem : data)
        {
            size += serializedSize(elem);
        }
        return size;
    }
    // std::array
    else if constexpr (is_std_array_v<T>)
    {
        size_t size = 0;
        for (auto const& elem : data)
        {
            size += serializedSize(elem);
        }
        return size;
    }
    // std::pair
    else if constexpr (is_std_pair_v<T>)
    {
        return serializedSize(data.first) + serializedSize(data.second);
    }
    // Optional
    else if constexpr (std::is_same_v<T, std::optional<typename ValueType<T>::type>>)
    {
        return sizeof(bool) + (data.has_value() ? serializedSize(data.value()) : 0);
    }
    else if constexpr (is_variant_v<T>)
    {
        size_t index = data.index();
        size_t size = sizeof(index);
        std::visit([&size](auto const& value) { size += serializedSize(value); }, data);
        return size;
    }
    else
    {
        static_assert(std::is_same_v<T, void>, "Unsupported type for serialization");
    }
}

// Serialize
template <typename T>
bool constexpr hasSerialize(...)
{
    return false;
}

template <typename T>
bool constexpr hasSerialize(
    decltype(Serialization::serialize(std::declval<T const&>(), std::declval<std::ostream&>()))*)
{
    return true;
}

static_assert(hasSerialize<RequestPerfMetrics>(nullptr));
static_assert(hasSerialize<Request>(nullptr));
static_assert(hasSerialize<SamplingConfig>(nullptr));
static_assert(hasSerialize<OutputConfig>(nullptr));
static_assert(hasSerialize<AdditionalModelOutput>(nullptr));
static_assert(hasSerialize<PromptTuningConfig>(nullptr));
static_assert(hasSerialize<MultimodalInput>(nullptr));
static_assert(hasSerialize<MropeConfig>(nullptr));
static_assert(hasSerialize<LoraConfig>(nullptr));
static_assert(hasSerialize<ExternalDraftTokensConfig>(nullptr));
static_assert(hasSerialize<Tensor>(nullptr));
static_assert(hasSerialize<SpeculativeDecodingFastLogitsInfo>(nullptr));
static_assert(hasSerialize<Result>(nullptr));
static_assert(hasSerialize<Response>(nullptr));
static_assert(hasSerialize<KvCacheConfig>(nullptr));
static_assert(hasSerialize<SchedulerConfig>(nullptr));
static_assert(hasSerialize<ParallelConfig>(nullptr));
static_assert(hasSerialize<PeftCacheConfig>(nullptr));
static_assert(hasSerialize<DecodingMode>(nullptr));
static_assert(hasSerialize<LookaheadDecodingConfig>(nullptr));
static_assert(hasSerialize<EagleConfig>(nullptr));
static_assert(hasSerialize<SpeculativeDecodingConfig>(nullptr));
static_assert(hasSerialize<GuidedDecodingConfig>(nullptr));
static_assert(hasSerialize<GuidedDecodingParams>(nullptr));
static_assert(hasSerialize<KvCacheRetentionConfig>(nullptr));
static_assert(hasSerialize<DecodingConfig>(nullptr));
static_assert(hasSerialize<kv_cache::CommState>(nullptr));
static_assert(hasSerialize<kv_cache::SocketState>(nullptr));
static_assert(hasSerialize<kv_cache::CacheState>(nullptr));
static_assert(hasSerialize<DataTransceiverState>(nullptr));
static_assert(hasSerialize<ContextPhaseParams>(nullptr));
static_assert(!hasSerialize<std::string>(nullptr));
static_assert(!hasSerialize<std::optional<float>>(nullptr));
static_assert(hasSerialize<CacheTransceiverConfig>(nullptr));
static_assert(hasSerialize<KVCacheEvent>(nullptr));
static_assert(hasSerialize<KVCacheCreatedData>(nullptr));
static_assert(hasSerialize<KVCacheStoredData>(nullptr));
static_assert(hasSerialize<KVCacheStoredBlockData>(nullptr));
static_assert(hasSerialize<KVCacheRemovedData>(nullptr));
static_assert(hasSerialize<KVCacheEventDiff<SizeType32>>(nullptr));
static_assert(hasSerialize<KVCacheUpdatedData>(nullptr));
static_assert(hasSerialize<tensorrt_llm::runtime::UniqueToken>(nullptr));

template <typename T>
void serialize(T const& data, std::ostream& os)
{
    // Fundamental types
    if constexpr (std::is_fundamental_v<T>)
    {
        os.write(reinterpret_cast<char const*>(&data), sizeof(data));
    }
    else if constexpr (hasSerialize<T>(nullptr))
    {
        return Serialization::serialize(data, os);
    }
    // Enum class
    else if constexpr (std::is_enum_v<T>)
    {
        using UnderlyingType = std::underlying_type_t<T>;
        auto value = static_cast<UnderlyingType>(data);
        os.write(reinterpret_cast<char const*>(&value), sizeof(value));
    }
    // Vectors, lists and strings
    else if constexpr (std::is_same_v<T, std::vector<typename ValueType<T>::type>>
        || std::is_same_v<T, std::list<typename ValueType<T>::type>> || std::is_same_v<T, std::string>)
    {
        size_t size = data.size();
        os.write(reinterpret_cast<char const*>(&size), sizeof(size));
        for (auto const& element : data)
        {
            serialize(element, os);
        }
    }
    // std::array
    else if constexpr (is_std_array_v<T>)
    {
        for (auto const& element : data)
        {
            serialize(element, os);
        }
    }
    // std::pair
    else if constexpr (is_std_pair_v<T>)
    {
        serialize(data.first, os);
        serialize(data.second, os);
    }
    // Optional
    else if constexpr (std::is_same_v<T, std::optional<typename ValueType<T>::type>>)
    {
        // Serialize a boolean indicating whether optional has a value
        bool hasValue = data.has_value();
        os.write(reinterpret_cast<char const*>(&hasValue), sizeof(hasValue));

        // Serialize the value if it exists
        if (hasValue)
        {
            serialize(data.value(), os);
        }
    }
    // std::variant
    else if constexpr (is_variant_v<T>)
    {
        // Store the index of the active variant
        size_t index = data.index();
        os.write(reinterpret_cast<char const*>(&index), sizeof(index));

        // Serialize the held value based on the index
        std::visit([&os](auto const& value) { serialize(value, os); }, data);
    }
    else
    {
        static_assert(std::is_same_v<T, void>, "Unsupported type for serialization");
    }
}

template <size_t I, typename T>
using variant_alternative_t = typename std::variant_alternative<I, T>::type;

template <typename T>
struct get_variant_alternative_type
{
    static variant_alternative_t<T::index(), T> get(T const& variant)
    {
        return std::get<T::index()>(variant);
    }
};

template <typename T>
T deserialize(std::istream& is);

// Helper function to deserialize variant by index using template recursion
template <typename T, std::size_t... Is>
T deserializeVariantByIndex(std::istream& is, std::size_t index, std::index_sequence<Is...> /*indices*/)
{
    T result;
    bool found = ((Is == index ? (result = deserialize<std::variant_alternative_t<Is, T>>(is), true) : false) || ...);
    if (!found)
    {
        TLLM_THROW("Invalid variant index during deserialization: " + std::to_string(index));
    }
    return result;
}

// Deserialize
template <typename T>
T deserialize(std::istream& is)
{
    // Fundamental types
    if constexpr (std::is_fundamental_v<T>)
    {
        T data;
        is.read(reinterpret_cast<char*>(&data), sizeof(data));
        return data;
    }
    // Enum class
    else if constexpr (std::is_enum_v<T>)
    {
        using UnderlyingType = std::underlying_type_t<T>;
        UnderlyingType value;
        is.read(reinterpret_cast<char*>(&value), sizeof(value));
        return static_cast<T>(value);
    }
    // deserialize from serialization class
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::RequestPerfMetrics::TimePoint>)
    {
        return Serialization::deserializeTimePoint(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::RequestPerfMetrics>)
    {
        return Serialization::deserializeRequestPerfMetrics(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::SamplingConfig>)
    {
        return Serialization::deserializeSamplingConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::OutputConfig>)
    {
        return Serialization::deserializeOutputConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::AdditionalModelOutput>)
    {
        return Serialization::deserializeAdditionalModelOutput(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::ExternalDraftTokensConfig>)
    {
        return Serialization::deserializeExternalDraftTokensConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::PromptTuningConfig>)
    {
        return Serialization::deserializePromptTuningConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::MultimodalInput>)
    {
        return Serialization::deserializeMultimodalInput(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::MropeConfig>)
    {
        return Serialization::deserializeMropeConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::LoraConfig>)
    {
        return Serialization::deserializeLoraConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::kv_cache::CommState>)
    {
        return Serialization::deserializeCommState(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::kv_cache::SocketState>)
    {
        return Serialization::deserializeSocketState(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::kv_cache::AgentState>)
    {
        return Serialization::deserializeAgentState(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::kv_cache::CacheState>)
    {
        return Serialization::deserializeCacheState(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::DataTransceiverState>)
    {
        return Serialization::deserializeDataTransceiverState(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::ContextPhaseParams>)
    {
        return Serialization::deserializeContextPhaseParams(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::Request>)
    {
        return Serialization::deserializeRequest(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::Tensor>)
    {
        return Serialization::deserializeTensor(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::SpeculativeDecodingFastLogitsInfo>)
    {
        return Serialization::deserializeSpecDecFastLogitsInfo(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::Result>)
    {
        return Serialization::deserializeResult(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::Response>)
    {
        return Serialization::deserializeResponse(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KvCacheConfig>)
    {
        return Serialization::deserializeKvCacheConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::DynamicBatchConfig>)
    {
        return Serialization::deserializeDynamicBatchConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::SchedulerConfig>)
    {
        return Serialization::deserializeSchedulerConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::ExtendedRuntimePerfKnobConfig>)
    {
        return Serialization::deserializeExtendedRuntimePerfKnobConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::ParallelConfig>)
    {
        return Serialization::deserializeParallelConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::PeftCacheConfig>)
    {
        return Serialization::deserializePeftCacheConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::OrchestratorConfig>)
    {
        return Serialization::deserializeOrchestratorConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::DecodingMode>)
    {
        return Serialization::deserializeDecodingMode(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::LookaheadDecodingConfig>)
    {
        return Serialization::deserializeLookaheadDecodingConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::EagleConfig>)
    {
        return Serialization::deserializeEagleConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::SpeculativeDecodingConfig>)
    {
        return Serialization::deserializeSpeculativeDecodingConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::GuidedDecodingConfig>)
    {
        return Serialization::deserializeGuidedDecodingConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::GuidedDecodingParams>)
    {
        return Serialization::deserializeGuidedDecodingParams(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KvCacheRetentionConfig>)
    {
        return Serialization::deserializeKvCacheRetentionConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KvCacheRetentionConfig::TokenRangeRetentionConfig>)
    {
        return Serialization::deserializeTokenRangeRetentionConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::DecodingConfig>)
    {
        return Serialization::deserializeDecodingConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::DebugConfig>)
    {
        return Serialization::deserializeDebugConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KvCacheStats>)
    {
        return Serialization::deserializeKvCacheStats(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::StaticBatchingStats>)
    {
        return Serialization::deserializeStaticBatchingStats(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::InflightBatchingStats>)
    {
        return Serialization::deserializeInflightBatchingStats(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::SpecDecodingStats>)
    {
        return Serialization::deserializeSpecDecodingStats(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::IterationStats>)
    {
        return Serialization::deserializeIterationStats(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::ExecutorConfig>)
    {
        return Serialization::deserializeExecutorConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::DisServingRequestStats>)
    {
        return Serialization::deserializeDisServingRequestStats(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::RequestStage>)
    {
        return Serialization::deserializeRequestStage(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::RequestStats>)
    {
        return Serialization::deserializeRequestStats(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::RequestStatsPerIteration>)
    {
        return Serialization::deserializeRequestStatsPerIteration(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::AdditionalOutput>)
    {
        return Serialization::deserializeAdditionalOutput(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::CacheTransceiverConfig>)
    {
        return Serialization::deserializeCacheTransceiverConfig(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KVCacheEvent>)
    {
        return Serialization::deserializeKVCacheEvent(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KVCacheCreatedData>)
    {
        return Serialization::deserializeKVCacheCreatedData(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KVCacheStoredData>)
    {
        return Serialization::deserializeKVCacheStoredData(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KVCacheStoredBlockData>)
    {
        return Serialization::deserializeKVCacheStoredBlockData(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KVCacheRemovedData>)
    {
        return Serialization::deserializeKVCacheRemovedData(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KVCacheEventDiff<SizeType32>>)
    {
        return Serialization::deserializeKVCacheEventDiff<SizeType32>(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::executor::KVCacheUpdatedData>)
    {
        return Serialization::deserializeKVCacheUpdatedData(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::runtime::UniqueToken>)
    {
        return Serialization::deserializeUniqueToken(is);
    }
    else if constexpr (std::is_same_v<T, tensorrt_llm::batch_manager::kv_cache_manager::BlockKey>)
    {
        return Serialization::deserializeBlockKey(is);
    }
    // Optional
    else if constexpr (std::is_same_v<T, std::optional<typename ValueType<T>::type>>)
    {
        bool hasValue = false;
        is.read(reinterpret_cast<char*>(&hasValue), sizeof(hasValue));

        if (hasValue)
        {
            auto value = deserialize<typename ValueType<T>::type>(is);
            return std::optional<typename ValueType<T>::type>(std::move(value));
        }

        return std::nullopt;
    }
    // Vectors, lists and strings
    else if constexpr (std::is_same_v<T, std::vector<typename ValueType<T>::type>>
        || std::is_same_v<T, std::list<typename ValueType<T>::type>> || std::is_same_v<T, std::string>)
    {
        size_t size = 0;
        is.read(reinterpret_cast<char*>(&size), sizeof(size));

        T container;
        for (size_t i = 0; i < size; ++i)
        {
            auto element = deserialize<typename ValueType<T>::type>(is);
            container.push_back(std::move(element));
        }
        return container;
    }
    // std::array
    else if constexpr (is_std_array_v<T>)
    {
        T container{};
        for (std::size_t i = 0; i < array_size_v<T>; ++i)
        {
            container[i] = deserialize<array_value_type_t<T>>(is);
        }
        return container;
    }
    // std::pair
    else if constexpr (is_std_pair_v<T>)
    {
        auto first = deserialize<typename is_std_pair<T>::first_type>(is);
        auto second = deserialize<typename is_std_pair<T>::second_type>(is);
        return T{std::move(first), std::move(second)};
    }
    // std::variant
    else if constexpr (is_variant_v<T>)
    {
        // Get the index of the active type
        std::size_t index = 0;
        is.read(reinterpret_cast<char*>(&index), sizeof(index));

        return deserializeVariantByIndex<T>(is, index, std::make_index_sequence<std::variant_size_v<T>>{});
    }
    else
    {
        static_assert(std::is_same_v<T, void>, "Unsupported type for deserialization");
        return T();
    }
}

// https://stackoverflow.com/a/75741832
template <typename T>
struct method_return_type;

template <typename ReturnT, typename ClassT, typename... Args>
struct method_return_type<ReturnT (ClassT::*)(Args...) const>
{
    using type = ReturnT;
};

template <typename MethodT>
using method_return_type_t = typename method_return_type<MethodT>::type;

template <typename ObjectMethodT>
auto deserializeWithGetterType(std::istream& is)
{
    return deserialize<method_return_type_t<ObjectMethodT>>(is);
}

} // namespace tensorrt_llm::executor::serialize_utils
