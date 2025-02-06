/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"

#include <optional>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

enum class CacheType
{
    kSELF = 0,
    kCROSS = 1,
};

//! @brief Encapsulates parameters to configure paged KV cache.
class KvCacheConfig
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    explicit KvCacheConfig(std::optional<SizeType32> maxTokens = std::nullopt,
        std::optional<std::vector<SizeType32>> maxAttentionWindowVec = std::nullopt,
        std::optional<SizeType32> sinkTokenLength = std::nullopt,
        std::optional<float> freeGpuMemoryFraction = std::nullopt, bool enableBlockReuse = true, bool useUvm = false,
        std::optional<size_t> hostCacheSize = std::nullopt, bool onboardBlocks = true,
        std::optional<float> crossKvCacheFraction = std::nullopt,
        std::optional<SizeType32> secondaryOffloadMinPriority = std::nullopt, size_t eventBufferMaxSize = 0)
        : maxTokens{maxTokens}
        , maxAttentionWindowVec{std::move(maxAttentionWindowVec)}
        , sinkTokenLength{sinkTokenLength}
        , freeGpuMemoryFraction{freeGpuMemoryFraction}
        , enableBlockReuse(enableBlockReuse)
        , useUvm(useUvm)
        , hostCacheSize(hostCacheSize)
        , onboardBlocks(onboardBlocks)
        , crossKvCacheFraction{crossKvCacheFraction}
        , secondaryOffloadMinPriority(secondaryOffloadMinPriority)
        , eventBufferMaxSize(eventBufferMaxSize)
    {
    }

    explicit KvCacheConfig(executor::KvCacheConfig const& kvCacheConfig)
        : KvCacheConfig(kvCacheConfig.getMaxTokens(), kvCacheConfig.getMaxAttentionWindowVec(),
            kvCacheConfig.getSinkTokenLength(), kvCacheConfig.getFreeGpuMemoryFraction(),
            kvCacheConfig.getEnableBlockReuse(), false, kvCacheConfig.getHostCacheSize(),
            kvCacheConfig.getOnboardBlocks(), kvCacheConfig.getCrossKvCacheFraction(),
            kvCacheConfig.getSecondaryOffloadMinPriority(), kvCacheConfig.getEventBufferMaxSize())
    {
    }

    bool operator==(KvCacheConfig const& other) const
    {
        return maxTokens == other.maxTokens && maxAttentionWindowVec == other.maxAttentionWindowVec
            && sinkTokenLength == other.sinkTokenLength && freeGpuMemoryFraction == other.freeGpuMemoryFraction
            && enableBlockReuse == other.enableBlockReuse && useUvm == other.useUvm
            && hostCacheSize == other.hostCacheSize && onboardBlocks == other.onboardBlocks
            && crossKvCacheFraction == other.crossKvCacheFraction
            && secondaryOffloadMinPriority == other.secondaryOffloadMinPriority
            && eventBufferMaxSize == other.eventBufferMaxSize;
    }

    friend std::ostream& operator<<(std::ostream& os, KvCacheConfig const& self);

    std::optional<SizeType32> maxTokens;
    std::optional<std::vector<SizeType32>> maxAttentionWindowVec;
    std::optional<SizeType32> sinkTokenLength;
    std::optional<float> freeGpuMemoryFraction;
    bool enableBlockReuse;
    static constexpr auto kDefaultGpuMemFraction = 0.9F;
    bool useUvm;
    std::optional<size_t> hostCacheSize;
    bool onboardBlocks;
    // Cross will use crossKvCacheFraction of KV Cache and self attention will use the rest.
    std::optional<float> crossKvCacheFraction;
    // The minimum priority level to allow blocks to be offloaded to secondary memory.
    std::optional<SizeType32> secondaryOffloadMinPriority;
    // Maximum size of the KV Cache event buffer
    size_t eventBufferMaxSize;
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
