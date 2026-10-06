/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager
{

namespace cache_formatter_utils
{

using SizeType32 = runtime::SizeType32;
using CacheState = executor::kv_cache::CacheState;

/**
 * @brief Check if this rank should send cache data, given a pre-computed TargetRanksInfo.
 *
 * Callers that need RNN-specific target ranks should pass the result of targetIRanksForRnn().
 */
inline bool needSendCache(CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx,
    executor::kv_cache::TargetRanksInfo const& targetInfo)
{
    if (targetInfo.mDupHeadFactor <= 1)
    {
        return true;
    }

    int selfCpSize = selfConfig.getParallelConfig().mContextParallelism;
    int selfTpRank = (selfIdx % (selfConfig.getParallelConfig().mTensorParallelism * selfCpSize)) / selfCpSize;
    int selfTpRankInDpGroup = selfTpRank;

    if (selfConfig.getParallelConfig().mEnableAttentionDP)
    {
        int selfTPNumInDPGroup
            = selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize;
        selfTpRankInDpGroup = selfTpRank % selfTPNumInDPGroup;
    }

    int destDPRank = destConfig.getParallelConfig().mEnableAttentionDP ? destConfig.getParallelConfig().mDPrank : 0;

    return (destDPRank % targetInfo.mDupHeadFactor) == (selfTpRankInDpGroup % targetInfo.mDupHeadFactor);
}

/// @brief Convenience overload that computes KV-cache target ranks automatically.
inline bool needSendCache(CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx)
{
    return needSendCache(
        selfConfig, destConfig, selfIdx, executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx));
}

/**
 * @brief Pick send connections, given a pre-computed TargetRanksInfo.
 */
inline std::vector<size_t> pickSendConnections(size_t numConnections, CacheState const& selfConfig, SizeType32 selfIdx,
    CacheState const& destConfig, std::vector<SizeType32> const& counterPartRanks,
    executor::kv_cache::TargetRanksInfo const& targetInfo)
{
    TLLM_CHECK(numConnections == counterPartRanks.size());

    // NO duplicate head filtering - format/sendBuffer handles that
    std::vector<size_t> indices;
    for (auto rank : targetInfo.mIRanks)
    {
        auto it = std::find(counterPartRanks.begin(), counterPartRanks.end(), rank);
        TLLM_CHECK_WITH_INFO(it != counterPartRanks.end(), "Required rank %d not found in counterPartRanks", rank);
        indices.push_back(std::distance(counterPartRanks.begin(), it));
    }
    return indices;
}

/// @brief Convenience overload that computes KV-cache target ranks automatically.
inline std::vector<size_t> pickSendConnections(size_t numConnections, CacheState const& selfConfig, SizeType32 selfIdx,
    CacheState const& destConfig, std::vector<SizeType32> const& counterPartRanks)
{
    return pickSendConnections(numConnections, selfConfig, selfIdx, destConfig, counterPartRanks,
        executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx));
}

/**
 * @brief Pick receive connections and their corresponding local rank indices, given a pre-computed TargetRanksInfo.
 * @return Pair of (pickUpConnections, localRankIndices) where pickUpConnections are indices into counterPartRanks
 *         and localRankIndices are the corresponding indices into targetInfo.mIRanks.
 */
inline std::pair<std::vector<size_t>, std::vector<size_t>> pickRecvConnections(size_t numConnections,
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig,
    std::vector<SizeType32> const& counterPartRanks, executor::kv_cache::TargetRanksInfo const& targetInfo)
{
    if (targetInfo.mIRanks.empty())
    {
        return {{}, {}};
    }

    auto baseIndices
        = pickSendConnections(numConnections, selfConfig, selfIdx, destConfig, counterPartRanks, targetInfo);

    if (targetInfo.mPeerDupHeadFactor <= 1)
    {
        std::vector<size_t> localRankIndices(baseIndices.size());
        std::iota(localRankIndices.begin(), localRankIndices.end(), 0);
        return {baseIndices, localRankIndices};
    }

    int selfDPRank = selfConfig.getParallelConfig().mEnableAttentionDP ? selfConfig.getParallelConfig().mDPrank : 0;

    std::vector<size_t> pickUpConnections;
    std::vector<size_t> localRankIndices;
    for (int i = 0; i < targetInfo.mDomainTPSize; i++)
    {
        if ((i % targetInfo.mPeerDupHeadFactor) == (selfDPRank % targetInfo.mPeerDupHeadFactor))
        {
            for (int j = 0; j < targetInfo.mDomainPPSize; j++)
            {
                size_t localIdx = (i * targetInfo.mDomainPPSize) + j;
                pickUpConnections.push_back(baseIndices.at(localIdx));
                localRankIndices.push_back(localIdx);
            }
        }
    }
    return {pickUpConnections, localRankIndices};
}

/// @brief Convenience overload that computes KV-cache target ranks automatically.
inline std::pair<std::vector<size_t>, std::vector<size_t>> pickRecvConnections(size_t numConnections,
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig,
    std::vector<SizeType32> const& counterPartRanks)
{
    return pickRecvConnections(numConnections, selfConfig, selfIdx, destConfig, counterPartRanks,
        executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx));
}

} // namespace cache_formatter_utils

} // namespace tensorrt_llm::batch_manager
