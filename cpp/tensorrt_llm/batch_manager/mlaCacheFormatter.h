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

#include "cacheFormatter.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

/**
 * @brief Calculate the number of blocks allocated to a specific Context Parallelism (CP) rank.
 *
 * This function determines how many blocks should be allocated to a given CP rank when
 * distributing a total number of blocks across multiple CP ranks. It supports two distribution
 * modes: strict and non-strict.
 *
 * @param cpRank The rank (index) of the current CP process. Must be in range [0, cpSize).
 * @param cpSize The total number of CP ranks/processes in the parallel group.
 * @param numTotalBlocks The total number of blocks to be distributed across all CP ranks.
 * @param strict Flag controlling the distribution strategy:
 *               - true: Use strict round-robin distribution with exact allocation
 *               - false: Use ceiling division which may over-allocate
 *
 * @return The number of blocks allocated to the specified CP rank.
 */
int getBlockNumAccountingForCP(int cpRank, int cpSize, int numTotalBlocks, bool strict);

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class MLACacheFormatter final : public BaseCacheFormatter
{
public:
    MLACacheFormatter(BaseKVCacheManager* cacheManager, CacheTransBufferManager* cacheTransBufferManager)
        : mCacheManager{cacheManager}
        , mCacheTransBufferManager{cacheTransBufferManager}
    {
        TLLM_CHECK(mCacheManager);
        TLLM_CHECK(mCacheTransBufferManager);
    }

    void format(tensorrt_llm::batch_manager::TransferSession& session) override;

    void unformat(tensorrt_llm::batch_manager::TransferSession& session) override;

    [[nodiscard]] bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const override;

    [[nodiscard]] std::vector<SizeType32> getCounterparts(
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const override
    {
        return executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx).mIRanks;
    }

    [[nodiscard]] BaseKVCacheManager* getCacheManager() const noexcept override
    {
        return mCacheManager;
    }

    static bool needSendCache(CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx);
    std::vector<size_t> pickRecvConnections(size_t numConnections, CacheState const& selfConfig, SizeType32 selfIdx,
        CacheState const& destConfig) const override;

private:
    BaseKVCacheManager* mCacheManager;
    CacheTransBufferManager* mCacheTransBufferManager;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
