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

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class MLACacheFormatter final : public BaseCacheFormatter
{
public:
    MLACacheFormatter(BaseKVCacheManager* cacheManager, std::vector<CacheTransBufferManager*> cacheTransBufferManagers)
        : mCacheManager{cacheManager}
        , mCacheTransBufferManagers{cacheTransBufferManagers}
    {
        TLLM_CHECK(mCacheManager);
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
    std::vector<CacheTransBufferManager*> mCacheTransBufferManagers;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
