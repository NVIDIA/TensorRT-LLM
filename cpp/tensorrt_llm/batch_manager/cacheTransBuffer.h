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

#include "tensorrt_llm/batch_manager/baseTransBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class FabricMemory
{
public:
    explicit FabricMemory(size_t size);
    ~FabricMemory();

    FabricMemory(FabricMemory const&) = delete;
    FabricMemory& operator=(FabricMemory const&) = delete;

    FabricMemory(FabricMemory&&) noexcept;
    FabricMemory& operator=(FabricMemory&&) noexcept;

    void* getPtr() const;
    size_t getSize() const;

    static size_t getAlignedSize(size_t size);
    static bool supportFbaricMemory();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/// @brief KV Cache specific transfer buffer manager.
/// Inherits common buffer management from BaseTransBufferManager.
class CacheTransBufferManager : public BaseTransBufferManager
{
public:
    CacheTransBufferManager(KVCacheManager::BaseKVCacheManager* cacheManager,
        std::optional<size_t> maxNumTokens = std::nullopt, bool transferIndexerKCache = false);

    static size_t preAllocBufferSize(std::map<SizeType32, SizeType32> const& cacheSizeBytesPerTokenPerWindow,
        SizeType32 tokensPerBlock,
        std::optional<executor::CacheTransceiverConfig> const& cacheTransceiverConfig = std::nullopt);

    /// @brief Get the KV cache manager.
    [[nodiscard]] KVCacheManager::BaseKVCacheManager* getCacheManager() const noexcept
    {
        return mCacheManager;
    }

private:
    /// @brief Compute transfer buffer size from KV cache configuration.
    static size_t computeTransferBufferSize(KVCacheManager::BaseKVCacheManager* cacheManager,
        std::optional<size_t> maxNumTokens, bool transferIndexerKCache);

    KVCacheManager::BaseKVCacheManager* mCacheManager;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
