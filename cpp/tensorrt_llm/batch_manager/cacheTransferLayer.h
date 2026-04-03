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

#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/common.h"

#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class TransferSession;

namespace kv_cache_manager
{
class BaseCacheFormatter;
} // namespace kv_cache_manager

using BaseCacheFormatter = kv_cache_manager::BaseCacheFormatter;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

/// @brief Bundles all cache states and formatters into a single object that CacheSender/CacheReceiver
/// interact with. Provides unified methods for validation, counterpart computation, and formatting
/// so that the data transceiver is agnostic to the specific cache types.
class CacheTransferLayer
{
public:
    /// @brief Constructor.
    /// @param cacheState The cache state (KV, and optionally RNN if hasRnnConfig() is true).
    /// @param kvFormatter The KV cache formatter.
    /// @param rnnFormatter Optional RNN cache formatter.
    CacheTransferLayer(executor::kv_cache::CacheState cacheState, std::unique_ptr<BaseCacheFormatter> kvFormatter,
        std::unique_ptr<RnnCacheFormatter> rnnFormatter = nullptr);

    ~CacheTransferLayer();
    CacheTransferLayer(CacheTransferLayer&&) noexcept;
    CacheTransferLayer& operator=(CacheTransferLayer&&) noexcept;

    /// @brief Validates all cache types against the peer state. Throws on incompatibility.
    /// @param peerState The peer's DataTransceiverState.
    void validateSupport(executor::DataTransceiverState const& peerState) const;

    /// @brief Computes counterparts for all cache types and returns the union (KV + RNN if present).
    /// @param selfIdx The sequential index of the current executor process.
    /// @param peerState The peer's DataTransceiverState.
    /// @return Union of counterparts across all cache types.
    [[nodiscard]] std::vector<SizeType32> computeCounterparts(
        SizeType32 selfIdx, executor::DataTransceiverState const& peerState) const;

    /// @brief Calls all formatters sequentially.
    /// @param session The transfer session.
    void format(TransferSession& session) const;

    /// @brief Calls all unformatters sequentially.
    /// @param session The transfer session.
    void unformat(TransferSession& session) const;

    [[nodiscard]] executor::kv_cache::CacheState const& getCacheState() const noexcept;

    [[nodiscard]] kv_cache_manager::BaseKVCacheManager* getCacheManager() const noexcept;

    [[nodiscard]] BaseCacheFormatter* getKvFormatter() const noexcept;

private:
    executor::kv_cache::CacheState mCacheState;
    std::unique_ptr<BaseCacheFormatter> mKvFormatter;
    std::unique_ptr<RnnCacheFormatter> mRnnFormatter;
};

} // namespace tensorrt_llm::batch_manager
