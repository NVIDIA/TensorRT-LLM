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

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/rnnCacheTransBuffer.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/common.h"

#include <vector>

namespace tensorrt_llm::batch_manager
{

class TransferSession;

namespace rnn_state_manager
{
class RnnStateManager;
class RnnCacheTransBufferManager;
} // namespace rnn_state_manager

/// @brief RNN Cache Formatter for formatting/unformatting RNN states during transfer.
/// Supports two operating modes:
/// - Slot mode: uses RnnStateManager (for CppMambaCacheManager, separate tensor storage)
/// - Unified pool mode: uses BaseKVCacheManager (for CppMambaHybridCacheManager, block-indexed pool)
class RnnCacheFormatter : public kv_cache_manager::BaseCacheFormatter
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheState = executor::kv_cache::CacheState;
    using RequestIdType = tensorrt_llm::batch_manager::RequestIdType;

    /// @brief Constructor for slot-based mode (CppMambaCacheManager with RnnStateManager).
    /// @param rnnStateManager Pointer to the RNN state manager.
    /// @param rnnCacheTransBufferManager Pointer to the RNN cache transfer buffer manager.
    RnnCacheFormatter(rnn_state_manager::RnnStateManager* rnnStateManager,
        rnn_state_manager::RnnCacheTransBufferManager* rnnCacheTransBufferManager);

    /// @brief Constructor for unified pool mode (CppMambaHybridCacheManager).
    /// @param kvCacheManager Pointer to the KV cache manager with unified pool.
    /// @param rnnCacheTransBufferManager Pointer to the RNN cache transfer buffer manager.
    RnnCacheFormatter(kv_cache_manager::BaseKVCacheManager* kvCacheManager,
        rnn_state_manager::RnnCacheTransBufferManager* rnnCacheTransBufferManager);

    /// @brief Format RNN states for sending.
    /// @param session The transfer session.
    void format(TransferSession& session) override;

    /// @brief Unformat received RNN states.
    /// @param session The transfer session.
    void unformat(TransferSession& session) override;

    /// @brief Check if transfer is supported between two configurations.
    [[nodiscard]] bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const override;

    /// @brief Get the counterpart ranks for communication.
    [[nodiscard]] std::vector<SizeType32> getCounterparts(
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const override;

    /// @brief Pick receive connections and their corresponding local rank indices.
    [[nodiscard]] std::pair<std::vector<size_t>, std::vector<size_t>> pickRecvConnections(size_t numConnections,
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig,
        std::vector<SizeType32> const& counterPartRanks) const override;

    /// @brief Returns the KV cache manager (non-null in unified pool mode).
    [[nodiscard]] kv_cache_manager::BaseKVCacheManager* getCacheManager() const noexcept override
    {
        return mKvCacheManager;
    }

    /// @brief Get the RNN state manager (non-null in slot mode).
    /// @return Pointer to the RNN state manager.
    [[nodiscard]] rnn_state_manager::RnnStateManager* getRnnStateManager() const noexcept
    {
        return mRnnStateManager;
    }

    /// @brief Check if operating in unified pool mode.
    [[nodiscard]] bool isUnifiedPoolMode() const noexcept
    {
        return mKvCacheManager != nullptr;
    }

private:
    /// @brief Format logic for slot-based path (RnnStateManager).
    void formatSlotMode(TransferSession& session);

    /// @brief Unformat logic for slot-based path (RnnStateManager).
    void unformatSlotMode(TransferSession& session);

    /// @brief Format logic for unified pool path (BaseKVCacheManager).
    void formatUnifiedPoolMode(TransferSession& session);

    /// @brief Unformat logic for unified pool path (BaseKVCacheManager).
    void unformatUnifiedPoolMode(TransferSession& session);

    rnn_state_manager::RnnStateManager* mRnnStateManager{nullptr};
    rnn_state_manager::RnnCacheTransBufferManager* mRnnCacheTransBufferManager;
    kv_cache_manager::BaseKVCacheManager* mKvCacheManager{nullptr};
};

} // namespace tensorrt_llm::batch_manager
