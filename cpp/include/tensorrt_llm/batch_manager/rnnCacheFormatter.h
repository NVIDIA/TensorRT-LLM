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
class RnnCacheFormatter : public kv_cache_manager::BaseCacheFormatter
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheState = executor::kv_cache::CacheState;
    using RequestIdType = tensorrt_llm::batch_manager::RequestIdType;

    /// @brief Constructor.
    /// @param rnnStateManager Pointer to the RNN state manager.
    /// @param rnnCacheTransBufferManager Pointer to the RNN cache transfer buffer manager.
    RnnCacheFormatter(rnn_state_manager::RnnStateManager* rnnStateManager,
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

    /// @brief Returns nullptr since RNN cache formatter doesn't use KV cache manager.
    [[nodiscard]] kv_cache_manager::BaseKVCacheManager* getCacheManager() const noexcept override
    {
        return nullptr;
    }

    /// @brief Get the RNN state manager.
    /// @return Pointer to the RNN state manager.
    [[nodiscard]] rnn_state_manager::RnnStateManager* getRnnStateManager() const noexcept
    {
        return mRnnStateManager;
    }

private:
    rnn_state_manager::RnnStateManager* mRnnStateManager;
    rnn_state_manager::RnnCacheTransBufferManager* mRnnCacheTransBufferManager;
};

} // namespace tensorrt_llm::batch_manager
