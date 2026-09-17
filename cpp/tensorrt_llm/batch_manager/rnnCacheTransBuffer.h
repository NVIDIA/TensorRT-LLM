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

#include "tensorrt_llm/batch_manager/baseTransBuffer.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstddef>
#include <optional>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class BaseKVCacheManager;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

/// @brief RNN Cache specific transfer buffer manager.
/// Inherits common buffer management from BaseTransBufferManager.
class RnnCacheTransBufferManager : public BaseTransBufferManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheState = executor::kv_cache::CacheState;

    /// @brief Constructor for unified pool path (CppMambaHybridCacheManager).
    /// Computes buffer sizes from the KV cache manager's recurrent state pool metadata.
    /// @param kvCacheManager Pointer to the KV cache manager with unified pool.
    /// @param cacheState The CacheState containing RNN model config.
    /// @param maxNumTokens Optional maximum number of tokens for buffer sizing.
    RnnCacheTransBufferManager(kv_cache_manager::BaseKVCacheManager* kvCacheManager,
        executor::kv_cache::CacheState const& cacheState, std::optional<size_t> maxNumTokens = std::nullopt);

    /// @brief Calculate pre-allocated buffer size for RNN state transfer.
    /// @param rnnStateSizeBytes Total size of RNN state in bytes.
    /// @param cacheTransceiverConfig Optional transceiver configuration.
    /// @return Total pre-allocated buffer size.
    static size_t preAllocBufferSize(
        size_t rnnStateSizeBytes, std::optional<executor::CacheTransceiverConfig> const& cacheTransceiverConfig);

    [[nodiscard]] BufferKind getBufferKind() const override
    {
        return BufferKind::kRNN;
    }

private:
    /// @brief Compute transfer buffer size from unified pool metadata.
    static size_t computeTransferBufferSizeFromPool(kv_cache_manager::BaseKVCacheManager* kvCacheManager,
        executor::kv_cache::CacheState const& cacheState, std::optional<size_t> maxNumTokens);
};

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
