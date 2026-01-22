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
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstddef>
#include <optional>

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

/// @brief RNN Cache specific transfer buffer manager.
/// Inherits common buffer management from BaseTransBufferManager.
class RnnCacheTransBufferManager : public BaseTransBufferManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    // using RnnCacheState = executor::rnn_cache::RnnCacheState;

    /// @brief Constructor.
    /// @param rnnStateManager Pointer to the RNN state manager.
    /// @param maxNumTokens Optional maximum number of tokens for buffer sizing.
    RnnCacheTransBufferManager(RnnStateManager* rnnStateManager, std::optional<size_t> maxNumTokens = std::nullopt);

    /// @brief Calculate pre-allocated buffer size for RNN state transfer.
    /// @param rnnStateSizeBytes Total size of RNN state in bytes.
    /// @param cacheTransceiverConfig Optional transceiver configuration.
    /// @return Total pre-allocated buffer size.
    static size_t preAllocBufferSize(
        size_t rnnStateSizeBytes, std::optional<executor::CacheTransceiverConfig> const& cacheTransceiverConfig);

    /// @brief Get the RNN state manager.
    [[nodiscard]] RnnStateManager* getRnnStateManager() const noexcept
    {
        return mRnnStateManager;
    }

    /// @brief set dtypes
    // void setDtypes(RnnCacheState const& cacheState) noexcept;

private:
    /// @brief Compute transfer buffer size from RNN state configuration.
    static size_t computeTransferBufferSize(RnnStateManager* rnnStateManager, std::optional<size_t> maxNumTokens);

    RnnStateManager* mRnnStateManager;
    nvinfer1::DataType mConvStateDataType;
    nvinfer1::DataType mSsmStateDataType;
};

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
