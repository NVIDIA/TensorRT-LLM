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

#include "tensorrt_llm/executor/dataTransceiverState.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::rnn_cache_routing
{

using CacheState = executor::kv_cache::CacheState;
using SizeType32 = runtime::SizeType32;

//! Check RNN model, state datatype, and context-parallel compatibility for cache transfer.
[[nodiscard]] bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig);

//! Select peer ranks that overlap this rank's RNN layers and tensor-parallel state.
[[nodiscard]] std::vector<SizeType32> getCounterparts(
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig);

//! Select receive connections and local rank indices from the combined KV/RNN counterpart list.
[[nodiscard]] std::pair<std::vector<size_t>, std::vector<size_t>> pickRecvConnections(size_t numConnections,
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig,
    std::vector<SizeType32> const& counterPartRanks);

} // namespace tensorrt_llm::batch_manager::rnn_cache_routing
