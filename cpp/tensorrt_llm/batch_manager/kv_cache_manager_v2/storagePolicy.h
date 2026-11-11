/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/utils/typedIndex.h"
#include "tensorrt_llm/common/smallVector.h"

#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Inline capacities of the KV cache manager's small containers. A block holds the pages of up to
//! kInlineLifeCycles life cycles and kInlineBeams beams inline, so walking a request's blocks stays
//! local in memory; more life cycles or beams spill to the heap.
inline constexpr uint32_t kInlineLifeCycles = 4;
inline constexpr uint32_t kInlineBeams = 1;
//! A request-private page's finish events; only shared prefix pages spill to the heap.
inline constexpr uint32_t kInlineFinishEvents = 2;

template <typename T>
using LifeCycleVec = TypedVec<LifeCycleId, T, tensorrt_llm::common::SmallVector<T, kInlineLifeCycles>>;

template <typename T>
using BeamVec = TypedVec<BeamIndex, T, tensorrt_llm::common::SmallVector<T, kInlineBeams>>;

template <typename T>
using FinishEventVec = tensorrt_llm::common::SmallVector<T, kInlineFinishEvents>;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
