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

#include "kv_cache_manager_v2/tokenIdExt.h" // TokenId, Digest, TokenIdExt
#include "kv_cache_manager_v2/utils/typedIndex.h"
#include "tensorrt_llm/batch_manager/common.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Debug flag — true when TLLM_DEBUG_MODE=1.
// Delegates to DebugConfig::isCheckDebugEnabled() for consistency with TLLM_CHECK_DEBUG.
// ---------------------------------------------------------------------------
extern bool const gDebug; // true == debug mode (expensive assertions enabled)

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

enum class PageStatus : int
{
    LOCKED = 0,    // Required in GPU. Eviction/dropping not allowed.
    HELD = 1,      // Allow eviction but not dropping.
    DROPPABLE = 2, // Allow eviction and dropping.
};

enum class CacheTier : int
{
    GPU_MEM = 0,
    HOST_MEM = 1,
    DISK = 2,
};

// PageIndexMode — how converted page indices relate to layers within a layer group.
// Mirrors _common.py::PageIndexMode.
enum class PageIndexMode : int
{
    // Converted index list is shared across layers in the same LayerGroup.
    // Base pointer is per-layer (includes attr.offset).
    SHARED = 0,
    // Converted index list is per-layer.
    // Base pointer is shared (pool group base, no attr.offset).
    PER_LAYER = 1,
};

// ---------------------------------------------------------------------------
// Strongly-typed integer aliases (mirroring Python NewType wrappers).
// ---------------------------------------------------------------------------

// Index of a cache level (0 = GPU, 1 = host, 2 = disk, ...).
using CacheLevel = StrongIndex<int, struct CacheLevelTag, 0>;
inline constexpr CacheLevel kGpuLevel{0};

// Opaque request identifier shared with the rest of the batch manager.
using RequestIdType = tensorrt_llm::batch_manager::RequestIdType;

// Opaque LoRA task identifier shared with the rest of the batch manager.
using LoraTaskIdType = tensorrt_llm::runtime::LoraTaskIdType;

// Ordinal index of a KV cache block (sequence of tokens).
using BlockOrdinal = StrongIndex<int, struct BlockOrdinalTag, -1>;
inline constexpr BlockOrdinal kBadBlockOrdinal{-1};

// Identifier of an attention layer.
using LayerId = int;

// Raw CUDA stream handle (CUstream cast to uintptr_t).
using CudaStream = uintptr_t;

// Index of a beam in beam-search.
using BeamIndex = StrongIndex<int, struct BeamIndexTag, 0>;
inline constexpr BeamIndex kDefaultBeamIndex{0};

// User-defined request/session identifier.
using UserId = int64_t;

// Host or device memory address (uintptr_t).
using MemAddress = std::uintptr_t;

// OS file descriptor.
using FileDescriptor = int;
inline constexpr FileDescriptor kBadFileDescriptor = -1;

// Index into a page table.
using PageIndex = StrongIndex<int, struct PageIndexTag, -1>;
inline constexpr PageIndex kBadPageIndex{-1};

// Eviction priority (0 = highest priority to evict, 100 = lowest).
using Priority = int;
inline constexpr Priority kPriorityMin = 0;
inline constexpr Priority kPriorityMax = 100;
inline constexpr Priority kPriorityDefault = 35;

// Optional sliding window size (nullopt = no sliding window).
using SlidingWindowSize = std::optional<int>;

// ---------------------------------------------------------------------------
// Address types
// ---------------------------------------------------------------------------

// Disk address: (fd, byte-offset).
struct DiskAddress
{
    int fd = kBadFileDescriptor;
    ssize_t pos = 0;

    bool operator==(DiskAddress const& o) const noexcept
    {
        return fd == o.fd && pos == o.pos;
    }
};

// Unified address: either a host/device memory pointer or a disk address.
using Address = std::variant<MemAddress, DiskAddress>;

// ---------------------------------------------------------------------------
// DataRole — string-typed tag for a buffer inside one attention layer.
// ---------------------------------------------------------------------------
using DataRole = std::string;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
