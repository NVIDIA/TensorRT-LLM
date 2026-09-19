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
#include "tensorrt_llm/common/assert.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <sys/types.h>
#include <variant>
#include <vector>

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

// Index of a cache level (0 = hot; subsequent levels are colder storage tiers).
using CacheLevel = StrongIndex<int, struct CacheLevelTag, 0>;
// The kernel-facing hot level; colder levels may also use GPU memory.
inline constexpr CacheLevel kHotLevel{0};

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
//! Non-owning view into a contiguous buffer (a C++17 stand-in for std::span).
//!
//! The referenced buffer must outlive the view. This remains an aggregate, so
//! `Span<T>{}` creates an empty view and `Span<T>{ptr, len}` is plain brace-init.
//! Supports operator[] for uniform access with std::vector<T>.
// ---------------------------------------------------------------------------
template <typename T>
struct Span
{
    T* ptr = nullptr;
    int len = 0;

    T& operator[](int idx)
    {
        return ptr[idx];
    }

    T const& operator[](int idx) const
    {
        return ptr[idx];
    }

    int size() const noexcept
    {
        return len;
    }

    T* data() const noexcept
    {
        return ptr;
    }

    T* begin() const noexcept
    {
        return ptr;
    }

    T* end() const noexcept
    {
        return ptr + len;
    }
};

//! Create a non-owning const Span over a std::vector.
//!
//! The source vector must outlive the returned view and must not reallocate while
//! the view is in use.
template <typename T>
inline Span<T const> toSpan(std::vector<T> const& vec) noexcept
{
    TLLM_CHECK_DEBUG(vec.size() <= static_cast<size_t>(std::numeric_limits<int>::max()));
    return Span<T const>{vec.data(), static_cast<int>(vec.size())};
}

//! Non-owning view of a token sequence; the source buffer must outlive the view.
//!
//! Used on the hot ingest path: a digest-free int32 token buffer can be
//! reinterpret_cast to TokenIdExt const* and matched/hashed with no per-token
//! copy. TokenIdExt is 4 bytes and bit-identical to a normal int32 token (see
//! tokenIdExt.h).
using TokenSpan = Span<TokenIdExt const>;

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
