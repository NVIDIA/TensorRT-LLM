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

// Thread-local free lists for KVCM2's per-page bookkeeping objects (control blocks, list and deque
// nodes, short per-block vectors). Claiming or releasing a long prefix allocates or frees them by the
// thousand in one burst, which overflows the general-purpose allocator's thread cache.
//
// Blocks come from ::operator new and go back to ::operator delete past the per-class cap. The cache
// is trivially destructible, so thread and process exit are safe; a thread's cached blocks are not
// returned when it exits (the cap bounds them).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace detail
{

inline constexpr size_t kPoolGranule = 16;
inline constexpr size_t kPoolMaxSize = 512;
inline constexpr size_t kPoolNumClasses = kPoolMaxSize / kPoolGranule;
//! Blocks kept per size class and thread before frees go back to the heap.
inline constexpr uint32_t kPoolMaxCached = 1u << 17;

struct PoolFreeNode
{
    PoolFreeNode* next;
};

struct PoolThreadCache
{
    PoolFreeNode* head[kPoolNumClasses];
    uint32_t count[kPoolNumClasses];
};

inline thread_local constinit PoolThreadCache tPoolThreadCache{};

//! Whether a type of this size and alignment is served by the pool.
constexpr bool poolServes(size_t size, size_t align) noexcept
{
    return size != 0 && size <= kPoolMaxSize && align <= kPoolGranule && align <= __STDCPP_DEFAULT_NEW_ALIGNMENT__;
}

constexpr size_t poolClass(size_t size) noexcept
{
    return (size + kPoolGranule - 1) / kPoolGranule - 1;
}

inline void* poolAllocate(size_t size)
{
    size_t const cls = poolClass(size);
    auto& cache = tPoolThreadCache;
    if (PoolFreeNode* node = cache.head[cls])
    {
        cache.head[cls] = node->next;
        --cache.count[cls];
        return node;
    }
    return ::operator new((cls + 1) * kPoolGranule);
}

inline void poolDeallocate(void* ptr, size_t size) noexcept
{
    size_t const cls = poolClass(size);
    auto& cache = tPoolThreadCache;
    if (cache.count[cls] < kPoolMaxCached)
    {
        auto* node = static_cast<PoolFreeNode*>(ptr);
        node->next = cache.head[cls];
        cache.head[cls] = node;
        ++cache.count[cls];
        return;
    }
    ::operator delete(ptr);
}

} // namespace detail

//! Standard allocator over the thread-local pool; larger or over-aligned allocations use std::allocator.
template <typename T>
struct PoolAllocator
{
    using value_type = T;

    PoolAllocator() noexcept = default;

    template <typename U>
    PoolAllocator(PoolAllocator<U> const&) noexcept // NOLINT(google-explicit-constructor)
    {
    }

    T* allocate(size_t n)
    {
        if (n <= detail::kPoolMaxSize / sizeof(T) && detail::poolServes(n * sizeof(T), alignof(T)))
        {
            return static_cast<T*>(detail::poolAllocate(n * sizeof(T)));
        }
        // std::allocator handles over-alignment and size overflow.
        return std::allocator<T>{}.allocate(n);
    }

    void deallocate(T* ptr, size_t n) noexcept
    {
        if (n <= detail::kPoolMaxSize / sizeof(T) && detail::poolServes(n * sizeof(T), alignof(T)))
        {
            detail::poolDeallocate(ptr, n * sizeof(T));
            return;
        }
        std::allocator<T>{}.deallocate(ptr, n);
    }

    template <typename U>
    bool operator==(PoolAllocator<U> const&) const noexcept
    {
        return true;
    }
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
