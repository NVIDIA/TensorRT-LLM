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

#include <cstddef>
#include <cstdint>
#include <memory>

// Reuse the KV-cache transfer buffers' fabric allocator (batch_manager) rather than a private copy:
// one MNNVL/GPUDirect-RDMA implementation, shared. Forward-declared here (the member is held by
// unique_ptr with an out-of-line dtor); the full definition is included in the .cpp.
namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class FabricMemory;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// BounceArena — the ONE shared bounce data buffer (DESIGN)
// ----------------------------------------------------------------------------
// Role
//   Pure STORAGE: a single contiguous `bytes`-byte device buffer registered ONCE with NIXL and
//   shared by BOTH roles. CreditScheduler (its BuddyAllocator) carves variable-size regions out of
//   it by byte offset — receiver grants (remote senders' RDMA-write targets) and the local sender's
//   gather staging both draw from the same arena. Unlike the old fixed-slot pool, a chunk gets a
//   region sized to its actual bytes, so many small requests pack densely (high concurrency, no
//   per-slot waste) while a request larger than the arena streams through chunk-by-chunk.
//
// Allocation
//   On MNNVL parts (GH200/GB200) the buffer must be fabric memory (cuMemCreate +
//   CU_MEM_HANDLE_TYPE_FABRIC) to be NVLink-fabric reachable + GPUDirect-RDMA capable, matching how
//   the KV-cache transfer buffers are allocated. Elsewhere (and CI / x86, or when fabric is force-
//   disabled) it falls back to cudaMalloc. The device-pointer surface is identical either way.
//
// Threading
//   base()/baseAddr()/at() are const lookups into an immutable buffer; safe from the IO thread and
//   the scatter workers concurrently. WHICH region a role may touch is arbitrated by CreditScheduler
//   (a region is granted/acquired to exactly one user until it is freed).
// ============================================================================
class BounceArena
{
public:
    /// Allocate one `bytes`-byte device buffer. When `allowFabric` and the device is fabric-capable,
    /// it is fabric memory; otherwise cudaMalloc. Throws on CUDA allocation failure.
    BounceArena(std::size_t bytes, int deviceId, bool allowFabric);
    ~BounceArena();

    BounceArena(BounceArena const&) = delete;
    BounceArena& operator=(BounceArena const&) = delete;

    [[nodiscard]] void* base() const noexcept
    {
        return mBase;
    }

    [[nodiscard]] std::uint64_t baseAddr() const noexcept
    {
        return reinterpret_cast<std::uint64_t>(mBase);
    }

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return mBytes;
    }

    /// Device address of `offset` bytes into the arena (an arena region's start).
    [[nodiscard]] void* at(std::uint64_t offset) const noexcept
    {
        return static_cast<char*>(mBase) + offset;
    }

    [[nodiscard]] bool isFabric() const noexcept
    {
        return mIsFabric;
    }

private:
    int mDeviceId{0};
    std::size_t mBytes{0};
    void* mBase{nullptr}; // the registered RDMA src/dst buffer (offset 0)
    // Non-null (and mIsFabric=true) on MNNVL when fabric-backed; else null and mBase is cudaMalloc'd.
    std::unique_ptr<tensorrt_llm::batch_manager::kv_cache_manager::FabricMemory> mFabric;
    bool mIsFabric{false};
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
