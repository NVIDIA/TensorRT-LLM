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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Smallest unit the backing allocates in, and the alignment every size it is
//! given must satisfy.
inline constexpr size_t kHostMemAlignment = 2ULL << 20;

struct HostMemBackingOptions
{
    //! Unit memory is registered and released in. Also the alignment every
    //! commit and decommit bound must satisfy, and therefore how finely a resize
    //! can give memory back.
    //!
    //! Each unit is a separate registration, so it must divide whatever ranges
    //! are copied in one call: the driver rejects a copy that straddles two
    //! registrations. It need not be a multiple of the page size -- registration
    //! accepts any alignment, and the madvise and mprotect calls round to pages
    //! themselves, outwards when they must cover the range and inwards when they
    //! must not reach past it.
    size_t commitUnit = kHostMemAlignment;

    //! Whether a page may come from a NUMA node other than the one the GPU
    //! attaches to, when the local node cannot satisfy it.
    //!
    //! Applied as MPOL_PREFERRED when true and MPOL_BIND when false. False caps
    //! the tier at one node's memory, so a tier sized from total system memory
    //! cannot be filled. It also requires that a memory policy be settable: a
    //! container without CAP_SYS_NICE cannot apply MPOL_BIND, and construction
    //! fails rather than silently running with first-touch placement, which
    //! would take a remote page instead of refusing.
    bool allowRemoteNumaFallback = true;
};

//! Physical backing for the host tier: reserve an address range once, then
//! commit it a unit at a time.
//!
//! The base address returned by reserve() is fixed for the lifetime of the
//! object. Growing and shrinking touch only the delta, so a resize costs
//! O(delta) rather than O(total) and never invalidates a pointer the allocator
//! above has handed out.
//!
//! Pages are populated through the kernel's fault path (MADV_POPULATE_WRITE)
//! and then handed to the driver with cuMemHostRegister. Allocating and zeroing
//! in the kernel rather than inside cuMemCreate is what lets several processes
//! fill at once: the driver serializes that work on a per-process lock, which on
//! an 8-GPU node collapsed per-process fill throughput by roughly 5x and blocked
//! unrelated CUDA calls on the same process for seconds at a time.
//!
//! Registration happens in units rather than over the whole range so that a
//! shrink can unregister exactly what it releases; a registration can only be
//! dropped whole. Each unit stays far below the 2 GiB-per-call pinning limit on
//! Linux 6.11.
class HostMemBacking
{
public:
    explicit HostMemBacking(HostMemBackingOptions const& options);
    ~HostMemBacking();

    HostMemBacking(HostMemBacking const&) = delete;
    HostMemBacking& operator=(HostMemBacking const&) = delete;
    HostMemBacking(HostMemBacking&&) = delete;
    HostMemBacking& operator=(HostMemBacking&&) = delete;

    //! Reserves `size` bytes of address space, page-aligned as mmap returns it
    //! and no coarser. No page is resident and none is reachable: the whole
    //! range is PROT_NONE until commit() opens it, so a read that runs ahead of
    //! the fill faults instead of returning zeros.
    //!
    //! Must be called exactly once, before any commit.
    [[nodiscard]] MemAddress reserve(size_t size);

    //! Makes [offset, offset + size) resident and reachable by both the CPU and
    //! the GPU. Both bounds must be multiples of commitGranularity().
    //!
    //! Throws HostOOMError when the system cannot back the range; that is the
    //! expected failure and callers are required to handle it.
    //!
    //! With `populated` true the caller has already run open() and populate()
    //! on the range, and commit() only registers it.
    void commit(size_t offset, size_t size, bool populated = false);

    //! The first two steps of commit(): open() makes the range reachable and
    //! populate() allocates its pages. Neither unwinds on failure; decommit()
    //! releases what they took.
    //!
    //! open() changes the mapping, which waits for every populate() in flight
    //! on any thread: open a range once, then populate it in parallel. Both
    //! are safe to call while another thread commits a disjoint range.
    void open(size_t offset, size_t size);
    void populate(size_t offset, size_t size) const;

    //! Releases the physical backing of [offset, offset + size), which stays
    //! reserved and may be committed again. Both bounds must be multiples of
    //! commitGranularity().
    void decommit(size_t offset, size_t size) noexcept;

    //! Releases everything, including the reservation. Idempotent.
    void destroy() noexcept;

    //! The unit memory is populated, registered and released in.
    [[nodiscard]] size_t commitGranularity() const noexcept
    {
        return mUnit;
    }

private:
    void unregisterRange(size_t offset, size_t size) noexcept;

    size_t mUnit = 0;
    bool mAllowRemoteNumaFallback = true;
    MemAddress mAddr = 0;
    size_t mSize = 0;

    //! Bytes actually mapped: mSize rounded up to a page, since a unit is a
    //! whole number of slots and need not end on one.
    size_t mMapped = 0;

    //! Offsets of the units currently registered. A unit is registered exactly
    //! when it is committed, so this is also what destroy() must undo.
    std::map<size_t, size_t> mRegistered;

    //! Captured in reserve(), which runs on the thread that owns the context.
    //! cuMemHostRegister needs a current context and the fill worker only sets
    //! the device, which leaves it without one.
    void* mContext = nullptr;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
