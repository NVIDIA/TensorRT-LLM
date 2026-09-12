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
#include <memory>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Physical backing for the host tier: reserve an address range once, then
//! commit it in chunks.
//!
//! The base address returned by reserve() is fixed for the lifetime of the
//! object. Growing and shrinking touch only the delta, so a resize costs
//! O(delta) rather than O(total) and never invalidates a pointer the allocator
//! above has handed out.
//!
//! Two implementations exist because the fastest route to host memory differs
//! by platform:
//!
//!   kMmap  Plain anonymous mmap with no cuMemHostRegister. Requires ATS, where
//!          the GPU shares the CPU's page tables and reaches host memory at full
//!          C2C bandwidth without the driver pinning anything.
//!   kVmm   cuMemCreate(HOST_NUMA) mapped into a cuMemAddressReserve range.
//!          Used where ATS is unavailable: the memory is driver-allocated and
//!          device-mapped, so copies stay asynchronous without registration.
//!
//! Neither implementation calls cuMemHostRegister, which is why neither is
//! subject to the 2 GiB-per-call pinning limit on Linux 6.11.
class IHostMemBacking
{
public:
    virtual ~IHostMemBacking() = default;

    IHostMemBacking(IHostMemBacking const&) = delete;
    IHostMemBacking& operator=(IHostMemBacking const&) = delete;
    IHostMemBacking(IHostMemBacking&&) = delete;
    IHostMemBacking& operator=(IHostMemBacking&&) = delete;

    //! Reserves `size` bytes of address space. No physical memory is committed
    //! and no page is resident. Must be called exactly once, before any commit.
    [[nodiscard]] virtual MemAddress reserve(size_t size) = 0;

    //! Makes [offset, offset + size) resident and reachable by both the CPU and
    //! the GPU. `offset` and `size` must be multiples of commitGranularity().
    //! Committing a range twice is not allowed.
    //!
    //! A backing that allocates physical memory splits the request into
    //! commitGranularity()-sized allocations, so callers are free to commit in
    //! whatever span suits them without coupling that span to the unit a later
    //! decommit can release.
    //!
    //! Throws HostOOMError when the system cannot back the range; that is the
    //! expected failure and callers are required to handle it.
    virtual void commit(size_t offset, size_t size) = 0;

    //! Releases the physical backing of [offset, offset + size). The address
    //! range stays reserved and may be committed again later.
    //!
    //! Both bounds must be multiples of commitGranularity(), which is what lets
    //! this be exact: the backing allocates in units of that size, so a release
    //! never has to split one.
    virtual void decommit(size_t offset, size_t size) noexcept = 0;

    //! Releases everything, including the reservation. Idempotent.
    virtual void destroy() noexcept = 0;

    //! The unit this backing allocates and releases in. Commit and decommit
    //! offsets and sizes must be multiples of it.
    //!
    //! This is deliberately independent of how much the filler commits per
    //! call: it sets how finely a later resize can give memory back.
    [[nodiscard]] virtual size_t commitGranularity() const noexcept = 0;

    //! Stable identifier for logging and for tests that assert which backing ran.
    [[nodiscard]] virtual char const* name() const noexcept = 0;

protected:
    IHostMemBacking() = default;
};

enum class HostMemBackingKind : std::uint8_t
{
    kMmap,
    kVmm,
};

//! Picks kMmap only when the GPU reaches pageable host memory through the CPU's
//! page tables (ATS) *and* the link is coherent.
//!
//! Both conditions are required, not just ATS. Together they identify the
//! platform class where mmap was measured to win: a coherent link is what makes
//! BatchedPageCopier take its SM-kernel path, which reads host memory directly.
//! Where the copy engine runs instead, VMM is the faster source.
//!
//! A caller that needs a specific backing names it to createHostMemBacking()
//! rather than going through this.
[[nodiscard]] HostMemBackingKind selectHostMemBackingKind();

//! `commitUnit` becomes the backing's commitGranularity(); it is rounded up to
//! whatever the platform requires. Larger units make the fill cheaper and a
//! resize coarser.
[[nodiscard]] std::unique_ptr<IHostMemBacking> createHostMemBacking(HostMemBackingKind kind, size_t commitUnit);

//! Convenience overload using selectHostMemBackingKind().
[[nodiscard]] std::unique_ptr<IHostMemBacking> createHostMemBacking(size_t commitUnit);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
