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
#include <stack>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! What every size a caller asks of the host tier must be a multiple of.
//!
//! A granularity, not an address alignment: the reservation's base comes from mmap and is
//! only page-aligned, so this constrains sizes and offsets within it rather than addresses.
//!
//! One value for the process, settled from the page size on first use.
[[nodiscard]] size_t hostMemSizeGranularity() noexcept;

struct HostMemBackingOptions
{
    //! What every registration bound is a multiple of, so no registration ever
    //! splits a range of this size. The registered prefix therefore trails the
    //! committed one by up to this much.
    //!
    //! A caller sets it to the largest range it hands to one cudaMemcpy, which
    //! the driver rejects if it straddles two registrations. 1 imposes no
    //! constraint, which is what a caller that never copies wants.
    size_t registerUnit = 1;

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

//! Physical backing for the host tier: an address range reserved on
//! construction, whose committed prefix then moves towards a target a chunk at
//! a time.
//!
//! The base address is fixed for the lifetime of the object. Growing and
//! shrinking touch only the delta, so a resize costs O(delta) rather than
//! O(total) and never invalidates a pointer the allocator above has handed out.
//!
//! Pages are populated through the kernel's fault path (MADV_POPULATE_WRITE)
//! and then handed to the driver with cuMemHostRegister. Allocating and zeroing
//! in the kernel rather than inside cuMemCreate is what lets several processes
//! fill at once: the driver serializes that work on a per-process lock, which on
//! an 8-GPU node collapsed per-process fill throughput by roughly 5x and blocked
//! unrelated CUDA calls on the same process for seconds at a time.
//!
//! Registration happens in chunks rather than over the whole range so that a
//! shrink can unregister close to exactly what it releases; a registration can
//! only be dropped whole. Each chunk stays far below the 2 GiB-per-call pinning
//! limit on Linux 6.11.
//!
//! A step stops on a chunk boundary and leaves the remainder to its successor,
//! so the range carries one registration per chunk rather than one per step.
class HostMemBacking
{
public:
    //! Reserves `vmSize` bytes of address space, page-aligned as mmap returns it
    //! and no coarser, and takes no memory for it. No page is resident and none
    //! is reachable: the whole range is PROT_NONE until a step opens it, so a
    //! read that runs ahead of the fill faults instead of returning zeros.
    //!
    //! `vmSize` must be a multiple of hostMemSizeGranularity(). Zero reserves
    //! nothing, for an owner that has no range to back.
    //!
    //! Needs a current CUDA context, which is captured here: registration needs
    //! one and the fill worker only sets the device, which does not give it one.
    explicit HostMemBacking(size_t vmSize, HostMemBackingOptions const& options = {});
    ~HostMemBacking();

    HostMemBacking(HostMemBacking const&) = delete;
    HostMemBacking& operator=(HostMemBacking const&) = delete;
    HostMemBacking(HostMemBacking&&) = delete;
    HostMemBacking& operator=(HostMemBacking&&) = delete;

    //! Base of the reservation, fixed for the lifetime of the object. Zero once
    //! destroy() has run, and for a backing reserving nothing.
    [[nodiscard]] MemAddress address() const noexcept
    {
        return mAddr;
    }

    //! Bytes of address space held, i.e. the largest target this accepts.
    [[nodiscard]] size_t reservedSize() const noexcept
    {
        return mSize;
    }

    //! The first two steps a commit takes: open() makes the range reachable and
    //! populate() allocates its pages. Neither unwinds on failure; decommitFrom()
    //! releases what they took.
    //!
    //! open() changes the mapping, which waits for every populate() in flight
    //! on any thread: open a range once, then populate it in parallel. Both
    //! are safe to call while another thread commits a disjoint range.
    //!
    //! Together they let a caller populate a step ahead of resizeStep(), which it
    //! then passes `populated` so the step only registers what is already there.
    void open(size_t offset, size_t size);
    void populate(size_t offset, size_t size) const;

    //! Undoes open() and populate() over a range that was never committed, which
    //! is what a fill that populated ahead of its commits leaves behind when it
    //! fails. Only valid at or above committedSize(); committed bytes are released
    //! by lowering the target instead.
    void discardUncommitted(size_t offset, size_t size) noexcept;

    //! Declares how much of the reservation should end up committed. Must be a
    //! multiple of commitGranularity() and no larger than the reservation.
    //!
    //! Takes no memory by itself; resizeStep() does the work. The previous target
    //! must have been reached: the caller finishes or abandons an in-flight fill first.
    void setTargetSize(size_t size);

    //! Moves committedSize() one chunk towards the target and returns
    //! registeredSize().
    //!
    //! A no-op once committedSize() == targetSize(), which is how a loop over it
    //! terminates.
    //!
    //! Throws HostOOMError when the system cannot back the chunk; that is the
    //! expected failure and callers are required to handle it. The step then
    //! takes nothing: committedSize() is left where it was found.
    //!
    //! Pass `populated` when the range up to nextStepEnd() has already been
    //! opened and populated, which lets the step skip straight to registering it.
    size_t resizeStep(bool populated = false);

    //! Where the next growing step will stop, so a caller can populate that range
    //! before asking for it. Equals committedSize() when there is no step to take.
    [[nodiscard]] size_t nextStepEnd() const noexcept;

    //! Bytes a growing step advances by. Steps land on multiples of it, so a caller
    //! populating ahead can derive every step boundary without taking one.
    [[nodiscard]] static size_t commitChunkSize() noexcept;

    //! Sets the target and steps until it is reached, for a caller with nothing
    //! to do in between.
    void resizeTo(size_t size);

    //! Releases everything, including the reservation. Idempotent.
    void destroy() noexcept;

    //! The granularity of every size this object is asked for.
    [[nodiscard]] size_t commitGranularity() const noexcept
    {
        return mUnit;
    }

    [[nodiscard]] size_t targetSize() const noexcept
    {
        return mTarget;
    }

    //! Bytes resident and reachable, from the base. Moves one chunk per step.
    [[nodiscard]] size_t committedSize() const noexcept
    {
        return mCommitted;
    }

    //! Bytes the GPU may be copied to or from, from the base. Never exceeds
    //! committedSize(), and trails it by less than one register unit once the
    //! target is reached.
    [[nodiscard]] size_t registeredSize() const noexcept
    {
        return mRegisteredEnd;
    }

    //! What registration bounds are multiples of, as the options asked, or 1 when
    //! they imposed no constraint.
    [[nodiscard]] size_t registerUnit() const noexcept
    {
        return mRegUnit;
    }

private:
    //! Populate up to `end`, which must be at or above mCommitted.
    void commitUpTo(size_t end, bool populated = false);

    //! Release from `end`, which must be at or below mCommitted.
    void decommitFrom(size_t end) noexcept;

    //! Extend registration towards the last whole register unit at or below `end`.
    //! Unless `allowPartialChunk`, stops at the last whole chunk boundary so the
    //! remainder joins the next chunk rather than becoming a registration of its
    //! own.
    void registerUpTo(size_t end, bool allowPartialChunk);

    //! Drop registration above the last whole register unit at or below `end`.
    void unregisterFrom(size_t end) noexcept;

    //! Remember `start` if it is not where arithmetic would look for it.
    void recordRegistrationStart(size_t start);

    //! Where the registration ending at mRegisteredEnd began. Requires one to exist.
    [[nodiscard]] size_t innermostRegistrationStart() const noexcept;

    //! Unregister that one and pull mRegisteredEnd back to its start, which is
    //! returned.
    size_t dropInnermostRegistration() noexcept;

    //! cuMemHostUnregister takes only a base address; the size is the driver's.
    void unregisterAt(size_t offset) noexcept;

    size_t mUnit = 0;
    size_t mRegUnit = 1;
    //! Bytes per registration: the most whole register units that fit in the target
    //! chunk size, or one register unit when that is the larger.
    size_t mRegChunk = 0;
    bool mAllowRemoteNumaFallback = true;
    MemAddress mAddr = 0;

    size_t mSize = 0;

    size_t mTarget = 0;
    size_t mCommitted = 0;

    //! Bytes actually mapped: mSize rounded up to a page.
    size_t mMapped = 0;

    //! Innermost last. Registrations are contiguous and grow and shrink only at
    //! the end, so a stack is the whole structure.
    //!
    //! Only the unaligned ones, because the rest are arithmetic: a step registers
    //! whole chunks from a boundary, so the start of the registration covering a
    //! byte is normally the chunk boundary below it. One case breaks that -- a
    //! fill resumed at a higher target begins where the previous target's partial
    //! tail ended -- and those are what this holds, one per resumed fill rather
    //! than one per chunk of the tier.
    std::stack<size_t> mUnalignedRegStarts;

    //! End of the registered prefix, always a whole number of register units.
    size_t mRegisteredEnd = 0;

    //! The context cuMemHostRegister runs under; the fill worker only sets the device.
    void* mContext = nullptr;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
