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
#include "kv_cache_manager_v2/utils/hostMemBacking.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <thread>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// HostMem — the host tier's memory, filled progressively.
//
// The address range is reserved up front, so address() is valid immediately and
// never moves. Physical memory is committed chunk by chunk on a worker thread,
// and waitUsable() blocks until a given offset is backed.
//
// Address space that has not been committed yet is unreachable, so a reader
// that runs ahead of the watermark faults rather than reading undefined bytes.
// ---------------------------------------------------------------------------
class HostMem
{
public:
    //! What every size passed to HostMem must be a multiple of. Only the single-size
    //! constructor rounds up on the caller's behalf; elsewhere a size that is not a
    //! multiple of this is a bug and is rejected.
    //!
    //! A granularity of sizes and offsets, not an alignment of addresses: the base
    //! comes from mmap and is only page-aligned.
    [[nodiscard]] static size_t sizeGranularity() noexcept
    {
        return hostMemSizeGranularity();
    }

    //! Reserves `vmSize` of address space and commits `initialSize` of it. Both must be
    //! multiples of sizeGranularity(); neither is rounded on the caller's behalf, since
    //! a reservation quietly made larger is one the caller does not know it can resize
    //! into, and resize() rejects the same way.
    //!
    //! The base address is fixed for the lifetime of the object: resize() only
    //! moves the committed boundary within the reservation, and a resize past
    //! `vmSize` is rejected rather than served by relocating. Reserving costs
    //! address space, not memory.
    //!
    //! The fill always runs on a worker thread. `waitForFill` selects whether
    //! the caller is exposed to that: when true, the constructor and resize()
    //! return only once the whole range is committed, which is what an owner
    //! that hands out raw pointers without any further bookkeeping needs.
    //!
    //! Pass false only when the range is covered by a waitUsable() somewhere
    //! above. One wait where the memory is handed out is what that takes, not a
    //! wait per read: once a byte is backed it stays backed, so an owner that
    //! waits before publishing a sub-range has already covered every later read
    //! of it. Reaching past the watermark faults, so a range that nothing waits
    //! for fails loudly rather than reading zeros.
    HostMem(size_t vmSize, size_t initialSize, HostMemBackingOptions const& options = {}, bool waitForFill = true);

    //! Reserves exactly as much as it commits, for an allocation that never
    //! resizes. Takes no waitForFill: a caller that cannot resize has no reason
    //! to observe a partially filled range.
    //!
    //! `size` may be any byte count: it is rounded up to sizeGranularity(), and size()
    //! reports the rounded value.
    explicit HostMem(size_t size)
        : HostMem(roundUp(size, sizeGranularity()), roundUp(size, sizeGranularity()), HostMemBackingOptions{},
            /*waitForFill=*/true)
    {
    }

    ~HostMem();

    HostMem(HostMem const&) = delete;
    HostMem& operator=(HostMem const&) = delete;

    //! Blocks until [0, endOffset) is committed and usable.
    //!
    //! Rethrows the worker's exception if the fill failed; callers that touch
    //! the range without calling this first may read memory that is not yet
    //! backed.
    void waitUsable(size_t endOffset) const;

    //! Commits or releases the difference between the current size and newSize.
    //! The base address does not change. Waits for any in-flight fill first, so
    //! the caller must guarantee no concurrent access to the range.
    //!
    //! Throws if newSize exceeds the reservation.
    void resize(size_t newSize);

    //! Bytes of address space reserved, i.e. the largest size resize() accepts.
    [[nodiscard]] size_t vmSize() const noexcept
    {
        return mVmSize;
    }

    //! Releases everything. Safe to call multiple times.
    void destroy();

    [[nodiscard]] MemAddress address() const noexcept
    {
        return mAddr;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return mSize;
    }

private:
    //! Sentinel stored in the watermark when the worker failed. Distinct from
    //! every reachable byte count, so a waiter cannot mistake it for progress.
    static constexpr size_t kFillFailed = SIZE_MAX;

    void fillToTarget(int helperCount);
    void startFill(int helperCount);
    void joinFill() noexcept;

    HostMemBacking mBacking;
    MemAddress mAddr = 0;
    size_t mSize = 0;
    size_t mVmSize = 0;
    bool mWaitForFill = true;

    //! Bytes a consumer may touch, from the base. Read on every allocation,
    //! written once per fill step. Kept on its own cache line so publishing
    //! progress does not invalidate the line that readers are spinning on.
    alignas(64) mutable std::atomic<size_t> mUsable{0};

    //! Written before the failure sentinel is stored, and only read after that
    //! sentinel has been observed, so no lock is needed.
    std::exception_ptr mFillFailure;
    std::thread mFillWorker;

    //! Captured on the constructing thread. A new thread defaults to device 0
    //! regardless of what the creator selected, so the fill worker has to be
    //! told which GPU this allocation belongs to.
    int mDevice = 0;
};

// ---------------------------------------------------------------------------
// Low-level wrappers used internally (also exposed for storage pool use).
// ---------------------------------------------------------------------------
//! Maps `size` bytes of address space, page-aligned, which is all mmap
//! promises and all any caller needs. Throws HostOOMError.
MemAddress hostMmap(size_t size);
void hostMunmap(MemAddress ptr, size_t size) noexcept;
void resizeFile(int fd, size_t newSize); // throws DiskOOMError

using HostMadviseFn = int (*)(void*, size_t, int);
using HostMemsetFn = void* (*) (void*, int, size_t);

bool hostUseThp();

//! Total host memory the OS reports, in bytes. Used as the default reservation
//! bound, since no host allocation can exceed it.
size_t hostTotalMemory();
void hostMadvisePageMode(MemAddress ptr, size_t size, bool useThp, HostMadviseFn madviseFn = nullptr) noexcept;
void hostPrefaultChunk(MemAddress ptr, size_t size, HostMadviseFn madviseFn = nullptr, HostMemsetFn memsetFn = nullptr);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
