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
// Which physical backing is used depends on the platform; see IHostMemBacking.
// Neither backing calls cuMemHostRegister.
// ---------------------------------------------------------------------------
class HostMem
{
public:
    //! Required alignment of every size passed to HostMem, and of the base
    //! address it returns. Set by the coarsest backing: cuMemCreate rejects a
    //! host allocation whose size is not a multiple of its granularity, which is
    //! 2 MiB. Sizes are not rounded up on the caller's behalf -- a request that
    //! is not a multiple of this is a bug and is rejected.
    static constexpr size_t kAlignment = 2ULL << 20; // 2 MB

    //! Bytes committed per watermark advance. Large enough that the commit cost
    //! dominates the notify, small enough that a waiter near the start of the
    //! range is released early.
    //!
    //! Unrelated to the unit memory is allocated and released in: this only sets
    //! how often the fill publishes progress. See IHostMemBacking.
    static constexpr size_t kFillChunkSize = 512ULL << 20; // 512 MB

    //! The fill always runs on a worker thread. `waitForFill` selects whether
    //! the caller is exposed to that: when true, the constructor and resize()
    //! return only once the whole range is committed, which is what an owner
    //! that hands out raw pointers without any further bookkeeping needs.
    //!
    //! Pass false only when every reader of this range calls waitUsable() with
    //! the offset it is about to touch. On the VMM backing, address space that
    //! has not been committed yet is unmapped, so reaching past the watermark
    //! faults rather than merely reading undefined bytes.
    //! Reserves `maxSize` of address space and commits `initialSize` of it.
    //!
    //! The base address is fixed for the lifetime of the object: resize() only
    //! moves the committed boundary within the reservation, and a resize past
    //! `maxSize` is rejected rather than served by relocating. Reserving costs
    //! address space, not memory.
    //! `commitUnit` is the granularity memory is allocated and released in, and
    //! therefore the alignment every size must satisfy. It is independent of
    //! kFillChunkSize, which only paces progress reporting.
    HostMem(size_t maxSize, size_t initialSize, size_t commitUnit = kAlignment, bool waitForFill = true);

    //! Reserves exactly as much as it commits, for an allocation that never
    //! resizes. Takes no waitForFill: a caller that cannot resize has no reason
    //! to observe a partially filled range.
    explicit HostMem(size_t size)
        : HostMem(size, size, kAlignment, /*waitForFill=*/true)
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
    [[nodiscard]] size_t maxSize() const noexcept
    {
        return mMaxSize;
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
    //! every real chunk count, so a waiter cannot mistake it for progress.
    static constexpr uint32_t kFillFailed = UINT32_MAX;

    void commitRange(size_t beginOffset);
    void startFill(size_t beginOffset);
    void joinFill() noexcept;
    [[nodiscard]] size_t chunkCount() const noexcept;

    std::unique_ptr<IHostMemBacking> mBacking;
    MemAddress mAddr = 0;
    size_t mSize = 0;
    size_t mMaxSize = 0;

    //! Bytes committed between watermark advances. A multiple of the backing's
    //! commit granularity, so a fill step never lands inside an allocation.
    size_t mChunkSize = 0;
    bool mWaitForFill = true;

    //! Read on every allocation, written once per committed chunk. Kept on its
    //! own cache line so publishing progress does not invalidate the line that
    //! readers are spinning on.
    alignas(64) mutable std::atomic<uint32_t> mFilledChunks{0};

    //! Written before the failure sentinel is stored, and only read after that
    //! sentinel has been observed, so no lock is needed.
    std::exception_ptr mFillFailure;
    std::thread mFillWorker;
};

// ---------------------------------------------------------------------------
// Low-level wrappers used internally (also exposed for storage pool use).
// ---------------------------------------------------------------------------
MemAddress hostMmap(size_t size);        // throws HostOOMError
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
