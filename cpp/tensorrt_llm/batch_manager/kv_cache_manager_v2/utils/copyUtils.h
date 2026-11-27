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

#include <cstdint>
#include <cuda.h>
#include <memory>
#include <sys/types.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

template <typename DstAddr, typename SrcAddr>
struct Task
{
    DstAddr dst;
    SrcAddr src;
};

//! Process-wide: run host <-> disk copies on a pool of \p numThreads threads. The first nonzero call sizes it.
void configureAsyncDiskIo(size_t numThreads) noexcept;

//! Process-wide: spread the page tasks of one host-side copy callback (host <-> host, disk <-> disk, and host <-> disk
//! while no async disk I/O pool runs) over \p numThreads threads, the callback thread included. The callback still
//! returns only once every task has finished. The first call with a value > 1 sets it for the life of the process;
//! values <= 1 change nothing. Without such a call every task runs on the callback thread, as before.
void configureHostCopyThreads(size_t numThreads) noexcept;

//! Threads a host-side copy callback may use now, the callback thread included.
[[nodiscard]] size_t hostCopyThreads() noexcept;

//! Testing only: set the host copy width to \p numThreads (1: one thread) whatever was configured. Returns the width
//! set, which is smaller when not every worker started.
size_t setHostCopyThreadsForTesting(size_t numThreads) noexcept;

//! Testing only: page tasks of host-side copy callbacks run by pool workers so far.
[[nodiscard]] uint64_t hostCopyHelperTasksForTesting() noexcept;

//! Process-wide: up to \p bytes of host memory for write-behind (0: off). With the async I/O pool running, a
//! host-to-disk page write that fits completes once the page is copied there, and a pool worker writes the file
//! afterwards; reads of the page are served from staging until then. Applies to writes issued from now on.
void configureDiskWriteBehind(size_t bytes) noexcept;

//! Call before closing (\p minPos 0) or shrinking a disk pool file, after waitForDiskIo() when closing: drops the
//! file's pages left in write-behind staging at offsets >= \p minPos (a failed flush keeps its page there).
void discardDiskWriteBehind(int fd, ssize_t minPos) noexcept;

//! Testing only: pages in write-behind staging (pending a flush, or kept after a failed one).
[[nodiscard]] size_t diskWriteBehindPagesForTesting() noexcept;

//! Testing only: a write-behind flush starts only once it has been queued for \p seconds (0: no delay), so reads are
//! served from staging meanwhile.
void delayDiskWriteBehindFlushesForTesting(double seconds) noexcept;

//! Whether the last write to the disk page at (\p fd, \p pos) failed; reports each failure once. A disk copy made by
//! a failed write holds stale bytes and must not be reused.
[[nodiscard]] bool consumeFailedDiskWrite(int fd, ssize_t pos) noexcept;

//! Testing only: make the next \p count disk write pieces fail without writing.
void injectDiskWriteFailuresForTesting(uint64_t count) noexcept;

//! Testing only: the async I/O pool starts a background write piece only once it has been queued for \p seconds (0:
//! no delay). Promoted pieces are required writes and are not delayed.
void delayBackgroundDiskWritesForTesting(double seconds) noexcept;

//! Whether host <-> disk copies run on the async I/O pool (configureAsyncDiskIo started it).
[[nodiscard]] bool asyncDiskIoEnabled() noexcept;

//! How the async I/O pool shares its workers between reads and write pieces (required writes, write-behind flushes,
//! write-through copies). All zero without the pool.
struct DiskIoWriteCapStats
{
    size_t threads = 0;                 //!< workers of the pool
    size_t maxWriters = 0;              //!< workers that may run write pieces at once (0: no cap)
    size_t peakWriters = 0;             //!< most workers seen running write pieces at once
    uint64_t readsPastCappedWrites = 0; //!< read pieces started while a write waited at the cap
    uint64_t expiredWriteHolds = 0;     //!< write pieces holdDiskWritePiecesForTesting released at its deadline
};

//! Testing only: the pool's write cap and its counters; \p reset zeroes the counters after reading them.
[[nodiscard]] DiskIoWriteCapStats diskIoWriteCapStatsForTesting(bool reset) noexcept;

//! Testing only: let at most \p maxWriters workers run write pieces at once (0: no cap). Returns the previous cap
//! (0: none), or 0 without the pool.
size_t setDiskIoMaxWriteThreadsForTesting(size_t maxWriters) noexcept;

//! Testing only: for the next \p seconds, a worker that starts a write piece waits before writing, as a buffered
//! write blocked in the kernel (the file's inode lock, dirty-page throttling) would; the deadline releases them. 0
//! releases them now. The deadline frees a test that would otherwise wait for them forever, without a Python thread.
void holdDiskWritePiecesForTesting(double seconds) noexcept;

//! Background writes issued in one BackgroundDiskWriteScope on the async I/O pool. It does not keep them alive: once a
//! write has finished, it is gone from here.
struct BackgroundDiskWrites;

//! While an instance is alive, host-to-disk copies issued by this thread are background writes. The I/O pool serves
//! them after reads and required writes. Write-through copies use this: they only save a later write.
class BackgroundDiskWriteScope
{
public:
    BackgroundDiskWriteScope() noexcept;
    ~BackgroundDiskWriteScope();
    BackgroundDiskWriteScope(BackgroundDiskWriteScope const&) = delete;
    BackgroundDiskWriteScope& operator=(BackgroundDiskWriteScope const&) = delete;

    //! The background writes issued so far in this scope, or null if there are none (no async I/O pool).
    [[nodiscard]] std::shared_ptr<BackgroundDiskWrites const> takeWrites();

private:
    BackgroundDiskWrites* mPrevious;
    std::unique_ptr<BackgroundDiskWrites> mWrites;
};

//! A required write now depends on \p writes (an eviction takes their disk slots back), so serve their pieces not
//! yet started as required writes instead of background ones.
void promoteBackgroundDiskWrites(BackgroundDiskWrites const& writes) noexcept;

//! Bytes of background writes that may still be issued: 0 without the async I/O pool, otherwise what is left under
//! a process-wide cap on background bytes issued and not yet written.
[[nodiscard]] size_t backgroundDiskWriteBudget() noexcept;

//! Call before closing a disk pool file: waits until the async I/O pool has no queued or running pieces, since a
//! piece for the file may still use its descriptor (and a closed descriptor number can be reused for another file).
//! The pool is process-wide, so this also waits for other managers' pieces.
void waitForDiskIo() noexcept;

CUresult copyDiskToDisk(std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDiskToHost(std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToDisk(std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToHost(std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToDevice(std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream);
CUresult copyDeviceToHost(std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream);
CUresult copyDeviceToDevice(std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
