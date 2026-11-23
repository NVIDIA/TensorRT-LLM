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
