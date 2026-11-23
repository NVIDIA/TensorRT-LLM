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

// The memory behind a remote cache level: host memory that donors (remote_donor.py, which describes the files)
// publish in a rendezvous directory, claimed in whole chunks with O_EXCL claim.<donor>.<chunk> files. Declaring and
// claiming it does not pick a transport: pages move through the client's RemoteTransferBackend (remoteTransfer.h),
// which makeRemoteMemClient makes from the tier's RemoteTransferConfig.

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/remoteTransfer.h"
#include "kv_cache_manager_v2/utils/sharedPtr.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! The memory a remote level declares: where donors publish it and how long to wait for them.
struct RemoteMemOptions
{
    //! Rendezvous directory the donors publish into.
    std::string directory;
    //! How long a claim waits for a donor with a free chunk (RemoteCacheTierConfig::waitTimeoutMs).
    int64_t waitTimeoutMs = 0;
    //! Donors the job starts (RemoteCacheTierConfig::expectedDonors); 0 if unknown.
    uint32_t expectedDonors = 0;

    bool operator==(RemoteMemOptions const&) const = default;
};

//! A file descriptor, closed with its owner.
class OwnedFd
{
public:
    explicit OwnedFd(FileDescriptor fd = kBadFileDescriptor) noexcept
        : mFd(fd)
    {
    }

    ~OwnedFd();
    OwnedFd(OwnedFd&& other) noexcept;
    OwnedFd(OwnedFd const&) = delete;
    OwnedFd& operator=(OwnedFd const&) = delete;
    OwnedFd& operator=(OwnedFd&&) = delete;

    [[nodiscard]] FileDescriptor get() const noexcept
    {
        return mFd;
    }

private:
    FileDescriptor mFd;
};

//! One chunk of donated memory this process owns.
struct RemoteChunk
{
    uint32_t donor = 0;      // index into the client's donor table
    uint64_t remoteAddr = 0; // the address transfers name the chunk by (RemoteTransferBackend::attachChunk)
    uint64_t bytes = 0;
};

//! A connection to the donors of one rendezvous directory: the donor table and the claimed chunks, and the backend
//! that moves pages. Owned by a StorageManager and shared by its levels that name the directory; the last one to go
//! releases the claims.
class RemoteMemClient
{
public:
    //! Reads the donors published so far. `name` is unique among the process's clients and names each claim's owner.
    //! `directory` is options.directory, opened and checked (openRendezvousDirectory): every later access goes through
    //! it, so that the path is never resolved again.
    RemoteMemClient(std::string name, RemoteMemOptions options, OwnedFd directory, RemoteTransferConfig transferConfig,
        std::unique_ptr<RemoteTransferBackend> backend);
    ~RemoteMemClient();

    RemoteMemClient(RemoteMemClient const&) = delete;
    RemoteMemClient& operator=(RemoteMemClient const&) = delete;

    //! A chunk that holds at least one slot of `minBytes`: from this process's free list, else claimed from a donor,
    //! waiting up to waitTimeoutMs for one, or until all expectedDonors have reported. Throws RemoteMemError when
    //! none appears.
    [[nodiscard]] RemoteChunk acquireChunk(size_t minBytes);
    //! Return a chunk to this process's free list. The claim is kept until the client is destroyed.
    void releaseChunk(RemoteChunk chunk) noexcept;

    //! RemoteTransferBackend::streamOrdered().
    [[nodiscard]] bool streamOrdered() const noexcept
    {
        return mStreamOrdered;
    }

    //! Register local memory transfers read from or write into with the backend (reference-counted by base).
    void registerLocal(MemAddress base, size_t bytes);
    //! Ignored once the client is abandoned: a transfer may still access the memory.
    void deregisterLocal(MemAddress base, size_t bytes) noexcept;

    //! Start a transfer (RemoteTransferBackend::submit). Throws RemoteMemError when an earlier one failed: the
    //! contents of the remote slots can no longer be trusted.
    [[nodiscard]] std::unique_ptr<RemoteTransferAttempt> submit(
        bool write, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream);

    //! Record the first transfer failure and throw it as RemoteMemError; the client has failed() from then on.
    [[noreturn]] void fail(std::string const& message);

    //! Throws RemoteMemError when an earlier transfer failed. Call it before leasing page staging for a transfer: a
    //! failed level's unquiesced transfers may still hold all of it.
    void checkUsable() const;

    //! Whether a transfer failed. A failed client throws RemoteMemError on every transfer and claim.
    [[nodiscard]] bool failed() const noexcept
    {
        return mFailed.load(std::memory_order_acquire);
    }

    //! Wait until every stream-ordered transfer enqueued so far has finished (teardown).
    void drain() noexcept;

    //! A transfer that never quiesced may still access the claimed chunks, the backend's registrations and its
    //! agent: keep the claims and leak the backend instead of releasing them.
    void abandon() noexcept;

    [[nodiscard]] std::string const& directory() const noexcept
    {
        return mOptions.directory;
    }

    [[nodiscard]] RemoteMemOptions const& options() const noexcept
    {
        return mOptions;
    }

    [[nodiscard]] RemoteTransferConfig const& transferConfig() const noexcept
    {
        return mTransferConfig;
    }

private:
    //! Add the donors newly published in the directory, mark the ones whose .ready file is gone as withdrawn, and
    //! count the donors that reported (.ready or .failed).
    void refreshDonors();
    //! Whether `donor` still publishes its .ready file; marks it withdrawn when not.
    bool stillPublished(RemoteDonor& donor);
    //! Claim a free chunk; a donor whose chunk cannot be attached (e.g. it crashed) is marked withdrawn.
    bool tryClaim(size_t minBytes, RemoteChunk& out);
    //! Caller holds mMutex. Throws RemoteMemError when an earlier transfer failed.
    void throwIfFailed() const;

    //! The contents of the donor file `file` in the directory, or nullopt when it cannot be read or is not a regular
    //! file of this user's: the donor withdrew after the listing, or the file is not trusted.
    [[nodiscard]] std::optional<std::string> readDonorFile(std::string const& file) const;

    std::string mName;
    RemoteMemOptions mOptions;
    OwnedFd mDirectory;
    RemoteTransferConfig mTransferConfig;
    std::unique_ptr<RemoteTransferBackend> mBackend;
    bool mStreamOrdered = false;
    std::vector<RemoteDonor> mDonors;
    mutable std::mutex mMutex;
    std::unordered_map<std::string, size_t> mDonorIndex;
    std::vector<RemoteChunk> mFreeChunks;
    std::vector<std::string> mClaims; // claim file names in the directory
    uint64_t mClaimedBytes = 0;
    size_t mReportedDonors = 0;       // donors of this transport with a .ready or .failed file at the last refresh
    std::unordered_set<std::string> mForeignDonors; // ready donors of another transport, passed over
    std::string mLastClaimError;                    // why the last donor was withdrawn at claim time
    bool mAbandoned = false;
    std::string mFailure;                           // the first transfer failure
    std::atomic<bool> mFailed{false};
};

//! Open `path` as a rendezvous directory: a directory, not a symlink, owned by this user and not writable by group or
//! others, as remote_donor.ensure_rendezvous_directory makes it. Throws otherwise.
[[nodiscard]] OwnedFd openRendezvousDirectory(std::string const& path);

//! The client of `options.directory`, moving pages through the backend `transferConfig` selects.
[[nodiscard]] SharedPtr<RemoteMemClient> makeRemoteMemClient(
    RemoteMemOptions options, RemoteTransferConfig const& transferConfig);

//! A pool's slots in remote memory: whole chunks, each holding floor(chunk bytes / slot size) slots, in order.
class RemoteRange
{
public:
    RemoteRange(SharedPtr<RemoteMemClient> client, size_t slotSize);
    ~RemoteRange();

    RemoteRange(RemoteRange const&) = delete;
    RemoteRange& operator=(RemoteRange const&) = delete;

    //! Hold enough chunks for `numSlots` slots. Growing claims chunks; shrinking keeps them (a later grow reuses them).
    void reserve(SlotCount numSlots);

    //! Where slot `slot` lives: (donor index, remote address).
    [[nodiscard]] std::pair<uint32_t, uint64_t> locate(SlotId slot) const;

    [[nodiscard]] RemoteMemClient& client() const noexcept
    {
        return *mClient;
    }

private:
    SharedPtr<RemoteMemClient> mClient;
    size_t mSlotSize = 0;
    std::vector<RemoteChunk> mChunks;
    std::vector<SlotCount> mFirstSlot; // first slot of each chunk; mFirstSlot.size() == mChunks.size()
    SlotCount mCapacity{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
