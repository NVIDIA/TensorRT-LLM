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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

namespace tensorrt_llm::runtime
{

#if ENABLE_MULTI_DEVICE

struct NcclUniqueIdRendezvousTags
{
    mpi::MpiTag ready;
    mpi::MpiTag id;
    mpi::MpiTag ack;
};

//! An MPI control channel dedicated to NCCL unique-ID rendezvous.
//!
//! The channel is created collectively by the initial NCCL group while every
//! member is healthy. It retains a process-lifetime MPI communicator whose
//! error handler is MPI_ERRORS_RETURN, leaving the parent/session communicator unchanged. The
//! original parent-communicator ranks are retained so a later survivor subset
//! can keep using stable world-rank IDs even though MPI ranks in this channel
//! are compact.
//!
//! This isolates rendezvous traffic and error handling, but it is not a ULFM
//! communicator repair. Post-failure point-to-point progress still requires an
//! MPI implementation and launcher configured to let survivors continue.
class NcclUniqueIdRendezvousComm
{
public:
    NcclUniqueIdRendezvousComm(mpi::MpiComm comm, std::vector<int> worldRanks, int worldRank);

    NcclUniqueIdRendezvousComm(NcclUniqueIdRendezvousComm const&) = delete;
    NcclUniqueIdRendezvousComm& operator=(NcclUniqueIdRendezvousComm const&) = delete;

    [[nodiscard]] mpi::MpiComm const& mpiComm() const noexcept;
    [[nodiscard]] std::vector<int> const& worldRanks() const noexcept;
    [[nodiscard]] int worldRank() const noexcept;
    [[nodiscard]] int commRank(int worldRank) const;
    [[nodiscard]] int worldRank(int commRank) const;

private:
    mpi::MpiComm mComm;
    std::vector<int> mWorldRanks;
    int mWorldRank;
};

//! Create a dedicated control channel over initialRanks.
//!
//! Every member of initialRanks must call this before any process failure.
//! Non-members must not call. creationTagSeed and the canonical group derive a
//! bounded, deterministic MPI creation tag so concurrently-created overlapping
//! groups do not normally cross-pair.
std::shared_ptr<NcclUniqueIdRendezvousComm> createNcclUniqueIdRendezvousComm(
    std::vector<int> const& initialRanks, int worldRank, mpi::MpiComm const& parentComm, int creationTagSeed);

// Exchange a fresh NCCL unique ID among only activeRanks. READY/ID/ACK tokens
// prevent delayed eager MPI messages from an earlier timed-out attempt from
// pairing different IDs across ranks. rendezvousId is a coordinator-provided
// logical attempt identity: all ranks in one attempt must pass the same value,
// and retries for the same logical communicator must use strictly increasing
// values. The protocol discards older same-communicator messages after seeing
// a later value while retaining messages for future values and other groups.
// All ranks use original/global rank IDs.
// Callers must keep one logical communicator per (control channel, tag set,
// active-rank set), or construct multiple instances in the same order on every
// rank. Recovery reuses the pre-failure control channel and performs no MPI
// collective.
ncclUniqueId exchangeNcclUniqueId(std::vector<int> const& activeRanks, NcclUniqueIdRendezvousComm const& controlComm,
    NcclUniqueIdRendezvousTags tags, std::uint64_t rendezvousId, std::chrono::steady_clock::time_point deadline);

#endif // ENABLE_MULTI_DEVICE

} // namespace tensorrt_llm::runtime
