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

#include "tensorrt_llm/runtime/utils/ncclUniqueIdRendezvous.h"

#if ENABLE_MULTI_DEVICE

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/utils/ncclHostApi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tensorrt_llm::runtime
{
namespace
{

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kProtocolVersion = 2;
constexpr auto kPollInterval = std::chrono::milliseconds{1};
constexpr auto kReadyRefreshInterval = std::chrono::milliseconds{100};
constexpr std::size_t kMaxTokensPerPeer = 1024;
constexpr std::size_t kMaxPendingMessages = 4096;

struct AttemptToken
{
    std::uint64_t incarnation;
    std::uint64_t sequence;
};

bool operator==(AttemptToken const& lhs, AttemptToken const& rhs) noexcept
{
    return lhs.incarnation == rhs.incarnation && lhs.sequence == rhs.sequence;
}

struct ReadyMessage
{
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t communicatorKey;
    std::uint64_t rendezvousId;
    AttemptToken clientToken;
};

struct IdMessage
{
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t communicatorKey;
    std::uint64_t rendezvousId;
    AttemptToken clientToken;
    AttemptToken serverToken;
    std::uint64_t idDigest;
    ncclUniqueId id;
};

struct AckMessage
{
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t communicatorKey;
    std::uint64_t rendezvousId;
    AttemptToken clientToken;
    AttemptToken serverToken;
    std::uint64_t idDigest;
};

template <typename Message>
struct ReceivedMessage
{
    int source;
    Message message;
};

struct RendezvousContext
{
    std::mutex exchangeMutex;
    std::deque<ReceivedMessage<ReadyMessage>> pendingReady;
    std::deque<ReceivedMessage<IdMessage>> pendingIds;
    std::deque<ReceivedMessage<AckMessage>> pendingAcks;
};

struct ControlCommCacheEntry
{
    ControlCommCacheEntry(MPI_Group parentGroup_, std::vector<int> initialRanks_, int worldRank_, int creationTagSeed_)
        : parentGroup(parentGroup_)
        , initialRanks(std::move(initialRanks_))
        , worldRank(worldRank_)
        , creationTagSeed(creationTagSeed_)
    {
    }

    std::mutex mutex;
    MPI_Group parentGroup{MPI_GROUP_NULL};
    std::vector<int> initialRanks;
    int worldRank;
    int creationTagSeed;
    std::shared_ptr<NcclUniqueIdRendezvousComm> comm;
    std::exception_ptr failure;
};

std::uint64_t groupIdentity(std::vector<int> const& ranks) noexcept
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (int const rank : ranks)
    {
        auto const* bytes = reinterpret_cast<unsigned char const*>(&rank);
        for (std::size_t index = 0; index < sizeof(rank); ++index)
        {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

using ContextKey = std::tuple<MPI_Fint, std::uint64_t, int, int, int>;

ContextKey getContextKey(NcclUniqueIdRendezvousTags const& tags, NcclUniqueIdRendezvousComm const& controlComm) noexcept
{
    return {MPI_Comm_c2f(static_cast<MPI_Comm>(controlComm.mpiComm())), groupIdentity(controlComm.worldRanks()),
        static_cast<int>(tags.ready), static_cast<int>(tags.id), static_cast<int>(tags.ack)};
}

RendezvousContext& getRendezvousContext(
    NcclUniqueIdRendezvousTags const& tags, NcclUniqueIdRendezvousComm const& controlComm)
{
    // Keep contexts immortal for the same reason as the NCCL host gate: a
    // communicator may be torn down from another process-lifetime singleton.
    static auto* registryMutex = new std::mutex;
    static auto* contexts = new std::map<ContextKey, std::unique_ptr<RendezvousContext>>;
    std::lock_guard<std::mutex> lock(*registryMutex);
    auto& context = (*contexts)[getContextKey(tags, controlComm)];
    if (context == nullptr)
    {
        context = std::make_unique<RendezvousContext>();
    }
    return *context;
}

std::shared_ptr<ControlCommCacheEntry> getControlCommCacheEntry(
    std::vector<int> const& initialRanks, int worldRank, mpi::MpiComm const& parentComm, int creationTagSeed)
{
    // A control communicator cannot be freed safely after a member dies. Keep
    // one channel per topology for the process lifetime and reuse it across PP
    // engine teardown/recreation instead of leaking a new MPI context each
    // time. Retain the parent's MPI group as a stable process-membership
    // identity: MPI communicator handles may be recycled after a custom parent
    // is freed. Per-entry locking permits unrelated groups to create
    // concurrently.
    static auto* registryMutex = new std::mutex;
    static auto* entries = new std::vector<std::shared_ptr<ControlCommCacheEntry>>;

    MPI_Group parentGroup = MPI_GROUP_NULL;
    int result = MPI_Comm_group(static_cast<MPI_Comm>(parentComm), &parentGroup);
    TLLM_CHECK_WITH_INFO(
        result == MPI_SUCCESS, "NCCL error: failed to retain MPI parent group identity (code %d)", result);
    try
    {
        std::lock_guard<std::mutex> lock(*registryMutex);
        for (auto const& entry : *entries)
        {
            if (entry->initialRanks != initialRanks || entry->worldRank != worldRank
                || entry->creationTagSeed != creationTagSeed)
            {
                continue;
            }
            int comparison = MPI_UNEQUAL;
            result = MPI_Group_compare(entry->parentGroup, parentGroup, &comparison);
            TLLM_CHECK_WITH_INFO(
                result == MPI_SUCCESS, "NCCL error: failed to compare MPI parent group identity (code %d)", result);
            if (comparison == MPI_IDENT)
            {
                MPI_Group_free(&parentGroup);
                return entry;
            }
        }

        auto entry = std::make_shared<ControlCommCacheEntry>(parentGroup, initialRanks, worldRank, creationTagSeed);
        entries->push_back(entry);
        parentGroup = MPI_GROUP_NULL;
        return entry;
    }
    catch (...)
    {
        if (parentGroup != MPI_GROUP_NULL)
        {
            MPI_Group_free(&parentGroup);
        }
        throw;
    }
}

template <typename Message>
bool takePending(std::deque<ReceivedMessage<Message>>& pending, int source, std::uint64_t key,
    std::uint64_t rendezvousId, ReceivedMessage<Message>& result)
{
    auto found = pending.begin();
    for (; found != pending.end(); ++found)
    {
        if ((source == MPI_ANY_SOURCE || found->source == source) && found->message.communicatorKey == key
            && found->message.rendezvousId == rendezvousId)
        {
            break;
        }
    }
    if (found == pending.end())
    {
        return false;
    }
    result = *found;
    pending.erase(found);
    return true;
}

template <typename Message>
void discardObsoletePending(
    std::deque<ReceivedMessage<Message>>& pending, std::uint64_t key, std::uint64_t rendezvousId)
{
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                      [key, rendezvousId](auto const& received)
                      {
                          return received.message.version == kProtocolVersion && received.message.communicatorKey == key
                              && received.message.rendezvousId < rendezvousId;
                      }),
        pending.end());
}

template <typename Message>
bool isCurrentAttempt(Message const& message, std::uint64_t key, std::uint64_t rendezvousId) noexcept
{
    return message.communicatorKey == key && message.rendezvousId == rendezvousId;
}

template <typename Message>
bool shouldStashForLater(Message const& message, std::uint64_t key, std::uint64_t rendezvousId) noexcept
{
    if (message.version != kProtocolVersion)
    {
        return false;
    }
    // Rendezvous IDs must increase for retries of one base communicator.
    // Messages from an older retry can never be consumed again, while a peer
    // may legitimately have entered a future retry or a different group first.
    return message.communicatorKey != key || message.rendezvousId > rendezvousId;
}

template <typename Message>
void stashPending(std::deque<ReceivedMessage<Message>>& pending, ReceivedMessage<Message> received)
{
    TLLM_CHECK_WITH_INFO(
        pending.size() < kMaxPendingMessages, "NCCL error: communicator rendezvous pending-message limit exceeded");
    pending.push_back(std::move(received));
}

std::uint64_t fnv1a(void const* data, std::size_t size, std::uint64_t hash = 14695981039346656037ULL) noexcept
{
    auto const* bytes = static_cast<unsigned char const*>(data);
    for (std::size_t index = 0; index < size; ++index)
    {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t communicatorKey(std::vector<int> const& activeRanks, NcclUniqueIdRendezvousTags const& tags) noexcept
{
    std::uint64_t hash = fnv1a(activeRanks.data(), activeRanks.size() * sizeof(activeRanks.front()));
    std::array<int, 3> const tagValues{
        static_cast<int>(tags.ready), static_cast<int>(tags.id), static_cast<int>(tags.ack)};
    return fnv1a(tagValues.data(), tagValues.size() * sizeof(tagValues.front()), hash);
}

std::uint64_t processIncarnation()
{
    static std::uint64_t const incarnation = []()
    {
        std::random_device random;
        std::array<std::uint32_t, 4> entropy{random(), random(), random(), random()};
        auto const wallClock = std::chrono::system_clock::now().time_since_epoch().count();
        std::uint64_t hash = fnv1a(entropy.data(), entropy.size() * sizeof(entropy.front()));
        hash = fnv1a(&wallClock, sizeof(wallClock), hash);
        return hash == 0 ? std::uint64_t{1} : hash;
    }();
    return incarnation;
}

AttemptToken makeAttemptToken()
{
    static std::atomic<std::uint64_t> sequence{1};
    auto const value = sequence.fetch_add(1, std::memory_order_relaxed);
    TLLM_CHECK_WITH_INFO(value != 0, "NCCL error: communicator rendezvous token sequence overflow");
    return AttemptToken{processIncarnation(), value};
}

int abandonRequest(MPI_Request& request) noexcept
{
    int firstError = MPI_SUCCESS;
    if (request != MPI_REQUEST_NULL)
    {
        firstError = MPI_Cancel(&request);
        // MPI_Wait can block forever after a peer failure, defeating the
        // recovery deadline. Free the local request handle and let MPI retire
        // it asynchronously; callers deliberately leak the tiny backing
        // payload so the implementation can no longer access freed storage.
        int const freeResult = MPI_Request_free(&request);
        if (firstError == MPI_SUCCESS)
        {
            firstError = freeResult;
        }
    }
    return firstError;
}

void waitForSend(MPI_Request& request, Clock::time_point deadline, char const* operation)
{
    while (true)
    {
        int complete = 0;
        int const result = MPI_Test(&request, &complete, MPI_STATUS_IGNORE);
        if (result != MPI_SUCCESS)
        {
            int const cleanupResult = abandonRequest(request);
            TLLM_THROW(
                "NCCL error: MPI_Test failed while %s (code %d, cleanup code %d)", operation, result, cleanupResult);
        }
        if (complete != 0)
        {
            return;
        }
        if (Clock::now() >= deadline)
        {
            int const cleanupResult = abandonRequest(request);
            TLLM_THROW("NCCL error: timed out while %s (MPI cleanup code %d)", operation, cleanupResult);
        }
        std::this_thread::sleep_for(kPollInterval);
    }
}

template <typename Message>
void sendMessage(Message const& message, int destination, mpi::MpiTag tag, mpi::MpiComm const& mpiComm,
    Clock::time_point deadline, char const* operation)
{
    auto payload = std::make_unique<Message>(message);
    MPI_Request request = MPI_REQUEST_NULL;
    int const result = MPI_Isend(payload.get(), static_cast<int>(sizeof(message)), MPI_BYTE, destination,
        static_cast<int>(tag), static_cast<MPI_Comm>(mpiComm), &request);
    if (result != MPI_SUCCESS)
    {
        int const cleanupResult = abandonRequest(request);
        static_cast<void>(payload.release());
        TLLM_THROW("NCCL error: MPI_Isend failed while %s to rank %d (code %d, cleanup code %d)", operation,
            destination, result, cleanupResult);
    }
    try
    {
        waitForSend(request, deadline, operation);
    }
    catch (...)
    {
        static_cast<void>(payload.release());
        throw;
    }
}

bool probeMessage(int source, mpi::MpiTag tag, mpi::MpiComm const& mpiComm, MPI_Status& status)
{
    int available = 0;
    int const result = MPI_Iprobe(source, static_cast<int>(tag), static_cast<MPI_Comm>(mpiComm), &available, &status);
    TLLM_CHECK_WITH_INFO(result == MPI_SUCCESS, "NCCL error: MPI_Iprobe failed with code %d", result);
    return available != 0;
}

template <typename Message>
Message receiveMessage(MPI_Status const& probedStatus, mpi::MpiTag tag, mpi::MpiComm const& mpiComm,
    Clock::time_point deadline, char const* operation)
{
    auto message = std::make_unique<Message>();
    MPI_Request request = MPI_REQUEST_NULL;
    int const result = MPI_Irecv(message.get(), static_cast<int>(sizeof(Message)), MPI_BYTE, probedStatus.MPI_SOURCE,
        static_cast<int>(tag), static_cast<MPI_Comm>(mpiComm), &request);
    if (result != MPI_SUCCESS)
    {
        int const cleanupResult = abandonRequest(request);
        static_cast<void>(message.release());
        TLLM_THROW("NCCL error: MPI_Irecv failed while %s from rank %d (code %d, cleanup code %d)", operation,
            probedStatus.MPI_SOURCE, result, cleanupResult);
    }

    while (true)
    {
        int complete = 0;
        MPI_Status status{};
        int const testResult = MPI_Test(&request, &complete, &status);
        if (testResult != MPI_SUCCESS)
        {
            int const cleanupResult = abandonRequest(request);
            static_cast<void>(message.release());
            TLLM_THROW("NCCL error: MPI_Test failed while %s (code %d, cleanup code %d)", operation, testResult,
                cleanupResult);
        }
        if (complete != 0)
        {
            int count = 0;
            int const countResult = MPI_Get_count(&status, MPI_BYTE, &count);
            TLLM_CHECK_WITH_INFO(countResult == MPI_SUCCESS && count == static_cast<int>(sizeof(Message)),
                "NCCL error: invalid MPI rendezvous payload while %s", operation);
            return *message;
        }
        if (Clock::now() >= deadline)
        {
            int const cleanupResult = abandonRequest(request);
            static_cast<void>(message.release());
            TLLM_THROW("NCCL error: timed out while %s (MPI cleanup code %d)", operation, cleanupResult);
        }
        std::this_thread::sleep_for(kPollInterval);
    }
}

bool isActivePeer(std::vector<int> const& activeRanks, int rank, int root)
{
    return rank != root && std::binary_search(activeRanks.begin(), activeRanks.end(), rank);
}

} // namespace

NcclUniqueIdRendezvousComm::NcclUniqueIdRendezvousComm(mpi::MpiComm comm, std::vector<int> worldRanks, int worldRank)
    : mComm(std::move(comm))
    , mWorldRanks(std::move(worldRanks))
    , mWorldRank(worldRank)
{
    TLLM_CHECK_WITH_INFO(!mWorldRanks.empty(), "NCCL error: rendezvous control group must not be empty");
    TLLM_CHECK_WITH_INFO(std::is_sorted(mWorldRanks.begin(), mWorldRanks.end()),
        "NCCL error: rendezvous control ranks must be in canonical order");
    TLLM_CHECK_WITH_INFO(std::adjacent_find(mWorldRanks.begin(), mWorldRanks.end()) == mWorldRanks.end(),
        "NCCL error: rendezvous control ranks contain duplicates");
    TLLM_CHECK_WITH_INFO(mComm.getSize() == static_cast<int>(mWorldRanks.size()),
        "NCCL error: rendezvous control communicator size %d does not match rank map size %zu", mComm.getSize(),
        mWorldRanks.size());
    TLLM_CHECK_WITH_INFO(worldRank == this->worldRank(mComm.getRank()),
        "NCCL error: world rank %d does not match rendezvous control rank %d", worldRank, mComm.getRank());
}

mpi::MpiComm const& NcclUniqueIdRendezvousComm::mpiComm() const noexcept
{
    return mComm;
}

std::vector<int> const& NcclUniqueIdRendezvousComm::worldRanks() const noexcept
{
    return mWorldRanks;
}

int NcclUniqueIdRendezvousComm::worldRank() const noexcept
{
    return mWorldRank;
}

int NcclUniqueIdRendezvousComm::commRank(int worldRank) const
{
    auto const found = std::lower_bound(mWorldRanks.begin(), mWorldRanks.end(), worldRank);
    TLLM_CHECK_WITH_INFO(found != mWorldRanks.end() && *found == worldRank,
        "NCCL error: world rank %d is not present in the rendezvous control channel", worldRank);
    return static_cast<int>(std::distance(mWorldRanks.begin(), found));
}

int NcclUniqueIdRendezvousComm::worldRank(int commRank) const
{
    TLLM_CHECK_WITH_INFO(commRank >= 0 && commRank < static_cast<int>(mWorldRanks.size()),
        "NCCL error: rendezvous control rank %d is outside [0, %zu)", commRank, mWorldRanks.size());
    return mWorldRanks[static_cast<std::size_t>(commRank)];
}

std::shared_ptr<NcclUniqueIdRendezvousComm> createNcclUniqueIdRendezvousComm(
    std::vector<int> const& initialRanks, int worldRank, mpi::MpiComm const& parentComm, int creationTagSeed)
{
    TLLM_CHECK_WITH_INFO(!initialRanks.empty(), "NCCL error: rendezvous control group must not be empty");
    TLLM_CHECK_WITH_INFO(std::is_sorted(initialRanks.begin(), initialRanks.end()),
        "NCCL error: rendezvous control ranks must be in canonical order");
    TLLM_CHECK_WITH_INFO(std::adjacent_find(initialRanks.begin(), initialRanks.end()) == initialRanks.end(),
        "NCCL error: rendezvous control ranks contain duplicates");
    TLLM_CHECK_WITH_INFO(std::binary_search(initialRanks.begin(), initialRanks.end(), worldRank),
        "NCCL error: world rank %d is not a member of the rendezvous control group", worldRank);
    TLLM_CHECK_WITH_INFO(parentComm.getRank() == worldRank,
        "NCCL error: world rank %d does not match MPI parent rank %d", worldRank, parentComm.getRank());
    TLLM_CHECK_WITH_INFO(initialRanks.front() >= 0 && initialRanks.back() < parentComm.getSize(),
        "NCCL error: rendezvous control ranks must be within MPI parent size %d", parentComm.getSize());

    auto cacheEntry = getControlCommCacheEntry(initialRanks, worldRank, parentComm, creationTagSeed);
    std::lock_guard<std::mutex> const cacheLock(cacheEntry->mutex);
    if (cacheEntry->comm != nullptr)
    {
        return cacheEntry->comm;
    }
    if (cacheEntry->failure != nullptr)
    {
        // Creation is collective over the initial group. Never let one rank
        // retry that collective after a prior asymmetric local failure while
        // peers may already have cached a successfully-created channel.
        std::rethrow_exception(cacheEntry->failure);
    }

    void* tagUpperBoundValue = nullptr;
    int hasTagUpperBound = 0;
    int result
        = MPI_Comm_get_attr(static_cast<MPI_Comm>(parentComm), MPI_TAG_UB, &tagUpperBoundValue, &hasTagUpperBound);
    TLLM_CHECK_WITH_INFO(
        result == MPI_SUCCESS, "NCCL error: failed to query MPI_TAG_UB for the control channel (code %d)", result);
    TLLM_CHECK_WITH_INFO(hasTagUpperBound != 0 && tagUpperBoundValue != nullptr,
        "NCCL error: MPI parent communicator does not expose MPI_TAG_UB");
    int const tagUpperBound = *static_cast<int*>(tagUpperBoundValue);
    constexpr int kFirstControlCreationTag = static_cast<int>(mpi::MpiTag::kNcclPpControl) + 1;
    TLLM_CHECK_WITH_INFO(creationTagSeed >= 0, "NCCL error: rendezvous control creation-tag seed must be nonnegative");
    int const tagParity = creationTagSeed & 1;
    TLLM_CHECK_WITH_INFO(tagUpperBound >= kFirstControlCreationTag + tagParity,
        "NCCL error: MPI_TAG_UB %d is too small for NCCL control-channel creation", tagUpperBound);
    std::uint64_t tagHash = groupIdentity(initialRanks);
    tagHash ^= static_cast<std::uint64_t>(creationTagSeed) + 0x9e3779b97f4a7c15ULL + (tagHash << 6) + (tagHash >> 2);
    // Even/odd slots separate raw-op and PP seeds; the group hash makes lazy
    // creation order-independent within each ownership path.
    auto const tagSlots = static_cast<std::uint64_t>(tagUpperBound - kFirstControlCreationTag - tagParity) / 2 + 1;
    int const creationTag = kFirstControlCreationTag + tagParity + static_cast<int>(2 * (tagHash % tagSlots));

    MPI_Group parentGroup = MPI_GROUP_NULL;
    MPI_Group controlGroup = MPI_GROUP_NULL;
    MPI_Comm controlComm = MPI_COMM_NULL;
    auto cleanup = [&]() noexcept
    {
        if (controlGroup != MPI_GROUP_NULL)
        {
            MPI_Group_free(&controlGroup);
        }
        if (parentGroup != MPI_GROUP_NULL)
        {
            MPI_Group_free(&parentGroup);
        }
        // Once MPI_Comm_create_group has returned a non-null communicator,
        // never call collective MPI_Comm_free from an exception path. A
        // local allocation or error-handler failure can be asymmetric, so
        // peers may already have published their process-lifetime channel.
        // Leaking that rare failed construction is safer than deadlocking.
    };

    try
    {
        result = MPI_Comm_group(static_cast<MPI_Comm>(parentComm), &parentGroup);
        TLLM_CHECK_WITH_INFO(result == MPI_SUCCESS, "NCCL error: failed to inspect MPI parent group (code %d)", result);
        result = MPI_Group_incl(parentGroup, static_cast<int>(initialRanks.size()), initialRanks.data(), &controlGroup);
        TLLM_CHECK_WITH_INFO(result == MPI_SUCCESS, "NCCL error: failed to create MPI control group (code %d)", result);
        // MPI_Comm_create_group is collective only over initialRanks. This is
        // the sole control-channel collective and must happen before failure.
        result = MPI_Comm_create_group(static_cast<MPI_Comm>(parentComm), controlGroup, creationTag, &controlComm);
        TLLM_CHECK_WITH_INFO(
            result == MPI_SUCCESS, "NCCL error: failed to create MPI control communicator (code %d)", result);
        TLLM_CHECK_WITH_INFO(controlComm != MPI_COMM_NULL,
            "NCCL error: MPI returned a null rendezvous control communicator for member rank %d", worldRank);
        result = MPI_Comm_set_errhandler(controlComm, MPI_ERRORS_RETURN);
        TLLM_CHECK_WITH_INFO(result == MPI_SUCCESS,
            "NCCL error: failed to configure MPI_ERRORS_RETURN on the control channel (code %d)", result);

        MPI_Group_free(&controlGroup);
        controlGroup = MPI_GROUP_NULL;
        MPI_Group_free(&parentGroup);
        parentGroup = MPI_GROUP_NULL;
        // Never call MPI_Comm_free on a successfully-published FT control
        // channel: freeing after a member dies is not guaranteed to make
        // progress. The communicator intentionally lives until process exit.
        auto ownedComm = mpi::MpiComm{controlComm, false};
        auto resultComm = std::make_shared<NcclUniqueIdRendezvousComm>(std::move(ownedComm), initialRanks, worldRank);
        controlComm = MPI_COMM_NULL;
        cacheEntry->comm = std::move(resultComm);
        return cacheEntry->comm;
    }
    catch (...)
    {
        cacheEntry->failure = std::current_exception();
        cleanup();
        throw;
    }
}

ncclUniqueId exchangeNcclUniqueId(std::vector<int> const& activeRanks, NcclUniqueIdRendezvousComm const& controlComm,
    NcclUniqueIdRendezvousTags tags, std::uint64_t rendezvousId, Clock::time_point deadline)
{
    auto const worldRank = controlComm.worldRank();
    auto const& mpiComm = controlComm.mpiComm();
    TLLM_CHECK_WITH_INFO(!activeRanks.empty(), "NCCL error: communicator rendezvous group must not be empty");
    TLLM_CHECK_WITH_INFO(std::is_sorted(activeRanks.begin(), activeRanks.end()),
        "NCCL error: communicator rendezvous ranks must be in canonical order");
    TLLM_CHECK_WITH_INFO(std::adjacent_find(activeRanks.begin(), activeRanks.end()) == activeRanks.end(),
        "NCCL error: communicator rendezvous ranks contain duplicates");
    TLLM_CHECK_WITH_INFO(std::binary_search(activeRanks.begin(), activeRanks.end(), worldRank),
        "NCCL error: world rank %d is not active in communicator rendezvous", worldRank);
    TLLM_CHECK_WITH_INFO(std::includes(controlComm.worldRanks().begin(), controlComm.worldRanks().end(),
                             activeRanks.begin(), activeRanks.end()),
        "NCCL error: communicator rendezvous group is not a subset of the pre-failure control group");

    auto& context = getRendezvousContext(tags, controlComm);
    std::unique_lock<std::mutex> const exchangeLock(context.exchangeMutex);
    int const root = activeRanks.front();
    std::uint64_t const key = communicatorKey(activeRanks, tags);
    discardObsoletePending(context.pendingReady, key, rendezvousId);
    discardObsoletePending(context.pendingIds, key, rendezvousId);
    discardObsoletePending(context.pendingAcks, key, rendezvousId);
    if (worldRank == root)
    {
        ncclUniqueId id{};
        {
            auto const hostApiLock = acquireNcclHostApiLock();
            auto const result = ncclGetUniqueId(&id);
            TLLM_CHECK_WITH_INFO(
                result == ncclSuccess, "NCCL error: ncclGetUniqueId failed: %s", ncclGetErrorString(result));
        }
        if (activeRanks.size() == 1)
        {
            return id;
        }

        AttemptToken const serverToken = makeAttemptToken();
        std::uint64_t const digest = fnv1a(&id, sizeof(id));
        std::unordered_map<int, std::vector<AttemptToken>> issuedTokens;
        std::unordered_set<int> acknowledged;
        while (acknowledged.size() + 1 < activeRanks.size())
        {
            TLLM_CHECK_WITH_INFO(
                Clock::now() < deadline, "NCCL error: timed out waiting for survivor communicator rendezvous");
            bool progressed = false;
            ReceivedMessage<ReadyMessage> receivedReady{};
            bool haveReady = takePending(context.pendingReady, MPI_ANY_SOURCE, key, rendezvousId, receivedReady);
            MPI_Status status{};
            if (!haveReady && probeMessage(MPI_ANY_SOURCE, tags.ready, mpiComm, status))
            {
                progressed = true;
                receivedReady = {controlComm.worldRank(status.MPI_SOURCE),
                    receiveMessage<ReadyMessage>(status, tags.ready, mpiComm, deadline, "receiving NCCL READY")};
                haveReady = isCurrentAttempt(receivedReady.message, key, rendezvousId);
                if (!haveReady && shouldStashForLater(receivedReady.message, key, rendezvousId))
                {
                    stashPending(context.pendingReady, receivedReady);
                }
            }
            if (haveReady)
            {
                progressed = true;
                ReadyMessage const& ready = receivedReady.message;
                int const source = receivedReady.source;
                if (ready.version == kProtocolVersion && isCurrentAttempt(ready, key, rendezvousId)
                    && isActivePeer(activeRanks, source, root))
                {
                    auto& tokens = issuedTokens[source];
                    if (std::find(tokens.begin(), tokens.end(), ready.clientToken) == tokens.end())
                    {
                        TLLM_CHECK_WITH_INFO(tokens.size() < kMaxTokensPerPeer,
                            "NCCL error: too many stale communicator rendezvous attempts from rank %d", source);
                        tokens.push_back(ready.clientToken);
                    }
                    IdMessage const proposal{
                        kProtocolVersion, 0, key, rendezvousId, ready.clientToken, serverToken, digest, id};
                    sendMessage(
                        proposal, controlComm.commRank(source), tags.id, mpiComm, deadline, "sending NCCL unique ID");
                }
            }

            ReceivedMessage<AckMessage> receivedAck{};
            bool haveAck = takePending(context.pendingAcks, MPI_ANY_SOURCE, key, rendezvousId, receivedAck);
            if (!haveAck && probeMessage(MPI_ANY_SOURCE, tags.ack, mpiComm, status))
            {
                progressed = true;
                receivedAck = {controlComm.worldRank(status.MPI_SOURCE),
                    receiveMessage<AckMessage>(status, tags.ack, mpiComm, deadline, "receiving NCCL ACK")};
                haveAck = isCurrentAttempt(receivedAck.message, key, rendezvousId);
                if (!haveAck && shouldStashForLater(receivedAck.message, key, rendezvousId))
                {
                    stashPending(context.pendingAcks, receivedAck);
                }
            }
            if (haveAck)
            {
                progressed = true;
                AckMessage const& ack = receivedAck.message;
                int const source = receivedAck.source;
                auto const issued = issuedTokens.find(source);
                if (ack.version == kProtocolVersion && isCurrentAttempt(ack, key, rendezvousId)
                    && ack.serverToken == serverToken && ack.idDigest == digest && issued != issuedTokens.end()
                    && std::find(issued->second.begin(), issued->second.end(), ack.clientToken) != issued->second.end())
                {
                    acknowledged.insert(source);
                }
            }

            if (!progressed)
            {
                TLLM_CHECK_WITH_INFO(
                    Clock::now() < deadline, "NCCL error: timed out waiting for survivor communicator rendezvous");
                std::this_thread::sleep_for(kPollInterval);
            }
        }
        return id;
    }

    AttemptToken const clientToken = makeAttemptToken();
    ReadyMessage const ready{kProtocolVersion, 0, key, rendezvousId, clientToken};
    int const rootCommRank = controlComm.commRank(root);
    sendMessage(ready, rootCommRank, tags.ready, mpiComm, deadline, "sending NCCL READY");
    auto nextReadyRefresh = Clock::now() + kReadyRefreshInterval;
    while (true)
    {
        TLLM_CHECK_WITH_INFO(Clock::now() < deadline, "NCCL error: timed out waiting for survivor communicator ID");
        if (Clock::now() >= nextReadyRefresh)
        {
            // A root attempt can consume READY and fail before accepting our
            // ACK. Refresh the same idempotent client token so a subsequent
            // root attempt can converge without waiting for this call to end.
            sendMessage(ready, rootCommRank, tags.ready, mpiComm, deadline, "refreshing NCCL READY");
            nextReadyRefresh = Clock::now() + kReadyRefreshInterval;
        }
        ReceivedMessage<IdMessage> receivedProposal{};
        bool haveProposal = takePending(context.pendingIds, root, key, rendezvousId, receivedProposal);
        MPI_Status status{};
        if (!haveProposal && probeMessage(rootCommRank, tags.id, mpiComm, status))
        {
            receivedProposal = {controlComm.worldRank(status.MPI_SOURCE),
                receiveMessage<IdMessage>(status, tags.id, mpiComm, deadline, "receiving NCCL unique ID")};
            haveProposal = isCurrentAttempt(receivedProposal.message, key, rendezvousId);
            if (!haveProposal && shouldStashForLater(receivedProposal.message, key, rendezvousId))
            {
                stashPending(context.pendingIds, receivedProposal);
            }
        }
        if (!haveProposal)
        {
            TLLM_CHECK_WITH_INFO(Clock::now() < deadline, "NCCL error: timed out waiting for survivor communicator ID");
            std::this_thread::sleep_for(kPollInterval);
            continue;
        }

        IdMessage const& proposal = receivedProposal.message;
        if (proposal.version != kProtocolVersion || !isCurrentAttempt(proposal, key, rendezvousId)
            || !(proposal.clientToken == clientToken) || proposal.idDigest != fnv1a(&proposal.id, sizeof(proposal.id)))
        {
            sendMessage(ready, rootCommRank, tags.ready, mpiComm, deadline, "refreshing NCCL READY");
            nextReadyRefresh = Clock::now() + kReadyRefreshInterval;
            continue;
        }

        AckMessage const ack{
            kProtocolVersion, 0, key, rendezvousId, clientToken, proposal.serverToken, proposal.idDigest};
        sendMessage(ack, rootCommRank, tags.ack, mpiComm, deadline, "sending NCCL ACK");
        return proposal.id;
    }
}

} // namespace tensorrt_llm::runtime

#endif // ENABLE_MULTI_DEVICE
