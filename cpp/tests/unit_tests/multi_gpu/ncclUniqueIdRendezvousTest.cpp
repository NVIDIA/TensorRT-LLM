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
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/ncclCommunicator.h"
#include "tensorrt_llm/runtime/utils/ncclHostApi.h"

#include <gtest/gtest.h>

#if ENABLE_MULTI_DEVICE

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace
{

using tensorrt_llm::runtime::NcclUniqueIdRendezvousComm;
using tensorrt_llm::runtime::createNcclUniqueIdRendezvousComm;
using tensorrt_llm::runtime::exchangeNcclUniqueId;

constexpr int kTestControlTagSeed = 0x4e43;
constexpr std::uint64_t kSubsetRendezvousId = 1;
constexpr std::uint64_t kStaleAttemptRendezvousId = 2;
constexpr std::uint64_t kRetryRendezvousId = 3;
constexpr std::uint64_t kStaleFloodFirstRendezvousId = 100;
constexpr std::uint64_t kStaleFloodCurrentRendezvousId = 10'000;
constexpr int kStaleFloodAttemptCount = 4'097;
constexpr std::uint64_t kRawRecoveryRendezvousId = 4;
constexpr std::uint64_t kRawCleanupRendezvousId = 5;
constexpr std::uint64_t kPpRecoveryRendezvousId = 6;
constexpr std::uint64_t kPpExcludedCleanupRendezvousId = 7;
constexpr std::uint64_t kProcessDeathRendezvousId = 8;

tensorrt_llm::runtime::NcclUniqueIdRendezvousTags rawRendezvousTags()
{
    return {tensorrt_llm::mpi::MpiTag::kNcclCommReady, tensorrt_llm::mpi::MpiTag::kNcclCommUniqueId,
        tensorrt_llm::mpi::MpiTag::kNcclCommAck};
}

std::vector<int> allSessionRanks()
{
    std::vector<int> ranks(COMM_SESSION.getSize());
    std::iota(ranks.begin(), ranks.end(), 0);
    return ranks;
}

std::shared_ptr<NcclUniqueIdRendezvousComm> createTestControlComm()
{
    return createNcclUniqueIdRendezvousComm(
        allSessionRanks(), COMM_SESSION.getRank(), COMM_SESSION, kTestControlTagSeed);
}

class EnvironmentVariableGuard
{
public:
    EnvironmentVariableGuard(char const* name, char const* value)
        : mName(name)
    {
        if (auto const* previous = std::getenv(mName.c_str()); previous != nullptr)
        {
            mPrevious = previous;
        }
#if defined(_WIN32)
        _putenv_s(mName.c_str(), value);
#else
        setenv(mName.c_str(), value, 1);
#endif
    }

    ~EnvironmentVariableGuard()
    {
#if defined(_WIN32)
        _putenv_s(mName.c_str(), mPrevious.value_or("").c_str());
#else
        if (mPrevious.has_value())
        {
            setenv(mName.c_str(), mPrevious->c_str(), 1);
        }
        else
        {
            unsetenv(mName.c_str());
        }
#endif
    }

    EnvironmentVariableGuard(EnvironmentVariableGuard const&) = delete;
    EnvironmentVariableGuard& operator=(EnvironmentVariableGuard const&) = delete;

private:
    std::string mName;
    std::optional<std::string> mPrevious;
};

bool isEnvironmentFlagEnabled(char const* name)
{
    auto const* value = std::getenv(name);
    return value != nullptr && std::string{value} == "1";
}

bool configureDistinctCudaDeviceForEveryRank(std::string& failure)
{
    int const rank = COMM_SESSION.getRank();
    int const worldSize = COMM_SESSION.getSize();
    int deviceCount = 0;
    cudaUUID_t deviceUuid{};
    int localReady = 0;
    auto const countResult = cudaGetDeviceCount(&deviceCount);
    if (countResult == cudaSuccess && deviceCount > 0)
    {
        int const device = rank % deviceCount;
        auto const setResult = cudaSetDevice(device);
        auto const uuidResult = setResult == cudaSuccess ? cudaDeviceGetUuid(&deviceUuid, device) : setResult;
        localReady = setResult == cudaSuccess && uuidResult == cudaSuccess ? 1 : 0;
    }

    std::vector<int> gatheredReady(static_cast<std::size_t>(worldSize));
    COMM_SESSION.allgather(&localReady, gatheredReady.data(), 1, tensorrt_llm::mpi::MpiType::kINT32);
    std::vector<char> gatheredUuids(
        static_cast<std::size_t>(worldSize) * static_cast<std::size_t>(sizeof(deviceUuid.bytes)));
    COMM_SESSION.allgather(deviceUuid.bytes, gatheredUuids.data(), static_cast<int>(sizeof(deviceUuid.bytes)),
        tensorrt_llm::mpi::MpiType::kBYTE);

    if (!std::all_of(gatheredReady.begin(), gatheredReady.end(), [](int ready) { return ready != 0; }))
    {
        failure = "every MPI rank must have a usable CUDA device";
        return false;
    }

    std::unordered_set<std::string> uniqueUuids;
    for (int peer = 0; peer < worldSize; ++peer)
    {
        auto const* bytes = gatheredUuids.data()
            + static_cast<std::size_t>(peer) * static_cast<std::size_t>(sizeof(deviceUuid.bytes));
        uniqueUuids.emplace(bytes, bytes + sizeof(deviceUuid.bytes));
    }
    if (uniqueUuids.size() != static_cast<std::size_t>(worldSize))
    {
        failure = "every MPI rank must use a distinct physical GPU";
        return false;
    }
    return true;
}

[[noreturn]] void exitProcessDeathTestFailure(int rank, std::string const& failure) noexcept
{
    std::fprintf(stderr, "TRTLLM_PROCESS_DEATH_RENDEZVOUS_FAIL rank=%d: %s\n", rank, failure.c_str());
    std::fflush(stderr);
    std::_Exit(EXIT_FAILURE);
}

void CUDART_CB waitForRelease(void* data)
{
    auto* release = static_cast<std::atomic<bool>*>(data);
    while (!release->load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

class HostCallbackReleaseGuard
{
public:
    explicit HostCallbackReleaseGuard(std::atomic<bool>& release)
        : mRelease(release)
    {
    }

    ~HostCallbackReleaseGuard()
    {
        mRelease.store(true, std::memory_order_release);
    }

private:
    std::atomic<bool>& mRelease;
};

TEST(NcclUniqueIdRendezvousTest, DedicatedControlCommPreservesParentErrorHandlerAndRankMap)
{
    MPI_Errhandler parentBefore = MPI_ERRHANDLER_NULL;
    ASSERT_EQ(MPI_Comm_get_errhandler(static_cast<MPI_Comm>(COMM_SESSION), &parentBefore), MPI_SUCCESS);

    auto control = createTestControlComm();

    MPI_Errhandler parentAfter = MPI_ERRHANDLER_NULL;
    MPI_Errhandler controlHandler = MPI_ERRHANDLER_NULL;
    ASSERT_EQ(MPI_Comm_get_errhandler(static_cast<MPI_Comm>(COMM_SESSION), &parentAfter), MPI_SUCCESS);
    ASSERT_EQ(MPI_Comm_get_errhandler(static_cast<MPI_Comm>(control->mpiComm()), &controlHandler), MPI_SUCCESS);
    EXPECT_EQ(parentAfter, parentBefore);
    EXPECT_EQ(controlHandler, MPI_ERRORS_RETURN);
    EXPECT_EQ(control->worldRank(), COMM_SESSION.getRank());
    EXPECT_EQ(control->worldRank(control->commRank(COMM_SESSION.getRank())), COMM_SESSION.getRank());

    ASSERT_EQ(MPI_Errhandler_free(&parentBefore), MPI_SUCCESS);
    ASSERT_EQ(MPI_Errhandler_free(&parentAfter), MPI_SUCCESS);
    ASSERT_EQ(MPI_Errhandler_free(&controlHandler), MPI_SUCCESS);
}

TEST(NcclUniqueIdRendezvousTest, DedicatedControlCommCacheReusesSameTopology)
{
    auto const firstControl = createTestControlComm();
    auto const secondControl = createTestControlComm();

    ASSERT_NE(firstControl, nullptr);
    ASSERT_NE(secondControl, nullptr);
    EXPECT_EQ(firstControl.get(), secondControl.get());
    EXPECT_EQ(MPI_Comm_c2f(static_cast<MPI_Comm>(firstControl->mpiComm())),
        MPI_Comm_c2f(static_cast<MPI_Comm>(secondControl->mpiComm())));
}

TEST(NcclUniqueIdRendezvousTest, DedicatedControlCommCacheSurvivesCustomParentHandleLifetime)
{
    auto const sessionControl = createTestControlComm();
    std::shared_ptr<NcclUniqueIdRendezvousComm> firstDuplicateControl;
    {
        MPI_Comm duplicate = MPI_COMM_NULL;
        ASSERT_EQ(MPI_Comm_dup(static_cast<MPI_Comm>(COMM_SESSION), &duplicate), MPI_SUCCESS);
        auto duplicateParent = tensorrt_llm::mpi::MpiComm{duplicate, true};
        firstDuplicateControl = createNcclUniqueIdRendezvousComm(
            allSessionRanks(), COMM_SESSION.getRank(), duplicateParent, kTestControlTagSeed);
    }

    std::shared_ptr<NcclUniqueIdRendezvousComm> secondDuplicateControl;
    {
        MPI_Comm duplicate = MPI_COMM_NULL;
        ASSERT_EQ(MPI_Comm_dup(static_cast<MPI_Comm>(COMM_SESSION), &duplicate), MPI_SUCCESS);
        auto duplicateParent = tensorrt_llm::mpi::MpiComm{duplicate, true};
        secondDuplicateControl = createNcclUniqueIdRendezvousComm(
            allSessionRanks(), COMM_SESSION.getRank(), duplicateParent, kTestControlTagSeed);
    }

    EXPECT_EQ(firstDuplicateControl.get(), sessionControl.get());
    EXPECT_EQ(secondDuplicateControl.get(), sessionControl.get());
}

TEST(NcclUniqueIdRendezvousTest, NonContiguousSurvivorSubsetExchangesWithoutExcludedRank)
{
    auto control = createTestControlComm();
    int const rank = COMM_SESSION.getRank();
    int const worldSize = COMM_SESSION.getSize();
    if (worldSize < 2)
    {
        GTEST_SKIP() << "Test requires at least two MPI processes";
    }

    // With three or more processes rank 1 is deliberately excluded, proving
    // that recovery rendezvous does not contain a hidden world collective.
    std::vector<int> activeRanks{0, worldSize - 1};
    std::array<unsigned char, sizeof(ncclUniqueId)> localId{};
    if (std::binary_search(activeRanks.begin(), activeRanks.end(), rank))
    {
        auto const id = exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(), kSubsetRendezvousId,
            std::chrono::steady_clock::now() + std::chrono::seconds{10});
        std::memcpy(localId.data(), &id, sizeof(id));
    }

    // This synchronization is test-only and happens after the survivor
    // exchange. The production recovery path performs no MPI collective.
    COMM_SESSION.barrier();
    std::vector<unsigned char> gathered(static_cast<std::size_t>(worldSize) * localId.size());
    COMM_SESSION.allgather(
        localId.data(), gathered.data(), static_cast<int>(localId.size()), tensorrt_llm::mpi::MpiType::kBYTE);

    auto const* first = gathered.data();
    auto const* last = gathered.data() + static_cast<std::size_t>(worldSize - 1) * localId.size();
    EXPECT_TRUE(std::any_of(first, first + localId.size(), [](unsigned char value) { return value != 0; }));
    EXPECT_TRUE(std::equal(first, first + localId.size(), last));
    if (worldSize >= 3)
    {
        auto const* excluded = gathered.data() + localId.size();
        EXPECT_TRUE(std::all_of(excluded, excluded + localId.size(), [](unsigned char value) { return value == 0; }));
    }
}

TEST(NcclUniqueIdRendezvousTest, RetryGenerationCannotPairWithStaleAttemptMessages)
{
    auto const control = createTestControlComm();
    int const rank = COMM_SESSION.getRank();
    int const worldSize = COMM_SESSION.getSize();
    if (worldSize < 2)
    {
        GTEST_SKIP() << "Test requires at least two MPI processes";
    }

    std::vector<int> const activeRanks{0, worldSize - 1};
    bool const isRoot = rank == activeRanks.front();
    bool const isClient = rank == activeRanks.back();
    bool staleAttemptTimedOut = false;
    bool retrySucceeded = false;
    std::array<unsigned char, sizeof(ncclUniqueId)> retryId{};

    // The root times out generation A before the client starts it, then enters
    // generation B. The client deliberately sends a late generation-A READY
    // while the root is already waiting in B. That message must be stashed,
    // not accepted as B's READY; both ranks converge only after the client also
    // advances to B.
    control->mpiComm().barrier();
    if (isRoot)
    {
        try
        {
            static_cast<void>(exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(),
                kStaleAttemptRendezvousId, std::chrono::steady_clock::now() + std::chrono::milliseconds{100}));
        }
        catch (std::exception const& error)
        {
            staleAttemptTimedOut = std::string{error.what()}.find("timed out") != std::string::npos;
        }

        try
        {
            auto const id = exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(), kRetryRendezvousId,
                std::chrono::steady_clock::now() + std::chrono::seconds{5});
            std::memcpy(retryId.data(), &id, sizeof(id));
            retrySucceeded = true;
        }
        catch (std::exception const&)
        {
            // Assert after every rank reaches the test-only barrier. If the
            // generation key regresses, the late A exchange can consume B and
            // leave this retry waiting until its bounded deadline.
        }
    }
    else if (isClient)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{500});
        try
        {
            static_cast<void>(exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(),
                kStaleAttemptRendezvousId, std::chrono::steady_clock::now() + std::chrono::milliseconds{150}));
        }
        catch (std::exception const& error)
        {
            staleAttemptTimedOut = std::string{error.what()}.find("timed out") != std::string::npos;
        }

        try
        {
            auto const id = exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(), kRetryRendezvousId,
                std::chrono::steady_clock::now() + std::chrono::seconds{5});
            std::memcpy(retryId.data(), &id, sizeof(id));
            retrySucceeded = true;
        }
        catch (std::exception const&)
        {
            // Keep the MPI test convergent on the regression path; the
            // assertions below retain the failure details.
        }
    }

    control->mpiComm().barrier();
    std::vector<unsigned char> gathered(static_cast<std::size_t>(worldSize) * retryId.size());
    control->mpiComm().allgather(
        retryId.data(), gathered.data(), static_cast<int>(retryId.size()), tensorrt_llm::mpi::MpiType::kBYTE);

    if (isRoot || isClient)
    {
        EXPECT_TRUE(staleAttemptTimedOut);
        EXPECT_TRUE(retrySucceeded);
    }
    auto const* rootId = gathered.data();
    auto const* clientId = gathered.data() + static_cast<std::size_t>(worldSize - 1) * retryId.size();
    EXPECT_TRUE(std::any_of(rootId, rootId + retryId.size(), [](unsigned char value) { return value != 0; }));
    EXPECT_TRUE(std::equal(rootId, rootId + retryId.size(), clientId));
}

TEST(NcclUniqueIdRendezvousTest, RepeatedStaleGenerationsDoNotExhaustPendingQueue)
{
    auto const control = createTestControlComm();
    int const rank = COMM_SESSION.getRank();
    int const worldSize = COMM_SESSION.getSize();
    if (worldSize < 2)
    {
        GTEST_SKIP() << "Test requires at least two MPI processes";
    }

    std::vector<int> const activeRanks{0, worldSize - 1};
    bool const isRoot = rank == activeRanks.front();
    bool const isClient = rank == activeRanks.back();
    bool currentAttemptSucceeded = false;
    int staleAttemptsTimedOut = 0;
    std::array<unsigned char, sizeof(ncclUniqueId)> currentId{};

    // The old opaque-key protocol stashed every lower-generation READY while
    // the root waited on the current generation, then failed at its 4,096
    // pending-message cap. Distinct increasing stale IDs honor the caller
    // contract while proving that same-communicator generations older than the
    // root's current ID are discarded instead of retained indefinitely.
    control->mpiComm().barrier();
    if (isRoot)
    {
        try
        {
            auto const id = exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(),
                kStaleFloodCurrentRendezvousId, std::chrono::steady_clock::now() + std::chrono::seconds{120});
            std::memcpy(currentId.data(), &id, sizeof(id));
            currentAttemptSucceeded = true;
        }
        catch (std::exception const&)
        {
            // Converge at the final test-only barrier even when the old queue
            // limit is hit; the assertions below report the regression.
        }
    }
    else if (isClient)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        for (int attempt = 0; attempt < kStaleFloodAttemptCount; ++attempt)
        {
            auto const staleId = kStaleFloodFirstRendezvousId + static_cast<std::uint64_t>(attempt);
            try
            {
                static_cast<void>(exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(), staleId,
                    std::chrono::steady_clock::now() + std::chrono::milliseconds{2}));
            }
            catch (std::exception const& error)
            {
                if (std::string{error.what()}.find("timed out waiting for survivor communicator ID")
                    != std::string::npos)
                {
                    ++staleAttemptsTimedOut;
                }
            }
        }

        try
        {
            auto const id = exchangeNcclUniqueId(activeRanks, *control, rawRendezvousTags(),
                kStaleFloodCurrentRendezvousId, std::chrono::steady_clock::now() + std::chrono::seconds{60});
            std::memcpy(currentId.data(), &id, sizeof(id));
            currentAttemptSucceeded = true;
        }
        catch (std::exception const&)
        {
            // The root may already have failed on the old pending-queue cap.
        }
    }

    control->mpiComm().barrier();
    std::vector<unsigned char> gathered(static_cast<std::size_t>(worldSize) * currentId.size());
    control->mpiComm().allgather(
        currentId.data(), gathered.data(), static_cast<int>(currentId.size()), tensorrt_llm::mpi::MpiType::kBYTE);

    if (isRoot || isClient)
    {
        EXPECT_TRUE(currentAttemptSucceeded);
    }
    if (isClient)
    {
        EXPECT_EQ(staleAttemptsTimedOut, kStaleFloodAttemptCount);
    }
    auto const* rootId = gathered.data();
    auto const* clientId = gathered.data() + static_cast<std::size_t>(worldSize - 1) * currentId.size();
    EXPECT_TRUE(std::any_of(rootId, rootId + currentId.size(), [](unsigned char value) { return value != 0; }));
    EXPECT_TRUE(std::equal(rootId, rootId + currentId.size(), clientId));
}

TEST(NcclUniqueIdRendezvousTest, WaitOperationDoesNotHoldGlobalGateForUnrelatedStreamPrefix)
{
    EnvironmentVariableGuard const faultToleranceMode{"TLLM_FAULT_TOLERANCE_MODE", "1"};
    EnvironmentVariableGuard const operationTimeout{"TRTLLM_NCCL_NONBLOCKING_TIMEOUT_MS", "200"};
    EnvironmentVariableGuard const watchdogPoll{"TRTLLM_NCCL_WATCHDOG_POLL_INTERVAL_MS", "10"};

    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount == 0)
    {
        GTEST_SKIP() << "Test requires a visible CUDA device";
    }
    int const rank = COMM_SESSION.getRank();
    ASSERT_EQ(cudaSetDevice(rank % deviceCount), cudaSuccess);

    auto comm = tensorrt_llm::getComm({rank});
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    std::atomic<bool> releasePrefix{false};
    HostCallbackReleaseGuard const releaseGuard{releasePrefix};
    ASSERT_EQ(cudaLaunchHostFunc(stream, waitForRelease, &releasePrefix), cudaSuccess);

    std::uint64_t token = 0;
    {
        auto lease = tensorrt_llm::acquireComm(comm);
        token = lease.begin(stream, "operation behind unrelated stream prefix");
        lease.track(token, stream);
    }

    std::atomic<bool> waiterStarted{false};
    std::exception_ptr waiterError;
    std::thread waiter(
        [&]()
        {
            try
            {
                waiterStarted.store(true, std::memory_order_release);
                tensorrt_llm::waitCommOperation(comm, token, "operation behind unrelated stream prefix");
            }
            catch (...)
            {
                waiterError = std::current_exception();
            }
        });
    while (!waiterStarted.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
    // Give the waiter time to enter its polling path before contending for the
    // process-wide NCCL host gate from an unrelated control thread.
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    std::atomic<bool> gateAcquired{false};
    std::thread gateContender(
        [&]()
        {
            auto gate = tensorrt_llm::runtime::acquireNcclHostApiLock();
            gateAcquired.store(true, std::memory_order_release);
        });
    auto const gateDeadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (!gateAcquired.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < gateDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    bool const acquiredBeforePrefixRelease = gateAcquired.load(std::memory_order_acquire);

    releasePrefix.store(true, std::memory_order_release);
    gateContender.join();
    waiter.join();

    EXPECT_TRUE(acquiredBeforePrefixRelease)
        << "waitCommOperation retained the process-wide NCCL host gate while waiting for an unrelated stream prefix";
    EXPECT_FALSE(waiterError);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(NcclUniqueIdRendezvousTest, FaultToleranceRejectsGraphCaptureAndTimesOutStalledOperation)
{
    EnvironmentVariableGuard const faultToleranceMode{"TLLM_FAULT_TOLERANCE_MODE", "1"};
    EnvironmentVariableGuard const operationTimeout{"TRTLLM_NCCL_NONBLOCKING_TIMEOUT_MS", "100"};
    EnvironmentVariableGuard const watchdogPoll{"TRTLLM_NCCL_WATCHDOG_POLL_INTERVAL_MS", "10"};

    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount == 0)
    {
        GTEST_SKIP() << "Test requires a visible CUDA device";
    }
    int const rank = COMM_SESSION.getRank();
    ASSERT_EQ(cudaSetDevice(rank % deviceCount), cudaSuccess);

    std::set<int> const singletonGroup{rank};
    auto comm = tensorrt_llm::getComm(singletonGroup);
    cudaStream_t stream = nullptr;
    int* captureScratch = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&captureScratch), sizeof(*captureScratch)), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(captureScratch, 0, sizeof(*captureScratch), stream), cudaSuccess);
    {
        auto lease = tensorrt_llm::acquireComm(comm);
        EXPECT_THROW(static_cast<void>(lease.begin(stream, "captured NCCL test operation")), std::exception);
    }
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    if (graph != nullptr)
    {
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    }
    ASSERT_EQ(cudaFree(captureScratch), cudaSuccess);

    std::atomic<bool> releaseCallback{false};
    HostCallbackReleaseGuard const releaseGuard{releaseCallback};
    {
        auto lease = tensorrt_llm::acquireComm(comm);
        auto const token = lease.begin(stream, "injected stalled NCCL operation");
        ASSERT_EQ(cudaLaunchHostFunc(stream, waitForRelease, &releaseCallback), cudaSuccess);
        lease.track(token, stream);
    }

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    std::optional<std::string> error;
    while (std::chrono::steady_clock::now() < deadline)
    {
        error = tensorrt_llm::getCommAsyncError(singletonGroup);
        if (error.has_value())
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    EXPECT_TRUE(error.has_value());
    if (error.has_value())
    {
        EXPECT_NE(error->find("timed out"), std::string::npos);
    }

    releaseCallback.store(true, std::memory_order_release);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(NcclUniqueIdRendezvousTest, FaultToleranceDisabledPreservesLegacyOperationPath)
{
    if (!isEnvironmentFlagEnabled("TRTLLM_TEST_FT_DEFAULT_OFF"))
    {
        GTEST_SKIP() << "Run in the isolated default-off CTest process";
    }
    ASSERT_FALSE(tensorrt_llm::mpi::isFaultToleranceModeEnabled());

    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount == 0)
    {
        GTEST_SKIP() << "Test requires a visible CUDA device";
    }
    int const rank = COMM_SESSION.getRank();
    ASSERT_EQ(cudaSetDevice(rank % deviceCount), cudaSuccess);

    std::set<int> const singletonGroup{rank};
    auto comm = tensorrt_llm::getComm(singletonGroup);
    auto lease = tensorrt_llm::acquireComm(comm);
    EXPECT_EQ(lease.begin(nullptr, "default-off legacy operation"), 0);
    EXPECT_EQ(lease.get(), *comm);
    EXPECT_THROW(
        tensorrt_llm::abortAndReinitComm(singletonGroup, singletonGroup, kRawRecoveryRendezvousId), std::exception);
}

TEST(NcclUniqueIdRendezvousTest, ProcessDeathRendezvousRequiresOptInFaultTolerantMpiLauncher)
{
    if (!isEnvironmentFlagEnabled("TRTLLM_TEST_NCCL_PROCESS_DEATH"))
    {
        GTEST_SKIP() << "This destructive test requires a dedicated fault-tolerant MPI launcher";
    }
    int const rank = COMM_SESSION.getRank();
    int const worldSize = COMM_SESSION.getSize();
    if (worldSize != 3)
    {
        exitProcessDeathTestFailure(rank, "the opt-in process-death test requires exactly three MPI processes");
    }

    try
    {
        std::string gpuFailure;
        if (!configureDistinctCudaDeviceForEveryRank(gpuFailure))
        {
            exitProcessDeathTestFailure(rank, gpuFailure);
        }

        // Establish both the production raw-NCCL communicator and its retained
        // MPI control channel while every rank is healthy. The separate test
        // control channel supplies only the pre-failure synchronization below.
        auto const control = createTestControlComm();
        auto const fullRanks = allSessionRanks();
        std::set<int> const fullGroup(fullRanks.begin(), fullRanks.end());
        auto const fullComm = tensorrt_llm::getComm(fullGroup);
        if (fullComm == nullptr || *fullComm == nullptr)
        {
            throw std::runtime_error("healthy full-group NCCL communicator is null");
        }
        control->mpiComm().barrier();
        if (rank == 1)
        {
            std::_Exit(EXIT_SUCCESS);
        }

        // This probes the stronger Mode-A contract: after one process exits,
        // surviving ranks must still exchange a fresh NCCL ID over the retained
        // MPI channel, abort the poisoned full communicator, and execute real
        // data-plane work on its replacement.
        std::this_thread::sleep_for(std::chrono::milliseconds{500});
        std::set<int> const survivorGroup{0, 2};
        tensorrt_llm::abortAndReinitComm(fullGroup, survivorGroup, kProcessDeathRendezvousId);
        auto const rebuiltComm = tensorrt_llm::getComm(survivorGroup);
        if (rebuiltComm == nullptr || *rebuiltComm == nullptr)
        {
            throw std::runtime_error("rebuilt survivor NCCL communicator is null");
        }

        auto const checkCuda = [](cudaError_t result, char const* operation)
        {
            if (result != cudaSuccess)
            {
                throw std::runtime_error(std::string{operation} + " failed: " + cudaGetErrorString(result));
            }
        };
        int const hostInput = rank + 1;
        int hostOutput = 0;
        int* deviceInput = nullptr;
        int* deviceOutput = nullptr;
        cudaStream_t stream = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInput), sizeof(hostInput)), "cudaMalloc(input)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), sizeof(hostOutput)), "cudaMalloc(output)");
        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        checkCuda(cudaMemcpyAsync(deviceInput, &hostInput, sizeof(hostInput), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(input)");

        std::uint64_t operationToken = 0;
        {
            auto lease = tensorrt_llm::acquireComm(rebuiltComm);
            operationToken = lease.begin(stream, "ncclAllReduce(process-death recovery test)");
            lease.check(ncclAllReduce(deviceInput, deviceOutput, 1, ncclInt32, ncclSum, lease.get(), stream),
                "ncclAllReduce(process-death recovery test)");
            lease.track(operationToken, stream);
        }
        tensorrt_llm::waitCommOperation(rebuiltComm, operationToken, "ncclAllReduce(process-death recovery test)");
        checkCuda(
            cudaMemcpy(&hostOutput, deviceOutput, sizeof(hostOutput), cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
        constexpr int kExpectedSurvivorSum = 4;
        if (hostOutput != kExpectedSurvivorSum)
        {
            throw std::runtime_error("post-recovery allreduce returned " + std::to_string(hostOutput) + ", expected "
                + std::to_string(kExpectedSurvivorSum));
        }

        if (rank == *survivorGroup.begin())
        {
            // Some fault-tolerant launchers still return nonzero because the
            // victim intentionally skipped MPI_Finalize. CTest keys off this
            // flushed sentinel, reached only after survivor communicator
            // rebuild and the expected allreduce, instead of that exit code.
            std::fprintf(stdout, "TRTLLM_PROCESS_DEATH_RENDEZVOUS_OK\n");
            std::fflush(stdout);
        }
        std::_Exit(EXIT_SUCCESS);
    }
    catch (std::exception const& error)
    {
        exitProcessDeathTestFailure(rank, error.what());
    }
    catch (...)
    {
        exitProcessDeathTestFailure(rank, "unknown exception");
    }
}

TEST(NcclUniqueIdRendezvousTest, RawCommunicatorRebuildRunsPostRecoveryCollective)
{
    int const rank = COMM_SESSION.getRank();
    int const worldSize = COMM_SESSION.getSize();
    if (worldSize < 3)
    {
        GTEST_SKIP() << "Test requires at least three MPI processes";
    }
    EnvironmentVariableGuard const faultToleranceMode{"TLLM_FAULT_TOLERANCE_MODE", "1"};

    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount == 0)
    {
        GTEST_SKIP() << "Test requires a visible CUDA device";
    }
    int const device = rank % deviceCount;
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);

    // A launcher may expose all GPUs to every rank or bind each rank to one
    // different physical GPU. Compare UUIDs rather than local ordinals so both
    // layouts work, and never run multiple NCCL ranks on the same device.
    cudaUUID_t deviceUuid{};
    ASSERT_EQ(cudaDeviceGetUuid(&deviceUuid, device), cudaSuccess);
    std::vector<char> gatheredUuids(
        static_cast<std::size_t>(worldSize) * static_cast<std::size_t>(sizeof(deviceUuid.bytes)));
    COMM_SESSION.allgather(deviceUuid.bytes, gatheredUuids.data(), static_cast<int>(sizeof(deviceUuid.bytes)),
        tensorrt_llm::mpi::MpiType::kBYTE);
    std::unordered_set<std::string> uniqueUuids;
    for (int peer = 0; peer < worldSize; ++peer)
    {
        auto const* bytes = gatheredUuids.data()
            + static_cast<std::size_t>(peer) * static_cast<std::size_t>(sizeof(deviceUuid.bytes));
        uniqueUuids.emplace(bytes, bytes + sizeof(deviceUuid.bytes));
    }
    if (uniqueUuids.size() != static_cast<std::size_t>(worldSize))
    {
        GTEST_SKIP() << "Test requires one distinct physical GPU per MPI rank";
    }

    auto const fullRanks = allSessionRanks();
    std::set<int> const fullGroup(fullRanks.begin(), fullRanks.end());
    std::set<int> const survivorGroup{0, worldSize - 1};
    auto const testSync = createTestControlComm();
    auto oldComm = tensorrt_llm::getComm(fullGroup);
    ASSERT_NE(oldComm, nullptr);
    EXPECT_EQ(tensorrt_llm::getCommWorldRank(oldComm), rank);

    // Keep the old registry state alive so its storage cannot be recycled;
    // pointer inequality then proves recovery installed a fresh state.
    std::shared_ptr<ncclComm_t> cachedSurvivorComm;
    ncclComm_t* cachedSurvivorState = nullptr;
    if (survivorGroup.count(rank) != 0)
    {
        cachedSurvivorComm = tensorrt_llm::getComm(survivorGroup);
        ASSERT_NE(cachedSurvivorComm, nullptr);
        cachedSurvivorState = cachedSurvivorComm.get();
    }

    // Ensure the survivor-target cache entry exists before any rank begins
    // recovery. Use a dedicated test communicator so the excluded rank does
    // not enter a parent-communicator collective while survivors create their
    // subgroup control channel.
    testSync->mpiComm().barrier();

    if (survivorGroup.count(rank) != 0)
    {
        tensorrt_llm::abortAndReinitComm(fullGroup, survivorGroup, kRawRecoveryRendezvousId);
        auto rebuiltComm = tensorrt_llm::getComm(survivorGroup);
        ASSERT_NE(rebuiltComm, nullptr);
        EXPECT_EQ(tensorrt_llm::getCommWorldRank(rebuiltComm), rank);
        EXPECT_NE(rebuiltComm.get(), cachedSurvivorState);
        EXPECT_EQ(tensorrt_llm::getComm(survivorGroup).get(), rebuiltComm.get());

        int const hostInput = rank + 1;
        int hostOutput = 0;
        int* deviceInput = nullptr;
        int* deviceOutput = nullptr;
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&deviceInput), sizeof(hostInput)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), sizeof(hostOutput)), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(deviceInput, &hostInput, sizeof(hostInput), cudaMemcpyHostToDevice, stream), cudaSuccess);

        std::uint64_t operationToken = 0;
        {
            auto lease = tensorrt_llm::acquireComm(rebuiltComm);
            operationToken = lease.begin(stream, "ncclAllReduce(recovery test)");
            lease.check(ncclAllReduce(deviceInput, deviceOutput, 1, ncclInt32, ncclSum, lease.get(), stream),
                "ncclAllReduce(recovery test)");
            lease.track(operationToken, stream);
        }
        tensorrt_llm::waitCommOperation(rebuiltComm, operationToken, "ncclAllReduce(recovery test)");

        ASSERT_EQ(cudaMemcpy(&hostOutput, deviceOutput, sizeof(hostOutput), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(hostOutput, worldSize + 1);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        EXPECT_EQ(cudaFree(deviceOutput), cudaSuccess);
        EXPECT_EQ(cudaFree(deviceInput), cudaSuccess);

        // Leave only a singleton communicator for process teardown. This is
        // test cleanup, not another simulated survivor-membership decision.
        tensorrt_llm::abortAndReinitComm(survivorGroup, {rank}, kRawCleanupRendezvousId);
    }
    else
    {
        // The excluded rank deliberately does not participate in the survivor
        // rendezvous. Once survivors have started rebuilding, replace its old
        // local communicator with a singleton so test-process teardown never
        // calls ncclCommDestroy on a half-aborted full-group communicator.
        tensorrt_llm::abortAndReinitComm(fullGroup, {rank}, kRawCleanupRendezvousId);
    }

    // Test-only synchronization after every local recovery path has finished.
    testSync->mpiComm().barrier();
}

TEST(NcclUniqueIdRendezvousTest, PipelineCommunicatorRebuildRunsPostRecoverySendRecv)
{
    int const rank = COMM_SESSION.getRank();
    int const worldSize = COMM_SESSION.getSize();
    if (worldSize < 3)
    {
        GTEST_SKIP() << "Test requires at least three MPI processes";
    }
    EnvironmentVariableGuard const faultToleranceMode{"TLLM_FAULT_TOLERANCE_MODE", "1"};

    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount == 0)
    {
        GTEST_SKIP() << "Test requires a visible CUDA device";
    }
    int const device = rank % deviceCount;
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);

    cudaUUID_t deviceUuid{};
    ASSERT_EQ(cudaDeviceGetUuid(&deviceUuid, device), cudaSuccess);
    std::vector<char> gatheredUuids(
        static_cast<std::size_t>(worldSize) * static_cast<std::size_t>(sizeof(deviceUuid.bytes)));
    COMM_SESSION.allgather(deviceUuid.bytes, gatheredUuids.data(), static_cast<int>(sizeof(deviceUuid.bytes)),
        tensorrt_llm::mpi::MpiType::kBYTE);
    std::unordered_set<std::string> uniqueUuids;
    for (int peer = 0; peer < worldSize; ++peer)
    {
        auto const* bytes = gatheredUuids.data()
            + static_cast<std::size_t>(peer) * static_cast<std::size_t>(sizeof(deviceUuid.bytes));
        uniqueUuids.emplace(bytes, bytes + sizeof(deviceUuid.bytes));
    }
    if (uniqueUuids.size() != static_cast<std::size_t>(worldSize))
    {
        GTEST_SKIP() << "Test requires one distinct physical GPU per MPI rank";
    }

    auto const testSync = createTestControlComm();
    auto communicator = std::make_unique<tensorrt_llm::runtime::NcclCommunicator>(worldSize, rank);
    std::vector<int> const survivorRanks{0, worldSize - 1};
    bool const isSurvivor = std::binary_search(survivorRanks.begin(), survivorRanks.end(), rank);

    testSync->mpiComm().barrier();
    if (isSurvivor)
    {
        communicator->abortAndReinit(survivorRanks, kPpRecoveryRendezvousId);
        EXPECT_EQ(communicator->getActiveRanks(), survivorRanks);
    }
    else
    {
        communicator->abortAndReinit({rank}, kPpExcludedCleanupRendezvousId);
        EXPECT_EQ(communicator->getActiveRanks(), std::vector<int>{rank});
    }
    EXPECT_TRUE(communicator->getAsyncError().empty());

    int* deviceValue = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&deviceValue), sizeof(*deviceValue)), cudaSuccess);
    int hostValue = rank == survivorRanks.front() ? 0x1a07 : 0;
    ASSERT_EQ(cudaMemcpy(deviceValue, &hostValue, sizeof(hostValue), cudaMemcpyHostToDevice), cudaSuccess);
    auto buffer = tensorrt_llm::runtime::IBuffer::wrap(deviceValue, nvinfer1::DataType::kINT32, 1);
    tensorrt_llm::runtime::CudaStream stream;

    if (rank == survivorRanks.front())
    {
        communicator->send(*buffer, survivorRanks.back(), stream);
        stream.synchronize();
    }
    else if (rank == survivorRanks.back())
    {
        communicator->receive(*buffer, survivorRanks.front(), stream);
        stream.synchronize();
        ASSERT_EQ(cudaMemcpy(&hostValue, deviceValue, sizeof(hostValue), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(hostValue, 0x1a07);
    }

    buffer.reset();
    ASSERT_EQ(cudaFree(deviceValue), cudaSuccess);
    testSync->mpiComm().barrier();
    communicator.reset();
    testSync->mpiComm().barrier();
}

} // namespace

#endif // ENABLE_MULTI_DEVICE
