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

// Round-trip tests for copyHostToDevice / copyDeviceToHost in kvCacheManagerV2Utils.cu.
//
// A call whose tasks are at least hostCopyDmaMinBytes() each takes the copy-engine path (one cuMemcpyBatchAsync, or
// one cuMemcpy{HtoD,DtoH}Async per task on the legacy NULL stream, which the batched call does not accept); smaller
// tasks take the batched copy kernel. The knob is read once per process, so instead of changing the environment
// variable mid-process the tests pick sizes on both sides of the default and ask the policy predicate which path a size
// takes; the raw-value semantics of the knob are tested through hostCopyDmaMinBytesFor().

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/tllmException.h"

// After the header that brings in ATen: nvcc fails to parse the torch headers when this one comes first.
#include "kvCacheManagerV2TestUtils.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::test;

using MMTask = Task<MemAddress, MemAddress>;

// Page sizes of a production hybrid-attention recipe: an attention page, which stays on the kernel at the default
// threshold, and the conv-state and bf16 SSM-state pages of its recurrent layers, which take the copy engines.
constexpr size_t kAttentionPageBytes = 884736;
constexpr size_t kConvStateBytes = 15261696;
constexpr size_t kSsmStateBytes = 217055232;
constexpr size_t kDefaultThreshold = size_t{4} << 20; // kvCacheManagerV2Utils.cu default

struct CudaStreamDeleter
{
    void operator()(cudaStream_t stream) const noexcept
    {
        cudaStreamDestroy(stream);
    }
};

struct CudaEventDeleter
{
    void operator()(cudaEvent_t event) const noexcept
    {
        cudaEventDestroy(event);
    }
};

using CudaStreamHandle = std::unique_ptr<CUstream_st, CudaStreamDeleter>;
using CudaEventHandle = std::unique_ptr<CUevent_st, CudaEventDeleter>;

// Stalls a stream from inside a host function until release(), so work enqueued behind it provably has not run when
// later calls are issued (same device as HostGate in kvCacheManagerV2StagingBufferTest.cu).
struct HostGate
{
    std::atomic<bool> open{false};

    static void CUDART_CB wait(void* data)
    {
        auto& gate = *static_cast<HostGate*>(data);
        while (!gate.open.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

    void release() noexcept
    {
        open.store(true, std::memory_order_release);
    }
};

// Opens the gate and drains the device on scope exit, also when an assertion returns early: a shut gate would block
// the stream forever, and the buffers the pending work targets are freed on the way out.
struct HostGateReleaseGuard
{
    HostGate& gate;

    ~HostGateReleaseGuard()
    {
        gate.release();
        cudaDeviceSynchronize();
    }
};

// Pseudo-random 64-bit words from a fixed seed: deterministic, yet no two slots of the buffer carry the same bytes.
// An affine byte pattern repeats every 256 B, and every production page size in this file is a multiple of 256 B, so
// with such a pattern all slots of a buffer would be byte-identical and a copy that permuted source slots, or read
// every slot from slot 0, would still pass the memcmp in roundTrip.
void fillPattern(unsigned char* bytes, size_t numBytes, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    size_t const numWords = numBytes / sizeof(uint64_t);
    for (size_t i = 0; i < numWords; ++i)
    {
        uint64_t const word = rng();
        std::memcpy(bytes + i * sizeof(word), &word, sizeof(word));
    }
    for (size_t i = numWords * sizeof(uint64_t); i < numBytes; ++i)
    {
        bytes[i] = static_cast<unsigned char>(rng());
    }
}

size_t roundUpToHostMemAlignment(size_t numBytes)
{
    return (numBytes + HostMem::kAlignment - 1) / HostMem::kAlignment * HostMem::kAlignment;
}

// HostMem (pinned, cuMemHostRegister'ed like the production host pools) -> device -> HostMem, byte-exact.
// numTasks contiguous slots of numBytes each, the slot layout of a host slot pool (base + slot * slot_size).
void roundTrip(size_t numBytes, size_t numTasks, cudaStream_t stream)
{
    // Keep HostMem construction cheap in the test: no prefault threads, regular pages.
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "0");

    size_t const total = numBytes * numTasks;
    HostMem src(roundUpToHostMemAlignment(total));
    HostMem dst(roundUpToHostMemAlignment(total));
    CudaAllocation device = allocateCuda(total);
    ASSERT_NE(device.get(), nullptr);

    // Drain the stream before the buffers above go away, also when an assertion below returns early.
    struct DrainOnExit
    {
        cudaStream_t stream;

        ~DrainOnExit()
        {
            cudaStreamSynchronize(stream);
        }
    } const drain{stream};

    auto* srcBytes = reinterpret_cast<unsigned char*>(src.address());
    auto* dstBytes = reinterpret_cast<unsigned char*>(dst.address());
    fillPattern(srcBytes, total, numTasks);
    std::memset(dstBytes, 0, total);
    ASSERT_EQ(cudaMemsetAsync(device.get(), 0xEE, total, stream), cudaSuccess);

    auto const deviceBase = reinterpret_cast<MemAddress>(device.get());
    std::vector<MMTask> hostToDevice;
    std::vector<MMTask> deviceToHost;
    hostToDevice.reserve(numTasks);
    deviceToHost.reserve(numTasks);
    for (size_t i = 0; i < numTasks; ++i)
    {
        hostToDevice.push_back(MMTask{deviceBase + i * numBytes, src.address() + i * numBytes});
        deviceToHost.push_back(MMTask{dst.address() + i * numBytes, deviceBase + i * numBytes});
    }

    ASSERT_EQ(copyHostToDevice(hostToDevice, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(stream)),
        CUDA_SUCCESS);
    ASSERT_EQ(copyDeviceToHost(deviceToHost, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(stream)),
        CUDA_SUCCESS);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(std::memcmp(srcBytes, dstBytes, total), 0) << "numBytes=" << numBytes << " numTasks=" << numTasks;
}

class KvCacheManagerV2HostCopyTest : public testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    cudaStream_t mStream = nullptr;
};

// The knob's documented semantics on raw values, so the process-wide cache need not be reset to reach them.
TEST(KvCacheManagerV2HostCopyPolicyTest, ThresholdFromRawValue)
{
    constexpr size_t kNever = std::numeric_limits<size_t>::max();
    EXPECT_EQ(hostCopyDmaMinBytesFor(nullptr), kDefaultThreshold);
    EXPECT_EQ(hostCopyDmaMinBytesFor(""), kDefaultThreshold);
    EXPECT_EQ(hostCopyDmaMinBytesFor("4abc"), kDefaultThreshold);                 // reported, default kept
    EXPECT_EQ(hostCopyDmaMinBytesFor("99999999999999999999"), kDefaultThreshold); // out of range, default kept
    EXPECT_EQ(hostCopyDmaMinBytesFor("0"), size_t{0});
    EXPECT_EQ(hostCopyDmaMinBytesFor("-1"), kNever);
    EXPECT_EQ(hostCopyDmaMinBytesFor("16777216"), size_t{16} << 20);
}

TEST_F(KvCacheManagerV2HostCopyTest, ThresholdFollowsEnvironment)
{
    // The cached accessor is the parser checked above applied to the variable this process started with.
    EXPECT_EQ(hostCopyDmaMinBytes(), hostCopyDmaMinBytesFor(std::getenv(kHostCopyDmaMinBytesEnv)));
    bool const chunked = HostMem::shouldUseChunkedRegistration();
    for (size_t numBytes :
        {size_t{16}, kAttentionPageBytes, kDefaultThreshold - 1, kDefaultThreshold, kConvStateBytes, kSsmStateBytes})
    {
        bool const expected = !chunked && numBytes >= hostCopyDmaMinBytes();
        EXPECT_EQ(useCopyEngineForHostCopy(numBytes), expected) << "numBytes=" << numBytes;
    }
}

// An empty call enqueues nothing whatever numBytes says: the dispatch guard in copyHostToDevice / copyDeviceToHost
// (`!tasks.empty()`) keeps it on the kernel branch and launchBatchedCopy returns CUDA_SUCCESS at batchSize 0, so
// cuMemcpyBatchAsync never sees count == 0, which its attrsIdxs/numAttrs constraints forbid.
TEST_F(KvCacheManagerV2HostCopyTest, EmptyTaskListIsANoOp)
{
    std::vector<MMTask> const none;
    for (size_t numBytes : {kAttentionPageBytes, kSsmStateBytes})
    {
        EXPECT_EQ(
            copyHostToDevice(none, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)), CUDA_SUCCESS);
        EXPECT_EQ(
            copyDeviceToHost(none, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)), CUDA_SUCCESS);
    }
    EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
}

// A negative byte count is rejected before any CUDA work on both entry points.
TEST_F(KvCacheManagerV2HostCopyTest, NegativeByteCountIsRejected)
{
    std::vector<MMTask> const tasks(1, MMTask{MemAddress{0x1000}, MemAddress{0x2000}});
    EXPECT_THROW(copyHostToDevice(tasks, ssize_t{-16}, reinterpret_cast<CUstream>(mStream)),
        tensorrt_llm::common::TllmException);
    EXPECT_THROW(copyDeviceToHost(tasks, ssize_t{-16}, reinterpret_cast<CUstream>(mStream)),
        tensorrt_llm::common::TllmException);
}

class HostCopyRoundTripTest : public KvCacheManagerV2HostCopyTest,
                              public testing::WithParamInterface<std::pair<size_t, size_t>>
{
};

TEST_P(HostCopyRoundTripTest, HostToDeviceToHost)
{
    auto const [numBytes, numTasks] = GetParam();
    roundTrip(numBytes, numTasks, mStream);
}

// 300 tasks > 256 covers the kernel path's multi-launch remainder; the state pages take the copy-engine path at
// the default threshold. (kDefaultThreshold, 8) is the inclusive threshold boundary; (kDefaultThreshold, 64) and
// (kConvStateBytes, 12) are batches of many large tasks that travel in one cuMemcpyBatchAsync.
INSTANTIATE_TEST_SUITE_P(Shapes, HostCopyRoundTripTest,
    testing::Values(std::make_pair(kAttentionPageBytes, size_t{1}), std::make_pair(kAttentionPageBytes, size_t{140}),
        std::make_pair(kAttentionPageBytes, size_t{300}), std::make_pair(size_t{16}, size_t{300}),
        std::make_pair(kConvStateBytes, size_t{1}), std::make_pair(kSsmStateBytes, size_t{1}),
        std::make_pair(kDefaultThreshold, size_t{8}), std::make_pair(kDefaultThreshold, size_t{64}),
        std::make_pair(kConvStateBytes, size_t{12})));

// The kernel requires numBytes % 16 == 0 and 16 B-aligned task addresses, and rejects a bad byte count with TLLM_CHECK
// before enqueuing work; the copy-engine path has no such constraint. Which path a size takes depends on the threshold
// this process started with (24 B is below the default, 4 MiB + 8 B above it), so each size is either round-tripped
// or asserted to throw according to the policy predicate.
TEST_F(KvCacheManagerV2HostCopyTest, NonMultipleOf16)
{
    constexpr size_t kNumTasks = 3;
    for (size_t numBytes : {size_t{24}, kDefaultThreshold + 8})
    {
        if (useCopyEngineForHostCopy(numBytes))
        {
            roundTrip(numBytes, kNumTasks, mStream);
            continue;
        }
        // Rejected before any CUDA work: the addresses are never dereferenced, so they need not be real.
        std::vector<MMTask> const tasks(kNumTasks, MMTask{MemAddress{0x1000}, MemAddress{0x2000}});
        EXPECT_THROW(copyHostToDevice(tasks, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
            tensorrt_llm::common::TllmException)
            << "numBytes=" << numBytes;
        EXPECT_THROW(copyDeviceToHost(tasks, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
            tensorrt_llm::common::TllmException)
            << "numBytes=" << numBytes;
    }
}

// The kernel launch is legal on the NULL stream; the copy-engine path falls back to one cuMemcpy{HtoD,DtoH}Async per
// task there, because cuMemcpyBatchAsync does not accept the legacy NULL stream, as 0 or as the CU_STREAM_LEGACY
// handle. All of them must round-trip.
TEST_F(KvCacheManagerV2HostCopyTest, NullStreamAccepted)
{
    roundTrip(kAttentionPageBytes, 4, nullptr);
    roundTrip(kConvStateBytes, 1, nullptr);
    roundTrip(kConvStateBytes, 3, nullptr);
    roundTrip(kConvStateBytes, 1, cudaStreamLegacy);
}

// Work queued earlier on the stream must be visible to the copy, and a second stream waiting on an event recorded
// after the copy must see the copied bytes (the runtime tracks completion only through such events). A host-function
// gate holds the producer stream until everything is enqueued, so the checks are deterministic: a copy that read its
// source out of stream order would see the pre-fill zeros, not a fill that happened to finish first.
TEST_F(KvCacheManagerV2HostCopyTest, StreamOrderingThroughEvents)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "0");

    for (size_t numBytes : {kAttentionPageBytes, kConvStateBytes})
    {
        HostMem host(roundUpToHostMemAlignment(numBytes));
        CudaAllocation device = allocateCuda(numBytes);
        CudaAllocation deviceCheck = allocateCuda(numBytes);
        ASSERT_NE(device.get(), nullptr);
        ASSERT_NE(deviceCheck.get(), nullptr);
        auto* hostBytes = reinterpret_cast<unsigned char*>(host.address());
        std::memset(hostBytes, 0, numBytes);
        // Known non-0xA5 contents before the gate: an out-of-order read must not find 0xA5 left behind by the previous
        // iteration's allocation.
        ASSERT_EQ(cudaMemsetAsync(device.get(), 0x00, numBytes, mStream), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(deviceCheck.get(), 0x00, numBytes, mStream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

        // Created before the gate is shut (stream creation can wait for host callbacks) and owned, so an early return
        // releases them.
        cudaEvent_t readyRaw = nullptr;
        ASSERT_EQ(cudaEventCreateWithFlags(&readyRaw, cudaEventDisableTiming), cudaSuccess);
        CudaEventHandle const ready{readyRaw};
        cudaStream_t consumerRaw = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&consumerRaw, cudaStreamNonBlocking), cudaSuccess);
        CudaStreamHandle const consumer{consumerRaw};

        // Hold mStream until the copy, the event and the consumer's wait are enqueued: the 0xA5 fill provably has not
        // run yet, so only stream-ordered reads can see it. Declared after the buffers so it runs before they go.
        HostGate gate;
        HostGateReleaseGuard const releaseGate{gate};
        ASSERT_EQ(cudaLaunchHostFunc(mStream, HostGate::wait, &gate), cudaSuccess);

        // Producer on mStream: fill device, then device -> host, then record.
        ASSERT_EQ(cudaMemsetAsync(device.get(), 0xA5, numBytes, mStream), cudaSuccess);
        std::vector<MMTask> deviceToHost{MMTask{host.address(), reinterpret_cast<MemAddress>(device.get())}};
        ASSERT_EQ(copyDeviceToHost(deviceToHost, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
            CUDA_SUCCESS);
        ASSERT_EQ(cudaEventRecord(ready.get(), mStream), cudaSuccess);

        // Consumer on a second stream: wait on the event, host -> deviceCheck, read back.
        ASSERT_EQ(cudaStreamWaitEvent(consumer.get(), ready.get(), 0), cudaSuccess);
        std::vector<MMTask> hostToDevice{MMTask{reinterpret_cast<MemAddress>(deviceCheck.get()), host.address()}};
        ASSERT_EQ(
            copyHostToDevice(hostToDevice, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(consumer.get())),
            CUDA_SUCCESS);
        std::vector<unsigned char> readBack(numBytes);
        // readBack is pageable, so this cudaMemcpyAsync waits for the consumer stream on the host: open the gate first
        // or the call deadlocks against the host function above. Everything that matters has been enqueued by now.
        gate.release();
        ASSERT_EQ(cudaMemcpyAsync(readBack.data(), deviceCheck.get(), numBytes, cudaMemcpyDeviceToHost, consumer.get()),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(consumer.get()), cudaSuccess);
        EXPECT_TRUE(std::all_of(readBack.begin(), readBack.end(), [](unsigned char v) { return v == 0xA5; }))
            << "numBytes=" << numBytes;
        EXPECT_TRUE(std::all_of(hostBytes, hostBytes + numBytes, [](unsigned char v) { return v == 0xA5; }))
            << "numBytes=" << numBytes;
    }
}

// Informational only: event-to-event GB/s of one SSM-state page in each direction and the path that moved it.
// Not asserted, because the figure depends on the link, the host NUMA placement and whatever else shares the
// machine. On the system that motivated the change the kernel path moved such a page at 1.8-17.5 GB/s per task and
// the copy engines in about a millisecond.
TEST_F(KvCacheManagerV2HostCopyTest, StateSlotBandwidthInfo)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");

    size_t const numBytes = kSsmStateBytes;
    HostMem host(roundUpToHostMemAlignment(numBytes));
    CudaAllocation device = allocateCuda(numBytes);
    ASSERT_NE(device.get(), nullptr);
    std::memset(reinterpret_cast<void*>(host.address()), 0x3C, numBytes);
    std::vector<MMTask> hostToDevice{MMTask{reinterpret_cast<MemAddress>(device.get()), host.address()}};
    std::vector<MMTask> deviceToHost{MMTask{host.address(), reinterpret_cast<MemAddress>(device.get())}};

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    auto timeGbPerSec = [&](auto&& issue) -> double
    {
        constexpr int kWarmup = 3;
        constexpr int kIters = 10;
        for (int i = 0; i < kWarmup; ++i)
        {
            issue();
        }
        EXPECT_EQ(cudaEventRecord(start, mStream), cudaSuccess);
        for (int i = 0; i < kIters; ++i)
        {
            issue();
        }
        EXPECT_EQ(cudaEventRecord(stop, mStream), cudaSuccess);
        EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);
        float ms = 0.0F;
        EXPECT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
        return static_cast<double>(numBytes) * kIters / (static_cast<double>(ms) * 1e-3) / 1e9;
    };

    double const h2d = timeGbPerSec(
        [&]
        {
            EXPECT_EQ(
                copyHostToDevice(hostToDevice, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
                CUDA_SUCCESS);
        });
    double const d2h = timeGbPerSec(
        [&]
        {
            EXPECT_EQ(
                copyDeviceToHost(deviceToHost, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
                CUDA_SUCCESS);
        });
    bool const copyEngine = useCopyEngineForHostCopy(numBytes);
    std::cout << "StateSlotBandwidthInfo: " << numBytes << " B page, path=" << (copyEngine ? "copy-engine" : "kernel")
              << ", host->device " << h2d << " GB/s, device->host " << d2h << " GB/s" << std::endl;
    RecordProperty("path", copyEngine ? "copy-engine" : "kernel");
    RecordProperty("host_to_device_gb_per_s", std::to_string(h2d));
    RecordProperty("device_to_host_gb_per_s", std::to_string(d2h));

    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
}

} // namespace
