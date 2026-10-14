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
// A call of at most hostCopyDmaMaxTasks() tasks of at least hostCopyDmaMinBytes() each takes the copy-engine path
// (cuMemcpy{HtoD,DtoH}Async); smaller tasks or larger batches take the batched copy kernel. Both knobs are read once
// per process, so instead of changing the environment variables mid-process the tests pick sizes and task counts on
// both sides of the defaults and reach both branches that way.

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

using MMTask = Task<MemAddress, MemAddress>;

// Page sizes of a production hybrid-attention recipe: an attention page, which stays on the kernel at the default
// threshold, and the conv-state and bf16 SSM-state pages of its recurrent layers, which take the copy engines.
constexpr size_t kAttentionPageBytes = 884736;
constexpr size_t kConvStateBytes = 15261696;
constexpr size_t kSsmStateBytes = 217055232;
constexpr size_t kDefaultThreshold = size_t{4} << 20; // kvCacheManagerV2Utils.cu default
constexpr size_t kDefaultMaxTasks = 8;                // kvCacheManagerV2Utils.cu default

struct CudaDeleter
{
    void operator()(void* ptr) const noexcept
    {
        cudaFree(ptr);
    }
};

using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation allocateCuda(size_t numBytes)
{
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, numBytes) != cudaSuccess)
    {
        return CudaAllocation{};
    }
    return CudaAllocation{ptr};
}

class ScopedEnv
{
public:
    ScopedEnv(char const* name, std::string const& value)
        : mName(name)
    {
        if (char const* oldValue = std::getenv(name); oldValue != nullptr)
        {
            mOldValue = oldValue;
            mHadOldValue = true;
        }
        if (::setenv(name, value.c_str(), /*overwrite=*/1) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "setenv failed");
        }
    }

    ~ScopedEnv()
    {
        if (mHadOldValue)
        {
            ::setenv(mName.c_str(), mOldValue.c_str(), /*overwrite=*/1);
        }
        else
        {
            ::unsetenv(mName.c_str());
        }
    }

private:
    std::string mName;
    std::string mOldValue;
    bool mHadOldValue = false;
};

// The integer value of an environment variable, or nullopt when it is unset or not an integer, which is how the
// policy reads its two knobs.
std::optional<long long> integerEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr || *env == '\0')
    {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    long long const value = std::strtoll(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0')
    {
        return std::nullopt;
    }
    return value;
}

// What hostCopyDmaMinBytes() must return for the environment this process started with.
size_t expectedThresholdFromEnv()
{
    auto const parsed = integerEnv("TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES");
    if (!parsed.has_value())
    {
        return kDefaultThreshold;
    }
    return *parsed < 0 ? std::numeric_limits<size_t>::max() : static_cast<size_t>(*parsed);
}

// What hostCopyDmaMaxTasks() must return for the environment this process started with.
size_t expectedMaxTasksFromEnv()
{
    auto const parsed = integerEnv("TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MAX_TASKS");
    if (!parsed.has_value())
    {
        return kDefaultMaxTasks;
    }
    return *parsed <= 0 ? std::numeric_limits<size_t>::max() : static_cast<size_t>(*parsed);
}

void fillPattern(unsigned char* bytes, size_t numBytes, unsigned char seed)
{
    for (size_t i = 0; i < numBytes; ++i)
    {
        bytes[i] = static_cast<unsigned char>((i * 131U + seed) & 0xFFU);
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
    ASSERT_NE(device, nullptr);

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
    fillPattern(srcBytes, total, static_cast<unsigned char>(numTasks));
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

TEST_F(KvCacheManagerV2HostCopyTest, ThresholdFollowsEnvironment)
{
    EXPECT_EQ(hostCopyDmaMinBytes(), expectedThresholdFromEnv());
    EXPECT_EQ(hostCopyDmaMaxTasks(), expectedMaxTasksFromEnv());
    bool const chunked = HostMem::shouldUseChunkedRegistration();
    for (size_t numBytes :
        {size_t{16}, kAttentionPageBytes, kDefaultThreshold - 1, kDefaultThreshold, kConvStateBytes, kSsmStateBytes})
    {
        for (size_t numTasks : {size_t{1}, size_t{2}, kDefaultMaxTasks, kDefaultMaxTasks + 1, size_t{140}})
        {
            bool const expected = !chunked && numBytes >= hostCopyDmaMinBytes() && numTasks <= hostCopyDmaMaxTasks();
            EXPECT_EQ(useCopyEngineForHostCopy(numBytes, numTasks), expected)
                << "numBytes=" << numBytes << " numTasks=" << numTasks;
        }
    }
}

// Both branches accept an empty call and enqueue nothing.
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
// the default threshold. (kDefaultThreshold, 8) is the last call on the copy-engine path at the default task
// limit, (kDefaultThreshold, 9) and (kConvStateBytes, 12) are large-task batches that the task limit keeps on the
// kernel.
INSTANTIATE_TEST_SUITE_P(Shapes, HostCopyRoundTripTest,
    testing::Values(std::make_pair(kAttentionPageBytes, size_t{1}), std::make_pair(kAttentionPageBytes, size_t{140}),
        std::make_pair(kAttentionPageBytes, size_t{300}), std::make_pair(size_t{16}, size_t{300}),
        std::make_pair(kConvStateBytes, size_t{1}), std::make_pair(kSsmStateBytes, size_t{1}),
        std::make_pair(kDefaultThreshold, kDefaultMaxTasks), std::make_pair(kDefaultThreshold, kDefaultMaxTasks + 1),
        std::make_pair(kConvStateBytes, size_t{12})));

// The kernel requires numBytes % 16 == 0 and rejects anything else with TLLM_CHECK before enqueuing work; the
// copy-engine path has no such constraint. A size below the threshold always takes the kernel, so its rejection is
// asserted on every host; the copy-engine round trip runs wherever that path is selected.
TEST_F(KvCacheManagerV2HostCopyTest, NonMultipleOf16)
{
    size_t const numTasks = 3;
    // Rejected before any CUDA work: the addresses are never dereferenced, so they need not be real.
    size_t const kernelBytes = 24;
    ASSERT_FALSE(useCopyEngineForHostCopy(kernelBytes, numTasks));
    std::vector<MMTask> const tasks(numTasks, MMTask{MemAddress{0x1000}, MemAddress{0x2000}});
    EXPECT_THROW(copyHostToDevice(tasks, static_cast<ssize_t>(kernelBytes), reinterpret_cast<CUstream>(mStream)),
        tensorrt_llm::common::TllmException);
    EXPECT_THROW(copyDeviceToHost(tasks, static_cast<ssize_t>(kernelBytes), reinterpret_cast<CUstream>(mStream)),
        tensorrt_llm::common::TllmException);

    size_t const copyEngineBytes = kDefaultThreshold + 8;
    if (useCopyEngineForHostCopy(copyEngineBytes, numTasks))
    {
        roundTrip(copyEngineBytes, numTasks, mStream);
    }
    else
    {
        EXPECT_THROW(
            copyHostToDevice(tasks, static_cast<ssize_t>(copyEngineBytes), reinterpret_cast<CUstream>(mStream)),
            tensorrt_llm::common::TllmException);
    }
}

// cuMemcpy{HtoD,DtoH}Async are legal on the NULL stream; the kernel launch is too. Both branches must accept it.
TEST_F(KvCacheManagerV2HostCopyTest, NullStreamAccepted)
{
    roundTrip(kAttentionPageBytes, 4, nullptr);
    roundTrip(kConvStateBytes, 1, nullptr);
}

// Work queued earlier on the stream must be visible to the copy, and a second stream waiting on an event recorded
// after the copy must see the copied bytes (the runtime tracks completion only through such events).
TEST_F(KvCacheManagerV2HostCopyTest, StreamOrderingThroughEvents)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "0");

    for (size_t numBytes : {kAttentionPageBytes, kConvStateBytes})
    {
        HostMem host(roundUpToHostMemAlignment(numBytes));
        CudaAllocation device = allocateCuda(numBytes);
        CudaAllocation deviceCheck = allocateCuda(numBytes);
        ASSERT_NE(device, nullptr);
        ASSERT_NE(deviceCheck, nullptr);
        auto* hostBytes = reinterpret_cast<unsigned char*>(host.address());
        std::memset(hostBytes, 0, numBytes);

        // Producer on mStream: fill device, then device -> host, then record.
        ASSERT_EQ(cudaMemsetAsync(device.get(), 0xA5, numBytes, mStream), cudaSuccess);
        std::vector<MMTask> deviceToHost{MMTask{host.address(), reinterpret_cast<MemAddress>(device.get())}};
        ASSERT_EQ(copyDeviceToHost(deviceToHost, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
            CUDA_SUCCESS);
        cudaEvent_t ready = nullptr;
        ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(ready, mStream), cudaSuccess);

        // Consumer on a second stream: wait on the event, host -> deviceCheck, read back.
        cudaStream_t consumer = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(consumer, ready, 0), cudaSuccess);
        std::vector<MMTask> hostToDevice{MMTask{reinterpret_cast<MemAddress>(deviceCheck.get()), host.address()}};
        ASSERT_EQ(copyHostToDevice(hostToDevice, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(consumer)),
            CUDA_SUCCESS);
        std::vector<unsigned char> readBack(numBytes);
        ASSERT_EQ(cudaMemcpyAsync(readBack.data(), deviceCheck.get(), numBytes, cudaMemcpyDeviceToHost, consumer),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
        EXPECT_TRUE(std::all_of(readBack.begin(), readBack.end(), [](unsigned char v) { return v == 0xA5; }))
            << "numBytes=" << numBytes;
        EXPECT_TRUE(std::all_of(hostBytes, hostBytes + numBytes, [](unsigned char v) { return v == 0xA5; }))
            << "numBytes=" << numBytes;

        ASSERT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
    }
}

// Informational only: event-to-event GB/s of one SSM-state page in each direction and the path that moved it.
// Not asserted, because the figure depends on the link, the host NUMA placement and whatever else shares the
// machine. On the system that motivated the change the kernel path moved such a page at 1.8-17.5 GB/s per task and
// the copy engines in well under a millisecond.
TEST_F(KvCacheManagerV2HostCopyTest, StateSlotBandwidthInfo)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");

    size_t const numBytes = kSsmStateBytes;
    HostMem host(roundUpToHostMemAlignment(numBytes));
    CudaAllocation device = allocateCuda(numBytes);
    ASSERT_NE(device, nullptr);
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
    bool const copyEngine = useCopyEngineForHostCopy(numBytes, 1);
    std::cout << "StateSlotBandwidthInfo: " << numBytes << " B page, path=" << (copyEngine ? "copy-engine" : "kernel")
              << ", host->device " << h2d << " GB/s, device->host " << d2h << " GB/s" << std::endl;
    RecordProperty("path", copyEngine ? "copy-engine" : "kernel");
    RecordProperty("host_to_device_gb_per_s", std::to_string(h2d));
    RecordProperty("device_to_host_gb_per_s", std::to_string(d2h));

    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
}

} // namespace
