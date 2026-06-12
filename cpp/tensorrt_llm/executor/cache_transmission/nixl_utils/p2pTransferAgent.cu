/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/p2pTransferAgent.h"

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cub/device/device_memcpy.cuh>
#include <errno.h>
#include <poll.h>
#include <random>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace tensorrt_llm::executor::kv_cache
{

// ============================================================================
// Unix Domain Socket helpers for POSIX FD passing (SCM_RIGHTS)
// ============================================================================

namespace
{

bool sendFd(int socket, int fd)
{
    struct msghdr msg = {};
    struct iovec iov[1];
    char buf[1] = {0};
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    std::memset(cmsgbuf, 0, sizeof(cmsgbuf));

    iov[0].iov_base = buf;
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

    return sendmsg(socket, &msg, 0) >= 0;
}

int recvFd(int socket)
{
    struct msghdr msg = {};
    struct iovec iov[1];
    char buf[1];
    char cmsgbuf[CMSG_SPACE(sizeof(int))];
    std::memset(cmsgbuf, 0, sizeof(cmsgbuf));

    iov[0].iov_base = buf;
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);

    if (recvmsg(socket, &msg, 0) < 0)
    {
        return -1;
    }

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
    {
        int fd;
        std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
        return fd;
    }
    return -1;
}

std::vector<int> recvFds(int socket, size_t count)
{
    std::vector<int> fds;
    fds.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        int fd = recvFd(socket);
        if (fd < 0)
        {
            for (int openFd : fds)
            {
                ::close(openFd);
            }
            return {};
        }
        fds.push_back(fd);
    }
    return fds;
}

int createUdsServer(char const* path)
{
    ::unlink(path);
    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0)
    {
        return -1;
    }
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (::bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        ::close(sock);
        return -1;
    }
    if (::listen(sock, 5) < 0)
    {
        ::close(sock);
        return -1;
    }
    return sock;
}

int connectUds(char const* path, int maxRetries = 50)
{
    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0)
    {
        return -1;
    }
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    for (int i = 0; i < maxRetries; ++i)
    {
        if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0)
        {
            return sock;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ::close(sock);
    return -1;
}

std::string generateUdsPath()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t randomId = dist(gen);
    return "/tmp/trt_llm_p2p_fd_" + std::to_string(::getpid()) + "_" + std::to_string(randomId) + ".sock";
}

} // anonymous namespace

// ============================================================================
// Serialization of P2pMemChunk / P2pMemPool / P2pMemInfo
// ============================================================================

void P2pMemChunk::serialize(std::ostream& os) const
{
    os.write(reinterpret_cast<char const*>(&virtAddrOffset), sizeof(virtAddrOffset));
    os.write(reinterpret_cast<char const*>(&size), sizeof(size));
    os.write(reinterpret_cast<char const*>(fabricHandle), sizeof(fabricHandle));
}

P2pMemChunk P2pMemChunk::deserialize(std::istream& is)
{
    P2pMemChunk chunk;
    is.read(reinterpret_cast<char*>(&chunk.virtAddrOffset), sizeof(chunk.virtAddrOffset));
    is.read(reinterpret_cast<char*>(&chunk.size), sizeof(chunk.size));
    is.read(reinterpret_cast<char*>(chunk.fabricHandle), sizeof(chunk.fabricHandle));
    return chunk;
}

void P2pMemPool::serialize(std::ostream& os) const
{
    os.write(reinterpret_cast<char const*>(&deviceId), sizeof(deviceId));
    os.write(reinterpret_cast<char const*>(&poolBaseAddr), sizeof(poolBaseAddr));
    os.write(reinterpret_cast<char const*>(&poolTotalSize), sizeof(poolTotalSize));
    os.write(reinterpret_cast<char const*>(&registeredAddr), sizeof(registeredAddr));
    os.write(reinterpret_cast<char const*>(&registeredSize), sizeof(registeredSize));
    os.write(reinterpret_cast<char const*>(&mappedOffset), sizeof(mappedOffset));
    os.write(reinterpret_cast<char const*>(&mappedSize), sizeof(mappedSize));
    uint64_t numChunks = chunks.size();
    os.write(reinterpret_cast<char const*>(&numChunks), sizeof(numChunks));
    for (auto const& chunk : chunks)
    {
        chunk.serialize(os);
    }
}

P2pMemPool P2pMemPool::deserialize(std::istream& is)
{
    P2pMemPool pool;
    is.read(reinterpret_cast<char*>(&pool.deviceId), sizeof(pool.deviceId));
    is.read(reinterpret_cast<char*>(&pool.poolBaseAddr), sizeof(pool.poolBaseAddr));
    is.read(reinterpret_cast<char*>(&pool.poolTotalSize), sizeof(pool.poolTotalSize));
    is.read(reinterpret_cast<char*>(&pool.registeredAddr), sizeof(pool.registeredAddr));
    is.read(reinterpret_cast<char*>(&pool.registeredSize), sizeof(pool.registeredSize));
    is.read(reinterpret_cast<char*>(&pool.mappedOffset), sizeof(pool.mappedOffset));
    is.read(reinterpret_cast<char*>(&pool.mappedSize), sizeof(pool.mappedSize));
    uint64_t numChunks;
    is.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));
    constexpr uint64_t kMaxChunks = 1000000;
    if (numChunks > kMaxChunks)
    {
        is.setstate(std::ios::failbit);
        return pool;
    }
    pool.chunks.reserve(numChunks);
    for (uint64_t i = 0; i < numChunks; ++i)
    {
        pool.chunks.push_back(P2pMemChunk::deserialize(is));
    }
    return pool;
}

std::string P2pMemInfo::serialize() const
{
    std::ostringstream oss;
    oss.write(reinterpret_cast<char const*>(&kMagic), sizeof(kMagic));
    oss.write(reinterpret_cast<char const*>(&kVersion), sizeof(kVersion));
    uint8_t supportedFlag = supported ? 1 : 0;
    oss.write(reinterpret_cast<char const*>(&supportedFlag), sizeof(supportedFlag));
    uint8_t handleTypeVal = static_cast<uint8_t>(handleType);
    oss.write(reinterpret_cast<char const*>(&handleTypeVal), sizeof(handleTypeVal));
    uint32_t udsPathLen = static_cast<uint32_t>(udsPath.size());
    oss.write(reinterpret_cast<char const*>(&udsPathLen), sizeof(udsPathLen));
    if (udsPathLen > 0)
    {
        oss.write(udsPath.data(), udsPathLen);
    }
    uint64_t numPools = pools.size();
    oss.write(reinterpret_cast<char const*>(&numPools), sizeof(numPools));
    for (auto const& pool : pools)
    {
        pool.serialize(oss);
    }
    return oss.str();
}

std::optional<P2pMemInfo> P2pMemInfo::deserialize(std::string_view data)
{
    if (data.size() < sizeof(uint32_t) * 2 + sizeof(uint8_t) * 2 + sizeof(uint32_t))
    {
        return std::nullopt;
    }
    std::istringstream iss{std::string{data}};
    uint32_t magic, version;
    iss.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    iss.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != P2pMemInfo::kMagic || version != P2pMemInfo::kVersion)
    {
        return std::nullopt;
    }

    P2pMemInfo info;
    uint8_t supportedFlag;
    iss.read(reinterpret_cast<char*>(&supportedFlag), sizeof(supportedFlag));
    info.supported = (supportedFlag != 0);

    uint8_t handleTypeVal;
    iss.read(reinterpret_cast<char*>(&handleTypeVal), sizeof(handleTypeVal));
    info.handleType = static_cast<VmmHandleType>(handleTypeVal);

    uint32_t udsPathLen;
    iss.read(reinterpret_cast<char*>(&udsPathLen), sizeof(udsPathLen));
    if (udsPathLen > 0)
    {
        info.udsPath.resize(udsPathLen);
        iss.read(info.udsPath.data(), udsPathLen);
    }

    uint64_t numPools;
    iss.read(reinterpret_cast<char*>(&numPools), sizeof(numPools));
    constexpr uint64_t kMaxPools = 100000;
    if (numPools > kMaxPools)
    {
        return std::nullopt;
    }
    info.pools.reserve(numPools);
    for (uint64_t i = 0; i < numPools; ++i)
    {
        info.pools.push_back(P2pMemPool::deserialize(iss));
    }
    if (!iss)
    {
        return std::nullopt;
    }
    return info;
}

// ============================================================================
// BatchCopyWorkerPool
// ============================================================================

BatchCopyWorkerPool::BatchCopyWorkerPool(int numWorkers, int cudaDevice)
{
    for (int i = 0; i < numWorkers; ++i)
    {
        mWorkers.emplace_back(
            [this, cudaDevice]()
            {
                TLLM_CUDA_CHECK(cudaSetDevice(cudaDevice));
                workerLoop();
            });
    }
}

BatchCopyWorkerPool::~BatchCopyWorkerPool()
{
    {
        std::lock_guard<std::mutex> lk(mMutex);
        mShutdown = true;
    }
    mCv.notify_all();
    for (auto& w : mWorkers)
    {
        w.join();
    }
}

void BatchCopyWorkerPool::submit(BatchCopyTask&& task)
{
    {
        std::lock_guard<std::mutex> lk(mMutex);
        mQueue.push(std::move(task));
        mPending.fetch_add(1, std::memory_order_relaxed);
    }
    mCv.notify_one();
}

bool BatchCopyWorkerPool::isDone() const
{
    return mPending.load(std::memory_order_acquire) == 0;
}

void BatchCopyWorkerPool::waitAll()
{
    std::unique_lock<std::mutex> lk(mMutex);
    mDoneCv.wait(lk, [this]() { return mPending.load(std::memory_order_relaxed) == 0; });
}

void BatchCopyWorkerPool::workerLoop()
{
    while (true)
    {
        BatchCopyTask task;
        {
            std::unique_lock<std::mutex> lk(mMutex);
            mCv.wait(lk, [this]() { return mShutdown || !mQueue.empty(); });
            if (mShutdown && mQueue.empty())
            {
                return;
            }
            task = std::move(mQueue.front());
            mQueue.pop();
        }

        size_t count = task.dst.size();
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;

        std::vector<size_t> idx(count, 0);
        TLLM_CUDA_CHECK(cudaMemcpyBatchAsync(
            task.dst.data(), task.src.data(), task.sizes.data(), count, &attr, idx.data(), 1, task.stream));

        if (task.completionEvent)
        {
            TLLM_CUDA_CHECK(cudaEventRecord(task.completionEvent->get(), task.stream));
            task.completionEvent.reset();
        }

        if (task.batchPending)
        {
            task.batchPending->fetch_sub(1, std::memory_order_release);
            task.batchPending.reset();
        }

        if (mPending.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            mDoneCv.notify_all();
        }
    }
}

// ============================================================================
// CudaEventPool
// ============================================================================

std::shared_ptr<runtime::CudaEvent> CudaEventPool::acquire()
{
    std::unique_ptr<runtime::CudaEvent> event;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mFreeEvents.empty())
        {
            event = std::move(mFreeEvents.back());
            mFreeEvents.pop_back();
        }
    }
    if (!event)
    {
        event = std::make_unique<runtime::CudaEvent>();
    }

    auto weak = weak_from_this();
    auto* rawPtr = event.release();
    return std::shared_ptr<runtime::CudaEvent>(rawPtr,
        [weak](runtime::CudaEvent* e)
        {
            auto pool = weak.lock();
            if (pool)
            {
                pool->release(e);
            }
            else
            {
                delete e;
            }
        });
}

void CudaEventPool::release(runtime::CudaEvent* event)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mFreeEvents.push_back(std::unique_ptr<runtime::CudaEvent>(event));
}

// ============================================================================
// P2pTransferStatus
// ============================================================================

P2pTransferStatus::P2pTransferStatus(
    std::shared_ptr<runtime::CudaStream> stream, std::shared_ptr<runtime::CudaEvent> completionEvent)
    : mStream(std::move(stream))
    , mCompletionEvent(std::move(completionEvent))
{
}

P2pTransferStatus::P2pTransferStatus(
    std::shared_ptr<std::atomic<int>> batchPending, std::vector<std::shared_ptr<runtime::CudaEvent>> workerEvents)
    : mBatchPending(std::move(batchPending))
    , mWorkerEvents(std::move(workerEvents))
{
}

bool P2pTransferStatus::isCompleted() const
{
    if (mCompleted.load(std::memory_order_acquire))
    {
        return true;
    }
    if (mBatchPending)
    {
        if (mBatchPending->load(std::memory_order_acquire) > 0)
        {
            return false;
        }
        for (auto const& event : mWorkerEvents)
        {
            auto result = cudaEventQuery(event->get());
            if (result == cudaErrorNotReady)
            {
                return false;
            }
            if (result != cudaSuccess)
            {
                TLLM_LOG_ERROR("P2pTransfer: cudaEventQuery returned error %d (%s)", static_cast<int>(result),
                    cudaGetErrorString(result));
                return false;
            }
        }
        mCompleted.store(true, std::memory_order_release);
        return true;
    }

    auto result = cudaEventQuery(mCompletionEvent->get());
    if (result == cudaSuccess)
    {
        mCompleted.store(true, std::memory_order_release);
        return true;
    }
    if (result != cudaErrorNotReady)
    {
        TLLM_LOG_ERROR(
            "P2pTransfer: cudaEventQuery returned error %d (%s)", static_cast<int>(result), cudaGetErrorString(result));
    }
    return false;
}

TransferState P2pTransferStatus::wait(int64_t timeout_ms) const
{
    if (mCompleted.load(std::memory_order_acquire))
    {
        return TransferState::kSUCCESS;
    }

    auto const startTime = std::chrono::steady_clock::now();
    auto timedOut = [&]()
    {
        if (timeout_ms < 0)
        {
            return false;
        }
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        return elapsed >= timeout_ms;
    };

    if (mBatchPending)
    {
        // Phase 1: wait until every worker has recorded its completionEvent (batchPending hits 0).
        // batchPending is decremented with release ordering after cudaEventRecord, so an acquire load
        // of 0 here guarantees every event is visible for query/sync below.
        while (mBatchPending->load(std::memory_order_acquire) > 0)
        {
            if (timedOut())
            {
                return TransferState::kIN_PROGRESS;
            }
            std::this_thread::yield();
        }

        if (timeout_ms < 0)
        {
            for (auto const& event : mWorkerEvents)
            {
                event->synchronize();
            }
            mCompleted.store(true, std::memory_order_release);
            return TransferState::kSUCCESS;
        }

        while (true)
        {
            bool allDone = true;
            for (auto const& event : mWorkerEvents)
            {
                auto result = cudaEventQuery(event->get());
                if (result == cudaErrorNotReady)
                {
                    allDone = false;
                    break;
                }
                if (result != cudaSuccess)
                {
                    TLLM_LOG_ERROR("P2pTransfer: cudaEventQuery returned error %d (%s)", static_cast<int>(result),
                        cudaGetErrorString(result));
                    return TransferState::kFAILURE;
                }
            }
            if (allDone)
            {
                mCompleted.store(true, std::memory_order_release);
                return TransferState::kSUCCESS;
            }

            if (timedOut())
            {
                return TransferState::kIN_PROGRESS;
            }
            std::this_thread::yield();
        }
    }

    if (timeout_ms < 0)
    {
        mCompletionEvent->synchronize();
        mCompleted.store(true, std::memory_order_release);
        return TransferState::kSUCCESS;
    }

    while (true)
    {
        auto result = cudaEventQuery(mCompletionEvent->get());
        if (result == cudaSuccess)
        {
            mCompleted.store(true, std::memory_order_release);
            return TransferState::kSUCCESS;
        }
        if (result != cudaErrorNotReady)
        {
            TLLM_LOG_ERROR("P2pTransfer: cudaEventQuery returned error %d (%s)", static_cast<int>(result),
                cudaGetErrorString(result));
            return TransferState::kFAILURE;
        }
        if (timedOut())
        {
            return TransferState::kIN_PROGRESS;
        }
        std::this_thread::yield();
    }
}

// ============================================================================
// MixedTransferStatus
// ============================================================================
//
// Shape the timeout split naively: spend all remaining time on P2P first (typically
// the faster half because it is just a CUDA stream sync), then the remainder on NIXL.
// Callers in the KV transfer path pass timeout_ms = -1 (wait indefinitely) in the vast
// majority of cases, so the split rarely matters.

bool MixedTransferStatus::isCompleted() const
{
    bool p2pDone = !mP2p || mP2p->isCompleted();
    bool nixlDone = !mNixl || mNixl->isCompleted();
    return p2pDone && nixlDone;
}

TransferState MixedTransferStatus::wait(int64_t timeout_ms) const
{
    auto const start = std::chrono::steady_clock::now();

    TransferState p2pResult = mP2p ? mP2p->wait(timeout_ms) : TransferState::kSUCCESS;
    if (p2pResult == TransferState::kFAILURE || p2pResult == TransferState::kIN_PROGRESS)
    {
        // Short-circuit: don't wait on NIXL if P2P already timed out or failed.
        return p2pResult;
    }

    int64_t remaining = timeout_ms;
    if (timeout_ms >= 0)
    {
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        remaining = std::max<int64_t>(0, timeout_ms - elapsed);
    }

    TransferState nixlResult = mNixl ? mNixl->wait(remaining) : TransferState::kSUCCESS;
    if (nixlResult == TransferState::kFAILURE)
    {
        return TransferState::kFAILURE;
    }
    if (nixlResult == TransferState::kIN_PROGRESS)
    {
        return TransferState::kIN_PROGRESS;
    }
    return TransferState::kSUCCESS;
}

// ============================================================================
// P2pHandleExporter
// ============================================================================

P2pHandleExporter::P2pHandleExporter(CUdevice localDevice)
    : mLocalDevice(localDevice)
{
}

P2pHandleExporter::~P2pHandleExporter()
{
    stopUdsServer();
    for (int fd : mExportedFds)
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }
    mExportedFds.clear();
}

size_t P2pHandleExporter::getVmmGranularity() const
{
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<int>(mLocalDevice);

    size_t granularity = 0;
    auto err = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (err != CUDA_SUCCESS || granularity == 0)
    {
        granularity = 2 * 1024 * 1024;
    }
    return granularity;
}

void P2pHandleExporter::detectAndExportChunks(CUdeviceptr scanStart, size_t scanSize, CUdeviceptr poolBase,
    size_t poolTotalSize, std::vector<P2pMemChunk>& chunks)
{
    if (mDetectedHandleType == VmmHandleType::kCudaIpc)
    {
        exportSingleChunkCudaIpc(poolBase, poolTotalSize, chunks);
        return;
    }

    CUdeviceptr current = scanStart;
    CUdeviceptr scanEnd = scanStart + scanSize;
    while (current < scanEnd)
    {
        CUdeviceptr chunkBase;
        size_t chunkSize;
        CUresult err = cuMemGetAddressRange(&chunkBase, &chunkSize, current);
        if (err != CUDA_SUCCESS)
        {
            size_t granularity = getVmmGranularity();
            CUdeviceptr nextAligned = ((current / granularity) + 1) * granularity;
            current = nextAligned;
            continue;
        }

        if (mDetectedHandleType == VmmHandleType::kPosixFd)
        {
            exportSingleChunkPosixFd(chunkBase, chunkSize, poolBase, chunks);
        }
        else
        {
            exportSingleChunkFabric(chunkBase, chunkSize, poolBase, chunks);
        }

        current = chunkBase + chunkSize;
    }
}

void P2pHandleExporter::exportSingleChunkFabric(
    CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, std::vector<P2pMemChunk>& chunks)
{
    uint64_t offset = chunkBase - poolBase;
    for (auto const& existing : chunks)
    {
        if (existing.virtAddrOffset == offset)
        {
            return;
        }
    }

    CUmemGenericAllocationHandle allocHandle;
    auto err = cuMemRetainAllocationHandle(&allocHandle, reinterpret_cast<void*>(chunkBase));
    if (err != CUDA_SUCCESS)
    {
        return;
    }

    P2pMemChunk chunk;
    chunk.virtAddrOffset = offset;
    chunk.size = chunkSize;

    err = cuMemExportToShareableHandle(
        reinterpret_cast<void*>(chunk.fabricHandle), allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(allocHandle));

    if (err == CUDA_SUCCESS)
    {
        TLLM_LOG_DEBUG("P2pTransfer: exported fabric chunk offset=%lu size=%zu", chunk.virtAddrOffset, chunk.size);
        chunks.push_back(std::move(chunk));
    }
}

void P2pHandleExporter::exportSingleChunkPosixFd(
    CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, std::vector<P2pMemChunk>& chunks)
{
    uint64_t offset = chunkBase - poolBase;
    for (auto const& existing : chunks)
    {
        if (existing.virtAddrOffset == offset)
        {
            return;
        }
    }

    CUmemGenericAllocationHandle allocHandle;
    auto err = cuMemRetainAllocationHandle(&allocHandle, reinterpret_cast<void*>(chunkBase));
    if (err != CUDA_SUCCESS)
    {
        return;
    }

    int fd = -1;
    err = cuMemExportToShareableHandle(&fd, allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(allocHandle));

    if (err != CUDA_SUCCESS)
    {
        TLLM_LOG_WARNING("P2pTransfer: failed to export POSIX FD for chunk at 0x%lx, error=%d", chunkBase, err);
        return;
    }

    P2pMemChunk chunk;
    chunk.virtAddrOffset = offset;
    chunk.size = chunkSize;
    std::memset(chunk.fabricHandle, 0, sizeof(chunk.fabricHandle));

    TLLM_LOG_DEBUG(
        "P2pTransfer: exported POSIX FD=%d for chunk offset=%lu size=%zu", fd, chunk.virtAddrOffset, chunk.size);

    chunks.push_back(std::move(chunk));

    {
        std::lock_guard<std::mutex> fdsLock(mExportedFdsMutex);
        mExportedFds.push_back(fd);
    }
}

void P2pHandleExporter::exportSingleChunkCudaIpc(
    CUdeviceptr poolBase, size_t poolTotalSize, std::vector<P2pMemChunk>& chunks)
{
    if (!chunks.empty())
    {
        return;
    }

    cudaIpcMemHandle_t ipcHandle;
    auto err = cudaIpcGetMemHandle(&ipcHandle, reinterpret_cast<void*>(poolBase));
    if (err != cudaSuccess)
    {
        TLLM_LOG_WARNING(
            "P2pTransfer: failed to get CUDA IPC handle for ptr=0x%lx, error=%d", poolBase, static_cast<int>(err));
        return;
    }

    P2pMemChunk chunk;
    chunk.virtAddrOffset = 0;
    chunk.size = poolTotalSize;
    static_assert(sizeof(cudaIpcMemHandle_t) == sizeof(chunk.fabricHandle),
        "cudaIpcMemHandle_t must be 64 bytes to fit in fabricHandle");
    std::memcpy(chunk.fabricHandle, &ipcHandle, sizeof(ipcHandle));

    TLLM_LOG_DEBUG("P2pTransfer: exported CudaIpc handle for pool at 0x%lx, size=%zu", poolBase, poolTotalSize);
    chunks.push_back(std::move(chunk));
}

void P2pHandleExporter::exportHandles(RegisterDescs const& descs)
{
    TLLM_CUDA_CHECK(cudaSetDevice(static_cast<int>(mLocalDevice)));

    std::unordered_map<uint64_t, size_t> processedPools;
    for (size_t i = 0; i < mLocalInfo.pools.size(); ++i)
    {
        processedPools.emplace(mLocalInfo.pools[i].poolBaseAddr, i);
    }

    for (auto const& desc : descs.getDescs())
    {
        CUdeviceptr ptr = static_cast<CUdeviceptr>(desc.getAddr());
        size_t descSize = desc.getLen();

        int legacy_capable = 0;
        unsigned long long allowed_handle_types = 0;

        CUpointer_attribute attr_type[2]
            = {CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES};
        void* attr_data[2] = {&legacy_capable, &allowed_handle_types};

        auto err = cuPointerGetAttributes(2, attr_type, attr_data, ptr);
        if (err != CUDA_SUCCESS)
        {
            continue;
        }

        VmmHandleType descHandleType;
        if (legacy_capable)
        {
            descHandleType = VmmHandleType::kCudaIpc;
        }
        else
        {
            bool fabricSupported = (allowed_handle_types & CU_MEM_HANDLE_TYPE_FABRIC) != 0;
            bool posixFdSupported = (allowed_handle_types & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != 0;
            if (!fabricSupported && !posixFdSupported)
            {
                continue;
            }
            descHandleType = fabricSupported ? VmmHandleType::kFabric : VmmHandleType::kPosixFd;
        }

        if (mDetectedHandleType == VmmHandleType::kNone)
        {
            mDetectedHandleType = descHandleType;
        }
        else if (mDetectedHandleType != descHandleType)
        {
            TLLM_LOG_WARNING("P2pTransfer: mixed handle types detected (existing=%d new=%d), skipping desc",
                static_cast<int>(mDetectedHandleType), static_cast<int>(descHandleType));
            continue;
        }

        CUdeviceptr basePtr;
        size_t totalSize;
        err = cuMemGetAddressRange(&basePtr, &totalSize, ptr);
        if (err != CUDA_SUCCESS)
        {
            continue;
        }

        if (descSize > 1)
        {
            CUdeviceptr endBasePtr;
            size_t endTotalSize;
            CUresult endErr = cuMemGetAddressRange(&endBasePtr, &endTotalSize, ptr + descSize - 1);
            if (endErr == CUDA_SUCCESS)
            {
                CUdeviceptr poolStart = std::min(basePtr, endBasePtr);
                CUdeviceptr poolEndAddr = std::max(basePtr + totalSize, endBasePtr + endTotalSize);
                basePtr = poolStart;
                totalSize = poolEndAddr - poolStart;
            }
        }

        auto poolIt = processedPools.find(basePtr);
        if (poolIt != processedPools.end())
        {
            P2pMemPool& existingPool = mLocalInfo.pools[poolIt->second];
            uint64_t existingStart = existingPool.registeredAddr;
            uint64_t existingEnd = existingStart + existingPool.registeredSize;
            uint64_t newStart = ptr;
            uint64_t newEnd = ptr + descSize;

            if (newStart < existingStart)
            {
                size_t extendSize = std::min(existingStart, newEnd) - newStart;
                detectAndExportChunks(newStart, extendSize, basePtr, existingPool.poolTotalSize, existingPool.chunks);
            }
            if (newEnd > existingEnd)
            {
                CUdeviceptr extendStart = std::max(existingEnd, newStart);
                size_t extendSize = newEnd - extendStart;
                detectAndExportChunks(
                    extendStart, extendSize, basePtr, existingPool.poolTotalSize, existingPool.chunks);
            }

            existingPool.registeredAddr = std::min(existingStart, newStart);
            existingPool.registeredSize = std::max(existingEnd, newEnd) - existingPool.registeredAddr;

            uint64_t minOffset = UINT64_MAX;
            uint64_t maxEnd = 0;
            for (auto const& chunk : existingPool.chunks)
            {
                minOffset = std::min(minOffset, chunk.virtAddrOffset);
                maxEnd = std::max(maxEnd, chunk.virtAddrOffset + chunk.size);
            }
            existingPool.mappedOffset = minOffset;
            existingPool.mappedSize = maxEnd - minOffset;

            TLLM_LOG_DEBUG(
                "P2pTransfer: pool at 0x%lx updated, chunks=%zu registered=[0x%lx, 0x%lx) mapped=[0x%lx, 0x%lx)",
                basePtr, existingPool.chunks.size(), existingPool.registeredAddr,
                existingPool.registeredAddr + existingPool.registeredSize,
                existingPool.poolBaseAddr + existingPool.mappedOffset,
                existingPool.poolBaseAddr + existingPool.mappedOffset + existingPool.mappedSize);
            continue;
        }

        P2pMemPool pool;
        pool.deviceId = desc.getDeviceId();
        pool.poolBaseAddr = basePtr;
        pool.poolTotalSize = totalSize;
        pool.registeredAddr = ptr;
        pool.registeredSize = descSize;

        detectAndExportChunks(ptr, descSize, basePtr, totalSize, pool.chunks);

        if (!pool.chunks.empty())
        {
            uint64_t minOffset = UINT64_MAX;
            uint64_t maxEnd = 0;
            for (auto const& chunk : pool.chunks)
            {
                minOffset = std::min(minOffset, chunk.virtAddrOffset);
                maxEnd = std::max(maxEnd, chunk.virtAddrOffset + chunk.size);
            }
            pool.mappedOffset = minOffset;
            pool.mappedSize = maxEnd - minOffset;

            TLLM_LOG_DEBUG(
                "P2pTransfer: detected pool at 0x%lx chunks=%zu registered=[0x%lx, 0x%lx) mapped=[0x%lx, 0x%lx)",
                pool.poolBaseAddr, pool.chunks.size(), pool.registeredAddr, pool.registeredAddr + pool.registeredSize,
                pool.poolBaseAddr + pool.mappedOffset, pool.poolBaseAddr + pool.mappedOffset + pool.mappedSize);
            processedPools[basePtr] = mLocalInfo.pools.size();
            mLocalInfo.pools.push_back(std::move(pool));
        }
    }

    mLocalInfo.supported = !mLocalInfo.pools.empty();
    mLocalInfo.handleType = mDetectedHandleType;

    if (mLocalInfo.supported)
    {
        TLLM_LOG_INFO("P2pTransfer: enabled with %zu pools handleType=%s", mLocalInfo.pools.size(),
            handleTypeToString(mDetectedHandleType));
        if (mDetectedHandleType == VmmHandleType::kPosixFd)
        {
            startUdsServer();
            mLocalInfo.udsPath = mUdsPath;
        }
    }
}

void P2pHandleExporter::removeHandles(RegisterDescs const& descs)
{
    TLLM_CUDA_CHECK(cudaSetDevice(static_cast<int>(mLocalDevice)));

    if (mLocalInfo.pools.empty())
    {
        return;
    }

    std::unordered_set<uint64_t> poolBasesToRemove;
    for (auto const& desc : descs.getDescs())
    {
        CUdeviceptr ptr = static_cast<CUdeviceptr>(desc.getAddr());
        CUdeviceptr basePtr;
        size_t totalSize;
        auto err = cuMemGetAddressRange(&basePtr, &totalSize, ptr);
        if (err != CUDA_SUCCESS)
        {
            continue;
        }
        poolBasesToRemove.insert(basePtr);
    }

    if (poolBasesToRemove.empty())
    {
        return;
    }

    bool isPosixFdMode = (mDetectedHandleType == VmmHandleType::kPosixFd);
    for (auto it = mLocalInfo.pools.begin(); it != mLocalInfo.pools.end();)
    {
        if (poolBasesToRemove.find(it->poolBaseAddr) != poolBasesToRemove.end())
        {
            if (isPosixFdMode)
            {
                std::lock_guard<std::mutex> fdsLock(mExportedFdsMutex);
                if (!mExportedFds.empty())
                {
                    size_t fdOffset = 0;
                    for (auto prev = mLocalInfo.pools.begin(); prev != it; ++prev)
                    {
                        fdOffset += prev->chunks.size();
                    }
                    size_t fdCount = it->chunks.size();
                    size_t fdEnd = std::min(fdOffset + fdCount, mExportedFds.size());
                    for (size_t i = fdOffset; i < fdEnd; ++i)
                    {
                        if (mExportedFds[i] >= 0)
                        {
                            ::close(mExportedFds[i]);
                        }
                    }
                    if (fdOffset < mExportedFds.size())
                    {
                        mExportedFds.erase(mExportedFds.begin() + static_cast<ptrdiff_t>(fdOffset),
                            mExportedFds.begin() + static_cast<ptrdiff_t>(fdEnd));
                    }
                }
            }

            TLLM_LOG_DEBUG("P2pTransfer: removed pool at base=0x%lx registered=[0x%lx, +%zu)", it->poolBaseAddr,
                it->registeredAddr, it->registeredSize);
            it = mLocalInfo.pools.erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (mLocalInfo.pools.empty())
    {
        mLocalInfo.supported = false;
        if (isPosixFdMode)
        {
            stopUdsServer();
            for (int fd : mExportedFds)
            {
                if (fd >= 0)
                {
                    ::close(fd);
                }
            }
            mExportedFds.clear();
        }
        TLLM_LOG_INFO("P2pTransfer: all pools removed, disabled");
    }
    else
    {
        TLLM_LOG_DEBUG("P2pTransfer: %zu pools remaining after removal", mLocalInfo.pools.size());
    }
}

void P2pHandleExporter::startUdsServer()
{
    if (mExportedFds.empty())
    {
        return;
    }

    // Idempotent: exportHandles() runs this on every successful registerMemory call,
    // but the server only needs to be started once per exporter. Reassigning the
    // std::thread while the previous one is still joinable would trigger std::terminate.
    // The already-running server reads mExportedFds under mExportedFdsMutex, so new FDs
    // added by a later registerMemory are served automatically — no restart required.
    if (mUdsServerRunning.load(std::memory_order_acquire))
    {
        return;
    }

    mUdsPath = generateUdsPath();
    mUdsServerSocket = createUdsServer(mUdsPath.c_str());
    if (mUdsServerSocket < 0)
    {
        TLLM_LOG_ERROR("P2pTransfer: failed to create UDS server at %s", mUdsPath.c_str());
        mUdsPath.clear();
        return;
    }

    mUdsServerRunning.store(true, std::memory_order_release);

    mUdsServerThread = std::thread(
        [this]()
        {
            TLLM_LOG_DEBUG("P2pTransfer: UDS server started at %s with %zu FDs", mUdsPath.c_str(), mExportedFds.size());

            while (mUdsServerRunning.load(std::memory_order_acquire))
            {
                struct pollfd pfd;
                pfd.fd = mUdsServerSocket;
                pfd.events = POLLIN;
                pfd.revents = 0;

                int ret = ::poll(&pfd, 1, 1000);
                if (ret <= 0)
                {
                    continue;
                }

                int clientSocket = ::accept(mUdsServerSocket, nullptr, nullptr);
                if (clientSocket < 0)
                {
                    if (mUdsServerRunning.load(std::memory_order_acquire))
                    {
                        TLLM_LOG_WARNING("P2pTransfer: UDS accept failed errno=%d", errno);
                    }
                    continue;
                }

                uint32_t numExpected = 0;
                ssize_t bytesRead = ::read(clientSocket, &numExpected, sizeof(numExpected));
                if (bytesRead != sizeof(numExpected))
                {
                    TLLM_LOG_WARNING("P2pTransfer: UDS client protocol error (read numExpected)");
                    ::close(clientSocket);
                    continue;
                }

                std::lock_guard<std::mutex> fdsLock(mExportedFdsMutex);

                uint32_t numAvailable = static_cast<uint32_t>(mExportedFds.size());
                ssize_t bytesWritten = ::write(clientSocket, &numAvailable, sizeof(numAvailable));
                if (bytesWritten != sizeof(numAvailable))
                {
                    TLLM_LOG_WARNING("P2pTransfer: UDS client protocol error (write numAvailable)");
                    ::close(clientSocket);
                    continue;
                }

                uint32_t numToSend = std::min(numExpected, numAvailable);
                bool sendOk = true;
                for (uint32_t i = 0; i < numToSend; ++i)
                {
                    if (!sendFd(clientSocket, mExportedFds[i]))
                    {
                        TLLM_LOG_WARNING("P2pTransfer: UDS failed to send FD[%u]", i);
                        sendOk = false;
                        break;
                    }
                }

                if (sendOk)
                {
                    TLLM_LOG_DEBUG("P2pTransfer: UDS sent %u FDs to client", numToSend);
                }

                ::close(clientSocket);
            }

            TLLM_LOG_DEBUG("P2pTransfer: UDS server stopped");
        });
}

void P2pHandleExporter::stopUdsServer()
{
    if (!mUdsServerRunning.load(std::memory_order_acquire))
    {
        return;
    }
    mUdsServerRunning.store(false, std::memory_order_release);
    if (mUdsServerSocket >= 0)
    {
        ::close(mUdsServerSocket);
        mUdsServerSocket = -1;
    }
    if (mUdsServerThread.joinable())
    {
        mUdsServerThread.join();
    }
    if (!mUdsPath.empty())
    {
        ::unlink(mUdsPath.c_str());
    }
}

// ============================================================================
// P2pRemoteMappingRegistry
// ============================================================================

namespace
{

void releaseRemoteP2pMapping(RemoteP2pMapping& mapping)
{
    if (mapping.handleType == VmmHandleType::kCudaIpc)
    {
        for (auto& poolMapping : mapping.pools)
        {
            if (poolMapping.localVirtAddr != 0)
            {
                TLLM_CUDA_CHECK_FREE_RESOURCE(
                    cudaIpcCloseMemHandle(reinterpret_cast<void*>(poolMapping.localVirtAddr)));
            }
        }
        return;
    }

    for (auto& poolMapping : mapping.pools)
    {
        if (poolMapping.localVirtAddr != 0)
        {
            TLLM_CU_CHECK_FREE_RESOURCE(cuMemUnmap(poolMapping.localVirtAddr, poolMapping.mappedSize));
        }
        for (auto handle : poolMapping.importedHandles)
        {
            TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(handle));
        }
        if (poolMapping.localVirtAddr != 0)
        {
            TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(poolMapping.localVirtAddr, poolMapping.mappedSize));
        }
    }
}

} // namespace

P2pRemoteMappingRegistry::P2pRemoteMappingRegistry(CUdevice localDevice)
    : mLocalDevice(localDevice)
{
}

P2pRemoteMappingRegistry::~P2pRemoteMappingRegistry()
{
    TLLM_CUDA_CHECK_FREE_RESOURCE(cudaSetDevice(static_cast<int>(mLocalDevice)));
    for (auto& [name, mappingPtr] : mMappings)
    {
        releaseRemoteP2pMapping(*mappingPtr);
    }
    mMappings.clear();
}

bool P2pRemoteMappingRegistry::hasMapping(std::string const& name) const
{
    std::shared_lock lock(mMutex);
    return mMappings.find(name) != mMappings.end() && mFailed.find(name) == mFailed.end();
}

bool P2pRemoteMappingRegistry::hasImportFailed(std::string const& name) const
{
    std::shared_lock lock(mMutex);
    return mFailed.find(name) != mFailed.end();
}

std::shared_ptr<RemoteP2pMapping const> P2pRemoteMappingRegistry::get(std::string const& name) const
{
    std::shared_lock lock(mMutex);
    auto it = mMappings.find(name);
    if (it == mMappings.end())
    {
        return nullptr;
    }
    return it->second;
}

void P2pRemoteMappingRegistry::importAndMap(std::string const& name, P2pMemInfo const& info)
{
    TLLM_CUDA_CHECK(cudaSetDevice(static_cast<int>(mLocalDevice)));

    {
        std::shared_lock lock(mMutex);
        if (mFailed.find(name) != mFailed.end())
        {
            TLLM_LOG_DEBUG("P2pTransfer: skipping import for '%s' (previously failed)", name.c_str());
            return;
        }
    }

    // Receive FDs from remote UDS server (POSIX FD mode only)
    std::vector<int> receivedFds;
    if (info.handleType == VmmHandleType::kPosixFd)
    {
        if (info.udsPath.empty())
        {
            TLLM_LOG_WARNING("P2pTransfer: POSIX FD mode but no UDS path for '%s'", name.c_str());
            std::unique_lock lock(mMutex);
            mFailed.insert(name);
            return;
        }

        uint32_t totalChunks = 0;
        for (auto const& pool : info.pools)
        {
            totalChunks += static_cast<uint32_t>(pool.chunks.size());
        }

        int udsSocket = connectUds(info.udsPath.c_str(), 50);
        if (udsSocket < 0)
        {
            TLLM_LOG_WARNING(
                "P2pTransfer: failed to connect to UDS at '%s' for '%s'", info.udsPath.c_str(), name.c_str());
            std::unique_lock lock(mMutex);
            mFailed.insert(name);
            return;
        }

        ssize_t bytesWritten = ::write(udsSocket, &totalChunks, sizeof(totalChunks));
        if (bytesWritten != sizeof(totalChunks))
        {
            TLLM_LOG_WARNING("P2pTransfer: UDS protocol error (write numExpected) for '%s'", name.c_str());
            ::close(udsSocket);
            std::unique_lock lock(mMutex);
            mFailed.insert(name);
            return;
        }

        uint32_t numAvailable = 0;
        ssize_t bytesRead = ::read(udsSocket, &numAvailable, sizeof(numAvailable));
        if (bytesRead != sizeof(numAvailable))
        {
            TLLM_LOG_WARNING("P2pTransfer: UDS protocol error (read numAvailable) for '%s'", name.c_str());
            ::close(udsSocket);
            std::unique_lock lock(mMutex);
            mFailed.insert(name);
            return;
        }

        uint32_t numToRecv = std::min(totalChunks, numAvailable);
        receivedFds = recvFds(udsSocket, numToRecv);
        ::close(udsSocket);

        if (receivedFds.size() != numToRecv)
        {
            TLLM_LOG_WARNING(
                "P2pTransfer: received %zu/%u FDs from UDS for '%s'", receivedFds.size(), numToRecv, name.c_str());
            for (int fd : receivedFds)
            {
                ::close(fd);
            }
            std::unique_lock lock(mMutex);
            mFailed.insert(name);
            return;
        }

        TLLM_LOG_DEBUG("P2pTransfer: received %zu FDs from UDS for '%s'", receivedFds.size(), name.c_str());
    }

    RemoteP2pMapping mapping;
    mapping.remoteName = name;
    mapping.handleType = info.handleType;
    bool anyImportFailed = false;
    size_t fdIndex = 0;

    for (auto const& pool : info.pools)
    {
        RemoteP2pPoolMapping poolMapping;
        poolMapping.remoteBaseAddr = pool.poolBaseAddr;
        poolMapping.totalSize = pool.poolTotalSize;
        poolMapping.remoteRegisteredAddr = pool.registeredAddr;
        poolMapping.registeredSize = pool.registeredSize;
        poolMapping.remoteMappedOffset = pool.mappedOffset;
        poolMapping.mappedSize = pool.mappedSize;

        if (info.handleType == VmmHandleType::kCudaIpc)
        {
            if (pool.chunks.empty())
            {
                TLLM_LOG_WARNING("P2pTransfer: CudaIpc pool has no chunks for '%s'", name.c_str());
                anyImportFailed = true;
                continue;
            }
            auto const& chunk = pool.chunks[0];
            cudaIpcMemHandle_t ipcHandle;
            std::memcpy(&ipcHandle, chunk.fabricHandle, sizeof(ipcHandle));

            void* devPtr = nullptr;
            auto cudaErr = cudaIpcOpenMemHandle(&devPtr, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
            if (cudaErr != cudaSuccess)
            {
                TLLM_LOG_WARNING("P2pTransfer: cudaIpcOpenMemHandle failed for '%s' error=%d (%s)", name.c_str(),
                    static_cast<int>(cudaErr), cudaGetErrorString(cudaErr));
                anyImportFailed = true;
                continue;
            }

            poolMapping.localVirtAddr = reinterpret_cast<CUdeviceptr>(devPtr);
            mapping.pools.push_back(std::move(poolMapping));
            TLLM_LOG_DEBUG("P2pTransfer: CudaIpc mapped remote pool at 0x%lx -> local 0x%lx size=%zu",
                pool.poolBaseAddr, reinterpret_cast<CUdeviceptr>(devPtr), pool.poolTotalSize);
            continue;
        }

        // VMM path (Fabric / PosixFd)
        CUdeviceptr localVa;
        auto err = cuMemAddressReserve(&localVa, pool.mappedSize, 0, 0, 0);
        if (err != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("P2pTransfer: failed to reserve VA (mappedSize=%zu) error=%d", pool.mappedSize, err);
            anyImportFailed = true;
            fdIndex += pool.chunks.size();
            continue;
        }
        poolMapping.localVirtAddr = localVa;

        bool allChunksMapped = true;
        std::vector<std::pair<CUdeviceptr, size_t>> mappedRanges;
        size_t fdIndexBeforePool = fdIndex;
        for (auto const& chunk : pool.chunks)
        {
            if (chunk.virtAddrOffset < pool.mappedOffset)
            {
                TLLM_LOG_ERROR(
                    "P2pTransfer: chunk offset %lu < mapped offset %lu", chunk.virtAddrOffset, pool.mappedOffset);
                if (info.handleType == VmmHandleType::kPosixFd && fdIndex < receivedFds.size())
                {
                    ::close(receivedFds[fdIndex++]);
                }
                allChunksMapped = false;
                continue;
            }
            uint64_t localOffset = chunk.virtAddrOffset - pool.mappedOffset;

            CUmemGenericAllocationHandle importedHandle;
            if (info.handleType == VmmHandleType::kPosixFd)
            {
                if (fdIndex >= receivedFds.size())
                {
                    TLLM_LOG_ERROR(
                        "P2pTransfer: ran out of FDs (index=%zu available=%zu)", fdIndex, receivedFds.size());
                    allChunksMapped = false;
                    anyImportFailed = true;
                    break;
                }
                int fd = receivedFds[fdIndex++];
                err = cuMemImportFromShareableHandle(&importedHandle,
                    reinterpret_cast<void*>(static_cast<uintptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
                ::close(fd);
            }
            else
            {
                err = cuMemImportFromShareableHandle(&importedHandle,
                    const_cast<void*>(reinterpret_cast<void const*>(chunk.fabricHandle)), CU_MEM_HANDLE_TYPE_FABRIC);
            }

            if (err != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING("P2pTransfer: failed to import handle for '%s' type=%s error=%d", name.c_str(),
                    handleTypeToString(info.handleType), err);
                allChunksMapped = false;
                anyImportFailed = true;
                break;
            }

            err = cuMemMap(localVa + localOffset, chunk.size, 0, importedHandle, 0);
            if (err != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "P2pTransfer: failed to map remote chunk at localOffset=%lu error=%d", localOffset, err);
                TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(importedHandle));
                allChunksMapped = false;
                anyImportFailed = true;
                break;
            }
            mappedRanges.emplace_back(localVa + localOffset, chunk.size);

            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = static_cast<int>(mLocalDevice);
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            err = cuMemSetAccess(localVa + localOffset, chunk.size, &accessDesc, 1);
            if (err != CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "P2pTransfer: failed to set access for remote chunk at localOffset=%lu error=%d", localOffset, err);
                TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(importedHandle));
                allChunksMapped = false;
                anyImportFailed = true;
                break;
            }

            poolMapping.importedHandles.push_back(importedHandle);
        }

        if (allChunksMapped)
        {
            mapping.pools.push_back(std::move(poolMapping));
            TLLM_LOG_DEBUG("P2pTransfer: mapped remote pool at 0x%lx -> local 0x%lx size=%zu", pool.poolBaseAddr,
                localVa, pool.poolTotalSize);
        }
        else
        {
            if (info.handleType == VmmHandleType::kPosixFd)
            {
                size_t expectedFdsForPool = pool.chunks.size();
                size_t poolFdEnd = std::min(fdIndexBeforePool + expectedFdsForPool, receivedFds.size());
                for (size_t i = fdIndex; i < poolFdEnd; ++i)
                {
                    ::close(receivedFds[i]);
                }
                fdIndex = std::max(fdIndex, poolFdEnd);
            }
            for (auto const& [mappedAddr, mappedSize] : mappedRanges)
            {
                TLLM_CU_CHECK_FREE_RESOURCE(cuMemUnmap(mappedAddr, mappedSize));
            }
            for (auto handle : poolMapping.importedHandles)
            {
                TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(handle));
            }
            TLLM_CU_CHECK_FREE_RESOURCE(cuMemAddressFree(localVa, pool.mappedSize));
        }
    }

    for (size_t i = fdIndex; i < receivedFds.size(); ++i)
    {
        ::close(receivedFds[i]);
    }

    {
        std::unique_lock lock(mMutex);
        if (!mapping.pools.empty())
        {
            auto mappingPtr = std::make_shared<RemoteP2pMapping>(std::move(mapping));
            size_t numPools = mappingPtr->pools.size();
            mMappings[name] = std::move(mappingPtr);
            TLLM_LOG_INFO("P2pTransfer: mapping established for '%s' with %zu/%zu pools type=%s", name.c_str(),
                numPools, info.pools.size(), handleTypeToString(info.handleType));
        }
        else if (anyImportFailed)
        {
            mFailed.insert(name);
            TLLM_LOG_WARNING("P2pTransfer: import failed for '%s', will use NIXL fallback", name.c_str());
        }
    }
}

void P2pRemoteMappingRegistry::cleanup(std::string const& name)
{
    TLLM_CUDA_CHECK(cudaSetDevice(static_cast<int>(mLocalDevice)));

    std::shared_ptr<RemoteP2pMapping> mappingPtr;
    {
        std::unique_lock lock(mMutex);
        mFailed.erase(name);
        auto it = mMappings.find(name);
        if (it == mMappings.end())
        {
            return;
        }
        mappingPtr = std::move(it->second);
        mMappings.erase(it);
    }

    releaseRemoteP2pMapping(*mappingPtr);
}

void* P2pRemoteMappingRegistry::translate(RemoteP2pMapping const& mapping, uintptr_t remoteAddr, size_t transferSize)
{
    for (auto const& pool : mapping.pools)
    {
        uint64_t registeredEnd = pool.remoteRegisteredAddr + pool.registeredSize;
        if (remoteAddr >= pool.remoteRegisteredAddr && remoteAddr < registeredEnd)
        {
            uint64_t transferEnd = remoteAddr + transferSize;
            if (transferEnd > registeredEnd)
            {
                TLLM_LOG_ERROR("P2pTransfer: transfer range [0x%lx, 0x%lx) exceeds registered range [0x%lx, 0x%lx)",
                    remoteAddr, transferEnd, pool.remoteRegisteredAddr, registeredEnd);
                TLLM_THROW("Transfer range [0x%lx, 0x%lx) exceeds registered memory range [0x%lx, 0x%lx)", remoteAddr,
                    transferEnd, pool.remoteRegisteredAddr, registeredEnd);
            }
            uint64_t offsetFromPoolBase = remoteAddr - pool.remoteBaseAddr;
            if (offsetFromPoolBase < pool.remoteMappedOffset)
            {
                TLLM_LOG_ERROR("P2pTransfer: address underflow offset %lu < mappedOffset %lu for addr 0x%lx",
                    offsetFromPoolBase, pool.remoteMappedOffset, remoteAddr);
                return nullptr;
            }
            uint64_t localOffset = offsetFromPoolBase - pool.remoteMappedOffset;
            return reinterpret_cast<void*>(pool.localVirtAddr + localOffset);
        }
        if (remoteAddr >= pool.remoteBaseAddr && remoteAddr < pool.remoteBaseAddr + pool.totalSize)
        {
            TLLM_LOG_ERROR(
                "P2pTransfer: address 0x%lx within pool [0x%lx, 0x%lx) but outside registered range [0x%lx, 0x%lx)",
                remoteAddr, pool.remoteBaseAddr, pool.remoteBaseAddr + pool.totalSize, pool.remoteRegisteredAddr,
                registeredEnd);
            TLLM_THROW("Transfer address 0x%lx outside registered memory range [0x%lx, 0x%lx)", remoteAddr,
                pool.remoteRegisteredAddr, registeredEnd);
        }
    }
    return nullptr;
}

// ============================================================================
// P2pTransferContext
// ============================================================================

P2pTransferContext::P2pTransferContext(CUdevice localDevice, std::shared_ptr<CudaEventPool> eventPool,
    int batchCopyThreads, size_t multiThreadMinOps, bool cubZeroCopy, std::shared_ptr<P2pAgentCounters> counters)
    : mLocalDevice(localDevice)
    , mCubZeroCopy(cubZeroCopy)
    , mBatchCopyThreads(batchCopyThreads)
    , mMultiThreadMinOps(multiThreadMinOps)
    , mEventPool(std::move(eventPool))
    , mCounters(std::move(counters))
{
    // This may be called from a background thread where the device is not yet set.
    TLLM_CUDA_CHECK(cudaSetDevice(static_cast<int>(mLocalDevice)));
    mSubmitStream = std::make_shared<runtime::CudaStream>();
    mBufferManager = std::make_shared<runtime::BufferManager>(mSubmitStream);
}

void P2pTransferContext::ensureBuffers(size_t batchSize, size_t cubTempBytes)
{
    if (batchSize <= mMaxBatchSize && cubTempBytes <= mCubTempStorageSize)
    {
        return;
    }

    constexpr size_t kDefaultBatchSize = 8192 * 2;
    constexpr size_t kDefaultCubTempSize = 64 * 1024 * 1024;

    size_t newBatchSize = std::max(batchSize, mMaxBatchSize * 2);
    newBatchSize = std::max(newBatchSize, kDefaultBatchSize);
    size_t newCubTempSize = std::max(cubTempBytes, mCubTempStorageSize * 2);
    newCubTempSize = std::max(newCubTempSize, kDefaultCubTempSize);

    TLLM_LOG_DEBUG("P2pTransfer: reallocating prealloc buffers batch %zu -> %zu cubTemp %zu -> %zu", mMaxBatchSize,
        newBatchSize, mCubTempStorageSize, newCubTempSize);

    mSubmitStream->synchronize();

    size_t combinedSize = newBatchSize * (sizeof(void*) + sizeof(void*) + sizeof(size_t));
    mCombinedPinned = runtime::BufferManager::pinned(combinedSize);
    if (!mCubZeroCopy)
    {
        mCombinedGpu = mBufferManager->gpu(combinedSize);
    }
    else
    {
        mCombinedGpu = nullptr;
    }
    mCubTempStorage = mBufferManager->gpu(newCubTempSize);
    mMaxBatchSize = newBatchSize;
    mCubTempStorageSize = newCubTempSize;
}

void P2pTransferContext::ensureWorkerPoolAndStreams()
{
    std::call_once(mWorkerPoolInit,
        [this]
        {
            if (mBatchCopyThreads <= 1)
            {
                return;
            }
            TLLM_CUDA_CHECK(cudaSetDevice(static_cast<int>(mLocalDevice)));
            mBatchCopyStreams.reserve(mBatchCopyThreads);
            for (int i = 0; i < mBatchCopyThreads; ++i)
            {
                mBatchCopyStreams.push_back(std::make_shared<runtime::CudaStream>());
            }
            mWorkerPool = std::make_unique<BatchCopyWorkerPool>(mBatchCopyThreads, static_cast<int>(mLocalDevice));
            TLLM_LOG_INFO("P2pTransfer: per-context batch copy worker pool started with %d threads (tid=%s)",
                mBatchCopyThreads, std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())).c_str());
        });
}

std::unique_ptr<TransferStatus> P2pTransferContext::submitWithCubBatched(
    std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes)
{
    size_t numBuffers = srcPtrs.size();
    if (numBuffers == 0)
    {
        auto event = mEventPool->acquire();
        mSubmitStream->record(*event);
        return std::make_unique<P2pTransferStatus>(mSubmitStream, event);
    }

    size_t cubTempBytes = 0;
    cub::DeviceMemcpy::Batched(nullptr, cubTempBytes, static_cast<void const* const*>(nullptr),
        static_cast<void* const*>(nullptr), static_cast<size_t const*>(nullptr), numBuffers, mSubmitStream->get());

    ensureBuffers(numBuffers, cubTempBytes);

    size_t const srcBytes = numBuffers * sizeof(void*);
    size_t const dstBytes = numBuffers * sizeof(void*);
    size_t const sizesBytes = numBuffers * sizeof(size_t);
    size_t const totalBytes = srcBytes + dstBytes + sizesBytes;

    auto* pinnedBase = static_cast<uint8_t*>(mCombinedPinned->data());
    std::memcpy(pinnedBase, srcPtrs.data(), srcBytes);
    std::memcpy(pinnedBase + srcBytes, dstPtrs.data(), dstBytes);
    std::memcpy(pinnedBase + srcBytes + dstBytes, sizes.data(), sizesBytes);

    void const** cubSrcPtrs;
    void** cubDstPtrs;
    size_t* cubSizes;
    if (mCubZeroCopy)
    {
        cubSrcPtrs = reinterpret_cast<void const**>(pinnedBase);
        cubDstPtrs = reinterpret_cast<void**>(pinnedBase + srcBytes);
        cubSizes = reinterpret_cast<size_t*>(pinnedBase + srcBytes + dstBytes);
    }
    else
    {
        auto* gpuBase = static_cast<uint8_t*>(mCombinedGpu->data());
        TLLM_CUDA_CHECK(cudaMemcpyAsync(gpuBase, pinnedBase, totalBytes, cudaMemcpyHostToDevice, mSubmitStream->get()));
        cubSrcPtrs = reinterpret_cast<void const**>(gpuBase);
        cubDstPtrs = reinterpret_cast<void**>(gpuBase + srcBytes);
        cubSizes = reinterpret_cast<size_t*>(gpuBase + srcBytes + dstBytes);
    }

    size_t actualTempBytes = mCubTempStorageSize;
    TLLM_CUDA_CHECK(cub::DeviceMemcpy::Batched(
        mCubTempStorage->data(), actualTempBytes, cubSrcPtrs, cubDstPtrs, cubSizes, numBuffers, mSubmitStream->get()));

    auto completionEvent = mEventPool->acquire();
    mSubmitStream->record(*completionEvent);
    return std::make_unique<P2pTransferStatus>(mSubmitStream, completionEvent);
}

std::unique_ptr<TransferStatus> P2pTransferContext::submitWithMemcpyBatch(
    std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes)
{
    size_t numOps = srcPtrs.size();
    if (numOps == 0)
    {
        auto event = mEventPool->acquire();
        mSubmitStream->record(*event);
        return std::make_unique<P2pTransferStatus>(mSubmitStream, event);
    }

    std::vector<void const*> constSrcPtrs(srcPtrs.begin(), srcPtrs.end());

    // Single-thread path: chosen when the worker-pool dispatch cost would outweigh
    // the CPU savings from parallel API calls. Thresholds:
    //   - mBatchCopyThreads <= 1     : multi-thread explicitly disabled
    //   - numOps < mMultiThreadMinOps: batch too small to be worth splitting (default 4096)
    //   - numOps < mBatchCopyThreads : can't evenly split to N workers
    // In the single-thread path the caller itself issues ONE cudaMemcpyBatchAsync on its
    // own mSubmitStream — no worker pool is constructed, no cross-caller contention.
    if (mBatchCopyThreads <= 1 || numOps < mMultiThreadMinOps || numOps < static_cast<size_t>(mBatchCopyThreads))
    {
        if (mCounters)
        {
            mCounters->memcpyBatchSingleThread.fetch_add(1, std::memory_order_relaxed);
        }
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        std::vector<size_t> idx(numOps, 0);

        TLLM_CUDA_CHECK(cudaMemcpyBatchAsync(
            dstPtrs.data(), constSrcPtrs.data(), sizes.data(), numOps, &attr, idx.data(), 1, mSubmitStream->get()));

        auto completionEvent = mEventPool->acquire();
        mSubmitStream->record(*completionEvent);
        return std::make_unique<P2pTransferStatus>(mSubmitStream, completionEvent);
    }

    // Multi-thread path — lazily construct this caller's own worker pool + streams.
    // Each caller thread thus drives N independent workers with no cross-caller queue contention.
    if (mCounters)
    {
        mCounters->memcpyBatchMultiThread.fetch_add(1, std::memory_order_relaxed);
    }
    ensureWorkerPoolAndStreams();

    int numWorkers = mBatchCopyThreads;
    size_t perWorker = numOps / numWorkers;
    auto batchPending = std::make_shared<std::atomic<int>>(numWorkers);

    std::vector<std::shared_ptr<runtime::CudaEvent>> workerEvents;
    workerEvents.reserve(numWorkers);

    for (int t = 0; t < numWorkers; ++t)
    {
        size_t off = t * perWorker;
        size_t cnt = (t == numWorkers - 1) ? (numOps - off) : perWorker;

        auto event = mEventPool->acquire();

        BatchCopyTask task;
        task.dst.assign(dstPtrs.begin() + off, dstPtrs.begin() + off + cnt);
        task.src.assign(constSrcPtrs.begin() + off, constSrcPtrs.begin() + off + cnt);
        task.sizes.assign(sizes.begin() + off, sizes.begin() + off + cnt);
        task.stream = mBatchCopyStreams[t]->get();
        task.completionEvent = event;
        task.batchPending = batchPending;

        mWorkerPool->submit(std::move(task));
        workerEvents.push_back(std::move(event));
    }

    return std::make_unique<P2pTransferStatus>(std::move(batchPending), std::move(workerEvents));
}

// ============================================================================
// P2pTransferContextPool
// ============================================================================

P2pTransferContextPool::P2pTransferContextPool(CUdevice localDevice, std::shared_ptr<CudaEventPool> eventPool,
    int batchCopyThreads, size_t multiThreadMinOps, bool cubZeroCopy, std::shared_ptr<P2pAgentCounters> counters)
    : mLocalDevice(localDevice)
    , mEventPool(std::move(eventPool))
    , mBatchCopyThreads(batchCopyThreads)
    , mMultiThreadMinOps(multiThreadMinOps)
    , mCubZeroCopy(cubZeroCopy)
    , mCounters(std::move(counters))
{
}

P2pTransferContext& P2pTransferContextPool::contextForCurrentThread()
{
    // Look up by thread id under the mutex. A thread_local pointer cache is unsafe here
    // because a previous pool at the same address (common for stack-allocated or reused
    // heap allocations) would leave a stale entry pointing at a freed Context. The mutex
    // cost is small compared to the CUDA work that follows.
    auto tid = std::this_thread::get_id();
    std::unique_lock lock(mMutex);
    auto it = mContexts.find(tid);
    if (it != mContexts.end())
    {
        return *it->second;
    }
    auto ctx = std::make_unique<P2pTransferContext>(
        mLocalDevice, mEventPool, mBatchCopyThreads, mMultiThreadMinOps, mCubZeroCopy, mCounters);
    P2pTransferContext* raw = ctx.get();
    mContexts[tid] = std::move(ctx);
    return *raw;
}

// ============================================================================
// P2pTransferAgent (facade)
// ============================================================================

namespace
{
CUdevice queryCurrentDevice()
{
    CUdevice device = 0;
    auto err = cuCtxGetDevice(&device);
    if (err != CUDA_SUCCESS)
    {
        TLLM_LOG_WARNING("P2pTransfer: failed to get current CUDA device error=%d", err);
    }
    return device;
}
} // anonymous namespace

P2pTransferAgent::P2pTransferAgent()
    : mLocalDevice(queryCurrentDevice())
    , mEventPool(std::make_shared<CudaEventPool>())
    , mBatchCopyThreads(std::max(1, common::getEnvKvTransferP2pBatchCopyThreads()))
    , mMultiThreadMinOps(common::getEnvKvTransferP2pBatchCopyMinOps())
    , mCubZeroCopy(common::getEnvKvTransferP2pCubZeroCopy())
    , mCounters(std::make_shared<P2pAgentCounters>())
    , mExporter(mLocalDevice)
    , mRegistry(mLocalDevice)
    , mContextPool(mLocalDevice, mEventPool, mBatchCopyThreads, mMultiThreadMinOps, mCubZeroCopy, mCounters)
{
    if (mBatchCopyThreads > 1)
    {
        TLLM_LOG_INFO(
            "P2pTransfer: multi-thread memcpyBatch enabled (threads=%d, minOps=%zu); worker pools "
            "are created per caller thread on first eligible submit",
            mBatchCopyThreads, mMultiThreadMinOps);
    }
}

} // namespace tensorrt_llm::executor::kv_cache
