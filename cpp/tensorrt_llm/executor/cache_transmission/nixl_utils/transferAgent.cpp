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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/ipUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cuda.h>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <limits>
#include <mutex>
#include <netdb.h>
#include <netinet/in.h>
#include <nixl_types.h>
#include <numeric>
#include <set>
#include <shared_mutex>
#include <sys/file.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef TLLM_BOUNCE_V2
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceConfig.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ExecPool.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"
#include <cuda_runtime_api.h>
#include <future>
#endif

namespace tensorrt_llm::executor::kv_cache
{

// ============================================================================
// Bounce v2 integration (opt-in via TRTLLM_NIXL_BOUNCE_ENABLE). When disabled, transfers remain on
// the standard NIXL path.
// ============================================================================
#ifdef TLLM_BOUNCE_V2
namespace bounce
{
struct NixlBounceState
{
    BounceConfig cfg;
    int deviceId{};
    NixlTransferAgent* owner{nullptr};       // the agent this state belongs to (owns this state)
    bool arenaRegistered{false};             // arena was NIXL-registered -> dtor must deregister it
    std::unique_ptr<ControlChannel> channel; // ZmqControlChannel
    std::unique_ptr<BounceArena> arena;      // ONE shared buffer: receiver targets + local gather staging
    std::unique_ptr<ExecPool> exec;          // gather/scatter exec contexts (streams/scratch)
    // Declared last -> destroyed first: BounceTransport::~ joins its threads (which use the
    // agent/channel/arena/exec) before those are torn down.
    std::unique_ptr<BounceTransport> transport;

    ~NixlBounceState()
    {
        transport.reset(); // join the IO/worker threads before anything they use goes away
        if (owner != nullptr && arenaRegistered && arena)
        {
            // Deregister the arena from NIXL while its memory is still alive (cudaFree follows).
            owner->deregisterRegionImpl(arena->base(), arena->bytes(), deviceId);
        }
    }
};
} // namespace bounce

namespace
{
/// TransferStatus over the bounce transport's completion future (the value Python's wait() reads).
class BounceTransferStatus final : public TransferStatus
{
public:
    explicit BounceTransferStatus(std::shared_future<bounce::BounceResult> fut)
        : mFut(std::move(fut))
    {
    }

    [[nodiscard]] bool isCompleted() const override
    {
        return mFut.valid() && mFut.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    [[nodiscard]] TransferState wait(int64_t timeoutMs) const override
    {
        if (!mFut.valid())
        {
            return TransferState::kFAILURE;
        }
        if (timeoutMs < 0)
        {
            return mFut.get().state;
        }
        if (mFut.wait_for(std::chrono::milliseconds(timeoutMs)) == std::future_status::ready)
        {
            return mFut.get().state;
        }
        return TransferState::kIN_PROGRESS;
    }

    /// Failure cause recorded by the bounce transport (empty while in flight / on success).
    [[nodiscard]] std::string getLastStatusStr() const override
    {
        if (!mFut.valid() || mFut.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        {
            return {};
        }
        auto const& result = mFut.get();
        return result.reason == bounce::BounceFailReason::kNone ? std::string{} : bounce::toString(result.reason);
    }

private:
    std::shared_future<bounce::BounceResult> mFut;
};
} // namespace

void NixlTransferAgent::maybeInitBounce(std::optional<bool> agentBufferEnable)
{
    auto cfg = bounce::BounceConfig::fromEnv();
    // An explicit agent_buffer_enable from CacheTransceiverConfig / BaseAgentConfig overrides the
    // TRTLLM_NIXL_BOUNCE_ENABLE environment variable; unset keeps the env-var behavior.
    if (agentBufferEnable.has_value())
    {
        cfg.enabled = agentBufferEnable.value();
    }
    if (!cfg.enabled)
    {
        return;
    }
    // A single chunk must fit a fresh arena; otherwise its request cannot make progress before
    // requestTimeoutMs. Clamp the per-chunk cap to the configured arena size first. BounceTransport
    // applies a second clamp to the buddy allocator's smaller usable capacity when necessary.
    if (cfg.maxChunkSizeBytes > cfg.arenaSizeBytes)
    {
        TLLM_LOG_WARNING("NixlTransferAgent(%s): maxChunkSizeBytes (%zu) > arenaSizeBytes (%zu) -> clamping to arena",
            mName.c_str(), cfg.maxChunkSizeBytes, cfg.arenaSizeBytes);
        cfg.maxChunkSizeBytes = cfg.arenaSizeBytes;
    }
    // A chunk's packed size travels in 32-bit wire fields, so it must fit in 32 bits even though the
    // arena (and thus region offsets) may exceed 4 GiB. Clamp rather than abort on a large config.
    if (cfg.maxChunkSizeBytes > std::numeric_limits<std::uint32_t>::max())
    {
        TLLM_LOG_WARNING(
            "NixlTransferAgent(%s): maxChunkSizeBytes (%zu) > 4 GiB -> clamping "
            "(chunk size is 32-bit)",
            mName.c_str(), cfg.maxChunkSizeBytes);
        cfg.maxChunkSizeBytes = std::numeric_limits<std::uint32_t>::max();
    }
    // Any setup failure (e.g. the arena/ExecPool cudaMalloc fails on a busy GPU, or fabric alloc
    // throws) must NOT take down agent construction — bounce is an opt-in fast path. Catch, warn,
    // and leave mBounce null so the agent runs the standard per-desc NIXL path unchanged.
    try
    {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess)
        {
            // Don't silently bind the arena/exec to device 0 on a multi-GPU host — warn and clear the
            // sticky error; the enclosing try still proceeds (best-effort) with dev=0.
            (void) cudaGetLastError();
            TLLM_LOG_WARNING("NixlTransferAgent(%s): cudaGetDevice failed; bounce assuming device 0", mName.c_str());
        }
        auto st = std::make_unique<bounce::NixlBounceState>();
        st->cfg = cfg;
        st->deviceId = dev;
        // Bind the control channel to a ROUTABLE interface (not loopback) so peers on OTHER nodes can
        // reach it — the bounce endpoint is advertised cross-node via AgentDesc and the receiver
        // self-bootstraps a DEALER to it from WANT. Pick the IP the same way the NIXL agent does
        // (TRTLLM_NIXL_INTERFACE NIC if set, else auto-detect via outbound route / hostname; the
        // shared common::getLocalIp util). IPv6 needs brackets in a zmq tcp endpoint.
        std::string const localIp = common::getLocalIp(common::getEnvNixlInterface(), mpi::MpiComm::world().getRank());
        std::string const bindAddr
            = (localIp.find(':') != std::string::npos) ? "tcp://[" + localIp + "]:*" : "tcp://" + localIp + ":*";
        st->channel = std::make_unique<bounce::ZmqControlChannel>(mName, bindAddr);
        std::string const controlDesc = st->channel->localEndpoint();
        st->owner = this;
        std::size_t const maxDescs = std::max<std::size_t>(1024ULL, cfg.maxChunkSizeBytes / 256ULL);
        // ONE shared arena for both roles (receiver RDMA-write targets + local gather staging),
        // carved into variable-size regions by the scheduler. Register it ONCE NOW (before any
        // metadata exchange) so peers' loaded MD includes it. Exec contexts (streams/scratch) are a
        // separate small pool borrowed per gather/scatter kernel.
        st->arena = std::make_unique<bounce::BounceArena>(cfg.arenaSizeBytes, dev, !cfg.disableFabricMemory);
        if (!registerRegionImpl(st->arena->base(), st->arena->bytes(), dev))
        {
            // Arena couldn't be NIXL-registered -> bounce can't move data. Leave mBounce null so the
            // agent falls back transparently to the standard per-desc NIXL path (no partial enable).
            TLLM_LOG_WARNING(
                "NixlTransferAgent(%s): bounce arena registerMem failed -> bounce disabled "
                "(NIXL fallback)",
                mName.c_str());
            return;
        }
        st->arenaRegistered = true;
        st->exec = std::make_unique<bounce::ExecPool>(cfg.copyStreamCount, maxDescs, dev, cfg.useZeroCopyArguments);
        st->transport = std::make_unique<bounce::BounceTransport>(
            mName, cfg, dev, st->channel.get(), *this, st->arena.get(), st->exec.get());
        mBounce = std::move(st);
        TLLM_LOG_INFO(
            "NixlTransferAgent(%s): bounce v2 enabled "
            "(arena=%zuB chunk<=%zuB maxInflight=%u copyStreams=%u control=%s)",
            mName.c_str(), cfg.arenaSizeBytes, cfg.maxChunkSizeBytes,
            static_cast<unsigned>(cfg.maxInflightChunksPerRequest), static_cast<unsigned>(cfg.copyStreamCount),
            controlDesc.c_str());
    }
    catch (std::exception const& e)
    {
        mBounce.reset();
        TLLM_LOG_WARNING("NixlTransferAgent(%s): bounce init failed (%s) -> bounce disabled (NIXL fallback)",
            mName.c_str(), e.what());
    }
}

bool NixlTransferAgent::shouldUseBounce(TransferRequest const& request) const
{
    if (!mBounce)
    {
        return false;
    }
    auto const& cfg = mBounce->cfg;
    if (request.getOp() != TransferOp::kWRITE)
    {
        return false;
    }
    if (request.getSrcDescs().getType() != MemoryType::kVRAM || request.getDstDescs().getType() != MemoryType::kVRAM)
    {
        return false;
    }
    if (request.getSyncMessage().has_value())
    {
        return false; // sync message rides the standard notif path
    }
    if (!mBounce->transport->hasPeerHandshake(request.getRemoteName()))
    {
        // The peer did not advertise a compatible bounce handshake (bounce disabled, missing or
        // malformed handshake, or version/control-kind/chunk-size mismatch). Without a compatible
        // receiver, WANT would remain unanswered until requestTimeoutMs, so use standard NIXL.
        return false;
    }
    auto const& srcs = request.getSrcDescs().getDescs();
    auto const& dsts = request.getDstDescs().getDescs();
    if (srcs.empty() || srcs.size() != dsts.size() || srcs.size() < cfg.minDescriptorCount)
    {
        return false;
    }
    auto const sourceDeviceId = srcs.front().getDeviceId();
    auto const destinationDeviceId = dsts.front().getDeviceId();
    if (sourceDeviceId != static_cast<std::uint32_t>(mBounce->deviceId))
    {
        return false;
    }
    // Screen every precondition BounceTransferPlan::build enforces with TLLM_CHECK: an admitted
    // request must never throw out of submitTransferRequests — it either runs on bounce or is
    // routed to the standard per-descriptor NIXL path.
    std::uint64_t totalBytes = 0;
    // BounceTransport may clamp the configured cap further to the buddy allocator's usable capacity.
    auto const maxChunkSizeBytes = mBounce->transport->maxChunkSizeBytes();
    for (std::size_t i = 0; i < srcs.size(); ++i)
    {
        auto const len = srcs[i].getLen();
        if (len != dsts[i].getLen() || len > maxChunkSizeBytes || srcs[i].getDeviceId() != sourceDeviceId
            || dsts[i].getDeviceId() != destinationDeviceId)
        {
            return false;
        }
        totalBytes += len;
    }
    std::uint64_t const avg = totalBytes / srcs.size();
    return avg <= cfg.maxAverageDescriptorSizeBytes;
}
#else  // !TLLM_BOUNCE_V2 — bounce not built; the member stays null and these are no-ops.
namespace bounce
{
struct NixlBounceState
{
};
} // namespace bounce

void NixlTransferAgent::maybeInitBounce(std::optional<bool> agentBufferEnable)
{
    if (agentBufferEnable.value_or(false))
    {
        TLLM_LOG_WARNING("agent_buffer_enable requested but bounce support is not built; ignoring");
    }
}

bool NixlTransferAgent::shouldUseBounce(TransferRequest const&) const
{
    return false;
}
#endif // TLLM_BOUNCE_V2

class FileLock
{
private:
    int fd_;
    std::string lockFile_;
    bool locked_;

public:
    explicit FileLock(std::string const& lockFile)
        : fd_(-1)
        , lockFile_(lockFile)
        , locked_(false)
    {
    }

    ~FileLock()
    {
        unlock();
    }

    bool lock()
    {
        if (locked_)
            return true;

        size_t pos = lockFile_.find_last_of('/');
        if (pos != std::string::npos)
        {
            std::string dir = lockFile_.substr(0, pos);
            mkdir(dir.c_str(), 0755);
        }

        fd_ = open(lockFile_.c_str(), O_CREAT | O_WRONLY, 0644);
        if (fd_ == -1)
        {
            TLLM_LOG_ERROR("Failed to open lock file: %s", lockFile_.c_str());
            return false;
        }

        if (flock(fd_, LOCK_EX) == -1)
        {
            TLLM_LOG_ERROR("Failed to acquire file lock: %s", lockFile_.c_str());
            close(fd_);
            fd_ = -1;
            return false;
        }

        locked_ = true;
        return true;
    }

    void unlock()
    {
        if (locked_ && fd_ != -1)
        {
            flock(fd_, LOCK_UN);
            close(fd_);
            fd_ = -1;
            locked_ = false;
        }
    }
};

uint16_t getAvailablePort(std::string const& ip = "0.0.0.0")
{
    struct addrinfo hints
    {
    };

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo* res;
    int ret = getaddrinfo(ip.c_str(), "0", &hints, &res);
    TLLM_CHECK(ret == 0);

    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    TLLM_CHECK(sockfd != -1);

    ret = bind(sockfd, res->ai_addr, res->ai_addrlen);
    TLLM_CHECK(ret == 0);

    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    ret = getsockname(sockfd, (struct sockaddr*) &addr, &addrlen);
    TLLM_CHECK(ret == 0);

    uint16_t port = ntohs(addr.sin_port);
    close(sockfd);
    freeaddrinfo(res);

    return port;
}

[[nodiscard]] nixl_mem_t NixlHelper::convert(MemoryType type)
{
    switch (type)
    {
    case MemoryType::kDRAM: return DRAM_SEG;
    case MemoryType::kVRAM: return VRAM_SEG;
    case MemoryType::kBLK: return BLK_SEG;
    case MemoryType::kOBJ: return OBJ_SEG;
    case MemoryType::kFILE: return FILE_SEG;
    default: TLLM_THROW("Unknown MemoryType value");
    }
}

[[nodiscard]] nixlBasicDesc NixlHelper::convert(MemoryDesc const& desc)
{
    return nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()};
}

[[nodiscard]] nixl_reg_dlist_t NixlHelper::convertRegDlist(RegisterDescs const& descs)
{
    nixl_reg_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBlobDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

[[nodiscard]] nixl_reg_dlist_t NixlHelper::convertRegDlist(FileDescs const& descs)
{
    nixl_reg_dlist_t list(FILE_SEG);
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBlobDesc{0, desc.getLen(), desc.getFd()});
    }
    return list;
}

[[nodiscard]] nixl_xfer_op_t NixlHelper::convert(TransferOp const& op)
{
    switch (op)
    {
    case TransferOp::kREAD: return NIXL_READ;
    case TransferOp::kWRITE: return NIXL_WRITE;
    default: TLLM_THROW("Unknown TransferOp value");
    }
}

[[nodiscard]] nixl_xfer_dlist_t NixlHelper::convertXferDist(TransferDescs const& descs)
{
    nixl_xfer_dlist_t list{convert(descs.getType())};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBasicDesc{desc.getAddr(), desc.getLen(), desc.getDeviceId()});
    }
    return list;
}

[[nodiscard]] nixl_xfer_dlist_t NixlHelper::convertXferDist(FileDescs const& descs)
{
    nixl_xfer_dlist_t list{FILE_SEG};
    for (auto const& desc : descs.getDescs())
    {
        list.addDesc(nixlBasicDesc{0, desc.getLen(), desc.getFd()});
    }
    return list;
}

void NixlHelper::posixGpuToFileFallback(MemoryDescs const& memoryDescs, FileDescs const& fileDescs)
{
    auto const& memVec = memoryDescs.getDescs();
    auto const& fileVec = fileDescs.getDescs();
    std::size_t i;

    for (i = 0; i < std::min(memVec.size(), fileVec.size()); i++)
    {
        auto& memDesc = memVec[i];
        auto& fileDesc = fileVec[i];

        ssize_t numBytes = static_cast<ssize_t>(memDesc.getLen());
        std::vector<uint8_t> hostBuffer(numBytes);

        cudaError_t cpyErr = cudaMemcpy(
            hostBuffer.data(), reinterpret_cast<void*>(memDesc.getAddr()), numBytes, cudaMemcpyDeviceToHost);
        TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to host failed, error=%d", cpyErr);

        ssize_t written = ::write(fileDesc.getFd(), hostBuffer.data(), numBytes);
        TLLM_CHECK_WITH_INFO(written >= 0, "POSIX write error=%zd", written);
    }
}

void NixlHelper::posixFileToGpuFallback(MemoryDescs const& memoryDescs, FileDescs const& fileDescs)
{
    auto const& memVec = memoryDescs.getDescs();
    auto const& fileVec = fileDescs.getDescs();
    std::size_t i;

    for (i = 0; i < std::min(memVec.size(), fileVec.size()); i++)
    {
        auto& memDesc = memVec[i];
        auto& fileDesc = fileVec[i];

        ssize_t numBytes = static_cast<ssize_t>(memDesc.getLen());
        std::vector<uint8_t> hostBuffer(numBytes);

        ssize_t bytesRead = ::read(fileDesc.getFd(), hostBuffer.data(), numBytes);
        TLLM_CHECK_WITH_INFO(bytesRead == numBytes, "POSIX read error=%zd", bytesRead);

        cudaError_t cpyErr = cudaMemcpy(
            reinterpret_cast<void*>(memDesc.getAddr()), hostBuffer.data(), numBytes, cudaMemcpyHostToDevice);
        TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to device failed, error=%d", cpyErr);
    }
}

NixlTransferStatus::NixlTransferStatus(std::weak_ptr<nixlAgent> agent, nixlXferReqH* handle)
    : mWeakAgent{std::move(agent)}
    , mHandle{handle}
    , mSynchronizeHandleAccess{common::getEnvDisaggEnableInflightCancel()}
{
    TLLM_CHECK(!mWeakAgent.expired());
    TLLM_CHECK(mHandle);
}

NixlTransferStatus::~NixlTransferStatus() noexcept
{
    try
    {
        if (!release())
        {
            TLLM_LOG_WARNING(
                "NIXL transfer handle release failed during destruction; backend handle may remain active");
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("~NixlTransferStatus: releaseXferReq threw: %s", e.what());
    }
    catch (...)
    {
        TLLM_LOG_WARNING("~NixlTransferStatus: releaseXferReq threw unknown exception");
    }
}

TransferState NixlTransferStatus::wait(int64_t timeout_ms) const
{
    auto startTime = std::chrono::steady_clock::now();

    while (true)
    {
        auto const status = queryStatus();
        if (status == NIXL_SUCCESS)
        {
            return TransferState::kSUCCESS;
        }
        else if (status != NIXL_IN_PROG)
        {
            return TransferState::kFAILURE;
        }

        // If timeout_ms < 0, wait indefinitely until status is not NIXL_IN_PROG
        if (timeout_ms < 0)
        {
            std::this_thread::yield();
            continue;
        }

        // Check if timeout has elapsed
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        if (elapsed >= timeout_ms)
        {
            return TransferState::kIN_PROGRESS;
        }

        std::this_thread::yield();
    }
}

int NixlTransferStatus::getLastStatus() const noexcept
{
    return mLastStatus.load(std::memory_order_relaxed);
}

std::string NixlTransferStatus::getLastStatusStr() const
{
    return nixlEnumStrings::statusStr(static_cast<nixl_status_t>(getLastStatus()));
}

[[nodiscard]] bool NixlTransferStatus::isCompleted() const
{
    return queryStatus() == NIXL_SUCCESS;
}

nixl_status_t NixlTransferStatus::queryStatus() const
{
    auto const query = [this]()
    {
        if (mHandle == nullptr)
        {
            mLastStatus.store(static_cast<int>(NIXL_ERR_INVALID_PARAM), std::memory_order_relaxed);
            return NIXL_ERR_INVALID_PARAM;
        }
        auto agent = mWeakAgent.lock();
        if (!agent)
        {
            // Owning agent was reset; report failure so callers don't deref a null status.
            mLastStatus.store(static_cast<int>(NIXL_ERR_INVALID_PARAM), std::memory_order_relaxed);
            return NIXL_ERR_INVALID_PARAM;
        }
        auto const status = agent->getXferStatus(mHandle);
        mLastStatus.store(static_cast<int>(status), std::memory_order_relaxed);
        return status;
    };

    if (mSynchronizeHandleAccess)
    {
        std::lock_guard<std::mutex> lock(mHandleMutex);
        return query();
    }
    return query();
}

[[nodiscard]] bool NixlTransferStatus::release()
{
    std::lock_guard<std::mutex> lock(mHandleMutex);
    if (mHandle == nullptr)
    {
        return true;
    }

    auto agent = mWeakAgent.lock();
    if (!agent)
    {
        mHandle = nullptr;
        mLastStatus.store(static_cast<int>(NIXL_ERR_INVALID_PARAM), std::memory_order_relaxed);
        return true;
    }

    auto status = agent->releaseXferReq(mHandle);
    mLastStatus.store(static_cast<int>(status), std::memory_order_relaxed);
    if (status == NIXL_SUCCESS)
    {
        mHandle = nullptr;
        return true;
    }

    TLLM_LOG_WARNING("NIXL releaseXferReq failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
    return false;
}

NixlTransferAgent::NixlTransferAgent(BaseAgentConfig const& config)
    : mName{config.mName}
{
    bool const mpiEnabled = !common::getBoolEnv("TLLM_DISABLE_MPI");
    TLLM_CHECK_WITH_INFO(config.rank.has_value() == config.worldSize.has_value(),
        "NIXL agent config fields 'rank' and 'worldSize' must be specified together");

    if (config.rank.has_value())
    {
        mRank = config.rank.value();
        mWorldSize = config.worldSize.value();
        TLLM_CHECK_WITH_INFO(mWorldSize > 0, "NIXL world size must be positive, got %d", mWorldSize);
        TLLM_CHECK_WITH_INFO(
            mRank >= 0 && mRank < mWorldSize, "NIXL rank must be in [0, %d), got %d", mWorldSize, mRank);
    }
    else if (config.useListenThread && mpiEnabled)
    {
        mRank = mpi::MpiComm::session().getRank();
        mWorldSize = mpi::MpiComm::session().getSize();
    }
    else if (config.useListenThread)
    {
        TLLM_LOG_WARNING("NIXL rank parameters are not configured; defaulting to one process");
    }

    nixl_status_t status;
    if (config.useListenThread)
    {
        FileLock lock("/tmp/trtllm_nixl_port.lock");
        if (!lock.lock())
        {
            TLLM_THROW("Failed to lock /tmp/trtllm_nixl_port.lock");
        }
        uint16_t port = getAvailablePort();
        uint32_t numWorker = config.backendParams.find("num_workers") != config.backendParams.end()
            ? std::stoi(config.backendParams.at("num_workers"))
            : 1;
        nixlAgentConfig nixlConfig{config.useProgThread, true, port, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, numWorker,
            0, 10000, config.enableTelemetry};
        std::string localIp = common::getLocalIp(common::getEnvNixlInterface(), mRank);
        // Bracket IPv6 literals (RFC 3986) so the last ':' always separates the port;
        // IPv4 keeps the legacy "ip:port" format for cross-version compatibility.
        if (localIp.find(':') != std::string::npos)
        {
            localIp = "[" + localIp + "]";
        }
        mAddress = localIp + ":" + std::to_string(port);
        mRawAgent = std::make_shared<nixlAgent>(config.mName, std::move(nixlConfig));
    }
    else
    {
        uint32_t numWorker = config.backendParams.find("num_workers") != config.backendParams.end()
            ? std::stoi(config.backendParams.at("num_workers"))
            : 1;
        mAddress.clear();
        nixlAgentConfig nixlConfig{config.useProgThread, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, numWorker,
            0, 10000, config.enableTelemetry};
        mRawAgent = std::make_shared<nixlAgent>(config.mName, std::move(nixlConfig));
    }

    std::string nixlBackend = common::getEnvNixlBackend();
    // List of supported backends - extend this list as new backends are added
    static std::set<std::string> const kSUPPORTED_BACKENDS = {"UCX", "LIBFABRIC"};

    if (kSUPPORTED_BACKENDS.find(nixlBackend) == kSUPPORTED_BACKENDS.end())
    {
        TLLM_LOG_WARNING("Unsupported NIXL backend: %s, fallback to UCX", nixlBackend.c_str());
        nixlBackend = "UCX";
    }

    TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent using NIXL backend: %s", nixlBackend.c_str());

    // Get default plugin params first, then override with user-provided backendParams
    // NOTE: getPluginParams overwrites init1, so we must call it BEFORE setting user params
    nixl_b_params_t init1;
    nixl_mem_list_t mems1;
    status = mRawAgent->getPluginParams(nixlBackend.c_str(), mems1, init1);
    TLLM_CHECK(status == NIXL_SUCCESS);

    // Override default params with user-provided backendParams
    for (auto const& [key, value] : config.backendParams)
    {
        init1[key] = value;
        TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent backendParams: %s: %s", key.c_str(), value.c_str());
    }

    status = mRawAgent->createBackend(nixlBackend.c_str(), init1, mRawBackend);
    if (status != NIXL_SUCCESS || !mRawBackend)
    {
        TLLM_THROW("Failed to create NIXL backend: %s", nixlBackend.c_str());
    }
    mExtraParams.backends.push_back(mRawBackend);
    TLLM_LOG_INFO("NixlTransferAgent::NixlTransferAgent mAddress: %s", mAddress.c_str());

    // Bring up bounce v2 now, if enabled, so its shared arena is registered before any peer fetches
    // our metadata. This is a no-op when bounce is disabled or not built.
    maybeInitBounce(config.agentBufferEnable);
}

void NixlTransferAgent::registerMemory(RegisterDescs const& descs)
{
    std::unique_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::registerMemory called after shutdown");
    // Split VRAM descriptors at VMM chunk boundaries so each sub-descriptor
    // falls within a single cuMemCreate allocation (required by gdr_copy / cuda_ipc).
    size_t detectedChunkSize = 0;
    auto splitDescs = VmmDescSplitter::splitVmmDescs(descs, detectedChunkSize);

    // Record per-desc VMM chunk info for use in deregisterMemory / submitTransferRequests
    auto detectedRegionMap = VmmDescSplitter::detectVramRegionMap(descs);
    mLocalVramRegionInfo.merge(detectedRegionMap);

    nixl_status_t status;
    status = mRawAgent->registerMem(NixlHelper::convertRegDlist(splitDescs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    std::string localMD;
    status = mRawAgent->getLocalMD(localMD);
    TLLM_CHECK(status == NIXL_SUCCESS);
}

void NixlTransferAgent::deregisterMemory(RegisterDescs const& descs)
{
    std::unique_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::deregisterMemory called after shutdown");
    // Split using per-region registry info to match what was registered
    auto splitDescs = VmmDescSplitter::splitDescsWithRegionMap(descs, mLocalVramRegionInfo);

    nixl_status_t status;
    status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(splitDescs), &mExtraParams);
    TLLM_CHECK(status == NIXL_SUCCESS);

    // Remove entries from registry
    if (descs.getType() == MemoryType::kVRAM)
    {
        for (auto const& desc : descs.getDescs())
        {
            mLocalVramRegionInfo.erase(desc.getAddr());
        }
    }
}

void NixlTransferAgent::loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc)
{
    std::unique_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::loadRemoteAgent called after shutdown");
    nixl_status_t status;
    std::string remoteName;
    status = mRawAgent->loadRemoteMD(agentDesc.getBackendAgentDesc(), remoteName);
    TLLM_CHECK(status == NIXL_SUCCESS);
    TLLM_CHECK_WITH_INFO(
        name == remoteName, "loadRemoteAgent gets error agent name: %s != %s", name.c_str(), remoteName.c_str());
#ifdef TLLM_BOUNCE_V2
    // Validate the peer's bounce capability handshake from AgentDesc and register its control
    // channel. Only peers with the same wire version, control kind and effective maxChunkSizeBytes
    // pass; missing or incompatible handshakes leave the peer on standard NIXL.
    if (mBounce)
    {
        mBounce->transport->registerPeerHandshake(name, agentDesc.getBounceHandshake());
    }
#endif

    // Store remote VMM region info for chunk boundary calculations in
    // VmmDescSplitter::splitAndCoalesceTransferDescs. Per-agent map because different remote agents may have
    // overlapping virtual addresses.
    auto const& regions = agentDesc.getVramRegions();
    if (!regions.empty())
    {
        auto& remoteMap = mRemoteVramRegionInfo[name];
        for (auto const& r : regions)
        {
            remoteMap[r.baseAddr] = {r.totalLen, r.chunkSize};
        }
    }
}

AgentDesc NixlTransferAgent::getLocalAgentDesc()
{
    std::shared_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::getLocalAgentDesc called after shutdown");
    nixl_blob_t nixlBlob;
    nixl_status_t status = mRawAgent->getLocalMD(nixlBlob);
    TLLM_CHECK(status == NIXL_SUCCESS);

    // Pack ALL local region info (VMM multi-chunk and single-allocation alike) so remote agents can
    // compute chunk boundaries and never coalesce transfer descs across separately registered regions.
    std::vector<VramRegionMeta> regions;
    regions.reserve(mLocalVramRegionInfo.size());
    for (auto const& [base, info] : mLocalVramRegionInfo)
    {
        regions.push_back({base, info.totalLen, info.chunkSize});
    }

    std::string bounceHandshake;
#ifdef TLLM_BOUNCE_V2
    // Include the capability handshake in the structured AgentDesc. Callers must exchange the full
    // AgentDesc serialization for the peer to receive and validate it.
    if (mBounce)
    {
        bounceHandshake = mBounce->transport->localHandshakeBlob();
    }
#endif
    return AgentDesc{nixlBlob, std::move(regions), std::move(bounceHandshake)};
}

void NixlTransferAgent::invalidateRemoteAgent(std::string const& name)
{
    std::unique_lock<std::shared_mutex> lock(mLock);
    if (mShutdown.load())
    {
        // shutdown() already cleaned everything; treat as no-op for late callers.
        return;
    }
    // Clean up remote VMM region info before invalidating the remote agent.
    mRemoteVramRegionInfo.erase(name);
#ifdef TLLM_BOUNCE_V2
    // Drop bounce credits and in-flight requests tied to this peer so it cannot retain receiver
    // allocations or leave a sender request pending. forgetPeer also removes the handshake record;
    // bounce re-engages only after a fresh loadRemoteAgent validates new metadata.
    if (mBounce && mBounce->transport)
    {
        mBounce->transport->forgetPeer(name);
    }
#endif
    mRawAgent->invalidateRemoteMD(name);
}

[[nodiscard]] std::unique_ptr<TransferStatus> NixlTransferAgent::submitTransferRequests(TransferRequest const& request)
{
    std::shared_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::submitTransferRequests called after shutdown");

#ifdef TLLM_BOUNCE_V2
    // Bounce fast path for many-small-desc VRAM writes (opt-in). Falls through to the standard
    // NIXL path for everything else. The returned future resolves once every chunk is
    // scattered+ACKed at the peer (or FAILURE) — never hangs. Admission is final: once a request
    // enters bounce, a failure (peer loss / requestTimeoutMs) fails the transfer with no automatic
    // fallback to the standard path — the dominant failure causes (dead peer, partition) would fail
    // there too, and silent fallback would mask bounce-layer faults. Fallbacks happen only at
    // admission time (this gate, unhandshaked peer, disabled/failed arena setup).
    if (shouldUseBounce(request))
    {
        mBounceSubmitCount.fetch_add(1, std::memory_order_relaxed);
        // Debug-level so it's silent by default but lets ops confirm the bounce fast path actually
        // engaged for a given write (enable via TLLM_LOG_LEVEL_BY_MODULE "debug:executor").
        // Programmatic check: getBounceSubmitCount() / bounce_submit_count on the Python binding.
        TLLM_LOG_DEBUG("NixlTransferAgent(%s): bounce path engaged for write to %s (%zu descs)", mName.c_str(),
            request.getRemoteName().c_str(), request.getSrcDescs().getDescs().size());
        auto fut = mBounce->transport->submit(request.getSrcDescs(), request.getDstDescs(), request.getRemoteName());
        return std::make_unique<BounceTransferStatus>(std::move(fut));
    }
#endif

    // Split transfer descriptors at VMM chunk boundaries to match registered memory, then coalesce
    // contiguous pieces. A coalesced descriptor never crosses a chunk boundary or a registered
    // region boundary on either side, so every descriptor still falls within a single registered
    // memory region on both local and remote sides. Set TRTLLM_NIXL_DISABLE_COALESCE=1 to fall back
    // to split-only descriptors. Find remote agent's region map (empty map if not found — e.g. the
    // peer's AgentDesc carried no region info; addresses missing from a map are never coalesced,
    // so an empty remote map degrades to split-only rather than risking merges across unknown
    // remote chunk/registration boundaries).
    static VramRegionMap const kEmptyMap;
    auto remoteIt = mRemoteVramRegionInfo.find(request.getRemoteName());
    auto const& remoteRegionMap = (remoteIt != mRemoteVramRegionInfo.end()) ? remoteIt->second : kEmptyMap;

    auto [xferSrc, xferDst] = VmmDescSplitter::splitAndCoalesceTransferDescs(request.getSrcDescs(),
        request.getDstDescs(), mLocalVramRegionInfo, remoteRegionMap, !common::getEnvNixlDisableCoalesce());

    auto status = postXferRequest(request.getOp(), xferSrc, xferDst, request.getRemoteName(), request.getSyncMessage());
    TLLM_CHECK_WITH_INFO(status != nullptr,
        " rank: %d createXferReq failed (see warning above) selfname: %s remoteAgent name: %s", mRank, mName.c_str(),
        request.getRemoteName().c_str());
    return status;
}

std::unique_ptr<TransferStatus> NixlTransferAgent::postXferRequest(TransferOp op, TransferDescs const& srcDescs,
    TransferDescs const& dstDescs, std::string const& remoteName, std::optional<SyncMessage> const& syncMessage)
{
    // No mLock and no shutdown gate here: the public path already holds the shared lock, and the
    // bounce IO thread (the other caller) is joined before agent teardown. mExtraParams is set once
    // at construction and only read afterwards — its hasNotif is never set, so the no-notif case
    // (every bounce chunk write, on the sub-ms pipeline) passes it directly without a copy. Only a
    // notif-carrying call takes a local copy (hasNotif / notifMsg vary per call; mutating the shared
    // mExtraParams would race between concurrent submits even under shared_lock).
    nixl_opt_args_t notifParams;
    nixl_opt_args_t const* reqParams = &mExtraParams;
    if (syncMessage.has_value())
    {
        notifParams = mExtraParams;
        notifParams.hasNotif = true;
        notifParams.notifMsg = syncMessage.value();
        reqParams = &notifParams;
    }

    nixl_status_t status;
    nixlXferReqH* handle = nullptr;
    {
        NVTX3_SCOPED_RANGE(createXferReq);
        status = mRawAgent->createXferReq(NixlHelper::convert(op), NixlHelper::convertXferDist(srcDescs),
            NixlHelper::convertXferDist(dstDescs), remoteName, handle, reqParams);
    }
    if (status != NIXL_SUCCESS)
    {
        TLLM_LOG_WARNING("NixlTransferAgent(%s): createXferReq to %s failed: %s", mName.c_str(), remoteName.c_str(),
            nixlEnumStrings::statusStr(status).c_str());
        if (handle != nullptr)
        {
            (void) mRawAgent->releaseXferReq(handle);
        }
        return nullptr;
    }
    {
        NVTX3_SCOPED_RANGE(postXferReq);
        status = mRawAgent->postXferReq(handle, reqParams);
    }
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG)
    {
        TLLM_LOG_WARNING("NixlTransferAgent(%s): postXferReq to %s failed: %s", mName.c_str(), remoteName.c_str(),
            nixlEnumStrings::statusStr(status).c_str());
        // The status object still owns the handle; its release()/dtor retries the backend release.
    }
    return std::make_unique<NixlTransferStatus>(std::weak_ptr<nixlAgent>(mRawAgent), handle);
}

bool NixlTransferAgent::registerRegionImpl(void* base, std::size_t bytes, int deviceId)
{
    nixl_reg_dlist_t list{VRAM_SEG};
    list.addDesc(nixlBlobDesc{reinterpret_cast<uintptr_t>(base), bytes, static_cast<uint64_t>(deviceId)});
    nixl_status_t const st = mRawAgent->registerMem(list);
    if (st != NIXL_SUCCESS)
    {
        TLLM_LOG_WARNING(
            "NixlTransferAgent(%s): registerMem failed: %s", mName.c_str(), nixlEnumStrings::statusStr(st).c_str());
        return false;
    }
    return true;
}

void NixlTransferAgent::deregisterRegionImpl(void* base, std::size_t bytes, int deviceId)
{
    try
    {
        nixl_reg_dlist_t list{VRAM_SEG};
        list.addDesc(nixlBlobDesc{reinterpret_cast<uintptr_t>(base), bytes, static_cast<uint64_t>(deviceId)});
        nixl_status_t const st = mRawAgent->deregisterMem(list);
        if (st != NIXL_SUCCESS)
        {
            TLLM_LOG_WARNING("NixlTransferAgent(%s): deregisterMem failed: %s", mName.c_str(),
                nixlEnumStrings::statusStr(st).c_str());
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("NixlTransferAgent(%s): deregisterMem threw: %s", mName.c_str(), e.what());
    }
}

void NixlTransferAgent::notifySyncMessage(std::string const& name, SyncMessage const& syncMessage)
{
    std::shared_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::notifySyncMessage called after shutdown");
    auto status = mRawAgent->genNotif(name, syncMessage);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "genNotif failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
}

[[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> NixlTransferAgent::getNotifiedSyncMessages()
{
    std::shared_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::getNotifiedSyncMessages called after shutdown");
    nixl_notifs_t notifs;
    auto status = mRawAgent->getNotifs(notifs);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "getNotifs failed with status: %s", nixlEnumStrings::statusStr(status).c_str());

    return notifs;
}

ConnectionInfoType NixlTransferAgent::getLocalConnectionInfo()
{
    // mAddress is set in ctor and never mutated; no lock needed. This connection-info string does
    // not contain the bounce handshake; callers that need bounce must also exchange AgentDesc.
    return mAddress;
}

void NixlTransferAgent::loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo)
{
    std::unique_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::loadRemoteAgent called after shutdown");
    // Bounce bootstrap is handled on the AgentDesc overload, which is the production path.
    auto const separator = connectionInfo.rfind(':');
    TLLM_CHECK_WITH_INFO(separator != std::string::npos,
        "Invalid NIXL connection info, expected 'ip:port' or '[ipv6]:port': %s", connectionInfo.c_str());
    std::string ip = connectionInfo.substr(0, separator);
    std::string port = connectionInfo.substr(separator + 1);
    if (ip.size() >= 2 && ip.front() == '[' && ip.back() == ']')
    {
        ip = ip.substr(1, ip.size() - 2);
    }
    TLLM_LOG_DEBUG(mRank, "NixlTransferAgent::loadRemoteAgent loadRemoteAgent to %s remoteagent name: %s",
        connectionInfo.c_str(), name.c_str());
    TLLM_CHECK_WITH_INFO(!ip.empty() && !port.empty(), "loadRemoteAgent get empty ip or port, connectionInfo: %s",
        connectionInfo.c_str());
    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = ip;
    md_extra_params.port = std::stoi(port);
    auto status = mRawAgent->fetchRemoteMD(name, &md_extra_params);
    TLLM_CHECK_WITH_INFO(
        status == NIXL_SUCCESS, "fetchRemoteMD failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
    // status = mRawAgent->sendLocalMD(&md_extra_params);
    // TLLM_CHECK_WITH_INFO(
    //     status == NIXL_SUCCESS, "sendLocalMD failed with status: %s", nixlEnumStrings::statusStr(status).c_str());

    status = NIXL_ERR_NOT_FOUND;
    nixl_xfer_dlist_t descs{DRAM_SEG};
    while (status == NIXL_ERR_NOT_FOUND)
    {
        status = mRawAgent->checkRemoteMD(name, descs);
        TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS || status == NIXL_ERR_NOT_FOUND,
            "checkRemoteMD failed with status: %s", nixlEnumStrings::statusStr(status).c_str());
        if (status == NIXL_ERR_NOT_FOUND)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    TLLM_LOG_DEBUG(mRank,
        "NixlTransferAgent::loadRemoteAgent loadRemoteAgent to %s remoteagent name: %s success status: %s",
        connectionInfo.c_str(), name.c_str(), nixlEnumStrings::statusStr(status).c_str());
}

bool NixlTransferAgent::checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs)
{
    std::shared_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlTransferAgent::checkRemoteDescs called after shutdown");
    auto status = mRawAgent->checkRemoteMD(name, NixlHelper::convertXferDist(memoryDescs));
    TLLM_CHECK_WITH_INFO(status == NIXL_SUCCESS || status == NIXL_ERR_NOT_FOUND, "checkRemoteMD failed with status: %s",
        nixlEnumStrings::statusStr(status).c_str());
    return status == NIXL_SUCCESS;
}

void NixlTransferAgent::shutdown() noexcept
{
    // unique_lock drains all in-flight shared_lock holders (submit / getDesc / etc.).
    // A concurrent second shutdown() blocks here, then sees mShutdown=true and returns.
    std::unique_lock<std::shared_mutex> lock(mLock);
    if (mShutdown.exchange(true))
    {
        return;
    }
    TLLM_LOG_DEBUG("NixlTransferAgent::shutdown");

#ifdef TLLM_BOUNCE_V2
    // Stop the bounce transport first: its IO/scatter threads poll mRawAgent (getXferStatus),
    // so they must be joined before the agent is torn down. Failing pending futures here means
    // no submit() waiter hangs across shutdown.
    if (mBounce && mBounce->transport)
    {
        mBounce->transport->shutdown();
    }
    mBounce.reset();
#endif

    if (mRawAgent)
    {
        // Inline invalidate: invalidateRemoteAgent() would re-enter the non-recursive lock.
        for (auto const& [name, _] : mRemoteVramRegionInfo)
        {
            try
            {
                mRawAgent->invalidateRemoteMD(name);
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_WARNING(
                    "NixlTransferAgent::shutdown: invalidateRemoteMD(%s) threw: %s", name.c_str(), e.what());
            }
            catch (...)
            {
            }
        }
    }

    mExtraParams.backends.clear();
    mRawBackend = nullptr;
    try
    {
        mRawAgent.reset();
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("NixlTransferAgent::shutdown: ~nixlAgent threw: %s", e.what());
    }
    catch (...)
    {
        TLLM_LOG_WARNING("NixlTransferAgent::shutdown: ~nixlAgent threw unknown exception");
    }
    mLocalVramRegionInfo.clear();
    mRemoteVramRegionInfo.clear();
}

NixlTransferAgent::~NixlTransferAgent()
{
    TLLM_LOG_DEBUG("NixlTransferAgent::~NixlTransferAgent");
    shutdown();
}

NixlLoopbackAgent::NixlLoopbackAgent(BaseAgentConfig const& config)
    : mName{config.mName}
{
    nixlAgentConfig nixlConfig{config.useProgThread};
    nixlBackendH* backend;
    nixl_status_t status;
    nixl_b_params_t init;

    mRawAgent = std::make_shared<nixlAgent>(config.mName, std::move(nixlConfig));
    init["batch_pool_size"] = std::to_string(8);
    init["batch_limit"] = std::to_string(128);
    init["max_request_size"] = std::to_string(16 * 1024 * 1024);

    if (config.multiThread)
    {
        status = mRawAgent->createBackend("GDS_MT", init, backend);
        if (status != NIXL_SUCCESS || !backend)
            TLLM_THROW("Failed to create NIXL GDS_MT backend, status = %d", status);
    }
    else
    {
        status = mRawAgent->createBackend("GDS", init, backend);
        if (status != NIXL_SUCCESS || !backend)
            TLLM_THROW("Failed to create NIXL GDS backend, status = %d", status);
    }
}

void NixlLoopbackAgent::shutdown() noexcept
{
    // unique_lock drains all in-flight shared_lock holders before destroying the agent.
    std::unique_lock<std::shared_mutex> lock(mLock);
    if (mShutdown.exchange(true))
    {
        return;
    }
    try
    {
        mRawAgent.reset();
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("NixlLoopbackAgent::shutdown: ~nixlAgent threw: %s", e.what());
    }
    catch (...)
    {
        TLLM_LOG_WARNING("NixlLoopbackAgent::shutdown: ~nixlAgent threw unknown exception");
    }
}

NixlLoopbackAgent::~NixlLoopbackAgent()
{
    shutdown();
}

int NixlLoopbackAgent::registerMemory(MemoryDescs const& descs)
{
    nixl_status_t status = mRawAgent->registerMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

int NixlLoopbackAgent::deregisterMemory(MemoryDescs const& descs)
{
    nixl_status_t status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

int NixlLoopbackAgent::registerFiles(FileDescs const& descs)
{
    nixl_status_t status = mRawAgent->registerMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

int NixlLoopbackAgent::deregisterFiles(FileDescs const& descs)
{
    nixl_status_t status = mRawAgent->deregisterMem(NixlHelper::convertRegDlist(descs));
    if (status != NIXL_SUCCESS)
        return -1;

    return 0;
}

std::unique_ptr<TransferStatus> NixlLoopbackAgent::submitLoopbackRequests(
    MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload)
{
    nixl_xfer_dlist_t vram_seg = NixlHelper::convertXferDist(memoryDescs);
    nixl_xfer_dlist_t file_seg = NixlHelper::convertXferDist(fileDescs);
    nixl_xfer_dlist_t& src = isOffload ? vram_seg : file_seg;
    nixl_xfer_dlist_t& dst = isOffload ? file_seg : vram_seg;
    nixl_xfer_op_t op = isOffload ? NIXL_WRITE : NIXL_READ;
    nixlXferReqH* handle = nullptr;

    nixl_status_t status = mRawAgent->createXferReq(op, src, dst, mName, handle);
    TLLM_CHECK(status == NIXL_SUCCESS && handle);
    status = mRawAgent->postXferReq(handle);
    TLLM_CHECK(status == NIXL_IN_PROG);

    return std::make_unique<NixlTransferStatus>(std::weak_ptr<nixlAgent>(mRawAgent), handle);
}

void NixlLoopbackAgent::executeLoopbackRequest(
    MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload)
{
    std::shared_lock<std::shared_mutex> lock(mLock);
    TLLM_CHECK_WITH_INFO(!mShutdown.load(), "NixlLoopbackAgent::executeLoopbackRequest called after shutdown");
    bool fallback = false;
    int ret;

    ret = this->registerFiles(fileDescs);
    if (ret < 0)
    { // register can fail if no GDS support
        TLLM_LOG_DEBUG("NIXL GDS register files failed, using POSIX fallback");
        fallback = true;
    }
    else
    {
        ret = this->registerMemory(memoryDescs);
        if (ret < 0)
        { // register can fail if no GDS support
            TLLM_LOG_DEBUG("NIXL GDS register memory failed, using POSIX fallback");
            this->deregisterFiles(fileDescs);
            fallback = true;
        }
    }

    if (fallback)
    {
        if (isOffload)
        {
            NixlHelper::posixGpuToFileFallback(memoryDescs, fileDescs);
        }
        else
        {
            NixlHelper::posixFileToGpuFallback(memoryDescs, fileDescs);
        }

        return;
    }

    std::unique_ptr<TransferStatus> status = this->submitLoopbackRequests(memoryDescs, fileDescs, isOffload);
    TLLM_CHECK_WITH_INFO(status != nullptr, "submitLoopbackRequests failed");
    TransferState transferState = status->wait();
    TLLM_CHECK_WITH_INFO(transferState == TransferState::kSUCCESS, "submitLoopbackRequests failed");

    this->deregisterMemory(memoryDescs);
    this->deregisterFiles(fileDescs);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    std::unique_ptr<BaseTransferAgent> createNixlTransferAgent(BaseAgentConfig const* config)
    {
        TLLM_CHECK(config);
        return std::make_unique<NixlTransferAgent>(*config);
    }
}

extern "C"
{
    std::shared_ptr<BaseLoopbackAgent> createNixlLoopbackAgent(BaseAgentConfig const* config)
    {
        TLLM_CHECK(config);
        return std::make_shared<NixlLoopbackAgent>(*config);
    }
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
