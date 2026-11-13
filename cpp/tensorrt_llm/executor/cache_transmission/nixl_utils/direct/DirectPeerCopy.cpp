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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/direct/DirectPeerCopy.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache::direct
{

namespace
{

constexpr std::uint32_t kHandshakeVersion = 1;
constexpr char kHandshakeMagic[4] = {'T', 'L', 'D', 'X'};
constexpr char kExtensionMagic[8] = {'T', 'L', 'L', 'M', 'A', 'X', 'T', '1'};
constexpr std::size_t kInitialPlanEntries = 64 * 1024;
constexpr std::size_t kPlanEntryBytes = 2 * sizeof(std::uint64_t) + sizeof(std::uint32_t);
//! Copy kernel grid size cap and block size.
constexpr std::uint32_t kMaxCtas = 32;
constexpr std::uint32_t kThreads = 512;
//! Largest kernel work unit; a multiple of 16 so every unit of an aligned descriptor stays 16-byte aligned.
constexpr std::uint64_t kUnitBytes = 256ULL << 10;
//! Plans whose work units average at most this many bytes (per-layer head slices of KV blocks) use one warp per unit;
//! larger units use one CTA per unit.
constexpr std::uint64_t kWarpUnitMeanBytes = 32ULL << 10;
//! Background import threads (peers are imported in parallel; each import maps one peer's chunks in order).
constexpr std::size_t kImportThreads = 4;
//! INFO counters every this many direct writes.
constexpr std::uint64_t kLogEvery = 2000;

std::uint64_t envU64(char const* name, std::uint64_t dflt)
{
    char const* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
    {
        return dflt;
    }
    char* end = nullptr;
    unsigned long long const x = std::strtoull(v, &end, 10);
    if (end == v)
    {
        TLLM_LOG_WARNING("Ignoring invalid %s=%s", name, v);
        return dflt;
    }
    return static_cast<std::uint64_t>(x);
}

std::string readBootId()
{
    std::ifstream f("/proc/sys/kernel/random/boot_id");
    std::string s;
    std::getline(f, s);
    return s;
}

std::string hostName()
{
    char buf[256] = {};
    if (gethostname(buf, sizeof(buf) - 1) != 0)
    {
        return {};
    }
    return std::string(buf);
}

char const* cuErrStr(CUresult r)
{
    char const* s = nullptr;
    if (cuGetErrorName(r, &s) != CUDA_SUCCESS || s == nullptr)
    {
        return "CUDA_ERROR_?";
    }
    return s;
}

class Writer
{
public:
    template <typename T>
    void pod(T const& v)
    {
        auto const* p = reinterpret_cast<char const*>(&v);
        mBuf.append(p, sizeof(T));
    }

    void str(std::string const& s)
    {
        pod(static_cast<std::uint32_t>(s.size()));
        mBuf.append(s);
    }

    void raw(void const* p, std::size_t n)
    {
        mBuf.append(static_cast<char const*>(p), n);
    }

    std::string take()
    {
        return std::move(mBuf);
    }

private:
    std::string mBuf;
};

class Reader
{
public:
    explicit Reader(std::string const& s)
        : mData(s.data())
        , mLen(s.size())
    {
    }

    template <typename T>
    bool pod(T& v)
    {
        if (mPos + sizeof(T) > mLen)
        {
            return false;
        }
        std::memcpy(&v, mData + mPos, sizeof(T));
        mPos += sizeof(T);
        return true;
    }

    bool str(std::string& s, std::size_t maxLen = 1U << 20)
    {
        std::uint32_t n = 0;
        if (!pod(n) || n > maxLen || mPos + n > mLen)
        {
            return false;
        }
        s.assign(mData + mPos, n);
        mPos += n;
        return true;
    }

    bool raw(void* p, std::size_t n)
    {
        if (mPos + n > mLen)
        {
            return false;
        }
        std::memcpy(p, mData + mPos, n);
        mPos += n;
        return true;
    }

    [[nodiscard]] bool done() const noexcept
    {
        return mPos == mLen;
    }

private:
    char const* mData;
    std::size_t mLen;
    std::size_t mPos{0};
};

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Config / wire formats
// ---------------------------------------------------------------------------------------------------------------

DirectConfig DirectConfig::fromEnv()
{
    DirectConfig c;
    c.enabled = envU64("TRTLLM_NIXL_DIRECT_COPY", 0) != 0;
    return c;
}

std::string encodeHandshake(DirectHandshake const& hs)
{
    Writer w;
    w.raw(kHandshakeMagic, sizeof(kHandshakeMagic));
    w.pod(hs.version);
    w.str(hs.host);
    w.str(hs.bootId);
    w.pod(static_cast<std::uint32_t>(hs.regions.size()));
    for (auto const& r : hs.regions)
    {
        w.pod(r.base);
        w.pod(r.len);
        w.pod(static_cast<std::uint32_t>(r.chunks.size()));
        for (auto const& c : r.chunks)
        {
            w.pod(c.va);
            w.pod(c.size);
            w.raw(&c.handle, sizeof(c.handle));
        }
    }
    return w.take();
}

bool decodeHandshake(std::string const& blob, DirectHandshake& out)
{
    Reader r(blob);
    char magic[sizeof(kHandshakeMagic)];
    if (!r.raw(magic, sizeof(magic)) || std::memcmp(magic, kHandshakeMagic, sizeof(magic)) != 0)
    {
        return false;
    }
    DirectHandshake hs;
    std::uint32_t nRegions = 0;
    if (!r.pod(hs.version) || hs.version != kHandshakeVersion || !r.str(hs.host) || !r.str(hs.bootId)
        || !r.pod(nRegions) || nRegions > (1U << 16))
    {
        return false;
    }
    hs.regions.resize(nRegions);
    for (auto& reg : hs.regions)
    {
        std::uint32_t nChunks = 0;
        if (!r.pod(reg.base) || !r.pod(reg.len) || !r.pod(nChunks) || nChunks > (1U << 22))
        {
            return false;
        }
        reg.chunks.resize(nChunks);
        for (auto& c : reg.chunks)
        {
            if (!r.pod(c.va) || !r.pod(c.size) || !r.raw(&c.handle, sizeof(c.handle)))
            {
                return false;
            }
        }
    }
    if (!r.done())
    {
        return false;
    }
    out = std::move(hs);
    return true;
}

std::string packExtensions(std::vector<std::pair<ExtensionTag, std::string>> const& sections)
{
    Writer w;
    w.raw(kExtensionMagic, sizeof(kExtensionMagic));
    w.pod(static_cast<std::uint32_t>(sections.size()));
    for (auto const& [tag, data] : sections)
    {
        w.pod(static_cast<std::uint32_t>(tag));
        w.pod(static_cast<std::uint64_t>(data.size()));
        w.raw(data.data(), data.size());
    }
    return w.take();
}

std::unordered_map<std::uint32_t, std::string> unpackExtensions(std::string const& blob)
{
    std::unordered_map<std::uint32_t, std::string> out;
    if (blob.size() < sizeof(kExtensionMagic)
        || std::memcmp(blob.data(), kExtensionMagic, sizeof(kExtensionMagic)) != 0)
    {
        if (!blob.empty())
        {
            out[static_cast<std::uint32_t>(ExtensionTag::kBounce)] = blob;
        }
        return out;
    }
    Reader r(blob);
    char magic[sizeof(kExtensionMagic)];
    std::uint32_t n = 0;
    if (!r.raw(magic, sizeof(magic)) || !r.pod(n))
    {
        return {};
    }
    for (std::uint32_t i = 0; i < n; ++i)
    {
        std::uint32_t tag = 0;
        std::uint64_t len = 0;
        if (!r.pod(tag) || !r.pod(len) || len > blob.size())
        {
            return {};
        }
        std::string data(static_cast<std::size_t>(len), '\0');
        if (!r.raw(data.data(), data.size()))
        {
            return {};
        }
        out[tag] = std::move(data);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Copy contexts: a stream, a blocking-sync completion event and plan buffers per concurrent transfer
// ---------------------------------------------------------------------------------------------------------------

struct CopyCtx
{
    cudaStream_t stream{nullptr};
    cudaEvent_t event{nullptr};
    char* host{nullptr};        ///< mapped pinned plan buffer
    char* dev{nullptr};         ///< the device alias of `host` the kernel reads the plan through
    std::size_t capEntries{0};
    std::vector<char*> retired; ///< outgrown buffers, freed at teardown (no runtime cudaFreeHost)
};

class CopyCtxPool
{
public:
    explicit CopyCtxPool(int deviceId)
        : mDeviceId(deviceId)
    {
    }

    ~CopyCtxPool()
    {
        (void) cudaSetDevice(mDeviceId);
        for (auto& c : mAll)
        {
            if (c->stream != nullptr)
            {
                (void) cudaStreamSynchronize(c->stream);
            }
            freeBuffer(c->host);
            for (auto* h : c->retired)
            {
                freeBuffer(h);
            }
            if (c->event != nullptr)
            {
                (void) cudaEventDestroy(c->event);
            }
            if (c->stream != nullptr)
            {
                (void) cudaStreamDestroy(c->stream);
            }
        }
    }

    /// A free context with room for `entries` plan entries, or nullptr (CUDA allocation failure).
    CopyCtx* acquire(std::size_t entries)
    {
        CopyCtx* ctx = nullptr;
        {
            std::lock_guard<std::mutex> lk(mMu);
            if (!mFree.empty())
            {
                ctx = mFree.back();
                mFree.pop_back();
            }
        }
        if (ctx == nullptr)
        {
            auto fresh = std::make_unique<CopyCtx>();
            if (cudaStreamCreateWithFlags(&fresh->stream, cudaStreamNonBlocking) != cudaSuccess
                || cudaEventCreateWithFlags(&fresh->event, cudaEventDisableTiming | cudaEventBlockingSync)
                    != cudaSuccess)
            {
                TLLM_LOG_WARNING("NixlDirectCopy: stream/event creation failed");
                if (fresh->stream != nullptr)
                {
                    (void) cudaStreamDestroy(fresh->stream);
                }
                return nullptr;
            }
            ctx = fresh.get();
            std::lock_guard<std::mutex> lk(mMu);
            mAll.push_back(std::move(fresh));
        }
        if (ctx->capEntries < entries && !grow(*ctx, entries))
        {
            release(ctx);
            return nullptr;
        }
        return ctx;
    }

    void release(CopyCtx* ctx)
    {
        if (ctx == nullptr)
        {
            return;
        }
        std::lock_guard<std::mutex> lk(mMu);
        mFree.push_back(ctx);
    }

    void synchronizeAll() noexcept
    {
        std::vector<cudaStream_t> streams;
        {
            std::lock_guard<std::mutex> lk(mMu);
            for (auto const& c : mAll)
            {
                streams.push_back(c->stream);
            }
        }
        for (auto s : streams)
        {
            (void) cudaStreamSynchronize(s);
        }
    }

private:
    static void freeBuffer(char* host)
    {
        if (host != nullptr)
        {
            (void) cudaFreeHost(host); // also releases its mapped alias
        }
    }

    bool grow(CopyCtx& ctx, std::size_t entries)
    {
        std::size_t cap = std::max(ctx.capEntries * 2, kInitialPlanEntries);
        while (cap < entries)
        {
            cap *= 2;
        }
        // The kernel reads the plan from mapped pinned host memory: no H2D plan copy queues on the copy engines
        // (behind the KV cache host-tier copies).
        char* host = nullptr;
        char* dev = nullptr;
        bool const ok
            = cudaHostAlloc(reinterpret_cast<void**>(&host), cap * kPlanEntryBytes, cudaHostAllocMapped) == cudaSuccess
            && cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev), host, 0) == cudaSuccess;
        if (!ok)
        {
            TLLM_LOG_WARNING("NixlDirectCopy: plan buffer allocation (%zu entries) failed", cap);
            freeBuffer(host);
            return false;
        }
        if (ctx.host != nullptr)
        {
            ctx.retired.push_back(ctx.host); // the context is idle, but freeing pinned memory would sync the device
        }
        ctx.host = host;
        ctx.dev = dev;
        ctx.capEntries = cap;
        return true;
    }

    int mDeviceId{0};
    std::mutex mMu;
    std::vector<std::unique_ptr<CopyCtx>> mAll;
    std::vector<CopyCtx*> mFree;
};

namespace
{

/// Completion = the context's event. The context goes back to the pool once the event has completed.
class DirectTransferStatus final : public TransferStatus
{
public:
    DirectTransferStatus(std::shared_ptr<CopyCtxPool> pool, CopyCtx* ctx)
        : mPool(std::move(pool))
        , mCtx(ctx)
    {
    }

    /// An already-terminal failure (the context was synchronized and returned by the caller).
    explicit DirectTransferStatus(std::string error)
        : mDone(true)
        , mOk(false)
        , mError(std::move(error))
    {
    }

    ~DirectTransferStatus() override
    {
        try
        {
            (void) wait(-1);
        }
        catch (...)
        {
        }
    }

    [[nodiscard]] bool isCompleted() const override
    {
        std::lock_guard<std::mutex> lk(mMu);
        if (mDone)
        {
            return mOk;
        }
        cudaError_t const e = cudaEventQuery(mCtx->event);
        if (e == cudaErrorNotReady)
        {
            return false;
        }
        finishLocked(e);
        return mOk;
    }

    TransferState wait(int64_t timeoutMs) const override
    {
        {
            std::lock_guard<std::mutex> lk(mMu);
            if (mDone)
            {
                return mOk ? TransferState::kSUCCESS : TransferState::kFAILURE;
            }
        }
        if (timeoutMs < 0)
        {
            // Blocking-sync event: the thread sleeps in the driver instead of polling it.
            cudaError_t const e = cudaEventSynchronize(mCtx->event);
            std::lock_guard<std::mutex> lk(mMu);
            if (!mDone)
            {
                finishLocked(e);
            }
            return mOk ? TransferState::kSUCCESS : TransferState::kFAILURE;
        }
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true)
        {
            {
                std::lock_guard<std::mutex> lk(mMu);
                if (mDone)
                {
                    return mOk ? TransferState::kSUCCESS : TransferState::kFAILURE;
                }
                cudaError_t const e = cudaEventQuery(mCtx->event);
                if (e != cudaErrorNotReady)
                {
                    finishLocked(e);
                    return mOk ? TransferState::kSUCCESS : TransferState::kFAILURE;
                }
            }
            if (std::chrono::steady_clock::now() >= deadline)
            {
                return TransferState::kIN_PROGRESS;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    /// The copy kernel always terminates: waiting for it proves the destination is no longer written.
    [[nodiscard]] bool release() override
    {
        (void) wait(-1);
        return true;
    }

    [[nodiscard]] std::string getLastStatusStr() const override
    {
        std::lock_guard<std::mutex> lk(mMu);
        return mError;
    }

private:
    void finishLocked(cudaError_t e) const
    {
        mDone = true;
        mOk = e == cudaSuccess;
        if (!mOk)
        {
            mError = std::string("direct peer copy failed: ") + cudaGetErrorName(e);
        }
        mPool->release(mCtx);
        mCtx = nullptr;
    }

    std::shared_ptr<CopyCtxPool> mPool;
    mutable std::mutex mMu;
    mutable CopyCtx* mCtx{nullptr};
    mutable bool mDone{false};
    mutable bool mOk{false};
    mutable std::string mError;
};

bool lookupRegion(VramRegionMap const& map, std::uint64_t addr, std::uint64_t len)
{
    auto it = map.upper_bound(static_cast<uintptr_t>(addr));
    if (it == map.begin())
    {
        return false;
    }
    --it;
    return addr >= it->first && addr + len <= it->first + it->second.totalLen && addr + len >= addr;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// DirectPeerCopy
// ---------------------------------------------------------------------------------------------------------------

DirectPeerCopy::DirectPeerCopy(int deviceId)
    : mDeviceId(deviceId)
    , mHost(hostName())
    , mBootId(readBootId())
    , mPool(std::make_shared<CopyCtxPool>(deviceId))
{
    TLLM_LOG_INFO("NixlDirectCopy: enabled on device %d (host %s, ctas %u, threads %u, unit %llu B)", mDeviceId,
        mHost.c_str(), kMaxCtas, kThreads, static_cast<unsigned long long>(kUnitBytes));
}

DirectPeerCopy::~DirectPeerCopy()
{
    {
        std::lock_guard<std::mutex> lk(mImportMu);
        mStopping.store(true);
        mImportQueue.clear();
        mGenerations.clear();
    }
    mImportCv.notify_all();
    for (auto& t : mImportThreads)
    {
        t.join();
    }
    waitAllInflight();
    std::unique_lock<std::shared_mutex> lk(mPeersMu);
    for (auto& [name, peer] : mPeers)
    {
        unmapPeer(*peer);
    }
    mPeers.clear();
}

std::size_t DirectPeerCopy::mappedPeerCount() const
{
    std::shared_lock<std::shared_mutex> lk(mPeersMu);
    return mPeers.size();
}

std::string DirectPeerCopy::localHandshake(VramRegionMap const& regions)
{
    std::lock_guard<std::mutex> lk(mExportMu);
    auto const sameMap = [&]()
    {
        if (mExportedFrom.size() != regions.size())
        {
            return false;
        }
        for (auto const& [base, info] : regions)
        {
            auto it = mExportedFrom.find(base);
            if (it == mExportedFrom.end() || it->second.totalLen != info.totalLen)
            {
                return false;
            }
        }
        return true;
    };
    if (mExportDone && sameMap())
    {
        return mExportBlob;
    }
    (void) cudaSetDevice(mDeviceId);
    (void) cudaFree(nullptr); // make the primary context current for the driver calls below

    DirectHandshake hs;
    hs.version = kHandshakeVersion;
    hs.host = mHost;
    hs.bootId = mBootId;
    std::size_t nChunks = 0;
    std::uint64_t bytes = 0;
    auto const t0 = std::chrono::steady_clock::now();
    for (auto const& [base, info] : regions)
    {
        ExportedRegion reg;
        reg.base = base;
        reg.len = info.totalLen;
        std::uint64_t p = base;
        std::uint64_t const end = base + info.totalLen;
        char const* failure = nullptr;
        CUresult failCode = CUDA_SUCCESS;
        while (p < end)
        {
            CUdeviceptr allocBase = 0;
            std::size_t allocSize = 0;
            CUresult r = cuMemGetAddressRange(&allocBase, &allocSize, static_cast<CUdeviceptr>(p));
            if (r != CUDA_SUCCESS || allocSize == 0 || allocBase > p)
            {
                failure = "cuMemGetAddressRange";
                failCode = r;
                break;
            }
            CUmemGenericAllocationHandle h{};
            r = cuMemRetainAllocationHandle(&h, reinterpret_cast<void*>(allocBase));
            if (r != CUDA_SUCCESS)
            {
                failure = "not a VMM allocation (cuMemRetainAllocationHandle)";
                failCode = r;
                break;
            }
            CUmemAllocationProp prop{};
            r = cuMemGetAllocationPropertiesFromHandle(&prop, h);
            bool const fabric = r == CUDA_SUCCESS
                && (static_cast<unsigned>(prop.requestedHandleTypes) & static_cast<unsigned>(CU_MEM_HANDLE_TYPE_FABRIC))
                    != 0U;
            ExportedChunk chunk;
            chunk.va = allocBase;
            chunk.size = allocSize;
            if (fabric)
            {
                r = cuMemExportToShareableHandle(&chunk.handle, h, CU_MEM_HANDLE_TYPE_FABRIC, 0);
            }
            (void) cuMemRelease(h);
            if (!fabric)
            {
                failure = "allocation not created with CU_MEM_HANDLE_TYPE_FABRIC";
                break;
            }
            if (r != CUDA_SUCCESS)
            {
                failure = "cuMemExportToShareableHandle(FABRIC)";
                failCode = r;
                break;
            }
            reg.chunks.push_back(chunk);
            p = allocBase + allocSize;
        }
        if (failure != nullptr)
        {
            TLLM_LOG_WARNING(
                "NixlDirectCopy: region 0x%llx (+%llu B) is not exported (%s, %s): writes into it keep the NIXL path",
                static_cast<unsigned long long>(base), static_cast<unsigned long long>(info.totalLen), failure,
                cuErrStr(failCode));
            continue;
        }
        nChunks += reg.chunks.size();
        bytes += reg.len;
        hs.regions.push_back(std::move(reg));
    }
    double const ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    mExportBlob = hs.regions.empty() ? std::string{} : encodeHandshake(hs);
    mExportedFrom = regions;
    mExportDone = true;
    TLLM_LOG_INFO("NixlDirectCopy: exported %zu of %zu regions (%zu chunks, %.2f GB) in %.1f ms, handshake %zu B",
        hs.regions.size(), regions.size(), nChunks, static_cast<double>(bytes) / 1e9, ms, mExportBlob.size());
    return mExportBlob;
}

void DirectPeerCopy::unmapPeer(MappedPeer& peer) noexcept
{
    for (auto& [base, reg] : peer.regions)
    {
        for (auto const& c : reg.chunks)
        {
            (void) cuMemUnmap(reg.localVa + c.offset, c.size);
        }
        if (reg.localVa != 0)
        {
            (void) cuMemAddressFree(reg.localVa, reg.mapLen);
        }
        reg.chunks.clear();
        reg.localVa = 0;
    }
    peer.regions.clear();
}

void DirectPeerCopy::waitAllInflight() noexcept
{
    if (mPool)
    {
        mPool->synchronizeAll();
    }
}

std::shared_ptr<DirectPeerCopy::MappedPeer> DirectPeerCopy::takeMappedLocked(std::string const& name)
{
    std::unique_lock<std::shared_mutex> lk(mPeersMu);
    auto it = mPeers.find(name);
    if (it == mPeers.end())
    {
        return nullptr;
    }
    auto peer = std::move(it->second);
    mPeers.erase(it);
    return peer;
}

void DirectPeerCopy::releaseMapped(std::shared_ptr<MappedPeer> peer) noexcept
{
    if (peer)
    {
        waitAllInflight();
        unmapPeer(*peer);
    }
}

void DirectPeerCopy::importPeer(std::string const& name, std::string const& blob)
{
    DirectHandshake hs;
    bool const usable = !blob.empty() && decodeHandshake(blob, hs);
    if (blob.empty())
    {
        TLLM_LOG_INFO("NixlDirectCopy: peer %s exports no regions -> NIXL path", name.c_str());
    }
    else if (!usable)
    {
        TLLM_LOG_WARNING(
            "NixlDirectCopy: peer %s sent an undecodable handshake (%zu B) -> NIXL path", name.c_str(), blob.size());
    }
    std::shared_ptr<MappedPeer> old;
    {
        std::lock_guard<std::mutex> lk(mImportMu);
        if (mStopping.load())
        {
            return;
        }
        // A new handshake under a known name replaces whatever this name mapped before.
        old = takeMappedLocked(name);
        if (!usable)
        {
            mGenerations.erase(name);
        }
        else
        {
            std::uint64_t const generation = ++mNextGeneration;
            mGenerations[name] = generation;
            mImportQueue.push_back(ImportTask{name, std::move(hs), generation});
            while (mImportThreads.size() < kImportThreads)
            {
                mImportThreads.emplace_back([this] { importLoop(); });
            }
        }
    }
    mImportCv.notify_one();
    releaseMapped(std::move(old));
}

void DirectPeerCopy::importLoop()
{
    (void) cudaSetDevice(mDeviceId);
    (void) cudaFree(nullptr); // make the primary context current for the driver calls of mapPeer
    for (;;)
    {
        ImportTask task;
        {
            std::unique_lock<std::mutex> lk(mImportMu);
            mImportCv.wait(lk, [this] { return mStopping.load() || !mImportQueue.empty(); });
            if (mStopping.load())
            {
                return;
            }
            task = std::move(mImportQueue.front());
            mImportQueue.pop_front();
            auto it = mGenerations.find(task.name);
            if (it == mGenerations.end() || it->second != task.generation)
            {
                continue; // imported again or forgotten while queued
            }
        }
        auto peer = mapPeer(task.name, task.handshake);
        if (!peer)
        {
            continue;
        }
        bool published = false;
        {
            std::lock_guard<std::mutex> lk(mImportMu);
            auto it = mGenerations.find(task.name);
            if (!mStopping.load() && it != mGenerations.end() && it->second == task.generation)
            {
                std::unique_lock<std::shared_mutex> plk(mPeersMu);
                mPeers[task.name] = peer;
                published = true;
            }
        }
        if (!published)
        {
            unmapPeer(*peer); // never visible to trySubmit, so no copy can be using it
        }
    }
}

std::shared_ptr<DirectPeerCopy::MappedPeer> DirectPeerCopy::mapPeer(std::string const& name, DirectHandshake const& hs)
{
    bool const remoteHost = hs.host != mHost || hs.bootId != mBootId;
    auto peer = std::make_shared<MappedPeer>();
    peer->remoteHost = remoteHost;
    auto const t0 = std::chrono::steady_clock::now();
    std::size_t nChunks = 0;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = mDeviceId;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (auto const& reg : hs.regions)
    {
        if (reg.chunks.empty())
        {
            continue;
        }
        MappedRegion m;
        m.remoteBase = reg.base;
        m.remoteEnd = reg.base + reg.len;
        m.mapStart = reg.chunks.front().va;
        auto const& last = reg.chunks.back();
        m.mapLen = last.va + last.size - m.mapStart;
        bool ordered = m.mapStart <= m.remoteBase && m.remoteEnd <= m.mapStart + m.mapLen;
        for (std::size_t i = 1; ordered && i < reg.chunks.size(); ++i)
        {
            ordered = reg.chunks[i].va >= reg.chunks[i - 1].va + reg.chunks[i - 1].size;
        }
        if (!ordered)
        {
            TLLM_LOG_WARNING("NixlDirectCopy: peer %s region 0x%llx has an inconsistent chunk list -> NIXL path",
                name.c_str(), static_cast<unsigned long long>(reg.base));
            unmapPeer(*peer);
            return nullptr;
        }
        std::uint64_t align = reg.chunks.front().size;
        if ((align & (align - 1)) != 0)
        {
            align = 2ULL << 20;
        }
        CUresult r = cuMemAddressReserve(&m.localVa, m.mapLen, align, 0, 0);
        if (r != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING("NixlDirectCopy: cuMemAddressReserve(%llu B) for peer %s failed (%s) -> NIXL path",
                static_cast<unsigned long long>(m.mapLen), name.c_str(), cuErrStr(r));
            m.localVa = 0;
            unmapPeer(*peer);
            return nullptr;
        }
        char const* failure = nullptr;
        for (auto const& c : reg.chunks)
        {
            if (mStopping.load(std::memory_order_relaxed))
            {
                failure = "shutdown";
                break;
            }
            CUmemGenericAllocationHandle h{};
            r = cuMemImportFromShareableHandle(
                &h, const_cast<void*>(static_cast<void const*>(&c.handle)), CU_MEM_HANDLE_TYPE_FABRIC);
            if (r != CUDA_SUCCESS)
            {
                failure = "cuMemImportFromShareableHandle(FABRIC)";
                break;
            }
            std::uint64_t const offset = c.va - m.mapStart;
            r = cuMemMap(m.localVa + offset, c.size, 0, h, 0);
            (void) cuMemRelease(h); // the mapping holds the reference
            if (r != CUDA_SUCCESS)
            {
                failure = "cuMemMap";
                break;
            }
            m.chunks.push_back({offset, c.size});
            r = cuMemSetAccess(m.localVa + offset, c.size, &access, 1);
            if (r != CUDA_SUCCESS)
            {
                failure = "cuMemSetAccess";
                break;
            }
        }
        std::uint64_t const remoteBase = m.remoteBase;
        nChunks += m.chunks.size();
        peer->bytes += reg.len;
        peer->regions.emplace(remoteBase, std::move(m));
        if (failure != nullptr)
        {
            unmapPeer(*peer);
            if (std::strcmp(failure, "shutdown") == 0)
            {
                return nullptr;
            }
            // Another IMEX channel (a separate job or ComputeDomain) or another NVLink domain refuses the import.
            bool const outsideDomain = remoteHost && (r == CUDA_ERROR_NOT_PERMITTED || r == CUDA_ERROR_NOT_SUPPORTED);
            TLLM_LOG_WARNING(
                "NixlDirectCopy: mapping peer %s (%s host %s) region 0x%llx failed at %s (%s)%s -> NIXL path",
                name.c_str(), remoteHost ? "remote" : "same", hs.host.c_str(),
                static_cast<unsigned long long>(remoteBase), failure, cuErrStr(r),
                outsideDomain ? "; the peer is not in this process's NVLink domain and IMEX channel" : "");
            return nullptr;
        }
    }
    double const ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    TLLM_LOG_INFO("NixlDirectCopy: mapped peer %s (%s host %s): %zu regions, %zu chunks, %.2f GB in %.1f ms",
        name.c_str(), remoteHost ? "remote" : "same", hs.host.c_str(), peer->regions.size(), nChunks,
        static_cast<double>(peer->bytes) / 1e9, ms);
    return peer;
}

void DirectPeerCopy::forgetPeer(std::string const& name)
{
    std::shared_ptr<MappedPeer> old;
    {
        std::lock_guard<std::mutex> lk(mImportMu);
        mGenerations.erase(name); // a queued or running import of this name is now stale
        old = takeMappedLocked(name);
    }
    releaseMapped(std::move(old));
}

void DirectPeerCopy::declined(std::string const& peer, char const* reason, std::size_t descs)
{
    mDeclines.fetch_add(1, std::memory_order_relaxed);
    bool first = false;
    {
        std::lock_guard<std::mutex> lk(mWarnMu);
        auto& w = mWarned[reason];
        first = !w;
        w = true;
    }
    if (first)
    {
        TLLM_LOG_WARNING("NixlDirectCopy: write to %s (%zu descs) declined: %s -> NIXL path (later: DEBUG)",
            peer.c_str(), descs, reason);
    }
    else
    {
        TLLM_LOG_DEBUG("NixlDirectCopy: write to %s (%zu descs) declined: %s", peer.c_str(), descs, reason);
    }
}

std::unique_ptr<TransferStatus> DirectPeerCopy::trySubmit(TransferRequest const& request, VramRegionMap const& local)
{
    if (request.getOp() != TransferOp::kWRITE || request.getSrcDescs().getType() != MemoryType::kVRAM
        || request.getDstDescs().getType() != MemoryType::kVRAM || request.getSyncMessage().has_value())
    {
        return nullptr;
    }
    std::shared_ptr<MappedPeer> peer;
    {
        std::shared_lock<std::shared_mutex> lk(mPeersMu);
        auto it = mPeers.find(request.getRemoteName());
        if (it == mPeers.end())
        {
            return nullptr;
        }
        peer = it->second;
    }
    auto const& srcs = request.getSrcDescs().getDescs();
    auto const& dsts = request.getDstDescs().getDescs();
    std::string const& name = request.getRemoteName();
    if (srcs.size() != dsts.size())
    {
        declined(name, "src/dst descriptor count mismatch", srcs.size());
        return nullptr;
    }
    if (srcs.empty())
    {
        return nullptr;
    }
    // The kernel splits descriptors into work units of at most kUnitBytes.
    std::uint64_t const unitBytes = kUnitBytes;
    std::uint64_t units = 0;
    for (auto const& d : srcs)
    {
        units += (d.getLen() + unitBytes - 1) / unitBytes;
    }
    if (units > std::numeric_limits<std::uint32_t>::max() / 2)
    {
        declined(name, "too many work units", srcs.size());
        return nullptr;
    }
    int cur = -1;
    if (cudaGetDevice(&cur) != cudaSuccess || cur != mDeviceId)
    {
        (void) cudaSetDevice(mDeviceId);
    }
    CopyCtx* ctx = mPool->acquire(static_cast<std::size_t>(units));
    if (ctx == nullptr)
    {
        declined(name, "no copy context", srcs.size());
        return nullptr;
    }
    auto* hSrc = reinterpret_cast<std::uint64_t*>(ctx->host);
    auto* hDst = hSrc + units;
    auto* hSize = reinterpret_cast<std::uint32_t*>(hDst + units);
    std::uint64_t n = 0;
    std::uint64_t bytes = 0;
    char const* reason = nullptr;
    // The peer's regions are few (one per pool): find each destination's region from the previous hit first.
    MappedRegion const* hit = nullptr;
    for (std::size_t i = 0; i < srcs.size(); ++i)
    {
        auto const& s = srcs[i];
        auto const& d = dsts[i];
        std::uint64_t const len = s.getLen();
        if (len != d.getLen())
        {
            reason = "src/dst length mismatch";
            break;
        }
        if (len == 0)
        {
            continue;
        }
        if (static_cast<int>(s.getDeviceId()) != mDeviceId)
        {
            reason = "source on another device";
            break;
        }
        std::uint64_t const sa = s.getAddr();
        std::uint64_t const da = d.getAddr();
        if (!lookupRegion(local, sa, len))
        {
            reason = "source outside the local registered regions";
            break;
        }
        if (hit == nullptr || da < hit->remoteBase || da + len > hit->remoteEnd)
        {
            hit = nullptr;
            auto it = peer->regions.upper_bound(da);
            if (it != peer->regions.begin())
            {
                --it;
                if (da >= it->second.remoteBase && da + len <= it->second.remoteEnd && da + len >= da)
                {
                    hit = &it->second;
                }
            }
            if (hit == nullptr)
            {
                reason = "destination outside the peer's mapped regions";
                break;
            }
        }
        std::uint64_t const ld = hit->localVa + (da - hit->mapStart);
        for (std::uint64_t off = 0; off < len; off += unitBytes)
        {
            hSrc[n] = sa + off;
            hDst[n] = ld + off;
            hSize[n] = static_cast<std::uint32_t>(std::min<std::uint64_t>(unitBytes, len - off));
            ++n;
        }
        bytes += len;
    }
    if (reason != nullptr)
    {
        mPool->release(ctx);
        declined(name, reason, srcs.size());
        return nullptr;
    }
    if (n == 0)
    {
        mPool->release(ctx);
        return nullptr;
    }
    // Compact the size array behind the n used src/dst entries: the kernel reads the plan as three packed arrays
    // through the device alias of ctx->host.
    auto* pDst = hSrc + n;
    auto* pSize = reinterpret_cast<std::uint32_t*>(pDst + n);
    if (n != units)
    {
        std::memmove(pDst, hDst, n * sizeof(std::uint64_t));
        std::memmove(pSize, hSize, n * sizeof(std::uint32_t));
    }
    auto const* dSrc = reinterpret_cast<std::uint64_t const*>(ctx->dev);
    auto const* dDst = dSrc + n;
    auto const* dSize = reinterpret_cast<std::uint32_t const*>(dDst + n);
    // Head slices of KV blocks (a few KiB each) leave most of a CTA idle when one CTA copies one unit.
    bool const warpPerUnit = bytes <= n * kWarpUnitMeanBytes;
    auto const nUnits = static_cast<std::uint32_t>(n);
    cudaError_t e = warpPerUnit
        ? launchDirectCopyWarpPerEntry(dSrc, dDst, dSize, nUnits, kMaxCtas, kThreads, ctx->stream)
        : launchDirectCopy(dSrc, dDst, dSize, nUnits, kMaxCtas, kThreads, ctx->stream);
    if (e == cudaSuccess)
    {
        e = cudaEventRecord(ctx->event, ctx->stream);
    }
    if (e != cudaSuccess)
    {
        (void) cudaStreamSynchronize(ctx->stream);
        mPool->release(ctx);
        TLLM_LOG_ERROR("NixlDirectCopy: copy launch to %s failed: %s", name.c_str(), cudaGetErrorName(e));
        return std::make_unique<DirectTransferStatus>(
            std::string("direct peer copy launch failed: ") + cudaGetErrorName(e));
    }
    if (peer->remoteHost)
    {
        mCrossNodeSubmits.fetch_add(1, std::memory_order_relaxed);
    }
    noteSubmit(name, srcs.size(), n, bytes);
    return std::make_unique<DirectTransferStatus>(mPool, ctx);
}

void DirectPeerCopy::noteSubmit(std::string const& name, std::size_t descs, std::uint64_t units, std::uint64_t bytes)
{
    std::uint64_t const k = mSubmits.fetch_add(1, std::memory_order_relaxed) + 1;
    mDescs.fetch_add(descs, std::memory_order_relaxed);
    mBytes.fetch_add(bytes, std::memory_order_relaxed);
    if (k == 1 || k % kLogEvery == 0)
    {
        TLLM_LOG_INFO(
            "NixlDirectCopy: %llu direct writes (%llu to other nodes, %llu descs, %.2f GB, last to %s: %zu descs / "
            "%llu entries / %.1f MB), %llu declined",
            static_cast<unsigned long long>(k), static_cast<unsigned long long>(mCrossNodeSubmits.load()),
            static_cast<unsigned long long>(mDescs.load()), static_cast<double>(mBytes.load()) / 1e9, name.c_str(),
            descs, static_cast<unsigned long long>(units), static_cast<double>(bytes) / 1e6,
            static_cast<unsigned long long>(mDeclines.load()));
    }
}

} // namespace tensorrt_llm::executor::kv_cache::direct
