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

#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/serializeUtils.h"

#include <algorithm>
#include <cuda.h>
#include <dlfcn.h>
#include <mutex>
#include <numeric>
#include <sstream>
#include <tuple>

namespace tensorrt_llm::executor::kv_cache
{

void promoteHostLibraryToGlobalScope()
{
    static std::once_flag once;
    std::call_once(once,
        []()
        {
            Dl_info info{};
            if (dladdr(reinterpret_cast<void*>(&promoteHostLibraryToGlobalScope), &info) == 0
                || info.dli_fname == nullptr)
            {
                TLLM_LOG_DEBUG("dladdr failed; skipping global-scope promotion");
                return;
            }
            // RTLD_NOLOAD promotes the visibility of the already-loaded library without
            // loading a second copy. It fails when this code is statically linked into an
            // executable (e.g. unit tests) — nothing to promote there.
            void* handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
            if (handle == nullptr)
            {
                TLLM_LOG_DEBUG("global-scope promotion skipped for %s: %s", info.dli_fname, dlerror());
                return;
            }
            TLLM_LOG_DEBUG("promoted %s to the global symbol scope", info.dli_fname);
            // Keep the extra reference: the library must stay resident for the wrappers.
        });
}

[[nodiscard]] DynLibLoader& DynLibLoader::getInstance()
{
    static DynLibLoader instance;
    return instance;
}

[[nodiscard]] void* DynLibLoader::getHandle(std::string const& name)
{
    promoteHostLibraryToGlobalScope();
    std::lock_guard<std::mutex> lock(mDllMutex);
    auto it = mHandlers.find(name);
    if (it != mHandlers.end())
    {
        return it->second;
    }
    TLLM_LOG_INFO("dlopen: " + name);
    void* handler = dlopen(name.c_str(), RTLD_LAZY);
    TLLM_CHECK_WITH_INFO(handler, "%s can not be loaded correctly: %s", name.c_str(), dlerror());
    mHandlers[name] = handler;
    return handler;
}

[[nodiscard]] void* DynLibLoader::dlSym(void* handle, char const* symbol)
{
    return dlsym(handle, symbol);
}

DynLibLoader::~DynLibLoader()
{
    std::lock_guard<std::mutex> lock(mDllMutex);
    for (auto const& pair : mHandlers)
    {
        dlclose(pair.second);
    }
}

std::string AgentDesc::serialize() const
{
    namespace su = executor::serialize_utils;
    std::ostringstream os;
    su::serialize(mBackendAgentDesc, os);
    size_t numRegions = mVramRegions.size();
    su::serialize(numRegions, os);
    for (auto const& r : mVramRegions)
    {
        su::serialize(r.baseAddr, os);
        su::serialize(r.totalLen, os);
        su::serialize(r.chunkSize, os);
    }
    su::serialize(mBounceHandshake, os); // NIXL-bounce capability handshake (empty when bounce disabled)
    return os.str();
}

AgentDesc AgentDesc::deserialize(std::string const& data)
{
    namespace su = executor::serialize_utils;
    std::istringstream is(data);
    auto backendAgentDesc = su::deserialize<std::string>(is);
    auto numRegions = su::deserialize<size_t>(is);
    std::vector<VramRegionMeta> regions;
    regions.reserve(numRegions);
    for (size_t i = 0; i < numRegions; ++i)
    {
        auto baseAddr = su::deserialize<uintptr_t>(is);
        auto totalLen = su::deserialize<size_t>(is);
        auto chunkSize = su::deserialize<size_t>(is);
        regions.push_back({baseAddr, totalLen, chunkSize});
    }
    // Peers built before the handshake field existed end the stream right after the regions. An empty handshake
    // means "bounce not advertised" and the peer stays on standard NIXL (registerPeerHandshake treats it that way).
    std::string bounceHandshake;
    if (is.peek() != std::char_traits<char>::eof())
    {
        bounceHandshake = su::deserialize<std::string>(is);
    }
    TLLM_CHECK_WITH_INFO(!is.fail(),
        "AgentDesc::deserialize failed: stream error after reading %zu/%zu regions (data size=%zu)", regions.size(),
        numRegions, data.size());
    return AgentDesc{std::move(backendAgentDesc), std::move(regions), std::move(bounceHandshake)};
}

// ── VmmDescSplitter utilities (backend-agnostic, no NIXL dependency) ──

std::pair<size_t, uintptr_t> VmmDescSplitter::lookupChunkInfo(uintptr_t addr, VramRegionMap const& regionMap)
{
    auto it = regionMap.upper_bound(addr);
    if (it != regionMap.begin())
    {
        --it;
        if (addr >= it->first && addr < it->first + it->second.totalLen)
        {
            return {it->second.chunkSize, it->first};
        }
    }
    return {0, 0};
}

MemoryDescs VmmDescSplitter::splitDescsWithRegionMap(MemoryDescs const& descs, VramRegionMap const& regionMap)
{
    if (descs.getType() != MemoryType::kVRAM)
        return descs;

    auto const& descVec = descs.getDescs();
    if (descVec.empty())
        return descs;

    std::vector<MemoryDesc> result;
    result.reserve(descVec.size());

    for (auto const& desc : descVec)
    {
        auto [chunkSize, regionBase] = lookupChunkInfo(desc.getAddr(), regionMap);
        if (chunkSize == 0)
        {
            result.push_back(desc);
            continue;
        }

        uintptr_t addr = desc.getAddr();
        size_t remaining = desc.getLen();
        uint32_t deviceId = desc.getDeviceId();

        while (remaining > 0)
        {
            // A region is one run of equal chunks; a pool mapped from chunks of several sizes is
            // several adjacent regions, so the region is looked up again for every piece.
            auto [pieceChunkSize, pieceBase] = lookupChunkInfo(addr, regionMap);
            size_t pieceSize = remaining;
            if (pieceChunkSize > 0)
            {
                size_t offsetInChunk = static_cast<size_t>((addr - pieceBase) % pieceChunkSize);
                pieceSize = std::min(remaining, pieceChunkSize - offsetInChunk);
            }
            result.emplace_back(addr, pieceSize, deviceId);
            addr += pieceSize;
            remaining -= pieceSize;
        }
    }

    return MemoryDescs{descs.getType(), std::move(result)};
}

std::pair<MemoryDescs, MemoryDescs> VmmDescSplitter::splitAndCoalesceTransferDescs(MemoryDescs const& srcDescs,
    MemoryDescs const& dstDescs, VramRegionMap const& localRegionMap, VramRegionMap const& remoteRegionMap,
    bool enableCoalesce)
{
    if (srcDescs.getType() != MemoryType::kVRAM)
        return {srcDescs, dstDescs};

    auto const& srcVec = srcDescs.getDescs();
    auto const& dstVec = dstDescs.getDescs();
    TLLM_CHECK(srcVec.size() == dstVec.size());
    if (srcVec.empty())
        return {srcDescs, dstDescs};

    // Sort pair indices by (src deviceId, src addr) so pairs that are contiguous in memory become
    // adjacent, maximizing coalescing regardless of input order. Pairs are independent transfers,
    // so reordering is safe. The sort only serves coalescing; with it disabled, keep input order.
    std::vector<size_t> order(srcVec.size());
    std::iota(order.begin(), order.end(), 0);
    if (enableCoalesce)
    {
        std::sort(order.begin(), order.end(),
            [&srcVec](size_t lhs, size_t rhs)
            {
                if (srcVec[lhs].getDeviceId() != srcVec[rhs].getDeviceId())
                {
                    return srcVec[lhs].getDeviceId() < srcVec[rhs].getDeviceId();
                }
                return srcVec[lhs].getAddr() < srcVec[rhs].getAddr();
            });
    }

    std::vector<MemoryDesc> outSrc, outDst;
    outSrc.reserve(srcVec.size());
    outDst.reserve(dstVec.size());

    // Region info of the last emitted piece. Invariant: every emitted desc lies within a single
    // chunk on both sides, so a merge is legal iff the new piece is contiguous, in the same region,
    // and does not start on a chunk boundary (starting on a boundary means the merge would cross it).
    size_t prevSrcChunkSize = 0, prevDstChunkSize = 0;
    uintptr_t prevSrcBase = 0, prevDstBase = 0;

    auto emitPiece
        = [&](uintptr_t srcAddr, uintptr_t dstAddr, size_t len, uint32_t srcDev, uint32_t dstDev, size_t srcChunkSize,
              uintptr_t srcBase, size_t dstChunkSize, uintptr_t dstBase, bool regionsKnown)
    {
        if (enableCoalesce && regionsKnown && !outSrc.empty())
        {
            auto const& lastSrc = outSrc.back();
            auto const& lastDst = outDst.back();
            bool contiguous = lastSrc.getAddr() + lastSrc.getLen() == srcAddr && lastSrc.getDeviceId() == srcDev
                && lastDst.getAddr() + lastDst.getLen() == dstAddr && lastDst.getDeviceId() == dstDev;
            bool sameSrcRegion = srcChunkSize == prevSrcChunkSize && srcBase == prevSrcBase;
            bool sameDstRegion = dstChunkSize == prevDstChunkSize && dstBase == prevDstBase;
            bool srcWithinChunk = srcChunkSize == 0 || (srcAddr - srcBase) % srcChunkSize != 0;
            bool dstWithinChunk = dstChunkSize == 0 || (dstAddr - dstBase) % dstChunkSize != 0;
            if (contiguous && sameSrcRegion && sameDstRegion && srcWithinChunk && dstWithinChunk)
            {
                outSrc.back() = MemoryDesc{lastSrc.getAddr(), lastSrc.getLen() + len, srcDev};
                outDst.back() = MemoryDesc{lastDst.getAddr(), lastDst.getLen() + len, dstDev};
                return;
            }
        }
        outSrc.emplace_back(srcAddr, len, srcDev);
        outDst.emplace_back(dstAddr, len, dstDev);
        prevSrcChunkSize = srcChunkSize;
        prevSrcBase = srcBase;
        prevDstChunkSize = dstChunkSize;
        prevDstBase = dstBase;
    };

    // One-entry region cache per side: after sorting, consecutive pairs almost always fall in the
    // same region (typically one KV pool), so the O(log R) map lookup is skipped on cache hits.
    struct RegionCache
    {
        uintptr_t base = 0;
        size_t totalLen = 0;
        size_t chunkSize = 0;
        bool valid = false;
    };

    // Returns {chunkSize, regionBase, found}. A miss means the address is not covered by any
    // region metadata (e.g. the peer did not send its region info). Two misses both look like
    // {0, 0} yet may belong to two distinct regions, so pieces with a missed lookup on either
    // side are never merged — a merge could silently cross a chunk or registration boundary.
    auto cachedLookup = [](uintptr_t addr, VramRegionMap const& regionMap, RegionCache& cache)
    {
        if (cache.valid && addr >= cache.base && addr - cache.base < cache.totalLen)
        {
            return std::tuple<size_t, uintptr_t, bool>{cache.chunkSize, cache.base, true};
        }
        auto it = regionMap.upper_bound(addr);
        if (it != regionMap.begin())
        {
            --it;
            if (addr >= it->first && addr - it->first < it->second.totalLen)
            {
                cache = {it->first, it->second.totalLen, it->second.chunkSize, true};
                return std::tuple<size_t, uintptr_t, bool>{cache.chunkSize, cache.base, true};
            }
        }
        return std::tuple<size_t, uintptr_t, bool>{0, 0, false};
    };

    RegionCache srcCache, dstCache;
    size_t numPieces = 0;
    for (size_t idx : order)
    {
        auto const& src = srcVec[idx];
        auto const& dst = dstVec[idx];

        uintptr_t srcAddr = src.getAddr();
        uintptr_t dstAddr = dst.getAddr();
        size_t remaining = src.getLen();

        while (remaining > 0)
        {
            // Looked up per piece: a pool mapped from chunks of several sizes is several adjacent
            // regions, and a pair can cross from one into the next.
            auto [srcChunkSize, srcBase, srcFound] = cachedLookup(srcAddr, localRegionMap, srcCache);
            auto [dstChunkSize, dstBase, dstFound] = cachedLookup(dstAddr, remoteRegionMap, dstCache);
            size_t srcPieceSize = remaining;
            if (srcChunkSize > 0)
            {
                srcPieceSize = srcChunkSize - static_cast<size_t>((srcAddr - srcBase) % srcChunkSize);
            }

            size_t dstPieceSize = remaining;
            if (dstChunkSize > 0)
            {
                dstPieceSize = dstChunkSize - static_cast<size_t>((dstAddr - dstBase) % dstChunkSize);
            }

            size_t pieceSize = std::min({remaining, srcPieceSize, dstPieceSize});
            emitPiece(srcAddr, dstAddr, pieceSize, src.getDeviceId(), dst.getDeviceId(), srcChunkSize, srcBase,
                dstChunkSize, dstBase, srcFound && dstFound);
            srcAddr += pieceSize;
            dstAddr += pieceSize;
            remaining -= pieceSize;
            ++numPieces;
        }
    }

    if (outSrc.size() != srcVec.size())
    {
        TLLM_LOG_DEBUG("VmmDescSplitter::splitAndCoalesceTransferDescs: %zu pairs -> %zu pieces -> %zu transfers",
            srcVec.size(), numPieces, outSrc.size());
    }

    return {MemoryDescs{srcDescs.getType(), std::move(outSrc)}, MemoryDescs{dstDescs.getType(), std::move(outDst)}};
}

namespace
{

//! One run of equal-size VMM chunks inside a descriptor: [base, base + len), chunks of chunkSize
//! bytes from base (chunkSize 0: the descriptor lies inside one allocation and is not split).
struct VmmRun
{
    uintptr_t base;
    size_t len;
    size_t chunkSize;
};

//! The runs of `desc`, found by walking its allocations with cuMemGetAddressRange. A descriptor
//! inside one allocation (or not VMM memory) is one run with chunkSize 0. Chunk sizes may differ
//! from run to run (a KV-V2 pool mapped from larger chunks where its offsets allow it).
std::vector<VmmRun> vmmRunsOf(MemoryDesc const& desc)
{
    uintptr_t const addr = desc.getAddr();
    size_t const len = desc.getLen();
    CUdeviceptr startBase = 0, endBase = 0;
    size_t startSize = 0, endSize = 0;
    CUresult const startErr = cuMemGetAddressRange(&startBase, &startSize, static_cast<CUdeviceptr>(addr));
    CUresult const endErr = cuMemGetAddressRange(&endBase, &endSize, static_cast<CUdeviceptr>(addr + len - 1));
    // If either query fails, or both are in the same allocation -> no split needed
    if (startErr != CUDA_SUCCESS || endErr != CUDA_SUCCESS || startBase == endBase)
    {
        return {VmmRun{addr, len, 0}};
    }
    TLLM_CHECK_WITH_INFO(addr == static_cast<uintptr_t>(startBase),
        "VMM descriptor start address 0x%lx does not match chunk base 0x%lx. "
        "Descriptor must start at a chunk boundary.",
        static_cast<unsigned long>(addr), static_cast<unsigned long>(startBase));

    std::vector<VmmRun> runs;
    uintptr_t const end = addr + len;
    uintptr_t cur = addr;
    while (cur < end)
    {
        CUdeviceptr base = 0;
        size_t size = 0;
        CUresult const err = cuMemGetAddressRange(&base, &size, static_cast<CUdeviceptr>(cur));
        TLLM_CHECK_WITH_INFO(err == CUDA_SUCCESS && static_cast<uintptr_t>(base) == cur && size > 0,
            "VMM descriptor [0x%lx, 0x%lx) has an unmapped hole or misaligned chunk at 0x%lx.",
            static_cast<unsigned long>(addr), static_cast<unsigned long>(end), static_cast<unsigned long>(cur));
        size_t const piece = std::min(size, static_cast<size_t>(end - cur));
        if (!runs.empty() && runs.back().chunkSize == size && runs.back().base + runs.back().len == cur
            && runs.back().len % size == 0)
        {
            runs.back().len += piece;
        }
        else
        {
            runs.push_back(VmmRun{cur, piece, size});
        }
        cur += piece;
    }
    return runs;
}

} // namespace

MemoryDescs VmmDescSplitter::splitVmmDescs(MemoryDescs const& descs, size_t& detectedChunkSize)
{
    detectedChunkSize = 0;
    auto const& descVec = descs.getDescs();
    if (descVec.empty() || descs.getType() != MemoryType::kVRAM)
        return descs;

    std::vector<MemoryDesc> result;
    result.reserve(descVec.size());

    for (auto const& desc : descVec)
    {
        uint32_t deviceId = desc.getDeviceId();
        if (desc.getLen() == 0)
            continue;

        for (auto const& run : vmmRunsOf(desc))
        {
            if (run.chunkSize == 0)
            {
                result.emplace_back(run.base, run.len, deviceId);
                continue;
            }
            // The largest chunk seen, for logging: chunks may differ in size within and across descs.
            detectedChunkSize = std::max(detectedChunkSize, run.chunkSize);
            for (size_t off = 0; off < run.len; off += run.chunkSize)
            {
                result.emplace_back(run.base + off, std::min(run.chunkSize, run.len - off), deviceId);
            }
        }
    }

    if (result.size() != descVec.size())
    {
        TLLM_LOG_DEBUG("VmmDescSplitter::splitVmmDescs: split %zu -> %zu VRAM entries (max chunkSize=%zu)",
            descVec.size(), result.size(), detectedChunkSize);
    }

    return MemoryDescs{descs.getType(), std::move(result)};
}

VramRegionMap VmmDescSplitter::detectVramRegionMap(MemoryDescs const& descs)
{
    VramRegionMap regionMap;
    if (descs.getType() != MemoryType::kVRAM)
        return regionMap;

    for (auto const& desc : descs.getDescs())
    {
        if (desc.getLen() <= 1)
        {
            regionMap[desc.getAddr()] = {desc.getLen(), 0};
            continue;
        }
        // One region per run of equal chunks; a pool mapped from one chunk size is one region.
        for (auto const& run : vmmRunsOf(desc))
        {
            regionMap[run.base] = {run.len, run.chunkSize};
        }
    }
    return regionMap;
}

} // namespace tensorrt_llm::executor::kv_cache
