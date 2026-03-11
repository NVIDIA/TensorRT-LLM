/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <numeric>
#include <sstream>

namespace tensorrt_llm::executor::kv_cache
{

[[nodiscard]] DynLibLoader& DynLibLoader::getInstance()
{
    static DynLibLoader instance;
    return instance;
}

[[nodiscard]] void* DynLibLoader::getHandle(std::string const& name)
{
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
    TLLM_CHECK_WITH_INFO(!is.fail(),
        "AgentDesc::deserialize failed: stream error after reading %zu/%zu regions (data size=%zu)", regions.size(),
        numRegions, data.size());
    return AgentDesc{std::move(backendAgentDesc), std::move(regions)};
}

// ── VmmDescSplitter utilities (backend-agnostic, no NIXL dependency) ──

VmmDescSplitter::RegionLookupResult VmmDescSplitter::lookupChunkInfo(uintptr_t addr, VramRegionMap const& regionMap)
{
    auto it = regionMap.upper_bound(addr);
    if (it != regionMap.begin())
    {
        --it;
        if (addr >= it->first && addr < it->first + it->second.totalLen)
        {
            return {it->second.chunkSize, it->first, it->second.totalLen};
        }
    }
    return {0, 0, 0};
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
        uintptr_t addr = desc.getAddr();
        size_t remaining = desc.getLen();
        uint32_t deviceId = desc.getDeviceId();

        while (remaining > 0)
        {
            auto lookup = lookupChunkInfo(addr, regionMap);
            if (lookup.regionBase == 0)
            {
                // Not in any known region — emit remainder as-is.
                result.emplace_back(addr, remaining, deviceId);
                break;
            }

            // Respect region boundary.
            size_t regionRemaining = static_cast<size_t>((lookup.regionBase + lookup.regionTotalLen) - addr);
            size_t pieceSize = std::min(remaining, regionRemaining);

            // Also respect chunk boundary when chunks are present.
            if (lookup.chunkSize > 0)
            {
                size_t offsetInChunk = static_cast<size_t>((addr - lookup.regionBase) % lookup.chunkSize);
                size_t chunkRemaining = lookup.chunkSize - offsetInChunk;
                pieceSize = std::min(pieceSize, chunkRemaining);
            }

            result.emplace_back(addr, pieceSize, deviceId);
            addr += pieceSize;
            remaining -= pieceSize;
        }
    }

    return MemoryDescs{descs.getType(), std::move(result)};
}

namespace
{

/// Compute the maximum piece size at @p addr that does not cross any region or chunk boundary.
/// Returns @p remaining when no region info is available.
size_t computePieceLimit(uintptr_t addr, size_t remaining, VmmDescSplitter::RegionLookupResult const& lookup)
{
    if (lookup.regionBase == 0)
    {
        return remaining;
    }
    // Respect region boundary.
    size_t regionRemaining = static_cast<size_t>((lookup.regionBase + lookup.regionTotalLen) - addr);
    size_t limit = std::min(remaining, regionRemaining);
    // Also respect chunk boundary when chunks are present.
    if (lookup.chunkSize > 0)
    {
        size_t offsetInChunk = static_cast<size_t>((addr - lookup.regionBase) % lookup.chunkSize);
        size_t chunkRemaining = lookup.chunkSize - offsetInChunk;
        limit = std::min(limit, chunkRemaining);
    }
    return limit;
}

/// Check whether a piece starting at @p addr can be merged with the previous piece
/// (same registered memory region and same physical chunk).
bool canCoalesceWith(VmmDescSplitter::RegionLookupResult const& curLookup,
    VmmDescSplitter::RegionLookupResult const& prevLookup, uintptr_t addr)
{
    // Both must be in a known registered region, and the same one.
    // If either side has no region info, we cannot verify boundaries → refuse merge.
    if (curLookup.regionBase == 0 || prevLookup.regionBase == 0)
    {
        return false;
    }
    if (curLookup.regionBase != prevLookup.regionBase)
    {
        return false;
    }
    // Same region. Check chunk boundary: addr must NOT be on a chunk boundary.
    // If addr is exactly on a chunk boundary, the previous piece ended at one physical
    // chunk and this piece starts the next → different physical chunks → refuse merge.
    if (curLookup.chunkSize > 0)
    {
        if ((addr - curLookup.regionBase) % curLookup.chunkSize == 0)
        {
            return false;
        }
    }
    return true;
}

} // namespace

namespace
{

/// Split a single (src, dst) descriptor pair at chunk and region boundaries.
/// When @p enableCoalesce is true, tries to merge the emitted pieces with the
/// last piece already in splitSrc / splitDst.
template <bool kEnableCoalesce>
void splitOnePair(MemoryDesc const& srcDesc, MemoryDesc const& dstDesc, VramRegionMap const& localRegionMap,
    VramRegionMap const& remoteRegionMap, VmmDescSplitter::RegionLookupResult& cachedSrcLookup,
    VmmDescSplitter::RegionLookupResult& cachedDstLookup, std::vector<MemoryDesc>& splitSrc,
    std::vector<MemoryDesc>& splitDst)
{
    uintptr_t srcAddr = srcDesc.getAddr();
    uintptr_t dstAddr = dstDesc.getAddr();
    size_t remaining = srcDesc.getLen();
    uint32_t srcDeviceId = srcDesc.getDeviceId();
    uint32_t dstDeviceId = dstDesc.getDeviceId();

    auto srcLookup = VmmDescSplitter::lookupChunkInfo(srcAddr, localRegionMap);
    auto dstLookup = VmmDescSplitter::lookupChunkInfo(dstAddr, remoteRegionMap);

    // Fast path: neither side in any known region → emit as-is, skip while loop.
    if (srcLookup.regionBase == 0 && dstLookup.regionBase == 0)
    {
        splitSrc.emplace_back(srcAddr, remaining, srcDeviceId);
        splitDst.emplace_back(dstAddr, remaining, dstDeviceId);
        return;
    }

    while (remaining > 0)
    {
        // Re-lookup if address has advanced past its current region.
        if (srcLookup.regionBase != 0 && srcAddr >= srcLookup.regionBase + srcLookup.regionTotalLen)
        {
            srcLookup = VmmDescSplitter::lookupChunkInfo(srcAddr, localRegionMap);
        }
        if (dstLookup.regionBase != 0 && dstAddr >= dstLookup.regionBase + dstLookup.regionTotalLen)
        {
            dstLookup = VmmDescSplitter::lookupChunkInfo(dstAddr, remoteRegionMap);
        }

        size_t srcLimit = computePieceLimit(srcAddr, remaining, srcLookup);
        size_t dstLimit = computePieceLimit(dstAddr, remaining, dstLookup);
        size_t pieceSize = std::min({remaining, srcLimit, dstLimit});

        // Try to coalesce with the last emitted piece.
        bool merged = false;
        if constexpr (kEnableCoalesce)
        {
            if (!splitSrc.empty())
            {
                auto const& lastSrc = splitSrc.back();
                auto const& lastDst = splitDst.back();

                bool srcContig
                    = (lastSrc.getAddr() + lastSrc.getLen() == srcAddr) && (lastSrc.getDeviceId() == srcDeviceId);
                bool dstContig
                    = (lastDst.getAddr() + lastDst.getLen() == dstAddr) && (lastDst.getDeviceId() == dstDeviceId);

                if (srcContig && dstContig && canCoalesceWith(srcLookup, cachedSrcLookup, srcAddr)
                    && canCoalesceWith(dstLookup, cachedDstLookup, dstAddr))
                {
                    splitSrc.back() = MemoryDesc(lastSrc.getAddr(), lastSrc.getLen() + pieceSize, srcDeviceId);
                    splitDst.back() = MemoryDesc(lastDst.getAddr(), lastDst.getLen() + pieceSize, dstDeviceId);
                    merged = true;
                }
            }
        }

        if (!merged)
        {
            splitSrc.emplace_back(srcAddr, pieceSize, srcDeviceId);
            splitDst.emplace_back(dstAddr, pieceSize, dstDeviceId);
            cachedSrcLookup = srcLookup;
            cachedDstLookup = dstLookup;
        }

        srcAddr += pieceSize;
        dstAddr += pieceSize;
        remaining -= pieceSize;
    }
}

} // namespace

std::pair<MemoryDescs, MemoryDescs> VmmDescSplitter::splitTransferDescsWithRegionMaps(MemoryDescs const& srcDescs,
    MemoryDescs const& dstDescs, VramRegionMap const& localRegionMap, VramRegionMap const& remoteRegionMap,
    bool enableCoalesce)
{
    if (srcDescs.getType() != MemoryType::kVRAM)
        return {srcDescs, dstDescs};

    auto const& srcVec = srcDescs.getDescs();
    auto const& dstVec = dstDescs.getDescs();
    size_t const numDescs = srcVec.size();
    TLLM_CHECK(numDescs == dstVec.size());

    std::vector<MemoryDesc> splitSrc, splitDst;
    splitSrc.reserve(numDescs);
    splitDst.reserve(numDescs);

    RegionLookupResult cachedSrcLookup{0, 0, 0};
    RegionLookupResult cachedDstLookup{0, 0, 0};

    if (enableCoalesce)
    {
        // Sort descriptor pairs by (src deviceId, src addr) so that contiguous
        // KV blocks become adjacent regardless of upper-layer ordering.
        std::vector<size_t> order(numDescs);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
            [&srcVec](size_t lhs, size_t rhs)
            {
                if (srcVec[lhs].getDeviceId() != srcVec[rhs].getDeviceId())
                {
                    return srcVec[lhs].getDeviceId() < srcVec[rhs].getDeviceId();
                }
                return srcVec[lhs].getAddr() < srcVec[rhs].getAddr();
            });
        for (size_t idx = 0; idx < numDescs; ++idx)
        {
            splitOnePair<true>(srcVec[order[idx]], dstVec[order[idx]], localRegionMap, remoteRegionMap, cachedSrcLookup,
                cachedDstLookup, splitSrc, splitDst);
        }
    }
    else
    {
        // Fast path: no allocation, no sort, coalesce code compiled out via template.
        for (size_t i = 0; i < numDescs; ++i)
        {
            splitOnePair<false>(srcVec[i], dstVec[i], localRegionMap, remoteRegionMap, cachedSrcLookup, cachedDstLookup,
                splitSrc, splitDst);
        }
    }

    return {MemoryDescs{srcDescs.getType(), std::move(splitSrc)}, MemoryDescs{dstDescs.getType(), std::move(splitDst)}};
}

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
        uintptr_t addr = desc.getAddr();
        size_t len = desc.getLen();
        uint32_t deviceId = desc.getDeviceId();

        if (len == 0)
            continue;

        // Query start and end of the descriptor
        CUdeviceptr startBase = 0;
        size_t startSize = 0;
        CUresult startErr = cuMemGetAddressRange(&startBase, &startSize, static_cast<CUdeviceptr>(addr));

        CUdeviceptr endBase = 0;
        size_t endSize = 0;
        CUresult endErr = cuMemGetAddressRange(&endBase, &endSize, static_cast<CUdeviceptr>(addr + len - 1));

        // If either query fails, or both are in the same allocation -> no split needed
        if (startErr != CUDA_SUCCESS || endErr != CUDA_SUCCESS || startBase == endBase)
        {
            result.emplace_back(addr, len, deviceId);
            continue;
        }

        // Multi-chunk VMM detected: use first chunk's size as uniform chunk size
        size_t chunkSize = startSize;

        TLLM_CHECK_WITH_INFO(endSize == chunkSize,
            "VMM chunk size mismatch: first chunk %zu bytes, last chunk %zu bytes. "
            "All VMM chunks must be the same size.",
            chunkSize, endSize);

        TLLM_CHECK_WITH_INFO(addr == static_cast<uintptr_t>(startBase),
            "VMM descriptor start address 0x%lx does not match chunk base 0x%lx. "
            "Descriptor must start at a chunk boundary.",
            static_cast<unsigned long>(addr), static_cast<unsigned long>(startBase));

        if (detectedChunkSize > 0)
        {
            TLLM_CHECK_WITH_INFO(chunkSize == detectedChunkSize,
                "Inconsistent VMM chunk sizes across descriptors: %zu vs %zu. "
                "All VMM pools must use the same chunk size.",
                chunkSize, detectedChunkSize);
        }
        detectedChunkSize = chunkSize;

        uintptr_t current = addr;
        size_t remaining = len;

        while (remaining > 0)
        {
            size_t offsetInChunk = static_cast<size_t>((current - static_cast<uintptr_t>(startBase)) % chunkSize);
            size_t pieceSize = std::min(remaining, chunkSize - offsetInChunk);
            result.emplace_back(current, pieceSize, deviceId);
            current += pieceSize;
            remaining -= pieceSize;
        }
    }

    if (result.size() != descVec.size())
    {
        TLLM_LOG_DEBUG("VmmDescSplitter::splitVmmDescs: split %zu -> %zu VRAM entries (chunkSize=%zu)", descVec.size(),
            result.size(), detectedChunkSize);
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
        uintptr_t addr = desc.getAddr();
        size_t len = desc.getLen();
        size_t chunkSize = 0;

        if (len > 1)
        {
            CUdeviceptr startBase = 0, endBase = 0;
            size_t startSize = 0, endSize = 0;
            CUresult startErr = cuMemGetAddressRange(&startBase, &startSize, static_cast<CUdeviceptr>(addr));
            CUresult endErr = cuMemGetAddressRange(&endBase, &endSize, static_cast<CUdeviceptr>(addr + len - 1));
            if (startErr == CUDA_SUCCESS && endErr == CUDA_SUCCESS && startBase != endBase)
            {
                chunkSize = startSize;
            }
        }

        regionMap[addr] = {len, chunkSize};
    }
    return regionMap;
}

} // namespace tensorrt_llm::executor::kv_cache
