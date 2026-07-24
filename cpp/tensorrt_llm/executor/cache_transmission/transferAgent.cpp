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

#include <cuda.h>
#include <dlfcn.h>
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
            size_t offsetInChunk = static_cast<size_t>((addr - regionBase) % chunkSize);
            size_t pieceSize = std::min(remaining, chunkSize - offsetInChunk);
            result.emplace_back(addr, pieceSize, deviceId);
            addr += pieceSize;
            remaining -= pieceSize;
        }
    }

    return MemoryDescs{descs.getType(), std::move(result)};
}

std::pair<MemoryDescs, MemoryDescs> VmmDescSplitter::splitTransferDescsWithRegionMaps(MemoryDescs const& srcDescs,
    MemoryDescs const& dstDescs, VramRegionMap const& localRegionMap, VramRegionMap const& remoteRegionMap)
{
    if (srcDescs.getType() != MemoryType::kVRAM)
        return {srcDescs, dstDescs};

    auto const& srcVec = srcDescs.getDescs();
    auto const& dstVec = dstDescs.getDescs();
    TLLM_CHECK(srcVec.size() == dstVec.size());

    std::vector<MemoryDesc> splitSrc, splitDst;
    splitSrc.reserve(srcVec.size());
    splitDst.reserve(dstVec.size());

    for (size_t i = 0; i < srcVec.size(); ++i)
    {
        auto [srcChunkSize, srcBase] = lookupChunkInfo(srcVec[i].getAddr(), localRegionMap);
        auto [dstChunkSize, dstBase] = lookupChunkInfo(dstVec[i].getAddr(), remoteRegionMap);

        // If neither side is multi-chunk VMM, no splitting is needed.
        if (srcChunkSize == 0 && dstChunkSize == 0)
        {
            splitSrc.push_back(srcVec[i]);
            splitDst.push_back(dstVec[i]);
            continue;
        }

        uintptr_t srcAddr = srcVec[i].getAddr();
        uintptr_t dstAddr = dstVec[i].getAddr();
        size_t remaining = srcVec[i].getLen();

        while (remaining > 0)
        {
            size_t srcPieceSize = remaining;
            if (srcChunkSize > 0)
            {
                size_t srcOffsetInChunk = static_cast<size_t>((srcAddr - srcBase) % srcChunkSize);
                srcPieceSize = srcChunkSize - srcOffsetInChunk;
            }

            size_t dstPieceSize = remaining;
            if (dstChunkSize > 0)
            {
                size_t dstOffsetInChunk = static_cast<size_t>((dstAddr - dstBase) % dstChunkSize);
                dstPieceSize = dstChunkSize - dstOffsetInChunk;
            }

            size_t pieceSize = std::min({remaining, srcPieceSize, dstPieceSize});
            splitSrc.emplace_back(srcAddr, pieceSize, srcVec[i].getDeviceId());
            splitDst.emplace_back(dstAddr, pieceSize, dstVec[i].getDeviceId());
            srcAddr += pieceSize;
            dstAddr += pieceSize;
            remaining -= pieceSize;
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
