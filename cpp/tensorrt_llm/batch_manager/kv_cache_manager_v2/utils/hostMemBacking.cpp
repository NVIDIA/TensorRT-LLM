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

#include "kv_cache_manager_v2/utils/hostMemBacking.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/hostMem.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/topologyDetector.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cuda.h>
#include <map>
#include <numaif.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

//! Minimum granularity madvise accepts. The fill worker commits in much larger
//! chunks; this is only the alignment floor.
//!
//! Aligning further, to the huge page size, is deliberately not done: populating
//! a sub-range of a MADV_HUGEPAGE mapping keeps the pages huge and does not
//! split the VMA, so a chunk edge inside a huge page costs nothing.
size_t commitAlignment()
{
    return static_cast<size_t>(::sysconf(_SC_PAGESIZE));
}

//! Places the pages of [addr, addr + size) on the NUMA node the current GPU
//! attaches to.
//!
//! The policy is attached to the address range, not to a thread, so it holds
//! whichever thread happens to fault a page in. Setting it on the whole range
//! before anything is committed also avoids splitting the mapping, which a
//! per-chunk policy would do.
//!
//! Best effort, with a working fallback rather than no placement at all. Where
//! mbind cannot be applied -- no libnuma, a single node, a GPU that reports no
//! affinity, or a container whose seccomp profile rejects the call -- the range
//! keeps MPOL_DEFAULT, which faults each page onto the node of the CPU that
//! first touches it. The fill worker binds itself to the GPU's node before
//! committing, and it is the thread that first touches every page, so the
//! common path still places correctly without mbind.
//!
//! What mbind adds is independence from *which* thread faults a page: a reader
//! that races past the watermark, or a re-fault after decommit, need not be the
//! bound worker. That is why the policy is attached to the range even though
//! first touch would usually suffice.
//!
//! The fallback is not equivalent for allowRemoteFallback == false. First touch
//! prefers the local node and silently takes a remote page when that node
//! cannot satisfy it, which is MPOL_PREFERRED's behaviour. Only MPOL_BIND
//! refuses, and nothing unprivileged reproduces that -- set_mempolicy is gated
//! with mbind. So where mbind is denied, a strict request degrades to a
//! preference, which is why that case is reported differently below.
//!
//! Note for containers: mbind, set_mempolicy and get_mempolicy sit behind
//! CAP_SYS_NICE in Docker's default seccomp profile, so an unprivileged
//! container falls back to first touch.
void bindRangeToGpuNumaNode(MemAddress addr, size_t size, bool allowRemoteFallback)
{
    if (numa_available() < 0 || numa_max_node() < 1)
    {
        return;
    }
    int const node = runtime::TopologyDetector::getInstance().getCurrentGpuNumaId();
    if (node < 0)
    {
        TLLM_LOG_WARNING("KVCM2 host tier: GPU reports no NUMA node; leaving page placement to first touch");

        return;
    }

    std::unique_ptr<bitmask, decltype(&numa_free_nodemask)> mask{numa_allocate_nodemask(), &numa_free_nodemask};
    if (mask == nullptr)
    {
        return;
    }
    numa_bitmask_setbit(mask.get(), static_cast<unsigned>(node));

    // MPOL_BIND fails an allocation the node cannot satisfy; MPOL_PREFERRED
    // takes a remote page instead, trading bandwidth for capacity.
    int const mode = allowRemoteFallback ? MPOL_PREFERRED : MPOL_BIND;
    if (::mbind(reinterpret_cast<void*>(addr), size, mode, mask->maskp, mask->size + 1, 0) != 0)
    {
        if (!allowRemoteFallback)
        {
            // First touch prefers the local node but takes a remote page rather
            // than failing, so it cannot stand in for a strict binding. Refusing
            // here is better than running as though the request was honoured.
            throw LogicError(
                "KVCM2 host tier: allow_remote_numa_fallback=false requires mbind, which failed with errno "
                + std::to_string(errno) + " for NUMA node " + std::to_string(node)
                + ". In a container this usually means CAP_SYS_NICE is absent; grant it, or set "
                  "allow_remote_numa_fallback=true to accept first-touch placement.");
        }
        TLLM_LOG_WARNING(
            "KVCM2 host tier: mbind to NUMA node %d failed with errno %d; falling back to first-touch placement by "
            "the fill worker, which is bound to that node (in a container this usually means CAP_SYS_NICE is "
            "absent). Placement is preserved; no behaviour is lost here.",
            node, errno);
        return;
    }
    TLLM_LOG_INFO("KVCM2 host tier: pages bound to NUMA node %d (%s)", node,
        allowRemoteFallback ? "remote fallback allowed" : "strict");
}

// ---------------------------------------------------------------------------
// mmap backing
// ---------------------------------------------------------------------------
class MmapHostMemBacking final : public IHostMemBacking
{
public:
    explicit MmapHostMemBacking(HostMemBackingOptions const& options)
        : mAlignment(std::max(options.commitUnit, commitAlignment()))
        , mAllowRemoteNumaFallback(options.allowRemoteNumaFallback)
    {
    }

    ~MmapHostMemBacking() override
    {
        MmapHostMemBacking::destroy();
    }

    MemAddress reserve(size_t size) override
    {
        TLLM_CHECK_WITH_INFO(mAddr == 0, "HostMem backing reserved twice");
        mAddr = hostMmap(size);
        mSize = size;
        //! Applied once over the whole mapping. Per-chunk page-mode advice would
        //! change VMA flags and split the mapping into one VMA per chunk.
        hostMadvisePageMode(mAddr, mSize, hostUseThp());
        bindRangeToGpuNumaNode(mAddr, mSize, mAllowRemoteNumaFallback);
        return mAddr;
    }

    void commit(size_t offset, size_t size) override
    {
        TLLM_CHECK_DEBUG(mAddr != 0 && offset + size <= mSize);
        hostPrefaultChunk(mAddr + offset, size);
    }

    void decommit(size_t offset, size_t size) noexcept override
    {
        if (mAddr == 0)
        {
            return;
        }
        TLLM_CHECK_DEBUG(offset + size <= mSize);
        //! MADV_DONTNEED drops the pages without changing VMA flags, so unlike
        //! mprotect it does not split the mapping.
        ::madvise(reinterpret_cast<void*>(mAddr + offset), size, MADV_DONTNEED);
    }

    void destroy() noexcept override
    {
        if (mAddr != 0)
        {
            hostMunmap(mAddr, mSize);
            mAddr = 0;
            mSize = 0;
        }
    }

    [[nodiscard]] size_t commitGranularity() const noexcept override
    {
        return mAlignment;
    }

    [[nodiscard]] char const* name() const noexcept override
    {
        return "mmap";
    }

private:
    MemAddress mAddr = 0;
    size_t mSize = 0;
    size_t const mAlignment;
    bool const mAllowRemoteNumaFallback;
};

// ---------------------------------------------------------------------------
// CUDA VMM backing
// ---------------------------------------------------------------------------
class VmmHostMemBacking final : public IHostMemBacking
{
public:
    explicit VmmHostMemBacking(HostMemBackingOptions const& options)
        : mAllowRemoteNumaFallback(options.allowRemoteNumaFallback)
    {
        CUdevice device{};
        cuCheck(cuCtxGetDevice(&device));
        mDevice = device;

        int numaId = 0;
        cuCheck(cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
        //! -1 means the platform reports no NUMA affinity for this device.
        mNumaId = numaId >= 0 ? numaId : 0;

        mProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        mProp.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        mProp.location.id = mNumaId;
        size_t required = 0;
        cuCheck(cuMemGetAllocationGranularity(&required, &mProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        // The driver's granularity is a floor; the caller's unit sets how finely
        // a resize can release memory, traded against per-allocation cost.
        mGranularity = std::max(roundUp(options.commitUnit, required), required);

        mAccess[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        mAccess[0].location.id = mNumaId;
        mAccess[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        mAccess[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        mAccess[1].location.id = static_cast<int>(mDevice);
        mAccess[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    ~VmmHostMemBacking() override
    {
        VmmHostMemBacking::destroy();
    }

    MemAddress reserve(size_t size) override
    {
        TLLM_CHECK_WITH_INFO(mAddr == 0, "HostMem backing reserved twice");
        CUdeviceptr addr = 0;
        cuCheck(cuMemAddressReserve(&addr, size, mGranularity, 0, 0));
        mAddr = static_cast<MemAddress>(addr);
        mSize = size;
        return mAddr;
    }

    //! Split into granularity-sized allocations so that a later decommit can
    //! release any granularity-aligned sub-range without having to split one.
    void commit(size_t offset, size_t size) override
    {
        TLLM_CHECK_DEBUG(mAddr != 0 && offset + size <= mSize);
        TLLM_CHECK_DEBUG(offset % mGranularity == 0 && size % mGranularity == 0);

        size_t committed = 0;
        try
        {
            for (; committed < size; committed += mGranularity)
            {
                commitOne(offset + committed);
            }
            // Inside the try: a failure here would otherwise leave every granule
            // above mapped until destroy(), and the caller treats a failed
            // commit as having taken nothing.
            //
            // Access applies to a range, so it costs the same whether the range
            // was one allocation or many.
            cuCheck(cuMemSetAccess(static_cast<CUdeviceptr>(mAddr + offset), size, mAccess.data(), mAccess.size()));
        }
        catch (...)
        {
            decommit(offset, committed);
            throw;
        }
    }

    //! Exact, because every allocation is one granule and the bounds are
    //! granule-aligned: no allocation ever straddles the released range.
    void decommit(size_t offset, size_t size) noexcept override
    {
        TLLM_CHECK_DEBUG(offset % mGranularity == 0 && size % mGranularity == 0);
        auto chunk = mChunks.lower_bound(offset);
        while (chunk != mChunks.end() && chunk->first < offset + size)
        {
            cuMemUnmap(static_cast<CUdeviceptr>(mAddr + chunk->first), chunk->second.size);
            cuMemRelease(chunk->second.handle);
            chunk = mChunks.erase(chunk);
        }
    }

    void destroy() noexcept override
    {
        if (mAddr == 0)
        {
            return;
        }
        for (auto const& [offset, chunk] : mChunks)
        {
            cuMemUnmap(static_cast<CUdeviceptr>(mAddr + offset), chunk.size);
            cuMemRelease(chunk.handle);
        }
        mChunks.clear();
        cuMemAddressFree(static_cast<CUdeviceptr>(mAddr), mSize);
        mAddr = 0;
        mSize = 0;
    }

    [[nodiscard]] size_t commitGranularity() const noexcept override
    {
        return mGranularity;
    }

    [[nodiscard]] char const* name() const noexcept override
    {
        return "vmm";
    }

private:
    void commitOne(size_t offset)
    {
        CUmemGenericAllocationHandle handle{};
        CUresult created = cuMemCreate(&handle, mGranularity, &mProp, 0);
        if (created == CUDA_ERROR_OUT_OF_MEMORY && mAllowRemoteNumaFallback)
        {
            // A named node is strict in the driver: os_alloc_pages_node() passes
            // __GFP_THISNODE and additionally frees and fails any page the
            // kernel returns from a different node, so cuMemCreate reports OOM
            // rather than spilling. Honouring the flag therefore means retrying
            // without naming a node and letting any of them serve it.
            CUmemAllocationProp anyNode = mProp;
            anyNode.location.type = CU_MEM_LOCATION_TYPE_HOST;
            anyNode.location.id = 0;
            created = cuMemCreate(&handle, mGranularity, &anyNode, 0);
            if (created == CUDA_SUCCESS)
            {
                TLLM_LOG_WARNING(
                    "KVCM2 host tier: NUMA node %d is exhausted; allocating from another node, which the GPU reaches "
                    "more slowly. Set allow_remote_numa_fallback=false to fail instead.",
                    mNumaId);
            }
        }
        if (created == CUDA_ERROR_OUT_OF_MEMORY)
        {
            throw HostOOMError("cuMemCreate(HOST_NUMA) failed: out of memory");
        }
        cuCheck(created);

        auto const base = static_cast<CUdeviceptr>(mAddr + offset);
        try
        {
            cuCheck(cuMemMap(base, mGranularity, 0, handle, 0));
        }
        catch (...)
        {
            cuMemRelease(handle);
            throw;
        }
        mChunks.emplace(offset, Chunk{mGranularity, handle});
    }

    struct Chunk
    {
        size_t size;
        CUmemGenericAllocationHandle handle;
    };

    MemAddress mAddr = 0;
    size_t mSize = 0;
    size_t mGranularity = 0;
    CUdevice mDevice = 0;
    int mNumaId = 0;
    bool mAllowRemoteNumaFallback = true;
    CUmemAllocationProp mProp{};
    std::array<CUmemAccessDesc, 2> mAccess{};
    std::map<size_t, Chunk> mChunks;
};

} // namespace

HostMemBackingKind selectHostMemBackingKind()
{
    static HostMemBackingKind const kind = []
    {
        //! The answer is cached, so a query made before a context exists would
        //! fix the wrong backing for the process lifetime. Fail loudly instead
        //! of guessing; every caller reaches here from HostMem construction,
        //! which already requires a context.
        CUdevice device{};
        cuCheck(cuCtxGetDevice(&device));
        int usesHostPageTables = 0;
        int coherent = 0;
        cuDeviceGetAttribute(
            &usesHostPageTables, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device);
        cuDeviceGetAttribute(&coherent, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device);
        auto const selected
            = (usesHostPageTables != 0 && coherent != 0) ? HostMemBackingKind::kMmap : HostMemBackingKind::kVmm;
        // Which route the host tier takes decides its bandwidth, so record it
        // alongside the attributes that chose it: a host-tier performance report
        // is not interpretable without knowing which one ran.
        TLLM_LOG_INFO("KVCM2 host memory backing: %s (host page tables: %d, coherent link: %d)",
            selected == HostMemBackingKind::kMmap ? "mmap" : "vmm", usesHostPageTables, coherent);
        return selected;
    }();
    return kind;
}

std::unique_ptr<IHostMemBacking> createHostMemBacking(HostMemBackingKind kind, HostMemBackingOptions const& options)
{
    switch (kind)
    {
    case HostMemBackingKind::kMmap: return std::make_unique<MmapHostMemBacking>(options);
    case HostMemBackingKind::kVmm: return std::make_unique<VmmHostMemBacking>(options);
    }
    TLLM_THROW("Unknown HostMemBackingKind");
}

std::unique_ptr<IHostMemBacking> createHostMemBacking(HostMemBackingOptions const& options)
{
    return createHostMemBacking(selectHostMemBackingKind(), options);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
