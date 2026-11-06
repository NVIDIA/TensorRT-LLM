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
#include <utility>

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

} // namespace

size_t hostMemSizeGranularity() noexcept
{
    // A page is the only hard bound; 2 MiB is the floor because that is the huge page the
    // backing asks for, and a size quantized below it would straddle one for no gain.
    static size_t const unit = std::max(size_t{2} << 20, commitAlignment());
    return unit;
}

namespace
{

//! Bytes one step populates. Large enough that the call overhead disappears against the
//! faulting, small enough that a caller publishing progress between steps releases an
//! early waiter early.
constexpr size_t kCommitChunkSize = 32ULL << 20;

//! Bytes one registration covers, before the register unit rounds it down. Near the commit
//! chunk so the two advance together, and small enough that the tail a shrink has to split
//! is cheap to put back.
constexpr size_t kRegisterChunkTarget = 32ULL << 20;

//! The pages [addr, addr + size) touches, widened to whole pages.
//!
//! For calls that must cover at least the range: populating or opening a few
//! bytes either side is harmless, and a unit may start or end mid-page because
//! units are sized in slots rather than pages.
std::pair<MemAddress, size_t> pagesCovering(MemAddress addr, size_t size)
{
    size_t const page = commitAlignment();
    MemAddress const begin = addr & ~static_cast<MemAddress>(page - 1);
    MemAddress const end = roundUp(addr + size, page);
    return {begin, static_cast<size_t>(end - begin)};
}

//! The pages wholly inside [addr, addr + size).
//!
//! For calls that must not reach past the range: a page shared with the
//! neighbouring unit is left alone, because discarding or protecting it would
//! take bytes that still belong to that unit. At most one page per boundary
//! stays resident, which is the price of not corrupting the neighbour.
std::pair<MemAddress, size_t> pagesWithin(MemAddress addr, size_t size)
{
    size_t const page = commitAlignment();
    MemAddress const begin = roundUp(addr, page);
    MemAddress const end = (addr + size) & ~static_cast<MemAddress>(page - 1);
    return {begin, end > begin ? static_cast<size_t>(end - begin) : 0};
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

} // namespace

HostMemBacking::HostMemBacking(size_t vmSize, HostMemBackingOptions const& options)
    : mUnit(hostMemSizeGranularity())
    , mRegUnit(std::max<size_t>(options.registerUnit, 1))
    // Whole register units, so no bound falls inside one.
    , mRegChunk(std::max(kRegisterChunkTarget / mRegUnit * mRegUnit, mRegUnit))
    , mAllowRemoteNumaFallback(options.allowRemoteNumaFallback)
{
    if (vmSize == 0)
    {
        return;
    }
    // The caller rounds: a reservation silently made larger is one the caller does not
    // know it can resize into.
    TLLM_CHECK_WITH_INFO(
        vmSize % mUnit == 0, "HostMemBacking reservation %zu is not a multiple of the granularity %zu", vmSize, mUnit);

    cuCheck(cuCtxGetCurrent(reinterpret_cast<CUcontext*>(&mContext)));
    TLLM_CHECK_WITH_INFO(mContext != nullptr, "HostMemBacking needs a current CUDA context");

    // No alignment is asked of the mapping: registration needs none, and the
    // madvise and mprotect calls round to pages themselves. The mapped length is
    // page-rounded so the guard below covers whole pages.
    mMapped = roundUp(vmSize, commitAlignment());
    mAddr = hostMmap(mMapped);
    mSize = vmSize;

    //! Applied once over the whole mapping. Per-chunk page-mode advice would
    //! change VMA flags and split the mapping into one VMA per chunk.
    hostMadvisePageMode(mAddr, mSize, hostUseThp());
    bindRangeToGpuNumaNode(mAddr, mSize, mAllowRemoteNumaFallback);

    // Nothing is reachable until a step opens it, so a read that runs ahead of
    // the fill faults rather than returning zeros from an untouched page. Costs
    // one VMA and is independent of the size: with no resident pages there are
    // no page tables to rewrite.
    auto const [guardAddr, guardSize] = pagesCovering(mAddr, mSize);
    if (::mprotect(reinterpret_cast<void*>(guardAddr), guardSize, PROT_NONE) != 0)
    {
        hostMunmap(mAddr, mMapped);
        mAddr = 0;
        mSize = 0;
        mMapped = 0;
        throw HostOOMError(std::string("mprotect(PROT_NONE) failed: ") + std::strerror(errno));
    }
}

HostMemBacking::~HostMemBacking()
{
    HostMemBacking::destroy();
}

void HostMemBacking::setTargetSize(size_t size)
{
    TLLM_CHECK_WITH_INFO(mAddr != 0, "HostMemBacking::setTargetSize on a backing that reserved nothing");
    TLLM_CHECK_WITH_INFO(size <= mSize, "HostMemBacking target %zu exceeds the reservation of %zu", size, mSize);
    TLLM_CHECK_WITH_INFO(
        size % mUnit == 0, "HostMemBacking target %zu is not a multiple of the granularity %zu", size, mUnit);
    TLLM_CHECK_WITH_INFO(mCommitted == mTarget, "HostMemBacking: a new target with %zu of %zu bytes still to commit",
        mTarget - mCommitted, mTarget);
    mTarget = size;
}

size_t HostMemBacking::resizeStep(bool populated)
{
    if (mCommitted == mTarget)
    {
        return mRegisteredEnd;
    }
    // cuMemHostRegister needs a current context; the fill worker has only set
    // the device, which does not give it one.
    cuCheck(cuCtxSetCurrent(static_cast<CUcontext>(mContext)));

    if (mCommitted < mTarget)
    {
        // Rounding up from one past the end lands on the next chunk boundary
        // whether or not the previous step ended on one, so a step that resumed
        // a partial chunk completes it rather than carrying the offset forward.
        commitUpTo(std::min(roundUp(mCommitted + 1, kCommitChunkSize), mTarget), populated);
    }
    else
    {
        decommitFrom(std::max(mTarget, (mCommitted - 1) / kCommitChunkSize * kCommitChunkSize));
    }
    return mRegisteredEnd;
}

size_t HostMemBacking::commitChunkSize() noexcept
{
    return kCommitChunkSize;
}

size_t HostMemBacking::nextStepEnd() const noexcept
{
    return mCommitted < mTarget ? std::min(roundUp(mCommitted + 1, kCommitChunkSize), mTarget) : mCommitted;
}

void HostMemBacking::resizeTo(size_t size)
{
    setTargetSize(size);
    while (mCommitted != mTarget)
    {
        resizeStep();
    }
}

void HostMemBacking::commitUpTo(size_t end, bool populated)
{
    // With `populated` the caller has already opened and populated this range;
    // the step only has to register it.
    if (!populated)
    {
        open(mCommitted, end - mCommitted);
        populate(mCommitted, end - mCommitted);
    }

    size_t const begin = mCommitted;
    mCommitted = end;
    try
    {
        registerUpTo(end, /*allowPartialChunk=*/end == mTarget);
    }
    catch (...)
    {
        // A failed step must take nothing, so what it populated goes back.
        decommitFrom(begin);
        throw;
    }
}

void HostMemBacking::open(size_t offset, size_t size)
{
    TLLM_CHECK_DEBUG(mAddr != 0 && offset + size <= mSize);
    auto const [openAddr, openSize] = pagesCovering(mAddr + offset, size);
    if (::mprotect(reinterpret_cast<void*>(openAddr), openSize, PROT_READ | PROT_WRITE) != 0)
    {
        throw HostOOMError(std::string("mprotect(PROT_READ|PROT_WRITE) failed: ") + std::strerror(errno));
    }
}

void HostMemBacking::populate(size_t offset, size_t size) const
{
    TLLM_CHECK_DEBUG(mAddr != 0 && offset + size <= mSize);
    // Allocating and zeroing through the kernel's fault path is the point of
    // this backing: it scales across processes, where the driver serializes the
    // same work on a per-process lock.
    //
    // Each call holds the process's mmap_lock for reading throughout, and
    // whatever needs it for writing -- mprotect and registration by other
    // fills, allocations on the executor -- waits for the call to finish.
    // Short calls bound that wait.
    constexpr size_t kPopulateCall = 4ULL << 20;
    auto const [fillAddr, fillSize] = pagesCovering(mAddr + offset, size);
    for (size_t done = 0; done < fillSize; done += kPopulateCall)
    {
        hostPrefaultChunk(fillAddr + done, std::min(kPopulateCall, fillSize - done));
    }
}

void HostMemBacking::registerUpTo(size_t end, bool allowPartialChunk)
{
    size_t const target = end / mRegUnit * mRegUnit;
    while (mRegisteredEnd < target)
    {
        // Rounding up from one past the start stops at the next chunk boundary, so
        // no registration ever spans one.
        size_t const chunkEnd = std::min(roundUp(mRegisteredEnd + 1, mRegChunk), target);
        if (!allowPartialChunk && chunkEnd % mRegChunk != 0)
        {
            return;
        }
        CUresult const registered = cuMemHostRegister(reinterpret_cast<void*>(mAddr + mRegisteredEnd),
            chunkEnd - mRegisteredEnd, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
        if (registered == CUDA_ERROR_OUT_OF_MEMORY)
        {
            throw HostOOMError("cuMemHostRegister failed: out of memory");
        }
        cuCheck(registered);
        recordRegistrationStart(mRegisteredEnd);
        mRegisteredEnd = chunkEnd;
    }
}

void HostMemBacking::recordRegistrationStart(size_t start)
{
    // A start on a chunk boundary is the one innermostRegistrationStart() derives,
    // so remembering it would only duplicate the arithmetic.
    if (start % mRegChunk != 0)
    {
        mUnalignedRegStarts.push(start);
    }
}

size_t HostMemBacking::innermostRegistrationStart() const noexcept
{
    TLLM_CHECK_DEBUG(mRegisteredEnd > 0);
    // A registration never reaches past the chunk boundary above where it began,
    // so the boundary below its last byte is where it began -- unless it began
    // off a boundary, which is the case the stack exists to record.
    size_t const onChunkBoundary = (mRegisteredEnd - 1) / mRegChunk * mRegChunk;
    return mUnalignedRegStarts.empty() ? onChunkBoundary : std::max(mUnalignedRegStarts.top(), onChunkBoundary);
}

size_t HostMemBacking::dropInnermostRegistration() noexcept
{
    size_t const start = innermostRegistrationStart();
    unregisterAt(start);
    if (!mUnalignedRegStarts.empty() && mUnalignedRegStarts.top() == start)
    {
        mUnalignedRegStarts.pop();
    }
    // Registrations are contiguous, so the one below ends where this began.
    mRegisteredEnd = start;
    return start;
}

void HostMemBacking::unregisterAt(size_t offset) noexcept
{
    CUresult const result = cuMemHostUnregister(reinterpret_cast<void*>(mAddr + offset));
    if (result != CUDA_SUCCESS)
    {
        char const* name = nullptr;
        cuGetErrorName(result, &name);
        // The driver and this object disagree about what is registered, which the
        // pages are about to be freed on the assumption of.
        Poison::set("HostMemBacking::unregisterAt", name != nullptr ? name : "cuMemHostUnregister failed");
    }
}

void HostMemBacking::unregisterFrom(size_t end) noexcept
{
    size_t const target = end / mRegUnit * mRegUnit;
    while (mRegisteredEnd > target)
    {
        size_t const start = dropInnermostRegistration();
        if (start >= target)
        {
            continue;
        }
        // That one straddled `target`. A registration cannot be shortened, so it
        // had to go whole and its surviving prefix is registered again. Leaving
        // it would pin pages madvise is about to discard.
        CUresult const result = cuMemHostRegister(reinterpret_cast<void*>(mAddr + start), target - start,
            CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
        if (result != CUDA_SUCCESS)
        {
            char const* name = nullptr;
            cuGetErrorName(result, &name);
            // The prefix stays committed but unregistered, so a copy over it would
            // fail rather than read the wrong pages. Nothing can repair that here.
            Poison::set("HostMemBacking::unregisterFrom", name != nullptr ? name : "cuMemHostRegister failed");
            return;
        }
        recordRegistrationStart(start);
        mRegisteredEnd = target;
        return;
    }
}

void HostMemBacking::decommitFrom(size_t end) noexcept
{
    if (mAddr == 0 || end >= mCommitted)
    {
        return;
    }
    // Order matters and nothing enforces it: registration does not hold the
    // pages, so discarding them first destroys memory the GPU still has
    // registered, with every call reporting success.
    unregisterFrom(end);
    auto const [dropAddr, dropSize] = pagesWithin(mAddr + end, mCommitted - end);
    mCommitted = end;
    if (dropSize == 0)
    {
        return;
    }
    //! MADV_DONTNEED drops the pages without changing VMA flags, so unlike
    //! mprotect it does not split the mapping.
    if (::madvise(reinterpret_cast<void*>(dropAddr), dropSize, MADV_DONTNEED) != 0)
    {
        TLLM_LOG_WARNING("KVCM2 host tier: madvise(MADV_DONTNEED) failed with errno %d", errno);
    }
    // Back to unreachable, so a reader that outlives the shrink faults instead
    // of silently re-faulting a zero page.
    if (::mprotect(reinterpret_cast<void*>(dropAddr), dropSize, PROT_NONE) != 0)
    {
        TLLM_LOG_WARNING("KVCM2 host tier: mprotect(PROT_NONE) failed with errno %d", errno);
    }
}

void HostMemBacking::discardUncommitted(size_t offset, size_t size) noexcept
{
    if (mAddr == 0 || size == 0)
    {
        return;
    }
    // Never registered -- registration happens in the commit -- so unlike
    // decommitFrom() this only has pages to drop.
    TLLM_CHECK_DEBUG(offset >= mCommitted);
    auto const [dropAddr, dropSize] = pagesWithin(mAddr + offset, size);
    if (dropSize == 0)
    {
        return;
    }
    if (::madvise(reinterpret_cast<void*>(dropAddr), dropSize, MADV_DONTNEED) != 0)
    {
        TLLM_LOG_WARNING("KVCM2 host tier: madvise(MADV_DONTNEED) failed with errno %d", errno);
    }
    // Back to unreachable, undoing the open() that let it be populated.
    if (::mprotect(reinterpret_cast<void*>(dropAddr), dropSize, PROT_NONE) != 0)
    {
        TLLM_LOG_WARNING("KVCM2 host tier: mprotect(PROT_NONE) failed with errno %d", errno);
    }
}

void HostMemBacking::destroy() noexcept
{
    if (mAddr == 0)
    {
        return;
    }
    // Skipped once KVCM2 is poisoned: teardown then also skips the waits that
    // let this range's CUDA consumers finish, and unmapping under one corrupts
    // whatever is mapped there next. The range is kept instead, which costs a
    // tier in a process whose cache is already permanently unusable.
    KVCM2_POISON_ON_EXCEPT(
        [&]
        {
            unregisterFrom(0);
            hostMunmap(mAddr, mMapped);
        });
    mUnalignedRegStarts = {};
    mRegisteredEnd = 0;
    mAddr = 0;
    mSize = 0;
    mMapped = 0;
    mTarget = 0;
    mCommitted = 0;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
