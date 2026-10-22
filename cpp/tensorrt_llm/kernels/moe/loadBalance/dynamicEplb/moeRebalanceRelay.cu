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

// HALO-R relay copy. Unicast data path: every slice is loaded once from its owner with a bulk copy into shared memory
// and stored with bulk copies into the helper slot of every receiver (moeRebalanceRelayWork.h). Lane 0 of each issuer
// warp pipelines two banks of shared-memory slots: the loads of one bank overlap the stores of the other, and a bank
// is reused only after its stores have read it. Slices whose addresses or size are not 16-byte aligned are copied
// with plain stores. READY is published once every store of every CTA has completed: a release/acquire chain over the
// CTAs elects the last one, which stores the epoch through the multicast view of the READY rows.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloR.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloRPlanner.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceRelay.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceRelayWork.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

#if CUDART_VERSION < 12080
#error "The HALO-R relay copy requires CUDA 12.8 or newer (bulk copies into CTA shared memory)"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error "The HALO-R relay copy requires SM90 or newer"
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using halo_r::kRelayMaxPlanes;
using halo_r::kRelaySliceBytes;

//! The relay uses the CTA shape of the TMA copy, seven worker pairs of 64 threads, so that both copies occupy the
//! same SM resources next to the MoE kernel.
constexpr int kRelayThreads = 7 * MEGAMOE_TMA_COPY_THREADS_PER_WORKER;
//! An issuer warp needs two banks of at least two slots to overlap loads with stores.
constexpr int kRelayMinSlotsPerIssuer = 4;
//! Slices per bank batch; it sizes the per-lane batch bookkeeping.
constexpr int kRelayMaxBatch = 16;
//! The MegaMoE READY gate reserves bit 63 of a READY cell for errors.
constexpr std::uint64_t kReadyErrorBit = 1ULL << 63;
constexpr std::uint64_t kBulkAlignment = 16;
constexpr std::uint64_t kWordAlignment = 4;
//! Back-off between polls of a slot's load barrier.
constexpr unsigned kAwaitBackoffNs = 32;

//! This rank's received broadcasts in plan order with item prefix sums. Built by thread 0 of every CTA at the start
//! of the dynamic shared memory.
struct RelayWork
{
    int entries;
    int broadcast[halo_r::kMaxBroadcasts];
    int prefix[halo_r::kMaxBroadcasts + 1];
    int destinations[halo_r::kMaxBroadcasts];
    int index[halo_r::kMaxBroadcasts];
    long long slices;
    unsigned long long planeBytes[kRelayMaxPlanes];
};

constexpr unsigned kRelayWorkBytes = static_cast<unsigned>((sizeof(RelayWork) + 127) & ~std::size_t{127});

struct RelaySlotControl
{
    unsigned long long barrier;
    unsigned phase;
    unsigned pad;
};

constexpr unsigned kRelaySlotBytes = static_cast<unsigned>(kRelaySliceBytes + sizeof(RelaySlotControl));

__device__ __forceinline__ unsigned sharedAddress(void const* pointer)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void relayLoad(
    unsigned char* buffer, RelaySlotControl* slot, std::uint64_t source, unsigned bytes)
{
    unsigned const barrier = sharedAddress(&slot->barrier);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(bytes)
                 : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     sharedAddress(buffer)),
                 "l"(source), "r"(bytes), "r"(barrier)
                 : "memory");
}

__device__ __forceinline__ void relayAwait(RelaySlotControl* slot)
{
    unsigned ready;
    unsigned const barrier = sharedAddress(&slot->barrier);
    unsigned const phase = slot->phase;
    do
    {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
            "selp.u32 %0, 1, 0, p; }"
            : "=r"(ready)
            : "r"(barrier), "r"(phase)
            : "memory");
        if (!ready)
        {
            __nanosleep(kAwaitBackoffNs);
        }
    } while (!ready);
    slot->phase = phase ^ 1U;
}

__device__ __forceinline__ void relayStore(std::uint64_t destination, unsigned char const* buffer, unsigned bytes)
{
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(destination),
                 "r"(sharedAddress(buffer)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void relayScalarCopy(std::uint64_t destination, std::uint64_t source, std::uint64_t bytes)
{
    if (((destination | source | bytes) % kWordAlignment) == 0)
    {
        for (std::uint64_t i = 0; i < bytes; i += kWordAlignment)
        {
            *reinterpret_cast<unsigned*>(destination + i) = *reinterpret_cast<unsigned const*>(source + i);
        }
    }
    else
    {
        for (std::uint64_t i = 0; i < bytes; ++i)
        {
            *reinterpret_cast<unsigned char*>(destination + i) = *reinterpret_cast<unsigned char const*>(source + i);
        }
    }
}

__device__ __forceinline__ void multicastStoreReady(std::uint64_t* flag, std::uint64_t value)
{
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    asm volatile("multimem.st.release.sys.global.u64 [%0], %1;" ::"l"(flag), "l"(value) : "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
}

__device__ __forceinline__ bool validPlaneCount(int planes)
{
    return planes >= 1 && planes <= kRelayMaxPlanes;
}

//! Address of `offset` within helper row (home + helper slot) of `destination` for broadcast b.
__device__ __forceinline__ std::uint64_t relayDestination(
    MoeRebalanceRelayParams const& p, RelayWork const& work, int b, int destination, int plane, std::uint64_t offset)
{
    int const home = p.experts / p.ep;
    int const row = home
        + halo_r::helperIndex(static_cast<unsigned>(halo_r::planField(p.plan, halo_r::PlanField::kHelperLowMask)[b]),
            static_cast<unsigned>(halo_r::planField(p.plan, halo_r::PlanField::kHelperHighMask)[b]), destination);
    return p.destinationTable[static_cast<std::size_t>(destination) * p.planes + plane]
        + static_cast<std::uint64_t>(row) * work.planeBytes[plane] + offset;
}

void __device__ prepareRelayWork(MoeRebalanceRelayParams const& p, RelayWork* work)
{
    work->entries = 0;
    work->prefix[0] = 0;
    work->slices = 0;
    if (!validPlaneCount(p.planes))
    {
        return;
    }
    for (int plane = 0; plane < p.planes; ++plane)
    {
        work->planeBytes[plane] = p.planeBytes[plane];
    }
    work->slices = halo_r::relayBundleSlices(work->planeBytes, p.planes);
    int const broadcasts = p.plan[1];
    int const* masks = halo_r::planField(p.plan, halo_r::PlanField::kDestinationMask);
    int entries = 0;
    for (int b = 0; b < broadcasts && b < halo_r::kMaxBroadcasts; ++b)
    {
        int destinations = 0;
        int const index = halo_r::relayDestinationIndex(static_cast<unsigned>(masks[b]), p.localRank, &destinations);
        if (index < 0)
        {
            continue;
        }
        work->broadcast[entries] = b;
        work->destinations[entries] = destinations;
        work->index[entries] = index;
        work->prefix[entries + 1]
            = work->prefix[entries] + static_cast<int>(halo_r::relaySliceCount(work->slices, destinations, index));
        ++entries;
    }
    work->entries = entries;
}

//! Issuer warps and slots per issuer warp for a dynamic shared-memory size.
__device__ __forceinline__ int relayIssuers(unsigned dynamicBytes, int warps)
{
    int const totalSlots
        = dynamicBytes > kRelayWorkBytes ? static_cast<int>((dynamicBytes - kRelayWorkBytes) / kRelaySlotBytes) : 0;
    if (totalSlots >= kRelayMinSlotsPerIssuer)
    {
        int const issuers = totalSlots / kRelayMinSlotsPerIssuer;
        return issuers < warps ? issuers : warps;
    }
    return totalSlots ? 1 : 0;
}

//! Lane 0 of one issuer warp: copy every work item dealt to this warp.
__device__ void relayIssue(MoeRebalanceRelayParams const& p, RelayWork const& work, unsigned char* buffer,
    RelaySlotControl* control, int slots, int issuers, int warp)
{
    for (int s = 0; s < slots; ++s)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sharedAddress(&control[s].barrier)) : "memory");
        control[s].phase = 0;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    int const* experts = halo_r::planField(p.plan, halo_r::PlanField::kExpert);
    int const* masks = halo_r::planField(p.plan, halo_r::PlanField::kDestinationMask);
    int const banks = slots >= 2 ? 2 : 1;
    int const bankSlots = slots / banks;
    int const batchLimit = bankSlots < kRelayMaxBatch ? bankSlots : kRelayMaxBatch;
    long long const total = work.prefix[work.entries];
    long long const stride = static_cast<long long>(gridDim.x) * issuers;
    long long item = static_cast<long long>(blockIdx.x) * issuers + warp;
    bool pending[2] = {false, false};
    for (int bank = 0; item < total; bank = (bank + 1) % banks)
    {
        if (pending[bank])
        {
            // The bank's shared-memory slots may be refilled once its stores have read them.
            if (banks == 2 && pending[bank ^ 1])
            {
                asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
            }
            else
            {
                asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
            pending[bank] = false;
        }
        int entryOf[kRelayMaxBatch];
        std::uint64_t offsetOf[kRelayMaxBatch];
        unsigned bytesOf[kRelayMaxBatch];
        int planeOf[kRelayMaxBatch];
        int count = 0;
        for (; count < batchLimit && item < total; item += stride)
        {
            long long k;
            int const entry = halo_r::relayLocateItem(work.prefix, work.entries, item, &k);
            long long const slice = halo_r::relaySliceOf(k, work.destinations[entry], work.index[entry]);
            int plane = 0;
            unsigned long long offset = 0;
            unsigned long long bytes = 0;
            halo_r::relayLocateSlice(work.planeBytes, p.planes, slice, &plane, &offset, &bytes);
            int const b = work.broadcast[entry];
            unsigned const mask = static_cast<unsigned>(masks[b]);
            std::uint64_t const source
                = p.sourceTable[static_cast<std::size_t>(experts[b]) * p.planes + plane] + offset;
            bool aligned = ((source | bytes) % kBulkAlignment) == 0;
            for (int d = 0; d < p.ep && aligned; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    aligned = (relayDestination(p, work, b, d, plane, offset) % kBulkAlignment) == 0;
                }
            }
            if (!aligned)
            {
                for (int d = 0; d < p.ep; ++d)
                {
                    if ((mask >> d) & 1U)
                    {
                        relayScalarCopy(relayDestination(p, work, b, d, plane, offset), source, bytes);
                    }
                }
                continue;
            }
            int const s = bank * bankSlots + count;
            relayLoad(buffer + static_cast<std::size_t>(s) * kRelaySliceBytes, &control[s], source,
                static_cast<unsigned>(bytes));
            entryOf[count] = entry;
            offsetOf[count] = offset;
            bytesOf[count] = static_cast<unsigned>(bytes);
            planeOf[count] = plane;
            ++count;
        }
        for (int c = 0; c < count; ++c)
        {
            int const s = bank * bankSlots + c;
            relayAwait(&control[s]);
            int const b = work.broadcast[entryOf[c]];
            unsigned const mask = static_cast<unsigned>(masks[b]);
            for (int d = 0; d < p.ep; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    relayStore(relayDestination(p, work, b, d, planeOf[c], offsetOf[c]),
                        buffer + static_cast<std::size_t>(s) * kRelaySliceBytes, bytesOf[c]);
                }
            }
        }
        if (count)
        {
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            pending[bank] = true;
        }
    }
    // READY covers the completion of the destination writes, not only their shared-memory reads.
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

//! The value the last CTA publishes: the plan epoch, with the READY error bit when this launch could not copy, any
//! scheduler status word records a failure (timeouts, planner, invalid or unmappable routes, READY), or the epoch
//! differs from this copy's own launch count.
__device__ std::uint64_t relayReadyValue(MoeRebalanceRelayParams const& p, bool copyFailed)
{
    int const epoch = *reinterpret_cast<int const volatile*>(p.plan);
    unsigned const launches = atomicAdd(p.launchCount, 1U) + 1U;
    int const volatile* status = p.status;
    bool const schedulerFailed = status[kHaloRStatusExchangeTimeout] != 0 || status[kHaloRStatusGridTimeout] != 0
        || status[kHaloRStatusPlanWaitTimeout] != 0 || status[kHaloRStatusInvalidRoute] != 0
        || status[kHaloRStatusSlotOutsideMask] != 0 || status[kHaloRStatusPlanner] != 0
        || status[kHaloRStatusReady] != 0;
    bool const error = copyFailed || schedulerFailed || epoch != static_cast<int>(launches);
    std::uint64_t const value = static_cast<std::uint64_t>(static_cast<unsigned>(epoch));
    return error ? (value | kReadyErrorBit) : value;
}

__global__ void haloRRelayKernel(MoeRebalanceRelayParams const p)
{
    // The work table and every slot live in dynamic shared memory, so the footprint is the requested size.
    extern __shared__ __align__(128) unsigned char relayShared[];
    RelayWork& work = *reinterpret_cast<RelayWork*>(relayShared);
    if (threadIdx.x == 0)
    {
        prepareRelayWork(p, &work);
    }
    __syncthreads();

    int const warps = blockDim.x >> 5;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    unsigned dynamicBytes;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamicBytes));
    int const issuers = relayIssuers(dynamicBytes, warps);
    unsigned const perWarpBytes
        = issuers ? ((dynamicBytes - kRelayWorkBytes) / static_cast<unsigned>(issuers)) & ~127U : 0U;
    int const slots = warp < issuers ? static_cast<int>(perWarpBytes / kRelaySlotBytes) : 0;
    unsigned char* buffer = relayShared + kRelayWorkBytes + static_cast<std::size_t>(warp) * perWarpBytes;
    auto* control = reinterpret_cast<RelaySlotControl*>(buffer + static_cast<std::size_t>(slots) * kRelaySliceBytes);
    if (lane == 0 && slots > 0)
    {
        relayIssue(p, work, buffer, control, slots, issuers, warp);
    }
    __syncthreads();
    if (threadIdx.x != 0)
    {
        return;
    }
    unsigned previous;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(previous) : "l"(p.completedCtas) : "memory");
    if (previous == gridDim.x - 1)
    {
        // Every CTA has arrived; the next launch is stream-ordered after this one, so the counter can be rearmed.
        asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(p.completedCtas) : "memory");
        bool const copyFailed = !validPlaneCount(p.planes) || issuers == 0;
        multicastStoreReady(p.readyFlagMc, relayReadyValue(p, copyFailed));
    }
}

int relayDynamicSharedMemoryBytes(int device)
{
    static std::mutex mutex;
    static std::map<int, int> bytesByDevice;
    std::lock_guard<std::mutex> lock(mutex);
    auto const found = bytesByDevice.find(device);
    if (found != bytesByDevice.end())
    {
        return found->second;
    }
    int optinBytes = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&optinBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attributes{};
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attributes, haloRRelayKernel));
    int const available = optinBytes > static_cast<int>(attributes.sharedSizeBytes)
        ? optinBytes - static_cast<int>(attributes.sharedSizeBytes)
        : 0;
    // The same dynamic shared memory as the TMA copy.
    std::uint64_t const slotBytes = MEGAMOE_TMA_COPY_SLOT_DATA_BYTES + MEGAMOE_TMA_COPY_SLOT_CONTROL_BYTES;
    std::uint64_t const raw = static_cast<std::uint64_t>(megamoe_tma_copy_total_slots(available)) * slotBytes;
    int const bytes = static_cast<int>((raw + 127) & ~std::uint64_t{127});
    TLLM_CHECK_WITH_INFO(bytes >= static_cast<int>(kRelayWorkBytes + kRelayMinSlotsPerIssuer * kRelaySlotBytes),
        "HALO-R relay needs at least %u bytes of dynamic shared memory, the device provides %d",
        kRelayWorkBytes + kRelayMinSlotsPerIssuer * kRelaySlotBytes, bytes);
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(haloRRelayKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    bytesByDevice[device] = bytes;
    return bytes;
}

} // namespace

std::size_t getMoeRebalanceRelaySharedMemoryBytes()
{
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    return static_cast<std::size_t>(relayDynamicSharedMemoryBytes(device));
}

int getMoeRebalanceRelayThreads()
{
    return kRelayThreads;
}

void invokeMoeRebalanceRelay(MoeRebalanceRelayParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.plan != nullptr && params.sourceTable != nullptr && params.destinationTable != nullptr
            && params.planeBytes != nullptr && params.status != nullptr && params.completedCtas != nullptr
            && params.launchCount != nullptr && params.readyFlagMc != nullptr,
        "HALO-R relay buffers must not be null");
    TLLM_CHECK_WITH_INFO(params.ep >= 2 && params.ep <= halo_r::kMaxRanks && params.experts > 0
            && params.experts <= halo_r::kPlanFieldStride && params.experts % params.ep == 0 && params.localRank >= 0
            && params.localRank < params.ep,
        "Invalid HALO-R relay geometry: ep=%d experts=%d local_rank=%d", params.ep, params.experts, params.localRank);
    TLLM_CHECK_WITH_INFO(params.planes >= 1 && params.planes <= kRelayMaxPlanes,
        "HALO-R relay supports 1 to %d weight planes, got %d", kRelayMaxPlanes, params.planes);
    TLLM_CHECK_WITH_INFO(
        reinterpret_cast<std::uintptr_t>(params.readyFlagMc) % sizeof(std::uint64_t) == 0, "READY cell is misaligned");

    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int smCount = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    TLLM_CHECK_WITH_INFO(params.ctas >= 1 && params.ctas <= smCount,
        "HALO-R relay CTAs must be between 1 and the device SM count (%d), got %d", smCount, params.ctas);
    int const sharedBytes = relayDynamicSharedMemoryBytes(device);

    haloRRelayKernel<<<params.ctas, kRelayThreads, sharedBytes, stream>>>(params);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
