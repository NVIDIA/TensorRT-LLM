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

// HALO-R relay copy, the third phase of the fused HALO-R kernel (moeRebalanceHaloR.cu). Unicast data path: every slice
// is loaded once from its owner with a bulk copy into shared memory and stored with bulk copies into the helper slot of
// every receiver (moeRebalanceRelayWork.h). Lane 0 of each issuer warp pipelines two banks of shared-memory slots: the
// loads of one bank overlap the stores of the other, and a bank is reused only after its stores have read it. Slices
// whose addresses or size are not 16-byte aligned are copied with plain stores.
//
// The relay reuses the dynamic shared memory of the scheduler role, which every CTA has finished before it starts.

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloR.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloRPlanner.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceRelayWork.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if CUDART_VERSION < 12080
#error "The HALO-R relay copy requires CUDA 12.8 or newer (bulk copies into CTA shared memory)"
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

//! Issuer warps per CTA and slots per issuer warp: two banks of one slot, so one slice loads while the other stores.
//! Each SM's pull throughput is capped, so past about nine loads in flight per SM more loads only add latency.
//! Measured on pull-only plans, 9 issuers with two slots each copy about 30% more per CTA than 6 issuers with four
//! slots, and 13 issuers with two slots copy less.
constexpr int kRelayMaxIssuers = 9;
constexpr int kRelaySlotsPerIssuer = 2;
//! Slices per bank batch; it sizes the per-lane batch bookkeeping.
constexpr int kRelayMaxBatch = kRelaySlotsPerIssuer / 2;
constexpr std::uint64_t kRelayBulkAlignment = 16;
constexpr std::uint64_t kRelayWordAlignment = 4;
//! Back-off between polls of a slot's load barrier.
constexpr unsigned kRelayAwaitBackoffNs = 32;

//! The broadcasts this rank copies, in plan order, with item prefix sums: those whose need set it belongs to. Built
//! by thread 0 of every CTA at the start of the dynamic shared memory.
struct RelayWork
{
    int entries;
    int broadcast[halo_r::kMaxBroadcasts];
    unsigned need[halo_r::kMaxBroadcasts]; //!< The broadcast's need set N_b, the receivers it stores into.
    int prefix[halo_r::kMaxBroadcasts + 1];
    int destinations[halo_r::kMaxBroadcasts];
    int index[halo_r::kMaxBroadcasts];
    long long slices;
    unsigned long long planeBytes[halo_r::kRelayMaxPlanes];
};

constexpr unsigned kRelayWorkBytes = static_cast<unsigned>((sizeof(RelayWork) + 127) & ~std::size_t{127});

struct RelaySlotControl
{
    unsigned long long barrier;
    unsigned phase;
    unsigned pad;
};

constexpr unsigned kRelaySlotBytes = static_cast<unsigned>(halo_r::kRelaySliceBytes + sizeof(RelaySlotControl));

__device__ __forceinline__ unsigned relaySharedAddress(void const* pointer)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void relayLoad(
    unsigned char* buffer, RelaySlotControl* slot, std::uint64_t source, unsigned bytes)
{
    unsigned const barrier = relaySharedAddress(&slot->barrier);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(bytes)
                 : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     relaySharedAddress(buffer)),
                 "l"(source), "r"(bytes), "r"(barrier)
                 : "memory");
}

__device__ __forceinline__ void relayAwait(RelaySlotControl* slot)
{
    unsigned ready;
    unsigned const barrier = relaySharedAddress(&slot->barrier);
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
            __nanosleep(kRelayAwaitBackoffNs);
        }
    } while (!ready);
    slot->phase = phase ^ 1U;
}

__device__ __forceinline__ void relayStore(std::uint64_t destination, unsigned char const* buffer, unsigned bytes)
{
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(destination),
                 "r"(relaySharedAddress(buffer)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void relayScalarCopy(std::uint64_t destination, std::uint64_t source, std::uint64_t bytes)
{
    if (((destination | source | bytes) % kRelayWordAlignment) == 0)
    {
        for (std::uint64_t i = 0; i < bytes; i += kRelayWordAlignment)
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

//! Address of `offset` within helper row (home + helper slot) of `destination` for broadcast b.
__device__ __forceinline__ std::uint64_t relayDestination(
    MoeRebalanceHaloRParams const& p, RelayWork const& work, int b, int destination, int plane, std::uint64_t offset)
{
    int const home = p.experts / p.ep;
    int const row = home
        + halo_r::helperIndex(static_cast<unsigned>(halo_r::planField(p.plan, halo_r::PlanField::kHelperLowMask)[b]),
            static_cast<unsigned>(halo_r::planField(p.plan, halo_r::PlanField::kHelperHighMask)[b]), destination);
    return p.destinationTable[static_cast<std::size_t>(destination) * p.planes + plane]
        + static_cast<std::uint64_t>(row) * work.planeBytes[plane] + offset;
}

//! Thread 0: the broadcasts this rank pulls a share of, from the plan, which every CTA holds once its scheduler role is
//! done, and the residency table, which no CTA of this launch writes before every CTA has read it.
__device__ void prepareRelayWork(MoeRebalanceHaloRParams const& p, RelayWork* work)
{
    for (int plane = 0; plane < p.planes; ++plane)
    {
        work->planeBytes[plane] = p.planeBytes[plane];
    }
    work->slices = halo_r::relayBundleSlices(work->planeBytes, p.planes);
    work->prefix[0] = 0;
    int const broadcasts = p.plan[1];
    int const* experts = halo_r::planField(p.plan, halo_r::PlanField::kExpert);
    int const* masks = halo_r::planField(p.plan, halo_r::PlanField::kDestinationMask);
    int const* lowMasks = halo_r::planField(p.plan, halo_r::PlanField::kHelperLowMask);
    int const* highMasks = halo_r::planField(p.plan, halo_r::PlanField::kHelperHighMask);
    int entries = 0;
    for (int b = 0; b < broadcasts && b < halo_r::kMaxBroadcasts; ++b)
    {
        unsigned const need = halo_r::relayNeedMask(static_cast<unsigned>(masks[b]), static_cast<unsigned>(lowMasks[b]),
            static_cast<unsigned>(highMasks[b]), experts[b], p.residency, p.helpers);
        int destinations = 0;
        int const index = halo_r::relayDestinationIndex(need, p.localRank, &destinations);
        if (index < 0)
        {
            continue;
        }
        work->broadcast[entries] = b;
        work->need[entries] = need;
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
    if (totalSlots >= kRelaySlotsPerIssuer)
    {
        int const issuers = totalSlots / kRelaySlotsPerIssuer;
        int const limit = warps < kRelayMaxIssuers ? warps : kRelayMaxIssuers;
        return issuers < limit ? issuers : limit;
    }
    return totalSlots ? 1 : 0;
}

//! Lane 0 of one issuer warp: copy every work item dealt to this warp.
__device__ void relayIssue(MoeRebalanceHaloRParams const& p, RelayWork const& work, unsigned char* buffer,
    RelaySlotControl* control, int slots, int issuers, int warp)
{
    for (int s = 0; s < slots; ++s)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(relaySharedAddress(&control[s].barrier))
                     : "memory");
        control[s].phase = 0;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    int const* experts = halo_r::planField(p.plan, halo_r::PlanField::kExpert);
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
            unsigned const mask = work.need[entry];
            std::uint64_t const source
                = p.sourceTable[static_cast<std::size_t>(experts[b]) * p.planes + plane] + offset;
            bool aligned = ((source | bytes) % kRelayBulkAlignment) == 0;
            for (int d = 0; d < p.ep && aligned; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    aligned = (relayDestination(p, work, b, d, plane, offset) % kRelayBulkAlignment) == 0;
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
            relayLoad(buffer + static_cast<std::size_t>(s) * halo_r::kRelaySliceBytes, &control[s], source,
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
            unsigned const mask = work.need[entryOf[c]];
            for (int d = 0; d < p.ep; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    relayStore(relayDestination(p, work, b, d, planeOf[c], offsetOf[c]),
                        buffer + static_cast<std::size_t>(s) * halo_r::kRelaySliceBytes, bytesOf[c]);
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

//! This CTA's share of the relay copy. Every thread of the CTA calls it; returns false, uniformly, when the dynamic
//! shared memory holds no issuer slot, which the launch preparation rules out.
__device__ bool relayCopy(MoeRebalanceHaloRParams const& p)
{
    // The work table and every slot live in dynamic shared memory, so the footprint is the launch's size.
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
    if (issuers == 0)
    {
        return false;
    }
    unsigned const perWarpBytes = ((dynamicBytes - kRelayWorkBytes) / static_cast<unsigned>(issuers)) & ~127U;
    int const fitting = static_cast<int>(perWarpBytes / kRelaySlotBytes);
    int const slots = warp < issuers ? (fitting < kRelaySlotsPerIssuer ? fitting : kRelaySlotsPerIssuer) : 0;
    unsigned char* buffer = relayShared + kRelayWorkBytes + static_cast<std::size_t>(warp) * perWarpBytes;
    auto* control
        = reinterpret_cast<RelaySlotControl*>(buffer + static_cast<std::size_t>(slots) * halo_r::kRelaySliceBytes);
    // The other warps only join the barrier.
    if (lane == 0 && slots > 0)
    {
        relayIssue(p, work, buffer, control, slots, issuers, warp);
    }
    __syncthreads();
    return true;
}

} // namespace

} // namespace kernels

TRTLLM_NAMESPACE_END
