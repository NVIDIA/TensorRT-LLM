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
// every receiver (moeRebalanceRelayWork.h). Lane 0 of each issuer warp runs a ring of shared-memory slots: its loads
// run ahead of its stores, and a slot is reloaded as soon as the stores of its previous slice have read it. Slices
// whose addresses or size are not 16-byte aligned are copied with plain stores.
//
// The relay reuses the dynamic shared memory of the scheduler role, which every CTA has finished before it starts.

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/cudaAsyncOps.cuh"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloR.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloRPlanner.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceRelayWork.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if CUDART_VERSION < 13010
#error "The HALO-R relay copy requires CUDA 13.1 or newer, like the HALO-R kernel"
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

//! Issuer warps per CTA and ring slots per issuer: each issuer keeps up to two loads in flight. An issuer completes
//! about one slice per round trip whatever the slice size, so the SM's rate grows with the slice (kRelaySliceBytes).
constexpr int kRelayMaxIssuers = 2;
constexpr int kRelaySlotsPerIssuer = 3;
constexpr std::uint64_t kRelayBulkAlignment = 16;
constexpr std::uint64_t kRelayWordAlignment = 4;
//! Back-off between polls of a slot's load barrier.
constexpr unsigned kRelayAwaitBackoffNs = 32;

//! The broadcasts this rank copies, in plan order, with item prefix sums: those whose need set it belongs to. Built
//! by thread 0 of every CTA at the start of the dynamic shared memory.
struct RelayWork
{
    int entries;
    int broadcast[halo_r::kMaxHaloRBroadcasts];
    unsigned need[halo_r::kMaxHaloRBroadcasts]; //!< The broadcast's need set N_b, the receivers it stores into.
    int prefix[halo_r::kMaxHaloRBroadcasts + 1];
    int destinations[halo_r::kMaxHaloRBroadcasts];
    int index[halo_r::kMaxHaloRBroadcasts];
    long long slices;
    unsigned long long planeBytes[halo_r::kRelayMaxPlanes];
};

constexpr unsigned kRelayWorkBytes = static_cast<unsigned>((sizeof(RelayWork) + 127) & ~std::size_t{127});

struct RelaySlotControl
{
    std::uint64_t barrier;
    unsigned phase;
    unsigned pad;
};

constexpr unsigned kRelaySlotBytes = static_cast<unsigned>(halo_r::kRelaySliceBytes + sizeof(RelaySlotControl));

__device__ __forceinline__ void relayLoad(
    unsigned char* buffer, RelaySlotControl* slot, std::uint64_t source, unsigned bytes)
{
    mbarrier_arrive_expect_tx_no_state(&slot->barrier, bytes);
    cp_async_bulk_g2s(buffer, reinterpret_cast<void const*>(source), static_cast<int>(bytes), &slot->barrier);
}

__device__ __forceinline__ void relayAwait(RelaySlotControl* slot)
{
    // A volatile wait: cudaAsyncOps.cuh's mbarrier_try_wait_parity is not volatile, so the compiler could hoist it
    // out of this spin loop.
    unsigned const barrier = __as_ptr_smem(&slot->barrier);
    unsigned const phase = slot->phase;
    unsigned ready = 0;
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
    cp_async_bulk_s2g(reinterpret_cast<void*>(destination), buffer, static_cast<int>(bytes));
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
        + halo_r::helperIndex(
            static_cast<unsigned>(halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperLowMask)[b]),
            static_cast<unsigned>(halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperHighMask)[b]), destination);
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
    int const* experts = halo_r::planField(p.plan, halo_r::HaloRPlanField::kExpert);
    int const* masks = halo_r::planField(p.plan, halo_r::HaloRPlanField::kDestinationMask);
    int const* lowMasks = halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperLowMask);
    int const* highMasks = halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperHighMask);
    int entries = 0;
    for (int b = 0; b < broadcasts && b < halo_r::kMaxHaloRBroadcasts; ++b)
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

//! Issuer warps for a dynamic shared-memory size: one per kRelaySlotsPerIssuer slots, at most kRelayMaxIssuers. The
//! launch preparation gives the relay the work table and at least one issuer's slots, so there is at least one issuer,
//! and rounding its share down to 128 bytes leaves it at least two slots.
__device__ __forceinline__ int relayIssuers(unsigned dynamicBytes, int warps)
{
    int const issuers = static_cast<int>((dynamicBytes - kRelayWorkBytes) / kRelaySlotBytes) / kRelaySlotsPerIssuer;
    int const limit = warps < kRelayMaxIssuers ? warps : kRelayMaxIssuers;
    return issuers < limit ? issuers : limit;
}

static_assert(kRelaySlotBytes >= 128 && kRelaySlotsPerIssuer >= 3,
    "rounding an issuer's share of kRelaySlotsPerIssuer slots down to 128 bytes leaves at least two slots");

//! Waits until every store of this issuer has completed: READY covers the destination writes, not only their
//! shared-memory reads.
__device__ __forceinline__ void relayIssuerDone()
{
    cp_async_bulk_wait_group<0>();
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

//! Lane 0 of one issuer warp, with at least two slots: copy every work item dealt to this warp. Item n uses slot
//! n mod `slots`. Loads run up to `slots - 1` items ahead of the stores: once item n's stores are issued, the slot of
//! item n - 1 is refilled as soon as its stores have read it.
__device__ void relayIssue(MoeRebalanceHaloRParams const& p, RelayWork const& work, unsigned char* buffer,
    RelaySlotControl* control, int slots, int issuers, int warp)
{
    for (int s = 0; s < slots; ++s)
    {
        mbarrier_init(&control[s].barrier, 1);
        control[s].phase = 0;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    int const* experts = halo_r::planField(p.plan, halo_r::HaloRPlanField::kExpert);
    long long const total = work.prefix[work.entries];
    long long const stride = static_cast<long long>(gridDim.x) * issuers;
    long long const first = static_cast<long long>(blockIdx.x) * issuers + warp;
    long long const items = total > first ? (total - first + stride - 1) / stride : 0;
    int entryOf[kRelaySlotsPerIssuer];
    std::uint64_t sourceOf[kRelaySlotsPerIssuer];
    std::uint64_t offsetOf[kRelaySlotsPerIssuer];
    unsigned bytesOf[kRelaySlotsPerIssuer];
    int planeOf[kRelaySlotsPerIssuer];
    bool bulkOf[kRelaySlotsPerIssuer];
    // Item n into slot s: a bulk load, or nothing yet for a slice the bulk copy cannot take (copied in its turn).
    auto load = [&](long long n, int s)
    {
        long long k;
        int const entry = halo_r::relayLocateItem(work.prefix, work.entries, first + n * stride, &k);
        long long const slice = halo_r::relaySliceOf(k, work.destinations[entry], work.index[entry]);
        int plane = 0;
        unsigned long long offset = 0;
        unsigned long long bytes = 0;
        halo_r::relayLocateSlice(work.planeBytes, p.planes, slice, &plane, &offset, &bytes);
        int const b = work.broadcast[entry];
        unsigned const mask = work.need[entry];
        std::uint64_t const source = p.sourceTable[static_cast<std::size_t>(experts[b]) * p.planes + plane] + offset;
        bool aligned = ((source | bytes) % kRelayBulkAlignment) == 0;
        for (int d = 0; d < p.ep && aligned; ++d)
        {
            if ((mask >> d) & 1U)
            {
                aligned = (relayDestination(p, work, b, d, plane, offset) % kRelayBulkAlignment) == 0;
            }
        }
        entryOf[s] = entry;
        sourceOf[s] = source;
        offsetOf[s] = offset;
        bytesOf[s] = static_cast<unsigned>(bytes);
        planeOf[s] = plane;
        bulkOf[s] = aligned;
        if (aligned)
        {
            relayLoad(buffer + static_cast<std::size_t>(s) * halo_r::kRelaySliceBytes, &control[s], source,
                static_cast<unsigned>(bytes));
        }
    };
    for (int s = 0; s < slots && s < items; ++s)
    {
        load(s, s);
    }
    for (long long n = 0; n < items; ++n)
    {
        int const s = static_cast<int>(n % slots);
        int const b = work.broadcast[entryOf[s]];
        unsigned const mask = work.need[entryOf[s]];
        if (bulkOf[s])
        {
            relayAwait(&control[s]);
            for (int d = 0; d < p.ep; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    relayStore(relayDestination(p, work, b, d, planeOf[s], offsetOf[s]),
                        buffer + static_cast<std::size_t>(s) * halo_r::kRelaySliceBytes, bytesOf[s]);
                }
            }
            cp_async_bulk_commit_group();
        }
        else
        {
            for (int d = 0; d < p.ep; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    relayScalarCopy(relayDestination(p, work, b, d, planeOf[s], offsetOf[s]), sourceOf[s], bytesOf[s]);
                }
            }
        }
        if (n >= 1 && n - 1 + slots < items)
        {
            // Every group but this item's (when it has one) has read its slot, so item n - 1's slot is free.
            if (bulkOf[s])
            {
                cp_async_bulk_wait_group_read<1>();
            }
            else
            {
                cp_async_bulk_wait_group_read<0>();
            }
            load(n - 1 + slots, static_cast<int>((n - 1) % slots));
        }
    }
    relayIssuerDone();
}

//! This CTA's share of the relay copy. Every thread of the CTA calls it.
__device__ void relayCopy(MoeRebalanceHaloRParams const& p)
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
}

} // namespace

} // namespace kernels

TRTLLM_NAMESPACE_END
