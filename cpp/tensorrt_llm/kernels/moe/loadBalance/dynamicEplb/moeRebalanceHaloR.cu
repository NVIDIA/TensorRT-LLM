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

// Fused HALO-R kernel: the HALO-R scheduler and the relay copy of its plan in one grid of `auxiliary_sms` CTAs, one
// CTA per SM, on the copy stream (moeRebalanceHaloR.h). It follows the fused HALO-Q + TMA kernel
// (moeRebalanceTma.cu) and reuses its scheduler passes (moeRebalanceHaloQKernel.cuh), with four differences:
//   - The plan is HALO-R's own (moeRebalanceHaloRPlanner.h): arbitrary receiver masks, each receiver with its own
//     helper slot, copied with the unicast relay (moeRebalanceRelay.cuh) instead of multicast. With per-layer helper
//     slots, the relay skips each receiver whose slot already holds the expert, not a whole team.
//   - A CTA starts its relay share right after releasing its routes, without waiting for the other CTAs: CTA 0
//     publishes the plan only once it is complete, and the grid rendezvous already proved every CTA resident.
//   - A failing CTA traps before it releases its routes: every timeout and any nonzero status word, which also
//     catches the invalid routes and unmappable slots that the route passes record and continue past. The route
//     consumer therefore never passes its wait for a failed launch, and the last CTA, which re-checks the status words
//     before it publishes READY, publishes nothing.
//   - READY and the route counter take constant values (moeRebalanceHaloR.h), so a captured launch replays exactly.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceCompletion.cuh"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceFusedLaunch.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQKernel.cuh"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloR.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloRPlanner.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceRelay.cuh"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// CMakeLists.txt builds this file for the SM100 family with CUDA 13.1 or newer only.
#if CUDART_VERSION < 13010
#error "The HALO-R kernel requires CUDA 13.1 or newer"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "The HALO-R kernel requires SM100 or newer"
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

//! The shared passes partition routes over 16 warps per CTA, and one CTA must fit an SM for the grid rendezvous.
constexpr int kFusedThreads = 512;
constexpr int kFusedWarps = kFusedThreads / 32;
constexpr unsigned kFullWarpMask = 0xffffffffU;

static_assert(halo_r::kPlanFieldStride == kMaxBroadcasts, "HALO-R uses the shared plan field stride");
static_assert(static_cast<int>(halo_r::HaloRPlanField::kExpert) == kPlanExpert,
    "the shared worker passes read the selected experts from plan field 0");
static_assert(halo_r::kMaxRanks == kMaxEp, "the planner covers every supported EP size");

struct DeviceWarp
{
    static __device__ __forceinline__ int lane()
    {
        return static_cast<int>(threadIdx.x & 31U);
    }

    static __device__ __forceinline__ void sync()
    {
        __syncwarp();
    }

    static __device__ __forceinline__ unsigned reduceMax(unsigned value)
    {
        return __reduce_max_sync(kFullWarpMask, value);
    }

    static __device__ __forceinline__ int reduceAdd(int value)
    {
        return __reduce_add_sync(kFullWarpMask, value);
    }

    static __device__ __forceinline__ unsigned ballot(bool predicate)
    {
        return __ballot_sync(kFullWarpMask, predicate);
    }

    static __device__ __forceinline__ unsigned findFirstSet(unsigned value)
    {
        return static_cast<unsigned>(__ffs(static_cast<int>(value)));
    }

    static __device__ __forceinline__ unsigned popCount(unsigned value)
    {
        return static_cast<unsigned>(__popc(value));
    }
};

//! Why a CTA's scheduler role failed, uniform across the CTA; `reason` is null on success.
struct RoleResult
{
    char const* reason;
    int detail;
};

//! Thread 0 reports why this launch fails and traps, which faults the context. The calling CTA has not released its
//! routes and publishes no READY.
__device__ __noinline__ void trapLaunch(MoeRebalanceHaloRParams const& p, char const* reason, int detail)
{
    if (threadIdx.x == 0)
    {
        printf("HALO-R rank %d CTA %d: %s (%d); no routes or READY are published\n", p.localRank,
            static_cast<int>(blockIdx.x), reason, detail);
        asm volatile("trap;" ::: "memory");
    }
}

//! Index of the first nonzero status word, or -1. Thread 0, after a block barrier that orders this CTA's own records.
__device__ int firstFailedStatus(int const* status)
{
    for (int word = 0; word < kHaloRStatusWords; ++word)
    {
        if (*reinterpret_cast<int const volatile*>(status + word) != 0)
        {
            return word;
        }
    }
    return -1;
}

//! HALO-R's slot of a selected route (halo_r::selectedRouteSlot): the owner's home slot, or the receiver's own helper
//! slot.
struct HaloRSelectedRouteSlot
{
    int const* owners;
    int const* masks;
    int const* lowMasks;
    int const* highMasks;
    int home;
    int localSlots;

    __device__ __forceinline__ bool operator()(int b, int expert, int destination, int& slot) const
    {
        slot = halo_r::selectedRouteSlot(expert, destination, owners[b], static_cast<unsigned>(masks[b]),
            static_cast<unsigned>(lowMasks[b]), static_cast<unsigned>(highMasks[b]), home, localSlots);
        return slot >= 0;
    }
};

//! Slot of every selected route of this worker's span, through HALO-Q's selected-route pass with HALO-R's slots.
__device__ void materializeSelectedRoutes(MoeRebalanceHaloRParams const& p, int worker, int workers, Epoch epoch)
{
    int const home = p.experts / p.ep;
    HaloRSelectedRouteSlot const slotOf{halo_r::planField(p.plan, halo_r::HaloRPlanField::kOwner),
        halo_r::planField(p.plan, halo_r::HaloRPlanField::kDestinationMask),
        halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperLowMask),
        halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperHighMask), home, home + p.helpers};
    materializeCompactedSelectedRoutes(p.routes, p.outSlots, p.routePrefix, p.routeAux,
        p.status + kHaloRStatusSlotOutsideMask, p.validRouteCount, p.ep, p.experts, worker, workers, epoch, slotOf);
}

//! CTA 0 thread 0, before the count exchange of every launch: waits until every rank's previous launch of this layer
//! has published into this rank's READY row, then zeroes the row. The row starts at 1, as if a launch had preceded
//! the first, so no launch skips the drain, whatever its epoch. Every launch publishes the constant 1, and a peer
//! publishes this epoch's READY only after this epoch's exchange, which this rank enters only after the reset, so this
//! epoch's gate cannot pass on the previous epoch's cells. Returns the first rank whose cell did not arrive within the
//! bound, or -1.
__device__ int drainReadyRow(std::uint64_t* row, int ep, long long start, unsigned long long spinCycles)
{
    for (int source = 0; source < ep; ++source)
    {
        std::uint64_t value;
        for (;;)
        {
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(row + source) : "memory");
            if (value != 0)
            {
                break;
            }
            if (static_cast<unsigned long long>(clock64() - start) > spinCycles)
            {
                return source;
            }
        }
    }
    // Acquire the values published through the multicast alias before resetting through the unicast alias.
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    for (int source = 0; source < ep; ++source)
    {
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(row + source), "l"(0ULL) : "memory");
    }
    // Order the reset before this rank's exchange arrival; a peer's next READY follows its acquire of that arrival.
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    return -1;
}

//! CTA 0: READY drain, count exchange, plan, plan publication and route prefix. The drain and the exchange share one
//! spin bound.
__device__ RoleResult planAndPublish(MoeRebalanceHaloRParams const& p, Epoch epoch)
{
    extern __shared__ int shared[];
    int* counts = shared;
    int* histogram = counts + p.ep * p.experts;
    int* quota = histogram + p.experts;
    auto* scratch = reinterpret_cast<halo_r::PlannerScratch*>(quota + halo_r::kMaxHaloRBroadcasts * p.ep);
    __shared__ int missingReady;
    __shared__ unsigned long long exchangeCycles;

    if (threadIdx.x == 0)
    {
        long long const start = clock64();
        missingReady = drainReadyRow(p.readyFlags, p.ep, start, p.spinCycles);
        auto const elapsed = static_cast<unsigned long long>(clock64() - start);
        exchangeCycles = elapsed < p.spinCycles ? p.spinCycles - elapsed : 1;
    }
    __syncthreads();
    if (missingReady >= 0)
    {
        return RoleResult{"READY drain timeout", missingReady};
    }
    if (!exchange_counts(reinterpret_cast<unsigned long long const*>(p.peerBases), p.partial, shared, p.ep, p.experts,
            p.localRank, static_cast<int>(gridDim.x), epoch, exchangeCycles, p.status))
    {
        return RoleResult{"count exchange timeout", static_cast<int>(epoch)};
    }
    for (int expert = threadIdx.x; expert < p.experts; expert += blockDim.x)
    {
        int total = 0;
        for (int rank = 0; rank < p.ep; ++rank)
        {
            total += counts[rank * p.experts + expert];
        }
        histogram[expert] = total;
    }
    __syncthreads();

    // The residency table, which the previous launch of the layer updated before its READY, keeps a resident helper
    // expert in its slot.
    __shared__ int resident[halo_r::kMaxRanks * halo_r::kMaxHelperSlots];
    if (p.residency != nullptr)
    {
        for (int entry = threadIdx.x; entry < p.ep * p.helpers; entry += blockDim.x)
        {
            resident[entry] = p.residency[entry];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
    {
        halo_r::PlannerConfig const cfg{p.ep, p.experts, p.helpers, p.sendCount, p.localRank};
        halo_r::PlannerOutputs const outputs{p.plan, quota, p.outIds, p.outOwners, nullptr, nullptr};
        // checkMoeRebalanceHaloRParams accepted the configuration, so the planner always plans.
        halo_r::planHaloR<DeviceWarp>(histogram, cfg, scratch, outputs, p.residency != nullptr ? resident : nullptr);
    }
    __syncthreads();
    int const broadcasts = p.plan[1];
    if (threadIdx.x == 0)
    {
        store_release_device(p.plan, epoch);
    }
    __syncthreads();
    build_route_prefix(counts, quota, p.routePrefix, p.plan, broadcasts, p.ep, p.experts, p.localRank, blockDim.x);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        store_release_device(p.gridSync + 2, epoch);
    }
    return RoleResult{nullptr, 0};
}

//! A worker CTA: stable ordinals, the home slot of every unselected route and, once the prefix is published, the slot
//! of every selected route. Each wait has one spin bound.
__device__ RoleResult mapRoutes(MoeRebalanceHaloRParams const& p, Epoch epoch)
{
    int const worker = static_cast<int>(blockIdx.x) - 1;
    int const workers = static_cast<int>(gridDim.x) - 1;
    if (!stable_worker_ordinal_pass(
            p.routes, p.outSlots, p.routeAux, p.status, p.validRouteCount, p.experts, worker, workers, p.spinCycles))
    {
        return RoleResult{"worker rendezvous timeout", static_cast<int>(epoch)};
    }
    int* expertLut = worker_expert_lut(p.experts);
    prepare_worker_default_slots(expertLut, p.ep, p.experts, p.helpers);
    __shared__ int planReady;
    if (!wait_for_plan_epoch(p.plan, epoch, p.spinCycles, p.status, &planReady))
    {
        return RoleResult{"plan wait timeout", static_cast<int>(epoch)};
    }
    prepare_worker_selected_experts(expertLut, p.plan);
    pre_materialize_and_compact_worker_routes(
        p.routes, p.outSlots, p.routeAux, p.status, p.validRouteCount, p.experts, worker, workers, epoch);
    if (!wait_for_plan_epoch(p.gridSync + 2, epoch, p.spinCycles, p.status, &planReady))
    {
        return RoleResult{"route prefix wait timeout", static_cast<int>(epoch)};
    }
    materializeSelectedRoutes(p, worker, workers, epoch);
    return RoleResult{nullptr, 0};
}

//! Phase 1. Every return is uniform across the CTA.
__device__ RoleResult schedulerRole(MoeRebalanceHaloRParams const& p)
{
    // Only the caller's valid prefix is readable; clear the rest of the output capacity in the same kernel.
    for (int index = p.validRouteCount + blockIdx.x * blockDim.x + threadIdx.x; index < p.routeCapacity;
         index += gridDim.x * blockDim.x)
    {
        p.outSlots[index] = -1;
    }
    local_histogram_pass(p.routes, p.partial, p.validRouteCount, p.experts, static_cast<int>(gridDim.x));
    Epoch epoch;
    if (!grid_rendezvous(p.gridSync, static_cast<int>(gridDim.x), p.spinCycles, p.status, &epoch))
    {
        return RoleResult{"grid rendezvous timeout", 0};
    }
    return blockIdx.x == 0 ? planAndPublish(p, epoch) : mapRoutes(p, epoch);
}

//! Receivers that one thread's broadcasts copied to and skipped, for the diagnostic counters.
struct ReceiverCounts
{
    unsigned long long copies;
    unsigned long long skips;
};

//! The last CTA, every thread: each broadcast's expert recorded in every receiver's helper slot, after counting the
//! receivers copied to and skipped from the residency table as the relay read it, which the update overwrites. A plan
//! assigns each (receiver, helper slot) to one broadcast, so a thread reads and writes only its own broadcasts' table
//! entries.
__device__ ReceiverCounts recordResidency(MoeRebalanceHaloRParams const& p)
{
    ReceiverCounts counts{0, 0};
    int const broadcasts = p.plan[1];
    int const* experts = halo_r::planField(p.plan, halo_r::HaloRPlanField::kExpert);
    int const* masks = halo_r::planField(p.plan, halo_r::HaloRPlanField::kDestinationMask);
    int const* lowMasks = halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperLowMask);
    int const* highMasks = halo_r::planField(p.plan, halo_r::HaloRPlanField::kHelperHighMask);
    for (int b = threadIdx.x; b < broadcasts; b += blockDim.x)
    {
        unsigned const mask = static_cast<unsigned>(masks[b]);
        unsigned const low = static_cast<unsigned>(lowMasks[b]);
        unsigned const high = static_cast<unsigned>(highMasks[b]);
        unsigned const need = halo_r::relayNeedMask(mask, low, high, experts[b], p.residency, p.helpers);
        counts.copies += static_cast<unsigned long long>(__popc(need));
        counts.skips += static_cast<unsigned long long>(__popc(mask & ~need));
        if (p.residency == nullptr)
        {
            continue;
        }
        for (int d = 0; d < p.ep; ++d)
        {
            if ((mask >> d) & 1U)
            {
                p.residency[d * p.helpers + halo_r::helperIndex(low, high, d)] = experts[b];
            }
        }
    }
    return counts;
}

//! Phase 4: a release/acquire chain over the CTAs elects the last one, which publishes READY. No CTA waits for
//! another, and a rank with nothing to copy joins the chain all the same.
//!
//! The residency table changes before READY: every read of it in this launch precedes the chain, and READY orders the
//! change before any later launch of the layer that acquires it. Every rank applies the same change, so the tables
//! stay identical. The diagnostic counters, read only by tests, are summed after READY.
__device__ void completeLaunch(MoeRebalanceHaloRParams const& p)
{
    __shared__ int last;
    __shared__ int failedWord;
    __shared__ unsigned long long receiverCopies;
    __shared__ unsigned long long receiverSkips;
    if (threadIdx.x == 0)
    {
        last = moeRebalanceArriveLastCta(p.completedCtas);
        failedWord = last ? firstFailedStatus(p.status) : -1;
        receiverCopies = 0;
        receiverSkips = 0;
    }
    __syncthreads();
    if (!last)
    {
        return;
    }
    if (failedWord >= 0)
    {
        trapLaunch(p, "status word set before READY", failedWord);
        return;
    }
    ReceiverCounts const counts = recordResidency(p);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        moeRebalanceRearmCtaCounter(p.completedCtas);
        moeRebalancePublishReady(reinterpret_cast<std::uint64_t>(p.readyFlagMc), p.readyValue);
    }
    if (counts.copies != 0 || counts.skips != 0)
    {
        atomicAdd(&receiverCopies, counts.copies);
        atomicAdd(&receiverSkips, counts.skips);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        p.counters[kHaloRCounterLaunches] += 1;
        p.counters[kHaloRCounterReceiverCopies] += static_cast<long long>(receiverCopies);
        p.counters[kHaloRCounterReceiverSkips] += static_cast<long long>(receiverSkips);
    }
}

__global__ __launch_bounds__(kFusedThreads, 1) void haloRFusedKernel(MoeRebalanceHaloRParams const p)
{
    RoleResult const role = schedulerRole(p);
    if (role.reason != nullptr)
    {
        trapLaunch(p, role.reason, role.detail);
        return;
    }

    // Phase 2. The barrier orders every thread's route writes before thread 0's system-scope release. Some failures
    // are recorded by passes that continue, so the release also waits for every status word to be clear.
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int const word = firstFailedStatus(p.status);
        if (word >= 0)
        {
            trapLaunch(p, "status word set before route release", word);
        }
        else
        {
            asm volatile("red.release.sys.global.add.u64 [%0], 1;" ::"l"(p.routeCounter) : "memory");
        }
    }

    // Phase 3. The barrier hands the scheduler role's dynamic shared memory to the relay.
    __syncthreads();
    relayCopy(p);

    completeLaunch(p);
}

//! Dynamic shared memory of the scheduler role.
std::size_t schedulerSharedBytes(int ep, int experts)
{
    std::size_t const workerInts = static_cast<std::size_t>(kFusedWarps) * (experts + 1) + experts;
    std::size_t const plannerInts = static_cast<std::size_t>(ep) * experts + experts
        + static_cast<std::size_t>(halo_r::kMaxHaloRBroadcasts) * ep + sizeof(halo_r::PlannerScratch) / sizeof(int);
    return std::max(workerInts, plannerInts) * sizeof(int);
}

} // namespace

std::int64_t getMoeRebalanceHaloRRouteAuxEntries(int experts, int ctas, int routeCapacity)
{
    std::int64_t const bins = static_cast<std::int64_t>(experts) + 1;
    std::int64_t const passEntries = static_cast<std::int64_t>(ctas) * kFusedWarps * bins;
    std::int64_t const workerEntries = 2 + static_cast<std::int64_t>(ctas - 1) * bins + routeCapacity;
    return std::max(passEntries, workerEntries);
}

void checkMoeRebalanceHaloRParams(MoeRebalanceHaloRParams const& params, int ctas)
{
    TLLM_CHECK_WITH_INFO(params.outSlots != nullptr && params.outIds != nullptr && params.outOwners != nullptr
            && params.peerBases != nullptr && params.status != nullptr && params.partial != nullptr
            && params.routeAux != nullptr && params.gridSync != nullptr && params.plan != nullptr
            && params.routePrefix != nullptr && params.routeCounter != nullptr && params.sourceTable != nullptr
            && params.destinationTable != nullptr && params.planeBytes != nullptr && params.completedCtas != nullptr
            && params.counters != nullptr && params.readyFlags != nullptr && params.readyFlagMc != nullptr,
        "HALO-R kernel buffers must not be null");
    halo_r::PlannerConfig const cfg{params.ep, params.experts, params.helpers, params.sendCount, params.localRank};
    TLLM_CHECK_WITH_INFO(params.ep >= kHaloRMinEp && halo_r::validPlannerConfig(cfg),
        "Invalid HALO-R planner configuration: ep=%d experts=%d helpers=%d send_count=%d local_rank=%d", params.ep,
        params.experts, params.helpers, params.sendCount, params.localRank);
    TLLM_CHECK_WITH_INFO(params.routeCapacity > 0 && params.routeCapacity <= INT_MAX / params.ep,
        "Invalid HALO-R route capacity %d", params.routeCapacity);
    // A candidate key stores its rows above the order byte. A step moves at most the receiver's room below the
    // balanced target, ceil(total / ep), and each rank routes at most routeCapacity rows, so rows never exceed
    // routeCapacity.
    TLLM_CHECK_WITH_INFO(static_cast<long long>(params.routeCapacity) < halo_r::kMaxCandidateRows,
        "HALO-R route capacity %d (max_num_tokens x top_k per rank) overflows the planner's candidate keys",
        params.routeCapacity);
    TLLM_CHECK_WITH_INFO(params.planes >= 1 && params.planes <= halo_r::kRelayMaxPlanes,
        "HALO-R relay supports 1 to %d weight planes, got %d", halo_r::kRelayMaxPlanes, params.planes);
    TLLM_CHECK_WITH_INFO(reinterpret_cast<std::uintptr_t>(params.readyFlags) % sizeof(std::uint64_t) == 0
            && reinterpret_cast<std::uintptr_t>(params.readyFlagMc) % sizeof(std::uint64_t) == 0
            && reinterpret_cast<std::uintptr_t>(params.routeCounter) % sizeof(std::uint64_t) == 0,
        "HALO-R READY row, READY cell and route counter must be 8-byte aligned");
    TLLM_CHECK_WITH_INFO(ctas >= kHaloRMinCtas && ctas <= kHaloRMaxCtas,
        "HALO-R kernel needs between %d and %d CTAs, got %d", kHaloRMinCtas, kHaloRMaxCtas, ctas);
}

MoeRebalanceHaloRLaunch prepareMoeRebalanceHaloRLaunch(int ep, int experts, int ctas)
{
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int smCount = 0;
    int optinBytes = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&optinBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // The grid rendezvous needs every CTA resident at once, one CTA per SM.
    TLLM_CHECK_WITH_INFO(ctas >= kHaloRMinCtas && ctas <= std::min(kHaloRMaxCtas, smCount),
        "HALO-R kernel CTAs (%d) must be between %d and the device SM count (%d)", ctas, kHaloRMinCtas, smCount);

    void const* kernel = reinterpret_cast<void const*>(&haloRFusedKernel);
    cudaFuncAttributes attributes{};
    TLLM_CUDA_CHECK(loadMoeRebalanceFusedKernel(kernel, attributes));
    TLLM_CHECK_WITH_INFO(attributes.maxThreadsPerBlock >= kFusedThreads,
        "HALO-R kernel needs %d threads per CTA, the device allows %d", kFusedThreads, attributes.maxThreadsPerBlock);

    // The relay takes the shared memory of the TMA copy's slots; the scheduler role reuses the same allocation.
    int const available = std::max(0, optinBytes - static_cast<int>(attributes.sharedSizeBytes));
    std::size_t const slotBytes = MEGAMOE_TMA_COPY_SLOT_DATA_BYTES + MEGAMOE_TMA_COPY_SLOT_CONTROL_BYTES;
    std::size_t const relayBytes
        = (static_cast<std::size_t>(megamoe_tma_copy_total_slots(available)) * slotBytes + 127) & ~std::size_t{127};
    std::size_t const minRelayBytes = kRelayWorkBytes + kRelaySlotsPerIssuer * kRelaySlotBytes;
    TLLM_CHECK_WITH_INFO(relayBytes >= minRelayBytes,
        "HALO-R relay needs at least %zu bytes of dynamic shared memory, the device provides %zu", minRelayBytes,
        relayBytes);
    std::size_t const bytes = std::max(relayBytes, schedulerSharedBytes(ep, experts));
    TLLM_CHECK_WITH_INFO(bytes <= static_cast<std::size_t>(available),
        "HALO-R kernel needs %zu bytes of dynamic shared memory, the device allows %d", bytes, available);

    int active = 0;
    TLLM_CUDA_CHECK(sizeMoeRebalanceFusedKernel(kernel, kFusedThreads, static_cast<int>(bytes), active));
    // One CTA per SM: fewer cannot launch, and a second one would share the SM that the rendezvous counts on.
    TLLM_CHECK_WITH_INFO(active == 1, "HALO-R kernel must fit exactly one CTA per SM, got %d", active);

    MoeRebalanceHaloRLaunch launch;
    launch.ctas = ctas;
    launch.dynamicSharedBytes = static_cast<int>(bytes);
    // The relay's NVLink traffic runs in the remote memory-sync domain. Every edge to another domain is system scope
    // already: the route release, the READY publication and its release fence.
    launch.attributeCount = setMoeRebalanceRemoteSyncDomain(device, launch.attributes);
    return launch;
}

cudaError_t launchMoeRebalanceHaloR(MoeRebalanceHaloRLaunch const& launch, MoeRebalanceHaloRParams const& params,
    cudaStream_t stream, cudaEvent_t launched)
{
    // The prepared attributes, then the launch completion event.
    constexpr std::size_t kPrepared = sizeof(MoeRebalanceHaloRLaunch::attributes) / sizeof(cudaLaunchAttribute);
    cudaLaunchAttribute attributes[kPrepared + 1]{};
    std::copy_n(launch.attributes, launch.attributeCount, attributes);
    cudaLaunchAttribute& completion = attributes[launch.attributeCount];
    completion.id = cudaLaunchAttributeLaunchCompletionEvent;
    completion.val.launchCompletionEvent.event = launched;
    completion.val.launchCompletionEvent.flags = cudaEventRecordDefault;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(launch.ctas));
    config.blockDim = dim3(kFusedThreads);
    config.dynamicSmemBytes = static_cast<std::size_t>(launch.dynamicSharedBytes);
    config.stream = stream;
    config.attrs = attributes;
    config.numAttrs = launch.attributeCount + 1;
    // cudaLaunchKernelExC takes non-const argument pointers; pass a copy instead of casting const away.
    MoeRebalanceHaloRParams kernelParams = params;
    void* arguments[] = {&kernelParams};
    return cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(&haloRFusedKernel), arguments);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
