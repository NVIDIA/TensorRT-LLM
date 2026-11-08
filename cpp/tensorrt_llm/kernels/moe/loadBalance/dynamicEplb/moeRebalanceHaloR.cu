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
// (moeRebalanceTma.cu) and reuses its scheduler passes (moeRebalanceHaloQKernel.cuh), with three differences:
//   - The plan is HALO-R's own (moeRebalanceHaloRPlanner.h): arbitrary receiver masks, each receiver with its own
//     helper slot, copied with the unicast relay (moeRebalanceRelay.cuh) instead of multicast. With per-layer helper
//     slots, the relay skips each receiver whose slot already holds the expert, not a whole team.
//   - A CTA starts its relay share right after releasing its routes, without waiting for the other CTAs: CTA 0
//     publishes the plan only once it is complete, and the grid rendezvous already proved every CTA resident.
//   - A failing CTA traps before it releases its routes: every timeout, a planner status other than OK and any
//     nonzero status word, which also catches the invalid routes and unmappable slots that the route passes record
//     and continue past. The route consumer therefore never passes its wait for a failed launch, and the last CTA,
//     which re-checks the status words before it publishes READY, publishes nothing.
//   - READY and the route counter take constant values (moeRebalanceHaloR.h), so a captured launch replays exactly.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error "The HALO-R kernel requires SM90 or newer"
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

//! The shared passes partition routes over 16 warps per CTA, and one CTA must fit an SM for the grid rendezvous.
constexpr int kFusedThreads = 512;
constexpr int kFusedWarps = kFusedThreads / 32;
constexpr int kMinCtas = 2;
constexpr int kMaxCtas = 128;
constexpr int kMinEp = 2;
constexpr unsigned kFullWarpMask = 0xffffffffU;

static_assert(halo_r::kPlanFieldStride == kMaxBroadcasts, "HALO-R uses the shared plan field stride");
static_assert(static_cast<int>(halo_r::PlanField::kExpert) == kPlanExpert,
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

//! This warp's selected routes, as compacted by pre_materialize_and_compact_worker_routes: their route indices are
//! hotIndices[begin, begin + count) and their stable ordinals are still in the output.
struct WarpSelectedRoutes
{
    int begin;
    int count;
};

__device__ WarpSelectedRoutes warpSelectedRoutes(int routeCount, int experts, int worker, int workers)
{
    extern __shared__ int shared[];
    int const warp = threadIdx.x >> 5;
    int const warps = blockDim.x >> 5;
    int const workerSpan = (routeCount + workers - 1) / workers;
    int const workerBegin = worker * workerSpan;
    int const workerEnd = imin(routeCount, workerBegin + workerSpan);
    int const warpSpan = ((workerSpan + warps * 32 - 1) / (warps * 32)) * 32;
    int const begin = workerBegin + warp * warpSpan;
    int const end = imin(workerEnd, begin + warpSpan);
    return WarpSelectedRoutes{begin, imin(shared[warp * (experts + 1) + experts], imax(0, end - begin))};
}

//! Slot of every selected route of this worker's span, from the stable ordinals kept in shared memory.
__device__ void materializeSelectedRoutes(int const* routes, int* output, int const* routePrefix, int const* plan,
    int* routeAux, int* status, int routeCount, int ep, int experts, int helpers, int worker, int workers, Epoch epoch)
{
    extern __shared__ int shared[];
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const bins = experts + 1;
    int* expertLut = worker_expert_lut(experts);
    int* hotIndices = routeAux + 2 + workers * bins;
    int const home = experts / ep;
    int const localSlots = home + helpers;
    int const* owners = halo_r::planField(plan, halo_r::PlanField::kOwner);
    int const* masks = halo_r::planField(plan, halo_r::PlanField::kDestinationMask);
    int const* lowMasks = halo_r::planField(plan, halo_r::PlanField::kHelperLowMask);
    int const* highMasks = halo_r::planField(plan, halo_r::PlanField::kHelperHighMask);

    auto const [begin, hotCount] = warpSelectedRoutes(routeCount, experts, worker, workers);
    for (int offset = lane; offset < hotCount; offset += 32)
    {
        int const index = hotIndices[begin + offset];
        int const expert = routes[index];
        int const b = expertLut[expert];
        int const ordinal = output[index] + shared[warp * bins + expert];
        int const destination = route_destination(routePrefix + b * (ep + 1), ordinal, ep);
        int const slot = halo_r::selectedRouteSlot(expert, destination, owners[b], static_cast<unsigned>(masks[b]),
            static_cast<unsigned>(lowMasks[b]), static_cast<unsigned>(highMasks[b]), home, localSlots);
        if (slot < 0)
        {
            record_status(status + kHaloRStatusSlotOutsideMask, epoch);
        }
        output[index] = slot;
    }
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
    auto* scratch = reinterpret_cast<halo_r::PlannerScratch*>(quota + halo_r::kMaxBroadcasts * p.ep);
    __shared__ int planStatus;
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
        halo_r::PlannerConfig const cfg{p.ep, p.experts, p.helpers, p.sendCount, p.sendWeight, p.localRank};
        halo_r::PlannerOutputs const outputs{p.plan, quota, p.outIds, p.outOwners, nullptr, nullptr, nullptr};
        halo_r::PlanStatus const result = halo_r::planHaloR<DeviceWarp>(
            histogram, cfg, scratch, outputs, p.residency != nullptr ? resident : nullptr);
        if (threadIdx.x == 0)
        {
            planStatus = static_cast<int>(result);
            if (result != halo_r::PlanStatus::kOk)
            {
                record_status(p.status + kHaloRStatusPlanner, epoch);
            }
        }
    }
    __syncthreads();
    // Valid input never yields another status (moeRebalanceHaloRPlanner.h), so it is a bug, not a load condition.
    if (planStatus != static_cast<int>(halo_r::PlanStatus::kOk))
    {
        return RoleResult{"planner status", planStatus};
    }
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
    materializeSelectedRoutes(p.routes, p.outSlots, p.routePrefix, p.plan, p.routeAux, p.status, p.validRouteCount,
        p.ep, p.experts, p.helpers, worker, workers, epoch);
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

__device__ __forceinline__ void publishReady(std::uint64_t* flag, std::uint64_t value)
{
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.release.sys;" ::: "memory");
    asm volatile("multimem.st.release.sys.global.u64 [%0], %1;" ::"l"(flag), "l"(value) : "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
}

//! The last CTA, every thread: the copied and skipped receivers of each broadcast, from the residency table as the
//! relay read it, and then the broadcast's expert recorded in every receiver's helper slot. A plan assigns each
//! (receiver, helper slot) to one broadcast, so a thread reads and writes only its own broadcasts' table entries.
__device__ void accountBroadcasts(
    MoeRebalanceHaloRParams const& p, unsigned long long* receiverCopies, unsigned long long* receiverSkips)
{
    int const broadcasts = p.plan[1];
    int const* experts = halo_r::planField(p.plan, halo_r::PlanField::kExpert);
    int const* masks = halo_r::planField(p.plan, halo_r::PlanField::kDestinationMask);
    int const* lowMasks = halo_r::planField(p.plan, halo_r::PlanField::kHelperLowMask);
    int const* highMasks = halo_r::planField(p.plan, halo_r::PlanField::kHelperHighMask);
    for (int b = threadIdx.x; b < broadcasts; b += blockDim.x)
    {
        unsigned const mask = static_cast<unsigned>(masks[b]);
        unsigned const low = static_cast<unsigned>(lowMasks[b]);
        unsigned const high = static_cast<unsigned>(highMasks[b]);
        unsigned const need = halo_r::relayNeedMask(mask, low, high, experts[b], p.residency, p.helpers);
        atomicAdd(receiverCopies, static_cast<unsigned long long>(__popc(need)));
        atomicAdd(receiverSkips, static_cast<unsigned long long>(__popc(mask & ~need)));
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
}

//! Phase 4: a release/acquire chain over the CTAs elects the last one, which publishes READY. No CTA waits for
//! another, and a rank with nothing to copy joins the chain all the same.
//!
//! The residency table changes before READY: every read of it in this launch precedes the chain, and READY orders the
//! change before any later launch of the layer that acquires it. Every rank applies the same change, so the tables
//! stay identical.
__device__ void completeLaunch(MoeRebalanceHaloRParams const& p)
{
    __shared__ int last;
    __shared__ int failedWord;
    __shared__ unsigned long long receiverCopies;
    __shared__ unsigned long long receiverSkips;
    if (threadIdx.x == 0)
    {
        unsigned previous;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(previous) : "l"(p.completedCtas) : "memory");
        last = previous == gridDim.x - 1;
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
    accountBroadcasts(p, &receiverCopies, &receiverSkips);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        // Every CTA has arrived; the next launch is stream-ordered after this one, so the counter can be rearmed.
        asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(p.completedCtas) : "memory");
        publishReady(p.readyFlagMc, p.readyValue);
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
    if (!relayCopy(p))
    {
        trapLaunch(p, "no relay issuer fits the shared memory", 0);
        return;
    }

    completeLaunch(p);
}

//! Dynamic shared memory of the scheduler role.
std::size_t schedulerSharedBytes(int ep, int experts)
{
    std::size_t const workerInts = static_cast<std::size_t>(kFusedWarps) * (experts + 1) + experts;
    std::size_t const plannerInts = static_cast<std::size_t>(ep) * experts + experts
        + static_cast<std::size_t>(halo_r::kMaxBroadcasts) * ep + sizeof(halo_r::PlannerScratch) / sizeof(int);
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
    halo_r::PlannerConfig const cfg{
        params.ep, params.experts, params.helpers, params.sendCount, params.sendWeight, params.localRank};
    TLLM_CHECK_WITH_INFO(params.ep >= kMinEp && halo_r::validPlannerConfig(cfg),
        "Invalid HALO-R planner configuration: ep=%d experts=%d helpers=%d send_count=%d send_weight=%d "
        "local_rank=%d",
        params.ep, params.experts, params.helpers, params.sendCount, params.sendWeight, params.localRank);
    TLLM_CHECK_WITH_INFO(params.routeCapacity > 0 && params.routeCapacity <= INT_MAX / params.ep,
        "Invalid HALO-R route capacity %d", params.routeCapacity);
    // A candidate key stores weighted rows above the order byte; an expert's rows are bounded by all routed tokens.
    TLLM_CHECK_WITH_INFO(
        static_cast<long long>(params.ep) * params.routeCapacity * (1 + params.sendWeight) < halo_r::kMaxWeightedRows,
        "HALO-R route capacity %d at ep=%d and send_weight=%d overflows the planner's candidate keys",
        params.routeCapacity, params.ep, params.sendWeight);
    TLLM_CHECK_WITH_INFO(params.planes >= 1 && params.planes <= halo_r::kRelayMaxPlanes,
        "HALO-R relay supports 1 to %d weight planes, got %d", halo_r::kRelayMaxPlanes, params.planes);
    TLLM_CHECK_WITH_INFO(reinterpret_cast<std::uintptr_t>(params.readyFlags) % sizeof(std::uint64_t) == 0
            && reinterpret_cast<std::uintptr_t>(params.readyFlagMc) % sizeof(std::uint64_t) == 0
            && reinterpret_cast<std::uintptr_t>(params.routeCounter) % sizeof(std::uint64_t) == 0,
        "HALO-R READY row, READY cell and route counter must be 8-byte aligned");
    TLLM_CHECK_WITH_INFO(ctas >= kMinCtas && ctas <= kMaxCtas, "HALO-R kernel needs between %d and %d CTAs, got %d",
        kMinCtas, kMaxCtas, ctas);
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
    TLLM_CHECK_WITH_INFO(ctas >= kMinCtas && ctas <= std::min(kMaxCtas, smCount),
        "HALO-R kernel CTAs (%d) must be between %d and the device SM count (%d)", ctas, kMinCtas, smCount);

    void const* kernel = reinterpret_cast<void const*>(&haloRFusedKernel);
    cudaFuncAttributes attributes{};
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    std::size_t stack = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
    if (stack < attributes.localSizeBytes)
    {
        TLLM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, attributes.localSizeBytes));
    }
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

    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    int active = 0;
    TLLM_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, kFusedThreads, static_cast<int>(bytes)));
    // One CTA per SM: fewer cannot launch, and a second one would share the SM that the rendezvous counts on.
    TLLM_CHECK_WITH_INFO(active == 1, "HALO-R kernel must fit exactly one CTA per SM, got %d", active);

    MoeRebalanceHaloRLaunch launch;
    launch.ctas = ctas;
    launch.dynamicSharedBytes = static_cast<int>(bytes);
    // The relay's NVLink traffic runs in the remote memory-sync domain, so fences and completions of kernels in the
    // default domain do not wait for it. Every edge to another domain is system scope already: the route release, the
    // READY publication and its release fence.
    int domains = 0;
    if (cudaDeviceGetAttribute(&domains, cudaDevAttrMemSyncDomainCount, device) == cudaSuccess && domains >= 2)
    {
        launch.attributes[0].id = cudaLaunchAttributeMemSyncDomain;
        launch.attributes[0].val.memSyncDomain = cudaLaunchMemSyncDomainRemote;
        launch.attributes[1].id = cudaLaunchAttributeMemSyncDomainMap;
        launch.attributes[1].val.memSyncDomainMap.default_ = 0;
        launch.attributes[1].val.memSyncDomainMap.remote = 1;
        launch.attributeCount = 2;
    }
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
    void* arguments[] = {const_cast<MoeRebalanceHaloRParams*>(&params)};
    return cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(&haloRFusedKernel), arguments);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
