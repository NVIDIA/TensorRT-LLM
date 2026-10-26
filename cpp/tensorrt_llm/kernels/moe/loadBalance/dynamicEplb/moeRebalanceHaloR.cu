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

// HALO-R scheduler kernel. CTA 0 exchanges the routed-token counts with every EP rank, plans the layer's replicas
// with one warp and builds the route prefix of every selected expert; the worker CTAs compute stable per-expert
// ordinals, write the home slot of every unselected route and, once the prefix is published, the slot of every
// selected route.
//
// READY protocol. The MegaMoE READY gate passes when every cell of the layer's uint64[EP] READY row is at least the
// expected generation and carries no error bit. HALO-R hands the gate the constant generation 1 and keeps the
// protocol on the device, so that a captured layer replays without host bookkeeping:
//   - the relay copy of epoch g publishes g into its cell on every rank (bit 63 set on any detected error);
//   - before the count exchange of epoch g > 1, CTA 0 waits until every cell of this rank's row is nonzero (every
//     rank's epoch g - 1 copy has published here), records an error bit or a timeout in the sticky READY status word,
//     zeroes the row and only then arrives at the exchange. A peer's epoch g copy starts after that exchange, so the
//     gate of epoch g cannot pass on a value of epoch g - 1.
// A missing cell of this rank's own copy traps: the zeroed row would otherwise let this rank's gate read stale weights.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQKernel.cuh"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloR.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloRPlanner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error "The HALO-R scheduler requires SM90 or newer"
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

//! The shared passes partition routes over 16 warps per CTA, and one CTA must fit an SM for the grid rendezvous.
constexpr int kSchedulerThreads = 512;
constexpr int kSchedulerWarps = kSchedulerThreads / 32;
constexpr int kMinSchedulerCtas = 2;
constexpr int kMaxSchedulerCtas = 128;
constexpr int kMinEp = 2;
constexpr unsigned kFullWarpMask = 0xffffffffU;

//! The MegaMoE READY gate reserves bit 63 of a READY cell for errors.
constexpr std::uint64_t kReadyErrorBit = 1ULL << 63;
constexpr int kReadyStatusKindShift = 28;
constexpr int kReadyStatusErrorBit = 1 << kReadyStatusKindShift;
constexpr int kReadyStatusTimeout = 2 << kReadyStatusKindShift;
constexpr int kReadyStatusEpochMask = (1 << kReadyStatusKindShift) - 1;

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

__device__ __forceinline__ void systemFence()
{
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ std::uint64_t globalTimerNs()
{
    std::uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

//! Wait for every cell of this rank's READY row, record errors and zero the row. One thread. The wait is bounded in
//! wall-clock time (%globaltimer), independent of the SM clock.
__device__ void drainAndResetReadyRow(
    std::uint64_t* row, int ep, int localRank, int epoch, std::uint64_t spinNs, int* status)
{
    int kind = 0;
    std::uint64_t const start = globalTimerNs();
    for (int source = 0; source < ep; ++source)
    {
        std::uint64_t value;
        for (;;)
        {
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(row + source) : "memory");
            if (value != 0 || globalTimerNs() - start > spinNs)
            {
                break;
            }
        }
        if (value == 0)
        {
            if (source == localRank)
            {
                __trap();
            }
            kind |= kReadyStatusTimeout;
        }
        else if (value & kReadyErrorBit)
        {
            kind |= kReadyStatusErrorBit;
        }
    }
    // Acquire the values published through the multicast alias before resetting through the unicast alias.
    systemFence();
    if (kind)
    {
        atomicCAS(status + kHaloRStatusReady, 0, kind | (epoch & kReadyStatusEpochMask));
    }
    for (int source = 0; source < ep; ++source)
    {
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(row + source), "l"(0ULL) : "memory");
    }
    // Order the reset before this CTA's exchange arrival; a peer's next READY store follows its acquire of it.
    systemFence();
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

//! Timeout fail safe after the compaction: the owner's resident slot for every selected route of this worker's span,
//! which still holds a stable ordinal.
__device__ void materializeSelectedRoutesHome(int const* routes, int* output, int const* routeAux, int routeCount,
    int ep, int experts, int helpers, int worker, int workers)
{
    int const lane = threadIdx.x & 31;
    int const* hotIndices = routeAux + 2 + workers * (experts + 1);
    int const home = experts / ep;
    auto const [begin, hotCount] = warpSelectedRoutes(routeCount, experts, worker, workers);
    for (int offset = lane; offset < hotCount; offset += 32)
    {
        int const index = hotIndices[begin + offset];
        output[index] = halo_r::homeSlot(routes[index], home, home + helpers);
    }
}

//! CTA 0: count exchange, plan, plan publication and route prefix. Returns after publishing the prefix epoch.
__device__ void planAndPublish(MoeRebalanceHaloRParams const& p, Epoch epoch, unsigned long long spinCycles)
{
    extern __shared__ int shared[];
    int* counts = shared;
    int* histogram = counts + p.ep * p.experts;
    int* quota = histogram + p.experts;
    auto* scratch = reinterpret_cast<halo_r::PlannerScratch*>(quota + halo_r::kMaxBroadcasts * p.ep);
    __shared__ int planStatus;

    if (threadIdx.x == 0 && epoch > 1)
    {
        drainAndResetReadyRow(p.readyFlags, p.ep, p.localRank, static_cast<int>(epoch), p.spinNs, p.status);
    }
    __syncthreads();
    if (!exchange_counts<0>(reinterpret_cast<unsigned long long const*>(p.peerBases), p.partial, shared, p.ep,
            p.experts, p.localRank, p.ctas, epoch, spinCycles, p.status))
    {
        return;
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

    if (threadIdx.x < 32)
    {
        halo_r::PlannerConfig const cfg{p.ep, p.experts, p.helpers, p.sendCount, p.sendWeight, p.localRank};
        halo_r::PlannerOutputs const outputs{p.plan, quota, p.outIds, p.outOwners, nullptr, nullptr, nullptr};
        halo_r::PlanStatus const result = halo_r::planHaloR<DeviceWarp>(histogram, cfg, scratch, outputs);
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
    if (planStatus != static_cast<int>(halo_r::PlanStatus::kOk))
    {
        // Fail safe: no replicas, every route stays on its owner.
        if (threadIdx.x == 0)
        {
            p.plan[1] = 0;
        }
        if (threadIdx.x < p.helpers)
        {
            p.outIds[threadIdx.x] = -1;
            p.outOwners[threadIdx.x] = -1;
        }
        __syncthreads();
    }
    int const broadcasts = p.plan[1];
    if (threadIdx.x == 0)
    {
        store_release_device(p.plan, epoch);
    }
    build_route_prefix(counts, quota, p.routePrefix, p.plan, broadcasts, p.ep, p.experts, p.localRank, blockDim.x);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        store_release_device(p.gridSync + 2, epoch);
    }
}

//! spinCycles bounds the shared passes, which count SM clock cycles. It is the wall-clock budget converted at the
//! maximum SM clock, so those waits last at least params.spinNs; they all end before the routes are published.
__global__ __launch_bounds__(kSchedulerThreads, 1) void haloRSchedulerKernel(
    MoeRebalanceHaloRParams const p, unsigned long long const spinCycles)
{
    // The relay copy launched after this kernel counts its resident CTAs from zero (see moeRebalanceRelay.h).
    if (p.relayResidentCtas != nullptr && blockIdx.x == 0 && threadIdx.x == 0)
    {
        *p.relayResidentCtas = 0;
    }
    // Only the caller's valid prefix is readable; clear the rest of the output capacity in the same kernel.
    for (int index = p.validRouteCount + blockIdx.x * blockDim.x + threadIdx.x; index < p.routeCapacity;
         index += gridDim.x * blockDim.x)
    {
        p.outSlots[index] = -1;
    }
    local_histogram_pass(p.routes, p.partial, p.validRouteCount, p.experts, p.ctas);
    Epoch epoch;
    if (!grid_rendezvous(p.gridSync, p.ctas, spinCycles, p.status, &epoch))
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            // The epoch that timed out still resets the READY row, so this rank's gate cannot pass on the previous
            // epoch's values, and the sticky status makes every later READY publication carry the error bit.
            int const failed = *reinterpret_cast<int const volatile*>(p.status + kHaloRStatusGridTimeout);
            if (failed > 1)
            {
                drainAndResetReadyRow(p.readyFlags, p.ep, p.localRank, failed, p.spinNs, p.status);
            }
            atomicCAS(p.status + kHaloRStatusReady, 0, kReadyStatusTimeout | (failed & kReadyStatusEpochMask));
        }
        return;
    }

    if (blockIdx.x == 0)
    {
        planAndPublish(p, epoch, spinCycles);
        return;
    }

    // On a worker timeout, the ordinal pass has already written stable ordinals into the output, which MegaMoE would
    // read as slots. The worker routes its span to the owners' resident slots instead. The sticky status word still
    // sets the READY error bit, so the gate rejects every helper tile of the layer.
    int const worker = blockIdx.x - 1;
    int const workers = p.ctas - 1;
    bool const ordinalsReady = stable_worker_ordinal_pass(
        p.routes, p.outSlots, p.routeAux, p.status, p.validRouteCount, p.experts, worker, workers, spinCycles);
    int* expertLut = worker_expert_lut(p.experts);
    prepare_worker_default_slots(expertLut, p.ep, p.experts, p.helpers);
    // Before it publishes the plan, CTA 0 may spend up to spinNs draining READY and spinCycles on the count exchange,
    // and then runs the planner. spinCycles lasts at least spinNs, so two budgets cover both waits and the third
    // covers the planner. The host check on spinNs keeps spinCycles at most ULLONG_MAX / 10^6 + 1, so the product
    // cannot overflow.
    constexpr unsigned long long kPlanWaitBudgets = 3;
    __shared__ int planReady;
    if (!ordinalsReady || !wait_for_plan_epoch(p.plan, epoch, kPlanWaitBudgets * spinCycles, p.status, &planReady))
    {
        // The expert LUT holds only the default slots, so every route of the span goes to its owner.
        pre_materialize_and_compact_worker_routes(
            p.routes, p.outSlots, p.routeAux, p.status, p.validRouteCount, p.experts, worker, workers, epoch);
        return;
    }
    prepare_worker_selected_experts(expertLut, p.plan);
    pre_materialize_and_compact_worker_routes(
        p.routes, p.outSlots, p.routeAux, p.status, p.validRouteCount, p.experts, worker, workers, epoch);
    if (!wait_for_plan_epoch(p.gridSync + 2, epoch, spinCycles, p.status, &planReady))
    {
        materializeSelectedRoutesHome(
            p.routes, p.outSlots, p.routeAux, p.validRouteCount, p.ep, p.experts, p.helpers, worker, workers);
        return;
    }
    materializeSelectedRoutes(p.routes, p.outSlots, p.routePrefix, p.plan, p.routeAux, p.status, p.validRouteCount,
        p.ep, p.experts, p.helpers, worker, workers, epoch);
}

//! Per-device launch limits, queried once per device.
struct SchedulerDeviceLimits
{
    int smCount = 0;
    int clockKhz = 0;
    std::size_t sharedBytesLimit = 0; //!< Opt-in dynamic shared memory left after the kernel's static shared memory.
    std::size_t configuredSharedBytes = 0;
};

SchedulerDeviceLimits& schedulerDeviceLimits(int device, std::mutex& mutex)
{
    static std::map<int, SchedulerDeviceLimits> limits;
    std::lock_guard<std::mutex> lock(mutex);
    auto [entry, inserted] = limits.try_emplace(device);
    if (inserted)
    {
        int optinBytes = 0;
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&entry->second.smCount, cudaDevAttrMultiProcessorCount, device));
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&entry->second.clockKhz, cudaDevAttrClockRate, device));
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&optinBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        cudaFuncAttributes attributes{};
        TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attributes, haloRSchedulerKernel));
        entry->second.sharedBytesLimit = static_cast<std::size_t>(optinBytes) > attributes.sharedSizeBytes
            ? static_cast<std::size_t>(optinBytes) - attributes.sharedSizeBytes
            : 0;
        TLLM_CHECK_WITH_INFO(entry->second.clockKhz > 0, "Invalid SM clock rate %d kHz", entry->second.clockKhz);
    }
    return entry->second;
}

} // namespace

std::size_t getMoeRebalanceHaloRSharedMemoryBytes(int ep, int experts)
{
    std::size_t const workerInts = static_cast<std::size_t>(kSchedulerWarps) * (experts + 1) + experts;
    std::size_t const plannerInts = static_cast<std::size_t>(ep) * experts + experts
        + static_cast<std::size_t>(halo_r::kMaxBroadcasts) * ep + sizeof(halo_r::PlannerScratch) / sizeof(int);
    return std::max(workerInts, plannerInts) * sizeof(int);
}

std::int64_t getMoeRebalanceHaloRRouteAuxEntries(int experts, int ctas, int routeCapacity)
{
    std::int64_t const bins = static_cast<std::int64_t>(experts) + 1;
    std::int64_t const passEntries = static_cast<std::int64_t>(ctas) * kSchedulerWarps * bins;
    std::int64_t const workerEntries = 2 + static_cast<std::int64_t>(ctas - 1) * bins + routeCapacity;
    return std::max(passEntries, workerEntries);
}

void invokeMoeRebalanceHaloR(MoeRebalanceHaloRParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO((params.routes != nullptr || params.validRouteCount == 0) && params.outSlots != nullptr
            && params.outIds != nullptr && params.outOwners != nullptr && params.peerBases != nullptr
            && params.status != nullptr && params.partial != nullptr && params.routeAux != nullptr
            && params.gridSync != nullptr && params.plan != nullptr && params.routePrefix != nullptr
            && params.readyFlags != nullptr,
        "HALO-R scheduler buffers must not be null");
    halo_r::PlannerConfig const cfg{
        params.ep, params.experts, params.helpers, params.sendCount, params.sendWeight, params.localRank};
    TLLM_CHECK_WITH_INFO(params.ep >= kMinEp && halo_r::validPlannerConfig(cfg),
        "Invalid HALO-R planner configuration: ep=%d experts=%d helpers=%d send_count=%d send_weight=%d "
        "local_rank=%d",
        params.ep, params.experts, params.helpers, params.sendCount, params.sendWeight, params.localRank);
    TLLM_CHECK_WITH_INFO(params.routeCapacity > 0 && params.validRouteCount >= 0
            && params.validRouteCount <= params.routeCapacity && params.routeCapacity <= INT_MAX / params.ep,
        "Invalid HALO-R route capacity %d or valid route count %d", params.routeCapacity, params.validRouteCount);
    // A candidate key stores weighted rows above the order byte; an expert's rows are bounded by all routed tokens.
    TLLM_CHECK_WITH_INFO(
        static_cast<long long>(params.ep) * params.routeCapacity * (1 + params.sendWeight) < halo_r::kMaxWeightedRows,
        "HALO-R route capacity %d at ep=%d and send_weight=%d overflows the planner's candidate keys",
        params.routeCapacity, params.ep, params.sendWeight);
    TLLM_CHECK_WITH_INFO(params.ctas >= kMinSchedulerCtas && params.ctas <= kMaxSchedulerCtas,
        "HALO-R scheduler needs between %d and %d CTAs, got %d", kMinSchedulerCtas, kMaxSchedulerCtas, params.ctas);

    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    static std::mutex limitsMutex;
    SchedulerDeviceLimits& limits = schedulerDeviceLimits(device, limitsMutex);
    // The grid rendezvous needs every CTA resident at once, one CTA per SM.
    TLLM_CHECK_WITH_INFO(params.ctas <= limits.smCount, "HALO-R scheduler CTAs (%d) exceed the device SM count (%d)",
        params.ctas, limits.smCount);
    std::size_t const sharedBytes = getMoeRebalanceHaloRSharedMemoryBytes(params.ep, params.experts);
    TLLM_CHECK_WITH_INFO(sharedBytes <= limits.sharedBytesLimit,
        "HALO-R scheduler needs %zu bytes of shared memory, the device allows %zu", sharedBytes,
        limits.sharedBytesLimit);
    {
        std::lock_guard<std::mutex> lock(limitsMutex);
        if (limits.configuredSharedBytes < sharedBytes)
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(
                haloRSchedulerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
            limits.configuredSharedBytes = sharedBytes;
        }
    }
    TLLM_CHECK_WITH_INFO(params.spinNs > 0 && params.spinNs <= ULLONG_MAX / static_cast<unsigned>(limits.clockKhz),
        "HALO-R spin budget must be positive and convertible to SM clock cycles");
    // kHz are cycles per millisecond; round up so that the shared passes wait at least the wall-clock budget.
    unsigned long long const spinCycles
        = (static_cast<unsigned long long>(params.spinNs) * static_cast<unsigned>(limits.clockKhz) + 999'999ULL)
        / 1'000'000ULL;

    haloRSchedulerKernel<<<params.ctas, kSchedulerThreads, sharedBytes, stream>>>(params, spinCycles);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
