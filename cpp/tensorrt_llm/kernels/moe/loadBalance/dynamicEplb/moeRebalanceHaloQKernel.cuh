/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* HALO-Q scheduler device code, the first phase of the fused scheduler +
 * in-switch copy kernel. */
#ifndef TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_KERNEL_CUH
#define TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_KERNEL_CUH

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalancePlanPolicy.h"

#include <cuda/atomic>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
using sami_copy::group_contains;
using sami_copy::imax;
using sami_copy::imin;
using sami_copy::kMaxEp;
using sami_copy::kMinGroup;
constexpr int kMaxExperts = 384;
constexpr int kMaxBroadcasts = kMaxExperts;
constexpr int kHelperMaskWords = (kMaxBroadcasts + 31) / 32;
/* Rank loops run in fully unrolled chunks of this many predicated ranks, so
 * every load of a chunk is independent and in flight together. */
constexpr int kRankChunk = 8;
constexpr int kSymPayloadOffset = 64;
using Epoch = std::uint32_t;

enum PlanField
{
    kPlanExpert = 0,
    kPlanLevel = 1,
    kPlanGroupBegin = 2,
    kPlanGroupSize = 3,
    kPlanOwner = 4,
    kPlanHelper = 5,
};

__device__ __forceinline__ Epoch fetch_add_acq_rel_device(int* address, Epoch value)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_device> atom(*valueAddress);
    return atom.fetch_add(value, cuda::memory_order_acq_rel);
}

__device__ __forceinline__ Epoch load_acquire_device(int* address)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_device> atom(*valueAddress);
    return atom.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ Epoch load_relaxed_device(int* address)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_device> atom(*valueAddress);
    return atom.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ void store_relaxed_device(int* address, Epoch value)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_device> atom(*valueAddress);
    atom.store(value, cuda::memory_order_relaxed);
}

__device__ __forceinline__ void store_release_device(int* address, Epoch value)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_device> atom(*valueAddress);
    atom.store(value, cuda::memory_order_release);
}

__device__ __forceinline__ Epoch load_acquire_system(int* address)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_system> atom(*valueAddress);
    return atom.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ Epoch load_relaxed_system(int* address)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_system> atom(*valueAddress);
    return atom.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ void store_release_system(int* address, Epoch value)
{
    auto* valueAddress = reinterpret_cast<Epoch*>(address);
    cuda::atomic_ref<Epoch, cuda::thread_scope_system> atom(*valueAddress);
    atom.store(value, cuda::memory_order_release);
}

__device__ __forceinline__ bool epoch_before(Epoch observed, Epoch expected)
{
    // RFC-1982-style serial comparison: all arithmetic remains unsigned and
    // is unambiguous while peers differ by fewer than half the epoch space.
    return observed != expected && expected - observed < (Epoch{1} << 31);
}

__device__ __forceinline__ void record_status(int* address, Epoch epoch)
{
    // Zero is reserved for "never failed". Preserve a nonzero sticky error at
    // the one modular epoch whose bit pattern is zero.
    Epoch const value = epoch == 0 ? ~Epoch{0} : epoch;
    atomicExch(reinterpret_cast<Epoch*>(address), value);
}

__device__ __forceinline__ int* plan_field(int* plan, int field)
{
    return plan + 2 + field * kMaxBroadcasts;
}

__host__ __device__ __forceinline__ unsigned rank_group_mask(int begin, int size)
{
    return size == 32 ? 0xffffffffu : ((1u << size) - 1u) << begin;
}

__host__ __device__ __forceinline__ int hierarchy_depth(int ep)
{
    int depth = 1;
    for (int level = 1; (1 << level) <= ep; ++level)
    {
        int groups = 1 << level;
        if ((ep & (groups - 1)) || (ep >> level) < kMinGroup)
            break;
        ++depth;
    }
    return depth;
}

/* The planner's levels: the multicast ladder, then, for EP >= 4, aligned rank
 * pairs (pair_level_of; -1 without one). */
__host__ __device__ __forceinline__ int pair_level_of(int ep)
{
    return ep >= 4 ? hierarchy_depth(ep) : -1;
}

__host__ __device__ __forceinline__ int planner_depth(int ep)
{
    // A power of two from 2 up: the ladder ep .. 4 and pairs, log2(ep) levels.
    if (ep >= 2 && !(ep & (ep - 1)))
    {
#ifdef __CUDA_ARCH__
        return __ffs(ep) - 1;
#else
        return __builtin_ctz(static_cast<unsigned>(ep));
#endif
    }
    return hierarchy_depth(ep) + (ep >= 4 ? 1 : 0);
}

/* log2 of a power of two, at compile time. */
__host__ __device__ constexpr int log2_pow2(int x)
{
    return x > 1 ? 1 + log2_pow2(x >> 1) : 0;
}

/* Planner levels of any EP up to kMaxEp: group sizes 32, 16, 8, 4 and pairs. */
constexpr int kMaxLevels = 5;

/* x / ep for x >= 0, by a shift when ep is a power of two: a runtime division
 * costs a few dozen instructions on the planner's critical path. */
__host__ __device__ __forceinline__ int div_by_ep(int x, int ep)
{
    if (ep & (ep - 1))
        return x / ep;
#ifdef __CUDA_ARCH__
    return x >> (__ffs(ep) - 1);
#else
    return x >> __builtin_ctz(static_cast<unsigned>(ep));
#endif
}

/* Exports per owner: its heaviest S experts, at most a warp of them, so a rank
 * sends at most S distinct experts as it receives at most S. */
__host__ __device__ __forceinline__ int candidate_count(int ep, int experts, int helpers)
{
    int const home = div_by_ep(experts, ep);
    int const wanted = helpers < home ? helpers : home;
    return wanted < 32 ? wanted : 32;
}

/* Distinct experts and total receive capacity independently bound broadcasts:
 * every broadcast takes a helper slot on at least one rank.  Actual S is
 * retained in the public outputs; only scratch uses this bound. */
__host__ __device__ __forceinline__ int broadcast_capacity(int ep, int experts, int helpers)
{
    int const bounded_helpers = helpers < experts ? helpers : experts;
    int const receive_bound = ep * bounded_helpers;
    return receive_bound < experts ? receive_bound : experts;
}

/* Placement runs once per target, one warp each, and the plan with the lowest
 * maximum rank load wins (the lower target on a tie).  Warp w aims at
 * max(M0, ceil(A * (400 + margin) / (400 * ep))), margin = 0, 2, 4, 8: up to 2%
 * above the mean load A / ep.  M0 is the lowest maximum load any plan reaches
 * when a rank sends at most its candidates. */
constexpr int kPlannerWarps = 4;
constexpr int kPlannerScratchInts = 10752; // three dense EP32 S3 plans (3 x 3552) behind a direct warp 0

__host__ __device__ __forceinline__ int target_margin_400(int warp)
{
    return warp == 0 ? 0 : warp == 1 ? 2 : warp == 2 ? 4 : 8;
}

/* Ints of one target's dense plan: five plan fields and an ep-wide quota row
 * per broadcast. */
__host__ __device__ __forceinline__ int planner_dense_ints(int ep, int experts, int helpers)
{
    return (5 + ep) * broadcast_capacity(ep, experts, helpers);
}

/* Dense plans when every target's fits in kPlannerScratchInts, else compact
 * ones: five plan fields per broadcast, where each broadcast's receivers start,
 * and (rank, tokens) per receiver.  A receiver holds the expert in a helper
 * slot, so there are at most ep * S of them. */
__host__ __device__ __forceinline__ bool planner_dense(int ep, int experts, int helpers)
{
    return kPlannerWarps * planner_dense_ints(ep, experts, helpers) <= kPlannerScratchInts;
}

__host__ __device__ __forceinline__ int planner_warp_ints(int ep, int experts, int helpers)
{
    int const bounded_helpers = helpers < experts ? helpers : experts;
    return planner_dense(ep, experts, helpers)
        ? planner_dense_ints(ep, experts, helpers)
        : 6 * broadcast_capacity(ep, experts, helpers) + 1 + 2 * ep * bounded_helpers;
}

/* As many targets as kPlannerScratchInts holds, at least one. */
__host__ __device__ __forceinline__ int planner_targets(int ep, int experts, int helpers)
{
    // min(kPlannerWarps, max(1, kPlannerScratchInts / ints)) without a division.
    int const ints = planner_warp_ints(ep, experts, helpers);
    int fit = 1;
#pragma unroll
    for (int warps = 2; warps <= kPlannerWarps; ++warps)
        fit = warps * ints <= kPlannerScratchInts ? warps : fit;
    return fit;
}

/* When four dense plans do not fit (EP32 S3), warp 0 writes plan[] and the quota rows itself and the other
 * targets keep dense plans in targets - 1 slots (slot = warp - 1) if they fit. */
__host__ __device__ __forceinline__ bool planner_dense_spare(int ep, int experts, int helpers)
{
    return !planner_dense(ep, experts, helpers)
        && (planner_targets(ep, experts, helpers) - 1) * planner_dense_ints(ep, experts, helpers)
        <= kPlannerScratchInts;
}

__device__ bool grid_rendezvous(
    int* sync, int blocks, unsigned long long spin_cycles, int* status, Epoch* completedEpoch)
{
    __shared__ Epoch generation;
    __shared__ int success;
    /* The block barrier orders every writer's stores before thread zero's
     * device-scope release advertises this CTA. */
    __syncthreads();
    if (threadIdx.x == 0)
    {
        success = 1;
        Epoch const current = load_relaxed_device(sync + 1);
        Epoch const target = current + 1;
        generation = target;
        Epoch const old = fetch_add_acq_rel_device(sync, 1);
        if (old == static_cast<Epoch>(blocks - 1))
        {
            // The last CTA resets arrivals before publishing the next modular
            // epoch. Stream ordering prevents a new launch from entering until
            // every CTA in this launch has observed the release.
            store_relaxed_device(sync, 0);
            store_release_device(sync + 1, target);
        }
        else if (old >= static_cast<Epoch>(blocks))
        {
            record_status(status + 1, target);
            success = 0;
        }
        else
        {
            unsigned long long const start = clock64();
            Epoch observed = load_relaxed_device(sync + 1);
            while (epoch_before(observed, target))
            {
                if (clock64() - start > spin_cycles)
                {
                    record_status(status + 1, target);
                    success = 0;
                    break;
                }
                observed = load_relaxed_device(sync + 1);
            }
            if (!epoch_before(observed, target))
                (void) load_acquire_device(sync + 1);
        }
    }
    __syncthreads();
    *completedEpoch = generation;
    return success != 0;
}

__device__ void local_histogram_pass(int const* routes, int* partial, int route_count, int expert_count, int ctas)
{
    extern __shared__ int shared[];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const warps = blockDim.x >> 5;
    int const bins = expert_count + 1;
    int const shards = warps > 8 ? 8 : warps;
    int const shared_count = shards * bins;
    for (int index = tid; index < shared_count; index += blockDim.x)
        shared[index] = 0;
    __syncthreads();

    int const cta_span = (route_count + ctas - 1) / ctas;
    int const cta_begin = blockIdx.x * cta_span;
    int const cta_end = imin(route_count, cta_begin + cta_span);
    int const warp_span = ((cta_span + warps * 32 - 1) / (warps * 32)) * 32;
    int const begin = cta_begin + warp * warp_span;
    int const end = imin(cta_end, begin + warp_span);
    /* Issue a small load group before consuming it to expose memory-level
     * parallelism. Batching does not change which bin a route lands in, and
     * atomicAdd order does not matter because integer addition commutes. */
    constexpr int kBatch = 4;
    for (int base_index = begin; base_index < end; base_index += 32 * kBatch)
    {
        int expert[kBatch];
#pragma unroll
        for (int step = 0; step < kBatch; ++step)
        {
            int const index = base_index + step * 32 + lane;
            expert[step] = index < end ? routes[index] : -1;
        }
#pragma unroll
        for (int step = 0; step < kBatch; ++step)
        {
            int const index = base_index + step * 32 + lane;
            if (index < end)
            {
                int const value = expert[step];
                int const bin = (value >= 0 && value < expert_count) ? value : expert_count;
                atomicAdd(shared + (warp & 7) * bins + bin, 1);
            }
        }
    }
    __syncthreads();

    for (int expert = tid; expert < expert_count; expert += blockDim.x)
    {
        int count = 0;
        for (int shard = 0; shard < shards; ++shard)
            count += shared[shard * bins + expert];
        partial[blockIdx.x * expert_count + expert] = count;
    }
}

/* Re-rank all local routes on the C-1 worker CTAs while CTA0 performs the
 * cross-rank exchange and planner.  route_aux[0:2] is a reusable worker
 * rendezvous; the following NWORK*(E+1) entries hold per-worker totals.
 * Per-warp cursors remain in this CTA's shared memory until materialization. */
__device__ bool stable_worker_ordinal_pass(int const* routes, int* ordinal, int* route_aux, int* status,
    int route_count, int expert_count, int worker, int workers, unsigned long long spin_cycles)
{
    extern __shared__ int shared[];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const warps = blockDim.x >> 5;
    int const bins = expert_count + 1;
    int const shared_count = warps * bins;
    for (int index = tid; index < shared_count; index += blockDim.x)
        shared[index] = 0;
    __syncthreads();

    int const worker_span = (route_count + workers - 1) / workers;
    int const worker_begin = worker * worker_span;
    int const worker_end = imin(route_count, worker_begin + worker_span);
    int const warp_span = ((worker_span + warps * 32 - 1) / (warps * 32)) * 32;
    int const begin = worker_begin + warp * warp_span;
    int const end = imin(worker_end, begin + warp_span);
    for (int base_index = begin; base_index < end; base_index += 32)
    {
        int const index = base_index + lane;
        bool const active = index < end;
        int const expert = active ? routes[index] : -1;
        // Padding has its own key so it cannot contribute to real/invalid bins.
        int const bin = !active ? bins : ((expert >= 0 && expert < expert_count) ? expert : expert_count);
        unsigned const peers = __match_any_sync(0xffffffffu, bin);
        int const leader = __ffs(peers) - 1;
        int const rank = __popc(peers & ((1u << lane) - 1u));
        int base = 0;
        if (active && lane == leader)
            base = atomicAdd(shared + warp * bins + bin, __popc(peers));
        // One collective for the entire warp; each lane selects its own leader.
        base = __shfl_sync(0xffffffffu, base, leader);
        if (active)
            ordinal[index] = base + rank;
    }
    __syncthreads();

    int* worker_totals = route_aux + 2;
    for (int bin = tid; bin < bins; bin += blockDim.x)
    {
        int total = 0;
        for (int local_warp = 0; local_warp < warps; ++local_warp)
            total += shared[local_warp * bins + bin];
        worker_totals[worker * bins + bin] = total;
    }

    /* The first all-CTA rendezvous guarantees every worker is resident, so a
     * worker-only global barrier cannot wait on an unscheduled CTA. */
    Epoch workerEpoch;
    if (!grid_rendezvous(route_aux, workers, spin_cycles, status, &workerEpoch))
        return false;

    for (int bin = tid; bin < bins; bin += blockDim.x)
    {
        int cursor = 0;
        for (int previous = 0; previous < worker; ++previous)
            cursor += worker_totals[previous * bins + bin];
        for (int local_warp = 0; local_warp < warps; ++local_warp)
        {
            int index = local_warp * bins + bin;
            int count = shared[index];
            shared[index] = cursor;
            cursor += count;
        }
    }
    __syncthreads();
    return true;
}

__device__ bool exchange_counts(unsigned long long const* peer_bases, int const* partial, int* shared, int ep,
    int expert_count, int local_rank, int ctas, Epoch epoch, unsigned long long spin_cycles, int* status)
{
    int const tid = threadIdx.x;
    __shared__ int wait_failed;
    if (tid == 0)
        wait_failed = 0;
    /* Keep the simple accumulation loop; the compiler provides the required
     * pipelining without explicit batching. */
    for (int expert = tid; expert < expert_count; expert += blockDim.x)
    {
        int count = 0;
        for (int cta = 0; cta < ctas; ++cta)
            count += partial[cta * expert_count + expert];
        shared[expert] = count;
    }
    __syncthreads();

    int const parity = epoch & 1;
    /* One warp owns each destination row.  Its warp barrier orders every
     * lane's stores before lane zero's system-scope release publishes the
     * complete row to the peer acquire. */
    int const warp = tid >> 5;
    int const lane = tid & 31;
    for (int destination = warp; destination < ep; destination += blockDim.x >> 5)
    {
        int* peer = reinterpret_cast<int*>(peer_bases[destination]);
        int* row = peer + kSymPayloadOffset + (parity * ep + local_rank) * expert_count;
        /* Keep scalar remote stores for the publication path; the local mirror
         * copy below uses vector transfers when alignment permits. */
        for (int expert = lane; expert < expert_count; expert += 32)
            row[expert] = shared[expert];
        /* Warp synchronization orders every lane's row stores before lane
         * zero resumes; its system-scope release then publishes that complete
         * transitive happens-before set to the peer acquire. */
        __syncwarp();
        if (lane == 0)
            store_release_system(peer + local_rank, epoch);
    }
    __syncthreads();

    int* local = reinterpret_cast<int*>(peer_bases[local_rank]);
    if (tid < ep)
    {
        unsigned long long start = clock64();
        Epoch observed = load_relaxed_system(local + tid);
        while (epoch_before(observed, epoch))
        {
            if (clock64() - start > spin_cycles)
            {
                record_status(status + 0, epoch);
                atomicExch(&wait_failed, 1);
                break;
            }
            observed = load_relaxed_system(local + tid);
        }
        if (!epoch_before(observed, epoch))
            (void) load_acquire_system(local + tid);
    }
    __syncthreads();
    if (wait_failed)
        return false;
    int const* payload = local + kSymPayloadOffset + parity * ep * expert_count;
    int const total = ep * expert_count;
    /* Vectorize the local mirror copy when shape and alignment permit. */
    if ((total & 3) == 0 && (reinterpret_cast<uintptr_t>(payload) & 15) == 0)
    {
        int4 const* source = reinterpret_cast<int4 const*>(payload);
        int4* destination = reinterpret_cast<int4*>(shared);
        int const vectors = total >> 2;
        for (int vector = tid; vector < vectors; vector += blockDim.x)
            destination[vector] = source[vector];
    }
    else
    {
        for (int index = tid; index < total; index += blockDim.x)
            shared[index] = payload[index];
    }
    __syncthreads();
    return true;
}

/* Rank and home bounds of the register prepare path: an owner warp sums at
 * most 8 source ranks, and a home of at most 64 experts fits two 32-lane
 * rounds. */
constexpr int kRegisterEp = 8;
constexpr int kRegisterRounds = 2;

__device__ __forceinline__ bool uses_register_prepare(int ep, int experts)
{
    return ep <= kRegisterEp && div_by_ep(experts, ep) <= 32 * kRegisterRounds;
}

/* Pick the owner's hot list from the lane-resident home masses: the heaviest
 * remaining expert per slot, the lowest home index on a tie.  Each lane holds
 * home experts lane + 32 * round; entries outside the home carry -1. */
template <int Rounds>
__device__ __forceinline__ void select_owner_candidates(int (&local_mass)[Rounds], int* histogram, int* candidates,
    int owner, int home, int candidate_count, bool write_mass)
{
    int const lane = threadIdx.x & 31;
    for (int slot = 0; slot < candidate_count; ++slot)
    {
        int best_mass = -1;
        int best_local = INT_MAX;
#pragma unroll
        for (int round = 0; round < Rounds; ++round)
        {
            int const mass = local_mass[round];
            int const local = lane + 32 * round;
            if (mass > best_mass || (mass == best_mass && local < best_local))
            {
                best_mass = mass;
                best_local = local;
            }
        }
        int const warp_mass = __reduce_max_sync(0xffffffffu, best_mass);
        int const winner = __reduce_min_sync(0xffffffffu, best_mass == warp_mass ? best_local : INT_MAX);
        if (lane == 0)
        {
            int const expert = owner * home + winner;
            candidates[owner * candidate_count + slot] = expert;
            if (write_mass)
                histogram[expert] = warp_mass;
        }
#pragma unroll
        for (int round = 0; round < Rounds; ++round)
            if (lane + 32 * round == winner)
                local_mass[round] = -1;
    }
}

/* Rank loads and per-owner hot lists for uses_register_prepare(): one warp
 * per owner sums its home masses over the source ranks straight into
 * registers.  Every load is unconditional (out-of-range sources and experts
 * read a clamped valid address and contribute nothing), so all of them are in
 * flight together.  Only the candidates' masses reach histogram[]; the
 * planners read no other entry.  The owner warps, warp 0 among them, meet at
 * a named barrier; the other warps leave at once. */
__device__ void prepare_halo_m_register(
    int const* counts, int* histogram, int* load, int* candidates, int ep, int experts, int helpers)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    if (warp >= ep)
        return;
    int const owner = warp;
    int const home = experts / ep;
    int local_mass[kRegisterRounds];
    int rank_load = 0;
#pragma unroll
    for (int round = 0; round < kRegisterRounds; ++round)
    {
        int const local = lane + 32 * round;
        bool const in_home = local < home;
        int const expert = owner * home + imin(local, home - 1);
        int mass = 0;
#pragma unroll
        for (int source = 0; source < kRegisterEp; ++source)
        {
            int const count = counts[imin(source, ep - 1) * experts + expert];
            mass += source < ep ? count : 0;
        }
        rank_load += in_home ? mass : 0;
        local_mass[round] = in_home ? mass : -1;
    }
    rank_load = __reduce_add_sync(0xffffffffu, rank_load);
    if (lane == 0)
        load[owner] = rank_load;
    select_owner_candidates(
        local_mass, histogram, candidates, owner, home, candidate_count(ep, experts, helpers), true);
    asm volatile("bar.sync 4, %0;" ::"r"(ep * 32) : "memory");
}

/* Rounds of 32 home experts per lane; EP2 has the largest home, 192 experts. */
constexpr int kMaxHomeRounds = (kMaxExperts / 2 + 31) / 32;

/* Dense histogram, rank loads and per-owner hot lists for every other
 * geometry. */
__device__ void prepare_halo_m(
    int const* counts, int* histogram, int* load, int* candidates, int ep, int experts, int helpers)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const home = experts / ep;
    for (int expert = tid; expert < experts; expert += blockDim.x)
    {
        int mass = 0;
#pragma unroll
        for (int rank = 0; rank < ep; ++rank)
            mass += counts[rank * experts + expert];
        histogram[expert] = mass;
    }
    __syncthreads();

    /* Homes of at most 32 experts: one thread per expert finds its place in the
     * owner's hot list and the first one the owner's load; else a warp per owner. */
    int const candidates_per_owner = candidate_count(ep, experts, helpers);
    for (int expert = tid; home <= 32 && expert < ep * home; expert += blockDim.x)
    {
        int const owner = expert / home;
        int const local = expert - owner * home;
        int const mass = histogram[expert];
        int place = 0;
        int rank_load = 0;
        for (int other = 0; other < home; ++other)
        {
            int const other_mass = histogram[owner * home + other];
            rank_load += other_mass;
            place += other_mass > mass || (other_mass == mass && other < local);
        }
        if (place < candidates_per_owner)
            candidates[owner * candidates_per_owner + place] = expert;
        if (local == 0)
            load[owner] = rank_load;
    }
    for (int owner = warp; home > 32 && owner < ep; owner += blockDim.x >> 5)
    {
        int rank_load = 0;
        for (int local = lane; local < home; local += 32)
            rank_load += histogram[owner * home + local];
        rank_load = __reduce_add_sync(0xffffffffu, rank_load);
        if (lane == 0)
            load[owner] = rank_load;
        int local_mass[kMaxHomeRounds];
#pragma unroll
        for (int round = 0; round < kMaxHomeRounds; ++round)
        {
            int const local = lane + 32 * round;
            local_mass[round] = local < home ? histogram[owner * home + local] : -1;
        }
        select_owner_candidates(local_mass, histogram, candidates, owner, home, candidates_per_owner, false);
    }
    __syncthreads();
}

/* Candidates per owner the planner keeps in registers: S <= 4 needs no read. */
constexpr int kHeldCandidates = 4;

/* HALO-Q placement toward one target by one warp, lane = rank, all state in
 * registers.  Levels: the multicast ladder (hierarchy_depth(ep)) and, for
 * EP >= 4, aligned rank pairs (planner_depth).  A pair is a multicast group:
 * both members hold the expert and pay a helper slot (an owner outside it pays
 * none).  Each owner exports up to candidate_count() of its heaviest experts
 * to a group: a = the finest level of at least ceil(mass / target) ranks, or a
 * coarser one, whose best group has room; c = the finest level whose best
 * group takes the whole excess, else the one with the most room.  A coarser c
 * that takes the excess wins only if a needs more helper slots (its size plus
 * two per follow-up export).  A level's best group has the most room, then the
 * lowest index.  The plan goes to `out` in planner_warp_ints() layout and
 * *max_load gets the highest rank load left.  NarrowKeys packs a load or room
 * with a 5-bit index. */
template <bool NarrowKeys, int Depth>
__device__ __forceinline__ int plan_halo_q_impl(int const* histogram, int lane_load, int target, int const* candidates,
    int* out, int ep, int experts, int helpers, bool dense, int* max_load)
{
    constexpr unsigned mask = 0xffffffffu;
    int const lane = threadIdx.x & 31;
    bool const is_rank = lane < ep;
    int const count = candidate_count(ep, experts, helpers);
    int const plan_capacity = broadcast_capacity(ep, experts, helpers);
    int const pair_level = pair_level_of(ep);
    auto const group_size = [ep, pair_level](int level) { return level == pair_level ? 2 : ep >> level; };
    int* p_expert = out + kPlanExpert * plan_capacity;
    int* p_level = out + kPlanLevel * plan_capacity;
    int* p_begin = out + kPlanGroupBegin * plan_capacity;
    int* p_size = out + kPlanGroupSize * plan_capacity;
    int* p_owner = out + kPlanOwner * plan_capacity;
    int* p_rows = out + 5 * plan_capacity; // Dense: quota row b at b * ep.
    int* p_start = out + 5 * plan_capacity;
    int* p_entries = p_start + plan_capacity + 1;
    if (lane == 0 && !dense)
        p_start[0] = 0;
    int entries = 0;

    /* The lane's first kHeldCandidates candidates and their masses stay in
     * registers; a later one is read when it comes up. */
    int held_expert[kHeldCandidates];
    int held_mass[kHeldCandidates];
#pragma unroll
    for (int k = 0; k < kHeldCandidates; ++k)
    {
        held_expert[k] = is_rank && k < count ? candidates[lane * count + k] : 0;
        held_mass[k] = is_rank && k < count ? histogram[held_expert[k]] : 0;
    }
    int lane_expert = held_expert[0];
    int lane_mass = held_mass[0];

    /* Per multicast level, the lane's group: ranks, index and room. */
    unsigned group_ranks[Depth];
    int group_index[Depth];
    int group_room[Depth];
    long long capacity[Depth];
    {
        int prefix = is_rank ? imax(0, target - lane_load) : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            int const other = __shfl_up_sync(mask, prefix, offset);
            prefix += lane >= offset ? other : 0;
        }
#pragma unroll
        for (int level = 0; level < Depth; ++level)
        {
            int const size = group_size(level);
            int index;
            if (size & (size - 1)) // Warp-uniform: only EP 6, 12, 24 have other sizes.
                index = lane / size;
            else
                index = lane >> (__ffs(size) - 1);
            int const begin = index * size;
            int const last = __shfl_sync(mask, prefix, imin(begin + size - 1, 31));
            int const before = __shfl_sync(mask, prefix, imax(begin - 1, 0));
            group_ranks[level] = is_rank ? rank_group_mask(begin, size) : 0u;
            group_index[level] = index;
            group_room[level] = is_rank ? last - (begin > 0 ? before : 0) : 0;
            capacity[level] = static_cast<long long>(size) * target;
        }
    }
    __syncwarp();

    int lane_recv = 0;
    int lane_send = 0;
    /* An owner freezes as soon as it cannot export again (no candidate left,
     * or the next one has no tokens), not when it next comes up. */
    bool lane_frozen = !(count > 0 && lane_mass > 0);
    int broadcasts = 0;
    /* Each level's best group, which does not depend on the owner, changes only
     * with a broadcast.  A group is open only if every member, the owner too,
     * has a free slot.  With no owner left, or no room on any level (every
     * remaining owner would freeze), the plan is complete. */
    int level_room[Depth];
    int level_group[Depth];
    int any_room = 0;
    bool stale = true;
    for (int guard = 0; guard < ep * (count + 2); ++guard)
    {
        bool const eligible = is_rank && !lane_frozen && lane_load > target;
        int owner, owner_load;
        if constexpr (NarrowKeys)
        {
            unsigned const key = __reduce_max_sync(
                mask, eligible ? ((static_cast<unsigned>(lane_load) << 5) | (31u - static_cast<unsigned>(lane))) : 0u);
            owner = key ? 31 - static_cast<int>(key & 31u) : -1;
            owner_load = static_cast<int>(key >> 5);
        }
        else
        {
            owner_load = __reduce_max_sync(mask, eligible ? lane_load : -1);
            owner = owner_load < 0 ? -1 : __reduce_min_sync(mask, eligible && lane_load == owner_load ? lane : INT_MAX);
        }
        if (stale)
        {
            unsigned const full = __ballot_sync(mask, is_rank && lane_recv >= helpers);
            any_room = 0;
#pragma unroll
            for (int level = 0; level < Depth; ++level)
            {
                int const room = group_room[level];
                bool const valid = is_rank && room > 0 && !(full & group_ranks[level]);
                if constexpr (NarrowKeys)
                {
                    unsigned const key = __reduce_max_sync(mask,
                        valid ? ((static_cast<unsigned>(room) << 5) | (31u - static_cast<unsigned>(group_index[level])))
                              : 0u);
                    level_room[level] = static_cast<int>(key >> 5);
                    level_group[level] = 31 - static_cast<int>(key & 31u);
                }
                else
                {
                    level_room[level] = __reduce_max_sync(mask, valid ? room : 0);
                    level_group[level]
                        = __reduce_min_sync(mask, valid && room == level_room[level] ? group_index[level] : INT_MAX)
                        & 31;
                }
                any_room |= level_room[level];
            }
            stale = false;
        }
        if (owner < 0 || !any_room)
            break;
        int const expert = __shfl_sync(mask, lane_expert, owner);
        int const mass = __shfl_sync(mask, lane_mass, owner);
        int const owner_send = __shfl_sync(mask, lane_send, owner);
        int const give = imin(owner_load - target, mass);
        if (mass <= 0 || give <= 0)
        {
            lane_frozen |= lane == owner;
            continue;
        }
        if (broadcasts >= plan_capacity)
            break;

        int want = 0;
#pragma unroll
        for (int level = 0; level < Depth; ++level)
            want = capacity[level] >= mass ? level : want;
        int a = -1, room_a = 0, group_a = 0;
        int c = -1, group_c = 0;
        int most = -1, room_most = 0, group_most = 0;
#pragma unroll
        for (int level = Depth - 1; level >= 0; --level)
        {
            int const room = level_room[level];
            int const group = level_group[level];
            bool const take_a = a < 0 && level <= want && room > 0;
            bool const take_c = c < 0 && room >= give;
            bool const take_most = room > 0 && (most < 0 || room > room_most);
            room_a = take_a ? room : room_a;
            group_a = take_a ? group : group_a;
            a = take_a ? level : a;
            group_c = take_c ? group : group_c;
            c = take_c ? level : c;
            room_most = take_most ? room : room_most;
            group_most = take_most ? group : group_most;
            most = take_most ? level : most;
        }
        bool const whole = c >= 0;
        group_c = whole ? group_c : group_most;
        c = whole ? c : most;
        int pick = c >= 0 ? c : a;
        int pick_group = c >= 0 ? group_c : group_a;
        if (a >= 0 && whole && c < a)
        {
            /* Lane j: the owner's next j + 1 candidates' mass. */
            int const next = owner_send + 1 + lane;
            int sum = next < count ? histogram[candidates[owner * count + next]] : 0;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1)
            {
                int const other = __shfl_up_sync(mask, sum, offset);
                sum += lane >= offset ? other : 0;
            }
            sum = next < count ? sum : -1;
            int const excess = owner_load - target;
            int const after_a = excess - room_a;
            int const after_c = excess - give;
            unsigned const cover_a = __ballot_sync(mask, sum >= 0 && sum >= after_a);
            unsigned const cover_c = __ballot_sync(mask, sum >= 0 && sum >= after_c);
            int const follow_a = after_a <= 0 ? 0 : cover_a ? __ffs(cover_a) : -1;
            int const follow_c = after_c <= 0 ? 0 : cover_c ? __ffs(cover_c) : -1;
            if (follow_a >= 0 && (follow_c < 0 || group_size(a) + 2 * follow_a <= group_size(c) + 2 * follow_c))
            {
                pick = a;
                pick_group = group_a;
            }
        }
        if (pick < 0)
        {
            lane_frozen |= lane == owner;
            continue;
        }
        int const size = group_size(pick);
        int const begin = pick_group * size;
        bool const in_group = lane >= begin && lane < begin + size;

        /* Emptiest member first, the lower rank on a tie; members come in
         * rising load, so the first one without room ends the spread. */
        bool const member = is_rank && in_group && lane != owner;
        int lane_quota = lane == owner ? mass : 0;
        int remaining = give;
        auto const take = [&](int destination, int amount)
        {
            lane_quota += lane == destination ? amount : lane == owner ? -amount : 0;
            remaining -= amount;
#pragma unroll
            for (int level = 0; level < Depth; ++level)
                group_room[level] -= (group_ranks[level] >> destination) & 1u ? amount : 0;
        };
        if (size == 2)
        {
            /* The same order with the pair's two loads in hand: the emptier
             * member, the lower rank on a tie, then the other one. */
            int const load_low = __shfl_sync(mask, lane_load, begin);
            int const load_high = __shfl_sync(mask, lane_load, begin + 1);
            bool const low_member = begin != owner;
            bool const high_member = begin + 1 != owner;
            bool const low_first = low_member && (!high_member || load_low <= load_high);
            int const first = low_first ? begin : begin + 1;
            int const room_first = target - (low_first ? load_low : load_high);
            if (room_first > 0)
            {
                take(first, imin(room_first, remaining));
                int const room_second = target - (low_first ? load_high : load_low);
                if (remaining > 0 && low_member && high_member && room_second > 0)
                    take(low_first ? begin + 1 : begin, imin(room_second, remaining));
            }
        }
        unsigned visited = 0;
        for (int spread = 0; size > 2 && spread < size && remaining > 0; ++spread)
        {
            bool const can_receive = member && !((visited >> lane) & 1u);
            int destination, destination_load;
            if constexpr (NarrowKeys)
            {
                unsigned const key = __reduce_min_sync(mask,
                    can_receive ? ((static_cast<unsigned>(lane_load) << 5) | static_cast<unsigned>(lane))
                                : 0xffffffffu);
                if (key == 0xffffffffu)
                    break;
                destination = static_cast<int>(key & 31u);
                destination_load = static_cast<int>(key >> 5);
            }
            else
            {
                destination_load = __reduce_min_sync(mask, can_receive ? lane_load : INT_MAX);
                destination = __reduce_min_sync(mask, can_receive && lane_load == destination_load ? lane : INT_MAX);
                if (destination == INT_MAX)
                    break;
            }
            visited |= 1u << destination;
            int const room = target - destination_load;
            if (room <= 0)
                break;
            take(destination, imin(room, remaining));
        }
        int const moved = give - remaining;
        if (moved <= 0)
        {
            lane_frozen |= lane == owner;
            continue;
        }

        int const b = broadcasts++;
        if (lane == 0)
        {
            p_expert[b] = expert;
            p_level[b] = pick;
            p_begin[b] = begin;
            p_size[b] = size;
            p_owner[b] = owner;
        }
        if (dense) // Warp-uniform.
        {
            if (is_rank)
                p_rows[b * ep + lane] = lane_quota;
        }
        else
        {
            unsigned const receivers = __ballot_sync(mask, is_rank && lane != owner && lane_quota > 0);
            if (lane == 0)
                p_start[b + 1] = entries + __popc(receivers);
            if ((receivers >> lane) & 1u)
            {
                int const entry = entries + __popc(receivers & ((1u << lane) - 1u));
                p_entries[2 * entry] = lane;
                p_entries[2 * entry + 1] = lane_quota;
            }
            entries += __popc(receivers);
        }
        lane_recv += is_rank && in_group ? 1 : 0;
        stale = true;
        /* Branch-free: the owner sheds what moved and steps to its next
         * candidate, a receiver gains its tokens. */
        bool const is_owner = lane == owner;
        lane_load += is_owner ? -moved : imax(lane_quota, 0);
        lane_send += is_owner ? 1 : 0;
        int next_expert = 0, next_mass = 0;
#pragma unroll
        for (int k = 1; k < kHeldCandidates; ++k)
        {
            next_expert = lane_send == k ? held_expert[k] : next_expert;
            next_mass = lane_send == k ? held_mass[k] : next_mass;
        }
        if (is_owner && lane_send >= kHeldCandidates && lane_send < count)
        {
            next_expert = candidates[lane * count + lane_send];
            next_mass = histogram[next_expert];
        }
        lane_expert = is_owner ? next_expert : lane_expert;
        lane_mass = is_owner ? next_mass : lane_mass;
        lane_frozen = is_owner ? next_mass <= 0 : lane_frozen;
    }
    *max_load = __reduce_max_sync(mask, is_rank ? lane_load : 0);
    return broadcasts;
}

/* plan_halo_q_impl's plans with 5-bit-index keys and at most kHeldCandidates
 * candidates per owner.  Lane j also holds the j-th group of the levels in
 * order (at most 31 groups, EP32: 1 + 2 + 4 + 8 + 16), so each step's owner and
 * best groups come from independent reductions.  kEp, kS > 0: the shape at
 * compile time (EP = 2^Depth, S slots, experts >= EP * S).  With direct_plan
 * (warp 0) the plan goes straight to plan[] and the dense quota rows instead of
 * `out`. */
template <int Depth, int kEp = 0, int kS = 0>
__device__ __forceinline__ int plan_halo_q_group_lanes(int const* histogram, int lane_load, int target,
    int const* candidates, int* out, int ep_arg, int experts, int helpers_arg, bool dense_arg, int* max_load,
    int* direct_plan, int* direct_quota, int const (&in_expert)[kHeldCandidates], int const (&in_mass)[kHeldCandidates])
{
    // Compile-time shape when the caller has one (kEp, kS > 0): EP = 2^Depth and S slots.
    int const ep = kEp > 0 ? kEp : ep_arg;
    int const helpers = kS > 0 ? kS : helpers_arg;
    constexpr int kHeld = kS > 0 ? kS : kHeldCandidates;
    // Specialized shapes always have four targets and a direct warp 0: dense iff three dense plans fit.
    bool const dense = kS > 0 ? (kPlannerWarps - 1) * (5 + kEp) * kEp * kS <= kPlannerScratchInts : dense_arg;
    constexpr unsigned mask = 0xffffffffu;
    constexpr unsigned kNoKey = 0xffffffffu;
    int const lane = threadIdx.x & 31;
    bool const is_rank = lane < ep;
    int const count = kS > 0 ? kS : candidate_count(ep, experts, helpers);
    int const plan_capacity = kS > 0 ? kEp * kS : broadcast_capacity(ep, experts, helpers);
    int const pair_level = pair_level_of(ep);
    auto const group_size = [ep, pair_level](int level) { return level == pair_level ? 2 : ep >> level; };
    int* p_expert = out + kPlanExpert * plan_capacity;
    int* p_level = out + kPlanLevel * plan_capacity;
    int* p_begin = out + kPlanGroupBegin * plan_capacity;
    int* p_size = out + kPlanGroupSize * plan_capacity;
    int* p_owner = out + kPlanOwner * plan_capacity;
    bool const direct = direct_plan != nullptr; // Warp-uniform.
    if (direct)
    {
        p_expert = plan_field(direct_plan, kPlanExpert);
        p_level = plan_field(direct_plan, kPlanLevel);
        p_begin = plan_field(direct_plan, kPlanGroupBegin);
        p_size = plan_field(direct_plan, kPlanGroupSize);
        p_owner = plan_field(direct_plan, kPlanOwner);
    }
    int* p_rows = direct ? direct_quota : out + 5 * plan_capacity; // Dense: quota row b at b * ep.
    int* p_start = out + 5 * plan_capacity;
    int* p_entries = p_start + plan_capacity + 1;
    if (lane == 0 && !dense)
        p_start[0] = 0;
    int entries = 0;

    // The candidates as the target computation loaded them (zero past count and on non-rank lanes).
    int held_expert[kHeld];
    int held_mass[kHeld];
#pragma unroll
    for (int k = 0; k < kHeld; ++k)
    {
        held_expert[k] = in_expert[k];
        held_mass[k] = in_mass[k];
    }
    int lane_expert = held_expert[0];
    int lane_mass = held_mass[0];

    /* The lane's group: level (-1: none), index within the level, ranks and
     * room (the members' slack max(0, target - load)); per level, the mass
     * its groups hold at target load (below 2^32: narrow keys bound the
     * target by 2^27). */
    int g_level = -1;
    int g_index = 0;
    int g_begin = 0;
    int g_size = 1;
    unsigned capacity[Depth];
    {
        int first_lane = 0;
#pragma unroll
        for (int level = 0; level < Depth; ++level)
        {
            int const size = group_size(level);
            int const groups = level == pair_level ? ep >> 1 : 1 << level; // ep / size
            bool const mine = lane >= first_lane && lane < first_lane + groups;
            g_level = mine ? level : g_level;
            g_index = mine ? lane - first_lane : g_index;
            g_size = mine ? size : g_size;
            first_lane += groups;
            capacity[level] = static_cast<unsigned>(size) * static_cast<unsigned>(target);
        }
        g_begin = g_index * g_size;
    }
    unsigned const g_ranks = g_level >= 0 ? rank_group_mask(g_begin, g_size) : 0u;
    int g_room;
    {
        int prefix = is_rank ? imax(0, target - lane_load) : 0;
        // Groups read the prefix of rank lanes only: offsets below EP suffice.
#pragma unroll
        for (int offset = 1; offset < (kEp > 0 ? kEp : 32); offset <<= 1)
        {
            int const other = __shfl_up_sync(mask, prefix, offset);
            prefix += lane >= offset ? other : 0;
        }
        int const last = __shfl_sync(mask, prefix, imin(g_begin + g_size - 1, 31));
        int const before = __shfl_sync(mask, prefix, imax(g_begin - 1, 0));
        g_room = g_level >= 0 ? last - (g_begin > 0 ? before : 0) : 0;
    }
    __syncwarp();

    int lane_recv = 0;
    int lane_send = 0;
    bool lane_frozen = !(count > 0 && lane_mass > 0);
    int broadcasts = 0;
    /* Per lane, refreshed when it exports: want, the finest level whose
     * groups hold its candidate's mass at target load, and the mass of its
     * next one, two and three candidates after it (-1 past the last). */
    auto const want_of = [&](int mass)
    {
        unsigned fits = 0;
#pragma unroll
        for (int level = 0; level < Depth; ++level)
            fits |= capacity[level] >= static_cast<unsigned>(mass) ? 1u << level : 0u;
        return fits ? 31 - __clz(fits) : 0;
    };
    // want of each held candidate (a frozen lane's want is never used).
    int held_want[kHeld];
#pragma unroll
    for (int k = 0; k < kHeld; ++k)
        held_want[k] = want_of(held_mass[k]);
    int lane_want = held_want[0];
    int lane_follow[kHeld - 1];
    auto const refresh_follow = [&](int send, int(&follow)[kHeld - 1])
    {
        int sum = 0;
#pragma unroll
        for (int k = 1; k < kHeld; ++k)
        {
            int const index = send + k;
            int next = 0;
#pragma unroll
            for (int h = 1; h < kHeld; ++h)
                next = index == h ? held_mass[h] : next;
            sum += next;
            follow[k - 1] = index < count ? sum : -1;
        }
    };
    refresh_follow(0, lane_follow);
    // The follow-up sums after each export count: an export selects its row.
    int follow_after[kHeld][kHeld - 1];
#pragma unroll
    for (int send = 1; send < kHeld; ++send)
        refresh_follow(send, follow_after[send]);
    for (int guard = 0; guard < ep * (count + 2); ++guard)
    {
        bool const eligible = is_rank && !lane_frozen && lane_load > target;
        unsigned const owner_key = __reduce_max_sync(
            mask, eligible ? ((static_cast<unsigned>(lane_load) << 5) | (31u - static_cast<unsigned>(lane))) : 0u);
        /* Each level's best group: open (every member, the owner too, has a
         * free slot) with the most room, then the lowest index. */
        unsigned const full = __ballot_sync(mask, is_rank && lane_recv >= helpers);
        bool const valid = g_level >= 0 && g_room > 0 && !(full & g_ranks);
        unsigned const g_key
            = valid ? ((static_cast<unsigned>(g_room) << 5) | (31u - static_cast<unsigned>(g_index))) : 0u;
        int level_room[Depth];
        unsigned level_groups = 0; // 4 bits a level: at most 16 groups.
        unsigned open = 0;
#pragma unroll
        for (int level = 0; level < Depth; ++level)
        {
            unsigned const key = __reduce_max_sync(mask, g_level == level ? g_key : 0u);
            level_room[level] = static_cast<int>(key >> 5);
            level_groups |= ((31u - (key & 31u)) & 15u) << (4 * level);
            open |= key ? 1u << level : 0u;
        }
        if (!owner_key || !open)
            break;

        /* The lane's level as the owner, plus one (0: none).  a: the finest
         * level up to want (the finest whose groups hold the mass at target)
         * with room; c: the finest level whose best group takes the whole
         * give, else the level with the most room (the finer on a tie); a
         * replaces a coarser whole c unless it needs more helper slots. */
        int most;
        {
            // The most room, the finer level on a tie; none when every level is closed.
            unsigned best = 0;
#pragma unroll
            for (int level = 0; level < Depth; ++level)
                best = max(best, (static_cast<unsigned>(level_room[level]) << 3) | static_cast<unsigned>(level));
            most = best >> 3 ? static_cast<int>(best & 7u) : -1;
        }
        int lane_choice;
        {
            int const give = imin(lane_load - target, lane_mass);
            unsigned whole = 0;
#pragma unroll
            for (int level = 0; level < Depth; ++level)
                whole |= level_room[level] >= give ? 1u << level : 0u;
            int const want = lane_want;
            unsigned const reach = open & ((2u << want) - 1u);
            int const a = reach ? 31 - __clz(static_cast<int>(reach)) : -1;
            int const c = whole ? 31 - __clz(static_cast<int>(whole)) : -1;
            int pick = c >= 0 ? c : most >= 0 ? most : a;
            {
                /* Branch-free: every lane evaluates the slot cost; only a
                 * lane whose whole c is coarser than its a uses it. */
                int room_a = 0;
#pragma unroll
                for (int level = 0; level < Depth; ++level)
                    room_a = level == a ? level_room[level] : room_a;
                int const size_a = group_size(imax(a, 0));
                int const size_c = group_size(imax(c, 0));
                /* Follow-up exports the owner's next candidates need to cover
                 * the excess each choice leaves. */
                int const excess = lane_load - target;
                int const after_a = excess - room_a;
                int const after_c = excess - give;
                int follow_a = after_a <= 0 ? 0 : -1;
                int follow_c = after_c <= 0 ? 0 : -1;
#pragma unroll
                for (int k = 1; k < kHeld; ++k)
                {
                    follow_a = follow_a < 0 && lane_follow[k - 1] >= after_a ? k : follow_a;
                    follow_c = follow_c < 0 && lane_follow[k - 1] >= after_c ? k : follow_c;
                }
                bool const use_a = a >= 0 && c >= 0 && c < a && follow_a >= 0
                    && (follow_c < 0 || size_a + 2 * follow_a <= size_c + 2 * follow_c);
                pick = use_a ? a : pick;
            }
            lane_choice = pick + 1;
        }

        int const owner = 31 - static_cast<int>(owner_key & 31u);
        int const owner_load = static_cast<int>(owner_key >> 5);
        int const choice = __shfl_sync(mask, lane_choice, owner);
        int const expert = __shfl_sync(mask, lane_expert, owner);
        int const mass = __shfl_sync(mask, lane_mass, owner);
        int const give = imin(owner_load - target, mass);
        int const pick = choice - 1;
        /* The export is worked out before the rare exits are taken; nothing
         * changes state until then.  The chosen group's two emptiest ranks
         * (rising load, the lower rank on a tie), the owner among them if
         * inside; the first one without room (an owner inside, above target,
         * has none) ends the spread. */
        int const pick_level = imax(pick, 0);
        int const pick_group = static_cast<int>((level_groups >> (4 * pick_level)) & 15u);
        int const size = group_size(pick_level);
        int const begin = pick_group * size;
        bool const in_group = lane >= begin && lane < begin + size;
        unsigned const lane_key = (static_cast<unsigned>(lane_load) << 5) | static_cast<unsigned>(lane);
        bool const in_pick = is_rank && in_group;
        unsigned const first = __reduce_min_sync(mask, in_pick ? lane_key : kNoKey);
        unsigned const second = __reduce_min_sync(mask, in_pick && lane_key != first ? lane_key : kNoKey);
        int const first_rank = static_cast<int>(first & 31u);
        int const first_room = first == kNoKey ? 0 : target - static_cast<int>(first >> 5);
        int const first_amount = first_room > 0 ? imin(first_room, give) : 0;
        int remaining = give - first_amount;
        int const second_room = second == kNoKey ? 0 : target - static_cast<int>(second >> 5);
        bool const takes_second = remaining > 0 && first_amount > 0 && second_room > 0;
        int const second_rank = takes_second ? static_cast<int>(second & 31u) : -1;
        int const second_amount = takes_second ? imin(second_room, remaining) : 0;
        remaining -= second_amount;
        bool const ready = mass > 0 && give > 0;
        if (ready && broadcasts >= plan_capacity)
            break;
        if (!ready || pick < 0)
        {
            lane_frozen |= lane == owner;
            continue;
        }
        int lane_quota = (lane == first_rank ? first_amount : 0) + (lane == second_rank ? second_amount : 0);
        g_room -= ((g_ranks >> first_rank) & 1u ? first_amount : 0)
            + (second_rank >= 0 && ((g_ranks >> second_rank) & 1u) ? second_amount : 0);
        if (size > 2 && remaining > 0 && second_amount > 0)
        {
            /* A third receiver and on: groups of 4 or more only. */
            unsigned visited = (1u << first_rank) | (1u << second_rank);
            for (int spread = 2; spread < size && remaining > 0; ++spread)
            {
                bool const can_receive = is_rank && in_group && lane != owner && !((visited >> lane) & 1u);
                unsigned const key = __reduce_min_sync(mask, can_receive ? lane_key : kNoKey);
                if (key == kNoKey)
                    break;
                int const destination = static_cast<int>(key & 31u);
                visited |= 1u << destination;
                int const room = target - static_cast<int>(key >> 5);
                if (room <= 0)
                    break;
                int const amount = imin(room, remaining);
                lane_quota += lane == destination ? amount : 0;
                remaining -= amount;
                g_room -= (g_ranks >> destination) & 1u ? amount : 0;
            }
        }
        int const moved = give - remaining;
        if (moved <= 0)
        {
            lane_frozen |= lane == owner;
            continue;
        }
        lane_quota = lane == owner ? mass - moved : lane_quota;

        int const b = broadcasts++;
        if (lane == 0)
        {
            p_expert[b] = expert;
            p_level[b] = pick;
            p_begin[b] = begin;
            p_size[b] = size;
            p_owner[b] = owner;
            if (direct)
                plan_field(direct_plan, kPlanHelper)[b] = -1;
        }
        if (dense || direct) // Warp-uniform.
        {
            if (is_rank)
                p_rows[b * ep + lane] = lane_quota;
        }
        else
        {
            unsigned const receivers = __ballot_sync(mask, is_rank && lane != owner && lane_quota > 0);
            if (lane == 0)
                p_start[b + 1] = entries + __popc(receivers);
            if ((receivers >> lane) & 1u)
            {
                int const entry = entries + __popc(receivers & ((1u << lane) - 1u));
                p_entries[2 * entry] = lane;
                p_entries[2 * entry + 1] = lane_quota;
            }
            entries += __popc(receivers);
        }
        lane_recv += is_rank && in_group ? 1 : 0;
        bool const is_owner = lane == owner;
        lane_load += is_owner ? -moved : imax(lane_quota, 0);
        lane_send += is_owner ? 1 : 0;
        int next_expert = 0, next_mass = 0;
#pragma unroll
        for (int k = 1; k < kHeld; ++k)
        {
            next_expert = lane_send == k ? held_expert[k] : next_expert;
            next_mass = lane_send == k ? held_mass[k] : next_mass;
        }
        lane_expert = is_owner ? next_expert : lane_expert;
        lane_mass = is_owner ? next_mass : lane_mass;
        lane_frozen = is_owner ? next_mass <= 0 : lane_frozen;
        int next_want = 0;
#pragma unroll
        for (int k = 1; k < kHeld; ++k)
            next_want = lane_send == k ? held_want[k] : next_want;
        lane_want = is_owner ? next_want : lane_want;
#pragma unroll
        for (int k = 0; k < kHeld - 1; ++k)
        {
            int next_follow = -1; // past the held candidates: frozen
#pragma unroll
            for (int send = 1; send < kHeld; ++send)
                next_follow = lane_send == send ? follow_after[send][k] : next_follow;
            lane_follow[k] = is_owner ? next_follow : lane_follow[k];
        }
    }
    *max_load = __reduce_max_sync(mask, is_rank ? lane_load : 0);
    return broadcasts;
}

/* Flag in a target's result: its plan is already in plan[] and the quota rows. */
constexpr int kDirectPlan = 1 << 30;

/* Plan layouts in the scratch: compact, dense (slot = warp), or dense behind a direct warp 0 (slot = warp - 1). */
constexpr int kLayoutCompact = 0;
constexpr int kLayoutDense = 1;
constexpr int kLayoutSpare = 2;

/* Placement by warps 0 .. targets - 1, one target each: each leaves its plan in
 * scratch and (highest rank load, broadcasts) in results; with
 * plan_halo_q_group_lanes, warp 0 writes its plan to plan[] and quota itself.
 * 5-bit-index keys need a total below 2^26.  Returns the layout used. */
template <int kEp = 0, int kS = 0>
__device__ int plan_halo_q(int const* histogram, int const* load, int const* candidates, int* scratch, int* results,
    int* quota, int* plan, int ep_arg, int experts, int helpers_arg, int targets)
{
    constexpr int kShapeDepth = log2_pow2(kEp);
    constexpr bool kDense4 = kEp > 0 && kPlannerWarps * (5 + kEp) * kEp * kS <= kPlannerScratchInts;
    int const ep = kEp > 0 ? kEp : ep_arg;
    int const helpers = kS > 0 ? kS : helpers_arg;
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const count = kS > 0 ? kS : candidate_count(ep, experts, helpers);
    int const lane_load = lane < ep ? load[lane] : 0;
    int const total = __reduce_add_sync(0xffffffffu, lane_load);
    int held = 0;
    int held_expert[kHeldCandidates];
    int held_mass[kHeldCandidates];
#pragma unroll
    for (int k = 0; k < kHeldCandidates; ++k)
        held_expert[k] = held_mass[k] = 0;
    if (count <= kHeldCandidates)
    {
#pragma unroll
        for (int k = 0; k < kHeldCandidates; ++k)
        {
            held_expert[k] = lane < ep && k < count ? candidates[lane * count + k] : 0;
            held_mass[k] = lane < ep && k < count ? histogram[held_expert[k]] : 0;
            held += held_mass[k];
        }
    }
    else
    {
        for (int k = 0; lane < ep && k < count; ++k)
            held += histogram[candidates[lane * count + k]];
    }
    int const shed = lane < ep ? lane_load - held : 0;
    int const mean = kEp > 0 ? total >> kShapeDepth : div_by_ep(total, ep);
    int const floor_load = imax(mean + (total != mean * ep), __reduce_max_sync(0xffffffffu, shed));
    int const margin = 400 + target_margin_400(warp);
    unsigned const divisor = 400u * static_cast<unsigned>(ep);
    // 32-bit when total * 412 + divisor fits; the 64-bit division is slow.  At power-of-two EP,
    // ceil(a / (25 * 16 * ep)) = ceil(ceil(a / (16 * ep)) / 25): a shift and a division by a constant.
    int share;
    if (!(ep & (ep - 1)) && total <= 0x00980000)
    {
        int const shift = kEp > 0 ? 4 + kShapeDepth : 3 + __ffs(ep);
        unsigned const scaled = (static_cast<unsigned>(total) * margin + (1u << shift) - 1u) >> shift;
        share = static_cast<int>((scaled + 24u) / 25u);
    }
    else
        share = total <= 0x00980000
            ? static_cast<int>((static_cast<unsigned>(total) * margin + divisor - 1) / divisor)
            : static_cast<int>((static_cast<long long>(total) * margin + divisor - 1) / divisor);
    int const target = imax(floor_load, share);
    int const dense_ints = kEp > 0 ? (5 + kEp) * kEp * kS : planner_dense_ints(ep, experts, helpers);
    int* out = scratch + warp * (kDense4 ? dense_ints : planner_warp_ints(ep, experts, helpers));
    bool dense = kEp > 0 ? kDense4 : planner_dense(ep, experts, helpers);
    int layout = dense ? kLayoutDense : kLayoutCompact;
    if (!dense && total <= 0x03ffffff && count <= kHeldCandidates && (targets - 1) * dense_ints <= kPlannerScratchInts)
    {
        dense = true;
        layout = kLayoutSpare;
        out = scratch + imax(warp - 1, 0) * dense_ints;
    }
    int max_load = 0;
    int broadcasts = 0;
    bool direct = false;
    auto const place = [&](auto narrow, auto depth)
    {
        if constexpr (decltype(narrow)::value)
        {
            if (count <= kHeldCandidates)
            {
                constexpr int kDepth = decltype(depth)::value;
                direct = warp == 0;
                int* const direct_plan = direct ? plan : nullptr;
                int* const direct_quota = direct ? quota : nullptr;
                if constexpr (kEp > 0)
                    broadcasts = plan_halo_q_group_lanes<kDepth, kEp, kS>(histogram, lane_load, target, candidates, out,
                        ep, experts, helpers, dense, &max_load, direct_plan, direct_quota, held_expert, held_mass);
                else if (ep == (1 << kDepth) && experts >= ep * helpers && helpers == 2)
                    broadcasts = plan_halo_q_group_lanes<kDepth, (1 << kDepth), 2>(histogram, lane_load, target,
                        candidates, out, ep, experts, helpers, dense, &max_load, direct_plan, direct_quota, held_expert,
                        held_mass);
                else if (ep == (1 << kDepth) && experts >= ep * helpers && helpers == 3)
                    broadcasts = plan_halo_q_group_lanes<kDepth, (1 << kDepth), 3>(histogram, lane_load, target,
                        candidates, out, ep, experts, helpers, dense, &max_load, direct_plan, direct_quota, held_expert,
                        held_mass);
                else
                    broadcasts = plan_halo_q_group_lanes<kDepth>(histogram, lane_load, target, candidates, out, ep,
                        experts, helpers, dense, &max_load, direct_plan, direct_quota, held_expert, held_mass);
                return;
            }
        }
        broadcasts = plan_halo_q_impl<decltype(narrow)::value, decltype(depth)::value>(
            histogram, lane_load, target, candidates, out, ep, experts, helpers, dense, &max_load);
    };
    auto const by_depth = [&](auto narrow)
    {
        if constexpr (kEp > 0)
            place(narrow, std::integral_constant<int, kShapeDepth>{});
        else
        {
            switch (planner_depth(ep))
            {
            case 1: place(narrow, std::integral_constant<int, 1>{}); break;
            case 2: place(narrow, std::integral_constant<int, 2>{}); break;
            case 3: place(narrow, std::integral_constant<int, 3>{}); break;
            case 4: place(narrow, std::integral_constant<int, 4>{}); break;
            default: place(narrow, std::integral_constant<int, kMaxLevels>{}); break;
            }
        }
    };
    if (total <= 0x03ffffff)
        by_depth(std::true_type{});
    else
        by_depth(std::false_type{});
    if (lane == 0)
    {
        results[2 * warp] = max_load;
        results[2 * warp + 1] = broadcasts | (direct ? kDirectPlan : 0);
    }
    return layout;
}

/* The winning target's plan (lowest highest rank load, the lower target on a
 * tie) into plan[] and dense quota rows, by the whole CTA: an owner keeps the
 * expert's mass less what its receivers take. */
template <int kEp = 0, int kS = 0>
__device__ int adopt_best_plan(int const* histogram, int const* scratch, int const* results, int* quota, int* plan,
    int ep_arg, int experts, int helpers_arg, int threads, int targets, int layout)
{
    int const ep = kEp > 0 ? kEp : ep_arg;
    int const helpers = kS > 0 ? kS : helpers_arg;
    int const plan_capacity = kEp > 0 ? kEp * kS : broadcast_capacity(ep, experts, helpers);
    // Independent reads; the strict < keeps the lower target on a tie.
    int best = 0;
    int best_max = results[0];
#pragma unroll
    for (int warp = 1; warp < kPlannerWarps; ++warp)
    {
        int const value = warp < targets ? results[2 * warp] : best_max;
        best = value < best_max ? warp : best;
        best_max = value < best_max ? value : best_max;
    }
    int const meta = results[2 * best + 1];
    int const broadcasts = meta & (kDirectPlan - 1);
    if (!(meta & kDirectPlan))
    {
        if (layout != kLayoutCompact)
        {
            // Dense slots of dense_ints: slot best, or best - 1 behind a direct warp 0.
            int const* out = scratch
                + (best - (layout == kLayoutSpare ? 1 : 0))
                    * (kEp > 0 ? (5 + kEp) * kEp * kS : planner_dense_ints(ep, experts, helpers));
            int const* rows = out + 5 * plan_capacity;
            for (int index = threadIdx.x; index < broadcasts * ep; index += threads)
                quota[index] = rows[index];
            for (int b = threadIdx.x; b < broadcasts; b += threads)
            {
#pragma unroll
                for (int field = 0; field <= kPlanHelper; ++field)
                    plan_field(plan, field)[b] = field == kPlanHelper ? -1 : out[field * plan_capacity + b];
            }
        }
        else
        {
            /* One thread per broadcast: its fields, its quota row cleared, then the receivers' tokens and the
             * owner's mass less what they take. */
            int const* out = scratch + best * planner_warp_ints(ep, experts, helpers);
            int const* start = out + 5 * plan_capacity;
            int const* entries = start + plan_capacity + 1;
            for (int b = threadIdx.x; b < broadcasts; b += threads)
            {
                int const expert = out[kPlanExpert * plan_capacity + b];
                int const owner = out[kPlanOwner * plan_capacity + b];
                int const first = start[b];
                int const last = start[b + 1];
#pragma unroll
                for (int field = 0; field <= kPlanHelper; ++field)
                    plan_field(plan, field)[b] = field == kPlanHelper ? -1 : out[field * plan_capacity + b];
                int* const row = quota + b * ep;
                for (int rank = 0; rank < ep; ++rank)
                    row[rank] = 0;
                int moved = 0;
                for (int entry = first; entry < last; ++entry)
                {
                    int const tokens = entries[2 * entry + 1];
                    row[entries[2 * entry]] = tokens;
                    moved += tokens;
                }
                row[owner] = histogram[expert] - moved;
            }
        }
    }
    if (threadIdx.x == 0)
        plan[1] = broadcasts;
    return broadcasts;
}

/* HALO-Q placement by the whole planner CTA after prepare: the planner warps
 * plan toward their targets, then the best plan is adopted.  running[0..1]
 * holds the broadcast count and layout; the results and per-target scratch
 * follow it.  Returns the broadcast count.  With a
 * compile-time shape (kEp, kS > 0) see place_halo_q. */
template <int kEp = 0, int kS = 0>
__device__ int place_halo_q_shape(int const* histogram, int const* load, int* quota, int const* candidates,
    int* running, int* plan, int ep_arg, int experts, int helpers_arg)
{
    // Four dense plans, or three behind a direct warp 0 (planner_dense_spare).
    static_assert(kEp == 0 || (kPlannerWarps - 1) * (5 + kEp) * kEp * kS <= kPlannerScratchInts, "dense plans");
    constexpr bool kDense4 = kEp > 0 && kPlannerWarps * (5 + kEp) * kEp * kS <= kPlannerScratchInts;
    int const ep = kEp > 0 ? kEp : ep_arg;
    int const helpers = kS > 0 ? kS : helpers_arg;
    int* const results = running + 2;
    int* const scratch = results + 2 * kPlannerWarps;
    /* Prepare ends at a block barrier, or at the owner warps' named barrier,
     * which every planner warp joins when EP >= the target count. */
    int const targets = kEp > 0 ? kPlannerWarps : planner_targets(ep, experts, helpers);
    if (uses_register_prepare(ep, experts) && ep < targets)
        __syncthreads();
    /* Dense plans: the planner warps adopt the best one among themselves while
     * the other warps wait at the block barrier.  Compact plans expand faster
     * with the whole CTA.  When four dense plans do not fit,
     * the planner warps tell the others in running[1] whether theirs were
     * dense (planner_dense_spare). */
    bool const dense4 = kEp > 0 ? kDense4 : planner_dense(ep, experts, helpers);
    if (threadIdx.x < 32 * targets)
    {
        int const layout = plan_halo_q<kEp, kS>(
            histogram, load, candidates, scratch, results, quota, plan, ep, experts, helpers, targets);
        if (layout != kLayoutCompact)
        {
            asm volatile("bar.sync 5, %0;" ::"r"(32 * targets) : "memory");
            int const broadcasts = adopt_best_plan<kEp, kS>(
                histogram, scratch, results, quota, plan, ep, experts, helpers, 32 * targets, targets, layout);
            if (threadIdx.x == 0)
                running[0] = broadcasts;
        }
        if (!dense4 && threadIdx.x == 0)
            running[1] = layout;
    }
    __syncthreads();
    if (dense4 || running[1] != kLayoutCompact)
        return running[0];
    int const broadcasts = adopt_best_plan<kEp, kS>(
        histogram, scratch, results, quota, plan, ep, experts, helpers, blockDim.x, targets, kLayoutCompact);
    __syncthreads();
    return broadcasts;
}

__device__ int place_halo_q(int const* histogram, int const* load, int* quota, int const* candidates, int* running,
    int* plan, int ep, int experts, int helpers)
{
    // The shape at compile time where S candidates a rank fit: EP8/16/32 at S = 2 or 3.
    if (experts >= ep * helpers)
    {
        if (ep == 8 && helpers == 2)
            return place_halo_q_shape<8, 2>(histogram, load, quota, candidates, running, plan, ep, experts, helpers);
        if (ep == 8 && helpers == 3)
            return place_halo_q_shape<8, 3>(histogram, load, quota, candidates, running, plan, ep, experts, helpers);
        if (ep == 16 && helpers == 2)
            return place_halo_q_shape<16, 2>(histogram, load, quota, candidates, running, plan, ep, experts, helpers);
        if (ep == 16 && helpers == 3)
            return place_halo_q_shape<16, 3>(histogram, load, quota, candidates, running, plan, ep, experts, helpers);
        if (ep == 32 && helpers == 2)
            return place_halo_q_shape<32, 2>(histogram, load, quota, candidates, running, plan, ep, experts, helpers);
        if (ep == 32 && helpers == 3)
            return place_halo_q_shape<32, 3>(histogram, load, quota, candidates, running, plan, ep, experts, helpers);
    }
    return place_halo_q_shape(histogram, load, quota, candidates, running, plan, ep, experts, helpers);
}

__device__ __forceinline__ int warp_sum(int value)
{
    for (int offset = 16; offset; offset >>= 1)
    {
        int const other = __shfl_down_sync(0xffffffffu, value, offset);
        if ((threadIdx.x & 31) + offset < 32)
            value += other;
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ int warp_exclusive_scan(int value)
{
    int sum = value;
    int const lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1)
    {
        int other = __shfl_up_sync(0xffffffffu, sum, offset);
        if (lane >= offset)
            sum += other;
    }
    return sum - value;
}

/* Pair repair of more than 32 broadcasts by one warp, a tile of 32 at a time. */
__device__ void halo_q_repair(int receiver, int donor, int broadcasts, int const* order, int const* host_mask,
    int const* counts, int const* p_expert, int* quota, int ep, int experts)
{
    int const lane = threadIdx.x & 31;
    int local_pull = 0, local_give = 0;
    for (int base = 0; base < broadcasts; base += 32)
    {
        int const b = base + lane < broadcasts ? order[base + lane] : -1;
        if (b >= 0 && (host_mask[b] & (1u << receiver)) && (host_mask[b] & (1u << donor)))
        {
            int const expert = p_expert[b];
            int const cr = counts[receiver * experts + expert];
            int const cd = counts[donor * experts + expert];
            int const qr = quota[b * ep + receiver];
            int const qd = quota[b * ep + donor];
            local_pull += imin(imax(0, cr - qr), imax(0, qd - cd));
            local_give += imax(0, qr - cr);
        }
    }
    int const delta = imin(warp_sum(local_pull), warp_sum(local_give));
    int pull_before_tiles = 0, give_before_tiles = 0;
    for (int base = 0; base < broadcasts; base += 32)
    {
        int const b = base + lane < broadcasts ? order[base + lane] : -1;
        int pull = 0, give = 0;
        if (b >= 0 && (host_mask[b] & (1u << receiver)) && (host_mask[b] & (1u << donor)))
        {
            int const expert = p_expert[b];
            int const cr = counts[receiver * experts + expert];
            int const cd = counts[donor * experts + expert];
            int const qr = quota[b * ep + receiver];
            int const qd = quota[b * ep + donor];
            pull = imin(imax(0, cr - qr), imax(0, qd - cd));
            give = imax(0, qr - cr);
        }
        int const pull_before = pull_before_tiles + warp_exclusive_scan(pull);
        int const give_before = give_before_tiles + warp_exclusive_scan(give);
        int const take = imin(pull, imax(0, delta - pull_before));
        int const release = imin(give, imax(0, delta - give_before));
        pull_before_tiles += warp_sum(pull);
        give_before_tiles += warp_sum(give);
        if (b >= 0)
        {
            quota[b * ep + receiver] += take - release;
            quota[b * ep + donor] += release - take;
        }
        __syncwarp();
    }
}

/* Pair repair of the scheduled pairs of plans with at most 32 broadcasts, one
 * lane per broadcast. */
__device__ __forceinline__ void halo_q_direction(
    bool common, int receiver_count, int donor_count, int& receiver_quota, int& donor_quota)
{
    int pull = common ? imin(imax(0, receiver_count - receiver_quota), imax(0, donor_quota - donor_count)) : 0;
    int give = common ? imax(0, receiver_quota - receiver_count) : 0;
    int pull_total = __reduce_add_sync(0xffffffffu, pull);
    int give_total = __reduce_add_sync(0xffffffffu, give);
    int delta = imin(pull_total, give_total);
    if (!delta)
        return;
    int pull_before = warp_exclusive_scan(pull);
    int give_before = warp_exclusive_scan(give);
    int pull_take = imin(pull, imax(0, delta - pull_before));
    int give_take = imin(give, imax(0, delta - give_before));
    receiver_quota += pull_take - give_take;
    donor_quota += give_take - pull_take;
}

/* Fuse both ordered directions of one rank pair.  Each broadcast lane loads
 * its two count/quota entries once, executes left<-right then right<-left in
 * registers, and writes the final pair once.
 *
 * `b`, `hosts` and `expert` are the lane's own broadcast identity and do not
 * change between phases or rounds, so the caller hoists them.  Reading them
 * here would put a four-deep dependent load chain (order[lane] -> b -> host_mask[b]
 * and the GLOBAL p_expert[b] -> expert -> counts[...]) at the head of every
 * phase. */
__device__ __forceinline__ void halo_q_pair(
    int left, int right, int b, int hosts, int expert, int const* counts, int* quota, int ep, int experts)
{
    bool common = false;
    int count_left = 0;
    int count_right = 0;
    int quota_left = 0;
    int quota_right = 0;
    if (b >= 0)
    {
        common = (hosts & (1u << left)) && (hosts & (1u << right));
        if (common)
        {
            count_left = counts[left * experts + expert];
            count_right = counts[right * experts + expert];
            quota_left = quota[b * ep + left];
            quota_right = quota[b * ep + right];
        }
    }

    /* Most pairs move nothing: a direction moves quota only if the receiver is
     * short where the donor is over and over somewhere; the first leaves the
     * quotas as they are when it moves nothing. */
    bool const left_short = count_left > quota_left;
    bool const left_over = quota_left > count_left;
    bool const right_short = count_right > quota_right;
    bool const right_over = quota_right > count_right;
    unsigned const flags = __reduce_or_sync(0xffffffffu,
        common ? (left_short && right_over ? 1u : 0u) | (left_over ? 2u : 0u) | (right_short && left_over ? 4u : 0u)
                | (right_over ? 8u : 0u)
               : 0u);
    if ((flags & 3u) != 3u && (flags & 12u) != 12u)
        return;
    halo_q_direction(common, count_left, count_right, quota_left, quota_right);
    halo_q_direction(common, count_right, count_left, quota_right, quota_left);

    if (common)
    {
        quota[b * ep + left] = quota_left;
        quota[b * ep + right] = quota_right;
    }
}

__device__ __forceinline__ void halo_q_active_barrier(int threads)
{
    asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
}

__device__ __forceinline__ void scheduler_tail_barrier()
{
    asm volatile("bar.sync 3, 64;" ::: "memory");
}

/* Build order[] and host_mask[] with one lane per broadcast.
 *
 * order[]:  lane b's position is the number of broadcasts that sort before it,
 *           ordered by expert.  The `j < lane` tiebreak orders equal experts by
 *           broadcast index, so the order is stable.
 * host_mask: the destination group is a contiguous rank range, so the mask is
 *           closed form; no loop over group members is needed.
 *           rank_group_mask handles the full EP32 word without a shift by32.
 */
__device__ __forceinline__ void halo_q_build_order_and_masks(int const* p_expert, int const* p_begin, int const* p_size,
    int const* p_owner, int* order, int* host_mask, int broadcasts)
{
    int const lane = threadIdx.x & 31;
    if (broadcasts > 32)
    {
        for (int b = lane; b < broadcasts; b += 32)
        {
            int const mine = p_expert[b];
            int position = 0;
            for (int j = 0; j < broadcasts; ++j)
                position += p_expert[j] < mine || (p_expert[j] == mine && j < b);
            order[position] = b;
            host_mask[b] = static_cast<int>(rank_group_mask(p_begin[b], p_size[b]) | (1u << p_owner[b]));
        }
        return;
    }
    /* Read p_expert cooperatively and distribute values with shuffles. All
     * lanes must participate before inactive lanes return; inactive lanes
     * carry INT_MAX and never contribute to a real comparison. */
    bool const active = lane < broadcasts;
    int const mine = active ? p_expert[lane] : INT_MAX;
    int position = 0;
    // Independent shuffles: unrolled so they overlap.
#pragma unroll 8
    for (int j = 0; j < broadcasts; ++j)
    {
        int const other = __shfl_sync(0xffffffffu, mine, j);
        position += (other < mine) || (other == mine && j < lane);
    }
    if (!active)
        return;
    order[position] = lane;
    host_mask[lane] = static_cast<int>(rank_group_mask(p_begin[lane], p_size[lane]) | (1u << p_owner[lane]));
}

/* run_halo_q's partner of `rank` in `phase`: ring position i holds rank 1 + (i -
 * phase) mod (ep - 1) and meets position ep - 1 - i, position 0 meets rank 0;
 * `left` receives first; no partner (odd ep) or a lane past ep gets itself. */
__device__ __forceinline__ int halo_q_partner(int rank, int phase, int ep, bool& left)
{
    int const ring = ep - 1;
    int position = rank - 1 + phase;
    position -= position >= ring ? ring : 0;
    int const other = ring - position;
    int index = other - phase;
    index += index < 0 ? ring : 0;
    int partner = position == 0 ? 0 : other == position ? rank : 1 + index;
    partner = rank == 0 ? (phase == 0 ? 1 : ep - phase) : partner;
    left = rank == 0 || (rank < ep && position > 0 && position < other);
    return rank < ep ? partner : rank;
}

/* The phases in which rank `lane` meets one of `partners`: halo_q_partner
 * stepped by phase over Phases >= ep - 1 phases, the extra ones dropped. */
template <int Phases>
__device__ __forceinline__ unsigned halo_q_live_phases(int lane, int ep, unsigned partners)
{
    int const ring = ep - 1;
    int position = lane - 1;
    int index = lane <= 1 ? 0 : ep - lane;
    unsigned live = 0;
#pragma unroll
    for (int phase = 0; phase < Phases; ++phase)
    {
        int partner = position == 0 ? 0 : 2 * position == ring ? lane : 1 + index;
        partner = lane == 0 ? (phase == 0 ? 1 : ep - phase) : partner;
        live |= (partners >> (partner & 31)) & 1u && partner != lane ? 1u << phase : 0u;
        position += position + 1 >= ring ? 1 - ring : 1;
        index += index < 2 ? ring - 2 : -2;
    }
    return lane < ep ? live & ((1u << ring) - 1u) : 0u;
}

/* Ints of `scratch` that the pair schedule of run_halo_q uses: two counters
 * and kScheduleSlots steps of up to kMaxEp / 2 pairs and their count. */
constexpr int kScheduleSlots = 8;
constexpr int kScheduleInts = 2 + kScheduleSlots * (kMaxEp / 2 + 1);

__device__ void run_halo_q(int const* counts, int* quota, int* plan, int broadcasts, int ep, int experts, int* ring,
    int* order, int* host_mask, int* scratch)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    if (broadcasts <= 32)
    {
        /* Two rounds of the round-robin pair tournament.  A pair moves quota
         * only through two broadcasts both ranks host, so pairs sharing fewer
         * are dropped; a pair runs once
         * neither rank has an earlier one left, so each rank keeps the tournament
         * order and the quota equals the two rounds'.  Warp 15 writes the steps
         * into `scratch`; up to 14 other warps, never warp 8, run the pairs. */
        int const workers = imin(ep / 2, 14);
        int const worker = warp < 8 ? warp : warp - 1;
        bool const is_worker = warp != 8 && worker < workers;
        bool const is_scheduler = warp == 15;
        int volatile* const published = scratch;
        int volatile* const consumed = scratch + 1;
        int* const slots = scratch + 2;
        if (is_worker || is_scheduler)
        {
            /* Each warp reads the plan's experts while warp 0 orders them. */
            int const plan_expert = lane < broadcasts ? p_expert[lane] : 0;
            if (warp == 0)
                halo_q_build_order_and_masks(p_expert, p_begin, p_size, p_owner, order, host_mask, broadcasts);
            if (is_scheduler && lane < 2)
                scratch[lane] = 0;
            halo_q_active_barrier((workers + 1) * 32);
            int const my_b = lane < broadcasts ? order[lane] : -1;
            int const my_hosts = my_b >= 0 ? host_mask[my_b] : 0;
            int const my_expert = __shfl_sync(0xffffffffu, plan_expert, my_b & 31);
            if (broadcasts >= 2 && is_scheduler)
            {
                /* From here lane = rank: the ranks that host two of its
                 * broadcasts with it, then the phases in which it meets one. */
                unsigned once[4] = {0, 0, 0, 0};
                unsigned twice[4] = {0, 0, 0, 0};
                for (int base = 0; base < broadcasts; base += 4)
                {
#pragma unroll
                    for (int chain = 0; chain < 4; ++chain)
                    {
                        unsigned const hosts
                            = __shfl_sync(0xffffffffu, static_cast<unsigned>(my_hosts), (base + chain) & 31);
                        unsigned const mine = base + chain < broadcasts && (hosts >> lane) & 1u ? hosts : 0u;
                        twice[chain] |= once[chain] & mine;
                        once[chain] |= mine;
                    }
                }
                unsigned const twice_hosts = twice[0] | twice[1] | twice[2] | twice[3] | (once[0] & once[1])
                    | ((once[0] | once[1]) & (once[2] | once[3])) | (once[2] & once[3]);
                unsigned const live = ep > 16 ? halo_q_live_phases<kMaxEp - 1>(lane, ep, twice_hosts)
                    : ep > 8                  ? halo_q_live_phases<15>(lane, ep, twice_hosts)
                                              : halo_q_live_phases<7>(lane, ep, twice_hosts);
                /* Each step: the pairs whose ranks have no earlier pair left; the
                 * lowest pending pair always qualifies, so an empty step ends. */
                unsigned first = live;
                unsigned second = live;
                for (int step = 0;; ++step)
                {
                    unsigned const pending = first ? first : second;
                    int const phase = __ffs(pending) - 1;
                    int const key = !pending ? -1 - lane : first ? phase : 32 + phase;
                    bool left;
                    int partner = halo_q_partner(lane, phase, ep, left);
                    partner = pending ? partner : lane;
                    int const partner_key = __shfl_sync(0xffffffffu, key, partner);
                    bool const ready = pending && partner_key == key;
                    unsigned const pairs = __ballot_sync(0xffffffffu, ready && left);
                    while (step - *consumed >= kScheduleSlots)
                        __nanosleep(32);
                    int* const slot = slots + (step % kScheduleSlots) * (kMaxEp / 2 + 1);
                    if (ready && left)
                        slot[1 + __popc(pairs & ((1u << lane) - 1u))] = lane | (partner << 8);
                    if (lane == 0)
                        slot[0] = __popc(pairs);
                    __syncwarp();
                    __threadfence_block();
                    if (lane == 0)
                        *published = step + 1;
                    if (!pairs)
                        break;
                    if (ready)
                    {
                        if (first)
                            first &= first - 1;
                        else
                            second &= second - 1;
                    }
                }
            }
            else if (broadcasts >= 2)
            {
                for (int step = 0;; ++step)
                {
                    while (*published <= step)
                        __nanosleep(32);
                    __threadfence_block();
                    int const* const slot = slots + (step % kScheduleSlots) * (kMaxEp / 2 + 1);
                    int const count = slot[0];
                    if (!count)
                        break;
                    for (int index = worker; index < count; index += workers)
                    {
                        int const pair = slot[1 + index];
                        halo_q_pair(pair & 255, pair >> 8, my_b, my_hosts, my_expert, counts, quota, ep, experts);
                    }
                    /* Unconditional for the reasons of the barrier above; it
                     * also ends the step before its slot is reused. */
                    halo_q_active_barrier(workers * 32);
                    if (worker == 0 && lane == 0)
                        *consumed = step + 1;
                }
            }
        }
        return;
    }
    if (warp == 0)
        halo_q_build_order_and_masks(p_expert, p_begin, p_size, p_owner, order, host_mask, broadcasts);
    if (tid < ep - 1)
        ring[tid] = tid + 1;
    __syncthreads();
    if (broadcasts < 2)
        return;

    for (int round = 0; round < 2; ++round)
    {
        for (int phase = 0; phase < ep - 1; ++phase)
        {
            if (warp < ep / 2)
            {
                int left, right;
                if (warp == 0)
                {
                    left = 0;
                    right = ring[0];
                }
                else
                {
                    left = ring[warp];
                    right = ring[ep - 1 - warp];
                }
                halo_q_repair(left, right, broadcasts, order, host_mask, counts, p_expert, quota, ep, experts);
                halo_q_repair(right, left, broadcasts, order, host_mask, counts, p_expert, quota, ep, experts);
            }
            __syncthreads();
            if (tid == 0)
            {
                int last = ring[ep - 2];
                for (int index = ep - 2; index > 0; --index)
                    ring[index] = ring[index - 1];
                ring[0] = last;
            }
            __syncthreads();
        }
    }
}

/* Residency rows the coloring keeps in registers: helper counts up to this. */
constexpr int kResidentSlots = 16;

/* Helper slots by level, coarse first, then broadcast: a slot free on the whole
 * group where every member already holds the expert (`resident`, the layer's
 * [ep][helpers] residency table; null without one), else the lowest slot free
 * on the whole group.  Every coarser group holds both ranks of a pair or
 * neither, so a pair's ranks have the same slots taken when its turn comes.
 * Every rank keeps the same residency table, so every rank picks the same
 * slots.  Warp 8 calls it; up to 32 broadcasts and helpers, lane b holds
 * broadcast b and lane r rank r's taken slots, otherwise lane 0 scans alone,
 * by the lowest free slot. */
__device__ void color_and_publish_local(int* plan, int broadcasts, int ep, int helpers, int local_rank, int* used,
    int* out_ids, int* out_levels, int* out_owners, int* status, Epoch epoch, int const* resident)
{
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_level = plan_field(plan, kPlanLevel);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    int* p_helper = plan_field(plan, kPlanHelper);
    int const lane = threadIdx.x & 31;
    if (broadcasts <= 32 && helpers >= 1 && helpers <= 32)
    {
        bool const is_broadcast = lane < broadcasts;
        int const level_of = is_broadcast ? p_level[lane] : -1;
        int const begin_of = is_broadcast ? p_begin[lane] : 0;
        int const size_of = is_broadcast ? p_size[lane] : 0;
        bool const holds_local = is_broadcast && group_contains(local_rank, begin_of, size_of);
        int const expert_all = is_broadcast ? p_expert[lane] : -1;
        int const expert_of = holds_local ? expert_all : 0;
        int const owner_of = holds_local ? p_owner[lane] : 0;
        unsigned const valid = helpers == 32 ? 0xffffffffu : (1u << helpers) - 1u;
        /* Lane r: the experts rank r's helper slots hold. */
        bool const reuse = resident != nullptr && helpers <= kResidentSlots;
        int held[kResidentSlots];
#pragma unroll
        for (int slot = 0; slot < kResidentSlots; ++slot)
            held[slot] = reuse && lane < ep && slot < helpers ? resident[lane * helpers + slot] : -1;
        unsigned taken = 0;
        int helper_of = -1;
        bool no_slot = false;
        for (int level = 0; level < planner_depth(ep); ++level)
        {
            for (unsigned pending = __ballot_sync(0xffffffffu, level_of == level); pending; pending &= pending - 1)
            {
                int const b = __ffs(pending) - 1;
                int const begin = __shfl_sync(0xffffffffu, begin_of, b);
                int const size = __shfl_sync(0xffffffffu, size_of, b);
                bool const member = group_contains(lane, begin, size);
                unsigned const open_slots = ~__reduce_or_sync(0xffffffffu, member ? taken : 0u) & valid;
                unsigned holding = ~0u;
                if (reuse) // Warp-uniform.
                {
                    int const expert = __shfl_sync(0xffffffffu, expert_all, b);
                    unsigned mine = 0;
#pragma unroll
                    for (int slot = 0; slot < kResidentSlots; ++slot)
                        mine |= held[slot] == expert ? 1u << slot : 0u;
                    holding = __reduce_and_sync(0xffffffffu, member ? mine : ~0u);
                }
                unsigned const kept = reuse ? open_slots & holding : 0u;
                int const helper = kept ? __ffs(kept) - 1 : open_slots ? __ffs(open_slots) - 1 : 0;
                no_slot |= !open_slots;
                taken |= member ? 1u << helper : 0u;
                helper_of = lane == b ? helper : helper_of;
            }
        }
        if (no_slot && lane == 0)
            record_status(status + 4, epoch);
        if (helper_of >= 0)
            p_helper[lane] = helper_of;
        if (lane < helpers)
        {
            out_ids[lane] = -1;
            out_levels[lane] = -1;
            out_owners[lane] = -1;
        }
        __syncwarp();
        /* The last broadcast holding local_rank fills its slot; a slot two
         * of them hold records the status. */
        unsigned filled = 0;
        for (unsigned writers = __ballot_sync(0xffffffffu, holds_local && helper_of >= 0); writers;)
        {
            int const b = 31 - __clz(writers);
            writers &= ~(1u << b);
            int const slot = __shfl_sync(0xffffffffu, helper_of, b);
            bool const twice = (filled >> slot) & 1u;
            if (lane == b && !twice)
            {
                out_ids[slot] = expert_of;
                out_levels[slot] = level_of;
                out_owners[slot] = owner_of;
            }
            if (twice && lane == 0)
                record_status(status + 4, epoch);
            filled |= 1u << slot;
        }
        return;
    }
    if (lane != 0)
        return;
    int const active_helpers = imin(helpers, kMaxBroadcasts);
    int const words = (active_helpers + 31) / 32;
    for (int rank = 0; rank < ep; ++rank)
        for (int word = 0; word < words; ++word)
            used[rank * kHelperMaskWords + word] = 0;
    for (int level = 0; level < planner_depth(ep); ++level)
    {
        for (int b = 0; b < broadcasts; ++b)
        {
            if (p_level[b] != level)
                continue;
            /* The lowest slot free on every group rank, one mask word at a
             * time, so the scan keeps no per-word array. */
            int helper = -1;
            for (int word = 0; word < words && helper < 0; ++word)
            {
                unsigned unavailable = 0;
                for (int rank = p_begin[b]; rank < p_begin[b] + p_size[b]; ++rank)
                    unavailable |= static_cast<unsigned>(used[rank * kHelperMaskWords + word]);
                int const slots = imin(32, active_helpers - word * 32);
                unsigned const valid = slots == 32 ? 0xffffffffu : (1u << slots) - 1u;
                unsigned const open_slots = ~unavailable & valid;
                if (open_slots)
                    helper = word * 32 + __ffs(open_slots) - 1;
            }
            if (helper < 0)
            {
                record_status(status + 4, epoch);
                helper = 0;
            }
            p_helper[b] = helper;
            for (int rank = p_begin[b]; rank < p_begin[b] + p_size[b]; ++rank)
                used[rank * kHelperMaskWords + helper / 32] |= static_cast<int>(1u << (helper % 32));
        }
    }
    for (int slot = 0; slot < helpers; ++slot)
    {
        out_ids[slot] = -1;
        out_levels[slot] = -1;
        out_owners[slot] = -1;
    }
    for (int b = 0; b < broadcasts; ++b)
    {
        if (!group_contains(local_rank, p_begin[b], p_size[b]))
            continue;
        int helper = p_helper[b];
        if (out_ids[helper] != -1)
            record_status(status + 4, epoch);
        out_ids[helper] = p_expert[b];
        out_levels[helper] = p_level[b];
        out_owners[helper] = p_owner[b];
    }
}

__device__ void build_route_prefix(int const* counts, int const* quota, int* prefix, int const* plan, int broadcasts,
    int ep, int experts, int local_rank, int participating_threads)
{
    int* p_expert = plan_field(const_cast<int*>(plan), kPlanExpert);
    for (int b = threadIdx.x; b < broadcasts; b += participating_threads)
    {
        int expert = p_expert[b];
        int source_before = 0;
        /* Ranks past the bound load a clamped valid entry and add nothing,
         * so every load of a chunk is unconditional. */
        for (int base = 0; base < local_rank; base += kRankChunk)
        {
#pragma unroll
            for (int step = 0; step < kRankChunk; ++step)
            {
                int const rank = imin(base + step, local_rank - 1);
                int const count = counts[rank * experts + expert];
                int const diagonal = imin(count, quota[b * ep + rank]);
                source_before += base + step < local_rank ? count - diagonal : 0;
            }
        }
        int source_count = counts[local_rank * experts + expert];
        int source_diagonal = imin(source_count, quota[b * ep + local_rank]);
        int residual_supply = source_count - source_diagonal;
        /* route_destination loads prefix[destination + step] only under
         * `candidate < ep`, so entry ep is never read: the loop stops before
         * destination == ep and skips the accumulation after the last
         * destination, which would only feed that entry.  Entry 0 is written
         * although route_destination never reads it: it costs one store and
         * leaves no stale entry below ep. */
        int destination_before = 0;
        for (int base = 0; base < ep; base += kRankChunk)
        {
            /* Load the whole chunk before the first prefix store, which the
             * compiler cannot order after loads it might alias. */
            int destination_supply[kRankChunk];
#pragma unroll
            for (int step = 0; step < kRankChunk; ++step)
            {
                int const destination = imin(base + step, ep - 1);
                int const destination_quota = quota[b * ep + destination];
                destination_supply[step]
                    = destination_quota - imin(counts[destination * experts + expert], destination_quota);
            }
#pragma unroll
            for (int step = 0; step < kRankChunk; ++step)
            {
                int const destination = base + step;
                int const overlap = imax(0, imin(residual_supply, destination_before - source_before));
                int const diagonal_prefix = local_rank < destination ? source_diagonal : 0;
                if (destination < ep)
                    prefix[b * (ep + 1) + destination] = diagonal_prefix + overlap;
                destination_before += destination + 1 < ep ? destination_supply[step] : 0;
            }
        }
    }
}

__device__ __forceinline__ int* worker_expert_lut(int experts)
{
    extern __shared__ int shared[];
    return shared + (blockDim.x >> 5) * (experts + 1);
}

__device__ void prepare_worker_default_slots(int* expert_lut, int ep, int experts, int helpers)
{
    int const home = experts / ep;
    int const local_slots = home + helpers;
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x)
    {
        int owner = expert / home;
        int local = expert - owner * home;
        /* Negative values distinguish the precomputed cold slot from a
         * nonnegative broadcast index.  Slot zero therefore maps to -1. */
        expert_lut[expert] = -(owner * local_slots + local) - 1;
    }
    __syncthreads();
}

__device__ void prepare_worker_selected_experts(int* expert_lut, int const* plan)
{
    int const broadcasts = plan[1];
    int const* p_expert = plan + 2 + kPlanExpert * kMaxBroadcasts;
    for (int b = threadIdx.x; b < broadcasts; b += blockDim.x)
        expert_lut[p_expert[b]] = b;
    __syncthreads();
}

__device__ bool wait_for_plan_epoch(int* marker, Epoch epoch, unsigned long long spin_cycles, int* status, int* ready)
{
    if (threadIdx.x == 0)
    {
        *ready = 1;
        unsigned long long start = clock64();
        Epoch observed = load_relaxed_device(marker);
        while (epoch_before(observed, epoch))
        {
            if (clock64() - start > spin_cycles)
            {
                record_status(status + 2, epoch);
                *ready = 0;
                break;
            }
            observed = load_relaxed_device(marker);
        }
        if (!epoch_before(observed, epoch))
            (void) load_acquire_device(marker);
    }
    __syncthreads();
    return *ready != 0;
}

/* Prefix entries are monotonic, so binary lifting returns the number of
 * thresholds <= ordinal, counting repeated entries. */
__device__ __forceinline__ int route_destination(int const* prefix, int ordinal, int ep)
{
    int destination = 0;
#pragma unroll
    for (int step = 16; step; step >>= 1)
    {
        int candidate = destination + step;
        if (candidate < ep && ordinal >= prefix[candidate])
            destination = candidate;
    }
    return destination;
}

/* Once HALO-M publishes its selected experts, workers can finish every cold
 * route while CTA0 runs HALO-Q.  Selected route indices are compacted into
 * disjoint slices of route_aux; their ordinal values remain in output. */
__device__ void pre_materialize_and_compact_worker_routes(int const* routes, int* output, int* route_aux, int* status,
    int route_count, int experts, int worker, int workers, Epoch epoch)
{
    extern __shared__ int shared[];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const warps = blockDim.x >> 5;
    int const bins = experts + 1;
    int* expert_lut = worker_expert_lut(experts);
    int* hot_indices = route_aux + 2 + workers * bins;

    int const worker_span = (route_count + workers - 1) / workers;
    int const worker_begin = worker * worker_span;
    int const worker_end = imin(route_count, worker_begin + worker_span);
    int const warp_span = ((worker_span + warps * 32 - 1) / (warps * 32)) * 32;
    int const begin = worker_begin + warp * warp_span;
    int const end = imin(worker_end, begin + warp_span);
    int hot_count = 0;
    for (int base = begin; base < end; base += 32)
    {
        int index = base + lane;
        bool active = index < end;
        int expert = active ? routes[index] : -1;
        int slot = -1;
        bool hot = false;
        if (active && expert >= 0 && expert < experts)
        {
            int entry = expert_lut[expert];
            hot = entry >= 0;
            if (!hot)
                slot = -entry - 1;
        }
        else if (active && expert != -1)
        {
            record_status(status + 3, epoch);
        }
        unsigned hot_mask = __ballot_sync(0xffffffffu, hot);
        if (hot)
        {
            int offset = __popc(hot_mask & ((1u << lane) - 1u));
            hot_indices[begin + hot_count + offset] = index;
        }
        else if (active)
        {
            output[index] = slot;
        }
        hot_count += __popc(hot_mask);
    }
    if (lane == 0)
        shared[warp * bins + experts] = hot_count;
}

/* The final epoch only has to resolve selected routes.  Stable ordinal
 * cursors remain in the per-warp shared histogram from the initial pass. */
__device__ void materialize_worker_hot_routes(int const* routes, int* output, int const* route_prefix, int const* plan,
    int* route_aux, int* status, int route_count, int ep, int experts, int helpers, int worker, int workers,
    Epoch epoch)
{
    extern __shared__ int shared[];
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const warps = blockDim.x >> 5;
    int const bins = experts + 1;
    int* expert_lut = worker_expert_lut(experts);
    int* hot_indices = route_aux + 2 + workers * bins;
    int const home = experts / ep;
    int const local_slots = home + helpers;
    int const* p_begin = plan + 2 + kPlanGroupBegin * kMaxBroadcasts;
    int const* p_size = plan + 2 + kPlanGroupSize * kMaxBroadcasts;
    int const* p_owner = plan + 2 + kPlanOwner * kMaxBroadcasts;
    int const* p_helper = plan + 2 + kPlanHelper * kMaxBroadcasts;

    int const worker_span = (route_count + workers - 1) / workers;
    int const worker_begin = worker * worker_span;
    int const worker_end = imin(route_count, worker_begin + worker_span);
    int const warp_span = ((worker_span + warps * 32 - 1) / (warps * 32)) * 32;
    int const begin = worker_begin + warp * warp_span;
    int const end = imin(worker_end, begin + warp_span);
    int hot_count = shared[warp * bins + experts];
    hot_count = imin(hot_count, imax(0, end - begin));
    /* Each lane resolves kHotBatch routes at once: every dependent load of a
     * stage is issued for the whole batch before the next stage consumes it.
     * Batch members are distinct routes, so the order of their stores does
     * not matter. */
    constexpr int kHotBatch = 4;
    for (int first = lane; first < hot_count; first += 32 * kHotBatch)
    {
        int index[kHotBatch];
        int expert[kHotBatch];
        int ordinal[kHotBatch];
#pragma unroll
        for (int step = 0; step < kHotBatch; ++step)
        {
            int const offset = first + 32 * step;
            index[step] = offset < hot_count ? hot_indices[begin + offset] : -1;
        }
#pragma unroll
        for (int step = 0; step < kHotBatch; ++step)
        {
            expert[step] = index[step] >= 0 ? routes[index[step]] : 0;
            ordinal[step] = index[step] >= 0 ? output[index[step]] : 0;
        }
#pragma unroll
        for (int step = 0; step < kHotBatch; ++step)
        {
            if (index[step] < 0)
                continue;
            int b = expert_lut[expert[step]];
            int const* prefix = route_prefix + b * (ep + 1);
            int destination = route_destination(prefix, ordinal[step] + shared[warp * bins + expert[step]], ep);
            int slot = -1;
            if (group_contains(destination, p_begin[b], p_size[b]))
            {
                slot = destination * local_slots + home + p_helper[b];
            }
            else if (destination == p_owner[b])
            {
                slot = destination * local_slots + expert[step] - p_owner[b] * home;
            }
            else
            {
                record_status(status + 5, epoch);
            }
            output[index[step]] = slot;
        }
    }
}

__device__ void materialize_worker_routes(int const* routes, int* output, int const* route_prefix, int const* plan,
    int* status, int route_count, int ep, int experts, int helpers, int worker, int workers, Epoch epoch)
{
    extern __shared__ int shared[];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const warps = blockDim.x >> 5;
    int const bins = experts + 1;
    int* expert_lut = worker_expert_lut(experts);
    int const home = experts / ep;
    int const local_slots = home + helpers;
    int const* p_begin = plan + 2 + kPlanGroupBegin * kMaxBroadcasts;
    int const* p_size = plan + 2 + kPlanGroupSize * kMaxBroadcasts;
    int const* p_owner = plan + 2 + kPlanOwner * kMaxBroadcasts;
    int const* p_helper = plan + 2 + kPlanHelper * kMaxBroadcasts;

    int const worker_span = (route_count + workers - 1) / workers;
    int const worker_begin = worker * worker_span;
    int const worker_end = imin(route_count, worker_begin + worker_span);
    int const warp_span = ((worker_span + warps * 32 - 1) / (warps * 32)) * 32;
    int const begin = worker_begin + warp * warp_span;
    int const end = imin(worker_end, begin + warp_span);
    for (int index = begin + lane; index < end; index += 32)
    {
        int expert = routes[index];
        int slot = -1;
        if (expert >= 0 && expert < experts)
        {
            int entry = expert_lut[expert];
            if (entry < 0)
            {
                slot = -entry - 1;
            }
            else
            {
                int b = entry;
                int ordinal = output[index] + shared[warp * bins + expert];
                int const* prefix = route_prefix + b * (ep + 1);
                int destination = route_destination(prefix, ordinal, ep);
                if (group_contains(destination, p_begin[b], p_size[b]))
                {
                    slot = destination * local_slots + home + p_helper[b];
                }
                else if (destination == p_owner[b])
                {
                    slot = destination * local_slots + expert - p_owner[b] * home;
                }
                else
                {
                    record_status(status + 5, epoch);
                }
            }
        }
        else if (expert != -1)
        {
            record_status(status + 3, epoch);
        }
        output[index] = slot;
    }
}

/* One scheduler generation on a grid of ctas resident CTAs of 512 threads:
 * CTA 0 plans and publishes the helper placement, the other CTAs write routes.
 * The fused scheduler + in-switch copy kernel runs it as its first phase. */
__device__ __forceinline__ void halo_q_scheduler_body(int const* routes, int* out_slots, int* out_ids, int* out_levels,
    int* out_owners, unsigned long long const* peer_bases, int* status, int* partial, int* route_aux, int* grid_sync,
    int* plan, int* route_prefix, int ep, int experts, int helpers, int local_rank, int route_capacity, int ctas,
    unsigned long long spin_cycles, int capacity, int route_count, int const* resident)
{
    // Only the caller's valid prefix is readable. Keep the public output at
    // its configured capacity, clearing stale routes inside this same kernel.
    // Empty ranks still execute every collective; no route load occurs at N=0.
    for (int index = route_count + blockIdx.x * blockDim.x + threadIdx.x; index < route_capacity;
         index += gridDim.x * blockDim.x)
        out_slots[index] = -1;
    local_histogram_pass(routes, partial, route_count, experts, ctas);
    Epoch epoch;
    if (!grid_rendezvous(grid_sync, ctas, spin_cycles, status, &epoch))
        return;

    if (blockIdx.x == 0)
    {
        extern __shared__ int shared[];
        int offset = ep * experts;
        int* counts = shared;
        int* histogram = shared + offset;
        offset += experts;
        int* load = shared + offset;
        offset += ep;
        // Reuse the exact capacity already validated/computed by launch().
        int* quota = shared + offset;
        offset += capacity * ep;
        int* selected_flags = shared + offset;
        offset += experts;
        int* recv_used = shared + offset;
        offset += ep * kHelperMaskWords;
        int* ring = shared + offset;
        offset += ep;
        int* order = shared + offset;
        offset += capacity;
        int* host_mask = shared + offset;
        offset += capacity;
        int* schedule = shared + offset;
        offset += kScheduleInts;
        int* running = shared + offset;

        if (!exchange_counts(peer_bases, partial, shared, ep, experts, local_rank, ctas, epoch, spin_cycles, status))
            return;
        if (uses_register_prepare(ep, experts))
            prepare_halo_m_register(counts, histogram, load, selected_flags, ep, experts, helpers);
        else
            prepare_halo_m(counts, histogram, load, selected_flags, ep, experts, helpers);

        int const broadcasts
            = place_halo_q(histogram, load, quota, selected_flags, running, plan, ep, experts, helpers);
        /* place_halo_q() ends at a block barrier after the plan fields and
         * plan[1] are written.  Publish this placement-only epoch before HALO-Q
         * so workers can construct their selected-expert LUT while quota repair
         * is still running. */
        if (ctas > 1 && threadIdx.x == 0)
        {
            store_release_device(plan + 0, epoch);
        }
        /* Repair of <= 32 broadcasts excludes warp8 and can overlap coloring. Generic
         * repair first meets the coloring warp at its full-block barrier. */
        if (threadIdx.x >> 5 == 8)
            color_and_publish_local(plan, broadcasts, ep, helpers, local_rank, recv_used, out_ids, out_levels,
                out_owners, status, epoch, resident);
        run_halo_q(counts, quota, plan, broadcasts, ep, experts, ring, order, host_mask, schedule);
        if (ctas > 1)
        {
            int const warp = threadIdx.x >> 5;
            /* build_route_prefix is independent of warp 8's outputs. Defer their
             * rendezvous until both paths reach their first shared consumer so
             * prefix construction and coloring can overlap safely. */
            if (warp == 0)
            {
                build_route_prefix(counts, quota, route_prefix, plan, broadcasts, ep, experts, local_rank, 32);
                __syncwarp();
            }
            if (warp == 0 || warp == 8)
                scheduler_tail_barrier();
            /* The worker CTAs gate on grid_sync+2 and consume route_prefix,
             * which is complete above. */
            if (threadIdx.x == 0)
                store_release_device(grid_sync + 2, epoch);
            return;
        }
        if (threadIdx.x == 0)
            store_release_device(plan + 0, epoch);
        __syncthreads();
        build_route_prefix(counts, quota, route_prefix, plan, broadcasts, ep, experts, local_rank, blockDim.x);
        /* The block barrier orders route-prefix/helper writes before release. */
        __syncthreads();
    }
    else
    {
        int const worker = blockIdx.x - 1;
        int const workers = ctas - 1;
        if (!stable_worker_ordinal_pass(
                routes, out_slots, route_aux, status, route_count, experts, worker, workers, spin_cycles))
            return;
        int* expert_lut = worker_expert_lut(experts);
        prepare_worker_default_slots(expert_lut, ep, experts, helpers);
        __shared__ int plan_ready;
        if (!wait_for_plan_epoch(plan + 0, epoch, spin_cycles, status, &plan_ready))
            return;
        prepare_worker_selected_experts(expert_lut, plan);
        pre_materialize_and_compact_worker_routes(
            routes, out_slots, route_aux, status, route_count, experts, worker, workers, epoch);
        if (!wait_for_plan_epoch(grid_sync + 2, epoch, spin_cycles, status, &plan_ready))
            return;
        materialize_worker_hot_routes(routes, out_slots, route_prefix, plan, route_aux, status, route_count, ep,
            experts, helpers, worker, workers, epoch);
        return;
    }

    /* CTA-count one remains a correct serial fallback. */
    if (!stable_worker_ordinal_pass(routes, out_slots, route_aux, status, route_count, experts, 0, 1, spin_cycles))
        return;
    int* expert_lut = worker_expert_lut(experts);
    prepare_worker_default_slots(expert_lut, ep, experts, helpers);
    prepare_worker_selected_experts(expert_lut, plan);
    materialize_worker_routes(
        routes, out_slots, route_prefix, plan, status, route_count, ep, experts, helpers, 0, 1, epoch);
}

/* Dynamic shared-memory ints that the planner CTA and a worker CTA need. */
__host__ __device__ __forceinline__ int halo_q_shared_ints(int ep, int experts, int helpers, int threads)
{
    int const capacity = broadcast_capacity(ep, experts, helpers);
    int const workerInts = threads / 32 * (experts + 1) + experts;
    int const plannerInts = ep * experts + experts + ep + capacity * ep + experts + (kHelperMaskWords + 1) * ep
        + 2 * capacity + kScheduleInts + 2 + 2 * kPlannerWarps
        + (planner_dense_spare(ep, experts, helpers)
                ? (planner_targets(ep, experts, helpers) - 1) * planner_dense_ints(ep, experts, helpers)
                : planner_targets(ep, experts, helpers) * planner_warp_ints(ep, experts, helpers));
    return workerInts > plannerInts ? workerInts : plannerInts;
}

} // namespace

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_KERNEL_CUH
