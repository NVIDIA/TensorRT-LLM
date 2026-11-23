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

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
constexpr int kMaxEp = 32;
constexpr int kMaxExperts = 384;
constexpr int kMaxBroadcasts = kMaxExperts;
constexpr int kHelperMaskWords = (kMaxBroadcasts + 31) / 32;
constexpr int kMinGroup = 4;
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

__device__ __forceinline__ int imin(int a, int b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ int imax(int a, int b)
{
    return a > b ? a : b;
}

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

__device__ __forceinline__ bool group_contains(int rank, int begin, int size)
{
    return rank >= begin && rank < begin + size;
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
        if (ep % groups || ep / groups < kMinGroup)
            break;
        ++depth;
    }
    return depth;
}

/* Distinct experts and total receive capacity independently bound broadcasts.
 * Actual S is retained in the public outputs; only scratch uses this bound. */
__host__ __device__ __forceinline__ int broadcast_capacity(int ep, int experts, int helpers)
{
    int const bounded_helpers = helpers < experts ? helpers : experts;
    /* Every hierarchy level divides ep exactly. This cancels two integer
     * divisions without changing capacity, including non-power-of-two EP. */
    int const groups = 1 << (hierarchy_depth(ep) - 1);
    int const receive_bound = groups * bounded_helpers;
    return receive_bound < experts ? receive_bound : experts;
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

/* Rank and home bounds of the register planner path.  Every hierarchy level
 * of EP <= 8 fits one 8-lane warp segment, and a home of at most 64 experts
 * fits two 32-lane rounds. */
constexpr int kRegisterEp = 8;
constexpr int kRegisterRounds = 2;

__device__ __forceinline__ bool uses_register_prepare(int ep, int experts)
{
    return ep <= kRegisterEp && experts / ep <= 32 * kRegisterRounds;
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

__device__ __forceinline__ void reset_owner_state(
    int owner, int rank_load, int* load, int* recv_used, int* send_used, int* next_index, int* frozen)
{
    load[owner] = rank_load;
    recv_used[owner] = 0;
    send_used[owner] = 0;
    next_index[owner] = 0;
    frozen[owner] = 0;
}

/* Rank loads and per-owner hot lists for uses_register_prepare(): one warp
 * per owner sums its home masses over the source ranks straight into
 * registers.  Every load is unconditional (out-of-range sources and experts
 * read a clamped valid address and contribute nothing), so all of them are in
 * flight together.  Only the candidates' masses reach histogram[]; the
 * planners read no other entry.  The owner warps, warp 0 among them, meet at
 * a named barrier; the other warps leave at once. */
__device__ void prepare_halo_m_register(int const* counts, int* histogram, int* load, int* candidates, int* recv_used,
    int* send_used, int* next_index, int* frozen, int ep, int experts, int helpers)
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
        reset_owner_state(owner, rank_load, load, recv_used, send_used, next_index, frozen);
    select_owner_candidates(local_mass, histogram, candidates, owner, home, imin(helpers, home), true);
    asm volatile("bar.sync 4, %0;" ::"r"(ep * 32) : "memory");
}

/* Rounds of 32 home experts per lane; EP2 has the largest home, 192 experts. */
constexpr int kMaxHomeRounds = (kMaxExperts / 2 + 31) / 32;

/* Dense histogram, rank loads and per-owner hot lists for every other
 * geometry, with one warp per owner. */
__device__ void prepare_halo_m(int const* counts, int* histogram, int* load, int* candidates, int* recv_used,
    int* send_used, int* next_index, int* frozen, int ep, int experts, int helpers)
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

    for (int owner = warp; owner < ep; owner += blockDim.x >> 5)
    {
        int rank_load = 0;
        for (int local = lane; local < home; local += 32)
            rank_load += histogram[owner * home + local];
        rank_load = __reduce_add_sync(0xffffffffu, rank_load);
        if (lane == 0)
            reset_owner_state(owner, rank_load, load, recv_used, send_used, next_index, frozen);
        int local_mass[kMaxHomeRounds];
#pragma unroll
        for (int round = 0; round < kMaxHomeRounds; ++round)
        {
            int const local = lane + 32 * round;
            local_mass[round] = local < home ? histogram[owner * home + local] : -1;
        }
        select_owner_candidates(local_mass, histogram, candidates, owner, home, imin(helpers, home), false);
    }
    __syncthreads();
}

/* EP<=8 fast path: every participating lane owns one rank's complete
 * mutable HALO-M state.  Group and destination decisions use only shuffles,
 * avoiding repeated shared-memory rank-state traffic and warp barriers. */
template <bool NarrowKeys>
__device__ __forceinline__ int plan_halo_m_register8_impl(
    int const* histogram, int lane_load, int total, int* quota, int const* candidates, int* plan, int ep, int helpers)
{
    int const lane = threadIdx.x & 31;
    constexpr unsigned mask = 0xffu;
    int lane_recv = 0;
    int lane_send = 0;
    int lane_next = 0;
    int lane_frozen = 0;

    int const target = total / ep + (total % ep != 0);
    int const depth = hierarchy_depth(ep);
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_level = plan_field(plan, kPlanLevel);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    int* p_helper = plan_field(plan, kPlanHelper);
    int broadcasts = 0;

    int trips = 0;
    for (int guard = 0; guard < ep * (helpers + 2); ++guard)
    {
        ++trips;
        bool const eligible = lane < ep && !lane_frozen && lane_load > target;
        int owner, owner_load;
        if constexpr (NarrowKeys)
        {
            unsigned const key = __reduce_max_sync(
                mask, eligible ? ((static_cast<unsigned>(lane_load) << 4) | (15u - static_cast<unsigned>(lane))) : 0u);
            if (!key)
                break;
            owner = 15 - static_cast<int>(key & 15u);
            owner_load = static_cast<int>(key >> 4);
        }
        else
        {
            owner_load = __reduce_max_sync(mask, eligible ? lane_load : -1);
            if (owner_load < 0)
                break;
            owner = __reduce_min_sync(mask, eligible && lane_load == owner_load ? lane : INT_MAX);
        }

        unsigned const packed
            = __shfl_sync(mask, (static_cast<unsigned>(lane_send) << 4) | static_cast<unsigned>(lane_next), owner, 8);
        int const owner_send = static_cast<int>(packed >> 4);
        int const owner_next = static_cast<int>(packed & 15u);
        if (owner_send >= helpers || owner_next >= helpers)
        {
            if (lane == owner)
                lane_frozen = 1;
            continue;
        }

        /* owner and owner_next are warp-uniform here, so every lane can read
         * candidates[] and histogram[] directly.  Both are written by
         * the prepare step before its barrier and never inside this
         * loop, so every lane loads the same values and no broadcast from the
         * owner is needed.  Only the owner advances lane_next, and it does so
         * before the mass test. */
        int const expert = candidates[owner * helpers + owner_next];
        if (lane == owner)
            ++lane_next;
        int const mass = histogram[expert];
        if (mass == 0)
        {
            if (lane == owner)
                lane_frozen = 1;
            continue;
        }

        int give = imin(owner_load - target, mass);
        if (give <= 0)
        {
            if (lane == owner)
                lane_frozen = 1;
            continue;
        }

        /* EP <= 8 has at most two hierarchy levels: level 0 is every rank
         * and, at EP8, level 1 is the two halves of four ranks.  Two
         * independent full-mask sums give the two half sums.  Integer addition
         * is exact, so zeroing the other half's addends yields each half sum
         * exactly; the two reduces do not depend on each other, and neither
         * redux.sync uses a per-lane membermask.  The lower half wins a tie. */
        int const half = ep >> 1;
        unsigned const rank_mask = (1u << ep) - 1u;
        unsigned const low_mask = (1u << half) - 1u;
        unsigned const high_mask = rank_mask & ~low_mask;
        int const slack = lane < ep && lane != owner ? imax(0, target - lane_load) : 0;
        int const room_low = __reduce_add_sync(mask, lane < half ? slack : 0);
        int const room_high = __reduce_add_sync(mask, lane >= half ? slack : 0);
        unsigned const receive_mask = __ballot_sync(mask, lane < ep && lane_recv < helpers);
        int chosen_level = -1;
        int chosen_begin = -1;
        int chosen_size = 0;
        if (depth > 1 && mass <= half * target)
        {
            bool const valid_low = (receive_mask & low_mask) == low_mask && room_low > 0;
            bool const valid_high = (receive_mask & high_mask) == high_mask && room_high > 0;
            if (valid_low || valid_high)
            {
                chosen_level = 1;
                chosen_begin = valid_low && (!valid_high || room_low >= room_high) ? 0 : half;
                chosen_size = half;
            }
        }
        if (chosen_level < 0 && (receive_mask & rank_mask) == rank_mask && room_low + room_high > 0)
        {
            chosen_level = 0;
            chosen_begin = 0;
            chosen_size = ep;
        }
        if (chosen_level < 0)
        {
            if (lane == owner)
                lane_frozen = 1;
            continue;
        }
        if (broadcasts >= kMaxBroadcasts)
            break;

        int lane_quota = lane == owner ? mass : 0;
        int remaining = give;
        unsigned visited = 0;
        for (int step = 0; step < chosen_size && remaining > 0; ++step)
        {
            bool can_receive = lane >= chosen_begin && lane < chosen_begin + chosen_size && lane != owner
                && !(visited & (1u << lane));
            int destination, destination_load;
            if constexpr (NarrowKeys)
            {
                unsigned const key = __reduce_min_sync(mask,
                    can_receive ? ((static_cast<unsigned>(lane_load) << 4) | static_cast<unsigned>(lane))
                                : 0xffffffffu);
                if (key == 0xffffffffu)
                    break;
                destination = static_cast<int>(key & 15u);
                destination_load = static_cast<int>(key >> 4);
            }
            else
            {
                destination_load = __reduce_min_sync(mask, can_receive ? lane_load : INT_MAX);
                destination = __reduce_min_sync(mask, can_receive && lane_load == destination_load ? lane : INT_MAX);
                if (destination == INT_MAX)
                    break;
            }
            visited |= 1u << destination;
            int room = target - destination_load;
            if (room <= 0)
                break;
            int amount = imin(room, remaining);
            if (lane == owner)
                lane_quota -= amount;
            if (lane == destination)
                lane_quota += amount;
            remaining -= amount;
        }

        int moved = give - remaining;
        if (moved <= 0)
        {
            if (lane == owner)
                lane_frozen = 1;
            continue;
        }

        int b = broadcasts++;
        if (lane == 0)
        {
            p_expert[b] = expert;
            p_level[b] = chosen_level;
            p_begin[b] = chosen_begin;
            p_size[b] = chosen_size;
            p_owner[b] = owner;
            p_helper[b] = -1;
        }
        if (lane < ep)
            quota[b * ep + lane] = lane_quota;

        bool in_group = lane >= chosen_begin && lane < chosen_begin + chosen_size;
        if (in_group)
            ++lane_recv;
        if (lane == owner)
        {
            ++lane_send;
            lane_load -= moved;
        }
        else if (in_group)
        {
            lane_load += lane_quota;
        }
    }
    return broadcasts;
}

/* Select the key representation once, from the actual complete rank load.
 * This guard keeps the owner and destination load keys representable.
 * Keep the wide path for large valid int32 inputs without putting a runtime
 * representation branch on each iteration of either planning loop. */
__device__ __forceinline__ int plan_halo_m_register8(
    int const* histogram, int const* load, int* quota, int const* candidates, int* plan, int ep, int helpers)
{
    int const lane = threadIdx.x & 31;
    int const lane_load = lane < ep ? load[lane] : 0;
    int const total = __reduce_add_sync(0xffu, lane_load);
    if (total <= 0x0fffffff)
        return plan_halo_m_register8_impl<true>(histogram, lane_load, total, quota, candidates, plan, ep, helpers);
    return plan_halo_m_register8_impl<false>(histogram, lane_load, total, quota, candidates, plan, ep, helpers);
}

/* Exact HALO-M state machine executed by warp 0.  Each lane owns one rank;
 * rank/group/destination scans are warp reductions rather than serial loops. */
__device__ int plan_halo_m_warp(int const* histogram, int* load, int* quota, int const* candidates, int* recv_used,
    int* send_used, int* next_index, int* frozen, int* plan, int ep, int experts, int helpers)
{
    int const candidate_count = imin(helpers, experts / ep);
    int const lane = threadIdx.x & 31;
    int const reduce_width = ep <= 8 ? 8 : 32;
    int const reduce_start = reduce_width >> 1;
    unsigned const reduce_mask = ep <= 8 ? 0xffu : 0xffffffffu;
    int total = lane < ep ? load[lane] : 0;
    for (int offset = reduce_start; offset; offset >>= 1)
    {
        int const other = __shfl_down_sync(reduce_mask, total, offset, reduce_width);
        if (lane + offset < reduce_width)
            total += other;
    }
    total = __shfl_sync(reduce_mask, total, 0, reduce_width);
    int const target = total / ep + (total % ep != 0);
    int const depth = hierarchy_depth(ep);
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_level = plan_field(plan, kPlanLevel);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    int* p_helper = plan_field(plan, kPlanHelper);
    int broadcasts = 0;

    for (int guard = 0; guard < ep * (candidate_count + 2); ++guard)
    {
        bool eligible = lane < ep && !frozen[lane] && load[lane] > target;
        int best_load = eligible ? load[lane] : -1;
        int best_rank = eligible ? lane : INT_MAX;
        for (int offset = reduce_start; offset; offset >>= 1)
        {
            int other_load = __shfl_down_sync(reduce_mask, best_load, offset, reduce_width);
            int other_rank = __shfl_down_sync(reduce_mask, best_rank, offset, reduce_width);
            if (lane + offset < reduce_width
                && (other_load > best_load || (other_load == best_load && other_rank < best_rank)))
            {
                best_load = other_load;
                best_rank = other_rank;
            }
        }
        int owner = __shfl_sync(reduce_mask, best_rank, 0, reduce_width);
        if (owner == INT_MAX)
            break;
        /* One lane owns the capacity check and shared cursor transition.
         * Publish either the chosen expert or -1 for exhaustion explicitly;
         * no reader can observe a partially advanced cursor. */
        int expert = -1;
        if (lane == 0)
        {
            if (send_used[owner] >= helpers || next_index[owner] >= candidate_count)
            {
                frozen[owner] = 1;
            }
            else
            {
                int const candidate_index = next_index[owner]++;
                expert = candidates[owner * candidate_count + candidate_index];
            }
        }
        expert = __shfl_sync(reduce_mask, expert, 0, reduce_width);
        __syncwarp(reduce_mask);
        if (expert < 0)
            continue;
        int mass = histogram[expert];
        if (mass == 0)
        {
            if (lane == 0)
                frozen[owner] = 1;
            __syncwarp(reduce_mask);
            continue;
        }
        int give = imin(load[owner] - target, mass);
        if (give <= 0)
        {
            if (lane == 0)
                frozen[owner] = 1;
            __syncwarp(reduce_mask);
            continue;
        }

        int need = imax(1, mass / imax(target, 1) + (mass % imax(target, 1) != 0));
        int wanted = depth - 1;
        for (int level = depth - 1; level >= 0; --level)
        {
            if (ep / (1 << level) >= need)
            {
                wanted = level;
                break;
            }
        }
        int chosen_level = -1;
        int chosen_begin = -1;
        int chosen_size = 0;
        for (int level = wanted; level >= 0 && chosen_level < 0; --level)
        {
            int size = ep / (1 << level);
            int begin = lane * size;
            bool active = begin < ep;
            bool affordable = active;
            int room = 0;
            if (active)
            {
                for (int rank = begin; rank < begin + size; ++rank)
                {
                    affordable &= recv_used[rank] < helpers;
                    if (rank != owner)
                        room += imax(0, target - load[rank]);
                }
            }
            int candidate_room = affordable ? room : -1;
            int candidate_begin = affordable ? begin : INT_MAX;
            for (int offset = reduce_start; offset; offset >>= 1)
            {
                int other_room = __shfl_down_sync(reduce_mask, candidate_room, offset, reduce_width);
                int other_begin = __shfl_down_sync(reduce_mask, candidate_begin, offset, reduce_width);
                if (lane + offset < reduce_width
                    && (other_room > candidate_room || (other_room == candidate_room && other_begin < candidate_begin)))
                {
                    candidate_room = other_room;
                    candidate_begin = other_begin;
                }
            }
            int best_room = __shfl_sync(reduce_mask, candidate_room, 0, reduce_width);
            int best_begin = __shfl_sync(reduce_mask, candidate_begin, 0, reduce_width);
            if (best_begin != INT_MAX && best_room > 0)
            {
                chosen_level = level;
                chosen_begin = best_begin;
                chosen_size = size;
            }
        }
        if (chosen_level < 0)
        {
            if (lane == 0)
                frozen[owner] = 1;
            __syncwarp(reduce_mask);
            continue;
        }
        if (broadcasts >= kMaxBroadcasts)
            break;

        int b = broadcasts++;
        if (lane == 0)
        {
            p_expert[b] = expert;
            p_level[b] = chosen_level;
            p_begin[b] = chosen_begin;
            p_size[b] = chosen_size;
            p_owner[b] = owner;
            p_helper[b] = -1;
        }
        if (lane < ep)
            quota[b * ep + lane] = 0;
        __syncwarp(reduce_mask);
        if (lane == 0)
            quota[b * ep + owner] = mass;
        __syncwarp(reduce_mask);

        int remaining = give;
        unsigned visited = 0;
        for (int step = 0; step < chosen_size && remaining > 0; ++step)
        {
            bool can_receive = lane >= chosen_begin && lane < chosen_begin + chosen_size && lane != owner
                && !(visited & (1u << lane));
            int cold_load = can_receive ? load[lane] : INT_MAX;
            int cold_rank = can_receive ? lane : INT_MAX;
            for (int offset = reduce_start; offset; offset >>= 1)
            {
                int other_load = __shfl_down_sync(reduce_mask, cold_load, offset, reduce_width);
                int other_rank = __shfl_down_sync(reduce_mask, cold_rank, offset, reduce_width);
                if (lane + offset < reduce_width
                    && (other_load < cold_load || (other_load == cold_load && other_rank < cold_rank)))
                {
                    cold_load = other_load;
                    cold_rank = other_rank;
                }
            }
            int destination = __shfl_sync(reduce_mask, cold_rank, 0, reduce_width);
            if (destination == INT_MAX)
                break;
            visited |= 1u << destination;
            int room = target - load[destination];
            if (room <= 0)
                continue;
            int amount = imin(room, remaining);
            if (lane == 0)
            {
                quota[b * ep + owner] -= amount;
                quota[b * ep + destination] += amount;
            }
            remaining -= amount;
            __syncwarp(reduce_mask);
        }
        int moved = give - remaining;
        if (moved <= 0)
        {
            --broadcasts;
            if (lane == 0)
                frozen[owner] = 1;
            __syncwarp(reduce_mask);
            continue;
        }
        if (lane >= chosen_begin && lane < chosen_begin + chosen_size)
            ++recv_used[lane];
        if (lane == owner)
        {
            ++send_used[owner];
            load[owner] -= moved;
        }
        else if (lane >= chosen_begin && lane < chosen_begin + chosen_size)
        {
            load[lane] += quota[b * ep + lane];
        }
        __syncwarp(reduce_mask);
    }
    return broadcasts;
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

__device__ void halo_q_repair(int receiver, int donor, int broadcasts, int const* order, int const* host_mask,
    int const* counts, int const* p_expert, int* quota, int ep, int experts)
{
    int const lane = threadIdx.x & 31;
    if (broadcasts > 32)
    {
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
        return;
    }
    int b = lane < broadcasts ? order[lane] : -1;
    int pull = 0;
    int give = 0;
    if (b >= 0 && (host_mask[b] & (1u << receiver)) && (host_mask[b] & (1u << donor)))
    {
        int expert = p_expert[b];
        int cr = counts[receiver * experts + expert];
        int cd = counts[donor * experts + expert];
        int qr = quota[b * ep + receiver];
        int qd = quota[b * ep + donor];
        pull = imin(imax(0, cr - qr), imax(0, qd - cd));
        give = imax(0, qr - cr);
    }
    int delta = imin(warp_sum(pull), warp_sum(give));
    int pull_before = warp_exclusive_scan(pull);
    int give_before = warp_exclusive_scan(give);
    int pull_take = imin(pull, imax(0, delta - pull_before));
    int give_take = imin(give, imax(0, delta - give_before));
    if (b >= 0)
    {
        quota[b * ep + receiver] += pull_take - give_take;
        quota[b * ep + donor] += give_take - pull_take;
    }
    __syncwarp();
}

__device__ __forceinline__ int warp_sum8(int value)
{
    return __reduce_add_sync(0xffu, value);
}

__device__ __forceinline__ int warp_exclusive_scan8(int value)
{
    int sum = value;
    int const lane = threadIdx.x & 7;
    for (int offset = 1; offset < 8; offset <<= 1)
    {
        int other = __shfl_up_sync(0xffu, sum, offset, 8);
        if (lane >= offset)
            sum += other;
    }
    return sum - value;
}

__device__ __forceinline__ void halo_q_direction8(
    bool common, int receiver_count, int donor_count, int& receiver_quota, int& donor_quota)
{
    int pull = common ? imin(imax(0, receiver_count - receiver_quota), imax(0, donor_quota - donor_count)) : 0;
    int give = common ? imax(0, receiver_quota - receiver_count) : 0;
    int pull_total = warp_sum8(pull);
    int give_total = warp_sum8(give);
    int delta = imin(pull_total, give_total);
    if (!delta)
        return;
    int pull_before = warp_exclusive_scan8(pull);
    int give_before = warp_exclusive_scan8(give);
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
__device__ __forceinline__ void halo_q_pair8(
    int left, int right, int b, int hosts, int expert, int const* counts, int* quota, int ep, int experts)
{
    int const lane = threadIdx.x & 31;
    if (lane >= 8)
        return;
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

    halo_q_direction8(common, count_left, count_right, quota_left, quota_right);
    halo_q_direction8(common, count_right, count_left, quota_right, quota_left);

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

__device__ void run_halo_q(int const* counts, int* quota, int* plan, int broadcasts, int ep, int experts, int* ring,
    int* order, int* host_mask)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    if (ep <= kRegisterEp && broadcasts <= 8)
    {
        if (warp < ep / 2)
        {
            if (warp == 0)
                halo_q_build_order_and_masks(p_expert, p_begin, p_size, p_owner, order, host_mask, broadcasts);
            halo_q_active_barrier((ep / 2) * 32);
            /* Loop-invariant broadcast identity, read once after the barrier
             * that publishes order[] and host_mask[].  All ep / 2 participating
             * warps read the same lane-indexed entries. */
            int const my_b = lane < broadcasts ? order[lane] : -1;
            int const my_hosts = my_b >= 0 ? host_mask[my_b] : 0;
            int const my_expert = my_b >= 0 ? p_expert[my_b] : -1;
            if (broadcasts >= 2)
            {
                int const ring_size = ep - 1;
                /* Run two repair rounds unconditionally. A phase with no required
                 * repair is a no-op, so the plan equals running only the required
                 * repairs. */
                int const repair_rounds = 2;
                for (int round = 0; round < repair_rounds; ++round)
                {
                    /* At most kRegisterEp - 1 phases, unrolled; the guard is
                     * uniform across the block, so every pair warp meets
                     * every barrier it executes. */
#pragma unroll
                    for (int phase = 0; phase < kRegisterEp - 1; ++phase)
                    {
                        if (phase >= ep - 1)
                            break;
                        int right_index = warp == 0 ? 0 : ep - 1 - warp;
                        int right_position = right_index - phase;
                        if (right_position < 0)
                            right_position += ring_size;
                        int right = 1 + right_position;
                        int left = 0;
                        if (warp != 0)
                        {
                            int left_position = warp - phase;
                            if (left_position < 0)
                                left_position += ring_size;
                            left = 1 + left_position;
                        }
                        halo_q_pair8(left, right, my_b, my_hosts, my_expert, counts, quota, ep, experts);
                        /* Must stay unconditional.  Omitting it after a
                         * round's last phase races twice: warp 0 enters the
                         * next round's phase 0 while the last pair warp still
                         * writes the same quota[], and after the final phase
                         * the other pair warps leave run_halo_q while warp 0
                         * -- which only meets warp 8 at scheduler_tail_barrier
                         * -- walks into build_route_prefix and reads every
                         * quota[]. */
                        halo_q_active_barrier((ep / 2) * 32);
                    }
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
    (void) lane;
}

__device__ void color_and_publish_local(int* plan, int broadcasts, int ep, int helpers, int local_rank, int* used,
    int* out_ids, int* out_levels, int* out_owners, int* status, Epoch epoch)
{
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_level = plan_field(plan, kPlanLevel);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    int* p_helper = plan_field(plan, kPlanHelper);
    int const active_helpers = imin(helpers, kMaxBroadcasts);
    int const words = (active_helpers + 31) / 32;
    for (int rank = 0; rank < ep; ++rank)
        for (int word = 0; word < words; ++word)
            used[rank * kHelperMaskWords + word] = 0;
    for (int level = 0; level < hierarchy_depth(ep); ++level)
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
    unsigned long long spin_cycles, int capacity, int route_count)
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
        int* send_used = shared + offset;
        offset += ep;
        int* next_index = shared + offset;
        offset += ep;
        int* frozen = shared + offset;
        offset += ep;
        int* ring = shared + offset;
        offset += ep;
        int* order = shared + offset;
        offset += capacity;
        int* host_mask = shared + offset;

        if (!exchange_counts(peer_bases, partial, shared, ep, experts, local_rank, ctas, epoch, spin_cycles, status))
            return;
        if (uses_register_prepare(ep, experts))
            prepare_halo_m_register(counts, histogram, load, selected_flags, recv_used, send_used, next_index, frozen,
                ep, experts, helpers);
        else
            prepare_halo_m(counts, histogram, load, selected_flags, recv_used, send_used, next_index, frozen, ep,
                experts, helpers);

        int broadcasts = 0;
        if (threadIdx.x < (ep <= 8 ? 8 : 32))
        {
            // These short-circuit bounds keep helpers*ep <= 120.
            if (ep <= 8 && helpers <= 15 && helpers * ep <= experts)
            {
                broadcasts = plan_halo_m_register8(histogram, load, quota, selected_flags, plan, ep, helpers);
            }
            else
            {
                broadcasts = plan_halo_m_warp(histogram, load, quota, selected_flags, recv_used, send_used, next_index,
                    frozen, plan, ep, experts, helpers);
            }
            if (threadIdx.x == 0)
                plan[1] = broadcasts;
        }
        __syncthreads();
        broadcasts = plan[1];
        /* p_expert and plan[1] are produced by thread zero.  Publish this
         * placement-only epoch before HALO-Q so workers can construct their
         * selected-expert LUT while quota repair is still running. */
        if (ctas > 1 && threadIdx.x == 0)
        {
            store_release_device(plan + 0, epoch);
        }
        /* Small-EP repair excludes warp8 and can overlap coloring. Generic
         * repair first meets the coloring warp at its full-block barrier. */
        if (threadIdx.x == 8 * 32)
            color_and_publish_local(
                plan, broadcasts, ep, helpers, local_rank, recv_used, out_ids, out_levels, out_owners, status, epoch);
        run_halo_q(counts, quota, plan, broadcasts, ep, experts, ring, order, host_mask);
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
__host__ __device__ __forceinline__ int halo_q_shared_ints(int ep, int experts, int capacity, int threads)
{
    int const workerInts = threads / 32 * (experts + 1) + experts;
    int const plannerInts
        = ep * experts + experts + ep + capacity * ep + experts + (kHelperMaskWords + 4) * ep + 2 * capacity;
    return workerInts > plannerInts ? workerInts : plannerInts;
}

} // namespace

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_KERNEL_CUH
