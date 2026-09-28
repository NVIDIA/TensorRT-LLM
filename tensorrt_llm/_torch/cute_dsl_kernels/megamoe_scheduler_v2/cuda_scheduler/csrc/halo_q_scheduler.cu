/* Pure-CUDA fused physical-slot scheduler for supported architectures.
 *
 * One launch performs stable local-route ordinal counting, an in-kernel
 * peer-visible histogram exchange, either legacy GAR-N or HALO-M + HALO-Q,
 * mapped PlanChannel publication, and final physical-slot materialization.
 * Every CTA triggers CUDA Programmatic Dependent Launch at kernel entry.  The
 * remaining scheduler work stays on its small grid while an independent
 * same-stream successor may occupy unused SMs after every CTA has signalled.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>

#include <cuda/atomic>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "../../csrc/in_switch_copy/plan_policy.cuh"
#include <limits.h>
#include <stdint.h>

namespace
{

constexpr int kMaxEp = 32;
constexpr int kMaxExperts = 384;
constexpr int kMaxBroadcasts = kMaxExperts;
constexpr int kHelperMaskWords = (kMaxBroadcasts + 31) / 32;
constexpr int kMinGroup = 4;
constexpr int kSymPayloadOffset = 64;

enum PlanField
{
    kPlanExpert = 0,
    kPlanLevel = 1,
    kPlanGroupBegin = 2,
    kPlanGroupSize = 3,
    kPlanOwner = 4,
    kPlanHelper = 5,
};

// Adapt placement storage to the copy layer without exposing scheduler internals.
__device__ __forceinline__ sami_copy::PlacementView copy_placement_view(int const* plan)
{
    return {plan[1], kMaxBroadcasts, plan + 2 + kPlanExpert * kMaxBroadcasts, plan + 2 + kPlanLevel * kMaxBroadcasts,
        plan + 2 + kPlanGroupBegin * kMaxBroadcasts, plan + 2 + kPlanGroupSize * kMaxBroadcasts,
        plan + 2 + kPlanOwner * kMaxBroadcasts, plan + 2 + kPlanHelper * kMaxBroadcasts};
}

__device__ __forceinline__ int imin(int a, int b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ int imax(int a, int b)
{
    return a > b ? a : b;
}

__device__ __forceinline__ int fetch_add_release_device(int* address, int value)
{
    cuda::atomic_ref<int, cuda::thread_scope_device> atom(*address);
    return atom.fetch_add(value, cuda::memory_order_release);
}

__device__ __forceinline__ int load_acquire_device(int* address)
{
    cuda::atomic_ref<int, cuda::thread_scope_device> atom(*address);
    return atom.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ int load_relaxed_device(int* address)
{
    cuda::atomic_ref<int, cuda::thread_scope_device> atom(*address);
    return atom.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ void store_release_device(int* address, int value)
{
    cuda::atomic_ref<int, cuda::thread_scope_device> atom(*address);
    atom.store(value, cuda::memory_order_release);
}

__device__ __forceinline__ int load_acquire_system(int* address)
{
    cuda::atomic_ref<int, cuda::thread_scope_system> atom(*address);
    return atom.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ int load_relaxed_system(int* address)
{
    cuda::atomic_ref<int, cuda::thread_scope_system> atom(*address);
    return atom.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ void store_release_system(int* address, int value)
{
    cuda::atomic_ref<int, cuda::thread_scope_system> atom(*address);
    atom.store(value, cuda::memory_order_release);
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

__device__ int grid_rendezvous(int* sync, int blocks, unsigned long long spin_cycles, int* status)
{
    __shared__ int generation;
    __shared__ int success;
    /* The block barrier orders every writer's stores before thread zero's
     * device-scope release advertises this CTA. */
    __syncthreads();
    if (threadIdx.x == 0)
    {
        success = 1;
        /* One monotonic counter is graph-replay safe and exactly mirrors the
         * reset-free rendezvous. */
        int old = fetch_add_release_device(sync, 1);
        int target = (old / blocks + 1) * blocks;
        generation = target / blocks;
        unsigned long long start = clock64();
        int observed = load_relaxed_device(sync);
        while (observed < target)
        {
            if (clock64() - start > spin_cycles)
            {
                atomicExch(status + 1, generation);
                success = 0;
                break;
            }
            observed = load_relaxed_device(sync);
        }
        if (observed >= target)
            (void) load_acquire_device(sync);
    }
    __syncthreads();
    return success ? generation : -1;
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
     * parallelism. The binning is unchanged, and atomicAdd regrouping is
     * safe because integer addition commutes. */
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
 * cross-rank exchange and planner.  route_aux[0] is a reset-free worker
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

    int* worker_totals = route_aux + 1;
    for (int bin = tid; bin < bins; bin += blockDim.x)
    {
        int total = 0;
        for (int local_warp = 0; local_warp < warps; ++local_warp)
            total += shared[local_warp * bins + bin];
        worker_totals[worker * bins + bin] = total;
    }

    /* The first all-CTA rendezvous guarantees every worker is resident, so a
     * worker-only global barrier cannot wait on an unscheduled CTA. */
    if (grid_rendezvous(route_aux, workers, spin_cycles, status) < 0)
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

template <int FixedEp>
__device__ bool exchange_counts(unsigned long long const* peer_bases, int const* partial, int* shared, int ep,
    int expert_count, int local_rank, int ctas, int epoch, unsigned long long spin_cycles, int* status)
{
    int const exchange_ep = FixedEp > 0 ? FixedEp : ep;
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
    for (int destination = warp; destination < exchange_ep; destination += blockDim.x >> 5)
    {
        int* peer = reinterpret_cast<int*>(peer_bases[destination]);
        int* row = peer + kSymPayloadOffset + (parity * exchange_ep + local_rank) * expert_count;
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
    if (tid < exchange_ep)
    {
        unsigned long long start = clock64();
        int observed = load_relaxed_system(local + tid);
        while (observed < epoch)
        {
            if (clock64() - start > spin_cycles)
            {
                atomicExch(status + 0, epoch);
                atomicExch(&wait_failed, 1);
                break;
            }
            observed = load_relaxed_system(local + tid);
        }
        if (observed >= epoch)
            (void) load_acquire_system(local + tid);
    }
    __syncthreads();
    if (wait_failed)
        return false;
    int const* payload = local + kSymPayloadOffset + parity * exchange_ep * expert_count;
    int const total = exchange_ep * expert_count;
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

__device__ void prepare_halo_m_sparse8(int const* counts, int* histogram, int* load, int* candidates, int experts)
{
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    if (warp >= 8)
        return;
    int const home = experts / 8;
    int local_index[2];
    int local_mass[2];
    int rank_load = 0;
#pragma unroll
    for (int round = 0; round < 2; ++round)
    {
        int local = lane + 32 * round;
        int mass = -1;
        if (local < home)
        {
            mass = 0;
            int expert = warp * home + local;
#pragma unroll
            for (int source = 0; source < 8; ++source)
                mass += counts[source * experts + expert];
            rank_load += mass;
        }
        local_index[round] = local;
        local_mass[round] = mass;
    }
    rank_load = __reduce_add_sync(0xffffffffu, rank_load);
    if (lane == 0)
        load[warp] = rank_load;
    for (int slot = 0; slot < 4; ++slot)
    {
        int best_mass = -1;
        int best_local = INT_MAX;
#pragma unroll
        for (int round = 0; round < 2; ++round)
        {
            int mass = local_mass[round];
            int local = local_index[round];
            if (mass > best_mass || (mass == best_mass && local < best_local))
            {
                best_mass = mass;
                best_local = local;
            }
        }
        int warp_mass = __reduce_max_sync(0xffffffffu, best_mass);
        int winner = __reduce_min_sync(0xffffffffu, best_mass == warp_mass ? best_local : INT_MAX);
        if (lane == 0)
        {
            int expert = warp * home + winner;
            candidates[warp * 4 + slot] = expert;
            histogram[expert] = warp_mass;
        }
#pragma unroll
        for (int round = 0; round < 2; ++round)
            if (local_index[round] == winner)
                local_mass[round] = -1;
    }
    asm volatile("bar.sync 4, 256;" ::: "memory");
}

/* Build HALO-M's dense histogram, rank loads and per-owner hot lists with one
 * warp per rank.  The former thread-0 implementation dominated the kernel. */
template <int FixedEp, int FixedHelpers>
__device__ void prepare_halo_m(int const* counts, int* histogram, int* load, int* candidates, int* recv_used,
    int* send_used, int* next_index, int* frozen, int ep, int experts, int helpers)
{
    int const prepare_ep = FixedEp > 0 ? FixedEp : ep;
    int const prepare_helpers = imin(FixedHelpers > 0 ? FixedHelpers : helpers, experts / prepare_ep);
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const home = experts / prepare_ep;
    for (int expert = tid; expert < experts; expert += blockDim.x)
    {
        int mass = 0;
#pragma unroll
        for (int rank = 0; rank < prepare_ep; ++rank)
            mass += counts[rank * experts + expert];
        histogram[expert] = mass;
    }
    __syncthreads();

    for (int owner = warp; owner < prepare_ep; owner += blockDim.x >> 5)
    {
        int rank_load = 0;
        for (int local = lane; local < home; local += 32)
            rank_load += histogram[owner * home + local];
        rank_load = __reduce_add_sync(0xffffffffu, rank_load);
        if (lane == 0)
        {
            load[owner] = rank_load;
            recv_used[owner] = 0;
            send_used[owner] = 0;
            next_index[owner] = 0;
            frozen[owner] = 0;
        }

        /* EP2 is the largest home: 192 experts, six entries per lane. */
        constexpr int rounds = FixedEp == 8 ? 2 : 6;
        int local_index[rounds];
        int local_mass[rounds];
#pragma unroll
        for (int round = 0; round < rounds; ++round)
        {
            int local = lane + 32 * round;
            local_index[round] = local;
            local_mass[round] = local < home ? histogram[owner * home + local] : -1;
        }
        for (int slot = 0; slot < prepare_helpers; ++slot)
        {
            int best_mass = -1;
            int best_local = INT_MAX;
#pragma unroll
            for (int round = 0; round < rounds; ++round)
            {
                int mass = local_mass[round];
                int local = local_index[round];
                if (mass > best_mass || (mass == best_mass && local < best_local))
                {
                    best_mass = mass;
                    best_local = local;
                }
            }
            int warp_mass = __reduce_max_sync(0xffffffffu, best_mass);
            int winner = __reduce_min_sync(0xffffffffu, best_mass == warp_mass ? best_local : INT_MAX);
            if (lane == 0)
                candidates[owner * prepare_helpers + slot] = owner * home + winner;
#pragma unroll
            for (int round = 0; round < rounds; ++round)
                if (local_index[round] == winner)
                    local_mass[round] = -1;
        }
    }
    __syncthreads();
}

/* EP<=8 fast path: every participating lane owns one rank's complete
 * mutable HALO-M state.  Group and destination decisions use only shuffles,
 * avoiding repeated shared-memory rank-state traffic and warp barriers. */
template <int FixedEp, int FixedHelpers, bool NarrowKeys>
__device__ __forceinline__ int plan_halo_m_register8_impl(
    int const* histogram, int lane_load, int total, int* quota, int const* candidates, int* plan, int ep, int helpers)
{
    int const plan_ep = FixedEp > 0 ? FixedEp : ep;
    int const plan_helpers = FixedHelpers > 0 ? FixedHelpers : helpers;
    int const lane = threadIdx.x & 31;
    constexpr unsigned mask = 0xffu;
    int lane_recv = 0;
    int lane_send = 0;
    int lane_next = 0;
    int lane_frozen = 0;

    int const target = total / plan_ep + (total % plan_ep != 0);
    int const depth = hierarchy_depth(plan_ep);
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_level = plan_field(plan, kPlanLevel);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    int* p_helper = plan_field(plan, kPlanHelper);
    int broadcasts = 0;

    int trips = 0;
    for (int guard = 0; guard < plan_ep * (plan_helpers + 2); ++guard)
    {
        ++trips;
        bool const eligible = lane < plan_ep && !lane_frozen && lane_load > target;
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
        if (owner_send >= plan_helpers || owner_next >= plan_helpers)
        {
            if (lane == owner)
                lane_frozen = 1;
            continue;
        }

        /* owner and owner_next are now warp-uniform, so every lane can read
         * candidates[] and histogram[] directly.  Both are written by
         * prepare_halo_m_sparse8 before its bar.sync and never inside this
         * loop, so a uniform load returns exactly what the owner would have
         * broadcast.  The increment stays owner-only and stays ahead of the
         * mass test, as before. */
        int const expert = candidates[owner * plan_helpers + owner_next];
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

        int positive_target = imax(target, 1);
        int need = imax(1, mass / positive_target + (mass % positive_target != 0));
        int wanted = depth - 1;
        for (int level = depth - 1; level >= 0; --level)
        {
            if (plan_ep / (1 << level) >= need)
            {
                wanted = level;
                break;
            }
        }

        int chosen_level = -1;
        int chosen_begin = -1;
        int chosen_size = 0;
        if constexpr (FixedEp == 8 && FixedHelpers == 4)
        {
            int slack = lane != owner ? imax(0, target - lane_load) : 0;
            /* Two independent full-mask sums instead of one lane-varying-mask
             * reduce followed by two broadcasts.  Integer addition is exact, so
             * masking the addend gives the same half sums; the two reduces do
             * not depend on each other, so the chain is depth 1 instead of 3,
             * and the redux.sync no longer runs with a per-lane membermask. */
            int room0 = __reduce_add_sync(mask, lane < 4 ? slack : 0);
            int room4 = __reduce_add_sync(mask, lane >= 4 ? slack : 0);
            unsigned receive_mask = __ballot_sync(mask, lane_recv < plan_helpers);
            bool valid0 = (receive_mask & 0x0fu) == 0x0fu && room0 > 0;
            bool valid4 = (receive_mask & 0xf0u) == 0xf0u && room4 > 0;
            bool want_half = mass <= 4 * target;
            if (want_half && (valid0 || valid4))
            {
                chosen_level = 1;
                chosen_begin = valid0 && (!valid4 || room0 >= room4) ? 0 : 4;
                chosen_size = 4;
            }
            else if (receive_mask == mask && room0 + room4 > 0)
            {
                chosen_level = 0;
                chosen_begin = 0;
                chosen_size = 8;
            }
        }
        else
        {
            for (int level = wanted; level >= 0 && chosen_level < 0; --level)
            {
                int size = plan_ep / (1 << level);
                bool active = lane < plan_ep;
                int begin = (lane / size) * size;
                unsigned group_mask = rank_group_mask(begin, size);
                int local_room = active && lane != owner ? imax(0, target - lane_load) : 0;
                int room = 0;
                if (active)
                    room = __reduce_add_sync(group_mask, local_room);
                unsigned receive_mask = __ballot_sync(mask, active && lane_recv < plan_helpers);
                bool group_leader = active && lane == begin && (receive_mask & group_mask) == group_mask;
                unsigned leader_mask = __ballot_sync(mask, group_leader);
                int best_room = -1;
                int best_begin = INT_MAX;
                if (leader_mask)
                {
                    int const first_leader = __ffs(leader_mask) - 1;
                    if constexpr (NarrowKeys)
                    {
                        unsigned key = 0;
                        if (group_leader)
                            key = __reduce_max_sync(
                                leader_mask, (static_cast<unsigned>(room) << 4) | (15u - static_cast<unsigned>(begin)));
                        key = __shfl_sync(mask, key, first_leader, 8);
                        best_room = static_cast<int>(key >> 4);
                        best_begin = 15 - static_cast<int>(key & 15u);
                    }
                    else
                    {
                        if (group_leader)
                        {
                            best_room = __reduce_max_sync(leader_mask, room);
                            best_begin = __reduce_min_sync(leader_mask, room == best_room ? begin : INT_MAX);
                        }
                        best_room = __shfl_sync(mask, best_room, first_leader, 8);
                        best_begin = __shfl_sync(mask, best_begin, first_leader, 8);
                    }
                }
                if (best_begin != INT_MAX && best_room > 0)
                {
                    chosen_level = level;
                    chosen_begin = best_begin;
                    chosen_size = size;
                }
            }
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
        if (lane < plan_ep)
            quota[b * plan_ep + lane] = lane_quota;

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
 * This guard keeps owner/destination loads and group-room keys representable.
 * Keep the wide path for large valid int32 inputs without putting a runtime
 * representation branch on each iteration of either planning loop. */
template <int FixedEp, int FixedHelpers>
__device__ __forceinline__ int plan_halo_m_register8(
    int const* histogram, int const* load, int* quota, int const* candidates, int* plan, int ep, int helpers)
{
    int const plan_ep = FixedEp > 0 ? FixedEp : ep;
    int const lane = threadIdx.x & 31;
    int const lane_load = lane < plan_ep ? load[lane] : 0;
    int const total = __reduce_add_sync(0xffu, lane_load);
    if (total <= 0x0fffffff)
        return plan_halo_m_register8_impl<FixedEp, FixedHelpers, true>(
            histogram, lane_load, total, quota, candidates, plan, ep, helpers);
    return plan_halo_m_register8_impl<FixedEp, FixedHelpers, false>(
        histogram, lane_load, total, quota, candidates, plan, ep, helpers);
}

/* Exact HALO-M state machine executed by warp 0.  Each lane owns one rank;
 * warp reductions replace all serial rank/group/destination scans. */
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
 * here put a four-deep dependent load chain (order[lane] -> b -> host_mask[b]
 * and the GLOBAL p_expert[b] -> expert -> counts[...]) at the head of all
 * fourteen phases. */
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
 * This replaces serial insertion-sort and mask-construction loops while
 * preserving the same stable order.
 *
 * order[]:  lane b's position is the number of broadcasts that sort before it.
 *           The `j < lane` tiebreak makes this the same stable order the
 *           insertion sort produced (its shift test is a strict `>`), so the
 *           result is bit-identical, not merely equivalent.
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

template <int FixedEp>
__device__ void run_halo_q(int const* counts, int* quota, int* plan, int broadcasts, int ep, int experts, int* ring,
    int* order, int* host_mask)
{
    int const plan_ep = FixedEp > 0 ? FixedEp : ep;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int* p_expert = plan_field(plan, kPlanExpert);
    int* p_begin = plan_field(plan, kPlanGroupBegin);
    int* p_size = plan_field(plan, kPlanGroupSize);
    int* p_owner = plan_field(plan, kPlanOwner);
    if (plan_ep <= 8 && broadcasts <= 8)
    {
        if (warp < plan_ep / 2)
        {
            if (warp == 0)
                halo_q_build_order_and_masks(p_expert, p_begin, p_size, p_owner, order, host_mask, broadcasts);
            halo_q_active_barrier((plan_ep / 2) * 32);
            /* Loop-invariant broadcast identity, read once after the barrier
             * that publishes order[] and host_mask[].  All four participating
             * warps read the same lane-indexed entries. */
            int const my_b = lane < broadcasts ? order[lane] : -1;
            int const my_hosts = my_b >= 0 ? host_mask[my_b] : 0;
            int const my_expert = my_b >= 0 ? p_expert[my_b] : -1;
            if (broadcasts >= 2)
            {
                int const ring_size = plan_ep - 1;
                /* Use two unconditional repair rounds. Phases with no required repair
                 * are no-ops, so this preserves the conditional algorithm's result. */
                int const repair_rounds = 2;
                for (int round = 0; round < repair_rounds; ++round)
                {
                    for (int phase = 0; phase < plan_ep - 1; ++phase)
                    {
                        int right_index = warp == 0 ? 0 : plan_ep - 1 - warp;
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
                        halo_q_pair8(left, right, my_b, my_hosts, my_expert, counts, quota, plan_ep, experts);
                        if constexpr (FixedEp == 8)
                        {
                            /* Must stay unconditional.  Omitting it after a
                             * round's last phase races twice: warp 0 enters the
                             * next round's phase 0 while warp 3 still writes the
                             * same quota[], and after the final phase warps 1-3
                             * leave run_halo_q while warp 0 -- which only meets
                             * warp 8 at scheduler_tail_barrier -- walks into
                             * build_route_prefix and reads every quota[]. */
                            halo_q_active_barrier(128);
                        }
                        else
                        {
                            halo_q_active_barrier((plan_ep / 2) * 32);
                        }
                    }
                }
            }
        }
        /* The unconditional repair rounds above replace the conditional scan. */
        return;
    }
    if (warp == 0)
        halo_q_build_order_and_masks(p_expert, p_begin, p_size, p_owner, order, host_mask, broadcasts);
    if (tid < plan_ep - 1)
        ring[tid] = tid + 1;
    __syncthreads();
    if (broadcasts < 2)
        return;

    for (int round = 0; round < 2; ++round)
    {
        for (int phase = 0; phase < plan_ep - 1; ++phase)
        {
            if (warp < plan_ep / 2)
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
                    right = ring[plan_ep - 1 - warp];
                }
                halo_q_repair(left, right, broadcasts, order, host_mask, counts, p_expert, quota, plan_ep, experts);
                halo_q_repair(right, left, broadcasts, order, host_mask, counts, p_expert, quota, plan_ep, experts);
            }
            __syncthreads();
            if (tid == 0)
            {
                int last = ring[plan_ep - 2];
                for (int index = plan_ep - 2; index > 0; --index)
                    ring[index] = ring[index - 1];
                ring[0] = last;
            }
            __syncthreads();
        }
    }
    (void) lane;
}

__device__ void color_and_publish_local(int* plan, int broadcasts, int ep, int helpers, int local_rank, int* used,
    int* out_ids, int* out_levels, int* out_owners, int* status, int epoch)
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
            unsigned unavailable[kHelperMaskWords] = {0};
            for (int rank = p_begin[b]; rank < p_begin[b] + p_size[b]; ++rank)
                for (int word = 0; word < words; ++word)
                    unavailable[word] |= static_cast<unsigned>(used[rank * kHelperMaskWords + word]);
            int helper = -1;
            for (int slot = 0; slot < active_helpers; ++slot)
                if (!(unavailable[slot / 32] & (1u << (slot % 32))))
                {
                    helper = slot;
                    break;
                }
            if (helper < 0)
            {
                atomicExch(status + 4, epoch);
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
            atomicExch(status + 4, epoch);
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
        for (int rank = 0; rank < local_rank; ++rank)
        {
            int diagonal = imin(counts[rank * experts + expert], quota[b * ep + rank]);
            source_before += counts[rank * experts + expert] - diagonal;
        }
        int source_count = counts[local_rank * experts + expert];
        int source_diagonal = imin(source_count, quota[b * ep + local_rank]);
        int residual_supply = source_count - source_diagonal;
        /* The loop used to run to destination == ep.  route_destination only
         * ever loads prefix[destination + step] under `candidate < ep`, so the
         * ep entry is never read, and the accumulation at ep-1 fed nothing but
         * that dead entry.  Both are gone; entry 0 is still written because it
         * costs one store and leaving it stale would depend on nobody ever
         * reading it, which is a weaker guarantee than writing the right value. */
        int destination_before = 0;
        for (int destination = 0; destination < ep; ++destination)
        {
            int overlap = destination_before - source_before;
            overlap = imax(0, imin(residual_supply, overlap));
            int diagonal_prefix = local_rank < destination ? source_diagonal : 0;
            prefix[b * (ep + 1) + destination] = diagonal_prefix + overlap;
            if (destination + 1 < ep)
            {
                int diagonal = imin(counts[destination * experts + expert], quota[b * ep + destination]);
                destination_before += quota[b * ep + destination] - diagonal;
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

__device__ bool wait_for_plan_epoch(int* marker, int epoch, unsigned long long spin_cycles, int* status, int* ready)
{
    if (threadIdx.x == 0)
    {
        *ready = 1;
        unsigned long long start = clock64();
        int observed = load_relaxed_device(marker);
        while (observed < epoch)
        {
            if (clock64() - start > spin_cycles)
            {
                atomicExch(status + 2, epoch);
                *ready = 0;
                break;
            }
            observed = load_relaxed_device(marker);
        }
        if (observed >= epoch)
            (void) load_acquire_device(marker);
    }
    __syncthreads();
    return *ready != 0;
}

/* Prefix entries are monotonic.  Binary lifting returns the same number of
 * thresholds <= ordinal as the old linear scan, including repeated entries. */
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
    int route_count, int experts, int worker, int workers, int epoch)
{
    extern __shared__ int shared[];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const warps = blockDim.x >> 5;
    int const bins = experts + 1;
    int* expert_lut = worker_expert_lut(experts);
    int* hot_indices = route_aux + 1 + workers * bins;

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
            atomicExch(status + 3, epoch);
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
    int* route_aux, int* status, int route_count, int ep, int experts, int helpers, int worker, int workers, int epoch)
{
    extern __shared__ int shared[];
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const warps = blockDim.x >> 5;
    int const bins = experts + 1;
    int* expert_lut = worker_expert_lut(experts);
    int* hot_indices = route_aux + 1 + workers * bins;
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
    for (int offset = lane; offset < hot_count; offset += 32)
    {
        int index = hot_indices[begin + offset];
        int expert = routes[index];
        int b = expert_lut[expert];
        int ordinal = output[index] + shared[warp * bins + expert];
        int const* prefix = route_prefix + b * (ep + 1);
        int destination = route_destination(prefix, ordinal, ep);
        int slot = -1;
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
            atomicExch(status + 5, epoch);
        }
        output[index] = slot;
    }
}

__device__ void materialize_worker_routes(int const* routes, int* output, int const* route_prefix, int const* plan,
    int* status, int route_count, int ep, int experts, int helpers, int worker, int workers, int epoch)
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
                    atomicExch(status + 5, epoch);
                }
            }
        }
        else if (expert != -1)
        {
            atomicExch(status + 3, epoch);
        }
        output[index] = slot;
    }
}

template <int FixedEp, int FixedExperts, int FixedHelpers>
__global__ __launch_bounds__(512, 1) void halo_q_scheduler_kernel(int const* routes, int* out_slots, int* out_ids,
    int* out_levels, int* out_owners, unsigned long long const* peer_bases, int* status, int* partial, int* route_aux,
    int* grid_sync, int* plan, int* route_prefix, int* plan_channel, int runtime_ep, int runtime_experts,
    int runtime_helpers, int local_rank, int route_capacity, int ctas, int enable_pdl, unsigned long long spin_cycles,
    int plan_abi_version, int plan_channel_words, int route_features, int capacity, int route_count)
{
    int const ep = FixedEp > 0 ? FixedEp : runtime_ep;
    int const experts = FixedExperts > 0 ? FixedExperts : runtime_experts;
    int const helpers = FixedHelpers > 0 ? FixedHelpers : runtime_helpers;
    (void) plan_channel_words; // Validated by the host launch wrapper.
    /* The scheduler grid is deliberately residency-safe (one CTA per SM and
     * CTA count <= SM count), so signal as soon as every CTA starts.  The
     * independent same-stream quantizer may then occupy all unused SMs while
     * these CTAs retain their own SMs for the complete scheduler lifetime. */
    if (enable_pdl && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
    // Only the caller's valid prefix is readable. Keep the public output at
    // its configured capacity, clearing stale routes inside this same kernel.
    // Empty ranks still execute every collective; no route load occurs at N=0.
    for (int index = route_count + blockIdx.x * blockDim.x + threadIdx.x; index < route_capacity;
         index += gridDim.x * blockDim.x)
        out_slots[index] = -1;
    local_histogram_pass(routes, partial, route_count, experts, ctas);
    int const epoch = grid_rendezvous(grid_sync, ctas, spin_cycles, status);
    if (epoch < 0)
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

        bool exchange_ok;
        if (ep == 8)
            exchange_ok = exchange_counts<8>(
                peer_bases, partial, shared, ep, experts, local_rank, ctas, epoch, spin_cycles, status);
        else
            exchange_ok = exchange_counts<0>(
                peer_bases, partial, shared, ep, experts, local_rank, ctas, epoch, spin_cycles, status);
        if (!exchange_ok)
            return;
        /* EP8 keeps only its candidate masses; other EPs retain the dense
         * histogram required by the generic planner path. */
        if (ep == 8 && helpers == 4 && experts >= 32)
        {
            prepare_halo_m_sparse8(counts, histogram, load, selected_flags, experts);
        }
        else
        {
            prepare_halo_m<0, 0>(counts, histogram, load, selected_flags, recv_used, send_used, next_index, frozen, ep,
                experts, helpers);
        }

        int broadcasts = 0;
        if (threadIdx.x < (ep <= 8 ? 8 : 32))
        {
            // These short-circuit bounds keep helpers*ep <= 120.
            if (ep <= 8 && helpers <= 15 && helpers * ep <= experts)
            {
                if (ep == 8 && helpers == 4)
                {
                    broadcasts = plan_halo_m_register8<8, 4>(histogram, load, quota, selected_flags, plan, ep, helpers);
                }
                else if (ep == 4 && helpers == 4)
                {
                    broadcasts = plan_halo_m_register8<4, 4>(histogram, load, quota, selected_flags, plan, ep, helpers);
                }
                else
                {
                    broadcasts = plan_halo_m_register8<0, 0>(histogram, load, quota, selected_flags, plan, ep, helpers);
                }
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
        if (ep == 8)
            run_halo_q<8>(counts, quota, plan, broadcasts, ep, experts, ring, order, host_mask);
        else
            run_halo_q<0>(counts, quota, plan, broadcasts, ep, experts, ring, order, host_mask);
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
            if (warp == 0)
            {
                /* Release the worker CTAs here rather than after the channel
                 * publish.  They gate on grid_sync+1 and consume route_prefix,
                 * which is complete above; plan_channel is consumed only by the
                 * copy stack, which acquires on the channel's own epoch word
                 * written with store_release_system below.  Publishing after
                 * the release lets the workers coloring pass -- the work that
                 * actually determines when the kernel ends -- overlap the whole
                 * publish instead of waiting behind it. */
                if (threadIdx.x == 0)
                    store_release_device(grid_sync + 1, epoch);
                sami_copy::publish_plan_channel_warp(plan_channel, out_ids, out_levels, out_owners,
                    copy_placement_view(plan), helpers, ep, local_rank, experts, plan_abi_version, route_features);
                __syncwarp();
            }
            return;
        }
        __syncthreads();
        build_route_prefix(counts, quota, route_prefix, plan, broadcasts, ep, experts, local_rank, blockDim.x);
        __syncthreads();
        sami_copy::publish_plan_channel_warp(plan_channel, out_ids, out_levels, out_owners, copy_placement_view(plan),
            helpers, ep, local_rank, experts, plan_abi_version, route_features);
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
        if (!wait_for_plan_epoch(grid_sync + 1, epoch, spin_cycles, status, &plan_ready))
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

int raise_cuda(cudaError_t error, char const* operation)
{
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s", operation, cudaGetErrorString(error));
    return -1;
}

bool valid_plan_channel_launch(
    unsigned long long pointer, int abi_version, int words, int features, int ep, int helpers)
{
    if (!pointer)
        return abi_version == 0 && words == 0 && features == 0;
    if (pointer % 16 || helpers <= 0)
        return false;
    const int64_t stride = (static_cast<int64_t>(helpers) + 3) & ~int64_t(3);
    const int64_t expected = 4 + 7 * stride;
    return abi_version == 6 && ep >= 2 && ep <= kMaxEp && ep % 2 == 0 && (features == 0 || features == 1)
        && expected <= INT_MAX && words == expected;
}

PyObject* launch(PyObject*, PyObject* args)
{
    unsigned long long routes, out_slots, out_ids, out_levels, out_owners;
    unsigned long long peer_bases, status, partial, route_aux, grid_sync;
    unsigned long long plan, route_prefix, plan_channel;
    unsigned long long stream, spin_cycles;
    int ep, experts, helpers, local_rank, route_count, ctas, threads;
    int enable_pdl;
    int plan_abi_version, plan_channel_words, route_features;
    int valid_route_count = -1;
    if (!PyArg_ParseTuple(args, "KKKKKKKKKKKKKiiiiiiiiKKiii|i", &routes, &out_slots, &out_ids, &out_levels, &out_owners,
            &peer_bases, &status, &partial, &route_aux, &grid_sync, &plan, &route_prefix, &plan_channel, &ep, &experts,
            &helpers, &local_rank, &route_count, &ctas, &threads, &enable_pdl, &stream, &spin_cycles, &plan_abi_version,
            &plan_channel_words, &route_features, &valid_route_count))
        return nullptr;
    // Omitted length preserves the existing full-capacity launch ABI.
    if (PyTuple_GET_SIZE(args) == 26)
        valid_route_count = route_count;
    if ((!routes && valid_route_count != 0) || !out_slots || !out_ids || !out_levels || !out_owners || !peer_bases
        || !status || !partial || !route_aux || !grid_sync || !plan || !route_prefix || ep < 2 || ep > kMaxEp || ep % 2
        || experts <= 0 || experts > kMaxExperts || experts % ep || helpers <= 0
        || helpers > INT_MAX / ep - experts / ep || local_rank < 0 || local_rank >= ep || valid_route_count < 0
        || valid_route_count > route_count || route_count <= 0 || route_count > INT_MAX / ep
        || route_count > INT_MAX - ctas || ctas <= 0 || ctas > 128 || threads != 512
        || (enable_pdl != 0 && enable_pdl != 1))
    {
        PyErr_SetString(PyExc_ValueError, "invalid pure-CUDA scheduler launch");
        return nullptr;
    }
    if (!valid_plan_channel_launch(plan_channel, plan_abi_version, plan_channel_words, route_features, ep, helpers))
    {
        PyErr_SetString(PyExc_ValueError, "invalid private PlanChannel launch ABI");
        return nullptr;
    }
    int const warps = threads / 32;
    int pass_ints = warps * (experts + 1);
    int worker_ints = pass_ints + experts;
    int capacity = broadcast_capacity(ep, experts, helpers);
    int planner_ints
        = ep * experts + experts + ep + capacity * ep + experts + (kHelperMaskWords + 4) * ep + 2 * capacity;
    size_t shared_bytes = static_cast<size_t>(worker_ints > planner_ints ? worker_ints : planner_ints) * sizeof(int);
    // Specialize only the two common geometries. Every other supported
    // EP/E/helper combination retains the identical generic kernel.
    void const* kernel = reinterpret_cast<void const*>(&halo_q_scheduler_kernel<0, 0, 0>);
    if (experts == 384 && helpers == 4)
    {
        if (ep == 8)
            kernel = reinterpret_cast<void const*>(&halo_q_scheduler_kernel<8, 384, 4>);
        else if (ep == 4)
            kernel = reinterpret_cast<void const*>(&halo_q_scheduler_kernel<4, 384, 4>);
    }
    if (shared_bytes > 48 * 1024)
    {
        cudaError_t configured
            = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
        if (configured != cudaSuccess)
        {
            raise_cuda(configured, "cudaFuncSetAttribute(scheduler dynamic shared memory)");
            return nullptr;
        }
    }
    dim3 grid(ctas, 1, 1);
    dim3 block(threads, 1, 1);
    void* parameters[] = {
        &routes,
        &out_slots,
        &out_ids,
        &out_levels,
        &out_owners,
        &peer_bases,
        &status,
        &partial,
        &route_aux,
        &grid_sync,
        &plan,
        &route_prefix,
        &plan_channel,
        &ep,
        &experts,
        &helpers,
        &local_rank,
        &route_count,
        &ctas,
        &enable_pdl,
        &spin_cycles,
        &plan_abi_version,
        &plan_channel_words,
        &route_features,
        &capacity,
        &valid_route_count,
    };
    cudaError_t error = cudaLaunchKernel(
        kernel, grid, block, parameters, shared_bytes, reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream)));
    if (error != cudaSuccess)
    {
        raise_cuda(error, "cudaLaunchKernel(halo_q_scheduler_kernel)");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"launch", launch, METH_VARARGS, "Launch the fused pure-CUDA scheduler."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "megamoe_halo_q_cuda",
    "Pure-CUDA legacy/HALO-Q physical-slot scheduler.",
    -1,
    methods,
};

} // namespace

PyMODINIT_FUNC PyInit_megamoe_halo_q_cuda(void)
{
    PyObject* result = PyModule_Create(&module);
    if (!result)
        return nullptr;
    PyModule_AddIntConstant(result, "ABI_VERSION", 5);
    PyModule_AddIntConstant(result, "MAX_BROADCASTS", kMaxBroadcasts);
    return result;
}
