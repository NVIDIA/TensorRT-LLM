#pragma once
#include <cuda/atomic>
#include <cuda_runtime.h>

// Copy-owned GPU policy adapter. The scheduler supplies only placement geometry.
// Fusing this adapter into its publication tail avoids another kernel launch or
// any CPU policy work. Without a bound copy channel, the adapter does nothing.
namespace sami_copy
{
constexpr int kMaxEp = 32;
constexpr int kMinGroup = 4;

struct PlacementView
{
    int count, capacity;
    int const *experts, *levels, *begins, *sizes, *owners, *helpers;
};

__device__ __forceinline__ int imin(int a, int b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ int imax(int a, int b)
{
    return a > b ? a : b;
}

__device__ __forceinline__ bool group_contains(int rank, int begin, int size)
{
    return rank >= begin && rank < begin + size;
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

__device__ __forceinline__ void store_release_system(int* address, int value)
{
    cuda::atomic_ref<int, cuda::thread_scope_system> atom(*address);
    atom.store(value, cuda::memory_order_release);
}

/* ABI6 publishes explicit per-slot modes and bounded owner jobs.  All scans
 * cover the complete global plan, including broadcasts beyond one warp.  Only
 * the releasing lane stores mapped words, preserving the SYS release contract. */
template <bool PublishHost = true>
__device__ void publish_plan_channel(int* channel, int const* ids, int const* levels, int const* owners,
    PlacementView const& plan, int helpers, int ep, int local_rank, int expert_count, int plan_abi_version,
    int route_features)
{
    if (!channel || threadIdx.x != 0)
        return;
    int const stride = (helpers + 3) & ~3;
    int const broadcasts = plan.count;
    int const* p_expert = plan.experts;
    int const* p_level = plan.levels;
    int const* p_begin = plan.begins;
    int const* p_size = plan.sizes;
    int const* p_owner = plan.owners;
    int const* p_helper = plan.helpers;
    int owner_count[kMaxEp] = {0};
    int status = plan_abi_version != 6 || (route_features & ~1) || broadcasts < 0 || broadcasts > plan.capacity;
    int global_max_send = 0;
    int common_size = broadcasts > 0 ? p_size[0] : 0;
    int foreign_begin = -1, target_active = 0;
    bool common_proper_size = common_size > 0 && common_size < ep;
    bool one_foreign_target = true;
    int const home = expert_count / ep;
    for (int b = 0; b < broadcasts && !status; ++b)
    {
        int const level = p_level[b];
        int const size = level >= 0 && level < hierarchy_depth(ep) ? ep / (1 << level) : 0;
        if (p_expert[b] < 0 || p_expert[b] >= expert_count || p_owner[b] != p_expert[b] / home || !size
            || p_size[b] != size || p_begin[b] < 0 || p_begin[b] > ep - size || p_begin[b] % size || p_helper[b] < 0
            || p_helper[b] >= helpers)
        {
            status = 1;
            break;
        }
        for (int j = 0; j < b; ++j)
        {
            bool const intersects = p_begin[j] < p_begin[b] + size && p_begin[b] < p_begin[j] + p_size[j];
            if (p_expert[j] == p_expert[b] || (intersects && p_helper[j] == p_helper[b]))
                status = 1;
        }
        if (group_contains(local_rank, p_begin[b], size)
            && (ids[p_helper[b]] != p_expert[b] || levels[p_helper[b]] != level || owners[p_helper[b]] != p_owner[b]))
            status = 1;
        int const count = ++owner_count[p_owner[b]];
        if (count > imin(helpers, home))
            status = 1;
        global_max_send = imax(global_max_send, count);
        common_proper_size &= size == common_size;
        if (!group_contains(p_owner[b], p_begin[b], size))
        {
            if (foreign_begin < 0)
                foreign_begin = p_begin[b];
            else
                one_foreign_target &= foreign_begin == p_begin[b];
        }
    }
    if (common_proper_size && foreign_begin >= 0)
        for (int b = 0; b < broadcasts; ++b)
            target_active += p_begin[b] == foreign_begin && p_size[b] == common_size;
    bool const external = !status && (route_features & 1) && common_proper_size && one_foreign_target
        && foreign_begin >= 0 && (global_max_send <= 1 || (global_max_send <= 2 && target_active >= 3));

    for (int slot = 0; slot < stride; ++slot)
    {
        bool const inactive = status || slot >= helpers;
        channel[4 + slot] = inactive ? -1 : ids[slot];
        channel[4 + stride + slot] = inactive ? -1 : levels[slot];
        channel[4 + 2 * stride + slot] = inactive ? -1 : owners[slot];
        channel[4 + 3 * stride + slot] = 0;
        channel[4 + 4 * stride + slot] = -1;
        channel[4 + 5 * stride + slot] = -1;
        channel[4 + 6 * stride + slot] = -1;
    }
    int outgoing = 0;
    for (int b = 0; b < broadcasts && !status; ++b)
    {
        int const owner = p_owner[b], begin = p_begin[b], size = p_size[b];
        bool const local_owner = group_contains(owner, begin, size);
        int mode = 0;
        if (local_owner)
        {
            bool pure = true, best = true;
            for (int j = 0; j < broadcasts; ++j)
            {
                bool const intersects = p_begin[j] < begin + size && begin < p_begin[j] + p_size[j];
                if (intersects
                    && (p_begin[j] != begin || p_size[j] != size || !group_contains(p_owner[j], begin, size)))
                    pure = false;
                if (p_owner[j] == owner && group_contains(owner, p_begin[j], p_size[j])
                    && (p_size[j] > size || (p_size[j] == size && j < b)))
                    best = false;
            }
            if (pure || best)
                mode = 1;
        }
        else if (external)
        {
            mode = 2;
            if (owner == local_rank)
            {
                int target_id = begin / size;
                for (int level = 1; level < p_level[b]; ++level)
                    target_id += 1 << level;
                channel[4 + 4 * stride + outgoing] = p_expert[b];
                channel[4 + 5 * stride + outgoing] = target_id;
                channel[4 + 6 * stride + outgoing] = p_helper[b];
                ++outgoing;
            }
        }
        if (group_contains(local_rank, begin, size))
            channel[4 + 3 * stride + p_helper[b]] = mode;
    }
    channel[1] = global_max_send;
    channel[2] = status;
    channel[3] = outgoing;
    if constexpr (PublishHost)
        store_release_system(channel, 0);
    else
        channel[0] = 0;
}

/* Small ABI6 plans fit in one warp. Load each record once and compare its
 * register values through shuffles. Two mode bits per helper fit in a register
 * for S<=16. Larger/malformed plans use the unchanged scalar publisher.
 * Only lane 0 writes mapped words; its SYS release orders every field. */
template <bool PublishHost = true>
__device__ void publish_plan_channel_warp(int* channel, int const* ids, int const* levels, int const* owners,
    PlacementView const& plan, int helpers, int ep, int local_rank, int expert_count, int plan_abi_version,
    int route_features)
{
    if (!channel || (threadIdx.x >> 5) != 0)
        return;
    // One shared cold fallback keeps its scalar O(B^2) implementation out
    // of the hot sections instead of inlining a copy at every failure gate.
    do
    {
        int const broadcasts = plan.count;
        if (broadcasts < 0 || broadcasts > plan.capacity || broadcasts > 32 || helpers <= 0 || helpers > 16
            || plan_abi_version != 6 || (route_features & ~1))
        {
            break;
        }
        constexpr unsigned mask = 0xffffffffu;
        int const lane = threadIdx.x & 31;
        bool const active = lane < broadcasts;
        int const expert = active ? plan.experts[lane] : -1;
        int const level = active ? plan.levels[lane] : 0;
        int const begin = active ? plan.begins[lane] : 0;
        int const size = active ? plan.sizes[lane] : 0;
        int const owner = active ? plan.owners[lane] : -1;
        int const helper = active ? plan.helpers[lane] : -1;
        // Coalesced helper reads avoid three dependent, short-circuited global
        // loads for every local record. Invalid helpers select a harmless lane
        // and are rejected below before any publication.
        int const slot_id = lane < helpers ? ids[lane] : -1;
        int const slot_level = lane < helpers ? levels[lane] : -1;
        int const slot_owner = lane < helpers ? owners[lane] : -1;
        int const check_id = __shfl_sync(mask, slot_id, helper & 31);
        int const check_level = __shfl_sync(mask, slot_level, helper & 31);
        int const check_owner = __shfl_sync(mask, slot_owner, helper & 31);
        int const home = expert_count / ep;
        bool const level_ok = static_cast<unsigned>(level) < static_cast<unsigned>(hierarchy_depth(ep));
        int const expected_size = level_ok ? ep >> level : 0;
        bool const aligned
            = (ep & (ep - 1)) == 0 ? (begin & (expected_size - 1)) == 0 : begin % imax(expected_size, 1) == 0;
        // Unsigned interval checks stay defined even for malformed int32 fields.
        // The native launch contract guarantees positive home and even EP<=32.
        bool const owner_ok = (static_cast<unsigned>(owner) < static_cast<unsigned>(ep))
            & (static_cast<unsigned>(expert) - static_cast<unsigned>(owner) * static_cast<unsigned>(home)
                < static_cast<unsigned>(home));
        bool const local
            = static_cast<unsigned>(local_rank) - static_cast<unsigned>(begin) < static_cast<unsigned>(expected_size);
        bool const record_valid = (expert >= 0) & (expert < expert_count) & owner_ok & (expected_size > 0)
            & (size == expected_size) & (begin >= 0) & (begin <= ep - expected_size) & aligned & (helper >= 0)
            & (helper < helpers);
        bool const local_valid = (check_id == expert) & (check_level == level) & (check_owner == owner);
        bool const valid = !active | (record_valid & (!local | local_valid));
        // Reject invalid addresses before the pairwise geometry calculations.
        if (__any_sync(mask, !valid))
        {
            break;
        }
        unsigned const same_owner = __match_any_sync(mask, owner);
        int const owned = active ? __popc(same_owner) : 0;
        int const global_max_send = __reduce_max_sync(mask, owned);
        unsigned const same_expert = __match_any_sync(mask, expert);
        int bad = active & ((__popc(same_expert) != 1) | (owned > imin(helpers, home)));
        int pure = 1, best = 1;
        // Validated EP<=32/S<=16 geometry needs only 19 bits. Exchange one
        // packed register per pair rather than four independent shuffles.
        unsigned const pair_record = static_cast<unsigned>(begin) | (static_cast<unsigned>(size - 1) << 5)
            | (static_cast<unsigned>(owner) << 10) | (static_cast<unsigned>(helper) << 15);
        // Eager bitwise predicates avoid divergent short-circuit branches and
        // repeated reconvergence inside this register-only comparison loop.
        // All lanes execute every shuffle, including inactive plan lanes.
#pragma unroll 1
        for (int j = 0; j < broadcasts; ++j)
        {
            unsigned const other = __shfl_sync(mask, pair_record, j);
            int const other_begin = other & 31u;
            int const other_size = ((other >> 5) & 31u) + 1;
            int const other_owner = (other >> 10) & 31u;
            int const other_helper = (other >> 15) & 15u;
            int const intersects = (other_begin < begin + size) & (begin < other_begin + other_size);
            bad |= active & (j < lane) & intersects & (other_helper == helper);
            pure &= !intersects
                | ((other_begin == begin) & (other_size == size) & (other_owner >= begin)
                    & (other_owner < begin + size));
            best &= !((other_owner == owner) & (owner >= other_begin) & (owner < other_begin + other_size)
                & ((other_size > size) | ((other_size == size) & (j < lane))));
        }
        if (__any_sync(mask, bad))
        {
            // Preserve even the scalar error header/partial owner-count semantics.
            break;
        }
        int const common_size = __shfl_sync(mask, size, 0);
        bool const common_proper_size
            = broadcasts > 0 && common_size < ep && !__any_sync(mask, active && size != common_size);
        bool const local_owner = active && group_contains(owner, begin, size);
        bool const foreign = active && !local_owner;
        unsigned const foreign_mask = __ballot_sync(mask, foreign);
        int const first_foreign = foreign_mask ? __ffs(foreign_mask) - 1 : 0;
        int const foreign_begin = __shfl_sync(mask, begin, first_foreign);
        bool const one_foreign_target = !__any_sync(mask, foreign && begin != foreign_begin);
        int const target_active = __popc(__ballot_sync(mask, active && begin == foreign_begin && size == common_size));
        bool const external = (route_features & 1) && common_proper_size && one_foreign_target && foreign_mask
            && (global_max_send <= 1 || (global_max_send <= 2 && target_active >= 3));
        int const mode = local_owner ? (pure | best) : (foreign && external ? 2 : 0);
        unsigned const local_mode
            = active && group_contains(local_rank, begin, size) ? static_cast<unsigned>(mode) << (2 * helper) : 0;
        unsigned const mode_bits = __reduce_or_sync(mask, local_mode);
        unsigned const outgoing_mask = __ballot_sync(mask, mode == 2 && owner == local_rank);
        int const stride = (helpers + 3) & ~3;
        // Reuse the coalesced helper registers from validation. All lanes
        // participate in every shuffle; lane 0 remains the only mapped writer.
        for (int slot = 0; slot < stride; slot += 4)
        {
            const int4 ids4 = make_int4(__shfl_sync(mask, slot_id, slot), __shfl_sync(mask, slot_id, slot + 1),
                __shfl_sync(mask, slot_id, slot + 2), __shfl_sync(mask, slot_id, slot + 3));
            const int4 levels4 = make_int4(__shfl_sync(mask, slot_level, slot), __shfl_sync(mask, slot_level, slot + 1),
                __shfl_sync(mask, slot_level, slot + 2), __shfl_sync(mask, slot_level, slot + 3));
            const int4 owners4 = make_int4(__shfl_sync(mask, slot_owner, slot), __shfl_sync(mask, slot_owner, slot + 1),
                __shfl_sync(mask, slot_owner, slot + 2), __shfl_sync(mask, slot_owner, slot + 3));
            if (lane == 0)
            {
                // Native channel/stride alignment permits 16-byte stores.
                const int4 empty = make_int4(-1, -1, -1, -1);
                *reinterpret_cast<int4*>(channel + 4 + slot) = ids4;
                *reinterpret_cast<int4*>(channel + 4 + stride + slot) = levels4;
                *reinterpret_cast<int4*>(channel + 4 + 2 * stride + slot) = owners4;
                unsigned const packed = mode_bits >> (2 * slot);
                *reinterpret_cast<int4*>(channel + 4 + 3 * stride + slot)
                    = make_int4(packed & 3u, (packed >> 2) & 3u, (packed >> 4) & 3u, (packed >> 6) & 3u);
                *reinterpret_cast<int4*>(channel + 4 + 4 * stride + slot) = empty;
                *reinterpret_cast<int4*>(channel + 4 + 5 * stride + slot) = empty;
                *reinterpret_cast<int4*>(channel + 4 + 6 * stride + slot) = empty;
            }
        }
        // Canonical target IDs enumerate proper hierarchy levels, widest first.
        int const target = mode == 2 ? begin / size + (1 << level) - 2 : -1;
        int outgoing = 0;
        for (unsigned remaining = outgoing_mask; remaining; remaining &= remaining - 1)
        {
            int const source_lane = __ffs(remaining) - 1;
            int const out_expert = __shfl_sync(mask, expert, source_lane);
            int const out_target = __shfl_sync(mask, target, source_lane);
            int const out_helper = __shfl_sync(mask, helper, source_lane);
            if (lane == 0)
            {
                channel[4 + 4 * stride + outgoing] = out_expert;
                channel[4 + 5 * stride + outgoing] = out_target;
                channel[4 + 6 * stride + outgoing] = out_helper;
            }
            ++outgoing;
        }
        if (lane == 0)
        {
            channel[1] = global_max_send;
            channel[2] = 0;
            channel[3] = outgoing;
            if constexpr (PublishHost)
                store_release_system(channel, 0);
            else
                channel[0] = 0;
        }
        return;
    } while (false);
    publish_plan_channel<PublishHost>(
        channel, ids, levels, owners, plan, helpers, ep, local_rank, expert_count, plan_abi_version, route_features);
}

} // namespace sami_copy
