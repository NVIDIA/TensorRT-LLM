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
/* Construct the existing TMA descriptor ABI inside the payload kernel.
 * Each CTA has private global scratch: 4*helper_count int32 slot words,
 * max_segments segments, max_ranges ranges, and one result. CTA thread zero
 * calls the builder and all threads synchronize before consuming its output.
 * No CPU plan read, descriptor DMA, allocation, atomics or device-wide barrier.
 * Scheduling already orders the producer before this kernel on its stream.
 * A rank writes at most helper_count local rows plus, as an owner, K =
 * min(4 * S, H, 32) outgoing records of up to two receivers each; max_segments
 * must cover (helper_count + 2 * K) * planes when outgoing routes are enabled.
 * With pair multicast an owner's own pair-level slot is one of its outgoing
 * records, not a local row.
 */
#ifndef MEGAMOE_TMA_GPU_PLAN_BUILDER_CUH
#define MEGAMOE_TMA_GPU_PLAN_BUILDER_CUH
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTmaGpuPlan.h"
#include <stddef.h>
#if defined(__CUDACC__)
#define MEGAMOE_GP_HD __host__ __device__
#else
#define MEGAMOE_GP_HD
#endif
namespace megamoe_gpu_plan_detail
{
/* Published owner-direct unicast mode: the owner writes this receiver's row. */
static constexpr int kSlotModeUnicast = MEGAMOE_TMA_GPU_PLAN_MODE_UNICAST;
/* A helper slot whose whole team already holds its expert needs no copy. This
 * marker is local to the builder, above every published mode. */
static constexpr int kSlotModeResident = 4;

struct Builder : MegamoeTmaGpuPlanConfig
{
    int32_t const* plan;
    int32_t const* resident; /* [world][helper_count] expert held per helper slot, or null. */
    int32_t* slot_scratch;
    MegamoeTmaCopySegment* tma_segments;
    MegamoeTmaCopyRange* tma_ranges;
    size_t max_commands, tma_max_ranges;
    size_t tma_segment_count, tma_range_count, last_commands;
    int last_direct_slots, last_scatter_slots;
    int last_tma_weighted_domains, last_tma_uniform_domains;
    int last_tma_mixed_source_domains, last_tma_foreign_source_domains;
    uint64_t last_tma_weighted_total_slices, last_tma_weighted_source_quota_slices;
    uint64_t last_tma_weighted_local_slices, last_tma_local_slices;
};

static MEGAMOE_GP_HD inline uint64_t tma_weighted_source_quota(uint64_t slices, unsigned members, unsigned percent)
{
    return slices * percent / (100U * members);
}

static MEGAMOE_GP_HD inline int target_geometry(Builder const* c, int target, int* level_out, int* begin_out)
{
    if (target < 0 || target >= c->target_count)
        return -1;
    for (int level = 1; level < c->level_count; ++level)
    {
        int count = c->world / c->group_sizes[level];
        if (target < count)
        {
            *level_out = level;
            *begin_out = target * c->group_sizes[level];
            return 0;
        }
        target -= count;
    }
    return -1;
}

/* Ranks [begin, begin + size) of a pair-level record, which this rank owns: one
 * rank or an even-aligned pair without this rank; with pair multicast a whole
 * even-aligned pair, this rank possibly inside. */
static MEGAMOE_GP_HD inline int unicast_geometry(Builder const* c, int target, int* begin_out, int* size_out)
{
    if (target < MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE)
        return -1;
    int const code = target - MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE;
    int const begin = code >> 1, size = (code & 1) + 1;
    bool const inside = c->rank >= begin && c->rank < begin + size;
    bool const pairs = (c->route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_PAIR_MULTICAST) != 0;
    if (begin > c->world - size || (size == 2 && (begin & 1)) || (pairs ? size != 2 : inside))
        return -1;
    *begin_out = begin;
    *size_out = size;
    return 0;
}

/* The ranks an outgoing record writes: a proper multicast team or the ranks
 * of a pair-level record. */
static MEGAMOE_GP_HD inline int outgoing_receivers(Builder const* c, int target, int* begin_out, int* size_out)
{
    if (target >= MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE)
        return unicast_geometry(c, target, begin_out, size_out);
    int level = 0;
    if (target_geometry(c, target, &level, begin_out))
        return -1;
    *size_out = c->group_sizes[level];
    return 0;
}

static MEGAMOE_GP_HD inline int append_tma_segment(
    Builder* c, size_t* segments, uint64_t* cursor, uint64_t src, uint64_t dst, size_t bytes)
{
    if (*segments >= c->max_commands || !src || !dst || !bytes || (src & 3) || (dst & 3) || (bytes & 3)
        || bytes > UINT64_MAX - *cursor || bytes > UINT64_MAX - src || bytes > UINT64_MAX - dst)
        return -1;
    MegamoeTmaCopySegment* s = &c->tma_segments[(*segments)++];
    s->src = src;
    s->dst = dst;
    s->bytes = bytes;
    s->virtual_begin = *cursor;
    *cursor += bytes;
    return 0;
}

/* The validated arena binds one UC object per (owner, plane) and one MC
 * object per (team, plane). The caller resets identity at plane/domain
 * boundaries and gives normal and external teams separate namespaces.
 * Numerical pointer adjacency alone is not enough to merge MC destinations. */
static MEGAMOE_GP_HD inline int append_tma_coalesced_segment(Builder* c, size_t* segments, uint64_t* cursor,
    uint64_t src, uint64_t dst, size_t bytes, uint64_t identity, uint64_t* previous_identity)
{
    if (!src || !dst || !bytes || (src & 3) || (dst & 3) || (bytes & 3) || bytes > UINT64_MAX - *cursor
        || bytes > UINT64_MAX - src || bytes > UINT64_MAX - dst)
        return -1;
    if (*segments && *previous_identity == identity)
    {
        MegamoeTmaCopySegment* previous = &c->tma_segments[*segments - 1];
        if (previous->bytes <= UINT64_MAX - previous->src && previous->bytes <= UINT64_MAX - previous->dst
            && previous->bytes <= UINT64_MAX - previous->virtual_begin && previous->src + previous->bytes == src
            && previous->dst + previous->bytes == dst && previous->virtual_begin + previous->bytes == *cursor
            && bytes <= UINT64_MAX - previous->bytes)
        {
            previous->bytes += bytes;
            *cursor += bytes;
            return 0;
        }
    }
    if (append_tma_segment(c, segments, cursor, src, dst, bytes))
        return -1;
    *previous_identity = identity;
    return 0;
}

static MEGAMOE_GP_HD inline int append_tma_range(Builder* c, size_t* ranges, uint64_t* prefix, size_t segment_begin,
    size_t segment_count, uint64_t bytes, uint64_t first, uint64_t step, uint64_t slices, uint64_t divisor,
    uint64_t reciprocal, uint64_t flags = 0)
{
    /* These constant bounds also avoid runtime division for overflow checks.
     * Valid 8KiB streams are smaller than this conservative slice bound. */
    if (!step || !divisor || divisor > 100U * MEGAMOE_TMA_GPU_PLAN_MAX_WORLD
        || slices > UINT64_MAX / (100U * MEGAMOE_TMA_GPU_PLAN_MAX_WORLD))
        return -1;
    uint64_t scaled_limit = slices * divisor;
    if (first >= scaled_limit)
        return 0;
    uint64_t count = 1 + (scaled_limit - 1 - first) / step;
    if (*ranges >= c->tma_max_ranges || count > UINT64_MAX - *prefix)
        return -1;
    MegamoeTmaCopyRange* r = &c->tma_ranges[(*ranges)++];
    r->segment_begin = segment_begin;
    r->segment_count = segment_count;
    r->bytes = bytes;
    r->first_slice = first;
    r->slice_stride = step;
    r->slice_count = count;
    *prefix += count;
    r->prefix_end = *prefix;
    r->slice_divisor = divisor;
    r->slice_reciprocal = reciprocal;
    r->flags = flags;
    return 0;
}

static MEGAMOE_GP_HD inline bool team_holds_expert(Builder const* c, int slot, int expert, int level)
{
    int group = c->group_sizes[level];
    int begin = c->rank / group * group;
    for (int rank = begin; rank < begin + group; ++rank)
        if (c->resident[rank * c->helper_count + slot] != expert)
            return false;
    return true;
}

static MEGAMOE_GP_HD inline size_t build_tma_descriptors(Builder* c, int outgoing_count, int* plan_error)
{
    *plan_error = 3;
    int slots = c->helper_count, stride = c->owner_stride;
    int *experts = c->slot_scratch, *levels = experts + slots;
    int *owners = levels + slots, *modes = owners + slots;
    size_t segments = 0, ranges = 0;
    uint64_t prefix = 0;
    int direct = 0, scatter = 0;
    c->last_tma_weighted_domains = c->last_tma_uniform_domains = 0;
    c->last_tma_mixed_source_domains = c->last_tma_foreign_source_domains = 0;
    c->last_tma_weighted_total_slices = c->last_tma_weighted_source_quota_slices = 0;
    c->last_tma_weighted_local_slices = c->last_tma_local_slices = 0;
    for (int slot = 0; slot < slots; ++slot)
    {
        // The owner writes external and owner-direct unicast slots.
        if (experts[slot] < 0 || modes[slot] == 2 || modes[slot] == kSlotModeUnicast)
            continue;
        if (c->resident && team_holds_expert(c, slot, experts[slot], levels[slot]))
        {
            modes[slot] = kSlotModeResident;
            continue;
        }
        if (c->tma_route == 1)
        {
            int group = c->group_sizes[levels[slot]];
            int begin = c->rank / group * group;
            if (owners[slot] < begin || owners[slot] >= begin + group)
            {
                *plan_error = 2;
                return 0;
            }
            modes[slot] = 1;
        }
        else if (c->tma_route == 2)
        {
            modes[slot] = 0;
        }
        direct += modes[slot] == 1;
        scatter += modes[slot] == 0;
    }
    /* Domain zero is source-owned work (including outgoing direct). A scatter
     * domain must have one common member set, so it is separated per level;
     * all ranks in that aligned team see the same plane/slot byte order. */
    for (int domain = 0; domain <= c->level_count; ++domain)
    {
        size_t begin_segment = segments;
        uint64_t cursor = 0;
        /* Keep one uninterrupted virtual stream per routing domain. Plane-major
         * order groups the large aligned weights before the scalar planes; the
         * physical arena and source ownership are unchanged. Within each plane,
         * normal source-owned slots precede existing external outgoing records. */
        for (int plane = 0; plane < c->planes; ++plane)
        {
            uint64_t previous_identity = UINT64_MAX;
            for (int slot = 0; slot < slots; ++slot)
            {
                if (experts[slot] < 0 || modes[slot] == 2)
                    continue;
                if (domain == 0 ? (modes[slot] != 1 || owners[slot] != c->rank)
                                : (modes[slot] != 0 || levels[slot] != domain - 1))
                    continue;
                int level = levels[slot];
                size_t si = (size_t) experts[slot] * c->planes + plane;
                size_t di = ((size_t) level * slots + slot) * c->planes + plane;
                uint64_t identity = ((uint64_t) owners[slot] << 32) | (uint32_t) level;
                if (append_tma_coalesced_segment(c, &segments, &cursor, c->src_table[si], c->dst_table[di],
                        c->plane_bytes[plane], identity, &previous_identity))
                    return 0;
            }
            if (domain == 0)
            {
                for (int i = 0; i < outgoing_count; ++i)
                {
                    int expert = c->plan[4 + 4 * stride + i];
                    int target = c->plan[4 + 5 * stride + i];
                    int helper = c->plan[4 + 6 * stride + i];
                    if (target >= MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE)
                        continue; /* The unicast domain below writes these receivers. */
                    size_t si = (size_t) expert * c->planes + plane;
                    size_t di = ((size_t) target * slots + helper) * c->planes + plane;
                    uint64_t identity = ((uint64_t) c->rank << 32) | (uint32_t) (c->level_count + target);
                    if (append_tma_coalesced_segment(c, &segments, &cursor, c->src_table[si], c->foreign_dst_table[di],
                            c->plane_bytes[plane], identity, &previous_identity))
                        return 0;
                }
            }
        }
        if (!cursor)
            continue;
        uint64_t step = domain ? (uint64_t) c->group_sizes[domain - 1] : 1;
        uint64_t first = domain ? (uint64_t) c->rank % step : 0;
        uint64_t slices = cursor / MEGAMOE_TMA_COPY_SLICE_BYTES + (cursor % MEGAMOE_TMA_COPY_SLICE_BYTES != 0);
        size_t before_ranges = ranges;
        int common_source = -1;
        int weighted = 0;
        if (domain && c->tma_source_load_percent != 100)
        {
            for (int slot = 0; slot < slots; ++slot)
            {
                if (experts[slot] < 0 || modes[slot] != 0 || levels[slot] != domain - 1)
                    continue;
                if (common_source == -1)
                    common_source = owners[slot];
                else if (common_source != owners[slot])
                {
                    common_source = -2;
                    break;
                }
            }
            int team_begin = c->rank / (int) step * (int) step;
            if (common_source == -2)
                ++c->last_tma_mixed_source_domains;
            else if (common_source < team_begin || common_source >= team_begin + (int) step)
                ++c->last_tma_foreign_source_domains;
            else
                weighted = 1;
        }
        if (weighted)
        {
            unsigned members = (unsigned) step;
            unsigned source_local = (unsigned) common_source % members;
            unsigned percent = (unsigned) c->tma_source_load_percent;
            uint64_t denominator = 100U * members;
            uint64_t before_prefix = prefix;
            uint64_t source_quota = tma_weighted_source_quota(slices, members, percent);
            if (slices > UINT64_MAX - c->last_tma_weighted_total_slices
                || source_quota > UINT64_MAX - c->last_tma_weighted_source_quota_slices)
                return 0;
            ++c->last_tma_weighted_domains;
            c->last_tma_weighted_total_slices += slices;
            c->last_tma_weighted_source_quota_slices += source_quota;
            uint64_t divisor, reciprocal;
            if (c->rank == common_source)
            {
                first = denominator - 1;
                step = denominator;
                divisor = percent;
                reciprocal = divisor == 1 ? 0 : UINT64_MAX / divisor;
            }
            else
            {
                unsigned peer_order = ((unsigned) c->rank % members + members - source_local) % members - 1;
                first = peer_order * denominator;
                step = (members - 1) * denominator;
                divisor = denominator - percent;
                reciprocal = divisor == 1 ? 0 : UINT64_MAX / divisor;
            }
            /* floor((first+j*step)/divisor) is the inverse of the source
             * quota events and peer round-robin. The whole existing virtual
             * stream stays intact, with increasing slices and a continuous
             * rank-local prefix across domains for SM/worker round-robin. */
            if (append_tma_range(c, &ranges, &prefix, begin_segment, segments - begin_segment, cursor, first, step,
                    slices, divisor, reciprocal))
                return 0;
            c->last_tma_weighted_local_slices += prefix - before_prefix;
        }
        else
        {
            if (domain)
                ++c->last_tma_uniform_domains;
            if (append_tma_range(
                    c, &ranges, &prefix, begin_segment, segments - begin_segment, cursor, first, step, slices, 1, 0))
                return 0;
        }
        if (ranges == before_ranges)
            segments = begin_segment; /* This rank owns none of this stream's slices. */
    }
    /* Pair-level records: this rank writes its own ones itself, as source-owned
     * streams after the multicast domains, so their slice order and worker
     * assignment stay unchanged.  With pair multicast, a plane that has a pair
     * team goes into it once (multicast stream), unless both ranks already hold
     * the expert in that row.  Every other plane goes into each rank's row that
     * does not hold it (unicast stream).  Segments coalesce only within one
     * mapping. */
    if (c->uc_dst_table)
    {
        for (int multicast = c->pair_dst_table != nullptr; multicast >= 0; --multicast)
        {
            size_t begin_segment = segments;
            uint64_t cursor = 0;
            for (int plane = 0; plane < c->planes; ++plane)
            {
                uint64_t previous_identity = UINT64_MAX;
                for (int i = 0; i < outgoing_count; ++i)
                {
                    int expert = c->plan[4 + 4 * stride + i];
                    int target = c->plan[4 + 5 * stride + i];
                    int helper = c->plan[4 + 6 * stride + i];
                    int begin = 0, size = 0;
                    if (target < MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE)
                        continue;
                    if (unicast_geometry(c, target, &begin, &size))
                        return 0;
                    size_t si = (size_t) expert * c->planes + plane;
                    uint64_t const team = c->pair_dst_table
                        ? c->pair_dst_table[((size_t) (begin / 2) * slots + helper) * c->planes + plane]
                        : 0;
                    if ((team != 0) != (multicast != 0))
                        continue;
                    if (team)
                    {
                        bool held = c->resident != nullptr;
                        for (int rank = begin; held && rank < begin + size; ++rank)
                            held = c->resident[(size_t) rank * slots + helper] == expert;
                        uint64_t identity = (UINT64_C(1) << 62) | ((uint64_t) (begin / 2) << 32);
                        if (!held
                            && append_tma_coalesced_segment(c, &segments, &cursor, c->src_table[si], team,
                                c->plane_bytes[plane], identity, &previous_identity))
                            return 0;
                        continue;
                    }
                    for (int receiver = begin; receiver < begin + size; ++receiver)
                    {
                        if (c->resident && c->resident[(size_t) receiver * slots + helper] == expert)
                            continue;
                        size_t di = ((size_t) receiver * slots + helper) * c->planes + plane;
                        uint64_t identity = (UINT64_C(1) << 63) | ((uint64_t) receiver << 32);
                        if (append_tma_coalesced_segment(c, &segments, &cursor, c->src_table[si], c->uc_dst_table[di],
                                c->plane_bytes[plane], identity, &previous_identity))
                            return 0;
                    }
                }
            }
            if (cursor)
            {
                uint64_t slices = cursor / MEGAMOE_TMA_COPY_SLICE_BYTES + (cursor % MEGAMOE_TMA_COPY_SLICE_BYTES != 0);
                if (append_tma_range(c, &ranges, &prefix, begin_segment, segments - begin_segment, cursor, 0, 1, slices,
                        1, 0, multicast ? 0 : MEGAMOE_TMA_COPY_RANGE_UNICAST))
                    return 0;
            }
        }
    }
    c->last_tma_local_slices = prefix;
    c->tma_segment_count = segments;
    c->tma_range_count = ranges;
    c->last_commands = segments;
    c->last_direct_slots = direct;
    c->last_scatter_slots = scatter;
    *plan_error = 0;
    return segments;
}

static MEGAMOE_GP_HD inline size_t build_descriptors(Builder* c, int* plan_error)
{
    const volatile int32_t* plan = c->plan;
    int const slots = c->helper_count;
    int const stride = c->owner_stride;
    // Records one owner may publish: HALO-Q's candidate count min(4 * S, H, 32).
    int const quota = megamoe_tma_gpu_plan_owner_records(slots, c->home_count);
    int* experts = c->slot_scratch;
    int* levels = experts + slots;
    int* owners = levels + slots;
    int* modes = owners + slots;
    int owner_counts[MEGAMOE_TMA_GPU_PLAN_MAX_WORLD] = {0};
    unsigned char seen_experts[384] = {0};
    int incoming_slots = 0;
    int active = 0, common_level = -1;
    int global_max_send = plan[1];
    int outgoing_count = plan[3];
    bool const external_route = (c->route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_EXTERNAL) && c->foreign_dst_table;
    bool const unicast_route = (c->route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_OWNER_UNICAST) && c->uc_dst_table;
    bool const pair_route = (c->route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_PAIR_MULTICAST) && c->pair_dst_table;
    int own_pair_slots = 0;
    *plan_error = 1;
    if (c->plan_abi_version != MEGAMOE_TMA_GPU_PLAN_ABI_VERSION || c->plan_words != 4 + 7 * stride || plan[2] != 0
        || global_max_send < 0 || global_max_send > quota || outgoing_count < 0 || outgoing_count > quota
        || (!c->route_features && outgoing_count))
        return 0;

    /* Published modes are global decisions. Validate the complete local row
     * before constructing any descriptor; inactive capacity never becomes work. */
    for (int slot = 0; slot < slots; ++slot)
    {
        int expert = experts[slot] = plan[4 + slot];
        int level = levels[slot] = plan[4 + stride + slot];
        int owner = owners[slot] = plan[4 + 2 * stride + slot];
        int mode = modes[slot] = plan[4 + 3 * stride + slot];
        if (expert == -1)
        {
            if (level != -1 || owner != -1 || mode != 0)
                return 0;
            continue;
        }
        /* The pair level follows the multicast levels. Exactly its slots are
         * owner-direct: their owner writes them, this rank itself only for its
         * own pair under pair multicast. */
        bool const unicast = level == c->level_count;
        if (expert < 0 || expert >= c->global_experts || level < 0 || level > c->level_count || owner < 0
            || owner >= c->world || owner != expert / c->home_count || mode < 0 || mode > kSlotModeUnicast
            || unicast != (mode == kSlotModeUnicast) || seen_experts[expert])
            return 0;
        seen_experts[expert] = 1;
        if (++owner_counts[owner] > global_max_send)
            return 0;
        if (unicast)
        {
            if (!unicast_route || (owner == c->rank && !pair_route))
                return 0;
            own_pair_slots += owner == c->rank;
            continue;
        }
        int group = c->group_sizes[level];
        int begin = c->rank / group * group;
        int local = owner >= begin && owner < begin + group;
        if ((mode == 1 && !local) || (mode == 2 && (!external_route || local || level == 0)))
            return 0;
        if (!active)
            common_level = level;
        else if (common_level != level)
            common_level = -2;
        ++active;
        incoming_slots += mode == 2;
    }
    /* Source-direct is valid whenever its owner belongs to the target team.
     * The scheduler may explicitly choose it for multiple weights or levels;
     * auto-policy heuristics are not descriptor safety constraints. The
     * external-route rules below cover the multicast-level slots only. */
    if (incoming_slots)
    {
        if (common_level <= 0 || global_max_send == 0 || global_max_send > 2 || (global_max_send == 2 && active < 3))
            return 0;
        int group = c->group_sizes[common_level];
        int begin = c->rank / group * group;
        for (int slot = 0; slot < slots; ++slot)
        {
            if (experts[slot] < 0 || modes[slot] == kSlotModeUnicast)
                continue;
            int external = owners[slot] < begin || owners[slot] >= begin + group;
            if (external != (modes[slot] == 2))
                return 0;
        }
    }
    int multicast_outgoing = 0;
    for (int i = 0; i < outgoing_count; ++i)
        multicast_outgoing += plan[4 + 5 * stride + i] < MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE;
    if (multicast_outgoing
        && (!external_route || global_max_send == 0 || global_max_send > 2 || (active && common_level <= 0)))
        return 0;
    if (outgoing_count > multicast_outgoing && !unicast_route)
        return 0;
    // An own pair-level slot is the same record as its outgoing record.
    if (owner_counts[c->rank] - own_pair_slots + outgoing_count > global_max_send)
        return 0;
    for (int i = 0; i < outgoing_count; ++i)
    {
        int expert = plan[4 + 4 * stride + i];
        int target = plan[4 + 5 * stride + i];
        int helper = plan[4 + 6 * stride + i];
        int begin, size;
        if (expert < c->rank * c->home_count || expert >= (c->rank + 1) * c->home_count || helper < 0 || helper >= slots
            || outgoing_receivers(c, target, &begin, &size))
            return 0;
        // A multicast target must not contain this rank. A pair-level record
        // holds it only under pair multicast, as the own slot of that record.
        if (c->rank >= begin && c->rank < begin + size ? target < MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE
                    || experts[helper] != expert || modes[helper] != kSlotModeUnicast
                                                       : seen_experts[expert])
            return 0;
        seen_experts[expert] = 1;
        // Records whose written ranks overlap must use different helper rows.
        for (int j = 0; j < i; ++j)
        {
            int other_begin, other_size;
            if (plan[4 + 6 * stride + j] != helper)
                continue;
            if (outgoing_receivers(c, plan[4 + 5 * stride + j], &other_begin, &other_size)
                || (other_begin < begin + size && begin < other_begin + other_size))
                return 0;
        }
    }
    for (int i = outgoing_count; i < stride; ++i)
        if (plan[4 + 4 * stride + i] != -1 || plan[4 + 5 * stride + i] != -1 || plan[4 + 6 * stride + i] != -1)
            return 0;
    for (int slot = slots; slot < stride; ++slot)
        if (plan[4 + slot] != -1 || plan[4 + stride + slot] != -1 || plan[4 + 2 * stride + slot] != -1
            || plan[4 + 3 * stride + slot] != 0)
            return 0;

    return build_tma_descriptors(c, outgoing_count, plan_error);
}

} // namespace megamoe_gpu_plan_detail

/* Host/device parity permits differential tests against the CPU submitter.
 * Counts remain zero on failure: callers must not publish successful READY.
 * Caller supplies valid memory for the full immutable tables and scratch.
 * A non-null resident table removes the slots whose team already holds the
 * planned expert from every routing domain.
 */
static MEGAMOE_GP_HD inline int megamoe_tma_build_gpu_plan(MegamoeTmaGpuPlanConfig const* config,
    int32_t const* device_plan, int32_t* slot_scratch, MegamoeTmaCopySegment* segments, uint64_t max_segments,
    MegamoeTmaCopyRange* ranges, uint64_t max_ranges, MegamoeTmaGpuPlanResult* result,
    int32_t const* resident = nullptr)
{
    if (!result)
        return 1;
    *result = MegamoeTmaGpuPlanResult{};
    result->error = 1;
    if (!config || !device_plan || !slot_scratch || !segments || !ranges || !max_segments || !max_ranges
        || max_segments > SIZE_MAX || max_ranges > SIZE_MAX)
        return 1;
    MegamoeTmaGpuPlanConfig const& g = *config;
    if (g.world < 2 || g.world > MEGAMOE_TMA_GPU_PLAN_MAX_WORLD || (g.world & 1) || g.rank < 0 || g.rank >= g.world
        || g.helper_count <= 0 || g.planes <= 0 || g.global_experts <= 0 || g.global_experts > 384
        || g.global_experts % g.world || g.home_count != g.global_experts / g.world || g.level_count <= 0
        || g.level_count > MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS
        || g.owner_stride != megamoe_tma_gpu_plan_stride(g.helper_count)
        || int64_t(g.plan_words) != 4 + 7 * int64_t(g.owner_stride)
        || g.plan_abi_version != MEGAMOE_TMA_GPU_PLAN_ABI_VERSION
        || (g.route_features & ~MEGAMOE_TMA_GPU_PLAN_ROUTE_MASK) || g.tma_route < 0 || g.tma_route > 2
        || g.tma_source_load_percent < 1 || g.tma_source_load_percent > 199 || !g.src_table || !g.dst_table
        || !g.plane_bytes
        || ((g.route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_EXTERNAL) != 0) != (g.foreign_dst_table != nullptr)
        || ((g.route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_OWNER_UNICAST) != 0) != (g.uc_dst_table != nullptr)
        || ((g.route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_PAIR_MULTICAST) != 0) != (g.pair_dst_table != nullptr)
        || ((g.route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_PAIR_MULTICAST) && !g.uc_dst_table) || device_plan[0] != 0)
        return 1;
    int targets = 0;
    // Below level 0 groups hold 4+ ranks, or 2 for a power-of-two world up to 16 (pair ladder).
    int const min_group = (g.world & (g.world - 1)) == 0 && g.world <= 16 ? 2 : 4;
    for (int level = 0; level < g.level_count; ++level)
    {
        int group = g.group_sizes[level];
        if (group < (level ? min_group : 2) || group > g.world || g.world % group || (!level && group != g.world)
            || (level && (group >= g.group_sizes[level - 1] || g.group_sizes[level - 1] % group)))
            return 1;
        if (level)
            targets += g.world / group;
    }
    if (g.target_count != targets || ((g.route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_EXTERNAL) && !targets))
        return 1;
    megamoe_gpu_plan_detail::Builder c{};
    static_cast<MegamoeTmaGpuPlanConfig&>(c) = g;
    c.plan = device_plan;
    c.resident = resident;
    c.slot_scratch = slot_scratch;
    c.tma_segments = segments;
    c.tma_ranges = ranges;
    c.max_commands = (size_t) max_segments;
    c.tma_max_ranges = (size_t) max_ranges;
    int error = 1;
    megamoe_gpu_plan_detail::build_descriptors(&c, &error);
    result->error = error;
    if (error)
        return error;
    result->direct_slots = c.last_direct_slots;
    result->scatter_slots = c.last_scatter_slots;
    result->weighted_domains = c.last_tma_weighted_domains;
    result->uniform_domains = c.last_tma_uniform_domains;
    result->mixed_source_domains = c.last_tma_mixed_source_domains;
    result->foreign_source_domains = c.last_tma_foreign_source_domains;
    result->segment_count = c.tma_segment_count;
    result->range_count = c.tma_range_count;
    result->total_slices = c.last_tma_local_slices;
    result->weighted_total_slices = c.last_tma_weighted_total_slices;
    result->weighted_source_quota_slices = c.last_tma_weighted_source_quota_slices;
    result->weighted_local_slices = c.last_tma_weighted_local_slices;
    return 0;
}

namespace megamoe_gpu_plan_detail
{
/* ROUTE_BALANCED. The issuers of a copy are its owner, when it is a member of
 * the destination group or the group is a pair it can reach through its team
 * object or with unicast stores, and the members other than the owner. An owner
 * slice reads local memory and costs 100, or 200 when the owner writes it to
 * both ranks of a pair with unicast stores; a member pulls its slices from the
 * owner and they cost tma_source_load_percent. Integer water-filling levels the
 * issuers' loads, copy by copy in placement order, then kBalancedPasses times more with each
 * copy taken out and put back (2 passes: within 1.3% of 3 at EP8 on real
 * layers, for three quarters of the device time). Every rank computes the same
 * split. */
static constexpr int kBalancedMaxCopies = 96;
static constexpr int kBalancedMaxIssuers = 17; // a group of 16 plus an owner outside it
static constexpr int kBalancedPasses = 2;
static constexpr uint32_t kBalancedOwnerCost = 100;
static constexpr uint32_t kBalancedUnicastOwnerCost = 2 * kBalancedOwnerCost;

struct BalancedCopy
{
    int expert, level, begin, size, owner, helper;
    int owner_issues;    // 1: issuer 0 is the owner
    int inside;          // the owner is a member of the destination group
    uint32_t slices, owner_cap;
    uint32_t owner_cost; // per owner slice: 100, or 200 for unicast stores into both ranks of a pair
    uint32_t take[kBalancedMaxIssuers];
};

static MEGAMOE_GP_HD inline int balanced_issuers(BalancedCopy const& b)
{
    return b.owner_issues + b.size - b.inside;
}

static MEGAMOE_GP_HD inline int balanced_rank(BalancedCopy const& b, int k)
{
    if (b.owner_issues && k == 0)
        return b.owner;
    int rank = b.begin + k - b.owner_issues;
    return b.inside && rank >= b.owner ? rank + 1 : rank;
}

/* Every rank's load, their sum and a finish one slice past any load stay below
 * 2^32: a copy adds at most slices * 200 over all of its issuers. */
static MEGAMOE_GP_HD inline bool balanced_loads_fit(int count, uint64_t slices)
{
    return slices <= UINT32_MAX && (uint64_t) (count > 0 ? count : 1) * slices * 200U <= UINT32_MAX;
}

static MEGAMOE_GP_HD inline uint64_t balanced_div(uint64_t a, uint64_t b)
{
    // 32-bit division when it fits: the 64-bit one is a long software sequence on the device.
    return (a | b) <= UINT32_MAX ? (uint64_t) ((uint32_t) a / (uint32_t) b) : a / b;
}

} // namespace megamoe_gpu_plan_detail

#if defined(__CUDACC__)
namespace megamoe_gpu_plan_detail
{
// Issuer index of `rank` in copy b, or -1: the inverse of balanced_rank.
static __device__ __forceinline__ int balanced_issuer_of(BalancedCopy const& b, int rank)
{
    if (b.owner_issues && rank == b.owner)
        return 0;
    if (rank < b.begin || rank >= b.begin + b.size)
        return -1;
    return b.owner_issues + rank - b.begin - (b.inside && rank > b.owner ? 1 : 0);
}

/* Puts each copy b onto the loads at the lowest finish level, in closed form and in
 * exact integers. Weighted by w = owner_cost * pull_cost / cost (pull_cost for an
 * owner issuer, owner_cost for a member), the level of an active set is
 * floor((slices * owner_cost * pull_cost + sum w * load) / sum w); issuers at or
 * above it leave the set. An owner whose share passes its cap is fixed at the
 * cap, and the members level the
 * rest. Each active issuer takes floor((level - load) / cost), which never
 * exceeds the slices; each remaining slice goes to the issuer that would finish
 * it first (the earlier issuer on a tie).
 * One warp places a copy: lane k is issuer k; load[] (shared, one word per
 * rank) starts at zero. balanced_loads_fit keeps every load and sum
 * below 2^32. */
static __device__ __forceinline__ void balanced_split_warp(
    BalancedCopy* copies, int n, uint32_t* load, uint32_t pull_cost, int passes, int lane)
{
    unsigned const all_lanes = 0xffffffffu;
    for (int pass = -1; pass < passes; ++pass)
        for (int j = 0; j < n; ++j)
        {
            BalancedCopy& b = copies[j];
            int const issuers = balanced_issuers(b);
            bool const is = lane < issuers;
            bool const own = b.owner_issues && lane == 0;
            uint32_t const owner_cost = b.owner_cost;
            uint32_t const cost = own ? owner_cost : pull_cost;
            uint32_t const cap = own ? b.owner_cap : UINT32_MAX;
            int const rank = is ? balanced_rank(b, lane) : 0;
            uint32_t l = is ? load[rank] : 0;
            if (pass >= 0 && is)
                l -= b.take[lane] * cost;
            uint32_t const everyone = __ballot_sync(all_lanes, is);
            uint32_t active = everyone, remaining = b.slices;
            bool fixed = false;
            uint64_t level = 0;
            for (int round = 0; round < 2; ++round)
            {
                for (int it = 0; it < issuers; ++it)
                {
                    bool const act = active >> lane & 1;
                    uint32_t const members = __reduce_add_sync(all_lanes, act && !own ? l : 0);
                    uint32_t const l0 = __shfl_sync(all_lanes, l, 0);
                    bool const own_active = b.owner_issues && (active & 1);
                    uint32_t const count = __popc(active) - (own_active ? 1 : 0);
                    uint64_t const num = (uint64_t) remaining * owner_cost * pull_cost + (uint64_t) owner_cost * members
                        + (own_active ? (uint64_t) pull_cost * l0 : 0);
                    uint64_t const den = (uint64_t) owner_cost * count + (own_active ? pull_cost : 0);
                    level = balanced_div(num, den);
                    uint32_t const keep = __ballot_sync(all_lanes, act && l < level);
                    if (keep == active || !keep)
                        break;
                    active = keep;
                }
                uint32_t const l0 = __shfl_sync(all_lanes, l, 0);
                if (round || fixed || !b.owner_issues || b.owner_cap == UINT32_MAX || !(active & 1) || level <= l0
                    || balanced_div(level - l0, owner_cost) <= b.owner_cap)
                    break;
                fixed = true;
                remaining -= b.owner_cap;
                active = remaining ? everyone & ~1u : 0;
                if (!active)
                    break;
            }
            uint32_t t = 0;
            if (fixed && own)
                t = b.owner_cap;
            else if ((active >> lane & 1) && level > l)
            {
                uint64_t const x = balanced_div(level - l, cost);
                t = (uint32_t) (x < cap ? x : cap);
            }
            uint32_t total = __reduce_add_sync(all_lanes, t);
            while (total < b.slices)
            {
                bool const open = is && !(own && t >= cap);
                uint32_t const finish = l + (t + 1) * cost;
                uint32_t const best = __reduce_min_sync(all_lanes, open ? finish : UINT32_MAX);
                int const winner = __ffs(__ballot_sync(all_lanes, open && finish == best)) - 1;
                t += lane == winner;
                ++total;
            }
            if (is)
            {
                b.take[lane] = t;
                load[rank] = l + t * cost;
            }
            __syncwarp();
        }
}
} // namespace megamoe_gpu_plan_detail

/* Builds this rank's descriptors of the ROUTE_BALANCED copy from HALO-Q's
 * placement (count entries; fields strided by capacity as in its workspace), on
 * one warp of the fused copy kernel: every lane calls it and gets the same
 * return value. A copy's virtual stream lists the planes its owner can write
 * through a team object first (pair_dst_table entries that are not zero), so an
 * owner's block, which always starts at slice 0, stays inside them; without team
 * objects (no pair_dst_table) an owner outside a pair writes its block of every
 * plane to each rank of the pair through uc_dst_table, one range per rank;
 * members write every plane through the level's address in dst_table. Copies
 * that the resident table shows complete on every member are skipped. Lanes read the placement records, the issuers
 * of a large copy and the planes of a descriptor at once. scratch holds the
 * records and one load word per rank: megamoe_tma_balanced_scratch_bytes(). */
static constexpr __host__ __device__ size_t megamoe_tma_balanced_scratch_bytes()
{
    return sizeof(megamoe_gpu_plan_detail::BalancedCopy) * megamoe_gpu_plan_detail::kBalancedMaxCopies
        + MEGAMOE_TMA_GPU_PLAN_MAX_WORLD * sizeof(uint32_t);
}

static __device__ inline int megamoe_tma_build_balanced_plan_warp(MegamoeTmaGpuPlanConfig const* config, int count,
    int capacity, int32_t const* placement_fields, int32_t const* resident, void* scratch, size_t scratch_bytes,
    MegamoeTmaCopySegment* segments, uint64_t max_segments, MegamoeTmaCopyRange* ranges, uint64_t max_ranges,
    MegamoeTmaGpuPlanResult* result)
{
    using namespace megamoe_gpu_plan_detail;
    unsigned const all_lanes = 0xffffffffu;
    int const lane = threadIdx.x & 31;
    if (!result)
        return 1;
    if (lane == 0)
    {
        *result = MegamoeTmaGpuPlanResult{};
        result->error = 1;
    }
    __syncwarp();
    if (!config || !placement_fields || !scratch || !segments || !ranges || !max_segments || !max_ranges)
        return 1;
    MegamoeTmaGpuPlanConfig const& g = *config;
    if (!(g.route_features & MEGAMOE_TMA_GPU_PLAN_ROUTE_BALANCED) || g.world < 2
        || g.world > MEGAMOE_TMA_GPU_PLAN_MAX_WORLD || g.rank < 0 || g.rank >= g.world || g.helper_count <= 0
        || g.planes <= 0 || g.planes > 32 || g.home_count <= 0 || g.global_experts != g.home_count * g.world
        || g.level_count <= 0 || g.level_count > MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS || g.tma_source_load_percent < 1
        || g.tma_source_load_percent > 199 || !g.src_table || !g.dst_table || !g.plane_bytes || count < 0
        || count > capacity || count > kBalancedMaxCopies
        || scratch_bytes < count * sizeof(BalancedCopy) + MEGAMOE_TMA_GPU_PLAN_MAX_WORLD * sizeof(uint32_t))
        return 1;
    uint32_t const pull_cost = (uint32_t) g.tma_source_load_percent;
    int const slots = g.helper_count;
    // Lane p holds plane p: whether a team writes it, its bytes, and its place in a copy's virtual stream.
    bool const plane_lane = lane < g.planes;
    bool const team = plane_lane && g.pair_dst_table && g.pair_dst_table[lane] != 0;
    uint64_t const bytes = plane_lane ? g.plane_bytes[lane] : 0;
    uint32_t const team_mask = __ballot_sync(all_lanes, team);
    // Without team objects an owner outside a pair writes both of its ranks with unicast stores.
    bool const unicast_pairs = !g.pair_dst_table && g.uc_dst_table;
    uint32_t const below = (1u << lane) - 1;
    int const team_planes = __popc(team_mask);
    uint32_t const plane_mask = g.planes >= 32 ? all_lanes : (1u << g.planes) - 1;
    int const position = team ? __popc(team_mask & below) : team_planes + __popc(~team_mask & below & plane_mask);
    uint64_t vbegin = 0, total_bytes = 0, team_bytes = 0;
    for (int q = 0; q < g.planes; ++q)
    {
        uint64_t const bq = __shfl_sync(all_lanes, bytes, q);
        int const pq = __shfl_sync(all_lanes, position, q);
        vbegin += pq < position ? bq : 0;
        total_bytes += bq;
        team_bytes += team_mask >> q & 1 ? bq : 0;
    }
    uint64_t const slices64 = (total_bytes + MEGAMOE_TMA_COPY_SLICE_BYTES - 1) / MEGAMOE_TMA_COPY_SLICE_BYTES;
    if (!total_bytes || !balanced_loads_fit(count, slices64))
        return 1;
    auto* copies = static_cast<BalancedCopy*>(scratch);
    int n = 0;
    for (int base = 0; base < count; base += 32)
    {
        int const i = base + lane;
        bool const in = i < count;
        int expert = 0, level = 0, begin = 0, size = 1, owner = 0, helper = 0;
        if (in)
        {
            expert = placement_fields[i];
            level = placement_fields[capacity + i];
            begin = placement_fields[2 * capacity + i];
            size = placement_fields[3 * capacity + i];
            owner = placement_fields[4 * capacity + i];
            helper = placement_fields[5 * capacity + i];
        }
        bool const bad = in
            && (expert < 0 || expert >= g.global_experts || level < 0 || level >= g.level_count
                || size != g.group_sizes[level] || size < 2 || size + 1 > kBalancedMaxIssuers || begin < 0
                || begin % size || begin + size > g.world || owner != expert / g.home_count || helper < 0
                || helper >= slots);
        if (__any_sync(all_lanes, bad))
        {
            if (lane == 0)
                result->error = 1;
            return 1;
        }
        bool held = in && resident;
        if (held)
        {
#pragma unroll 4
            for (int rank = begin; rank < begin + size; ++rank)
                held &= resident[rank * slots + helper] == expert;
        }
        bool const keep = in && !held;
        uint32_t const kept = __ballot_sync(all_lanes, keep);
        if (keep)
        {
            BalancedCopy& b = copies[n + __popc(kept & below)];
            b.expert = expert;
            b.level = level;
            b.begin = begin;
            b.size = size;
            b.owner = owner;
            b.helper = helper;
            b.inside = owner >= begin && owner < begin + size;
            b.owner_issues = b.inside || (size == 2 && (team_planes || unicast_pairs));
            b.slices = (uint32_t) slices64;
            b.owner_cap
                = b.inside || unicast_pairs ? UINT32_MAX : (uint32_t) (team_bytes / MEGAMOE_TMA_COPY_SLICE_BYTES);
            b.owner_cost = b.inside || !unicast_pairs ? kBalancedOwnerCost : kBalancedUnicastOwnerCost;
        }
        n += __popc(kept);
    }
    auto* load = reinterpret_cast<uint32_t*>(copies + n);
    load[lane] = 0;
    __syncwarp();
    balanced_split_warp(copies, n, load, pull_cost, kBalancedPasses, lane);
    uint64_t segs = 0, nranges = 0, prefix = 0;
    int owned = 0, pulled = 0, error = 0;
    for (int j = 0; j < n && !error; ++j)
    {
        BalancedCopy const& b = copies[j];
        int const me = balanced_issuer_of(b, g.rank);
        if (me < 0)
            continue;
        uint32_t const t = lane < balanced_issuers(b) ? b.take[lane] : 0;
        uint32_t const take = __shfl_sync(all_lanes, t, me);
        if (!take)
            continue;
        uint32_t const first = __reduce_add_sync(all_lanes, lane < me ? t : 0);
        bool const owner = b.owner_issues && me == 0;
        bool const outside = owner && !b.inside;
        bool const via_team = outside && !unicast_pairs;
        int const used = via_team ? team_planes : g.planes;
        uint64_t const cursor = via_team ? team_bytes : total_bytes;
        // An owner writing a pair with unicast stores issues its block once per rank of the pair.
        int const writes = outside && unicast_pairs ? 2 : 1;
        for (int w = 0; w < writes && !error; ++w)
        {
            bool bad = segs + used > max_segments || nranges >= max_ranges
                || first + take > (cursor + MEGAMOE_TMA_COPY_SLICE_BYTES - 1) / MEGAMOE_TMA_COPY_SLICE_BYTES;
            if (plane_lane && position < used && !bad)
            {
                uint64_t const src = g.src_table[(size_t) b.expert * g.planes + lane];
                uint64_t const dst = writes == 2
                    ? g.uc_dst_table[((size_t) (b.begin + w) * slots + b.helper) * g.planes + lane]
                    : via_team ? g.pair_dst_table[((size_t) (b.begin / 2) * slots + b.helper) * g.planes + lane]
                               : g.dst_table[((size_t) b.level * slots + b.helper) * g.planes + lane];
                bool const fault = !src || !dst || !bytes || (src & 3) || (dst & 3) || (bytes & 3)
                    || bytes > UINT64_MAX - vbegin || bytes > UINT64_MAX - src || bytes > UINT64_MAX - dst;
                if (!fault)
                {
                    MegamoeTmaCopySegment& s = segments[segs + position];
                    s.src = src;
                    s.dst = dst;
                    s.bytes = bytes;
                    s.virtual_begin = vbegin;
                }
                bad = fault;
            }
            if (__any_sync(all_lanes, bad))
            {
                error = 3;
                break;
            }
            if (lane == 0)
            {
                MegamoeTmaCopyRange& r = ranges[nranges];
                r.segment_begin = segs;
                r.segment_count = used;
                r.bytes = cursor;
                r.first_slice = first;
                r.slice_stride = 1;
                r.slice_count = take;
                r.prefix_end = prefix + take;
                r.slice_divisor = 1;
                r.slice_reciprocal = 0;
                r.flags = writes == 2 ? MEGAMOE_TMA_COPY_RANGE_UNICAST : 0;
            }
            segs += used;
            ++nranges;
            prefix += take;
        }
        owned += owner;
        pulled += !owner;
    }
    __syncwarp();
    if (lane == 0)
    {
        result->error = error;
        if (!error)
        {
            result->direct_slots = owned;
            result->scatter_slots = pulled;
            result->segment_count = segs;
            result->range_count = nranges;
            result->total_slices = prefix;
        }
    }
    __syncwarp();
    return error;
}
#endif

#undef MEGAMOE_GP_HD
#endif
