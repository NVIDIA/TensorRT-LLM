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
 */
#ifndef MEGAMOE_TMA_GPU_PLAN_BUILDER_CUH
#define MEGAMOE_TMA_GPU_PLAN_BUILDER_CUH
#include <stddef.h>
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTmaGpuPlan.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.h"
#if defined(__CUDACC__)
#define MEGAMOE_GP_HD __host__ __device__
#else
#define MEGAMOE_GP_HD
#endif
namespace megamoe_gpu_plan_detail {
struct Builder : MegamoeTmaGpuPlanConfig {
    const int32_t *plan;
    int32_t *slot_scratch;
    MegamoeTmaCopySegment *tma_segments;
    MegamoeTmaCopyRange *tma_ranges;
    size_t max_commands, tma_max_ranges;
    size_t tma_segment_count, tma_range_count, last_commands;
    int last_direct_slots, last_scatter_slots;
    int last_tma_weighted_domains, last_tma_uniform_domains;
    int last_tma_mixed_source_domains, last_tma_foreign_source_domains;
    uint64_t last_tma_weighted_total_slices, last_tma_weighted_source_quota_slices;
    uint64_t last_tma_weighted_local_slices, last_tma_local_slices;
};
static MEGAMOE_GP_HD inline uint64_t tma_weighted_source_quota(
    uint64_t slices, unsigned members, unsigned percent) {
    return slices * percent / (100U * members);
}
static MEGAMOE_GP_HD inline int target_geometry(const Builder *c, int target,
                           int *level_out, int *begin_out)
{
    if (target < 0 || target >= c->target_count)
        return -1;
    for (int level = 1; level < c->level_count; ++level) {
        int count = c->world / c->group_sizes[level];
        if (target < count) {
            *level_out = level;
            *begin_out = target * c->group_sizes[level];
            return 0;
        }
        target -= count;
    }
    return -1;
}

static MEGAMOE_GP_HD inline int append_tma_segment(Builder *c, size_t *segments,
                              uint64_t *cursor, uint64_t src, uint64_t dst,
                              size_t bytes)
{
    if (*segments >= c->max_commands || !src || !dst || !bytes ||
        (src & 3) || (dst & 3) || (bytes & 3) ||
        bytes > UINT64_MAX - *cursor || bytes > UINT64_MAX - src ||
        bytes > UINT64_MAX - dst)
        return -1;
    MegamoeTmaCopySegment *s = &c->tma_segments[(*segments)++];
    s->src = src; s->dst = dst; s->bytes = bytes; s->virtual_begin = *cursor;
    *cursor += bytes;
    return 0;
}

/* The validated arena binds one UC object per (owner, plane) and one MC
 * object per (team, plane). The caller resets identity at plane/domain
 * boundaries and gives normal and external teams separate namespaces.
 * Numerical pointer adjacency alone is not enough to merge MC destinations. */
static MEGAMOE_GP_HD inline int append_tma_coalesced_segment(Builder *c, size_t *segments,
                                       uint64_t *cursor, uint64_t src,
                                       uint64_t dst, size_t bytes,
                                       uint64_t identity, uint64_t *previous_identity)
{
    if (!src || !dst || !bytes || (src & 3) || (dst & 3) || (bytes & 3) ||
        bytes > UINT64_MAX - *cursor || bytes > UINT64_MAX - src ||
        bytes > UINT64_MAX - dst)
        return -1;
    if (*segments && *previous_identity == identity) {
        MegamoeTmaCopySegment *previous = &c->tma_segments[*segments - 1];
        if (previous->bytes <= UINT64_MAX - previous->src &&
            previous->bytes <= UINT64_MAX - previous->dst &&
            previous->bytes <= UINT64_MAX - previous->virtual_begin &&
            previous->src + previous->bytes == src &&
            previous->dst + previous->bytes == dst &&
            previous->virtual_begin + previous->bytes == *cursor &&
            bytes <= UINT64_MAX - previous->bytes) {
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

static MEGAMOE_GP_HD inline int append_tma_range(Builder *c, size_t *ranges, uint64_t *prefix,
                            size_t segment_begin, size_t segment_count,
                            uint64_t bytes, uint64_t first, uint64_t step,
                            uint64_t slices, uint64_t divisor, uint64_t reciprocal)
{
    /* These constant bounds also avoid runtime division for overflow checks.
     * Valid 8KiB streams are smaller than this conservative slice bound. */
    if (!step || !divisor || divisor > 100U * MEGAMOE_TMA_GPU_PLAN_MAX_WORLD ||
        slices > UINT64_MAX / (100U * MEGAMOE_TMA_GPU_PLAN_MAX_WORLD)) return -1;
    uint64_t scaled_limit = slices * divisor;
    if (first >= scaled_limit) return 0;
    uint64_t count = 1 + (scaled_limit - 1 - first) / step;
    if (*ranges >= c->tma_max_ranges || count > UINT64_MAX - *prefix)
        return -1;
    MegamoeTmaCopyRange *r = &c->tma_ranges[(*ranges)++];
    r->segment_begin = segment_begin; r->segment_count = segment_count;
    r->bytes = bytes; r->first_slice = first; r->slice_stride = step;
    r->slice_count = count; *prefix += count; r->prefix_end = *prefix;
    r->slice_divisor = divisor; r->slice_reciprocal = reciprocal;
    return 0;
}

static MEGAMOE_GP_HD inline size_t build_tma_descriptors(Builder *c, int outgoing_count,
                                   int *plan_error)
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
    for (int slot = 0; slot < slots; ++slot) {
        if (experts[slot] < 0 || modes[slot] == 2)
            continue;
        if (c->tma_route == 1) {
            int group = c->group_sizes[levels[slot]];
            int begin = c->rank / group * group;
            if (owners[slot] < begin || owners[slot] >= begin + group) {
                *plan_error = 2;
                return 0;
            }
            modes[slot] = 1;
        } else if (c->tma_route == 2) {
            modes[slot] = 0;
        }
        direct += modes[slot] == 1;
        scatter += modes[slot] == 0;
    }
    /* Domain zero is source-owned work (including outgoing direct). A scatter
     * domain must have one common member set, so it is separated per level;
     * all ranks in that aligned team see the same plane/slot byte order. */
    for (int domain = 0; domain <= c->level_count; ++domain) {
        size_t begin_segment = segments;
        uint64_t cursor = 0;
        /* Keep one uninterrupted virtual stream per routing domain. Plane-major
         * order groups the large aligned weights before the scalar planes; the
         * physical arena and source ownership are unchanged. Within each plane,
         * normal source-owned slots precede existing external outgoing records. */
        for (int plane = 0; plane < c->planes; ++plane) {
            uint64_t previous_identity = UINT64_MAX;
            for (int slot = 0; slot < slots; ++slot) {
                if (experts[slot] < 0 || modes[slot] == 2)
                    continue;
                if (domain == 0 ? (modes[slot] != 1 || owners[slot] != c->rank)
                                : (modes[slot] != 0 || levels[slot] != domain - 1))
                    continue;
                int level = levels[slot];
                size_t si = (size_t)experts[slot] * c->planes + plane;
                size_t di = ((size_t)level * slots + slot) * c->planes + plane;
                uint64_t identity = ((uint64_t)owners[slot] << 32) | (uint32_t)level;
                if (append_tma_coalesced_segment(c, &segments, &cursor,
                                                c->src_table[si], c->dst_table[di],
                                                c->plane_bytes[plane], identity,
                                                &previous_identity))
                    return 0;
            }
            if (domain == 0) {
                for (int i = 0; i < outgoing_count; ++i) {
                    int expert = c->plan[4 + 4 * stride + i];
                    int target = c->plan[4 + 5 * stride + i];
                    int helper = c->plan[4 + 6 * stride + i];
                    size_t si = (size_t)expert * c->planes + plane;
                    size_t di = ((size_t)target * slots + helper) * c->planes + plane;
                    uint64_t identity = ((uint64_t)c->rank << 32) |
                                        (uint32_t)(c->level_count + target);
                    if (append_tma_coalesced_segment(c, &segments, &cursor,
                                                    c->src_table[si], c->foreign_dst_table[di],
                                                    c->plane_bytes[plane], identity,
                                                    &previous_identity))
                        return 0;
                }
            }
        }
        if (!cursor)
            continue;
        uint64_t step = domain ? (uint64_t)c->group_sizes[domain - 1] : 1;
        uint64_t first = domain ? (uint64_t)c->rank % step : 0;
        uint64_t slices = cursor / MEGAMOE_TMA_COPY_SLICE_BYTES +
                          (cursor % MEGAMOE_TMA_COPY_SLICE_BYTES != 0);
        size_t before_ranges = ranges;
        int common_source = -1;
        int weighted = 0;
        if (domain && c->tma_source_load_percent != 100) {
            for (int slot = 0; slot < slots; ++slot) {
                if (experts[slot] < 0 || modes[slot] != 0 || levels[slot] != domain - 1)
                    continue;
                if (common_source == -1) common_source = owners[slot];
                else if (common_source != owners[slot]) { common_source = -2; break; }
            }
            int team_begin = c->rank / (int)step * (int)step;
            if (common_source == -2) ++c->last_tma_mixed_source_domains;
            else if (common_source < team_begin || common_source >= team_begin + (int)step)
                ++c->last_tma_foreign_source_domains;
            else weighted = 1;
        }
        if (weighted) {
            unsigned members = (unsigned)step;
            unsigned source_local = (unsigned)common_source % members;
            unsigned percent = (unsigned)c->tma_source_load_percent;
            uint64_t denominator = 100U * members;
            uint64_t before_prefix = prefix;
            uint64_t source_quota = tma_weighted_source_quota(slices, members, percent);
            if (slices > UINT64_MAX - c->last_tma_weighted_total_slices ||
                source_quota > UINT64_MAX - c->last_tma_weighted_source_quota_slices)
                return 0;
            ++c->last_tma_weighted_domains;
            c->last_tma_weighted_total_slices += slices;
            c->last_tma_weighted_source_quota_slices += source_quota;
            uint64_t divisor, reciprocal;
            if (c->rank == common_source) {
                first = denominator - 1;
                step = denominator;
                divisor = percent;
                reciprocal = divisor == 1 ? 0 : UINT64_MAX / divisor;
            } else {
                unsigned peer_order = ((unsigned)c->rank % members + members -
                                       source_local) % members - 1;
                first = peer_order * denominator;
                step = (members - 1) * denominator;
                divisor = denominator - percent;
                reciprocal = divisor == 1 ? 0 : UINT64_MAX / divisor;
            }
            /* floor((first+j*step)/divisor) is the inverse of the source
             * quota events and peer round-robin. The whole existing virtual
             * stream stays intact, with increasing slices and a continuous
             * rank-local prefix across domains for SM/worker round-robin. */
            if (append_tma_range(c, &ranges, &prefix, begin_segment,
                                 segments - begin_segment, cursor, first, step,
                                 slices, divisor, reciprocal))
                return 0;
            c->last_tma_weighted_local_slices += prefix - before_prefix;
        } else {
            if (domain) ++c->last_tma_uniform_domains;
            if (append_tma_range(c, &ranges, &prefix, begin_segment,
                                 segments - begin_segment, cursor, first, step, slices, 1, 0))
                return 0;
        }
        if (ranges == before_ranges)
            segments = begin_segment; /* This rank owns none of this stream's slices. */
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
static MEGAMOE_GP_HD inline size_t build_descriptors(Builder *c, int *plan_error)
{
    const volatile int32_t *plan = c->plan;
    const int slots = c->helper_count;
    const int stride = c->owner_stride;
    const int quota = slots < c->home_count ? slots : c->home_count;
    int *experts = c->slot_scratch;
    int *levels = experts + slots;
    int *owners = levels + slots;
    int *modes = owners + slots;
    int owner_counts[MEGAMOE_TMA_GPU_PLAN_MAX_WORLD] = {0};
    unsigned char seen_experts[384] = {0};
    int incoming_slots = 0;
    int active = 0, common_level = -1;
    int global_max_send = plan[1];
    int outgoing_count = plan[3];
    *plan_error = 1;
    if (c->plan_abi_version != 6 || c->plan_words != 4 + 7 * stride ||
        plan[2] != 0 || global_max_send < 0 || global_max_send > quota ||
        outgoing_count < 0 || outgoing_count > quota ||
        (!c->route_features && outgoing_count))
        return 0;

    /* Published modes are global decisions. Validate the complete local row
     * before constructing any descriptor; inactive capacity never becomes work. */
    for (int slot = 0; slot < slots; ++slot) {
        int expert = experts[slot] = plan[4 + slot];
        int level = levels[slot] = plan[4 + stride + slot];
        int owner = owners[slot] = plan[4 + 2 * stride + slot];
        int mode = modes[slot] = plan[4 + 3 * stride + slot];
        if (expert == -1) {
            if (level != -1 || owner != -1 || mode != 0)
                return 0;
            continue;
        }
        if (expert < 0 || expert >= c->global_experts ||
            level < 0 || level >= c->level_count ||
            owner < 0 || owner >= c->world || owner != expert / c->home_count ||
            mode < 0 || mode > 2 || seen_experts[expert])
            return 0;
        seen_experts[expert] = 1;
        if (++owner_counts[owner] > global_max_send)
            return 0;
        int group = c->group_sizes[level];
        int begin = c->rank / group * group;
        int local = owner >= begin && owner < begin + group;
        if ((mode == 1 && !local) ||
            (mode == 2 && (!c->route_features || !c->foreign_dst_table ||
                           local || level == 0)))
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
     * auto-policy heuristics are not descriptor safety constraints. */
    if (incoming_slots) {
        if (common_level <= 0 || global_max_send == 0 || global_max_send > 2 ||
            (global_max_send == 2 && active < 3))
            return 0;
        int group = c->group_sizes[common_level];
        int begin = c->rank / group * group;
        for (int slot = 0; slot < slots; ++slot) {
            if (experts[slot] < 0)
                continue;
            int external = owners[slot] < begin || owners[slot] >= begin + group;
            if (external != (modes[slot] == 2))
                return 0;
        }
    }
    if (outgoing_count && (global_max_send == 0 || global_max_send > 2 ||
                           (active && common_level <= 0)))
        return 0;
    if (owner_counts[c->rank] + outgoing_count > global_max_send)
        return 0;
    for (int i = 0; i < outgoing_count; ++i) {
        int expert = plan[4 + 4 * stride + i];
        int target = plan[4 + 5 * stride + i];
        int helper = plan[4 + 6 * stride + i];
        int level, begin;
        if (expert < c->rank * c->home_count ||
            expert >= (c->rank + 1) * c->home_count || seen_experts[expert] ||
            helper < 0 || helper >= slots ||
            target_geometry(c, target, &level, &begin))
            return 0;
        int group = c->group_sizes[level];
        if (c->rank >= begin && c->rank < begin + group)
            return 0;
        seen_experts[expert] = 1;
        for (int j = 0; j < i; ++j)
            if (plan[4 + 5 * stride + j] == target &&
                plan[4 + 6 * stride + j] == helper)
                return 0;
    }
    for (int i = outgoing_count; i < stride; ++i)
        if (plan[4 + 4 * stride + i] != -1 ||
            plan[4 + 5 * stride + i] != -1 ||
            plan[4 + 6 * stride + i] != -1)
            return 0;
    for (int slot = slots; slot < stride; ++slot)
        if (plan[4 + slot] != -1 || plan[4 + stride + slot] != -1 ||
            plan[4 + 2 * stride + slot] != -1 || plan[4 + 3 * stride + slot] != 0)
            return 0;

    return build_tma_descriptors(c, outgoing_count, plan_error);
}

} // namespace megamoe_gpu_plan_detail

/* Host/device parity permits differential tests against the CPU submitter.
 * Counts remain zero on failure: callers must not publish successful READY.
 * Caller supplies valid memory for the full immutable tables and scratch.
 */
static MEGAMOE_GP_HD inline int megamoe_tma_build_gpu_plan(
    const MegamoeTmaGpuPlanConfig *config, const int32_t *device_plan,
    int32_t *slot_scratch, MegamoeTmaCopySegment *segments, uint64_t max_segments,
    MegamoeTmaCopyRange *ranges, uint64_t max_ranges, MegamoeTmaGpuPlanResult *result) {
    if (!result) return 1;
    *result = MegamoeTmaGpuPlanResult{};
    result->error = 1;
    if (!config || !device_plan || !slot_scratch || !segments || !ranges ||
        !max_segments || !max_ranges || max_segments > SIZE_MAX || max_ranges > SIZE_MAX)
        return 1;
    const MegamoeTmaGpuPlanConfig &g = *config;
    if (g.world < 2 || g.world > MEGAMOE_TMA_GPU_PLAN_MAX_WORLD || (g.world & 1) ||
        g.rank < 0 || g.rank >= g.world || g.helper_count <= 0 ||
        g.planes <= 0 || g.global_experts <= 0 || g.global_experts > 384 ||
        g.global_experts % g.world || g.home_count != g.global_experts / g.world ||
        g.level_count <= 0 || g.level_count > MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS ||
        g.owner_stride != ((int64_t(g.helper_count) + 3) & ~INT64_C(3)) ||
        int64_t(g.plan_words) != 4 + 7 * int64_t(g.owner_stride) ||
        g.plan_abi_version != 6 || (g.route_features != 0 && g.route_features != 1) ||
        g.tma_route < 0 || g.tma_route > 2 ||
        g.tma_source_load_percent < 1 || g.tma_source_load_percent > 199 ||
        !g.src_table || !g.dst_table || !g.plane_bytes ||
        (g.route_features && !g.foreign_dst_table) ||
        (!g.route_features && g.foreign_dst_table) || device_plan[0] != 0)
        return 1;
    int targets = 0;
    for (int level = 0; level < g.level_count; ++level) {
        int group = g.group_sizes[level];
        if (group < (level ? 4 : 2) || group > g.world || g.world % group ||
            (!level && group != g.world) ||
            (level && (group >= g.group_sizes[level - 1] ||
                       g.group_sizes[level - 1] % group))) return 1;
        if (level) targets += g.world / group;
    }
    if (g.target_count != targets || (g.route_features && !targets)) return 1;
    megamoe_gpu_plan_detail::Builder c{};
    static_cast<MegamoeTmaGpuPlanConfig &>(c) = g;
    c.plan = device_plan;
    c.slot_scratch = slot_scratch;
    c.tma_segments = segments;
    c.tma_ranges = ranges;
    c.max_commands = (size_t)max_segments;
    c.tma_max_ranges = (size_t)max_ranges;
    int error = 1;
    megamoe_gpu_plan_detail::build_descriptors(&c, &error);
    result->error = error;
    if (error) return error;
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
#undef MEGAMOE_GP_HD
#endif
