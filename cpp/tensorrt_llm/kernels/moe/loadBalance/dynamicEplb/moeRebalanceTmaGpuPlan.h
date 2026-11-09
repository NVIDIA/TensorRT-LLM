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
/* Device-resident ABI8 plan input for the fused TMA copy kernel. */
#ifndef MEGAMOE_TMA_GPU_PLAN_H
#define MEGAMOE_TMA_GPU_PLAN_H
#include <stdint.h>

#if defined(__CUDACC__)
#define MEGAMOE_TMA_GPU_PLAN_HD __host__ __device__
#else
#define MEGAMOE_TMA_GPU_PLAN_HD
#endif

#define MEGAMOE_TMA_GPU_PLAN_MAX_WORLD 32
#define MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS 4
/* ABI7: pair-level owner-direct records and the wider channel stride. ABI8: a
 * pair-level record names a whole even-aligned pair, which may hold its owner. */
#define MEGAMOE_TMA_GPU_PLAN_ABI_VERSION 8
#define MEGAMOE_TMA_GPU_PLAN_MAX_OWNER_RECORDS 32

/* Records one owner may publish per plan: HALO-Q's candidate count
 * min(4 * S, H, 32). Each becomes at most one outgoing record. */
static inline MEGAMOE_TMA_GPU_PLAN_HD int megamoe_tma_gpu_plan_owner_records(int helpers, int home)
{
    int const candidates
        = helpers < MEGAMOE_TMA_GPU_PLAN_MAX_OWNER_RECORDS / 4 ? 4 * helpers : MEGAMOE_TMA_GPU_PLAN_MAX_OWNER_RECORDS;
    return candidates < home ? candidates : home;
}

/* Plan channel stride (owner_stride): it holds the S helper rows and up to
 * min(4 * S, 32) outgoing records, padded to whole int4 groups. The channel is
 * 4 header words and seven arrays of this stride. */
static inline MEGAMOE_TMA_GPU_PLAN_HD int megamoe_tma_gpu_plan_stride(int helpers)
{
    int const records
        = helpers < MEGAMOE_TMA_GPU_PLAN_MAX_OWNER_RECORDS / 4 ? 4 * helpers : MEGAMOE_TMA_GPU_PLAN_MAX_OWNER_RECORDS;
    int const rows = helpers > records ? helpers : records;
    return (rows + 3) & ~3;
}

/* route_features bits. EXTERNAL: an owner outside a proper multicast team writes
 * that team's multicast aperture (slot mode 2, foreign_dst_table). OWNER_UNICAST:
 * records at the pair level, the level index level_count right after the
 * multicast levels, are written by their owner itself (slot mode 3): without
 * PAIR_MULTICAST they name one receiver or an even-aligned pair that never holds
 * the owner, and the owner writes each receiver's helper row with unicast stores
 * (uc_dst_table). PAIR_MULTICAST (only with OWNER_UNICAST): they name a whole
 * even-aligned pair, the owner possibly inside, and the owner multicasts each
 * plane that has a pair team into it once (pair_dst_table); other planes still
 * go by unicast. */
#define MEGAMOE_TMA_GPU_PLAN_ROUTE_EXTERNAL 1
#define MEGAMOE_TMA_GPU_PLAN_ROUTE_OWNER_UNICAST 2
#define MEGAMOE_TMA_GPU_PLAN_ROUTE_PAIR_MULTICAST 4
/* BALANCED: the copy ignores the published modes and splits every HALO-Q copy
 * between its owner and its destination members, levelled per layer over all
 * ranks (megamoe_tma_build_balanced_plan_warp). PAIR_MULTICAST may come without
 * OWNER_UNICAST: then pair_dst_table only lets an owner outside a pair write
 * its share of the large planes through its team object. */
#define MEGAMOE_TMA_GPU_PLAN_ROUTE_BALANCED 8
#define MEGAMOE_TMA_GPU_PLAN_ROUTE_MASK 15

/* Published slot modes: 0 scatter, 1 owner multicast, 2 external multicast,
 * 3 owner-direct unicast. A receiver copies nothing for modes 2 and 3; their
 * owner gets an outgoing record (expert, target, helper) instead. */
#define MEGAMOE_TMA_GPU_PLAN_MODE_UNICAST 3

/* Outgoing targets below this base name proper multicast teams, widest level
 * first. A pair-level (owner-direct) record encodes its ranks
 * [begin, begin + size), size 1 or 2, as base + 2 * begin + size - 1. */
#define MEGAMOE_TMA_GPU_PLAN_UNICAST_TARGET_BASE 65536

/* Immutable geometry and device addresses. Upload these tables once at bind.
 * Source entries: [global_experts][planes]. Destinations: [level][helper][plane].
 * External destinations: [target][helper][plane]. Unicast destinations:
 * [receiver][helper][plane], row H + helper of the receiver's plane as this rank
 * maps it (its own row for this rank). Pair destinations: [pair][helper][plane],
 * the multicast address through which this rank writes row H + helper of both
 * ranks of pair j (ranks 2j and 2j + 1), or 0 for a plane without a pair team.
 * Bytes: [plane]. No entry here depends on the current HALO-Q result. All
 * pointer values are device pointers.
 */
typedef struct MegamoeTmaGpuPlanConfig
{
    int world, rank, helper_count, planes, global_experts, home_count;
    int owner_stride, level_count, target_count;
    int plan_abi_version, plan_words, route_features;
    int tma_route;               /* 0: published modes; 1: source; 2: scatter. */
    int tma_source_load_percent; /* Source quota = slices * percent / (100 * team); with ROUTE_BALANCED
                                    the cost of a pulled slice relative to 100 for an owner's local one. */
    int group_sizes[MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS];
    uint64_t const *src_table, *dst_table, *foreign_dst_table, *plane_bytes;
    uint64_t const* uc_dst_table;   /* Present exactly with ROUTE_OWNER_UNICAST. */
    uint64_t const* pair_dst_table; /* Present exactly with ROUTE_PAIR_MULTICAST. */
} MegamoeTmaGpuPlanConfig;

/* Per-CTA output; only successful construction makes counts consumable.
 * Error 1: malformed plan/config; 2: forced source outside its destination team;
 * 3: invalid address, byte count, capacity, or arithmetic overflow.
 */
typedef struct MegamoeTmaGpuPlanResult
{
    int error;
    int direct_slots, scatter_slots;
    int weighted_domains, uniform_domains, mixed_source_domains, foreign_source_domains;
    uint64_t segment_count, range_count, total_slices;
    uint64_t weighted_total_slices, weighted_source_quota_slices, weighted_local_slices;
} MegamoeTmaGpuPlanResult;
#endif
