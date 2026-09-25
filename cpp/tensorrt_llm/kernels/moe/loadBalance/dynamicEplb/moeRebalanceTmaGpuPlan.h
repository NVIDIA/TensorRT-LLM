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
/* Device-resident ABI6 plan input for the fused TMA copy kernel. */
#ifndef MEGAMOE_TMA_GPU_PLAN_H
#define MEGAMOE_TMA_GPU_PLAN_H
#include <stdint.h>

#define MEGAMOE_TMA_GPU_PLAN_MAX_WORLD 32
#define MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS 4

/* Immutable geometry and device addresses. Upload these tables once at bind.
 * Source entries: [global_experts][planes]. Destinations: [level][helper][plane].
 * External destinations: [target][helper][plane]. Bytes: [plane]. No entry here
 * depends on the current HALO-Q result. All pointer values are device pointers.
 */
typedef struct MegamoeTmaGpuPlanConfig {
    int world, rank, helper_count, planes, global_experts, home_count;
    int owner_stride, level_count, target_count;
    int plan_abi_version, plan_words, route_features;
    int tma_route; /* 0: published modes; 1: source; 2: scatter. */
    int tma_source_load_percent; /* Source quota = slices * percent / (100 * team). */
    int group_sizes[MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS];
    const uint64_t *src_table, *dst_table, *foreign_dst_table, *plane_bytes;
} MegamoeTmaGpuPlanConfig;

/* Per-CTA output; only successful construction makes counts consumable.
 * Error 1: malformed plan/config; 2: forced source outside its destination team;
 * 3: invalid address, byte count, capacity, or arithmetic overflow.
 */
typedef struct MegamoeTmaGpuPlanResult {
    int error;
    int direct_slots, scatter_slots;
    int weighted_domains, uniform_domains, mixed_source_domains, foreign_source_domains;
    uint64_t segment_count, range_count, total_slices;
    uint64_t weighted_total_slices, weighted_source_quota_slices, weighted_local_slices;
} MegamoeTmaGpuPlanResult;
#endif
