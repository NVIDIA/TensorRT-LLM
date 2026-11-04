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

// The row-norm exchange of the MoE hop boundaries (MhH1, MhF1) between the U > 1 units of a token row
// (MSO_MK_MH_XCHG, moe_hops_layout.cuh). A boundary phase reduces each warp's 8-column-per-thread partials to one
// FP32 slot (the stock boundary's warp slot); the row's 32 slots are then summed in slot order by every warp.
//
// kMhXchg == 2 (cluster DSMEM): the U units of a row are the U CTAs of one thread-block cluster (cluster rank = part
// q). A warp's lanes 0 .. U - 1 store its slot into the slot area of cluster rank `lane` (st.relaxed.cluster through
// the mapa address, the payload as its own flag), and every warp polls the 32 slots in its own CTA's shared memory
// (ld.relaxed.cluster, __all_sync exit).
//   Ownership: the slot area holds kMhXchgPhases x 32 words; each word of a phase has exactly one writer per launch
//   (the warp that owns the slot) and the CTA's own warps as readers; a phase is used once per launch, so there is no
//   in-launch reuse and no reset. Shared memory does not survive a launch: every CTA writes kEmpty into its area at
//   entry, then arrives on the cluster barrier (release); it waits on it (acquire) after griddepcontrol.wait and before
//   any boundary, so every peer's area is initialised before the first remote store. A CTA leaves the launch only
//   after it has read all words of its last phase, i.e. after the last remote store into its shared memory landed; its
//   own remote stores target peers that stay resident until they have read them.
//   Progress: the cluster's CTAs are co-scheduled; a unit waits only for the slots of the other units of its row,
//   which run the same boundary on the other CTAs of its cluster.
// kMhXchg == 1: the arena row of moe_hops.cuh, but only warp 0 polls it; the other warps take the slots from shared
// memory after a CTA barrier.

#pragma once

constexpr int kMhXchgPhases = 4;   // MhH1 norms 1-2, MhF1 norms 1-2
constexpr int kMhXchgSmall = 3072; // the slot area in the small shared-memory area: [phases][32] u32
static_assert(kSmallWeights + kMaxRoles * 3 * 8 <= kMhXchgSmall, "the slot area overlaps the weight-table row");
static_assert(kMhXchgSmall + kMhXchgPhases * 32 * 4 <= kSmemSmallBytes, "the slot area does not fit");
static_assert(kMhClusterDim == 1 || 160 % kMhClusterDim == 0, "row units (from CTA 160) start on a cluster boundary");

__device__ __forceinline__ uint32_t clusterCtaRank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ uint32_t clusterNumCtas()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
    return r;
}

// The shared::cluster address of `addr` (this CTA's shared::cta address) in cluster CTA `rank`.
__device__ __forceinline__ uint32_t mapaShared(uint32_t addr, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

__device__ __forceinline__ void stClusterRelaxed32(uint32_t addr, uint32_t v)
{
    asm volatile("st.relaxed.cluster.shared::cluster.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t ldClusterRelaxed32(uint32_t addr)
{
    uint32_t v;
    asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ void clusterArriveRelease()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWaitAcquire()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Kernel entry, every thread of every CTA (kMhClusterDim > 1): the launch must have the build's cluster shape; the slot
// area to kEmpty; arrive on the cluster barrier.
__device__ __forceinline__ void mhXchgEntry(uint8_t* smem)
{
    if constexpr (kMhClusterDim > 1)
    {
        if (clusterNumCtas() != static_cast<uint32_t>(kMhClusterDim))
        {
            __trap();
        }
        if (threadIdx.x < kMhXchgPhases * 32)
        {
            reinterpret_cast<uint32_t volatile*>(smem + kSmemSmall + kMhXchgSmall)[threadIdx.x] = kEmpty;
        }
        clusterArriveRelease();
    }
}

// After griddepcontrol.wait, every thread of every CTA: every peer's slot area is initialised.
__device__ __forceinline__ void mhXchgReady()
{
    if constexpr (kMhClusterDim > 1)
    {
        clusterWaitAcquire();
    }
}

// The 32 slots of one row-norm phase through the cluster's shared memory: warp slot v of `slot` (0..31) to every unit
// of the row; returns this lane's slot (lane l: slot l) once all 32 arrived. All 256 threads call it, once per phase.
template <int kV>
__device__ __forceinline__ float mhXchgDsmem(Ctx const& ctx, float const (&v)[kV], int const (&slot)[kV], int phase)
{
    int const lane = threadIdx.x & 31;
    uint32_t const area = ctx.smemBase + kSmemSmall + kMhXchgSmall + 4 * 32 * phase;
    if (lane < kMhRowU)
    {
#pragma unroll
        for (int j = 0; j < kV; ++j)
        {
            stClusterRelaxed32(
                mapaShared(area + 4 * slot[j], static_cast<uint32_t>(lane)), lamport(__float_as_uint(v[j])));
        }
    }
    uint32_t const a = area + 4 * lane;
    uint32_t w = ldClusterRelaxed32(a);
    uint64_t t0 = 0;
    while (!warpAll(w != kEmpty))
    {
        spinCheck(ctx, t0, kCodeMhSsq, phase, 2);
        if (w == kEmpty)
        {
            w = ldClusterRelaxed32(a);
        }
    }
    return __uint_as_float(w);
}
