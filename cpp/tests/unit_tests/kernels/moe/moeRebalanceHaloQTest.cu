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

/* The HALO-Q rebalance plan (placement, quota repair, helper-slot coloring) on routing counts given directly to one
 * CTA, without the cross-rank count exchange or the copy.  Each case checks the plan's invariants, recomputes the
 * coloring on the host, and pins the placement of fixed inputs, so a change to the policy shows up here first. */

#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQKernel.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace tensorrt_llm::tests::kernels::haloq
{
using namespace ::tensorrt_llm::kernels;

constexpr int kThreads = 512;
constexpr int kPlanInts = 2 + 6 * kMaxBroadcasts;

int planField(std::vector<int> const& plan, int field, int b)
{
    return plan[2 + field * kMaxBroadcasts + b];
}

/* halo_q_scheduler_body's planning phase after the count exchange, on its shared-memory layout: prepare,
 * place_halo_q, then warp 8 colors for local rank 0 while the quota repair runs.  The test only adds a snapshot of
 * the placement's quota rows before the repair, and colors again for every other local rank, each on a fresh copy of
 * the plan.  colors holds per rank the helper of every broadcast and the rank's out ids/levels/owners. */
__global__ void __launch_bounds__(kThreads, 1) planKernel(int const* allCounts, int ep, int experts, int helpers,
    int capacity, int sharedInts, int* plan, int* quotaPre, int* quotaPost, int* planCopy, int* colors, int* status)
{
    extern __shared__ int shared[];
    int offset = ep * experts;
    int* counts = shared;
    int* histogram = shared + offset;
    offset += experts;
    int* load = shared + offset;
    offset += ep;
    int* quota = shared + offset;
    offset += capacity * ep;
    int* selectedFlags = shared + offset;
    offset += experts;
    int* recvUsed = shared + offset;
    offset += ep * kHelperMaskWords;
    int* ring = shared + offset;
    offset += ep;
    int* order = shared + offset;
    offset += capacity;
    int* hostMask = shared + offset;
    offset += capacity;
    int* schedule = shared + offset;
    offset += kScheduleInts;
    int* running = shared + offset;

    /* The planner CTA's shared memory holds whatever ran before it, so every word starts poisoned; exchange_counts
     * then fills the counts and ends at a block barrier. */
    for (int i = threadIdx.x; i < sharedInts; i += blockDim.x)
        shared[i] = -1;
    __syncthreads();
    for (int i = threadIdx.x; i < ep * experts; i += blockDim.x)
        counts[i] = allCounts[i];
    __syncthreads();
    if (uses_register_prepare(ep, experts))
        prepare_halo_m_register(counts, histogram, load, selectedFlags, ep, experts, helpers);
    else
        prepare_halo_m(counts, histogram, load, selectedFlags, ep, experts, helpers);
    int const broadcasts = place_halo_q(histogram, load, quota, selectedFlags, running, plan, ep, experts, helpers);
    // place_halo_q ends at a block barrier.
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
        quotaPre[i] = quota[i];
    __syncthreads();
    if (threadIdx.x >> 5 == 8)
        color_and_publish_local(plan, broadcasts, ep, helpers, 0, recvUsed, colors + kMaxBroadcasts,
            colors + kMaxBroadcasts + helpers, colors + kMaxBroadcasts + 2 * helpers, status, 1, nullptr);
    run_halo_q(counts, quota, plan, broadcasts, ep, experts, ring, order, hostMask, schedule);
    __syncthreads();
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
        quotaPost[i] = quota[i];
    for (int i = threadIdx.x; i < broadcasts; i += blockDim.x)
        colors[i] = plan[2 + kPlanHelper * kMaxBroadcasts + i];
    __syncthreads();

    int const stride = kMaxBroadcasts + 3 * helpers;
    for (int rank = 1; rank < ep; ++rank)
    {
        for (int i = threadIdx.x; i < kPlanInts; i += blockDim.x)
            planCopy[i] = plan[i];
        __syncthreads();
        int* out = colors + rank * stride;
        if (threadIdx.x >> 5 == 8)
            color_and_publish_local(planCopy, broadcasts, ep, helpers, rank, recvUsed, out + kMaxBroadcasts,
                out + kMaxBroadcasts + helpers, out + kMaxBroadcasts + 2 * helpers, status, 1, nullptr);
        __syncthreads();
        for (int i = threadIdx.x; i < broadcasts; i += blockDim.x)
            out[i] = planCopy[2 + kPlanHelper * kMaxBroadcasts + i];
        __syncthreads();
    }
}

struct Case
{
    char const* name;
    int ep;
    int experts;
    int helpers;
    bool hotHalf;    // the first half of the ranks is three times as loaded, else hot experts are spread at random
    int broadcasts;  // of the placement
    uint64_t golden; // placementHash() of the placement
};

/* 8192 tokens x top-8 per source rank, with +-25% noise per source rank.  An expert's share follows 1 / (1 + j) for a
 * seeded permutation j, or with hotHalf is three units on the first half of the owners and one on the rest. */
std::vector<int> makeCounts(Case const& c)
{
    uint32_t state = static_cast<uint32_t>(c.ep * 1000003 + c.experts * 101 + c.helpers);
    auto next = [&state]
    {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };
    int const home = c.experts / c.ep;
    std::vector<int> perm(c.experts);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = c.experts - 1; i > 0; --i)
        std::swap(perm[i], perm[next() % (i + 1)]);
    std::vector<int64_t> weight(c.experts);
    int64_t total = 0;
    for (int e = 0; e < c.experts; ++e)
    {
        weight[e] = c.hotHalf ? (e / home < c.ep / 2 ? 3 : 1) * (int64_t{1} << 20) : (int64_t{1} << 20) / (1 + perm[e]);
        total += weight[e];
    }
    std::vector<int> counts(c.ep * c.experts);
    for (int rank = 0; rank < c.ep; ++rank)
        for (int e = 0; e < c.experts; ++e)
            counts[rank * c.experts + e]
                = static_cast<int>(int64_t{8192 * 8} * weight[e] / total * (75 + next() % 51) / 100);
    return counts;
}

uint64_t fnv1a(uint64_t hash, int const* values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        for (int byte = 0; byte < 4; ++byte)
            hash = (hash ^ ((static_cast<uint32_t>(values[i]) >> (8 * byte)) & 0xffu)) * 0x100000001b3ull;
    return hash;
}

/* FNV-1a over the broadcast count, each broadcast's expert, level, group begin, group size and owner, then the
 * placement's quota rows: the plan before repair and coloring. */
uint64_t placementHash(std::vector<int> const& plan, std::vector<int> const& pre, int ep)
{
    int const broadcasts = plan[1];
    uint64_t hash = fnv1a(0xcbf29ce484222325ull, &broadcasts, 1);
    for (int b = 0; b < broadcasts; ++b)
        for (int field : {kPlanExpert, kPlanLevel, kPlanGroupBegin, kPlanGroupSize, kPlanOwner})
        {
            int const value = planField(plan, field, b);
            hash = fnv1a(hash, &value, 1);
        }
    return fnv1a(hash, pre.data(), static_cast<size_t>(broadcasts) * ep);
}

class HaloQPlannerTest : public ::testing::TestWithParam<Case>
{
};

TEST_P(HaloQPlannerTest, PlanInvariantsAndGolden)
{
    Case const& c = GetParam();
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    int major = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), cudaSuccess);
    if (major != 10)
        GTEST_SKIP() << "HALO-Q runs on SM10x (Blackwell and Rubin) only";
    int const ep = c.ep;
    int const experts = c.experts;
    int const helpers = c.helpers;
    int const home = experts / ep;
    int const capacity = broadcast_capacity(ep, experts, helpers);
    int const sharedInts = halo_q_shared_ints(ep, experts, helpers, kThreads);
    int const sharedBytes = sharedInts * static_cast<int>(sizeof(int));
    int maxShared = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), cudaSuccess);
    if (sharedBytes > maxShared)
        GTEST_SKIP() << "needs " << sharedBytes << " B of shared memory per block";

    std::vector<int> const counts = makeCounts(c);
    int const stride = kMaxBroadcasts + 3 * helpers;
    int *dCounts, *dPlan, *dPre, *dPost, *dPlanCopy, *dColors, *dStatus;
    ASSERT_EQ(cudaMalloc(&dCounts, counts.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dPlan, kPlanInts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dPre, capacity * ep * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dPost, capacity * ep * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dPlanCopy, kPlanInts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dColors, ep * stride * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dStatus, 8 * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dCounts, counts.data(), counts.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(dPlan, 0xff, kPlanInts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dColors, 0xff, ep * stride * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dStatus, 0, 8 * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaFuncSetAttribute(planKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes), cudaSuccess);
    planKernel<<<1, kThreads, sharedBytes>>>(
        dCounts, ep, experts, helpers, capacity, sharedInts, dPlan, dPre, dPost, dPlanCopy, dColors, dStatus);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<int> plan(kPlanInts), pre(capacity * ep), post(capacity * ep), colors(ep * stride), status(8);
    ASSERT_EQ(cudaMemcpy(plan.data(), dPlan, plan.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(pre.data(), dPre, pre.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(post.data(), dPost, post.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(colors.data(), dColors, colors.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(status.data(), dStatus, status.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int* p : {dCounts, dPlan, dPre, dPost, dPlanCopy, dColors, dStatus})
        cudaFree(p);

    int const broadcasts = plan[1];
    ASSERT_GT(broadcasts, 0);
    ASSERT_LE(broadcasts, capacity);
    for (int word : status)
        EXPECT_EQ(word, 0) << "helper coloring failed";

    /* Levels: the multicast ladder ep >> level, then, for EP >= 4, aligned rank pairs.  A rank sends at most
     * candidate_count() = min(S, home, 32) experts; a multicast group may include its owner or not. */
    int const depth = planner_depth(ep);
    int const pairLevel = pair_level_of(ep);
    std::vector<int64_t> mass(experts, 0), resident(ep, 0), loadPre(ep, 0), loadPost(ep, 0);
    for (int rank = 0; rank < ep; ++rank)
        for (int e = 0; e < experts; ++e)
            mass[e] += counts[rank * experts + e];
    for (int e = 0; e < experts; ++e)
        resident[e / home] += mass[e];
    loadPre = resident;
    loadPost = resident;
    std::vector<int> exports(ep, 0), seen(experts, 0);
    for (int b = 0; b < broadcasts; ++b)
    {
        int const e = planField(plan, kPlanExpert, b);
        int const level = planField(plan, kPlanLevel, b);
        int const begin = planField(plan, kPlanGroupBegin, b);
        int const size = planField(plan, kPlanGroupSize, b);
        int const owner = planField(plan, kPlanOwner, b);
        ASSERT_TRUE(e >= 0 && e < experts) << "broadcast " << b;
        EXPECT_EQ(seen[e]++, 0) << "expert " << e << " broadcast twice";
        EXPECT_EQ(owner, e / home) << "broadcast " << b;
        EXPECT_LE(++exports[owner], candidate_count(ep, experts, helpers)) << "owner " << owner;
        ASSERT_TRUE(level >= 0 && level < depth && size == (level == pairLevel ? 2 : ep >> level) && begin % size == 0
            && begin + size <= ep)
            << "broadcast " << b << " group [" << begin << ", " << begin + size << ") at level " << level;
        loadPre[owner] -= mass[e];
        loadPost[owner] -= mass[e];
        for (auto const* rows : {&pre, &post})
        {
            int64_t sum = 0;
            for (int rank = 0; rank < ep; ++rank)
            {
                int const q = (*rows)[b * ep + rank];
                bool const host = rank == owner || (rank >= begin && rank < begin + size);
                EXPECT_TRUE(q >= 0 && (host || q == 0)) << "broadcast " << b << " rank " << rank << " quota " << q;
                sum += q;
            }
            EXPECT_EQ(sum, mass[e]) << "broadcast " << b << " quota does not add up to the expert's tokens";
        }
        for (int rank = 0; rank < ep; ++rank)
        {
            loadPre[rank] += pre[b * ep + rank];
            loadPost[rank] += post[b * ep + rank];
        }
    }
    EXPECT_EQ(loadPre, loadPost) << "quota repair changed a rank's load";
    EXPECT_LE(*std::max_element(loadPre.begin(), loadPre.end()), *std::max_element(resident.begin(), resident.end()))
        << "placement raised the busiest rank's load";

    /* Coloring without a residency table: levels coarse first, then broadcasts in order, each takes the lowest slot
     * free on its whole group.  Every rank colors alike, and each rank's slots name the broadcasts that cover it. */
    std::vector<char> taken(static_cast<size_t>(ep) * helpers, 0);
    std::vector<int> expected(broadcasts, -1);
    for (int level = 0; level < depth; ++level)
        for (int b = 0; b < broadcasts; ++b)
        {
            if (planField(plan, kPlanLevel, b) != level)
                continue;
            int const begin = planField(plan, kPlanGroupBegin, b);
            int const end = begin + planField(plan, kPlanGroupSize, b);
            for (int slot = 0; slot < helpers && expected[b] < 0; ++slot)
            {
                bool open = true;
                for (int rank = begin; rank < end; ++rank)
                    open = open && !taken[rank * helpers + slot];
                expected[b] = open ? slot : -1;
            }
            ASSERT_GE(expected[b], 0) << "broadcast " << b << " finds no free slot";
            for (int rank = begin; rank < end; ++rank)
                taken[rank * helpers + expected[b]] = 1;
        }
    for (int b = 0; b < broadcasts; ++b)
        for (int rank = 0; rank < ep; ++rank)
            EXPECT_EQ(colors[rank * stride + b], expected[b]) << "broadcast " << b << " local rank " << rank;
    for (int rank = 0; rank < ep; ++rank)
        for (int slot = 0; slot < helpers; ++slot)
        {
            int id = -1, level = -1, owner = -1;
            for (int b = 0; b < broadcasts; ++b)
            {
                int const begin = planField(plan, kPlanGroupBegin, b);
                if (expected[b] == slot && rank >= begin && rank < begin + planField(plan, kPlanGroupSize, b))
                {
                    id = planField(plan, kPlanExpert, b);
                    level = planField(plan, kPlanLevel, b);
                    owner = planField(plan, kPlanOwner, b);
                }
            }
            int const* out = colors.data() + rank * stride + kMaxBroadcasts;
            EXPECT_EQ(out[slot], id) << "rank " << rank << " slot " << slot;
            EXPECT_EQ(out[helpers + slot], level) << "rank " << rank << " slot " << slot;
            EXPECT_EQ(out[2 * helpers + slot], owner) << "rank " << rank << " slot " << slot;
        }

    /* The exact placement of these inputs, from the Python reference of the planner (haloq3_ref.place_best with
     * pairs="both" and planner_targets() targets).  The repair is checked by its invariants, the coloring by the host
     * recomputation above.  An intended policy change updates the values printed below. */
    uint64_t const hash = placementHash(plan, pre, ep);
    char hex[32];
    std::snprintf(hex, sizeof(hex), "0x%016llx", static_cast<unsigned long long>(hash));
    EXPECT_EQ(broadcasts, c.broadcasts) << c.name << ": placement hash " << hex;
    EXPECT_EQ(hash, c.golden) << c.name << ": placement hash " << hex << " (" << broadcasts << " broadcasts)";
}

/* Inputs make 1 to 69 broadcasts.  EP8/16/32 at S = 2 and 3 take the compile-time placement shapes: four dense
 * plans, except at EP32 S3, where warp 0 writes its plan directly and three dense plans stand behind it.  EP2 (no pair
 * level), EP4 S4, EP8 S16 and EP12 (not a power of two) take the runtime-shape placement with dense plans, EP32 S40
 * with two compact plans.  EP <= 8 prepares in registers (homes up to 64 experts), except EP2 (a 192-expert home).
 * Up to 32 broadcasts take the scheduled pair repair and the warp coloring; EP32 S40 (69 broadcasts, 40 helpers)
 * takes the generic repair and the serial coloring. */
INSTANTIATE_TEST_SUITE_P(HaloQ, HaloQPlannerTest,
    ::testing::Values(Case{"ep2_e384_s2", 2, 384, 2, false, 1, 0x6289361b7d1c018aull},
        Case{"ep4_e96_s4_hot_half", 4, 96, 4, true, 4, 0x20ef4247f37f6409ull},
        Case{"ep8_e384_s2", 8, 384, 2, false, 4, 0x5a329ac8c069c55cull},
        Case{"ep8_e384_s3", 8, 384, 3, false, 4, 0xb782d125f316ad9eull},
        Case{"ep8_e384_s16", 8, 384, 16, false, 5, 0xf46a992ac17e401eull},
        Case{"ep12_e384_s3", 12, 384, 3, false, 7, 0xf302a409d0dcf032ull},
        Case{"ep16_e384_s2", 16, 384, 2, false, 2, 0xe6e786693a67baf7ull},
        Case{"ep16_e384_s3", 16, 384, 3, false, 7, 0x62fc8a7e4f55da36ull},
        Case{"ep32_e384_s2", 32, 384, 2, false, 12, 0x2a8aa6ecb253137cull},
        Case{"ep32_e384_s3", 32, 384, 3, false, 13, 0x0d5cf5f85749022dull},
        Case{"ep32_e96_s3", 32, 96, 3, false, 12, 0x9cbc33c7e91f9ab2ull},
        Case{"ep32_e384_s40_hot_half", 32, 384, 40, true, 69, 0xda1f5369e873e662ull}),
    [](::testing::TestParamInfo<Case> const& info) { return std::string(info.param.name); });

} // namespace tensorrt_llm::tests::kernels::haloq
