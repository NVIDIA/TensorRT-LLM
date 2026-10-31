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
 * CTA, without the cross-rank count exchange or the copy.  Each case checks the plan's invariants and pins the plan of
 * fixed inputs, so a change to the policy shows up here first. */

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

/* halo_q_scheduler_body's planning phase after the count exchange.  Coloring then runs once per local rank, each on a
 * fresh copy of the plan; colors holds per rank the helper of every broadcast and the rank's out ids/levels/owners. */
__global__ void __launch_bounds__(kThreads, 1) planKernel(int const* allCounts, int ep, int experts, int helpers,
    int capacity, int* plan, int* quotaPre, int* quotaPost, int* planCopy, int* colors, int* status)
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
    int* candidates = shared + offset;
    offset += experts;
    int* recvUsed = shared + offset;
    offset += ep * kHelperMaskWords;
    int* sendUsed = shared + offset;
    offset += ep;
    int* nextIndex = shared + offset;
    offset += ep;
    int* frozen = shared + offset;
    offset += ep;
    int* ring = shared + offset;
    offset += ep;
    int* order = shared + offset;
    offset += capacity;
    int* hostMask = shared + offset;
    offset += capacity;
    int* schedule = shared + offset;

    for (int i = threadIdx.x; i < ep * experts; i += blockDim.x)
        counts[i] = allCounts[i];
    for (int i = threadIdx.x; i < capacity * ep; i += blockDim.x)
        quota[i] = 0;
    __syncthreads();
    if (uses_register_prepare(ep, experts))
        prepare_halo_m_register(
            counts, histogram, load, candidates, recvUsed, sendUsed, nextIndex, frozen, ep, experts, helpers);
    else
        prepare_halo_m(
            counts, histogram, load, candidates, recvUsed, sendUsed, nextIndex, frozen, ep, experts, helpers);
    if (threadIdx.x < (ep <= 8 ? 8 : 32))
    {
        int broadcasts;
        if (ep <= 8 && helpers <= 15 && helpers * ep <= experts)
            broadcasts = plan_halo_m_register8(histogram, load, quota, candidates, plan, ep, helpers);
        else
            broadcasts = plan_halo_m_warp(
                histogram, load, quota, candidates, recvUsed, sendUsed, nextIndex, frozen, plan, ep, experts, helpers);
        if (threadIdx.x == 0)
            plan[1] = broadcasts;
    }
    __syncthreads();
    int const broadcasts = plan[1];
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
        quotaPre[i] = quota[i];
    __syncthreads();
    run_halo_q(counts, quota, plan, broadcasts, ep, experts, ring, order, hostMask, schedule);
    __syncthreads();
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
        quotaPost[i] = quota[i];

    int const stride = kMaxBroadcasts + 3 * helpers;
    for (int rank = 0; rank < ep; ++rank)
    {
        for (int i = threadIdx.x; i < kPlanInts; i += blockDim.x)
            planCopy[i] = plan[i];
        __syncthreads();
        int* out = colors + rank * stride;
        if (threadIdx.x >> 5 == 8)
            color_and_publish_local(planCopy, broadcasts, ep, helpers, rank, recvUsed, out + kMaxBroadcasts,
                out + kMaxBroadcasts + helpers, out + kMaxBroadcasts + 2 * helpers, status, 1);
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
    bool hotHalf; // the first half of the ranks is three times as loaded, else hot experts are spread at random
    uint64_t golden;
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

uint64_t fnv1a(uint64_t hash, std::vector<int> const& values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        for (int byte = 0; byte < 4; ++byte)
            hash = (hash ^ ((static_cast<uint32_t>(values[i]) >> (8 * byte)) & 0xffu)) * 0x100000001b3ull;
    return hash;
}

class HaloQPlannerTest : public ::testing::TestWithParam<Case>
{
};

TEST_P(HaloQPlannerTest, PlanInvariantsAndGolden)
{
    Case const& c = GetParam();
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    int const ep = c.ep;
    int const experts = c.experts;
    int const helpers = c.helpers;
    int const home = experts / ep;
    int const capacity = broadcast_capacity(ep, experts, helpers);
    int const sharedBytes = halo_q_shared_ints(ep, experts, capacity, kThreads) * static_cast<int>(sizeof(int));
    int optin = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), cudaSuccess);
    if (sharedBytes > optin)
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
    ASSERT_EQ(cudaMemset(dPlan, 0, kPlanInts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dColors, 0xff, ep * stride * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dStatus, 0, 8 * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaFuncSetAttribute(planKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes), cudaSuccess);
    planKernel<<<1, kThreads, sharedBytes>>>(
        dCounts, ep, experts, helpers, capacity, dPlan, dPre, dPost, dPlanCopy, dColors, dStatus);
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
        EXPECT_LE(++exports[owner], std::min(helpers, home)) << "owner " << owner;
        ASSERT_TRUE(
            level >= 0 && level < hierarchy_depth(ep) && size == ep >> level && begin % size == 0 && begin + size <= ep)
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

    /* Every rank colors alike; broadcasts whose groups share a rank use different slots; each rank's slots name the
     * broadcasts that cover it. */
    for (int b = 0; b < broadcasts; ++b)
    {
        int const helper = colors[b];
        EXPECT_TRUE(helper >= 0 && helper < helpers) << "broadcast " << b << " helper " << helper;
        for (int rank = 1; rank < ep; ++rank)
            EXPECT_EQ(colors[rank * stride + b], helper) << "broadcast " << b << " rank " << rank;
        for (int other = 0; other < b; ++other)
        {
            int const lo = std::max(planField(plan, kPlanGroupBegin, b), planField(plan, kPlanGroupBegin, other));
            int const hi = std::min(planField(plan, kPlanGroupBegin, b) + planField(plan, kPlanGroupSize, b),
                planField(plan, kPlanGroupBegin, other) + planField(plan, kPlanGroupSize, other));
            EXPECT_FALSE(lo < hi && colors[other] == helper) << "broadcasts " << other << " and " << b;
        }
    }
    for (int rank = 0; rank < ep; ++rank)
        for (int slot = 0; slot < helpers; ++slot)
        {
            int id = -1, level = -1, owner = -1;
            for (int b = 0; b < broadcasts; ++b)
            {
                int const begin = planField(plan, kPlanGroupBegin, b);
                if (colors[b] == slot && rank >= begin && rank < begin + planField(plan, kPlanGroupSize, b))
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

    /* The exact plan of these inputs.  An intended policy change updates the golden value printed below. */
    uint64_t hash = 0xcbf29ce484222325ull;
    hash = fnv1a(hash, plan, 2 + 6 * kMaxBroadcasts);
    hash = fnv1a(hash, pre, broadcasts * ep);
    hash = fnv1a(hash, post, broadcasts * ep);
    hash = fnv1a(hash, colors, colors.size());
    char hex[32];
    std::snprintf(hex, sizeof(hex), "0x%016llx", static_cast<unsigned long long>(hash));
    EXPECT_EQ(hash, c.golden) << c.name << ": plan hash " << hex << " (" << broadcasts << " broadcasts)";
}

/* EP <= 8 with few helpers takes the 8-lane register planner and pair repair; more helpers, larger EP and non-powers
 * of two take the 32-lane planner and the scheduled repair; fewer experts than the repair schedule; more than 32
 * broadcasts and 32 helpers take the generic repair and serial coloring. */
INSTANTIATE_TEST_SUITE_P(HaloQ, HaloQPlannerTest,
    ::testing::Values(Case{"ep4_e96_s4_hot_half", 4, 96, 4, true, 0x7db396a2e3b3435aull},
        Case{"ep8_e384_s3", 8, 384, 3, false, 0x51289f6ecdd78fe2ull},
        Case{"ep8_e384_s16", 8, 384, 16, false, 0x6849100d88b04170ull},
        Case{"ep12_e384_s3", 12, 384, 3, false, 0x3b0f445dd222dfa0ull},
        Case{"ep16_e384_s3", 16, 384, 3, false, 0x8bfb8ef3a26b54eaull},
        Case{"ep32_e384_s3", 32, 384, 3, false, 0x78c7529ce5f2f0baull},
        Case{"ep32_e96_s3", 32, 96, 3, false, 0xf59361f1320337fcull},
        Case{"ep32_e384_s40_hot_half", 32, 384, 40, true, 0x9930d2f5211d1099ull}),
    [](::testing::TestParamInfo<Case> const& info) { return std::string(info.param.name); });

} // namespace tensorrt_llm::tests::kernels::haloq
