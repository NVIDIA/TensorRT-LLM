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

// The HALO-Q rebalance plan (placement, quota repair, helper-slot coloring) on routing counts given directly to one
// CTA, without the cross-rank count exchange or the copy.  Each case checks the plan's invariants, recomputes the
// coloring on the host, and pins the placement of fixed inputs, so a change to the policy shows up here first.

#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQKernel.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <optional>
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

// halo_q_scheduler_body's planning phase after the count exchange, on its shared-memory layout: prepare,
// place_halo_q, then warp 8 colors for local rank 0 while the quota repair runs.  The test only adds a snapshot of
// the placement's quota rows before the repair, and colors again for every other local rank, each on a fresh copy of
// the plan.  colors holds per rank the helper of every broadcast and the rank's out ids/levels/owners.
__global__ void __launch_bounds__(kThreads, 1) planKernel(int const* allCounts, int ep, int experts, int helpers,
    int capacity, int sharedInts, int* plan, int* quotaPre, int* quotaPost, int* planCopy, int* colors, int* status)
{
    extern __shared__ int shared[];
    int offset = ep * experts;
    int* const counts = shared;
    int* const histogram = shared + offset;
    offset += experts;
    int* const load = shared + offset;
    offset += ep;
    int* const quota = shared + offset;
    offset += capacity * ep;
    int* const selectedFlags = shared + offset;
    offset += experts;
    int* const slotMasks = shared + offset;
    offset += ep * kHelperMaskWords;
    int* const ring = shared + offset;
    offset += ep;
    int* const order = shared + offset;
    offset += capacity;
    int* const hostMask = shared + offset;
    offset += capacity;
    int* const schedule = shared + offset;
    offset += kScheduleInts;
    int* const running = shared + offset;

    // The planner CTA's shared memory holds whatever ran before it, so every word starts poisoned; exchange_counts
    // then fills the counts and ends at a block barrier.
    for (int i = threadIdx.x; i < sharedInts; i += blockDim.x)
    {
        shared[i] = -1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < ep * experts; i += blockDim.x)
    {
        counts[i] = allCounts[i];
    }
    __syncthreads();
    if (uses_register_prepare(ep, experts))
    {
        prepare_halo_m_register(counts, histogram, load, selectedFlags, ep, experts, helpers);
    }
    else
    {
        prepare_halo_m(counts, histogram, load, selectedFlags, ep, experts, helpers);
    }
    int const broadcasts = place_halo_q(histogram, load, quota, selectedFlags, running, plan, ep, experts, helpers);
    // place_halo_q ends at a block barrier.
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
    {
        quotaPre[i] = quota[i];
    }
    __syncthreads();
    if (threadIdx.x >> 5 == 8)
    {
        color_and_publish_local(plan, broadcasts, ep, helpers, 0, slotMasks, colors + kMaxBroadcasts,
            colors + kMaxBroadcasts + helpers, colors + kMaxBroadcasts + 2 * helpers, status, 1, nullptr);
    }
    run_halo_q(counts, quota, plan, broadcasts, ep, experts, ring, order, hostMask, schedule);
    __syncthreads();
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
    {
        quotaPost[i] = quota[i];
    }
    for (int i = threadIdx.x; i < broadcasts; i += blockDim.x)
    {
        colors[i] = plan[2 + kPlanHelper * kMaxBroadcasts + i];
    }
    __syncthreads();

    int const stride = kMaxBroadcasts + 3 * helpers;
    for (int rank = 1; rank < ep; ++rank)
    {
        for (int i = threadIdx.x; i < kPlanInts; i += blockDim.x)
        {
            planCopy[i] = plan[i];
        }
        __syncthreads();
        int* const out = colors + rank * stride;
        if (threadIdx.x >> 5 == 8)
        {
            color_and_publish_local(planCopy, broadcasts, ep, helpers, rank, slotMasks, out + kMaxBroadcasts,
                out + kMaxBroadcasts + helpers, out + kMaxBroadcasts + 2 * helpers, status, 1, nullptr);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < broadcasts; i += blockDim.x)
        {
            out[i] = planCopy[2 + kPlanHelper * kMaxBroadcasts + i];
        }
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
    std::vector<int> (*makeInput)() = nullptr;
    std::optional<int> expectedPeak = std::nullopt;
};

// 8192 tokens x top-8 per source rank, with +-25% noise per source rank.  An expert's share follows 1 / (1 + j) for a
// seeded permutation j, or with hotHalf is three units on the first half of the owners and one on the rest.
std::vector<int> makeCounts(Case const& c)
{
    uint32_t state = static_cast<uint32_t>(c.ep * 1000003 + c.experts * 101 + c.helpers);
    auto next = [&state]
    {
        state = state * 1664525U + 1013904223U;
        return state >> 8;
    };
    int const home = c.experts / c.ep;
    std::vector<int> perm(c.experts);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = c.experts - 1; i > 0; --i)
    {
        std::swap(perm[i], perm[next() % (i + 1)]);
    }
    std::vector<int64_t> weight(c.experts);
    int64_t total = 0;
    for (int e = 0; e < c.experts; ++e)
    {
        weight[e] = c.hotHalf ? (e / home < c.ep / 2 ? 3 : 1) * (int64_t{1} << 20) : (int64_t{1} << 20) / (1 + perm[e]);
        total += weight[e];
    }
    std::vector<int> counts(c.ep * c.experts);
    for (int rank = 0; rank < c.ep; ++rank)
    {
        for (int e = 0; e < c.experts; ++e)
        {
            counts[rank * c.experts + e]
                = static_cast<int>(int64_t{8192 * 8} * weight[e] / total * (75 + next() % 51) / 100);
        }
    }
    return counts;
}

std::vector<int> makeStandardL7Counts()
{
    constexpr int kEp = 16;
    constexpr int kExperts = 384;
    // Global counts from an EP16 layer with 8,192 input tokens per rank and top-k = 6.
    constexpr std::array<int, kExperts> kHistogram{129, 176, 12, 0, 38'840, 3, 0, 5, 1, 0, 0, 0, 0, 0, 173, 0, 2, 0, 2,
        0, 134, 12, 126'511, 1, 0, 0, 2, 938, 101, 0, 63, 0, 0, 2, 0, 33, 22'321, 10, 0, 1191, 0, 686, 0, 0, 0, 0, 14,
        187, 2, 0, 6, 0, 12, 0, 0, 20, 0, 1, 0, 0, 2, 6, 0, 0, 0, 7, 320, 0, 6, 0, 2, 0, 35, 0, 74, 10'655, 0, 0, 0, 70,
        0, 1, 0, 15, 0, 13'620, 0, 0, 1, 0, 0, 6, 0, 0, 0, 11, 0, 3, 4, 0, 38, 0, 60'229, 129'711, 0, 10, 42, 0, 368, 0,
        0, 0, 0, 34, 118, 0, 0, 0, 0, 477, 0, 0, 14, 0, 0, 0, 12, 0, 6, 0, 3112, 1, 0, 2, 0, 0, 0, 37'335, 0, 0, 166, 0,
        1, 3, 0, 0, 0, 9, 103, 9, 0, 90, 7, 1, 0, 281, 91, 179, 42, 0, 3, 2, 1, 3, 0, 96, 19, 295, 0, 0, 2, 87, 0, 0,
        171, 0, 1, 0, 2, 0, 4, 115, 0, 3, 7, 0, 8, 177, 3187, 4, 0, 27, 0, 143, 1, 0, 18, 0, 17, 1, 0, 0, 0, 0, 0, 0,
        72'895, 0, 0, 0, 0, 13, 1, 0, 0, 0, 0, 0, 16, 15'621, 0, 4, 3181, 0, 1, 60, 2, 0, 0, 20, 5, 0, 98, 1, 58, 0, 0,
        2, 3, 0, 0, 0, 4377, 0, 40, 0, 0, 5701, 0, 0, 1, 0, 0, 127, 1, 122'121, 10, 95'595, 147, 0, 3, 0, 0, 1, 1, 55,
        0, 37, 931, 4199, 0, 34, 21, 7, 0, 0, 4, 0, 1559, 0, 1969, 0, 0, 33, 56, 0, 0, 0, 0, 0, 52, 2, 0, 0, 1, 0, 0, 0,
        530, 0, 0, 0, 0, 91, 356, 3, 0, 293, 1, 13, 18, 44, 8, 717, 189, 0, 9, 3, 0, 0, 80, 54, 0, 89, 7, 0, 0, 0, 17,
        8, 12, 0, 138, 0, 163, 0, 0, 0, 2, 1, 16, 0, 0, 0, 0, 0, 3, 0, 0, 4, 1, 0, 29, 3, 0, 0, 102, 0, 1, 0, 40, 0, 0,
        116, 0, 0, 3, 5, 0, 0, 1, 20, 0, 0, 35, 0, 0, 3, 10, 12, 0, 0, 8, 795};
    std::vector<int> counts(kEp * kExperts);
    int firstExtraRank = 0;
    // Rotate the remainder across the ranks to preserve both the expert histogram and per-source volume.
    for (int expert = 0; expert < kExperts; ++expert)
    {
        int const base = kHistogram[expert] / kEp;
        int const remainder = kHistogram[expert] % kEp;
        for (int rank = 0; rank < kEp; ++rank)
        {
            counts[((firstExtraRank + rank) % kEp) * kExperts + expert] = base + (rank < remainder);
        }
        firstExtraRank = (firstExtraRank + remainder) % kEp;
    }
    return counts;
}

uint64_t fnv1a(uint64_t hash, int const* values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        for (int byte = 0; byte < 4; ++byte)
        {
            hash = (hash ^ ((static_cast<uint32_t>(values[i]) >> (8 * byte)) & 0xffU)) * 0x100000001b3ULL;
        }
    }
    return hash;
}

// FNV-1a over the broadcast count, each broadcast's expert, level, group begin, group size and owner, then the
// placement's quota rows: the plan before repair and coloring.
uint64_t placementHash(std::vector<int> const& plan, std::vector<int> const& pre, int ep)
{
    int const broadcasts = plan[1];
    uint64_t hash = fnv1a(0xcbf29ce484222325ULL, &broadcasts, 1);
    for (int b = 0; b < broadcasts; ++b)
    {
        for (int field : {kPlanExpert, kPlanLevel, kPlanGroupBegin, kPlanGroupSize, kPlanOwner})
        {
            int const value = planField(plan, field, b);
            hash = fnv1a(hash, &value, 1);
        }
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
    {
        GTEST_SKIP() << "HALO-Q runs on SM10x (Blackwell and Rubin) only";
    }
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
    {
        GTEST_SKIP() << "needs " << sharedBytes << " B of shared memory per block";
    }

    std::vector<int> const counts = c.makeInput ? c.makeInput() : makeCounts(c);
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
    {
        cudaFree(p);
    }

    int const broadcasts = plan[1];
    ASSERT_GT(broadcasts, 0);
    ASSERT_LE(broadcasts, capacity);
    for (int word : status)
    {
        EXPECT_EQ(word, 0) << "helper coloring failed";
    }

    // Levels: the multicast ladder ep >> level, then, for EP >= 4, aligned rank pairs.  A rank sends at most
    // candidate_count() = min(S, home, 32) experts; a multicast group may include its owner or not.
    int const depth = planner_depth(ep);
    int const pairLevel = pair_level_of(ep);
    std::vector<int64_t> mass(experts, 0), resident(ep, 0), loadPre(ep, 0), loadPost(ep, 0);
    for (int rank = 0; rank < ep; ++rank)
    {
        for (int e = 0; e < experts; ++e)
        {
            mass[e] += counts[rank * experts + e];
        }
    }
    for (int e = 0; e < experts; ++e)
    {
        resident[e / home] += mass[e];
    }
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
    if (c.expectedPeak)
    {
        EXPECT_EQ(*std::max_element(loadPost.begin(), loadPost.end()), *c.expectedPeak);
    }

    // Coloring without a residency table: levels coarse first, then broadcasts in order, each takes the lowest slot
    // free on its whole group.  Every rank colors alike, and each rank's slots name the broadcasts that cover it.
    std::vector<char> taken(static_cast<size_t>(ep) * helpers, 0);
    std::vector<int> expected(broadcasts, -1);
    for (int level = 0; level < depth; ++level)
    {
        for (int b = 0; b < broadcasts; ++b)
        {
            if (planField(plan, kPlanLevel, b) != level)
            {
                continue;
            }
            int const begin = planField(plan, kPlanGroupBegin, b);
            int const end = begin + planField(plan, kPlanGroupSize, b);
            for (int slot = 0; slot < helpers && expected[b] < 0; ++slot)
            {
                bool open = true;
                for (int rank = begin; rank < end; ++rank)
                {
                    open = open && !taken[rank * helpers + slot];
                }
                expected[b] = open ? slot : -1;
            }
            ASSERT_GE(expected[b], 0) << "broadcast " << b << " finds no free slot";
            for (int rank = begin; rank < end; ++rank)
            {
                taken[rank * helpers + expected[b]] = 1;
            }
        }
    }
    for (int b = 0; b < broadcasts; ++b)
    {
        for (int rank = 0; rank < ep; ++rank)
        {
            EXPECT_EQ(colors[rank * stride + b], expected[b]) << "broadcast " << b << " local rank " << rank;
        }
    }
    for (int rank = 0; rank < ep; ++rank)
    {
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
    }

    // The exact placement of these inputs, from a reference model of the planner with planner_targets() targets.  The
    // repair is checked by its invariants, the coloring by the host recomputation above.  An intended policy change
    // updates the values printed below.
    uint64_t const hash = placementHash(plan, pre, ep);
    char hex[32];
    std::snprintf(hex, sizeof(hex), "0x%016llx", static_cast<unsigned long long>(hash));
    EXPECT_EQ(broadcasts, c.broadcasts) << c.name << ": placement hash " << hex;
    EXPECT_EQ(hash, c.golden) << c.name << ": placement hash " << hex << " (" << broadcasts << " broadcasts)";
}

// Inputs make 1 to 69 broadcasts. EP8/16/32 at S = 2 and 3 take the compile-time placement shapes. EP32 S2 uses
// four dense plans, EP32 S3 writes one plan directly and three in scratch, and the smaller cases fit five dense
// plans. EP2 (no pair level), EP4 S4, EP8 S16 and EP12 (not a power of two) take runtime-shape placement with dense
// plans. EP32 S40 uses two compact plans. EP <= 8 prepares in registers (homes up to 64 experts), except EP2
// (a 192-expert home). Up to 32 broadcasts take the scheduled pair repair and warp coloring. EP32 S40 takes
// the generic repair and serial coloring.
INSTANTIATE_TEST_SUITE_P(HaloQ, HaloQPlannerTest,
    ::testing::Values(Case{"ep2_e384_s2", 2, 384, 2, false, 1, 0x6289361b7d1c018aULL},
        Case{"ep4_e96_s4_hot_half", 4, 96, 4, true, 4, 0x20ef4247f37f6409ULL},
        Case{"ep8_e384_s2", 8, 384, 2, false, 4, 0x5a329ac8c069c55cULL},
        Case{"ep8_e384_s3", 8, 384, 3, false, 4, 0xb782d125f316ad9eULL},
        Case{"ep8_e384_s16", 8, 384, 16, false, 5, 0xf46a992ac17e401eULL},
        Case{"ep12_e384_s3", 12, 384, 3, false, 7, 0xf302a409d0dcf032ULL},
        Case{"ep16_e384_s2", 16, 384, 2, false, 2, 0xe6e786693a67baf7ULL},
        Case{"ep16_e384_s2_standard_l7", 16, 384, 2, false, 7, 0xd6e0d5ed4a2d490cULL, makeStandardL7Counts, 50'627},
        Case{"ep16_e384_s3", 16, 384, 3, false, 7, 0x62fc8a7e4f55da36ULL},
        Case{"ep32_e384_s2", 32, 384, 2, false, 12, 0x2a8aa6ecb253137cULL},
        Case{"ep32_e384_s3", 32, 384, 3, false, 13, 0x0d5cf5f85749022dULL},
        Case{"ep32_e96_s3", 32, 96, 3, false, 12, 0x9cbc33c7e91f9ab2ULL},
        Case{"ep32_e384_s40_hot_half", 32, 384, 40, true, 69, 0xda1f5369e873e662ULL}),
    [](::testing::TestParamInfo<Case> const& info) { return std::string(info.param.name); });

} // namespace tensorrt_llm::tests::kernels::haloq
