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
// Two more tests compare the planner's parallel implementations with each other: the placements that take the same
// input, and the three encodings of the repair's pair tournament.

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

int planField(std::vector<int> const& plan, int field, int b)
{
    return plan[plan_field_offset(field) + b];
}

// halo_q_scheduler_body's planning phase after the count exchange, on its shared-memory layout: prepare,
// place_halo_q, then warp kColoringWarp colors for local rank 0 while the quota repair runs.  The test only adds a
// snapshot of the placement's quota rows before the repair, and colors again for every other local rank, each on a
// fresh copy of the plan.  colors holds per rank the helper of every broadcast and the rank's out ids/levels/owners.
__global__ void __launch_bounds__(kHaloQThreads, 1) planKernel(int const* allCounts, int ep, int experts, int helpers,
    int capacity, int sharedInts, int* plan, int* quotaPre, int* quotaPost, int* planCopy, int* colors, int* status)
{
    extern __shared__ int shared[];
    PlannerShared const mem = planner_shared(shared, ep, experts, capacity);

    // The planner CTA's shared memory holds whatever ran before it, so every word starts poisoned; exchange_counts
    // then fills the counts and ends at a block barrier.
    for (int i = threadIdx.x; i < sharedInts; i += blockDim.x)
    {
        shared[i] = -1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < ep * experts; i += blockDim.x)
    {
        mem.counts[i] = allCounts[i];
    }
    __syncthreads();
    if (uses_register_prepare(ep, experts))
    {
        prepare_halo_m_register(mem.counts, mem.histogram, mem.load, mem.selected_flags, ep, experts, helpers);
    }
    else
    {
        prepare_halo_m(mem.counts, mem.histogram, mem.load, mem.selected_flags, ep, experts, helpers);
    }
    int const broadcasts
        = place_halo_q(mem.histogram, mem.load, mem.quota, mem.selected_flags, mem.running, plan, ep, experts, helpers);
    // place_halo_q ends at a block barrier.
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
    {
        quotaPre[i] = mem.quota[i];
    }
    __syncthreads();
    if (threadIdx.x >> 5 == kColoringWarp)
    {
        color_and_publish_local(plan, broadcasts, ep, helpers, 0, mem.slot_masks, colors + kMaxBroadcasts,
            colors + kMaxBroadcasts + helpers, colors + kMaxBroadcasts + 2 * helpers, status, 1, nullptr);
    }
    run_halo_q(mem.counts, mem.quota, plan, broadcasts, ep, experts, mem.ring, mem.order, mem.host_mask, mem.schedule);
    __syncthreads();
    for (int i = threadIdx.x; i < broadcasts * ep; i += blockDim.x)
    {
        quotaPost[i] = mem.quota[i];
    }
    for (int i = threadIdx.x; i < broadcasts; i += blockDim.x)
    {
        colors[i] = plan_field(plan, kPlanHelper)[i];
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
        if (threadIdx.x >> 5 == kColoringWarp)
        {
            color_and_publish_local(planCopy, broadcasts, ep, helpers, rank, mem.slot_masks, out + kMaxBroadcasts,
                out + kMaxBroadcasts + helpers, out + kMaxBroadcasts + 2 * helpers, status, 1, nullptr);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < broadcasts; i += blockDim.x)
        {
            out[i] = plan_field(planCopy, kPlanHelper)[i];
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
    bool hotHalf;     // the first half of the ranks is three times as loaded, else hot experts are spread at random
    int broadcasts;   // of the placement
    uint64_t golden;  // placementHash() of the placement
    bool repairMoves; // the placement leaves the quota repair a move, see repairHasMove()
    std::vector<int> (*makeInput)() = nullptr;
    std::optional<int> expectedPeak = std::nullopt;
    int scale = 1; // every count times this; above 1 the total passes kNarrowKeyMaxTotal
};

// 8192 tokens x top-8 per source rank, with +-25% noise per source rank, times the case's scale.  An expert's share
// follows 1 / (1 + j) for a seeded permutation j, or with hotHalf is three units on the first half of the owners and
// one on the rest.
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
                = c.scale * static_cast<int>(int64_t{8192 * 8} * weight[e] / total * (75 + next() % 51) / 100);
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

// Whether `rank` holds broadcast b's expert: its owner, or a member of its group.
bool hostsBroadcast(std::vector<int> const& plan, int b, int rank)
{
    int const begin = planField(plan, kPlanGroupBegin, b);
    return rank == planField(plan, kPlanOwner, b)
        || (rank >= begin && rank < begin + planField(plan, kPlanGroupSize, b));
}

// The tokens that stay on the rank that routed them: over broadcasts b and ranks r, min(quota[b][r], counts[r][e_b]).
int64_t locality(
    std::vector<int> const& plan, std::vector<int> const& quota, std::vector<int> const& counts, int ep, int experts)
{
    int64_t local = 0;
    for (int b = 0; b < plan[1]; ++b)
    {
        int const e = planField(plan, kPlanExpert, b);
        for (int rank = 0; rank < ep; ++rank)
        {
            local += std::min(quota[b * ep + rank], counts[rank * experts + e]);
        }
    }
    return local;
}

// Whether the quota repair has a move on these quota rows: a receiver and a donor that both hold two broadcasts, one
// where the receiver's quota is below its own tokens and the donor's above, and one where the receiver's is above.  A
// move brings tokens of the receiver home and takes none of the donor's away, and nothing else changes a quota, so
// the repair raises the locality exactly when the placement leaves it a move.
bool repairHasMove(
    std::vector<int> const& plan, std::vector<int> const& quota, std::vector<int> const& counts, int ep, int experts)
{
    for (int receiver = 0; receiver < ep; ++receiver)
    {
        for (int donor = 0; donor < ep; ++donor)
        {
            bool pull = false;
            bool give = false;
            for (int b = 0; b < plan[1] && receiver != donor; ++b)
            {
                if (!hostsBroadcast(plan, b, receiver) || !hostsBroadcast(plan, b, donor))
                {
                    continue;
                }
                int const e = planField(plan, kPlanExpert, b);
                int const receiverShort = counts[receiver * experts + e] - quota[b * ep + receiver];
                int const donorOver = quota[b * ep + donor] - counts[donor * experts + e];
                pull = pull || (receiverShort > 0 && donorOver > 0);
                give = give || receiverShort < 0;
            }
            if (pull && give)
            {
                return true;
            }
        }
    }
    return false;
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
    int const sharedInts = halo_q_shared_ints(ep, experts, helpers, kHaloQThreads);
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
    planKernel<<<1, kHaloQThreads, sharedBytes>>>(
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
        ASSERT_TRUE(owner >= 0 && owner < ep) << "broadcast " << b;
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
    // Placement plans a total above kNarrowKeyMaxTotal with wide keys.
    int64_t const total = std::accumulate(mass.begin(), mass.end(), int64_t{0});
    EXPECT_EQ(total > kNarrowKeyMaxTotal, c.scale > 1) << "total " << total;

    // The repair only moves quota toward the rank that routed the tokens, so it never lowers the locality, and it
    // raises the locality exactly when the placement leaves it a move.
    int64_t const localityPre = locality(plan, pre, counts, ep, experts);
    int64_t const localityPost = locality(plan, post, counts, ep, experts);
    bool const hasMove = repairHasMove(plan, pre, counts, ep, experts);
    EXPECT_GE(localityPost, localityPre) << "quota repair lowered the locality";
    EXPECT_EQ(localityPost > localityPre, hasMove)
        << "locality " << localityPre << " before and " << localityPost << " after the repair";
    EXPECT_EQ(hasMove, c.repairMoves) << "locality " << localityPre << " before and " << localityPost << " after";

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
// plans. EP2 (no pair level), EP4, EP8 S6 and S16, EP12 and EP24 (not powers of two; EP24 has groups of 24, 12, 6
// and pairs) take place_halo_q's runtime-shape path with dense plans. EP32 S40 uses two compact plans. More than
// kHeldCandidates candidates per owner (EP8 S6 and S16, EP32 S40) plan with plan_halo_q_impl and narrow keys, a
// total above kNarrowKeyMaxTotal (EP12 at 128 times the counts) with plan_halo_q_impl and wide keys, and the other
// cases with plan_halo_q_group_lanes, which on the runtime-shape path still has the shape at compile time for EP2
// S2 and EP4 S3. EP <= 8 prepares in registers (homes up to 64 experts), except EP2 and EP4 with 384 experts (192-
// and 96-expert homes, a warp per owner). Up to 32 broadcasts take the scheduled pair repair and warp coloring; EP8
// S6 hot-half runs it for EP <= 8 with 12 broadcasts. EP32 S40 takes the generic repair and serial coloring.
std::vector<Case> allCases()
{
    return {Case{"ep2_e384_s2", 2, 384, 2, false, 1, 0x6289361b7d1c018aULL, false},
        Case{"ep4_e96_s4_hot_half", 4, 96, 4, true, 4, 0x20ef4247f37f6409ULL, true},
        Case{"ep4_e384_s3", 4, 384, 3, false, 3, 0xe4d307b5233330bdULL, true},
        Case{"ep8_e384_s2", 8, 384, 2, false, 4, 0x5a329ac8c069c55cULL, true},
        Case{"ep8_e384_s3", 8, 384, 3, false, 4, 0xb782d125f316ad9eULL, true},
        Case{"ep8_e384_s6_hot_half", 8, 384, 6, true, 12, 0x6f8dbb4be2edd1b4ULL, true},
        Case{"ep8_e384_s16", 8, 384, 16, false, 5, 0xf46a992ac17e401eULL, true},
        Case{"ep12_e384_s3", 12, 384, 3, false, 7, 0xf302a409d0dcf032ULL, true},
        Case{"ep12_e384_s3_wide", 12, 384, 3, false, 7, 0x2e91f9e962ab3559ULL, true, nullptr, std::nullopt, 128},
        Case{"ep16_e384_s2", 16, 384, 2, false, 2, 0xe6e786693a67baf7ULL, false},
        Case{"ep16_e384_s2_standard_l7", 16, 384, 2, false, 7, 0xd6e0d5ed4a2d490cULL, true, makeStandardL7Counts,
            50'627},
        Case{"ep16_e384_s3", 16, 384, 3, false, 7, 0x62fc8a7e4f55da36ULL, true},
        Case{"ep24_e384_s3", 24, 384, 3, false, 12, 0xa7cdfafb9397106dULL, true},
        Case{"ep32_e384_s2", 32, 384, 2, false, 12, 0x2a8aa6ecb253137cULL, true},
        Case{"ep32_e384_s3", 32, 384, 3, false, 13, 0x0d5cf5f85749022dULL, true},
        Case{"ep32_e96_s3", 32, 96, 3, false, 12, 0x9cbc33c7e91f9ab2ULL, true},
        Case{"ep32_e384_s40_hot_half", 32, 384, 40, true, 69, 0xda1f5369e873e662ULL, true}};
}

INSTANTIATE_TEST_SUITE_P(HaloQ, HaloQPlannerTest, ::testing::ValuesIn(allCases()),
    [](::testing::TestParamInfo<Case> const& info) { return std::string(info.param.name); });

// HALO-Q runs on SM10x (Blackwell and Rubin) only.
bool deviceRunsHaloQ()
{
    int device = 0;
    int major = 0;
    return cudaGetDevice(&device) == cudaSuccess
        && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess && major == 10;
}

// The placements of one target: plan_halo_q_impl with wide and with narrow keys, and plan_halo_q_group_lanes with
// its shape at run time and at compile time.
enum Placement
{
    kWideKeys = 0,
    kNarrowKeys = 1,
    kGroupLanes = 2,
    kGroupLanesShape = 3,
    kPlacements = 4,
};

// One warp's placement toward `target` by every implementation that takes the input, as plan_halo_q calls them,
// each into its own dense plan of planner_dense_ints() at out and its (broadcasts, highest rank load) at results.
// plan_halo_q_impl takes every input, with wide keys at any total.  plan_halo_q_group_lanes takes at most
// kHeldCandidates candidates per owner, and has the shape at compile time for a power-of-two EP at S = 2 or 3.
template <int Depth>
__device__ void placeEveryWay(int const* histogram, int const* load, int const* candidates, int target, int ep,
    int experts, int helpers, int* out, int* results)
{
    int const lane = threadIdx.x & 31;
    int const count = candidate_count(ep, experts, helpers);
    int const denseInts = planner_dense_ints(ep, experts, helpers);
    int const laneLoad = lane < ep ? load[lane] : 0;
    int heldExpert[kHeldCandidates];
    int heldMass[kHeldCandidates];
    for (int k = 0; k < kHeldCandidates; ++k)
    {
        heldExpert[k] = lane < ep && k < count ? candidates[lane * count + k] : 0;
        heldMass[k] = lane < ep && k < count ? histogram[heldExpert[k]] : 0;
    }
    auto const record = [&](int placement, int broadcasts, int maxLoad)
    {
        if (lane == 0)
        {
            results[2 * placement] = broadcasts;
            results[2 * placement + 1] = maxLoad;
        }
    };
    int maxLoad = 0;
    int broadcasts = plan_halo_q_impl<false, Depth>(
        histogram, laneLoad, target, candidates, out + kWideKeys * denseInts, ep, experts, helpers, true, &maxLoad);
    record(kWideKeys, broadcasts, maxLoad);
    broadcasts = plan_halo_q_impl<true, Depth>(
        histogram, laneLoad, target, candidates, out + kNarrowKeys * denseInts, ep, experts, helpers, true, &maxLoad);
    record(kNarrowKeys, broadcasts, maxLoad);
    if (count > kHeldCandidates)
    {
        return;
    }
    broadcasts = plan_halo_q_group_lanes<Depth>(histogram, laneLoad, target, candidates, out + kGroupLanes * denseInts,
        ep, experts, helpers, true, &maxLoad, nullptr, nullptr, heldExpert, heldMass);
    record(kGroupLanes, broadcasts, maxLoad);
    if (ep != (1 << Depth) || experts < ep * helpers)
    {
        return;
    }
    if (helpers == 2)
    {
        broadcasts = plan_halo_q_group_lanes<Depth, (1 << Depth), 2>(histogram, laneLoad, target, candidates,
            out + kGroupLanesShape * denseInts, ep, experts, helpers, true, &maxLoad, nullptr, nullptr, heldExpert,
            heldMass);
        record(kGroupLanesShape, broadcasts, maxLoad);
    }
    else if (helpers == 3)
    {
        broadcasts = plan_halo_q_group_lanes<Depth, (1 << Depth), 3>(histogram, laneLoad, target, candidates,
            out + kGroupLanesShape * denseInts, ep, experts, helpers, true, &maxLoad, nullptr, nullptr, heldExpert,
            heldMass);
        record(kGroupLanesShape, broadcasts, maxLoad);
    }
}

// Warp w places toward targets[w]; its plans start at out + w * kPlacements * planner_dense_ints().
__global__ void placementKernel(int const* histogram, int const* load, int const* candidates, int const* targets,
    int ep, int experts, int helpers, int* out, int* results)
{
    int const warp = threadIdx.x >> 5;
    int const target = targets[warp];
    int* const warpOut = out + warp * kPlacements * planner_dense_ints(ep, experts, helpers);
    int* const warpResults = results + warp * kPlacements * 2;
    switch (planner_depth(ep))
    {
    case 1: placeEveryWay<1>(histogram, load, candidates, target, ep, experts, helpers, warpOut, warpResults); break;
    case 2: placeEveryWay<2>(histogram, load, candidates, target, ep, experts, helpers, warpOut, warpResults); break;
    case 3: placeEveryWay<3>(histogram, load, candidates, target, ep, experts, helpers, warpOut, warpResults); break;
    case 4: placeEveryWay<4>(histogram, load, candidates, target, ep, experts, helpers, warpOut, warpResults); break;
    default:
        placeEveryWay<kMaxLevels>(histogram, load, candidates, target, ep, experts, helpers, warpOut, warpResults);
        break;
    }
}

class HaloQPlacementTest : public ::testing::TestWithParam<Case>
{
};

// plan_halo_q runs one placement per input: plan_halo_q_group_lanes for a total up to kNarrowKeyMaxTotal with at
// most kHeldCandidates candidates per owner, else plan_halo_q_impl, with narrow keys up to that total.  Every input
// of the first kind is also a legal input of the others, and they follow one policy, so their plans must be equal.
TEST_P(HaloQPlacementTest, ImplementationsAgree)
{
    Case const& c = GetParam();
    if (!deviceRunsHaloQ())
    {
        GTEST_SKIP() << "HALO-Q runs on SM10x (Blackwell and Rubin) only";
    }
    int const ep = c.ep;
    int const experts = c.experts;
    int const helpers = c.helpers;
    int const home = experts / ep;
    int const count = candidate_count(ep, experts, helpers);
    int const capacity = broadcast_capacity(ep, experts, helpers);
    int const denseInts = planner_dense_ints(ep, experts, helpers);
    int const targetCount = planner_targets(ep, experts, helpers);

    // What prepare_halo_m leaves the placement: expert masses, rank loads and each owner's heaviest experts, the
    // lower expert first on a tie.  Then plan_halo_q's targets.
    std::vector<int> const counts = c.makeInput ? c.makeInput() : makeCounts(c);
    std::vector<int> histogram(experts, 0), load(ep, 0), candidates(ep * count), targets(targetCount);
    for (int rank = 0; rank < ep; ++rank)
    {
        for (int e = 0; e < experts; ++e)
        {
            histogram[e] += counts[rank * experts + e];
        }
    }
    int64_t total = 0;
    int64_t floorLoad = 0;
    for (int owner = 0; owner < ep; ++owner)
    {
        std::vector<int> order(home);
        std::iota(order.begin(), order.end(), owner * home);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return histogram[a] > histogram[b]; });
        load[owner] = std::accumulate(histogram.begin() + owner * home, histogram.begin() + (owner + 1) * home, 0);
        int64_t kept = load[owner];
        for (int k = 0; k < count; ++k)
        {
            candidates[owner * count + k] = order[k];
            kept -= histogram[order[k]];
        }
        total += load[owner];
        floorLoad = std::max(floorLoad, kept);
    }
    ASSERT_LE(total, kNarrowKeyMaxTotal);
    floorLoad = std::max(floorLoad, (total + ep - 1) / ep);
    for (int warp = 0; warp < targetCount; ++warp)
    {
        int64_t const share = (total * (400 + target_margin_400(warp)) + 400 * ep - 1) / (400 * ep);
        targets[warp] = static_cast<int>(std::max(floorLoad, share));
    }

    size_t const outInts = static_cast<size_t>(targetCount) * kPlacements * denseInts;
    size_t const resultInts = static_cast<size_t>(targetCount) * kPlacements * 2;
    int *dHistogram, *dLoad, *dCandidates, *dTargets, *dOut, *dResults;
    ASSERT_EQ(cudaMalloc(&dHistogram, histogram.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dLoad, load.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dCandidates, candidates.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dTargets, targets.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOut, outInts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dResults, resultInts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(dHistogram, histogram.data(), histogram.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dLoad, load.data(), load.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dCandidates, candidates.data(), candidates.size() * sizeof(int), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dTargets, targets.data(), targets.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(dOut, 0xff, outInts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dResults, 0xff, resultInts * sizeof(int)), cudaSuccess);
    placementKernel<<<1, 32 * targetCount>>>(
        dHistogram, dLoad, dCandidates, dTargets, ep, experts, helpers, dOut, dResults);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<int> out(outInts), results(resultInts);
    ASSERT_EQ(cudaMemcpy(out.data(), dOut, out.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(results.data(), dResults, results.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int* p : {dHistogram, dLoad, dCandidates, dTargets, dOut, dResults})
    {
        cudaFree(p);
    }

    bool const shape
        = count <= kHeldCandidates && !(ep & (ep - 1)) && experts >= ep * helpers && (helpers == 2 || helpers == 3);
    char const* const names[kPlacements] = {"plan_halo_q_impl with wide keys", "plan_halo_q_impl with narrow keys",
        "plan_halo_q_group_lanes", "plan_halo_q_group_lanes with a compile-time shape"};
    for (int warp = 0; warp < targetCount; ++warp)
    {
        int const* const warpResults = results.data() + warp * kPlacements * 2;
        int const* const reference = out.data() + static_cast<size_t>(warp) * kPlacements * denseInts;
        int const broadcasts = warpResults[2 * kWideKeys];
        ASSERT_TRUE(broadcasts >= 0 && broadcasts <= capacity) << "target " << warp;
        for (int placement = kNarrowKeys; placement < kPlacements; ++placement)
        {
            bool const runs = placement == kNarrowKeys || (placement == kGroupLanes ? count <= kHeldCandidates : shape);
            ASSERT_EQ(warpResults[2 * placement] >= 0, runs) << names[placement] << ", target " << warp;
            if (!runs)
            {
                continue;
            }
            int const* const plan = reference + placement * denseInts;
            EXPECT_EQ(warpResults[2 * placement], broadcasts) << names[placement] << ", target " << warp;
            EXPECT_EQ(warpResults[2 * placement + 1], warpResults[2 * kWideKeys + 1])
                << names[placement] << ", target " << warp << ": highest rank load";
            for (int b = 0; b < std::min(broadcasts, warpResults[2 * placement]); ++b)
            {
                // The five plan fields of the broadcast, then its quota row.
                bool same = std::equal(plan + 5 * capacity + b * ep, plan + 5 * capacity + (b + 1) * ep,
                    reference + 5 * capacity + b * ep);
                for (int field = 0; field < 5; ++field)
                {
                    same = same && plan[field * capacity + b] == reference[field * capacity + b];
                }
                EXPECT_TRUE(same) << names[placement] << ", target " << warp << ", broadcast " << b << " of "
                                  << broadcasts;
            }
        }
    }
}

std::vector<Case> narrowKeyCases()
{
    std::vector<Case> cases = allCases();
    cases.erase(std::remove_if(cases.begin(), cases.end(), [](Case const& c) { return c.scale > 1; }), cases.end());
    return cases;
}

INSTANTIATE_TEST_SUITE_P(HaloQ, HaloQPlacementTest, ::testing::ValuesIn(narrowKeyCases()),
    [](::testing::TestParamInfo<Case> const& info) { return std::string(info.param.name); });

// For every even EP: thread = lane, entry (ep / 2 - 1) * 32 * 32 + lane * 32 + i holds halo_q_partner's partner and
// `left` in phase i, and the phases in which halo_q_live_phases, with the phase count run_halo_q gives it, has the
// lane meet rank i.
__global__ void pairingKernel(int* partners, int* lefts, unsigned* livePhases)
{
    int const lane = threadIdx.x;
    for (int ep = 2; ep <= kMaxEp; ep += 2)
    {
        int const row = ((ep / 2 - 1) * 32 + lane) * 32;
        for (int phase = 0; phase < ep - 1; ++phase)
        {
            bool left = false;
            partners[row + phase] = halo_q_partner(lane, phase, ep, left);
            lefts[row + phase] = left ? 1 : 0;
        }
        for (int other = 0; other < 32; ++other)
        {
            livePhases[row + other] = ep > 16 ? halo_q_live_phases<kMaxEp - 1>(lane, ep, 1U << other)
                : ep > 8                      ? halo_q_live_phases<15>(lane, ep, 1U << other)
                                              : halo_q_live_phases<7>(lane, ep, 1U << other);
        }
    }
}

// The pair tournament of the quota repair has three encodings: the ring[] rotation of the generic repair, and for
// the scheduled repair halo_q_partner (the partner of a rank in a phase) and the recurrence in halo_q_live_phases
// (the phases in which a rank meets given partners).  The scheduler takes a rank's phases from the recurrence and
// its partner from halo_q_partner, so they must name the same pairs, and the same ones as the rotation: the two
// repairs then visit the pairs in one order.
TEST(HaloQPairingTest, EncodingsAgree)
{
    if (!deviceRunsHaloQ())
    {
        GTEST_SKIP() << "HALO-Q runs on SM10x (Blackwell and Rubin) only";
    }
    constexpr size_t kEntries = (kMaxEp / 2) * 32 * 32;
    int *dPartners, *dLefts;
    unsigned* dLivePhases;
    ASSERT_EQ(cudaMalloc(&dPartners, kEntries * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dLefts, kEntries * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dLivePhases, kEntries * sizeof(unsigned)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dPartners, 0xff, kEntries * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dLefts, 0xff, kEntries * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dLivePhases, 0xff, kEntries * sizeof(unsigned)), cudaSuccess);
    pairingKernel<<<1, 32>>>(dPartners, dLefts, dLivePhases);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<int> partners(kEntries), lefts(kEntries);
    std::vector<unsigned> livePhases(kEntries);
    ASSERT_EQ(cudaMemcpy(partners.data(), dPartners, kEntries * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(lefts.data(), dLefts, kEntries * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(livePhases.data(), dLivePhases, kEntries * sizeof(unsigned), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dPartners);
    cudaFree(dLefts);
    cudaFree(dLivePhases);

    for (int ep = 2; ep <= kMaxEp; ep += 2)
    {
        auto const row = [ep](int lane) { return static_cast<size_t>((ep / 2 - 1) * 32 + lane) * 32; };
        // run_halo_q's generic repair: ring[] starts as ranks 1 .. ep - 1; in a phase warp 0 pairs rank 0 with
        // ring[0] and warp w pairs ring[w] with ring[ep - 1 - w], the first of each receiving first; then ring[]
        // rotates by one.
        std::vector<int> ring(ep - 1);
        std::iota(ring.begin(), ring.end(), 1);
        for (int phase = 0; phase < ep - 1; ++phase)
        {
            std::vector<int> ringPartner(ep, -1), ringLeft(ep, 0);
            for (int warp = 0; warp < ep / 2; ++warp)
            {
                int const left = warp == 0 ? 0 : ring[warp];
                int const right = warp == 0 ? ring[0] : ring[ep - 1 - warp];
                ringPartner[left] = right;
                ringPartner[right] = left;
                ringLeft[left] = 1;
            }
            for (int lane = 0; lane < ep; ++lane)
            {
                ASSERT_EQ(partners[row(lane) + phase], ringPartner[lane])
                    << "halo_q_partner, EP" << ep << " rank " << lane << " phase " << phase;
                ASSERT_EQ(lefts[row(lane) + phase], ringLeft[lane])
                    << "halo_q_partner's left, EP" << ep << " rank " << lane << " phase " << phase;
                for (int other = 0; other < 32; ++other)
                {
                    ASSERT_EQ((livePhases[row(lane) + other] >> phase) & 1U, other == ringPartner[lane] ? 1U : 0U)
                        << "halo_q_live_phases, EP" << ep << " rank " << lane << " phase " << phase << " partner "
                        << other;
                }
            }
            int const last = ring[ep - 2];
            for (int index = ep - 2; index > 0; --index)
            {
                ring[index] = ring[index - 1];
            }
            ring[0] = last;
        }
        // No phase past the tournament's ep - 1, and a lane that is no rank meets nobody and is its own partner.
        for (int lane = 0; lane < 32; ++lane)
        {
            for (int other = 0; other < 32; ++other)
            {
                ASSERT_EQ(livePhases[row(lane) + other] >> (lane < ep ? ep - 1 : 0), 0U)
                    << "halo_q_live_phases, EP" << ep << " lane " << lane << " partner " << other;
            }
            for (int phase = 0; phase < ep - 1 && lane >= ep; ++phase)
            {
                ASSERT_EQ(partners[row(lane) + phase], lane) << "halo_q_partner, EP" << ep << " lane " << lane;
                ASSERT_EQ(lefts[row(lane) + phase], 0) << "halo_q_partner's left, EP" << ep << " lane " << lane;
            }
        }
    }
}

} // namespace tensorrt_llm::tests::kernels::haloq
