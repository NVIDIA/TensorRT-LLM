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

// Host tests of the HALO-R planner, slot rule and relay work assignment. The planner text is the one the scheduler
// kernel runs; here a warp is 32 host threads that meet at a barrier for every collective, so a lane that skips or
// repeats a collective deadlocks the test instead of passing.

#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloRPlanner.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceRelayWork.h"

#include <gtest/gtest.h>

#include <barrier>
#include <cstring>
#include <thread>
#include <vector>

namespace halo_r = tensorrt_llm::kernels::halo_r;

namespace
{

constexpr int kWarpSize = 32;

//! 32 lockstep host threads; every collective is a full exchange between two barrier phases.
struct HostWarp
{
    struct State
    {
        std::barrier<> barrier{kWarpSize};
        unsigned slots[kWarpSize];
    };

    static inline State* state = nullptr;
    static inline thread_local int laneId = 0;

    enum class Op
    {
        kMax,
        kAdd,
        kBallot
    };

    static unsigned exchange(unsigned value, Op op)
    {
        state->slots[laneId] = value;
        state->barrier.arrive_and_wait();
        unsigned result = 0;
        for (int lane = 0; lane < kWarpSize; ++lane)
        {
            unsigned const x = state->slots[lane];
            if (op == Op::kMax)
            {
                result = x > result ? x : result;
            }
            else if (op == Op::kAdd)
            {
                result += x;
            }
            else
            {
                result |= (x ? 1U : 0U) << lane;
            }
        }
        state->barrier.arrive_and_wait();
        return result;
    }

    static int lane()
    {
        return laneId;
    }

    static void sync()
    {
        state->barrier.arrive_and_wait();
    }

    static unsigned reduceMax(unsigned value)
    {
        return exchange(value, Op::kMax);
    }

    static int reduceAdd(int value)
    {
        return static_cast<int>(exchange(static_cast<unsigned>(value), Op::kAdd));
    }

    static unsigned ballot(bool predicate)
    {
        return exchange(predicate ? 1U : 0U, Op::kBallot);
    }

    static unsigned findFirstSet(unsigned value)
    {
        return static_cast<unsigned>(__builtin_ffs(static_cast<int>(value)));
    }

    static unsigned popCount(unsigned value)
    {
        return static_cast<unsigned>(__builtin_popcount(value));
    }
};

struct PlanResult
{
    halo_r::PlanStatus status;
    std::vector<int> plan;
    std::vector<int> quota;
    std::vector<int> outIds;
    std::vector<int> outOwners;
    std::vector<int> hotTable;
    std::vector<int> loadAfter;
};

PlanResult runPlanner(std::vector<int> const& histogram, halo_r::PlannerConfig const& cfg)
{
    PlanResult result;
    result.plan.assign(halo_r::kPlanWords, -7);
    result.quota.assign(halo_r::kMaxBroadcasts * cfg.ep, -7);
    result.outIds.assign(cfg.helperSlots > 0 ? cfg.helperSlots : 1, -7);
    result.outOwners.assign(result.outIds.size(), -7);
    result.hotTable.assign(static_cast<std::size_t>(cfg.ep > 0 ? cfg.ep : 1) * halo_r::kMaxHelperSlots, -7);
    result.loadAfter.assign(cfg.ep > 0 ? cfg.ep : 1, -7);
    halo_r::PlannerScratch scratch;
    // Stale shared memory must not change the plan.
    std::memset(&scratch, 0xA5, sizeof(scratch));
    halo_r::PlannerOutputs const outputs{result.plan.data(), result.quota.data(), result.outIds.data(),
        result.outOwners.data(), result.hotTable.data(), result.loadAfter.data(), nullptr};
    std::vector<halo_r::PlanStatus> statuses(kWarpSize);

    HostWarp::State state;
    HostWarp::state = &state;
    std::vector<std::thread> lanes;
    for (int lane = 0; lane < kWarpSize; ++lane)
    {
        lanes.emplace_back(
            [&, lane]
            {
                HostWarp::laneId = lane;
                statuses[lane] = halo_r::planHaloR<HostWarp>(histogram.data(), cfg, &scratch, outputs);
            });
    }
    for (auto& thread : lanes)
    {
        thread.join();
    }
    HostWarp::state = nullptr;
    for (auto status : statuses)
    {
        EXPECT_EQ(status, statuses[0]) << "the planner status must be warp-uniform";
    }
    result.status = statuses[0];
    return result;
}

int field(PlanResult const& r, halo_r::PlanField f, int b)
{
    return halo_r::planField(r.plan.data(), f)[b];
}

//! Eight ranks of four experts. Rank 0 holds the hot expert 0 (300 of its 400 routes); ranks 1, 4 and 5 are idle and
//! the others sit at the balanced load of 100. The plan sends expert 0 to ranks 1, 4 and 5 with 100 routes each.
std::vector<int> workedExampleHistogram()
{
    return {300, 40, 30, 30, 0, 0, 0, 0, 40, 30, 20, 10, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0, 0, 0, 40, 30, 20, 10, 40, 30,
        20, 10};
}

} // namespace

TEST(MoeRebalanceHaloRPlannerTest, WorkedExampleReplicatesHotExpertToIdleRanks)
{
    halo_r::PlannerConfig const cfg{/*ep=*/8, /*experts=*/32, /*helperSlots=*/2, /*sendCount=*/4, /*sendWeight=*/0,
        /*localRank=*/4};
    PlanResult const r = runPlanner(workedExampleHistogram(), cfg);
    ASSERT_EQ(r.status, halo_r::PlanStatus::kOk);
    ASSERT_EQ(r.plan[1], 1);
    EXPECT_EQ(field(r, halo_r::PlanField::kExpert, 0), 0);
    EXPECT_EQ(field(r, halo_r::PlanField::kOwner, 0), 0);
    EXPECT_EQ(field(r, halo_r::PlanField::kDestinationMask, 0), (1 << 1) | (1 << 4) | (1 << 5));
    EXPECT_EQ(field(r, halo_r::PlanField::kHelperLowMask, 0), 0);
    EXPECT_EQ(field(r, halo_r::PlanField::kHelperHighMask, 0), 0);
    EXPECT_EQ(field(r, halo_r::PlanField::kRows, 0), 300);
    std::vector<int> const quota(r.quota.begin(), r.quota.begin() + cfg.ep);
    EXPECT_EQ(quota, (std::vector<int>{0, 100, 0, 0, 100, 100, 0, 0}));
    EXPECT_EQ(r.loadAfter, (std::vector<int>(cfg.ep, 100)));
    EXPECT_EQ(r.outIds, (std::vector<int>{0, -1}));
    EXPECT_EQ(r.outOwners, (std::vector<int>{0, -1}));
    for (int b = 1; b < halo_r::kPlanFieldStride; ++b)
    {
        ASSERT_EQ(field(r, halo_r::PlanField::kExpert, b), 0) << "unused plan entries must be zero";
    }
}

TEST(MoeRebalanceHaloRPlannerTest, PlanIsIdenticalOnEveryRank)
{
    std::vector<int> const histogram = workedExampleHistogram();
    halo_r::PlannerConfig cfg{8, 32, 2, 4, 0, 0};
    PlanResult const reference = runPlanner(histogram, cfg);
    for (int rank = 1; rank < cfg.ep; ++rank)
    {
        cfg.localRank = rank;
        PlanResult const r = runPlanner(histogram, cfg);
        EXPECT_EQ(r.plan, reference.plan);
        EXPECT_EQ(r.hotTable, reference.hotTable);
        bool const receives = rank == 1 || rank == 4 || rank == 5;
        EXPECT_EQ(r.outIds[0], receives ? 0 : -1);
    }
}

TEST(MoeRebalanceHaloRPlannerTest, HelperIndexUsesBothMaskBits)
{
    // Rank 0 is the only rank below the target and receives three experts, in ascending order, into helper slots
    // 0, 1 and 2; slot 2 is encoded in the high mask.
    std::vector<int> const histogram{0, 0, 80, 10, 40, 200, 40, 10, 10, 80, 200, 0, 0, 80, 0, 200};
    halo_r::PlannerConfig const cfg{4, 16, 4, 4, 0, 0};
    PlanResult const r = runPlanner(histogram, cfg);
    ASSERT_EQ(r.status, halo_r::PlanStatus::kOk);
    ASSERT_EQ(r.plan[1], 3);
    int const experts[] = {5, 10, 15};
    int const rows[] = {52, 52, 42};
    for (int b = 0; b < 3; ++b)
    {
        EXPECT_EQ(field(r, halo_r::PlanField::kExpert, b), experts[b]);
        EXPECT_EQ(field(r, halo_r::PlanField::kOwner, b), experts[b] / 4);
        EXPECT_EQ(field(r, halo_r::PlanField::kDestinationMask, b), 1);
        EXPECT_EQ(field(r, halo_r::PlanField::kRows, b), rows[b]);
        EXPECT_EQ(halo_r::helperIndex(static_cast<unsigned>(field(r, halo_r::PlanField::kHelperLowMask, b)),
                      static_cast<unsigned>(field(r, halo_r::PlanField::kHelperHighMask, b)), 0),
            b);
        EXPECT_EQ(r.quota[b * cfg.ep + 0], rows[b]);
        EXPECT_EQ(r.quota[b * cfg.ep + experts[b] / 4], histogram[experts[b]] - rows[b]);
    }
    EXPECT_EQ(r.outIds, (std::vector<int>{5, 10, 15, -1}));
    EXPECT_EQ(r.outOwners, (std::vector<int>{1, 2, 3, -1}));
    EXPECT_EQ(r.loadAfter, (std::vector<int>{236, 238, 238, 238}));
}

TEST(MoeRebalanceHaloRPlannerTest, BalancedOrEmptyLoadPlansNothing)
{
    halo_r::PlannerConfig const cfg{4, 16, 2, 2, 0, 1};
    PlanResult const balanced = runPlanner(std::vector<int>(16, 25), cfg);
    ASSERT_EQ(balanced.status, halo_r::PlanStatus::kOk);
    EXPECT_EQ(balanced.plan[1], 0);
    EXPECT_EQ(balanced.outIds, (std::vector<int>{-1, -1}));
    PlanResult const empty = runPlanner(std::vector<int>(16, 0), cfg);
    ASSERT_EQ(empty.status, halo_r::PlanStatus::kOk);
    EXPECT_EQ(empty.plan[1], 0);
}

TEST(MoeRebalanceHaloRPlannerTest, ReceiversNeverExceedTheirHelperSlots)
{
    // Every expert of rank 0 is hot; the other ranks can take at most `helperSlots` distinct experts each.
    std::vector<int> histogram(32 * 12, 1);
    for (int local = 0; local < 12; ++local)
    {
        histogram[local] = 1000 - local;
    }
    for (int helpers = 1; helpers <= halo_r::kMaxHelperSlots; ++helpers)
    {
        halo_r::PlannerConfig const cfg{32, 32 * 12, helpers, 4, 0, 0};
        PlanResult const r = runPlanner(histogram, cfg);
        ASSERT_EQ(r.status, halo_r::PlanStatus::kOk);
        std::vector<int> received(cfg.ep, 0);
        int moved = 0;
        for (int b = 0; b < r.plan[1]; ++b)
        {
            unsigned const mask = static_cast<unsigned>(field(r, halo_r::PlanField::kDestinationMask, b));
            EXPECT_EQ((mask >> field(r, halo_r::PlanField::kOwner, b)) & 1U, 0U) << "an owner never receives";
            for (int d = 0; d < cfg.ep; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    ++received[d];
                    moved += r.quota[b * cfg.ep + d];
                    EXPECT_LT(halo_r::helperIndex(static_cast<unsigned>(field(r, halo_r::PlanField::kHelperLowMask, b)),
                                  static_cast<unsigned>(field(r, halo_r::PlanField::kHelperHighMask, b)), d),
                        helpers);
                }
            }
            EXPECT_LE(field(r, halo_r::PlanField::kRows, b), histogram[field(r, halo_r::PlanField::kExpert, b)]);
        }
        for (int d = 0; d < cfg.ep; ++d)
        {
            EXPECT_LE(received[d], helpers);
        }
        EXPECT_GT(moved, 0);
    }
}

TEST(MoeRebalanceHaloRPlannerTest, RejectsConfigurationsOutsideItsLimits)
{
    std::vector<int> const histogram(64 * 12, 1);
    // ep above one warp, a helper index wider than two bits, more exports than the order field, more sends than home
    // experts, an unsupported send weight and a local rank outside the EP group.
    halo_r::PlannerConfig const invalid[] = {
        {33, 33 * 12, 2, 2, 0, 0},
        {8, 96, 5, 2, 0, 0},
        {32, 32 * 12, 2, 5, 0, 0},
        {8, 32, 2, 5, 0, 0},
        {8, 96, 2, 2, halo_r::kMaxSendWeight + 1, 0},
        {8, 96, 2, 2, 0, 8},
        {8, 97, 2, 2, 0, 0},
    };
    for (auto const& cfg : invalid)
    {
        EXPECT_FALSE(halo_r::validPlannerConfig(cfg));
        if (cfg.ep <= halo_r::kMaxRanks)
        {
            EXPECT_EQ(runPlanner(histogram, cfg).status, halo_r::PlanStatus::kBadConfig);
        }
    }
    EXPECT_TRUE(halo_r::validPlannerConfig({32, 384, 4, 4, 0, 31}));
}

TEST(MoeRebalanceHaloRSlotTest, SelectedRoutesGoToOwnerOrHelperSlot)
{
    int const home = 4;
    int const localSlots = home + 2;
    unsigned const mask = (1U << 1) | (1U << 3);
    unsigned const low = 1U << 3;
    // The owner keeps its home slot; receivers use home + helper index; any other rank is rejected.
    EXPECT_EQ(halo_r::selectedRouteSlot(9, 2, 2, mask, low, 0, home, localSlots), 2 * localSlots + 1);
    EXPECT_EQ(halo_r::selectedRouteSlot(9, 1, 2, mask, low, 0, home, localSlots), 1 * localSlots + home);
    EXPECT_EQ(halo_r::selectedRouteSlot(9, 3, 2, mask, low, 0, home, localSlots), 3 * localSlots + home + 1);
    EXPECT_EQ(halo_r::selectedRouteSlot(9, 0, 2, mask, low, 0, home, localSlots), -1);
    EXPECT_EQ(halo_r::helperIndex(1U << 5, 1U << 5, 5), 3);
}

TEST(MoeRebalanceHaloRSlotTest, TimeoutFallbackUsesTheOwnerResidentSlot)
{
    int const ep = 4;
    int const home = 4;
    int const localSlots = home + 2;
    // A scheduler worker that times out routes every expert to the slot its owner keeps when nothing is rebalanced.
    EXPECT_EQ(halo_r::homeSlot(0, home, localSlots), 0);
    EXPECT_EQ(halo_r::homeSlot(9, home, localSlots), 2 * localSlots + 1);
    EXPECT_EQ(halo_r::homeSlot(ep * home - 1, home, localSlots), (ep - 1) * localSlots + home - 1);
    for (int expert = 0; expert < ep * home; ++expert)
    {
        int const owner = expert / home;
        EXPECT_EQ(halo_r::homeSlot(expert, home, localSlots),
            halo_r::selectedRouteSlot(expert, owner, owner, 0U, 0U, 0U, home, localSlots));
    }
}

TEST(MoeRebalanceRelayWorkTest, DestinationsPartitionEverySlice)
{
    unsigned long long const planeBytes[] = {3 * halo_r::kRelaySliceBytes + 100, 4, halo_r::kRelaySliceBytes};
    int const planes = 3;
    long long const slices = halo_r::relayBundleSlices(planeBytes, planes);
    ASSERT_EQ(slices, 4 + 1 + 1);
    unsigned const mask = (1U << 2) | (1U << 7) | (1U << 30);
    std::vector<int> copies(slices, 0);
    for (int rank = 0; rank < 32; ++rank)
    {
        int destinations = 0;
        int const index = halo_r::relayDestinationIndex(mask, rank, &destinations);
        EXPECT_EQ(destinations, 3);
        if (index < 0)
        {
            EXPECT_EQ(halo_r::relaySliceCount(slices, destinations, index), 0);
            continue;
        }
        long long const count = halo_r::relaySliceCount(slices, destinations, index);
        for (long long k = 0; k < count; ++k)
        {
            ++copies[halo_r::relaySliceOf(k, destinations, index)];
        }
    }
    for (int count : copies)
    {
        EXPECT_EQ(count, 1) << "every slice is pulled by exactly one destination";
    }
    int plane = -1;
    unsigned long long offset = 0;
    unsigned long long bytes = 0;
    ASSERT_TRUE(halo_r::relayLocateSlice(planeBytes, planes, 3, &plane, &offset, &bytes));
    EXPECT_EQ(plane, 0);
    EXPECT_EQ(offset, 3 * halo_r::kRelaySliceBytes);
    EXPECT_EQ(bytes, 100U);
    ASSERT_TRUE(halo_r::relayLocateSlice(planeBytes, planes, 4, &plane, &offset, &bytes));
    EXPECT_EQ(plane, 1);
    EXPECT_EQ(bytes, 4U);
    EXPECT_FALSE(halo_r::relayLocateSlice(planeBytes, planes, slices, &plane, &offset, &bytes));
}

TEST(MoeRebalanceRelayWorkTest, ItemsMapBackToTheirBroadcast)
{
    int const prefix[] = {0, 5, 5, 12};
    long long k = -1;
    EXPECT_EQ(halo_r::relayLocateItem(prefix, 3, 0, &k), 0);
    EXPECT_EQ(k, 0);
    EXPECT_EQ(halo_r::relayLocateItem(prefix, 3, 4, &k), 0);
    EXPECT_EQ(k, 4);
    // Entry 1 has no items; item 5 belongs to entry 2.
    EXPECT_EQ(halo_r::relayLocateItem(prefix, 3, 5, &k), 2);
    EXPECT_EQ(k, 0);
    EXPECT_EQ(halo_r::relayLocateItem(prefix, 3, 11, &k), 2);
    EXPECT_EQ(k, 6);
}
