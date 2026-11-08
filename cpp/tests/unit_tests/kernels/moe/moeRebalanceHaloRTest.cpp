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

#include <algorithm>
#include <barrier>
#include <cstring>
#include <random>
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
    std::vector<int> plan;
    std::vector<int> quota;
    std::vector<int> outIds;
    std::vector<int> outOwners;
    std::vector<int> hotTable;
    std::vector<int> loadAfter;
};

PlanResult runPlanner(
    std::vector<int> const& histogram, halo_r::PlannerConfig const& cfg, std::vector<int> const* resident = nullptr)
{
    PlanResult result;
    result.plan.assign(halo_r::kPlanWords, -7);
    result.quota.assign(halo_r::kMaxHaloRBroadcasts * cfg.ep, -7);
    result.outIds.assign(cfg.helperSlots > 0 ? cfg.helperSlots : 1, -7);
    result.outOwners.assign(result.outIds.size(), -7);
    result.hotTable.assign(static_cast<std::size_t>(cfg.ep > 0 ? cfg.ep : 1) * halo_r::kMaxHelperSlots, -7);
    result.loadAfter.assign(cfg.ep > 0 ? cfg.ep : 1, -7);
    halo_r::PlannerScratch scratch;
    // Stale shared memory must not change the plan.
    std::memset(&scratch, 0xA5, sizeof(scratch));
    halo_r::PlannerOutputs const outputs{result.plan.data(), result.quota.data(), result.outIds.data(),
        result.outOwners.data(), result.hotTable.data(), result.loadAfter.data()};

    HostWarp::State state;
    HostWarp::state = &state;
    std::vector<std::thread> lanes;
    for (int lane = 0; lane < kWarpSize; ++lane)
    {
        lanes.emplace_back(
            [&, lane]
            {
                HostWarp::laneId = lane;
                halo_r::planHaloR<HostWarp>(
                    histogram.data(), cfg, &scratch, outputs, resident != nullptr ? resident->data() : nullptr);
            });
    }
    for (auto& thread : lanes)
    {
        thread.join();
    }
    HostWarp::state = nullptr;
    return result;
}

int field(PlanResult const& r, halo_r::HaloRPlanField f, int b)
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
    halo_r::PlannerConfig const cfg{/*ep=*/8, /*experts=*/32, /*helperSlots=*/2, /*sendCount=*/4, /*localRank=*/4};
    PlanResult const r = runPlanner(workedExampleHistogram(), cfg);
    ASSERT_EQ(r.plan[1], 1);
    EXPECT_EQ(field(r, halo_r::HaloRPlanField::kExpert, 0), 0);
    EXPECT_EQ(field(r, halo_r::HaloRPlanField::kOwner, 0), 0);
    EXPECT_EQ(field(r, halo_r::HaloRPlanField::kDestinationMask, 0), (1 << 1) | (1 << 4) | (1 << 5));
    EXPECT_EQ(field(r, halo_r::HaloRPlanField::kHelperLowMask, 0), 0);
    EXPECT_EQ(field(r, halo_r::HaloRPlanField::kHelperHighMask, 0), 0);
    EXPECT_EQ(field(r, halo_r::HaloRPlanField::kRows, 0), 300);
    std::vector<int> const quota(r.quota.begin(), r.quota.begin() + cfg.ep);
    EXPECT_EQ(quota, (std::vector<int>{0, 100, 0, 0, 100, 100, 0, 0}));
    EXPECT_EQ(r.loadAfter, (std::vector<int>(cfg.ep, 100)));
    EXPECT_EQ(r.outIds, (std::vector<int>{0, -1}));
    EXPECT_EQ(r.outOwners, (std::vector<int>{0, -1}));
    for (int b = 1; b < halo_r::kPlanFieldStride; ++b)
    {
        ASSERT_EQ(field(r, halo_r::HaloRPlanField::kExpert, b), 0) << "unused plan entries must be zero";
    }
}

TEST(MoeRebalanceHaloRPlannerTest, PlanIsIdenticalOnEveryRank)
{
    std::vector<int> const histogram = workedExampleHistogram();
    halo_r::PlannerConfig cfg{8, 32, 2, 4, 0};
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
    halo_r::PlannerConfig const cfg{4, 16, 4, 4, 0};
    PlanResult const r = runPlanner(histogram, cfg);
    ASSERT_EQ(r.plan[1], 3);
    int const experts[] = {5, 10, 15};
    int const rows[] = {52, 52, 42};
    for (int b = 0; b < 3; ++b)
    {
        EXPECT_EQ(field(r, halo_r::HaloRPlanField::kExpert, b), experts[b]);
        EXPECT_EQ(field(r, halo_r::HaloRPlanField::kOwner, b), experts[b] / 4);
        EXPECT_EQ(field(r, halo_r::HaloRPlanField::kDestinationMask, b), 1);
        EXPECT_EQ(field(r, halo_r::HaloRPlanField::kRows, b), rows[b]);
        EXPECT_EQ(halo_r::helperIndex(static_cast<unsigned>(field(r, halo_r::HaloRPlanField::kHelperLowMask, b)),
                      static_cast<unsigned>(field(r, halo_r::HaloRPlanField::kHelperHighMask, b)), 0),
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
    halo_r::PlannerConfig const cfg{4, 16, 2, 2, 1};
    PlanResult const balanced = runPlanner(std::vector<int>(16, 25), cfg);
    EXPECT_EQ(balanced.plan[1], 0);
    EXPECT_EQ(balanced.outIds, (std::vector<int>{-1, -1}));
    PlanResult const empty = runPlanner(std::vector<int>(16, 0), cfg);
    EXPECT_EQ(empty.plan[1], 0);
}

TEST(MoeRebalanceHaloRPlannerTest, ReceiversNeverExceedTheirHelperSlots)
{
    // Two ranks of eight experts. Rank 0 holds eight experts of 100 routes each and offers its four hottest, 0 to 3;
    // rank 1 is idle and has room for all four (target 400), so only its helper slots limit what it receives:
    // experts 0, 1, ... one per slot, 100 routes each, in slot order.
    std::vector<int> histogram(16, 0);
    for (int expert = 0; expert < 8; ++expert)
    {
        histogram[expert] = 100;
    }
    for (int helpers = 1; helpers <= halo_r::kMaxHelperSlots; ++helpers)
    {
        SCOPED_TRACE(helpers);
        halo_r::PlannerConfig const cfg{/*ep=*/2, /*experts=*/16, helpers, /*sendCount=*/4, /*localRank=*/1};
        PlanResult const r = runPlanner(histogram, cfg);
        ASSERT_EQ(r.plan[1], helpers);
        for (int b = 0; b < helpers; ++b)
        {
            EXPECT_EQ(field(r, halo_r::HaloRPlanField::kExpert, b), b);
            EXPECT_EQ(field(r, halo_r::HaloRPlanField::kOwner, b), 0);
            EXPECT_EQ(field(r, halo_r::HaloRPlanField::kDestinationMask, b), 0b10);
            EXPECT_EQ(field(r, halo_r::HaloRPlanField::kRows, b), 100);
            EXPECT_EQ(halo_r::helperIndex(static_cast<unsigned>(field(r, halo_r::HaloRPlanField::kHelperLowMask, b)),
                          static_cast<unsigned>(field(r, halo_r::HaloRPlanField::kHelperHighMask, b)), 1),
                b);
            EXPECT_EQ(r.quota[b * cfg.ep + 0], 0);
            EXPECT_EQ(r.quota[b * cfg.ep + 1], 100);
            EXPECT_EQ(r.outIds[b], b);
            EXPECT_EQ(r.outOwners[b], 0);
        }
        EXPECT_EQ(r.loadAfter[0], 800 - 100 * helpers);
        EXPECT_EQ(r.loadAfter[1], 100 * helpers);
    }
}

TEST(MoeRebalanceHaloRPlannerTest, RejectsConfigurationsOutsideItsLimits)
{
    // ep above one warp (within the 384-expert cap), a helper index wider than two bits, more exports than the order
    // field, more sends than home experts, a local rank outside the EP group and experts that do not divide over the
    // ranks.
    halo_r::PlannerConfig const invalid[] = {
        {33, 33 * 11, 2, 2, 0},
        {8, 96, 5, 2, 0},
        {32, 32 * 12, 2, 5, 0},
        {8, 32, 2, 5, 0},
        {8, 96, 2, 2, 8},
        {8, 97, 2, 2, 0},
    };
    for (auto const& cfg : invalid)
    {
        EXPECT_FALSE(halo_r::validPlannerConfig(cfg));
    }
    EXPECT_TRUE(halo_r::validPlannerConfig({32, 384, 4, 4, 31}));
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

TEST(MoeRebalanceHaloRSlotTest, OwnerRoutesKeepTheirResidentSlot)
{
    int const home = 4;
    int const localSlots = home + 2;
    // A selected route that the prefix leaves on the owner goes to the slot the owner keeps when nothing is
    // rebalanced, whatever the receivers: rank r's home slots are r * 6 + 0..3, before its two helper slots.
    int const resident[] = {0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21};
    unsigned const mask = (1U << 0) | (1U << 3);
    for (int expert = 0; expert < 16; ++expert)
    {
        int const owner = expert / home;
        EXPECT_EQ(halo_r::selectedRouteSlot(expert, owner, owner, 0U, 0U, 0U, home, localSlots), resident[expert]);
        EXPECT_EQ(halo_r::selectedRouteSlot(expert, owner, owner, mask & ~(1U << owner), 1U << 3, 0U, home, localSlots),
            resident[expert]);
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

TEST(MoeRebalanceRelayWorkTest, NeedSetsDropReceiversWhoseSlotHoldsTheExpert)
{
    // Eight ranks with two helper slots each; expert 9 goes to ranks 1, 2, 5 and 6, into helper slots 1, 0, 1 and 0.
    unsigned const mask = (1U << 1) | (1U << 2) | (1U << 5) | (1U << 6);
    unsigned const low = (1U << 1) | (1U << 5);
    int const helpers = 2;
    // [rank][slot]: rank 1 holds 9 in slot 1 (its slot), rank 2 holds 9 in slot 1 (not its slot 0), rank 5 holds 4 in
    // slot 1, rank 6 holds 9 in slot 0 (its slot). Ranks 0 and 3 hold 9 but receive nothing.
    std::vector<int> const table{9, -1, -1, 9, -1, 9, 9, -1, -1, -1, -1, 4, 9, -1, -1, -1};
    EXPECT_EQ(halo_r::relayNeedMask(mask, low, 0U, 9, table.data(), helpers), (1U << 2) | (1U << 5));
    // A table that holds the expert nowhere needs every receiver, and one that holds it in every receiver's slot needs
    // none.
    std::vector<int> const unknown(16, -1);
    EXPECT_EQ(halo_r::relayNeedMask(mask, low, 0U, 9, unknown.data(), helpers), mask);
    std::vector<int> const resident{-1, -1, -1, 9, 9, -1, -1, -1, -1, -1, -1, 9, 9, -1, -1, -1};
    EXPECT_EQ(halo_r::relayNeedMask(mask, low, 0U, 9, resident.data(), helpers), 0U);
    // The high mask selects slots 2 and 3: rank 1 holds 9 in slot 3.
    std::vector<int> const wide{-1, -1, -1, -1, -1, -1, -1, 9};
    EXPECT_EQ(halo_r::relayNeedMask(0b11U, 0b10U, 0b10U, 9, wide.data(), 4), 0b01U);
    // Without a residency table every receiver needs the copy.
    EXPECT_EQ(halo_r::relayNeedMask(0b1010U, 0U, 0U, 5, nullptr, 2), 0b1010U);
}

TEST(MoeRebalanceRelayWorkTest, NeedSetsSplitEverySliceOverTheReceiversThatNeedIt)
{
    std::mt19937 rng(20887);
    for (int trial = 0; trial < 10000; ++trial)
    {
        int const ep = 2 * (1 + static_cast<int>(rng() % 16));
        // Any subset of the ranks, empty included.
        unsigned const need = static_cast<unsigned>(rng()) & (ep == 32 ? ~0U : ((1U << ep) - 1U));
        long long const slices = 1 + static_cast<long long>(rng() % 40);
        int const members = __builtin_popcount(need);
        std::vector<int> pulls(slices, 0);
        for (int rank = 0; rank < ep; ++rank)
        {
            int destinations = 0;
            int const index = halo_r::relayDestinationIndex(need, rank, &destinations);
            ASSERT_EQ(destinations, members);
            long long const count = halo_r::relaySliceCount(slices, destinations, index);
            if (!((need >> rank) & 1U))
            {
                ASSERT_EQ(index, -1);
                ASSERT_EQ(count, 0) << "a rank outside the need set pulls nothing";
            }
            for (long long k = 0; k < count; ++k)
            {
                ++pulls[halo_r::relaySliceOf(k, destinations, index)];
            }
        }
        for (long long slice = 0; slice < slices; ++slice)
        {
            ASSERT_EQ(pulls[slice], need ? 1 : 0) << "trial " << trial << " slice " << slice;
        }
    }
}

namespace
{

//! A skewed random histogram: a few hot experts on random ranks over a random background.
std::vector<int> randomHistogram(std::mt19937& rng, int experts)
{
    std::vector<int> histogram(experts);
    for (int& count : histogram)
    {
        count = rng() % 4 == 0 ? 0 : static_cast<int>(rng() % 200);
    }
    for (int hot = 0; hot < 1 + static_cast<int>(rng() % 6); ++hot)
    {
        histogram[rng() % experts] += static_cast<int>(rng() % 4000);
    }
    return histogram;
}

std::vector<int> hotRow(PlanResult const& r, halo_r::PlannerConfig const& cfg, int rank)
{
    auto const begin = r.hotTable.begin() + rank * cfg.helperSlots;
    return std::vector<int>(begin, begin + cfg.helperSlots);
}

} // namespace

TEST(MoeRebalanceHaloRPlannerTest, ResidentExpertsKeepTheirSlotsAndNothingElseChanges)
{
    std::mt19937 rng(4);
    int hits = 0;
    for (int trial = 0; trial < 300; ++trial)
    {
        int const ep = 2 * (1 + static_cast<int>(rng() % 16));
        int const home = 4 + static_cast<int>(rng() % 9);
        if (ep * home > halo_r::kPlanFieldStride)
        {
            continue;
        }
        int const helpers = 1 + static_cast<int>(rng() % halo_r::kMaxHelperSlots);
        halo_r::PlannerConfig cfg{ep, ep * home, helpers, 4, static_cast<int>(rng() % ep)};
        std::vector<int> const histogram = randomHistogram(rng, cfg.experts);
        PlanResult const ascending = runPlanner(histogram, cfg);

        // Nothing resident: bitwise the ascending rule.
        std::vector<int> const unknown(static_cast<std::size_t>(ep) * helpers, -1);
        PlanResult const none = runPlanner(histogram, cfg, &unknown);
        ASSERT_EQ(none.plan, ascending.plan) << "trial " << trial;
        ASSERT_EQ(none.quota, ascending.quota);
        ASSERT_EQ(none.outIds, ascending.outIds);
        ASSERT_EQ(none.hotTable, ascending.hotTable);

        // A random table that holds some received experts, in any slot, sometimes twice.
        std::vector<int> table(unknown.size());
        for (int d = 0; d < ep; ++d)
        {
            std::vector<int> const row = hotRow(ascending, cfg, d);
            for (int q = 0; q < helpers; ++q)
            {
                int const pick = static_cast<int>(rng() % 3);
                int const received = row[rng() % helpers];
                table[d * helpers + q]
                    = pick == 0 ? -1 : (pick == 1 ? received : static_cast<int>(rng() % cfg.experts));
            }
        }
        PlanResult const r = runPlanner(histogram, cfg, &table);
        // Moves, rows, quotas and load are unchanged; only the helper slot bits may move.
        ASSERT_EQ(r.plan[1], ascending.plan[1]);
        for (auto field : {halo_r::HaloRPlanField::kExpert, halo_r::HaloRPlanField::kOwner,
                 halo_r::HaloRPlanField::kDestinationMask, halo_r::HaloRPlanField::kRows})
        {
            for (int b = 0; b < halo_r::kPlanFieldStride; ++b)
            {
                ASSERT_EQ(
                    halo_r::planField(r.plan.data(), field)[b], halo_r::planField(ascending.plan.data(), field)[b]);
            }
        }
        ASSERT_EQ(r.quota, ascending.quota);
        ASSERT_EQ(r.loadAfter, ascending.loadAfter);
        for (int d = 0; d < ep; ++d)
        {
            std::vector<int> row = hotRow(r, cfg, d);
            std::vector<int> sortedRow = row;
            std::vector<int> sortedAscending = hotRow(ascending, cfg, d);
            std::sort(sortedRow.begin(), sortedRow.end());
            std::sort(sortedAscending.begin(), sortedAscending.end());
            ASSERT_EQ(sortedRow, sortedAscending) << "the same experts, possibly in other slots";
            for (int q = 0; q < helpers; ++q)
            {
                int const expert = row[q];
                if (expert < 0)
                {
                    continue;
                }
                ASSERT_EQ(std::count(row.begin(), row.end(), expert), 1) << "an expert holds one slot of a row";
                int lowest = -1;
                for (int k = 0; k < helpers && lowest < 0; ++k)
                {
                    lowest = table[d * helpers + k] == expert ? k : -1;
                }
                if (lowest >= 0)
                {
                    ASSERT_EQ(q, lowest) << "a resident expert keeps the lowest slot that holds it";
                    ++hits;
                }
            }
        }
        // The plan's helper bits agree with the table of slots.
        for (int b = 0; b < r.plan[1]; ++b)
        {
            unsigned const mask = static_cast<unsigned>(field(r, halo_r::HaloRPlanField::kDestinationMask, b));
            for (int d = 0; d < ep; ++d)
            {
                if ((mask >> d) & 1U)
                {
                    int const slot = halo_r::helperIndex(
                        static_cast<unsigned>(field(r, halo_r::HaloRPlanField::kHelperLowMask, b)),
                        static_cast<unsigned>(field(r, halo_r::HaloRPlanField::kHelperHighMask, b)), d);
                    ASSERT_EQ(r.hotTable[d * helpers + slot], field(r, halo_r::HaloRPlanField::kExpert, b));
                }
            }
        }
        // Every simulated rank computes the same plan from the same table.
        cfg.localRank = (cfg.localRank + 1) % ep;
        ASSERT_EQ(runPlanner(histogram, cfg, &table).plan, r.plan);
    }
    EXPECT_GT(hits, 0) << "the trials must cover resident experts";
}
