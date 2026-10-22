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

// HALO-R replica planner: plan layout, the warp-parallel greedy planner and the selected-route slot rule.
//
// The planner runs as one warp with one lane per EP rank. Every rank computes the same plan from the same global
// routed-token histogram. Each overloaded rank offers its hottest home experts; on every step the planner moves rows of
// one expert from its owner to the eligible rank with the most room below the balanced target load, and records which
// helper slot of that rank receives the expert.
//
// The planner is written against a warp policy so that the same text runs on the GPU (one warp of the scheduler
// kernel) and in host unit tests (32 threads in lockstep). A policy provides static functions lane(), sync(),
// reduceMax(unsigned), reduceAdd(int), ballot(bool), findFirstSet(unsigned) and popCount(unsigned); all 32 lanes call
// every collective in the same order, and lanes at or above the EP size carry no rank.

#pragma once

#include "tensorrt_llm/common/config.h"

#if defined(__CUDACC__)
#define TLLM_HALO_R_HOST_DEVICE __host__ __device__
#else
#define TLLM_HALO_R_HOST_DEVICE
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace halo_r
{

//! Ranks are lanes of one warp and rank sets are 32-bit masks.
constexpr int kMaxRanks = 32;
//! Upper bound of the per-rank hottest-expert table that overloaded ranks offer.
constexpr int kMaxSendsPerRank = 16;
//! The helper index of a receiver is stored as two bits split over two rank masks.
constexpr int kMaxHelperSlots = 4;
//! Exported experts are ordered by an index below kOrderOpen, so at most this many experts are exported.
constexpr int kMaxBroadcasts = 128;
//! The plan uses the shared per-layer dynamic EPLB layout: six fields with one entry per possible expert.
constexpr int kPlanFieldStride = 384;
constexpr int kPlanFieldCount = 6;
constexpr int kPlanHeaderWords = 2;
constexpr int kPlanWords = kPlanHeaderWords + kPlanFieldCount * kPlanFieldStride;
//! Candidate keys carry a candidate order in their low byte: exported experts use [0, kOrderOpen) and newly opened
//! experts use kOrderOpen + owner rank. A larger key wins, so the order is stored inverted.
constexpr int kOrderBits = 8;
constexpr int kOrderOpen = 1 << (kOrderBits - 1);
constexpr int kOrderMax = (1 << kOrderBits) - 1;
//! The weighted row count of a candidate occupies the key bits above the order.
constexpr long long kMaxWeightedRows = 1LL << (32 - kOrderBits);
//! The room key keeps the lane in its low five bits.
constexpr int kLaneBits = 5;
//! Largest supported extra weight of a fan-out of an already exported expert over opening a new one.
constexpr int kMaxSendWeight = 32;
//! Each greedy step records one move and adds one receiver to an eligible rank, so a valid run makes at most
//! kMaxRanks * kMaxHelperSlots moves; the next step finds no candidate.
constexpr int kMaxPlannerIterations = kMaxRanks * kMaxHelperSlots + 1;

static_assert(kMaxRanks * kMaxHelperSlots <= kMaxBroadcasts, "every move must fit the move table");
static_assert(kMaxBroadcasts <= kOrderOpen, "exported experts must fit the order field");
static_assert(kMaxBroadcasts <= kPlanFieldStride, "the plan fields must hold every exported expert");

//! Plan fields after the [epoch, broadcast count] header. A broadcast b sends expert[b] from owner[b] to every rank in
//! destinationMask[b]; rank d stores it in helper slot (helperHighMask bit d) << 1 | (helperLowMask bit d).
enum class PlanField : int
{
    kExpert = 0,
    kOwner = 1,
    kDestinationMask = 2,
    kHelperLowMask = 3,
    kRows = 4,
    kHelperHighMask = 5,
};

enum class PlanStatus : int
{
    kOk = 0,
    kNoDestination = 1,
    kOverBudget = 2,
    kIterationCap = 3,
    kBadConfig = 4,
};

//! Shared scratch of the planner warp.
struct PlannerScratch
{
    int load[kMaxRanks];
    int received[kMaxRanks];
    int sent[kMaxRanks];
    int hottest[kMaxRanks * kMaxSendsPerRank];
    int exportExpert[kMaxBroadcasts];
    int exportRemaining[kMaxBroadcasts];
    int moveExpert[kMaxBroadcasts];
    int moveDestination[kMaxBroadcasts];
    int moveRows[kMaxBroadcasts];
    int broadcastExpert[kMaxBroadcasts];
    int hot[kMaxRanks * kMaxHelperSlots];
};

struct PlannerConfig
{
    int ep;
    int experts;
    int helperSlots;
    int sendCount;
    int sendWeight;
    int localRank;
};

struct PlannerOutputs
{
    int* plan;       //!< kPlanWords words; the planner writes plan[1] and every field, not the epoch plan[0].
    int* quota;      //!< [broadcast][rank] rows routed to each rank, kMaxBroadcasts * ep entries.
    int* outIds;     //!< This rank's helper slots: the expert held in each slot, or -1.
    int* outOwners;  //!< This rank's helper slots: the owner of that expert, or -1.
    int* hotTable;   //!< Optional [rank][helper slot] experts of every rank.
    int* loadAfter;  //!< Optional per-rank load after the moves.
    int* iterations; //!< Optional greedy step count.
};

TLLM_HALO_R_HOST_DEVICE inline int planMin(int a, int b)
{
    return a < b ? a : b;
}

TLLM_HALO_R_HOST_DEVICE inline int* planField(int* plan, PlanField field)
{
    return plan + kPlanHeaderWords + static_cast<int>(field) * kPlanFieldStride;
}

TLLM_HALO_R_HOST_DEVICE inline int const* planField(int const* plan, PlanField field)
{
    return plan + kPlanHeaderWords + static_cast<int>(field) * kPlanFieldStride;
}

TLLM_HALO_R_HOST_DEVICE inline bool validPlannerConfig(PlannerConfig const& cfg)
{
    return cfg.ep >= 1 && cfg.ep <= kMaxRanks && cfg.experts > 0 && cfg.experts % cfg.ep == 0
        && cfg.experts <= kPlanFieldStride && cfg.helperSlots >= 1 && cfg.helperSlots <= kMaxHelperSlots
        && cfg.sendCount >= 1 && cfg.sendCount <= kMaxSendsPerRank && cfg.ep * cfg.sendCount <= kMaxBroadcasts
        && cfg.sendCount <= cfg.experts / cfg.ep && cfg.sendWeight >= 0 && cfg.sendWeight <= kMaxSendWeight
        && cfg.localRank >= 0 && cfg.localRank < cfg.ep;
}

//! Helper slot of `rank` for a broadcast with the given helper masks.
TLLM_HALO_R_HOST_DEVICE inline int helperIndex(unsigned lowMask, unsigned highMask, int rank)
{
    return static_cast<int>((((highMask >> rank) & 1U) << 1) | ((lowMask >> rank) & 1U));
}

//! Physical slot of a selected route of `expert` that the route prefix assigned to `destination`: the owner's home
//! slot, or the destination's helper slot. Returns -1 when the destination is neither the owner nor a receiver.
TLLM_HALO_R_HOST_DEVICE inline int selectedRouteSlot(int expert, int destination, int owner, unsigned destinationMask,
    unsigned lowMask, unsigned highMask, int homeExperts, int localSlots)
{
    if (destination == owner)
    {
        return destination * localSlots + expert - owner * homeExperts;
    }
    if ((destinationMask >> destination) & 1U)
    {
        return destination * localSlots + homeExperts + helperIndex(lowMask, highMask, destination);
    }
    return -1;
}

//! Candidate key: weighted rows above the inverted candidate order.
TLLM_HALO_R_HOST_DEVICE inline unsigned candidateKey(int weightedRows, int order)
{
    return (static_cast<unsigned>(weightedRows) << kOrderBits) | static_cast<unsigned>(kOrderMax - order);
}

//! The two ranks with the most room below the target. Ties go to the lower rank.
struct ReceiverRooms
{
    int bestRank;
    int bestRoom;
    int secondRoom;
};

TLLM_HALO_R_HOST_DEVICE inline int roomExcluding(ReceiverRooms const& rooms, int rank)
{
    return rooms.bestRank == rank ? rooms.secondRoom : rooms.bestRoom;
}

//! The planner's view of one greedy step.
struct Move
{
    int expert;
    int owner;
    int rows;
    int exportIndex;
    bool opensExport;
};

//! Each rank's hottest home experts, hottest first, ties to the lower local expert. Returns the lane's load.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline int prepareRankLoads(int const* histogram, PlannerConfig const& cfg, PlannerScratch* s)
{
    int const lane = Warp::lane();
    int const home = cfg.experts / cfg.ep;
    int load = 0;
    if (lane < cfg.ep)
    {
        for (int local = 0; local < home; ++local)
        {
            load += histogram[lane * home + local];
        }
        for (int k = 0; k < cfg.sendCount; ++k)
        {
            int best = -1;
            int bestMass = -1;
            for (int local = 0; local < home; ++local)
            {
                bool used = false;
                for (int q = 0; q < k; ++q)
                {
                    used |= s->hottest[lane * kMaxSendsPerRank + q] == local;
                }
                int const mass = histogram[lane * home + local];
                if (!used && mass > bestMass)
                {
                    bestMass = mass;
                    best = local;
                }
            }
            s->hottest[lane * kMaxSendsPerRank + k] = best;
        }
        s->load[lane] = load;
        s->received[lane] = 0;
        s->sent[lane] = 0;
    }
    return load;
}

//! Warp collective: the best and second-best room among eligible ranks.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline ReceiverRooms findReceiverRooms(bool eligible, int room)
{
    int const lane = Warp::lane();
    unsigned const key = eligible ? (static_cast<unsigned>(room) << kLaneBits) | static_cast<unsigned>(31 - lane) : 0U;
    unsigned const best = Warp::reduceMax(key);
    int const bestRank = best ? 31 - static_cast<int>(best & 31U) : -1;
    unsigned const second = Warp::reduceMax(lane == bestRank ? 0U : key);
    return ReceiverRooms{bestRank, static_cast<int>(best >> kLaneBits), static_cast<int>(second >> kLaneBits)};
}

//! This lane's best candidate: fan-outs of exported experts (strided over lanes) and the next hottest expert of the
//! lane's own rank when it is still overloaded.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline unsigned bestLaneCandidate(int const* histogram, PlannerConfig const& cfg, int target,
    int exports, ReceiverRooms const& rooms, PlannerScratch const* s)
{
    int const lane = Warp::lane();
    int const home = cfg.experts / cfg.ep;
    unsigned best = 0;
    for (int j = lane; j < exports; j += 32)
    {
        int const owner = s->exportExpert[j] / home;
        int const remaining = s->exportRemaining[j];
        int const excess = s->load[owner] - target;
        if (remaining <= 0 || excess <= 0)
        {
            continue;
        }
        int const rows = planMin(planMin(roomExcluding(rooms, owner), remaining), excess);
        if (rows <= 0)
        {
            continue;
        }
        unsigned const key = candidateKey(rows * (1 + cfg.sendWeight), j);
        best = key > best ? key : best;
    }
    if (lane < cfg.ep)
    {
        int const excess = s->load[lane] - target;
        int const sent = s->sent[lane];
        if (excess > 0 && sent < cfg.sendCount)
        {
            int const mass = histogram[lane * home + s->hottest[lane * kMaxSendsPerRank + sent]];
            if (mass > 0)
            {
                int const rows = planMin(planMin(roomExcluding(rooms, lane), mass), excess);
                if (rows > 0)
                {
                    unsigned const key = candidateKey(rows, kOrderOpen + lane);
                    best = key > best ? key : best;
                }
            }
        }
    }
    return best;
}

TLLM_HALO_R_HOST_DEVICE inline Move decodeWinner(
    unsigned winner, PlannerConfig const& cfg, int exports, PlannerScratch const* s)
{
    int const home = cfg.experts / cfg.ep;
    int const order = kOrderMax - static_cast<int>(winner & static_cast<unsigned>(kOrderMax));
    unsigned const weightedRows = winner >> kOrderBits;
    if (order < kOrderOpen)
    {
        int const expert = s->exportExpert[order];
        return Move{expert, expert / home, static_cast<int>(weightedRows / static_cast<unsigned>(1 + cfg.sendWeight)),
            order, false};
    }
    int const owner = order - kOrderOpen;
    int const expert = owner * home + s->hottest[owner * kMaxSendsPerRank + s->sent[owner]];
    return Move{expert, owner, static_cast<int>(weightedRows), exports, true};
}

//! Greedy moves until no candidate remains. Returns the status; the export and move counts are written back.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline PlanStatus runGreedyMoves(int const* histogram, PlannerConfig const& cfg, int target,
    PlannerScratch* s, int* exportCount, int* moveCount, int* iterations)
{
    int const lane = Warp::lane();
    bool const active = lane < cfg.ep;
    int exports = 0;
    int moves = 0;
    int steps = 0;
    PlanStatus status = PlanStatus::kOk;
    for (;;)
    {
        if (++steps > kMaxPlannerIterations)
        {
            status = PlanStatus::kIterationCap;
            break;
        }
        int const room = target - (active ? s->load[lane] : 0);
        bool const eligible = active && s->received[lane] < cfg.helperSlots && room > 0;
        ReceiverRooms const rooms = findReceiverRooms<Warp>(eligible, room);
        unsigned const winner = Warp::reduceMax(bestLaneCandidate<Warp>(histogram, cfg, target, exports, rooms, s));
        if (!winner)
        {
            break;
        }
        Move const move = decodeWinner(winner, cfg, exports, s);
        unsigned const receivers = Warp::ballot(eligible && lane != move.owner && room >= move.rows);
        if (!receivers)
        {
            status = PlanStatus::kNoDestination;
            break;
        }
        int const destination = static_cast<int>(Warp::findFirstSet(receivers)) - 1;
        Warp::sync();
        if (lane == 0)
        {
            if (move.opensExport)
            {
                s->sent[move.owner] += 1;
                s->exportExpert[move.exportIndex] = move.expert;
                s->exportRemaining[move.exportIndex] = histogram[move.expert];
            }
            s->load[move.owner] -= move.rows;
            s->load[destination] += move.rows;
            s->received[destination] += 1;
            s->exportRemaining[move.exportIndex] -= move.rows;
            s->moveExpert[moves] = move.expert;
            s->moveDestination[moves] = destination;
            s->moveRows[moves] = move.rows;
        }
        exports += move.opensExport ? 1 : 0;
        moves += 1;
        Warp::sync();
    }
    *exportCount = exports;
    *moveCount = moves;
    *iterations = steps;
    return status;
}

//! Each rank's distinct received experts in ascending order, one per helper slot. Returns kOverBudget on any rank
//! that would receive more distinct experts than it has helper slots.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline PlanStatus buildHelperTable(PlannerConfig const& cfg, int moves, PlannerScratch* s)
{
    int const lane = Warp::lane();
    bool overBudget = false;
    if (lane < cfg.ep)
    {
        int received[kMaxHelperSlots];
        int used = 0;
        for (int m = 0; m < moves; ++m)
        {
            if (s->moveDestination[m] != lane)
            {
                continue;
            }
            int const expert = s->moveExpert[m];
            bool duplicate = false;
            for (int q = 0; q < used; ++q)
            {
                duplicate |= received[q] == expert;
            }
            if (duplicate)
            {
                continue;
            }
            if (used >= cfg.helperSlots)
            {
                overBudget = true;
                break;
            }
            received[used++] = expert;
        }
        if (!overBudget)
        {
            for (int a = 1; a < used; ++a)
            {
                for (int q = a; q > 0 && received[q] < received[q - 1]; --q)
                {
                    int const t = received[q];
                    received[q] = received[q - 1];
                    received[q - 1] = t;
                }
            }
            for (int q = 0; q < cfg.helperSlots; ++q)
            {
                s->hot[lane * kMaxHelperSlots + q] = q < used ? received[q] : -1;
            }
        }
    }
    return Warp::ballot(overBudget) ? PlanStatus::kOverBudget : PlanStatus::kOk;
}

//! Distinct moved experts in ascending order, one broadcast each. Returns the broadcast count.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline int collectBroadcastExperts(int moves, PlannerScratch* s)
{
    int const lane = Warp::lane();
    int broadcasts = 0;
    for (int base = 0; base < kMaxBroadcasts; base += 32)
    {
        int const m = base + lane;
        bool first = false;
        int rank = 0;
        if (m < moves)
        {
            int const expert = s->moveExpert[m];
            first = true;
            for (int q = 0; q < m; ++q)
            {
                first &= s->moveExpert[q] != expert;
            }
            if (first)
            {
                for (int q = 0; q < moves; ++q)
                {
                    bool firstOfItsExpert = true;
                    for (int t = 0; t < q; ++t)
                    {
                        firstOfItsExpert &= s->moveExpert[t] != s->moveExpert[q];
                    }
                    rank += (firstOfItsExpert && s->moveExpert[q] < expert) ? 1 : 0;
                }
            }
        }
        if (first)
        {
            s->broadcastExpert[rank] = s->moveExpert[m];
        }
        broadcasts += static_cast<int>(Warp::popCount(Warp::ballot(first)));
    }
    Warp::sync();
    return broadcasts;
}

//! Plan fields and route quotas of every broadcast. Fields beyond the broadcast count are zero.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline void writeBroadcastTables(int const* histogram, PlannerConfig const& cfg, int moves,
    int broadcasts, PlannerScratch const* s, int* plan, int* quota)
{
    int const lane = Warp::lane();
    int const home = cfg.experts / cfg.ep;
    for (int w = lane; w < kPlanFieldCount * kPlanFieldStride; w += 32)
    {
        plan[kPlanHeaderWords + w] = 0;
    }
    for (int w = lane; w < broadcasts * cfg.ep; w += 32)
    {
        quota[w] = 0;
    }
    Warp::sync();
    for (int b = lane; b < broadcasts; b += 32)
    {
        int const expert = s->broadcastExpert[b];
        int const owner = expert / home;
        unsigned destinationMask = 0;
        unsigned lowMask = 0;
        unsigned highMask = 0;
        int rows = 0;
        for (int m = 0; m < moves; ++m)
        {
            if (s->moveExpert[m] != expert)
            {
                continue;
            }
            int const destination = s->moveDestination[m];
            destinationMask |= 1U << destination;
            quota[b * cfg.ep + destination] += s->moveRows[m];
            rows += s->moveRows[m];
        }
        for (int destination = 0; destination < cfg.ep; ++destination)
        {
            if (!((destinationMask >> destination) & 1U))
            {
                continue;
            }
            int slot = 0;
            while (slot < cfg.helperSlots && s->hot[destination * kMaxHelperSlots + slot] != expert)
            {
                ++slot;
            }
            lowMask |= static_cast<unsigned>(slot & 1) << destination;
            highMask |= static_cast<unsigned>((slot >> 1) & 1) << destination;
        }
        quota[b * cfg.ep + owner] = histogram[expert] - rows;
        planField(plan, PlanField::kExpert)[b] = expert;
        planField(plan, PlanField::kOwner)[b] = owner;
        planField(plan, PlanField::kDestinationMask)[b] = static_cast<int>(destinationMask);
        planField(plan, PlanField::kHelperLowMask)[b] = static_cast<int>(lowMask);
        planField(plan, PlanField::kRows)[b] = rows;
        planField(plan, PlanField::kHelperHighMask)[b] = static_cast<int>(highMask);
    }
    if (lane == 0)
    {
        plan[1] = broadcasts;
    }
}

//! Plan one layer from the global routed-token histogram. Called by all 32 lanes of one warp; returns a warp-uniform
//! status. On a non-OK status the plan and outputs are unspecified and the caller must keep every route home.
template <typename Warp>
TLLM_HALO_R_HOST_DEVICE inline PlanStatus planHaloR(
    int const* histogram, PlannerConfig const& cfg, PlannerScratch* s, PlannerOutputs const& out)
{
    if (!validPlannerConfig(cfg))
    {
        return PlanStatus::kBadConfig;
    }
    int const lane = Warp::lane();
    int const myLoad = prepareRankLoads<Warp>(histogram, cfg, s);
    int const total = Warp::reduceAdd(lane < cfg.ep ? myLoad : 0);
    int const target = (total + cfg.ep - 1) / cfg.ep;
    Warp::sync();

    int exports = 0;
    int moves = 0;
    int iterations = 0;
    PlanStatus status = runGreedyMoves<Warp>(histogram, cfg, target, s, &exports, &moves, &iterations);
    if (out.iterations != nullptr && lane == 0)
    {
        *out.iterations = iterations;
    }
    if (status != PlanStatus::kOk)
    {
        return status;
    }
    status = buildHelperTable<Warp>(cfg, moves, s);
    if (status != PlanStatus::kOk)
    {
        return status;
    }
    int const broadcasts = collectBroadcastExperts<Warp>(moves, s);
    writeBroadcastTables<Warp>(histogram, cfg, moves, broadcasts, s, out.plan, out.quota);

    int const home = cfg.experts / cfg.ep;
    if (lane < cfg.ep)
    {
        if (out.hotTable != nullptr)
        {
            for (int q = 0; q < cfg.helperSlots; ++q)
            {
                out.hotTable[lane * cfg.helperSlots + q] = s->hot[lane * kMaxHelperSlots + q];
            }
        }
        if (out.loadAfter != nullptr)
        {
            out.loadAfter[lane] = s->load[lane];
        }
    }
    if (lane < cfg.helperSlots)
    {
        int const expert = s->hot[cfg.localRank * kMaxHelperSlots + lane];
        out.outIds[lane] = expert;
        out.outOwners[lane] = expert >= 0 ? expert / home : -1;
    }
    Warp::sync();
    return PlanStatus::kOk;
}

} // namespace halo_r
} // namespace kernels

TRTLLM_NAMESPACE_END
