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

// Per-copy shares of one MoE layer for the adaptive expert load balancer (CPU op).
//
// Given the rows every expert receives per decode step and a slot layout, the op splits the rows of every expert with
// copies on two or more GPUs among those GPUs so that the largest number of rows on any GPU is smallest. It bisects on
// that target, a max flow deciding whether a target is feasible. A lower bound of the smallest feasible target, found
// with Newton's method, decides the bisection steps away from it without a max flow; when the bound does not converge,
// every step runs the max flow. Copies of an expert on one GPU share its rows there evenly; an expert without rows or
// on one GPU keeps the even split over its copies.
//
// With the costs of the decision's time model, the op minimizes the busiest GPU's modeled time instead: the max flow
// takes every GPU's rows held alone as their modeled time over the time of one more row, plus the read of every copy
// of a replicated expert there; then, while the busiest GPU can shed time, a replicated expert's rows on it move to
// another GPU holding it, all of them (its copies stop reading weights) or those above its copies' last full tile,
// whichever move leaves the larger of the two GPUs' times smallest, as long as that is below the busiest GPU's time.
//
// Every floating-point operation runs in a fixed order, so every EP rank, running the same binary on the same rows,
// computes the same shares.

#include "tensorrt_llm/common/config.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr double kFlowNeedRel = 1e-12;   // Relative shortfall of a max flow still counted as feasible.
constexpr double kFlowEpsRel = 1e-13;    // Residual capacity at or below this fraction of the rows counts as none.
constexpr double kBalanceTolRel = 1e-9;  // Relative bisection tolerance on the target rows.
constexpr double kBalanceBandRel = 1e-9; // Band around the Newton bound in which max flows decide a bisection step.
constexpr int kBalanceMaxNewton = 64;    // Newton steps before the bisection runs without a bound.
constexpr double kInfinity = std::numeric_limits<double>::infinity();

//! The largest value, the first one on ties.
double maxOf(std::vector<double> const& values)
{
    double top = values[0];
    for (size_t r = 1; r < values.size(); ++r)
    {
        if (values[r] > top)
        {
            top = values[r];
        }
    }
    return top;
}

//! Sum by Neumaier's compensated summation, the compensation added once at the end (the first term added to 0.0,
//! a non-finite compensation dropped).
double compensatedSum(std::vector<double> const& values)
{
    if (values.empty())
    {
        return 0.0;
    }
    double total = 0.0 + values[0];
    double comp = 0.0;
    for (size_t i = 1; i < values.size(); ++i)
    {
        double const x = values[i];
        double const t = total + x;
        if (std::fabs(total) >= std::fabs(x))
        {
            comp += (total - t) + x;
        }
        else
        {
            comp += (x - t) + total;
        }
        total = t;
    }
    if (comp != 0.0 && std::isfinite(comp))
    {
        total += comp;
    }
    return total;
}

//! A directed graph whose edge k ^ 1 is the reverse of edge k. Edges are listed per node in creation order, which
//! fixes the order in which the max flow explores them.
class FlowGraph
{
public:
    explicit FlowGraph(int numNodes)
        : mNumNodes(numNodes)
    {
    }

    //! Adds u -> v with capacity cap and its reverse without capacity; returns the forward edge.
    int addEdge(int u, int v, double cap)
    {
        int const k = static_cast<int>(mTo.size());
        mTo.push_back(v);
        mCap0.push_back(cap);
        mOwner.push_back(u);
        mTo.push_back(u);
        mCap0.push_back(0.0);
        mOwner.push_back(v);
        return k;
    }

    //! Builds the per-node edge lists; called once after the last edge.
    void finalize()
    {
        mPtr.assign(mNumNodes + 1, 0);
        for (int const u : mOwner)
        {
            ++mPtr[u + 1];
        }
        for (int u = 0; u < mNumNodes; ++u)
        {
            mPtr[u + 1] += mPtr[u];
        }
        std::vector<int> fill(mPtr.begin(), mPtr.end() - 1);
        mIdx.assign(mOwner.size(), 0);
        for (int k = 0; k < static_cast<int>(mOwner.size()); ++k)
        {
            mIdx[fill[mOwner[k]]++] = k;
        }
        mLevel.assign(mNumNodes, -1);
        mCurrent.assign(mNumNodes, 0);
        mQueue.assign(mNumNodes, 0);
    }

    std::vector<double> const& initialCapacities() const
    {
        return mCap0;
    }

    //! Dinic's max flow from src to snk; cap is updated to the residual capacities. BFS levels over edges with residual
    //! capacity above eps, then depth-first augmenting paths with a current-edge pointer per node until one carries no
    //! more than eps. Afterwards reachable(u) tells whether u is reachable from src in the residual graph.
    double maxFlow(std::vector<double>& cap, int src, int snk, double eps)
    {
        double flow = 0.0;
        while (true)
        {
            std::fill(mLevel.begin(), mLevel.end(), -1);
            mLevel[src] = 0;
            int head = 0;
            int tail = 0;
            mQueue[tail++] = src;
            while (head < tail)
            {
                int const u = mQueue[head++];
                for (int k = mPtr[u]; k < mPtr[u + 1]; ++k)
                {
                    int const e = mIdx[k];
                    int const v = mTo[e];
                    if (mLevel[v] < 0 && cap[e] > eps)
                    {
                        mLevel[v] = mLevel[u] + 1;
                        mQueue[tail++] = v;
                    }
                }
            }
            if (mLevel[snk] < 0)
            {
                return flow;
            }
            std::fill(mCurrent.begin(), mCurrent.end(), 0);
            while (true)
            {
                double const f = augment(cap, src, kInfinity, snk, eps);
                if (f <= eps)
                {
                    break;
                }
                flow += f;
            }
        }
    }

    bool reachable(int u) const
    {
        return mLevel[u] >= 0;
    }

private:
    double augment(std::vector<double>& cap, int u, double f, int snk, double eps)
    {
        if (u == snk)
        {
            return f;
        }
        int const base = mPtr[u];
        int const degree = mPtr[u + 1] - base;
        while (mCurrent[u] < degree)
        {
            int const e = mIdx[base + mCurrent[u]];
            int const v = mTo[e];
            if (cap[e] > eps && mLevel[v] == mLevel[u] + 1)
            {
                double const c = cap[e];
                double const d = augment(cap, v, c < f ? c : f, snk, eps);
                if (d > eps)
                {
                    cap[e] -= d;
                    cap[e ^ 1] += d;
                    return d;
                }
            }
            ++mCurrent[u];
        }
        return 0.0;
    }

    int mNumNodes;
    std::vector<int> mTo;
    std::vector<double> mCap0;
    std::vector<int> mOwner;
    std::vector<int> mPtr;
    std::vector<int> mIdx;
    std::vector<int> mLevel;
    std::vector<int> mCurrent;
    std::vector<int> mQueue;
};

//! Bisection on the largest rows per GPU, solved as a max flow. With m replicated experts and R GPUs, node 0 is the
//! source, node 1 + i the replicated expert i, node 1 + m + r GPU r and node m + R + 1 the sink. A target T gives
//! GPU r the capacity max(T - fixed[r], 0), where fixed[r] are the rows of the experts that only GPU r holds; T is
//! feasible when the max flow carries all rows of the replicated experts.
class RowsBalance
{
public:
    RowsBalance(std::vector<double> const& rows, std::vector<std::vector<int>> const& ranks,
        std::vector<double> const& fixed, double total)
        : mRows(rows)
        , mFixed(fixed)
        , mNumExperts(static_cast<int>(rows.size()))
        , mSnk(static_cast<int>(rows.size() + fixed.size()) + 1)
        , mGraph(static_cast<int>(rows.size() + fixed.size()) + 2)
        , mNeed(compensatedSum(rows) * (1.0 - kFlowNeedRel))
        , mEps(kFlowEpsRel * total)
    {
        double const big = 2.0 * total;
        for (int i = 0; i < mNumExperts; ++i)
        {
            mGraph.addEdge(kSrc, 1 + i, rows[i]);
            std::vector<int> edges;
            for (int const r : ranks[i])
            {
                edges.push_back(mGraph.addEdge(1 + i, 1 + mNumExperts + r, big));
            }
            mMidEdges.push_back(std::move(edges));
        }
        for (int r = 0; r < static_cast<int>(fixed.size()); ++r)
        {
            mRankEdges.push_back(mGraph.addEdge(1 + mNumExperts + r, mSnk, 0.0));
        }
        mGraph.finalize();
    }

    //! Rows of every replicated expert on each of its GPUs; false when no target is feasible.
    bool solve(double lo, double hi, std::vector<std::vector<double>>& perRank)
    {
        std::vector<double> best;
        double bound = 0.0;
        Outcome outcome = newtonBound(lo, bound) ? bisect(lo, hi, true, bound, best) : Outcome::kFinalInfeasible;
        if (outcome == Outcome::kFinalInfeasible)
        {
            outcome = bisect(lo, hi, false, 0.0, best);
        }
        if (outcome != Outcome::kFound)
        {
            return false;
        }
        perRank.clear();
        for (auto const& edges : mMidEdges)
        {
            std::vector<double> x;
            for (int const k : edges)
            {
                x.push_back(best[k ^ 1]);
            }
            perRank.push_back(std::move(x));
        }
        return true;
    }

private:
    enum class Outcome
    {
        kFound,
        kNotFound,
        kFinalInfeasible,
    };

    static constexpr int kSrc = 0;

    bool flowAt(double target, std::vector<double>& cap)
    {
        cap = mGraph.initialCapacities();
        for (int r = 0; r < static_cast<int>(mRankEdges.size()); ++r)
        {
            double const x = target - mFixed[r];
            cap[mRankEdges[r]] = x > 0.0 ? x : 0.0;
        }
        return mGraph.maxFlow(cap, kSrc, mSnk, mEps) >= mNeed;
    }

    //! A lower bound of the optimum close to it, from lo: while the target is infeasible, the GPUs reachable from the
    //! source in the residual graph form a set whose rows per GPU, the next target, bound the optimum from below.
    //! False when the steps do not settle.
    bool newtonBound(double lo, double& bound)
    {
        std::vector<double> cap;
        double target = lo;
        int steps = 0;
        while (true)
        {
            if (flowAt(target, cap))
            {
                bound = target;
                return true;
            }
            if (++steps > kBalanceMaxNewton)
            {
                return false;
            }
            double num = 0.0;
            int count = 0;
            for (int r = 0; r < static_cast<int>(mFixed.size()); ++r)
            {
                if (mGraph.reachable(1 + mNumExperts + r))
                {
                    num += mFixed[r];
                    ++count;
                }
            }
            for (int i = 0; i < mNumExperts; ++i)
            {
                if (mGraph.reachable(1 + i))
                {
                    num += mRows[i];
                }
            }
            double const next = count > 0 ? num / count : target;
            if (!(next > target))
            {
                return false;
            }
            target = next;
        }
    }

    //! Bisection on the target; best gets the residual capacities at the final target. With a Newton bound, a step
    //! whose target lies outside a narrow band around it is decided by comparison and only the steps inside the band
    //! and the final target run max flows; without one every step runs one.
    Outcome bisect(double lo, double hi, bool guided, double bound, std::vector<double>& best)
    {
        double const below = guided ? bound * (1.0 - kBalanceBandRel) : -kInfinity;
        double const above = guided ? bound * (1.0 + kBalanceBandRel) : kInfinity;
        std::vector<double> cap;
        bool found = false;
        bool have = false;
        while (hi - lo > kBalanceTolRel * hi)
        {
            double const mid = 0.5 * (lo + hi);
            if (mid < below)
            {
                lo = mid;
            }
            else if (mid > above)
            {
                hi = mid;
                found = true;
                have = false;
            }
            else if (flowAt(mid, cap))
            {
                hi = mid;
                best = cap;
                found = true;
                have = true;
            }
            else
            {
                lo = mid;
            }
        }
        if (found && !have)
        {
            if (!flowAt(hi, cap))
            {
                return Outcome::kFinalInfeasible;
            }
            best = cap;
        }
        return found ? Outcome::kFound : Outcome::kNotFound;
    }

    std::vector<double> const& mRows;
    std::vector<double> const& mFixed;
    int mNumExperts;
    int mSnk;
    FlowGraph mGraph;
    double mNeed;
    double mEps;
    std::vector<std::vector<int>> mMidEdges;
    std::vector<int> mRankEdges;
};

//! The shares of one layer (see the file comment); slotsOf lists every expert's slots in ascending order. offset adds
//! its positive entries to the rows every GPU holds alone (none when empty).
std::vector<double> rowsBalancedShares(std::vector<double> const& load, std::vector<std::vector<int>> const& slotsOf,
    int numRanks, int localSlots, int numSlots, std::vector<double> const& offset)
{
    std::vector<double> shares(numSlots, 0.0);
    std::vector<double> fixed(numRanks, 0.0);
    std::vector<double> rows;
    std::vector<std::vector<int>> ranks;
    std::vector<int> experts;
    for (int e = 0; e < static_cast<int>(slotsOf.size()); ++e)
    {
        auto const& sl = slotsOf[e];
        std::vector<int> held;
        for (int const s : sl)
        {
            if (held.empty() || held.back() != s / localSlots)
            {
                held.push_back(s / localSlots);
            }
        }
        double const n = load[e];
        if (n <= 0.0 || held.size() == 1)
        {
            for (int const s : sl)
            {
                shares[s] = 1.0 / static_cast<double>(sl.size());
            }
            if (n > 0.0)
            {
                fixed[held[0]] += n;
            }
            continue;
        }
        rows.push_back(n);
        ranks.push_back(std::move(held));
        experts.push_back(e);
    }
    for (size_t r = 0; r < offset.size(); ++r)
    {
        fixed[r] += offset[r] > 0.0 ? offset[r] : 0.0;
    }
    if (rows.empty())
    {
        return shares;
    }
    std::vector<double> even(fixed);
    for (size_t i = 0; i < experts.size(); ++i)
    {
        auto const& sl = slotsOf[experts[i]];
        for (int const s : sl)
        {
            even[s / localSlots] += rows[i] / static_cast<double>(sl.size());
        }
    }
    double const total = compensatedSum(fixed) + compensatedSum(rows);
    double const maxFixed = maxOf(fixed);
    double const perRankTotal = total / numRanks;
    double const lo = perRankTotal > maxFixed ? perRankTotal : maxFixed;
    double const hi = maxOf(even);
    std::vector<std::vector<double>> perRank;
    bool const solved = RowsBalance(rows, ranks, fixed, total).solve(lo, hi, perRank);
    for (size_t i = 0; i < experts.size(); ++i)
    {
        auto const& sl = slotsOf[experts[i]];
        if (!solved)
        {
            for (int const s : sl)
            {
                shares[s] = 1.0 / static_cast<double>(sl.size());
            }
            continue;
        }
        auto const& x = perRank[i];
        double const tot = compensatedSum(x);
        double const n = rows[i];
        for (int const s : sl)
        {
            int const r = s / localSlots;
            int copies = 0;
            for (int const t : sl)
            {
                copies += t / localSlots == r ? 1 : 0;
            }
            int const j = static_cast<int>(std::find(ranks[i].begin(), ranks[i].end(), r) - ranks[i].begin());
            double const planned
                = (tot > 0 ? n * x[j] / tot : n / static_cast<double>(ranks[i].size())) / static_cast<double>(copies);
            shares[s] = planned / n;
        }
    }
    return shares;
}

//! The time model's costs in us: a copy's weight read, a tile beyond its first, a row, a tile beyond the first of the
//! GPU's busiest copy; and the rows of a tile. The model's scale multiplies every GPU alike, so it is left out.
struct TimeCosts
{
    double read;
    double tile;
    double row;
    double hot;
    double tileRows;
};

//! Rows of an expert on one GPU, split evenly over its copies there.
struct Holding
{
    int rank;
    int expert;
    int copies;
    double rows;
};

//! Modeled time of a GPU's holdings (indices into all), the rows of holding changed plus delta.
double gpuTime(std::vector<Holding> const& all, std::vector<int> const& mine, TimeCosts const& c, int changed = -1,
    double delta = 0.0)
{
    double t = 0.0;
    double top = 0.0;
    for (int const h : mine)
    {
        double const y = h == changed ? all[h].rows + delta : all[h].rows;
        if (!(y > 0.0))
        {
            continue;
        }
        double const k = all[h].copies;
        double const q = std::ceil(y / k / c.tileRows);
        t += c.read * k + c.tile * k * std::max(q - 1.0, 0.0) + c.row * y;
        top = std::max(top, q);
    }
    return t + c.hot * std::max(top - 1.0, 0.0);
}

//! Holdings of every expert with rows on every GPU, in slot order, and the indices per GPU and per expert.
std::vector<Holding> holdingsOf(std::vector<double> const& load, int64_t const* layout, int numSlots, int localSlots,
    std::vector<double> const& shares, std::vector<std::vector<int>>& perRank, std::vector<std::vector<int>>& perExpert)
{
    std::vector<Holding> all;
    for (int s = 0; s < numSlots; ++s)
    {
        int const e = static_cast<int>(layout[s]);
        if (e < 0)
        {
            continue;
        }
        int const r = s / localSlots;
        auto const& mine = perExpert[e];
        auto it = std::find_if(mine.begin(), mine.end(), [&](int h) { return all[h].rank == r; });
        int h = it == mine.end() ? -1 : *it;
        if (h < 0)
        {
            h = static_cast<int>(all.size());
            all.push_back({r, e, 0, 0.0});
            perRank[r].push_back(h);
            perExpert[e].push_back(h);
        }
        all[h].copies += 1;
        all[h].rows += load[e] * shares[s];
    }
    return all;
}

//! Per GPU, its rows held alone as modeled time, with the read of each copy of a replicated expert with rows, over the
//! time of one more row, minus those rows; less the smallest of them, as a common offset changes no split.
std::vector<double> timeOffsets(std::vector<double> const& load, int64_t const* layout, int numSlots, int numRanks,
    int localSlots, std::vector<std::vector<int>> const& slotsOf, TimeCosts const& c)
{
    std::vector<Holding> alone;
    std::vector<std::vector<int>> mine(numRanks);
    std::vector<int> copies(numRanks, 0);
    std::vector<double> fixedRows(numRanks, 0.0);
    for (int s = 0; s < numSlots; ++s)
    {
        int const e = static_cast<int>(layout[s]);
        if (e < 0 || !(load[e] > 0.0))
        {
            continue;
        }
        int const r = s / localSlots;
        auto const& sl = slotsOf[e];
        if (sl.front() / localSlots != sl.back() / localSlots)
        {
            copies[r] += 1;
        }
        else if (sl.front() == s)
        {
            mine[r].push_back(static_cast<int>(alone.size()));
            alone.push_back({r, e, static_cast<int>(sl.size()), load[e]});
            fixedRows[r] += load[e];
        }
    }
    double const slope = c.row + c.tile / c.tileRows;
    std::vector<double> offset(numRanks);
    for (int r = 0; r < numRanks; ++r)
    {
        offset[r] = gpuTime(alone, mine[r], c) / slope + c.read * copies[r] / slope - fixedRows[r];
    }
    double const low = *std::min_element(offset.begin(), offset.end());
    for (double& x : offset)
    {
        x -= low;
    }
    return offset;
}

//! Moves rows of replicated experts off the busiest GPU while that lowers its modeled time (see the file comment).
void shedBusiest(std::vector<double> const& load, int64_t const* layout, int numSlots, int numRanks, int localSlots,
    TimeCosts const& c, std::vector<double>& shares)
{
    std::vector<std::vector<int>> perRank(numRanks);
    std::vector<std::vector<int>> perExpert(load.size());
    std::vector<Holding> all = holdingsOf(load, layout, numSlots, localSlots, shares, perRank, perExpert);
    std::vector<double> times(numRanks);
    for (int r = 0; r < numRanks; ++r)
    {
        times[r] = gpuTime(all, perRank[r], c);
    }
    auto movable = [&](int e) { return perExpert[e].size() > 1 && load[e] > 0.0; };
    while (true)
    {
        int const a = static_cast<int>(std::max_element(times.begin(), times.end()) - times.begin());
        double const ta = times[a];
        double best = kInfinity;
        int from = -1;
        int to = -1;
        double moved = 0.0;
        double timeFrom = 0.0;
        double timeTo = 0.0;
        for (int const h : perRank[a])
        {
            double const y = all[h].rows;
            if (!movable(all[h].expert) || !(y > 0.0))
            {
                continue;
            }
            double const k = all[h].copies;
            double const q = std::ceil(y / k / c.tileRows);
            double const amounts[2] = {y, (y / k - (q - 1.0) * c.tileRows) * k};
            for (int i = 0; i < (q >= 2.0 ? 2 : 1); ++i)
            {
                double const ta2 = gpuTime(all, perRank[a], c, h, -amounts[i]);
                if (!(ta2 < ta))
                {
                    continue;
                }
                for (int const g : perExpert[all[h].expert])
                {
                    if (all[g].rank == a)
                    {
                        continue;
                    }
                    double const tb2 = gpuTime(all, perRank[all[g].rank], c, g, amounts[i]);
                    double const v = std::max(ta2, tb2);
                    if (v < ta && v < best)
                    {
                        best = v;
                        from = h;
                        to = g;
                        moved = amounts[i];
                        timeFrom = ta2;
                        timeTo = tb2;
                    }
                }
            }
        }
        if (from < 0)
        {
            break;
        }
        all[from].rows = std::max(all[from].rows - moved, 0.0);
        all[to].rows += moved;
        times[a] = timeFrom;
        times[all[to].rank] = timeTo;
    }
    for (int s = 0; s < numSlots; ++s)
    {
        int const e = static_cast<int>(layout[s]);
        if (e >= 0 && movable(e))
        {
            auto const& mine = perExpert[e];
            int const h = *std::find_if(mine.begin(), mine.end(), [&](int i) { return all[i].rank == s / localSlots; });
            shares[s] = all[h].rows / all[h].copies / load[e];
        }
    }
}

} // namespace

//! Shares of every slot's expert rows (see the file comment). loadRows: [num_experts] float64 rows per step of every
//! expert, rows <= 0 counting as none; layout: [num_slots] int64 expert of every slot, -1 when free, rank-major with
//! num_slots / ep_size slots per rank. Returns [num_slots] float64: the fraction of its expert's rows every slot
//! receives, 0 on a free slot. costs: the time model's (read, tile, row, hot tile in us, scale, tile rows) to minimize
//! the busiest GPU's modeled time instead of its rows. Raises ValueError on a load that is not finite, an expert index
//! outside [-1, num_experts), an expert with rows but no slot, or costs that are not six finite values, none negative
//! and the scale and the tile rows positive.
at::Tensor adaptiveEplbSolveShares(
    at::Tensor const& loadRows, at::Tensor const& layout, int64_t epSize, c10::optional<at::ArrayRef<double>> costs)
{
    TORCH_CHECK(loadRows.device().is_cpu() && layout.device().is_cpu(), "adaptive_eplb_solve_shares takes CPU tensors");
    TORCH_CHECK(loadRows.scalar_type() == at::kDouble && loadRows.dim() == 1, "load_rows must be a 1-D float64 tensor");
    TORCH_CHECK(layout.scalar_type() == at::kLong && layout.dim() == 1, "layout must be a 1-D int64 tensor");
    TORCH_CHECK(epSize > 0 && layout.size(0) % epSize == 0, "the slots must divide evenly over ep_size ranks");
    at::Tensor const loadC = loadRows.contiguous();
    at::Tensor const layoutC = layout.contiguous();
    int64_t const numExperts = loadC.size(0);
    int64_t const numSlots = layoutC.size(0);
    TORCH_CHECK_VALUE(numExperts > 0 && numSlots > 0, "the layer needs experts and slots");
    double const* loadPtr = loadC.data_ptr<double>();
    std::vector<double> const load(loadPtr, loadPtr + numExperts);
    for (double const v : load)
    {
        TORCH_CHECK_VALUE(std::isfinite(v), "load_rows must be finite");
    }
    int64_t const* layoutPtr = layoutC.data_ptr<int64_t>();
    std::vector<std::vector<int>> slotsOf(numExperts);
    for (int64_t s = 0; s < numSlots; ++s)
    {
        int64_t const e = layoutPtr[s];
        TORCH_CHECK_VALUE(e >= -1 && e < numExperts, "layout holds expert ", e, " outside [-1, ", numExperts, ")");
        if (e >= 0)
        {
            slotsOf[e].push_back(static_cast<int>(s));
        }
    }
    for (int64_t e = 0; e < numExperts; ++e)
    {
        TORCH_CHECK_VALUE(!(load[e] > 0.0) || !slotsOf[e].empty(), "expert ", e, " has rows but no slot");
    }
    int const ranks = static_cast<int>(epSize);
    int const local = static_cast<int>(numSlots / epSize);
    int const n = static_cast<int>(numSlots);
    std::vector<double> offset;
    TimeCosts c{};
    if (costs.has_value())
    {
        auto const& v = costs.value();
        bool const valid
            = v.size() == 6 && std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x) && x >= 0.0; });
        TORCH_CHECK_VALUE(valid && v[4] > 0.0 && v[5] > 0.0,
            "costs must be six finite values, none negative, the scale and the tile rows positive");
        c = TimeCosts{v[0], v[1], v[2], v[3], v[5]}; // the scale multiplies every GPU alike
        offset = timeOffsets(load, layoutPtr, n, ranks, local, slotsOf, c);
    }
    std::vector<double> shares = rowsBalancedShares(load, slotsOf, ranks, local, n, offset);
    if (costs.has_value())
    {
        shedBusiest(load, layoutPtr, n, ranks, local, c, shares);
    }
    at::Tensor out = at::empty({numSlots}, at::TensorOptions().dtype(at::kDouble));
    std::copy(shares.begin(), shares.end(), out.data_ptr<double>());
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("adaptive_eplb_solve_shares(Tensor load_rows, Tensor layout, int ep_size, float[]? costs=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CPU, m)
{
    m.impl("adaptive_eplb_solve_shares", &tensorrt_llm::torch_ext::adaptiveEplbSolveShares);
}
