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

//! The shares of one layer (see the file comment); slotsOf lists every expert's slots in ascending order.
std::vector<double> rowsBalancedShares(std::vector<double> const& load, std::vector<std::vector<int>> const& slotsOf,
    int numRanks, int localSlots, int numSlots)
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

} // namespace

//! Shares of every slot's expert rows (see the file comment). loadRows: [num_experts] float64 rows per step of every
//! expert, rows <= 0 counting as none; layout: [num_slots] int64 expert of every slot, -1 when free, rank-major with
//! num_slots / ep_size slots per rank. Returns [num_slots] float64: the fraction of its expert's rows every slot
//! receives, 0 on a free slot. Raises ValueError on a load that is not finite, an expert index outside
//! [-1, num_experts) or an expert with rows but no slot.
at::Tensor adaptiveEplbSolveShares(at::Tensor const& loadRows, at::Tensor const& layout, int64_t epSize)
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
    std::vector<double> const shares = rowsBalancedShares(
        load, slotsOf, static_cast<int>(epSize), static_cast<int>(numSlots / epSize), static_cast<int>(numSlots));
    at::Tensor out = at::empty({numSlots}, at::TensorOptions().dtype(at::kDouble));
    std::copy(shares.begin(), shares.end(), out.data_ptr<double>());
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("adaptive_eplb_solve_shares(Tensor load_rows, Tensor layout, int ep_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CPU, m)
{
    m.impl("adaptive_eplb_solve_shares", &tensorrt_llm::torch_ext::adaptiveEplbSolveShares);
}
