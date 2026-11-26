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

// Host-side transitions of the MiniMax-M3 family's draft prob1 agreement and exchange protocols
// (tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/m31/prob1Agreement.h, prob1Exchange.h), driven by a
// sequential model of the ranks.

#include "minimax_m3/csrc/m31/prob1Agreement.h"
#include "minimax_m3/csrc/m31/prob1Exchange.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <stdexcept>
#include <vector>

namespace
{

using namespace TRTLLM_MV2_NS::prob1Agreement;
namespace prob1Exchange = TRTLLM_MV2_NS::prob1Exchange;

constexpr int kWorlds[] = {2, 4, 8};

struct Model
{
    std::atomic<Word> committed{0};
    std::vector<Word> last, votes, gates;

    explicit Model(int world)
        : last(world)
        , votes(world)
        , gates(world)
    {
    }

    Word begin(int rank, bool error = false)
    {
        Word seq = last[rank] + 1;
        if (seq > kMaxSequence)
        {
            throw std::logic_error("sequence exhausted");
        }
        gates[rank] = producerGate(committed.load());
        votes[rank] = encode(seq, error ? Abort : Success);
        return seq;
    }

    bool healthy(Word seq) const
    {
        return std::all_of(
            votes.begin(), votes.end(), [seq](Word v) { return sequence(v) == seq && state(v) == Success; });
    }

    State propose(int rank, Word seq, State desired)
    {
        // Same sequential-per-rank precondition as the device's sequence increment.
        if (seq != last[rank] + 1)
        {
            throw std::invalid_argument("skipped local sequence");
        }
        if (desired == Success && !healthy(seq))
        {
            throw std::invalid_argument("missing or bad votes");
        }
        Word old = committed.load();
        Word next = proposal(old, seq, desired);
        committed.compare_exchange_strong(old, next);
        return resolve(committed.load(), seq);
    }

    State finish(int rank)
    {
        Word seq = last[rank] + 1;
        State result = resolve(committed.load(), seq);
        if (result == Pending)
        {
            throw std::logic_error("finished before a decision");
        }
        last[rank] = seq;
        return result;
    }
};

TEST(MiniMaxM3Prob1Agreement, LatePeerAbortIsStickyUntilANewWorkspace)
{
    for (int world : kWorlds)
    {
        SCOPED_TRACE(world);
        // Early timeout versus a genuinely late peer: every rank aborts, no producer-state reset.
        Model m(world);
        m.begin(0);
        EXPECT_EQ(m.propose(0, 1, Abort), Abort);
        for (int r = 1; r < world; ++r)
        {
            m.begin(r);
            EXPECT_EQ(m.gates[r], 0u);
        }
        for (int r = 0; r < world; ++r)
        {
            EXPECT_EQ(m.finish(r), Abort);
        }
        Word sticky = m.committed.load();
        for (int round = 0; round < 12; ++round)
        {
            for (int r = 0; r < world; ++r)
            {
                m.begin(r);
            }
            for (int r = 0; r < world; ++r)
            {
                EXPECT_EQ(m.gates[r], 0u);
                EXPECT_EQ(m.finish(r), Abort);
            }
            EXPECT_EQ(m.committed.load(), sticky);
        }
        // Recovery needs a freshly constructed collective workspace; there is no in-place reset.
        Model rebuilt(world);
        for (int r = 0; r < world; ++r)
        {
            rebuilt.begin(r);
        }
        EXPECT_EQ(rebuilt.propose(0, 1, Success), Success);
        for (int r = 0; r < world; ++r)
        {
            EXPECT_EQ(rebuilt.finish(r), Success);
        }
    }
}

TEST(MiniMaxM3Prob1Agreement, LaterAbortKeepsAnEarlierSuccess)
{
    for (int world : kWorlds)
    {
        SCOPED_TRACE(world);
        Model later(world);
        for (int r = 0; r < world; ++r)
        {
            later.begin(r);
        }
        later.propose(0, 1, Success);
        EXPECT_EQ(later.finish(0), Success);
        later.begin(0);
        later.propose(0, 2, Abort);
        for (int r = 1; r < world; ++r)
        {
            EXPECT_EQ(later.finish(r), Success);
            later.begin(r);
            EXPECT_EQ(later.gates[r], 0u);
        }
        for (int r = 0; r < world; ++r)
        {
            EXPECT_EQ(later.finish(r), Abort);
        }
        EXPECT_EQ(resolve(later.committed.load(), 1), Success);
    }
}

TEST(MiniMaxM3Prob1Agreement, SuccessAndTimeoutRaceResolvesToTheFirstCommit)
{
    // Both linearization orders, including the stale observation of the losing compare-exchange.
    for (State first : {Success, Abort})
    {
        std::atomic<Word> word{0};
        Word observedA = 0, observedB = 0;
        Word a = proposal(observedA, 1, first);
        Word b = proposal(observedB, 1, first == Success ? Abort : Success);
        ASSERT_TRUE(word.compare_exchange_strong(observedA, a));
        ASSERT_FALSE(word.compare_exchange_strong(observedB, b));
        EXPECT_EQ(resolve(word.load(), 1), first);
        EXPECT_EQ(resolve(observedB, 1), first);
    }
}

TEST(MiniMaxM3Prob1Agreement, BadVotesAndSkippedSequencesAreRejected)
{
    for (int world : kWorlds)
    {
        SCOPED_TRACE(world);
        Model bad(world);
        for (int r = 0; r < world; ++r)
        {
            bad.begin(r, r == world - 1);
        }
        EXPECT_THROW(bad.propose(0, 1, Success), std::invalid_argument);
        EXPECT_EQ(bad.propose(0, 1, Abort), Abort);
        EXPECT_THROW(bad.propose(0, 3, Abort), std::invalid_argument);
    }
}

TEST(MiniMaxM3Prob1Agreement, ProducerGateFollowsTheSnapshot)
{
    // The snapshot taken at prepare time gates the whole producer grid, even if a rank aborts mid-grid.
    EXPECT_EQ(producerGate(0), 1u);
    EXPECT_EQ(producerGate(encode(1, Abort)), 0u);
}

TEST(MiniMaxM3Prob1Exchange, UnwrittenAndStaleTagsAreNotReady)
{
    for (Word ep : {16384ull, 32768ull, 49152ull})
    {
        EXPECT_FALSE(prob1Exchange::ready(0, ep));
        EXPECT_FALSE(prob1Exchange::ready(ep - 16384, ep));
        EXPECT_FALSE(prob1Exchange::ready(ep + 16384, ep));
        EXPECT_TRUE(prob1Exchange::ready(ep, ep));
    }
    EXPECT_FALSE(prob1Exchange::ready(0, 0));
}

TEST(MiniMaxM3Prob1Exchange, ProducerOpensOnlyAfterEveryReaderFinished)
{
    for (int world : kWorlds)
    {
        SCOPED_TRACE(world);
        // The second row starts unwritten even where an older packed tag would read as zero.
        std::array<Word, prob1Exchange::kAllocationWords> xs{};
        int const slot = world + (world - 1);
        EXPECT_FALSE(prob1Exchange::ready(xs[prob1Exchange::tagIndex(slot)], 32768));
        xs[prob1Exchange::tagIndex(slot)] = 16384;
        EXPECT_FALSE(prob1Exchange::ready(xs[prob1Exchange::tagIndex(slot)], 32768));
        xs[prob1Exchange::tagIndex(slot)] = 32768;
        EXPECT_TRUE(prob1Exchange::ready(xs[prob1Exchange::tagIndex(slot)], 32768));

        // A fast rank completed the agreement while a slow reader still owns its cache read.
        std::vector<Word> acks(world, 1);
        acks.back() = 0;
        auto allDone
            = [&]() { return std::all_of(acks.begin(), acks.end(), [](Word ack) { return readerDone(ack, 1); }); };
        Word decision = encode(1, Success);
        EXPECT_EQ(readerAction(decision, allDone(), false), WaitReaders);
        EXPECT_EQ(readerAction(decision, allDone(), true), AbortReaders);
        EXPECT_FALSE(allDone());
        // A bounded prepare timeout aborts the next call; the late reader finishing cannot reopen the producer.
        Word aborted = proposal(decision, 2, Abort);
        EXPECT_EQ(resolve(aborted, 1), Success);
        EXPECT_EQ(resolve(aborted, 2), Abort);
        acks.back() = 1;
        EXPECT_TRUE(allDone());
        EXPECT_EQ(producerGate(aborted), 0u);
        EXPECT_EQ(readerAction(aborted, allDone(), false), StopProducer);
        EXPECT_EQ(readerAction(decision, allDone(), true), OpenProducer);
        EXPECT_EQ(producerGate(decision), 1u);
        // A rank may finish the next reader while another still prepares; the comparison is >=.
        acks.front() = 2;
        EXPECT_TRUE(allDone());
        EXPECT_TRUE(readerDone(0, 0));

        // No partial prefix of any reader completion order opens the producer.
        std::vector<int> order(world);
        for (int r = 0; r < world; ++r)
        {
            order[r] = r;
        }
        do
        {
            std::fill(acks.begin(), acks.end(), 0);
            for (int i = 0; i < world; ++i)
            {
                ASSERT_EQ(readerAction(decision, allDone(), false), WaitReaders);
                acks[order[i]] = 1;
            }
            ASSERT_EQ(readerAction(decision, allDone(), false), OpenProducer);
        } while (std::next_permutation(order.begin(), order.end()));
    }
}

TEST(MiniMaxM3Prob1Agreement, SequenceSpaceCannotWrap)
{
    static_assert(kMaxSequence == (1ull << 60) - 1, "16 epochs per call cannot wrap a 64-bit generation");
    static_assert(kMaxSequence * 16 < ~0ull, "the last producer epoch is representable");
    EXPECT_TRUE(sequenceAvailable(kMaxSequence - 1));
    EXPECT_FALSE(sequenceAvailable(kMaxSequence));
    EXPECT_FALSE(sequenceAvailable(~0ull));
    EXPECT_EQ(sequence(encode(kMaxSequence, Abort)), kMaxSequence);
}

} // namespace
