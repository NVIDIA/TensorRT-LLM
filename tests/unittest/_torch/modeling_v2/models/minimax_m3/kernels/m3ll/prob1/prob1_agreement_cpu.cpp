// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// A host model of the draft prob1 agreement and exchange protocols, compiled against the family's own headers
// (csrc/m31/prob1Agreement.h, prob1Exchange.h) by test_m31_prob1_agreement_cpu.py.
#include "minimax_m3/csrc/m31/prob1Agreement.h"
#include "minimax_m3/csrc/m31/prob1Exchange.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace TRTLLM_MV2_NS::prob1Agreement;
namespace prob1Exchange = TRTLLM_MV2_NS::prob1Exchange;

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
        assert(seq <= kMaxSequence);
        gates[rank] = producerGate(committed.load());
        votes[rank] = encode(seq, error ? Abort : Success);
        return seq;
    }

    bool healthy(Word seq) const
    {
        for (Word v : votes)
            if (sequence(v) != seq || state(v) != Success)
                return false;
        return true;
    }

    State propose(int rank, Word seq, State desired)
    {
        // Same sequential-per-rank precondition as the device's seqp increment.
        if (seq != last[rank] + 1)
            throw std::invalid_argument("skipped local sequence");
        if (desired == Success && !healthy(seq))
            throw std::invalid_argument("missing or bad votes");
        Word old = committed.load();
        Word next = proposal(old, seq, desired);
        committed.compare_exchange_strong(old, next);
        return resolve(committed.load(), seq);
    }

    State finish(int rank)
    {
        Word seq = last[rank] + 1;
        State result = resolve(committed.load(), seq);
        assert(result != Pending);
        last[rank] = seq;
        return result;
    }
};

int main()
{
    for (int world : {2, 4, 8})
    {
        // Early timeout versus a genuinely late peer: every rank aborts, no producer-state reset.
        Model m(world);
        m.begin(0);
        assert(m.propose(0, 1, Abort) == Abort);
        for (int r = 1; r < world; ++r)
        {
            m.begin(r);
            assert(m.gates[r] == 0);
        }
        for (int r = 0; r < world; ++r)
            assert(m.finish(r) == Abort);
        Word sticky = m.committed.load();
        for (int round = 0; round < 12; ++round)
        {
            for (int r = 0; r < world; ++r)
                m.begin(r);
            for (int r = 0; r < world; ++r)
            {
                assert(m.gates[r] == 0);
                assert(m.finish(r) == Abort);
            }
            assert(m.committed.load() == sticky);
        }
        // Recover only by a freshly constructed collective workspace; no in-place reset API exists.
        Model rebuilt(world);
        for (int r = 0; r < world; ++r)
            rebuilt.begin(r);
        assert(rebuilt.propose(0, 1, Success) == Success);
        for (int r = 0; r < world; ++r)
            assert(rebuilt.finish(r) == Success);
        // Later ABORT must not rewrite an earlier committed SUCCESS seen by a delayed peer.
        Model later(world);
        for (int r = 0; r < world; ++r)
            later.begin(r);
        later.propose(0, 1, Success);
        assert(later.finish(0) == Success);
        later.begin(0);
        later.propose(0, 2, Abort);
        for (int r = 1; r < world; ++r)
        {
            assert(later.finish(r) == Success);
            later.begin(r);
            assert(later.gates[r] == 0);
        }
        for (int r = 0; r < world; ++r)
            assert(later.finish(r) == Abort);
        assert(resolve(later.committed.load(), 1) == Success);
        // SUCCESS-vs-timeout CAS: enumerate both linearization orders, including stale CAS observations.
        for (State first : {Success, Abort})
        {
            std::atomic<Word> word{0};
            Word observedA = 0, observedB = 0;
            Word a = proposal(observedA, 1, first), b = proposal(observedB, 1, first == Success ? Abort : Success);
            assert(word.compare_exchange_strong(observedA, a));
            assert(!word.compare_exchange_strong(observedB, b));
            assert(resolve(word.load(), 1) == first && resolve(observedB, 1) == first);
        }
        Model bad(world);
        for (int r = 0; r < world; ++r)
            bad.begin(r, r == world - 1);
        bool rejected = false;
        try
        {
            bad.propose(0, 1, Success);
        }
        catch (std::invalid_argument const&)
        {
            rejected = true;
        }
        assert(rejected);
        assert(bad.propose(0, 1, Abort) == Abort);
        bool skipped = false;
        try
        {
            bad.propose(0, 3, Abort);
        }
        catch (std::invalid_argument const&)
        {
            skipped = true;
        }
        assert(skipped);
        // The prep snapshot is immutable for the entire producer grid, even if another rank aborts mid-grid.
        Word gate = producerGate(0);
        Word aborted = encode(1, Abort);
        for (int cta = 0; cta < 512; ++cta)
            assert(gate == 1);
        assert(producerGate(aborted) == 0);
        for (int cta = 0; cta < 512; ++cta)
            assert(producerGate(aborted) == 0);
    }
    // Actual shared readiness helper: zero/unwritten and nonzero stale values cannot alias full tags.
    for (Word ep : {16384ull, 32768ull, 49152ull})
    {
        assert(!prob1Exchange::ready(0, ep));
        assert(!prob1Exchange::ready(ep - 16384, ep));
        assert(!prob1Exchange::ready(ep + 16384, ep));
        assert(prob1Exchange::ready(ep, ep));
    }
    assert(!prob1Exchange::ready(0, 0));
    for (int world : {2, 4, 8})
    {
        // C1 -> C2: second row starts unwritten even when the old packed14 tag would equal zero.
        std::array<Word, prob1Exchange::kAllocationWords> xs{};
        int const slot = (1 * world) + (world - 1);
        assert(!prob1Exchange::ready(xs[prob1Exchange::tagIndex(slot)], 32768));
        xs[prob1Exchange::tagIndex(slot)] = 16384;
        assert(!prob1Exchange::ready(xs[prob1Exchange::tagIndex(slot)], 32768));
        xs[prob1Exchange::tagIndex(slot)] = 32768;
        assert(prob1Exchange::ready(xs[prob1Exchange::tagIndex(slot)], 32768));
        // Fast rank completes agreement/adapter1; a slow old adapter still owns its cache read.
        std::vector<Word> acks(world, 1);
        acks.back() = 0;
        auto allDone = [&]()
        {
            for (Word ack : acks)
                if (!readerDone(ack, 1))
                    return false;
            return true;
        };
        assert(readerAction(encode(1, Success), allDone(), false) == WaitReaders);
        assert(readerAction(encode(1, Success), allDone(), true) == AbortReaders);
        assert(!allDone());                          // producer2 must not overwrite any mirrored cache byte.
        Word decision = encode(1, Success);
        Word aborted = proposal(decision, 2, Abort); // bounded prep timeout; no payload produced.
        assert(resolve(aborted, 1) == Success && resolve(aborted, 2) == Abort);
        acks.back() = 1; // late old reader finishes, but terminal abort cannot reopen producer2.
        assert(allDone() && producerGate(aborted) == 0);
        assert(readerAction(aborted, allDone(), false) == StopProducer);
        assert(readerAction(decision, allDone(), true) == OpenProducer);
        // Healthy schedule opens producer2 only after the last old adapter completion.
        assert(allDone() && producerGate(decision) == 1);
        for (int cta = 0; cta < 512; ++cta)
            assert(readerDone(acks.back(), 1));
        // A rank may already finish adapter2 while another rank still prepares2; >= is intentional.
        acks.front() = 2;
        assert(allDone());
        assert(readerDone(0, 0)); // first invocation needs no predecessor.
        // Exhaust every ordering of peer adapter finishes; no partial prefix may open producer2.
        std::vector<int> order(world);
        for (int r = 0; r < world; ++r)
            order[r] = r;
        do
        {
            std::fill(acks.begin(), acks.end(), 0);
            for (int i = 0; i < world; ++i)
            {
                assert(readerAction(decision, allDone(), false) == WaitReaders);
                acks[order[i]] = 1;
            }
            assert(readerAction(decision, allDone(), false) == OpenProducer);
        } while (std::next_permutation(order.begin(), order.end()));
    }
    static_assert(kMaxSequence == (1ull << 60) - 1, "16 epochs/call cannot wrap full64 generation");
    static_assert(kMaxSequence * 16 < ~0ull, "last allowed producer epoch is representable");
    assert(sequenceAvailable(kMaxSequence - 1));
    assert(!sequenceAvailable(kMaxSequence));
    assert(!sequenceAvailable(~0ull));
    assert(sequence(encode(kMaxSequence, Abort)) == kMaxSequence);
    std::cout << "PROB1_AGREEMENT_CPU_PASS worlds2/4/8 latepeer stickyabort earlier-success CASrace gate recovery\n";
}
