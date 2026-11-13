/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>
#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

// One coordinator decision, never reset in place. All ranks call sequences consecutively from one.
// SUCCESS at N implies all earlier decisions succeeded. ABORT at N is terminal for N and all later calls;
// earlier committed successes remain successes. Reuse requires collective workspace AND graph rebuild.
TRTLLM_MV2_NAMESPACE_BEGIN

namespace prob1Agreement
{
using Word = unsigned long long;

enum State : Word
{
    Pending = 0,
    Success = 1,
    Abort = 2
};

constexpr int kDecisionOffset = 8;
constexpr int kGateOffset = 9;
constexpr int kDoneOffset = 10;
constexpr Word kMaxSequence = (1ull << 60) - 1;
#if defined(__CUDACC__)
#define PROB1_HD __host__ __device__
#else
#define PROB1_HD
#endif
PROB1_HD constexpr Word encode(Word sequence, State state)
{
    return (sequence << 2) | state;
}

PROB1_HD constexpr Word sequence(Word word)
{
    return word >> 2;
}

PROB1_HD constexpr State state(Word word)
{
    return static_cast<State>(word & 3);
}

PROB1_HD constexpr State resolve(Word word, Word requested)
{
    return state(word) == Abort                                 ? (requested < sequence(word) ? Success : Abort)
        : state(word) == Success && requested <= sequence(word) ? Success
                                                                : Pending;
}

PROB1_HD constexpr Word proposal(Word observed, Word requested, State desired)
{
    return resolve(observed, requested) == Pending ? encode(requested, desired) : observed;
}

PROB1_HD constexpr Word producerGate(Word committed)
{
    return state(committed) == Abort ? 0 : 1;
}

enum ReaderAction
{
    WaitReaders,
    OpenProducer,
    StopProducer,
    AbortReaders
};

PROB1_HD constexpr ReaderAction readerAction(Word committed, bool allDone, bool expired)
{
    return !producerGate(committed) ? StopProducer : allDone ? OpenProducer : expired ? AbortReaders : WaitReaders;
}

PROB1_HD constexpr bool sequenceAvailable(Word prior)
{
    return prior < kMaxSequence;
}

PROB1_HD constexpr bool readerDone(Word completed, Word prior)
{
    return completed >= prior;
}

#undef PROB1_HD
#if defined(__CUDACC__)
__device__ __forceinline__ Word load(Word const* p)
{
    Word v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store(Word* p, Word v)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ Word clockNs()
{
    Word v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}

// Called by prep lane zero before the producer. A completed prior agreement is insufficient:
// all peers must also have finished reading the mirrored cache in the adapter.
__device__ __forceinline__ Word prepare(int world, Word* const* peers, Word const* seqp, int64_t capNs)
{
    Word const priorSeq = *seqp;
    if (!sequenceAvailable(priorSeq))
    {
        __trap();
        return 0;
    }
    Word* const decision = peers[0] + kDecisionOffset;
    Word const start = clockNs();
    while (true)
    {
        Word observed = load(decision);
        bool ready = true;
        for (int r = 0; r < world; ++r)
            ready = ready && readerDone(load(peers[r] + kDoneOffset), priorSeq);
        ReaderAction const action = readerAction(observed, ready, clockNs() - start > static_cast<Word>(capNs));
        if (action == StopProducer)
            return 0;
        if (action == OpenProducer)
            return 1;
        if (action == AbortReaders)
        {
            Word const next = proposal(observed, priorSeq + 1, Abort);
            cuda::atomic_ref<Word, cuda::thread_scope_system> sharedDecision(*decision);
            Word old = observed;
            if (sharedDecision.compare_exchange_strong(
                    old, next, cuda::std::memory_order_acq_rel, cuda::std::memory_order_acquire))
            {
                // A SUCCESS for this unproduced call would violate consecutive collective invocation.
                if (resolve(next, priorSeq + 1) != Abort)
                    __trap();
                return 0;
            }
        }
    }
}

// A separate launch after writeKernel completes acknowledges every adapter CTA, not merely its last writer.
template <int Backend>
__global__ void finishReaders(Word* ack, Word const* seqp)
{
    if (threadIdx.x == 0)
        store(ack, *seqp);
}

__device__ __forceinline__ void agree(
    int rank, int world, int const* err, Word* const* peers, Word* seqp, int64_t capNs, int* fault, int64_t* faultCount)
{
    // Single lane: CAS on rank0's system-atomic-capable MNNVL word is the sole linearization point.
    if (threadIdx.x != 0)
        return;
    if (!sequenceAvailable(*seqp))
    {
        __trap();
        return;
    } // rebuild before finite sequence-space exhaustion
    Word const seq = *seqp + 1;
    Word* const decision = peers[0] + kDecisionOffset;
    Word const vote = encode(seq, *err ? Abort : Success);
    for (int r = 0; r < world; ++r)
        store(peers[r] + rank, vote);
    Word const start = clockNs();
    State outcome = Pending;
    while (outcome == Pending)
    {
        Word observed = load(decision);
        outcome = resolve(observed, seq);
        if (outcome != Pending)
            break;
        bool ready = true, bad = *err != 0;
        for (int r = 0; r < world; ++r)
        {
            Word const v = load(peers[rank] + r);
            ready = ready && sequence(v) == seq;
            bad = bad || (sequence(v) == seq && state(v) == Abort);
        }
        if (bad || ready || clockNs() - start > static_cast<Word>(capNs))
        {
            State const desired = bad || !ready ? Abort : Success;
            Word const proposed = proposal(observed, seq, desired);
            cuda::atomic_ref<Word, cuda::thread_scope_system> sharedDecision(*decision);
            Word prior = observed;
            bool const won = sharedDecision.compare_exchange_strong(
                prior, proposed, cuda::std::memory_order_acq_rel, cuda::std::memory_order_acquire);
            outcome = resolve(won ? proposed : prior, seq);
        }
    }
    *seqp = seq;
    *fault = outcome == Abort ? 1 : 0;
    if (outcome == Abort)
        *faultCount += 1;
    // Never clear producer state: a late peer may still be running that sequence.
    __threadfence_system();
}
#endif
} // namespace prob1Agreement

TRTLLM_MV2_NAMESPACE_END
