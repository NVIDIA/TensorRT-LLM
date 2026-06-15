/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#include <cassert>

#include "suffixAutomatonKernels.h"
#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

__global__ void suffixAutomatonExtendKernel(int batchSize, int draftLength, int maxSlots, size_t stateSize,
    void* slotsMemory, int const* batchIndices, int* matchLenOut, int* draftTokensOut, int const* acceptedTokensIn,
    int const* acceptedLensIn)
{
    // Only one thread per block does the work
    if (threadIdx.x > 0)
    {
        return;
    }

    int i = blockIdx.x;
    if (i >= batchSize)
    {
        return;
    }

    int batchIndex = batchIndices[i];
    assert(batchIndex >= 0 && batchIndex < maxSlots);

    // Calculate slot pointer based on dynamic state size
    uint8_t* slotMemory = static_cast<uint8_t*>(slotsMemory) + static_cast<size_t>(batchIndex) * stateSize;
    SuffixAutomaton* slot = reinterpret_cast<SuffixAutomaton*>(slotMemory);

    int numNewTokens = acceptedLensIn[i];
    // Bounds check: numNewTokens must be in valid range to prevent out-of-bounds access
    assert(numNewTokens >= 0 && numNewTokens <= draftLength + 1);

    // Extend the automaton with accepted tokens
    for (int j = 0; j < numNewTokens; j++)
    {
        slot->extend(Token(acceptedTokensIn[i * (draftLength + 1) + j]));
    }

    // Lookup the longest suffix match
    auto result = slot->lookup();
    if (result.hasValue())
    {
        matchLenOut[i] = result->len;
        slot->getDraftTokens(&draftTokensOut[i * draftLength], draftLength, result->pos);
    }
    else
    {
        matchLenOut[i] = 0;
    }
}

void invokeSuffixAutomatonExtend(SuffixAutomatonExtendParams const& params, cudaStream_t stream)
{
    params.checkParams();

    int batchSize = params.batchSize;
    int maxSlots = params.maxSlots;
    if (batchSize > maxSlots)
    {
        batchSize = maxSlots;
    }

    size_t stateSize = getSuffixAutomatonStateSize(params.maxSeqLen);

    // Launch one block per sequence, one thread per block
    suffixAutomatonExtendKernel<<<batchSize, 1, 0, stream>>>(batchSize, params.draftLength, maxSlots, stateSize,
        params.slots, params.batchIndices, params.matchLenOut, params.draftTokensOut, params.acceptedTokensIn,
        params.acceptedLensIn);
}

__global__ void suffixAutomatonExtendNgramKernel(int batchSize, int draftLength, int maxNgramSize, int maxSlots,
    size_t stateSize, void* slotsMemory, int const* batchIndices, int* matchLenOut, int* draftTokensOut,
    int const* acceptedTokensIn, int const* acceptedLensIn)
{
    // Only one thread per block does the work
    if (threadIdx.x > 0)
    {
        return;
    }

    int i = blockIdx.x;
    if (i >= batchSize)
    {
        return;
    }

    int batchIndex = batchIndices[i];
    assert(batchIndex >= 0 && batchIndex < maxSlots);

    // Calculate slot pointer based on dynamic state size
    uint8_t* slotMemory = static_cast<uint8_t*>(slotsMemory) + static_cast<size_t>(batchIndex) * stateSize;
    SuffixAutomaton* slot = reinterpret_cast<SuffixAutomaton*>(slotMemory);

    int numNewTokens = acceptedLensIn[i];
    // Bounds check: numNewTokens must be in valid range to prevent out-of-bounds access
    assert(numNewTokens >= 0 && numNewTokens <= draftLength + 1);

    // Extend the automaton with accepted tokens
    for (int j = 0; j < numNewTokens; j++)
    {
        slot->extend(Token(acceptedTokensIn[i * (draftLength + 1) + j]));
    }

    // Perform lookup based on maxNgramSize
    SAOptional<SuffixAutomaton::LookupResult> result;

    if (maxNgramSize == -1)
    {
        // Longest match mode
        result = slot->lookup();
    }
    else
    {
        // Fixed-size ngram mode - try sizes from maxNgramSize down to 1
        for (int size = maxNgramSize; size >= 1; size--)
        {
            result = slot->lookupFixed(size);
            if (result.hasValue())
            {
                break;
            }
        }
    }

    if (result.hasValue())
    {
        matchLenOut[i] = result->len;
        slot->getDraftTokens(&draftTokensOut[i * draftLength], draftLength, result->pos);
    }
    else
    {
        matchLenOut[i] = 0;
    }
}

void invokeSuffixAutomatonExtendNgram(SuffixAutomatonExtendNgramParams const& params, cudaStream_t stream)
{
    params.checkParams();

    int batchSize = params.batchSize;
    int maxSlots = params.maxSlots;
    if (batchSize > maxSlots)
    {
        batchSize = maxSlots;
    }

    size_t stateSize = getSuffixAutomatonStateSize(params.maxSeqLen);

    // Launch one block per sequence, one thread per block
    suffixAutomatonExtendNgramKernel<<<batchSize, 1, 0, stream>>>(batchSize, params.draftLength, params.maxNgramSize,
        maxSlots, stateSize, params.slots, params.batchIndices, params.matchLenOut, params.draftTokensOut,
        params.acceptedTokensIn, params.acceptedLensIn);
}

// =====================================================================
// Global search kernels (cross-request pattern sharing)
// =====================================================================

// Kernel 1: Extend all SAs with accepted tokens.
// Separate kernel ensures all mutations complete before cross-SA reads.
__global__ void suffixAutomatonGlobalExtendKernel(int batchSize, int draftLength, int maxSlots, size_t stateSize,
    void* slotsMemory, int const* batchIndices, int const* acceptedTokensIn, int const* acceptedLensIn)
{
    int reqIdx = blockIdx.x;
    if (reqIdx >= batchSize)
    {
        return;
    }

    int ownSlotIdx = batchIndices[reqIdx];
    assert(ownSlotIdx >= 0 && ownSlotIdx < maxSlots);
    uint8_t* slotMemory = static_cast<uint8_t*>(slotsMemory) + static_cast<size_t>(ownSlotIdx) * stateSize;
    SuffixAutomaton* ownSlot = reinterpret_cast<SuffixAutomaton*>(slotMemory);

    int numNewTokens = acceptedLensIn[reqIdx];
    assert(numNewTokens >= 0 && numNewTokens <= draftLength + 1);

    for (int j = 0; j < numNewTokens; j++)
    {
        ownSlot->extend(Token(acceptedTokensIn[reqIdx * (draftLength + 1) + j]));
    }
}

// Per-thread match result for shared-memory parallel reduction
struct SlotMatch
{
    int matchLen;
    int continuationLen;
    int isOwnSlot;
    int slotIdx;
    TextIndex pos;
};

// kMaxGlobalSuffixLen is defined in suffixAutomatonParams.h.
// With maxNgramSize == -1, longer sequences are silently truncated to that limit.

// Kernel 2: Search all active SAs in parallel, reduce to best match per request.
// All SAs are read-only (const) — launched after the extend kernel on the same stream.
__global__ void suffixAutomatonGlobalSearchKernel(int batchSize, int draftLength, int maxNgramSize, int maxSlots,
    size_t stateSize, void const* slotsMemory, int const* batchIndices, int const* activeSlotMask, int* matchLenOut,
    int* matchSlotOut, int* draftTokensOut)
{
    extern __shared__ SlotMatch sharedMatches[];

    int reqIdx = blockIdx.x;
    int slotIdx = threadIdx.x;

    if (reqIdx >= batchSize)
    {
        return;
    }

    int ownSlotIdx = batchIndices[reqIdx];
    assert(ownSlotIdx >= 0 && ownSlotIdx < maxSlots);

    // Step 1: Extract suffix from own SA into shared memory
    __shared__ Token sharedSuffix[kMaxGlobalSuffixLen];
    __shared__ int suffixLen;

    if (slotIdx == 0)
    {
        uint8_t const* slotMem = static_cast<uint8_t const*>(slotsMemory) + static_cast<size_t>(ownSlotIdx) * stateSize;
        SuffixAutomaton const* ownSlot = reinterpret_cast<SuffixAutomaton const*>(slotMem);

        int maxSuffixLen = (maxNgramSize > 0) ? maxNgramSize : kMaxGlobalSuffixLen;
        int textLen = +ownSlot->mTokens.size();
        suffixLen = (maxSuffixLen < textLen) ? maxSuffixLen : textLen;

        for (int i = 0; i < suffixLen; i++)
        {
            sharedSuffix[i] = ownSlot->mTokens.at(TextIndex(textLen - suffixLen + i));
        }
    }
    __syncthreads();

    // Step 2: Each thread searches one slot
    SlotMatch myMatch = {0, 0, 0, -1, TextIndex(0)};

    if (slotIdx < maxSlots && activeSlotMask[slotIdx])
    {
        uint8_t const* slotMem = static_cast<uint8_t const*>(slotsMemory) + static_cast<size_t>(slotIdx) * stateSize;
        SuffixAutomaton const* slot = reinterpret_cast<SuffixAutomaton const*>(slotMem);

        auto result = slot->lookupWithSuffix(sharedSuffix, suffixLen);
        if (result.hasValue())
        {
            myMatch.matchLen = result->len;
            myMatch.continuationLen = +slot->mTokens.size() - +result->pos;
            myMatch.isOwnSlot = (slotIdx == ownSlotIdx) ? 1 : 0;
            myMatch.slotIdx = slotIdx;
            myMatch.pos = result->pos;
        }
    }

    sharedMatches[slotIdx] = myMatch;
    __syncthreads();

    // Step 3: Parallel reduction — three-level comparison:
    //   1. Prefer longer match (higher matchLen)
    //   2. Among equal matchLen, prefer own slot
    //   3. Among equal matchLen and same locality, prefer longer continuation
    // Requires blockDim.x to be a power of 2 (guaranteed by nextPowerOf2 in the host launcher).
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (slotIdx < stride)
        {
            auto& current = sharedMatches[slotIdx];
            auto& candidate = sharedMatches[slotIdx + stride];
            bool replace = false;
            if (candidate.matchLen > current.matchLen)
            {
                replace = true;
            }
            else if (candidate.matchLen == current.matchLen && candidate.matchLen > 0)
            {
                if (candidate.isOwnSlot > current.isOwnSlot)
                {
                    replace = true;
                }
                else if (candidate.isOwnSlot == current.isOwnSlot
                    && candidate.continuationLen > current.continuationLen)
                {
                    replace = true;
                }
            }
            if (replace)
            {
                current = candidate;
            }
        }
        __syncthreads();
    }

    // Step 4: Thread 0 writes output
    if (slotIdx == 0)
    {
        SlotMatch best = sharedMatches[0];

        if (best.matchLen > 0 && best.slotIdx >= 0)
        {
            matchLenOut[reqIdx] = best.matchLen;
            matchSlotOut[reqIdx] = best.slotIdx;

            uint8_t const* slotMem
                = static_cast<uint8_t const*>(slotsMemory) + static_cast<size_t>(best.slotIdx) * stateSize;
            SuffixAutomaton const* slot = reinterpret_cast<SuffixAutomaton const*>(slotMem);
            slot->getDraftTokens(&draftTokensOut[reqIdx * draftLength], draftLength, best.pos);
        }
        else
        {
            matchLenOut[reqIdx] = 0;
            matchSlotOut[reqIdx] = -1;
        }
    }
}

namespace
{

int nextPowerOf2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return (v < 1) ? 1 : v;
}

} // anonymous namespace

void invokeSuffixAutomatonGlobalSearch(SuffixAutomatonGlobalSearchParams const& params, cudaStream_t stream)
{
    params.checkParams();

    int batchSize = params.batchSize;
    int maxSlots = params.maxSlots;
    if (batchSize > maxSlots)
    {
        batchSize = maxSlots;
    }

    size_t stateSize = getSuffixAutomatonStateSize(params.maxSeqLen);

    // Kernel 1: Extend all SAs (1 thread per block, 1 block per request)
    suffixAutomatonGlobalExtendKernel<<<batchSize, 1, 0, stream>>>(batchSize, params.draftLength, maxSlots, stateSize,
        params.slots, params.batchIndices, params.acceptedTokensIn, params.acceptedLensIn);

    // Kernel 2: Global search + reduce (N threads per block, 1 block per request)
    int threadsPerBlock = nextPowerOf2(maxSlots);
    threadsPerBlock = (threadsPerBlock < 1024) ? threadsPerBlock : 1024;

    size_t sharedMemSize = static_cast<size_t>(threadsPerBlock) * sizeof(SlotMatch);

    suffixAutomatonGlobalSearchKernel<<<batchSize, threadsPerBlock, sharedMemSize, stream>>>(batchSize,
        params.draftLength, params.maxNgramSize, maxSlots, stateSize, params.slots, params.batchIndices,
        params.activeSlotMask, params.matchLenOut, params.matchSlotOut, params.draftTokensOut);
}

size_t getSuffixAutomatonStateSize(size_t maxSeqLen)
{
    return SuffixAutomaton::getRequiredMemorySize(maxSeqLen);
}

void initAutomaton(void* memory, size_t maxSeqLen)
{
    SuffixAutomaton* sa = reinterpret_cast<SuffixAutomaton*>(memory);
    // Use placement new to construct the struct, then initialize
    new (sa) SuffixAutomaton();
    sa->init(memory, maxSeqLen);
}

void buildAutomatonFromTokens(SuffixAutomaton* sa, int const* tokens, int numTokens)
{
    // Extend the automaton with each token
    for (int i = 0; i < numTokens; i++)
    {
        sa->extend(Token(tokens[i]));
    }
}

void relocateAutomaton(SuffixAutomaton* sa, void* oldBase, void* newBase)
{
    sa->relocate(oldBase, newBase);
}

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
