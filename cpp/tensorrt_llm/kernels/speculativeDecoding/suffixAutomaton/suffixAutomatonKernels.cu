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
