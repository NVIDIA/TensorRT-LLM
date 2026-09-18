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

#include <cstdint>

#include "ngramKernels.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::ngram
{

namespace
{

constexpr int kExtendBlockSize = 32;
constexpr int kDraftBlockSize = 256;

// A candidate match is ranked by a single 64-bit key so the best one falls out of a max reduction.
// Most significant first: match length, own-slot flag, position score, slot score.
constexpr int kSlotScoreBits = 16;
constexpr int kPositionScoreBits = 31;
constexpr int kOwnSlotBit = kSlotScoreBits + kPositionScoreBits;
constexpr int kMatchLenShift = kOwnSlotBit + 1;
constexpr uint64_t kSlotScoreMask = (uint64_t{1} << kSlotScoreBits) - 1;
constexpr uint64_t kPositionScoreMask = (uint64_t{1} << kPositionScoreBits) - 1;

__device__ __forceinline__ uint64_t makeMatchKey(int matchLen, bool ownSlot, int positionScore, int slotScore)
{
    return (static_cast<uint64_t>(matchLen) << kMatchLenShift) | (static_cast<uint64_t>(ownSlot) << kOwnSlotBit)
        | (static_cast<uint64_t>(positionScore) << kSlotScoreBits) | static_cast<uint64_t>(slotScore);
}

__device__ __forceinline__ uint64_t blockMax(uint64_t val, uint64_t* shared)
{
    shared[threadIdx.x] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            uint64_t const other = shared[threadIdx.x + stride];
            if (other > shared[threadIdx.x])
            {
                shared[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void ngramExtendHistoryKernel(NGramDraftParams params)
{
    int const row = blockIdx.x;
    if (params.rowMask[row] == 0)
    {
        return;
    }

    int const slot = params.slotIds[row];
    int const len = params.historyLens[slot];
    int numNew = min(params.numAcceptedTokens[row], params.acceptedStride);
    numNew = max(0, min(numNew, params.maxSeqLen - len));

    int* history = params.historyTokens + static_cast<size_t>(slot) * params.maxSeqLen + len;
    int const* accepted = params.acceptedTokens + static_cast<size_t>(row) * params.acceptedStride;
    for (int j = threadIdx.x; j < numNew; j += blockDim.x)
    {
        history[j] = accepted[j];
    }
    if (threadIdx.x == 0)
    {
        params.historyLens[slot] = len + numNew;
    }
}

//! Scan one slot's history for occurrences of the row's suffix and fold the best candidate into ``best``.
//!
//! A candidate ends at position ``end`` (the last token of the occurrence) and needs at least one token
//! after it to draft from, so ``end + 1 < searchedLen``. For the row's own slot this also excludes the
//! suffix itself, which ends at ``len - 1``.
__device__ __forceinline__ void searchHistory(
    NGramDraftParams const& params, int const* suffixEnd, int maxNgram, int searchedSlot, int ownSlot, uint64_t& best)
{
    int const searchedLen = params.historyLens[searchedSlot];
    int const* searched = params.historyTokens + static_cast<size_t>(searchedSlot) * params.maxSeqLen;
    bool const isOwn = searchedSlot == ownSlot;
    int const slotScore = params.numSlots - 1 - searchedSlot;
    int const positionScoreMax = params.maxSeqLen - 1;

    for (int end = threadIdx.x; end + 1 < searchedLen; end += blockDim.x)
    {
        int matchLen = 0;
        while (matchLen < maxNgram && end - matchLen >= 0 && searched[end - matchLen] == suffixEnd[-matchLen])
        {
            ++matchLen;
        }
        if (matchLen > 0)
        {
            int const positionScore = params.useOldest ? positionScoreMax - end : end;
            uint64_t const key = makeMatchKey(matchLen, isOwn, positionScore, slotScore);
            if (key > best)
            {
                best = key;
            }
        }
    }
}

__global__ void ngramDraftKernel(NGramDraftParams params)
{
    __shared__ uint64_t sharedKeys[kDraftBlockSize];

    int const row = blockIdx.x;
    int const draftLen = params.draftLen;
    int* draftOut = params.draftTokensOut + static_cast<size_t>(row) * draftLen;

    int const slot = params.slotIds[row];
    int const len = params.historyLens[slot];
    int const maxNgram = min(params.maxNgramSize, len - 1);
    bool const active = params.rowMask[row] != 0 && maxNgram > 0;

    uint64_t best = 0;
    if (active)
    {
        int const* suffixEnd = params.historyTokens + static_cast<size_t>(slot) * params.maxSeqLen + len - 1;
        searchHistory(params, suffixEnd, maxNgram, slot, slot, best);
        if (params.publicPool)
        {
            for (int other = 0; other < params.numSlots; ++other)
            {
                if (other != slot && params.historyLens[other] > 0)
                {
                    searchHistory(params, suffixEnd, maxNgram, other, slot, best);
                }
            }
        }
    }
    best = blockMax(best, sharedKeys);

    if (best == 0)
    {
        for (int j = threadIdx.x; j < draftLen; j += blockDim.x)
        {
            draftOut[j] = 0;
        }
        if (threadIdx.x == 0)
        {
            params.matchLenOut[row] = 0;
        }
        return;
    }

    int const matchLen = static_cast<int>(best >> kMatchLenShift);
    int const matchedSlot = params.numSlots - 1 - static_cast<int>(best & kSlotScoreMask);
    int const positionScore = static_cast<int>((best >> kSlotScoreBits) & kPositionScoreMask);
    int const end = params.useOldest ? params.maxSeqLen - 1 - positionScore : positionScore;
    int const matchedLen = params.historyLens[matchedSlot];
    int const* matched = params.historyTokens + static_cast<size_t>(matchedSlot) * params.maxSeqLen;

    for (int j = threadIdx.x; j < draftLen; j += blockDim.x)
    {
        int const pos = end + 1 + j;
        draftOut[j] = pos < matchedLen ? matched[pos] : 0;
    }
    if (threadIdx.x == 0)
    {
        params.matchLenOut[row] = matchLen;
    }
}

} // namespace

void invokeNGramExtendAndDraft(NGramDraftParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.batchSize >= 0, "batchSize must be non-negative");
    TLLM_CHECK_WITH_INFO(params.draftLen >= 0, "draftLen must be non-negative");
    TLLM_CHECK_WITH_INFO(params.acceptedStride > 0, "acceptedStride must be positive");
    TLLM_CHECK_WITH_INFO(params.maxNgramSize > 0 && params.maxNgramSize <= kNGramMaxNgramSize,
        "maxNgramSize must be in [1, %d]", kNGramMaxNgramSize);
    TLLM_CHECK_WITH_INFO(
        params.numSlots > 0 && params.numSlots <= kNGramMaxSlots, "numSlots must be in [1, %d]", kNGramMaxSlots);
    TLLM_CHECK_WITH_INFO(params.maxSeqLen > 0, "maxSeqLen must be positive");
    if (params.batchSize == 0)
    {
        return;
    }

    ngramExtendHistoryKernel<<<params.batchSize, kExtendBlockSize, 0, stream>>>(params);
    sync_check_cuda_error(stream);

    if (params.draftLen > 0)
    {
        ngramDraftKernel<<<params.batchSize, kDraftBlockSize, 0, stream>>>(params);
        sync_check_cuda_error(stream);
    }
}

} // namespace kernels::speculative_decoding::ngram

TRTLLM_NAMESPACE_END
