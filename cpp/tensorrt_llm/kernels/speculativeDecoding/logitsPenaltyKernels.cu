/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/speculativeDecoding/logitsPenaltyKernels.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

template <typename T, typename TokenT>
__global__ void applySpeculativeTokenPenaltiesKernel(
    T* logits, TokenT const* tokenIds, float const* penaltyValues, int32_t width, int32_t vocabSize)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const rowTokenIds = tokenIds + row * width;
    auto const rowPenaltyValues = penaltyValues + row * width;
    auto rowLogits = logits + row * vocabSize;

    if (width <= 32)
    {
        if (threadIdx.x != 0)
        {
            return;
        }
        for (auto idx = 0; idx < width; ++idx)
        {
            auto const tokenId = static_cast<int64_t>(rowTokenIds[idx]);
            auto const penalty = rowPenaltyValues[idx];
            if (penalty != 0.0f && tokenId >= 0 && tokenId < vocabSize)
            {
                auto const offset = static_cast<int32_t>(tokenId);
                auto logit = static_cast<float>(rowLogits[offset]);
                logit -= penalty;
                rowLogits[offset] = static_cast<T>(logit);
            }
        }
        return;
    }

    for (auto idx = static_cast<int32_t>(threadIdx.x); idx < width; idx += static_cast<int32_t>(blockDim.x))
    {
        auto const tokenId = static_cast<int64_t>(rowTokenIds[idx]);
        auto const penalty = rowPenaltyValues[idx];
        if (penalty != 0.0f && tokenId >= 0 && tokenId < vocabSize)
        {
            auto const offset = static_cast<int32_t>(tokenId);
            auto logit = static_cast<float>(rowLogits[offset]);
            logit -= penalty;
            rowLogits[offset] = static_cast<T>(logit);
        }
    }
}

template <typename T, typename TokenT>
void invokeApplySpeculativeTokenPenalties(T* logits, TokenT const* tokenIds, float const* penaltyValues, int32_t numRows,
    int32_t width, int32_t vocabSize, cudaStream_t stream)
{
    if (numRows == 0 || width == 0)
    {
        return;
    }

    dim3 const grid(numRows);
    dim3 const block(std::min(width, 256));
    applySpeculativeTokenPenaltiesKernel<T, TokenT><<<grid, block, 0, stream>>>(
        logits, tokenIds, penaltyValues, width, vocabSize);
}

template void invokeApplySpeculativeTokenPenalties<float, int32_t>(
    float*, int32_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeTokenPenalties<float, int64_t>(
    float*, int64_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeTokenPenalties<half, int32_t>(
    half*, int32_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeTokenPenalties<half, int64_t>(
    half*, int64_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeTokenPenalties<__nv_bfloat16, int32_t>(
    __nv_bfloat16*, int32_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeTokenPenalties<__nv_bfloat16, int64_t>(
    __nv_bfloat16*, int64_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);

__global__ void applySpeculativeHistoryFrequencyPenaltyKernel(float* logits, int32_t const* historyTokens,
    int32_t const* historyLens, int32_t const* rowSlots, float const* frequencyPenalties, int32_t historyCapacity,
    int32_t vocabSize)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const frequencyPenalty = frequencyPenalties[row];
    if (frequencyPenalty == 0.0f)
    {
        return;
    }

    auto const slot = rowSlots[row];
    if (slot < 0)
    {
        return;
    }

    auto const historyLen = min(max(historyLens[slot], 0), historyCapacity);
    auto const rowHistory = historyTokens + static_cast<int64_t>(slot) * historyCapacity;
    auto rowLogits = logits + static_cast<int64_t>(row) * vocabSize;

    for (auto idx = static_cast<int32_t>(threadIdx.x); idx < historyLen; idx += static_cast<int32_t>(blockDim.x))
    {
        auto const tokenId = rowHistory[idx];
        if (tokenId >= 0 && tokenId < vocabSize)
        {
            atomicAdd(rowLogits + tokenId, -frequencyPenalty);
        }
    }
}

void invokeApplySpeculativeHistoryFrequencyPenalty(float* logits, int32_t const* historyTokens,
    int32_t const* historyLens, int32_t const* rowSlots, float const* frequencyPenalties, int32_t numRows,
    int32_t historyCapacity, int32_t vocabSize, cudaStream_t stream)
{
    if (numRows == 0 || historyCapacity == 0)
    {
        return;
    }

    dim3 const grid(numRows);
    dim3 const block(256);
    applySpeculativeHistoryFrequencyPenaltyKernel<<<grid, block, 0, stream>>>(
        logits, historyTokens, historyLens, rowSlots, frequencyPenalties, historyCapacity, vocabSize);
}

__global__ void appendSpeculativeAcceptedTokensKernel(int32_t* historyTokens, int32_t* historyLens,
    int32_t const* seqSlots, int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t acceptedStride,
    int32_t historyCapacity)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const slot = seqSlots[row];
    if (slot < 0)
    {
        return;
    }

    auto const acceptedLen = max(acceptedLens[row], 0);
    if (acceptedLen == 0)
    {
        return;
    }

    auto const oldLen = min(max(historyLens[slot], 0), historyCapacity);
    auto const writeLen = min(acceptedLen, max(historyCapacity - oldLen, 0));
    auto const rowAccepted = acceptedTokens + static_cast<int64_t>(row) * acceptedStride;
    auto rowHistory = historyTokens + static_cast<int64_t>(slot) * historyCapacity;

    for (auto idx = static_cast<int32_t>(threadIdx.x); idx < writeLen; idx += static_cast<int32_t>(blockDim.x))
    {
        rowHistory[oldLen + idx] = rowAccepted[idx];
    }

    if (threadIdx.x == 0)
    {
        historyLens[slot] = oldLen + writeLen;
    }
}

void invokeAppendSpeculativeAcceptedTokens(int32_t* historyTokens, int32_t* historyLens, int32_t const* seqSlots,
    int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t numRows, int32_t acceptedStride,
    int32_t historyCapacity, cudaStream_t stream)
{
    if (numRows == 0 || acceptedStride == 0 || historyCapacity == 0)
    {
        return;
    }

    dim3 const grid(numRows);
    dim3 const block(std::min(acceptedStride, 256));
    appendSpeculativeAcceptedTokensKernel<<<grid, block, 0, stream>>>(
        historyTokens, historyLens, seqSlots, acceptedTokens, acceptedLens, acceptedStride, historyCapacity);
}

template <typename T>
__global__ void applySpeculativeCountFrequencyPenaltyKernel(T* logits, int32_t const* tokenCounts,
    int32_t const* rowSlots, float const* frequencyPenalties, int32_t vocabSize)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const tokenId = static_cast<int32_t>(blockIdx.y) * static_cast<int32_t>(blockDim.x)
        + static_cast<int32_t>(threadIdx.x);
    if (tokenId >= vocabSize)
    {
        return;
    }

    auto const frequencyPenalty = frequencyPenalties[row];
    if (frequencyPenalty == 0.0f)
    {
        return;
    }

    auto const slot = rowSlots[row];
    if (slot < 0)
    {
        return;
    }

    auto const count = tokenCounts[static_cast<int64_t>(slot) * vocabSize + tokenId];
    if (count <= 0)
    {
        return;
    }

    auto rowLogits = logits + static_cast<int64_t>(row) * vocabSize;
    auto logit = static_cast<float>(rowLogits[tokenId]);
    logit -= frequencyPenalty * static_cast<float>(count);
    rowLogits[tokenId] = static_cast<T>(logit);
}

template <typename T>
void invokeApplySpeculativeCountFrequencyPenalty(T* logits, int32_t const* tokenCounts,
    int32_t const* rowSlots, float const* frequencyPenalties, int32_t numRows, int32_t vocabSize, cudaStream_t stream)
{
    if (numRows == 0 || vocabSize == 0)
    {
        return;
    }

    dim3 const block(256);
    dim3 const grid(numRows, (vocabSize + static_cast<int32_t>(block.x) - 1) / static_cast<int32_t>(block.x));
    applySpeculativeCountFrequencyPenaltyKernel<T><<<grid, block, 0, stream>>>(
        logits, tokenCounts, rowSlots, frequencyPenalties, vocabSize);
}

template void invokeApplySpeculativeCountFrequencyPenalty<float>(
    float*, int32_t const*, int32_t const*, float const*, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeCountFrequencyPenalty<half>(
    half*, int32_t const*, int32_t const*, float const*, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeCountFrequencyPenalty<__nv_bfloat16>(
    __nv_bfloat16*, int32_t const*, int32_t const*, float const*, int32_t, int32_t, cudaStream_t);

__global__ void appendSpeculativeAcceptedTokenCountsKernel(int32_t* tokenCounts, int32_t const* seqSlots,
    int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t acceptedStride, int32_t vocabSize)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const slot = seqSlots[row];
    if (slot < 0)
    {
        return;
    }

    auto const acceptedLen = min(max(acceptedLens[row], 0), acceptedStride);
    auto const rowAccepted = acceptedTokens + static_cast<int64_t>(row) * acceptedStride;
    auto rowCounts = tokenCounts + static_cast<int64_t>(slot) * vocabSize;

    for (auto idx = static_cast<int32_t>(threadIdx.x); idx < acceptedLen; idx += static_cast<int32_t>(blockDim.x))
    {
        auto const tokenId = rowAccepted[idx];
        if (tokenId >= 0 && tokenId < vocabSize)
        {
            atomicAdd(rowCounts + tokenId, 1);
        }
    }
}

void invokeAppendSpeculativeAcceptedTokenCounts(int32_t* tokenCounts, int32_t const* seqSlots,
    int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t numRows, int32_t acceptedStride,
    int32_t vocabSize, cudaStream_t stream)
{
    if (numRows == 0 || acceptedStride == 0 || vocabSize == 0)
    {
        return;
    }

    dim3 const grid(numRows);
    dim3 const block(std::min(acceptedStride, 256));
    appendSpeculativeAcceptedTokenCountsKernel<<<grid, block, 0, stream>>>(
        tokenCounts, seqSlots, acceptedTokens, acceptedLens, acceptedStride, vocabSize);
}

template <typename T>
__global__ void applySpeculativeSparseCountFrequencyPenaltyKernel(T* logits, int32_t const* tokenIds,
    int32_t const* tokenCounts, int32_t const* countLens, int32_t const* rowSlots, float const* frequencyPenalties,
    int32_t numRows, int32_t countCapacity, int32_t vocabSize)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const frequencyPenalty = frequencyPenalties[row];
    if (frequencyPenalty == 0.0f)
    {
        return;
    }

    auto const slot = rowSlots[row];
    if (slot < 0)
    {
        return;
    }

    if (row > 0 && rowSlots[row - 1] == slot && frequencyPenalties[row - 1] == frequencyPenalty)
    {
        return;
    }

    auto rowEnd = row + 1;
    while (rowEnd < numRows && rowSlots[rowEnd] == slot && frequencyPenalties[rowEnd] == frequencyPenalty)
    {
        ++rowEnd;
    }

    auto const countLen = min(max(countLens[slot], 0), countCapacity);
    auto const rowTokenIds = tokenIds + static_cast<int64_t>(slot) * countCapacity;
    auto const rowTokenCounts = tokenCounts + static_cast<int64_t>(slot) * countCapacity;

    for (auto idx = static_cast<int32_t>(threadIdx.x); idx < countLen; idx += static_cast<int32_t>(blockDim.x))
    {
        auto const tokenId = rowTokenIds[idx];
        auto const count = rowTokenCounts[idx];
        if (count > 0 && tokenId >= 0 && tokenId < vocabSize)
        {
            for (auto applyRow = row; applyRow < rowEnd; ++applyRow)
            {
                auto rowLogits = logits + static_cast<int64_t>(applyRow) * vocabSize;
                auto logit = static_cast<float>(rowLogits[tokenId]);
                logit -= frequencyPenalty * static_cast<float>(count);
                rowLogits[tokenId] = static_cast<T>(logit);
            }
        }
    }
}

template <typename T>
void invokeApplySpeculativeSparseCountFrequencyPenalty(T* logits, int32_t const* tokenIds,
    int32_t const* tokenCounts, int32_t const* countLens, int32_t const* rowSlots, float const* frequencyPenalties,
    int32_t numRows, int32_t countCapacity, int32_t vocabSize, cudaStream_t stream)
{
    if (numRows == 0 || countCapacity == 0 || vocabSize == 0)
    {
        return;
    }

    dim3 const grid(numRows);
    dim3 const block(256);
    applySpeculativeSparseCountFrequencyPenaltyKernel<T><<<grid, block, 0, stream>>>(logits, tokenIds, tokenCounts,
        countLens, rowSlots, frequencyPenalties, numRows, countCapacity, vocabSize);
}

template void invokeApplySpeculativeSparseCountFrequencyPenalty<float>(float*, int32_t const*, int32_t const*,
    int32_t const*, int32_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeSparseCountFrequencyPenalty<half>(half*, int32_t const*, int32_t const*,
    int32_t const*, int32_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);
template void invokeApplySpeculativeSparseCountFrequencyPenalty<__nv_bfloat16>(__nv_bfloat16*, int32_t const*,
    int32_t const*, int32_t const*, int32_t const*, float const*, int32_t, int32_t, int32_t, cudaStream_t);

__global__ void appendSpeculativeSparseTokenCountsKernel(int32_t* tokenIds, int32_t* tokenCounts, int32_t* countLens,
    int32_t const* seqSlots, int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t acceptedStride,
    int32_t countCapacity, int32_t vocabSize)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const slot = seqSlots[row];
    if (slot < 0)
    {
        return;
    }

    auto len = min(max(countLens[slot], 0), countCapacity);
    auto rowTokenIds = tokenIds + static_cast<int64_t>(slot) * countCapacity;
    auto rowTokenCounts = tokenCounts + static_cast<int64_t>(slot) * countCapacity;
    auto const rowAccepted = acceptedTokens + static_cast<int64_t>(row) * acceptedStride;
    auto const acceptedLen = min(max(acceptedLens[row], 0), acceptedStride);

    __shared__ int32_t lenShared;
    __shared__ int32_t tokenIdShared;
    __shared__ int32_t foundIdx;

    if (threadIdx.x == 0)
    {
        lenShared = len;
    }
    __syncthreads();

    for (auto acceptedIdx = 0; acceptedIdx < acceptedLen; ++acceptedIdx)
    {
        if (threadIdx.x == 0)
        {
            tokenIdShared = rowAccepted[acceptedIdx];
            foundIdx = -1;
        }
        __syncthreads();

        auto const tokenId = tokenIdShared;
        if (tokenId >= 0 && tokenId < vocabSize)
        {
            auto const currentLen = lenShared;
            for (auto idx = static_cast<int32_t>(threadIdx.x); idx < currentLen;
                 idx += static_cast<int32_t>(blockDim.x))
            {
                if (rowTokenIds[idx] == tokenId)
                {
                    atomicCAS(&foundIdx, -1, idx);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0)
            {
                if (foundIdx >= 0)
                {
                    rowTokenCounts[foundIdx] += 1;
                }
                else if (lenShared < countCapacity)
                {
                    rowTokenIds[lenShared] = tokenId;
                    rowTokenCounts[lenShared] = 1;
                    ++lenShared;
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0)
    {
        countLens[slot] = lenShared;
    }
}

void invokeAppendSpeculativeSparseTokenCounts(int32_t* tokenIds, int32_t* tokenCounts, int32_t* countLens,
    int32_t const* seqSlots, int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t numRows,
    int32_t acceptedStride, int32_t countCapacity, int32_t vocabSize, cudaStream_t stream)
{
    if (numRows == 0 || acceptedStride == 0 || countCapacity == 0 || vocabSize == 0)
    {
        return;
    }

    dim3 const grid(numRows);
    dim3 const block(std::min(countCapacity, 256));
    appendSpeculativeSparseTokenCountsKernel<<<grid, block, 0, stream>>>(tokenIds, tokenCounts, countLens, seqSlots,
        acceptedTokens, acceptedLens, acceptedStride, countCapacity, vocabSize);
}

__global__ void initSpeculativeSparseTokenCountsKernel(int32_t* tokenIds, int32_t* tokenCounts, int32_t* countLens,
    int32_t const* promptTokenIds, int32_t const* promptTokenCounts, int32_t const* promptLens,
    int32_t const* seqSlots, int32_t promptCapacity, int32_t countCapacity, int32_t vocabSize)
{
    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const slot = seqSlots[row];
    if (slot < 0)
    {
        return;
    }

    auto const len = min(min(max(promptLens[row], 0), promptCapacity), countCapacity);
    auto rowTokenIds = tokenIds + static_cast<int64_t>(slot) * countCapacity;
    auto rowTokenCounts = tokenCounts + static_cast<int64_t>(slot) * countCapacity;
    auto const rowPromptTokenIds = promptTokenIds + static_cast<int64_t>(row) * promptCapacity;
    auto const rowPromptTokenCounts = promptTokenCounts + static_cast<int64_t>(row) * promptCapacity;

    for (auto idx = static_cast<int32_t>(threadIdx.x); idx < len; idx += static_cast<int32_t>(blockDim.x))
    {
        auto const tokenId = rowPromptTokenIds[idx];
        auto const count = rowPromptTokenCounts[idx];
        if (tokenId >= 0 && tokenId < vocabSize && count > 0)
        {
            rowTokenIds[idx] = tokenId;
            rowTokenCounts[idx] = count;
        }
        else
        {
            rowTokenIds[idx] = 0;
            rowTokenCounts[idx] = 0;
        }
    }

    if (threadIdx.x == 0)
    {
        countLens[slot] = len;
    }
}

void invokeInitSpeculativeSparseTokenCounts(int32_t* tokenIds, int32_t* tokenCounts, int32_t* countLens,
    int32_t const* promptTokenIds, int32_t const* promptTokenCounts, int32_t const* promptLens,
    int32_t const* seqSlots, int32_t numRows, int32_t promptCapacity, int32_t countCapacity, int32_t vocabSize,
    cudaStream_t stream)
{
    if (numRows == 0 || promptCapacity == 0 || countCapacity == 0 || vocabSize == 0)
    {
        return;
    }

    dim3 const grid(numRows);
    dim3 const block(std::min(promptCapacity, 256));
    initSpeculativeSparseTokenCountsKernel<<<grid, block, 0, stream>>>(tokenIds, tokenCounts, countLens, promptTokenIds,
        promptTokenCounts, promptLens, seqSlots, promptCapacity, countCapacity, vocabSize);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
