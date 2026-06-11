/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_runtime.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

template <typename T, typename TokenT>
void invokeApplySpeculativeTokenPenalties(T* logits, TokenT const* tokenIds, float const* penaltyValues, int32_t numRows,
    int32_t width, int32_t vocabSize, cudaStream_t stream);

void invokeApplySpeculativeHistoryFrequencyPenalty(float* logits, int32_t const* historyTokens,
    int32_t const* historyLens, int32_t const* rowSlots, float const* frequencyPenalties, int32_t numRows,
    int32_t historyCapacity, int32_t vocabSize, cudaStream_t stream);

void invokeAppendSpeculativeAcceptedTokens(int32_t* historyTokens, int32_t* historyLens, int32_t const* seqSlots,
    int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t numRows, int32_t acceptedStride,
    int32_t historyCapacity, cudaStream_t stream);

template <typename T>
void invokeApplySpeculativeCountFrequencyPenalty(T* logits, int32_t const* tokenCounts,
    int32_t const* rowSlots, float const* frequencyPenalties, int32_t numRows, int32_t vocabSize, cudaStream_t stream);

void invokeAppendSpeculativeAcceptedTokenCounts(int32_t* tokenCounts, int32_t const* seqSlots,
    int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t numRows, int32_t acceptedStride,
    int32_t vocabSize, cudaStream_t stream);

template <typename T>
void invokeApplySpeculativeSparseCountFrequencyPenalty(T* logits, int32_t const* tokenIds,
    int32_t const* tokenCounts, int32_t const* countLens, int32_t const* rowSlots, float const* frequencyPenalties,
    int32_t numRows, int32_t countCapacity, int32_t vocabSize, cudaStream_t stream);

void invokeAppendSpeculativeSparseTokenCounts(int32_t* tokenIds, int32_t* tokenCounts, int32_t* countLens,
    int32_t const* seqSlots, int32_t const* acceptedTokens, int32_t const* acceptedLens, int32_t numRows,
    int32_t acceptedStride, int32_t countCapacity, int32_t vocabSize, cudaStream_t stream);

void invokeInitSpeculativeSparseTokenCounts(int32_t* tokenIds, int32_t* tokenCounts, int32_t* countLens,
    int32_t const* promptTokenIds, int32_t const* promptTokenCounts, int32_t const* promptLens,
    int32_t const* seqSlots, int32_t numRows, int32_t promptCapacity, int32_t countCapacity, int32_t vocabSize,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
