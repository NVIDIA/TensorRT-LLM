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

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Largest numTokens (rows of token_selected_experts) invokeKimiK3MoeSortKf accepts, and the multiple it requires (the
// rows of the DEP16 dispatch buffer are 16 x tokens-per-source-rank, so the ids are a whole number of 256-id blocks).
constexpr int32_t kKimiK3MoeSortKfMaxTokens = 8192;
constexpr int32_t kKimiK3MoeSortKfTokenMultiple = 16;

// moe_sort of one expert-parallel rank of Kimi K3 (top_k 16, 56 local experts [localExpertOffset, localExpertOffset +
// 56), tile 128) as one cluster launch: the same layout trtllm::moe_sort produces (entries of a local expert are
// consecutive rows, every expert is padded to a multiple of 128 rows, non-local entries of expandedIdxToPermutedIdx are
// -1, tile tables and the two scalars as in moe_sort). The order of the rows inside an expert is deterministic but not
// the ascending one. Outputs for numTokens rows: tileIdxToExpertIdx / tileIdxToMnLimit [maxTiles], maxTiles = 56 + (16
// * numTokens - 56) / 128; expandedIdxToPermutedIdx [numTokens, 16]; permutedIdxToExpandedIdx [maxTiles * 128];
// totalNumPaddedTokens / numNonExitingTiles [1]. tokenSelectedExperts [numTokens, 16] and expandedIdxToPermutedIdx
// must be 16-byte aligned. SM107 only (the kernel has not been run on other SM10x parts); numTokens a multiple of
// kKimiK3MoeSortKfTokenMultiple in [kKimiK3MoeSortKfTokenMultiple, kKimiK3MoeSortKfMaxTokens].
void invokeKimiK3MoeSortKf(int32_t const* tokenSelectedExperts, int32_t numTokens, int32_t localExpertOffset,
    int32_t* tileIdxToExpertIdx, int32_t* tileIdxToMnLimit, int32_t* expandedIdxToPermutedIdx,
    int32_t* permutedIdxToExpandedIdx, int32_t* totalNumPaddedTokens, int32_t* numNonExitingTiles, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
