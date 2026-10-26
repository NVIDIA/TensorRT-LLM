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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! TAIL_BK3 (B): the S4FE position-0 Markov embedding in one launch, bitwise the in-tree
//! ``F.embedding(masked prev, w1).masked_fill(invalid, 0)``: out[i, :] = w1[prev[i], :] when 0 <= prev[i] < vocab, else
//! +0. prev: int64 [n]; w1: contiguous bf16 [vocab, rank]; out: contiguous bf16 [n, rank]. Graph-safe.
void invokeMarkovPrevEmbed(int64_t const* prev, int32_t n, __nv_bfloat16 const* w1, int64_t vocab, int32_t rank,
    __nv_bfloat16* out, cudaStream_t stream);

//! TAIL_BK3 (C): the block draft sampler's per-row parameters in one launch, bitwise the in-tree
//! ``request_*[start:end].repeat_interleave(K)`` and ``_rng_state_per_request(..., repeat=K, slot=s)``. For row
//! r = i * K + j (i < n requests, j < K): temps / topKs / topPs / minPs [r] = the request's value; seeds[r] = seeds[i];
//! offsets[r] = offsets[i] + slotOffset + j * span (int64, wrapping as torch does). Inputs are the request slices
//! (contiguous, n entries); outputs are contiguous [n * K]. Graph-safe.
void invokeExpandBlockSampling(float const* temps, int32_t const* topKs, float const* topPs, float const* minPs,
    int64_t const* seeds, int64_t const* offsets, int32_t n, int32_t K, int64_t slotOffset, int64_t span,
    float* tempsOut, int32_t* topKsOut, float* topPsOut, float* minPsOut, int64_t* seedsOut, int64_t* offsetsOut,
    cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
