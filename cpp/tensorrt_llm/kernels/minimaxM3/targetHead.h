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
namespace minimaxM3
{
//! The MiniMax-M3 target lm_head shard GEMV over a lossless 12-bit coding of its BF16 weight (verify rows <= 8,
//! sm_103a): BF16 x BF16 MMAs with FP32 accumulation over the full K, one BF16 rounding per logit. See targetHead.cu.
namespace targetHead
{

constexpr int kMaxRows = 8; // activation rows (verify rows) of one launch
constexpr int kWarps = 16;  // warps per CTA
constexpr int kStages = 3;  // coded units in flight per warp (cp.async ring)

struct Params
{
    uint4 const* w;          // [tiles][K / 64][3][32] coded weight (see targetHead.cu)
    __nv_bfloat16 const* x;  // [M, K] activation rows
    long long x_stride;      // elements between rows of x
    __nv_bfloat16* out;      // [M, N] logits
    long long out_stride;    // elements between rows of out
    float* part;             // [grid * kWarps][2][4 * 32] partial sums of tiles split between warps
    int* counter;            // [tiles] finishers per split tile, zero between calls (re-armed by the kernel)
    int const* exc_off;      // [tiles + 1] exception list offsets
    uint32_t const* exc_pos; // row in tile << 16 | k, per exception
    float2 const* exc_w;     // {w, 0} per exception: the exact weight
    uint32_t exp_base;       // exponent base << 7 in both BF16 halves (base << 7 | base << 23)
    int M;                   // rows, 1..kMaxRows
    int N;                   // logits per row (shard rows)
};

//! Shared memory a launch over hidden size K takes (the staged activation and the cp.async rings).
size_t targetHeadSmemBytes(int K);

//! Launch over `grid` 512-thread CTAs (K a multiple of 64).
cudaError_t invokeTargetHead(Params const& p, int K, int grid, bool pdl, cudaStream_t stream);

} // namespace targetHead
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
