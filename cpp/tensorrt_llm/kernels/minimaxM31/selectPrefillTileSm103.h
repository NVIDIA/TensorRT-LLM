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

// MiniMax-M3.1 GB300 (sm_103) target-local copy of the #20629 prefill tile select
// (cpp/tensorrt_llm/kernels/minimaxM3/selectPrefillTile.* @ cf8a77e816); namespace and op name are M3.1's.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimaxM31Sm103
{

//! Largest row (blocks per query) the prefill tile select stages in shared memory: 8 query rows of keys plus the
//! per-warp scratch stay under 200 KB.
constexpr int32_t kMinimaxM3SelectPrefillTileMaxBlocks = 5856;

//! MiniMax-M3 top-16 block select for prefill: per-KV-head scores [numKvHeads, numBlocks, totalQ] with unit query
//! stride, one CTA per (KV head, 8 consecutive queries) so every 32-byte sector of the scores is read once.
//!
//! Same contract as minimax_m3_select_blocks: per (query, head) row the clamp(nValidBlocks[query], 0, numBlocks)
//! leading blocks; the last localBlocks of them (counted from the unclamped count) forced to 1e29 and the first
//! initBlocks to 1e30, local winning; ranked by the descending ordered key of the float bits (NaN above +inf, +0 above
//! -0, negative NaN below -inf), ties to the smaller block; picks whose forced score is -inf emitted as -1; ascending
//! ids with -1 tail padding. Writes out[query * outQueryStride + head * outHeadStride + 0..15].
cudaError_t invokeMinimaxM3SelectPrefillTile(float const* scores, int64_t headStride, int64_t blockStride,
    int32_t const* nValidBlocks, int32_t* out, int64_t outQueryStride, int64_t outHeadStride, int32_t numKvHeads,
    int32_t numBlocks, int32_t totalQ, int32_t initBlocks, int32_t localBlocks, cudaStream_t stream);

} // namespace kernels::minimaxM31Sm103

TRTLLM_NAMESPACE_END
