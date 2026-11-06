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
namespace minimaxM3Sm107Tp4
{

//! Largest row (blocks per query) the long-row top-16 select covers: 9 keys per thread of one 1024-thread CTA.
constexpr int32_t kSelectLong2MaxBlocks = 9216;

//! MiniMax-M3 top-16 block select over per-KV-head scores [numKvHeads, numBlocks, totalQ] (element strides given),
//! one 1024-thread CTA per (query, KV head). Same contract as minimax_m3_select_blocks: init / local forcing, the
//! per-query valid-block count, ties to the smaller block, forced -inf picks emitted as -1, ascending ids with -1
//! tail padding. Writes out[query * outQueryStride + head * outHeadStride + 0..15]. The SM107 TP4 copy of
//! minimaxM3/selectLong with a choice of where the dependent grid (the sparse decode) is released: at this grid's exit
//! (triggerPhase 0), right after every score of the row is loaded (1), or, with earlyTrigger, right after this grid's
//! dependency wait. Every release point follows this grid's own dependency wait. fastTail ranks rows of at most 64
//! candidates on warp 0 and reads the valid count before the wait (the same picks); false keeps the all-thread rank
//! (dispatch.SELECT_LONG_SL3 = False).
cudaError_t invokeSelectLong2(float const* scores, int64_t headStride, int64_t blockStride, int64_t queryStride,
    int32_t const* nValidBlocks, int32_t* out, int64_t outQueryStride, int64_t outHeadStride, int32_t numKvHeads,
    int32_t numBlocks, int32_t totalQ, int32_t initBlocks, int32_t localBlocks, bool usePdl, bool earlyTrigger,
    int32_t triggerPhase, bool fastTail, cudaStream_t stream);

} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
