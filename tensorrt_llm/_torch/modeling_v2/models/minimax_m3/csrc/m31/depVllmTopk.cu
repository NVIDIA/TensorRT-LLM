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

#include "minimax_m3/csrc/m31/depVllmTopk.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

// The served DEP8 rows run past the upstream 8193-column contract: KV cache manager V2 pads max_blocks_per_seq to a
// multiple of 4 (8196 at a 1M max_seq_len with DSpark K7). One more block-family rung (36 keys per thread) covers
// 9216 candidates; a row takes the narrowest rung that covers it, so shorter rows rank as before.
#define M3_MAX_NVM 9216
#define M3_RUNGS 2, 4, 8, 12, 16, 20, 24, 28, 32, 36
#include "minimax_m3/csrc/msa_vllm_topk/m3_topk.cuh"

#include <limits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{

static_assert(kDepVllmTopkMaxColumns == ::minimaxM31VllmTopk::kMaxNvm + 1,
    "the DEP column cap is the vendored select's m3_max_n()");

cudaError_t invokeDepVllmTopk(float const* scores, int64_t headStride, int64_t rowStride, int32_t const* nValidBlocks,
    int32_t* out, int32_t numHeads, int32_t numRows, int32_t numColumns, bool usePdl, cudaStream_t stream)
{
    int64_t const rows = static_cast<int64_t>(numHeads) * numRows;
    if (rows == 0)
        return cudaSuccess;
    if (numHeads <= 0 || numRows <= 0 || rows > kDepVllmTopkMaxRows || numColumns <= 0
        || numColumns > kDepVllmTopkMaxColumns || rowStride < numColumns
        || rowStride > std::numeric_limits<int32_t>::max())
        return cudaErrorInvalidValue;
    ::minimaxM31VllmTopk::m3_launch(scores, nValidBlocks, out, numColumns, static_cast<int>(rowStride),
        static_cast<int>(rows), numRows, headStride, stream, usePdl);
    return cudaGetLastError();
}

} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
