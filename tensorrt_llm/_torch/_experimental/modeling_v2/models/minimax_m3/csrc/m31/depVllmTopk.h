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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{

//! Widest row (score columns) the vendored select ranks: its m3_max_n(), kMaxNvm + 1 (nvm counts the candidates
//! before the row's own block), with the DEP build's kMaxNvm of 9216. The served DEP8 bound is the KV manager's
//! max_blocks_per_seq, 8196 at a 1M max_seq_len with DSpark K7 (8193 blocks padded to a multiple of 4).
constexpr int32_t kDepVllmTopkMaxColumns = 9217;
//! Rows (heads x query rows) below the vendored select's float4 families, whose id writes are not sorted.
constexpr int32_t kDepVllmTopkMaxRows = 2047;

//! DEP8 decode top-16 select with vLLM's MiniMax-M3.1 ranker (vllm-project/MSA indexer_topk_select, vendored in
//! csrc/msa_vllm_topk): for KV head h and query row t, the scores of row t of group h lie at
//! scores + h * headStride + t * rowStride with unit column stride. nValidBlocks[t] (clamped to numColumns) counts the
//! row's candidate blocks, its own block last and forced; 15 more are ranked by a 16-bit key quantized over the row's
//! [min, max] (ties to the lower block); rows of at most 16 candidates give 0 .. n - 1 and -1. Writes the 16 ids
//! ascending to out[(h * numRows + t) * 16 + 0..15] (the head-major backing of a [rows, heads, 16] table). PDL: waits
//! for the scores' producer and releases its dependents right after.
cudaError_t invokeDepVllmTopk(float const* scores, int64_t headStride, int64_t rowStride, int32_t const* nValidBlocks,
    int32_t* out, int32_t numHeads, int32_t numRows, int32_t numColumns, bool usePdl, cudaStream_t stream);

} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
