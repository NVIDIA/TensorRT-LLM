/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
/// fp32 indexer TopK decode — Scheme X v1.2 dispatcher (GVR Heuristic /
/// Insertion / Radix / Radix-split-work). Production V2e path.
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, float* heuristicScratch = nullptr, cudaStream_t const stream = 0);

/// bf16 indexer TopK decode (4d_multi_dtype_unified, 2026-05-06).
/// GVR-Heuristic path only — bf16/fp16 inputs have no radix fallback (the
/// production radix kernel is fp32-only). Caller MUST satisfy GVR conditions
/// (preIdx provided, numRows < kBsLarge, numColumns ∈ [kSeqSmall, kSplitWork))
/// or cast to fp32 first.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices,
    int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, __nv_bfloat16* heuristicScratch = nullptr, cudaStream_t const stream = 0);

/// fp16 indexer TopK decode (4d_multi_dtype_unified, 2026-05-06). See bf16 doc.
void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0, int const preIdxCount = 0,
    __half* heuristicScratch = nullptr, cudaStream_t const stream = 0);

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK = 2048,
    cudaStream_t const stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
