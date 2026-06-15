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
/// Indexer TopK decode. Three tiers:
///   - GVR Heuristic     (preIdx provided, K in {512,1024,2048}, numColumns in
///                       [kSeqSmall, splitWorkThreshold), numRows below the
///                       architecture-derived wave/L2 bound).
///   - Single-block     (numColumns < split-work threshold)
///   - Multi-pass radix (numColumns >= split-work threshold; requires
///                       `scratch` sized via indexerTopKDecodeScratchBytes,
///                       zero-init on first call and may be reused).
///
/// `is_prefill = true` forces single-block (split-work suppressed).
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0, int const preIdxCount = 0,
    float* heuristicScratch = nullptr, cudaStream_t const stream = 0, void* scratch = nullptr, size_t scratchBytes = 0,
    bool is_prefill = false);

/// Size of the multi-pass radix `scratch` buffer for these shapes.
size_t indexerTopKDecodeScratchBytes(int numRows, int numColumns, int topK);

/// bf16 overload; same contract.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices,
    int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, __nv_bfloat16* heuristicScratch = nullptr, cudaStream_t const stream = 0,
    void* scratch = nullptr, size_t scratchBytes = 0, bool is_prefill = false);

/// fp16 overload; same contract.
void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0, int const preIdxCount = 0,
    __half* heuristicScratch = nullptr, cudaStream_t const stream = 0, void* scratch = nullptr, size_t scratchBytes = 0,
    bool is_prefill = false);

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK = 2048,
    cudaStream_t const stream = 0);

/// True iff invokeIndexerTopKDecode would pick the GVR tier for this shape:
/// K in {512,1024,2048}, numColumns in [kSeqSmall, splitWorkThreshold), and
/// numRows below the architecture-derived wave/L2 bound. Lets callers
/// provision preIdx / heuristicScratch only when needed.
bool canIndexerTopKDecodeUseGvr(int numRows, int numColumns, int topK, int bytesPerElem = 4);

} // namespace kernels

TRTLLM_NAMESPACE_END
