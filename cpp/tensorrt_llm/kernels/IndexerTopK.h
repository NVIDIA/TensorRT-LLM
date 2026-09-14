/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
// Number of blocks-per-row used by the multi-block split + merge dispatch path of
// invokeIndexerTopKDecode. Returns 1 when the single-block path is preferred.
// Callers that allocate aux buffers must use this same helper to size them, and
// must pass the same splitWorkThreshold they will pass to invokeIndexerTopKDecode
// (a value <= 0 selects the internal default).
int computeIndexerTopKDecodeBlocksPerRow(int numRows, int numColumns, int splitWorkThreshold = 0);

/// fp32 indexer TopK decode — three dispatch tiers:
///   - Insertion sort   (N < kSortingAlgorithmThreshold)
///   - Radix sort       (kSortingAlgorithmThreshold ≤ N < splitWork)
///   - Radix split-work (N ≥ splitWork — uses outLogitsAux / outIndicesAux)
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK = 2048, int const compressRatio = 1,
    cudaStream_t const stream = 0);

/// bf16 indexer TopK decode — same dispatch tiers as the fp32 entry, except
/// the split-work tier is unsupported (the bf16/fp16 entry does not expose
/// the float aux buffers required for split-work). Insertion + radix tiers
/// share topKPerRowDecode with fp32 — histogram and sort run on float keys
/// after a static_cast<float>(InputT) at HBM-read sites.
///
/// Aborts with TLLM_CHECK if numColumns ≥ splitWorkThreshold; callers in
/// that regime must use the fp32 entry.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices,
    int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const next_n, int const topK = 2048, int const compressRatio = 1, cudaStream_t const stream = 0);

/// fp16 indexer TopK decode — see bf16 overload for dispatcher contract.
void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const compressRatio = 1, cudaStream_t const stream = 0);

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK = 2048,
    cudaStream_t const stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
