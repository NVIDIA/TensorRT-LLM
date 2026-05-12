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
/// fp32 indexer TopK decode — L2-aware BS-threshold dispatcher with four
/// fallback tiers:
///   - GVR Heuristic    (preIdx provided, kSeqSmall ≤ N < splitWork, BS < kBsLarge, K ∈ {512,1024,2048})
///   - Insertion sort   (N < kSortingAlgorithmThreshold)
///   - Radix sort       (kSortingAlgorithmThreshold ≤ N < splitWork)
///   - Radix split-work (N ≥ splitWork — uses outLogitsAux / outIndicesAux)
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, float* heuristicScratch = nullptr, cudaStream_t const stream = 0);

/// bf16 indexer TopK decode — same dispatch axes as the fp32 entry, except
/// kBsL2 uses sizeof(__nv_bfloat16) bytes/elem (L2 footprint is half) and
/// the split-work tier is unsupported (the bf16/fp16 entry does not expose
/// the float aux buffers required for split-work). Insertion + radix tiers
/// share topKPerRowDecode with fp32 — histogram and sort run on float keys
/// after a static_cast<float>(InputT) at HBM-read sites.
///
/// Aborts with TLLM_CHECK if numColumns ≥ splitWorkThreshold; callers in
/// that regime must use the fp32 entry.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices,
    int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, __nv_bfloat16* heuristicScratch = nullptr, cudaStream_t const stream = 0);

/// fp16 indexer TopK decode — see bf16 overload for dispatcher contract.
void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0, int const preIdxCount = 0,
    __half* heuristicScratch = nullptr, cudaStream_t const stream = 0);

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK = 2048,
    cudaStream_t const stream = 0);

/// Returns true iff invokeIndexerTopKDecode would route to the GVR Heuristic
/// kernel for this (numRows, numColumns, topK) triple, assuming valid preIdx
/// is provided and stride1 == 1. Useful for callers that need to provision a
/// preIdx tensor or heuristicScratch buffer only when GVR will be selected.
///
/// Mirrors the gating logic of the dispatcher: K ∈ {512, 1024, 2048},
/// numColumns ∈ [kSeqSmall, splitWorkThreshold), numRows < kBsLarge, where
/// kBsLarge = min(kBsWave, kBsL2) and kBsL2 scales with bytesPerElem.
///
/// @param numRows         logits rows (batch · next_n)
/// @param numColumns      logits columns (max sequence length)
/// @param topK            requested output size
/// @param bytesPerElem    element size of logits (4 for fp32, 2 for bf16/fp16)
bool canIndexerTopKDecodeUseGvr(int numRows, int numColumns, int topK, int bytesPerElem = 4);

} // namespace kernels

TRTLLM_NAMESPACE_END
