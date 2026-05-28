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
/// Indexer TopK decode — GVR ↔ TPR dispatcher with three tiers, all
/// available for fp32, bf16, and fp16 inputs:
///   - GVR Heuristic        (preIdx provided, numRows >= 64 ∧ numColumns >= 32768, K ∈ {512,1024,2048})
///   - Single-block adaptive (numColumns < splitWork — sort algorithm picked
///                            at runtime inside topKPerRowJob)
///   - Multi-pass radix     (numColumns ≥ splitWork — 4 cooperative radix
///                           passes via DRAM scratch; pass 3's last block
///                           emits the final top-K inline. Requires `scratch`
///                           of at least `indexerTopKDecodeScratchBytes(numRows,
///                           numColumns, topK)` bytes.)
///
/// All TPR-family kernels accept `InputT` (fp32 / bf16 / fp16); logits are
/// cast to float at HBM-read sites and the histogram/sort run on float keys.
///
/// Required buffers for the split-work tier:
///   - scratch / scratchBytes: uint8 scratch sized via
///     `indexerTopKDecodeScratchBytes(numRows, numColumns, topK)`. Pass
///     nullptr when numColumns < splitWork. The caller may zero-init the
///     buffer once and reuse across compatible-shape calls.
///   - is_prefill: hint that the actual rows are tiny (lengths = [1..bs]).
///     Suppresses split-work entirely (single-block always handles prefill).
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0, int const preIdxCount = 0,
    float* heuristicScratch = nullptr, cudaStream_t const stream = 0, void* scratch = nullptr,
    size_t scratchBytes = 0, bool is_prefill = false);

/// Size in bytes of the `scratch` buffer required by `invokeIndexerTopKDecode`'s
/// multi-pass radix split-work tier for the given (numRows, numColumns, topK).
/// The buffer must be allocated by the caller whenever numColumns is at or
/// above the split-work threshold; it may be reused across calls of
/// compatible shape.
size_t indexerTopKDecodeScratchBytes(int numRows, int numColumns, int topK);

/// bf16 indexer TopK decode — same dispatcher contract as the fp32 entry.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices,
    int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, __nv_bfloat16* heuristicScratch = nullptr, cudaStream_t const stream = 0,
    void* scratch = nullptr, size_t scratchBytes = 0, bool is_prefill = false);

/// fp16 indexer TopK decode — same dispatcher contract as the fp32 entry.
void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0, int const preIdxCount = 0,
    __half* heuristicScratch = nullptr, cudaStream_t const stream = 0, void* scratch = nullptr,
    size_t scratchBytes = 0, bool is_prefill = false);

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK = 2048,
    cudaStream_t const stream = 0);

/// Returns true iff invokeIndexerTopKDecode would route to the GVR Heuristic
/// kernel for this (numRows, numColumns, topK) triple, assuming valid preIdx
/// is provided and stride1 == 1. Useful for callers that need to provision a
/// preIdx tensor or heuristicScratch buffer only when GVR will be selected.
///
/// The rule is identical for fp32 / bf16 / fp16 (all TPR tiers are
/// InputT-templated): GVR is preferred when K ∈ {512, 1024, 2048} AND
/// numColumns >= 32768 AND numRows >= 64.
///
/// @param numRows       logits rows (batch · next_n)
/// @param numColumns    logits columns (max sequence length)
/// @param topK          requested output size
/// @param bytesPerElem  retained for source compatibility; no longer used.
bool canIndexerTopKDecodeUseGvr(int numRows, int numColumns, int topK, int bytesPerElem = 4);

} // namespace kernels

TRTLLM_NAMESPACE_END
