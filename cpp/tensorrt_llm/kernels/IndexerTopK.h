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
/// Indexer TopK decode — GVR ↔ TPR dispatcher with five fallback tiers, all
/// available for fp32, bf16, and fp16 inputs:
///   - GVR Heuristic      (preIdx provided, numColumns > 16384 ∧ numRows > 32, K ∈ {512,1024,2048})
///   - Multi-pass radix   (scratch provided, low-bs/long-seq corner; opt-in)
///   - Insertion sort     (N < kSortingAlgorithmThreshold, see .cu file)
///   - Single-block radix (kSortingAlgorithmThreshold ≤ N < splitWork)
///   - Split-work radix   (N ≥ splitWork — uses outLogitsAux / outIndicesAux,
///                         which stay fp32 regardless of input dtype.
///                         Fused single-launch when doneCounterScratch != nullptr,
///                         else legacy 2-launch part1 + part2.)
///
/// All TPR-family kernels accept `InputT` (fp32 / bf16 / fp16); logits are
/// cast to float at HBM-read sites and the histogram/sort run on float keys.
/// Aux buffers (`outLogitsAux` for split-work, `scratch` for multi-pass radix)
/// stay fp32 across all three entries.
///
/// Optional buffers (defaults preserve original behavior):
///   - doneCounterScratch: one int per row, zero-initialized. When provided,
///     the N ≥ splitWork tier uses a fused single-launch variant where the
///     last block in each row performs an in-kernel merge. Up to 20% faster
///     than the 2-launch part1+part2 path on H100/B200 for typical BS×N
///     shapes (BS≥1, N≥splitWork).
///   - scratch / scratchBytes: optional uint8 scratch for the multi-pass
///     radix path used at low-bs / long-seq decode shapes
///     (BS≤32 / N≥65k, BS≤64 / N≥131k, BS≤256 / N≥524k). When non-null AND
///     !is_prefill AND the shape is in the multi-pass-radix-eligible zone,
///     the kernel uses this path instead of the single-block radix.
///     Required size can be queried via `indexerTopKDecodeScratchBytes`.
///   - is_prefill: hint that the actual rows are tiny (lengths = [1..bs]).
///     Suppresses the multi-pass radix path; the fused / single-block paths'
///     short-row short-circuit handles tiny rows faster.
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, float* heuristicScratch = nullptr, int* doneCounterScratch = nullptr,
    cudaStream_t const stream = 0, void* scratch = nullptr, size_t scratchBytes = 0, bool is_prefill = false);

/// Size in bytes of the `scratch` buffer required by `invokeIndexerTopKDecode`
/// for the given (numRows, numColumns, topK). The buffer must be allocated by
/// the caller and may be re-used across calls of compatible shape. Pass
/// nullptr for `scratch` if you don't intend to use the multi-pass radix path.
size_t indexerTopKDecodeScratchBytes(int numRows, int numColumns, int topK);

/// bf16 indexer TopK decode — same dispatcher contract as the fp32 entry.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, __nv_bfloat16* heuristicScratch = nullptr, int* doneCounterScratch = nullptr,
    cudaStream_t const stream = 0, void* scratch = nullptr, size_t scratchBytes = 0, bool is_prefill = false);

/// fp16 indexer TopK decode — same dispatcher contract as the fp32 entry.
void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, __half* heuristicScratch = nullptr, int* doneCounterScratch = nullptr,
    cudaStream_t const stream = 0, void* scratch = nullptr, size_t scratchBytes = 0, bool is_prefill = false);

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
/// numColumns > 16384 AND numRows > 32.
///
/// @param numRows       logits rows (batch · next_n)
/// @param numColumns    logits columns (max sequence length)
/// @param topK          requested output size
/// @param bytesPerElem  retained for source compatibility; no longer used.
bool canIndexerTopKDecodeUseGvr(int numRows, int numColumns, int topK, int bytesPerElem = 4);

} // namespace kernels

TRTLLM_NAMESPACE_END
