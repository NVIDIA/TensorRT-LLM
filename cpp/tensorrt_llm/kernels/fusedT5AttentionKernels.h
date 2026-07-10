/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <cstddef>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// ---------------------------------------------------------------------------
// Fused T5 attention — encoder self-attention with additive relative position
// bias, fused as a single CUDA kernel.
//
// Fuses: QKV split + QK GEMM + T5 relative-bias add + online softmax
//        + SV GEMM + output transpose.
//
// Supports:
//   * dtype:       half, __nv_bfloat16 (when ENABLE_BF16)
//   * head_size:   32, 64, 128
//   * max_seq_len: up to kMaxSupportedSeqLen (2048 by default)
//   * num_buckets: multiples of 2, up to kMaxSupportedNumBuckets (128)
//   * SM:          80, 86, 89, 90 (WMMA fast path)
//                  70, 75         (SIMT reference fallback for correctness only)
//   * padding:     both padded [B, S, 3*H*D] and packed [num_tokens, 3*H*D]
//                  with cu_seqlens.
//   * attention:   bidirectional (T5 encoder). Causal T5 self-attention is
//                  not supported by this kernel and callers should fall back
//                  to the standard attention path.
//
// This kernel is intended as an OPTIONAL fast path. The default is disabled;
// enable via:
//   * runtime env  `TRTLLM_ENABLE_FUSED_T5_ATTENTION=1`, or
//   * per-call     `FusedT5AttentionParams::forceEnable = true`.
//
// If the requested shape/arch combination is not supported the runner will
// return `false` from `isSupported()` and callers MUST fall back to the
// legacy path.
// ---------------------------------------------------------------------------

// Maximum sequence length supported by the fused kernel. The T1 lookup table
// grows as O(2 * max_seq_len - 1) bytes, so this bound also caps that table.
constexpr int kFusedT5MaxSeqLen = 2048;

// Maximum bucket count. T5-base uses 32; some variants use 64.
constexpr int kFusedT5MaxNumBuckets = 128;

// Supported head sizes (compile-time specializations).
constexpr int kFusedT5HeadSize32  = 32;
constexpr int kFusedT5HeadSize64  = 64;
constexpr int kFusedT5HeadSize128 = 128;

// Legacy alias retained for downstream consumers. Prefer `kFusedT5MaxSeqLen`
// and `kFusedT5MaxNumBuckets` in new code.
constexpr int kFusedT5SeqLen     = kFusedT5MaxSeqLen;
constexpr int kFusedT5HeadSize   = kFusedT5HeadSize64;
constexpr int kFusedT5NumBuckets = 32;

// ---------------------------------------------------------------------------
// Parameter block — everything the runner needs to know for one launch.
// ---------------------------------------------------------------------------
struct FusedT5AttentionParams
{
    // Layout.
    int batchSize      = 0;
    int numHeads       = 0;
    int headSize       = 0;  // one of {32, 64, 128}
    int maxSeqLen      = 0;  // padded/max sequence length, S
    int numBuckets     = 32; // T5 bucket count (must be even)
    int maxDistance    = 128;// T5 max_distance (log-scale saturation)
    bool isBidirectional = true; // encoder self-attention

    // Runtime toggles.
    bool removePadding = false;  // true → variable-length packed input
    bool forceEnable   = false;  // true → bypass env-var gate
    bool forceSimt     = false;  // test-only → force the SIMT reference path

    // Scaling: typically 1/sqrt(head_size) / q_scaling.
    float qkScale = 0.f;
};

// ---------------------------------------------------------------------------
// Host-side utilities (unit-testable, no CUDA dependency at call site).
// ---------------------------------------------------------------------------

// Reference implementation of the T5 relative-position bucketing formula.
// Bit-exact match with HuggingFace `T5Attention._relative_position_bucket`
// in bidirectional mode.
int hostT5RelativeBucket(int relativePosition, int numBuckets, int maxDistance, bool bidirectional);

// Populate a host-side [2 * maxSeqLen - 1] bucket lookup table (T1). Values
// fit in int16_t because `numBuckets <= kFusedT5MaxNumBuckets`.
void hostBuildT5BucketTable(
    int16_t* table, int maxSeqLen, int numBuckets, int maxDistance, bool bidirectional);

// ---------------------------------------------------------------------------
// Device-side one-time initialization.
//
// Populates a caller-owned device buffer `bucketTable` of size
// `2 * params.maxSeqLen - 1` int16_t entries with the T5 bucket-id lookup.
// The buffer must remain alive for as long as the runner uses it (one buffer
// per (maxSeqLen, numBuckets, maxDistance, bidirectional) tuple).
//
// This replaces the previous `__constant__`-memory design, so multiple
// concurrent T5 models with different shapes can coexist within one process.
// ---------------------------------------------------------------------------
void initFusedT5BucketTable(
    int16_t* bucketTable,
    int maxSeqLen,
    int numBuckets,
    int maxDistance,
    bool bidirectional,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Explicit bias extraction — for callers that already have a precomputed
// [1, H, S, S] additive bias, this reduces it to the compact
// [H, numBuckets] representation the fused kernel consumes.
//
// The bucket table must have been initialized by `initFusedT5BucketTable`.
// ---------------------------------------------------------------------------
template <typename T>
void extractExplicitBiasToTable2(
    T const* explicitTable,          // [1, H, maxSeqLen, maxSeqLen]
    T* bucketBiasOut,                // [H, numBuckets]
    int16_t const* bucketTable,      // [2 * maxSeqLen - 1] from initFusedT5BucketTable
    int numHeads,
    int maxSeqLen,
    int numBuckets,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Runner — capability query + launch.
//
// Thread-safe with respect to `isSupported`; `run` is not thread-safe on the
// same runner instance from multiple host threads (typical of TRT-LLM Op
// wrappers, which own one runner per Op).
// ---------------------------------------------------------------------------
class FusedT5AttentionRunner
{
public:
    FusedT5AttentionRunner() = default;

    // Returns true iff the fused kernel can handle this parameter set on the
    // current device. Consults both the env-var gate and hard architectural
    // constraints (head_size / max_seq_len / SM version / bidirectional).
    static bool isSupported(FusedT5AttentionParams const& params);

    // Convenience: same as `isSupported` but ignores the env-var gate.
    static bool isShapeSupported(FusedT5AttentionParams const& params);

    // Launch the fused kernel. All device pointers must be valid on `stream`.
    //
    // Preconditions (checked with TLLM_CHECK_WITH_INFO):
    //   * `isSupported(params)` returns true.
    //   * `bucketTable` was populated by `initFusedT5BucketTable` with the
    //     same (maxSeqLen, numBuckets, maxDistance, bidirectional).
    //
    // Layouts:
    //   * QKV:         [numTokens, 3*H*D] if removePadding else [B, S, 3*H*D]
    //   * bucketBias:  [H, numBuckets]
    //   * out:         [numTokens, H*D]   if removePadding else [B, S, H*D]
    //   * inputLengths:[B]                actual (unpadded) lengths
    //   * cuSeqlens:   [B+1]              required when removePadding, else nullptr
    template <typename T>
    void run(
        FusedT5AttentionParams const& params,
        T const* qkv,
        T const* bucketBias,
        int16_t const* bucketTable,
        int const* inputLengths,
        int const* cuSeqlens,
        T* out,
        cudaStream_t stream) const;
};

} // namespace kernels

TRTLLM_NAMESPACE_END
