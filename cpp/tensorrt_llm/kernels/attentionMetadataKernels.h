/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Backend-agnostic helpers for the per-iteration attention-metadata rebuild.
// Nothing here depends on a particular sparse-attention algorithm: they are the
// device-side forms of tensor patterns that several backends currently build
// with element-wise ATen chains on the host critical path.

// Upper bound on the scheduler batch handled by the single-block scan below.
constexpr int32_t kMaxTokenPositionScanBatch = 4096;

// Computes cu_seq_lens (optional), req_idx_per_token, and token_positions.
//
// Device-side form of:
//   cu_seq_lens      = pad(cumsum(seq_lens), (1, 0))
//   req_idx_per_token = repeat_interleave(arange(batch_size), seq_lens)
//   token_positions   = cached_tokens[req_idx] + (t - cu_seq_lens[req_idx])
// where the last line is the searchsorted(cu_seq_lens[1:], t, right=True) gather.
//
// `tokenPositions` may be null when only the request index is needed;
// `cachedTokens` is then unused. When `computeCuSeqLens` is false, `cuSeqLens`
// is read as an already-populated input and `batchSize` is not bounded by
// kMaxTokenPositionScanBatch.
void invokeComputeTokenPositions(int32_t const* seqLens, int32_t const* cachedTokens, int32_t* cuSeqLens,
    int32_t* reqIdxPerToken, int32_t* tokenPositions, int32_t batchSize, int32_t numTokens, bool computeCuSeqLens,
    cudaStream_t stream);

// Builds one shared-page block table from the host block-offset buffer.
//
// Device-side form of:
//   base = block_offsets[pool_id, copy_idx, 0, :]
//   out  = where(base == kBadPageIndex, kBadPageIndex, base * scale)
//
// `blockOffsets` is laid out [numPools, copyIdxCapacity, 2, maxBlocksPerSeq].
// Rows past `numTables` are left untouched, so padded CUDA-graph slots keep
// whatever the caller put there.
void invokeComputeSharedBlockTable(int32_t const* blockOffsets, int32_t const* copyIdx, int32_t* output,
    int32_t poolId, int32_t scale, int32_t copyIdxCapacity, int32_t numTables, int32_t maxBlocksPerSeq,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
