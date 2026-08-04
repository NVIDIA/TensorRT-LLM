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

// Upper bound on the scheduler batch handled by the single-block scan.
constexpr int32_t kMaxTokenPositionScanBatch = 4096;

// Computes cu_seq_lens (optional), req_idx_per_token, and token_positions.
// `tokenPositions` may be null when only the request index is needed;
// `cachedTokens` is then unused. When `computeCuSeqLens` is false, `cuSeqLens`
// is read as an already-populated input.
void invokeDeepseekV4ComputeTokenPositions(int32_t const* seqLens, int32_t const* cachedTokens,
    int32_t* cuSeqLens, int32_t* reqIdxPerToken, int32_t* tokenPositions, int32_t batchSize,
    int32_t numTokens, bool computeCuSeqLens, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
