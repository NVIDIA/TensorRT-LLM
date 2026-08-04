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

// DeepSeek-V4 restricts compression ratios to {1, 4, 128}; one slot spare.
constexpr int32_t kMaxCompressRatios = 4;
// Upper bound on the scheduler batch handled by the single-block shared-memory
// scan (2 * 4096 * 4B = 32KB static shared memory, within the 48KB budget).
constexpr int32_t kMaxScanBatch = 4096;

// Per-ratio buffers are persistent across iterations, and there are at most 4
// ratios, so they travel by value in the kernel launch packet. This keeps the
// whole step host-copy-free -- passing device pointer arrays would need an H2D
// memcpy per call and negate the point of the port.
struct PerRatioKvLensParams
{
    int32_t* compressedKvLens[kMaxCompressRatios];
    int32_t* pastKvLens[kMaxCompressRatios];
    int32_t* newCompKvLens[kMaxCompressRatios];
    int32_t* cuNewCompKv[kMaxCompressRatios];
    int32_t ratios[kMaxCompressRatios];
};

struct CompressedMaskParams
{
    int32_t const* newCompKvLens[kMaxCompressRatios];
    int32_t const* cuNewCompKv[kMaxCompressRatios];
    bool* mask[kMaxCompressRatios];
    int32_t totalTokens[kMaxCompressRatios];
};

// Shared by the context and generation position-id kernels. `counts` is the
// per-ratio element count; `offsets` is the compact output offset (generation
// only, zero for context).
struct CompressedPositionIdsParams
{
    int32_t const* pastKvLens[kMaxCompressRatios];
    int32_t const* cuNewCompKv[kMaxCompressRatios];
    int32_t* positionIds[kMaxCompressRatios];
    int32_t ratios[kMaxCompressRatios];
    int32_t counts[kMaxCompressRatios];
    int32_t offsets[kMaxCompressRatios];
};

void invokeDeepseekV4ComputePerRatioKvLens(int32_t const* kvLens, int32_t const* cachedTokens,
    PerRatioKvLensParams const& params, int32_t numRatios, int32_t batchSize, cudaStream_t stream);

void invokeDeepseekV4ComputeCompressedMask(CompressedMaskParams const& params, int32_t maxTotalTokens,
    int32_t numRatios, int32_t batchSize, cudaStream_t stream);

void invokeDeepseekV4ComputeCtxCompressedPositionIds(CompressedPositionIdsParams const& params, int32_t maxCount,
    int32_t numRatios, int32_t numContexts, cudaStream_t stream);

void invokeDeepseekV4ComputeGenCompressedPositionIds(CompressedPositionIdsParams const& params, int32_t maxCount,
    int32_t numRatios, int32_t numContexts, int32_t batchSize, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
