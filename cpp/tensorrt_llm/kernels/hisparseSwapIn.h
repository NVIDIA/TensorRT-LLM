/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Block/page-granular HiSparse host->device swap-in.
//
// Ported from SGLang HiSparse (python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh,
// load_cache_to_device_buffer_kernel). The original operates at token granularity for
// an MLA (single) KV tensor; here the "token" unit is reinterpreted as a fixed-size KV
// page/block (one MiniMax-M3 sparse block == one 128-token KV page) and the copy moves
// separate K and V paged pools. Each item is copied as a contiguous byte blob of
// itemSizeBytes, which covers FP8 (1 byte/element) and any other fixed-size page layout.
//
// One CUDA block processes one request. For every request it:
//   1) fast-path returns device slots directly when the request fits the hot buffer;
//   2) hash-matches selected blocks against the resident hot buffer (hits);
//   3) LRU-orders the hot buffer, choosing eviction victims for misses;
//   4) copies each missed page host->device (K then V) and writes its device slot.
//
// All problem sizes (numTopK, hotBufferSize, cudaBlockSize) are runtime parameters;
// shared memory is allocated dynamically to match, so no per-shape JIT is required.
//
// Indexing mirrors the SGLang contract:
//   - topKBlocks / topKDeviceLocs / seqLensBlocks / reqPoolIndices are indexed by the
//     in-batch request id (block/grid id), rows = numReqs.
//   - deviceBufferBlocks / deviceBufferLocs / hostBlockLocs / lruSlots are indexed by
//     the request-pool id rid = reqPoolIndices[bid], rows = maxReqs.
void invokeHiSparseSwapInBlocks(int32_t const* topKBlocks, int32_t* deviceBufferBlocks, int64_t const* hostBlockLocs,
    int32_t const* deviceBufferLocs, void const* hostCacheK, void const* hostCacheV, void* deviceBufferK,
    void* deviceBufferV, int32_t* topKDeviceLocs, void const* reqPoolIndices, bool reqPoolIndicesIsInt64,
    void const* seqLensBlocks, bool seqLensIsInt64, int16_t* lruSlots, int32_t const* numRealReqs, int32_t numReqs,
    int32_t numTopK, int32_t hotBufferSize, int64_t bufferStride0, int64_t hostStride, int64_t lruSlotStride0,
    int64_t topKStride, int64_t topKDeviceLocsStride, int64_t itemSizeBytes, int32_t cudaBlockSize,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
