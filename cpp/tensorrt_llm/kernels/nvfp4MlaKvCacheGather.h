/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstddef>
#include <cstdint>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeNvFp4MlaKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool, int32_t const* globalIndices,
    __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale, int32_t numRows, int32_t topK,
    int32_t headDim, int32_t residualDim, int64_t numPoolTokens, cudaStream_t stream = 0);

//! One layer of invokeNvFp4MlaKvCacheRestore. Every member is 64-bit, so a contiguous int64 tensor of shape
//! [layers, 10] holds the array.
struct NvFp4MlaKvCacheRestoreLayer
{
    int64_t dataPool;          //!< Address of the layer's NVFP4 data rows.
    int64_t scalePool;         //!< Address of their E4M3 scales.
    int64_t high;              //!< Address of the layer's FP8 high rows, head_dim bytes each.
    int64_t numPoolTokens;     //!< Rows the data pool holds.
    int64_t sourceTable;       //!< Page table that maps the layer's source pages.
    int64_t highTable;         //!< Page table that maps its high pages.
    int64_t sourceRowsPerPage; //!< Source rows one source page index spans.
    int64_t sourceRowOffset;   //!< The layer's first row within a source page.
    int64_t rowsPerPage;       //!< Rows one page holds.
    int64_t tokensPerRow;      //!< Tokens one row holds: 1, or a compression ratio.
};

//! Decode the whole rows of tokens [tokenBegin, tokenEnd) of every layer from its NVFP4 pool into its FP8 high
//! rows in one launch, as invokeNvFp4MlaKvCacheGather decodes them. Row r of a layer lies in page r / rowsPerPage
//! of its two page tables; pageTables is a [tables, maxPages] array.
void invokeNvFp4MlaKvCacheRestore(NvFp4MlaKvCacheRestoreLayer const* layers, int32_t numLayers,
    int32_t const* pageTables, int32_t maxPages, int32_t tokenBegin, int32_t tokenEnd, int32_t headDim,
    int32_t residualDim, float dequantScale, cudaStream_t stream = 0);

size_t getNvFp4MlaContextKvCacheGatherDirectWorkspaceSize(
    int32_t numRequests, int32_t maxKvTokens, cudaStream_t stream = 0);

void invokeNvFp4MlaContextKvCacheGatherDirect(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* localTopKIndices, int32_t const* queryReqIndices, int32_t const* compressedKvLengths,
    int32_t* globalIndices, __nv_fp8_e4m3* output, float const* globalDequantScale, void* workspace,
    size_t workspaceSize, int32_t numQueryRows, int32_t topK, int32_t numRequests, int32_t maxKvTokens,
    int32_t outputCapacity, int32_t headDim, int32_t residualDim, int64_t numPoolTokens, cudaStream_t stream = 0);

size_t getNvFp4MlaContextKvCacheGatherWorkspaceSize(int32_t totalKvTokens, cudaStream_t stream = 0);

void invokeNvFp4MlaContextKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* localTopKIndices, int32_t const* queryReqIndices, int32_t const* blockTable,
    int64_t const* cuKvLengths, __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale,
    void* workspace, size_t workspaceSize, int32_t numQueryRows, int32_t topK, int32_t numRequests,
    int32_t maxBlocksPerRequest, int32_t totalKvTokens, int32_t outputCapacity, int32_t tokensPerBlock,
    int32_t pageStride, int32_t layerId, int32_t headDim, int32_t residualDim, int64_t numPoolTokens,
    cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
