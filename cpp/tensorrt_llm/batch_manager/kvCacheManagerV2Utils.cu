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

#include "kvCacheManagerV2Utils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
constexpr uint32_t copyBlockCtaSize = 128;
constexpr uint32_t copyBlocknbBufs = 2;
// Loads per thread of one block-offset copy CTA: each CTA copies 1024 entries of a row,
// so a long row is read from host memory by many CTAs in parallel.
constexpr uint32_t copyBlockItersPerCta = 2;

__device__ __host__ inline uint32_t divUp(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

void copyBasePageRowsToDevice(ITensor const& input, ITensor& output, SizeType32 numRows, CUstream stream)
{
    using namespace tensorrt_llm::runtime;

    auto const& srcShape = input.getShape();
    auto const& dstShape = output.getShape();
    constexpr SizeType32 kKvFactor = 2;
    TLLM_CHECK_WITH_INFO(srcShape.nbDims == 4 && dstShape.nbDims == 4,
        "Base-page copy expects 4-D input and output, got %d-D and %d-D tensors.", srcShape.nbDims, dstShape.nbDims);
    TLLM_CHECK_WITH_INFO(srcShape.d[0] == dstShape.d[0],
        "Base-page copy pool counts must match, got %d input pools and %d output pools.", srcShape.d[0], dstShape.d[0]);
    TLLM_CHECK_WITH_INFO(srcShape.d[2] == kKvFactor && dstShape.d[2] == kKvFactor,
        "Base-page copy expects K/V factor %d, got %d for input and %d for output.", kKvFactor, srcShape.d[2],
        dstShape.d[2]);
    TLLM_CHECK_WITH_INFO(srcShape.d[3] == dstShape.d[3],
        "Base-page copy block widths must match, got %d for input and %d for output.", srcShape.d[3], dstShape.d[3]);
    TLLM_CHECK_WITH_INFO(numRows >= 0 && numRows <= srcShape.d[1] && numRows <= dstShape.d[1],
        "Base-page copy row count %d must fit input capacity %d and output capacity %d.", numRows, srcShape.d[1],
        dstShape.d[1]);
    if (numRows == 0)
    {
        return;
    }

    auto const* srcPtr = input.getBufferPtr<SizeType32 const>();
    auto* dstPtr = output.getBufferPtr<SizeType32>();
    auto const numPools = srcShape.d[0];
    auto const numBlocks = srcShape.d[3];
    auto const rowStride = static_cast<size_t>(kKvFactor) * static_cast<size_t>(numBlocks);
    auto const srcPoolStride = static_cast<size_t>(srcShape.d[1]) * rowStride;
    auto const dstPoolStride = static_cast<size_t>(dstShape.d[1]) * rowStride;
    auto const rowBytes = static_cast<size_t>(numBlocks) * sizeof(SizeType32);
    auto const pitchBytes = static_cast<size_t>(rowStride) * sizeof(SizeType32);
    for (SizeType32 pool = 0; pool < numPools; ++pool)
    {
        TLLM_CUDA_CHECK(cudaMemcpy2DAsync(dstPtr + pool * dstPoolStride, pitchBytes, srcPtr + pool * srcPoolStride,
            pitchBytes, rowBytes, numRows, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    }
}

// dst_tensor[:, :num_seqs, 0] = src_tensor[:, copy_idx]
// dst_tensor[:, :num_seqs, 1] = dst_tensor[:, :num_seqs, 0] + 1
// Grid: (pool, sequence, chunk of copyBlockItersPerCta loads per thread along the row).
__global__ void copyBatchBlockOffsetsToDeviceKernel(SizeType32 const* __restrict__ srcPtr,
    SizeType32* __restrict__ dstPtr, SizeType32 const srcMaxNumSequences, SizeType32 const dstMaxNumSequences,
    SizeType32 numBlocksPerSeq, SizeType32 const* __restrict__ copyIndex, SizeType32 const* __restrict__ indexScales,
    SizeType32 const* __restrict__ kvOffset)
{
    constexpr uint32_t kvFactor = 2;
    constexpr auto elemPerAccess = sizeof(PackedInt) / sizeof(SizeType32);

    __shared__ PackedInt data[copyBlocknbBufs][copyBlockCtaSize];

    auto const iterPerSeq = divUp(numBlocksPerSeq * sizeof(SizeType32), sizeof(PackedInt) * copyBlockCtaSize);
    auto const iterBeg = blockIdx.z * copyBlockItersPerCta;
    auto const numIters = min(copyBlockItersPerCta, iterPerSeq - iterBeg);
    auto const tid = threadIdx.x;
    auto const poolIdx = blockIdx.x;
    auto const seqIdx = blockIdx.y;
    auto const seqDimStride = kvFactor * numBlocksPerSeq;
    uint32_t const chunkBeg = iterBeg * copyBlockCtaSize * elemPerAccess;
    uint32_t const srcIdxBeg
        = chunkBeg + tid * elemPerAccess + (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride;
    uint32_t const dstIdxKBeg = chunkBeg + tid * elemPerAccess + (poolIdx * dstMaxNumSequences + seqIdx) * seqDimStride;
    uint32_t const dstIdxVBeg = dstIdxKBeg + numBlocksPerSeq;

    uint32_t const srcIdxEnd = (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride + numBlocksPerSeq;

    for (uint32_t i = 0; i < numIters + copyBlocknbBufs; i++)
    {
        uint32_t const idxBuf = i % copyBlocknbBufs;
        if (i >= copyBlocknbBufs)
        {
            uint32_t const stIter = i - copyBlocknbBufs;
            assert(idxBuf == (stIter % copyBlocknbBufs));
            auto const offset = copyBlockCtaSize * stIter * elemPerAccess;
            SizeType32 const srcIdx = srcIdxBeg + offset;
            SizeType32 const dstIdxK = dstIdxKBeg + offset;
            SizeType32 const dstIdxV = dstIdxVBeg + offset;
            PackedInt const& src = data[idxBuf][tid];
            PackedInt& dstK = *reinterpret_cast<PackedInt*>(dstPtr + dstIdxK);
            PackedInt& dstV = *reinterpret_cast<PackedInt*>(dstPtr + dstIdxV);
            asm volatile("cp.async.wait_group %0;\n" ::"n"(copyBlocknbBufs - 1) : "memory");
            if (srcIdx < srcIdxEnd)
            {
#pragma unroll
                for (uint32_t j = 0; j < elemPerAccess; j++)
                {
                    auto const val = src.unpacked[j];
                    dstK.unpacked[j] = (val == BAD_PAGE_INDEX) ? 0 : (indexScales[poolIdx] * val);
                    dstV.unpacked[j] = (val == BAD_PAGE_INDEX) ? 0 : (indexScales[poolIdx] * val + kvOffset[poolIdx]);
                }
            }
        }
        uint32_t const ldIter = i;
        PackedInt* const dst = &data[idxBuf][tid];
        uint32_t const srcIdx = srcIdxBeg + copyBlockCtaSize * ldIter * elemPerAccess;
        PackedInt const* const src = reinterpret_cast<PackedInt const*>(srcPtr + srcIdx);
        if (ldIter < numIters && srcIdx < srcIdxEnd)
        {
            uint32_t const size = sizeof(PackedInt);
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"l"(__cvta_generic_to_shared(dst)),
                         "l"(src), "n"(size), "r"(size)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;\n" : : : "memory");
    }
}

// Host-side launcher
void copyBatchBlockOffsetsToDevice(ITensor const& input, ITensor& output, ITensor const& copyIndex,
    ITensor const& indexScales, ITensor const& kvOffset, CUstream stream) noexcept
{
    using namespace tensorrt_llm::runtime;

    auto const* srcPtr = input.getBufferPtr<tk::KVCacheIndex::UnderlyingType const>();
    // [numPools, maxNumSequences, kvFactor, numBlocksPerSeq]
    auto* dstPtr = output.getBufferPtr<tk::KVCacheIndex::UnderlyingType>();
    auto const* copyIndexPtr = copyIndex.getBufferPtr<SizeType32 const>();
    auto const* indexScalesPtr = indexScales.getBufferPtr<SizeType32 const>();
    auto const* kvOffsetPtr = kvOffset.getBufferPtr<SizeType32 const>();
    auto const& srcShape = input.getShape();
    auto const& dstShape = output.getShape();
    auto const& copyIndexShape = copyIndex.getShape();

    TLLM_CHECK(srcShape.nbDims == 4); // [numPools, srcMaxNumSequences, kvFactor, numBlocksPerSeq]
    TLLM_CHECK(dstShape.nbDims == 4); // [numPools, dstMaxNumSequences, kvFactor, numBlocksPerSeq]

    SizeType32 numPools = srcShape.d[0];
    SizeType32 srcMaxNumSequences = srcShape.d[1];
    SizeType32 dstMaxNumSequences = dstShape.d[1];
    SizeType32 numBlocksPerSeq = srcShape.d[3];
    SizeType32 numSeqs = copyIndexShape.d[0];

    if (numSeqs == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO((numBlocksPerSeq * sizeof(SizeType32)) % sizeof(PackedInt) == 0,
        "Not implemented case: numBlocksPerSeq * sizeof(SizeType32) = %zu must be a multiple of %zu.",
        static_cast<size_t>(numBlocksPerSeq * sizeof(SizeType32)), static_cast<size_t>(sizeof(PackedInt)));

    auto const iterPerSeq = divUp(static_cast<uint32_t>(numBlocksPerSeq * sizeof(SizeType32)),
        static_cast<uint32_t>(sizeof(PackedInt) * copyBlockCtaSize));
    auto const numChunks = divUp(iterPerSeq, copyBlockItersPerCta);
    if (numChunks == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(
        numChunks <= 65535, "numBlocksPerSeq = %d needs %u CTAs per row, more than 65535.", numBlocksPerSeq, numChunks);
    dim3 gridDim(numPools, numSeqs, numChunks);
    dim3 blockDim(copyBlockCtaSize);

    copyBatchBlockOffsetsToDeviceKernel<<<gridDim, blockDim, 0, stream>>>(srcPtr, dstPtr, srcMaxNumSequences,
        dstMaxNumSequences, numBlocksPerSeq, copyIndexPtr, indexScalesPtr, kvOffsetPtr);
}

// Build per-layer K/V page tables for the active batch. Each thread owns one
// (pool, row, block) entry of the base table: it reads that entry once and
// writes it to every K/V lane stored in the pool, applying the lane's scale
// and offset and preserving BAD_PAGE_INDEX. The input may be pinned host
// memory, where every read crosses the host link, so an entry must not be
// re-read per lane. Only the first maxBlocks columns of each row are written.
__global__ void copyPerLayerBlockOffsetsToDeviceKernel(SizeType32 const* __restrict__ input,
    SizeType32* __restrict__ output, SizeType32 const* __restrict__ copyIndex,
    SizeType32 const* __restrict__ converters, SizeType32 const* __restrict__ poolLaneOffsets,
    SizeType32 const* __restrict__ poolLanes, int64_t sourceRows, int64_t outputRows, int64_t tableWidth,
    int64_t numSeqs, int64_t maxBlocks)
{
    auto const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numSeqs * maxBlocks)
    {
        return;
    }
    auto const row = index / maxBlocks;
    auto const block = index % maxBlocks;
    auto const pool = static_cast<int64_t>(blockIdx.y);
    auto const base = input[(pool * sourceRows + copyIndex[row]) * 2 * tableWidth + block];
    for (auto i = poolLaneOffsets[pool]; i < poolLaneOffsets[pool + 1]; ++i)
    {
        auto const lane = poolLanes[i];
        auto const scale = converters[lane * 3 + 1];
        auto const offset = converters[lane * 3 + 2];
        auto const layer = static_cast<int64_t>(lane / 2);
        auto const role = lane % 2;
        output[((layer * outputRows + row) * 2 + role) * tableWidth + block]
            = base == BAD_PAGE_INDEX ? BAD_PAGE_INDEX : base * scale + offset;
    }
}

void copyPerLayerBlockOffsetsToDevice(ITensor const& input, ITensor& output, ITensor const& copyIndex,
    ITensor const& converters, ITensor const& poolLaneOffsets, ITensor const& poolLanes, SizeType32 maxBlocks,
    CUstream stream)
{
    constexpr int64_t kThreads = 256;
    auto const& sourceShape = input.getShape();
    auto const& outputShape = output.getShape();
    auto const& converterShape = converters.getShape();
    auto const& copyIndexShape = copyIndex.getShape();
    TLLM_CHECK_WITH_INFO(
        sourceShape.nbDims == 4 && outputShape.nbDims == 4 && sourceShape.d[2] == 2 && outputShape.d[2] == 2,
        "input and output must be [tables, rows, 2, blocks]");
    TLLM_CHECK_WITH_INFO(sourceShape.d[3] == outputShape.d[3], "input and output block widths must match");
    TLLM_CHECK_WITH_INFO(converterShape.nbDims == 3 && converterShape.d[0] == outputShape.d[0]
            && converterShape.d[1] == 2 && converterShape.d[2] == 3,
        "converters must be [output tables, 2, 3]");
    auto const numPools = sourceShape.d[0];
    TLLM_CHECK_WITH_INFO(poolLaneOffsets.getShape().nbDims == 1 && poolLaneOffsets.getShape().d[0] == numPools + 1,
        "pool_lane_offsets must hold one entry per input pool plus one");
    TLLM_CHECK_WITH_INFO(poolLanes.getShape().nbDims == 1 && poolLanes.getShape().d[0] == outputShape.d[0] * 2,
        "pool_lanes must list every output K/V lane once");
    TLLM_CHECK_WITH_INFO(maxBlocks >= 0 && maxBlocks <= sourceShape.d[3], "max_blocks %d exceeds the table width %ld",
        maxBlocks, static_cast<long>(sourceShape.d[3]));
    TLLM_CHECK_WITH_INFO(copyIndexShape.nbDims == 1, "copy_index must be one-dimensional");
    auto const numSeqs = static_cast<int64_t>(copyIndexShape.d[0]);
    TLLM_CHECK_WITH_INFO(numSeqs <= outputShape.d[1], "output must have at least one row per copy_index entry");
    auto const count = numSeqs * maxBlocks;
    if (count == 0 || numPools == 0)
    {
        return;
    }
    auto const numCtas = (count + kThreads - 1) / kThreads;
    TLLM_CHECK_WITH_INFO(numCtas <= std::numeric_limits<int32_t>::max() && numPools <= 65535,
        "Per-layer page-table copy exceeds the CUDA grid limits");
    dim3 const grid(static_cast<uint32_t>(numCtas), static_cast<uint32_t>(numPools));
    copyPerLayerBlockOffsetsToDeviceKernel<<<grid, kThreads, 0, stream>>>(input.getBufferPtr<SizeType32 const>(),
        output.getBufferPtr<SizeType32>(), copyIndex.getBufferPtr<SizeType32 const>(),
        converters.getBufferPtr<SizeType32 const>(), poolLaneOffsets.getBufferPtr<SizeType32 const>(),
        poolLanes.getBufferPtr<SizeType32 const>(), sourceShape.d[1], outputShape.d[1], sourceShape.d[3], numSeqs,
        maxBlocks);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
