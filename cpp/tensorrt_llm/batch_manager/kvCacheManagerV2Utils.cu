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
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
constexpr uint32_t copyBlockCtaSize = 128;
constexpr uint32_t copyBlocknbBufs = 2;

__device__ __host__ inline uint32_t divUp(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// dst_tensor[:, :num_seqs, 0] = src_tensor[:, copy_idx]
// dst_tensor[:, :num_seqs, 1] = dst_tensor[:, :num_seqs, 0] + 1
__global__ void copyBatchBlockOffsetsToDeviceKernel(SizeType32 const* __restrict__ srcPtr,
    SizeType32* __restrict__ dstPtr, SizeType32 const srcMaxNumSequences, SizeType32 const dstMaxNumSequences,
    SizeType32 numBlocksPerSeq, SizeType32 const* __restrict__ copyIndex, SizeType32 const* __restrict__ indexScales,
    SizeType32 const* __restrict__ kvOffset)
{
    constexpr uint32_t kvFactor = 2;
    constexpr auto elemPerAccess = sizeof(PackedInt) / sizeof(SizeType32);

    __shared__ PackedInt data[copyBlocknbBufs][copyBlockCtaSize];

    auto const iterPerSeq = divUp(numBlocksPerSeq * sizeof(SizeType32), sizeof(PackedInt) * copyBlockCtaSize);
    auto const tid = threadIdx.x;
    auto const poolIdx = blockIdx.x;
    auto const seqIdx = blockIdx.y;
    auto const seqDimStride = kvFactor * numBlocksPerSeq;
    uint32_t const srcIdxBeg = tid * elemPerAccess + (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride;
    uint32_t const dstIdxKBeg = tid * elemPerAccess + (poolIdx * dstMaxNumSequences + seqIdx) * seqDimStride;
    uint32_t const dstIdxVBeg = dstIdxKBeg + numBlocksPerSeq;

    uint32_t const srcIdxEnd = (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride + numBlocksPerSeq;

    for (uint32_t i = 0; i < iterPerSeq + copyBlocknbBufs; i++)
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
        if (srcIdx < srcIdxEnd)
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

    dim3 gridDim(numPools, numSeqs, 1);
    dim3 blockDim(copyBlockCtaSize);

    copyBatchBlockOffsetsToDeviceKernel<<<gridDim, blockDim, 0, stream>>>(srcPtr, dstPtr, srcMaxNumSequences,
        dstMaxNumSequences, numBlocksPerSeq, copyIndexPtr, indexScalesPtr, kvOffsetPtr);
}

// Build per-layer K/V page tables for the active batch directly from pinned
// host indices, avoiding a full-table device upload. Each entry selects its
// source pool and request row, applies the layer/role's scale and offset,
// and preserves BAD_PAGE_INDEX. The copy writes at most maxBlocks columns
// per row and runs only when the batch has no active scratch slots.
__global__ void copyPerLayerBlockOffsetsToDeviceKernel(SizeType32 const* __restrict__ input,
    SizeType32* __restrict__ output, SizeType32 const* __restrict__ copyIndex,
    SizeType32 const* __restrict__ converters, SizeType32 sourceRows, SizeType32 outputRows, SizeType32 tableWidth,
    SizeType32 numLayers, SizeType32 numSeqs, SizeType32 maxBlocks)
{
    auto const index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= numLayers * numSeqs * 2 * maxBlocks)
    {
        return;
    }
    auto const block = index % maxBlocks;
    auto const role = index / maxBlocks % 2;
    auto const row = index / (maxBlocks * 2) % numSeqs;
    auto const layer = index / (maxBlocks * 2 * numSeqs);
    auto const lane = layer * 2 + role;
    auto const pool = converters[lane * 3];
    auto const scale = converters[lane * 3 + 1];
    auto const offset = converters[lane * 3 + 2];
    auto const base = input[(static_cast<int64_t>(pool) * sourceRows + copyIndex[row]) * 2 * tableWidth + block];
    auto const destination = ((static_cast<int64_t>(layer) * outputRows + row) * 2 + role) * tableWidth + block;
    output[destination] = base == BAD_PAGE_INDEX ? BAD_PAGE_INDEX : base * scale + offset;
}

void copyPerLayerBlockOffsetsToDevice(ITensor const& input, ITensor& output, ITensor const& copyIndex,
    ITensor const& converters, SizeType32 maxBlocks, CUstream stream) noexcept
{
    auto const& sourceShape = input.getShape();
    auto const& outputShape = output.getShape();
    auto const& converterShape = converters.getShape();
    TLLM_CHECK(sourceShape.nbDims == 4 && outputShape.nbDims == 4);
    TLLM_CHECK(sourceShape.d[2] == 2 && outputShape.d[2] == 2);
    TLLM_CHECK(sourceShape.d[3] == outputShape.d[3]);
    TLLM_CHECK(converterShape.nbDims == 3 && converterShape.d[0] == outputShape.d[0] && converterShape.d[1] == 2
        && converterShape.d[2] == 3);
    TLLM_CHECK(maxBlocks >= 0 && maxBlocks <= sourceShape.d[3]);
    auto const numSeqs = static_cast<SizeType32>(copyIndex.getShape().d[0]);
    TLLM_CHECK(numSeqs <= outputShape.d[1]);
    auto const numLayers = static_cast<SizeType32>(outputShape.d[0]);
    auto const count = numLayers * numSeqs * 2 * maxBlocks;
    if (count == 0)
    {
        return;
    }
    copyPerLayerBlockOffsetsToDeviceKernel<<<divUp(count, 256), 256, 0, stream>>>(
        input.getBufferPtr<SizeType32 const>(), output.getBufferPtr<SizeType32>(),
        copyIndex.getBufferPtr<SizeType32 const>(), converters.getBufferPtr<SizeType32 const>(), sourceShape.d[1],
        outputShape.d[1], sourceShape.d[3], numLayers, numSeqs, maxBlocks);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
