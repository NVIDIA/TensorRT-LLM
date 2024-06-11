/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/speculativeDecoding/common.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels::speculative_decoding
{
template <int32_t BLOCK_SIZE>
__global__ void packAcceptedPaths(SizeType32* acceptedLengthsCumSum, SizeType32* pathsOffsets,
    SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds, SizeType32 const* paths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 numPaths, SizeType32 maxPathLen,
    bool isPathsLinearBatchIdx)
{
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;

    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage tempStorage;
    auto const batchSizeRounded = ((batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    __shared__ SizeType32 currentCumSum;
    if (threadIdx.x == 0)
    {
        currentCumSum = 0;
    }

    __syncthreads();

    for (auto bi = static_cast<SizeType32>(threadIdx.x); bi < batchSizeRounded;
         bi += static_cast<SizeType32>(blockDim.x))
    {
        auto const valid = bi < batchSize;
        auto const batchSlot = valid ? batchSlots[bi] : 0;
        auto const acceptedLen = valid ? acceptedLengths[batchSlot] - 1 : 0;
        SizeType32 cumSum;
        BlockScan(tempStorage).ExclusiveSum(acceptedLen + currentCumSum, cumSum);
        if (threadIdx.x == blockDim.x - 1)
        {
            currentCumSum = cumSum;
        }
        __syncthreads();

        if (valid)
        {
            acceptedLengthsCumSum[bi] = cumSum;
            auto const pathBatchIdx = isPathsLinearBatchIdx ? bi : batchSlot;
            auto const bestPathIdx = bestPathIds[pathBatchIdx];
            auto const pathIdx = flat_index3(pathBatchIdx, bestPathIdx, 0, numPaths, maxPathLen);
            for (SizeType32 ti = 0; ti < acceptedLen; ++ti)
            {
                pathsOffsets[cumSum + ti] = paths[pathIdx + ti + 1] - 1;
            }
        }
    }
    if (threadIdx.x == 0)
    {
        acceptedLengthsCumSum[batchSize] = currentCumSum;
    }
}

void invokePackAcceptedPaths(SizeType32* acceptedLengthsCumSum, SizeType32* pathsOffsets,
    SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds, SizeType32 const* paths,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 numPaths, SizeType32 maxPathLen,
    bool isPathsLinearBatchIdx, cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 1024;
    packAcceptedPaths<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(acceptedLengthsCumSum, pathsOffsets, acceptedLengths,
        bestPathIds, paths, batchSlots, batchSize, numPaths, maxPathLen, isPathsLinearBatchIdx);
}
} // namespace tensorrt_llm::kernels::speculative_decoding
