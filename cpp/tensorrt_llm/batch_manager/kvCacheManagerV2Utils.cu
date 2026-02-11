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
using Grain = uint4;
constexpr uint32_t ctaSize = 128;
constexpr uint32_t copyBlockCtaSize = 128;
constexpr uint32_t copyBlocknbBufs = 2;
constexpr uint32_t nbBufs = 4;
constexpr uint32_t grainBytes = sizeof(Grain);

using MMTask = Task<MemAddress, MemAddress>;

__device__ __host__ inline uint32_t divUp(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <uint32_t N>
__global__ void batchedCopy(std::array<MMTask, N> const __grid_constant__ tasks, uint32_t nbBytes)
{
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n");
#endif
    assert(nbBytes % sizeof(Grain) == 0);
    __shared__ Grain data[nbBufs][ctaSize];

    uint32_t const nbTasks = gridDim.y;
    assert(nbTasks <= N);
    auto const& task = tasks[blockIdx.y];
    uint32_t const nbSplits = gridDim.x;
    uint32_t const idxSplit = blockIdx.x;
    uint32_t const tid = threadIdx.x;

    constexpr uint32_t bytesPerIter = grainBytes * ctaSize;

    uint32_t const totalIters = divUp(nbBytes, bytesPerIter);
    uint32_t const maxItersPerCta = divUp(totalIters, nbSplits);
    uint32_t const idxGrainBeg = ctaSize * maxItersPerCta * idxSplit + tid;
    uint32_t const idxGrainEnd = std::min(idxGrainBeg + ctaSize * maxItersPerCta, nbBytes / grainBytes);

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n");
#endif
    for (uint32_t i = 0; i < maxItersPerCta + nbBufs; i++)
    {
        uint32_t const idxBuf = i % nbBufs;
        if (i >= nbBufs)
        {
            uint32_t const stIter = i - nbBufs;
            assert(idxBuf == (stIter % nbBufs));
            Grain const& src = data[idxBuf][tid];
            uint32_t const idxGrain = idxGrainBeg + ctaSize * stIter;
            Grain& dst = reinterpret_cast<Grain*>(task.dst)[idxGrain];
            asm volatile("cp.async.wait_group %0;\n" ::"n"(nbBufs - 1) : "memory");
            if (idxGrain < idxGrainEnd)
            {
                dst = src;
            }
        }
        uint32_t const ldIter = i;
        Grain* const dst = &data[idxBuf][tid];
        uint32_t const idxGrain = idxGrainBeg + ctaSize * ldIter;
        Grain const* const src = &reinterpret_cast<Grain const*>(task.src)[idxGrain];
        if (idxGrain < idxGrainEnd)
        {
            uint32_t const size = grainBytes;
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"l"(__cvta_generic_to_shared(dst)),
                         "l"(src), "n"(grainBytes), "r"(size)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;\n" : : : "memory");
    }
}

template <uint32_t N>
CUresult launchBatchedCopyImpl(
    bool lowBandwidth, MMTask const* tasks, uint32_t nbTasks, uint32_t nbBytes, cudaStream_t stream)
{
    TLLM_CHECK(nbTasks <= N);
    TLLM_CHECK_WITH_INFO(
        nbBytes % sizeof(Grain) == 0, "Not implemented case: nbBytes = %d must be a multiple of 16.", nbBytes);
    std::array<MMTask, N> const* pTasks;
    std::array<MMTask, N> tmp;
    if (nbTasks < N)
    {
        std::copy_n(tasks, nbTasks, tmp.begin());
        pTasks = &tmp;
    }
    else
    {
        pTasks = reinterpret_cast<std::array<MMTask, N> const*>(tasks);
    }
    uint32_t const nbSplits = lowBandwidth ? 1 : divUp(nbBytes, grainBytes * ctaSize * 2);
    void* args[] = {(void*) pTasks, (void*) &nbBytes};
    static CUkernel const kernel = [] -> CUkernel
    {
        cudaKernel_t kernel = nullptr;
        TLLM_CUDA_CHECK(cudaGetKernel(&kernel, reinterpret_cast<void const*>(&batchedCopy<N>)));
        return kernel;
    }();
    return common::CUDADriverWrapper::getInstance()->cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), nbSplits,
        nbTasks, 1,    // gridDimX, gridDimY, gridDimZ
        ctaSize, 1, 1, // blockDimX, blockDimY, blockDimZ
        0,             // sharedMemBytes
        stream, args, nullptr);
}

// When bandwidth is low, e.g. when host memory is involved, we avoid splitting as fewer CTAs should be enough to
// saturate the bandwidth.
CUresult launchBatchedCopy(bool lowBandwidth, std::vector<MMTask> const& tasks, uint32_t nbBytes, cudaStream_t stream)
{
    constexpr uint32_t maxN = 256;
    uint32_t const nbWholeBatches = tasks.size() / maxN;
    for (uint32_t i = 0; i < nbWholeBatches; i++)
    {
        CUresult const err = launchBatchedCopyImpl<maxN>(lowBandwidth, tasks.data() + maxN * i, maxN, nbBytes, stream);
        if (err != CUDA_SUCCESS)
        {
            return err;
        }
    }
    {
        auto const* const pTasks = tasks.data() + maxN * nbWholeBatches;
        auto const batchSize = tasks.size() % maxN;
        if (batchSize == 0)
        {
            return CUDA_SUCCESS;
        }
        if (batchSize > maxN / 2)
        {
            return launchBatchedCopyImpl<maxN>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
        }
        if (batchSize > maxN / 4)
        {
            return launchBatchedCopyImpl<maxN / 2>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
        }
        if (batchSize > maxN / 8)
        {
            return launchBatchedCopyImpl<maxN / 4>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
        }
        return launchBatchedCopyImpl<maxN / 8>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
    }
}

CUresult copyHostToDevice(std::vector<MMTask> const& tasks, ssize_t numBytes, CUstream stream) noexcept
{
    return launchBatchedCopy(true, tasks, numBytes, stream);
}

CUresult copyDeviceToHost(std::vector<MMTask> const& tasks, ssize_t numBytes, CUstream stream) noexcept
{
    return launchBatchedCopy(true, tasks, numBytes, stream);
}

CUresult copyDeviceToDevice(std::vector<MMTask> const& tasks, ssize_t numBytes, CUstream stream) noexcept
{
    return launchBatchedCopy(false, tasks, numBytes, stream);
}

// dst_tensor[:, :num_seqs, 0] = src_tensor[:, copy_idx]
// dst_tensor[:, :num_seqs, 1] = dst_tensor[:, :num_seqs, 0] + 1
template <bool COPY_V_IDX = true>
__global__ void copyBatchBlockOffsetsToDeviceKernel(SizeType32 const* __restrict__ srcPtr,
    SizeType32* __restrict__ dstPtr, SizeType32 const srcMaxNumSequences, SizeType32 const dstMaxNumSequences,
    SizeType32 numBlocksPerSeq, SizeType32 const* __restrict__ copyIndex)
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
                dstK = src;
                if (COPY_V_IDX)
                {
                    dstV = src;
                }
                else
                {
#pragma unroll
                    for (uint32_t j = 0; j < elemPerAccess; j++)
                    {
                        auto const val = src.unpacked[j];
                        dstV.unpacked[j] = (val == BAD_PAGE_INDEX) ? val : (val + 1);
                    }
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
void copyBatchBlockOffsetsToDevice(
    ITensor const& input, ITensor& output, ITensor const& copyIndex, bool copyVIdx, CUstream stream) noexcept
{
    using namespace tensorrt_llm::runtime;

    auto const* srcPtr = bufferCast<tk::KVCacheIndex::UnderlyingType const>(input);
    auto* dstPtr = bufferCast<tk::KVCacheIndex::UnderlyingType>(
        output); // [numPools, maxNumSequences, kvFactor, numBlocksPerSeq]
    auto const* copyIndexPtr = bufferCast<SizeType32 const>(copyIndex);
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

    if (copyVIdx)
    {
        copyBatchBlockOffsetsToDeviceKernel<true><<<gridDim, blockDim, 0, stream>>>(
            srcPtr, dstPtr, srcMaxNumSequences, dstMaxNumSequences, numBlocksPerSeq, copyIndexPtr);
    }
    else
    {
        copyBatchBlockOffsetsToDeviceKernel<false><<<gridDim, blockDim, 0, stream>>>(
            srcPtr, dstPtr, srcMaxNumSequences, dstMaxNumSequences, numBlocksPerSeq, copyIndexPtr);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
