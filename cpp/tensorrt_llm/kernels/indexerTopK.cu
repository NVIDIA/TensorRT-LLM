/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "moeTopKFuncs.cuh"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/heuristicTopKDecode.h"
#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include <algorithm>
#include <cfloat>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mutex>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

template <int step>
static inline __device__ uint32_t extractBinIdx(float x)
{
    if constexpr (step == 0)
    {
        __half hx = __float2half(x);
        uint16_t bits = __half_as_ushort(hx);
        bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
        return bits >> 5;
    }
    else
    {
        uint32_t bits = __float_as_uint(x);
        bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

        if constexpr (step == 1)
        {
            return bits >> 21;
        }
        else if constexpr (step == 2)
        {
            return (bits >> 10) & 0x7ff;
        }
        else if constexpr (step == 3)
        {
            return bits & 0x3ff;
        }
    }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern)
{
    if constexpr (shift == 0)
    {
        return true;
    }
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    return (bits ^ pattern) >> shift == 0;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, T const* in, idxT len, Func f)
{
    constexpr int WARP_SIZE = 32;
    using WideT = float4;
    if constexpr (sizeof(T) >= sizeof(WideT))
    {
        for (idxT i = thread_rank; i < len; i += num_threads)
        {
            f(in[i], i);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

        // TODO: it's UB
        union
        {
            WideT scalar;
            T array[items_per_scalar];
        } wide;

        int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
        if (skip_cnt > len)
        {
            skip_cnt = len;
        }
        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        idxT const len_cast = (len - skip_cnt) / items_per_scalar;

        for (idxT i = thread_rank; i < len_cast; i += num_threads)
        {
            wide.scalar = in_cast[i];
            idxT const real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for (int j = 0; j < items_per_scalar; ++j)
            {
                f(wide.array[j], real_i + j);
            }
        }

        static_assert(WARP_SIZE >= items_per_scalar);
        // and because items_per_scalar > skip_cnt, WARP_SIZE > skip_cnt
        // no need to use loop
        if (thread_rank < skip_cnt)
        {
            f(in[thread_rank], thread_rank);
        }
        // because len_cast = (len - skip_cnt) / items_per_scalar,
        // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
        // and so
        // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
        // WARP_SIZE no need to use loop
        idxT const remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if (remain_i < len)
        {
            f(in[remain_i], remain_i);
        }
    }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems, bool multipleBlocksPerRow,
    bool mergeBlocks, typename SmemFinalType, typename SmemOutputType, typename InputT = float>
__device__ bool processHistogramStep(int const* indices, InputT const* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx, int* smemFinalDstIdx,
    int* smemFinalBinSize, int* smemFoundTopKValues, SmemFinalType& smemFinal, int stride1, int rowStart, int topK)
{
    // Clear the histogram.
#pragma unroll
    for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock)
    {
        smemFinal.histo.data[idx] = 0;
    }

    // Make sure the histogram is ready.
    __syncthreads();

    // Update pattern
    constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
    if constexpr (step == 2)
    {
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }
    else if constexpr (step == 3)
    {
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }

    auto distributeToBins = [&](InputT logitIn, int /* idx */ = 0)
    {
        float const logit = static_cast<float>(logitIn);
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            atomicAdd(&smemFinal.histo.data[binIdx], 1);
        }
    };

    // Distribute the elements to the histogram bins.
    if (stride1 == 1)
    {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
    }
    else
    {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
        {
            distributeToBins(logits[idx * stride1], idx);
        }
    }
    // Make sure the histogram is ready.
    __syncthreads();

    // Reads the value of the starting position in the smemOutput array
    int lastValue = smemFoundTopKValues[0];

    for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++)
    {
        // Read the values from SMEM.
        int idx = threadIdx.x + kNumThreadsPerBlock * round;
        int binCount{0};
        binCount = smemFinal.histo.data[idx];

        // Make sure each thread has read its value.
        __syncthreads();

        // Compute the prefix sum.
        int prefixSum{0}, totalSum{0};
        using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

        // Update the histogram with the prefix sums.
        prefixSum += lastValue;
        totalSum += lastValue;
        smemFinal.histo.data[idx] = prefixSum;

        // Make sure the data is in shared memory.
        __syncthreads();

        // Find the last valid bin.
        bool foundThreshold = false;
        if (prefixSum < topK)
        {
            int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];

            if (nextPrefixSum >= topK)
            {
                smemThresholdBinIdx[0] = idx;
                smemFinalBinSize[0] = nextPrefixSum - prefixSum;
                foundThreshold = true;
            }
        }

        // Early exit: if any thread found the threshold, we can skip remaining
        // rounds
        if (__syncthreads_or(foundThreshold))
        {
            break;
        }

        lastValue = totalSum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The threshold bin.
    thresholdBinIdx = smemThresholdBinIdx[0];

    auto processBins = [&](InputT logitIn, int idx)
    {
        float const logit = static_cast<float>(logitIn);
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            if (binIdx < thresholdBinIdx)
            {
                // The element is part of the top-k selection
                int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

                if constexpr (mergeBlocks)
                {
                    smemOutput[dstIdx] = indices[idx];
                }
                else if constexpr (multipleBlocksPerRow)
                {
                    smemOutput[dstIdx] = idx + rowStart;
                    reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                }
                else
                {
                    smemOutput[dstIdx] = idx;
                }
            }
            if constexpr (step < 3)
            {
                // Only fill the final items for sorting if the threshold bin fits
                if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems)
                {
                    int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
                    smemFinal.items.logits[dstIdx] = logit;
                    if constexpr (mergeBlocks)
                    {
                        smemFinal.items.indices[dstIdx] = indices[idx];
                    }
                    else if constexpr (multipleBlocksPerRow)
                    {
                        smemFinal.items.indices[dstIdx] = idx + rowStart;
                    }
                    else
                    {
                        smemFinal.items.indices[dstIdx] = idx;
                    }
                }
            }
            else
            {
                if (binIdx == thresholdBinIdx)
                {
                    // The elements in the threshold bin share the same 32 bits at step 3
                    int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
                    if (dstIdx < topK)
                    {
                        if constexpr (mergeBlocks)
                        {
                            smemOutput[dstIdx] = indices[idx];
                        }
                        else if constexpr (multipleBlocksPerRow)
                        {
                            smemOutput[dstIdx] = idx + rowStart;
                            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                        }
                        else
                        {
                            smemOutput[dstIdx] = idx;
                        }
                    }
                }
            }
        }
    };

    if (stride1 == 1)
    {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
    }
    else
    {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
        {
            processBins(logits[idx * stride1], idx);
        }
    }

    // Make sure the elements are in shared memory.
    __syncthreads();

    // Check if we should continue to next step
    return smemFinalBinSize[0] > kNumFinalItems;
}

// Follows half - 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort, bool multipleBlocksPerRow = false,
    bool mergeBlocks = false, typename InputT = float>
static __device__ void topKPerRowJob(int const* indices, InputT const* logits, int rowStart, int rowEnd,
    int* outIndices, float* outLogits, int stride1, int topK)
{
    // The number of slots for the final pass.
    static constexpr int kNumFinalItems = 2048;
    // The number of elements per thread for the final sort.
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
    // The class to sort the elements during the final pass.
    using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
    using FinalSortTempStorage = std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
    // The class to compute the inclusive prefix-sum over the histogram.
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

    // The structure to store the final items (for the final pass).
    struct FinalItems
    {
        // Shared memory to store the indices for the final pass.
        int indices[kNumFinalItems];
        // Shared memory to store the logits for the final pass.
        float logits[kNumFinalItems];
    };

    struct Histogram
    {
        typename Scan::TempStorage scan;
        int data[kNumBins];
    };

    // Shared memory to compute the block sort.
    __shared__ union
    {
        FinalItems items;
        FinalSortTempStorage finalSort;
        Histogram histo;
    } smemFinal;

    // Shared memory to store the selected indices.
    // If we are processing using multiple blocks, we need to store the logits and
    // indices.
    extern __shared__ int32_t smemOutput[];

    // Shared memory to store the threshold bin.
    __shared__ int smemThresholdBinIdx[1];
    // Shared memory counter to register the candidates for the final phase.
    __shared__ int smemFinalDstIdx[1];
    // Shared memory to determine if the threshold bin fits in the final items.
    __shared__ int smemFinalBinSize[1];
    // Shared memory to keep track of the top-k values found so far by the
    // previous iterations
    __shared__ int smemFoundTopKValues[1];

    // The length of the row.
    int rowLen = rowEnd - rowStart;

    // Shortcut if the length of the row is smaller than Top-K. Indices are not
    // sorted by their corresponding logit.
    if (rowLen <= topK)
    {
        for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock)
        {
            if constexpr (multipleBlocksPerRow)
            {
                outIndices[rowIt] = rowIt + rowStart;
                outLogits[rowIt] = static_cast<float>(logits[rowIt + rowStart]);
            }
            else
            {
                outIndices[rowIt] = rowIt;
            }
        }
        for (int rowIt = rowLen + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock)
        {
            outIndices[rowIt] = -1;
            if constexpr (multipleBlocksPerRow)
            {
                outLogits[rowIt] = -FLT_MAX;
            }
        }

        return;
    }
    // Initialize values
    if (threadIdx.x == 0)
    {
        smemFinalDstIdx[0] = 0;
        smemFoundTopKValues[0] = 0;
    }
    __syncthreads();
    int thresholdBinIdx = -1;
    uint32_t logitPattern = 0;

    // Step 0: Process first 11 bits of half representation
    bool continueToNextStep
        = processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx,
            smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

    if (continueToNextStep)
    {
        // Step 1: Process next 11 bits
        continueToNextStep
            = processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
                smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    }

    if (continueToNextStep)
    {
        // Step 2: Process next 11 bits
        continueToNextStep
            = processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
                smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    }

    if (continueToNextStep)
    {
        // Step 3: Process last 10 bits
        processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx, smemFinalDstIdx,
            smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    }

    if (!continueToNextStep)
    {
        // The histogram did not proceed to the final 10 bits, therefore we need to
        // sort the final items The logits of the elements to be sorted in the final
        // pass.
        if constexpr (useRadixSort)
        {
            // Sorting with radix sort
            float finalLogits[kNumFinalItemsPerThread];
            // The indices of the elements to be sorted in the final pass.
            int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                finalLogits[ii] = -FLT_MAX;
            }

            // Read the elements from SMEM.
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                if (srcIdx < smemFinalDstIdx[0])
                {
                    finalLogits[ii] = smemFinal.items.logits[srcIdx];
                    finalIndices[ii] = smemFinal.items.indices[srcIdx];
                }
            }
            // Make sure the shared memory has been read.
            __syncthreads();

            // Sort the elements.
            FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

            // Copy the data back to the shared memory storage.
            int baseIdx = smemFoundTopKValues[0];

#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                int dstIdx = baseIdx + srcIdx;

                if (dstIdx < topK)
                {
                    smemOutput[dstIdx] = finalIndices[ii];
                    if constexpr (multipleBlocksPerRow)
                    {
                        reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = finalLogits[ii];
                    }
                }
            }
        }
        else
        {
            // Sorting with insertion sort
            auto baseIdx = smemFoundTopKValues[0];
            for (int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock)
            {
                int outIndex = 0;
                auto logit = smemFinal.items.logits[i];
                for (int j = 0; j < smemFinalDstIdx[0]; j++)
                {
                    auto otherLogit = smemFinal.items.logits[j];
                    if (logit < otherLogit || (logit == otherLogit && i < j))
                    {
                        outIndex++;
                    }
                }
                // Store if outIndex is in bounds
                if (outIndex + baseIdx < topK)
                {
                    smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
                    if constexpr (multipleBlocksPerRow)
                    {
                        reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = smemFinal.items.logits[i];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store to global memory.
    for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock)
    {
        if constexpr (multipleBlocksPerRow)
        {
            outIndices[i] = smemOutput[i];
            outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
        }
        else
        {
            if (stride1 == 1)
            {
                // stride1 == 1 will use vectorized_process, which indexes already skip the rowStart.
                outIndices[i] = smemOutput[i];
            }
            else
            {
                outIndices[i] = smemOutput[i] - rowStart;
            }
        }
    }
}
} // namespace

template <int kNumThreadsPerBlock, bool useRadixSort, typename InputT = float>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(InputT const* logits,
    int const* rowStarts, int const* rowEnds, int* outIndices, int stride0, int stride1, int const topK,
    int const offsetIndex)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x + offsetIndex;

    // The range of logits within the row.
    int rowStart = rowStarts[rowIdx];
    int rowEnd = rowEnds[rowIdx];

    // Local pointers to this block
    outIndices += static_cast<int64_t>(rowIdx) * topK;
    logits += static_cast<int64_t>(rowIdx) * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
        nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock, bool useRadixSort, bool multipleBlocksPerRow = false, bool mergeBlocks = false,
    typename InputT = float>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(InputT const* logits, int const* seqLens,
    int* outIndices, int stride0, int stride1, int const topK, int next_n, int compressRatio,
    float* outLogits = nullptr,
    int const numBlocksToMerge = 0, int const* indices = nullptr)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x;

    // The range of logits within the row.
    int rowStart = 0;
    int seq_len = seqLens[rowIdx / next_n];
    int actual_kv_len = seq_len - next_n + (rowIdx % next_n) + 1;
    int rowEnd = actual_kv_len / compressRatio;

    // Local pointers to this block
    if constexpr (!multipleBlocksPerRow && !mergeBlocks)
    {
        outIndices += static_cast<int64_t>(rowIdx) * topK;
    }
    else if constexpr (multipleBlocksPerRow)
    {
        auto const blockSize = rowEnd / gridDim.y; // 16384 / 2 = 8192
        rowStart = blockSize * blockIdx.y;         // 8192 * 1 = 8192
        rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
        outIndices += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
        outLogits += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
    }
    else if constexpr (mergeBlocks)
    {
        rowEnd = numBlocksToMerge * topK;
        indices += static_cast<int64_t>(rowIdx) * numBlocksToMerge * topK;
        outIndices += static_cast<int64_t>(rowIdx) * topK;
    }
    logits += static_cast<int64_t>(rowIdx) * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

namespace
{

// Scheme X bound calculator — shared between fp32 and bf16/fp16 dispatchers.
// Caches hardware attrs (SM count, L2 capacity) and the small-N threshold
// once per process via std::call_once. Per-call cost is just two reads
// from cached static variables plus a small arithmetic block, no syscalls.
struct SchemeXBounds
{
    int smCount;
    int l2Bytes;
    int kBsWave;
    int kBsL2;
    int kBsLarge;
    int kSeqSmall;
};

inline SchemeXBounds getSchemeXBounds(int numColumns, int bytesPerElem)
{
    static std::once_flag sOnce;
    static int sSm = 0;
    static int sL2 = 0;
    static int sNMin = 0;
    std::call_once(sOnce,
        []()
        {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sSm, cudaDevAttrMultiProcessorCount, dev);
            cudaDeviceGetAttribute(&sL2, cudaDevAttrL2CacheSize, dev);
            constexpr int kSeqSmallDefault = 12288;
            char const* env = std::getenv("TRTLLM_HEURISTIC_NMIN");
            if (env != nullptr)
            {
                int const v = std::atoi(env);
                sNMin = (v >= 1024 && v <= 200000) ? v : kSeqSmallDefault;
            }
            else
            {
                sNMin = kSeqSmallDefault;
            }
        });

    SchemeXBounds b;
    b.smCount = sSm;
    b.l2Bytes = sL2;
    b.kBsWave = (sSm > 0) ? (sSm * 3 - sSm / 8) : 426;
    b.kBsL2 = (sL2 > 0 && numColumns > 0)
        ? static_cast<int>(static_cast<int64_t>(sL2) * 9 / 10 / (static_cast<int64_t>(numColumns) * bytesPerElem))
        : b.kBsWave;
    b.kBsLarge = std::min(b.kBsWave, b.kBsL2 > 0 ? b.kBsL2 : b.kBsWave);
    b.kSeqSmall = sNMin;
    return b;
}

} // anonymous namespace

void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK, int const* preIdx, int const preIdxStride,
    int const preIdxCount, float* heuristicScratch, int const compressRatio, cudaStream_t const stream)
{

    // INVARIANT: kSortingAlgorithmThreshold is the ORIGINAL TRT-LLM Radix-path
    // internal boundary (Insertion vs Radix-radix). v1.2.X dispatcher leaves it
    // at 12288 — the GVR Heuristic axis (kSeqSmall, see below) is INDEPENDENT,
    // so when canUseHeuristic is false (e.g. preIdx missing, BS too large, or
    // numColumns < kSeqSmall), this function falls back to BYTE-IDENTICAL
    // original radix dispatcher behavior. Do not touch this constant.
    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kDefaultSplitWorkThreshold = 200 * 1000;
    constexpr int kNumThreadsPerBlock = 512;
    int const effectiveSplitWorkThreshold = splitWorkThreshold > 0 ? splitWorkThreshold : kDefaultSplitWorkThreshold;

    // ========================================================================
    // Small-N dispatch axis.
    //
    // GVR Heuristic Top-K has a *fixed* per-launch overhead from Phase-1
    // (preIdx stats reduction over M=2048) and Phase-4 (2048-bin histogram
    // snap), totaling ~11 µs regardless of N. For small N (≤16K), this
    // fixed cost dominates and the kernel loses to the existing
    // insertion-sort/radix path. Empirically (random data, B200 BS=1):
    //   N=8192   : Heuristic 16.5 µs vs Radix 11.2 µs (radix 1.47× faster)
    //   N=16384  : Heuristic 21.9 µs vs Radix 22.0 µs (parity)
    //   N=32768  : Heuristic 26.1 µs vs Radix 32.9 µs (heuristic 1.26× faster)
    //   N=131072 : Heuristic 43.4 µs vs Radix 76.1 µs (heuristic 1.75× faster)
    //
    // Route N < kSeqSmall to the existing Radix/Insertion path (which itself
    // splits at kSortingAlgorithmThreshold=12288). kSeqSmall is set at the
    // empirical crossover point.
    //
    // ========================================================================
    // Architecture-derived BS-threshold dispatch — jointly bounded by
    // occupancy AND L2 cache capacity.
    //
    // Two physical constraints bound when the per-row heuristic kernel
    // remains faster than a radix streaming kernel:
    //
    //   (A) Occupancy bound — 3·SM − SM/8 (wave geometry + setup margin)
    //        Each CTA uses ~58 KB SMEM (fixed, independent of N), so B200's
    //        228 KB dynamic SMEM allows max 3 CTA/SM. Above 3·SM rows per
    //        launch, tail-wave imbalance causes stragglers. The -SM/8 margin
    //        (~1/8 wave) covers CTA setup + L2 ingestion overhead.
    //        On B200(148 SM): 3×148 − 18 = 426.
    //
    //   (B) L2 cache bound — 0.9·L2 / (4·N) per-CTA logits fit
    //        Each CTA streams its row (N×4B) through L2 per Phase-2 iter.
    //        With num_concurrent_CTAs × N × 4B > L2, eviction dominates.
    //        On B200(126 MB L2) with N=70K: 0.9·126MB/(4·70690) ≈ 440,
    //        which is ~ equal to (A)=426 — the two constraints cross over
    //        near the SWE-Bench data point.
    //        For N > 73K the L2 bound tightens below (A) and must take
    //        over; e.g. N=128K → kBsL2=238, N=196K → kBsL2=155.
    //
    // Dispatch threshold = min(kBsWave, kBsL2), still data-agnostic (only
    // queries hardware attrs). At N≈70K both bounds produce ~426, so the
    // L2 axis is a no-op there; for larger N it auto-tightens the threshold.
    //
    // Small-N lower bound `kSeqSmall` (default 12288) lets the Heuristic
    // axis take over wherever the original Radix-radix branch would have
    // triggered. Random-data benchmarks suggest the crossover is 16384,
    // but workloads with strongly preIdx-correlated logits make P1 stats
    // accurate and P2 converge in 1-2 iterations, shifting the real
    // crossover into the [12288, 16384] band. Configurable via
    // TRTLLM_HEURISTIC_NMIN env (>=1024).
    // ========================================================================
    auto const bounds = getSchemeXBounds(numColumns, /*bytesPerElem=*/4);
    int const kBsWave = bounds.kBsWave;
    int const kBsL2 = bounds.kBsL2;
    int const kBsLarge = bounds.kBsLarge;
    int const kSeqSmall = bounds.kSeqSmall;

    bool const isSupportedTopK = (topK == 512 || topK == 1024 || topK == 2048);
    bool const canUseHeuristic = compressRatio == 1 && preIdx != nullptr && stride1 == 1 && isSupportedTopK
        && preIdxCount == topK && preIdxStride >= preIdxCount && numColumns < effectiveSplitWorkThreshold
        && numColumns >= kSeqSmall && heuristicScratch != nullptr && numRows < kBsLarge;

    // Optional env-gated dispatch trace (set TRTLLM_SCHEMEX_DEBUG=1 to enable)
    {
        static std::once_flag sDebugOnceFlag;
        static bool sDebug = false;
        std::call_once(sDebugOnceFlag,
            []()
            {
                char const* env = std::getenv("TRTLLM_SCHEMEX_DEBUG");
                sDebug = (env != nullptr && env[0] == '1');
            });
        if (sDebug)
        {
            fprintf(stderr,
                "[Scheme X] numRows=%d numColumns=%d kBsWave=%d kBsL2=%d kBsLarge=%d kSeqSmall=%d smCount=%d "
                "L2=%dMB -> %s path%s\n",
                numRows, numColumns, kBsWave, kBsL2, kBsLarge, kSeqSmall, bounds.smCount,
                bounds.l2Bytes / (1024 * 1024), canUseHeuristic ? "Heuristic" : "Radix",
                (numColumns < kSeqSmall) ? " (small-N route)" : "");
        }
    }

    if (canUseHeuristic)
    {
        launchHeuristicTopKDecode(logits, seqLens, preIdx, indices, heuristicScratch, stride0, next_n, topK,
            preIdxStride, preIdxCount, numRows, stream);
    }
    else if (numColumns < kSortingAlgorithmThreshold)
    {
        // Use insertion sort
        auto* kernel_instance = &topKPerRowDecode<kNumThreadsPerBlock, false>;

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = kNumThreadsPerBlock;
        config.dynamicSmemBytes = topK * sizeof(int32_t);
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kernel_instance, logits, seqLens, indices, stride0, stride1, topK, next_n,
            compressRatio, nullptr, 0, nullptr);
    }
    else if (numColumns < effectiveSplitWorkThreshold)
    {
        // From this threshold, use radix sort instead
        auto* kernel_instance = &topKPerRowDecode<kNumThreadsPerBlock, true>;

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = kNumThreadsPerBlock;
        config.dynamicSmemBytes = topK * sizeof(int32_t);
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kernel_instance, logits, seqLens, indices, stride0, stride1, topK, next_n,
            compressRatio, nullptr, 0, nullptr);
    }
    else
    {
        // Long sequences are run in two steps
        constexpr auto multipleBlocksPerRowConfig = 10;
        auto* kernel_instance_part1 = &topKPerRowDecode<kNumThreadsPerBlock, true, true>;
        cudaLaunchConfig_t config_part1;
        config_part1.gridDim = dim3(numRows, multipleBlocksPerRowConfig);
        config_part1.blockDim = kNumThreadsPerBlock;
        config_part1.dynamicSmemBytes = 2 * topK * sizeof(int32_t);
        config_part1.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config_part1.numAttrs = 1;
        config_part1.attrs = attrs;

        cudaLaunchKernelEx(&config_part1, kernel_instance_part1, logits, seqLens, outIndicesAux, stride0, stride1, topK,
            next_n, compressRatio, outLogitsAux, 0, nullptr);

        constexpr int kNumThreadsPerBlockMerge = 1024;
        auto* kernel_instance_part2 = &topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>;
        cudaLaunchConfig_t config_part2;
        config_part2.gridDim = numRows;
        config_part2.blockDim = kNumThreadsPerBlockMerge;
        config_part2.dynamicSmemBytes = topK * sizeof(int32_t);
        config_part2.stream = stream;
        // Reuse attrs array since part1 kernel has already been launched
        config_part2.numAttrs = 1;
        config_part2.attrs = attrs;

        cudaLaunchKernelEx(&config_part2, kernel_instance_part2, outLogitsAux, seqLens, indices,
            multipleBlocksPerRowConfig * topK, 1, topK, next_n, 1, nullptr, multipleBlocksPerRowConfig, outIndicesAux);
    }
    sync_check_cuda_error(stream);
}

// ============================================================================
// bf16 / fp16 dispatcher overloads
// ============================================================================
// Reuses the BS-threshold + small-N dispatch axes (kBsLarge, kSeqSmall) from
// the fp32 dispatcher, except kBsL2 uses sizeof(InputT) bytes/element instead
// of 4 — L2 footprint is half, so bf16/fp16 path remains valid for larger BS
// than fp32 at the same N.
//
// Fallback chain when GVR-Heuristic preconditions are not met (preIdx
// missing, BS too large, or numColumns < kSeqSmall):
//   numColumns < kSortingAlgorithmThreshold (12288)            → insertion sort
//   kSortingAlgorithmThreshold ≤ numColumns < splitWorkThreshold → radix sort
//   numColumns ≥ splitWorkThreshold (200K default)             → unsupported
//
// Insertion + radix tiers use the same topKPerRowDecode kernel as fp32 with
// InputT propagated through; the histogram and sort steps operate on float
// keys after a static_cast<float>(InputT) at HBM-read sites, so accuracy is
// identical to casting input to fp32 before the kernel.
//
// The split-work tier requires float aux buffers (outLogitsAux /
// outIndicesAux) that the bf16/fp16 entry does not expose; callers in that
// regime must use the fp32 entry.

namespace
{

template <typename InputT>
void invokeIndexerTopKDecodeDtype(InputT const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n, int const topK,
    int const* preIdx, int const preIdxStride, int const preIdxCount, InputT* heuristicScratch,
    int const compressRatio, cudaStream_t const stream)
{
    static_assert(std::is_same_v<InputT, __nv_bfloat16> || std::is_same_v<InputT, __half>,
        "invokeIndexerTopKDecodeDtype is for bf16/fp16 only");

    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kDefaultSplitWorkThreshold = 200 * 1000;
    constexpr int kNumThreadsPerBlock = 512;
    int const effectiveSplitWorkThreshold = splitWorkThreshold > 0 ? splitWorkThreshold : kDefaultSplitWorkThreshold;

    // bf16/fp16: bytes_per_element = sizeof(InputT) = 2 → kBsL2 doubles vs fp32.
    auto const bounds = getSchemeXBounds(numColumns, /*bytesPerElem=*/static_cast<int>(sizeof(InputT)));
    int const kBsLarge = bounds.kBsLarge;
    int const kSeqSmall = bounds.kSeqSmall;

    bool const isSupportedTopK = (topK == 512 || topK == 1024 || topK == 2048);
    bool const canUseHeuristic = compressRatio == 1 && preIdx != nullptr && stride1 == 1 && isSupportedTopK
        && preIdxCount == topK && preIdxStride >= preIdxCount && numColumns < effectiveSplitWorkThreshold
        && numColumns >= kSeqSmall && heuristicScratch != nullptr && numRows < kBsLarge;

    if (canUseHeuristic)
    {
        launchHeuristicTopKDecode(logits, seqLens, preIdx, indices, heuristicScratch, stride0, next_n, topK,
            preIdxStride, preIdxCount, numRows, stream);
    }
    else if (numColumns < kSortingAlgorithmThreshold)
    {
        // Insertion sort path — InputT propagated; histogram/sort run on float keys.
        auto* kernel_instance = &topKPerRowDecode<kNumThreadsPerBlock, /*useRadixSort=*/false,
            /*multipleBlocksPerRow=*/false, /*mergeBlocks=*/false, InputT>;

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = kNumThreadsPerBlock;
        config.dynamicSmemBytes = topK * sizeof(int32_t);
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kernel_instance, logits, seqLens, indices, stride0, stride1, topK, next_n,
            compressRatio, nullptr, 0, nullptr);
    }
    else if (numColumns < effectiveSplitWorkThreshold)
    {
        // Radix sort path — InputT propagated; histogram/sort run on float keys.
        auto* kernel_instance = &topKPerRowDecode<kNumThreadsPerBlock, /*useRadixSort=*/true,
            /*multipleBlocksPerRow=*/false, /*mergeBlocks=*/false, InputT>;

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = kNumThreadsPerBlock;
        config.dynamicSmemBytes = topK * sizeof(int32_t);
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kernel_instance, logits, seqLens, indices, stride0, stride1, topK, next_n,
            compressRatio, nullptr, 0, nullptr);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "indexer_topk_decode bf16/fp16 path does not support numColumns >= splitWorkThreshold "
            "(split-work path requires float aux buffers not exposed in the bf16/fp16 entry). "
            "Got numColumns=%d splitWorkThreshold=%d. Use the fp32 entry for this regime.",
            numColumns, effectiveSplitWorkThreshold);
    }

    sync_check_cuda_error(stream);
}

} // anonymous namespace

void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices,
    int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const next_n, int const topK, int const* preIdx, int const preIdxStride, int const preIdxCount,
    __nv_bfloat16* heuristicScratch, int const compressRatio, cudaStream_t const stream)
{
    invokeIndexerTopKDecodeDtype<__nv_bfloat16>(logits, seqLens, indices, splitWorkThreshold, numRows, numColumns,
        stride0, stride1, next_n, topK, preIdx, preIdxStride, preIdxCount, heuristicScratch, compressRatio, stream);
}

void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n, int const topK,
    int const* preIdx, int const preIdxStride, int const preIdxCount, __half* heuristicScratch,
    int const compressRatio, cudaStream_t const stream)
{
    invokeIndexerTopKDecodeDtype<__half>(logits, seqLens, indices, splitWorkThreshold, numRows, numColumns, stride0,
        stride1, next_n, topK, preIdx, preIdxStride, preIdxCount, heuristicScratch, compressRatio, stream);
}

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK,
    cudaStream_t const stream)
{
    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kNumThreadsPerBlock = 512;

    int numInsertionBlocks = std::min(numRows, kSortingAlgorithmThreshold);
    topKPerRowPrefill<kNumThreadsPerBlock, false>
        <<<numInsertionBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits, rowStarts, rowEnds, indices, stride0, stride1, topK, 0);

    if (numRows > kSortingAlgorithmThreshold)
    {
        int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
        topKPerRowPrefill<kNumThreadsPerBlock, true>
            <<<numRadixBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
                logits, rowStarts, rowEnds, indices, stride0, stride1, topK, kSortingAlgorithmThreshold);
    }

    sync_check_cuda_error(stream);
}

bool canIndexerTopKDecodeUseGvr(int numRows, int numColumns, int topK, int bytesPerElem)
{
    bool const isSupportedTopK = (topK == 512 || topK == 1024 || topK == 2048);
    if (!isSupportedTopK)
    {
        return false;
    }
    constexpr int kDefaultSplitWorkThreshold = 200 * 1000;
    auto const bounds = getSchemeXBounds(numColumns, bytesPerElem);
    return numColumns >= bounds.kSeqSmall && numColumns < kDefaultSplitWorkThreshold && numRows < bounds.kBsLarge;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
