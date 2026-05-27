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

    // At step 0, auto-promoting items in higher half-precision bins is only
    // safe if no subsequent step will run — otherwise step 2's
    // isPartialMatch<21> filter cannot exclude them (half-precision binning
    // does not correspond to bits[31:21] of full-precision), and they would
    // be auto-promoted a second time at step 2. When step 0 will continue,
    // defer all promotion to the later steps (whose bit-pattern filters
    // correctly carry forward).
    bool const step0WillContinue = (step == 0) && (smemFinalBinSize[0] > kNumFinalItems);

    auto processBins = [&](InputT logitIn, int idx)
    {
        float const logit = static_cast<float>(logitIn);
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            if (binIdx < thresholdBinIdx && !step0WillContinue)
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

// Type holder for the per-block "final" union used by topKPerRowJob. The
// union always carries the BlockRadixSort temp storage so the final-sort
// step in topKPerRowJob can pick insertion vs radix sort entirely at
// runtime based on the actual candidate count (see the sort-selection block
// further below). In practice BlockRadixSort::TempStorage stays under
// FinalItems = 16KB at our shapes (kNumFinalItems=2048,
// kNumThreadsPerBlock ∈ {256, 512, 1024}), so the union size does not grow
// versus the previous useRadixSort=false layout.
template <int kNumThreadsPerBlock, int kNumBins>
struct TopKSmem
{
    static constexpr int kNumFinalItems = 2048;
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
    using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
    using FinalSortTempStorage = typename FinalSort::TempStorage;
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

    struct FinalItems
    {
        int indices[kNumFinalItems];
        float logits[kNumFinalItems];
    };

    struct Histogram
    {
        typename Scan::TempStorage scan;
        int data[kNumBins];
    };

    union Final
    {
        FinalItems items;
        FinalSortTempStorage finalSort;
        Histogram histo;
    };

    Final smemFinal;
    int smemThresholdBinIdx[1];
    int smemFinalDstIdx[1];
    int smemFinalBinSize[1];
    int smemFoundTopKValues[1];
};

// Follows half - 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock, int kNumBins, bool multipleBlocksPerRow = false, bool mergeBlocks = false,
    typename InputT = float>
static __device__ void topKPerRowJob(int const* indices, InputT const* logits, int rowStart, int rowEnd,
    int* outIndices, float* outLogits, int stride1, int topK,
    TopKSmem<kNumThreadsPerBlock, kNumBins>& smem)
{
    static constexpr int kNumFinalItems = TopKSmem<kNumThreadsPerBlock, kNumBins>::kNumFinalItems;
    static constexpr int kNumFinalItemsPerThread
        = TopKSmem<kNumThreadsPerBlock, kNumBins>::kNumFinalItemsPerThread;
    using FinalSort = typename TopKSmem<kNumThreadsPerBlock, kNumBins>::FinalSort;

    auto& smemFinal = smem.smemFinal;
    int* smemThresholdBinIdx = smem.smemThresholdBinIdx;
    int* smemFinalDstIdx = smem.smemFinalDstIdx;
    int* smemFinalBinSize = smem.smemFinalBinSize;
    int* smemFoundTopKValues = smem.smemFoundTopKValues;

    // Shared memory to store the selected indices.
    // If we are processing using multiple blocks, we need to store the logits and
    // indices.
    extern __shared__ int32_t smemOutput[];

    // The length of the row.
    int rowLen = rowEnd - rowStart;

    // Shortcut if the length of the row is smaller than Top-K. Indices are not
    // sorted by their corresponding logit. Unreachable when mergeBlocks=true:
    // both merge callers pass rowLen = numBlocksPerRow * topK > topK.
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
        // The histogram did not proceed to the final 10 bits, so we now sort
        // the candidates that landed in the threshold bin. Runtime branch on
        // `smemFinalDstIdx[0]` (count bounded by kNumFinalItems=2048): for
        // small counts the O(n²/T) insertion sort beats BlockRadixSort, which
        // always pays its full kNumFinalItems-padded cost regardless of n.
        //
        // Threshold 512 picked from a 3-run averaged sweep over bs=1..128 /
        // seq=1k..256k (gaussian, k=2048) on Blackwell sm_100. Matches the
        // theoretical breakeven n²/T ≈ const BlockRadixSort cost at n ~ 450-500.
        constexpr int kInsertionSortBranchThreshold = 512;
        int const finalCount = smemFinalDstIdx[0];
        if (finalCount > kInsertionSortBranchThreshold)
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
                if (srcIdx < finalCount)
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
            // Sorting with insertion sort.
            auto baseIdx = smemFoundTopKValues[0];
            for (int i = threadIdx.x; i < finalCount; i += kNumThreadsPerBlock)
            {
                int outIndex = 0;
                auto logit = smemFinal.items.logits[i];
                for (int j = 0; j < finalCount; j++)
                {
                    auto otherLogit = smemFinal.items.logits[j];
                    if (logit < otherLogit || (logit == otherLogit && i < j))
                    {
                        outIndex++;
                    }
                }
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
        else if constexpr (mergeBlocks)
        {
            outIndices[i] = smemOutput[i];
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

// Per-row block count picked by the fused split-work dispatcher (see
// invokeIndexerTopKDecodeImpl). Defaults to 10 (matching the legacy
// 2-launch split-work tier); bumped at very-low-bs corners where 10
// blocks/row leaves the GPU under-filled. The merge phase processes
// numBlocksPerRow * topK candidates in a single block, so we never bump
// past where the merge would dominate. Cutoffs come from the
// mid-long-decode sweep. Exposed here so callers can size the aux
// buffers (outIndicesAux / outLogitsAux) to numRows × numBlocksPerRow ×
// topK — sizing them at the legacy 10 blocks/row triggers OOB writes for
// the bumped tiers.
int indexerTopKDecodeFusedAuxBlocksPerRow(int numRows, int numColumns)
{
    int numBlocksPerRow = 10;
    if (numRows == 1)
    {
        // 132 SMs / 1 row → cap at 32 (aux buffer max), gated so per-block
        // scan stays >= 30k (the 1024-thread wide-block threshold).
        int target = 32;
        if (target > numColumns / 30000)
            target = numColumns / 30000;
        if (target > numBlocksPerRow)
            numBlocksPerRow = target;
    }
    else if (numRows == 2 && numColumns >= 400000)
        numBlocksPerRow = 16;
    else if (numRows == 4 && numColumns >= 400000)
        numBlocksPerRow = 14;
    else if (numRows == 8 && numColumns >= 500000)
        numBlocksPerRow = 12;
    return numBlocksPerRow;
}

template <int kNumThreadsPerBlock, typename InputT = float>
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

    __shared__ TopKSmem<kNumThreadsPerBlock, kNumBins> smem;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins>(
        nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK, smem);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock, bool multipleBlocksPerRow = false, bool mergeBlocks = false,
    typename InputT = float>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(InputT const* logits, int const* seqLens,
    int* outIndices, int stride0, int stride1, int const topK, int next_n, float* outLogits = nullptr,
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
    int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

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

    __shared__ TopKSmem<kNumThreadsPerBlock, kNumBins> smem;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK, smem);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock, typename InputT = float>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecodeFused(InputT const* logits,
    int const* seqLens, int* outIndices, int* outIndicesAux, float* outLogitsAux, int* doneCounter, int stride0,
    int stride1, int const topK, int const next_n, int const numBlocksPerRow)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    static constexpr int kNumBins = 2048;

    int rowIdx = blockIdx.x;
    int blockInRow = blockIdx.y;

    int seq_len = seqLens[rowIdx / next_n];
    int rowEndFull = seq_len - next_n + (rowIdx % next_n) + 1;

    InputT const* rowLogits = logits + static_cast<int64_t>(rowIdx) * stride0;

    // One smem allocation shared between part1 and the merge phase. Both calls
    // use the same kNumThreadsPerBlock/kNumBins, and the multipleBlocksPerRow /
    // mergeBlocks template flags don't affect smem layout, so the union /
    // scratch can be reused (each call re-initializes its working slots at
    // entry).
    __shared__ TopKSmem<kNumThreadsPerBlock, kNumBins> smem;

    // Short-row short-circuit: if the actual row is too small to benefit from
    // multi-block fan-out (e.g. the prefill harness uses lengths=1..bs which
    // are tiny no matter how large numColumns is), have block 0 do the full
    // single-block top-k and have all other blocks bail without touching the
    // atomic counter. Without this, fused-path overhead dominates for these
    // configs.
    constexpr int kFusedShortRowThreshold = 16384;
    if (rowEndFull < kFusedShortRowThreshold)
    {
        if (blockInRow == 0)
        {
            int* mergedOut = outIndices + static_cast<int64_t>(rowIdx) * topK;
            topKPerRowJob<kNumThreadsPerBlock, kNumBins,
                /*multipleBlocksPerRow=*/false, /*mergeBlocks=*/false>(nullptr, rowLogits, /*rowStart=*/0, rowEndFull,
                mergedOut,
                /*outLogits=*/nullptr, stride1, topK, smem);
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    int blockSize = rowEndFull / numBlocksPerRow;
    int rowStart = blockSize * blockInRow;
    int rowEnd = (blockInRow == numBlocksPerRow - 1) ? rowEndFull : rowStart + blockSize;

    int64_t auxOffset = (static_cast<int64_t>(rowIdx) * numBlocksPerRow + blockInRow) * topK;
    int* myOutIndicesAux = outIndicesAux + auxOffset;
    float* myOutLogitsAux = outLogitsAux + auxOffset;

    // Part 1: per-block top-k into aux buffers.
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, /*multipleBlocksPerRow=*/true,
        /*mergeBlocks=*/false>(
        nullptr, rowLogits, rowStart, rowEnd, myOutIndicesAux, myOutLogitsAux, stride1, topK, smem);

    // Make sure all writes from this block to outIndicesAux/outLogitsAux are
    // visible to every other block before we publish via the counter.
    __threadfence();
    __syncthreads();

    __shared__ int s_isLast;
    if (threadIdx.x == 0)
    {
        int prev = atomicAdd(&doneCounter[rowIdx], 1);
        s_isLast = (prev == numBlocksPerRow - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_isLast)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // Reset the counter so the buffer is reusable on the next call without
    // requiring an external memset.
    if (threadIdx.x == 0)
    {
        doneCounter[rowIdx] = 0;
    }

    // Last block: merge the numBlocksPerRow per-block top-ks into the final
    // top-k. Same shared-memory union is reused (the part1 phase already
    // returned smemFinal/smemThresholdBinIdx/etc to a known-undefined state,
    // and topKPerRowJob initializes them again at entry).
    int64_t mergeBase = static_cast<int64_t>(rowIdx) * numBlocksPerRow * topK;
    int* mergeIndicesIn = outIndicesAux + mergeBase;
    float* mergeLogitsIn = outLogitsAux + mergeBase;
    int* mergedOut = outIndices + static_cast<int64_t>(rowIdx) * topK;
    int mergeRowEnd = numBlocksPerRow * topK;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, /*multipleBlocksPerRow=*/false,
        /*mergeBlocks=*/true>(mergeIndicesIn, mergeLogitsIn, /*rowStart=*/0, mergeRowEnd, mergedOut,
        /*outLogits=*/nullptr,
        /*stride1=*/1, topK, smem);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

namespace
{

// ===========================================================================
// Multi-pass radix path (opt-in via scratch != nullptr).
//
// One launch per radix pass (top-11 half-precision bits, then bits [21..32),
// [10..21), [0..10)). State per row lives in a small RadixState struct in
// DRAM; the candidate data is two ping-pong buffers sized to the input row
// length. After 4 passes any remaining bit-level ties are emitted directly
// to fill the top-k.
//
// Eligibility (require !is_prefill): three nested tiers where the
// single-block radix-final pass and the fused last-block-merge kernel both
// under-use the GPU.
//   inner: bs ≤  32 / seq ≥  65k
//   mid  : bs ≤  64 / seq ≥ 131k
//   high : bs ≤ 256 / seq ≥ 524k
// ===========================================================================
static constexpr int kMultiPassRadixLowMaxRows = 32;
static constexpr int kMultiPassRadixLowMinSeqLen = 65536;
static constexpr int kMultiPassRadixMidMaxRows = 64;
static constexpr int kMultiPassRadixMidMinSeqLen = 131072;
static constexpr int kMultiPassRadixHighMaxRows = 256;
static constexpr int kMultiPassRadixHighMinSeqLen = 524288;

static constexpr int kRadixBins = 2048;

static inline bool multi_pass_radix_eligible(int numRows, int numColumns)
{
    if (numRows <= kMultiPassRadixLowMaxRows && numColumns >= kMultiPassRadixLowMinSeqLen)
        return true;
    if (numRows <= kMultiPassRadixMidMaxRows && numColumns >= kMultiPassRadixMidMinSeqLen)
        return true;
    if (numRows <= kMultiPassRadixHighMaxRows && numColumns >= kMultiPassRadixHighMinSeqLen)
        return true;
    return false;
}

// Per-row state for the multi-pass radix path (in DRAM scratch). Each pass's
// last block (selected via the `finishedBlocks` atomic) prefix-scans the
// global histogram and picks the threshold for the next pass — same pattern
// as `topKPerRowDecodeFused`'s in-block merge, generalised across passes.
struct alignas(64) RadixState
{
    int candCount;      // candidates entering current pass
    int outIdx;         // running outIndices write position
    int kRemaining;     // top-k slots remaining; equals topK - outIdx
    int filterCnt;      // running candidate-buf write position
    int thresholdBin;   // threshold bin from PRIOR pass (read at filter
                        // time; overwritten at last-block stage with this
                        // pass's threshold for the next pass to consume)
    int finishedBlocks; // last-block atomic counter, reset between passes
    int thresholdLess;  // cumsum prefix at thresholdBin (count of bins below it);
                        // used by pass-3 final-emit to route ties (bin == threshold)
                        // into a disjoint slot range so they don't race definite
                        // top-k items on the output counter.
    int padding[1];
};

// Common last-block trailer: prefix-scan the merged global histogram, locate
// the bin where the running count crosses kRemaining, stash to state for the
// next pass. step < 3 also resets the per-row global histogram. In step 1
// the trailer writes st.candCount = full row length for pass 2 to consume
// (pass 1 reads the length inline from seqLens, so no init kernel is needed).
template <int kThreads, int step>
__device__ __forceinline__ void radixLastBlockTrailer(int* gHist, RadixState& st, int topK, int rowFullLen)
{
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scanStorage;
    __shared__ int s_thresholdBin;
    __shared__ int s_runningBefore;
    __shared__ int s_thresholdCount;

    if (threadIdx.x == 0)
    {
        s_thresholdBin = -1;
        s_runningBefore = 0;
        s_thresholdCount = 0;
    }
    __syncthreads();

    // kRemaining for THIS pass:
    //   step 1: topK (no auto-promotes have happened yet).
    //   step 2/3: topK - outIdx (where outIdx was atomic-incremented during
    //             the filter loop). The histogram for the next pass's
    //             threshold pick must target this fresh value, not the
    //             stale state.kRemaining left over from the previous pass.
    int const kRem = (step == 1) ? topK : (topK - st.outIdx);
    if (kRem <= 0)
    {
        // All top-k slots already filled by auto-promotes in this pass —
        // no need for a next pass to emit anything.
        if (threadIdx.x == 0)
        {
            st.thresholdBin = -1;
            st.candCount = 0;
            st.kRemaining = 0;
            st.finishedBlocks = 0;
            st.filterCnt = 0;
            (void) rowFullLen;
        }
        if constexpr (step < 3)
        {
            __syncthreads();
            for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
                gHist[i] = 0;
        }
        return;
    }
    constexpr int kRoundsPerScan = kRadixBins / kThreads;
    int running = 0;
    for (int r = 0; r < kRoundsPerScan; ++r)
    {
        int bin = r * kThreads + threadIdx.x;
        int c = gHist[bin];
        int prefix, total;
        Scan(scanStorage).ExclusiveSum(c, prefix, total);
        prefix += running;
        int next = prefix + c;
        if (prefix < kRem && next >= kRem && s_thresholdBin == -1)
        {
            atomicCAS(&s_thresholdBin, -1, bin);
            if (s_thresholdBin == bin)
            {
                s_runningBefore = prefix;
                s_thresholdCount = c;
            }
        }
        running += total;
        __syncthreads();
        if (s_thresholdBin != -1)
            break;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // If the cumsum over the whole histogram never reached kRem the row
        // has fewer items than we still need (e.g. decode rows shorter than
        // topK). Set thresholdBin to a sentinel above any valid bin so the
        // next pass's `bin < thresholdBin` test accepts every surviving
        // candidate as auto-promote.
        st.thresholdBin = (s_thresholdBin == -1) ? kRadixBins : s_thresholdBin;
        // s_runningBefore is the count of histogram items in bins < threshold.
        // In the sentinel case it stays at its init 0; that's fine because
        // pass-3's inline final-emit only uses thresholdLess to position the
        // ties (bin == threshold) write base, and the sentinel branch has
        // no items in the threshold bin (the sentinel is above all valid bins).
        st.thresholdLess = s_runningBefore;
        st.finishedBlocks = 0;
        if constexpr (step == 1)
        {
            // Pass 2 also scans the full row from `logits`, so it reads
            // st.candCount = full row length. Pass 1 did not write to it
            // and the state struct started at zero (cudaMemsetAsync).
            st.candCount = rowFullLen;
            (void) s_thresholdCount;
            (void) kRem;
        }
        else
        {
            st.candCount = st.filterCnt;
            int newKRem = topK - st.outIdx;
            if (newKRem < 0)
                newKRem = 0;
            st.kRemaining = newKRem;
            st.filterCnt = 0;
        }
    }
    if constexpr (step < 3)
    {
        __syncthreads();
        for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
            gHist[i] = 0;
    }
}

// Fused histogram + filter pass kernel.
//
//   step == 1: pass 1, no filter, just histogram top-11 bits of the row.
//   step == 2: pass 2 reads `logits` (pass 1 didn't write a candidate buffer),
//              for each item:
//                bin1 < thresholdBin1 → write to outIndices (auto-promote)
//                bin1 == thresholdBin1 → append to candBufOut, count its bin2
//                                        in the histogram for pass 2
//                else                 → drop
//   step == 3: same as step 2 but reads `candBufIn` (pass 2's output) and
//              uses extractBinIdx<2> for the prior-bits check, extractBinIdx<3>
//              for the histogram.
//
// Last block of every pass runs `radixLastBlockTrailer` to compute the
// next pass's threshold and reset cross-pass state.
template <int kThreads, int step, typename InputT = float>
static __global__ __launch_bounds__(kThreads) void radixPassKernel(InputT const* logits, int const* seqLens,
    int* outIndices, int const* candBufIn, int* candBufOut, int* histograms, RadixState* state, int stride0, int next_n,
    int topK)
{
    int rowIdx = blockIdx.y;
    int blockInRow = blockIdx.x;
    int blocksPerRow = gridDim.x;

    RadixState& st = state[rowIdx];
    int* gHist = histograms + static_cast<int64_t>(rowIdx) * kRadixBins;

    __shared__ int sHist[kRadixBins];
    for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
        sHist[i] = 0;
    __syncthreads();

    if constexpr (step == 1)
    {
        // Read seqLen inline so pass 1 does not depend on st.candCount being
        // pre-initialised by a separate init kernel; the cudaMemsetAsync that
        // zeroes state+histograms together is enough. Pass-1 trailer below
        // writes st.candCount = seqLens[rowIdx] for pass 2 to consume.
        int const rowEnd = seqLens[rowIdx / next_n] - next_n + (rowIdx % next_n) + 1;
        InputT const* in = logits + static_cast<int64_t>(rowIdx) * stride0;
        size_t threadRank = static_cast<size_t>(blockInRow) * kThreads + threadIdx.x;
        size_t numThreads = static_cast<size_t>(blocksPerRow) * kThreads;
        auto f = [&](InputT vIn, size_t /*idx*/)
        {
            float const v = static_cast<float>(vIn);
            uint32_t bin = extractBinIdx<step>(v);
            atomicAdd(&sHist[bin], 1);
        };
        vectorized_process(threadRank, numThreads, in, static_cast<int>(rowEnd), f);
    }
    else if constexpr (step == 2)
    {
        int const rowEnd = st.candCount;
        InputT const* in = logits + static_cast<int64_t>(rowIdx) * stride0;
        int* outIdxArr = outIndices + static_cast<int64_t>(rowIdx) * topK;
        int* candArr = candBufOut + static_cast<int64_t>(rowIdx) * stride0;
        int const prevThresh = st.thresholdBin;
        size_t threadRank = static_cast<size_t>(blockInRow) * kThreads + threadIdx.x;
        size_t numThreads = static_cast<size_t>(blocksPerRow) * kThreads;
        auto f = [&](InputT vIn, size_t i)
        {
            float const v = static_cast<float>(vIn);
            int bin1 = static_cast<int>(extractBinIdx<1>(v));
            if (bin1 < prevThresh)
            {
                int pos = atomicAdd(&st.outIdx, 1);
                if (pos < topK)
                    outIdxArr[pos] = static_cast<int>(i);
            }
            else if (bin1 == prevThresh)
            {
                int pos = atomicAdd(&st.filterCnt, 1);
                candArr[pos] = static_cast<int>(i);
                uint32_t bin2 = extractBinIdx<step>(v);
                atomicAdd(&sHist[bin2], 1);
            }
        };
        vectorized_process(threadRank, numThreads, in, static_cast<int>(rowEnd), f);
    }
    else // step == 3
    {
        int const candCnt = st.candCount;
        int const* candArrIn = candBufIn + static_cast<int64_t>(rowIdx) * stride0;
        InputT const* in = logits + static_cast<int64_t>(rowIdx) * stride0;
        int* outIdxArr = outIndices + static_cast<int64_t>(rowIdx) * topK;
        int* candArrOut = candBufOut + static_cast<int64_t>(rowIdx) * stride0;
        int const prevThresh = st.thresholdBin;
        for (int i = blockInRow * kThreads + threadIdx.x; i < candCnt; i += blocksPerRow * kThreads)
        {
            int srcIdx = candArrIn[i];
            float v = static_cast<float>(in[srcIdx]);
            int bin2 = static_cast<int>(extractBinIdx<2>(v));
            if (bin2 < prevThresh)
            {
                int pos = atomicAdd(&st.outIdx, 1);
                if (pos < topK)
                    outIdxArr[pos] = srcIdx;
            }
            else if (bin2 == prevThresh)
            {
                int pos = atomicAdd(&st.filterCnt, 1);
                candArrOut[pos] = srcIdx;
                uint32_t bin3 = extractBinIdx<step>(v);
                atomicAdd(&sHist[bin3], 1);
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
    {
        int c = sHist[i];
        if (c)
            atomicAdd(&gHist[i], c);
    }

    __threadfence();
    __shared__ int isLast;
    if (threadIdx.x == 0)
    {
        int prev = atomicAdd(&st.finishedBlocks, 1);
        isLast = (prev == blocksPerRow - 1) ? 1 : 0;
    }
    __syncthreads();
    if (!isLast)
        return;

    int const rowFullLen = (step == 1) ? (seqLens[rowIdx / next_n] - next_n + (rowIdx % next_n) + 1) : 0;
    radixLastBlockTrailer<kThreads, step>(gHist, st, topK, rowFullLen);

    if constexpr (step == 3)
    {
        // Final emit, folded into the last block of pass 3. Scan candBufOut
        // (top 22 bits == thresholdBin2) and route items into outIndices by
        // bin3 vs thresholdBin3. Two-counter scheme so threshold-bin ties
        // don't race definite top-k items on the same atomic:
        //   bin3 <  thresh3  → slots [ltBase,           ltBase + prefix3)
        //   bin3 == thresh3  → slots [ltBase + prefix3, topK)
        __syncthreads();
        int const filterCnt = st.candCount;
        int const thresh3 = st.thresholdBin;
        int const prefix3 = st.thresholdLess;
        int const* candArr = candBufOut + static_cast<int64_t>(rowIdx) * stride0;
        InputT const* in = logits + static_cast<int64_t>(rowIdx) * stride0;
        int* outIdxArr = outIndices + static_cast<int64_t>(rowIdx) * topK;
        int const ltBase = st.outIdx;    // already at outBase here
        int const eqBase = ltBase + prefix3;
        int const eqCap = topK - eqBase; // ≥ 0 by trailer invariant
        __shared__ int sEqEmitted;
        if (threadIdx.x == 0)
            sEqEmitted = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < filterCnt; i += kThreads)
        {
            int srcIdx = candArr[i];
            float v = static_cast<float>(in[srcIdx]);
            int bin3 = static_cast<int>(extractBinIdx<3>(v));
            if (bin3 < thresh3)
            {
                // atomicAdd on st.outIdx is safe: by construction exactly
                // prefix3 items fall in this branch, so pos stays in
                // [ltBase, eqBase) which is strictly inside [0, topK).
                int pos = atomicAdd(&st.outIdx, 1);
                outIdxArr[pos] = srcIdx;
            }
            else if (bin3 == thresh3)
            {
                int pos = atomicAdd(&sEqEmitted, 1);
                if (pos < eqCap)
                    outIdxArr[eqBase + pos] = srcIdx;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int eq = sEqEmitted < eqCap ? sEqEmitted : eqCap;
            int filled = eqBase + eq;
            if (filled > topK)
                filled = topK;
            for (int i = filled; i < topK; ++i)
                outIdxArr[i] = -1;
        }
        // Reset the per-row global histogram and st.outIdx so the next call
        // sees a clean state without a per-call cudaMemsetAsync. Caller must
        // zero-initialize the scratch buffer before the first call.
        __syncthreads();
        for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
            gHist[i] = 0;
        if (threadIdx.x == 0)
            st.outIdx = 0;
    }
}

// Scratch layout (uint8 buffer, 64-byte aligned regions):
//   RadixState[numRows]
//   int histograms[numRows * kRadixBins]    (zeroed on first call by
//                                           torch::zeros allocator; pass-3
//                                           trailer zeroes for subsequent
//                                           calls)
//   int candBuf1[numRows * stride0]         (pass 2 → pass 3 input)
//   int candBuf2[numRows * stride0]         (pass 3 → fused final filter)
static size_t radixScratchBytes(int numRows, int numColumns)
{
    auto roundUp = [](size_t x) { return (x + 63) & ~size_t(63); };
    size_t s = 0;
    s += roundUp(sizeof(RadixState) * numRows);
    s += roundUp(sizeof(int) * static_cast<size_t>(numRows) * kRadixBins);
    s += roundUp(sizeof(int) * static_cast<size_t>(numRows) * numColumns);
    s += roundUp(sizeof(int) * static_cast<size_t>(numRows) * numColumns);
    return s;
}

template <typename InputT>
static void launchMultiPassRadix(void* scratch, InputT const* logits, int const* seqLens, int* outIndices, int numRows,
    int numColumns, int topK, int stride0, int next_n, cudaLaunchAttribute const* attrs, cudaStream_t stream)
{
    auto roundUp = [](size_t x) { return (x + 63) & ~size_t(63); };
    char* base = static_cast<char*>(scratch);
    RadixState* state = reinterpret_cast<RadixState*>(base);
    base += roundUp(sizeof(RadixState) * numRows);
    int* histograms = reinterpret_cast<int*>(base);
    base += roundUp(sizeof(int) * static_cast<size_t>(numRows) * kRadixBins);
    int* candBuf1 = reinterpret_cast<int*>(base);
    base += roundUp(sizeof(int) * static_cast<size_t>(numRows) * numColumns);
    int* candBuf2 = reinterpret_cast<int*>(base);

    int sm_cnt = 132;
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev);
    }
    // Block fan-out heuristic: target ~4 active blocks/SM (one wave at the
    // achievable occupancy of radixPassKernel<512, 1>), with a per-block work
    // floor of 2048 items (4 items/thread at 512-wide).
    int targetTotalBlocks = sm_cnt * 4;
    int numBlocksPerRow = (targetTotalBlocks + numRows - 1) / numRows;
    int maxByCols = numColumns / 2048;
    if (numBlocksPerRow > maxByCols)
        numBlocksPerRow = maxByCols;
    if (numBlocksPerRow < 1)
        numBlocksPerRow = 1;

    constexpr int kPassThreads = 512;

    auto launchPass = [&](void const* kernel, int const* candIn, int* candOut)
    {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(numBlocksPerRow, numRows);
        cfg.blockDim = kPassThreads;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cfg.numAttrs = 1;
        cfg.attrs = const_cast<cudaLaunchAttribute*>(attrs);
        void* args[] = {(void*) &logits, (void*) &seqLens, (void*) &outIndices, (void*) &candIn, (void*) &candOut,
            (void*) &histograms, (void*) &state, (void*) &stride0, (void*) &next_n, (void*) &topK};
        cudaLaunchKernelExC(&cfg, kernel, args);
    };

    launchPass(reinterpret_cast<void const*>(&radixPassKernel<kPassThreads, 1, InputT>), (int const*) nullptr,
        (int*) nullptr);
    launchPass(reinterpret_cast<void const*>(&radixPassKernel<kPassThreads, 2, InputT>), (int const*) nullptr, candBuf1);
    // Pass 3 emits the final top-K inline in its last-block trailer (see
    // radixPassKernel<step=3>) instead of requiring a separate filter launch.
    launchPass(
        reinterpret_cast<void const*>(&radixPassKernel<kPassThreads, 3, InputT>), (int const*) candBuf1, candBuf2);
}

// ============================================================================
// Unified dispatcher (fp32, bf16, fp16 all go through this).
// ============================================================================
// The TPR-family kernels (insertion-sort, single-block radix, 2-launch
// split-work, fused split-work, multi-pass radix) are all templated on
// `InputT` — logits are read with `static_cast<float>(InputT)` at HBM-read
// sites, so accuracy is identical to up-casting the input tensor to fp32
// before invoking the kernel. Aux buffers (outLogitsAux / scratch) stay fp32
// regardless of InputT.
template <typename InputT>
void invokeIndexerTopKDecodeImpl(InputT const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK, int const* preIdx, int const preIdxStride,
    int const preIdxCount, InputT* heuristicScratch, int* doneCounterScratch, cudaStream_t const stream, void* scratch,
    size_t scratchBytes, bool is_prefill)
{
    // Opt-in fast paths (added in the TPRv2 port — see commit message):
    //   - multi-pass radix path (low-bs/long-seq corner that the
    //     single-block radix loses on). Requires the caller to provide
    //     `scratch` of at least `indexerTopKDecodeScratchBytes(numRows,
    //     numColumns, topK)` bytes. Suppressed when is_prefill (the prefill
    //     entry handles tiny rows with its own short-row short-circuit).
    //   - Fused single-launch multi-block + last-block-merge variant of the
    //     `numColumns ≥ splitWorkThreshold` tier. Requires the caller to
    //     provide `doneCounterScratch` (one int per row, zero-initialized).
    //     When omitted, the original 2-launch split-work path is used.
    bool const useMultiPassRadixPath
        = !is_prefill && multi_pass_radix_eligible(numRows, numColumns) && scratch != nullptr && scratchBytes > 0;

    // kSortingAlgorithmThreshold no longer selects the final-sort algorithm
    // (that's runtime now — see the sort-selection block in topKPerRowJob).
    // It now gates the CTA-width choice inside the single-block tier between
    // a 256-thread "narrow" block (more occupancy, used when the per-row scan
    // is short) and a 512-thread "wide" block (more per-row parallelism).
    // The numRows + is_prefill aware thresholds are kept from the previous
    // insertion-vs-radix tuning since they correspond to the same shape
    // boundaries where narrow CTAs stay profitable.
    //
    //   is_prefill:           per-row work is bounded by the small (1..bs)
    //                         row lengths the prefill harness emits, so the
    //                         inner short-row short-circuit handles it; lift
    //                         the threshold out of the way so the narrow
    //                         block stays in use.
    //   numRows >= 4096:      narrow block stretches up to the split-work
    //                         boundary; the cheap path saturates the GPU.
    //   numRows >= 1024:      narrow block to 100k (covers seq~64k-98k).
    //   else:                 12288 (upstream default).
    int const kSortingAlgorithmThreshold = is_prefill ? (1 << 30)
        : (numRows >= 4096)                           ? 200 * 1000
        : (numRows >= 1024)                           ? 100 * 1000
                                                      : 12288;
    // Split-work tier cutoff: callers can still override with `splitWorkThreshold > 0`.
    //   numRows >= 128:       single-block always saturates; never split.
    //   numRows >= 64:        single-block to ~524k.
    //   numRows > 8:          200k (upstream default).
    //   numRows <= 8:         fused split-work earlier (65k) — low-bs path
    //                         under-uses the GPU otherwise.
    int const adaptiveSplitWorkThreshold = is_prefill ? (1 << 30)
        : (numRows >= 128)                            ? (1 << 30)
        : (numRows >= 64)                             ? 524 * 1024
        : (numRows > 8)                               ? 200 * 1000
                                                      : 65 * 1024;
    int const effectiveSplitWorkThreshold = splitWorkThreshold > 0 ? splitWorkThreshold : adaptiveSplitWorkThreshold;
    constexpr int kNumThreadsPerBlock = 512;
    constexpr int kNumThreadsPerBlockNarrow = 256;

    // Within the single-block path, 256-thread CTAs unlock more occupancy at
    // high bs but 512-thread CTAs win for low bs (per-row parallelism). The
    // bound on totalWork avoids picking 256 threads for high-bs but tiny
    // rows. The 256-thread variant is only safe when the per-row scan is
    // short enough — kSortingAlgorithmThreshold caps that.
    bool const useNarrowBlock = (numRows >= 256)
        && (static_cast<int64_t>(numRows) * numColumns >= (4LL << 20))
        && (numColumns < kSortingAlgorithmThreshold);

    // GVR ↔ TPR routing rule (applies to fp32, bf16, fp16 alike now that all
    // TPR tiers are InputT-templated):
    //   numColumns <= 16384  → TPR (small-N: GVR's fixed Phase-1/4 overhead
    //                                ~11 µs dominates; insertion / single-block
    //                                radix wins. 16K itself stays on TPR.)
    //   numRows    <= 32     → TPR (low-BS: GVR can't recoup setup at any N;
    //                                also keeps the multi-pass radix path
    //                                eligibility zone (BS ≤ 32) reachable.)
    //   otherwise            → GVR (when its eligibility predicate also holds;
    //                                otherwise fall through to TPR).
    bool const isSupportedTopK = (topK == 512 || topK == 1024 || topK == 2048);
    bool const canUseHeuristic = preIdx != nullptr && stride1 == 1 && isSupportedTopK && preIdxCount == topK
        && preIdxStride >= preIdxCount && heuristicScratch != nullptr && numColumns > 16384 && numRows > 32;

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
            fprintf(stderr, "[Scheme X] numRows=%d numColumns=%d -> %s path\n", numRows, numColumns,
                canUseHeuristic ? "Heuristic" : "Radix");
        }
    }

    if (canUseHeuristic)
    {
        launchHeuristicTopKDecode(logits, seqLens, preIdx, indices, heuristicScratch, stride0, next_n, topK,
            preIdxStride, preIdxCount, numRows, stream);
    }
    else if (useMultiPassRadixPath)
    {
        TLLM_CHECK_WITH_INFO(scratchBytes >= radixScratchBytes(numRows, numColumns),
            "indexer top-k multi-pass radix path: scratch buffer too small.");
        cudaLaunchAttribute radixAttrs[1];
        radixAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        radixAttrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        launchMultiPassRadix<InputT>(
            scratch, logits, seqLens, indices, numRows, numColumns, topK, stride0, next_n, radixAttrs, stream);
    }
    else if (numColumns < effectiveSplitWorkThreshold)
    {
        // Single-block adaptive: insertion vs radix sort is picked at runtime
        // inside topKPerRowJob based on the threshold-bin candidate count.
        // CTA width is picked here based on useNarrowBlock (256 vs 512).
        auto launchSingleBlock = [&](auto kernel_instance, int blockDim)
        {
            cudaLaunchConfig_t config;
            config.gridDim = numRows;
            config.blockDim = blockDim;
            config.dynamicSmemBytes = topK * sizeof(int32_t);
            config.stream = stream;
            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
            config.numAttrs = 1;
            config.attrs = attrs;

            cudaLaunchKernelEx(&config, kernel_instance, logits, seqLens, indices, stride0, stride1, topK, next_n,
                /*outLogits=*/nullptr, /*numBlocksToMerge=*/0, /*indices=*/nullptr);
        };

        if (useNarrowBlock)
        {
            launchSingleBlock(&topKPerRowDecode<kNumThreadsPerBlockNarrow, /*multipleBlocksPerRow=*/false,
                                  /*mergeBlocks=*/false, InputT>,
                kNumThreadsPerBlockNarrow);
        }
        else
        {
            launchSingleBlock(&topKPerRowDecode<kNumThreadsPerBlock, /*multipleBlocksPerRow=*/false,
                                  /*mergeBlocks=*/false, InputT>,
                kNumThreadsPerBlock);
        }
    }
    else if (doneCounterScratch != nullptr)
    {
        // Fused multi-block path: per-block top-k followed by last-block
        // in-kernel merge via a global atomic counter on doneCounterScratch.
        // Single launch replaces the two-launch part1+part2 split-work below.
        //
        // numBlocksPerRow defaults to 10; bumped at the very-low-bs corners
        // where 10 blocks/row leaves the GPU under-filled. The merge phase
        // processes (numBlocksPerRow * topK) candidates in a single block, so
        // we never bump past where the merge would dominate. Cutoffs below
        // come from the mid-long-decode sweep.
        TLLM_CHECK_WITH_INFO(outLogitsAux != nullptr && outIndicesAux != nullptr,
            "Fused split-work path requires both outLogitsAux and outIndicesAux to be non-null.");
        int const numBlocksPerRow = indexerTopKDecodeFusedAuxBlocksPerRow(numRows, numColumns);

        // 1024-thread variant wins once per-block scan is large enough to
        // amortize block-scope syncs (~30k); below that the 512-thread
        // narrow block hides sync cost better.
        bool const useWideBlock = (numColumns / numBlocksPerRow) >= 30000;
        constexpr int kFusedNarrowThreadsPerBlock = 512;
        int const blockDim = useWideBlock ? 1024 : kFusedNarrowThreadsPerBlock;

        auto* kernel_512 = &topKPerRowDecodeFused<kFusedNarrowThreadsPerBlock, InputT>;
        auto* kernel_1024 = &topKPerRowDecodeFused<1024, InputT>;
        // Opt-in to the smem budget the kernel actually needs. The kernel's
        // static smem (≈19KB for the 512-thread variant, ≈38KB for the
        // 1024-thread one) plus the 2*topK*sizeof(int32_t) dynamic smem
        // requested at launch can exceed the default per-block cap (48KB on
        // many archs, including sm_120). Setting MaxDynamicSharedMemorySize
        // to a value that, together with the kernel's static smem, exceeds
        // cudaDevAttrMaxSharedMemoryPerBlockOptin causes cudaFuncSetAttribute
        // to fail and the subsequent cudaLaunchKernelEx to return
        // cudaErrorInvalidValue. Sizing the attribute to exactly the dynamic
        // smem the launch uses keeps this comfortably under the optin cap on
        // every Hopper / Blackwell arch we target.
        size_t const fusedDynamicSmemBytes = 2 * topK * sizeof(int32_t);
        static bool s_attr_512 = false;
        static bool s_attr_1024 = false;
        if (useWideBlock && !s_attr_1024)
        {
            cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel_1024),
                cudaFuncAttributeMaxDynamicSharedMemorySize, fusedDynamicSmemBytes);
            s_attr_1024 = true;
        }
        else if (!useWideBlock && !s_attr_512)
        {
            cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel_512),
                cudaFuncAttributeMaxDynamicSharedMemorySize, fusedDynamicSmemBytes);
            s_attr_512 = true;
        }

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();

        cudaLaunchConfig_t config{};
        config.gridDim = dim3(numRows, numBlocksPerRow);
        config.blockDim = blockDim;
        config.dynamicSmemBytes = 2 * topK * sizeof(int32_t);
        config.stream = stream;
        config.numAttrs = 1;
        config.attrs = attrs;

        if (useWideBlock)
            cudaLaunchKernelEx(&config, kernel_1024, logits, seqLens, indices, outIndicesAux, outLogitsAux,
                doneCounterScratch, stride0, stride1, topK, next_n, numBlocksPerRow);
        else
            cudaLaunchKernelEx(&config, kernel_512, logits, seqLens, indices, outIndicesAux, outLogitsAux,
                doneCounterScratch, stride0, stride1, topK, next_n, numBlocksPerRow);
    }
    else
    {
        // Long sequences are run in two steps.
        TLLM_CHECK_WITH_INFO(outLogitsAux != nullptr && outIndicesAux != nullptr,
            "Split-work path requires both outLogitsAux and outIndicesAux to be non-null.");
        constexpr auto multipleBlocksPerRowConfig = 10;
        // Part 1: reads InputT logits, writes per-block top-K into the fp32
        // aux buffers (outLogitsAux / outIndicesAux).
        auto* kernel_instance_part1
            = &topKPerRowDecode<kNumThreadsPerBlock, /*multipleBlocksPerRow=*/true,
                /*mergeBlocks=*/false, InputT>;
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
            next_n, outLogitsAux, 0, nullptr);

        // Part 2 (merge): reads the fp32 aux buffer as its "logits" input, so
        // it always runs with InputT=float regardless of the original input
        // dtype.
        constexpr int kNumThreadsPerBlockMerge = 1024;
        auto* kernel_instance_part2 = &topKPerRowDecode<kNumThreadsPerBlockMerge,
            /*multipleBlocksPerRow=*/false, /*mergeBlocks=*/true>;
        cudaLaunchConfig_t config_part2;
        config_part2.gridDim = numRows;
        config_part2.blockDim = kNumThreadsPerBlockMerge;
        config_part2.dynamicSmemBytes = topK * sizeof(int32_t);
        config_part2.stream = stream;
        // Reuse attrs array since part1 kernel has already been launched
        config_part2.numAttrs = 1;
        config_part2.attrs = attrs;

        cudaLaunchKernelEx(&config_part2, kernel_instance_part2, outLogitsAux, seqLens, indices,
            multipleBlocksPerRowConfig * topK, 1, topK, next_n, nullptr, multipleBlocksPerRowConfig, outIndicesAux);
    }
    sync_check_cuda_error(stream);
}

} // anonymous namespace

void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK, int const* preIdx, int const preIdxStride,
    int const preIdxCount, float* heuristicScratch, int* doneCounterScratch, cudaStream_t const stream, void* scratch,
    size_t scratchBytes, bool is_prefill)
{
    invokeIndexerTopKDecodeImpl<float>(logits, seqLens, indices, outLogitsAux, outIndicesAux, splitWorkThreshold,
        numRows, numColumns, stride0, stride1, next_n, topK, preIdx, preIdxStride, preIdxCount, heuristicScratch,
        doneCounterScratch, stream, scratch, scratchBytes, is_prefill);
}

size_t indexerTopKDecodeScratchBytes(int numRows, int numColumns, int /*topK*/)
{
    // Returns the bytes the multi-pass radix path needs (state +
    // histograms + two candidate buffers).
    return radixScratchBytes(numRows, numColumns);
}

// bf16 / fp16 entries — full feature parity with fp32 via the unified
// invokeIndexerTopKDecodeImpl<InputT> template. All TPR-family kernels
// (insertion / single-block radix / 2-launch split-work / fused split-work /
// multi-pass radix) accept InputT now; aux buffers (outLogitsAux / scratch)
// remain fp32 regardless of dtype.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK, int const* preIdx, int const preIdxStride,
    int const preIdxCount, __nv_bfloat16* heuristicScratch, int* doneCounterScratch, cudaStream_t const stream,
    void* scratch, size_t scratchBytes, bool is_prefill)
{
    invokeIndexerTopKDecodeImpl<__nv_bfloat16>(logits, seqLens, indices, outLogitsAux, outIndicesAux, splitWorkThreshold,
        numRows, numColumns, stride0, stride1, next_n, topK, preIdx, preIdxStride, preIdxCount, heuristicScratch,
        doneCounterScratch, stream, scratch, scratchBytes, is_prefill);
}

void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK, int const* preIdx, int const preIdxStride,
    int const preIdxCount, __half* heuristicScratch, int* doneCounterScratch, cudaStream_t const stream, void* scratch,
    size_t scratchBytes, bool is_prefill)
{
    invokeIndexerTopKDecodeImpl<__half>(logits, seqLens, indices, outLogitsAux, outIndicesAux, splitWorkThreshold,
        numRows, numColumns, stride0, stride1, next_n, topK, preIdx, preIdxStride, preIdxCount, heuristicScratch,
        doneCounterScratch, stream, scratch, scratchBytes, is_prefill);
}

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK,
    cudaStream_t const stream)
{
    constexpr int kNumThreadsPerBlock = 512;

    // Single launch over all rows: topKPerRowPrefill / topKPerRowJob runtime-picks
    // insertion vs radix sort per row based on the threshold-bin candidate count
    // (see the sort-selection block in topKPerRowJob). Previously this function
    // split the row range at 12288 to compile-time-select the sort algorithm.
    topKPerRowPrefill<kNumThreadsPerBlock>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits, rowStarts, rowEnds, indices, stride0, stride1, topK, 0);

    sync_check_cuda_error(stream);
}

bool canIndexerTopKDecodeUseGvr(int numRows, int numColumns, int topK, int /*bytesPerElem*/)
{
    // The dispatcher now applies the same GVR ↔ TPR routing rule for fp32,
    // bf16, and fp16 (all TPR tiers are InputT-templated). `bytesPerElem` is
    // kept in the signature for source compatibility but no longer affects
    // the answer.
    bool const isSupportedTopK = (topK == 512 || topK == 1024 || topK == 2048);
    if (!isSupportedTopK)
    {
        return false;
    }
    return numColumns > 16384 && numRows > 32;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
