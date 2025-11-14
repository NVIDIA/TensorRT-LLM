/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t
{
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x)
{
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 0)
    {
        return bits >> 21;
    }
    else if constexpr (step == 1)
    {
        return (bits >> 10) & 0x7ff;
    }
    else
    {
        return bits & 0x3ff;
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
        const idxT len_cast = (len - skip_cnt) / items_per_scalar;

        for (idxT i = thread_rank; i < len_cast; i += num_threads)
        {
            wide.scalar = in_cast[i];
            const idxT real_i = skip_cnt + i * items_per_scalar;
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
        const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if (remain_i < len)
        {
            f(in[remain_i], remain_i);
        }
    }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kTopK, int kNumFinalItems, bool multipleBlocksPerRow,
    bool mergeBlocks, typename SmemFinalType, typename SmemOutputType>
__device__ bool processHistogramStep(int const* indices, float const* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx, int* smemFinalDstIdx,
    int* smemFinalBinSize, int* smemFoundTopKValues, SmemFinalType& smemFinal, int stride1, int rowStart)
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
    constexpr auto patternShift = step == 0 ? 0 : step == 1 ? 21 : 10;
    if constexpr (step == 1)
    {
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }
    else if constexpr (step == 2)
    {
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }

    auto distributeToBins = [&](float logit, int /* idx */ = 0)
    {
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
            float logit = logits[idx * stride1];
            distributeToBins(logit, idx);
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
        if (prefixSum < kTopK)
        {
            int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];

            if (nextPrefixSum >= kTopK)
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

    auto processBins = [&](float logit, int idx)
    {
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            if (binIdx < thresholdBinIdx)
            {
                // The element is part of the top-k selection
                int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

                if constexpr (mergeBlocks)
                {
                    smemOutput.indices[dstIdx] = indices[idx];
                }
                else if constexpr (multipleBlocksPerRow)
                {
                    smemOutput.indices[dstIdx] = idx + rowStart;
                    smemOutput.logits[dstIdx] = logit;
                }
                else
                {
                    smemOutput.indices[dstIdx] = idx;
                }
            }
            if constexpr (step < 2)
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
                else if (binIdx == thresholdBinIdx && smemFinalBinSize[0] > kNumFinalItems)
                {
                    // Load elements for next it
                }
            }
            else
            {
                if (binIdx == thresholdBinIdx)
                {
                    // The elements in the threshold bin share the same 32 bits at step 2
                    int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
                    if (dstIdx < kTopK)
                    {
                        if constexpr (mergeBlocks)
                        {
                            smemOutput.indices[dstIdx] = indices[idx];
                        }
                        else if constexpr (multipleBlocksPerRow)
                        {
                            smemOutput.indices[dstIdx] = idx + rowStart;
                            smemOutput.logits[dstIdx] = logit;
                        }
                        else
                        {
                            smemOutput.indices[dstIdx] = idx;
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
            float logit = logits[idx * stride1];
            processBins(logit, idx);
        }
    }

    // Make sure the elements are in shared memory.
    __syncthreads();

    // Check if we should continue to next step
    return smemFinalBinSize[0] > kNumFinalItems;
}

// Follows 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock, int kNumBins, int kTopK, bool useRadixSort, bool multipleBlocksPerRow = false,
    bool mergeBlocks = false>
static __device__ void topKPerRowJob(
    int const* indices, float const* logits, int rowStart, int rowEnd, int* outIndices, float* outLogits, int stride1)
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
    struct SmemOutputIndices
    {
        int indices[kTopK];
    };

    struct SmemOutputLogitsAndIndices
    {
        int indices[kTopK];
        float logits[kTopK];
    };

    using SmemOutput_t = std::conditional_t<multipleBlocksPerRow, SmemOutputLogitsAndIndices, SmemOutputIndices>;
    __shared__ SmemOutput_t smemOutput;

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
    if (rowLen <= kTopK)
    {
        for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock)
        {
            if constexpr (multipleBlocksPerRow)
            {
                outIndices[rowIt] = rowIt + rowStart;
                outLogits[rowIt] = logits[rowIt + rowStart];
            }
            else
            {
                outIndices[rowIt] = rowIt;
            }
        }
        for (int rowIt = rowLen + threadIdx.x; rowIt < kTopK; rowIt += kNumThreadsPerBlock)
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

    // Step 0: Process first 11 bits
    bool continueToNextStep = processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems,
        multipleBlocksPerRow, mergeBlocks>(indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart);

    if (continueToNextStep)
    {
        // Step 1: Process next 11 bits
        continueToNextStep = processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems,
            multipleBlocksPerRow, mergeBlocks>(indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart);

        if (continueToNextStep)
        {
            // Step 2: Process final 10 bits
            processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems, multipleBlocksPerRow,
                mergeBlocks>(indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
                smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart);
        }
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

                if (dstIdx < kTopK)
                {
                    smemOutput.indices[dstIdx] = finalIndices[ii];
                    if constexpr (multipleBlocksPerRow)
                    {
                        smemOutput.logits[dstIdx] = finalLogits[ii];
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
                if (outIndex + baseIdx < kTopK)
                {
                    smemOutput.indices[outIndex + baseIdx] = smemFinal.items.indices[i];
                    if constexpr (multipleBlocksPerRow)
                    {
                        smemOutput.logits[outIndex + baseIdx] = smemFinal.items.logits[i];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store to global memory.
    for (int i = threadIdx.x; i < kTopK; i += kNumThreadsPerBlock)
    {
        if constexpr (multipleBlocksPerRow)
        {
            outIndices[i] = smemOutput.indices[i];
            outLogits[i] = smemOutput.logits[i];
        }
        else
        {
            outIndices[i] = smemOutput.indices[i] - rowStart;
        }
    }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(float const* logits,
    int const* rowStarts, int const* rowEnds, int* outIndices, int stride0, int stride1, int const offsetIndex)
{
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The top-k width.
    static constexpr int kTopK = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x + offsetIndex;

    // The range of logits within the row.
    int rowStart = rowStarts[rowIdx];
    int rowEnd = rowEnds[rowIdx];

    // Local pointers to this block
    outIndices += rowIdx * kTopK;
    logits += rowIdx * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, kTopK, useRadixSort>(
        nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1);
}

template <int kNumThreadsPerBlock, bool useRadixSort, bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(float const* logits, int const* seqLens,
    int* outIndices, int stride0, int stride1, int next_n, float* outLogits = nullptr, int const numBlocksToMerge = 0,
    int const* indices = nullptr)
{
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The top-k width.
    static constexpr int kTopK = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x;

    // The range of logits within the row.
    int rowStart = 0;
    int seq_len = seqLens[rowIdx / next_n];
    int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

    // Local pointers to this block
    if constexpr (!multipleBlocksPerRow && !mergeBlocks)
    {
        outIndices += rowIdx * kTopK;
    }
    else if constexpr (multipleBlocksPerRow)
    {
        auto const blockSize = rowEnd / gridDim.y; // 16384 / 2 = 8192
        rowStart = blockSize * blockIdx.y;         // 8192 * 1 = 8192
        rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
        outIndices += rowIdx * gridDim.y * kTopK + blockIdx.y * kTopK;
        outLogits += rowIdx * gridDim.y * kTopK + blockIdx.y * kTopK;
    }
    else if constexpr (mergeBlocks)
    {
        rowEnd = numBlocksToMerge * kTopK;
        indices += rowIdx * numBlocksToMerge * kTopK;
        outIndices += rowIdx * kTopK;
    }
    logits += rowIdx * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, kTopK, useRadixSort, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1);
}

template <int kNumThreadsPerBlock = 512>
static __global__ void topKPerRowDecode(
    float const* logits, int const* seqLens, int* outIndices, int stride0, int stride1, int next_n)
{
    // The number of bins in the histogram.
    static constexpr int kNumBins = 512;

    // The top-k width.
    static constexpr int kTopK = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x;

    // The range of logits within the row.
    int rowStart = 0;
    int seq_len = seqLens[rowIdx / next_n];
    int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, kTopK>(logits, rowStart, rowEnd, rowIdx, outIndices, stride0, stride1);
}

void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* outIndices, float* auxLogits,
    int* auxIndices, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const index_topk, cudaStream_t const stream)
{
    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kNumThreadsPerBlock = 512;
    constexpr int kTopK = 2048;
    assert(index_topk == kTopK);

    if (numColumns < kSortingAlgorithmThreshold)
    {
        // Use insertion sort
        topKPerRowDecode<kNumThreadsPerBlock, false>
            <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits, seqLens, outIndices, stride0, stride1, next_n);
    }
    else if (numColumns < splitWorkThreshold)
    {
        // From this threshold, use radix sort instead
        topKPerRowDecode<kNumThreadsPerBlock, true>
            <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits, seqLens, outIndices, stride0, stride1, next_n);
    }
    else
    {
        // Long sequences are run in two steps
        constexpr auto multipleBlocksPerRowConfig = 10;
        topKPerRowDecode<kNumThreadsPerBlock, true, true>
            <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock, 0, stream>>>(
                logits, seqLens, outIndices, stride0, stride1, next_n);

        constexpr int kNumThreadsPerBlockMerge = 1024;
        topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>
            <<<numRows, kNumThreadsPerBlockMerge, 0, stream>>>(auxLogits, seqLens, outIndices,
                multipleBlocksPerRowConfig * kTopK, 1, next_n, nullptr, multipleBlocksPerRowConfig, auxIndices);
    }
    sync_check_cuda_error(stream);
}

void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const index_topk,
    cudaStream_t const stream)
{
    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kNumThreadsPerBlock = 512;
    assert(index_topk == 2048);

    int numInsertionBlocks = std::min(numRows, kSortingAlgorithmThreshold);
    topKPerRowPrefill<kNumThreadsPerBlock, false><<<numInsertionBlocks, kNumThreadsPerBlock, 0, stream>>>(
        logits, rowStarts, rowEnds, outIndices, stride0, stride1, 0);

    if (numRows > kSortingAlgorithmThreshold)
    {
        int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
        topKPerRowPrefill<kNumThreadsPerBlock, true><<<numRadixBlocks, kNumThreadsPerBlock, 0, stream>>>(
            logits, rowStarts, rowEnds, outIndices, stride0, stride1, kSortingAlgorithmThreshold);
    }
    sync_check_cuda_error(stream);
}

} // namespace tensorrt_llm::kernels
