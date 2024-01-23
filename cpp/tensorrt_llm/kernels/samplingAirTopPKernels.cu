/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include <cuda/std/limits>
#include <cuda_fp16.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
template <typename T, typename IdxT, typename AccT>
struct alignas(128) Counter
{
    // Address for input value and index
    T const* in;
    IdxT const* inIdx;

    // The original length of the input
    IdxT oriLen;

    // We are processing the values in multiple passes, from most significant to least
    // significant. In each pass, we keep the length of input (`len`) and the `sum` of
    // current pass, and update them at the end of the pass.
    AccT sum;
    IdxT len;
    float p;

    //  `previousLen` is the length of input in previous pass. Note that `previousLen`
    //  rather than `len` is used for the filtering step because filtering is indeed for
    //  previous pass.
    IdxT previousLen;

    // We determine the bits of the k_th value inside the mask processed by the pass. The
    // already known bits are stored in `kthValueBits`. It's used to discriminate a
    // element is a result (written to `out`), a candidate for next pass (written to
    // `outBuf`), or not useful (discarded). The bits that are not yet processed do not
    // matter for this purpose.
    typename cub::Traits<T>::UnsignedBits kthValueBits;

    // Record how many elements have passed filtering. It's used to determine the position
    // in the `outBuf` where an element should be written.
    alignas(128) IdxT filterCnt;

    // For a row inside a batch, we may launch multiple thread blocks. This counter is
    // used to determine if the current block is the last running block. If so, this block
    // will execute scan() and chooseBucket().
    alignas(128) unsigned int finishedBlockCnt;
};

/*******************************Functions*********************************/
using WideT = float4;

//! \brief Provide a ceiling division operation ie. ceil(a / b)
//! \tparam IntType supposed to be only integers for now!
template <typename IntType>
constexpr __host__ __device__ IntType ceilDiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}

//! \brief Provide an alignment function ie. ceil(a / b) * b
//! \tparam IntType supposed to be only integers for now!
template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b)
{
    return ceilDiv(a, b) * b;
}

//! \brief Calcute the number of buckets based on the number of bits per pass.
//! \tparam BitsPerPass. If BitsPerPass==11, the number of buckets is 2048. If BitsPerPass==8, the number of buckets is
//! 256.
template <int BitsPerPass>
__host__ __device__ int constexpr calcNumBuckets()
{
    return 1 << BitsPerPass;
}

//! \brief Calcute the number of passes based on the number of bits per pass.
//! \tparam BitsPerPass. If BitsPerPass==11, the number of passes is 3. If BitsPerPass==8, the number of passes is 4.
template <typename T, int BitsPerPass>
__host__ __device__ int constexpr calcNumPasses()
{
    return ceilDiv<int>(sizeof(T) * 8, BitsPerPass);
}

/**
 * This implementation processes input from the most to the least significant bit (Bit 0 is the least
 * significant (rightmost)). This way, we can skip some passes in the end at the cost of having an unsorted output.
 */
template <typename T, int BitsPerPass>
__device__ int constexpr calcsStartBit(int pass)
{
    int startBit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    if (startBit < 0)
    {
        startBit = 0;
    }
    return startBit;
}

template <typename T, int BitsPerPass>
__device__ unsigned constexpr calcMask(int pass)
{
    static_assert(BitsPerPass <= 31);
    int numBits = calcsStartBit<T, BitsPerPass>(pass - 1) - calcsStartBit<T, BitsPerPass>(pass);
    return (1 << numBits) - 1;
}

/**
 * Use CUB to twiddle bits.
 */
template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddleIn(T key, bool selectMin)
{
    auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
    bits = cub::Traits<T>::TwiddleIn(bits);
    if (!selectMin)
    {
        bits = ~bits;
    }
    return bits;
}

template <typename T>
__device__ T twiddleOut(typename cub::Traits<T>::UnsignedBits bits, bool selectMin)
{
    if (!selectMin)
    {
        bits = ~bits;
    }
    bits = cub::Traits<T>::TwiddleOut(bits);
    return reinterpret_cast<T&>(bits);
}

/**
 * Find the bucket based on the radix
 */
template <typename T, int BitsPerPass>
__device__ int calcBucket(T x, int startBit, unsigned mask, bool selectMin)
{
    static_assert(BitsPerPass <= sizeof(int) * 8 - 1, "BitsPerPass is too large that the result type could not be int");
    return (twiddleIn(x, selectMin) >> startBit) & mask;
}

/**
 * This function calculate the bufLen, which is the size of buffer.
 * When the number of candidates for next pass exceeds the bufLen, we choose not to store the candidates. Otherwise, we
 * will load candidates from the original input data.
 */
template <typename T, typename IdxT>
__host__ __device__ IdxT calcBufLen(IdxT len)
{
    // This ratio is calculated based on the element number.
    // If we choose to write the buffers, it means (sizeof(T)+sizeof(IdxT))*bufLen bytes of storing and loading.
    // To ensure we do not access more than len*sizeof(T) bytes. bufLen should be smaller than:
    // len*sizeof(T)/2*(sizeof(T) + sizeof(IdxT)) = len/(2 + sizeof(IdxT) * 2 / sizeof(T))).
    IdxT constexpr ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
    // Even such estimation is too conservative (due to the global coalescing access). So based on our experiments, we
    // further decrease bufLen by 1/8
    IdxT bufLen = len / (ratio * 8);

    // Align the address to 256 bytes
    bufLen = alignTo(bufLen, 256);
    return bufLen;
}

/**
 * Use ping-pong buffer and set the inBuf and outBuf based on the pass value.
 */
template <typename T, typename IdxT>
__host__ __device__ void setBufPointers(T const* in, IdxT const* inIdx, T* buf1, IdxT* idxBuf1, T* buf2, IdxT* idxBuf2,
    int pass, T const*& inBuf, IdxT const*& inIdxBuf, T*& outBuf, IdxT*& outIdxBuf)
{
    if (pass == 0)
    {
        inBuf = in;
        inIdxBuf = nullptr;
        outBuf = nullptr;
        outIdxBuf = nullptr;
    }
    else if (pass == 1)
    {
        inBuf = in;
        inIdxBuf = inIdx;
        outBuf = buf1;
        outIdxBuf = idxBuf1;
    }
    else if (pass % 2 == 0)
    {
        inBuf = buf1;
        inIdxBuf = idxBuf1;
        outBuf = buf2;
        outIdxBuf = idxBuf2;
    }
    else
    {
        inBuf = buf2;
        inIdxBuf = idxBuf2;
        outBuf = buf1;
        outIdxBuf = idxBuf1;
    }
}

//! \brief Map a Func over the input data, using vectorized load instructions if possible.
//! \tparam T element type
//! \tparam IdxT indexing type
//! \tparam Func void (T x, IdxT idx)
//! \param threadRank rank of the calling thread among all participating threads
//! \param numThreads number of the threads that participate in processing
//! \param in the input data
//! \param len the number of elements to read
//! \param f the lambda taking two arguments (T x, IdxT idx)
template <typename T, typename IdxT, typename Func>
__device__ void vectorizedProcess(size_t threadRank, size_t numThreads, T const* in, IdxT len, Func f)
{
    int constexpr WARP_SIZE = 32;
    if constexpr (sizeof(T) >= sizeof(WideT))
    {
        for (IdxT i = threadRank; i < len; i += numThreads)
        {
            f(in[i], i);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        int constexpr itemsPerScalar = sizeof(WideT) / sizeof(T);

        // TODO: it's UB
        union
        {
            WideT scalar;
            T array[itemsPerScalar];
        } wide;

        int skipCnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
        if (skipCnt > len)
        {
            skipCnt = len;
        }
        WideT const* inCast = reinterpret_cast<decltype(inCast)>(in + skipCnt);
        IdxT const lenCast = (len - skipCnt) / itemsPerScalar;

        for (IdxT i = threadRank; i < lenCast; i += numThreads)
        {
            wide.scalar = inCast[i];
            IdxT const real_i = skipCnt + i * itemsPerScalar;
#pragma unroll
            for (int j = 0; j < itemsPerScalar; ++j)
            {
                f(wide.array[j], real_i + j);
            }
        }

        static_assert(WARP_SIZE >= itemsPerScalar);
        // and because itemsPerScalar > skipCnt, WARP_SIZE > skipCnt
        // no need to use loop
        if (threadRank < skipCnt)
        {
            f(in[threadRank], threadRank);
        }
        // because lenCast = (len - skipCnt) / itemsPerScalar,
        // lenCast * itemsPerScalar + itemsPerScalar > len - skipCnt;
        // and so
        // len - (skipCnt + lenCast * itemsPerScalar) < itemsPerScalar <=
        // WARP_SIZE no need to use loop
        IdxT const remain_i = skipCnt + lenCast * itemsPerScalar + threadRank;
        if (remain_i < len)
        {
            f(in[remain_i], remain_i);
        }
    }
}

/**
 * Fused filtering of the current pass and building histogram for the next pass (see steps 4 & 1 in `airTopPSsampling`
 * description).
 */
template <typename T, typename IdxT, typename AccT, int BitsPerPass>
__device__ __forceinline__ void filterAndHistogram(T const* inBuf, IdxT const* inIdxBuf, T* outBuf, IdxT* outIdxBuf,
    int previousLen, Counter<T, IdxT, AccT>* counter, AccT* histogram, IdxT* countHistogram, int pass,
    float* outputLogProbs, float* cumLogProbs, IdxT** ids, IdxT const* endIds, IdxT* sequenceLengths,
    FinishedState* finishedOutput, int const batchId, bool earlyStop)
{
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");

    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    bool constexpr selectMin = false;
    __shared__ AccT histogramSmem[numBuckets];
    __shared__ IdxT countHistogramSmem[numBuckets];
    for (IdxT i = threadIdx.x; i < numBuckets; i += blockDim.x)
    {
        histogramSmem[i] = 0;
        countHistogramSmem[i] = 0;
    }
    __syncthreads();

    int const startBit = calcsStartBit<T, BitsPerPass>(pass);
    unsigned const mask = calcMask<T, BitsPerPass>(pass);

    if (pass == 0)
    {
        // Passed to vectorizedProcess, this function executes in all blocks in
        // parallel, i.e. the work is split along the input (both, in batches and
        // chunks of a single row). Later, the histograms are merged using
        // atomicAdd.
        auto f = [selectMin, startBit, mask](T value, IdxT)
        {
            int bucket = calcBucket<T, BitsPerPass>(value, startBit, mask, selectMin);
            if constexpr (std::is_same_v<T, half>)
            {
                atomicAdd(histogramSmem + bucket, __half2float(value));
            }
            else
            {
                atomicAdd(histogramSmem + bucket, value);
            }

            atomicAdd(countHistogramSmem + bucket, static_cast<IdxT>(1));
        };
        vectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(blockDim.x) * gridDim.x, inBuf, previousLen, f);
    }
    else
    {
        IdxT* pFilterCnt = &counter->filterCnt;
        auto const kthValueBits = counter->kthValueBits;
        int const previousStartBit = calcsStartBit<T, BitsPerPass>(pass - 1);

        // See the remark above on the distributed execution of `f` using
        // vectorizedProcess.
        auto f = [inIdxBuf, outBuf, outIdxBuf, selectMin, startBit, mask, previousStartBit, kthValueBits, pFilterCnt,
                     outputLogProbs, cumLogProbs, ids, endIds, sequenceLengths, finishedOutput, batchId,
                     earlyStop](T value, IdxT i)
        {
            auto const previousBits = (twiddleIn(value, selectMin) >> previousStartBit) << previousStartBit;
            if (previousBits == kthValueBits)
            {
                if (earlyStop)
                {
                    int const currentStep = sequenceLengths[batchId];
                    IdxT index = inIdxBuf ? inIdxBuf[i] : i;
                    ids[batchId][currentStep] = index;
                    epilogue(
                        value, index, outputLogProbs, cumLogProbs, endIds, sequenceLengths, finishedOutput, batchId);
                }
                if (outBuf)
                {
                    IdxT pos = atomicAdd(pFilterCnt, static_cast<IdxT>(1));
                    outBuf[pos] = value;
                    outIdxBuf[pos] = inIdxBuf ? inIdxBuf[i] : i;
                }

                int bucket = calcBucket<T, BitsPerPass>(value, startBit, mask, selectMin);
                if constexpr (std::is_same_v<T, half>)
                {
                    atomicAdd(histogramSmem + bucket, __half2float(value));
                }
                else
                {
                    atomicAdd(histogramSmem + bucket, value);
                }

                atomicAdd(countHistogramSmem + bucket, static_cast<IdxT>(1));
            }
        };
        vectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(blockDim.x) * gridDim.x, inBuf, previousLen, f);
    }

    __syncthreads();
    if (earlyStop)
    {
        return;
    }

    // merge histograms produced by individual blocks
    for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
    {
        if (histogramSmem[i] != 0)
        {
            atomicAdd(histogram + i, histogramSmem[i]);
        }
        if (countHistogramSmem[i] != 0)
        {
            atomicAdd(countHistogram + i, countHistogramSmem[i]);
        }
    }
}

/**
 *  Replace histogram with its own prefix sum (step 2 in `airTopPSsampling` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(volatile IdxT* histogram)
{
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    if constexpr (numBuckets >= BlockSize)
    {
        static_assert(numBuckets % BlockSize == 0);
        int constexpr itemsPerThread = numBuckets / BlockSize;
        typedef cub::BlockLoad<IdxT, BlockSize, itemsPerThread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
        typedef cub::BlockStore<IdxT, BlockSize, itemsPerThread, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
        typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

        __shared__ union
        {
            typename BlockLoad::TempStorage load;
            typename BlockScan::TempStorage scan;
            typename BlockStore::TempStorage store;
        } tempStorage;

        IdxT threadData[itemsPerThread];

        BlockLoad(tempStorage.load).Load(histogram, threadData);
        __syncthreads();

        BlockScan(tempStorage.scan).InclusiveSum(threadData, threadData);
        __syncthreads();

        BlockStore(tempStorage.store).Store(histogram, threadData);
    }
    else
    {
        typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
        __shared__ typename BlockScan::TempStorage tempStorage;

        IdxT threadData = 0;
        if (threadIdx.x < numBuckets)
        {
            threadData = histogram[threadIdx.x];
        }

        BlockScan(tempStorage).InclusiveSum(threadData, threadData);
        __syncthreads();

        if (threadIdx.x < numBuckets)
        {
            histogram[threadIdx.x] = threadData;
        }
    }
}

/**
 * Calculate in which bucket the k-th value will fall
 *  (steps 3 in `airTopPSsampling` description)
 */
template <typename T, typename IdxT, typename AccT, int BitsPerPass>
__device__ void chooseBucket(
    Counter<T, IdxT, AccT>* counter, AccT const* histogram, IdxT const* countHistogram, AccT const sum, int const pass)
{
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
    {
        AccT prev = (i == 0) ? 0 : histogram[i - 1];
        AccT cur = histogram[i];

        // one and only one thread will satisfy this condition, so counter is
        // written by only one thread
        if ((prev < sum && cur >= sum) || (sum <= 0 && i == 0))
        {
            counter->sum = sum - prev;        // how many values still are there to find
            counter->len = countHistogram[i]; // cur - prev; // number of values in next pass
            typename cub::Traits<T>::UnsignedBits bucket = i;
            int startBit = calcsStartBit<T, BitsPerPass>(pass);
            counter->kthValueBits |= bucket << startBit;
        }
    }
}

/**
 * Computes sequenceLength, finished state, outputLogProbs, and cumLogProbs.
 */
template <typename T, typename IdxT>
__device__ void epilogue(T const value, IdxT const index, float* outputLogProbs, float* cumLogProbs, IdxT const* endIds,
    IdxT* sequenceLengths, FinishedState* finishedOutput, int const batchId)
{
    if (outputLogProbs != nullptr || cumLogProbs != nullptr)
    {
        float res = logf(value);
        if (outputLogProbs)
        {
            outputLogProbs[batchId] = res;
        }
        if (cumLogProbs)
        {
            cumLogProbs[batchId] += res;
        }
    }
    if (index == endIds[batchId])
    {
        if (finishedOutput != nullptr)
        {
            finishedOutput[batchId].setFinishedEOS();
        }
        // Do not increase seq len when EOS is generated. Seq len should always contain only tokens to be outputted
    }
    else
    {
        // We don't need to set output finished state as it is assumed to be in non finished state
        sequenceLengths[batchId] += 1;
    }
}

/**
 *  Find the target element.
 *  (steps 4 in `airTopPSsampling` description)
 */
template <typename T, typename IdxT, typename AccT, int BitsPerPass>
__device__ void lastFilter(T const* inBuf, IdxT const* inIdxBuf, IdxT currentLen, Counter<T, IdxT, AccT>* counter,
    float* outputLogProbs, float* cumLogProbs, IdxT** ids, IdxT const* endIds, IdxT* sequenceLengths,
    FinishedState* finishedOutput, int const batchId)
{
    auto const kthValueBits = counter->kthValueBits;
    auto const equalValue = twiddleOut<T>(kthValueBits, false);
    int const currentStep = sequenceLengths[batchId];
    IdxT* outIdx = &ids[batchId][currentStep];
    if (threadIdx.x == 0)
    {
        *outIdx = cuda::std::numeric_limits<IdxT>::max();
    }
    __syncthreads();

    for (IdxT i = threadIdx.x; i < currentLen; i += blockDim.x)
    {
        if (inBuf[i] == equalValue)
        {
            atomicMin(outIdx, inIdxBuf ? inIdxBuf[i] : i);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        epilogue(equalValue, *outIdx, outputLogProbs, cumLogProbs, endIds, sequenceLengths, finishedOutput, batchId);
    }
}

/******************************Kernel**********************************/
/**
 * We call this parallel top-p algorithm AIR Top-P, because this method is based on our previous work called AIR Top-K.
 * Details about AIR Top-K can be found here https://dl.acm.org/doi/10.1145/3581784.360706, the open-source code is here
 * https://github.com/rapidsai/raft/blob/main/cpp/include/raft/matrix/detail/select_radix.cuh
 *
 * It is expected to call this kernel multiple times (passes), in each pass we process a radix,
 * going from the most significant towards the least significant bits (MSD).
 *
 * Conceptually, each pass consists of 4 steps:
 *
 * 1. Calculate histogram
 *      First, transform bits into a digit, the value of which is in the range
 *      [0, 2^{BITS_PER_PASS}-1]. Then count the frequency of each digit value along with the summation of corresponding
 * elements and the result is a countHistogram and histogram. That is, countHistogram[i] contains the count of inputs
 * having value i.
 *
 * 2. Scan the histogram
 *      Inclusive prefix sum is computed for the histogram. After this step, histogram[i] contains
 *      the prefix-sum of inputs having value <= i.
 *
 * 3. Find the bucket j of the histogram that just exceed the p*total_sum value falls into
 *
 * 4. Filtering
 *      Input elements whose digit value <j are the top-p elements. Since the k-th value must be in
 *      the bucket j, we write all elements in bucket j into a intermediate buffer out_buf. For the
 *      next pass, these elements are used as input, and we update the counter->sum accordingly. T
 *
 * In the implementation, the filtering step is delayed to the next pass so the filtering and
 * histogram computation are fused. In this way, inputs are read once rather than twice.
 *
 * During the filtering step, we won't write candidates (elements in bucket j) to `out_buf` if the
 * number of candidates is larger than the length of `out_buf` (this could happen when the leading
 * bits of input values are almost the same). And then in the next pass, inputs are read from `in`
 * rather than from `in_buf`. The benefit is that we can save the cost of writing candidates and
 * their indices.
 */
template <typename T, typename IdxT, typename AccT, int BitsPerPass, int BlockSize, bool is_fused_filter = false>
__global__ void airTopPSsampling(Counter<T, IdxT, AccT>* counters, AccT* histograms, IdxT* countHistograms, IdxT** ids,
    int* sequenceLengths, FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, IdxT const* endIds, int const batchSize, bool const* skipDecode, int const pass, T* buf1,
    IdxT* idxBuf1, T* buf2, IdxT* idxBuf2)
{
    assert(sequenceLengths != nullptr);
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");

    int const tid = threadIdx.x;
    int const batchId = blockIdx.y;
    auto counter = counters + batchId;

    // Skip kernel if this sampling method is not chosen
    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchId] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchId]) || (finishState.isSkipDecoding()))
    {
        return;
    }

    // Exit early if sequence has finished
    if (finishState.isFinished())
    {
        if (pass == 0 && tid == 0)
        {
            if (finishedOutput != nullptr)
            {
                finishedOutput[batchId] = finishState;
            }
            ids[batchId][sequenceLengths[batchId]] = endIds[batchId];
        }
        return;
    }

    /// Set length
    AccT currentSum;
    IdxT previousLen;
    IdxT currentLen;

    if (pass == 0)
    {
        currentSum = 0;
        previousLen = counter->len;
        // Need to do this so setting counter->previousLen for the next pass is correct.
        // This value is meaningless for pass 0, but it's fine because pass 0 won't be the
        // last pass in this implementation so pass 0 won't hit the "if (pass ==
        // numPasses - 1)" branch.
        currentLen = counter->len;
    }
    else
    {
        currentSum = counter->sum;
        currentLen = counter->len;
        previousLen = counter->previousLen;
    }
    if (currentLen == 0)
    {
        return;
    }
    bool const earlyStop = (currentLen == 1);
    IdxT const bufLen = calcBufLen<T>(counter->oriLen);

    /// Set address
    T const* inBuf = nullptr;
    IdxT const* inIdxBuf = nullptr;
    T* outBuf = nullptr;
    IdxT* outIdxBuf = nullptr;

    setBufPointers(counter->in, counter->inIdx, buf1 + bufLen * batchId, idxBuf1 + bufLen * batchId,
        buf2 + bufLen * batchId, idxBuf2 + bufLen * batchId, pass, inBuf, inIdxBuf, outBuf, outIdxBuf);

    // "previousLen > bufLen" means previous pass skips writing buffer
    if (pass == 0 || pass == 1 || previousLen > bufLen)
    {
        inBuf = counter->in;
        inIdxBuf = counter->inIdx;
        previousLen = counter->oriLen;
    }
    // "currentLen > bufLen" means current pass will skip writing buffer
    if (pass == 0 || currentLen > bufLen)
    {
        outBuf = nullptr;
        outIdxBuf = nullptr;
    }
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    auto histogram = histograms + batchId * numBuckets;
    auto countHistogram = countHistograms + batchId * numBuckets;

    filterAndHistogram<T, IdxT, AccT, BitsPerPass>(inBuf, inIdxBuf, outBuf, outIdxBuf, previousLen, counter, histogram,
        countHistogram, pass, outputLogProbs, cumLogProbs, ids, endIds, sequenceLengths, finishedOutput, batchId,
        earlyStop);

    __syncthreads();

    bool isLastBlock = false;
    if (threadIdx.x == 0)
    {
        unsigned int finished = atomicInc(&counter->finishedBlockCnt, gridDim.x - 1);
        isLastBlock = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(isLastBlock))
    {
        if (earlyStop)
        {
            return;
        }
        scan<AccT, BitsPerPass, BlockSize>(histogram);
        __syncthreads();
        if (pass == 0)
        {
            currentSum = histogram[numBuckets - 1] * counter->p;
        }
        __syncthreads();

        chooseBucket<T, IdxT, AccT, BitsPerPass>(counter, histogram, countHistogram, currentSum, pass);
        __syncthreads();

        int constexpr numPasses = calcNumPasses<T, BitsPerPass>();
        // reset for next pass
        if (pass != numPasses - 1)
        {
            for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
            {
                histogram[i] = 0;
                countHistogram[i] = 0;
            }
        }
        if (threadIdx.x == 0)
        {
            counter->previousLen = currentLen;
            // not necessary for the last pass, but put it here anyway
            counter->filterCnt = 0;
        }

        if (pass == numPasses - 1)
        {
            if constexpr (is_fused_filter)
            {
                lastFilter<T, IdxT, AccT, BitsPerPass>(outBuf ? outBuf : inBuf, outIdxBuf ? outIdxBuf : inIdxBuf,
                    outBuf ? currentLen : counter->oriLen, counter, outputLogProbs, cumLogProbs, ids, endIds,
                    sequenceLengths, finishedOutput, batchId);

                __syncthreads();
            }
        }
    }
}

/**
 * Initialize the Counter<T, IdxT, AccT> and the histogram and countHistogram.
 */
template <typename T, typename IdxT, typename AccT, int BitsPerPass, int BlockSize>
__global__ void airTopPInitialize(Counter<T, IdxT, AccT>* counters, int const batchSize, int const len, T const* in,
    IdxT const* inIdx, float const topP, float const* topPs, curandState_t* curandstate, AccT* histograms,
    IdxT* countHistograms)
{
    auto const batchIdx = blockIdx.x;
    Counter<T, IdxT, AccT>* counter = counters + batchIdx;
    IdxT offset = batchIdx * len;
    IdxT bufOffset = batchIdx * calcBufLen<T>(len);
    if (threadIdx.x == 0)
    {
        counter->in = in + offset;
        counter->inIdx = nullptr;
        if (inIdx)
        {
            counter->inIdx = inIdx + offset;
        }

        counter->len = len;
        counter->oriLen = len;
        counter->previousLen = len;

        float const probThreshold = (topPs != nullptr) ? topPs[batchIdx] : topP;
        float const randP = curand_uniform(curandstate + batchIdx) * probThreshold;
        counter->p = randP;
        counter->sum = 0;

        counter->kthValueBits = 0;
        counter->finishedBlockCnt = 0;
        counter->filterCnt = 0;
    }

    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    AccT* histogram = histograms + batchIdx * numBuckets;
    for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
    {
        histogram[i] = 0;
    }

    IdxT* countHistogram = nullptr;
    if (countHistograms)
    {
        countHistogram = countHistograms + batchIdx * numBuckets;
        for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
        {
            countHistogram[i] = 0;
        }
    }
}

/*
 *  Calculate the number of blocks based on the batchSize and len to avoid tailing effect.
 */
template <typename T, typename IdxT, typename AccT, int BitsPerPass, int BlockSize>
unsigned calcAirTopPBlockNum(int batchSize, IdxT len, int smCnt)
{
    int constexpr VECTORIZED_READ_SIZE = 16;
    static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);

    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks, airTopPSsampling<T, IdxT, AccT, BitsPerPass, BlockSize, false>, BlockSize, 0);
    activeBlocks *= smCnt;

    IdxT bestNumBlocks = 0;
    float bestTailWavePenalty = 1.0f;
    IdxT const maxNumBlocks = ceilDiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize);
    for (int numWaves = 1;; ++numWaves)
    {
        IdxT numBlocks = std::min(maxNumBlocks, static_cast<IdxT>(std::max(numWaves * activeBlocks / batchSize, 1)));
        IdxT itemsPerThread = ceilDiv<IdxT>(len, numBlocks * BlockSize);
        itemsPerThread = alignTo<IdxT>(itemsPerThread, VECTORIZED_READ_SIZE / sizeof(T));
        numBlocks = ceilDiv<IdxT>(len, itemsPerThread * BlockSize);
        float actualNumWaves = static_cast<float>(numBlocks) * batchSize / activeBlocks;
        float tailWavePenalty = (ceilf(actualNumWaves) - actualNumWaves) / ceilf(actualNumWaves);

        // 0.15 is determined experimentally. It also ensures breaking the loop
        // early, e.g. when numWaves > 7, tailWavePenalty will always <0.15
        if (tailWavePenalty < 0.15)
        {
            bestNumBlocks = numBlocks;
            break;
        }
        else if (tailWavePenalty < bestTailWavePenalty)
        {
            bestNumBlocks = numBlocks;
            bestTailWavePenalty = tailWavePenalty;
        }

        if (numBlocks == maxNumBlocks)
        {
            break;
        }
    }
    return bestNumBlocks;
}

template <typename T>
void invokeBatchAirTopPSampling(void* workspace, size_t& workspaceSize, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    T const* logProbs, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded, int const* endIds,
    float const maxTopP, float const* topPs, cudaStream_t stream, int blockNum, bool const* skipDecode)
{
    using IdxT = int;
    using AccT = float;
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");

    IdxT const vocabSize = vocabSizePadded;
    int constexpr BitsPerPass = 11;
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    IdxT const bufLen = calcBufLen<T>(vocabSize);

    int constexpr SAMPLING_BLOCK_SIZE = 512;
    int constexpr THREADS_PER_CTA_TOP_P_INIT = 1024;

    Counter<T, IdxT, AccT>* counters = nullptr;
    AccT* histograms = nullptr;
    IdxT* countHistograms = nullptr;
    T* buf1 = nullptr;
    IdxT* idxBuf1 = nullptr;
    T* buf2 = nullptr;
    IdxT* idxBuf2 = nullptr;
    std::vector<size_t> sizes = {sizeof(*counters) * batchSize, sizeof(*histograms) * numBuckets * batchSize,
        sizeof(*countHistograms) * numBuckets * batchSize, sizeof(*buf1) * bufLen * batchSize,
        sizeof(*idxBuf1) * bufLen * batchSize, sizeof(*buf2) * bufLen * batchSize,
        sizeof(*idxBuf2) * bufLen * batchSize};
    size_t totalSize = calcAlignedSize(sizes);
    if (workspace == nullptr)
    {
        workspaceSize = totalSize;
        return;
    }
    std::vector<void*> alignedPointers;
    calcAlignedPointers(alignedPointers, workspace, sizes);
    counters = static_cast<decltype(counters)>(alignedPointers[0]);
    histograms = static_cast<decltype(histograms)>(alignedPointers[1]);
    countHistograms = static_cast<decltype(countHistograms)>(alignedPointers[2]);
    buf1 = static_cast<decltype(buf1)>(alignedPointers[3]);
    idxBuf1 = static_cast<decltype(idxBuf1)>(alignedPointers[4]);
    buf2 = static_cast<decltype(buf2)>(alignedPointers[5]);
    idxBuf2 = static_cast<decltype(idxBuf2)>(alignedPointers[6]);

    airTopPInitialize<T, IdxT, AccT, BitsPerPass, THREADS_PER_CTA_TOP_P_INIT>
        <<<batchSize, THREADS_PER_CTA_TOP_P_INIT, 0, stream>>>(counters, batchSize, vocabSize, logProbs, nullptr,
            maxTopP, topPs, curandstate, histograms, countHistograms);
    sync_check_cuda_error();

    dim3 grid(blockNum, batchSize);
    // Sample with Top P given sorted tokens
    int constexpr numPasses = calcNumPasses<T, BitsPerPass>();
    auto kernel = airTopPSsampling<T, IdxT, AccT, BitsPerPass, SAMPLING_BLOCK_SIZE, false>;

    for (int pass = 0; pass < numPasses; ++pass)
    {
        if (pass == numPasses - 1)
        {
            kernel = airTopPSsampling<T, IdxT, AccT, BitsPerPass, SAMPLING_BLOCK_SIZE, true>;
        }

        kernel<<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(counters, histograms, countHistograms, outputIds,
            sequenceLength, finishedInput, finishedOutput, cumLogProbs, outputLogProbs, endIds, batchSize, skipDecode,
            pass, buf1, idxBuf1, buf2, idxBuf2);
        sync_check_cuda_error();
    }
}

template void invokeBatchAirTopPSampling(void* workspace, size_t& workspaceSize, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    float const* logProbs, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded,
    int const* endIds, float const maxTopP, float const* topPs, cudaStream_t stream, int blockNum,
    bool const* skipDecode);

template void invokeBatchAirTopPSampling(void* workspace, size_t& workspaceSize, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    half const* logProbs, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded,
    int const* endIds, float const maxTopP, float const* topPs, cudaStream_t stream, int blockNum,
    bool const* skipDecode);

template <typename T>
void invokeAirTopPSampling(void* workspace, size_t& workspaceSize, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    T const* logProbs, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded, int const* endIds,
    float const topP, cudaStream_t stream, int blockNum, bool const* skipDecode)
{
    invokeBatchAirTopPSampling(workspace, workspaceSize, outputIds, sequenceLength, finishedInput, finishedOutput,
        cumLogProbs, outputLogProbs, logProbs, curandstate, batchSize, vocabSizePadded, endIds, topP, nullptr, stream,
        blockNum, skipDecode);
}

template void invokeAirTopPSampling(void* workspace, size_t& workspaceSize, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    float const* logProbs, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded,
    int const* endIds, float const topP, cudaStream_t stream, int blockNum, bool const* skipDecode);

template void invokeAirTopPSampling(void* workspace, size_t& workspaceSize, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    half const* logProbs, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded,
    int const* endIds, float const topP, cudaStream_t stream, int blockNum, bool const* skipDecode);

template unsigned calcAirTopPBlockNum<float, int, float>(int batchSize, int len, int smCnt);
template unsigned calcAirTopPBlockNum<half, int, float>(int batchSize, int len, int smCnt);

} // namespace kernels
} // namespace tensorrt_llm
