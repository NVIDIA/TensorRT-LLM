/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda/atomic>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda/std/limits>
#include <cuda_fp16.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

using IdxT = int;
using AccT = float;

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
    // used to determine if the current block is the last running block.
    alignas(128) uint32_t finishedBlockCnt;
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

//! \brief Calculate the number of buckets based on the number of bits per pass.
//! \tparam BitsPerPass. If BitsPerPass==11, the number of buckets is 2048. If BitsPerPass==8, the number of buckets is
//! 256.
template <int BitsPerPass>
__host__ __device__ int constexpr calcNumBuckets()
{
    return 1 << BitsPerPass;
}

//! \brief Calculate the number of passes based on the number of bits per pass.
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
__device__ int constexpr calcStartBit(int pass)
{
    int startBit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    if (startBit < 0)
    {
        startBit = 0;
    }
    return startBit;
}

template <typename T, int BitsPerPass>
__device__ uint32_t constexpr calcMask(int pass)
{
    static_assert(BitsPerPass <= 31);
    int numBits = calcStartBit<T, BitsPerPass>(pass - 1) - calcStartBit<T, BitsPerPass>(pass);
    return (1 << numBits) - 1;
}

template <typename T>
__device__ constexpr uint32_t getNumTotalMantissa()
{
    if constexpr (std::is_same_v<T, half>)
    {
        return 10;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return 23;
    }
}

template <typename T>
__device__ uint32_t calcMantissa(T value);

template <>
__device__ uint32_t calcMantissa(float value)
{
    union
    {
        uint32_t bits;
        float value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<float>();
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return input.bits & mask;
}

__device__ uint32_t calcMantissa(half value)
{
    union
    {
        uint16_t bits;
        half value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<half>();
    uint32_t t = 0u | input.bits;
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return t & mask;
}

template <typename T>
__device__ uint32_t calcExponent(T value);

template <>
__device__ uint32_t calcExponent(float value)
{
    union
    {
        uint32_t bits;
        float value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<float>();
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return input.bits & ~mask;
}

template <>
__device__ uint32_t calcExponent(half value)
{
    union
    {
        uint16_t bits;
        half value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<half>();
    uint32_t t = 0u | input.bits;
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return t & ~mask;
}

__device__ float calcHalfValue(uint32_t count, uint32_t exponent, uint32_t sign, uint64_t bitSum)
{
    constexpr uint32_t numTotalBits = 64; // The bit number of uint64_t
    constexpr uint32_t numOffset = 16;    // The bits number difference between float and half data type
    constexpr uint32_t numTotalMantissaHalf
        = getNumTotalMantissa<half>();    // The bit number of mantissa for half data type
    constexpr uint32_t numTotalMantissaFloat
        = getNumTotalMantissa<float>();   // The bit number of mantissa for float data type

    uint64_t extraInMatissa = (bitSum >> numTotalMantissaHalf);

    // Count the bit number for exceeding mantissa and the extra unwritten 1s
    uint32_t numExtra = 0;
    uint32_t numDeNorm = 0;
    int numNorm = 0;
    uint32_t mask = 0;
    extraInMatissa = (exponent == 0) ? extraInMatissa : extraInMatissa + count;
    numExtra = numTotalBits - __clzll(extraInMatissa);
    numNorm = (exponent == 0) ? 0 : -1;
    if (extraInMatissa == 0)
    {
        numDeNorm = numTotalMantissaHalf - (numTotalBits - __clzll(bitSum));
    }
    exponent = exponent + ((numExtra + numNorm + 127 - 15 - numDeNorm) << numTotalMantissaHalf);
    // As extra bits (extraInMatissa) need to be part of the mantissa, we have to move the current
    // mantissa within the range of [0-23]bits.
    // This is the only step cause precision loss
    uint32_t mantissa;
    if (extraInMatissa != 0)
    {
        int numMove = numTotalMantissaFloat - (numExtra - 1);
        mask = (1u << (numExtra - 1)) - 1;
        // As the first bit of extraInMatissa is the unwritten 1,
        // we need to mask that to zero
        extraInMatissa = extraInMatissa & mask;
        if (numMove > 0)
        {
            extraInMatissa = extraInMatissa << numMove;
            mask = (1u << numTotalMantissaHalf) - 1;
            mantissa = (((bitSum & mask) << (numTotalMantissaFloat - numTotalMantissaHalf)) >> (numExtra - 1))
                | extraInMatissa;
        }
        else
        {
            mantissa = extraInMatissa >> (-1 * numMove);
        }
    }
    else
    {
        mask = (1u << numTotalMantissaHalf) - 1;
        mantissa = bitSum << (numDeNorm + 1);
        mantissa = mantissa & mask;
        mantissa = mantissa << (numTotalMantissaFloat - numTotalMantissaHalf);
    }

    uint32_t bitFloat = (sign << numOffset) | (exponent << (numTotalMantissaFloat - numTotalMantissaHalf)) | mantissa;
    return reinterpret_cast<float&>(bitFloat);
}

__device__ float calcFloatValue(uint32_t count, uint32_t exponent, uint64_t bitSum)
{
    constexpr uint32_t numTotalBits = 64;
    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<float>();
    uint64_t extraInMatissa = (bitSum >> numTotalMantissa);
    // Count the bit number for exceeding mantissa and the extra unwritten 1s
    uint32_t numExtra;
    int numNorm = 0;
    uint32_t mask = 0;
    extraInMatissa = (exponent == 0) ? extraInMatissa : extraInMatissa + count;
    numExtra = numTotalBits - __clzll(extraInMatissa);
    numNorm = (exponent == 0) ? 0 : -1;
    exponent = exponent + ((numExtra + numNorm) << numTotalMantissa);
    // As extra integers need to be part of the mantissa, we have to move the current
    // mantissa within the range of [0-23]bits.
    // This is the only step cause precision loss
    uint32_t mantissa;
    if (extraInMatissa != 0)
    {
        int numMove = numTotalMantissa - (numExtra - 1);
        // As the first bit of extraInMatissa is the unwritten 1,
        // we need to mask that to zero
        mask = (1u << (numExtra - 1)) - 1;
        extraInMatissa = extraInMatissa & mask;
        if (numMove > 0)
        {
            extraInMatissa = extraInMatissa << numMove;
            mask = (1u << numTotalMantissa) - 1;
            mantissa = ((bitSum & mask) >> (numExtra - 1)) | extraInMatissa;
        }
        else
        {
            mantissa = extraInMatissa >> (-1 * numMove);
        }
    }
    else
    {
        mantissa = bitSum;
    }
    uint32_t bitFloat = exponent | mantissa;
    return reinterpret_cast<float&>(bitFloat);
}

template <typename T, typename HisT, bool isDeterministic = false>
__device__ constexpr void calcAtomicAdd(HisT* dst, T value)
{
    if constexpr (isDeterministic)
    {
        uint32_t mantissa = calcMantissa(value);
        if constexpr (std::is_same_v<T, half>)
        {
            atomicAdd(dst, mantissa);
        }
        else
        {
            // Have to use reinterpret_cast() to convert uint64_t to "unsigned long long"
            // Otherwise, the complication will report the follow error:
            //"error: no instance of overloaded function "atomicAdd" matches the argument list
            // argument types are: (uint64_t *, uint64_t)"
            atomicAdd(reinterpret_cast<unsigned long long*>(dst), static_cast<HisT>(mantissa));
        }
    }
    else
    {
        if constexpr (std::is_same_v<T, half>)
        {
            atomicAdd(dst, __half2float(value));
        }
        else
        {
            atomicAdd(dst, value);
        }
    }
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
__device__ int calcBucket(T x, int startBit, uint32_t mask, bool selectMin)
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
 * Fused filtering of the current pass and building histogram for the next pass (see steps 4 & 1 in `airTopPSampling`
 * description).
 */
template <typename T, typename IdxT, typename AccT, typename HisT, int BitsPerPass, bool isDeterministic = false>
__device__ __forceinline__ void filterAndHistogram(T const* inBuf, IdxT const* inIdxBuf, T* outBuf, IdxT* outIdxBuf,
    int previousLen, Counter<T, IdxT, AccT>* counter, HisT* histogram, IdxT* countHistogram, HisT* histogramSmem,
    IdxT* countHistogramSmem, int pass, float* outputLogProbs, float* cumLogProbs, IdxT** ids, IdxT const* endIds,
    IdxT* sequenceLengths, FinishedState* finishedOutput, int const batchId, int maxBatchSize, bool earlyStop)
{
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");

    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    bool constexpr selectMin = false;

    for (IdxT i = threadIdx.x; i < numBuckets; i += blockDim.x)
    {
        histogramSmem[i] = 0;
        countHistogramSmem[i] = 0;
    }
    __syncthreads();

    int const startBit = calcStartBit<T, BitsPerPass>(pass);
    uint32_t const mask = calcMask<T, BitsPerPass>(pass);

    if (pass == 0)
    {
        // Passed to vectorizedProcess, this function executes in all blocks in
        // parallel, i.e. the work is split along the input (both, in batches and
        // chunks of a single row). Later, the histograms are merged using
        // atomicAdd.
        auto f = [selectMin, startBit, mask, histogramSmem, countHistogramSmem](T value, IdxT)
        {
            int bucket = calcBucket<T, BitsPerPass>(value, startBit, mask, selectMin);
            calcAtomicAdd<T, HisT, isDeterministic>(histogramSmem + bucket, value);
            atomicAdd(countHistogramSmem + bucket, static_cast<IdxT>(1));
        };
        vectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(blockDim.x) * gridDim.x, inBuf, previousLen, f);
    }
    else
    {
        IdxT* pFilterCnt = &counter->filterCnt;
        auto const kthValueBits = counter->kthValueBits;
        int const previousStartBit = calcStartBit<T, BitsPerPass>(pass - 1);

        // See the remark above on the distributed execution of `f` using
        // vectorizedProcess.
        auto f = [inIdxBuf, outBuf, outIdxBuf, selectMin, startBit, mask, previousStartBit, kthValueBits, pFilterCnt,
                     histogramSmem, countHistogramSmem, outputLogProbs, cumLogProbs, ids, endIds, sequenceLengths,
                     finishedOutput, batchId, maxBatchSize, earlyStop](T value, IdxT i)
        {
            auto const previousBits = (twiddleIn(value, selectMin) >> previousStartBit) << previousStartBit;
            if (previousBits == kthValueBits)
            {
                if (earlyStop)
                {

                    int const currentStep = sequenceLengths ? sequenceLengths[batchId] : 0;
                    IdxT index = inIdxBuf ? inIdxBuf[i] : i;
                    ids[batchId][currentStep] = index;
                    float valueFloat;
                    if constexpr (std::is_same_v<T, half>)
                    {
                        valueFloat = __half2float(value);
                    }
                    else
                    {
                        valueFloat = value;
                    }
                    epilogue(valueFloat, index, outputLogProbs, cumLogProbs, endIds, sequenceLengths, finishedOutput,
                        batchId, maxBatchSize);
                }
                if (outBuf)
                {
                    IdxT pos = atomicAdd(pFilterCnt, static_cast<IdxT>(1));
                    outBuf[pos] = value;
                    outIdxBuf[pos] = inIdxBuf ? inIdxBuf[i] : i;
                }

                int bucket = calcBucket<T, BitsPerPass>(value, startBit, mask, selectMin);
                calcAtomicAdd<T, HisT, isDeterministic>(histogramSmem + bucket, value);
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
            if constexpr ((isDeterministic) && (std::is_same_v<T, float>) )
            {
                // Have to use reinterpret_cast() to convert uint64_t to "unsigned long long"
                // Otherwise, the complication will report the follow error:
                //"error: no instance of overloaded function "atomicAdd" matches the argument list
                // argument types are: (uint64_t *, uint64_t)"
                atomicAdd(reinterpret_cast<unsigned long long*>(histogram + i), histogramSmem[i]);
            }
            else
            {
                atomicAdd(histogram + i, histogramSmem[i]);
            }
        }
        if (countHistogramSmem[i] != 0)
        {
            atomicAdd(countHistogram + i, countHistogramSmem[i]);
        }
    }
}

/**
 *  Replace histogram with its own prefix sum (step 2 in `airTopPSampling` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(IdxT volatile* histogram, IdxT* histogramOut)
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

        BlockStore(tempStorage.store).Store(histogramOut, threadData);
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
            histogramOut[threadIdx.x] = threadData;
        }
    }
}

/**
 * Computes sequenceLength, finished state, outputLogProbs, and cumLogProbs.
 */
template <typename T, typename IdxT>
__device__ void epilogue(T const value, IdxT const index, float* outputLogProbs, float* cumLogProbs, IdxT const* endIds,
    IdxT* sequenceLengths, FinishedState* finishedOutput, int const batchId, int maxBatchSize)
{
    if (outputLogProbs != nullptr || cumLogProbs != nullptr)
    {
        float res = logf(value);
        if (outputLogProbs)
        {
            auto const curLen = sequenceLengths ? sequenceLengths[batchId] : 0;
            outputLogProbs[curLen * maxBatchSize + batchId] = res;
        }
        if (cumLogProbs)
        {
            cumLogProbs[batchId] += res;
        }
    }
    if (endIds && index == endIds[batchId])
    {
        if (finishedOutput != nullptr)
        {
            finishedOutput[batchId].setFinishedEOS();
        }
        // Do not increase seq len when EOS is generated. Seq len should always contain only tokens to be outputted
    }
    else if (sequenceLengths != nullptr)
    {
        // We don't need to set output finished state as it is assumed to be in non finished state
        sequenceLengths[batchId] += 1;
    }
}

/**
 *  Find the target element.
 *  (steps 4 in `airTopPSampling` description)
 */
template <typename T, typename IdxT, typename AccT, int BitsPerPass, int BlockSize, bool isDeterministic = false>
__device__ void lastFilter(T const* inBuf, IdxT const* inIdxBuf, IdxT currentLen, Counter<T, IdxT, AccT>* counter,
    float* outputLogProbs, float* cumLogProbs, IdxT** ids, IdxT const* endIds, IdxT* sequenceLengths,
    FinishedState* finishedOutput, int const batchId, int maxBatchSize, IdxT* lastIdxBuf, IdxT* countHistogram)
{
    auto const kthValueBits = counter->kthValueBits;
    auto const equalValue = twiddleOut<T>(kthValueBits, false);
    int const currentStep = sequenceLengths ? sequenceLengths[batchId] : 0;
    IdxT* outIdx = &ids[batchId][currentStep];

    float equalValueFloat;
    if constexpr (std::is_same_v<T, half>)
    {
        equalValueFloat = __half2float(equalValue);
    }
    else
    {
        equalValueFloat = equalValue;
    }
    if constexpr (!isDeterministic)
    {

        for (IdxT i = threadIdx.x; i < currentLen; i += blockDim.x)
        {
            if (inBuf[i] == equalValue)
            {
                *outIdx = inIdxBuf ? inIdxBuf[i] : i;
                break;
            }
        }
    }
    else
    {
        IdxT const bufLen = calcBufLen<T>(counter->oriLen);
        IdxT neededNumOfKth = counter->sum > 0 ? ceil(counter->sum / equalValueFloat) : 1;

        if (counter->len < neededNumOfKth)
        {
            neededNumOfKth = counter->len;
        }

        if (neededNumOfKth < bufLen)
        {
            for (int i = threadIdx.x; i < neededNumOfKth; i += blockDim.x)
            {
                lastIdxBuf[i] = cuda::std::numeric_limits<IdxT>::max();
            }
            __threadfence_block();
            __syncthreads();

            cuda::atomic_ref<IdxT, cuda::thread_scope_block> refLast(lastIdxBuf[neededNumOfKth - 1]);

            for (IdxT i = threadIdx.x; i < currentLen; i += blockDim.x)
            {
                if (inBuf[i] == equalValue)
                {
                    IdxT newIdx = inIdxBuf ? inIdxBuf[i] : i;
                    if (newIdx < refLast.load(cuda::memory_order_relaxed))
                    {
                        for (int j = 0; j < neededNumOfKth; j++)
                        {
                            IdxT preIdx = atomicMin_block(&lastIdxBuf[j], newIdx);
                            if (preIdx > newIdx)
                            {
                                newIdx = preIdx;
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                *outIdx = refLast.load(cuda::memory_order_relaxed);
            }
        }
        else
        {
            int numPass = calcNumPasses<IdxT, BitsPerPass>();
            int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
            __shared__ typename cub::Traits<IdxT>::UnsignedBits kthValueBitsIdx;
            __shared__ IdxT neededNumOfKthSmem;
            if (threadIdx.x == 0)
            {
                kthValueBitsIdx = 0;
                neededNumOfKthSmem = neededNumOfKth;
            }
            __syncthreads();
            for (int pass = 0; pass < numPass; pass++)
            {
                for (IdxT i = threadIdx.x; i < numBuckets; i += blockDim.x)
                {
                    countHistogram[i] = 0;
                }
                __syncthreads();

                int preNeededNumOfKth = neededNumOfKthSmem;
                int const startBit = calcStartBit<IdxT, BitsPerPass>(pass);
                uint32_t const mask = calcMask<IdxT, BitsPerPass>(pass);
                for (IdxT j = threadIdx.x; j < currentLen; j += blockDim.x)
                {
                    if (inBuf[j] == equalValue)
                    {
                        IdxT newIdx = inIdxBuf ? inIdxBuf[j] : j;
                        bool isQualified = (pass == 0) ? true : false;
                        if (pass > 0)
                        {
                            int const previousStartBit = calcStartBit<IdxT, BitsPerPass>(pass - 1);
                            auto const previousBits = (twiddleIn(newIdx, true) >> previousStartBit) << previousStartBit;
                            if (previousBits == kthValueBitsIdx)
                            {
                                isQualified = true;
                            }
                        }
                        if (isQualified)
                        {
                            int bucket = calcBucket<IdxT, BitsPerPass>(newIdx, startBit, mask, true);
                            atomicAdd(countHistogram + bucket, static_cast<IdxT>(1));
                        }
                    }
                } // end histogram
                __syncthreads();

                scan<IdxT, BitsPerPass, BlockSize>(countHistogram, countHistogram); // prefix sum
                __syncthreads();
                // Locate the bucket
                for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
                {
                    IdxT prev = (i == 0) ? 0 : countHistogram[i - 1];
                    IdxT cur = countHistogram[i];
                    // one and only one thread will satisfy this condition, so counter is
                    // written by only one thread
                    if (prev < preNeededNumOfKth && preNeededNumOfKth <= cur)
                    {
                        neededNumOfKthSmem = neededNumOfKthSmem - prev;
                        typename cub::Traits<IdxT>::UnsignedBits bucket = i;
                        kthValueBitsIdx |= bucket << startBit;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                *outIdx = twiddleOut<IdxT>(kthValueBitsIdx, true);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        epilogue(equalValueFloat, *outIdx, outputLogProbs, cumLogProbs, endIds, sequenceLengths, finishedOutput,
            batchId, maxBatchSize);
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
template <typename T, typename IdxT, typename AccT, typename HisT, int BitsPerPass, int BlockSize,
    bool isFusedFilter = false, bool isDeterministic = false>
__global__ void airTopPSampling(Counter<T, IdxT, AccT>* counters, HisT* histograms, IdxT* countHistograms, IdxT** ids,
    int* sequenceLengths, FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, IdxT const* endIds, int const maxBatchSize, bool const* skipDecode, int const pass, T* buf1,
    IdxT* idxBuf1, T* buf2, IdxT* idxBuf2, int32_t const* batchSlots)
{
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");

    int const tid = threadIdx.x;
    int const batchId = blockIdx.y;
    auto const batchSlot = batchSlots ? batchSlots[batchId] : batchId;
    auto counter = counters + batchId;

    // Skip kernel if this sampling method is not chosen
    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchSlot] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchSlot]) || (finishState.isSkipDecoding()))
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
                finishedOutput[batchSlot] = finishState;
            }
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
    __shared__ HisT histogramSmem[numBuckets];
    __shared__ IdxT countHistogramSmem[numBuckets];
    AccT* histValueSmem = reinterpret_cast<AccT*>(histogramSmem);

    filterAndHistogram<T, IdxT, AccT, HisT, BitsPerPass, isDeterministic>(inBuf, inIdxBuf, outBuf, outIdxBuf,
        previousLen, counter, histogram, countHistogram, histogramSmem, countHistogramSmem, pass, outputLogProbs,
        cumLogProbs, ids, endIds, sequenceLengths, finishedOutput, batchSlot, maxBatchSize, earlyStop);

    __syncthreads();
    __threadfence();

    bool isLastBlock = false;
    if (threadIdx.x == 0)
    {
        uint32_t finished = atomicInc(&counter->finishedBlockCnt, gridDim.x - 1);
        isLastBlock = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(isLastBlock))
    {
        if (earlyStop)
        {
            if (threadIdx.x == 0)
            {
                // avoid duplicated epilgue()
                counter->previousLen = 0;
                counter->len = 0;
            }
            return;
        }

        if constexpr (isDeterministic)
        {
            for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
            {
                uint64_t value = (uint64_t) histogram[i];
                IdxT count = countHistogram[i];

                if (count != 0)
                {
                    uint32_t startBit = calcStartBit<T, BitsPerPass>(pass);
                    [[maybe_unused]] float bucketValueFloat;
                    if constexpr (std::is_same_v<T, half>)
                    {
                        // To acquire the summation in single-precision format, we need to get the original exponent
                        // value first counter->kthValueBits stores the bits selected by previous pass, which contains
                        // the bit corresponds to the exponent value
                        uint16_t bucketValue = counter->kthValueBits;

                        // For the first pass, different bucket indices correspond to different exponents.
                        // The bucket index can be used to deduce the exponent.
                        if (pass == 0)
                        {
                            // Right shift the bucket index with startBit bits (5 bits for half-precision when pass==0),
                            // so that the bucket index fills the bit related to exponent.
                            bucketValue = i << startBit;
                        }
                        uint32_t exponent = calcExponent(twiddleOut<T>(bucketValue, false));
                        uint32_t mask = (1u << (sizeof(half) * CHAR_BIT - 1)) - 1;
                        uint32_t sign = exponent & (~mask);
                        exponent = exponent & mask;
                        float tmp = calcHalfValue((uint32_t) count, exponent, sign, value);
                        histValueSmem[i] = tmp;
                    }
                    else
                    {
                        // To acquire the summation in single-precision format, we need to get the original exponent
                        // value first
                        uint32_t bucketValue = counter->kthValueBits;
                        if (pass == 0)
                        {
                            // Right shift the bucket index with startBit bits (22 bits for single-precision when
                            // pass==0), so that the bucket index fills the bit related to exponent.
                            bucketValue = i << startBit;
                        }
                        bucketValueFloat = twiddleOut<T>(bucketValue, false);
                        uint32_t exponent = calcExponent(bucketValueFloat);
                        histValueSmem[i] = calcFloatValue((uint32_t) count, exponent, value);
                    }
                }
                else
                {
                    histValueSmem[i] = 0.0f;
                }
            }
        }

        // To avoid the error related to the prefix sum from cub, we find the bucket sequentially.
        int constexpr WARP_SIZE = 32;
        int constexpr WARP_COUNT = numBuckets / WARP_SIZE;
        namespace cg = cooperative_groups;
        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        AccT* histPtr = isDeterministic ? histValueSmem : reinterpret_cast<AccT*>(histogram);
        __shared__ AccT warpSum[WARP_COUNT];
        __shared__ cuda::atomic<AccT, cuda::thread_scope_block> blockSum;
        if constexpr (BitsPerPass != 11)
        {
            for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
            {
                warpSum[i] = 0;
            }
            __syncthreads();
        }

        // Acquire the summation of each 32 buckets
        for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
        {
            reduce_store_async(warp, warpSum + i / WARP_SIZE, histPtr[i], cg::plus<float>{});
        }
        __syncthreads();

        // Acquire the summation of all the 2048 buckets
        if (threadIdx.x < WARP_SIZE)
        {
            reduce_store_async(warp, blockSum, warpSum[threadIdx.x], cg::plus<float>{});
            if constexpr (BitsPerPass == 11)
            {
                reduce_update_async(warp, blockSum, warpSum[threadIdx.x + WARP_SIZE], cg::plus<float>{});
            }
        }
        __syncthreads();

        // Update currentSum
        if (pass == 0)
        {
            currentSum = blockSum * counter->p;
        }

        if (threadIdx.x == 0)
        {
            AccT prev = 0;

            // Add 32 elements each step
            int iStep = 0;
            int targetStep = 0;
            for (; iStep < WARP_COUNT; iStep++)
            {
                if (warpSum[iStep])
                {
                    targetStep = iStep;
                    if ((prev + warpSum[iStep]) >= currentSum)
                    {
                        break;
                    }
                    prev += warpSum[iStep];
                }
            }

            int targetIdx = 0;
            for (int i = targetStep * WARP_SIZE; i < numBuckets; i++)
            {
                if (countHistogram[i])
                {
                    targetIdx = i;
                    if ((prev + histPtr[i]) >= currentSum)
                    {
                        break;
                    }
                    prev += histPtr[i];
                }
            }

            counter->sum = currentSum - prev;         // how many values still are there to find
            counter->len = countHistogram[targetIdx]; // cur - prev; // number of values in next pass
            typename cub::Traits<T>::UnsignedBits bucket = targetIdx;
            int startBit = calcStartBit<T, BitsPerPass>(pass);
            counter->kthValueBits |= bucket << startBit;
        }
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
            // Used when isDeterministic==true
            // idxBuf1 and idxBuf2 are ping-pong buffers used in previous iterations to store candidates.
            // In the last pass (pass==2 for single-precision and pass==1 for half-precision),
            // we reuse the buffer didn't store the candidates (idxBuf1 for single-precision and idxBuf2 for
            // half-precision) to help find the correct index of the result.
            [[maybe_unused]] IdxT* lastIdxBuf
                = (pass % 2 == 0) ? idxBuf1 + bufLen * batchId : idxBuf2 + bufLen * batchId;
            if constexpr (isFusedFilter)
            {
                lastFilter<T, IdxT, AccT, BitsPerPass, BlockSize, isDeterministic>(outBuf ? outBuf : inBuf,
                    outIdxBuf ? outIdxBuf : inIdxBuf, outBuf ? currentLen : counter->oriLen, counter, outputLogProbs,
                    cumLogProbs, ids, endIds, sequenceLengths, finishedOutput, batchSlot, maxBatchSize, lastIdxBuf,
                    countHistogramSmem);
                __syncthreads();
            }
        }
    }
}

/**
 * Initialize the Counter<T, IdxT, AccT> and the histogram and countHistogram.
 */
template <typename T, typename IdxT, typename AccT, typename HisT, int BitsPerPass, int BlockSize>
__global__ void airTopPInitialize(Counter<T, IdxT, AccT>* counters, int const batchSize, int const len, T const* in,
    IdxT const* inIdx, float const* topPs, curandState_t* curandState, float const* randomVals, HisT* histograms,
    IdxT* countHistograms, int32_t const* batchSlots)
{
    auto const batchIdx = blockIdx.x;
    auto const batchSlot = batchSlots ? batchSlots[batchIdx] : batchIdx;
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

        float const probThreshold = topPs[batchSlot];
        auto const randomNumber = randomVals ? randomVals[batchSlot] : curand_uniform(curandState + batchSlot);
        float const randP = randomNumber * probThreshold;
        counter->p = randP;
        counter->sum = 0;

        counter->kthValueBits = 0;
        counter->finishedBlockCnt = 0;
        counter->filterCnt = 0;
    }

    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    HisT* histogram = histograms + batchIdx * numBuckets;
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
template <typename T>
uint32_t calcAirTopPBlockNum(int batchSize, int len, int smCnt, bool isDeterministic)
{
    int constexpr BitsPerPass = 11;
    int constexpr BlockSize = 512;
    int constexpr VECTORIZED_READ_SIZE = 16;
    static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);
    TLLM_CHECK_WITH_INFO(
        smCnt > 0, "AIR Top-P needs the count of multiprocessor to calculate the proper block dimension settings");

    int activeBlocks;
    if (isDeterministic)
    {
        using HisT = std::conditional_t<std::is_same_v<T, float>, uint64_t, uint32_t>;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, airTopPSampling<T, IdxT, AccT, HisT, BitsPerPass, BlockSize, false, true>, BlockSize, 0);
    }
    else
    {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, airTopPSampling<T, IdxT, AccT, float, BitsPerPass, BlockSize, false, false>, BlockSize, 0);
    }
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

template <typename T, bool isDeterministic = false>
[[nodiscard]] std::vector<size_t> getAirTopPWorkspaceSizes(int32_t batchSize, int32_t vocabSize)
{
    using HisT
        = std::conditional_t<isDeterministic, std::conditional_t<std::is_same_v<T, float>, uint64_t, uint32_t>, float>;
    int constexpr BitsPerPass = 11;
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    IdxT const bufLen = calcBufLen<T>(vocabSize);

    size_t countersSize = sizeof(Counter<T, IdxT, AccT>) * batchSize;
    size_t histogramsSize = sizeof(HisT) * numBuckets * batchSize;
    size_t countHistogramsSize = sizeof(IdxT) * numBuckets * batchSize;
    size_t buf1Size = sizeof(T) * bufLen * batchSize;
    size_t idxBuf1Size = sizeof(IdxT) * bufLen * batchSize;
    size_t buf2Size = sizeof(T) * bufLen * batchSize;
    size_t idxBuf2Size = sizeof(IdxT) * bufLen * batchSize;

    std::vector<size_t> sizes
        = {countersSize, histogramsSize, countHistogramsSize, buf1Size, idxBuf1Size, buf2Size, idxBuf2Size};

    return sizes;
}

template std::vector<size_t> getAirTopPWorkspaceSizes<float, true>(int32_t batchSize, int32_t vocabSize);
template std::vector<size_t> getAirTopPWorkspaceSizes<float, false>(int32_t batchSize, int32_t vocabSize);
template std::vector<size_t> getAirTopPWorkspaceSizes<half, true>(int32_t batchSize, int32_t vocabSize);
template std::vector<size_t> getAirTopPWorkspaceSizes<half, false>(int32_t batchSize, int32_t vocabSize);

template <typename T, bool isDeterministic = false>
void invokeAirTopPSamplingWithDeterministicPara(TopPSamplingKernelParams<T> const& params, cudaStream_t stream)
{
    using HisT
        = std::conditional_t<isDeterministic, std::conditional_t<std::is_same_v<T, float>, uint64_t, uint32_t>, float>;

    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");
    TLLM_CHECK_WITH_INFO(((std::is_same_v<T, half>) &&(params.vocabSizePadded < pow(2, 22)) && isDeterministic)
            || ((std::is_same_v<T, float>) &&(params.vocabSizePadded < pow(2, 41)) && isDeterministic)
            || (!isDeterministic),
        "For Deterministic AIR Top-P, the maximum vocab_size we support is pow(2,22) for half-precision and pow(2,41) "
        "for single-precision");

    IdxT const vocabSize = params.vocabSizePadded;
    int constexpr BitsPerPass = 11;

    int constexpr SAMPLING_BLOCK_SIZE = 512;
    int constexpr THREADS_PER_CTA_TOP_P_INIT = 1024;

    Counter<T, IdxT, AccT>* counters = nullptr;
    HisT* histograms = nullptr;
    IdxT* countHistograms = nullptr;
    T* buf1 = nullptr;
    IdxT* idxBuf1 = nullptr;
    T* buf2 = nullptr;
    IdxT* idxBuf2 = nullptr;

    auto const workspaceSizes = getAirTopPWorkspaceSizes<T, isDeterministic>(params.batchSize, vocabSize);
    calcAlignedPointers(params.workspace, workspaceSizes)(
        counters, histograms, countHistograms, buf1, idxBuf1, buf2, idxBuf2);

    airTopPInitialize<T, IdxT, AccT, HisT, BitsPerPass, THREADS_PER_CTA_TOP_P_INIT>
        <<<params.batchSize, THREADS_PER_CTA_TOP_P_INIT, 0, stream>>>(counters, params.batchSize, vocabSize,
            params.probs, nullptr, params.topPs, params.curandState, params.randomVals, histograms, countHistograms,
            params.batchSlots);

    dim3 grid(params.blockNum, params.batchSize);
    // Sample with Top P given sorted tokens
    int constexpr numPasses = calcNumPasses<T, BitsPerPass>();
    auto kernel = airTopPSampling<T, IdxT, AccT, HisT, BitsPerPass, SAMPLING_BLOCK_SIZE, false, isDeterministic>;

    for (int pass = 0; pass < numPasses; ++pass)
    {
        if (pass == numPasses - 1)
        {
            kernel = airTopPSampling<T, IdxT, AccT, HisT, BitsPerPass, SAMPLING_BLOCK_SIZE, true, isDeterministic>;
        }

        kernel<<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(counters, histograms, countHistograms, params.outputIdsPtrs,
            params.sequenceLength, params.finishedInput, params.finishedOutput, params.cumLogProbs,
            params.outputLogProbs, params.endIds, params.maxBatchSize, params.skipDecode, pass, buf1, idxBuf1, buf2,
            idxBuf2, params.batchSlots);
    }
}

template <typename T>
void invokeBatchAirTopPSampling(TopPSamplingKernelParams<T> const& params, cudaStream_t stream)
{
    if (params.isDeterministic)
    {
        invokeAirTopPSamplingWithDeterministicPara<T, true>(params, stream);
    }
    else
    {
        invokeAirTopPSamplingWithDeterministicPara<T, false>(params, stream);
    }
}

template void invokeBatchAirTopPSampling(TopPSamplingKernelParams<float> const& params, cudaStream_t stream);

template void invokeBatchAirTopPSampling(TopPSamplingKernelParams<half> const& params, cudaStream_t stream);

template <typename T>
size_t getAirTopPWorkspaceSize(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic)
{
    std::vector<size_t> workspaceSizes;
    if (isDeterministic == true)
    {
        workspaceSizes = getAirTopPWorkspaceSizes<T, true>(batchSize, vocabSizePadded);
    }
    else
    {
        workspaceSizes = getAirTopPWorkspaceSizes<T, false>(batchSize, vocabSizePadded);
    }
    return calcAlignedSize(workspaceSizes, 256);
}

template size_t getAirTopPWorkspaceSize<float>(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic);
template size_t getAirTopPWorkspaceSize<half>(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic);

template uint32_t calcAirTopPBlockNum<float>(int batchSize, int len, int smCnt, bool isDeterministic);
template uint32_t calcAirTopPBlockNum<half>(int batchSize, int len, int smCnt, bool isDeterministic);
} // namespace kernels
} // namespace tensorrt_llm
