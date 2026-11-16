/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "kimi_k3_vl/csrc/k3MnnvlAllreduce.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/quantization.cuh"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::kimiK3MnnvlOneShot
{
using namespace ::tensorrt_llm::kernels;

using tensorrt_llm::common::isNegZero;
using tensorrt_llm::common::isLamportDirty;
using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::cuda_cast;
using tensorrt_llm::common::getMultiProcessorCount;
using tensorrt_llm::common::getDTypeSize;
using tensorrt_llm::common::loadPackedVolatile;

namespace detail
{
template <typename PackedType, typename T>
union PackedVec
{
    PackedType packed;
    T elements[sizeof(PackedType) / sizeof(T)];

    __device__ PackedVec& operator+=(PackedVec& other)
    {
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            elements[i] += other.elements[i];
        }
        return *this;
    }
};

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr)
{
    return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr)
{
    return *reinterpret_cast<PackedType const*>(ptr);
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;

template <typename T>
inline __device__ T warpReduceSumPartial(T val)
{
    int laneId = threadIdx.x & kLANE_ID_MASK;

    int warpSize = blockDim.x - (threadIdx.x & ~(kWARP_SIZE - 1));
    unsigned int active_mask = (1U << warpSize) - 1;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        int targetLane = laneId ^ mask;
        auto tmp = __shfl_xor_sync(active_mask, val, mask, kWARP_SIZE);
        val += targetLane < warpSize ? tmp : 0;
    }
    return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val)
{
    __shared__ T smem[kWARP_SIZE];
    int laneId = threadIdx.x & kLANE_ID_MASK;
    int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
    int warpNum = (blockDim.x + kWARP_SIZE - 1) >> kLOG2_WARP_SIZE;

    val = (warpId == warpNum - 1) ? warpReduceSumPartial(val) : tensorrt_llm::common::warpReduceSum(val);
    if (laneId == 0)
    {
        smem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0)
    {
        val = (laneId < warpNum) ? smem[laneId] : (T) 0.f;

        val = (warpNum == 1) ? warpReduceSumPartial(val) : tensorrt_llm::common::warpReduceSum(val);

        if constexpr (SYNC)
        {
            if (laneId == 0)
            {
                smem[warpId] = val;
            }
        }
    }
    if constexpr (SYNC)
    {
        __syncthreads();
        val = smem[0];
    }
    return val;
}

template <typename T>
inline __device__ T blockReduceSumFull(T val)
{
    __shared__ T smem[kWARP_SIZE];
    int lane_id = threadIdx.x & kLANE_ID_MASK;
    int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
    int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

    val = tensorrt_llm::common::warpReduceSum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();

    val = (lane_id < warp_num) ? smem[lane_id] : (T) 0.f;
    val = tensorrt_llm::common::warpReduceSum(val);

    return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSum(T val)
{
    bool hasPartialWarp = (blockDim.x & kLANE_ID_MASK) != 0;
    if (hasPartialWarp)
    {
        return blockReduceSumPartial<T, SYNC>(val);
    }
    else
    {
        return blockReduceSumFull<T>(val);
    }
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

inline bool requiresTwoAccessScaleGroup(ar_fusion::AllReduceFusionPattern pattern)
{
    return pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant
        || pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant;
}

template <bool UseCluster>
std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread, int accessGroupSize = 1)
{

    int clusterSize = UseCluster ? 8 : 1;
    int blockSize = 128;
    int threadsNeeded = divUp(dim, eltsPerThread);
    int loadsPerThread = 1;

    blockSize = divUp(threadsNeeded, clusterSize);
    if constexpr (UseCluster)
    {

        while ((threadsNeeded % clusterSize != 0 || divUp(threadsNeeded, clusterSize) % accessGroupSize != 0)
            && clusterSize > 1)
        {
            clusterSize /= 2;
        }
        int const maxDivisibleClusterSize = clusterSize;
        blockSize = divUp(threadsNeeded, clusterSize);

        while (blockSize < 128 && clusterSize >= 2)
        {
            blockSize *= 2;
            clusterSize /= 2;
        }
        int smCount = getMultiProcessorCount();

        while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512)
        {
            blockSize *= 2;
            clusterSize /= 2;
        }

        while (clusterSize < maxDivisibleClusterSize)
        {
            int const candidateClusterSize = clusterSize * 2;
            int const candidateBlockSize = divUp(threadsNeeded, candidateClusterSize);
            if (candidateBlockSize < 64 || candidateBlockSize % accessGroupSize != 0
                || numTokens * candidateClusterSize > smCount)
            {
                break;
            }
            clusterSize = candidateClusterSize;
            blockSize = candidateBlockSize;
        }
    }

    while (blockSize > 1024)
    {
        if constexpr (UseCluster)
        {
            if (clusterSize < 8)
            {
                clusterSize = clusterSize << 1;
            }
            else
            {
                if (loadsPerThread < 8)
                {
                    loadsPerThread += 1;
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            if (loadsPerThread < 8)
            {
                loadsPerThread += 1;
            }
            else
            {
                break;
            }
        }
        blockSize = divUp(threadsNeeded, clusterSize * loadsPerThread);
    }

    while (blockSize % accessGroupSize != 0 && loadsPerThread < 8)
    {
        loadsPerThread += 1;
        blockSize = divUp(threadsNeeded, clusterSize * loadsPerThread);
    }

    return {blockSize, clusterSize, loadsPerThread};
}

template <typename T>
struct MnnvlAllReduceKernelParams
{
    T* outputPtr;
    T* residualOutPtr;
    T const* shardPtr;
    T const* residualInPtr;
    T const* gammaPtr;
    T** inputPtrs;
    T* bufferInputPtr;
    T* mcastPtr;
    void* quantOutPtr;
    void* scaleOutPtr;
    float const* scaleFactorPtr;
    int numTokens;
    int tokenDim;
    int nRanks;
    int rank;
    float epsilon;
    uint32_t* bufferFlags;
    bool waitForResults;
    QuantizationSFLayout layout;
    bool earlyTrigger;
};

template <typename PackedType, typename T>
inline __device__ void sanitizeLamportPayload(PackedVec<PackedType, T>& value)
{
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
    {
        if (isNegZero(value.elements[i]))
        {
            value.elements[i] = cuda_cast<T, float>(0.F);
        }
    }
}

template <int Rank, uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ bool pollOneshotRemoteRank(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    if constexpr (Rank == LocalRank)
    {
        return true;
    }
    else
    {
        auto loaded = loadPackedVolatile<PackedType>(
            &stagePtrLocal[token * tokenDim * WorldSize + Rank * tokenDim + packedIdx * kELTS_PER_THREAD]);
        remoteValues[Rank].packed = loaded.packed;
        return !isLamportDirty(loaded);
    }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD, int... Ranks>
inline __device__ bool pollOneshotRemoteRanks(PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token,
    int tokenDim, int packedIdx, std::integer_sequence<int, Ranks...>)
{
    bool valid = true;
    ((valid &= pollOneshotRemoteRank<Ranks, WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
          remoteValues, stagePtrLocal, token, tokenDim, packedIdx)),
        ...);
    return valid;
}

template <typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulatePacked(float (&accum)[kELTS_PER_THREAD], PackedVec<PackedType, T> const& value)
{
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] += cuda_cast<float, T>(value.elements[i]);
    }
}

template <uint8_t WorldSize, int kRankChunk, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulateLamportRanksChunked(
    float (&accum)[kELTS_PER_THREAD], T const* input, int token, int tokenDim, int packedIdx)
{
    static_assert(kRankChunk > 0);
    static_assert(WorldSize % kRankChunk == 0);

#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] = 0.F;
    }

#pragma unroll 1
    for (int rankBase = 0; rankBase < WorldSize; rankBase += kRankChunk)
    {
        float chunkAccum[kELTS_PER_THREAD];
        while (1)
        {
            bool valid = true;
#pragma unroll
            for (int i = 0; i < kELTS_PER_THREAD; i++)
            {
                chunkAccum[i] = 0.F;
            }
#pragma unroll
            for (int rr = 0; rr < kRankChunk; rr++)
            {
                int const r = rankBase + rr;
                auto loaded = loadPackedVolatile<PackedType>(
                    &input[token * tokenDim * WorldSize + r * tokenDim + packedIdx * kELTS_PER_THREAD]);
                PackedVec<PackedType, T> value;
                value.packed = loaded.packed;
                valid &= !isLamportDirty(loaded);
                accumulatePacked<T, PackedType, kELTS_PER_THREAD>(chunkAccum, value);
            }
            if (valid)
            {
                break;
            }
        }
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            accum[i] += chunkAccum[i];
        }
    }
}

template <int Rank, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulateOneshotRank(float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue)
{
    if constexpr (Rank == LocalRank)
    {
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, localValue);
    }
    else
    {
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, remoteValues[Rank]);
    }
}

template <uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD, int... Ranks>
inline __device__ void accumulateOneshotRanks(float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue,
    std::integer_sequence<int, Ranks...>)
{
    (accumulateOneshotRank<Ranks, LocalRank, T, PackedType, kELTS_PER_THREAD>(accum, remoteValues, localValue), ...);
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void waitOneshotRemoteRanks(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    static_assert(LocalRank < WorldSize);
    while (1)
    {
        bool const valid = pollOneshotRemoteRanks<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
            remoteValues, stagePtrLocal, token, tokenDim, packedIdx, std::make_integer_sequence<int, WorldSize>{});
        if (valid)
        {
            break;
        }
    }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministic(
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue)
{
    static_assert(LocalRank < WorldSize);
    float accum[kELTS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] = 0.F;
    }
    accumulateOneshotRanks<LocalRank, T, PackedType, kELTS_PER_THREAD>(
        accum, remoteValues, localValue, std::make_integer_sequence<int, WorldSize>{});

    PackedVec<PackedType, T> packedAccum;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
    }
    return packedAccum;
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministicFastPath(
    PackedVec<PackedType, T> const& localValue, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    PackedVec<PackedType, T> remoteValues[WorldSize];
    waitOneshotRemoteRanks<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
        remoteValues, stagePtrLocal, token, tokenDim, packedIdx);
    return reduceOneshotDeterministic<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(remoteValues, localValue);
}

template <ar_fusion::AllReduceFusionPattern Pattern, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void quantizeEpilogue(PackedVec<PackedType, T> const& value,
    MnnvlAllReduceKernelParams<T> const& params, int packedAccessIdx, int accessIdInToken, int token)
{
    if constexpr (ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kFP4)
    {
        static_assert(kELTS_PER_THREAD == 8, "NVFP4 quantization expects eight elements per 16-byte access.");
        constexpr int kSFVecSize = 16;
        using QuantPackedVec = tensorrt_llm::kernels::PackedVec<T>;
        QuantPackedVec packVal = *reinterpret_cast<QuantPackedVec const*>(&value.packed);
        auto* sfOut = cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt, token, accessIdInToken, std::nullopt,
            params.tokenDim / kSFVecSize, reinterpret_cast<uint32_t*>(params.scaleOutPtr), params.layout);
        reinterpret_cast<uint32_t*>(params.quantOutPtr)[packedAccessIdx]
            = cvt_warp_fp16_to_fp4<T, kSFVecSize, false>(packVal, *params.scaleFactorPtr, sfOut);
    }
    else if constexpr (ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kFP8)
    {
        float const scale = 1.F / *params.scaleFactorPtr;
        using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
        PackedQuantizedType quantized;
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; ++i)
        {
            reinterpret_cast<__nv_fp8_e4m3*>(&quantized)[i]
                = static_cast<__nv_fp8_e4m3>(static_cast<float>(value.elements[i]) * scale);
        }
        reinterpret_cast<PackedQuantizedType*>(params.quantOutPtr)[packedAccessIdx] = quantized;
    }
    else
    {
        static_assert(ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kNone, "Invalid quant type");
    }
}

template <ar_fusion::AllReduceFusionPattern Pattern, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void writeEpilogueOutput(PackedVec<PackedType, T> const& value,
    MnnvlAllReduceKernelParams<T> const& params, int threadOffset, int packedAccessIdx, int accessIdInToken, int token)
{
    if constexpr (ar_fusion::HasAllReduceOut<Pattern> || ar_fusion::HasNormOut<Pattern>)
    {
        if (params.outputPtr != nullptr)
        {
            reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = value.packed;
        }
    }
    if constexpr (ar_fusion::GetQuantType<Pattern> != ar_fusion::QuantType::kNone)
    {
        quantizeEpilogue<Pattern, T, PackedType, kELTS_PER_THREAD>(
            value, params, packedAccessIdx, accessIdInToken, token);
    }
}

}

using detail::PackedVec;
using detail::loadPacked;
using detail::blockReduceSum;
using detail::divUp;
using detail::MnnvlAllReduceKernelParams;
using detail::sanitizeLamportPayload;
using detail::accumulateLamportRanksChunked;
using detail::reduceOneshotDeterministicFastPath;
using detail::writeEpilogueOutput;

template <uint8_t WorldSize, typename T,
    ar_fusion::AllReduceFusionPattern Pattern = ar_fusion::AllReduceFusionPattern::kAllReduce,
    typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshotAllreduceFusionKernel(MnnvlAllReduceKernelParams<T> params)
{
    constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr uint32_t kELT_SIZE = sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int packedIdx = cluster.thread_rank();
    int token = blockIdx.x;
    int threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;

    cudaGridDependencySynchronize();
    if (params.earlyTrigger)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#else
    int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;

    int threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;
#endif

    LamportFlags<PackedType> flag(params.bufferFlags, 1);
    T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
    T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));
    bool const inBounds = packedIdx * kELTS_PER_THREAD < params.tokenDim;

    PackedVec<PackedType, T> val;
    if (inBounds)
    {
        val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
        sanitizeLamportPayload(val);

        reinterpret_cast<PackedType*>(
            &stagePtrMcast[token * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
            = val.packed;
    }

    flag.ctaArrive();

    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

    if (!inBounds)
    {
        return;
    }

    PackedVec<PackedType, T> packedAccum;
    if constexpr (WorldSize <= 8)
    {
        packedAccum = val;
#define RUN_ONESHOT_LOCAL_RANK(LOCAL_RANK)                                                                             \
    case LOCAL_RANK:                                                                                                   \
        if constexpr (WorldSize > LOCAL_RANK)                                                                          \
        {                                                                                                              \
            packedAccum = reduceOneshotDeterministicFastPath<WorldSize, LOCAL_RANK, T, PackedType, kELTS_PER_THREAD>(  \
                val, stagePtrLocal, token, params.tokenDim, packedIdx);                                                \
        }                                                                                                              \
        break

        switch (params.rank)
        {
            RUN_ONESHOT_LOCAL_RANK(0);
            RUN_ONESHOT_LOCAL_RANK(1);
            RUN_ONESHOT_LOCAL_RANK(2);
            RUN_ONESHOT_LOCAL_RANK(3);
            RUN_ONESHOT_LOCAL_RANK(4);
            RUN_ONESHOT_LOCAL_RANK(5);
            RUN_ONESHOT_LOCAL_RANK(6);
            RUN_ONESHOT_LOCAL_RANK(7);
        }
#undef RUN_ONESHOT_LOCAL_RANK
    }
    else
    {

        constexpr int kRankChunk = 8;
        float accum[kELTS_PER_THREAD];
        accumulateLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS_PER_THREAD>(
            accum, stagePtrLocal, token, params.tokenDim, packedIdx);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if constexpr (ar_fusion::HasResidual<Pattern>)
    {

        PackedVec<PackedType, T> residualIn;
        residualIn.packed = *reinterpret_cast<PackedType const*>(&params.residualInPtr[threadOffset]);
        packedAccum += residualIn;
        if constexpr (ar_fusion::HasResidualOut<Pattern>)
        {
            *reinterpret_cast<PackedType*>(&params.residualOutPtr[threadOffset]) = packedAccum.packed;
        }
    }
    if constexpr (ar_fusion::HasRMSNorm<Pattern>)
    {

        PackedVec<PackedType, T> gamma;
        gamma.packed = *reinterpret_cast<PackedType const*>(&params.gammaPtr[packedIdx * kELTS_PER_THREAD]);

        float threadSum = 0.F;
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {

            threadSum += cuda_cast<float, T>(packedAccum.elements[i] * packedAccum.elements[i]);
        }
        float blockSum = blockReduceSum<float, true>(threadSum);

        float fullSum = blockSum;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        __shared__ float sharedVal[8];
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        int const numBlocks = cluster.num_blocks();
        if (numBlocks > 1)
        {
            fullSum = 0.F;

            int const blockRank = cluster.block_rank();

            cluster.sync();
            if (threadIdx.x < numBlocks)
            {
                cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
            }
            cluster.barrier_wait(cluster.barrier_arrive());
            for (int i = 0; i < numBlocks; ++i)
            {
                fullSum += sharedVal[i];
            }
        }
#endif
        float rcpRms = rsqrtf(fullSum / params.tokenDim + params.epsilon);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(
                cuda_cast<float, T>(packedAccum.elements[i]) * rcpRms * cuda_cast<float, T>(gamma.elements[i]));
        }
    }
    int const packedAccessIdx = threadOffset / kELTS_PER_THREAD;
    writeEpilogueOutput<Pattern, T, PackedType, kELTS_PER_THREAD>(
        packedAccum, params, threadOffset, packedAccessIdx, packedIdx, token);
    flag.waitAndUpdate({static_cast<uint32_t>(params.numTokens * params.tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

using detail::adjustGridConfig;
using detail::requiresTwoAccessScaleGroup;

void oneshotAllreduceFusionOp(AllReduceFusionParams const& params)
{

    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSMVersion >= 90, "[MNNVL AllReduceOneShot] requires SM 90 or newer.");
    int const numTokens = params.numTokens;
    int const tokenDim = params.tokenDim;
    int const eltsPerThread = sizeof(float4) / getDTypeSize(params.dType);
    int const accessGroupSize = requiresTwoAccessScaleGroup(params.pattern) ? 2 : 1;

    auto [blockSize, clusterSize, loadsPerThread]
        = adjustGridConfig<true>(numTokens, tokenDim, eltsPerThread, accessGroupSize);
    dim3 grid(numTokens, clusterSize, 1);

    TLLM_LOG_DEBUG(
        "[MNNVL AllReduceOneShot] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
        "loads_per_thread: %d, deterministic_reduction_order: 1, threads_needed: %d",
        numTokens, clusterSize, blockSize, clusterSize, loadsPerThread, divUp(tokenDim, eltsPerThread));

    TLLM_CHECK_WITH_INFO(blockSize <= 1024 && loadsPerThread == 1,
        "Hidden Dimension %d exceeds the maximum supported hidden dimension (%d)", tokenDim, 1024 * 8 * eltsPerThread);

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 1;
    attrs[1].val.clusterDim.y = clusterSize;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{
        .gridDim = grid,
        .blockDim = blockSize,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = attrs,
        .numAttrs = 2U,
    };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, PATTERN)                                                                \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &oneshotAllreduceFusionKernel<WORLD_SIZE, T, PATTERN>, kernelParams));
#define DISPATCH_ALLREDUCE_PATTERN(WORLD_SIZE, T)                                                                      \
    if (params.pattern == ar_fusion::AllReduceFusionPattern::kAllReduce)                                               \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kAllReduce);                         \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm)                                  \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm);                 \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant)                          \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant);         \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant)                       \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);      \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant)                          \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant);     \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceOneShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant)                       \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant);  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceOneShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARRMSNorm)                                          \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARRMSNorm);                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceOneShot] Unsupported fusion pattern.");                           \
    }

    auto dispatchImpl = [&](auto* type_ptr) -> bool
    {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
        T* mcPtr = reinterpret_cast<T*>(params.multicastPtr);
        T* output = reinterpret_cast<T*>(params.output);
        T* residualOut = reinterpret_cast<T*>(params.residualOut);
        T const* input = reinterpret_cast<T const*>(params.input);
        T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
        T const* gamma = reinterpret_cast<T const*>(params.gamma);
        MnnvlAllReduceKernelParams<T> kernelParams{output, residualOut, input, residualIn, gamma, ucPtrs,
            reinterpret_cast<T*>(params.bufferPtrLocal), mcPtr, params.quantOut, params.scaleOut, params.scaleFactor,
            numTokens, tokenDim, params.nRanks, params.rank, static_cast<float>(params.epsilon), params.bufferFlags,
            false, params.layout, params.earlyTrigger};

        switch (params.nRanks)
        {

        case 2: DISPATCH_ALLREDUCE_PATTERN(2, T); return true;
        case 4: DISPATCH_ALLREDUCE_PATTERN(4, T); return true;
        case 8: DISPATCH_ALLREDUCE_PATTERN(8, T); return true;
        case 16: DISPATCH_ALLREDUCE_PATTERN(16, T); return true;
        case 32: DISPATCH_ALLREDUCE_PATTERN(32, T); return true;
        case 64: DISPATCH_ALLREDUCE_PATTERN(64, T); return true;
        }
        return false;
    };
#undef LAUNCH_ALLREDUCE_KERNEL
#undef DISPATCH_ALLREDUCE_PATTERN
    bool launched = (params.dType == tensorrt_llm::DataType::kBF16 && dispatchImpl((__nv_bfloat16*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kFLOAT && dispatchImpl((float*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kHALF && dispatchImpl((__nv_half*) nullptr));
    if (!launched)
    {
        TLLM_CHECK_WITH_INFO(false, "Failed to dispatch MNNVL AllReduceOneShot kernel.");
    }
}

}

TRTLLM_MV2_NAMESPACE_END
