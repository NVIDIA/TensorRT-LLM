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
#include "mnnvlAllreduceKernels.h"
#include "tensorrt_llm/common/config.h"
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mnnvl
{

using tensorrt_llm::common::isNegZero;
using tensorrt_llm::common::isLamportDirty;
using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::cuda_cast;
using tensorrt_llm::common::getMultiProcessorCount;
using tensorrt_llm::common::getDTypeSize;
using tensorrt_llm::common::loadPackedVolatile;

// Guard the helper function used for this kernel.
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

    __device__ PackedVec operator+(PackedVec& other)
    {
        PackedVec result;
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            result.elements[i] = elements[i] + other.elements[i];
        }
        return result;
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

template <typename T_IN>
inline __device__ void copyF4(T_IN* dst, T_IN const* src)
{
    float4* dst4 = reinterpret_cast<float4*>(dst);
    float4 const* src4 = reinterpret_cast<float4 const*>(src);
    __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;

template <typename T>
inline __device__ T warpReduceSumPartial(T val)
{
    int laneId = threadIdx.x & kLANE_ID_MASK;
    // We make sure only the last warp will call this function
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

// SYNC:
//  - True: share the sume across all threads
//  - False: only thread 0 get the sum; Other thread's value is undefined.
template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val)
{
    __shared__ T smem[kWARP_SIZE];
    int laneId = threadIdx.x & kLANE_ID_MASK;
    int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
    int warpNum = (blockDim.x + kWARP_SIZE - 1) >> kLOG2_WARP_SIZE; // Ceiling division to include partial warps

    val = (warpId == warpNum - 1) ? warpReduceSumPartial(val) : tensorrt_llm::common::warpReduceSum(val);
    if (laneId == 0)
    {
        smem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0)
    {
        val = (laneId < warpNum) ? smem[laneId] : (T) 0.f;
        // Need to consider the corner case where we only have one warp and it is partial
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

// blockReduceSum in reduceKernelUtils.cuh returns result only on warp0
// So we need a duplicate implementation here where all threads get the result
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

// We have to define this again since the one in mathUtils.h is shadowed by the one from cudaUtils.h, which is a
// host-only function!
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

// A helper function to tune the grid configuration for fused oneshot and RMSNorm kernels.
// Return (block_size, cluster_size, loads_per_thread).
template <bool UseCluster>
std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread, int accessGroupSize = 1)
{
    // Step 1: Start from the widest cluster we are willing to launch. The caller can request a no-CGA
    // fallback for multi-load RMSNorm rows, in which case the cluster width stays at one CTA.
    int clusterSize = UseCluster ? 8 : 1;
    int blockSize = 128;
    int threadsNeeded = divUp(dim, eltsPerThread);
    int loadsPerThread = 1;

    blockSize = divUp(threadsNeeded, clusterSize);
    if constexpr (UseCluster)
    {
        // Step 2: Shrink the cluster until the hidden dimension partitions cleanly across CTAs
        // and each CTA contains an integral packed-access group. NVFP4 scale generation uses a
        // two-access group, so it must not straddle a CTA boundary.
        while ((threadsNeeded % clusterSize != 0 || divUp(threadsNeeded, clusterSize) % accessGroupSize != 0)
            && clusterSize > 1)
        {
            clusterSize /= 2;
        }
        int const maxDivisibleClusterSize = clusterSize;
        blockSize = divUp(threadsNeeded, clusterSize);
        // Step 3: If divisibility leaves each CTA too small, trade cluster width for at least
        // a 128-thread CTA. This improves occupancy and avoids tiny CTAs when the row is narrow.
        while (blockSize < 128 && clusterSize >= 2)
        {
            blockSize *= 2;
            clusterSize /= 2;
        }
        int smCount = getMultiProcessorCount();
        // Step 4: If the token grid already has enough CTAs to cover the GPU, reduce cluster
        // width and make CTAs larger. This avoids over-partitioning one token across too many CTAs.
        while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512)
        {
            blockSize *= 2;
            clusterSize /= 2;
        }
        // Step 5: If the token grid still underfills the GPU, restore cluster width up to the
        // divisibility limit. We accept 64-thread CTAs here to expose more CTAs per token.
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

    // Step 6: For very wide rows, first increase cluster width on SM90+ and then increase
    // per-thread loads. The goal is to keep block_size within CUDA's 1024-thread limit.
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
        auto* sfOut = cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt /* batchIdx */, token, accessIdInToken,
            std::nullopt /* numRows */, params.tokenDim / kSFVecSize, reinterpret_cast<uint32_t*>(params.scaleOutPtr),
            params.layout);
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

} // namespace detail

using detail::PackedVec;
using detail::loadPacked;
using detail::blockReduceSum;
using detail::divUp;
using detail::copyF4;
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
#else
    int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;
#endif

    // We only use 1 stage for the oneshot allreduce. Arrivals are counted per CTA: the RMSNorm epilogue's cluster
    // reduction below must be each thread's only phase of the cluster barrier.
    LamportFlags<PackedType> flag(params.bufferFlags, 1);
    T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
    T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));
    bool const inBounds = packedIdx * kELTS_PER_THREAD < params.tokenDim;

    // ==================== Broadcast tokens to each rank =============================
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
    // ======================= Lamport Sync and clear the output buffer from previous iteration
    // =============================
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

    if (!inBounds)
    {
        return;
    }

    // ======================= Reduction =============================
    // Fully deterministic: every rank uses the exact same reduction order. For WorldSize <= 8, specialize the local
    // slot so the fast path reuses `val` from registers without a dynamic `remoteValues[params.rank]` store. Larger
    // world sizes use the compact fallback because the benefit is thin but specializing every rank significantly
    // increases compile time.
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
        // Chunk Lamport polling so only a bounded rank set is live at once, avoiding register spills for large
        // world sizes.
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
        // =============================== Residual ===============================
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
        // =============================== Rmsnorm ================================
        PackedVec<PackedType, T> gamma;
        gamma.packed = *reinterpret_cast<PackedType const*>(&params.gammaPtr[packedIdx * kELTS_PER_THREAD]);

        float threadSum = 0.F;
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            // FIXME: Use float square if accuracy issue
            threadSum += cuda_cast<float, T>(packedAccum.elements[i] * packedAccum.elements[i]);
        }
        float blockSum = blockReduceSum<float, true>(threadSum);

        float fullSum = blockSum;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        __shared__ float sharedVal[8]; // Temporary variable to share the sum within block
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        int const numBlocks = cluster.num_blocks();
        if (numBlocks > 1)
        {
            fullSum = 0.F;
            // Need to reduce over the entire cluster
            int const blockRank = cluster.block_rank();
            // Every CTA of the cluster must have started before another CTA writes its shared memory.
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
    // C++17 compatible alternative using a template function
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
            false, params.layout};

        switch (params.nRanks)
        {
            // FIXME: Do we need other world sizes?
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

enum MNNVLTwoShotStage : uint8_t
{
    SCATTER = 0,
    BROADCAST = 1,
    NUM_STAGES = 2,
};

template <uint8_t WorldSize, typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(MnnvlAllReduceKernelParams<T> params)
{
    constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr uint32_t kELT_SIZE = sizeof(T);

    int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;

    int destRank = token % WorldSize;
    int destTokenOffset = token / WorldSize;
    bool const inBounds = packedIdx * kELTS_PER_THREAD < params.tokenDim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    LamportFlags<PackedType> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);

    T* scatterBufLocal
        = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER));
    T* scatterBufDest
        = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
    T* broadcastBufW = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, MNNVLTwoShotStage::BROADCAST));
    T* broadcastBufR
        = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST));

    // =============================== Scatter ===============================

    // Load vectorized data
    PackedVec<PackedType, T> val;
    if (inBounds)
    {
        val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
        sanitizeLamportPayload(val);

        // Store vectorized data
        reinterpret_cast<PackedType*>(
            &scatterBufDest[destTokenOffset * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
            = val.packed;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER);

    // =============================== Reduction and Broadcast ===============================

    if (inBounds && (token % WorldSize) == params.rank)
    {
        int localToken = token / WorldSize;

        // Store vectorized result
        PackedVec<PackedType, T> packedAccum;
        // Chunk Lamport polling so only a bounded rank set is live at once, avoiding register spills for large
        // world sizes.
        constexpr int kRankChunk = WorldSize < 16 ? WorldSize : 16;
        float accum[kELTS_PER_THREAD];
        accumulateLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS_PER_THREAD>(
            accum, scatterBufLocal, localToken, params.tokenDim, packedIdx);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
        }
        sanitizeLamportPayload(packedAccum);
        reinterpret_cast<PackedType*>(&broadcastBufW[token * params.tokenDim])[packedIdx] = packedAccum.packed;
    }

    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST);

    // Optionally wait for results if the next layer isn't doing the Lamport check
    if (params.waitForResults)
    {
        // Update the atomic counter to indicate the block has read the offsets
        flag.ctaArrive();

        if (inBounds)
        {
            auto loaded = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
            while (isLamportDirty(loaded))
            {
                loaded = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
            }
            if (params.outputPtr)
            {
                reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = loaded.packed;
            }
        }

        // Update the buffer flags
        flag.waitAndUpdate({static_cast<uint32_t>(divUp<uint32_t>(params.numTokens, WorldSize) * WorldSize
                                * params.tokenDim * kELT_SIZE),                    // Clear Size for scatter stage
            static_cast<uint32_t>(params.numTokens * params.tokenDim * kELT_SIZE), // Clear Size for broadcast stage
            0, 0});
        // If not wait for results, we will rely on the following kernel to update the buffer
    }
}

// =============================== E4M3 two-shot payload ===============================
// AllReduceFusionParams::fp8Payload (BF16, H % 1024 == 0, residual RMSNorm fusion, 2 / 4 / 8 ranks): both Lamport
// stages carry E4M3 instead of BF16. A row is H E4M3 bytes followed by H / 256 FP32 scales, one per warp of the
// all-reduce kernel (32 lanes x 8 elements): values x * (448 / amax) rounded to E4M3 (RN, satfinite), scale
// amax * (1 / 448). The owner of a token sums the other ranks' dequantized vectors and its own exact input in rank
// order (FP32), quantizes the sum the same way and multicasts it; rmsNormLamport takes BF16(E4M3 x scale). Each stage
// moves half the bytes; the reduced value carries two E4M3 roundings, so this is not bitwise the BF16 path. A word
// equal to the Lamport sentinel (E4M3 +0, +0, +0, -0) is stored as four +0.
namespace fp8payload
{
constexpr uint32_t kSentinel = 0x80000000u;
constexpr uint32_t kScaleBlock = 256;

__host__ __device__ inline uint32_t rowBytes(uint32_t tokenDim)
{
    return tokenDim + tokenDim / kScaleBlock * 4;
}

inline __device__ uint32_t ldVolatileU32(void const* ptr)
{
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

inline __device__ uint2 ldVolatileU32x2(void const* ptr)
{
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
    return v;
}

// Two FP32 -> two E4M3 (RN, satfinite) in the low 16 bits, lo first. The PTX needs SM 8.9+.
inline __device__ uint32_t e4m3x2(float lo, float hi)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    uint16_t h;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h) : "f"(hi), "f"(lo));
    return h;
#else
    return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
#endif
}

// Four E4M3 (byte i = element i) -> four FP32 (exact, through FP16).
inline __device__ void e4m3x4ToFloat(uint32_t w, float* out)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    uint32_t h01, h23;
    asm("{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n\t}"
        : "=r"(h01), "=r"(h23)
        : "r"(w));
    asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\tcvt.f32.f16 %0, a;\n\tcvt.f32.f16 %1, b;\n\t}"
        : "=f"(out[0]), "=f"(out[1])
        : "r"(h01));
    asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\tcvt.f32.f16 %0, a;\n\tcvt.f32.f16 %1, b;\n\t}"
        : "=f"(out[2]), "=f"(out[3])
        : "r"(h23));
#else
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        out[i]
            = __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(w >> (8 * i)), __NV_E4M3)));
    }
#endif
}

// max |x| over the warp's 32 x 8 values (every lane gets it), at least 1e-30.
inline __device__ float warpAmax(float const (&x)[8])
{
    float m = fabsf(x[0]);
#pragma unroll
    for (int i = 1; i < 8; i++)
    {
        m = fmaxf(m, fabsf(x[i]));
    }
#pragma unroll
    for (int k = 16; k > 0; k >>= 1)
    {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, k));
    }
    return fmaxf(m, 1e-30f);
}

// Eight values -> eight E4M3 (two words) on the warp's scale; returns the scale. Every lane of the warp must call it.
inline __device__ float quantize8(float const (&x)[8], uint2& q)
{
    float const amax = warpAmax(x);
    float const inv = __fdiv_rn(448.F, amax);
    uint32_t const w0 = e4m3x2(__fmul_rn(x[0], inv), __fmul_rn(x[1], inv))
        | (e4m3x2(__fmul_rn(x[2], inv), __fmul_rn(x[3], inv)) << 16);
    uint32_t const w1 = e4m3x2(__fmul_rn(x[4], inv), __fmul_rn(x[5], inv))
        | (e4m3x2(__fmul_rn(x[6], inv), __fmul_rn(x[7], inv)) << 16);
    q.x = w0 == kSentinel ? 0U : w0;
    q.y = w1 == kSentinel ? 0U : w1;
    return __fmul_rn(amax, 1.F / 448.F);
}
} // namespace fp8payload

// The two-shot all-reduce with the E4M3 payload (grid and stages as twoshotAllreduceKernel; H % 1024 == 0, so every
// thread is in bounds and a warp covers one 256-element scale block). The fused rmsNormLamport (Fp8Payload) waits for
// the BROADCAST stage and updates the flags.
template <uint8_t WorldSize>
__global__ __launch_bounds__(128) void twoshotAllreduceKernelFp8(MnnvlAllReduceKernelParams<__nv_bfloat16> params)
{
    using T = __nv_bfloat16;
    constexpr int kELTS_PER_THREAD = 8;

    int const packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int const token = blockIdx.x;
    int const threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;
    int const destRank = token % WorldSize;
    int const destTokenOffset = token / WorldSize;
    uint32_t const rowBytes = fp8payload::rowBytes(params.tokenDim);
    uint32_t const dataOff = packedIdx * kELTS_PER_THREAD;
    uint32_t const scaleOff = params.tokenDim + packedIdx / 32 * 4;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    LamportFlags<float4> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);

    char* scatterBufLocal
        = reinterpret_cast<char*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER));
    char* scatterBufDest
        = reinterpret_cast<char*>(flag.getCurLamportBuf(params.inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
    char* broadcastBufW = reinterpret_cast<char*>(flag.getCurLamportBuf(params.mcastPtr, MNNVLTwoShotStage::BROADCAST));

    // =============================== Scatter: E4M3 + the warp's scale ===============================
    float x[kELTS_PER_THREAD];
    {
        PackedVec<float4, T> val;
        val.packed = loadPacked<float4>(&params.shardPtr[threadOffset]);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            x[i] = cuda_cast<float, T>(val.elements[i]);
        }
        uint2 q;
        float const scale = fp8payload::quantize8(x, q);
        char* row = scatterBufDest + (static_cast<size_t>(destTokenOffset) * WorldSize + params.rank) * rowBytes;
        *reinterpret_cast<uint2*>(row + dataOff) = q;
        if (threadIdx.x % 32 == 0)
        {
            *reinterpret_cast<uint32_t*>(row + scaleOff) = __float_as_uint(scale);
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER);

    // =============================== Reduction and Broadcast ===============================
    // Owner (uniform per CTA): every rank's E4M3 vector and scale, FP32 sum in rank order with its own input exact,
    // then the sum quantized the same way into every rank's BROADCAST stage.
    if (destRank == params.rank)
    {
        char const* rows = scatterBufLocal + static_cast<size_t>(destTokenOffset) * WorldSize * rowBytes;
        uint32_t got[WorldSize][3];
        bool valid = false;
        while (!valid)
        {
            valid = true;
#pragma unroll
            for (int r = 0; r < WorldSize; r++)
            {
                uint2 const d = fp8payload::ldVolatileU32x2(rows + r * rowBytes + dataOff);
                uint32_t const sw = fp8payload::ldVolatileU32(rows + r * rowBytes + scaleOff);
                got[r][0] = d.x;
                got[r][1] = d.y;
                got[r][2] = sw;
                valid &= d.x != fp8payload::kSentinel && d.y != fp8payload::kSentinel && sw != fp8payload::kSentinel;
            }
        }
        float accum[kELTS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            accum[i] = 0.F;
        }
#pragma unroll
        for (int r = 0; r < WorldSize; r++)
        {
            float const sc = __uint_as_float(got[r][2]);
            float d[kELTS_PER_THREAD];
            fp8payload::e4m3x4ToFloat(got[r][0], d);
            fp8payload::e4m3x4ToFloat(got[r][1], d + 4);
#pragma unroll
            for (int i = 0; i < kELTS_PER_THREAD; i++)
            {
                accum[i] = __fadd_rn(accum[i], r == params.rank ? x[i] : __fmul_rn(d[i], sc));
            }
        }
        uint2 q;
        float const scale = fp8payload::quantize8(accum, q);
        char* row = broadcastBufW + static_cast<size_t>(token) * rowBytes;
        *reinterpret_cast<uint2*>(row + dataOff) = q;
        if (threadIdx.x % 32 == 0)
        {
            *reinterpret_cast<uint32_t*>(row + scaleOff) = __float_as_uint(scale);
        }
    }

    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST);
}

// This kernel works performant when loads_per_thread is 1.
// For this mode, we are able to support up to 1024 (threads) x 8 (elements) = 8192 hidden dimension.
// There are two options for further scaling up:
//      1. Use CGA if supported. It expands the hidden dimension to 8k x 8 = 64k.
//      2. Set loads_per_thread >1. Which can be used if CGA is not supported. Note that this will be limited by the
//      shared memory size and register count.
template <typename T, ar_fusion::AllReduceFusionPattern Pattern, bool UseCGA, int LoadsPerThread = 1,
    bool Fp8Payload = false>
__global__ __launch_bounds__(1024) void rmsNormLamport(MnnvlAllReduceKernelParams<T> params)
{
    static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T);

    uint32_t const token = blockIdx.x;
    uint32_t const blockSize = blockDim.x;
    uint32_t const threadOffset = threadIdx.x;

    uint32_t numThreads = blockSize;
    uint32_t clusterSize = 1;
    uint32_t blockOffset = 0;
    if constexpr (UseCGA)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        numThreads = cluster.num_threads();
        clusterSize = cluster.num_blocks();
        blockOffset = cluster.block_rank();
#endif
    }
    uint32_t const dimPadded
        = divUp(static_cast<uint32_t>(params.tokenDim), kELTS_PER_LOAD * numThreads) * kELTS_PER_LOAD * numThreads;
    uint32_t const elemsPerThread = dimPadded / numThreads;
    uint32_t const loadStride = blockSize;

    extern __shared__ uint8_t smem[];
    float rInput[LoadsPerThread * kELTS_PER_LOAD];
    uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

    uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T);
    T* smemInput = reinterpret_cast<T*>(&smem[0]);
    T* smemResidual = reinterpret_cast<T*>(&smem[smemBufferSize]);
    T* smemGamma = reinterpret_cast<T*>(&smem[2 * smemBufferSize]);

    // Arrivals are counted per CTA: with UseCGA the RMSNorm cluster reduction below must be each thread's only phase
    // of the cluster barrier.
    LamportFlags<float4> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
    T* input = reinterpret_cast<T*>(
        flag.getCurLamportBuf(reinterpret_cast<void*>(params.bufferInputPtr), MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    // The offset that current thread should load from. Note that the hidden dimension is split by CGA size and each
    // block loads a contiguous chunk;
    // The size of chunk that each block processes
    uint32_t const blockChunkSize
        = divUp(static_cast<uint32_t>(params.tokenDim), clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
    uint32_t const blockLoadOffset = token * params.tokenDim + blockOffset * blockChunkSize;

#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++)
    {
        // Each block load a contiguous chunk of tokens
        uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        offsets[i] = blockLoadOffset + threadLoadOffset;
    }

    if constexpr (ar_fusion::HasResidual<Pattern>)
    {
#pragma unroll
        for (uint32_t i = 0; i < LoadsPerThread; i++)
        {
            uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
            if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
            {
                copyF4(&smemResidual[threadLoadOffset], &params.residualInPtr[blockLoadOffset + threadLoadOffset]);
            }
        }
        __pipeline_commit();
    }
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++)
    {
        uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
        {
            copyF4(&smemGamma[threadLoadOffset], &params.gammaPtr[blockOffset * blockChunkSize + threadLoadOffset]);
        }
    }
    __pipeline_commit();

    flag.ctaArrive();
    bool valid = false;
    // ACQBLK if not lamport
    while (!valid)
    {
        valid = true;
#pragma unroll
        for (uint32_t i = 0; i < LoadsPerThread; i++)
        {
            uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;

            if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
            {

                float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
                if constexpr (Fp8Payload)
                {
                    // This token's E4M3 row (H bytes, then H / 256 FP32 scales): BF16(E4M3 x scale) into smem.
                    uint32_t const elt = blockOffset * blockChunkSize + threadLoadOffset;
                    char const* row = reinterpret_cast<char const*>(input)
                        + static_cast<size_t>(token) * fp8payload::rowBytes(params.tokenDim);
                    uint2 const d = fp8payload::ldVolatileU32x2(row + elt);
                    uint32_t const sw
                        = fp8payload::ldVolatileU32(row + params.tokenDim + elt / fp8payload::kScaleBlock * 4);
                    valid
                        &= d.x != fp8payload::kSentinel && d.y != fp8payload::kSentinel && sw != fp8payload::kSentinel;
                    float f[8];
                    fp8payload::e4m3x4ToFloat(d.x, f);
                    fp8payload::e4m3x4ToFloat(d.y, f + 4);
                    float const sc = __uint_as_float(sw);
                    PackedVec<float4, T> deq;
#pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        deq.elements[j] = cuda_cast<T, float>(__fmul_rn(f[j], sc));
                    }
                    *dst4 = deq.packed;
                }
                else
                {
                    float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

                    auto loaded = loadPackedVolatile<float4>(src4);
                    // Assume that the 16B were written atomically, so we only need to check one value
                    valid &= !isLamportDirty(loaded);
                    *dst4 = loaded.packed;
                }
            }
        }
    }

    __pipeline_wait_prior(1);
    __syncthreads();

    float threadSum = 0.f;
#pragma unroll
    for (int i = 0; i < LoadsPerThread; i++)
    {
        int threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
        {
            PackedVec<float4, T> inp{.packed = loadPacked<float4>(&smemInput[threadLoadOffset])};
            PackedVec<float4, T> inpPlusRes = inp;
            if constexpr (ar_fusion::HasResidual<Pattern>)
            {
                PackedVec<float4, T> res{.packed = loadPacked<float4>(&smemResidual[threadLoadOffset])};
                inpPlusRes = inp + res;
                if constexpr (ar_fusion::HasResidualOut<Pattern>)
                {
                    *reinterpret_cast<float4*>(&params.residualOutPtr[blockLoadOffset + threadLoadOffset])
                        = inpPlusRes.packed;
                }
            }

#pragma unroll
            for (int j = 0; j < kELTS_PER_LOAD; j++)
            {
                rInput[i * kELTS_PER_LOAD + j] = cuda_cast<float, T>(inpPlusRes.elements[j]);
                threadSum += cuda_cast<float, T>(inpPlusRes.elements[j] * inpPlusRes.elements[j]);
            }
        }
    }

    __pipeline_wait_prior(0);

    float blockSum = blockReduceSum<float, true>(threadSum);

    float fullSum = blockSum;
    // Use CGA Reduction if supported
    if constexpr (UseCGA)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        __shared__ float sharedVal[8];
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        int const numBlocks = cluster.num_blocks();
        if (numBlocks > 1)
        {
            fullSum = 0.F;
            // Need to reduce over the entire cluster
            int const blockRank = cluster.block_rank();
            // Every CTA of the cluster must have started before another CTA writes its shared memory.
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
    }

    float rcpRms = rsqrtf(fullSum / params.tokenDim + params.epsilon);

#pragma unroll
    for (int i = 0; i < LoadsPerThread; i++)
    {
        PackedVec<float4, T> r_out;
        uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
        {
            PackedVec<float4, T> gamma = {.packed = loadPacked<float4>(&smemGamma[threadLoadOffset])};

#pragma unroll
            for (uint32_t j = 0; j < kELTS_PER_LOAD; j++)
            {
                r_out.elements[j] = cuda_cast<T, float>(
                    cuda_cast<float, T>(gamma.elements[j]) * rInput[i * kELTS_PER_LOAD + j] * rcpRms);
            }

            int const accessIdInToken = (blockOffset * blockChunkSize + threadLoadOffset) / kELTS_PER_LOAD;
            int const packedAccessIdx = (blockLoadOffset + threadLoadOffset) / kELTS_PER_LOAD;
            writeEpilogueOutput<Pattern, T, float4, kELTS_PER_LOAD>(
                r_out, params, blockLoadOffset + threadLoadOffset, packedAccessIdx, accessIdInToken, token);
        }
    }
    constexpr int kELTS_SIZE = sizeof(T);

    // Issue ACQBLK at the end. Assuming preceding kernel will not modify the buffer_flags.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // Update the buffer pointers
    if constexpr (Fp8Payload)
    {
        uint32_t const rowBytes = fp8payload::rowBytes(params.tokenDim);
        flag.waitAndUpdate({divUp<uint32_t>(params.numTokens, params.nRanks) * params.nRanks * rowBytes,
            static_cast<uint32_t>(params.numTokens) * rowBytes, 0, 0});
    }
    else
    {
        flag.waitAndUpdate({static_cast<uint32_t>(divUp<uint32_t>(params.numTokens, params.nRanks) * params.nRanks
                                * params.tokenDim * kELTS_SIZE),
            static_cast<uint32_t>(params.numTokens * params.tokenDim * kELTS_SIZE), 0, 0});
    }
}

void twoshotAllreduceFusionOp(AllReduceFusionParams const& params)
{
    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSMVersion >= 90, "[MNNVL AllReduceTwoShot] requires SM 90 or newer.");
    int const numTokens = params.numTokens;
    int const tokenDim = params.tokenDim;
    int const numEltsPerThread = sizeof(float4) / getDTypeSize(params.dType);
    TLLM_CHECK_WITH_INFO(tokenDim % numEltsPerThread == 0, "[MNNVL AllReduceTwoShot] token_dim must be divisible by %d",
        numEltsPerThread);

    int const arNumThreads = divUp(tokenDim, numEltsPerThread);
    int const arNumBlocksPerToken = divUp(arNumThreads, 128);

    dim3 arGrid(numTokens, arNumBlocksPerToken);

    // E4M3 Lamport payload (fp8payload above); the norm kernel reads the E4M3 rows.
    bool const fp8Payload = params.fp8Payload && params.dType == tensorrt_llm::DataType::kBF16 && params.rmsNormFusion
        && tokenDim % 1024 == 0 && (params.nRanks == 2 || params.nRanks == 4 || params.nRanks == 8)
        && (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm
            || params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant
            || params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);

    cudaLaunchAttribute arAttrs[1];
    arAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    arAttrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;

    cudaLaunchConfig_t arConfig{
        .gridDim = arGrid,
        .blockDim = 128,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = arAttrs,
        .numAttrs = 1,
    };

    TLLM_LOG_DEBUG(
        "[MNNVL AllReduceTwoShot] Dispatch: grid size: (%d, %d, 1), block_size: 128", numTokens, arNumBlocksPerToken);

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T)                                                                         \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&arConfig, &twoshotAllreduceKernel<WORLD_SIZE, T>, kernelParams));
#define LAUNCH_ALLREDUCE_KERNEL_FP8(WORLD_SIZE)                                                                        \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&arConfig, &twoshotAllreduceKernelFp8<WORLD_SIZE>, kernelParams));
    auto dispatchAR = [&](auto* type_ptr) -> bool
    {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
        T* mcastPtr = reinterpret_cast<T*>(params.multicastPtr);
        T* output = reinterpret_cast<T*>(params.output);
        T const* input = reinterpret_cast<T const*>(params.input);
        MnnvlAllReduceKernelParams<T> kernelParams{output, reinterpret_cast<T*>(params.residualOut), input,
            reinterpret_cast<T const*>(params.residualIn), reinterpret_cast<T const*>(params.gamma), ucPtrs,
            reinterpret_cast<T*>(params.bufferPtrLocal), mcastPtr, params.quantOut, params.scaleOut, params.scaleFactor,
            numTokens, tokenDim, params.nRanks, params.rank, static_cast<float>(params.epsilon), params.bufferFlags,
            !params.rmsNormFusion, params.layout};
        if constexpr (std::is_same_v<T, __nv_bfloat16>)
        {
            if (fp8Payload)
            {
                switch (params.nRanks)
                {
                case 2: LAUNCH_ALLREDUCE_KERNEL_FP8(2); return true;
                case 4: LAUNCH_ALLREDUCE_KERNEL_FP8(4); return true;
                case 8: LAUNCH_ALLREDUCE_KERNEL_FP8(8); return true;
                }
                return false;
            }
        }
        switch (params.nRanks)
        {
        case 2: LAUNCH_ALLREDUCE_KERNEL(2, T); return true;
        case 4: LAUNCH_ALLREDUCE_KERNEL(4, T); return true;
        case 8: LAUNCH_ALLREDUCE_KERNEL(8, T); return true;
        case 16: LAUNCH_ALLREDUCE_KERNEL(16, T); return true;
        case 32: LAUNCH_ALLREDUCE_KERNEL(32, T); return true;
        case 64: LAUNCH_ALLREDUCE_KERNEL(64, T); return true;
        }
        return false;
    };

#undef LAUNCH_ALLREDUCE_KERNEL
#undef LAUNCH_ALLREDUCE_KERNEL_FP8

    bool launched = (params.dType == tensorrt_llm::DataType::kFLOAT && dispatchAR((float*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kBF16 && dispatchAR((__nv_bfloat16*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kHALF && dispatchAR((__nv_half*) nullptr));
    if (!launched)
    {
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch twoshotAllreduce kernel.");
    }
    // Launch the rmsnorm lamport kernel if fusion is enabled
    if (params.rmsNormFusion)
    {
        int const accessGroupSize = requiresTwoAccessScaleGroup(params.pattern) ? 2 : 1;
        auto gridConfig = adjustGridConfig<true>(numTokens, tokenDim, numEltsPerThread, accessGroupSize);
        int rnBlockSize = std::get<0>(gridConfig);
        int rnClusterSize = std::get<1>(gridConfig);
        int rnLoadsPerThread = std::get<2>(gridConfig);
        bool rnUseCGA = rnClusterSize > 1 && rnLoadsPerThread == 1;
        if (!rnUseCGA)
        {
            gridConfig = adjustGridConfig<false>(numTokens, tokenDim, numEltsPerThread, accessGroupSize);
            rnBlockSize = std::get<0>(gridConfig);
            rnClusterSize = std::get<1>(gridConfig);
            rnLoadsPerThread = std::get<2>(gridConfig);
        }

        int rnNumThreads = rnClusterSize * rnBlockSize;
        dim3 rnGrid(numTokens, rnClusterSize, 1);
        cudaLaunchConfig_t rnConfig;
        cudaLaunchAttribute rnAttrs[2];
        rnConfig.stream = params.stream;
        rnConfig.gridDim = rnGrid;
        rnConfig.blockDim = rnBlockSize;
        rnConfig.attrs = rnAttrs;
        rnAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        rnAttrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
        rnConfig.numAttrs = 1U;
        if (rnUseCGA)
        {
            rnAttrs[1].id = cudaLaunchAttributeClusterDimension;
            rnAttrs[1].val.clusterDim.x = 1;
            rnAttrs[1].val.clusterDim.y = rnClusterSize;
            rnAttrs[1].val.clusterDim.z = 1;
            rnConfig.numAttrs = 2U;
        }
        int const dimPadded = divUp(tokenDim, numEltsPerThread * rnNumThreads) * numEltsPerThread * rnNumThreads;
        int const iters = dimPadded / rnNumThreads;

        size_t const smemSize = 3 * rnBlockSize * iters * getDTypeSize(params.dType);

        TLLM_LOG_DEBUG(
            "[MNNVL AllReduceTwoShotRMSNorm] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
            "loads_per_thread: %d, "
            "threads_needed: %d",
            numTokens, rnClusterSize, rnBlockSize, rnClusterSize, rnLoadsPerThread, divUp(tokenDim, numEltsPerThread));

#define RUN_RMSNORM_KERNEL_IMPL(T, PATTERN, USE_CGA, LOADS_PER_THREAD, FP8)                                            \
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(&rmsNormLamport<T, PATTERN, USE_CGA, LOADS_PER_THREAD, FP8>,                  \
        cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize));                                                       \
    rnConfig.dynamicSmemBytes = smemSize;                                                                              \
    TLLM_CUDA_CHECK(                                                                                                   \
        cudaLaunchKernelEx(&rnConfig, &rmsNormLamport<T, PATTERN, USE_CGA, LOADS_PER_THREAD, FP8>, kernelParams));
#define RUN_RMSNORM_KERNEL(T, PATTERN, USE_CGA, LOADS_PER_THREAD)                                                      \
    RUN_RMSNORM_KERNEL_IMPL(T, PATTERN, USE_CGA, LOADS_PER_THREAD, false)

// The E4M3-payload norm (fp8Payload: BF16 and one of these three patterns).
#define DISPATCH_RMSNORM_PATTERN_FP8(T, USE_CGA, LOADS_PER_THREAD)                                                     \
    if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm)                                       \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL_IMPL(                                                                                       \
            T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm, USE_CGA, LOADS_PER_THREAD, true);                \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant)                          \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL_IMPL(                                                                                       \
            T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant, USE_CGA, LOADS_PER_THREAD, true);        \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL_IMPL(                                                                                       \
            T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, USE_CGA, LOADS_PER_THREAD, true);     \
    }

#define DISPATCH_RMSNORM_PATTERN(T, USE_CGA, LOADS_PER_THREAD)                                                         \
    if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm)                                       \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm, USE_CGA, LOADS_PER_THREAD);       \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant)                          \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(                                                                                            \
            T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant, USE_CGA, LOADS_PER_THREAD);              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant)                       \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(                                                                                            \
            T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, USE_CGA, LOADS_PER_THREAD);           \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant)                          \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            RUN_RMSNORM_KERNEL(                                                                                        \
                T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant, USE_CGA, LOADS_PER_THREAD);          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant)                       \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            RUN_RMSNORM_KERNEL(                                                                                        \
                T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, USE_CGA, LOADS_PER_THREAD);       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARRMSNorm)                                          \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(T, ar_fusion::AllReduceFusionPattern::kARRMSNorm, USE_CGA, LOADS_PER_THREAD);               \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Unsupported RMSNorm fusion pattern.");                   \
    }

        // C++ 17 does not support capturing structured bindings
        auto dispatchRN = [&, rnLoadsPerThread](auto* type_ptr)
        {
            using T = std::remove_pointer_t<decltype(type_ptr)>;
            MnnvlAllReduceKernelParams<T> kernelParams{reinterpret_cast<T*>(params.output),
                reinterpret_cast<T*>(params.residualOut), reinterpret_cast<T const*>(params.input),
                reinterpret_cast<T const*>(params.residualIn), reinterpret_cast<T const*>(params.gamma),
                reinterpret_cast<T**>(params.bufferPtrsDev), reinterpret_cast<T*>(params.bufferPtrLocal),
                reinterpret_cast<T*>(params.multicastPtr), params.quantOut, params.scaleOut, params.scaleFactor,
                numTokens, tokenDim, params.nRanks, params.rank, static_cast<float>(params.epsilon), params.bufferFlags,
                false, params.layout};
            if constexpr (std::is_same_v<T, __nv_bfloat16>)
            {
                if (fp8Payload)
                {
                    if (rnUseCGA)
                    {
                        DISPATCH_RMSNORM_PATTERN_FP8(T, true, 1);
                    }
                    else
                    {
                        switch (rnLoadsPerThread)
                        {
                        case 1: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 1); break;
                        case 2: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 2); break;
                        case 3: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 3); break;
                        case 4: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 4); break;
                        case 5: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 5); break;
                        case 6: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 6); break;
                        case 7: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 7); break;
                        case 8: DISPATCH_RMSNORM_PATTERN_FP8(T, false, 8); break;
                        default: return false;
                        }
                    }
                    return true;
                }
            }
            if (rnUseCGA)
            {
                DISPATCH_RMSNORM_PATTERN(T, true, 1);
            }
            else
            {
                switch (rnLoadsPerThread)
                {
                case 1: DISPATCH_RMSNORM_PATTERN(T, false, 1); break;
                case 2: DISPATCH_RMSNORM_PATTERN(T, false, 2); break;
                case 3: DISPATCH_RMSNORM_PATTERN(T, false, 3); break;
                case 4: DISPATCH_RMSNORM_PATTERN(T, false, 4); break;
                case 5: DISPATCH_RMSNORM_PATTERN(T, false, 5); break;
                case 6: DISPATCH_RMSNORM_PATTERN(T, false, 6); break;
                case 7: DISPATCH_RMSNORM_PATTERN(T, false, 7); break;
                case 8: DISPATCH_RMSNORM_PATTERN(T, false, 8); break;
                default: return false;
                }
            }
            return true;
        };

        launched = (params.dType == tensorrt_llm::DataType::kFLOAT && dispatchRN((float*) nullptr))
            || (params.dType == tensorrt_llm::DataType::kBF16 && dispatchRN((__nv_bfloat16*) nullptr))
            || (params.dType == tensorrt_llm::DataType::kHALF && dispatchRN((__nv_half*) nullptr));
        if (!launched)
        {
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch rmsnorm lamport kernel.");
        }
#undef RUN_RMSNORM_KERNEL
#undef RUN_RMSNORM_KERNEL_IMPL
#undef DISPATCH_RMSNORM_PATTERN
#undef DISPATCH_RMSNORM_PATTERN_FP8
    }
}

} // namespace kernels::mnnvl

TRTLLM_NAMESPACE_END
