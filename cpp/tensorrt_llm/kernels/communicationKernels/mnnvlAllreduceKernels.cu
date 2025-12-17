/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_pipeline.h>
#include <tuple>
#include <type_traits>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mnnvl
{

using tensorrt_llm::common::isNegZero;
using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::cuda_cast;
using tensorrt_llm::common::getMultiProcessorCount;
using tensorrt_llm::common::getDTypeSize;

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

template <typename PackedType>
inline __device__ PackedType loadPackedVolatile(void const* ptr)
{
    static_assert(sizeof(PackedType) == 0, "Not implemented");
    return PackedType{};
}

template <>
inline __device__ float4 loadPackedVolatile<float4>(void const* ptr)
{
    float4 returnValue;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(returnValue.x), "=f"(returnValue.y), "=f"(returnValue.z), "=f"(returnValue.w)
                 : "l"(ptr));
    return returnValue;
}

template <>
inline __device__ float2 loadPackedVolatile<float2>(void const* ptr)
{
    float2 returnValue;
    asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(returnValue.x), "=f"(returnValue.y) : "l"(ptr));
    return returnValue;
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

// A helper function to tune the grid configuration for fused oneshot and rmsnorm kernels
// Return (block_size, cluster_size, loads_per_thread)
std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread)
{
    static int SM = tensorrt_llm::common::getSMVersion();

    int clusterSize = SM >= 90 ? 8 : 1;
    int blockSize = 128;
    // ========================== Adjust the grid configuration ==========================
    int threadsNeeded = divUp(dim, eltsPerThread);
    int loadsPerThread = 1;

    blockSize = divUp(threadsNeeded, clusterSize);
    if (clusterSize > 1)
    {
        while (threadsNeeded % clusterSize != 0 && clusterSize > 1)
        {
            clusterSize /= 2;
        }
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
    }

    // Trying to scale up use multiple loads or CGA
    while (blockSize > 1024)
    {
        // Scale up with CGA if supported
        if (SM >= 90)
        {
            if (clusterSize < 8)
            {
                clusterSize = clusterSize << 1;
            }
            else
            {
                break;
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
    return {blockSize, clusterSize, loadsPerThread};
}

} // namespace detail

using detail::PackedVec;
using detail::loadPacked;
using detail::loadPackedVolatile;
using detail::blockReduceSum;
using detail::divUp;
using detail::copyF4;

template <uint8_t WorldSize, typename T, bool RMSNormFusion = false, typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshotAllreduceFusionKernel(T* outputPtr, T* prenormedPtr, T const* shardPtr,
    T const* residualInPtr, T const* gammaPtr, T** inputPtrs, T* mcastPtr, int const numTokens, int const tokenDim,
    float epsilon, int const rank, uint32_t* bufferFlags)
{
    constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
    constexpr uint32_t kELT_SIZE = sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int packedIdx = cluster.thread_rank();
    int token = blockIdx.x;
    int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

    cudaGridDependencySynchronize();
#else
    int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;
#endif

    // We only use 1 stage for the oneshot allreduce
    LamportFlags<PackedType> flag(bufferFlags, 1);
    T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, 0));
    T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], 0));

    if (packedIdx * kELTS_PER_THREAD >= tokenDim)
    {
        flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
        return;
    }

    // ==================== Broadcast tokens to each rank =============================
    PackedVec<PackedType, T> val;
    val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        if (isNegZero(val.elements[i]))
            val.elements[i] = cuda_cast<T, float>(0.f);
    }

    reinterpret_cast<PackedType*>(&stagePtrMcast[token * tokenDim * WorldSize + rank * tokenDim])[packedIdx]
        = val.packed;

    flag.ctaArrive();
    // ======================= Lamport Sync and clear the output buffer from previous iteration
    // =============================
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);

    PackedVec<PackedType, float> valuesLamport[WorldSize];
    while (1)
    {
        bool valid = true;
#pragma unroll
        for (int r = 0; r < WorldSize; r++)
        {
            valuesLamport[r].packed = loadPackedVolatile<PackedType>(
                &stagePtrLocal[token * tokenDim * WorldSize + r * tokenDim + packedIdx * kELTS_PER_THREAD]);

#pragma unroll
            for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++)
            {
                valid &= !isNegZero(valuesLamport[r].elements[i]);
            }
        }
        if (valid)
        {
            break;
        }
    }

    auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
    // ======================= Reduction =============================
    float accum[kELTS_PER_THREAD];
    PackedVec<PackedType, T> packedAccum;

#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] = cuda_cast<float, T>(values[0].elements[i]);
    }

#pragma unroll
    for (int r = 1; r < WorldSize; r++)
    {
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            accum[i] += cuda_cast<float, T>(values[r].elements[i]);
        }
    }

#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if constexpr (RMSNormFusion)
    {
        // =============================== Residual ===============================
        PackedVec<PackedType, T> residualIn;
        residualIn.packed = *reinterpret_cast<PackedType const*>(&residualInPtr[threadOffset]);
        packedAccum += residualIn;
        *reinterpret_cast<PackedType*>(&prenormedPtr[threadOffset]) = packedAccum.packed;
        // =============================== Rmsnorm ================================
        PackedVec<PackedType, T> gamma;
        gamma.packed = *reinterpret_cast<PackedType const*>(&gammaPtr[packedIdx * kELTS_PER_THREAD]);

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
            if (threadIdx.x < numBlocks)
            {
                cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
            }
            // cluster.sync();
            cluster.barrier_wait(cluster.barrier_arrive());
            for (int i = 0; i < numBlocks; ++i)
            {
                fullSum += sharedVal[i];
            }
        }
#endif
        float rcpRms = rsqrtf(fullSum / tokenDim + epsilon);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(
                cuda_cast<float, T>(packedAccum.elements[i]) * rcpRms * cuda_cast<float, T>(gamma.elements[i]));
        }
    }
    reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = packedAccum.packed;
    flag.waitAndUpdate({static_cast<uint32_t>(numTokens * tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

using detail::adjustGridConfig;

void oneshotAllreduceFusionOp(AllReduceFusionParams const& params)
{

    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    int const numTokens = params.numTokens;
    int const tokenDim = params.tokenDim;
    int const eltsPerThread = sizeof(float4) / getDTypeSize(params.dType);

    auto [blockSize, clusterSize, loadsPerThread] = adjustGridConfig(numTokens, tokenDim, eltsPerThread);
    dim3 grid(numTokens, clusterSize, 1);

    TLLM_LOG_DEBUG(
        "[MNNVL AllReduceOneShot] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
        "loads_per_thread: %d, "
        "threads_needed: %d",
        numTokens, clusterSize, blockSize, clusterSize, loadsPerThread, divUp(tokenDim, eltsPerThread));

    TLLM_CHECK_WITH_INFO(blockSize <= 1024 && loadsPerThread == 1,
        "Hidden Dimension %d exceeds the maximum supported hidden dimension (%d)", tokenDim,
        1024 * (kSMVersion >= 90 ? 8 : 1) * eltsPerThread);

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
        .numAttrs = kSMVersion >= 90 ? 2U : 1U,
    };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, RMSNORM)                                                                \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &oneshotAllreduceFusionKernel<WORLD_SIZE, T, RMSNORM>, output,         \
        residualOut, input, residualIn, gamma, ucPtrs, mcPtr, numTokens, tokenDim, static_cast<float>(params.epsilon), \
        params.rank, params.bufferFlags));
#define DISPATCH_ALLREDUCE_KERNEL(WORLD_SIZE, T)                                                                       \
    if (params.rmsNormFusion)                                                                                          \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, true);                                                                  \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, false);                                                                 \
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

        switch (params.nRanks)
        {
            // FIXME: Do we need other world sizes?
        case 2: DISPATCH_ALLREDUCE_KERNEL(2, T); return true;
        case 4: DISPATCH_ALLREDUCE_KERNEL(4, T); return true;
        case 8: DISPATCH_ALLREDUCE_KERNEL(8, T); return true;
        case 16: DISPATCH_ALLREDUCE_KERNEL(16, T); return true;
        case 32: DISPATCH_ALLREDUCE_KERNEL(32, T); return true;
        case 64: DISPATCH_ALLREDUCE_KERNEL(64, T); return true;
        }
        return false;
    };
#undef LAUNCH_ALLREDUCE_KERNEL
#undef DISPATCH_ALLREDUCE_KERNEL
    bool launched = (params.dType == nvinfer1::DataType::kBF16 && dispatchImpl((__nv_bfloat16*) nullptr))
        || (params.dType == nvinfer1::DataType::kFLOAT && dispatchImpl((float*) nullptr))
        || (params.dType == nvinfer1::DataType::kHALF && dispatchImpl((__nv_half*) nullptr));
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
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(T* outputPtr, T const* shardPtr, T** inputPtrs,
    T* mcastPtr, uint32_t const numTokens, uint32_t const tokenDim, uint32_t const rank, uint32_t* bufferFlags,
    bool const wait_for_results)
{
    constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
    constexpr uint32_t kELT_SIZE = sizeof(T);

    int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

    int destRank = token % WorldSize;
    int destTokenOffset = token / WorldSize;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    LamportFlags<PackedType> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);

    T* scatterBufLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER));
    T* scatterBufDest = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
    T* broadcastBufW = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, MNNVLTwoShotStage::BROADCAST));
    T* broadcastBufR = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    // Make sure the clear function is called before OOB thread exits
    if (packedIdx * kELTS_PER_THREAD >= tokenDim)
    {
        flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
        return;
    }

    // =============================== Scatter ===============================

    // Load vectorized data
    PackedVec<PackedType, T> val;
    val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        if (isNegZero(val.elements[i]))
        {
            val.elements[i] = cuda_cast<T, float>(0.F);
        }
    }

    // Store vectorized data
    reinterpret_cast<PackedType*>(&scatterBufDest[destTokenOffset * tokenDim * WorldSize + rank * tokenDim])[packedIdx]
        = val.packed;

    flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER);

    // =============================== Reduction and Broadcast ===============================

    if ((token % WorldSize) == rank)
    {
        int localToken = token / WorldSize;
        float accum[kELTS_PER_THREAD] = {0.F};

        // Use float as we only check each float value for validity
        PackedVec<PackedType, float> valuesLamport[WorldSize];
        while (1)
        {
            bool valid = true;
#pragma unroll
            for (int r = 0; r < WorldSize; r++)
            {
                valuesLamport[r].packed = loadPackedVolatile<PackedType>(
                    &scatterBufLocal[localToken * tokenDim * WorldSize + r * tokenDim + packedIdx * kELTS_PER_THREAD]);

                // Check validity across all elements
#pragma unroll
                for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++)
                {
                    valid &= !isNegZero(valuesLamport[r].elements[i]);
                }
            }
            if (valid)
            {
                break;
            }
        }

        // Now we view it as the value for reduction
        auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
#pragma unroll
        for (int r = 0; r < WorldSize; r++)
        {

#pragma unroll
            for (int i = 0; i < kELTS_PER_THREAD; i++)
            {
                accum[i] += cuda_cast<float, T>(values[r].elements[i]);
            }
        }

        // Store vectorized result
        PackedVec<PackedType, T> packedAccum;
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
        }
        reinterpret_cast<PackedType*>(&broadcastBufW[token * tokenDim])[packedIdx] = packedAccum.packed;
    }

    flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST);

    // Optionally wait for results if the next layer isn't doing the Lamport check
    if (wait_for_results)
    {
        // Update the atomic counter to indicate the block has read the offsets
        flag.ctaArrive();

        PackedVec<PackedType, float> valLamport;
        valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
        while (isNegZero(valLamport.elements[0]))
        {
            valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
        }
        if (outputPtr)
        {
            reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = valLamport.packed;
        }

        // Update the buffer flags
        flag.waitAndUpdate({static_cast<uint32_t>(divUp<uint32_t>(numTokens, WorldSize) * WorldSize * tokenDim
                                * kELT_SIZE),                        // Clear Size for scatter stage
            static_cast<uint32_t>(numTokens * tokenDim * kELT_SIZE), // Clear Size for broadcast stage
            0, 0});
        // If not wait for results, we will rely on the following kernel to update the buffer
    }
}

// This kernel works performant when loads_per_thread is 1.
// For this mode, we are able to support up to 1024 (threads) x 8 (elements) = 8192 hidden dimension.
// There are two options for further scaling up:
//      1. Use CGA if supported. It expands the hidden dimension to 8k x 8 = 64k.
//      2. Set loads_per_thread >1. Which can be used if CGA is not supported. Note that this will be limited by the
//      shared memory size and register count.
template <typename T_IN, typename T_OUT, int LoadsPerThread = 1>
__global__ __launch_bounds__(1024) void rmsNormLamport(T_IN* outputPreNorm, T_OUT* outputNorm, T_IN* bufferInput,
    T_IN const* gamma, float epsilon, T_IN const* residual, uint32_t numTokens, uint32_t dim, uint32_t worldSize,
    uint32_t* bufferFlags)
{
    static_assert(std::is_same_v<T_IN, T_OUT>, "T_IN and T_OUT must be the same type");
    static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T_IN);

    uint32_t const token = blockIdx.x;
    uint32_t const blockSize = blockDim.x;
    uint32_t const threadOffset = threadIdx.x;

    uint32_t numThreads = blockSize;
    uint32_t clusterSize = 1;
    uint32_t blockOffset = 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    numThreads = cluster.num_threads();
    clusterSize = cluster.num_blocks();
    blockOffset = cluster.block_rank();
#endif
    uint32_t const dimPadded = divUp(dim, kELTS_PER_LOAD * numThreads) * kELTS_PER_LOAD * numThreads;
    uint32_t const elemsPerThread = dimPadded / numThreads;
    uint32_t const loadStride = blockSize;

    extern __shared__ uint8_t smem[];
    float rInput[LoadsPerThread * kELTS_PER_LOAD];
    uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

    uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T_IN);
    T_IN* smemInput = (T_IN*) &smem[0];
    T_IN* smemResidual = (T_IN*) &smem[smemBufferSize];
    T_IN* smemGamma = (T_IN*) &smem[2 * smemBufferSize];

    LamportFlags<float4> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
    T_IN* input = reinterpret_cast<T_IN*>(
        flag.getCurLamportBuf(reinterpret_cast<void*>(bufferInput), MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    // The offset that current thread should load from. Note that the hidden dimension is split by CGA size and each
    // block loads a contiguous chunk;
    // The size of chunk that each block processes
    uint32_t const blockChunkSize = divUp(dim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
    uint32_t const blockLoadOffset = token * dim + blockOffset * blockChunkSize;

#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++)
    {
        // Each block load a contiguous chunk of tokens
        uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        offsets[i] = blockLoadOffset + threadLoadOffset;
    }

#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++)
    {
        uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < dim)
        {
            copyF4(&smemResidual[threadLoadOffset], &residual[blockLoadOffset + threadLoadOffset]);
        }
    }
    __pipeline_commit();
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++)
    {
        uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < dim)
        {
            copyF4(&smemGamma[threadLoadOffset], &gamma[blockOffset * blockChunkSize + threadLoadOffset]);
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

            if (blockOffset * blockChunkSize + threadLoadOffset < dim)
            {

                float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
                float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

                float4 value = loadPackedVolatile<float4>(src4);
                // Assume that the 16B were written atomically, so we only need to check one value
                valid &= !isNegZero(value.x);
                *dst4 = value;
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
        if (blockOffset * blockChunkSize + threadLoadOffset < dim)
        {
            PackedVec<float4, T_IN> inp{.packed = loadPacked<float4>(&smemInput[threadLoadOffset])};
            PackedVec<float4, T_IN> res{.packed = loadPacked<float4>(&smemResidual[threadLoadOffset])};

            PackedVec<float4, T_IN> inp_plus_res = inp + res;
#pragma unroll
            for (int j = 0; j < kELTS_PER_LOAD; j++)
            {
                rInput[i * kELTS_PER_LOAD + j] = cuda_cast<float, T_IN>(inp_plus_res.elements[j]);
                threadSum += cuda_cast<float, T_IN>(inp_plus_res.elements[j] * inp_plus_res.elements[j]);
            }

            *reinterpret_cast<float4*>(&outputPreNorm[blockLoadOffset + threadLoadOffset]) = inp_plus_res.packed;
        }
    }

    __pipeline_wait_prior(0);

    float blockSum = blockReduceSum<float, true>(threadSum);

    float fullSum = blockSum;
    // Use CGA Reduction if supported
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    __shared__ float sharedVal[8];
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1)
    {
        fullSum = 0.F;
        // Need to reduce over the entire cluster
        int const blockRank = cluster.block_rank();
        if (threadIdx.x < numBlocks)
        {
            cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
        }
        // cluster.sync();
        cluster.barrier_wait(cluster.barrier_arrive());
        for (int i = 0; i < numBlocks; ++i)
        {
            fullSum += sharedVal[i];
        }
    }
#endif

    float rcpRms = rsqrtf(fullSum / dim + epsilon);

#pragma unroll
    for (int i = 0; i < LoadsPerThread; i++)
    {
        PackedVec<float4, T_OUT> r_out;
        uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < dim)
        {
            PackedVec<float4, T_IN> gamma = {.packed = loadPacked<float4>(&smemGamma[threadLoadOffset])};

#pragma unroll
            for (uint32_t j = 0; j < kELTS_PER_LOAD; j++)
            {
                r_out.elements[j] = cuda_cast<T_OUT, float>(
                    cuda_cast<float, T_IN>(gamma.elements[j]) * rInput[i * kELTS_PER_LOAD + j] * rcpRms);
            }

            *reinterpret_cast<float4*>(&outputNorm[blockLoadOffset + threadLoadOffset]) = r_out.packed;
        }
    }
    constexpr int kELTS_SIZE = sizeof(T_IN);

    // Issue ACQBLK at the end. Assuming preceding kernel will not modify the buffer_flags.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // Update the buffer pointers
    flag.waitAndUpdate({static_cast<uint32_t>(divUp<uint32_t>(numTokens, worldSize) * worldSize * dim * kELTS_SIZE),
        static_cast<uint32_t>(numTokens * dim * kELTS_SIZE), 0, 0});
}

void twoshotAllreduceFusionOp(AllReduceFusionParams const& params)
{
    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    int const numTokens = params.numTokens;
    int const tokenDim = params.tokenDim;
    int const numEltsPerThread = sizeof(float4) / getDTypeSize(params.dType);
    TLLM_CHECK_WITH_INFO(tokenDim % numEltsPerThread == 0, "[MNNVL AllReduceTwoShot] token_dim must be divisible by %d",
        numEltsPerThread);

    int const arNumThreads = divUp(tokenDim, numEltsPerThread);
    int const arNumBlocksPerToken = divUp(arNumThreads, 128);

    dim3 arGrid(numTokens, arNumBlocksPerToken);

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
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&arConfig, &twoshotAllreduceKernel<WORLD_SIZE, T>, output, input, ucPtrs,       \
        mcastPtr, numTokens, tokenDim, params.rank, params.bufferFlags, (!params.rmsNormFusion)));
    auto dispatchAR = [&](auto* type_ptr) -> bool
    {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
        T* mcastPtr = reinterpret_cast<T*>(params.multicastPtr);
        T* output = reinterpret_cast<T*>(params.output);
        T const* input = reinterpret_cast<T const*>(params.input);
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

    bool launched = (params.dType == nvinfer1::DataType::kFLOAT && dispatchAR((float*) nullptr))
        || (params.dType == nvinfer1::DataType::kBF16 && dispatchAR((__nv_bfloat16*) nullptr))
        || (params.dType == nvinfer1::DataType::kHALF && dispatchAR((__nv_half*) nullptr));
    if (!launched)
    {
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch twoshotAllreduce kernel.");
    }
    // Launch the rmsnorm lamport kernel if fusion is enabled
    if (params.rmsNormFusion)
    {
        auto gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread);
        int rnBlockSize = std::get<0>(gridConfig);
        int rnClusterSize = std::get<1>(gridConfig);
        int rnLoadsPerThread = std::get<2>(gridConfig);

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
        rnAttrs[1].id = cudaLaunchAttributeClusterDimension;
        rnAttrs[1].val.clusterDim.x = 1;
        rnAttrs[1].val.clusterDim.y = rnClusterSize;
        rnAttrs[1].val.clusterDim.z = 1;
        rnConfig.numAttrs = (kSMVersion >= 90) ? 2U : 1U;

        bool const rnUseCGA = kSMVersion >= 90 && rnClusterSize > 1;
        int const dimPadded = divUp(tokenDim, numEltsPerThread * rnNumThreads) * numEltsPerThread * rnNumThreads;
        int const iters = dimPadded / rnNumThreads;

        size_t const smemSize = 3 * rnBlockSize * iters * getDTypeSize(params.dType);

        TLLM_LOG_DEBUG(
            "[MNNVL AllReduceTwoShotRMSNorm] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
            "loads_per_thread: %d, "
            "threads_needed: %d",
            numTokens, rnClusterSize, rnBlockSize, rnClusterSize, rnLoadsPerThread, divUp(tokenDim, numEltsPerThread));

#define RUN_RMSNORM_KERNEL(T_IN, T_OUT, LOADS_PER_THREAD)                                                              \
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(                                                                              \
        &rmsNormLamport<T_IN, T_OUT, LOADS_PER_THREAD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize));       \
    rnConfig.dynamicSmemBytes = smemSize;                                                                              \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&rnConfig, &rmsNormLamport<T_IN, T_OUT, LOADS_PER_THREAD>, residualOut, output, \
        bufferInput, gamma, static_cast<float>(params.epsilon), residualIn, numTokens, tokenDim, params.nRanks,        \
        params.bufferFlags));

        // C++ 17 does not support capturing structured bindings
        auto dispatchRN = [&, rnLoadsPerThread](auto* type_ptr)
        {
            using T_IN = std::remove_pointer_t<decltype(type_ptr)>;
            using T_OUT = T_IN;
            T_OUT* residualOut = reinterpret_cast<T_OUT*>(params.residualOut);
            T_OUT* output = reinterpret_cast<T_OUT*>(params.output);
            T_IN* bufferInput = reinterpret_cast<T_IN*>(params.bufferPtrLocal);
            T_IN const* gamma = reinterpret_cast<T_IN const*>(params.gamma);
            T_IN const* residualIn = reinterpret_cast<T_IN const*>(params.residualIn);
            if (rnUseCGA)
            {
                RUN_RMSNORM_KERNEL(T_IN, T_OUT, 1);
            }
            else
            {
                switch (rnLoadsPerThread)
                {
                case 1: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 1); break;
                case 2: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 2); break;
                case 3: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 3); break;
                case 4: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 4); break;
                case 5: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 5); break;
                case 6: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 6); break;
                case 7: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 7); break;
                case 8: RUN_RMSNORM_KERNEL(T_IN, T_OUT, 8); break;
                default: return false;
                }
            }
            return true;
        };

        launched = (params.dType == nvinfer1::DataType::kFLOAT && dispatchRN((float*) nullptr))
            || (params.dType == nvinfer1::DataType::kBF16 && dispatchRN((__nv_bfloat16*) nullptr))
            || (params.dType == nvinfer1::DataType::kHALF && dispatchRN((__nv_half*) nullptr));
        if (!launched)
        {
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch rmsnorm lamport kernel.");
        }
#undef RUN_RMSNORM_KERNEL
    }
}

} // namespace kernels::mnnvl

TRTLLM_NAMESPACE_END
