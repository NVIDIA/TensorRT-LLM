/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

// Helper functions for lamport-based synchronization

#ifndef TRTLLM_CUDA_LAMPORT_UTILS_CUH
#define TRTLLM_CUDA_LAMPORT_UTILS_CUH

#include "tensorrt_llm/common/config.h"
#include <array>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include <cooperative_groups.h>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace common
{

constexpr uint16_t kNEGZERO_FP16 = 0x8000U;

template <typename T>
union Fp16BitCast
{
    T mFp;
    uint16_t mInt;

    constexpr Fp16BitCast()
        : mInt(0)
    {
    }

    constexpr Fp16BitCast(T val)
        : mFp(val)
    {
    }

    constexpr Fp16BitCast(uint16_t val)
        : mInt(val)
    {
    }
};

template <typename T>
static constexpr __device__ __host__ T negZero()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return -0.0F;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>)
    {
        return Fp16BitCast<T>(kNEGZERO_FP16).mFp;
    }
    else
    {
        static_assert(sizeof(T) == 0, "negativeZero not specialized for this type");
    }
    return T{}; // Never reached, but needed for compilation
}

template <typename T>
static inline __device__ bool isNegZero(T val)
{

    if constexpr (std::is_same_v<T, float>)
    {
        return val == 0.F && signbit(val);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>)
    {
        return Fp16BitCast<T>(val).mInt == kNEGZERO_FP16;
    }
    else
    {
        static_assert(sizeof(T) == 0, "isNegZero not specialized for this type");
    }
    return false; // Never reached, but needed for compilation
}

template <typename PackedType, typename T>
constexpr __device__ __host__ PackedType getPackedLamportInit()
{
    static_assert(sizeof(PackedType) % sizeof(T) == 0, "PackedType size must be divisible by T size");
    constexpr int kNumElements = sizeof(PackedType) / sizeof(T);

    union PackedT
    {
        PackedType mPacked;
        std::array<T, kNumElements> mElements;

        constexpr PackedT()
            : mElements{}
        {
            for (int i = 0; i < kNumElements; i++)
            {
                mElements[i] = negZero<T>();
            }
        }
    };

    PackedT initValue{};
    return initValue.mPacked;
}

// A helper class to get the correct base pointer for a given layout
struct LamportBufferLayout
{
    uint32_t numStages = 1;
    uint32_t bytesPerBuffer = 0;
    static constexpr uint32_t sNumLamportBuffers = 3;

    // Implicitly inlined
    [[nodiscard]] __device__ __host__ size_t getTotalBytes() const
    {
        return numStages * static_cast<size_t>(bytesPerBuffer / numStages) * sNumLamportBuffers;
    }

    // Implicitly inlined
    [[nodiscard]] __device__ __host__ void* getStagePtr(
        void* bufferBasePtr, uint32_t lamportIndex, uint32_t stageIndex) const
    {
        // Typecast to avoid warnings
        return reinterpret_cast<void*>(reinterpret_cast<char*>(bufferBasePtr)
            + static_cast<size_t>(
                (lamportIndex * numStages + stageIndex) * static_cast<size_t>(bytesPerBuffer / numStages)));
    }
};
// Current Index
// Dirty Index
// bytes_per_buffer
// Dirty num_stages
// Dirty bytes_to_clear = {stage0, stage1, stage2, stage3}  # We fix this to 4 stages
// offset_access_ptr

namespace cg = cooperative_groups;

// PackedType is the one used in kernel for Lamport buffer (LDG.128 or LDG.64)
template <typename PackedType = float4>
__device__ struct __attribute__((aligned(32))) LamportFlags
{
public:
    __device__ explicit LamportFlags(uint32_t* bufferFlags, uint32_t numStages = 1)
        : mBufferFlagsPtr(bufferFlags)
        , mFlagAccessPtr(&bufferFlags[8])
    {
        mCurBufferLayout.numStages = numStages;
        uint4 flag = reinterpret_cast<uint4*>(bufferFlags)[0];
        mCurrentIndex = flag.x;
        mDirtyIndex = flag.y;
        // Buffer size is unchanged as the flag should be coupled to each buffer
        mCurBufferLayout.bytesPerBuffer = flag.z;
        mDirtyBufferLayout.bytesPerBuffer = flag.z;
        mDirtyBufferLayout.numStages = flag.w;
        *reinterpret_cast<uint4*>(&mBytesToClear) = reinterpret_cast<uint4*>(bufferFlags)[1];
    }

    // Return the base pointer of the lamport buffer indexed by mCurrentIndex and the stageIdx
    [[nodiscard]] __device__ void* getCurLamportBuf(void* bufferBasePtr, int stageIdx = 0) const
    {
        return mCurBufferLayout.getStagePtr(bufferBasePtr, mCurrentIndex, stageIdx);
    }

    // Fill the dirty lamport buffer with the init value; Use stageIdx to select the stage to clear, -1 to clear all
    // FIXME: Current kernel may use less stages than the dirty numStages; How to guarantee the correctness?
    // CAUTION: This function requires all threads in the grid to participate and ASSUME 1D thread block layout!
    __device__ void clearDirtyLamportBuf(void* bufferBasePtr, int stageIdx = -1)
    {
        // Rasterize the threads to 1D for flexible clearing

        uint32_t globalCtaIdx = blockIdx.x * gridDim.y + blockIdx.y;
        uint32_t globalTid = globalCtaIdx * blockDim.x + threadIdx.x;
        uint32_t numThreads = gridDim.x * gridDim.y * blockDim.x;

        if (stageIdx == -1)
        {
            // Clear all stages
            for (uint32_t i = 0; i < mDirtyBufferLayout.numStages; i++)
            {
                clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[i], mDirtyIndex, i);
            }
        }
        else if (stageIdx < mDirtyBufferLayout.numStages)
        {
            clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[stageIdx], mDirtyIndex, stageIdx);
        }
    }

    __device__ void ctaArrive()
    {
        int tid{0};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

        cg::cluster_group cluster = cg::this_cluster();
        // We update the atomic counter per cluster
        tid = cluster.thread_rank();
        cluster.sync();
#else
        tid = threadIdx.x;
        __syncthreads();
#endif
        if (tid == 0)
        {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
            asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
            asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#else
            atomicAdd(mFlagAccessPtr, 1);
#endif
        }
    }

    __device__ void waitAndUpdate(uint4 bytesToClearPerStage)
    {
        bool isLastCtaT0{false};
        int targetCount{0};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cg::grid_group grid = cg::this_grid();
        // Use the first thread instead of the last thread as the last thread may exit early
        isLastCtaT0 = grid.thread_rank() == 0;
        targetCount = grid.num_clusters();
#else
        isLastCtaT0 = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0;
        targetCount = gridDim.x * gridDim.y;
#endif
        if (isLastCtaT0)
        {
            uint4* flagPtr = reinterpret_cast<uint4*>(mBufferFlagsPtr);
            while (*reinterpret_cast<uint32_t volatile*>(mFlagAccessPtr) < targetCount)
            {
            }
            // 'Current' becomes 'Dirty'
            flagPtr[0] = {(mCurrentIndex + 1) % 3, // Current index
                mCurrentIndex,                     // Dirty index
                mCurBufferLayout.bytesPerBuffer,   // Buffer size
                mCurBufferLayout.numStages};       // Dirty - Number of stages
            flagPtr[1] = bytesToClearPerStage;
            *mFlagAccessPtr = 0;
        }
    }

private:
    uint32_t* mBufferFlagsPtr;
    uint32_t* mFlagAccessPtr;

    uint32_t mCurrentIndex, mDirtyIndex;
    // So that we can access it with uint4
    alignas(16) std::array<uint32_t, 4> mBytesToClear;
    LamportBufferLayout mCurBufferLayout, mDirtyBufferLayout;

    inline __device__ void clearPackedBuf(void* bufferBasePtr, uint32_t globalTid, uint32_t numThreads,
        uint32_t bytesToClear, uint8_t dirtyIndex, uint8_t stageIdx)
    {
        // Round up to the float4 boundary
        // For the same reason that the divUp is shadowed, we have to define it again here.
        uint32_t clearBoundary = (bytesToClear + sizeof(PackedType) - 1) / sizeof(PackedType);
        for (uint32_t packedIdx = globalTid; packedIdx < clearBoundary; packedIdx += numThreads)
        {
            reinterpret_cast<PackedType*>(
                mDirtyBufferLayout.getStagePtr(bufferBasePtr, dirtyIndex, stageIdx))[packedIdx]
                = getPackedLamportInit<PackedType, float>();
        }
    }
};

} // namespace common

TRTLLM_NAMESPACE_END
#endif // TRTLLM_CUDA_LAMPORT_UTILS_CUH
