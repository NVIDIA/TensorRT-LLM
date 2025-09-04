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

#pragma once

#include <array>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include <cooperative_groups.h>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"

namespace tensorrt_llm::common
{

constexpr uint16_t NEGZERO_FP16 = 0x8000U;

template <typename T>
union fp16_bit_cast
{
    T fp;
    uint16_t i;

    constexpr fp16_bit_cast()
        : i(0)
    {
    }

    constexpr fp16_bit_cast(T val)
        : fp(val)
    {
    }

    constexpr fp16_bit_cast(uint16_t val)
        : i(val)
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
        return fp16_bit_cast<T>(NEGZERO_FP16).fp;
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
        return val == 0.f && signbit(val);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>)
    {
        return fp16_bit_cast<T>(val).i == NEGZERO_FP16;
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
    constexpr int num_elements = sizeof(PackedType) / sizeof(T);

    union packed_t
    {
        PackedType packed;
        std::array<T, num_elements> elements;

        constexpr packed_t()
            : elements{}
        {
            for (int i = 0; i < num_elements; i++)
            {
                elements[i] = negZero<T>();
            }
        }
    };

    packed_t init_value{};
    return init_value.packed;
}

// A helper class to get the correct base pointer for a given layout
struct LamportBufferLayout
{
    uint32_t num_stages = 1;
    uint32_t bytes_per_buffer = 0;
    static constexpr uint32_t num_lamport_buffers = 3;

    // Implicitly inlined
    [[nodiscard]] __device__ __host__ size_t getTotalBytes() const
    {
        return num_stages * (bytes_per_buffer / num_stages) * num_lamport_buffers;
    }

    // Implicitly inlined
    [[nodiscard]] __device__ __host__ void* getStagePtr(
        void* buffer_base_ptr, uint32_t lamport_index, uint32_t stage_index) const
    {
        // Typecast to avoid warnings
        return reinterpret_cast<void*>(reinterpret_cast<char*>(buffer_base_ptr)
            + ((lamport_index * num_stages + stage_index) * (bytes_per_buffer / num_stages)));
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
template <typename PackedType = float4, bool USE_CGA = false>
__device__ struct __attribute__((aligned(32))) LamportFlags
{
public:
    __device__ explicit LamportFlags(uint32_t* buffer_flags, uint32_t num_stages = 1)
        : mBufferFlagsPtr(buffer_flags)
        , mFlagAccessPtr(&buffer_flags[8])
    {
        mCurBufferLayout.num_stages = num_stages;
        uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
        mCurrentIndex = flag.x;
        mDirtyIndex = flag.y;
        // Buffer size is unchanged as the flag should be coupled to each buffer
        mCurBufferLayout.bytes_per_buffer = flag.z;
        mDirtyBufferLayout.bytes_per_buffer = flag.z;
        mDirtyBufferLayout.num_stages = flag.w;
        *reinterpret_cast<uint4*>(&mBytesToClear) = reinterpret_cast<uint4*>(buffer_flags)[1];
    }

    // Return the base pointer of the lamport buffer indexed by mCurrentIndex and the stage_idx
    [[nodiscard]] __device__ void* getCurLamportBuf(void* buffer_base_ptr, int stage_idx = 0) const
    {
        return mCurBufferLayout.getStagePtr(buffer_base_ptr, mCurrentIndex, stage_idx);
    }

    // Fill the dirty lamport buffer with the init value; Use stage_idx to select the stage to clear, -1 to clear all
    // FIXME: Current kernel may use less stages than the dirty num_stages; How to guarantee the correctness?
    // CAUTION: This function requires all threads in the grid to participate and ASSUME 1D thread block layout!
    __device__ void clearDirtyLamportBuf(void* buffer_base_ptr, int stage_idx = -1)
    {
        // Rasterize the threads to 1D for flexible clearing

        uint32_t global_cta_idx = blockIdx.x * gridDim.y + blockIdx.y;
        uint32_t global_tid = global_cta_idx * blockDim.x + threadIdx.x;
        uint32_t num_threads = gridDim.x * gridDim.y * blockDim.x;

        if (stage_idx == -1)
        {
            // Clear all stages
            for (uint32_t i = 0; i < mDirtyBufferLayout.num_stages; i++)
            {
                clear_packed_buf(buffer_base_ptr, global_tid, num_threads, mBytesToClear[i], mDirtyIndex, i);
            }
        }
        else if (stage_idx < mDirtyBufferLayout.num_stages)
        {
            clear_packed_buf(
                buffer_base_ptr, global_tid, num_threads, mBytesToClear[stage_idx], mDirtyIndex, stage_idx);
        }
    }

    __device__ void cta_arrive()
    {
        int tid{0};
        if constexpr (USE_CGA)
        {
            cg::cluster_group cluster = cg::this_cluster();
            cg::grid_group grid = cg::this_grid();
            // We update the atomic counter per cluster
            tid = cluster.thread_rank();
            cluster.sync();
        }
        else
        {
            tid = threadIdx.x;
            __syncthreads();
        }
        if (tid == 0)
        {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
            asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            asm volatile("red.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#else
            atomicAdd(mFlagAccessPtr, 1);
#endif
        }
    }

    __device__ void wait_and_update(uint4 bytes_to_clear_per_stage)
    {
        bool is_last_cta_t0{false};
        int target_count{0};
        if constexpr (USE_CGA)
        {
            cg::grid_group grid = cg::this_grid();
            // Use the first thread instead of the last thread as the last thread may exit early
            is_last_cta_t0 = grid.thread_rank() == 0;
            target_count = grid.num_clusters();
        }
        else
        {
            is_last_cta_t0 = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0;
            target_count = gridDim.x * gridDim.y;
        }
        if (is_last_cta_t0)
        {
            uint4* flag_ptr = reinterpret_cast<uint4*>(mBufferFlagsPtr);
            while (*reinterpret_cast<uint32_t volatile*>(mFlagAccessPtr) < target_count)
            {
            }
            // 'Current' becomes 'Dirty'
            flag_ptr[0] = {(mCurrentIndex + 1) % 3, // Current index
                mCurrentIndex,                      // Dirty index
                mCurBufferLayout.bytes_per_buffer,  // Buffer size
                mCurBufferLayout.num_stages};       // Dirty - Number of stages
            flag_ptr[1] = bytes_to_clear_per_stage;
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

    inline __device__ void clear_packed_buf(void* buffer_base_ptr, uint32_t global_tid, uint32_t num_threads,
        uint32_t bytes_to_clear, uint8_t dirty_index, uint8_t stage_idx)
    {
        // Round up to the float4 boundary
        // For the same reason that the divUp is shadowed, we have to define it again here.
        uint32_t clear_boundry = (bytes_to_clear + sizeof(PackedType) - 1) / sizeof(PackedType);
        for (uint32_t packed_idx = global_tid; packed_idx < clear_boundry; packed_idx += num_threads)
        {
            reinterpret_cast<PackedType*>(
                mDirtyBufferLayout.getStagePtr(buffer_base_ptr, dirty_index, stage_idx))[packed_idx]
                = getPackedLamportInit<PackedType, float>();
        }
    }
};

} // namespace tensorrt_llm::common
