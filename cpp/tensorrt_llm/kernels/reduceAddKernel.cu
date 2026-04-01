/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "reduceAddKernel.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tensorrt_llm
{
namespace kernels
{

// Helper traits for type-specific operations
template <typename T>
struct TypeTraits
{
};

template <>
struct TypeTraits<half>
{
    using Vec2Type = half2;
    using Vec8Type = uint4; // 128 bits = 8 * fp16

    __device__ __forceinline__ static half zero()
    {
        return __float2half(0.0f);
    }

    __device__ __forceinline__ static half2 add2(half2 a, half2 b)
    {
        return __hadd2(a, b);
    }

    __device__ __forceinline__ static half add(half a, half b)
    {
        return __hadd(a, b);
    }
};

template <>
struct TypeTraits<__nv_bfloat16>
{
    using Vec2Type = __nv_bfloat162;
    using Vec8Type = uint4; // 128 bits = 8 * bf16

    __device__ __forceinline__ static __nv_bfloat16 zero()
    {
        return __float2bfloat16(0.0f);
    }

    __device__ __forceinline__ static __nv_bfloat162 add2(__nv_bfloat162 a, __nv_bfloat162 b)
    {
        return __hadd2(a, b);
    }

    __device__ __forceinline__ static __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b)
    {
        return __hadd(a, b);
    }
};

// Vectorized add for 8 elements (128-bit)
template <typename T>
__device__ __forceinline__ typename TypeTraits<T>::Vec8Type add_vec8(
    typename TypeTraits<T>::Vec8Type a, typename TypeTraits<T>::Vec8Type b)
{
    typename TypeTraits<T>::Vec8Type result;
    using Vec2Type = typename TypeTraits<T>::Vec2Type;

    Vec2Type* a_ptr = reinterpret_cast<Vec2Type*>(&a);
    Vec2Type* b_ptr = reinterpret_cast<Vec2Type*>(&b);
    Vec2Type* r_ptr = reinterpret_cast<Vec2Type*>(&result);

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        r_ptr[i] = TypeTraits<T>::add2(a_ptr[i], b_ptr[i]);
    }
    return result;
}

// Helper functions for type conversion to/from float
__device__ __forceinline__ float to_float(half val)
{
    return __half2float(val);
}

__device__ __forceinline__ float to_float(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

__device__ __forceinline__ half from_float_to_half(float val)
{
    return __float2half(val);
}

__device__ __forceinline__ __nv_bfloat16 from_float_to_bfloat16(float val)
{
    return __float2bfloat16(val);
}

// Persistent kernel with loop over tokens
// Uses compile-time topk for complete loop unrolling and optimization
template <typename T, int TOPK, int VEC_SIZE = 8>
__global__ void reduceAddKernel(T const* __restrict__ input, T const* __restrict__ residual, T* __restrict__ output,
    int32_t num_tokens, int32_t hidden_size)
{
    int32_t const tid = threadIdx.x;
    int32_t const threads_per_block = blockDim.x;
    int32_t const num_blocks = gridDim.x;

    using VecType = typename TypeTraits<T>::Vec8Type;
    int32_t const num_vec_elements = hidden_size / VEC_SIZE;
    int32_t const remaining_start = num_vec_elements * VEC_SIZE;

    // Persistent kernel: each block processes multiple tokens
    for (int32_t token_idx = blockIdx.x; token_idx < num_tokens; token_idx += num_blocks)
    {
        // Pre-calculate base pointers to reduce address calculation overhead
        // TOPK is compile-time constant, so topk * hidden_size becomes a simple shift/add
        T const* input_token_base = input + token_idx * TOPK * hidden_size;
        T const* residual_token_base = residual + token_idx * hidden_size;
        T* output_token_base = output + token_idx * hidden_size;

        // Vectorized processing: 128-bit loads (8 elements) + FP32 accumulation
        for (int32_t vec_idx = tid; vec_idx < num_vec_elements; vec_idx += threads_per_block)
        {
            int32_t const h_offset = vec_idx * VEC_SIZE;

            // FP32 accumulator array for precision
            float acc[VEC_SIZE];
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i)
            {
                acc[i] = 0.0f;
            }

            // Reduce across TOPK: compile-time constant allows complete unrolling
            // Compiler can optimize away all multiplications and generate optimal code
            T const* input_ptr_base = input_token_base + h_offset;

#pragma unroll
            for (int32_t k = 0; k < TOPK; ++k)
            {
                // With TOPK as template parameter, compiler generates:
                // load from (base + 0), (base + H), (base + 2*H), ... (base + (TOPK-1)*H)
                // All offsets known at compile time!
                VecType const input_vec = *reinterpret_cast<VecType const*>(input_ptr_base);
                T const* input_ptr = reinterpret_cast<T const*>(&input_vec);

                // Convert to FP32 and accumulate
#pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i)
                {
                    acc[i] += to_float(input_ptr[i]);
                }

                // Increment by hidden_size (compiler knows this at compile time for unrolled loop)
                input_ptr_base += hidden_size;
            }

            // Vectorized load residual (simple offset)
            VecType const residual_vec = *reinterpret_cast<VecType const*>(residual_token_base + h_offset);
            T const* residual_ptr = reinterpret_cast<T const*>(&residual_vec);

            // Prepare output vector
            VecType output_vec;
            T* output_ptr = reinterpret_cast<T*>(&output_vec);

// Add residual and convert back to T
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i)
            {
                acc[i] += to_float(residual_ptr[i]);
                if constexpr (std::is_same_v<T, half>)
                {
                    output_ptr[i] = from_float_to_half(acc[i]);
                }
                else if constexpr (std::is_same_v<T, __nv_bfloat16>)
                {
                    output_ptr[i] = from_float_to_bfloat16(acc[i]);
                }
            }

            // 128-bit vectorized store (simple offset)
            *reinterpret_cast<VecType*>(output_token_base + h_offset) = output_vec;
        }

        // Handle remaining elements (if hidden_size is not multiple of VEC_SIZE)
        for (int32_t h = remaining_start + tid; h < hidden_size; h += threads_per_block)
        {
            float sum_val = 0.0f;

            // TOPK is compile-time constant, loop will be completely unrolled
#pragma unroll
            for (int32_t k = 0; k < TOPK; ++k)
            {
                sum_val += to_float(input_token_base[k * hidden_size + h]);
            }

            sum_val += to_float(residual_token_base[h]);

            if constexpr (std::is_same_v<T, half>)
            {
                output_token_base[h] = from_float_to_half(sum_val);
            }
            else if constexpr (std::is_same_v<T, __nv_bfloat16>)
            {
                output_token_base[h] = from_float_to_bfloat16(sum_val);
            }
        }
    }
}

// Helper function to get SM count with caching
inline int32_t getSMCount()
{
    static int cached_sm_count = -1;
    static int cached_device_id = -1;

    int device_id;
    cudaGetDevice(&device_id);

    // Cache SM count per device
    if (cached_device_id != device_id || cached_sm_count == -1)
    {
        cudaDeviceGetAttribute(&cached_sm_count, cudaDevAttrMultiProcessorCount, device_id);
        cached_device_id = device_id;
    }

    return cached_sm_count;
}

// Helper function to calculate optimal grid size
inline int32_t getOptimalGridSize(int32_t num_tokens)
{
    // Use persistent kernel approach to avoid launching too many blocks
    // Heuristic: launch enough blocks to keep all SMs busy, but not too many
    // Typically 2-4 blocks per SM for good occupancy and load balancing
    constexpr int blocks_per_sm = 4;
    int32_t const sm_count = getSMCount();
    int32_t const max_blocks = sm_count * blocks_per_sm;

    // For small num_tokens, use fewer blocks to avoid overhead
    // For large num_tokens, cap at max_blocks for persistent kernel benefits
    return std::min(num_tokens, max_blocks);
}

// Helper to dispatch based on runtime topk value to compile-time template
template <typename T>
void launchReduceAddKernel(T const* input, T const* residual, T* output, int32_t num_tokens, int32_t topk,
    int32_t hidden_size, cudaStream_t stream)
{
    constexpr int threads_per_block = 256;
    int32_t const grid_size = getOptimalGridSize(num_tokens);
    dim3 grid(grid_size);
    dim3 block(threads_per_block);

    // Dispatch to template specialization based on topk value
    // Supported topk values: 2, 4, 6, 8, 16
    switch (topk)
    {
    case 2: reduceAddKernel<T, 2><<<grid, block, 0, stream>>>(input, residual, output, num_tokens, hidden_size); break;
    case 4: reduceAddKernel<T, 4><<<grid, block, 0, stream>>>(input, residual, output, num_tokens, hidden_size); break;
    case 6: reduceAddKernel<T, 6><<<grid, block, 0, stream>>>(input, residual, output, num_tokens, hidden_size); break;
    case 8: reduceAddKernel<T, 8><<<grid, block, 0, stream>>>(input, residual, output, num_tokens, hidden_size); break;
    case 16:
        reduceAddKernel<T, 16><<<grid, block, 0, stream>>>(input, residual, output, num_tokens, hidden_size);
        break;
    default:
        // Throw error for unsupported topk values
        TLLM_CHECK_WITH_INFO(false, "Unsupported topk value: %d. Supported values are: 2, 4, 6, 8, 16", topk);
    }
}

// Template specializations for invokeReduceAdd
template <>
void invokeReduceAdd<half>(half const* input, half const* residual, half* output, int32_t num_tokens, int32_t topk,
    int32_t hidden_size, cudaStream_t stream)
{
    launchReduceAddKernel<half>(input, residual, output, num_tokens, topk, hidden_size, stream);
}

template <>
void invokeReduceAdd<__nv_bfloat16>(__nv_bfloat16 const* input, __nv_bfloat16 const* residual, __nv_bfloat16* output,
    int32_t num_tokens, int32_t topk, int32_t hidden_size, cudaStream_t stream)
{
    launchReduceAddKernel<__nv_bfloat16>(input, residual, output, num_tokens, topk, hidden_size, stream);
}

} // namespace kernels
} // namespace tensorrt_llm
