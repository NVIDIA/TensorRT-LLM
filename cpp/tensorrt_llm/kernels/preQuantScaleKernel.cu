/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace
{
template <typename T>
struct Vec2Type;

template <>
struct Vec2Type<half>
{
    using type = half2;
};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
template <>
struct Vec2Type<__nv_bfloat16>
{
    using type = __nv_bfloat162;
};
#endif
}; // namespace

template <typename T_in, typename T_out, int kProcessRows, typename AccessType>
__global__ void apply_per_channel_scale(
    T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale, int rows, int cols)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T_in);
    T_in scale[kElems], act_vec[kElems];
    int col_offset = blockIdx.y * blockDim.x + threadIdx.x;
    int row_offset = blockIdx.x;
    if (col_offset * kElems >= cols || row_offset * kProcessRows >= rows)
        return;
    act += row_offset * kProcessRows * cols;
    smoothed_act += row_offset * kProcessRows * cols;
    *reinterpret_cast<AccessType*>(scale) = reinterpret_cast<AccessType const*>(per_channel_scale)[col_offset];
#pragma unroll
    for (int i = 0; i < kProcessRows; ++i)
    {
        *reinterpret_cast<AccessType*>(act_vec) = reinterpret_cast<AccessType const*>(act + i * cols)[col_offset];
        if constexpr ((std::is_same_v<T_in, half>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
                          || std::is_same_v<T_in, __nv_bfloat16>
#endif
                          ) &&(kElems % 2 == 0))
        {
            using Vec2 = typename Vec2Type<T_in>::type;
#pragma unroll
            for (int j = 0; j < kElems; j += 2)
            {
                *reinterpret_cast<Vec2*>(act_vec + j)
                    = __hmul2(*reinterpret_cast<Vec2*>(act_vec + j), *reinterpret_cast<Vec2*>(scale + j));
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElems; ++j)
            {
                act_vec[j] = static_cast<T_in>(static_cast<float>(act_vec[j]) * static_cast<float>(scale[j]));
            }
        }
        if constexpr (std::is_same_v<T_in, T_out>)
        {
            reinterpret_cast<AccessType*>(smoothed_act + i * cols)[col_offset]
                = *reinterpret_cast<AccessType*>(act_vec);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElems; ++j)
            {
                (smoothed_act + i * cols)[col_offset * kElems + j] = static_cast<T_out>(act_vec[j]);
            }
        }
    }
}

template <typename T_in, typename T_out, int kProcessRows, typename AccessType = float4>
void apply_per_channel_scale_kernel_launcher_(
    T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale, int rows, int cols, cudaStream_t stream = 0)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T_in);
    dim3 block(128);
    dim3 grid((rows + kProcessRows - 1) / kProcessRows, (cols / kElems + block.x - 1) / block.x);
    apply_per_channel_scale<T_in, T_out, kProcessRows, AccessType>
        <<<grid, block, 0, stream>>>(smoothed_act, act, per_channel_scale, rows, cols);
}

template <typename T_in, typename T_out>
void apply_per_channel_scale_kernel_launcher(
    T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale, int rows, int cols, cudaStream_t stream)
{
    uint64_t elems = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
    if (elems < 2048 * 2048)
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 1, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
    else if (elems < 4096 * 4096)
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 4, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
    else if (elems < 8192 * 8192)
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 8, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
    else
    {
        apply_per_channel_scale_kernel_launcher_<T_in, T_out, 16, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
}

#define INSTANTIATE_PREQUANT_SCALE(T_in, T_out)                                                                        \
    template void apply_per_channel_scale_kernel_launcher<T_in, T_out>(                                                \
        T_out * smoothed_act, const T_in* act, const T_in* per_channel_scale, int rows, int cols, cudaStream_t stream)

INSTANTIATE_PREQUANT_SCALE(half, half);
#if defined(ENABLE_FP8)
INSTANTIATE_PREQUANT_SCALE(half, __nv_fp8_e4m3);
#endif

#if defined(ENABLE_BF16)
INSTANTIATE_PREQUANT_SCALE(__nv_bfloat16, __nv_bfloat16);
#if defined(ENABLE_FP8)
INSTANTIATE_PREQUANT_SCALE(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

template <typename T_in, typename T_out, int kProcessRows, typename AccessType>
__global__ void apply_per_expert_scale(T_out* smoothed_act, T_in const* act, T_in const* per_expert_scale,
    int const* permuted_token_selected_experts, int64_t const* num_valid_tokens_ptr, int rows, int cols)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T_in);
    T_in act_vec[kElems];
    int col_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int row_offset = blockIdx.y;
    int expert_idx = permuted_token_selected_experts[row_offset];
    T_in scale = per_expert_scale[expert_idx];
    if (col_offset * kElems >= cols || row_offset * kProcessRows >= rows)
        return;
    if (num_valid_tokens_ptr && (row_offset * kProcessRows >= *num_valid_tokens_ptr))
        return;
    act += row_offset * kProcessRows * cols;
    smoothed_act += row_offset * kProcessRows * cols;
#pragma unroll
    for (int i = 0; i < kProcessRows; ++i)
    {
        *reinterpret_cast<AccessType*>(act_vec) = reinterpret_cast<AccessType const*>(act + i * cols)[col_offset];
        if constexpr ((std::is_same_v<T_in, half>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
                          || std::is_same_v<T_in, __nv_bfloat16>
#endif
                          ) &&(kElems % 2 == 0))
        {
            using Vec2 = typename Vec2Type<T_in>::type;
#pragma unroll
            for (int j = 0; j < kElems; j += 2)
            {
                if constexpr (std::is_same_v<T_in, half>)
                {
                    *reinterpret_cast<Vec2*>(act_vec + j)
                        = __hmul2(*reinterpret_cast<Vec2*>(act_vec + j), __half2half2(scale));
                }
                else
                {
                    *reinterpret_cast<Vec2*>(act_vec + j)
                        = __hmul2(*reinterpret_cast<Vec2*>(act_vec + j), __bfloat162bfloat162(scale));
                }
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElems; ++j)
            {
                act_vec[j] = static_cast<T_in>(static_cast<float>(act_vec[j]) * static_cast<float>(scale));
            }
        }
        if constexpr (std::is_same_v<T_in, T_out>)
        {
            reinterpret_cast<AccessType*>(smoothed_act + i * cols)[col_offset]
                = *reinterpret_cast<AccessType*>(act_vec);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElems; ++j)
            {
                (smoothed_act + i * cols)[col_offset * kElems + j] = static_cast<T_out>(act_vec[j]);
            }
        }
    }
}

template <typename T_in, typename T_out, int kProcessRows, typename AccessType = float4>
void apply_per_expert_scale_kernel_launcher_(T_out* smoothed_act, T_in const* act, T_in const* per_expert_scale,
    int const* permuted_token_selected_experts, int64_t const* num_valid_tokens_ptr, int rows, int cols,
    cudaStream_t stream = 0)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T_in);
    dim3 block(128);
    dim3 grid((cols / kElems + block.x - 1) / block.x, (rows + kProcessRows - 1) / kProcessRows);
    apply_per_expert_scale<T_in, T_out, kProcessRows, AccessType><<<grid, block, 0, stream>>>(
        smoothed_act, act, per_expert_scale, permuted_token_selected_experts, num_valid_tokens_ptr, rows, cols);
}

template <typename T_in, typename T_out>
void apply_per_expert_scale_kernel_launcher(T_out* smoothed_act, T_in const* act, T_in const* per_expert_scale,
    int const* permuted_token_selected_experts, int64_t const* num_valid_tokens_ptr, int rows, int cols,
    cudaStream_t stream)
{
    int elems = rows * cols;
    if (elems < 2048 * 2048)
    {
        apply_per_expert_scale_kernel_launcher_<T_in, T_out, 1, float4>(smoothed_act, act, per_expert_scale,
            permuted_token_selected_experts, num_valid_tokens_ptr, rows, cols, stream);
    }
    else if (elems < 4096 * 4096)
    {
        apply_per_expert_scale_kernel_launcher_<T_in, T_out, 4, float4>(smoothed_act, act, per_expert_scale,
            permuted_token_selected_experts, num_valid_tokens_ptr, rows, cols, stream);
    }
    else if (elems < 8192 * 8192)
    {
        apply_per_expert_scale_kernel_launcher_<T_in, T_out, 8, float4>(smoothed_act, act, per_expert_scale,
            permuted_token_selected_experts, num_valid_tokens_ptr, rows, cols, stream);
    }
    else
    {
        apply_per_expert_scale_kernel_launcher_<T_in, T_out, 16, float4>(smoothed_act, act, per_expert_scale,
            permuted_token_selected_experts, num_valid_tokens_ptr, rows, cols, stream);
    }
}

#define INSTANTIATE_PEREXPERT_SCALE(T_in, T_out)                                                                       \
    template void apply_per_expert_scale_kernel_launcher<T_in, T_out>(T_out * smoothed_act, T_in const* act,           \
        T_in const* per_expert_scale, int const* permuted_token_selected_experts, int64_t const* num_valid_tokens_ptr, \
        int rows, int cols, cudaStream_t stream)

INSTANTIATE_PEREXPERT_SCALE(half, half);
#if defined(ENABLE_FP8)
INSTANTIATE_PEREXPERT_SCALE(half, __nv_fp8_e4m3);
#endif

#if defined(ENABLE_BF16)
INSTANTIATE_PEREXPERT_SCALE(__nv_bfloat16, __nv_bfloat16);
#if defined(ENABLE_FP8)
INSTANTIATE_PEREXPERT_SCALE(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

} // namespace kernels
} // namespace tensorrt_llm
