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

#pragma once
#include <array>
#include <cstdint>
#include <cub/cub.cuh>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <optional>
#include <string>
#include <vector>

#include "ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh"
#include "fp8_blockscale_mma_utils.cuh"
#include "fp8_blockscale_tma_utils.cuh"
#include "sm120_blockwise_gemm/sm120_fp8_gemm_1d1d.cuh"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/deep_gemm/fp8_gemm.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernel_utils
{

inline void find_divisor(uint32_t& mul, uint32_t& shr, int x)
{

    auto find_log_2 = [](int x, bool round_up = false)
    {
        auto clz = [](int x)
        {
            for (int i = 31; i >= 0; --i)
            {
                if ((1 << i) & x)
                {
                    return 31 - i;
                }
            }
            return 32;
        };

        int a = 31 - clz(x);
        if (round_up)
        {
            a += (x & (x - 1)) ? 1 : 0;
        }
        return a;
    };

    assert(x != 0);
    if (x == 1)
    {
        // If dividing by 1, reduced math doesn't work because mul_coeff would need
        // to be 2^32, which doesn't fit into unsigned int.  the div() routine
        // handles this special case separately.
        mul = 0;
        shr = 0;
    }
    else
    {
        // To express the division N/D in terms of a multiplication, what we first
        // imagine is simply N*(1/D).  However, 1/D will always evaluate to 0 (for
        // D>1), so we need another way.  There's nothing that says we have to use
        // exactly the fraction 1/D; instead it could be any X/Y that reduces to 1/D
        // (i.e., Y=X*D), or at least to "close enough" to it.  If we pick Y that is
        // a power of two, then the N*(X/Y) can be N*X followed by a right-shift by
        // some amount. The power of two we should pick should be at least 2^32,
        // because in the div() routine we'll use umulhi(), which returns only the
        // upper 32 bits -- this being equivalent to a right-shift by 32.  But we
        // might want a higher power of two for better accuracy depending on the
        // magnitude of the denominator. Once we've picked Y, then X [our mul_coeff
        // value] is simply Y/D, rounding up, and we save shift_coeff as whatever
        // further shift we have to do beyond what the umulhi() implies.
        uint32_t p = 31 + find_log_2(x, true);
        uint32_t m = (uint32_t) (((1ull << p) + (uint32_t) x - 1) / (uint32_t) x);

        mul = m;
        shr = p - 32;
    }
}

__device__ __forceinline__ void fast_divmod(uint32_t& div, uint32_t& mod, int x, int y, uint32_t mul, uint32_t shr)
{
    if (y == 1)
    {
        div = x;
        mod = 0;
    }
    else
    {
        div = __umulhi((uint32_t) x, mul) >> shr;
        mod = x - div * y;
    }
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

template <>
__inline__ __device__ __nv_bfloat16 warpReduceSum(__nv_bfloat16 val)
{
    constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = __hmax(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

__inline__ __device__ uint32_t elect_one_sync([[maybe_unused]] int lane_id)
{
    uint32_t pred = 0;
#if __CUDA_ARCH__ >= 900
    uint32_t laneid = 0;
    asm volatile(
        "\n\
    {\n\
        .reg .b32 %rx;\n\
        .reg .pred %px;\n\
        elect.sync %rx|%px, %2;\n\
        @%px mov.s32 %1, 1;\n\
        mov.s32 %0, %rx;\n\
    }\n\
  "
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
#else
    return lane_id == 0;
#endif
    return pred;
}

} // namespace kernel_utils

namespace kernels::fp8_blockscale_gemm
{

template <typename T>
__device__ __host__ constexpr T div_up(T a, int b)
{
    return (a + b - 1) / b;
}

template <typename T>
__forceinline__ __device__ T find_max_elem_in_warp(T value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value = T(std::max(float(value), __shfl_down_sync(0xFFFFFFFF, float(value), offset)));
    }
    value = T(__shfl_sync(0xffffffff, float(value), 0));
    return value;
}

template <typename InputType, typename OutputType, typename ScaleType = float, bool USE_UE8M0 = false>
__global__ void scale_1x128_kernel(
    OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    size_t scales_along_dim_x = div_up(dim_x, 128);
    size_t scales_along_dim_y = div_up(dim_y, 1);
    size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;
    using Input2Type = typename std::conditional<std::is_same<InputType, half>::value, half2, __nv_bfloat162>::type;
    for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         warp_idx < scales_along_dim_x * scales_along_dim_y; warp_idx += gridDim.x * blockDim.x / 32)
    {
        int scales_idx_y = warp_idx / scales_along_dim_x;
        int scales_idx_x = warp_idx % scales_along_dim_x;

        InputType const* input_line = input + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
        InputType input_amax = InputType(0);
        // Each thread reads 2 elements from input_line
        int lane_id = threadIdx.x % 32 * 2;

        Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                input_frag2[i] = *((Input2Type*) (input_line) + lane_id / 2);
            }
            input_line += 64;
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                input_amax = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
            }
        }

        InputType amax = find_max_elem_in_warp(input_amax);
        ScaleType quant_scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;
        ScaleType dequant_scale;

        if constexpr (USE_UE8M0)
        {
            // Round dequant scale to UE8M0 (power of 2)
            ScaleType dequant_scale_raw = 1.f / quant_scale;
            __nv_fp8_e8m0 ue8m0_scale;
            ue8m0_scale.__x = __nv_cvt_float_to_e8m0(float(dequant_scale_raw), __NV_SATFINITE, cudaRoundPosInf);
            // Cast back to float automatically decodes E8M0 format
            dequant_scale = ScaleType(static_cast<float>(ue8m0_scale));
            // Recompute quant scale from rounded dequant scale for consistency
            quant_scale = dequant_scale != ScaleType(0.f) ? 1.f / dequant_scale : 1.f;
        }
        else
        {
            dequant_scale = 1.f / quant_scale;
        }

        if (lane_id == 0)
        {
            scales[(size_t) scales_idx_x * stride_scale_dim_y + scales_idx_y] = dequant_scale;
        }

        OutputType* output_line = output + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                ScaleType value_1 = ScaleType(input_frag2[i].x) * quant_scale;
                ScaleType value_2 = ScaleType(input_frag2[i].y) * quant_scale;
                output_line[lane_id] = OutputType(value_1);
                output_line[lane_id + 1] = OutputType(value_2);
            }
            output_line += 64;
        }
    }
#endif
}

template <bool UseBinarySearch, typename InputType, typename OutputType>
__global__ void scale_1x128_kernel(OutputType* output, float* scales, InputType const* input,
    int64_t const* problem_m_offsets, int num_problems, int dim_x, int64_t scale_leading_dim, uint32_t scale_dim_x_mul,
    uint32_t scale_dim_x_shr)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    extern __shared__ char shared_memory[];
    int64_t* smem_problem_m_boundaries = reinterpret_cast<int64_t*>(shared_memory);

    // problem_m_offsets[0] is omitted because its value is known to be 0
    for (int i = threadIdx.x; i < num_problems; i += blockDim.x)
    {
        smem_problem_m_boundaries[i] = problem_m_offsets[i + 1];
    }
    __syncthreads();

    size_t scales_along_dim_x = div_up(dim_x, 128);
    size_t scales_along_dim_y = smem_problem_m_boundaries[num_problems - 1];
    size_t total_scales = scales_along_dim_x * scales_along_dim_y;

    int problem_idx = 0;
    int64_t padded_offset = 0;
    int64_t boundary_left, boundary_right;
    if constexpr (UseBinarySearch)
    {
        boundary_left = smem_problem_m_boundaries[0];
        boundary_right = scales_along_dim_y;
    }
    else
    {
        boundary_left = 0;
        boundary_right = smem_problem_m_boundaries[0];
    }

    for (size_t warp_idx = (threadIdx.x + blockIdx.x * blockDim.x) / 32; warp_idx < total_scales;
         warp_idx += (blockDim.x * gridDim.x) / 32)
    {
        uint32_t scales_idx_y; // = warp_idx / scales_along_dim_x;
        uint32_t scales_idx_x; // = warp_idx % scales_along_dim_x;
        kernel_utils::fast_divmod(
            scales_idx_y, scales_idx_x, warp_idx, scales_along_dim_x, scale_dim_x_mul, scale_dim_x_shr);

        if constexpr (UseBinarySearch)
        {
            int idx_right = num_problems - 1;
            int64_t val_right = boundary_right;
            if (scales_idx_y >= boundary_left)
            {
                while (problem_idx + 1 < idx_right)
                {
                    int idx_mid = (problem_idx + idx_right) >> 1;
                    int64_t val_mid = smem_problem_m_boundaries[idx_mid];
                    if (scales_idx_y < val_mid)
                    {
                        idx_right = idx_mid;
                        val_right = val_mid;
                    }
                    else
                    {
                        problem_idx = idx_mid;
                        boundary_left = val_mid;
                    }
                }
                padded_offset = deep_gemm::compute_padded_offset(boundary_left, problem_idx + 1) - boundary_left;
                boundary_left = val_right;
            }
        }
        else
        {
            if (boundary_right <= scales_idx_y)
            {
                while (problem_idx < num_problems - 1)
                {
                    boundary_left = boundary_right;
                    boundary_right = smem_problem_m_boundaries[++problem_idx];
                    if (scales_idx_y < boundary_right)
                    {
                        break;
                    }
                }
                padded_offset = deep_gemm::compute_padded_offset(boundary_left, problem_idx) - boundary_left;
            }
        }

        auto warp_offset = (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
        InputType const* input_line = input + warp_offset;
        OutputType* output_line = output + warp_offset;
        auto& scale_output = scales[(size_t) scales_idx_x * scale_leading_dim + scales_idx_y + padded_offset];

        int lane_id = threadIdx.x % 32;
        InputType input_frag[4];

        for (int i = 0; i < 4; i++)
        {
            input_frag[i] = (scales_idx_x * 128 + i * 32 + lane_id < dim_x) ? input_line[lane_id] : InputType(0);
            input_line += 32;
        }

        InputType amax = kernel_utils::warpReduceSum(max(max(fabs(float(input_frag[0])), fabs(float(input_frag[1]))),
            max(fabs(float(input_frag[2])), fabs(float(input_frag[3])))));

        // Half seems to be slower, probably because we need float values below
        // anyway. InputType amax = kernel_utils::warpReduceSum(
        //     __hmax(__hmax(__habs(input_frag[0]), __habs(input_frag[1])),
        //         __hmax(__habs(input_frag[2]), __habs(input_frag[3]))));

        float scale = amax != InputType(0.f) ? 448.f / float(amax) : 1.f;

        if (kernel_utils::elect_one_sync(lane_id))
        {
            scale_output = float(1.f / scale);
        }

        for (int i = 0; i < 4; i++)
        {
            float value = float(input_frag[i]) * scale;
            if (scales_idx_x * 128 + i * 32 + lane_id < dim_x)
            {
                output_line[lane_id] = OutputType(value);
            }
            output_line += 32;
        }
    }
#endif
}

// input: [dim_y, dim_h, dim_x]
// output: [dim_h, dim_y, dim_x], cs[dim_h, dim_x/128, padding(dim_y)]
template <typename InputType, typename OutputType, typename ScaleType = float>
__global__ void scale_1x128_reshape_kernel(
    OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_h, int dim_y, int stride_x)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    size_t scales_along_dim_x = div_up(dim_x, 128);
    size_t scales_along_dim_y = div_up(dim_y, 1);
    size_t scales_along_dim_h = div_up(dim_h, 1);
    size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;
    using Input2Type = typename std::conditional<std::is_same<InputType, half>::value, half2, __nv_bfloat162>::type;
    for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         warp_idx < scales_along_dim_x * scales_along_dim_y * scales_along_dim_h;
         warp_idx += gridDim.x * blockDim.x / 32)
    {
        int scales_idx_y = warp_idx / (scales_along_dim_x * scales_along_dim_h);
        int scales_idx_h = (warp_idx % (scales_along_dim_x * scales_along_dim_h)) / scales_along_dim_x;
        int scales_idx_x = warp_idx % scales_along_dim_x;

        InputType const* input_line
            = input + (size_t) scales_idx_y * stride_x * dim_h + (size_t) scales_idx_h * stride_x + scales_idx_x * 128;
        InputType input_amax = InputType(0);
        int lane_id = threadIdx.x % 32 * 2;

        Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                input_frag2[i] = *((Input2Type*) (input_line) + lane_id / 2);
            }
            input_line += 64;
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                input_amax = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
            }
        }

        InputType amax = find_max_elem_in_warp(input_amax);
        ScaleType scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;

        if (lane_id == 0)
        {
            scales[(size_t) scales_idx_h * scales_along_dim_x * stride_scale_dim_y
                + (size_t) scales_idx_x * stride_scale_dim_y + scales_idx_y]
                = ScaleType(1.f / scale);
        }

        OutputType* output_line
            = output + (size_t) scales_idx_h * dim_y * dim_x + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                ScaleType value_1 = ScaleType(input_frag2[i].x) * scale;
                ScaleType value_2 = ScaleType(input_frag2[i].y) * scale;
                output_line[lane_id] = OutputType(value_1);
                output_line[lane_id + 1] = OutputType(value_2);
            }
            output_line += 64;
        }
    }
#endif
}

template <typename InputType, typename OutputType, typename ScaleType = float>
__global__ void scale_128x128_kernel(
    OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    int scales_along_dim_x = div_up(dim_x, 128);
    int scales_along_dim_y = div_up(dim_y, 128);

    for (int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         warp_idx < scales_along_dim_x * scales_along_dim_y; warp_idx += gridDim.x * blockDim.x / 32)
    {
        int scales_idx_y = warp_idx / scales_along_dim_x;
        int scales_idx_x = warp_idx % scales_along_dim_x;

        InputType const* input_line = input + scales_idx_y * 128 * dim_x + scales_idx_x * 128;
        InputType input_amax = InputType(0);
        int lane_id = threadIdx.x % 32;

        for (int i = 0; i < 128; i++)
        {
            if (scales_idx_y * 128 + i >= dim_y)
            {
                break;
            }
            InputType const* input_d = input_line;

            for (int j = 0; j < 4; j++)
            {
                if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x)
                {
                    break;
                }
                else
                {
                    input_amax = InputType(std::max(float(input_amax), std::fabs(float(input_d[lane_id]))));
                }
                input_d += 32;
            }
            input_line += dim_x;
        }

        InputType amax = find_max_elem_in_warp(input_amax);
        ScaleType scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;

        if (lane_id == 0)
        {
            scales[scales_idx_y * scales_along_dim_x + scales_idx_x] = ScaleType(1.f / scale);
        }

        input_line = input + scales_idx_y * 128 * dim_x + scales_idx_x * 128;
        OutputType* output_line = output + scales_idx_y * 128 * dim_x + scales_idx_x * 128;

        for (int i = 0; i < 128; i++)
        {
            if (scales_idx_y * 128 + i >= dim_y)
            {
                break;
            }
            InputType const* input_d = input_line;
            OutputType* output_d = output_line;

            for (int j = 0; j < 4; j++)
            {
                if (scales_idx_x * 128 + j * 32 + lane_id >= dim_x)
                {
                    break;
                }
                else
                {
                    output_d[lane_id] = OutputType(ScaleType(input_d[lane_id]) * scale);
                }
                input_d += 32;
                output_d += 32;
            }

            input_line += dim_x;
            output_line += dim_x;
        }
    }
#endif
}

template <typename OutputType>
__global__ void fill_kernel(OutputType* output, size_t num_elems, float value)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elems; idx += gridDim.x * blockDim.x)
    {
        output[idx] = OutputType(value);
    }
}

template <typename InputType, typename OutputType>
__global__ void convert_kernel(OutputType* output, InputType const* const input, size_t num_elems)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elems; idx += gridDim.x * blockDim.x)
    {
        float value = float(input[idx]);
        if (std::isnan(value))
        {
            output[idx] = OutputType(448);
        }
        else
        {
            output[idx] = OutputType(value);
        }
    }
}

static int kNumDeviceSMs = -1;

void fp8_1x128_cs(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
    cudaStream_t stream, bool use_ue8m0 = false)
{
    if (kNumDeviceSMs < 0)
    {
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    if (use_ue8m0)
    {
        scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, true>
            <<<kNumDeviceSMs * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
    }
    else
    {
        scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, false>
            <<<kNumDeviceSMs * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
    }
}

void fp8_1x128_cs_reshape(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_h,
    int shape_y, int stride_x, cudaStream_t stream)
{
    if (kNumDeviceSMs < 0)
    {
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    scale_1x128_reshape_kernel<<<kNumDeviceSMs * 8, 256, 0, stream>>>(
        mat_quant, scales, mat, shape_x, shape_h, shape_y, stride_x);
}

void fp8_128x128_cs(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream)
{
    if (kNumDeviceSMs < 0)
    {
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    convert_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(mat_quant, mat, shape_x * shape_y);
    fill_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(scales, div_up(shape_x, 128) * div_up(shape_y, 128), 1);
}

void gemm_dispatch(void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,
    float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
    {
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }

    constexpr uint32_t block_k = 128;
    constexpr uint32_t num_problems = 1;

    uint32_t m_threshold = 32;
    if (shape_m >= m_threshold)
    {
        // Select the best configuration based on shape dimensions
        auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size]
            = deep_gemm::jit::get_best_gemm_config(shape_m, shape_n, shape_k, num_problems, num_device_sms);

        auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, best_block_m, best_block_n, block_k,
            num_problems, best_num_stages, best_num_tma_multicast, deep_gemm::GemmType::Normal);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        deep_gemm::runGemm(kernel, mat_a, ld_a, mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, shape_m, shape_n, shape_k,
            best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast, deep_gemm::GemmType::Normal,
            static_cast<int*>(nullptr), stream, num_device_sms, static_cast<uint32_t>(best_smem_size));
    }
    else
    {
        auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size]
            = deep_gemm::jit::get_best_gemm_config(
                shape_n, shape_m, shape_k, num_problems, num_device_sms, false, true);
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, best_block_m, best_block_n, block_k,
            num_problems, best_num_stages, best_num_tma_multicast, deep_gemm::GemmType::Normal, true);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        deep_gemm::runGemmSwapAB(kernel, mat_b, ld_b, mat_a, ld_a, mat_d, ld_d, scales_b, scales_a, shape_n, shape_m,
            shape_k, best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
            deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream, num_device_sms,
            static_cast<uint32_t>(best_smem_size));
    }
}

void gemm_dispatch_sm89(void* mat_a, void* mat_b, void* mat_d, float* scales_a, float* scales_b, uint32_t shape_m,
    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
    {
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementAccum = float;
    using ElementBlockScale = float;
    static constexpr int Stages = 3;
    using TileShape = cutlass::gemm::GemmShape<32, 128, 128>;
    using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<ElementInput, ElementOutput, ElementAccum, ElementBlockScale,
        Stages, TileShape::kM, TileShape::kN, TileShape::kK>;
    using GemmKernel = ada_blockwise_gemm::AdaBlockwiseGemmKernel<KT>;

    static constexpr int kSmemSize = KT::kSmemSize;
    static constexpr int kThreadCount = KT::kThreadCount;
    int grid_m = (shape_m + KT::kTileM - 1) / KT::kTileM;
    int grid_n = (shape_n + KT::kTileN - 1) / KT::kTileN;
    int grid_k = 1;
    dim3 grid = dim3(grid_m, grid_n, grid_k);
    dim3 block = dim3(kThreadCount, 1, 1);

    auto result = cudaFuncSetAttribute(ada_blockwise_gemm::sm89_fp8_gemm_1d1d_impl<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm89 gemm kernel cannot launch: %s", cudaGetErrorString(result));

    ada_blockwise_gemm::sm89_fp8_gemm_1d1d_impl<GemmKernel>
        <<<grid, block, kSmemSize, stream>>>(shape_m, shape_n, shape_k, mat_a, mat_b, mat_d, scales_a, scales_b);

    result = cudaGetLastError();
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm89 gemm kernel runtime error: %s", cudaGetErrorString(result));
}

void gemm_dispatch_sm120(void* mat_a, void* mat_b, void* mat_d, float* scales_a, float* scales_b, uint32_t shape_m,
    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
    {
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementAccum = float;
    using ElementBlockScale = int32_t;
    using KT = sm120_blockscaled_gemm::SM120BlockScaledBuilder<32, 128>;
    using GemmKernel = sm120_blockscaled_gemm::SM120BlockScaledKernel<KT>;
    using Params = typename GemmKernel::Params;
    using Arguments = typename GemmKernel::Arguments;
    using ProblemShape = typename GemmKernel::ProblemShape;
    ProblemShape problem_shape = make_shape((int) shape_m, (int) shape_n, (int) shape_k, 1);

    auto ptr_A = reinterpret_cast<ElementInput*>(mat_a);
    auto ptr_B = reinterpret_cast<ElementInput*>(mat_b);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale*>(scales_a);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale*>(scales_b);
    auto ptr_D = reinterpret_cast<ElementOutput*>(mat_d);

    int32_t ld_a = shape_k;
    int32_t stride_a = shape_m * shape_k;
    int32_t ld_b = shape_k;
    int32_t stride_b = shape_n * shape_k;
    int32_t ld_d = shape_n;
    int32_t stride_d = shape_m * shape_n;

    typename KT::StrideA dA = make_stride(ld_a, Int<1>{}, stride_a);
    typename KT::StrideB dB = make_stride(ld_b, Int<1>{}, stride_b);
    typename KT::StrideSFA dSFA = KT::deduce_sfa_layout(problem_shape).stride();
    typename KT::StrideSFB dSFB = KT::deduce_sfb_layout(problem_shape).stride();
    typename KT::StrideD dD = make_stride(ld_d, Int<1>{}, stride_d);

    Arguments args = {ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD};

    Params kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
    auto result = cudaGetLastError();
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 gemm kernel cannot launch: %s", cudaGetErrorString(result));

    cudaLaunchConfig_t launch_config;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    launch_config.gridDim = GemmKernel::get_grid_shape(kernel_params);
    launch_config.blockDim = GemmKernel::get_block_shape();
    launch_config.dynamicSmemBytes = GemmKernel::kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;

    cudaLaunchKernelEx(&launch_config, kernel_ptr, kernel_params);

    result = cudaGetLastError();
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 gemm kernel runtime error: %s", cudaGetErrorString(result));
}

void fp8_gemm_run(__nv_fp8_e4m3* mat_a, int ld_a, __nv_fp8_e4m3* mat_b, int ld_b, __nv_bfloat16* mat_d, int ld_d,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, float* scales_a, float* scales_b, cudaStream_t stream)
{
    if (shape_m == 0)
    {
        return;
    }
#ifndef PLACEHOLDER_KERNELS
    int arch = tensorrt_llm::common::getSMVersion();
    if (arch == 89)
    {
        gemm_dispatch_sm89(mat_a, mat_b, mat_d, scales_a, scales_b, shape_m, shape_n, shape_k, stream);
        return;
    }
    if (arch == 120)
    {
        gemm_dispatch_sm120(mat_a, mat_b, mat_d, scales_a, scales_b, shape_m, shape_n, shape_k, stream);
        return;
    }
    gemm_dispatch(mat_a, ld_a, mat_b, ld_b, mat_d, ld_d, scales_a, scales_b, shape_m, shape_n, shape_k, stream);
#endif
}

void fp8_gemm_run(__nv_bfloat16 const* mat_a, __nv_fp8_e4m3* fp8_mat_a, int ld_a, float* scales_a,
    __nv_bfloat16 const* mat_b, __nv_fp8_e4m3* fp8_mat_b, int ld_b, float* scales_b, __nv_bfloat16* mat_d, int ld_d,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, bool internal_quantize_a = true,
    bool internal_quantize_b = true)
{
    if (shape_m == 0)
    {
        return;
    }
    if (kNumDeviceSMs < 0)
    {
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }

    if (internal_quantize_a)
    {
        scale_1x128_kernel<<<kNumDeviceSMs * 8, 256, 0, stream>>>(fp8_mat_a, scales_a, mat_a, shape_k, shape_m);
    }
    if (internal_quantize_b)
    {
        scale_128x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(fp8_mat_b, scales_b, mat_b, shape_k, shape_n);
    }
    fp8_gemm_run(fp8_mat_a, ld_a, fp8_mat_b, ld_b, mat_d, ld_d, shape_m, shape_n, shape_k, scales_a, scales_b, stream);
}

void grouped_gemm_dispatch(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d, uint32_t num_problems,
    int64_t const* problem_m_offsets, uint32_t expected_m, uint32_t max_shape_m, uint32_t max_shape_m_padded,
    uint32_t shape_n, uint32_t shape_k, float* scales_a, float* scales_b, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
    {
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }

    constexpr uint32_t block_k = 128;
    uint32_t m_per_expert_threshold = num_device_sms == 78 ? 64 : 32; // 64 for H20(sms=78), 32 for H100/H200
    if (expected_m >= m_per_expert_threshold)
    {
        auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size]
            = deep_gemm::jit::get_best_gemm_config(expected_m, shape_n, shape_k, num_problems, num_device_sms);

        auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, best_block_m, best_block_n, block_k,
            num_problems, best_num_stages, best_num_tma_multicast, deep_gemm::GemmType::GroupedWithOffset);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
        deep_gemm::runGemm(kernel, mat_a, 0, mat_b, 0, mat_d, 0, scales_a, scales_b, max_shape_m, shape_n, shape_k,
            best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
            deep_gemm::GemmType::GroupedWithOffset, const_cast<int64_t*>(problem_m_offsets), stream, num_device_sms,
            static_cast<uint32_t>(best_smem_size), max_shape_m_padded);
    }
    else
    {
        auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size]
            = deep_gemm::jit::get_best_gemm_config(
                shape_n, expected_m, shape_k, num_problems, num_device_sms, false, true);
        auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, best_block_m, best_block_n, block_k,
            num_problems, best_num_stages, best_num_tma_multicast, deep_gemm::GemmType::GroupedWithOffset, true);
        auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());

        deep_gemm::runGemmSwapAB(kernel, mat_b, 0, mat_a, 0, mat_d, 0, scales_b, scales_a, shape_n, max_shape_m,
            shape_k, best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
            deep_gemm::GemmType::GroupedWithOffset, const_cast<int64_t*>(problem_m_offsets), stream, num_device_sms,
            static_cast<uint32_t>(best_smem_size), max_shape_m_padded);
    }
}

void fp8_grouped_gemm_run(__nv_bfloat16 const* mat_a, __nv_fp8_e4m3* fp8_mat_a, float* scales_a,
    __nv_bfloat16 const* mat_b, __nv_fp8_e4m3* fp8_mat_b, float* scales_b, __nv_bfloat16* mat_d,
    int64_t const* problem_m_offsets, int num_problems, int64_t expected_m, int64_t max_shape_m,
    int64_t max_shape_m_padded, int shape_n, int shape_k, cudaStream_t stream, bool internal_quantize_a = true,
    bool internal_quantize_b = true)
{
    if (kNumDeviceSMs < 0)
    {
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }

    if (internal_quantize_a)
    {
        constexpr int NumThreads = 256;
        int scales_dim_x = div_up(shape_k, 128);
        uint32_t scale_dim_x_mul, scale_dim_x_shr;
        kernel_utils::find_divisor(scale_dim_x_mul, scale_dim_x_shr, scales_dim_x);

        int smem_size = num_problems * sizeof(int64_t);
        int num_blocks
            = std::min(static_cast<int64_t>(kNumDeviceSMs), div_up(max_shape_m * scales_dim_x, NumThreads / 32));
        // Binary search is expected to have lower complexity when max_shape_m is small
        bool use_binary_search
            = static_cast<double>(max_shape_m) * scales_dim_x / static_cast<double>(NumThreads * num_blocks / 32)
            <= static_cast<double>(num_problems) / std::log2(static_cast<double>(num_problems));
        auto kernel = use_binary_search ? scale_1x128_kernel<true, __nv_bfloat16, __nv_fp8_e4m3>
                                        : scale_1x128_kernel<false, __nv_bfloat16, __nv_fp8_e4m3>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<num_blocks, NumThreads, smem_size, stream>>>(fp8_mat_a, scales_a, mat_a, problem_m_offsets,
            num_problems, shape_k, max_shape_m_padded, scale_dim_x_mul, scale_dim_x_shr);
    }

    if (internal_quantize_b)
    {
        __nv_fp8_e4m3* fp8_mat_b_tmp = fp8_mat_b;
        float* scales_b_tmp = scales_b;
        __nv_bfloat16 const* mat_b_tmp = mat_b;

        for (int i = 0; i < num_problems; i++)
        {
            scale_128x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(
                fp8_mat_b_tmp, scales_b_tmp, mat_b_tmp, shape_k, shape_n);
            fp8_mat_b_tmp += shape_n * shape_k;
            mat_b_tmp += shape_n * shape_k;
            scales_b_tmp += div_up(shape_n, 128) * div_up(shape_k, 128);
        }
    }

    grouped_gemm_dispatch(fp8_mat_a, fp8_mat_b, mat_d, num_problems, problem_m_offsets, expected_m, max_shape_m,
        max_shape_m_padded, shape_n, shape_k, scales_a, scales_b, stream);
}

void strided_batch_gemm_dispatch(__nv_fp8_e4m3* mat_a, int ld_a, int stride_a, __nv_fp8_e4m3* mat_b, int ld_b,
    int stride_b, __nv_bfloat16* mat_d, int ld_d, int stride_d, float* scales_a, float* scales_b, uint32_t num_problems,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
    {
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }

    constexpr uint32_t block_k = 128;

    // Select the best configuration based on shape dimensions
    auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size]
        = deep_gemm::jit::get_best_gemm_config(shape_m, shape_n, shape_k, num_problems, num_device_sms);

    auto runtime = deep_gemm::jit::getGlobalCompiler().build(shape_n, shape_k, best_block_m, best_block_n, block_k,
        num_problems, best_num_stages, best_num_tma_multicast, deep_gemm::GemmType::StridedBatched);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    deep_gemm::runGemm(kernel, mat_a, static_cast<uint64_t>(ld_a), static_cast<uint64_t>(stride_a), mat_b,
        static_cast<uint64_t>(ld_b), static_cast<uint64_t>(stride_b), mat_d, static_cast<uint64_t>(ld_d),
        static_cast<uint64_t>(stride_d), scales_a, scales_b, shape_m, shape_n, shape_k, best_block_m, best_block_n,
        block_k, num_problems, best_num_tma_multicast, deep_gemm::GemmType::StridedBatched, stream, num_device_sms,
        static_cast<uint32_t>(best_smem_size));
}

void strided_batch_gemm_dispatch_sm89(__nv_fp8_e4m3* mat_a, int ld_a, int stride_a, __nv_fp8_e4m3* mat_b, int ld_b,
    int stride_b, __nv_bfloat16* mat_d, int ld_d, int stride_d, float* scales_a, int stride_scales_a, float* scales_b,
    uint32_t num_problems, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{

    if (num_device_sms < 0)
    {
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementAccum = float;
    using ElementBlockScale = float;
    static constexpr int Stages = 3;
    using TileShape = cutlass::gemm::GemmShape<32, 128, 128>;
    using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<ElementInput, ElementOutput, ElementAccum, ElementBlockScale,
        Stages, TileShape::kM, TileShape::kN, TileShape::kK>;
    using GemmKernel = ada_blockwise_gemm::AdaBlockwiseGemmKernel<KT>;

    static constexpr int kSmemSize = KT::kSmemSize;
    static constexpr int kThreadCount = KT::kThreadCount;
    int grid_m = (shape_m + KT::kTileM - 1) / KT::kTileM;
    int grid_n = (shape_n + KT::kTileN - 1) / KT::kTileN;
    int grid_k = num_problems;
    dim3 grid = dim3(grid_m, grid_n, grid_k);
    dim3 block = dim3(kThreadCount, 1, 1);

    int stride_scales_b = ((shape_n + 128 - 1) / 128) * ((shape_k + 128 - 1) / 128);

    if (kSmemSize > (48 << 10))
    {
        cudaFuncSetAttribute(ada_blockwise_gemm::sm89_fp8_bmm_1d1d_impl<GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
        auto result = cudaGetLastError();
        TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm89 gemm kernel cannot launch: %s", cudaGetErrorString(result));
    }
    ada_blockwise_gemm::sm89_fp8_bmm_1d1d_impl<GemmKernel><<<grid, block, kSmemSize, stream>>>(shape_m, shape_n,
        shape_k, mat_a, mat_b, mat_d, scales_a, scales_b, stride_a, stride_b, stride_d, stride_scales_a,
        stride_scales_b);
}

void strided_batch_gemm_dispatch_sm120(__nv_fp8_e4m3* mat_a, int ld_a, int stride_a, __nv_fp8_e4m3* mat_b, int ld_b,
    int stride_b, __nv_bfloat16* mat_d, int ld_d, int stride_d, float* scales_a, int stride_scales_a, float* scales_b,
    uint32_t num_problems, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, cudaStream_t stream,
    int num_device_sms = kNumDeviceSMs)
{
    if (num_device_sms < 0)
    {
        num_device_sms = kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    using ElementInput = cute::float_e4m3_t;
    using ElementOutput = cute::bfloat16_t;
    using ElementAccum = float;
    using ElementBlockScale = int32_t;
    using KT = sm120_blockscaled_gemm::SM120BlockScaledBuilder<32, 128>;
    using GemmKernel = sm120_blockscaled_gemm::SM120BlockScaledKernel<KT>;
    using Params = typename GemmKernel::Params;
    using Arguments = typename GemmKernel::Arguments;
    using ProblemShape = typename GemmKernel::ProblemShape;
    ProblemShape problem_shape = make_shape((int) shape_m, (int) shape_n, (int) shape_k, (int) num_problems);

    auto ptr_A = reinterpret_cast<ElementInput*>(mat_a);
    auto ptr_B = reinterpret_cast<ElementInput*>(mat_b);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale*>(scales_a);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale*>(scales_b);
    auto ptr_D = reinterpret_cast<ElementOutput*>(mat_d);

    typename KT::StrideA dA = make_stride(ld_a, Int<1>{}, stride_a);
    typename KT::StrideB dB = make_stride(ld_b, Int<1>{}, stride_b);
    typename KT::StrideSFA dSFA = KT::deduce_sfa_layout(problem_shape).stride();
    typename KT::StrideSFB dSFB = KT::deduce_sfb_layout(problem_shape).stride();
    typename KT::StrideD dD = make_stride(ld_d, Int<1>{}, stride_d);

    Arguments args = {ptr_A, dA, ptr_B, dB, ptr_SFA, dSFA, ptr_SFB, dSFB, ptr_D, dD};

    Params kernel_params = GemmKernel::to_underlying_arguments(problem_shape, args);
    auto kernel_ptr = &cutlass::device_kernel<GemmKernel>;

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::kSmemSize);
    auto result = cudaGetLastError();
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 gemm kernel cannot launch: %s", cudaGetErrorString(result));

    cudaLaunchConfig_t launch_config;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    launch_config.gridDim = GemmKernel::get_grid_shape(kernel_params);
    launch_config.blockDim = GemmKernel::get_block_shape();
    launch_config.dynamicSmemBytes = GemmKernel::kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;

    cudaLaunchKernelEx(&launch_config, kernel_ptr, kernel_params);

    result = cudaGetLastError();
    TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm120 gemm kernel runtime error: %s", cudaGetErrorString(result));
}

void fp8_stride_batch_gemm_run(__nv_bfloat16 const* mat_a, __nv_fp8_e4m3* fp8_mat_a, float* scales_a, int ld_a,
    int stride_a, int stride_scales_a, __nv_bfloat16 const* mat_b, __nv_fp8_e4m3* fp8_mat_b, float* scales_b, int ld_b,
    int stride_b, __nv_bfloat16* mat_d, int ld_d, int stride_d, uint32_t num_problems, uint32_t shape_m,
    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, bool internal_quantize_a = true,
    bool internal_quantize_b = true)
{
    if (shape_m == 0)
    {
        return;
    }

    if (kNumDeviceSMs < 0)
    {
        kNumDeviceSMs = tensorrt_llm::common::getMultiProcessorCount();
    }
    if (internal_quantize_a)
    {
        scale_1x128_kernel<<<kNumDeviceSMs * 8, 256, 0, stream>>>(
            fp8_mat_a, scales_a, mat_a, shape_k, shape_m * num_problems);
    }
    if (internal_quantize_b)
    {
        scale_128x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(
            fp8_mat_b, scales_b, mat_b, shape_k, shape_n * num_problems);
    }

    int arch = tensorrt_llm::common::getSMVersion();
    if (arch == 89)
    {
        strided_batch_gemm_dispatch_sm89(fp8_mat_a, ld_a, stride_a, fp8_mat_b, ld_b, stride_b, mat_d, ld_d, stride_d,
            scales_a, stride_scales_a, scales_b, num_problems, shape_m, shape_n, shape_k, stream);
        return;
    }
    if (arch == 120)
    {
        strided_batch_gemm_dispatch_sm120(fp8_mat_a, ld_a, stride_a, fp8_mat_b, ld_b, stride_b, mat_d, ld_d, stride_d,
            scales_a, stride_scales_a, scales_b, num_problems, shape_m, shape_n, shape_k, stream);
        return;
    }
    strided_batch_gemm_dispatch(fp8_mat_a, ld_a, stride_a, fp8_mat_b, ld_b, stride_b, mat_d, ld_d, stride_d, scales_a,
        scales_b, num_problems, shape_m, shape_n, shape_k, stream);
}

} // namespace kernels::fp8_blockscale_gemm

TRTLLM_NAMESPACE_END
