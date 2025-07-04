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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/quantTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/layernormKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename Tf, typename T>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, T const* gamma, T const* beta, int i)
{
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (beta != nullptr)
    {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

/* Computes the layernorm https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 * normed_output <- ( (input - E[input]) / Sqrt(Var[input] + eps) ) * gamma + beta
 * input is [tokens, hidden_dim]. Mean and Variance are per-row (i.e. per-token)
 *
 * One CTA handles one row.
 *
 * with USE_DIFF_OF_SQUARES set to false:
 * First pass (loop) computes the mean.
 * Second computes the variance via Var[x] = E[(x - E[x])²].
 * Third pass computes and writes normed_output
 *
 * with USE_DIFF_OF_SQUARES set to true (may be faster but less accurate):
 * First pass (loop) computes the mean and variance via Var[x] = E[x²] - E[x]²
 * Second pass computes and writes normed_output
 *
 * QuantT is the quantized data type (e.g. int8_t, __nv_fp8_e4m3)
 *
 * USE_SHMEM controls if we cache input values into shared memory
 *
 * Optional: with dynamic scaling, the last pass doesn't write immediately but finds the
 *           amax per row. A final pass scales to int8 accordingly, and writes output to
 *           normed_output_quant.
 */
template <typename T, typename QuantT, bool USE_SHMEM, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm(T const* input, T const* gamma, T const* beta, T* normed_output, float const eps,
    int tokens, int hidden_dim, float const* clamp_ptr, float const* scale_orig_quant_per_tensor,
    float* scale_orig_quant_per_token, float* sum_per_token, QuantT* normed_output_quant, bool has_fp8_min_scaling)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using QuantT_packed_t = typename packed_as<QuantT, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    // The clamping minimum / maximum values.
    T const clamp_min = cuda_cast<T>(clamp_ptr ? clamp_ptr[0] : -FLT_MAX);
    T const clamp_max = cuda_cast<T>(clamp_ptr ? clamp_ptr[1] : FLT_MAX);

    // The quantized data type's maximum value (upper-bound).
    static constexpr float MAX_QUANT_VAL = QuantTypeStaticVals<QuantT>::MAX_VAL;
    // The minimum scaling factor (lower-bound)
    static constexpr float MIN_SCALING_FACTOR = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR;
    static constexpr float MIN_SCALING_FACTOR_RCP = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR_RCP;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    int const tidx = threadIdx.x;
    int const bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    int const n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if constexpr (USE_SHMEM)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if constexpr (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if constexpr (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if constexpr (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if constexpr (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = USE_SHMEM ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    bool const with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    bool const with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    bool const with_per_token_sum = sum_per_token != nullptr;

    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? *scale_orig_quant_per_tensor : 0.0f);
    T_scalar amax = 1e-6f;
    local_sum = 0.f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        int const index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : input[index]);
        T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            val = cuda_clamp(val, clamp_min, clamp_max);
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if constexpr (USE_SHMEM)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            val = cuda_clamp(val, clamp_min, clamp_max);
            reinterpret_cast<QuantT_packed_t*>(normed_output_quant)[index]
                = cuda_cast<QuantT_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }

        if (with_per_token_sum)
        {
            local_sum += cuda_sum<float>(cuda_cast<float_packed_t>(val));
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        float const dynamic_per_token_scale = has_fp8_min_scaling
            ? fminf(MAX_QUANT_VAL / abs_max_f, MIN_SCALING_FACTOR_RCP)
            : (MAX_QUANT_VAL / abs_max_f);
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            int const index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : input[index]);
            if constexpr (!USE_SHMEM)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<QuantT_packed_t*>(normed_output_quant)[index]
                = cuda_cast<QuantT_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = has_fp8_min_scaling
                ? cuda_max(abs_max_f / MAX_QUANT_VAL, MIN_SCALING_FACTOR)
                : abs_max_f / MAX_QUANT_VAL;
        }
    }

    if (with_per_token_sum)
    {
        float packed_sum[1] = {local_sum};
        blockReduceSumV2<float, 1>(packed_sum);
        if (tidx == 0)
        {
            sum_per_token[bidx] = packed_sum[0];
        }
    }
}

template <bool USE_DIFF_OF_SQUARES, typename T, typename QuantT>
void dispatch_layernorm_type_square_method(T const* input, T const* gamma, T const* beta, T* normed_output,
    float const eps, int tokens, int hidden_dim, float const* clamp_ptr, float const* scale_orig_quant_per_tensor,
    float* scale_orig_quant_per_token, float* sum_per_token, QuantT* normed_output_quant,
    bool const has_fp8_min_scaling, dim3 const grid, dim3 const block, size_t const shmem_size, cudaStream_t stream)
{
    // Do we use shared memory to cache intermediate results
    bool use_shmem = true;
    if (shmem_size >= (48 << 10))
    {
        cudaError_t ret = cudaFuncSetAttribute(generalLayerNorm<T, QuantT, true, USE_DIFF_OF_SQUARES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        // Use shared memory when the capacity is enough
        use_shmem = (ret == cudaSuccess);
    }

    if (use_shmem)
    {
        generalLayerNorm<T, QuantT, true, USE_DIFF_OF_SQUARES><<<grid, block, shmem_size, stream>>>(input, gamma, beta,
            normed_output, eps, tokens, hidden_dim, clamp_ptr, scale_orig_quant_per_tensor, scale_orig_quant_per_token,
            sum_per_token, normed_output_quant, has_fp8_min_scaling);
    }
    else
    {
        generalLayerNorm<T, QuantT, false, USE_DIFF_OF_SQUARES><<<grid, block, 0, stream>>>(input, gamma, beta,
            normed_output, eps, tokens, hidden_dim, clamp_ptr, scale_orig_quant_per_tensor, scale_orig_quant_per_token,
            sum_per_token, normed_output_quant, has_fp8_min_scaling);
    }
}

template <typename T, typename QuantT>
void dispatch_layernorm_type(T const* input, T const* gamma, T const* beta, T* normed_output, float const eps,
    int tokens, int hidden_dim, float const* clamp_ptr, float const* scale_orig_quant_per_tensor,
    float* scale_orig_quant_per_token, float* sum_per_token, QuantT* normed_output_quant,
    bool const has_fp8_min_scaling, dim3 const grid, dim3 const block, size_t const shmem_size, cudaStream_t stream,
    bool const use_diff_of_squares)
{
    if (use_diff_of_squares)
    {
        dispatch_layernorm_type_square_method<true>(input, gamma, beta, normed_output, eps, tokens, hidden_dim,
            clamp_ptr, scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token, normed_output_quant,
            has_fp8_min_scaling, grid, block, shmem_size, stream);
    }
    else
    {
        dispatch_layernorm_type_square_method<false>(input, gamma, beta, normed_output, eps, tokens, hidden_dim,
            clamp_ptr, scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token, normed_output_quant,
            has_fp8_min_scaling, grid, block, shmem_size, stream);
    }
}

template <typename T, typename QuantT>
void invokeGeneralLayerNorm(T* out, T const* input, T const* gamma, T const* beta, float const eps, int const tokens,
    int const hidden_dim, QuantMode quantMode, cudaStream_t stream, bool use_diff_of_squares, float const* clamp_ptr,
    float const* scale, float* dynamic_scale, float* sum_per_token, QuantT* normed_output_quant)
{
    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size = 2;
    const size_t shmem_size = hidden_dim * sizeof(T);
    bool const use_vec_type = (hidden_dim % vec_size == 0)
        && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        );

    // Enable min_scaling factor if it is fp8 row-wise per-token quantization
    bool has_fp8_min_scaling = quantMode.hasFp8RowWise();

    if (use_vec_type)
    {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_layernorm_type(reinterpret_cast<Tp const*>(input), reinterpret_cast<Tp const*>(gamma),
            reinterpret_cast<Tp const*>(beta), reinterpret_cast<Tp*>(out), eps, tokens, hidden_dim, clamp_ptr, scale,
            dynamic_scale, sum_per_token, normed_output_quant, has_fp8_min_scaling, grid, block, shmem_size, stream,
            use_diff_of_squares);
    }
    else
    {
        dispatch_layernorm_type(input, gamma, beta, out, eps, tokens, hidden_dim, clamp_ptr, scale, dynamic_scale,
            sum_per_token, normed_output_quant, has_fp8_min_scaling, grid, block, shmem_size, stream,
            use_diff_of_squares);
    }
}

#define INSTANTIATE_GENERAL_LAYERNORM(T, QuantT)                                                                       \
    template void invokeGeneralLayerNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps,       \
        const int tokens, const int hidden_dim, QuantMode quantMode, cudaStream_t stream, bool use_diff_of_squares,    \
        const float* clamp_ptr, float const* scale, float* dynamic_scale, float* sum_per_token,                        \
        QuantT* normed_output_quant);

INSTANTIATE_GENERAL_LAYERNORM(float, int8_t);
INSTANTIATE_GENERAL_LAYERNORM(half, int8_t);

#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_LAYERNORM(__nv_bfloat16, int8_t);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_GENERAL_LAYERNORM(float, __nv_fp8_e4m3);
INSTANTIATE_GENERAL_LAYERNORM(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_LAYERNORM(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

} // namespace kernels
} // namespace tensorrt_llm
