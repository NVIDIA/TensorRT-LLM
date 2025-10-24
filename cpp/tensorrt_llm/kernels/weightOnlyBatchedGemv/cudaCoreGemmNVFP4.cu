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

#include "cutlass/numeric_conversion.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmNVFP4.h"
#include <cub/cub.cuh>

namespace tensorrt_llm
{
namespace kernels
{
namespace cuda_core_gemm_nvfp4
{
template <typename InputType, typename OutputType, typename ScaleType, SizeType32 TILE_M, SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE>
__device__ void cudaCoreGemmImpl(InputType const* __restrict__ act, InputType const* __restrict__ weight,
    ScaleType const* __restrict__ scale_a, ScaleType const* __restrict__ scale_w, float const alpha,
    OutputType* __restrict__ output, SizeType32 m, SizeType32 n, SizeType32 k)
{
    using VecType = int4;

    using ScaleVecType = __nv_fp8x2_e4m3;
    using CvtInputType = typename tensorrt_llm::kernels::cutlass_kernels::TllmToCutlassTypeAdapter<InputType>::type;
    static constexpr SizeType32 step_k = static_cast<SizeType32>(128 / cutlass::sizeof_bits<CvtInputType>::value);
    static constexpr SizeType32 nvfp4_scale_granularity = 16;
    static constexpr SizeType32 step_k_scale = step_k / nvfp4_scale_granularity;
    static constexpr SizeType32 tile_k = step_k * BLOCK_SIZE;
    auto tile_id_m = static_cast<SizeType32>(blockIdx.x * TILE_M);
    auto tile_id_n = static_cast<SizeType32>(blockIdx.y * TILE_N);
    auto tid = static_cast<SizeType32>(threadIdx.x);
    float tile_a[step_k];
    float tile_w[TILE_N * step_k];
    float tile_a_scale[step_k_scale];
    float tile_w_scale[TILE_N * step_k_scale];
    float acc[TILE_M * TILE_N];

    static_assert(step_k % 4 == 0);
    using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 8>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    static constexpr SizeType32 k_cvt_count = static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
    for (SizeType32 i = 0; i < TILE_M * TILE_N; ++i)
    {
        acc[i] = 0;
    }
    act += tile_id_m * k / 2;
    weight += tile_id_n * k / 2;
    output += tile_id_m * n + tile_id_n;

    scale_a += tile_id_m * k / nvfp4_scale_granularity;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int const num_cols_sf = k / nvfp4_scale_granularity;
    int const num_sf_tiles_k = (num_cols_sf + 4 - 1) / 4;
    for (SizeType32 idx_k = tid * step_k; idx_k < k; idx_k += tile_k)
    {
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
            auto tile_w_quantized = reinterpret_cast<VecType const*>(weight + (j * k + idx_k) / 2)[0];
#pragma unroll
            for (SizeType32 cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx)
            {
                reinterpret_cast<CvtResType*>(tile_w)[j * k_cvt_count + cvt_idx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvt_idx]);
            }
        }
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
            int const row_idx = tile_id_n + j;
            int const col_idx = idx_k / nvfp4_scale_granularity;
            int const tile_offset = ((row_idx / 128) * num_sf_tiles_k + col_idx / 4) * 512;
            int const dst_idx = tile_offset + (row_idx % 32) * 16 + ((row_idx % 128) / 32) * 4 + col_idx % 4;
            auto tile_w_scale_fp8x2 = reinterpret_cast<ScaleVecType const*>(scale_w + dst_idx)[0];
            const char2 tmp = reinterpret_cast<char2 const&>(tile_w_scale_fp8x2);
            tile_w_scale[j * step_k_scale + 0] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
            tile_w_scale[j * step_k_scale + 1] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));
        }
#pragma unroll
        for (SizeType32 i = 0; i < TILE_M; ++i)
        {
            auto tile_a_quantized = reinterpret_cast<VecType const*>(act + (i * k + idx_k) / 2)[0];
#pragma unroll
            for (SizeType32 cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx)
            {
                reinterpret_cast<CvtResType*>(tile_a)[cvt_idx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvt_idx]);
            }
            auto tile_a_scale_fp8x2
                = reinterpret_cast<ScaleVecType const*>(scale_a + (i * k + idx_k) / nvfp4_scale_granularity)[0];
            const char2 tmp = reinterpret_cast<char2 const&>(tile_a_scale_fp8x2);
            tile_a_scale[0] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
            tile_a_scale[1] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));
#pragma unroll
            for (SizeType32 j = 0; j < TILE_N; ++j)
            {
#pragma unroll
                for (SizeType32 l = 0; l < step_k; ++l)
                {
                    acc[i * TILE_N + j] = fma(alpha * tile_a[l] * tile_a_scale[l / nvfp4_scale_granularity],
                        tile_w[j * step_k + l] * tile_w_scale[j * step_k_scale + l / nvfp4_scale_granularity],
                        acc[i * TILE_N + j]);
                }
            }
        }
    }

    typedef cub::WarpReduce<float> WarpReduce;

    static constexpr SizeType32 warp_size = 32;
    static constexpr SizeType32 warp_num = BLOCK_SIZE / warp_size;
    SizeType32 warp_id = tid / warp_size, lane_id = tid % warp_size;
    __shared__ float shmem[TILE_M * TILE_N * warp_num];
    __shared__ typename WarpReduce::TempStorage temp_storage[warp_num];
#pragma unroll
    for (SizeType32 mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (SizeType32 ni = 0; ni < TILE_N; ++ni)
        {
            float val = WarpReduce(temp_storage[warp_id]).Sum(acc[mi * TILE_N + ni]);
            if (lane_id == 0)
            {
                shmem[mi * TILE_N + ni + warp_id * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();
    for (SizeType32 ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE)
    {
        SizeType32 mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (SizeType32 jj = 0; jj < warp_num; ++jj)
        {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename OutputType, typename ScaleType, SizeType32 TILE_M, SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE>
__global__ void cudaCoreGemmFp4(InputType const* __restrict__ act, InputType const* __restrict__ weight,
    ScaleType const* __restrict__ scale_a, ScaleType const* __restrict__ scale_w, float const* alpha_ptr,
    OutputType* __restrict__ output, SizeType32 m, SizeType32 n, SizeType32 k)
{
    float alpha = alpha_ptr[0];
    cudaCoreGemmImpl<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>(
        reinterpret_cast<InputType const*>(act), reinterpret_cast<InputType const*>(weight),
        reinterpret_cast<ScaleType const*>(scale_a), reinterpret_cast<ScaleType const*>(scale_w), alpha,
        reinterpret_cast<OutputType*>(output), m, n, k);
}

template <typename InputType, typename OutputType, typename ScaleType, SizeType32 TILE_M, SizeType32 TILE_N,
    SizeType32 BLOCK_SIZE>
void cudaCoreGemmKernel(Params const& params, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(params.m / TILE_M, params.n / TILE_N);

    if (tensorrt_llm::common::getEnvEnablePDL())
    {
        TLLM_LOG_DEBUG("Enable PDL in fp8_gemm_plugin");
        cudaLaunchConfig_t kernelConfig = {0};
        kernelConfig.gridDim = grid;
        kernelConfig.blockDim = block;
        kernelConfig.dynamicSmemBytes = 0;
        kernelConfig.stream = stream;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        kernelConfig.attrs = attribute;
        kernelConfig.numAttrs = 1;

        if (params.scale_a && params.scale_b && params.alpha_ptr)
        {
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(&kernelConfig,
                cudaCoreGemmFp4<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>,
                reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight),
                reinterpret_cast<ScaleType const*>(params.scale_a), reinterpret_cast<ScaleType const*>(params.scale_b),
                params.alpha_ptr, reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k));
        }
    }
    else
    {
        if (params.scale_a && params.scale_b && params.alpha_ptr)
        {
            cudaCoreGemmFp4<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE><<<grid, block, 0, stream>>>(
                reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight),
                reinterpret_cast<ScaleType const*>(params.scale_a), reinterpret_cast<ScaleType const*>(params.scale_b),
                params.alpha_ptr, reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k);
        }
    }
}

template <typename InputType, typename OutputType, typename ScaleType, int TILE_M, int TILE_N, int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(Params const& params, cudaStream_t stream)
{
    constexpr int cudaCoreGemmTemplateMaxM = 16;
    if (params.m == TILE_M)
    {
        cudaCoreGemmKernel<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>(params, stream);
        return true;
    }
    if constexpr (TILE_M < cudaCoreGemmTemplateMaxM)
    {
        return cudaCoreGemmTemplateCaller<InputType, OutputType, ScaleType, TILE_M + 1, TILE_N, BLOCK_SIZE>(
            params, stream);
    }
    return false;
}

template <typename InputType, typename OutputType, typename ScaleType = float>
bool cudaCoreGemmLauncher(Params const& params, cudaStream_t stream)
{
    return cudaCoreGemmTemplateCaller<InputType, OutputType, ScaleType, 1, 2, 128>(params, stream);
}

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream)
{
    bool dispatched = true;
    if (params.n % 2 != 0)
    {
        dispatched = false;
    }
    else if (params.inputType == CUDA_R_8U)
    {
        if (params.k % 16 != 0)
        {
            // Expect k % 16 == 0 for nvfp4 scaling granularity
            dispatched = false;
        }
        else if (params.outputType == CUDA_R_16F)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp4_e2m1, half, __nv_fp8_e4m3>(params, stream);
        }
        else if (params.outputType == CUDA_R_16BF)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp4_e2m1, __nv_bfloat16, __nv_fp8_e4m3>(params, stream);
        }
        else if (params.outputType == CUDA_R_32F)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp4_e2m1, float, __nv_fp8_e4m3>(params, stream);
        }
        else
        {
            dispatched = false;
        }
    }
    else
    {
        dispatched = false;
    }

    if (!dispatched)
    {
        TLLM_LOG_WARNING(
            "tensorrt_llm::kernels::cuda_core_gemm_nvfp4::cudaCoreGemmDispatcher [NOT DISPATCHED], inputType=%d, "
            "outputType=%d, "
            "m=%d, "
            "n=%d, k=%d",
            params.inputType, params.outputType, params.m, params.n, params.k);
    }
    return dispatched;
}

} // namespace cuda_core_gemm_nvfp4
} // namespace kernels
} // namespace tensorrt_llm
