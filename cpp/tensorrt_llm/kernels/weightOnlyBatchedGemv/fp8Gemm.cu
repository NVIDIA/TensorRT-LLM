/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/fp8Gemm.h"
#include <cub/cub.cuh>

namespace tensorrt_llm
{
namespace kernels
{
namespace fp8_gemm
{
template <typename InputType, typename OutputType, SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
__global__ void fp8Gemm(InputType const* __restrict__ act, InputType const* __restrict__ weight, float alpha,
    OutputType* __restrict__ output, SizeType32 m, SizeType32 n, SizeType32 k)
{
    using VecType = int4;
    static constexpr SizeType32 kStepK = static_cast<SizeType32>(128 / (8 * sizeof(InputType)));
    static constexpr SizeType32 kTileK = kStepK * BLOCK_SIZE;
    auto tileIdM = static_cast<SizeType32>(blockIdx.x * TILE_M);
    auto tileIdN = static_cast<SizeType32>(blockIdx.y * TILE_N);
    auto tid = static_cast<SizeType32>(threadIdx.x);
    float tile_a[kStepK], tile_w[TILE_N * kStepK];
    float acc[TILE_M * TILE_N];

    static_assert(kStepK % 4 == 0);
    using CvtInputType
        = std::conditional_t<std::is_same_v<InputType, __nv_fp8_e4m3>, cutlass::float_e4m3_t, cutlass::float_e5m2_t>;
    using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 4>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    static constexpr SizeType32 kCvtCount = static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
    for (SizeType32 i = 0; i < TILE_M * TILE_N; ++i)
    {
        acc[i] = 0;
    }
    act += tileIdM * k;
    weight += tileIdN * k;
    output += tileIdM * n + tileIdN;
    for (SizeType32 idxK = tid * kStepK; idxK < k; idxK += kTileK)
    {
#pragma unroll
        for (SizeType32 i = 0; i < TILE_N; ++i)
        {
            auto tile_w_quantized = reinterpret_cast<VecType const*>(weight + i * k + idxK)[0];
#pragma unroll
            for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_w)[i * kCvtCount + cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvtIdx]);
            }
        }
#pragma unroll
        for (SizeType32 i = 0; i < TILE_M; ++i)
        {
            auto tile_a_quantized = reinterpret_cast<VecType const*>(act + i * k + idxK)[0];
#pragma unroll
            for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_a)[cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvtIdx]);
            }
#pragma unroll
            for (SizeType32 j = 0; j < TILE_N; ++j)
            {
#pragma unroll
                for (SizeType32 l = 0; l < kStepK; ++l)
                {
                    acc[i * TILE_N + j] = fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
                }
            }
        }
    }

    typedef cub::WarpReduce<float> WarpReduce;

    static constexpr SizeType32 kWarpSize = 32;
    static constexpr SizeType32 kWarpNum = BLOCK_SIZE / kWarpSize;
    SizeType32 warpId = tid / kWarpSize, laneId = tid % kWarpSize;
    __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
    for (SizeType32 mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (SizeType32 ni = 0; ni < TILE_N; ++ni)
        {
            float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
            if (laneId == 0)
            {
                shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (SizeType32 ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE)
    {
        SizeType32 mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (SizeType32 jj = 0; jj < kWarpNum; ++jj)
        {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val * alpha);
    }
}

template <typename InputType, typename OutputType, SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
void fp8GemmKernel(Params& params, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(params.m / TILE_M, params.n / TILE_N);
    fp8Gemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight), params.alpha,
        reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k);
}

template <typename InputType, typename OutputType>
void fp8GemmLauncher(Params& params, cudaStream_t stream)
{
#define DISPATCH(TargetM, TILE_M, TILE_N, BLOCK_SIZE)                                                                  \
    if (params.m == TargetM)                                                                                           \
    {                                                                                                                  \
        fp8GemmKernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(params, stream);                              \
        return;                                                                                                        \
    }
    DISPATCH(1, 1, 2, 128);
    DISPATCH(2, 2, 2, 128);
    DISPATCH(3, 3, 2, 128);
    DISPATCH(4, 4, 2, 128);
#undef DISPATCH
}

template void fp8GemmLauncher<__nv_fp8_e4m3, float>(Params& params, cudaStream_t stream);
template void fp8GemmLauncher<__nv_fp8_e4m3, half>(Params& params, cudaStream_t stream);
template void fp8GemmLauncher<__nv_fp8_e4m3, __nv_bfloat16>(Params& params, cudaStream_t stream);
} // namespace fp8_gemm
} // namespace kernels
} // namespace tensorrt_llm
