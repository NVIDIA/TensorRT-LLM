/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "swigluBlockFp8.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_VG_NAMESPACE_BEGIN

namespace swiglu_block_fp8
{
namespace
{

constexpr int kWarpsPerBlock = 4;
constexpr int kWarpSize = 32;

__global__ void visualGenSwiGLUBlockFp8Kernel(__nv_bfloat16 const* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output, int32_t* __restrict__ scales, int const rows, int const width,
    int64_t const paddedRows)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    int const lane = threadIdx.x % kWarpSize;
    int const word = blockIdx.x;
    int const col = word * kColsPerWarp + lane * kElemsPerLane;
    int64_t const row
        = (static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * kWarpsPerBlock + threadIdx.x / kWarpSize;

    uint32_t packed = 0;
    if (row < rows)
    {
        union Bf16Pack
        {
            double4 data;
            __nv_bfloat16 values[kElemsPerLane];
        };

        Bf16Pack gate{};
        Bf16Pack up{};
        if (col < width)
        {
            auto const* start = input + row * 2 * width + col;
            gate.data = *reinterpret_cast<double4 const*>(start);
            up.data = *reinterpret_cast<double4 const*>(start + width);
        }

        __nv_bfloat162 activated[kElemsPerLane / 2];
        __nv_bfloat162 maxPair = __float2bfloat162_rn(0.0F);
#pragma unroll
        for (int pair = 0; pair < kElemsPerLane / 2; ++pair)
        {
            float const g0 = static_cast<float>(gate.values[2 * pair]);
            float const g1 = static_cast<float>(gate.values[2 * pair + 1]);
            float const y0 = (g0 / (1.0F + __expf(-g0))) * static_cast<float>(up.values[2 * pair]);
            float const y1 = (g1 / (1.0F + __expf(-g1))) * static_cast<float>(up.values[2 * pair + 1]);
            // Quantize the rounded BF16 activation, including the reduction for its scale.
            activated[pair] = __floats2bfloat162_rn(y0, y1);
            maxPair = __hmax2(maxPair, __habs2(activated[pair]));
        }

        // Eight lanes cover one quantization group. Each warp writes one four-scale word.
        float amax = fmaxf(static_cast<float>(__low2bfloat16(maxPair)), static_cast<float>(__high2bfloat16(maxPair)));
#pragma unroll
        for (int offset = 4; offset > 0; offset /= 2)
        {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, offset, 8));
        }
        amax = fmaxf(amax, 1e-10F);
        float reciprocalFp8Max;
        asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocalFp8Max) : "f"(448.0F));
        uint32_t const scale = __nv_cvt_float_to_e8m0(amax * reciprocalFp8Max, __NV_SATFINITE, cudaRoundPosInf);
        float const quantScale = scale == 0 ? 1.0F : exp2f(static_cast<float>(127U) - static_cast<float>(scale));

        union Fp8Pack
        {
            float4 data;
            __nv_fp8x2_e4m3 pairs[kElemsPerLane / 2];
        };

        Fp8Pack quantized;
#pragma unroll
        for (int pair = 0; pair < kElemsPerLane / 2; ++pair)
        {
            float2 y = __bfloat1622float2(activated[pair]);
            y.x *= quantScale;
            y.y *= quantScale;
            quantized.pairs[pair] = __nv_fp8x2_e4m3(y);
        }
        if (col < width)
        {
            *reinterpret_cast<float4*>(output + row * width + col) = quantized.data;
        }

#pragma unroll
        for (int group = 0; group < kGroupsPerWord; ++group)
        {
            uint32_t const byte = __shfl_sync(0xFFFFFFFFU, scale, group * 8);
            if (word * kColsPerWarp + group * kGroupWidth < width)
            {
                packed |= byte << (8 * group);
            }
        }
    }
    if (lane == 0 && row < paddedRows)
    {
        scales[static_cast<int64_t>(word) * paddedRows + row] = static_cast<int32_t>(packed);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace

void launchSwiGLUBlockFp8(
    void const* input, void* output, int32_t* scales, int rows, int width, int64_t paddedRows, cudaStream_t stream)
{
    int64_t const rowBlocks = paddedRows / kWarpsPerBlock;
    constexpr int kMaxGridY = 65535;
    auto const gridZ = static_cast<unsigned int>((rowBlocks + kMaxGridY - 1) / kMaxGridY);
    dim3 const grid(static_cast<unsigned int>(1 + (width - 1) / kColsPerWarp),
        static_cast<unsigned int>((rowBlocks + gridZ - 1) / gridZ), gridZ);
    common::launchWithPdlWhenEnabled("visualGenSwiGLUBlockFp8Kernel", visualGenSwiGLUBlockFp8Kernel, grid,
        dim3(kWarpsPerBlock * kWarpSize), 0, stream, static_cast<__nv_bfloat16 const*>(input),
        static_cast<__nv_fp8_e4m3*>(output), scales, rows, width, paddedRows);
}

} // namespace swiglu_block_fp8

TRTLLM_VG_NAMESPACE_END
