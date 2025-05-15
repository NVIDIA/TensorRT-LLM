/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "fused_multihead_flash_attention_kernel_noloop.h"
#include "gtest/gtest.h"
#include <cuda.h>

using Kernel_traits_nl
    = fmha::Kernel_traits_v2<fmha::Ampere_hmma_fp16_traits, 16, 160, 64, 4, 1, 1, 0x07u | 0x200 /* no_loop flag */>;

extern "C" __global__ void fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl(
    bert::Fused_multihead_attention_params_v2 params)
{
    fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl>(params);
}

void run_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_nl(
    bert::Fused_multihead_attention_params_v2 const& params, cudaStream_t stream)
{

    constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
    if (smem_size >= 48 * 1024)
    {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    // runtime q_loop_iters
    int loop_iters = (params.s + 64 - 1) / 64;
    dim3 grid(params.h, params.b, loop_iters);
    fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl<<<grid, Kernel_traits_nl::THREADS,
        Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
}

TEST(FMHA_v2_FlashAttention_nl, InvalidConfig)
{
    run_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_nl(bert::Fused_multihead_attention_params_v2{}, cudaStream_t{});
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    EXPECT_EQ(error, cudaError::cudaErrorInvalidConfiguration);
}
