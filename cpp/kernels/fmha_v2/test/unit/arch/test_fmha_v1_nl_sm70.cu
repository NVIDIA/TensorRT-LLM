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

#include "fused_multihead_attention_kernel_1xN_noloop.h"
#include "gtest/gtest.h"
#include <cuda.h>

using Kernel_traits_nl
    = fmha::Kernel_traits_v1<fmha::Volta_hmma_fp16_traits, 512, 32, 32, 1, 1 * 8, 1, 0x08u | 0x200 /* no_loop flag */>;

static_assert(Kernel_traits_nl::CTAS_PER_HEAD == 1, "");

extern "C" __global__ void fmha_v1_fp16_512_32_sm70_kernel_nl(bert::Fused_multihead_attention_params_v1 params)
{
    fused_multihead_attention::device_1xN_nl<Kernel_traits_nl>(params);
}

void run_fmha_v1_fp16_512_32_sm70_nl(bert::Fused_multihead_attention_params_v1 const& params, cudaStream_t stream)
{

    constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
    if (smem_size >= 48 * 1024)
    {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(
            fmha_v1_fp16_512_32_sm70_kernel_nl, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    constexpr int loop_iters = (512 + 32 - 1) / 32;
    static_assert(loop_iters * 32 >= 512, "");
    dim3 grid(params.h, params.b, loop_iters);
    fmha_v1_fp16_512_32_sm70_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(
        params);
}

TEST(FMHA_v1_nl, InvalidConfig)
{
    run_fmha_v1_fp16_512_32_sm70_nl(bert::Fused_multihead_attention_params_v1{}, cudaStream_t{});
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    EXPECT_EQ(error, cudaError::cudaErrorInvalidConfiguration);
}
