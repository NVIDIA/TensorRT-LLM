/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "fused_multihead_attention_dgrad_kernel_1xN_reload.h"
#include "fused_multihead_attention_fprop.h"

using Kernel_traits = fmha::Kernel_traits_v2<fmha::Ampere_hmma_fp32_traits, 128, 64, 16, 1, 4, 1, 0x28u>;
static_assert(!Kernel_traits::HEADS_INTERLEAVED);

extern "C" __global__ void fmha_dgrad_v2_fp16_128_64_sm80_kernel(Fused_multihead_attention_fprop_params params)
{
    fused_multihead_attention::compute_dv_1xN<Kernel_traits>(params);
    fused_multihead_attention::compute_dq_dk_1xN<Kernel_traits>(params);
}

void run_fmha_dgrad_v2_fp16_128_64_sm80(Fused_multihead_attention_fprop_params const& params, cudaStream_t stream)
{

    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_o = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

    using Smem_tile_s = Smem_tile_mma_transposed<fmha::Ampere_hmma_fp32_traits, Kernel_traits::Cta_tile_p>;
    constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;
    static_assert(smem_size_s == 16 * 128 * 2);
    static_assert(smem_size_o == 16 * 64 * 4 * Kernel_traits::Cta_tile_p::WARPS_N);

    constexpr int smem_size_dv = smem_size_s + 2 * smem_size_q + smem_size_v + smem_size_softmax;
    constexpr int smem_size_dq_dk = smem_size_s + smem_size_o + smem_size_q + smem_size_v;
    constexpr int smem_size = std::max(smem_size_dv, smem_size_dq_dk);

    if (smem_size >= 48 * 1024)
    {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(
            fmha_dgrad_v2_fp16_128_64_sm80_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    dim3 grid(params.h, params.b);
    fmha_dgrad_v2_fp16_128_64_sm80_kernel<<<grid, Kernel_traits::THREADS, smem_size, stream>>>(params);
}
