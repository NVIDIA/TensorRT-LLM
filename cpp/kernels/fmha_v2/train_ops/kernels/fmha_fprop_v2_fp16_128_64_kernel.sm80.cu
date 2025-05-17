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

#include "fused_multihead_attention_fprop.h"
#include "fused_multihead_attention_fprop_kernel_1xN.h"

using Kernel_traits = fmha::Kernel_traits_v2<fmha::Ampere_hmma_fp32_traits, 128, 64, 16, 1, 4, 1, 0x20u>;

template <bool IS_TRAINING>
__global__ void fmha_fprop_v2_fp16_128_64_sm80_kernel(
    Fused_multihead_attention_fprop_params params, int const total_heads)
{

    fused_multihead_attention::device_1xN<Kernel_traits, IS_TRAINING>(params, total_heads);
}

template <bool IS_TRAINING>
__global__ void fmha_fprop_v2_fp16_128_64_sm80_kernel_nl(Fused_multihead_attention_fprop_params params,
    int const num_full_heads, int const num_main_groups, int const main_group_size, int const main_steps,
    int const rest_steps)
{

    fused_multihead_attention::device_1xN<Kernel_traits, IS_TRAINING>(
        params, num_full_heads, num_main_groups, main_group_size, main_steps, rest_steps);
}

void run_fmha_v2_fp16_128_64_sm80_(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure)
{

    auto kernel = launch_params.is_training ? &fmha_fprop_v2_fp16_128_64_sm80_kernel<true>
                                            : &fmha_fprop_v2_fp16_128_64_sm80_kernel<false>;

    constexpr int smem_size = fused_multihead_attention::get_dynamic_smem_size<Kernel_traits>();

    if (smem_size >= 48 * 1024)
    {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    int const sm_count = launch_params.props->multiProcessorCount;
    int ctas_per_sm;
    FMHA_CHECK_CUDA(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size));
    int total_ctas = sm_count * ctas_per_sm;

    int const heads_total = launch_params.params.b * launch_params.params.h;
    if (configure)
    {

        using Mma_tile_p = typename Kernel_traits::Traits_p::template Mma_tile<typename Kernel_traits::Cta_tile_p>;
        constexpr size_t STEPS = Kernel_traits::Cta_tile_p::N / Kernel_traits::Cta_tile_p::M;
        constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
        constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;

        size_t heads_per_cta = ((heads_total + total_ctas - 1) / total_ctas);
        size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8;
        launch_params.elts_per_thread = heads_per_cta * elts_per_head;
        return;
    }

    dim3 grid(total_ctas);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(launch_params.params, heads_total);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

void run_fmha_v2_fp16_128_64_sm80_nl_(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure)
{

    auto kernel = launch_params.is_training ? &fmha_fprop_v2_fp16_128_64_sm80_kernel_nl<true>
                                            : &fmha_fprop_v2_fp16_128_64_sm80_kernel_nl<false>;

    constexpr int smem_size = fused_multihead_attention::get_dynamic_smem_size<Kernel_traits>();

    if (smem_size >= 48 * 1024)
    {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    int const sm_count = launch_params.props->multiProcessorCount;
    int ctas_per_sm;
    FMHA_CHECK_CUDA(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size));
    int total_ctas = sm_count * ctas_per_sm;

    if (configure)
    {
        int const heads_total = launch_params.params.b * launch_params.params.h;
        std::tie(launch_params.num_full_heads, launch_params.num_main_groups, launch_params.heads_last_wave,
            launch_params.main_steps, launch_params.rest_steps, launch_params.elts_per_thread)
            = work_dist<Kernel_traits>(launch_params.params.s, total_ctas, heads_total);
        return;
    }

    dim3 grid(total_ctas);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(launch_params.params,
        launch_params.num_full_heads, launch_params.num_main_groups, launch_params.heads_last_wave,
        launch_params.main_steps, launch_params.rest_steps);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

void run_fmha_v2_fp16_128_64_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure)
{
    if (launch_params.is_nl)
    {
        run_fmha_v2_fp16_128_64_sm80_nl_(launch_params, configure);
    }
    else
    {
        run_fmha_v2_fp16_128_64_sm80_(launch_params, configure);
    }
}
