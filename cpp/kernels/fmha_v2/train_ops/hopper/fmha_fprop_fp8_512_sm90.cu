/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <fmha/gmem_tile_qkv_packed.h>
#include <fmha/hopper/compute_tile.h>
#include <fmha/hopper/gmem_tile_o_packed.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/hopper/smem_tile.h>
#include <fused_multihead_attention_kernel.h>
#include <fused_multihead_attention_utils.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <fmha/numeric_types.h>
#include <fmha/traits.h>
#include <hopper/fmha_fprop.h>
#include <hopper/fmha_fprop_kernel_4x1.h>
#include <hopper/kernel_traits.h>
#include <philox.h>

namespace fmha
{
namespace hopper
{
namespace fprop
{

// We encounter three different M.N.K GEMM configs. We specify separate GMMA traits.

// Type 1: STEP.S.D - as in fprop.
using Traits_p = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, false, false>;

// FP8 Recipe: everything e4m3 in fprop.
using A_type = fmha::e4m3_t;
using Traits_o = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, true, false, A_type>;

static constexpr uint32_t S_MAX = 512;
static constexpr uint32_t D = 64;
static constexpr uint32_t STEP_M = 64;
static constexpr uint32_t STEP_N = 64;
static constexpr uint32_t WARPS_M = 4;
static constexpr uint32_t WARPS_N = 1;
static constexpr uint32_t FLAGS = 0x28u;

using Kernel_traits
    = fmha::hopper::Kernel_traits_fprop<Traits_p, Traits_o, S_MAX, D, STEP_M, STEP_N, WARPS_M, WARPS_N, FLAGS>;

template <bool IS_TRAINING>
__global__ __launch_bounds__(Kernel_traits::THREADS, 3) void fmha_fprop_fp8_sm90_kernel(
    __grid_constant__ const Fmha_fprop_params params)
{

    extern __shared__ char smem[];

    int const bidb = blockIdx.y;
    int const bidh = blockIdx.x;
    int const tidx = threadIdx.x;

    Fprop_4x1<Kernel_traits, IS_TRAINING> kernel(params, bidb, bidh, tidx, smem);
    auto [seed, offset] = at::cuda::philox::unpack(params.philox_args);

    kernel(seed, offset);
}

void run_fmha_fprop_fp8_512_64_sm90(Launch_params& launch_params, bool const configure)
{

    auto& params = launch_params.params;
    auto kernel = launch_params.is_training ? &fmha_fprop_fp8_sm90_kernel<true> : &fmha_fprop_fp8_sm90_kernel<false>;

    if (configure)
    {
        params.template set_strides<Kernel_traits>();
        size_t elts_per_cta = params.s * params.s;
        launch_params.elts_per_thread = (elts_per_cta + Kernel_traits::THREADS - 1) / Kernel_traits::THREADS;
        return;
    }

    constexpr int SMEM_BYTES = Kernel_traits::SMEM_BYTES;
    if (SMEM_BYTES >= 48 * 1024)
    {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    }
    // printf("SMEM %d\n", SMEM_BYTES);

    dim3 grid(params.h, params.b, 1);

    kernel<<<grid, Kernel_traits::THREADS, SMEM_BYTES, launch_params.stream>>>(params);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

} // namespace fprop
} // namespace hopper
} // namespace fmha
