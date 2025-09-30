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
#include <hopper/fmha_dgrad.h>
#include <hopper/fmha_dgrad_kernel_4x1.h>
#include <hopper/kernel_traits.h>
#include <philox.h>

namespace fmha
{
namespace hopper
{
namespace dgrad
{

using fmha::e4m3_t;
using fmha::e5m2_t;
// Aggressive recipe, where everything is in E4M3 except dP, so dQ, dK are mixed FP8 QGMMAs.

// P  = Q   x K'
using Traits_p = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, false, false, e4m3_t, e4m3_t, e4m3_t>;
// T  = dO  x O'
using Traits_t = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, false, false, e4m3_t, e4m3_t, e4m3_t>;
// dS = dO  x V'
using Traits_ds = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, false, false, e4m3_t, e4m3_t, e4m3_t>;
// dQ = dP  x K
using Traits_dq = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, true, false, e5m2_t, e4m3_t, e4m3_t>;
// dK = dP' x Q
using Traits_dk = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, true, false, e5m2_t, e4m3_t, e4m3_t>;
// dV = S'  x dO
using Traits_dv = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, true, false, e4m3_t, e4m3_t, e4m3_t>;

static constexpr uint32_t S_MAX = 512;
static constexpr uint32_t D = 64;
static constexpr uint32_t STEP_M = 64;
static constexpr uint32_t STEP_N = 64;
static constexpr uint32_t WARPS_M = 4;
static constexpr uint32_t WARPS_N = 1;
static constexpr uint32_t FLAGS = 0x28u;

using Kernel_traits = fmha::hopper::Kernel_traits_dgrad<Traits_p, Traits_t, Traits_ds, Traits_dq, Traits_dk, Traits_dv,
    S_MAX, D, STEP_M, STEP_N, WARPS_M, WARPS_N, FLAGS>;

namespace grad_all_e5m2
{

// P = Q    x K'
using Traits_p = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, false, false, e4m3_t, e4m3_t, e4m3_t>;
// T = dO   x O'
using Traits_t = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, false, false, e5m2_t, e4m3_t, e5m2_t>;
// dS = dO  x V'
using Traits_ds = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, false, false, e5m2_t, e4m3_t, e5m2_t>;
// dQ = dP  x K
using Traits_dq = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, true, false, e5m2_t, e4m3_t, e5m2_t>;
// dK = dP' x Q
using Traits_dk = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, true, false, e5m2_t, e4m3_t, e5m2_t>;
// dV = S'  x dO
using Traits_dv = fmha::Hopper_qgmma_fp8_fp32_traits<64, 64, 32, true, false, e4m3_t, e5m2_t, e5m2_t>;

static constexpr uint32_t S_MAX = 512;
static constexpr uint32_t D = 64;
static constexpr uint32_t STEP_M = 64;
static constexpr uint32_t STEP_N = 64;
static constexpr uint32_t WARPS_M = 4;
static constexpr uint32_t WARPS_N = 1;
static constexpr uint32_t FLAGS = 0x28u;

using Kernel_traits = fmha::hopper::Kernel_traits_dgrad<Traits_p, Traits_t, Traits_ds, Traits_dq, Traits_dk, Traits_dv,
    S_MAX, D, STEP_M, STEP_N, WARPS_M, WARPS_N, FLAGS>;

} // namespace grad_all_e5m2

template <typename Kernel_traits>
__global__ __launch_bounds__(Kernel_traits::THREADS, 2) void fmha_dgrad_fp8_sm90_kernel(
    __grid_constant__ const Fmha_dgrad_params params)
{

    extern __shared__ char smem[];

    int const bidb = blockIdx.y;
    int const bidh = blockIdx.x;
    int const tidx = threadIdx.x;

    // auto [seed, offset] = at::cuda::philox::unpack(params.philox_args);
    uint64_t seed = params.ptr_philox_unpacked[0];
    uint64_t offset = params.ptr_philox_unpacked[1];

    Dgrad_4x1<Kernel_traits> kernel(params, bidb, bidh, tidx, smem);

    kernel(seed, offset);
}

void run_fmha_dgrad_fp8_512_64_sm90(Launch_params& launch_params, bool const configure)
{

    auto& params = launch_params.params;

    auto kernel = launch_params.all_e5m2 ? fmha_dgrad_fp8_sm90_kernel<grad_all_e5m2::Kernel_traits>
                                         : fmha_dgrad_fp8_sm90_kernel<Kernel_traits>;

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

} // namespace dgrad
} // namespace hopper
} // namespace fmha
