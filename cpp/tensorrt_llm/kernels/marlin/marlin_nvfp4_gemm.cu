/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Single-expert Marlin NVFP4 GEMM dispatcher.
// BF16 activations + FP4 E2M1 weights + FP8 E4M3 scales.

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

#define MARLIN_DECLARE_SINGLE_EXPERT_KERNEL
#include "marlin_nvfp4.h"
#include "marlin_nvfp4_template.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace marlin
{

using namespace marlin_nvfp4_dispatch;

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// Single-expert shared-memory size (no block-meta overhead).
int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks, int prob_n, int prob_k, int num_bits,
    int group_size, int stages)
{
    int pack_factor = 32 / num_bits;
    int tb_k = th_config.thread_k;
    int tb_n = th_config.thread_n;
    int tb_m = thread_m_blocks * 16;
    int sh_a_size = stages * (tb_m * tb_k) * 2;
    int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
    int sh_red_size = tb_m * (tb_n + 8) * 2;
    int sh_bias_size = tb_n * 2;
    int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
    tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);
    int sh_s_size = get_scales_cache_size(th_config, prob_n, prob_k, num_bits, group_size, stages);
    return tmp_size + sh_a_size + sh_s_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks, int prob_n, int prob_k, int num_bits,
    int group_size, int stages, int max_shared_mem)
{
    if (!is_config_feasible(th_config, prob_n, prob_k))
        return false;
    return get_kernel_cache_size(th_config, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages)
        <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(int thread_m_blocks, int thread_n_blocks, int thread_k_blocks, bool m_block_size_8,
    int group_blocks, int threads, int stages)
{
#define MARLIN_KERNEL_MATCH(T, M, N, K, M8)                                                                            \
    (threads == (T) && thread_m_blocks == (M) && thread_n_blocks == (N) && thread_k_blocks == (K)                      \
        && m_block_size_8 == (M8) && stages == 4 && group_blocks == 1)
#define MARLIN_KERNEL_IF(T, M, N, K, M8)                                                                               \
    if (MARLIN_KERNEL_MATCH(T, M, N, K, M8))                                                                           \
        return Marlin<nv_bfloat16, T, M, N, K, M8, 4, 1>;

    MARLIN_KERNEL_IF(256, 1, 8, 8, true)
    MARLIN_KERNEL_IF(128, 1, 8, 4, true)
    MARLIN_KERNEL_IF(128, 1, 4, 8, true)
    MARLIN_KERNEL_IF(256, 1, 8, 8, false)
    MARLIN_KERNEL_IF(128, 1, 8, 4, false)
    MARLIN_KERNEL_IF(128, 1, 4, 8, false)
    MARLIN_KERNEL_IF(256, 2, 16, 4, false)
    MARLIN_KERNEL_IF(128, 2, 8, 4, false)
    MARLIN_KERNEL_IF(128, 2, 4, 8, false)
    MARLIN_KERNEL_IF(256, 3, 16, 4, false)
    MARLIN_KERNEL_IF(128, 3, 8, 4, false)
    MARLIN_KERNEL_IF(128, 3, 4, 8, false)
    MARLIN_KERNEL_IF(256, 4, 16, 4, false)
    MARLIN_KERNEL_IF(128, 4, 8, 4, false)
    MARLIN_KERNEL_IF(128, 4, 4, 8, false)

#undef MARLIN_KERNEL_MATCH
#undef MARLIN_KERNEL_IF
    return MarlinDefault;
}

exec_config_t determine_exec_config(int prob_m, int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, int stages, int max_shared_mem, int sms)
{
    exec_config_t exec_cfg{1, {-1, -1, -1}};
    thread_config_t const* cfgs = thread_m_blocks > 1 ? kLargeBatchConfigs : kSmallBatchConfigs;
    int cfg_count = thread_m_blocks > 1 ? kLargeBatchConfigCount : kSmallBatchConfigCount;

    for (int i = 0; i < cfg_count; i++)
    {
        thread_config_t th = cfgs[i];
        if (!is_valid_config(th, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages, max_shared_mem - 512))
            continue;
        int group_blocks = group_size == -1 ? -1 : group_size / 16;
        auto kernel = get_marlin_kernel(
            thread_m_blocks, th.thread_n / 16, th.thread_k / 16, m_block_size_8, group_blocks, th.num_threads, stages);
        if (kernel == MarlinDefault)
            continue;
        return {1, th};
    }
    return exec_cfg;
}

void marlin_mm_nvfp4(void const* A, void const* B, void* C, void* C_tmp, void const* b_s, void const* g_s, int prob_m,
    int prob_n, int prob_k, int* locks, int num_groups, int group_size, bool use_fp32_reduce, int dev,
    cudaStream_t stream)
{
    constexpr int num_bits = 4;

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    int4 const* A_ptr = (int4 const*) A;
    int4 const* B_ptr = (int4 const*) B;
    int4* C_ptr = (int4*) C;
    int4* C_tmp_ptr = (int4*) C_tmp;
    int4 const* b_s_ptr = (int4 const*) b_s;
    uint16_t const* g_s_ptr = (uint16_t const*) g_s;

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    int stages = 4;
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    int max_par_val = 16;
    if (prob_n <= 4096)
        max_par_val = 16 * 8;
    int max_shared_mem_new = max_shared_mem;
    int rest_m = prob_m;
    int max_thread_m_blocks = 4;
    int lda = prob_k;

    while (rest_m)
    {
        int par_count = std::min(rest_m / (max_thread_m_blocks * 16), max_par_val);
        int prob_m_split = par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

        int thread_m_blocks = std::min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
        bool m_block_size_8 = prob_m_split <= 8;

        exec_config_t exec_cfg = determine_exec_config(prob_m_split, prob_n, prob_k, thread_m_blocks, m_block_size_8,
            num_bits, group_size, stages, max_shared_mem, sms);
        thread_config_t thread_tfg = exec_cfg.tb_cfg;

        if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1)
        {
            max_thread_m_blocks--;
            continue;
        }

        if (thread_tfg.thread_k == -1)
        {
            break;
        }

        // Small wave optimization
        if (thread_tfg.thread_n != -1)
        {
            if (prob_n / thread_tfg.thread_n * div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <= sms)
            {
                if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages,
                        max_shared_mem_new))
                {
                    thread_tfg = {128, 64, 128};
                    exec_cfg = {1, thread_tfg};
                }
            }
        }

        int num_threads = thread_tfg.num_threads;
        int thread_k = thread_tfg.thread_k;
        int thread_n = thread_tfg.thread_n;
        int blocks = sms * exec_cfg.blocks_per_sm;
        if (exec_cfg.blocks_per_sm > 1)
            max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

        int thread_k_blocks = thread_k / 16;
        int thread_n_blocks = thread_n / 16;

        auto kernel = get_marlin_kernel(
            thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, group_blocks, num_threads, stages);

        if (kernel == MarlinDefault)
        {
            break;
        }

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new);

        // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr,
        nullptr,  // b_bias
        nullptr,  // a_scales
        b_s_ptr, g_s_ptr,
        nullptr,  // zp
        nullptr,  // g_idx
        num_groups, prob_m_split, prob_n, prob_k, lda, locks,
        false,  // has_bias
        false,  // use_atomic_add
        use_fp32_reduce, max_shared_mem_new);
        // clang-format on

        A_ptr += prob_m_split * (lda / 8);
        C_ptr += prob_m_split * (prob_n / 8);
        rest_m -= prob_m_split;
    }
}

// Explicit template instantiations for BF16 + NVFP4 Marlin kernels.
// clang-format off
template __global__ void Marlin<nv_bfloat16, 256, 1, 8, 8, true,  4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, true,  4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, true,  4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 1, 8, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 2, 16, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 2, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 2, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 3, 16, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 3, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 3, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 4, 16, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 4, 8, 4, false, 4, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 4, 4, 8, false, 4, 1>( MARLIN_KERNEL_PARAMS );

// clang-format on

// =========================================================================
// FP4 E2M1 -> BF16 dequantization kernel for activations.
// Each byte contains 2 FP4 values. Block scales are FP8 E4M3 (swizzled).
// output[i] = fp4_to_bf16(act[i]) * block_scale[i/16] * global_scale
// =========================================================================
__global__ void dequant_fp4_act_kernel(uint8_t const* __restrict__ act_fp4, // [M, K/2] packed FP4
    uint8_t const* __restrict__ act_sf,                                     // FP8 E4M3 block scales (swizzled)
    float const* __restrict__ alpha,                                        // global scale
    nv_bfloat16* __restrict__ out,                                          // [M, K] BF16 output
    int M, int K)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = M * (K / 2);
    if (idx >= total_pairs)
        return;

    float global_s = *alpha;
    int row = idx / (K / 2);
    int col_pair = idx % (K / 2);

    uint8_t packed = act_fp4[idx];

    // Unpack two FP4 E2M1 values (low nibble first)
    auto fp4_to_float = [](uint8_t nibble) -> float
    {
        // FP4 E2M1: 1 sign + 2 exponent + 1 mantissa
        uint8_t sign = (nibble >> 3) & 1;
        uint8_t exp = (nibble >> 1) & 0x3;
        uint8_t mant = nibble & 1;
        float val;
        if (exp == 0)
        {
            // subnormal: (-1)^s * 0.mantissa * 2^(1-bias) = (-1)^s * mant * 0.5
            val = mant * 0.5f;
        }
        else
        {
            // normal: (-1)^s * 1.mantissa * 2^(exp-bias), bias=1
            val = (1.0f + mant * 0.5f) * (float) (1 << (exp - 1));
        }
        return sign ? -val : val;
    };

    float v0 = fp4_to_float(packed & 0x0F);
    float v1 = fp4_to_float((packed >> 4) & 0x0F);

    // Block scale: one FP8 E4M3 per 16 FP4 elements = per 8 bytes
    // The scale layout is swizzled 128x4 — for now use linear indexing
    // as a reasonable approximation. TODO: handle swizzled layout properly.
    int elem0 = col_pair * 2;
    int scale_idx = row * (K / 16) + elem0 / 16;
    uint8_t sf_byte = act_sf[scale_idx];
    // FP8 E4M3 -> float: reinterpret as __nv_fp8_e4m3
    __nv_fp8_e4m3 sf_fp8 = *reinterpret_cast<__nv_fp8_e4m3 const*>(&sf_byte);
    float sf = float(sf_fp8);

    float scale = sf * global_s;
    int out_idx = row * K + col_pair * 2;
    out[out_idx] = __float2bfloat16(v0 * scale);
    out[out_idx + 1] = __float2bfloat16(v1 * scale);
}

} // namespace marlin

namespace marlin_nvfp4
{

void dequantFp4Activations(
    void const* act_fp4, void const* act_sf, float const* alpha, void* act_bf16, int m, int k, cudaStream_t stream)
{

    int total_pairs = m * (k / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    ::marlin::dequant_fp4_act_kernel<<<blocks, threads, 0, stream>>>(
        (uint8_t const*) act_fp4, (uint8_t const*) act_sf, alpha, (nv_bfloat16*) act_bf16, m, k);
}

void marlinNvfp4Gemm(void const* act_bf16, void const* weight, void* output, void* C_tmp, void const* weight_sf,
    void const* global_scale_bf16, int m, int n, int k, int* workspace, int num_groups, int group_size,
    bool use_fp32_reduce, cudaStream_t stream)
{

    int dev;
    cudaGetDevice(&dev);

    ::marlin::marlin_mm_nvfp4(act_bf16, weight, output, C_tmp, weight_sf, global_scale_bf16, m, n, k, workspace,
        num_groups, group_size, use_fp32_reduce, dev, stream);
}

} // namespace marlin_nvfp4
