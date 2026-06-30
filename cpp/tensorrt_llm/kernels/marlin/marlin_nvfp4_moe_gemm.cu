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

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include "marlin_nvfp4.h"
#include "marlin_nvfp4_moe_template.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace marlin_moe_wna16
{

using namespace marlin_nvfp4_dispatch;

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// MoE shared-memory size includes block-meta overhead for sorted_token_ids.
int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks, int prob_n, int prob_k, int num_bits,
    int group_size, int stages)
{
    int pack_factor = 32 / num_bits;
    int tb_k = th_config.thread_k;
    int tb_n = th_config.thread_n;
    int tb_m = thread_m_blocks * 16;
    int sh_block_meta_size = tb_m * 16;
    int sh_a_size = stages * (tb_m * tb_k) * 2;
    int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
    int sh_red_size = tb_m * (tb_n + 8) * 2;
    int sh_bias_size = tb_n * 2;
    int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
    tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);
    int sh_s_size = get_scales_cache_size(th_config, prob_n, prob_k, num_bits, group_size, stages);
    return tmp_size + sh_a_size + sh_s_size + sh_block_meta_size;
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

// MoE config selection with occupancy-based multi-block logic.
exec_config_t determine_exec_config(int prob_m, int prob_n, int prob_k, int num_experts, int top_k, int thread_m_blocks,
    bool m_block_size_8, int num_bits, int group_size, int stages, int max_shared_mem, int sms)
{
    exec_config_t exec_cfg{1, {-1, -1, -1}};
    thread_config_t const* cfgs = thread_m_blocks > 1 ? kLargeBatchConfigs : kSmallBatchConfigs;
    int cfg_count = thread_m_blocks > 1 ? kLargeBatchConfigCount : kSmallBatchConfigCount;

    int count = 0;
    constexpr int device_max_reg_size = 255 * 1024;
    int group_blocks = group_size == -1 ? -1 : (group_size / 16);

    for (int i = 0; i < cfg_count; i++)
    {
        thread_config_t th = cfgs[i];
        if (!is_valid_config(th, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages, max_shared_mem - 512))
            continue;

        int cache_size = get_kernel_cache_size(th, thread_m_blocks, prob_n, prob_k, num_bits, group_size, stages);

        auto kernel = get_marlin_kernel(
            thread_m_blocks, th.thread_n / 16, th.thread_k / 16, m_block_size_8, group_blocks, th.num_threads, stages);
        if (kernel == MarlinDefault)
            continue;

        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, kernel);
        int reg_size = std::max(attr.numRegs, 1) * th.num_threads * 4;
        int allow_count = std::min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1536));
        if (thread_m_blocks == 1)
            allow_count = std::max(std::min(allow_count, 4), 1);
        else
            allow_count = std::max(std::min(allow_count, 2), 1);

        if (prob_n / th.thread_n * prob_m * top_k * 4 < sms * allow_count)
            allow_count = std::max(prob_n / th.thread_n * prob_m * top_k * 4 / sms, 1);

        if (allow_count > count)
        {
            count = allow_count;
            exec_cfg = {count, th};
        };
    }
    return exec_cfg;
}

void marlin_mm_moe_nvfp4(void const* A, void const* B, void* C, void* C_tmp, void const* b_s, void const* g_s,
    void const* sorted_token_ids, void const* expert_ids, void const* num_tokens_past_padded, void const* topk_weights,
    int moe_block_size, int num_experts, int top_k, bool mul_topk_weights, int prob_m, int prob_n, int prob_k,
    int* locks, int num_groups, int group_size, bool use_fp32_reduce, bool use_atomic_add, int dev, cudaStream_t stream)
{

    constexpr int num_bits = 4;

    int thread_m_blocks = div_ceil(moe_block_size, 16);
    bool m_block_size_8 = moe_block_size == 8;
    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int stages = 4;
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    exec_config_t exec_cfg = determine_exec_config(prob_m, prob_n, prob_k, num_experts, top_k, thread_m_blocks,
        m_block_size_8, num_bits, group_size, stages, max_shared_mem, sms);
    thread_config_t thread_tfg = exec_cfg.tb_cfg;

    if (thread_tfg.thread_k == -1)
        return;

    int num_threads = thread_tfg.num_threads;
    int thread_k = thread_tfg.thread_k;
    int thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
        max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    auto kernel = get_marlin_kernel(
        thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, group_blocks, num_threads, stages);

    if (kernel == MarlinDefault)
    {
        TLLM_LOG_ERROR(
            "xuantengh debug error: kernel is MarlinDefault, cannot find corresponding instantiated kernel for threads "
            "= %d, "
            "thread_n_blocks = %d, thread_k_blocks = %d, m_block_size_8 = %d, group_blocks = %d, num_threads = %d, "
            "stages = %d",
            num_threads, thread_n_blocks, thread_k_blocks, m_block_size_8, group_blocks, num_threads, stages);
        return;
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);

    int4 const* A_ptr = (int4 const*) A;
    int4 const* B_ptr = (int4 const*) B;
    int4* C_ptr = (int4*) C;
    int4* C_tmp_ptr = (int4*) C_tmp;
    int4 const* b_s_ptr = (int4 const*) b_s;
    uint16_t const* g_s_ptr = (uint16_t const*) g_s;
    int32_t const* sorted_token_ids_ptr = (int32_t const*) sorted_token_ids;
    int32_t const* expert_ids_ptr = (int32_t const*) expert_ids;
    int32_t const* num_tokens_past_padded_ptr = (int32_t const*) num_tokens_past_padded;
    float const* topk_weights_ptr = (float const*) topk_weights;

    // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr,
      nullptr,  // b_bias
      nullptr,  // a_scales
      b_s_ptr, g_s_ptr,
      nullptr,  // zp
      nullptr,  // g_idx
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, num_groups,
      prob_m, prob_n, prob_k, locks,
      false,  // has_bias
      use_atomic_add, use_fp32_reduce);
    // clang-format on
}

// Explicit template instantiations for BF16 + NVFP4 MoE Marlin kernels.
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

} // namespace marlin_moe_wna16

namespace marlin_nvfp4
{

void marlinNvfp4MoeGemmDispatcher(void const* A, void const* B, void* C, void* C_tmp, void const* b_scales,
    void const* global_scale, void const* sorted_token_ids, void const* expert_ids, void const* num_tokens_past_padded,
    void const* topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int prob_m, int prob_n, int prob_k,
    void* workspace, int num_groups, int group_size, bool use_fp32_reduce, bool use_atomic_add, cudaDataType_t outType,
    cudaStream_t stream)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(
        sm >= 90 && sm < 100, "Marlin NVFP4 MoE GEMM is only supported on Hopper (SM 9.x); current SM = %d", sm);

    int dev;
    cudaGetDevice(&dev);

    int num_experts = 1; // Not used in kernel dispatch, only in config selection

    ::marlin_moe_wna16::marlin_mm_moe_nvfp4(A, B, C, C_tmp, b_scales, global_scale, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, num_experts, top_k, mul_topk_weights, prob_m, prob_n,
        prob_k, (int*) workspace, num_groups, group_size, use_fp32_reduce, use_atomic_add, dev, stream);
}

} // namespace marlin_nvfp4
