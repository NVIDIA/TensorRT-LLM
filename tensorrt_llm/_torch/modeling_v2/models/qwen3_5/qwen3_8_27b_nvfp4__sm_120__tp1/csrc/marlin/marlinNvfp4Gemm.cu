/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define MARLIN_NAMESPACE_NAME marlin
#endif

#define MARLIN_DECLARE_SINGLE_EXPERT_KERNEL
#include "marlinNvfp4.h"
#include "marlinNvfp4Template.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

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

// Deeper-pipeline single-m-block 16-row kernels (decode rows 9-16), see get_m16_launch.
MarlinFuncPtr get_m16_kernel(thread_config_t const& th, int stages)
{
#define MARLIN_M16_IF(T, N, K, S)                                                                                      \
    if (th.num_threads == (T) && th.thread_n == (N) *16 && th.thread_k == (K) *16 && stages == (S))                    \
        return Marlin<nv_bfloat16, T, 1, N, K, false, S, 1>;
    MARLIN_M16_IF(128, 4, 8, 4)
    MARLIN_M16_IF(128, 4, 8, 6)
    MARLIN_M16_IF(128, 4, 8, 8)
    MARLIN_M16_IF(128, 8, 4, 4)
    MARLIN_M16_IF(128, 8, 4, 6)
    MARLIN_M16_IF(128, 8, 4, 8)
    MARLIN_M16_IF(256, 8, 8, 4)
    MARLIN_M16_IF(256, 8, 8, 6)
#undef MARLIN_M16_IF
    return MarlinDefault;
}

struct m16_launch_t
{
    thread_config_t th;
    int blocks;
    int stages;
};

struct m16_table_entry_t
{
    int prob_n;
    int prob_k;
    m16_launch_t launch;
};

// Decode rows 9-16 (one 16-row m block) launch shapes per (N, K) of the tuned decode GEMM shapes on
// 188-SM SM120 GPUs (launch hint kMarlinNvfp4LaunchHintDecodeTables). Exact (N, K) match only.
//  - (5120, 6144) GDN out_proj / attn o_proj: 160 CTAs = 2 stripes per 64-wide column, one
//    BF16 reduction step, no stripe wrap.
//  - (16384, 5120) GDN in_proj_qkvz: one full-K CTA per 128-wide column.
//  - (14336, 5120) attn qkv_proj: one full-K CTA per 128-wide column.
m16_table_entry_t const kM16Table[] = {
    {5120, 6144, {{128, 64, 128}, 160, 8}},
    {16384, 5120, {{64, 128, 128}, 128, 6}},
    {14336, 5120, {{64, 128, 128}, 112, 6}},
};

bool get_m16_launch(int prob_n, int prob_k, int sms, m16_launch_t& out)
{
    if (sms != 188)
        return false;
    for (auto const& e : kM16Table)
    {
        if (e.prob_n == prob_n && e.prob_k == prob_k)
        {
            out = e.launch;
            return true;
        }
    }
    return false;
}

// Deeper-pipeline two-m-block (32-row) kernels (decode rows 17-32), see get_m32_launch.
MarlinFuncPtr get_m32_kernel(thread_config_t const& th, int stages)
{
#define MARLIN_M32_IF(T, N, K, S)                                                                                      \
    if (th.num_threads == (T) && th.thread_n == (N) *16 && th.thread_k == (K) *16 && stages == (S))                    \
        return Marlin<nv_bfloat16, T, 2, N, K, false, S, 1>;
    MARLIN_M32_IF(128, 8, 4, 4)
    MARLIN_M32_IF(128, 8, 4, 6)
    MARLIN_M32_IF(128, 8, 4, 8)
    MARLIN_M32_IF(128, 4, 8, 4)
    MARLIN_M32_IF(128, 4, 8, 6)
    MARLIN_M32_IF(256, 16, 4, 4)
    MARLIN_M32_IF(256, 16, 4, 6)
#undef MARLIN_M32_IF
    return MarlinDefault;
}

// Decode rows 17-32 (one 32-row m block, FP32 split-K reduce, workspace_mode 2) launch
// shapes per (N, K) of the tuned decode GEMM shapes on 188-SM SM120 GPUs (launch hint
// kMarlinNvfp4LaunchHintDecodeTables). Exact (N, K) match only; without an entry, rows 17-32 keep
// the generic launch.
//  - (5120, 6144) GDN out_proj / attn o_proj: 160 CTAs = 2 stripes per 64-wide column, one
//    FP32 reduce step through C_tmp.
//  - (16384, 5120) GDN in_proj_qkvz: one full-K CTA per 128-wide column (no reduce).
//  - (14336, 5120) attn qkv_proj: one full-K CTA per 128-wide column (no reduce).
m16_table_entry_t const kM32Table[] = {
    {5120, 6144, {{128, 64, 128}, 160, 4}},
    {16384, 5120, {{64, 128, 128}, 128, 6}},
    {14336, 5120, {{64, 128, 128}, 112, 6}},
};

bool get_m32_launch(int prob_n, int prob_k, int sms, m16_launch_t& out)
{
    if (sms != 188)
        return false;
    for (auto const& e : kM32Table)
    {
        if (e.prob_n == prob_n && e.prob_k == prob_k)
        {
            out = e.launch;
            return true;
        }
    }
    return false;
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

// Dynamic shared memory the full_k kernel addresses: the kernel's own layout for
// BF16 A, 8-row m block, no act_order / zero points, plus the stripe stash.
int full_k_smem_bytes(int threads, int thread_n_blocks, int thread_k_blocks, int group_blocks, int stages)
{
    int const b_sh_stage = ((thread_n_blocks * 16) * 16 / 8 / 4) * thread_k_blocks;
    int const sh_red = (2 * thread_n_blocks + 1) * 16;
    int const sh_b = stages * b_sh_stage;
    int const sh_bias = thread_n_blocks * 16 / 8;
    int const red_b = std::max(std::max(sh_red, sh_b), std::min(sh_red, sh_b) + sh_bias);
    int const s_tb_groups = group_blocks < thread_k_blocks ? thread_k_blocks / group_blocks : 1;
    int const s_sh_stage = s_tb_groups * thread_n_blocks;
    int const a_sh_stage = (16 * thread_k_blocks / 8) * 8;
    int const int4s = red_b + stages * s_sh_stage + stages * a_sh_stage;
    return int4s * 16 + kMarlinNvfp4FullKMaxStripes * 16 * threads * static_cast<int>(sizeof(float));
}

// The instances compiled with the pre-wait prefetch (marlin_prewait_instance in the
// template: 128 threads, n tile 128, k tile 64, 6 stages, not full_k): the qkv / qkvz
// decode kernels at rows <= 8 (full-K), 9-16 (kM16Table) and 17-32 (kM32Table).
bool marlin_prewait_kernel(MarlinFuncPtr kernel)
{
    static_assert(marlin_prewait_instance<128, 8, 4, 6, false>(), "pre-wait instance set mismatch");
    static_assert(!marlin_prewait_instance<128, 4, 8, 8, false>(), "(5120,6144) kM16Table must stay stock");
    static_assert(!marlin_prewait_instance<128, 4, 8, 4, false>(), "(5120,6144) kM32Table must stay stock");
    static_assert(!marlin_prewait_instance<128, 4, 8, 8, true>(), "(5120,6144) full_k must stay stock");
    return kernel == Marlin<nv_bfloat16, 128, 1, 8, 4, true, 6, 1>
        || kernel == Marlin<nv_bfloat16, 128, 1, 8, 4, false, 6, 1>
        || kernel == Marlin<nv_bfloat16, 128, 2, 8, 4, false, 6, 1>;
}

// Largest number of split-K stripes the default `grid`-CTA launch gives one column.
int full_k_max_stripes(int k_tiles, int n_tiles, int grid)
{
    int const iters = div_ceil(k_tiles * n_tiles, grid);
    int most = 0;
    for (int c = 0; c < n_tiles; c++)
    {
        most = std::max(most, ((c + 1) * k_tiles - 1) / iters - (c * k_tiles) / iters + 1);
    }
    return most;
}

void marlin_mm_nvfp4(void const* A, void const* B, void* C, void* C_tmp, void const* b_s, void const* g_s, int prob_m,
    int prob_n, int prob_k, int* locks, int num_groups, int group_size, bool use_fp32_reduce, int dev,
    cudaStream_t stream, bool enable_pdl, int launch_hint)
{
    // The decode launch tables (full-K and per-(N, K) launches, pre-wait weight prefetch) apply only when
    // the caller asks for them; otherwise this is the stock dispatcher.
    bool const decode_table_launches = launch_hint == marlin_nvfp4::kMarlinNvfp4LaunchHintDecodeTables;
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
    // The pre-wait flag goes only to the pre-wait instances (only the decode-table launches pick
    // them); every other launch gets exactly the stock flag word.
    int const prewait_flags = enable_pdl ? kMarlinNvfp4PreWaitFlag : 0;
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

        // GDN out_proj / attention o_proj (n=5120, k=6144) at decode rows <= 8: one CTA
        // per 64-wide column with an 8-stage pipeline instead of `blocks` split-K CTAs
        // with 4 stages. The kernel replays the split-K launch's reduction chain, so the
        // output is bitwise equal to it.
        int grid = blocks;
        int full_k_flags = 0;
        if (decode_table_launches && m_block_size_8 && thread_m_blocks == 1 && num_threads == 128 && thread_n == 64
            && thread_k == 128 && group_blocks == 1 && prob_n == 5120 && prob_k == 6144 && exec_cfg.blocks_per_sm == 1
            && !use_fp32_reduce && blocks <= kMarlinNvfp4FullKGridMask && prob_n / thread_n <= blocks
            && max_shared_mem_new < (1 << kMarlinNvfp4FullKGridShift)
            && full_k_max_stripes(prob_k / thread_k, prob_n / thread_n, blocks) <= kMarlinNvfp4FullKMaxStripes
            && full_k_smem_bytes(num_threads, thread_n_blocks, thread_k_blocks, group_blocks, 8)
                <= max_shared_mem_new - 512)
        {
            kernel = Marlin<nv_bfloat16, 128, 1, 4, 8, true, 8, 1, true>;
            grid = prob_n / thread_n;
            full_k_flags = blocks << kMarlinNvfp4FullKGridShift;
        }

        // GDN in_proj_qkvz (n=16384) / attention qkv_proj (n=14336), k=5120, at decode
        // rows <= 8: one CTA per 128-wide column (128 threads, {64k, 128n}, 6 stages)
        // streaming the whole K (iters == k_tiles, slice_count 1, no lock use) instead
        // of `blocks` split-K CTAs whose partial sums chain through the BF16 output.
        // Reordered: each column's fp32 sum is rounded to BF16 once.
        thread_config_t const in_proj_cfg = {64, 128, 128};
        if (decode_table_launches && full_k_flags == 0 && prob_m <= 8 && m_block_size_8 && thread_m_blocks == 1
            && group_blocks == 1 && exec_cfg.blocks_per_sm == 1 && !use_fp32_reduce && prob_k == 5120
            && (prob_n == 16384 || prob_n == 14336) && prob_n / in_proj_cfg.thread_n <= blocks
            && is_valid_config(
                in_proj_cfg, thread_m_blocks, prob_n, prob_k, num_bits, group_size, 6, max_shared_mem_new - 512))
        {
            kernel = Marlin<nv_bfloat16, 128, 1, 8, 4, true, 6, 1>;
            num_threads = in_proj_cfg.num_threads;
            grid = prob_n / in_proj_cfg.thread_n;
        }

        // Decode rows 9-16 (one 16-row m block): per-(N, K) thread config, grid and
        // cp.async stages (kM16Table). With grid == n_tiles each CTA streams one full-K
        // column (no split-K BF16 chain through C, no stripe wrap). Reordered.
        m16_launch_t m16;
        if (decode_table_launches && full_k_flags == 0 && !m_block_size_8 && thread_m_blocks == 1 && prob_m_split > 8
            && group_blocks == 1 && exec_cfg.blocks_per_sm == 1 && !use_fp32_reduce
            && get_m16_launch(prob_n, prob_k, sms, m16) && m16.blocks >= 1 && m16.blocks <= sms
            && is_valid_config(
                m16.th, thread_m_blocks, prob_n, prob_k, num_bits, group_size, m16.stages, max_shared_mem_new - 512)
            && get_m16_kernel(m16.th, m16.stages) != MarlinDefault)
        {
            kernel = get_m16_kernel(m16.th, m16.stages);
            num_threads = m16.th.num_threads;
            grid = m16.blocks;
        }

        // Decode rows 17-32 (one 32-row m block) with the FP32 split-K reduce: per-(N, K)
        // thread config, grid and cp.async stages (kM32Table). grid == n_tiles streams one
        // full-K column per CTA (no reduce); a larger grid splits each column into stripes
        // that reduce in FP32 through C_tmp (one lock / C_tmp slot per column < sms).
        m16_launch_t m32;
        if (decode_table_launches && full_k_flags == 0 && !m_block_size_8 && thread_m_blocks == 2 && prob_m_split > 16
            && group_blocks == 1 && exec_cfg.blocks_per_sm == 1 && use_fp32_reduce
            && get_m32_launch(prob_n, prob_k, sms, m32) && m32.blocks >= 1 && m32.blocks <= sms
            && is_valid_config(
                m32.th, thread_m_blocks, prob_n, prob_k, num_bits, group_size, m32.stages, max_shared_mem_new - 512)
            && get_m32_kernel(m32.th, m32.stages) != MarlinDefault)
        {
            kernel = get_m32_kernel(m32.th, m32.stages);
            num_threads = m32.th.num_threads;
            grid = m32.blocks;
        }

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new);

        // A PDL launch orders its A, lock and C accesses after the predecessor grid
        // (griddepcontrol.wait); that also serializes the launches of this loop, which
        // share `locks`. Only L2 prefetches of the constant weights / weight scales of
        // the first pipeline stages may be issued before the wait (kMarlinNvfp4PreWaitFlag).
        // Only a PDL launch (flag in the last argument) triggers dependents early.
        int4 const* b_bias_null = nullptr;
        float const* a_scales_null = nullptr;
        int4 const* zp_null = nullptr;
        int const* g_idx_null = nullptr;
        // clang-format off
        tensorrt_llm::common::launchWithPdl("marlin_nvfp4", enable_pdl, kernel, dim3(grid), dim3(num_threads),
            static_cast<size_t>(max_shared_mem_new), stream,
            A_ptr, B_ptr, C_ptr, C_tmp_ptr,
            b_bias_null,    // b_bias
            a_scales_null,  // a_scales
            b_s_ptr, g_s_ptr,
            zp_null,        // zp
            g_idx_null,     // g_idx
            num_groups, prob_m_split, prob_n, prob_k, lda, locks,
            false,  // has_bias
            false,  // use_atomic_add
            use_fp32_reduce, max_shared_mem_new | full_k_flags | (enable_pdl ? kMarlinNvfp4PdlLaunchFlag : 0)
                | (marlin_prewait_kernel(kernel) ? prewait_flags : 0));
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
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, true,  8, 1, true>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, true,  6, 1>( MARLIN_KERNEL_PARAMS );
// Decode rows 9-16 deeper-pipeline variants (get_m16_kernel).
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, false, 6, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 4, 8, false, 8, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, false, 6, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 1, 8, 4, false, 8, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 1, 8, 8, false, 6, 1>( MARLIN_KERNEL_PARAMS );
// Decode rows 17-32 deeper-pipeline variants (get_m32_kernel).
template __global__ void Marlin<nv_bfloat16, 128, 2, 8, 4, false, 6, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 2, 8, 4, false, 8, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 128, 2, 4, 8, false, 6, 1>( MARLIN_KERNEL_PARAMS );
template __global__ void Marlin<nv_bfloat16, 256, 2, 16, 4, false, 6, 1>( MARLIN_KERNEL_PARAMS );

// clang-format on

} // namespace marlin

namespace marlin_nvfp4
{

void marlinNvfp4Gemm(void const* act_bf16, void const* weight, void* output, void* C_tmp, void const* weight_sf,
    void const* global_scale_bf16, int m, int n, int k, int* workspace, int num_groups, int group_size,
    bool use_fp32_reduce, cudaStream_t stream, bool enable_pdl, int launch_hint)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(isMarlinNvfp4DenseSmSupported(sm),
        "Marlin NVFP4 GEMM is only supported on SM89, SM90-99, and the SM120 family; current SM = %d", sm);

    int dev;
    cudaGetDevice(&dev);

    // griddepcontrol needs SM90+; on SM89 launch without the PDL attribute.
    bool const pdl = enable_pdl && sm >= 90;
    marlin::marlin_mm_nvfp4(act_bf16, weight, output, C_tmp, weight_sf, global_scale_bf16, m, n, k, workspace,
        num_groups, group_size, use_fp32_reduce, dev, stream, pdl, launch_hint);
}

} // namespace marlin_nvfp4

TRTLLM_MV2_NAMESPACE_END
