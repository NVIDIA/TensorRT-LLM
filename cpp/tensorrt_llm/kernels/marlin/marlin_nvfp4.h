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

// Host-side header for the Marlin NVFP4 W4A16 kernels.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace marlin_nvfp4
{

void dequantFp4Activations(
    void const* act_fp4, void const* act_sf, float const* alpha, void* act_bf16, int m, int k, cudaStream_t stream);

void marlinNvfp4Gemm(void const* act_bf16, void const* weight, void* output, void* C_tmp, void const* weight_sf,
    void const* global_scale_bf16, int m, int n, int k, int* workspace, int num_groups, int group_size,
    bool use_fp32_reduce, cudaStream_t stream);

void marlinNvfp4MoeGemmDispatcher(void const* A, void const* B, void* C, void* C_tmp, void const* b_scales,
    void const* global_scale, void const* sorted_token_ids, void const* expert_ids, void const* num_tokens_past_padded,
    void const* topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int prob_m, int prob_n, int prob_k,
    void* workspace, int num_groups, int group_size, bool use_fp32_reduce, bool use_atomic_add, cudaDataType_t outType,
    cudaStream_t stream);

void gptq_marlin_repack_dispatch(uint32_t const* b_q_weight_ptr, uint32_t const* perm_ptr, uint32_t* out_ptr,
    int size_k, int size_n, int num_bits, bool has_perm, bool is_a_8bit, cudaStream_t stream);

} // namespace marlin_nvfp4

namespace marlin_nvfp4_dispatch
{

struct thread_config_t
{
    int thread_k;
    int thread_n;
    int num_threads;
};

struct exec_config_t
{
    int blocks_per_sm;
    thread_config_t tb_cfg;
};

extern thread_config_t const kSmallBatchConfigs[];
extern thread_config_t const kLargeBatchConfigs[];

extern int const kSmallBatchConfigCount;
extern int const kLargeBatchConfigCount;

int get_scales_cache_size(
    thread_config_t const& th_config, int prob_n, int prob_k, int num_bits, int group_size, int stages);

bool is_config_feasible(thread_config_t const& cfg, int prob_n, int prob_k);

} // namespace marlin_nvfp4_dispatch
