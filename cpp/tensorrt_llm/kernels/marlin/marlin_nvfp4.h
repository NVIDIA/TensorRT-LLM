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

// Unified host-side header for the Marlin NVFP4 W4A16 kernels.
//
// Replaces:
//   - marlin_nvfp4_gemm.h            (single-expert GEMM + activation dequant)
//   - marlin_nvfp4_moe_gemm.h        (MoE fused GEMM dispatcher)
//   - marlin_repack.h                (GPTQ-style weight repack)
//   - marlin_nvfp4_dispatch_utils.h  (thread/exec config shared by both GEMMs)
//
// All entries are pure C-style functions / POD structs that the Torch
// bindings under cpp/tensorrt_llm/thop/ link against; no device-side
// templates leak through this header.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace marlin_nvfp4
{

// =========================================================================
// Single-expert NVFP4 GEMM
// =========================================================================

// Dequantize FP4 E2M1 activations to BF16.
// act_fp4:  [M, K/2] packed FP4 (uint8)
// act_sf:   activation block scales (FP8 E4M3, swizzled, uint8)
// alpha:    global scale (float32, device pointer)
// act_bf16: [M, K] output BF16 buffer (pre-allocated by caller)
void dequantFp4Activations(
    void const* act_fp4, void const* act_sf, float const* alpha, void* act_bf16, int m, int k, cudaStream_t stream);

// Run Marlin W4A16 GEMM: BF16 activations + FP4 weights.
// act_bf16:  [M, K] BF16 activations (pre-dequantized by caller)
// weight:    [N, K/2] FP4 weights (row-major packed)
// output:    [M, N] output buffer
// weight_sf: weight block scales (FP8 E4M3, swizzled)
// global_scale_bf16: global scale as BF16 (device pointer, uint16_t)
// workspace: [sms] int32 lock buffer (pre-allocated, zeroed by caller)
void marlinNvfp4Gemm(void const* act_bf16, void const* weight, void* output, void* C_tmp, void const* weight_sf,
    void const* global_scale_bf16, int m, int n, int k, int* workspace, int num_groups, int group_size,
    bool use_fp32_reduce, cudaStream_t stream);

// =========================================================================
// MoE fused NVFP4 GEMM
// =========================================================================

void marlinNvfp4MoeGemmDispatcher(void const* A, void const* B, void* C, void* C_tmp, void const* b_scales,
    void const* global_scale, void const* sorted_token_ids, void const* expert_ids, void const* num_tokens_past_padded,
    void const* topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int prob_m, int prob_n, int prob_k,
    void* workspace, int num_groups, int group_size, bool use_fp32_reduce, bool use_atomic_add, cudaDataType_t outType,
    cudaStream_t stream);

// =========================================================================
// GPTQ-style weight repack into Marlin tiled format
// =========================================================================

void gptq_marlin_repack_dispatch(uint32_t const* b_q_weight_ptr, uint32_t const* perm_ptr, uint32_t* out_ptr,
    int size_k, int size_n, int num_bits, bool has_perm, bool is_a_8bit, cudaStream_t stream);

} // namespace marlin_nvfp4

// =========================================================================
// Shared dispatch utilities for single-expert and MoE GEMMs
// =========================================================================
//
// Implementations live in marlin_nvfp4_dispatch_utils.cpp.  Kept in a
// separate namespace so the dispatcher tables are not part of the public
// API surface above; only the *.cu dispatchers need them.

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

// Thread configs ordered by priority.
extern thread_config_t const kSmallBatchConfigs[];
extern thread_config_t const kLargeBatchConfigs[];

extern int const kSmallBatchConfigCount;
extern int const kLargeBatchConfigCount;

// Scale cache size shared by both single-expert and MoE dispatchers.
int get_scales_cache_size(
    thread_config_t const& th_config, int prob_n, int prob_k, int num_bits, int group_size, int stages);

// Basic config validation shared by both dispatchers.
bool is_config_feasible(thread_config_t const& cfg, int prob_n, int prob_k);

} // namespace marlin_nvfp4_dispatch
