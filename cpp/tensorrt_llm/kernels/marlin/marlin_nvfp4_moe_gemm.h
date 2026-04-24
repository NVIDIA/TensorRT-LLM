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
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace marlin_nvfp4
{

void marlinNvfp4MoeGemmDispatcher(void const* A, void const* B, void* C, void* C_tmp, void const* b_scales,
    void const* global_scale, void const* sorted_token_ids, void const* expert_ids, void const* num_tokens_past_padded,
    void const* topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int prob_m, int prob_n, int prob_k,
    void* workspace, int num_groups, int group_size, bool use_fp32_reduce, bool use_atomic_add, cudaDataType_t outType,
    cudaStream_t stream);

} // namespace marlin_nvfp4
