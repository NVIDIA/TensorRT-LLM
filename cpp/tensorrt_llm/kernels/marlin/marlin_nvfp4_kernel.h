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

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"

#define MARLIN_KERNEL_PARAMS                                                                                           \
    const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,            \
        const int4 *__restrict__ b_bias_ptr, const float *__restrict__ a_scales_ptr,                                   \
        const int4 *__restrict__ scales_ptr, const uint16_t *__restrict__ global_scale_ptr,                            \
        const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, int num_groups, int prob_m, int prob_n,        \
        int prob_k, int lda, int *locks, bool has_bias, bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem

namespace MARLIN_NAMESPACE_NAME
{

template <typename scalar_t,   // compute type (nv_bfloat16)
    int const threads,         // threads per block (128 or 256)
    int const thread_m_blocks, // 16x16 blocks in M dimension
    int const thread_n_blocks, // 16x16 blocks in N dimension
    int const thread_k_blocks, // 16x16 blocks in K dimension
    bool const m_block_size_8, // use 8-row M blocks (thread_m_blocks==1)
    int const stages,          // async pipeline stages
    int const group_blocks           // consecutive blocks per scale group
    >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
