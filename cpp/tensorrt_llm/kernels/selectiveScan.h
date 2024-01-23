/*
 * Adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.h
 * Copyright (c) 2023, Tri Dao.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Not a contribution
 * Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
 * NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{

struct SSMParamsBase
{
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t state_batch_stride;
    index_t state_d_stride;

    // Common data pointers.
    void* __restrict__ A_ptr;
    void* __restrict__ B_ptr;
    void* __restrict__ C_ptr;
    void* __restrict__ D_ptr;
    void* __restrict__ u_ptr;
    void* __restrict__ delta_ptr;
    void* __restrict__ delta_bias_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ x_ptr;
    void* __restrict__ z_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t, typename weight_t>
void invokeSelectiveScan(SSMParamsBase& params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void invokeSelectiveScanUpdate(SSMParamsBase& params, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
