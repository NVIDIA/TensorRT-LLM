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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{

struct SSMParamsBase
{
    int batch, dim, dstate, dt_rank, nheads, ngroups, chunk_size;
    int max_seqlen; // only valid for padded input.
    int num_tokens;
    bool remove_padding;
    bool delta_softplus;
    bool is_mamba2;

    // Common data pointers.
    void* __restrict__ A_ptr;
    void* __restrict__ BC_ptr;
    void* __restrict__ D_ptr;
    void* __restrict__ u_ptr;
    void* __restrict__ delta_ptr;
    void* __restrict__ delta_bias_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ x_ptr;
    void* __restrict__ z_ptr;
    // Workspace data pointers.
    void* __restrict__ Os_ptr;
    void* __restrict__ St_ptr;
    void* __restrict__ dc_ptr;
    void* __restrict__ dA_ptr;
    void* __restrict__ CB_ptr;
    void* __restrict__ desc_ptr;
    int const* __restrict__ last_token_ids_ptr;
    int const* __restrict__ slot_mapping_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t, typename weight_t>
void invokeSelectiveScan(SSMParamsBase& params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void invokeChunkScan(SSMParamsBase& params, cudaStream_t stream, tensorrt_llm::common::CUDADriverWrapper* driver);

template <typename input_t, typename weight_t>
void invokeSelectiveScanUpdate(SSMParamsBase& params, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
