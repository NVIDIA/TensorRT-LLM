/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

enum class Mamba2Dtype : int32_t
{
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
};

struct Mamba2MTPSSMCacheParams
{
    // State tensors
    void* ssm;                 // [batch, nheads, head_dim, ssm_dim]
    void* intermediate_states; // [batch, cache_steps, nheads, head_dim, ssm_dim]

    // Input tensors
    void const* x;  // [bs, cache_steps, nheads, head_dim]
    void const* dt; // [bs, cache_steps, nheads, head_dim] (strided)
    void const* A;  // [nheads, head_dim, ssm_dim] (strided)
    void const* B;  // [bs, cache_steps, ngroups, ssm_dim]
    void const* C;  // [bs, cache_steps, ngroups, ssm_dim]

    // Output
    void* out; // [bs, cache_steps, nheads, head_dim]

    // Optional tensors (nullptr if not present)
    void const* D;                              // [nheads, head_dim]
    void const* z;                              // [bs, cache_steps, nheads, head_dim]
    void const* dt_bias;                        // [nheads, head_dim]
    int32_t const* ssm_batch_indices;           // [bs]
    int32_t const* intermediate_states_indices; // [bs]
    int32_t const* retrieve_parent_token;       // [bs, cache_steps]

    // Flags
    bool dt_softplus;

    // Dimensions
    int cache_steps;
    int pad_slot_id;
    bool disable_state_update;
    int bs;
    int nheads;
    int head_dim;
    int ssm_dim;
    int ngroups;

    // Data types
    Mamba2Dtype ssm_dtype;
    Mamba2Dtype in_out_dtype;
    Mamba2Dtype weight_dtype;
    Mamba2Dtype a_dtype;
};

void invokeMamba2MTPSSMCacheUpdate(Mamba2MTPSSMCacheParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
