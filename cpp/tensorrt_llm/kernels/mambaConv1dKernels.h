/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{

struct MambaConv1dParamsBase
{
    int batch, dim, max_seqlen, dconv, pre_stride, post_stride;
    bool remove_padding;
    bool apply_silu;
    void* __restrict__ in_ptr;
    void* state_in_ptr;
    void* state_out_ptr;
    void* __restrict__ weight_ptr;
    void* __restrict__ bias_ptr;
    void* __restrict__ out_ptr;
    int const* __restrict__ last_token_ids_ptr;
    int const* __restrict__ state_slot_mapping_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t>
void invokeMambaConv1dContext(MambaConv1dParamsBase& params, cudaStream_t stream);

template <typename input_t>
void invokeMambaConv1dGeneration(MambaConv1dParamsBase& params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
