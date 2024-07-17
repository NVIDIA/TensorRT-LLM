/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

namespace tensorrt_llm
{
namespace kernels
{

struct lruParams
{
    int batch, width;
    int max_seqlen; // only valid for padded input.
    int block_size; // used for the cases that enable fused gate
    bool remove_padding;

    // Common data pointers.
    void* __restrict__ A_ptr;
    void* __restrict__ x_ptr;
    void* __restrict__ y_ptr;
    void* __restrict__ y_bias_ptr;
    void* __restrict__ gate_ptr;
    void* __restrict__ gate_bias_ptr;
    void* __restrict__ gate_x_ptr;
    void* __restrict__ gate_x_bias_ptr;
    void* __restrict__ gate_a_ptr;
    void* __restrict__ gate_a_bias_ptr;
    void* __restrict__ state_ptr;
    void* __restrict__ out_ptr;
    int const* __restrict__ last_token_ids_ptr;
    int const* __restrict__ slot_mapping_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void invokeRGLRU(lruParams& params, cudaStream_t stream);

template <typename T>
void invokeRGLRUUpdate(lruParams& params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
