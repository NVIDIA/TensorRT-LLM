/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cute_dsl
{
template <typename InputType, typename SFType>
void moePermute(InputType const* input, InputType* permuted_output, SFType const* input_sf, SFType* permuted_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens, int32_t const hidden_size,
    int32_t const top_k, int32_t const tile_size, cudaStream_t stream);

template <typename InputType, typename TopKScaleType>
void moeUnpermute(InputType const* permuted_input, InputType* output, int32_t const* expanded_idx_to_permuted_idx,
    TopKScaleType const* topk_scales, int32_t const num_tokens, int32_t const hidden_size, int32_t const top_k,
    cudaStream_t stream);

template <typename InputType>
void moeOutputMemset(InputType* input, int32_t const* tile_idx_to_mn_limit, int32_t const* expanded_idx_to_permuted_idx,
    int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,
    int32_t const max_num_permuted_tokens, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size,
    cudaStream_t stream);

template <typename InputType, typename OutputType, typename SFType>
void moeActivation(InputType const* input, OutputType* output, float const* global_sf, SFType* output_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* num_non_exiting_tiles,
    cutlass_kernels::ActivationParams activation_params, int32_t const max_num_permuted_tokens,
    int32_t const interm_size, int32_t const tile_size, cudaStream_t stream);

} // namespace kernels::cute_dsl

TRTLLM_NAMESPACE_END
