/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass_kernels/include/moe_kernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tensorrt_llm::kernels
{
bool fusedBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* unpermuted_token_selected_experts,
    int* permuted_source_token_ids, int64_t* expert_first_token_offset, int64_t const num_tokens,
    int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
    cudaStream_t stream);

void buildExpertMaps(int const* token_selected_experts, int* unpermuted_token_selected_experts,
    int* unpermuted_source_token_ids, int64_t const num_tokens, int const num_experts_per_node,
    int const experts_per_token, int const start_expert, int const end_expert, cudaStream_t stream);

void generateTokenPermutation(int const* unpermuted_token_selected_experts, int const* unpermuted_source_token_ids,
    int* permuted_token_selected_experts, int* permuted_source_token_ids, int64_t* expert_first_token_offset,
    int64_t num_rows, int64_t num_experts_per_node, int64_t k, cutlass_kernels::CubKeyValueSorter& sorter,
    void* sorter_ws, cudaStream_t stream);

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input,
    ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
    int const* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
    int const num_experts_per_node, float const* fc1_act_global_scale, int64_t* expert_first_token_offset,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream);

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* final_scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const num_rows,
    int64_t const cols, int64_t const experts_per_token, int64_t const* num_valid_ptr,
    cutlass_kernels::MOEParallelismConfig parallelism_config, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
