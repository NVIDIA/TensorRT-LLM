/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

// Device-side description of one LoRA module (fc1, fc2, or gated) for the
// MoE per-token A/B pointer-table expansion. All pointers refer to device
// memory.
//
// Inputs (per source token, indexed by the pre-permutation source row index):
//   ranks_src: int32 [num_rows], per-source-token LoRA rank.
//   ptrs_src:  int64 [num_rows * 2], pointer bits laid out as (A_ptr, B_ptr)
//              per source token.
//
// Outputs (per permuted row, sized expanded_num_rows == num_rows * top_k):
//   ranks_out: int32 [expanded_num_rows], per-permuted-row LoRA rank.
//   ptrs_out:  int64 [expanded_num_rows * 2], per-permuted-row
//              (A_ptr + offset, B_ptr + offset). The per-expert offset is
//              weight_index * dim * rank * lora_dtype_bytes, so the consumer
//              can reinterpret directly as the LoRA scalar type.
//
// dim_a and dim_b are the non-rank dimension of A and B respectively:
// fc1/gated use (hidden_size, inter_size); fc2 uses (inter_size, hidden_size).
struct MoeLoraExpandModule
{
    int32_t const* ranks_src = nullptr;
    int64_t const* ptrs_src = nullptr;
    int64_t dim_a = 0;
    int64_t dim_b = 0;
    int32_t* ranks_out = nullptr;
    int64_t* ptrs_out = nullptr;
};

// Device-side replacement for the host-CPU pointer fan-out in
// CutlassMoeFCRunner::setupLoraWorkspace. Reads per-source-token LoRA metadata
// and permuted_rows, and writes per-permuted-row pointer tables directly into
// device memory. It performs no host synchronization and no cudaMemcpyAsync
// staging, so it is safe to launch from a captured CUDA graph.
//
// expert_first_token_offset has shape [num_experts_per_node + 1] (int64,
// device-resident). The kernel uses it both to find the expert a permuted row
// belongs to and to derive weight_index = local_expert_idx + start_expert for
// the per-expert weight-buffer stride.
//
// lora_dtype_bytes is the size in bytes of the LoRA matrix scalar (e.g. 2 for
// bf16/fp16). It scales the stride applied to the A/B pointers so consumers can
// reinterpret the result directly as the appropriate scalar type.
//
// gated may be nullptr for non-gated activations; when non-null, the gated
// module's outputs are produced in the same pass.
void launchMoeLoraPointerExpand(int32_t const* permuted_rows, int64_t const* expert_first_token_offset,
    int32_t num_experts_per_node, int32_t start_expert, int64_t num_rows, int64_t expanded_num_rows,
    int64_t lora_dtype_bytes, MoeLoraExpandModule const& fc1, MoeLoraExpandModule const& fc2,
    MoeLoraExpandModule const* gated, cudaStream_t stream);

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
