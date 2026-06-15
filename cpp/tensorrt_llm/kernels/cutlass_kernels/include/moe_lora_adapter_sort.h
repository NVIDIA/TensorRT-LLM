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

// "Adapter as secondary sort key" for routed-expert MoE LoRA
// (lora_gemm_by_adapter.md, Piece B).
//
// The base expert sort (fusedBuildExpertMapsSortFirstToken) groups permuted
// rows by expert and produces the forward/inverse index maps plus
// expert_first_token_offset. This post-pass regroups the rows *within* each
// expert block so that rows using the same LoRA adapter slot become
// contiguous, which lets the builder-side run-length aggregation (Piece A)
// merge them into one fat-M grouped-GEMM problem.
//
// Because the reordering happens strictly inside each expert block, the
// per-expert counts do not change: expert_first_token_offset is left
// untouched, and only the two index maps are permuted. Per Fact 2 in
// lora_gemm_by_adapter.md, intra-block row order is arbitrary for the expert
// GEMM, and the finalize/combine un-permutes through the (updated) maps, so
// this is correctness-preserving. Grouping (not a stable sort) is sufficient:
// the order of rows within one slot is irrelevant downstream.
//
// Slot-indexed mode only: token_to_slot must be a dense per-token adapter id
// in [0, num_slots); any value outside that range (e.g. -1 for a no-adapter
// token) is folded into a single trailing "no-adapter" bucket so those rows
// still group together (forming a rank-0 run for Piece A).
//
// Arguments:
//   permuted_row_to_unpermuted_row: [expanded_num_rows] in/out. Rewritten in
//       place for the live region [0, expert_first_token_offset[num_experts]);
//       the invalid tail is left as the base sort produced it.
//   unpermuted_row_to_permuted_row: [expanded_num_rows] out. Inverse map,
//       updated to match the new positions.
//   p2u_scratch: [expanded_num_rows] device temp. The launcher copies the
//       current permuted_row_to_unpermuted_row into it (a stable read source so
//       reads and writes do not alias).
//   expert_first_token_offset: [num_experts_per_node + 1] device, read-only.
//   token_to_slot: [num_tokens] device, dense per-token adapter slot id.
//   num_slots: number of distinct adapter slots (histogram width). The kernel
//       uses num_slots + 1 shared-memory bins (the +1 is the no-adapter
//       bucket).
//
// Returns false without launching when num_slots is too large to fit the
// shared-memory histogram; the caller must then keep the base ordering (Piece A
// still applies, just with shorter runs).
bool launchMoeLoraAdapterRegroup(int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int* p2u_scratch, int64_t const* expert_first_token_offset, int32_t const* token_to_slot,
    int num_experts_per_node, int64_t num_tokens, int64_t expanded_num_rows, int num_slots, cudaStream_t stream);

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
