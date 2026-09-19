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
// slot -> per-source-token expansion. All pointers refer to device memory.
//
// Inputs (indexed by adapter slot):
//   slot_ranks: int32 [num_slots], per-slot LoRA rank (0 == slot inactive).
//   slot_ptrs:  int64 [num_slots * 3], pointer bits laid out as
//               (A_ptr, B_ptr, dora_ptr) per slot; dora is ignored.
//
// Outputs (indexed by source token, sized [num_tokens]):
//   ranks_out: int32 [num_tokens], per-source-token LoRA rank.
//   ptrs_out:  int64 [num_tokens * 2], per-source-token (A_ptr, B_ptr).
//
// These outputs match the (ranks_src, ptrs_src) inputs that
// launchMoeLoraPointerExpand consumes, so its results feed directly into the
// pointer-expand stage.
struct MoeLoraSlotExpandModule
{
    int32_t const* slot_ranks = nullptr;
    int64_t const* slot_ptrs = nullptr;
    int32_t* ranks_out = nullptr;
    int64_t* ptrs_out = nullptr;
};

// Device-side slot -> per-source-token expansion for routed-expert MoE LoRA.
// For each source token t in [0, num_tokens), reads slot = token_to_slot[t]
// and copies the slot's (rank, A_ptr, B_ptr) into the per-token output tables
// for every module. Runs entirely on the stream with no host synchronization,
// so it is safe to record into a CUDA graph: replaying it re-reads the
// (in-place updated) device slot tables and token_to_slot, picking up new
// adapter assignments without re-capture.
//
// token_to_slot: int32 [num_tokens] (device). Entries outside [0, num_slots)
// are treated as inactive (rank 0, null pointers) rather than indexing out of
// bounds.
//
// gated may be nullptr for non-gated activations; when non-null, the gated
// module's outputs are produced in the same pass.
void launchMoeLoraSlotExpand(int32_t const* token_to_slot, int64_t num_tokens, int64_t num_slots,
    MoeLoraSlotExpandModule const& fc1, MoeLoraSlotExpandModule const& fc2, MoeLoraSlotExpandModule const* gated,
    cudaStream_t stream);

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
