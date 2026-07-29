/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Fused permute + scatter for Ulysses A2A (peer-WRITE variant).
//
// Replaces the .permute(2,0,1,3,4).contiguous() materialization that would
// otherwise happen pre-A2A. Reads `input [B, S_local, H, D]` once and
// scatters each (b,s,h,d) element to one of two destinations:
//   - peer != my_rank → send_buf[peer,    b, s, h-peer*H_local,  d]   (local)
//   - peer == my_rank → recv_buf[my_rank, b, s, h-my_rank*H_local, d] (symm-mem)
//
// After this kernel runs, the caller fires (P-1) cudaMemcpyBatchAsync
// entries to push send_buf[p] → peer[p].recv_buf[my_rank], then an LSA
// barrier (both folded into ulysses_a2a_async).
//
// Layout (all contiguous bf16):
//   input    : [B, S_local, H,        D]   row-major
//   send_buf : [P, B, S_local, H/P,   D]   row-major
//   recv_buf : [P, B, S_local, H/P,   D]   row-major
//
// Requires: D % 8 == 0 (int4 vec load); H % P == 0; bf16 only.
void launchUlyssesPermuteScatter(void const* input, // bf16 [B, S_local, H, D]
    void* send_buf,                                 // bf16 [P, B, S_local, H/P, D]
    void* recv_buf,                                 // bf16 [P, B, S_local, H/P, D]
    int my_rank, int B, int S_local, int H, int D, int P, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
