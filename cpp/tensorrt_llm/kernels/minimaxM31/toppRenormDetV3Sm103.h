/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstddef>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! TOPP_DET v3: the same renorm as invokeToppRenormDet (toppRenormDetSm103.h: semantics, the bitwise contract vs
//! FlashInfer's deterministic top_p_renorm_probs, the top_p >= 1 keep-all rule, --use_fast_math, the co-residency
//! rule for ctasPerRow), with the output bitwise equal to v1.2's on every row. What changes is the synchronization:
//!   - no leader CTA: after each radix pass every CTA of the row waits for all of them, then runs FlashInfer's
//!     selection itself on the row's global integer histograms of that pass (identical inputs and code, so identical
//!     bits in every CTA); the norm is likewise reduced by every CTA from the same 1024 chain sums;
//!   - one global histogram buffer per pass, cleared by CTA 0 one barrier later (once every CTA has read it);
//!   - per-barrier arrival counters that only grow (each launch adds 1024 per barrier it runs, whatever its
//!     ctasPerRow), so a workspace needs no reset between launches and replays, and launches with different
//!     ctasPerRow may share it.
//! candidateList: when pass 0's threshold bucket holds at most 4096 elements, its members are listed once (one barrier)
//! and the remaining radix passes run locally in every CTA on the list (the same integer histograms, so the same
//! bits); larger buckets take the global passes.
//! pdl: launched as a programmatic dependent; everything that reads probs, top_p or the workspace waits for the
//! preceding grid, and dependents are released once the workspace is ready for the next launch.
//! Workspace: toppRenormDetV3WorkspaceBytesPerRow() x rows bytes, ZERO-initialized once before the first launch,
//! then reused as is (CUDA-graph safe). Launches sharing one workspace must be stream-ordered.
cudaError_t invokeToppRenormDetV3(float const* probs, float* out, float const* topP, int rows, int d, int ctasPerRow,
    void* workspace, int* keptOut, bool pdl, bool candidateList, cudaStream_t stream);

//! Workspace bytes per row of invokeToppRenormDetV3.
size_t toppRenormDetV3WorkspaceBytesPerRow();

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
