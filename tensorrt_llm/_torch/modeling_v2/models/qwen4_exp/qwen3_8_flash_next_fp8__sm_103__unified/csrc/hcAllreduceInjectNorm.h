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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

//! IPC one-shot Lamport all-reduce of a block output (stock `allreduce_fusion_kernel_oneshot_lamport` workspace and
//! protocol) fused with a hyper-connection site's `hc_out = bf16(fma(inject, block, hc))` and grouped RMS norm
//! `normed = bf16((hc_out * rstd) * (1 + w))`; bitwise that all-reduce followed by the Triton `_hc_inject_norm_kernel`.
//! With `shared` and `gate`, the block is first `bf16(fma(sigmoid(gate), shared, routed))`.
struct HcInjectNormParams
{
    int nranks;
    int rank;
    int rows;
    int hidden_dim;          // 2560
    int hc_count;            // 4
    int program_warps;       // 4 or 2: warps of the Triton program whose reduction order is reproduced
    void** workspace;        // the stock all-reduce fusion IPC workspace
    void const* partial;     // [rows, hidden] bf16, contiguous
    void const* hc;          // [rows, hc_count * hidden] bf16
    int64_t hc_stride;
    void const* inject;      // [rows, >= hc_count] bf16
    int64_t inject_stride;
    void const* shared;      // optional [rows, hidden] bf16
    int64_t shared_stride;
    void const* gate;        // optional [rows, 1] bf16
    int64_t gate_stride;
    void const* norm_weight; // [hc_count * hidden] bf16
    void* hc_out;            // [rows, hc_count * hidden] bf16
    int64_t hc_out_stride;
    void* normed;            // [rows, hc_count * hidden] bf16
    int64_t normed_stride;
    float eps;
    bool trigger_completion_at_end;
    cudaStream_t stream;
};

void hc_allreduce_inject_norm_op(HcInjectNormParams const& params);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
