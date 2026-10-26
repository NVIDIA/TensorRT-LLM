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

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! NATIVE_FINALIZE: out[t, h] = bf16(sum_k fma(w[t, k], fc2[idx[t, k], h], acc)) over k in order
//! (idx -1 skipped), fp32; with shared: bf16(float(shared[t, h]) + float(that)). Bit for bit kernels/m3_moe_finalize
//! (Triton) and the TRTLLM-Gen runner's finalize (+ shared.add_). weights: bf16 or fp32 [tokens, topk]; topk 1..10;
//! fc2 rows 16-byte aligned (hidden % 8 == 0, ldFc2 % 8 == 0). cfg: the (threads, vectors per thread) table entry 0..7
//! (only the work split; every element's arithmetic is the same in all of them).
cudaError_t invokeMoeFinalize(__nv_bfloat16 const* fc2, int64_t ldFc2, void const* weights, bool weightsFp32,
    int32_t const* idx, __nv_bfloat16 const* shared, __nv_bfloat16* out, int64_t tokens, int topk, int hidden, int cfg,
    cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
