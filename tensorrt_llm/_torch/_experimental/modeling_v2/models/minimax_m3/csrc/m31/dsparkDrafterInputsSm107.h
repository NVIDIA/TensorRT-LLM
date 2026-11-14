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
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! The DSpark / DFlash drafter's step inputs in one launch (ops/dspark/dspark_drafter_inputs; replaces its
//! former Triton _inputs_kernel bit for bit). For each generation request r (n >= 1):
//!   bonus[r]         = accepted[r, max(nacc[r] - 1, 0)]                                       (int64)
//!   qpos[r, q]       = min(ctxLen[slots[r]] + nacc[r], maxCtx) + q,  q < blockSize              (int64)
//!   cpos[r, c]       = ctxLen[slots[r]] + c,                         c < tokensPerRequest       (int64)
//!   noise[r, 0, :]   = table[bonus[r]] (raw bytes) if 0 <= bonus[r] < vocab, else all-zero bytes
//!   noise[r, j>0, :] = maskRow
//! accepted: int32 / int64 [n, acceptedStride] (unit column stride); nacc: int32 / int64 [n] contiguous; slots /
//! ctxLen: int64 contiguous; table: [vocab, rowBytes / elementSize] contiguous; maskRow: rowBytes contiguous; outputs
//! contiguous. Integer arithmetic and row copies only. Graph-safe; PDL (griddepcontrol.wait at entry).
void invokeDsparkDrafterInputs(void const* accepted, bool acceptedIs64, int64_t acceptedStride, void const* nacc,
    bool naccIs64, int64_t const* slots, int64_t const* ctxLen, int64_t maxCtx, void const* table, void const* maskRow,
    int64_t* bonus, int64_t* qpos, int64_t* cpos, void* noise, int64_t rowBytes, int64_t vocab, int32_t n,
    int32_t blockSize, int32_t tokensPerRequest, cudaStream_t stream);

//! invokeDsparkDrafterInputs plus this step's block-table refresh of rows 0..n-1 in the same launch (replaces the
//! former Triton _inputs_refresh_kernel bit for bit): with v = src[r, j] (int32 / int64 offsets, strides in elements),
//!   tables[r, j] = int32(max(v, 0) // divisor),  counts[r] = number of v >= 0 over the row (int64, stored).
void invokeDsparkDrafterInputsRefresh(void const* accepted, bool acceptedIs64, int64_t acceptedStride, void const* nacc,
    bool naccIs64, int64_t const* slots, int64_t const* ctxLen, int64_t maxCtx, void const* table, void const* maskRow,
    int64_t* bonus, int64_t* qpos, int64_t* cpos, void* noise, int64_t rowBytes, int64_t vocab, int32_t n,
    int32_t blockSize, int32_t tokensPerRequest, void const* src, bool srcIs64, int64_t srcStrideRow,
    int64_t srcStrideCol, int32_t* tables, int64_t tablesStride, int64_t* counts, int32_t width, int64_t divisor,
    cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
