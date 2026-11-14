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

//! The widest row the launch grid covers (65,534 decode CTAs of 2,048 entries).
constexpr int32_t kDsparkRefreshMaxWidth = 65534 * 2048;

//! DSpark / DFlash draft block-table refresh (ops/dspark/dspark_block_tables; replaces its former Triton
//! kernels bit for bit). For r < rows, j < width, with v = src[r * srcStrideRow + j * srcStrideCol] (int32 or int64,
//! srcIs64; strides in elements):
//!   tables[r * tablesStride + j] = int32(max(v, 0) // divisor)
//!   counts[r]                    = number of v >= 0 over the row (int64, stored, not accumulated)
//! divisor >= 1 (a power of two is a shift; an int32 source with divisor > 2^31 - 1 decodes to 0). counts contiguous.
//! Integer only; no atomics or scratch; graph-safe; PDL (griddepcontrol.wait at entry).
void invokeDsparkRefreshBlockTables(void const* src, bool srcIs64, int64_t srcStrideRow, int64_t srcStrideCol,
    int32_t* tables, int64_t tablesStride, int64_t* counts, int32_t rows, int32_t width, int64_t divisor,
    cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
