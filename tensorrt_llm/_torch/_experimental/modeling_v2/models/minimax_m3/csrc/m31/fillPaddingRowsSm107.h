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

//! DEP_PAD_FILL: for every row r with isPadding[r], rows[r, :] = T(value) and, when hasCol0, rows[r, 0] = T(col0); the
//! real rows are neither read nor written. rows: contiguous [nRows, cols] of T = float32 (elemSize 4) or bf16 / fp16
//! (elemSize 2, isBf16 picks bf16); T(v) rounds to nearest even, so bit for bit the stock masked fills. 16-byte aligned
//! rows of whole 16-byte vectors with nRows <= 2048 take the compact kernel (one CTA wave: each CTA lists the padded
//! rows from the mask in shared memory and writes its column slice of every padded row, 128 threads x 2 vectors);
//! anything else takes the per-row grid kernel (256 threads x 8 iterations, nRows <= 65535). PDL: waits on its
//! predecessor before the first load, releases its dependents after the last store.
cudaError_t invokeFillPaddingRows(void* rows, int elemSize, bool isBf16, bool const* isPadding, int64_t nRows,
    int64_t cols, float value, float col0, bool hasCol0, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
