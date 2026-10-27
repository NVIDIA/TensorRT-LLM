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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! TAIL16-G2: the speculative worker's _zero_padding_rows (uniform layout) in place and in one launch:
//!   logits[r, :] = 0.0f   iff   slotIds[r / rowsPerRequest] == dummySlotRow
//! for r in 0..rows-1 (slotIds already offset to the first generation request). Stock: an eq, a repeat_interleave and
//! an out-of-place masked_fill over every row. Bitwise the stock result: real rows are untouched, padding rows are
//! +0.0. logits: fp32 rows of d with row stride rowStride elements. Graph-safe (no host state).
void invokeZeroPaddingRows(float* logits, int64_t rowStride, int64_t const* slotIds, int64_t dummySlotRow, int32_t rows,
    int32_t rowsPerRequest, int32_t d, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
