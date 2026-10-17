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
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! In place, for every row: row[i] <- row[(i - shift) mod cols], the same convention as torch.roll:
//! a positive shift moves elements toward higher indices (right), a negative one toward lower (left).
//! Columns must be unit-stride; rows may be strided. Three reversal passes, no scratch memory.
//! \param data      device pointer to the first element
//! \param rows      number of rows
//! \param cols      elements per row
//! \param rowStride elements between the starts of consecutive rows
//! \param shift     any value; reduced modulo cols
//! \param elemSize  element size in bytes: 1, 2, 4, 8 or 16 (the rotation is a byte permutation, so any dtype)
void invokeRotateRows(
    void* data, int64_t rows, int64_t cols, int64_t rowStride, int64_t shift, int elemSize, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
