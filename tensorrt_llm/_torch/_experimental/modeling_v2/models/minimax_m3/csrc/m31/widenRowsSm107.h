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

//! DEP_LOGITS_WIDEN: out[r, c] = float(in[r * inRowStride + c]) for bf16 (isBf16) or fp16 rows, out contiguous float32
//! [rows, cols]. Widening is exact, so every value (+-0, +-inf, subnormals, NaN payloads) is bit for bit
//! at::Tensor::to(float32) of the strided view (the padded lm_head's vocabulary slice). Grid (column chunks, rows),
//! 256 threads x 2 iterations of one 16-byte load and two 16-byte streaming stores (st.global.cs) per CTA when the
//! row bases allow it, a scalar path otherwise. PDL: waits on its predecessor before the first load, releases its
//! dependents after the last store.
cudaError_t invokeWidenRows(uint16_t const* in, int64_t inRowStride, float* out, int64_t rows, int64_t cols,
    bool isBf16, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
