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

//! DEP_MARKOV_GLUE's widening add: out[r, c] = base[r * baseRowStride + c] + float(bias[r, c]). base: float32 rows
//! (gen_logits[:, step]); bias: contiguous bf16 (biasIsBf16) or float32 [rows, cols]; out: contiguous float32
//! [rows, cols]. One IEEE fp32 add of two exactly widened values, so bit for bit torch's
//! gen_logits[:, step] + bias.to(float32). Grid (column chunks, rows), 128 threads x 8 iterations of 4 columns per CTA;
//! 16-byte base loads / out stores (8- or 16-byte bias loads) when the row bases allow it, a scalar path otherwise.
//! PDL: waits on its predecessor before the first load, releases its dependents after the last store.
cudaError_t invokeWidenAddRows(float const* base, int64_t baseRowStride, void const* bias, bool biasIsBf16, float* out,
    int64_t rows, int64_t cols, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
