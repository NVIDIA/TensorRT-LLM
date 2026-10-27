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

//! out[i] = float(in[i]) for n bf16 values: the widening places the 16 bf16 bits in the high half of the fp32 word, so
//! every value (+-0, +-inf, subnormals, NaN payloads) is bit for bit at::Tensor::to(float32). 16-byte vector loads and
//! stores when both pointers allow it, a scalar tail otherwise. The MiniMax-M3.1 router gate's input cast
//! (TP2CTX_GATE_CAST): torch's generic dtype-converting copy ran at ~2.8 TB/s on a 16K x 6144 chunk.
void launchCastBf16ToFp32(uint16_t const* in, float* out, int64_t n, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
