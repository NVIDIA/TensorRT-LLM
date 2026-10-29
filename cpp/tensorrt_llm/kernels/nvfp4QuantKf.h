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

//! BF16 [m, n] (rows contiguous) -> NVFP4: E2M1 codes [m, n / 2] (two per byte, the even element in the low nibble)
//! and one E4M3 scale per 16 elements in the linear layout [m, n / 16]. On finite inputs the bytes equal those of
//! invokeFP4Quantization<__nv_bfloat16, 16>(..., useUE8M0 = false, QuantizationSFLayout::LINEAR, ...).

//! True when the kernel can run: SM10x, m > 0, n > 0 and a multiple of 16, m * n / 16 within int32, a 32-byte aligned
//! input and an 8-byte aligned output.
bool isNvfp4QuantKfSupported(void const* input, void const* output, int64_t m, int64_t n);

//! Quantizes input: output holds m * n / 2 bytes, scales m * n / 16 bytes, globalScale points to one device float.
void invokeNvfp4QuantKf(__nv_bfloat16 const* input, float const* globalScale, uint8_t* output, uint8_t* scales,
    int64_t m, int64_t n, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
