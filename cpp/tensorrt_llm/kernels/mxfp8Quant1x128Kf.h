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

// BF16 [m, n] -> FP8 E4M3 [m, n] with one UE8M0 scale per 1x128 block, replicated into the four K32 slots of the
// R128c4 (128x4 swizzled) scale layout [ceil(m / 128), n / 128, 32, 4, 4]; the scale bytes of rows m ..
// ceil(m / 128) * 128 - 1 are zero. On finite inputs the bytes equal those of
// invokeMxFP8Quantization<__nv_bfloat16, 128, 32>(..., QuantizationSFLayout::SWIZZLED, ...).

// True when the kernel can run: SM10x, m > 0 and n within int32, n a multiple of 128, and 32-byte aligned input,
// output and scales.
bool isMxFP8Quant1x128KfSupported(void const* input, void const* output, void const* scales, int64_t m, int64_t n);

// output holds m * n bytes, scales ceil(m / 128) * 128 * n / 32 bytes.
void invokeMxFP8Quant1x128Kf(
    __nv_bfloat16 const* input, uint8_t* output, uint8_t* scales, int64_t m, int64_t n, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
