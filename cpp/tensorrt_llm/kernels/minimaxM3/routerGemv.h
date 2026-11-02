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

//! Low-M fp32 router GEMV: `out[M, E] = float(x[M, K]) @ w[E, K]^T` with exact fp32 FMA, one CTA per expert,
//! 1 <= M <= 16, K % 1024 == 0. With `qOut` the CTAs also emit the NVFP4 activation (packed e2m1 [M, K / 2], linear
//! E4M3 scales [M, K / 16]) for global scale `*sfScale`, bit-identical to `fp4_quantize(x, sfScale, 16, False,
//! False)`.
void invokeMinimaxM3RouterGemv(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t k, float const* w, int32_t e,
    float* out, uint8_t* qOut, uint8_t* sfOut, float const* sfScale, bool pdl, cudaStream_t stream);

//! TF32 tensor-core router for 128 experts and 1 <= M <= 48 (K % 1024 == 0): `out[M, 128]` with the weights rounded
//! to TF32 (round to nearest, ties away) and fp32 accumulation, independent of the cluster size picked. Optional
//! NVFP4 activation as in `invokeMinimaxM3RouterGemv`.
void invokeMinimaxM3RouterTc(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t k, float const* w, float* out,
    uint8_t* qOut, uint8_t* sfOut, float const* sfScale, bool pdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
