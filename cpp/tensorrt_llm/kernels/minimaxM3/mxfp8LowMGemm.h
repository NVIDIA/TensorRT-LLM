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

//! Whether `invokeMinimaxM3Mxfp8LowMGemm` is built for this (warps, clusterK, rowTiles) configuration at M rows:
//! (4, 4, 2), (8, 2, 4), (4, 4, 1) and (4, 2, 2), each for 1 <= M <= 8.
bool isMinimaxM3Mxfp8LowMGemmSupported(int32_t warps, int32_t clusterK, int32_t rowTiles, int32_t m);

//! Fused MXFP8 activation quantization + low-M MXFP8 GEMM: `out[M, N] = mxfp8(x)[M, K] @ w[N, K]^T` in BF16.
//!
//! `x` is BF16 [M, K] with row stride `ldx`; `w` is E4M3 [N, K] row-major with UE8M0 scales `wsf` in the 128x4
//! swizzled layout of `mxfp8_quantize(w, True)`; `out` is BF16 [M, N] with row stride `ldo`. Each CTA owns
//! 16 * rowTiles weight rows and K / clusterK columns; `warps` warps per CTA.
void invokeMinimaxM3Mxfp8LowMGemm(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t n, int32_t k,
    uint8_t const* w, uint8_t const* wsf, __nv_bfloat16* out, int32_t ldo, int32_t warps, int32_t clusterK,
    int32_t rowTiles, bool pdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
