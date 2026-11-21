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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

// SM120 NVFP4 W4A4 gate_up GEMM + SwiGLU + NVFP4 quantize epilogue (see nvfp4GateUpSwigluSm120.cu for the
// gate_up weight-row layout it expects). A: [m, k / 2] packed E2M1 activations with inputSf (128x4-swizzled), B:
// [n, k / 2] packed E2M1 permuted gate_up weights with weightSf, alpha: the GEMM's global scale, outScale: down_proj's
// global input scale. Writes fp4Out [m, n / 4] bytes and sfOut (128x4-swizzled, padding rows zeroed).
// tile 0: 256x128x128; tile 1: 128x128x256; tile 2: 128x128x256 with the MMAs of tile rows 64-127 skipped (m <= 64).
// Returns the workspace size when A, B and fp4Out are all null.
size_t nvfp4GateUpSwigluSm120(void const* A, void const* B, void const* inputSf, void const* weightSf,
    float const* alpha, float const* outScale, uint8_t* fp4Out, uint8_t* sfOut, int m, int n, int k, int tile,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, bool enablePdl);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
