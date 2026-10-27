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

namespace kernels::minimaxM31Sm103
{

//! MiniMax-M3 CTX prefill: the residual add + RMSNorm of the all-reduce's RESIDUAL_RMS_NORM epilogue
//! (rms_norm_rows_kernel, bitwise the same) fused with the MXFP8 quantization of its output that the next MXFP8
//! linear performs (quantize_mxfp8_rows: K32 UE8M0 scales, swizzled 128x4 layout, bitwise the same).
//! In place: `inter` (the all-reduce output) becomes `inter + residual`, as in the stock epilogue.
//! normOut may be null (the BF16 norm output is then not written). With fp4Out (and fp4SfOut / fp4GlobalScale) set,
//! it also writes fp4_quantize(norm, globalScale, 16, ue8m0 = false, swizzled = false) of the same rows.
//! With f32Out set (requires normOut; NVFP4 optional), it also writes normOut widened to FP32 (bit-exact
//! normOut.float()). hidden % 256 == 0, hidden / 8 <= 1024.
void invokeCtxResidualRmsNormMxfp8(__nv_bfloat16* inter, __nv_bfloat16 const* residual, __nv_bfloat16 const* weight,
    __nv_bfloat16* normOut, uint64_t* qOut, uint32_t* sfOut, uint32_t* fp4Out, uint32_t* fp4SfOut,
    float const* fp4GlobalScale, float* f32Out, int rows, int hidden, float eps, bool enablePdl, cudaStream_t stream);

} // namespace kernels::minimaxM31Sm103

TRTLLM_NAMESPACE_END
