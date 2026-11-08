/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Fused concat + FP8 1x128 quantization.
///
/// Given two BF16 input matrices `pe` [M, pe_dim] and `nope` [M, nope_dim],
/// this kernel concatenates them along the last dimension (pe first, nope second),
/// then quantizes each row to FP8 E4M3 with one scale factor per row.
///
/// Inputs need not be fully contiguous — only the innermost dimension must be
/// contiguous (stride 1).  The row stride for each input is provided explicitly
/// via pe_row_stride / nope_row_stride, which allows processing non-contiguous
/// views (e.g. from torch.split()) without a prior contiguous copy.
///
/// @param fp8_out         Output FP8 data [M, head_dim], row-major.
/// @param scale_out       Output scales [M, 1], float32.  When use_ue8m0 is true,
///                        the scale is stored as UE8M0 (power-of-two) in float bits.
/// @param pe              Input PE part, BF16. Each row has pe_dim contiguous elements.
/// @param nope            Input non-PE part, BF16. Each row has nope_dim contiguous elements.
/// @param M               Number of rows (product of all dims except the last).
/// @param pe_dim          Dimension of PE input (must satisfy pe_dim + nope_dim == head_dim).
/// @param nope_dim        Dimension of non-PE input.
/// @param head_dim        Total head dimension (must be 128, power of 2).
/// @param pe_row_stride   Stride (in elements) between consecutive rows of pe.
///                        For contiguous layout this equals pe_dim; for non-contiguous
///                        views (e.g. from torch.split) it may be larger.
/// @param nope_row_stride Stride (in elements) between consecutive rows of nope.
/// @param use_ue8m0       If true, use UE8M0 (power-of-two) scale format.
/// @param stream          CUDA stream.
void invokeFusedCatFp8(__nv_fp8_e4m3* fp8_out, float* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride,
    bool use_ue8m0, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
