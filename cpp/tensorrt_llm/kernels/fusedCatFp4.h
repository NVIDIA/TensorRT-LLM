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
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Fused concat + FP4 E2M1 per-block-32 quantization for the DSA indexer.
///
/// Given two BF16 input matrices `pe` [M, pe_dim] and `nope` [M, nope_dim],
/// this kernel concatenates them along the last dimension (pe first, nope second),
/// then quantizes each row to FP4 E2M1 with one UE8M0 scale per 32-element block
/// (head_dim=128 => 4 scales per row packed as one int32, little-endian).
///
/// Inputs need not be fully contiguous — only the innermost dimension must be
/// contiguous (stride 1). The row stride for each input is provided explicitly
/// via pe_row_stride / nope_row_stride, allowing processing non-contiguous
/// views (e.g. from torch.split()) without a prior contiguous copy.
///
/// Output packing (matches DeepGEMM `per_token_cast_to_fp4`, gran_k=32,
/// use_packed_ue8m0=True):
///   - packed[row, 2t + 0] = code[4t] | (code[4t+1] << 4)
///   - packed[row, 2t + 1] = code[4t+2] | (code[4t+3] << 4)
///   - scale[row] = ue8m0[block0] | (ue8m0[block1]<<8) | (ue8m0[block2]<<16) | (ue8m0[block3]<<24)
///
/// @param packed_out      Output packed FP4 bytes [M, head_dim/2], int8, row-major.
///                        Two E2M1 codes per byte (even-index in low nibble,
///                        odd-index in high nibble).
/// @param scale_out       Output UE8M0 scales [M, 1], int32. Each int32 packs
///                        four 8-bit exponents little-endian (block 0 → bits 0..7).
/// @param pe              Input PE part, BF16. Each row has pe_dim contiguous elements.
/// @param nope            Input non-PE part, BF16. Each row has nope_dim contiguous elements.
/// @param M               Number of rows (product of all dims except the last).
/// @param pe_dim          Dimension of PE input (must satisfy pe_dim + nope_dim == head_dim,
///                        and pe_dim must be a multiple of 4).
/// @param nope_dim        Dimension of non-PE input.
/// @param head_dim        Total head dimension (must be 128).
/// @param pe_row_stride   Stride (in elements) between consecutive rows of pe.
///                        For contiguous layout this equals pe_dim; for non-contiguous
///                        views (e.g. from torch.split) it may be larger.
/// @param nope_row_stride Stride (in elements) between consecutive rows of nope.
/// @param stream          CUDA stream.
void invokeFusedCatFp4(int8_t* packed_out, int32_t* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride,
    cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
