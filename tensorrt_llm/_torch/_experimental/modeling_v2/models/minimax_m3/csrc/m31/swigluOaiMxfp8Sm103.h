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

//! SWIGLU_MXFP8 (prefill): the swigluoai activation of invokeSwigluOai (bit for bit the Triton silu_and_mul at limit 7,
//! alpha 1.702, beta 1) followed, while the result is still in registers, by the MXFP8 quantize of
//! mxfp8_quantize(h, swizzled = true) (quantize_with_block_size<FP16_TO_MXFP8, bf16, 32, UE8M0>: E4M3 [rows, d] and
//! UE8M0 K32 scales in the swizzled 128x4 layout, the scale rows that pad the row count to a multiple of 128 zeroed).
//! x bf16 [rows, 2d] contiguous; qOut [rows, d] E4M3 (as uint64 x 8); sfOut pad128(rows) * d / 32 bytes; hOut (the bf16
//! activation, [rows, d]) is optional (nullptr skips it). d must be a multiple of 256 and at most 8192. With `pdl` the
//! grid waits on its dependency and reports completion after its last store (as quantize_with_block_size).
//!
//! SWIGLU_MXFP8_ROWS (default on, from 1024 rows): each CTA keeps kRows rows in flight (all 2 * kRows 16-byte loads
//! issued before the first row's math; kRows 4) on a grid of
//! min(pad128(rows), resident CTAs x SMs) with the resident CTAs from the occupancy API, where the one-row launch
//! assumes 2048 threads per SM (sm_107 holds 1024, so two 384-thread CTAs at d 3072, not five). Only the row-to-CTA map
//! changes: every output byte is the one-row kernel's. An occupancy query that fails or finds no resident CTA keeps
//! the one-row launch. rowsForm = false (the SWIGLU_MXFP8_ROWS kill switch) keeps the one-row launch at every row
//! count; kRows picks the rows in flight per CTA (2, 4 or 8; other values: 4).
cudaError_t invokeSwigluOaiMxfp8(uint4 const* x, uint64_t* qOut, uint32_t* sfOut, uint4* hOut, int64_t rows, int64_t d,
    int multiProcessorCount, bool pdl, cudaStream_t stream, bool rowsForm = true, int kRows = 4);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
