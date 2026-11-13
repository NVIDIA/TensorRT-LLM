/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// MiniMax-M3.1 NVFP4 index-K cache: the writer and the prefill gather.
//
// Page layout (the vLLM / MSA Q8KV4 contract, one 128-token page, one index head, head_dim 128):
//   bytes [0, 8192): packed E2M1 codes, token t at t * 64, low nibble = even element;
//   bytes [8192, 9216): E4M3 scales of 16-element groups, token t group g at 8192 + t * 8 + g (linear).
// No global scale. A page may sit at any 16-byte-aligned stride.

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::minimaxM31
{

//! Quantize index-K rows into their tokens' NVFP4 pages (idxkNvfp4QuantStore's recipe, one warp per token).
//! rows: numTokens rows of 128 elements, bf16 (rowsFp8 false) or E4M3 (rowsFp8 true), unit inner stride, row stride
//! rowStride elements. With rowsBySlot the rows are instead an FP8 index-K pool [pages, 1, 128, 128] (row stride
//! rowStride, page stride rowsPageStride elements) read at each token's slot. Slots come from loc64 (int64) or loc32
//! (int32), exactly one non-null; slots < 0 are skipped. pages: the NVFP4 cache, page stride pageStride bytes.
cudaError_t invokeNvfp4IdxkWrite(void const* rows, bool rowsFp8, int64_t rowStride, int64_t rowsPageStride,
    bool rowsBySlot, int64_t const* loc64, int32_t const* loc32, int numTokens, uint8_t* pages, int64_t pageStride,
    cudaStream_t stream);

//! Dequantize numPages NVFP4 pages (pageIds[i] < 0: zeros) into out [numPages, 128, 128] E4M3: each element becomes
//! E4M3_RN_satfinite(e2m1 * scale), the exact fp32 product rounded once.
cudaError_t invokeNvfp4IdxkGatherFp8(
    uint8_t const* pages, int64_t pageStride, int32_t const* pageIds, int numPages, uint8_t* out, cudaStream_t stream);

} // namespace kernels::minimaxM31

TRTLLM_MV2_NAMESPACE_END
