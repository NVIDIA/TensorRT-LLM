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

#include <cstddef>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! FlashInfer's Path A slice size (OnlineSoftmax DEFAULT_SLICE_SIZE): the Map kernel's grid is rows x
//! ceil(d / kSoftmaxMcSliceSize) CTAs, one partial (max, denominator) float2 each.
constexpr int kSoftmaxMcSliceSize = 8192;

//! ST-01-SM: FlashInfer softmax(logits, temperature) Path A (OnlineSoftmaxMapKernel + OnlineSoftmaxReduceKernel,
//! batch <= 128 and vocab >= 24576) with the Reduce kernel's write pass spread over writeCtas CTAs per row instead of
//! one (FlashInfer: one 1024-thread CTA per row, ~19 us at 7-8 rows of a 200064 vocab on GB300).
//!
//! Semantics: logits fp32 [rows, d] (contiguous, d % 4 == 0, 16-byte aligned) -> out fp32 [rows, d] (same layout),
//! out[r, i] = softmax(logits[r,:] / T_r)[i] with T_r = temperatureArr[r] when temperatureArr is not null, else
//! temperatureVal (inverse temperature 0 when T_r == 0, as FlashInfer).
//!
//! Bitwise contract vs flashinfer.sampling.softmax(logits, temperature) (FlashInfer 0.6.15 == 0.6.18, Path A):
//!   - the Map kernel is FlashInfer's OnlineSoftmaxMapKernel<1024, 4, float> (same grid, same cub BlockReduce calls,
//!     same per-element source form), writing the same per-slice partials;
//!   - every one of a row's writeCtas CTAs runs FlashInfer's Reduce prologue unchanged (the same cub
//!     BlockReduce<float2, 1024> with Float2SoftmaxReduceOp over the same partials in the same order), so each gets
//!     FlashInfer's final max and 1 / denominator bit for bit, and writes only its share of the row's 4096-element
//!     tiles (tiles writeCtas apart) with the Reduce kernel's per-element arithmetic. The write is elementwise, so the
//!     split changes no value.
//! Validated bitwise at writeCtas 4 / 8 / 16 on GB300 (microbenchmark): 9.2 vs FlashInfer 21.8 us per call
//! at 7-8 rows with writeCtas 16.
//!
//! MUST be compiled with --use_fast_math (FlashInfer's JIT flags: __expf, the approximate 1 / x, flush-to-zero);
//! cpp/tensorrt_llm/kernels/CMakeLists.txt sets the flag on this one source.
//!
//! partials: rows x ceil(d / kSoftmaxMcSliceSize) float2 (softmaxMcPartialsBytes), 8-byte aligned; fully overwritten
//! by every launch (no initialization, CUDA-graph safe). Launches that share it must be stream-ordered. Two plain
//! launches (no PDL), no co-residency requirement. The caller keeps the contract rows <= 128 and d >= 24576 (the
//! shapes where FlashInfer itself takes Path A); 1 <= writeCtas <= 65535.
cudaError_t invokeSoftmaxMc(float const* logits, float* out, float const* temperatureArr, float temperatureVal,
    int rows, int d, int writeCtas, void* partials, cudaStream_t stream);

//! Bytes of invokeSoftmaxMc's partials workspace for rows x d.
size_t softmaxMcPartialsBytes(int rows, int d);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
