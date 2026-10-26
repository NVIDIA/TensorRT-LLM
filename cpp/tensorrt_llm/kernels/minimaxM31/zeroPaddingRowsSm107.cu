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

#include "tensorrt_llm/kernels/minimaxM31/zeroPaddingRowsSm107.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

constexpr int kThreads = 256;

// One CTA per (row, chunk): a row belongs to request row / rowsPerRequest; it is zeroed only when that request's slot
// id is the CUDA-graph padding marker. A real row is never touched, so at a full batch the kernel writes nothing.
__global__ void __launch_bounds__(kThreads) zeroPaddingRowsKernel(float* __restrict__ logits, int64_t rowStride,
    int64_t const* __restrict__ slotIds, int64_t dummySlotRow, int32_t rowsPerRequest, int32_t d, int32_t chunk)
{
    int64_t const row = blockIdx.y;
    if (slotIds[row / rowsPerRequest] != dummySlotRow)
    {
        return;
    }
    int32_t const begin = blockIdx.x * chunk;
    int32_t const end = min(d, begin + chunk);
    float* dst = logits + row * rowStride;
    for (int32_t i = begin + threadIdx.x; i < end; i += kThreads)
    {
        dst[i] = 0.0f;
    }
}

} // namespace

void invokeZeroPaddingRows(float* logits, int64_t rowStride, int64_t const* slotIds, int64_t dummySlotRow, int32_t rows,
    int32_t rowsPerRequest, int32_t d, cudaStream_t stream)
{
    int32_t const chunk = 16384;
    dim3 const grid((d + chunk - 1) / chunk, rows);
    zeroPaddingRowsKernel<<<grid, kThreads, 0, stream>>>(
        logits, rowStride, slotIds, dummySlotRow, rowsPerRequest, d, chunk);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
