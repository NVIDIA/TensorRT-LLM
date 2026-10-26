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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/foldTail.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{

namespace
{

// Row i of the step (lane c = column), as the Triton fold tail's program i: the store's padded rows, then the next
// step's inputs read from them (zero past each source width).
__global__ void __launch_bounds__(32) foldTailKernel(FoldTailParams const p)
{
    int const i = blockIdx.x;
    if (i >= *p.ctrl)
    {
        return;
    }
    foldTailRow(p, i, static_cast<int>(threadIdx.x));
}

__global__ void __launch_bounds__(32) foldTailOneWarpKernel(FoldTailParams const p)
{
    foldTailWarp(p, static_cast<int>(threadIdx.x));
}

} // namespace

cudaError_t invokeFoldTail(FoldTailParams const& p, int32_t rows, cudaStream_t stream)
{
    foldTailKernel<<<rows, 32, 0, stream>>>(p);
    return cudaGetLastError();
}

cudaError_t invokeFoldTailOneWarp(FoldTailParams const& p, cudaStream_t stream)
{
    foldTailOneWarpKernel<<<1, 32, 0, stream>>>(p);
    return cudaGetLastError();
}

} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
