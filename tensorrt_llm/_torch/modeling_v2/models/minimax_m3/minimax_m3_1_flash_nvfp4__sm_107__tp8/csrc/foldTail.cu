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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/csrc/foldTail.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{

namespace
{

// One CTA per row: row i's next-input tokens, then the fold tail of row i (lane c = column, the index arithmetic of
// next_new_tokens_kernel).
__global__ void __launch_bounds__(32) foldTailNextKernel(FoldTailParams const p, FoldTailNextParams const q)
{
    int const i = blockIdx.x;
    int const c = static_cast<int>(threadIdx.x);
    if (c <= q.draftLen)
    {
        q.next[i * q.nx0 + c] = c == 0
            ? q.accepted[static_cast<int64_t>(q.batchIdx[i]) * q.acc0 + (static_cast<int64_t>(q.numAccepted[i]) - 1)]
            : q.nextDraft[i * q.nd0 + (c - 1)];
    }
    if (i >= *p.ctrl)
    {
        return;
    }
    foldTailRow(p, i, c);
}

} // namespace

cudaError_t invokeFoldTailNext(FoldTailParams const& p, FoldTailNextParams const& q, int32_t rows, cudaStream_t stream)
{
    foldTailNextKernel<<<rows, 32, 0, stream>>>(p, q);
    return cudaGetLastError();
}

} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
