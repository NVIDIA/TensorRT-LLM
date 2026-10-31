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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/foldTail.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp2
{

namespace
{

// Row i of the step (lane c = column), as the Triton fold tail's program i: the store's padded rows, then the next
// step's inputs read from them (zero past each source width).
__global__ void __launch_bounds__(32) foldTailKernel(FoldTailParams const p)
{
    int const i = blockIdx.x;
    int const c = threadIdx.x;
    if (i >= *p.ctrl)
    {
        return;
    }
    int64_t const s = p.slots[i];
    int32_t const nw = c < p.snW ? p.srcNew[i * p.sn0 + c * p.sn1] : 0;
    if (c < p.dnW)
    {
        p.dstNew[s * p.dn0 + c * p.dn1] = nw;
    }
    int32_t const nxt = c < p.sxW ? p.srcNext[i * p.sx0 + c * p.sx1] : 0;
    if (c < p.dxW)
    {
        p.dstNext[s * p.dx0 + c * p.dx1] = nxt;
    }
    int32_t const drf = c < p.sdW ? p.srcDraft[i * p.sd0 + c * p.sd1] : 0;
    if (c < p.ddW)
    {
        p.dstDraft[s * p.dd0 + c * p.dd1] = drf;
    }
    int32_t const ln = p.srcLens[i];
    int32_t const t = p.tokensPerStep;
    int32_t const d = t - 1;
    if (c == 0)
    {
        p.dstLens[s] = ln;
        p.kvOff[i] = ln - t;
    }
    if (c < t)
    {
        p.ids[i * t + c] = c < p.dxW ? nxt : 0;
        p.posOff[i * t + c] = ln;
    }
    if (c < d)
    {
        p.draftIds[i * d + c] = c < p.ddW ? drf : 0;
    }
}

} // namespace

cudaError_t invokeFoldTail(FoldTailParams const& p, int32_t rows, cudaStream_t stream)
{
    foldTailKernel<<<rows, 32, 0, stream>>>(p);
    return cudaGetLastError();
}

} // namespace minimaxM3Sm107Tp2
} // namespace kernels

TRTLLM_NAMESPACE_END
