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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp4
{

//! Operands of the speculative step's forward-tail store and next-step staging (kernels/m3_fold_tail.py). Every tensor
//! is int32 except `slots` (int64); strides are in elements. The store views are [slots, width], the sources
//! [rows, width].
struct FoldTailParams
{
    int32_t const* ctrl; // ctrl[0]: the rows the step armed (0: write nothing)
    int64_t const* slots;
    int32_t const* srcNew;
    int64_t sn0, sn1;
    int32_t snW;
    int32_t* dstNew;
    int64_t dn0, dn1;
    int32_t dnW;
    int32_t const* srcNext;
    int64_t sx0, sx1;
    int32_t sxW;
    int32_t* dstNext;
    int64_t dx0, dx1;
    int32_t dxW;
    int32_t const* srcDraft;
    int64_t sd0, sd1;
    int32_t sdW;
    int32_t* dstDraft;
    int64_t dd0, dd1;
    int32_t ddW;
    int32_t const* srcLens;
    int32_t* dstLens;
    int32_t* ids;      // [rows * T]
    int32_t* draftIds; // [rows * (T - 1)]
    int32_t* posOff;   // [rows * T]
    int32_t* kvOff;    // [rows]
    int32_t tokensPerStep;
};

//! One CTA of 32 lanes per row, `rows` CTAs; the first ctrl[0] rows write. Widths and tokensPerStep must be <= 32.
cudaError_t invokeFoldTail(FoldTailParams const& p, int32_t rows, cudaStream_t stream);

//! The same writes from ONE warp (one CTA of 32 lanes looping over the rows, foldTailWarp): the in-kernel form a caller
//! such as the drafter's last-pass draft_step runs in its last CTA; launched alone it is that form's oracle.
cudaError_t invokeFoldTailOneWarp(FoldTailParams const& p, cudaStream_t stream);

#ifdef __CUDACC__
//! Row i of the tail at lane c (0..31), the work of foldTailKernel's CTA i, for a caller that runs the tail inside its
//! own kernel (e.g. the drafter's last-pass draft_step in its last CTA, after its final tokens are written and fenced).
//! The caller runs rows i < *p.ctrl (one warp per row, or one warp looping over the rows); same code as the kernel, so
//! the writes are bit for bit the standalone tail's.
__device__ __forceinline__ void foldTailRow(FoldTailParams const& p, int i, int c)
{
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

//! The whole tail from one warp of a running kernel (lane = threadIdx.x % 32): rows i < *p.ctrl in order.
__device__ __forceinline__ void foldTailWarp(FoldTailParams const& p, int lane)
{
    int const rows = *p.ctrl;
    for (int i = 0; i < rows; ++i)
    {
        foldTailRow(p, i, lane);
    }
}
#endif

} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
