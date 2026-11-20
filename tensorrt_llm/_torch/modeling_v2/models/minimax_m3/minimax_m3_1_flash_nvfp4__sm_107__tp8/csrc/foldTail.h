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
namespace minimaxM3Sm107Tp8
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

//! FOLD_F2: the step's next-input tokens the tail writes itself, spec_next_new_tokens' values
//! (ops/dspark/strict_accept_kernel.py next_new_tokens_kernel): row i is
//! accepted[batchIdx[i] * acc0 + numAccepted[i] - 1], then nextDraft[i * nd0 + 0 .. draftLen). `next` is
//! FoldTailParams::srcNext (the step's next_new_tokens output, unit column stride). All int32.
struct FoldTailNextParams
{
    int32_t const* accepted;
    int64_t acc0;
    int32_t const* numAccepted; // [rows]
    int32_t const* batchIdx;    // [rows]
    int32_t const* nextDraft;
    int64_t nd0;
    int32_t draftLen;
    int32_t* next; // [rows, draftLen + 1]
    int64_t nx0;
};

//! One CTA of 32 lanes per row, `rows` CTAs: every row's next-input tokens (FoldTailNextParams; draftLen + 1 <= 32),
//! then the first ctrl[0] rows store and stage them as trtllm::minimax_m3_sm107_tp8_fold_tail does (each lane reads
//! back only its own write). Widths and tokensPerStep must be <= 32.
cudaError_t invokeFoldTailNext(FoldTailParams const& p, FoldTailNextParams const& q, int32_t rows, cudaStream_t stream);

#ifdef __CUDACC__
//! Row i of the tail at lane c (0..31): the stores and next-step staging of trtllm::minimax_m3_sm107_tp8_fold_tail's
//! CTA i, so the writes are bit for bit the standalone tail's.
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
#endif

} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
