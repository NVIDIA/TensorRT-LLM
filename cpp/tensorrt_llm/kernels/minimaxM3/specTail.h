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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
//! Small fused kernels of the MiniMax-M3 Eagle3 step: the drafter's fc_norm RMSNorms over the captured hidden states
//! and the step's input / output bookkeeping (GB300, decode). All launches use programmatic dependent launch where
//! noted.
namespace specTail
{

//! Threads per CTA of the norm kernels, and the hidden-size bound of their register tiles:
//! H % 8 == 0 and H <= kMaxIters * kThreads * 8.
constexpr int kThreads = 256;
constexpr int kMaxIters = 4;

//! out[r] = [rmsnorm(table[ids[r]]) * wEmbed, rmsnorm(hidden[rows ? rows[r] : r]) * wHidden] ([M, 2H] BF16; ids
//! outside [0, vocab) embed to zero); residual[r] = hidden[rows[r]] when residual is set.
cudaError_t invokeDraftInput(int const* ids, int vocab, __nv_bfloat16 const* table, __nv_bfloat16 const* hidden,
    long long const* rows, __nv_bfloat16 const* wEmbed, __nv_bfloat16 const* wHidden, float epsEmbed, float epsHidden,
    int H, int M, __nv_bfloat16* out, __nv_bfloat16* residual, bool pdl, cudaStream_t stream);

//! dst[r, c*H:(c+1)*H] = rmsnorm(src[r, c*H:(c+1)*H]) * w[c] for c < C (src row stride srcStride), dst [M, C*H].
cudaError_t invokeChunkNorm(__nv_bfloat16 const* src, long long srcStride, __nv_bfloat16 const* w, float eps, int H,
    int C, int M, __nv_bfloat16* dst, bool pdl, cudaStream_t stream);

//! dst[r, :H] = rmsnorm(bf16(h[r] + res[r])) * w (res optional), with chunk-norm rounding.
cudaError_t invokeCaptureNorm(__nv_bfloat16 const* h, long long hStride, __nv_bfloat16 const* res, long long resStride,
    __nv_bfloat16 const* w, float eps, int H, int M, __nv_bfloat16* dst, long long dstStride, cudaStream_t stream);

//! The overlap scheduler's per-step input gathers (see step_gather_kernel); work = the largest of its four loops.
cudaError_t invokeStepGather(int const* newTok, long long ntS0, long long ntS1, int const* nextDraft, long long ndS0,
    long long ndS1, int const* lens, int const* slots, int const* ppi, int P, int T, int D, int* ids, int ids0,
    int* draft, int d0, int* posOff, int nPos, int* kvOff, int nKv, int b0, int work, cudaStream_t stream);

//! A strided int32 2-D view: element (i, c) at data[i * s0 + c * s1], `width` columns.
struct IntView
{
    int* data;
    long long s0;
    long long s1;
    int width;
};

//! The spec sampler's slot stores (see slot_scatter_kernel): for i < n, da / db / dd rows slots[i] get sa / sb / sd
//! row i, zero-padded to the destination widths, and dl[slots[i]] = sl[i].
cudaError_t invokeSlotScatter(long long const* slots, int n, IntView sa, IntView da, IntView sb, IntView db,
    int const* sl, int* dl, IntView sd, IntView dd, cudaStream_t stream);

} // namespace specTail
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
