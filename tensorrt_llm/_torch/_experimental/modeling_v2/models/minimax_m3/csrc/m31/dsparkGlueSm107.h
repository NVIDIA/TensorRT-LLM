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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! DSPARK_GLUE_CPP: the DSpark / DFlash drafter's context advance + rebased read view, bit for bit the Triton
//! _advance_kernel (ops/dspark/dspark_ctx_advance.py). For each generation request r (one block each, so the
//! in-place ctx_len update keeps the one-program-per-request ordering):
//!   ctxLen[slot] = min(ctxLen[slot] + nacc[r], maxCtx)    (not for the dummy slot); ctxLen[dummySlot] = 0 (request 0)
//!   numCtx       = min(new (0 on the dummy slot), max(counts[row] * page - blockSize, 0))
//!   rebase       = min(basePages[slot], numCtx / page);  out[r] = max(numCtx - basePages[slot] * page, 0)
//!   readView[row, j] = tables[row, min(j + rebase, width - 1)]                                            (int32)
//! nacc is int32 (naccIs64 false) or int64. Integer arithmetic and copies only. Graph-safe.
void invokeDsparkAdvance(int64_t const* slots, int64_t const* rows, void const* nacc, bool naccIs64, int64_t* ctxLen,
    int64_t const* counts, int64_t const* basePages, int32_t const* tables, int32_t* readView, int64_t* out, int64_t n,
    int64_t width, int64_t tablesStride, int64_t readStride, int64_t maxCtx, int64_t dummySlot, int64_t page,
    int64_t blockSize, cudaStream_t stream);

//! DSPARK_GLUE_CPP: the paged dflash_forward's index prelude in one launch, bit for bit the Triton _indices_kernel
//! (ops/dspark/dspark_forward_indices.py), all outputs int32 (int64 -> int32 keeps the low 32 bits):
//!   seqlens = numCtx; batchOut = batchIdx; after = seqlens + block; last = floor-mod(after - 1, pageSize) + 1;
//!   indptr[b] = b * pagesPerSlot (b <= B); append[t] = seqlens[t / block] + t % block;
//!   qflat[t] = qpos[t / block, t % block]  (qpos strided by q0 / q1 elements).
//! numCtx / batchIdx / qpos are int32 or int64 (the *Is64 flags). Graph-safe.
void invokeDsparkIndices(void const* numCtx, bool numCtxIs64, void const* batchIdx, bool batchIdxIs64, void const* qpos,
    bool qposIs64, int64_t q0, int64_t q1, int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr,
    int32_t* last, int32_t* append, int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize,
    cudaStream_t stream);

//! DSPARK_GLUE_CPP (ADV_PAR): invokeDsparkAdvance's outputs in two launches, bit for bit the Triton
//! _advance_only_kernel + _read_view_kernel pair it replaces: an advance-only launch (one thread per request) stores
//! each request's rebase into the int64 [n] scratch rebase, then a copy launch writes the read view with grid
//! (n, ceil(width / 512)), 128 threads x 4 columns per CTA. Same values as invokeDsparkAdvance (a pure index copy).
//! enablePdl: launched with programmatic stream serialization; each kernel runs griddepcontrol.wait (then
//! launch_dependents) before its first dependent read. n <= 0: nothing launched; width 0: no copy launch.
//! Graph-safe.
cudaError_t invokeDsparkAdvancePar(int64_t const* slots, int64_t const* rows, void const* nacc, bool naccIs64,
    int64_t* ctxLen, int64_t const* counts, int64_t const* basePages, int32_t const* tables, int32_t* readView,
    int64_t* out, int64_t* rebase, int64_t n, int64_t width, int64_t tablesStride, int64_t readStride, int64_t maxCtx,
    int64_t dummySlot, int64_t page, int64_t blockSize, bool enablePdl, cudaStream_t stream);

//! DSPARK_GLUE_CPP: the DSpark / DFlash drafter block's q / k RMSNorm + partial NeoX RoPE in one launch (head_dim 128),
//! bit for bit the Triton _qk_norm_rope_kernel it replaces (ops/dspark/dspark_qk_norm_rope.py, itself bit for
//! bit flashinfer RMSNorm x2 + apply_rope_with_cos_sin_cache), with the optional paged HND K / V append
//! (DRAFT_APPEND_FUSE). One warp per (row t, head h < HQ + HKV), 4 warps per CTA:
//!   RMSNorm: 16 partial sums of 8 contiguous x^2 (sequential, mul.rn / add.rn), pair tree, total * (1 / 128) + eps,
//!     rstd = rsqrt.approx.ftz; y = bf16_rn((x * rstd) * w).
//!   RoPE (rotated dims [0, 2 half)): x1 / x2 = float(y_i) / float(y_{i + half});
//!     o1 = bf16_rn(fma.rn(x1, cos_i, (-x2) * sin_i)), o2 = bf16_rn(fma.rn(x2, cos_i, x1 * sin_i)).
//! qkv: bf16 rows of stride qkvStride (q heads at column 0, k heads at HQ * 128, v at vCol); pos: int32 [T];
//! cs: fp32 [*, 2 half] rows of stride csStride; wq / wk: bf16 [128]; qOut [T, HQ * 128] / kOut [T, HKV * 128].
//! store: also write K (kOut's bits) and V (copied from qkv) into cache [pages, 2, HKV, ps, 128] (page stride
//! cachePageStride) at page kvi[kvp[bidx[t]] + apos[t] / ps], slot apos[t] % ps; idx64 bit j = the j-th index tensor
//! (bidx, apos, kvi, kvp) is int64, else int32. Vector loads when every operand is aligned for them, else the same
//! kernel's scalar-load instantiation (same bits). half must be a multiple of 4 in [4, 64]. T <= 0: nothing launched.
cudaError_t invokeDsparkQkNormRope(__nv_bfloat16 const* qkv, int64_t qkvStride, int32_t const* pos, float const* cs,
    int64_t csStride, __nv_bfloat16 const* wq, __nv_bfloat16 const* wk, __nv_bfloat16* qOut, __nv_bfloat16* kOut,
    float epsQ, float epsK, __nv_bfloat16* cache, int64_t cachePageStride, void const* bidx, void const* apos,
    void const* kvi, void const* kvp, int32_t idx64, int64_t vCol, int32_t T, int32_t HQ, int32_t HKV, int32_t half,
    int32_t ps, bool store, bool enablePdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
