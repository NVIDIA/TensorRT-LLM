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
namespace minimaxM31
{

//! DSPARK_GLUE_CPP: the DSpark / DFlash drafter's context advance + rebased read view, bit for bit the Triton
//! _advance_kernel (catalog/spec_decode/dspark_ctx_advance.py). For each generation request r (one block each, so the
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
//! (catalog/spec_decode/dspark_forward_indices.py), all outputs int32 (int64 -> int32 keeps the low 32 bits):
//!   seqlens = numCtx; batchOut = batchIdx; after = seqlens + block; last = floor-mod(after - 1, pageSize) + 1;
//!   indptr[b] = b * pagesPerSlot (b <= B); append[t] = seqlens[t / block] + t % block;
//!   qflat[t] = qpos[t / block, t % block]  (qpos strided by q0 / q1 elements).
//! numCtx / batchIdx / qpos are int32 or int64 (the *Is64 flags). Graph-safe.
void invokeDsparkIndices(void const* numCtx, bool numCtxIs64, void const* batchIdx, bool batchIdxIs64, void const* qpos,
    bool qposIs64, int64_t q0, int64_t q1, int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr,
    int32_t* last, int32_t* append, int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize,
    cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
