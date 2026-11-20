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
namespace minimaxM31
{

//! TAILGLUE_A: the DSpark drafter's post-join glue of a forked decode step in one launch, bit for bit the four
//! launches it replaces (and the Triton _tail_glue_kernel of the TP4 target's kernels/m3_tail_glue.py):
//!  * the context K / V store (invokeDsparkCtxStore with colFromPos and bound: columns from the positions, clamped to
//!    counts[row] * storePage - 1 and 0), one warp per (row, layer, kv head);
//!  * the context advance (invokeDsparkAdvancePar's advance-only kernel), one thread per request, which also writes
//!    the request's forward indices (invokeDsparkIndices with numCtx = the advertised length just computed,
//!    batchIdx = rows, pagesPerSlot = width);
//!  * the rebased read view (invokeDsparkAdvancePar's copy kernel), one CTA per (request, 512 columns). The rebase is
//!    recomputed from the request's context length before the advance, read as pos[r * kp1] (the drafter inputs
//!    wrote ctx_len[slot] + j there), with the advance's own integer lines.
//! No CTA reads what another CTA of the launch writes (cache pages / context lengths + outputs / read table), so the
//! launch order does not matter. Index operands: pos / rows / counts / slots / basePages / ctxLen int64, table and
//! readView int32 (unit column stride), nacc and qpos int32 or int64. PDL-aware (griddepcontrol.wait at entry).
struct TailGlueParams
{
    // ---- the store
    void const* kv;        // bf16 [n * kp1, L * 2 * NKV * headDim], kvStrideRow elements per row
    int64_t kvStrideRow;
    void const* k;         // bf16 [n * kp1, L, NKV, headDim] contiguous
    void const* w;         // bf16 [L, headDim] contiguous (foldW)
    float const* cosSin;   // fp32 [maxPos, rot] contiguous (cos | sin)
    int64_t const* pos;    // [n * kp1]
    void const* nacc;      // [n]
    int64_t const* rows;   // [n] block-table row per request
    int64_t const* counts; // [table rows] pages per row
    int32_t const* table;  // [table rows, width] absolute page ids, tableStride elements per row
    int64_t tableStride;
    void* cache;           // stacked [L, pages, 2, NKV, storePage, headDim] bf16, or int64 [L] addresses (ptrList)
    int64_t cacheStrideLayer;
    int32_t kp1;
    int32_t numLayers;
    int32_t numKvHeads;
    int32_t headDim;   // 16 .. 256, a power of two
    int32_t rot;       // even, <= headDim
    int32_t storePage; // the cache's page size
    bool naccIs64;
    bool ptrList;
    bool foldW;
    // ---- the advance
    int64_t const* slots;     // [n]
    int64_t* ctxLen;          // [slots], updated in place
    int64_t const* basePages; // [slots]
    int64_t* out;             // [n] advertised context lengths
    int64_t maxCtx;
    int64_t dummySlot;
    int64_t page;
    int64_t blockSize;
    // ---- the read view
    int32_t* readView; // [table rows, width], readStride elements per row
    int64_t readStride;
    int64_t width;
    // ---- the forward indices (int32 outputs)
    void const* qpos; // [n, iblock], strided by q0 / q1 elements
    bool qposIs64;
    int64_t q0;
    int64_t q1;
    int32_t* seqlens;  // [n]
    int32_t* batchOut; // [n]
    int32_t* after;    // [n]
    int32_t* indptr;   // [n + 1]
    int32_t* last;     // [n]
    int32_t* append;   // [n * iblock]
    int32_t* qflat;    // [n * iblock]
    int32_t iblock;
    int32_t ipage;
    int32_t n;
};

//! Launches the fused glue (n >= 1). Returns the launch status.
cudaError_t invokeTailGlue(TailGlueParams const& params, bool enablePdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
