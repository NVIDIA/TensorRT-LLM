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

//! DSPARK_CTX_STORE_CPP (replaces the Triton _ctx_store_kernel of ops/dspark/dspark_ctx_store.py, bit
//! for bit): the DSpark / DFlash drafter's context K / V store after the fused K / V GEMM, every layer in one launch.
//! Row r (request r / kp1, block offset r % kp1) of layer l, kv head h:
//!   k = kIn[r, l, h] (FOLD_W: bf16_rn(f32(k) * f32(w[l]))), NeoX RoPE over the first rot dims from cosSin[pos[r]]
//!   (fma.rn(a, c, (-b) * s) / fma.rn(b, c, a * s), rounded to bf16), dims rot..headDim unrotated;
//!   k, v (= kv[r, (l * 2 + 1) * NKV + h]) *= m, m = 1 if r % kp1 < numAccepted[r / kp1] else 0 (bf16);
//!   stored at cache[l][table[row, col / pageSize], K | V, h, col % pageSize] (HND), row = requestRows[r / kp1],
//!   col = cols[r], or from pos[r] (colFromPos) clamped to counts[row] * pageSize - 1 and 0 (bound) or to maxCol.
//!   Of the rows of one request that share a column only the last writes.
//! Index tensors are int32 or int64 (idxBits: bit 0 pos, 1 numAccepted, 2 requestRows, 3 cols, 4 counts, 5 table set
//! = int64). The cache is a stacked [L, pages, 2, NKV, pageSize, headDim] bf16 tensor (cacheStrideLayer elements per
//! layer) or, with ptrList, an int64 array of L per-layer cache addresses.
struct DsparkCtxStoreParams
{
    void const* kv;          // bf16 [n, L * 2 * NKV * headDim], kvStrideRow elements per row
    int64_t kvStrideRow;
    void const* k;           // bf16 [n, L, NKV, headDim] contiguous
    void const* w;           // bf16 [L, headDim] contiguous (foldW)
    float const* cosSin;     // fp32 [maxPos, rot] contiguous (cos | sin)
    void const* pos;         // [n]
    void const* numAccepted; // [n / kp1]
    void const* requestRows; // [n / kp1]
    void const* cols;        // [n] (unless colFromPos)
    void const* counts;      // [table rows] (bound)
    int64_t maxCol;
    void const* table;       // [table rows, tableWidth] page ids, unit column stride
    int64_t tableStride;
    int64_t tableWidth;
    void* cache;
    int64_t cacheStrideLayer;
    int32_t n;
    int32_t kp1;
    int32_t numLayers;
    int32_t numKvHeads;
    int32_t headDim; // 16 .. 256, a power of two
    int32_t rot;     // even, <= headDim
    int32_t pageSize;
    int32_t idxBits;
    bool ptrList;
    bool colFromPos;
    bool bound; // colFromPos on a bound pool (else the private arena's maxCol)
    bool foldW;
};

//! One warp per (row, layer, kv head), PDL-aware (griddepcontrol.wait at entry, dependents released right after it).
void invokeDsparkCtxStore(DsparkCtxStoreParams const& params, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
