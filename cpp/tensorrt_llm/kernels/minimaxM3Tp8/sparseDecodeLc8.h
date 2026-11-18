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

//! Operands of the MiniMax-M3 TP8 one-launch long-context sparse decode. Strides are in elements of the pointed-to
//! type. Geometry is fixed: 8 query heads on one KV head, head_dim 128, page 128, top-k 16.
struct MinimaxM3Tp8SparseDecodeLc8Params
{
    uint8_t const* q; // e4m3 [T, 8, 128]
    int64_t q_tok, q_head;
    uint8_t const* k; // packed e2m1 [pages, 1, 128, 64]
    uint8_t const* v;
    int64_t k_blk, v_blk;
    uint8_t const* ksc; // e4m3 [pages, 1, 128, 8], token-major
    uint8_t const* vsc; // e4m3, 4x4 token-quad order
    int64_t ksc_blk, vsc_blk;
    float const* kgs;
    float const* vgs;
    int const* topk; // [T, 16] view (kv head 0)
    int64_t t_tok, t_k;
    int const* block_table;
    int64_t bt_stride;
    int const* seq_lens;
    int dql;
    float sm_scale;
    __nv_bfloat16* out; // [T, 1024]
    int64_t out_stride;
    uint8_t* q8;        // e4m3 [rows, num_sf_k_tiles * 128]
    uint8_t* sf;
    int num_sf_k_tiles;
};

//! One cluster of 16 CTAs per query token: CTA r attends the token's r-th selected block with transposed MMAs (the 8
//! heads fill the MMA n dimension), then CTA r merges (head r / 2, dims (r % 2) * 64 + [0, 64)) of the 16 partials
//! over DSMEM into the BF16 [T, 1024] row (and, when mxfp8, its MXFP8 copy with swizzled UE8M0 scales). Dependents
//! are released right after the grid dependency wait.
//!
//! With prewait, the request's seq_lens entry, its q rows (widened to FP16) and its block-table row (up to 8192
//! entries, in dynamic shared memory; longer rows are read from global memory) are staged before the grid dependency
//! wait; outputs are bit-identical to prewait=false.
//!
//! Instantiated variants: linearSoftmax x mxfp8 x prewait, all in {false, true}. The device code is SM100-family
//! only; on other architectures the kernel compiles to an empty body, so callers check the device first.
cudaError_t invokeMinimaxM3Tp8SparseDecodeLc8(MinimaxM3Tp8SparseDecodeLc8Params const& params, int32_t numTokens,
    bool linearSoftmax, bool mxfp8, bool prewait, bool usePdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
