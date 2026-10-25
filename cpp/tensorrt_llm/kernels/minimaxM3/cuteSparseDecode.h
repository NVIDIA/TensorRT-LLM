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

//! Operands of the MiniMax-M3 TP2 block-sparse verify decode (16 q heads per KV head, head_dim 128, page 128,
//! top-k <= 32). Strides are in bytes for the one-byte tensors (q, K, V, block scales) and in elements otherwise.
struct MinimaxM3CuteSparseDecodeParams
{
    uint8_t const* q;   // e4m3 [total_q, H * 16, 128]
    uint8_t const* k;   // packed e2m1 [pages, H, 128, 64]
    uint8_t const* v;   // packed e2m1 [pages, H, 128, 64]
    uint8_t const* ksc; // e4m3 [pages, H, 128, 8], token-major
    uint8_t const* vsc; // e4m3 [pages, H, 128, 8], 4x4 token-quad order
    float const* kgs;
    float const* vgs;
    int32_t const* topk; // [H, total_q, max_topk]
    int32_t const* bt;   // [num_reqs, max_blocks]
    int32_t const* seq_lens;
    __nv_bfloat16* out;  // [total_q, H * 16, 128]
    int total_q;
    int num_kv_heads;
    int max_topk;
    int dql;
    float sm_scale;
    int pad0;
    long long sq_n, sq_h;
    long long sk_blk, sk_h, sv_blk, sv_h;
    long long sks_blk, sks_h, svs_blk, svs_h;
    long long st_h, st_n, st_k;
    long long so_n, so_h;
    long long sbt_b;
};

//! One CTA per (top-k split, KV head, query row), clusters of `splits` CTAs merging over DSMEM; `stages` smem ring
//! stages; `dequant` 0 = cvt.e2m1x2, 1 = integer bit placement. splits in {1, 2, 4, 8}, stages in {2, 3, 4}.
cudaError_t invokeMinimaxM3CuteSparseDecode(
    MinimaxM3CuteSparseDecodeParams const& p, int splits, int stages, int dequant, bool pdl, cudaStream_t stream);

//! Kernel attributes of one policy: registers, local bytes, dynamic smem, CTAs per SM, binary version.
cudaError_t minimaxM3CuteSparseDecodeAttrs(int splits, int stages, int dequant, int* out5);

} // namespace kernels

TRTLLM_NAMESPACE_END
