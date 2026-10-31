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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{
namespace drafterRopeFp8Kv
{

constexpr int kHeadDim = 128;

struct Params
{
    //! [tokens, (numHeadsQ + 2 * numHeadsKv) * 128] BF16: the drafter layer's packed Q, K, V rows.
    __nv_bfloat16 const* qkv;
    //! [positions, rotaryDim / 2] (cos, sin) pairs: the layer's rotary_cos_sin table.
    float2 const* cosSin;
    //! [requests] cache lengths including this step's rows.
    int const* seqLens;
    //! [requests, qLen] spec-decoding position offsets of a multi-token pass, or null.
    int const* specPositionOffsets;
    //! [tokens, numHeadsQ, 128] E4M3 query out.
    __nv_fp8_e4m3* qOut;
    //! Base of the layer's K / V sub-pages, [*, numHeadsKv, tokensPerBlock, 128] E4M3.
    __nv_fp8_e4m3* pool;
    //! [requests, 2, maxBlocks] sub-page indices: K in row 0, V in row 1.
    int const* blockTable;
    long long blockTableRequestStride;
    long long blockTableKvStride;
    int numTokens;
    int qLen;
    int numHeadsQ;
    int numHeadsKv;
    int rotaryDim;
    int tokensPerBlock;
    //! How the rotation's two products are summed: 0 two products then the add, 1 fma(cos, x, sin * pair),
    //! 2 fma(sin, pair, cos * x).
    int ropeMode;
};

//! The MiniMax-M3 TP8 Eagle3 drafter layer's generation preprocess: GPT-NeoX RoPE from the cos/sin table on Q and K,
//! rounded to BF16 (the in-tree preprocess's q output and cache input), then Q, K and V to saturating E4M3: K / V
//! as the in-tree preprocess stores them at unit KV scale, Q as torch's CUDA cast of the BF16 query. Rows map to
//! (request token / qLen, row token % qLen) at cache position seqLen - qLen + row.
//!
//! PDL (pdl): the launch is programmatic; griddepcontrol.wait is the kernel's first instruction and the kernel
//! releases its dependent right after it. The dependent may read only step-invariant data before its own wait.
cudaError_t launch(Params const& params, bool pdl, bool maxSharedCarveout, cudaStream_t stream);

} // namespace drafterRopeFp8Kv
} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
