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

#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp2
{
namespace denseQkNormRopeFp8
{

constexpr int kHeadDim = 128;

//! A layer's FP8 K/V pool in the HND paged layout [pages, 2 (K, V), kv heads, tokens per block, 128] (element
//! strides), with the step's per-token cache slots (page * tokensPerBlock + within) for its first numLiveTokens rows.
struct KvCacheView
{
    __nv_fp8_e4m3* base = nullptr;
    int const* outCacheLoc = nullptr;
    long long pageStride = 0;
    long long kvStride = 0;
    long long headStride = 0;
    long long tokenStride = 0;
    int tokensPerBlock = 1;
    int numLiveTokens = 0;
    //! Optional: every row's Q also stored compact, [tokens, q heads, 128].
    __nv_fp8_e4m3* qCompact = nullptr;
};

//! The MiniMax-M3 TP2 dense layers' per-head QK RMSNorm + partial RoPE of a packed BF16 [tokens, (q + k + v) * 128]
//! row, written out of place as FP8 E4M3 with V copy-cast. Q and K are rounded to BF16 before E4M3, as the stock path
//! (the BF16 fused_qk_norm_rope output cast to FP8) rounds them. With cache, the K and V rows of the live tokens are
//! also stored into the paged pool at their slots (the bytes of the qkvOut rows).
void launch(void const* qkvIn, void* qkvOut, int numTokens, int numHeadsQ, int numHeadsK, int numHeadsV, int rotaryDim,
    float eps, void const* qWeight, void const* kWeight, float base, bool interleave, int const* positionIds,
    bool useGemma, KvCacheView const* cache, cudaStream_t stream);

} // namespace denseQkNormRopeFp8
} // namespace minimaxM3Sm107Tp2
} // namespace kernels

TRTLLM_NAMESPACE_END
