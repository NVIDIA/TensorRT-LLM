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

//! Operands of the MiniMax-M3 multicast all-reduce + residual + RMSNorm kernel (decode, at most 48 tokens).
//!
//! Per rank the kernel forms the partial `bf16(x + x2)` (or, in finalize mode,
//! `bf16(bf16(sum_k scale * fc2[perm]) + x2)`, the routed sum rounded before the shared expert is added, as stock),
//! pushes it through the multicast mapping `wsMc` into every rank's exchange, polls its own unicast view `wsUc`,
//! sums the ranks in FP32 in rank order, adds the residual (BF16 round) and applies RMSNorm. With `outQuant` it
//! also writes the MXFP8 copy of the normed rows (E4M3 [numTokens, hidden] and swizzled UE8M0 scales).
struct MinimaxM3McAllreduceNormParams
{
    __nv_bfloat16 const* x{nullptr};   //!< [numTokens, hidden] partial; in finalize mode the FC2 rows [rows, hidden].
    __nv_bfloat16 const* x2{nullptr};  //!< Optional second addend [numTokens, hidden].
    __nv_bfloat16 const* fc2{nullptr}; //!< Finalize mode: the permuted FC2 output (same buffer as x); null otherwise.
    int32_t const* perm{nullptr};      //!< Finalize mode: [numTokens, topk] expanded index -> permuted row, -1 dropped.
    void const* scales{nullptr};       //!< Finalize mode: [numTokens, topk] routing weights (fp32 or bf16).
    bool scaleFp32{false};
    int32_t topk{0};
    int32_t numTokens{0};
    int32_t hidden{0};
    __nv_bfloat16 const* residual{nullptr};   //!< [numTokens, hidden]
    __nv_bfloat16 const* normWeight{nullptr}; //!< [hidden]
    float eps{0.F};
    int32_t* wsUc{nullptr};  //!< This rank's unicast view of the exchange (int32 words, re-armed after reading).
    int32_t* wsMc{nullptr};  //!< The multicast mapping of the exchange.
    int32_t* flags{nullptr}; //!< Per-CTA call-parity counters, [maxTokens * 8].
    int32_t rank{0};
    int32_t world{0};
    __nv_bfloat16* outNormed{nullptr};
    __nv_bfloat16* outResidual{nullptr};
    uint8_t* outQuant{nullptr}; //!< Optional E4M3 [numTokens, hidden].
    uint8_t* outSf{nullptr};    //!< Optional UE8M0 scales, 128x4-swizzled.
    //! 0: push x (+ x2) here; 2: a producer already pushed x and launched this grid after its pushes; 3: as 2, with x
    //! in exchange set A and x2 in set B.
    int32_t prepushed{0};
};

//! The (bucket, epilogue options) pairs this launcher is built for: (8, 7), (8, 15), (16, 15) and (48, 15).
bool isMinimaxM3McAllreduceNormSupported(int32_t maxTokens, int32_t opts);

//! Launch with 8-CTA clusters of 128 threads, one cluster per token. `maxTokens` selects the exchange bucket;
//! `opts` the epilogue variant (every variant is bit-identical).
void invokeMinimaxM3McAllreduceNorm(
    MinimaxM3McAllreduceNormParams const& params, int32_t maxTokens, int32_t opts, bool pdl, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
