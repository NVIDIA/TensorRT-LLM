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
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace post_attention_hop
{

//! Fused post-attention hop.
//!
//! For ``numRows`` rows of hidden size ``kHiddenDim`` on a TP group of ``kNumRanks`` ranks it computes, on every
//! rank and with identical bits on every rank:
//!   sum      = bf16(sum over ranks 0..7 of float(partial_rank))          (one-shot Lamport exchange, fixed order)
//!   residual = sum + residualIn                                         (bf16 add)
//!   normed   = bf16(float(residual) * rsqrt(mean(bf16(residual^2)) + eps) * float(gamma))
//!   logits   = normed @ routerWeight^T in FP32                          (optional, [numRows, kNumExperts])
//! ``sum``, ``residual`` and ``normed`` follow the arithmetic and reduction tree of the native MNNVL one-shot
//! AR + residual + RMSNorm kernel, so they are bit-identical to it.
//!
//! The exchange uses a private multicast workspace of ``kNumLamportBuffers`` rotating buffers. ``flags`` holds the
//! device-side protocol state ``{current buffer, bytes to clear in the previous buffer, arrival counter, unused}``
//! and must be zero-initialized together with the buffers filled with the Lamport sentinel (all 32-bit words equal
//! to 0x80000000) on every rank before the first call.
struct PostAttnHopParams
{
    void const* input = nullptr;        //!< [numRows, kHiddenDim] bf16, rank-local partial (o_proj output)
    void const* residualIn = nullptr;   //!< [numRows, kHiddenDim] bf16
    void const* gamma = nullptr;        //!< [kHiddenDim] bf16 RMSNorm weight
    void const* routerWeight = nullptr; //!< [kNumExperts, kHiddenDim] bf16 or nullptr (no router logits)
    void* normOut = nullptr;            //!< [numRows, kHiddenDim] bf16
    void* residualOut = nullptr;        //!< [numRows, kHiddenDim] bf16
    float* routerLogits = nullptr;      //!< [numRows, kNumExperts] fp32, required iff routerWeight != nullptr
    void* ucBuffer = nullptr;           //!< local unicast view of the multicast workspace
    void* mcBuffer = nullptr;           //!< multicast address of the same workspace
    uint32_t* flags = nullptr;          //!< [4] device-side protocol state of this workspace
    int numRows = 0;
    int rank = 0;
    float eps = 1e-5F;
    bool enablePdl = true;
    cudaStream_t stream = nullptr;
};

constexpr int kHiddenDim = 6144;
constexpr int kNumRanks = 8;
constexpr int kNumExperts = 256;
constexpr int kMaxRows = 8;
constexpr int kNumLamportBuffers = 3;

//! Bytes of one Lamport buffer: [kMaxRows][kNumRanks][kHiddenDim] bf16.
constexpr size_t kLamportBufferBytes = size_t(kMaxRows) * kNumRanks * kHiddenDim * 2;
//! Bytes of the whole multicast workspace.
constexpr size_t kWorkspaceBytes = kLamportBufferBytes * kNumLamportBuffers;

//! Launch the fused hop on ``params.stream``. Returns the launch status.
cudaError_t invokePostAttentionHop(PostAttnHopParams const& params);

#ifdef TLLM_GLM_LL_W2_TIMING
//! Development builds only: phase timestamps of the most recent launch, [kTimingMaxCtas][kTimingStamps] ns.
constexpr int kTimingMaxCtas = 128;
constexpr int kTimingStamps = 10;
cudaError_t copyPostAttnHopTiming(unsigned long long* dst, cudaStream_t stream);
#endif

} // namespace post_attention_hop

TRTLLM_MV2_NAMESPACE_END
