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

namespace draft_live_rows
{

//! Live-row post-attention hop (MTP draft step 0).
//!
//! A fork of the router hop of postAttentionHop.h that serves only the ``numRows`` live rows of a call with
//! ``numInputRows`` rows. Compact row ``r`` (0 <= r < numRows) is source row ``rowIds[r]`` of ``input`` and
//! ``residualIn``; for it the kernel computes, on every rank and with identical bits on every rank:
//!   sum      = bf16(sum over ranks 0..7 of float(partial_rank))          (one-shot Lamport exchange, fixed order)
//!   residual = sum + residualIn                                         (bf16 add)
//!   normed   = bf16(float(residual) * rsqrt(mean(bf16(residual^2)) + eps) * float(gamma))
//!   logits   = normed @ routerWeight^T in FP32                          ([numRows, kNumExperts])
//! Every quantity of a row is computed by the same per-row arithmetic as the post-attention hop, so compact row ``r``
//! equals row ``rowIds[r]`` of a post-attention hop over all ``numInputRows`` rows.
//!
//! ``rowIds`` is read on the device after the kernel's programmatic dependency wait, never before it; it may be
//! written by any earlier kernel of the stream. Ids outside ``[0, numInputRows)`` are clamped into that range.
//!
//! The exchange runs on the post-attention hop's multicast workspace and protocol state, with the same layout
//! (``[row][source rank][kHiddenDim]`` bf16 per Lamport buffer, ``kNumLamportBuffers`` rotating buffers, flags
//! ``{current buffer, bytes to clear in the previous buffer, arrival counter, unused}``), so calls of both kernels
//! may be issued in any order on the stream that issues the group's hops.
struct LiveRowHopParams
{
    void const* input = nullptr;        //!< [numInputRows, kHiddenDim] bf16, rank-local partial (o_proj output)
    void const* residualIn = nullptr;   //!< [numInputRows, kHiddenDim] bf16
    int64_t const* rowIds = nullptr;    //!< [numRows] device ids of the live rows in [0, numInputRows)
    void const* gamma = nullptr;        //!< [kHiddenDim] bf16 RMSNorm weight
    void const* routerWeight = nullptr; //!< [kNumExperts, kHiddenDim] bf16
    void* normOut = nullptr;            //!< [numRows, kHiddenDim] bf16
    void* residualOut = nullptr;        //!< [numRows, kHiddenDim] bf16
    float* routerLogits = nullptr;      //!< [numRows, kNumExperts] fp32
    void* ucBuffer = nullptr;           //!< local unicast view of the multicast workspace
    void* mcBuffer = nullptr;           //!< multicast address of the same workspace
    uint32_t* flags = nullptr;          //!< [4] device-side protocol state of this workspace
    int numRows = 0;                    //!< live rows, 1 <= numRows <= numInputRows
    int numInputRows = 0;               //!< rows of ``input`` / ``residualIn``, <= kMaxRows
    int rank = 0;
    float eps = 1e-5F;
    bool enablePdl = true;
    cudaStream_t stream = nullptr;
};

// The post-attention hop's workspace geometry (the kernel runs on that workspace).
constexpr int kHiddenDim = 6144;
constexpr int kNumRanks = 8;
constexpr int kNumExperts = 256;
constexpr int kMaxRows = 8;
constexpr int kNumLamportBuffers = 3;

//! Bytes of one Lamport buffer: [kMaxRows][kNumRanks][kHiddenDim] bf16.
constexpr size_t kLamportBufferBytes = size_t(kMaxRows) * kNumRanks * kHiddenDim * 2;
//! Bytes of the whole multicast workspace.
constexpr size_t kWorkspaceBytes = kLamportBufferBytes * kNumLamportBuffers;

//! Launch the live-row hop on ``params.stream``. Returns the launch status.
cudaError_t invokeDraftLiveRowHop(LiveRowHopParams const& params);

} // namespace draft_live_rows

TRTLLM_MV2_NAMESPACE_END
