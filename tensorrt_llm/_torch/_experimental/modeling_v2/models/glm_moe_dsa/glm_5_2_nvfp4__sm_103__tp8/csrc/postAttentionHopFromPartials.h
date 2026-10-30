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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHop.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace attention_output_into_hop
{

//! Post-attention hop that takes its input from the transport: the hop of postAttentionHop.h with its input taken
//! from the transport instead of from a partial tensor.
//!
//! The producer of this call (the attention output tail's o_proj kernel, publishing into the hop) has already
//! written every rank's sanitized rank-local partial (bf16 -0.0 mapped to +0.0) into the current Lamport buffer of
//! the post-attention hop's workspace, slot ``[row][rank][kHiddenDim]``, with multicast stores. This kernel polls
//! all ``kNumRanks`` slots of its rows (its own included) and then performs exactly the post-attention hop's
//! arithmetic: the rank sum in rank order 0..7, the bf16 residual add, RMSNorm with the native reduction tree and
//! the optional router logits with the post-attention hop's MMA decomposition and summation order. Given the same
//! partials, ``normed``, ``residualOut`` and ``routerLogits`` are bit-identical to ``postAttnHop`` consuming the
//! ordinary partial (whose own contribution is the same sanitized value, taken from registers there).
//!
//! The workspace, buffer rotation and protocol state are the post-attention hop's
//! (``post_attention_hop``), so calls of either kind can follow each other on one workspace.
//! Memory-order contract with the producer (programmatic dependent launch):
//!   * the producer reads the protocol state and writes the transport only after its own ``griddepcontrol.wait`` and
//!     triggers its dependents only after that wait, so when this kernel starts, every grid before the producer has
//!     completed: the protocol state written by the previous hop and ``residualIn`` are final;
//!   * with ``earlyPoll`` this kernel reads the protocol state and polls the transport before its
//!     ``griddepcontrol.wait`` (the transport is private to the package and Lamport-validated); it reads
//!     ``residualIn`` and writes every output only after the wait, i.e. after the producer grid has completed.
//!     Without ``earlyPoll`` it waits first. The early read relies on every kernel between two hops waiting on its
//!     predecessor before it completes, which holds for the served graph's kernels (each consumes its predecessor's
//!     output).
struct PostAttentionHopFromPartialsParams
{
    void const* residualIn = nullptr;   //!< [numRows, kHiddenDim] bf16
    void const* gamma = nullptr;        //!< [kHiddenDim] bf16 RMSNorm weight
    void const* routerWeight = nullptr; //!< [kNumExperts, kHiddenDim] bf16 or nullptr (no router logits)
    void* normOut = nullptr;            //!< [numRows, kHiddenDim] bf16
    void* residualOut = nullptr;        //!< [numRows, kHiddenDim] bf16
    float* routerLogits = nullptr;      //!< [numRows, kNumExperts] fp32, required iff routerWeight != nullptr
    void* ucBuffer = nullptr;           //!< local unicast view of the post-attention hop's multicast workspace
    uint32_t* flags = nullptr;          //!< [4] protocol state of that workspace
    int numRows = 0;
    int rank = 0;
    float eps = 1e-5F;
    bool enablePdl = true;
    //! true: read the protocol state and poll the transport before ``griddepcontrol.wait`` (relies on the producer
    //! contract above); false: wait first, as postAttentionHop does, then read the state and poll.
    bool earlyPoll = true;
    //! Where the router-weight MMA fragments are loaded (same values and arithmetic in every case): 0 after the
    //! barrier that ends the rank sum, 1 right after the rank sum (default), 2 right before the MMAs.
    int routerLoad = 1;
    cudaStream_t stream = nullptr;
};

using post_attention_hop::kHiddenDim;
using post_attention_hop::kLamportBufferBytes;
using post_attention_hop::kMaxRows;
using post_attention_hop::kNumExperts;
using post_attention_hop::kNumLamportBuffers;
using post_attention_hop::kNumRanks;
using post_attention_hop::kWorkspaceBytes;

//! Launch the transport-input hop on ``params.stream``. Returns the launch status.
cudaError_t invokePostAttentionHopFromPartials(PostAttentionHopFromPartialsParams const& params);

} // namespace attention_output_into_hop

TRTLLM_MV2_NAMESPACE_END
