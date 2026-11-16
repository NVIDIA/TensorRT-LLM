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

namespace hop_routing
{

//! Pre-MoE hop with routing: the transport-input hop (postAttentionHopFromPartials.h) with router logits, plus
//! the native DeepSeek-V3 top-8 routing of every row computed from those logits.
//!
//! Normalized hidden, residual and router logits are produced exactly as by
//! ``attention_output_into_hop::invokePostAttentionHopFromPartials`` (same work decomposition, arithmetic
//! and protocol on the same post-attention hop workspace), so they are bit-identical to that hop's.
//!
//! Routing (the native ``SigmoidBias`` + ``ScaledSumNormalize`` routing of the TRT-LLM-Gen routing runner with one
//! expert group): every logit owner computes ``score = 0.5f * tanhf(0.5f * logit) + 0.5f + float(bias[e])`` and stores
//! its bits into the gather buffer; one finisher CTA per row collects the row's 256 scores and selects the 8 largest
//! native keys ``(TwiddleIn(score) << 32) | (65535 - e)`` (ties go to the lower expert) in descending order, and writes
//! ``topkIds[r][k]`` and ``topkWeights[r][k] = bf16((score_k - bias_k) * routedScaling / sum)`` (stored as FP32), where
//! ``sum`` adds the eight ``score_k - bias_k`` in the native warp-butterfly order ((s0+s4)+(s2+s6))+((s1+s5)+(s3+s7)).
//!
//! Gather protocol (private to this kernel): ``gather`` holds two buffers of ``kMaxRows x kNumExperts`` 32-bit words
//! and ``zstate[0]`` the parity ``p`` of the buffer this call uses. A score is never -0.0f, so the word 0x80000000
//! marks an empty slot. Each call resets buffer ``p ^ 1`` (the previous call's) to that sentinel and CTA 0 flips the
//! parity together with the hop's state advance, after every CTA has read both states. Calls of other hop kinds on the
//! same hop workspace do not touch the routing state.
//!
//! Memory-order contract with the producer (programmatic dependent launch): as
//! ``attention_output_into_hop``. Before ``griddepcontrol.wait`` the kernel reads only the RMSNorm
//! weight, the router weight (L2 prefetch) and the router bias, which no kernel writes while serving; with
//! ``earlyPoll`` it also reads the protocol state of the post-attention hop workspace and polls the transport early
//! (see postAttentionHopFromPartials.h). The routing state, the gather buffers and every output are read or written
//! only after the wait. Finishers spin on gather words that every CTA writes before it waits on anything outside its
//! own cluster, so the wait-for graph is acyclic.
struct PostAttentionHopRoutingParams
{
    void const* residualIn = nullptr;   //!< [numRows, kHiddenDim] bf16
    void const* gamma = nullptr;        //!< [kHiddenDim] bf16 RMSNorm weight
    void const* routerWeight = nullptr; //!< [kNumExperts, kHiddenDim] bf16
    void const* routingBias = nullptr;  //!< [kNumExperts] bf16 score-correction bias
    //! Test only: when set, the routing scores are computed from these [numRows, kNumExperts] FP32 logits instead of
    //! the computed router logits (which are still written to ``routerLogits``). Read after the dependency wait.
    float const* logitsIn = nullptr;
    void* normOut = nullptr;       //!< [numRows, kHiddenDim] bf16
    void* residualOut = nullptr;   //!< [numRows, kHiddenDim] bf16
    float* routerLogits = nullptr; //!< [numRows, kNumExperts] fp32
    int32_t* topkIds = nullptr;    //!< [numRows, kTopK] int32, descending native key order
    float* topkWeights = nullptr;  //!< [numRows, kTopK] fp32 holding bf16-rounded weights
    void* ucBuffer = nullptr;      //!< local unicast view of the post-attention hop's multicast workspace
    uint32_t* flags = nullptr;     //!< [4] protocol state of that workspace
    uint32_t* gather = nullptr;    //!< [kGatherWords] routing gather buffers
    uint32_t* zstate = nullptr;    //!< [4] routing state; word 0 is the parity
    //! Diagnostic builds only (``TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS``): [gridCtas, kNumStamps] globaltimer values.
    unsigned long long* stamps = nullptr;
    int numRows = 0;
    int rank = 0;
    float eps = 1e-5F;
    float routedScaling = 2.5F;
    bool enablePdl = true;
    bool earlyPoll = false;
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

constexpr int kTopK = 8;
//! Words of one gather buffer and of both.
constexpr int kGatherBufferWords = kMaxRows * kNumExperts;
constexpr int kGatherWords = 2 * kGatherBufferWords;
constexpr int kStateWords = 4;
//! Diagnostic stamp fields per CTA: entry, wait, last logit store, last gather store, poll complete, level-1 done,
//! routing stored, end.
constexpr int kNumStamps = 8;
//! Upper bound of the grid size (4 expert groups x 3 row groups x 8 CTAs).
constexpr int kMaxGridCtas = 96;

//! Whether this build records diagnostic stamps.
bool stampsCompiled();

//! Launch the routing hop on ``params.stream``. Returns the launch status.
cudaError_t invokePostAttentionHopRouting(PostAttentionHopRoutingParams const& params);

} // namespace hop_routing

TRTLLM_MV2_NAMESPACE_END
