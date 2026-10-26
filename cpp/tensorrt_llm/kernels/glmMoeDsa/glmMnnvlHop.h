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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! Routed experts per token of the MoE-finalize hop.
constexpr int kMnnvlHopMoeTopK = 8;

//! \brief One BF16 one-shot MNNVL hop: the all-reduce of [numTokens, tokenDim] rows over nRanks ranks, plus the
//! residual and RMSNorm, on an MNNVL all-reduce workspace (three Lamport buffers and their flags, -0.0 as the empty
//! value; the workspace of the one-shot MNNVL all-reduce).
struct MnnvlHopParams
{
    int nRanks;             //!< Ranks of the group: 2, 4, 8 or 16
    int rank;               //!< This rank
    int numTokens;          //!< Rows
    int tokenDim;           //!< Elements per row
    void** bufferPtrsDev;   //!< Unicast addresses of every rank's workspace (a device array)
    void* multicastPtr;     //!< Multicast address of the workspace
    uint32_t* bufferFlags;  //!< Lamport flags of the workspace
    void const* input;      //!< This rank's partial rows [numTokens, tokenDim]; not read when shardPublished
    void const* residualIn; //!< [numTokens, tokenDim]
    void const* gamma;      //!< RMSNorm weight [tokenDim]
    float epsilon;          //!< RMSNorm epsilon
    void* residualOut;      //!< Rows summed over the ranks plus residualIn [numTokens, tokenDim]
    void* normOut;          //!< RMSNorm of residualOut times gamma [numTokens, tokenDim]
    bool shardPublished;    //!< The producer of input already stored this rank's rows into the workspace
    cudaStream_t stream;
};

//! \brief The routed experts' rows of the MoE-finalize hop: the do_finalize=false outputs of the MoE and the
//! shared-expert output added to them.
struct MnnvlMoeFinalizeParams
{
    void const* fc2Output;                   //!< FC2 rows [*, hiddenDimPadded], BF16
    int hiddenDimPadded;                     //!< Row pitch of fc2Output in elements
    void const* expertWeights;               //!< Routing weights [numTokens, kMnnvlHopMoeTopK], BF16
    int32_t const* expandedIdxToPermutedIdx; //!< FC2 row of each (token, k) [numTokens, kMnnvlHopMoeTopK]; -1 skips
    void const* addend;                      //!< Added to the finalized rows [numTokens, tokenDim], BF16
};

//! Whether the hops serve a BF16 [numTokens, tokenDim] all-reduce over nRanks ranks: 2, 4, 8 or 16 ranks, one-shot
//! (up to 1 MiB of Lamport data per rank, and up to 1.5 MiB on 16 ranks), and a grid of one row per cluster of 2 to 8
//! CTAs of at most 128 threads (whole warps), eight elements per thread, that covers the row exactly.
bool mnnvlHopSupported(int numTokens, int tokenDim, int nRanks);

//! Bytes of one Lamport buffer that a hop of numTokens x tokenDim BF16 elements over nRanks ranks uses.
int64_t mnnvlHopBufferBytes(int numTokens, int tokenDim, int nRanks);

//! \brief The one-shot all-reduce with the residual add and RMSNorm (FastEpilogue, FastProtocol, one poll loop).
//!
//! Every rank stores its rows (-0.0 written as +0.0) through the multicast address at [token][rank][column] of the
//! current Lamport buffer, empties the dirty buffer, and polls all slots in one loop. The sum is in rank order in FP32,
//! from zero per chunk of 8 ranks, the chunks added in order, and rounded to BF16; the residual is added in BF16; the
//! sum of squares of the BF16 values goes through a transaction barrier of the cluster. The PDL dependents launch
//! right after the grid dependency wait, so they must read the outputs only after their own wait.
//!
//! With shardPublished the producer of input (the o_proj GEMM's Lamport publish) has stored this rank's rows into
//! every rank's current buffer: the kernel polls every slot, its own included, does not wait for its primary before
//! the exchange and reads the flags and the residual with volatile loads; it counts its CTA arrival after its poll
//! and waits for its primary at its end. The producer must trigger its dependents only after its own grid dependency
//! wait.
void mnnvlHop(MnnvlHopParams const& params);

//! \brief The MoE finalize fused into the hop: each thread finalizes its eight elements of every routed row (the
//! token's FC2 rows weighted in k order with an FMA, -1 slots skipped, rounded to BF16, the addend added in FP32 and
//! the sum rounded once more) and publishes them as its rows; the rest is mnnvlHop without a published shard.
//! params.input and params.shardPublished are not read. Eight threads per CTA read the token's FC2 row indices and
//! routing weights before the grid dependency wait: the kernel that wrote them must have completed when this grid
//! launches (the MoE's FC2 triggers its dependents only after its own wait).
void mnnvlMoeFinalizeHop(MnnvlHopParams const& params, MnnvlMoeFinalizeParams const& moe);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
