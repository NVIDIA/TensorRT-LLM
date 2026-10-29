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
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! Rows (requests) one launch serves.
constexpr int kDraftSampleGlueMaxRows = 8;
//! Hidden size of the eh_proj input glue (FlashInfer RMSNormKernel geometry: 64 threads x 8-element vectors x 12).
constexpr int kDraftSampleGlueHidden = 6144;
//! Bytes of one exchanged (index, value) packet.
constexpr int kDraftSampleGluePacketBytes = 16;

//! How the next step's KV lengths follow from the current ones.
enum class DraftSampleGlueKvUpdate : int
{
    kNone = 0,      //!< unchanged
    kIncrement = 1, //!< kvLens[r] += 1
    kRewind = 2,    //!< kvLens[r] -= draftLen - kvRewindAccepted[r]
};

//! One MTP draft step after the vocabulary-sharded draft LM head, in one CTA:
//! - the greedy draft token: the local argmax of this rank's logits shard, exchanged with the other ranks through a
//!   dedicated MNNVL Lamport workspace (layout and flags of the MNNVL all-gather), and the global argmax of the pairs;
//! - the state of the next draft step: position ids, sequence and KV lengths, the DSA decode metadata of
//!   fused_dsa_decode_metadata;
//! - the next step's eh_proj input: this rank's columns of cat([enorm(embed(token)), hnorm(hidden)], -1);
//! - after the last draft step, the next-new-tokens rows of the output.
//! Every part except the token is optional (skipped when its output pointer is null).
struct DraftSampleGlueParams
{
    // Draft logits of this rank, [numRows, vocabShard] BF16 with row stride logitsStride (elements).
    __nv_bfloat16 const* logits{nullptr};
    int64_t logitsStride{0};
    int numRows{0};
    int vocabShard{0};
    int rank{0};
    int tpSize{1};

    // Lamport workspace: three buffers of kDraftSampleGlueMaxRows * tpSize packets, the local unicast view, the
    // multicast view and the nine 32-bit flags of the MNNVL all-gather.
    void* exchangeLocal{nullptr};
    void* exchangeMcast{nullptr};
    uint32_t* exchangeFlags{nullptr};

    // Draft tokens [numRows, numDraftTokens] int32; this step writes column draftStep.
    int32_t* draftTokens{nullptr};
    int numDraftTokens{1};
    int draftStep{0};

    // Next step's eh_proj input (skipped when glueOut is null): columns [colLo, colHi) of
    // cat([enorm(embedTable[token]), hnorm(hidden)], -1), stored at glueOut[row * glueOutStride + col - colLo].
    __nv_bfloat16 const* embedTable{nullptr};
    int64_t vocabSize{0};
    __nv_bfloat16 const* hidden{nullptr};
    int64_t hiddenStride{0};
    __nv_bfloat16 const* enormWeight{nullptr};
    __nv_bfloat16 const* hnormWeight{nullptr};
    float eps{0.F};
    int colLo{0};
    int colHi{0};
    __nv_bfloat16* glueOut{nullptr};
    int64_t glueOutStride{0};

    // Next step's position ids (skipped when positionsOut is null):
    // positionsOut[r] = positionsIn[gatherIds ? gatherIds[r] : r] + 1, int32 or int64.
    void const* positionsIn{nullptr};
    void* positionsOut{nullptr};
    bool positions64{false};
    int64_t const* gatherIds{nullptr};

    // Sequence and KV lengths of the next step: seqLens[r] = 1 when fillSeqLens; kvLens[r] updated as kvUpdate says.
    int32_t* seqLens{nullptr};
    bool fillSeqLens{false};
    int32_t* kvLens{nullptr};
    DraftSampleGlueKvUpdate kvUpdate{DraftSampleGlueKvUpdate::kNone};
    int32_t const* kvRewindAccepted{nullptr};
    int draftLen{0};

    // DSA decode metadata of the next step (skipped when reqIdxPerToken is null), from the sequence and KV lengths
    // above: what fused_dsa_decode_metadata writes for numRows requests and numTokens tokens. maxQueryLanes is its
    // per-request token lanes (the power of two at or above the maximum query length).
    int32_t const* blockOffsets{nullptr};
    int64_t blockOffsetsStride0{0};
    int64_t blockOffsetsStride1{1};
    int maxBlocks{0};
    int numTokens{0};
    int maxQueryLanes{1};
    int tokensPerBlock{0};
    int64_t dataBytesPerToken{0};
    int64_t scaleSize{0};
    int64_t blockStride{0};
    int64_t scaleBaseOffset{0};
    int32_t* reqIdxPerToken{nullptr};
    int64_t* slotMappingFp8{nullptr};
    int64_t* slotMappingScale{nullptr};
    int64_t* genKvIndptr{nullptr};
    int64_t* genCachedTokenIndptr{nullptr};

    // Output rows after the last draft step (skipped when nextNewTokens is null), [numRows, numDraftTokens + 1]
    // int32: nextNewTokens[r, 0] = acceptedTokens[batchIndices[r], numAccepted[r] - 1], then the draft tokens.
    int32_t const* acceptedTokens{nullptr};
    int64_t acceptedTokensStride{0};
    int32_t const* batchIndices{nullptr};
    int32_t const* numAccepted{nullptr};
    int32_t* nextNewTokens{nullptr};
};

void invokeDraftSampleGlue(DraftSampleGlueParams const& params, cudaStream_t stream);

//! CTAs of the cluster that runs invokeDraftEhprojTable; each owns kDraftSampleGlueHidden / 8 hidden columns.
constexpr int kDraftEhprojTableClusterSize = 8;
//! Largest TP size of invokeDraftEhprojTable (2, 4, 8 or 16 ranks: each rank's share of the hidden-half outputs is
//! whole 8-column vectors).
constexpr int kDraftEhprojTableMaxRanks = 16;

//! Whether invokeDraftEhprojTable serves tpSize ranks.
constexpr bool draftEhprojTableSupportsTpSize(int tpSize)
{
    return tpSize >= 2 && tpSize <= kDraftEhprojTableMaxRanks && kDraftSampleGlueHidden % (8 * tpSize) == 0;
}

//! Bytes one rank publishes per row in invokeDraftEhprojTable: its (index, value) packet, the BF16 token-table row of
//! its candidate token and its kDraftSampleGlueHidden / tpSize FP32 hidden-half eh_proj outputs.
constexpr int draftEhprojTableSlotBytes(int tpSize)
{
    return kDraftSampleGluePacketBytes + 2 * kDraftSampleGlueHidden + 4 * kDraftSampleGlueHidden / tpSize;
}

//! One MTP draft step after the vocabulary-sharded draft LM head that also replaces the next step's eh_proj and its
//! all-reduce + input_layernorm, in one cluster of kDraftEhprojTableClusterSize CTAs:
//! - the greedy draft token and the next step's state, as invokeDraftSampleGlue computes them;
//! - the next step's eh_proj output x = BF16(tokenTable[token] + hiddenHalf), one FP32 add, where tokenTable holds
//!   eh_proj's embedding half applied to enorm(embed(t)) for the tokens t of this rank's vocabulary shard (BF16, one
//!   rounding of the FP32 sum), and hiddenHalf is eh_proj's hidden half applied to hnorm(hidden) in FP32, the ranks
//!   computing H / tpSize outputs each. Every rank publishes the token-table row of its candidate and its share of
//!   hiddenHalf with its (index, value) packet in one exchange; every rank then takes the winner's row;
//! - input_layernorm(x) and x, what eh_proj's all-reduce with input_layernorm and a zero residual returns.
struct DraftEhprojTableParams
{
    //! Token, state and exchange workspace as for invokeDraftSampleGlue; glueOut and nextNewTokens must be null. The
    //! workspace buffers hold numRows * tpSize slots of draftEhprojTableSlotBytes(tpSize) bytes.
    DraftSampleGlueParams step;
    //! This rank's hiddenHalf columns [rank * H / tpSize, (rank + 1) * H / tpSize), FP32 [numRows, H / tpSize] with
    //! row stride hiddenHalfStride.
    float const* hiddenHalf{nullptr};
    int64_t hiddenHalfStride{0};
    //! [vocabShard, H]: row t is eh_proj's embedding half applied to enorm(embed(rank * vocabShard + t)).
    __nv_bfloat16 const* tokenTable{nullptr};
    //! input_layernorm weight [H] and epsilon.
    __nv_bfloat16 const* normWeight{nullptr};
    float normEps{0.F};
    //! Outputs [numRows, H] with row stride outStride: input_layernorm(x) and x.
    __nv_bfloat16* normedOut{nullptr};
    __nv_bfloat16* residualOut{nullptr};
    int64_t outStride{0};
};

void invokeDraftEhprojTable(DraftEhprojTableParams const& params, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
