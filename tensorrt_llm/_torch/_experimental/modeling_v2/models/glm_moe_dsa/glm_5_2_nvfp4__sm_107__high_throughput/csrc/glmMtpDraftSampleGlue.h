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
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::glm_rubin_mtp
{

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
//! - the state of the next draft step: position ids, sequence and KV lengths;
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

    // Lamport workspace: three buffers of tpSize * segmentRows * tpSize packets, the local unicast view, the
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

    // Output rows after the last draft step (skipped when nextNewTokens is null), [numRows, numDraftTokens + 1]
    // int32: nextNewTokens[r, 0] = acceptedTokens[batchIndices[r], numAccepted[r] - 1], then the draft tokens.
    int32_t const* acceptedTokens{nullptr};
    int64_t acceptedTokensStride{0};
    int32_t const* batchIndices{nullptr};
    int32_t const* numAccepted{nullptr};
    int32_t* nextNewTokens{nullptr};
};

//! Largest rows per rank (segment) of invokeDraftSampleGlueAdp: one lane of a warp per row.
constexpr int kDraftSampleGlueAdpMaxSegmentRows = 32;
//! Largest LM-head tensor-parallel group of invokeDraftSampleGlueAdp: one lane of a warp per rank.
constexpr int kDraftSampleGlueAdpMaxRanks = 32;

//! One MTP draft step after the draft LM head under attention DP with the LM head tensor-parallel over a group of
//! ranks (each rank multiplies the group's stacked rows, every rank's requests padded to segmentRows rows, by its
//! vocabulary shard): what invokeDraftSampleGlue computes, for this rank's rows, with one CTA per stacked row.
//! - Every CTA scans its row of this rank's logits shard, publishes the row's (index, value) packet through the
//!   multicast address of a dedicated Lamport workspace (the layout and flags of the MNNVL all-gather) and reads the
//!   row's packets of every rank: each rank publishes every row, and each published packet is read before its buffer
//!   is emptied two calls later.
//! - The CTAs of this rank's rows pick the global argmax with the order of torch.argmax (NaN first, lowest index on
//!   ties) and write the token, the row's eh_proj input window and, after the last draft step, its next-new-tokens
//!   row; the CTA of this rank's first row writes the next step's state.
struct DraftSampleGlueAdpParams
{
    //! As for invokeDraftSampleGlue, except: logits holds this rank's vocabulary shard of the stacked rows,
    //! [tpSize * segmentRows, vocabShard], rank r's requests at rows [r * segmentRows, r * segmentRows + its numRows);
    //! numRows is this rank's requests, and every per-row output and state field covers this rank's rows. The
    //! workspace buffers hold tpSize * segmentRows * tpSize packets.
    DraftSampleGlueParams step;
    //! Rows of each rank's segment (its requests padded), 1 to kDraftSampleGlueAdpMaxSegmentRows.
    int segmentRows{0};
};

void invokeDraftSampleGlueAdp(DraftSampleGlueAdpParams const& params, cudaStream_t stream);

} // namespace kernels::glm_rubin_mtp

TRTLLM_MV2_NAMESPACE_END
