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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

// U-A70's DSpark Markov-block sampler (Gumbel-max, one draw per position) as CUDA C++, bit for bit the Triton kernels
// of speculative/markov_sampling.py it replaced. The block runs, per row of numGens:
// invokeMarkovPick (the anchor's w1 row into emb), then per position pos: emb @ w2^T (the caller's GEMM) and
// invokeMarkovDraw with the fused pick (arrivals zeroed once per block), then invokeMarkovProposal. Columns in chunks
// of 2048, numChunks = ceil(vocab / 2048) <= 128; 128 threads per CTA.

//! The pick: emb[row, :rank] = w1[token, :rank] (0 for an anchor outside [0, vocab)); fromDraw takes the token from
//! the draw's per-chunk argmax (keys, indices) and stores it at tokens[row, pos], else from anchors[row].
struct MarkovPickArgs
{
    float const* keys;
    int32_t const* indices;
    int32_t* tokens;
    void const* anchors; // int64 (anchors64) or int32 [numGens]
    bool anchors64;
    int64_t pos, numPos;
    void const* w1; // bf16 (w1Bf16) or fp32 [>= vocab, rank], row stride w1Stride
    bool w1Bf16;
    int64_t w1Stride;
    void* emb; // [numGens, rank], w1's dtype
    int64_t rank, vocab, numChunks, numGens;
    bool fromDraw, pdl;
};

//! The draw of position pos: biased[row, pos] = base + bias (fp32), the Gumbel-max per-chunk argmax keys / indices and
//! the per-chunk softmax statistics runMax / runSum; with arrivals (int32 [numGens], zero), tokens, w1 and emb the
//! row's last chunk CTA also runs the fromDraw pick (w1 / emb of the bias dtype).
struct MarkovDrawArgs
{
    void const* logits; // contiguous [numGens, numPos, vocab], bf16 (logitsBf16) or fp32
    bool logitsBf16;
    float* biased;      // contiguous fp32 [numGens, numPos, vocab]
    void const* bias;   // contiguous [numGens, vocab], bf16 (biasBf16) or fp32
    bool biasBf16;
    float const* temps;
    int64_t const* seeds;
    int64_t const* offsets;
    int64_t const* slots;
    float* keys;
    int32_t* indices;
    float* runMax;
    float* runSum;
    int64_t pos, numPos, offsetStep, paddingSlot, vocab, numChunks, numGens;
    int32_t* arrivals;
    int32_t* tokens;
    void const* w1;
    int64_t w1Stride;
    void* emb;
    int64_t rank;
    bool pdl;
};

//! The proposals: q_k = softmax(biased[row, k] / T) into probs[slots[row], k, :vocab] for every row and position.
struct MarkovProposalArgs
{
    float const* biased;
    float const* temps;
    float const* runMax;
    float const* runSum;
    int64_t const* slots;
    float* probs;
    int64_t probsSlotStride, probsPosStride, vocab, numChunks, numPos, numGens;
    bool pdl;
};

cudaError_t invokeMarkovPick(MarkovPickArgs const& args, cudaStream_t stream);
cudaError_t invokeMarkovDraw(MarkovDrawArgs const& args, cudaStream_t stream);
cudaError_t invokeMarkovProposal(MarkovProposalArgs const& args, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
