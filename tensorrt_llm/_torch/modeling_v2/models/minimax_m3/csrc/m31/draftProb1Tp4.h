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

#include <cuda_runtime.h>
#include <stdint.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace minimaxM3Sm107Tp4
{

//! The peer's probabilistic DSpark Markov draft (dk_mk_prob1: all K steps of one draft in one launch,
//! Gumbel-max on this rank's vocab shard, one (key, id) word per row and step exchanged across the TP ranks, the biased
//! bf16 draft logits mirrored into every rank's draft_logits cache) retaining measured arithmetic: draftProb1Tp4/ holds
//! the peer's dspark_markov_prob1.cuh, dspark_markov_prob.cuh and dspark_rng.h with formatting-only qualifier changes
//! and the documented uniform producer gate, full-generation exchange readiness, and invalid-anchor zero embedding. The
//! launch body is the peer extension's steps() without its pybind layer; its cudaMalloc / cudaIpc workspace
//! (dspark_markov_prob_ext.cu ws_create / ws_open / cache_alloc / cache_open) is replaced by an MNNVL allocation the
//! Python host makes (kernels/m3ll/prob1).
//!
//! Around it, five small kernels of ours (not the peer's): prep (the per-row request index and Philox position), agree
//! (a device-side, cross-rank fault decision: the peer's bounded waits set err instead of trapping; every rank
//! publishes its err to every rank and all ranks take the same decision), the reader-completion acknowledgement, and
//! the draft_probs adapter (stats + write): the verifier's proposal q = softmax(lg / T) of the same bf16 logits the
//! Gumbel-max drew from, one-hot for T == 0, uniform for CUDA-graph padding requests, and on an agreed fault token 0
//! with a one-hot q at 0 for every position of every row (each stored q then equals its deterministic draw, so
//! rejection stays exact).
namespace draftProb1
{

constexpr int kMaxRows = 8;                          // dkmkp::kMaxC
constexpr int kMaxWorld = 4;                         // dkmkp::kMaxW
constexpr int kMaxK = 16;                            // dkmkp1::kMaxK
constexpr int kRank = 256;                           // dkmkp::kR (the Markov rank)
constexpr int kMaxGrid = 512;                        // dkmkp::kMaxGrid
constexpr int kSlotWords = 2 * kMaxRows * kMaxWorld; // the exchange slots: [2][kMaxC][kMaxW] u64
constexpr int kStatSplit = 8;                        // vocab chunks per row in the adapter's stats pass

//! The peer kernel's arguments (dkmkp1::Args), pointers only; the op fills it with the peer steps() checks.
struct Prob1Params
{
    void const* base; // bf16 [C*k][ld]: row (r*k + step) of this rank's vocab shard
    int64_t ld;
    void const* w1;   // bf16 [V][256]
    int64_t w1Rows;   // actual table row bound for the stock invalid-anchor zero embedding
    void const* w2s;  // bf16 [Vs][256]
    int vs;
    int voff;
    int rows;              // C
    int k;
    int64_t const* prevIn; // [C] step-0 anchors
    int64_t* prev0;
    int64_t* prev1;
    int64_t* tokOut;          // [C][tokLd]
    int tokLd;
    int32_t const* idxMap;    // [C*k] request row (-1 = padding: no cache write, temperature 0)
    int64_t const* samplePos; // [C*k]
    float const* temperature; // [C]
    int64_t const* seeds;     // [C]
    void* caches[kMaxWorld];  // [0] own draft_logits, then the peers' (mirror)
    int ncache;
    int64_t cs0, cs1;         // draft_logits strides (row, step)
    unsigned long long* part; // [kMaxC][kMaxGrid]
    float* partMs;            // [kMaxC][kMaxGrid][2]
    unsigned int* counter;
    unsigned long long* epoch;
    unsigned long long* mine; // this rank's exchange slots [2][kMaxC][kMaxW]
    unsigned long long* peers[kMaxWorld];
    int* err;
    unsigned long long const* launchGate; // stable per-call device gate prepared before producer launch
    int rank, world;
};

//! Largest co-resident grid of the peer kernel at `rows` rows on the current device (<= kMaxGrid); 0 if rows is out
//! of range.
int maxGrid(int rows);

//! dkmkp1::launch (programmatic stream serialization as `pdl`).
cudaError_t launchProb1(Prob1Params const& p, int grid, bool pdl, cudaStream_t stream);

//! idxMap[r*k + s] = r (or -1 when slotIds[r] == dummySlotRow, dummySlotRow > 0), samplePos[r*k + s] = offsets[r] +
//! rngBase + s * span + 1 (the peer kernel draws step s of row r at Philox position samplePos - 1).
cudaError_t launchPrep(int rows, int k, int64_t const* slotIds, int64_t dummySlotRow, int64_t slotCount, int* err,
    int64_t const* offsets, int64_t rngBase, int64_t span, int32_t* idxMap, int64_t* samplePos,
    unsigned long long* const* faultPeers, int world, unsigned long long const* seq, int64_t capNs,
    unsigned long long* gate, cudaStream_t stream);

//! Linearizable shared rank0 fault decision. Fault-area words0..7 are votes; word8 on rank0 is the
//! monotonically committed(sequence,state), word9 locally is the immutable per-call producer gate; word10 is reader
//! completion. ABORT is sticky. Never reset err/counter/epoch/slots in place; recover only by collective workspace and
//! graph rebuild. Sequences must be consecutive from1 and the workspace rebuilt before2^60-1 calls.
cudaError_t launchAgree(int rank, int world, int* err, unsigned int* counter, unsigned long long* epoch,
    unsigned long long* mine, unsigned long long* faultMine, unsigned long long* const* faultPeers,
    unsigned long long* seq, int64_t capNs, int* fault, int64_t* faultCount, cudaStream_t stream);

//! The verifier's proposal for the K draft rows of each request, from the own draft_logits cache (full vocab after the
//! mirror). Pass 1: per (row, step, chunk) (max, sum exp) of lg / T. Pass 2: draftProbs[slot[r]][s][:V] = q,
//! tokens32[r][s] = the drawn token; T == 0 -> one-hot at the token; padding -> uniform 1 / V; *fault -> token 0 and
//! one-hot at 0 for every (r, s).
struct ProbsParams
{
    unsigned long long* readerAck;
    unsigned long long const* seq;
    void const* cache;        // bf16 [>= C][>= k][>= V], strides cs0 / cs1 / 1
    int64_t cs0, cs1;
    int64_t const* tok;       // [C][tokLd]
    int tokLd;
    float const* temperature; // [C]
    int64_t const* slotIds;   // [C]
    int64_t dummySlotRow;     // > 0: rows whose slot equals it are padding
    int64_t slotCount;        // capacity of draftProbs; invalid slot ids fault without a write
    int const* fault;
    float* stats;             // [C][k][kStatSplit][2]
    float* draftProbs;        // fp32, [slot][step][vocab] with strides ds0 / ds1 / 1
    int64_t ds0, ds1;
    int32_t* tokens32;        // [C][k]
    int rows, k, vocab;
};

cudaError_t launchProbs(ProbsParams const& p, cudaStream_t stream);

} // namespace draftProb1

} // namespace minimaxM3Sm107Tp4

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
