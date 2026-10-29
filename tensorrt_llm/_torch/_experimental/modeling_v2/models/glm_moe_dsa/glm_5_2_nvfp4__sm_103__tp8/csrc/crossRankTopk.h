/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
#include <cuda_runtime_api.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace cross_rank_topk
{

// Context-partitioned DSA indexer top-k.
//
// The indexer context of one generation request is partitioned over the ranks of a tensor-parallel group by whole
// 64-token pages: global page p belongs to rank p % worldSize, and a rank's local column c maps to global token
// ((c / 64) * worldSize + rank) * 64 + c % 64. Every rank scores only its own pages (the caller supplies the local
// logits). Within a rank, each verify row is handled by kSourcesPerRank CTAs; CTA s owns one 8-token block of every
// local page (block (s + page) % kSourcesPerRank), so the (rank, CTA) "sources" interleave the context finely and a
// fully selected page is spread over all sources of its rank. Each source
//   1. selects its local top keys (kSourceBudget at 8 ranks, see below) with a block radix select,
//   2. broadcasts them as (order-preserving key, global token) pairs, plus a header (count, exclusive bound of the
//      keys it withheld), through a multicast Lamport workspace.
// One CTA per row and rank then merges the candidates of all sources into the exact top-kTopK of the row,
// identically on every rank. A source that withheld a key above the merged threshold makes every rank run a
// second round in which that source's top keys between the threshold and its bound are sent (at most
// kRound2Capacity, by the merge CTA of the source's rank, from the local logits), so the result is the tie-aware
// exact top-kTopK for any input. In the second round, a source whose cut falls inside a run of equal keys sends the
// keys above the cut and the first equal ones in source order up to its capacity (round 1 sends only the keys above
// such a cut, so the two rounds never send the same key).
// Rows of at most kTopK valid tokens produce the identity selection followed by -1, without an exchange.
// A launch whose rows all have at most kTopK valid tokens (the same on every rank: the lengths come from the shared
// KV length) is an identity launch: it touches neither the workspace nor the epoch. Every other launch is an
// exchange launch. Exchange launches rotate kNumLamportBuffers buffers by a device epoch that counts exchange
// launches; in exchange launch x the CTAs that do not merge reset, on their own rank, the buffer of exchange launch
// x - 1. Every rank's merges of x - 1 have read their copy of it, and a peer writes it again only in exchange
// launch x + 2, which starts after the peer's merge of x + 1 received this rank's sources of x + 1, i.e. after this
// rank completed launch x (identity launches in between neither write nor wait, so they do not shorten this chain).
//
// A merge that waits longer than the timeout (CpTopkParams::timeoutNs, %globaltimer) for a peer's words records the
// wait in the state's error word and aborts the kernel with a trap: the exchange cannot complete correctly any more,
// and a late peer write could land in a buffer this rank reuses, so the run fails instead of continuing.

//! Selected indices per verify row (DSA index_topk).
constexpr int32_t kTopK = 2048;
//! Ranks of the context-parallel group.
constexpr int32_t kMaxRanks = 8;
//! Verify rows per call.
constexpr int32_t kMaxRows = 8;
//! Tokens per indexer K page; the context is partitioned by whole pages.
constexpr int32_t kPageTokens = 64;
//! CTAs (exchange sources) per rank and verify row.
constexpr int32_t kSourcesPerRank = 8;
//! Exchange sources per row.
constexpr int32_t kMaxSources = kMaxRanks * kSourcesPerRank;
//! Round-1 slots per (source, row).
constexpr int32_t kSourceCapacity = 256;
//! Keys each saturated source aims to send in round 1 at 8 ranks (64 sources): three times the even share of kTopK
//! (the largest share of the exact top-2048 that one source of this layout held on real GLM-5.2 indexer logits was
//! 69). A source stops refining as soon as its keys at or above a bin edge number at most four times the even
//! share, so it sends between 96 and 128 keys at 64 sources. Other group sizes scale both with ceil(kTopK / sources).
constexpr int32_t kSourceBudget = 96;
//! Round-2 slots per (source, row).
constexpr int32_t kRound2Capacity = 2048;
//! Threads per CTA.
constexpr int32_t kThreads = 1024;
//! Lamport buffers rotated by the device epoch.
constexpr int32_t kNumLamportBuffers = 3;
//! 32-bit word that marks an empty Lamport slot; never a valid key, token index, count or bound.
constexpr uint32_t kEmptyWord = 0xFFFFFFFFu;

//! 32-bit words of one (source, row) slot: a 16-byte header (count, bound, two unused words) then two words per
//! candidate, so candidates can be written and read with 16-byte vector accesses.
constexpr int64_t kRound1SlotWords = 4 + 2 * int64_t{kSourceCapacity};
constexpr int64_t kRound2SlotWords = 4 + 2 * int64_t{kRound2Capacity};
//! 32-bit words of one Lamport buffer (round-1 region, then round-2 region).
constexpr int64_t kBufferWords = int64_t{kMaxSources} * kMaxRows * (kRound1SlotWords + kRound2SlotWords);
//! 32-bit words of the whole exchange workspace; every word must start as kEmptyWord.
constexpr int64_t kWorkspaceWords = kNumLamportBuffers * kBufferWords;
//! int32 words of the per-rank device state: epoch (exchange launches), CTA arrivals, second rounds taken, error
//! word (nonzero = the wait that timed out, written right before the kernel aborts).
constexpr int32_t kStateWords = 4;
//! Default wait for a peer's Lamport words before the kernel aborts
//! (TRTLLM_STAIRCASE_GLM_LL_CROSS_RANK_TOPK_TIMEOUT_MS overrides it).
constexpr int64_t kDefaultTimeoutMs = 120000;

//! Values of the error word.
enum CpTopkWait : int32_t
{
    kWaitRound1Header = 1,
    kWaitRound1Candidates = 2,
    kWaitRound2Header = 3,
    kWaitRound2Candidates = 4
};

struct CpTopkParams
{
    //! [numRows, logitsStride] fp32 logits of this rank's local columns.
    float const* logits;
    int64_t logitsStride;
    //! Local columns the logits view holds (each row's local length is clamped to it).
    int32_t localCapacity;
    //! [1] int32 global indexer KV length of the request; row r has kvLen - nextN + r + 1 valid tokens.
    int32_t const* kvLens;
    //! [numRows, kTopK] int32 output: global token indices, -1 for unused slots.
    int32_t* out;
    int32_t numRows;
    int32_t nextN;
    int32_t rank;
    int32_t worldSize;
    //! Multicast address of the exchange workspace (kWorkspaceWords words, identical layout on every rank).
    uint32_t* mcWorkspace;
    //! This rank's unicast address of the same workspace.
    uint32_t* ucWorkspace;
    //! [kStateWords] int32 per-rank device state, zero-initialized.
    int32_t* state;
    //! Optional [numRows, kSourcesPerRank, kDebugSlots] int64 phase timestamps (diagnosis only; nullptr = off).
    int64_t* debugTimes = nullptr;
    //! Longest wait for a peer's words before the kernel aborts (ns); 0 = the default (kDefaultTimeoutMs, or
    //! TRTLLM_STAIRCASE_GLM_LL_CROSS_RANK_TOPK_TIMEOUT_MS).
    uint64_t timeoutNs = 0;
};

//! Timestamps per CTA in CpTopkParams::debugTimes: [0, 16) %globaltimer (ns), [16, 32) clock64 per phase,
//! [32, 42) local select (keys, then clock64 at the start of / after the histogram of / after the bin search of
//! each pass), [42, 45) send (counter ready, keys taken, stores issued), 45 candidates polled before the key-range
//! reduction, 46 round-1 keys sent by this source, 47 local select passes (+10 when the cut was searched among the
//! negative keys), [48, 58) merge select (as the local select), [58, 62) emission (threshold decided, positions
//! scanned, row assembled, row written), 62 merge candidates, 63 merge select passes (+10 likewise); clock64 and
//! values of thread 0.
constexpr int32_t kDebugSlots = 64;

//! Effective wait timeout of a launch with params.timeoutNs == 0 (ms).
int64_t cpTopkDefaultTimeoutMs();

//! Dynamic shared memory the kernel needs (bytes).
size_t cpTopkSmemBytes(int32_t localCapacity, int32_t worldSize);

//! Largest local capacity the kernel accepts for a world size.
int32_t cpTopkMaxLocalCapacity(int32_t worldSize);

void invokeCpTopk(CpTopkParams const& params, cudaStream_t stream);

} // namespace cross_rank_topk

TRTLLM_MV2_NAMESPACE_END
