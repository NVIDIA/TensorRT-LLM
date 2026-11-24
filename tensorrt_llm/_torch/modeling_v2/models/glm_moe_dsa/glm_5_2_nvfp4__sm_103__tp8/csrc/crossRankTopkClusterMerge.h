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

namespace cross_rank_topk_cluster_merge
{

// Context-partitioned DSA indexer top-k with a cluster merge.
//
// It selects exactly what the kernel of crossRankTopk.h selects, in the same output order, with the
// same context partition, exchange sources, round-1 / round-2 protocol and Lamport workspace layout:
//   - global 64-token page p belongs to rank p % worldSize; within a rank, each verify row is handled by
//     kSourcesPerRank CTAs, CTA s owning the 8-token block (s + page) % kSourcesPerRank of every local page
//     (exchange source q = rank * kSourcesPerRank + s);
//   - each source selects its local top keys with crossRankTopk's rule and broadcasts them as (order-preserving key,
//     global token) pairs in source order, plus a header (count, exclusive bound of the withheld keys; this kernel adds
//     the key statistics of the keys sent), through the multicast Lamport workspace.
// What differs is how the row is merged and how a launch ends:
//   - the kSourcesPerRank CTAs of one row form a thread-block cluster. CTA s merges the candidates of sources
//     [s * worldSize, (s + 1) * worldSize) (at 8 ranks: the 8 sources of rank s). The cluster runs crossRankTopk's
//     merge select (the same digits and early-stop rules) on the cluster-summed histograms, and every CTA writes its
//     selected candidates directly into the output row at crossRankTopk's positions (keys above the threshold in
//     candidate order, then ties in candidate order, then -1). Each round-1 slot header carries its source's key
//     statistics, so every CTA derives the row's statistics from the headers it polls; the CTAs exchange only the
//     select's histogram totals, with st.async stores into each other's shared memory completed on transaction
//     barriers, and derive every CTA's output counts from the per-CTA tallies of the select. The only cluster-wide
//     barrier orders the barriers' initialization before the first exchange. A row in which a source withheld keys
//     above the threshold (identical decision on every rank) is merged instead by CTA 0 with crossRankTopk's single-CTA
//     merge, including its second round;
//   - every CTA keeps its own epoch word (one per (row, source CTA) slot): its thread 0 reads the word after the grid
//     dependency wait (only the previous exchange launch of this op writes it), the value is published to the CTA at
//     the block barrier that follows the key loads, and the CTA advances the word with a plain store after that
//     barrier (nothing else in the launch reads it; the next launch reads it after its own grid dependency wait); the
//     CTAs of the last row also advance the words of the rows the launch does not have, so every word counts the
//     exchange launches. There is no arrival counter and no end-of-kernel fence.
//
// Rows of at most kTopK valid tokens produce the identity selection followed by -1, without an exchange. A launch whose
// rows all have at most kTopK valid tokens (the same on every rank: the lengths come from the shared KV length) is an
// identity launch: it writes neither the workspace nor the epoch words. Every other launch is an exchange launch.
// Exchange launches rotate kNumLamportBuffers buffers by the epoch, which counts exchange launches; in exchange launch
// x every CTA resets its share of its own rank's copy of the buffer of exchange launch x - 1 between its send and its
// merge (the headers of its share are fetched once the epoch is known, after the grid dependency wait). Every rank's
// merges of x - 1 have read their copy of it, and a peer writes it again only in exchange launch x + 2, which starts
// after the peer's merges of x + 1 received this rank's sources of x + 1, i.e. after this rank completed launch x
// (identity launches in between neither write nor wait, so they do not shorten this chain). Every global memory
// access of the kernel follows its grid dependency wait, which orders it after the completion (and memory
// visibility) of the preceding kernels of the stream, including the previous launch of this op; the kernel never
// triggers its dependents early.
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
//! CTAs (exchange sources) per rank and verify row; also the cluster size (one cluster per row).
constexpr int32_t kSourcesPerRank = 8;
//! Exchange sources per row.
constexpr int32_t kMaxSources = kMaxRanks * kSourcesPerRank;
//! Round-1 slots per (source, row).
constexpr int32_t kSourceCapacity = 256;
//! Keys each saturated source aims to send in round 1 at 8 ranks (crossRankTopk's rule: three times the even share of
//! kTopK; refining stops at the first bin edge with at most four times the even share of keys at or above it).
constexpr int32_t kSourceBudget = 96;
//! Round-2 slots per (source, row).
constexpr int32_t kRound2Capacity = 2048;
//! Threads per CTA.
constexpr int32_t kThreads = 1024;
//! Lamport buffers rotated by the epoch.
constexpr int32_t kNumLamportBuffers = 3;
//! 32-bit word that marks an empty Lamport slot; never a valid key, token index, count or bound.
constexpr uint32_t kEmptyWord = 0xFFFFFFFFu;

//! 32-bit words of one (source, row) slot: a 32-byte header then two words per candidate, so candidates can be written
//! and read with 16-byte vector accesses. Round-1 header: count, exclusive bound of the withheld keys, then the key
//! statistics of the keys sent (largest key, smallest key + 1, keys >= +0.0, largest key below +0.0, smallest key
//! >= +0.0 plus 1; the "+ 1" maps the "none" value 0xFFFFFFFF to 0), then 0. Round-2 header: count, bound.
constexpr int64_t kSlotHeaderWords = 8;
constexpr int64_t kRound1SlotWords = kSlotHeaderWords + 2 * int64_t{kSourceCapacity};
constexpr int64_t kRound2SlotWords = kSlotHeaderWords + 2 * int64_t{kRound2Capacity};
//! 32-bit words of one Lamport buffer (round-1 region, then round-2 region).
constexpr int64_t kBufferWords = int64_t{kMaxSources} * kMaxRows * (kRound1SlotWords + kRound2SlotWords);
//! 32-bit words of the whole exchange workspace; every word must start as kEmptyWord.
constexpr int64_t kWorkspaceWords = kNumLamportBuffers * kBufferWords;
//! int32 words of the per-rank device state, zero-initialized: [0, 2) unused, [2] second rounds taken, [3] error word
//! (nonzero = the wait that timed out, written right before the kernel aborts), then one epoch word per
//! (row, source CTA) slot: state[kEpochBase + row * kSourcesPerRank + s].
constexpr int32_t kEpochBase = 4;
constexpr int32_t kStateWords = kEpochBase + kMaxRows * kSourcesPerRank;
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

//! Timestamps per CTA in CpTopkParams::debugTimes: [0, 16) %globaltimer (ns) and [16, 32) clock64 per phase (0 start,
//! 10 grid dependency wait returned, 1 keys loaded, 2 local select, 3 sent, 9 previous buffer reset (%globaltimer of
//! the first reset warp only), 4 candidates polled, 5 cluster statistics agreed, 6 merge select, 7 row written, 8 end),
//! [32, 42) local select (keys, then clock64 at the start of / after the histogram of / after the bin search of each
//! pass), [42, 45) send (start, keys taken, stores issued), 46 round-1 keys sent by this source, 47 local select passes
//! (+10 when the cut was searched among the negative keys), [48, 58) merge select (keys = this CTA's candidates; per
//! pass clock64 at its start / after the group totals arrived / after the bin was decided), clock64 of the merge: 45
//! candidates polled, 61 statistics combined, 58 emission positions scanned, 60 row written; 62 candidates of the row
//! (all CTAs), 63 merge select passes (+10 likewise; +100 when the row took crossRankTopk's single-CTA merge, which
//! then reuses 45 and 58-61 as crossRankTopk does); clock64 and values of thread 0.
constexpr int32_t kDebugSlots = 64;

//! Effective wait timeout of a launch with params.timeoutNs == 0 (ms).
int64_t cpTopkDefaultTimeoutMs();

//! Dynamic shared memory the kernel needs (bytes).
size_t cpTopkSmemBytes(int32_t localCapacity, int32_t worldSize);

//! Largest local capacity the kernel accepts for a world size.
int32_t cpTopkMaxLocalCapacity(int32_t worldSize);

//! Clusters of the kernel's launch configuration (largest local capacity) that can be resident at once on this
//! device (cudaOccupancyMaxActiveClusters, queried once per world size and device); 0 when the query fails. No row
//! waits for another row of its rank, but a row's cluster waits for the same row's sources on every peer rank, and
//! the ranks may make their rows resident in different orders; the caller therefore requires kMaxRows co-resident
//! clusters, so every cluster of a launch can be resident at once (other kernels sharing the SMs are finite and do
//! not wait for this one).
int32_t cpTopkMaxActiveClusters(int32_t worldSize);

void invokeCpTopk(CpTopkParams const& params, cudaStream_t stream);

} // namespace cross_rank_topk_cluster_merge

TRTLLM_MV2_NAMESPACE_END
