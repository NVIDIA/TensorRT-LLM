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
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::hist_topk
{

// Key-sharded DSA indexer top-k of one generation request, exact, by histogram exchange.
//
// The indexer context of the request is split over the ranks of a tensor-parallel group into contiguous shards:
// S = ceil(ceil(kvLen / worldSize) / kShardAlign) * kShardAlign, rank r holds the tokens [r S, r S + n_r) with
// n_r = clamp(kvLen - r S, 0, S), and scores only those (the caller supplies the shard's logits). Verify row i of
// nextN sees the tokens [0, kvLen - nextN + i + 1).
//
// Per row, one cluster of kCtasPerRow CTAs holds the row's shard keys (order-preserving keys of the FP32 logits).
// Every rank
//   1. builds the histogram of its keys' top 11 bits (per CTA in shared memory, reduced over the cluster: every CTA
//      owns 1/kCtasPerRow of the bins and receives the other CTAs' counts of them as asynchronous stores into its
//      shared memory, which complete on its barrier) and stores it once through the multicast address: the 32-bit
//      totals of its 32-bin groups, and two 16-bit bin counts per word, saturated at 0xFFFF (at most one bin of a
//      group can saturate, as two would hold more than kMaxShard keys; the receivers recover its count from the group
//      total);
//   2. reads all ranks' group totals, finds the group that holds the kTopK-th largest key, reads that group's bins
//      of all ranks and finds the cut bin: the keys above it are selected, and every rank knows each rank's count
//      above it and in it;
//   3. when that bin is not taken whole and holds more than kBoundaryCap keys, refines the same way with the next
//      11 bits of the keys in the bin, then with the last 10 bits (the exact key, whose ties are taken from the
//      higher ranks first: the larger tokens first);
//   4. stores its selected tokens through the multicast address into every rank's output stage, at its offset (rank,
//      then CTA), and, when the boundary bin is neither taken whole nor the exact key, its boundary (key, token) pairs
//      into every rank's boundary stage. After a cut on the top 11 bits (the common case) a CTA's tokens and pairs are
//      prefixes of buckets it fills while it waits for the peers' histograms: its keys in the bins that cut cannot be
//      above, by bin from the top, in the order of shared-memory atomics within a bin; after a later cut they are in
//      position order;
//   5. copies the output stage into the output row; one CTA selects the boundary pairs that complete the row, by key
//      and then the larger token, and appends them in stage order.
// Every rank derives each decision from the same multicast data, so all ranks write the same bits: the exact top-kTopK
// of the row by (logit, token), the larger token first among equal logits. The order within the row is the stages'
// order: the same on all ranks, but after a cut on the top 11 bits it can differ between calls with the same logits.
// Rows of at most kTopK tokens get [0, n) and -1 without an exchange, and a launch whose rows all have at most kTopK
// tokens touches neither the workspace nor the state.
//
// Workspace: kNumBuffers Lamport buffers, rotated by a device epoch that counts exchange launches. In exchange launch x
// a rank clears, on its own copy, what launch x - 1 wrote into buffer (x - 1) % kNumBuffers (the sizes are recorded
// in the state per row), each histogram word after it reads the word written (launch x - 1 read only the group
// totals and one group's bins); a peer writes that buffer again only in launch x + 2, after this rank completed
// launch x + 1.
//
// A wait for a peer's word that exceeds the timeout records the wait in the state's error word and aborts the kernel:
// the selection can no longer complete, and a late peer write could land in a buffer this rank reuses.

constexpr int32_t kTopK = 2048;
constexpr int32_t kMaxRanks = 16;
constexpr int32_t kMaxRows = 8;
//! CTAs per row (one cluster).
constexpr int32_t kCtasPerRow = 8;
constexpr int32_t kShardAlign = 128;
//! Largest shard (tokens per rank): three float4 per thread of kCtasPerRow CTAs of 1024 threads. At 16 ranks a request
//! of up to 1,572,864 indexer tokens.
constexpr int32_t kMaxShard = 98304;
//! Boundary-bin keys exchanged per row; a fuller boundary bin is refined with the next digit instead.
constexpr int32_t kBoundaryCap = 4096;
constexpr int32_t kNumBuffers = 3;
//! Bins of the first two digits (11 bits); the last digit has 10 bits.
constexpr int32_t kBins = 2048;
//! Bins per group of the coarse summary.
constexpr int32_t kGroupBins = 32;
//! Lamport words of one (level, row, rank) histogram: kBins / 2 words of two 16-bit bin counts, then the 32-bit totals
//! of the kBins / kGroupBins groups.
constexpr int32_t kHistWords = kBins / 2 + kBins / kGroupBins;
constexpr int32_t kLevels = 3;
constexpr int64_t kBufferWords = int64_t{kLevels} * kMaxRows * kMaxRanks * kHistWords + int64_t{kMaxRows} * kTopK
    + int64_t{kMaxRows} * kBoundaryCap * 2;
//! 32-bit words of the exchange workspace; every word must start as 0xFFFFFFFF.
constexpr int64_t kWorkspaceWords = kNumBuffers * kBufferWords;
//! int32 words of the per-rank device state (zero-initialized): a 64-bit count of the CTAs that completed exchange
//! launches (the epoch is that count / (kCtasPerRow * kMaxRows)), the error word, a spare word, then per row the
//! levels, output-stage words and boundary pairs the last exchange launch wrote.
constexpr int32_t kStateWords = 4 + kMaxRows * 4;
//! Wait for a peer's words before the kernel aborts, when HistTopKParams::timeoutNs is 0.
constexpr int64_t kDefaultTimeoutMs = 120000;
//! Debug slots per CTA (phase %globaltimer stamps and per-row statistics), for diagnosis only.
constexpr int32_t kDebugSlots = 64;

struct HistTopKParams
{
    //! [numRows, logitsStride] FP32 logits of this rank's shard (columns [0, localCapacity), localCapacity >= S).
    float const* logits;
    int64_t logitsStride;
    int32_t localCapacity;
    //! [1] int32 indexer KV length of the request.
    int32_t const* kvLens;
    //! [numRows, kTopK] int32 output: request-local token indices, -1 for unused slots.
    int32_t* out;
    int32_t numRows;
    int32_t nextN;
    int32_t rank;
    int32_t worldSize;
    //! Multicast and this rank's unicast address of the exchange workspace (kWorkspaceWords words).
    uint32_t* mcWorkspace;
    uint32_t* ucWorkspace;
    //! [kStateWords] int32 per-rank device state, 8-byte aligned.
    int32_t* state;
    //! Optional [kMaxRows, kCtasPerRow, kDebugSlots] int64 (nullptr = off).
    int64_t* debug = nullptr;
    //! Longest wait for a peer's words (ns); 0 = the default.
    uint64_t timeoutNs = 0;
};

void invokeDsaHistTopK(HistTopKParams const& params, cudaStream_t stream);

//! Shard of the generation request for `rank`: shardCtx[0] = max(n_r, 1) (an empty shard scores one token no row
//! reaches) and shardBlockTable[j] = blockTable[r S / tokensPerBlock + j] for the shard's pages, 0 beyond. One CTA.
void invokeDsaHistShardMeta(int32_t const* kvLens, int32_t const* blockTable, int32_t blockTableWidth,
    int32_t worldSize, int32_t rank, int32_t tokensPerBlock, int32_t* shardCtx, int32_t* shardBlockTable,
    int32_t shardBlockTableWidth, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa::hist_topk

TRTLLM_NAMESPACE_END
