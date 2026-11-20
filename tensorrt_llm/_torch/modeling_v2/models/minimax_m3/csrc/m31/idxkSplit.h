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
namespace minimaxM3
{
namespace idxkSplit
{

//! IDXK_SPLIT: the MSA index-K pages of a TP8 decode step split over the 8 tensor-parallel ranks, with an
//! exact cross-rank top-16 merge whose table is the FSEL table (fselTopkSm107.h) bit for bit.
//!
//! M3.1 has 4 index heads; at TP8 rank r holds head r / 2, so today every rank scores every index-K page for one
//! head (ranks 2h and 2h + 1 the same head). Here rank r scores only the pages p with p % 8 == r, for all 4 heads
//! (one pass of the existing scorer at 4 heads x 8 query slots = its 32-column tile), then:
//!   1. compact (once per step, plain launch): the rank's compact block table (compact block j = global block
//!      j * 8 + r), the length handed to the scorer (n_local * 128 + qlen: no causal mask on any owned block; the
//!      causal cut is applied by the top-k from the per-query valid-block count), and the step epoch (+1);
//!   2. qx (per layer): ranks 0, 2, 4, 6 publish their index-q head to every rank; every rank gathers the 4 heads
//!      into the [rows, 4, 128] FP8 tile the scorer reads;
//!   3. the scorer (unchanged CuTe DSL kernel, 4 heads) over the compact table;
//!   4. topk (per layer): per (query row, head) the exact top 16 of the rank's candidate blocks by
//!      (score desc on the fp32 bit order, block asc), published to every rank;
//!   5. merge (per layer): per query row, the exact top `need` of the 8 x 16 candidates of this rank's head plus the
//!      FSEL forced blocks, written as the FSEL table (ascending block ids, -1 tail).
//! The global top `need` lies in the union of the per-rank top 16s (need <= 16), and the order is total (blocks are
//! unique), so the merged table equals the single-rank FSEL table whenever the per-block scores are equal, which
//! they are: the scorer's per-column arithmetic does not depend on the head count, and no candidate block holds a
//! position past its query (candidates end before the query's own block).
//!
//! Exchange: Lamport-tagged 64-bit words (no flags), written through a multicast address (one store reaches every
//! rank) or, without one, through each rank's unicast address. Halves alternate by epoch parity. The C tag is 18 bits,
//! so the merge re-arms every slot it consumes with a tag of the other parity, and the workspace starts with half 0's
//! C words at tag 1 (half 1 at tag 0): a slot then never matches before its producer rewrites it, however long its
//! row stayed idle.
//! Workspace per layer (64-bit words): Q [2][4 heads][rowsMax][32] then C [2][8 src][4 heads][rowsMax][16].
//!   Q word = epoch << 32 | 4 bytes of the FP8 row; C word = key << 32 | (epoch & 0x3FFFF) << 14 | (0x3FFF - block).

constexpr int32_t kWorld = 8;
constexpr int32_t kHeads = 4;
constexpr int32_t kTopk = 16;
constexpr int32_t kPage = 128;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kQWords = kHeadDim / 4;
//! Blocks per row the select contract covers (kFselTopkMaxBlocks).
constexpr int32_t kMaxBlocks = 9216;
//! Compact blocks per rank per row at kMaxBlocks.
constexpr int32_t kMaxLocal = (kMaxBlocks + kWorld - 1) / kWorld;

//! 64-bit words of one layer's workspace at rowsMax query rows.
inline int64_t layerWords(int32_t rowsMax)
{
    return static_cast<int64_t>(2) * kHeads * rowsMax * kQWords
        + static_cast<int64_t>(2) * kWorld * kHeads * rowsMax * kTopk;
}

//! Where a rank's stores go: `mc` (multicast, one store reaches every rank) when non-null, else `dst[0..world)`
//! (every rank's workspace base for this layer, unicast; the single-GPU emulation passes its emulated ranks' bases).
struct Dest
{
    unsigned long long* mc;
    unsigned long long* dst[kWorld];
    int32_t world;
};

//! Step setup (plain launch): cbt[b, j] = blockTable[b, j * 8 + rank] for j < n_local (else blockTable[b, 0]),
//! scoreLens[b] = n_local * 128 + qlen with n_local = ceil(((seqLens[b] - 1) / 128 - rank) / 8) clamped to
//! [0, cbtWidth - 1]; epoch[0] += 1.
cudaError_t invokeCompact(int32_t const* blockTable, int64_t btStride, int32_t btWidth, int32_t const* seqLens,
    int32_t batch, int32_t* cbt, int64_t cbtStride, int32_t cbtWidth, int32_t* scoreLens, int32_t* epoch, int32_t rank,
    int32_t qlen, cudaStream_t stream);

//! Index-q exchange of one layer. idxQ: this rank's head, rows x 128 FP8 bytes at qRowStride bytes. publish: store
//! the rows (only when rank is even: rank 2h publishes head h); gather: wait for the 4 heads of this epoch in the
//! local workspace and write qTile [rows, 4, 128] (contiguous). usePdl: wait for the producer grid first.
cudaError_t invokeQx(uint8_t const* idxQ, int64_t qRowStride, int32_t rows, int32_t rowsMax, Dest const& dest,
    unsigned long long const* local, uint8_t* qTile, int32_t const* epoch, int32_t rank, bool publish, bool gather,
    bool usePdl, cudaStream_t stream);

//! Per (query row, head): the exact top 16 of this rank's candidate blocks (global block j * 8 + rank, in
//! [initBlocks, min(cap, n_valid - localBlocks)) with the FSEL cap), from scores[head, row, j] (element strides),
//! published to every rank. Rows the FSEL table takes without a rank (at most 16 valid blocks, or every slot
//! forced) publish nothing.
cudaError_t invokeTopk(float const* scores, int64_t headStride, int64_t rowStride, int64_t blockStride,
    int32_t const* nValidBlocks, int32_t rows, int32_t rowsMax, int32_t numCols, int32_t initBlocks,
    int32_t localBlocks, Dest const& dest, int32_t const* epoch, int32_t rank, bool usePdl, cudaStream_t stream);

//! Per query row: the FSEL table of head `head` from the 8 ranks' candidates in the local workspace (each consumed
//! slot re-armed): out[row * outRowStride + 0..15]. triggerPhase 0 releases the dependent grid after the grid wait, 2
//! at exit.
cudaError_t invokeMerge(unsigned long long* local, int32_t const* nValidBlocks, int32_t rows, int32_t rowsMax,
    int32_t numCols, int32_t initBlocks, int32_t localBlocks, int32_t head, int32_t* out, int64_t outRowStride,
    int32_t const* epoch, bool usePdl, int32_t triggerPhase, cudaStream_t stream);

} // namespace idxkSplit
} // namespace minimaxM3
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
