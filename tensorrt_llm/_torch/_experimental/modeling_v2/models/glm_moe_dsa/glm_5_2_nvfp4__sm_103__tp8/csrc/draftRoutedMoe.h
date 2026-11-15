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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace draft_routed_moe
{

//! The routed experts of the GLM-5.2 MTP draft layer (BF16 weights, TP8/EP1 rank shard) on the
//! FlashInfer TRT-LLM-Gen BF16 layouts, bit-identical to ``trtllm_bf16_moe`` (DeepSeek-V3 routing with
//! ``n_group <= 1``, SwiGLU, ``do_finalize=True``).
//!
//! Two kernels, chained with programmatic dependent launch:
//!
//! * ``fc1``: every CTA computes the native top-8 routing of all rows from the FP32 router logits (sigmoid via
//!   ``tanh.approx``, selection key ``sigmoid + bias``, ties to the lower expert index, NaN keys first, weights
//!   ``bf16(sigmoid * scale / (sum + 1e-20))`` with the native FP32 sum order and approximate division), then its
//!   warps stream 16-row tiles of the distinct experts' gate/up weights (BlockMajorK, rows shuffled for the gated
//!   epilogue) and run ``mma.sync.m16n8k16`` over K = 6144 in natural order with one FP32 accumulator per tile,
//!   followed by the native SwiGLU epilogue. The BF16 intermediate goes to a scratch buffer.
//! * ``fc2``: after the FC1 grid has completed, each warp computes 16 down-projection rows for every distinct
//!   expert (K = 256 in natural order), rounds each row to BF16 like the native FC2 epilogue, and finalizes the
//!   rows of every token in top-k slot order (``fma`` from 0, one BF16 rounding), as ``finalizeKernel`` does.
//!
//! The weights are read in place: ``w13`` is ``[256][96][512][64]`` and ``w2`` is ``[256][4][6144][64]`` (BF16).

constexpr int kHidden = 6144;
constexpr int kInter = 256;   //!< Per-rank intermediate size (TP8).
constexpr int kFc1Rows = 512; //!< Gate and up rows per expert per rank.
constexpr int kNumExperts = 256;
constexpr int kTopK = 8;
constexpr int kMaxRows = 8;
constexpr int kMaxDistinct = kMaxRows * kTopK;
constexpr int kKChunk = 64; //!< BlockMajorK block (elements, 128 bytes).
constexpr int kFc1Chunks = kHidden / kKChunk;
constexpr int kFc2Chunks = kInter / kKChunk;
constexpr int kFc1Tiles = kFc1Rows / 16;
constexpr int kFc2Tiles = kHidden / 16;

//! Routing record written by the FC1 grid and read by the FC2 grid (int32 words).
constexpr int kRouteDistinctOffset = 1;                                 //!< [kMaxDistinct] expert ids, ascending.
constexpr int kRouteSlotOffset = kRouteDistinctOffset + kMaxDistinct;   //!< [kMaxRows][kTopK] distinct index.
constexpr int kRouteWeightOffset = kRouteSlotOffset + kMaxRows * kTopK; //!< [kMaxRows][kTopK] float bits of bf16.
constexpr int kRouteWords = kRouteWeightOffset + kMaxRows * kTopK;
//! Intermediate scratch: [kMaxDistinct][kMaxRows][kInter] BF16.
constexpr int64_t kScratchHElems = int64_t{kMaxDistinct} * kMaxRows * kInter;

//! Optional behaviour (``flags``).
enum DraftRoutedMoeFlags : int
{
    kFlagPdl = 1,          //!< Programmatic dependent launch for both kernels (as the native kernels).
    kFlagPrefetchFc2 = 2,  //!< FC1 warps prefetch the distinct experts' FC2 weights into L2 after their tiles.
    kFlagEarlyTrigger = 4, //!< FC1 triggers its dependents after routing instead of after its tiles.
    kFlagFc1Pairs = 8,     //!< FC1 warps interleave two tiles (two independent accumulation chains).
    kFlagFc2Pairs = 16,    //!< FC2 warps interleave two experts (two independent accumulation chains).
    kFlagDeepRing = 32,    //!< Deeper weight rings (FC1 20, FC2 12 chunks of 2 KB per warp instead of 12 and 8).
    kFlagTma = 64,         //!< TMA-fed kernels: one producer thread streams 16W-row weight boxes into a ring that W
                   //!< MMA warps consume (same per-chain arithmetic). Needs the tensor maps of invokeDraftRoutedMoe.
                   //!< With it, kFlagFc1Pairs gives every FC1 MMA warp two 16-row chains (boxes of 32W
                   //!< rows), kFlagFc2Pairs gives every FC2 MMA warp two experts per step (two stages, two
                   //!< accumulators), kFlagPrefetchFc2 is issued by the FC1 producer after its last weight
                   //!< copy, and kFlagDeepRing does not apply (the rings are sized to the shared memory).
    kFlagL2Prefetch = 128, //!< TMA FC1: the producer also prefetches its weight boxes into L2, 2 x ring depth ahead.
    kFlagPrefetchFc2Early = 256, //!< TMA FC1: the producer prefetches its share of the FC2 weights right after routing.
    //! GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE builds only: MMA warps wait for and release stages without reading them
    //! (memory schedule alone; the output is meaningless).
    kFlagProbeNoMma = 512,
    //! GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE builds only: the FC1 producer fills the ring once and the MMA warps cycle
    //! over the resident stages without waiting (MMA warps alone; meaningless output).
    kFlagProbeResident = 1024,
    //! TMA FC1 with 4 MMA warps: when the call's tiles (decided on the device from the number of distinct experts)
    //! would need more than one round of the grid, every CTA instead takes three consecutive 64-row sub-tiles (three
    //! chains per MMA warp, three TMA boxes per stage), if those fit in one round. Needs
    //! DraftRoutedMoeTensorMaps::w13Triple.
    kFlagFc1Triples = 4096,
    //! TMA FC1: the early trigger of kFlagEarlyTrigger for calls on the kFlagFc1Triples path only (other calls trigger
    //! after their tiles).
    kFlagEarlyTriggerTriples = 8192,
    //! TMA FC1: the weight boxes are loaded with an L2 evict-first policy (each is read once per call), so that they
    //! do not displace FC2 weights prefetched into L2.
    kFlagFc1EvictFirst = 32768,
    //! TMA FC1 with 4 MMA warps and tile pairs (ignored otherwise): the activation rows are not kept in shared
    //! memory; every ring stage also carries the activation K chunks it needs (boxes of DraftRoutedMoeTensorMaps::x),
    //! so the weight ring is 6 stages deep (4 on the triples path) for every row count. Needs
    //! DraftRoutedMoeTensorMaps::x.
    kFlagStreamX = 65536
};

//! Event record of the probe builds (compiled with ``GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE=1``): int64
//! ``[2][kTimelineCtas][kTimelineSlots]`` (FC1, FC2), one row per CTA; see TimelineSlot in draftRoutedMoe.cu.
constexpr int kTimelineCtas = 1024;
constexpr int kTimelineSlots = 16;

//! K chunks per FC1 stage of the TMA-fed kernel for all its 16-row chains together (32 KB per stage and box).
constexpr int kTmaStageChunks = 16;

//! Sub-tiles per CTA, rows per sub-tile and K chunks per stage of the kFlagFc1Triples path (4 MMA warps): a stage is
//! three boxes of 64 rows x 2 K chunks (16 KB each).
constexpr int kTripleSubTiles = 3;
constexpr int kTripleRows = 64;
constexpr int kTripleStageChunks = 2;

//! Activation box of kFlagStreamX: 8 tokens (rows beyond the call's are zero-filled) x kXBoxChunks K chunks.
constexpr int kXBoxChunks = 2;
constexpr int kXBoxBytes = 8 * kXBoxChunks * kKChunk * 2;

//! Tensor maps of the TMA-fed kernels (kFlagTma): 3D views [slab][row][64] BF16 of ``w13`` (256 * 96 slabs of 512
//! rows) and ``w2`` (256 * 4 slabs of 6144 rows), 128-byte swizzle; boxes of 64 x fc1TmaBoxRows x fc1TmaBoxSlabs
//! (FC1) and 64 x 16 * fc2Warps x 4 (FC2).
struct DraftRoutedMoeTensorMaps
{
    CUtensorMap w13;
    CUtensorMap w2;
    CUtensorMap w13Triple; //!< kFlagFc1Triples: view of ``w13`` with boxes of 64 x kTripleRows x kTripleStageChunks.
    //! kFlagStreamX: 3D view [token][K chunk][64] of the call's ``x`` (dims 64, 96, rows), boxes of 64 x kXBoxChunks x
    //! 8, 128-byte swizzle.
    CUtensorMap x;
};

struct DraftRoutedMoeParams
{
    float const* logits;       //!< [rows, 256] FP32.
    __nv_bfloat16 const* bias; //!< [256] router bias.
    __nv_bfloat16 const* x;    //!< [rows, 6144] activation (16-byte aligned).
    __nv_bfloat16 const* w13;  //!< [256][96][512][64].
    __nv_bfloat16 const* w2;   //!< [256][4][6144][64].
    __nv_bfloat16* out;        //!< [rows, 6144] finalized routed output.
    __nv_bfloat16* h;          //!< Scratch, kScratchHElems.
    int32_t* route;            //!< Scratch, kRouteWords.
    __nv_bfloat16* debugFc2;   //!< Optional [rows, 8, 6144]: FC2 rows per (token, slot) in hidden order.
    int64_t* timeline; //!< Optional event record (GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE builds; ignored otherwise).
    int rows;
    float routedScaling;
    int fc1Ctas;
    int fc1Warps; //!< 1, 2 or 4 warps per FC1 CTA.
    int fc2Ctas;
    int fc2Warps; //!< 1, 2 or 4 warps per FC2 CTA (also 3 with kFlagTma: 128 groups of 48 rows).
    int flags;
};

//! Dynamic shared memory of the FC1 kernel for ``rows`` and ``warps`` (bytes); ``deep``: kFlagDeepRing.
int fc1SmemBytes(int rows, int warps, bool deep);

//! Dynamic shared memory of the FC2 kernel for ``warps`` (bytes); ``deep``: kFlagDeepRing.
int fc2SmemBytes(int warps, bool deep);

//! Dynamic shared memory of the TMA-fed FC1 kernel (bytes, incl. 1 KB alignment slack); stages: ring depth.
int fc1TmaSmemBytes(int rows, int stages);

//! Dynamic shared memory of the TMA-fed FC1 kernel with the kFlagFc1Triples path (bytes): the larger of the two
//! rings (``stages`` of 32 KB, ``tripleStages`` of three sub-tile boxes) next to the activation rows.
int fc1TmaTriplesSmemBytes(int rows, int stages, int tripleStages);

//! Dynamic shared memory of the TMA-fed FC1 kernel with kFlagStreamX (bytes): the larger of the 6-stage regular and
//! the 4-stage triples rings, activation boxes included.
int fc1TmaStreamXSmemBytes();

//! Dynamic shared memory of the TMA-fed FC2 kernel (bytes, incl. 1 KB alignment slack); stages: ring depth.
int fc2TmaSmemBytes(int warps, int stages);

//! Launches the FC1 and FC2 kernels on ``stream``; ``fc1Warps``/``fc2Warps`` must be 1, 2 or 4 (``fc2Warps`` also 3
//! with kFlagTma). ``maps`` is required
//! with kFlagTma (see DraftRoutedMoeTensorMaps) and ignored otherwise.
cudaError_t invokeDraftRoutedMoe(
    DraftRoutedMoeParams const& params, DraftRoutedMoeTensorMaps const* maps, cudaStream_t stream);

//! Rows of the FC1 weight box of the TMA-fed kernel (16 per chain).
inline int fc1TmaBoxRows(int flags, int fc1Warps)
{
    return 16 * fc1Warps * ((flags & kFlagFc1Pairs) ? 2 : 1);
}

//! K chunks (slabs) of the FC1 weight box of the TMA-fed kernel.
inline int fc1TmaBoxSlabs(int flags, int fc1Warps)
{
    return kTmaStageChunks * 16 / fc1TmaBoxRows(flags, fc1Warps);
}

} // namespace draft_routed_moe

TRTLLM_MV2_NAMESPACE_END
