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

// Sentinel transport of the NVLink one-sided MoE all-to-all for small steps: one dispatch and one
// combine kernel, no completion flags and no fences between data and its readiness.
//
// Sentinel arrival. Every landing row a peer writes into is armed with a sentinel word that no
// payload word takes (senders rewrite the few words that could, see canonicalize_* in
// moeAlltoAllSentinel.cu). A receiver polls the data itself: a 16-byte vector has arrived when none
// of its four 32-bit words is the sentinel (32-bit stores are single-copy atomic, so a torn vector
// still shows one). The receiver copies the vector out and re-arms it; it is the vector's only
// reader.
//
// Landing rows of a rank's workspace, row (src, token) = token * epSize + src (the same rows in
// every round, a token slot's rows adjacent):
//   payload  hidden states in wire format        sentinel 0xFFFFFFFF
//   scales   block scales in wire format         sentinel 0xFFFFFFFF
//   weights  top-k routing weights (optional)    sentinel 0xFFFFFFFF
//   ids      top-k expert ids                    sentinel 0x88888888   two sets
//   combine  partial output of this rank's token from rank src, row src * maxTokensPerRank +
//            token (this step's tokens per rank)
// A source stores its ids row (src, token) on EVERY alive peer, all invalidExpertId where the
// peer is no target: the ids row tells that slot `token` arrived and whether it holds a token.
//
// Re-arming. A dispatch ends only after every alive peer's ids rows of the round arrived, so no
// rank starts round n + 2 before every peer finished round n. Payload, scale and weights rows go
// to targets only, and the sender's combine waits for every target's partial: a row is rewritten
// only after its reader re-armed it, so one set suffices. Ids rows also reach non-targets, which
// may still be in round n when the sender writes round n + 1, so they alternate two sets by an
// epoch in device memory (CUDA graphs replay it) that the last combine block advances; the
// combine re-arms them. A combine row is written in round n only by a rank that received the
// reader's round-n ids rows, sent after the reader's previous combine re-armed the row.
//
// Dispatch, grid maxTokensPerRank x blocksPerToken (a block: one token slot, one chunk of
// kNumThreads 16-byte vectors of the wire row, payload vectors first, then scale vectors):
//   D1 route     warp 0 maps the token's top-k ids to target ranks (each rank once, at its first
//                slot; masked ranks dropped) and stages the ids and weights rows; thread 0 reads
//                the epoch.
//   D2 push      each thread stores its vector into the landing row (epRank, token) of every
//                target; for this rank itself it goes straight into the receive buffer.
//   D3 ids       the chunk-0 block stores the ids row to every alive peer, the weights to targets,
//                its threads taking the (peer, vector) stores in turn.
//   D4 poll ids  every source's ids row of slot `token` -> recvSourceMask; warp 0 lists the
//                sources that sent a token.
//   D5 receive   poll, copy out and re-arm this thread's vector of every source that sent one,
//                8 sources of the slot's source list per round.
//   D6 weights   in the same rounds, the chunk-0 block does the same for the weights rows.
// Rank loops walk the set bits of a rank mask in ascending rank order (popLowestRank).
// Combine, same grid:
//   C1 sources   warp 0 lists the token's target ranks in top-k slot order.
//   C2 push      the expert output of row (src, token) goes to src's combine row (epRank, token).
//   C3 reduce    poll this rank's partials 8 sources at a time, re-arm them and add them in the
//                order of the fence combine (vectorized_combine_impl), then round to bf16.
//   C4 tail      the chunk-0 block re-arms the slot's ids rows; the last block advances the epoch.
// The only fence is the device-scope one before the combine's finished-block count.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::sentinel
{

static constexpr int kMaxRanks = 64; // the active-rank mask is one 64-bit word
static constexpr int kMaxTopK = 22;  // the NVLink one-sided kernels' bound
static constexpr int kNumThreads = 128;
// Resident dispatch blocks per SM (a register bound); the default token limit keeps the dispatch
// grid within one wave of them.
static constexpr int kDispatchBlocksPerSM = 6;
// Landing rows per source rank of a workspace, at most (bounds memory, not code).
static constexpr int kMaxTokensPerRank = 512;

// 16-byte vectors of an ids or weights row (topK 4-byte words, padded).
__host__ __device__ constexpr int topkRowVectors(int topK)
{
    return (topK + 3) / 4;
}

static constexpr uint32_t kIdsSentinel = 0x88888888U;
static constexpr uint32_t kDataSentinel = 0xFFFFFFFFU;
// NVFP4 data words travel XORed with this, so that the one word that could equal the sentinel
// after canonicalisation (eight -0 nibbles) cannot.
static constexpr uint32_t kNvfp4WireXor = 0x77777777U;

// Payload formats of the landing rows. The values are stored in the workspace metainfo and exported
// to Python (MOE_A2A_SENTINEL_FORMAT_*).
enum class PayloadFormat : int
{
    kBf16 = 0,
    kMxfp8 = 1, // e4m3 [T, H] + UE8M0 scales [T, H / 32] (16-bit lanes on the wire)
    // e4m3 [T, H] + fp32 scales [T, H / 128]; no quant algo maps to it yet (kept from FlashInfer, tested)
    kFp8Block128 = 2,
    kNvfp4 = 3, // e2m1 x2 [T, H / 2] + e4m3 scales [T, H / 16]
};

// Kernel families by the payload's element width (scales are handled at run time).
enum class PayloadKind : int
{
    kBf16 = 0,
    kFp8 = 1,
    kNvfp4 = 2,
};

// int32 words of the round state.
static constexpr int kEpoch = 0;
static constexpr int kFinishedCombineBlocks = 1;
static constexpr int kTimeoutReported = 2;

// Landing regions of every rank's workspace slice (byte offsets).
struct LandingLayout
{
    int64_t payloadOffset;
    int64_t scaleOffset;
    int64_t weightsOffset;
    int64_t idsOffset; // ids set 0; set 1 follows at + idsSetBytes
    int64_t idsSetBytes;
    int64_t combineOffset;
};

struct DispatchParams
{
    // Every rank's workspace slice as mapped here; rankWorkspaces[epRank] is this rank's.
    uint8_t* rankWorkspaces[kMaxRanks];
    LandingLayout landing;
    // This rank's tokens.
    uint8_t const* hiddenStates;         // [localNumTokens, payloadRowBytes]
    uint8_t const* hiddenStatesScale;    // [localNumTokens, scaleRowBytes]; null for BF16
    int32_t const* tokenSelectedExperts; // [localNumTokens, topK]
    uint32_t const* topkWeights;         // [localNumTokens, topK] 4-byte words, or bf16 (weightsBf16)
    // Receive buffers, row (src, token) = src * maxTokensPerRank + token.
    uint8_t* recvHiddenStates;
    uint8_t* recvHiddenStatesScale;
    int32_t* recvTopkIds;
    uint32_t* recvTopkWeights; // in the dtype of topkWeights
    // Round state shared with the combine.
    int32_t* topkTargetRanks; // [maxTokensPerRank, topK]: rank of slot k at its first
                              // occurrence in the token's top-k, else -1
    uint64_t* recvSourceMask; // [maxTokensPerRank]: bit s = rank s sent its token slot here
    int32_t* roundState;      // the int32 words kEpoch, kFinishedCombineBlocks, kTimeoutReported
    int epSize;
    int epRank;
    int numExperts;
    int expertsPerRank; // numExperts / epSize (an even split)
    int topK;
    int localNumTokens;
    int maxTokensPerRank; // of this step: the receive buffers' rows per source
    int hasTopkWeights;   // the step carries routing weights (every rank alike)
    int weightsBf16;      // the routing weights are bf16: each travels as the fp32 word of its value
    int mxScales;         // MXFP8: UE8M0 scale bytes travel as 16-bit lanes
    int payloadRowBytes;  // landing and receive
    int scaleRowBytes;    // input and receive scale row
    int wireScaleRowBytes;
    int payloadVectors;   // 16-byte vectors per payload row
    int wireScaleVectors;
    int blocksPerToken;
    int32_t invalidExpertId;
    uint64_t activeRankMask; // bit r: rank r takes part (all ranks: the ops take no rank mask)
    int64_t timeoutCycles;
};

struct CombineParams
{
    uint8_t* rankWorkspaces[kMaxRanks];
    LandingLayout landing;
    uint8_t const* expertOutput; // [epSize * maxTokensPerRank, hiddenSize] bf16, row (src, t)
    uint8_t* output;             // [localNumTokens, hiddenSize] bf16
    int32_t const* topkTargetRanks;
    uint64_t const* recvSourceMask;
    int32_t* roundState;
    int epSize;
    int epRank;
    int topK;
    int localNumTokens;
    int maxTokensPerRank;
    int hiddenSize;
    int rowVectors; // 16-byte vectors per bf16 row
    int blocksPerToken;
    uint64_t activeRankMask;
    int64_t timeoutCycles;
};

// Whether the sentinel kernels can run on the current device: it has compute capability 10.0, 10.3
// or 10.7 (where the transport is validated) and the code loaded for it carries the kernels (see
// MOE_A2A_SENTINEL_COMPILE in moeAlltoAllSentinel.cu).
bool moeA2ASentinelKernelsAvailable();

// Grid maxTokensPerRank * blocksPerToken; launched with PDL when TRTLLM_ENABLE_PDL is set.
// Only the all-ranks-active kernels are instantiated: the ops reject a rank mask on this transport.
void moeA2ASentinelDispatchLaunch(PayloadKind payloadKind, DispatchParams const& params, cudaStream_t stream);
void moeA2ASentinelCombineLaunch(CombineParams const& params, cudaStream_t stream);

} // namespace kernels::moe_comm::sentinel

TRTLLM_NAMESPACE_END
