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
// Landing rows of a rank's workspace, row (src, token) = token * ep_size + src (the same rows in
// every round, a token slot's rows adjacent):
//   payload  hidden states in wire format        sentinel 0xFFFFFFFF
//   scales   block scales in wire format         sentinel 0xFFFFFFFF
//   weights  top-k routing weights (optional)    sentinel 0xFFFFFFFF
//   ids      top-k expert ids                    sentinel 0x88888888   two sets
//   combine  partial output of this rank's token from rank src, row src * max_tokens_per_rank +
//            token (this step's tokens per rank)
// A source stores its ids row (src, token) on EVERY alive peer, all invalid_expert_id where the
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
// Dispatch, grid max_tokens_per_rank x blocks_per_token (a block: one token slot, one chunk of
// kNumThreads 16-byte vectors of the wire row, payload vectors first, then scale vectors):
//   D1 route     warp 0 maps the token's top-k ids to target ranks (each rank once, at its first
//                slot; masked ranks dropped) and stages the ids and weights rows; thread 0 reads
//                the epoch.
//   D2 push      each thread stores its vector into the landing row (ep_rank, token) of every
//                target; for this rank itself it goes straight into the receive buffer.
//   D3 ids       the chunk-0 block stores the ids row to every alive peer, the weights to targets,
//                its threads taking the (peer, vector) stores in turn.
//   D4 poll ids  every source's ids row of slot `token` -> recv_source_mask; warp 0 lists the
//                sources that sent a token.
//   D5 receive   poll, copy out and re-arm this thread's vector of every source that sent one,
//                8 sources of the slot's source list per round.
//   D6 weights   in the same rounds, the chunk-0 block does the same for the weights rows.
// Rank loops walk the set bits of a rank mask in ascending rank order (pop_lowest_rank).
// Combine, same grid:
//   C1 sources   warp 0 lists the token's target ranks in top-k slot order.
//   C2 push      the expert output of row (src, token) goes to src's combine row (ep_rank, token).
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

// 16-byte vectors of an ids or weights row (top_k 4-byte words, padded).
__host__ __device__ constexpr int topk_row_vectors(int top_k)
{
    return (top_k + 3) / 4;
}

static constexpr uint32_t kIdsSentinel = 0x88888888u;
static constexpr uint32_t kDataSentinel = 0xFFFFFFFFu;
// NVFP4 data words travel XORed with this, so that the one word that could equal the sentinel
// after canonicalisation (eight -0 nibbles) cannot.
static constexpr uint32_t kNvfp4WireXor = 0x77777777u;

// Payload formats of the landing rows.
enum PayloadFormat : int
{
    kBf16 = 0,
    kMxfp8 = 1,       // e4m3 [T, H] + UE8M0 scales [T, H / 32] (16-bit lanes on the wire)
    kFp8Block128 = 2, // e4m3 [T, H] + fp32 scales [T, H / 128]
    kNvfp4 = 3,       // e2m1 x2 [T, H / 2] + e4m3 scales [T, H / 16]
    kNumPayloadFormats
};

// Kernel families by the payload's element width (scales are handled at run time).
enum PayloadKind : int
{
    kKindBf16 = 0,
    kKindFp8 = 1,
    kKindNvfp4 = 2
};

// int32 words of the round state.
enum RoundStateWord : int
{
    kEpoch = 0,
    kFinishedCombineBlocks = 1,
    kTimeoutReported = 2,
    kRoundStateWords = 4
};

// Landing regions of every rank's workspace slice (byte offsets).
struct LandingLayout
{
    int64_t payload_offset;
    int64_t scale_offset;
    int64_t weights_offset;
    int64_t ids_offset; // ids set 0; set 1 follows at + ids_set_bytes
    int64_t ids_set_bytes;
    int64_t combine_offset;
};

struct DispatchParams
{
    // Every rank's workspace slice as mapped here; rank_workspaces[ep_rank] is this rank's.
    uint8_t* rank_workspaces[kMaxRanks];
    LandingLayout landing;
    // This rank's tokens.
    uint8_t const* hidden_states;          // [local_num_tokens, payload_row_bytes]
    uint8_t const* hidden_states_scale;    // [local_num_tokens, scale_row_bytes]; null for BF16
    int32_t const* token_selected_experts; // [local_num_tokens, top_k]
    uint32_t const* topk_weights;          // [local_num_tokens, top_k] 4-byte words
    // Receive buffers, row (src, token) = src * max_tokens_per_rank + token.
    uint8_t* recv_hidden_states;
    uint8_t* recv_hidden_states_scale;
    int32_t* recv_topk_ids;
    uint32_t* recv_topk_weights;
    // Round state shared with the combine.
    int32_t* topk_target_ranks; // [max_tokens_per_rank, top_k]: rank of slot k at its first
                                // occurrence in the token's top-k, else -1
    uint64_t* recv_source_mask; // [max_tokens_per_rank]: bit s = rank s sent its token slot here
    int32_t* round_state;       // RoundStateWord
    int ep_size;
    int ep_rank;
    int num_experts;
    int experts_per_rank; // num_experts / ep_size (an even split)
    int top_k;
    int local_num_tokens;
    int max_tokens_per_rank; // of this step: the receive buffers' rows per source
    int has_topk_weights;    // the step carries routing weights (every rank alike)
    int mx_scales;           // MXFP8: UE8M0 scale bytes travel as 16-bit lanes
    int payload_row_bytes;   // landing and receive
    int scale_row_bytes;     // input and receive scale row
    int wire_scale_row_bytes;
    int payload_vectors;     // 16-byte vectors per payload row
    int wire_scale_vectors;
    int blocks_per_token;
    int32_t invalid_expert_id;
    uint64_t active_rank_mask; // bit r: rank r takes part (rank-mask kernels only)
    int64_t timeout_cycles;
};

struct CombineParams
{
    uint8_t* rank_workspaces[kMaxRanks];
    LandingLayout landing;
    uint8_t const* expert_output; // [ep_size * max_tokens_per_rank, hidden_size] bf16, row (src, t)
    uint8_t* output;              // [local_num_tokens, hidden_size] bf16
    int32_t const* topk_target_ranks;
    uint64_t const* recv_source_mask;
    int32_t* round_state;
    int ep_size;
    int ep_rank;
    int top_k;
    int local_num_tokens;
    int max_tokens_per_rank;
    int hidden_size;
    int row_vectors; // 16-byte vectors per bf16 row
    int blocks_per_token;
    uint64_t active_rank_mask;
    int64_t timeout_cycles;
};

// Whether the code loaded for the current device carries the kernels. They are built for compute
// capability 10.0, 10.3 and 10.7; other architectures get kernels that trap.
bool moe_a2a_sentinel_kernels_available();

// Grid max_tokens_per_rank * blocks_per_token; launched with PDL when TRTLLM_ENABLE_PDL is set.
void moe_a2a_sentinel_dispatch_launch(
    int payload_kind, DispatchParams const& params, bool enable_rank_mask, cudaStream_t stream);
void moe_a2a_sentinel_combine_launch(CombineParams const& params, bool enable_rank_mask, cudaStream_t stream);

} // namespace kernels::moe_comm::sentinel

TRTLLM_NAMESPACE_END
