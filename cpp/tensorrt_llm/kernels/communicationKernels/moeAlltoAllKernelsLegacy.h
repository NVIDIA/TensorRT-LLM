/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

// The previous MoE all-to-all kernels, kept beside the current ones (moeAlltoAllKernels.cu) so one
// build can A/B them at runtime: moeAlltoAllOp.cpp launches these unless TLLM_MOE_A2A_FUSED_ROUND=1. This header is
// the previous moeAlltoAllKernels.h in namespace moe_comm::legacy, except that the public types below
// (CftPeerLeIds, PayloadDescriptor, MoeA2ADispatchParams, MoeA2ACombineParams), which both versions define
// identically, come from moeAlltoAllKernels.h, so the op fills one parameter struct for either version.
// Do not change the legacy kernels: a fix or an optimization belongs in moeAlltoAllKernels.cu.

#pragma once
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm::legacy
{

// Configuration constants
static constexpr int kMaxTopK = 22;      // Maximum top-k experts per token
static constexpr int kMaxPayloads = 4;   // Maximum number of different payload types
static constexpr int kMaxRanks = 128;    // Maximum supported EP size (covers NVL72 with headroom)
static constexpr int kRankMaskWords = 2; // uint64 words to hold the active-rank bitmask
                                         // (kRankMaskWords * 64 must be >= kMaxRanks)
static_assert(kRankMaskWords * 64 >= kMaxRanks, "active_rank_mask too small for kMaxRanks");

// CFT counted write counter stride: 256B per counter to avoid L2 XBAR port camping
// when concurrent counters update. See NCCL CFT perf study (CST tracking structure).
static constexpr size_t kCftCounterStride = 256;
static constexpr size_t kCftCounterStrideU64 = kCftCounterStride / sizeof(uint64_t);

// Smem slot reserved for an mbarrier in the CFT dispatch / push-combine kernels.
// The mbarrier itself is only 8 B (a uint64_t initialized via mbarrier.init.shared.b64);
// the 64 B reservation is padding so the staging buffer that immediately follows is
// 16 B-aligned (cp.async.bulk requires 16 B-aligned source AND destination).
static constexpr int kCftMbarrierSlotBytes = 64;

// Public types shared with moeAlltoAllKernels.h (defined identically in the previous version).
using moe_comm::CftPeerLeIds;
using moe_comm::PayloadDescriptor;
using moe_comm::MoeA2ADispatchParams;
using moe_comm::MoeA2ACombineParams;
// Those structs size their arrays with moe_comm's constants, and the op lays out the workspace with them.
static_assert(kMaxTopK == moe_comm::kMaxTopK && kMaxPayloads == moe_comm::kMaxPayloads
        && kMaxRanks == moe_comm::kMaxRanks && kRankMaskWords == moe_comm::kRankMaskWords,
    "legacy MoE A2A limits must equal moe_comm's");
static_assert(
    kCftCounterStride == moe_comm::kCftCounterStride && kCftMbarrierSlotBytes == moe_comm::kCftMbarrierSlotBytes,
    "legacy CFT layout constants must equal moe_comm's");

// Kernel pointers packed into a struct for device access
// Dispatch kernel pointers - const source data
struct DispatchKernelPointers
{
    // Payload pointers
    void const* src_data_ptrs[kMaxPayloads];     // Array of source data pointers
    void* recv_buffers[kMaxRanks][kMaxPayloads]; // 2D array of receive buffer pointers
    int payload_bytes_per_token[kMaxPayloads];   // Bytes per token for each payload
    // Completion flags for synchronization (fence-based path)
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint32_t* flag_val;                    // The value of the flag for this round (stored on the local rank)

    // LE dispatch counters: HW-incremented byte counters for dispatch data.
    // le_dispatch_counters[target_rank][source_rank] is incremented by fabric engine
    // by the number of bytes written from source_rank to target_rank.
    uint64_t* le_dispatch_counters[kMaxRanks];

    // Local aux data pointers
    int* send_counters; // [ep_size] How many tokens have been sent to each target rank
    // Each rank owns recv_counters[parity][source_rank]. The two parity banks
    // alternate between A2A rounds.
    int* recv_counters[kMaxRanks];
    int* local_token_counter; // Atomic counter for completed tokens

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int* topk_target_ranks; // target rank per k, -1 for invalid or duplicate routes
    int* topk_send_indices; // dst index per k, -1 for invalid or duplicate routes

    // Optional: Statistics for EPLB
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank

    // CFT handle-based counted writes (fabric.try_put.counted via Logical Endpoints)
    uint32_t peer_le_ids[kMaxRanks];           // LE ID per target rank (from CftLeManager)
    uint64_t le_payload_offsets[kMaxPayloads]; // Byte offset of each payload's recv region within LE
    uint64_t le_counter_base;                  // Base byte offset for per-source-rank 8B counters within LE

    // Cumulative counter baseline for dispatch data counters (regular device memory, not LE-backed).
    // LE counters grow monotonically; the baseline tracks the previous cumulative value.
    uint64_t* dispatch_counter_baseline; // [ep_size]

    // Optional CFT dispatch-side expert-id sanitization.  Invalid padding tokens in the received
    // expert-id payload are filled after recv_counters are known, avoiding a separate sanitize kernel.
    bool sanitize_expert_ids;
    int expert_id_payload_index;
    int32_t invalid_expert_id;

    // Active-rank bitmask: bit i set => rank i is alive and participates in this collective.
    // Word 0 covers ranks 0..63; word 1 covers ranks 64..127. Tokens routed to a masked
    // rank are dropped (topk_*[k] = -1); flag writes/waits to/from masked peers are skipped.
    // The local rank's own bit must always be set; this is checked at launch time.
    uint64_t active_rank_mask[kRankMaskWords];
};

// Combine kernel pointers - non-const output in src_data_ptrs[0], const recv buffers
struct CombineKernelPointers
{
    // Payload pointers
    void* src_data_ptrs[kMaxPayloads];                 // src_data_ptrs[0] is output
    void const* recv_buffers[kMaxRanks][kMaxPayloads]; // 2D array of receive buffer pointers (const)

    // Completion flags for synchronization (fence-based path)
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint32_t* flag_val;                    // The value of the flag for this round (stored on the local rank)

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int const* topk_target_ranks; // target rank per k, -1 for invalid or duplicate routes
    int const* topk_send_indices; // dst index per k, -1 for invalid or duplicate routes

    // ---- CFT combine (counted-write) fields. Unused by the fence combine path. ----
    // Local LE combine counters: per receive-slot HW-incremented byte counters.
    uint64_t* combine_counters;
    // Cumulative per-slot counter baselines (regular device memory, single-buffer, advanced in place by reduce).
    uint64_t* combine_counter_baseline; // [ep_size * max_tokens_per_rank]
    int combine_counter_ep_stride = 0;  // STABLE static stride (maxNumTokens) for counter/baseline slot indexing
    // Active-rank bitmask: see DispatchKernelPointers::active_rank_mask. Combine skips
    // completion flag writes/waits to/from inactive peers.
    uint64_t active_rank_mask[kRankMaskWords];
};

// Dispatch kernels
void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params);
// Prepare for dispatch: zero send_counters, local_token_counter and increment flag_val
void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params);

// Combine kernels
void moe_a2a_combine_launch(MoeA2ACombineParams const& params);

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params);

// CFT combine push: processing rank pushes expert output back to originating rank's LE.
void moe_a2a_cft_combine_push_launch(MoeA2ACombineParams const& params);

// Sanitize expert IDs for invalid tokens
// expert_ids: [ep_size, max_tokens_per_rank, top_k] (int32)
// recv_counters: [2, ep_size] (int32), number of valid tokens per source
// invalid_id: value to fill for invalid tokens' expert ids
void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, uint32_t const* flag_val,
    int32_t invalid_id, int ep_size, int max_tokens_per_rank, int top_k, cudaStream_t stream);

} // namespace kernels::moe_comm::legacy

TRTLLM_NAMESPACE_END
