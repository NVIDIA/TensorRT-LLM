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

#pragma once
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

// Configuration constants
static constexpr int kMaxTopK = 22;      // Maximum top-k experts per token
static constexpr int kMaxPayloads = 4;   // Maximum number of different payload types
static constexpr int kMaxRanks = 256;    // Maximum supported EP size (covers NVL576 EP256)
static constexpr int kRankMaskWords = 4; // uint64 words to hold the active-rank bitmask
                                         // (kRankMaskWords * 64 must be >= kMaxRanks)
static_assert(kRankMaskWords * 64 >= kMaxRanks, "active_rank_mask too small for kMaxRanks");

// Nominal clock64 rate used to convert timeout seconds to SM cycles.
static constexpr int64_t kAssumedClockHz = 2000ll * 1000ll * 1000ll;
static constexpr int64_t kDefaultTimeoutCycles = 300ll * kAssumedClockHz;

// Waits on a peer shorter than kAbortPollCycles never read the host abort state.
// Longer waits read it once per kAbortPollCycles per block.
static constexpr int64_t kAbortPollCycles = 100ll * 1000ll;

// A wait on a peer longer than kPublishWaitedFlagCycles publishes the flag value it waits for in
// MoeA2AAbortState::waited_flag, so the host can tell which peer this rank waits on.
static constexpr int64_t kPublishWaitedFlagCycles = 20ll * 1000ll * 1000ll;

//! \brief Abort state of the rank-mask collectives, in pinned host memory.
//!
//! The host sets `requested` to end the waits on peers. A kernel that sees it stops waiting and sets `status` to
//! kCancelled. While `requested` or `status` is nonzero, the outputs of every collective that uses the abort state
//! are invalid. The incomplete-round word below decides what the following collectives exchange.
struct MoeA2AAbortState
{
    static constexpr int32_t kCancelled = 1;

    int32_t requested;
    // kCancelled once a kernel stopped waiting because of `requested`, or the incomplete-round check found the word
    // set, 0 otherwise
    int32_t status;
    // Per peer, the round flag value that a long wait on the peer waits for, 0 otherwise
    uint32_t waited_flag[kMaxRanks];
};

// Incomplete-round word: a uint32_t in the workspace of each rank of a layout built with a rank mask. A collective
// that ends without the data of some active source stores its incomplete-round value (its round flag value with bit 0
// set, so never 0) there, unless the word is already nonzero. The word keeps that value until the workspace is reset.
// While it is nonzero, every later collective of this rank exchanges with this rank only, so no peer reads rows of an
// incomplete round. The MoE after a dispatch may still overwrite rows of this rank that an active peer's fence combine
// reads. So a fence dispatch that aborts, or that starts with the word set, first stores the value in the word of
// every active rank, and a check after each forward that finds its word set marks the abort state cancelled
// (moe_a2a_check_incomplete_round_launch). The incomplete-round value of round N's combine, which round N + 1's
// dispatch shares, is ignored within round N: this rank's combine or a peer's next dispatch may store it during the
// round, and the prepare, the sanitize and every block of the combine must exchange with the same peers.

// Describes a single payload type to be communicated
struct PayloadDescriptor
{
    void const* src_data;   // Source data pointer [local_num_tokens, elements_per_token]
    int element_size;       // Size of each element in bytes
    int elements_per_token; // Number of elements per token (e.g., hidden_size, top_k)
};

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

    // Local aux data pointers
    int* send_counters; // [ep_size] How many tokens have been sent to each target rank
    // Each rank owns recv_counters[parity][source_rank]. The two parity banks
    // alternate between A2A rounds.
    int* recv_counters[kMaxRanks];
    int* local_token_counter; // Atomic counter for completed tokens

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int* topk_target_ranks;   // target rank per k, -1 for invalid or duplicate routes
    int* topk_target_indices; // dst index per k, -1 for invalid or duplicate routes

    // Optional: Statistics for EPLB
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank

    // Active-rank bitmask: bit i set => rank i is alive and participates in this collective.
    // Word w covers ranks 64*w..64*w+63 for w in [0, kRankMaskWords). Tokens routed to a masked
    // rank are dropped (topk_*[k] = -1); flag writes/waits to/from masked peers are skipped.
    // The address is fixed, so captured CUDA graphs read a mask rewritten in place while no collective runs.
    uint64_t const* active_rank_mask{nullptr};
    // Byte offset from completion_flags[rank] to the incomplete-round word (see above) of that rank, or 0 without one.
    // Every rank's workspace has the same layout, so one offset locates the word of every rank.
    int64_t incomplete_round_offset{0};
    MoeA2AAbortState* abort{nullptr};
    // The rank-mask fields above fill kRankMaskWords words, so the parameter offsets of the fields and of the kernel
    // arguments after them do not depend on them.
    uint64_t rank_mask_reserved{0};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};
};

// Gather one contribution slice per expert rank.
struct CombineKernelPointers
{
    void* output;
    // Peer combine-input slices.
    uint8_t const* source_buffers[kMaxRanks];

    // Combine readiness flags.
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint32_t* flag_val;                    // The value of the flag for this round (stored on the local rank)

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int const* topk_target_ranks;   // target rank per k, -1 for invalid or duplicate routes
    int const* topk_target_indices; // dst index per k, -1 for invalid or duplicate routes

    // Active-rank bitmask: see DispatchKernelPointers::active_rank_mask. Combine skips
    // completion flag writes/waits to/from inactive peers.
    uint64_t const* active_rank_mask{nullptr};
    // Incomplete-round word in the local workspace (see above), or nullptr without one.
    uint32_t* incomplete_round{nullptr};
    MoeA2AAbortState* abort{nullptr};
    // The rank-mask fields above fill kRankMaskWords words, as in DispatchKernelPointers.
    uint64_t rank_mask_reserved{0};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};
};

// The kernels take these structs by value, followed by scalar arguments, so the rank-mask fields keep a fixed
// kRankMaskWords-word slot and timeout_cycles stays the last field. The parameter offsets of timeout_cycles and of
// the scalar arguments do not depend on the rank-mask fields, and neither does the generated code of the kernels
// without a rank mask.
static_assert(offsetof(DispatchKernelPointers, timeout_cycles)
    == offsetof(DispatchKernelPointers, active_rank_mask) + kRankMaskWords * sizeof(uint64_t));
static_assert(sizeof(DispatchKernelPointers) == offsetof(DispatchKernelPointers, timeout_cycles) + sizeof(int64_t));
static_assert(offsetof(CombineKernelPointers, timeout_cycles)
    == offsetof(CombineKernelPointers, active_rank_mask) + kRankMaskWords * sizeof(uint64_t));
static_assert(sizeof(CombineKernelPointers) == offsetof(CombineKernelPointers, timeout_cycles) + sizeof(int64_t));

// Dispatch phase parameters
struct MoeA2ADispatchParams
{
    // EP configuration
    int ep_size;     // Number of EP ranks
    int ep_rank;     // Current EP rank
    int num_experts; // Total number of experts

    // Token configuration
    int local_num_tokens;    // Number of tokens on this rank
    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation TODO: Rename to runtime_max_tokens_per_rank
    int top_k;               // Number of experts per token

    // Expert routing information
    int32_t const* token_selected_experts; // [local_num_tokens, top_k]

    // Generic payloads
    int num_payloads;                         // Number of different payload types
    PayloadDescriptor payloads[kMaxPayloads]; // Array of payload descriptors

    // Local aux data
    uint32_t* flag_val;       // The value of the flag for this round (stored on the local rank)
    int* local_token_counter; // Atomic counter for completed tokens on this rank
    int* send_counters;       // [ep_size] atomic counters - tracks tokens sent to each target rank
    int* topk_target_ranks; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), target rank
                            // per k, -1 for duplicates
    int* topk_target_indices; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), dst index
                              // per k, -1 for duplicates

    // Distributed aux data and recv buffers
    // Each rank owns recv_counters[parity][source_rank]. The two parity banks
    // alternate between A2A rounds.
    int* recv_counters[kMaxRanks];
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    // Byte offset from completion_flags[rank] to the incomplete-round word (see above) of that rank, or 0 without one.
    int64_t incomplete_round_offset{0};
    void* recv_buffers[kMaxRanks][kMaxPayloads]; // Per-rank receive buffers for each payload

    // Optional: Statistics for EPLB
    bool enable_eplb;                    // Whether to enable EPLB
    int eplb_stats_num_experts;          // Number of experts for EPLB stats
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank

    // Whether to instantiate a kernel with active-rank checks. An abort state requires it.
    bool enable_rank_mask{false};

    // Active-rank bitmask: see DispatchKernelPointers::active_rank_mask. Required when enable_rank_mask is true.
    uint64_t const* active_rank_mask{nullptr};
    MoeA2AAbortState* abort{nullptr};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};

    // CUDA stream
    cudaStream_t stream;
};

// Dispatch kernels
void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params);
// Prepare for dispatch: zero send_counters, local_token_counter and increment flag_val
void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params);

// Combine phase parameters
struct MoeA2ACombineParams
{
    // EP configuration
    int ep_size; // Number of EP ranks
    int ep_rank; // Current EP rank

    // Token configuration
    int local_num_tokens;    // Number of tokens on this rank
    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation TODO: Rename to runtime_max_tokens_per_rank
    int top_k;               // Number of experts per token

    // Resolved payload plan. The source always points to the MoE output tensor;
    // the host determines all storage strides before launching any kernel.
    void const* source_payload;
    int source_stride_per_token;
    int workspace_stride_per_token; // prepare writes and combine reads the combine input at this stride
    int prepare_num_tokens;

    // Output tensor
    void* output_data; // Output buffer [local_num_tokens, elements_per_token]
    // Payload information
    int elements_per_token;       // Number of elements per token
    tensorrt_llm::DataType dtype; // Data type of the payload (used for combine kernel dispatch)
    bool
        use_low_precision; // If true, prepare kernel quantizes payload→FP8; combine kernel accumulates FP8→output dtype

    // Local aux data
    uint32_t* flag_val;     // The value of the flag for this round (stored on the local rank)
    int* topk_target_ranks; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), target rank
                            // per k, -1 for duplicates
    int* topk_target_indices; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), dst index
                              // per k, -1 for duplicates
    // Incomplete-round word in the local workspace (see above), or nullptr without one.
    uint32_t* incomplete_round{nullptr};
    // Local recv_counters[parity][source_rank]. The two parity banks alternate
    // between A2A rounds.
    int const* recv_counters;

    // Distributed aux data and recv buffers
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint8_t* combine_input_buffers[kMaxRanks]; // Expert-output/staging region on each rank

    // Whether to instantiate a kernel with active-rank checks in peer synchronization. An abort state requires it.
    bool enable_rank_mask{false};

    // Active-rank bitmask: see DispatchKernelPointers::active_rank_mask. Required when enable_rank_mask is true.
    uint64_t const* active_rank_mask{nullptr};
    MoeA2AAbortState* abort{nullptr};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};

    // CUDA stream
    cudaStream_t stream;
};

// Combine kernels
void moe_a2a_combine_launch(MoeA2ACombineParams const& params);

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params);

// Mark `abort` cancelled if the incomplete-round word (see above) is set, ordered after every earlier fence combine in
// `stream`. Launch once per forward, after its last combine.
void moe_a2a_check_incomplete_round_launch(
    uint32_t const* incomplete_round, MoeA2AAbortState* abort, cudaStream_t stream);

// Sanitize expert IDs for invalid tokens
// expert_ids: [ep_size, max_tokens_per_rank, top_k] (int32)
// recv_counters: [2, ep_size] (int32), number of valid tokens per source
// incomplete_round: incomplete-round word (see above), or nullptr. While it is nonzero, only the tokens from ep_rank
// are valid.
// invalid_id: value to fill for invalid tokens' expert ids
void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, uint32_t const* flag_val,
    uint32_t const* incomplete_round, int32_t invalid_id, int ep_rank, int ep_size, int max_tokens_per_rank, int top_k,
    cudaStream_t stream);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
