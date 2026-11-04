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
    // The local rank's own bit must always be set; this is checked at launch time.
    uint64_t active_rank_mask[kRankMaskWords];

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
    uint64_t active_rank_mask[kRankMaskWords];

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};
};

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
    void* recv_buffers[kMaxRanks][kMaxPayloads]; // Per-rank receive buffers for each payload

    // Optional: Statistics for EPLB
    bool enable_eplb;                    // Whether to enable EPLB
    int eplb_stats_num_experts;          // Number of experts for EPLB stats
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank

    // Whether to instantiate a kernel with active-rank checks.
    // This is a launch-lifetime mode, independent of future execution-abort handling.
    bool enable_rank_mask{false};

    // Active-rank bitmask: see DispatchKernelPointers::active_rank_mask. Used only when
    // enable_rank_mask is true; defaults to all-ones for backwards-compatible behavior.
    // The mask is copied by value into kernel arguments. Rank-mask mode must reject
    // CUDA graph replay until generation-scoped invalidation and recapture are available.
    uint64_t active_rank_mask[kRankMaskWords] = {~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}};

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
    int workspace_stride_per_token;
    int wire_bytes_per_token;
    int prepare_first_token;
    int prepare_num_tokens;
    int reduce_stride_per_token;

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
    // Local recv_counters[parity][source_rank]. The two parity banks alternate
    // between A2A rounds.
    int const* recv_counters;

    // Distributed aux data and recv buffers
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint8_t* combine_input_buffers[kMaxRanks]; // Expert-output/staging region on each rank

    // Whether to instantiate a kernel with active-rank checks in peer synchronization.
    // This is a launch-lifetime mode, independent of future execution-abort handling.
    bool enable_rank_mask{false};

    // Active-rank bitmask: see DispatchKernelPointers::active_rank_mask. Used only when
    // enable_rank_mask is true; defaults to all-ones for backwards-compatible behavior.
    // The mask is copied by value into kernel arguments. Rank-mask mode must reject
    // CUDA graph replay until generation-scoped invalidation and recapture are available.
    uint64_t active_rank_mask[kRankMaskWords] = {~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}, ~uint64_t{0}};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};

    // CUDA stream
    cudaStream_t stream;
};

// Combine kernels
void moe_a2a_combine_launch(MoeA2ACombineParams const& params);

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params);

// Sanitize expert IDs for invalid tokens
// expert_ids: [ep_size, max_tokens_per_rank, top_k] (int32)
// recv_counters: [2, ep_size] (int32), number of valid tokens per source
// invalid_id: value to fill for invalid tokens' expert ids
void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, uint32_t const* flag_val,
    int32_t invalid_id, int ep_size, int max_tokens_per_rank, int top_k, cudaStream_t stream);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
