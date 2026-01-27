/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

// Configuration constants
static constexpr int kMaxTopK = 22;    // Maximum top-k experts per token
static constexpr int kMaxPayloads = 4; // Maximum number of different payload types
static constexpr int kMaxRanks = 64;   // Maximum supported EP size

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

    // Completion flags for synchronization
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint32_t* flag_val;                    // The value of the flag for this round (stored on the local rank)

    // Local aux data pointers
    int* send_counters;            // [ep_size] How many tokens have been sent to each target rank
    int* recv_counters[kMaxRanks]; // How many tokens have been received from each source rank. Each rank has [ep_size]
                                   // counters
    int* local_token_counter;      // Atomic counter for completed tokens

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int* topk_target_ranks; // target rank per k, -1 for duplicates
    int* topk_send_indices; // dst index per k, -1 for duplicates

    // Optional: Statistics for EPLB
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank
};

// Combine kernel pointers - non-const output in src_data_ptrs[0], const recv buffers
struct CombineKernelPointers
{
    // Payload pointers
    void* src_data_ptrs[kMaxPayloads];                 // src_data_ptrs[0] is output
    void const* recv_buffers[kMaxRanks][kMaxPayloads]; // 2D array of receive buffer pointers (const)

    // Completion flags for synchronization
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint32_t* flag_val;                    // The value of the flag for this round (stored on the local rank)

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int const* topk_target_ranks; // target rank per k, -1 for duplicates
    int const* topk_send_indices; // dst index per k, -1 for duplicates
};

// Dispatch phase parameters
struct MoeA2ADispatchParams
{
    // Threading policy
    bool one_block_per_token; // True: one block per token, False: one warp per token

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
    int* topk_send_indices; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), dst index
                            // per k, -1 for duplicates

    // Distributed aux data and recv buffers
    int* recv_counters[kMaxRanks]; // tracks tokens received from each source rank. Each rank has [ep_size] counters
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    void* recv_buffers[kMaxRanks][kMaxPayloads]; // Per-rank receive buffers for each payload

    // Optional: Statistics for EPLB
    bool enable_eplb;                    // Whether to enable EPLB
    int eplb_stats_num_experts;          // Number of experts for EPLB stats
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank

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
    bool one_block_per_token; // True: one block per token, False: one warp per token

    // EP configuration
    int ep_size; // Number of EP ranks
    int ep_rank; // Current EP rank

    // Token configuration
    int local_num_tokens;    // Number of tokens on this rank
    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation TODO: Rename to runtime_max_tokens_per_rank
    int top_k;               // Number of experts per token

    // Prepare-only field: original payload tensor pointer used to stage into workspace
    void const* prepare_payload;

    // Output tensor
    void* output_data; // Output buffer [local_num_tokens, elements_per_token]
    // Payload information
    int elements_per_token;   // Number of elements per token
    nvinfer1::DataType dtype; // Data type for proper summation

    // Local aux data
    uint32_t* flag_val;     // The value of the flag for this round (stored on the local rank)
    int* topk_target_ranks; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), target rank
                            // per k, -1 for duplicates
    int* topk_send_indices; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), dst index
                            // per k, -1 for duplicates
    int const* recv_counters; // [ep_size] number of valid tokens per source rank for this target

    // Distributed aux data and recv buffers
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    void const* recv_buffers[kMaxRanks];   // Per-rank receive buffers (only for single payload)

    // CUDA stream
    cudaStream_t stream;
};

// Combine kernels
void moe_a2a_combine_launch(MoeA2ACombineParams const& params);

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params);

// Sanitize expert IDs for invalid tokens
// expert_ids: [ep_size, max_tokens_per_rank, top_k] (int32)
// recv_counters: [ep_size] (int32), number of valid tokens per source
// invalid_id: value to fill for invalid tokens' expert ids
void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, int32_t invalid_id,
    int ep_size, int max_tokens_per_rank, int top_k, cudaStream_t stream);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
