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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllSentinel.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace moe_comm
{

// Per-rank layout: round state, dispatch control/payload, combine control/input, then the sentinel regions.
// Fence combine pulls from peer input buffers.
// Region boundaries are fixed at allocation; token slices remain runtime-packed.
// OFFSET fields are byte offsets from the per-rank workspace base; SIZE fields are byte capacities.
// Array extents below use allocation-time max_tokens, top_k and ep_size, not runtime token counts.
enum MoeA2AMetaInfoIndex : int64_t
{
    // Shared round state.
    // uint32_t scalar advanced by dispatch/combine prepare; supplies sync epochs and round parity.
    FLAG_VAL_OFFSET_INDEX = 0,

    // Dispatch control, routing metadata and receive payload.
    // int32_t scalar counting token CTAs to elect the last CTA that publishes dispatch counts.
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = 1,
    // int32_t[ep_size]: outgoing token counts / slot allocators, indexed by destination rank.
    SEND_COUNTERS_OFFSET_INDEX = 2,
    // int32_t[2][ep_size]: incoming counts by round parity and sender; dispatch writes, combine reads.
    RECV_COUNTERS_OFFSET_INDEX = 3,
    // uint32_t[ep_size]: per-peer epoch flags for fence dispatch synchronization.
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = 4,
    // int32_t[max_tokens][top_k]: destination ranks for local tokens, reused by combine to gather results.
    TOPK_TARGET_RANKS_OFFSET_INDEX = 5,
    // int32_t[max_tokens][top_k]: matching receive-slot indices within each destination's sender slice.
    TOPK_TARGET_INDICES_OFFSET_INDEX = 6,
    // int32_t[ep_size][eplb_stats_num_experts]: gathered per-rank expert statistics; empty without EPLB.
    EPLB_GATHERED_STATS_OFFSET_INDEX = 7,
    // Receive region for dispatched activations, scales, expert IDs/weights and optional extra payloads.
    DISPATCH_PAYLOAD_OFFSET_INDEX = 8,
    // Reserved dispatch payload capacity, including payload alignment padding.
    DISPATCH_PAYLOAD_SIZE_INDEX = 9,

    // Combine control and expert-output staging.
    // uint32_t[ep_size]: peer readiness epochs for fence combine synchronization.
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = 10,
    // Expert-output / staging region: fence peers pull from it; MoE may write directly into it.
    COMBINE_INPUT_OFFSET_INDEX = 11,
    // Reserved combine input capacity, large enough for original-dtype MoE output even with FP8 wire data.
    COMBINE_INPUT_SIZE_INDEX = 12,

    // Allocation-time configuration used to construct and validate the layout, not mutable round state.
    // Maximum input token count per rank; bounds routing storage and per-rank token slots.
    MAX_NUM_TOKENS_INDEX = 13,
    // Configured experts selected per token; determines routing-table capacity.
    TOP_K_INDEX = 14,
    // Number of ranks in the EP group; determines peer-array sizes.
    EP_SIZE_INDEX = 15,
    // Number of expert statistics gathered from each rank; zero disables EPLB storage.
    EPLB_STATS_NUM_EXPERTS_INDEX = 16,
    // Total required bytes per rank, including all control buffers, payloads and alignment padding.
    WORKSPACE_SIZE_INDEX = 17,

    // Sentinel transport (moeAlltoAllSentinel.h), appended after the regions above by
    // moe_a2a_sentinel_extend_workspace_layout. Every field is zero when it is not allocated.
    // Landing rows per source rank: the largest runtime_max_tokens_per_rank of a sentinel step.
    SENTINEL_MAX_NUM_TOKENS_INDEX = 18,
    // Payload format (kernels::moe_comm::sentinel::PayloadFormat) and hidden size the landing rows hold.
    SENTINEL_PAYLOAD_FORMAT_INDEX = 19,
    SENTINEL_HIDDEN_SIZE_INDEX = 20,
    // Round state: int32 words (epoch, ...), then a uint64 receive-source mask per token slot.
    SENTINEL_STATE_OFFSET_INDEX = 21,
    // Landing rows: payload, scale and weight rows, then two sets of expert-id rows.
    SENTINEL_LANDING_OFFSET_INDEX = 22,
    // Combine landing rows: one bf16 partial row per (source rank, token).
    SENTINEL_COMBINE_LANDING_OFFSET_INDEX = 23,
    // Receive buffers of a sentinel dispatch, packed like the dispatch payload region.
    SENTINEL_RECV_OFFSET_INDEX = 24,
    // Reserved receive capacity, including payload alignment padding.
    SENTINEL_RECV_SIZE_INDEX = 25,

    // Number of int64_t entries in the metadata tensor; not a stored layout field.
    NUM_METAINFO_FIELDS = 26
};

using MoeA2AWorkspaceLayout = std::array<int64_t, NUM_METAINFO_FIELDS>;
static constexpr int64_t kWorkspaceAlignment = 256;

inline std::vector<std::pair<char const*, int64_t>> getMoeA2AMetaInfoIndexPairs()
{
    using namespace tensorrt_llm::kernels::moe_comm;
    return {
        {"MOE_A2A_FLAG_VAL_OFFSET_INDEX", FLAG_VAL_OFFSET_INDEX},
        {"MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX", LOCAL_TOKEN_COUNTER_OFFSET_INDEX},
        {"MOE_A2A_SEND_COUNTERS_OFFSET_INDEX", SEND_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_RECV_COUNTERS_OFFSET_INDEX", RECV_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX", DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX},
        {"MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX", TOPK_TARGET_RANKS_OFFSET_INDEX},
        {"MOE_A2A_TOPK_TARGET_INDICES_OFFSET_INDEX", TOPK_TARGET_INDICES_OFFSET_INDEX},
        {"MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX", EPLB_GATHERED_STATS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_PAYLOAD_OFFSET_INDEX", DISPATCH_PAYLOAD_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_PAYLOAD_SIZE_INDEX", DISPATCH_PAYLOAD_SIZE_INDEX},
        {"MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX", COMBINE_COMPLETION_FLAGS_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_INPUT_OFFSET_INDEX", COMBINE_INPUT_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_INPUT_SIZE_INDEX", COMBINE_INPUT_SIZE_INDEX},
        {"MOE_A2A_MAX_NUM_TOKENS_INDEX", MAX_NUM_TOKENS_INDEX},
        {"MOE_A2A_TOP_K_INDEX", TOP_K_INDEX},
        {"MOE_A2A_EP_SIZE_INDEX", EP_SIZE_INDEX},
        {"MOE_A2A_EPLB_STATS_NUM_EXPERTS_INDEX", EPLB_STATS_NUM_EXPERTS_INDEX},
        {"MOE_A2A_WORKSPACE_SIZE_INDEX", WORKSPACE_SIZE_INDEX},
        {"MOE_A2A_SENTINEL_MAX_NUM_TOKENS_INDEX", SENTINEL_MAX_NUM_TOKENS_INDEX},
        {"MOE_A2A_SENTINEL_PAYLOAD_FORMAT_INDEX", SENTINEL_PAYLOAD_FORMAT_INDEX},
        {"MOE_A2A_SENTINEL_HIDDEN_SIZE_INDEX", SENTINEL_HIDDEN_SIZE_INDEX},
        {"MOE_A2A_SENTINEL_STATE_OFFSET_INDEX", SENTINEL_STATE_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_LANDING_OFFSET_INDEX", SENTINEL_LANDING_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_COMBINE_LANDING_OFFSET_INDEX", SENTINEL_COMBINE_LANDING_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_RECV_OFFSET_INDEX", SENTINEL_RECV_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_RECV_SIZE_INDEX", SENTINEL_RECV_SIZE_INDEX},
        {"MOE_A2A_NUM_METAINFO_FIELDS", NUM_METAINFO_FIELDS},
        {"MOE_A2A_MAX_RANKS", kMaxRanks},
        {"MOE_A2A_MAX_TOP_K", kMaxTopK},
        {"MOE_A2A_MAX_PAYLOADS", kMaxPayloads},
        {"MOE_A2A_WORKSPACE_ALIGNMENT", kWorkspaceAlignment},
        // Limits of the sentinel transport and its payload formats.
        {"MOE_A2A_SENTINEL_MAX_RANKS", sentinel::kMaxRanks},
        {"MOE_A2A_SENTINEL_MAX_TOKENS_PER_RANK", sentinel::kMaxTokensPerRank},
        {"MOE_A2A_SENTINEL_DISPATCH_BLOCKS_PER_SM", sentinel::kDispatchBlocksPerSM},
        {"MOE_A2A_SENTINEL_FORMAT_BF16", sentinel::kBf16},
        {"MOE_A2A_SENTINEL_FORMAT_MXFP8", sentinel::kMxfp8},
        {"MOE_A2A_SENTINEL_FORMAT_FP8_BLOCK128", sentinel::kFp8Block128},
        {"MOE_A2A_SENTINEL_FORMAT_NVFP4", sentinel::kNvfp4},
    };
}

} // namespace moe_comm
} // namespace torch_ext

TRTLLM_NAMESPACE_END
