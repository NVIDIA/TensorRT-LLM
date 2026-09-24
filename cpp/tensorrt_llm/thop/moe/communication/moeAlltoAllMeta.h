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
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace moe_comm
{

// Per-rank layout: round state, dispatch control/payload, combine control/input/receive.
// Fence combine pulls from peer input buffers; CFT pushes into the local receive inbox.
// Region boundaries are fixed at allocation; token slices remain runtime-packed.
// OFFSET fields are byte offsets from the per-rank workspace base; SIZE fields are byte capacities.
// Array extents below use allocation-time max_tokens, top_k and ep_size, not runtime token counts.
// CFT-only offsets and sizes are zero when CFT storage is not allocated.
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
    // CFT-only cumulative received-byte counters per sender; uint64_t at kCftCounterStride byte spacing.
    DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX = 5,
    // CFT-only uint64_t[ep_size]: consumed-byte baselines for dispatch counters; ordinary device memory.
    DISPATCH_COUNTER_BASELINE_OFFSET_INDEX = 6,
    // int32_t[max_tokens][top_k]: destination ranks for local tokens, reused by combine to gather results.
    TOPK_TARGET_RANKS_OFFSET_INDEX = 7,
    // int32_t[max_tokens][top_k]: matching receive-slot indices within each destination's sender slice.
    TOPK_TARGET_INDICES_OFFSET_INDEX = 8,
    // int32_t[ep_size][eplb_stats_num_experts]: gathered per-rank expert statistics; empty without EPLB.
    EPLB_GATHERED_STATS_OFFSET_INDEX = 9,
    // Receive region for dispatched activations, scales, expert IDs/weights and optional extra payloads.
    DISPATCH_PAYLOAD_OFFSET_INDEX = 10,
    // Reserved dispatch payload capacity, including payload alignment padding.
    DISPATCH_PAYLOAD_SIZE_INDEX = 11,

    // Combine control, expert-output staging and the CFT-only receive inbox.
    // uint32_t[ep_size]: peer readiness epochs, used by fence combine and the CFT cross-round guard.
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = 12,
    // CFT-only cumulative received-byte counters per expert-rank/token slot, kCftCounterStride bytes apart.
    COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX = 13,
    // CFT-only uint64_t[ep_size][max_tokens]: consumed-byte baselines for combine counters.
    COMBINE_COUNTER_BASELINE_OFFSET_INDEX = 14,
    // Expert-output / staging region: fence peers pull from it; MoE may write directly into it.
    COMBINE_INPUT_OFFSET_INDEX = 15,
    // Reserved combine input capacity, large enough for original-dtype MoE output even with FP8 wire data.
    COMBINE_INPUT_SIZE_INDEX = 16,
    // CFT-only receive inbox: peer pushes and the self contribution are gathered here for local reduction.
    COMBINE_RECV_OFFSET_INDEX = 17,
    // CFT receive payload capacity in wire-format bytes, excluding counters and baselines.
    COMBINE_RECV_SIZE_INDEX = 18,

    // Allocation-time configuration used to construct and validate the layout, not mutable round state.
    // Maximum input token count per rank; bounds routing storage and per-rank token slots.
    MAX_NUM_TOKENS_INDEX = 19,
    // Configured experts selected per token; determines routing-table capacity.
    TOP_K_INDEX = 20,
    // Number of ranks in the EP group; determines peer-array sizes.
    EP_SIZE_INDEX = 21,
    // Number of expert statistics gathered from each rank; zero disables EPLB storage.
    EPLB_STATS_NUM_EXPERTS_INDEX = 22,
    // Whether CFT counters, baselines and receive storage are allocated; not the current transport choice.
    CFT_ENABLED_INDEX = 23,
    // Total required bytes per rank, including all control buffers, payloads and alignment padding.
    WORKSPACE_SIZE_INDEX = 24,
    // Number of int64_t entries in the metadata tensor; not a stored layout field.
    NUM_METAINFO_FIELDS = 25
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
        {"MOE_A2A_DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX", DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_COUNTER_BASELINE_OFFSET_INDEX", DISPATCH_COUNTER_BASELINE_OFFSET_INDEX},
        {"MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX", TOPK_TARGET_RANKS_OFFSET_INDEX},
        {"MOE_A2A_TOPK_TARGET_INDICES_OFFSET_INDEX", TOPK_TARGET_INDICES_OFFSET_INDEX},
        {"MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX", EPLB_GATHERED_STATS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_PAYLOAD_OFFSET_INDEX", DISPATCH_PAYLOAD_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_PAYLOAD_SIZE_INDEX", DISPATCH_PAYLOAD_SIZE_INDEX},
        {"MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX", COMBINE_COMPLETION_FLAGS_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX", COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_COUNTER_BASELINE_OFFSET_INDEX", COMBINE_COUNTER_BASELINE_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_INPUT_OFFSET_INDEX", COMBINE_INPUT_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_INPUT_SIZE_INDEX", COMBINE_INPUT_SIZE_INDEX},
        {"MOE_A2A_COMBINE_RECV_OFFSET_INDEX", COMBINE_RECV_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_RECV_SIZE_INDEX", COMBINE_RECV_SIZE_INDEX},
        {"MOE_A2A_MAX_NUM_TOKENS_INDEX", MAX_NUM_TOKENS_INDEX},
        {"MOE_A2A_TOP_K_INDEX", TOP_K_INDEX},
        {"MOE_A2A_EP_SIZE_INDEX", EP_SIZE_INDEX},
        {"MOE_A2A_EPLB_STATS_NUM_EXPERTS_INDEX", EPLB_STATS_NUM_EXPERTS_INDEX},
        {"MOE_A2A_CFT_ENABLED_INDEX", CFT_ENABLED_INDEX},
        {"MOE_A2A_WORKSPACE_SIZE_INDEX", WORKSPACE_SIZE_INDEX},
        {"MOE_A2A_NUM_METAINFO_FIELDS", NUM_METAINFO_FIELDS},
        {"MOE_A2A_MAX_RANKS", kMaxRanks},
        {"MOE_A2A_MAX_TOP_K", kMaxTopK},
        {"MOE_A2A_MAX_PAYLOADS", kMaxPayloads},
        {"MOE_A2A_WORKSPACE_ALIGNMENT", kWorkspaceAlignment},
    };
}

} // namespace moe_comm
} // namespace torch_ext

TRTLLM_NAMESPACE_END
