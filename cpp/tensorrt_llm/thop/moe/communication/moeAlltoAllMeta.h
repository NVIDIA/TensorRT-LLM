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

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace moe_comm
{

// Enum for indexing into moe_a2a_metainfo tensor
enum MoeA2AMetaInfoIndex : int64_t
{
    FLAG_VAL_OFFSET_INDEX = 0,
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = 1,
    SEND_COUNTERS_OFFSET_INDEX = 2,
    RECV_COUNTERS_OFFSET_INDEX = 3,
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = 4,
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = 5,
    // Counted-write counters: uint64 per slot, kCftCounterStride-aligned.
    DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX = 6,
    COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX = 7,
    TOPK_TARGET_RANKS_OFFSET_INDEX = 8,
    TOPK_SEND_INDICES_OFFSET_INDEX = 9,
    EPLB_GATHERED_STATS_OFFSET_INDEX = 10,
    PAYLOAD_DATA_OFFSET_INDEX = 11,
    // Static max tokens/rank (a count, not a byte offset).
    MAX_NUM_TOKENS_INDEX = 12,
    DISPATCH_COUNTER_BASELINE_OFFSET_INDEX = 13,
    COMBINE_COUNTER_BASELINE_OFFSET_INDEX = 14,
    NUM_METAINFO_FIELDS = 15
};

using MoeA2ADataOffsets = std::array<int64_t, NUM_METAINFO_FIELDS>;

inline std::vector<std::pair<char const*, int64_t>> getMoeA2AMetaInfoIndexPairs()
{
    return {
        {"MOE_A2A_FLAG_VAL_OFFSET_INDEX", FLAG_VAL_OFFSET_INDEX},
        {"MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX", LOCAL_TOKEN_COUNTER_OFFSET_INDEX},
        {"MOE_A2A_SEND_COUNTERS_OFFSET_INDEX", SEND_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_RECV_COUNTERS_OFFSET_INDEX", RECV_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX", DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX", COMBINE_COMPLETION_FLAGS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX", DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX", COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX", TOPK_TARGET_RANKS_OFFSET_INDEX},
        {"MOE_A2A_TOPK_SEND_INDICES_OFFSET_INDEX", TOPK_SEND_INDICES_OFFSET_INDEX},
        {"MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX", EPLB_GATHERED_STATS_OFFSET_INDEX},
        {"MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX", PAYLOAD_DATA_OFFSET_INDEX},
        {"MOE_A2A_MAX_NUM_TOKENS_INDEX", MAX_NUM_TOKENS_INDEX},
        {"MOE_A2A_DISPATCH_COUNTER_BASELINE_OFFSET_INDEX", DISPATCH_COUNTER_BASELINE_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_COUNTER_BASELINE_OFFSET_INDEX", COMBINE_COUNTER_BASELINE_OFFSET_INDEX},
        {"MOE_A2A_NUM_METAINFO_FIELDS", NUM_METAINFO_FIELDS},
    };
}

} // namespace moe_comm
} // namespace torch_ext

TRTLLM_NAMESPACE_END
