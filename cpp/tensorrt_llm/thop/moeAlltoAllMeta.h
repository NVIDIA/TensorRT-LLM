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

#include <cstdint>
#include <utility>
#include <vector>

namespace torch_ext
{

// Enum for indexing into moe_a2a_metainfo tensor
enum MoeA2AMetaInfoIndex
{
    FLAG_VAL_OFFSET_INDEX = 0,
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = 1,
    SEND_COUNTERS_OFFSET_INDEX = 2,
    RECV_COUNTERS_OFFSET_INDEX = 3,
    // Dispatch completion flags offset
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = 4,
    // Combine completion flags offset
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = 5,
    PAYLOAD_DATA_OFFSET_INDEX = 6,
    NUM_METAINFO_FIELDS = 7
};

inline std::vector<std::pair<char const*, int64_t>> getMoeA2AMetaInfoIndexPairs()
{
    return {
        {"MOE_A2A_FLAG_VAL_OFFSET_INDEX", static_cast<int64_t>(FLAG_VAL_OFFSET_INDEX)},
        {"MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX", static_cast<int64_t>(LOCAL_TOKEN_COUNTER_OFFSET_INDEX)},
        {"MOE_A2A_SEND_COUNTERS_OFFSET_INDEX", static_cast<int64_t>(SEND_COUNTERS_OFFSET_INDEX)},
        {"MOE_A2A_RECV_COUNTERS_OFFSET_INDEX", static_cast<int64_t>(RECV_COUNTERS_OFFSET_INDEX)},
        {"MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX",
            static_cast<int64_t>(DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX)},
        {"MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX", static_cast<int64_t>(COMBINE_COMPLETION_FLAGS_OFFSET_INDEX)},
        {"MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX", static_cast<int64_t>(PAYLOAD_DATA_OFFSET_INDEX)},
    };
}
} // namespace torch_ext
