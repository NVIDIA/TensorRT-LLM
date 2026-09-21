/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "kv_cache_manager_v2/common.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class Block;

using EventTokenId = std::variant<int64_t, std::string>;

struct MmKey
{
    std::string hash;
    int startOffset = 0;
    std::optional<std::string> uuid;
    bool hasUuidField = false;

    bool operator==(MmKey const& other) const
    {
        return hash == other.hash && startOffset == other.startOffset && uuid == other.uuid
            && hasUuidField == other.hasUuidField;
    }
};

struct DecodedEventBlock
{
    std::vector<EventTokenId> tokenIds;
    std::vector<MmKey> mmKeys;
};

[[nodiscard]] std::string digestToHex(Digest const& digest);

//! Decode the V2 digest-first multimodal representation used by KV event consumers.
//! When mmTokenIdOffset is absent, tokens are still preserved but no MM segments are derived.
[[nodiscard]] DecodedEventBlock decodeEventBlock(Block const& block, std::optional<int> mmTokenIdOffset);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
