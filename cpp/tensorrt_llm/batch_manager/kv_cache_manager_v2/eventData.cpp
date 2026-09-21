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

#include "kv_cache_manager_v2/eventData.h"

#include "kv_cache_manager_v2/blockRadixTree.h"

#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

std::string digestToHex(Digest const& digest)
{
    constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.resize(digest.size() * 2);
    for (size_t i = 0; i < digest.size(); ++i)
    {
        auto const value = std::to_integer<uint8_t>(digest[i]);
        result[2 * i] = kHex[value >> 4U];
        result[2 * i + 1] = kHex[value & 0x0FU];
    }
    return result;
}

DecodedEventBlock decodeEventBlock(Block const& block, std::optional<int> mmTokenIdOffset)
{
    DecodedEventBlock result;
    result.tokenIds.reserve(block.tokens.size());

    Digest const* itemDigest = nullptr;
    if (mmTokenIdOffset.has_value() && block.prev != nullptr && block.prev->type() == NodeBase::Type::kBLOCK)
    {
        itemDigest = static_cast<Block const*>(block.prev)->getLastTokenDigest().get();
    }
    bool inMmRun = false;
    for (auto const& token : block.tokens)
    {
        if (token.isDigest())
        {
            result.tokenIds.emplace_back(std::in_place_index<1>, digestToHex(token.digest()));
            if (mmTokenIdOffset.has_value())
            {
                itemDigest = &token.digest();
                result.mmKeys.push_back(
                    {std::string(reinterpret_cast<char const*>(itemDigest->data()), itemDigest->size()), 0,
                        std::nullopt, false});
                inMmRun = true;
            }
            continue;
        }

        auto const tokenId = token.tokenId();
        result.tokenIds.emplace_back(std::in_place_index<0>, tokenId);
        if (itemDigest != nullptr && tokenId > *mmTokenIdOffset)
        {
            if (!inMmRun)
            {
                result.mmKeys.push_back(
                    {std::string(reinterpret_cast<char const*>(itemDigest->data()), itemDigest->size()),
                        tokenId - *mmTokenIdOffset, std::nullopt, false});
            }
            inMmRun = true;
        }
        else
        {
            // Text separates runs of the same item, so retain its digest for later continuations.
            inMmRun = false;
        }
    }
    return result;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
