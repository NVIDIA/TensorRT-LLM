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

#include "kv_cache_manager_v2/streamingEventSink.h"

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/page.h"
#include "tensorrt_llm/common/logger.h"

#include <cinttypes>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

StreamingEventSink::StreamingEventSink(int tokensPerBlock, int maxEntries)
    : mTokensPerBlock(tokensPerBlock)
    , mMaxEntries(maxEntries)
{
    if (mTokensPerBlock <= 0)
    {
        throw std::invalid_argument("tokensPerBlock must be positive");
    }
    if (mMaxEntries <= 0)
    {
        throw std::invalid_argument("maxEntries must be positive");
    }
}

void StreamingEventSink::setTargetLifeCycle(LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mTargetLifeCycle = lifeCycle;
}

std::vector<StreamingEventData> StreamingEventSink::drainIterationEvents()
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto events = std::move(mPendingEvents);
    mPendingEvents.clear();
    mPendingEntries = 0;
    return events;
}

StreamingEventStats StreamingEventSink::getStats() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mStats;
}

void StreamingEventSink::addStoredBlock(Block const& block)
{
    std::lock_guard<std::mutex> lock(mMutex);
    addStoredBlockUnlocked(block);
}

void StreamingEventSink::addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mTargetLifeCycle.has_value())
    {
        return;
    }
    if (lifeCycle != *mTargetLifeCycle)
    {
        ++mStats.nonTargetLifeCyclesIgnored;
        return;
    }
    addStoredBlockUnlocked(block);
}

void StreamingEventSink::addRemovedBlock(Digest const& blockKey)
{
    std::lock_guard<std::mutex> lock(mMutex);
    addRemovedBlockUnlocked(blockKey);
}

void StreamingEventSink::addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mTargetLifeCycle.has_value())
    {
        return;
    }
    if (lifeCycle != *mTargetLifeCycle)
    {
        ++mStats.nonTargetLifeCyclesIgnored;
        return;
    }
    addRemovedBlockUnlocked(blockKey);
}

void StreamingEventSink::addCacheLevelUpdated(Digest const&, CacheLevel, CacheLevel, LifeCycleId)
{
    // The streaming protocol currently tracks radix-tree residency, not cache-tier movement.
}

void StreamingEventSink::addStoredBlockUnlocked(Block const& block)
{
    if (!mTargetLifeCycle.has_value() || *mTargetLifeCycle >= block.storage.size())
    {
        return;
    }
    auto const* page = block.getPage(*mTargetLifeCycle);
    if (page == nullptr)
    {
        return;
    }
    if (!block.isFull() || page->numTokensInBlock < static_cast<int>(block.tokens.size()))
    {
        ++mStats.partialBlocksSuppressed;
        return;
    }
    if (mStoredBlocks.count(block.key) != 0)
    {
        return;
    }
    for (auto const& token : block.tokens)
    {
        if (token.isDigest())
        {
            ++mStats.multimodalBlocksSuppressed;
            return;
        }
    }
    if (!reserveEntryUnlocked())
    {
        return;
    }
    if (block.prev == nullptr)
    {
        throw std::logic_error("Cannot publish an orphan KV cache block");
    }

    int64_t const blockHash = wireHash(block.key);
    std::optional<int64_t> parentHash;
    if (block.prev->type() == NodeBase::Type::kBLOCK)
    {
        parentHash = wireHash(static_cast<Block const*>(block.prev)->key);
    }

    std::vector<TokenId> tokenIds;
    tokenIds.reserve(block.tokens.size());
    for (auto const& token : block.tokens)
    {
        tokenIds.push_back(token.tokenId());
    }

    mStoredBlocks.emplace(block.key, blockHash);
    if (!mPendingEvents.empty())
    {
        auto* stored = std::get_if<StreamingBlockStoredData>(&mPendingEvents.back());
        if (stored != nullptr && !stored->blockHashes.empty() && parentHash.has_value()
            && stored->blockHashes.back() == *parentHash)
        {
            stored->blockHashes.push_back(blockHash);
            stored->tokenIds.insert(stored->tokenIds.end(), tokenIds.begin(), tokenIds.end());
            ++mStats.storedBlocks;
            return;
        }
    }
    mPendingEvents.emplace_back(StreamingBlockStoredData{{blockHash}, parentHash, std::move(tokenIds)});
    ++mStats.storedBlocks;
}

void StreamingEventSink::addRemovedBlockUnlocked(Digest const& blockKey)
{
    auto const stored = mStoredBlocks.find(blockKey);
    if (stored == mStoredBlocks.end())
    {
        return;
    }
    int64_t const blockHash = stored->second;
    mStoredBlocks.erase(stored);
    addRemovedHashUnlocked(blockHash);
}

void StreamingEventSink::addRemovedHashUnlocked(int64_t blockHash)
{
    if (!mPendingEvents.empty())
    {
        auto* removed = std::get_if<StreamingBlockRemovedData>(&mPendingEvents.back());
        if (removed != nullptr)
        {
            removed->blockHashes.push_back(blockHash);
            ++mStats.removedBlocks;
            return;
        }
    }
    mPendingEvents.emplace_back(StreamingBlockRemovedData{{blockHash}});
    ++mStats.removedBlocks;
}

bool StreamingEventSink::reserveEntryUnlocked()
{
    if (mPendingEntries < mMaxEntries)
    {
        ++mPendingEntries;
        return true;
    }
    ++mStats.droppedEvents;
    int64_t const dropped = mStats.droppedEvents;
    if (dropped == 1 || (dropped & (dropped - 1)) == 0)
    {
        TLLM_LOG_WARNING(
            "Dropping streaming KV events because the per-iteration safety cap was exceeded; "
            "dropped_events=%" PRId64,
            dropped);
    }
    return false;
}

int64_t StreamingEventSink::wireHash(Digest const& digest)
{
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i)
    {
        value = (value << 8U) | std::to_integer<uint8_t>(digest[i]);
    }
    uint64_t constexpr kSignedMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (value <= kSignedMax)
    {
        return static_cast<int64_t>(value);
    }
    return -static_cast<int64_t>(~value) - 1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
