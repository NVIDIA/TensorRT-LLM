/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/executor.h"

namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

KVCacheEventManager::KVCacheEventManager(size_t maxKVEventEntries)
    : mRun{true}
    , mMaxSize{maxKVEventEntries}
    , mEventId{0}
{
    TLLM_CHECK(mMaxSize > 0);
    // mWorkerThread = std::thread(std::bind(&KVCacheEventManager::worker, this));
    mWorkerThread = std::thread([this]() { this->worker(); });
};

KVCacheEventManager::~KVCacheEventManager()
{
    mRun = false;
    mPendingEmptyCV.notify_all();
    mEmptyCV.notify_all();
    mWorkerThread.join();
}

void KVCacheEventManager::enqueueCreatedEvent(std::vector<SizeType32> const& numBlocksPerCacheLevel)
{
    enqueueEvent({mEventId++, tle::KVCacheCreatedData{numBlocksPerCacheLevel}});
}

void KVCacheEventManager::enqueueStoredEvent(std::vector<BlockPtr> const& blocks)
{
    if (blocks.empty())
    {
        return;
    }

    auto const parentBlock = blocks.front()->getPrevBlock();
    auto const parent = (parentBlock != nullptr && parentBlock->getBlockId() >= 0)
        ? std::optional<size_t>(parentBlock->getHash())
        : std::nullopt;

    tle::KVCacheStoredData data{parent, {}};

    for (auto const& block : blocks)
    {
        data.blocks.emplace_back(block->getHash(), block->getUniqueTokens(), block->getBlockKey().loraTaskId,
            block->isPrimary() ? kPrimaryLevel : kSecondaryLevel, block->getPriority());
    }

    enqueueEvent({mEventId++, data});
}

void KVCacheEventManager::enqueueRemovedEvent(BlockPtr const& block)
{
    if (!mEventQueue.empty() && std::holds_alternative<tle::KVCacheRemovedData>(mEventQueue.back().data))
    {
        std::get<tle::KVCacheRemovedData>(mEventQueue.back().data).blockHashes.push_back(block->getHash());
    }
    else
    {
        enqueueEvent({mEventId++, tle::KVCacheRemovedData{{block->getHash()}}});
    }
}

void KVCacheEventManager::enqueueUpdatedEvent(tle::KVCacheUpdatedData const& data)
{
    enqueueEvent({mEventId++, data});
}

void KVCacheEventManager::enqueueEvent(tle::KVCacheEvent&& event)
{
    mEventQueue.emplace_back(event);
}

std::deque<tle::KVCacheEvent> KVCacheEventManager::getEvents(std::optional<std::chrono::milliseconds> timeout)
{
    std::unique_lock<std::mutex> lck(mEventsMutex);
    auto pred = [this] { return !mEvents.empty() || !mRun; };

    if (timeout.has_value())
    {
        mEmptyCV.wait_for(lck, *timeout, pred);
    }
    else
    {
        mEmptyCV.wait(lck, pred);
    }

    return std::exchange(mEvents, {});
}

void KVCacheEventManager::flush()
{
    auto eventQueue = std::exchange(mEventQueue, {});
    std::unique_lock<std::mutex> lck(mPendingEventsMutex);
    mPendingEvents.push_back(std::move(eventQueue));
    mPendingEmptyCV.notify_one();
}

void KVCacheEventManager::worker()
{
    while (true)
    {
        std::deque<tle::KVCacheEvent> events;
        {
            std::unique_lock<std::mutex> pendingLock(mPendingEventsMutex);
            mPendingEmptyCV.wait(pendingLock, [this] { return !mPendingEvents.empty() || !mRun; });
            if (!mRun)
            {
                return;
            }
            events = mPendingEvents.front();
            mPendingEvents.pop_front();
        }

        std::unique_lock<std::mutex> lck(mEventsMutex);

        SizeType32 elementsToRemove = mEvents.size() + events.size() - mMaxSize;

        // First, take elements from mEvents since they are the oldest.
        if (elementsToRemove > 0)
        {
            SizeType32 numRemoved = std::min(static_cast<SizeType32>(mEvents.size()), elementsToRemove);
            mEvents.erase(mEvents.begin(), mEvents.begin() + numRemoved);
            elementsToRemove -= numRemoved;
            TLLM_LOG_WARNING("The event queue has reached the max size of %d. Events have been discarded.", mMaxSize);
        }

        // If there's still too many events, take from the front of the events queue.
        mEvents.insert(mEvents.end(), events.begin() + std::max(0, elementsToRemove), events.end());
        mEmptyCV.notify_one();
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
