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
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

KVCacheEventManager::KVCacheEventManager(size_t maxKVEventEntries, std::optional<SizeType32> attentionDpRank,
    std::optional<SizeType32> attentionDpSize, SizeType32 attentionDpEventsGatherPeriodMs)
    : mRun{true}
    , mMaxSize{maxKVEventEntries}
    , mEventId{0}
    , mAttentionDpRank{attentionDpRank}
    , mAttentionDpSize{attentionDpSize}
    , mAttentionDpEventsGatherPeriodMs(attentionDpEventsGatherPeriodMs)
{
    TLLM_CHECK(mMaxSize > 0);
    if (mAttentionDpRank)
    {
        TLLM_CHECK_WITH_INFO(
            mAttentionDpSize.has_value(), "If attention DP rank is set, the attention DP size must also be set");
        TLLM_CHECK_WITH_INFO(mAttentionDpRank.value() < mAttentionDpSize.value(),
            "Attention DP rank must be less than attention DP size");
        if (mAttentionDpRank.value() == 0)
        {
            // Rank 0 will gather events from all other ranks
            // Need to increase size
            mMaxSize *= mAttentionDpSize.value();
        }
        // Create a communicator to be used for event exchange
        mMpiComm = std::make_unique<tensorrt_llm::mpi::MpiComm>(COMM_SESSION.split(0, mAttentionDpRank.value()));
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            !mAttentionDpSize.has_value(), "If attention DP rank is not set, the attention DP size must not be set");
    }
    mWorkerThread = std::thread([this]() { this->worker(); });
#if ENABLE_MULTI_DEVICE
    if (mAttentionDpRank)
    {
        mExchangeAttentionDpThread = std::thread([this]() { this->exchangeAttentionDpThread(); });
    }
#endif
};

KVCacheEventManager::~KVCacheEventManager()
{
    mRun = false;
    mPendingEmptyCV.notify_all();
    mEmptyCV.notify_all();
    mWorkerThread.join();
#if ENABLE_MULTI_DEVICE
    if (mAttentionDpRank)
    {
        mExchangeAttentionDpThread.join();
    }
#endif
}

void KVCacheEventManager::enqueueCreatedEvent(
    std::vector<SizeType32> const& numBlocksPerCacheLevel, SizeType32 windowSize)
{
    enqueueEvent({mEventId++, tle::KVCacheCreatedData{numBlocksPerCacheLevel}, windowSize, mAttentionDpRank});
}

void KVCacheEventManager::enqueueStoredEvent(std::vector<BlockPtr> const& blocks, SizeType32 windowSize)
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

    enqueueEvent({mEventId++, data, windowSize, mAttentionDpRank});
}

void KVCacheEventManager::enqueueRemovedEvent(BlockPtr const& block, SizeType32 windowSize)
{
    // We can only batch the removed block events if the same sliding window size is used.
    if (!mEventQueue.empty() && mEventQueue.back().windowSize == windowSize
        && std::holds_alternative<tle::KVCacheRemovedData>(mEventQueue.back().data))
    {
        std::get<tle::KVCacheRemovedData>(mEventQueue.back().data).blockHashes.push_back(block->getHash());
    }
    else
    {
        enqueueEvent({mEventId++, tle::KVCacheRemovedData{{block->getHash()}}, windowSize, mAttentionDpRank});
    }
}

void KVCacheEventManager::enqueueUpdatedEvent(tle::KVCacheUpdatedData const& data, SizeType32 windowSize)
{
    enqueueEvent({mEventId++, data, windowSize, mAttentionDpRank});
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

    if (eventQueue.empty())
    {
        return;
    }

    std::unique_lock<std::mutex> lck(mPendingEventsMutex);
    mPendingEvents.push_back(std::move(eventQueue));
    mPendingEmptyCV.notify_one();
}

void KVCacheEventManager::exchangeAttentionDpThread()
{
#if ENABLE_MULTI_DEVICE
    while (true)
    {
        TLLM_CHECK(mAttentionDpRank);

        // Check if any of the ranks have been shutdown
        int32_t numFinished = 0;
        int32_t finished = mRun ? 0 : 1;
        mMpiComm->allreduce(&finished, &numFinished, 1, mpi::MpiType::kINT32, mpi::MpiOp::SUM);
        if (numFinished > 0)
        {
            TLLM_LOG_INFO("One of the rank has been shut down, exiting");
            break;
        }

        // If we are not rank 0, send events to rank 0
        if (mAttentionDpRank.value() != 0)
        {
            std::vector<char> serializedEvents;
            uint64_t numEvents = 0;
            {
                std::lock_guard<std::mutex> lck(mEventsMutex);
                serializedEvents = executor::Serialization::serialize(mEvents);
                numEvents = mEvents.size();
                mEvents.clear();
            }
            uint64_t vecSize = numEvents > 0 ? serializedEvents.size() : 0;
            mMpiComm->send(&vecSize, 1, mpi::MpiType::kUINT64, 0, mpi::MpiTag::kKvCacheEventSize);
            if (vecSize > 0)
            {
                mMpiComm->send(serializedEvents.data(), serializedEvents.size(), mpi::MpiType::kCHAR, 0,
                    mpi::MpiTag::kKvCacheEvent);
            }
        }
        else
        {
            TLLM_CHECK(mAttentionDpSize.has_value());
            // Loop until have received events from all ranks
            for (int rank = 1; rank < mAttentionDpSize.value(); ++rank)
            {
                uint64_t vecSize{0};
                mMpiComm->recv(&vecSize, 1, mpi::MpiType::kUINT64, rank, mpi::MpiTag::kKvCacheEventSize);
                if (vecSize > 0)
                {
                    std::vector<char> serializedEvents(vecSize);
                    mMpiComm->recv(
                        serializedEvents.data(), vecSize, mpi::MpiType::kCHAR, rank, mpi::MpiTag::kKvCacheEvent);

                    // Deserialize the events and add them to the local queue
                    auto rankEvents = executor::Serialization::deserializeKVCacheEvents(serializedEvents);
                    {
                        std::lock_guard<std::mutex> lck(mEventsMutex);
                        mEvents.insert(mEvents.end(), rankEvents.begin(), rankEvents.end());
                        mEmptyCV.notify_one();
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(mAttentionDpEventsGatherPeriodMs));
    }
#else
    TLLM_THROW("Multi device support is disabled.");
#endif
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

        // Notify the empty condition variable to wake up any waiting threads
        mEmptyCV.notify_one();
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
