/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/executor/executor.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

using SizeType32 = tensorrt_llm::runtime::SizeType32;

class KVCacheBlock;
using BlockPtr = std::shared_ptr<KVCacheBlock>;

class KVCacheEventManager
{
public:
    explicit KVCacheEventManager(size_t maxKVEventEntries, std::optional<SizeType32> attentionDpRank = std::nullopt,
        std::optional<SizeType32> attentionDpSize = std::nullopt, SizeType32 attentionDpEventsGatherPeriodMs = 5);

    ~KVCacheEventManager();
    KVCacheEventManager(KVCacheEventManager& other) = delete;
    KVCacheEventManager& operator=(KVCacheEventManager& other) = delete;
    KVCacheEventManager(KVCacheEventManager&& other) = delete;
    KVCacheEventManager& operator=(KVCacheEventManager&& other) = delete;

    void enqueueCreatedEvent(std::vector<SizeType32> const& numBlocksPerCacheLevel, SizeType32 windowSize);

    void enqueueStoredEvent(std::vector<BlockPtr> const& blocks, SizeType32 windowSize);

    void enqueueRemovedEvent(BlockPtr const& block, SizeType32 windowSize);

    void enqueueUpdatedEvent(executor::KVCacheUpdatedData const& data, SizeType32 windowSize);

    // Get events in mEvents. If there are no events, wait for a maximum of `timeout` milliseconds.
    std::deque<executor::KVCacheEvent> getEvents(std::optional<std::chrono::milliseconds> timeout);

    // Clear the event buffer, and asynchronously move events to the event queue.
    void flush();

    // Worker thread which adds events to mEvents.
    void worker();

    // Thread which exchanges events if attentionDP is enabled
    void exchangeAttentionDpThread();

private:
    // Add an event to mEventQueue
    void enqueueEvent(executor::KVCacheEvent&& event);

    /// @brief Flag to terminate the worker
    std::atomic<bool> mRun;
    /// @brief Worker thread
    std::thread mWorkerThread;
    /// @brief Exchange thread for attention DP events
    std::thread mExchangeAttentionDpThread;

    /// @brief The deque of events
    std::deque<executor::KVCacheEvent> mEvents;
    /// @brief Lock for mEvents
    std::mutex mEventsMutex;
    /// @brief Condition variable for blocking read
    std::condition_variable mEmptyCV;

    /// @brief List of buffers waiting awaiting insertion into mEvents. Consumed by the worker.
    std::deque<std::deque<executor::KVCacheEvent>> mPendingEvents;
    /// @brief Lock for mPendingEvents
    std::mutex mPendingEventsMutex;
    /// @brief Condition variable to notify worker thread
    std::condition_variable mPendingEmptyCV;

    /// @brief Buffer of events waiting to be added to the eventQueue. Only ever accessed by forward pass thread.
    std::deque<executor::KVCacheEvent> mEventQueue;

    /// @brief The maximum size of the deque
    size_t mMaxSize;
    /// @brief An auto-incrementing event id counter
    size_t mEventId;

    /// @brief Attention DP ranks and size
    /// If set, we will exchange KV cache events and accumulate on rank 0
    std::optional<SizeType32> mAttentionDpRank;
    std::optional<SizeType32> mAttentionDpSize;

    /// @brief The period in milliseconds to gather attention DP events across rank
    SizeType32 mAttentionDpEventsGatherPeriodMs;

    /// @brief MPI communicator for attention DP
    std::unique_ptr<tensorrt_llm::mpi::MpiComm> mMpiComm;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
