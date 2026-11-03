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

#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/exceptions.h"

#include <cstdlib>
#include <unordered_map>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Stream-ordered events: record registry and wait selection
// ---------------------------------------------------------------------------

namespace
{

bool readStreamOrderedEventsEnv() noexcept
{
    char const* value = std::getenv("TLLM_KV_CACHE_MANAGER_V2_STREAM_ORDERED_EVENTS");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

//! Identity of a stream for the stream-order rules, or 0 when it is unknown or must not be used:
//! the NULL / legacy / per-thread default stream handles name different streams in different
//! contexts or threads. cuStreamGetId is unique for the lifetime of the context, so a destroyed
//! stream whose handle value is reused does not alias the new one.
uint64_t streamIdentity(CUstream stream) noexcept
{
    if (stream == nullptr || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)
    {
        return 0;
    }
    unsigned long long id = 0;
    if (cuStreamGetId(stream, &id) != CUDA_SUCCESS)
    {
        return 0;
    }
    return static_cast<uint64_t>(id) + 1; // 0 stays reserved for "unknown"
}

struct EventRecordInfo
{
    uint64_t streamId = 0; // 0: unknown, never elided
    uint64_t seq = 0;
};

//! The latest record of every pooled CUevent. Pooled events are only recorded through
//! CachedCudaEvent, and a CUevent is recorded again only after its PooledEvent has retired it, so
//! the entry for a handle loaded from a live CachedCudaEvent describes that event's record.
class EventRecordRegistry
{
public:
    static EventRecordRegistry& instance()
    {
        static EventRecordRegistry registry;
        return registry;
    }

    //! Record `event` on `stream`. The sequence number is taken under the same mutex as the
    //! record, so of two records on one stream the larger number is the later one in stream
    //! order, even when they come from different threads.
    void record(CUevent event, CUstream stream)
    {
        uint64_t const streamId = streamIdentity(stream);
        std::lock_guard<std::mutex> const lock(mMutex);
        cuCheck(cuEventRecord(event, stream));
        mRecords[event] = EventRecordInfo{streamId, ++mSeq};
    }

    void lookup(CUevent const* events, size_t count, EventRecordInfo* out) const
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        for (size_t i = 0; i < count; ++i)
        {
            auto const it = mRecords.find(events[i]);
            out[i] = it == mRecords.end() ? EventRecordInfo{} : it->second;
        }
    }

private:
    mutable std::mutex mMutex;
    std::unordered_map<CUevent, EventRecordInfo> mRecords;
    uint64_t mSeq = 0;
};

struct AtomicStreamOrderedCounters
{
    std::atomic<uint64_t> waitsIssued{0};
    std::atomic<uint64_t> waitsElided{0};
    std::atomic<uint64_t> mergesWithoutStream{0};
    std::atomic<uint64_t> mergesWithStream{0};
};

AtomicStreamOrderedCounters gStreamOrderedCounters;

//! Indices into `infos` of the events that a stream with identity `waiterId` (0: none) still has
//! to wait for: every event of unknown stream, plus the latest event of each known stream other
//! than the waiter's own. Unknown-stream events come first, in input order, then one event per
//! stream in first-seen order.
void selectStreamOrderedWaits(uint64_t waiterId, std::vector<EventRecordInfo> const& infos, std::vector<size_t>& out)
{
    struct Latest
    {
        uint64_t streamId;
        uint64_t seq;
        size_t index;
    };

    out.clear();
    thread_local std::vector<Latest> latest;
    thread_local std::unordered_map<uint64_t, size_t> slotOfStream;
    latest.clear();
    if (!slotOfStream.empty())
    {
        // clear() resets every bucket, so skip it in the common case where the map was not used.
        slotOfStream.clear();
    }
    constexpr size_t kLinearSearchLimit = 16;
    for (size_t i = 0; i < infos.size(); ++i)
    {
        auto const& info = infos[i];
        if (info.streamId == 0)
        {
            out.push_back(i);
            continue;
        }
        if (info.streamId == waiterId)
        {
            continue;
        }
        size_t slot = latest.size();
        if (latest.size() <= kLinearSearchLimit)
        {
            for (size_t j = 0; j < latest.size(); ++j)
            {
                if (latest[j].streamId == info.streamId)
                {
                    slot = j;
                    break;
                }
            }
        }
        else if (auto const it = slotOfStream.find(info.streamId); it != slotOfStream.end())
        {
            slot = it->second;
        }
        if (slot == latest.size())
        {
            latest.push_back({info.streamId, info.seq, i});
            if (latest.size() == kLinearSearchLimit + 1)
            {
                for (size_t j = 0; j < latest.size(); ++j)
                {
                    slotOfStream.emplace(latest[j].streamId, j);
                }
            }
            else if (latest.size() > kLinearSearchLimit + 1)
            {
                slotOfStream.emplace(info.streamId, slot);
            }
        }
        else if (info.seq > latest[slot].seq)
        {
            latest[slot].seq = info.seq;
            latest[slot].index = i;
        }
    }
    for (auto const& entry : latest)
    {
        out.push_back(entry.index);
    }
}

} // namespace

bool streamOrderedEventsEnabled() noexcept
{
    static bool const enabled = readStreamOrderedEventsEnv();
    return enabled;
}

StreamOrderedEventCounters streamOrderedEventCounters() noexcept
{
    return StreamOrderedEventCounters{gStreamOrderedCounters.waitsIssued.load(std::memory_order_relaxed),
        gStreamOrderedCounters.waitsElided.load(std::memory_order_relaxed),
        gStreamOrderedCounters.mergesWithoutStream.load(std::memory_order_relaxed),
        gStreamOrderedCounters.mergesWithStream.load(std::memory_order_relaxed)};
}

void streamWaitEvents(CudaStream stream, std::vector<CUevent> events)
{
    events.erase(std::remove(events.begin(), events.end(), nullptr), events.end());
    if (events.empty())
    {
        return;
    }
    auto const cuStream = reinterpret_cast<CUstream>(stream);
    if (!streamOrderedEventsEnabled())
    {
        std::sort(events.begin(), events.end());
        events.erase(std::unique(events.begin(), events.end()), events.end());
        for (CUevent h : events)
        {
            cuCheck(cuStreamWaitEvent(cuStream, h, 0));
        }
        return;
    }
    // Select first: copies of one event share its stream and sequence number, so the selection
    // already collapses the duplicates of known streams and only the few remaining events need
    // the sort-and-unique pass.
    thread_local std::vector<EventRecordInfo> infos;
    thread_local std::vector<size_t> selected;
    infos.resize(events.size());
    EventRecordRegistry::instance().lookup(events.data(), events.size(), infos.data());
    selectStreamOrderedWaits(streamIdentity(cuStream), infos, selected);
    std::vector<CUevent> waits;
    waits.reserve(selected.size());
    for (size_t const index : selected)
    {
        waits.push_back(events[index]);
    }
    std::sort(waits.begin(), waits.end());
    waits.erase(std::unique(waits.begin(), waits.end()), waits.end());
    gStreamOrderedCounters.waitsIssued.fetch_add(waits.size(), std::memory_order_relaxed);
    gStreamOrderedCounters.waitsElided.fetch_add(events.size() - waits.size(), std::memory_order_relaxed);
    for (CUevent h : waits)
    {
        cuCheck(cuStreamWaitEvent(cuStream, h, 0));
    }
}

// ---------------------------------------------------------------------------
// CudaEventPool / CudaStreamPool singleton implementations
// ---------------------------------------------------------------------------

CudaEventPool::CudaEventPool()
    : SimplePool(
        []() -> CUevent
        {
            CUevent ev;
            cuCheck(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING));
            return ev;
        },
        [](CUevent ev) { cuEventDestroy(ev); },
        /*initSize=*/1024)
{
}

CudaEventPool& CudaEventPool::instance()
{
    static CudaEventPool pool;
    return pool;
}

CudaStreamPool::CudaStreamPool()
    : SimplePool(
        []() -> CUstream
        {
            CUstream s;
            cuCheck(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
            return s;
        },
        [](CUstream s) { cuStreamDestroy(s); },
        /*initSize=*/128)
{
}

CudaStreamPool& CudaStreamPool::instance()
{
    static CudaStreamPool pool;
    return pool;
}

// ---------------------------------------------------------------------------
// CachedCudaEvent implementation
// ---------------------------------------------------------------------------

CachedCudaEvent CachedCudaEvent::makeNull() noexcept
{
    return CachedCudaEvent{};
}

CachedCudaEvent::CachedCudaEvent(CudaStream stream) noexcept
{
    KVCM2_POISON_ON_EXCEPT(
        [&]()
        {
            mEvent = std::make_shared<PooledEvent>(CudaEventPool::instance().get());
            if (streamOrderedEventsEnabled())
            {
                EventRecordRegistry::instance().record(mEvent->load(), reinterpret_cast<CUstream>(stream));
            }
            else
            {
                cuCheck(cuEventRecord(mEvent->load(), reinterpret_cast<CUstream>(stream)));
            }
        });
}

bool CachedCudaEvent::queryComplete() const
{
    CUevent event = handle();
    if (event == nullptr)
    {
        return true;
    }
    CUresult result = cuEventQuery(event);
    if (result == CUDA_SUCCESS)
    {
        close();
        return true;
    }
    if (result == CUDA_ERROR_NOT_READY)
    {
        return false;
    }
    throw CuError(result);
}

void CachedCudaEvent::synchronize() const
{
    CUevent event = handle();
    if (event == nullptr)
    {
        return;
    }
    cuCheck(cuEventSynchronize(event));
    close();
}

void CachedCudaEvent::waitInStream(CudaStream stream) const
{
    CUevent event = handle();
    if (event == nullptr)
    {
        return;
    }
    auto const cuStream = reinterpret_cast<CUstream>(stream);
    if (streamOrderedEventsEnabled())
    {
        EventRecordInfo info;
        EventRecordRegistry::instance().lookup(&event, 1, &info);
        if (info.streamId != 0 && info.streamId == streamIdentity(cuStream))
        {
            gStreamOrderedCounters.waitsElided.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        gStreamOrderedCounters.waitsIssued.fetch_add(1, std::memory_order_relaxed);
    }
    cuCheck(cuStreamWaitEvent(cuStream, event, 0));
}

void CachedCudaEvent::close() const noexcept
{
    if (mEvent)
    {
        mEvent->retire();
    }
}

// ---------------------------------------------------------------------------
// CachedCudaStream implementation
// ---------------------------------------------------------------------------

CachedCudaStream::CachedCudaStream()
    : mPoolItem(CudaStreamPool::instance().get())
{
}

CachedCudaEvent CachedCudaStream::recordEvent() noexcept
{
    return CachedCudaEvent{reinterpret_cast<CudaStream>(handle())};
}

void CachedCudaStream::synchronize()
{
    cuCheck(cuStreamSynchronize(handle()));
}

// ---------------------------------------------------------------------------
// TemporaryCudaStream implementation
// ---------------------------------------------------------------------------

TemporaryCudaStream::TemporaryCudaStream(std::vector<CUevent> priorEvents)
    : mStream()
{
    CudaStream cs = reinterpret_cast<CudaStream>(mStream.handle());
    streamWaitEvents(cs, std::move(priorEvents));
}

// ---------------------------------------------------------------------------
// mergeEvents — merge multiple CUDA events into one.
// Mirrors Python's merge_events() in _utils.py.
// ---------------------------------------------------------------------------

CachedCudaEvent mergeEvents(std::vector<CachedCudaEvent>& events)
{
    // A single live event is returned as is rather than merged, so track one live event and
    // how many there are.
    CachedCudaEvent* onlyLive = nullptr;
    size_t numLive = 0;
    for (auto& ev : events)
    {
        if (!ev.isClosed())
        {
            ++numLive;
            onlyLive = &ev;
        }
    }
    if (numLive == 0)
        return CachedCudaEvent::makeNull();
    if (numLive == 1)
        return std::move(*onlyLive);
    if (streamOrderedEventsEnabled())
    {
        // Keep the latest event of every known stream (it completes after every other event
        // recorded there) and each event of unknown stream, then drop those already complete.
        // Only when more than one is left does the merge need a temporary stream.
        std::vector<CUevent> handles;
        std::vector<size_t> liveIndex;
        handles.reserve(numLive);
        liveIndex.reserve(numLive);
        for (size_t i = 0; i < events.size(); ++i)
        {
            CUevent const h = events[i].handle();
            if (h != nullptr)
            {
                handles.push_back(h);
                liveIndex.push_back(i);
            }
        }
        std::vector<EventRecordInfo> infos(handles.size());
        EventRecordRegistry::instance().lookup(handles.data(), handles.size(), infos.data());
        std::vector<size_t> selected;
        selectStreamOrderedWaits(/*waiterId=*/0, infos, selected);
        std::vector<size_t> keep;
        keep.reserve(selected.size());
        for (size_t const s : selected)
        {
            // Copies of one unknown-stream event share a handle; keep it once.
            bool const duplicate
                = std::any_of(keep.begin(), keep.end(), [&](size_t k) { return handles[k] == handles[s]; });
            if (!duplicate)
            {
                keep.push_back(s);
            }
        }
        if (keep.size() > 1)
        {
            std::erase_if(keep, [&](size_t k) { return events[liveIndex[k]].queryComplete(); });
        }
        if (keep.size() <= 1)
        {
            gStreamOrderedCounters.mergesWithoutStream.fetch_add(1, std::memory_order_relaxed);
            return keep.empty() ? CachedCudaEvent::makeNull() : std::move(events[liveIndex[keep.front()]]);
        }
        gStreamOrderedCounters.mergesWithStream.fetch_add(1, std::memory_order_relaxed);
        std::vector<CUevent> priors;
        priors.reserve(keep.size());
        for (size_t const k : keep)
        {
            priors.push_back(handles[k]);
        }
        TemporaryCudaStream tempStream(std::move(priors));
        {
            auto scope = tempStream.enter();
        }
        return tempStream.takeFinishEvent();
    }
    // Multiple live events: merge via TemporaryCudaStream. Closed events yield a null handle,
    // which streamWaitEvents drops.
    std::vector<CUevent> priors;
    priors.reserve(events.size());
    for (auto const& ev : events)
        priors.push_back(ev.handle());
    TemporaryCudaStream tempStream(std::move(priors));
    {
        auto scope = tempStream.enter();
    }
    return tempStream.takeFinishEvent();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
