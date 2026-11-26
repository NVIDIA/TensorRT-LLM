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

#include "tensorrt_llm/common/logger.h"

#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

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
            cuCheck(cuEventRecord(mEvent->load(), reinterpret_cast<CUstream>(stream)));
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
    cuCheck(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream), event, 0));
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

CachedCudaStream::CachedCudaStream(CudaStreamPool::PoolItem item) noexcept
    : mPoolItem(std::move(item))
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

namespace
{

std::atomic<size_t> gBusyStreamSkip{0};
std::atomic<uint64_t> gBusyStreamsSkipped{0};

//! A pooled stream, skipping up to busyStreamSkip() streams that still have work pending. The last one taken is
//! returned even if it is busy.
CudaStreamPool::PoolItem acquireTemporaryStream()
{
    auto& pool = CudaStreamPool::instance();
    auto item = pool.get();
    size_t const maxBusy = gBusyStreamSkip.load(std::memory_order_relaxed);
    if (maxBusy == 0 || cuStreamQuery(item.get()) != CUDA_ERROR_NOT_READY)
    {
        return item;
    }
    // Skipped streams stay out of the pool until this returns, so none is offered twice; they go back to the pool's
    // tail when `busy` is destroyed. A stream that reports an error is not busy: its user sees the error.
    std::vector<CudaStreamPool::PoolItem> busy;
    do
    {
        busy.push_back(std::move(item));
        item = pool.get();
    } while (busy.size() < maxBusy && cuStreamQuery(item.get()) == CUDA_ERROR_NOT_READY);
    gBusyStreamsSkipped.fetch_add(busy.size(), std::memory_order_relaxed);
    return item;
}

} // namespace

void configureBusyStreamSkip(size_t maxBusy) noexcept
{
    if (maxBusy == 0)
    {
        return;
    }
    static std::atomic<size_t> configured{0};
    size_t expected = 0;
    if (!configured.compare_exchange_strong(expected, maxBusy))
    {
        if (expected != maxBusy)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2] copy streams already skip up to %zu busy streams in this process; ignoring a "
                "request for %zu",
                expected, maxBusy);
        }
        return;
    }
    gBusyStreamSkip.store(maxBusy, std::memory_order_relaxed);
    TLLM_LOG_INFO("[kvCacheManagerV2] copy streams skip up to %zu busy pooled streams", maxBusy);
}

size_t busyStreamSkip() noexcept
{
    return gBusyStreamSkip.load(std::memory_order_relaxed);
}

void setBusyStreamSkipForTesting(size_t maxBusy) noexcept
{
    gBusyStreamSkip.store(maxBusy, std::memory_order_relaxed);
}

uint64_t busyStreamsSkippedForTesting() noexcept
{
    return gBusyStreamsSkipped.load(std::memory_order_relaxed);
}

TemporaryCudaStream::TemporaryCudaStream(std::vector<CUevent> priorEvents)
    : mStream(acquireTemporaryStream())
{
    CudaStream cs = reinterpret_cast<CudaStream>(mStream.handle());
    streamWaitEvents(cs, std::move(priorEvents));
}

// ---------------------------------------------------------------------------
// mergeEvents — merge multiple CUDA events into one.
// Mirrors Python's merge_events() in _utils.py.
// ---------------------------------------------------------------------------

CachedCudaEvent mergeEvents(CachedCudaEvent* events, size_t count)
{
    // A single live event is returned as is rather than merged, so track one live event and
    // how many distinct ones there are.
    CachedCudaEvent* onlyLive = nullptr;
    size_t numLive = 0;
    auto const countLive = [&]()
    {
        onlyLive = nullptr;
        numLive = 0;
        for (size_t i = 0; i < count; ++i)
        {
            CUevent const handle = events[i].handle();
            if (handle != nullptr
                && std::none_of(events, events + i, [handle](auto const& prior) { return prior.handle() == handle; }))
            {
                ++numLive;
                onlyLive = &events[i];
            }
        }
    };
    countLive();
    if (numLive > 1)
    {
        // A merge costs a pooled stream, a wait per event and a record. Completed events order
        // nothing: dropping them leaves a released page with its owner's finish event alone.
        for (size_t i = 0; i < count; ++i)
            (void) events[i].queryComplete();
        countLive();
    }
    if (numLive == 0)
        return CachedCudaEvent::makeNull();
    if (numLive == 1)
        return std::move(*onlyLive);
    // Multiple live events: merge via TemporaryCudaStream. Closed events yield a null handle,
    // which streamWaitEvents drops.
    std::vector<CUevent> priors;
    priors.reserve(count);
    for (size_t i = 0; i < count; ++i)
        priors.push_back(events[i].handle());
    TemporaryCudaStream tempStream(std::move(priors));
    {
        auto scope = tempStream.enter();
    }
    return tempStream.takeFinishEvent();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
