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
    KVCM2_ABORT_ON_EXCEPT(
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
