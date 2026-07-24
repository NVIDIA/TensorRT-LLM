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

CachedCudaEvent::CachedCudaEvent(CudaStream stream)
    : mEvent(std::make_shared<CudaEventPool::PoolItem>(CudaEventPool::instance().get()))
{
    cuCheck(cuEventRecord(mEvent->get(), reinterpret_cast<CUstream>(stream)));
}

bool CachedCudaEvent::queryComplete()
{
    if (isClosed())
    {
        return true;
    }
    CUresult result = cuEventQuery(mEvent->get());
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

void CachedCudaEvent::synchronize()
{
    if (isClosed())
    {
        return;
    }
    cuCheck(cuEventSynchronize(mEvent->get()));
    close();
}

void CachedCudaEvent::waitInStream(CudaStream stream) const
{
    if (isClosed())
    {
        return;
    }
    cuCheck(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream), mEvent->get(), 0));
}

void CachedCudaEvent::close()
{
    if (mEvent)
    {
        mEvent->reset();
    }
}

// ---------------------------------------------------------------------------
// CachedCudaStream implementation
// ---------------------------------------------------------------------------

CachedCudaStream::CachedCudaStream()
    : mPoolItem(CudaStreamPool::instance().get())
{
}

CachedCudaEvent CachedCudaStream::recordEvent()
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

TemporaryCudaStream::TemporaryCudaStream(std::vector<CachedCudaEvent const*> const& priorEvents)
    : mStream()
{
    CudaStream cs = reinterpret_cast<CudaStream>(mStream.handle());
    streamWaitEvents(cs, priorEvents);
}

// ---------------------------------------------------------------------------
// mergeEvents — merge multiple CUDA events into one.
// Mirrors Python's merge_events() in _utils.py.
// ---------------------------------------------------------------------------

CachedCudaEvent mergeEvents(std::vector<CachedCudaEvent>& events)
{
    // Filter out closed events (optimization: skip cuStreamWaitEvent calls).
    std::vector<CachedCudaEvent*> live;
    for (auto& ev : events)
    {
        if (!ev.isClosed())
            live.push_back(&ev);
    }
    if (live.empty())
        return CachedCudaEvent::makeNull();
    if (live.size() == 1)
        return std::move(*live[0]);
    // Multiple live events: merge via TemporaryCudaStream.
    std::vector<CachedCudaEvent const*> priors;
    priors.reserve(live.size());
    for (auto* ev : live)
        priors.push_back(ev);
    TemporaryCudaStream tempStream(priors);
    {
        auto scope = tempStream.enter();
    }
    return tempStream.takeFinishEvent();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
