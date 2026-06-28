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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ExecPool.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::executor::kv_cache::bounce
{

ExecPool::ExecPool(std::uint32_t count, std::size_t maxDescsPerChunk, int deviceId)
    : mDeviceId(deviceId)
{
    TLLM_CHECK_WITH_INFO(count > 0, "ExecPool: count must be > 0");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
    std::size_t const scratchBytes = maxDescsPerChunk * (2 * sizeof(std::uint64_t) + sizeof(std::uint32_t));
    mCtxs.resize(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        auto& c = mCtxs[i];
        c.id = i;
        c.scratchBytes = scratchBytes;
        TLLM_CUDA_CHECK(cudaMalloc(&c.scratch, scratchBytes));
        TLLM_CUDA_CHECK(cudaHostAlloc(&c.hostPinned, scratchBytes, cudaHostAllocDefault));
        TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&c.event, cudaEventDisableTiming));
        mFree.push_back(i);
    }
}

ExecPool::~ExecPool()
{
    // Select the device the resources live on before freeing — otherwise on a multi-GPU process the
    // cudaFree/cudaStreamDestroy/cudaEventDestroy target whatever device is current on this thread.
    // A failure here means the frees below would leak on the wrong device, so warn (a dtor can't
    // throw). The frees themselves stay best-effort: nothing to recover at teardown.
    if (cudaSetDevice(mDeviceId) != cudaSuccess)
    {
        (void) cudaGetLastError();
        TLLM_LOG_WARNING("ExecPool::~ExecPool: cudaSetDevice(%d) failed; contexts may leak", mDeviceId);
    }
    for (auto& c : mCtxs)
    {
        if (c.scratch != nullptr)
            (void) cudaFree(c.scratch);
        if (c.hostPinned != nullptr)
            (void) cudaFreeHost(c.hostPinned);
        if (c.stream != nullptr)
            (void) cudaStreamDestroy(c.stream);
        if (c.event != nullptr)
            (void) cudaEventDestroy(c.event);
    }
}

ExecCtx* ExecPool::tryAcquire()
{
    std::lock_guard<std::mutex> lk(mMu);
    if (mFree.empty())
    {
        return nullptr;
    }
    std::uint32_t const id = mFree.front();
    mFree.pop_front();
    return &mCtxs[id];
}

void ExecPool::release(ExecCtx* ctx)
{
    if (ctx == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lk(mMu);
    mFree.push_back(ctx->id);
}

std::size_t ExecPool::freeCount()
{
    std::lock_guard<std::mutex> lk(mMu);
    return mFree.size();
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
