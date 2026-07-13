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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/GatherScatterKernel.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::executor::kv_cache::bounce
{

ExecPool::ExecPool(std::uint32_t count, std::size_t maxDescsPerChunk, int deviceId, bool zeroCopyArgs, bool cubCopy)
    : mDeviceId(deviceId)
{
    TLLM_CHECK_WITH_INFO(count > 0, "ExecPool: count must be > 0");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
    std::size_t const scratchBytes = maxDescsPerChunk * (2 * sizeof(std::uint64_t) + sizeof(std::uint32_t));
    // cub workspace size depends only on the buffer count -> size it once for the worst case (maxDescs);
    // per-call we re-query for the actual n (<= maxDescs, so <= this) and validate it fits.
    std::size_t cubTempBytes = 0;
    if (cubCopy)
    {
        TLLM_CUDA_CHECK(batchedCopyCubTempBytes(static_cast<std::uint32_t>(maxDescsPerChunk), cubTempBytes));
    }
    // Gather/scatter copies sit on the KV-transfer critical path but share the GPU with model
    // kernels (prefill/decode). At default priority a copy kernel queues behind those (measured
    // ~260us avg, >1ms tail, vs ~65us of actual copy time). Greatest priority lets its blocks be
    // scheduled as soon as any SM frees up, without preempting running work.
    int leastPriority = 0;
    int greatestPriority = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    mCtxs.resize(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        auto& c = mCtxs[i];
        c.id = i;
        c.scratchBytes = scratchBytes;
        TLLM_CUDA_CHECK(cudaMalloc(&c.scratch, scratchBytes));
        // zeroCopyArgs: map the pinned buffer into the device address space so the kernel can read the
        // plan arrays in place (no H2D). cudaHostAllocMapped is otherwise a normal pinned buffer.
        TLLM_CUDA_CHECK(
            cudaHostAlloc(&c.hostPinned, scratchBytes, zeroCopyArgs ? cudaHostAllocMapped : cudaHostAllocDefault));
        if (zeroCopyArgs)
        {
            TLLM_CUDA_CHECK(cudaHostGetDevicePointer(&c.hostPinnedDev, c.hostPinned, 0));
        }
        if (cubCopy && cubTempBytes > 0)
        {
            TLLM_CUDA_CHECK(cudaMalloc(&c.cubTemp, cubTempBytes));
            c.cubTempBytes = cubTempBytes;
        }
        TLLM_CUDA_CHECK(cudaStreamCreateWithPriority(&c.stream, cudaStreamNonBlocking, greatestPriority));
        TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&c.event, cudaEventDisableTiming));
        mFree.push_back(i);
    }
}

ExecPool::~ExecPool()
{
    // Select the device the resources live on before freeing — otherwise on a multi-GPU process the
    // cudaFree/cudaStreamDestroy/cudaEventDestroy target whatever device is current on this thread.
    // Select the owning device before freeing (multi-GPU: otherwise these target the thread's current
    // device). A dtor can't throw, so use the project's warn-only cleanup check on every teardown call
    // rather than discarding the result (matches tllmBuffers / cudaMemPool).
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mDeviceId));
    for (auto& c : mCtxs)
    {
        if (c.scratch != nullptr)
            TLLM_CUDA_CHECK_WARN(cudaFree(c.scratch));
        if (c.cubTemp != nullptr)
            TLLM_CUDA_CHECK_WARN(cudaFree(c.cubTemp));
        if (c.hostPinned != nullptr)
            TLLM_CUDA_CHECK_WARN(cudaFreeHost(c.hostPinned)); // also frees its hostPinnedDev alias
        if (c.stream != nullptr)
            TLLM_CUDA_CHECK_WARN(cudaStreamDestroy(c.stream));
        if (c.event != nullptr)
            TLLM_CUDA_CHECK_WARN(cudaEventDestroy(c.event));
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
