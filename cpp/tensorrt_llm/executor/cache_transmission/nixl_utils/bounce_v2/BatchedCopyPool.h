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

#pragma once

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce_v2/CompletionPoller.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include <cuda_runtime_api.h>

namespace tensorrt_llm::executor::kv_cache::bounce_v2
{

// ============================================================================
// BatchedCopyPool — gather/scatter copy engine (streams + plan buffers + events)
// ----------------------------------------------------------------------------
// Owns N CUDA streams at GREATEST priority (copies sit on the KV-transfer critical path but share
// the GPU with model kernels; at default priority a copy kernel queues behind them — measured
// ~260 µs avg / >1 ms tail vs ~65 µs of copy time), each bundled with a pinned zero-copy plan
// buffer and a completion event into a stream context.
//
// submitCopy() fills the plan arrays IN PLACE in the context's pinned buffer — splitting large
// runs into ≤64 KiB pieces so the one-thread-block-per-entry kernel keeps its grid-level
// parallelism — launches launchBatchedCopy() on the context's stream (the kernel reads the plan
// straight from pinned memory via the device alias; no H2D staging), records the event, and
// registers it with the CompletionPoller. The context returns to the free list from the poller
// thread when the event fires; Python only ever sees the returned completion id.
//
// Non-blocking contract: if every stream context is busy, submitCopy() returns kBusy (-1) and the
// caller retries on its next tick — it never blocks. CUDA errors throw.
// ============================================================================
class BatchedCopyPool
{
public:
    /// submitCopy() result when no stream context is free (caller retries later).
    static constexpr std::int64_t kBusy = -1;

    /// Allocate `numStreams` stream contexts, each with pinned plan buffers sized for
    /// `maxPlanEntries` (src,dst,size) entries. `poller` must outlive this pool (the Python binding
    /// enforces it via keep_alive). Throws on CUDA allocation failure.
    BatchedCopyPool(std::uint32_t numStreams, std::size_t maxPlanEntries, int deviceId, CompletionPoller& poller);
    ~BatchedCopyPool();

    BatchedCopyPool(BatchedCopyPool const&) = delete;
    BatchedCopyPool& operator=(BatchedCopyPool const&) = delete;

    /// Copy sizes[i] bytes from srcs[i] to dsts[i] for i in [0, n) as one batched kernel launch.
    /// srcs/dsts/sizes are HOST arrays of FINAL device addresses; n must be <= maxPlanEntries().
    /// Returns the CompletionPoller id of the recorded completion event, or kBusy when no stream
    /// context is free. Thread-safe. Throws on CUDA failure (the context is returned first).
    [[nodiscard]] std::int64_t submitCopy(
        std::uint64_t const* srcs, std::uint64_t const* dsts, std::uint32_t const* sizes, std::size_t n);

    [[nodiscard]] std::size_t maxPlanEntries() const noexcept
    {
        return mMaxPlanEntries;
    }

    [[nodiscard]] std::uint32_t size() const noexcept
    {
        return static_cast<std::uint32_t>(mCtxs.size());
    }

    [[nodiscard]] std::size_t freeCount();

private:
    struct Ctx
    {
        std::uint32_t id{};
        cudaStream_t stream{nullptr};
        cudaEvent_t event{nullptr};   // copy-completion event (recorded per submit, poller-queried)
        void* hostPinned{nullptr};    // pinned plan arrays [srcs(n) | dsts(n) | sizes(n)]
        void* hostPinnedDev{nullptr}; // device alias of hostPinned (kernel reads the plan in place)
    };

    [[nodiscard]] Ctx* tryAcquire();
    void release(Ctx* ctx);
    /// Free every allocated context resource (never throws; used by the destructor and the
    /// constructor's failure path).
    void destroyContexts();

    int mDeviceId{0};
    std::size_t mMaxPlanEntries{0};
    std::size_t mPlanBytes{0};
    CompletionPoller& mPoller;
    std::vector<Ctx> mCtxs;
    std::mutex mMu;
    std::deque<std::uint32_t> mFree;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
