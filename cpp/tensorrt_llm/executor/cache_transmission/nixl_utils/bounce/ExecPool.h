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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include <cuda_runtime_api.h>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// ExecPool — gather/scatter execution contexts, decoupled from the data region
// ----------------------------------------------------------------------------
// With the variable-region arena, the data buffer a chunk uses is just a region (offset,len) held
// until ACK — a LONG lifetime. The CUDA resources needed to RUN one gather/scatter kernel
// (a stream, a completion event, device `scratch` for the plan arrays, and pinned `hostPinned`
// H2D staging) are only needed DURING the kernel — a SHORT lifetime. Bundling them per-region (as
// the old per-slot model did) would force one stream/scratch per concurrent region, which doesn't
// scale when the arena holds many small regions. So they live in a small fixed pool of E contexts,
// borrowed for one gather/scatter and returned when the kernel completes. E bounds GPU kernel
// concurrency (independent of region count); the sender's gather (IO thread) and the receiver's
// scatter (worker threads) both borrow here, so acquire/release are thread-safe.
// ============================================================================

struct ExecCtx
{
    std::uint32_t id{};
    cudaStream_t stream{nullptr};
    cudaEvent_t event{nullptr};   // gather-completion event (cudaEventRecord/Query, no blocking sync)
    void* scratch{nullptr};       // device plan arrays (srcs|dsts|sizes); unused when zeroCopyArgs
    void* hostPinned{nullptr};    // pinned staging for the plan arrays (H2D source, or read in-kernel)
    void* hostPinnedDev{nullptr}; // device-accessible alias of hostPinned (only set when zeroCopyArgs)
    void* cubTemp{nullptr};       // cub::DeviceMemcpy::Batched workspace (only set when cubCopy)
    std::size_t cubTempBytes{0};  // capacity of cubTemp
    std::size_t scratchBytes{0};
};

class ExecPool
{
public:
    /// Allocate `count` contexts, each with a stream/event + scratch/hostPinned sized for
    /// `maxDescsPerChunk` plan entries. Throws on CUDA allocation failure.
    /// @param zeroCopyArgs map hostPinned into the device address space so a kernel can read the plan
    ///        arrays directly (skip the H2D); @param cubCopy pre-allocate a cub batched-memcpy
    ///        workspace per context. Both default OFF (experimental — see BounceConfig).
    ExecPool(std::uint32_t count, std::size_t maxDescsPerChunk, int deviceId, bool zeroCopyArgs = false,
        bool cubCopy = false);
    ~ExecPool();

    ExecPool(ExecPool const&) = delete;
    ExecPool& operator=(ExecPool const&) = delete;

    /// Borrow a free context, or nullptr if all are in use (non-blocking — caller parks/retries,
    /// never blocks the IO thread). Thread-safe.
    [[nodiscard]] ExecCtx* tryAcquire();

    /// Return a context borrowed via tryAcquire(). Thread-safe.
    void release(ExecCtx* ctx);

    [[nodiscard]] std::uint32_t size() const noexcept
    {
        return static_cast<std::uint32_t>(mCtxs.size());
    }

    [[nodiscard]] std::size_t freeCount();

private:
    int mDeviceId{0};
    std::vector<ExecCtx> mCtxs;
    std::mutex mMu;
    std::deque<std::uint32_t> mFree;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
