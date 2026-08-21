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
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

namespace tensorrt_llm::executor::kv_cache::bounce_v2
{

// One raw scatter RUN exactly as it travels in a bounce_v2 DATA message (the Python codec's
// SCATTER_RUN_DTYPE / the v2 C++ BounceScatterRun): `count` equal pieces of `pieceSize` bytes;
// piece p copies region[bounceOffset + p*bounceStride ...] to dstAddr + p*dstStride. Packed
// 36-byte little-endian wire layout — submitScatterRuns() consumes the DATA payload directly.
#pragma pack(push, 1)

struct ScatterRunWire
{
    std::uint64_t bounceOffset;
    std::uint64_t dstAddr;
    std::uint64_t dstStride;
    std::uint32_t bounceStride;
    std::uint32_t pieceSize;
    std::uint32_t count;
};

#pragma pack(pop)
static_assert(sizeof(ScatterRunWire) == 36, "ScatterRunWire must match the 36-byte wire layout");

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
    /// submitScatterRuns() result when the runs failed bounds/size validation (details logged).
    /// The caller must NOT ack the chunk (the data did not land) but should release the region.
    static constexpr std::int64_t kScatterRejected = -2;

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
    ///
    /// `reserveChainId` (optional, gather->RDMA chain): forwarded to
    /// CompletionPoller::registerEvent so the chain reservation is taken ATOMICALLY with the event
    /// registration (guaranteed — never loses the race to the poll sweep). 0 when unavailable
    /// (poller shut down); untouched on kBusy/throw.
    [[nodiscard]] std::int64_t submitCopy(std::uint64_t const* srcs, std::uint64_t const* dsts,
        std::uint32_t const* sizes, std::size_t n, std::uint64_t* reserveChainId = nullptr);

    /// Register one sender request's ENTIRE gather plan (per-request plan handle): flat per-desc
    /// arrays over all chunks, sliced by `chunkStarts` ([nChunks + 1] monotonic desc-index
    /// boundaries; chunkStarts[0] == 0, chunkStarts[nChunks] == nDescs). `bounceOffsets` are
    /// REGION-RELATIVE staging offsets — the absolute destination is stagingBase + offset,
    /// resolved per launchChunk() call, so the registered plan never references arena regions.
    /// The arrays are copied into pool-owned memory (the caller's buffers may be freed after this
    /// returns). Returns the plan handle. Thread-safe. Throws std::invalid_argument on malformed
    /// boundaries or a chunk exceeding maxPlanEntries().
    ///
    /// Lifecycle: the caller frees a handle explicitly via releasePlan() on the request's
    /// terminal paths; every remaining plan is dropped at pool destruction. A launchChunk()
    /// concurrent with releasePlan() is safe (the launch pins the plan via shared ownership);
    /// a launch AFTER release throws std::invalid_argument (unknown handle) deterministically.
    [[nodiscard]] std::uint64_t registerPlan(std::uint64_t const* srcs, std::uint64_t const* bounceOffsets,
        std::uint32_t const* sizes, std::uint64_t const* chunkStarts, std::size_t nDescs, std::size_t nChunks);

    /// Drop a registered plan (idempotent for unknown handles).
    void releasePlan(std::uint64_t handle);

    /// Launch ONE chunk of a registered plan: gather chunk `chunkIdx`'s descs from their source
    /// addresses to stagingBase + bounceOffsets (with the same <= 64 KiB run splitting as
    /// submitCopy), entirely from the pre-marshalled plan — no per-call array marshalling.
    /// Returns the CompletionPoller id of the completion event, or kBusy. `reserveChainId` as in
    /// submitCopy. Thread-safe. Throws std::invalid_argument on an unknown handle or chunk index,
    /// and on CUDA failure like submitCopy.
    [[nodiscard]] std::int64_t launchChunk(
        std::uint64_t handle, std::size_t chunkIdx, std::uint64_t stagingBase, std::uint64_t* reserveChainId = nullptr);

    /// Receiver-side scatter of one DATA chunk in ONE call: validate the RAW wire runs against the
    /// granted region, expand them to per-piece copies (with the same <= 64 KiB run splitting as
    /// submitCopy) straight into a context's pinned plan buffer, launch, and register the
    /// completion event. Replaces the Python reactor's per-chunk numpy run expansion
    /// (_validate_and_expand_runs + submit_copy), whose interpreter cost dominated the receiver's
    /// onData path.
    ///
    /// `regionBase` is the ABSOLUTE device address of the flow's granted arena region and
    /// `regionBytes` its granted length; the CALLER must already have verified the region lies
    /// inside the registered arena (the Python reactor's check (1) — it owns the arena bounds).
    /// Validation here mirrors the Python fallback's remaining checks EXACTLY (at least as
    /// strict; see submitScatterRuns in BatchedCopyPool.cpp for the check-by-check mapping).
    ///
    /// Returns the CompletionPoller id of the recorded completion event, kBusy when no stream
    /// context is free (retry later — validation is repeated, it is deterministic), or
    /// kScatterRejected when validation failed (logged; never launches, never takes a context).
    /// Thread-safe. Throws on CUDA failure (the context is returned first).
    [[nodiscard]] std::int64_t submitScatterRuns(
        std::uint64_t regionBase, std::uint64_t regionBytes, ScatterRunWire const* runs, std::size_t nRuns);

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

    /// One registered per-request gather plan (see registerPlan). Immutable after construction;
    /// held by shared_ptr so an in-flight launchChunk survives a concurrent releasePlan.
    struct RequestPlan
    {
        std::vector<std::uint64_t> srcs;
        std::vector<std::uint64_t> bounceOffsets;
        std::vector<std::uint32_t> sizes;
        std::vector<std::uint64_t> chunkStarts;
    };

    [[nodiscard]] Ctx* tryAcquire();
    void release(Ctx* ctx);
    /// Shared launch body of submitCopy/launchChunk: dst[i] = dstBase + dstOffsets[i] (submitCopy
    /// passes absolute dsts with dstBase 0).
    [[nodiscard]] std::int64_t launchGather(std::uint64_t const* srcs, std::uint64_t const* dstOffsets,
        std::uint64_t dstBase, std::uint32_t const* sizes, std::size_t n, std::uint64_t* reserveChainId);
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

    std::mutex mPlanMu; // guards mPlans / mNextPlanId (separate from mMu: plan lookups must not
                        // serialize behind context acquire/release)
    std::uint64_t mNextPlanId{1};
    std::unordered_map<std::uint64_t, std::shared_ptr<RequestPlan const>> mPlans;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
