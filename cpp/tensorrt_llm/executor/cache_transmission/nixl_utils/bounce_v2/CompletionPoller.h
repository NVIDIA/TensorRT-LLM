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

#include "tensorrt_llm/executor/transferAgent.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

namespace tensorrt_llm::executor::kv_cache::bounce_v2
{

// ============================================================================
// CompletionPoller — the ONE polling thread for all async bounce completions
// ----------------------------------------------------------------------------
// Owns a ~50 µs poll loop over (a) CUDA events (gather/scatter kernel completion — the events
// themselves stay owned by BatchedCopyPool, which guarantees an event is not reused or destroyed
// while registered here) and (b) RDMA TransferStatus handles (per-chunk writes posted through
// NixlTransferAgent::postXferRequest). Terminal states are pushed as {id, kind, ok} completions
// into a mutex-guarded vector that Python drains in ONE batch (one GIL acquisition per drain,
// instead of one C→Python transition per completion).
//
// TransferStatus handles are released inside C++ on terminal state; if release() fails the handle
// is retired and its release retried by the destructor (the object is never leaked to Python).
// Threading: registerEvent/registerXfer are safe from any thread; drain() is intended for the
// single Python reactor thread but is also thread-safe. shutdown() is idempotent; pending entries
// are terminated with ok=0 so no drain() caller waits forever.
// ============================================================================
class CompletionPoller
{
public:
    static constexpr std::int64_t kKindEvent = 0; // CUDA event (gather/scatter)
    static constexpr std::int64_t kKindXfer = 1;  // RDMA transfer

    // fulfillChain() results (see fulfillChain).
    static constexpr std::int64_t kFulfillDeclined = 0; // terminal row already published/pending; do NOT post
    static constexpr std::int64_t kFulfillArmed = 1;    // poster attached; posts when the gather event completes
    static constexpr std::int64_t kFulfillPosted = 2;   // gather already done; the write was posted inline

    struct Completion
    {
        std::uint64_t id;
        std::int64_t kind; // kKindEvent / kKindXfer
        std::int64_t ok;   // 1 = success, 0 = failure (CUDA error / FAILURE state / shutdown)
    };

    /// Starts the poll thread. `pollIntervalUs` is the sleep between poll sweeps.
    explicit CompletionPoller(std::uint32_t pollIntervalUs = 50);
    ~CompletionPoller();

    CompletionPoller(CompletionPoller const&) = delete;
    CompletionPoller& operator=(CompletionPoller const&) = delete;

    /// Track `event` (owner-managed: the caller keeps it alive and unreused until the completion is
    /// reported). `onTerminal` (optional) runs on the poll thread right before the completion is
    /// published — BatchedCopyPool uses it to return the stream context to its free list, so exec
    /// contexts are recycled in C++ without waiting for Python to drain. Returns the completion id.
    ///
    /// `reserveChainId` (optional, gather->RDMA chain): take a chain reservation ATOMICALLY with
    /// the registration — in the same critical section, before any poll sweep can see the event —
    /// so it can never lose the race to the event's own completion. From then on the event's own
    /// completion is consumed in C++ and the chunk's ONE terminal row publishes under the reserved
    /// id (see fulfillChain for the row contract). On success *reserveChainId is the reserved id;
    /// when the poller is already shut down it is 0 (no reservation; the event id itself resolves
    /// classically).
    [[nodiscard]] std::uint64_t registerEvent(
        cudaEvent_t event, std::function<void()> onTerminal = {}, std::uint64_t* reserveChainId = nullptr);

    /// Take ownership of an RDMA transfer handle and poll it with wait(0). On terminal state the
    /// handle is release()d (kept for a dtor retry if release fails). Returns the completion id.
    [[nodiscard]] std::uint64_t registerXfer(std::unique_ptr<TransferStatus> status);

    /// Chain poster: posts the armed RDMA write; runs on the POLL thread (no GIL, no poller lock).
    /// Returns the transfer handle, or nullptr when the post failed (the reserved id is then
    /// published with kind=kKindXfer, ok=0). The referenced agent must outlive the poller's armed
    /// chains — guaranteed by the BounceEngine teardown order (reactor -> poller.shutdown() ->
    /// agent) and belt-and-braces by the binding's keep_alive.
    using ChainPoster = std::function<std::unique_ptr<TransferStatus>()>;

    /// Attach the RDMA poster to a registerEvent(reserveChainId) reservation (the C++-internal
    /// half of the pool/agent binding's launch_chunk_chained; not exposed to Python on its own).
    /// The reserved id carries the chunk's ONE terminal row:
    ///   {reservedId, kKindXfer, ok}       write posted and reached a terminal state
    ///   {reservedId, kKindXfer, 0}        post failed / shutdown before the write resolved
    ///   {reservedId, kKindEvent, 0}       the gather event itself failed (write never posted)
    /// Return codes:
    ///   kFulfillArmed    gather still pending -> the poll thread posts when the event fires;
    ///   kFulfillPosted   gather already done OK -> the write was posted INLINE on the calling
    ///                    thread (executeChains path, mChainsInFlight accounted) and its handle is
    ///                    now polled under the reserved id;
    ///   kFulfillDeclined a terminal row for the reserved id is already published or pending
    ///                    (gather failed, shutdown swept it, or the chain was already fulfilled) —
    ///                    do NOT post; just wait for that row.
    /// Every outcome preserves exactly ONE terminal row per reserved id. Thread-safe.
    [[nodiscard]] std::int64_t fulfillChain(std::uint64_t reservedId, ChainPoster poster);

    /// Return (and clear) ALL pending completions, blocking up to `timeoutMs` for the first one
    /// (0 = non-blocking; may return empty). Never blocks after shutdown().
    [[nodiscard]] std::vector<Completion> drain(int timeoutMs);

    /// Remove every still-pending event entry whose event handle is in `events`, WITHOUT running
    /// its onTerminal or publishing a completion (the ids simply never report — acceptable only at
    /// the event owner's teardown, e.g. ~BatchedCopyPool detaching before destroying its events).
    /// An ARMED or RESERVED chain on a removed entry is dropped the same way: its reserved id never reports
    /// either (consistent with the entry's own id; a Python waiter unblocks via the reactor's
    /// request-timeout sweep / shutdown failAll / stall watchdog, never by this id).
    /// Synchronization is two-part: (1) taking the internal mutex synchronizes with the poll
    /// sweep, so no removed entry is being queried or called back concurrently; (2) because chain
    /// POSTERS run OUTSIDE that mutex by design, this additionally waits — bounded, GIL-free — for
    /// every in-flight poster to finish (mChainsInFlight == 0) before returning, so the caller may
    /// destroy anything a poster could still reference. Never blocks on anything that needs the
    /// GIL (the poll thread never takes it), so it is safe from a nanobind tp_dealloc. Returns the
    /// number of entries removed.
    std::size_t unregisterEvents(std::vector<cudaEvent_t> const& events);

    /// Stop the poll thread and terminate every still-pending entry with ok=0 (running its
    /// onTerminal / releasing its handle). Idempotent.
    void shutdown() noexcept;

private:
    struct EventEntry
    {
        std::uint64_t id;
        cudaEvent_t event;
        std::function<void()> onTerminal;
        // 0 = no chain. chainId != 0 with a poster = ARMED (fulfilled reservation); chainId != 0
        // WITHOUT a poster = RESERVED (registerEvent's reserveChainId), awaiting fulfillChain.
        std::uint64_t chainId{0};
        ChainPoster chainPoster;
    };

    /// A chain whose event completed OK, awaiting its post (runs outside mMu on the poll thread).
    struct PendingChain
    {
        std::uint64_t id;
        ChainPoster poster;
    };

    struct XferEntry
    {
        std::uint64_t id;
        std::unique_ptr<TransferStatus> status;
    };

    void pollLoop();
    /// Poll every pending entry once; completed ones move to mDone, and successfully-completed
    /// events with an armed chain move their poster into `chainsOut` (nothing published for them
    /// yet). Called under mMu. Returns true if any completion was published.
    bool pollOnceLocked(std::vector<PendingChain>& chainsOut);
    /// Post every collected chain OUTSIDE mMu (postXferRequest takes the agent lock and can take
    /// tens of us; drain()/register*() must not stall behind it), then either enqueue the handle
    /// for polling under the reserved id or publish {reservedId, kKindXfer, 0} on failure.
    void executeChains(std::vector<PendingChain>& chains);
    /// Release a terminal xfer handle; retire it for a dtor retry if release fails. Under mMu.
    void releaseXferLocked(XferEntry& entry);

    std::uint32_t const mPollIntervalUs;

    std::mutex mMu; // guards mEvents / mXfers / mDone / mRetired / mChainsInFlight /
                    // mGatherDoneChains
    std::condition_variable mCv;
    std::vector<EventEntry> mEvents;
    // Reserved chain ids whose gather event completed OK before fulfillChain arrived (nothing
    // published for them yet; fulfillChain posts inline, shutdown terminates them).
    std::vector<std::uint64_t> mGatherDoneChains;
    // Chain posters currently executing OUTSIDE mMu on the poll thread (incremented under mMu when
    // pollOnceLocked hands a chain out, decremented under mMu as executeChains finishes each one).
    // unregisterEvents waits for zero so event owners can tear down safely.
    std::size_t mChainsInFlight{0};
    std::vector<XferEntry> mXfers;
    std::vector<Completion> mDone;
    // release() failed on these; retried (best effort) at destruction.
    std::vector<std::unique_ptr<TransferStatus>> mRetired;

    std::atomic<std::uint64_t> mNextId{1};
    std::atomic<bool> mStop{false};
    std::mutex mShutdownMu; // serializes shutdown() callers
    bool mShutdownDone{false};
    std::thread mThread;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
