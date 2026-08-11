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

#include "tensorrt_llm/common/nvtxUtils.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>

#ifndef NVTX_DISABLE
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace tensorrt_llm::executor::kv_cache::bounce
{

// NVTX instrumentation for the bounce pipeline (perf analysis with nsys). Everything here
// compiles to a no-op when the build defines NVTX_DISABLE (the default; build with
// -DNVTX_DISABLE=OFF to profile).
//
// Two kinds of spans:
//   - BounceNvtxScope: same-thread RAII push/pop, for synchronous sections (buildPlan, the
//     gather launch, the scatter kernel + sync in a worker).
//   - bounceRangeStart()/bounceRangeEnd(): process-wide start/end ranges for the ASYNC legs —
//     started on one thread and ended on another (gather in flight: IO-thread launch -> IO-thread
//     event poll later; RDMA write: post -> poll Done; ACK wait: DATA sent -> ACK received;
//     scatter queueing: IO-thread enqueue -> worker dequeue). The handle is a plain uint64 so the
//     reactor structs that carry it across threads need no NVTX include.

/// Dedicated domain: bounce ranges get their own row group in nsys instead of mixing with the
/// global TRT-LLM ranges.
struct BounceNvtxDomain
{
    static constexpr char const* name{"trtllm.disagg.bounce"};
};

// Span colors (ARGB), one per pipeline stage so the nsys timeline reads at a glance.
inline constexpr std::uint32_t kNvtxBuildPlan = 0xFF9E9E9EU;     // gray
inline constexpr std::uint32_t kNvtxRequest = 0xFF2196F3U;       // blue: submit -> resolve
inline constexpr std::uint32_t kNvtxGrantWait = 0xFFFF9800U;     // orange: WANT sent -> first GRANT
inline constexpr std::uint32_t kNvtxGatherLaunch = 0xFF8BC34AU;  // light green
inline constexpr std::uint32_t kNvtxGather = 0xFF4CAF50U;        // green: gather launched -> event done
inline constexpr std::uint32_t kNvtxNixlWrite = 0xFF3F51B5U;     // indigo: postWrite -> poll Done
inline constexpr std::uint32_t kNvtxAckWait = 0xFFE91E63U;       // pink: DATA sent -> ACK
inline constexpr std::uint32_t kNvtxScatterQueue = 0xFFFFC107U;  // dark yellow: enqueue -> worker dequeue
inline constexpr std::uint32_t kNvtxScatter = 0xFFFFEB3BU;       // yellow: scatter kernel + sync
inline constexpr std::uint32_t kNvtxCreditStarved = 0xFFFF5722U; // deep orange: out of credits -> next GRANT
inline constexpr std::uint32_t kNvtxArenaStarved = 0xFF795548U;  // brown: credits parked on local arena/exec

// Fine-grained control-path spans decomposing ackWait (DATA sent -> ACK received). Together with
// the coarse spans these localize where the ACK round-trip actually goes: sender-side DATA
// build/enqueue, wire+peer time (the ackWait residual), receiver decode, scatter prep vs the real
// GPU wait, ACK enqueue, and the IO-thread bookkeeping drain.
inline constexpr std::uint32_t kNvtxDataSend = 0xFF00BCD4U;    // cyan: build entries + encode + zmq enqueue of DATA
inline constexpr std::uint32_t kNvtxOnData = 0xFF009688U;      // teal: DATA decode + scatter-job enqueue (receiver IO)
inline constexpr std::uint32_t kNvtxScatterPrep = 0xFFCDDC39U; // lime: scatter plan-array build + kernel launch
inline constexpr std::uint32_t kNvtxScatterSync = 0xFFB2A429U; // dark lime: cudaStreamSynchronize (the GPU wait)
inline constexpr std::uint32_t kNvtxAckSend = 0xFFF06292U;     // light pink: encode + zmq enqueue of ACK (worker)
inline constexpr std::uint32_t kNvtxOnAck = 0xFFAD1457U;       // dark pink: ACK dispatch incl. mReqMu wait (sender IO)
inline constexpr std::uint32_t kNvtxDoneDrain = 0xFF607D8BU;   // blue gray: drainScatterDone region bookkeeping
inline constexpr std::uint32_t kNvtxZmqSend = 0xFF9C27B0U;     // purple: zmq sendTo (lock + msg copy + enqueue)
inline constexpr std::uint32_t kNvtxZmqRecv = 0xFF673AB7U;     // deep purple: zmq frame reads + blob copies

/// Scoped range with a printf-formatted message, e.g.
/// `BounceNvtxScope s(kNvtxScatter, "scatter rid=%llu chunk=%u", rid, chunk);`
/// RAII (begins on construction, ends on destruction — early-exit safe), but implemented with
/// start/end ranges rather than push/pop: start/end is PROCESS-scoped, so these spans land in the
/// single `trtllm.disagg.bounce` domain row in nsys-ui together with the async spans, instead of
/// being scattered across per-thread rows (push/pop is thread-scoped and only shows under the
/// pushing thread, which for scatter workers is easy to miss).
class BounceNvtxScope
{
public:
    BounceNvtxScope(BounceNvtxScope const&) = delete;
    BounceNvtxScope& operator=(BounceNvtxScope const&) = delete;

#ifndef NVTX_DISABLE
    BounceNvtxScope(std::uint32_t argb, char const* fmt, ...)
    {
        char msg[128];
        std::va_list args;
        va_start(args, fmt);
        std::vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        mHandle = ::nvtx3::start_range_in<BounceNvtxDomain>(msg, ::nvtx3::color{argb});
    }

    ~BounceNvtxScope()
    {
        ::nvtx3::end_range_in<BounceNvtxDomain>(mHandle);
    }

private:
    ::nvtx3::range_handle mHandle{};
#else
    BounceNvtxScope(std::uint32_t /*argb*/, char const* /*fmt*/, ...) {}
#endif
};

/// Start a cross-thread span. Returns an opaque handle (0 == no range, e.g. NVTX disabled);
/// end it — possibly on a different thread — with bounceRangeEnd().
#ifndef NVTX_DISABLE
inline std::uint64_t bounceRangeStart(std::uint32_t argb, char const* fmt, ...)
{
    char msg[128];
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    return ::nvtx3::start_range_in<BounceNvtxDomain>(msg, ::nvtx3::color{argb}).get_value();
}
#else
inline std::uint64_t bounceRangeStart(std::uint32_t /*argb*/, char const* /*fmt*/, ...)
{
    return 0;
}
#endif

/// Label the calling thread in profiler timelines (e.g. "bounceIO", "bounceScatter"), so the
/// bounce threads are identifiable rows in nsys-ui — PushPop ranges (like the scatter span)
/// appear under the thread that pushed them, which is otherwise an anonymous worker.
inline void bounceNameThread(char const* name)
{
#ifndef NVTX_DISABLE
    nvtxNameOsThreadA(static_cast<std::uint32_t>(::syscall(SYS_gettid)), name);
#else
    (void) name;
#endif
}

/// End a span started by bounceRangeStart() and zero the handle. Safe on a 0 handle (no-op),
/// so failure paths can end every handle unconditionally.
inline void bounceRangeEnd(std::uint64_t& handle)
{
#ifndef NVTX_DISABLE
    if (handle != 0)
    {
        ::nvtx3::end_range_in<BounceNvtxDomain>(::nvtx3::range_handle{handle});
    }
#endif
    handle = 0;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
