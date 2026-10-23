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

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Process-wide "this cache is no longer trustworthy" latch.
//
// A broken invariant is detected in places that cannot throw — destructors, noexcept functions —
// so the failure is recorded here instead. Once set, KVCM2 refuses further work: public entry
// points raise, and destructors skip their cleanup rather than act on structures that are known
// to be inconsistent. Skipping leaks slots and pages, which is the safe direction; the manager
// will never serve another request.
//
// The latch is process-wide because the objects that detect violations — Slot, SlotAllocator,
// the singleton pools — have no route back to an owning manager.
//
// Clearing is reachable only through takePoison(), declared below the class. Whether it is safe to
// clear is a question about live PoisonHolds, which this class deliberately knows nothing about —
// hence the friendship rather than a check here.
class Poison
{
public:
    // Hot path. Relaxed because the latch carries no data of its own: reason() takes the mutex.
    [[nodiscard]] static bool poisoned() noexcept
    {
        return sPoisoned.load(std::memory_order_relaxed);
    }

    // Records the first violation and ignores later ones, which are cascade from the first.
    static void set(char const* context, char const* what) noexcept;

    // The first recorded violation, or nullopt. Never clears, so it is safe to poll.
    [[nodiscard]] static std::optional<std::string> reason();

private:
    // Dropping the latch is not offered as an operation: it is only safe once nothing that skipped
    // its cleanup is still reachable, and it must happen under the same lock as the decision to do
    // it. takePoison() is the one place that holds that lock, so it is the one place that can.
    friend std::optional<std::string> takePoison() noexcept;

    static std::atomic<bool> sPoisoned;
};

// Registers its owner for as long as it exists, and blocks clearing of the latch meanwhile: an
// owner that was alive when a violation was recorded skipped its own cleanup, so re-enabling it
// would resume work on structures known to be inconsistent.
//
// Registration is serialized with the latch itself, so an owner that appears concurrently with
// takePoison() either blocks the clear or starts after it has completed; the two can never
// interleave into an owner running on a cleared latch it did not observe.
class PoisonHold
{
public:
    PoisonHold() noexcept;
    ~PoisonHold();

    PoisonHold(PoisonHold const&) = delete;
    PoisonHold& operator=(PoisonHold const&) = delete;
    PoisonHold(PoisonHold&&) = delete;
    PoisonHold& operator=(PoisonHold&&) = delete;

    //! Number of registrations currently live.
    [[nodiscard]] static uint32_t count() noexcept;
};

// Reports the recorded violation and clears the latch, but only once no PoisonHold is live. The
// reason is returned either way; when it cannot be cleared the latch stays set, so a following
// Poison::reason() still reports it.
std::optional<std::string> takePoison() noexcept;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
