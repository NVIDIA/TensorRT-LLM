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
// Clearing is reachable only through takePoison(), declared below the class. Whether it is safe
// to clear is a question about live managers, which this class deliberately knows nothing about —
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
    // Drops the latch and the recorded reason. Private because it is only safe once nothing that
    // skipped its cleanup is still reachable; takePoison() enforces that.
    friend std::optional<std::string> takePoison() noexcept;
    static void clear() noexcept;

    static std::atomic<bool> sPoisoned;
};

// Reports the recorded violation and clears the latch, but only once no manager is alive: such a
// manager skipped its own cleanup when it was poisoned, so re-enabling it would resume work on
// structures known to be inconsistent. The reason is returned either way; when it cannot be
// cleared the latch stays set, so a following Poison::reason() still reports it.
std::optional<std::string> takePoison() noexcept;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
