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

#include "kv_cache_manager_v2/utils/poison.h"

#include "tensorrt_llm/common/logger.h"

#include <mutex>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

// Guards sReason. Held only when a violation is recorded or read, never on the poisoned() path.
std::mutex& reasonMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::string& reasonStorage()
{
    static std::string reason;
    return reason;
}

} // namespace

std::atomic<bool> Poison::sPoisoned{false};

void Poison::set(char const* context, char const* what) noexcept
{
    try
    {
        // Logged from a copy: clear() may run between releasing the mutex and the log, and it
        // mutates the stored string.
        std::string recorded;
        {
            std::lock_guard<std::mutex> lock(reasonMutex());
            if (sPoisoned.load(std::memory_order_relaxed))
            {
                // Already poisoned: this is cascade from the first violation, which is the one
                // worth keeping.
                return;
            }
            reasonStorage() = std::string(context != nullptr ? context : "<unknown context>") + ": "
                + (what != nullptr ? what : "<no message>");
            recorded = reasonStorage();
            sPoisoned.store(true, std::memory_order_release);
        }
        TLLM_LOG_ERROR("KVCM2 poisoned, refusing further work: %s", recorded.c_str());
    }
    catch (...)
    {
        // Allocation failed while recording the reason. The latch still has to be set, since
        // continuing is what this exists to prevent.
        sPoisoned.store(true, std::memory_order_release);
    }
}

std::optional<std::string> Poison::reason()
{
    if (!sPoisoned.load(std::memory_order_acquire))
    {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(reasonMutex());
    return reasonStorage();
}

void Poison::clear() noexcept
{
    try
    {
        std::lock_guard<std::mutex> lock(reasonMutex());
        reasonStorage().clear();
        sPoisoned.store(false, std::memory_order_release);
    }
    catch (...)
    {
        // Locking failed; leaving the latch set is the safe outcome.
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
