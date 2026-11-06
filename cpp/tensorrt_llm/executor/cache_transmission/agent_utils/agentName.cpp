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

#include "agentName.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unistd.h>

namespace tensorrt_llm::executor::kv_cache
{

std::string genUniqueAgentName()
{
    static std::atomic<uint64_t> counter{0};

    // Generate a per-process random suffix to disambiguate agents across containers
    // that may share the same hostname (--network host) and PID namespace.
    static uint64_t const sRandomSuffix = []()
    {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) | rd();
    }();

    constexpr std::size_t kHostnameBufferSize = 1024;
    char hostname[kHostnameBufferSize];
    gethostname(hostname, sizeof(hostname));
    auto const pid = static_cast<uint64_t>(::getpid());
    return std::string(hostname) + "_" + std::to_string(pid) + "_" + std::to_string(sRandomSuffix) + "_"
        + std::to_string(counter++);
}

} // namespace tensorrt_llm::executor::kv_cache
