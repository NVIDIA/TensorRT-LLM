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

#include <mutex>

namespace tensorrt_llm::runtime
{

// NCCL host APIs must not run concurrently from different threads for
// communicators associated with the same CUDA device. TRT-LLM can own both a
// PP communicator and several cached raw-op communicators, each with its own
// watchdog, so a per-communicator mutex is insufficient. A process-wide gate
// is deliberately stricter than a per-device gate and avoids relying on CUDA
// thread-local device state in watchdog threads.
//
// Recursive locking lets low-level error paths abort a communicator while the
// operation wrapper already owns the gate. A grouped operation keeps one lock
// alive from ncclGroupStart through ncclGroupEnd.
inline std::recursive_mutex& getNcclHostApiMutex()
{
    // Communicator registries and watchdogs are also process-lifetime
    // singletons, with cross-translation-unit destruction order unspecified.
    // Keep the gate alive until process exit so late communicator teardown can
    // never touch an already-destroyed mutex.
    static auto* mutex = new std::recursive_mutex;
    return *mutex;
}

using NcclHostApiLock = std::unique_lock<std::recursive_mutex>;

inline NcclHostApiLock acquireNcclHostApiLock()
{
    return NcclHostApiLock{getNcclHostApiMutex()};
}

} // namespace tensorrt_llm::runtime
