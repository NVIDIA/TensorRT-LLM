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

#include <cstdint>
#include <optional>
#include <string>

namespace tensorrt_llm::nanobind::visual_gen
{

//! Start a detached native thread that terminates this process when its coordinator exits.
//! Returns a warning when pidfd_open is unavailable and parent-PID polling is used instead.
std::optional<std::string> startCoordinatorWatchdog(std::int64_t coordinatorPid);

namespace testing
{

//! Exercise the pidfd_open error path without changing the process seccomp policy.
std::optional<std::string> startCoordinatorWatchdogWithPidfdError(std::int64_t coordinatorPid, int pidfdErrorCode);

} // namespace testing

} // namespace tensorrt_llm::nanobind::visual_gen
