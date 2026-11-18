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

#include <string>

namespace tensorrt_llm::executor::kv_cache
{

//! Generate a unique agent name for NIXL/UCX connection identity.
//! Format: {hostname}_{pid}_{random64}_{counter}
//! The per-process random suffix prevents collisions across Docker containers
//! that share hostname (--network host) and PID namespace.
std::string genUniqueAgentName();

} // namespace tensorrt_llm::executor::kv_cache
