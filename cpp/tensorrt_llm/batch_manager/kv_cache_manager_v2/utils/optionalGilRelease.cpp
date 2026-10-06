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

#include "kv_cache_manager_v2/utils/optionalGilRelease.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{
//! Written once, from the nanobind module initializer, before any object that could read it can
//! exist -- so a plain pointer is enough. It is never cleared, and must not become mutable at run
//! time without revisiting that.
GilHooks const* gGilHooks = nullptr;
} // namespace

void setGilHooks(GilHooks const* hooks) noexcept
{
    gGilHooks = hooks;
}

GilHooks const* gilHooks() noexcept
{
    return gGilHooks;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
