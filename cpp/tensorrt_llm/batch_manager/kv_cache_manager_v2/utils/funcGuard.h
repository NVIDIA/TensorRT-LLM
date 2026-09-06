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

#include "kv_cache_manager_v2/exceptions.h"

#include <type_traits>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Generic RAII scope guard that calls a void() callable on destruction.
//! Callback failure during destruction terminates; explicit run() propagates it.
//! Movable (moved-from instance is disarmed). Not copyable.
template <typename F>
class FuncGuard
{
public:
    explicit FuncGuard(F&& func)
        : mFunc(std::forward<F>(func))
    {
    }

    ~FuncGuard() noexcept
    {
        terminateOnException("FuncGuard callback failed during destruction", [this]() { run(); });
    }

    FuncGuard(FuncGuard&& other) noexcept
        : mFunc(std::move(other.mFunc))
        , mActive(other.mActive)
    {
        other.mActive = false;
    }

    FuncGuard(FuncGuard const&) = delete;
    FuncGuard& operator=(FuncGuard const&) = delete;
    FuncGuard& operator=(FuncGuard&&) = delete;

    void cancel() noexcept
    {
        mActive = false;
    }

    void run()
    {
        if (std::exchange(mActive, false))
        {
            mFunc();
        }
    }

    [[nodiscard]] bool isActive() const noexcept
    {
        return mActive;
    }

private:
    std::decay_t<F> mFunc;
    bool mActive = true;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
