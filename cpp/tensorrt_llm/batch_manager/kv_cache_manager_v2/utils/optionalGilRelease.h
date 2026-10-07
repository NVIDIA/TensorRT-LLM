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

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Lets the core release the GIL without depending on the Python C API.
//!
//! Motivation: destructors. A binding method can declare `nb::call_guard<nb::gil_scoped_release>`,
//! but a *destructor* has no such hook: nanobind runs it from `tp_dealloc` with the GIL held, which
//! is exactly the state that deadlocks against a thread holding the API lock and waiting on
//! `gil_scoped_acquire` to run a Python callback. Objects here are also destroyed on pure-C++ paths
//! where there may be no interpreter at all, so releasing unconditionally is wrong.
//!
//! The core therefore only calls through these hooks; the nanobind layer supplies the Python C API
//! implementation via setGilHooks(). They stay null in a build with no bindings, which makes
//! OptionalGilRelease a no-op there. The GIL is process-wide, so a single global is enough.
struct GilHooks
{
    //! Releases the GIL and returns an opaque token, or nullptr when this thread does not hold it.
    void* (*release)() noexcept;
    //! Re-acquires the GIL. Called only with a non-null token returned by release().
    void (*restore)(void* token) noexcept;
};

//! Installs the hooks. `hooks` must have static storage duration: the core holds the pointer for
//! the lifetime of the process. Called once from the nanobind module initializer.
void setGilHooks(GilHooks const* hooks) noexcept;

//! Returns the installed hooks, or nullptr when no bindings are loaded.
[[nodiscard]] GilHooks const* gilHooks() noexcept;

//! Releases the GIL for the enclosing scope if -- and only if -- this thread currently holds it.
class OptionalGilRelease
{
public:
    OptionalGilRelease() noexcept
        : mHooks(gilHooks())
    {
        if (mHooks != nullptr)
        {
            mToken = mHooks->release();
        }
    }

    ~OptionalGilRelease()
    {
        if (mToken != nullptr)
        {
            mHooks->restore(mToken);
        }
    }

    OptionalGilRelease(OptionalGilRelease const&) = delete;
    OptionalGilRelease& operator=(OptionalGilRelease const&) = delete;
    OptionalGilRelease(OptionalGilRelease&&) = delete;
    OptionalGilRelease& operator=(OptionalGilRelease&&) = delete;

private:
    GilHooks const* mHooks;
    void* mToken = nullptr;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
