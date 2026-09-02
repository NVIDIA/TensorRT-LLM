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

#include <Python.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Releases the GIL for the enclosing scope if -- and only if -- this thread currently holds it.
//!
//! Motivation: destructors. A binding method can declare `nb::call_guard<nb::gil_scoped_release>`,
//! but a *destructor* has no such hook: nanobind runs it from `tp_dealloc` with the GIL held,
//! which is exactly the state that deadlocks against a thread holding the API lock and waiting on
//! `gil_scoped_acquire` to run a Python callback. Objects here are also destroyed on pure-C++
//! paths where there may be no interpreter at all, so an unconditional `nb::gil_scoped_release`
//! is wrong; hence the runtime probe.
//!
//! `Py_IsInitialized()` is safe to call without an interpreter (libpython is linked into
//! libtensorrt_llm), and `PyGILState_Check()` answers precisely the question that matters: does
//! *this* thread hold the GIL right now.
class OptionalGilRelease
{
public:
    OptionalGilRelease()
    {
        if (Py_IsInitialized() != 0 && PyGILState_Check() != 0)
        {
            mState = PyEval_SaveThread();
        }
    }

    ~OptionalGilRelease()
    {
        if (mState != nullptr)
        {
            PyEval_RestoreThread(mState);
        }
    }

    OptionalGilRelease(OptionalGilRelease const&) = delete;
    OptionalGilRelease& operator=(OptionalGilRelease const&) = delete;
    OptionalGilRelease(OptionalGilRelease&&) = delete;
    OptionalGilRelease& operator=(OptionalGilRelease&&) = delete;

private:
    PyThreadState* mState = nullptr;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
