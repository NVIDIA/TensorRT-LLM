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

#include <cuda.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Makes a CUDA context current for the enclosing scope, for teardown that may run on a thread
//! that has none: destructors run on whichever thread drops the last reference, and a thread
//! that never touched CUDA has no current context, so the owner's context is pushed and popped
//! again on exit, leaving the thread as it was found.
//!
//! usable() is false when no context could be made current: the thread has none and the owner
//! recorded none, or the driver refused the push because the context, or the driver itself, is
//! already gone (process exit).
class CudaContextScope
{
public:
    //! The calling thread's current context, or nullptr when it has none or the driver is gone.
    [[nodiscard]] static CUcontext current() noexcept;

    //! Pushes `context` unless it is already current. A null `context` pushes nothing, and the
    //! scope is usable only if the thread has a context of its own.
    explicit CudaContextScope(CUcontext context) noexcept;
    ~CudaContextScope();

    CudaContextScope(CudaContextScope const&) = delete;
    CudaContextScope& operator=(CudaContextScope const&) = delete;
    CudaContextScope(CudaContextScope&&) = delete;
    CudaContextScope& operator=(CudaContextScope&&) = delete;

    //! A context is current on this thread for the scope's duration.
    [[nodiscard]] bool usable() const noexcept
    {
        return mUsable;
    }

    //! The context was pushed by this scope: the thread had none, or a different one.
    [[nodiscard]] bool pushed() const noexcept
    {
        return mPushed;
    }

    //! Why the context could not be made current; CUDA_SUCCESS when usable().
    [[nodiscard]] CUresult error() const noexcept
    {
        return mError;
    }

private:
    bool mPushed = false;
    bool mUsable = false;
    CUresult mError = CUDA_SUCCESS;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
