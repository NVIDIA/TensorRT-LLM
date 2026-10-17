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

//! The CUDA context a KVCM2 object's resources belong to, made current for its teardown.
//!
//! A destructor runs on whichever thread drops the last reference. From Python that is often a
//! thread with no current CUDA context: the cyclic garbage collector frees a cycle on the thread
//! whose allocation triggered the collection, e.g. a watchdog timer. Every driver call the
//! teardown makes (event records, frees, unregistration) would then fail with
//! CUDA_ERROR_INVALID_CONTEXT.
//!
//! Declare it before every member whose destruction touches the driver, and call
//! enterForTeardown() first thing in the owner's destructor. The context stays current for the
//! destructor body and the member destructors that follow, and is popped when this member, the
//! last of them, is destroyed. A thread that already has the context current is left untouched.
class CudaContextBinding
{
public:
    //! Binds the context current on the calling thread (none if the thread has no context).
    CudaContextBinding() noexcept
    {
        if (cuCtxGetCurrent(&mContext) != CUDA_SUCCESS)
        {
            mContext = nullptr;
        }
    }

    //! Binds the given context, e.g. the one recorded by the owning manager.
    explicit CudaContextBinding(CUcontext context) noexcept
        : mContext(context)
    {
    }

    CudaContextBinding(CudaContextBinding const&) = delete;
    CudaContextBinding& operator=(CudaContextBinding const&) = delete;

    ~CudaContextBinding()
    {
        if (mPushed)
        {
            CUcontext popped{};
            cuCtxPopCurrent(&popped);
        }
    }

    [[nodiscard]] CUcontext context() const noexcept
    {
        return mContext;
    }

    //! Makes the bound context current on this thread until this object is destroyed. If pushing
    //! fails the teardown runs as before and its first driver call reports the error.
    void enterForTeardown() noexcept
    {
        if (mContext == nullptr || mPushed)
        {
            return;
        }
        CUcontext current{};
        if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == mContext)
        {
            return;
        }
        mPushed = cuCtxPushCurrent(mContext) == CUDA_SUCCESS;
    }

private:
    CUcontext mContext{};
    bool mPushed = false;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
