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

#include "kv_cache_manager_v2/utils/cudaContextScope.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

CUcontext CudaContextScope::current() noexcept
{
    CUcontext context = nullptr;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS)
    {
        return nullptr;
    }
    return context;
}

CudaContextScope::CudaContextScope(CUcontext context) noexcept
{
    CUcontext before = nullptr;
    mError = cuCtxGetCurrent(&before);
    if (mError != CUDA_SUCCESS)
    {
        // The driver itself is gone, which no push can repair.
        return;
    }
    if (context == nullptr || before == context)
    {
        mUsable = before != nullptr;
        if (!mUsable)
        {
            mError = CUDA_ERROR_INVALID_CONTEXT;
        }
        return;
    }
    // A context the thread already had is not a substitute: the owner's resources live in
    // `context`, and if that one cannot be made current they cannot be released from here.
    mError = cuCtxPushCurrent(context);
    if (mError == CUDA_SUCCESS)
    {
        mPushed = true;
        mUsable = true;
    }
}

CudaContextScope::~CudaContextScope()
{
    if (mPushed)
    {
        CUcontext popped = nullptr;
        // Nothing to do on failure: the thread is being left, and a pop fails only once the
        // driver is being torn down.
        cuCtxPopCurrent(&popped);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
