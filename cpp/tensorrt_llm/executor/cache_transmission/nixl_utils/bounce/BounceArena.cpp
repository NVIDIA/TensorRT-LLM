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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::executor::kv_cache::bounce
{

BounceArena::BounceArena(std::size_t bytes, int deviceId, bool allowFabric)
    : mDeviceId(deviceId)
    , mBytes(bytes)
{
    TLLM_CHECK_WITH_INFO(bytes > 0, "BounceArena: bytes must be > 0");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

    // On MNNVL parts (GH200/GB200) the arena (RDMA src/dst, NIXL-registered) must be fabric memory
    // to be reachable over the NVLink fabric + GPUDirect-RDMA capable. Elsewhere (and CI, or when
    // fabric is force-disabled) fall back to cudaMalloc.
    mIsFabric = allowFabric && tensorrt_llm::common::FabricMemory::supportFbaricMemory();
    if (mIsFabric)
    {
        mFabric = std::make_unique<tensorrt_llm::common::FabricMemory>(bytes);
        mBase = mFabric->getPtr();
        TLLM_LOG_DEBUG("BounceArena: %zuB backed by fabric memory", bytes);
    }
    else
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mBase, bytes));
    }
}

BounceArena::~BounceArena()
{
    if (!mIsFabric && mBase != nullptr)
    {
        // Select the owning device before freeing (multi-GPU: cudaFree otherwise targets the thread's
        // current device). Fabric-backed arena is freed by ~FabricMemory (mFabric). A dtor can't
        // throw, so warn on failure rather than swallow.
        if (cudaSetDevice(mDeviceId) != cudaSuccess)
        {
            (void) cudaGetLastError();
            TLLM_LOG_WARNING("BounceArena::~BounceArena: cudaSetDevice(%d) failed; arena may leak", mDeviceId);
        }
        if (cudaFree(mBase) != cudaSuccess)
        {
            (void) cudaGetLastError();
            TLLM_LOG_WARNING("BounceArena::~BounceArena: cudaFree failed");
        }
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
