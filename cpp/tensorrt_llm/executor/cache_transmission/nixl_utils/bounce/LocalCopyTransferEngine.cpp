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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/LocalCopyTransferEngine.h"

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::executor::kv_cache::bounce
{

LocalCopyTransferEngine::~LocalCopyTransferEngine()
{
    std::lock_guard<std::mutex> lk(mMu);
    for (auto& [id, ev] : mEvents)
    {
        TLLM_CUDA_CHECK_WARN(cudaEventDestroy(ev)); // warn-only cleanup (dtor can't throw)
    }
}

std::uint64_t LocalCopyTransferEngine::postWrite(std::string const& peer, void const* src, std::uint64_t dstAddr,
    std::uint32_t remoteDevId, std::uint32_t bytes, cudaStream_t stream)
{
    (void) peer;        // same process; address is directly usable
    (void) remoteDevId; // same process: dst pointer already absolute, device implied by the address
    // Check every CUDA call: if any fails we must NOT register a handle, else poll() would query a
    // garbage/never-recorded event and could report kDone for a copy that never ran (a false Done
    // here masks real bugs, since this engine is the loopback correctness oracle). Return 0, which
    // the reactor treats as a failed post (poll(0) -> kFailed), mirroring NixlTransferEngine.
    cudaError_t st = cudaMemcpyAsync(reinterpret_cast<void*>(dstAddr), src, bytes, cudaMemcpyDeviceToDevice, stream);
    if (st != cudaSuccess)
    {
        (void) cudaGetLastError();
        return 0;
    }
    cudaEvent_t ev{};
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess)
    {
        (void) cudaGetLastError();
        return 0;
    }
    if (cudaEventRecord(ev, stream) != cudaSuccess)
    {
        (void) cudaGetLastError();
        TLLM_CUDA_CHECK_WARN(cudaEventDestroy(ev));
        return 0;
    }
    std::lock_guard<std::mutex> lk(mMu);
    std::uint64_t const id = mNext++;
    mEvents.emplace(id, ev);
    return id;
}

XferState LocalCopyTransferEngine::poll(std::uint64_t handle)
{
    cudaEvent_t ev{};
    {
        std::lock_guard<std::mutex> lk(mMu);
        auto it = mEvents.find(handle);
        if (it == mEvents.end())
        {
            return XferState::kFailed;
        }
        ev = it->second;
    }
    cudaError_t const st = cudaEventQuery(ev);
    if (st == cudaSuccess)
    {
        return XferState::kDone;
    }
    if (st == cudaErrorNotReady)
    {
        return XferState::kInProgress;
    }
    (void) cudaGetLastError(); // real error -> clear sticky state so it doesn't poison later calls
    return XferState::kFailed;
}

void LocalCopyTransferEngine::release(std::uint64_t handle)
{
    std::lock_guard<std::mutex> lk(mMu);
    auto it = mEvents.find(handle);
    if (it != mEvents.end())
    {
        TLLM_CUDA_CHECK_WARN(cudaEventDestroy(it->second));
        mEvents.erase(it);
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
