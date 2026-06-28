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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/NixlTransferEngine.h"

#include "tensorrt_llm/common/logger.h"

#include "nixl.h"
#include "nixl_types.h"

namespace tensorrt_llm::executor::kv_cache::bounce
{

NixlTransferEngine::NixlTransferEngine(nixlAgent* agent, int deviceId)
    : mRawAgent(agent)
    , mDeviceId(deviceId)
{
}

NixlTransferEngine::~NixlTransferEngine()
{
    std::lock_guard<std::mutex> lk(mMu);
    for (auto& [id, h] : mHandles)
    {
        try
        {
            mRawAgent->releaseXferReq(static_cast<nixlXferReqH*>(h));
        }
        catch (...)
        {
        }
    }
    // Deregister every region we registered, so we don't leave a stale registration on the borrowed
    // agent after the backing buffer is freed. The caller (NixlBounceState) destroys this engine
    // BEFORE the BounceArena, so the memory is still valid here.
    for (auto const& [addr, bytes] : mRegistered)
    {
        try
        {
            nixl_reg_dlist_t list{VRAM_SEG};
            list.addDesc(nixlBlobDesc{reinterpret_cast<uintptr_t>(addr), bytes, static_cast<uint64_t>(mDeviceId)});
            (void) mRawAgent->deregisterMem(list);
        }
        catch (...)
        {
        }
    }
}

bool NixlTransferEngine::registerRegion(void* addr, std::size_t bytes)
{
    nixl_reg_dlist_t list{VRAM_SEG};
    list.addDesc(nixlBlobDesc{reinterpret_cast<uintptr_t>(addr), bytes, static_cast<uint64_t>(mDeviceId)});
    nixl_status_t const st = mRawAgent->registerMem(list);
    if (st != NIXL_SUCCESS)
    {
        TLLM_LOG_WARNING("NixlTransferEngine: registerMem failed: %s", nixlEnumStrings::statusStr(st).c_str());
        return false;
    }
    mRegistered.emplace_back(addr, bytes); // remember so the dtor can deregisterMem
    return true;
}

std::uint64_t NixlTransferEngine::postWrite(std::string const& peer, void const* src, std::uint64_t dstAddr,
    std::uint32_t remoteDevId, std::uint32_t bytes, cudaStream_t stream)
{
    (void) stream; // NIXL is not ordered against the gather stream; caller synced it already.
    // Local src is on this agent's GPU (mDeviceId); the remote dst is on the peer's GPU, whose
    // index is carried in the credit (remoteDevId). Do NOT assume sender/receiver share a device
    // index — that is false when ranks aren't single-GPU-per-rank.
    nixl_xfer_dlist_t local{VRAM_SEG};
    local.addDesc(nixlBasicDesc{reinterpret_cast<uintptr_t>(src), bytes, static_cast<uint64_t>(mDeviceId)});
    nixl_xfer_dlist_t remote{VRAM_SEG};
    remote.addDesc(nixlBasicDesc{static_cast<uintptr_t>(dstAddr), bytes, static_cast<uint64_t>(remoteDevId)});

    nixlXferReqH* handle = nullptr;
    nixl_status_t st = mRawAgent->createXferReq(NIXL_WRITE, local, remote, peer, handle);
    if (st != NIXL_SUCCESS || handle == nullptr)
    {
        TLLM_LOG_WARNING(
            "NixlTransferEngine: createXferReq to %s failed: %s", peer.c_str(), nixlEnumStrings::statusStr(st).c_str());
        return 0; // 0 == invalid handle; poll() reports kFailed
    }
    st = mRawAgent->postXferReq(handle);
    if (st != NIXL_SUCCESS && st != NIXL_IN_PROG)
    {
        TLLM_LOG_WARNING(
            "NixlTransferEngine: postXferReq to %s failed: %s", peer.c_str(), nixlEnumStrings::statusStr(st).c_str());
        try
        {
            mRawAgent->releaseXferReq(handle);
        }
        catch (...)
        {
        }
        return 0;
    }
    std::lock_guard<std::mutex> lk(mMu);
    std::uint64_t const id = mNext++;
    mHandles.emplace(id, static_cast<void*>(handle));
    return id;
}

XferState NixlTransferEngine::poll(std::uint64_t handle)
{
    nixlXferReqH* h = nullptr;
    {
        std::lock_guard<std::mutex> lk(mMu);
        auto it = mHandles.find(handle);
        if (it == mHandles.end())
        {
            return XferState::kFailed; // unknown / invalid (incl. id 0)
        }
        h = static_cast<nixlXferReqH*>(it->second);
    }
    nixl_status_t const st = mRawAgent->getXferStatus(h);
    if (st == NIXL_SUCCESS)
    {
        return XferState::kDone;
    }
    if (st == NIXL_IN_PROG)
    {
        return XferState::kInProgress;
    }
    return XferState::kFailed;
}

void NixlTransferEngine::release(std::uint64_t handle)
{
    nixlXferReqH* h = nullptr;
    {
        std::lock_guard<std::mutex> lk(mMu);
        auto it = mHandles.find(handle);
        if (it == mHandles.end())
        {
            return;
        }
        h = static_cast<nixlXferReqH*>(it->second);
        mHandles.erase(it);
    }
    try
    {
        mRawAgent->releaseXferReq(h);
    }
    catch (...)
    {
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
