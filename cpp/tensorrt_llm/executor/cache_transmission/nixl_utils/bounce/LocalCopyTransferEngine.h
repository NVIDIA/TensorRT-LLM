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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/TransferEngine.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// LocalCopyTransferEngine — single-process loopback TransferEngine (DESIGN)
// ----------------------------------------------------------------------------
// Stands in for RDMA when sender and receiver live in the same process on the same
// GPU: postWrite() is a cudaMemcpyAsync(D2D) from the sender slot to the receiver
// slot (whose device address arrived as a credit), ordered on the gather stream;
// poll() queries a recorded event. This lets the full bounce pipeline (reactor +
// scheduler + slot pools + gather/scatter kernels) run end-to-end and verify
// byte-exact data movement without NIXL or a second node. NOT for production.
// ============================================================================
class LocalCopyTransferEngine : public TransferEngine
{
public:
    LocalCopyTransferEngine() = default;
    ~LocalCopyTransferEngine() override;

    [[nodiscard]] bool registerRegion(void* addr, std::size_t bytes) override
    {
        (void) addr;
        (void) bytes; // same-process: nothing to register
        return true;
    }

    [[nodiscard]] std::uint64_t postWrite(std::string const& peer, void const* src, std::uint64_t dstAddr,
        std::uint32_t remoteDevId, std::uint32_t bytes, cudaStream_t stream) override;
    [[nodiscard]] XferState poll(std::uint64_t handle) override;
    void release(std::uint64_t handle) override;

private:
    std::mutex mMu;
    std::uint64_t mNext{1};
    std::unordered_map<std::uint64_t, cudaEvent_t> mEvents;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
