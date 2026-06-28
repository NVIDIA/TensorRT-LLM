/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nixl.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <thread>

namespace tensorrt_llm::executor::kv_cache
{

namespace bounce
{
// Pimpl holding the bounce v2 transport + its pools/engine/channel. Defined only in the .cpp
// (and only when built with TLLM_BOUNCE_V2 / ENABLE_UCX); the member below is always present so
// the NixlTransferAgent layout is identical across all translation units.
struct NixlBounceState;
} // namespace bounce

struct NixlHelper
{
    [[nodiscard]] static nixl_mem_t convert(MemoryType type);
    [[nodiscard]] static nixlBasicDesc convert(MemoryDesc const& desc);
    [[nodiscard]] static nixl_reg_dlist_t convertRegDlist(RegisterDescs const& descs);
    [[nodiscard]] static nixl_reg_dlist_t convertRegDlist(FileDescs const& descs);
    [[nodiscard]] static nixl_xfer_op_t convert(TransferOp const& op);
    [[nodiscard]] static nixl_xfer_dlist_t convertXferDist(TransferDescs const& descs);
    [[nodiscard]] static nixl_xfer_dlist_t convertXferDist(FileDescs const& descs);
    static void posixGpuToFileFallback(MemoryDescs const& memoryDesc, FileDescs const& fileDescs);
    static void posixFileToGpuFallback(MemoryDescs const& memoryDesc, FileDescs const& fileDescs);

    /// @brief Coalesce contiguous memory regions to reduce memory registration overhead.
    /// Adjacent memory regions with the same deviceId will be merged into a single region.
    /// @param descs Memory descriptors to coalesce
    /// @return Coalesced MemoryDescs
    [[nodiscard]] static MemoryDescs coalesceMemoryDescs(MemoryDescs const& descs);

    /// @brief Coalesce contiguous memory regions in src and dst to reduce transfer count.
    /// If src[i] and src[i+1] are contiguous, and dst[i] and dst[i+1] are also contiguous
    /// (with same deviceId), they will be merged into a single transfer.
    /// @param srcDescs Source memory descriptors
    /// @param dstDescs Destination memory descriptors
    /// @return Pair of coalesced (src, dst) MemoryDescs
    [[nodiscard]] static std::pair<MemoryDescs, MemoryDescs> coalesceTransferDescs(
        TransferDescs const& srcDescs, TransferDescs const& dstDescs);
};

class NixlTransferStatus final : public TransferStatus
{
public:
    NixlTransferStatus(std::weak_ptr<nixlAgent> agent, nixlXferReqH* handle);
    ~NixlTransferStatus() noexcept override;

    NixlTransferStatus(NixlTransferStatus const&) = delete;
    NixlTransferStatus& operator=(NixlTransferStatus const&) = delete;
    NixlTransferStatus(NixlTransferStatus&&) = delete;
    NixlTransferStatus& operator=(NixlTransferStatus&&) = delete;

    [[nodiscard]] bool isCompleted() const override;

    [[nodiscard]] TransferState wait(int64_t timeout_ms = -1) const override;

    [[nodiscard]] int getLastStatus() const noexcept;
    [[nodiscard]] std::string getLastStatusStr() const;

private:
    // weak_ptr so the status outliving the owning agent is safe (lock() returns null after reset).
    std::weak_ptr<nixlAgent> mWeakAgent;
    nixlXferReqH* mHandle{};
    mutable std::atomic<int> mLastStatus{0};
};

class NixlTransferAgent final : public BaseTransferAgent
{
public:
    NixlTransferAgent(BaseAgentConfig const& config);
    ~NixlTransferAgent();

    /// Synchronously release NIXL agent / UCX / prog_thread. Idempotent.
    void shutdown() noexcept;

    void registerMemory(RegisterDescs const& descs) override;

    void deregisterMemory(RegisterDescs const& descs) override;

    void loadRemoteAgent(std::string const& name, AgentDesc const& agentDesc) override;

    AgentDesc getLocalAgentDesc() override;

    void invalidateRemoteAgent(std::string const& name) override;

    [[nodiscard]] std::unique_ptr<TransferStatus> submitTransferRequests(TransferRequest const& request) override;

    [[nodiscard]] nixlAgent* getRawAgent() const noexcept
    {
        return mRawAgent.get();
    }

    nixl_opt_args_t* getExtraParams() noexcept
    {
        return &mExtraParams;
    }

    void notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) override;

    [[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> getNotifiedSyncMessages() override;

    ConnectionInfoType getLocalConnectionInfo() override;

    void loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) override;

    bool checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs) override;

private:
    // shared_ptr so outstanding NixlTransferStatus (via weak_ptr) can detect agent reset.
    std::shared_ptr<nixlAgent> mRawAgent;
    nixlBackendH* mRawBackend{};
    nixl_opt_args_t mExtraParams;
    std::string mName;
    std::string mAddress;
    std::atomic<bool> mShutdown{false};

    /// Serializes (a) wrapper-map mutations vs reads and (b) drain-on-shutdown.
    /// Writers (register/deregister/load/invalidate/shutdown) take unique_lock;
    /// readers (submit / getLocalAgentDesc / checkRemoteDescs / etc.) take shared_lock.
    mutable std::shared_mutex mLock;

    /// Local VMM region info (from registerMemory). Keyed by local virtual address.
    VramRegionMap mLocalVramRegionInfo;

    /// Remote VMM region info (from loadRemoteAgent). Keyed by {agentName → {addr → info}}.
    /// Per-agent maps because different remote agents may have overlapping virtual addresses.
    std::unordered_map<std::string, VramRegionMap> mRemoteVramRegionInfo;

    /// Bounce v2 transport (opt-in via TRTLLM_NIXL_BOUNCE_ENABLE). Null unless enabled & built;
    /// when null the agent behaves exactly as before. See bounce/DESIGN.md.
    std::unique_ptr<bounce::NixlBounceState> mBounce;

    /// Lazily create the bounce transport (ctor, before any metadata exchange) when enabled.
    void maybeInitBounce();
    /// Heuristic gate: is this request eligible for the bounce fast path?
    [[nodiscard]] bool shouldUseBounce(TransferRequest const& request) const;
};

class NixlLoopbackAgent final : public BaseLoopbackAgent
{
public:
    NixlLoopbackAgent(BaseAgentConfig const& config);
    ~NixlLoopbackAgent() override;

    /// Synchronously release the NIXL agent. Idempotent; drains in-flight requests.
    void shutdown() noexcept;

    virtual void executeLoopbackRequest(
        MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload) override;

private:
    int registerMemory(MemoryDescs const& descs);
    int deregisterMemory(MemoryDescs const& descs);
    int registerFiles(FileDescs const& descs);
    int deregisterFiles(FileDescs const& descs);

    [[nodiscard]] std::unique_ptr<TransferStatus> submitLoopbackRequests(
        MemoryDescs const& memoryDescs, FileDescs const& filedescs, bool isOffload);

    std::shared_ptr<nixlAgent> mRawAgent;
    std::string mName;
    std::atomic<bool> mShutdown{false};
    /// Drain-on-shutdown: executeLoopbackRequest takes shared_lock; shutdown takes unique_lock.
    mutable std::shared_mutex mLock;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    [[nodiscard]] std::unique_ptr<BaseTransferAgent> createNixlTransferAgent(BaseAgentConfig const* config);
}

extern "C"
{
    [[nodiscard]] std::shared_ptr<BaseLoopbackAgent> createNixlLoopbackAgent(BaseAgentConfig const* config);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
