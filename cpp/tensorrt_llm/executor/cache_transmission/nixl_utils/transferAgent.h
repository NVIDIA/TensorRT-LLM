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
#include <mutex>
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
    [[nodiscard]] std::string getLastStatusStr() const override;

    [[nodiscard]] bool release() override;

private:
    [[nodiscard]] nixl_status_t queryStatus() const;

    // weak_ptr so the status outliving the owning agent is safe (lock() returns null after reset).
    std::weak_ptr<nixlAgent> mWeakAgent;
    nixlXferReqH* mHandle{};
    mutable std::atomic<int> mLastStatus{0};
    bool const mSynchronizeHandleAccess;
    mutable std::mutex mHandleMutex;
};

// Not `final`: the low-level virtuals below (postXferRequest / registerRegionImpl) are the
// fault-injection seam — bounce failure tests subclass this agent and override them.
class NixlTransferAgent : public BaseTransferAgent
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

    // ---- Low-level transfer primitives (below the VMM splitter) -------------------------------
    // submitTransferRequests() = bounce fork + VMM split/coalesce + postXferRequest(). The bounce
    // transport calls these directly: its remote address comes from a credit (already final, no
    // VMM resolution) and its per-chunk cadence cannot afford the full public path. Virtual so
    // bounce failure tests can inject deterministic transfer faults.

    /// Post one transfer whose descriptors are already FINAL device addresses (no VMM splitting,
    /// no bounce fork). Returns nullptr on submission failure (logged) instead of aborting.
    /// Takes no agent lock: safe from the bounce IO thread, which is joined before agent teardown.
    [[nodiscard]] virtual std::unique_ptr<TransferStatus> postXferRequest(TransferOp op, TransferDescs const& srcDescs,
        TransferDescs const& dstDescs, std::string const& remoteName, std::optional<SyncMessage> const& syncMessage);

    /// Register/deregister one raw device range with NIXL, WITHOUT the VMM split or the AgentDesc
    /// VRAM-region bookkeeping (the bounce arena must not enter the splitter's region maps).
    [[nodiscard]] virtual bool registerRegionImpl(void* base, std::size_t bytes, int deviceId);
    virtual void deregisterRegionImpl(void* base, std::size_t bytes, int deviceId);

    nixl_opt_args_t* getExtraParams() noexcept
    {
        return &mExtraParams;
    }

    void notifySyncMessage(std::string const& name, SyncMessage const& syncMessage) override;

    [[nodiscard]] std::unordered_map<std::string, std::vector<SyncMessage>> getNotifiedSyncMessages() override;

    ConnectionInfoType getLocalConnectionInfo() override;

    void loadRemoteAgent(std::string const& name, ConnectionInfoType const& connectionInfo) override;

    bool checkRemoteDescs(std::string const& name, MemoryDescs const& memoryDescs) override;

    /// Whether the bounce v2 transport is active on this agent (built + enabled + init succeeded).
    /// Programmatic alternative to grepping logs (deployment checks and tests).
    [[nodiscard]] bool isBounceEnabled() const noexcept
    {
        return mBounce != nullptr;
    }

    /// Number of transfer requests routed to the bounce fast path so far.
    [[nodiscard]] std::uint64_t getBounceSubmitCount() const noexcept
    {
        return mBounceSubmitCount.load(std::memory_order_relaxed);
    }

private:
    /// Counts requests admitted by shouldUseBounce (see getBounceSubmitCount).
    std::atomic<std::uint64_t> mBounceSubmitCount{0};

    // shared_ptr so outstanding NixlTransferStatus (via weak_ptr) can detect agent reset.
    std::shared_ptr<nixlAgent> mRawAgent;
    nixlBackendH* mRawBackend{};
    nixl_opt_args_t mExtraParams;
    std::string mName;
    std::string mAddress;
    int mRank{0};
    int mWorldSize{1};
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
    /// @param agentBufferEnable explicit enable/disable from BaseAgentConfig; unset falls back to
    /// the TRTLLM_NIXL_BOUNCE_ENABLE environment variable.
    void maybeInitBounce(std::optional<bool> agentBufferEnable);
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
