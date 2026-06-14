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
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/p2pTransferAgent.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include <atomic>
#include <thread>

namespace tensorrt_llm::executor::kv_cache
{

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
    NixlTransferStatus(nixlAgent* agent, nixlXferReqH* handle);

    [[nodiscard]] bool isCompleted() const override;

    [[nodiscard]] TransferState wait(int64_t timeout_ms = -1) const override;

private:
    nixlAgent* mRawAgent{};
    nixlXferReqH* mHandle{};
};

// MixedTransferStatus is declared alongside P2pTransferStatus in p2pTransferAgent.h —
// keeping it out of this NIXL-dependent header lets unit tests compose it without
// pulling in nixl.h.

/// @brief Test-only aggregate of every path-decision counter in NixlTransferAgent::submitTransferRequests.
/// Used to assert that eligibility short-circuits, cub-vs-memcpyBatch thresholds, and single-vs-
/// multi-thread batch-copy thresholds actually route through the intended branch instead of silently
/// falling through. See NixlTransferAgent::getPathCounters().
struct PathCounterSnapshot
{
    /// NixlTransferAgent-level terminal branches (sum equals total submit count).
    uint64_t pureP2p;  ///< Case A — every segment mapped, no NIXL half.
    uint64_t mixed;    ///< Case C — partial mapping; returns MixedTransferStatus.
    uint64_t pureNixl; ///< P2P disabled / non-VRAM / sync-message / no remote mapping / full fallback.

    /// P2P submit path selection (sum equals pureP2p + mixed).
    uint64_t cubSubmit;         ///< avgSegmentSize < threshold, routes to cub::DeviceMemcpy::Batched.
    uint64_t memcpyBatchSubmit; ///< avgSegmentSize >= threshold, routes to cudaMemcpyBatchAsync.

    /// cudaMemcpyBatchAsync sub-path (sum equals memcpyBatchSubmit; forwarded from P2pAgentCounters).
    uint64_t memcpyBatchSingleThread;
    uint64_t memcpyBatchMultiThread;
};

class NixlTransferAgent final : public BaseTransferAgent
{
public:
    NixlTransferAgent(BaseAgentConfig const& config);
    ~NixlTransferAgent();

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

    /// @brief Test-only snapshot of path counters. Reads relaxed atomics; safe to call any time.
    [[nodiscard]] PathCounterSnapshot getPathCounters() const noexcept;

private:
    /// @brief Build and post a NIXL xfer request for the given descs + optional sync message.
    /// Extracted so both the full-NIXL path and the mixed path (which only sends unmapped
    /// segments via NIXL) can share the createXferReq/postXferReq bookkeeping.
    [[nodiscard]] std::unique_ptr<TransferStatus> submitNixlTransferInternal(TransferOp op, MemoryDescs const& srcDescs,
        MemoryDescs const& dstDescs, std::string const& remoteName, std::optional<SyncMessage> const& syncMessage);

    std::unique_ptr<nixlAgent> mRawAgent;
    nixlBackendH* mRawBackend{};
    nixl_opt_args_t mExtraParams;
    std::string mName;
    std::string mAddress;

    std::vector<char> mDRamSrcBuffer;
    std::vector<char> mDRamDstBuffer;

    /// Local VMM region info (from registerMemory). Keyed by local virtual address.
    VramRegionMap mLocalVramRegionInfo;

    /// Remote VMM region info (from loadRemoteAgent). Keyed by {agentName → {addr → info}}.
    /// Per-agent maps because different remote agents may have overlapping virtual addresses.
    std::unordered_map<std::string, VramRegionMap> mRemoteVramRegionInfo;

    /// Optional P2P fast-path agent. nullptr when TRTLLM_KV_TRANSFER_P2P_DISABLE=1.
    /// Owns the handle exporter, remote mapping registry, per-thread transfer contexts,
    /// and shared batch-copy worker pool / event pool. When unset or when a remote agent's
    /// import failed, submitTransferRequests falls back to the NIXL path below.
    std::unique_ptr<P2pTransferAgent> mP2pAgent;

    /// @brief Test-only atomic counters for submitTransferRequests path selection.
    /// Five counters live here (three terminal branches + cub/memcpyBatch decision); the
    /// two sub-counters inside P2pTransferContext::submitWithMemcpyBatch live on P2pAgentCounters
    /// and are forwarded by getPathCounters(). One atomic fetch_add per submit is negligible
    /// relative to the NIXL/CUDA work that follows.
    mutable std::atomic<uint64_t> mPureP2pCount{0};
    mutable std::atomic<uint64_t> mMixedCount{0};
    mutable std::atomic<uint64_t> mPureNixlCount{0};
    mutable std::atomic<uint64_t> mCubSubmitCount{0};
    mutable std::atomic<uint64_t> mMemcpyBatchSubmitCount{0};
};

class NixlLoopbackAgent final : public BaseLoopbackAgent
{
public:
    NixlLoopbackAgent(BaseAgentConfig const& config);
    virtual ~NixlLoopbackAgent() = default;

    virtual void executeLoopbackRequest(
        MemoryDescs const& memoryDescs, FileDescs const& fileDescs, bool isOffload) override;

private:
    int registerMemory(MemoryDescs const& descs);
    int deregisterMemory(MemoryDescs const& descs);
    int registerFiles(FileDescs const& descs);
    int deregisterFiles(FileDescs const& descs);

    [[nodiscard]] std::unique_ptr<TransferStatus> submitLoopbackRequests(
        MemoryDescs const& memoryDescs, FileDescs const& filedescs, bool isOffload);

    std::unique_ptr<nixlAgent> mRawAgent;
    std::string mName;
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
