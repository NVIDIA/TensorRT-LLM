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

#include "tensorrt_llm/executor/transferAgent.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::direct
{

// ============================================================================
// Direct peer copy: one batched copy kernel per VRAM -> VRAM write to a peer whose registered memory this process
// has mapped through CUDA fabric handles.
// ----------------------------------------------------------------------------
// The standard NIXL/UCX path issues one copy plus event record per descriptor (thousands per KV send, tens of
// thousands when the two sides hold different KV heads per rank and every block moves as per-layer head slices), and
// those driver calls contend with the model's launches on the CUDA driver locks. KVCacheManagerV2 pools are
// cuMemCreate'd with CU_MEM_HANDLE_TYPE_FABRIC, so the receiver advertises one fabric handle per physical chunk of
// each registered region; the sender imports and maps them, then serves a write with one copy kernel launch (stores
// over NVLink; the kernel reads its plan from mapped pinned host memory, so no copy-engine work) and one event
// record. Completion is the event.
//
// Reach: any peer whose GPU this GPU reaches over NVLink and whose fabric handles this process may import, that is
// the same node, or another node of the same NVLink domain (one NVL72 rack) when both processes hold the same IMEX
// channel (one Slurm job under switch/nvidia_imex, or one Kubernetes ComputeDomain). For any other peer the import
// fails (CUDA_ERROR_NOT_PERMITTED for another IMEX channel), and that peer keeps the NIXL path.
//
// Imports run on background threads (mapping a peer's pools costs about 0.2 ms per physical chunk on one node and up
// to about 1 ms across nodes, seconds per peer), so registering a peer never blocks the caller; writes to a peer
// whose mapping is not ready yet take the NIXL path. Everything else (non-VRAM, sync messages, unmapped peers,
// descriptors outside the mapped regions) keeps the NIXL path too.
//
// Opt-in: TRTLLM_NIXL_DIRECT_COPY=1 on both the exporting (receiver) and importing (sender) agent. When bounce v2 is
// also enabled, the direct path is tried first, so bounce only carries writes to peers this path cannot map.
// ============================================================================

struct DirectConfig
{
    bool enabled{false}; ///< TRTLLM_NIXL_DIRECT_COPY (0/1)

    [[nodiscard]] static DirectConfig fromEnv();
};

/// One physical chunk of an exported region, at its virtual address in the exporting process.
struct ExportedChunk
{
    std::uint64_t va{0};
    std::uint64_t size{0};
    CUmemFabricHandle handle{};
};

/// A registered region [base, base + len) and the physical chunks that back it (they may start before base).
struct ExportedRegion
{
    std::uint64_t base{0};
    std::uint64_t len{0};
    std::vector<ExportedChunk> chunks;
};

struct DirectHandshake
{
    std::uint32_t version{0};
    std::string host;
    std::string bootId;
    std::vector<ExportedRegion> regions;
};

[[nodiscard]] std::string encodeHandshake(DirectHandshake const& handshake);
[[nodiscard]] bool decodeHandshake(std::string const& blob, DirectHandshake& out);

/// AgentDesc extension envelope. AgentDesc carries one opaque per-agent blob (historically the bounce handshake);
/// it now holds either a raw bounce handshake (older peers) or an envelope of tagged sections.
enum class ExtensionTag : std::uint32_t
{
    kBounce = 1,
    kDirect = 2,
};

[[nodiscard]] std::string packExtensions(std::vector<std::pair<ExtensionTag, std::string>> const& sections);
/// Splits an extension blob. A blob without the envelope magic is returned as the bounce section.
[[nodiscard]] std::unordered_map<std::uint32_t, std::string> unpackExtensions(std::string const& blob);

/// Launch the batched copy: entry i copies sizes[i] bytes from srcs[i] to dsts[i]. Device-pointer arrays.
[[nodiscard]] cudaError_t launchDirectCopy(std::uint64_t const* srcs, std::uint64_t const* dsts,
    std::uint32_t const* sizes, std::uint32_t n, std::uint32_t maxCtas, std::uint32_t threads, cudaStream_t stream);

/// Same plan, one warp per entry instead of one CTA per entry (for plans of small entries).
[[nodiscard]] cudaError_t launchDirectCopyWarpPerEntry(std::uint64_t const* srcs, std::uint64_t const* dsts,
    std::uint32_t const* sizes, std::uint32_t n, std::uint32_t maxCtas, std::uint32_t threads, cudaStream_t stream);

struct CopyCtx;
class CopyCtxPool;

class DirectPeerCopy
{
public:
    explicit DirectPeerCopy(int deviceId);
    ~DirectPeerCopy();

    DirectPeerCopy(DirectPeerCopy const&) = delete;
    DirectPeerCopy& operator=(DirectPeerCopy const&) = delete;

    /// Receiver side: the handshake blob exporting every region of `regions` that is fully backed by
    /// fabric-exportable VMM chunks (cached until the region map changes). Empty when nothing is exportable.
    [[nodiscard]] std::string localHandshake(VramRegionMap const& regions);

    /// Sender side: queue the import and mapping of a peer's exported regions on a background thread and return.
    /// Writes to the peer take the NIXL path until its mapping is ready. Never throws; on any failure (a peer outside
    /// this NVLink domain or IMEX channel included) the peer stays on NIXL. A mapping made earlier under this name is
    /// dropped first.
    void importPeer(std::string const& name, std::string const& blob);

    /// Drop a peer: cancel its pending import and unmap it (waits for in-flight copies first).
    void forgetPeer(std::string const& name);

    /// Serve `request` with one copy kernel when it is a VRAM -> VRAM write whose every destination lies in a
    /// mapped region of the peer and every source in a local registered region; nullptr when not eligible (the
    /// caller uses NIXL). Never throws for an eligible-looking request: a CUDA error after work was enqueued
    /// comes back as a failed status.
    [[nodiscard]] std::unique_ptr<TransferStatus> trySubmit(TransferRequest const& request, VramRegionMap const& local);

    [[nodiscard]] std::uint64_t submitCount() const noexcept
    {
        return mSubmits.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t declineCount() const noexcept
    {
        return mDeclines.load(std::memory_order_relaxed);
    }

    /// Writes served to peers on another node (a subset of submitCount()).
    [[nodiscard]] std::uint64_t crossNodeSubmitCount() const noexcept
    {
        return mCrossNodeSubmits.load(std::memory_order_relaxed);
    }

    /// Peers whose regions are mapped, so that the direct path serves writes to them.
    [[nodiscard]] std::size_t mappedPeerCount() const;

private:
    struct MappedChunk
    {
        std::uint64_t offset; ///< from the region's mapping start
        std::uint64_t size;
    };

    struct MappedRegion
    {
        std::uint64_t remoteBase{0}; ///< registered range start (remote VA)
        std::uint64_t remoteEnd{0};  ///< registered range end (remote VA)
        std::uint64_t mapStart{0};   ///< remote VA of the first mapped chunk
        std::uint64_t mapLen{0};
        CUdeviceptr localVa{0};      ///< local VA of mapStart
        std::vector<MappedChunk> chunks;
    };

    struct MappedPeer
    {
        std::map<std::uint64_t, MappedRegion> regions; ///< keyed by remoteBase
        std::uint64_t bytes{0};
        bool remoteHost{false};                        ///< the peer runs on another node of the NVLink domain
    };

    struct ImportTask
    {
        std::string name;
        DirectHandshake handshake;
        std::uint64_t generation{0};
    };

    /// Import and map every region of `handshake`; nullptr on any failure (partial mappings are undone).
    [[nodiscard]] std::shared_ptr<MappedPeer> mapPeer(std::string const& name, DirectHandshake const& handshake);
    void importLoop();
    /// Remove `name` from the mapped peers (caller holds mImportMu); the result goes to releaseMapped() after the
    /// lock is dropped.
    [[nodiscard]] std::shared_ptr<MappedPeer> takeMappedLocked(std::string const& name);
    /// Unmap a peer taken out of mPeers once the copies in flight are done.
    void releaseMapped(std::shared_ptr<MappedPeer> peer) noexcept;
    void unmapPeer(MappedPeer& peer) noexcept;
    void declined(std::string const& peer, char const* reason, std::size_t descs);
    void noteSubmit(std::string const& name, std::size_t descs, std::uint64_t units, std::uint64_t bytes);
    void waitAllInflight() noexcept;

    int mDeviceId{0};
    std::string mHost;
    std::string mBootId;

    std::mutex mExportMu;
    std::string mExportBlob;
    VramRegionMap mExportedFrom;
    bool mExportDone{false};

    mutable std::shared_mutex mPeersMu;
    std::unordered_map<std::string, std::shared_ptr<MappedPeer>> mPeers;

    // Background imports. Lock order: mImportMu before mPeersMu. A task is stale once the generation of its name has
    // moved on (imported again or forgotten); a stale mapping is undone instead of published.
    std::mutex mImportMu;
    std::condition_variable mImportCv;
    std::deque<ImportTask> mImportQueue;
    std::unordered_map<std::string, std::uint64_t> mGenerations;
    std::uint64_t mNextGeneration{0};
    std::vector<std::thread> mImportThreads;
    std::atomic<bool> mStopping{false};

    std::shared_ptr<CopyCtxPool> mPool;

    std::atomic<std::uint64_t> mSubmits{0};
    std::atomic<std::uint64_t> mDeclines{0};
    std::atomic<std::uint64_t> mCrossNodeSubmits{0};
    std::atomic<std::uint64_t> mDescs{0};
    std::atomic<std::uint64_t> mBytes{0};
    std::mutex mWarnMu;
    std::unordered_map<std::string, bool> mWarned;
};

} // namespace tensorrt_llm::executor::kv_cache::direct
