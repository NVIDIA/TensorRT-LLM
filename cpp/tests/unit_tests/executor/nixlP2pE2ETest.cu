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

// End-to-end tests that drive two in-process NixlTransferAgent instances through the
// full P2P fast-path contract (registerMemory -> getLocalAgentDesc with p2pBlob ->
// loadRemoteAgent -> submitTransferRequests -> MixedTransferStatus). Covers all three
// terminal branches of NixlTransferAgent::submitTransferRequests (pure-P2P, mixed,
// pure-NIXL) plus the cub-vs-memcpyBatch and single-vs-multi-thread sub-paths via
// PathCounterSnapshot. The unit-test suites in p2pTransferAgentTest.cu / p2pMemInfoTest.cu
// exercise the P2P layer in isolation; this file covers the NIXL agent hooks that
// stitch them together.
//
// Topology requirements:
//   - At least one CUDA GPU with VMM shareable handle support (POSIX_FILE_DESCRIPTOR
//     preferred, FABRIC fallback). Tests GTEST_SKIP() when neither is available.
//   - Same-host (UDS path for POSIX-FD handoff).
// Not suitable for l0 — gated in CMake and should land on a dedicated P2P CI stage.

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/p2pTransferAgent.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace tensorrt_llm::executor::kv_cache;

namespace
{

// ============================================================================
// VMM buffer helper — P2P-exportable GPU memory via cuMemCreate.
// ============================================================================
//
// Tries CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR first (works on any Linux host with
// CUDA IPC capable GPUs; no NVSwitch required), then CU_MEM_HANDLE_TYPE_FABRIC. If
// neither is allowed by the driver/device, returns nullptr so the test can skip.
//
// The returned pointer is in a dedicated VA reservation so cuMemGetAddressRange inside
// P2pHandleExporter sees a single chunk that matches the full allocation. That is the
// same allocation pattern the real KV-cache transfer pool uses (cacheTransBuffer.cpp),
// so exercising it here tests the realistic shape of local VA metadata that the P2P
// layer serializes into p2pBlob.

class VmmGpuBuffer
{
public:
    static std::unique_ptr<VmmGpuBuffer> tryCreate(size_t size, int deviceId)
    {
        // Detect which shareable handle type the device supports. Prefer POSIX-FD
        // because it has no NVLink/NVSwitch requirement.
        CUmemAllocationHandleType const candidates[]
            = {CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, CU_MEM_HANDLE_TYPE_FABRIC};
        for (auto handleType : candidates)
        {
            try
            {
                return std::unique_ptr<VmmGpuBuffer>(new VmmGpuBuffer(size, deviceId, handleType));
            }
            catch (std::runtime_error const&)
            {
                continue;
            }
        }
        return nullptr;
    }

    ~VmmGpuBuffer()
    {
        if (mDevicePtr)
        {
            cuMemUnmap(mDevicePtr, mAllocSize);
            cuMemAddressFree(mDevicePtr, mAllocSize);
        }
        if (mHandle)
        {
            cuMemRelease(mHandle);
        }
    }

    VmmGpuBuffer(VmmGpuBuffer const&) = delete;
    VmmGpuBuffer& operator=(VmmGpuBuffer const&) = delete;

    [[nodiscard]] void* ptr() const noexcept
    {
        return reinterpret_cast<void*>(mDevicePtr);
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return mAllocSize;
    }

    [[nodiscard]] int deviceId() const noexcept
    {
        return mDeviceId;
    }

    [[nodiscard]] MemoryDesc desc() const
    {
        return MemoryDesc{ptr(), size(), static_cast<uint32_t>(mDeviceId)};
    }

private:
    VmmGpuBuffer(size_t requested, int deviceId, CUmemAllocationHandleType handleType)
        : mDeviceId(deviceId)
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = deviceId;
        prop.requestedHandleTypes = handleType;
        prop.allocFlags.gpuDirectRDMACapable = (handleType == CU_MEM_HANDLE_TYPE_FABRIC) ? 1 : 0;

        size_t granularity = 0;
        auto err = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (err != CUDA_SUCCESS || granularity == 0)
        {
            throw std::runtime_error("cuMemGetAllocationGranularity failed");
        }
        mAllocSize = (requested + granularity - 1) / granularity * granularity;

        err = cuMemCreate(&mHandle, mAllocSize, &prop, 0);
        if (err != CUDA_SUCCESS)
        {
            mHandle = 0;
            throw std::runtime_error("cuMemCreate failed");
        }

        err = cuMemAddressReserve(&mDevicePtr, mAllocSize, granularity, 0, 0);
        if (err != CUDA_SUCCESS)
        {
            cuMemRelease(mHandle);
            mHandle = 0;
            throw std::runtime_error("cuMemAddressReserve failed");
        }

        err = cuMemMap(mDevicePtr, mAllocSize, 0, mHandle, 0);
        if (err != CUDA_SUCCESS)
        {
            cuMemAddressFree(mDevicePtr, mAllocSize);
            cuMemRelease(mHandle);
            mDevicePtr = 0;
            mHandle = 0;
            throw std::runtime_error("cuMemMap failed");
        }

        CUmemAccessDesc accessDesc{};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = deviceId;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        err = cuMemSetAccess(mDevicePtr, mAllocSize, &accessDesc, 1);
        if (err != CUDA_SUCCESS)
        {
            cuMemUnmap(mDevicePtr, mAllocSize);
            cuMemAddressFree(mDevicePtr, mAllocSize);
            cuMemRelease(mHandle);
            mDevicePtr = 0;
            mHandle = 0;
            throw std::runtime_error("cuMemSetAccess failed");
        }
    }

    int mDeviceId{0};
    size_t mAllocSize{0};
    CUmemGenericAllocationHandle mHandle{0};
    CUdeviceptr mDevicePtr{0};
};

// Plain cudaMalloc buffer — used for "not-P2P-exportable" region to force the mixed
// or NIXL-only path. Not all cudaMalloc memory fails the exporter (CUDA IPC path can
// pick it up), but for the tests here we only read/write into it and observe the
// counters, so either outcome is fine: the assertions are on data + path correctness,
// not on which handle-type was selected.
class CudaMallocBuffer
{
public:
    CudaMallocBuffer(size_t size, int deviceId)
        : mSize(size)
        , mDeviceId(deviceId)
    {
        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        TLLM_CUDA_CHECK(cudaMalloc(&mPtr, mSize));
    }

    ~CudaMallocBuffer()
    {
        cudaFree(mPtr);
    }

    CudaMallocBuffer(CudaMallocBuffer const&) = delete;
    CudaMallocBuffer& operator=(CudaMallocBuffer const&) = delete;

    [[nodiscard]] void* ptr() const noexcept
    {
        return mPtr;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return mSize;
    }

    [[nodiscard]] MemoryDesc desc() const
    {
        return MemoryDesc{mPtr, mSize, static_cast<uint32_t>(mDeviceId)};
    }

private:
    void* mPtr{nullptr};
    size_t mSize{0};
    int mDeviceId{0};
};

// ============================================================================
// Helpers: fill / verify GPU buffers, poll for remote descs availability.
// ============================================================================

void fillGpu(void* dst, uint8_t pattern, size_t bytes)
{
    TLLM_CUDA_CHECK(cudaMemset(dst, pattern, bytes));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
}

void expectGpuPattern(void const* src, uint8_t pattern, size_t bytes, char const* label)
{
    std::vector<uint8_t> host(bytes);
    TLLM_CUDA_CHECK(cudaMemcpy(host.data(), src, bytes, cudaMemcpyDeviceToHost));
    // Spot-check head, mid, tail; full memcmp if small enough. Keeps assertion output readable.
    if (bytes <= 4096)
    {
        for (size_t i = 0; i < bytes; ++i)
        {
            ASSERT_EQ(host[i], pattern) << label << " mismatch at byte " << i;
        }
        return;
    }
    size_t const checkpoints[] = {0, bytes / 2, bytes - 1};
    for (size_t off : checkpoints)
    {
        ASSERT_EQ(host[off], pattern) << label << " mismatch at byte " << off;
    }
}

void waitForRemoteDescs(NixlTransferAgent& agent, std::string const& remote, MemoryDescs const& descs)
{
    // NIXL backend exchanges metadata asynchronously on a progress thread; poll briefly.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!agent.checkRemoteDescs(remote, descs))
    {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "timed out waiting for remote descs";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// ============================================================================
// Fixture: two in-process NixlTransferAgents, one CUDA device.
// ============================================================================

class NixlP2pE2ETest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int deviceCount = 0;
        ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
        if (deviceCount == 0)
        {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS);

        // Probe whether the device supports any P2P-shareable handle type BEFORE spinning
        // up agents. Without this the agents would still construct but exporter().isSupported()
        // would be false, getLocalAgentDesc would return an empty p2pBlob, and every test
        // would just exercise the NIXL fallback — missing the whole point of this file.
        auto probe = VmmGpuBuffer::tryCreate(4096, 0);
        if (!probe)
        {
            GTEST_SKIP() << "Neither POSIX_FILE_DESCRIPTOR nor FABRIC VMM handle supported on this device";
        }
    }

    std::unique_ptr<NixlTransferAgent> makeAgent(std::string const& name)
    {
        BaseAgentConfig cfg{name, /*useProgThread=*/true, /*multiThread=*/false, /*useListenThread=*/true,
            /*enableTelemetry=*/false, /*backendParams=*/{}};
        return std::make_unique<NixlTransferAgent>(cfg);
    }

    // Wires agent0 <-> agent1 in both directions using AgentDesc-with-p2pBlob
    // (the code path taken by submitTransferRequests in production; the
    // getLocalConnectionInfo/string overload does NOT propagate p2pBlob).
    void connectWithAgentDesc(NixlTransferAgent& a, NixlTransferAgent& b, std::string const& bName)
    {
        AgentDesc bDesc = b.getLocalAgentDesc();
        a.loadRemoteAgent(bName, bDesc);
    }
};

} // namespace

// ============================================================================
// Case A — pure-P2P WRITE and READ, round-tripped both directions.
// ============================================================================
//
// Each agent registers one VMM VRAM region (P2P-exportable). We drive a WRITE from
// agent0 into agent1's region, then a READ from agent0 out of agent1's region, and
// repeat symmetrically. Assertions:
//   1) TransferState::kSUCCESS on every status
//   2) Destination pattern matches source (memcpy on device -> pattern readback)
//   3) pureP2pCount == number of submits; pureNixlCount == 0; mixedCount == 0
//   4) cubSubmit + memcpyBatchSubmit == pureP2p (so no submit silently bypassed the P2P path)

TEST_F(NixlP2pE2ETest, CaseA_PureP2pWriteAndRead)
{
    constexpr size_t kBufBytes = 256 * 1024; // 256 KiB — > default 16 KiB threshold, routes to memcpyBatch
    auto a0Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    auto a1Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a0Buf, nullptr);
    ASSERT_NE(a1Buf, nullptr);

    auto agent0 = makeAgent("agent0");
    auto agent1 = makeAgent("agent1");

    MemoryDescs a0Descs{MemoryType::kVRAM, {a0Buf->desc()}};
    MemoryDescs a1Descs{MemoryType::kVRAM, {a1Buf->desc()}};

    agent0->registerMemory(a0Descs);
    agent1->registerMemory(a1Descs);

    // Bidirectional connect: each side needs the other's AgentDesc (p2pBlob) before
    // any submit goes out.
    connectWithAgentDesc(*agent0, *agent1, "agent1");
    connectWithAgentDesc(*agent1, *agent0, "agent0");

    waitForRemoteDescs(*agent0, "agent1", a1Descs);
    waitForRemoteDescs(*agent1, "agent0", a0Descs);

    // -- WRITE agent0 -> agent1 --
    constexpr uint8_t kWritePattern = 0xA3;
    fillGpu(a0Buf->ptr(), kWritePattern, kBufBytes);
    fillGpu(a1Buf->ptr(), 0x00, kBufBytes);
    {
        TransferRequest req{TransferOp::kWRITE, a0Descs, a1Descs, "agent1"};
        auto status = agent0->submitTransferRequests(req);
        ASSERT_NE(status, nullptr);
        EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    }
    expectGpuPattern(a1Buf->ptr(), kWritePattern, kBufBytes, "after WRITE a0->a1");

    // -- READ agent0 <- agent1 (pull a1's contents into a0) --
    constexpr uint8_t kReadPattern = 0x5C;
    fillGpu(a1Buf->ptr(), kReadPattern, kBufBytes);
    fillGpu(a0Buf->ptr(), 0x00, kBufBytes);
    {
        TransferRequest req{TransferOp::kREAD, a0Descs, a1Descs, "agent1"};
        auto status = agent0->submitTransferRequests(req);
        ASSERT_NE(status, nullptr);
        EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    }
    expectGpuPattern(a0Buf->ptr(), kReadPattern, kBufBytes, "after READ a0<-a1");

    // -- Path counters: both submits went through the pure-P2P branch --
    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, 2u) << "expected 2 pure-P2P submits";
    EXPECT_EQ(snap.mixed, 0u);
    EXPECT_EQ(snap.pureNixl, 0u);
    EXPECT_EQ(snap.cubSubmit + snap.memcpyBatchSubmit, snap.pureP2p + snap.mixed)
        << "cub+memcpyBatch must equal number of submits that entered the P2P path";
    EXPECT_EQ(snap.memcpyBatchSingleThread + snap.memcpyBatchMultiThread, snap.memcpyBatchSubmit)
        << "single+multi thread must equal total memcpyBatch submits";

    // -- Symmetric direction: agent1 -> agent0 --
    fillGpu(a1Buf->ptr(), 0x11, kBufBytes);
    fillGpu(a0Buf->ptr(), 0x00, kBufBytes);
    {
        TransferRequest req{TransferOp::kWRITE, a1Descs, a0Descs, "agent0"};
        auto status = agent1->submitTransferRequests(req);
        ASSERT_NE(status, nullptr);
        EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    }
    expectGpuPattern(a0Buf->ptr(), 0x11, kBufBytes, "after WRITE a1->a0");

    auto snap1 = agent1->getPathCounters();
    EXPECT_EQ(snap1.pureP2p, 1u);
    EXPECT_EQ(snap1.mixed, 0u);
    EXPECT_EQ(snap1.pureNixl, 0u);

    agent0->invalidateRemoteAgent("agent1");
    agent1->invalidateRemoteAgent("agent0");
    agent0->deregisterMemory(a0Descs);
    agent1->deregisterMemory(a1Descs);
}

// ============================================================================
// Sanity: a submit before loadRemoteAgent must fall through to pure-NIXL (no mapping
// exists). This is the bottom of the eligibility chain at transferAgent.cpp:796.
// Kept in this file because it shares the fixture — dedicated eligibility-matrix
// coverage lives in a separate test body once #2 lands.
// ============================================================================

TEST_F(NixlP2pE2ETest, NoRemoteMapping_FallsBackToPureNixl)
{
    constexpr size_t kBufBytes = 128 * 1024;
    auto a0Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    auto a1Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a0Buf, nullptr);
    ASSERT_NE(a1Buf, nullptr);

    auto agent0 = makeAgent("agent0_nomap");
    auto agent1 = makeAgent("agent1_nomap");

    MemoryDescs a0Descs{MemoryType::kVRAM, {a0Buf->desc()}};
    MemoryDescs a1Descs{MemoryType::kVRAM, {a1Buf->desc()}};

    agent0->registerMemory(a0Descs);
    agent1->registerMemory(a1Descs);

    // Connect via the string-connectionInfo overload — this path does NOT carry p2pBlob,
    // so agent0 never imports agent1's P2P handles. submitTransferRequests should therefore
    // see remoteMapping == nullptr and take the pure-NIXL branch.
    auto connInfo1 = agent1->getLocalConnectionInfo();
    agent0->loadRemoteAgent("agent1_nomap", connInfo1);
    waitForRemoteDescs(*agent0, "agent1_nomap", a1Descs);

    fillGpu(a0Buf->ptr(), 0x77, kBufBytes);
    fillGpu(a1Buf->ptr(), 0x00, kBufBytes);

    TransferRequest req{TransferOp::kWRITE, a0Descs, a1Descs, "agent1_nomap"};
    auto status = agent0->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    expectGpuPattern(a1Buf->ptr(), 0x77, kBufBytes, "after pure-NIXL WRITE");

    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, 0u);
    EXPECT_EQ(snap.mixed, 0u);
    EXPECT_EQ(snap.pureNixl, 1u) << "submit without a P2P mapping must route through NIXL";
    EXPECT_EQ(snap.cubSubmit, 0u);
    EXPECT_EQ(snap.memcpyBatchSubmit, 0u);

    agent0->invalidateRemoteAgent("agent1_nomap");
    agent0->deregisterMemory(a0Descs);
    agent1->deregisterMemory(a1Descs);
}

// ============================================================================
// Case C — mixed P2P + NIXL in a single request.
// ============================================================================
//
// To force a partial remote mapping we exploit the exporter's "same handle type only"
// rule (p2pTransferAgent.cu: see mDetectedHandleType mismatch check): if agent1
// registers a VMM buffer first, the exporter locks in POSIX-FD/FABRIC. When we then
// register a cudaMalloc buffer, its CudaIpc handle type mismatches and the cudaMalloc
// region is silently excluded from the p2pBlob. On agent0 the imported mapping
// therefore covers only the VMM region; a transfer that targets both regions in a
// single TransferRequest splits at translate() — VMM segment goes P2P, cudaMalloc
// segment falls through to NIXL — and the composite MixedTransferStatus waits on both.

TEST_F(NixlP2pE2ETest, CaseC_MixedP2pAndNixl)
{
    constexpr size_t kBufBytes = 128 * 1024;

    // Agent1 hosts two regions: exportable VMM + non-exportable cudaMalloc.
    auto a1VmmBuf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a1VmmBuf, nullptr);
    CudaMallocBuffer a1CudaBuf(kBufBytes, 0);

    // Agent0's local src buffers. a0CudaBuf MUST be cudaMalloc (not VMM) so its
    // byte length matches a1CudaBuf's on the transfer desc — NIXL's createXferReq
    // rejects mismatched src/dst lengths, and VmmGpuBuffer rounds size up to the
    // VMM granularity (typically 2 MiB), which would not equal a cudaMalloc request.
    auto a0VmmBuf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a0VmmBuf, nullptr);
    CudaMallocBuffer a0CudaBuf(kBufBytes, 0);

    auto agent0 = makeAgent("agent0_mixed");
    auto agent1 = makeAgent("agent1_mixed");

    // IMPORTANT: register VMM first so the exporter locks to POSIX-FD / FABRIC. The
    // cudaMalloc register call that follows will be skipped from p2pBlob due to the
    // handle-type mismatch rule, which is exactly the asymmetric mapping we need.
    MemoryDescs a1VmmDescs{MemoryType::kVRAM, {a1VmmBuf->desc()}};
    MemoryDescs a1CudaDescs{MemoryType::kVRAM, {a1CudaBuf.desc()}};
    MemoryDescs a0VmmDescs{MemoryType::kVRAM, {a0VmmBuf->desc()}};
    MemoryDescs a0CudaDescs{MemoryType::kVRAM, {a0CudaBuf.desc()}};

    agent1->registerMemory(a1VmmDescs);
    agent1->registerMemory(a1CudaDescs);
    agent0->registerMemory(a0VmmDescs);
    agent0->registerMemory(a0CudaDescs);

    connectWithAgentDesc(*agent0, *agent1, "agent1_mixed");
    waitForRemoteDescs(*agent0, "agent1_mixed", a1VmmDescs);
    waitForRemoteDescs(*agent0, "agent1_mixed", a1CudaDescs);

    constexpr uint8_t kVmmPattern = 0xEE;
    constexpr uint8_t kCudaPattern = 0x22;
    fillGpu(a0VmmBuf->ptr(), kVmmPattern, kBufBytes);
    fillGpu(a0CudaBuf.ptr(), kCudaPattern, kBufBytes);
    fillGpu(a1VmmBuf->ptr(), 0x00, kBufBytes);
    fillGpu(a1CudaBuf.ptr(), 0x00, kBufBytes);

    // Single request, two descs: index 0 targets the P2P-mapped region, index 1 the
    // unmapped region. The bucketing loop in submitTransferRequests will split them.
    MemoryDescs srcMixed{MemoryType::kVRAM, {a0VmmBuf->desc(), a0CudaBuf.desc()}};
    MemoryDescs dstMixed{MemoryType::kVRAM, {a1VmmBuf->desc(), a1CudaBuf.desc()}};

    TransferRequest req{TransferOp::kWRITE, srcMixed, dstMixed, "agent1_mixed"};
    auto status = agent0->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(), TransferState::kSUCCESS);

    // Both halves must have reached the destination.
    expectGpuPattern(a1VmmBuf->ptr(), kVmmPattern, kBufBytes, "VMM half (via P2P)");
    expectGpuPattern(a1CudaBuf.ptr(), kCudaPattern, kBufBytes, "cudaMalloc half (via NIXL)");

    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, 0u) << "partial mapping must not fire the pure-P2P branch";
    EXPECT_EQ(snap.mixed, 1u) << "exactly one mixed submit expected";
    EXPECT_EQ(snap.pureNixl, 0u) << "neither segment should force full NIXL fallback";
    // Mixed path submits both halves — the P2P half counts one cub/memcpyBatch select.
    EXPECT_EQ(snap.cubSubmit + snap.memcpyBatchSubmit, 1u) << "mixed path submits exactly one P2P half";

    agent0->invalidateRemoteAgent("agent1_mixed");
    agent0->deregisterMemory(a0VmmDescs);
    agent0->deregisterMemory(a0CudaDescs);
    agent1->deregisterMemory(a1VmmDescs);
    agent1->deregisterMemory(a1CudaDescs);
}

// ============================================================================
// Eligibility: sync-message present forces pure-NIXL even with a valid mapping.
// ============================================================================
//
// NIXL's notification channel is required to deliver the sync message to the peer,
// and the P2P fast path has no such channel — so the eligibility gate at
// transferAgent.cpp:796 must route the entire request to NIXL the moment a sync
// message is attached, regardless of whether P2P would otherwise be available.

TEST_F(NixlP2pE2ETest, Eligibility_SyncMessageForcesNixl)
{
    constexpr size_t kBufBytes = 128 * 1024;
    auto a0Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    auto a1Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a0Buf, nullptr);
    ASSERT_NE(a1Buf, nullptr);

    auto agent0 = makeAgent("agent0_sync");
    auto agent1 = makeAgent("agent1_sync");

    MemoryDescs a0Descs{MemoryType::kVRAM, {a0Buf->desc()}};
    MemoryDescs a1Descs{MemoryType::kVRAM, {a1Buf->desc()}};

    agent0->registerMemory(a0Descs);
    agent1->registerMemory(a1Descs);

    // Full AgentDesc exchange so a P2P mapping exists; this is exactly the setup where
    // we want to prove the sync-message path disables the fast path.
    connectWithAgentDesc(*agent0, *agent1, "agent1_sync");
    waitForRemoteDescs(*agent0, "agent1_sync", a1Descs);

    fillGpu(a0Buf->ptr(), 0x99, kBufBytes);
    fillGpu(a1Buf->ptr(), 0x00, kBufBytes);

    SyncMessage syncMsg = "test-sync-marker";
    TransferRequest req{TransferOp::kWRITE, a0Descs, a1Descs, "agent1_sync", syncMsg};
    auto status = agent0->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    expectGpuPattern(a1Buf->ptr(), 0x99, kBufBytes, "sync-forced NIXL WRITE");

    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, 0u);
    EXPECT_EQ(snap.mixed, 0u);
    EXPECT_EQ(snap.pureNixl, 1u) << "sync_message must force pure-NIXL even with a valid mapping";

    agent0->invalidateRemoteAgent("agent1_sync");
    agent0->deregisterMemory(a0Descs);
    agent1->deregisterMemory(a1Descs);
}

// ============================================================================
// Eligibility: DRAM<->DRAM transfers bypass the P2P fast path unconditionally.
// ============================================================================

TEST_F(NixlP2pE2ETest, Eligibility_DramBypassesP2p)
{
    constexpr size_t kBufBytes = 4096;
    std::vector<char> a0Host(kBufBytes, char{0x4C});
    std::vector<char> a1Host(kBufBytes, char{0x00});

    auto agent0 = makeAgent("agent0_dram");
    auto agent1 = makeAgent("agent1_dram");

    MemoryDescs a0Descs{MemoryType::kDRAM, {MemoryDesc{a0Host}}};
    MemoryDescs a1Descs{MemoryType::kDRAM, {MemoryDesc{a1Host}}};

    agent0->registerMemory(a0Descs);
    agent1->registerMemory(a1Descs);

    connectWithAgentDesc(*agent0, *agent1, "agent1_dram");
    waitForRemoteDescs(*agent0, "agent1_dram", a1Descs);

    TransferRequest req{TransferOp::kWRITE, a0Descs, a1Descs, "agent1_dram"};
    auto status = agent0->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    EXPECT_EQ(std::memcmp(a0Host.data(), a1Host.data(), kBufBytes), 0);

    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, 0u);
    EXPECT_EQ(snap.mixed, 0u);
    EXPECT_EQ(snap.pureNixl, 1u) << "non-VRAM transfers must bypass P2P";

    agent0->invalidateRemoteAgent("agent1_dram");
    agent0->deregisterMemory(a0Descs);
    agent1->deregisterMemory(a1Descs);
}

// ============================================================================
// Eligibility under env flags: separate-process tests.
// ============================================================================
//
// TRTLLM_KV_TRANSFER_P2P_DISABLE and TRTLLM_KV_TRANSFER_P2P_MIXED_DISABLE are both
// read through `static bool const = getBoolEnv(...)` in envUtils.cpp. The value is
// latched on first call, so we cannot flip them at runtime from inside a test body.
// These two cases run only when the env is set at process launch; otherwise they
// GTEST_SKIP. Drive them from CTest via set_tests_properties(... ENVIRONMENT ...)
// or a wrapper shell that sets the env before invoking the binary with
// --gtest_filter=NixlP2pE2ETest.EnvGated_P2pDisabled. Same pattern for MixedDisabled.

TEST_F(NixlP2pE2ETest, EnvGated_P2pDisabled)
{
    if (!std::getenv("TRTLLM_KV_TRANSFER_P2P_DISABLE"))
    {
        GTEST_SKIP() << "Set TRTLLM_KV_TRANSFER_P2P_DISABLE=1 before launching the test binary";
    }

    constexpr size_t kBufBytes = 128 * 1024;
    auto a0Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    auto a1Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a0Buf, nullptr);
    ASSERT_NE(a1Buf, nullptr);

    auto agent0 = makeAgent("agent0_p2pdisabled");
    auto agent1 = makeAgent("agent1_p2pdisabled");

    MemoryDescs a0Descs{MemoryType::kVRAM, {a0Buf->desc()}};
    MemoryDescs a1Descs{MemoryType::kVRAM, {a1Buf->desc()}};

    agent0->registerMemory(a0Descs);
    agent1->registerMemory(a1Descs);

    // Even with a full AgentDesc exchange, mP2pAgent is null on both sides so no
    // mapping is imported; every submit must route through pure-NIXL.
    connectWithAgentDesc(*agent0, *agent1, "agent1_p2pdisabled");
    waitForRemoteDescs(*agent0, "agent1_p2pdisabled", a1Descs);

    fillGpu(a0Buf->ptr(), 0xB5, kBufBytes);
    fillGpu(a1Buf->ptr(), 0x00, kBufBytes);

    TransferRequest req{TransferOp::kWRITE, a0Descs, a1Descs, "agent1_p2pdisabled"};
    auto status = agent0->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    expectGpuPattern(a1Buf->ptr(), 0xB5, kBufBytes, "P2P-disabled WRITE");

    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, 0u);
    EXPECT_EQ(snap.mixed, 0u);
    EXPECT_EQ(snap.pureNixl, 1u);
    EXPECT_EQ(snap.cubSubmit, 0u);
    EXPECT_EQ(snap.memcpyBatchSubmit, 0u);
    EXPECT_EQ(snap.memcpyBatchSingleThread, 0u);
    EXPECT_EQ(snap.memcpyBatchMultiThread, 0u);
}

TEST_F(NixlP2pE2ETest, EnvGated_MixedDisabledFallsBackFullNixl)
{
    if (!std::getenv("TRTLLM_KV_TRANSFER_P2P_MIXED_DISABLE"))
    {
        GTEST_SKIP() << "Set TRTLLM_KV_TRANSFER_P2P_MIXED_DISABLE=1 before launching the test binary";
    }

    // Reproduce CaseC_MixedP2pAndNixl's topology (one VMM + one cudaMalloc on the
    // receiver, so the mapping is partial). With MIXED_DISABLE set we expect the
    // mixedDisabled safety valve to fire: the entire request is pushed through NIXL
    // and both halves should still arrive correctly.
    constexpr size_t kBufBytes = 128 * 1024;
    auto a1VmmBuf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a1VmmBuf, nullptr);
    CudaMallocBuffer a1CudaBuf(kBufBytes, 0);
    auto a0VmmBuf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a0VmmBuf, nullptr);
    CudaMallocBuffer a0CudaBuf(kBufBytes, 0);

    auto agent0 = makeAgent("agent0_mixdisabled");
    auto agent1 = makeAgent("agent1_mixdisabled");

    MemoryDescs a1VmmDescs{MemoryType::kVRAM, {a1VmmBuf->desc()}};
    MemoryDescs a1CudaDescs{MemoryType::kVRAM, {a1CudaBuf.desc()}};
    MemoryDescs a0VmmDescs{MemoryType::kVRAM, {a0VmmBuf->desc()}};
    MemoryDescs a0CudaDescs{MemoryType::kVRAM, {a0CudaBuf.desc()}};

    agent1->registerMemory(a1VmmDescs);
    agent1->registerMemory(a1CudaDescs);
    agent0->registerMemory(a0VmmDescs);
    agent0->registerMemory(a0CudaDescs);
    connectWithAgentDesc(*agent0, *agent1, "agent1_mixdisabled");
    waitForRemoteDescs(*agent0, "agent1_mixdisabled", a1VmmDescs);
    waitForRemoteDescs(*agent0, "agent1_mixdisabled", a1CudaDescs);

    fillGpu(a0VmmBuf->ptr(), 0xD1, kBufBytes);
    fillGpu(a0CudaBuf.ptr(), 0x3F, kBufBytes);
    fillGpu(a1VmmBuf->ptr(), 0x00, kBufBytes);
    fillGpu(a1CudaBuf.ptr(), 0x00, kBufBytes);

    MemoryDescs srcMixed{MemoryType::kVRAM, {a0VmmBuf->desc(), a0CudaBuf.desc()}};
    MemoryDescs dstMixed{MemoryType::kVRAM, {a1VmmBuf->desc(), a1CudaBuf.desc()}};

    TransferRequest req{TransferOp::kWRITE, srcMixed, dstMixed, "agent1_mixdisabled"};
    auto status = agent0->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(), TransferState::kSUCCESS);
    expectGpuPattern(a1VmmBuf->ptr(), 0xD1, kBufBytes, "VMM via full NIXL");
    expectGpuPattern(a1CudaBuf.ptr(), 0x3F, kBufBytes, "cudaMalloc via full NIXL");

    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, 0u);
    EXPECT_EQ(snap.mixed, 0u) << "MIXED_DISABLE must skip the mixed branch";
    EXPECT_EQ(snap.pureNixl, 1u) << "partial mapping + MIXED_DISABLE -> full NIXL fallback";
}

// ============================================================================
// Lifecycle: invalidateRemoteAgent -> reload must restore the P2P fast path.
// ============================================================================
//
// NixlTransferAgent::invalidateRemoteAgent unloads BOTH the NIXL remote metadata
// AND the P2P registry entry — after it runs, further submits throw NIXL_ERR_NOT_FOUND.
// The only meaningful lifecycle check is therefore: after a full invalidate+reload
// cycle, does the P2P fast path come back online (i.e., registry.cleanup is idempotent
// and importAndMap can run a second time against the same peer name)?
// Analogous to p2pTransferAgentTest.cu:1178 (SequentialAgentsReusingPoolAddress) but
// at the NixlTransferAgent level. A bug in registry cleanup or UDS server restart
// would surface as either a P2pTransfer: mapping failed log, a silent NIXL fallback,
// or a crash when the second UDS connect runs.

TEST_F(NixlP2pE2ETest, Lifecycle_InvalidateAndReload)
{
    constexpr size_t kBufBytes = 128 * 1024;
    auto a0Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    auto a1Buf = VmmGpuBuffer::tryCreate(kBufBytes, 0);
    ASSERT_NE(a0Buf, nullptr);
    ASSERT_NE(a1Buf, nullptr);

    auto agent0 = makeAgent("agent0_life");
    auto agent1 = makeAgent("agent1_life");

    MemoryDescs a0Descs{MemoryType::kVRAM, {a0Buf->desc()}};
    MemoryDescs a1Descs{MemoryType::kVRAM, {a1Buf->desc()}};
    agent0->registerMemory(a0Descs);
    agent1->registerMemory(a1Descs);

    auto submitWrite = [&](uint8_t pattern)
    {
        fillGpu(a0Buf->ptr(), pattern, kBufBytes);
        fillGpu(a1Buf->ptr(), 0x00, kBufBytes);
        TransferRequest req{TransferOp::kWRITE, a0Descs, a1Descs, "agent1_life"};
        auto status = agent0->submitTransferRequests(req);
        ASSERT_NE(status, nullptr);
        ASSERT_EQ(status->wait(), TransferState::kSUCCESS);
        expectGpuPattern(a1Buf->ptr(), pattern, kBufBytes, "lifecycle WRITE");
    };

    // Phase 1: connect and submit via P2P.
    connectWithAgentDesc(*agent0, *agent1, "agent1_life");
    waitForRemoteDescs(*agent0, "agent1_life", a1Descs);
    submitWrite(0xAA);
    {
        auto s = agent0->getPathCounters();
        EXPECT_EQ(s.pureP2p, 1u);
        EXPECT_EQ(s.pureNixl, 0u);
        EXPECT_EQ(s.mixed, 0u);
    }

    // Phase 2: full invalidate — this clears NIXL remote MD AND P2P registry entry.
    // A submit between now and the reload would throw NIXL_ERR_NOT_FOUND, so we go
    // straight to the reload.
    agent0->invalidateRemoteAgent("agent1_life");

    // Phase 3: reload with a fresh AgentDesc; P2P must come back online.
    connectWithAgentDesc(*agent0, *agent1, "agent1_life");
    waitForRemoteDescs(*agent0, "agent1_life", a1Descs);
    submitWrite(0xCC);
    {
        auto s = agent0->getPathCounters();
        EXPECT_EQ(s.pureP2p, 2u) << "reload should restore P2P fast path";
        EXPECT_EQ(s.pureNixl, 0u) << "neither submit should have fallen back to NIXL";
        EXPECT_EQ(s.mixed, 0u);
    }

    agent0->invalidateRemoteAgent("agent1_life");
    agent0->deregisterMemory(a0Descs);
    agent1->deregisterMemory(a1Descs);
}

// ============================================================================
// Concurrency: multi-thread caller submits on a single agent pair.
// ============================================================================
//
// Four host threads call submitTransferRequests concurrently on disjoint regions.
// Stresses (1) per-context worker pool ownership (c9f56a0a), (2) registry thread
// safety on the read side, (3) cross-thread counter correctness. Each thread does
// kPerThread submits so expected pureP2pCount == kThreads * kPerThread.

TEST_F(NixlP2pE2ETest, Concurrency_MultiThreadSubmit)
{
    constexpr int kThreads = 4;
    constexpr int kPerThread = 4;
    constexpr size_t kBufBytes = 64 * 1024;

    // Each thread owns its own src/dst pair — disjoint regions avoid data races on
    // the buffers themselves; the concurrency is in the path through the agent.
    struct BufPair
    {
        std::unique_ptr<VmmGpuBuffer> src;
        std::unique_ptr<VmmGpuBuffer> dst;
    };

    std::vector<BufPair> bufs;
    bufs.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t)
    {
        BufPair bp;
        bp.src = VmmGpuBuffer::tryCreate(kBufBytes, 0);
        bp.dst = VmmGpuBuffer::tryCreate(kBufBytes, 0);
        ASSERT_NE(bp.src, nullptr);
        ASSERT_NE(bp.dst, nullptr);
        bufs.push_back(std::move(bp));
    }

    auto agent0 = makeAgent("agent0_conc");
    auto agent1 = makeAgent("agent1_conc");

    // Register all src regions on agent0 and all dst regions on agent1.
    std::vector<MemoryDescs> srcDescsVec, dstDescsVec;
    srcDescsVec.reserve(kThreads);
    dstDescsVec.reserve(kThreads);
    for (auto const& bp : bufs)
    {
        srcDescsVec.emplace_back(MemoryType::kVRAM, std::vector<MemoryDesc>{bp.src->desc()});
        dstDescsVec.emplace_back(MemoryType::kVRAM, std::vector<MemoryDesc>{bp.dst->desc()});
        agent0->registerMemory(srcDescsVec.back());
        agent1->registerMemory(dstDescsVec.back());
    }
    connectWithAgentDesc(*agent0, *agent1, "agent1_conc");
    for (auto const& d : dstDescsVec)
    {
        waitForRemoteDescs(*agent0, "agent1_conc", d);
    }

    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t)
    {
        workers.emplace_back(
            [&, t]()
            {
                TLLM_CUDA_CHECK(cudaSetDevice(0));
                for (int it = 0; it < kPerThread; ++it)
                {
                    uint8_t const pattern = static_cast<uint8_t>((t * 37 + it * 11 + 1) & 0xFF);
                    fillGpu(bufs[t].src->ptr(), pattern, kBufBytes);
                    fillGpu(bufs[t].dst->ptr(), 0x00, kBufBytes);

                    TransferRequest req{TransferOp::kWRITE, srcDescsVec[t], dstDescsVec[t], "agent1_conc"};
                    auto status = agent0->submitTransferRequests(req);
                    if (!status || status->wait() != TransferState::kSUCCESS)
                    {
                        failures.fetch_add(1);
                        return;
                    }

                    // Verify this thread's own region. Other threads are writing to
                    // their own disjoint dst, so there's no cross-thread interference.
                    std::vector<uint8_t> host(kBufBytes);
                    TLLM_CUDA_CHECK(cudaMemcpy(host.data(), bufs[t].dst->ptr(), kBufBytes, cudaMemcpyDeviceToHost));
                    if (host[0] != pattern || host[kBufBytes - 1] != pattern)
                    {
                        failures.fetch_add(1);
                        return;
                    }
                }
            });
    }
    for (auto& w : workers)
    {
        w.join();
    }
    EXPECT_EQ(failures.load(), 0);

    auto snap = agent0->getPathCounters();
    EXPECT_EQ(snap.pureP2p, static_cast<uint64_t>(kThreads * kPerThread));
    EXPECT_EQ(snap.mixed, 0u);
    EXPECT_EQ(snap.pureNixl, 0u);
    // Invariants: the decomposition into sub-counters is consistent with the terminal counts.
    EXPECT_EQ(snap.cubSubmit + snap.memcpyBatchSubmit, snap.pureP2p + snap.mixed);
    EXPECT_EQ(snap.memcpyBatchSingleThread + snap.memcpyBatchMultiThread, snap.memcpyBatchSubmit);

    agent0->invalidateRemoteAgent("agent1_conc");
    for (auto const& d : srcDescsVec)
    {
        agent0->deregisterMemory(d);
    }
    for (auto const& d : dstDescsVec)
    {
        agent1->deregisterMemory(d);
    }
}
