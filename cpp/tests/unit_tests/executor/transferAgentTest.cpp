/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serialization.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

using namespace tensorrt_llm::executor::kv_cache;

std::vector<std::string> getAvailableBackends()
{
    std::vector<std::string> backends;

#ifdef TEST_NIXL_BACKEND
    backends.push_back("nixl");
#endif

#ifdef TEST_MOONCAKE_BACKEND
    backends.push_back("mooncake");
#endif

    return backends;
}

class RegisteredHostMemory
{
public:
    RegisteredHostMemory(MemoryDescs mems, BaseTransferAgent* agent)
        : mDescs{std::move(mems)}
        , mAgentPtr{agent}
    {
        TLLM_CHECK(mAgentPtr);
        mAgentPtr->registerMemory(mDescs);
    }

    ~RegisteredHostMemory()
    {
        TLLM_CHECK(mAgentPtr);
        mAgentPtr->deregisterMemory(mDescs);
    }

    [[nodiscard]] MemoryDescs const& getDescs() const noexcept
    {
        return mDescs;
    }

private:
    MemoryDescs mDescs;
    BaseTransferAgent* mAgentPtr{};
};

class TransferAgentTest : public ::testing::TestWithParam<std::string> // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override
    {
        backend = GetParam();
    }

    void TearDown() override {}

    [[nodiscard]] std::unique_ptr<BaseTransferAgent> makeTransferAgent(BaseAgentConfig const& config)
    {
        return tensorrt_llm::executor::kv_cache::makeTransferAgent(backend, &config);
    }

    std::string backend;
};

TEST_P(TransferAgentTest, Basic)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
        // wait for regMem is unpacked by xferAgent0
    } while (!checked);

    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CHECK(memory0 == memory1);
    xferAgent0->invalidateRemoteAgent(agent1);
}

TEST_P(TransferAgentTest, Basic2)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked);

    TransferRequest readReq{TransferOp::kREAD, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(readReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CHECK(memory0 == memory1);

    xferAgent0->invalidateRemoteAgent(agent1);
}

TEST_P(TransferAgentTest, DeviceMemory)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);
    char* dev_ptr0;
    char* dev_ptr1;
    size_t size = 100;
    uint32_t deviceId = 0;
    cudaMalloc(&dev_ptr0, size);
    cudaMalloc(&dev_ptr1, size);
    std::vector<char> memory0(size, 10);
    std::vector<char> memory1(size, 1);
    TLLM_CUDA_CHECK(cudaMemcpy(dev_ptr0, memory0.data(), size, cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(dev_ptr1, memory1.data(), size, cudaMemcpyHostToDevice));
    RegisteredHostMemory regMem0(
        MemoryDescs{MemoryType::kVRAM, {MemoryDesc{dev_ptr0, size, deviceId}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(
        MemoryDescs{MemoryType::kVRAM, {MemoryDesc{dev_ptr1, size, deviceId}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked);
    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CUDA_CHECK(cudaMemcpy(memory0.data(), dev_ptr0, size, cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaMemcpy(memory1.data(), dev_ptr1, size, cudaMemcpyDeviceToHost));

    TLLM_CHECK(memory0 == memory1);

    TLLM_CUDA_CHECK(cudaFree(dev_ptr0));
    TLLM_CUDA_CHECK(cudaFree(dev_ptr1));
    xferAgent0->invalidateRemoteAgent(agent1);
}

TEST_P(TransferAgentTest, Connect)
{

    std::string const agent0{"agent0"}, agent1{"agent1"}, agent2{"agent2"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true},
        config2{agent2, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);
    auto xferAgent2 = makeTransferAgent(config2);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);
    MemoryDescs memDescs0{MemoryType::kDRAM, {MemoryDesc{memory0}}};
    MemoryDescs memDescs1{MemoryType::kDRAM, {MemoryDesc{memory1}}};

    xferAgent0->registerMemory(memDescs0);
    xferAgent1->registerMemory(memDescs1);
    xferAgent2->registerMemory(memDescs0);

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, memDescs1);
    } while (!checked);
    TransferRequest writeReq{TransferOp::kWRITE, memDescs0, memDescs1, agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);

    TLLM_CHECK(memory0 == memory1);
    xferAgent2->loadRemoteAgent(agent1, connectionInfo);
    checked = false;
    do
    {
        checked = xferAgent2->checkRemoteDescs(agent1, memDescs1);
    } while (!checked);
    TransferRequest writeReq2{TransferOp::kWRITE, memDescs0, memDescs1, agent1};
    auto status2 = xferAgent2->submitTransferRequests(writeReq2);
    TLLM_CHECK(status2->wait() == TransferState::kSUCCESS);
    TLLM_CHECK(memory0 == memory1);
    xferAgent0->invalidateRemoteAgent(agent1);
    xferAgent2->invalidateRemoteAgent(agent1);
    xferAgent0->deregisterMemory(memDescs0);
    xferAgent1->deregisterMemory(memDescs1);
    xferAgent2->deregisterMemory(memDescs0);
}

TEST_P(TransferAgentTest, SyncMessage)
{
    constexpr std::size_t MAX_QUERY_TIMES = std::numeric_limits<size_t>::max();
    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true, false, true}, config1{agent1, true, false, true};
    auto xferAgent0 = makeTransferAgent(config0);
    auto xferAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(xferAgent0);
    TLLM_CHECK(xferAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent0.get());

    RegisteredHostMemory regMem2(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, xferAgent1.get());
    RegisteredHostMemory regMem3(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, xferAgent1.get());

    // xferAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = xferAgent1->getLocalConnectionInfo();
    xferAgent0->loadRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = xferAgent0->checkRemoteDescs(agent1, regMem3.getDescs());
    } while (!checked);
    auto syncMessage = std::string("agent_sync_message");
    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem3.getDescs(), agent1};
    auto status = xferAgent0->submitTransferRequests(writeReq);
    xferAgent0->notifySyncMessage(agent1, syncMessage);

    auto notif = xferAgent1->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif.size() == 0; i++)
    {
        notif = xferAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(status->wait() == TransferState::kSUCCESS);
    TLLM_CHECK(status->isCompleted());
    TLLM_CHECK(notif.size() == 1);
    TLLM_CHECK(notif[agent0].size() == 1);
    TLLM_CHECK(notif[agent0][0] == syncMessage);

    TLLM_CHECK(memory0 == memory1);

    std::string syncMessage2 = "two_agent_sync_message";
    xferAgent0->notifySyncMessage(agent1, syncMessage2);
    auto notif2 = xferAgent1->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif2.size() == 0; i++)
    {
        notif2 = xferAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif2.size() == 1);
    TLLM_CHECK(notif2[agent0].size() == 1);
    TLLM_CHECK(notif2[agent0][0] == syncMessage2);

    // xferAgent1->loadRemoteAgent(agent0);
    auto connectionInfo2 = xferAgent0->getLocalConnectionInfo();
    xferAgent1->loadRemoteAgent(agent0, connectionInfo2);
    std::string syncMessage3 = "three_agent_sync_message";
    xferAgent1->notifySyncMessage(agent0, syncMessage3);
    auto notif3 = xferAgent0->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif3.size() == 0; i++)
    {
        notif3 = xferAgent0->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif3.size() == 1);
    TLLM_CHECK(notif3[agent1].size() == 1);
    TLLM_CHECK(notif3[agent1][0] == syncMessage3);

    bool checked2 = false;
    do
    {
        checked2 = xferAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked2);

    std::string syncMessage4 = "four_agent_sync_message";
    TransferRequest writeReq1{TransferOp::kWRITE, regMem2.getDescs(), regMem1.getDescs(), agent0};
    auto status1 = xferAgent1->submitTransferRequests(writeReq1);
    xferAgent1->notifySyncMessage(agent0, syncMessage4);

    auto notif4 = xferAgent0->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif4.size() == 0; i++)
    {
        notif4 = xferAgent0->getNotifiedSyncMessages();
    }
    TLLM_CHECK(status1->wait() == TransferState::kSUCCESS);
    TLLM_CHECK(status1->isCompleted());
    TLLM_CHECK(notif4.size() == 1);
    TLLM_CHECK(notif4[agent1].size() == 1);
    TLLM_CHECK(notif4[agent1][0] == syncMessage4);

    TLLM_CHECK(memory0 == memory1);

    // serialization

    CommState state{std::vector<SocketState>{SocketState{1234, "127.0.0.1"}}, 0};
    using namespace tensorrt_llm::executor;
    std::stringstream ss;
    Serialization::serialize(state, ss);
    std::string serializedState = ss.str();
    xferAgent0->notifySyncMessage(agent1, serializedState);
    auto notif5 = xferAgent1->getNotifiedSyncMessages();
    for (size_t i = 0; i < MAX_QUERY_TIMES && notif5.size() == 0; i++)
    {
        notif5 = xferAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif5.size() == 1);
    TLLM_CHECK(notif5[agent0].size() == 1);
    TLLM_CHECK(notif5[agent0][0] == serializedState);
    std::stringstream ss2(notif5[agent0][0]);
    auto state2 = Serialization::deserializeCommState(ss2);
    TLLM_CHECK(state2 == state);

    xferAgent0->invalidateRemoteAgent(agent1);
    xferAgent1->invalidateRemoteAgent(agent0);
}

INSTANTIATE_TEST_SUITE_P(AvailableBackends, TransferAgentTest, ::testing::ValuesIn(getAvailableBackends()),
    [](::testing::TestParamInfo<TransferAgentTest::ParamType> const& info) { return info.param; });

// ── AgentDesc serialization tests (no backend needed) ──

TEST(AgentDescTest, SerializeDeserializeEmpty)
{
    std::string nixlBlob = "some_nixl_metadata_blob";
    AgentDesc original{nixlBlob};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    EXPECT_EQ(decoded.getBackendAgentDesc(), nixlBlob);
    EXPECT_TRUE(decoded.getVramRegions().empty());
}

TEST(AgentDescTest, SerializeDeserializeWithRegions)
{
    std::string nixlBlob = "nixl_blob_data_here";
    std::vector<VramRegionMeta> regions{
        {0x7f0000000000UL, 67108864, 2097152},   // 64MB pool, 2MB chunks
        {0x7f0004000000UL, 134217728, 33554432}, // 128MB pool, 32MB chunks
    };
    AgentDesc original{nixlBlob, regions};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    EXPECT_EQ(decoded.getBackendAgentDesc(), nixlBlob);
    ASSERT_EQ(decoded.getVramRegions().size(), 2);
    EXPECT_EQ(decoded.getVramRegions()[0].baseAddr, 0x7f0000000000UL);
    EXPECT_EQ(decoded.getVramRegions()[0].totalLen, 67108864);
    EXPECT_EQ(decoded.getVramRegions()[0].chunkSize, 2097152);
    EXPECT_EQ(decoded.getVramRegions()[1].baseAddr, 0x7f0004000000UL);
    EXPECT_EQ(decoded.getVramRegions()[1].totalLen, 134217728);
    EXPECT_EQ(decoded.getVramRegions()[1].chunkSize, 33554432);
}

TEST(AgentDescTest, SerializeDeserializeBinaryBlob)
{
    // NIXL blob can contain arbitrary binary data including null bytes
    std::string nixlBlob(256, '\0');
    for (size_t i = 0; i < nixlBlob.size(); ++i)
    {
        nixlBlob[i] = static_cast<char>(i);
    }
    std::vector<VramRegionMeta> regions{{0xdeadbeefUL, 4096, 4096}};
    AgentDesc original{nixlBlob, regions};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    EXPECT_EQ(decoded.getBackendAgentDesc(), nixlBlob);
    ASSERT_EQ(decoded.getVramRegions().size(), 1);
    EXPECT_EQ(decoded.getVramRegions()[0].baseAddr, 0xdeadbeefUL);
}

TEST(AgentDescTest, SerializeDeserializeUnalignedBase)
{
    // Simulate an unaligned VMM pool base (what happens with cuMemAddressReserve alignment=0)
    uintptr_t unalignedBase = 0x7f0000200000UL; // 2MB aligned but not 32MB aligned
    size_t chunkSize = 33554432;                // 32MB
    size_t totalLen = chunkSize * 4;            // 128MB pool

    std::vector<VramRegionMeta> regions{{unalignedBase, totalLen, chunkSize}};
    AgentDesc original{"blob", regions};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    ASSERT_EQ(decoded.getVramRegions().size(), 1);
    auto const& r = decoded.getVramRegions()[0];
    EXPECT_EQ(r.baseAddr, unalignedBase);
    EXPECT_EQ(r.totalLen, totalLen);
    EXPECT_EQ(r.chunkSize, chunkSize);

    // Verify chunk boundary calculation: (addr - base) % chunkSize
    uintptr_t addr = unalignedBase + chunkSize + 100;
    size_t offsetInChunk = (addr - r.baseAddr) % r.chunkSize;
    EXPECT_EQ(offsetInChunk, 100);

    // addr % chunkSize would give the WRONG answer for unaligned base
    size_t wrongOffset = addr % chunkSize;
    // wrongOffset != 100 in general for unaligned base
    // (only equal if base happens to be chunk-aligned)
    EXPECT_NE(wrongOffset, offsetInChunk);
}

TEST(AgentDescTest, DeserializeTruncatedData)
{
    // Serialize a valid AgentDesc, then truncate the data
    std::string nixlBlob = "nixl_blob_data";
    std::vector<VramRegionMeta> regions{{0x1000UL, 4096, 4096}};
    AgentDesc original{nixlBlob, regions};
    auto serialized = original.serialize();

    // Truncate to half the data
    std::string truncated = serialized.substr(0, serialized.size() / 2);
    EXPECT_THROW(AgentDesc::deserialize(truncated), std::exception);
}

// ── VmmDescSplitter tests (backend-agnostic, no NIXL dependency) ──

TEST(VmmDescSplitterTest, LookupChunkInfoHit)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x300000, 0x100000}; // base=1MB, len=3MB, chunk=1MB
    auto result = VmmDescSplitter::lookupChunkInfo(0x200000, regionMap);
    EXPECT_EQ(result.chunkSize, 0x100000);
    EXPECT_EQ(result.regionBase, 0x100000);
    EXPECT_EQ(result.regionTotalLen, 0x300000);
}

TEST(VmmDescSplitterTest, LookupChunkInfoMiss)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x100000, 0x100000};
    // Address outside the region
    auto result = VmmDescSplitter::lookupChunkInfo(0x300000, regionMap);
    EXPECT_EQ(result.chunkSize, 0u);
    EXPECT_EQ(result.regionBase, 0u);
    EXPECT_EQ(result.regionTotalLen, 0u);
}

TEST(VmmDescSplitterTest, LookupChunkInfoMultipleRegions)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x200000, 0x100000}; // region1: [0x100000, 0x300000)
    regionMap[0x400000] = {0x200000, 0x80000};  // region2: [0x400000, 0x600000)

    auto r1 = VmmDescSplitter::lookupChunkInfo(0x150000, regionMap);
    EXPECT_EQ(r1.chunkSize, 0x100000);
    EXPECT_EQ(r1.regionBase, 0x100000);

    auto r2 = VmmDescSplitter::lookupChunkInfo(0x500000, regionMap);
    EXPECT_EQ(r2.chunkSize, 0x80000);
    EXPECT_EQ(r2.regionBase, 0x400000);

    // Gap between regions
    auto r3 = VmmDescSplitter::lookupChunkInfo(0x350000, regionMap);
    EXPECT_EQ(r3.chunkSize, 0u);
}

TEST(VmmDescSplitterTest, SplitDescsAlignedBase)
{
    // Pool base is chunk-aligned: base=0x200000, chunkSize=0x100000 (1MB)
    VramRegionMap regionMap;
    regionMap[0x200000] = {0x400000, 0x100000}; // 4MB pool, 1MB chunks

    // A 2.5MB descriptor spanning 3 chunks
    std::vector<MemoryDesc> descs{{0x200000, 0x280000, 0}};
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    auto const& out = result.getDescs();

    ASSERT_EQ(out.size(), 3);
    EXPECT_EQ(out[0].getAddr(), 0x200000);
    EXPECT_EQ(out[0].getLen(), 0x100000); // first full chunk
    EXPECT_EQ(out[1].getAddr(), 0x300000);
    EXPECT_EQ(out[1].getLen(), 0x100000); // second full chunk
    EXPECT_EQ(out[2].getAddr(), 0x400000);
    EXPECT_EQ(out[2].getLen(), 0x80000);  // remaining 0.5MB
}

TEST(VmmDescSplitterTest, SplitDescsUnalignedBase)
{
    // Pool base is NOT chunk-aligned: base=0x200000 (2MB), chunkSize=0x2000000 (32MB)
    uintptr_t base = 0x200000;
    size_t chunkSize = 0x2000000;
    VramRegionMap regionMap;
    regionMap[base] = {chunkSize * 3, chunkSize};

    // Descriptor starting at base, spanning 2.5 chunks
    size_t descLen = chunkSize * 2 + chunkSize / 2;
    std::vector<MemoryDesc> descs{{base, descLen, 0}};
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    auto const& out = result.getDescs();

    ASSERT_EQ(out.size(), 3);
    EXPECT_EQ(out[0].getAddr(), base);
    EXPECT_EQ(out[0].getLen(), chunkSize);
    EXPECT_EQ(out[1].getAddr(), base + chunkSize);
    EXPECT_EQ(out[1].getLen(), chunkSize);
    EXPECT_EQ(out[2].getAddr(), base + chunkSize * 2);
    EXPECT_EQ(out[2].getLen(), chunkSize / 2);
}

TEST(VmmDescSplitterTest, SplitDescsNonVmm)
{
    VramRegionMap regionMap;
    // Address not in any region → no split
    std::vector<MemoryDesc> descs{{0x1000, 4096, 0}};
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getLen(), 4096);
}

TEST(VmmDescSplitterTest, SplitDescsNonVramPassthrough)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x200000, 0x100000};
    // DRAM descs should pass through unchanged regardless of region map
    std::vector<MemoryDesc> descs{{0x100000, 0x200000, 0}};
    MemoryDescs input{MemoryType::kDRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getLen(), 0x200000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsDifferentChunkSizes)
{
    // src: 1MB chunks, dst: 512KB chunks
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x100000}; // 4MB, 1MB chunks
    remoteMap[0x800000] = {0x400000, 0x80000}; // 4MB, 512KB chunks

    // Transfer 2MB from src to dst
    std::vector<MemoryDesc> srcDescs{{0x100000, 0x200000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x200000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst]
        = VmmDescSplitter::splitTransferDescsWithRegionMaps(srcInput, dstInput, localMap, remoteMap);

    // dst has smaller chunks (512KB), so we get 4 pieces: 512K, 512K, 512K, 512K
    ASSERT_EQ(splitSrc.getDescs().size(), 4);
    ASSERT_EQ(splitDst.getDescs().size(), 4);
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(splitSrc.getDescs()[i].getLen(), 0x80000);
        EXPECT_EQ(splitDst.getDescs()[i].getLen(), 0x80000);
        EXPECT_EQ(splitSrc.getDescs()[i].getAddr(), 0x100000 + i * 0x80000);
        EXPECT_EQ(splitDst.getDescs()[i].getAddr(), 0x800000 + i * 0x80000);
    }
}

TEST(VmmDescSplitterTest, SplitTransferDescsUnalignedBothSides)
{
    // Both src and dst have unaligned bases with different chunk sizes
    uintptr_t srcBase = 0x200000;  // 2MB (not 4MB aligned)
    uintptr_t dstBase = 0x1800000; // 24MB (not 32MB aligned)
    size_t srcChunk = 0x400000;    // 4MB
    size_t dstChunk = 0x200000;    // 2MB

    VramRegionMap localMap, remoteMap;
    localMap[srcBase] = {srcChunk * 4, srcChunk};
    remoteMap[dstBase] = {dstChunk * 8, dstChunk};

    // Transfer exactly 4MB
    std::vector<MemoryDesc> srcDescs{{srcBase, 0x400000, 0}};
    std::vector<MemoryDesc> dstDescs{{dstBase, 0x400000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst]
        = VmmDescSplitter::splitTransferDescsWithRegionMaps(srcInput, dstInput, localMap, remoteMap);

    // src has 4MB chunks starting at srcBase → 1 piece from src side
    // dst has 2MB chunks starting at dstBase → 2 pieces from dst side
    // min produces 2 pieces: 2MB, 2MB
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x200000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x200000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsNoDstRegion)
{
    // src has VMM chunks, dst has no VMM info (empty remote map) → only src boundaries
    VramRegionMap localMap;
    VramRegionMap emptyRemoteMap;
    localMap[0x100000] = {0x400000, 0x100000}; // 4MB, 1MB chunks

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x200000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x900000, 0x200000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst]
        = VmmDescSplitter::splitTransferDescsWithRegionMaps(srcInput, dstInput, localMap, emptyRemoteMap);

    // Only src boundaries: 2 pieces of 1MB each
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x100000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(splitDst.getDescs()[1].getLen(), 0x100000);
}

// ── splitDescsWithRegionMap: region boundary tests ──

TEST(VmmDescSplitterTest, SplitDescsRegionBoundaryNoChunks)
{
    // Two adjacent regions with chunkSize=0 (cudaMalloc).
    // A descriptor spanning both should be split at the region boundary.
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x100000, 0};                    // regionA: [0x100000, 0x200000), no chunks
    regionMap[0x200000] = {0x100000, 0};                    // regionB: [0x200000, 0x300000), no chunks

    std::vector<MemoryDesc> descs{{0x100000, 0x200000, 0}}; // spans both regions
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    auto const& out = result.getDescs();

    ASSERT_EQ(out.size(), 2);
    EXPECT_EQ(out[0].getAddr(), 0x100000);
    EXPECT_EQ(out[0].getLen(), 0x100000); // regionA portion
    EXPECT_EQ(out[1].getAddr(), 0x200000);
    EXPECT_EQ(out[1].getLen(), 0x100000); // regionB portion
}

TEST(VmmDescSplitterTest, SplitDescsRegionBoundaryWithChunks)
{
    // Region with chunks: desc spans chunks AND region boundary.
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x200000, 0x100000}; // [0x100000, 0x300000), 1MB chunks
    regionMap[0x300000] = {0x100000, 0};        // [0x300000, 0x400000), no chunks

    // Desc spans from middle of first region into second region
    std::vector<MemoryDesc> descs{{0x100000, 0x280000, 0}}; // 2.5MB from 0x100000
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    auto const& out = result.getDescs();

    // First region: 2 chunks of 1MB each
    // Second region: remainder 0x80000
    ASSERT_EQ(out.size(), 3);
    EXPECT_EQ(out[0].getAddr(), 0x100000);
    EXPECT_EQ(out[0].getLen(), 0x100000); // chunk 0
    EXPECT_EQ(out[1].getAddr(), 0x200000);
    EXPECT_EQ(out[1].getLen(), 0x100000); // chunk 1 (ends at region boundary)
    EXPECT_EQ(out[2].getAddr(), 0x300000);
    EXPECT_EQ(out[2].getLen(), 0x80000);  // into second region
}

// ── splitTransferDescsWithRegionMaps: coalesce tests ──

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceDisabled)
{
    // Two contiguous descs within same chunk, coalesce=false → no merging.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x400000}; // 4MB, single 4MB chunk
    remoteMap[0x800000] = {0x400000, 0x400000};

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x80000, 0}, {0x180000, 0x80000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x80000, 0}, {0x880000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/false);

    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceContiguousSameChunk)
{
    // Two contiguous descs within same chunk, coalesce=true → merged into one.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x400000}; // 4MB, single 4MB chunk
    remoteMap[0x800000] = {0x400000, 0x400000};

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x80000, 0}, {0x180000, 0x80000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x80000, 0}, {0x880000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    ASSERT_EQ(splitSrc.getDescs().size(), 1);
    EXPECT_EQ(splitSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000); // 512KB + 512KB = 1MB
    ASSERT_EQ(splitDst.getDescs().size(), 1);
    EXPECT_EQ(splitDst.getDescs()[0].getAddr(), 0x800000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x100000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceAcrossChunkBoundary)
{
    // Two contiguous descs that each fill one chunk → NOT merged (chunk boundary).
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x100000}; // 4MB, 1MB chunks
    remoteMap[0x800000] = {0x400000, 0x100000};

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x100000, 0}, {0x200000, 0x100000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x100000, 0}, {0x900000, 0x100000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // Not merged: 0x200000 is on a chunk boundary
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x100000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceAcrossRegionBoundary)
{
    // Two contiguous descs in different regions (chunkSize=0) → NOT merged.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x100000, 0}; // regionA: [0x100000, 0x200000)
    localMap[0x200000] = {0x100000, 0}; // regionB: [0x200000, 0x300000)
    remoteMap[0x800000] = {0x100000, 0};
    remoteMap[0x900000] = {0x100000, 0};

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x100000, 0}, {0x200000, 0x100000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x100000, 0}, {0x900000, 0x100000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // Not merged: different regions
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalescePartialMerge)
{
    // Three descs: [0]+[1] contiguous in same chunk, [2] in next chunk.
    // With coalesce: [0]+[1] merge, [2] stays separate → 2 pieces.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x200000}; // 4MB, 2MB chunks
    remoteMap[0x800000] = {0x400000, 0x200000};

    std::vector<MemoryDesc> srcDescs{
        {0x100000, 0x80000, 0}, // [0] starts at chunk 0
        {0x180000, 0x80000, 0}, // [1] still in chunk 0, contiguous with [0]
        {0x300000, 0x80000, 0}, // [2] in chunk 1 (starts at 0x300000)
    };
    std::vector<MemoryDesc> dstDescs{
        {0x800000, 0x80000, 0},
        {0x880000, 0x80000, 0},
        {0xA00000, 0x80000, 0},
    };
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // [0]+[1] merged (contiguous, same chunk), [2] separate (gap, different chunk)
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000); // merged [0]+[1]
    EXPECT_EQ(splitSrc.getDescs()[1].getAddr(), 0x300000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x80000);  // [2]
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceNonContiguous)
{
    // Two descs in same chunk but NOT contiguous (gap) → NOT merged.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x400000}; // single chunk
    remoteMap[0x800000] = {0x400000, 0x400000};

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x80000, 0}, {0x200000, 0x80000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x80000, 0}, {0x900000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // Not merged: gap between descs
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceSrcOnlyChunkBoundary)
{
    // src crosses chunk boundary, dst doesn't → NOT merged.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x100000};  // 1MB chunks
    remoteMap[0x800000] = {0x400000, 0x400000}; // single 4MB chunk

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x100000, 0}, {0x200000, 0x100000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x100000, 0}, {0x900000, 0x100000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // src at 0x200000 is on a chunk boundary → NOT merged
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
}

TEST(VmmDescSplitterTest, SplitTransferDescsNoSrcRegionWithCoalesce)
{
    // src has no region info (chunkSize=0, not in map) → split still respects dst boundaries.
    VramRegionMap localMap;                     // empty — src not registered
    VramRegionMap remoteMap;
    remoteMap[0x800000] = {0x400000, 0x100000}; // dst: 4MB, 1MB chunks

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x200000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x200000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // dst has 1MB chunks → 2 pieces
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x100000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceUnsortedInput)
{
    // Descs given in reverse order — internal sort brings them in order → merged.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x400000}; // single 4MB chunk
    remoteMap[0x800000] = {0x400000, 0x400000};

    // Reverse order: second half first, first half second.
    std::vector<MemoryDesc> srcDescs{{0x180000, 0x80000, 0}, {0x100000, 0x80000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x880000, 0x80000, 0}, {0x800000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // After internal sort by src addr: (0x100000) then (0x180000) → contiguous → merged.
    ASSERT_EQ(splitSrc.getDescs().size(), 1);
    EXPECT_EQ(splitSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    ASSERT_EQ(splitDst.getDescs().size(), 1);
    EXPECT_EQ(splitDst.getDescs()[0].getAddr(), 0x800000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x100000);
}

// ── Coalesce within chunkSize=0 (cudaMalloc) region ──

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceWithinCudaMallocRegion)
{
    // Two contiguous descs in the same chunkSize=0 region → MERGED.
    // chunkSize=0 means no chunk boundaries to respect, only region boundaries.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0}; // 4MB, no chunks (cudaMalloc)
    remoteMap[0x800000] = {0x400000, 0};

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x80000, 0}, {0x180000, 0x80000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x80000, 0}, {0x880000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // Both contiguous in same region with no chunk boundaries → merged
    ASSERT_EQ(splitSrc.getDescs().size(), 1);
    EXPECT_EQ(splitSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000); // 512KB + 512KB
    ASSERT_EQ(splitDst.getDescs().size(), 1);
    EXPECT_EQ(splitDst.getDescs()[0].getAddr(), 0x800000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x100000);
}

// ── Coalesce across original descriptor boundaries ──

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceAcrossOriginalDescBoundary)
{
    // desc[0] ends where desc[1] begins, both in the same chunk.
    // The coalesce logic checks splitSrc.back() regardless of which original desc
    // produced it, so the last piece of desc[0] and the first piece of desc[1]
    // should merge.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x400000}; // 4MB, single 4MB chunk
    remoteMap[0x800000] = {0x400000, 0x400000};

    // desc[0]: [0x100000, 0x180000)  desc[1]: [0x180000, 0x200000)
    std::vector<MemoryDesc> srcDescs{{0x100000, 0x80000, 0}, {0x180000, 0x80000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x80000, 0}, {0x880000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // Merged across desc boundary
    ASSERT_EQ(splitSrc.getDescs().size(), 1);
    EXPECT_EQ(splitSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    ASSERT_EQ(splitDst.getDescs().size(), 1);
    EXPECT_EQ(splitDst.getDescs()[0].getAddr(), 0x800000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x100000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceAcrossDescBoundaryBlocked)
{
    // desc[0] ends at chunk boundary, desc[1] starts at next chunk.
    // Even though they are contiguous in virtual address, they span a chunk
    // boundary → NOT merged.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x100000};  // 4MB, 1MB chunks
    remoteMap[0x800000] = {0x400000, 0x400000}; // single 4MB chunk

    // desc[0] fills chunk [0x100000, 0x200000), desc[1] starts chunk [0x200000, 0x300000)
    std::vector<MemoryDesc> srcDescs{{0x100000, 0x100000, 0}, {0x200000, 0x80000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x100000, 0}, {0x900000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // src at 0x200000 is on a chunk boundary → NOT merged across desc boundary
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x80000);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
}

// ── Different deviceId prevents coalesce ──

TEST(VmmDescSplitterTest, SplitTransferDescsCoalesceDifferentDeviceId)
{
    // Two contiguous descs in same chunk but with different deviceIds → NOT merged.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x400000}; // single chunk
    remoteMap[0x800000] = {0x400000, 0x400000};

    // Same addresses but different src deviceIds
    std::vector<MemoryDesc> srcDescs{{0x100000, 0x80000, /*deviceId=*/0}, {0x180000, 0x80000, /*deviceId=*/1}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x80000, 0}, {0x880000, 0x80000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/true);

    // Different src deviceIds → NOT merged
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getDeviceId(), 0u);
    EXPECT_EQ(splitSrc.getDescs()[1].getDeviceId(), 1u);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
}

// ── Transfer desc spanning multiple regions (re-lookup path) ──

TEST(VmmDescSplitterTest, SplitTransferDescsRelookupAcrossRegions)
{
    // A single large transfer desc that spans 3 regions on the src side.
    // Verifies the re-lookup logic in splitTransferDescsWithRegionMaps when
    // srcAddr advances past the current region boundary.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x100000, 0x100000}; // regionA: [0x100000, 0x200000), 1MB chunks
    localMap[0x200000] = {0x100000, 0};        // regionB: [0x200000, 0x300000), no chunks
    localMap[0x300000] = {0x100000, 0x80000};  // regionC: [0x300000, 0x400000), 512KB chunks
    remoteMap[0x800000] = {0x400000, 0};       // single dst region, no chunks

    // src desc spans all 3 regions: 3MB from 0x100000
    std::vector<MemoryDesc> srcDescs{{0x100000, 0x300000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x300000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/false);

    // regionA: 1 chunk of 1MB → 1 piece
    // regionB: 1MB, no chunks → 1 piece
    // regionC: 1MB with 512KB chunks → 2 pieces
    // Total: 4 pieces
    ASSERT_EQ(splitSrc.getDescs().size(), 4);
    ASSERT_EQ(splitDst.getDescs().size(), 4);

    // regionA chunk
    EXPECT_EQ(splitSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    // regionB whole
    EXPECT_EQ(splitSrc.getDescs()[1].getAddr(), 0x200000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x100000);
    // regionC chunk 0
    EXPECT_EQ(splitSrc.getDescs()[2].getAddr(), 0x300000);
    EXPECT_EQ(splitSrc.getDescs()[2].getLen(), 0x80000);
    // regionC chunk 1
    EXPECT_EQ(splitSrc.getDescs()[3].getAddr(), 0x380000);
    EXPECT_EQ(splitSrc.getDescs()[3].getLen(), 0x80000);

    // dst pieces should sum correctly
    size_t totalDst = 0;
    for (auto const& d : splitDst.getDescs())
    {
        totalDst += d.getLen();
    }
    EXPECT_EQ(totalDst, 0x300000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsRelookupBothSides)
{
    // Both src AND dst span multiple regions simultaneously.
    // Verifies re-lookup happens independently on each side.
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x100000, 0};  // srcA: [0x100000, 0x200000), no chunks
    localMap[0x200000] = {0x100000, 0};  // srcB: [0x200000, 0x300000), no chunks
    remoteMap[0x800000] = {0x80000, 0};  // dstA: [0x800000, 0x880000), no chunks
    remoteMap[0x880000] = {0x180000, 0}; // dstB: [0x880000, 0xA00000), no chunks

    // 2MB transfer: src spans srcA+srcB, dst spans dstA+dstB
    std::vector<MemoryDesc> srcDescs{{0x100000, 0x200000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x200000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitTransferDescsWithRegionMaps(
        srcInput, dstInput, localMap, remoteMap, /*enableCoalesce=*/false);

    // First piece: min(srcA remaining=1MB, dstA remaining=512KB) = 512KB
    // Second piece: min(srcA remaining=512KB, dstB remaining=1.5MB) = 512KB
    // Third piece: min(srcB remaining=1MB, dstB remaining=1MB) = 1MB
    // Total: 3 pieces
    ASSERT_EQ(splitSrc.getDescs().size(), 3);
    ASSERT_EQ(splitDst.getDescs().size(), 3);

    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x80000);  // 512KB
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x80000);  // 512KB
    EXPECT_EQ(splitSrc.getDescs()[2].getLen(), 0x100000); // 1MB

    EXPECT_EQ(splitDst.getDescs()[0].getAddr(), 0x800000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x80000);
    EXPECT_EQ(splitDst.getDescs()[1].getAddr(), 0x880000);
    EXPECT_EQ(splitDst.getDescs()[1].getLen(), 0x80000);
    EXPECT_EQ(splitDst.getDescs()[2].getAddr(), 0x900000);
    EXPECT_EQ(splitDst.getDescs()[2].getLen(), 0x100000);
}

// Skip LoopbackAgentTest for mooncake backend for now
#ifdef TEST_NIXL_BACKEND

class LoopbackAgentTest : public ::testing::Test,
                          public ::testing::WithParamInterface<bool> // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override
    {
        static int file_num = 0;
        std::string filename = std::string("test_agent") + std::to_string(file_num++);
        auto dirPath = fs::absolute(filename);
        std::error_code ec;
        fs::create_directories(dirPath, ec);
        TLLM_CHECK_WITH_INFO(!ec, "Failed to create test directory: %s", ec.message().c_str());
        mDirectory = dirPath.string();
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(mDirectory, ec);
        if (ec)
            std::cerr << "Warning: Failed to clean up test directory: " << ec.message() << std::endl;
    }

    [[nodiscard]] std::shared_ptr<BaseLoopbackAgent> makeLoopbackAgent(BaseAgentConfig const& config)
    {
        return tensorrt_llm::executor::kv_cache::makeLoopbackAgent("nixl", &config);
    }

    [[nodiscard]] std::string getDirectory() const
    {
        return mDirectory;
    }

private:
    std::string mDirectory;
};

TEST_P(LoopbackAgentTest, FileToGpu)
{
    std::string const agentName{"loopbackAgent"};
    BaseAgentConfig config{agentName, true, GetParam()};
    auto loopbackAgent = makeLoopbackAgent(config);

    TLLM_CHECK(loopbackAgent);

    std::vector<char> memory(100, 1);
    char* cuda_mem;
    TLLM_CUDA_CHECK(cudaMalloc(&cuda_mem, 100));
    TLLM_CUDA_CHECK(cudaMemcpy(cuda_mem, memory.data(), 100, cudaMemcpyHostToDevice));
    std::string filename = getDirectory() + std::string("/file2gpu.bin");

    int fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for writing", filename.c_str());

    std::vector<char> fileData(100, 10);
    ssize_t bytesWritten = ::write(fd, fileData.data(), fileData.size());
    TLLM_CHECK_WITH_INFO(bytesWritten == static_cast<ssize_t>(fileData.size()), "Failed to write to file");
    ::close(fd);

    {
        MemoryDesc mem_desc(cuda_mem, 100, 0);
        MemoryDescs memDescs{MemoryType::kVRAM, {mem_desc}};

        std::vector<FileDesc> fileDescVec;
        fileDescVec.emplace_back(filename, O_RDONLY, 0664, 100);
        FileDescs fileDescs{std::move(fileDescVec)};

        loopbackAgent->executeLoopbackRequest(memDescs, fileDescs, false);
    }

    TLLM_CUDA_CHECK(cudaMemcpy(memory.data(), cuda_mem, 100, cudaMemcpyDeviceToHost));

    TLLM_CHECK(memory == fileData);
    TLLM_CUDA_CHECK(cudaFree(cuda_mem));
}

TEST_P(LoopbackAgentTest, GpuToFile)
{
    std::string const agentName{"loopbackAgent"};
    BaseAgentConfig config{agentName, true, GetParam()};
    auto loopbackAgent = makeLoopbackAgent(config);

    TLLM_CHECK(loopbackAgent);

    std::vector<char> memory(100, 1);
    char* cuda_mem;
    TLLM_CUDA_CHECK(cudaMalloc(&cuda_mem, 100));
    TLLM_CUDA_CHECK(cudaMemcpy(cuda_mem, memory.data(), 100, cudaMemcpyHostToDevice));
    std::string filename = getDirectory() + std::string("/gpu2file.bin");

    {
        MemoryDesc mem_desc(cuda_mem, 100, 0);
        MemoryDescs memDescs{MemoryType::kVRAM, {mem_desc}};

        std::vector<FileDesc> fileDescVec;
        fileDescVec.emplace_back(filename, O_CREAT | O_WRONLY, 0664, 100);
        FileDescs fileDescs{std::move(fileDescVec)};

        loopbackAgent->executeLoopbackRequest(memDescs, fileDescs, true);
    }

    int fd = ::open(filename.c_str(), O_RDONLY, 0664);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for reading", filename.c_str());

    std::vector<char> fileData(100);
    ssize_t bytesRead = ::read(fd, fileData.data(), fileData.size());
    TLLM_CHECK_WITH_INFO(bytesRead == static_cast<ssize_t>(fileData.size()), "Failed to read from file");
    ::close(fd);

    TLLM_CHECK(fileData == memory);
    TLLM_CUDA_CHECK(cudaFree(cuda_mem));
}

INSTANTIATE_TEST_SUITE_P(, LoopbackAgentTest, ::testing::Values(true, false));

#endif // TEST_NIXL_BACKEND
