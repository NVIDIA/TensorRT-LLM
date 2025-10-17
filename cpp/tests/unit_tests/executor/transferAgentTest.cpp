/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serialization.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace tensorrt_llm::executor::kv_cache;

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

class TransferAgentTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}

    [[nodiscard]] std::unique_ptr<BaseTransferAgent> makeTransferAgent(BaseAgentConfig const& config)
    {
        return tensorrt_llm::executor::kv_cache::makeTransferAgent("nixl", &config);
    }
};

TEST_F(TransferAgentTest, Basic)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true}, config1{agent1, true};
    auto nixlAgent0 = makeTransferAgent(config0);
    auto nixlAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(nixlAgent0);
    TLLM_CHECK(nixlAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, nixlAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, nixlAgent1.get());

    // nixlAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = nixlAgent1->getConnectionInfo();
    nixlAgent0->connectRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = nixlAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
        // wait for regMem is unpacked by nixlAgent0
    } while (!checked);

    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = nixlAgent0->submitTransferRequests(writeReq);
    status->wait();

    TLLM_CHECK(memory0 == memory1);

    nixlAgent0->invalidateRemoteAgent(agent1);
}

TEST_F(TransferAgentTest, Basic2)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true}, config1{agent1, true};
    auto nixlAgent0 = makeTransferAgent(config0);
    auto nixlAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(nixlAgent0);
    TLLM_CHECK(nixlAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, nixlAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, nixlAgent1.get());

    // nixlAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = nixlAgent1->getConnectionInfo();
    nixlAgent0->connectRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = nixlAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked);

    TransferRequest readReq{TransferOp::kREAD, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = nixlAgent0->submitTransferRequests(readReq);
    status->wait();

    TLLM_CHECK(memory0 == memory1);

    nixlAgent0->invalidateRemoteAgent(agent1);
}

TEST_F(TransferAgentTest, DeviceMemory)
{

    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true}, config1{agent1, true};
    auto nixlAgent0 = makeTransferAgent(config0);
    auto nixlAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(nixlAgent0);
    TLLM_CHECK(nixlAgent1);
    char* dev_ptr0;
    char* dev_ptr1;
    size_t size = 100;
    uint32_t deviceId = 0;
    cudaMalloc(&dev_ptr0, size);
    cudaMalloc(&dev_ptr1, size);
    std::vector<char> memory0(size, 10);
    std::vector<char> memory1(size, 1);
    cudaMemcpy(dev_ptr0, memory0.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptr1, memory1.data(), size, cudaMemcpyHostToDevice);
    RegisteredHostMemory regMem0(
        MemoryDescs{MemoryType::kVRAM, {MemoryDesc{dev_ptr0, size, deviceId}}}, nixlAgent0.get());
    RegisteredHostMemory regMem1(
        MemoryDescs{MemoryType::kVRAM, {MemoryDesc{dev_ptr1, size, deviceId}}}, nixlAgent1.get());

    // nixlAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = nixlAgent1->getConnectionInfo();
    nixlAgent0->connectRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = nixlAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked);
    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = nixlAgent0->submitTransferRequests(writeReq);
    status->wait();

    cudaMemcpy(memory0.data(), dev_ptr0, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(memory1.data(), dev_ptr1, size, cudaMemcpyDeviceToHost);

    TLLM_CHECK(memory0 == memory1);
    TLLM_CUDA_CHECK(cudaFree(dev_ptr0));
    TLLM_CUDA_CHECK(cudaFree(dev_ptr1));
    nixlAgent0->invalidateRemoteAgent(agent1);
}

TEST_F(TransferAgentTest, Connect)
{

    std::string const agent0{"agent0"}, agent1{"agent1"}, agent2{"agent2"};
    BaseAgentConfig config0{agent0, true}, config1{agent1, true}, config2{agent2, true};
    auto nixlAgent0 = makeTransferAgent(config0);
    auto nixlAgent1 = makeTransferAgent(config1);
    auto nixlAgent2 = makeTransferAgent(config2);

    TLLM_CHECK(nixlAgent0);
    TLLM_CHECK(nixlAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);
    MemoryDescs memDescs0{MemoryType::kDRAM, {MemoryDesc{memory0}}};
    MemoryDescs memDescs1{MemoryType::kDRAM, {MemoryDesc{memory1}}};

    nixlAgent0->registerMemory(memDescs0);
    nixlAgent1->registerMemory(memDescs1);
    nixlAgent2->registerMemory(memDescs0);

    // nixlAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = nixlAgent1->getConnectionInfo();
    nixlAgent0->connectRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = nixlAgent0->checkRemoteDescs(agent1, memDescs1);
    } while (!checked);
    TransferRequest writeReq{TransferOp::kWRITE, memDescs0, memDescs1, agent1};
    auto status = nixlAgent0->submitTransferRequests(writeReq);
    status->wait();

    TLLM_CHECK(memory0 == memory1);
    nixlAgent2->connectRemoteAgent(agent1, connectionInfo);
    checked = false;
    do
    {
        checked = nixlAgent2->checkRemoteDescs(agent1, memDescs1);
    } while (!checked);
    TransferRequest writeReq2{TransferOp::kWRITE, memDescs0, memDescs1, agent1};
    auto status2 = nixlAgent2->submitTransferRequests(writeReq2);
    status2->wait();
    TLLM_CHECK(memory0 == memory1);
    nixlAgent0->invalidateRemoteAgent(agent1);
    nixlAgent2->invalidateRemoteAgent(agent1);
    nixlAgent0->deregisterMemory(memDescs0);
    nixlAgent1->deregisterMemory(memDescs1);
    nixlAgent2->deregisterMemory(memDescs0);
}

TEST_F(TransferAgentTest, SyncMessage)
{
    constexpr std::size_t MAX_QUERY_TIMES = std::numeric_limits<size_t>::max();
    std::string const agent0{"agent0"}, agent1{"agent1"};
    BaseAgentConfig config0{agent0, true}, config1{agent1, true};
    auto nixlAgent0 = makeTransferAgent(config0);
    auto nixlAgent1 = makeTransferAgent(config1);

    TLLM_CHECK(nixlAgent0);
    TLLM_CHECK(nixlAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, nixlAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, nixlAgent0.get());

    RegisteredHostMemory regMem2(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, nixlAgent1.get());
    RegisteredHostMemory regMem3(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, nixlAgent1.get());

    // nixlAgent0->loadRemoteAgent(agent1);
    auto connectionInfo = nixlAgent1->getConnectionInfo();
    nixlAgent0->connectRemoteAgent(agent1, connectionInfo);
    bool checked = false;
    do
    {
        checked = nixlAgent0->checkRemoteDescs(agent1, regMem3.getDescs());
    } while (!checked);
    auto syncMessage = std::string("agent_sync_message");
    nixlAgent0->notifySyncMessage(agent1, syncMessage);
    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem3.getDescs(), agent1};
    auto status = nixlAgent0->submitTransferRequests(writeReq);

    auto notif = nixlAgent1->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif.size() == 0; i++)
    {
        notif = nixlAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(status->isCompleted());
    TLLM_CHECK(notif.size() == 1);
    TLLM_CHECK(notif[agent0].size() == 1);
    TLLM_CHECK(notif[agent0][0] == syncMessage);

    TLLM_CHECK(memory0 == memory1);

    std::string syncMessage2 = "two_agent_sync_message";
    nixlAgent0->notifySyncMessage(agent1, syncMessage2);
    auto notif2 = nixlAgent1->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif2.size() == 0; i++)
    {
        notif2 = nixlAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif2.size() == 1);
    TLLM_CHECK(notif2[agent0].size() == 1);
    TLLM_CHECK(notif2[agent0][0] == syncMessage2);

    // nixlAgent1->loadRemoteAgent(agent0);
    auto connectionInfo2 = nixlAgent0->getConnectionInfo();
    nixlAgent1->connectRemoteAgent(agent0, connectionInfo2);
    std::string syncMessage3 = "three_agent_sync_message";
    nixlAgent1->notifySyncMessage(agent0, syncMessage3);
    auto notif3 = nixlAgent0->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif3.size() == 0; i++)
    {
        notif3 = nixlAgent0->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif3.size() == 1);
    TLLM_CHECK(notif3[agent1].size() == 1);
    TLLM_CHECK(notif3[agent1][0] == syncMessage3);

    bool checked2 = false;
    do
    {
        checked2 = nixlAgent0->checkRemoteDescs(agent1, regMem1.getDescs());
    } while (!checked2);

    std::string syncMessage4 = "four_agent_sync_message";
    nixlAgent1->notifySyncMessage(agent0, syncMessage4);
    TransferRequest writeReq1{TransferOp::kWRITE, regMem2.getDescs(), regMem1.getDescs(), agent0};
    auto status1 = nixlAgent1->submitTransferRequests(writeReq1);
    auto notif4 = nixlAgent0->getNotifiedSyncMessages();
    for (std::size_t i = 0; i < MAX_QUERY_TIMES && notif4.size() == 0; i++)
    {
        notif4 = nixlAgent0->getNotifiedSyncMessages();
    }
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
    nixlAgent0->notifySyncMessage(agent1, serializedState);
    auto notif5 = nixlAgent1->getNotifiedSyncMessages();
    for (size_t i = 0; i < MAX_QUERY_TIMES && notif5.size() == 0; i++)
    {
        notif5 = nixlAgent1->getNotifiedSyncMessages();
    }
    TLLM_CHECK(notif5.size() == 1);
    TLLM_CHECK(notif5[agent0].size() == 1);
    TLLM_CHECK(notif5[agent0][0] == serializedState);
    std::stringstream ss2(notif5[agent0][0]);
    auto state2 = Serialization::deserializeCommState(ss2);
    TLLM_CHECK(state2 == state);

    nixlAgent0->invalidateRemoteAgent(agent1);
    nixlAgent1->invalidateRemoteAgent(agent0);
}
