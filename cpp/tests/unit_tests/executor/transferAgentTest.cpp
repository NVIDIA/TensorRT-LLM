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
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/interfaces.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace tensorrt_llm::executor::kv_cache;

class LocalAgentRegistrar final : public tensorrt_llm::executor::kv_cache::AgentRegistrar
{
public:
    [[nodiscard]] AgentDesc const* getAgentDesc(char const* agentName) const
    {
        auto it = mAgentDescs.find(agentName);
        TLLM_CHECK(it != mAgentDescs.end());
        return &it->second;
    }

    void addAgentDesc(char const* agentName, AgentDesc desc)
    {
        mAgentDescs.insert(std::make_pair(agentName, std::move(desc)));
    }

    void removeAgentDesc(char const* agentName)
    {
        mAgentDescs.erase(agentName);
    }

private:
    std::unordered_map<std::string, AgentDesc> mAgentDescs;
};

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

    [[nodiscard]] std::unique_ptr<BaseTransferAgent> makeTransferAgent(
        BaseAgentConfig const& config, AgentRegistrar* registrar)
    {
        return tensorrt_llm::executor::kv_cache::createNixlTransferAgent(&config, registrar);
    }
};

TEST_F(TransferAgentTest, Basic)
{
    LocalAgentRegistrar registrar;

    char const *agent0{"agent0"}, *agent1{"agent1"};
    BaseAgentConfig config0{agent0, true}, config1{agent1, true};
    auto nixlAgent0 = makeTransferAgent(config0, &registrar);
    auto nixlAgent1 = makeTransferAgent(config1, &registrar);

    TLLM_CHECK(nixlAgent0);
    TLLM_CHECK(nixlAgent1);

    std::vector<char> memory0(100, 10);
    std::vector<char> memory1(100, 1);

    RegisteredHostMemory regMem0(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory0}}}, nixlAgent0.get());
    RegisteredHostMemory regMem1(MemoryDescs{MemoryType::kDRAM, {MemoryDesc{memory1}}}, nixlAgent1.get());

    nixlAgent0->loadRemoteAgent(agent1);

    TransferRequest writeReq{TransferOp::kWRITE, regMem0.getDescs(), regMem1.getDescs(), agent1};
    auto status = nixlAgent0->submitTransferRequests(writeReq);
    status->wait();

    TLLM_CHECK(memory0 == memory1);

    nixlAgent0->invalidateRemoteAgent(agent1);
}
