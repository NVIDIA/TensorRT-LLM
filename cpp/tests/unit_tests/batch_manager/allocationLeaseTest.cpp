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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tb = tensorrt_llm::batch_manager;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

namespace
{

TEST(AllocationLeaseAccountingTest, UnknownDefaultFailsClosed)
{
    tbk::AllocationLeaseAccounting const accounting;

    EXPECT_FALSE(accounting.leaseStateKnown);
    EXPECT_FALSE(accounting.safeToReleasePools());
}

class AllocationLeaseSettlementGuard
{
public:
    AllocationLeaseSettlementGuard(tbk::KVCacheManager& manager, tbk::AllocationLeaseSnapshot const& lease)
        : mManager(manager)
        , mLease(lease)
    {
    }

    ~AllocationLeaseSettlementGuard()
    {
        (void) mManager.settleAllocationLease(
            mLease.getLeaseId(), mLease.getIdentity(), tb::PhysicalDisposition::kNOT_EXPOSED);
    }

private:
    tbk::KVCacheManager& mManager;
    tbk::AllocationLeaseSnapshot const& mLease;
};

class AllocationLeaseTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP();
        }
    }

    static std::unique_ptr<tbk::KVCacheManager> makeManager()
    {
        constexpr tbk::SizeType32 kNumLayers = 2;
        constexpr tbk::SizeType32 kNumKvHeads = 2;
        constexpr tbk::SizeType32 kSizePerHead = 8;
        constexpr tbk::SizeType32 kTokensPerBlock = 4;
        constexpr tbk::SizeType32 kNumBlocks = 16;
        constexpr tbk::SizeType32 kMaxNumSequences = 4;
        constexpr tbk::SizeType32 kMaxBeamWidth = 2;
        constexpr tbk::SizeType32 kMaxSequenceLength = 32;
        constexpr tbk::SizeType32 kSinkTokenLength = 0;
        auto stream = std::make_shared<tr::CudaStream>();
        tbk::BlocksPerWindow const blocksPerWindow{{kMaxSequenceLength, {kNumBlocks, 0}}};
        auto manager = std::make_unique<tbk::KVCacheManager>(kNumLayers, kNumKvHeads, kSizePerHead, kTokensPerBlock,
            blocksPerWindow, kMaxNumSequences, kMaxBeamWidth, std::vector<tbk::SizeType32>{kMaxSequenceLength},
            tensorrt_llm::DataType::kHALF, kSinkTokenLength, stream, kMaxSequenceLength, kMaxSequenceLength,
            /*enableBlockReuse=*/false);
        manager->allocatePools(false);
        return manager;
    }

    static std::shared_ptr<tb::LlmRequest> addRequest(
        tbk::KVCacheManager& manager, tb::LlmRequest::RequestIdType requestId, tbk::SizeType32 beamWidth)
    {
        constexpr tbk::SizeType32 kInputLength = 5;
        auto request = makeRequest(requestId, beamWidth, kInputLength);
        manager.addSequenceBatch({{{requestId, kInputLength, beamWidth}}}, {std::ref(*request)});
        return request;
    }

    static std::shared_ptr<tb::LlmRequest> makeRequest(
        tb::LlmRequest::RequestIdType requestId, tbk::SizeType32 beamWidth, tbk::SizeType32 inputLength = 5)
    {
        auto inputTokens = std::make_shared<tb::LlmRequest::VecTokens>(inputLength);
        std::iota(inputTokens->begin(), inputTokens->end(), 0);
        tr::SamplingConfig samplingConfig{beamWidth};
        return std::make_shared<tb::LlmRequest>(
            requestId, /*maxNewTokens=*/0, inputTokens, samplingConfig, /*isStreaming=*/false);
    }
};

TEST_F(AllocationLeaseTest, LogicalRemovalRetainsExactBlocksUntilReusableSettlement)
{
    constexpr tb::LlmRequest::RequestIdType kRequestId = 41;
    constexpr tbk::SizeType32 kBeamWidth = 2;
    auto manager = makeManager();
    auto request = addRequest(*manager, kRequestId, kBeamWidth);
    auto const initialFreeBlocks = manager->getNumFreeBlocks();

    auto const identity = manager->getAllocationIdentity(kRequestId);
    ASSERT_TRUE(identity.has_value());
    auto const lease = manager->snapshotAndLease(*identity);
    ASSERT_TRUE(lease.has_value());
    AllocationLeaseSettlementGuard leaseGuard{*manager, *lease};
    EXPECT_EQ(lease->getBlocks().size(), 4);

    auto accounting = manager->getAllocationLeaseAccounting();
    EXPECT_EQ(accounting.outstandingLeaseCount, 1);
    EXPECT_EQ(accounting.outstandingBlockPinCount, 3);
    EXPECT_TRUE(accounting.leaseStateKnown);
    EXPECT_FALSE(accounting.safeToReleasePools());
    EXPECT_THROW(manager->allocatePools(false), std::runtime_error);

    EXPECT_THROW(manager->releasePools(), std::runtime_error);
    accounting = manager->getAllocationLeaseAccounting();
    EXPECT_TRUE(accounting.shutdownStarted);
    EXPECT_TRUE(accounting.leaseStateKnown);
    EXPECT_FALSE(accounting.safeToReleasePools());
    EXPECT_FALSE(manager->snapshotAndLease(*identity).has_value());
    EXPECT_THROW(manager->allocatePools(false), std::runtime_error);

    (void) manager->removeSequence(kRequestId, request);
    EXPECT_FALSE(manager->getAllocationIdentity(kRequestId).has_value());
    EXPECT_EQ(manager->getNumFreeBlocks(), initialFreeBlocks);

    EXPECT_EQ(
        manager->settleAllocationLease(lease->getLeaseId(), lease->getIdentity(), tb::PhysicalDisposition::kACTIVE),
        tbk::AllocationLeaseSettlement::kNOT_QUIESCED);
    EXPECT_EQ(manager->getNumFreeBlocks(), initialFreeBlocks);

    auto staleIdentity = lease->getIdentity();
    ++staleIdentity.allocationGeneration;
    EXPECT_EQ(
        manager->settleAllocationLease(lease->getLeaseId(), staleIdentity, tb::PhysicalDisposition::kQUIESCED_SUCCESS),
        tbk::AllocationLeaseSettlement::kSTALE_GENERATION);
    EXPECT_EQ(manager->getNumFreeBlocks(), initialFreeBlocks);

    EXPECT_EQ(manager->settleAllocationLease(
                  lease->getLeaseId(), lease->getIdentity(), tb::PhysicalDisposition::kQUIESCED_SUCCESS),
        tbk::AllocationLeaseSettlement::kRELEASED);
    EXPECT_GT(manager->getNumFreeBlocks(), initialFreeBlocks);
    EXPECT_EQ(manager->settleAllocationLease(
                  lease->getLeaseId(), lease->getIdentity(), tb::PhysicalDisposition::kQUIESCED_SUCCESS),
        tbk::AllocationLeaseSettlement::kALREADY_RELEASED);

    accounting = manager->getAllocationLeaseAccounting();
    EXPECT_EQ(accounting.outstandingLeaseCount, 0);
    EXPECT_EQ(accounting.outstandingBlockPinCount, 0);
    EXPECT_TRUE(accounting.leaseStateKnown);
    EXPECT_TRUE(accounting.shutdownStarted);
    EXPECT_TRUE(accounting.safeToReleasePools());
    manager->releasePools();
}

TEST_F(AllocationLeaseTest, SliceDeduplicatesSharedBeamPinsAndOldIdentityCannotLeaseReuse)
{
    constexpr tb::LlmRequest::RequestIdType kRequestId = 73;
    constexpr std::uint64_t kUnknownLeaseId = 999999;
    constexpr tbk::SizeType32 kBeamWidth = 2;
    auto manager = makeManager();
    auto firstRequest = addRequest(*manager, kRequestId, kBeamWidth);
    auto const oldIdentity = manager->getAllocationIdentity(kRequestId);
    ASSERT_TRUE(oldIdentity.has_value());

    tbk::AllocationLeaseSliceSpec sliceSpec;
    sliceSpec.firstBlockIndex = 0;
    sliceSpec.blockCount = 1;
    auto const lease = manager->snapshotAndLease(*oldIdentity, sliceSpec);
    ASSERT_TRUE(lease.has_value());
    AllocationLeaseSettlementGuard leaseGuard{*manager, *lease};
    EXPECT_EQ(lease->getBlocks().size(), 2);
    EXPECT_EQ(lease->getBlocks().at(0).getBlockId(), lease->getBlocks().at(1).getBlockId());
    EXPECT_EQ(manager->getAllocationLeaseAccounting().outstandingBlockPinCount, 1);

    (void) manager->removeSequence(kRequestId, firstRequest);
    auto secondRequest = addRequest(*manager, kRequestId, kBeamWidth);
    auto const newIdentity = manager->getAllocationIdentity(kRequestId);
    ASSERT_TRUE(newIdentity.has_value());
    EXPECT_EQ(newIdentity->allocatorDomainId, oldIdentity->allocatorDomainId);
    EXPECT_GT(newIdentity->allocationGeneration, oldIdentity->allocationGeneration);
    EXPECT_FALSE(manager->snapshotAndLease(*oldIdentity, sliceSpec).has_value());

    EXPECT_EQ(
        manager->settleAllocationLease(lease->getLeaseId(), lease->getIdentity(), tb::PhysicalDisposition::kIN_DOUBT),
        tbk::AllocationLeaseSettlement::kNOT_QUIESCED);
    EXPECT_EQ(manager->getAllocationLeaseAccounting().outstandingBlockPinCount, 1);
    EXPECT_EQ(manager->settleAllocationLease(
                  lease->getLeaseId(), lease->getIdentity(), tb::PhysicalDisposition::kQUIESCED_FAILURE),
        tbk::AllocationLeaseSettlement::kRELEASED);

    (void) manager->removeSequence(kRequestId, secondRequest);
    EXPECT_EQ(manager->settleAllocationLease(kUnknownLeaseId, *newIdentity, tb::PhysicalDisposition::kQUIESCED_SUCCESS),
        tbk::AllocationLeaseSettlement::kNOT_FOUND);
    EXPECT_NO_THROW(manager->allocatePools(false));
    manager->releasePools();
}

TEST_F(AllocationLeaseTest, BatchSetupFailureRollsBackSequenceAndGenerationTogether)
{
    constexpr tb::LlmRequest::RequestIdType kExistingRequestId = 101;
    constexpr tb::LlmRequest::RequestIdType kRolledBackRequestId = 102;
    constexpr tbk::SizeType32 kBeamWidth = 1;
    constexpr tbk::SizeType32 kInputLength = 5;
    auto manager = makeManager();
    auto existingRequest = addRequest(*manager, kExistingRequestId, kBeamWidth);
    auto rolledBackRequest = makeRequest(kRolledBackRequestId, kBeamWidth, kInputLength);
    std::vector<std::tuple<tb::LlmRequest::RequestIdType, tbk::SizeType32, tbk::SizeType32>> const requestInfos{
        {kRolledBackRequestId, kInputLength, kBeamWidth}, {kExistingRequestId, kInputLength, kBeamWidth}};

    EXPECT_THROW(manager->addSequenceBatch(requestInfos, {std::ref(*rolledBackRequest), std::ref(*existingRequest)}),
        std::runtime_error);
    EXPECT_FALSE(manager->getAllocationIdentity(kRolledBackRequestId).has_value());
    EXPECT_TRUE(manager->getAllocationIdentity(kExistingRequestId).has_value());

    manager->addSequenceBatch({{{kRolledBackRequestId, kInputLength, kBeamWidth}}}, {std::ref(*rolledBackRequest)});
    EXPECT_TRUE(manager->getAllocationIdentity(kRolledBackRequestId).has_value());

    (void) manager->removeSequence(kRolledBackRequestId, rolledBackRequest);
    (void) manager->removeSequence(kExistingRequestId, existingRequest);
    manager->releasePools();
}

} // namespace
