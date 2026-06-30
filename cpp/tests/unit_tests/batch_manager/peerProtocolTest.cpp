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
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "tensorrt_llm/executor/cache_transmission/agent_utils/peerProtocol.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{
namespace
{

constexpr std::uint64_t kSessionToken{0x123456789abcdef0ULL};

std::string makeAgentName(bool const enabled, std::uint64_t const sessionToken = kSessionToken)
{
    return appendPeerProtocolDescriptor("opaque_nixl_agent_identity",
        makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/enabled, sessionToken));
}

TEST(PeerProtocolTest, DescriptorRoundTripsWithoutChangingIdentityPrefix)
{
    auto const descriptor = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    auto const agentName = appendPeerProtocolDescriptor("opaque_nixl_agent_identity", descriptor);
    EXPECT_EQ(agentName.find("opaque_nixl_agent_identity"), 0);

    auto const parsed = parsePeerProtocolDescriptor(agentName);
    ASSERT_EQ(parsed.status, PeerProtocolParseStatus::kValid);
    ASSERT_TRUE(parsed.descriptor.has_value());
    EXPECT_EQ(parsed.descriptor.value(), descriptor);
}

TEST(PeerProtocolTest, MissingDescriptorIsAllowedOnlyForBaselineMode)
{
    auto const baseline = validatePeerProtocol(PeerCancellationMode::kBaseline, {"legacy_agent"});
    EXPECT_TRUE(baseline.compatible);
    EXPECT_TRUE(baseline.hasLegacyPeer);
    EXPECT_FALSE(baseline.allPeersProtocolAware);
    EXPECT_FALSE(baseline.allPeersCanRejectBeforeDma);

    auto const enabled = validatePeerProtocol(PeerCancellationMode::kEnabled, {"legacy_agent"});
    EXPECT_FALSE(enabled.compatible);
    EXPECT_TRUE(enabled.hasLegacyPeer);
    EXPECT_FALSE(enabled.allPeersProtocolAware);
    EXPECT_FALSE(enabled.allPeersCanRejectBeforeDma);
    EXPECT_NE(enabled.reason.find("does not advertise"), std::string::npos);
}

TEST(PeerProtocolTest, MatchingEnabledPeersAreCompatible)
{
    auto const compatibility = validatePeerProtocol(
        PeerCancellationMode::kEnabled, {makeAgentName(/*enabled=*/true), makeAgentName(true, kSessionToken + 1)});
    EXPECT_TRUE(compatibility.compatible);
    EXPECT_FALSE(compatibility.hasLegacyPeer);
    EXPECT_TRUE(compatibility.allPeersProtocolAware);
    EXPECT_TRUE(compatibility.allPeersCanRejectBeforeDma);
    EXPECT_EQ(compatibility.selectedVersion, 1);
}

TEST(PeerProtocolTest, BaselineRolloutAssumesDescriptorlessPeersHaveCancellationOff)
{
    auto const compatibility
        = validatePeerProtocol(PeerCancellationMode::kBaseline, {makeAgentName(/*enabled=*/false), "legacy_agent"});
    EXPECT_TRUE(compatibility.compatible);
    EXPECT_TRUE(compatibility.hasLegacyPeer);
    EXPECT_FALSE(compatibility.allPeersProtocolAware);
    EXPECT_FALSE(compatibility.allPeersCanRejectBeforeDma);
    EXPECT_FALSE(compatibility.selectedVersion.has_value());
}

TEST(PeerProtocolTest, EffectiveModeMismatchIsRejected)
{
    auto const enabledLocal = validatePeerProtocol(PeerCancellationMode::kEnabled, {makeAgentName(/*enabled=*/false)});
    EXPECT_FALSE(enabledLocal.compatible);
    EXPECT_TRUE(enabledLocal.allPeersProtocolAware);
    EXPECT_TRUE(enabledLocal.allPeersCanRejectBeforeDma);
    EXPECT_NE(enabledLocal.reason.find("mode mismatch"), std::string::npos);

    auto const baselineLocal = validatePeerProtocol(PeerCancellationMode::kBaseline, {makeAgentName(/*enabled=*/true)});
    EXPECT_FALSE(baselineLocal.compatible);
    EXPECT_TRUE(baselineLocal.allPeersProtocolAware);
    EXPECT_TRUE(baselineLocal.allPeersCanRejectBeforeDma);
    EXPECT_NE(baselineLocal.reason.find("mode mismatch"), std::string::npos);
}

TEST(PeerProtocolTest, NonOverlappingVersionIsRejected)
{
    auto descriptor = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    descriptor.minVersion = 2;
    descriptor.maxVersion = 3;
    auto const agentName = appendPeerProtocolDescriptor("future_agent", descriptor);

    auto const compatibility = validatePeerProtocol(PeerCancellationMode::kEnabled, {agentName});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_NE(compatibility.reason.find("supports protocol versions 2-3"), std::string::npos);
}

TEST(PeerProtocolTest, BaselinePeersIgnoreCancellationProtocolVersion)
{
    auto descriptor = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/false, kSessionToken);
    descriptor.minVersion = 2;
    descriptor.maxVersion = 3;
    auto const agentName = appendPeerProtocolDescriptor("future_baseline_agent", descriptor);

    auto const compatibility = validatePeerProtocol(PeerCancellationMode::kBaseline, {agentName});
    EXPECT_TRUE(compatibility.compatible);
    EXPECT_FALSE(compatibility.selectedVersion.has_value());
}

TEST(PeerProtocolTest, EnabledPeerMustAdvertiseCancellationCapability)
{
    auto descriptor = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    descriptor.capabilities = static_cast<std::uint64_t>(PeerProtocolCapability::kPreDmaProtocolRejection);
    auto const agentName = appendPeerProtocolDescriptor("incapable_agent", descriptor);

    auto const compatibility = validatePeerProtocol(PeerCancellationMode::kEnabled, {agentName});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_NE(compatibility.reason.find("lacks the in-flight cancellation capability"), std::string::npos);
}

TEST(PeerProtocolTest, MalformedAdvertisedDescriptorIsRejectedEvenInBaselineMode)
{
    auto const compatibility = validatePeerProtocol(
        PeerCancellationMode::kBaseline, {"agent__trtllm_protocol_v1_1_not-a-version__trtllm_protocol_end"});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_NE(compatibility.reason.find("malformed"), std::string::npos);
}

TEST(PeerProtocolTest, EmbeddedHostnameMarkerDoesNotForgeLegacyIdentity)
{
    auto const parsed
        = parsePeerProtocolDescriptor("legacy__trtllm_protocol_v1_1_1_3_1_1__trtllm_protocol_end_123_456_0");
    EXPECT_EQ(parsed.status, PeerProtocolParseStatus::kMissing);
}

TEST(PeerProtocolTest, ZeroSessionTokenIsRejected)
{
    auto const agentName = makeAgentName(/*enabled=*/false, /*sessionToken=*/0);
    auto const parsed = parsePeerProtocolDescriptor(agentName);
    EXPECT_EQ(parsed.status, PeerProtocolParseStatus::kMalformed);
    EXPECT_NE(parsed.reason.find("session token"), std::string::npos);
}

TEST(PeerProtocolTest, V1EnvelopeRejectsUnversionedFutureFields)
{
    auto const suffixPos = makeAgentName(/*enabled=*/true).find("__trtllm_protocol_end");
    ASSERT_NE(suffixPos, std::string::npos);
    auto agentName = makeAgentName(/*enabled=*/true);
    agentName.insert(suffixPos, "_future_optional_field");
    auto const parsed = parsePeerProtocolDescriptor(agentName);
    EXPECT_EQ(parsed.status, PeerProtocolParseStatus::kMalformed);
}

TEST(PeerProtocolTest, MixedProtocolAwareAndLegacyPeersCannotUseReadyRejection)
{
    auto const compatibility
        = validatePeerProtocol(PeerCancellationMode::kBaseline, {makeAgentName(/*enabled=*/true), "legacy_agent"});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_TRUE(compatibility.hasLegacyPeer);
    EXPECT_FALSE(compatibility.allPeersProtocolAware);
    EXPECT_FALSE(compatibility.allPeersCanRejectBeforeDma);
}

TEST(PeerProtocolTest, MismatchRequiresEveryPeerToAdvertisePreDmaRejection)
{
    auto descriptor = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/false, kSessionToken);
    descriptor.capabilities &= ~static_cast<std::uint64_t>(PeerProtocolCapability::kPreDmaProtocolRejection);
    auto const compatibility = validatePeerProtocol(
        PeerCancellationMode::kEnabled, {appendPeerProtocolDescriptor("old_protocol_peer", descriptor)});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_TRUE(compatibility.allPeersProtocolAware);
    EXPECT_FALSE(compatibility.allPeersCanRejectBeforeDma);
}

TEST(PeerProtocolTest, EnabledPeerMustSupportPreDmaProtocolRejection)
{
    auto descriptor = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    descriptor.capabilities = static_cast<std::uint64_t>(PeerProtocolCapability::kInflightCancellation);
    auto const compatibility = validatePeerProtocol(
        PeerCancellationMode::kEnabled, {appendPeerProtocolDescriptor("old_protocol_peer", descriptor)});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_FALSE(compatibility.allPeersCanRejectBeforeDma);
    EXPECT_NE(compatibility.reason.find("pre-DMA protocol-rejection"), std::string::npos);
}

TEST(PeerProtocolTest, EnabledLocalDescriptorMustAdvertiseRequiredCapabilities)
{
    auto local = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    auto const peer = makeAgentName(/*enabled=*/true, kSessionToken + 1);

    local.capabilities = static_cast<std::uint64_t>(PeerProtocolCapability::kPreDmaProtocolRejection);
    auto const missingCancellation = validatePeerProtocol(local, {peer});
    EXPECT_FALSE(missingCancellation.compatible);
    EXPECT_NE(missingCancellation.reason.find("local protocol descriptor lacks the in-flight"), std::string::npos);

    local.capabilities = static_cast<std::uint64_t>(PeerProtocolCapability::kInflightCancellation);
    auto const missingRejection = validatePeerProtocol(local, {peer});
    EXPECT_FALSE(missingRejection.compatible);
    EXPECT_NE(missingRejection.reason.find("local protocol descriptor lacks the pre-DMA"), std::string::npos);
}

TEST(PeerProtocolTest, SelectsOneVersionCommonToEveryPeerRank)
{
    auto local = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    local.minVersion = 1;
    local.maxVersion = 2;

    auto peer0 = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken + 1);
    peer0.minVersion = 1;
    peer0.maxVersion = 2;
    auto peer1 = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken + 2);
    peer1.minVersion = 2;
    peer1.maxVersion = 3;
    auto const compatible = validatePeerProtocol(
        local, {appendPeerProtocolDescriptor("peer0", peer0), appendPeerProtocolDescriptor("peer1", peer1)});
    EXPECT_TRUE(compatible.compatible);
    EXPECT_EQ(compatible.selectedVersion, 2);

    peer0.minVersion = 1;
    peer0.maxVersion = 1;
    auto const noCommonVersion = validatePeerProtocol(
        local, {appendPeerProtocolDescriptor("peer0", peer0), appendPeerProtocolDescriptor("peer1", peer1)});
    EXPECT_FALSE(noCommonVersion.compatible);
    EXPECT_FALSE(noCommonVersion.selectedVersion.has_value());
    EXPECT_NE(noCommonVersion.reason.find("no version is common"), std::string::npos);
}

TEST(PeerProtocolTest, LocalRanksMustAdvertiseOneExactProtocolProfile)
{
    auto rank0 = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    auto rank1 = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken + 1);
    auto const matching = validateLocalPeerProtocol(PeerCancellationMode::kEnabled,
        {appendPeerProtocolDescriptor("rank0", rank0), appendPeerProtocolDescriptor("rank1", rank1)});
    EXPECT_TRUE(matching.compatible);

    rank1.maxVersion = 2;
    auto const mixedVersions = validateLocalPeerProtocol(PeerCancellationMode::kEnabled,
        {appendPeerProtocolDescriptor("rank0", rank0), appendPeerProtocolDescriptor("rank1", rank1)});
    EXPECT_FALSE(mixedVersions.compatible);
    EXPECT_NE(mixedVersions.reason.find("different protocol version"), std::string::npos);

    auto const legacyRank
        = validateLocalPeerProtocol(PeerCancellationMode::kBaseline, {makeAgentName(/*enabled=*/false), "legacy_rank"});
    EXPECT_FALSE(legacyRank.compatible);
    EXPECT_NE(legacyRank.reason.find("does not advertise"), std::string::npos);
}

} // namespace
} // namespace tensorrt_llm::executor::kv_cache
