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

    auto const enabled = validatePeerProtocol(PeerCancellationMode::kEnabled, {"legacy_agent"});
    EXPECT_FALSE(enabled.compatible);
    EXPECT_TRUE(enabled.hasLegacyPeer);
    EXPECT_FALSE(enabled.allPeersProtocolAware);
    EXPECT_NE(enabled.reason.find("does not advertise"), std::string::npos);
}

TEST(PeerProtocolTest, MatchingEnabledPeersAreCompatible)
{
    auto const compatibility = validatePeerProtocol(
        PeerCancellationMode::kEnabled, {makeAgentName(/*enabled=*/true), makeAgentName(true, kSessionToken + 1)});
    EXPECT_TRUE(compatibility.compatible);
    EXPECT_FALSE(compatibility.hasLegacyPeer);
    EXPECT_TRUE(compatibility.allPeersProtocolAware);
}

TEST(PeerProtocolTest, BaselineRollingUpgradeAcceptsAdvertisedAndLegacyRanks)
{
    auto const compatibility
        = validatePeerProtocol(PeerCancellationMode::kBaseline, {makeAgentName(/*enabled=*/false), "legacy_agent"});
    EXPECT_TRUE(compatibility.compatible);
    EXPECT_TRUE(compatibility.hasLegacyPeer);
    EXPECT_FALSE(compatibility.allPeersProtocolAware);
}

TEST(PeerProtocolTest, EffectiveModeMismatchIsRejected)
{
    auto const enabledLocal = validatePeerProtocol(PeerCancellationMode::kEnabled, {makeAgentName(/*enabled=*/false)});
    EXPECT_FALSE(enabledLocal.compatible);
    EXPECT_TRUE(enabledLocal.allPeersProtocolAware);
    EXPECT_NE(enabledLocal.reason.find("mode mismatch"), std::string::npos);

    auto const baselineLocal = validatePeerProtocol(PeerCancellationMode::kBaseline, {makeAgentName(/*enabled=*/true)});
    EXPECT_FALSE(baselineLocal.compatible);
    EXPECT_TRUE(baselineLocal.allPeersProtocolAware);
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
}

TEST(PeerProtocolTest, EnabledPeerMustAdvertiseCancellationCapability)
{
    auto descriptor = makePeerProtocolDescriptor(/*inflightCancellationEnabled=*/true, kSessionToken);
    descriptor.capabilities = 0;
    auto const agentName = appendPeerProtocolDescriptor("incapable_agent", descriptor);

    auto const compatibility = validatePeerProtocol(PeerCancellationMode::kEnabled, {agentName});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_NE(compatibility.reason.find("lacks the in-flight cancellation capability"), std::string::npos);
}

TEST(PeerProtocolTest, MalformedAdvertisedDescriptorIsRejectedEvenInBaselineMode)
{
    auto const compatibility
        = validatePeerProtocol(PeerCancellationMode::kBaseline, {"agent__trtllm_p_1_not-a-version"});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_NE(compatibility.reason.find("malformed"), std::string::npos);
}

TEST(PeerProtocolTest, ZeroSessionTokenIsRejected)
{
    auto const agentName = makeAgentName(/*enabled=*/false, /*sessionToken=*/0);
    auto const parsed = parsePeerProtocolDescriptor(agentName);
    EXPECT_EQ(parsed.status, PeerProtocolParseStatus::kMalformed);
    EXPECT_NE(parsed.reason.find("session token"), std::string::npos);
}

TEST(PeerProtocolTest, FutureDescriptorFieldsAreIgnored)
{
    auto const agentName = makeAgentName(/*enabled=*/true) + "_future_optional_field";
    auto const parsed = parsePeerProtocolDescriptor(agentName);
    ASSERT_EQ(parsed.status, PeerProtocolParseStatus::kValid);

    auto const compatibility = validatePeerProtocol(PeerCancellationMode::kEnabled, {agentName});
    EXPECT_TRUE(compatibility.compatible);
    EXPECT_TRUE(compatibility.allPeersProtocolAware);
}

TEST(PeerProtocolTest, MixedProtocolAwareAndLegacyPeersCannotUseReadyRejection)
{
    auto const compatibility
        = validatePeerProtocol(PeerCancellationMode::kBaseline, {makeAgentName(/*enabled=*/true), "legacy_agent"});
    EXPECT_FALSE(compatibility.compatible);
    EXPECT_TRUE(compatibility.hasLegacyPeer);
    EXPECT_FALSE(compatibility.allPeersProtocolAware);
}

} // namespace
} // namespace tensorrt_llm::executor::kv_cache
