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

#pragma once

#include <charconv>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

//! Effective cancellation behavior for a cache-transceiver process.
enum class PeerCancellationMode : std::uint8_t
{
    kBaseline = 0,
    kEnabled = 1,
};

//! Protocol capabilities advertised through the NIXL agent identity.
enum class PeerProtocolCapability : std::uint64_t
{
    kInflightCancellation = std::uint64_t{1} << 0,
};

struct PeerProtocolDescriptor
{
    std::uint16_t minVersion{1};
    std::uint16_t maxVersion{1};
    std::uint64_t capabilities{0};
    PeerCancellationMode cancellationMode{PeerCancellationMode::kBaseline};
    std::uint64_t sessionToken{0};

    [[nodiscard]] bool operator==(PeerProtocolDescriptor const& other) const noexcept
    {
        return minVersion == other.minVersion && maxVersion == other.maxVersion && capabilities == other.capabilities
            && cancellationMode == other.cancellationMode && sessionToken == other.sessionToken;
    }
};

enum class PeerProtocolParseStatus : std::uint8_t
{
    kMissing,
    kValid,
    kMalformed,
};

struct PeerProtocolParseResult
{
    PeerProtocolParseStatus status{PeerProtocolParseStatus::kMissing};
    std::optional<PeerProtocolDescriptor> descriptor;
    std::string reason;
};

struct PeerProtocolCompatibility
{
    bool compatible{false};
    bool hasLegacyPeer{false};
    bool allPeersProtocolAware{false};
    std::string reason;
};

namespace detail
{

constexpr std::string_view kPeerProtocolMarker{"__trtllm_p_"};
constexpr std::uint16_t kPeerProtocolMinVersion{1};
constexpr std::uint16_t kPeerProtocolMaxVersion{1};
constexpr std::uint64_t kInflightCancellationCapability
    = static_cast<std::uint64_t>(PeerProtocolCapability::kInflightCancellation);

inline bool parseUnsigned(std::string_view text, int const base, std::uint64_t& value)
{
    if (text.empty())
    {
        return false;
    }
    auto const* begin = text.data();
    auto const* end = begin + text.size();
    auto const result = std::from_chars(begin, end, value, base);
    return result.ec == std::errc{} && result.ptr == end;
}

inline std::vector<std::string_view> splitProtocolFields(std::string_view payload)
{
    std::vector<std::string_view> fields;
    while (true)
    {
        auto const separator = payload.find('_');
        fields.push_back(payload.substr(0, separator));
        if (separator == std::string_view::npos)
        {
            break;
        }
        payload.remove_prefix(separator + 1);
    }
    return fields;
}

inline char const* cancellationModeName(PeerCancellationMode const mode)
{
    switch (mode)
    {
    case PeerCancellationMode::kBaseline: return "baseline";
    case PeerCancellationMode::kEnabled: return "enabled";
    }
    return "unknown";
}

} // namespace detail

//! Construct the descriptor supported by this binary.
inline PeerProtocolDescriptor makePeerProtocolDescriptor(
    bool const inflightCancellationEnabled, std::uint64_t const sessionToken)
{
    return PeerProtocolDescriptor{detail::kPeerProtocolMinVersion, detail::kPeerProtocolMaxVersion,
        detail::kInflightCancellationCapability,
        inflightCancellationEnabled ? PeerCancellationMode::kEnabled : PeerCancellationMode::kBaseline, sessionToken};
}

//! Append an opaque, backward-compatible protocol suffix to the actual NIXL agent identity.
inline std::string appendPeerProtocolDescriptor(std::string agentName, PeerProtocolDescriptor const& descriptor)
{
    std::ostringstream suffix;
    suffix << detail::kPeerProtocolMarker << descriptor.minVersion << '_' << descriptor.maxVersion << '_' << std::hex
           << descriptor.capabilities << '_' << std::dec << static_cast<unsigned>(descriptor.cancellationMode) << '_'
           << std::hex << std::setw(16) << std::setfill('0') << descriptor.sessionToken;
    agentName += suffix.str();
    return agentName;
}

//! Parse a descriptor suffix without changing the transport identity string.
inline PeerProtocolParseResult parsePeerProtocolDescriptor(std::string_view const agentName)
{
    auto const markerPos = agentName.rfind(detail::kPeerProtocolMarker);
    if (markerPos == std::string_view::npos)
    {
        return {PeerProtocolParseStatus::kMissing, std::nullopt, "peer does not advertise a protocol descriptor"};
    }

    auto const fields = detail::splitProtocolFields(agentName.substr(markerPos + detail::kPeerProtocolMarker.size()));
    constexpr size_t kRequiredFieldCount = 5;
    if (fields.size() < kRequiredFieldCount)
    {
        return {PeerProtocolParseStatus::kMalformed, std::nullopt,
            "protocol descriptor must contain version range, capabilities, mode, and session token"};
    }

    std::uint64_t minVersion = 0;
    std::uint64_t maxVersion = 0;
    std::uint64_t capabilities = 0;
    std::uint64_t mode = 0;
    std::uint64_t sessionToken = 0;
    bool const parsed = detail::parseUnsigned(fields[0], /*base=*/10, minVersion)
        && detail::parseUnsigned(fields[1], /*base=*/10, maxVersion)
        && detail::parseUnsigned(fields[2], /*base=*/16, capabilities)
        && detail::parseUnsigned(fields[3], /*base=*/10, mode)
        && detail::parseUnsigned(fields[4], /*base=*/16, sessionToken);
    if (!parsed || minVersion > std::numeric_limits<std::uint16_t>::max()
        || maxVersion > std::numeric_limits<std::uint16_t>::max())
    {
        return {PeerProtocolParseStatus::kMalformed, std::nullopt,
            "protocol descriptor contains a non-numeric or out-of-range field"};
    }
    if (minVersion == 0 || minVersion > maxVersion)
    {
        return {PeerProtocolParseStatus::kMalformed, std::nullopt, "protocol version range is invalid"};
    }
    if (mode > static_cast<std::uint64_t>(PeerCancellationMode::kEnabled))
    {
        return {PeerProtocolParseStatus::kMalformed, std::nullopt, "effective cancellation mode is invalid"};
    }
    if (sessionToken == 0)
    {
        return {PeerProtocolParseStatus::kMalformed, std::nullopt, "protocol session token must be nonzero"};
    }

    return {PeerProtocolParseStatus::kValid,
        PeerProtocolDescriptor{static_cast<std::uint16_t>(minVersion), static_cast<std::uint16_t>(maxVersion),
            capabilities, static_cast<PeerCancellationMode>(mode), sessionToken},
        {}};
}

//! Validate every peer rank before a ready signal or destination-buffer advertisement.
inline PeerProtocolCompatibility validatePeerProtocol(
    PeerCancellationMode const localMode, std::vector<std::string> const& peerAgentNames)
{
    if (peerAgentNames.empty())
    {
        return {false, false, false, "peer agent-state list is empty"};
    }

    bool hasLegacyPeer = false;
    std::vector<std::optional<PeerProtocolDescriptor>> peerDescriptors;
    peerDescriptors.reserve(peerAgentNames.size());
    for (size_t rank = 0; rank < peerAgentNames.size(); ++rank)
    {
        auto const parsed = parsePeerProtocolDescriptor(peerAgentNames[rank]);
        if (parsed.status == PeerProtocolParseStatus::kMissing)
        {
            hasLegacyPeer = true;
            peerDescriptors.emplace_back(std::nullopt);
            continue;
        }
        if (parsed.status == PeerProtocolParseStatus::kMalformed)
        {
            return {false, hasLegacyPeer, false,
                "peer rank " + std::to_string(rank) + " has a malformed protocol descriptor: " + parsed.reason};
        }
        peerDescriptors.push_back(parsed.descriptor);
    }

    bool const allPeersProtocolAware = !hasLegacyPeer;
    if (localMode == PeerCancellationMode::kEnabled && hasLegacyPeer)
    {
        return {false, true, false,
            "local cancellation mode is enabled but one or more peer ranks do not advertise compatible "
            "cancellation semantics"};
    }

    for (size_t rank = 0; rank < peerDescriptors.size(); ++rank)
    {
        if (!peerDescriptors[rank].has_value())
        {
            continue;
        }
        auto const& peer = peerDescriptors[rank].value();
        if (peer.cancellationMode != localMode)
        {
            return {false, hasLegacyPeer, allPeersProtocolAware,
                "effective cancellation mode mismatch at peer rank " + std::to_string(rank)
                    + ": local=" + detail::cancellationModeName(localMode)
                    + ", peer=" + detail::cancellationModeName(peer.cancellationMode)};
        }
        // Cancellation protocol versions do not affect the baseline wire path. Keeping baseline-to-baseline
        // transfers compatible lets deployments roll this protocol independently while cancellation remains off.
        if (localMode == PeerCancellationMode::kBaseline)
        {
            continue;
        }

        bool const versionOverlaps
            = peer.minVersion <= detail::kPeerProtocolMaxVersion && peer.maxVersion >= detail::kPeerProtocolMinVersion;
        if (!versionOverlaps)
        {
            return {false, hasLegacyPeer, allPeersProtocolAware,
                "peer rank " + std::to_string(rank) + " supports protocol versions " + std::to_string(peer.minVersion)
                    + "-" + std::to_string(peer.maxVersion) + ", but this binary supports "
                    + std::to_string(detail::kPeerProtocolMinVersion) + "-"
                    + std::to_string(detail::kPeerProtocolMaxVersion)};
        }
        if ((peer.capabilities & detail::kInflightCancellationCapability) == 0)
        {
            return {false, hasLegacyPeer, allPeersProtocolAware,
                "peer rank " + std::to_string(rank) + " lacks the in-flight cancellation capability"};
        }
    }

    return {true, hasLegacyPeer, allPeersProtocolAware,
        hasLegacyPeer ? "compatible legacy baseline peer" : "compatible protocol and effective mode"};
}

} // namespace tensorrt_llm::executor::kv_cache
