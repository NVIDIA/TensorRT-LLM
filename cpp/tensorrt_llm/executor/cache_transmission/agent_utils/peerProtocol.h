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

#include <algorithm>
#include <array>
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
    //! The peer can settle a protocol mismatch with ready=false before DMA.
    kPreDmaProtocolRejection = std::uint64_t{1} << 1,
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
    bool allPeersCanRejectBeforeDma{false};
    std::optional<std::uint16_t> selectedVersion;
    std::string reason;
};

namespace detail
{

// The terminal envelope avoids interpreting an operator-controlled hostname fragment as protocol metadata. A future
// descriptor schema must use a new versioned marker rather than adding fields to this v1 envelope.
constexpr std::string_view kPeerProtocolMarker{"__trtllm_protocol_v1_"};
constexpr std::string_view kPeerProtocolTerminator{"__trtllm_protocol_end"};
constexpr std::uint16_t kPeerProtocolMinVersion{1};
constexpr std::uint16_t kPeerProtocolMaxVersion{1};
constexpr std::uint64_t kInflightCancellationCapability
    = static_cast<std::uint64_t>(PeerProtocolCapability::kInflightCancellation);
constexpr std::uint64_t kPreDmaProtocolRejectionCapability
    = static_cast<std::uint64_t>(PeerProtocolCapability::kPreDmaProtocolRejection);
constexpr std::uint64_t kAdvertisedCapabilities = kInflightCancellationCapability | kPreDmaProtocolRejectionCapability;

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

inline std::optional<std::array<std::string_view, 5>> splitProtocolFields(std::string_view payload)
{
    std::array<std::string_view, 5> fields;
    for (size_t field = 0; field < fields.size(); ++field)
    {
        auto const separator = payload.find('_');
        if (field + 1 == fields.size())
        {
            if (separator != std::string_view::npos)
            {
                return std::nullopt;
            }
            fields[field] = payload;
            return fields;
        }
        if (separator == std::string_view::npos)
        {
            return std::nullopt;
        }
        fields[field] = payload.substr(0, separator);
        payload.remove_prefix(separator + 1);
    }
    return std::nullopt;
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
        detail::kAdvertisedCapabilities,
        inflightCancellationEnabled ? PeerCancellationMode::kEnabled : PeerCancellationMode::kBaseline, sessionToken};
}

//! Append an opaque, backward-compatible protocol suffix to the actual NIXL agent identity.
inline std::string appendPeerProtocolDescriptor(std::string agentName, PeerProtocolDescriptor const& descriptor)
{
    std::ostringstream suffix;
    suffix << detail::kPeerProtocolMarker << descriptor.minVersion << '_' << descriptor.maxVersion << '_' << std::hex
           << descriptor.capabilities << '_' << std::dec << static_cast<unsigned>(descriptor.cancellationMode) << '_'
           << std::hex << std::setw(16) << std::setfill('0') << descriptor.sessionToken;
    suffix << detail::kPeerProtocolTerminator;
    agentName += suffix.str();
    return agentName;
}

//! Parse a descriptor suffix without changing the transport identity string.
inline PeerProtocolParseResult parsePeerProtocolDescriptor(std::string_view const agentName)
{
    if (agentName.size() < detail::kPeerProtocolTerminator.size()
        || agentName.substr(agentName.size() - detail::kPeerProtocolTerminator.size())
            != detail::kPeerProtocolTerminator)
    {
        return {PeerProtocolParseStatus::kMissing, std::nullopt, "peer does not advertise a protocol descriptor"};
    }

    auto const envelopeEnd = agentName.size() - detail::kPeerProtocolTerminator.size();
    auto const markerPos = agentName.rfind(detail::kPeerProtocolMarker, envelopeEnd);
    if (markerPos == std::string_view::npos)
    {
        return {PeerProtocolParseStatus::kMissing, std::nullopt, "peer does not advertise a protocol descriptor"};
    }

    auto const payloadBegin = markerPos + detail::kPeerProtocolMarker.size();
    auto const fields = detail::splitProtocolFields(agentName.substr(payloadBegin, envelopeEnd - payloadBegin));
    if (!fields.has_value())
    {
        return {PeerProtocolParseStatus::kMalformed, std::nullopt,
            "protocol descriptor must contain exactly version range, capabilities, mode, and session token"};
    }

    std::uint64_t minVersion = 0;
    std::uint64_t maxVersion = 0;
    std::uint64_t capabilities = 0;
    std::uint64_t mode = 0;
    std::uint64_t sessionToken = 0;
    bool const parsed = detail::parseUnsigned(fields.value()[0], /*base=*/10, minVersion)
        && detail::parseUnsigned(fields.value()[1], /*base=*/10, maxVersion)
        && detail::parseUnsigned(fields.value()[2], /*base=*/16, capabilities)
        && detail::parseUnsigned(fields.value()[3], /*base=*/10, mode)
        && detail::parseUnsigned(fields.value()[4], /*base=*/16, sessionToken);
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
    PeerProtocolDescriptor const& localDescriptor, std::vector<std::string_view> const& peerAgentNames)
{
    if (peerAgentNames.empty())
    {
        return {false, false, false, false, std::nullopt, "peer agent-state list is empty"};
    }
    if (localDescriptor.minVersion == 0 || localDescriptor.minVersion > localDescriptor.maxVersion)
    {
        return {false, false, false, false, std::nullopt, "local protocol version range is invalid"};
    }
    if (localDescriptor.sessionToken == 0)
    {
        return {false, false, false, false, std::nullopt, "local protocol session token must be nonzero"};
    }
    if (localDescriptor.cancellationMode == PeerCancellationMode::kEnabled
        && (localDescriptor.capabilities & detail::kInflightCancellationCapability) == 0)
    {
        return {false, false, false, false, std::nullopt,
            "local protocol descriptor lacks the in-flight cancellation capability"};
    }
    if (localDescriptor.cancellationMode == PeerCancellationMode::kEnabled
        && (localDescriptor.capabilities & detail::kPreDmaProtocolRejectionCapability) == 0)
    {
        return {false, false, false, false, std::nullopt,
            "local protocol descriptor lacks the pre-DMA protocol-rejection capability"};
    }

    bool hasLegacyPeer = false;
    bool allPeersCanRejectBeforeDma = true;
    for (size_t rank = 0; rank < peerAgentNames.size(); ++rank)
    {
        auto const parsed = parsePeerProtocolDescriptor(peerAgentNames[rank]);
        if (parsed.status == PeerProtocolParseStatus::kMissing)
        {
            hasLegacyPeer = true;
            allPeersCanRejectBeforeDma = false;
            continue;
        }
        if (parsed.status == PeerProtocolParseStatus::kMalformed)
        {
            return {false, hasLegacyPeer, false, false, std::nullopt,
                "peer rank " + std::to_string(rank) + " has a malformed protocol descriptor: " + parsed.reason};
        }
        allPeersCanRejectBeforeDma
            &= (parsed.descriptor->capabilities & detail::kPreDmaProtocolRejectionCapability) != 0;
    }

    bool const allPeersProtocolAware = !hasLegacyPeer;
    auto const localMode = localDescriptor.cancellationMode;
    if (localMode == PeerCancellationMode::kEnabled && hasLegacyPeer)
    {
        return {false, true, false, false, std::nullopt,
            "local cancellation mode is enabled but one or more peer ranks do not advertise compatible "
            "cancellation semantics"};
    }

    std::uint16_t commonMinVersion = localDescriptor.minVersion;
    std::uint16_t commonMaxVersion = localDescriptor.maxVersion;
    for (size_t rank = 0; rank < peerAgentNames.size(); ++rank)
    {
        auto const parsed = parsePeerProtocolDescriptor(peerAgentNames[rank]);
        if (parsed.status == PeerProtocolParseStatus::kMissing)
        {
            continue;
        }
        auto const& peer = parsed.descriptor.value();
        if (peer.cancellationMode != localMode)
        {
            return {false, hasLegacyPeer, allPeersProtocolAware, allPeersCanRejectBeforeDma, std::nullopt,
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

        commonMinVersion = std::max(commonMinVersion, peer.minVersion);
        commonMaxVersion = std::min(commonMaxVersion, peer.maxVersion);
        if (commonMinVersion > commonMaxVersion)
        {
            return {false, hasLegacyPeer, allPeersProtocolAware, allPeersCanRejectBeforeDma, std::nullopt,
                "peer rank " + std::to_string(rank) + " supports protocol versions " + std::to_string(peer.minVersion)
                    + "-" + std::to_string(peer.maxVersion) + ", but this binary supports "
                    + std::to_string(localDescriptor.minVersion) + "-" + std::to_string(localDescriptor.maxVersion)
                    + " and no version is common to every peer rank"};
        }
        if ((peer.capabilities & detail::kInflightCancellationCapability) == 0)
        {
            return {false, hasLegacyPeer, allPeersProtocolAware, allPeersCanRejectBeforeDma, std::nullopt,
                "peer rank " + std::to_string(rank) + " lacks the in-flight cancellation capability"};
        }
        if ((peer.capabilities & detail::kPreDmaProtocolRejectionCapability) == 0)
        {
            return {false, hasLegacyPeer, allPeersProtocolAware, false, std::nullopt,
                "peer rank " + std::to_string(rank) + " lacks the pre-DMA protocol-rejection capability"};
        }
    }

    return {true, hasLegacyPeer, allPeersProtocolAware, allPeersCanRejectBeforeDma,
        localMode == PeerCancellationMode::kEnabled ? std::make_optional(commonMaxVersion) : std::nullopt,
        hasLegacyPeer ? "compatible legacy baseline peer" : "compatible protocol and effective mode"};
}

inline PeerProtocolCompatibility validatePeerProtocol(
    PeerCancellationMode const localMode, std::vector<std::string_view> const& peerAgentNames)
{
    auto localDescriptor = makePeerProtocolDescriptor(localMode == PeerCancellationMode::kEnabled, /*sessionToken=*/1);
    return validatePeerProtocol(localDescriptor, peerAgentNames);
}

//! Require one exact protocol profile across the ranks of a service instance. Session tokens intentionally differ.
inline PeerProtocolCompatibility validateLocalPeerProtocol(
    PeerCancellationMode const localMode, std::vector<std::string_view> const& localAgentNames)
{
    auto compatibility = validatePeerProtocol(localMode, localAgentNames);
    if (!compatibility.compatible)
    {
        return compatibility;
    }

    std::optional<PeerProtocolDescriptor> reference;
    for (size_t rank = 0; rank < localAgentNames.size(); ++rank)
    {
        auto const parsed = parsePeerProtocolDescriptor(localAgentNames[rank]);
        if (parsed.status != PeerProtocolParseStatus::kValid)
        {
            return {false, parsed.status == PeerProtocolParseStatus::kMissing, false, false, std::nullopt,
                "local rank " + std::to_string(rank) + " does not advertise a valid protocol descriptor"};
        }
        if (!reference.has_value())
        {
            reference = parsed.descriptor;
            continue;
        }
        auto const& descriptor = parsed.descriptor.value();
        if (descriptor.minVersion != reference->minVersion || descriptor.maxVersion != reference->maxVersion
            || descriptor.capabilities != reference->capabilities
            || descriptor.cancellationMode != reference->cancellationMode)
        {
            return {false, false, true, compatibility.allPeersCanRejectBeforeDma, std::nullopt,
                "local rank " + std::to_string(rank)
                    + " advertises a different protocol version, capability set, or effective mode"};
        }
    }
    return compatibility;
}

} // namespace tensorrt_llm::executor::kv_cache
