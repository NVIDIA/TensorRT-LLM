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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceMessage.h"

#include "tensorrt_llm/executor/serializeUtils.h"

#include <cstring>
#include <sstream>

namespace tensorrt_llm::executor::kv_cache::bounce
{

namespace
{

BounceMsgHeader makeHeader(BounceMsgType type, std::uint64_t requestId, std::uint32_t chunkIdx, std::uint32_t numChunks,
    std::uint64_t regionHandle, std::uint32_t count, std::uint32_t payloadBytes)
{
    BounceMsgHeader header{};
    header.magic = kBounceMagic;
    header.version = kBounceVersion;
    header.msgType = static_cast<std::uint16_t>(type);
    header.requestId = requestId;
    header.chunkIdx = chunkIdx;
    header.numChunks = numChunks;
    header.regionHandle = regionHandle;
    header.count = count;
    header.payloadBytes = payloadBytes;
    return header;
}

template <typename Entry>
std::string encodeWithEntries(BounceMsgHeader const& header, std::vector<Entry> const& entries)
{
    std::string blob;
    blob.resize(sizeof(BounceMsgHeader) + header.payloadBytes);
    std::memcpy(blob.data(), &header, sizeof(BounceMsgHeader));
    if (!entries.empty())
    {
        std::memcpy(blob.data() + sizeof(BounceMsgHeader), entries.data(), entries.size() * sizeof(Entry));
    }
    return blob;
}

template <typename Entry>
bool decodeEntries(std::string const& blob, BounceMsgHeader const& header, std::vector<Entry>& out)
{
    std::size_t const expectBytes = static_cast<std::size_t>(header.count) * sizeof(Entry);
    if (expectBytes != header.payloadBytes)
    {
        return false;
    }
    if (blob.size() < sizeof(BounceMsgHeader) + expectBytes)
    {
        return false;
    }
    out.resize(header.count);
    if (header.count > 0)
    {
        std::memcpy(out.data(), blob.data() + sizeof(BounceMsgHeader), expectBytes);
    }
    return true;
}

std::string encodeHeaderOnly(BounceMsgHeader const& header)
{
    std::string blob;
    blob.resize(sizeof(BounceMsgHeader));
    std::memcpy(blob.data(), &header, sizeof(BounceMsgHeader));
    return blob;
}

} // namespace

std::string encodeWant(
    std::uint64_t requestId, std::vector<std::uint32_t> const& chunkBytes, std::string const& endpoint)
{
    auto const n = static_cast<std::uint32_t>(chunkBytes.size());
    auto const sizesBytes = static_cast<std::uint32_t>(chunkBytes.size() * sizeof(std::uint32_t));
    auto const epLen = static_cast<std::uint32_t>(endpoint.size());
    auto const payloadBytes = sizesBytes + static_cast<std::uint32_t>(sizeof(std::uint32_t)) + epLen;
    auto h = makeHeader(BounceMsgType::kWANT, requestId, 0, 0, 0, n, payloadBytes);
    std::string blob;
    blob.resize(sizeof(BounceMsgHeader) + payloadBytes);
    std::memcpy(blob.data(), &h, sizeof(BounceMsgHeader));
    auto* payload = blob.data() + sizeof(BounceMsgHeader);
    if (sizesBytes > 0)
    {
        std::memcpy(payload, chunkBytes.data(), sizesBytes);
    }
    std::memcpy(payload + sizesBytes, &epLen, sizeof(epLen));
    if (epLen > 0)
    {
        std::memcpy(payload + sizesBytes + sizeof(epLen), endpoint.data(), epLen);
    }
    return blob;
}

std::string encodeCancel(std::uint64_t requestId, std::string const& endpoint)
{
    // A cancel IS an empty-chunk WANT (same wire form) — see the header. Keeping it a thin wrapper
    // (rather than a new message type) reuses the receiver's onWant/reclaim path verbatim.
    return encodeWant(requestId, {}, endpoint);
}

std::string encodeGrant(std::uint64_t requestId, std::vector<BounceCreditEntry> const& credits)
{
    auto const bytes = static_cast<std::uint32_t>(credits.size() * sizeof(BounceCreditEntry));
    auto h = makeHeader(BounceMsgType::kGRANT, requestId, 0, 0, 0, static_cast<std::uint32_t>(credits.size()), bytes);
    return encodeWithEntries(h, credits);
}

std::string encodeData(std::uint64_t requestId, std::uint32_t chunkIdx, std::uint32_t numChunks,
    std::uint64_t regionHandle, std::vector<BounceScatterRun> const& entries)
{
    auto const bytes = static_cast<std::uint32_t>(entries.size() * sizeof(BounceScatterRun));
    auto h = makeHeader(BounceMsgType::kDATA, requestId, chunkIdx, numChunks, regionHandle,
        static_cast<std::uint32_t>(entries.size()), bytes);
    return encodeWithEntries(h, entries);
}

std::string encodeAck(std::uint64_t requestId, std::uint32_t chunkIdx, std::uint64_t regionHandle)
{
    return encodeHeaderOnly(makeHeader(BounceMsgType::kACK, requestId, chunkIdx, 0, regionHandle, 0, 0));
}

std::string encodeNack(std::uint64_t requestId, std::uint32_t chunkIdx, std::uint64_t regionHandle)
{
    return encodeHeaderOnly(makeHeader(BounceMsgType::kNACK, requestId, chunkIdx, 0, regionHandle, 0, 0));
}

bool decodeHeader(std::string const& blob, BounceMsgHeader& outHeader)
{
    if (blob.size() < sizeof(BounceMsgHeader))
    {
        return false;
    }
    std::memcpy(&outHeader, blob.data(), sizeof(BounceMsgHeader));
    if (outHeader.magic != kBounceMagic || outHeader.version != kBounceVersion)
    {
        return false;
    }
    // Guard against a payloadBytes that the blob can't satisfy.
    if (blob.size() < sizeof(BounceMsgHeader) + static_cast<std::size_t>(outHeader.payloadBytes))
    {
        return false;
    }
    return true;
}

bool decodeCredits(std::string const& blob, BounceMsgHeader const& header, std::vector<BounceCreditEntry>& out)
{
    return decodeEntries(blob, header, out);
}

bool decodeScatter(std::string const& blob, BounceMsgHeader const& header, std::vector<BounceScatterRun>& out)
{
    return decodeEntries(blob, header, out);
}

namespace
{
// The handshake travels inside AgentDesc (whose fields use executor::serialize_utils), so it uses
// the SAME serialization utility rather than the raw-memcpy control-message codec above: a magic +
// format-version prefix, then each field via su::serialize. `formatVersion` versions THIS blob's
// layout (so the handshake itself can evolve); `wireVersion` inside is the control-message codec
// version being negotiated.
constexpr std::uint32_t kHandshakeMagic = 0x424E4853U; // 'B''N''H''S'
constexpr std::uint16_t kHandshakeFormatVersion = 1U;
} // namespace

std::string encodeHandshake(BounceHandshake const& handshake)
{
    namespace su = tensorrt_llm::executor::serialize_utils;
    std::ostringstream os;
    su::serialize(kHandshakeMagic, os);
    su::serialize(kHandshakeFormatVersion, os);
    su::serialize(handshake.wireVersion, os);
    su::serialize(static_cast<std::uint8_t>(handshake.controlKind), os);
    su::serialize(handshake.arenaUsableCapacityBytes, os);
    su::serialize(handshake.maxChunkSizeBytes, os);
    su::serialize(handshake.requestTimeoutMs, os);
    su::serialize(handshake.endpoint, os);
    return os.str();
}

bool decodeHandshake(std::string const& blob, BounceHandshake& out)
{
    namespace su = tensorrt_llm::executor::serialize_utils;
    std::istringstream is(blob);
    auto const magic = su::deserialize<std::uint32_t>(is);
    auto const formatVersion = su::deserialize<std::uint16_t>(is);
    if (is.fail() || magic != kHandshakeMagic || formatVersion != kHandshakeFormatVersion)
    {
        return false; // absent / foreign / future-format blob -> peer treated as non-bounce
    }
    out.wireVersion = su::deserialize<std::uint16_t>(is);
    out.controlKind = static_cast<BounceControlKind>(su::deserialize<std::uint8_t>(is));
    out.arenaUsableCapacityBytes = su::deserialize<std::uint64_t>(is);
    out.maxChunkSizeBytes = su::deserialize<std::uint64_t>(is);
    out.requestTimeoutMs = su::deserialize<std::int32_t>(is);
    auto const endpointLen = su::deserialize<std::size_t>(is);
    if (is.fail())
    {
        return false; // truncated fixed fields (deserialize on a failed stream yields garbage)
    }
    // The endpoint is the final field, so its length must equal exactly the bytes left in the blob.
    // Bounding it here keeps the "malformed blob -> false" contract: handing a corrupt 64-bit length
    // to std::string would throw (length_error/bad_alloc) out of a peer-input path instead.
    auto const consumed = static_cast<std::size_t>(is.tellg());
    if (endpointLen != blob.size() - consumed)
    {
        return false;
    }
    out.endpoint = blob.substr(consumed);
    return true;
}

bool decodeWant(std::string const& blob, BounceMsgHeader const& header, std::vector<std::uint32_t>& outChunkBytes,
    std::string& outEndpoint)
{
    std::size_t const sizesBytes = static_cast<std::size_t>(header.count) * sizeof(std::uint32_t);
    std::size_t const off = sizeof(BounceMsgHeader);
    // Need the chunk sizes plus the u32 endpoint-length prefix.
    if (blob.size() < off + sizesBytes + sizeof(std::uint32_t))
    {
        return false;
    }
    std::uint32_t epLen = 0;
    std::memcpy(&epLen, blob.data() + off + sizesBytes, sizeof(epLen));
    std::size_t const expectPayload = sizesBytes + sizeof(std::uint32_t) + epLen;
    // payloadBytes must match exactly, and the blob must hold the whole endpoint.
    if (header.payloadBytes != expectPayload || blob.size() < off + expectPayload)
    {
        return false;
    }
    outChunkBytes.resize(header.count);
    if (header.count > 0)
    {
        std::memcpy(outChunkBytes.data(), blob.data() + off, sizesBytes);
    }
    outEndpoint.assign(blob.data() + off + sizesBytes + sizeof(std::uint32_t), epLen);
    return true;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
