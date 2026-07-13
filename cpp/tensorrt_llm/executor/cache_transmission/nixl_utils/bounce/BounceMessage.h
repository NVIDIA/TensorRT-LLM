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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// Bounce v2 control-plane wire format. Carried over the ControlChannel (zmq by default).
/// Pure encode/decode — no IO — so it is fully unit-testable. NOTE: this is NOT the NIXL
/// notifMsg (v2 does not use notifMsg at all); it is our own message set.
///
/// ENDIANNESS: fields are memcpy'd raw (host byte order), so this assumes both peers are
/// little-endian — true for all NVIDIA GPU hosts (x86_64 / aarch64 LE). It is NOT a portable
/// cross-endian format; if a big-endian peer is ever introduced, add explicit byte-order
/// normalization here. `decodeHeader` validates magic + version + payload length; `dispatch`
/// treats any unknown `msgType` as a no-op (default branch), so a bogus type can't misroute.
enum class BounceMsgType : std::uint16_t
{
    kWANT = 1,  // sender -> receiver: per-chunk byte sizes[] for requestId (empty list cancels)
    kGRANT = 2, // receiver -> sender: credits[] for requestId
    kDATA = 3,  // sender -> receiver: region written; scatter entries[] (after getXferStatus SUCCESS)
    kACK = 4,   // receiver -> sender: chunk scattered; region freed
};

#pragma pack(push, 1)

/// Fixed 44-byte header prefixing every message. Field meaning is per-type (documented at each
/// encode function); unused fields are 0.
struct BounceMsgHeader
{
    std::uint32_t magic;     // kMagic
    std::uint16_t version;   // kVersion
    std::uint16_t msgType;   // BounceMsgType
    std::uint64_t requestId; // WANT/GRANT/DATA/ACK
    std::uint64_t
        regionHandle; // DATA/ACK: the arena region offset of this chunk; 0 elsewhere (64-bit -> arena may exceed 4 GiB)
    std::uint32_t chunkIdx;     // DATA/ACK
    std::uint32_t numChunks;    // DATA/ACK
    std::uint32_t count;        // number of trailing entries (credits / chunk sizes / scatter)
    std::uint32_t payloadBytes; // bytes of trailing payload
    std::uint32_t aux;          // WANT: numChunks; 0 elsewhere
};

/// Credit = one variable-size receiver region the sender may RDMA-write into. `addr` is its
/// absolute remote address, `len` its size (the chunk's packed bytes), `devId` the RECEIVER's GPU
/// (so it works under non-symmetric device indices), and `regionHandle` a 64-bit opaque id (the
/// receiver's arena offset) the sender echoes back in DATA so the receiver locates + frees it.
struct BounceCreditEntry
{
    std::uint64_t addr;
    std::uint32_t len;
    std::uint32_t devId;
    std::uint64_t regionHandle; // 64-bit (was 32-bit + pad) -> arena offsets may exceed 4 GiB
};

/// One scatter RUN inside a DATA message: `count` equal pieces of `pieceSize` bytes; piece p copies
///   region[bounceOffset + p*bounceStride .. +pieceSize)  ->  dstAddr + p*dstStride.
/// count==1 is a single plain extent (strides unused, 0). Runs compress BOTH dst layouts a KV
/// transfer produces: a fully CONTIGUOUS dst (one count==1 run whose pieceSize covers the chunk —
/// e.g. ctx tp1 -> gen tp4, the gen rank's head-slice pool is dense) and a uniformly STRIDED dst
/// (one count==N run — e.g. ctx tp4 -> gen DP, each ctx rank's head slice lands every
/// bytes-per-layer in the gen rank's full-head pool). Either way a whole chunk's scatter plan is a
/// handful of runs instead of one entry per desc, keeping the DATA message (ACK critical path) tiny.
struct BounceScatterRun
{
    std::uint64_t bounceOffset; // first piece's offset within the granted region
    std::uint64_t dstAddr;      // first piece's destination address
    std::uint64_t dstStride;    // dst step between pieces (0 when count==1)
    std::uint32_t bounceStride; // region-offset step between pieces (0 when count==1)
    std::uint32_t pieceSize;    // bytes per piece
    std::uint32_t count;        // number of pieces (>= 1)
};

#pragma pack(pop)

static_assert(sizeof(BounceMsgHeader) == 44, "BounceMsgHeader must be 44 bytes");
static_assert(sizeof(BounceCreditEntry) == 24, "BounceCreditEntry must be 24 bytes");
static_assert(sizeof(BounceScatterRun) == 36, "BounceScatterRun must be 36 bytes");

inline constexpr std::uint32_t kBounceMagic = 0x424E4332U; // 'B''N''C''2'
// v2: DATA scatter entries became strided RUNS (BounceScatterRun). decodeHeader rejects mismatched
// versions, so a mixed-version pair degrades to leaseTimeout (deploy sender+receiver together).
inline constexpr std::uint16_t kBounceVersion = 2U;

// ---- encode (each returns a self-contained blob: header + payload) ----
/// WANT carries the per-chunk byte sizes the sender will write (the receiver allocates a region of
/// each size as it grants) AND the sender's own bounce control endpoint. An EMPTY size list cancels.
/// The endpoint lets the receiver self-bootstrap the reverse control path (addPeer the sender) so
/// GRANT/ACK can flow back even though the disagg metadata exchange is one-directional — the KV
/// sender loads our metadata, but we never load the sender's. Payload layout:
///   [count * u32 chunkBytes][u32 endpointLen][endpointLen bytes endpoint]
/// decode via decodeWant.
[[nodiscard]] std::string encodeWant(
    std::uint64_t requestId, std::vector<std::uint32_t> const& chunkBytes, std::string const& endpoint);
/// Cancel/retract a request: a WANT with an EMPTY chunk list (the receiver frees everything it
/// allocated/held for `requestId`). This is the ONLY meaning of an empty WANT — submit() never sends
/// a zero-chunk WANT (a 0-chunk transfer resolves SUCCESS without any WANT) — so an empty chunk list
/// is unambiguously a cancel. Thin named wrapper over encodeWant to make the intent explicit at call
/// sites; the wire form is identical (still a WANT), so it reuses the receiver's onWant/reclaim path.
[[nodiscard]] std::string encodeCancel(std::uint64_t requestId, std::string const& endpoint);
[[nodiscard]] std::string encodeGrant(std::uint64_t requestId, std::vector<BounceCreditEntry> const& credits);
[[nodiscard]] std::string encodeData(std::uint64_t requestId, std::uint32_t chunkIdx, std::uint32_t numChunks,
    std::uint64_t regionHandle, std::vector<BounceScatterRun> const& entries);
[[nodiscard]] std::string encodeAck(std::uint64_t requestId, std::uint32_t chunkIdx, std::uint64_t regionHandle);

// ---- decode ----
/// Lightweight prefix check: does `blob` start with the bounce magic? (Lets a shared channel
/// distinguish bounce control traffic from other notifications.)
[[nodiscard]] bool hasBounceMagic(std::string const& blob);

/// Decode the fixed header. Returns false on short blob, bad magic, or version mismatch.
[[nodiscard]] bool decodeHeader(std::string const& blob, BounceMsgHeader& outHeader);

/// Decode credit entries (GRANT payload). `header.count` entries expected.
[[nodiscard]] bool decodeCredits(
    std::string const& blob, BounceMsgHeader const& header, std::vector<BounceCreditEntry>& out);

/// Decode scatter entries (DATA payload). `header.count` entries expected.
[[nodiscard]] bool decodeScatter(
    std::string const& blob, BounceMsgHeader const& header, std::vector<BounceScatterRun>& out);

/// Decode a WANT: its per-chunk byte sizes (`header.count` entries; empty == cancel) and the
/// sender's bounce control endpoint. Returns false on a malformed/short blob.
[[nodiscard]] bool decodeWant(std::string const& blob, BounceMsgHeader const& header,
    std::vector<std::uint32_t>& outChunkBytes, std::string& outEndpoint);

/// Does a decoded WANT mean "cancel/retract"? (An empty chunk list — see encodeCancel.) Makes the
/// receiver-side intent explicit instead of an inline `chunkBytes.empty()` check.
[[nodiscard]] inline bool isCancelWant(std::vector<std::uint32_t> const& chunkBytes)
{
    return chunkBytes.empty();
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
