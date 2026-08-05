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

#include "tensorrt_llm/common/assert.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// Vocabulary token identifier (normal tokens only). 32-bit, matching the
// runtime-wide tensorrt_llm::runtime::TokenIdType — real token ids are < vocab
// size (~10^5-10^6), well within int32.
using TokenId = std::int32_t;

// 32-byte aligned to enable SIMD.
inline constexpr int kDIGEST_LEN = 32;

struct alignas(kDIGEST_LEN) Digest : std::array<std::byte, kDIGEST_LEN>
{
    // Custom operator== needed to emit SIMD code
    bool operator==(Digest const& o) const noexcept
    {
        return std::memcmp(this, &o, kDIGEST_LEN) == 0;
    }

    bool operator!=(Digest const& o) const noexcept
    {
        return !(*this == o);
    }
};

// ---------------------------------------------------------------------------
// TokenIdExt — 4-byte self-describing token handle (RAII value type).
//
// One uint32_t; the high bit tags the low 31 bits:
//   - tag 0: normal token id (stored verbatim). An all-normal array is a
//     contiguous little-endian int32 array, hashed in one CSHA256::Write(N*4).
//   - tag 1: multi-modal digest; low bits index a slot in an internal pool that
//     holds the 32-byte Digest.
//
// A digest handle owns its pool slot: construct from a Digest to allocate, copy
// to clone into a fresh slot, and destroy to free. Normal handles own nothing.
// Pool-touching members are defined out-of-line in tokenIdExt.cpp.
// ---------------------------------------------------------------------------
class TokenIdExt
{
public:
    static constexpr uint32_t kTagMask = 0x80000000U;
    static constexpr uint32_t kValueMask = 0x7FFFFFFFU;
    // Sentinel for default / moved-from handles: a *digest* (tag 1) whose index is
    // the reserved value kValueMask — never a real slot. Tagging it as a digest
    // leaves the whole tag-0 space to real token ids, and its destructor is a
    // no-op because DigestPool::free ignores that index. Copying or dereferencing
    // a moved-from handle is a bug and fails loudly (duplicate/get hit a bad slot).
    static constexpr uint32_t kBadToken = kTagMask | kValueMask; // 0xFFFFFFFF
    // Maximum normal token id — the full 31-bit range. (Digest slot indices stay
    // strictly below kValueMask, which is reserved for the sentinel above.)
    static constexpr uint32_t kMaxValue = kValueMask;

    TokenIdExt() noexcept = default; // kBadToken (see mBits initializer)

    // Normal token id (tag 0). Precondition: 0 <= id <= kMaxValue (so the value
    // fits the 31-bit field and leaves the digest tag bit clear).
    explicit TokenIdExt(TokenId id)
        : mBits(static_cast<uint32_t>(id))
    {
        TLLM_CHECK_DEBUG(id >= 0 && id <= static_cast<TokenId>(kMaxValue));
    }

    // Multi-modal digest (tag 1): copies `digest` into a fresh pool slot.
    explicit TokenIdExt(Digest const& digest);

    ~TokenIdExt();
    TokenIdExt(TokenIdExt const& other);            // clones a digest slot
    TokenIdExt& operator=(TokenIdExt const& other); // clones a digest slot

    TokenIdExt(TokenIdExt&& other) noexcept
        : mBits(other.mBits)
    {
        other.mBits = kBadToken; // steal the slot; leave source empty
    }

    TokenIdExt& operator=(TokenIdExt&& other) noexcept;

    [[nodiscard]] bool isDigest() const noexcept
    {
        return (mBits & kTagMask) != 0;
    }

    // Valid iff !isDigest(). A normal handle's tag bit is clear by construction
    // (the ctor range-checks id <= kMaxValue), so the raw bits are already the id.
    [[nodiscard]] TokenId tokenId() const noexcept
    {
        TLLM_CHECK_DEBUG(!isDigest());
        return static_cast<TokenId>(mBits);
    }

    // The pooled 32-byte digest. Precondition: isDigest().
    [[nodiscard]] Digest const& digest() const;

    // Raw 4-byte payload — consumed by the bulk-hash fast path.
    [[nodiscard]] uint32_t raw() const noexcept
    {
        return mBits;
    }

    // Value equality: normal/tag-mismatch compare raw bits; digest-vs-digest
    // compares the pooled 32 bytes (equal content in different slots is equal).
    bool operator==(TokenIdExt const& other) const;

    bool operator!=(TokenIdExt const& other) const
    {
        return !(*this == other);
    }

private:
    [[nodiscard]] uint32_t digestIndex() const noexcept
    {
        TLLM_CHECK_DEBUG(isDigest());
        return mBits & kValueMask;
    }

    uint32_t mBits{kBadToken};
};

static_assert(sizeof(TokenIdExt) == 4, "TokenIdExt must be exactly 4 bytes for bulk hashing");
static_assert(std::is_standard_layout_v<TokenIdExt>, "TokenIdExt must be standard-layout for byte-stream hashing");

// The digest pool that backs digest-tagged TokenIdExt is an implementation
// detail hidden entirely in tokenIdExt.cpp (anonymous namespace). Only this
// introspection hook is exposed, for tests.
namespace detail
{

// Live digest-pool slot count. Compare as a delta vs a captured baseline because
// the pool is a process-global singleton.
[[nodiscard]] size_t digestPoolLiveCount();

} // namespace detail

// Token sequences are held directly as std::vector<TokenIdExt>. TokenIdExt is an
// RAII value type (copy = clone the digest slot, destroy = free it), so a plain
// vector copies/destroys correctly with no wrapper. The "is this sequence
// digest-free" summary that once lived here is now an explicit knownNoDigest /
// text_only flag threaded from the request/model level (see blockRadixTree Hasher
// and KvCache::textOnly).

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2

// std::hash specialization for Digest/BlockKey so unordered_map works without a custom hasher.
template <>
struct std::hash<tensorrt_llm::batch_manager::kv_cache_manager_v2::Digest>
{
    size_t operator()(tensorrt_llm::batch_manager::kv_cache_manager_v2::Digest const& k) const noexcept
    {
        // First 8 bytes of a SHA-256 digest are already well-distributed.
        uint64_t v;
        std::memcpy(&v, k.data(), sizeof(v));
        return static_cast<size_t>(v);
    }
};
