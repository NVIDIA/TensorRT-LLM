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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/tokenIdExt.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

// Build a deterministic 32-byte digest from a seed byte.
Digest makeDigest(std::byte seed)
{
    Digest digest;
    for (size_t i = 0; i < kDIGEST_LEN; ++i)
    {
        digest[i] = static_cast<std::byte>(static_cast<uint8_t>(seed) + static_cast<uint8_t>(i));
    }
    return digest;
}

// Core cache-key-stability guarantee: hashing an all-normal token block in one
// bulk Write must be bit-identical to the per-element path.
TEST(DigestPoolTest, BulkHashEqualsPerElementForNormalTokens)
{
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 37; ++i) // odd, non-power-of-two count
    {
        tokens.emplace_back(TokenId{i * 7 + 1});
    }
    ASSERT_FALSE(std::any_of(tokens.begin(), tokens.end(), [](TokenIdExt const& t) { return t.isDigest(); }));

    // Per-element (slow) path.
    Hasher slow;
    for (auto const& tok : tokens)
    {
        slow.update(tok);
    }

    // Bulk (fast) path — knownNoDigest=true does one bulk Write, no per-element scan.
    Hasher fast;
    fast.update(tokens.data(), tokens.size(), /*knownNoDigest=*/true);

    EXPECT_EQ(slow.digest(), fast.digest());
}

// Two digests with identical bytes in DISTINCT pool slots must compare equal
// (equality dereferences the pool) and hash identically. Slots are released when
// the owning tokens go out of scope.
TEST(DigestPoolTest, DigestValueEqualityAcrossDistinctSlots)
{
    size_t const baseline = detail::digestPoolLiveCount();
    Digest const bytes = makeDigest(std::byte{0x42});

    TokenIdExt const tokA(bytes);
    TokenIdExt const tokB(bytes);
    EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 2); // two distinct slots
    ASSERT_TRUE(tokA.isDigest());
    EXPECT_EQ(tokA, tokB);                                  // by-value (pooled) equality
    EXPECT_NE(tokA, TokenIdExt(TokenId{5}));                // digest != normal

    // Hashing the two distinct-slot digests yields the same contribution.
    Hasher hashA;
    hashA.update(tokA);
    Hasher hashB;
    hashB.update(tokB);
    EXPECT_EQ(hashA.digest(), hashB.digest());
}

// A copied digest token clones its slot; both are freed on destruction.
TEST(DigestPoolTest, CopyDigestTokenClonesSlot)
{
    size_t const baseline = detail::digestPoolLiveCount();
    Digest const bytes = makeDigest(std::byte{0x5A});
    {
        TokenIdExt const original(bytes);
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 1);
        TokenIdExt const copy = original; // clone → second slot
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 2);
        EXPECT_EQ(original, copy);
        EXPECT_EQ(copy.digest(), bytes);
    }
    EXPECT_EQ(detail::digestPoolLiveCount(), baseline); // both slots freed
}

// Moving transfers ownership without cloning or double-freeing a digest slot.
TEST(DigestPoolTest, MoveTransfersSlotWithoutCloning)
{
    size_t const baseline = detail::digestPoolLiveCount();
    Digest const bytes = makeDigest(std::byte{0x33});
    Digest const other = makeDigest(std::byte{0x77});
    {
        TokenIdExt source(bytes);
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 1);

        TokenIdExt moved(std::move(source));
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 1);
        EXPECT_EQ(source.raw(), TokenIdExt::kBadToken);
        EXPECT_EQ(moved.digest(), bytes);
        EXPECT_THROW((void) (source == moved), std::out_of_range);

        TokenIdExt target(other);
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 2);
        target = std::move(moved);
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 1);
        EXPECT_EQ(moved.raw(), TokenIdExt::kBadToken);
        EXPECT_EQ(target.digest(), bytes);
    }
    EXPECT_EQ(detail::digestPoolLiveCount(), baseline);
}

// clone-on-copy: the copy owns an independent slot; destroying the original
// leaves the copy valid, and slots return to the free-list (liveCount delta 0).
TEST(DigestPoolTest, CloneIndependenceAndFreeReuse)
{
    size_t const baseline = detail::digestPoolLiveCount();
    Digest const bytes = makeDigest(std::byte{0x11});

    {
        std::vector<TokenIdExt> original;
        original.emplace_back(TokenId{1});
        original.emplace_back(bytes);
        original.emplace_back(TokenId{2});
        ASSERT_TRUE(std::any_of(original.begin(), original.end(), [](TokenIdExt const& t) { return t.isDigest(); }));
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 1);

        std::vector<TokenIdExt> copy(original); // deep clone → a second slot
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 2);

        // Destroy the original; the copy's digest bytes must remain valid.
        {
            std::vector<TokenIdExt> dying(std::move(original));
        } // dying (holding original's slot) destructs here
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 1);

        ASSERT_EQ(copy.size(), 3U);
        ASSERT_TRUE(copy[1].isDigest());
        EXPECT_EQ(copy[1].digest(), bytes);
    }                                                   // copy destructs

    EXPECT_EQ(detail::digestPoolLiveCount(), baseline); // all slots freed / reused
}

// Exercise front-packed allocation + tail-shrink reclamation: allocate well past
// the shrink threshold, release, and confirm content integrity survives the
// intervening deque shrinks and that every slot is reclaimed.
TEST(DigestPoolTest, FrontPackAndTailShrinkChurn)
{
    size_t const baseline = detail::digestPoolLiveCount();

    auto distinctDigest = [](int seed)
    {
        Digest digest;
        for (size_t i = 0; i < kDIGEST_LEN; ++i)
        {
            digest[i] = static_cast<std::byte>((seed + static_cast<int>(i)) & 0xFF);
        }
        // Embed the full seed so digests are distinct beyond 256 entries.
        std::memcpy(digest.data(), &seed, sizeof(seed));
        return digest;
    };

    int const count = 600; // > kSlackHigh (256), forces growth then shrink
    std::vector<TokenIdExt> tokens;
    tokens.reserve(count);
    for (int i = 0; i < count; ++i)
    {
        tokens.emplace_back(distinctDigest(i));
    }
    EXPECT_EQ(detail::digestPoolLiveCount(), baseline + static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        EXPECT_EQ(tokens[i].digest(), distinctDigest(i)) << "content at " << i;
    }

    uint32_t const reclaimedSlot = tokens.front().raw();

    // Release the first half (front slots), then confirm the survivors are intact
    // after the shrink churn triggered by freeing.
    tokens.erase(tokens.begin(), tokens.begin() + count / 2);
    EXPECT_EQ(detail::digestPoolLiveCount(), baseline + static_cast<size_t>(count - count / 2));
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        EXPECT_EQ(tokens[i].digest(), distinctDigest(static_cast<int>(i) + count / 2)) << "survivor at " << i;
    }

    // A fresh allocation must front-pack into a reclaimed low slot.
    {
        TokenIdExt const refill(distinctDigest(9999));
        EXPECT_EQ(refill.raw(), reclaimedSlot);
        EXPECT_EQ(refill.digest(), distinctDigest(9999));

        tokens.clear();
        // `refill` still alive here.
        EXPECT_EQ(detail::digestPoolLiveCount(), baseline + 1);
    }
    // `refill` destroyed: its slot must return to the pool, or a leak in the refill
    // allocation path would go unnoticed and silently raise the baseline for later tests.
    EXPECT_EQ(detail::digestPoolLiveCount(), baseline);
}

// A block containing a sparse digest still hashes deterministically; the
// per-slice fast path applies to the all-text portion.
TEST(DigestPoolTest, MixedBlockHashesDeterministically)
{
    auto build = [](Digest const& mm)
    {
        std::vector<TokenIdExt> tokens;
        tokens.emplace_back(TokenId{10});
        tokens.emplace_back(mm);
        tokens.emplace_back(TokenId{20});
        tokens.emplace_back(TokenId{30});
        return tokens;
    };
    Digest const mm = makeDigest(std::byte{0x7E});
    std::vector<TokenIdExt> const a = build(mm);
    std::vector<TokenIdExt> const b = build(mm); // distinct slots, identical content

    Hasher ha;
    ha.update(a.data(), a.size());
    Hasher hb;
    hb.update(b.data(), b.size());
    EXPECT_EQ(ha.digest(), hb.digest());
}
} // namespace
