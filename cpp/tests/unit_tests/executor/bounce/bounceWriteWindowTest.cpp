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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceWriteWindow.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

TEST(BounceReceiveWindow, ScopeIsRetiredBeforeLocalMemoryCanBeReused)
{
    b::BounceWriteWindows windows;
    auto window = windows.add("writer", "endpoint", std::string(16, 'a'), {{64, 8}, {72, 8}, {120, 8}});
    auto const& id = window->id();
    EXPECT_EQ(windows.find(id, "writer"), window);
    EXPECT_EQ(windows.find(id, "another-writer"), nullptr);
    EXPECT_TRUE(window->contains(68, 12));
    EXPECT_FALSE(window->contains(79, 2));

    auto copy = window->acquire();
    ASSERT_TRUE(copy.owns_lock());
    auto retired = std::async(std::launch::async, [&] { windows.removeIntersecting(64, 16); });
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (windows.find(id, "writer") && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }
    EXPECT_EQ(windows.find(id, "writer"), nullptr);
    EXPECT_EQ(retired.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    copy.unlock();
    retired.get();
    EXPECT_FALSE(window->acquire().owns_lock());
}

TEST(BounceReceiveWindow, InterleavedWritersReleaseTheirOwnRanges)
{
    b::BounceWriteWindows windows;
    auto first = windows.add("writer-a", "endpoint-a", std::string(16, 'a'), {{96, 16}, {64, 8}, {72, 8}});
    auto second = windows.add("writer-b", "endpoint-b", std::string(16, 'b'), {{80, 16}, {112, 16}});
    EXPECT_TRUE(first->contains(64, 16));
    EXPECT_FALSE(first->contains(79, 2));
    EXPECT_TRUE(second->contains(80, 16));
    EXPECT_FALSE(second->contains(95, 2));

    windows.removeIntersecting(80, 16);
    EXPECT_EQ(windows.find(second->id(), "writer-b"), nullptr);
    EXPECT_FALSE(second->acquire().owns_lock());
    EXPECT_EQ(windows.find(first->id(), "writer-a"), first);
    EXPECT_TRUE(first->acquire().owns_lock());

    auto replacement = windows.add("writer-b", "endpoint-b", std::string(16, 'c'), {{80, 16}});
    windows.remove(second->id());
    EXPECT_EQ(windows.find(replacement->id(), "writer-b"), replacement);
    EXPECT_TRUE(replacement->acquire().owns_lock());
}

TEST(BounceReceiveWindow, RangeBoundariesAndDuplicateRegistration)
{
    constexpr auto kEnd = std::numeric_limits<std::uint64_t>::max();
    b::BounceWriteWindows windows;
    std::string const key(16, 'a');
    auto window = windows.add("writer", "endpoint", key, {{kEnd - 8, 8}});
    EXPECT_TRUE(window->contains(kEnd - 1, 1));
    EXPECT_FALSE(window->contains(kEnd - 1, 2));
    EXPECT_FALSE(window->contains(kEnd, 1));
    EXPECT_FALSE(window->intersects(kEnd, 1));
    EXPECT_FALSE(window->intersects(kEnd - 1, 0));
    EXPECT_THROW(windows.add("writer", "endpoint", key, {{32, 8}}), std::invalid_argument);
    EXPECT_EQ(windows.find(window->id(), "writer"), window);
    EXPECT_FALSE(window->contains(32, 8));
}

TEST(BounceReceiveWindow, SignedMessageUsesTheIssuedWindow)
{
    std::string const key(16, 'a');
    std::string const otherKey(16, 'b');
    EXPECT_EQ(b::bounceWindowId(key), "\x77\x64\xd0\x2a\xdb\xa0\x6f\x44\xf4\x25\x3e\x8c\x32\xa1\xc7\xd3");
    EXPECT_EQ(b::bounceWindowId(otherKey), "\xcc\x87\xc9\x23\xce\xac\xcd\xe5\xfc\x39\x7b\x75\x67\x4e\xf4\xa9");
    std::string const message = "completed";
    auto const envelope = b::signBounceFrame(message, key);
    std::string id;
    ASSERT_TRUE(b::readBounceWindowId(envelope, message.size(), id));
    EXPECT_EQ(id, b::bounceWindowId(key));
    EXPECT_TRUE(b::verifyBounceFrame(envelope, key));
    EXPECT_FALSE(b::verifyBounceFrame(envelope, otherKey));
}

TEST(BounceReceiveWindow, SelectedSlotsPreservePaddingHolesAndWriterPieces)
{
    std::array<std::int64_t, 3> const slots{2, -1, 0};
    std::array<std::int64_t, 2> const offsets{0, 12};
    std::array<std::int64_t, 2> const lengths{4, 4};
    b::BounceWriteWindows windows;
    auto window
        = windows.addSlots("writer", "endpoint", std::string(16, 'w'), slots, {64, 32, 20, 3, offsets, lengths});
    for (auto base : {64U, 128U})
    {
        EXPECT_TRUE(window->contains(base, 4));
        EXPECT_TRUE(window->contains(base + 12, 4));
        EXPECT_FALSE(window->contains(base + 3, 2));
        EXPECT_FALSE(window->contains(base + 16, 1));
        EXPECT_FALSE(window->contains(base + 20, 1));
    }
    EXPECT_FALSE(window->contains(96, 1));
}

TEST(BounceReceiveWindow, InvalidSlotLayoutDoesNotPublishAWindow)
{
    std::array<std::int64_t, 1> const offsets{8};
    std::array<std::int64_t, 1> const lengths{8};
    std::array<std::int64_t, 2> const slots{0, 2};
    b::BounceSlotLayout layout{64, 32, 16, 2, offsets, lengths};
    std::string const key(16, 'w');
    b::BounceWriteWindows windows;
    EXPECT_THROW(windows.addSlots("writer", "endpoint", key, slots, layout), std::invalid_argument);
    layout.numSlots = 3;
    layout.slotBytes = 12;
    EXPECT_THROW(windows.addSlots("writer", "endpoint", key, slots, layout), std::invalid_argument);
    layout.slotBytes = 16;
    layout.base = std::numeric_limits<std::uint64_t>::max() - 64;
    EXPECT_THROW(windows.addSlots("writer", "endpoint", key, slots, layout), std::invalid_argument);
    EXPECT_EQ(windows.find(b::bounceWindowId(key), "writer"), nullptr);
}

TEST(BounceReceiveWindow, SealedWindowRetainsItsWriterUntilRetirement)
{
    b::BounceWriteWindows windows;
    auto window = windows.add("writer", "endpoint", std::string(16, 'a'), {{32, 16}});
    windows.seal(window->id());
    auto sealed = windows.find(window->id(), "writer");
    ASSERT_EQ(sealed, window);
    EXPECT_FALSE(sealed->acquire().owns_lock());
    EXPECT_EQ(windows.find(window->id(), "other-writer"), nullptr);
    windows.remove(window->id());
    EXPECT_EQ(windows.find(window->id(), "writer"), nullptr);
}
