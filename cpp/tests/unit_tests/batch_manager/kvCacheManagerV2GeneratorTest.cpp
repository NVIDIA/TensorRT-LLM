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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/generator.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

//! Sets the flag it was given when it goes, so a test can see whether a frame was unwound.
class Flag
{
public:
    explicit Flag(bool* destroyed) noexcept
        : mDestroyed(destroyed)
    {
    }

    Flag(Flag const&) = delete;
    Flag(Flag&&) = delete;
    Flag& operator=(Flag const&) = delete;
    Flag& operator=(Flag&&) = delete;

    ~Flag()
    {
        *mDestroyed = true;
    }

private:
    bool* mDestroyed;
};

Generator<int> evens(int limit)
{
    for (int value = 0; value < limit; value += 2)
    {
        co_yield value;
    }
}

// The bodies that need state are free functions taking a pointer, not capturing lambdas: a
// coroutine copies its parameters into the frame, but a lambda's closure is only referred to,
// so an immediately-invoked capturing lambda leaves the frame pointing at a destroyed object.
Generator<int> marksEntry(bool* entered)
{
    *entered = true;
    co_yield 1;
}

Generator<int, int> holdsAFlagAcrossTwoYields(bool* destroyed)
{
    Flag const flag{destroyed};
    static_cast<void>(co_yield 1);
    static_cast<void>(co_yield 2);
}

Generator<int> holdsAFlagFromItsFirstYield(bool* destroyed)
{
    Flag const flag{destroyed};
    co_yield 1;
}

TEST(KvCacheManagerV2Generator, OneWayYieldsLazilyAndInOrder)
{
    bool entered = false;
    auto body = marksEntry(&entered);
    EXPECT_FALSE(entered) << "the body ran before the first pull";
    EXPECT_EQ(body.value(), 1);
    EXPECT_TRUE(entered);

    std::vector<int> seen;
    for (auto value : evens(7))
    {
        seen.push_back(value);
    }
    EXPECT_EQ(seen, (std::vector<int>{0, 2, 4, 6}));
}

TEST(KvCacheManagerV2Generator, OneWayOverAnEmptySequenceYieldsNothing)
{
    auto sequence = evens(0);
    EXPECT_EQ(sequence.begin(), sequence.end());
    EXPECT_TRUE(sequence.done());
}

TEST(KvCacheManagerV2Generator, TwoWayCarriesAValueBackIntoTheBody)
{
    // Two rounds, so the second request is answered from what the first was given.
    auto body = []() -> Generator<int, int>
    {
        int const first = co_yield 10;
        int const second = co_yield first + 1;
        static_cast<void>(co_yield second * 2);
    }();

    ASSERT_EQ(body.value(), 10);
    body.send(5);
    ASSERT_EQ(body.value(), 6);
    body.send(7);
    EXPECT_EQ(body.value(), 14);
}

TEST(KvCacheManagerV2Generator, TwoWayReportsDoneOnlyAfterTheBodyReturns)
{
    auto body = []() -> Generator<int, int> { static_cast<void>(co_yield 1); }();

    ASSERT_FALSE(body.done());
    body.send(0);
    EXPECT_TRUE(body.done());
    EXPECT_FALSE(body.value().has_value()) << "a finished body is told apart by having no value";
}

TEST(KvCacheManagerV2Generator, ReturnValueArrivesOnlyWhenTheBodyFinishes)
{
    auto body = []() -> Generator<int, int, std::string>
    {
        int const given = co_yield 1;
        co_return std::string(static_cast<size_t>(given), 'x');
    }();

    ASSERT_EQ(body.value(), 1);
    EXPECT_FALSE(body.result().has_value());
    body.send(3);
    ASSERT_TRUE(body.result().has_value());
    EXPECT_EQ(*body.result(), "xxx");
}

TEST(KvCacheManagerV2Generator, ExceptionSurfacesOutOfWhicheverCallResumedTheBody)
{
    auto beforeFirstYield = []() -> Generator<int>
    {
        throw std::runtime_error("priming");
        co_return;
    }();
    EXPECT_THROW((void) beforeFirstYield.value(), std::runtime_error);

    auto afterFirstYield = []() -> Generator<int, int>
    {
        static_cast<void>(co_yield 1);
        throw std::runtime_error("resuming");
    }();
    ASSERT_EQ(afterFirstYield.value(), 1);
    EXPECT_THROW(afterFirstYield.send(0), std::runtime_error);
}

TEST(KvCacheManagerV2Generator, DestroyingASuspendedBodyRunsItsLocals)
{
    // A layout body holds its guards and buffers across the yield, so abandoning it -- which
    // is what a throw between asking and answering does -- has to release them.
    bool destroyed = false;
    {
        auto body = holdsAFlagAcrossTwoYields(&destroyed);
        ASSERT_EQ(body.value(), 1);
        ASSERT_FALSE(destroyed);
    }
    EXPECT_TRUE(destroyed);
}

TEST(KvCacheManagerV2Generator, DestroyingAnUnstartedBodyRunsNothing)
{
    bool destroyed = false;
    {
        auto body = holdsAFlagFromItsFirstYield(&destroyed);
        (void) body;
    }
    EXPECT_FALSE(destroyed) << "the local is not alive until the body has run to its first yield";
}

TEST(KvCacheManagerV2Generator, MovingAPrimedBodyCarriesThatItHasStarted)
{
    // Moving an already-suspended body and then resuming it must not re-prime it: that would
    // resume it with nothing sent, and the body would read whatever the empty slot held.
    auto body = []() -> Generator<int, int>
    {
        int const given = co_yield 1;
        static_cast<void>(co_yield given * 10);
    }();
    ASSERT_EQ(body.value(), 1);

    std::vector<Generator<int, int>> moved;
    moved.push_back(std::move(body));
    moved.back().send(4);
    EXPECT_EQ(moved.back().value(), 40);
}

} // namespace
