/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/optionalRef.h"
#include <gtest/gtest.h>
#include <memory>

namespace tc = tensorrt_llm::common;

TEST(TestOptionalRef, TestOptionalRefBehavesAsExpected)
{
    tc::OptionalRef<int> a;
    EXPECT_FALSE(a.has_value());

    a = std::nullopt;
    EXPECT_FALSE(a.has_value());

    // Assign a nullptr container. It should still be empty:
    std::unique_ptr<int> b;
    a = b;
    EXPECT_FALSE(a.has_value());

    std::shared_ptr<int> c;
    a = c;
    EXPECT_FALSE(a.has_value());

    // Give values and check it takes the correct value:
    int d = 1;
    a = d;
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(*a, d);

    *a = 2;
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(*a, d);

    b = std::make_unique<int>(3);
    a = b;
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(*a, *b);

    *a = 4;
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(*a, *b);

    c = std::make_shared<int>(5);
    a = c;
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(*a, *c);

    *a = 6;
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(*a, *c);

    // Create an object of some class and test it can invoke it:
    struct A
    {
        bool foo()
        {
            return true;
        }
    };

    A aObj{};
    tc::OptionalRef<A> optA = aObj;
    EXPECT_TRUE(optA.has_value());
    EXPECT_TRUE(optA->foo());

    // A const optionalref can ingest a non-const container:
    auto nonConstAContainer = std::make_shared<A>();
    tc::OptionalRef<A const> constA = nonConstAContainer;
    EXPECT_TRUE(constA.has_value());

    auto nonConstAUniqueContainer = std::make_unique<A>();
    constA = nonConstAUniqueContainer;
    EXPECT_TRUE(constA.has_value());

    A& aReference = aObj;
    constA = aReference;
    EXPECT_TRUE(constA.has_value());

    // Use the explicit bool operator:
    EXPECT_TRUE(constA);
}

// A non-const optional ref cannot ingest a const container
static_assert(!std::is_constructible_v<tc::OptionalRef<int>, std::shared_ptr<int const>>);
static_assert(!std::is_constructible_v<tc::OptionalRef<int>, std::unique_ptr<int const>>);
static_assert(!std::is_constructible_v<tc::OptionalRef<int>, int const&>);
