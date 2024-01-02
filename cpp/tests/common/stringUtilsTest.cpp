/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/stringUtils.h"

#include <sstream>

using namespace tensorrt_llm::common;

namespace
{

auto constexpr SHORT_STRING = "Let's format this string 5 times.";

inline std::string formatShort()
{
    return fmtstr("Let's %s this string %d times.", "format", 5);
}

auto const LONG_STRING = std::string(10000, '?');
auto constexpr LONG_PREFIX = "add me";

std::string formatLong()
{
    return fmtstr("add me%s", LONG_STRING.c_str());
}

std::ostringstream priceFormatStream;

template <typename P>
std::string formatFixed(P price)
{
    priceFormatStream.str("");
    priceFormatStream.clear();
    priceFormatStream << price;
    return priceFormatStream.str();
}

} // namespace

TEST(StringUtil, ShortStringFormat)
{
    EXPECT_EQ(SHORT_STRING, formatShort());
}

TEST(StringUtil, LongStringFormat)
{
    EXPECT_EQ(LONG_PREFIX + LONG_STRING, formatLong());
}

TEST(StringUtil, FormatFixedDecimals)
{
    auto num = 0.123456789;

    for (auto d = 1; d <= 9; ++d)
    {
        auto const fmt = std::string("%.") + std::to_string(d) + "f";
        auto prefix = fmtstr(fmt.c_str(), num);
        priceFormatStream.precision(d);
        EXPECT_EQ(prefix, formatFixed(num));
        EXPECT_EQ(prefix, formatFixed(num));
        EXPECT_EQ(prefix, formatFixed(num));
    }
}
