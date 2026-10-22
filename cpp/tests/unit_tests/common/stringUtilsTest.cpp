/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

using namespace tensorrt_llm::common;

TEST(StringUtil, str2set)
{
    {
        char delimiter{','};

        std::vector<std::string> inputs{
            {"apple,car,dog"}, {",apple,car,dog"}, {"apple,car,dog"}, {"apple,car,dog,,"}, {"apple,,,car,dog,"}};

        for (auto const& input : inputs)
        {

            auto out = str2set(input, delimiter);

            EXPECT_EQ(out.size(), 3);
            EXPECT_EQ(out.count("apple"), 1);
            EXPECT_EQ(out.count("car"), 1);
            EXPECT_EQ(out.count("dog"), 1);
            EXPECT_EQ(out.count("cat"), 0);
        }
    }
}
