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

#include <functional>
#include <list>

#include "tensorrt_llm/common/stlUtils.h"

TEST(StlUtils, ExclusiveScan)
{
    std::list<int> data{3, 1, 4, 1, 5}, l;
    auto it = tensorrt_llm::common::stl_utils::exclusiveScan(
        data.begin(), data.end(), std::insert_iterator<std::list<int>>(l, std::next(l.begin())), 0);
    tensorrt_llm::common::stl_utils::basicExclusiveScan(data.begin(), data.end(), it, 1, std::multiplies<>{});
    EXPECT_EQ(l, (std::list<int>{0, 3, 4, 8, 9, 1, 3, 3, 12, 12}));
}

TEST(StlUtils, InclusiveScan)
{
    std::list<int> data{3, 1, 4, 1, 5}, l;
    auto it = tensorrt_llm::common::stl_utils::inclusiveScan(
        data.begin(), data.end(), std::insert_iterator<std::list<int>>(l, std::next(l.begin())));
    tensorrt_llm::common::stl_utils::basicInclusiveScan(data.begin(), data.end(), it, std::multiplies<>{});
    EXPECT_EQ(l, (std::list<int>{3, 4, 8, 9, 14, 3, 3, 12, 12, 60}));
}
