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

#include "tensorrt_llm/thop/thUtils.h"
#include <memory>

using namespace torch_ext;

TEST(ThUtils, ConvertShape2D)
{
    at::Tensor a = at::ones({2, 5}, at::kInt);
    auto const shape = convert_shape(a);
    ASSERT_EQ(shape.d[0], 2);
    ASSERT_EQ(shape.d[1], 5);
    ASSERT_EQ(shape.nbDims, 2);
}

TEST(ThUtils, ConvertShape1D)
{
    at::Tensor a = at::ones({20}, at::kInt);
    auto const shape = convert_shape(a);
    ASSERT_EQ(shape.d[0], 20);
    ASSERT_EQ(shape.nbDims, 1);
}
