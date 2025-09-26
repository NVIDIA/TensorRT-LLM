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

#include "tensorrt_llm/common/quantization.h"

#include <memory>

using namespace tensorrt_llm::common;

TEST(Quantization, Constructor)
{
    auto const defaultQuantMode = std::make_shared<QuantMode>();
    EXPECT_EQ(*defaultQuantMode, QuantMode::none());
    static_assert(QuantMode() == QuantMode::none());
    static_assert(QuantMode::int4Weights().hasInt4Weights());
    static_assert(QuantMode::int8Weights().hasInt8Weights());
    static_assert(QuantMode::activations().hasActivations());
    static_assert(QuantMode::perChannelScaling().hasPerChannelScaling());
    static_assert(QuantMode::perTokenScaling().hasPerTokenScaling());
    static_assert(QuantMode::int8KvCache().hasInt8KvCache());
    static_assert(QuantMode::fp8KvCache().hasFp8KvCache());
    static_assert(QuantMode::fp8Qdq().hasFp8Qdq());
}

TEST(Quantization, PlusMinus)
{
    QuantMode quantMode{};
    quantMode += QuantMode::activations() + QuantMode::perChannelScaling();
    EXPECT_TRUE(quantMode.hasActivations());
    EXPECT_TRUE(quantMode.hasPerChannelScaling());
    quantMode -= QuantMode::activations();
    EXPECT_FALSE(quantMode.hasActivations());
    EXPECT_TRUE(quantMode.hasPerChannelScaling());
    quantMode -= QuantMode::perChannelScaling();
    EXPECT_FALSE(quantMode.hasActivations());
    EXPECT_FALSE(quantMode.hasPerChannelScaling());
    EXPECT_EQ(quantMode, QuantMode::none());
}
