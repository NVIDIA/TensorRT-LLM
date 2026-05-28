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

#include "tensorrt_llm/common/quantization.h"

#include <memory>
#include <optional>

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
    static_assert(QuantMode::turboQuant4KvCache().hasTurboQuant4KvCache());
    static_assert(QuantMode::fp8Qdq().hasFp8Qdq());
    static_assert(QuantMode::w4a8Nvfp4Fp8().hasW4a8Nvfp4Fp8());
    static_assert(QuantMode::w4a16Mxfp4().hasW4a16Mxfp4());
}

TEST(Quantization, TurboQuant4KvCache)
{
    auto const quantMode = QuantMode::turboQuant4KvCache();
    EXPECT_TRUE(quantMode.hasTurboQuant4KvCache());
    EXPECT_TRUE(quantMode.hasKvCacheQuant());
    EXPECT_FALSE(quantMode.hasFp4KvCache());
    EXPECT_EQ(quantMode.value(), QuantMode::BaseType(1u) << 18);

    auto const fromDescription = QuantMode::fromDescription(false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, true);
    EXPECT_TRUE(fromDescription.hasTurboQuant4KvCache());

    auto const fromQuantAlgo = QuantMode::fromQuantAlgo(std::nullopt, "TURBOQUANT4");
    EXPECT_TRUE(fromQuantAlgo.hasTurboQuant4KvCache());
    EXPECT_TRUE(fromQuantAlgo.hasKvCacheQuant());
}

TEST(Quantization, PythonFlagCompatibility)
{
    EXPECT_EQ(QuantMode::w4a8Nvfp4Fp8().value(), QuantMode::BaseType(1u) << 14);
    EXPECT_EQ(QuantMode::w4a8Mxfp4Fp8().value(), QuantMode::BaseType(1u) << 15);
    EXPECT_EQ(QuantMode::w4a8Mxfp4Mxfp8().value(), QuantMode::BaseType(1u) << 16);
    EXPECT_EQ(QuantMode::w4a16Mxfp4().value(), QuantMode::BaseType(1u) << 17);
    EXPECT_EQ(QuantMode::turboQuant4KvCache().value(), QuantMode::BaseType(1u) << 18);

    auto const fromQuantAlgo = QuantMode::fromQuantAlgo("W4A8_NVFP4_FP8", std::nullopt);
    EXPECT_TRUE(fromQuantAlgo.hasW4a8Nvfp4Fp8());

    auto const fromDescription = QuantMode::fromDescription(false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, true);
    EXPECT_TRUE(fromDescription.hasW4a8Nvfp4Fp8());
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
