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

#include <gtest/gtest.h>

#include <climits>

#include "tensorrt_llm/kernels/fmhaDispatcher.h"

namespace
{

using tensorrt_llm::kernels::ContextAttentionMaskType;
using tensorrt_llm::kernels::TrtllmGenAttentionMaskType;
using tensorrt_llm::kernels::detail::selectTrtllmGenMask;

TEST(FmhaDispatcherTest, SelectsVariableWindowWithoutOverrides)
{
    auto const selection = selectTrtllmGenMask(ContextAttentionMaskType::VARIABLE_WINDOW, 4, 2, 8);

    EXPECT_EQ(selection.maskType, TrtllmGenAttentionMaskType::VariableWindow);
    EXPECT_EQ(selection.leftSlidingWindow, -1);
    EXPECT_EQ(selection.rightSlidingWindow, -1);
    EXPECT_EQ(selection.chunkedAttentionSize, 0);
}

TEST(FmhaDispatcherTest, SelectsSlidingWindow)
{
    auto const selection = selectTrtllmGenMask(ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL, 4, INT_MAX, 8);

    EXPECT_EQ(selection.maskType, TrtllmGenAttentionMaskType::SlidingOrChunkedCausal);
    EXPECT_EQ(selection.leftSlidingWindow, 3);
    EXPECT_EQ(selection.rightSlidingWindow, 0);
    EXPECT_EQ(selection.chunkedAttentionSize, 0);
}

TEST(FmhaDispatcherTest, SelectsCausalForDenseEquivalentSlidingWindow)
{
    auto const selection = selectTrtllmGenMask(ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL, 8, INT_MAX, 8);

    EXPECT_EQ(selection.maskType, TrtllmGenAttentionMaskType::Causal);
    EXPECT_EQ(selection.leftSlidingWindow, -1);
    EXPECT_EQ(selection.rightSlidingWindow, -1);
    EXPECT_EQ(selection.chunkedAttentionSize, 0);
}

TEST(FmhaDispatcherTest, SelectsChunkedAttention)
{
    auto const selection = selectTrtllmGenMask(ContextAttentionMaskType::CAUSAL, 0, 4, 8);

    EXPECT_EQ(selection.maskType, TrtllmGenAttentionMaskType::SlidingOrChunkedCausal);
    EXPECT_EQ(selection.leftSlidingWindow, -1);
    EXPECT_EQ(selection.rightSlidingWindow, -1);
    EXPECT_EQ(selection.chunkedAttentionSize, 4);
}

TEST(FmhaDispatcherTest, RejectsSimultaneousSlidingAndChunkedAttention)
{
    EXPECT_ANY_THROW(selectTrtllmGenMask(ContextAttentionMaskType::CAUSAL, 4, 2, 8));
}

} // namespace
