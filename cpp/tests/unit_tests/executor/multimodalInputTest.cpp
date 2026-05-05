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

#include "tensorrt_llm/executor/executor.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace tle = tensorrt_llm::executor;

namespace
{
std::vector<std::vector<tle::SizeType32>> makeHashes(size_t numItems)
{
    std::vector<std::vector<tle::SizeType32>> hashes;
    hashes.reserve(numItems);
    for (size_t itemIdx = 0; itemIdx < numItems; ++itemIdx)
    {
        std::vector<tle::SizeType32> hash;
        hash.reserve(8);
        for (tle::SizeType32 partIdx = 0; partIdx < 8; ++partIdx)
        {
            hash.push_back(static_cast<tle::SizeType32>(itemIdx * 16 + partIdx));
        }
        hashes.push_back(std::move(hash));
    }
    return hashes;
}
} // namespace

TEST(MultimodalInputTest, CreateAcceptsValidSparseItemRuns)
{
    auto input = tle::MultimodalInput(makeHashes(2),
        {{tle::MultimodalItemRun{1, 1, {}}, tle::MultimodalItemRun{4, 2, {1}}}, {tle::MultimodalItemRun{8, 3, {0}}}},
        std::vector<std::optional<std::string>>{std::string{"image-0"}, std::nullopt}, 12);

    ASSERT_EQ(input.getMultimodalItemRuns().size(), 2);
    EXPECT_EQ(input.getMultimodalItemRuns()[0][0].promptStart, 1);
    EXPECT_EQ(input.getMultimodalItemRuns()[0][1].runLength, 2);
    EXPECT_EQ(input.getMultimodalItemRuns()[1][0].nonEmbedOffsets, std::vector<tle::SizeType32>{0});
}

TEST(MultimodalInputTest, CreateRejectsInvalidItemRuns)
{
    auto const hashes = makeHashes(1);
    auto makeInput = [&](tle::MultimodalItemRuns itemRuns)
    { return tle::MultimodalInput(hashes, std::move(itemRuns), std::nullopt); };
    auto makeBoundedInput = [&](tle::MultimodalItemRuns itemRuns, tle::SizeType32 promptLen)
    { return tle::MultimodalInput(hashes, std::move(itemRuns), std::nullopt, promptLen); };

    EXPECT_THROW(
        (void) tle::MultimodalInput({{1, 2, 3}}, {{tle::MultimodalItemRun{1, 1, {}}}}, std::nullopt), std::exception);
    EXPECT_NO_THROW((void) tle::MultimodalInput(
        hashes, {{tle::MultimodalItemRun{1, 1, {}}}}, std::vector<std::optional<std::string>>{}));
    EXPECT_THROW((void) tle::MultimodalInput(hashes, {{tle::MultimodalItemRun{1, 1, {}}}},
                     std::vector<std::optional<std::string>>{std::nullopt, std::nullopt}),
        std::exception);
    EXPECT_THROW(
        (void) tle::MultimodalInput(makeHashes(2), {{tle::MultimodalItemRun{1, 1, {}}}}, std::nullopt), std::exception);
    EXPECT_THROW((void) makeInput({{}}), std::exception);
    EXPECT_THROW((void) makeInput({{tle::MultimodalItemRun{1, 0, {}}}}), std::exception);
    EXPECT_THROW((void) makeInput({{tle::MultimodalItemRun{-1, 1, {}}}}), std::exception);
    EXPECT_THROW((void) makeBoundedInput({{tle::MultimodalItemRun{4, 2, {}}}}, 5), std::exception);
    EXPECT_THROW(
        (void) makeInput({{tle::MultimodalItemRun{1, 2, {}}, tle::MultimodalItemRun{2, 1, {}}}}), std::exception);
    EXPECT_THROW((void) tle::MultimodalInput(makeHashes(2),
                     {{tle::MultimodalItemRun{1, 2, {}}}, {tle::MultimodalItemRun{2, 1, {}}}}, std::nullopt),
        std::exception);
    EXPECT_THROW((void) makeInput({{tle::MultimodalItemRun{1, 2, {-1}}}}), std::exception);
    EXPECT_THROW((void) makeInput({{tle::MultimodalItemRun{1, 2, {2}}}}), std::exception);
    EXPECT_THROW((void) makeInput({{tle::MultimodalItemRun{1, 3, {2, 1}}}}), std::exception);
    EXPECT_THROW((void) makeInput({{tle::MultimodalItemRun{1, 3, {1, 1}}}}), std::exception);
}
