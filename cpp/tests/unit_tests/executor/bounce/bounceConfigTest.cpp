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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceConfig.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{
// Sets a group of env vars for one test and restores the previous values on destruction, so
// tests don't leak state into each other (or into a developer's shell-provided environment).
class ScopedEnv
{
public:
    void set(char const* name, char const* value)
    {
        char const* old = std::getenv(name);
        mSaved.emplace_back(name, old != nullptr ? std::optional<std::string>{old} : std::nullopt);
        ::setenv(name, value, /*overwrite=*/1);
    }

    ~ScopedEnv()
    {
        for (auto const& [name, old] : mSaved)
        {
            if (old.has_value())
            {
                ::setenv(name.c_str(), old->c_str(), 1);
            }
            else
            {
                ::unsetenv(name.c_str());
            }
        }
    }

private:
    std::vector<std::pair<std::string, std::optional<std::string>>> mSaved;
};

std::size_t arenaBytesFor(char const* value)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_BYTES", value);
    return b::BounceConfig::fromEnv().arenaBytes;
}
} // namespace

TEST(BounceConfig, ByteSuffixesParse)
{
    EXPECT_EQ(arenaBytesFor("12345"), 12345u);
    EXPECT_EQ(arenaBytesFor("12345B"), 12345u);
    EXPECT_EQ(arenaBytesFor("512K"), 512ULL << 10);
    EXPECT_EQ(arenaBytesFor("512KB"), 512ULL << 10);
    EXPECT_EQ(arenaBytesFor("512kib"), 512ULL << 10);
    EXPECT_EQ(arenaBytesFor("256M"), 256ULL << 20);
    EXPECT_EQ(arenaBytesFor("256MB"), 256ULL << 20);
    EXPECT_EQ(arenaBytesFor("256mb"), 256ULL << 20);
    EXPECT_EQ(arenaBytesFor("256MiB"), 256ULL << 20);
    EXPECT_EQ(arenaBytesFor("1G"), 1ULL << 30);
    EXPECT_EQ(arenaBytesFor("1GB"), 1ULL << 30);
    EXPECT_EQ(arenaBytesFor("1gb"), 1ULL << 30);
    EXPECT_EQ(arenaBytesFor("2GiB"), 2ULL << 30);
}

TEST(BounceConfig, ByteGarbageFallsBackToDefault)
{
    std::size_t const def = b::BounceConfig{}.arenaBytes;
    EXPECT_EQ(arenaBytesFor("abc"), def);           // no digits
    EXPECT_EQ(arenaBytesFor("256XB"), def);         // unknown suffix
    EXPECT_EQ(arenaBytesFor("256 MB"), def);        // space before suffix
    EXPECT_EQ(arenaBytesFor("256MBx"), def);        // trailing junk
    EXPECT_EQ(arenaBytesFor(""), def);              // empty -> default
    EXPECT_EQ(arenaBytesFor("999999999999G"), def); // multiply would overflow u64
}

TEST(BounceConfig, AllByteVarsAcceptSuffix)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_BYTES", "1GB");
    env.set("TRTLLM_NIXL_BOUNCE_MIN_BLOCK", "2mb");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_BYTES", "64MB");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_AVG", "32kb");
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.arenaBytes, 1ULL << 30);
    EXPECT_EQ(cfg.minBlock, 2ULL << 20);
    EXPECT_EQ(cfg.maxChunkBytes, 64ULL << 20);
    EXPECT_EQ(cfg.maxAvgDescBytes, 32ULL << 10);
}

TEST(BounceConfig, PlainCountsRejectSuffix)
{
    // Non-byte vars keep strict integer parsing: a suffix is garbage -> default.
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_DEPTH", "4MB");
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.windowDepth, b::BounceConfig{}.windowDepth);
}
