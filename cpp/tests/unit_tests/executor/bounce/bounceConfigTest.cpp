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
#include <limits>
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

std::size_t arenaSizeBytesFor(char const* value)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_SIZE_BYTES", value);
    return b::BounceConfig::fromEnv().arenaSizeBytes;
}
} // namespace

TEST(BounceConfig, ByteSuffixesParse)
{
    EXPECT_EQ(arenaSizeBytesFor("12345"), 12345u);
    EXPECT_EQ(arenaSizeBytesFor("12345B"), 12345u);
    EXPECT_EQ(arenaSizeBytesFor("512K"), 512ULL << 10);
    EXPECT_EQ(arenaSizeBytesFor("512KB"), 512ULL << 10);
    EXPECT_EQ(arenaSizeBytesFor("512kib"), 512ULL << 10);
    EXPECT_EQ(arenaSizeBytesFor("256M"), 256ULL << 20);
    EXPECT_EQ(arenaSizeBytesFor("256MB"), 256ULL << 20);
    EXPECT_EQ(arenaSizeBytesFor("256mb"), 256ULL << 20);
    EXPECT_EQ(arenaSizeBytesFor("256MiB"), 256ULL << 20);
    EXPECT_EQ(arenaSizeBytesFor("1G"), 1ULL << 30);
    EXPECT_EQ(arenaSizeBytesFor("1GB"), 1ULL << 30);
    EXPECT_EQ(arenaSizeBytesFor("1gb"), 1ULL << 30);
    EXPECT_EQ(arenaSizeBytesFor("2GiB"), 2ULL << 30);
}

TEST(BounceConfig, ByteGarbageFallsBackToDefault)
{
    std::size_t const def = b::BounceConfig{}.arenaSizeBytes;
    EXPECT_EQ(arenaSizeBytesFor("abc"), def);           // no digits
    EXPECT_EQ(arenaSizeBytesFor("256XB"), def);         // unknown suffix
    EXPECT_EQ(arenaSizeBytesFor("256 MB"), def);        // space before suffix
    EXPECT_EQ(arenaSizeBytesFor("256MBx"), def);        // trailing junk
    EXPECT_EQ(arenaSizeBytesFor(""), def);              // empty -> default
    EXPECT_EQ(arenaSizeBytesFor("999999999999G"), def); // multiply would overflow u64
}

TEST(BounceConfig, AllByteVarsAcceptSuffix)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_SIZE_BYTES", "1GB");
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_ALLOCATION_GRANULARITY_BYTES", "2mb");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_SIZE_BYTES", "64MB");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_AVERAGE_DESCRIPTOR_SIZE_BYTES", "32kb");
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.arenaSizeBytes, 1ULL << 30);
    EXPECT_EQ(cfg.arenaAllocationGranularityBytes, 2ULL << 20);
    EXPECT_EQ(cfg.maxChunkSizeBytes, 64ULL << 20);
    EXPECT_EQ(cfg.maxAverageDescriptorSizeBytes, 32ULL << 10);
}

TEST(BounceConfig, PlainCountsRejectSuffix)
{
    // Non-byte vars keep strict integer parsing: a suffix is garbage -> default.
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST", "4MB");
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.maxInflightChunksPerRequest, b::BounceConfig{}.maxInflightChunksPerRequest);
}

TEST(BounceConfig, DescriptiveNamesParse)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ENABLE", "true");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST", "5");
    env.set("TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT", "6");
    env.set("TRTLLM_NIXL_BOUNCE_SCATTER_WORKER_COUNT", "7");
    env.set("TRTLLM_NIXL_BOUNCE_MIN_DESCRIPTOR_COUNT", "8");
    env.set("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", "900");
    env.set("TRTLLM_NIXL_BOUNCE_DISABLE_FABRIC_MEMORY", "yes");
    env.set("TRTLLM_NIXL_BOUNCE_ENABLE_EAGER_GATHER", "false");
    env.set("TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS", "false");

    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_TRUE(cfg.enabled);
    EXPECT_EQ(cfg.maxInflightChunksPerRequest, 5u);
    EXPECT_EQ(cfg.copyStreamCount, 6u);
    EXPECT_EQ(cfg.scatterWorkerCount, 7u);
    EXPECT_EQ(cfg.minDescriptorCount, 8u);
    EXPECT_EQ(cfg.requestTimeoutMs, 900);
    // Lease/quarantine are derived from the request timeout, not independent env knobs.
    EXPECT_EQ(cfg.receiverFlowTimeoutMs, 1800);
    EXPECT_EQ(cfg.quarantineMs, 900);
    EXPECT_TRUE(cfg.disableFabricMemory);
    EXPECT_FALSE(cfg.enableEagerGather);
    EXPECT_FALSE(cfg.useZeroCopyArguments);
}

TEST(BounceConfig, InvalidResourceCountsFallBackToDefaults)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_SIZE_BYTES", "0");
    env.set("TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT", "0");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST", "4294967296");
    env.set("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", "2147483648");

    auto const defaults = b::BounceConfig{};
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.arenaSizeBytes, defaults.arenaSizeBytes);
    EXPECT_EQ(cfg.copyStreamCount, defaults.copyStreamCount);
    EXPECT_EQ(cfg.maxInflightChunksPerRequest, defaults.maxInflightChunksPerRequest);
    EXPECT_EQ(cfg.requestTimeoutMs, defaults.requestTimeoutMs);
}

// envInt admits any timeout up to INT_MAX, so the 2x lease derivation must clamp instead of
// overflowing: a wrapped-negative receiverFlowTimeoutMs reads as "lease disabled" and silently
// turns off the dead-sender region reclaim.
TEST(BounceConfig, LeaseDerivationClampsInsteadOfOverflowing)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", "2000000000"); // valid, but > INT_MAX / 2

    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.requestTimeoutMs, 2000000000);
    EXPECT_EQ(cfg.receiverFlowTimeoutMs, std::numeric_limits<int>::max());
    EXPECT_EQ(cfg.quarantineMs, 2000000000);
}
