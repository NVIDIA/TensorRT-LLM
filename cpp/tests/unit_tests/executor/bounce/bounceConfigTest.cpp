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
#include <unordered_map>
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

// Resolve one byte-valued knob through the params (dict) path.
std::size_t maxChunkSizeBytesForParam(char const* value)
{
    std::unordered_map<std::string, std::string> const params{{"max_chunk_size", value}};
    return b::BounceConfig::fromParams(params, b::BounceConfig{}).maxChunkSizeBytes;
}
} // namespace

// The on/off switch and the arena size come ONLY from CacheTransceiverConfig
// (agent_bounce_buffer_enable + kv_cache_bounce_size_mb);
// the legacy TRTLLM_NIXL_BOUNCE_ENABLE / TRTLLM_NIXL_BOUNCE_ARENA_SIZE_BYTES env vars are dead.
TEST(BounceConfig, EnableAndArenaAreNotEnvBacked)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ENABLE", "1");
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_SIZE_BYTES", "1GB");
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.arenaSizeBytes, b::BounceConfig{}.arenaSizeBytes);
}

TEST(BounceConfig, ByteSuffixesParse)
{
    EXPECT_EQ(maxChunkSizeBytesForParam("12345"), 12345u);
    EXPECT_EQ(maxChunkSizeBytesForParam("12345B"), 12345u);
    EXPECT_EQ(maxChunkSizeBytesForParam("512K"), 512ULL << 10);
    EXPECT_EQ(maxChunkSizeBytesForParam("512KB"), 512ULL << 10);
    EXPECT_EQ(maxChunkSizeBytesForParam("512kib"), 512ULL << 10);
    EXPECT_EQ(maxChunkSizeBytesForParam("256M"), 256ULL << 20);
    EXPECT_EQ(maxChunkSizeBytesForParam("256MB"), 256ULL << 20);
    EXPECT_EQ(maxChunkSizeBytesForParam("256mb"), 256ULL << 20);
    EXPECT_EQ(maxChunkSizeBytesForParam("256MiB"), 256ULL << 20);
    EXPECT_EQ(maxChunkSizeBytesForParam("1G"), 1ULL << 30);
    EXPECT_EQ(maxChunkSizeBytesForParam("1GB"), 1ULL << 30);
    EXPECT_EQ(maxChunkSizeBytesForParam("1gb"), 1ULL << 30);
    EXPECT_EQ(maxChunkSizeBytesForParam("2GiB"), 2ULL << 30);
}

TEST(BounceConfig, ByteGarbageFallsBackToDefault)
{
    std::size_t const def = b::BounceConfig{}.maxChunkSizeBytes;
    EXPECT_EQ(maxChunkSizeBytesForParam("abc"), def);           // no digits
    EXPECT_EQ(maxChunkSizeBytesForParam("256XB"), def);         // unknown suffix
    EXPECT_EQ(maxChunkSizeBytesForParam("256 MB"), def);        // space before suffix
    EXPECT_EQ(maxChunkSizeBytesForParam("256MBx"), def);        // trailing junk
    EXPECT_EQ(maxChunkSizeBytesForParam(""), def);              // empty -> default
    EXPECT_EQ(maxChunkSizeBytesForParam("999999999999G"), def); // multiply would overflow u64
    EXPECT_EQ(maxChunkSizeBytesForParam("0"), def);             // zero chunk cap is not usable
}

TEST(BounceConfig, AllByteVarsAcceptSuffixFromEnv)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_ARENA_ALLOCATION_GRANULARITY_BYTES", "2mb");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_SIZE_BYTES", "64MB");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_AVERAGE_DESCRIPTOR_SIZE_BYTES", "32kb");
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.arenaAllocationGranularityBytes, 2ULL << 20);
    EXPECT_EQ(cfg.maxChunkSizeBytes, 64ULL << 20);
    EXPECT_EQ(cfg.maxAverageDescriptorSizeBytes, 32ULL << 10);
}

TEST(BounceConfig, PlainCountsRejectSuffix)
{
    // Non-byte knobs keep strict integer parsing: a suffix is garbage -> default. Same rule on
    // both the env and the params path.
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST", "4MB");
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.maxInflightChunksPerRequest, b::BounceConfig{}.maxInflightChunksPerRequest);

    std::unordered_map<std::string, std::string> const params{{"max_inflight_chunks_per_request", "4MB"}};
    auto const cfgP = b::BounceConfig::fromParams(params, b::BounceConfig{});
    EXPECT_EQ(cfgP.maxInflightChunksPerRequest, b::BounceConfig{}.maxInflightChunksPerRequest);
}

TEST(BounceConfig, DescriptiveNamesParse)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST", "5");
    env.set("TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT", "6");
    env.set("TRTLLM_NIXL_BOUNCE_SCATTER_WORKER_COUNT", "7");
    env.set("TRTLLM_NIXL_BOUNCE_MIN_DESCRIPTOR_COUNT", "8");
    env.set("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", "900");
    env.set("TRTLLM_NIXL_BOUNCE_DISABLE_FABRIC_MEMORY", "yes");
    env.set("TRTLLM_NIXL_BOUNCE_ENABLE_EAGER_GATHER", "false");
    env.set("TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS", "false");

    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.maxInflightChunksPerRequest, 5u);
    EXPECT_EQ(cfg.copyStreamCount, 6u);
    EXPECT_EQ(cfg.scatterWorkerCount, 7u);
    EXPECT_EQ(cfg.minDescriptorCount, 8u);
    EXPECT_EQ(cfg.requestTimeoutMs, 900);
    // Lease/quarantine are derived from the request timeout, not independent knobs.
    EXPECT_EQ(cfg.receiverFlowTimeoutMs, 1800);
    EXPECT_EQ(cfg.quarantineMs, 900);
    EXPECT_TRUE(cfg.disableFabricMemory);
    EXPECT_FALSE(cfg.enableEagerGather);
    EXPECT_FALSE(cfg.useZeroCopyArguments);
}

TEST(BounceConfig, InvalidResourceCountsFallBackToDefaults)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT", "0");
    env.set("TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST", "4294967296");
    env.set("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", "2147483648");

    auto const defaults = b::BounceConfig{};
    auto const cfg = b::BounceConfig::fromEnv();
    EXPECT_EQ(cfg.copyStreamCount, defaults.copyStreamCount);
    EXPECT_EQ(cfg.maxInflightChunksPerRequest, defaults.maxInflightChunksPerRequest);
    EXPECT_EQ(cfg.requestTimeoutMs, defaults.requestTimeoutMs);
}

// The timeout parser admits any value up to INT_MAX, so the 2x lease derivation must clamp instead
// of overflowing: a wrapped-negative receiverFlowTimeoutMs reads as "lease disabled" and silently
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

// agent_bounce_params dict > env var > default: a key present in both takes the dict value; keys
// only in the env keep the env value; everything else keeps the default.
TEST(BounceConfig, ParamsOverrideEnv)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT", "6");
    env.set("TRTLLM_NIXL_BOUNCE_SCATTER_WORKER_COUNT", "7");

    std::unordered_map<std::string, std::string> const params{
        {"copy_stream_count", "3"}, {"max_chunk_size", "64MB"}, // suffix parsing works on the params path too
    };
    auto const cfg = b::BounceConfig::fromParams(params, b::BounceConfig::fromEnv());
    EXPECT_EQ(cfg.copyStreamCount, 3u);                 // dict wins over env
    EXPECT_EQ(cfg.scatterWorkerCount, 7u);              // env-only knob keeps env value
    EXPECT_EQ(cfg.maxChunkSizeBytes, 64ULL << 20);      // dict-only knob applies
    EXPECT_EQ(cfg.maxInflightChunksPerRequest,          // untouched knob keeps the default
        b::BounceConfig{}.maxInflightChunksPerRequest); //
}

// A dict value that fails to parse keeps the env/base value instead of clobbering it. "0" counts
// as unparsable for allowZero=false knobs (arena_allocation_granularity) but is a valid value for
// allowZero=true ones (max_average_descriptor_size: a zero gate routes everything to standard NIXL).
TEST(BounceConfig, ParamsGarbageKeepsBaseValue)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT", "6");
    std::unordered_map<std::string, std::string> const params{
        {"copy_stream_count", "banana"},
        {"arena_allocation_granularity", "0"},
        {"max_average_descriptor_size", "0"},
    };
    auto const cfg = b::BounceConfig::fromParams(params, b::BounceConfig::fromEnv());
    EXPECT_EQ(cfg.copyStreamCount, 6u);
    EXPECT_EQ(cfg.arenaAllocationGranularityBytes, b::BounceConfig{}.arenaAllocationGranularityBytes);
    EXPECT_EQ(cfg.maxAverageDescriptorSizeBytes, 0u);
}

// Unknown keys are skipped with a warning (must not crash or affect known knobs).
TEST(BounceConfig, UnknownParamKeyIsIgnored)
{
    std::unordered_map<std::string, std::string> const params{
        {"definitely_not_a_bounce_knob", "42"},
        {"copy_stream_count", "3"},
    };
    auto const cfg = b::BounceConfig::fromParams(params, b::BounceConfig{});
    EXPECT_EQ(cfg.copyStreamCount, 3u);
    EXPECT_EQ(cfg.maxChunkSizeBytes, b::BounceConfig{}.maxChunkSizeBytes);
}

// A dict-provided request_timeout_ms must re-derive the receiver lease and quarantine values
// (otherwise the lease would keep the env/default-based derivation and break its "must exceed the
// peers' request timeout" invariant).
TEST(BounceConfig, ParamsRequestTimeoutRederivesLease)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", "900");

    std::unordered_map<std::string, std::string> const params{{"request_timeout_ms", "1000"}};
    auto const cfg = b::BounceConfig::fromParams(params, b::BounceConfig::fromEnv());
    EXPECT_EQ(cfg.requestTimeoutMs, 1000);
    EXPECT_EQ(cfg.receiverFlowTimeoutMs, 2000);
    EXPECT_EQ(cfg.quarantineMs, 1000);

    // The 64-bit clamp holds on the params path too.
    std::unordered_map<std::string, std::string> const bigParams{{"request_timeout_ms", "2000000000"}};
    auto const bigCfg = b::BounceConfig::fromParams(bigParams, b::BounceConfig::fromEnv());
    EXPECT_EQ(bigCfg.requestTimeoutMs, 2000000000);
    EXPECT_EQ(bigCfg.receiverFlowTimeoutMs, std::numeric_limits<int>::max());
    EXPECT_EQ(bigCfg.quarantineMs, 2000000000);

    // Edge values: "0" is accepted (timeout disabled, derived values 0), "-1" is rejected (strict
    // unsigned parsing) and keeps the base value untouched.
    auto const zeroCfg = b::BounceConfig::fromParams({{"request_timeout_ms", "0"}}, b::BounceConfig{});
    EXPECT_EQ(zeroCfg.requestTimeoutMs, 0);
    EXPECT_EQ(zeroCfg.receiverFlowTimeoutMs, 0);
    EXPECT_EQ(zeroCfg.quarantineMs, 0);
    auto const negCfg = b::BounceConfig::fromParams({{"request_timeout_ms", "-1"}}, b::BounceConfig{});
    EXPECT_EQ(negCfg.requestTimeoutMs, b::BounceConfig{}.requestTimeoutMs);
    EXPECT_EQ(negCfg.receiverFlowTimeoutMs, b::BounceConfig{}.receiverFlowTimeoutMs);
}

// A dict WITHOUT request_timeout_ms must not re-derive: an env-derived lease survives, and
// directly-set lease/quarantine values on the base (as failure-injection tests use) stay intact.
TEST(BounceConfig, ParamsWithoutRequestTimeoutLeaveLeaseAlone)
{
    ScopedEnv env;
    env.set("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", "900");
    std::unordered_map<std::string, std::string> const params{{"copy_stream_count", "3"}};
    auto const cfg = b::BounceConfig::fromParams(params, b::BounceConfig::fromEnv());
    EXPECT_EQ(cfg.copyStreamCount, 3u);
    EXPECT_EQ(cfg.requestTimeoutMs, 900);
    EXPECT_EQ(cfg.receiverFlowTimeoutMs, 1800);
    EXPECT_EQ(cfg.quarantineMs, 900);

    b::BounceConfig base;
    base.receiverFlowTimeoutMs = 123456;
    base.quarantineMs = 654321;
    auto const cfgDirect = b::BounceConfig::fromParams(params, base);
    EXPECT_EQ(cfgDirect.receiverFlowTimeoutMs, 123456);
    EXPECT_EQ(cfgDirect.quarantineMs, 654321);
}
