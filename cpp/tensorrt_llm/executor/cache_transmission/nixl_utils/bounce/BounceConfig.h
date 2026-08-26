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

#pragma once

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ---------------------------------------------------------------------------
// Reusable string parsers, shared by the TRTLLM_NIXL_BOUNCE_* env fallback and
// the CacheTransceiverConfig agent_bounce_params dict. All are defensive: a
// value that fails to parse yields nullopt so the caller can keep its current
// (default or env-resolved) value and warn instead of aborting or silently
// producing 0 (a 0 would later trip TLLM_CHECKs, e.g. in BounceTransferPlan).
// ---------------------------------------------------------------------------

/// Case-insensitive: 0/false/no/off -> false, 1/true/yes/on -> true, anything else -> nullopt.
[[nodiscard]] inline std::optional<bool> parseBoolValue(std::string const& raw)
{
    std::string s;
    s.reserve(raw.size());
    for (char c : raw)
    {
        s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (s == "0" || s == "false" || s == "no" || s == "off")
    {
        return false;
    }
    if (s == "1" || s == "true" || s == "yes" || s == "on")
    {
        return true;
    }
    return std::nullopt;
}

/// Strict unsigned decimal integer: garbage (typo like "abc"), trailing junk, or overflow -> nullopt.
[[nodiscard]] inline std::optional<std::uint64_t> parseU64Value(std::string const& raw)
{
    if (raw.empty() || !std::isdigit(static_cast<unsigned char>(raw[0])))
    {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    std::uint64_t const parsed = std::strtoull(raw.c_str(), &end, 10);
    if (errno == ERANGE || end == raw.c_str() || *end != '\0')
    {
        return std::nullopt;
    }
    return parsed;
}

/// Byte size: unsigned integer with an optional binary suffix — K/KB/KiB, M/MB/MiB, G/GB/GiB
/// (case-insensitive, no space), e.g. "256MB", "1gb", "512kib". All suffixes are powers of two
/// (MB == MiB == 2^20). Bare numbers and a trailing "B" stay bytes. Unknown suffix, trailing
/// junk, or multiplication overflow -> nullopt.
[[nodiscard]] inline std::optional<std::uint64_t> parseBytesValue(std::string const& raw)
{
    if (raw.empty() || !std::isdigit(static_cast<unsigned char>(raw[0])))
    {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    std::uint64_t const parsed = std::strtoull(raw.c_str(), &end, 10);
    if (errno == ERANGE || end == raw.c_str())
    {
        return std::nullopt;
    }
    std::string suffix;
    for (char const* p = end; *p != '\0'; ++p)
    {
        suffix.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
    }
    std::uint64_t mult = 1;
    if (suffix.empty() || suffix == "b")
    {
        mult = 1;
    }
    else if (suffix == "k" || suffix == "kb" || suffix == "kib")
    {
        mult = 1ULL << 10;
    }
    else if (suffix == "m" || suffix == "mb" || suffix == "mib")
    {
        mult = 1ULL << 20;
    }
    else if (suffix == "g" || suffix == "gb" || suffix == "gib")
    {
        mult = 1ULL << 30;
    }
    else
    {
        return std::nullopt; // unknown suffix
    }
    if (mult > 1 && parsed > std::numeric_limits<std::uint64_t>::max() / mult)
    {
        return std::nullopt; // would overflow
    }
    return parsed * mult;
}

/// POD config for the bounce v2 pipeline. There is no `enabled` field: the on/off switch is
/// CacheTransceiverConfig's agent_bounce_buffer_enable + kv_cache_bounce_size_mb (which the Python
/// frontend folds into the agent's arena size in MiB: 0 = off, >0 = arena size), and at runtime
/// "bounce on" simply means the owning agent built its bounce state (mBounce != nullptr).
/// `arenaSizeBytes` is NOT env-backed either — the owning agent sets it from the same knob. The
/// expert knobs resolve as: agent_bounce_params dict > TRTLLM_NIXL_BOUNCE_* env var > built-in
/// default (`fromEnv()` reads the env, `fromParams()` layers the dict on top). Byte-valued knobs
/// accept an optional case-insensitive binary suffix such as "256MB", "1gb", or "512KiB"
/// (K/M/G == KiB/MiB/GiB, powers of two).
struct BounceConfig
{
    // Overwritten with the agent arena size (MiB << 20, derived from CacheTransceiverConfig's
    // kv_cache_bounce_size_mb + agent_bounce_buffer_enable) on every production path
    // (maybeInitBounce); the
    // default only serves unit tests that construct a BounceConfig{} directly.
    std::size_t arenaSizeBytes{512ULL << 20};
    std::size_t arenaAllocationGranularityBytes{1ULL << 20}; // arena_allocation_granularity
    std::size_t maxChunkSizeBytes{32ULL << 20};              // max_chunk_size
    std::uint32_t maxInflightChunksPerRequest{8};            // max_inflight_chunks_per_request
    std::uint32_t copyStreamCount{8};                        // copy_stream_count
    std::uint32_t scatterWorkerCount{4};                     // scatter_worker_count
    std::size_t minDescriptorCount{1024};                    // min_descriptor_count
    std::size_t maxAverageDescriptorSizeBytes{16ULL << 10};  // max_average_descriptor_size
    int requestTimeoutMs{30000}; // request_timeout_ms; must be > 0 — the whole failure model
                                 // (abandoned-flow resolution, receiver lease, quarantine) hangs off
                                 // this timer, so applyParam rejects 0 (negatives already fail the
                                 // strict unsigned parse) and keeps the current value.
    // Receiver-side lease on granted regions — DERIVED in deriveDependentTimeouts() as
    // 2 x requestTimeoutMs, not an independent knob (tests may still set the field directly). A
    // dead sender emits neither DATA nor a cancel, which is unobservable through the protocol
    // alone — so a flow whose grants see no progress (no GRANT sent, no DATA received) for this
    // long is reclaimed and its regions quarantined (below) before reuse. The lease must EXCEED
    // the peers' requestTimeoutMs (a live sender abandons + cancels first, so only
    // dead/unreachable peers ever hit this) — the 2x derivation assumes both ends run the SAME
    // request_timeout_ms.
    int receiverFlowTimeoutMs{60000};
    // How long a receiver-reclaimed, possibly-still-being-written region stays out of the arena
    // before reuse — DERIVED in deriveDependentTimeouts() as requestTimeoutMs. A one-sided RDMA
    // write cannot be aborted, so time is the only barrier against re-granting a region a gone
    // peer's NIC may still be writing.
    int quarantineMs{30000};
    bool disableFabricMemory{false}; // disable_fabric_memory
    // enable_eager_gather: launch a chunk's gather at submit() time, before the receiver's GRANT
    // arrives, overlapping the WANT->GRANT control round-trip with the gather kernel. Eager
    // (credit-less) staging regions are capped at HALF the arena so that on a bidirectional
    // deployment both sides can always still grant incoming regions (no mutual eager-starvation);
    // the credit-backed path is unaffected by the cap.
    bool enableEagerGather{true};
    // use_zero_copy_arguments: the copy kernel reads [srcs|dsts|sizes] directly from pinned host
    // memory instead of staging them in device scratch first. Faster at every plan size (same
    // bytes over the bus, but no H2D-then-kernel serialization), so on by default.
    bool useZeroCopyArguments{true};

    /// Re-derive the lease/quarantine values from the one user-visible timeout (see the field
    /// comments): the lease must exceed the peers' request timeout, and time is the only write
    /// barrier for a reclaimed region. The config layer guarantees requestTimeoutMs > 0 (applyParam
    /// rejects 0); the > 0 guard below only matters for directly-constructed configs. Must be
    /// re-run whenever requestTimeoutMs changes: fromEnv() always calls it, fromParams() only when
    /// the dict actually provides request_timeout_ms — so a caller-tweaked receiverFlowTimeoutMs /
    /// quarantineMs on the base config (tests set the fields directly) survives an unrelated dict.
    /// The doubling is done in 64-bit and clamped: requestTimeoutMs admits values up to INT_MAX,
    /// and a signed overflow here would wrap the lease negative — silently disabling the
    /// dead-sender reclaim it drives.
    void deriveDependentTimeouts()
    {
        receiverFlowTimeoutMs = requestTimeoutMs > 0 ? static_cast<int>(std::min<std::int64_t>(
                                    std::int64_t{2} * requestTimeoutMs, std::numeric_limits<int>::max()))
                                                     : 0;
        quarantineMs = requestTimeoutMs;
    }

    /// Apply one knob given as a (key, value) string pair. Returns false when the key is unknown.
    /// A value that fails to parse or is out of range keeps the current field value and warns
    /// (`origin` names the source — the env var or the config dict — for the log line).
    static bool applyParam(BounceConfig& cfg, std::string const& key, std::string const& value, char const* origin)
    {
        auto warnBad = [&]
        {
            TLLM_LOG_WARNING("BounceConfig: invalid value '%s' for '%s' (from %s) -> keeping the current value",
                value.c_str(), key.c_str(), origin);
        };
        auto setBool = [&](bool& field)
        {
            if (auto v = parseBoolValue(value))
            {
                field = *v;
            }
            else
            {
                warnBad();
            }
        };
        auto setSizeBytes = [&](std::size_t& field, bool allowZero)
        {
            auto const v = parseBytesValue(value);
            // The size_t bound only bites on 32-bit size_t builds; it is trivially true on 64-bit.
            if (v.has_value() && (allowZero || *v != 0) && *v <= std::numeric_limits<std::size_t>::max())
            {
                field = static_cast<std::size_t>(*v);
            }
            else
            {
                warnBad();
            }
        };
        auto setSize = [&](std::size_t& field, bool allowZero)
        {
            auto const v = parseU64Value(value);
            // The size_t bound only bites on 32-bit size_t builds; it is trivially true on 64-bit.
            if (v.has_value() && (allowZero || *v != 0) && *v <= std::numeric_limits<std::size_t>::max())
            {
                field = static_cast<std::size_t>(*v);
            }
            else
            {
                warnBad();
            }
        };
        auto setU32 = [&](std::uint32_t& field, bool allowZero)
        {
            auto const v = parseU64Value(value);
            if (v.has_value() && (allowZero || *v != 0) && *v <= std::numeric_limits<std::uint32_t>::max())
            {
                field = static_cast<std::uint32_t>(*v);
            }
            else
            {
                warnBad();
            }
        };
        auto setPositiveInt = [&](int& field)
        {
            auto const v = parseU64Value(value);
            if (v.has_value() && *v != 0 && *v <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            {
                field = static_cast<int>(*v);
            }
            else
            {
                warnBad();
            }
        };

        if (key == "max_chunk_size")
        {
            setSizeBytes(cfg.maxChunkSizeBytes, /*allowZero=*/false);
        }
        else if (key == "arena_allocation_granularity")
        {
            setSizeBytes(cfg.arenaAllocationGranularityBytes, /*allowZero=*/false);
        }
        else if (key == "max_average_descriptor_size")
        {
            setSizeBytes(cfg.maxAverageDescriptorSizeBytes, /*allowZero=*/true);
        }
        else if (key == "max_inflight_chunks_per_request")
        {
            setU32(cfg.maxInflightChunksPerRequest, /*allowZero=*/false);
        }
        else if (key == "copy_stream_count")
        {
            setU32(cfg.copyStreamCount, /*allowZero=*/false);
        }
        else if (key == "scatter_worker_count")
        {
            setU32(cfg.scatterWorkerCount, /*allowZero=*/false);
        }
        else if (key == "min_descriptor_count")
        {
            setSize(cfg.minDescriptorCount, /*allowZero=*/true);
        }
        else if (key == "request_timeout_ms")
        {
            setPositiveInt(cfg.requestTimeoutMs);
        }
        else if (key == "disable_fabric_memory")
        {
            setBool(cfg.disableFabricMemory);
        }
        else if (key == "enable_eager_gather")
        {
            setBool(cfg.enableEagerGather);
        }
        else if (key == "use_zero_copy_arguments")
        {
            setBool(cfg.useZeroCopyArguments);
        }
        else
        {
            return false;
        }
        return true;
    }

    /// Defaults overridden by any set TRTLLM_NIXL_BOUNCE_* env var. Each call reads the current
    /// environment. The on/off switch and the arena size are NOT read here — they only come from
    /// CacheTransceiverConfig (agent_bounce_buffer_enable + kv_cache_bounce_size_mb).
    [[nodiscard]] static BounceConfig fromEnv()
    {
        // paramKey -> env var name. Byte-valued knobs keep the historical _BYTES env suffix.
        // KEEP IN SYNC with AGENT_BOUNCE_PARAM_KEYS in
        // tensorrt_llm/_torch/disaggregation/nixl/bounce_knobs.py, which the llm_args validator
        // uses to reject unknown agent_bounce_params keys upfront.
        static constexpr std::array<std::pair<char const*, char const*>, 11> kEnvKnobs{{
            {"max_chunk_size", "TRTLLM_NIXL_BOUNCE_MAX_CHUNK_SIZE_BYTES"},
            {"arena_allocation_granularity", "TRTLLM_NIXL_BOUNCE_ARENA_ALLOCATION_GRANULARITY_BYTES"},
            {"max_average_descriptor_size", "TRTLLM_NIXL_BOUNCE_MAX_AVERAGE_DESCRIPTOR_SIZE_BYTES"},
            {"max_inflight_chunks_per_request", "TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST"},
            {"copy_stream_count", "TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT"},
            {"scatter_worker_count", "TRTLLM_NIXL_BOUNCE_SCATTER_WORKER_COUNT"},
            {"min_descriptor_count", "TRTLLM_NIXL_BOUNCE_MIN_DESCRIPTOR_COUNT"},
            {"request_timeout_ms", "TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS"},
            {"disable_fabric_memory", "TRTLLM_NIXL_BOUNCE_DISABLE_FABRIC_MEMORY"},
            {"enable_eager_gather", "TRTLLM_NIXL_BOUNCE_ENABLE_EAGER_GATHER"},
            {"use_zero_copy_arguments", "TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS"},
        }};
        BounceConfig cfg;
        for (auto const& [paramKey, envName] : kEnvKnobs)
        {
            char const* v = std::getenv(envName);
            if (v != nullptr && v[0] != '\0') // unset or empty -> keep the default
            {
                applyParam(cfg, paramKey, v, envName);
            }
        }
        cfg.deriveDependentTimeouts();
        return cfg;
    }

    /// Layer the CacheTransceiverConfig agent_bounce_params dict over `base` (normally the
    /// fromEnv() result, so dict > env > default). Unknown keys warn and are skipped (defensive
    /// backstop — the llm_args validator already rejects them at the Python boundary, but other
    /// entry points can reach this directly). The lease/quarantine values are re-derived only
    /// when the dict provides a request_timeout_ms that actually parses (a rejected value keeps
    /// the base timeout, so re-deriving would only clobber directly-set values on `base`).
    [[nodiscard]] static BounceConfig fromParams(
        std::unordered_map<std::string, std::string> const& params, BounceConfig base)
    {
        for (auto const& [key, value] : params)
        {
            if (!applyParam(base, key, value, "agent_bounce_params"))
            {
                TLLM_LOG_WARNING("BounceConfig: unknown agent_bounce_params key '%s' -> ignored", key.c_str());
            }
        }
        // Same validity check as applyParam's request_timeout_ms setter: only a value that was
        // actually accepted triggers the re-derivation.
        auto const it = params.find("request_timeout_ms");
        if (it != params.end())
        {
            auto const parsed = parseU64Value(it->second);
            if (parsed.has_value() && *parsed != 0
                && *parsed <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            {
                base.deriveDependentTimeouts();
            }
        }
        return base;
    }
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
