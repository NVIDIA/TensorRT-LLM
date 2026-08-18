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

#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// POD config for the bounce v2 pipeline. Each `fromEnv()` call reads the current
/// `TRTLLM_NIXL_BOUNCE_*` environment.
/// Byte-valued variables accept an optional case-insensitive binary suffix such as "256MB", "1gb",
/// or "512KiB" (K/M/G == KiB/MiB/GiB, powers of two).
struct BounceConfig
{
    bool enabled{false};                                     // TRTLLM_NIXL_BOUNCE_ENABLE
    std::size_t arenaSizeBytes{512ULL << 20};                // TRTLLM_NIXL_BOUNCE_ARENA_SIZE_BYTES
    std::size_t arenaAllocationGranularityBytes{1ULL << 20}; // TRTLLM_NIXL_BOUNCE_ARENA_ALLOCATION_GRANULARITY_BYTES
    std::size_t maxChunkSizeBytes{32ULL << 20};              // TRTLLM_NIXL_BOUNCE_MAX_CHUNK_SIZE_BYTES
    std::uint32_t maxInflightChunksPerRequest{8};            // TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST
    std::uint32_t copyStreamCount{8};                        // TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT
    std::uint32_t scatterWorkerCount{4};                     // TRTLLM_NIXL_BOUNCE_SCATTER_WORKER_COUNT
    std::size_t minDescriptorCount{1024};                    // TRTLLM_NIXL_BOUNCE_MIN_DESCRIPTOR_COUNT
    std::size_t maxAverageDescriptorSizeBytes{16ULL << 10};  // TRTLLM_NIXL_BOUNCE_MAX_AVERAGE_DESCRIPTOR_SIZE_BYTES
    int requestTimeoutMs{30000}; // TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS; <=0 DISABLES the timeout
                                 // (checkTimeouts no-ops; used by tests that intentionally wait)
    // Receiver-side lease on granted regions — DERIVED in fromEnv() as 2 x requestTimeoutMs, not an
    // independent env knob (tests may still set the field directly). A dead sender emits neither
    // DATA nor a cancel, which is unobservable through the protocol alone — so a flow whose grants
    // see no progress (no GRANT sent, no DATA received) for this long is reclaimed and its regions
    // quarantined (below) before reuse. The lease must EXCEED the peers' requestTimeoutMs (a live
    // sender abandons + cancels first, so only dead/unreachable peers ever hit this) — the 2x
    // derivation assumes both ends run the SAME TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS.
    int receiverFlowTimeoutMs{60000};
    // How long a receiver-reclaimed, possibly-still-being-written region stays out of the arena
    // before reuse — DERIVED in fromEnv() as requestTimeoutMs. A one-sided RDMA write cannot be
    // aborted, so time is the only barrier against re-granting a region a gone peer's NIC may
    // still be writing.
    int quarantineMs{30000};
    bool disableFabricMemory{false}; // TRTLLM_NIXL_BOUNCE_DISABLE_FABRIC_MEMORY
    // TRTLLM_NIXL_BOUNCE_ENABLE_EAGER_GATHER: launch a chunk's gather at submit() time, before the
    // receiver's GRANT arrives, overlapping the WANT->GRANT control round-trip with the gather
    // kernel. Eager (credit-less) staging regions are capped at HALF the arena so that on a
    // bidirectional deployment both sides can always still grant incoming regions (no mutual
    // eager-starvation); the credit-backed path is unaffected by the cap.
    bool enableEagerGather{true};
    // TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS: the copy kernel reads [srcs|dsts|sizes] directly
    // from pinned host memory instead of staging them in device scratch first. Faster at every plan
    // size (same bytes over the bus, but no H2D-then-kernel serialization), so on by default.
    bool useZeroCopyArguments{true};

    [[nodiscard]] static BounceConfig fromEnv()
    {
        BounceConfig cfg;
        auto envBool = [](char const* name, bool def) -> bool
        {
            char const* v = std::getenv(name);
            if (v == nullptr || v[0] == '\0')
            {
                return def; // unset or empty -> default (don't treat "" as enabled)
            }
            // Case-insensitive: 0/false/no/off -> false, 1/true/yes/on -> true, anything else -> def.
            std::string s;
            for (char const* p = v; *p != '\0'; ++p)
            {
                s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
            }
            if (s == "0" || s == "false" || s == "no" || s == "off")
            {
                return false;
            }
            if (s == "1" || s == "true" || s == "yes" || s == "on")
            {
                return true;
            }
            return def;
        };
        auto envU64 = [](char const* name, std::uint64_t def) -> std::uint64_t
        {
            char const* v = std::getenv(name);
            if (v == nullptr || !std::isdigit(static_cast<unsigned char>(v[0])))
            {
                return def;
            }
            // Parse strictly: a garbage value (typo like "abc", or trailing junk) falls back to the
            // default instead of yielding 0 — a 0 here would later abort the process (e.g.
            // maxChunkSizeBytes=0 trips a TLLM_CHECK in BounceTransferPlan::build).
            char* end = nullptr;
            errno = 0;
            std::uint64_t const parsed = std::strtoull(v, &end, 10);
            if (errno == ERANGE || end == v || *end != '\0')
            {
                return def;
            }
            return parsed;
        };
        // Byte sizes additionally accept a binary suffix — K/KB/KiB, M/MB/MiB, G/GB/GiB
        // (case-insensitive, no space), e.g. "256MB", "1gb", "512kib". All suffixes are
        // powers of two (MB == MiB == 2^20). Bare numbers and a trailing "B" stay bytes.
        auto envBytes = [](char const* name, std::uint64_t def) -> std::uint64_t
        {
            char const* v = std::getenv(name);
            if (v == nullptr || !std::isdigit(static_cast<unsigned char>(v[0])))
            {
                return def;
            }
            char* end = nullptr;
            errno = 0;
            std::uint64_t const parsed = std::strtoull(v, &end, 10);
            if (errno == ERANGE || end == v)
            {
                return def;
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
                return def; // unknown suffix -> default, same rationale as envU64
            }
            if (mult > 1 && parsed > std::numeric_limits<std::uint64_t>::max() / mult)
            {
                return def; // would overflow
            }
            return parsed * mult;
        };
        auto envSize = [&envU64](char const* name, std::size_t def, bool allowZero = true) -> std::size_t
        {
            auto const parsed = envU64(name, def);
            if ((!allowZero && parsed == 0) || parsed > std::numeric_limits<std::size_t>::max())
            {
                return def;
            }
            return static_cast<std::size_t>(parsed);
        };
        auto envSizeBytes = [&envBytes](char const* name, std::size_t def, bool allowZero = true) -> std::size_t
        {
            auto const parsed = envBytes(name, def);
            if ((!allowZero && parsed == 0) || parsed > std::numeric_limits<std::size_t>::max())
            {
                return def;
            }
            return static_cast<std::size_t>(parsed);
        };
        auto envU32 = [&envU64](char const* name, std::uint32_t def, bool allowZero = true) -> std::uint32_t
        {
            auto const parsed = envU64(name, def);
            if ((!allowZero && parsed == 0) || parsed > std::numeric_limits<std::uint32_t>::max())
            {
                return def;
            }
            return static_cast<std::uint32_t>(parsed);
        };
        auto envInt = [&envU64](char const* name, int def, bool allowZero = true) -> int
        {
            auto const parsed = envU64(name, static_cast<std::uint64_t>(def));
            if ((!allowZero && parsed == 0) || parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            {
                return def;
            }
            return static_cast<int>(parsed);
        };

        cfg.enabled = envBool("TRTLLM_NIXL_BOUNCE_ENABLE", cfg.enabled);
        cfg.arenaSizeBytes = envSizeBytes("TRTLLM_NIXL_BOUNCE_ARENA_SIZE_BYTES", cfg.arenaSizeBytes, false);
        cfg.arenaAllocationGranularityBytes = envSizeBytes(
            "TRTLLM_NIXL_BOUNCE_ARENA_ALLOCATION_GRANULARITY_BYTES", cfg.arenaAllocationGranularityBytes, false);
        cfg.maxChunkSizeBytes = envSizeBytes("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_SIZE_BYTES", cfg.maxChunkSizeBytes, false);
        cfg.maxInflightChunksPerRequest
            = envU32("TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST", cfg.maxInflightChunksPerRequest, false);
        cfg.copyStreamCount = envU32("TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT", cfg.copyStreamCount, false);
        cfg.scatterWorkerCount = envU32("TRTLLM_NIXL_BOUNCE_SCATTER_WORKER_COUNT", cfg.scatterWorkerCount, false);
        cfg.minDescriptorCount = envSize("TRTLLM_NIXL_BOUNCE_MIN_DESCRIPTOR_COUNT", cfg.minDescriptorCount);
        cfg.maxAverageDescriptorSizeBytes
            = envSizeBytes("TRTLLM_NIXL_BOUNCE_MAX_AVERAGE_DESCRIPTOR_SIZE_BYTES", cfg.maxAverageDescriptorSizeBytes);
        cfg.requestTimeoutMs = envInt("TRTLLM_NIXL_BOUNCE_REQUEST_TIMEOUT_MS", cfg.requestTimeoutMs);
        // Lease/quarantine derive from the one user-visible timeout (see the field comments): the
        // lease must exceed the peers' request timeout, and time is the only write barrier for a
        // reclaimed region. Disabling the request timeout (<=0) disables both.
        cfg.receiverFlowTimeoutMs = cfg.requestTimeoutMs > 0 ? 2 * cfg.requestTimeoutMs : 0;
        cfg.quarantineMs = cfg.requestTimeoutMs;
        cfg.disableFabricMemory = envBool("TRTLLM_NIXL_BOUNCE_DISABLE_FABRIC_MEMORY", cfg.disableFabricMemory);
        cfg.useZeroCopyArguments = envBool("TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS", cfg.useZeroCopyArguments);
        cfg.enableEagerGather = envBool("TRTLLM_NIXL_BOUNCE_ENABLE_EAGER_GATHER", cfg.enableEagerGather);
        return cfg;
    }
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
