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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// POD config for the bounce v2 pipeline. `fromEnv()` snapshots `TRTLLM_NIXL_BOUNCE_*` once.
struct BounceConfig
{
    bool enabled{false};                      // TRTLLM_NIXL_BOUNCE_ENABLE
    std::size_t arenaBytes{256ULL << 20};     // TRTLLM_NIXL_BOUNCE_ARENA_BYTES (shared region arena, 256 MiB)
    std::size_t minBlock{1ULL << 20};         // TRTLLM_NIXL_BOUNCE_MIN_BLOCK (buddy min region, 1 MiB)
    std::size_t maxChunkBytes{32ULL << 20};   // TRTLLM_NIXL_BOUNCE_MAX_CHUNK_BYTES (per-chunk byte cap, 32 MiB)
    std::uint32_t windowDepth{8};             // TRTLLM_NIXL_BOUNCE_DEPTH (default per-flow in-flight region cap)
    std::uint32_t window{0};                  // TRTLLM_NIXL_BOUNCE_WINDOW (per-flow cap W override; 0 == windowDepth)
    std::uint32_t execCtxCount{8};            // TRTLLM_NIXL_BOUNCE_EXEC_CTX (gather/scatter exec contexts)
    std::uint32_t scatterWorkers{4};          // TRTLLM_NIXL_BOUNCE_SCATTER_WORKERS
    std::size_t minDescCount{1024};           // TRTLLM_NIXL_BOUNCE_MIN_DESC (heuristic gate)
    std::size_t maxAvgDescBytes{16ULL << 10}; // TRTLLM_NIXL_BOUNCE_MAX_AVG (16 KiB)
    int leaseTimeoutMs{30000};                // TRTLLM_NIXL_BOUNCE_LEASE_TIMEOUT_MS
    bool forceFallback{false};                // TRTLLM_NIXL_BOUNCE_FORCE_FALLBACK (no fabric mem; CI)
    // --- experimental gather/scatter copy backends (default OFF; benchmark before enabling) ---
    bool cubCopy{false};      // TRTLLM_NIXL_BOUNCE_CUB_COPY: use cub::DeviceMemcpy::Batched vs the custom kernel
    bool zeroCopyArgs{false}; // TRTLLM_NIXL_BOUNCE_ZEROCOPY_ARGS: kernel reads the [srcs|dsts|sizes] plan arrays
                              // directly from pinned host (skip their H2D copy) — likely a loss for large n

    /// Effective per-flow window: explicit `window` if set, else `windowDepth` (per-flow region cap).
    [[nodiscard]] std::uint32_t effectiveWindow() const noexcept
    {
        return window > 0 ? window : windowDepth;
    }

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
            if (v == nullptr || v[0] == '\0')
            {
                return def;
            }
            // Parse strictly: a garbage value (typo like "abc", or trailing junk) falls back to the
            // default instead of yielding 0 — a 0 here would later abort the process (e.g.
            // maxChunkBytes=0 trips a TLLM_CHECK in BounceTransferPlan::build).
            char* end = nullptr;
            std::uint64_t const parsed = std::strtoull(v, &end, 10);
            if (end == v || *end != '\0')
            {
                return def;
            }
            return parsed;
        };

        cfg.enabled = envBool("TRTLLM_NIXL_BOUNCE_ENABLE", false);
        cfg.arenaBytes = static_cast<std::size_t>(envU64("TRTLLM_NIXL_BOUNCE_ARENA_BYTES", cfg.arenaBytes));
        cfg.minBlock = static_cast<std::size_t>(envU64("TRTLLM_NIXL_BOUNCE_MIN_BLOCK", cfg.minBlock));
        cfg.maxChunkBytes = static_cast<std::size_t>(envU64("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_BYTES", cfg.maxChunkBytes));
        cfg.windowDepth = static_cast<std::uint32_t>(envU64("TRTLLM_NIXL_BOUNCE_DEPTH", cfg.windowDepth));
        cfg.window = static_cast<std::uint32_t>(envU64("TRTLLM_NIXL_BOUNCE_WINDOW", cfg.window));
        cfg.execCtxCount = static_cast<std::uint32_t>(envU64("TRTLLM_NIXL_BOUNCE_EXEC_CTX", cfg.execCtxCount));
        cfg.scatterWorkers
            = static_cast<std::uint32_t>(envU64("TRTLLM_NIXL_BOUNCE_SCATTER_WORKERS", cfg.scatterWorkers));
        cfg.minDescCount = static_cast<std::size_t>(envU64("TRTLLM_NIXL_BOUNCE_MIN_DESC", cfg.minDescCount));
        cfg.maxAvgDescBytes = static_cast<std::size_t>(envU64("TRTLLM_NIXL_BOUNCE_MAX_AVG", cfg.maxAvgDescBytes));
        cfg.leaseTimeoutMs = static_cast<int>(
            envU64("TRTLLM_NIXL_BOUNCE_LEASE_TIMEOUT_MS", static_cast<std::uint64_t>(cfg.leaseTimeoutMs)));
        cfg.forceFallback = envBool("TRTLLM_NIXL_BOUNCE_FORCE_FALLBACK", false);
        cfg.cubCopy = envBool("TRTLLM_NIXL_BOUNCE_CUB_COPY", false);
        cfg.zeroCopyArgs = envBool("TRTLLM_NIXL_BOUNCE_ZEROCOPY_ARGS", false);
        return cfg;
    }
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
