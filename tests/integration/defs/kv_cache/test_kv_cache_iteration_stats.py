# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for per-iteration KV cache statistics (kvCacheIterationStats).

Tests verify that the 18 stat fields are correctly populated across
different inference scenarios: cold start, block reuse (partial/full),
shared prefix, batch generation, long context, and rapid-fire.

Usage:
    # Via pytest (recommended):
    pytest tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py
    pytest tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py -k "cold_start"
    pytest tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py -s   # show prints
    pytest tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py -s --verbose-stats

    # Standalone (still supported):
    python3 tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py
    python3 tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py --verbose
    python3 tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py --test 2 3
    python3 tests/integration/defs/kv_cache/test_kv_cache_iteration_stats.py --list
"""

from ..conftest import llm_models_root

MODEL = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"

ALL_FIELDS = [
    # Instantaneous gauges — primary (GPU) pool
    "primaryMaxNumBlocks",
    "primaryFreeNumBlocks",
    "primaryUsedNumBlocks",
    # Instantaneous gauges — secondary (host) pool
    "secondaryMaxNumBlocks",
    "secondaryFreeNumBlocks",
    "secondaryUsedNumBlocks",
    # Per-iteration deltas — context phase
    "iterAllocTotalBlocks",
    "iterAllocNewBlocks",
    "iterReusedBlocks",
    "iterFullReusedBlocks",
    "iterPartialReusedBlocks",
    "iterMissedBlocks",
    "iterCacheHitRate",
    # Per-iteration deltas — generation phase
    "iterGenAllocBlocks",
    # Per-iteration deltas — transfer traffic
    "iterOnboardBlocks",
    "iterOnboardBytes",
    "iterOffloadBlocks",
    "iterOffloadBytes",
    # Intra-device (GPU → GPU) block copies
    "iterIntraDeviceCopyBlocks",
    "iterIntraDeviceCopyBytes",
]

TEST_NAMES = {
    1: "Cold start",
    2: "Partial block reuse",
    3: "Full block reuse",
    4: "Shared prefix",
    5: "Batch generation",
    6: "Long context",
    7: "Rapid-fire",
    8: "Field completeness",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_verbose(request):
    """Check if verbose stats output is requested (pytest or standalone)."""
    if request is not None:
        return request.config.getoption("--verbose-stats", default=False)
    return False


def print_kv_stats(label, stats_list):
    """Print all 18 fields for every stats entry."""
    print(f"\n{'=' * 60}")
    print(f" {label}: {len(stats_list)} stats entries")
    print(f"{'=' * 60}")
    found = False
    for i, s in enumerate(stats_list):
        ki = s.get("kvCacheIterationStats")
        if ki:
            found = True
            for ws, v in ki.items():
                print(f"\n  --- entry[{i}] window_size={ws} ---")
                for field in ALL_FIELDS:
                    val = v.get(field, "<MISSING>")
                    print(f"    {field:30s} = {val}")
        else:
            keys = list(s.keys())[:8]
            print(f"  entry[{i}]: no kvCacheIterationStats (keys: {keys})")
    if not found:
        print("  WARNING: no entry contained kvCacheIterationStats!")


def collect_stats(llm, all_collected):
    """Get stats and append to the cumulative list."""
    stats = llm.get_stats(timeout=2)
    all_collected.extend(stats)
    return stats


def find_kv_entries(stats_list):
    """Extract all (entry_index, window_size, fields_dict) from stats."""
    results = []
    for i, s in enumerate(stats_list):
        ki = s.get("kvCacheIterationStats")
        if ki:
            for ws, v in ki.items():
                results.append((i, ws, v))
    return results
