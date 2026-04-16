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

import argparse

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def llm_instance():
    """Create a shared LLM instance for all tests in this module."""
    llm = LLM(
        model=MODEL,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True, iteration_stats_interval=1),
        enable_iter_perf_stats=True,
        return_perf_metrics=True,
    )
    yield llm
    llm.shutdown()


@pytest.fixture(scope="module")
def all_collected():
    """Shared list to accumulate stats across tests."""
    return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.threadleak(enabled=False)
class TestKvCacheIterationStats:
    def test_cold_start(self, llm_instance, all_collected, request):
        """Cold start — 3 unique prompts, no reuse expected."""
        prompts = [
            "The history of ancient Rome begins with",
            "Photosynthesis is the process by which plants",
            "The Pythagorean theorem states that in a right triangle",
        ]
        # Collect stats between each generate() call so that the context-phase
        # iteration stats (where iterMissedBlocks > 0) are captured before being
        # diluted by many generation/idle iterations with zero deltas.
        all_kv = []
        for p in prompts:
            o = llm_instance.generate([p], SamplingParams(max_tokens=64))
            print(f"  Input:  {p[:50]}...")
            print(f"  Output: {o[0].outputs[0].text!r}")
            stats = collect_stats(llm_instance, all_collected)
            if _is_verbose(request):
                print_kv_stats(f"Cold start ({p[:20]}...)", stats)
            all_kv.extend(find_kv_entries(stats))

        assert len(all_kv) > 0, "kvCacheIterationStats not found"

        # The overlap scheduler processes stats for the *previous* batch,
        # so per-iteration deltas may not align with the iteration that
        # allocated context blocks.  Check the cumulative kvCacheStats first
        # (always reliable), then the per-iteration deltas.
        cumul_missed = max(
            (s.get("kvCacheStats", {}).get("missedBlocks", 0) for s in all_collected),
            default=0,
        )
        iter_missed_found = any(v["iterMissedBlocks"] > 0 for _, _, v in all_kv)
        assert cumul_missed > 0 or iter_missed_found, (
            "No cold misses detected: cumulative missedBlocks = "
            f"{cumul_missed}, iterMissedBlocks > 0 in {sum(1 for _, _, v in all_kv if v['iterMissedBlocks'] > 0)}"
            f"/{len(all_kv)} entries"
        )
        assert any(v["iterAllocTotalBlocks"] > 0 for _, _, v in all_kv), (
            "iterAllocTotalBlocks = 0 in all entries"
        )
        assert any(v["iterGenAllocBlocks"] > 0 for _, _, v in all_kv), (
            "iterGenAllocBlocks = 0 in all entries"
        )

    def test_partial_block_reuse(self, llm_instance, all_collected, request):
        """Partial block reuse — short prompt (< 1 block) repeated x3.

        With tokens_per_block=32, a short prompt fits within a single block
        without filling it completely. On repeat, the block is reused but
        classified as partial (iterPartialReusedBlocks).
        """
        repeated = "The theory of general relativity tells us that gravity is"
        print(f"  Input:  {repeated!r}")
        for i in range(3):
            o = llm_instance.generate([repeated], SamplingParams(max_tokens=64))
            print(f"  Output (repeat {i + 1}): {o[0].outputs[0].text!r}")
        stats = collect_stats(llm_instance, all_collected)
        if _is_verbose(request):
            print_kv_stats("Partial block reuse", stats)
        kv = find_kv_entries(stats)

        assert len(kv) > 0, "kvCacheIterationStats not found"
        assert any(v["iterPartialReusedBlocks"] > 0 for _, _, v in kv), (
            "iterPartialReusedBlocks = 0 in all entries (expected partial reuse)"
        )
        assert any(v["iterCacheHitRate"] > 0 for _, _, v in kv), (
            "iterCacheHitRate = 0 in all entries (expected cache hits)"
        )

    def test_full_block_reuse(self, llm_instance, all_collected, request):
        """Full block reuse — prompt spanning 3+ blocks, repeated.

        With tokens_per_block=32, a prompt of ~120 tokens spans ~4 blocks.
        The first N-1 fully-filled blocks should register as iterFullReusedBlocks
        on the second request.
        """
        long_prompt = (
            "The quick brown fox jumps over the lazy dog and then runs across "
            "the wide open field where the tall green grass sways gently in the "
            "warm summer breeze while birds sing melodiously in the trees above "
            "and the river flows calmly through the valley carrying leaves and "
            "small stones downstream toward the distant ocean where waves crash "
            "against the rocky shore and seagulls circle overhead looking for "
            "fish beneath the sparkling surface of the deep blue water that"
        )

        # First request — cold, populates the radix tree
        print(f"  Input:  {long_prompt[:80]!r}... (~{len(long_prompt.split())} words)")
        o1 = llm_instance.generate([long_prompt], SamplingParams(max_tokens=16))
        print(f"  Output (1st, cold): {o1[0].outputs[0].text!r}")
        collect_stats(llm_instance, all_collected)  # drain stats from first request

        # Second request — identical prompt, should reuse full blocks
        o2 = llm_instance.generate([long_prompt], SamplingParams(max_tokens=16))
        print(f"  Output (2nd, warm): {o2[0].outputs[0].text!r}")
        stats = collect_stats(llm_instance, all_collected)
        if _is_verbose(request):
            print_kv_stats("Full block reuse", stats)
        kv = find_kv_entries(stats)

        assert len(kv) > 0, "kvCacheIterationStats not found"
        max_full = max(v["iterFullReusedBlocks"] for _, _, v in kv)
        max_reused = max(v["iterReusedBlocks"] for _, _, v in kv)
        assert max_full > 0, "iterFullReusedBlocks = 0 in all entries (expected full block reuse)"
        assert max_reused > max_full, (
            f"iterReusedBlocks = {max_reused} == iterFullReusedBlocks = {max_full} "
            "(expected at least one partial block too)"
        )

    def test_shared_prefix(self, llm_instance, all_collected, request):
        """Shared prefix — common prefix, 5 different suffixes."""
        prefix = "In the field of machine learning, neural networks are commonly used for "
        suffixes = [
            "image classification where the input data is",
            "natural language processing where the model learns to",
            "training with backpropagation which involves computing",
            "building layers of neurons that can represent",
            "learning complex patterns in data such as",
        ]
        print(f"  Prefix: {prefix!r}")
        for s in suffixes:
            full = prefix + s
            o = llm_instance.generate([full], SamplingParams(max_tokens=64))
            print(f"  Input:  ...{s!r}")
            print(f"  Output: {o[0].outputs[0].text!r}")
        stats = collect_stats(llm_instance, all_collected)
        if _is_verbose(request):
            print_kv_stats("Shared prefix", stats)
        kv = find_kv_entries(stats)

        assert len(kv) > 0, "kvCacheIterationStats not found"
        assert any(v["iterReusedBlocks"] > 0 for _, _, v in kv), (
            "iterReusedBlocks = 0 in all entries (expected prefix reuse)"
        )

    def test_batch_generation(self, llm_instance, all_collected, request):
        """Batch generation — 4 prompts in one generate() call."""
        batch = [
            "The capital of France is known for its",
            "The capital of Germany is a city that",
            "The capital of Japan is famous for its",
            "The capital of Brazil was designed by",
        ]
        outputs = llm_instance.generate(batch, SamplingParams(max_tokens=64))
        for p, o in zip(batch, outputs):
            print(f"  Input:  {p!r}")
            print(f"  Output: {o.outputs[0].text!r}")
        stats = collect_stats(llm_instance, all_collected)
        if _is_verbose(request):
            print_kv_stats("Batch generation", stats)
        kv = find_kv_entries(stats)

        assert len(kv) > 0, "kvCacheIterationStats not found"
        assert any(v["iterAllocTotalBlocks"] > 0 for _, _, v in kv), (
            "iterAllocTotalBlocks = 0 in all entries"
        )
        assert any(v["iterGenAllocBlocks"] > 0 for _, _, v in kv), (
            "iterGenAllocBlocks = 0 in all entries"
        )
        assert any(v["primaryUsedNumBlocks"] > 0 for _, _, v in kv), (
            "primaryUsedNumBlocks = 0 in all entries"
        )

    def test_long_context(self, llm_instance, all_collected, request):
        """Long context — single long prompt to allocate many blocks."""
        long_prompt = " ".join(
            [
                "The quick brown fox jumps over the lazy dog.",
                "A journey of a thousand miles begins with a single step.",
                "To be or not to be, that is the question.",
                "All that glitters is not gold.",
                "The only thing we have to fear is fear itself.",
                "In the beginning, there was nothing but darkness and void.",
                "Science is organized knowledge; wisdom is organized life.",
                "The unexamined life is not worth living.",
                "I think, therefore I am.",
                "That which does not kill us makes us stronger.",
                "The greatest glory in living lies not in never falling,",
                "but in rising every time we fall.",
                "Life is what happens when you are busy making other plans.",
                "The way to get started is to quit talking and begin doing.",
                "If you look at what you have in life,",
                "you will always have more. In conclusion, the meaning of",
            ]
        )
        o = llm_instance.generate([long_prompt], SamplingParams(max_tokens=128))
        print(f"  Input:  ({len(long_prompt.split())} words) {long_prompt[:80]!r}...")
        print(f"  Output: {o[0].outputs[0].text!r}")
        stats = collect_stats(llm_instance, all_collected)
        if _is_verbose(request):
            print_kv_stats("Long context", stats)
        kv = find_kv_entries(stats)

        assert len(kv) > 0, "kvCacheIterationStats not found"
        max_used = max(v["primaryUsedNumBlocks"] for _, _, v in kv)
        max_alloc = max(v["iterAllocTotalBlocks"] for _, _, v in kv)
        assert max_used > 0, "primaryUsedNumBlocks = 0 in all entries"
        assert max_alloc > 0, "iterAllocTotalBlocks = 0 in all entries"
        assert any(v["iterGenAllocBlocks"] > 0 for _, _, v in kv), (
            "iterGenAllocBlocks = 0 in all entries"
        )

    def test_rapid_fire(self, llm_instance, all_collected, request):
        """Rapid-fire — 20 short requests to accumulate deltas."""
        for i in range(20):
            prompt = f"Count to {i}: "
            o = llm_instance.generate([prompt], SamplingParams(max_tokens=32))
            if i % 5 == 0:
                print(f"  Input:  {prompt!r}")
                print(f"  Output: {o[0].outputs[0].text!r}")
        stats = collect_stats(llm_instance, all_collected)
        if _is_verbose(request):
            print_kv_stats("Rapid-fire", stats)
        kv = find_kv_entries(stats)

        assert len(kv) > 0, "kvCacheIterationStats not found"
        total_gen = sum(v["iterGenAllocBlocks"] for _, _, v in kv)
        total_alloc = sum(v["iterAllocTotalBlocks"] for _, _, v in kv)
        assert total_gen > 0, "iterGenAllocBlocks = 0 across all entries"
        assert total_alloc > 0, "iterAllocTotalBlocks = 0 across all entries"

    def test_field_completeness(self, llm_instance, all_collected, request):
        """Field completeness — verify all 18 fields present across all collected stats."""
        # If running standalone (no prior tests), generate some traffic
        if not all_collected:
            llm_instance.generate(["Hello world"], SamplingParams(max_tokens=16))
            collect_stats(llm_instance, all_collected)

        entries_with_kv = 0
        missing_fields = set()
        for s in all_collected:
            ki = s.get("kvCacheIterationStats")
            if ki:
                entries_with_kv += 1
                for ws, v in ki.items():
                    for field in ALL_FIELDS:
                        if field not in v:
                            missing_fields.add(field)

        print(f"  Entries with kvCacheIterationStats: {entries_with_kv}/{len(all_collected)}")
        assert entries_with_kv > 0, "no entries contain kvCacheIterationStats"
        assert len(missing_fields) == 0, f"Missing fields: {sorted(missing_fields)}"


# ---------------------------------------------------------------------------
# Standalone execution (python3 ... directly)
# ---------------------------------------------------------------------------
_STANDALONE_TEST_FUNCS = {
    1: "test_cold_start",
    2: "test_partial_block_reuse",
    3: "test_full_block_reuse",
    4: "test_shared_prefix",
    5: "test_batch_generation",
    6: "test_long_context",
    7: "test_rapid_fire",
    8: "test_field_completeness",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Dump all 18 KV cache stat fields for every stats entry",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=int,
        nargs="+",
        metavar="N",
        help="Run only the specified test(s) by number (e.g. --test 2 3)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List available tests and exit")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for num, name in TEST_NAMES.items():
            print(f"  {num}  {name}")
        return

    selected = args.test if args.test else sorted(_STANDALONE_TEST_FUNCS.keys())
    invalid = [t for t in selected if t not in _STANDALONE_TEST_FUNCS]
    if invalid:
        parser.error(f"Unknown test number(s): {invalid}. Use --list to see available tests.")

    # Create a fake request object for verbose flag
    class FakeConfig:
        def getoption(self, name, default=False):
            return args.verbose

    class FakeRequest:
        config = FakeConfig()

    fake_request = FakeRequest()

    print("Starting LLM with block_reuse + iteration_stats_interval=1")
    llm = LLM(
        model=MODEL,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True, iteration_stats_interval=1),
        enable_iter_perf_stats=True,
        return_perf_metrics=True,
    )
    all_collected = []
    results = {}
    test_cls = TestKvCacheIterationStats()

    for t in selected:
        name = f"Test {t}: {TEST_NAMES[t]}"
        method = getattr(test_cls, _STANDALONE_TEST_FUNCS[t])
        try:
            method(llm, all_collected, fake_request)
            results[name] = True
            print("  PASS")
        except AssertionError as e:
            results[name] = False
            print(f"  FAIL: {e}")

    llm.shutdown()

    print(f"\n{'=' * 60}")
    print(" SUMMARY")
    print(f"{'=' * 60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n  {passed}/{total} tests passed.")
    if passed < total:
        print("  Some tests FAILED — review output above.")
    else:
        print("  All tests passed!")


if __name__ == "__main__":
    main()
