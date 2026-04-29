# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for KVCacheManager._estimate_post_reuse_compute() — DYN-2868.

The helper is effectively pure: it depends only on the arguments and
``self.tokens_per_block``. Tests construct a stub holding tokens_per_block
and bind the unbound method, avoiding any real KVCacheManager initialization.

Tier 1A: branch coverage of the helper.
Tier 1B: divergence vs ``_prepare_tp_inputs`` actual position-id count.
Tier 1C: precise property tests — undercharges in the reuse branch
         (0 < P < prompt_len), short-circuits to chunk_size outside it.
"""
import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


class _Stub:
    """Minimal stand-in for KVCacheManager exposing only tokens_per_block."""

    def __init__(self, tokens_per_block):
        self.tokens_per_block = tokens_per_block

    _estimate_post_reuse_compute = KVCacheManager._estimate_post_reuse_compute


def _model_engine_forward_count(begin_compute, chunk_size, prompt_len):
    """Mirror model_engine.py:2293-2297 — len(prompt_tokens) per ctx req."""
    return max(0, min(chunk_size, prompt_len - begin_compute))


# ---------------------------------------------------------------------------
# Tier 1A: branch coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "P, chunk, prompt_len, block, expected, branch",
    [
        # P <= 0 → return chunk_size
        (0, 256, 1024, 64, 256, "no_reuse_zero"),
        (-5, 256, 1024, 64, 256, "no_reuse_negative"),
        # P >= prompt_len → return chunk_size
        (1024, 256, 1024, 64, 256, "full_reuse_eq"),
        (2048, 256, 1024, 64, 256, "full_reuse_gt"),
        # last-chunk branch (P + chunk >= prompt_len): max(1, prompt_len - P)
        (128, 1024, 200, 64, 72, "last_chunk_partial"),
        (199, 4, 200, 64, 1, "last_chunk_just_fits"),
        # non-last chunk: aligned_end = floor((P+chunk)/block)*block, max(1, aligned_end - P)
        (64, 128, 1024, 64, 128, "nonlast_block_aligned"),
        (70, 130, 1024, 64, 122, "nonlast_off_block"),
        (100, 4, 1024, 64, 1, "nonlast_max_clamp"),
        (50, 100, 1024, 1, 100, "block_size_one"),
    ],
)
def test_branch_coverage(P, chunk, prompt_len, block, expected, branch):
    stub = _Stub(tokens_per_block=block)
    actual = stub._estimate_post_reuse_compute(P, chunk, prompt_len)
    assert actual == expected, (
        f"branch={branch}: expected {expected}, got {actual} "
        f"(P={P}, chunk={chunk}, prompt_len={prompt_len}, block={block})"
    )


# ---------------------------------------------------------------------------
# Tier 1B: helper vs model_engine — production-relevant divergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "P, chunk_size, prompt_len, tokens_per_block, expect_undercharge",
    [
        # Block-aligned chunk + block-aligned reuse: helper == model_engine
        (64, 128, 1024, 64, False),
        (0, 256, 1024, 64, False),
        (128, 256, 200, 64, False),  # last chunk, exact match (helper=72, actual=72)
        (192, 64, 1024, 64, False),
        # Block-aligned reuse + UNALIGNED chunk: helper undercharges. This
        # case is legal in production per
        # tests/unittest/_torch/executor/test_kv_cache_v2_scheduler.py:243-247
        # (chunk_unit_size=100 with tokens_per_block=64).
        (64, 100, 1000, 64, True),   # aligned_end=128, helper=64,  actual=100
        (128, 50, 1024, 64, True),   # aligned_end=128, helper=1,   actual=50
        # Block-unaligned reuse: also undercharges
        (70, 130, 1024, 64, True),   # aligned_end=192, helper=122, actual=130
    ],
)
def test_helper_vs_model_engine_per_case(P, chunk_size, prompt_len,
                                         tokens_per_block,
                                         expect_undercharge):
    """Document where the helper diverges from _prepare_tp_inputs token
    counting. The aligned cases agree; the unaligned cases undercharge —
    which means the budget guard is conservative-enough to catch run-14's
    +2 overshoot but not a strict ≤ max guarantee under all legal scheduler
    outputs. Surfaced as a caveat in the PR description."""
    stub = _Stub(tokens_per_block)
    helper = stub._estimate_post_reuse_compute(P, chunk_size, prompt_len)
    actual = _model_engine_forward_count(P, chunk_size, prompt_len)
    if expect_undercharge:
        assert helper < actual, (
            f"Expected undercharge for unaligned (P={P}, chunk={chunk_size}, "
            f"block={tokens_per_block}): helper={helper}, actual={actual}. "
            f"If this fails, the helper has been tightened — update this "
            f"test and review whether Tier 3 needs new scenarios."
        )
    else:
        assert helper == actual, (
            f"Helper/model_engine mismatch under aligned inputs: "
            f"helper={helper}, actual={actual}, "
            f"(P={P}, chunk={chunk_size}, prompt_len={prompt_len}, "
            f"block={tokens_per_block})"
        )


# ---------------------------------------------------------------------------
# Tier 1C: precise property tests
# ---------------------------------------------------------------------------
def test_helper_undercharges_in_reuse_branch():
    """When 0 < P < prompt_len, helper <= actual model_engine forward count.
    This is the invariant the budget guard relies on for the run-14
    scenario (block-aligned reuse + block-aligned chunk_size). Counts as
    the load-bearing property — a regression that breaks this is a real bug."""
    counterexamples = []
    for P in [1, 32, 63, 64, 65, 70, 128, 199]:
        for chunk in [1, 50, 64, 100, 128, 256]:
            for prompt_len in [128, 200, 512, 1024]:
                if P >= prompt_len:
                    continue  # outside the reuse branch
                for block in [1, 32, 64]:
                    stub = _Stub(block)
                    h = stub._estimate_post_reuse_compute(P, chunk, prompt_len)
                    a = _model_engine_forward_count(P, chunk, prompt_len)
                    if h > a:
                        counterexamples.append(
                            (P, chunk, prompt_len, block, h, a))
    assert not counterexamples, (
        f"helper > actual in reuse branch (first 5): {counterexamples[:5]}"
    )


def test_helper_short_circuits_outside_reuse_range():
    """When P <= 0 or P >= prompt_len, helper returns chunk_size verbatim
    regardless of prompt_len — so the helper may OVERCHARGE actual forward
    tokens here. This is intentional: outside the reuse branch the guard
    falls back to the no-credit path, mirroring the pre-fix behavior."""
    stub = _Stub(tokens_per_block=64)
    # P=0: no reuse credit → return chunk_size even when chunk > prompt_len
    assert stub._estimate_post_reuse_compute(0, 256, 128) == 256
    # P negative: same path
    assert stub._estimate_post_reuse_compute(-5, 256, 128) == 256
    # P beyond prompt_len: same path
    assert stub._estimate_post_reuse_compute(2048, 256, 128) == 256
