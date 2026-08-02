# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which CUDA-graph misses the ragged gate is allowed to blame on ragged.

A miss can mean opposite things and the counter alone cannot tell them apart.
``peer_not_gen_only`` is ordinary continuous batching -- some rank had context
requests, which drops the whole attention-DP world out of replay and happens on
the uniform path too. ``key_not_captured`` means the batch was graphable, every
rank agreed, and the shape was still missing: the token bucket grid is wrong.
``peer_shape_mismatch`` means one rank chose a different ragged bucket, which is
the attention-DP divergence this feature exists to avoid.

Gating on the raw count made the check unsatisfiable on any real workload; not
gating at all let a real bucket bug through. These tests pin the split.
"""

import pytest

from tensorrt_llm._torch.speculative.dspark_observability import (
    DSparkRaggedStats, RaggedVerifyMode)


def _stats(**reasons):
    """Stats that look like a healthy ragged run apart from the given misses."""
    stats = DSparkRaggedStats(mode=RaggedVerifyMode.COMPACT, max_draft_len=5)
    stats.steps_total = 100
    stats.steps_ragged = 80
    # distinct_verify_lens is derived from this histogram, so populate the
    # histogram rather than the property.
    stats.verify_len_hist.update({1: 10, 3: 10, 5: 10})
    stats.ceiling_tokens = 1000
    stats.delivered_tokens = 800  # trim_ratio 0.2 > 0, so require_trim passes
    stats.window_tokens = 800
    stats.graph_replays = 500
    for reason, count in reasons.items():
        stats.graph_eager += count
        stats.graph_miss_reasons[reason] = count
    return stats


def test_continuous_batching_misses_are_not_blamed_on_ragged():
    """A rank with context requests is normal, not a ragged defect."""
    _stats(peer_not_gen_only=9).assert_ragged_active(require_trim=True)


@pytest.mark.parametrize(
    "reason", ["key_not_captured", "bs_not_supported", "peer_shape_mismatch"])
def test_ragged_attributable_misses_raise(reason):
    with pytest.raises(AssertionError) as excinfo:
        _stats(**{reason: 1}).assert_ragged_active(require_trim=True)
    assert reason in str(excinfo.value)


def test_mixed_misses_raise_and_name_only_the_attributable_ones():
    with pytest.raises(AssertionError) as excinfo:
        _stats(peer_not_gen_only=20,
               key_not_captured=2).assert_ragged_active(require_trim=True)
    message = str(excinfo.value)
    assert "key_not_captured" in message
    # The tolerated reason must not appear in the *blame* dict, or the message
    # sends the reader to the wrong place. It still appears in the summary.
    blame = message.split("(", 1)[1].split(")", 1)[0]
    assert "peer_not_gen_only" not in blame


def test_unattributed_misses_are_a_broken_probe_not_a_clean_run():
    """graph_eager > 0 with no reasons recorded must not read as success.

    This is what actually happened: a build skew left the reason unrecorded and
    the summary showed misses with an empty reason map. Silently passing there
    would make a broken probe indistinguishable from a healthy run.
    """
    stats = _stats()
    stats.graph_eager = 9  # counted, but no reason recorded
    with pytest.raises(AssertionError, match="cannot be attributed"):
        stats.assert_ragged_active(require_trim=True)


def test_clean_run_passes():
    _stats().assert_ragged_active(require_trim=True)


def test_ramp_up_steps_do_not_count_toward_the_gate_floor():
    """A batch of one is uniform by construction, not a scheduler failure.

    trtllm-bench ramps concurrency from zero, so a step-count floor of 64 can be
    reached while only two requests are in flight. Every window is then
    trivially identical, and gating on total steps failed a run whose scheduler
    had done nothing wrong -- observed as "Broadcasting event-loop error to 2
    pending request(s)". The floor counts steps whose batch could actually be
    ragged.
    """
    stats = DSparkRaggedStats(mode=RaggedVerifyMode.COMPACT, max_draft_len=5)
    for _ in range(64):
        stats.record_step(num_gen_requests=1, verify_lens=[5])
    assert stats.steps_total == 64
    assert stats.steps_multi_request == 0, (
        "single-request steps must not count toward the gate floor")

    for _ in range(64):
        stats.record_step(num_gen_requests=4, verify_lens=[1, 3, 5, 5])
    assert stats.steps_multi_request == 64
