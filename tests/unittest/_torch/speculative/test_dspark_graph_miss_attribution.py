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


class _Runner:
    """Just the fields will_pad_to reads."""

    def __init__(self, supported, cfg_batch_size):
        self.supported_batch_sizes = supported
        self.config = type("C", (), {"batch_size": cfg_batch_size})()


def test_will_pad_to_rejects_row_counts_the_ladder_cannot_reach():
    """The ragged fit must not derive a grid from rows that never appear.

    A run fitted its bucket grid to padded_bs=256 while the batch stayed at 193
    -- the captured ladder held 192 and 256 but not 193 -- so the filled total
    449 matched no captured bucket and every rank lost a graph replay on 16 of
    318 steps. Only reachable with real trimming: without it the total is
    padded_bs * (max_verify_len + 1), a captured bucket by construction.
    """
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
        CUDAGraphRunner

    runner = _Runner([1, 2, 4, 8, 128, 192, 256], cfg_batch_size=256)
    will = CUDAGraphRunner.will_pad_to

    assert will(runner, 256, 193), "193 -> 256 is reachable and should be taken"
    assert not will(runner, 193, 193), "193 is not a captured size"
    assert will(runner, 192, 192), "already at a captured size"
    # Padding past the configured batch size is refused by _get_padded_batch,
    # so the fit must not assume it.
    assert not will(_Runner([256, 512], cfg_batch_size=256), 512, 300)
    assert not will(runner, 0, 10)
    assert not will(runner, 256, 0)


def test_graph_key_uses_the_agreed_bucket_not_a_fresh_sum():
    """Every attention-DP rank must key on the same bucket, by construction.

    The key used to re-sum 1 + py_verify_len over generation_requests, which
    walks the batch *after* padding is appended -- a second, independent
    derivation of a value all ranks must match exactly. When the two
    derivations diverged the shape gate dropped every rank out of replay
    (peer_shape_mismatch: 2/265 steps without trimming, 16/318 with it, since
    trimming widens the per-rank window spread). Reading the fitted value
    instead makes agreement structural: it comes from allgathered peer stats
    through rules every rank runs identically.
    """
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
        CUDAGraphRunner

    import types

    class _R:
        spec_config = type("S", (), {"enable_ragged_verify": True})()
        agreed_ragged_bucket = None

    def _batch(*verify_lens):
        return types.SimpleNamespace(generation_requests=[
            types.SimpleNamespace(py_verify_len=v) for v in verify_lens
        ])

    r = _R()
    get = CUDAGraphRunner._ragged_verify_bucket
    windowed = _batch(3, 2, 1)

    # Nothing fitted yet -> not ragged, key unchanged from the uniform one.
    assert get(r, windowed) is None

    # A fitted bucket is returned verbatim: the batch's own token total is
    # never re-summed here, which is what makes every rank agree.
    r.agreed_ragged_bucket = 449
    assert get(r, windowed) == 449
    # ... including when that total plainly differs from the fitted value.
    assert get(r, _batch(1, 1)) == 449

    # But a request without a window means the token-major staging chain never
    # ran this step, so a ragged key would replay the PREVIOUS ragged step's
    # row maps -- the ragged IMA. Decline, whatever the fit published.
    assert get(r, _batch(3, None, 1)) is None

    # Cleared between steps so a stale bucket cannot key into the wrong graph.
    r.agreed_ragged_bucket = None
    assert get(r, windowed) is None

    # Config gate still wins: a non-ragged engine keeps the original key shape.
    r.agreed_ragged_bucket = 512
    r.spec_config = type("S", (), {"enable_ragged_verify": False})()
    assert get(r, windowed) is None


def test_capture_publishes_the_bucket_it_shaped_the_batch_to():
    """The other half of the test above, and the regression it let through.

    Once the key reads a fitted value instead of re-summing the batch, whoever
    shapes a batch has to publish the shape. Real steps do both in
    fit_ragged_verify_lens. CUDA-graph capture shapes its warmup batch in
    _set_warmup_ragged_windows and never called the fit, so the bucket stayed
    None while the batch really held `verify_bucket` tokens.

    Both consequences are silent in different ways. The capture-time forward
    ran with metadata describing a uniform batch over ragged tensors and died
    hundreds of frames later in the MoE on `unmatched tensor shape`. Had it
    survived, all three of a batch size's buckets would have been stored under
    one key with no token axis -- colliding with each other, and matching no
    runtime step, so every ragged step would have fallen back to eager without
    an error anywhere.
    """
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    runner = types.SimpleNamespace(agreed_ragged_bucket=None)
    requests = [types.SimpleNamespace(py_verify_len=None) for _ in range(4)]
    batch = types.SimpleNamespace(generation_requests=requests)
    engine = types.SimpleNamespace(cuda_graph_runner=runner)

    # bs=4, draft_len=5 -> the t=5 bucket is 4 * 6 = 24 tokens.
    PyTorchModelEngine._set_warmup_ragged_windows(engine, batch, 24, 5)

    assert sum(1 + r.py_verify_len for r in requests) == 24, (
        "the warmup batch must hit the bucket exactly")
    assert runner.agreed_ragged_bucket == 24, (
        "capture shaped the batch to 24 tokens but did not publish it, so the "
        "graph would be keyed as if the batch were uniform")

    # A narrower bucket at the same batch size, to prove it is per-entry and
    # not set once: capture walks every tier for every batch size.
    PyTorchModelEngine._set_warmup_ragged_windows(engine, batch, 8, 5)
    assert sum(1 + r.py_verify_len for r in requests) == 8
    assert runner.agreed_ragged_bucket == 8

    # An empty batch publishes nothing rather than a stale or zero bucket.
    runner.agreed_ragged_bucket = None
    PyTorchModelEngine._set_warmup_ragged_windows(
        engine, types.SimpleNamespace(generation_requests=[]), 24, 5)
    assert runner.agreed_ragged_bucket is None
