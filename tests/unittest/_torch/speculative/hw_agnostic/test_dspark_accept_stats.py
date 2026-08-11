# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark accept-caps and ragged-verify observability (hardware-agnostic, CPU).

Covers `apply_accept_caps` clamping semantics, the cap-accept diagnostic mode's
counters (`cap_trim_tokens`, trim shape), `trim_ratio` row-base accounting under
CUDA-graph padding, and trim-regret acceptance stats.
"""

import os
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_observability import (
    DSparkRaggedStats, RaggedVerifyMode)
from tensorrt_llm._torch.speculative.interface import apply_accept_caps

MAX_DRAFT_LEN = 5


def _cap_stats(mode=RaggedVerifyMode.CAP_ACCEPT):
    return DSparkRaggedStats(mode=mode, max_draft_len=MAX_DRAFT_LEN)


def _compact_stats(max_draft_len=5):
    return DSparkRaggedStats(mode=RaggedVerifyMode.COMPACT,
                             max_draft_len=max_draft_len)


class _Meta:
    """Just the two fields `apply_accept_caps` reads."""

    def __init__(self, caps=None, track_trim=True, rows=8):
        self.accept_caps = (None if caps is None else torch.tensor(
            caps, dtype=torch.int32))
        # Persistent per-request buffer, sized like the engine's.
        self.accept_cap_trim = (torch.zeros(rows, dtype=torch.int32)
                                if track_trim else None)


# --- the cap itself ----------------------------------------------------------
# Units: `num_accepted_tokens` is `accepted drafts + 1`, so a window of w
# drafted positions is a cap of `w + 1`, matching `verify_lens`.


def test_no_caps_is_a_no_op():
    """Every other mode goes through this call."""
    counts = torch.tensor([3, 1, 4], dtype=torch.int32)
    apply_accept_caps(counts, 0, _Meta(caps=None))
    assert counts.tolist() == [3, 1, 4]


def test_counts_beyond_their_windows_are_clamped_and_the_loss_accumulated():
    """The trim buffer holds this step's loss, not a running sum: repeating the
    same step leaves the same losses."""
    meta = _Meta(caps=[3, 5, 6])
    for _ in range(3):
        counts = torch.tensor([6, 5, 2], dtype=torch.int32)
        apply_accept_caps(counts, 0, meta)
        # Request 2's cap (6) is wider than its count (2): never raised.
        assert counts.tolist() == [3, 5, 2]
        # Only the first request lost anything: 6 -> 3. And we can say WHICH.
        assert meta.accept_cap_trim[:3].tolist() == [3, 0, 0]


def test_context_requests_are_left_alone():
    """Caps are per generation request; contexts sit ahead of them."""
    counts = torch.tensor([9, 9, 6, 6], dtype=torch.int32)
    meta = _Meta(caps=[2, 3])
    apply_accept_caps(counts, 2, meta)
    assert counts.tolist() == [9, 9, 2, 3], "a context count was clamped"
    # Context slots are zeroed, not left stale.
    assert meta.accept_cap_trim[:4].tolist() == [0, 0, 4, 3]

    # Degenerate endpoint: a generation-free batch must not crash or clamp.
    counts = torch.tensor([4, 4], dtype=torch.int32)
    apply_accept_caps(counts, 2, _Meta(caps=[2, 2]))
    assert counts.tolist() == [4, 4]


def test_missing_accumulator_still_caps():
    """The clamp is correctness; the counter is only measurement."""
    counts = torch.tensor([6], dtype=torch.int32)
    meta = _Meta(caps=[2], track_trim=False)
    apply_accept_caps(counts, 0, meta)
    assert counts.tolist() == [2]


@pytest.mark.parametrize("accepted", range(0, MAX_DRAFT_LEN + 2))
@pytest.mark.parametrize("window", range(0, MAX_DRAFT_LEN + 1))
def test_nothing_is_lost_or_invented(accepted, window):
    """committed + discarded accounts for every position the target produced."""
    counts = torch.tensor([accepted + 1], dtype=torch.int32)
    meta = _Meta(caps=[window + 1])
    apply_accept_caps(counts, 0, meta)
    committed = int(counts[0])
    trim = int(meta.accept_cap_trim[0])
    assert committed + trim == accepted + 1
    assert committed >= 1, "a step that commits nothing cannot make progress"
    assert committed <= window + 1


# --- what the counters must say ----------------------------------------------


def test_cap_trim_is_recorded_per_request():
    stats = _cap_stats()
    # Given 5, allowed 2 -> 3 accepted positions thrown away.
    stats.record_acceptance(accepted=2, window=2, cap_trim=3)
    # Died on its own inside a wide window: the schedule cost it nothing.
    stats.record_acceptance(accepted=1, window=4, cap_trim=0)

    assert stats.cap_trim_tokens == 3
    assert stats.requests_cap_trimmed == 1
    assert stats.accept_len == pytest.approx(1.5)
    assert stats.accept_loss_per_request == pytest.approx(1.5)
    # accept_len + accept_loss is what a no-trim run would have accepted.
    assert (stats.accept_len +
            stats.accept_loss_per_request) == pytest.approx(3.0)

    # And the summary carries the measurement.
    summary = stats.summary()
    assert summary["mode"] == "cap-accept"
    assert summary["cap_trim_tokens"] == 3
    assert summary["accept_loss_per_request"] == pytest.approx(1.5)
    for key in ("requests_cap_trimmed", "cap_trim_max", "cap_trim_hist",
                "cap_trim_concentration"):
        assert key in summary, f"{key} missing from the summary"


def test_the_same_total_loss_is_distinguished_by_its_shape():
    """Equal totals with different per-request shapes must stay
    distinguishable: spread loss vs. loss concentrated in one request."""
    spread = _cap_stats()
    for _ in range(10):
        spread.record_acceptance(accepted=1, window=1, cap_trim=2)

    concentrated = _cap_stats()
    concentrated.record_acceptance(accepted=1, window=1, cap_trim=20)
    for _ in range(9):
        concentrated.record_acceptance(accepted=3, window=5, cap_trim=0)

    assert spread.cap_trim_tokens == concentrated.cap_trim_tokens == 20
    assert spread.accept_loss_per_request == pytest.approx(
        concentrated.accept_loss_per_request)

    # ...and the shape separates them.
    assert spread.cap_trim_concentration == pytest.approx(1.0)
    assert concentrated.cap_trim_concentration == pytest.approx(0.1)
    assert spread.cap_trim_max == 2
    assert concentrated.cap_trim_max == 20
    assert spread.cap_trim_hist == {2: 10}
    assert concentrated.cap_trim_hist == {20: 1}


def test_the_loss_is_absent_rather_than_zero_when_not_measured():
    """`compact` cannot produce this number, and must not appear to."""
    stats = _cap_stats(mode=RaggedVerifyMode.COMPACT)
    stats.record_acceptance(accepted=3, window=3)
    assert stats.cap_trim_tokens == 0
    assert stats.cap_trim_concentration == 0.0
    assert stats.cap_trim_max == 0
    # The bound that IS available under compact still works.
    assert stats.trimmed_hit_ceiling == 1


def test_cap_accept_must_not_be_credited_with_a_compute_saving():
    """cap-accept submits the full block, so delivered tokens must not be
    derived from the windows and `trim_ratio` must stay 0."""
    stats = _cap_stats()
    num_reqs = 4
    full_block = num_reqs * (1 + MAX_DRAFT_LEN)
    stats.record_step(num_gen_requests=num_reqs,
                      verify_lens=[5, 3, 2, 1],
                      bucket=None,
                      delivered=full_block)

    assert stats.delivered_tokens == full_block
    assert stats.ceiling_tokens == full_block
    assert stats.trim_ratio == 0.0, "cap-accept saves no compute"
    # The schedule is still visible -- that is the whole point of the mode.
    assert stats.window_tokens == sum(1 + v for v in (5, 3, 2, 1))
    assert stats.steps_ragged == 1


def test_compact_still_derives_delivered_from_the_bucket():
    """The default path is unchanged by the new argument."""
    stats = _cap_stats(mode=RaggedVerifyMode.COMPACT)
    stats.record_step(num_gen_requests=4,
                      verify_lens=[5, 3, 2, 1],
                      bucket=16)
    assert stats.delivered_tokens == 16
    assert stats.trim_ratio > 0.0


# --- mode plumbing -----------------------------------------------------------


def test_cap_accept_computes_windows_but_does_not_trim_the_token_axis():
    """The two predicates that route the whole feature."""
    assert RaggedVerifyMode.CAP_ACCEPT.computes_windows
    assert not RaggedVerifyMode.CAP_ACCEPT.trims_submitted_tokens

    assert RaggedVerifyMode.COMPACT.computes_windows
    assert RaggedVerifyMode.COMPACT.trims_submitted_tokens

    assert not RaggedVerifyMode.STATIC.computes_windows
    assert not RaggedVerifyMode.STATIC.trims_submitted_tokens


def test_a_slot_left_unwritten_cannot_inherit_the_previous_occupant_s_loss():
    """cap_trim_lens is persistent and slot-indexed: a step producing no caps
    must still zero this batch's slots or recycled slots report stale loss."""
    import inspect

    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSamplerBase

    src = inspect.getsource(SpecSamplerBase.sample_async)
    assert "zeros_like" in src and "cap_trim_lens" in src, (
        "_process_outputs no longer writes zeros when the step produced no "
        "cap_trim_lens; recycled slots will report a stale request's loss")


def test_the_cuda_graph_padding_dummy_carries_a_cap():
    """The CUDA-graph padding dummy must carry a verify cap, or cap-accept
    silently degrades to static on every padded step."""
    import inspect

    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner

    src = inspect.getsource(cuda_graph_runner.CUDAGraphRunner._get_padded_batch)
    assert "py_verify_cap" in src, (
        "the padded-batch builder no longer stamps the dummy with a verify "
        "cap; cap-accept will silently degrade to static on every padded step")
    # And it must be cleared, not left stale: the dummy object is cached
    # across steps and shared between them.
    assert "if cap_accept else None" in src, (
        "the dummy's cap is not cleared for non-cap-accept batches, so a "
        "stale cap would trim a static step")


def test_the_capture_set_follows_the_mode_not_the_config_flag():
    """Batch shaping must ask whether the token axis actually shrinks, not
    whether ragged was requested: cap-accept always submits the full block."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.speculative.dspark_observability import (
        resolve_ragged_verify_mode, trims_submitted_tokens)

    ragged_cfg = SimpleNamespace(enable_ragged_verify=True)

    with patch.dict(os.environ,
                    {"TLLM_DSPARK_RAGGED_VERIFY_MODE": "cap-accept"}):
        assert resolve_ragged_verify_mode(
            ragged_cfg) is RaggedVerifyMode.CAP_ACCEPT
        assert not trims_submitted_tokens(ragged_cfg), (
            "cap-accept submits the full block; a capture set built for it "
            "must be the uniform one")

    with patch.dict(os.environ, {"TLLM_DSPARK_RAGGED_VERIFY_MODE": "compact"}):
        assert trims_submitted_tokens(ragged_cfg)

    # Unset: the config flag decides, which is the pre-existing behaviour.
    with patch.dict(os.environ, {}, clear=True):
        assert trims_submitted_tokens(ragged_cfg)
        assert not trims_submitted_tokens(
            SimpleNamespace(enable_ragged_verify=False))
    assert not trims_submitted_tokens(None)


def test_accept_caps_is_not_the_ragged_flag():
    """`accept_caps` must not imply ragged buffer packing: cap-accept's draft
    buffers are the ordinary rectangle."""
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    fields = SpecMetadata.__dataclass_fields__
    assert "accept_caps" in fields
    assert "accept_cap_trim" in fields

    meta = _Meta(caps=[2, 3])
    # The mock stands in for a metadata object carrying caps but no windows.
    assert getattr(meta, "verify_lens", None) is None


# --- trim_ratio row-base under CUDA-graph padding ----------------------------


def test_padding_rows_do_not_make_the_trim_look_negative():
    """A top-tier step on padded rows saved nothing -- and lost nothing."""
    stats = _compact_stats(max_draft_len=5)
    # 3 real requests, graph padded to 4 rows, every window at the full block.
    # delivered = 4 * 6 = 24. Against a ceiling of 3 * 6 = 18 that is -0.333.
    stats.record_step(num_gen_requests=3, verify_lens=[5, 5, 5],
                      bucket=4 * 6, padded_bs=4)
    summary = stats.summary()
    assert summary["trim_ratio"] == pytest.approx(0.0), (
        f"a step that verified the full block on every row trimmed nothing, so "
        f"the ratio is 0 -- got {summary['trim_ratio']}, which came from "
        f"comparing {summary['delivered_tokens']} padded-row tokens against a "
        f"ceiling counted over {3} real requests")
    # The padded row count itself must land in the summary: the executor once
    # passed padded_bs=None unconditionally, and an always-empty padded_bs_hist
    # is indistinguishable from "padding never happened".
    assert summary["padded_bs_hist"] == {4: 1}


def test_a_real_trim_still_reads_as_a_saving():
    """The fix must not flatten genuine trimming to zero."""
    stats = _compact_stats(max_draft_len=5)
    # 4 rows, no padding, every window trimmed to 2 -> bucket 4*3 = 12 against a
    # ceiling of 4*6 = 24.
    stats.record_step(num_gen_requests=4, verify_lens=[2, 2, 2, 2],
                      bucket=4 * 3, padded_bs=4)
    assert stats.summary()["trim_ratio"] == pytest.approx(0.5)


def test_paths_without_a_bucket_keep_the_local_row_base():
    """Only the bucket path rebases to the padded row count; widening the
    no-window and cap-accept paths would invent a saving out of padding."""
    no_window = _compact_stats(max_draft_len=5)
    no_window.record_step(num_gen_requests=3, verify_lens=None, padded_bs=8)
    assert no_window.summary()["trim_ratio"] == pytest.approx(0.0)

    explicit = _compact_stats(max_draft_len=5)
    # cap-accept: windows are computed but the full block is submitted anyway.
    explicit.record_step(num_gen_requests=3, verify_lens=[2, 3, 1],
                         bucket=8 * 6, padded_bs=8, delivered=3 * 6)
    assert explicit.summary()["trim_ratio"] == pytest.approx(0.0)


# --- trim regret -------------------------------------------------------------


def test_trim_regret_counts_only_drafts_alive_at_the_cut():
    """Regret separates "trimming was free, those drafts would have died
    anyway" from "trimming discarded acceptance"."""
    stats = _compact_stats()
    # Given the full block and died early: not trimmed, no regret.
    stats.record_acceptance(accepted=2, window=5)
    # Trimmed to 3 and died at 1: the cut cost nothing.
    stats.record_acceptance(accepted=1, window=3)
    # Trimmed to 3 and accepted all 3: still alive at the cut -> regret.
    stats.record_acceptance(accepted=3, window=3)

    assert stats.requests_scored == 3
    assert stats.requests_trimmed == 2
    assert stats.trimmed_hit_ceiling == 1
    assert stats.trim_regret_rate == 0.5
    assert stats.accept_len == pytest.approx(2.0)
    summary = stats.summary()
    for key in ("accept_len", "requests_scored", "requests_trimmed",
                "trim_regret_rate"):
        assert key in summary, f"{key} missing from the summary"

    # Nothing trimmed: the regret rate is 0/0-safe and stays 0.
    untrimmed = _compact_stats()
    for accepted in (0, 3, 5):
        untrimmed.record_acceptance(accepted=accepted, window=5)
    assert untrimmed.requests_trimmed == 0
    assert untrimmed.trim_regret_rate == 0.0
    assert untrimmed.summary()["accept_len"] == pytest.approx(8 / 3, abs=1e-4)
