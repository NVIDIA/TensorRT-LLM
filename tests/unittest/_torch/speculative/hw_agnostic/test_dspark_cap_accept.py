# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cap-accept``: run the schedule's policy without the layout that pays for it.

The mode exists to split one question into two. ``compact`` changes both what
the scheduler commits *and* how the batch is laid out, so when its output moves,
nothing says which half did it. ``cap-accept`` applies the same per-request
windows to commitment while submitting the full block, giving two independent
assertions instead of one entangled one:

    cap-accept output != static output   ->  the scheduling policy is wrong
    compact output    != cap-accept      ->  the layout compaction is wrong

It also produces a number ``compact`` structurally cannot: ``cap_trim_tokens``,
the positions the target accepted and the window then discarded. Under
``compact`` those positions are never scored, so the acceptance cost of trimming
can only be bounded (``trim_regret_rate``), never measured.

The price is that it saves no compute at all, which is why it is a diagnostic
mode and never a serving one -- and why the counters must not report it as
though it had saved any.
"""

import pytest

from tensorrt_llm._torch.speculative.dspark_observability import (
    DSparkRaggedStats, RaggedVerifyMode)
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSamplerBase

MAX_DRAFT_LEN = 5


def _stats(mode=RaggedVerifyMode.CAP_ACCEPT):
    return DSparkRaggedStats(mode=mode, max_draft_len=MAX_DRAFT_LEN)


# --- the cap itself ---------------------------------------------------------
#
# `num_new_tokens` is the uncapped chain: accepted drafts + the bonus at the
# first unaccepted position. A window of `cap` drafted positions admits
# `cap + 1` tokens.


def test_no_cap_leaves_the_commit_untouched():
    """Every non-cap-accept path goes through here; it must be a no-op."""
    assert SpecSamplerBase._apply_verify_cap(4, None) == (4, 0)


@pytest.mark.parametrize("num_new_tokens,cap", [(1, 5), (3, 5), (6, 5),
                                                (1, 0), (2, 1), (4, 3)])
def test_a_commit_inside_its_window_is_not_trimmed(num_new_tokens, cap):
    """The draft died before the cut, so the cut cost nothing."""
    committed, trim = SpecSamplerBase._apply_verify_cap(num_new_tokens, cap)
    assert (committed, trim) == (num_new_tokens, 0)


def test_a_commit_beyond_its_window_is_truncated_to_the_window():
    # Window of 2 drafted positions admits 3 tokens; the target accepted 5.
    committed, trim = SpecSamplerBase._apply_verify_cap(6, 2)
    assert committed == 3
    assert trim == 3


def test_a_zero_window_still_commits_the_bonus():
    """min_verify_len can be 0 drafted positions; the step must still advance.

    A request that commits nothing makes no progress and would spin forever.
    """
    committed, trim = SpecSamplerBase._apply_verify_cap(6, 0)
    assert committed == 1
    assert trim == 5


@pytest.mark.parametrize("num_new_tokens", range(1, MAX_DRAFT_LEN + 2))
@pytest.mark.parametrize("cap", range(0, MAX_DRAFT_LEN + 1))
def test_nothing_is_lost_or_invented(num_new_tokens, cap):
    """committed + discarded accounts for every token the target produced."""
    committed, trim = SpecSamplerBase._apply_verify_cap(num_new_tokens, cap)
    assert committed + trim == num_new_tokens
    assert committed >= 1, "a step that commits nothing cannot make progress"
    assert trim >= 0


# --- what the counters must say ---------------------------------------------


def test_cap_trim_is_recorded_as_the_exact_acceptance_loss():
    stats = _stats()
    # Given 5, allowed 2, so 3 accepted positions were thrown away.
    stats.record_acceptance(accepted=2, window=2, cap_trim=3)
    # Given 1, allowed 4: died on its own, the window cost nothing.
    stats.record_acceptance(accepted=1, window=4, cap_trim=0)

    assert stats.cap_trim_tokens == 3
    assert stats.accept_len == pytest.approx(1.5)
    assert stats.accept_loss_per_request == pytest.approx(1.5)
    # accept_len + accept_loss is what a no-trim run would have accepted.
    assert (stats.accept_len +
            stats.accept_loss_per_request) == pytest.approx(3.0)


def test_the_loss_is_absent_rather_than_zero_when_not_measured():
    """`compact` cannot produce this number, and must not appear to."""
    stats = _stats(mode=RaggedVerifyMode.COMPACT)
    stats.record_acceptance(accepted=3, window=3)
    assert stats.cap_trim_tokens == 0
    # The bound that IS available under compact still works.
    assert stats.trimmed_hit_ceiling == 1


def test_cap_accept_must_not_be_credited_with_a_compute_saving():
    """The regression this guards: cap-accept booking the windows as delivered.

    It submits the full block by construction. Deriving delivered tokens from
    the windows would report a trim ratio for a run that trimmed no compute at
    all -- and that ratio is the headline number for whether the feature works.
    """
    stats = _stats()
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
    stats = _stats(mode=RaggedVerifyMode.COMPACT)
    stats.record_step(num_gen_requests=4,
                      verify_lens=[5, 3, 2, 1],
                      bucket=16)
    assert stats.delivered_tokens == 16
    assert stats.trim_ratio > 0.0


# --- mode plumbing ----------------------------------------------------------


def test_cap_accept_computes_windows_but_does_not_trim_the_token_axis():
    """The two predicates that route the whole feature."""
    assert RaggedVerifyMode.CAP_ACCEPT.computes_windows
    assert not RaggedVerifyMode.CAP_ACCEPT.trims_submitted_tokens

    assert RaggedVerifyMode.COMPACT.computes_windows
    assert RaggedVerifyMode.COMPACT.trims_submitted_tokens

    assert not RaggedVerifyMode.STATIC.computes_windows
    assert not RaggedVerifyMode.STATIC.trims_submitted_tokens


def test_the_active_gate_does_not_demand_a_saving_cap_accept_never_makes():
    """The gate must stay satisfiable in the mode it exists to validate.

    `require_trim` asks "did delivered tokens come in under the no-trim
    ceiling". cap-accept answers no by construction, so keying the gate on the
    ratio alone would fail every cap-accept run -- the reference run against
    which the others are judged.
    """
    stats = _stats()
    stats.record_step(num_gen_requests=4,
                      verify_lens=[5, 3, 2, 1],
                      bucket=None,
                      delivered=4 * (1 + MAX_DRAFT_LEN))
    assert stats.trim_ratio == 0.0
    stats.assert_ragged_active(require_trim=True)


def test_the_active_gate_still_demands_a_saving_from_compact():
    """The regression the clause above must not introduce."""
    stats = _stats(mode=RaggedVerifyMode.COMPACT)
    stats.record_step(num_gen_requests=4,
                      verify_lens=[5, 3, 2, 1],
                      bucket=4 * (1 + MAX_DRAFT_LEN))
    assert stats.trim_ratio == 0.0
    with pytest.raises(AssertionError, match="nothing was saved"):
        stats.assert_ragged_active(require_trim=True)


def test_the_summary_carries_the_measurement():
    stats = _stats()
    stats.record_acceptance(accepted=2, window=2, cap_trim=3)
    summary = stats.summary()
    assert summary["mode"] == "cap-accept"
    assert summary["cap_trim_tokens"] == 3
    assert summary["accept_loss_per_request"] == pytest.approx(3.0)
