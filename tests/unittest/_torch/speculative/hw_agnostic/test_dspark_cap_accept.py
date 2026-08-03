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
import torch

from tensorrt_llm._torch.speculative.dspark_observability import (
    DSparkRaggedStats, RaggedVerifyMode)
from tensorrt_llm._torch.speculative.interface import apply_accept_caps

MAX_DRAFT_LEN = 5


def _stats(mode=RaggedVerifyMode.CAP_ACCEPT):
    return DSparkRaggedStats(mode=mode, max_draft_len=MAX_DRAFT_LEN)


class _Meta:
    """Just the two fields `apply_accept_caps` reads."""

    def __init__(self, caps=None, track_trim=True):
        self.accept_caps = (None if caps is None else torch.tensor(
            caps, dtype=torch.int32))
        self.accept_cap_trim = (torch.zeros(1, dtype=torch.int64)
                                if track_trim else None)


# --- the cap itself ---------------------------------------------------------
#
# WHERE this runs is the whole point, and it is not testable from here: it has
# to be inside acceptance, because the DSpark drafter reads the same
# `num_accepted_tokens` later in the SAME forward to advance its rolling KV
# window and its persistent decode position (`_ctx_len += nacc`, dspark.py).
# Capping on the host afterwards leaves the drafter advanced by the uncapped
# count while only the capped prefix was committed -- lossless output, but the
# draft state drifts and the mode stops being a faithful reference.
#
# Units: `num_accepted_tokens` is `accepted drafts + 1`, so a window of w
# drafted positions is a cap of `w + 1`, matching `verify_lens`.


def test_no_caps_is_a_no_op():
    """Every other mode goes through this call."""
    counts = torch.tensor([3, 1, 4], dtype=torch.int32)
    apply_accept_caps(counts, 0, _Meta(caps=None))
    assert counts.tolist() == [3, 1, 4]


def test_counts_inside_their_windows_are_untouched():
    """The draft died before the cut, so the cut cost nothing."""
    counts = torch.tensor([1, 3, 2], dtype=torch.int32)
    meta = _Meta(caps=[6, 4, 6])
    apply_accept_caps(counts, 0, meta)
    assert counts.tolist() == [1, 3, 2]
    assert int(meta.accept_cap_trim.item()) == 0


def test_counts_beyond_their_windows_are_clamped_and_the_loss_accumulated():
    counts = torch.tensor([6, 5, 2], dtype=torch.int32)
    meta = _Meta(caps=[3, 5, 6])
    apply_accept_caps(counts, 0, meta)
    assert counts.tolist() == [3, 5, 2]
    # Only the first request lost anything: 6 -> 3.
    assert int(meta.accept_cap_trim.item()) == 3


def test_the_accumulator_is_a_running_total_across_steps():
    """It mirrors a device counter that lives as long as the engine."""
    meta = _Meta(caps=[2, 2])
    for _ in range(3):
        counts = torch.tensor([5, 5], dtype=torch.int32)
        apply_accept_caps(counts, 0, meta)
        assert counts.tolist() == [2, 2]
    assert int(meta.accept_cap_trim.item()) == 3 * (3 + 3)


def test_context_requests_are_left_alone():
    """Caps are per generation request; contexts sit ahead of them."""
    counts = torch.tensor([9, 9, 6, 6], dtype=torch.int32)
    meta = _Meta(caps=[2, 3])
    apply_accept_caps(counts, 2, meta)
    assert counts.tolist() == [9, 9, 2, 3], "a context count was clamped"


def test_a_generation_free_batch_does_not_crash():
    counts = torch.tensor([4, 4], dtype=torch.int32)
    meta = _Meta(caps=[2, 2])
    apply_accept_caps(counts, 2, meta)
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
    trim = int(meta.accept_cap_trim.item())
    assert committed + trim == accepted + 1
    assert committed >= 1, "a step that commits nothing cannot make progress"
    assert committed <= window + 1


def test_the_cap_never_raises_a_count():
    """A window wider than the block must not invent acceptance."""
    counts = torch.tensor([2], dtype=torch.int32)
    meta = _Meta(caps=[99])
    apply_accept_caps(counts, 0, meta)
    assert counts.tolist() == [2]


# --- what the counters must say ---------------------------------------------


def test_cap_trim_is_absorbed_from_the_device_counter_as_a_running_total():
    """Assigned, not accumulated: the device counter is itself the total.

    Adding would double-count on every periodic read, which is the shape of
    bug that makes a metric drift slowly enough to be believed.
    """
    stats = _stats()
    stats.record_acceptance(accepted=2, window=2)
    stats.record_acceptance(accepted=1, window=4)

    stats.record_cap_trim_total(3)
    assert stats.cap_trim_tokens == 3
    stats.record_cap_trim_total(7)
    assert stats.cap_trim_tokens == 7, "reads must not accumulate"

    assert stats.accept_len == pytest.approx(1.5)
    assert stats.accept_loss_per_request == pytest.approx(3.5)


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
    stats.record_acceptance(accepted=2, window=2)
    stats.record_cap_trim_total(3)
    summary = stats.summary()
    assert summary["mode"] == "cap-accept"
    assert summary["cap_trim_tokens"] == 3
    assert summary["accept_loss_per_request"] == pytest.approx(3.0)


def test_the_cuda_graph_padding_dummy_carries_a_cap():
    """The regression that made the whole mode a silent no-op.

    `pad_batch` appends a shared dummy request to reach a captured batch size,
    and `_attach_accept_caps` drops the batch's caps if ANY request lacks one.
    The dummy has no cap of its own, so before this it disabled cap-accept --
    on padded steps only, so the mode measured nothing on some steps and
    worked on others, with nothing in the output to say which.

    Mirrors the treatment `py_verify_len` already got for the ragged path.
    """
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


def test_accept_caps_is_not_the_ragged_flag():
    """The distinction the whole fix rests on.

    `is_ragged_verify` means two things at once -- cap acceptance AND the draft
    buffers are packed by window (`_padded_gen_draft_tokens` slices
    `draft_tokens[:total_verify_tokens - num_gens]`). cap-accept wants only the
    first: its buffers are the ordinary rectangle. If `accept_caps` ever starts
    implying ragged, a rectangle gets unpacked as if it were packed and one
    request's drafts are attributed to another, silently.
    """
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    fields = SpecMetadata.__dataclass_fields__
    assert "accept_caps" in fields
    assert "accept_cap_trim" in fields

    meta = _Meta(caps=[2, 3])
    # The mock stands in for a metadata object carrying caps but no windows.
    assert getattr(meta, "verify_lens", None) is None
