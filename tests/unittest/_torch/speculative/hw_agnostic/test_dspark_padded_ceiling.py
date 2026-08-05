# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trim_ratio`` must not report a saving the run did not make -- or a loss.

Four runs reported a NEGATIVE ``trim_ratio`` (jobs 2560992 -0.001, 2561519
-0.0438, 2563581 -0.0082), which reads as "the ragged path delivered more tokens
than a no-trim run". It did not. ``delivered_tokens`` books the fitted bucket,
which is ``padded_bs * (tier + 1)`` where ``padded_bs`` is the CUDA-graph-rounded
maximum row count *across attention-DP ranks*; ``ceiling_tokens`` booked
``num_gen_requests * (1 + block)`` over this rank's real, unpadded requests. Two
different row bases in one ratio.

The gate reads that ratio: ``assert_ragged_active`` fires on ``trim_ratio <= 0``
with the message "delivered as many tokens as a no-trim run, so nothing was
saved" -- the opposite of what a negative value means, so the defect also
misdiagnoses itself.
"""

import pytest

from tensorrt_llm._torch.speculative.dspark_observability import (
    DSparkRaggedStats, RaggedVerifyMode)


def _stats(block_size: int = 5) -> DSparkRaggedStats:
    return DSparkRaggedStats(mode=RaggedVerifyMode.COMPACT,
                             max_draft_len=block_size)


def test_padding_rows_do_not_make_the_trim_look_negative():
    """A top-tier step on padded rows saved nothing -- and lost nothing."""
    stats = _stats(block_size=5)
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
    stats = _stats(block_size=5)
    # 4 rows, no padding, every window trimmed to 2 -> bucket 4*3 = 12 against a
    # ceiling of 4*6 = 24.
    stats.record_step(num_gen_requests=4, verify_lens=[2, 2, 2, 2],
                      bucket=4 * 3, padded_bs=4)
    assert stats.summary()["trim_ratio"] == pytest.approx(0.5)


def test_paths_without_a_bucket_keep_the_local_row_base():
    """Only the bucket path rebases; widening the others understates the trim.

    The no-window path and the cap-accept path each book ``delivered`` over the
    real rows, so pulling their ceiling up to the padded count would invent a
    saving out of padding that those paths never submitted.
    """
    no_window = _stats(block_size=5)
    no_window.record_step(num_gen_requests=3, verify_lens=None, padded_bs=8)
    assert no_window.summary()["trim_ratio"] == pytest.approx(0.0)

    explicit = _stats(block_size=5)
    # cap-accept: windows are computed but the full block is submitted anyway.
    explicit.record_step(num_gen_requests=3, verify_lens=[2, 3, 1],
                         bucket=8 * 6, padded_bs=8, delivered=3 * 6)
    assert explicit.summary()["trim_ratio"] == pytest.approx(0.0)
