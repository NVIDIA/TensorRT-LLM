# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which entries of an lm-eval result dict count as scores.

The dict mixes metrics with bookkeeping. ``alias`` holds the task name and
``samples`` holds the example count, neither of which is an accuracy. Treating
them as scores produced two failures in sequence on the same GSM8K run: first a
``TypeError: the resolved dtypes are not compatible with add.reduce`` from
averaging the string, then -- after that was "fixed" by keeping only numeric
entries -- a reported accuracy of 44030.98, because gsm8k's 1319 samples had
been scaled to 131900 and averaged in. The second is the dangerous one: it
cleared a reference of 96.0 and the test passed.

lm-eval keys metrics as ``"<metric>,<filter>"``, so the comma is the structural
discriminator.
"""

import pytest

from tensorrt_llm.evaluate.lm_eval import _is_metric_key


@pytest.mark.parametrize("key", [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "acc,none",
    "acc_norm,none",
    "exact_match_stderr,strict-match",
])
def test_metric_keys_are_recognized(key):
    assert _is_metric_key(key)


@pytest.mark.parametrize("key", ["alias", "samples"])
def test_bookkeeping_keys_are_rejected(key):
    assert not _is_metric_key(key)


def test_gsm8k_shaped_dict_averages_to_the_metrics_only():
    """The exact dict shape that produced 44030.98."""
    scores = {
        "alias": "gsm8k",
        "exact_match,strict-match": 96.5125,
        "exact_match_stderr,strict-match": 0.5053,
        "exact_match,flexible-extract": 96.4367,
        "exact_match_stderr,flexible-extract": 0.5106,
        "samples": 131900,  # 1319 examples, already scaled by 100
    }
    kept = [
        v for k, v in scores.items()
        if _is_metric_key(k) and "_stderr" not in k
    ]
    assert sorted(kept) == [96.4367, 96.5125]
    assert 96.0 < sum(kept) / len(kept) < 97.0
