# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal accuracy-panel checks without starting a model."""

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

source = Path(__file__).parents[1] / "integration/defs/accuracy/modeling_v2/multimodal.py"
spec = importlib.util.spec_from_file_location("multimodal_accuracy", source)
multimodal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multimodal)

pytestmark = pytest.mark.cpu_only

IMG, BREAK, END = 10, 12, 13


@pytest.mark.parametrize(
    "text,expected,passed",
    [
        ("Red, blue", ["red", "blue"], True),
        ("First image: red. Second image: blue. Red, blue.", ["red", "blue"], True),
        ("A red circle on a white background.", ["red"], True),
        ("Blue, red", ["red", "blue"], False),
        ("Red, green, blue", ["red", "blue"], False),
        (
            "I can't see the images. If they are blue and red, I can name them.",
            ["blue", "red"],
            False,
        ),
        ("Second. The first was red.", ["second"], True),
        ("The first, not second.", ["second"], False),
        ("", ["red"], False),
    ],
)
def test_answer_matches_names_and_order(text, expected, passed):
    assert multimodal.answer_matches(text, expected) is passed


def test_image_spans_follow_prompt_order():
    ids = [1, 3, 7, IMG, IMG, BREAK, IMG, END, 7, IMG, END, 4]
    assert multimodal.image_spans(ids, IMG, END) == [(3, 7), (9, 10)]
    with pytest.raises(AssertionError):
        multimodal.image_spans([1, IMG, IMG, 4], IMG, END)


def test_pair_checks_separate_illegal_hits_from_missing_evidence():
    bounds = {"cold": 0, "other": 64, "warm": 199}
    assert (
        multimodal.pair_failures({"cold": 0, "other": 64, "warm": 192}, bounds, {"warm": 70}) == []
    )
    failures = multimodal.pair_failures(
        {"cold": 0, "other": 96, "warm": None}, bounds, {"warm": 70}
    )
    assert failures == [
        "FAIL: other: illegal cache hit 96 > 64",
        "INCONCLUSIVE: warm: missing cache telemetry",
    ]
    assert multimodal.pair_failures({"cold": 0, "other": 0, "warm": 64}, bounds, {"warm": 70}) == [
        "INCONCLUSIVE: warm: no warm image-block witness (64)"
    ]


def _row(seconds, rank, tokens, requests=1):
    stamp = datetime.fromtimestamp(seconds).strftime("%m-%d-%Y %H:%M:%S.%f")
    row = {
        "timestamp": stamp,
        "inflightBatchingStats": {"numContextRequests": requests, "numCtxTokens": tokens},
    }
    if rank is not None:
        row["attentionDpRank"] = rank
    return row


def test_rank_evidence_needs_the_cold_prefill_inside_each_window():
    windows = {0: (100.0, 110.0, 300), 1: (120.0, 130.0, 300)}
    rows = [_row(101.0, 0, 300), _row(121.0, 1, 299), _row(140.0, 1, 300), _row(125.0, 1, 0, 0)]
    assert multimodal.rank_evidence(rows, windows, enable_attention_dp=True) == [
        "INCONCLUSIVE: rank 1 shows no image prefill in its pinned window"
    ]
    rows.append(_row(129.0, 1, 512))
    assert multimodal.rank_evidence(rows, windows, enable_attention_dp=True) == []
    # Without attention DP the rows carry no rank: they are rank 0's.
    assert multimodal.rank_evidence([_row(101.0, None, 300)], {0: windows[0]}, False) == []
