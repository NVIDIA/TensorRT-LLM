# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the offload CI gate; no model weights or GPU required."""

import copy
import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_HELPERS = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] / "integration/defs/accuracy/sparse_offload_accuracy.py"
    )
)
assert_token_parity = _HELPERS["assert_token_parity"]
assert_host_fetches = _HELPERS["assert_host_fetches"]


def _completion() -> dict:
    return {
        "id": "case",
        "prompt_sha256": "prompt",
        "token_ids": [7, 8, 9],
        "finish_reason": "stop",
        "answer": "B",
        "correct": True,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"id": "wrong-case"},
        {"prompt_sha256": "different-prompt"},
        {"token_ids": []},
        {"token_ids": [7, 8]},
        {"token_ids": [7, 8, 10]},
        {"token_ids": [8, 7, 9]},
        {"finish_reason": "length"},
        {"answer": "C"},
        {"correct": False},
    ],
)
def test_token_gate_rejects_regressions(change: dict) -> None:
    reference = _completion()
    candidate = {**reference, **change}
    with pytest.raises(AssertionError):
        assert_token_parity(reference, candidate, label="negative control")


def test_token_gate_accepts_exact_match() -> None:
    reference = _completion()
    assert_token_parity(reference, copy.deepcopy(reference), label="exact match")


def _events() -> list[dict]:
    return [
        {
            "rank": rank,
            "history_tokens": 8192,
            "selectable_units": [2048],
            "unit_tokens": 4,
            "index_topk": 512,
            "hits": 0,
            "misses": 512,
        }
        for rank in range(2)
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"rank": 0},
        {"misses": 0},
        {"misses": -1},
        {"history_tokens": 2048},
        {"selectable_units": [512]},
        {"unit_tokens": 1},
        {"index_topk": 2048},
    ],
)
def test_activation_gate_rejects_missing_or_wrong_fetches(change: dict) -> None:
    events = _events()
    events[1].update(change)
    with pytest.raises(AssertionError):
        assert_host_fetches(
            events, prompt_lengths=[8192], world_size=2, index_topk=512, unit_tokens=4
        )


@pytest.mark.parametrize("events", [[], _events()[:1], _events() + _events()])
def test_activation_gate_requires_every_rank_and_request(events: list[dict]) -> None:
    with pytest.raises(AssertionError):
        assert_host_fetches(
            events, prompt_lengths=[8192], world_size=2, index_topk=512, unit_tokens=4
        )


def test_activation_gate_rejects_short_prompt() -> None:
    with pytest.raises(AssertionError, match="activate offload"):
        assert_host_fetches(
            _events(), prompt_lengths=[2048], world_size=2, index_topk=512, unit_tokens=4
        )


def test_activation_gate_accepts_real_fetches_on_every_rank() -> None:
    assert_host_fetches(
        _events(), prompt_lengths=[8192], world_size=2, index_topk=512, unit_tokens=4
    )
