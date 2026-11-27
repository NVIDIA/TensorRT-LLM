# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the offload CI gate; no model weights or GPU required."""

import copy
import runpy
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_HELPERS = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] / "integration/defs/accuracy/sparse_offload_accuracy.py"
    )
)
assert_token_parity = _HELPERS["assert_token_parity"]
assert_host_fetches = _HELPERS["assert_host_fetches"]
assert_forced_token_parity = _HELPERS["assert_forced_token_parity"]
assert_history_integrity = _HELPERS["assert_history_integrity"]
SparseHistoryVerifier = _HELPERS["SparseHistoryVerifier"]


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


def _forced() -> dict:
    return {**_completion(), "token_ids": [7, 8, 9, 10, 11], "finish_reason": "length"}


def test_forced_gate_allows_only_post_stop_tokens() -> None:
    reference = _forced()
    candidate = {**reference, "token_ids": [7, 8, 9, 12, 13]}
    assert (
        assert_forced_token_parity(
            reference,
            candidate,
            ordinary_reference=_completion(),
            ordinary_candidate=_completion(),
            label="forced",
        )
        == 3
    )
    assert (
        assert_forced_token_parity(
            reference,
            copy.deepcopy(reference),
            ordinary_reference=_completion(),
            ordinary_candidate=_completion(),
            label="identical",
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        {"token_ids": [7, 8, 12, 10, 11]},
        {"token_ids": [7, 8, 9, 10]},
        {"id": "other"},
        {"prompt_sha256": "other"},
        {"finish_reason": "stop"},
        {"answer": "C"},
        {"correct": False},
    ],
)
def test_forced_gate_preserves_prefix_and_metadata(change: dict) -> None:
    with pytest.raises(AssertionError):
        assert_forced_token_parity(
            _forced(),
            {**_forced(), **change},
            ordinary_reference=_completion(),
            ordinary_candidate=_completion(),
            label="invalid forced",
        )


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_forced_gate_checks_each_prefix_against_ordinary_completion(finish_reason: str) -> None:
    forced = {**_forced(), "token_ids": [7, 8, 12, 10, 11]}
    ordinary = {**_completion(), "finish_reason": finish_reason}
    with pytest.raises(AssertionError, match="valid prefix"):
        assert_forced_token_parity(
            forced,
            copy.deepcopy(forced),
            ordinary_reference=ordinary,
            ordinary_candidate=ordinary,
            label="wrong shared prefix",
        )


def test_forced_gate_keeps_full_equality_without_an_ordinary_stop() -> None:
    ordinary = {**_completion(), "finish_reason": "length"}
    assert (
        assert_forced_token_parity(
            _forced(),
            _forced(),
            ordinary_reference=ordinary,
            ordinary_candidate=ordinary,
            label="matching no stop",
        )
        is None
    )
    with pytest.raises(AssertionError, match="first different token"):
        assert_forced_token_parity(
            _forced(),
            {**_forced(), "token_ids": [7, 8, 9, 12, 13]},
            ordinary_reference=ordinary,
            ordinary_candidate=ordinary,
            label="no stop",
        )


def test_forced_gate_reports_suffix_before_metadata_failure(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(AssertionError, match="extracted answers differ"):
        assert_forced_token_parity(
            _forced(),
            {**_forced(), "token_ids": [7, 8, 9, 12, 13], "answer": "C"},
            ordinary_reference=_completion(),
            ordinary_candidate=_completion(),
            label="suffix metadata",
        )
    diagnostic = capfd.readouterr().out
    assert "First post-stop token diff case:" in diagnostic
    assert "'offset': 3" in diagnostic
    assert "'resident_token': 10" in diagnostic
    assert "'offload_token': 12" in diagnostic


def test_history_verifier_rejects_injected_wrong_slots() -> None:
    host_view = torch.tensor([[[11, 12], [21, 22]], [[31, 32], [41, 42]]], dtype=torch.uint8)
    host_pages = [1, 0]
    arena = torch.tensor([[21, 22], [91, 92], [31, 32]], dtype=torch.uint8)
    units = torch.tensor([0, 3], dtype=torch.int64)
    slots = torch.tensor([2, 0], dtype=torch.int64)
    original_host, original_arena, original_units = host_view.clone(), arena.clone(), units.clone()
    verifier = SparseHistoryVerifier()
    verifier.check_tensors(0, units, slots, host_view, host_pages, arena, token_position=8)
    with pytest.raises(AssertionError, match="Offload history row mismatch"):
        verifier.check_tensors(
            0, units, slots.roll(1), host_view, host_pages, arena, token_position=9
        )
    assert verifier.mismatched_rows == 2
    assert torch.equal(host_view, original_host)
    assert torch.equal(arena, original_arena)
    assert torch.equal(units, original_units)


def test_history_verifier_counts_eviction_and_refetch_per_layer() -> None:
    verifier = SparseHistoryVerifier()
    verifier.check(0, [10, 11], [0, 1], b"ab", b"ab", row_bytes=1, token_position=8)
    verifier.check(0, [12, 11], [0, 1], b"cb", b"cb", row_bytes=1, token_position=9)
    verifier.check(0, [10, 11], [0, 1], b"ab", b"ab", row_bytes=1, token_position=10)
    verifier.check(1, [12, 11], [0, 1], b"cb", b"cb", row_bytes=1, token_position=8)
    assert verifier.metrics() == {
        "checked_rows": 8,
        "mismatched_rows": 0,
        "evictions": 2,
        "refetches": 1,
        "checked_positions": {"0": [8, 9, 10], "1": [8]},
    }


def test_history_verifier_rejects_shared_physical_slot() -> None:
    with pytest.raises(AssertionError, match="share an attention slot"):
        SparseHistoryVerifier().check(
            0, [10, 11], [0, 0], b"aa", b"aa", row_bytes=1, token_position=8
        )


def _integrity_event(metrics: dict | None) -> dict:
    return {
        "rank": 0,
        "request_id": 1,
        "history_tokens": 8,
        "offload_layers": [0],
        "integrity": metrics,
    }


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        {"checked_rows": 0, "mismatched_rows": 0, "evictions": 0, "refetches": 0},
        {"checked_rows": 1, "mismatched_rows": 1, "evictions": 1, "refetches": 1},
        {"checked_rows": 1, "mismatched_rows": 0, "evictions": 0, "refetches": 1},
        {"checked_rows": 1, "mismatched_rows": 0, "evictions": 1, "refetches": 0},
    ],
)
def test_history_gate_requires_checked_evicted_and_refetched_data(metrics: dict | None) -> None:
    if metrics is not None:
        metrics = {**metrics, "checked_positions": {"0": [8]}}
    with pytest.raises(AssertionError):
        assert_history_integrity(
            [_integrity_event(metrics)],
            prompt_lengths=[8],
            completion_lengths=[2],
            world_size=1,
            require_eviction=True,
        )


def test_history_gate_does_not_require_eviction_from_short_completions() -> None:
    assert_history_integrity(
        [
            _integrity_event(
                {
                    "checked_rows": 1,
                    "mismatched_rows": 0,
                    "evictions": 0,
                    "refetches": 0,
                    "checked_positions": {"0": [8]},
                }
            )
        ],
        prompt_lengths=[8],
        completion_lengths=[2],
        world_size=1,
        require_eviction=False,
    )


def _history_event(
    rank: int,
    request_id: int,
    prompt_length: int,
    *,
    omitted: tuple[int, int | None] | None = None,
) -> dict:
    verifier = SparseHistoryVerifier()
    for layer in (0, 2):
        for offset, (unit, value) in enumerate(((10, b"a"), (12, b"b"), (10, b"a"))):
            if omitted is not None and layer == omitted[0] and omitted[1] in (None, offset):
                continue
            verifier.check(
                layer, [unit], [0], value, value, row_bytes=1, token_position=prompt_length + offset
            )
    return {
        "rank": rank,
        "request_id": request_id,
        "history_tokens": prompt_length,
        "offload_layers": [0, 2],
        "integrity": verifier.metrics(),
    }


def test_history_gate_reconciles_complete_requests_on_every_rank() -> None:
    events = [
        _history_event(rank, request_id, prompt_length)
        for rank in (1, 0)
        for request_id, prompt_length in ((2, 12), (1, 8))
    ]
    assert_history_integrity(
        events,
        prompt_lengths=[8, 12],
        completion_lengths=[4, 4],
        world_size=2,
        require_eviction=True,
    )


@pytest.mark.parametrize(
    ("omitted", "message"),
    [((2, None), "incomplete offload layers"), ((2, 1), "incomplete decode steps")],
)
def test_history_gate_requires_every_request_layer_and_decode_step(
    omitted: tuple[int, int | None], message: str
) -> None:
    events = [
        _history_event(
            rank,
            request_id,
            prompt_length,
            omitted=omitted if rank == 1 and request_id == 2 else None,
        )
        for rank in (0, 1)
        for request_id, prompt_length in ((1, 8), (2, 12))
    ]
    assert all(event["integrity"]["evictions"] > 0 for event in events)
    assert all(event["integrity"]["refetches"] > 0 for event in events)
    with pytest.raises(AssertionError, match=message):
        assert_history_integrity(
            events,
            prompt_lengths=[8, 12],
            completion_lengths=[4, 4],
            world_size=2,
            require_eviction=True,
        )
