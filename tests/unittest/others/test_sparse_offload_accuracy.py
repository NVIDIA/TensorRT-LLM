# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the offload CI gate; no model weights or GPU required."""

import copy
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    ("change", "message"),
    [
        ({"token_ids": [7, 8, 12, 10, 11]}, "valid prefix differs"),
        ({"token_ids": [7, 8, 9, 10]}, "forced completion lengths differ"),
        ({"id": "other"}, "ordinary sample ID differs"),
        ({"prompt_sha256": "other"}, "ordinary tokenized prompt differs"),
        ({"finish_reason": "stop"}, "finish reasons differ"),
        ({"answer": "C"}, "extracted answers differ"),
        ({"correct": False}, "answer correctness differs"),
    ],
)
def test_forced_gate_preserves_prefix_and_metadata(change: dict, message: str) -> None:
    reference = _forced()
    candidate = {**reference, **change}
    if "id" in change or "prompt_sha256" in change:
        reference = copy.deepcopy(candidate)
    with pytest.raises(AssertionError, match=message):
        assert_forced_token_parity(
            reference,
            candidate,
            ordinary_reference=_completion(),
            ordinary_candidate=_completion(),
            label="invalid forced",
        )


@pytest.mark.parametrize("tokens", [[], [7, 8]])
def test_forced_gate_rejects_empty_or_short_shared_rows(tokens: list[int]) -> None:
    forced = {**_forced(), "token_ids": tokens}
    with pytest.raises(AssertionError, match="empty or shorter than the ordinary"):
        assert_forced_token_parity(
            forced,
            copy.deepcopy(forced),
            ordinary_reference=_completion(),
            ordinary_candidate=_completion(),
            label="short forced",
        )


def test_forced_gate_allows_suffix_derived_answer_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm.evaluate import interface, longbench_v2

    monkeypatch.setattr(longbench_v2, "load_dataset", lambda *arguments, **keywords: [])
    with monkeypatch.context() as setup:
        setup.setattr(interface.random, "seed", lambda seed: None)
        setup.setattr(interface.np.random, "seed", lambda seed: None)
        setup.setattr(interface.torch, "manual_seed", lambda seed: None)
        evaluator = longbench_v2.LongBenchV2(length="short")
    prefix = "The correct answer is (B)."
    reference_answer = evaluator._extract_answer(
        evaluator._post_process(prefix + "</think>The correct answer is (B).")
    )
    candidate_answer = evaluator._extract_answer(
        evaluator._post_process(prefix + "</think>The correct answer is (C).")
    )
    assert reference_answer == "B" and candidate_answer == "C"
    reference = {**_forced(), "answer": reference_answer, "correct": True}
    candidate = {
        **_forced(),
        "token_ids": [7, 8, 9, 12, 13],
        "answer": candidate_answer,
        "correct": False,
    }
    assert (
        assert_forced_token_parity(
            reference,
            candidate,
            ordinary_reference=_completion(),
            ordinary_candidate=_completion(),
            label="post-stop answer",
        )
        == 3
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
    with pytest.raises(AssertionError, match="finish reasons differ"):
        assert_forced_token_parity(
            _forced(),
            {**_forced(), "token_ids": [7, 8, 9, 12, 13], "finish_reason": "stop"},
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
    verifier.check_tensors(
        0,
        units,
        slots,
        host_view,
        host_pages,
        arena,
        consumer_pool=arena[:, None, :],
        token_position=8,
    )
    verifier.check_tensors(
        0,
        units,
        slots.roll(1),
        host_view,
        host_pages,
        arena,
        consumer_pool=arena[:, None, :],
        token_position=9,
    )
    assert verifier.mismatched_rows == 2
    assert verifier.metrics()["first_mismatch"] == {"layer": 0, "unit": 0, "slot": 0, "position": 9}
    with pytest.raises(AssertionError, match="rank 0, request 1: Offload history row mismatch"):
        assert_history_integrity(
            [_integrity_event(verifier.metrics())],
            prompt_lengths=[8],
            completion_lengths=[3],
            world_size=1,
            require_eviction=False,
        )
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
        "aliased_rows": 0,
        "pool_mismatches": 0,
        "evictions": 2,
        "refetches": 1,
        "checked_positions": {"0": [8, 9, 10], "1": [8]},
        "first_mismatch": None,
        "first_alias": None,
        "first_pool_mismatch": None,
    }


@pytest.mark.parametrize("expected", [b"ab", b"aa"])
def test_history_verifier_defers_shared_physical_slot(expected: bytes) -> None:
    verifier = SparseHistoryVerifier()
    verifier.check(
        0,
        [0, 1],
        [0, 0],
        expected,
        expected[:1] * 2,
        row_bytes=1,
        token_position=8,
    )
    metrics = verifier.metrics()
    assert metrics["checked_rows"] == 2
    assert metrics["aliased_rows"] == 1
    assert metrics["mismatched_rows"] == int(expected[0] != expected[1])
    assert metrics["first_alias"] == {
        "layer": 0,
        "unit": 1,
        "previous_unit": 0,
        "slot": 0,
        "position": 8,
    }
    with pytest.raises(AssertionError, match="rank 0, request 1: Different history units share"):
        assert_history_integrity(
            [_integrity_event(metrics)],
            prompt_lengths=[8],
            completion_lengths=[2],
            world_size=1,
            require_eviction=False,
        )


@pytest.mark.parametrize("binding", ["same", "different_storage", "wrong_stride"])
def test_history_observer_checks_consumer_pool_binding(
    binding: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_dir = Path(__file__).resolve().parents[2] / "integration/defs/accuracy"
    monkeypatch.syspath_prepend(str(helper_dir))
    integrity_helpers = runpy.run_path(str(helper_dir / "sparse_offload_integrity.py"))

    arena = torch.tensor([[11, 12], [21, 22]], dtype=torch.uint8)
    host_view = arena.clone().view(1, 2, 2)
    consumer_pool = arena[:, None, :]
    if binding == "different_storage":
        consumer_pool = consumer_pool.flip(0)
    elif binding == "wrong_stride":
        consumer_pool = arena.as_strided((2, 1, 2), (1, 2, 1))
    views = {host_view.data_ptr(): host_view, arena.data_ptr(): arena}
    utils = ModuleType("tensorrt_llm._utils")
    monkeypatch.setattr(utils, "TensorWrapper", lambda pointer, *args: pointer, raising=False)
    monkeypatch.setattr(utils, "convert_to_torch_tensor", views.__getitem__, raising=False)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._utils", utils)
    descriptor = SimpleNamespace(
        data_buffer_index=0,
        buffer_ids=[1],
        unit_tokens=4,
        arena_base_addresses=[arena.data_ptr()],
        arena_row_counts=[2],
    )
    host = SimpleNamespace(
        size=4,
        stride=4,
        base=host_view.data_ptr(),
        layer_group_id=0,
        buffers=[SimpleNamespace(id=1, expansion=1)],
    )
    cache = SimpleNamespace(history_length=8, get_aggregated_page_indices=lambda group: [0])
    manager = SimpleNamespace(
        tokens_per_block=8,
        _sparse_batch_members={1: SimpleNamespace(row=0, cache=cache)},
        impl=SimpleNamespace(get_aggregated_pages=lambda buffers: [host]),
        get_sparse_compress_buffers=lambda layer: (consumer_pool, None),
    )
    metadata = SimpleNamespace(
        sparse_offload_state=SimpleNamespace(
            layers={0: descriptor},
            prepared=True,
            batch=object(),
            batch_rows=torch.tensor([0]),
            active_request_count=torch.tensor([1]),
        ),
        kv_cache_manager=manager,
        request_ids=[1],
        num_contexts=0,
        num_tokens=1,
        max_compressed_indices={4: 2},
        req_idx_per_token=torch.tensor([0]),
        token_positions_cuda=torch.tensor([8]),
        compressed_kv_lens_cuda={4: torch.tensor([2])},
    )
    attention = SimpleNamespace(
        layer_idx=0,
        compress_ratio=4,
        _uses_nvfp4_compress=False,
        head_dim=2,
        sparse_attention_config=SimpleNamespace(window_size=2),
    )
    forward_args = SimpleNamespace(
        sparse_runtime_params=SimpleNamespace(aux_kv_cache_pool_ptr=arena.data_ptr()),
        sparse_backend_args=SimpleNamespace(topk_indices=torch.tensor([[0, 1]])),
    )
    try:
        integrity_helpers["_verify_generation"](
            attention, metadata, forward_args, torch.tensor([[0, 1, 0, 1]])
        )
    finally:
        metrics = integrity_helpers["finish_request"](manager, 1, cache)
    assert metrics["checked_rows"] == 2
    assert metrics["mismatched_rows"] == 0
    if binding == "same":
        assert metrics.get("pool_mismatches", 0) == 0
        assert_history_integrity(
            [_integrity_event(metrics)],
            prompt_lengths=[8],
            completion_lengths=[2],
            world_size=1,
            require_eviction=False,
        )
    else:
        with pytest.raises(
            AssertionError, match="rank 0, request 1: Attention consumer pool differs"
        ):
            assert_history_integrity(
                [_integrity_event(metrics)],
                prompt_lengths=[8],
                completion_lengths=[2],
                world_size=1,
                require_eviction=False,
            )
        assert metrics["pool_mismatches"] == 1
        assert metrics["first_pool_mismatch"]["layer"] == 0
        assert metrics["first_pool_mismatch"]["position"] == 8


def _integrity_event(metrics: dict | None) -> dict:
    return {
        "rank": 0,
        "request_id": 1,
        "history_tokens": 8,
        "offload_layers": [0],
        "integrity": metrics,
    }


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (None, "integrity check did not run"),
        (
            {"checked_rows": 0, "mismatched_rows": 0, "evictions": 1, "refetches": 1},
            "integrity check did not run",
        ),
        (
            {"checked_rows": 1, "mismatched_rows": 1, "evictions": 1, "refetches": 1},
            "Offload history row mismatch",
        ),
        (
            {"checked_rows": 1, "mismatched_rows": 0, "evictions": 0, "refetches": 1},
            "No observed sparse-cache eviction",
        ),
        (
            {"checked_rows": 1, "mismatched_rows": 0, "evictions": 1, "refetches": 0},
            "No observed sparse-history re-fetch",
        ),
    ],
)
def test_history_gate_requires_checked_evicted_and_refetched_data(
    metrics: dict | None, message: str
) -> None:
    if metrics is not None:
        metrics = {
            **metrics,
            "aliased_rows": 0,
            "pool_mismatches": 0,
            "checked_positions": {"0": [8]},
        }
    with pytest.raises(AssertionError, match=message):
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
                    "aliased_rows": 0,
                    "pool_mismatches": 0,
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
