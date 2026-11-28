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
assert_concurrent_decode = _HELPERS["assert_concurrent_decode"]
make_boundary_prompts = _HELPERS["make_boundary_prompts"]
DecodeBatchGate = _HELPERS["DecodeBatchGate"]


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


@pytest.mark.parametrize(
    "diagnostics", [{}, {"answer": None, "correct": False}, {"answer": "C", "correct": False}]
)
def test_token_gate_does_not_require_correct_or_parseable_answers(diagnostics: dict) -> None:
    reference = _completion()
    candidate = {key: value for key, value in reference.items() if key not in ("answer", "correct")}
    candidate.update(diagnostics)
    assert_token_parity(reference, candidate, label="scoring is informational")
    with pytest.raises(AssertionError, match="answers differ"):
        assert_token_parity(
            reference, candidate, label="FP8 scoring remains exact", check_answer_metadata=True
        )


def test_boundary_prompts_preserve_input_tokens_and_exceed_cache_capacity() -> None:
    prompts = [list(range(10000)), list(range(20000, 40000))]
    originals = copy.deepcopy(prompts)
    cases = make_boundary_prompts(prompts, cache_tokens=4224, tokens_per_block=128)
    assert [len(prompt) for _, prompt in cases] == [
        4479,
        4480,
        4481,
        8191,
        8192,
        8193,
        16383,
        16384,
        16385,
    ]
    assert len({case_id for case_id, _ in cases}) == len(cases)
    assert prompts == originals
    for _, prompt in cases:
        assert any(prompt == original[: len(prompt)] for original in originals)


def test_boundary_prompts_reject_incomplete_coverage() -> None:
    with pytest.raises(AssertionError, match="source prompt"):
        make_boundary_prompts([list(range(10000))], cache_tokens=4224, tokens_per_block=128)


@pytest.mark.parametrize("counts", [[], [1] * 256, [2] * 127 + [1] * 256])
def test_concurrency_gate_rejects_insufficient_overlap(counts: list[int]) -> None:
    stats = [{"inflightBatchingStats": {"numGenRequests": count}} for count in counts]
    with pytest.raises(AssertionError, match="batched decode"):
        assert_concurrent_decode(stats, min_batched_steps=128)


def test_concurrency_gate_requires_retirement_after_overlap() -> None:
    stats = [{"inflightBatchingStats": {"numGenRequests": count}} for count in [1] * 10 + [2] * 128]
    assert_concurrent_decode(stats, min_batched_steps=128)
    with pytest.raises(AssertionError, match="after one request retired"):
        assert_concurrent_decode(stats, min_batched_steps=128, require_retirement=True)
    stats.append({"inflightBatchingStats": {"numGenRequests": 1}})
    assert_concurrent_decode(stats, min_batched_steps=128, require_retirement=True)


def test_concurrency_gate_rejects_an_incomplete_first_batch() -> None:
    stats = [{"inflightBatchingStats": {"numGenRequests": count}} for count in [0, 1] + [2] * 128]
    with pytest.raises(AssertionError, match="Parity wave started with 1 of 2 requests"):
        assert_concurrent_decode(stats, min_batched_steps=128, expected_initial_batch_size=2)


def test_decode_gate_waits_orders_and_allows_retirement() -> None:
    gate = DecodeBatchGate()
    config = {"wave": 1, "expected_lengths": [24749, 4479]}
    assert gate.select(**config, request_lengths=[4479], decoding_iters=[1]) == []
    assert gate.select(**config, request_lengths=[4479, 24749], decoding_iters=[1, 1]) == [1, 0]
    assert gate.select(**config, request_lengths=[24749], decoding_iters=[257]) == [0]
    config["wave"] = 2
    assert gate.select(**config, request_lengths=[4479], decoding_iters=[1]) == []


def test_decode_gate_leaves_serial_and_empty_batches_alone() -> None:
    gate = DecodeBatchGate()
    assert gate.select(
        wave=1, expected_lengths=[], request_lengths=[4479], decoding_iters=[15]
    ) == [0]
    assert (
        gate.select(wave=2, expected_lengths=[4479, 24749], request_lengths=[], decoding_iters=[])
        == []
    )


@pytest.mark.parametrize(
    "lengths, iterations",
    [([4479], [2]), ([5000], [1]), ([4479, 4479], [1, 1]), ([4479, 24749, 8192], [1, 1, 1])],
)
def test_decode_gate_rejects_wrong_or_already_running_requests(
    lengths: list[int], iterations: list[int]
) -> None:
    with pytest.raises(AssertionError):
        DecodeBatchGate().select(
            wave=1,
            expected_lengths=[4479, 24749],
            request_lengths=lengths,
            decoding_iters=iterations,
        )


def test_decode_gate_times_out_instead_of_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = DecodeBatchGate()
    config = dict(
        wave=1, expected_lengths=[4479, 24749], request_lengths=[4479], decoding_iters=[1]
    )
    monkeypatch.setattr(_HELPERS["time"], "monotonic", lambda: 100.0)
    assert gate.select(**config) == []
    monkeypatch.setattr(_HELPERS["time"], "monotonic", lambda: 131.0)
    with pytest.raises(AssertionError, match="Timed out"):
        gate.select(**config)


def test_scheduler_gate_preserves_transfer_admission_and_request_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    control = tmp_path / "batch.json"
    control.write_text(json.dumps({"wave": 1, "prompt_lengths": [24749, 4479]}))
    monkeypatch.setenv("TLLM_TEST_SPARSE_BATCH_CONTROL", str(control))
    helpers = ModuleType("sparse_offload_accuracy")
    helpers.DecodeBatchGate = DecodeBatchGate
    monkeypatch.setitem(sys.modules, "sparse_offload_accuracy", helpers)
    observer = runpy.run_path(
        str(Path(_HELPERS["__file__"]).with_name("sparse_offload_observer.py"))
    )

    class Executor:
        def __init__(self, result: tuple) -> None:
            self.result = result

        def _schedule(self) -> tuple:
            for request in self.result[0].generation_requests:
                request.capacity += 1
                request.inflight.append(0)
            return self.result

        def _revert_unexecuted_alloc(self, batch) -> None:
            for request in batch.generation_requests:
                request.capacity -= 1 + request.inflight.pop()

    module = ModuleType("test_executor")
    module.PyExecutor = Executor
    observer["_observe_scheduler"](module)
    short = SimpleNamespace(py_orig_prompt_len=4479, py_decoding_iter=1, capacity=4479, inflight=[])
    long = SimpleNamespace(
        py_orig_prompt_len=24749, py_decoding_iter=1, capacity=24749, inflight=[]
    )
    batch = SimpleNamespace(generation_requests=[short])
    pending_transfers = [object()]
    executor = Executor((batch, pending_transfers, 2))
    for _ in range(3):
        batch.generation_requests = [short]
        result = executor._schedule()
        assert result[0] is batch and result[1] is pending_transfers and result[2] == 2
        assert batch.generation_requests == []
        assert short.capacity == 4479 and short.inflight == []
    batch.generation_requests = [short, long]
    executor._schedule()
    assert batch.generation_requests[0] is long and batch.generation_requests[1] is short
    assert short.capacity == 4480 and short.inflight == [0]
    assert long.capacity == 24750 and long.inflight == [0]


def _events() -> list[dict]:
    return [
        {
            "rank": rank,
            "request_id": 1,
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


@pytest.mark.parametrize("requests", [1, 2])
def test_activation_gate_accepts_real_fetches_on_every_rank(requests: int) -> None:
    events = [
        {**event, "request_id": request_id}
        for event in _events()
        for request_id in range(1, requests + 1)
    ]
    assert_host_fetches(
        events, prompt_lengths=[8192] * requests, world_size=2, index_topk=512, unit_tokens=4
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
@pytest.mark.parametrize("resident_only", [False, True])
def test_forced_gate_checks_each_prefix_against_ordinary_completion(
    finish_reason: str, resident_only: bool
) -> None:
    forced = {**_forced(), "token_ids": [7, 8, 12, 10, 11]}
    candidate = _forced() if resident_only else copy.deepcopy(forced)
    ordinary = {**_completion(), "finish_reason": finish_reason}
    with pytest.raises(AssertionError, match="valid prefix differs"):
        assert_forced_token_parity(
            forced,
            candidate,
            ordinary_reference=ordinary,
            ordinary_candidate=ordinary,
            label="wrong forced prefix",
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


@pytest.mark.parametrize("move_unit", [False, True])
def test_history_verifier_counts_eviction_and_refetch_per_layer(move_unit: bool) -> None:
    verifier = SparseHistoryVerifier()
    verifier.check(0, [10, 11], [0, 1], b"ab", b"ab", row_bytes=1, token_position=8)
    position = 9
    if move_unit:
        verifier.check(0, [10, 11], [2, 1], b"ab", b"ab", row_bytes=1, token_position=position)
        position += 1
    verifier.check(0, [12, 11], [0, 1], b"cb", b"cb", row_bytes=1, token_position=position)
    verifier.check(
        0,
        [10, 11],
        [2 if move_unit else 0, 1],
        b"ab",
        b"ab",
        row_bytes=1,
        token_position=position + 1,
    )
    verifier.check(1, [12, 11], [0, 1], b"cb", b"cb", row_bytes=1, token_position=8)
    assert verifier.metrics() == {
        "checked_rows": 10 if move_unit else 8,
        "mismatched_rows": 0,
        "aliased_rows": 0,
        "pool_mismatches": 0,
        "evictions": 0 if move_unit else 2,
        "refetches": 0 if move_unit else 1,
        "checked_positions": {"0": list(range(8, position + 2)), "1": [8]},
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


@pytest.mark.parametrize("binding", ["same", "same_fp8", "different_storage"])
def test_history_verifier_defers_consumer_storage_mismatch(binding: str) -> None:
    verifier = SparseHistoryVerifier()
    arena = torch.tensor([[11, 12], [21, 22]], dtype=torch.uint8)
    host_view = arena.clone().view(1, 2, 2)
    consumer_pool = arena[:, None, :]
    if binding == "different_storage":
        consumer_pool = consumer_pool.clone()
    elif binding == "same_fp8":
        consumer_pool = consumer_pool.view(torch.float8_e4m3fn)
    verifier.check_tensors(
        0,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        host_view,
        [0],
        arena,
        consumer_pool=consumer_pool,
        token_position=8,
    )
    metrics = verifier.metrics()
    assert metrics["checked_rows"] == 2
    assert metrics["mismatched_rows"] == 0
    if binding != "different_storage":
        assert metrics["pool_mismatches"] == 0
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


@pytest.mark.parametrize("stride", [(1, 2, 1), (2, 1, 1)])
def test_history_verifier_rejects_invalid_consumer_pool_layout(
    stride: tuple[int, int, int],
) -> None:
    verifier = SparseHistoryVerifier()
    arena = torch.tensor([[11, 12], [21, 22]], dtype=torch.uint8)
    with pytest.raises(
        AssertionError,
        match="layer 0, position 8: Attention consumer pool layout differs",
    ):
        verifier.check_tensors(
            0,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            arena.clone().view(1, 2, 2),
            [0],
            arena,
            consumer_pool=arena.as_strided((2, 1, 2), stride),
            token_position=8,
        )
    assert verifier.metrics()["checked_rows"] == 0


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


@pytest.mark.parametrize("second_prompt_length", [8, 12])
def test_history_gate_reconciles_complete_requests_on_every_rank(
    second_prompt_length: int,
) -> None:
    events = [
        _history_event(rank, request_id, prompt_length)
        for rank in (1, 0)
        for request_id, prompt_length in ((2, second_prompt_length), (1, 8))
    ]
    assert_history_integrity(
        events,
        prompt_lengths=[8, second_prompt_length],
        completion_lengths=[4, 4],
        world_size=2,
        require_eviction=True,
    )


@pytest.mark.parametrize("fp8_prefix_integrity", [False, True])
def test_request_identity_check_is_fp8_integrity_only(fp8_prefix_integrity: bool) -> None:
    events = [
        {
            **_history_event(rank, request_id, 8),
            "selectable_units": [2],
            "unit_tokens": 4,
            "index_topk": 1,
            "hits": 0,
            "misses": 2,
        }
        for rank in (0, 1)
        for request_id in (1, 2)
    ]
    events[-1]["request_id"] = 1
    assert_host_fetches(events, prompt_lengths=[8, 8], world_size=2, index_topk=1, unit_tokens=4)
    if fp8_prefix_integrity:
        with pytest.raises(AssertionError, match="duplicate or split request"):
            assert_history_integrity(
                events,
                prompt_lengths=[8, 8],
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
