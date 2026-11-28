# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the offload CI gate; no model weights or GPU required."""

import copy
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_HELPERS = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] / "integration/defs/accuracy/sparse_offload_accuracy.py"
    )
)
assert_token_parity = _HELPERS["assert_token_parity"]
assert_host_fetches = _HELPERS["assert_host_fetches"]
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
