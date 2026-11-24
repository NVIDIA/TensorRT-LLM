# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forced acceptance applies only to the generation rows that verify drafts the drafter made.

``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` overrides the accepted-token count of generation
rows. A row whose drafts are placeholders keeps its natural count: a request's first generation
step after a context step of a parallel-draft mode (those workers give context requests zero
drafts), or after a disaggregated handoff whose context server sent none.
``SpecMetadata.real_draft_rows`` marks the rows to force. The input preparation fills it from
each request's ``py_has_real_drafts``, which it keeps up to date step by step.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.speculative.interface import (
    FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR,
    SpeculativeDecodingMode,
    SpecWorkerBase,
)

pytestmark = pytest.mark.cpu_only

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_K = 7


class _StubSpecWorker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return _K

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _worker(monkeypatch, value) -> _StubSpecWorker:
    if value is None:
        monkeypatch.delenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, value)
    return _StubSpecWorker()


def _spec(real_rows, *, is_cuda_graph: bool = False) -> SimpleNamespace:
    rows = None
    if real_rows is not None:
        rows = torch.tensor(real_rows, dtype=torch.bool, device=_DEVICE)
    return SimpleNamespace(is_cuda_graph=is_cuda_graph, real_draft_rows=rows)


def _counts(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=_DEVICE)


def _force(worker, natural, num_contexts, spec) -> list[int]:
    out = worker._apply_force_accepted_tokens(
        _counts(natural), num_contexts, _K, spec_metadata=spec
    )
    return out.tolist()


# --- SpecWorkerBase._apply_force_accepted_tokens -----------------------------------------------


def test_a_placeholder_row_keeps_its_natural_count(monkeypatch):
    worker = _worker(monkeypatch, "2")
    # One context row, then a generation row whose drafts are placeholders: its natural count
    # stays, whether the target rejected the first placeholder (1) or matched it (2).
    assert _force(worker, [1, 1], 1, _spec([False])) == [1, 1]
    assert _force(worker, [1, 2], 1, _spec([False])) == [1, 2]


def test_a_row_with_real_drafts_is_forced(monkeypatch):
    worker = _worker(monkeypatch, "2")
    assert _force(worker, [1, 1], 1, _spec([True])) == [1, 3]
    assert _force(worker, [1, 6], 1, _spec([True])) == [1, 3]


def test_a_mixed_batch_forces_only_the_rows_with_real_drafts(monkeypatch):
    masked = _worker(monkeypatch, "2.68")
    reference = _worker(monkeypatch, "2.68")
    natural = [1, 1, 2, 1, 5, 1, 8]
    real = [False, True, False, True, True, False]
    for _ in range(8):
        out = _force(masked, natural, 1, _spec(real))
        # The same draws as forcing every row: the selection does not move the pool.
        forced = _force(reference, natural, 1, _spec(None))
        expected = [natural[0]] + [f if r else n for f, n, r in zip(forced[1:], natural[1:], real)]
        assert out == expected
        assert all(v in (3, 4) for v, r in zip(out[1:], real) if r)


def test_unmarked_rows_are_all_forced(monkeypatch):
    # A step that marks no rows (or metadata without the field) forces every generation row.
    worker = _worker(monkeypatch, "2")
    assert _force(worker, [1, 1, 1], 1, _spec(None)) == [1, 3, 3]
    assert _force(worker, [1, 1, 1], 1, SimpleNamespace(is_cuda_graph=False)) == [1, 3, 3]
    assert _force(worker, [1, 1, 1], 1, None) == [1, 3, 3]


@pytest.mark.parametrize("value", [None, "0"])
def test_without_the_setting_nothing_changes(value, monkeypatch):
    worker = _worker(monkeypatch, value)
    natural = [1, 1, 2, 5]
    assert _force(worker, natural, 1, _spec([False, True, True])) == natural
    assert _force(worker, natural, 1, _spec(None)) == natural
    assert worker._force_accept_rng_pool is None


@pytest.mark.parametrize("real", [[True, True], [False, True], None])
def test_graph_warmup_forces_nothing(real, monkeypatch):
    # An eager pass of a CUDA graph step (warmup) leaves every count as it is.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _worker(monkeypatch, "2.68")
    counter = worker._force_accept_rng_counter.clone()
    assert _force(worker, [1, 1, 2], 1, _spec(real, is_cuda_graph=True)) == [1, 1, 2]
    assert torch.equal(worker._force_accept_rng_counter, counter)


# --- The rows the input preparation marks ------------------------------------------------------


def _request(name: str, real: bool = False) -> SimpleNamespace:
    return SimpleNamespace(name=name, py_has_real_drafts=real)


def _mark(mode, draft_len, *, context=(), extend=(), generation=()) -> list[bool]:
    """One input-preparation pass; returns its rows in batch order."""
    host = SimpleNamespace(
        lists=SimpleNamespace(
            real_draft_rows=[], extend_requests=list(extend), generation_requests=list(generation)
        )
    )
    DecoderInputPreparer._mark_real_draft_rows(
        host,
        SimpleNamespace(context_requests=list(context)),
        SimpleNamespace(spec_dec_mode=mode),
        draft_len,
    )
    return host.lists.real_draft_rows


@pytest.mark.parametrize(
    "mode", [mode for mode in SpeculativeDecodingMode if mode.is_parallel_draft()]
)
def test_the_first_step_after_a_parallel_draft_context_step_is_not_forced(mode):
    request = _request("a")
    assert _mark(mode, _K, context=[request]) == []
    # The worker gave the context request zero placeholders: its first verify reads them.
    assert _mark(mode, _K, extend=[request]) == [False]
    # From then on it verifies the drafter's drafts.
    assert _mark(mode, _K, extend=[request]) == [True]
    assert _mark(mode, _K, extend=[request]) == [True]


@pytest.mark.parametrize(
    "mode",
    [
        SpeculativeDecodingMode.MTP,
        SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL,
        SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
        SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL,
        SpeculativeDecodingMode.SA,
    ],
)
def test_modes_that_draft_on_the_context_step_force_the_first_step(mode):
    request = _request("a")
    _mark(mode, _K, context=[request])
    assert _mark(mode, _K, extend=[request]) == [True]


def test_a_step_without_drafting_leaves_placeholders_for_the_next():
    request = _request("a", real=True)
    assert _mark(SpeculativeDecodingMode.MTP, 0, extend=[request]) == [True]
    # The draft length was 0: the next verify reads padding, not drafts.
    assert _mark(SpeculativeDecodingMode.MTP, _K, extend=[request]) == [False]
    assert _mark(SpeculativeDecodingMode.MTP, _K, extend=[request]) == [True]


def test_rows_follow_the_batch_order():
    steady, arrived, dummy, plain = (
        _request("steady", True),
        _request("arrived", False),
        SimpleNamespace(name="dummy"),
        _request("plain", True),
    )
    rows = _mark(
        SpeculativeDecodingMode.DFLASH,
        _K,
        context=[_request("new")],
        extend=[steady, arrived, dummy],
        generation=[plain],
    )
    # A request without the flag reads as placeholders.
    assert rows == [True, False, False, True]
    assert all(r.py_has_real_drafts for r in (steady, arrived, dummy, plain))


def test_context_requests_draft_only_outside_the_parallel_draft_modes():
    for mode in SpeculativeDecodingMode:
        assert mode.drafts_context_requests() == (not mode.is_parallel_draft())
