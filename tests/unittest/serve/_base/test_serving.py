# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The serving bases and the shared termination helper in ``serve/_base/serving.py``."""

import asyncio
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve._base import serving as base_serving
from tensorrt_llm.serve._base.serving import (
    BaseServing,
    GenerateBaseServing,
    _record_generator_termination,
)

pytestmark = pytest.mark.cpu_only

# Constructor argument -> the attribute the moved code reads it as.
GENERATE_ATTRIBUTES = {
    "model": "model",
    "generator": "generator",
    "tokenizer": "tokenizer",
    "model_config": "model_config",
    "collect_perf_metrics": "_collect_perf_metrics",
    "metrics_collector": "metrics_collector",
    "adjusted_steady_clock": "_adjusted_steady_clock",
    "iteration_stats_wakeup_event": "_iteration_stats_wakeup_event",
    "iteration_stats_collector_task": "_iteration_stats_collector_task_getter",
    "internal_disagg_auth_key": "_internal_disagg_auth_key",
    "per_request_spec_decode_stats": "_per_request_spec_decode_stats",
    "spec_decode_num_spec_tokens": "_spec_decode_num_spec_tokens",
    "input_proc_executor": "_input_proc_executor",
    "postproc": "_postproc",
}
GENERATE_REQUIRED = tuple(GENERATE_ATTRIBUTES)


def _generate_dependencies():
    return {
        "model": "fake-model",
        "generator": object(),
        "tokenizer": object(),
        "model_config": None,
        "collect_perf_metrics": False,
        "metrics_collector": None,
        "adjusted_steady_clock": object(),
        "iteration_stats_wakeup_event": object(),
        "iteration_stats_collector_task": lambda: None,
        "internal_disagg_auth_key": None,
        "per_request_spec_decode_stats": False,
        "spec_decode_num_spec_tokens": None,
        "input_proc_executor": None,
        "postproc": object(),
    }


def test_base_serving_has_no_constructor_and_no_state():
    assert "__init__" not in vars(BaseServing), (
        "BaseServing has no constructor: subclasses set every attribute they read themselves"
    )
    assert vars(BaseServing()) == {}


def test_generate_base_serving_requires_each_dependency():
    dependencies = _generate_dependencies()
    serving = GenerateBaseServing(**dependencies)  # the control
    expected = {attr: dependencies[arg] for arg, attr in GENERATE_ATTRIBUTES.items()}
    # A generate serving object is never VisualGen; postproc_worker_enabled reads it.
    expected["_is_visual_gen"] = False
    assert vars(serving) == expected
    for name in GENERATE_REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            GenerateBaseServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"GenerateBaseServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )


def test_iteration_stats_collector_task_reads_the_roots_current_binding():
    # The composition root's lifespan creates the collector task after the
    # serving objects exist, so the attribute is a getter-backed property that
    # follows the root's binding instead of freezing the value at construction.
    root = SimpleNamespace(task=None)
    serving = GenerateBaseServing(
        **{**_generate_dependencies(), "iteration_stats_collector_task": lambda: root.task}
    )
    assert serving._iteration_stats_collector_task is None
    root.task = object()
    assert serving._iteration_stats_collector_task is root.task
    with pytest.raises(AttributeError):
        serving._iteration_stats_collector_task = None  # read-only: the root owns the binding


class _Request:
    """What ``await_disconnected`` reads of a starlette request."""

    def __init__(self, disconnected, no_client_connection=False):
        self._disconnected = list(disconnected)  # one answer per poll; a missing one is a test bug
        self.state = SimpleNamespace()
        if no_client_connection:
            self.state.no_client_connection = True
        self.client = ("127.0.0.1", 1234)
        self.polls = 0

    async def is_disconnected(self):
        self.polls += 1
        return self._disconnected.pop(0)


class _Promise:
    def __init__(self, finished):
        self.finished = finished
        self.request_id = 7
        self.aborted = False

    def abort(self):
        self.aborted = True


def _await_disconnected(request, promise):
    serving = GenerateBaseServing(**_generate_dependencies())
    asyncio.run(serving.await_disconnected(request, promise))


def test_await_disconnected_returns_at_once_without_a_request():
    promise = _Promise(finished=False)
    _await_disconnected(None, promise)
    assert not promise.aborted


def test_await_disconnected_returns_at_once_for_a_synthetic_request():
    # A batched request has no client socket: it must not be polled at all.
    request = _Request(disconnected=[], no_client_connection=True)
    promise = _Promise(finished=False)
    _await_disconnected(request, promise)
    assert request.polls == 0
    assert not promise.aborted


def test_await_disconnected_aborts_an_unfinished_promise_once_the_client_leaves(monkeypatch):
    sleeps = []
    real_sleep = asyncio.sleep

    async def fast_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(base_serving.asyncio, "sleep", fast_sleep)
    request = _Request(disconnected=[False, True])
    promise = _Promise(finished=False)
    _await_disconnected(request, promise)
    assert request.polls == 2
    assert sleeps == [1]  # polled once a second
    assert promise.aborted


def test_await_disconnected_leaves_a_finished_promise_alone():
    request = _Request(disconnected=[True])
    promise = _Promise(finished=True)
    _await_disconnected(request, promise)
    assert not promise.aborted


def test_record_generator_termination_blames_the_worker_behind_a_proxy(monkeypatch):
    outcomes = []
    # Patched where the moved helper looks the name up, not on the composition root.
    monkeypatch.setattr(base_serving, "record_termination_observation", outcomes.append)
    _record_generator_termination(SimpleNamespace(engine_in_worker_process=True))
    (outcome,) = outcomes
    assert outcome.termination_kind == "worker_failure"
    assert outcome.component == "engine_worker"
    assert outcome.reporting_source == "executor_proxy"
    assert outcome.exit_code_known is False
