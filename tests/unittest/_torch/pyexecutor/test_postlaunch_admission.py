# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-launch admission on an attention-DP generation server.

Covers PyExecutor._init_postlaunch_admit gating, the staging of routed
generation-only items in _fetch_new_requests, _run_postlaunch_admission
(build, validation, input-copy wait before gen-init preparation, no collective)
and the shutdown/control flush in _fetch_and_activate_new_requests.
"""

import types
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import FinishReason, RequestType

pytestmark = pytest.mark.cpu_only

GEN = RequestType.REQUEST_TYPE_GENERATION_ONLY
CTX = RequestType.REQUEST_TYPE_CONTEXT_ONLY
ENV_NAMES = (
    "TRTLLM_DISABLE_POSTLAUNCH_ADMISSION",
    "TRTLLM_DISABLE_POSTLAUNCH_ADMISSION_BUILD",
    "TRTLLM_DISABLE_POSTLAUNCH_ADMISSION_GENPREP",
    "TRTLLM_POSTLAUNCH_ADMISSION_MIN_TOKENS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _item(rid, request_type=GEN, prompt_len=8):
    return types.SimpleNamespace(
        id=rid,
        request=types.SimpleNamespace(request_type=request_type, num_input_tokens=prompt_len),
        child_req_ids=None,
    )


class _Built:
    def __init__(self, item):
        self.py_request_id = item.id
        self.prompt_len = item.request.num_input_tokens


class _StrictDist:
    """Records every attribute access other than the plain rank fields."""

    plain = {"rank", "tp_rank", "pp_size", "cp_size", "cp_rank", "cp_config", "tp_size"}

    def __init__(self, rank=0):
        self.__dict__["calls"] = []
        self.__dict__["rank"] = rank
        self.__dict__["tp_rank"] = rank
        self.__dict__["tp_size"] = 8
        self.__dict__["pp_size"] = 1
        self.__dict__["cp_size"] = 1
        self.__dict__["cp_rank"] = 0
        self.__dict__["cp_config"] = {}

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append(name)
            return MagicMock()

        return call


def _executor(*, build=True, genprep=True, in_loop=True, rank=0):
    ex = object.__new__(PyExecutor)
    ex.dist = _StrictDist(rank)
    ex.active_requests = []
    ex.iter_counter = 7
    ex.is_shutdown = False
    ex.control_requests = []
    ex._postlaunch_items = []
    ex._postlaunch_gen_init = []
    ex.enable_iter_perf_stats = False
    ex._pending_num_new_active_requests = 0
    ex._postlaunch_build = build
    ex._postlaunch_genprep = genprep
    ex._postlaunch_in_loop = in_loop
    ex._postlaunch_min_tokens = 0
    ex.should_exclude_last_generation_logits = False
    ex._num_embeddings = None
    # Off, unlike the merge default, so the staged build is seen to pass it on.
    ex.track_unique_tokens = False
    ex._mm_encoder_item_scheduling_enabled = False
    ex._validate_request = Mock()
    ex._handle_errors = Mock()
    ex._disagg_coordinator = Mock()
    ex.kv_cache_transceiver = None
    ex.model_engine = Mock()
    return ex


@pytest.fixture
def built_by_merge(monkeypatch):
    calls = []

    def fake_merge(items, **kwargs):
        assert kwargs["track_unique_tokens"] is False
        calls.append([item.id for item in items])
        return [_Built(item) for item in items]

    monkeypatch.setattr(py_executor_module, "merge_requests", fake_merge)
    return calls


# --------------------------------------------------------------------------- gating


def _init_executor(
    *, adp=True, transceiver=True, need_sync=False, benchmark=False, pp=1, opted_in=True
):
    ex = object.__new__(PyExecutor)
    if opted_in:
        ex._postlaunch_admission_opted_in = True
    ex.kv_cache_transceiver = (
        types.SimpleNamespace(_gen_need_sync=need_sync) if transceiver else None
    )
    ex.enable_attention_dp = adp
    ex.is_benchmark_disagg = benchmark
    ex.dist = types.SimpleNamespace(rank=1, pp_size=pp, cp_size=1)
    ex._init_postlaunch_admit()
    return ex


@pytest.mark.parametrize(
    "kwargs, env, build, genprep",
    [
        pytest.param({}, {}, True, True, id="adp_generation_server"),
        # The class default stands for an executor whose model sets no RuntimeHooks opt-in.
        pytest.param(dict(opted_in=False), {}, False, False, id="not_opted_in"),
        pytest.param(dict(adp=False), {}, False, False, id="no_adp"),
        pytest.param(dict(transceiver=False), {}, False, False, id="no_transceiver"),
        pytest.param(dict(pp=2), {}, False, False, id="pipeline_parallel"),
        # Gen-init preparation needs a rank-local receive poll.
        pytest.param(dict(need_sync=True), {}, True, False, id="need_sync"),
        pytest.param(
            {}, {"TRTLLM_DISABLE_POSTLAUNCH_ADMISSION_BUILD": "1"}, False, True, id="kill_build"
        ),
        pytest.param(
            {}, {"TRTLLM_DISABLE_POSTLAUNCH_ADMISSION_GENPREP": "1"}, True, False, id="kill_genprep"
        ),
    ],
)
def test_init_gating(monkeypatch, kwargs, env, build, genprep):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    ex = _init_executor(**kwargs)
    assert (ex._postlaunch_build, ex._postlaunch_genprep) == (build, genprep)


# --------------------------------------------------------------------------- staging


def test_stage_keeps_generation_only_items_and_builds_the_rest_now():
    ex = _executor()
    items = [_item(1), _item(2, CTX), _item(3)]
    assert [i.id for i in ex._stage_postlaunch_items(items)] == [2]
    assert [i.id for i in ex._postlaunch_items] == [1, 3]


@pytest.mark.parametrize("build, in_loop", [(False, True), (True, False)])
def test_stage_inactive(build, in_loop):
    ex = _executor(build=build, in_loop=in_loop)
    items = [_item(1), _item(2)]
    assert ex._stage_postlaunch_items(items) is items
    assert ex._postlaunch_items == []


# --------------------------------------------------------------------------- post-launch run


def test_postlaunch_builds_then_waits_then_prepares_without_collectives(monkeypatch):
    ex = _executor()
    order = []
    ex.model_engine.wait_for_input_copy = Mock(side_effect=lambda: order.append("wait"))
    ex.disagg.receive_gen_init = Mock(side_effect=lambda reqs: order.append("prep"))
    ex._stage_postlaunch_items([_item(1), _item(2)])
    gen_init = [
        types.SimpleNamespace(
            prompt_len=5, state=LlmRequestState.DISAGG_GENERATION_INIT, is_finished=False
        )
    ]
    ex._postlaunch_gen_init.extend(gen_init)

    def merge_spy(items, **kwargs):
        order.append("build")
        return [_Built(i) for i in items]

    monkeypatch.setattr(py_executor_module, "merge_requests", merge_spy)
    ex._run_postlaunch_admission()

    assert order == ["build", "wait", "prep"]
    assert [r.py_request_id for r in ex.active_requests] == [1, 2]
    ex.disagg.receive_gen_init.assert_called_once_with(gen_init)
    assert ex._postlaunch_items == [] and ex._postlaunch_gen_init == []
    assert ex.dist.calls == []


def test_postlaunch_build_only_does_not_wait(built_by_merge):
    ex = _executor()
    ex.model_engine.wait_for_input_copy = Mock()
    ex._stage_postlaunch_items([_item(4)])
    ex._run_postlaunch_admission()
    ex.model_engine.wait_for_input_copy.assert_not_called()
    ex.disagg.receive_gen_init.assert_not_called()
    assert built_by_merge == [[4]]
    assert ex.dist.calls == []


def test_postlaunch_invalid_request_fails_once_and_is_not_activated(built_by_merge):
    ex = _executor()

    def validate(request):
        if request.py_request_id == 2:
            raise ValueError("bad request")

    ex._validate_request = Mock(side_effect=validate)
    ex._stage_postlaunch_items([_item(1), _item(2), _item(3)])
    ex._run_postlaunch_admission()
    assert [r.py_request_id for r in ex.active_requests] == [1, 3]
    ex._handle_errors.assert_called_once()
    assert ex._handle_errors.call_args.kwargs["requests"][0].py_request_id == 2


def test_postlaunch_skips_a_gen_init_request_a_failed_step_completed(built_by_merge):
    """_handle_errors after a tolerated forward error completes the parked request."""
    ex = _executor()
    ex.model_engine.wait_for_input_copy = Mock()
    live = types.SimpleNamespace(
        prompt_len=5, state=LlmRequestState.DISAGG_GENERATION_INIT, is_finished=False
    )
    failed = types.SimpleNamespace(
        prompt_len=7, state=LlmRequestState.GENERATION_COMPLETE, is_finished=True
    )
    ex._postlaunch_gen_init.extend([failed, live])

    ex._run_postlaunch_admission()

    ex.disagg.receive_gen_init.assert_called_once_with([live])
    assert ex._postlaunch_gen_init == []


def test_postlaunch_failed_only_receives_nothing(built_by_merge):
    ex = _executor()
    ex.model_engine.wait_for_input_copy = Mock()
    ex._postlaunch_gen_init.append(
        types.SimpleNamespace(
            prompt_len=7, state=LlmRequestState.GENERATION_COMPLETE, is_finished=True
        )
    )
    ex._run_postlaunch_admission()
    ex.disagg.receive_gen_init.assert_not_called()


def test_postlaunch_counts_built_valid_requests_once(built_by_merge):
    ex = _executor()
    ex.enable_iter_perf_stats = True

    def validate(request):
        if request.py_request_id == 2:
            raise ValueError("bad request")

    ex._validate_request = Mock(side_effect=validate)
    ex._stage_postlaunch_items([_item(1), _item(2), _item(3)])
    assert ex._pending_num_new_active_requests == 0  # staging counts nothing
    ex._run_postlaunch_admission()
    assert ex._pending_num_new_active_requests == 2


# --------------------------------------------------------------------------- fetch integration


def _fetch_executor(ex, routed_items):
    """Stand in for the routed fetch: stage, then build what is left now."""

    def fetch(waiting_queue, active_requests):
        items = ex._stage_postlaunch_items(list(routed_items))
        return py_executor_module.merge_requests(items, track_unique_tokens=ex.track_unique_tokens)

    ex.waiting_queue = None
    ex._fetch_new_requests = fetch
    ex.scheduler = None
    return ex


def test_activate_counts_staged_and_flushes_on_shutdown(built_by_merge):
    ex = _fetch_executor(_executor(), [_item(1), _item(2)])
    activated = ex._fetch_and_activate_new_requests()
    assert activated == []
    assert [i.id for i in ex._postlaunch_items] == [1, 2]

    ex._postlaunch_items = []
    _fetch_executor(ex, [_item(3)])
    ex.is_shutdown = True
    activated = ex._fetch_and_activate_new_requests()
    assert [r.py_request_id for r in activated] == [3]
    assert [r.py_request_id for r in ex.active_requests] == [3]
    assert ex._postlaunch_items == []


def test_activate_flushes_when_a_control_request_arrived(built_by_merge):
    ex = _fetch_executor(_executor(), [_item(5)])
    ex.control_requests = [object()]
    activated = ex._fetch_and_activate_new_requests()
    assert [r.py_request_id for r in activated] == [5]
    assert ex._postlaunch_items == []


@pytest.mark.parametrize("staged", [True, False])
def test_cancellation_of_a_staged_item_outlives_the_fetch(staged):
    """A staged item's cancel id is kept for the pass that sees the built request."""
    ex = PyExecutor.__new__(PyExecutor)
    ex.kv_cache_transceiver = types.SimpleNamespace(
        supports_idle_request_cancellation=lambda request=None: True
    )
    ex.canceled_req_ids = [7]
    ex.waiting_queue = Mock()
    ex._cancelled_disagg_model_request_ids = lambda: set()
    ex.previous_batch = None
    ex.active_requests = []
    ex._postlaunch_items = [_item(7)] if staged else []

    assert ex._cancel_waiting_disagg_requests([]) == []
    ex.waiting_queue.remove_by_ids.assert_called_once_with({7})
    assert ex.canceled_req_ids == ([7] if staged else [])
    if not staged:
        return

    # The next pass sees the built request waiting for its KV and cancels it.
    request = Mock(
        is_child=False,
        py_request_id=7,
        is_finished=False,
        is_context_only_request=False,
        is_generation_only_request=True,
        state=LlmRequestState.DISAGG_GENERATION_INIT,
        py_total_draft_tokens=0,
        py_disaggregated_params=None,
    )
    ex._postlaunch_items = []
    ex.active_requests = [request]
    ex._disagg_coordinator = Mock()  # read through the disagg property
    ex._disagg_coordinator.cancel_gen_logically.return_value = True
    ex._disagg_coordinator.requires_gen_cancellation_consensus = False  # no replicated GEN vote
    ex._held_transfer_failures = lambda: set()
    ex._pending_transfer_responses = []
    ex._pending_response_terminations = []

    assert ex._cancel_waiting_disagg_requests([request]) == []
    request.finish_by_reason.assert_called_once_with(FinishReason.CANCELLED)
    assert ex.active_requests == []
    assert ex._pending_response_terminations == [request]
    assert ex.canceled_req_ids == []


def test_postlaunch_admission_is_off_in_default_runtime_hooks():
    assert RuntimeHooks().postlaunch_admission is False
