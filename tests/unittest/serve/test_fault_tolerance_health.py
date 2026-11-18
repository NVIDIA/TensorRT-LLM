# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Wide-EP fault-tolerance status endpoint, metrics and request errors."""

import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import uvicorn
from starlette.routing import Match
from test_common.serve.app_factory import AggConfig, isolated_serve_env

from tensorrt_llm._torch.distributed.membership import (
    AUTHORITY_SILENCE_TIMEOUT_S,
    HANG_TIMEOUT_S,
    RECORD_KEY,
    STALLED_SINCE_KEY,
    TRANSACTION_KEY,
    MembershipRecord,
    MembershipState,
    SignedStore,
    TransactionSummary,
)
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.rank_failure_requests import RANK_FAILURE_CODE, RankFailureError
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve import fault_tolerance_health as health_module
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve.anthropic_adapter import reframe_openai_stream
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.fault_tolerance_health import FaultToleranceHealth
from tensorrt_llm.serve.openai_protocol import RerankRequest
from tensorrt_llm.serve.openai_server import _stream_error_events
from tensorrt_llm.serve.responses_utils import (
    classify_stream_termination,
    describe_stream_termination,
)

# threadleak is off: the status refresh thread and the TestClient threads end in the teardown of
# isolated_serve_env, which runs after the leak check.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

WORLD = 4
FAILED = 2
_SECRET = "ab" * 32


def _record(state, active_mask=0b1011, generation=1, given_up_mask=0):
    return MembershipRecord(
        generation=generation,
        state=state,
        active_mask=active_mask,
        incarnations=(0,) * WORLD,
        given_up_mask=given_up_mask,
    )


class FakeRendezvousStore:
    def __init__(self, record, transaction=None):
        self.values = {
            key: value.model_dump_json().encode()
            for key, value in ((RECORD_KEY, record), (TRANSACTION_KEY, transaction))
            if value is not None
        }

    def set(self, key, value):
        self.values[key] = value

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]


@pytest.fixture
def new_reader(monkeypatch):
    """Return a factory of started readers that open ``store`` and refresh only once an hour."""
    monkeypatch.setattr(health_module, "_STATUS_REFRESH_PERIOD_S", 3600.0)
    readers = []

    def new_reader(store=None):
        if store is not None:
            monkeypatch.setattr(health_module, "open_store", Mock(return_value=store))
        readers.append(FaultToleranceHealth("host:1", _SECRET))
        readers[-1].start()
        return readers[-1]

    yield new_reader
    for reader in readers:
        reader.close()


@pytest.mark.parametrize(
    "record, state",
    [
        (_record(MembershipState.COMMITTED, active_mask=0b1111), "healthy"),
        (_record(MembershipState.COMMITTED), "degraded"),
        (_record(MembershipState.INSTALL), "recovering"),
        (_record(MembershipState.ABORTED), "failed"),
        (_record(MembershipState.SHUTDOWN), "stopped"),
        (None, "unprotected"),
    ],
)
def test_health_state(new_reader, record, state):
    """Check that the status answered before any refresh is the membership in the store."""
    assert new_reader(FakeRendezvousStore(record)).last_status().state == state


def test_reader_reads_nothing_until_started(monkeypatch):
    """Check that a reader opens no store and starts no thread until it is started."""
    monkeypatch.setattr(health_module, "open_store", open_store := Mock())
    threads = set(threading.enumerate())
    health = FaultToleranceHealth("host:1", _SECRET)
    assert set(threading.enumerate()) == threads
    open_store.assert_not_called()
    assert (health.last_status().state, health.last_status().authority_alive) == (
        "unprotected",
        False,
    )
    health.close()


def test_failed_refresh_reports_store_unreadable(monkeypatch, new_reader):
    """Check that a refresh that raises leaves the status with the store unreadable, not frozen."""
    monkeypatch.setattr(health_module, "_STATUS_REFRESH_PERIOD_S", 0.01)
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED, active_mask=0b1111))
    health = new_reader(store)
    store.check = Mock(side_effect=KeyError("unexpected"))
    # The refresh thread ends after the read that raised, with one last read.
    health._refresh_thread.join(10.0)
    assert not health._refresh_thread.is_alive()
    status = health.last_status()
    assert (status.state, status.authority_alive) == ("unprotected", False)


@pytest.mark.parametrize("active_mask", [0b1111, 0b1011], ids=["healthy", "degraded"])
def test_stalled_after_half_the_hang_timeout(monkeypatch, new_reader, active_mask):
    """Check that a committed membership reads stalled once a stall lasts half the hang timeout."""
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED, active_mask=active_mask))
    now_s = [4.0]
    monkeypatch.setattr(
        health_module, "time", SimpleNamespace(time=lambda: now_s[0], monotonic=time.monotonic)
    )
    health = new_reader(store)
    serving = "healthy" if active_mask == 0b1111 else "degraded"
    store.values[STALLED_SINCE_KEY] = b"4000"
    # A long step, such as a long prefill, keeps the other ranks waiting for a shorter time.
    now_s[0] += HANG_TIMEOUT_S / 2 - 1
    assert health.read_status().state == serving
    now_s[0] += 1
    assert health.read_status().state == "stalled"
    # The authority clears the stall with an empty value.
    store.values[STALLED_SINCE_KEY] = b""
    assert health.read_status().state == serving
    # A stall time that does not parse counts as no stall.
    store.values[STALLED_SINCE_KEY] = b"not a time"
    status = health.read_status()
    assert (status.state, status.stalled_seconds) == (serving, 0.0)


def test_unsigned_value_keeps_last_status(new_reader):
    """Check that a record not signed with the deployment's secret keeps the last status read."""
    values = FakeRendezvousStore(None)
    store = SignedStore(values, _SECRET)
    store.set(RECORD_KEY, _record(MembershipState.COMMITTED).model_dump_json().encode())
    health = new_reader(store)
    values.values[RECORD_KEY] = _record(MembershipState.ABORTED).model_dump_json().encode()
    status = health.read_status()
    assert (status.state, status.authority_alive) == ("degraded", True)


def test_older_generation_keeps_last_status(new_reader):
    """Check that a record and transaction of an earlier generation, written back, are ignored."""
    store = SignedStore(FakeRendezvousStore(None), _SECRET)

    def publish(record, transaction):
        store.set(RECORD_KEY, record.model_dump_json().encode())
        store.set(TRANSACTION_KEY, transaction.model_dump_json().encode())

    later_failover = FAILOVER.model_copy(update={"generation": 2})
    publish(_record(MembershipState.COMMITTED, generation=2), later_failover)
    health = new_reader(store)
    publish(_record(MembershipState.COMMITTED, active_mask=0b1111), FAILOVER)
    status = health.read_status()
    assert (status.state, status.generation) == ("degraded", 2)
    assert status.last_transaction.generation == 2


def test_invalid_secret_fails_at_start(monkeypatch):
    """Check that a secret the ranks would refuse fails the reader at once, not each read."""
    monkeypatch.setattr(health_module, "open_store", open_store := Mock())
    with pytest.raises(ValueError, match="at least 32 bytes"):
        FaultToleranceHealth("host:1", "ab" * 31)
    open_store.assert_not_called()


def test_stopped_after_authority_ended(monkeypatch, new_reader):
    """Check that a deployment that shut down stays stopped once its authority ended."""
    monkeypatch.setattr(
        health_module.socket, "create_connection", Mock(side_effect=ConnectionRefusedError)
    )
    store = FakeRendezvousStore(_record(MembershipState.SHUTDOWN))
    health = new_reader(store)
    store.check = Mock(side_effect=RuntimeError("the store does not answer"))
    status = health.read_status()
    assert (status.state, status.authority_alive) == ("stopped", False)


@pytest.mark.parametrize(
    "probe_error, state",
    [(TimeoutError(), "recovering"), (ConnectionRefusedError(), "failed")],
    ids=["transient", "authority-ended"],
)
def test_health_after_failed_read(monkeypatch, new_reader, probe_error, state):
    """Check that a failed read keeps the last state, or reads failed once the store refuses."""
    now = [0.0]
    monkeypatch.setattr(health_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(health_module.socket, "create_connection", Mock(side_effect=probe_error))
    store = FakeRendezvousStore(_record(MembershipState.INSTALL))
    health = new_reader(store)
    assert health.read_status().state == "recovering"
    answers = store.check
    store.check = Mock(side_effect=RuntimeError("the store does not answer"))
    for _ in range(3):
        status = health.read_status()
        assert status.state == state
        assert status.authority_alive == (state != "failed")
        assert status.active_ranks == [0, 1, 3]
    assert health_module.open_store.call_count == 1
    store.check = answers
    now[0] += health_module._STATUS_READ_TIMEOUT_S
    assert health.read_status().state == state


def test_health_unprotected_while_store_silent(monkeypatch, new_reader):
    """Check that the status is unprotected until the store answers and once it stays silent."""
    now = [0.0]
    monkeypatch.setattr(health_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(health_module.socket, "create_connection", Mock(side_effect=TimeoutError))
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED, active_mask=0b1111))
    open_store = Mock(side_effect=OSError("no store"))
    monkeypatch.setattr(health_module, "open_store", open_store)
    health = new_reader()
    status = health.read_status()
    assert status.state == "unprotected"
    assert status.active_ranks is None
    open_store.side_effect = None
    open_store.return_value = store
    now[0] += health_module._STATUS_READ_TIMEOUT_S
    assert health.read_status().state == "healthy"
    answers = store.check
    store.check = Mock(side_effect=RuntimeError("the store does not answer"))
    now[0] += AUTHORITY_SILENCE_TIMEOUT_S
    assert health.read_status().state == "healthy"
    now[0] += health_module._STATUS_READ_TIMEOUT_S
    status = health.read_status()
    assert status.state == "unprotected"
    assert not status.authority_alive
    store.check = answers
    now[0] += health_module._STATUS_READ_TIMEOUT_S
    assert health.read_status().state == "healthy"


def test_closed_reader_stops_reading(new_reader):
    """Check that a closed reader never reads its store again."""
    health = new_reader(FakeRendezvousStore(_record(MembershipState.COMMITTED)))
    assert health.read_status().authority_alive
    health.close()
    assert not health.read_status().authority_alive
    assert health_module.open_store.call_count == 1


def test_close_waits_for_read_in_progress(monkeypatch, new_reader):
    """Check that closing a reader during a read returns once its refresh thread has ended."""
    monkeypatch.setattr(health_module, "_STATUS_REFRESH_PERIOD_S", 0.01)
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED))
    threads = set(threading.enumerate())
    health = new_reader(store)
    reading = threading.Event()
    answers = store.check

    def slow_check(keys):
        reading.set()
        time.sleep(0.1)  # The store takes a while to answer.
        return answers(keys)

    store.check = slow_check
    assert reading.wait(10.0)
    health.close()
    assert set(threading.enumerate()) <= threads


@pytest.mark.parametrize("err_type", ["BadRequestError", "internal_error"])
@pytest.mark.parametrize(
    "error, status, code, expected_type",
    [
        (RankFailureError(), 503, RANK_FAILURE_CODE, "ServiceUnavailableError"),
        # The type decides, not the text.
        (RequestError(str(RankFailureError())), 400, 400, None),
    ],
)
def test_rank_failure_answers_503_with_code(
    serve_env, error, status, code, expected_type, err_type
):
    """Check that only a RankFailureError, by type, answers 503 rank_failure without Retry-After.

    Any other error keeps the caller's error type.
    """
    server = serve_env.build_openai_server(AggConfig())
    response = server.create_exception_error_response(error, err_type)
    body = json.loads(response.body)
    assert response.status_code == status
    assert body["code"] == code
    assert body["type"] == (expected_type or err_type)
    assert body["message"] == str(error)
    assert "retry-after" not in response.headers


def test_rank_failure_ends_streams_with_code():
    """Check that a stream that a rank failure ends sends the rank failure's code or message."""
    error = RankFailureError()
    cause = classify_stream_termination(error)
    assert cause == RANK_FAILURE_CODE
    assert describe_stream_termination(error, cause) == f"RankFailureError: {error}"
    events = _stream_error_events(error)
    assert json.loads(events[0][len("data: ") :])["error"]["code"] == RANK_FAILURE_CODE

    async def chat_stream():
        for event in events:
            yield event

    async def collect():
        return [frame async for frame in reframe_openai_stream(chat_stream(), "model")]

    name, data = asyncio.run(collect())[-1].strip().split("\n")
    assert name == "event: error"
    assert json.loads(data[len("data: ") :])["error"] == {
        "type": "overloaded_error",
        "message": str(error),
    }


def test_rank_failure_is_an_anthropic_overloaded_error(serve_env):
    """Check that a Messages request and a batched one that a rank failure ends get overloaded_error."""
    error = RankFailureError()

    async def openai_chat(chat_request, raw_request):
        return server.create_exception_error_response(error)

    server = serve_env.build_openai_server(AggConfig(model_name="model"))
    server.openai_chat = openai_chat
    request = AnthropicMessagesRequest(
        model="model", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
    )
    raw_request = SimpleNamespace(state=SimpleNamespace(), headers={})
    response = asyncio.run(server.anthropic_messages(request, raw_request))
    assert response.status_code == 503
    expected = {"type": "overloaded_error", "message": str(error)}
    assert json.loads(response.body)["error"] == expected
    assert asyncio.run(server._run_batched_message(request)) == ("errored", expected)


@pytest.mark.parametrize(
    "error, status, code",
    [
        (RankFailureError(), 503, RANK_FAILURE_CODE),
        (RuntimeError("encoder failed"), 500, 500),
    ],
)
def test_rerank_rank_failure_answers_503_with_code(serve_env, error, status, code):
    """Check that a rerank request that a rank failure ends answers 503 rank_failure.

    Any other error keeps the route's 500.
    """
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.RERANK))
    server.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: [ord(char) for char in text]
    )

    async def submit(token_ids):
        raise error

    server.encode_batcher = SimpleNamespace(validate_input=lambda token_ids: None, submit=submit)
    request = RerankRequest(model="reranker", query="query", documents=["a"])
    response = asyncio.run(server._rerank(request))
    assert response.status_code == status
    assert json.loads(response.body)["code"] == code


def _server(env, monkeypatch, store, fault_tolerance=True, **config):
    """Return a server that reads ``store``, which stalled 6 s ago, or no store if it is None."""
    monkeypatch.setattr(
        health_module,
        "open_store",
        Mock(side_effect=OSError("no store")) if store is None else Mock(return_value=store),
    )
    monkeypatch.setattr(
        health_module, "time", SimpleNamespace(time=lambda: 10.0, monotonic=time.monotonic)
    )
    if store is not None:
        store.values[STALLED_SINCE_KEY] = b"4000"
    return env.build_openai_server(AggConfig(fault_tolerance=fault_tolerance, **config))


FAILOVER = TransactionSummary(
    generation=1,
    outcome="COMMITTED",
    removed_ranks=(FAILED,),
    causes={FAILED: "hung"},
    added_ranks=(),
    detected_unix_ms=988,
    began_unix_ms=1000,
    ready_unix_ms=1400,
    ended_unix_ms=2500,
    changes={"failover": 1},
)


def test_status_endpoint_reports_membership(monkeypatch, tmp_path):
    # Rank 1 serves again, so its earlier mark does not count.
    record = _record(MembershipState.COMMITTED, active_mask=0b0011, given_up_mask=0b1010)
    store = FakeRendezvousStore(record, FAILOVER)
    with isolated_serve_env(tmp_path) as env:
        client = env.client(_server(env, monkeypatch, store).app)
        response = client.get("/fault_tolerance_status")
    assert response.status_code == 200
    assert response.json() == {
        "state": "degraded",
        "authority_alive": True,
        "stalled_seconds": 6.0,
        "generation": 1,
        "active_ranks": [0, 1],
        "failed_ranks": [FAILED, 3],
        "replacements": {str(FAILED): "pending", "3": "given_up"},
        "incarnations": [0] * WORLD,
        "last_transaction": {
            "generation": 1,
            "outcome": "COMMITTED",
            "removed_ranks": [FAILED],
            "causes": {str(FAILED): "hung"},
            "added_ranks": [],
            "detected_unix_ms": 988,
            "began_unix_ms": 1000,
            "ready_unix_ms": 1400,
            "ended_unix_ms": 2500,
            "changes": {"failover": 1},
            "reason": "",
        },
    }


@pytest.mark.parametrize(
    "config",
    [
        {"return_perf_metrics": True},
        {},
        {"return_perf_metrics": True, "fault_tolerance": False},
    ],
    ids=["with_perf_metrics", "alone", "off"],
)
def test_prometheus_metrics_follow_flag(monkeypatch, tmp_path, config):
    """Check that the Prometheus endpoint reports the membership whenever fault tolerance is on."""
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED), FAILOVER)
    with isolated_serve_env(tmp_path) as env:
        if not config.get("return_perf_metrics"):
            # Only perf metrics set up the multiprocess directory.
            del os.environ["PROMETHEUS_MULTIPROC_DIR"]
        client = env.client(_server(env, monkeypatch, store, **config).app)
        lines = client.get("/prometheus/metrics").text.splitlines()
    labels = 'engine_type="pytorch",model_name="fake-model"'
    expected = [
        f'trtllm_fault_tolerance_state{{{labels},state="stalled"}} 0.0',
        f'trtllm_fault_tolerance_state{{{labels},state="degraded"}} 1.0',
        f'trtllm_fault_tolerance_state{{{labels},state="stopped"}} 0.0',
        f"trtllm_fault_tolerance_stalled_seconds{{{labels}}} 6.0",
        f"trtllm_fault_tolerance_active_ranks{{{labels}}} 3.0",
        f"trtllm_fault_tolerance_failed_ranks{{{labels}}} 1.0",
        f'trtllm_fault_tolerance_replacements{{{labels},state="pending"}} 1.0',
        f'trtllm_fault_tolerance_replacements{{{labels},state="given_up"}} 0.0',
        f"trtllm_fault_tolerance_rank_restarts_total{{{labels}}} 0.0",
        'trtllm_fault_tolerance_membership_changes_total{engine_type="pytorch",kind="failover",'
        'model_name="fake-model"} 1.0',
        'trtllm_fault_tolerance_membership_changes_total{engine_type="pytorch",kind="abort",'
        'model_name="fake-model"} 0.0',
        # From the detection of the failure to the end of the change.
        f"trtllm_fault_tolerance_last_change_pause_seconds{{{labels}}} 1.512",
    ]
    if config.get("fault_tolerance", True):
        assert set(expected) <= set(lines)
    else:
        assert not [line for line in lines if "fault_tolerance" in line]


@pytest.mark.parametrize("fault_tolerance", [True, False], ids=["on", "off"])
def test_attached_frontend_forwards_status(monkeypatch, tmp_path, fault_tolerance):
    """Check that only the launcher reads the status, and attached frontends forward its routes."""
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED))
    with isolated_serve_env(tmp_path) as env:
        server = _server(
            env,
            monkeypatch,
            store,
            fault_tolerance=fault_tolerance,
            return_perf_metrics=True,
            multi_frontend_attached=True,
        )
        assert server.fault_tolerance_health is None
        assert not health_module.open_store.called
        for path in ("/fault_tolerance_status", "/prometheus/metrics"):
            scope = {"type": "http", "method": "GET", "path": path, "path_params": {}}
            route = next((r for r in server.app.routes if r.matches(scope)[0] == Match.FULL), None)
            forwarded = getattr(route, "endpoint", None) == server._launcher_forwarder.forward
            assert forwarded == fault_tolerance, path


@pytest.mark.parametrize(
    "store_readable, rank_0_ended",
    [(True, False), (False, False), (True, True)],
    ids=["serving", "unreadable", "rank_0_ended"],
)
def test_frontends_agree_on_health(monkeypatch, tmp_path, store_readable, rank_0_ended):
    """Check that every frontend answers /health from its executor, whatever the membership."""
    # A refresh that finds the deployment failed shuts the server down.
    monkeypatch.setattr(openai_server_module.signal, "raise_signal", Mock())
    store = FakeRendezvousStore(_record(MembershipState.ABORTED)) if store_readable else None
    with isolated_serve_env(tmp_path) as env:
        launcher, attached = (
            _server(env, monkeypatch, store, multi_frontend_attached=is_attached)
            for is_attached in (False, True)
        )
        try:
            if rank_0_ended:
                # The executor proxy reaps an ended rank 0 as ``_check_mpi_workers`` does.
                for server in (launcher, attached):
                    executor = server.generator._executor
                    GenerationExecutor._set_fatal_error(
                        executor, RuntimeError("MPI worker rank 0 exited unexpectedly")
                    )
                    executor.doing_shutdown = True
            codes = [asyncio.run(server.health()).status_code for server in (launcher, attached)]
        finally:
            launcher.fault_tolerance_health.close()
    assert codes == ([503, 503] if rank_0_ended else [200, 200])


@pytest.mark.parametrize("shutting_down", [False, True], ids=["serving", "shutting_down"])
def test_failed_deployment_shuts_the_server_down(monkeypatch, tmp_path, shutting_down):
    """Check that a failed deployment raises SIGINT once, and none while the server shuts down.

    Uvicorn takes a second SIGINT as a forced exit, which cuts the open streams.
    """
    monkeypatch.setattr(health_module, "_STATUS_REFRESH_PERIOD_S", 0.01)
    raise_signal = Mock()
    monkeypatch.setattr(openai_server_module.signal, "raise_signal", raise_signal)
    store = FakeRendezvousStore(_record(MembershipState.COMMITTED))
    record_checks = 0
    checked = threading.Condition()
    answers = store.check

    def counted_check(keys):
        nonlocal record_checks
        with checked:
            record_checks += RECORD_KEY in keys
            checked.notify_all()
        return answers(keys)

    def wait_for_reads(count):
        # Each read checks the record once, so once the read after them checks it, the next
        # ``count`` reads have ended.
        with checked:
            target = record_checks + count + 1
            assert checked.wait_for(lambda: record_checks >= target, timeout=10.0)

    with isolated_serve_env(tmp_path) as env:
        server = _server(env, monkeypatch, store)
        server._uvicorn_server = uvicorn.Server(uvicorn.Config(server.app))
        server._uvicorn_server.should_exit = shutting_down
        store.check = counted_check
        try:
            wait_for_reads(3)
            raise_signal.assert_not_called()
            store.values[RECORD_KEY] = _record(MembershipState.ABORTED).model_dump_json().encode()
            # Later reads that find the deployment failed raise no second signal.
            wait_for_reads(3)
            assert server.fault_tolerance_health.last_status().state == "failed"
        finally:
            server.fault_tolerance_health.close()
    if shutting_down:
        raise_signal.assert_not_called()
    else:
        raise_signal.assert_called_once_with(openai_server_module.signal.SIGINT)


def test_status_never_waits_for_store(monkeypatch, tmp_path):
    """Check that the status endpoint answers from the last status during a read."""
    # The released read finds the deployment failed, which shuts the server down.
    monkeypatch.setattr(openai_server_module.signal, "raise_signal", Mock())
    store = FakeRendezvousStore(_record(MembershipState.ABORTED))
    reading, unblocked = threading.Event(), threading.Event()
    ended_reads = []

    def blocked_read(keys):
        reading.set()
        unblocked.wait(30.0)
        ended_reads.append(keys)

    with isolated_serve_env(tmp_path) as env:
        server = _server(env, monkeypatch, store)
        client = env.client(server.app)
        store.check = blocked_read
        try:
            assert reading.wait(10.0)
            response = client.get("/fault_tolerance_status")
            assert not ended_reads
            assert response.status_code == 200
        finally:
            unblocked.set()
            server.fault_tolerance_health.close()
    assert response.json()["state"] == "failed"
