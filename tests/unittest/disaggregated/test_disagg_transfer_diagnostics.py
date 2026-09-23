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

import json
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.disaggregation import diagnostics

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _isolate_diagnostic_sink(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    diagnostics._reset_diagnostic_sink_for_tests()
    monkeypatch.delenv("TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID", raising=False)
    yield
    diagnostics._reset_diagnostic_sink_for_tests()


def test_suppress_diagnostic_errors_does_not_interrupt_request_progress() -> None:
    progress = []

    with diagnostics.suppress_diagnostic_errors():
        progress.append("diagnostic_started")
        raise RuntimeError("diagnostic preparation failed")

    progress.append("request_progressed")
    assert progress == ["diagnostic_started", "request_progressed"]


def test_disabled_emit_event_does_no_diagnostic_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)

    unexpected_call = MagicMock(side_effect=AssertionError("disabled diagnostics did work"))
    monkeypatch.setattr(diagnostics, "_host_identity", unexpected_call)
    monkeypatch.setattr(diagnostics, "_get_sink", unexpected_call)
    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(getpid=unexpected_call, write=unexpected_call, getenv=unexpected_call),
    )
    monkeypatch.setattr(
        diagnostics,
        "time",
        SimpleNamespace(monotonic_ns=unexpected_call, time_ns=unexpected_call),
    )
    monkeypatch.setattr(diagnostics, "json", SimpleNamespace(dumps=unexpected_call))
    monkeypatch.setattr(
        diagnostics, "uuid", SimpleNamespace(uuid4=unexpected_call, UUID=unexpected_call)
    )

    diagnostics.emit_event("ctx_send_ready", side="ctx", request_id=17)

    unexpected_call.assert_not_called()


def test_disabled_request_helpers_do_not_inspect_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)
    unexpected_call = MagicMock(side_effect=AssertionError("disabled diagnostics did work"))
    monkeypatch.setattr(diagnostics, "get_request_id", unexpected_call)
    monkeypatch.setattr(diagnostics, "get_request_id_scope", unexpected_call)
    monkeypatch.setattr(diagnostics, "emit_event", unexpected_call)
    monkeypatch.setattr(diagnostics, "capture_timestamp", unexpected_call)

    class _OpaqueInput:
        def __getattribute__(self, name: str) -> object:
            return unexpected_call(name)

    # Record inspection even if the helper's error isolation swallows the
    # spy's exception before any other patched diagnostic function is reached.
    request = _OpaqueInput()
    dist = _OpaqueInput()

    diagnostics.emit_request_event("gen_ingress", request, side="gen", dist=dist)
    diagnostics.emit_transfer_timeout(
        "transfer_timeout_started", request, side="gen", dist=dist, timeout_ms=1000
    )

    unexpected_call.assert_not_called()


def test_request_helper_preserves_identity_rank_and_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emit = MagicMock()
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    request = SimpleNamespace(
        request_id=17,
        py_request_id=42,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=99),
    )
    dist = SimpleNamespace(rank=3, tp_rank=1, pp_rank=0, cp_rank=2, dp_rank=7)

    diagnostics.emit_request_event(
        "gen_ingress",
        request,
        side="gen",
        dist=dist,
        rank=5,
        timestamp=(100, 200),
        prompt_tokens=256,
    )

    emit.assert_called_once_with(
        "gen_ingress",
        side="gen",
        request_id=99,
        request_id_scope="run",
        local_request_id=42,
        rank=5,
        rank_info=None,
        timestamp=(100, 200),
        prompt_tokens=256,
        tp_rank=1,
        pp_rank=0,
        cp_rank=2,
    )


def test_request_helper_preserves_rank_info_without_inventing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emit = MagicMock()
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    request = SimpleNamespace(request_id=17, py_request_id=42, py_disaggregated_params=None)
    rank_info = SimpleNamespace(instance_name="gen", instance_rank=3)

    diagnostics.emit_request_event(
        "gen_receive_requested", request, side="gen", rank_info=rank_info, slice_id=0
    )

    emit.assert_called_once_with(
        "gen_receive_requested",
        side="gen",
        request_id=42,
        request_id_scope="process",
        local_request_id=42,
        rank=None,
        rank_info=rank_info,
        timestamp=None,
        slice_id=0,
    )


@pytest.mark.parametrize(
    ("disagg_id", "ctx_id", "expected_id", "expected_scope"),
    [
        (99, 7, 99, "run"),
        (0, 7, 0, "run"),
        (None, 7, 7, "process"),
        (None, 0, 0, "process"),
        (None, None, 41, "process"),
    ],
)
def test_request_identity_matches_transport_without_promoting_local_ids(
    monkeypatch: pytest.MonkeyPatch,
    disagg_id: int | None,
    ctx_id: int | None,
    expected_id: int,
    expected_scope: str,
) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emit = MagicMock()
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    request = SimpleNamespace(
        request_id=41,
        py_request_id=41,
        py_disaggregated_params=SimpleNamespace(disagg_request_id=disagg_id, ctx_request_id=ctx_id),
    )

    diagnostics.emit_request_event("gen_receive_start", request, side="gen")

    assert emit.call_args.kwargs["request_id"] == expected_id
    assert emit.call_args.kwargs["request_id_scope"] == expected_scope
    assert emit.call_args.kwargs["local_request_id"] == 41
    assert request.request_id == request.py_request_id == 41


@pytest.mark.parametrize("failure", ["identity", "request", "rank", "emit"])
def test_request_helper_isolates_metadata_and_emission_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emit = MagicMock(
        side_effect=RuntimeError("broken diagnostic sink") if failure == "emit" else None
    )
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    request = SimpleNamespace(request_id=17, py_disaggregated_params=None)
    if failure != "request":
        request.py_request_id = 42
    dist = SimpleNamespace(rank=3, tp_rank=1, pp_rank=0, cp_rank=2)
    if failure == "rank":
        del dist.tp_rank
    if failure == "identity":
        monkeypatch.setattr(
            diagnostics, "get_request_id", MagicMock(side_effect=RuntimeError("missing identity"))
        )

    diagnostics.emit_request_event("gen_ingress", request, side="gen", dist=dist)

    assert emit.call_count == (1 if failure == "emit" else 0)


@pytest.mark.parametrize("event", ["transfer_timeout_started", "transfer_timeout_observed"])
def test_timeout_helper_observes_without_changing_deadline(
    monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emit = MagicMock()
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    request = SimpleNamespace(
        request_id=17,
        py_request_id=42,
        py_disaggregated_params=None,
        py_kv_transfer_start_time=1.25,
        state=SimpleNamespace(name="DISAGG_GENERATION_TRANS_IN_PROGRESS"),
    )
    dist = SimpleNamespace(rank=3, tp_rank=1, pp_rank=0, cp_rank=2)

    diagnostics.emit_transfer_timeout(
        event,
        request,
        side="gen",
        dist=dist,
        timeout_ms=1000,
        elapsed_ms=1500,
        cancellation_requested=True,
    )

    emit.assert_called_once_with(
        event,
        side="gen",
        request_id=42,
        request_id_scope="process",
        local_request_id=42,
        rank=3,
        rank_info=None,
        timestamp=None,
        tp_rank=1,
        pp_rank=0,
        cp_rank=2,
        timeout_ms=1000,
        timeout_owner="pyexecutor",
        timer_start_monotonic_ns=1_250_000_000,
        state="DISAGG_GENERATION_TRANS_IN_PROGRESS",
        elapsed_ms=1500,
        cancellation_requested=True,
    )
    assert request.py_kv_transfer_start_time == 1.25
    assert request.state.name == "DISAGG_GENERATION_TRANS_IN_PROGRESS"


def test_timeout_helper_isolates_invalid_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    emit = MagicMock()
    monkeypatch.setattr(diagnostics, "emit_event", emit)
    request = SimpleNamespace(py_kv_transfer_start_time=None)

    diagnostics.emit_transfer_timeout(
        "transfer_timeout_started", request, side="gen", dist=SimpleNamespace(), timeout_ms=1000
    )

    emit.assert_not_called()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork support")
def test_forked_child_replaces_inherited_diagnostic_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    run_uuid = str(uuid.uuid4())
    monkeypatch.setenv("TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID", run_uuid)
    diagnostics._reset_diagnostic_sink_for_tests()
    parent_pid = os.getpid()
    parent_sink = diagnostics._get_sink(parent_pid)
    read_fd, write_fd = os.pipe()

    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            reset_in_child = diagnostics._sink is None
            child_sink = diagnostics._get_sink(os.getpid())
            result = (
                reset_in_child
                and child_sink is not parent_sink
                and child_sink.pid == os.getpid()
                and child_sink._thread.is_alive()
                and child_sink._identity["process_uuid"] != parent_sink._identity["process_uuid"]
                and child_sink._identity["run_uuid"]
                == parent_sink._identity["run_uuid"]
                == run_uuid
            )
            diagnostics._reset_diagnostic_sink_for_tests()
            os.write(write_fd, b"ok" if result else b"failed")
        except BaseException as error:
            os.write(write_fd, f"error: {error!r}".encode())
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    try:
        child_result = os.read(read_fd, 4096)
        _, status = os.waitpid(child_pid, 0)
    finally:
        os.close(read_fd)
        diagnostics._reset_diagnostic_sink_for_tests()

    assert os.waitstatus_to_exitcode(status) == 0
    assert child_result == b"ok"


@pytest.mark.parametrize(
    ("configured_run_id", "expected_run_id", "expected_status"),
    [
        (None, None, "unset"),
        ("", None, "invalid"),
        ("not-a-run-uuid", None, "invalid"),
        (
            "9C74CDA0AA094C668F24B8B38F40A958",
            "9c74cda0-aa09-4c66-8f24-b8b38f40a958",
            "shared",
        ),
    ],
)
def test_sink_records_validated_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    configured_run_id: str | None,
    expected_run_id: str | None,
    expected_status: str,
) -> None:
    if configured_run_id is not None:
        monkeypatch.setenv("TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID", configured_run_id)
    records = []
    monkeypatch.setattr(
        diagnostics._AsyncDiagnosticSink,
        "_write",
        staticmethod(lambda record: records.append(record.copy())),
    )
    # Leak detection runs before fixture teardown, so stop the writer in the test body.
    with closing(diagnostics._get_sink(os.getpid())) as sink:
        sink.submit({"event": "diagnostic_capabilities", "request_id": None})
        sink.flush()

    assert not sink._thread.is_alive()
    assert len(records) == 1
    record = records[0]
    assert record["run_uuid"] == expected_run_id
    assert record["run_uuid_status"] == expected_status
    assert str(uuid.UUID(record["process_uuid"])) == record["process_uuid"]


def test_event_identity_is_cached_and_cannot_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    run_uuid = str(uuid.uuid4())
    monkeypatch.setenv("TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID", run_uuid)
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "_host_identity", lambda: "node-a")
    records = []
    monkeypatch.setattr(
        diagnostics._AsyncDiagnosticSink,
        "_write",
        staticmethod(lambda record: records.append(record.copy())),
    )
    unexpected_call = MagicMock(side_effect=AssertionError("event regenerated identity"))
    with closing(diagnostics._get_sink(os.getpid())) as sink:
        monkeypatch.setattr(
            diagnostics, "uuid", SimpleNamespace(uuid4=unexpected_call, UUID=unexpected_call)
        )
        monkeypatch.setattr(
            diagnostics, "os", SimpleNamespace(getpid=os.getpid, getenv=unexpected_call)
        )

        for event in ("diagnostic_capabilities", "ctx_send_ready"):
            diagnostics.emit_event(
                event,
                side="ctx",
                request_id=17,
                run_uuid="forged-run",
                process_uuid="forged-process",
                run_uuid_status="invalid",
                host="forged-host",
                pid=-1,
            )
        sink.flush()

    assert not sink._thread.is_alive()
    unexpected_call.assert_not_called()
    assert len(records) == 2
    for record in records:
        assert record["run_uuid"] == run_uuid
        assert record["run_uuid_status"] == "shared"
        assert record["process_uuid"] == sink._identity["process_uuid"]
        assert record["host"] == "node-a"
        assert record["pid"] == os.getpid()


@pytest.mark.parametrize("restart", ["same_pid", "changed_pid"])
def test_recreated_sink_has_fresh_process_identity(
    monkeypatch: pytest.MonkeyPatch, restart: str
) -> None:
    run_uuid = str(uuid.uuid4())
    monkeypatch.setenv("TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID", run_uuid)
    pid = os.getpid()
    with closing(diagnostics._get_sink(pid)) as original_sink:
        assert diagnostics._get_sink(pid) is original_sink
        if restart == "same_pid":
            diagnostics._reset_diagnostic_sink_for_tests()
            replacement_pid = pid
        else:
            # Model PID replacement without inheriting a live parent thread.
            original_sink.close()
            replacement_pid = pid + 1
        with closing(diagnostics._get_sink(replacement_pid)) as replacement_sink:
            assert replacement_sink is not original_sink
            assert (
                replacement_sink._identity["process_uuid"]
                != original_sink._identity["process_uuid"]
            )
            assert (
                replacement_sink._identity["run_uuid"]
                == original_sink._identity["run_uuid"]
                == run_uuid
            )

    assert not original_sink._thread.is_alive()
    assert not replacement_sink._thread.is_alive()


@pytest.mark.parametrize("failing_dependency", ["uuid", "environment"])
def test_identity_initialization_failure_does_not_affect_request_progress(
    monkeypatch: pytest.MonkeyPatch, failing_dependency: str
) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    failure = MagicMock(side_effect=OSError("diagnostic identity unavailable"))
    if failing_dependency == "uuid":
        monkeypatch.setattr(diagnostics, "uuid", SimpleNamespace(uuid4=failure))
    else:
        monkeypatch.setattr(diagnostics, "os", SimpleNamespace(getpid=os.getpid, getenv=failure))

    diagnostics.emit_event("gen_decode_ready", side="gen", request_id=42)

    failure.assert_called_once()
    assert diagnostics._sink is None


def test_enabled_emit_event_records_request_and_rank_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics._reset_diagnostic_sink_for_tests()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "_host_identity", lambda: "node-a")
    request_thread_id = threading.get_ident()
    writes = []

    def write(fd: int, data: bytes) -> int:
        writes.append((threading.get_ident(), fd, data))
        return len(data)

    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(
            getpid=lambda: 321,
            write=write,
            getenv=lambda _key: None,
        ),
    )
    monkeypatch.setattr(
        diagnostics,
        "time",
        SimpleNamespace(monotonic_ns=lambda: 111, time_ns=lambda: 222),
    )
    rank_info = SimpleNamespace(
        instance_name="ctx_0",
        instance_rank=4,
        tp_rank=1,
        pp_rank=2,
        cp_rank=3,
        dp_rank=0,
    )

    diagnostics.emit_event(
        "ctx_backend_submitted",
        side="ctx",
        request_id=99,
        local_request_id=7,
        rank_info=rank_info,
        slice_id=5,
        peer_rank=8,
        transfer_bytes=4096,
        source_kv_request_owned=True,
        source_kv_reuse_pinned=False,
        timestamp=(1_234, 5_678),
    )
    diagnostics._flush_diagnostic_sink_for_tests()

    assert len(writes) == 1
    writer_thread_id, fd, encoded_message = writes[0]
    assert writer_thread_id != request_thread_id
    assert fd == 1
    message = encoded_message.decode("utf-8").removesuffix("\n")
    assert message.startswith(diagnostics.DIAGNOSTICS_LOG_PREFIX)
    payload = message.removeprefix(diagnostics.DIAGNOSTICS_LOG_PREFIX)
    process_uuid = json.loads(payload)["process_uuid"]
    assert str(uuid.UUID(process_uuid)) == process_uuid
    assert json.loads(payload) == {
        "schema_version": diagnostics.DIAGNOSTICS_SCHEMA_VERSION,
        "event": "ctx_backend_submitted",
        "side": "ctx",
        "request_id": 99,
        "local_request_id": 7,
        "host": "node-a",
        "pid": 321,
        "process_uuid": process_uuid,
        "run_uuid": None,
        "run_uuid_status": "unset",
        "monotonic_ns": 1_234,
        "wall_ns": 5_678,
        "instance": "ctx_0",
        "rank": 4,
        "tp_rank": 1,
        "pp_rank": 2,
        "cp_rank": 3,
        "dp_rank": 0,
        "slice_id": 5,
        "peer_rank": 8,
        "transfer_bytes": 4096,
        "source_kv_request_owned": True,
        "source_kv_reuse_pinned": False,
    }
    assert payload == json.dumps(json.loads(payload), separators=(",", ":"), sort_keys=True)
    diagnostics._reset_diagnostic_sink_for_tests()


def test_enabled_emit_event_accepts_executor_rank_context(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics._reset_diagnostic_sink_for_tests()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "_host_identity", lambda: "node-b")
    writes = []

    def write(fd: int, data: bytes) -> int:
        writes.append((fd, data))
        return len(data)

    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(
            getpid=lambda: 654,
            write=write,
            getenv=lambda _key: None,
        ),
    )
    monkeypatch.setattr(
        diagnostics,
        "time",
        SimpleNamespace(monotonic_ns=lambda: 9_000, time_ns=lambda: 10_000),
    )
    diagnostics.emit_event(
        "gen_kv_admission_result",
        side="gen",
        request_id=101,
        rank=6,
        instance="gen_0",
        outcome="deferred",
    )
    diagnostics._flush_diagnostic_sink_for_tests()

    assert len(writes) == 1
    _, encoded_message = writes[0]
    message = encoded_message.decode("utf-8").removesuffix("\n")
    payload = json.loads(message.removeprefix(diagnostics.DIAGNOSTICS_LOG_PREFIX))
    assert payload["rank"] == 6
    assert payload["instance"] == "gen_0"
    assert payload["outcome"] == "deferred"
    assert "slice_id" not in payload
    assert "peer_rank" not in payload
    diagnostics._reset_diagnostic_sink_for_tests()


def test_sink_write_completes_partial_stdout_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"event": "partial_write", "request_id": 42}
    expected = (
        f"{diagnostics.DIAGNOSTICS_LOG_PREFIX}"
        f"{json.dumps(record, separators=(',', ':'), sort_keys=True)}\n"
    ).encode("utf-8")
    accepted_chunks = []
    calls = []

    def write(fd: int, data: bytes) -> int:
        calls.append((fd, data))
        written = min(7, len(data))
        accepted_chunks.append(data[:written])
        return written

    monkeypatch.setattr(diagnostics, "os", SimpleNamespace(write=write))

    diagnostics._AsyncDiagnosticSink._write(record)

    assert len(calls) > 2
    assert all(fd == 1 for fd, _ in calls)
    assert [data for _, data in calls] == [
        expected[offset:] for offset in range(0, len(expected), 7)
    ]
    assert b"".join(accepted_chunks) == expected


def test_sink_write_rejects_zero_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(write=lambda _fd, _data: 0),
    )

    with pytest.raises(OSError, match="made no progress"):
        diagnostics._AsyncDiagnosticSink._write({"event": "no_progress"})


def test_async_sink_failure_does_not_stop_later_diagnostic_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics._reset_diagnostic_sink_for_tests()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "_host_identity", lambda: "node-a")
    writes = []

    def write(fd: int, data: bytes) -> int:
        writes.append((fd, data))
        if len(writes) == 1:
            raise OSError("diagnostic sink unavailable")
        return len(data)

    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(
            getpid=lambda: 123,
            write=write,
            getenv=lambda _key: None,
        ),
    )

    diagnostics.emit_event("gen_decode_ready", side="gen", request_id=42)
    diagnostics._flush_diagnostic_sink_for_tests()
    sink = diagnostics._sink
    assert sink is not None
    assert sink._thread.is_alive()

    diagnostics.emit_event("gen_decode_ready", side="gen", request_id=43)
    diagnostics._flush_diagnostic_sink_for_tests()

    assert len(writes) == 3
    dropped = json.loads(
        writes[1][1].decode("utf-8").removeprefix(diagnostics.DIAGNOSTICS_LOG_PREFIX)
    )
    assert dropped["event"] == "diagnostics_events_dropped"
    assert dropped["dropped_events"] == 1
    assert b'"request_id":43' in writes[2][1]
    diagnostics._reset_diagnostic_sink_for_tests()


def test_drop_record_failure_preserves_loss_count_and_current_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = []
    failed_once = False

    def write(record) -> None:
        nonlocal failed_once
        records.append(record.copy())
        if record.get("event") == "diagnostics_events_dropped" and not failed_once:
            failed_once = True
            raise OSError("transient diagnostic sink failure")

    monkeypatch.setattr(diagnostics._AsyncDiagnosticSink, "_write", staticmethod(write))
    sink = diagnostics._AsyncDiagnosticSink(os.getpid())
    try:
        sink._record_dropped(3)
        sink.submit({"event": "first"})
        sink.submit({"event": "second"})
        sink.flush()
    finally:
        sink.close()

    assert [record["event"] for record in records] == [
        "diagnostics_events_dropped",
        "first",
        "diagnostics_events_dropped",
        "second",
    ]
    assert records[0]["dropped_events"] == records[2]["dropped_events"] == 3


def test_bounded_close_accounts_for_abandoned_queued_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "_DIAGNOSTIC_SHUTDOWN_TIMEOUT_S", 0.01)
    writer_started = threading.Event()
    release_writer = threading.Event()
    records = []

    def write(record) -> None:
        if record.get("event") == "first":
            writer_started.set()
            assert release_writer.wait(timeout=1.0)
        records.append(record.copy())

    monkeypatch.setattr(diagnostics._AsyncDiagnosticSink, "_write", staticmethod(write))
    sink = diagnostics._AsyncDiagnosticSink(os.getpid())
    try:
        sink.submit({"event": "first"})
        assert writer_started.wait(timeout=1.0)
        sink.submit({"event": "second"})
        sink.submit({"event": "third"})

        sink.close()
        assert sink._thread.is_alive()
        release_writer.set()
        sink._thread.join(timeout=1.0)
    finally:
        release_writer.set()
        sink.close()

    assert not sink._thread.is_alive()
    assert [record["event"] for record in records] == [
        "first",
        "diagnostics_events_dropped",
    ]
    assert records[-1]["dropped_events"] == 2


def test_sink_creation_failure_does_not_affect_request_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    get_sink = MagicMock(side_effect=RuntimeError("diagnostic sink unavailable"))
    monkeypatch.setattr(diagnostics, "_get_sink", get_sink)

    diagnostics.emit_event("gen_decode_ready", side="gen", request_id=42)

    get_sink.assert_called_once()


def test_full_diagnostic_queue_reports_dropped_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "_DIAGNOSTIC_QUEUE_CAPACITY", 1)
    writer_started = threading.Event()
    release_writer = threading.Event()
    records = []

    def write(record) -> None:
        records.append(record)
        if record.get("event") == "first":
            writer_started.set()
            assert release_writer.wait(timeout=1.0)

    monkeypatch.setattr(diagnostics._AsyncDiagnosticSink, "_write", staticmethod(write))
    sink = diagnostics._AsyncDiagnosticSink(os.getpid())
    try:
        sink.submit({"event": "first"})
        assert writer_started.wait(timeout=1.0)
        sink.submit({"event": "second"})
        sink.submit({"event": "dropped"})
        release_writer.set()
        sink.flush()
    finally:
        release_writer.set()
        sink.close()

    drop_record = next(
        record for record in records if record["event"] == "diagnostics_events_dropped"
    )
    assert drop_record["dropped_events"] == 1
    for record in records:
        assert record["process_uuid"] == sink._identity["process_uuid"]
        assert record["run_uuid"] is None
        assert record["run_uuid_status"] == "unset"


def test_scheduler_kv_admission_guard_avoids_telemetry_state_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
        KVCacheV2Scheduler,
        ScheduleAction,
    )

    inspection = MagicMock(side_effect=AssertionError("disabled diagnostics inspected state"))

    class _OpaqueRequest:
        @property
        def py_request_id(self) -> int:
            return inspection("py_request_id")

        @property
        def prompt_len(self) -> int:
            return inspection("prompt_len")

    class _KVCacheManager:
        def prepare_disagg_gen_init(self, _request: _OpaqueRequest) -> bool:
            return True

        @property
        def kv_cache_map(self) -> dict[int, object]:
            return inspection("kv_cache_map")

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = _KVCacheManager()
    scheduler.tokens_per_block = 32

    action, tokens = scheduler._try_schedule_disagg_gen_init(_OpaqueRequest(), None)

    assert action is ScheduleAction.SCHEDULED
    assert tokens == 0
    inspection.assert_not_called()


def test_scheduler_kv_admission_continues_when_diagnostic_inspection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
        KVCacheV2Scheduler,
        ScheduleAction,
    )

    class _KVCacheManager:
        def prepare_disagg_gen_init(self, _request) -> bool:
            return True

        @property
        def kv_cache_map(self):
            raise RuntimeError("diagnostic KV inspection failed")

    request = SimpleNamespace(py_request_id=17)
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = _KVCacheManager()
    scheduler.tokens_per_block = 32
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)

    action, tokens = scheduler._try_schedule_disagg_gen_init(request, None)

    assert action is ScheduleAction.SCHEDULED
    assert tokens == 0
