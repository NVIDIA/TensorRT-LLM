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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.disaggregation import diagnostics

pytestmark = pytest.mark.cpu_only


def test_disabled_emit_event_does_no_diagnostic_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)

    unexpected_call = MagicMock(side_effect=AssertionError("disabled diagnostics did work"))
    monkeypatch.setattr(diagnostics, "_host_identity", unexpected_call)
    monkeypatch.setattr(diagnostics, "_get_sink", unexpected_call)
    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(getpid=unexpected_call, write=unexpected_call),
    )
    monkeypatch.setattr(
        diagnostics,
        "time",
        SimpleNamespace(monotonic_ns=unexpected_call, time_ns=unexpected_call),
    )
    monkeypatch.setattr(diagnostics, "json", SimpleNamespace(dumps=unexpected_call))

    diagnostics.emit_event("ctx_send_ready", side="ctx", request_id=17)

    unexpected_call.assert_not_called()


def test_enabled_emit_event_records_request_and_rank_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics._reset_diagnostic_sink_for_tests()
    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(diagnostics, "_host_identity", lambda: "node-a")
    request_thread_id = threading.get_ident()
    writes = []
    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(
            getpid=lambda: 321,
            write=lambda fd, data: writes.append((threading.get_ident(), fd, data)),
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
    assert json.loads(payload) == {
        "schema_version": diagnostics.DIAGNOSTICS_SCHEMA_VERSION,
        "event": "ctx_backend_submitted",
        "side": "ctx",
        "request_id": 99,
        "local_request_id": 7,
        "host": "node-a",
        "pid": 321,
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
    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(
            getpid=lambda: 654,
            write=lambda fd, data: writes.append((fd, data)),
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
        ),
    )

    diagnostics.emit_event("gen_decode_ready", side="gen", request_id=42)
    diagnostics._flush_diagnostic_sink_for_tests()
    sink = diagnostics._sink
    assert sink is not None
    assert sink._thread.is_alive()

    diagnostics.emit_event("gen_decode_ready", side="gen", request_id=43)
    diagnostics._flush_diagnostic_sink_for_tests()

    assert len(writes) == 2
    assert b'"request_id":43' in writes[1][1]
    diagnostics._reset_diagnostic_sink_for_tests()


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


def test_scheduler_kv_admission_guard_avoids_telemetry_state_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
        KVCacheV2Scheduler,
        ScheduleAction,
    )

    class _OpaqueRequest:
        @property
        def py_request_id(self) -> int:
            raise AssertionError("disabled diagnostics inspected the request")

        @property
        def prompt_len(self) -> int:
            raise AssertionError("disabled diagnostics inspected the request")

    class _KVCacheManager:
        def prepare_disagg_gen_init(self, _request: _OpaqueRequest) -> bool:
            return True

        @property
        def kv_cache_map(self) -> dict[int, object]:
            raise AssertionError("disabled diagnostics inspected the KV cache map")

    monkeypatch.setattr(diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False)
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = _KVCacheManager()
    scheduler.tokens_per_block = 32

    action, tokens = scheduler._try_schedule_disagg_gen_init(_OpaqueRequest(), None)

    assert action is ScheduleAction.SCHEDULED
    assert tokens == 0
