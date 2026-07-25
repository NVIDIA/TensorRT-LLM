# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from tensorrt_llm._torch.disaggregation import diagnostics
from tensorrt_llm._torch.disaggregation.diagnostics import (
    DisaggCorrelation,
    DisaggLifecycleDecision,
    DisaggLifecycleEmitter,
    DisaggLifecycleEmitterName,
    DisaggLifecycleEvent,
    DisaggLifecycleGate,
    DisaggLifecycleRole,
    DisaggLifecycleScheduleStyle,
    get_disagg_correlation,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue


def _parse_event(message: str) -> dict[str, str]:
    _, payload = message.split("] ", maxsplit=1)
    return dict(field.split("=", maxsplit=1) for field in payload.split())


def test_disabled_emitter_does_not_read_clocks_or_log(monkeypatch):
    monkeypatch.delenv(diagnostics.DISAGG_DIAGNOSTICS_ENV, raising=False)
    emitter = DisaggLifecycleEmitter.from_environment(rank=0, runtime="PYTHON", backend="NIXL")
    write_record = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", write_record)

    def unexpected_clock_read():
        raise AssertionError("disabled diagnostics read a clock")

    monkeypatch.setattr(diagnostics.time, "monotonic_ns", unexpected_clock_read)
    monkeypatch.setattr(diagnostics.time, "time_ns", unexpected_clock_read)

    emitter.emit(
        DisaggLifecycleEvent.GEN_ARRIVED,
        emitter=DisaggLifecycleEmitterName.EXECUTOR_QUEUE,
        role=DisaggLifecycleRole.GEN,
        correlation=DisaggCorrelation(1, 2, 3),
    )

    write_record.assert_not_called()


def test_enabled_stream_does_not_depend_on_general_logger_level(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="PYTHON",
        backend="NIXL",
        clock_id="test-clock",
        async_output=False,
    )
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: 101)
    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: 201)
    write_record = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", write_record)
    log_info = Mock(side_effect=AssertionError("general logger was used"))
    monkeypatch.setattr(diagnostics.logger, "info", log_info)

    emitter.emit(
        DisaggLifecycleEvent.GEN_ARRIVED,
        emitter=DisaggLifecycleEmitterName.EXECUTOR_QUEUE,
        role=DisaggLifecycleRole.GEN,
        correlation=DisaggCorrelation(1, 2, 3),
    )

    write_record.assert_called_once()
    log_info.assert_not_called()


def test_async_writer_drops_on_saturation_without_blocking(monkeypatch):
    writer_entered = threading.Event()
    release_writer = threading.Event()
    written_records = []

    def blocked_writer(record):
        writer_entered.set()
        assert release_writer.wait(timeout=5)
        written_records.append(record)

    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", blocked_writer)
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="PYTHON",
        backend="NIXL",
        clock_id="test-clock",
        queue_capacity=1,
    )
    event_kwargs = {
        "emitter": DisaggLifecycleEmitterName.PYEXECUTOR,
        "role": DisaggLifecycleRole.GEN,
        "correlation": DisaggCorrelation(1, 2, 3),
    }

    emitter.emit(DisaggLifecycleEvent.GEN_ARRIVED, **event_kwargs)
    assert writer_entered.wait(timeout=5)
    emitter.emit(DisaggLifecycleEvent.GEN_DEQUEUED, **event_kwargs)
    publisher = threading.Thread(
        target=emitter.emit,
        args=(DisaggLifecycleEvent.GEN_DISPATCHED,),
        kwargs=event_kwargs,
    )
    publisher.start()
    publisher.join(timeout=1)
    writer_thread = emitter._writer_thread

    assert not publisher.is_alive()
    assert writer_thread is not None
    assert emitter.dropped_record_count == 1
    close_result = []
    closer = threading.Thread(target=lambda: close_result.append(emitter.close(timeout=0.01)))
    closer.start()
    closer.join(timeout=1)
    assert not closer.is_alive()
    assert close_result == [False]
    release_writer.set()
    writer_thread.join(timeout=1)
    assert not writer_thread.is_alive()
    assert any("event=records_dropped" in record for record in written_records)
    assert emitter.close(timeout=0)


def test_async_writer_close_drains_tail_and_stops(monkeypatch):
    written_records = []
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", written_records.append)
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="PYTHON",
        backend="NIXL",
        clock_id="test-clock",
        queue_capacity=4,
    )
    writer_thread = emitter._writer_thread
    event_kwargs = {
        "emitter": DisaggLifecycleEmitterName.PYEXECUTOR,
        "role": DisaggLifecycleRole.GEN,
        "correlation": DisaggCorrelation(1, 2, 3),
    }

    emitter.emit(DisaggLifecycleEvent.GEN_ARRIVED, **event_kwargs)
    emitter.emit(DisaggLifecycleEvent.GEN_DEQUEUED, **event_kwargs)

    assert emitter.close(timeout=1)
    assert writer_thread is not None
    assert not writer_thread.is_alive()
    assert [
        record.split("event=", maxsplit=1)[1].split(maxsplit=1)[0] for record in written_records
    ] == ["gen_arrived", "gen_dequeued"]
    assert emitter.close(timeout=0)


def test_async_writer_reports_sink_failure_once(monkeypatch):
    sink_calls = 0

    def failing_writer(_record):
        nonlocal sink_calls
        sink_calls += 1
        raise RuntimeError("sink unavailable")

    warning = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", failing_writer)
    monkeypatch.setattr(diagnostics.logger, "warning", warning)
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="PYTHON",
        backend="NIXL",
        clock_id="test-clock",
        queue_capacity=4,
    )
    event_kwargs = {
        "emitter": DisaggLifecycleEmitterName.PYEXECUTOR,
        "role": DisaggLifecycleRole.GEN,
        "correlation": DisaggCorrelation(1, 2, 3),
    }

    emitter.emit(DisaggLifecycleEvent.GEN_ARRIVED, **event_kwargs)
    emitter.emit(DisaggLifecycleEvent.GEN_DEQUEUED, **event_kwargs)

    assert emitter.close(timeout=1)
    assert sink_calls == 2
    warning.assert_called_once()


def test_async_queue_failure_does_not_prevent_enqueue(monkeypatch):
    warning = Mock()
    monkeypatch.setattr(diagnostics.logger, "warning", warning)
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="PYTHON",
        backend="NIXL",
        clock_id="test-clock",
        queue_capacity=4,
    )
    assert emitter._output_queue is not None
    monkeypatch.setattr(
        emitter._output_queue,
        "put_nowait",
        Mock(side_effect=RuntimeError("queue unavailable")),
    )
    request_queue = ExecutorRequestQueue(
        dist=Mock(),
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
        disagg_lifecycle_emitter=emitter,
    )
    request = SimpleNamespace(
        disagg_request_id=42,
        request_type=SimpleNamespace(name="REQUEST_TYPE_GENERATION_ONLY"),
        sampling_config=SimpleNamespace(beam_width=1, num_return_sequences=1),
        disaggregated_params=None,
    )

    assert request_queue.enqueue_request(request) == 42
    assert request_queue.request_queue.get_nowait().id == 42
    warning.assert_called_once()
    assert emitter.close(timeout=1)


def test_correlation_does_not_fall_back_between_id_domains():
    request = SimpleNamespace(
        disagg_request_id=None,
        ctx_request_id=17,
        request_id=29,
        disaggregated_params=None,
        context_phase_params=None,
    )

    correlation = get_disagg_correlation(request)

    assert correlation == DisaggCorrelation(
        disagg_request_id=None,
        ctx_request_id=17,
        local_request_id=29,
    )


def test_context_phase_request_id_is_not_relabelled_as_context_request_id():
    request = SimpleNamespace(
        disagg_request_id=42,
        ctx_request_id=None,
        request_id=29,
        disaggregated_params=None,
        context_phase_params=SimpleNamespace(req_id=42),
    )

    correlation = get_disagg_correlation(request)

    assert correlation == DisaggCorrelation(
        disagg_request_id=42,
        ctx_request_id=None,
        local_request_id=29,
    )


def test_emitter_writes_versioned_ordered_events(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=3,
        runtime="PYTHON",
        backend="NIXL",
        clock_id="test-clock",
        async_output=False,
    )
    monotonic_values = iter((101, 102))
    wall_values = iter((201, 202))
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: next(wall_values))
    write_record = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", write_record)

    for decision in (DisaggLifecycleDecision.DEFER, DisaggLifecycleDecision.ADMIT):
        emitter.emit(
            DisaggLifecycleEvent.GEN_ADMISSION_CHANGED,
            emitter=DisaggLifecycleEmitterName.PYEXECUTOR,
            role=DisaggLifecycleRole.GEN,
            correlation=DisaggCorrelation(11, 12, 13),
            gate=DisaggLifecycleGate.TRANSFER,
            decision=decision,
            schedule_style=DisaggLifecycleScheduleStyle.CONTEXT_FIRST,
        )

    first = _parse_event(write_record.call_args_list[0].args[0])
    second = _parse_event(write_record.call_args_list[1].args[0])
    assert first["schema"] == "1"
    assert first["event"] == "gen_admission_changed"
    assert first["mono_ns"] == "101"
    assert first["wall_ns"] == "201"
    assert first["clock_id"] == "test-clock"
    assert first["seq"] == "1"
    assert first["runtime"] == "python"
    assert first["backend"] == "nixl"
    assert first["rank"] == "3"
    assert first["disagg_request_id"] == "11"
    assert first["ctx_request_id"] == "12"
    assert first["local_request_id"] == "13"
    assert first["decision"] == "defer"
    assert first["schedule_style"] == "context_first"
    assert second["seq"] == "2"
    assert second["decision"] == "admit"


def test_rank_filter_is_applied_when_emitter_is_created(monkeypatch):
    monkeypatch.setenv(diagnostics.DISAGG_DIAGNOSTICS_ENV, "1")
    monkeypatch.setenv(diagnostics.DISAGG_DIAGNOSTICS_RANKS_ENV, "0,2")

    assert DisaggLifecycleEmitter.from_environment(rank=0, runtime="CPP", backend="NIXL").enabled
    assert not DisaggLifecycleEmitter.from_environment(
        rank=1, runtime="CPP", backend="NIXL"
    ).enabled


def test_request_state_is_named_and_domain_qualified(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="PYTHON",
        backend="NIXL",
        clock_id="test-clock",
        async_output=False,
    )
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: 101)
    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: 201)
    write_record = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", write_record)
    request = SimpleNamespace(
        state=SimpleNamespace(name="DISAGG_GENERATION_INIT"),
        disagg_request_id=42,
        ctx_request_id=None,
        request_id=29,
        disaggregated_params=None,
    )

    emitter.emit_for_request(
        DisaggLifecycleEvent.GEN_LOCAL_SCHEDULER_ACTIVATED,
        request,
        emitter=DisaggLifecycleEmitterName.PYEXECUTOR,
        role=DisaggLifecycleRole.GEN,
    )

    event = _parse_event(write_record.call_args.args[0])
    assert event["local_state"] == "disagg_generation_init"
    assert event["state_domain"] == "llm_request"


def test_generation_arrival_timestamp_precedes_engine_queue_insert(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="CPP",
        backend="NIXL",
        clock_id="test-clock",
        async_output=False,
    )
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: 101)
    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: 201)
    write_record = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", write_record)
    request_queue = ExecutorRequestQueue(
        dist=Mock(),
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
        disagg_lifecycle_emitter=emitter,
    )
    request_queue.request_queue = Mock()
    capture_for_role = emitter.capture_for_role
    capture_stamp = emitter.capture_stamp
    call_order = []

    def capture_outside_enqueue_lock(**kwargs):
        assert not request_queue.enqueue_lock.locked()
        return capture_for_role(**kwargs)

    def stamp_before_insert():
        call_order.append("stamp")
        return capture_stamp()

    monkeypatch.setattr(emitter, "capture_for_role", capture_outside_enqueue_lock)
    monkeypatch.setattr(emitter, "capture_stamp", stamp_before_insert)

    def insert_after_stamp(_):
        call_order.append("put")
        write_record.assert_not_called()

    request_queue.request_queue.put.side_effect = insert_after_stamp
    request = SimpleNamespace(
        disagg_request_id=42,
        ctx_request_id=None,
        request_type=SimpleNamespace(name="REQUEST_TYPE_GENERATION_ONLY"),
        sampling_config=SimpleNamespace(beam_width=1, num_return_sequences=1),
        disaggregated_params=None,
        context_phase_params=None,
        schedule_style=SimpleNamespace(name="GENERATION_FIRST"),
    )

    assert request_queue.enqueue_request(request) == 42

    assert call_order[:2] == ["stamp", "put"]
    event = _parse_event(write_record.call_args.args[0])
    assert event["event"] == "gen_arrived"
    assert event["runtime"] == "cpp"
    assert event["disagg_request_id"] == "42"
    assert event["local_request_id"] == "42"
    assert event["schedule_style"] == "generation_first"


def test_root_queue_segments_share_ingress_clock(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="CPP",
        backend="NIXL",
        clock_id="ingress-clock",
        async_output=False,
    )
    monotonic_values = iter((101, 102, 202, 601))
    wall_values = iter((201, 202, 302, 701))
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: next(wall_values))
    write_record = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", write_record)
    request_queue = ExecutorRequestQueue(
        dist=Mock(),
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
        disagg_lifecycle_emitter=emitter,
    )
    request = SimpleNamespace(
        disagg_request_id=42,
        ctx_request_id=None,
        request_type=SimpleNamespace(name="REQUEST_TYPE_GENERATION_ONLY"),
        sampling_config=SimpleNamespace(beam_width=1, num_return_sequences=1),
        disaggregated_params=None,
        context_phase_params=None,
        schedule_style=SimpleNamespace(name="GENERATION_FIRST"),
    )

    request_queue.enqueue_request(request)
    item = request_queue.request_queue.get_nowait()
    request_queue.emit_disagg_dequeue_events([item])
    request_queue.emit_disagg_dispatch_events([item])
    request_queue.emit_disagg_waiting_release_events([item])

    arrival = _parse_event(write_record.call_args_list[0].args[0])
    dequeue = _parse_event(write_record.call_args_list[1].args[0])
    dispatch = _parse_event(write_record.call_args_list[2].args[0])
    waiting_release = _parse_event(write_record.call_args_list[3].args[0])
    assert arrival["event"] == "gen_arrived"
    assert dequeue["event"] == "gen_dequeued"
    assert dispatch["event"] == "gen_dispatched"
    assert waiting_release["event"] == "gen_waiting_released"
    assert (
        arrival["clock_id"]
        == dequeue["clock_id"]
        == dispatch["clock_id"]
        == waiting_release["clock_id"]
        == "ingress-clock"
    )
    assert int(dequeue["mono_ns"]) > int(arrival["mono_ns"])
    assert int(dispatch["mono_ns"]) - int(dequeue["mono_ns"]) == 100
    assert int(waiting_release["mono_ns"]) - int(dispatch["mono_ns"]) == 399


def test_nonzero_adp_scheduler_segment_uses_one_local_clock(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=3,
        runtime="CPP",
        backend="NIXL",
        clock_id="scheduler-clock",
        async_output=False,
    )
    monotonic_values = iter((701, 901))
    wall_values = iter((801, 1001))
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: next(wall_values))
    write_record = Mock()
    monkeypatch.setattr(diagnostics, "_write_lifecycle_record", write_record)
    request = SimpleNamespace(
        state=SimpleNamespace(name="DISAGG_GENERATION_INIT"),
        disagg_request_id=42,
        ctx_request_id=None,
        py_request_id=42,
        py_llm_request_type=SimpleNamespace(name="LLMREQUEST_TYPE_GENERATION_ONLY"),
        py_disaggregated_params=None,
    )

    emitter.emit_for_role(
        ctx_event=DisaggLifecycleEvent.CTX_LOCAL_SCHEDULER_ACTIVATED,
        gen_event=DisaggLifecycleEvent.GEN_LOCAL_SCHEDULER_ACTIVATED,
        request=request,
        emitter=DisaggLifecycleEmitterName.PYEXECUTOR,
    )
    emitter.emit_for_request(
        DisaggLifecycleEvent.GEN_ADMISSION_CHANGED,
        request,
        emitter=DisaggLifecycleEmitterName.PYEXECUTOR,
        role=DisaggLifecycleRole.GEN,
        gate=DisaggLifecycleGate.SCHEDULER,
        decision=DisaggLifecycleDecision.ELIGIBLE,
    )

    activation = _parse_event(write_record.call_args_list[0].args[0])
    eligibility = _parse_event(write_record.call_args_list[1].args[0])
    assert activation["rank"] == eligibility["rank"] == "3"
    assert activation["clock_id"] == eligibility["clock_id"] == "scheduler-clock"
    assert int(eligibility["mono_ns"]) - int(activation["mono_ns"]) == 200


def test_diagnostic_failures_do_not_prevent_enqueue(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="CPP",
        backend="NIXL",
        clock_id="test-clock",
        async_output=False,
    )
    monkeypatch.setattr(
        diagnostics.time,
        "monotonic_ns",
        Mock(side_effect=RuntimeError("clock unavailable")),
    )
    monkeypatch.setattr(diagnostics.logger, "warning", Mock())
    request_queue = ExecutorRequestQueue(
        dist=Mock(),
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
        disagg_lifecycle_emitter=emitter,
    )
    request = SimpleNamespace(
        disagg_request_id=42,
        request_type=SimpleNamespace(name="REQUEST_TYPE_GENERATION_ONLY"),
        sampling_config=SimpleNamespace(beam_width=1, num_return_sequences=1),
        disaggregated_params=None,
    )

    assert request_queue.enqueue_request(request) == 42
    assert request_queue.request_queue.get_nowait().id == 42


def test_logger_failure_does_not_prevent_enqueue(monkeypatch):
    emitter = DisaggLifecycleEmitter(
        enabled=True,
        rank=0,
        runtime="CPP",
        backend="NIXL",
        clock_id="test-clock",
        async_output=False,
    )
    monkeypatch.setattr(diagnostics.time, "monotonic_ns", lambda: 101)
    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: 201)
    monkeypatch.setattr(
        diagnostics,
        "_write_lifecycle_record",
        Mock(side_effect=RuntimeError("logger unavailable")),
    )
    monkeypatch.setattr(diagnostics.logger, "warning", Mock())
    request_queue = ExecutorRequestQueue(
        dist=Mock(),
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
        disagg_lifecycle_emitter=emitter,
    )
    request = SimpleNamespace(
        disagg_request_id=42,
        request_type=SimpleNamespace(name="REQUEST_TYPE_GENERATION_ONLY"),
        sampling_config=SimpleNamespace(beam_width=1, num_return_sequences=1),
        disaggregated_params=None,
    )

    assert request_queue.enqueue_request(request) == 42
    assert request_queue.request_queue.get_nowait().id == 42
