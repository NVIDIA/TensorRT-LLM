# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the serve-layer orchestrator lifecycle emitter."""

import json

import pytest

from tensorrt_llm.serve.disagg_lifecycle import (
    DisaggOrchestratorLifecycle,
    OrchestratorEvent,
    OrchestratorScheduleStyle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emitter(enabled: bool = True, node_id: str = "n0") -> DisaggOrchestratorLifecycle:
    return DisaggOrchestratorLifecycle(enabled=enabled, node_id=node_id)


def _capture(capsys):
    """Return parsed JSON records from stdout since last capture."""
    out = capsys.readouterr().out
    records = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("[DISAGG_DIAG][orchestrator]"):
            payload = line[len("[DISAGG_DIAG][orchestrator]") :].strip()
            records.append(json.loads(payload))
    return records


# ---------------------------------------------------------------------------
# Disabled emitter
# ---------------------------------------------------------------------------


class TestDisabledEmitter:
    def test_disabled_produces_no_output(self, capsys):
        em = _make_emitter(enabled=False)
        em.emit(OrchestratorEvent.CTX_DISPATCH, disagg_request_id=1)
        records = _capture(capsys)
        assert records == []

    def test_disabled_tracer_is_noop(self, capsys):
        em = _make_emitter(enabled=False)
        tracer = em.tracer(disagg_request_id=42)
        tracer.ctx_dispatch("ctx:8000")
        tracer.ctx_complete()
        tracer.gen_dispatch("gen:9000")
        tracer.gen_complete()
        records = _capture(capsys)
        assert records == []


# ---------------------------------------------------------------------------
# Enabled emitter — basic emit
# ---------------------------------------------------------------------------


class TestEnabledEmitter:
    def test_emit_contains_required_fields(self, capsys):
        em = _make_emitter()
        em.emit(OrchestratorEvent.CTX_DISPATCH, disagg_request_id=7, ctx_server="h:8")
        records = _capture(capsys)
        assert len(records) == 1
        r = records[0]
        assert r["event"] == "ctx_dispatch"
        assert r["rid"] == 7
        assert r["ctx"] == "h:8"
        assert "wall_ns" in r
        assert "seq" in r
        assert r["v"] == 1

    def test_sequence_increments(self, capsys):
        em = _make_emitter()
        em.emit(OrchestratorEvent.CTX_DISPATCH)
        em.emit(OrchestratorEvent.GEN_DISPATCH)
        records = _capture(capsys)
        assert records[1]["seq"] == records[0]["seq"] + 1

    def test_optional_fields_absent_when_none(self, capsys):
        em = _make_emitter()
        em.emit(OrchestratorEvent.ABORT, error="oops")
        records = _capture(capsys)
        r = records[0]
        assert "ctx" not in r
        assert "gen" not in r
        assert r["error"] == "oops"

    def test_http_status_included(self, capsys):
        em = _make_emitter()
        em.emit(OrchestratorEvent.GEN_REJECTED, http_status=503, error="no capacity")
        records = _capture(capsys)
        r = records[0]
        assert r["event"] == "gen_rejected"
        assert r["http_status"] == 503

    def test_error_truncated_to_256(self, capsys):
        em = _make_emitter()
        long_error = "x" * 512
        em.emit(OrchestratorEvent.ABORT, error=long_error)
        records = _capture(capsys)
        assert len(records[0]["error"]) == 256


# ---------------------------------------------------------------------------
# Tracer — paired events
# ---------------------------------------------------------------------------


class TestTracer:
    def test_ctx_dispatch_complete(self, capsys):
        em = _make_emitter()
        tracer = em.tracer(
            disagg_request_id=99, schedule_style=OrchestratorScheduleStyle.CONTEXT_FIRST
        )
        tracer.ctx_dispatch("ctx:7000")
        tracer.ctx_complete(ctx_server="ctx:7000")
        records = _capture(capsys)
        assert len(records) == 2
        assert records[0]["event"] == "ctx_dispatch"
        assert records[1]["event"] == "ctx_complete"
        assert records[0]["rid"] == 99
        assert records[1]["sched"] == "context_first"
        # ctx_complete elapsed_ms is measured from ctx_dispatch, not tracer start
        assert records[1]["elapsed_ms"] is not None
        assert records[1]["elapsed_ms"] >= 0

    def test_gen_error_with_http_status(self, capsys):
        em = _make_emitter()
        tracer = em.tracer(disagg_request_id=5)
        tracer.gen_error("worker unavailable", gen_server="gen:9000", http_status=503)
        records = _capture(capsys)
        assert len(records) == 1
        r = records[0]
        assert r["event"] == "gen_rejected"
        assert r["http_status"] == 503
        assert r["gen"] == "gen:9000"

    def test_gen_error_without_http_status(self, capsys):
        em = _make_emitter()
        tracer = em.tracer()
        tracer.gen_error("transport error")
        records = _capture(capsys)
        assert records[0]["event"] == "gen_error"
        assert "http_status" not in records[0]

    def test_abort(self, capsys):
        em = _make_emitter()
        tracer = em.tracer(disagg_request_id=1)
        tracer.abort("client disconnected")
        records = _capture(capsys)
        assert records[0]["event"] == "abort"
        assert "disconnected" in records[0]["error"]

    def test_client_disconnect(self, capsys):
        em = _make_emitter()
        tracer = em.tracer()
        tracer.client_disconnect()
        records = _capture(capsys)
        assert records[0]["event"] == "client_disconnect"


# ---------------------------------------------------------------------------
# from_environment
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_disabled_by_default(self, monkeypatch, capsys):
        monkeypatch.delenv("TRTLLM_DISAGG_ORCHESTRATOR_DIAGNOSTICS", raising=False)
        em = DisaggOrchestratorLifecycle.from_environment()
        assert not em.enabled
        em.emit(OrchestratorEvent.CTX_DISPATCH)
        assert _capture(capsys) == []

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE"])
    def test_enabled_by_env(self, monkeypatch, val, capsys):
        monkeypatch.setenv("TRTLLM_DISAGG_ORCHESTRATOR_DIAGNOSTICS", val)
        em = DisaggOrchestratorLifecycle.from_environment()
        assert em.enabled
        em.emit(OrchestratorEvent.CTX_DISPATCH)
        assert len(_capture(capsys)) == 1

    @pytest.mark.parametrize("val", ["0", "", "false", "False", "FALSE"])
    def test_disabled_by_env(self, monkeypatch, val, capsys):
        monkeypatch.setenv("TRTLLM_DISAGG_ORCHESTRATOR_DIAGNOSTICS", val)
        em = DisaggOrchestratorLifecycle.from_environment()
        assert not em.enabled
