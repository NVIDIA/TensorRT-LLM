# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU contract tests for the shared profiler primitives."""

import builtins
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import TracebackType
from typing import cast

import pytest
import torch
from profiler_test_utils import (
    decode_node_metadata,
    decode_profiler_metadata,
    execution_trace_observer,
    profiler_nodes,
    select_regions,
)

from tensorrt_llm import _profiler as profiler

pytestmark = pytest.mark.cpu_only


def test_json_envelope_has_separate_version_and_utf8_region_length() -> None:
    """A JSON consumer must decode framing and values without Python literals."""
    fields = {"text": "snow 雪\x00", "nested": {"ok": True, "value": None}, "ids": [[1, -2]]}
    payload = profiler.encode_metadata("区域", fields)
    assert payload.startswith("TRTLLM1:6:区域")
    assert json.loads(payload.removeprefix("TRTLLM1:6:区域")) == fields
    assert decode_profiler_metadata(payload) == ("区域", fields)


@pytest.mark.parametrize("kind", ["list", "dict", "nested-key"])
def test_metadata_rejects_cycles_and_nested_nonstring_keys(kind: str) -> None:
    """Reject malformed values before recording; do not coerce JSON object keys."""
    value: object
    if kind == "list":
        value = []
        value.append(value)
    elif kind == "dict":
        value = {}
        value["cycle"] = value
    else:
        value = {"nested": {1: "ambiguous"}}
    with pytest.raises((TypeError, ValueError)):
        profiler.encode_metadata("workflow", {"value": cast(profiler.MetadataValue, value)})


def test_failed_metadata_keeps_named_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken metadata producer must retain the observed operation's interval."""
    select_regions(monkeypatch, "workflow")

    def broken() -> dict[str, profiler.MetadataValue]:
        raise ValueError("metadata failed")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with profiler.record_function_metadata_scope("named_without_metadata", "workflow", broken):
            result = 42
    assert result == 42
    assert sum(event.name == "named_without_metadata" for event in trace.events()) == 1


@pytest.mark.parametrize(
    "case", json.loads(Path(__file__).with_name("profiler_selector_cases.json").read_text())
)
def test_selector_matches_shared_golden_table(
    case: dict[str, object], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Python and native parsing use the same valid and invalid startup examples."""
    selection = cast(str | None, case["value"])
    expected = frozenset(cast(list[str], case["expected"]))
    if case["valid"]:
        assert profiler._parse_custom_regions(selection) == expected
    else:
        with pytest.raises(ValueError, match="TLLM_PROFILE_CUSTOM_REGIONS"):
            profiler._parse_custom_regions(selection)
    if selection is None:
        monkeypatch.delenv("TLLM_PROFILE_CUSTOM_REGIONS", raising=False)
    else:
        monkeypatch.setenv("TLLM_PROFILE_CUSTOM_REGIONS", selection)
    assert profiler._load_custom_regions() == expected
    assert ("using default custom regions" in caplog.text) == (not case["valid"])


def test_selection_is_cached_at_module_initialization() -> None:
    env = os.environ.copy()
    env["TLLM_PROFILE_CUSTOM_REGIONS"] = "workflow,moe"
    script = (
        "import os, runpy; "
        f"p = runpy.run_path({str(Path(profiler.__file__).resolve())!r}); "
        "os.environ['TLLM_PROFILE_CUSTOM_REGIONS'] = 'none'; "
        "assert p['custom_region_enabled']('workflow'); "
        "assert p['custom_region_enabled']('moe'); "
        "assert not p['custom_region_enabled']('kv')"
    )
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_invalid_startup_configuration_warns_and_uses_defaults() -> None:
    """Verify invalid startup configuration warns and enables defaults."""
    env = os.environ.copy()
    env["TLLM_PROFILE_CUSTOM_REGIONS"] = "workflow,unknown"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"p = runpy.run_path({str(Path(profiler.__file__).resolve())!r}); "
                "assert p['_CUSTOM_REGIONS'] == p['_CUSTOM_REGION_DEFAULTS']"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr.count("Invalid TLLM_PROFILE_CUSTOM_REGIONS") == 1


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            "TRTLLM18:workflow{'quoted\\'key':'slash\\\\line\\n\\t\\'雪','rank':-7,'ids':[]}",
            {"quoted'key": "slash\\line\n\t'雪", "rank": -7, "ids": []},
        ),
        (
            "TRTLLM16:区域{'nested':{'values':[None,True,1.5,('x',-2)]}}",
            {"nested": {"values": [None, True, 1.5, ("x", -2)]}},
        ),
    ],
)
def test_decoder_handles_native_and_unicode_golden_payloads(
    payload: str, expected: dict[str, profiler.MetadataValue]
) -> None:
    region, metadata = decode_profiler_metadata(payload)
    assert region in {"workflow", "区域"}
    assert metadata == expected


def test_python_encoder_preserves_supported_values() -> None:
    metadata: dict[str, profiler.MetadataValue] = {
        "quoted'key": "slash\\line\n\t'雪\x00",
        "rank": -(2**63),
        "ids": [],
        "nested": {"values": [None, True, 1.5, ["x", -2]]},
    }
    assert decode_profiler_metadata(profiler.encode_metadata("workflow", metadata)) == (
        "workflow",
        metadata,
    )


@pytest.mark.parametrize("value", [object(), {"x"}, float("inf"), float("nan")])
def test_encoder_rejects_values_outside_literal_contract(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        profiler.encode_metadata("workflow", {"value": cast(profiler.MetadataValue, value)})


def test_encoder_rejects_builtin_subclasses_with_custom_repr() -> None:
    class CustomInt(int):
        def __repr__(self) -> str:
            return "not a literal"

    with pytest.raises(TypeError):
        profiler.encode_metadata("workflow", {"value": CustomInt(1)})


@pytest.mark.parametrize("payload", ["bad", "TRTLLM1-1:x{}", "TRTLLM18:work{}", "TRTLLM12:kv[]"])
def test_decoder_rejects_invalid_envelopes(payload: str) -> None:
    with pytest.raises(ValueError):
        decode_profiler_metadata(payload)


@pytest.mark.parametrize("enabled", [False, True])
def test_inactive_scopes_do_not_evaluate_metadata(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """Avoid metadata evaluation while the PyTorch profiler is inactive."""
    select_regions(monkeypatch, *(("workflow",) if enabled else ()))
    calls = 0

    def forbidden() -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("inactive factory evaluated")

    with profiler.record_function_metadata_scope("inactive", "workflow", forbidden):
        result = 42
    assert result == 42
    assert calls == 0


def test_none_selection_emits_no_custom_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress custom events and metadata when no regions are selected."""
    select_regions(monkeypatch)
    calls = 0

    def forbidden() -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("disabled factory evaluated")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with profiler.record_function_metadata_scope("disabled", "workflow", forbidden):
            result = torch.ones(1) + 1
    assert result.item() == 2
    assert calls == 0
    assert "disabled" not in {event.name for event in trace.events()}


def test_active_scope_preserves_result_and_records_real_cpu_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Record a real CPU event while preserving the observed operation."""
    select_regions(monkeypatch, "workflow")
    calls = 0

    def metadata() -> dict[str, profiler.MetadataValue]:
        nonlocal calls
        calls += 1
        return {"rank": -7, "ids": []}

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with profiler.record_function_metadata_scope("foundation", "workflow", metadata):
            result = torch.ones(1) + 1
    assert result.item() == 2
    assert calls == 1
    trace_file = tmp_path / "trace.json"
    trace.export_chrome_trace(str(trace_file))
    events = json.loads(trace_file.read_text())["traceEvents"]
    assert any(event.get("name") == "foundation" for event in events)


@pytest.mark.skipif(
    not hasattr(torch.profiler, "ExecutionTraceObserver"), reason="requires ExecutionTraceObserver"
)
def test_real_execution_trace_receives_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve typed fields in a real execution-trace observer output."""
    select_regions(monkeypatch, "workflow")
    trace_file = tmp_path / "execution.json"
    with execution_trace_observer(trace_file) as observer:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
        ):
            with profiler.record_function_metadata_scope(
                "payload", "workflow", lambda: {"rank": -7, "ids": []}
            ):
                torch.ones(1)
    (node,) = profiler_nodes(trace_file, "payload")
    assert decode_node_metadata(node) == (
        "workflow",
        {"rank": -7, "ids": []},
    )


@pytest.mark.skipif(
    not hasattr(torch.profiler, "ExecutionTraceObserver"), reason="requires ExecutionTraceObserver"
)
def test_real_execution_trace_preserves_preencoded_metadata_and_body_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve encoded bytes, body side effects, and return identity."""
    select_regions(monkeypatch, "workflow")
    payload = "TRTLLM18:workflow{'rank':-7,'ids':[],'label':'preencoded'}"
    result = object()
    effects: list[str] = []

    @profiler.record_function_metadata_scope("preencoded", "workflow", lambda: payload)
    def operation() -> object:
        effects.append("body")
        torch.ones(1)
        return result

    trace_file = tmp_path / "preencoded-execution.json"
    with execution_trace_observer(trace_file) as observer:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
        ):
            assert operation() is result
    assert effects == ["body"]
    nodes = profiler_nodes(trace_file, "preencoded")
    assert len(nodes) == 1
    recorded = nodes[0]["inputs"]["values"][0]
    assert recorded == payload
    assert decode_profiler_metadata(recorded) == (
        "workflow",
        {"rank": -7, "ids": [], "label": "preencoded"},
    )


def test_contextmanager_decorator_preserves_arguments_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve decorated call arguments, result, and CPU scope emission."""
    select_regions(monkeypatch, "workflow")

    @profiler.record_function_metadata_scope("decorated", "workflow", lambda: {"rank": -7})
    def wrapped(value: int, *, offset: int) -> int:
        return value + offset

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        assert wrapped(4, offset=2) == 6
    assert "decorated" in {event.name for event in trace.events()}


def test_metadata_failures_log_once_per_scope_at_warning_level(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A repeated metadata error must be visible without flooding the logs."""
    from tensorrt_llm.logger import logger

    select_regions(monkeypatch, "workflow")
    monkeypatch.setattr(logger, "_appeared_keys", set())
    monkeypatch.setattr(logger, "_min_severity", "debug")
    monkeypatch.setattr(logger, "_module_levels", {})
    diagnostic_logger = logging.getLogger(f"{__name__}.metadata_failures")
    monkeypatch.setattr(logger, "_logger", diagnostic_logger)
    original = RuntimeError("workload failed")

    def metadata() -> dict[str, profiler.MetadataValue]:
        raise TypeError("unexpected metadata keyword")

    with caplog.at_level(logging.DEBUG, logger=diagnostic_logger.name):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
            for _ in range(2):
                with profiler.record_function_metadata_scope(
                    "broken_metadata", "workflow", metadata
                ):
                    result = 42
            with pytest.raises(RuntimeError) as caught:
                with profiler.record_function_metadata_scope(
                    "broken_metadata", "workflow", metadata
                ):
                    raise original
            with profiler.record_function_metadata_scope("another_scope", "workflow", metadata):
                pass
            with pytest.raises(RuntimeError):
                with profiler.record_function_metadata_scope(
                    "body_failure", "workflow", lambda: {}
                ):
                    raise original

    assert result == 42
    assert caught.value is original
    records = [record for record in caplog.records if record.name == diagnostic_logger.name]
    assert len(records) == 2
    assert all(record.levelno == logging.WARNING for record in records)
    assert all("unexpected metadata keyword" in record.getMessage() for record in records)
    assert sum("broken_metadata" in record.getMessage() for record in records) == 1
    assert sum("another_scope" in record.getMessage() for record in records) == 1


@pytest.mark.parametrize("boundary", ["import", "logging"])
def test_metadata_diagnostic_failures_preserve_workload_outcome(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Diagnostic import and emission failures remain observational too."""
    from tensorrt_llm.logger import logger

    select_regions(monkeypatch, "workflow")
    original = LookupError("workload failed")

    def metadata() -> dict[str, profiler.MetadataValue]:
        raise TypeError("unexpected metadata keyword")

    def fail_logging(*args: object, **kwargs: object) -> None:
        raise RuntimeError("diagnostic failed")

    real_import = builtins.__import__

    def fail_logger_import(name: str, *args: object, **kwargs: object) -> object:
        """Fail logger imports while preserving all unrelated imports."""
        if name in {"logger", "tensorrt_llm.logger"}:
            raise ImportError("diagnostic import failed")
        return real_import(name, *args, **kwargs)

    if boundary == "logging":
        monkeypatch.setattr(logger, "warning_once", fail_logging)
    else:
        monkeypatch.setattr(builtins, "__import__", fail_logger_import)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with profiler.record_function_metadata_scope("diagnostic_result", "workflow", metadata):
            result = 42
        with pytest.raises(LookupError) as caught:
            with profiler.record_function_metadata_scope("diagnostic_error", "workflow", metadata):
                raise original
    assert result == 42
    assert caught.value is original


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("metadata_fails", [False, True])
def test_metadata_failures_preserve_body_and_original_exception(
    monkeypatch: pytest.MonkeyPatch, active: bool, metadata_fails: bool
) -> None:
    """Keep body results and exceptions independent of metadata failures."""
    select_regions(monkeypatch, *(("workflow",) if active else ()))
    original = RuntimeError("operation failed")

    def metadata() -> dict[str, profiler.MetadataValue]:
        """Return literal fields or inject the selected metadata failure."""
        if metadata_fails:
            raise LookupError("observational metadata failed")
        return {"rank": -7}

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with profiler.record_function_metadata_scope("result", "workflow", metadata):
            result = 42
        with pytest.raises(RuntimeError) as caught:
            with profiler.record_function_metadata_scope("error", "workflow", metadata):
                raise original
    assert result == 42
    assert caught.value is original


@pytest.mark.parametrize("boundary", ["setup", "cleanup"])
def test_profiler_failures_preserve_body_and_original_exception(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Inject upstream failures only at the observational entry or exit boundary."""
    select_regions(monkeypatch, "workflow")
    real_exit = torch.profiler.record_function.__exit__

    def fail_enter(scope: torch.profiler.record_function) -> None:
        raise RuntimeError("profiler entry failed")

    def fail_exit(
        scope: torch.profiler.record_function,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        real_exit(scope, error_type, error, traceback)
        raise RuntimeError("profiler cleanup failed")

    method = "__enter__" if boundary == "setup" else "__exit__"
    monkeypatch.setattr(
        torch.profiler.record_function, method, fail_enter if boundary == "setup" else fail_exit
    )
    original = LookupError("body failed")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with profiler.record_function_metadata_scope("result", "workflow", lambda: {}):
            result = 42
        with pytest.raises(LookupError) as caught:
            with profiler.record_function_metadata_scope("failure", "workflow", lambda: {}):
                raise original
    assert result == 42
    assert caught.value is original


@pytest.mark.parametrize("boundary", ["entry", "cleanup"])
def test_profiler_operation_failures_warn_once(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Warn once for repeated scope entry or cleanup failures."""
    from tensorrt_llm.logger import logger

    select_regions(monkeypatch, "workflow")
    monkeypatch.setattr(logger, "_appeared_keys", set())
    messages: list[str] = []
    monkeypatch.setattr(logger, "log", lambda level, *parts: messages.append(" ".join(parts)))
    method = "__enter__" if boundary == "entry" else "__exit__"
    original = getattr(torch.profiler.record_function, method)

    def fail(scope: object, *args: object) -> None:
        """Inject the selected scope failure after any required real cleanup."""
        if boundary == "cleanup":
            original(scope, *args)
        raise RuntimeError("injected observational failure")

    monkeypatch.setattr(torch.profiler.record_function, method, fail)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        for _ in range(2):
            with profiler.record_function_metadata_scope(
                "warning-contract", "workflow", lambda: {}
            ):
                result = 7
    assert result == 7
    assert len(messages) == 1
    assert boundary in messages[0] and "injected observational failure" in messages[0]
