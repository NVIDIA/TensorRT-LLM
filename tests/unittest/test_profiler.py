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
from profiler_test_utils import decode_profiler_metadata

from tensorrt_llm import _profiler as profiler

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (None, frozenset({"workflow", "mpi", "nccl", "nanobind"})),
        ("", frozenset({"workflow", "mpi", "nccl", "nanobind"})),
        (" \t\n", frozenset({"workflow", "mpi", "nccl", "nanobind"})),
        ("workflow,moe", frozenset({"workflow", "moe"})),
        (" workflow , kv ", frozenset({"workflow", "kv"})),
        ("none", frozenset()),
    ],
)
def test_selector_controls_regions(selection: str | None, expected: frozenset[str]) -> None:
    """Removing defaults, opt-ins, or trimming breaks region selection."""
    assert profiler._parse_custom_regions(selection) == expected


@pytest.mark.parametrize(
    "selection",
    ["workflow,", ",workflow", "workflow,,mpi", "unknown", "mpi,mpi", "none,kv"],
)
def test_selector_rejects_malformed_configuration(selection: str) -> None:
    with pytest.raises(ValueError, match="TLLM_PROFILE_CUSTOM_REGIONS"):
        profiler._parse_custom_regions(selection)


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
        "nested": {"values": [None, True, 1.5, ("x", -2)]},
    }
    assert decode_profiler_metadata(profiler._encode_metadata("workflow", metadata)) == (
        "workflow",
        metadata,
    )


@pytest.mark.parametrize("value", [object(), {"x"}, float("inf"), float("nan")])
def test_encoder_rejects_values_outside_literal_contract(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        profiler._encode_metadata("workflow", {"value": cast(profiler.MetadataValue, value)})


def test_encoder_rejects_builtin_subclasses_with_custom_repr() -> None:
    class CustomInt(int):
        def __repr__(self) -> str:
            return "not a literal"

    with pytest.raises(TypeError):
        profiler._encode_metadata("workflow", {"value": CustomInt(1)})


@pytest.mark.parametrize("payload", ["bad", "TRTLLM1-1:x{}", "TRTLLM18:work{}", "TRTLLM12:kv[]"])
def test_decoder_rejects_invalid_envelopes(payload: str) -> None:
    with pytest.raises(ValueError):
        decode_profiler_metadata(payload)


@pytest.mark.parametrize("enabled", [False, True])
def test_inactive_scopes_do_not_evaluate_metadata(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    monkeypatch.setattr(
        profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}) if enabled else frozenset()
    )
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
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset())
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
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
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
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    trace_file = tmp_path / "execution.json"
    observer = torch.profiler.ExecutionTraceObserver().register_callback(str(trace_file))
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
        ):
            with profiler.record_function_metadata_scope(
                "payload", "workflow", lambda: {"rank": -7, "ids": []}
            ):
                torch.ones(1)
    finally:
        observer.unregister_callback()
    node = next(
        node for node in json.loads(trace_file.read_text())["nodes"] if node["name"] == "payload"
    )
    assert decode_profiler_metadata(node["inputs"]["values"][0]) == (
        "workflow",
        {"rank": -7, "ids": []},
    )


def test_contextmanager_decorator_preserves_arguments_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))

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

    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
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

    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    original = LookupError("workload failed")

    def metadata() -> dict[str, profiler.MetadataValue]:
        raise TypeError("unexpected metadata keyword")

    def fail_logging(*args: object, **kwargs: object) -> None:
        raise RuntimeError("diagnostic failed")

    real_import = builtins.__import__

    def fail_logger_import(name: str, *args: object, **kwargs: object) -> object:
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
    monkeypatch.setattr(
        profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}) if active else frozenset()
    )
    original = RuntimeError("operation failed")

    def metadata() -> dict[str, profiler.MetadataValue]:
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
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
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
    from tensorrt_llm.logger import logger

    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    monkeypatch.setattr(logger, "_appeared_keys", set())
    messages: list[str] = []
    monkeypatch.setattr(logger, "log", lambda level, *parts: messages.append(" ".join(parts)))
    method = "__enter__" if boundary == "entry" else "__exit__"
    original = getattr(torch.profiler.record_function, method)

    def fail(scope: object, *args: object) -> None:
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
