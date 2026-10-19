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
"""Tests for the perf-sanity AgentX client's prefill-only (ctx_only) path."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AGENTX_CLIENT_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "perf" / "agentx_client.py"

_BASE_ARGS = ["--model", "/m", "--concurrency", "8", "--dataset", "d", "--host", "h", "--port", "1"]


@pytest.fixture
def client() -> ModuleType:
    spec = importlib.util.spec_from_file_location("perf_agentx_client", AGENTX_CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export(client: ModuleType, with_decode_intervals: bool) -> dict:
    """A minimal aiperf export carrying every metric report_metrics reads."""
    export = {field: {"unit": unit, "avg": 2.0} for _, field, unit in client.THROUGHPUT_METRICS}
    for _, field in client.LATENCY_METRICS:
        if with_decode_intervals or field not in client.DECODE_INTERVAL_FIELDS:
            export[field] = {"unit": "ms", "avg": 5.0, "p50": 4.0, "p99": 9.0}
    return export


def _extra_inputs(cmd: list) -> list:
    """Every value passed to ``--extra-inputs`` in an aiperf command line."""
    return [cmd[i + 1] for i, arg in enumerate(cmd[:-1]) if arg == "--extra-inputs"]


def test_max_output_tokens_overrides_every_turn(client: ModuleType, monkeypatch):
    monkeypatch.delenv("AGENTX_EXTRA_ARGS", raising=False)
    args = client.parse_args(_BASE_ARGS + ["--max-output-tokens", "1"])
    cmd = client.build_aiperf_cmd(args, "http://h:1", "/tmp/a")
    # aiperf merges --extra-inputs over each turn's own max_completion_tokens.
    assert "max_completion_tokens:1" in _extra_inputs(cmd)


def test_trace_output_lengths_kept_by_default(client: ModuleType, monkeypatch):
    monkeypatch.delenv("AGENTX_EXTRA_ARGS", raising=False)
    cmd = client.build_aiperf_cmd(client.parse_args(_BASE_ARGS), "http://h:1", "/tmp/a")
    assert not any(v.startswith("max_completion_tokens:") for v in _extra_inputs(cmd))


def test_max_output_tokens_must_be_positive(client: ModuleType):
    with pytest.raises(SystemExit):
        client.parse_args(_BASE_ARGS + ["--max-output-tokens", "0"])


def test_prefill_only_reports_absent_decode_intervals_as_zero(client: ModuleType, capsys):
    client.report_metrics(_export(client, with_decode_intervals=False), prefill_only=True)
    lines = dict(
        line.rsplit(":", 1) for line in capsys.readouterr().out.splitlines() if line.strip()
    )
    for stat in ("Mean", "Median", "P99"):
        assert float(lines[f"{stat} TPOT (ms)"]) == 0.0
        assert float(lines[f"{stat} ITL (ms)"]) == 0.0
    # Prefill-side metrics are still measured.
    assert float(lines["Median TTFT (ms)"]) == 4.0
    assert float(lines["P99 E2EL (ms)"]) == 9.0


def test_prefill_only_still_reports_measured_decode_intervals(client: ModuleType, capsys):
    client.report_metrics(_export(client, with_decode_intervals=True), prefill_only=True)
    assert "Mean TPOT (ms):                          5.00" in capsys.readouterr().out


def test_missing_decode_intervals_fail_outside_prefill_only(client: ModuleType):
    with pytest.raises(RuntimeError, match="no metric 'inter_chunk_latency'"):
        client.report_metrics(_export(client, with_decode_intervals=False))


PERF_SANITY_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "perf" / "test_perf_sanity.py"


def _load_perf_sanity_names(*names: str) -> dict:
    """Execute only the named top-level definitions of test_perf_sanity.py.

    The module imports the TRT-LLM test fixtures (and through them torch) at
    load time, so it cannot be imported in this CPU-only test; the helpers
    under test only need the standard library.
    """
    import ast
    from typing import Dict

    tree = ast.parse(PERF_SANITY_PATH.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            continue
        if defined & set(names):
            nodes.append(node)
    namespace = {"Dict": Dict}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(PERF_SANITY_PATH), "exec"), namespace)
    return namespace


def test_ctx_only_client_env_shortens_agentx_warmup_and_window():
    ns = _load_perf_sanity_names(
        "to_env_dict", "AGENTX_CTX_ONLY_CLIENT_ENV", "agentx_ctx_only_client_env"
    )
    yaml_env = "AGENTX_MAX_CTX=996579 AGENTX_DURATION=3600 AGENTX_WARMUP_PER_LANE=3"
    env = ns["to_env_dict"](ns["agentx_ctx_only_client_env"](yaml_env))
    assert env == {
        "AGENTX_MAX_CTX": "996579",
        "AGENTX_DURATION": "1800",
        "AGENTX_WARMUP_PER_LANE": "0",
    }
    # The agentx scenario rejects a window below 900 s.
    assert int(env["AGENTX_DURATION"]) >= 900


def _empty_content(count: int) -> dict:
    return {
        "error_details": {"type": "InvalidInferenceResultError", "message": "no content"},
        "count": count,
    }


def test_small_share_of_empty_content_responses_is_tolerated(client: ModuleType):
    # The DSv4 ctx_only lane: 3 empty-content responses in 1752 + 3 requests.
    client.check_request_errors([_empty_content(3)], requests=1752)


def test_empty_content_responses_above_the_tolerance_fail(client: ModuleType):
    with pytest.raises(RuntimeError, match="exceed the 0.5% tolerance"):
        client.check_request_errors([_empty_content(20)], requests=1000)


def test_other_request_errors_always_fail(client: ModuleType):
    errors = [_empty_content(1), {"error_details": {"type": "HTTPError"}, "count": 1}]
    with pytest.raises(RuntimeError, match="aiperf reported request errors"):
        client.check_request_errors(errors, requests=100000)
