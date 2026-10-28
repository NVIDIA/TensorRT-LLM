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
"""The perf-sanity client loops must keep every warmup pass out of what they measure.

Drives AggrTestCmds.run_cmd and DisaggTestCmds.run_cmd with fake servers and a fake
benchmark_serving that appends one perf_metrics record per request, so the warmup
pass, the time breakdown and the gen-log windows run as in a real lane.
"""

import importlib
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import List, Tuple

import pytest

pytestmark = pytest.mark.cpu_only

_TESTS_DIR = Path(__file__).resolve().parents[2]
_DEFS_DIR = _TESTS_DIR / "integration" / "defs"

# The time breakdown's ctx_queue span, per kind of request the fake client sends.
_WARMUP_PASS_QUEUE_S = 5.0
_MEASURED_QUEUE_S = 0.1


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _package(name: str, path: Path) -> types.ModuleType:
    package = _module(name)
    package.__path__ = [str(path)]
    return package


@pytest.fixture
def tps(monkeypatch):
    """test_perf_sanity.py, with the imports that need a CI test environment stubbed."""
    noop = lambda *args, **kwargs: None  # noqa: E731
    stubs = {
        "defs": _package("defs", _DEFS_DIR),
        "defs.perf": _package("defs.perf", _DEFS_DIR / "perf"),
        "defs.common": _module("defs.common", wait_for_reported_addr=noop),
        "defs.trt_test_alternative": _module(
            "defs.trt_test_alternative", print_info=print, print_warning=print
        ),
        "defs.conftest": _module(
            "defs.conftest",
            get_llm_root=lambda: str(_TESTS_DIR.parent),
            llm_models_root=lambda: "/models",
        ),
        "defs.perf.perf_regression_utils": _module(
            "defs.perf.perf_regression_utils",
            _percentile=lambda values, percentile: 0.0,
            get_job_info=dict,
            process_and_upload_test_results=noop,
        ),
        "test_common": _package("test_common", _TESTS_DIR / "test_common"),
        "test_common.error_utils": _module("test_common.error_utils", report_error=noop),
        "test_common.http_utils": _module(
            "test_common.http_utils", fail_if_proc_died=noop, wait_for_endpoint_ready=noop
        ),
        "test_common.perf_sanity_matching": _module(
            "test_common.perf_sanity_matching", get_test_case_match_keys=list
        ),
    }
    # Set aside any copy imported under the real packages, so the import below binds
    # to the stubs.
    for name in [n for n in sys.modules if n.startswith("defs.perf.")]:
        monkeypatch.delitem(sys.modules, name)
    loaded_before = set(sys.modules)
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    module = importlib.import_module("defs.perf.test_perf_sanity")
    # A fake server process, ready at once, that the runner can stop.
    monkeypatch.setattr(module.subprocess, "Popen", _FakeServer)
    monkeypatch.setattr(module, "wait_for_reported_addr", lambda *a, **k: ("localhost", 8000))
    monkeypatch.setattr(module, "wait_for_endpoint_ready", noop)
    monkeypatch.setattr(module, "collect_startup_observation", lambda *a, **k: {})
    monkeypatch.setattr(module, "write_startup_observations", noop)
    monkeypatch.setattr(module, "PERF_METRICS_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(module, "PERF_METRICS_SETTLE_TIMEOUT", 10.0)
    yield module
    for name in set(sys.modules) - loaded_before:
        if name.split(".")[0] in ("defs", "test_common"):
            sys.modules.pop(name, None)


class _FakeServer:
    def __init__(self, *args, **kwargs):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class _FakeBenchmarkServing:
    """Records each run and appends one perf_metrics record per request it sends.

    A warmup pass's requests queue for _WARMUP_PASS_QUEUE_S and measured ones for
    _MEASURED_QUEUE_S, so any pass record in the breakdown moves its mean.
    """

    def __init__(self, perf_metrics_dir: str):
        # No directory: the time breakdown is off and the server writes no records.
        self.jsonl = (
            os.path.join(perf_metrics_dir, "perf_metrics-server-node0-1-x.jsonl")
            if perf_metrics_dir
            else ""
        )
        self.runs: List[Tuple[str, int]] = []
        self._arrival = 0.0

    def __call__(self, cmd, env, log_path, timeout=0):
        num_requests = int(cmd[cmd.index("--num-prompts") + 1])
        is_warmup_pass = "warmup-pass" in os.path.basename(log_path)
        self.runs.append((os.path.basename(log_path), num_requests))
        if self.jsonl:
            queue_s = _WARMUP_PASS_QUEUE_S if is_warmup_pass else _MEASURED_QUEUE_S
            # Without --no-test-input, a single test request goes first.
            num_records = num_requests + (0 if "--no-test-input" in cmd else 1)
            with open(self.jsonl, "a") as handle:
                for _ in range(num_records):
                    self._arrival += 1.0
                    timing = {
                        "arrival_time": self._arrival,
                        "first_scheduled_time": self._arrival + queue_s,
                    }
                    handle.write(json.dumps({"perf_metrics": {"timing_metrics": timing}}) + "\n")
        return f"Successful requests: {num_requests}\nFailed requests: 0\n"


def _clients(tps, concurrencies, warmup: bool):
    return [
        tps.ClientConfig(
            {"name": f"con{c}", "concurrency": c, "iterations": 2, "isl": 1024, "osl": 1},
            "m",
            warmup=warmup,
        )
        for c in concurrencies
    ]


def _run_aggr(tps, tmp_path, monkeypatch, benchmark_mode: str, time_breakdown: bool):
    perf_metrics_dir = tmp_path / "perf_metrics" if time_breakdown else None
    if perf_metrics_dir:
        perf_metrics_dir.mkdir()
    client = _FakeBenchmarkServing(str(perf_metrics_dir) if perf_metrics_dir else "")
    monkeypatch.setattr(tps, "_run_benchmark_with_log", client)
    configs = _clients(tps, (4, 8), warmup=benchmark_mode == "ctx_only")
    cmds = tps.AggrTestCmds(
        server_cmds=[["trtllm-serve", "m"]],
        client_cmds={0: [config.to_cmd() for config in configs]},
        timeout=tps.DEFAULT_TIMEOUT,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
        client_configs={0: configs},
        model_name="m",
        server_configs=[],
        perf_metrics_output_dir=str(perf_metrics_dir) if perf_metrics_dir else "",
        benchmark_mode=benchmark_mode,
    )
    return cmds.run_cmd(0), client.runs


def test_time_breakdown_leaves_out_every_warmup_pass_with_two_clients(
    tps, tmp_path, monkeypatch
) -> None:
    outputs, runs = _run_aggr(tps, tmp_path, monkeypatch, "ctx_only", time_breakdown=True)

    # Both clients' rows carry the pooled breakdown: no warmup pass record in it.
    for output in outputs:
        ctx_queue_mean = re.search(r"Time Breakdown ctx_queue mean \(ms\): ([\d.]+)", output)
        assert ctx_queue_mean, output
        assert float(ctx_queue_mean.group(1)) == pytest.approx(_MEASURED_QUEUE_S * 1000)
    # That takes giving only the first client a pass: the breakdown leaves out only
    # the records written before it.
    assert [name for name, _ in runs] == [
        "trtllm-benchmark-warmup-pass.0.0.log",
        "trtllm-benchmark.0.0.log",
        "trtllm-benchmark.0.1.log",
    ]


def test_every_client_gets_a_warmup_pass_without_the_time_breakdown(
    tps, tmp_path, monkeypatch
) -> None:
    _, runs = _run_aggr(tps, tmp_path, monkeypatch, "", time_breakdown=False)

    assert runs == [
        ("trtllm-benchmark-warmup-pass.0.0.log", 4),
        ("trtllm-benchmark.0.0.log", 8),
        ("trtllm-benchmark-warmup-pass.0.1.log", 8),
        ("trtllm-benchmark.0.1.log", 16),
    ]


def test_gen_log_windows_leave_out_the_next_clients_warmup_pass(tps, tmp_path, monkeypatch) -> None:
    client = _FakeBenchmarkServing("")
    monkeypatch.setattr(tps, "_run_benchmark_with_log", client)
    # Each snapshot's value is 100 x the benchmark runs before it, so it names its place.
    monkeypatch.setattr(tps, "gen_worker_log_sizes", lambda *a: [100 * len(client.runs)])
    monkeypatch.setattr(tps, "expected_gen_only_ngen", lambda *a, **k: None)
    windows = []
    cls = tps.DisaggTestCmds
    monkeypatch.setattr(cls, "_get_disagg_server_hostname_and_port", lambda self, i: ("h", 1))
    monkeypatch.setattr(cls, "_collect_worker_startup_observations", lambda self, i: [])
    monkeypatch.setattr(cls, "_run_connection_warmup", lambda self, *a: 0)
    monkeypatch.setattr(
        cls,
        "_append_gen_worker_device_step_time",
        lambda self, pending, outputs: windows.extend(pending),
    )
    configs = _clients(tps, (4, 8), warmup=True)
    cmds = cls(
        server_cmds=[(["ctx"], ["gen"], ["disagg"])],
        client_cmds={0: [config.to_cmd() for config in configs]},
        timeout=tps.DEFAULT_TIMEOUT,
        hostname="bench-host",
        disagg_serving_type="BENCHMARK",
        num_ctx_servers=1,
        num_gen_servers=1,
        output_dir=str(tmp_path),
        test_output_dir=str(tmp_path),
        client_configs={0: configs},
        server_configs=[
            (None, None, types.SimpleNamespace(benchmark_mode="e2e", server_env_var=""))
        ],
    )
    cmds.run_cmd(0)

    assert [name for name, _ in client.runs] == [
        "trtllm-benchmark-warmup-pass.0.0.log",
        "trtllm-benchmark.0.0.log",
        "trtllm-benchmark-warmup-pass.0.1.log",
        "trtllm-benchmark.0.1.log",
    ]
    # Client 0's window opens after its pass and closes before client 1's pass, which
    # comes before client 1's window; the last window reads to the end of the log.
    assert [(w["start_offsets"], w["end_offsets"]) for w in windows] == [
        ([100], [200]),
        ([300], None),
    ]


def test_warmup_plan_matches_the_run_loops(tps) -> None:
    configs = _clients(tps, (4, 8), warmup=True)

    assert tps.warmup_pass_plan_for(configs, "e2e", time_breakdown=True) == [4, 0]
    assert tps.warmup_pass_plan_for(configs, "e2e", time_breakdown=False) == [4, 8]
    assert tps.warmup_pass_plan_for(configs, "gen_only", time_breakdown=False) == [0, 0]
