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
r"""Integration tests for prefix-aware scheduling with trtllm-serve.

Launches trtllm-serve with Qwen2-0.5B and runs the LMBenchmark
multi-round QA workload that originally triggered the over-admission bug:
  total_num_tokens (13985) should be less than or equal to max_num_tokens (8192)

The workload sends concurrent requests sharing a system prompt prefix
at various QPS levels, exercising the prefix-aware scheduler's reuse
estimation logic under realistic serving conditions.
"""

import csv
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import pytest
import requests as req_lib
import yaml
from requests.exceptions import RequestException

from ..common import get_free_port_in_ci
from ..conftest import llm_models_root
from ..local_venv import PythonVenvRunnerImpl
from ..trt_test_alternative import popen, print_error, print_info

MODEL_PATH = f"{llm_models_root()}/Qwen2-0.5B"
MODEL_NAME = "Qwen2-0.5B"

# c1e05a70 "add timeout, skip-ssl-verify, gap-between-requests, itl/throughput
# metrics to real multi-round qa (#36)" by Ziwen Ning, 2026-01-28.
LMBENCHMARK_SHA = "c1e05a708a5a1fd04a9ec09d215edbdcccdf92cb"
LMBENCHMARK_SCRIPT = ""
_LMBENCHMARK_PACKAGE = "benchmark"
_LMBENCHMARK_SCRIPT_IN_PACKAGE = "synthetic-multi-round-qa/multi-round-qa.py"
_LMBENCHMARK_INSTALL_SPEC = (
    f"{_LMBENCHMARK_PACKAGE} @ git+https://github.com/LMCache/LMBenchmark.git@{LMBENCHMARK_SHA}"
)
CsvMetrics: TypeAlias = dict[str, int | float]

# ---------------------------------------------------------------------------
# Scheduler / KV-cache config combinations
# ---------------------------------------------------------------------------
#
# Each entry is a plain dict that will be serialised to YAML and passed via
# `trtllm-serve serve --config`.  All combinations test a distinct scheduling
# code path so that functional regressions in any path are caught.
#
# Dimensions covered:
#   capacity_scheduler_policy : GUARANTEED_NO_EVICT (default), MAX_UTILIZATION
#   enable_chunked_prefill    : true, false
#   use_python_scheduler      : false (default), true
#   max_attention_window      : unset (full KV), [2048] (SWA)
#   disable_overlap_scheduler : false (default), true

SCHED_CONFIGS = [
    # ── Baseline: GUARANTEED_NO_EVICT + chunked prefill ─────────────────────
    pytest.param(
        {
            "kv_cache_config": {"max_tokens": 200000},
            "enable_chunked_prefill": True,
            "print_iter_log": True,
        },
        id="guaranteed-chunked",
    ),
    # ── Aggressive eviction policy ───────────────────────────────────────────
    pytest.param(
        {
            "kv_cache_config": {"max_tokens": 200000},
            "enable_chunked_prefill": True,
            "scheduler_config": {"capacity_scheduler_policy": "MAX_UTILIZATION"},
            "print_iter_log": True,
        },
        id="max-util-chunked",
    ),
    # ── No chunked prefill (full-context scheduling) ─────────────────────────
    pytest.param(
        {
            "kv_cache_config": {"max_tokens": 200000},
            "enable_chunked_prefill": False,
            "print_iter_log": True,
        },
        id="guaranteed-no-chunked",
    ),
    # ── Pure-Python scheduler (parity with C++ path) ─────────────────────────
    pytest.param(
        {
            "kv_cache_config": {"max_tokens": 200000},
            "enable_chunked_prefill": True,
            "scheduler_config": {"use_python_scheduler": True},
            "print_iter_log": True,
        },
        id="python-scheduler",
    ),
    # ── Sliding-window attention (SWA, 2048-token window) ────────────────────
    # Qwen2-0.5B has 24 layers; max_attention_window=[2048] broadcasts to all.
    # The 1000-token system prompt fits within the window, but older history
    # rounds are evicted, exercising the SWA eviction path in the scheduler.
    pytest.param(
        {
            "kv_cache_config": {
                "max_tokens": 200000,
                "max_attention_window": [2048],
            },
            "enable_chunked_prefill": True,
            "print_iter_log": True,
        },
        id="swa-chunked",
    ),
    # ── Overlap scheduler disabled ────────────────────────────────────────────
    pytest.param(
        {
            "kv_cache_config": {"max_tokens": 200000},
            "enable_chunked_prefill": True,
            "disable_overlap_scheduler": True,
            "print_iter_log": True,
        },
        id="no-overlap-chunked",
    ),
    # ── KV-cache host offload (secondary memory tier) ────────────────────────
    # Exercises the offload path: on eviction from device, blocks move to
    # host memory instead of being dropped, and prefix-aware scheduling
    # must account for the larger effective reuse pool. host_cache_size is
    # deliberately small (1 GiB) relative to chat-history volume so that
    # offload + promotion both happen during the QPS sweep.
    pytest.param(
        {
            "kv_cache_config": {
                "max_tokens": 200000,
                "host_cache_size": 1024 * 1024 * 1024,
                "free_gpu_memory_fraction": 0.2,
            },
            "enable_chunked_prefill": True,
            "print_iter_log": True,
        },
        id="offload-chunked",
    ),
    pytest.param(
        {
            "kv_cache_config": {
                "max_tokens": 200000,
                "host_cache_size": 1024 * 1024 * 1024,
                "free_gpu_memory_fraction": 0.2,
            },
            # enable_chunked_prefill defaults to False here. The multi-round-QA
            # workload sends prompts up to ~21k tokens (system_prompt=1000 +
            # chat_history=20000); without chunking they must fit in a single
            # forward pass, so max_num_tokens must exceed the worst-case prompt
            # length. Default is 8192, which rejects the request at admission
            # with "sum of prompt length (21044) should not exceed
            # max_num_tokens (8192)". Size to 32768 to accept the workload.
            "max_num_tokens": 32768,
            "print_iter_log": True,
        },
        id="offload-no-chunked",
    ),
]

# ---------------------------------------------------------------------------
# Error / NaN detection helpers
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = [
    "should be less than or equal to max_num_tokens",  # original over-admission bug
    "CUDA error",
    "Traceback (most recent call last)",  # uncaught Python exception
    "AssertionError",
]

_NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)


def _check_server_errors(server_log: str) -> str | None:
    """Return the first error line found in *server_log*, or None if clean."""
    try:
        with open(server_log) as f:
            for line in f:
                if any(p in line for p in _ERROR_PATTERNS):
                    return line.strip()
    except OSError:
        pass
    return None


def _tail_log(log_path: str, n: int = 40) -> str:
    """Return the last *n* lines of *log_path* as a single string."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError:
        return "(log not available)"


def _assert_no_server_errors(server_log: str) -> None:
    """Assert the server log is error-free; include log tail on failure."""
    err = _check_server_errors(server_log)
    assert err is None, (
        f"Server error detected: {err!r}\n"
        f"--- last lines of {server_log} ---\n"
        f"{_tail_log(server_log)}"
    )


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------


def _wait_for_server_ready(
    proc: subprocess.Popen,
    port: int,
    timeout: int = 300,
    interval: int = 2,
    server_log: str | None = None,
) -> None:
    """Wait for trtllm-serve /health to return 200.

    If *server_log* is provided, the log is scanned for error patterns on each
    poll iteration so that crashes during startup are caught immediately.
    """
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if req_lib.get(url, timeout=interval).status_code == 200:
                print_info(f"Server ready in {time.time() - start:.1f}s on port {port}")
                return
        except RequestException:
            pass
        rc = proc.poll()
        if rc is not None and rc != 0:
            tail = _tail_log(server_log) if server_log else ""
            raise RuntimeError(f"trtllm-serve exited unexpectedly with code {rc}.\n{tail}")
        if server_log:
            err = _check_server_errors(server_log)
            if err:
                raise RuntimeError(
                    f"trtllm-serve reported error during startup: {err!r}\n{_tail_log(server_log)}"
                )
        time.sleep(interval)
    raise TimeoutError(f"trtllm-serve not ready within {timeout}s")


def _make_server_cmd(port: int, config_path: str) -> list[str]:
    """Build the trtllm-serve command used by all tests."""
    return [
        "trtllm-serve",
        "serve",
        MODEL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--config",
        config_path,
    ]


def _write_config(tmp_path: Path, cfg: dict[str, object], name: str = "config.yml") -> str:
    """Serialise *cfg* to YAML in *tmp_path* and return the file path."""
    path = str(tmp_path / name)
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _run_lmbenchmark(
    llm_venv: PythonVenvRunnerImpl,
    port: int,
    output_csv: str,
    qps: float = 0.5,
    num_users: int = 4,
    num_rounds: int = 3,
    system_prompt: int = 500,
    chat_history: int = 2000,
    answer_len: int = 20,
    duration: int = 30,
    server_log: str | None = None,
) -> int:
    """Run the LMBenchmark multi-round-qa script and return the exit code.

    stdout and stderr are drained in background threads so that the main
    watchdog loop can run deadline / server-log / /health checks on a fixed
    cadence regardless of how chatty the benchmark is.  A blocking readline
    in the main loop would otherwise stall indefinitely when the benchmark
    goes quiet (which is exactly what happens when the server stalls),
    defeating the point of the watchdog.
    """
    assert LMBENCHMARK_SCRIPT, "ensure_lmbenchmark fixture must resolve LMBenchmark first"
    script_dir = os.path.dirname(LMBENCHMARK_SCRIPT)
    cmd = [
        LMBENCHMARK_SCRIPT,
        "--num-users",
        str(num_users),
        "--num-rounds",
        str(num_rounds),
        "--qps",
        str(qps),
        "--shared-system-prompt",
        str(system_prompt),
        "--user-history-prompt",
        str(chat_history),
        "--answer-len",
        str(answer_len),
        "--model",
        MODEL_NAME,
        "--base-url",
        f"http://localhost:{port}",
        "--init-user-id",
        "1",
        "--output",
        output_csv,
        "--log-interval",
        "30",
        "--time",
        str(duration),
    ]
    print_info(f"Running LMBenchmark: {' '.join(cmd)}")

    t_start = time.time()

    def _popen_lmbenchmark(call_args: list[str], env: Mapping[str, str]) -> subprocess.Popen:
        return subprocess.Popen(
            call_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_dir,
            env=env,
        )

    proc = llm_venv.run_cmd(cmd, caller=_popen_lmbenchmark)

    stdout_q: queue.Queue[str | None] = queue.Queue()
    stderr_lines: list[str] = []

    def _drain_stdout() -> None:
        try:
            for line in iter(proc.stdout.readline, ""):
                stdout_q.put(line)
        finally:
            stdout_q.put(None)  # EOF sentinel

    def _drain_stderr() -> None:
        for line in iter(proc.stderr.readline, ""):
            stderr_lines.append(line)

    stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    # Grace window above the scripted duration. LMBenchmark keeps requests
    # in flight after `--time` expires, and at higher QPS the backlog the
    # server still has to drain scales roughly linearly with offered load.
    # Observed worst-case drain on H100 PCIe + Qwen2-0.5B (5 scheduler
    # variants, 3 stages each): drain ≈ 1.7·qps + 1 s. The coefficient below
    # gives ≥2.8× margin over the observed worst case at every tested qps
    # while still catching genuine stalls within a few minutes.
    grace_s = max(45, int(qps * 5))
    deadline = time.time() + duration + grace_s
    last_health_ok = time.time()
    last_poll = time.time()

    def _kill(reason: str) -> int:
        elapsed_at_kill = time.time() - t_start
        print_error(
            f"{reason} [timing] qps={qps} duration={duration}s "
            f"elapsed={elapsed_at_kill:.1f}s drain={elapsed_at_kill - duration:+.1f}s "
            f"grace_s={grace_s}s"
        )
        proc.kill()
        proc.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        return -1

    def _cleanup_if_running() -> None:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    def _drain_queued_stdout() -> int | None:
        """Pop all queued stdout lines; return -1 if NaN seen, else None."""
        while True:
            try:
                line = stdout_q.get_nowait()
            except queue.Empty:
                return None
            if line is None:
                return None  # EOF sentinel; stdout thread done
            line = line.rstrip()
            if line and _NAN_RE.search(line):
                return _kill(f"NaN detected in benchmark output: {line!r}")

    completed = False
    try:
        while proc.poll() is None:
            rc = _drain_queued_stdout()
            if rc is not None:
                return rc

            now = time.time()
            if now > deadline:
                return _kill(
                    f"LMBenchmark exceeded deadline ({duration + grace_s}s, "
                    f"duration={duration}s + grace={grace_s}s at qps={qps}); "
                    "server may have stalled"
                )

            # Periodic checks (every 5 s).  Do not gate these on stdout
            # activity -- the whole point of the watchdog is to fire even
            # when the benchmark is quiet.
            if now - last_poll >= 5:
                last_poll = now

                if server_log and _check_server_errors(server_log):
                    return _kill("Server error detected in log, killing benchmark")

                try:
                    # 5s (was 2s) — under high offered load the async /health
                    # endpoint can be briefly starved by the busy event loop.
                    resp = req_lib.get(f"http://localhost:{port}/health", timeout=5)
                    if resp.status_code == 200:
                        last_health_ok = now
                except RequestException:
                    pass
                # 60s (was 30s) — extreme QPS stages need more slack before
                # declaring the server stalled; genuine hangs still get caught.
                if now - last_health_ok > 60:
                    return _kill("Server stopped responding to /health (stalled)")

            time.sleep(0.2)
        completed = True
    finally:
        if not completed:
            _cleanup_if_running()

    # Benchmark has exited.  Wait for drain threads to finish so we see
    # every remaining line, then surface any leftover output.
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    while True:
        try:
            line = stdout_q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            continue
        line = line.rstrip()
        if not line:
            continue
        if _NAN_RE.search(line):
            print_error(f"NaN detected in benchmark output: {line!r}")
            return -1
        print_info(f"  [benchmark] {line}")

    elapsed = time.time() - t_start
    drain = elapsed - duration
    print_info(
        f"[timing] qps={qps} duration={duration}s elapsed={elapsed:.1f}s "
        f"drain={drain:+.1f}s grace_s={grace_s}s "
        f"grace_used_frac={(drain / grace_s if grace_s else float('nan')):.2f}"
    )

    if proc.returncode != 0:
        print_error(f"LMBenchmark exited with code {proc.returncode}")
        for line in "".join(stderr_lines).strip().split("\n")[-10:]:
            if line:
                print_error(f"  [stderr] {line}")
    return proc.returncode


def _parse_csv_metrics(csv_path: str) -> CsvMetrics | None:
    """Parse LMBenchmark output CSV and return TTFT metrics.

    Returns a dict with keys:
        num_requests  – total rows
        nan_count     – rows with NaN/Inf ttft (should be 0)
        ttft_count    – rows with a usable (parseable, finite) ttft value
        ttft_avg, ttft_p50, ttft_p99  – latency percentiles (when available)
    """
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    nan_count = 0
    ttfts: list[float] = []
    for r in rows:
        raw = r.get("ttft")
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            nan_count += 1
            continue
        if math.isnan(v) or math.isinf(v):
            nan_count += 1
        else:
            ttfts.append(v)

    metrics: CsvMetrics = {
        "num_requests": len(rows),
        "nan_count": nan_count,
        "ttft_count": len(ttfts),
    }
    if not ttfts:
        return metrics

    ttfts.sort()
    metrics.update(
        {
            "ttft_avg": sum(ttfts) / len(ttfts),
            "ttft_p50": ttfts[len(ttfts) // 2],
            "ttft_p99": ttfts[int(len(ttfts) * 0.99)],
        }
    )
    return metrics


def _run_and_assert_stage(
    label: str,
    llm_venv: PythonVenvRunnerImpl,
    port: int,
    output_csv: str,
    server_log: str,
    *,
    qps: float,
    num_users: int,
    num_rounds: int,
    system_prompt: int,
    chat_history: int,
    answer_len: int,
    duration: int,
) -> CsvMetrics:
    """Run one LMBenchmark stage, assert success, and return the metrics dict."""
    rc = _run_lmbenchmark(
        llm_venv=llm_venv,
        port=port,
        output_csv=output_csv,
        qps=qps,
        num_users=num_users,
        num_rounds=num_rounds,
        system_prompt=system_prompt,
        chat_history=chat_history,
        answer_len=answer_len,
        duration=duration,
        server_log=server_log,
    )
    assert rc == 0, f"{label} failed with rc={rc}. See {server_log}\n{_tail_log(server_log)}"
    metrics = _parse_csv_metrics(output_csv)
    assert metrics and metrics.get("num_requests", 0) > 0, (
        f"{label} produced no completed requests (csv={output_csv})"
    )
    assert metrics["nan_count"] == 0, f"{label} produced {metrics['nan_count']} NaN ttft values"
    assert metrics["ttft_count"] > 0, (
        f"{label} produced {metrics['num_requests']} rows but zero usable ttft "
        f"values — benchmark output is broken (csv={output_csv})"
    )
    return metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _validate_lmbenchmark_direct_url(direct_url: str | None) -> None:
    """Verify the installed LMBenchmark package records the pinned Git revision."""
    if direct_url is None:
        pytest.fail("LMBenchmark package is missing direct_url.json provenance.")
    try:
        direct_url_data = json.loads(direct_url)
    except json.JSONDecodeError as exc:
        pytest.fail(f"LMBenchmark direct_url.json is not valid JSON: {exc}")

    vcs_info = direct_url_data.get("vcs_info", {})
    recorded_revisions = (vcs_info.get("commit_id"), vcs_info.get("requested_revision"))
    if LMBENCHMARK_SHA not in recorded_revisions:
        pytest.fail(
            f"LMBenchmark package revision {recorded_revisions} does not match "
            f"expected {LMBENCHMARK_SHA}."
        )


def _resolve_lmbenchmark_install(llm_venv: PythonVenvRunnerImpl) -> str:
    """Return the benchmark script from the fixture Python environment."""
    query = f"""
import importlib.metadata
import json
import os

package = {_LMBENCHMARK_PACKAGE!r}
script_in_package = {_LMBENCHMARK_SCRIPT_IN_PACKAGE!r}

try:
    dist = importlib.metadata.distribution(package)
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(
        "LMBenchmark package is not installed after fixture setup. "
        f"Expected {{script_in_package}}."
    )

print(
    json.dumps(
        {{
            "direct_url": dist.read_text("direct_url.json"),
            "script": os.fspath(dist.locate_file(script_in_package)),
        }}
    )
)
"""
    try:
        payload = json.loads(llm_venv.run_output(query))
    except (RuntimeError, json.JSONDecodeError) as exc:
        pytest.fail(f"Failed to resolve installed LMBenchmark package: {exc}")

    _validate_lmbenchmark_direct_url(payload.get("direct_url"))
    script = payload["script"]
    if not os.path.exists(script):
        pytest.fail(
            f"LMBenchmark package installed, but {_LMBENCHMARK_SCRIPT_IN_PACKAGE} was not found."
        )
    return script


@pytest.fixture(scope="module")
def ensure_lmbenchmark(llm_venv: PythonVenvRunnerImpl) -> PythonVenvRunnerImpl:
    """Install and resolve the pinned LMBenchmark script used by this module."""
    global LMBENCHMARK_SCRIPT

    llm_venv.run_cmd(["-m", "pip", "install", _LMBENCHMARK_INSTALL_SPEC], timeout=600)
    LMBENCHMARK_SCRIPT = _resolve_lmbenchmark_install(llm_venv)
    return llm_venv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServePrefixAwareScheduling:
    """E2E: trtllm-serve with shared prefixes.

    Tests prefix-aware scheduling under shared-prefix workloads that
    originally triggered the total_num_tokens > max_num_tokens over-admission
    bug. Includes both self-contained OpenAI-client tests and LMBenchmark-
    based regression tests.

    Config variants (SCHED_CONFIGS) exercise:
      * capacity_scheduler_policy: GUARANTEED_NO_EVICT, MAX_UTILIZATION
      * enable_chunked_prefill: true / false
      * use_python_scheduler: false / true
      * max_attention_window (SWA): unset / 2048 tokens
      * disable_overlap_scheduler: false / true
    """

    def test_multi_round_qa_shared_prefix_smoke(
        self, tmp_path: Path, ensure_lmbenchmark: PythonVenvRunnerImpl
    ) -> None:
        """Pre-merge smoke: two-stage LMBenchmark run to catch the original bug.

        Uses the baseline scheduler config (GUARANTEED_NO_EVICT + chunked
        prefill).  Stage 1 seeds the radix tree; Stage 2 applies load at
        QPS=32 — the level that originally triggered the over-admission crash.
        The two stages share a single server so that accumulated radix-tree
        state from Stage 1 is present when Stage 2 runs, matching the
        conditions under which the original bug manifested.

        Estimated runtime: ~2.5 min (server start + 20s warmup + 45s main).
        """
        baseline_cfg = {
            "kv_cache_config": {"max_tokens": 200000},
            "enable_chunked_prefill": True,
            "print_iter_log": True,
        }
        config_path = _write_config(tmp_path, baseline_cfg)
        port = get_free_port_in_ci()
        cmd = _make_server_cmd(port, config_path)
        server_log = str(tmp_path / "server.log")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        with open(server_log, "w") as log_f:
            with popen(cmd, stderr=log_f, stdout=log_f, env=env) as proc:
                _wait_for_server_ready(proc, port, server_log=server_log)

                # Stage 1: seed radix tree (mimics the low-QPS warmup that
                # accumulated state before the crash in the original report).
                print_info("Smoke stage 1: seeding radix tree...")
                _run_and_assert_stage(
                    "Smoke warmup",
                    ensure_lmbenchmark,
                    port,
                    str(tmp_path / "smoke_warmup.csv"),
                    server_log,
                    qps=4,
                    num_users=3,
                    num_rounds=3,
                    system_prompt=1000,
                    chat_history=20000,
                    answer_len=10,
                    duration=20,
                )

                # Stage 2: main load at QPS=32 (the originally failing level).
                print_info("Smoke stage 2: main load at QPS=32...")
                _run_and_assert_stage(
                    "Smoke main stage",
                    ensure_lmbenchmark,
                    port,
                    str(tmp_path / "smoke_main.csv"),
                    server_log,
                    qps=32,
                    num_users=8,
                    num_rounds=5,
                    system_prompt=1000,
                    chat_history=20000,
                    answer_len=10,
                    duration=45,
                )

                assert proc.poll() is None, (
                    f"Server exited unexpectedly. See {server_log}\n{_tail_log(server_log)}"
                )

        _assert_no_server_errors(server_log)

    @pytest.mark.parametrize(
        "sched_cfg",
        [
            c
            for c in SCHED_CONFIGS
            if c.id
            in (
                "guaranteed-chunked",
                "guaranteed-no-chunked",
                "max-util-chunked",
                "python-scheduler",
                "swa-chunked",
                "no-overlap-chunked",
                "offload-chunked",
                "offload-no-chunked",
            )
        ],
    )
    def test_multi_round_qa_shared_prefix(
        self,
        tmp_path: Path,
        sched_cfg: dict[str, object],
        ensure_lmbenchmark: PythonVenvRunnerImpl,
    ) -> None:
        """Full QPS-sweep regression: shared prefix at escalating load.

        Launches trtllm-serve with block reuse enabled, then runs the full
        QPS escalation sequence (8 -> 32) against a single
        server process. The original over-admission bug manifests after
        accumulated radix-tree state from earlier sweeps, so restarting
        per QPS would mask the failure.

        Each stage is preceded by a warmup that seeds the radix tree.
        After every stage the test asserts:
          * the benchmark process exited with rc=0
          * the output CSV exists and contains completed request rows
          * the CSV contains no NaN ttft values
          * the server log is free of error patterns

        NOTE: This test is intended for post-merge / nightly CI due to its
        ~10 min runtime per config.
        """
        config_path = _write_config(tmp_path, sched_cfg)
        port = get_free_port_in_ci()
        cmd = _make_server_cmd(port, config_path)
        server_log = str(tmp_path / "server.log")
        # Parameters match the long_input_short_output workload that
        # originally surfaced the bug: 1000-token shared system prompt,
        # 20000-token per-user chat history, 100-token answers.
        qps_values = [8, 32]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        with open(server_log, "w") as log_f:
            with popen(cmd, stderr=log_f, stdout=log_f, env=env) as proc:
                _wait_for_server_ready(proc, port, server_log=server_log)

                # Warmup: seed the radix tree with the shared system prompt.
                print_info("Warmup to seed the radix tree with the shared system prompt.")
                _run_and_assert_stage(
                    "Warmup to seed the radix tree with the shared system prompt.",
                    ensure_lmbenchmark,
                    port,
                    str(tmp_path / "warmup_1u_qps2.csv"),
                    server_log,
                    qps=2,
                    num_users=1,
                    num_rounds=2,
                    system_prompt=1000,
                    chat_history=20000,
                    answer_len=100,
                    duration=10,
                )

                for qps in qps_values:
                    print_info(f"Benchmark: 15 users, QPS={qps}...")
                    metrics = _run_and_assert_stage(
                        f"Benchmark at QPS={qps}",
                        ensure_lmbenchmark,
                        port,
                        str(tmp_path / f"benchmark_15u_qps{qps}.csv"),
                        server_log,
                        qps=qps,
                        num_users=15,
                        num_rounds=3,
                        system_prompt=1000,
                        chat_history=20000,
                        answer_len=100,
                        duration=30,
                    )
                    print_info(
                        f"QPS={qps}: {metrics['num_requests']} requests, "
                        f"ttft_p50={metrics.get('ttft_p50', float('nan')):.3f}s, "
                        f"ttft_p99={metrics.get('ttft_p99', float('nan')):.3f}s"
                    )

                    _assert_no_server_errors(server_log)
                    assert proc.poll() is None, (
                        f"Server exited unexpectedly during QPS={qps}. "
                        f"See {server_log}\n{_tail_log(server_log)}"
                    )

        # Final server-log sanity check across the full sweep.
        _assert_no_server_errors(server_log)
