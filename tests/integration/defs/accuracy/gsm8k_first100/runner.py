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
"""One explicitly triggered first-100 run: verify, serve, ask, record and judge.

Nothing is read, staged or launched before the explicit trigger is checked; without it the
run skips. Once triggered, every missing prerequisite (hardware, timeout, inputs, recipe,
criterion in regression mode) fails the run instead of skipping it.

A run verifies the pinned dataset items and the staged inputs, materializes the recipe,
starts ``trtllm-serve`` on four GPUs, asks the not yet completed questions through
``/v1/chat/completions`` with at most ``max_batch_size`` requests in flight, writes each reply
to ``results.jsonl`` as it arrives, stops the server and writes ``summary.json``. The results
directory is ``<--output-dir>/nvfp4_serving_gsm8k_first100`` (a fresh temporary directory
without ``--output-dir``); running again with the same directory and run identity resumes.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Mapping, NoReturn, Optional

import pytest
import requests

from ...common import wait_for_reported_addr
from ...trt_test_alternative import popen
from . import deployment, protocol, records

RESULTS_DIRNAME = "nvfp4_serving_gsm8k_first100"
#: Seconds of the pytest timeout kept for teardown and the summary.
TEARDOWN_RESERVE_S = 300
#: Grace for the stopped server before its process tree is killed.
STOP_GRACE_S = 120
#: Upper bound on server startup (model load) inside the run's budget.
STARTUP_LIMIT_S = 3600
POLL_S = 5.0
CONNECT_TIMEOUT_S = 10.0
SKIP_REASON = (
    f"explicit trigger only: set {deployment.GUARD_ENV}=1 to run the first-100 GSM8K "
    "serving qualification"
)


class ServerError(RuntimeError):
    """The server did not come up or stopped serving."""


@contextlib.contextmanager
def launch_server(
    argv: list[str], env: Mapping[str, str], log_path: Path
) -> Iterator[subprocess.Popen]:
    """Start the server in its own session; its whole process tree is cleaned up on exit."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        with popen(argv, env=dict(env), stdout=log, stderr=subprocess.STDOUT) as process:
            yield process


@dataclasses.dataclass
class RunDeps:
    """Process, HTTP, GPU and clock entry points of a run (fakes replace them in unit tests)."""

    launch: Callable[..., ContextManager[Any]] = launch_server
    wait_for_addr: Callable[..., tuple[str, int]] = wait_for_reported_addr
    get: Callable[..., Any] = requests.get
    post: Callable[..., Any] = requests.post
    query_gpus: Callable[[Mapping[str, str]], list[dict[str, str]]] = deployment.query_gpus
    memory_poller: Callable[..., ContextManager[Any]] = deployment.gpu_memory_poller
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


def _fail(message: str) -> NoReturn:
    pytest.fail(f"first-100 GSM8K qualification: {message}", pytrace=False)


def _enabled(environ: Mapping[str, str]) -> bool:
    try:
        return deployment.explicit_trigger(environ)
    except ValueError as error:
        _fail(str(error))


def require_hardware_when_enabled(
    environ: Mapping[str, str], deps: Optional[RunDeps] = None
) -> None:
    """Fail, never skip, an explicitly triggered run without four GB300 (SM103) GPUs."""
    if not _enabled(environ):
        return
    deps = deps or RunDeps()
    try:
        gpus = deps.query_gpus(environ)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        _fail(f"cannot identify the GPUs: {error}")
    problems = deployment.check_hardware(gpus)
    if problems:
        _fail("; ".join(problems))


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _results_dir(output_dir: Optional[str]) -> Path:
    if not output_dir:
        return Path(tempfile.mkdtemp(prefix=f"{RESULTS_DIRNAME}-"))
    path = Path(output_dir) / RESULTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_revision(repo: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _content_identity(description: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "files": {name: entry["sha256"] for name, entry in description["files"].items()},
        "weights_inventory_sha256": description["weights"]["inventory_sha256"],
        "num_hidden_layers": description["num_hidden_layers"],
    }


def _wait_ready(base: str, server: Any, limit: float, deps: RunDeps) -> str:
    """Poll ``/health`` until the server answers; return the one served model id."""
    while True:
        code = server.poll()
        if code is not None:
            raise ServerError(f"trtllm-serve exited with code {code} before it was ready")
        if deps.monotonic() >= limit:
            raise ServerError("trtllm-serve was not ready within its startup budget")
        try:
            if deps.get(f"{base}/health", timeout=CONNECT_TIMEOUT_S).status_code == 200:
                models = deps.get(f"{base}/v1/models", timeout=CONNECT_TIMEOUT_S)
                ids = [entry["id"] for entry in models.json()["data"]]
                if len(ids) != 1:
                    raise ServerError(f"/v1/models lists {ids}; expected one served model")
                return ids[0]
        except requests.exceptions.RequestException:
            pass
        except (ValueError, KeyError, TypeError) as error:
            raise ServerError(f"malformed /v1/models reply: {error}") from error
        deps.sleep(POLL_S)


def _stop(server: Any) -> Optional[int]:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
    return server.poll()


@dataclasses.dataclass
class _Session:
    """What one server session asks and records."""

    pending: list[protocol.Problem]
    ledger: records.Ledger
    identity: str
    attempt: int
    max_seq_len: int
    concurrency: int
    deadline: float


def _ask(session: _Session, base: str, model: str, server: Any, deps: RunDeps) -> dict[str, Any]:
    """Ask every pending question; each reply is recorded before the next result is read."""
    stop = threading.Event()
    url = f"{base}/v1/chat/completions"

    def ask(problem: protocol.Problem) -> Optional[dict[str, Any]]:
        if stop.is_set() or deps.monotonic() >= session.deadline or server.poll() is not None:
            return None
        body = protocol.build_request(model, problem.prompt)
        started_at = records.utc_now()
        begin = deps.monotonic()
        try:
            reply = deps.post(
                url,
                json=body,
                timeout=(CONNECT_TIMEOUT_S, max(1.0, session.deadline - begin)),
            )
            status, error, fields = records.interpret_reply(
                reply.status_code, reply.text, max_seq_len=session.max_seq_len
            )
        except requests.exceptions.Timeout as exc:
            status, error, fields = records.TIMEOUT, f"no reply before the run deadline: {exc}", {}
        except requests.exceptions.RequestException as exc:
            status, error, fields = records.ERROR, f"request failed: {exc}", {}
        record = records.make_record(
            problem,
            body,
            status,
            error,
            fields,
            identity=session.identity,
            attempt=session.attempt,
            started_at=started_at,
            finished_at=records.utc_now(),
            latency_s=deps.monotonic() - begin,
        )
        session.ledger.append(record)
        return record

    first = deps.monotonic()
    asked: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=session.concurrency, thread_name_prefix="gsm8k-first100"
    ) as pool:
        waiting = {pool.submit(ask, problem) for problem in session.pending}
        try:
            while waiting:
                done, waiting = concurrent.futures.wait(
                    waiting, timeout=POLL_S, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    record = future.result()
                    if record is not None:
                        asked.append(record)
                if server.poll() is not None or deps.monotonic() >= session.deadline:
                    stop.set()
        finally:
            stop.set()
    asked_ids = {record["id"] for record in asked}
    return {
        "evaluation_s": deps.monotonic() - first,
        "asked": len(asked),
        "not_asked": [p.id for p in session.pending if p.id not in asked_ids],
        "server_exit_code_during_evaluation": server.poll(),
    }


def _serve(
    session: _Session,
    argv: list[str],
    env: Mapping[str, str],
    files: Mapping[str, Path],
    gpu_ids: list[str],
    deps: RunDeps,
) -> dict[str, Any]:
    """One server session: launch, wait until ready, ask, stop; returns its timings."""
    timing: dict[str, Any] = {"startup_error": None}
    with deps.memory_poller(files["memory"], gpu_ids):
        launched = deps.monotonic()
        stopping: Optional[float] = None
        try:
            with deps.launch(argv, env, files["log"]) as server:
                try:
                    limit = min(session.deadline, launched + STARTUP_LIMIT_S)
                    try:
                        host, port = deps.wait_for_addr(
                            str(files["addr"]), max(1.0, limit - deps.monotonic()), server
                        )
                        timing["address_s"] = deps.monotonic() - launched
                        base = f"http://{host}:{port}"
                        model = _wait_ready(base, server, limit, deps)
                    except (RuntimeError, TimeoutError) as error:
                        timing["startup_error"] = f"{type(error).__name__}: {error}"
                    else:
                        timing["startup_s"] = deps.monotonic() - launched
                        timing["served_model"] = model
                        timing.update(_ask(session, base, model, server, deps))
                finally:
                    stopping = deps.monotonic()
                    timing["server_exit_code"] = _stop(server)
        finally:
            ended = deps.monotonic()
            timing["teardown_s"] = ended - stopping if stopping is not None else None
            timing["server_lifetime_s"] = ended - launched
    return timing


def _report(summary: Mapping[str, Any], timing: Mapping[str, Any]) -> str:
    def seconds(key: str) -> str:
        value = timing.get(key)
        return "n/a" if value is None else f"{value:.1f} s"

    memory = timing.get("gpu_memory") or {}
    lines = [
        f"first-100 GSM8K ({summary['protocol_id']}, {summary['mode']} mode): "
        f"{summary['correct']}/{summary['total']} correct",
        f"  completed {summary['distinct_completed']}/{summary['expected']}; parse failures "
        f"{len(summary['parse_failures'])}; errors {len(summary['errors'])}; timeouts "
        f"{len(summary['timeouts'])}; truncations {len(summary['truncated'])}; missing "
        f"{len(summary['missing'])}",
        f"  attempt {summary['attempt']}: verification {seconds('verification_s')}, startup "
        f"{seconds('startup_s')}, evaluation {seconds('evaluation_s')}, teardown "
        f"{seconds('teardown_s')}, total {seconds('total_s')}",
        f"  peak GPU memory {memory.get('max_peak_used_mib')} MiB (largest GPU); GPU-hours "
        f"{timing.get('gpu_hours')} (all attempts {summary['cost']['gpu_hours_all_attempts']})",
        f"  results: {summary['results_dir']}",
    ]
    return "\n".join(lines)


def run_first100(
    *,
    environ: Mapping[str, str],
    models_root: Callable[[], str],
    repo_root: Callable[[], str],
    timeout_s: Optional[float],
    output_dir: Optional[str],
    deps: Optional[RunDeps] = None,
    manifest_path: Path = protocol.MANIFEST_PATH,
    criterion_path: Path = records.CRITERION_PATH,
) -> dict[str, Any]:
    """Run or resume the qualification and return its summary; skip unless triggered.

    Args:
        environ: The process environment (trigger, mode and staged input paths).
        models_root: Returns the models root; called only after the trigger check.
        repo_root: Returns the repository root holding the curated recipe.
        timeout_s: The test's pytest timeout in seconds; a finite value is required.
        output_dir: The pytest ``--output-dir``; results go to a subdirectory of it.
        deps: Process/HTTP/GPU/clock entry points; the real ones by default.
        manifest_path: The frozen item manifest.
        criterion_path: The installed regression criterion (regression mode only).
    """
    if not _enabled(environ):
        pytest.skip(SKIP_REASON)
    deps = deps or RunDeps()
    started = deps.monotonic()
    started_at = records.utc_now()
    try:
        mode = deployment.run_mode(environ)
    except ValueError as error:
        _fail(str(error))
    if timeout_s is None or timeout_s <= TEARDOWN_RESERVE_S + 60:
        _fail(
            f"a finite pytest timeout above {TEARDOWN_RESERVE_S + 60} s is required (test-db "
            f"TIMEOUT or --timeout), got {timeout_s!r}"
        )
    deadline = started + float(timeout_s) - TEARDOWN_RESERVE_S
    manifest_sha256 = protocol.sha256_file(manifest_path)
    criterion = None
    if mode == "regression":
        try:
            criterion = records.load_criterion(criterion_path, manifest_sha256=manifest_sha256)
        except records.CriterionError as error:
            _fail(f"regression mode refused: {error}")

    try:
        manifest = protocol.load_manifest(manifest_path)
        inputs = deployment.resolve_inputs(environ, models_root())
        problems, dataset_files = protocol.load_problems(inputs.dataset_dir, manifest)
        target = deployment.describe_checkpoint(
            inputs.target, deployment.TARGET_FILES, deployment.TARGET_OPTIONAL_FILES
        )
        drafter = deployment.describe_checkpoint(
            inputs.drafter, deployment.DRAFTER_FILES, deployment.DRAFTER_OPTIONAL_FILES
        )
        kv_scales = deployment.describe_kv_scales(
            inputs.kv_scales, target["num_hidden_layers"], drafter["num_hidden_layers"]
        )
        gpus = deps.query_gpus(environ)
    except (AssertionError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        _fail(f"inputs are not usable: {type(error).__name__}: {error}")
    hardware_problems = deployment.check_hardware(gpus)
    if hardware_problems:
        _fail("; ".join(hardware_problems))
    gpus = gpus[: deployment.REQUIRED_GPU_COUNT]

    try:
        recipe_text, recipe = deployment.materialize_recipe(
            os.path.join(repo_root(), deployment.RECIPE_RELATIVE_PATH), inputs.drafter
        )
    except (OSError, ValueError) as error:
        _fail(f"the recipe is not usable: {error}")

    identity = {
        "protocol_id": protocol.PROTOCOL_ID,
        "manifest_sha256": manifest_sha256,
        "request_fields": protocol.SAMPLING_FIELDS,
        "recipe_template_sha256": recipe["template_sha256"],
        "serve_command": deployment.serve_command("<TARGET>", "<CONFIG>", "<ADDR>"),
        "server_environment": {
            **deployment.SERVER_ENV,
            deployment.KV_SCALES_SERVER_ENV: "<KV_SCALES>",
        },
        "target": _content_identity(target),
        "drafter": _content_identity(drafter),
        "kv_scales": kv_scales,
        "gpus": [[gpu["name"], gpu["compute_cap"]] for gpu in gpus],
        "tensorrt_llm": _package_version("tensorrt_llm"),
        "source_revision": _source_revision(repo_root()),
    }
    identity_sha256 = protocol.sha256_text(protocol.canonical_json(identity))
    results_dir = _results_dir(output_dir)
    manifest_file = results_dir / "run_manifest.json"
    run_manifest = _read_json(manifest_file)
    if run_manifest and run_manifest["identity_sha256"] != identity_sha256:
        _fail(
            f"{results_dir} holds a run with identity {run_manifest['identity_sha256']}, not "
            f"{identity_sha256} (changed protocol, inputs, recipe, hardware or source); "
            "use a fresh --output-dir"
        )
    if criterion is not None:
        mismatched = records.criterion_mismatches(criterion, identity)
        if mismatched:
            _fail(
                "regression mode refused: the installed criterion was set for a different "
                f"{', '.join(mismatched)}"
            )
    ledger = records.Ledger(
        results_dir / "results.jsonl",
        identity_sha256,
        {problem.id: (problem.index, problem.prompt_sha256) for problem in problems},
    )
    try:
        state = ledger.load()
    except records.LedgerError as error:
        _fail(f"cannot resume {results_dir}: {error}")
    pending = [problem for problem in problems if problem.id not in state.completed]

    if run_manifest is None:
        run_manifest = {
            "protocol_id": protocol.PROTOCOL_ID,
            "identity_sha256": identity_sha256,
            "identity": identity,
            "attempts": [],
        }
    attempt = len(run_manifest["attempts"]) + 1
    config_path = results_dir / f"recipe-attempt-{attempt}.yaml"
    config_path.write_text(recipe_text, encoding="utf-8")
    addr_path = results_dir / f"server-attempt-{attempt}.addr"
    argv = deployment.serve_command(inputs.target, str(config_path), str(addr_path))
    attempt_info: dict[str, Any] = {
        "attempt": attempt,
        "mode": mode,
        "started_at": started_at,
        "pytest_timeout_s": timeout_s,
        "staged_inputs": inputs.relative,
        "dataset_files": dataset_files,
        "target": target,
        "drafter": drafter,
        "recipe": recipe,
        "server_command": argv,
        "server_environment": {
            **deployment.SERVER_ENV,
            deployment.KV_SCALES_SERVER_ENV: inputs.relative[deployment.KV_SCALES_ENV],
        },
        "gpus": gpus,
        "python": platform.python_version(),
        "already_completed": len(problems) - len(pending),
        "pending": len(pending),
        "quarantined_partial_writes": state.quarantined,
    }
    run_manifest["attempts"].append(attempt_info)
    _write_json(manifest_file, run_manifest)

    timing: dict[str, Any] = {"verification_s": deps.monotonic() - started}
    if pending:
        session = _Session(
            pending=pending,
            ledger=ledger,
            identity=identity_sha256,
            attempt=attempt,
            max_seq_len=recipe["max_seq_len"],
            concurrency=recipe["max_batch_size"],
            deadline=deadline,
        )
        files = {
            "addr": addr_path,
            "log": results_dir / f"server-attempt-{attempt}.log",
            "memory": results_dir / f"gpu-memory-attempt-{attempt}.csv",
        }
        timing.update(
            _serve(
                session,
                argv,
                deployment.server_environment(environ, inputs.kv_scales),
                files,
                [gpu["index"] for gpu in gpus],
                deps,
            )
        )
        if files["memory"].exists():
            timing["gpu_memory"] = records.gpu_memory_peaks(files["memory"].read_text())
    timing["total_s"] = deps.monotonic() - started
    timing["gpu_hours"] = records.gpu_hours(len(gpus), timing["total_s"])
    timing["server_gpu_hours"] = records.gpu_hours(len(gpus), timing.get("server_lifetime_s", 0))
    attempt_info.update(finished_at=records.utc_now(), timing=timing)

    summary = records.summarize(ledger.load(), problems)
    summary.update(
        mode=mode,
        run_identity=identity_sha256,
        attempt=attempt,
        results_dir=str(results_dir),
        timing=timing,
        cost={
            "gpu_count": len(gpus),
            "gpu_hours_all_attempts": round(
                sum(a.get("timing", {}).get("gpu_hours", 0) for a in run_manifest["attempts"]), 4
            ),
            "server_gpu_hours_all_attempts": round(
                sum(
                    a.get("timing", {}).get("server_gpu_hours", 0) for a in run_manifest["attempts"]
                ),
                4,
            ),
        },
    )
    summary["verdict"] = records.verdict(summary, mode, criterion)
    attempt_info["verdict"] = summary["verdict"]
    _write_json(manifest_file, run_manifest)
    _write_json(results_dir / "summary.json", summary)
    print(_report(summary, timing))
    problems_found = list(summary["verdict"]["problems"])
    if timing.get("startup_error"):
        problems_found.insert(0, f"server startup failed: {timing['startup_error']}")
    if problems_found:
        _fail("; ".join(problems_found) + f" (results: {results_dir})")
    return summary
