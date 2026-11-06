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
"""One evaluation run: verify, (optionally) serve, ask, record and judge.

A run verifies the pinned dataset items, then asks every not yet completed selected question
through ``/v1/chat/completions`` with at most ``concurrency`` requests in flight, writing each
reply to ``results.jsonl`` as it arrives, and finally writes ``summary.json``. The endpoint is
either an existing server (endpoint mode: never started, stopped or signalled here) or a
``trtllm-serve`` process this run launches and owns (managed mode: always stopped on success,
failure, timeout or interruption). Running again with the same output directory and run
identity resumes; a different identity is refused. The replies of a run all come from one
served model: before its first request an attempt records the model it resolved, and a model
other than the one the recorded replies were asked of is refused without asking anything.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping, Optional

import requests

from . import protocol, records, server

#: Seconds between polls of the server and the in-flight requests.
POLL_S = 5.0
CONNECT_TIMEOUT_S = 10.0
#: Upper bound on managed-server startup (model load) inside the run's budget.
STARTUP_LIMIT_S = 3600.0


class EvaluationError(RuntimeError):
    """The run cannot start or resume (inputs, identity or results directory)."""


class ServerError(RuntimeError):
    """The server did not come up or stopped serving."""


@dataclasses.dataclass(frozen=True)
class ManagedServer:
    """A ``trtllm-serve`` launch this run owns.

    Args:
        argv: The launch command without ``--host``, ``--port`` and ``--report_addr``.
        env: The complete launch environment.
        gpu_ids: GPU indices whose used memory is polled while the server lives (optional).
        settings: Recorded launch settings (e.g. the rendered config's digest).
    """

    argv: tuple[str, ...]
    env: Mapping[str, str]
    gpu_ids: tuple[str, ...] = ()
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RunDeps:
    """Process, HTTP and clock entry points of a run (fakes replace them in unit tests)."""

    launch: Callable[..., ContextManager[Any]] = server.launch
    wait_for_addr: Callable[..., tuple[str, int]] = server.wait_for_addr
    memory_poller: Callable[..., ContextManager[Any]] = server.gpu_memory_poller
    get: Callable[..., Any] = requests.get
    post: Callable[..., Any] = requests.post
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


class _ExternalEndpoint:
    """An existing endpoint: it is never polled for an exit code and never stopped."""

    @staticmethod
    def poll() -> None:
        return None


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def served_model_ids(base: str, deps: RunDeps) -> Optional[list[str]]:
    """The model ids an OpenAI-compatible endpoint lists at ``/v1/models``.

    ``None`` when the endpoint has no such route (HTTP 404), like the router of
    ``trtllm-serve disaggregated``.
    """
    reply = deps.get(f"{base}/v1/models", timeout=CONNECT_TIMEOUT_S)
    if reply.status_code == 404:
        return None
    if reply.status_code != 200:
        raise ServerError(f"/v1/models answered HTTP {reply.status_code}")
    try:
        return [entry["id"] for entry in reply.json()["data"]]
    except (ValueError, KeyError, TypeError) as error:
        raise ServerError(f"malformed /v1/models reply: {error}") from error


def resolve_model(base: str, requested: Optional[str], deps: RunDeps) -> tuple[str, bool]:
    """The served model to ask and whether the endpoint listed it.

    ``requested`` if listed, else the endpoint's only model; an endpoint that lists no models
    is asked for ``requested``, which must then be given.
    """
    ids = served_model_ids(base, deps)
    if ids is None:
        if requested is None:
            raise ServerError(
                "the endpoint does not list its models (/v1/models answered HTTP 404); "
                "name the model with --served-model-name"
            )
        return requested, False
    if requested is not None:
        if requested not in ids:
            raise ServerError(f"the endpoint serves {ids}, not {requested!r}")
        return requested, True
    if len(ids) != 1:
        raise ServerError(f"/v1/models lists {ids}; name the model with --served-model-name")
    return ids[0], True


def _wait_ready(
    base: str, process: Any, limit: float, requested: Optional[str], deps: RunDeps
) -> tuple[str, bool]:
    """Poll ``/health`` of a managed server until it answers; return ``resolve_model``'s answer."""
    while True:
        code = process.poll()
        if code is not None:
            raise ServerError(f"trtllm-serve exited with code {code} before it was ready")
        if deps.monotonic() >= limit:
            raise ServerError("trtllm-serve was not ready within its startup budget")
        try:
            if deps.get(f"{base}/health", timeout=CONNECT_TIMEOUT_S).status_code == 200:
                return resolve_model(base, requested, deps)
        except requests.exceptions.RequestException:
            pass
        deps.sleep(POLL_S)


@dataclasses.dataclass
class _Session:
    """What one asking session sends and records."""

    pending: list[protocol.Problem]
    ledger: records.Ledger
    identity: str
    attempt: int
    max_seq_len: Optional[int]
    concurrency: int
    deadline: float
    request_timeout_s: Optional[float]


def _ask(session: _Session, base: str, model: str, process: Any, deps: RunDeps) -> dict[str, Any]:
    """Ask every pending question; each reply is recorded before the next result is read."""
    stop = threading.Event()
    url = f"{base}/v1/chat/completions"

    def ask(problem: protocol.Problem) -> Optional[dict[str, Any]]:
        if stop.is_set() or deps.monotonic() >= session.deadline or process.poll() is not None:
            return None
        body = protocol.build_request(model, problem.prompt)
        started_at = records.utc_now()
        begin = deps.monotonic()
        read_timeout = max(1.0, session.deadline - begin)
        if session.request_timeout_s is not None:
            read_timeout = min(read_timeout, session.request_timeout_s)
        try:
            reply = deps.post(url, json=body, timeout=(CONNECT_TIMEOUT_S, read_timeout))
            status, error, fields = records.interpret_reply(
                reply.status_code, reply.text, max_seq_len=session.max_seq_len
            )
        except requests.exceptions.Timeout as exc:
            status, error, fields = (
                records.TIMEOUT,
                f"no reply within {read_timeout:.0f} s: {exc}",
                {},
            )
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

    # Daemon workers: an interruption leaves at once (the caller then stops an owned server);
    # replies still in flight are not recorded and are asked again on resume.
    todo: queue.SimpleQueue[protocol.Problem] = queue.SimpleQueue()
    for problem in session.pending:
        todo.put(problem)
    asked: list[dict[str, Any]] = []
    failures: list[BaseException] = []
    lock = threading.Lock()
    workers = min(session.concurrency, len(session.pending))
    remaining = [workers]
    all_done = threading.Event()

    def worker() -> None:
        try:
            while not stop.is_set():
                try:
                    problem = todo.get_nowait()
                except queue.Empty:
                    return
                record = ask(problem)
                if record is None:
                    return
                with lock:
                    asked.append(record)
        except BaseException as error:  # noqa: BLE001 - re-raised in the asking thread
            with lock:
                failures.append(error)
            stop.set()
        finally:
            with lock:
                remaining[0] -= 1
                if remaining[0] == 0:
                    all_done.set()

    first = deps.monotonic()
    if workers == 0:
        all_done.set()
    for number in range(workers):
        threading.Thread(target=worker, name=f"gsm8k-eval-{number}", daemon=True).start()
    try:
        while not all_done.wait(timeout=POLL_S):
            if process.poll() is not None or deps.monotonic() >= session.deadline:
                stop.set()
    finally:
        stop.set()
    if failures:
        raise failures[0]
    asked_ids = {record["id"] for record in asked}
    return {
        "evaluation_s": deps.monotonic() - first,
        "asked": len(asked),
        "not_asked": [p.id for p in session.pending if p.id not in asked_ids],
        "server_exit_code_during_evaluation": process.poll(),
    }


def _serve_and_ask(
    session: _Session,
    managed: ManagedServer,
    requested_model: Optional[str],
    files: Mapping[str, Path],
    admit: Callable[[str], Optional[str]],
    deps: RunDeps,
) -> dict[str, Any]:
    """One owned server session: launch, wait until ready, ask, stop; returns its timings.

    ``admit`` receives the served model before any question is asked and returns why the run
    refuses that model, if it does; the server is then stopped without asking.
    """
    timing: dict[str, Any] = {"startup_error": None}
    argv = server.serve_argv(managed.argv, files["addr"])
    with deps.memory_poller(files["memory"], list(managed.gpu_ids)):
        launched = deps.monotonic()
        stopping: Optional[float] = None
        process = None
        try:
            with deps.launch(argv, managed.env, files["log"]) as process:
                try:
                    limit = min(session.deadline, launched + STARTUP_LIMIT_S)
                    try:
                        host, port = deps.wait_for_addr(
                            files["addr"], max(1.0, limit - deps.monotonic()), process
                        )
                        timing["address_s"] = deps.monotonic() - launched
                        base = f"http://{host}:{port}"
                        model, listed = _wait_ready(base, process, limit, requested_model, deps)
                    except (RuntimeError, TimeoutError) as error:
                        timing["startup_error"] = f"{type(error).__name__}: {error}"
                    else:
                        timing["startup_s"] = deps.monotonic() - launched
                        timing["served_model"] = model
                        timing["served_model_listed"] = listed
                        refusal = admit(model)
                        if refusal is not None:
                            timing["refused"] = refusal
                        else:
                            timing.update(_ask(session, base, model, process, deps))
                finally:
                    stopping = deps.monotonic()
        finally:
            # Leaving deps.launch stopped the server; record how it ended.
            ended = deps.monotonic()
            timing["server_exit_code"] = process.poll() if process is not None else None
            timing["teardown_s"] = ended - stopping if stopping is not None else None
            timing["server_lifetime_s"] = ended - launched
    return timing


def _report(summary: Mapping[str, Any], timing: Mapping[str, Any]) -> str:
    def seconds(key: str) -> str:
        value = timing.get(key)
        return "n/a" if value is None else f"{value:.1f} s"

    memory = timing.get("gpu_memory") or {}
    lines = [
        f"GSM8K ({summary['protocol_id']}, first {summary['total']} questions, "
        f"{summary['verdict']['mode']} mode, {summary['transport']} transport): "
        f"{summary['correct']}/{summary['total']} correct",
        f"  served model: {summary['served_model']}",
        f"  completed {summary['distinct_completed']}/{summary['expected']}; parse failures "
        f"{len(summary['parse_failures'])}; errors {len(summary['errors'])}; timeouts "
        f"{len(summary['timeouts'])}; truncations {len(summary['truncated'])}; missing "
        f"{len(summary['missing'])}",
        f"  attempt {summary['attempt']}: verification {seconds('verification_s')}, startup "
        f"{seconds('startup_s')}, evaluation {seconds('evaluation_s')}, teardown "
        f"{seconds('teardown_s')}, total {seconds('total_s')}",
    ]
    if memory:
        lines.append(
            f"  peak GPU memory {memory.get('max_peak_used_mib')} MiB (largest GPU); GPU-hours "
            f"{timing.get('gpu_hours')}"
        )
    lines.append(f"  results: {summary['results_dir']}")
    return "\n".join(lines)


def run(
    *,
    dataset_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    num_questions: int = protocol.NUM_QUESTIONS,
    concurrency: int,
    time_budget_s: float,
    endpoint: Optional[str] = None,
    served_model: Optional[str] = None,
    managed: Optional[ManagedServer] = None,
    max_seq_len: Optional[int] = None,
    min_correct: Optional[int] = None,
    request_timeout_s: Optional[float] = None,
    identity_extra: Optional[Mapping[str, Any]] = None,
    attempt_extra: Optional[Mapping[str, Any]] = None,
    manifest_path: str | os.PathLike = protocol.MANIFEST_PATH,
    deps: Optional[RunDeps] = None,
    print_report: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run or resume one evaluation and return its summary (with ``verdict``).

    Exactly one of ``endpoint`` (an existing server's base URL) and ``managed`` (a server
    this run launches and owns) is given. Raises EvaluationError when the inputs, the run
    identity, the results directory or the served model refuse the run; nothing is asked in
    that case, and a refused served model is recorded with its attempt in the run manifest.
    """
    deps = deps or RunDeps()
    started = deps.monotonic()
    started_at = records.utc_now()
    if (endpoint is None) == (managed is None):
        raise EvaluationError("give exactly one of an existing endpoint or a managed server")
    try:
        protocol.check_count(num_questions)
    except ValueError as error:
        raise EvaluationError(str(error)) from error
    if type(concurrency) is not int or concurrency < 1:
        raise EvaluationError(f"concurrency must be a positive integer, got {concurrency!r}")
    if not time_budget_s or time_budget_s <= 0:
        raise EvaluationError(f"a positive time budget is required, got {time_budget_s!r}")
    if min_correct is not None and (
        type(min_correct) is not int or not 0 <= min_correct <= num_questions
    ):
        raise EvaluationError(
            f"min_correct must be an integer in 0..{num_questions}, got {min_correct!r}"
        )
    deadline = started + float(time_budget_s)
    manifest_sha256 = protocol.sha256_file(manifest_path)
    try:
        manifest = protocol.load_manifest(manifest_path)
        problems, dataset_files = protocol.load_problems(dataset_dir, manifest, num_questions)
    except (OSError, ValueError) as error:
        raise EvaluationError(
            f"the dataset is not usable: {type(error).__name__}: {error}"
        ) from error

    transport = "endpoint" if endpoint is not None else "managed"
    identity = {
        "protocol_id": protocol.PROTOCOL_ID,
        "manifest_sha256": manifest_sha256,
        "selection": protocol.selection(num_questions),
        "request_fields": protocol.SAMPLING_FIELDS,
        "transport": transport,
        "endpoint": endpoint.rstrip("/") if endpoint is not None else None,
        "served_model": served_model,
        "max_seq_len": max_seq_len,
        **(identity_extra or {}),
    }
    identity_sha256 = protocol.sha256_text(protocol.canonical_json(identity))
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = results_dir / "run_manifest.json"
    run_manifest = _read_json(manifest_file)
    if run_manifest and run_manifest["identity_sha256"] != identity_sha256:
        raise EvaluationError(
            f"{results_dir} holds a run with identity {run_manifest['identity_sha256']}, not "
            f"{identity_sha256} (changed protocol, selection, endpoint, inputs or settings); "
            "use a fresh output directory"
        )
    ledger = records.Ledger(
        results_dir / "results.jsonl",
        identity_sha256,
        {problem.id: (problem.index, problem.prompt_sha256) for problem in problems},
    )
    try:
        state = ledger.load()
    except records.LedgerError as error:
        raise EvaluationError(f"cannot resume {results_dir}: {error}") from error
    asked_models = state.served_models()
    if len(asked_models) > 1:
        raise EvaluationError(
            f"cannot resume {results_dir}: its replies were asked of more than one model "
            f"({', '.join(map(repr, asked_models))}); use a fresh output directory"
        )
    pending = [problem for problem in problems if problem.id not in state.completed]

    if run_manifest is None:
        run_manifest = {
            "protocol_id": protocol.PROTOCOL_ID,
            "identity_sha256": identity_sha256,
            "identity": identity,
            "attempts": [],
        }
    attempt = len(run_manifest["attempts"]) + 1
    attempt_info: dict[str, Any] = {
        "attempt": attempt,
        "transport": transport,
        "started_at": started_at,
        "time_budget_s": time_budget_s,
        "concurrency": concurrency,
        "request_timeout_s": request_timeout_s,
        "min_correct": min_correct,
        "dataset_files": dataset_files,
        "python": platform.python_version(),
        "already_completed": len(problems) - len(pending),
        "pending": len(pending),
        "quarantined_partial_writes": state.quarantined,
        **(attempt_extra or {}),
    }
    if managed is not None:
        attempt_info["server_command"] = list(
            server.serve_argv(managed.argv, results_dir / f"server-attempt-{attempt}.addr")
        )
        attempt_info["server_settings"] = dict(managed.settings)
    run_manifest["attempts"].append(attempt_info)
    _write_json(manifest_file, run_manifest)

    def admit(model: str) -> Optional[str]:
        """Record the served model before the attempt's first request; why it is refused, if so."""
        attempt_info["served_model"] = model
        if asked_models and asked_models != [model]:
            return (
                f"{results_dir} holds replies of model {asked_models[0]!r}, but {model!r} is "
                "served now; a run never mixes models: use a fresh output directory"
            )
        _write_json(manifest_file, run_manifest)
        return None

    timing: dict[str, Any] = {"verification_s": deps.monotonic() - started}
    if pending:
        session = _Session(
            pending=pending,
            ledger=ledger,
            identity=identity_sha256,
            attempt=attempt,
            max_seq_len=max_seq_len,
            concurrency=concurrency,
            deadline=deadline,
            request_timeout_s=request_timeout_s,
        )
        if managed is not None:
            files = {
                "addr": results_dir / f"server-attempt-{attempt}.addr",
                "log": results_dir / f"server-attempt-{attempt}.log",
                "memory": results_dir / f"gpu-memory-attempt-{attempt}.csv",
            }
            timing.update(_serve_and_ask(session, managed, served_model, files, admit, deps))
            if files["memory"].exists():
                timing["gpu_memory"] = records.gpu_memory_peaks(files["memory"].read_text())
        else:
            base = endpoint.rstrip("/")
            try:
                model, listed = resolve_model(base, served_model, deps)
            except (ServerError, requests.exceptions.RequestException) as error:
                timing["startup_error"] = f"{type(error).__name__}: {error}"
            else:
                timing["served_model"] = model
                timing["served_model_listed"] = listed
                refusal = admit(model)
                if refusal is not None:
                    timing["refused"] = refusal
                else:
                    timing.update(_ask(session, base, model, _ExternalEndpoint(), deps))
    timing["total_s"] = deps.monotonic() - started
    if managed is not None and managed.gpu_ids:
        timing["gpu_hours"] = records.gpu_hours(len(managed.gpu_ids), timing["total_s"])
        timing["server_gpu_hours"] = records.gpu_hours(
            len(managed.gpu_ids), timing.get("server_lifetime_s", 0)
        )
    attempt_info.update(finished_at=records.utc_now(), timing=timing)
    if "refused" in timing:
        _write_json(manifest_file, run_manifest)
        raise EvaluationError(timing["refused"])

    final_state = ledger.load()
    summary = records.summarize(final_state, problems)
    summary.update(
        transport=transport,
        served_model=next(iter(final_state.served_models()), None),
        num_questions=num_questions,
        run_identity=identity_sha256,
        attempt=attempt,
        results_dir=str(results_dir),
        timing=timing,
        cost={
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
    summary["verdict"] = records.verdict(summary, min_correct)
    if timing.get("startup_error"):
        failed = "server startup failed" if managed is not None else "the endpoint is not usable"
        summary["verdict"]["problems"].insert(0, f"{failed}: {timing['startup_error']}")
        summary["verdict"]["passed"] = False
        summary["verdict"]["operational_ok"] = False
    attempt_info["verdict"] = summary["verdict"]
    _write_json(manifest_file, run_manifest)
    _write_json(results_dir / "summary.json", summary)
    print_report(_report(summary, timing))
    return summary
