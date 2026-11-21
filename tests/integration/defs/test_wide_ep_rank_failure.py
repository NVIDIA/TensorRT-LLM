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
"""End-to-end tests of Wide-EP fault tolerance, serving DeepSeek under load while ranks fail.

The tests start DeepSeek-V3-Lite with attention DP and EP over 4 GPUs as plain ``trtllm-serve``,
whose node supervisor starts the ranks but 0, and fail its ranks. The membership comes from the
records the authority publishes in the rendezvous store.

Load and GSM8K clients run throughout. After a ``healthy`` phase, each failure adds the phases
``failover`` (the requests in flight at the failure), ``degraded``, ``admission`` (in flight when
the replacement is admitted) and ``rejoined``. A request that only the failed rank held gets no
answer, so the clients give up on it after REQUEST_TIMEOUT_S.
"""

import contextlib
import glob
import http.client
import json
import math
import os
import random
import re
import signal
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import psutil
import pytest
import yaml

from tensorrt_llm._torch.distributed.membership import (
    RECORD_KEY,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    MembershipRecord,
    MembershipState,
    open_store,
)
from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.llmapi.mpi_session import split_mpi_env

from .common import get_free_port_in_ci
from .conftest import llm_models_root

WORLD_SIZE = 4
VICTIM = 2
STARTUP_TIMEOUT_S = 1800
STORE_POLL_S = 0.01

# Closed-loop load. Half of the clients stream, and half of each group samples with a seed.
CLIENTS_PER_RANK = 16
MAX_CLIENTS = 256
LOAD_INPUT_LEN = (128, 512)
LOAD_MAX_TOKENS = 512
SAMPLING = {"temperature": 0.8, "top_p": 0.95}
REQUEST_TIMEOUT_S = 300

# GSM8K with the lm-eval task on the first EVAL_QUESTIONS questions. A window phase lasts until
# every question is answered in it, at most EVAL_COVER_EXTRA_S past WINDOW_S.
EVAL_CLIENTS_PER_RANK = 4
MAX_EVAL_CLIENTS = 64
EVAL_QUESTIONS = 256
EVAL_COVER_EXTRA_S = 120
# A phase must not lose paired accuracy against the healthy phase: a one-sided exact sign test
# (McNemar) at ACCURACY_ALPHA, Bonferroni-corrected by the number of later phases. Each test's paired
# sign test fails without an accuracy loss with probability at most 5%.
ACCURACY_FILTER = "strict-match"
ACCURACY_ALPHA = 0.05

WARMUP_S = 20
SETTLE_S = 10
WINDOW_S = 60
# A stall is a gap of this long without a streamed token while the load runs.
STALL_S = 5.0

# The test runs on a shared CI node, so it checks a generous bound of the pauses at the failure and
# while the replacement is admitted.
PAUSE_BOUND_S = 30.0
RECOVERY_TIMEOUT_S = 120
REJOIN_TIMEOUT_S = 1200
SHUTDOWN_TIMEOUT_S = 120

# A rank logs this per graph shape when it captures, which only the warmup may do.
CAPTURE_LINE = re.compile(r"\[RANK (\d+)\].*(CUDA graph capture|Capturing \d+ graphs)")
GPU_DEVICE = re.compile(r"/dev/nvidia\d+")
SEQUENCES = ["crash"]


def _error_body(error: urllib.error.HTTPError) -> bytes:
    """Return the body of an HTTP error response, or empty bytes if it breaks off."""
    try:
        return error.read()
    except (OSError, http.client.HTTPException):
        return b""


def _get_status(url: str, timeout_s: float = 5.0) -> Optional[int]:
    """Return the HTTP status, or None without an answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, http.client.HTTPException):
        return None


def _post(base: str, path: str, body: dict):
    request = urllib.request.Request(
        f"{base}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S)


@dataclass
class Deployment:
    base: str
    model: str
    logs: str  # glob pattern of the files the ranks log to
    secret: str
    world_size: int
    victim: int


def _serving_config(model: Path) -> dict:
    num_experts = json.loads((model / "config.json").read_text())["n_routed_experts"]
    return {
        "enable_wide_ep_fault_tolerance": True,
        "enable_attention_dp": True,
        "disable_overlap_scheduler": False,
        # A request continued by recompute prefills up to max_seq_len tokens in one context.
        "max_seq_len": 4096,
        "cuda_graph_config": {"enable_padding": True, "max_batch_size": 32},
        "kv_cache_config": {"free_gpu_memory_fraction": 0.5, "use_kv_cache_manager_v2": True},
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
        # Static EPLB with one slot per expert. Fault tolerance adds the spare slots the survivors
        # need to hold every expert, and restores a failed rank's experts there from host weights.
        "moe_config": {"load_balancer": {"num_slots": num_experts}},
    }


def _wait_until_healthy(base: str, alive: Callable[[], bool]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while _get_status(f"{base}/health") != 200:
        assert alive(), "the server exited during startup"
        assert time.monotonic() < deadline, "the server did not become healthy"
        time.sleep(5)


def _chosen_secret(since: float) -> str:
    """Return the shared secret a server spawning its ranks chose, read from its node supervisor."""
    for process in psutil.process_iter(["cmdline", "create_time"]):
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            cmdline = process.info["cmdline"] or []
            if process.info["create_time"] >= since and any(
                arg.endswith("_llmapi_node_supervisor.py") for arg in cmdline
            ):
                return process.environ()[SHARED_SECRET_ENV]
    raise LookupError("no node supervisor of the deployment runs on this node")


@contextlib.contextmanager
def _deploy(tmp_path: Path):
    """Start a deployment as plain ``trtllm-serve``, which spawns its ranks."""
    model = Path(llm_models_root()) / "DeepSeek-V3-Lite" / "bf16"
    if not model.exists():
        pytest.skip(f"{model} does not exist")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_serving_config(model)))
    port = get_free_port_in_ci()
    base = f"http://localhost:{port}"
    log = tmp_path / "server.log"
    serve = [
        "trtllm-serve",
        str(model),
        f"--tp_size={WORLD_SIZE}",
        f"--ep_size={WORLD_SIZE}",
        f"--port={port}",
        f"--config={config}",
    ]
    # A single process outside any MPI or Slurm job, as a user starts it. Set info level on every
    # rank so that each rank logs its CUDA graph captures. Only the leader takes the server's level.
    env = {**split_mpi_env()[0], "TLLM_LOG_LEVEL": "info"}
    if "PMIX_HOSTNAME" in os.environ:
        env["PMIX_HOSTNAME"] = os.environ["PMIX_HOSTNAME"]
    launched_s = time.time()
    with open(log, "w") as log_file:
        server = subprocess.Popen(
            serve, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
    secret = None
    try:
        _wait_until_healthy(base, lambda: server.poll() is None)
        secret = _chosen_secret(launched_s)
        yield Deployment(base, str(model), str(log), secret, WORLD_SIZE, VICTIM)
    finally:
        if secret is not None:
            for process in _node_processes(secret):
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.resume()  # a stopped process does not take the interrupt
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGINT)
        try:
            server.wait(SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        left_running = []
        if secret is not None:  # replacements run outside the server's process group
            _, left_running = psutil.wait_procs(list(_node_processes(secret)), timeout=60)
            for process in left_running:
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.kill()
            psutil.wait_procs(left_running, timeout=60)
    assert not left_running, (
        f"processes of the deployment still ran 60 s after it ended: {left_running}"
    )


@pytest.fixture
def spawned_deployment(tmp_path: Path):
    """Start the deployment with ``trtllm-serve``, which spawns the ranks and picks the secrets."""
    with _deploy(tmp_path) as deployment:
        yield deployment


def _holds_gpu(pid: int) -> bool:
    try:
        return any(
            GPU_DEVICE.fullmatch(os.readlink(f"/proc/{pid}/fd/{fd}"))
            for fd in os.listdir(f"/proc/{pid}/fd")
        )
    except OSError:
        return False


def _node_processes(secret: str) -> Dict[psutil.Process, Dict[str, str]]:
    """Return the environment of each process of the deployment on this node, except this test."""
    test = psutil.Process()
    own = {test.pid, *(p.pid for p in test.parents())}
    found = {}
    for process in psutil.process_iter():
        if process.pid in own:
            continue
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            env = process.environ()
            if env.get(SHARED_SECRET_ENV) == secret:
                found[process] = env
    return found


def _rank_workers(secret: str) -> Dict[int, psutil.Process]:
    """Return each rank's outermost process on this node that holds a GPU."""
    candidates = {}
    for process, env in _node_processes(secret).items():
        identity = env.get(FAULT_TOLERANCE_RANK_ENV)
        if identity is not None and _holds_gpu(process.pid):
            candidates[process.pid] = (int(identity.split(",")[0]), process)
    workers = {}
    for pid, (rank, process) in candidates.items():
        try:
            parent = process.ppid()
        except psutil.NoSuchProcess:
            continue
        if candidates.get(parent, (None,))[0] == rank:
            continue
        if rank in workers:
            raise LookupError(
                f"rank {rank} has two worker processes: {workers[rank].pid} and {pid}"
            )
        workers[rank] = process
    return workers


def _rendezvous(secret: str) -> str:
    """Return the rendezvous store address from a process of the deployment on this node."""
    for env in _node_processes(secret).values():
        if RENDEZVOUS_ENV in env:
            return env[RENDEZVOUS_ENV]
    raise LookupError("no process of the deployment on this node has the rendezvous address")


def _log_lines(logs: str):
    """Yield every line of the deployment's logs, given as a glob pattern."""
    for log in sorted(glob.glob(logs)):
        with open(log, errors="replace") as lines:
            yield from lines


def _capture_lines(logs: str) -> Dict[int, int]:
    """Count the CUDA graph capture lines of each rank in the deployment's logs."""
    counts: Dict[int, int] = {}
    for match in filter(None, map(CAPTURE_LINE.search, _log_lines(logs))):
        rank = int(match.group(1))
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def _nvidia_smi(kind: str, fields: str) -> List[List[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[v.strip() for v in line.split(",")] for line in out.splitlines() if line.strip()]


def _rank_gpus(workers: Dict[int, psutil.Process]) -> Dict[int, List[int]]:
    """Return the GPU indices of each rank's worker, from nvidia-smi's compute processes."""
    index = {uuid: int(i) for i, uuid in _nvidia_smi("gpu", "index,uuid")}
    gpus: Dict[int, List[int]] = {}
    for pid, uuid in _nvidia_smi("compute-apps", "pid,gpu_uuid"):
        gpus.setdefault(int(pid), []).append(index[uuid])
    return {r: sorted(gpus.get(p.pid, [])) for r, p in sorted(workers.items())}


def _prompt(seed: int, length: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 30000) for _ in range(length)]


class Gsm8k:
    """GSM8K questions and their scoring with the lm-eval task ``tensorrt_llm.evaluate.GSM8K``."""

    def __init__(self, num_questions: int):
        from tensorrt_llm.evaluate import GSM8K

        path = Path(llm_models_root()) / "datasets" / "openai" / "gsm8k"
        assert path.exists(), f"GSM8K is not available offline at {path}"
        self._task = GSM8K(dataset_path=str(path), random_seed=0).task_dict["gsm8k"]
        self._task.build_all_requests(limit=num_questions, rank=0, world_size=1)
        self.instances = self._task.instances

    def request(self, question: int) -> dict:
        """Return the completion request for a question with its few-shot context."""
        context, gen_kwargs = self.instances[question].args
        return {
            "prompt": context,
            "max_tokens": gen_kwargs.get("max_gen_toks", 256),
            "temperature": gen_kwargs.get("temperature", 0.0),
            "stop": gen_kwargs.get("until", []),
        }

    def score(self, answers: Dict[int, str]) -> Dict[int, float]:
        """Score one answer per question with ACCURACY_FILTER, as the lm-eval evaluator does."""
        for question, instance in enumerate(self.instances):
            instance.resps = [answers.get(question, "")]
            instance.filtered_resps = {}
        self._task.apply_filters()
        return {
            question: float(
                self._task.process_results(
                    self.instances[question].doc,
                    [self.instances[question].filtered_resps[ACCURACY_FILTER]],
                )["exact_match"]
            )
            for question in answers
        }


@dataclass
class Request:
    stream: bool
    start: float
    question: Optional[int] = None  # an eval request's GSM8K question
    end: float = 0.0
    # The arrival time of each streamed chunk with tokens.
    arrivals: List[float] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    completion_tokens: Optional[int] = None
    error: Optional[str] = None

    def spans(self, t: float) -> bool:
        return self.start < t < self.end


def _thread(target: Callable[..., None], args: tuple, failures: List[str]) -> threading.Thread:
    """Return a daemon thread that runs ``target`` and appends its exception to ``failures``."""

    def run() -> None:
        try:
            target(*args)
        except Exception:  # re-raised, recorded so that the run fails on it
            failures.append(f"{threading.current_thread().name} ended: {traceback.format_exc()}")
            raise

    return threading.Thread(target=run, daemon=True)


def _join(threads: List[threading.Thread], timeout_s: float, failures: List[str]) -> List[str]:
    """Join ``threads`` within ``timeout_s`` and return ``failures`` plus one per running thread."""
    deadline = time.monotonic() + timeout_s
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return failures + [
        f"{thread.name} still runs {timeout_s} s after it was told to stop"
        for thread in threads
        if thread.is_alive()
    ]


class Clients:
    """Closed-loop load and eval clients, as the module docstring describes."""

    def __init__(self, deployment: Deployment, gsm8k: Gsm8k):
        self._deployment = deployment
        self._gsm8k = gsm8k
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._seed = 0
        self._question = 0
        self.requests: List[Request] = []
        self._failures: List[str] = []
        world = deployment.world_size
        self._threads = [
            _thread(self._load_client, (i,), self._failures)
            for i in range(min(CLIENTS_PER_RANK * world, MAX_CLIENTS))
        ] + [
            _thread(self._eval_client, (), self._failures)
            for _ in range(min(EVAL_CLIENTS_PER_RANK * world, MAX_EVAL_CLIENTS))
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> List[str]:
        """Stop sending, wait for the requests in flight, and return the clients that failed."""
        self._stop.set()
        return _join(self._threads, REQUEST_TIMEOUT_S + 60, self._failures)

    def restart_questions(self) -> None:
        """Restart the eval clients from the first question of the subset."""
        with self._lock:
            self._question = 0

    def answered(self, since: float) -> set:
        """Return the questions answered by requests that started at ``since`` or later."""
        with self._lock:
            requests = list(self.requests)
        return {
            r.question
            for r in requests
            if r.question is not None and r.error is None and r.start >= since
        }

    def _next(self, attribute: str) -> int:
        with self._lock:
            value = getattr(self, attribute)
            setattr(self, attribute, value + 1)
            return value

    def _load_client(self, index: int) -> None:
        stream, sampled = index % 2 == 0, index // 2 % 2 == 1
        while not self._stop.is_set():
            seed = self._next("_seed")
            body = {
                "prompt": _prompt(seed, random.Random(seed).randint(*LOAD_INPUT_LEN)),
                "max_tokens": LOAD_MAX_TOKENS,
                "ignore_eos": True,
                "detokenize": False,
                **({**SAMPLING, "seed": seed} if sampled else {"temperature": 0.0}),
            }
            self._send(body, stream)

    def _eval_client(self) -> None:
        while not self._stop.is_set():
            question = self._next("_question") % len(self._gsm8k.instances)
            self._send(self._gsm8k.request(question), False, question)

    def _send(self, body: dict, stream: bool, question: Optional[int] = None) -> Request:
        request = Request(stream=stream, start=time.monotonic(), question=question)
        body = {"model": self._deployment.model, **body, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        try:
            with _post(self._deployment.base, "/v1/completions", body) as response:
                if stream:
                    self._read_stream(response, request)
                else:
                    result = json.load(response)
                    choice = result["choices"][0]
                    request.token_ids = choice.get("token_ids") or []
                    request.text = choice.get("text") or ""
                    request.completion_tokens = result["usage"]["completion_tokens"]
        except urllib.error.HTTPError as error:
            request.error = f"HTTP {error.code}: {_error_body(error).decode(errors='replace')}"
        except (OSError, http.client.HTTPException, ValueError, LookupError, TypeError) as error:
            # No answer within REQUEST_TIMEOUT_S, a response that broke off (IncompleteRead,
            # BadStatusLine), or one that is not a completion.
            request.error = repr(error)
        finally:
            # Record the request even when another exception ends the client, which fails the run.
            request.end = time.monotonic()
            with self._lock:
                self.requests.append(request)
        return request

    @staticmethod
    def _read_stream(response, request: Request) -> None:
        for line in response:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if data == b"[DONE]":
                return
            chunk = json.loads(data)
            if "error" in chunk:
                request.error = chunk["error"]["message"]
                continue
            now = time.monotonic()
            for choice in chunk.get("choices") or []:
                ids = choice.get("token_ids") or []
                if ids:
                    request.arrivals.append(now)
                    request.token_ids.extend(ids)
            if chunk.get("usage"):
                request.completion_tokens = chunk["usage"]["completion_tokens"]
        # A stream that breaks off ends the iteration without an exception.
        request.error = "the stream ended before [DONE]"

    def load_requests(self) -> List[Request]:
        return [r for r in self.requests if r.question is None]

    def stream_arrivals(self) -> List[float]:
        return sorted(t for r in self.load_requests() for t in r.arrivals)


def _largest_gap(times: List[float], start: float, end: float) -> Tuple[float, float]:
    """Return the length and start of the longest interval from [start, end] to the next arrival."""
    inside = [start] + [t for t in times if start < t < end]
    later = [t for t in times if t >= end]
    inside.append(later[0] if later else end)
    return max((b - a, a) for a, b in zip(inside, inside[1:]))


class Monitor:
    """Sample the server's ``/health``, and record each membership record in the store."""

    def __init__(self, deployment: Deployment):
        self._base = deployment.base
        self._stop = threading.Event()
        self._changed = threading.Condition()
        self._failures: List[str] = []
        self.health: List[Tuple[float, Optional[int]]] = []
        self.records: List[Tuple[float, MembershipRecord]] = []
        self._address = _rendezvous(deployment.secret)
        self._secret = deployment.secret
        self._store = open_store(self._address, timeout_s=60.0, secret=self._secret)
        self._threads = [
            _thread(self._sample_health, (), self._failures),
            _thread(self._sample_records, (), self._failures),
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> List[str]:
        """Stop watching and return the samplers that raised or still run."""
        self._stop.set()
        return _join(self._threads, 60.0, self._failures)

    def latest(self) -> MembershipRecord:
        """Return the record the store holds now."""
        return MembershipRecord.model_validate_json(self._store.get(RECORD_KEY))

    def _sample_health(self) -> None:
        while not self._stop.is_set():
            self.health.append((time.monotonic(), _get_status(f"{self._base}/health")))
            self._stop.wait(0.25)

    def _sample_records(self) -> None:
        # Its own client, since the test's thread reads the store too.
        store = open_store(self._address, timeout_s=60.0, secret=self._secret)
        while not self._stop.is_set():
            value = store.get(RECORD_KEY) if store.check([RECORD_KEY]) else None
            record = None if value is None else MembershipRecord.model_validate_json(value)
            t = time.monotonic()
            if record is not None and (not self.records or record != self.records[-1][1]):
                with self._changed:
                    self.records.append((t, record))
                    self._changed.notify_all()
            self._stop.wait(STORE_POLL_S)

    def wait_for(
        self, predicate: Callable[[MembershipRecord], bool], since: float, timeout_s: float
    ) -> Tuple[float, MembershipRecord]:
        """Return the time and the first record after ``since`` that satisfies ``predicate``."""
        deadline = time.monotonic() + timeout_s
        checked = 0
        with self._changed:
            while True:
                for t, record in self.records[checked:]:
                    if t > since and predicate(record):
                        return t, record
                checked = len(self.records)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._failures:
                    last = self.records[-1] if self.records else None
                    raise AssertionError(
                        f"not reached within {timeout_s} s; last record: {last}; "
                        f"failed samplers: {self._failures}"
                    )
                self._changed.wait(min(remaining, 1.0))

    def wait_for_health(self, since: float, timeout_s: float) -> None:
        """Wait until ``/health`` answered 200 after ``since``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._failures:
            if any(t > since and code == 200 for t, code in list(self.health)):
                return
            time.sleep(0.1)
        raise AssertionError(
            f"/health did not answer 200 within {timeout_s} s; last samples: "
            f"{self.health[-3:]}; failed samplers: {self._failures}"
        )


def _active_ranks(record: MembershipRecord, world: int) -> List[int]:
    return [rank for rank in range(world) if record.is_active(rank)]


def _committed_after(generation: int) -> Callable[[MembershipRecord], bool]:
    return lambda r: r.state == MembershipState.COMMITTED and r.generation > generation


def _all_active(world: int) -> Callable[[MembershipRecord], bool]:
    return lambda r: r.state == MembershipState.COMMITTED and _active_ranks(r, world) == list(
        range(world)
    )


def _admitting(rank: int, after_generation: int) -> Callable[[MembershipRecord], bool]:
    return lambda r: r.generation > after_generation and r.is_joining(rank)


@dataclass
class Phase:
    name: str
    cycle: int
    start: float
    end: float
    # For a phase that spans an event, the phase's requests are those in flight at that time.
    event: Optional[float] = None

    def covers(self, request: Request) -> bool:
        if self.event is not None:
            return request.spans(self.event)
        return self.start <= request.start and request.end <= self.end


class Run:
    """Run the load, the monitor and failures on one deployment, then check."""

    def __init__(self, deployment: Deployment):
        self.monitor = Monitor(deployment)
        record = self.monitor.latest()
        assert _all_active(deployment.world_size)(record), record
        self.deployment = deployment
        self.record_after_warmup = record
        self.captures_after_warmup = _capture_lines(deployment.logs)
        self.gsm8k = Gsm8k(EVAL_QUESTIONS)
        self.clients = Clients(deployment, self.gsm8k)
        self.phases: List[Phase] = []
        self.cycles = 0
        # Only requests in flight at a rank failure may fail.
        self.failure_times: List[float] = []
        self.failures: List[str] = []
        self.t0 = time.monotonic()

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.failures.append(message)

    def __enter__(self) -> "Run":
        self.t0 = time.monotonic()
        self.monitor.start()
        self.clients.start()
        try:
            time.sleep(WARMUP_S)
            self.window("healthy", 0)
        except BaseException:
            self.clients.stop()
            self.monitor.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for failure in self.clients.stop() + self.monitor.stop():
            self.check(False, failure)
        if exc[0] is None:
            self._check_accuracy()
            self._check_requests()
            self._check_captures()
            assert not self.failures, "\n".join(self.failures)

    def window(self, name: str, cycle: int) -> Phase:
        """Serve for WINDOW_S, then until every eval question is answered or EVAL_COVER_EXTRA_S."""
        self.clients.restart_questions()
        start = time.monotonic()
        time.sleep(WINDOW_S)
        deadline = time.monotonic() + EVAL_COVER_EXTRA_S
        while len(self.clients.answered(start)) < EVAL_QUESTIONS and time.monotonic() < deadline:
            time.sleep(1)
        phase = Phase(name, cycle, start, time.monotonic())
        self.phases.append(phase)
        return phase

    def cycle(self, failure: str) -> None:
        """Kill the victim rank, wait for N ranks, and serve a window."""
        assert failure == "crash", failure
        self.cycles += 1
        d, index = self.deployment, self.cycles
        world, all_ranks = d.world_size, list(range(d.world_size))
        before = self.monitor.latest()
        assert _all_active(world)(before), before
        workers_before = _rank_workers(d.secret)  # scanning the processes takes seconds
        gpus_before = _rank_gpus(workers_before)
        captures_before = _capture_lines(d.logs)
        assert d.victim in workers_before, (
            f"rank {d.victim} does not run on this node (ranks here: {sorted(workers_before)})"
        )
        survivors = [r for r in all_ranks if r != d.victim]

        t_fail = time.monotonic()
        workers_before[d.victim].kill()
        self.failure_times.append(t_fail)
        t_degraded, failover = self.monitor.wait_for(
            _committed_after(before.generation), t_fail, RECOVERY_TIMEOUT_S
        )
        self.monitor.wait_for_health(t_degraded, RECOVERY_TIMEOUT_S)
        self.phases.append(Phase("failover", index, t_fail, t_degraded, event=t_fail))
        time.sleep(SETTLE_S)
        t_settled = time.monotonic()
        self.clients.restart_questions()

        # The node supervisor starts a replacement, which the authority admits.
        t_admission, _ = self.monitor.wait_for(
            _admitting(d.victim, failover.generation), t_degraded, REJOIN_TIMEOUT_S
        )
        t_rejoined, rejoined = self.monitor.wait_for(
            _all_active(world), t_admission, REJOIN_TIMEOUT_S
        )
        self.monitor.wait_for_health(t_rejoined, RECOVERY_TIMEOUT_S)
        degraded_end = t_admission - 1.0
        self.phases.append(Phase("degraded", index, t_settled, degraded_end))
        self.phases.append(Phase("admission", index, t_admission, t_rejoined, event=t_admission))
        time.sleep(SETTLE_S)
        rejoined_window = self.window("rejoined", index)
        # Streamed tokens must not stall between the commit at N ranks and the window's end.
        stall_s = _largest_gap(self.clients.stream_arrivals(), t_rejoined, rejoined_window.end)[0]
        self.check(
            stall_s <= STALL_S,
            f"cycle {index}: serving stalled after the admission: "
            f"no streamed token for {stall_s:.1f} s",
        )

        workers_after = _rank_workers(d.secret)
        gpus_after = _rank_gpus(workers_after)
        captures_after = _capture_lines(d.logs)

        # A pause is the longest time without a streamed token, at the failure and at the admission.
        arrivals = self.clients.stream_arrivals()
        pause_s = _largest_gap(arrivals, t_fail - 1.0, degraded_end)[0]
        admission_pause_s = _largest_gap(arrivals, t_admission - 1.0, t_rejoined + SETTLE_S)[0]
        for what, seconds in (("the failure", pause_s), ("the admission", admission_pause_s)):
            self.check(
                seconds <= PAUSE_BOUND_S,
                f"cycle {index}: pause {seconds:.2f} s at {what} exceeds {PAUSE_BOUND_S} s",
            )
        spanning = [r for r in self.clients.load_requests() if r.spans(t_fail)]
        for stream in (False, True):
            self.check(
                any(r.stream == stream and r.error is None for r in spanning),
                f"cycle {index}: no {'streamed' if stream else 'non-streamed'} request "
                "in flight at the failure completed",
            )
        self.check(
            _active_ranks(failover, world) == survivors,
            f"cycle {index}: record after the failover: {failover}",
        )
        # No survivor restarted. Run._check_captures checks that none captured a graph since.
        for rank in survivors:
            if rank in workers_before:
                self.check(
                    rank in workers_after and workers_after[rank].pid == workers_before[rank].pid,
                    f"cycle {index}: survivor rank {rank} was restarted",
                )
        # A survivor keeps its incarnation, and the replacement takes the next one.
        expected = [n + (r == d.victim) for r, n in enumerate(before.incarnations)]
        self.check(
            list(rejoined.incarnations) == expected,
            f"cycle {index}: incarnations {before.incarnations} -> {rejoined.incarnations}, "
            f"expected {expected}",
        )
        # The replacement runs on the GPU of the process it replaces, and captures its own CUDA
        # graphs at startup.
        self.check(
            bool(gpus_before[d.victim]) and gpus_after.get(d.victim) == gpus_before[d.victim],
            f"cycle {index}: rank {d.victim}'s replacement runs on GPUs "
            f"{gpus_after.get(d.victim)}, the process it replaces ran on {gpus_before[d.victim]}",
        )
        self.check(
            captures_after.get(d.victim, 0) > captures_before.get(d.victim, 0),
            f"cycle {index}: no CUDA graph capture lines of rank {d.victim}'s replacement",
        )

    def _check_accuracy(self) -> None:
        """Check that the healthy phase answered every GSM8K question and no later phase lost accuracy."""
        requests = sorted(
            (r for r in self.clients.requests if r.question is not None and r.error is None),
            key=lambda r: r.start,
        )
        healthy = None
        alpha = ACCURACY_ALPHA / max(1, len(self.phases) - 1)
        for phase in self.phases:
            # Score each question by its first answer in the phase.
            texts: Dict[int, str] = {}
            for r in requests:
                if phase.covers(r):
                    texts.setdefault(r.question, r.text)
            answers = self.gsm8k.score(texts)
            label = f"cycle {phase.cycle} {phase.name}"
            if healthy is None:
                healthy = answers
                self.check(
                    len(answers) == EVAL_QUESTIONS,
                    f"{label}: {len(answers)} of the {EVAL_QUESTIONS} GSM8K questions answered",
                )
            else:
                paired = set(answers) & set(healthy)
                lost = sum(1 for q in paired if healthy[q] == 1.0 and answers[q] == 0.0)
                gained = sum(1 for q in paired if healthy[q] == 0.0 and answers[q] == 1.0)
                p_loss = _sign_test_p(lost, gained)
                self.check(
                    bool(paired),
                    f"{label}: no GSM8K question answered here and in the healthy phase",
                )
                self.check(
                    p_loss >= alpha,
                    f"{label}: GSM8K lost accuracy against the healthy phase: of {len(paired)} "
                    f"questions both answered, {lost} went right -> wrong and {gained} wrong -> "
                    f"right (p {p_loss:.2g} < {alpha:.2g})",
                )

    def _check_requests(self) -> None:
        """Check that every request completed in full, except requests in flight at a failure."""
        for r in self.clients.requests:
            kind = ("streamed" if r.stream else "non-streamed") + (
                " eval" if r.question is not None else ""
            )
            if r.error is None:
                if r.question is None:
                    self.check(
                        len(r.token_ids) == r.completion_tokens == LOAD_MAX_TOKENS,
                        f"a {kind} request returned {len(r.token_ids)} tokens "
                        f"(usage {r.completion_tokens}), not {LOAD_MAX_TOKENS}",
                    )
                continue
            self.check(
                any(r.spans(t) for t in self.failure_times),
                f"a {kind} request failed (started {r.start - self.t0:.1f} s, "
                f"ended {r.end - self.t0:.1f} s into the run, failures at "
                f"{[round(t - self.t0, 1) for t in self.failure_times]}): {r.error[:500]}",
            )

    def _check_captures(self) -> None:
        """Check that every rank captured CUDA graphs at startup and no original rank since."""
        d = self.deployment
        start, end = self.record_after_warmup.incarnations, self.monitor.latest()
        first_processes = [
            r for r in _active_ranks(end, d.world_size) if end.incarnations[r] == start[r]
        ]
        before, after = self.captures_after_warmup, _capture_lines(d.logs)
        for rank in range(d.world_size):
            self.check(before.get(rank, 0) > 0, f"no CUDA graph capture lines of rank {rank}")
        for rank in first_processes:
            self.check(
                after.get(rank, 0) == before.get(rank, 0),
                f"rank {rank} captured CUDA graphs after warmup: {before.get(rank, 0)} -> "
                f"{after.get(rank, 0)} capture lines",
            )


def _sign_test_p(lost: int, gained: int) -> float:
    """Return the p-value of a one-sided exact sign test (McNemar on discordant pairs)."""
    n = lost + gained
    return sum(math.comb(n, k) for k in range(lost, n + 1)) / 2**n


@pytest.mark.skip_less_device(WORLD_SIZE)
class TestRankFailures:
    """Tests that fail ranks of one deployment under load, each ending with every rank active."""

    @pytest.mark.parametrize("sequence", SEQUENCES)
    @pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
    def test_rank_failure_sequence(self, spawned_deployment: Deployment, sequence: str):
        """Fail a rank per ``-``-separated step of ``sequence``, each followed by a rejoin."""
        with Run(spawned_deployment) as run:
            for failure in sequence.split("-"):
                run.cycle(failure)
