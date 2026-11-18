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
"""End-to-end tests of Wide-EP fault tolerance, serving DeepSeek-V3-Lite on 4 GPUs while ranks fail.

Load and GSM8K clients run throughout. After a ``healthy`` phase, each failure adds the phases
``failover`` (the requests in flight at the failure), ``degraded``, ``admission`` (in flight when
the replacement is admitted) and ``rejoined``.
"""

import concurrent.futures
import contextlib
import glob
import http.client
import json
import math
import operator
import os
import random
import re
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from unittest import mock

import psutil
import pytest
import yaml

from tensorrt_llm._torch.distributed.membership import (
    HANG_TIMEOUT_S,
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    host_weights_name,
    open_liveness_connection,
    open_store,
)
from tensorrt_llm._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV
from tensorrt_llm.executor.rank_failure_requests import RANK_FAILURE_CODE
from tensorrt_llm.llmapi._llmapi_node_supervisor import _MAX_FAILED_REPLACEMENTS
from tensorrt_llm.llmapi.mpi_session import split_mpi_env
from tensorrt_llm.serve.fault_tolerance_health import (
    _STATUS_READ_TIMEOUT_S,
    _STATUS_REFRESH_PERIOD_S,
    FaultToleranceHealth,
)

from .common import PORTS_IN_USE, get_free_port_in_ci
from .conftest import get_device_count, llm_models_root

WORLD_SIZE = 4
# The ranks that tests fail. The node supervisor gives a rank up after _MAX_FAILED_REPLACEMENTS of
# its replacements in a row end soon after their admission, so the tests of a class, which share a
# deployment, kill each rank at most 3 times: the first kill ends its first process.
VICTIM = 2
IDLE_VICTIM = 1
REFUSED_JOIN_VICTIM = 3
STARTUP_TIMEOUT_S = 1800
STATUS_PATH = "/fault_tolerance_status"
# The endpoint refreshes once a second, so the test times membership events from the store.
STORE_POLL_S = 0.01
# Each serving frontend refreshes its metrics on its own, so a stale scrape is repeated this long.
METRICS_REFRESH_TIMEOUT_S = 10.0

# Closed-loop load. Half of the clients stream, and half of each group samples with a seed.
CLIENTS_PER_RANK = 16
MAX_CLIENTS = 256
LOAD_INPUT_LEN = (128, 512)
LOAD_MAX_TOKENS = 512
SAMPLING = {"temperature": 0.8, "top_p": 0.95}
REQUEST_TIMEOUT_S = 900

# GSM8K with the lm-eval task on the first EVAL_QUESTIONS questions. A window phase lasts until
# every question is answered in it, at most EVAL_COVER_EXTRA_S past WINDOW_S.
EVAL_CLIENTS_PER_RANK = 4
MAX_EVAL_CLIENTS = 64
EVAL_QUESTIONS = 256
EVAL_COVER_EXTRA_S = 120
# A phase must not lose paired accuracy against the healthy phase, by a one-sided exact sign test
# at ACCURACY_ALPHA, Bonferroni-corrected by the number of later phases.
ACCURACY_FILTER = "strict-match"
ACCURACY_ALPHA = 0.05

WARMUP_S = 20
SETTLE_S = 10
WINDOW_S = 60
# Timings are noisy on a shared CI node, so pauses and stalls are checked against a relaxed bound
# (after the detection of a hang). A failed check names the measured one.
RELAXED_PAUSE_BOUND_S = 30.0
RECOVERY_TIMEOUT_S = 120
REJOIN_TIMEOUT_S = 1200
EXCLUDED_EXIT_TIMEOUT_S = 60
AUTHORITY_LOSS_END_S = 120
AUTHORITY_MODULE = "tensorrt_llm._torch.distributed.membership"
REFUSAL_TIMEOUT_S = 60
# The authority logs this and the reason when it refuses a join.
REFUSAL_LOG = "Closing a liveness connection: "
RETRYABLE_LOG = re.compile(
    r"Membership generation (\d+) ended (\d+) requests that no active rank holds"
)
# The authority logs each record it publishes.
MEMBERSHIP_LOG = re.compile(r"Membership: MembershipRecord\(.*?state=<MembershipState\.(\w+)")
# trtllm-llmapi-launch logs the PID of rank 0's trtllm-serve, which a SIGINT stops.
SERVER_PID_LOG = re.compile(r"Rank0 task guard PID: \d+; workload PGID: (\d+)")
SHUTDOWN_TIMEOUT_S = 120
# uvicorn logs this once a frontend's connections closed. The launcher's frontend then stops the engine.
APP_SHUTDOWN_LOG = "Waiting for application shutdown"
# Load requests of this length outlast a replacement's CUDA graph capture, and fit max_seq_len.
DRAIN_MAX_TOKENS = 3500

# A rank logs this per graph shape when it captures.
CAPTURE_LINE = re.compile(r"\[RANK (\d+)\].*(CUDA graph capture|Capturing \d+ graphs)")
# The node supervisor logs the end of each process of the victim, and the victim why it exits.
VICTIM_END_LOG = re.compile(rf"Rank {VICTIM} (\(.*\) ended with status|exits: )")
RANK_ENV = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
GPU_DEVICE = re.compile(r"/dev/nvidia\d+")
# The states a run may read. During a hang, until the hung rank is removed, it may also read stalled.
ALLOWED_HEALTH_STATES = {"healthy", "recovering", "degraded"}


def _free_port_run(count: int) -> int:
    """Return the first of ``count`` consecutive free ports."""
    while True:
        port = get_free_port_in_ci()
        following = range(port + 1, port + count)
        if PORTS_IN_USE.intersection(following):
            continue
        with contextlib.ExitStack() as sockets, contextlib.suppress(OSError, OverflowError):
            for other in following:
                sockets.enter_context(socket.socket()).bind(("", other))
            PORTS_IN_USE.update(following)
            return port


def _error_body(error: urllib.error.HTTPError) -> bytes:
    """Return the body of an HTTP error response, or empty bytes if it breaks off."""
    try:
        return error.read()
    except (OSError, http.client.HTTPException):
        return b""


def _get_json(url: str, timeout_s: float = 5.0) -> Tuple[Optional[int], Optional[dict]]:
    """Return the HTTP status and JSON body, or (None, None) without a complete answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, _error_body(error)
    except (OSError, http.client.HTTPException):
        return None, None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def _membership_changes_metric(base: str, minimum: Dict[str, int]) -> Dict[str, float]:
    """Return the membership changes metric by kind, scraped until each reaches ``minimum``."""
    sample = re.compile(
        r'^trtllm_fault_tolerance_membership_changes_total\{[^}]*kind="(\w+)"[^}]*\} (\S+)$', re.M
    )
    deadline = time.monotonic() + METRICS_REFRESH_TIMEOUT_S
    while True:
        try:
            with urllib.request.urlopen(f"{base}/prometheus/metrics", timeout=5.0) as response:
                text = response.read().decode()
        except (OSError, http.client.HTTPException):
            text = ""
        changes = {kind: float(value) for kind, value in sample.findall(text)}
        if (
            all(changes.get(kind, 0) >= count for kind, count in minimum.items())
            or time.monotonic() >= deadline
        ):
            return changes
        time.sleep(0.1)


def _status(base: str) -> Optional[dict]:
    """Return the deployment's status, or None if the endpoint does not answer."""
    code, body = _get_json(f"{base}{STATUS_PATH}")
    return body if code == 200 else None


@dataclass
class Deployment:
    base: str
    model: str
    logs: str  # glob pattern of the files the ranks log to
    secret: str
    launcher: subprocess.Popen
    # The base URL of each serving frontend's own port, if the test reaches them one by one.
    frontends: List[str] = field(default_factory=list)

    @property
    def bases(self) -> List[str]:
        """Return the base URLs that the clients and the monitor use."""
        return self.frontends or [self.base]

    def check_running(self) -> None:
        """Fail if the launcher exited, with its exit code and the last line of the server log."""
        code = self.launcher.poll()
        if code is not None:
            last = deque(_log_lines(self.logs), maxlen=1)
            raise AssertionError(
                f"the launcher exited with {code}; last server log line: "
                f"{last[0].rstrip() if last else None!r}"
            )

    def status(self) -> dict:
        """Return the deployment's status. Fail at once if the launcher exited."""
        self.check_running()
        status = _status(self.base)
        assert status is not None, f"{self.base}{STATUS_PATH} did not answer"
        return status


# The supported targets at 4 GPUs: the checkpoint under the models root and the options its default
# configuration adds to the generated ones.
TARGETS = {
    "bf16": ("DeepSeek-V3-Lite/bf16", {}),
    **{
        name: (
            "DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
            {
                "kv_cache_config": {"dtype": "fp8"},
                "moe_config": {"backend": backend, "use_low_precision_moe_combine": True},
            },
        )
        for name, backend in (("nvfp4", "CUTEDSL"), ("nvfp4_cutlass", "CUTLASS"))
    },
    # Online EPLB: V3-Lite's 72 routed experts and a redundant slot per rank, one layer's
    # placement updated per iteration while serving.
    "bf16_online_eplb": (
        "DeepSeek-V3-Lite/bf16",
        {
            "moe_config": {
                "load_balancer": {"num_slots": 72 + WORLD_SIZE, "layer_updates_per_iter": 1}
            }
        },
    ),
}
DEFAULT_TARGET = "bf16"


def _merged(base: dict, options: dict) -> dict:
    """Return ``base`` with ``options`` merged in, nested dictionaries key by key."""
    return {
        **base,
        **{
            key: _merged(base[key], value)
            if isinstance(value, dict) and isinstance(base.get(key), dict)
            else value
            for key, value in options.items()
        },
    }


def _serving_config(model: Path, target_options: dict) -> dict:
    num_experts = json.loads((model / "config.json").read_text())["n_routed_experts"]
    config = {
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
    return _merged(config, target_options)


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
def _deploy(
    tmp_path: Path,
    server_spawns_ranks: bool = False,
    frontends: int = 0,
    target: str = DEFAULT_TARGET,
):
    """Start a deployment of ``target``.

    With ``server_spawns_ranks``, plain ``trtllm-serve`` spawns its ranks. With ``frontends``, the
    server runs that many serving frontends, each also on its own port. A started deployment must
    leave none of its host weights in /dev/shm.
    """
    checkpoint, target_options = TARGETS[target]
    model = Path(llm_models_root()) / checkpoint
    if not model.exists():
        pytest.skip(f"{model} does not exist")
    serving_config = _serving_config(model, target_options)
    if frontends:
        # Frontend i also listens on port + i + 1.
        port = _free_port_run(frontends + 1)
        serving_config.update(num_serve_frontends=frontends, serve_frontend_affinity_port_stride=1)
    else:
        port = get_free_port_in_ci()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(serving_config))
    shm_before = set(os.listdir("/dev/shm"))
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
    # Info level on every rank, so that each rank logs its CUDA graph captures.
    env = {**os.environ, "TLLM_LOG_LEVEL": "info"}
    secret = None
    if server_spawns_ranks:
        # A single process outside any MPI or Slurm job, as a user starts it.
        env = {**split_mpi_env()[0], "TLLM_LOG_LEVEL": "info"}
        if "PMIX_HOSTNAME" in os.environ:
            env["PMIX_HOSTNAME"] = os.environ["PMIX_HOSTNAME"]
        command = serve
    else:
        # The launcher's node supervisor starts the ranks with the launch's shared secret and
        # rendezvous address. --enable-recovery keeps the launcher running when a rank exits.
        secret = os.urandom(32).hex()
        env[SHARED_SECRET_ENV] = secret
        env[RENDEZVOUS_ENV] = f"{socket.gethostname()}:{get_free_port_in_ci()}"
        command = ["mpirun", "--enable-recovery", "--allow-run-as-root", "-np", "1"]
        command += ["trtllm-llmapi-launch", *serve]
    launched_s = time.time()
    with open(log, "w") as log_file:
        launcher = subprocess.Popen(
            command, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        # /health does not report the state, so it comes from the status endpoint.
        while not (
            _get_json(f"{base}/health")[0] == 200
            and (_status(base) or {}).get("state") == "healthy"
        ):
            assert launcher.poll() is None, "the server exited during startup"
            assert time.monotonic() < deadline, "the server did not become healthy"
            time.sleep(5)
        if server_spawns_ranks:
            secret = _chosen_secret(launched_s)
        with mock.patch.dict(os.environ, {SHARED_SECRET_ENV: secret}):
            # Every rank runs on this node, node 0.
            host_weights_prefix = f"{host_weights_name(0)}_"
        assert any(name.startswith(host_weights_prefix) for name in os.listdir("/dev/shm"))
        frontend_bases = [f"http://localhost:{port + i + 1}" for i in range(frontends)]
        yield Deployment(base, str(model), str(log), secret, launcher, frontend_bases)
    finally:
        if secret is not None:
            for process in _node_processes(secret):
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.resume()  # a stopped process does not take the interrupt
        # A test may have ended the deployment, and with it the launcher's process group.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(launcher.pid, signal.SIGINT)
        try:
            launcher.wait(SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(launcher.pid, signal.SIGKILL)
            launcher.wait()
        left_running = []
        if secret is not None:  # replacements run outside the launcher's process group
            _, left_running = psutil.wait_procs(list(_node_processes(secret)), timeout=60)
            for process in left_running:
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.kill()
            psutil.wait_procs(left_running, timeout=60)
    assert not left_running, (
        f"processes of the deployment still ran 60 s after it ended: {left_running}"
    )
    # The host weights are the deployment's only /dev/shm segments. Other libraries, such as Open
    # MPI, clean up their own.
    left = sorted(
        name
        for name in set(os.listdir("/dev/shm")) - shm_before
        if name.startswith(host_weights_prefix)
    )
    assert not left, f"host weights the deployment left in /dev/shm after it ended: {left}"


class _ClassDeployment:
    """The deployment that the tests of a class share, started again after a test broke it."""

    def __init__(self, tmp_path_factory: pytest.TempPathFactory):
        self._tmp_path_factory = tmp_path_factory
        self._running = contextlib.ExitStack()
        self.deployment: Optional[Deployment] = None

    def start(self) -> None:
        self.deployment = self._running.enter_context(
            _deploy(self._tmp_path_factory.mktemp("deployment"))
        )

    def end(self) -> None:
        self.deployment = None
        self._running.close()


def _serves_every_rank(deployment: Deployment) -> bool:
    """Return whether the deployment runs with every rank active, as each test leaves it."""
    if deployment.launcher.poll() is not None:
        return False
    status = _status(deployment.base)
    return status is not None and status["state"] == "healthy"


@pytest.fixture(scope="class")
def class_deployment(tmp_path_factory: pytest.TempPathFactory):
    # A class-scoped fixture is set up before the skip_less_device check runs.
    if get_device_count() < WORLD_SIZE:
        pytest.skip(f"needs {WORLD_SIZE} GPUs")
    shared = _ClassDeployment(tmp_path_factory)
    try:
        shared.start()
        yield shared
    finally:
        shared.end()


@pytest.fixture
def shared_deployment(class_deployment: _ClassDeployment):
    """Return the deployment of the test's class, which each test ends with every rank active.

    A test that leaves it otherwise ends it, so that the next test starts a new one instead of
    failing on the state that test left.
    """
    if class_deployment.deployment is None:
        class_deployment.start()
    yield class_deployment.deployment
    if not _serves_every_rank(class_deployment.deployment):
        class_deployment.end()


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
        # A process the node supervisor started has the rank first in its identity.
        names = (FAULT_TOLERANCE_RANK_ENV, *RANK_ENV)
        rank = next((env[name].split(",")[0] for name in names if name in env), None)
        if rank is not None and _holds_gpu(process.pid):
            candidates[process.pid] = (int(rank), process)
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
    return Counter(int(m.group(1)) for m in map(CAPTURE_LINE.search, _log_lines(logs)) if m)


def _retryable_failures_logged(logs: str) -> Dict[int, int]:
    """Return the requests rank 0 logged as failed with a retryable error, by membership generation."""
    matches = filter(None, map(RETRYABLE_LOG.search, _log_lines(logs)))
    return {int(m.group(1)): int(m.group(2)) for m in matches}


def _rank_gpus(workers: Dict[int, psutil.Process]) -> Dict[int, List[str]]:
    """Return the GPU UUIDs of each rank's worker, from nvidia-smi's compute processes."""
    command = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]
    gpus: Dict[int, List[str]] = {}
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    for pid, uuid in (map(str.strip, line.split(",")) for line in out.splitlines() if line.strip()):
        gpus.setdefault(int(pid), []).append(uuid)
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
    base: str  # the base URL of the frontend that the request went through
    stream: bool
    start: float
    question: Optional[int] = None  # an eval request's GSM8K question
    retry: bool = False  # sent again after a retryable failure
    end: float = 0.0
    # The arrival time of each streamed chunk with tokens.
    arrivals: List[float] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    completion_tokens: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[Union[int, str]] = None  # the OpenAI error's ``code``

    def spans(self, t: float) -> bool:
        return self.start < t < self.end

    @property
    def retryable(self) -> bool:
        return self.error_code == RANK_FAILURE_CODE


class _Threads:
    """Daemon threads that record their exceptions, started and stopped together."""

    def __init__(self):
        self._stop = threading.Event()
        self._failures: List[str] = []
        self._threads: List[threading.Thread] = []

    def _spawn(self, target: Callable[..., None], *args) -> None:
        def run() -> None:
            try:
                target(*args)
            except Exception:  # re-raised, recorded so that the run fails on it
                name = threading.current_thread().name
                self._failures.append(f"{name} ended: {traceback.format_exc()}")
                raise

        self._threads.append(threading.Thread(target=run, daemon=True))

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def _join(self, timeout_s: float) -> List[str]:
        """Stop the threads, join them within ``timeout_s`` and return the ones that failed."""
        self._stop.set()
        deadline = time.monotonic() + timeout_s
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return self._failures + [
            f"{thread.name} still runs {timeout_s} s after it was told to stop"
            for thread in self._threads
            if thread.is_alive()
        ]


class Clients(_Threads):
    """Closed-loop load and eval clients, as the module docstring describes."""

    def __init__(self, deployment: Deployment, gsm8k: Gsm8k, max_tokens: int = LOAD_MAX_TOKENS):
        super().__init__()
        self._deployment = deployment
        self._gsm8k = gsm8k
        self._max_tokens = max_tokens
        self._lock = threading.Lock()
        self._seed = 0
        self._question = 0
        self.requests: List[Request] = []
        bases = deployment.bases
        for i in range(min(CLIENTS_PER_RANK * WORLD_SIZE, MAX_CLIENTS)):
            self._spawn(self._load_client, i, bases[i % len(bases)])
        for i in range(min(EVAL_CLIENTS_PER_RANK * WORLD_SIZE, MAX_EVAL_CLIENTS)):
            self._spawn(self._eval_client, bases[i % len(bases)])

    def stop(self) -> List[str]:
        """Stop sending, wait for the requests in flight, and return the clients that failed."""
        return self._join(REQUEST_TIMEOUT_S)

    def stop_sending(self) -> None:
        """Send no new request, without waiting for the requests in flight."""
        self._stop.set()

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

    def _load_client(self, index: int, base: str) -> None:
        stream, sampled = index % 2 == 0, index // 2 % 2 == 1
        while not self._stop.is_set():
            seed = self._next("_seed")
            body = {
                "prompt": _prompt(seed, random.Random(seed).randint(*LOAD_INPUT_LEN)),
                "max_tokens": self._max_tokens,
                "ignore_eos": True,
                "detokenize": False,
                **({**SAMPLING, "seed": seed} if sampled else {"temperature": 0.0}),
            }
            self._send_with_retry(base, body, stream)

    def _eval_client(self, base: str) -> None:
        while not self._stop.is_set():
            question = self._next("_question") % len(self._gsm8k.instances)
            self._send_with_retry(base, self._gsm8k.request(question), False, question)

    def _send_with_retry(
        self, base: str, body: dict, stream: bool, question: Optional[int] = None
    ) -> None:
        """Send a request, and send it once more if it fails with a retryable error."""
        if self._send(base, body, stream, question).retryable:
            self._send(base, body, stream, question, retry=True)

    def _send(
        self, base: str, body: dict, stream: bool, question: Optional[int], retry: bool = False
    ) -> Request:
        request = Request(
            base=base, stream=stream, start=time.monotonic(), question=question, retry=retry
        )
        try:
            _send_completion(self._deployment.model, request, body)
        finally:
            # Record the request even when another exception ends the client, which fails the run.
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
                request.error_code = chunk["error"].get("code")
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


def _send_completion(model: str, request: Request, body: dict) -> Request:
    """Send ``request``, a completion of ``model`` with ``body``, and record its outcome."""
    body = {"model": model, **body, "stream": request.stream}
    if request.stream:
        body["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json"}
    url = f"{request.base}/v1/completions"
    post = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(post, timeout=REQUEST_TIMEOUT_S) as response:
            if request.stream:
                Clients._read_stream(response, request)
            else:
                result = json.load(response)
                choice = result["choices"][0]
                request.token_ids = choice.get("token_ids") or []
                request.text = choice.get("text") or ""
                request.completion_tokens = result["usage"]["completion_tokens"]
    except urllib.error.HTTPError as error:
        body = _error_body(error)
        request.error = f"HTTP {error.code}: {body.decode(errors='replace')}"
        with contextlib.suppress(ValueError, AttributeError):
            request.error_code = json.loads(body).get("code")
    except (OSError, http.client.HTTPException, ValueError, LookupError, TypeError) as error:
        # No answer, a response that broke off, or one that is not a completion.
        request.error = repr(error)
    finally:
        request.end = time.monotonic()
    return request


def _largest_gap(times: List[float], start: float, end: float) -> float:
    """Return the length of the longest interval from [start, end] to the next arrival."""
    inside = [start] + [t for t in times if start < t < end]
    later = [t for t in times if t >= end]
    inside.append(later[0] if later else end)
    return max(b - a for a, b in zip(inside, inside[1:]))


class Monitor(_Threads):
    """Sample each frontend's ``/health`` and status, and record each membership change."""

    def __init__(self, deployment: Deployment):
        super().__init__()
        self._deployment = deployment
        self._changed = threading.Condition()
        self.health: Dict[str, List[Tuple[float, Optional[int], Optional[str]]]] = {
            base: [] for base in deployment.bases
        }
        self.membership: List[Tuple[float, dict]] = []
        self._reader = FaultToleranceHealth(_rendezvous(deployment.secret), deployment.secret)
        for base in self.health:
            self._spawn(self._sample_health, base)
        self._spawn(self._sample_membership, self._reader)

    def stop(self) -> List[str]:
        """Stop watching and return the samplers that raised or still run."""
        failures = self._join(60.0)
        self._reader.close()
        return failures

    def _sample_health(self, base: str) -> None:
        while not self._stop.is_set():
            code, _ = _get_json(f"{base}/health")
            status = _status(base)
            self.health[base].append(
                (time.monotonic(), code, None if status is None else status["state"])
            )
            self._stop.wait(0.25)

    def _sample_membership(self, reader: FaultToleranceHealth) -> None:
        while not self._stop.is_set():
            status = reader.read_status().model_dump()
            # The public status leaves out the record's own state, which the checks need to time
            # kills in a change.
            record = reader._record
            status["membership_state"] = None if record is None else record.state.name
            t = time.monotonic()
            if not self.membership or status != self.membership[-1][1]:
                with self._changed:
                    self.membership.append((t, status))
                    self._changed.notify_all()
            self._stop.wait(STORE_POLL_S)

    def wait_for(
        self, predicate: Callable[[dict], bool], since: float, timeout_s: float
    ) -> Tuple[float, dict]:
        """Return the time and the first status after ``since`` that satisfies ``predicate``."""
        deadline = time.monotonic() + timeout_s
        checked = 0
        with self._changed:
            while True:
                for t, status in self.membership[checked:]:
                    if t > since and predicate(status):
                        return t, status
                checked = len(self.membership)
                self._deployment.check_running()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._failures:
                    last = self.membership[-1] if self.membership else None
                    raise AssertionError(
                        f"not reached within {timeout_s} s; last membership read: {last}; "
                        f"failed samplers: {self._failures}"
                    )
                self._changed.wait(min(remaining, 1.0))

    def wait_for_health(self, code: int, state: str, since: float, timeout_s: float) -> None:
        """Wait until each frontend's ``/health`` answered ``code`` in ``state`` after ``since``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._failures:
            if all(
                any(t > since and (c, s) == (code, state) for t, c, s in list(samples))
                for samples in self.health.values()
            ):
                return
            self._deployment.check_running()
            time.sleep(0.1)
        last = {base: samples[-1] if samples else None for base, samples in self.health.items()}
        raise AssertionError(
            f"/health did not answer {code} in state {state} within {timeout_s} s; last samples: {last}; "
            f"failed samplers: {self._failures}"
        )


def _committed_after(generation: int) -> Callable[[dict], bool]:
    return lambda s: s["membership_state"] == "COMMITTED" and s["generation"] > generation


def _running(process: psutil.Process) -> bool:
    """Return whether ``process`` runs, so it has neither ended nor waits to be reaped."""
    try:
        return process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_for_replacement(
    deployment: Deployment, replaced_pid: int, rank: int = VICTIM
) -> Dict[int, psutil.Process]:
    """Wait for and return the worker process of ``rank`` that replaces ``replaced_pid``.

    Fail at once when the node supervisor gives the rank up instead.
    """
    # The node supervisor logs one of these when it gives the rank up.
    given_up_log = re.compile(
        rf"replacements of rank {rank} failed in a row|The replacement of rank {rank} cannot start"
    )
    deadline = time.monotonic() + REJOIN_TIMEOUT_S
    while (
        process := _rank_workers(deployment.secret).get(rank)
    ) is None or process.pid == replaced_pid:
        assert time.monotonic() < deadline, f"no replacement of rank {rank} started"
        given_up = [line for line in _log_lines(deployment.logs) if given_up_log.search(line)]
        assert not given_up, f"no replacement of rank {rank} started: {given_up[-1].strip()}"
        time.sleep(1)
    return {rank: process}


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
    """Run the load, the monitor and failures of rank ``victim`` on one deployment, then check."""

    def __init__(self, deployment: Deployment, victim: int = VICTIM):
        status = deployment.status()
        assert status["active_ranks"] == list(range(WORLD_SIZE)), status
        assert not status["failed_ranks"], status
        self.deployment = deployment
        self.victim = victim
        self.status_after_warmup = status
        self.captures_after_warmup = _capture_lines(deployment.logs)
        self.gsm8k = Gsm8k(EVAL_QUESTIONS)
        self.clients = Clients(deployment, self.gsm8k)
        self.phases: List[Phase] = []
        self.cycles = 0
        # Only requests in flight at a rank failure may fail.
        self.failure_times: List[float] = []
        # From each hang to the commit that removes the hung rank.
        self.hangs: List[Tuple[float, float]] = []
        self.failures: List[str] = []
        self.t0 = time.monotonic()
        # Last, since its status reader runs from its construction until it stops.
        self.monitor = Monitor(deployment)

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
            # A status may read stalled until one refresh period and one read after a hang ends.
            lag_s = _STATUS_REFRESH_PERIOD_S + _STATUS_READ_TIMEOUT_S
            states = {
                str(state)
                for samples in self.monitor.health.values()
                for t, _, state in samples
                if state != "stalled" or not any(a <= t <= b + lag_s for a, b in self.hangs)
            }
            self.check(states <= ALLOWED_HEALTH_STATES, f"states outside a hang: {sorted(states)}")
            retryable = sum(1 for r in self.clients.requests if r.retryable)
            logged = _retryable_failures_logged(self.deployment.logs)
            # Rank 0 logs the count at every membership change, so this also checks the pattern.
            self.check(
                len(logged) >= self.cycles,
                f"rank 0 logged {len(logged)} lines matching {RETRYABLE_LOG.pattern!r} in "
                f"{self.cycles} failure cycles",
            )
            self.check(
                retryable <= sum(logged.values()),
                f"{retryable} requests failed as retryable, but rank 0 logged only "
                f"{sum(logged.values())}",
            )
            assert not self.failures, "\n".join(self.failures)

    def window(self, name: str, cycle: int) -> Phase:
        """Serve for WINDOW_S, then until every eval question is answered or EVAL_COVER_EXTRA_S."""
        self.clients.restart_questions()
        start = time.monotonic()
        time.sleep(WINDOW_S)
        deadline = time.monotonic() + EVAL_COVER_EXTRA_S
        while len(self.clients.answered(start)) < EVAL_QUESTIONS and time.monotonic() < deadline:
            self.deployment.check_running()
            time.sleep(1)
        phase = Phase(name, cycle, start, time.monotonic())
        self.phases.append(phase)
        return phase

    def _committed_generations(self) -> set:
        """Return the generations of the membership changes that committed so far."""
        return {
            s["last_transaction"]["generation"]
            for _, s in self.monitor.membership
            if s["last_transaction"] is not None
        }

    def cycle(
        self,
        failure: str,
        at_admission: Optional[Callable[[], Dict[int, psutil.Process]]] = None,
        at_admission_state: Optional[str] = None,
    ) -> None:
        """Fail a rank with ``crash`` or ``hang``, wait for N ranks, and serve a window.

        ``at_admission`` returns processes to kill once the store shows the admission in
        ``at_admission_state``, which the suspended joining rank holds open.
        """
        self.cycles += 1
        d, index = self.deployment, self.cycles
        status = d.status()
        assert status["active_ranks"] == list(range(WORLD_SIZE)), status
        assert status["failed_ranks"] == [], status
        workers_before = _rank_workers(d.secret)  # scanning the processes takes seconds
        gpus_before = _rank_gpus(workers_before)
        captures_before = _capture_lines(d.logs)
        assert self.victim in workers_before, f"ranks on this node: {sorted(workers_before)}"
        victim = workers_before[self.victim]

        t_fail = time.monotonic()
        if failure == "hang":
            victim.suspend()
        else:
            victim.kill()
        self.failure_times.append(t_fail)
        detection_s = HANG_TIMEOUT_S if failure == "hang" else 0.0
        t_degraded, failover_status = self.monitor.wait_for(
            _committed_after(status["generation"]), t_fail, detection_s + RECOVERY_TIMEOUT_S
        )
        self.monitor.wait_for_health(200, "degraded", t_degraded, RECOVERY_TIMEOUT_S)
        self.phases.append(Phase("failover", index, t_fail, t_degraded, event=t_fail))
        time.sleep(SETTLE_S)
        t_settled = time.monotonic()
        self.clients.restart_questions()
        if failure == "hang":
            self.hangs.append((t_fail, t_degraded))
            # The hung rank keeps the others waiting for the hang timeout, so it reads stalled.
            self.check(
                all(
                    any(state == "stalled" and t >= t_fail for t, _, state in list(samples))
                    for samples in self.monitor.health.values()
                ),
                f"cycle {index}: the status did not read stalled during the hang",
            )
            # The node supervisor kills the stopped process once the membership drops it, so it
            # never runs again.
            try:
                victim.wait(EXCLUDED_EXIT_TIMEOUT_S)
            except psutil.TimeoutExpired:
                self.failures.append(
                    f"cycle {index}: the excluded process still exists "
                    f"{EXCLUDED_EXIT_TIMEOUT_S} s after the failover"
                )

        killed_at_admission: List[int] = []
        if at_admission is not None:
            joining = _wait_for_replacement(d, victim.pid, self.victim)[self.victim]
            processes = at_admission()
            killed_at_admission = sorted(processes)

            def admitting_in(states: Tuple[str, ...]) -> Callable[[dict], bool]:
                return lambda s: (
                    s["membership_state"] in states
                    and s["generation"] > failover_status["generation"]
                    and self.victim in s["active_ranks"]
                )

            # Suspend the joining rank once the store shows the admission. The admission then
            # cannot commit, since the joining rank must install it, and the survivors still move
            # it from PROPOSE to INSTALL, as a joining rank's readiness counts as its reply. The
            # authority gives a suspended joining rank up only after ADMISSION_TIMEOUT_S.
            t_admitting, _ = self.monitor.wait_for(
                admitting_in(("PROPOSE", "INSTALL")), t_degraded, REJOIN_TIMEOUT_S
            )
            joining.suspend()
            try:
                # Kill them in the admission's first PROPOSE or INSTALL, or in that state.
                states = (
                    ("PROPOSE", "INSTALL") if at_admission_state is None else (at_admission_state,)
                )
                _, admitting = self.monitor.wait_for(
                    admitting_in(states), t_degraded, RECOVERY_TIMEOUT_S
                )
                for process in processes.values():
                    process.kill()
                t_kill = time.monotonic()
                self.failure_times.append(t_kill)
            finally:
                with contextlib.suppress(psutil.NoSuchProcess):
                    joining.resume()
            # The kill must land before the admission commits. The admission's generation then never
            # commits, and the next committed record leaves the killed ranks out.
            t_after_kill, after_kill = self.monitor.wait_for(
                _committed_after(admitting["generation"]), t_kill, RECOVERY_TIMEOUT_S
            )
            self.phases.append(
                Phase("admission_failure", index, t_kill, t_after_kill, event=t_kill)
            )
            self.check(
                admitting["generation"] not in self._committed_generations()
                and not set(killed_at_admission) & set(after_kill["active_ranks"]),
                f"cycle {index}: the kill did not land during the admission: {after_kill}",
            )

        t_rejoined, rejoined_status = self.monitor.wait_for(
            lambda s: s["membership_state"] == "COMMITTED"
            and s["active_ranks"] == list(range(WORLD_SIZE)),
            t_degraded,
            REJOIN_TIMEOUT_S,
        )
        self.monitor.wait_for_health(200, "healthy", t_rejoined, RECOVERY_TIMEOUT_S)
        admission = rejoined_status["last_transaction"]
        # The commit is read within STORE_POLL_S. The authority measures the transaction's duration.
        t_admission = t_rejoined - (admission["ended_unix_ms"] - admission["began_unix_ms"]) / 1e3
        changes = _membership_changes_metric(d.base, {"failover": index, "admission": index})
        self.check(
            changes.get("failover", 0) >= index and changes.get("admission", 0) >= index,
            f"cycle {index}: trtllm_fault_tolerance_membership_changes after the rejoin: {changes}",
        )
        degraded_end = t_admission - 1.0
        self.phases.append(Phase("degraded", index, t_settled, degraded_end))
        self.phases.append(Phase("admission", index, t_admission, t_rejoined, event=t_admission))
        time.sleep(SETTLE_S)
        rejoined = self.window("rejoined", index)
        workers_after = _rank_workers(d.secret)
        gpus_after = _rank_gpus(workers_after)
        captures_after = _capture_lines(d.logs)
        replaced = sorted({self.victim, *killed_at_admission})

        # The longest time without a streamed token at the failure, at the admission, at a kill
        # during the admission, and from the commit at N ranks to the end of the window.
        gaps = {
            "pause_s": (
                t_fail - 1.0,
                degraded_end if at_admission is None else t_admitting - 1.0,
                detection_s + RELAXED_PAUSE_BOUND_S,
                operator.le,
            ),
            "admission_pause_s": (
                t_admission - 1.0,
                t_rejoined + SETTLE_S,
                RELAXED_PAUSE_BOUND_S,
                operator.lt,
            ),
            "stall_s": (t_rejoined, rejoined.end, RELAXED_PAUSE_BOUND_S, operator.le),
        }
        if at_admission is not None:
            gaps["admission_failure_pause_s"] = (
                t_admitting - 1.0,
                t_after_kill + SETTLE_S,
                RELAXED_PAUSE_BOUND_S,
                operator.lt,
            )
        arrivals = self.clients.stream_arrivals()
        for name, (start, end, bound_s, within) in gaps.items():
            gap_s = _largest_gap(arrivals, start, end)
            self.check(
                within(gap_s, bound_s),
                f"cycle {index}: no streamed token for {gap_s:.2f} s ({name}), bound {bound_s} s",
            )
        spanning = [r for r in self.clients.load_requests() if r.spans(t_fail)]
        for stream in (False, True):
            self.check(
                any(r.stream == stream and r.error is None for r in spanning),
                f"cycle {index}: no {'streamed' if stream else 'non-streamed'} request "
                "in flight at the failure completed",
            )
        # Requests sent again after a retryable failure at this cycle's failures.
        retries = [
            r for r in self.clients.requests if r.retry and t_fail <= r.start <= degraded_end
        ]
        self.check(
            any(r.error is None for r in retries),
            f"cycle {index}: none of the {len(retries)} requests sent again "
            "after a retryable failure succeeded",
        )
        self.check(
            failover_status["generation"] > status["generation"]
            and failover_status["failed_ranks"] == [self.victim]
            and failover_status["active_ranks"]
            == [r for r in range(WORLD_SIZE) if r != self.victim]
            and failover_status["authority_alive"],
            f"cycle {index}: status after the failover: {failover_status}",
        )
        # No survivor restarted. Run._check_captures checks that none captured a graph since.
        for rank in set(workers_before) - set(replaced):
            self.check(
                rank in workers_after and workers_after[rank].pid == workers_before[rank].pid,
                f"cycle {index}: survivor rank {rank} was restarted",
            )
        self.check(
            rejoined_status["failed_ranks"] == []
            and rejoined_status["generation"] > failover_status["generation"],
            f"cycle {index}: status after the rejoin: {rejoined_status}",
        )
        # A replacement takes the next incarnation, and one killed during its admission is
        # replaced once more.
        expected = [
            n + (r == self.victim) + (r in killed_at_admission)
            for r, n in enumerate(status["incarnations"])
        ]
        self.check(
            rejoined_status["incarnations"] == expected,
            f"cycle {index}: incarnations {status['incarnations']} -> "
            f"{rejoined_status['incarnations']}, expected {expected}",
        )
        # A replacement runs on the GPU of the process it replaces and captures its own CUDA graphs.
        for rank in replaced:
            self.check(
                rank not in workers_before
                or (bool(gpus_before[rank]) and gpus_after.get(rank) == gpus_before[rank]),
                f"cycle {index}: rank {rank}'s replacement runs on GPUs {gpus_after.get(rank)}, "
                f"the process it replaces ran on {gpus_before.get(rank)}",
            )
            self.check(
                captures_after.get(rank, 0) > captures_before.get(rank, 0),
                f"cycle {index}: no CUDA graph capture lines of rank {rank}'s replacement",
            )

    @contextlib.contextmanager
    def refused_join(self, rank: int, replacement: bool, reason: str):
        """Join as ``rank`` and yield the join time and whether the authority logged ``reason`` yet."""
        d, line = self.deployment, REFUSAL_LOG + reason

        def logged() -> int:
            return sum(line in log_line for log_line in _log_lines(d.logs))

        before = logged()
        store = open_store(_rendezvous(d.secret), timeout_s=60.0, secret=d.secret)
        t = time.monotonic()
        with mock.patch.dict(os.environ, {SHARED_SECRET_ENV: d.secret}):
            connection = open_liveness_connection(store, rank, replacement)
        with connection:
            yield t, lambda: logged() > before
        self.check(
            logged() > before,
            f"the authority logged no refusal of a hello for rank {rank} "
            f"(replacement {replacement}): {line!r}",
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
        """Check that every request completed in full, except retryable streams across a failure."""
        for r in self.clients.requests:
            kind = ("retried " if r.retry else "") + ("streamed" if r.stream else "non-streamed")
            kind += " eval" if r.question is not None else ""
            if r.error is None:
                if r.question is None:
                    self.check(
                        len(r.token_ids) == r.completion_tokens == LOAD_MAX_TOKENS,
                        f"a {kind} request returned {len(r.token_ids)} tokens "
                        f"(usage {r.completion_tokens}), not {LOAD_MAX_TOKENS}",
                    )
                continue
            spanned = [t for t in self.failure_times if r.spans(t)]
            self.check(
                bool(not r.retry and r.stream and r.retryable and spanned and r.token_ids),
                f"a {kind} request failed (started {r.start - self.t0:.1f} s, "
                f"ended {r.end - self.t0:.1f} s into the run, failures at "
                f"{[round(t - self.t0, 1) for t in self.failure_times]}, "
                f"{len(r.token_ids)} tokens): {r.error[:500]}",
            )
            self.check(
                len(r.token_ids) < LOAD_MAX_TOKENS, "a failed stream returned too many tokens"
            )

    def _check_captures(self) -> None:
        """Check that every rank captured CUDA graphs at startup and no original rank since."""
        d = self.deployment
        start, end = self.status_after_warmup["incarnations"], d.status()
        first_processes = [r for r in end["active_ranks"] if end["incarnations"][r] == start[r]]
        before, after = self.captures_after_warmup, _capture_lines(d.logs)
        for rank in range(WORLD_SIZE):
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


def _check_refused_join(run: Run, kind: str, replacement: bool, reason: str) -> None:
    """Send a join for the victim's rank that the authority refuses, and check nothing changes."""
    d = run.deployment
    before = d.status()
    pids = {r: p.pid for r, p in _rank_workers(d.secret).items()}

    def change() -> Optional[str]:
        d.check_running()
        status = _status(d.base)
        if (
            status is None
            or status["generation"] != before["generation"]
            or status["active_ranks"] != list(range(WORLD_SIZE))
        ):
            return f"changed the membership: {before} -> {status}"
        after = {r: p.pid for r, p in _rank_workers(d.secret).items()}
        return f"restarted a process: {pids} -> {after}" if after != pids else None

    with run.refused_join(run.victim, replacement, reason) as (t_join, refused):
        deadline = time.monotonic() + REFUSAL_TIMEOUT_S
        while not refused() and time.monotonic() < deadline:
            time.sleep(1)
        # The endpoint serves a status at most one refresh period and one read old.
        fresh = time.monotonic() + _STATUS_REFRESH_PERIOD_S + _STATUS_READ_TIMEOUT_S
        while True:
            checked_s = time.monotonic()
            if (changed := change()) is not None or checked_s >= fresh:
                break
            time.sleep(0.1)
    run.phases.append(Phase(f"{kind}_join", run.cycles, t_join, time.monotonic()))
    run.check(changed is None, f"a {kind} join for rank {run.victim} {changed}")


def _second_rank(deployment: Deployment) -> Dict[int, psutil.Process]:
    """Return the process of a rank on this node to fail second, neither rank 0 nor the victim."""
    workers = _rank_workers(deployment.secret)
    second = min((r for r in workers if r not in (0, VICTIM)), default=None)
    assert second is not None, f"no second rank to fail on this node: {sorted(workers)}"
    return {second: workers[second]}


def _crash_during_admission(deployment: Deployment, state: Optional[str] = None) -> None:
    """Crash the victim, then a second rank on this node during the victim's admission."""
    with Run(deployment) as run:
        second = _second_rank(deployment)
        run.cycle("crash", at_admission=lambda: second, at_admission_state=state)


class TestRankFailures:
    """Tests that fail ranks of one deployment, each ending with every rank active.

    They kill rank 2 three times and rank 1 once.
    """

    @pytest.mark.parametrize("sequence", ["crash", "crash-hang"])
    @pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
    def test_rank_failure_sequence(self, shared_deployment: Deployment, sequence: str):
        """Fail a rank per ``-``-separated step of ``sequence``, each followed by a rejoin."""
        with Run(shared_deployment) as run:
            for failure in sequence.split("-"):
                run.cycle(failure)

    @pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
    def test_rank_stopped_while_idle(self, shared_deployment: Deployment):
        """Stop a rank while the server is idle, send a request, and check that it completes.

        With no request, every rank but 0 waits in the request broadcast with its hang detector
        paused. The node supervisor sees the stopped process, so the rank does not count as paused:
        once the request makes the other ranks wait on it, the authority removes it after the hang
        timeout, the request completes on the other ranks, and a replacement rejoins.
        """
        d = shared_deployment
        status = d.status()
        assert status["active_ranks"] == list(range(WORLD_SIZE)), status
        victim = _rank_workers(d.secret)[IDLE_VICTIM]
        body = {
            "prompt": _prompt(0, LOAD_INPUT_LEN[0]),
            "max_tokens": 16,
            "temperature": 0.0,
            "detokenize": False,
        }

        def send(retry: bool = False) -> Request:
            request = Request(base=d.base, stream=False, start=time.monotonic(), retry=retry)
            return _send_completion(d.model, request, body)

        sender = concurrent.futures.ThreadPoolExecutor(1)
        monitor = Monitor(d)
        monitor.start()
        try:
            time.sleep(SETTLE_S)
            victim.suspend()
            t_request = time.monotonic()
            sending = sender.submit(send)
            t_removed, removed = monitor.wait_for(
                _committed_after(status["generation"]),
                t_request,
                HANG_TIMEOUT_S + RELAXED_PAUSE_BOUND_S,
            )
            assert removed["failed_ranks"] == [IDLE_VICTIM], removed
            # An exception in the request's thread is raised here.
            request = sending.result(REQUEST_TIMEOUT_S)
            # A request that ran on the removed rank fails as retryable, and is sent again.
            if request.retryable:
                request = send(retry=True)
            assert request.error is None, request
            t_rejoined, _ = monitor.wait_for(
                lambda s: s["membership_state"] == "COMMITTED"
                and s["active_ranks"] == list(range(WORLD_SIZE)),
                t_removed,
                REJOIN_TIMEOUT_S,
            )
            monitor.wait_for_health(200, "healthy", t_rejoined, RECOVERY_TIMEOUT_S)
        finally:
            # Without the removal, the stopped rank continues, so the deployment can end.
            with contextlib.suppress(psutil.NoSuchProcess):
                victim.resume()
            sender.shutdown(wait=False)
            failures = monitor.stop()
        assert not failures, failures


class TestAdmissionFailures:
    """Tests that refuse joins or fail ranks during admissions, each ending with every rank active.

    They kill rank 3 three times, rank 2 twice and rank 1 twice.
    """

    @pytest.mark.timeout(STARTUP_TIMEOUT_S + 3 * REJOIN_TIMEOUT_S + 1200)
    def test_refused_joins_and_admission_kill(self, shared_deployment: Deployment):
        """Send joins the authority refuses, and kill a replacement during its admission."""
        d, rank = shared_deployment, REFUSED_JOIN_VICTIM
        first_process_refusal = (
            f"rank {rank} joins only as a replacement once the deployment has started"
        )
        with Run(d, rank) as run:
            with run.refused_join(rank, False, first_process_refusal):
                run.cycle("crash")
            _check_refused_join(run, "stale", False, first_process_refusal)
            _check_refused_join(
                run, "duplicate", True, f"rank {rank} is active: no replacement joins for it"
            )

            victim_pid = _rank_workers(d.secret)[rank].pid
            run.cycle("crash", at_admission=lambda: _wait_for_replacement(d, victim_pid, rank))

    @pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
    def test_failure_during_admission(self, shared_deployment: Deployment):
        """Crash a second rank while the first one's replacement is admitted."""
        _crash_during_admission(shared_deployment)

    @pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * REJOIN_TIMEOUT_S + 1200)
    def test_failure_during_install(self, shared_deployment: Deployment):
        """Crash a second rank while the active ranks install the first one's admission."""
        _crash_during_admission(shared_deployment, "INSTALL")


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + 2 * RECOVERY_TIMEOUT_S + 600)
def test_failure_during_failover_ends_deployment(tmp_path: Path):
    """Crash a second rank during the first one's failover, and check the deployment ends.

    On one node the spare slots hold the experts of one lost rank, so the survivors cannot restore
    two: they reply FAILED to the INSTALL, and the job ends with a failure status and the reason.
    """
    with _deploy(tmp_path) as deployment:
        generation = deployment.status()["generation"]
        workers = _rank_workers(deployment.secret)
        second = _second_rank(deployment)
        holding = min(r for r in workers if r not in (0, VICTIM, *second))
        monitor = Monitor(deployment)
        monitor.start()
        try:
            t_fail = time.monotonic()
            workers[VICTIM].kill()
            # Suspend a survivor once the store shows the failover. It then neither replies nor
            # installs, so the failover cannot commit before the second kill.
            monitor.wait_for(
                lambda s: s["membership_state"] in ("PROPOSE", "INSTALL")
                and s["generation"] > generation,
                t_fail,
                RECOVERY_TIMEOUT_S,
            )
            workers[holding].suspend()
            try:
                for process in second.values():
                    process.kill()
            finally:
                with contextlib.suppress(psutil.NoSuchProcess):
                    workers[holding].resume()
        finally:
            # The deployment may end before the samplers stop, so their failures are not checked.
            monitor.stop()
        assert deployment.launcher.wait(RECOVERY_TIMEOUT_S) != 0
        lines = list(_log_lines(deployment.logs))
        assert any("cannot hold expert" in line for line in lines), (
            "no rank logged that the spare slots cannot hold a lost expert"
        )
        assert any("Deployment aborted: in INSTALL" in line for line in lines), (
            "the deployment did not abort in the failover's INSTALL"
        )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + AUTHORITY_LOSS_END_S + 600)
def test_authority_loss_ends_deployment(tmp_path: Path):
    """Kill the authority and check that /health fails and the job ends with a failure status."""
    with _deploy(tmp_path) as deployment:
        authority = [
            p for p in _node_processes(deployment.secret) if AUTHORITY_MODULE in p.cmdline()
        ]
        assert len(authority) == 1, f"authority processes on this node: {authority}"
        authority[0].kill()
        deadline = time.monotonic() + AUTHORITY_LOSS_END_S
        while (code := _get_json(f"{deployment.base}/health")[0]) == 200:
            assert time.monotonic() < deadline, (
                f"/health still answers 200 {AUTHORITY_LOSS_END_S} s after the authority ended"
            )
            time.sleep(0.2)
        # Rank 0 ends once its connection to the authority closes, maybe before a probe reads 503.
        assert code in (503, None), f"/health answered {code} after the authority ended"
        assert deployment.launcher.wait(AUTHORITY_LOSS_END_S) != 0
        # Rank 0 ended because its connection to the authority failed, not for another reason.
        assert any(
            "Rank 0 exits: the membership client failed" in line
            for line in _log_lines(deployment.logs)
        ), "rank 0 did not log that it exits for the loss of the authority"


def _interrupt_server(deployment: Deployment, clients: Clients) -> float:
    """Stop the clients sending, send SIGINT to rank 0's trtllm-serve, and return the time."""
    (server_pid,) = {
        int(m.group(1)) for m in map(SERVER_PID_LOG.search, _log_lines(deployment.logs)) if m
    }
    # The clients send no new request, so every request was sent before the stop.
    clients.stop_sending()
    os.kill(server_pid, signal.SIGINT)
    return time.monotonic()


def _wait_for_shutdown(deployment: Deployment, t_stop: float) -> None:
    """Wait for the launcher to end after the SIGINT, and check that the deployment shut down."""
    try:
        deployment.launcher.wait(max(0.0, t_stop + SHUTDOWN_TIMEOUT_S - time.monotonic()))
    except subprocess.TimeoutExpired:
        pytest.fail(f"the launcher still ran {SHUTDOWN_TIMEOUT_S} s after the SIGINT")
    states = {m.group(1) for m in map(MEMBERSHIP_LOG.search, _log_lines(deployment.logs)) if m}
    assert "SHUTDOWN" in states and "ABORTED" not in states, states


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
def test_shutdown_after_rank_given_up(tmp_path: Path):
    """Give up a killed rank, serve without it, then stop the deployment with SIGINT.

    Each replacement of the rank is killed once it holds its GPU, until the node supervisor gives
    the rank up. SIGINT to trtllm-serve then drains the requests in flight, the authority publishes
    SHUTDOWN, and the launcher ends on its own.
    """
    with _deploy(tmp_path) as deployment:
        with Run(deployment) as run:
            status = deployment.status()
            victim = _rank_workers(deployment.secret)[VICTIM]
            victim.kill()
            t_fail = time.monotonic()
            run.failure_times.append(t_fail)
            t_degraded, failover = run.monitor.wait_for(
                _committed_after(status["generation"]), t_fail, RECOVERY_TIMEOUT_S
            )
            for _ in range(_MAX_FAILED_REPLACEMENTS):
                victim = _wait_for_replacement(deployment, victim.pid)[VICTIM]
                victim.kill()
            t_given_up, given_up = run.monitor.wait_for(
                lambda s: s["replacements"] == {VICTIM: "given_up"}, t_degraded, RECOVERY_TIMEOUT_S
            )
            # Publishing the given-up rank starts no membership change.
            serving = [r for r in range(WORLD_SIZE) if r != VICTIM]
            assert (given_up["generation"], given_up["active_ranks"]) == (
                failover["generation"],
                serving,
            ), given_up
            run.monitor.wait_for_health(200, "degraded", t_given_up, RECOVERY_TIMEOUT_S)
            run.window("degraded", 1)
            assert VICTIM not in _rank_workers(deployment.secret), "the given-up rank was replaced"

        clients = Clients(deployment, run.gsm8k)
        clients.start()
        time.sleep(SETTLE_S)
        t_stop = _interrupt_server(deployment, clients)
        assert not clients.stop()
        _wait_for_shutdown(deployment, t_stop)
        assert any(r.spans(t_stop) for r in clients.requests), (
            "no request was in flight at the stop"
        )
        incomplete = [
            r
            for r in clients.requests
            if r.error is not None or (r.question is None and len(r.token_ids) != LOAD_MAX_TOKENS)
        ]
        assert not incomplete, (
            f"{len(incomplete)} requests did not complete, the first with "
            f"{len(incomplete[0].token_ids)} tokens and the error {incomplete[0].error}"
        )


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
def test_shutdown_during_admission(tmp_path: Path):
    """Stop the deployment with SIGINT while a replacement starts, so that it is admitted in the drain.

    The replacement is held at its CUDA graph capture until the launcher's frontend stops the engine.
    Long requests through the attached frontend keep the engine draining until rank 0 admits the
    replacement, which then stops with the other ranks: the authority publishes SHUTDOWN, the
    replacement exits 0 on its own, and the launcher ends.
    """
    with _deploy(tmp_path, frontends=2) as deployment:
        captures = _capture_lines(deployment.logs)[VICTIM]
        victim = _rank_workers(deployment.secret)[VICTIM]
        victim.kill()
        replacement = _wait_for_replacement(deployment, victim.pid)[VICTIM]
        deadline = time.monotonic() + REJOIN_TIMEOUT_S
        while _capture_lines(deployment.logs)[VICTIM] == captures:
            assert time.monotonic() < deadline, f"rank {VICTIM}'s replacement captured no graph"
            if not _running(replacement):
                ends = [line for line in _log_lines(deployment.logs) if VICTIM_END_LOG.search(line)]
                pytest.fail(
                    f"rank {VICTIM}'s replacement ended before a capture:\n{''.join(ends[-2:])}"
                )
            time.sleep(0.1)
        replacement.suspend()
        attached = replace(deployment, frontends=deployment.frontends[1:])
        clients = Clients(attached, Gsm8k(EVAL_QUESTIONS), max_tokens=DRAIN_MAX_TOKENS)
        clients.start()
        time.sleep(SETTLE_S)
        t_stop = _interrupt_server(deployment, clients)
        while not any(APP_SHUTDOWN_LOG in line for line in _log_lines(deployment.logs)):
            assert time.monotonic() < t_stop + SHUTDOWN_TIMEOUT_S, "the engine was not stopped"
            time.sleep(0.1)
        replacement.resume()
        assert not clients.stop()
        _wait_for_shutdown(deployment, t_stop)
        # A request completes, or the attached frontend rejects it with an error response.
        broken = [
            r
            for r in clients.requests
            if r.error and r.error_code is None and not r.error.startswith("HTTP ")
        ]
        assert not broken, f"{len(broken)} requests broke off, the first with {broken[0].error}"
        lines = list(_log_lines(deployment.logs))
        stop = next(i for i, line in enumerate(lines) if APP_SHUTDOWN_LOG in line)
        assert any(
            "outcome='COMMITTED'" in line and f"added_ranks=({VICTIM},)" in line
            for line in lines[stop:]
        ), "the replacement was not admitted after the engine was stopped"
        after_stop = "".join(lines[stop:])
        assert f"Rank {VICTIM} exits: the executor completed" in after_stop
        assert "The deployment shut down" in after_stop and "did not end within" not in after_stop


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
@pytest.mark.parametrize("target", [t for t in TARGETS if t != DEFAULT_TARGET])
def test_rejoin_on_target(tmp_path: Path, target: str):
    """Fail over and rejoin a killed rank on another target's default configuration."""
    with _deploy(tmp_path, target=target) as deployment:
        with Run(deployment) as run:
            run.cycle("crash")


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
def test_rejoin_when_server_spawns_ranks(tmp_path: Path):
    """Fail over and rejoin a killed rank under plain trtllm-serve, which chooses the secret."""
    with _deploy(tmp_path, server_spawns_ranks=True) as deployment:
        with Run(deployment) as run:
            run.cycle("crash")


@pytest.mark.skip_less_device(WORLD_SIZE)
@pytest.mark.timeout(STARTUP_TIMEOUT_S + REJOIN_TIMEOUT_S + 1200)
def test_rejoin_through_two_frontends(tmp_path: Path):
    """Fail over and rejoin a killed rank while the clients and the monitor use both frontends."""
    with _deploy(tmp_path, server_spawns_ranks=True, frontends=2) as deployment:
        with Run(deployment) as run:
            run.cycle("crash")
        (t_fail,) = run.failure_times
        for base in deployment.frontends:
            in_flight = [r for r in run.clients.requests if r.base == base and r.spans(t_fail)]
            assert any(r.error is None for r in in_flight), (
                f"no request through {base} in flight at the failure completed"
            )
