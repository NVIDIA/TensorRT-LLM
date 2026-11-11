# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The servers of a deployment: started behind its launch prefix, owned, asked, and stopped.

A server is ``trtllm-serve <checkpoint> --config <options> --port 0 --report_addr <file>
--num_serve_frontends=1`` behind the deployment's ``launch`` prefix: the kernel picks the
port, the server writes ``<host>:<port>`` once it is bound (its host is rank 0's), and
the flow reads it back. A disaggregated deployment is a context server, a generation
server and ``trtllm-serve disaggregated`` with a proxy config written from the two
addresses. ``launch`` is the site's own prefix (a scheduler, a container) and nothing
else: where the options ask for more than one rank the flow puts the LLM API's launcher
before the program itself (:func:`launcher`).

The flow owns what it starts. Each server runs in its own process group and under a name
(``MV2FLOW_SERVER``) the site's ``launch`` and ``stop`` strings may use for their step.
What a later process needs to end it is recorded in the workspace before the server is
waited for: the host, the group, its leader's start time, the ``stop`` command and the
environment it runs in. Stopping is a SIGTERM to the group, a wait, the site's ``stop``,
and then a look through the same prefix at what holds the devices: a holder that was not
there before the start means ranks were left behind, the check has no verdict, and the
record stays until the devices are seen free. A look that could not be taken is said
(:data:`UNVERIFIED`), never taken for clean. A new start first ends what an earlier
process recorded: it signals a recorded group only on the same host and while its leader
is the process that was started, and otherwise runs the recorded ``stop`` alone. Nothing
is ever killed by a pattern, and every wait is bounded by the deployment's ``timeout``.

One load at a time: :func:`engine_lock` is one lock per workspace, taken around every
check, a worker's own gate command included.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from ..infra.compute.environment import LLMAPI_LAUNCH, MPI_SETTINGS_KEPT, SINGLE_PROCESS_WORKER_ENV
from ..infra.compute.server_runtime import rank_launch
from ..infra.files import write_text_atomic
from ..spec.deployments import ROLES, Deployment, Server, write_options
from .check_runtime import MODE_REQUIRE, engine_env

#: What this run started and has not seen ended, in the workspace.
OWNED_FILENAME = "servers.json"
#: The name a server runs under, exported to its launch and stop commands.
SERVER_NAME_ENV = "MV2FLOW_SERVER"
#: How long a server gets to exit after SIGTERM before its group is killed.
STOP_GRACE_S = 90
#: How long the devices get to empty after a stop, and one look at them at most.
RELEASE_TIMEOUT_S = 120
PROBE_TIMEOUT_S = 120
#: A deployment's start timeout when it states none: a load of a large checkpoint.
START_TIMEOUT_S = 1800
#: What a launch must not inherit from the orchestrator: the MPI and scheduler identity
#: of the process that happens to start it. The prefix brings its own.
SCRUBBED_PREFIXES: tuple[str, ...] = ("PMI_", "PMIX_", "OMPI_", "MPI_", "SLURM_", "MV2_")
#: The role names ``trtllm-serve --server_role`` takes for the two sides.
SERVER_ROLES: dict[str, str] = {"ctx": "context", "gen": "generation"}
#: What a verdict's line says when the devices could not be looked at after a stop.
UNVERIFIED = "devices not verified free"
#: Where the engine locks are: on this node whatever ``TMPDIR`` a caller has.
LOCK_DIRS: tuple[str, ...] = ("/dev/shm", "/var/tmp", "/tmp")  # nosec B108
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# Each process holding a device, with the host it is on: one prefix may land on several.
# A host whose devices cannot be listed fails the look; it is not a host with no holder.
_HOLDERS = (
    "bash -o pipefail -c 'h=$(hostname); nvidia-smi --query-compute-apps=pid --format=csv,noheader "
    '| while read -r p; do echo "$h $p"; done\''
)


class ServerError(RuntimeError):
    """A server that did not start or did not answer; the message names its log."""


class Leftover(ServerError):
    """Ranks still hold a device after a stop: whatever was measured is no verdict."""


@dataclass
class Owned:
    """One server this run started: enough for another process to end it, and no more."""

    name: str
    pid: int
    pgid: int
    host: str = ""
    #: The leader's start time (``/proc/<pid>/stat``): the group is this server's only
    #: while its leader is still that process.
    started: str = ""
    launch: str = ""
    stop: str = ""
    #: The environment its ``launch`` and ``stop`` commands were given by the spec.
    env: dict[str, str] = field(default_factory=dict)
    address_file: str = ""
    log: str = ""
    #: The device holders seen through the prefix before the start; ``None`` when the
    #: look could not be taken.
    before: list[str] | None = None


@dataclass
class Running:
    """A started deployment: the address to ask, and what to stop."""

    deployment: Deployment
    address: str
    owned: list[Owned]
    workspace: Path
    #: The seconds every wait of its start had, and every wait of its stop has.
    budget: float = START_TIMEOUT_S

    @property
    def url(self) -> str:
        return f"http://{self.address}"

    def log(self, role: str) -> str:
        """What the server of ``role`` printed so far: where its ranks say what they run."""
        name = self.deployment.name + ("" if role == "server" else f".{role}")
        path = next((record.log for record in self.owned if record.name == name), "")
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace") if path else ""
        except OSError:
            return ""


@contextlib.contextmanager
def engine_lock(workspace: str | Path | None) -> Iterator[None]:
    """Hold the workspace's one engine lock: the run's checks and a worker's never overlap.

    The lock file is in a fixed node-local directory, named by the user and the workspace:
    a shared filesystem does not always honour ``flock``, and two callers with different
    ``TMPDIR`` must still meet at one file. Without a workspace nothing is held.
    """
    if workspace is None:
        yield
        return
    digest = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
    directory = next(d for d in LOCK_DIRS if os.access(d, os.W_OK))
    with (Path(directory) / f"mv2flow-engine-{os.getuid()}-{digest}.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- what is owned


def files_of(workspace: str | Path, deployment: str) -> Path:
    """Where the servers of ``deployment`` keep their options, addresses and logs."""
    return Path(workspace) / "deployments" / deployment


def load_owned(workspace: str | Path) -> list[Owned]:
    try:
        rows = json.loads((Path(workspace) / OWNED_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    known = set(Owned.__dataclass_fields__)
    return [Owned(**{k: v for k, v in row.items() if k in known}) for row in rows or ()]


def _save_owned(workspace: Path, owned: Sequence[Owned]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    write_text_atomic(workspace / OWNED_FILENAME, json.dumps([asdict(o) for o in owned], indent=1))


def _start_time(pid: int) -> str:
    """When ``pid`` started, in the kernel's clock ticks; ``""`` when there is no such process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return ""
    return stat.rpartition(")")[2].split()[19]


def _exited(owned: Owned) -> bool:
    """Whether the leader this process started has exited.

    It is left unreaped: its start time stays readable, so :func:`_end` still knows the
    group for its own and ends whatever the leader left behind.
    """
    try:
        return os.waitid(os.P_PID, owned.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return not _alive(owned.pgid)


def _alive(pgid: int) -> bool:
    """Whether any process of the group still runs; a leader that exited is reaped first."""
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pgid, os.WNOHANG)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _site_env(owned: Owned) -> dict[str, str]:
    """What ``owned``'s launch and stop commands run with: the spec's words, its own name."""
    return {**launch_env(owned.launch, owned.env), SERVER_NAME_ENV: owned.name}


def _end(owned: Owned, grace_s: float) -> None:
    """End one server within ``grace_s``: its own group while it is still its own, then the site's ``stop``.

    The time is shared: SIGTERM gets what the kill and the site's ``stop`` do not need,
    and each wait is for what then remains.
    """
    # Read before anything is reaped: a leader that exited keeps its start time until then.
    ours = owned.host == socket.gethostname() and _start_time(owned.pid) == owned.started != ""
    deadline = time.monotonic() + grace_s
    kill_s, stop_s = min(10.0, grace_s / 6), grace_s / 3 if owned.stop else 0.0
    for sig, until in (
        (signal.SIGTERM, deadline - stop_s - kill_s),
        (signal.SIGKILL, deadline - stop_s),
    ):
        if not ours or not _alive(owned.pgid):
            break
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(owned.pgid, sig)
        while time.monotonic() < until and _alive(owned.pgid):
            time.sleep(min(0.2, max(0.0, until - time.monotonic())))
    if owned.stop:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["bash", "-c", owned.stop],
                env=_site_env(owned),
                capture_output=True,
                timeout=max(stop_s, deadline - time.monotonic()),
                check=False,
            )


def _holders(launch: str, env: Mapping[str, str], timeout: float) -> list[str] | None:
    """``<host> <pid>`` of each process holding a device where ``launch`` lands; ``None`` unknown."""
    try:
        done = subprocess.run(
            ["bash", "-c", f"{launch} {_HOLDERS}".strip()],
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode:
        return None
    return sorted({" ".join(line.split()) for line in done.stdout.splitlines() if line.strip()})


def _released(owned: Owned, release_s: float) -> str:
    """Look at the devices of ``owned``, which was ended: ``""`` free, :data:`UNVERIFIED`, or who holds them."""
    deadline = time.monotonic() + release_s
    while True:
        # Each look and each pause is for the time that remains, never a fresh allowance.
        left = max(0.1, deadline - time.monotonic())
        now = _holders(owned.launch, _site_env(owned), min(PROBE_TIMEOUT_S, left))
        if now is None or owned.before is None:
            return UNVERIFIED
        stray = sorted(set(now) - set(owned.before))
        left = deadline - time.monotonic()
        if not stray or left <= 0:
            return ", ".join(stray)
        time.sleep(min(1.0, left / 2))


def _retire(workspace: Path, records: Sequence[Owned], budget: float) -> tuple[list[str], bool]:
    """End ``records``, last started first; drop from the workspace those whose devices are not held.

    Every one is ended before the devices are looked at: the ranks of a deployment's other
    server, still up, are not ranks left behind. Returns what still holds a device, by
    server, and whether any look could not be taken.
    """
    held, unverified = [], False
    kept = {o.name: o for o in load_owned(workspace)}
    for owned in reversed(list(records)):
        _end(owned, min(STOP_GRACE_S, budget))
    for owned in reversed(list(records)):
        said = _released(owned, min(RELEASE_TIMEOUT_S, budget))
        if said and said != UNVERIFIED:
            held.append(f"{owned.name}: {said}")
            kept[owned.name] = owned
            continue
        unverified |= said == UNVERIFIED
        kept.pop(owned.name, None)
    _save_owned(workspace, list(kept.values()))
    return held, unverified


def clean_owned(workspace: str | Path, budget: float = STOP_GRACE_S) -> None:
    """End every server an earlier process of this run recorded, each in its own environment.

    Raises :class:`Leftover` while ranks of one still hold a device: the next load would
    share them, so it does not start, and the record stays for the next try.
    """
    held, _ = _retire(Path(workspace), load_owned(workspace), budget)
    if held:
        raise Leftover(
            "ranks an earlier check left still hold a device (" + "; ".join(held) + "): nothing "
            "is started on them; end them (the site's `stop`), then start again"
        )


# ------------------------------------------------------------------------------ a launch


def launch_env(launch: str, declared: Mapping[str, str]) -> dict[str, str]:
    """The environment of a server's command: the process's, the spec's, and no borrowed identity.

    Behind a prefix nothing of the orchestrator's MPI or scheduler identity is passed on
    (:data:`SCRUBBED_PREFIXES`); a variable the spec declares is, whatever its name.
    """
    env = dict(os.environ)
    if launch.strip():
        env = {k: v for k, v in env.items() if not k.startswith(SCRUBBED_PREFIXES)}
    env.update({str(k): str(v) for k, v in declared.items()})
    return env


def launcher(server: Server) -> list[str]:
    """What the flow puts before a program that runs ``server``'s ranks.

    Nothing for one rank. Behind a launch prefix, which has placed the ranks, the LLM
    API's launcher; on this host, ``mpirun`` and the launcher, as every in-process check
    starts them.
    """
    if server.ranks <= 1:
        return []
    return [LLMAPI_LAUNCH] if server.launch.strip() else rank_launch(server.ranks, LLMAPI_LAUNCH)


def local_env(
    server: Server, devices: Sequence[int], env: Mapping[str, str], declared: Sequence[str]
) -> dict[str, str]:
    """``env`` for a server with no launch prefix: its own devices, and no MPI world it did not start.

    One rank runs its worker in its own process, with the bootstrap variables this
    process may have inherited from an allocation taken out (all but those the spec
    ``declared``). Behind a prefix the site places both, and ``env`` is returned as it is.
    """
    env = dict(env)
    if server.launch.strip():
        return env
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    if server.ranks > 1:
        return env
    kept = {"SLURM_PROCID", "SLURM_LOCALID", *MPI_SETTINGS_KEPT, *declared}
    env = {
        k: v
        for k, v in env.items()
        if k in kept or not k.startswith(("PMI", "MPI", "OMPI", "SLURM"))
    }
    env.setdefault(SINGLE_PROCESS_WORKER_ENV, "1")
    return env


def serve_argv(
    model: str, options: str | Path, address_file: str | Path, *, role: str | None = None
) -> list[str]:
    """``trtllm-serve`` for one server: the kernel picks the port and the server reports it."""
    argv = ["trtllm-serve", str(model), "--config", str(options)]
    # Bound to every interface, so the address it reports names its host, not localhost.
    argv += ["--host", "0.0.0.0", "--port", "0", "--report_addr", str(address_file)]  # nosec B104
    argv.append("--num_serve_frontends=1")
    if role is not None:
        argv += ["--server_role", SERVER_ROLES[role]]
    return argv


def proxy_config(model: str, ctx_address: str, gen_address: str) -> dict[str, Any]:
    """The disaggregated server's config: one context and one generation server, by address."""
    return {
        "hostname": "0.0.0.0",  # nosec B104
        "port": 0,
        "model": str(model),
        "backend": "pytorch",
        "context_servers": {"num_instances": 1, "urls": [ctx_address]},
        "generation_servers": {"num_instances": 1, "urls": [gen_address]},
    }


#: What a log line that states an error looks like: the cause of a server that died is
#: one of these, often a hundred lines above the bottom of a backtrace.
_ERROR_LINE = re.compile(r"Error|Exception|Invalid|Traceback|\bE\d{4} ")


def _tail(record: Owned, lines: int = 8, errors: int = 6) -> str:
    """Where ``record``'s log is, its last lines that state an error, then its last lines."""
    try:
        text = Path(record.log).read_text(errors="replace").splitlines()
    except OSError:
        text = []
    last = text[-lines:]
    said = [line for line in text[: len(text) - len(last)] if _ERROR_LINE.search(line)][-errors:]
    shown = [*said, *(["  ..."] if said else []), *last]
    return f"({record.log}):\n" + "\n".join(shown)


def _await(record: Owned, deadline: float, watched: Sequence[Owned] = (), address: str = "") -> str:
    """Wait for ``record``'s address (or, given one, its health); fail when a server ends first.

    ``watched`` are the servers it stands on: a context or generation server that dies
    while the proxy is waited for fails the start at once, by its own name and log.
    """
    path = Path(record.address_file)
    what = "healthy" if address else "bound to an address"
    while time.monotonic() < deadline:
        for server in (record, *watched):
            if _exited(server):
                raise ServerError(
                    f"{server.name} exited before {record.name} was {what} {_tail(server)}"
                )
        if not address and path.is_file() and path.read_text().strip():
            return path.read_text().strip()
        if address:
            left = max(1.0, min(10.0, deadline - time.monotonic()))
            with contextlib.suppress(urllib.error.URLError, OSError):
                with _DIRECT.open(f"http://{address}/health", timeout=left) as reply:
                    if reply.status == 200:
                        return address
        time.sleep(0.2)
    why = "was not healthy" if address else "reported no address"
    raise ServerError(f"{record.name} {why} within its timeout {_tail(record)}")


def start(
    deployment: Deployment,
    *,
    model: str,
    workspace: str | Path,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
    tree: str | Path | None = None,
    python: str | None = None,
    devices: Sequence[int] = (),
) -> Running:
    """Start ``deployment``'s servers and wait until the one to ask is healthy.

    Ends first whatever an earlier process of the run left recorded (:func:`clean_owned`).
    ``env`` is the spec's environment; the deployment's own is laid over it. Every server
    runs from ``tree`` with the Modeling V2 switch on and the rank hook loaded
    (``check_runtime.engine_env``), so its ranks say what they run. A server with no
    launch prefix takes its ranks' share of ``devices``, in the order the servers start.
    Every wait ends at the deployment's ``timeout``. On any failure everything started
    here is stopped before the error is raised.
    """
    workspace = Path(workspace)
    budget = float(timeout or deployment.timeout or START_TIMEOUT_S)
    deadline = time.monotonic() + budget
    clean_owned(workspace, budget)
    declared = engine_env({**(env or {}), **deployment.env}, tree=tree, python=python)
    directory = files_of(workspace, deployment.name)
    directory.mkdir(parents=True, exist_ok=True)
    options = write_options(deployment, directory, {"modeling_v2": MODE_REQUIRE})
    left = list(devices)
    owned: list[Owned] = []
    running = Running(deployment, "", owned, workspace, budget)

    def begin(name: str, server: Server, argv: Sequence[str], address_file: Path) -> Owned:
        """Start ``argv`` behind ``server``'s prefix in its own group, and record it."""
        # A server of this host with options takes its ranks' devices; the proxy has neither.
        takes = server.ranks if server.options and not server.launch.strip() else 0
        own = [left.pop(0) for _ in range(min(takes, len(left)))]
        process_env = local_env(server, own, launch_env(server.launch, declared), list(declared))
        process_env[SERVER_NAME_ENV] = name
        address_file.unlink(missing_ok=True)
        line = f"{server.launch} {shlex.join([*launcher(server), *argv])}".strip()
        # The line as it is started, beside the log, for whoever reads a start that failed.
        (directory / f"{name}.launch").write_text(line + "\n", encoding="utf-8")
        look = max(0.1, min(PROBE_TIMEOUT_S, deadline - time.monotonic()))
        before = _holders(server.launch, process_env, look)
        with (directory / f"{name}.log").open("wb") as sink:
            process = subprocess.Popen(
                ["bash", "-c", line],
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=process_env,
                cwd=str(tree) if tree is not None else None,
                start_new_session=True,
            )
        record = Owned(
            name=name,
            pid=process.pid,
            pgid=process.pid,
            host=socket.gethostname(),
            started=_start_time(process.pid),
            launch=server.launch,
            stop=server.stop,
            # Every declared value, one that equals this process's own too: a later
            # process that ends the server may run with another.
            env=dict(declared),
            address_file=str(address_file),
            log=str(directory / f"{name}.log"),
            before=before,
        )
        owned.append(record)
        _save_owned(workspace, [*load_owned(workspace), record])
        return record

    try:
        if deployment.server is not None:
            address_file = directory / "address"
            argv = serve_argv(model, options["server"], address_file)
            record = begin(deployment.name, deployment.server, argv, address_file)
            running.address = _await(record, deadline, address=_await(record, deadline))
            return running
        sides = []
        for role in ROLES:
            address_file = directory / f"address.{role}"
            argv = serve_argv(model, options[role], address_file, role=role)
            sides.append(
                begin(f"{deployment.name}.{role}", getattr(deployment, role), argv, address_file)
            )
        ctx, gen = (_await(side, deadline) for side in sides)
        config = directory / "proxy.yaml"
        config.write_text(yaml.safe_dump(proxy_config(model, ctx, gen)), "utf-8")
        remaining = max(1, int(deadline - time.monotonic()))
        argv = ["trtllm-serve", "disaggregated", "-c", str(config)]
        argv += ["--server_start_timeout", str(remaining), "-r", "360000"]
        argv += ["--report_addr", str(directory / "address.proxy")]
        proxy = begin(
            f"{deployment.name}.proxy", Server(options={}), argv, directory / "address.proxy"
        )
        running.address = _await(proxy, deadline, sides, _await(proxy, deadline, sides))
        return running
    except BaseException:
        with contextlib.suppress(ServerError):
            stop(running)
        raise


def stop(running: Running) -> str:
    """Stop every server of ``running``, last started first, and look at the devices.

    Returns ``""`` when they are seen free, :data:`UNVERIFIED` when a look could not be
    taken. Raises :class:`Leftover` naming what still holds a device where a prefix lands
    and was not there before the start: ranks were left behind, so whatever was measured
    is not a verdict, and their record stays in the workspace until they are gone.
    """
    held, unverified = _retire(running.workspace, running.owned, running.budget)
    running.owned.clear()
    if held:
        raise Leftover(
            "ranks were left holding a device after the stop (" + "; ".join(held) + "): the "
            "check has no verdict, and the site's `stop` must end them before the next load"
        )
    return UNVERIFIED if unverified else ""


# ------------------------------------------------------------------------------ asking


def _post(
    running: Running, path: str, body: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        f"{running.url}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    name = running.deployment.name
    try:
        with _DIRECT.open(request, timeout=timeout) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # The body of an error response is the server's own word on what is wrong.
        body = exc.read().decode("utf-8", errors="replace").strip()[:300]
        raise ServerError(f"{name} did not answer {path}: {exc}: {body}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ServerError(f"{name} did not answer {path}: {exc}") from exc
    choice = (payload.get("choices") or [None])[0] if isinstance(payload, dict) else None
    # An answer counts once it completed: a response cut off mid-way has no finish reason.
    if not isinstance(choice, dict) or not choice.get("finish_reason"):
        raise ServerError(f"{name} answered {path} without a completed choice: {payload}")
    return payload


def complete(
    running: Running, prompt: str, *, max_tokens: int, timeout: float = 600
) -> Mapping[str, Any]:
    """One greedy completion from the deployment's endpoint: the response, as the server sent it."""
    body = {
        "model": "deployment",
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
    }
    return _post(running, "/v1/completions", body, timeout)


def ask(running: Running, prompt: str, *, max_tokens: int, timeout: float = 600) -> str:
    """One greedy completion from the deployment's endpoint; its text."""
    payload = complete(running, prompt, max_tokens=max_tokens, timeout=timeout)
    return str(payload["choices"][0].get("text") or "")


def ask_image(
    running: Running, prompt: str, image: str | Path, *, max_tokens: int, timeout: float = 600
) -> str:
    """One greedy chat completion over ``prompt`` and the image file ``image``; its text."""
    try:
        data = base64.b64encode(Path(image).read_bytes()).decode("ascii")
    except OSError as exc:
        raise ServerError(f"the image {image} cannot be read: {exc}") from exc
    kind = Path(image).suffix.lstrip(".").lower().replace("jpg", "jpeg") or "png"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{data}"}},
    ]
    body = {
        "model": "deployment",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
    }
    payload = _post(running, "/v1/chat/completions", body, timeout)
    return str((payload["choices"][0].get("message") or {}).get("content") or "")


def cached_tokens(payload: Mapping[str, Any]) -> int | None:
    """The prompt tokens a completion resumed from the cache, where the server reports them."""
    details = (payload.get("usage") or {}).get("prompt_tokens_details") or {}
    count = details.get("cached_tokens")
    return count if isinstance(count, int) and not isinstance(count, bool) else None
